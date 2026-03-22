"""Qulacs-accelerated exact-state routines for selected QCANT algorithms."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Sequence

from .adapt.adaptvqe import (
    _ansatz_type_from_pool_type,
    _build_operator_pool,
    _iter_chunks,
    _normalize_pool_type,
    _resolve_chunk_size,
    _resolve_worker_count,
)
from .cvqe.cvqe import (
    _bits_to_index,
    _build_uccsd_excitation_metadata,
    _extract_active_ccsd_amplitudes,
    _index_to_bits,
    _initialize_lucj_params,
    _initialize_uccsd_params,
    _parameter_slices,
    _parameter_slices_uccsd,
    _restore_history,
    _select_new_determinant,
    _to_plain_data,
    _unpack_uccsd_params,
    _validate_inputs as _validate_cvqe_inputs,
    _write_json_payload,
)


_PAULI_TO_ID = {"X": 1, "Y": 2, "Z": 3}
_PRIMITIVE_PARAM_GATE_NAMES = {"RX", "RY", "RZ", "MultiRZ"}
_FIXED_SINGLE_QUBIT_GATES = {
    "Hadamard",
    "PauliX",
    "PauliY",
    "PauliZ",
    "S",
    "Adjoint(S)",
    "T",
    "Adjoint(T)",
}
_FIXED_TWO_QUBIT_GATES = {"CNOT", "CZ", "SWAP"}


@dataclass
class _CompiledParametricCircuit:
    """Compiled Qulacs parametric circuit with user-parameter remapping."""

    qubits: int
    circuit: object
    user_parameter_count: int
    parameter_groups: list[list[tuple[int, float]]] = field(default_factory=list)

    def set_user_parameters(self, user_values, np) -> None:
        values = np.asarray(user_values, dtype=float)
        if values.size != int(self.user_parameter_count):
            raise ValueError(
                f"expected {self.user_parameter_count} user parameters, received {values.size}"
            )
        for user_idx, group in enumerate(self.parameter_groups):
            value = float(values[user_idx])
            for circuit_idx, scale in group:
                self.circuit.set_parameter(int(circuit_idx), float(scale) * value)


def _adjust_user_gradient(raw_gradient, parameter_groups, np):
    adjusted = np.zeros(len(parameter_groups), dtype=float)
    for user_idx, group in enumerate(parameter_groups):
        total = 0.0
        for circuit_idx, scale in group:
            total += float(scale) * float(raw_gradient[int(circuit_idx)])
        adjusted[user_idx] = total
    return adjusted


def _create_quantum_state_from_bits(bits, qubits: int, QuantumState, np):
    return _state_from_vector(
        _basis_vector_from_bits(bits, qubits, np),
        qubits=qubits,
        QuantumState=QuantumState,
        np=np,
    )


def _make_excitation_operator(theta: float, excitation, *, ansatz_type: str, qml):
    excitation = [int(value) for value in excitation]
    angle = float(theta)
    if ansatz_type == "fermionic":
        if len(excitation) == 2:
            return qml.FermionicSingleExcitation(
                weight=angle,
                wires=list(range(excitation[0], excitation[1] + 1)),
            )
        if len(excitation) == 4:
            return qml.FermionicDoubleExcitation(
                weight=angle,
                wires1=list(range(excitation[0], excitation[1] + 1)),
                wires2=list(range(excitation[2], excitation[3] + 1)),
            )
    elif ansatz_type == "qubit_excitation":
        if len(excitation) == 2:
            return qml.SingleExcitation(
                angle,
                wires=[int(excitation[0]), int(excitation[1])],
            )
        if len(excitation) == 4:
            return qml.DoubleExcitation(
                angle,
                wires=[int(excitation[0]), int(excitation[1]), int(excitation[2]), int(excitation[3])],
            )

    raise ValueError(
        "Each excitation must have length 2 (single) or 4 (double); "
        f"received {excitation!r} for ansatz_type='{ansatz_type}'."
    )


def _flatten_qml_primitives(operator, *, qml, np):
    name = str(getattr(operator, "name", type(operator).__name__))
    wires = tuple(int(wire) for wire in getattr(operator, "wires", []))
    params = [complex(param) for param in getattr(operator, "parameters", [])]

    if name in _PRIMITIVE_PARAM_GATE_NAMES and len(params) == 1:
        return [
            {
                "kind": "param_primitive",
                "name": name,
                "wires": wires,
                "angle": complex(params[0]),
            }
        ]
    if name in _FIXED_SINGLE_QUBIT_GATES.union(_FIXED_TWO_QUBIT_GATES) and len(params) == 0:
        return [
            {
                "kind": "fixed_gate",
                "name": name,
                "wires": wires,
            }
        ]

    decomposition = None
    try:
        decomposition = operator.decomposition()
    except Exception:
        decomposition = None

    if decomposition:
        primitives = []
        for sub_op in decomposition:
            primitives.extend(_flatten_qml_primitives(sub_op, qml=qml, np=np))
        return primitives

    matrix = np.asarray(qml.matrix(operator, wire_order=list(wires)), dtype=complex)
    return [
        {
            "kind": "fixed_dense",
            "wires": wires,
            "matrix": matrix,
        }
    ]


def _primitive_signature(primitive, np):
    if primitive["kind"] == "param_primitive":
        return (primitive["kind"], primitive["name"], tuple(primitive["wires"]))
    if primitive["kind"] == "fixed_gate":
        return (primitive["kind"], primitive["name"], tuple(primitive["wires"]))
    return (primitive["kind"], tuple(primitive["wires"]), tuple(np.asarray(primitive["matrix"]).shape))


def _append_fixed_gate(circuit, primitive, *, qubits: int, qg):
    wires = tuple(int(wire) for wire in primitive["wires"])
    name = str(primitive["name"])
    if name == "Hadamard":
        circuit.add_H_gate(_qulacs_wire(qubits, wires[0]))
        return
    if name == "PauliX":
        circuit.add_X_gate(_qulacs_wire(qubits, wires[0]))
        return
    if name == "PauliY":
        circuit.add_Y_gate(_qulacs_wire(qubits, wires[0]))
        return
    if name == "PauliZ":
        circuit.add_Z_gate(_qulacs_wire(qubits, wires[0]))
        return
    if name == "S":
        circuit.add_S_gate(_qulacs_wire(qubits, wires[0]))
        return
    if name == "Adjoint(S)":
        circuit.add_Sdag_gate(_qulacs_wire(qubits, wires[0]))
        return
    if name == "T":
        circuit.add_T_gate(_qulacs_wire(qubits, wires[0]))
        return
    if name == "Adjoint(T)":
        circuit.add_Tdag_gate(_qulacs_wire(qubits, wires[0]))
        return
    if name == "CNOT":
        circuit.add_CNOT_gate(_qulacs_wire(qubits, wires[0]), _qulacs_wire(qubits, wires[1]))
        return
    if name == "CZ":
        circuit.add_CZ_gate(_qulacs_wire(qubits, wires[0]), _qulacs_wire(qubits, wires[1]))
        return
    if name == "SWAP":
        circuit.add_SWAP_gate(_qulacs_wire(qubits, wires[0]), _qulacs_wire(qubits, wires[1]))
        return
    raise ValueError(f"Unsupported fixed primitive gate: {name}")


def _append_fixed_rotation(circuit, primitive, *, qubits: int, qg, np):
    name = str(primitive["name"])
    wires = tuple(int(wire) for wire in primitive["wires"])
    angle = float(np.real_if_close(primitive["angle"]))

    if abs(angle) < 1e-15:
        return
    if name == "RX":
        circuit.add_RotX_gate(_qulacs_wire(qubits, wires[0]), angle)
        return
    if name == "RY":
        circuit.add_RotY_gate(_qulacs_wire(qubits, wires[0]), angle)
        return
    if name == "RZ":
        circuit.add_RotZ_gate(_qulacs_wire(qubits, wires[0]), angle)
        return
    if name == "MultiRZ":
        qulacs_wires = _dense_gate_wires(qubits, wires)
        circuit.add_gate(qg.PauliRotation(qulacs_wires, [3] * len(qulacs_wires), -angle))
        return
    raise ValueError(f"Unsupported fixed rotation primitive: {name}")


def _append_parametric_rotation(circuit, primitive, *, qubits: int):
    name = str(primitive["name"])
    wires = tuple(int(wire) for wire in primitive["wires"])
    if name == "RX":
        circuit.add_parametric_RX_gate(_qulacs_wire(qubits, wires[0]), 0.0)
    elif name == "RY":
        circuit.add_parametric_RY_gate(_qulacs_wire(qubits, wires[0]), 0.0)
    elif name == "RZ":
        circuit.add_parametric_RZ_gate(_qulacs_wire(qubits, wires[0]), 0.0)
    elif name == "MultiRZ":
        circuit.add_parametric_multi_Pauli_rotation_gate(
            _dense_gate_wires(qubits, wires),
            [3] * len(wires),
            0.0,
        )
    else:
        raise ValueError(f"Unsupported parametric rotation primitive: {name}")
    return int(circuit.get_parameter_count() - 1)


def _parametric_gate_scale(name: str, coeff: float) -> float:
    # Qulacs parametric RX/RY/RZ and Pauli rotations use the opposite sign
    # convention from PennyLane's RX/RY/RZ/MultiRZ primitives.
    if name in {"RX", "RY", "RZ", "MultiRZ"}:
        return -float(coeff)
    raise ValueError(f"Unsupported parametric primitive name: {name}")


def _compile_single_parameter_factory_into_circuit(
    circuit,
    parameter_groups,
    user_param_index: int,
    operator_factory: Callable[[float], object],
    *,
    qubits: int,
    qml,
    qg,
    np,
):
    sample_values = [0.0, 1.0, 2.0]
    primitive_samples = [
        _flatten_qml_primitives(operator_factory(sample), qml=qml, np=np)
        for sample in sample_values
    ]
    reference = primitive_samples[0]
    if not all(len(sample) == len(reference) for sample in primitive_samples[1:]):
        raise ValueError("parametric decomposition changed size across samples")

    for idx in range(len(reference)):
        signatures = [_primitive_signature(sample[idx], np) for sample in primitive_samples]
        if not all(signature == signatures[0] for signature in signatures[1:]):
            raise ValueError("parametric decomposition changed structure across samples")

        primitive0 = primitive_samples[0][idx]
        primitive1 = primitive_samples[1][idx]
        primitive2 = primitive_samples[2][idx]

        if primitive0["kind"] == "param_primitive":
            angle0 = float(np.real_if_close(primitive0["angle"]))
            angle1 = float(np.real_if_close(primitive1["angle"]))
            angle2 = float(np.real_if_close(primitive2["angle"]))
            coeff = angle1 - angle0
            linearity_residual = abs((angle2 - angle0) - (2.0 * coeff))
            if linearity_residual > 1e-10:
                raise ValueError(
                    "Only affine parametric decompositions are supported; "
                    f"residual was {linearity_residual:.3e}."
                )
            _append_fixed_rotation(circuit, primitive0, qubits=qubits, qg=qg, np=np)
            if abs(coeff) > 1e-15:
                circuit_param_index = _append_parametric_rotation(
                    circuit,
                    primitive0,
                    qubits=qubits,
                )
                parameter_groups[int(user_param_index)].append(
                    (int(circuit_param_index), _parametric_gate_scale(str(primitive0["name"]), coeff))
                )
        elif primitive0["kind"] == "fixed_gate":
            _append_fixed_gate(circuit, primitive0, qubits=qubits, qg=qg)
        elif primitive0["kind"] == "fixed_dense":
            matrix0 = np.asarray(primitive0["matrix"], dtype=complex)
            matrix1 = np.asarray(primitive1["matrix"], dtype=complex)
            matrix2 = np.asarray(primitive2["matrix"], dtype=complex)
            if not (np.allclose(matrix0, matrix1, atol=1e-10, rtol=0.0) and np.allclose(matrix0, matrix2, atol=1e-10, rtol=0.0)):
                raise ValueError("Encountered unsupported parameter-dependent dense primitive during compilation")
            circuit.add_gate(qg.DenseMatrix(_dense_gate_wires(qubits, primitive0["wires"]), matrix0))
        else:
            raise ValueError(f"Unsupported primitive kind: {primitive0['kind']}")


def _evaluate_compiled_state(
    compiled_circuit: _CompiledParametricCircuit,
    user_values,
    *,
    QuantumState,
    np,
    init_bits=None,
    init_vector=None,
):
    compiled_circuit.set_user_parameters(user_values, np)
    if init_vector is not None:
        state = _state_from_vector(
            np.asarray(init_vector, dtype=complex),
            qubits=compiled_circuit.qubits,
            QuantumState=QuantumState,
            np=np,
        )
    elif init_bits is not None:
        state = _create_quantum_state_from_bits(
            init_bits,
            compiled_circuit.qubits,
            QuantumState,
            np,
        )
    else:
        state = QuantumState(int(compiled_circuit.qubits))
        state.set_zero_state()
    compiled_circuit.circuit.update_quantum_state(state)
    return state


def _build_adam_result(x, fun, nit, grad_norm):
    return {
        "x": x,
        "fun": float(fun),
        "nit": int(nit),
        "success": True,
        "grad_norm": float(grad_norm),
    }


def _optimize_with_adam(
    x0,
    cost_grad_fn: Callable[[object], tuple[float, object]],
    *,
    maxiter: int,
    step_size: float,
    np,
):
    x = np.asarray(x0, dtype=float).copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    last_fun = None
    last_grad = np.zeros_like(x)

    for iteration in range(1, int(maxiter) + 1):
        last_fun, grad = cost_grad_fn(x)
        grad = np.asarray(grad, dtype=float)
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad * grad)
        m_hat = m / (1.0 - beta1**iteration)
        v_hat = v / (1.0 - beta2**iteration)
        x = x - float(step_size) * m_hat / (np.sqrt(v_hat) + epsilon)
        last_grad = grad

    if last_fun is None:
        last_fun, last_grad = cost_grad_fn(x)
        last_grad = np.asarray(last_grad, dtype=float)
    return _build_adam_result(x, last_fun, int(maxiter), np.linalg.norm(last_grad))


def _import_optional_stack():
    try:
        import numpy as np
        import pennylane as qml
        import pyscf
        from pyscf import cc, gto, mcscf, scf
        from qulacs import GeneralQuantumOperator, Observable, ParametricQuantumCircuit, QuantumCircuit, QuantumState
        from qulacs import gate as qg
        from scipy.optimize import minimize
        from scipy.sparse.linalg import expm_multiply
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Qulacs-accelerated routines require optional dependencies. Install at least: "
            "`pip install qulacs numpy scipy pennylane pyscf`."
        ) from exc

    return {
        "np": np,
        "qml": qml,
        "pyscf": pyscf,
        "cc": cc,
        "gto": gto,
        "mcscf": mcscf,
        "scf": scf,
        "GeneralQuantumOperator": GeneralQuantumOperator,
        "Observable": Observable,
        "ParametricQuantumCircuit": ParametricQuantumCircuit,
        "QuantumCircuit": QuantumCircuit,
        "QuantumState": QuantumState,
        "qg": qg,
        "minimize": minimize,
        "expm_multiply": expm_multiply,
    }


def _qulacs_wire(qubits: int, wire: int) -> int:
    return int(qubits) - 1 - int(wire)


def _dense_gate_wires(qubits: int, wires: Sequence[int]) -> list[int]:
    return [_qulacs_wire(qubits, wire) for wire in wires[::-1]]


def _basis_vector_from_bits(bits, qubits: int, np):
    state = np.zeros(2**int(qubits), dtype=complex)
    state[_bits_to_index(bits)] = 1.0
    return state


def _state_from_vector(vector, qubits: int, QuantumState, np):
    state = QuantumState(int(qubits))
    state.load(np.asarray(vector, dtype=complex))
    return state


def _normalize_state(state, QuantumState, np):
    vector = np.asarray(state.get_vector(), dtype=complex)
    norm = np.linalg.norm(vector)
    if norm < 1e-14:
        raise ValueError("state became numerically zero")
    if abs(norm - 1.0) > 1e-12:
        state = _state_from_vector(vector / norm, qubits=state.get_qubit_count(), QuantumState=QuantumState, np=np)
    return state


def _threshold_state(state, basis_threshold: float, QuantumState, np):
    if basis_threshold <= 0:
        return state
    vector = np.asarray(state.get_vector(), dtype=complex)
    mask = np.abs(vector) >= float(basis_threshold)
    if not np.any(mask):
        mask[int(np.argmax(np.abs(vector)))] = True
    filtered = np.where(mask, vector, 0.0)
    norm = np.linalg.norm(filtered)
    if norm < 1e-14:
        raise ValueError("thresholded state has zero norm")
    return _state_from_vector(filtered / norm, qubits=state.get_qubit_count(), QuantumState=QuantumState, np=np)


def _sanitize_hamiltonian(operator, np, qml):
    if hasattr(operator, "terms"):
        coeffs, ops = operator.terms()
    else:
        coeffs, ops = getattr(operator, "coeffs", []), getattr(operator, "ops", [])
    coeffs = np.asarray(coeffs, dtype=complex)
    if coeffs.size > 0 and (np.any(np.abs(coeffs.imag) > 1e-12) or coeffs.dtype.kind == "c"):
        operator = qml.Hamiltonian(coeffs.real.astype(float), ops)
    return operator


def _qml_operator_to_qulacs_operator(operator, qubits: int, *, qml, operator_cls):
    qulacs_operator = operator_cls(int(qubits))
    term_specs = []

    if hasattr(operator, "terms"):
        coeffs, ops = operator.terms()
        operator_terms = zip(coeffs, ops)
    else:
        operator_terms = [(1.0, operator)]

    for coeff, term in operator_terms:
        pauli_sentence = getattr(term, "pauli_rep", None)
        if pauli_sentence is None:
            pauli_sentence = qml.pauli.pauli_sentence(term)

        for word, word_coeff in pauli_sentence.items():
            total_coeff = complex(coeff) * complex(word_coeff)
            mapped_items = [(_qulacs_wire(qubits, int(wire)), str(label)) for wire, label in word.items()]
            pauli_string = " ".join(f"{label} {wire}" for wire, label in mapped_items)
            qulacs_operator.add_operator(total_coeff, pauli_string)
            term_specs.append(
                {
                    "coeff": complex(total_coeff),
                    "indices": [int(wire) for wire, _ in mapped_items],
                    "pauli_ids": [_PAULI_TO_ID[str(label)] for _, label in mapped_items],
                    "pauli_string": pauli_string,
                }
            )

    return qulacs_operator, term_specs


def _build_hamiltonian_payload(
    symbols: Sequence[str],
    geometry,
    *,
    basis: str,
    charge: int,
    spin: int,
    active_electrons: int,
    active_orbitals: int,
    hamiltonian_cutoff: float,
    hamiltonian_source: str = "casci",
):
    stack = _import_optional_stack()
    np = stack["np"]
    qml = stack["qml"]
    pyscf = stack["pyscf"]
    gto = stack["gto"]
    mcscf = stack["mcscf"]
    scf = stack["scf"]
    Observable = stack["Observable"]

    atom = [(symbols[i], tuple(float(x) for x in geometry[i])) for i in range(len(symbols))]
    mol = gto.Mole()
    mol.atom = atom
    mol.unit = "Angstrom"
    mol.basis = basis
    mol.charge = charge
    mol.spin = spin
    mol.symmetry = False
    mol.build()

    mf = scf.RHF(mol)
    mf.level_shift = 0.5
    mf.diis_space = 12
    mf.max_cycle = 100
    mf.kernel()
    if not mf.converged:
        mf = scf.newton(mf).run()

    hamiltonian_source = str(hamiltonian_source).strip().lower()
    if hamiltonian_source == "casci":
        mycas = mcscf.CASCI(mf, active_orbitals, active_electrons)
        h1ecas, ecore = mycas.get_h1eff(mf.mo_coeff)
        h2ecas = mycas.get_h2eff(mf.mo_coeff)
        two_mo = pyscf.ao2mo.restore("1", h2ecas, norb=int(mycas.ncas))
        two_mo = np.swapaxes(two_mo, 1, 3)
        h_fermionic = qml.qchem.fermionic_observable(
            np.array([ecore]),
            h1ecas,
            two_mo,
            cutoff=hamiltonian_cutoff,
        )
        h_qml = qml.jordan_wigner(h_fermionic)
        h_qml = _sanitize_hamiltonian(h_qml, np=np, qml=qml)
        qubits = 2 * int(mycas.ncas)
        effective_active_electrons = int(sum(mycas.nelecas))
    elif hamiltonian_source == "molecular":
        h_qml, qubits = qml.qchem.molecular_hamiltonian(
            symbols,
            geometry,
            basis=basis,
            method="pyscf",
            active_electrons=int(active_electrons),
            active_orbitals=int(active_orbitals),
            charge=int(charge),
        )
        h_qml = _sanitize_hamiltonian(h_qml, np=np, qml=qml)
        mycas = None
        effective_active_electrons = int(active_electrons)
    else:
        raise ValueError("hamiltonian_source must be one of {'casci', 'molecular'}")

    h_qulacs, pauli_terms = _qml_operator_to_qulacs_operator(
        h_qml,
        qubits=qubits,
        qml=qml,
        operator_cls=Observable,
    )
    hf_bits = np.asarray(qml.qchem.hf_state(effective_active_electrons, qubits), dtype=int)

    return {
        **stack,
        "mol": mol,
        "mf": mf,
        "mycas": mycas,
        "h_qml": h_qml,
        "h_qulacs": h_qulacs,
        "pauli_terms": pauli_terms,
        "qubits": int(qubits),
        "hf_bits": hf_bits,
        "active_electrons": int(effective_active_electrons),
    }


def _build_projection_matrices(basis_states, h_qulacs, np, *, max_workers: Optional[int]):
    basis_array = np.stack([np.asarray(state.get_vector(), dtype=complex) for state in basis_states], axis=0)
    overlap = basis_array.conj() @ basis_array.T
    size = len(basis_states)
    h_proj = np.zeros((size, size), dtype=complex)

    def _compute_row(row_idx: int):
        values = []
        bra = basis_states[row_idx]
        for col_idx in range(row_idx, size):
            ket = basis_states[col_idx]
            values.append((col_idx, complex(h_qulacs.get_transition_amplitude(bra, ket))))
        return row_idx, values

    worker_count = 1 if max_workers is None else max(1, int(max_workers))
    if worker_count > 1 and size > 2:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(_compute_row, row_idx) for row_idx in range(size)]
            for future in as_completed(futures):
                row_idx, values = future.result()
                for col_idx, value in values:
                    h_proj[row_idx, col_idx] = value
                    if col_idx != row_idx:
                        h_proj[col_idx, row_idx] = np.conj(value)
    else:
        for row_idx in range(size):
            _, values = _compute_row(row_idx)
            for col_idx, value in values:
                h_proj[row_idx, col_idx] = value
                if col_idx != row_idx:
                    h_proj[col_idx, row_idx] = np.conj(value)

    return overlap, h_proj, basis_array


def _diagonalize_projected(overlap, h_proj, overlap_tol: float, np):
    s_vals, s_vecs = np.linalg.eigh((overlap + overlap.conj().T) / 2.0)
    keep = s_vals > float(overlap_tol)
    if not keep.any():
        raise ValueError("overlap matrix is numerically singular; basis collapsed")
    projector = s_vecs[:, keep] / np.sqrt(s_vals[keep])[None, :]
    h_ortho = projector.conj().T @ h_proj @ projector
    h_ortho = (h_ortho + h_ortho.conj().T) / 2.0
    return np.linalg.eigvalsh(h_ortho).real


def _project_hamiltonian_energies(basis_states, h_qulacs, overlap_tol: float, np, *, max_workers: Optional[int]):
    overlap, h_proj, basis_array = _build_projection_matrices(
        basis_states,
        h_qulacs,
        np,
        max_workers=max_workers,
    )
    energies = _diagonalize_projected(overlap, h_proj, overlap_tol, np)
    return energies, basis_array


def _build_time_evolution_circuit(qubits: int, pauli_terms, time_step: float, trotter_steps: int, qg, QuantumCircuit, np):
    circuit = QuantumCircuit(int(qubits))
    if abs(float(time_step)) < 1e-15:
        return circuit

    scaled_time = float(time_step) / float(trotter_steps)
    for _ in range(int(trotter_steps)):
        for term in pauli_terms:
            if len(term["indices"]) == 0:
                continue
            angle = -2.0 * float(np.real(term["coeff"])) * scaled_time
            if abs(angle) < 1e-15:
                continue
            circuit.add_gate(qg.PauliRotation(term["indices"], term["pauli_ids"], angle))
    return circuit


def _evolve_state(state, *, evolution_mode: str, sparse_hamiltonian, circuit, time_step: float, QuantumState, np, expm_multiply):
    if evolution_mode == "sparse":
        vector = expm_multiply(-1j * float(time_step) * sparse_hamiltonian, np.asarray(state.get_vector(), dtype=complex))
        return _state_from_vector(vector, qubits=state.get_qubit_count(), QuantumState=QuantumState, np=np)

    evolved = state.copy()
    circuit.update_quantum_state(evolved)
    return evolved


def _apply_dense_gate(state, wires, matrix, *, qubits: int, qg):
    qg.DenseMatrix(_dense_gate_wires(qubits, wires), matrix).update_quantum_state(state)


def _build_excitation_matrix(excitation, weight: float, *, ansatz_type: str, qml, np):
    excitation = [int(value) for value in excitation]
    angle = float(weight)
    if ansatz_type == "fermionic":
        if len(excitation) == 2:
            wires = list(range(excitation[0], excitation[1] + 1))
            op = qml.FermionicSingleExcitation(weight=angle, wires=wires)
        elif len(excitation) == 4:
            wires1 = list(range(excitation[0], excitation[1] + 1))
            wires2 = list(range(excitation[2], excitation[3] + 1))
            wires = wires1 + wires2
            op = qml.FermionicDoubleExcitation(weight=angle, wires1=wires1, wires2=wires2)
        else:
            raise ValueError(f"Unexpected fermionic excitation shape: {excitation!r}")
    elif ansatz_type == "qubit_excitation":
        if len(excitation) == 2:
            wires = [int(excitation[0]), int(excitation[1])]
            op = qml.SingleExcitation(angle, wires=wires)
        elif len(excitation) == 4:
            wires = [int(excitation[0]), int(excitation[1]), int(excitation[2]), int(excitation[3])]
            op = qml.DoubleExcitation(angle, wires=wires)
        else:
            raise ValueError(f"Unexpected qubit excitation shape: {excitation!r}")
    else:
        raise ValueError(f"Unsupported ansatz_type: {ansatz_type}")

    matrix = np.asarray(qml.matrix(op, wire_order=wires), dtype=complex)
    return wires, matrix


def _apply_excitation_sequence(state, params, excitations, *, ansatz_type: str, qubits: int, qml, qg, np):
    for weight, excitation in zip(params, excitations):
        if abs(float(weight)) < 1e-15:
            continue
        wires, matrix = _build_excitation_matrix(
            excitation,
            float(weight),
            ansatz_type=ansatz_type,
            qml=qml,
            np=np,
        )
        _apply_dense_gate(state, wires, matrix, qubits=qubits, qg=qg)
    return state


def _build_adapt_state(params, excitations, hf_bits, *, qubits: int, ansatz_type: str, QuantumState, qml, qg, np):
    state = _state_from_vector(_basis_vector_from_bits(hf_bits, qubits, np), qubits=qubits, QuantumState=QuantumState, np=np)
    return _apply_excitation_sequence(
        state,
        params,
        excitations,
        ansatz_type=ansatz_type,
        qubits=qubits,
        qml=qml,
        qg=qg,
        np=np,
    )


def _apply_qulacs_operator(operator, state, *, qubits: int, QuantumState):
    dst_state = QuantumState(int(qubits))
    try:
        operator.apply_to_state(state, dst_state)
    except TypeError:
        work_state = QuantumState(int(qubits))
        operator.apply_to_state(work_state, state, dst_state)
    return dst_state


def _commutator_score(state, h_state, h_qulacs, pool_operator, *, qubits: int, QuantumState, np):
    a_state = _apply_qulacs_operator(pool_operator, state, qubits=qubits, QuantumState=QuantumState)
    term_ha = h_qulacs.get_transition_amplitude(state, a_state)
    term_ah = pool_operator.get_transition_amplitude(state, h_state)
    return float(np.abs(2.0 * (term_ha - term_ah)))


def _build_adapt_compiled_circuit(
    excitations,
    hf_bits,
    *,
    ansatz_type: str,
    qubits: int,
    ParametricQuantumCircuit,
    qml,
    qg,
    np,
):
    circuit = ParametricQuantumCircuit(int(qubits))
    for wire, bit in enumerate(hf_bits):
        if int(bit):
            circuit.add_X_gate(_qulacs_wire(qubits, wire))

    parameter_groups = [[] for _ in range(len(excitations))]
    for user_idx, excitation in enumerate(excitations):
        _compile_single_parameter_factory_into_circuit(
            circuit,
            parameter_groups,
            int(user_idx),
            lambda theta, excitation=tuple(excitation): _make_excitation_operator(
                theta,
                excitation,
                ansatz_type=ansatz_type,
                qml=qml,
            ),
            qubits=qubits,
            qml=qml,
            qg=qg,
            np=np,
        )

    return _CompiledParametricCircuit(
        qubits=int(qubits),
        circuit=circuit,
        user_parameter_count=len(excitations),
        parameter_groups=parameter_groups,
    )


def _optimize_adapt_compiled_circuit(
    compiled_circuit: _CompiledParametricCircuit,
    params,
    *,
    h_qulacs,
    optimizer_method: str,
    optimizer_maxiter: int,
    minimize,
    QuantumState,
    np,
):
    if compiled_circuit.user_parameter_count == 0:
        state = _evaluate_compiled_state(
            compiled_circuit,
            np.zeros(0, dtype=float),
            QuantumState=QuantumState,
            np=np,
        )
        energy = float(np.real(h_qulacs.get_expectation_value(state)))
        return {
            "x": np.zeros(0, dtype=float),
            "fun": energy,
            "nit": 0,
            "success": True,
            "grad_norm": 0.0,
        }

    method = str(optimizer_method).strip().lower()
    x0 = np.asarray(params, dtype=float)

    def _cost_and_grad(user_values):
        compiled_circuit.set_user_parameters(user_values, np)
        state = QuantumState(int(compiled_circuit.qubits))
        state.set_zero_state()
        compiled_circuit.circuit.update_quantum_state(state)
        energy = float(np.real(h_qulacs.get_expectation_value(state)))
        raw_grad = np.asarray(compiled_circuit.circuit.backprop(h_qulacs), dtype=float)
        user_grad = _adjust_user_gradient(raw_grad, compiled_circuit.parameter_groups, np)
        return energy, user_grad

    if method == "adam":
        return _optimize_with_adam(
            x0,
            _cost_and_grad,
            maxiter=int(optimizer_maxiter),
            step_size=0.05,
            np=np,
        )

    def _objective(user_values):
        energy, grad = _cost_and_grad(user_values)
        return energy, grad

    result = minimize(
        _objective,
        x0,
        method=optimizer_method,
        jac=True,
        tol=1e-12,
        options={"disp": False, "maxiter": int(optimizer_maxiter)},
    )
    grad_norm = 0.0
    if hasattr(result, "jac") and result.jac is not None:
        grad_norm = float(np.linalg.norm(np.asarray(result.jac, dtype=float)))
    return {
        "x": np.asarray(result.x, dtype=float),
        "fun": float(result.fun),
        "nit": int(0 if getattr(result, "nit", None) is None else result.nit),
        "success": bool(getattr(result, "success", False)),
        "grad_norm": grad_norm,
    }


def _append_number_pair_phase_terms(circuit, parameter_groups, user_param_index: int, wire_a: int, wire_b: int, *, qubits: int):
    circuit.add_parametric_RZ_gate(_qulacs_wire(qubits, wire_a), 0.0)
    parameter_groups[int(user_param_index)].append((int(circuit.get_parameter_count() - 1), 0.5))
    circuit.add_parametric_RZ_gate(_qulacs_wire(qubits, wire_b), 0.0)
    parameter_groups[int(user_param_index)].append((int(circuit.get_parameter_count() - 1), 0.5))
    circuit.add_parametric_multi_Pauli_rotation_gate(
        _dense_gate_wires(qubits, [wire_a, wire_b]),
        [3, 3],
        0.0,
    )
    parameter_groups[int(user_param_index)].append((int(circuit.get_parameter_count() - 1), -0.5))


def _build_lucj_compiled_ansatz(
    *,
    active_orbitals: int,
    qubits: int,
    ParametricQuantumCircuit,
    qml,
    qg,
    np,
):
    n_pairs = max(0, int(active_orbitals) - 1)
    circuit = ParametricQuantumCircuit(int(qubits))
    parameter_groups = [[] for _ in range(3 * n_pairs)]

    for pair_idx in range(n_pairs):
        alpha_p = 2 * int(pair_idx)
        beta_p = alpha_p + 1
        alpha_q = alpha_p + 2
        beta_q = alpha_p + 3
        pair_wires = [alpha_p, beta_p, alpha_q, beta_q]

        orbital_index = int(pair_idx)
        _compile_single_parameter_factory_into_circuit(
            circuit,
            parameter_groups,
            orbital_index,
            lambda theta, wires=tuple(pair_wires): qml.OrbitalRotation(-float(theta), wires=list(wires)),
            qubits=qubits,
            qml=qml,
            qg=qg,
            np=np,
        )

        same_spin_index = n_pairs + int(pair_idx)
        _append_number_pair_phase_terms(
            circuit,
            parameter_groups,
            same_spin_index,
            alpha_p,
            alpha_q,
            qubits=qubits,
        )
        _append_number_pair_phase_terms(
            circuit,
            parameter_groups,
            same_spin_index,
            beta_p,
            beta_q,
            qubits=qubits,
        )

        opposite_spin_index = (2 * n_pairs) + int(pair_idx)
        _append_number_pair_phase_terms(
            circuit,
            parameter_groups,
            opposite_spin_index,
            alpha_p,
            beta_q,
            qubits=qubits,
        )
        _append_number_pair_phase_terms(
            circuit,
            parameter_groups,
            opposite_spin_index,
            beta_p,
            alpha_q,
            qubits=qubits,
        )

        _compile_single_parameter_factory_into_circuit(
            circuit,
            parameter_groups,
            orbital_index,
            lambda theta, wires=tuple(pair_wires): qml.OrbitalRotation(float(theta), wires=list(wires)),
            qubits=qubits,
            qml=qml,
            qg=qg,
            np=np,
        )

    return _CompiledParametricCircuit(
        qubits=int(qubits),
        circuit=circuit,
        user_parameter_count=3 * n_pairs,
        parameter_groups=parameter_groups,
    )


def _build_uccsd_compiled_ansatz(
    uccsd_metadata,
    *,
    qubits: int,
    ParametricQuantumCircuit,
    qml,
    qg,
    np,
):
    circuit = ParametricQuantumCircuit(int(qubits))
    parameter_groups = [[] for _ in range(len(uccsd_metadata))]

    for user_idx, meta in enumerate(uccsd_metadata):
        if meta["kind"] == "single":
            operator_factory = lambda theta, meta=meta: qml.FermionicSingleExcitation(
                float(theta),
                wires=list(meta["wires"]),
            )
        else:
            operator_factory = lambda theta, meta=meta: qml.FermionicDoubleExcitation(
                float(theta),
                wires1=list(meta["wires1"]),
                wires2=list(meta["wires2"]),
            )
        _compile_single_parameter_factory_into_circuit(
            circuit,
            parameter_groups,
            int(user_idx),
            operator_factory,
            qubits=qubits,
            qml=qml,
            qg=qg,
            np=np,
        )

    return _CompiledParametricCircuit(
        qubits=int(qubits),
        circuit=circuit,
        user_parameter_count=len(uccsd_metadata),
        parameter_groups=parameter_groups,
    )


def _build_linear_reference_state(reference_determinants, amplitudes, *, qubits: int, np):
    amplitudes = np.asarray(amplitudes, dtype=float)
    if amplitudes.ndim != 1 or amplitudes.size != len(reference_determinants):
        raise ValueError("reference amplitudes must align with reference determinants")
    state = np.zeros(2**int(qubits), dtype=complex)
    for amplitude, determinant in zip(amplitudes, reference_determinants):
        state[_bits_to_index(determinant)] += float(amplitude)
    norm = np.linalg.norm(state)
    if norm < 1e-14:
        raise ValueError("reference amplitudes produced a zero state")
    return state / norm


def _phase_fix_real_vector(vector, np):
    vec = np.asarray(vector, dtype=float).copy()
    if vec.size == 0:
        return vec
    pivot = int(np.argmax(np.abs(vec)))
    if vec[pivot] < 0:
        vec = -vec
    return vec


def _project_cvqe_reference(
    compiled_ansatz: _CompiledParametricCircuit,
    ansatz_values,
    reference_determinants,
    *,
    h_qulacs,
    QuantumState,
    np,
):
    basis_states = [
        _evaluate_compiled_state(
            compiled_ansatz,
            ansatz_values,
            QuantumState=QuantumState,
            np=np,
            init_bits=determinant,
        )
        for determinant in reference_determinants
    ]
    size = len(basis_states)
    projected_h = np.zeros((size, size), dtype=float)
    for row_idx, bra in enumerate(basis_states):
        for col_idx in range(row_idx, size):
            value = complex(h_qulacs.get_transition_amplitude(bra, basis_states[col_idx]))
            real_value = float(np.real(value))
            projected_h[row_idx, col_idx] = real_value
            projected_h[col_idx, row_idx] = real_value
    evals, evecs = np.linalg.eigh(projected_h)
    coeffs = _phase_fix_real_vector(evecs[:, 0], np)
    coeffs = coeffs / np.linalg.norm(coeffs)
    energy = float(evals[0])
    return energy, coeffs, basis_states, projected_h


def _evaluate_cvqe_energy_for_reference(
    compiled_ansatz: _CompiledParametricCircuit,
    ansatz_values,
    reference_determinants,
    reference_amplitudes,
    *,
    h_qulacs,
    QuantumState,
    np,
):
    compiled_ansatz.set_user_parameters(ansatz_values, np)
    return _evaluate_current_cvqe_energy_for_reference(
        compiled_ansatz,
        reference_determinants,
        reference_amplitudes,
        h_qulacs=h_qulacs,
        QuantumState=QuantumState,
        np=np,
    )


def _evaluate_current_cvqe_energy_for_reference(
    compiled_ansatz: _CompiledParametricCircuit,
    reference_determinants,
    reference_amplitudes,
    *,
    h_qulacs,
    QuantumState,
    np,
):
    state = _state_from_vector(
        _build_linear_reference_state(
            reference_determinants,
            reference_amplitudes,
            qubits=compiled_ansatz.qubits,
            np=np,
        ),
        qubits=compiled_ansatz.qubits,
        QuantumState=QuantumState,
        np=np,
    )
    compiled_ansatz.circuit.update_quantum_state(state)
    return float(np.real(h_qulacs.get_expectation_value(state))), state


def _parameter_shift_user_gradient(
    compiled_ansatz: _CompiledParametricCircuit,
    ansatz_values,
    energy_from_current_circuit: Callable[[], float],
    *,
    np,
):
    compiled_ansatz.set_user_parameters(ansatz_values, np)
    if compiled_ansatz.user_parameter_count == 0:
        return np.zeros(compiled_ansatz.user_parameter_count, dtype=float)
    circuit_param_count = int(compiled_ansatz.circuit.get_parameter_count())
    raw_grad = np.zeros(circuit_param_count, dtype=float)

    for param_idx in range(circuit_param_count):
        original = float(compiled_ansatz.circuit.get_parameter(param_idx))
        compiled_ansatz.circuit.set_parameter(param_idx, original + (np.pi / 2.0))
        plus_energy = float(energy_from_current_circuit())
        compiled_ansatz.circuit.set_parameter(param_idx, original - (np.pi / 2.0))
        minus_energy = float(energy_from_current_circuit())
        compiled_ansatz.circuit.set_parameter(param_idx, original)
        raw_grad[param_idx] = 0.5 * (plus_energy - minus_energy)

    return _adjust_user_gradient(raw_grad, compiled_ansatz.parameter_groups, np)


def _optimize_cvqe_reduced_objective(
    compiled_ansatz: _CompiledParametricCircuit,
    ansatz_values,
    reference_determinants,
    *,
    h_qulacs,
    optimizer_method: str,
    optimizer_maxiter: int,
    minimize,
    QuantumState,
    np,
):
    x0 = np.asarray(ansatz_values, dtype=float)
    cache = {}

    def _cost_and_grad(user_values):
        key = tuple(np.asarray(user_values, dtype=float))
        if key in cache:
            return cache[key]

        energy, coeffs, _basis_states, _projected_h = _project_cvqe_reference(
            compiled_ansatz,
            user_values,
            reference_determinants,
            h_qulacs=h_qulacs,
            QuantumState=QuantumState,
            np=np,
        )

        compiled_ansatz.set_user_parameters(user_values, np)

        def _energy_from_current_shifted_circuit():
            shifted_energy, _ = _evaluate_current_cvqe_energy_for_reference(
                compiled_ansatz,
                reference_determinants,
                coeffs,
                h_qulacs=h_qulacs,
                QuantumState=QuantumState,
                np=np,
            )
            return shifted_energy

        gradient = _parameter_shift_user_gradient(
            compiled_ansatz,
            user_values,
            _energy_from_current_shifted_circuit,
            np=np,
        )
        payload = (float(energy), np.asarray(gradient, dtype=float), np.asarray(coeffs, dtype=float))
        cache[key] = payload
        return payload

    method = str(optimizer_method).strip().lower()
    if method == "adam":
        result = _optimize_with_adam(
            x0,
            lambda values: _cost_and_grad(values)[:2],
            maxiter=int(optimizer_maxiter),
            step_size=0.05,
            np=np,
        )
        final_energy, _final_grad, final_coeffs = _cost_and_grad(result["x"])
        result.update({"fun": float(final_energy), "coeffs": np.asarray(final_coeffs, dtype=float)})
        return result

    def _objective(user_values):
        energy, grad, _coeffs = _cost_and_grad(user_values)
        return energy, grad

    result = minimize(
        _objective,
        x0,
        method=optimizer_method,
        jac=True,
        tol=1e-12,
        options={"disp": False, "maxiter": int(optimizer_maxiter)},
    )
    final_energy, final_grad, final_coeffs = _cost_and_grad(result.x)
    return {
        "x": np.asarray(result.x, dtype=float),
        "fun": float(final_energy),
        "nit": int(0 if getattr(result, "nit", None) is None else result.nit),
        "success": bool(getattr(result, "success", False)),
        "grad_norm": float(np.linalg.norm(np.asarray(final_grad, dtype=float))),
        "coeffs": np.asarray(final_coeffs, dtype=float),
    }


def qkud_qulacs(
    symbols: Sequence[str],
    geometry,
    *,
    n_steps: int,
    epsilon: float,
    active_electrons: int,
    active_orbitals: int,
    basis: str = "sto-3g",
    charge: int = 0,
    spin: int = 0,
    overlap_tol: float = 1e-10,
    normalize_basis: bool = True,
    basis_threshold: float = 0.0,
    return_min_energy_history: bool = False,
    evolution_mode: str = "sparse",
    trotter_steps: int = 1,
    max_workers: Optional[int] = None,
):
    """Qulacs-backed QKUD using exact sparse or Trotterized real-time updates.

    For larger qubit counts, prefer ``evolution_mode="trotter"`` to avoid
    materializing the Hamiltonian matrix.
    """
    if len(symbols) == 0:
        raise ValueError("symbols must be non-empty")
    if n_steps < 0:
        raise ValueError("n_steps must be >= 0")
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    if overlap_tol <= 0:
        raise ValueError("overlap_tol must be > 0")
    if trotter_steps < 1:
        raise ValueError("trotter_steps must be >= 1")
    if evolution_mode not in {"sparse", "trotter"}:
        raise ValueError("evolution_mode must be one of {'sparse', 'trotter'}")

    payload = _build_hamiltonian_payload(
        symbols,
        geometry,
        basis=basis,
        charge=charge,
        spin=spin,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
        hamiltonian_cutoff=1e-20,
    )
    np = payload["np"]
    QuantumState = payload["QuantumState"]
    QuantumCircuit = payload["QuantumCircuit"]
    qg = payload["qg"]
    expm_multiply = payload["expm_multiply"]
    qubits = payload["qubits"]
    h_qulacs = payload["h_qulacs"]

    current = _state_from_vector(
        _basis_vector_from_bits(payload["hf_bits"], qubits, np),
        qubits=qubits,
        QuantumState=QuantumState,
        np=np,
    )
    current = _threshold_state(current, basis_threshold, QuantumState, np)
    basis_states = [current.copy()]

    sparse_hamiltonian = h_qulacs.get_matrix().tocsr() if evolution_mode == "sparse" else None
    forward_circuit = None
    backward_circuit = None
    if evolution_mode == "trotter":
        forward_circuit = _build_time_evolution_circuit(
            qubits,
            payload["pauli_terms"],
            float(epsilon),
            trotter_steps,
            qg,
            QuantumCircuit,
            np,
        )
        backward_circuit = _build_time_evolution_circuit(
            qubits,
            payload["pauli_terms"],
            -float(epsilon),
            trotter_steps,
            qg,
            QuantumCircuit,
            np,
        )

    for _ in range(int(n_steps)):
        forward = _evolve_state(
            current,
            evolution_mode=evolution_mode,
            sparse_hamiltonian=sparse_hamiltonian,
            circuit=forward_circuit,
            time_step=float(epsilon),
            QuantumState=QuantumState,
            np=np,
            expm_multiply=expm_multiply,
        )
        backward = _evolve_state(
            current,
            evolution_mode=evolution_mode,
            sparse_hamiltonian=sparse_hamiltonian,
            circuit=backward_circuit,
            time_step=-float(epsilon),
            QuantumState=QuantumState,
            np=np,
            expm_multiply=expm_multiply,
        )
        vector = (1j * (np.asarray(forward.get_vector(), dtype=complex) - np.asarray(backward.get_vector(), dtype=complex))) / (2.0 * float(epsilon))
        if normalize_basis:
            norm = np.linalg.norm(vector)
            if norm < 1e-14:
                raise ValueError("QKUD vector has zero norm")
            vector = vector / norm
        current = _state_from_vector(vector, qubits=qubits, QuantumState=QuantumState, np=np)
        current = _threshold_state(current, basis_threshold, QuantumState, np)
        basis_states.append(current.copy())

    energies, basis_array = _project_hamiltonian_energies(
        basis_states,
        h_qulacs,
        overlap_tol,
        np,
        max_workers=max_workers,
    )
    if return_min_energy_history:
        min_history = []
        for end in range(2, len(basis_states) + 1):
            evals, _ = _project_hamiltonian_energies(
                basis_states[:end],
                h_qulacs,
                overlap_tol,
                np,
                max_workers=max_workers,
            )
            min_history.append(float(evals[0]))
        return energies, basis_array, np.asarray(min_history, dtype=float)
    return energies, basis_array


def qrte_qulacs(
    symbols: Sequence[str],
    geometry,
    *,
    delta_t: float,
    n_steps: int,
    active_electrons: int,
    active_orbitals: int,
    basis: str = "sto-3g",
    charge: int = 0,
    spin: int = 0,
    overlap_tol: float = 1e-10,
    basis_threshold: float = 0.0,
    return_min_energy_history: bool = False,
    evolution_mode: str = "sparse",
    trotter_steps: int = 1,
    max_workers: Optional[int] = None,
):
    """Qulacs-backed QRTE using exact sparse or Trotterized evolution.

    For larger qubit counts, prefer ``evolution_mode="trotter"`` to avoid
    materializing the Hamiltonian matrix.
    """
    if len(symbols) == 0:
        raise ValueError("symbols must be non-empty")
    if delta_t <= 0:
        raise ValueError("delta_t must be > 0")
    if n_steps < 0:
        raise ValueError("n_steps must be >= 0")
    if overlap_tol <= 0:
        raise ValueError("overlap_tol must be > 0")
    if trotter_steps < 1:
        raise ValueError("trotter_steps must be >= 1")
    if evolution_mode not in {"sparse", "trotter"}:
        raise ValueError("evolution_mode must be one of {'sparse', 'trotter'}")

    payload = _build_hamiltonian_payload(
        symbols,
        geometry,
        basis=basis,
        charge=charge,
        spin=spin,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
        hamiltonian_cutoff=1e-20,
    )
    np = payload["np"]
    QuantumState = payload["QuantumState"]
    QuantumCircuit = payload["QuantumCircuit"]
    qg = payload["qg"]
    expm_multiply = payload["expm_multiply"]
    qubits = payload["qubits"]
    h_qulacs = payload["h_qulacs"]

    current = _state_from_vector(
        _basis_vector_from_bits(payload["hf_bits"], qubits, np),
        qubits=qubits,
        QuantumState=QuantumState,
        np=np,
    )
    current = _threshold_state(current, basis_threshold, QuantumState, np)
    basis_states = [current.copy()]
    min_history = []

    sparse_hamiltonian = h_qulacs.get_matrix().tocsr() if evolution_mode == "sparse" else None
    circuit = None
    if evolution_mode == "trotter":
        circuit = _build_time_evolution_circuit(
            qubits,
            payload["pauli_terms"],
            float(delta_t),
            trotter_steps,
            qg,
            QuantumCircuit,
            np,
        )

    for _ in range(int(n_steps)):
        current = _evolve_state(
            current,
            evolution_mode=evolution_mode,
            sparse_hamiltonian=sparse_hamiltonian,
            circuit=circuit,
            time_step=float(delta_t),
            QuantumState=QuantumState,
            np=np,
            expm_multiply=expm_multiply,
        )
        current = _normalize_state(current, QuantumState, np)
        current = _threshold_state(current, basis_threshold, QuantumState, np)
        basis_states.append(current.copy())
        if return_min_energy_history:
            evals, _ = _project_hamiltonian_energies(
                basis_states,
                h_qulacs,
                overlap_tol,
                np,
                max_workers=max_workers,
            )
            min_history.append(float(evals[0]))

    energies, basis_array = _project_hamiltonian_energies(
        basis_states,
        h_qulacs,
        overlap_tol,
        np,
        max_workers=max_workers,
    )
    times = np.arange(int(n_steps) + 1, dtype=float) * float(delta_t)
    if return_min_energy_history:
        return energies, basis_array, times, np.asarray(min_history, dtype=float)
    return energies, basis_array, times


def qrte_pmte_qulacs(
    symbols: Sequence[str],
    geometry,
    *,
    delta_t: float,
    n_steps: int,
    active_electrons: int,
    active_orbitals: int,
    basis: str = "sto-3g",
    charge: int = 0,
    spin: int = 0,
    overlap_tol: float = 1e-10,
    basis_threshold: float = 0.0,
    return_min_energy_history: bool = False,
    evolution_mode: str = "sparse",
    trotter_steps: int = 1,
    max_workers: Optional[int] = None,
):
    """Qulacs-backed symmetric QRTE basis growth.

    For larger qubit counts, prefer ``evolution_mode="trotter"`` to avoid
    materializing the Hamiltonian matrix.
    """
    if len(symbols) == 0:
        raise ValueError("symbols must be non-empty")
    if delta_t <= 0:
        raise ValueError("delta_t must be > 0")
    if n_steps < 0:
        raise ValueError("n_steps must be >= 0")
    if overlap_tol <= 0:
        raise ValueError("overlap_tol must be > 0")
    if trotter_steps < 1:
        raise ValueError("trotter_steps must be >= 1")
    if evolution_mode not in {"sparse", "trotter"}:
        raise ValueError("evolution_mode must be one of {'sparse', 'trotter'}")

    payload = _build_hamiltonian_payload(
        symbols,
        geometry,
        basis=basis,
        charge=charge,
        spin=spin,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
        hamiltonian_cutoff=1e-20,
    )
    np = payload["np"]
    QuantumState = payload["QuantumState"]
    QuantumCircuit = payload["QuantumCircuit"]
    qg = payload["qg"]
    expm_multiply = payload["expm_multiply"]
    qubits = payload["qubits"]
    h_qulacs = payload["h_qulacs"]

    psi0 = _state_from_vector(
        _basis_vector_from_bits(payload["hf_bits"], qubits, np),
        qubits=qubits,
        QuantumState=QuantumState,
        np=np,
    )
    psi0 = _threshold_state(psi0, basis_threshold, QuantumState, np)
    basis_states = [psi0.copy()]
    psi_plus = psi0.copy()
    psi_minus = psi0.copy()
    min_history = []

    sparse_hamiltonian = h_qulacs.get_matrix().tocsr() if evolution_mode == "sparse" else None
    plus_circuit = None
    minus_circuit = None
    if evolution_mode == "trotter":
        plus_circuit = _build_time_evolution_circuit(
            qubits,
            payload["pauli_terms"],
            float(delta_t),
            trotter_steps,
            qg,
            QuantumCircuit,
            np,
        )
        minus_circuit = _build_time_evolution_circuit(
            qubits,
            payload["pauli_terms"],
            -float(delta_t),
            trotter_steps,
            qg,
            QuantumCircuit,
            np,
        )

    for _ in range(int(n_steps)):
        psi_plus = _evolve_state(
            psi_plus,
            evolution_mode=evolution_mode,
            sparse_hamiltonian=sparse_hamiltonian,
            circuit=plus_circuit,
            time_step=float(delta_t),
            QuantumState=QuantumState,
            np=np,
            expm_multiply=expm_multiply,
        )
        psi_plus = _normalize_state(psi_plus, QuantumState, np)
        psi_plus = _threshold_state(psi_plus, basis_threshold, QuantumState, np)

        psi_minus = _evolve_state(
            psi_minus,
            evolution_mode=evolution_mode,
            sparse_hamiltonian=sparse_hamiltonian,
            circuit=minus_circuit,
            time_step=-float(delta_t),
            QuantumState=QuantumState,
            np=np,
            expm_multiply=expm_multiply,
        )
        psi_minus = _normalize_state(psi_minus, QuantumState, np)
        psi_minus = _threshold_state(psi_minus, basis_threshold, QuantumState, np)

        basis_states.extend([psi_plus.copy(), psi_minus.copy()])
        if return_min_energy_history:
            evals, _ = _project_hamiltonian_energies(
                basis_states,
                h_qulacs,
                overlap_tol,
                np,
                max_workers=max_workers,
            )
            min_history.append(float(evals[0]))

    energies, basis_array = _project_hamiltonian_energies(
        basis_states,
        h_qulacs,
        overlap_tol,
        np,
        max_workers=max_workers,
    )
    times = [0.0]
    for step in range(1, int(n_steps) + 1):
        value = float(step) * float(delta_t)
        times.extend([value, -value])
    times = np.asarray(times, dtype=float)
    if return_min_energy_history:
        return energies, basis_array, times, np.asarray(min_history, dtype=float)
    return energies, basis_array, times


def adapt_vqe_qulacs(
    symbols: Sequence[str],
    geometry,
    *,
    adapt_it: int,
    basis: str = "sto-6g",
    charge: int = 0,
    spin: int = 0,
    active_electrons: int,
    active_orbitals: int,
    hamiltonian_cutoff: float = 1e-20,
    hamiltonian_source: str = "casci",
    pool_type: str = "fermionic_sd",
    pool_sample_size: Optional[int] = None,
    pool_seed: Optional[int] = None,
    parallel_gradients: bool = True,
    max_workers: Optional[int] = None,
    gradient_chunk_size: Optional[int] = None,
    optimizer_method: str = "BFGS",
    optimizer_maxiter: int = 100_000_000,
    return_history: bool = False,
    iteration_callback: Optional[Callable[[Mapping[str, object]], None]] = None,
):
    """Qulacs-backed analytic ADAPT-VQE with threaded commutator scoring."""
    if len(symbols) == 0:
        raise ValueError("symbols must be non-empty")
    if adapt_it < 0:
        raise ValueError("adapt_it must be >= 0")
    if hamiltonian_cutoff < 0:
        raise ValueError("hamiltonian_cutoff must be >= 0")
    if pool_sample_size is not None and pool_sample_size <= 0:
        raise ValueError("pool_sample_size must be > 0")
    if max_workers is not None and max_workers <= 0:
        raise ValueError("max_workers must be > 0")
    if gradient_chunk_size is not None and gradient_chunk_size <= 0:
        raise ValueError("gradient_chunk_size must be > 0")

    payload = _build_hamiltonian_payload(
        symbols,
        geometry,
        basis=basis,
        charge=charge,
        spin=spin,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
        hamiltonian_cutoff=hamiltonian_cutoff,
        hamiltonian_source=hamiltonian_source,
    )
    np = payload["np"]
    qml = payload["qml"]
    QuantumState = payload["QuantumState"]
    qg = payload["qg"]
    minimize = payload["minimize"]
    qubits = payload["qubits"]
    h_qulacs = payload["h_qulacs"]
    hf_bits = [int(bit) for bit in np.asarray(payload["hf_bits"], dtype=int)]

    pool_type_canonical = _normalize_pool_type(pool_type)
    ansatz_type = _ansatz_type_from_pool_type(pool_type_canonical)
    pool_excitations, operator_pool_ops = _build_operator_pool(
        qml,
        int(payload["active_electrons"]),
        qubits,
        pool_type_canonical,
    )
    pool_qulacs_ops = [
        _qml_operator_to_qulacs_operator(pool_op, qubits, qml=qml, operator_cls=payload["GeneralQuantumOperator"])[0]
        for pool_op in operator_pool_ops
    ]

    params = np.zeros(0, dtype=float)
    ash_excitation = []
    energies = []
    history = []
    rng = np.random.default_rng(pool_seed)

    worker_count = _resolve_worker_count(max_workers)

    for iteration in range(int(adapt_it)):
        current_compiled = _build_adapt_compiled_circuit(
            ash_excitation,
            hf_bits,
            ansatz_type=ansatz_type,
            qubits=qubits,
            ParametricQuantumCircuit=payload["ParametricQuantumCircuit"],
            qml=qml,
            qg=qg,
            np=np,
        )
        current_state = _evaluate_compiled_state(
            current_compiled,
            params,
            QuantumState=QuantumState,
            np=np,
        )
        h_state = _apply_qulacs_operator(h_qulacs, current_state, qubits=qubits, QuantumState=QuantumState)

        if pool_sample_size is None or pool_sample_size >= len(pool_qulacs_ops):
            candidate_indices = list(range(len(pool_qulacs_ops)))
        else:
            candidate_indices = [int(idx) for idx in rng.choice(len(pool_qulacs_ops), size=int(pool_sample_size), replace=False)]

        candidate_positions = list(enumerate(candidate_indices))
        scores = {}

        def _score_chunk(chunk_positions):
            chunk_scores = {}
            for position, pool_idx in chunk_positions:
                chunk_scores[position] = _commutator_score(
                    current_state,
                    h_state,
                    h_qulacs,
                    pool_qulacs_ops[pool_idx],
                    qubits=qubits,
                    QuantumState=QuantumState,
                    np=np,
                )
            return chunk_scores

        if parallel_gradients and worker_count > 1 and len(candidate_positions) > 1:
            chunk_size = _resolve_chunk_size(
                total_items=len(candidate_positions),
                worker_count=worker_count,
                user_chunk_size=gradient_chunk_size,
            )
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = [
                    executor.submit(_score_chunk, chunk)
                    for chunk in _iter_chunks(candidate_positions, chunk_size)
                ]
                for future in as_completed(futures):
                    scores.update(future.result())
        else:
            scores.update(_score_chunk(candidate_positions))

        best_value = float("-inf")
        best_excitation = None
        best_index = None
        for position, pool_idx in candidate_positions:
            current_value = float(scores[position])
            if current_value > best_value:
                best_value = current_value
                best_excitation = [int(value) for value in pool_excitations[pool_idx]]
                best_index = int(pool_idx)

        if best_excitation is None or best_index is None:
            raise RuntimeError("No ADAPT operator selected.")

        ash_excitation.append(best_excitation)
        x0 = np.concatenate([np.asarray(params, dtype=float), np.array([0.0], dtype=float)])
        compiled_for_optimization = _build_adapt_compiled_circuit(
            ash_excitation,
            hf_bits,
            ansatz_type=ansatz_type,
            qubits=qubits,
            ParametricQuantumCircuit=payload["ParametricQuantumCircuit"],
            qml=qml,
            qg=qg,
            np=np,
        )
        result = _optimize_adapt_compiled_circuit(
            compiled_for_optimization,
            x0,
            h_qulacs=h_qulacs,
            optimizer_method=optimizer_method,
            optimizer_maxiter=optimizer_maxiter,
            minimize=minimize,
            QuantumState=QuantumState,
            np=np,
        )
        params = np.asarray(result["x"], dtype=float)
        energy = float(result["fun"])
        energies.append(energy)

        snapshot = {
            "iteration": int(iteration + 1),
            "selected_pool_index": int(best_index),
            "selected_excitation": [int(value) for value in best_excitation],
            "adapt_max_gradient": float(best_value),
            "energy": float(energy),
            "params": np.asarray(params, dtype=float).copy(),
            "optimizer_method": str(optimizer_method),
            "optimizer_nit": int(result["nit"]),
            "optimizer_success": bool(result["success"]),
            "optimizer_grad_norm": float(result["grad_norm"]),
            "ash_excitation": [[int(value) for value in excitation] for excitation in ash_excitation],
        }
        if return_history:
            history.append(snapshot)
        if iteration_callback is not None:
            iteration_callback(snapshot)

    if return_history:
        return params, ash_excitation, energies, history
    return params, ash_excitation, energies


def cvqe_qulacs(
    symbols: Sequence[str],
    geometry,
    *,
    adapt_it: int,
    ansatz: str = "lucj",
    basis: str = "sto-3g",
    charge: int = 0,
    spin: int = 0,
    active_electrons: int,
    active_orbitals: int,
    shots: Optional[int] = 0,
    optimizer_method: str = "BFGS",
    optimizer_maxiter: int = 100,
    hamiltonian_cutoff: float = 1e-20,
    selection_seed: Optional[int] = None,
    print_progress: bool = True,
    return_details: bool = False,
    resume_state: Optional[dict[str, object]] = None,
    checkpoint_path: Optional[str] = None,
    iteration_callback: Optional[Callable[[dict[str, object]], None]] = None,
):
    """Qulacs-backed exact-state CVQE."""
    _validate_cvqe_inputs(
        symbols=symbols,
        geometry=geometry,
        adapt_it=adapt_it,
        shots=shots,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
        spin=spin,
    )
    ansatz = str(ansatz).strip().lower()
    if ansatz not in {"lucj", "uccsd"}:
        raise ValueError("ansatz must be one of {'lucj', 'uccsd'}")

    stack = _import_optional_stack()
    np = stack["np"]
    qml = stack["qml"]
    pyscf = stack["pyscf"]
    cc = stack["cc"]
    gto = stack["gto"]
    mcscf = stack["mcscf"]
    scf = stack["scf"]
    Observable = stack["Observable"]
    QuantumState = stack["QuantumState"]
    qg = stack["qg"]
    minimize = stack["minimize"]

    atom = [(symbols[i], tuple(float(x) for x in geometry[i])) for i in range(len(symbols))]
    mol_ref = gto.Mole()
    mol_ref.atom = atom
    mol_ref.unit = "Angstrom"
    mol_ref.basis = basis
    mol_ref.charge = charge
    mol_ref.spin = spin
    mol_ref.symmetry = False
    mol_ref.build()

    mf_ref = scf.RHF(mol_ref)
    mf_ref.level_shift = 0.5
    mf_ref.diis_space = 12
    mf_ref.max_cycle = 100
    mf_ref.kernel()
    if not mf_ref.converged:
        mf_ref = scf.newton(mf_ref).run()

    mycc = cc.CCSD(mf_ref)
    mycc.kernel()

    mycas_ref = mcscf.CASCI(mf_ref, active_orbitals, active_electrons)
    h1ecas, ecore = mycas_ref.get_h1eff(mf_ref.mo_coeff)
    h2ecas = mycas_ref.get_h2eff(mf_ref.mo_coeff)

    ncas = int(mycas_ref.ncas)
    two_mo = pyscf.ao2mo.restore("1", h2ecas, norb=ncas)
    two_mo = np.swapaxes(two_mo, 1, 3)
    h_fermionic = qml.qchem.fermionic_observable(
        np.array([ecore]),
        h1ecas,
        two_mo,
        cutoff=hamiltonian_cutoff,
    )
    h_qml = qml.jordan_wigner(h_fermionic)
    h_qml = _sanitize_hamiltonian(h_qml, np=np, qml=qml)
    qubits = 2 * ncas
    h_qulacs, _pauli_terms = _qml_operator_to_qulacs_operator(
        h_qml,
        qubits=qubits,
        qml=qml,
        operator_cls=Observable,
    )

    hf_bits = np.asarray(qml.qchem.hf_state(active_electrons, qubits), dtype=int)
    hf_bits_list = [int(bit) for bit in hf_bits]
    hf_index = _bits_to_index(hf_bits_list)
    occupancy_table = np.asarray([_index_to_bits(index, qubits) for index in range(2**qubits)], dtype=float)
    fixed_electron_indices = np.where(np.sum(occupancy_table, axis=1).astype(int) == int(active_electrons))[0]

    t1_active, t2_active = _extract_active_ccsd_amplitudes(
        mycc,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
        np=np,
    )
    n_lucj_pairs = max(0, active_orbitals - 1)
    uccsd_metadata = None
    if ansatz == "lucj":
        current_ansatz_payload = _initialize_lucj_params(
            t1_active,
            t2_active,
            active_orbitals=active_orbitals,
            active_electrons=active_electrons,
            np=np,
        )
    else:
        uccsd_metadata = _build_uccsd_excitation_metadata(
            active_electrons=active_electrons,
            qubits=qubits,
            qml=qml,
        )
        current_ansatz_payload = _initialize_uccsd_params(
            t1_active,
            t2_active,
            uccsd_metadata,
            active_electrons=active_electrons,
            np=np,
        )

    added_determinants: list[list[int]] = []
    selected_indices = {int(hf_index)}
    rng = np.random.default_rng(selection_seed)
    energies: list[float] = []
    history: list[dict[str, object]] = []

    if ansatz == "lucj":
        compiled_ansatz = _build_lucj_compiled_ansatz(
            active_orbitals=active_orbitals,
            qubits=qubits,
            ParametricQuantumCircuit=stack["ParametricQuantumCircuit"],
            qml=qml,
            qg=qg,
            np=np,
        )
        current_ansatz_values = np.concatenate(
            [np.asarray(block, dtype=float) for block in current_ansatz_payload]
        )
    else:
        compiled_ansatz = _build_uccsd_compiled_ansatz(
            uccsd_metadata,
            qubits=qubits,
            ParametricQuantumCircuit=stack["ParametricQuantumCircuit"],
            qml=qml,
            qg=qg,
            np=np,
        )
        current_ansatz_values = np.asarray(current_ansatz_payload, dtype=float)

    def _ansatz_values_to_payload(values):
        values = np.asarray(values, dtype=float)
        if ansatz == "lucj":
            return (
                values[:n_lucj_pairs],
                values[n_lucj_pairs : 2 * n_lucj_pairs],
                values[2 * n_lucj_pairs : 3 * n_lucj_pairs],
            )
        return values

    def _pack_all_params(det_coeffs, ansatz_values):
        ansatz_payload = _ansatz_values_to_payload(ansatz_values)
        if ansatz == "lucj":
            return np.concatenate(
                [
                    np.asarray(det_coeffs, dtype=float),
                    np.asarray(ansatz_payload[0], dtype=float),
                    np.asarray(ansatz_payload[1], dtype=float),
                    np.asarray(ansatz_payload[2], dtype=float),
                ]
            )
        return _pack_uccsd_params(det_coeffs, ansatz_payload, np)

    def _unpack_all_params(flat_params, n_det_params: int):
        flat_params = np.asarray(flat_params, dtype=float)
        if ansatz == "lucj":
            det_coeffs, orbital_angles, same_spin, opposite_spin = _unpack_params(
                flat_params,
                n_det_params=n_det_params,
                n_orb_pairs=n_lucj_pairs,
                np=np,
            )
            return det_coeffs, np.concatenate([orbital_angles, same_spin, opposite_spin])
        det_coeffs, uccsd_weights = _unpack_uccsd_params(
            flat_params,
            n_det_params=n_det_params,
            n_uccsd_params=len(uccsd_metadata),
            np=np,
        )
        return det_coeffs, np.asarray(uccsd_weights, dtype=float)

    def _parameter_slices_for_count(n_det_params: int):
        if ansatz == "lucj":
            return _parameter_slices(n_det_params=n_det_params, n_orb_pairs=n_lucj_pairs)
        return _parameter_slices_uccsd(
            n_det_params=n_det_params,
            n_uccsd_params=len(uccsd_metadata),
        )

    def _reference_determinants():
        return [[int(bit) for bit in hf_bits_list]] + [[int(bit) for bit in det] for det in added_determinants]

    def _determinant_coeffs_from_reference(reference_amplitudes):
        reference_amplitudes = np.asarray(reference_amplitudes, dtype=float)
        if reference_amplitudes.size <= 1:
            return np.zeros(0, dtype=float)
        scale = float(reference_amplitudes[0])
        if abs(scale) < 1e-12:
            scale = 1.0
        return np.asarray(reference_amplitudes[1:] / scale, dtype=float)

    current_reference_amplitudes = np.array([1.0], dtype=float)
    initial_energy, current_state = _evaluate_cvqe_energy_for_reference(
        compiled_ansatz,
        current_ansatz_values,
        _reference_determinants(),
        current_reference_amplitudes,
        h_qulacs=h_qulacs,
        QuantumState=QuantumState,
        np=np,
    )
    current_det_coeffs = _determinant_coeffs_from_reference(current_reference_amplitudes)
    current_params = _pack_all_params(current_det_coeffs, current_ansatz_values)

    def _build_details_payload(*, plain: bool):
        details = {
            "ansatz": str(ansatz),
            "backend": "qulacs",
            "initial_energy": float(initial_energy),
            "hf_determinant": [int(bit) for bit in hf_bits_list],
            "reference_determinants": _reference_determinants(),
            "reference_amplitudes": np.asarray(current_reference_amplitudes, dtype=float).copy(),
            "fixed_electron_sector_size": int(len(fixed_electron_indices)),
            "parameter_slices": _parameter_slices_for_count(len(added_determinants)),
            "history": history,
        }
        if ansatz == "uccsd":
            details["uccsd_excitations"] = [
                [int(idx) for idx in meta["excitation"]] for meta in uccsd_metadata
            ]
        if plain:
            return _to_plain_data(details, np)
        return details

    def _build_checkpoint_payload():
        return {
            "version": 2,
            "config": {
                "symbols": [str(symbol) for symbol in symbols],
                "geometry": _to_plain_data(geometry, np),
                "adapt_it": int(adapt_it),
                "ansatz": str(ansatz),
                "basis": str(basis),
                "charge": int(charge),
                "spin": int(spin),
                "active_electrons": int(active_electrons),
                "active_orbitals": int(active_orbitals),
                "shots": None if shots is None else int(shots),
                "optimizer_method": str(optimizer_method),
                "optimizer_maxiter": int(optimizer_maxiter),
                "hamiltonian_cutoff": float(hamiltonian_cutoff),
                "selection_seed": None if selection_seed is None else int(selection_seed),
                "backend": "qulacs",
            },
            "ansatz": str(ansatz),
            "backend": "qulacs",
            "completed_iterations": int(len(energies)),
            "current_params": _to_plain_data(np.asarray(current_params, dtype=float), np),
            "added_determinants": [[int(bit) for bit in determinant] for determinant in added_determinants],
            "energies": [float(energy) for energy in energies],
            "details": _build_details_payload(plain=True),
        }

    def _emit_progress():
        payload = _build_checkpoint_payload()
        if checkpoint_path is not None:
            _write_json_payload(checkpoint_path, payload)
        if iteration_callback is not None:
            iteration_callback(payload)

    if resume_state is not None:
        resume_details = dict(resume_state.get("details", {}))
        added_determinants = [[int(bit) for bit in determinant] for determinant in resume_state.get("added_determinants", [])]
        energies = [float(energy) for energy in resume_state.get("energies", [])]
        history = _restore_history(resume_details.get("history", []), np)
        current_params = np.asarray(resume_state["current_params"], dtype=float)
        _resume_det_coeffs, current_ansatz_values = _unpack_all_params(
            current_params,
            n_det_params=len(added_determinants),
        )
        selected_indices = {int(hf_index)}
        for determinant in added_determinants:
            selected_indices.add(_bits_to_index(determinant))
        _, current_reference_amplitudes, _basis_states, _projected_h = _project_cvqe_reference(
            compiled_ansatz,
            current_ansatz_values,
            _reference_determinants(),
            h_qulacs=h_qulacs,
            QuantumState=QuantumState,
            np=np,
        )
        resumed_energy, current_state = _evaluate_cvqe_energy_for_reference(
            compiled_ansatz,
            current_ansatz_values,
            _reference_determinants(),
            current_reference_amplitudes,
            h_qulacs=h_qulacs,
            QuantumState=QuantumState,
            np=np,
        )
        current_det_coeffs = _determinant_coeffs_from_reference(current_reference_amplitudes)
        current_params = _pack_all_params(current_det_coeffs, current_ansatz_values)
        initial_energy = float(resume_details.get("initial_energy", initial_energy))
        if len(energies) > 0 and abs(float(energies[-1]) - float(resumed_energy)) > 1e-8:
            energies[-1] = float(resumed_energy)

    if print_progress:
        if len(energies) == 0:
            print(f"Initial CVQE energy (Qulacs): {initial_energy}", flush=True)
        else:
            print(
                f"Resuming Qulacs CVQE from iteration {len(energies)} with current energy {energies[-1]}",
                flush=True,
            )

    for iteration in range(len(energies), int(adapt_it)):
        probabilities = np.abs(np.asarray(current_state.get_vector(), dtype=complex)) ** 2
        selected_index, selected_metric, selected_exact_prob, counts = _select_new_determinant(
            probabilities,
            selected_indices,
            candidate_indices=fixed_electron_indices,
            shots=shots,
            rng=rng,
            np=np,
        )
        if selected_index is None:
            if print_progress:
                print("No additional determinant found; stopping early.", flush=True)
            break

        selected_bits = _index_to_bits(int(selected_index), qubits)
        added_determinants.append(selected_bits)
        selected_indices.add(int(selected_index))

        result = _optimize_cvqe_reduced_objective(
            compiled_ansatz,
            current_ansatz_values,
            _reference_determinants(),
            h_qulacs=h_qulacs,
            optimizer_method=optimizer_method,
            optimizer_maxiter=optimizer_maxiter,
            minimize=minimize,
            QuantumState=QuantumState,
            np=np,
        )
        current_ansatz_values = np.asarray(result["x"], dtype=float)
        current_reference_amplitudes = np.asarray(result["coeffs"], dtype=float)
        energy, current_state = _evaluate_cvqe_energy_for_reference(
            compiled_ansatz,
            current_ansatz_values,
            _reference_determinants(),
            current_reference_amplitudes,
            h_qulacs=h_qulacs,
            QuantumState=QuantumState,
            np=np,
        )
        energies.append(float(energy))

        current_det_coeffs = _determinant_coeffs_from_reference(current_reference_amplitudes)
        current_params = _pack_all_params(current_det_coeffs, current_ansatz_values)

        entry = {
            "iteration": int(iteration + 1),
            "ansatz": str(ansatz),
            "backend": "qulacs",
            "selected_determinant": [int(bit) for bit in selected_bits],
            "selected_index": int(selected_index),
            "selection_mode": "exact" if shots is None or int(shots) == 0 else "sampled",
            "selection_shots": int(0 if shots is None else shots),
            "selection_metric": float(selected_metric),
            "selected_exact_probability": float(selected_exact_prob),
            "energy": float(energy),
            "optimizer_method": str(optimizer_method),
            "optimizer_maxiter": int(optimizer_maxiter),
            "optimizer_nit": int(result["nit"]),
            "optimizer_success": bool(result["success"]),
            "optimizer_grad_norm": float(result["grad_norm"]),
            "determinant_coeffs": np.asarray(current_det_coeffs, dtype=float).copy(),
            "reference_amplitudes": np.asarray(current_reference_amplitudes, dtype=float).copy(),
            "sample_counts": None if counts is None else np.asarray(counts, dtype=int),
        }
        if ansatz == "lucj":
            orbital_angles, same_spin, opposite_spin = _ansatz_values_to_payload(current_ansatz_values)
            entry.update(
                {
                    "lucj_orbital": np.asarray(orbital_angles, dtype=float).copy(),
                    "lucj_same_spin": np.asarray(same_spin, dtype=float).copy(),
                    "lucj_opposite_spin": np.asarray(opposite_spin, dtype=float).copy(),
                }
            )
        else:
            entry["uccsd_weights"] = np.asarray(current_ansatz_values, dtype=float).copy()
        history.append(entry)
        _emit_progress()

        if print_progress:
            print("Energies are", energies, flush=True)

    if return_details:
        return current_params, added_determinants, energies, _build_details_payload(plain=False)
    return current_params, added_determinants, energies
