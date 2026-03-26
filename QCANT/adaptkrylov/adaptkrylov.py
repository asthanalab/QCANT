"""ADAPT-VQE with exact first/second-order Krylov post-processing."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Mapping, Optional, Sequence

from ..adapt.adaptvqe import (
    _ansatz_type_from_pool_type,
    _apply_excitation_gate,
    _normalize_pool_type,
    _resolve_worker_count,
    adapt_vqe,
)
from ..qulacs_accel import (
    _adapt_vqe_qulacs_from_payload,
    _apply_qulacs_operator,
    _build_adapt_compiled_circuit,
    _build_hamiltonian_payload,
    _evaluate_compiled_state,
    _project_hamiltonian_energies,
)


def _make_device(qml, name: Optional[str], wires: int):
    """Create an analytic PennyLane device for exact-state reconstruction."""
    if name is not None:
        try:
            return qml.device(name, wires=wires)
        except Exception:
            return qml.device("default.qubit", wires=wires)
    try:
        return qml.device("lightning.qubit", wires=wires)
    except Exception:
        return qml.device("default.qubit", wires=wires)


def _build_hamiltonian_qml(
    *,
    symbols: Sequence[str],
    geometry,
    basis: str,
    charge: int,
    spin: int,
    active_electrons: int,
    active_orbitals: int,
    hamiltonian_cutoff: float,
    hamiltonian_source: str,
):
    """Build the active-space Hamiltonian for PennyLane-based Krylov diagnostics."""
    try:
        import numpy as np
        import pennylane as qml
        import pyscf
        from pyscf import gto, mcscf, scf
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "adaptKrylov requires dependencies. Install at least: "
            "`pip install numpy scipy pennylane pyscf`."
        ) from exc

    hamiltonian_source = str(hamiltonian_source).strip().lower()
    if hamiltonian_source not in {"casci", "molecular"}:
        raise ValueError("hamiltonian_source must be one of {'casci', 'molecular'}")

    if hamiltonian_source == "casci":
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

        mycas_ref = mcscf.CASCI(mf_ref, active_orbitals, active_electrons)
        h1ecas, ecore = mycas_ref.get_h1eff(mf_ref.mo_coeff)
        h2ecas = mycas_ref.get_h2eff(mf_ref.mo_coeff)

        ncas = int(mycas_ref.ncas)
        two_mo = pyscf.ao2mo.restore("1", h2ecas, norb=ncas)
        two_mo = np.swapaxes(two_mo, 1, 3)

        one_mo = h1ecas
        core_constant = np.array([ecore])

        h_fermionic = qml.qchem.fermionic_observable(
            core_constant, one_mo, two_mo, cutoff=hamiltonian_cutoff
        )
        hamiltonian = qml.jordan_wigner(h_fermionic)
        qubits = 2 * ncas
        active_electrons = int(sum(mycas_ref.nelecas))
    else:
        hamiltonian, qubits = qml.qchem.molecular_hamiltonian(
            symbols,
            geometry,
            basis=basis,
            method="pyscf",
            active_electrons=int(active_electrons),
            active_orbitals=int(active_orbitals),
            charge=int(charge),
        )
        active_electrons = int(active_electrons)

    hamiltonian_matrix = qml.matrix(hamiltonian, wire_order=range(qubits))
    hamiltonian_matrix = np.asarray(hamiltonian_matrix, dtype=complex)
    hamiltonian_matrix = 0.5 * (hamiltonian_matrix + hamiltonian_matrix.conj().T)

    return qml, np, hamiltonian_matrix, int(qubits), int(active_electrons)


def _project_ground_energy(np, basis_vectors, hamiltonian_matrix, overlap_tol: float) -> tuple[float, int]:
    """Solve the generalized eigenproblem in the span of ``basis_vectors``."""
    basis = np.asarray(basis_vectors, dtype=complex)
    overlap = basis.conj() @ basis.T
    h_proj = basis.conj() @ (hamiltonian_matrix @ basis.T)

    s_vals, s_vecs = np.linalg.eigh(overlap)
    keep = s_vals > float(overlap_tol)
    if not keep.any():
        raise ValueError("overlap matrix is numerically singular; Krylov basis collapsed")

    transform = s_vecs[:, keep] / np.sqrt(s_vals[keep])[None, :]
    h_ortho = transform.conj().T @ h_proj @ transform
    evals = np.linalg.eigvalsh(h_ortho).real
    return float(evals[0]), int(np.count_nonzero(keep))


def _solve_krylov_orders_dense(np, psi, hamiltonian_matrix, overlap_tol: float) -> tuple[float, float, int, int]:
    """Compute first- and second-order Krylov energies for ``psi`` with dense algebra."""
    psi = np.asarray(psi, dtype=complex)
    h_psi = hamiltonian_matrix @ psi
    h2_psi = hamiltonian_matrix @ h_psi

    order1_energy, order1_rank = _project_ground_energy(
        np,
        np.stack([psi, h_psi], axis=0),
        hamiltonian_matrix,
        overlap_tol,
    )
    order2_energy, order2_rank = _project_ground_energy(
        np,
        np.stack([psi, h_psi, h2_psi], axis=0),
        hamiltonian_matrix,
        overlap_tol,
    )
    return order1_energy, order2_energy, order1_rank, order2_rank


def _basis_rank_from_vectors(np, basis_vectors, overlap_tol: float) -> int:
    basis = np.stack([np.asarray(vector, dtype=complex) for vector in basis_vectors], axis=0)
    overlap = basis.conj() @ basis.T
    overlap = (overlap + overlap.conj().T) / 2.0
    s_vals = np.linalg.eigvalsh(overlap).real
    return int(np.count_nonzero(s_vals > float(overlap_tol)))


def _compute_exact_ground_energy_qulacs(h_qulacs, np) -> float:
    sparse_matrix = h_qulacs.get_matrix().tocsr()
    try:
        from scipy.sparse.linalg import eigsh

        value = eigsh(sparse_matrix, k=1, which="SA", return_eigenvectors=False, tol=0.0)[0]
        return float(np.real(value))
    except Exception:
        dense_matrix = np.asarray(sparse_matrix.toarray(), dtype=complex)
        dense_matrix = 0.5 * (dense_matrix + dense_matrix.conj().T)
        return float(np.linalg.eigvalsh(dense_matrix).real[0])


def _qulacs_compatible_kwargs(
    *,
    device_name: Optional[str],
    shots: Optional[int],
    commutator_shots: Optional[int],
    commutator_mode: str,
    commutator_debug: bool,
    device_kwargs: Optional[Mapping[str, Any]],
) -> bool:
    return (
        device_name is None
        and not device_kwargs
        and shots in {None, 0}
        and commutator_shots in {None, 0}
        and str(commutator_mode).strip().lower() == "ansatz"
        and not bool(commutator_debug)
    )


def _run_adapt_krylov_qulacs(
    symbols: Sequence[str],
    geometry,
    *,
    adapt_it: int,
    basis: str,
    charge: int,
    spin: int,
    active_electrons: int,
    active_orbitals: int,
    hamiltonian_cutoff: float,
    hamiltonian_source: str,
    pool_type: str,
    pool_sample_size: Optional[int],
    pool_seed: Optional[int],
    parallel_gradients: bool,
    max_workers: Optional[int],
    gradient_chunk_size: Optional[int],
    optimizer_method: str,
    optimizer_maxiter: int,
    overlap_tol: float,
    parallel_postprocessing: bool,
    postprocess_workers: Optional[int],
    iteration_callback: Optional[Callable[[Mapping[str, Any]], None]],
):
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
    params, ash_excitation, adapt_energies, adapt_history = _adapt_vqe_qulacs_from_payload(
        payload,
        adapt_it=adapt_it,
        pool_type=pool_type,
        pool_sample_size=pool_sample_size,
        pool_seed=pool_seed,
        parallel_gradients=parallel_gradients,
        max_workers=max_workers,
        gradient_chunk_size=gradient_chunk_size,
        optimizer_method=optimizer_method,
        optimizer_maxiter=optimizer_maxiter,
        return_history=True,
        iteration_callback=None,
    )

    np = payload["np"]
    qml = payload["qml"]
    qg = payload["qg"]
    QuantumState = payload["QuantumState"]
    ParametricQuantumCircuit = payload["ParametricQuantumCircuit"]
    qubits = int(payload["qubits"])
    h_qulacs = payload["h_qulacs"]
    hf_bits = [int(bit) for bit in np.asarray(payload["hf_bits"], dtype=int)]
    ansatz_type = _ansatz_type_from_pool_type(_normalize_pool_type(pool_type))

    exact_ground_energy = _compute_exact_ground_energy_qulacs(h_qulacs, np)
    worker_count = _resolve_worker_count(postprocess_workers if postprocess_workers is not None else max_workers)

    def _enrich_snapshot(index: int, snapshot: Mapping[str, Any]):
        compiled = _build_adapt_compiled_circuit(
            snapshot["ash_excitation"],
            hf_bits,
            ansatz_type=ansatz_type,
            qubits=qubits,
            ParametricQuantumCircuit=ParametricQuantumCircuit,
            qml=qml,
            qg=qg,
            np=np,
        )
        state = _evaluate_compiled_state(
            compiled,
            snapshot["params"],
            QuantumState=QuantumState,
            np=np,
        )
        h_state = _apply_qulacs_operator(h_qulacs, state, qubits=qubits, QuantumState=QuantumState)
        h2_state = _apply_qulacs_operator(h_qulacs, h_state, qubits=qubits, QuantumState=QuantumState)

        order1_energies, _order1_basis = _project_hamiltonian_energies(
            [state, h_state],
            h_qulacs,
            overlap_tol,
            np,
            max_workers=1,
        )
        order2_energies, _order2_basis = _project_hamiltonian_energies(
            [state, h_state, h2_state],
            h_qulacs,
            overlap_tol,
            np,
            max_workers=1,
        )

        order1_energy = float(order1_energies[0])
        order2_energy = float(order2_energies[0])
        order1_rank = _basis_rank_from_vectors(
            np,
            [state.get_vector(), h_state.get_vector()],
            overlap_tol,
        )
        order2_rank = _basis_rank_from_vectors(
            np,
            [state.get_vector(), h_state.get_vector(), h2_state.get_vector()],
            overlap_tol,
        )

        enriched_snapshot = dict(snapshot)
        enriched_snapshot.update(
            {
                "adapt_energy": float(snapshot["energy"]),
                "krylov_order1_energy": float(order1_energy),
                "krylov_order2_energy": float(order2_energy),
                "krylov_order1_basis_rank": int(order1_rank),
                "krylov_order2_basis_rank": int(order2_rank),
                "exact_ground_energy": float(exact_ground_energy),
            }
        )
        return int(index), enriched_snapshot

    ordered_history = [None] * len(adapt_history)
    if parallel_postprocessing and worker_count > 1 and len(adapt_history) > 1:
        with ThreadPoolExecutor(max_workers=min(worker_count, len(adapt_history))) as executor:
            futures = [
                executor.submit(_enrich_snapshot, index, snapshot)
                for index, snapshot in enumerate(adapt_history)
            ]
            for future in as_completed(futures):
                index, enriched_snapshot = future.result()
                ordered_history[index] = enriched_snapshot
    else:
        for index, snapshot in enumerate(adapt_history):
            resolved_index, enriched_snapshot = _enrich_snapshot(index, snapshot)
            ordered_history[resolved_index] = enriched_snapshot

    history = []
    krylov_order1_energies = []
    krylov_order2_energies = []
    for enriched_snapshot in ordered_history:
        history.append(enriched_snapshot)
        krylov_order1_energies.append(float(enriched_snapshot["krylov_order1_energy"]))
        krylov_order2_energies.append(float(enriched_snapshot["krylov_order2_energy"]))
        if iteration_callback is not None:
            iteration_callback(enriched_snapshot)

    details = {
        "backend": "qulacs",
        "krylov_order1_energies": np.asarray(krylov_order1_energies, dtype=float),
        "krylov_order2_energies": np.asarray(krylov_order2_energies, dtype=float),
        "exact_ground_energy": float(exact_ground_energy),
        "history": history,
        "parallel_postprocessing": bool(parallel_postprocessing),
        "parallel_postprocessing_workers": int(worker_count),
    }
    return params, ash_excitation, adapt_energies, details


def _run_adapt_krylov_pennylane(
    symbols: Sequence[str],
    geometry,
    *,
    adapt_it: int,
    basis: str,
    charge: int,
    spin: int,
    active_electrons: int,
    active_orbitals: int,
    device_name: Optional[str],
    shots: Optional[int],
    commutator_shots: Optional[int],
    commutator_mode: str,
    commutator_debug: bool,
    hamiltonian_cutoff: float,
    hamiltonian_source: str,
    pool_type: str,
    pool_sample_size: Optional[int],
    pool_seed: Optional[int],
    parallel_gradients: bool,
    parallel_backend: str,
    max_workers: Optional[int],
    gradient_chunk_size: Optional[int],
    optimizer_method: str,
    optimizer_maxiter: int,
    pauli_grouping: bool,
    grouping_type: str,
    device_kwargs: Optional[Mapping[str, Any]],
    overlap_tol: float,
    iteration_callback: Optional[Callable[[Mapping[str, Any]], None]],
):
    params, ash_excitation, adapt_energies, adapt_history = adapt_vqe(
        symbols=symbols,
        geometry=geometry,
        adapt_it=adapt_it,
        basis=basis,
        charge=charge,
        spin=spin,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
        device_name=device_name,
        shots=shots,
        commutator_shots=commutator_shots,
        commutator_mode=commutator_mode,
        commutator_debug=commutator_debug,
        hamiltonian_cutoff=hamiltonian_cutoff,
        hamiltonian_source=hamiltonian_source,
        pool_type=pool_type,
        pool_sample_size=pool_sample_size,
        pool_seed=pool_seed,
        parallel_gradients=parallel_gradients,
        parallel_backend=parallel_backend,
        max_workers=max_workers,
        gradient_chunk_size=gradient_chunk_size,
        optimizer_method=optimizer_method,
        optimizer_maxiter=optimizer_maxiter,
        pauli_grouping=pauli_grouping,
        grouping_type=grouping_type,
        device_kwargs=device_kwargs,
        return_history=True,
        iteration_callback=None,
    )

    qml, np, hamiltonian_matrix, qubits, effective_electrons = _build_hamiltonian_qml(
        symbols=symbols,
        geometry=geometry,
        basis=basis,
        charge=charge,
        spin=spin,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
        hamiltonian_cutoff=hamiltonian_cutoff,
        hamiltonian_source=hamiltonian_source,
    )

    exact_ground_energy = float(np.linalg.eigvalsh(hamiltonian_matrix).real[0])
    hf_state = qml.qchem.hf_state(effective_electrons, qubits)
    ansatz_type = getattr(ash_excitation, "ansatz_type", None)
    if ansatz_type is None:
        ansatz_type = _ansatz_type_from_pool_type(_normalize_pool_type(pool_type))

    dev_state = _make_device(qml, device_name, qubits)

    @qml.qnode(dev_state)
    def _statevector(params_local, excitations_local):
        qml.BasisState(hf_state, wires=range(qubits))
        for i, excitation in enumerate(excitations_local):
            _apply_excitation_gate(qml, excitation, params_local[i], ansatz_type)
        return qml.state()

    krylov_order1_energies = []
    krylov_order2_energies = []
    history = []

    for snapshot in adapt_history:
        state = _statevector(snapshot["params"], snapshot["ash_excitation"])
        order1_energy, order2_energy, order1_rank, order2_rank = _solve_krylov_orders_dense(
            np,
            state,
            hamiltonian_matrix,
            overlap_tol,
        )
        enriched_snapshot = dict(snapshot)
        enriched_snapshot.update(
            {
                "adapt_energy": float(snapshot["energy"]),
                "krylov_order1_energy": float(order1_energy),
                "krylov_order2_energy": float(order2_energy),
                "krylov_order1_basis_rank": int(order1_rank),
                "krylov_order2_basis_rank": int(order2_rank),
                "exact_ground_energy": float(exact_ground_energy),
            }
        )
        history.append(enriched_snapshot)
        krylov_order1_energies.append(float(order1_energy))
        krylov_order2_energies.append(float(order2_energy))
        if iteration_callback is not None:
            iteration_callback(enriched_snapshot)

    details = {
        "backend": "pennylane",
        "krylov_order1_energies": np.asarray(krylov_order1_energies, dtype=float),
        "krylov_order2_energies": np.asarray(krylov_order2_energies, dtype=float),
        "exact_ground_energy": float(exact_ground_energy),
        "history": history,
        "parallel_postprocessing": False,
        "parallel_postprocessing_workers": 1,
    }
    return params, ash_excitation, adapt_energies, details


def adaptKrylov(
    symbols: Sequence[str],
    geometry,
    *,
    adapt_it: int,
    basis: str = "sto-6g",
    charge: int = 0,
    spin: int = 0,
    active_electrons: int,
    active_orbitals: int,
    device_name: Optional[str] = None,
    shots: Optional[int] = None,
    commutator_shots: Optional[int] = None,
    commutator_mode: str = "ansatz",
    commutator_debug: bool = False,
    hamiltonian_cutoff: float = 1e-20,
    hamiltonian_source: str = "casci",
    pool_type: str = "fermionic_sd",
    pool_sample_size: Optional[int] = None,
    pool_seed: Optional[int] = None,
    parallel_gradients: Optional[bool] = None,
    parallel_backend: str = "auto",
    max_workers: Optional[int] = None,
    gradient_chunk_size: Optional[int] = None,
    optimizer_method: str = "BFGS",
    optimizer_maxiter: int = 100_000_000,
    pauli_grouping: bool = False,
    grouping_type: str = "qwc",
    device_kwargs: Optional[Mapping[str, Any]] = None,
    overlap_tol: float = 1e-10,
    backend: str = "auto",
    parallel_postprocessing: bool = True,
    postprocess_workers: Optional[int] = None,
    iteration_callback: Optional[Callable[[Mapping[str, Any]], None]] = None,
):
    """Run ADAPT-VQE and report exact order-1/order-2 Krylov energies per iteration.

    The Krylov subspaces are built from the optimized ADAPT state at each
    iteration using ``|psi>``, ``H|psi>``, and ``H^2|psi>``.

    ``backend="auto"`` prefers the Qulacs ADAPT/state backend when the request
    is fully analytic and does not pin a PennyLane device. Otherwise it falls
    back to the PennyLane reference implementation.
    """
    backend_normalized = str(backend).strip().lower()
    if backend_normalized not in {"auto", "qulacs", "pennylane"}:
        raise ValueError("backend must be one of {'auto', 'qulacs', 'pennylane'}")

    qulacs_compatible = _qulacs_compatible_kwargs(
        device_name=device_name,
        shots=shots,
        commutator_shots=commutator_shots,
        commutator_mode=commutator_mode,
        commutator_debug=commutator_debug,
        device_kwargs=device_kwargs,
    )

    if parallel_gradients is None:
        parallel_gradients_resolved = backend_normalized in {"auto", "qulacs"}
    else:
        parallel_gradients_resolved = bool(parallel_gradients)

    if backend_normalized == "qulacs" and not qulacs_compatible:
        raise ValueError(
            "backend='qulacs' currently supports only analytic ADAPT runs with "
            "device_name=None, shots=0, commutator_mode='ansatz', and no device_kwargs."
        )

    if backend_normalized in {"auto", "qulacs"} and qulacs_compatible:
        try:
            return _run_adapt_krylov_qulacs(
                symbols,
                geometry,
                adapt_it=adapt_it,
                basis=basis,
                charge=charge,
                spin=spin,
                active_electrons=active_electrons,
                active_orbitals=active_orbitals,
                hamiltonian_cutoff=hamiltonian_cutoff,
                hamiltonian_source=hamiltonian_source,
                pool_type=pool_type,
                pool_sample_size=pool_sample_size,
                pool_seed=pool_seed,
                parallel_gradients=parallel_gradients_resolved,
                max_workers=max_workers,
                gradient_chunk_size=gradient_chunk_size,
                optimizer_method=optimizer_method,
                optimizer_maxiter=optimizer_maxiter,
                overlap_tol=overlap_tol,
                parallel_postprocessing=parallel_postprocessing,
                postprocess_workers=postprocess_workers,
                iteration_callback=iteration_callback,
            )
        except ImportError:
            if backend_normalized == "qulacs":
                raise

    return _run_adapt_krylov_pennylane(
        symbols,
        geometry,
        adapt_it=adapt_it,
        basis=basis,
        charge=charge,
        spin=spin,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
        device_name=device_name,
        shots=shots,
        commutator_shots=commutator_shots,
        commutator_mode=commutator_mode,
        commutator_debug=commutator_debug,
        hamiltonian_cutoff=hamiltonian_cutoff,
        hamiltonian_source=hamiltonian_source,
        pool_type=pool_type,
        pool_sample_size=pool_sample_size,
        pool_seed=pool_seed,
        parallel_gradients=parallel_gradients_resolved,
        parallel_backend=parallel_backend,
        max_workers=max_workers,
        gradient_chunk_size=gradient_chunk_size,
        optimizer_method=optimizer_method,
        optimizer_maxiter=optimizer_maxiter,
        pauli_grouping=pauli_grouping,
        grouping_type=grouping_type,
        device_kwargs=device_kwargs,
        overlap_tol=overlap_tol,
        iteration_callback=iteration_callback,
    )


adapt_krylov = adaptKrylov
