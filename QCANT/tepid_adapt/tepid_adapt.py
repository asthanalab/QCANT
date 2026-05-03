"""Ancilla-free TEPID-ADAPT for truncated finite-temperature chemistry states.

This implementation follows the truncated Gibbs-state idea of the TEPID paper
while fitting the existing QCANT algorithm style:

1. Build a molecular active-space Hamiltonian.
2. Use the Hartree-Fock reference plus single/double excitations as the
   truncated computational basis by default.
3. Optimize an ADAPT-style ansatz that lowers the truncated free energy
   at a user-selected temperature.
4. Return the ansatz in the same ``(params, ash_excitation, history)`` shape
   already used elsewhere in QCANT so it can be replayed by :func:`QCANT.qscEOM`.

The implementation here uses the ancilla-free ensemble-average formulation
described in Appendix C of the paper and evaluates the thermal weights from the
current transformed basis energies at each optimization step.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Sequence

from .._accelerator import resolve_array_module, to_host_array


_FERMIONIC_POOL_ALIASES = {"fermionic_sd", "sd", "fermionic"}
_QUBIT_POOL_ALIASES = {"qubit_excitation", "qe", "qubit"}


class _ExcitationList(list):
    """List-like excitation container that preserves ansatz metadata."""

    def __init__(
        self,
        *args,
        ansatz_type: str = "fermionic",
        pool_type: str = "fermionic_sd",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.ansatz_type = str(ansatz_type)
        self.pool_type = str(pool_type)


def _normalize_pool_type(pool_type: str) -> str:
    """Normalize user pool-type aliases into a canonical value."""
    normalized = str(pool_type).strip().lower()
    if normalized in _FERMIONIC_POOL_ALIASES:
        return "fermionic_sd"
    if normalized in _QUBIT_POOL_ALIASES:
        return "qubit_excitation"
    raise ValueError(
        "pool_type must be one of {'fermionic_sd', 'sd', 'fermionic', "
        "'qubit_excitation', 'qe', 'qubit'}"
    )


def _ansatz_type_from_pool_type(pool_type: str) -> str:
    """Resolve the ansatz operator family implied by the ADAPT pool."""
    if pool_type == "fermionic_sd":
        return "fermionic"
    if pool_type == "qubit_excitation":
        return "qubit_excitation"
    raise ValueError(f"Unsupported canonical pool_type: {pool_type}")


def _validate_basis_occupations(
    basis_occupations,
    *,
    qubits: Optional[int],
    active_electrons: int,
):
    """Validate and normalize user-provided occupation configurations."""
    if basis_occupations is None:
        return None

    normalized = []
    for occ in basis_occupations:
        try:
            occ_list = sorted(int(idx) for idx in occ)
        except Exception as exc:
            raise ValueError("basis_occupations must be an iterable of occupied-index iterables") from exc

        if len(occ_list) != int(active_electrons):
            raise ValueError("each basis occupation must contain active_electrons occupied indices")
        if len(set(occ_list)) != len(occ_list):
            raise ValueError("basis occupations cannot contain repeated occupied indices")
        if qubits is not None and any(idx < 0 or idx >= int(qubits) for idx in occ_list):
            raise ValueError("basis occupation indices must lie in [0, qubits)")
        normalized.append(occ_list)

    return normalized


def _validate_inputs(
    symbols: Sequence[str],
    geometry,
    adapt_it: int,
    active_electrons: int,
    active_orbitals: int,
    spin: int,
    shots: int,
    gradient_eps: float,
    gradient_tol: Optional[float],
    pool_type: str,
    pool_sample_size: Optional[int],
    temperature: Optional[float],
    beta: Optional[float],
    basis_occupations,
) -> int:
    """Validate user inputs and return the number of atoms."""
    if len(symbols) == 0:
        raise ValueError("symbols must be non-empty")
    if adapt_it < 0:
        raise ValueError("adapt_it must be >= 0")
    if active_electrons <= 0:
        raise ValueError("active_electrons must be > 0")
    if active_orbitals <= 0:
        raise ValueError("active_orbitals must be > 0")
    if active_electrons > (2 * active_orbitals):
        raise ValueError("active_electrons cannot exceed 2 * active_orbitals")
    if spin < 0:
        raise ValueError("spin must be >= 0")
    if shots != 0:
        raise ValueError("tepid_adapt currently supports only analytic mode with shots=0")
    if gradient_eps <= 0:
        raise ValueError("gradient_eps must be > 0")
    if gradient_tol is not None and gradient_tol < 0:
        raise ValueError("gradient_tol must be >= 0")
    _normalize_pool_type(pool_type)
    if pool_sample_size is not None and pool_sample_size <= 0:
        raise ValueError("pool_sample_size must be > 0")
    if (temperature is None) == (beta is None):
        raise ValueError("Provide exactly one of temperature or beta")
    if temperature is not None and temperature <= 0:
        raise ValueError("temperature must be > 0")
    if beta is not None and beta <= 0:
        raise ValueError("beta must be > 0")

    try:
        n_atoms = len(symbols)
        if len(geometry) != n_atoms:
            raise ValueError
    except Exception as exc:
        raise ValueError("geometry must have the same length as symbols") from exc

    _validate_basis_occupations(
        basis_occupations,
        qubits=None,
        active_electrons=int(active_electrons),
    )

    return n_atoms


def tepid_boltzmann_weights(
    energies,
    *,
    temperature: Optional[float] = None,
    beta: Optional[float] = None,
):
    """Return normalized Boltzmann weights from a list of energies.

    Parameters
    ----------
    energies
        Iterable of energies.
    temperature
        Positive temperature in the same units as the energies with ``k_B = 1``.
    beta
        Positive inverse temperature. Exactly one of ``temperature`` or ``beta``
        must be supplied.

    Returns
    -------
    numpy.ndarray
        Normalized Boltzmann weights.
    """
    if (temperature is None) == (beta is None):
        raise ValueError("Provide exactly one of temperature or beta")

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise ImportError("tepid_boltzmann_weights requires numpy.") from exc

    beta_value = float(beta) if beta is not None else 1.0 / float(temperature)
    if beta_value <= 0:
        raise ValueError("beta must be > 0")

    energy_values = np.asarray(energies, dtype=float)
    if energy_values.ndim != 1:
        raise ValueError("energies must be one-dimensional")
    if energy_values.size == 0:
        raise ValueError("energies must be non-empty")

    shifted = -beta_value * energy_values
    shift = float(np.max(shifted))
    unnormalized = np.exp(shifted - shift)
    partition = float(np.sum(unnormalized))
    if partition <= 0:
        raise ValueError("failed to build a positive partition function from the energies")
    return unnormalized / partition


def _basis_index_from_occupations(occ, qubits: int) -> int:
    """Convert an occupied-orbital list into a computational-basis index."""
    bits = [0] * int(qubits)
    for idx in occ:
        bits[int(idx)] = 1
    return int("".join(str(int(bit)) for bit in bits), 2)


def _build_basis_columns(occupations, *, qubits: int, np):
    """Build a column-stacked basis matrix for the chosen occupations."""
    dim = 2 ** int(qubits)
    basis = np.zeros((dim, len(occupations)), dtype=complex)
    for col, occ in enumerate(occupations):
        basis[_basis_index_from_occupations(occ, qubits), col] = 1.0
    return basis


def _apply_local_operator_batch(states, op_matrix, target_wires, *, qubits: int, np):
    """Apply a local operator matrix to one or more statevectors."""
    arr = np.asarray(states, dtype=complex)
    squeeze = False
    if arr.ndim == 1:
        arr = arr[:, None]
        squeeze = True

    batch = int(arr.shape[1])
    targets = tuple(int(wire) for wire in target_wires)
    remaining = tuple(wire for wire in range(int(qubits)) if wire not in targets)

    state_tensor = arr.reshape((2,) * int(qubits) + (batch,))
    permutation = targets + remaining + (int(qubits),)
    inverse_permutation = tuple(sorted(range(len(permutation)), key=permutation.__getitem__))

    permuted = np.transpose(state_tensor, permutation)
    flat_state = permuted.reshape(2 ** len(targets), -1)
    updated = np.asarray(op_matrix, dtype=complex) @ flat_state
    restored = updated.reshape((2,) * int(qubits) + (batch,))
    out = np.transpose(restored, inverse_permutation).reshape(2 ** int(qubits), batch)

    if squeeze:
        return out[:, 0]
    return out


def _build_excitation_operation(qml, excitation, weight, ansatz_type: str):
    """Build a PennyLane excitation operation and return it with its wires."""
    if ansatz_type == "fermionic":
        if len(excitation) == 2:
            op = qml.FermionicSingleExcitation(
                weight=float(weight),
                wires=list(range(int(excitation[0]), int(excitation[1]) + 1)),
            )
            return op, tuple(int(wire) for wire in op.wires)
        if len(excitation) == 4:
            op = qml.FermionicDoubleExcitation(
                weight=float(weight),
                wires1=list(range(int(excitation[0]), int(excitation[1]) + 1)),
                wires2=list(range(int(excitation[2]), int(excitation[3]) + 1)),
            )
            return op, tuple(int(wire) for wire in op.wires)
    elif ansatz_type == "qubit_excitation":
        if len(excitation) == 2:
            op = qml.SingleExcitation(
                float(weight),
                wires=[int(excitation[0]), int(excitation[1])],
            )
            return op, tuple(int(wire) for wire in op.wires)
        if len(excitation) == 4:
            op = qml.DoubleExcitation(
                float(weight),
                wires=[
                    int(excitation[0]),
                    int(excitation[1]),
                    int(excitation[2]),
                    int(excitation[3]),
                ],
            )
            return op, tuple(int(wire) for wire in op.wires)

    raise ValueError(
        "Each excitation must have length 2 (single) or 4 (double); "
        f"received {excitation!r} for ansatz_type='{ansatz_type}'."
    )


def _build_operator_pool(qml, active_electrons: int, qubits: int, pool_type: str):
    """Build the excitation pool labels for TEPID-ADAPT."""
    singles, doubles = qml.qchem.excitations(int(active_electrons), int(qubits))
    pool_excitations = [list(map(int, excitation)) for excitation in singles]
    pool_excitations.extend(list(map(int, excitation)) for excitation in doubles)
    return pool_excitations


def _build_derivative_metadata(
    qml,
    pool_excitations,
    *,
    ansatz_type: str,
    gradient_eps: float,
    np,
):
    """Pre-compute local derivative matrices for candidate excitation gates."""
    metadata = []
    eps = float(gradient_eps)
    for excitation in pool_excitations:
        op_plus, target_wires = _build_excitation_operation(
            qml,
            excitation,
            +eps,
            ansatz_type,
        )
        op_minus, _ = _build_excitation_operation(
            qml,
            excitation,
            -eps,
            ansatz_type,
        )
        derivative = (
            np.asarray(qml.matrix(op_plus), dtype=complex)
            - np.asarray(qml.matrix(op_minus), dtype=complex)
        ) / (2.0 * eps)
        metadata.append(
            {
                "excitation": [int(v) for v in excitation],
                "target_wires": tuple(int(v) for v in target_wires),
                "derivative": derivative,
            }
        )
    return metadata


def _apply_ansatz_batch(
    states,
    params,
    ash_excitation,
    *,
    ansatz_type: str,
    qubits: int,
    qml,
    np,
):
    """Apply the current ansatz to a batch of basis states."""
    current = np.asarray(states, dtype=complex)
    for weight, excitation in zip(params, ash_excitation):
        op, target_wires = _build_excitation_operation(
            qml,
            excitation,
            weight,
            ansatz_type,
        )
        current = _apply_local_operator_batch(
            current,
            qml.matrix(op),
            target_wires,
            qubits=qubits,
            np=np,
        )
    return current


def _thermal_summary(energies, *, beta: float, np):
    """Compute Boltzmann weights, entropy, and free energy."""
    energy_values = np.asarray(energies, dtype=float)
    shifted = -float(beta) * energy_values
    shift = float(np.max(shifted))
    unnormalized = np.exp(shifted - shift)
    partition = float(np.sum(unnormalized))
    weights = unnormalized / partition
    safe_weights = np.clip(weights, 1e-300, None)
    entropy = float(-np.sum(weights * np.log(safe_weights)))
    log_partition = float(shift + np.log(partition))
    free_energy = float(-log_partition / float(beta))
    return weights, entropy, free_energy, log_partition


def tepid_adapt(
    symbols: Sequence[str],
    geometry,
    *,
    adapt_it: int,
    temperature: Optional[float] = None,
    beta: Optional[float] = None,
    basis: str = "sto-3g",
    charge: int = 0,
    spin: int = 0,
    active_electrons: int,
    active_orbitals: int,
    shots: int = 0,
    hamiltonian_cutoff: float = 1e-20,
    hamiltonian_source: str = "molecular",
    pool_type: str = "fermionic_sd",
    include_identity: bool = True,
    basis_occupations=None,
    pool_sample_size: Optional[int] = None,
    pool_seed: Optional[int] = None,
    array_backend: str = "auto",
    gradient_eps: float = 1e-7,
    gradient_tol: Optional[float] = None,
    optimizer_method: str = "BFGS",
    optimizer_maxiter: int = 200,
    return_details: bool = False,
    iteration_callback: Optional[Callable[[Mapping[str, Any]], None]] = None,
):
    """Run the ancilla-free TEPID-ADAPT workflow on a molecule.

    Parameters
    ----------
    symbols
        Atomic symbols, e.g. ``["H", "H"]``.
    geometry
        Nuclear coordinates, array-like with shape ``(n_atoms, 3)`` in Angstrom.
    adapt_it
        Maximum number of TEPID-ADAPT iterations.
    temperature
        Positive temperature in the same energy units as the Hamiltonian with
        ``k_B = 1``. Exactly one of ``temperature`` or ``beta`` must be given.
    beta
        Positive inverse temperature. Exactly one of ``temperature`` or
        ``beta`` must be given.
    basis
        Basis set name understood by PySCF / PennyLane.
    charge
        Total molecular charge.
    spin
        Spin parameter used by PySCF as ``2S``.
    active_electrons
        Number of active electrons in the chosen active space.
    active_orbitals
        Number of active spatial orbitals in the active space.
    shots
        Analytic-only placeholder for API symmetry. Must be ``0``.
    hamiltonian_cutoff
        Drop Hamiltonian terms below this absolute coefficient threshold when
        building the CASCI effective Hamiltonian.
    hamiltonian_source
        ``"molecular"`` (default) uses
        :func:`qml.qchem.molecular_hamiltonian`; ``"casci"`` uses the
        CASCI-effective active-space Hamiltonian.
    pool_type
        Operator-pool family used by the adaptive ansatz:
        ``"fermionic_sd"``/``"sd"``/``"fermionic"`` or
        ``"qubit_excitation"``/``"qe"``/``"qubit"``.
    include_identity
        If True (default), prepend the Hartree-Fock reference to the truncated
        computational basis before the single/double excitations.
    basis_occupations
        Optional custom occupation configurations. When omitted, the truncated
        basis defaults to HF + singles + doubles, matching qscEOM's basis
        construction.
    pool_sample_size
        If provided, randomly sample this many operators from the pool per
        adaptive iteration.
    pool_seed
        Seed for the operator-pool sampler.
    array_backend
        Dense linear algebra backend. ``"numpy"`` keeps CPU execution,
        ``"cupy"`` requests GPU dense linear algebra, and ``"auto"`` keeps
        CPU execution unless a GPU backend is explicitly requested.
    gradient_eps
        Finite-difference step used to build candidate gate derivatives at zero.
    gradient_tol
        Optional stopping threshold for the absolute free-energy gradient score.
    optimizer_method
        SciPy optimization method for the variational parameters.
    optimizer_maxiter
        Maximum SciPy iterations for the ansatz re-optimization after each
        adaptive selection.
    return_details
        If True, append a details dictionary containing the final truncated
        energies, thermal weights, basis metadata, and full iteration history.
    iteration_callback
        Optional callback invoked after each adaptive iteration with a snapshot
        dictionary.

    Returns
    -------
    tuple
        ``(params, ash_excitation, free_energies)``.
        When ``return_details=True``, a fourth item is appended:
        ``details``.

    Notes
    -----
    This implementation uses the ancilla-free ensemble objective from the TEPID
    paper's Appendix C. The thermal weights are updated analytically from the
    current transformed-basis energies at each objective evaluation.
    """
    # pylint: disable=too-many-locals,too-many-statements,too-many-branches
    n_atoms = _validate_inputs(
        symbols,
        geometry,
        adapt_it,
        active_electrons,
        active_orbitals,
        spin,
        shots,
        gradient_eps,
        gradient_tol,
        pool_type,
        pool_sample_size,
        temperature,
        beta,
        basis_occupations,
    )

    try:
        import warnings

        import numpy as np
        import pennylane as qml
        import pyscf
        from pyscf import gto, mcscf, scf
        from scipy.optimize import minimize

        from QCANT.qsceom.excitations import inite

        warnings.filterwarnings("ignore")
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "tepid_adapt requires dependencies. Install at least: "
            "`pip install numpy scipy pennylane pyscf`."
        ) from exc

    xp, array_backend_name, using_gpu = resolve_array_module(
        array_backend=array_backend,
        device_name=None,
        allow_gpu=True,
        context="tepid_adapt",
    )

    beta_value = float(beta) if beta is not None else 1.0 / float(temperature)
    temperature_value = 1.0 / beta_value

    pool_type_canonical = _normalize_pool_type(pool_type)
    ansatz_type = _ansatz_type_from_pool_type(pool_type_canonical)

    atom = [(symbols[i], tuple(float(x) for x in geometry[i])) for i in range(n_atoms)]
    hamiltonian_source = str(hamiltonian_source).strip().lower()

    if hamiltonian_source == "casci":
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

        mycas_ref = mcscf.CASCI(mf_ref, int(active_orbitals), int(active_electrons))
        h1ecas, ecore = mycas_ref.get_h1eff(mf_ref.mo_coeff)
        h2ecas = mycas_ref.get_h2eff(mf_ref.mo_coeff)

        mycas_ref.kernel()

        ncas = int(mycas_ref.ncas)
        two_mo = pyscf.ao2mo.restore("1", h2ecas, norb=ncas)
        two_mo = np.swapaxes(two_mo, 1, 3)
        one_mo = h1ecas
        core_constant = np.array([ecore])

        H_fermionic = qml.qchem.fermionic_observable(
            core_constant,
            one_mo,
            two_mo,
            cutoff=hamiltonian_cutoff,
        )
        H = qml.jordan_wigner(H_fermionic)
        qubits = 2 * ncas
        active_electrons = int(sum(mycas_ref.nelecas))
    elif hamiltonian_source == "molecular":
        H, qubits = qml.qchem.molecular_hamiltonian(
            symbols,
            geometry,
            basis=basis,
            method="pyscf",
            active_electrons=int(active_electrons),
            active_orbitals=int(active_orbitals),
            charge=int(charge),
        )
        active_electrons = int(active_electrons)
    else:
        raise ValueError("hamiltonian_source must be one of {'casci', 'molecular'}")

    basis_occupations = _validate_basis_occupations(
        basis_occupations,
        qubits=qubits,
        active_electrons=int(active_electrons),
    )
    hf_state = qml.qchem.hf_state(active_electrons, qubits)
    hf_occ = [int(idx) for idx, bit in enumerate(np.asarray(hf_state, dtype=int)) if int(bit) == 1]

    if basis_occupations is None:
        sd_occupations = [list(int(v) for v in occ) for occ in inite(active_electrons, qubits)]
    else:
        sd_occupations = [list(int(v) for v in occ) for occ in basis_occupations]

    if include_identity:
        basis_occ = [hf_occ] + [occ for occ in sd_occupations if occ != hf_occ]
    else:
        basis_occ = [occ for occ in sd_occupations if occ != hf_occ]

    seen = set()
    unique_basis_occ = []
    for occ in basis_occ:
        key = tuple(int(v) for v in occ)
        if key in seen:
            continue
        seen.add(key)
        unique_basis_occ.append(list(key))
    basis_occ = unique_basis_occ

    if len(basis_occ) == 0:
        raise ValueError("truncated TEPID basis is empty after include_identity filtering.")

    H_dense = xp.asarray(qml.matrix(H, wire_order=range(qubits)), dtype=complex) if using_gpu else None
    H_sparse = None if using_gpu else H.sparse_matrix(wire_order=range(qubits), format="csr")
    basis_columns = _build_basis_columns(basis_occ, qubits=qubits, np=xp)

    pool_excitations = _build_operator_pool(
        qml,
        active_electrons,
        qubits,
        pool_type_canonical,
    )
    derivative_metadata = _build_derivative_metadata(
        qml,
        pool_excitations,
        ansatz_type=ansatz_type,
        gradient_eps=float(gradient_eps),
        np=xp,
    )

    free_energies = []
    history = []
    ash_excitation = _ExcitationList(ansatz_type=ansatz_type, pool_type=pool_type_canonical)
    params = np.zeros(0, dtype=float)
    rng = np.random.default_rng(pool_seed)

    def _evaluate_snapshot(params_eval, excitations_eval):
        states = _apply_ansatz_batch(
            basis_columns,
            params_eval,
            excitations_eval,
            ansatz_type=ansatz_type,
            qubits=qubits,
            qml=qml,
            np=xp,
        )
        h_states = (H_dense @ states) if using_gpu else (H_sparse @ states)
        basis_energies = xp.real(xp.sum(xp.conj(states) * h_states, axis=0))
        weights, entropy, free_energy, log_partition = _thermal_summary(
            basis_energies,
            beta=beta_value,
            np=xp,
        )
        return {
            "states": xp.asarray(states, dtype=complex),
            "basis_energies": xp.asarray(basis_energies, dtype=float),
            "thermal_weights": xp.asarray(weights, dtype=float),
            "entropy": float(entropy),
            "free_energy": float(free_energy),
            "log_partition": float(log_partition),
        }

    def _objective(params_eval):
        snapshot = _evaluate_snapshot(params_eval, ash_excitation)
        return float(snapshot["free_energy"])

    initial_snapshot = _evaluate_snapshot(params, ash_excitation)
    current_snapshot = initial_snapshot

    for iteration in range(int(adapt_it)):
        if pool_sample_size is None or pool_sample_size >= len(pool_excitations):
            candidate_indices = list(range(len(pool_excitations)))
        else:
            candidate_indices = [
                int(idx)
                for idx in rng.choice(len(pool_excitations), size=int(pool_sample_size), replace=False)
            ]

        max_score = float("-inf")
        max_gradient = 0.0
        max_operator_index = None
        max_operator_excitation = None

        for idx in candidate_indices:
            meta = derivative_metadata[int(idx)]
            d_states = _apply_local_operator_batch(
                current_snapshot["states"],
                meta["derivative"],
                meta["target_wires"],
                qubits=qubits,
                np=xp,
            )
            hd_states = (H_dense @ d_states) if using_gpu else (H_sparse @ d_states)
            deriv_energies = 2.0 * xp.real(
                xp.sum(xp.conj(current_snapshot["states"]) * hd_states, axis=0)
            )
            weighted_gradient = float(
                to_host_array(
                    xp.dot(current_snapshot["thermal_weights"], xp.asarray(deriv_energies, dtype=float))
                ).item()
            )
            score = abs(weighted_gradient)
            if score > max_score:
                max_score = float(score)
                max_gradient = float(weighted_gradient)
                max_operator_index = int(idx)
                max_operator_excitation = [int(v) for v in meta["excitation"]]

        if max_operator_excitation is None or max_operator_index is None:
            raise RuntimeError("No operator selected during TEPID-ADAPT iteration.")

        if gradient_tol is not None and max_score < float(gradient_tol):
            break

        ash_excitation.append(max_operator_excitation)
        params = np.append(np.asarray(params, dtype=float), 0.0)

        result = minimize(
            _objective,
            params,
            method=optimizer_method,
            tol=1e-12,
            options={"disp": False, "maxiter": int(optimizer_maxiter)},
        )

        params = np.asarray(result.x, dtype=float)
        current_snapshot = _evaluate_snapshot(params, ash_excitation)
        free_energies.append(float(current_snapshot["free_energy"]))

        basis_energies_host = to_host_array(current_snapshot["basis_energies"], dtype=float)
        thermal_weights_host = to_host_array(current_snapshot["thermal_weights"], dtype=float)
        energy_order = np.argsort(basis_energies_host, kind="stable")
        iteration_snapshot = {
            "iteration": int(iteration + 1),
            "selected_pool_index": int(max_operator_index),
            "selected_excitation": [int(v) for v in max_operator_excitation],
            "tepid_gradient": float(max_gradient),
            "tepid_gradient_abs": float(max_score),
            "free_energy": float(current_snapshot["free_energy"]),
            "entropy": float(current_snapshot["entropy"]),
            "log_partition": float(current_snapshot["log_partition"]),
            "basis_energies": basis_energies_host.copy(),
            "thermal_weights": thermal_weights_host.copy(),
            "sorted_basis_energies": basis_energies_host[energy_order].copy(),
            "sorted_thermal_weights": thermal_weights_host[energy_order].copy(),
            "params": np.asarray(params, dtype=float).copy(),
            "ash_excitation": [[int(v) for v in excitation] for excitation in ash_excitation],
            "optimizer_success": bool(result.success),
            "optimizer_status": int(result.status),
            "optimizer_message": str(result.message),
            "optimizer_nit": getattr(result, "nit", None),
            "optimizer_nfev": getattr(result, "nfev", None),
        }
        history.append(iteration_snapshot)

        if iteration_callback is not None:
            iteration_callback(iteration_snapshot)

    initial_basis_energies = to_host_array(initial_snapshot["basis_energies"], dtype=float)
    initial_thermal_weights = to_host_array(initial_snapshot["thermal_weights"], dtype=float)
    final_basis_energies = to_host_array(current_snapshot["basis_energies"], dtype=float)
    final_thermal_weights = to_host_array(current_snapshot["thermal_weights"], dtype=float)
    final_order = np.argsort(final_basis_energies, kind="stable")
    details = {
        "beta": float(beta_value),
        "temperature": float(temperature_value),
        "array_backend": str(array_backend_name),
        "basis_occupations": [[int(v) for v in occ] for occ in basis_occ],
        "hf_occupation": [int(v) for v in hf_occ],
        "include_identity": bool(include_identity),
        "pool_type": pool_type_canonical,
        "ansatz_type": ansatz_type,
        "initial_free_energy": float(initial_snapshot["free_energy"]),
        "initial_entropy": float(initial_snapshot["entropy"]),
        "initial_basis_energies": initial_basis_energies.copy(),
        "initial_thermal_weights": initial_thermal_weights.copy(),
        "final_free_energy": float(current_snapshot["free_energy"]),
        "final_entropy": float(current_snapshot["entropy"]),
        "final_log_partition": float(current_snapshot["log_partition"]),
        "final_basis_energies": final_basis_energies.copy(),
        "final_thermal_weights": final_thermal_weights.copy(),
        "sorted_basis_energies": final_basis_energies[final_order].copy(),
        "sorted_basis_occupations": [basis_occ[int(idx)] for idx in final_order],
        "sorted_thermal_weights": final_thermal_weights[final_order].copy(),
        "history": history,
    }

    if return_details:
        return params, ash_excitation, free_energies, details
    return params, ash_excitation, free_energies
