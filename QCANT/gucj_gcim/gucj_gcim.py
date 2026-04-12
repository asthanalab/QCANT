"""Readable exact g-uCJ-GCIM solver for small active spaces.

The default path is intentionally simple:

1. Build the active-space Hamiltonian and the Hartree-Fock reference.
2. Build a fixed generalized same-spin one-body ``K`` operator.
3. Rotate the reference and Hamiltonian with ``exp(K)``.
4. Select one GCIM projector at a time with an ADAPT-like commutator score.
5. After each selection, rebuild the rank-0/1/2 projector-product basis.
6. Solve the generalized eigenvalue problem in that nonorthogonal basis.
7. Record the new ground-state Ritz root and diagnostics.

Advanced modes are still supported, but they live behind separate helpers so
the default frozen-``K`` algorithm can be read end-to-end without tracing every
compatibility branch. Algorithms 2 and 3 are kept only as legacy research
controls.
"""

from __future__ import annotations

from contextlib import nullcontext
import math
from dataclasses import dataclass
from itertools import combinations
from typing import Optional, Sequence


_SUPPORTED_METHOD_VARIANTS = {
    "frozen_k",
    "optimized_k_outer_loop",
    "alternating_k_and_subspace",
}
_LEGACY_METHOD_VARIANTS = {
    "optimized_k_outer_loop",
    "alternating_k_and_subspace",
}
_SUPPORTED_K_PARAM_SHAPES = {
    "ov_same_spin_vector",
    "generalized_same_spin_vector",
    "uniform_ov_same_spin",
}
_SUPPORTED_K_SOURCES = {
    "ccsd_t1",
    "manual",
    "heuristic",
}
_SUPPORTED_SELECTION_MODES = {
    "adaptive_commutator",
    "graded_lex",
}
_K_PARAM_SHAPE_ALIASES = {
    "vector_ov_same_spin": "ov_same_spin_vector",
    "full_ov_same_spin": "ov_same_spin_vector",
    "generalized": "generalized_same_spin_vector",
    "generalized_spin_preserving": "generalized_same_spin_vector",
    "spin_preserving_generalized": "generalized_same_spin_vector",
    "generalized_spin_preserving_vector": "generalized_same_spin_vector",
}
_K_SOURCE_ALIASES = {
    "ccsd": "ccsd_t1",
    "t1": "ccsd_t1",
    "ccsd-singles": "ccsd_t1",
}


@dataclass(frozen=True)
class _SystemData:
    """Cached molecular-system data for the exact g-uCJ-GRIM/GCIM solver."""

    h_matrix: object
    hf_state: object
    fci_energy: float
    hf_energy: float
    qubits: int
    projector_pairs: tuple[tuple[int, int], ...]
    projector_pair_masks: object
    subset_labels: tuple[tuple[tuple[int, int], ...], ...]
    subset_masks: object
    k_generator_matrices: object
    k_generator_pairs: tuple[tuple[int, int], ...]
    k_parameter_labels: tuple[str, ...]
    k_param_shape: str
    ccsd_t1_k_params: object


def _normalize_method_variant(method_variant: str) -> str:
    normalized = str(method_variant).strip().lower()
    if normalized not in _SUPPORTED_METHOD_VARIANTS:
        raise ValueError(
            "method_variant must be one of "
            "{'frozen_K', 'optimized_K_outer_loop', 'alternating_K_and_subspace'}"
        )
    return normalized


def _normalize_k_param_shape(k_param_shape: str) -> str:
    normalized = str(k_param_shape).strip().lower()
    normalized = _K_PARAM_SHAPE_ALIASES.get(normalized, normalized)
    if normalized not in _SUPPORTED_K_PARAM_SHAPES:
        raise ValueError(
            "k_param_shape must be one of "
            "{'ov_same_spin_vector', 'generalized_same_spin_vector', 'uniform_ov_same_spin'}"
        )
    return normalized


def _normalize_k_source(k_source: str) -> str:
    normalized = str(k_source).strip().lower()
    normalized = _K_SOURCE_ALIASES.get(normalized, normalized)
    if normalized not in _SUPPORTED_K_SOURCES:
        raise ValueError("k_source must be one of {'ccsd_t1', 'manual', 'heuristic'}")
    return normalized


def _normalize_selection_mode(selection_mode: str) -> str:
    normalized = str(selection_mode).strip().lower()
    if normalized not in _SUPPORTED_SELECTION_MODES:
        raise ValueError("selection_mode must be one of {'adaptive_commutator', 'graded_lex'}")
    return normalized


def _default_k_param_shape_for_method(method_variant: str) -> str:
    """Return the method-specific default K parameterization.

    The production default is intentionally local to ``frozen_K`` so a reader
    can immediately see that algorithm 1 uses the generalized same-spin fixed-K
    path, while the legacy algorithms 2 and 3 keep the older occupied-virtual
    control path.
    """
    if str(method_variant).strip().lower() == "frozen_k":
        return "generalized_same_spin_vector"
    return "ov_same_spin_vector"


def _is_legacy_method_variant(method_variant: str) -> bool:
    """Return True for the nondefault legacy K-optimization branches."""
    return str(method_variant).strip().lower() in _LEGACY_METHOD_VARIANTS


def _should_compute_ccsd_t1(
    *,
    method_variant: str,
    k_params,
    k_source: str,
    k_param_shape: str,
) -> bool:
    """Return True only for the explicit frozen-K CCSD control path."""
    return (
        str(method_variant).strip().lower() == "frozen_k"
        and k_params is None
        and str(k_source).strip().lower() == "ccsd_t1"
        and str(k_param_shape).strip().lower() == "ov_same_spin_vector"
    )


def _validate_inputs(
    *,
    symbols: Sequence[str],
    geometry,
    active_electrons: int,
    active_orbitals: int,
    subset_rank_max: int,
    overlap_tol: float,
    regularization: float,
    outer_opt_maxiter: int,
    alternating_opt_maxiter: int,
) -> int:
    """Validate core user inputs."""
    if len(symbols) == 0:
        raise ValueError("symbols must be non-empty")
    try:
        n_atoms = len(symbols)
        if len(geometry) != n_atoms:
            raise ValueError
    except Exception as exc:
        raise ValueError("geometry must have the same length as symbols") from exc
    if active_electrons <= 0:
        raise ValueError("active_electrons must be > 0")
    if active_orbitals <= 0:
        raise ValueError("active_orbitals must be > 0")
    if subset_rank_max < 0 or subset_rank_max > 2:
        raise ValueError("subset_rank_max must be between 0 and 2 for this v1 implementation")
    if overlap_tol <= 0:
        raise ValueError("overlap_tol must be > 0")
    if regularization < 0:
        raise ValueError("regularization must be >= 0")
    if outer_opt_maxiter <= 0:
        raise ValueError("outer_opt_maxiter must be > 0")
    if alternating_opt_maxiter <= 0:
        raise ValueError("alternating_opt_maxiter must be > 0")
    return n_atoms


def _threadpool_limit_context(num_threads: Optional[int]):
    """Return a context manager that limits BLAS/LAPACK threads when requested."""
    if num_threads is None:
        return nullcontext()
    if int(num_threads) <= 0:
        raise ValueError("num_threads must be > 0 when provided.")
    try:
        from threadpoolctl import threadpool_limits
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "num_threads control requires threadpoolctl. Install it with "
            "`pip install threadpoolctl`."
        ) from exc
    return threadpool_limits(limits=int(num_threads))


def _build_dense_basis_state(bits, *, np):
    """Build one computational-basis statevector in PennyLane's wire order."""
    qubits = int(len(bits))
    index = 0
    for wire, bit in enumerate(bits):
        if int(bit):
            index |= 1 << (qubits - 1 - int(wire))
    state = np.zeros(2**qubits, dtype=complex)
    state[int(index)] = 1.0
    return state


def _serialize_subset_labels(labels: Sequence[tuple[tuple[int, int], ...]]):
    """Convert subset labels into stable plain-python tuples."""
    return [tuple((int(p), int(q)) for p, q in label) for label in labels]


def _serialize_k_params(k_params, *, np) -> list[float]:
    """Serialize a parameter vector into plain Python floats."""
    params = np.asarray(k_params, dtype=float).reshape(-1)
    return [float(value) for value in params.tolist()]


def _serialize_projector_pair(pair: tuple[int, int]) -> tuple[int, int]:
    """Serialize one elementary projector label."""
    return (int(pair[0]), int(pair[1]))


def _public_record(record: dict[str, object]) -> dict[str, object]:
    """Drop private helper payloads before surfacing iteration diagnostics."""
    return {key: value for key, value in record.items() if not str(key).startswith("_")}


def _build_selected_subset_basis(
    system: _SystemData,
    *,
    selected_projector_indices: Sequence[int],
    subset_rank_max: int,
    np,
):
    """Build the selected-set subset basis up to the requested rank."""
    labels = [tuple()]
    masks = [np.ones(system.h_matrix.shape[0], dtype=bool)]
    projector_indices = [int(idx) for idx in selected_projector_indices]

    if int(subset_rank_max) >= 1:
        for projector_index in projector_indices:
            labels.append((system.projector_pairs[projector_index],))
            masks.append(np.asarray(system.projector_pair_masks[projector_index], dtype=bool))
    if int(subset_rank_max) >= 2:
        for first_pos, second_pos in combinations(range(len(projector_indices)), 2):
            first_index = projector_indices[first_pos]
            second_index = projector_indices[second_pos]
            labels.append(
                (
                    system.projector_pairs[first_index],
                    system.projector_pairs[second_index],
                )
            )
            masks.append(
                np.asarray(system.projector_pair_masks[first_index], dtype=bool)
                & np.asarray(system.projector_pair_masks[second_index], dtype=bool)
            )

    return tuple(labels), np.asarray(masks, dtype=bool)


def _resolve_basis_spec(
    system: _SystemData,
    *,
    num_basis_states: Optional[int],
    subset_labels,
    subset_masks,
    np,
):
    """Resolve a basis specification into explicit labels and masks."""
    if subset_labels is None and subset_masks is None:
        if num_basis_states is None:
            raise ValueError("Provide either num_basis_states or explicit subset_labels/subset_masks.")
        count = int(num_basis_states)
        if count <= 0:
            raise ValueError("num_basis_states must be > 0")
        return tuple(system.subset_labels[:count]), np.asarray(system.subset_masks[:count], dtype=bool)

    if subset_labels is None or subset_masks is None:
        raise ValueError("subset_labels and subset_masks must be provided together.")

    labels = tuple(tuple((int(p), int(q)) for p, q in label) for label in subset_labels)
    masks = np.asarray(subset_masks, dtype=bool)
    if masks.ndim != 2:
        raise ValueError("subset_masks must have shape (n_basis, hilbert_dim).")
    if len(labels) != int(masks.shape[0]):
        raise ValueError("subset_labels and subset_masks must have the same length.")
    if len(labels) == 0:
        raise ValueError("At least one basis label is required.")
    return labels, masks


def _default_initial_k_params(system: _SystemData, *, kappa: float, np):
    """Build a deterministic initial parameter vector from the user scalar seed.

    The production vector paths use a non-uniform ramp so that the frozen-K
    default explores a genuinely generalized rotation direction instead of a
    flat high-symmetry vector. The scalar control path keeps the single seed.
    """
    n_params = int(system.k_generator_matrices.shape[0])
    if n_params == 0:
        return np.zeros(0, dtype=float)
    if system.k_param_shape == "uniform_ov_same_spin":
        return np.asarray([float(kappa)], dtype=float)

    weights = np.linspace(1.0, float(n_params), num=n_params, dtype=float)
    weights /= float(n_params)
    return float(kappa) * weights


def _resolve_initial_k_params(
    system: _SystemData,
    *,
    kappa: float,
    k_params,
    np,
):
    """Resolve user-supplied K parameters into one validated float vector."""
    expected = int(system.k_generator_matrices.shape[0])
    if k_params is None:
        return _default_initial_k_params(system, kappa=kappa, np=np)

    params = np.asarray(k_params, dtype=float).reshape(-1)
    if params.size != expected:
        raise ValueError(
            f"k_params has length {int(params.size)}, but this system requires {expected} "
            f"parameter(s) for k_param_shape='{system.k_param_shape}'."
    )
    return np.asarray(params, dtype=float)


def _resolve_k_params_and_source(
    system: _SystemData,
    *,
    method_variant: str,
    kappa: float,
    k_params,
    k_source: str,
    np,
):
    """Resolve the effective K vector and metadata source for one run."""
    requested_k_source = _normalize_k_source(k_source)
    if k_params is not None:
        return (
            _resolve_initial_k_params(system, kappa=kappa, k_params=k_params, np=np),
            "manual",
            requested_k_source,
        )

    if requested_k_source == "manual":
        raise ValueError("k_source='manual' requires explicit k_params.")

    if (
        str(method_variant).strip().lower() == "frozen_k"
        and system.k_param_shape == "ov_same_spin_vector"
        and requested_k_source == "ccsd_t1"
    ):
        if system.ccsd_t1_k_params is None:
            raise ValueError("CCSD singles are unavailable for this frozen_K configuration.")
        return np.asarray(system.ccsd_t1_k_params, dtype=float), "ccsd_t1", requested_k_source

    return (
        _resolve_initial_k_params(system, kappa=kappa, k_params=None, np=np),
        "heuristic",
        requested_k_source,
    )


def _build_reference_mean_field(
    *,
    symbols: Sequence[str],
    geometry,
    basis: str,
    charge: int,
    spin: int,
    gto,
    scf,
):
    """Build the closed- or open-shell PySCF reference."""
    atom = [(symbols[i], tuple(float(x) for x in geometry[i])) for i in range(len(symbols))]
    molecule = gto.Mole()
    molecule.atom = atom
    molecule.unit = "Angstrom"
    molecule.basis = basis
    molecule.charge = int(charge)
    molecule.spin = int(spin)
    molecule.symmetry = False
    molecule.build()

    mean_field = scf.RHF(molecule) if int(spin) == 0 else scf.ROHF(molecule)
    mean_field.level_shift = 0.5
    mean_field.diis_space = 12
    mean_field.max_cycle = 100
    mean_field.kernel()
    if not mean_field.converged:
        mean_field = scf.newton(mean_field).run()
    return molecule, mean_field


def _build_active_space_reference(
    *,
    mean_field,
    active_orbitals: int,
    active_electrons: int,
    mcscf,
    np,
):
    """Build the CASCI reference data used everywhere else in the solver."""
    casci_ref = mcscf.CASCI(mean_field, int(active_orbitals), int(active_electrons))
    one_body_cas, core_energy = casci_ref.get_h1eff(mean_field.mo_coeff)
    two_body_cas = casci_ref.get_h2eff(mean_field.mo_coeff)
    return casci_ref, np.asarray(one_body_cas, dtype=float), two_body_cas, float(core_energy)


def _build_exact_small_system_hamiltonian(
    *,
    casci_ref,
    one_body_cas,
    two_body_cas,
    core_energy: float,
    active_electrons: int,
    hamiltonian_cutoff: float,
    ao2mo,
    fci,
    qml,
    np,
):
    """Build the exact qubit Hamiltonian, HF state, and FCI reference."""
    ncas = int(casci_ref.ncas)
    qubits = 2 * ncas
    wires = tuple(range(qubits))

    two_mo_for_qml = ao2mo.restore("1", two_body_cas, norb=ncas)
    two_mo_for_qml = np.swapaxes(two_mo_for_qml, 1, 3)
    core_constant = np.array([float(core_energy)], dtype=float)
    h_fermionic = qml.qchem.fermionic_observable(
        core_constant,
        np.asarray(one_body_cas, dtype=float),
        two_mo_for_qml,
        cutoff=float(hamiltonian_cutoff),
    )
    h_qubit = qml.jordan_wigner(h_fermionic)
    if hasattr(h_qubit, "terms"):
        coeffs, ops = h_qubit.terms()
    else:
        coeffs, ops = getattr(h_qubit, "coeffs", []), getattr(h_qubit, "ops", [])
    coeffs = np.asarray(coeffs, dtype=complex)
    if coeffs.size > 0 and (np.any(np.abs(coeffs.imag) > 1e-12) or coeffs.dtype.kind == "c"):
        h_qubit = qml.Hamiltonian(coeffs.real.astype(float), ops)

    h_matrix = np.asarray(qml.matrix(h_qubit, wire_order=wires), dtype=complex)
    h_matrix = 0.5 * (h_matrix + h_matrix.conj().T)

    hf_bits = np.asarray(qml.qchem.hf_state(int(active_electrons), qubits), dtype=int)
    hf_state = _build_dense_basis_state(hf_bits, np=np)
    hf_energy = float(np.real(np.vdot(hf_state, h_matrix @ hf_state)))

    two_mo_for_fci = ao2mo.restore("1", two_body_cas, norb=ncas)
    fci_energy, _ = fci.direct_spin1.kernel(
        np.asarray(one_body_cas, dtype=float),
        two_mo_for_fci,
        ncas,
        casci_ref.nelecas,
        ecore=float(core_energy),
    )
    return h_matrix, hf_bits, hf_state, float(hf_energy), float(fci_energy), int(qubits)


def _build_k_generator_data(
    *,
    hf_bits,
    qubits: int,
    k_param_shape: str,
    wires,
    from_string,
    qml,
    np,
):
    """Build the one-body K generators used by the chosen parameterization."""
    occupied = tuple(int(idx) for idx, bit in enumerate(hf_bits) if int(bit) == 1)
    virtual = tuple(int(idx) for idx, bit in enumerate(hf_bits) if int(bit) == 0)

    if k_param_shape == "generalized_same_spin_vector":
        generator_pairs = [
            (int(left), int(right))
            for left, right in combinations(range(qubits), 2)
            if (int(left) % 2) == (int(right) % 2)
        ]
    else:
        generator_pairs = [
            (int(occ), int(virt))
            for occ in occupied
            for virt in virtual
            if (int(virt) % 2) == (int(occ) % 2)
        ]

    raw_generator_matrices = []
    raw_generator_pairs = []
    raw_generator_labels = []
    for left, right in generator_pairs:
        term = from_string(f"{right}+ {left}-") - from_string(f"{left}+ {right}-")
        k_qubit = qml.jordan_wigner(term)
        generator_matrix = np.asarray(qml.matrix(k_qubit, wire_order=wires), dtype=complex)
        generator_matrix = 0.5 * (generator_matrix - generator_matrix.conj().T)
        raw_generator_matrices.append(np.asarray(generator_matrix, dtype=complex))
        raw_generator_pairs.append((int(left), int(right)))
        raw_generator_labels.append(f"{right}<-{left}")

    if k_param_shape == "uniform_ov_same_spin":
        if len(raw_generator_matrices) == 0:
            return np.zeros((0, 2**int(qubits), 2**int(qubits)), dtype=complex), tuple(), tuple()
        template = np.sum(np.asarray(raw_generator_matrices, dtype=complex), axis=0)
        return np.asarray([template], dtype=complex), tuple(raw_generator_pairs), ("uniform_ov_same_spin",)

    if len(raw_generator_matrices) == 0:
        k_generator_matrices = np.zeros((0, 2**int(qubits), 2**int(qubits)), dtype=complex)
    else:
        k_generator_matrices = np.asarray(raw_generator_matrices, dtype=complex)
    return k_generator_matrices, tuple(raw_generator_pairs), tuple(raw_generator_labels)


def _build_ccsd_t1_seed(
    *,
    mean_field,
    spin: int,
    active_electrons: int,
    active_orbitals: int,
    raw_generator_pairs: Sequence[tuple[int, int]],
    cc,
    extract_active_ccsd_amplitudes,
    np,
):
    """Map active-space RHF-CCSD singles onto the OV same-spin K slots."""
    if int(spin) != 0:
        raise ValueError("k_source='ccsd_t1' currently requires a closed-shell RHF reference (spin=0).")

    mycc = cc.CCSD(mean_field)
    mycc.kernel()
    t1_active, _t2_active = extract_active_ccsd_amplitudes(
        mycc,
        active_electrons=int(active_electrons),
        active_orbitals=int(active_orbitals),
        np=np,
    )

    active_occ = int(active_electrons // 2)
    active_virt = int(active_orbitals - active_occ)
    ccsd_t1_k_params = np.zeros(len(raw_generator_pairs), dtype=float)
    pair_to_index = {pair: idx for idx, pair in enumerate(raw_generator_pairs)}

    for occ_spatial in range(active_occ):
        occ_alpha = int(2 * occ_spatial)
        occ_beta = int(occ_alpha + 1)
        for virt_offset in range(active_virt):
            virt_spatial = int(active_occ + virt_offset)
            virt_alpha = int(2 * virt_spatial)
            virt_beta = int(virt_alpha + 1)
            value = float(np.real(t1_active[occ_spatial, virt_offset]))
            for pair in ((occ_alpha, virt_alpha), (occ_beta, virt_beta)):
                pair_idx = pair_to_index.get(pair)
                if pair_idx is not None:
                    ccsd_t1_k_params[pair_idx] = value
    return np.asarray(ccsd_t1_k_params, dtype=float)


def _build_projector_pair_masks(*, qubits: int, np):
    """Build boolean masks for all number-number projectors n_p n_q."""
    dim = 2**int(qubits)
    basis_indices = np.arange(dim, dtype=int)
    occupancy = np.asarray(
        [((basis_indices >> (qubits - 1 - wire)) & 1).astype(bool) for wire in range(qubits)],
        dtype=bool,
    )
    projector_pairs = tuple((int(p), int(q)) for p, q in combinations(range(qubits), 2))
    pair_masks = [occupancy[p] & occupancy[q] for p, q in projector_pairs]
    return projector_pairs, np.asarray(pair_masks, dtype=bool)


def _build_rank_limited_subset_basis(
    *,
    projector_pairs: Sequence[tuple[int, int]],
    pair_masks,
    subset_rank_max: int,
    dim: int,
    np,
):
    """Build the graded-lex subset basis used by the legacy fallback path."""
    subset_labels = [tuple()]
    subset_masks = [np.ones(int(dim), dtype=bool)]
    if int(subset_rank_max) >= 1:
        for pair_idx, pair in enumerate(projector_pairs):
            subset_labels.append((pair,))
            subset_masks.append(np.asarray(pair_masks[pair_idx], dtype=bool))
    if int(subset_rank_max) >= 2:
        for first_idx, second_idx in combinations(range(len(projector_pairs)), 2):
            subset_labels.append((projector_pairs[first_idx], projector_pairs[second_idx]))
            subset_masks.append(
                np.asarray(pair_masks[first_idx], dtype=bool)
                & np.asarray(pair_masks[second_idx], dtype=bool)
            )
    return tuple(subset_labels), np.asarray(subset_masks, dtype=bool)


def _build_system_data(
    *,
    symbols: Sequence[str],
    geometry,
    basis: str,
    charge: int,
    spin: int,
    active_electrons: int,
    active_orbitals: int,
    hamiltonian_cutoff: float,
    subset_rank_max: int,
    k_param_shape: str,
    compute_ccsd_t1: bool = False,
):
    """Build all exact small-system data used by the solver."""
    try:
        import numpy as np
        import pennylane as qml
        from pennylane.fermi import from_string
        from pyscf import ao2mo, cc, fci, gto, mcscf, scf
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "gucj_gcim requires dependencies. Install at least: "
            "`pip install numpy scipy pennylane pyscf`."
        ) from exc

    from QCANT.cvqe.cvqe import _extract_active_ccsd_amplitudes

    _molecule, mean_field = _build_reference_mean_field(
        symbols=symbols,
        geometry=geometry,
        basis=basis,
        charge=charge,
        spin=spin,
        gto=gto,
        scf=scf,
    )
    casci_ref, one_body_cas, two_body_cas, core_energy = _build_active_space_reference(
        mean_field=mean_field,
        active_orbitals=active_orbitals,
        active_electrons=active_electrons,
        mcscf=mcscf,
        np=np,
    )
    h_matrix, hf_bits, hf_state, hf_energy, fci_energy, qubits = _build_exact_small_system_hamiltonian(
        casci_ref=casci_ref,
        one_body_cas=one_body_cas,
        two_body_cas=two_body_cas,
        core_energy=core_energy,
        active_electrons=active_electrons,
        hamiltonian_cutoff=hamiltonian_cutoff,
        ao2mo=ao2mo,
        fci=fci,
        qml=qml,
        np=np,
    )
    wires = tuple(range(qubits))
    dim = int(h_matrix.shape[0])

    k_generator_matrices, raw_generator_pairs, k_parameter_labels = _build_k_generator_data(
        hf_bits=hf_bits,
        qubits=qubits,
        k_param_shape=k_param_shape,
        wires=wires,
        from_string=from_string,
        qml=qml,
        np=np,
    )

    ccsd_t1_k_params = None
    if compute_ccsd_t1:
        ccsd_t1_k_params = _build_ccsd_t1_seed(
            mean_field=mean_field,
            spin=spin,
            active_electrons=active_electrons,
            active_orbitals=active_orbitals,
            raw_generator_pairs=raw_generator_pairs,
            cc=cc,
            extract_active_ccsd_amplitudes=_extract_active_ccsd_amplitudes,
            np=np,
        )

    projector_pairs, pair_masks = _build_projector_pair_masks(qubits=qubits, np=np)
    subset_labels, subset_masks = _build_rank_limited_subset_basis(
        projector_pairs=projector_pairs,
        pair_masks=pair_masks,
        subset_rank_max=subset_rank_max,
        dim=dim,
        np=np,
    )

    return _SystemData(
        h_matrix=np.asarray(h_matrix, dtype=complex),
        hf_state=np.asarray(hf_state, dtype=complex),
        fci_energy=float(fci_energy),
        hf_energy=float(hf_energy),
        qubits=int(qubits),
        projector_pairs=tuple(projector_pairs),
        projector_pair_masks=np.asarray(pair_masks, dtype=bool),
        subset_labels=tuple(subset_labels),
        subset_masks=np.asarray(subset_masks, dtype=bool),
        k_generator_matrices=np.asarray(k_generator_matrices, dtype=complex),
        k_generator_pairs=tuple((int(occ), int(virt)) for occ, virt in raw_generator_pairs),
        k_parameter_labels=tuple(k_parameter_labels),
        k_param_shape=str(k_param_shape),
        ccsd_t1_k_params=None if ccsd_t1_k_params is None else np.asarray(ccsd_t1_k_params, dtype=float),
    )


def _unitary_from_k_params(system: _SystemData, *, k_params, np):
    """Build ``exp(K)`` from a validated real K-parameter vector."""
    from scipy.linalg import expm

    params = np.asarray(k_params, dtype=float).reshape(-1)
    n_params = int(system.k_generator_matrices.shape[0])
    if n_params == 0 or params.size == 0:
        dim = int(system.h_matrix.shape[0])
        return np.eye(dim, dtype=complex)

    if params.size != n_params:
        raise ValueError(
            f"Expected {n_params} K parameter(s), received {int(params.size)}."
        )

    k_matrix = np.tensordot(params, np.asarray(system.k_generator_matrices, dtype=complex), axes=(0, 0))
    return np.asarray(expm(k_matrix), dtype=complex)


def _rotation_artifacts_from_k_params(system: _SystemData, *, k_params, np):
    """Build the rotated reference state and Hamiltonian for one K vector."""
    unitary = _unitary_from_k_params(system, k_params=k_params, np=np)
    phi_k = unitary @ np.asarray(system.hf_state, dtype=complex)
    hbar_k = unitary @ np.asarray(system.h_matrix, dtype=complex) @ unitary.conj().T
    hbar_k = 0.5 * (hbar_k + hbar_k.conj().T)
    return np.asarray(phi_k, dtype=complex), np.asarray(hbar_k, dtype=complex)


def _evaluate_subspace(
    system: _SystemData,
    *,
    num_basis_states: Optional[int] = None,
    subset_labels=None,
    subset_masks=None,
    k_params,
    overlap_tol: float,
    regularization: float,
    include_iteration_matrices: bool,
    rotation_artifacts=None,
    return_state: bool = False,
):
    """Evaluate the lowest generalized Ritz root for one cumulative subspace."""
    import numpy as np

    params = np.asarray(k_params, dtype=float).reshape(-1)
    if rotation_artifacts is None:
        phi_k, hbar_k = _rotation_artifacts_from_k_params(system, k_params=params, np=np)
    else:
        phi_k, hbar_k = rotation_artifacts

    basis_labels, mask_block = _resolve_basis_spec(
        system,
        num_basis_states=num_basis_states,
        subset_labels=subset_labels,
        subset_masks=subset_masks,
        np=np,
    )
    basis_block = phi_k[:, None] * mask_block.T.astype(complex)

    zero_tol = max(float(regularization), 1e-14)
    basis_norms = np.real(np.sum(np.abs(basis_block) ** 2, axis=0))
    keep_nonzero = np.asarray(basis_norms > zero_tol, dtype=bool)
    basis_nonzero = np.asarray(basis_block[:, keep_nonzero], dtype=complex)
    labels_nonzero = [basis_labels[idx] for idx, keep in enumerate(keep_nonzero) if keep]
    norms_nonzero = [float(basis_norms[idx]) for idx, keep in enumerate(keep_nonzero) if keep]

    if basis_nonzero.shape[1] == 0:
        raise ValueError("All basis states were screened as zero-norm.")

    overlap = basis_nonzero.conj().T @ basis_nonzero
    overlap = 0.5 * (overlap + overlap.conj().T)
    projected_h = basis_nonzero.conj().T @ (hbar_k @ basis_nonzero)
    projected_h = 0.5 * (projected_h + projected_h.conj().T)

    overlap_eigvals_raw, overlap_eigvecs = np.linalg.eigh(overlap)
    overlap_eigvals_raw = np.real(overlap_eigvals_raw)
    overlap_eigvals = np.asarray(overlap_eigvals_raw, dtype=float)
    small_negative = overlap_eigvals < 0.0
    if np.any(small_negative & (overlap_eigvals < -float(regularization))):
        raise ValueError("Overlap matrix is not positive semidefinite within regularization.")
    overlap_eigvals[small_negative] = 0.0

    keep_modes = overlap_eigvals > float(overlap_tol)
    retained_rank = int(np.count_nonzero(keep_modes))
    if retained_rank == 0:
        raise ValueError("No overlap eigenmodes survived the overlap threshold.")

    xform = overlap_eigvecs[:, keep_modes] / np.sqrt(overlap_eigvals[keep_modes])[None, :]
    reduced_h = xform.conj().T @ projected_h @ xform
    reduced_h = 0.5 * (reduced_h + reduced_h.conj().T)
    reduced_eigvals, reduced_eigvecs = np.linalg.eigh(reduced_h)

    retained_eigvals = np.asarray(overlap_eigvals[keep_modes], dtype=float)
    lambda_min = float(retained_eigvals[0])
    lambda_max = float(retained_eigvals[-1])
    cond = float(lambda_max / lambda_min) if lambda_min > 0.0 else float("inf")
    energy = float(np.real(reduced_eigvals[0]))
    abs_error = float(abs(energy - float(system.fci_energy)))

    record = {
        "energy": energy,
        "abs_error_fci": abs_error,
        "k_params": _serialize_k_params(params, np=np),
        "k_param_norm": float(np.linalg.norm(params)),
        "kappa": float(params[0]) if params.size == 1 else float(np.linalg.norm(params)),
        "basis_dimension_before_screening": int(len(basis_labels)),
        "basis_dimension_after_zero_screening": int(basis_nonzero.shape[1]),
        "basis_dimension_after_screening": retained_rank,
        "retained_subset_labels": _serialize_subset_labels(labels_nonzero),
        "basis_norms_nonzero": [float(value) for value in norms_nonzero],
        "overlap_eigenvalues_raw": [float(value) for value in overlap_eigvals_raw.tolist()],
        "overlap_lambda_min_raw": float(np.min(overlap_eigvals_raw)),
        "overlap_lambda_min_retained": lambda_min,
        "overlap_lambda_max_retained": lambda_max,
        "overlap_condition_number": cond,
        "added_subset_label": _serialize_subset_labels([basis_labels[-1]])[0],
    }
    if return_state:
        reduced_ground = np.asarray(reduced_eigvecs[:, 0], dtype=complex)
        ritz_coefficients = np.asarray(xform @ reduced_ground, dtype=complex)
        ritz_state = np.asarray(basis_nonzero @ ritz_coefficients, dtype=complex)
        ritz_norm = float(np.linalg.norm(ritz_state))
        if ritz_norm <= 0.0:
            raise ValueError("Lowest Ritz state has zero norm.")
        ritz_state /= ritz_norm
        record["_ritz_state_vector"] = np.asarray(ritz_state, dtype=complex)
        record["_ritz_coefficients"] = np.asarray(ritz_coefficients, dtype=complex)
    if include_iteration_matrices:
        record["projected_hamiltonian"] = np.asarray(projected_h, dtype=complex)
        record["overlap_matrix"] = np.asarray(overlap, dtype=complex)
        record["reduced_hamiltonian"] = np.asarray(reduced_h, dtype=complex)
    return record


def _decorate_iteration_record(
    *,
    record: dict[str, object],
    iteration: int,
    method_variant: str,
    selection_mode: str,
    selected_projector,
    selected_projector_index,
    winning_selection_score,
    selection_scores,
):
    """Attach loop metadata to one raw subspace evaluation record."""
    decorated = dict(record)
    decorated["iteration"] = int(iteration)
    decorated["method_variant"] = str(method_variant)
    decorated["selection_mode"] = str(selection_mode)
    decorated["selected_projector"] = selected_projector
    decorated["selected_projector_index"] = selected_projector_index
    decorated["winning_selection_score"] = winning_selection_score
    decorated["selection_scores"] = selection_scores
    return decorated


def _append_iteration_result(
    *,
    iteration_record: dict[str, object],
    subset_label,
    iteration_records: list[dict[str, object]],
    k_history: list[object],
    energies: list[float],
    subset_labels: list[tuple[tuple[int, int], ...]],
):
    """Store one completed GCIM operator-selection iteration."""
    iteration_records.append(_public_record(iteration_record))
    k_value = iteration_record["k_params"]
    k_history.append(k_value if len(k_value) != 1 else float(k_value[0]))
    energies.append(float(iteration_record["energy"]))
    subset_labels.append(subset_label)


def _optimize_k_params(
    system: _SystemData,
    *,
    num_basis_states: Optional[int] = None,
    subset_labels=None,
    subset_masks=None,
    start_k_params,
    bounds,
    overlap_tol: float,
    regularization: float,
    maxiter: int,
    include_iteration_matrices: bool,
    return_state: bool = False,
):
    """Optimize a scalar or vector K parameterization over one bounded interval."""
    import numpy as np
    from scipy import optimize

    start = np.asarray(start_k_params, dtype=float).reshape(-1)
    if start.size == 0:
        record = _evaluate_subspace(
            system,
            num_basis_states=num_basis_states,
            subset_labels=subset_labels,
            subset_masks=subset_masks,
            k_params=start,
            overlap_tol=overlap_tol,
            regularization=regularization,
            include_iteration_matrices=include_iteration_matrices,
            return_state=return_state,
        )
        record["optimization_success"] = True
        record["optimization_nfev"] = 1
        record["optimization_message"] = "No K parameters to optimize."
        return start, record

    bounds_list = [(float(lower), float(upper)) for lower, upper in bounds]
    lower = np.asarray([bound[0] for bound in bounds_list], dtype=float)
    upper = np.asarray([bound[1] for bound in bounds_list], dtype=float)
    start = np.clip(start, lower, upper)

    cache: dict[tuple[float, ...], tuple[object, dict[str, object]]] = {}

    def _cache_key(params):
        arr = np.asarray(params, dtype=float).reshape(-1)
        return tuple(float(value) for value in np.round(arr, decimals=10).tolist())

    def _evaluate_cached(params):
        arr = np.clip(np.asarray(params, dtype=float).reshape(-1), lower, upper)
        key = _cache_key(arr)
        if key not in cache:
            cache[key] = (
                np.asarray(arr, dtype=float),
                _evaluate_subspace(
                    system,
                    num_basis_states=num_basis_states,
                    subset_labels=subset_labels,
                    subset_masks=subset_masks,
                    k_params=arr,
                    overlap_tol=overlap_tol,
                    regularization=regularization,
                    include_iteration_matrices=include_iteration_matrices,
                    return_state=return_state,
                ),
            )
        return cache[key]

    def _objective(x):
        _params, record = _evaluate_cached(x)
        return float(record["energy"])

    if start.size <= 2:
        result = optimize.minimize(
            _objective,
            x0=start,
            method="Powell",
            bounds=bounds_list,
            options={"maxiter": int(maxiter), "xtol": 1e-6, "ftol": 1e-12},
        )
    else:
        result = optimize.minimize(
            _objective,
            x0=start,
            method="L-BFGS-B",
            bounds=bounds_list,
            options={"maxiter": int(maxiter), "ftol": 1e-12, "gtol": 1e-6, "maxls": 20},
        )

    _evaluate_cached(result.x)
    best_params, best_record = min(cache.values(), key=lambda item: float(item[1]["energy"]))
    record = dict(best_record)
    record["optimization_success"] = bool(result.success)
    record["optimization_nfev"] = int(getattr(result, "nfev", -1))
    record["optimization_message"] = str(getattr(result, "message", ""))
    return np.asarray(best_params, dtype=float), record


def _projector_commutator_score(
    system: _SystemData,
    *,
    projector_index: int,
    ritz_state,
    hbar_k,
    np,
) -> float:
    """Evaluate the rotated-frame GCIM selection score for one projector."""
    mask = np.asarray(system.projector_pair_masks[int(projector_index)], dtype=bool).astype(complex)
    psi = np.asarray(ritz_state, dtype=complex).reshape(-1)
    hpsi = np.asarray(hbar_k, dtype=complex) @ psi
    ppsi = mask * psi
    comm_exp = 1j * (np.vdot(hpsi, ppsi) - np.vdot(ppsi, hpsi))
    return float(abs(2.0 * float(np.real_if_close(comm_exp))))


def _score_projector_candidates(
    system: _SystemData,
    *,
    selected_projector_indices: Sequence[int],
    ritz_state,
    hbar_k,
    np,
) -> list[dict[str, object]]:
    """Score all omitted projectors and return them in pool order."""
    selected_set = {int(idx) for idx in selected_projector_indices}
    scores = []
    for projector_index, pair in enumerate(system.projector_pairs):
        if projector_index in selected_set:
            continue
        scores.append(
            {
                "projector_index": int(projector_index),
                "projector": _serialize_projector_pair(pair),
                "score": _projector_commutator_score(
                    system,
                    projector_index=projector_index,
                    ritz_state=ritz_state,
                    hbar_k=hbar_k,
                    np=np,
                ),
            }
        )
    return scores


def _evaluate_current_basis_for_selection(
    system: _SystemData,
    *,
    selected_projector_indices: Sequence[int],
    subset_rank_max: int,
    current_k_params,
    overlap_tol: float,
    regularization: float,
    rotation_artifacts=None,
    np,
):
    """Evaluate the current selected-projector basis and return the Ritz state."""
    basis_labels, basis_masks = _build_selected_subset_basis(
        system,
        selected_projector_indices=selected_projector_indices,
        subset_rank_max=subset_rank_max,
        np=np,
    )
    if rotation_artifacts is None:
        rotation_artifacts = _rotation_artifacts_from_k_params(system, k_params=current_k_params, np=np)
    record = _evaluate_subspace(
        system,
        subset_labels=basis_labels,
        subset_masks=basis_masks,
        k_params=current_k_params,
        overlap_tol=overlap_tol,
        regularization=regularization,
        include_iteration_matrices=False,
        rotation_artifacts=rotation_artifacts,
        return_state=True,
    )
    return basis_labels, basis_masks, rotation_artifacts, record


def _build_run_details(
    system: _SystemData,
    *,
    method_variant: str,
    k_source: str,
    requested_k_source: str,
    initial_k_params,
    subset_labels,
    selected_projector_order,
    selection_mode: str,
    iteration_records,
    frozen_k_params,
    initial_reference_record,
    np,
):
    """Build the public details dictionary returned by ``gucj_gcim``."""
    details = {
        "method_variant": str(method_variant),
        "variant_status": "legacy" if _is_legacy_method_variant(method_variant) else "default",
        "k_source": str(k_source),
        "requested_k_source": str(requested_k_source),
        "k_param_shape": system.k_param_shape,
        "k_parameter_labels": list(system.k_parameter_labels),
        "initial_k_params": _serialize_k_params(initial_k_params, np=np),
        "fci_energy": float(system.fci_energy),
        "hf_energy": float(system.hf_energy),
        "projector_pairs": _serialize_subset_labels([(pair,) for pair in system.projector_pairs]),
        "subset_labels": _serialize_subset_labels(subset_labels),
        "selected_projector_order": [_serialize_projector_pair(pair) for pair in selected_projector_order],
        "selection_mode": str(selection_mode),
        "iteration_records": iteration_records,
    }
    if selection_mode == "adaptive_commutator":
        details["initial_reference_record"] = initial_reference_record
    if frozen_k_params is not None:
        details["frozen_k_params"] = _serialize_k_params(frozen_k_params, np=np)
    return details


def _maybe_reuse_current_k(
    system: _SystemData,
    *,
    num_basis_states: Optional[int] = None,
    subset_labels=None,
    subset_masks=None,
    current_k_params,
    overlap_tol: float,
    regularization: float,
    include_iteration_matrices: bool,
    return_state: bool,
):
    """Skip expensive K reoptimization when the current basis is already exact enough."""
    record = _evaluate_subspace(
        system,
        num_basis_states=num_basis_states,
        subset_labels=subset_labels,
        subset_masks=subset_masks,
        k_params=current_k_params,
        overlap_tol=overlap_tol,
        regularization=regularization,
        include_iteration_matrices=include_iteration_matrices,
        return_state=return_state,
    )
    if float(record["abs_error_fci"]) > max(1e-10, float(overlap_tol)):
        return None

    reused = dict(record)
    reused["optimization_success"] = True
    reused["optimization_nfev"] = 1
    reused["optimization_message"] = "Skipped: current K already exact within tolerance."
    return reused


def _evaluate_basis_for_method(
    system: _SystemData,
    *,
    method_variant: str,
    current_k_params,
    frozen_k_params,
    full_bounds,
    num_basis_states: Optional[int] = None,
    subset_labels=None,
    subset_masks=None,
    overlap_tol: float,
    regularization: float,
    outer_opt_maxiter: int,
    alternating_window: float,
    alternating_opt_maxiter: int,
    include_iteration_matrices: bool,
    return_state: bool,
    skip_if_exact: bool,
    frozen_rotation_artifacts=None,
    np,
):
    """Evaluate one basis for the chosen algorithm variant."""
    basis_kwargs = {
        "num_basis_states": num_basis_states,
        "subset_labels": subset_labels,
        "subset_masks": subset_masks,
        "overlap_tol": overlap_tol,
        "regularization": regularization,
        "include_iteration_matrices": include_iteration_matrices,
        "return_state": return_state,
    }

    if method_variant == "frozen_k":
        params = np.asarray(frozen_k_params, dtype=float)
        if frozen_rotation_artifacts is None:
            rotation_artifacts = _rotation_artifacts_from_k_params(system, k_params=params, np=np)
        else:
            rotation_artifacts = frozen_rotation_artifacts
        record = _evaluate_subspace(
            system,
            k_params=params,
            rotation_artifacts=rotation_artifacts,
            **basis_kwargs,
        )
        return params, record, rotation_artifacts

    if method_variant == "optimized_k_outer_loop":
        bounds = full_bounds
        maxiter = int(outer_opt_maxiter)
    elif method_variant == "alternating_k_and_subspace":
        bounds = [
            (
                max(-math.pi, float(value) - float(alternating_window)),
                min(math.pi, float(value) + float(alternating_window)),
            )
            for value in np.asarray(current_k_params, dtype=float)
        ]
        maxiter = int(alternating_opt_maxiter)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported method_variant: {method_variant}")

    if skip_if_exact:
        reused_record = _maybe_reuse_current_k(
            system,
            num_basis_states=num_basis_states,
            subset_labels=subset_labels,
            subset_masks=subset_masks,
            current_k_params=current_k_params,
            overlap_tol=overlap_tol,
            regularization=regularization,
            include_iteration_matrices=include_iteration_matrices,
            return_state=return_state,
        )
        if reused_record is not None:
            params = np.asarray(current_k_params, dtype=float)
            rotation_artifacts = _rotation_artifacts_from_k_params(system, k_params=params, np=np)
            return params, reused_record, rotation_artifacts

    params, record = _optimize_k_params(
        system,
        num_basis_states=num_basis_states,
        subset_labels=subset_labels,
        subset_masks=subset_masks,
        start_k_params=current_k_params,
        bounds=bounds,
        overlap_tol=overlap_tol,
        regularization=regularization,
        maxiter=maxiter,
        include_iteration_matrices=include_iteration_matrices,
        return_state=return_state,
    )
    rotation_artifacts = _rotation_artifacts_from_k_params(system, k_params=params, np=np)
    return np.asarray(params, dtype=float), record, rotation_artifacts


def _run_subset_rank_zero_variant(
    system: _SystemData,
    *,
    method_variant: str,
    initial_k_params,
    k_source: str,
    requested_k_source: str,
    selection_mode: str,
    subset_rank_max: int,
    overlap_tol: float,
    regularization: float,
    outer_opt_maxiter: int,
    alternating_window: float,
    alternating_opt_maxiter: int,
    include_iteration_matrices: bool,
):
    """Run the special rank-0 case where the basis only contains the reference."""
    import numpy as np

    iteration_records: list[dict[str, object]] = []
    k_history: list[object] = []
    energies: list[float] = []
    subset_label_history: list[tuple[tuple[int, int], ...]] = []

    current_k_params = np.asarray(initial_k_params, dtype=float).reshape(-1)
    full_bounds = [(-math.pi, math.pi)] * int(current_k_params.size)
    frozen_k_params = np.asarray(current_k_params, dtype=float) if method_variant == "frozen_k" else None
    basis_labels, basis_masks = _build_selected_subset_basis(
        system,
        selected_projector_indices=tuple(),
        subset_rank_max=subset_rank_max,
        np=np,
    )
    current_k_params, record, _rotation_artifacts = _evaluate_basis_for_method(
        system,
        method_variant=method_variant,
        current_k_params=current_k_params,
        frozen_k_params=frozen_k_params,
        full_bounds=full_bounds,
        subset_labels=basis_labels,
        subset_masks=basis_masks,
        overlap_tol=overlap_tol,
        regularization=regularization,
        outer_opt_maxiter=outer_opt_maxiter,
        alternating_window=alternating_window,
        alternating_opt_maxiter=alternating_opt_maxiter,
        include_iteration_matrices=include_iteration_matrices,
        return_state=False,
        skip_if_exact=False,
        np=np,
    )
    decorated = _decorate_iteration_record(
        record=record,
        iteration=0,
        method_variant=method_variant,
        selection_mode=selection_mode,
        selected_projector=None,
        selected_projector_index=None,
        winning_selection_score=None,
        selection_scores=[],
    )
    _append_iteration_result(
        iteration_record=decorated,
        subset_label=tuple(),
        iteration_records=iteration_records,
        k_history=k_history,
        energies=energies,
        subset_labels=subset_label_history,
    )
    details = _build_run_details(
        system,
        method_variant=method_variant,
        k_source=k_source,
        requested_k_source=requested_k_source,
        initial_k_params=initial_k_params,
        subset_labels=subset_label_history,
        selected_projector_order=[],
        selection_mode=selection_mode,
        iteration_records=iteration_records,
        frozen_k_params=frozen_k_params,
        initial_reference_record=None,
        np=np,
    )
    return k_history, _serialize_subset_labels(subset_label_history), energies, details


def _run_graded_lex_variant(
    system: _SystemData,
    *,
    method_variant: str,
    initial_k_params,
    k_source: str,
    requested_k_source: str,
    selection_mode: str,
    overlap_tol: float,
    regularization: float,
    outer_opt_maxiter: int,
    alternating_window: float,
    alternating_opt_maxiter: int,
    include_iteration_matrices: bool,
):
    """Run the advanced fixed graded-lexicographic basis-growth fallback."""
    import numpy as np

    iteration_records: list[dict[str, object]] = []
    k_history: list[object] = []
    energies: list[float] = []
    subset_label_history: list[tuple[tuple[int, int], ...]] = []

    current_k_params = np.asarray(initial_k_params, dtype=float).reshape(-1)
    frozen_k_params = np.asarray(current_k_params, dtype=float) if method_variant == "frozen_k" else None
    full_bounds = [(-math.pi, math.pi)] * int(current_k_params.size)

    for iteration in range(len(system.subset_labels)):
        num_basis_states = int(iteration + 1)
        current_k_params, record, _rotation_artifacts = _evaluate_basis_for_method(
            system,
            method_variant=method_variant,
            current_k_params=current_k_params,
            frozen_k_params=frozen_k_params,
            full_bounds=full_bounds,
            num_basis_states=num_basis_states,
            overlap_tol=overlap_tol,
            regularization=regularization,
            outer_opt_maxiter=outer_opt_maxiter,
            alternating_window=alternating_window,
            alternating_opt_maxiter=alternating_opt_maxiter,
            include_iteration_matrices=include_iteration_matrices,
            return_state=False,
            skip_if_exact=(method_variant != "frozen_k"),
            np=np,
        )
        decorated = _decorate_iteration_record(
            record=record,
            iteration=iteration,
            method_variant=method_variant,
            selection_mode=selection_mode,
            selected_projector=None,
            selected_projector_index=None,
            winning_selection_score=None,
            selection_scores=[],
        )
        _append_iteration_result(
            iteration_record=decorated,
            subset_label=system.subset_labels[iteration],
            iteration_records=iteration_records,
            k_history=k_history,
            energies=energies,
            subset_labels=subset_label_history,
        )

    details = _build_run_details(
        system,
        method_variant=method_variant,
        k_source=k_source,
        requested_k_source=requested_k_source,
        initial_k_params=initial_k_params,
        subset_labels=subset_label_history,
        selected_projector_order=[],
        selection_mode=selection_mode,
        iteration_records=iteration_records,
        frozen_k_params=frozen_k_params,
        initial_reference_record=None,
        np=np,
    )
    return k_history, _serialize_subset_labels(subset_label_history), energies, details


def _run_adaptive_variant(
    system: _SystemData,
    *,
    method_variant: str,
    initial_k_params,
    k_source: str,
    requested_k_source: str,
    selection_mode: str,
    subset_rank_max: int,
    overlap_tol: float,
    regularization: float,
    outer_opt_maxiter: int,
    alternating_window: float,
    alternating_opt_maxiter: int,
    include_iteration_matrices: bool,
):
    """Run the adaptive commutator-selected GCIM loop."""
    import numpy as np

    iteration_records: list[dict[str, object]] = []
    k_history: list[object] = []
    energies: list[float] = []
    subset_label_history: list[tuple[tuple[int, int], ...]] = []
    selected_projector_order: list[tuple[int, int]] = []

    current_k_params = np.asarray(initial_k_params, dtype=float).reshape(-1)
    frozen_k_params = np.asarray(current_k_params, dtype=float) if method_variant == "frozen_k" else None
    full_bounds = [(-math.pi, math.pi)] * int(current_k_params.size)

    selected_projector_indices: list[int] = []
    frozen_rotation_artifacts = None
    if method_variant == "frozen_k":
        frozen_rotation_artifacts = _rotation_artifacts_from_k_params(
            system,
            k_params=current_k_params,
            np=np,
        )
    # Start from the current selected-projector subspace so the first score is
    # evaluated with the current lowest Ritz state, just like ADAPT-style growth.
    _basis_labels, _basis_masks, scoring_rotation_artifacts, scoring_record = _evaluate_current_basis_for_selection(
        system,
        selected_projector_indices=selected_projector_indices,
        subset_rank_max=subset_rank_max,
        current_k_params=current_k_params,
        overlap_tol=overlap_tol,
        regularization=regularization,
        rotation_artifacts=frozen_rotation_artifacts,
        np=np,
    )
    initial_reference_record = _public_record(dict(scoring_record))

    for iteration in range(len(system.projector_pairs)):
        candidate_scores = _score_projector_candidates(
            system,
            selected_projector_indices=selected_projector_indices,
            ritz_state=scoring_record["_ritz_state_vector"],
            hbar_k=scoring_rotation_artifacts[1],
            np=np,
        )
        if len(candidate_scores) == 0:
            break

        winner = max(candidate_scores, key=lambda item: (float(item["score"]), -int(item["projector_index"])))
        winner_pair = tuple(winner["projector"])
        selected_projector_indices.append(int(winner["projector_index"]))
        selected_projector_order.append(winner_pair)

        # Rebuild the GCIM basis from all subsets over the selected projector
        # set, then solve again in that enlarged nonorthogonal space.
        current_basis_labels, current_basis_masks = _build_selected_subset_basis(
            system,
            selected_projector_indices=selected_projector_indices,
            subset_rank_max=subset_rank_max,
            np=np,
        )
        current_k_params, record, scoring_rotation_artifacts = _evaluate_basis_for_method(
            system,
            method_variant=method_variant,
            current_k_params=current_k_params,
            frozen_k_params=frozen_k_params,
            full_bounds=full_bounds,
            subset_labels=current_basis_labels,
            subset_masks=current_basis_masks,
            overlap_tol=overlap_tol,
            regularization=regularization,
            outer_opt_maxiter=outer_opt_maxiter,
            alternating_window=alternating_window,
            alternating_opt_maxiter=alternating_opt_maxiter,
            include_iteration_matrices=include_iteration_matrices,
            return_state=True,
            skip_if_exact=False,
            frozen_rotation_artifacts=frozen_rotation_artifacts,
            np=np,
        )
        scoring_record = dict(record)
        decorated = _decorate_iteration_record(
            record=scoring_record,
            iteration=iteration,
            method_variant=method_variant,
            selection_mode=selection_mode,
            selected_projector=_serialize_projector_pair(winner_pair),
            selected_projector_index=int(winner["projector_index"]),
            winning_selection_score=float(winner["score"]),
            selection_scores=candidate_scores,
        )
        _append_iteration_result(
            iteration_record=decorated,
            subset_label=((int(winner_pair[0]), int(winner_pair[1])),),
            iteration_records=iteration_records,
            k_history=k_history,
            energies=energies,
            subset_labels=subset_label_history,
        )

    details = _build_run_details(
        system,
        method_variant=method_variant,
        k_source=k_source,
        requested_k_source=requested_k_source,
        initial_k_params=initial_k_params,
        subset_labels=subset_label_history,
        selected_projector_order=selected_projector_order,
        selection_mode=selection_mode,
        iteration_records=iteration_records,
        frozen_k_params=frozen_k_params,
        initial_reference_record=initial_reference_record,
        np=np,
    )
    return k_history, _serialize_subset_labels(subset_label_history), energies, details


def _run_default_frozen_k_variant(
    system: _SystemData,
    *,
    initial_k_params,
    k_source: str,
    requested_k_source: str,
    selection_mode: str,
    subset_rank_max: int,
    overlap_tol: float,
    regularization: float,
    include_iteration_matrices: bool,
):
    """Run the student-facing default algorithm 1 path."""
    if int(subset_rank_max) == 0:
        return _run_subset_rank_zero_variant(
            system,
            method_variant="frozen_k",
            initial_k_params=initial_k_params,
            k_source=k_source,
            requested_k_source=requested_k_source,
            selection_mode=selection_mode,
            subset_rank_max=subset_rank_max,
            overlap_tol=overlap_tol,
            regularization=regularization,
            outer_opt_maxiter=1,
            alternating_window=0.0,
            alternating_opt_maxiter=1,
            include_iteration_matrices=include_iteration_matrices,
        )
    return _run_adaptive_variant(
        system,
        method_variant="frozen_k",
        initial_k_params=initial_k_params,
        k_source=k_source,
        requested_k_source=requested_k_source,
        selection_mode=selection_mode,
        subset_rank_max=subset_rank_max,
        overlap_tol=overlap_tol,
        regularization=regularization,
        outer_opt_maxiter=1,
        alternating_window=0.0,
        alternating_opt_maxiter=1,
        include_iteration_matrices=include_iteration_matrices,
    )


def _run_legacy_variant(
    system: _SystemData,
    *,
    method_variant: str,
    initial_k_params,
    k_source: str,
    requested_k_source: str,
    selection_mode: str,
    subset_rank_max: int,
    overlap_tol: float,
    regularization: float,
    outer_opt_maxiter: int,
    alternating_window: float,
    alternating_opt_maxiter: int,
    include_iteration_matrices: bool,
):
    """Run any legacy control path while keeping behavior unchanged."""
    if int(subset_rank_max) == 0:
        return _run_subset_rank_zero_variant(
            system,
            method_variant=method_variant,
            initial_k_params=initial_k_params,
            k_source=k_source,
            requested_k_source=requested_k_source,
            selection_mode=selection_mode,
            subset_rank_max=subset_rank_max,
            overlap_tol=overlap_tol,
            regularization=regularization,
            outer_opt_maxiter=outer_opt_maxiter,
            alternating_window=alternating_window,
            alternating_opt_maxiter=alternating_opt_maxiter,
            include_iteration_matrices=include_iteration_matrices,
        )
    if selection_mode == "graded_lex":
        return _run_graded_lex_variant(
            system,
            method_variant=method_variant,
            initial_k_params=initial_k_params,
            k_source=k_source,
            requested_k_source=requested_k_source,
            selection_mode=selection_mode,
            overlap_tol=overlap_tol,
            regularization=regularization,
            outer_opt_maxiter=outer_opt_maxiter,
            alternating_window=alternating_window,
            alternating_opt_maxiter=alternating_opt_maxiter,
            include_iteration_matrices=include_iteration_matrices,
        )
    return _run_adaptive_variant(
        system,
        method_variant=method_variant,
        initial_k_params=initial_k_params,
        k_source=k_source,
        requested_k_source=requested_k_source,
        selection_mode=selection_mode,
        subset_rank_max=subset_rank_max,
        overlap_tol=overlap_tol,
        regularization=regularization,
        outer_opt_maxiter=outer_opt_maxiter,
        alternating_window=alternating_window,
        alternating_opt_maxiter=alternating_opt_maxiter,
        include_iteration_matrices=include_iteration_matrices,
    )


def _run_variant(
    system: _SystemData,
    *,
    method_variant: str,
    initial_k_params,
    k_source: str,
    requested_k_source: str,
    selection_mode: str,
    subset_rank_max: int,
    overlap_tol: float,
    regularization: float,
    outer_opt_maxiter: int,
    alternating_window: float,
    alternating_opt_maxiter: int,
    include_iteration_matrices: bool,
):
    """Dispatch to the default path or one of the legacy control paths."""
    if method_variant == "frozen_k" and selection_mode == "adaptive_commutator":
        return _run_default_frozen_k_variant(
            system,
            initial_k_params=initial_k_params,
            k_source=k_source,
            requested_k_source=requested_k_source,
            selection_mode=selection_mode,
            subset_rank_max=subset_rank_max,
            overlap_tol=overlap_tol,
            regularization=regularization,
            include_iteration_matrices=include_iteration_matrices,
        )
    return _run_legacy_variant(
        system,
        method_variant=method_variant,
        initial_k_params=initial_k_params,
        k_source=k_source,
        requested_k_source=requested_k_source,
        selection_mode=selection_mode,
        subset_rank_max=subset_rank_max,
        overlap_tol=overlap_tol,
        regularization=regularization,
        outer_opt_maxiter=outer_opt_maxiter,
        alternating_window=alternating_window,
        alternating_opt_maxiter=alternating_opt_maxiter,
        include_iteration_matrices=include_iteration_matrices,
    )


def gucj_gcim(
    symbols: Sequence[str],
    geometry,
    *,
    method_variant: str = "frozen_K",
    basis: str = "sto-3g",
    charge: int = 0,
    spin: int = 0,
    active_electrons: int,
    active_orbitals: int,
    subset_rank_max: int = 2,
    kappa: float = math.pi / 4.0,
    k_params: Optional[Sequence[float]] = None,
    k_source: Optional[str] = "heuristic",
    k_param_shape: Optional[str] = None,
    selection_mode: str = "adaptive_commutator",
    overlap_tol: float = 1e-10,
    regularization: float = 1e-10,
    hamiltonian_cutoff: float = 1e-20,
    outer_opt_maxiter: int = 25,
    alternating_window: float = math.pi / 8.0,
    alternating_opt_maxiter: int = 10,
    num_threads: Optional[int] = None,
    include_iteration_matrices: bool = False,
    return_details: bool = False,
):
    """Run the exact v1 g-uCJ-GRIM/GCIM algorithm for small active spaces.

    Parameters
    ----------
    symbols
        Atomic symbols, e.g. ``["H", "H"]``.
    geometry
        Nuclear coordinates with shape ``(n_atoms, 3)`` in Angstrom.
    method_variant
        One of ``"frozen_K"``, ``"optimized_K_outer_loop"``, or
        ``"alternating_K_and_subspace"``. ``"frozen_K"`` is the production
        default. The other two variants are retained only as legacy research
        controls.
    basis
        Basis set name understood by PySCF.
    charge
        Total molecular charge.
    spin
        Spin parameter used by PySCF as ``2S``.
    active_electrons
        Number of active electrons.
    active_orbitals
        Number of active orbitals.
    subset_rank_max
        Maximum projector-subset rank. This v1 implementation supports only
        ranks ``0``, ``1``, and ``2``.
    kappa
        Scalar seed for the K-parameter initialization. In vector mode this
        scales a deterministic non-uniform generalized-K parameter ramp.
    k_params
        Optional explicit K-parameter vector. If omitted, one is built from
        ``kappa`` and ``k_param_shape``.
    k_source
        Frozen-K source selector. ``"heuristic"`` is the production default and
        uses the deterministic generalized-K initialization seeded by
        ``kappa``. ``"ccsd_t1"`` maps active-space RHF-CCSD singles onto the
        same-spin OV K slots for the explicit occupied-virtual control path.
        ``"manual"`` requires explicit ``k_params``.
    k_param_shape
        K-parameterization strategy. If omitted, ``"frozen_K"`` defaults to
        ``"generalized_same_spin_vector"`` while the optimized and alternating
        variants keep the occupied-virtual control path
        ``"ov_same_spin_vector"``. ``"uniform_ov_same_spin"`` keeps the old
        scalar control.
    selection_mode
        Basis-growth strategy. ``"adaptive_commutator"`` selects one new
        elementary projector per iteration using the rotated-frame commutator
        score. ``"graded_lex"`` keeps the legacy fixed graded-lexicographic
        subset-growth path for debugging and regression.
    overlap_tol
        Overlap-eigenvalue truncation threshold for canonical orthogonalization.
    regularization
        Numerical PSD tolerance used when screening small negative overlap
        eigenvalues due to floating-point noise.
    hamiltonian_cutoff
        Drop Hamiltonian terms below this absolute coefficient threshold.
    outer_opt_maxiter
        Maximum optimizer iterations for the ``"optimized_K_outer_loop"``
        variant.
    alternating_window
        Per-parameter local update window for ``"alternating_K_and_subspace"``.
    alternating_opt_maxiter
        Maximum optimizer iterations for the alternating local K update.
    num_threads
        Optional BLAS/LAPACK thread limit used by NumPy/SciPy kernels such as
        ``expm`` and ``eigh``. This is the meaningful multi-core control for
        the default frozen-K path; the GCIM operator-selection loop itself is
        sequential.
    include_iteration_matrices
        If True, store dense projected matrices in each iteration record.
        Leaving this False keeps the stretched-H4 benchmark memory-safe.
    return_details
        If True, also return a details dictionary containing per-iteration
        diagnostics, overlap data, and the exact FCI reference energy.

    Returns
    -------
    tuple
        ``(k_history, subset_labels, energies)`` where ``k_history`` contains
        either scalar values (uniform control mode) or full parameter vectors.
    tuple
        If ``return_details=True``, returns
        ``(k_history, subset_labels, energies, details)``.
    """
    _validate_inputs(
        symbols=symbols,
        geometry=geometry,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
        subset_rank_max=subset_rank_max,
        overlap_tol=overlap_tol,
        regularization=regularization,
        outer_opt_maxiter=outer_opt_maxiter,
        alternating_opt_maxiter=alternating_opt_maxiter,
    )

    method_variant_normalized = _normalize_method_variant(method_variant)
    if k_param_shape is None:
        k_param_shape_normalized = _default_k_param_shape_for_method(method_variant_normalized)
    else:
        k_param_shape_normalized = _normalize_k_param_shape(k_param_shape)
    k_source_normalized = _normalize_k_source("heuristic" if k_source is None else k_source)
    selection_mode_normalized = _normalize_selection_mode(selection_mode)

    # The public function reads top-to-bottom:
    # normalize user options -> build the molecular problem -> resolve K -> run.
    with _threadpool_limit_context(num_threads):
        system = _build_system_data(
            symbols=symbols,
            geometry=geometry,
            basis=basis,
            charge=charge,
            spin=spin,
            active_electrons=active_electrons,
            active_orbitals=active_orbitals,
            hamiltonian_cutoff=hamiltonian_cutoff,
            subset_rank_max=subset_rank_max,
            k_param_shape=k_param_shape_normalized,
            compute_ccsd_t1=_should_compute_ccsd_t1(
                method_variant=method_variant_normalized,
                k_params=k_params,
                k_source=k_source_normalized,
                k_param_shape=k_param_shape_normalized,
            ),
        )

        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "gucj_gcim requires NumPy. Install at least: `pip install numpy`."
            ) from exc

        initial_k_params, effective_k_source, requested_k_source = _resolve_k_params_and_source(
            system,
            method_variant=method_variant_normalized,
            kappa=float(kappa),
            k_params=k_params,
            k_source=k_source_normalized,
            np=np,
        )

        k_history, subset_labels, energies, details = _run_variant(
            system,
            method_variant=method_variant_normalized,
            initial_k_params=np.asarray(initial_k_params, dtype=float),
            k_source=str(effective_k_source),
            requested_k_source=str(requested_k_source),
            selection_mode=selection_mode_normalized,
            subset_rank_max=int(subset_rank_max),
            overlap_tol=float(overlap_tol),
            regularization=float(regularization),
            outer_opt_maxiter=int(outer_opt_maxiter),
            alternating_window=float(alternating_window),
            alternating_opt_maxiter=int(alternating_opt_maxiter),
            include_iteration_matrices=bool(include_iteration_matrices),
        )
    details["num_threads"] = None if num_threads is None else int(num_threads)

    if not return_details:
        return k_history, subset_labels, energies
    return k_history, subset_labels, energies, details
