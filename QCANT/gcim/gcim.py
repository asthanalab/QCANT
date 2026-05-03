"""GCIM implementation with an ADAPT-like public API.

This module promotes the legacy GCIM script into a reusable package function:
``QCANT.gcim``.

The implementation follows the same molecular Hamiltonian construction flow used
in other QCANT algorithms (PySCF CASCI -> PennyLane qubit Hamiltonian), then:

1. Builds a singles+doubles anti-Hermitian operator pool.
2. Selects one operator per iteration via largest commutator magnitude.
3. Grows a non-orthogonal state basis from individual and cumulative states.
4. Solves a generalized eigenvalue problem in that basis to estimate the ground
   energy at each iteration.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

from .._accelerator import build_qml_device


def _validate_inputs(
    symbols: Sequence[str],
    geometry,
    adapt_it: int,
    shots: Optional[int],
    hamiltonian_cutoff: float,
    pool_sample_size: Optional[int],
    theta: float,
    regularization: float,
    overlap_tol: float,
    pool_type: str,
) -> int:
    """Validate GCIM user inputs.

    Returns
    -------
    int
        Number of atoms.
    """
    if len(symbols) == 0:
        raise ValueError("symbols must be non-empty")
    if adapt_it < 0:
        raise ValueError("adapt_it must be >= 0")
    if shots is not None and shots != 0:
        raise ValueError("gcim currently requires analytic execution; set shots=None or 0")
    if hamiltonian_cutoff < 0:
        raise ValueError("hamiltonian_cutoff must be >= 0")
    if pool_sample_size is not None and pool_sample_size <= 0:
        raise ValueError("pool_sample_size must be > 0")
    if theta <= 0:
        raise ValueError("theta must be > 0")
    if regularization < 0:
        raise ValueError("regularization must be >= 0")
    if overlap_tol <= 0:
        raise ValueError("overlap_tol must be > 0")
    if pool_type not in {"sd", "singlet_sd", "gsd"}:
        raise ValueError("pool_type must be one of {'sd', 'singlet_sd', 'gsd'}")

    try:
        n_atoms = len(symbols)
        if len(geometry) != n_atoms:
            raise ValueError
    except Exception as exc:
        raise ValueError("geometry must have the same length as symbols") from exc

    return n_atoms


def gcim(
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
    device_kwargs: Optional[Mapping[str, object]] = None,
    shots: Optional[int] = None,
    hamiltonian_cutoff: float = 1e-20,
    pool_sample_size: Optional[int] = None,
    pool_seed: Optional[int] = None,
    pool_type: str = "sd",
    theta: float = 0.7853981633974483,  # pi/4
    regularization: float = 1e-10,
    overlap_tol: float = 1e-12,
    allow_repeated_operators: bool = True,
    print_progress: bool = True,
    return_details: bool = False,
):
    """Run a GCIM loop with ADAPT-style inputs and outputs.

    Parameters
    ----------
    symbols
        Atomic symbols, e.g. ``["H", "H"]``.
    geometry
        Nuclear coordinates, array-like with shape ``(n_atoms, 3)`` in Angstrom.
    adapt_it
        Kept for API compatibility. GCIM now hardcodes the number of iterations
        to the total number of operators in the selected pool.
    basis
        Basis set name understood by PySCF.
    charge
        Total molecular charge.
    spin
        Spin parameter used by PySCF as ``2S`` (e.g. 0 for singlet).
    active_electrons
        Number of active electrons in the CASCI reference.
    active_orbitals
        Number of active orbitals in the CASCI reference.
    device_name
        PennyLane device name, defaulting to ``lightning.qubit`` then
        ``default.qubit``.
    device_kwargs
        Optional keyword arguments forwarded to ``qml.device``.
    shots
        Present for API compatibility with :func:`QCANT.adapt_vqe`. GCIM needs
        analytic statevector execution, so only ``None`` or ``0`` is accepted.
    hamiltonian_cutoff
        Drop Hamiltonian terms below this absolute coefficient threshold.
    pool_sample_size
        If provided, sample this many operators from the pool per iteration.
    pool_seed
        RNG seed for pool sampling.
    pool_type
        Operator-pool construction strategy:

        - ``"sd"``: reference-based singles+doubles pool,
        - ``"singlet_sd"``: singlet-adapted paired singles+doubles pool
          (closed-shell active spaces), and
        - ``"gsd"``: generalized spin-conserving singles+doubles pool.
    theta
        Fixed generator angle used for each selected operator.
    regularization
        Diagonal Tikhonov regularization added to overlap matrix ``S`` before
        solving the generalized eigenproblem.
    overlap_tol
        Threshold for overlap-matrix eigenvalue filtering in canonical
        orthogonalization.
    allow_repeated_operators
        Kept for API compatibility. GCIM now always enforces unique operator
        selection, so repeated operators are not used.
    print_progress
        If True, print iteration/energy updates.
    return_details
        If True, return an additional dictionary with debug artifacts
        (state basis, selected indices, projected matrices history).

    Returns
    -------
    tuple
        ``(params, ash_excitation, energies)`` where:

        - ``params`` is a vector of selected angles (all equal to ``theta``),
        - ``ash_excitation`` is the selected excitation list (same style as
          ADAPT-VQE), and
        - ``energies`` is the per-iteration GCIM minimum energy.

        If ``return_details=True``, returns
        ``(params, ash_excitation, energies, details)``.
    """
    # pylint: disable=too-many-locals, too-many-statements, too-many-branches, too-many-arguments
    n_atoms = _validate_inputs(
        symbols=symbols,
        geometry=geometry,
        adapt_it=adapt_it,
        shots=shots,
        hamiltonian_cutoff=hamiltonian_cutoff,
        pool_sample_size=pool_sample_size,
        theta=theta,
        regularization=regularization,
        overlap_tol=overlap_tol,
        pool_type=pool_type,
    )

    try:
        import numpy as np
        import pennylane as qml
        import pyscf
        from itertools import combinations

        from pennylane.fermi import from_string
        from pyscf import gto, mcscf, scf
        from scipy.linalg import eigh
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "gcim requires dependencies. Install at least: "
            "`pip install numpy scipy pennylane pyscf`."
        ) from exc

    def _make_device(name: Optional[str], wires: int):
        return build_qml_device(
            qml,
            device_name=name,
            wires=wires,
            device_kwargs=device_kwargs,
            shots=None,
        )

    # ------------------------------------------------------------------
    # 1) Molecule and active-space Hamiltonian construction.
    # ------------------------------------------------------------------
    atom = [(symbols[i], tuple(float(x) for x in geometry[i])) for i in range(n_atoms)]
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

    if print_progress:
        en = mycas_ref.kernel()
        print("Ref.CASCI energy:", en[0], flush=True)

    ncas = int(mycas_ref.ncas)
    two_mo = pyscf.ao2mo.restore("1", h2ecas, norb=ncas)
    two_mo = np.swapaxes(two_mo, 1, 3)
    one_mo = h1ecas
    core_constant = np.array([ecore])
    h_fermionic = qml.qchem.fermionic_observable(
        core_constant, one_mo, two_mo, cutoff=hamiltonian_cutoff
    )
    h_qubit = qml.jordan_wigner(h_fermionic)

    # Guard against tiny imaginary coefficients from numerical conversion noise.
    if hasattr(h_qubit, "terms"):
        coeffs, ops = h_qubit.terms()
    else:
        coeffs, ops = getattr(h_qubit, "coeffs", []), getattr(h_qubit, "ops", [])
    coeffs = np.asarray(coeffs, dtype=complex)
    if coeffs.size > 0 and (np.any(np.abs(coeffs.imag) > 1e-12) or coeffs.dtype.kind == "c"):
        h_qubit = qml.Hamiltonian(coeffs.real.astype(float), ops)

    qubits = 2 * ncas
    wires = tuple(range(qubits))
    h_matrix = np.asarray(qml.matrix(h_qubit, wire_order=wires), dtype=complex)

    # ------------------------------------------------------------------
    # 2) Operator pool construction.
    # ------------------------------------------------------------------
    n_active_electrons = sum(mycas_ref.nelecas)

    def _build_sd_pool():
        singles, doubles = qml.qchem.excitations(n_active_electrons, qubits)
        labels = [list(ex) for ex in singles] + [list(ex) for ex in doubles]

        pool_fermi_local = []
        for ex in singles:
            pool_fermi_local.append(
                from_string(f"{ex[1]}+ {ex[0]}-") - from_string(f"{ex[0]}+ {ex[1]}-")
            )
        for ex in doubles:
            pool_fermi_local.append(
                from_string(f"{ex[3]}+ {ex[2]}+ {ex[1]}- {ex[0]}-")
                - from_string(f"{ex[0]}+ {ex[1]}+ {ex[2]}- {ex[3]}-")
            )
        return labels, pool_fermi_local

    def _build_singlet_sd_pool():
        n_alpha, n_beta = mycas_ref.nelecas
        if n_alpha != n_beta:
            raise ValueError("pool_type='singlet_sd' requires a closed-shell active space")

        n_occ_spatial = int(n_alpha)
        n_spatial = int(active_orbitals)
        occ = range(n_occ_spatial)
        virt = range(n_occ_spatial, n_spatial)

        labels = []
        pool_fermi_local = []

        # Singlet-adapted paired singles:
        # (a†_{a,alpha} a_{i,alpha} + a†_{a,beta} a_{i,beta}) - h.c.
        for i in occ:
            for a in virt:
                ia = 2 * i
                ib = 2 * i + 1
                aa = 2 * a
                ab = 2 * a + 1
                op = (
                    from_string(f"{aa}+ {ia}-")
                    + from_string(f"{ab}+ {ib}-")
                    - from_string(f"{ia}+ {aa}-")
                    - from_string(f"{ib}+ {ab}-")
                )
                labels.append([ia, aa, ib, ab])
                pool_fermi_local.append(op)

        # Singlet-adapted paired doubles:
        # a†_{a,alpha} a†_{a,beta} a_{i,beta} a_{i,alpha} - h.c.
        for i in occ:
            for a in virt:
                ia = 2 * i
                ib = 2 * i + 1
                aa = 2 * a
                ab = 2 * a + 1
                op = (
                    from_string(f"{ab}+ {aa}+ {ib}- {ia}-")
                    - from_string(f"{ia}+ {ib}+ {aa}- {ab}-")
                )
                labels.append([ia, ib, aa, ab])
                pool_fermi_local.append(op)

        return labels, pool_fermi_local

    def _build_gsd_pool():
        def spin_val(idx: int) -> int:
            # qml.qchem uses interleaved spin ordering in this codepath:
            # even -> alpha (+1), odd -> beta (-1).
            return 1 if (idx % 2 == 0) else -1

        labels = []
        pool_fermi_local = []

        # Generalized spin-conserving singles.
        for q in range(qubits):
            for p in range(q + 1, qubits):
                if spin_val(p) != spin_val(q):
                    continue
                op = from_string(f"{p}+ {q}-") - from_string(f"{q}+ {p}-")
                labels.append([q, p])
                pool_fermi_local.append(op)

        # Generalized spin-conserving doubles (disjoint indices only).
        pair_list = list(combinations(range(qubits), 2))
        for (r, s), (p, q) in combinations(pair_list, 2):
            if len({r, s, p, q}) < 4:
                continue
            if (spin_val(p) + spin_val(q)) != (spin_val(r) + spin_val(s)):
                continue
            op = (
                from_string(f"{q}+ {p}+ {s}- {r}-")
                - from_string(f"{r}+ {s}+ {p}- {q}-")
            )
            labels.append([r, s, p, q])
            pool_fermi_local.append(op)

        return labels, pool_fermi_local

    if pool_type == "sd":
        pool_excitations, pool_fermi = _build_sd_pool()
    elif pool_type == "singlet_sd":
        pool_excitations, pool_fermi = _build_singlet_sd_pool()
    else:
        pool_excitations, pool_fermi = _build_gsd_pool()

    pool_ops = [qml.jordan_wigner(op) for op in pool_fermi]

    if len(pool_ops) == 0:
        raise ValueError("operator pool is empty for the provided active space")
    fixed_iterations = int(len(pool_ops))

    # ------------------------------------------------------------------
    # 3) Device circuits for state prep and commutators.
    # ------------------------------------------------------------------
    dev = _make_device(device_name, qubits)
    hf_occ = qml.qchem.hf_state(n_active_electrons, qubits)

    @qml.qnode(dev)
    def _hf_statevector():
        qml.BasisState(hf_occ, wires=wires)
        return qml.state()

    @qml.qnode(dev)
    def _ansatz_state(selected_indices):
        qml.BasisState(hf_occ, wires=wires)
        for idx in selected_indices:
            qml.exp(pool_ops[int(idx)] * (float(theta) / 2.0))
        return qml.state()

    @qml.qnode(dev)
    def _single_state(idx):
        qml.BasisState(hf_occ, wires=wires)
        qml.exp(pool_ops[int(idx)] * (float(theta) / 2.0))
        return qml.state()

    @qml.qnode(dev)
    def _commutator_expectation(state, op_idx):
        qml.StatePrep(state, wires=wires)
        comm = qml.commutator(h_qubit, pool_ops[int(op_idx)])
        return qml.expval(comm)

    def _project_min_energy(state_basis):
        basis_array = np.asarray(state_basis, dtype=complex)
        s_mat = basis_array.conj() @ basis_array.T
        m_mat = basis_array.conj() @ (h_matrix @ basis_array.T)

        # Canonical orthogonalization improves stability before solving M c = S c E.
        s_vals, s_vecs = np.linalg.eigh((s_mat + s_mat.conj().T) / 2.0)
        keep = s_vals > float(overlap_tol)
        if not np.any(keep):
            raise ValueError("overlap matrix is numerically singular; basis collapsed")

        x_mat = s_vecs[:, keep] / np.sqrt(s_vals[keep])[None, :]
        h_ortho = x_mat.conj().T @ m_mat @ x_mat
        evals = np.linalg.eigvalsh((h_ortho + h_ortho.conj().T) / 2.0).real

        if regularization > 0.0:
            n = s_mat.shape[0]
            evals_reg = eigh(
                (m_mat + m_mat.conj().T) / 2.0,
                (s_mat + regularization * np.eye(n)).real,
                eigvals_only=True,
            )
            return float(np.real(evals_reg[0])), m_mat, s_mat
        return float(np.real(evals[0])), m_mat, s_mat

    # ------------------------------------------------------------------
    # 4) GCIM loop.
    # ------------------------------------------------------------------
    rng = np.random.default_rng(pool_seed)
    selected_indices = []
    selected_excitations = []
    energies = []
    states = []
    details_states = []
    details_m = []
    details_s = []

    hf_state = _hf_statevector()
    hf_state = hf_state / np.linalg.norm(hf_state)
    states.append(hf_state)

    if print_progress and int(adapt_it) != fixed_iterations:
        print(
            f"GCIM iterations are hardcoded to pool size={fixed_iterations}; "
            f"ignoring adapt_it={int(adapt_it)}.",
            flush=True,
        )
    if print_progress and bool(allow_repeated_operators):
        print(
            "GCIM enforces unique operator selection; "
            "allow_repeated_operators is ignored.",
            flush=True,
        )

    for it in range(fixed_iterations):
        if print_progress:
            print("The adapt iteration now is", it, flush=True)

        selected_set = set(int(i) for i in selected_indices)
        remaining = np.array(
            [idx for idx in range(len(pool_ops)) if idx not in selected_set], dtype=int
        )
        if remaining.size == 0:
            break

        # Candidate set (all remaining or sampled from remaining).
        if pool_sample_size is None or pool_sample_size >= remaining.size:
            candidates = remaining
        else:
            candidates = rng.choice(remaining, size=pool_sample_size, replace=False)

        current_state = states[-1]
        max_idx = None
        max_val = float("-inf")

        for op_idx in candidates:
            comm_val = _commutator_expectation(current_state, int(op_idx))
            cur = abs(2.0 * float(np.real_if_close(comm_val)))
            if cur > max_val:
                max_val = cur
                max_idx = int(op_idx)

        if max_idx is None:
            if print_progress:
                print("No selectable operator found; stopping early.", flush=True)
            break

        selected_indices.append(max_idx)
        selected_excitations.append(pool_excitations[max_idx])

        cumulative_state = _ansatz_state(tuple(selected_indices))
        cumulative_state = cumulative_state / np.linalg.norm(cumulative_state)

        # Match legacy growth pattern:
        # - first iteration: append cumulative state only
        # - subsequent iterations: append single(new op) then cumulative state
        if it >= 1:
            ind_state = _single_state(max_idx)
            ind_state = ind_state / np.linalg.norm(ind_state)
            states.append(ind_state)
        states.append(cumulative_state)

        e0, m_mat, s_mat = _project_min_energy(states)
        energies.append(float(e0))

        if return_details:
            details_states.append(np.asarray(states, dtype=complex))
            details_m.append(np.asarray(m_mat, dtype=complex))
            details_s.append(np.asarray(s_mat, dtype=complex))

        if print_progress:
            print("Energies are", energies, flush=True)

    params = np.full(len(selected_excitations), float(theta), dtype=float)
    if len(energies) > 0 and print_progress:
        print("energies:", energies[-1], flush=True)

    if return_details:
        details = {
            "selected_operator_indices": np.asarray(selected_indices, dtype=int),
            "state_basis_history": details_states,
            "projected_hamiltonians": details_m,
            "overlap_matrices": details_s,
            "hf_state": np.asarray(hf_state, dtype=complex),
        }
        return params, selected_excitations, energies, details

    return params, selected_excitations, energies
