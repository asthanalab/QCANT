"""Quantum Krylov using Unitary Decomposition (QKUD).

This module builds a Krylov basis using the QKUD recurrence
|psi_n> = (X + X^†) / (2 * epsilon) |psi_{n-1}| with
X = i * exp(-i * epsilon * H).
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple


def _validate_inputs(
    symbols: Sequence[str],
    geometry: "object",
    n_steps: int,
    epsilon: float,
    overlap_tol: float,
) -> int:
    """Validate user-provided inputs for the QKUD algorithm.

    Returns
    -------
    int
        The number of atoms.

    Raises
    ------
    ValueError
        If any of the input parameters are invalid.
    """
    if len(symbols) == 0:
        raise ValueError("symbols must be non-empty")
    if n_steps < 0:
        raise ValueError("n_steps must be >= 0")
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    if overlap_tol <= 0:
        raise ValueError("overlap_tol must be > 0")

    try:
        n_atoms = len(symbols)
        if len(geometry) != n_atoms:
            raise ValueError
    except Exception as exc:
        raise ValueError("geometry must have the same length as symbols") from exc

    return n_atoms


def qkud(
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
    method: str = "pyscf",
    device_name: Optional[str] = None,
    initial_state: Optional["object"] = None,
    overlap_tol: float = 1e-10,
    normalize_basis: bool = True,
    basis_threshold: float = 0.0,
    use_sparse: bool = False,
    return_min_energy_history: bool = False,
) -> Tuple["object", "object"] | Tuple["object", "object", "object"]:
    """Generate a QKUD Krylov basis and diagonalize the Hamiltonian in that basis.

    The basis is built using the recurrence in Eq. (3) of the QKUD formulation:
    |psi_n> = (X + X^†) / (2 * epsilon) |psi_{n-1}| with
    X = i * exp(-i * epsilon * H).

    Parameters
    ----------
    symbols
        Atomic symbols.
    geometry
        Nuclear coordinates, array-like with shape ``(n_atoms, 3)`` in Angstrom.
    n_steps
        Number of QKUD steps. The returned basis contains ``n_steps + 1``
        vectors (including the initial state).
    epsilon
        Error parameter used in the QKUD recurrence (must be > 0).
    active_electrons
        Number of active electrons.
    active_orbitals
        Number of active orbitals.
    basis
        Basis set name understood by PennyLane/PySCF (e.g. ``"sto-3g"``).
    charge
        Total molecular charge.
    spin
        Spin parameter used by PySCF as ``2S`` (e.g. 0 for singlet).
    method
        Backend used by PennyLane quantum chemistry tooling (default: ``"pyscf"``).
    device_name
        Unused; present for API compatibility with other algorithms.
    initial_state
        Optional statevector to seed the Krylov basis. If not provided, the
        Hartree–Fock state is used.
    overlap_tol
        Threshold for discarding near-linearly dependent basis vectors when
        orthonormalizing the basis via the overlap matrix eigen-decomposition.
    normalize_basis
        If True, normalize each Krylov vector to avoid numerical overflow.
    basis_threshold
        Drop amplitudes with absolute value below this threshold after each
        basis update. The thresholded state is re-normalized. Use 0.0 to
        disable thresholding.
    use_sparse
        If True, use a sparse Hamiltonian representation for state updates.
    return_min_energy_history
        If True, also return the minimum energy after each QKUD step.

    Notes
    -----
    This implementation enforces a real-valued Hamiltonian by dropping tiny
    imaginary parts in the coefficients. This keeps simulator backends like
    lightning.qubit stable when numerical noise introduces complex terms.

    Returns
    -------
    tuple
        ``(energies, basis_states)`` where:

        - ``energies`` is a real-valued array of eigenvalues obtained by
          diagonalizing the Hamiltonian projected into the generated basis.
        - ``basis_states`` is a complex-valued array with shape
          ``(n_steps+1, 2**n_qubits)``.

        If ``return_min_energy_history=True``, the function returns
        ``(energies, basis_states, min_energy_history)`` where
        ``min_energy_history`` has shape ``(n_steps,)`` and contains the
        minimum energy after each step (using the basis with ``k+1`` vectors).
    """
    # pylint: disable=too-many-locals, too-many-statements, too-many-branches, too-many-arguments, unused-argument

    # --------------------------------------------------------------------------
    # 1. Input validation and dependency imports.
    # --------------------------------------------------------------------------
    n_atoms = _validate_inputs(symbols, geometry, n_steps, epsilon, overlap_tol)

    try:
        import numpy as np
        import pennylane as qml
        import pyscf
        from pyscf import gto, mcscf, scf
        from scipy.sparse.linalg import expm_multiply
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "qkud requires dependencies. Install at least: "
            "`pip install numpy pennylane pyscf scipy`."
        ) from exc

    def _basis_state_vector(bits: Sequence[int]) -> "object":
        """Convert a bit string to a state vector."""
        idx = 0
        for bit in bits:
            idx = (idx << 1) | int(bit)
        state = np.zeros(2 ** len(bits), dtype=complex)
        state[idx] = 1.0
        return state

    # --------------------------------------------------------------------------
    # 2. Molecule and Hamiltonian setup.
    # --------------------------------------------------------------------------
    atom = [(symbols[i], tuple(float(x) for x in geometry[i])) for i in range(n_atoms)]
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

    mycas = mcscf.CASCI(mf, active_orbitals, active_electrons)
    h1ecas, ecore = mycas.get_h1eff(mf.mo_coeff)
    h2ecas = mycas.get_h2eff(mf.mo_coeff)

    two_mo = pyscf.ao2mo.restore("1", h2ecas, norb=mycas.ncas)
    two_mo = np.swapaxes(two_mo, 1, 3)
    one_mo = h1ecas
    core_constant = np.array([ecore])

    H_fermionic = qml.qchem.fermionic_observable(core_constant, one_mo, two_mo, cutoff=1e-20)
    H = qml.jordan_wigner(H_fermionic)
    n_qubits = 2 * mycas.ncas

    if hasattr(H, "terms"):
        coeffs, ops = H.terms()
    else:
        coeffs, ops = getattr(H, "coeffs", []), getattr(H, "ops", [])
    coeffs = np.asarray(coeffs, dtype=complex)
    if coeffs.size > 0 and (np.any(np.abs(coeffs.imag) > 1e-12) or coeffs.dtype.kind == "c"):
        H = qml.Hamiltonian(coeffs.real.astype(float), ops)

    wires = range(n_qubits)

    # --------------------------------------------------------------------------
    # 3. Initial state preparation.
    # --------------------------------------------------------------------------
    if initial_state is None:
        hf_occ = qml.qchem.hf_state(active_electrons, n_qubits)
        psi = _basis_state_vector(hf_occ)
    else:
        psi = np.asarray(initial_state, dtype=complex)
        if psi.ndim != 1:
            raise ValueError("initial_state must be a 1D statevector")
        expected_dim = 2**n_qubits
        if psi.size != expected_dim:
            raise ValueError(
                f"initial_state must have length {expected_dim}, got {psi.size}"
            )

    psi_norm = np.linalg.norm(psi)
    if psi_norm == 0:
        raise ValueError("initial_state has zero norm")
    psi = psi / psi_norm

    def _apply_basis_threshold(state):
        if basis_threshold <= 0:
            return state
        state = np.asarray(state, dtype=complex)
        mask = np.abs(state) >= basis_threshold
        if not np.any(mask):
            idx = int(np.argmax(np.abs(state)))
            mask[idx] = True
        state = np.where(mask, state, 0.0)
        norm = np.linalg.norm(state)
        if norm == 0:
            raise ValueError("thresholded basis vector has zero norm")
        return state / norm

    psi = _apply_basis_threshold(psi)

    # --------------------------------------------------------------------------
    # 4. QKUD basis construction.
    # --------------------------------------------------------------------------
    if use_sparse:
        try:
            import scipy.sparse  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError("use_sparse=True requires scipy") from exc
        if hasattr(H, "sparse_matrix") and getattr(H, "has_sparse_matrix", True):
            H_mat = H.sparse_matrix(wire_order=wires, format="csr")
        else:
            H_mat = qml.matrix(H, wire_order=wires)
    else:
        H_mat = qml.matrix(H, wire_order=wires)

    def _apply_qkud(state):
        """Apply the QKUD recurrence relation."""
        forward = expm_multiply(-1j * epsilon * H_mat, state)
        backward = expm_multiply(1j * epsilon * H_mat, state)
        return (1j * (forward - backward)) / (2.0 * epsilon)

    def _project_min_energy(current_basis_states):
        """Project and diagonalize the Hamiltonian, returning the minimum energy."""
        S = current_basis_states.conj() @ current_basis_states.T
        H_proj = current_basis_states.conj() @ (H_mat @ current_basis_states.T)

        s_vals, s_vecs = np.linalg.eigh(S)
        keep = s_vals > float(overlap_tol)
        if not keep.any():
            raise ValueError("overlap matrix is numerically singular; basis collapsed")

        X = s_vecs[:, keep] / np.sqrt(s_vals[keep])[None, :]
        H_ortho = X.conj().T @ H_proj @ X
        evals = np.linalg.eigvalsh(H_ortho).real
        return evals, float(evals[0])

    basis_states = [psi]
    current = psi
    for _ in range(n_steps):
        current = _apply_qkud(current)
        if normalize_basis:
            current_norm = np.linalg.norm(current)
            if current_norm == 0:
                raise ValueError("QKUD vector has zero norm")
            current = current / current_norm
        current = _apply_basis_threshold(current)
        basis_states.append(current)

    # --------------------------------------------------------------------------
    # 5. Diagonalize the Hamiltonian in the Krylov basis.
    # --------------------------------------------------------------------------
    basis_states = np.stack(basis_states, axis=0)
    energies, _e0 = _project_min_energy(basis_states)

    if return_min_energy_history:
        min_energy_history = []
        num_steps = basis_states.shape[0] - 1
        for k in range(1, num_steps + 1):
            _evals, e0 = _project_min_energy(basis_states[: k + 1])
            min_energy_history.append(e0)
        return energies, basis_states, np.asarray(min_energy_history, dtype=float)

    return energies, basis_states
