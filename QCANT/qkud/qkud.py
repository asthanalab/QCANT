"""Quantum Krylov using Unitary Decomposition (QKUD).

This module builds a Krylov basis using the QKUD recurrence
|psi_n> = (X + X^†) / (2 * epsilon) |psi_{n-1}| with
X = i * exp(-i * epsilon * H).
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple


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
    use_sparse
        If True, use a sparse Hamiltonian representation for state updates.
    return_min_energy_history
        If True, also return the minimum energy after each QKUD step.

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

    try:
        import numpy as np
        import pennylane as qml
        from scipy.sparse.linalg import expm_multiply
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "qkud requires dependencies. Install at least: "
            "`pip install numpy pennylane pyscf scipy`."
        ) from exc

    def _basis_state_vector(bits: Sequence[int]) -> "object":
        idx = 0
        for bit in bits:
            idx = (idx << 1) | int(bit)
        state = np.zeros(2 ** len(bits), dtype=complex)
        state[idx] = 1.0
        return state

    try:
        H, n_qubits = qml.qchem.molecular_hamiltonian(
            symbols,
            geometry,
            basis=basis,
            method=method,
            active_electrons=active_electrons,
            active_orbitals=active_orbitals,
            charge=charge,
            spin=spin,
        )
    except TypeError:
        # Older PennyLane versions do not accept a `spin` keyword.
        H, n_qubits = qml.qchem.molecular_hamiltonian(
            symbols,
            geometry,
            basis=basis,
            method=method,
            active_electrons=active_electrons,
            active_orbitals=active_orbitals,
            charge=charge,
        )

    wires = range(n_qubits)

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
        forward = expm_multiply(-1j * epsilon * H_mat, state)
        backward = expm_multiply(1j * epsilon * H_mat, state)
        return (1j * (forward - backward)) / (2.0 * epsilon)

    def _project_min_energy(current_basis_states):
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
        basis_states.append(current)

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
