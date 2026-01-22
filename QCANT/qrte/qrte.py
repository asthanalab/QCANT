"""Quantum Real Time Evolution (QRTE).

This module implements a simple real-time evolution loop that generates a
basis by repeatedly evolving a state by a fixed timestep ``delta_t``.

The starting state is the Hartree–Fock (HF) reference state for the requested
active space.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple


def qrte(
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
    method: str = "pyscf",
    device_name: Optional[str] = None,
    trotter_steps: int = 1,
    overlap_tol: float = 1e-10,
    use_sparse: bool = False,
    basis_threshold: float = 0.0,
    return_min_energy_history: bool = False,
) -> Tuple["object", "object", "object"] | Tuple["object", "object", "object", "object"]:
    """Run a quantum real-time evolution loop and return energies from the generated basis.

    At each step the current state is evolved by ``delta_t`` under the molecular
    Hamiltonian, producing a new state which is appended to the basis.

    Once the basis is generated, the molecular Hamiltonian is projected into this
    (generally non-orthogonal) basis and diagonalized to obtain approximate energies.

    Parameters
    ----------
    symbols
        Atomic symbols.
    geometry
        Nuclear coordinates, array-like with shape ``(n_atoms, 3)`` in Angstrom.
    delta_t
        Time step for each real-time evolution application.
    n_steps
        Number of time-evolution steps. The returned basis contains ``n_steps + 1``
        vectors (including the initial HF state).
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
        PennyLane device name (e.g. ``"default.qubit"``). If not provided,
        the function will prefer ``"lightning.qubit"`` if available.
    trotter_steps
        Number of Trotter steps used internally by :class:`pennylane.ApproxTimeEvolution`.
    overlap_tol
        Threshold for discarding near-linearly dependent basis vectors when
        orthonormalizing the basis via the overlap matrix eigen-decomposition.
    use_sparse
        If True, use a sparse Hamiltonian representation for projections.
    basis_threshold
        Drop amplitudes with absolute value below this threshold after each
        basis update. The thresholded state is re-normalized. Use 0.0 to
        disable thresholding.
    return_min_energy_history
        If True, also return an array containing the minimum energy after each
        iteration as the basis grows from 1 to ``n_steps + 1`` vectors.

    Returns
    -------
    tuple
                ``(energies, basis_states, times)`` where:

                - ``energies`` is a real-valued array of eigenvalues obtained by diagonalizing
                    the Hamiltonian projected into the generated basis

        - ``basis_states`` is a complex-valued array with shape ``(n_steps+1, 2**n_qubits)``
        - ``times`` is a float array with shape ``(n_steps+1,)`` giving the time associated
          with each basis vector

                If ``return_min_energy_history=True``, the function returns
                ``(energies, basis_states, times, min_energy_history)`` where
                ``min_energy_history`` has shape ``(n_steps,)`` and contains the minimum
                energy after each iteration (using the basis with ``k+1`` vectors).

    Raises
    ------
    ValueError
        If inputs are invalid (e.g. ``delta_t <= 0`` or ``n_steps < 0``).
    ImportError
        If required scientific dependencies are not installed.

    Notes
    -----
    This routine requires analytic execution (statevector access). It uses a
    statevector device and returns the full wavefunction after each step.

    The Hamiltonian projection uses a dense matrix by default, which scales as
    ``O(4**n_qubits)`` in memory. Set ``use_sparse=True`` to request a sparse
    representation when available.
    """

    if len(symbols) == 0:
        raise ValueError("symbols must be non-empty")
    if delta_t <= 0:
        raise ValueError("delta_t must be > 0")
    if n_steps < 0:
        raise ValueError("n_steps must be >= 0")
    if trotter_steps < 1:
        raise ValueError("trotter_steps must be >= 1")
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
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "qrte requires dependencies. Install at least: `pip install numpy pennylane pyscf`."
        ) from exc

    def _make_device(name: Optional[str], wires: int):
        if name is not None:
            try:
                return qml.device(name, wires=wires)
            except Exception:
                return qml.device("default.qubit", wires=wires)
        try:
            return qml.device("lightning.qubit", wires=wires)
        except Exception:
            return qml.device("default.qubit", wires=wires)

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
    hf_occ = qml.qchem.hf_state(active_electrons, n_qubits)

    dev = _make_device(device_name, n_qubits)

    @qml.qnode(dev)
    def _hf_statevector():
        qml.BasisState(hf_occ, wires=wires)
        return qml.state()

    @qml.qnode(dev)
    def _evolve(state):
        qml.StatePrep(state, wires=wires)
        qml.ApproxTimeEvolution(H, delta_t, trotter_steps)
        return qml.state()

    psi = _hf_statevector()
    psi = psi / np.linalg.norm(psi)

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
    min_energy_history = []

    for _ in range(n_steps):
        psi = _evolve(psi)
        psi = psi / np.linalg.norm(psi)
        psi = _apply_basis_threshold(psi)
        basis_states.append(psi)

        if return_min_energy_history:
            current = np.stack(basis_states, axis=0)
            _evals, e0 = _project_min_energy(current)
            min_energy_history.append(e0)

    times = np.arange(n_steps + 1, dtype=float) * float(delta_t)

    basis_states = np.stack(basis_states, axis=0)

    energies, _e0 = _project_min_energy(basis_states)

    if return_min_energy_history:
        return energies, basis_states, times, np.asarray(min_energy_history, dtype=float)

    return energies, basis_states, times
