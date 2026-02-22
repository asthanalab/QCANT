"""Quantum real-time Krylov basis generators.

This module provides two public routines:

- :func:`qrte` for forward-time basis growth using ``exp(-i dt H)``,
- :func:`qrte_pmte` for symmetric ``+/-`` time basis growth using both
  ``exp(-i dt H)`` and ``exp(+i dt H)``.

Both methods start from the Hartree-Fock (HF) reference state in the requested
active space and solve the projected generalized eigenvalue problem in the
resulting non-orthogonal basis.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple
import warnings


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
    print_hamiltonian: bool = False,
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
    print_hamiltonian
        If True, print the full dense Hamiltonian matrix used for evolution.

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
    This implementation enforces a real-valued Hamiltonian by dropping tiny
    imaginary parts in the coefficients. This keeps simulator backends like
    lightning.qubit stable when numerical noise introduces complex terms.
    This routine requires analytic execution (statevector access). It uses a
    statevector device and returns the full wavefunction after each step.

    The Hamiltonian projection uses a dense matrix by default, which scales as
    ``O(4**n_qubits)`` in memory. Set ``use_sparse=True`` to request a sparse
    representation when available.
    """
    # pylint: disable=too-many-locals, too-many-statements, too-many-branches, too-many-arguments

    # --------------------------------------------------------------------------
    # 1. Input validation and dependency imports.
    # --------------------------------------------------------------------------
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
        import pyscf
        from pyscf import gto, mcscf, scf
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "qrte requires dependencies. Install at least: `pip install numpy pennylane pyscf`."
        ) from exc

    # --------------------------------------------------------------------------
    # 2. Device and molecule setup.
    # --------------------------------------------------------------------------
    def _make_device(name: Optional[str], wires: int, *, force_default: bool):
        """Create a PennyLane device."""
        if force_default:
            return qml.device("default.qubit", wires=wires)
        if name is not None:
            try:
                return qml.device(name, wires=wires)
            except Exception:
                return qml.device("default.qubit", wires=wires)
        try:
            return qml.device("lightning.qubit", wires=wires)
        except Exception:
            return qml.device("default.qubit", wires=wires)

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

    # --------------------------------------------------------------------------
    # 3. Hamiltonian construction.
    # --------------------------------------------------------------------------
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
    if print_hamiltonian:
        h_dense = qml.matrix(H, wire_order=wires)
        print("Full Hamiltonian matrix (dense):")
        print(h_dense)

    # --------------------------------------------------------------------------
    # 4. Initial state preparation and time evolution setup.
    # --------------------------------------------------------------------------
    coeffs = np.asarray(getattr(H, "coeffs", []), dtype=complex)
    has_complex_coeffs = coeffs.size > 0 and np.any(np.abs(coeffs.imag) > 1e-12)
    if has_complex_coeffs:
        warnings.warn(
            "Hamiltonian has complex coefficients; falling back to default.qubit for QRTE.",
            RuntimeWarning,
        )
    hf_occ = qml.qchem.hf_state(active_electrons, n_qubits)

    dev = _make_device(device_name, n_qubits, force_default=has_complex_coeffs)

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

    # --------------------------------------------------------------------------
    # 5. Time evolution loop.
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

    # --------------------------------------------------------------------------
    # 6. Diagonalize the Hamiltonian in the generated basis.
    # --------------------------------------------------------------------------
    basis_states = np.stack(basis_states, axis=0)

    energies, _e0 = _project_min_energy(basis_states)

    if return_min_energy_history:
        return energies, basis_states, times, np.asarray(min_energy_history, dtype=float)

    return energies, basis_states, times


def qrte_pmte(
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
    print_hamiltonian: bool = False,
) -> Tuple["object", "object", "object"] | Tuple["object", "object", "object", "object"]:
    r"""Run QRTE with symmetric +/- time-evolved basis states.

    This routine builds a basis using both forward and backward real-time
    evolution at each step:

    .. math::
        \{ |\psi_0\rangle,\ e^{-i\Delta t H}|\psi_0\rangle,\ e^{+i\Delta t H}|\psi_0\rangle,\n
          e^{-i2\Delta t H}|\psi_0\rangle,\ e^{+i2\Delta t H}|\psi_0\rangle,\ldots \}

    The projected generalized eigenvalue problem is then solved in this
    non-orthogonal basis:

    .. math::
        M\\mathbf{c} = S\\mathbf{c}E

    where ``M`` is the projected Hamiltonian and ``S`` is the overlap matrix.

    Parameters
    ----------
    symbols
        Atomic symbols.
    geometry
        Nuclear coordinates, array-like with shape ``(n_atoms, 3)`` in Angstrom.
    delta_t
        Time step for each real-time evolution application.
    n_steps
        Number of +/- evolution steps. The returned basis contains
        ``1 + 2*n_steps`` vectors.
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
        +/- iteration as the basis grows from 1 to ``1 + 2*n_steps`` vectors.
    print_hamiltonian
        If True, print the full dense Hamiltonian matrix used for evolution.

    Returns
    -------
    tuple
        ``(energies, basis_states, times)`` where:

        - ``energies`` is a real-valued array of eigenvalues obtained by
          diagonalizing the projected Hamiltonian in the generated basis.
        - ``basis_states`` is a complex-valued array with shape
          ``(1 + 2*n_steps, 2**n_qubits)``.
        - ``times`` is a float array with shape ``(1 + 2*n_steps,)`` and
          entries ``[0, +dt, -dt, +2dt, -2dt, ...]`` matching the basis order.

        If ``return_min_energy_history=True``, returns
        ``(energies, basis_states, times, min_energy_history)`` where
        ``min_energy_history`` has shape ``(n_steps,)`` and stores the minimum
        energy after each +/- pair is added.
    """
    # pylint: disable=too-many-locals, too-many-statements, too-many-branches, too-many-arguments

    # --------------------------------------------------------------------------
    # 1. Input validation and dependency imports.
    # --------------------------------------------------------------------------
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
        import pyscf
        from pyscf import gto, mcscf, scf
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "qrte_pmte requires dependencies. Install at least: "
            "`pip install numpy pennylane pyscf`."
        ) from exc

    # --------------------------------------------------------------------------
    # 2. Device and molecule setup.
    # --------------------------------------------------------------------------
    def _make_device(name: Optional[str], wires: int, *, force_default: bool):
        """Create a PennyLane device."""
        if force_default:
            return qml.device("default.qubit", wires=wires)
        if name is not None:
            try:
                return qml.device(name, wires=wires)
            except Exception:
                return qml.device("default.qubit", wires=wires)
        try:
            return qml.device("lightning.qubit", wires=wires)
        except Exception:
            return qml.device("default.qubit", wires=wires)

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

    # --------------------------------------------------------------------------
    # 3. Hamiltonian construction.
    # --------------------------------------------------------------------------
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
    if print_hamiltonian:
        h_dense = qml.matrix(H, wire_order=wires)
        print("Full Hamiltonian matrix (dense):")
        print(h_dense)

    # --------------------------------------------------------------------------
    # 4. Initial state preparation and time evolution setup.
    # --------------------------------------------------------------------------
    coeffs = np.asarray(getattr(H, "coeffs", []), dtype=complex)
    has_complex_coeffs = coeffs.size > 0 and np.any(np.abs(coeffs.imag) > 1e-12)
    if has_complex_coeffs:
        warnings.warn(
            "Hamiltonian has complex coefficients; falling back to default.qubit for QRTE PMTE.",
            RuntimeWarning,
        )
    hf_occ = qml.qchem.hf_state(active_electrons, n_qubits)

    dev = _make_device(device_name, n_qubits, force_default=has_complex_coeffs)

    @qml.qnode(dev)
    def _hf_statevector():
        qml.BasisState(hf_occ, wires=wires)
        return qml.state()

    @qml.qnode(dev)
    def _evolve_plus(state):
        qml.StatePrep(state, wires=wires)
        qml.ApproxTimeEvolution(H, delta_t, trotter_steps)
        return qml.state()

    @qml.qnode(dev)
    def _evolve_minus(state):
        qml.StatePrep(state, wires=wires)
        qml.ApproxTimeEvolution(H, -delta_t, trotter_steps)
        return qml.state()

    psi0 = _hf_statevector()
    psi0 = psi0 / np.linalg.norm(psi0)

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

    psi0 = _apply_basis_threshold(psi0)

    # --------------------------------------------------------------------------
    # 5. Time evolution loop (+/- basis construction).
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

    basis_states = [psi0]
    min_energy_history = []
    psi_plus = psi0
    psi_minus = psi0

    for _ in range(n_steps):
        psi_plus = _evolve_plus(psi_plus)
        psi_plus = psi_plus / np.linalg.norm(psi_plus)
        psi_plus = _apply_basis_threshold(psi_plus)

        psi_minus = _evolve_minus(psi_minus)
        psi_minus = psi_minus / np.linalg.norm(psi_minus)
        psi_minus = _apply_basis_threshold(psi_minus)

        basis_states.extend([psi_plus, psi_minus])

        if return_min_energy_history:
            current = np.stack(basis_states, axis=0)
            _evals, e0 = _project_min_energy(current)
            min_energy_history.append(e0)

    times = [0.0]
    for k in range(1, n_steps + 1):
        t = float(k) * float(delta_t)
        times.extend([t, -t])
    times = np.asarray(times, dtype=float)

    # --------------------------------------------------------------------------
    # 6. Diagonalize the Hamiltonian in the generated basis.
    # --------------------------------------------------------------------------
    basis_states = np.stack(basis_states, axis=0)
    energies, _e0 = _project_min_energy(basis_states)

    if return_min_energy_history:
        return energies, basis_states, times, np.asarray(min_energy_history, dtype=float)

    return energies, basis_states, times
