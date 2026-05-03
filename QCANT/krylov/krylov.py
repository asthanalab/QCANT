"""Exact Krylov subspace algorithm.

This module builds a Krylov basis { |psi>, H|psi>, H^2|psi>, ... } and
diagonalizes the Hamiltonian in that basis.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple
import warnings

from .._accelerator import build_qml_device, resolve_array_module, to_host_array


def _validate_inputs(
    symbols: Sequence[str],
    geometry: "object",
    n_steps: int,
    overlap_tol: float,
    krylov_method: str,
) -> int:
    """Validate user-provided inputs for the exact_krylov algorithm.

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
    if overlap_tol <= 0:
        raise ValueError("overlap_tol must be > 0")

    if krylov_method.lower() not in {"exact", "lanczos"}:
        raise ValueError('krylov_method must be "exact" or "lanczos"')

    try:
        n_atoms = len(symbols)
        if len(geometry) != n_atoms:
            raise ValueError
    except Exception as exc:
        raise ValueError("geometry must have the same length as symbols") from exc

    return n_atoms


def exact_krylov(
    symbols: Sequence[str],
    geometry,
    *,
    n_steps: int,
    active_electrons: int,
    active_orbitals: int,
    basis: str = "sto-3g",
    charge: int = 0,
    spin: int = 0,
    method: str = "pyscf",
    device_name: Optional[str] = None,
    initial_state: Optional["object"] = None,
    array_backend: str = "auto",
    overlap_tol: float = 1e-10,
    normalize_basis: bool = True,
    basis_threshold: float = 0.0,
    krylov_method: str = "exact",
    use_sparse: bool = False,
    return_min_energy_history: bool = False,
) -> Tuple["object", "object"] | Tuple["object", "object", "object"]:
    """Generate a Krylov basis and diagonalize the Hamiltonian in that basis.

    The basis is built as ``{|psi>, H|psi>, H^2|psi>, ...}`` up to ``n_steps``.
    Energies are obtained by diagonalizing the Hamiltonian projected into the
    generated basis after overlap-based orthonormalization.

    Parameters
    ----------
    symbols
        Atomic symbols.
    geometry
        Nuclear coordinates, array-like with shape ``(n_atoms, 3)`` in Angstrom.
    n_steps
        Number of Krylov steps. The returned basis contains ``n_steps + 1``
        vectors (including the initial state).
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
        PennyLane device name used only to prepare the HF state if
        ``initial_state`` is not provided.
    initial_state
        Optional statevector to seed the Krylov basis. If not provided, the
        Hartree–Fock state is used.
    array_backend
        Dense linear algebra backend. ``"numpy"`` keeps CPU execution,
        ``"cupy"`` requests GPU dense linear algebra, and ``"auto"`` keeps
        CPU execution unless a GPU PennyLane device is explicitly requested.
    overlap_tol
        Threshold for discarding near-linearly dependent basis vectors when
        orthonormalizing the basis via the overlap matrix eigen-decomposition.
    normalize_basis
        If True, normalize each Krylov vector to avoid numerical overflow.
        This applies to ``krylov_method="exact"``.
    basis_threshold
        Drop amplitudes with absolute value below this threshold after each
        basis update. The thresholded state is re-normalized. Use 0.0 to
        disable thresholding.
    krylov_method
        Krylov construction method. Use ``"exact"`` for raw powers of ``H``,
        or ``"lanczos"`` to build an orthonormal Krylov basis.
    use_sparse
        If True, use a sparse Hamiltonian representation for state updates.
    return_min_energy_history
        If True, also return the minimum energy after each Krylov step.

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
    # pylint: disable=too-many-locals, too-many-statements, too-many-branches, too-many-arguments

    # --------------------------------------------------------------------------
    # 1. Input validation and dependency imports.
    # --------------------------------------------------------------------------
    n_atoms = _validate_inputs(symbols, geometry, n_steps, overlap_tol, krylov_method)

    try:
        import numpy as np
        import pennylane as qml
        import pyscf
        from pyscf import gto, mcscf, scf
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "exact_krylov requires dependencies. "
            "Install at least: `pip install numpy pennylane pyscf`."
        ) from exc

    # --------------------------------------------------------------------------
    # 2. Device and molecule setup.
    # --------------------------------------------------------------------------
    xp, _array_backend_name, _using_gpu = resolve_array_module(
        array_backend=array_backend,
        device_name=device_name,
        allow_gpu=not use_sparse,
        gpu_fallback_reason="Exact Krylov sparse execution remains CPU-only in v1; set use_sparse=False for GPU runs.",
        context="exact_krylov",
    )

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

    # --------------------------------------------------------------------------
    # 4. Initial state preparation.
    # --------------------------------------------------------------------------
    if initial_state is None:
        hf_occ = qml.qchem.hf_state(active_electrons, n_qubits)
        dev = build_qml_device(
            qml,
            device_name=device_name,
            wires=n_qubits,
            device_kwargs=None,
            shots=None,
        )

        @qml.qnode(dev)
        def _hf_statevector():
            qml.BasisState(hf_occ, wires=wires)
            return qml.state()

        psi = xp.asarray(_hf_statevector(), dtype=complex)
    else:
        psi = xp.asarray(initial_state, dtype=complex)
        if psi.ndim != 1:
            raise ValueError("initial_state must be a 1D statevector")
        expected_dim = 2**n_qubits
        if psi.size != expected_dim:
            raise ValueError(
                f"initial_state must have length {expected_dim}, got {psi.size}"
            )

    psi_norm = float(to_host_array(xp.linalg.norm(psi)).item())
    if psi_norm == 0:
        raise ValueError("initial_state has zero norm")
    psi = psi / psi_norm

    def _apply_basis_threshold(state):
        if basis_threshold <= 0:
            return state
        state = xp.asarray(state, dtype=complex)
        mask = xp.abs(state) >= basis_threshold
        if not bool(to_host_array(xp.any(mask)).item()):
            idx = int(to_host_array(xp.argmax(xp.abs(state))).item())
            mask[idx] = True
        state = xp.where(mask, state, 0.0)
        norm = float(to_host_array(xp.linalg.norm(state)).item())
        if norm == 0:
            raise ValueError("thresholded basis vector has zero norm")
        return state / norm

    psi = _apply_basis_threshold(psi)

    # --------------------------------------------------------------------------
    # 5. Krylov basis construction.
    # --------------------------------------------------------------------------
    if use_sparse:
        try:
            import scipy.sparse  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError("use_sparse=True requires scipy") from exc
        if hasattr(H, "sparse_matrix") and getattr(H, "has_sparse_matrix", True):
            H_mat = H.sparse_matrix(wire_order=wires, format="csr")
        else:
            warnings.warn(
                "Hamiltonian does not expose sparse_matrix; falling back to dense matrix.",
                RuntimeWarning,
            )
            H_mat = np.asarray(qml.matrix(H, wire_order=wires), dtype=complex)
    else:
        H_mat = xp.asarray(qml.matrix(H, wire_order=wires), dtype=complex)

    def _apply_h(state):
        return H_mat @ state

    def _project_min_energy(current_basis_states):
        """Project and diagonalize the Hamiltonian, returning the minimum energy."""
        S = current_basis_states.conj() @ current_basis_states.T
        H_proj = current_basis_states.conj() @ (H_mat @ current_basis_states.T)

        s_vals, s_vecs = xp.linalg.eigh(S)
        keep = s_vals > float(overlap_tol)
        if not bool(to_host_array(xp.any(keep)).item()):
            raise ValueError("overlap matrix is numerically singular; basis collapsed")

        X = s_vecs[:, keep] / xp.sqrt(s_vals[keep])[None, :]
        H_ortho = X.conj().T @ H_proj @ X
        evals = xp.linalg.eigvalsh(H_ortho).real
        return evals, float(to_host_array(evals[0]).item())

    if krylov_method == "exact":
        basis_states = [psi]
        current = psi
        for _ in range(n_steps):
            current = _apply_h(current)
            if normalize_basis:
                current_norm = float(to_host_array(xp.linalg.norm(current)).item())
                if current_norm == 0:
                    raise ValueError("Krylov vector has zero norm")
                current = current / current_norm
            current = _apply_basis_threshold(current)
            basis_states.append(current)

        basis_states = xp.stack(basis_states, axis=0)
        energies, _e0 = _project_min_energy(basis_states)

        if return_min_energy_history:
            min_energy_history = []
            num_steps = basis_states.shape[0] - 1
            for k in range(1, num_steps + 1):
                _evals, e0 = _project_min_energy(basis_states[: k + 1])
                min_energy_history.append(e0)
            return (
                to_host_array(energies, dtype=float),
                to_host_array(basis_states, dtype=complex),
                np.asarray(min_energy_history, dtype=float),
            )

        return to_host_array(energies, dtype=float), to_host_array(basis_states, dtype=complex)

    # --------------------------------------------------------------------------
    # 6. Lanczos iteration.
    # --------------------------------------------------------------------------
    basis_states = []
    alphas = []
    betas = []
    v = psi
    prev = None
    beta = 0.0

    for step in range(n_steps + 1):
        basis_states.append(v)
        w = _apply_h(v)
        alpha = np.vdot(v, w).real
        w = w - alpha * v
        if prev is not None:
            w = w - beta * prev
        beta = float(to_host_array(xp.linalg.norm(w)).item())
        alphas.append(alpha)
        if step == n_steps:
            break
        if beta == 0:
            raise ValueError("Lanczos breakdown before reaching n_steps")
        prev = v
        v = w / beta
        v = _apply_basis_threshold(v)
        betas.append(beta)

    basis_states = xp.stack(basis_states, axis=0)
    T = xp.diag(xp.asarray(alphas, dtype=float))
    if betas:
        off = xp.asarray(betas, dtype=float)
        T = T + xp.diag(off, 1) + xp.diag(off, -1)

    energies = xp.linalg.eigvalsh(T).real

    if return_min_energy_history:
        min_energy_history = []
        num_steps = basis_states.shape[0] - 1
        for k in range(1, num_steps + 1):
            sub_T = T[: k + 1, : k + 1]
            evals = xp.linalg.eigvalsh(sub_T).real
            min_energy_history.append(float(to_host_array(evals[0]).item()))
        return (
            to_host_array(energies, dtype=float),
            to_host_array(basis_states, dtype=complex),
            np.asarray(min_energy_history, dtype=float),
        )

    return to_host_array(energies, dtype=float), to_host_array(basis_states, dtype=complex)
