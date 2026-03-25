"""ADAPT-VQE with exact first/second-order Krylov post-processing."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Sequence

from ..adapt.adaptvqe import (
    _ansatz_type_from_pool_type,
    _apply_excitation_gate,
    _normalize_pool_type,
    adapt_vqe,
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


def _build_hamiltonian(
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
    """Build the active-space Hamiltonian used for exact Krylov diagnostics."""
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

    return qml, np, hamiltonian, hamiltonian_matrix, int(qubits), int(active_electrons)


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


def _solve_krylov_orders(np, psi, hamiltonian_matrix, overlap_tol: float) -> tuple[float, float, int, int]:
    """Compute first- and second-order Krylov energies for ``psi``."""
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
    parallel_gradients: bool = False,
    parallel_backend: str = "auto",
    max_workers: Optional[int] = None,
    gradient_chunk_size: Optional[int] = None,
    optimizer_method: str = "BFGS",
    optimizer_maxiter: int = 100_000_000,
    pauli_grouping: bool = False,
    grouping_type: str = "qwc",
    device_kwargs: Optional[Mapping[str, Any]] = None,
    overlap_tol: float = 1e-10,
    iteration_callback: Optional[Callable[[Mapping[str, Any]], None]] = None,
):
    """Run ADAPT-VQE and attach exact order-1/order-2 Krylov energies per iteration.

    The Krylov subspaces are built from the optimized ADAPT state at each
    iteration using the vectors ``|psi>``, ``H|psi>``, and ``H^2|psi>``.

    Returns
    -------
    tuple
        ``(params, ash_excitation, adapt_energies, details)`` where ``details``
        contains the exact ground-state energy plus the Krylov energies/history:

        - ``krylov_order1_energies``
        - ``krylov_order2_energies``
        - ``exact_ground_energy``
        - ``history``
    """
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

    qml, np, _hamiltonian, hamiltonian_matrix, qubits, effective_electrons = _build_hamiltonian(
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
        order1_energy, order2_energy, order1_rank, order2_rank = _solve_krylov_orders(
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
        "krylov_order1_energies": np.asarray(krylov_order1_energies, dtype=float),
        "krylov_order2_energies": np.asarray(krylov_order2_energies, dtype=float),
        "exact_ground_energy": float(exact_ground_energy),
        "history": history,
    }
    return params, ash_excitation, adapt_energies, details


adapt_krylov = adaptKrylov

