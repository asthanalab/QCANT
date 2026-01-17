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
) -> Tuple["object", "object"]:
    """Run a quantum real-time evolution loop and return a basis of states.

    At each step the current state is evolved by ``delta_t`` under the molecular
    Hamiltonian, producing a new state which is appended to the returned basis.

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

    Returns
    -------
    tuple
        ``(basis_states, times)`` where:

        - ``basis_states`` is a complex-valued array with shape ``(n_steps+1, 2**n_qubits)``
        - ``times`` is a float array with shape ``(n_steps+1,)`` giving the time associated
          with each basis vector

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
    """

    if len(symbols) == 0:
        raise ValueError("symbols must be non-empty")
    if delta_t <= 0:
        raise ValueError("delta_t must be > 0")
    if n_steps < 0:
        raise ValueError("n_steps must be >= 0")
    if trotter_steps < 1:
        raise ValueError("trotter_steps must be >= 1")

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
            return qml.device(name, wires=wires)
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

    basis_states = [psi]
    for _ in range(n_steps):
        psi = _evolve(psi)
        psi = psi / np.linalg.norm(psi)
        basis_states.append(psi)

    times = np.arange(n_steps + 1, dtype=float) * float(delta_t)
    return np.stack(basis_states, axis=0), times
