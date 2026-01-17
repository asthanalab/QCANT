"""qscEOM implementation.

This module was previously stored under ``QCANT/tests`` as an experiment/script.
It has been promoted into the package so it can be imported and documented.

Notes
-----
This code depends on optional scientific Python packages (e.g. PennyLane).
Imports are intentionally performed inside functions so that importing QCANT
does not require these optional dependencies.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

from .excitations import inite


def qscEOM(
    symbols: Sequence[str],
    geometry,
    active_electrons: int,
    active_orbitals: int,
    charge: int,
    params=None,
    ash_excitation=None,
    *,
    ansatz: Optional[Tuple[Any, Any, Any]] = None,
    basis: str = "sto-3g",
    method: str = "pyscf",
    shots: int = 0,
):
    """Compute qscEOM eigenvalues from an ansatz state.

    Parameters
    ----------
    symbols
        Atomic symbols.
    geometry
        Nuclear coordinates (as an array-like object).
    active_electrons
        Number of active electrons.
    active_orbitals
        Number of active orbitals.
    charge
        Total molecular charge.
    params
        Ansatz parameters.
    ash_excitation
        Excitation list describing the ansatz.
    shots
        If 0, run in analytic mode; otherwise use shot-based estimation.

    Returns
    -------
    list
        Sorted eigenvalues for the constructed effective matrix.
    """

    if ansatz is not None:
        try:
            params_from_adapt, ash_excitation_from_adapt, _energies = ansatz
        except Exception as exc:
            raise ValueError(
                "ansatz must be a 3-tuple like (params, ash_excitation, energies) "
                "as returned by QCANT.adapt_vqe"
            ) from exc

        params = params_from_adapt
        ash_excitation = ash_excitation_from_adapt

    if params is None or ash_excitation is None:
        raise TypeError(
            "qscEOM requires either (params, ash_excitation) or ansatz=(params, ash_excitation, energies)."
        )

    try:
        if len(params) != len(ash_excitation):
            raise ValueError
    except Exception as exc:
        raise ValueError("params and ash_excitation must have the same length") from exc

    try:
        import numpy as np
        import pennylane as qml
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "qscEOM requires dependencies. Install at least: "
            "`pip install numpy pennylane`."
        ) from exc

    H, qubits = qml.qchem.molecular_hamiltonian(
        symbols,
        geometry,
        basis=basis,
        method=method,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
        charge=charge,
    )

    hf_state = qml.qchem.hf_state(active_electrons, qubits)
    singles, doubles = qml.qchem.excitations(active_electrons, qubits)
    s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)
    wires = range(qubits)

    null_state = np.zeros(qubits, int)
    list1 = inite(active_electrons, qubits)
    values = []

    # Preserve original behavior (single iteration) from the prior script.
    for _ in range(1):
        if shots == 0:
            dev = qml.device("default.qubit", wires=qubits)
        else:
            dev = qml.device("default.qubit", wires=qubits, shots=shots)

        @qml.qnode(dev)
        def circuit_d(params, occ, wires, s_wires, d_wires, hf_state, ash_excitation):
            qml.BasisState(hf_state, wires=range(qubits))
            for w in occ:
                qml.X(wires=w)
            for i, excitations in enumerate(ash_excitation):
                if len(excitations) == 4:
                    qml.FermionicDoubleExcitation(
                        weight=params[i],
                        wires1=list(range(excitations[0], excitations[1] + 1)),
                        wires2=list(range(excitations[2], excitations[3] + 1)),
                    )
                elif len(excitations) == 2:
                    qml.FermionicSingleExcitation(
                        weight=params[i],
                        wires=list(range(excitations[0], excitations[1] + 1)),
                    )
            return qml.expval(H)

        @qml.qnode(dev)
        def circuit_od(params, occ1, occ2, wires, s_wires, d_wires, hf_state, ash_excitation):
            qml.BasisState(hf_state, wires=range(qubits))
            for w in occ1:
                qml.X(wires=w)
            first = -1
            for v in occ2:
                if v not in occ1:
                    if first == -1:
                        first = v
                        qml.Hadamard(wires=v)
                    else:
                        qml.CNOT(wires=[first, v])
            for v in occ1:
                if v not in occ2:
                    if first == -1:
                        first = v
                        qml.Hadamard(wires=v)
                    else:
                        qml.CNOT(wires=[first, v])
            for i, excitations in enumerate(ash_excitation):
                if len(excitations) == 4:
                    qml.FermionicDoubleExcitation(
                        weight=params[i],
                        wires1=list(range(excitations[0], excitations[1] + 1)),
                        wires2=list(range(excitations[2], excitations[3] + 1)),
                    )
                elif len(excitations) == 2:
                    qml.FermionicSingleExcitation(
                        weight=params[i],
                        wires=list(range(excitations[0], excitations[1] + 1)),
                    )
            return qml.expval(H)

        M = np.zeros((len(list1), len(list1)))
        for i in range(len(list1)):
            for j in range(len(list1)):
                if i == j:
                    M[i, i] = circuit_d(params, list1[i], wires, s_wires, d_wires, null_state, ash_excitation)

        for i in range(len(list1)):
            for j in range(len(list1)):
                if i != j:
                    Mtmp = circuit_od(
                        params,
                        list1[i],
                        list1[j],
                        wires,
                        s_wires,
                        d_wires,
                        null_state,
                        ash_excitation,
                    )
                    M[i, j] = Mtmp - M[i, i] / 2.0 - M[j, j] / 2.0

        eig, _ = np.linalg.eig(M)
        values.append(np.sort(eig))

    return values
