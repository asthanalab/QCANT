"""ADAPT-VQE implementation.

This module was previously stored under ``QCANT/tests`` as an experiment/script.
It has been promoted into the package so it can be imported and documented.

Notes
-----
This code uses optional heavy dependencies (PySCF, PennyLane, SciPy, etc.).
Imports are performed inside the main function so that importing QCANT does not
require these dependencies.
"""

from __future__ import annotations

from typing import Optional, Sequence


def adapt_vqe(
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
    optimizer_maxiter: int = 100_000_000,
):
    """Run an ADAPT-style VQE loop for a user-specified molecular geometry.

    The core ADAPT loop selects operators from a singles+doubles pool based on
    commutator magnitude, then optimizes the ansatz parameters at each
    iteration.

    Parameters
    ----------
    symbols
        Atomic symbols, e.g. ``["H", "H"]``.
    geometry
        Nuclear coordinates, array-like with shape ``(n_atoms, 3)`` in Angstrom.
    adapt_it
        Number of ADAPT iterations.
    basis
        Basis set name understood by PySCF (e.g. ``"sto-3g"``, ``"sto-6g"``).
    charge
        Total molecular charge.
    spin
        Spin multiplicity parameter used by PySCF as ``2S`` (e.g. 0 for singlet).
    active_electrons
        Number of active electrons in the CASCI reference.
    active_orbitals
        Number of active orbitals in the CASCI reference.

    Returns
    -------
    tuple
        ``(params, ash_excitation, energies)`` as produced by the optimization.

    Raises
    ------
    ValueError
        If ``symbols``/``geometry`` sizes are inconsistent.
    ImportError
        If required optional dependencies are not installed.
    """

    if len(symbols) == 0:
        raise ValueError("symbols must be non-empty")

    try:
        n_atoms = len(symbols)
        if len(geometry) != n_atoms:
            raise ValueError
    except Exception as exc:
        raise ValueError("geometry must have the same length as symbols") from exc

    try:
        import re
        import warnings

        import numpy as np
        import pennylane as qml
        import pyscf
        from pennylane import numpy as pnp
        from pyscf import gto, mcscf, scf
        from scipy.optimize import minimize

        warnings.filterwarnings("ignore")
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "adapt_vqe requires dependencies. Install at least: "
            "`pip install numpy scipy pennylane pyscf` "
            "(and optionally a faster PennyLane device backend, e.g. `pip install pennylane-lightning`)."
        ) from exc

    def _make_device(name: Optional[str], wires: int):
        if name is not None:
            return qml.device(name, wires=wires)
        # Backwards-compatible preference for lightning if available.
        try:
            return qml.device("lightning.qubit", wires=wires)
        except Exception:
            return qml.device("default.qubit", wires=wires)

    # Build the molecule from user-provided symbols/geometry.
    # PySCF accepts either a multiline string or a list of (symbol, (x,y,z)).
    atom = [(symbols[i], tuple(float(x) for x in geometry[i])) for i in range(n_atoms)]

    # ---------- Step 1: Reference CASCI calculation ----------
    mol_ref = gto.Mole()
    mol_ref.atom = atom
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

    en = mycas_ref.kernel()
    print("Ref.CASCI energy:", en[0])

    two_mo = pyscf.ao2mo.restore("1", h2ecas, norb=mycas_ref.mo_coeff.shape[1])
    two_mo = np.swapaxes(two_mo, 1, 3)

    one_mo = h1ecas
    core_constant = np.array([ecore])

    H_fermionic = qml.qchem.fermionic_observable(core_constant, one_mo, two_mo, cutoff=1e-20)
    H = qml.jordan_wigner(H_fermionic)

    qubits = 2 * (mycas_ref.mo_coeff.shape[1])
    active_electrons = sum(mycas_ref.nelecas)

    energies = []
    ash_excitation = []

    hf_state = qml.qchem.hf_state(active_electrons, qubits)
    dev = _make_device(device_name, qubits)

    @qml.qnode(dev)
    def commutator_0(H, w, k):
        qml.BasisState(k, wires=range(qubits))
        res = qml.commutator(H, w)
        return qml.expval(res)

    @qml.qnode(dev)
    def commutator_1(H, w, k):
        qml.StatePrep(k, wires=range(qubits))
        res = qml.commutator(H, w)
        return qml.expval(res)

    @qml.qnode(dev)
    def ash(params, ash_excitation, hf_state, H):
        [qml.PauliX(i) for i in np.nonzero(hf_state)[0]]
        for i, excitation in enumerate(ash_excitation):
            if len(ash_excitation[i]) == 4:
                qml.FermionicDoubleExcitation(
                    weight=params[i],
                    wires1=list(range(ash_excitation[i][0], ash_excitation[i][1] + 1)),
                    wires2=list(range(ash_excitation[i][2], ash_excitation[i][3] + 1)),
                )
            elif len(ash_excitation[i]) == 2:
                qml.FermionicSingleExcitation(
                    weight=params[i],
                    wires=list(range(ash_excitation[i][0], ash_excitation[i][1] + 1)),
                )
        return qml.expval(H)

    dev1 = _make_device(device_name, qubits)

    @qml.qnode(dev1)
    def new_state(hf_state, ash_excitation, params):
        [qml.PauliX(i) for i in np.nonzero(hf_state)[0]]
        for i, excitations in enumerate(ash_excitation):
            if len(ash_excitation[i]) == 4:
                qml.FermionicDoubleExcitation(
                    weight=params[i],
                    wires1=list(range(ash_excitation[i][0], ash_excitation[i][1] + 1)),
                    wires2=list(range(ash_excitation[i][2], ash_excitation[i][3] + 1)),
                )
            elif len(ash_excitation[i]) == 2:
                qml.FermionicSingleExcitation(
                    weight=params[i],
                    wires=list(range(ash_excitation[i][0], ash_excitation[i][1] + 1)),
                )
        return qml.state()

    def cost(params):
        return ash(params, ash_excitation, hf_state, H)

    singles, doubles = qml.qchem.excitations(active_electrons, qubits)
    op1 = [qml.fermi.FermiWord({(0, x[0]): "+", (1, x[1]): "-"}) for x in singles]
    op2 = [
        qml.fermi.FermiWord({(0, x[0]): "+", (1, x[1]): "+", (2, x[2]): "-", (3, x[3]): "-"})
        for x in doubles
    ]
    operator_pool = op1 + op2
    states = [hf_state]
    params = pnp.zeros(len(ash_excitation), requires_grad=True)

    for j in range(adapt_it):
        print("The adapt iteration now is", j, flush=True)
        max_value = float("-inf")
        max_operator = None
        k = states[-1] if states else hf_state

        for i in operator_pool:
            w = qml.fermi.jordan_wigner(i)
            if np.array_equal(k, hf_state):
                current_value = abs(2 * (commutator_0(H, w, k)))
            else:
                current_value = abs(2 * (commutator_1(H, w, k)))

            if current_value > max_value:
                max_value = current_value
                max_operator = i

        indices_str = re.findall(r"\d+", str(max_operator))
        excitations = [int(index) for index in indices_str]
        ash_excitation.append(excitations)

        params = np.append(np.asarray(params), 0.0)
        result = minimize(
            cost,
            params,
            method="BFGS",
            tol=1e-12,
            options={"disp": False, "maxiter": int(optimizer_maxiter)},
        )

        energies.append(result.fun)
        params = result.x
        print("Energies are", energies, flush=True)
        ostate = new_state(hf_state, ash_excitation, params)
        states.append(ostate)

    print("energies:", energies[-1])
    return params, ash_excitation, energies
