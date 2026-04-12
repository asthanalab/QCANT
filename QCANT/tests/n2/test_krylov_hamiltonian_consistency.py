"""Compare N2 Krylov Hamiltonians built via PySCF vs PennyLane qchem.

This reproduces the N2 (1.5 Å and 3.0 Å, 6e/6o, STO-6G) setups used in the
Krylov examples and checks that the Pauli-term coefficients and spectra match
once geometry units are aligned.
"""

from __future__ import annotations

import pytest

from QCANT.qchem_units import geometry_to_bohr


def _make_linear_n2_geometry(bond_length: float = 3.0):
    import numpy as np

    return np.array([[0.0, 0.0, 0.0], [0.0, 0.0, bond_length]], dtype=float)


def _build_pyscf_casci_hamiltonian(
    *,
    symbols: list[str],
    geometry,
    basis: str,
    charge: int,
    spin: int,
    active_electrons: int,
    active_orbitals: int,
    level_shift: float | None = 0.5,
    diis_space: int | None = 12,
    use_newton: bool = True,
):
    import numpy as np
    import pennylane as qml
    import pyscf
    from pyscf import gto, mcscf, scf

    atom = [(symbols[i], tuple(float(x) for x in geometry[i])) for i in range(len(symbols))]
    mol = gto.Mole()
    mol.atom = atom
    mol.unit = "Angstrom"
    mol.basis = basis
    mol.charge = charge
    mol.spin = spin
    mol.symmetry = False
    mol.build()

    mf = scf.RHF(mol)
    if level_shift is not None:
        mf.level_shift = level_shift
    if diis_space is not None:
        mf.diis_space = diis_space
    mf.max_cycle = 100
    mf.kernel()
    if use_newton and not mf.converged:
        mf = scf.newton(mf).run()

    mycas = mcscf.CASCI(mf, active_orbitals, active_electrons)
    h1ecas, ecore = mycas.get_h1eff(mf.mo_coeff)
    h2ecas = mycas.get_h2eff(mf.mo_coeff)

    two_mo = pyscf.ao2mo.restore("1", h2ecas, norb=mycas.ncas)
    two_mo = np.swapaxes(two_mo, 1, 3)
    one_mo = h1ecas
    core_constant = np.array([ecore])

    h_fermionic = qml.qchem.fermionic_observable(core_constant, one_mo, two_mo, cutoff=1e-20)
    h_qubit = qml.jordan_wigner(h_fermionic)
    n_qubits = 2 * mycas.ncas
    return h_qubit, n_qubits


def _build_pennylane_hamiltonian(
    *,
    symbols: list[str],
    geometry,
    basis: str,
    charge: int,
    spin: int,
    active_electrons: int,
    active_orbitals: int,
):
    import pennylane as qml

    try:
        return qml.qchem.molecular_hamiltonian(
            symbols,
            geometry,
            basis=basis,
            charge=charge,
            mult=spin + 1,
            active_electrons=active_electrons,
            active_orbitals=active_orbitals,
            method="pyscf",
        )
    except TypeError:
        return qml.qchem.molecular_hamiltonian(
            symbols,
            geometry,
            basis=basis,
            charge=charge,
            active_electrons=active_electrons,
            active_orbitals=active_orbitals,
            method="pyscf",
        )


def _flatten_op(op):
    if hasattr(op, "operands"):
        factors = []
        for child in op.operands:
            factors.extend(_flatten_op(child))
        return factors
    if hasattr(op, "obs"):
        factors = []
        for child in op.obs:
            factors.extend(_flatten_op(child))
        return factors
    return [op]


def _pauli_key(op, n_qubits: int) -> str:
    letters = ["I"] * n_qubits
    for factor in _flatten_op(op):
        name = getattr(factor, "name", factor.__class__.__name__)
        if name in {"Identity", "I"}:
            continue
        if name in {"PauliX", "X"}:
            letter = "X"
        elif name in {"PauliY", "Y"}:
            letter = "Y"
        elif name in {"PauliZ", "Z"}:
            letter = "Z"
        else:
            raise ValueError(f"Unsupported operator in Hamiltonian: {name}")

        for wire in factor.wires:
            letters[int(wire)] = letter
    return "".join(letters)


def _hamiltonian_to_dict(h_op, n_qubits: int, *, tol: float = 1e-12) -> dict[str, complex]:
    import numpy as np

    if hasattr(h_op, "terms"):
        coeffs, ops = h_op.terms()
    else:
        coeffs, ops = h_op.coeffs, h_op.ops

    term_map: dict[str, complex] = {}
    for coeff, op in zip(coeffs, ops):
        coeff = complex(coeff)
        if abs(coeff.imag) < tol:
            coeff = coeff.real
        key = _pauli_key(op, n_qubits)
        term_map[key] = term_map.get(key, 0.0) + coeff

    pruned = {k: v for k, v in term_map.items() if np.abs(v) > tol}
    return pruned


def _spectrum(h_op, n_qubits: int, n_vals: int = 1):
    import numpy as np
    import scipy.sparse.linalg as spla

    if hasattr(h_op, "sparse_matrix"):
        h_mat = h_op.sparse_matrix(wire_order=range(n_qubits), format="csr")
    else:
        import pennylane as qml

        h_mat = qml.matrix(h_op, wire_order=range(n_qubits))
    if hasattr(h_mat, "shape") and h_mat.shape[0] <= 256:
        evals = np.linalg.eigvalsh(h_mat.toarray() if hasattr(h_mat, "toarray") else h_mat)
        return np.sort(evals)[:n_vals]
    evals = spla.eigsh(h_mat, k=n_vals, which="SA", return_eigenvectors=False)
    return np.sort(evals.real)


@pytest.mark.parametrize("bond_length", [1.5, 3.0])
def test_n2_pyscf_vs_pennylane_hamiltonian_terms(bond_length: float):
    import numpy as np
    import pennylane  # noqa: F401
    import pyscf  # noqa: F401

    symbols = ["N", "N"]
    geometry = _make_linear_n2_geometry(bond_length)
    basis = "sto-6g"
    charge = 0
    spin = 0
    active_electrons = 6
    active_orbitals = 6

    h_pyscf, n_qubits = _build_pyscf_casci_hamiltonian(
        symbols=symbols,
        geometry=geometry,
        basis=basis,
        charge=charge,
        spin=spin,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
        level_shift=None,
        diis_space=None,
        use_newton=False,
    )
    geometry_bohr = geometry_to_bohr(geometry)
    h_qml, q_qubits = _build_pennylane_hamiltonian(
        symbols=symbols,
        geometry=geometry_bohr,
        basis=basis,
        charge=charge,
        spin=spin,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
    )

    assert n_qubits == q_qubits
    assert n_qubits == 2 * active_orbitals

    terms_pyscf = _hamiltonian_to_dict(h_pyscf, n_qubits)
    terms_qml = _hamiltonian_to_dict(h_qml, n_qubits)

    all_keys = set(terms_pyscf) | set(terms_qml)
    diffs = {
        key: abs(terms_pyscf.get(key, 0.0) - terms_qml.get(key, 0.0)) for key in all_keys
    }
    max_key = max(diffs, key=diffs.get)
    max_diff = diffs[max_key]
    identity_key = "I" * n_qubits
    identity_diff = abs(terms_pyscf.get(identity_key, 0.0) - terms_qml.get(identity_key, 0.0))

    tol = 1e-1
    assert max_diff < tol, (
        f"N2 {bond_length:.1f}A 6e/6o STO-6G: PySCF CASCI vs PennyLane qchem Hamiltonians differ. "
        f"max |Δcoeff|={max_diff:.3e} at term {max_key}; "
        f"|ΔI|={identity_diff:.3e}; "
        f"terms pyscf={len(terms_pyscf)} pennylane={len(terms_qml)}."
    )


@pytest.mark.parametrize("bond_length", [1.5, 3.0])
def test_n2_pyscf_vs_pennylane_hamiltonian_terms_without_unit_fix(bond_length: float):
    import pennylane  # noqa: F401
    import pyscf  # noqa: F401

    symbols = ["N", "N"]
    geometry = _make_linear_n2_geometry(bond_length)
    basis = "sto-6g"
    charge = 0
    spin = 0
    active_electrons = 6
    active_orbitals = 6

    h_pyscf, n_qubits = _build_pyscf_casci_hamiltonian(
        symbols=symbols,
        geometry=geometry,
        basis=basis,
        charge=charge,
        spin=spin,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
        level_shift=None,
        diis_space=None,
        use_newton=False,
    )
    h_qml, q_qubits = _build_pennylane_hamiltonian(
        symbols=symbols,
        geometry=geometry,
        basis=basis,
        charge=charge,
        spin=spin,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
    )

    assert n_qubits == q_qubits

    terms_pyscf = _hamiltonian_to_dict(h_pyscf, n_qubits)
    terms_qml = _hamiltonian_to_dict(h_qml, n_qubits)
    all_keys = set(terms_pyscf) | set(terms_qml)
    max_diff = max(abs(terms_pyscf.get(k, 0.0) - terms_qml.get(k, 0.0)) for k in all_keys)

    assert max_diff > 1e-2, (
        "Expected a noticeable mismatch when PennyLane is given Angstrom coordinates without "
        "unit conversion. If this fails, check whether PennyLane changed its PySCF unit handling."
    )


@pytest.mark.parametrize("bond_length", [1.5, 3.0])
def test_n2_pyscf_vs_pennylane_spectrum(bond_length: float):
    import numpy as np
    import pennylane  # noqa: F401
    import pyscf  # noqa: F401

    symbols = ["N", "N"]
    geometry = _make_linear_n2_geometry(bond_length)
    geometry_bohr = geometry_to_bohr(geometry)
    basis = "sto-6g"
    charge = 0
    spin = 0
    active_electrons = 6
    active_orbitals = 6

    h_pyscf, n_qubits = _build_pyscf_casci_hamiltonian(
        symbols=symbols,
        geometry=geometry,
        basis=basis,
        charge=charge,
        spin=spin,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
        level_shift=None,
        diis_space=None,
        use_newton=False,
    )
    h_qml, q_qubits = _build_pennylane_hamiltonian(
        symbols=symbols,
        geometry=geometry_bohr,
        basis=basis,
        charge=charge,
        spin=spin,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
    )

    assert n_qubits == q_qubits

    evals_pyscf = _spectrum(h_pyscf, n_qubits, n_vals=2)
    evals_qml = _spectrum(h_qml, n_qubits, n_vals=2)

    assert np.allclose(evals_pyscf, evals_qml, atol=1e-6)
