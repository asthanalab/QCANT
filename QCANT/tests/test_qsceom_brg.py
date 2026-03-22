"""Regression tests for qscEOM BRG support."""

from __future__ import annotations

import numpy as np
import pytest


def _geometry(symbol_count: int, bond_length: float):
    import QCANT

    geometry_angstrom = np.asarray([[0.0, 0.0, bond_length * idx] for idx in range(symbol_count)], dtype=float)
    return QCANT.geometry_to_bohr(geometry_angstrom)


def _roots(symbols, geometry, *, projector_backend: str, brg_tolerance=None):
    import QCANT

    kwargs = dict(
        symbols=symbols,
        geometry=geometry,
        active_electrons=len(symbols),
        active_orbitals=len(symbols),
        charge=0,
        params=np.asarray([], dtype=float),
        ash_excitation=[],
        basis="sto-3g",
        method="pyscf",
        shots=0,
        projector_backend=projector_backend,
    )
    if brg_tolerance is not None:
        kwargs["brg_tolerance"] = brg_tolerance
    values = QCANT.qscEOM(**kwargs)
    return np.asarray(values[0], dtype=float)


@pytest.mark.parametrize(
    ("symbol_count", "bond_length"),
    [(2, 0.74), (4, 1.5)],
)
def test_qsceom_dense_and_sparse_exact_match(symbol_count: int, bond_length: float):
    pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    pytest.importorskip("pennylane")
    pytest.importorskip("pyscf")
    pytest.importorskip("openfermion")

    symbols = ["H"] * symbol_count
    geometry = _geometry(symbol_count, bond_length)

    dense_roots = _roots(symbols, geometry, projector_backend="dense")
    sparse_roots = _roots(symbols, geometry, projector_backend="sparse_number_preserving")

    np.testing.assert_allclose(sparse_roots, dense_roots, atol=1e-10, rtol=0.0)


@pytest.mark.parametrize(
    ("symbol_count", "bond_length", "tolerance"),
    [(2, 0.74, 1e-10), (4, 1.5, 1e-8)],
)
def test_qsceom_dense_and_sparse_brg_match(symbol_count: int, bond_length: float, tolerance: float):
    pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    pytest.importorskip("pennylane")
    pytest.importorskip("pyscf")
    pytest.importorskip("openfermion")

    symbols = ["H"] * symbol_count
    geometry = _geometry(symbol_count, bond_length)

    dense_roots = _roots(
        symbols=symbols,
        geometry=geometry,
        projector_backend="dense",
        brg_tolerance=tolerance,
    )
    sparse_roots = _roots(
        symbols=symbols,
        geometry=geometry,
        projector_backend="sparse_number_preserving",
        brg_tolerance=tolerance,
    )

    np.testing.assert_allclose(sparse_roots, dense_roots, atol=1e-8, rtol=0.0)


def test_qsceom_sparse_brg_tight_matches_exact_h2():
    pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    pytest.importorskip("pennylane")
    pytest.importorskip("pyscf")
    pytest.importorskip("openfermion")

    import QCANT

    geometry = _geometry(2, 0.74)
    symbols = ["H", "H"]

    exact_values = QCANT.qscEOM(
        symbols=symbols,
        geometry=geometry,
        active_electrons=2,
        active_orbitals=2,
        charge=0,
        params=np.asarray([], dtype=float),
        ash_excitation=[],
        basis="sto-3g",
        method="pyscf",
        shots=0,
        projector_backend="sparse_number_preserving",
    )
    brg_values, brg_details = QCANT.qscEOM(
        symbols=symbols,
        geometry=geometry,
        active_electrons=2,
        active_orbitals=2,
        charge=0,
        params=np.asarray([], dtype=float),
        ash_excitation=[],
        basis="sto-3g",
        method="pyscf",
        shots=0,
        brg_tolerance=1e-10,
        projector_backend="sparse_number_preserving",
        return_details=True,
    )

    np.testing.assert_allclose(np.asarray(brg_values[0]), np.asarray(exact_values[0]), atol=1e-8, rtol=0.0)
    assert brg_details["brg_applied"] is True
    assert brg_details["brg_tolerance"] == pytest.approx(1e-10)
    assert isinstance(brg_details["brg_rank"], int)
    assert brg_details["brg_rank"] > 0
    assert brg_details["brg_truncation_value"] >= 0.0
    assert brg_details["projector_backend"] == "sparse_number_preserving"
    assert brg_details["sector_dimension"] == 6
    assert brg_details["hamiltonian_nnz"] > 0


def test_adapt_vqe_molecular_ansatz_runs_exact_and_brg_qsceom_h2():
    pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    pytest.importorskip("pennylane")
    pytest.importorskip("pyscf")
    pytest.importorskip("openfermion")

    import QCANT

    symbols = ["H", "H"]
    geometry = QCANT.geometry_to_bohr(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]], dtype=float))

    params, ash_excitation, energies = QCANT.adapt_vqe(
        symbols=symbols,
        geometry=geometry,
        adapt_it=1,
        basis="sto-3g",
        charge=0,
        spin=0,
        active_electrons=2,
        active_orbitals=2,
        device_name="default.qubit",
        optimizer_maxiter=25,
        hamiltonian_source="molecular",
    )

    values_exact = QCANT.qscEOM(
        symbols=symbols,
        geometry=geometry,
        active_electrons=2,
        active_orbitals=2,
        charge=0,
        ansatz=(params, ash_excitation, energies),
        basis="sto-3g",
        method="pyscf",
        shots=0,
        projector_backend="sparse_number_preserving",
    )
    values_brg, details_brg = QCANT.qscEOM(
        symbols=symbols,
        geometry=geometry,
        active_electrons=2,
        active_orbitals=2,
        charge=0,
        ansatz=(params, ash_excitation, energies),
        basis="sto-3g",
        method="pyscf",
        shots=0,
        brg_tolerance=1e-10,
        projector_backend="sparse_number_preserving",
        return_details=True,
    )

    assert np.all(np.isfinite(values_exact[0]))
    assert np.all(np.isfinite(values_brg[0]))
    np.testing.assert_allclose(np.asarray(values_brg[0]), np.asarray(values_exact[0]), atol=1e-8, rtol=0.0)
    assert details_brg["projector_backend"] == "sparse_number_preserving"
    assert details_brg["sector_dimension"] == 6
    assert details_brg["hamiltonian_nnz"] > 0
