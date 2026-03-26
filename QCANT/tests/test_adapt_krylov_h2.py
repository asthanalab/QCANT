"""Regression tests for ADAPT+Krylov post-processing."""

from __future__ import annotations

import numpy as np

import QCANT


def test_adapt_krylov_h2_reports_variational_krylov_energies():
    symbols = ["H", "H"]
    geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]], dtype=float)

    params, ash_excitation, adapt_energies, details = QCANT.adaptKrylov(
        symbols=symbols,
        geometry=geometry,
        adapt_it=2,
        basis="sto-3g",
        charge=0,
        spin=0,
        active_electrons=2,
        active_orbitals=2,
        device_name="default.qubit",
        optimizer_maxiter=25,
        hamiltonian_source="casci",
        backend="pennylane",
    )

    adapt_array = np.asarray(adapt_energies, dtype=float)
    order1 = np.asarray(details["krylov_order1_energies"], dtype=float)
    order2 = np.asarray(details["krylov_order2_energies"], dtype=float)
    history = details["history"]
    exact_ground = float(details["exact_ground_energy"])

    assert len(params) == len(ash_excitation) == 2
    assert adapt_array.shape == order1.shape == order2.shape == (2,)
    assert len(history) == 2
    assert np.all(np.isfinite(adapt_array))
    assert np.all(np.isfinite(order1))
    assert np.all(np.isfinite(order2))
    assert np.all(order1 <= adapt_array + 1e-8)
    assert np.all(order2 <= order1 + 1e-8)
    assert np.all(order2 >= exact_ground - 1e-8)
    assert details["backend"] == "pennylane"
    assert all("krylov_order1_energy" in item for item in history)
    assert all("krylov_order2_energy" in item for item in history)
