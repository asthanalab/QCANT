"""Tests for the GCIM implementation."""

from __future__ import annotations

import os

import numpy as np
import pytest

from QCANT.gcim import gcim
import QCANT


def test_gcim_h2_runs_and_returns_adapt_style_outputs():
    """GCIM should run on H2 and return (params, excitations, energies)."""
    symbols = ["H", "H"]
    geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]], dtype=float)

    params, excitations, energies = gcim(
        symbols=symbols,
        geometry=geometry,
        adapt_it=3,
        basis="sto-3g",
        charge=0,
        spin=0,
        active_electrons=2,
        active_orbitals=2,
        device_name="default.qubit",
        device_kwargs={},
        print_progress=False,
    )

    assert isinstance(excitations, list)
    assert isinstance(energies, list)
    assert len(params) == len(excitations) == len(energies) == 3
    assert all(np.isfinite(float(e)) for e in energies)


@pytest.mark.parametrize("pool_type", ["sd", "singlet_sd", "gsd"])
def test_gcim_h2_pool_types_run(pool_type: str):
    """GCIM should accept all supported pool types."""
    symbols = ["H", "H"]
    geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]], dtype=float)
    expected_pool_sizes = {"sd": 3, "singlet_sd": 2, "gsd": 4}

    params, excitations, energies = gcim(
        symbols=symbols,
        geometry=geometry,
        adapt_it=1,  # ignored: iterations are hardcoded to full pool size.
        basis="sto-3g",
        charge=0,
        spin=0,
        active_electrons=2,
        active_orbitals=2,
        device_name="default.qubit",
        pool_type=pool_type,
        print_progress=False,
    )

    assert len(params) == len(excitations) == len(energies) == expected_pool_sizes[pool_type]
    assert all(np.isfinite(float(e)) for e in energies)


def test_gcim_invalid_shots_raises():
    """GCIM currently requires analytic statevectors and should reject shots."""
    symbols = ["H", "H"]
    geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]], dtype=float)

    with pytest.raises(ValueError, match="gcim currently requires analytic execution"):
        gcim(
            symbols=symbols,
            geometry=geometry,
            adapt_it=1,
            basis="sto-3g",
            charge=0,
            spin=0,
            active_electrons=2,
            active_orbitals=2,
            shots=1000,
            print_progress=False,
        )


@pytest.mark.skipif(
    os.environ.get("QCANT_RUN_SLOW_GCIM", "0") != "1",
    reason="Set QCANT_RUN_SLOW_GCIM=1 to run the 50-iteration regression check.",
)
def test_gcim_matches_adapt_vqe_h2_after_50_iterations():
    """Slow regression: GCIM final energy should match ADAPT-VQE on H2 (50 iters)."""
    symbols = ["H", "H"]
    geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]], dtype=float)

    _params_adapt, _exc_adapt, energies_adapt = QCANT.adapt_vqe(
        symbols=symbols,
        geometry=geometry,
        adapt_it=50,
        basis="sto-3g",
        charge=0,
        spin=0,
        active_electrons=2,
        active_orbitals=2,
        device_name="default.qubit",
        optimizer_method="BFGS",
        optimizer_maxiter=200,
    )

    _params_gcim, _exc_gcim, energies_gcim = gcim(
        symbols=symbols,
        geometry=geometry,
        adapt_it=50,
        basis="sto-3g",
        charge=0,
        spin=0,
        active_electrons=2,
        active_orbitals=2,
        device_name="default.qubit",
        print_progress=False,
    )

    assert len(energies_adapt) == 50
    assert len(energies_gcim) == 50
    assert abs(float(energies_gcim[-1]) - float(energies_adapt[-1])) < 1e-3
