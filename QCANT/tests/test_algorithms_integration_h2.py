"""Integration test: ADAPT-VQE output feeds qscEOM.

This test is optional and will be skipped unless heavy scientific dependencies
are installed (PySCF, PennyLane, SciPy, NumPy).

It validates that:
- QCANT.adapt_vqe runs for H2 at 1.5 Å with STO-3G,
- QCANT.qscEOM can accept the (params, ash_excitation, energies) tuple returned
  by adapt_vqe via the `ansatz=` argument.
"""

from __future__ import annotations

import pytest


def test_adapt_vqe_output_runs_qsceom_h2_sto3g_1p5a():
    import numpy as np
    import scipy  # noqa: F401
    import pennylane  # noqa: F401
    import pyscf  # noqa: F401

    import QCANT

    symbols = ["H", "H"]
    geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]])

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
    )

    assert len(energies) == 1
    assert len(params) == len(ash_excitation)

    values = QCANT.qscEOM(
        symbols=symbols,
        geometry=geometry,
        active_electrons=2,
        active_orbitals=2,
        charge=0,
        ansatz=(params, ash_excitation, energies),
        basis="sto-3g",
        method="pyscf",
        shots=0,
    )

    assert isinstance(values, list)
    assert len(values) == 1
    assert np.all(np.isfinite(values[0]))
    assert values[0].size > 0
