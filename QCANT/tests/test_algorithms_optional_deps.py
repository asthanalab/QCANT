"""Tests for algorithm entry points.

These tests focus on argument/contract behavior that should be fast and
reliable.
"""

from __future__ import annotations

import pytest

import QCANT


def test_adapt_vqe_geometry_length_mismatch_raises_value_error():
    symbols = ["H", "H"]
    geometry = [[0.0, 0.0, 0.0]]

    with pytest.raises(ValueError, match=r"geometry must have the same length as symbols"):
        QCANT.adapt_vqe(
            symbols=symbols,
            geometry=geometry,
            adapt_it=1,
            active_electrons=2,
            active_orbitals=2,
        )


def test_qsceom_requires_ansatz_or_params_and_excitations():
    symbols = ["H", "H"]
    geometry = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]

    with pytest.raises(TypeError):
        QCANT.qscEOM(
            symbols=symbols,
            geometry=geometry,
            active_electrons=2,
            active_orbitals=2,
            charge=0,
        )


def test_qsceom_rejects_bad_ansatz_tuple():
    symbols = ["H", "H"]
    geometry = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]

    with pytest.raises(ValueError, match=r"ansatz must be a 3-tuple"):
        QCANT.qscEOM(
            symbols=symbols,
            geometry=geometry,
            active_electrons=2,
            active_orbitals=2,
            charge=0,
            ansatz=([], []),
        )
