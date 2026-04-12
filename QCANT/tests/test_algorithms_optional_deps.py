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


def test_qsceom_rejects_brg_tolerance_for_shot_mode():
    symbols = ["H", "H"]
    geometry = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]

    with pytest.raises(ValueError, match=r"brg_tolerance requires shots=0"):
        QCANT.qscEOM(
            symbols=symbols,
            geometry=geometry,
            active_electrons=2,
            active_orbitals=2,
            charge=0,
            params=[0.0],
            ash_excitation=[[0, 1]],
            shots=100,
            brg_tolerance=1e-6,
        )


def test_qsceom_rejects_sparse_projector_for_shot_mode():
    symbols = ["H", "H"]
    geometry = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]

    with pytest.raises(ValueError, match=r"projector_backend='sparse_number_preserving' requires shots=0"):
        QCANT.qscEOM(
            symbols=symbols,
            geometry=geometry,
            active_electrons=2,
            active_orbitals=2,
            charge=0,
            params=[0.0],
            ash_excitation=[[0, 1]],
            shots=100,
            projector_backend="sparse_number_preserving",
        )


def test_tepid_adapt_requires_exactly_one_temperature_parameter():
    symbols = ["H", "H"]
    geometry = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]

    with pytest.raises(ValueError, match=r"Provide exactly one of temperature or beta"):
        QCANT.tepid_adapt(
            symbols=symbols,
            geometry=geometry,
            adapt_it=1,
            active_electrons=2,
            active_orbitals=2,
            temperature=1.0,
            beta=1.0,
        )


def test_tepid_adapt_rejects_nonzero_shots():
    symbols = ["H", "H"]
    geometry = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]

    with pytest.raises(ValueError, match=r"shots=0"):
        QCANT.tepid_adapt(
            symbols=symbols,
            geometry=geometry,
            adapt_it=1,
            active_electrons=2,
            active_orbitals=2,
            beta=1.0,
            shots=100,
        )


def test_tepid_boltzmann_weights_normalize():
    np = pytest.importorskip("numpy")

    weights = QCANT.tepid_boltzmann_weights([0.0, 1.0, 2.0], beta=2.0)

    assert weights.shape == (3,)
    assert np.isclose(np.sum(weights), 1.0)
    assert weights[0] > weights[1] > weights[2]
