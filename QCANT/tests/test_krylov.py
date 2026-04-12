"""
Unit tests for the krylov module.
"""

import numpy as np
import pytest

import QCANT


def test_exact_krylov_lanczos():
    """Test the exact_krylov function with the lanczos method."""
    symbols = ["H", "H"]
    geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]])

    try:
        energies, basis_states = QCANT.exact_krylov(
            symbols=symbols,
            geometry=geometry,
            n_steps=10,
            active_electrons=2,
            active_orbitals=2,
            basis="sto-3g",
            krylov_method="lanczos",
        )
        assert energies.ndim == 1
        assert energies.size >= 1
        assert np.all(np.isfinite(energies))
        assert basis_states.shape[0] == 11
    except ValueError as e:
        assert "Lanczos breakdown" in str(e)


def test_exact_krylov_no_history():
    """Test the exact_krylov function without returning the energy history."""
    symbols = ["H", "H"]
    geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]])

    result = QCANT.exact_krylov(
        symbols=symbols,
        geometry=geometry,
        n_steps=2,
        active_electrons=2,
        active_orbitals=2,
        basis="sto-3g",
        return_min_energy_history=False,
    )
    assert len(result) == 2


def test_exact_krylov_sparse():
    """Test the exact_krylov function with sparse matrices."""
    symbols = ["H", "H"]
    geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]])

    energies, _, _ = QCANT.exact_krylov(
        symbols=symbols,
        geometry=geometry,
        n_steps=2,
        active_electrons=2,
        active_orbitals=2,
        basis="sto-3g",
        use_sparse=True,
        return_min_energy_history=True,
    )
    assert energies.ndim == 1
    assert energies.size >= 1


def test_exact_krylov_initial_state():
    """Test the exact_krylov function with a provided initial state."""
    symbols = ["H", "H"]
    geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]])
    n_qubits = 4
    initial_state = np.zeros(2**n_qubits)
    initial_state[0] = 1.0

    energies, _, _ = QCANT.exact_krylov(
        symbols=symbols,
        geometry=geometry,
        n_steps=2,
        active_electrons=2,
        active_orbitals=2,
        basis="sto-3g",
        initial_state=initial_state,
        return_min_energy_history=True,
    )
    assert energies.ndim == 1
    assert energies.size >= 1


def test_exact_krylov_basis_threshold():
    """Test the exact_krylov function with a basis threshold."""
    symbols = ["H", "H"]
    geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]])

    energies, _, _ = QCANT.exact_krylov(
        symbols=symbols,
        geometry=geometry,
        n_steps=2,
        active_electrons=2,
        active_orbitals=2,
        basis="sto-3g",
        basis_threshold=1e-6,
        return_min_energy_history=True,
    )
    assert energies.ndim == 1
    assert energies.size >= 1


def test_exact_krylov_invalid_input():
    """Test error handling for invalid inputs."""
    with pytest.raises(ValueError, match="symbols must be non-empty"):
        QCANT.exact_krylov(symbols=[], geometry=[], n_steps=1, active_electrons=1, active_orbitals=1)

    with pytest.raises(ValueError, match="n_steps must be >= 0"):
        QCANT.exact_krylov(symbols=["H"], geometry=[[0, 0, 0]], n_steps=-1, active_electrons=1, active_orbitals=1)

    with pytest.raises(ValueError, match="overlap_tol must be > 0"):
        QCANT.exact_krylov(
            symbols=["H"],
            geometry=[[0, 0, 0]],
            n_steps=1,
            active_electrons=1,
            active_orbitals=1,
            overlap_tol=0,
        )

    with pytest.raises(ValueError, match='krylov_method must be "exact" or "lanczos"'):
        QCANT.exact_krylov(
            symbols=["H"],
            geometry=[[0, 0, 0]],
            n_steps=1,
            active_electrons=1,
            active_orbitals=1,
            krylov_method="invalid",
        )
