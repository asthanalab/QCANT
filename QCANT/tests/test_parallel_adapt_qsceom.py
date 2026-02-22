"""Regression tests for optional ADAPT-VQE/qscEOM parallel execution paths."""

from __future__ import annotations

import pytest

import QCANT


def test_adapt_vqe_rejects_invalid_parallel_knobs():
    symbols = ["H", "H"]
    geometry = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]

    with pytest.raises(ValueError, match=r"max_workers must be > 0"):
        QCANT.adapt_vqe(
            symbols=symbols,
            geometry=geometry,
            adapt_it=1,
            active_electrons=2,
            active_orbitals=2,
            max_workers=0,
        )

    with pytest.raises(ValueError, match=r"gradient_chunk_size must be > 0"):
        QCANT.adapt_vqe(
            symbols=symbols,
            geometry=geometry,
            adapt_it=1,
            active_electrons=2,
            active_orbitals=2,
            gradient_chunk_size=0,
        )

    with pytest.raises(ValueError, match=r"parallel_backend must be one of"):
        QCANT.adapt_vqe(
            symbols=symbols,
            geometry=geometry,
            adapt_it=1,
            active_electrons=2,
            active_orbitals=2,
            parallel_backend="bad_backend",
        )


def test_qsceom_rejects_invalid_parallel_knobs():
    symbols = ["H", "H"]
    geometry = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]

    with pytest.raises(ValueError, match=r"max_workers must be > 0"):
        QCANT.qscEOM(
            symbols=symbols,
            geometry=geometry,
            active_electrons=2,
            active_orbitals=2,
            charge=0,
            params=[0.0],
            ash_excitation=[[0, 1]],
            max_workers=0,
        )

    with pytest.raises(ValueError, match=r"matrix_chunk_size must be > 0"):
        QCANT.qscEOM(
            symbols=symbols,
            geometry=geometry,
            active_electrons=2,
            active_orbitals=2,
            charge=0,
            params=[0.0],
            ash_excitation=[[0, 1]],
            matrix_chunk_size=0,
        )

    with pytest.raises(ValueError, match=r"parallel_backend must be one of"):
        QCANT.qscEOM(
            symbols=symbols,
            geometry=geometry,
            active_electrons=2,
            active_orbitals=2,
            charge=0,
            params=[0.0],
            ash_excitation=[[0, 1]],
            parallel_backend="bad_backend",
        )


def test_adapt_vqe_parallel_gradients_matches_serial_h2():
    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    pytest.importorskip("pennylane")
    pytest.importorskip("pyscf")

    symbols = ["H", "H"]
    geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]], dtype=float)
    kwargs = dict(
        symbols=symbols,
        geometry=geometry,
        adapt_it=1,
        basis="sto-3g",
        charge=0,
        spin=0,
        active_electrons=2,
        active_orbitals=2,
        device_name="default.qubit",
        optimizer_method="BFGS",
        optimizer_maxiter=25,
    )

    params_serial, exc_serial, energies_serial = QCANT.adapt_vqe(**kwargs)
    params_parallel, exc_parallel, energies_parallel = QCANT.adapt_vqe(
        **kwargs,
        parallel_gradients=True,
        max_workers=2,
        gradient_chunk_size=1,
    )

    assert exc_parallel == exc_serial
    assert len(energies_parallel) == len(energies_serial) == 1
    assert np.allclose(np.asarray(energies_parallel), np.asarray(energies_serial), atol=1e-8, rtol=1e-8)
    assert np.allclose(np.asarray(params_parallel), np.asarray(params_serial), atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("symmetric", [True, False])
def test_qsceom_parallel_matrix_matches_serial_h2(symmetric: bool):
    np = pytest.importorskip("numpy")
    pytest.importorskip("pennylane")
    pytest.importorskip("pyscf")

    symbols = ["H", "H"]
    geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]], dtype=float)
    kwargs = dict(
        symbols=symbols,
        geometry=geometry,
        active_electrons=2,
        active_orbitals=2,
        charge=0,
        params=np.array([0.0]),
        ash_excitation=[[0, 1]],
        basis="sto-3g",
        method="pyscf",
        shots=0,
        symmetric=symmetric,
        max_states=4,
        state_seed=13,
    )

    values_serial = QCANT.qscEOM(**kwargs)
    values_parallel = QCANT.qscEOM(
        **kwargs,
        parallel_matrix=True,
        max_workers=2,
        matrix_chunk_size=1,
    )

    assert isinstance(values_parallel, list)
    assert len(values_parallel) == len(values_serial) == 1
    assert np.allclose(np.asarray(values_parallel[0]), np.asarray(values_serial[0]), atol=1e-8, rtol=1e-8)
