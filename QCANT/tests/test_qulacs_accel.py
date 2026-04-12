"""Tests for the optional Qulacs-accelerated QCANT routines."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("qulacs")

import QCANT


def test_qkud_qulacs_matches_reference_h2():
    symbols = ["H", "H"]
    geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]], dtype=float)

    ref_energies, ref_basis = QCANT.qkud(
        symbols,
        geometry,
        n_steps=1,
        epsilon=0.1,
        active_electrons=2,
        active_orbitals=2,
        basis="sto-3g",
        charge=0,
        spin=0,
    )
    qulacs_energies, qulacs_basis = QCANT.qkud_qulacs(
        symbols,
        geometry,
        n_steps=1,
        epsilon=0.1,
        active_electrons=2,
        active_orbitals=2,
        basis="sto-3g",
        charge=0,
        spin=0,
    )

    np.testing.assert_allclose(qulacs_energies, ref_energies, atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(qulacs_basis, ref_basis, atol=1e-10, rtol=0.0)


def test_qrte_qulacs_matches_reference_h2():
    symbols = ["H", "H"]
    geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]], dtype=float)

    ref_energies, ref_basis, ref_times = QCANT.qrte(
        symbols,
        geometry,
        delta_t=0.1,
        n_steps=1,
        active_electrons=2,
        active_orbitals=2,
        basis="sto-3g",
        charge=0,
        spin=0,
    )
    qulacs_energies, qulacs_basis, qulacs_times = QCANT.qrte_qulacs(
        symbols,
        geometry,
        delta_t=0.1,
        n_steps=1,
        active_electrons=2,
        active_orbitals=2,
        basis="sto-3g",
        charge=0,
        spin=0,
    )

    np.testing.assert_allclose(qulacs_energies, ref_energies, atol=1e-9, rtol=0.0)
    np.testing.assert_allclose(qulacs_times, ref_times, atol=0.0, rtol=0.0)
    assert qulacs_basis.shape == ref_basis.shape
    np.testing.assert_allclose(np.linalg.norm(qulacs_basis, axis=1), 1.0, atol=1e-10, rtol=0.0)


def test_qrte_pmte_qulacs_runs_h2():
    symbols = ["H", "H"]
    geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]], dtype=float)

    energies, basis_states, times = QCANT.qrte_pmte_qulacs(
        symbols,
        geometry,
        delta_t=0.1,
        n_steps=1,
        active_electrons=2,
        active_orbitals=2,
        basis="sto-3g",
        charge=0,
        spin=0,
    )

    assert basis_states.shape == (3, 16)
    assert times.tolist() == [0.0, 0.1, -0.1]
    assert np.all(np.isfinite(energies))


def test_adapt_vqe_qulacs_matches_reference_h2():
    symbols = ["H", "H"]
    geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]], dtype=float)

    ref_params, ref_exc, ref_energies = QCANT.adapt_vqe(
        symbols,
        geometry,
        adapt_it=1,
        basis="sto-3g",
        charge=0,
        spin=0,
        active_electrons=2,
        active_orbitals=2,
        optimizer_maxiter=10,
    )
    qulacs_params, qulacs_exc, qulacs_energies = QCANT.adapt_vqe_qulacs(
        symbols,
        geometry,
        adapt_it=1,
        basis="sto-3g",
        charge=0,
        spin=0,
        active_electrons=2,
        active_orbitals=2,
        optimizer_maxiter=10,
        parallel_gradients=True,
        max_workers=2,
    )

    assert qulacs_exc == ref_exc
    np.testing.assert_allclose(qulacs_params, ref_params, atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(qulacs_energies, ref_energies, atol=1e-9, rtol=0.0)


def test_adapt_vqe_qulacs_sampled_qe_runs_h2():
    symbols = ["H", "H"]
    geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]], dtype=float)

    qulacs_params, qulacs_exc, qulacs_energies = QCANT.adapt_vqe_qulacs(
        symbols,
        geometry,
        adapt_it=1,
        basis="sto-3g",
        charge=0,
        spin=0,
        active_electrons=2,
        active_orbitals=2,
        optimizer_maxiter=5,
        pool_type="qe",
        pool_sample_size=1,
        pool_seed=3,
        hamiltonian_source="molecular",
        parallel_gradients=False,
    )

    assert len(qulacs_params) == 1
    assert len(qulacs_exc) == 1
    assert len(qulacs_energies) == 1
    assert np.all(np.isfinite(qulacs_params))
    assert np.all(np.isfinite(qulacs_energies))


def test_adapt_krylov_qulacs_matches_reference_h2():
    symbols = ["H", "H"]
    geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]], dtype=float)

    _ref_params, ref_exc, ref_energies, ref_details = QCANT.adaptKrylov(
        symbols=symbols,
        geometry=geometry,
        adapt_it=1,
        basis="sto-3g",
        charge=0,
        spin=0,
        active_electrons=2,
        active_orbitals=2,
        optimizer_maxiter=10,
        backend="pennylane",
    )
    qulacs_params, qulacs_exc, qulacs_energies, qulacs_details = QCANT.adaptKrylov(
        symbols=symbols,
        geometry=geometry,
        adapt_it=1,
        basis="sto-3g",
        charge=0,
        spin=0,
        active_electrons=2,
        active_orbitals=2,
        optimizer_maxiter=10,
        backend="qulacs",
        max_workers=2,
    )

    assert qulacs_exc == ref_exc
    assert qulacs_details["backend"] == "qulacs"
    np.testing.assert_allclose(qulacs_params, _ref_params, atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(qulacs_energies, ref_energies, atol=1e-9, rtol=0.0)
    np.testing.assert_allclose(
        qulacs_details["krylov_order1_energies"],
        ref_details["krylov_order1_energies"],
        atol=1e-9,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        qulacs_details["krylov_order2_energies"],
        ref_details["krylov_order2_energies"],
        atol=1e-9,
        rtol=0.0,
    )


def test_cvqe_qulacs_matches_reference_h2():
    symbols = ["H", "H"]
    geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]], dtype=float)

    ref_params, ref_dets, ref_energies, ref_details = QCANT.cvqe(
        symbols=symbols,
        geometry=geometry,
        adapt_it=1,
        basis="sto-3g",
        charge=0,
        spin=0,
        active_electrons=2,
        active_orbitals=2,
        shots=0,
        optimizer_maxiter=10,
        print_progress=False,
        return_details=True,
    )
    qulacs_params, qulacs_dets, qulacs_energies, qulacs_details = QCANT.cvqe_qulacs(
        symbols=symbols,
        geometry=geometry,
        adapt_it=1,
        basis="sto-3g",
        charge=0,
        spin=0,
        active_electrons=2,
        active_orbitals=2,
        shots=0,
        optimizer_maxiter=10,
        print_progress=False,
        return_details=True,
    )

    assert qulacs_dets == ref_dets
    assert qulacs_details["backend"] == "qulacs"
    assert qulacs_params.shape == ref_params.shape
    assert np.all(np.isfinite(qulacs_params))
    np.testing.assert_allclose(qulacs_energies, ref_energies, atol=5e-5, rtol=0.0)
