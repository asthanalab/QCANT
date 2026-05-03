"""Tests for the simplified CVQE implementation."""

from __future__ import annotations

import json

import numpy as np

import QCANT


def _assert_nonincreasing(values, tol: float = 1e-6):
    arr = np.asarray(values, dtype=float)
    diffs = np.diff(arr)
    assert np.all(diffs <= tol), f"energies must be non-increasing, got diffs={diffs}"


def test_cvqe_h2_exact_runs_and_records_history():
    """CVQE should run on H2 with exact determinant selection."""
    symbols = ["H", "H"]
    geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]], dtype=float)

    params, determinants, energies, details = QCANT.cvqe(
        symbols=symbols,
        geometry=geometry,
        adapt_it=2,
        basis="sto-3g",
        charge=0,
        spin=0,
        active_electrons=2,
        active_orbitals=2,
        shots=0,
        optimizer_method="BFGS",
        optimizer_maxiter=20,
        selection_seed=123,
        print_progress=False,
        array_backend="numpy",
        return_details=True,
    )

    assert isinstance(determinants, list)
    assert len(params) >= len(determinants)
    assert len(determinants) == len(energies) == len(details["history"]) == 2
    assert all(np.isfinite(float(e)) for e in energies)
    assert details["initial_energy"] >= float(energies[-1]) - 1e-6
    assert details["array_backend"] == "numpy"
    assert all(item["selection_mode"] == "exact" for item in details["history"])
    _assert_nonincreasing(energies)


def test_cvqe_h2_sampled_selection_runs_with_1000_shots():
    """Shot-based determinant selection should be available for CVQE."""
    symbols = ["H", "H"]
    geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]], dtype=float)

    params, determinants, energies, details = QCANT.cvqe(
        symbols=symbols,
        geometry=geometry,
        adapt_it=1,
        basis="sto-3g",
        charge=0,
        spin=0,
        active_electrons=2,
        active_orbitals=2,
        shots=1000,
        optimizer_method="BFGS",
        optimizer_maxiter=15,
        selection_seed=7,
        print_progress=False,
        return_details=True,
    )

    assert len(params) >= 1
    assert len(determinants) == len(energies) == 1
    assert np.isfinite(float(energies[0]))
    assert details["history"][0]["selection_mode"] == "sampled"
    assert details["history"][0]["selection_shots"] == 1000


def test_cvqe_h2_checkpoint_resume_matches_fresh_run(tmp_path):
    """CVQE should checkpoint after each iteration and resume from that checkpoint."""
    symbols = ["H", "H"]
    geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]], dtype=float)
    checkpoint_path = tmp_path / "cvqe_h2_checkpoint.json"

    _, dets_step1, energies_step1, details_step1 = QCANT.cvqe(
        symbols=symbols,
        geometry=geometry,
        adapt_it=1,
        basis="sto-3g",
        charge=0,
        spin=0,
        active_electrons=2,
        active_orbitals=2,
        shots=0,
        optimizer_method="BFGS",
        optimizer_maxiter=12,
        selection_seed=19,
        print_progress=False,
        return_details=True,
        checkpoint_path=checkpoint_path,
    )

    assert checkpoint_path.exists()
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint["completed_iterations"] == 1
    assert len(checkpoint["energies"]) == len(energies_step1) == len(dets_step1) == len(details_step1["history"]) == 1

    _, resumed_dets, resumed_energies, resumed_details = QCANT.cvqe(
        symbols=symbols,
        geometry=geometry,
        adapt_it=2,
        basis="sto-3g",
        charge=0,
        spin=0,
        active_electrons=2,
        active_orbitals=2,
        shots=0,
        optimizer_method="BFGS",
        optimizer_maxiter=12,
        selection_seed=19,
        print_progress=False,
        return_details=True,
        resume_state=checkpoint,
        checkpoint_path=checkpoint_path,
    )

    _, fresh_dets, fresh_energies, fresh_details = QCANT.cvqe(
        symbols=symbols,
        geometry=geometry,
        adapt_it=2,
        basis="sto-3g",
        charge=0,
        spin=0,
        active_electrons=2,
        active_orbitals=2,
        shots=0,
        optimizer_method="BFGS",
        optimizer_maxiter=12,
        selection_seed=19,
        print_progress=False,
        return_details=True,
    )

    assert resumed_dets == fresh_dets
    assert len(resumed_energies) == len(fresh_energies) == len(resumed_details["history"]) == len(fresh_details["history"]) == 2
    np.testing.assert_allclose(resumed_energies, fresh_energies, atol=1e-8, rtol=0.0)


def test_cvqe_h2_exact_runs_with_uccsd_ansatz():
    """CVQE should support exact selection with a jointly optimized UCCSD ansatz."""
    symbols = ["H", "H"]
    geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]], dtype=float)

    params, determinants, energies, details = QCANT.cvqe(
        symbols=symbols,
        geometry=geometry,
        adapt_it=1,
        ansatz="uccsd",
        basis="sto-3g",
        charge=0,
        spin=0,
        active_electrons=2,
        active_orbitals=2,
        shots=0,
        optimizer_method="BFGS",
        optimizer_maxiter=8,
        selection_seed=11,
        print_progress=False,
        return_details=True,
    )

    assert len(params) >= len(determinants)
    assert len(determinants) == len(energies) == len(details["history"]) == 1
    assert details["ansatz"] == "uccsd"
    assert "uccsd_weights" in details["history"][0]
    assert all(np.isfinite(float(e)) for e in energies)
    _assert_nonincreasing(energies, tol=1e-6)


def test_cvqe_h4_exact_runs_with_pt2_selection():
    """CVQE should run on a linear H4 chain with PT2-screened exact selection."""
    symbols = ["H", "H", "H", "H"]
    geometry = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.5],
            [0.0, 0.0, 3.0],
            [0.0, 0.0, 4.5],
        ],
        dtype=float,
    )

    params, determinants, energies, details = QCANT.cvqe(
        symbols=symbols,
        geometry=geometry,
        adapt_it=2,
        basis="sto-3g",
        charge=0,
        spin=0,
        active_electrons=4,
        active_orbitals=4,
        shots=0,
        optimizer_method="BFGS",
        optimizer_maxiter=10,
        selection_method="pt2",
        selection_topk=10,
        selection_seed=11,
        print_progress=False,
        return_details=True,
    )

    assert len(params) >= len(determinants)
    assert len(determinants) == len(energies) == len(details["history"]) == 2
    assert all(np.isfinite(float(e)) for e in energies)
    assert all(item["selection_strategy"] == "pt2" for item in details["history"])
    assert all(1 <= int(item["screened_candidate_count"]) <= 10 for item in details["history"])
    _assert_nonincreasing(energies, tol=1e-5)
