"""Tests for GPU backend controls and benchmark artifact generation."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
import warnings

import numpy as np
import pytest

import QCANT
import QCANT._accelerator as accelerator


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_SCRIPT = REPO_ROOT / "scripts" / "benchmark_gpu_speedups.py"


def _load_benchmark_module():
    spec = importlib.util.spec_from_file_location("benchmark_gpu_speedups", BENCHMARK_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_normalize_array_backend_rejects_invalid_value():
    with pytest.raises(ValueError, match=r"array_backend must be one of"):
        accelerator.normalize_array_backend("not-a-backend")


def test_resolve_gpu_parallelism_clamps_process_pool_to_single_worker():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        worker_count, backend = accelerator.resolve_gpu_parallelism(
            device_name="lightning.gpu",
            worker_count=4,
            parallel_backend="process",
            context="test",
        )

    assert worker_count == 1
    assert backend == "thread"
    messages = [str(item.message) for item in caught]
    assert any("Downgrading parallel_backend='process' to 'thread'" in message for message in messages)
    assert any("Clamping max_workers from 4 to 1" in message for message in messages)


def test_resolve_array_module_auto_gpu_request_falls_back_when_cupy_missing(monkeypatch):
    def _raise_import_error():
        raise ImportError("missing cupy")

    monkeypatch.setattr(accelerator, "import_cupy", _raise_import_error)

    with pytest.warns(RuntimeWarning, match=r"CuPy is not installed"):
        xp, backend_name, using_gpu = accelerator.resolve_array_module(
            array_backend="auto",
            device_name="lightning.gpu",
            context="test",
        )

    assert xp.__name__ == "numpy"
    assert backend_name == "numpy"
    assert using_gpu is False


def test_qkud_rejects_sparse_cupy_path():
    pytest.importorskip("pennylane")
    pytest.importorskip("pyscf")

    symbols = ["H", "H"]
    geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]], dtype=float)

    with pytest.raises(ValueError, match=r"QKUD sparse execution remains CPU-only"):
        QCANT.qkud(
            symbols=symbols,
            geometry=geometry,
            n_steps=1,
            epsilon=0.1,
            active_electrons=2,
            active_orbitals=2,
            use_sparse=True,
            array_backend="cupy",
        )


def test_exact_krylov_rejects_sparse_cupy_path():
    pytest.importorskip("pennylane")
    pytest.importorskip("pyscf")

    symbols = ["H", "H"]
    geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]], dtype=float)

    with pytest.raises(ValueError, match=r"Exact Krylov sparse execution remains CPU-only"):
        QCANT.exact_krylov(
            symbols=symbols,
            geometry=geometry,
            n_steps=1,
            active_electrons=2,
            active_orbitals=2,
            use_sparse=True,
            array_backend="cupy",
        )


def test_qsceom_sparse_projector_gpu_request_warns_and_runs_on_cpu():
    pytest.importorskip("pennylane")
    pytest.importorskip("pyscf")
    pytest.importorskip("openfermion")

    geometry = QCANT.geometry_to_bohr(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]], dtype=float))

    with pytest.warns(RuntimeWarning, match=r"sparse_number_preserving projector runs on CPU"):
        values = QCANT.qscEOM(
            symbols=["H", "H"],
            geometry=geometry,
            active_electrons=2,
            active_orbitals=2,
            charge=0,
            params=np.asarray([], dtype=float),
            ash_excitation=[],
            basis="sto-3g",
            method="pyscf",
            shots=0,
            device_name="lightning.gpu",
            projector_backend="sparse_number_preserving",
        )

    assert isinstance(values, list)
    assert len(values) == 1
    assert np.all(np.isfinite(values[0]))


def test_standalone_tepid_adapt_accepts_numpy_array_backend():
    pytest.importorskip("pennylane")
    pytest.importorskip("pyscf")

    from standalone.tepid_qsceom import tepid_adapt as standalone_tepid_adapt

    symbols = ["H", "H"]
    geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]], dtype=float)

    params, excitations, free_energies, details = standalone_tepid_adapt(
        symbols=symbols,
        geometry=geometry,
        adapt_it=1,
        beta=2.0,
        basis="sto-3g",
        charge=0,
        active_electrons=2,
        active_orbitals=2,
        optimizer_maxiter=12,
        array_backend="numpy",
        return_details=True,
    )

    assert len(params) == len(excitations) == len(free_energies) == 1
    assert details["array_backend"] == "numpy"
    assert np.isfinite(details["final_free_energy"])


def test_standalone_cli_override_threads_array_backend():
    from standalone.tepid_qsceom.core import _apply_cli_overrides, build_arg_parser

    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--config",
            "standalone/tepid_qsceom/configs/h2_sto3g.json",
            "--mode",
            "tepid_qsceom",
            "--array-backend",
            "numpy",
        ]
    )
    updated = _apply_cli_overrides({"tepid": {}, "qsceom": {}}, args)

    assert updated["array_backend"] == "numpy"


def test_gpu_benchmark_writes_csv_and_plot_artifacts(tmp_path):
    pytest.importorskip("matplotlib")

    benchmark = _load_benchmark_module()
    rows = [
        {
            "algorithm": "adapt_vqe",
            "backend": "cpu",
            "accelerator_kind": "cpu",
            "problem_profile": "h6",
            "basis": "sto-3g",
            "active_electrons": 6,
            "active_orbitals": 6,
            "n_qubits": 12,
            "median_runtime_s": 4.0,
            "stddev_runtime_s": 0.1,
            "speedup_vs_cpu": 1.0,
            "repeats": 2,
            "cpu_threads": 1,
            "cuda_visible_devices": "0",
            "status": "ok",
            "note": "",
        },
        {
            "algorithm": "adapt_vqe",
            "backend": "gpu",
            "accelerator_kind": "gpu",
            "problem_profile": "h6",
            "basis": "sto-3g",
            "active_electrons": 6,
            "active_orbitals": 6,
            "n_qubits": 12,
            "median_runtime_s": 2.0,
            "stddev_runtime_s": 0.1,
            "speedup_vs_cpu": 2.0,
            "repeats": 2,
            "cpu_threads": 1,
            "cuda_visible_devices": "0",
            "status": "ok",
            "note": "",
        },
        {
            "algorithm": "exact_krylov",
            "backend": "cpu",
            "accelerator_kind": "cpu",
            "problem_profile": "h6",
            "basis": "sto-3g",
            "active_electrons": 6,
            "active_orbitals": 6,
            "n_qubits": 12,
            "median_runtime_s": 3.0,
            "stddev_runtime_s": 0.2,
            "speedup_vs_cpu": 1.0,
            "repeats": 2,
            "cpu_threads": 1,
            "cuda_visible_devices": "0",
            "status": "ok",
            "note": "",
        },
        {
            "algorithm": "exact_krylov",
            "backend": "gpu",
            "accelerator_kind": "gpu",
            "problem_profile": "h6",
            "basis": "sto-3g",
            "active_electrons": 6,
            "active_orbitals": 6,
            "n_qubits": 12,
            "median_runtime_s": 1.5,
            "stddev_runtime_s": 0.1,
            "speedup_vs_cpu": 2.0,
            "repeats": 2,
            "cpu_threads": 1,
            "cuda_visible_devices": "0",
            "status": "ok",
            "note": "",
        },
    ]

    csv_path = tmp_path / "h6_speedups.csv"
    speedup_path = tmp_path / "h6_speedup_summary.png"
    runtime_path = tmp_path / "h6_runtime_summary.png"
    plots_dir = tmp_path / "plots"

    benchmark.write_csv(rows, csv_path)
    benchmark.write_speedup_plot(rows, speedup_path)
    benchmark.write_runtime_plot(rows, runtime_path)
    per_algorithm_plots = benchmark.write_per_algorithm_speedup_plots(rows, plots_dir)

    assert csv_path.exists()
    assert speedup_path.exists()
    assert runtime_path.exists()
    assert len(per_algorithm_plots) == 2
    assert (plots_dir / "adapt_vqe_speedup.png").exists()
    assert (plots_dir / "exact_krylov_speedup.png").exists()

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        written_rows = list(csv.DictReader(handle))

    assert len(written_rows) == len(rows)
    assert any(
        row["algorithm"] == "adapt_vqe" and row["backend"] == "gpu" and row["speedup_vs_cpu"] == "2.0"
        for row in written_rows
    )
    assert written_rows[0]["problem_profile"] == "h6"
    assert written_rows[0]["cpu_threads"] == "1"
