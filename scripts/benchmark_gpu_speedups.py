"""Benchmark QCANT CPU, GPU, and Qulacs speedups.

The default profile is a practical H6/STO-3G smoke benchmark intended for one
Talon GPU allocation. Each timed sample runs in a fresh subprocess so that
device construction, CUDA context setup, and large arrays do not leak between
backends.
"""

import argparse
import csv
from dataclasses import dataclass
import math
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import time
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_EXTRA_SYSPATH = os.environ.get("QCANT_BENCH_EXTRA_PATH", "").strip()
if _EXTRA_SYSPATH:
    for entry in _EXTRA_SYSPATH.split(os.pathsep):
        if entry and entry not in sys.path:
            sys.path.append(entry)

import matplotlib.pyplot as plt
import numpy as np

import QCANT
from standalone.tepid_qsceom import (
    qsceom as standalone_qsceom,
    tepid_adapt as standalone_tepid_adapt,
    tepid_qsceom as standalone_tepid_qsceom,
)


CPU_BACKEND = "cpu"
GPU_BACKEND = "gpu"
QULACS_BACKEND = "qulacs_cpu"
DEFAULT_CPU_THREADS = 1


@dataclass(frozen=True)
class Profile:
    """Problem profile shared by all benchmark callables."""

    name: str
    symbols: list[str]
    geometry: np.ndarray
    basis: str
    active_electrons: int
    active_orbitals: int
    spacing_angstrom: float | None = None

    @property
    def n_qubits(self) -> int:
        return int(2 * self.active_orbitals)


@dataclass(frozen=True)
class AlgorithmSpec:
    """Benchmark registry entry."""

    name: str
    backends: tuple[str, ...]
    runner: Callable[[str, Profile, str, str], None]
    note: str = ""


@dataclass(frozen=True)
class BenchmarkResult:
    """Timing result for one algorithm/backend pair."""

    algorithm: str
    backend: str
    accelerator_kind: str
    profile: Profile
    median_runtime_s: float | None
    stddev_runtime_s: float | None
    speedup_vs_cpu: float | None
    repeats: int
    cpu_threads: int
    cuda_visible_devices: str
    status: str
    note: str = ""

    def as_row(self) -> dict[str, str | float | int]:
        return {
            "algorithm": self.algorithm,
            "backend": self.backend,
            "accelerator_kind": self.accelerator_kind,
            "problem_profile": self.profile.name,
            "basis": self.profile.basis,
            "active_electrons": self.profile.active_electrons,
            "active_orbitals": self.profile.active_orbitals,
            "n_qubits": self.profile.n_qubits,
            "median_runtime_s": "" if self.median_runtime_s is None else round(self.median_runtime_s, 6),
            "stddev_runtime_s": "" if self.stddev_runtime_s is None else round(self.stddev_runtime_s, 6),
            "speedup_vs_cpu": "" if self.speedup_vs_cpu is None else round(self.speedup_vs_cpu, 6),
            "repeats": self.repeats,
            "cpu_threads": self.cpu_threads,
            "cuda_visible_devices": self.cuda_visible_devices,
            "status": self.status,
            "note": self.note,
        }


CSV_FIELDNAMES = [
    "algorithm",
    "backend",
    "accelerator_kind",
    "problem_profile",
    "basis",
    "active_electrons",
    "active_orbitals",
    "n_qubits",
    "median_runtime_s",
    "stddev_runtime_s",
    "speedup_vs_cpu",
    "repeats",
    "cpu_threads",
    "cuda_visible_devices",
    "status",
    "note",
]


def _set_thread_limits(cpu_threads: int = DEFAULT_CPU_THREADS) -> None:
    value = str(int(cpu_threads))
    os.environ["OMP_NUM_THREADS"] = value
    os.environ["OPENBLAS_NUM_THREADS"] = value
    os.environ["MKL_NUM_THREADS"] = value
    os.environ["VECLIB_MAXIMUM_THREADS"] = value
    os.environ["NUMEXPR_NUM_THREADS"] = value


def _time_once(fn: Callable[[], None]) -> float:
    start = time.perf_counter()
    fn()
    return float(time.perf_counter() - start)


def _summarize(samples: list[float]) -> tuple[float, float]:
    if len(samples) == 1:
        return float(samples[0]), 0.0
    return float(statistics.median(samples)), float(statistics.pstdev(samples))


def _linear_h_chain(n_atoms: int, spacing: float) -> tuple[list[str], np.ndarray]:
    symbols = ["H"] * int(n_atoms)
    geometry = np.asarray([[0.0, 0.0, float(i) * float(spacing)] for i in range(int(n_atoms))], dtype=float)
    return symbols, geometry


def _profile_geometry(profile: str, spacing: float) -> Profile:
    if profile == "small":
        symbols, geometry = _linear_h_chain(2, 1.5)
        return Profile(
            name="small",
            symbols=symbols,
            geometry=geometry,
            basis="sto-3g",
            active_electrons=2,
            active_orbitals=2,
            spacing_angstrom=1.5,
        )
    if profile == "large":
        symbols, geometry = _linear_h_chain(4, 0.9)
        return Profile(
            name="large",
            symbols=symbols,
            geometry=geometry,
            basis="sto-3g",
            active_electrons=4,
            active_orbitals=4,
            spacing_angstrom=0.9,
        )
    if profile == "h6":
        symbols, geometry = _linear_h_chain(6, spacing)
        return Profile(
            name="h6",
            symbols=symbols,
            geometry=geometry,
            basis="sto-3g",
            active_electrons=6,
            active_orbitals=6,
            spacing_angstrom=float(spacing),
        )
    raise ValueError("profile must be one of {'small', 'large', 'h6'}")


def _device_for_backend(backend: str, cpu_device: str, gpu_device: str) -> str:
    if backend == GPU_BACKEND:
        return gpu_device
    return cpu_device


def _array_backend(backend: str) -> str:
    return "cupy" if backend == GPU_BACKEND else "numpy"


def _base_kwargs(profile: Profile) -> dict[str, object]:
    return {
        "symbols": profile.symbols,
        "geometry": profile.geometry,
        "basis": profile.basis,
        "charge": 0,
        "active_electrons": profile.active_electrons,
        "active_orbitals": profile.active_orbitals,
    }


def _base_kwargs_with_spin(profile: Profile) -> dict[str, object]:
    kwargs = _base_kwargs(profile)
    kwargs["spin"] = 0
    return kwargs


def _run_adapt_vqe(backend: str, profile: Profile, cpu_device: str, gpu_device: str) -> None:
    QCANT.adapt_vqe(
        **_base_kwargs_with_spin(profile),
        adapt_it=1,
        device_name=_device_for_backend(backend, cpu_device, gpu_device),
        optimizer_method="BFGS",
        optimizer_maxiter=3,
        pool_sample_size=6,
        pool_seed=17,
        parallel_gradients=False,
    )


def _run_adapt_vqe_qulacs(backend: str, profile: Profile, cpu_device: str, gpu_device: str) -> None:
    if backend == QULACS_BACKEND:
        QCANT.adapt_vqe_qulacs(
            **_base_kwargs_with_spin(profile),
            adapt_it=1,
            optimizer_method="BFGS",
            optimizer_maxiter=3,
            pool_sample_size=6,
            pool_seed=17,
            parallel_gradients=True,
            max_workers=DEFAULT_CPU_THREADS,
        )
        return
    _run_adapt_vqe(CPU_BACKEND, profile, cpu_device, gpu_device)


def _run_qsceom(backend: str, profile: Profile, cpu_device: str, gpu_device: str) -> None:
    QCANT.qscEOM(
        profile.symbols,
        profile.geometry,
        profile.active_electrons,
        profile.active_orbitals,
        0,
        params=np.zeros(0, dtype=float),
        ash_excitation=[],
        basis=profile.basis,
        method="pyscf",
        shots=0,
        device_name=_device_for_backend(backend, cpu_device, gpu_device),
        ansatz_type="fermionic",
        include_identity=True,
        symmetric=True,
        parallel_matrix=False,
        projector_backend="dense",
    )


def _run_gcim(backend: str, profile: Profile, cpu_device: str, gpu_device: str) -> None:
    QCANT.gcim(
        **_base_kwargs_with_spin(profile),
        adapt_it=1,
        device_name=_device_for_backend(backend, cpu_device, gpu_device),
        pool_sample_size=6,
        pool_seed=19,
        pool_type="sd",
        print_progress=False,
    )


def _run_qrte(backend: str, profile: Profile, cpu_device: str, gpu_device: str) -> None:
    QCANT.qrte(
        **_base_kwargs_with_spin(profile),
        delta_t=0.05,
        n_steps=1,
        device_name=_device_for_backend(backend, cpu_device, gpu_device),
        trotter_steps=1,
        use_sparse=False,
    )


def _run_qrte_qulacs(backend: str, profile: Profile, cpu_device: str, gpu_device: str) -> None:
    if backend == QULACS_BACKEND:
        QCANT.qrte_qulacs(
            **_base_kwargs_with_spin(profile),
            delta_t=0.05,
            n_steps=1,
            evolution_mode="trotter",
            trotter_steps=1,
            max_workers=DEFAULT_CPU_THREADS,
        )
        return
    _run_qrte(CPU_BACKEND, profile, cpu_device, gpu_device)


def _run_qrte_pmte(backend: str, profile: Profile, cpu_device: str, gpu_device: str) -> None:
    QCANT.qrte_pmte(
        **_base_kwargs_with_spin(profile),
        delta_t=0.05,
        n_steps=1,
        device_name=_device_for_backend(backend, cpu_device, gpu_device),
        trotter_steps=1,
        use_sparse=False,
    )


def _run_qrte_pmte_qulacs(backend: str, profile: Profile, cpu_device: str, gpu_device: str) -> None:
    if backend == QULACS_BACKEND:
        QCANT.qrte_pmte_qulacs(
            **_base_kwargs_with_spin(profile),
            delta_t=0.05,
            n_steps=1,
            evolution_mode="trotter",
            trotter_steps=1,
            max_workers=DEFAULT_CPU_THREADS,
        )
        return
    _run_qrte_pmte(CPU_BACKEND, profile, cpu_device, gpu_device)


def _run_cvqe(backend: str, profile: Profile, cpu_device: str, gpu_device: str) -> None:
    QCANT.cvqe(
        **_base_kwargs_with_spin(profile),
        adapt_it=1,
        ansatz="lucj",
        shots=0,
        optimizer_method="BFGS",
        optimizer_maxiter=3,
        selection_method="probability",
        selection_topk=4,
        selection_seed=23,
        array_backend=_array_backend(backend),
        print_progress=False,
    )


def _run_cvqe_qulacs(backend: str, profile: Profile, cpu_device: str, gpu_device: str) -> None:
    if backend == QULACS_BACKEND:
        QCANT.cvqe_qulacs(
            **_base_kwargs_with_spin(profile),
            adapt_it=1,
            ansatz="lucj",
            shots=0,
            optimizer_method="BFGS",
            optimizer_maxiter=3,
            selection_method="probability",
            selection_topk=4,
            selection_seed=23,
            print_progress=False,
        )
        return
    _run_cvqe(CPU_BACKEND, profile, cpu_device, gpu_device)


def _run_tepid_adapt(backend: str, profile: Profile, cpu_device: str, gpu_device: str) -> None:
    QCANT.tepid_adapt(
        **_base_kwargs_with_spin(profile),
        adapt_it=1,
        beta=1.0,
        optimizer_method="BFGS",
        optimizer_maxiter=3,
        pool_sample_size=6,
        pool_seed=29,
        array_backend=_array_backend(backend),
        return_details=False,
    )


def _run_qkud(backend: str, profile: Profile, cpu_device: str, gpu_device: str) -> None:
    QCANT.qkud(
        **_base_kwargs_with_spin(profile),
        n_steps=1,
        epsilon=0.1,
        device_name=_device_for_backend(backend, cpu_device, gpu_device),
        array_backend=_array_backend(backend),
        use_sparse=False,
    )


def _run_qkud_qulacs(backend: str, profile: Profile, cpu_device: str, gpu_device: str) -> None:
    if backend == QULACS_BACKEND:
        QCANT.qkud_qulacs(
            **_base_kwargs_with_spin(profile),
            n_steps=1,
            epsilon=0.1,
            evolution_mode="trotter",
            trotter_steps=1,
            max_workers=DEFAULT_CPU_THREADS,
        )
        return
    _run_qkud(CPU_BACKEND, profile, cpu_device, gpu_device)


def _run_exact_krylov(backend: str, profile: Profile, cpu_device: str, gpu_device: str) -> None:
    QCANT.exact_krylov(
        **_base_kwargs_with_spin(profile),
        n_steps=1,
        device_name=_device_for_backend(backend, cpu_device, gpu_device),
        array_backend=_array_backend(backend),
        use_sparse=False,
    )


def _run_adapt_krylov(backend: str, profile: Profile, cpu_device: str, gpu_device: str) -> None:
    QCANT.adaptKrylov(
        **_base_kwargs_with_spin(profile),
        adapt_it=1,
        device_name=_device_for_backend(backend, cpu_device, gpu_device),
        optimizer_method="BFGS",
        optimizer_maxiter=3,
        pool_sample_size=6,
        pool_seed=31,
        backend="pennylane",
        parallel_gradients=False,
        parallel_postprocessing=False,
    )


def _run_adapt_krylov_qulacs(backend: str, profile: Profile, cpu_device: str, gpu_device: str) -> None:
    if backend == QULACS_BACKEND:
        QCANT.adaptKrylov(
            **_base_kwargs_with_spin(profile),
            adapt_it=1,
            optimizer_method="BFGS",
            optimizer_maxiter=3,
            pool_sample_size=6,
            pool_seed=31,
            backend="qulacs",
            parallel_gradients=True,
            max_workers=DEFAULT_CPU_THREADS,
            parallel_postprocessing=False,
        )
        return
    _run_adapt_krylov(CPU_BACKEND, profile, cpu_device, gpu_device)


def _run_standalone_tepid_adapt(backend: str, profile: Profile, cpu_device: str, gpu_device: str) -> None:
    standalone_tepid_adapt(
        **_base_kwargs(profile),
        adapt_it=1,
        beta=1.0,
        optimizer_method="BFGS",
        optimizer_maxiter=3,
        pool_sample_size=6,
        pool_seed=37,
        array_backend=_array_backend(backend),
        return_details=False,
    )


def _run_standalone_qsceom(backend: str, profile: Profile, cpu_device: str, gpu_device: str) -> None:
    standalone_qsceom(
        profile.symbols,
        profile.geometry,
        profile.active_electrons,
        profile.active_orbitals,
        0,
        params=np.zeros(0, dtype=float),
        ash_excitation=[],
        ansatz_type="fermionic",
        basis=profile.basis,
        method="pyscf",
        include_identity=True,
        array_backend=_array_backend(backend),
    )


def _run_standalone_tepid_qsceom(backend: str, profile: Profile, cpu_device: str, gpu_device: str) -> None:
    standalone_tepid_qsceom(
        **_base_kwargs(profile),
        adapt_it=1,
        beta=1.0,
        optimizer_method="BFGS",
        optimizer_maxiter=3,
        pool_type="fermionic_sd",
        include_identity=True,
        qsceom_include_identity=True,
        qsceom_each_iteration=False,
        array_backend=_array_backend(backend),
    )


ALGORITHM_SPECS: dict[str, AlgorithmSpec] = {
    "adapt_vqe": AlgorithmSpec("adapt_vqe", (CPU_BACKEND, GPU_BACKEND), _run_adapt_vqe),
    "qscEOM": AlgorithmSpec("qscEOM", (CPU_BACKEND, GPU_BACKEND), _run_qsceom),
    "gcim": AlgorithmSpec("gcim", (CPU_BACKEND, GPU_BACKEND), _run_gcim),
    "qrte": AlgorithmSpec("qrte", (CPU_BACKEND, GPU_BACKEND), _run_qrte),
    "qrte_pmte": AlgorithmSpec("qrte_pmte", (CPU_BACKEND, GPU_BACKEND), _run_qrte_pmte),
    "cvqe": AlgorithmSpec("cvqe", (CPU_BACKEND, GPU_BACKEND), _run_cvqe),
    "tepid_adapt": AlgorithmSpec("tepid_adapt", (CPU_BACKEND, GPU_BACKEND), _run_tepid_adapt),
    "qkud": AlgorithmSpec("qkud", (CPU_BACKEND, GPU_BACKEND), _run_qkud),
    "exact_krylov": AlgorithmSpec("exact_krylov", (CPU_BACKEND, GPU_BACKEND), _run_exact_krylov),
    "adaptKrylov": AlgorithmSpec("adaptKrylov", (CPU_BACKEND, GPU_BACKEND), _run_adapt_krylov),
    "standalone_tepid_adapt": AlgorithmSpec(
        "standalone_tepid_adapt",
        (CPU_BACKEND, GPU_BACKEND),
        _run_standalone_tepid_adapt,
    ),
    "standalone_qsceom": AlgorithmSpec(
        "standalone_qsceom",
        (CPU_BACKEND, GPU_BACKEND),
        _run_standalone_qsceom,
    ),
    "standalone_tepid_qsceom": AlgorithmSpec(
        "standalone_tepid_qsceom",
        (CPU_BACKEND, GPU_BACKEND),
        _run_standalone_tepid_qsceom,
    ),
    "adapt_vqe_qulacs": AlgorithmSpec(
        "adapt_vqe_qulacs",
        (CPU_BACKEND, QULACS_BACKEND),
        _run_adapt_vqe_qulacs,
        note="Qulacs is a CPU accelerator, not a GPU backend.",
    ),
    "cvqe_qulacs": AlgorithmSpec(
        "cvqe_qulacs",
        (CPU_BACKEND, QULACS_BACKEND),
        _run_cvqe_qulacs,
        note="Qulacs is a CPU accelerator, not a GPU backend.",
    ),
    "qkud_qulacs": AlgorithmSpec(
        "qkud_qulacs",
        (CPU_BACKEND, QULACS_BACKEND),
        _run_qkud_qulacs,
        note="Qulacs is a CPU accelerator, not a GPU backend.",
    ),
    "qrte_qulacs": AlgorithmSpec(
        "qrte_qulacs",
        (CPU_BACKEND, QULACS_BACKEND),
        _run_qrte_qulacs,
        note="Qulacs is a CPU accelerator, not a GPU backend.",
    ),
    "qrte_pmte_qulacs": AlgorithmSpec(
        "qrte_pmte_qulacs",
        (CPU_BACKEND, QULACS_BACKEND),
        _run_qrte_pmte_qulacs,
        note="Qulacs is a CPU accelerator, not a GPU backend.",
    ),
    "adaptKrylov_qulacs": AlgorithmSpec(
        "adaptKrylov_qulacs",
        (CPU_BACKEND, QULACS_BACKEND),
        _run_adapt_krylov_qulacs,
        note="Qulacs is a CPU accelerator, not a GPU backend.",
    ),
}
ALGORITHMS = tuple(ALGORITHM_SPECS)


def _accelerator_kind(backend: str) -> str:
    if backend == GPU_BACKEND:
        return "gpu"
    if backend == QULACS_BACKEND:
        return "cpu_accelerator"
    return "cpu"


def _sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _failure_status_and_note(backend: str, text: str) -> tuple[str, str]:
    lowered = text.lower()
    snippet = text.strip()[-500:]
    if backend == GPU_BACKEND and any(
        token in lowered
        for token in (
            "cupy",
            "lightning.gpu",
            "cuda",
            "gpu device",
            "no module named",
            "not installed",
        )
    ):
        return "skipped", f"GPU backend unavailable or missing optional dependencies. Trace: {snippet}"
    if backend == QULACS_BACKEND and any(
        token in lowered for token in ("qulacs", "no module named", "not installed")
    ):
        return "skipped", f"Qulacs backend unavailable or missing optional dependencies. Trace: {snippet}"
    return "error", snippet


def _run_algorithm_once(algorithm: str, backend: str, profile: Profile, cpu_device: str, gpu_device: str) -> None:
    spec = ALGORITHM_SPECS[algorithm]
    if backend not in spec.backends:
        raise ValueError(f"{algorithm} does not support backend {backend!r}")
    spec.runner(backend, profile, cpu_device, gpu_device)


def _run_isolated_single(
    algorithm: str,
    backend: str,
    profile_name: str,
    spacing: float,
    warmup: int,
    cpu_device: str,
    gpu_device: str,
    timeout: int,
    cpu_threads: int,
) -> tuple[str, float | None, str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--single-run",
        algorithm,
        "--single-backend",
        backend,
        "--profile",
        profile_name,
        "--spacing",
        str(spacing),
        "--warmup",
        str(warmup),
        "--cpu-device",
        cpu_device,
        "--gpu-device",
        gpu_device,
        "--cpu-threads",
        str(cpu_threads),
    ]
    _set_thread_limits(cpu_threads)
    env = os.environ.copy()
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            env=env,
            timeout=int(timeout),
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        note = f"Timed out after {timeout} seconds. {(stdout + stderr).strip()[-300:]}"
        return "timeout", None, note.strip()
    except subprocess.CalledProcessError as exc:
        combined = f"{exc.stdout}\n{exc.stderr}"
        status, note = _failure_status_and_note(backend, combined)
        return status, None, note
    for line in result.stdout.splitlines():
        if line.startswith("ELAPSED_SECONDS="):
            return "ok", float(line.split("=", 1)[1].strip()), ""
    return "error", None, f"Failed to parse elapsed time. stdout={result.stdout[-300:]}"


def _benchmark_backend(
    algorithm: str,
    backend: str,
    profile: Profile,
    repeats: int,
    warmup: int,
    cpu_device: str,
    gpu_device: str,
    timeout: int,
    cpu_threads: int,
) -> tuple[str, float | None, float | None, str]:
    samples: list[float] = []
    notes: list[str] = []
    terminal_status = "ok"
    for _ in range(int(repeats)):
        status, elapsed, note = _run_isolated_single(
            algorithm,
            backend,
            profile.name,
            profile.spacing_angstrom if profile.spacing_angstrom is not None else 0.9,
            warmup,
            cpu_device,
            gpu_device,
            timeout,
            cpu_threads,
        )
        if status != "ok":
            terminal_status = status
            if note:
                notes.append(note)
            break
        if elapsed is not None:
            samples.append(float(elapsed))
    if not samples:
        return terminal_status, None, None, " ".join(notes).strip()
    median, stddev = _summarize(samples)
    note_text = " ".join(notes).strip()
    return terminal_status, median, stddev, note_text


def _result_row(
    algorithm: str,
    backend: str,
    profile: Profile,
    repeats: int,
    cpu_threads: int,
    status: str,
    median: float | None,
    stddev: float | None,
    speedup: float | None,
    note: str,
) -> BenchmarkResult:
    return BenchmarkResult(
        algorithm=algorithm,
        backend=backend,
        accelerator_kind=_accelerator_kind(backend),
        profile=profile,
        median_runtime_s=median,
        stddev_runtime_s=stddev,
        speedup_vs_cpu=speedup,
        repeats=repeats,
        cpu_threads=cpu_threads,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        status=status,
        note=note,
    )


def _as_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def write_csv(rows: list[dict[str, object] | BenchmarkResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_row() if isinstance(row, BenchmarkResult) else row)


def _row_algorithm(row: dict[str, object] | BenchmarkResult) -> str:
    return row.algorithm if isinstance(row, BenchmarkResult) else str(row["algorithm"])


def _row_backend(row: dict[str, object] | BenchmarkResult) -> str:
    return row.backend if isinstance(row, BenchmarkResult) else str(row["backend"])


def _row_status(row: dict[str, object] | BenchmarkResult) -> str:
    return row.status if isinstance(row, BenchmarkResult) else str(row.get("status", "ok"))


def _row_speedup(row: dict[str, object] | BenchmarkResult) -> float | None:
    if isinstance(row, BenchmarkResult):
        return row.speedup_vs_cpu
    return _as_float(row.get("speedup_vs_cpu"))


def _row_runtime(row: dict[str, object] | BenchmarkResult) -> float | None:
    if isinstance(row, BenchmarkResult):
        return row.median_runtime_s
    return _as_float(row.get("median_runtime_s", row.get("median_seconds")))


def write_speedup_plot(rows: list[dict[str, object] | BenchmarkResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    accelerator_rows = [
        row
        for row in rows
        if _row_backend(row) != CPU_BACKEND and _row_status(row) == "ok" and _row_speedup(row) is not None
    ]

    fig, ax = plt.subplots(figsize=(12, 6))
    if accelerator_rows:
        labels = [f"{_row_algorithm(row)}\n{_row_backend(row)}" for row in accelerator_rows]
        speedups = [float(_row_speedup(row)) for row in accelerator_rows]
        colors = ["#1f77b4" if _row_backend(row) == GPU_BACKEND else "#2ca02c" for row in accelerator_rows]
        ax.bar(labels, speedups, color=colors)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
    else:
        ax.text(0.5, 0.5, "No successful accelerator runs", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
    ax.axhline(1.0, linestyle="--", linewidth=1, color="gray")
    ax.set_ylabel("Speedup vs one-thread CPU baseline")
    ax.set_title("QCANT H-chain Accelerator Speedups")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_runtime_plot(rows: list[dict[str, object] | BenchmarkResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok_rows = [row for row in rows if _row_status(row) == "ok" and _row_runtime(row) is not None]

    fig, ax = plt.subplots(figsize=(12, 6))
    if ok_rows:
        labels = [f"{_row_algorithm(row)}\n{_row_backend(row)}" for row in ok_rows]
        runtimes = [float(_row_runtime(row)) for row in ok_rows]
        colors = [
            "#7f7f7f" if _row_backend(row) == CPU_BACKEND else "#1f77b4" if _row_backend(row) == GPU_BACKEND else "#2ca02c"
            for row in ok_rows
        ]
        ax.bar(labels, runtimes, color=colors)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
    else:
        ax.text(0.5, 0.5, "No successful timing rows", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
    ax.set_ylabel("Median runtime (s)")
    ax.set_title("QCANT H-chain Median Runtime")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_algorithm_speedup_plot(rows: list[dict[str, object] | BenchmarkResult], algorithm: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    alg_rows = [row for row in rows if _row_algorithm(row) == algorithm]
    labels = [_row_backend(row) for row in alg_rows]
    values = [
        float(_row_speedup(row)) if _row_status(row) == "ok" and _row_speedup(row) is not None else 0.0
        for row in alg_rows
    ]
    colors = [
        "#7f7f7f" if label == CPU_BACKEND else "#1f77b4" if label == GPU_BACKEND else "#2ca02c"
        for label in labels
    ]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(labels), dtype=float)
    bars = ax.bar(x, values, color=colors)
    ax.axhline(1.0, linestyle="--", linewidth=1, color="gray")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Speedup vs one-thread CPU baseline")
    ax.set_title(f"{algorithm} Speedup")
    ax.grid(True, axis="y", alpha=0.25)
    for bar, row in zip(bars, alg_rows):
        status = _row_status(row)
        if status != "ok":
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                0.02,
                status,
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=8,
            )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_per_algorithm_speedup_plots(rows: list[dict[str, object] | BenchmarkResult], outdir: Path) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for algorithm in sorted({_row_algorithm(row) for row in rows}):
        path = outdir / f"{_sanitize_filename(algorithm)}_speedup.png"
        write_algorithm_speedup_plot(rows, algorithm, path)
        paths.append(path)
    return paths


def write_benchmark_readme(
    rows: list[dict[str, object] | BenchmarkResult],
    path: Path,
    *,
    command: str,
    profile: Profile,
    cpu_device: str,
    gpu_device: str,
    cpu_threads: int,
) -> None:
    ok_accelerators = [
        row
        for row in rows
        if _row_backend(row) != CPU_BACKEND and _row_status(row) == "ok" and _row_speedup(row) is not None
    ]
    best_line = "No successful accelerator speedups were recorded."
    if ok_accelerators:
        best = max(ok_accelerators, key=lambda row: float(_row_speedup(row) or 0.0))
        best_line = (
            f"Best observed accelerator row: `{_row_algorithm(best)}` on `{_row_backend(best)}` "
            f"at {float(_row_speedup(best) or 0.0):.2f}x versus CPU."
        )

    skipped = [
        f"- `{_row_algorithm(row)}` / `{_row_backend(row)}`: {_row_status(row)}"
        for row in rows
        if _row_status(row) != "ok"
    ]
    skipped_text = "\n".join(skipped) if skipped else "- No skipped or failed rows."

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# H6 GPU And Qulacs Benchmark",
                "",
                "This folder contains practical H6/STO-3G smoke benchmark artifacts for QCANT.",
                "Speedups are normalized to the matching one-thread CPU baseline for each code path.",
                "",
                "## Problem Profile",
                "",
                f"- Profile: `{profile.name}`",
                f"- Atoms: `{''.join(profile.symbols)}`",
                f"- Basis: `{profile.basis}`",
                f"- H-chain spacing: `{profile.spacing_angstrom}` Angstrom",
                f"- Active space: `{profile.active_electrons}` electrons in `{profile.active_orbitals}` orbitals",
                f"- Qubits: `{profile.n_qubits}`",
                "",
                "## Run Metadata",
                "",
                f"- Host: `{platform.node()}`",
                f"- Command: `{command}`",
                f"- CPU baseline threads: `{cpu_threads}`",
                f"- CPU PennyLane device: `{cpu_device}`",
                f"- GPU PennyLane device: `{gpu_device}`",
                f"- CUDA_VISIBLE_DEVICES: `{os.environ.get('CUDA_VISIBLE_DEVICES', '')}`",
                "",
                "## Outputs",
                "",
                "- `h6_speedups.csv`: runtime and speedup rows for CPU, GPU, and Qulacs backends.",
                "- `h6_speedup_summary.png`: combined accelerator speedup plot.",
                "- `plots/*_speedup.png`: one speedup plot per benchmarked code path.",
                "",
                "## Interpretation",
                "",
                best_line,
                "Rows labeled `qulacs_cpu` are CPU accelerator measurements, not GPU measurements.",
                "Rows labeled `skipped`, `timeout`, or `error` are retained in the CSV so the benchmark is reproducible.",
                "",
                "## Skipped Or Failed Rows",
                "",
                skipped_text,
                "",
            ]
        ),
        encoding="utf-8",
    )


def _parse_backend_filter(values: list[str] | None) -> set[str] | None:
    if values is None:
        return None
    allowed = {CPU_BACKEND, GPU_BACKEND, QULACS_BACKEND}
    requested = {value.strip() for value in values if value.strip()}
    unknown = requested - allowed
    if unknown:
        raise ValueError(f"unknown backend filter values: {sorted(unknown)}")
    return requested


def _run_full_benchmark(args: argparse.Namespace) -> list[BenchmarkResult]:
    profile = _profile_geometry(args.profile, args.spacing)
    backend_filter = _parse_backend_filter(args.backends)
    rows: list[BenchmarkResult] = []

    for algorithm in args.algorithms:
        spec = ALGORITHM_SPECS[algorithm]
        candidate_backends = [backend for backend in spec.backends if backend_filter is None or backend in backend_filter]
        if CPU_BACKEND not in candidate_backends:
            candidate_backends.insert(0, CPU_BACKEND)

        print(
            f"[benchmark] {algorithm}: running backends {', '.join(candidate_backends)}",
            flush=True,
        )
        backend_timings: dict[str, tuple[str, float | None, float | None, str]] = {}
        for backend in candidate_backends:
            print(f"[benchmark] {algorithm}/{backend}: start", flush=True)
            status, median, stddev, note = _benchmark_backend(
                algorithm,
                backend,
                profile,
                args.repeats,
                args.warmup,
                args.cpu_device,
                args.gpu_device,
                args.timeout,
                args.cpu_threads,
            )
            if spec.note and backend == QULACS_BACKEND:
                note = f"{spec.note} {note}".strip()
            backend_timings[backend] = (status, median, stddev, note)
            timing_text = "" if median is None else f" median={median:.3f}s"
            print(f"[benchmark] {algorithm}/{backend}: {status}{timing_text}", flush=True)

        cpu_status, cpu_median, cpu_stddev, cpu_note = backend_timings[CPU_BACKEND]
        rows.append(
            _result_row(
                algorithm,
                CPU_BACKEND,
                profile,
                args.repeats,
                args.cpu_threads,
                cpu_status,
                cpu_median,
                cpu_stddev,
                1.0 if cpu_status == "ok" and cpu_median is not None else None,
                cpu_note,
            )
        )

        for backend in candidate_backends:
            if backend == CPU_BACKEND:
                continue
            status, median, stddev, note = backend_timings[backend]
            speedup = None
            if status == "ok" and median and cpu_status == "ok" and cpu_median:
                speedup = float(cpu_median / median)
            rows.append(
                _result_row(
                    algorithm,
                    backend,
                    profile,
                    args.repeats,
                    args.cpu_threads,
                    status,
                    median,
                    stddev,
                    speedup,
                    note,
                )
            )
        if args.incremental:
            _write_artifacts(rows, args, profile)
    return rows


def _write_artifacts(rows: list[BenchmarkResult], args: argparse.Namespace, profile: Profile) -> list[Path]:
    csv_path = args.outdir / "h6_speedups.csv"
    speedup_png = args.outdir / "h6_speedup_summary.png"
    runtime_png = args.outdir / "h6_runtime_summary.png"
    plots_dir = args.outdir / "plots"
    readme_path = args.outdir / "README.md"
    write_csv(rows, csv_path)
    write_speedup_plot(rows, speedup_png)
    write_runtime_plot(rows, runtime_png)
    per_algorithm_paths = write_per_algorithm_speedup_plots(rows, plots_dir)
    if not args.no_readme:
        write_benchmark_readme(
            rows,
            readme_path,
            command=" ".join(sys.argv),
            profile=profile,
            cpu_device=args.cpu_device,
            gpu_device=args.gpu_device,
            cpu_threads=args.cpu_threads,
        )
    return per_algorithm_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark QCANT CPU, GPU, and Qulacs speedups.")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--profile", choices=["small", "large", "h6"], default="h6")
    parser.add_argument("--spacing", type=float, default=0.9, help="H-chain spacing in Angstrom for --profile h6.")
    parser.add_argument("--cpu-device", default="default.qubit")
    parser.add_argument("--gpu-device", default="lightning.gpu")
    parser.add_argument("--cpu-threads", type=int, default=DEFAULT_CPU_THREADS)
    parser.add_argument("--timeout", type=int, default=900, help="Per-sample subprocess timeout in seconds.")
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=list(ALGORITHMS),
        choices=list(ALGORITHMS),
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=None,
        choices=[CPU_BACKEND, GPU_BACKEND, QULACS_BACKEND],
        help="Optional backend filter. CPU baselines are always included.",
    )
    parser.add_argument("--require-gpu", action="store_true", help="Fail if no GPU benchmark row succeeds.")
    parser.add_argument("--fail-on-error", action="store_true", help="Fail if any row has status error or timeout.")
    parser.add_argument(
        "--no-incremental",
        action="store_false",
        dest="incremental",
        help="Only write CSV/plots after all algorithms complete.",
    )
    parser.add_argument("--no-readme", action="store_true", help="Do not write the benchmark README.")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("benchmarks/gpubenchmark"),
    )
    parser.add_argument("--single-run", choices=list(ALGORITHMS), default=None)
    parser.add_argument("--single-backend", choices=[CPU_BACKEND, GPU_BACKEND, QULACS_BACKEND], default=CPU_BACKEND)
    parser.add_argument("--single-mode", choices=[CPU_BACKEND, GPU_BACKEND], default=None)
    args = parser.parse_args()

    if args.repeats <= 0:
        raise ValueError("repeats must be > 0")
    if args.warmup < 0:
        raise ValueError("warmup must be >= 0")
    if args.spacing <= 0:
        raise ValueError("spacing must be > 0")
    if args.cpu_threads <= 0:
        raise ValueError("cpu_threads must be > 0")
    if args.timeout <= 0:
        raise ValueError("timeout must be > 0")

    _set_thread_limits(args.cpu_threads)
    profile = _profile_geometry(args.profile, args.spacing)

    if args.single_mode is not None:
        args.single_backend = args.single_mode

    if args.single_run is not None:
        for _ in range(args.warmup):
            _run_algorithm_once(args.single_run, args.single_backend, profile, args.cpu_device, args.gpu_device)
        elapsed = _time_once(
            lambda: _run_algorithm_once(
                args.single_run,
                args.single_backend,
                profile,
                args.cpu_device,
                args.gpu_device,
            )
        )
        print(f"ELAPSED_SECONDS={elapsed:.12f}")
        return

    rows = _run_full_benchmark(args)

    csv_path = args.outdir / "h6_speedups.csv"
    speedup_png = args.outdir / "h6_speedup_summary.png"
    runtime_png = args.outdir / "h6_runtime_summary.png"
    plots_dir = args.outdir / "plots"
    readme_path = args.outdir / "README.md"
    per_algorithm_paths = _write_artifacts(rows, args, profile)

    if args.require_gpu:
        gpu_success = any(row.backend == GPU_BACKEND and row.status == "ok" for row in rows)
        if not gpu_success:
            raise SystemExit("No GPU benchmark row succeeded; failing because --require-gpu was set.")
    if args.fail_on_error:
        bad_rows = [row for row in rows if row.status in {"error", "timeout"}]
        if bad_rows:
            details = ", ".join(f"{row.algorithm}/{row.backend}:{row.status}" for row in bad_rows)
            raise SystemExit(f"Benchmark had failing rows: {details}")

    print(f"CSV: {csv_path}")
    print(f"Speedup plot: {speedup_png}")
    print(f"Runtime plot: {runtime_png}")
    print(f"Per-algorithm plots: {len(per_algorithm_paths)} files under {plots_dir}")
    if not args.no_readme:
        print(f"README: {readme_path}")


if __name__ == "__main__":
    main()
