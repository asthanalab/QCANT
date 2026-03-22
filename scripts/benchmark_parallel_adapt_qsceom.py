"""Benchmark parallel ADAPT-VQE and qscEOM execution paths.

This script measures wall-clock runtime and speedup for worker counts
1/2/4/8 and writes:

- CSV table with raw timing statistics
- PNG plot of speedup curves

Usage
-----
python scripts/benchmark_parallel_adapt_qsceom.py
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import QCANT


DEFAULT_WORKERS = (1, 2, 4, 8)


def _set_thread_limits() -> None:
    """Limit BLAS/OpenMP threads to reduce oversubscription artifacts."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def _time_once(fn) -> float:
    start = time.perf_counter()
    fn()
    return float(time.perf_counter() - start)


def _summarize(samples: list[float]) -> tuple[float, float]:
    if len(samples) == 1:
        return float(samples[0]), 0.0
    return float(statistics.median(samples)), float(statistics.pstdev(samples))


def _run_adapt_once(workers: int, profile: str) -> None:
    """Run one ADAPT-VQE benchmark workload."""
    if profile == "large":
        symbols = ["H"] * 5
        geometry = np.array([[0.0, 0.0, 0.9 * i] for i in range(5)], dtype=float)
        kwargs = dict(
            symbols=symbols,
            geometry=geometry,
            adapt_it=3,
            basis="sto-3g",
            charge=0,
            spin=1,
            active_electrons=5,
            active_orbitals=5,
            device_name="default.qubit",
            optimizer_method="BFGS",
            optimizer_maxiter=6,
            parallel_gradients=(workers > 1),
            parallel_backend="process",
            max_workers=workers,
            gradient_chunk_size=2,
        )
    else:
        symbols = ["H", "H", "H", "H"]
        geometry = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.9],
                [0.0, 0.0, 1.8],
                [0.0, 0.0, 2.7],
            ],
            dtype=float,
        )
        kwargs = dict(
            symbols=symbols,
            geometry=geometry,
            adapt_it=4,
            basis="sto-3g",
            charge=0,
            spin=0,
            active_electrons=4,
            active_orbitals=4,
            device_name="default.qubit",
            optimizer_method="BFGS",
            optimizer_maxiter=8,
            parallel_gradients=(workers > 1),
            parallel_backend="process",
            max_workers=workers,
            gradient_chunk_size=2,
        )
    QCANT.adapt_vqe(**kwargs)


def benchmark_adapt(workers: int, repeats: int, warmup: int, profile: str) -> tuple[float, float]:
    """Benchmark ADAPT-VQE with a fixed H4 workload."""
    for _ in range(warmup):
        _run_adapt_once(workers, profile)
    samples = [_time_once(lambda: _run_adapt_once(workers, profile)) for _ in range(repeats)]
    return _summarize(samples)


def _run_qsceom_once(workers: int, profile: str) -> None:
    """Run one qscEOM benchmark workload."""
    if profile == "large":
        symbols = ["H"] * 6
        geometry = np.array([[0.0, 0.0, 0.9 * i] for i in range(6)], dtype=float)
        kwargs = dict(
            symbols=symbols,
            geometry=geometry,
            active_electrons=6,
            active_orbitals=6,
            charge=0,
            params=np.array([0.0]),
            ash_excitation=[[0, 1]],
            basis="sto-3g",
            method="pyscf",
            shots=0,
            device_name="default.qubit",
            symmetric=True,
            parallel_matrix=(workers > 1),
            parallel_backend="process",
            max_workers=workers,
            matrix_chunk_size=20,
        )
    else:
        symbols = ["H", "H", "H", "H"]
        geometry = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.9],
                [0.0, 0.0, 1.8],
                [0.0, 0.0, 2.7],
            ],
            dtype=float,
        )
        kwargs = dict(
            symbols=symbols,
            geometry=geometry,
            active_electrons=4,
            active_orbitals=4,
            charge=0,
            params=np.array([0.0]),
            ash_excitation=[[0, 1]],
            basis="sto-3g",
            method="pyscf",
            shots=0,
            device_name="default.qubit",
            symmetric=True,
            parallel_matrix=(workers > 1),
            parallel_backend="process",
            max_workers=workers,
            matrix_chunk_size=200,
        )
    QCANT.qscEOM(**kwargs)


def benchmark_qsceom(workers: int, repeats: int, warmup: int, profile: str) -> tuple[float, float]:
    """Benchmark qscEOM matrix construction with a fixed H4 workload."""
    for _ in range(warmup):
        _run_qsceom_once(workers, profile)
    samples = [_time_once(lambda: _run_qsceom_once(workers, profile)) for _ in range(repeats)]
    return _summarize(samples)


def _run_isolated_single(task: str, workers: int, warmup: int, profile: str) -> float:
    """Run one timed benchmark in a fresh Python process."""
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--single-run",
        task,
        "--single-workers",
        str(workers),
        "--warmup",
        str(warmup),
        "--profile",
        profile,
    ]
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    result = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
    elapsed_line = None
    for line in result.stdout.splitlines():
        if line.startswith("ELAPSED_SECONDS="):
            elapsed_line = line
    if elapsed_line is None:
        raise RuntimeError(
            "failed to parse isolated benchmark output.\n"
            f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
        )
    return float(elapsed_line.split("=", 1)[1].strip())


def write_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "algorithm",
        "workers",
        "median_seconds",
        "std_seconds",
        "speedup_vs_1",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_plot(
    workers: list[int],
    adapt_speedups: list[float],
    qsceom_speedups: list[float],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(workers, adapt_speedups, marker="o", linewidth=2, label="ADAPT-VQE gradients")
    ax.plot(workers, qsceom_speedups, marker="s", linewidth=2, label="qscEOM matrix")
    ax.plot(workers, [1.0] * len(workers), linestyle="--", linewidth=1, color="gray", label="No speedup")
    ax.set_xlabel("Workers")
    ax.set_ylabel("Speedup vs 1 worker")
    ax.set_title("Parallel Scaling: ADAPT-VQE and qscEOM")
    ax.set_xticks(workers)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark ADAPT/qscEOM parallel speedups.")
    parser.add_argument("--repeats", type=int, default=2, help="Timed repeats per worker count.")
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Untimed warmup runs per worker count.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("benchmarks/parallel_adapt_qsceom"),
        help="Output directory for CSV and plot.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        nargs="+",
        default=list(DEFAULT_WORKERS),
        help="Worker counts to benchmark (default: 1 2 4 8).",
    )
    parser.add_argument(
        "--profile",
        choices=["small", "large"],
        default="small",
        help="Benchmark workload profile.",
    )
    parser.add_argument(
        "--single-run",
        choices=["adapt", "qsceom"],
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--single-workers",
        type=int,
        default=1,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--isolate-process",
        dest="isolate_process",
        action="store_true",
        default=True,
        help="Run each timed sample in a fresh Python process (default: enabled).",
    )
    parser.add_argument(
        "--no-isolate-process",
        dest="isolate_process",
        action="store_false",
        help="Run all samples in the current process.",
    )
    args = parser.parse_args()

    if args.repeats <= 0:
        raise ValueError("repeats must be > 0")
    if args.warmup < 0:
        raise ValueError("warmup must be >= 0")
    if len(args.workers) == 0:
        raise ValueError("workers list must be non-empty")
    if 1 not in args.workers:
        raise ValueError("workers must include 1 to compute speedup baseline")

    _set_thread_limits()

    if args.single_run is not None:
        for _ in range(args.warmup):
            if args.single_run == "adapt":
                _run_adapt_once(args.single_workers, args.profile)
            else:
                _run_qsceom_once(args.single_workers, args.profile)
        if args.single_run == "adapt":
            elapsed = _time_once(lambda: _run_adapt_once(args.single_workers, args.profile))
        else:
            elapsed = _time_once(lambda: _run_qsceom_once(args.single_workers, args.profile))
        print(f"ELAPSED_SECONDS={elapsed:.12f}")
        return

    workers = sorted(set(int(w) for w in args.workers))
    adapt_stats: dict[int, tuple[float, float]] = {}
    qsceom_stats: dict[int, tuple[float, float]] = {}

    print(f"Benchmark workers: {workers}")
    print(f"Profile: {args.profile}")
    print(f"Repeats per worker: {args.repeats}")
    print(f"Warmup runs per worker: {args.warmup}")

    for w in workers:
        print(f"[ADAPT] workers={w} ...", flush=True)
        if args.isolate_process:
            adapt_samples = [
                _run_isolated_single("adapt", w, args.warmup, args.profile) for _ in range(args.repeats)
            ]
            adapt_stats[w] = _summarize(adapt_samples)
        else:
            adapt_stats[w] = benchmark_adapt(w, args.repeats, args.warmup, args.profile)
        print(
            f"  median={adapt_stats[w][0]:.4f}s std={adapt_stats[w][1]:.4f}s",
            flush=True,
        )

    for w in workers:
        print(f"[qscEOM] workers={w} ...", flush=True)
        if args.isolate_process:
            qsceom_samples = [
                _run_isolated_single("qsceom", w, args.warmup, args.profile) for _ in range(args.repeats)
            ]
            qsceom_stats[w] = _summarize(qsceom_samples)
        else:
            qsceom_stats[w] = benchmark_qsceom(w, args.repeats, args.warmup, args.profile)
        print(
            f"  median={qsceom_stats[w][0]:.4f}s std={qsceom_stats[w][1]:.4f}s",
            flush=True,
        )

    adapt_base = adapt_stats[1][0]
    qsceom_base = qsceom_stats[1][0]

    rows: list[dict[str, float | int | str]] = []
    adapt_speedups: list[float] = []
    qsceom_speedups: list[float] = []

    for w in workers:
        adapt_speedup = float(adapt_base / adapt_stats[w][0])
        qsceom_speedup = float(qsceom_base / qsceom_stats[w][0])
        adapt_speedups.append(adapt_speedup)
        qsceom_speedups.append(qsceom_speedup)

        rows.append(
            {
                "algorithm": "adapt_vqe",
                "workers": w,
                "median_seconds": round(adapt_stats[w][0], 6),
                "std_seconds": round(adapt_stats[w][1], 6),
                "speedup_vs_1": round(adapt_speedup, 6),
            }
        )
        rows.append(
            {
                "algorithm": "qscEOM",
                "workers": w,
                "median_seconds": round(qsceom_stats[w][0], 6),
                "std_seconds": round(qsceom_stats[w][1], 6),
                "speedup_vs_1": round(qsceom_speedup, 6),
            }
        )

    csv_path = args.outdir / "benchmark_parallel_adapt_qsceom.csv"
    png_path = args.outdir / "benchmark_parallel_adapt_qsceom_speedup.png"
    write_csv(rows, csv_path)
    write_plot(workers, adapt_speedups, qsceom_speedups, png_path)

    print("\nResults")
    print("-------")
    for w, s in zip(workers, adapt_speedups):
        print(f"ADAPT-VQE speedup @ {w} workers: {s:.3f}x")
    for w, s in zip(workers, qsceom_speedups):
        print(f"qscEOM speedup @ {w} workers: {s:.3f}x")
    print(f"\nCSV:  {csv_path}")
    print(f"Plot: {png_path}")


if __name__ == "__main__":
    main()
