"""Benchmark CPU+GPU hybrid dense H-chain Hamiltonian hot loops."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import statistics
import sys
import time

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
import pennylane as qml

from QCANT.qsceom.qsceom import _build_pyscf_active_space_hamiltonian


def _matrix_gib(qubits: int, *, dtype_bytes: int = 16) -> float:
    dim = 2 ** int(qubits)
    return float(dim * dim * dtype_bytes / (1024**3))


def _geometry_h_chain(atoms: int, spacing: float) -> np.ndarray:
    return np.asarray([[0.0, 0.0, spacing * idx] for idx in range(int(atoms))], dtype=float)


def _summarize(samples: list[float]) -> tuple[float, float]:
    if len(samples) == 1:
        return float(samples[0]), 0.0
    return float(statistics.median(samples)), float(statistics.pstdev(samples))


def _time_once(fn) -> float:
    start = time.perf_counter()
    fn()
    return float(time.perf_counter() - start)


def _build_hamiltonian_matrix(atoms: int, spacing: float):
    symbols = ["H"] * int(atoms)
    geometry = _geometry_h_chain(atoms, spacing)
    hamiltonian, qubits, *_rest = _build_pyscf_active_space_hamiltonian(
        symbols=symbols,
        geometry=geometry,
        basis="sto-3g",
        charge=0,
        active_electrons=int(atoms),
        active_orbitals=int(atoms),
    )
    return np.asarray(qml.matrix(hamiltonian, wire_order=range(qubits)), dtype=np.complex128), int(qubits)


def _state_batch(dim: int, batch_size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    states = rng.normal(size=(int(dim), int(batch_size))) + 1j * rng.normal(size=(int(dim), int(batch_size)))
    norms = np.linalg.norm(states, axis=0)
    return np.asarray(states / norms[None, :], dtype=np.complex128)


def _bench_cpu(H: np.ndarray, states: np.ndarray, repeats: int, warmup: int) -> tuple[float, float]:
    def _work():
        out = H @ states
        return float(np.real(np.vdot(out[:, 0], out[:, 0])))

    for _ in range(int(warmup)):
        _work()
    samples = [_time_once(_work) for _ in range(int(repeats))]
    return _summarize(samples)


def _bench_gpu(H: np.ndarray, states: np.ndarray, repeats: int, warmup: int) -> tuple[float, float]:
    import cupy as cp

    H_gpu = cp.asarray(H)
    states_gpu = cp.asarray(states)
    cp.cuda.Stream.null.synchronize()

    def _work():
        out = H_gpu @ states_gpu
        value = cp.real(cp.vdot(out[:, 0], out[:, 0]))
        cp.cuda.Stream.null.synchronize()
        return float(value.get())

    for _ in range(int(warmup)):
        _work()
    samples = [_time_once(_work) for _ in range(int(repeats))]
    return _summarize(samples)


def _bench_hybrid(
    H: np.ndarray,
    states: np.ndarray,
    *,
    gpu_fraction: float,
    repeats: int,
    warmup: int,
) -> tuple[float, float, int, int]:
    import cupy as cp

    batch_size = int(states.shape[1])
    gpu_cols = int(round(batch_size * float(gpu_fraction)))
    gpu_cols = min(max(gpu_cols, 0), batch_size)
    cpu_cols = batch_size - gpu_cols

    H_gpu = cp.asarray(H)
    cpu_states = states[:, :cpu_cols] if cpu_cols else None
    gpu_states = cp.asarray(states[:, cpu_cols:]) if gpu_cols else None
    cp.cuda.Stream.null.synchronize()

    def _cpu_work():
        if cpu_states is None:
            return 0.0
        out = H @ cpu_states
        return float(np.real(np.vdot(out[:, 0], out[:, 0])))

    def _work(executor: ThreadPoolExecutor):
        cpu_future = executor.submit(_cpu_work) if cpu_cols else None
        gpu_value = 0.0
        if gpu_states is not None:
            gpu_out = H_gpu @ gpu_states
            gpu_value = cp.real(cp.vdot(gpu_out[:, 0], gpu_out[:, 0]))
            cp.cuda.Stream.null.synchronize()
            gpu_value = float(gpu_value.get())
        cpu_value = float(cpu_future.result()) if cpu_future is not None else 0.0
        return cpu_value + gpu_value

    samples = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        for _ in range(int(warmup)):
            _work(executor)
        samples = [_time_once(lambda: _work(executor)) for _ in range(int(repeats))]
    median, std = _summarize(samples)
    return median, std, cpu_cols, gpu_cols


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "atoms",
        "qubits",
        "dimension",
        "dense_matrix_gib",
        "batch_size",
        "gpu_fraction",
        "cpu_columns",
        "gpu_columns",
        "cpu_median_seconds",
        "gpu_median_seconds",
        "hybrid_median_seconds",
        "hybrid_std_seconds",
        "gpu_speedup_vs_cpu",
        "hybrid_speedup_vs_cpu",
        "hybrid_speedup_vs_gpu",
        "status",
        "note",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_plot(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    timed_rows = [row for row in rows if row["status"] == "timed"]
    batches = sorted({int(row["batch_size"]) for row in timed_rows})
    fig, ax = plt.subplots(figsize=(9, 5))
    if timed_rows:
        gpu_by_batch = {}
        for row in timed_rows:
            gpu_by_batch.setdefault(int(row["batch_size"]), float(row["gpu_speedup_vs_cpu"]))
        ax.plot(
            batches,
            [gpu_by_batch[batch] for batch in batches],
            marker="o",
            linewidth=2,
            label="GPU only",
        )
        for fraction in sorted({float(row["gpu_fraction"]) for row in timed_rows}):
            fraction_rows = [row for row in timed_rows if float(row["gpu_fraction"]) == fraction]
            fraction_rows.sort(key=lambda row: int(row["batch_size"]))
            ax.plot(
                [int(row["batch_size"]) for row in fraction_rows],
                [float(row["hybrid_speedup_vs_cpu"]) for row in fraction_rows],
                marker="o",
                label=f"Hybrid GPU {fraction:.0%}",
            )
    ax.axhline(1.0, linestyle="--", linewidth=1, color="gray")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Statevector batch size")
    ax.set_ylabel("Speedup vs CPU-only")
    ax.set_title("H-Chain CPU+GPU Hybrid Dense Matmul")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark CPU+GPU hybrid dense H-chain matmul.")
    parser.add_argument("--atoms", type=int, default=6)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[128, 512, 1024])
    parser.add_argument("--gpu-fractions", nargs="+", type=float, default=[0.90, 0.95, 0.98])
    parser.add_argument("--spacing", type=float, default=0.9)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--max-dense-gib", type=float, default=12.0)
    parser.add_argument("--outdir", type=Path, default=Path("benchmarks/hchain_hybrid_capability"))
    args = parser.parse_args()

    qubits = 2 * int(args.atoms)
    dim = 2 ** qubits
    dense_gib = _matrix_gib(qubits)
    rows: list[dict[str, object]] = []

    if int(args.atoms) % 2:
        raise ValueError("current active-space helper uses closed-shell RHF; use an even H chain")

    if dense_gib > float(args.max_dense_gib):
        for batch_size in args.batch_sizes:
            for fraction in args.gpu_fractions:
                rows.append(
                    {
                        "atoms": int(args.atoms),
                        "qubits": qubits,
                        "dimension": dim,
                        "dense_matrix_gib": round(dense_gib, 6),
                        "batch_size": int(batch_size),
                        "gpu_fraction": float(fraction),
                        "cpu_columns": "",
                        "gpu_columns": "",
                        "cpu_median_seconds": "",
                        "gpu_median_seconds": "",
                        "hybrid_median_seconds": "",
                        "hybrid_std_seconds": "",
                        "gpu_speedup_vs_cpu": "",
                        "hybrid_speedup_vs_cpu": "",
                        "hybrid_speedup_vs_gpu": "",
                        "status": "skipped",
                        "note": f"dense complex128 Hamiltonian exceeds max_dense_gib={float(args.max_dense_gib):.2f}",
                    }
                )
    else:
        H, qubits = _build_hamiltonian_matrix(int(args.atoms), float(args.spacing))
        dim = int(H.shape[0])
        for batch_size in args.batch_sizes:
            states = _state_batch(dim, int(batch_size), seed=3000 + int(args.atoms) * 100 + int(batch_size))
            cpu_median, _cpu_std = _bench_cpu(H, states, int(args.repeats), int(args.warmup))
            gpu_median, _gpu_std = _bench_gpu(H, states, int(args.repeats), int(args.warmup))
            for fraction in args.gpu_fractions:
                hybrid_median, hybrid_std, cpu_cols, gpu_cols = _bench_hybrid(
                    H,
                    states,
                    gpu_fraction=float(fraction),
                    repeats=int(args.repeats),
                    warmup=int(args.warmup),
                )
                rows.append(
                    {
                        "atoms": int(args.atoms),
                        "qubits": qubits,
                        "dimension": dim,
                        "dense_matrix_gib": round(dense_gib, 6),
                        "batch_size": int(batch_size),
                        "gpu_fraction": round(float(fraction), 6),
                        "cpu_columns": int(cpu_cols),
                        "gpu_columns": int(gpu_cols),
                        "cpu_median_seconds": round(cpu_median, 6),
                        "gpu_median_seconds": round(gpu_median, 6),
                        "hybrid_median_seconds": round(hybrid_median, 6),
                        "hybrid_std_seconds": round(hybrid_std, 6),
                        "gpu_speedup_vs_cpu": round(float(cpu_median / gpu_median), 6),
                        "hybrid_speedup_vs_cpu": round(float(cpu_median / hybrid_median), 6),
                        "hybrid_speedup_vs_gpu": round(float(gpu_median / hybrid_median), 6),
                        "status": "timed",
                        "note": "CPU and GPU process disjoint state columns concurrently",
                    }
                )

    csv_path = args.outdir / "hchain_hybrid_capability.csv"
    plot_path = args.outdir / "hchain_hybrid_capability_speedup.png"
    write_csv(rows, csv_path)
    write_plot(rows, plot_path)
    print(f"CSV: {csv_path}")
    print(f"Speedup plot: {plot_path}")


if __name__ == "__main__":
    main()
