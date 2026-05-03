"""Benchmark dense H-chain Hamiltonian hot loops on CPU and GPU.

This script measures the GPU-accelerated region that QCANT can use after the
chemistry setup and dense Hamiltonian materialization have completed. It also
records the dense-memory wall for larger H chains.
"""

from __future__ import annotations

import argparse
import csv
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


def _set_thread_limits() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


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
    try:
        import cupy as cp
    except ImportError as exc:  # pragma: no cover
        raise ImportError("GPU benchmark requires CuPy. Install the QCANT GPU extra.") from exc

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


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "atoms",
        "qubits",
        "dimension",
        "dense_matrix_gib",
        "batch_size",
        "cpu_median_seconds",
        "cpu_std_seconds",
        "gpu_median_seconds",
        "gpu_std_seconds",
        "speedup_vs_cpu",
        "status",
        "note",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_speedup_plot(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    timed_rows = [row for row in rows if row["status"] == "timed"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for atoms in sorted({int(row["atoms"]) for row in timed_rows}):
        atom_rows = [row for row in timed_rows if int(row["atoms"]) == atoms]
        atom_rows.sort(key=lambda row: int(row["batch_size"]))
        ax.plot(
            [int(row["batch_size"]) for row in atom_rows],
            [float(row["speedup_vs_cpu"]) for row in atom_rows],
            marker="o",
            label=f"H{atoms}",
        )
    ax.axhline(1.0, linestyle="--", linewidth=1, color="gray")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Statevector batch size")
    ax.set_ylabel("Speedup vs CPU")
    ax.set_title("Dense H-Chain Hamiltonian Matmul Speedup")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_memory_plot(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_atoms = {}
    for row in rows:
        by_atoms[int(row["atoms"])] = float(row["dense_matrix_gib"])
    atoms = sorted(by_atoms)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([f"H{atom}" for atom in atoms], [by_atoms[atom] for atom in atoms], color="#7f7f7f")
    ax.axhline(16.0, linestyle="--", linewidth=1, color="#d62728", label="V100 16 GB")
    ax.set_yscale("log", base=2)
    ax.set_ylabel("Dense Hamiltonian memory (GiB)")
    ax.set_title("Dense Matrix Memory Wall")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark dense H-chain GPU hot-loop capability.")
    parser.add_argument("--atoms", nargs="+", type=int, default=[4, 6, 8, 10])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8, 32, 128])
    parser.add_argument("--spacing", type=float, default=0.9)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-dense-gib", type=float, default=12.0)
    parser.add_argument("--outdir", type=Path, default=Path("benchmarks/hchain_gpu_capability"))
    args = parser.parse_args()

    if args.repeats <= 0:
        raise ValueError("repeats must be > 0")
    if args.warmup < 0:
        raise ValueError("warmup must be >= 0")

    _set_thread_limits()

    rows: list[dict[str, object]] = []
    for atoms in args.atoms:
        if atoms <= 0:
            raise ValueError("all atom counts must be > 0")
        qubits = 2 * int(atoms)
        dim = 2 ** qubits
        dense_gib = _matrix_gib(qubits)
        if atoms % 2:
            note = "current PySCF active-space helper uses a closed-shell RHF reference; odd H chains are skipped"
            for batch_size in args.batch_sizes:
                rows.append(
                    {
                        "atoms": atoms,
                        "qubits": qubits,
                        "dimension": dim,
                        "dense_matrix_gib": round(dense_gib, 6),
                        "batch_size": int(batch_size),
                        "cpu_median_seconds": "",
                        "cpu_std_seconds": "",
                        "gpu_median_seconds": "",
                        "gpu_std_seconds": "",
                        "speedup_vs_cpu": "",
                        "status": "skipped",
                        "note": note,
                    }
                )
            continue
        if dense_gib > float(args.max_dense_gib):
            note = f"dense complex128 Hamiltonian exceeds max_dense_gib={float(args.max_dense_gib):.2f}"
            for batch_size in args.batch_sizes:
                rows.append(
                    {
                        "atoms": atoms,
                        "qubits": qubits,
                        "dimension": dim,
                        "dense_matrix_gib": round(dense_gib, 6),
                        "batch_size": int(batch_size),
                        "cpu_median_seconds": "",
                        "cpu_std_seconds": "",
                        "gpu_median_seconds": "",
                        "gpu_std_seconds": "",
                        "speedup_vs_cpu": "",
                        "status": "skipped",
                        "note": note,
                    }
                )
            continue

        H, qubits = _build_hamiltonian_matrix(atoms, args.spacing)
        dim = int(H.shape[0])
        for batch_size in args.batch_sizes:
            states = _state_batch(dim, int(batch_size), seed=1000 + int(atoms) * 100 + int(batch_size))
            cpu_median, cpu_std = _bench_cpu(H, states, args.repeats, args.warmup)
            gpu_median, gpu_std = _bench_gpu(H, states, args.repeats, args.warmup)
            rows.append(
                {
                    "atoms": atoms,
                    "qubits": qubits,
                    "dimension": dim,
                    "dense_matrix_gib": round(dense_gib, 6),
                    "batch_size": int(batch_size),
                    "cpu_median_seconds": round(cpu_median, 6),
                    "cpu_std_seconds": round(cpu_std, 6),
                    "gpu_median_seconds": round(gpu_median, 6),
                    "gpu_std_seconds": round(gpu_std, 6),
                    "speedup_vs_cpu": round(float(cpu_median / gpu_median), 6),
                    "status": "timed",
                    "note": "dense Hamiltonian matmul after CPU chemistry setup",
                }
            )

    csv_path = args.outdir / "hchain_gpu_capability.csv"
    speedup_path = args.outdir / "hchain_gpu_capability_speedup.png"
    memory_path = args.outdir / "hchain_dense_memory_wall.png"
    write_csv(rows, csv_path)
    write_speedup_plot(rows, speedup_path)
    write_memory_plot(rows, memory_path)
    print(f"CSV: {csv_path}")
    print(f"Speedup plot: {speedup_path}")
    print(f"Memory plot: {memory_path}")


if __name__ == "__main__":
    main()
