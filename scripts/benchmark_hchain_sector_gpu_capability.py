"""Benchmark fixed-electron H-chain sparse Hamiltonian hot loops on CPU/GPU."""

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

from QCANT.qsceom.qsceom import (
    _build_exact_fermion_operator,
    _build_number_preserving_sparse_hamiltonian,
)


def _set_thread_limits() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def _geometry_h_chain(atoms: int, spacing: float) -> np.ndarray:
    return np.asarray([[0.0, 0.0, spacing * idx] for idx in range(int(atoms))], dtype=float)


def _sector_dimension(qubits: int, electrons: int) -> int:
    from math import comb

    return int(comb(int(qubits), int(electrons)))


def _summarize(samples: list[float]) -> tuple[float, float]:
    if len(samples) == 1:
        return float(samples[0]), 0.0
    return float(statistics.median(samples)), float(statistics.pstdev(samples))


def _time_once(fn) -> float:
    start = time.perf_counter()
    fn()
    return float(time.perf_counter() - start)


def _state_batch(dim: int, batch_size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    states = rng.normal(size=(int(dim), int(batch_size))) + 1j * rng.normal(size=(int(dim), int(batch_size)))
    norms = np.linalg.norm(states, axis=0)
    return np.asarray(states / norms[None, :], dtype=np.complex128)


def _sparse_gib(matrix) -> float:
    return float((matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes) / (1024**3))


def _build_sector_sparse(atoms: int, spacing: float):
    symbols = ["H"] * int(atoms)
    geometry = _geometry_h_chain(atoms, spacing)
    fermion_operator = _build_exact_fermion_operator(
        symbols=symbols,
        geometry=geometry,
        basis="sto-3g",
        charge=0,
        active_electrons=int(atoms),
        active_orbitals=int(atoms),
    )
    return _build_number_preserving_sparse_hamiltonian(
        fermion_operator=fermion_operator,
        qubits=2 * int(atoms),
        active_electrons=int(atoms),
    )


def _bench_cpu(H, states: np.ndarray, repeats: int, warmup: int) -> tuple[float, float]:
    def _work():
        out = H @ states
        return float(np.real(np.vdot(out[:, 0], out[:, 0])))

    for _ in range(int(warmup)):
        _work()
    samples = [_time_once(_work) for _ in range(int(repeats))]
    return _summarize(samples)


def _bench_gpu(H, states: np.ndarray, repeats: int, warmup: int) -> tuple[float, float]:
    try:
        import cupy as cp
        import cupyx.scipy.sparse as cpx_sparse
    except ImportError as exc:  # pragma: no cover
        raise ImportError("GPU sparse benchmark requires CuPy and cupyx.") from exc

    H_gpu = cpx_sparse.csr_matrix(H)
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
        "sector_dimension",
        "nnz",
        "sparse_matrix_gib",
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
    ax.set_title("Number-Preserving H-Chain Sparse Matmul Speedup")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark fixed-electron H-chain sparse GPU hot loops.")
    parser.add_argument("--atoms", nargs="+", type=int, default=[6, 8, 10])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8, 32, 128])
    parser.add_argument("--spacing", type=float, default=0.9)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-sector-dim", type=int, default=50000)
    parser.add_argument("--outdir", type=Path, default=Path("benchmarks/hchain_sector_gpu_capability"))
    args = parser.parse_args()

    if args.repeats <= 0:
        raise ValueError("repeats must be > 0")
    if args.warmup < 0:
        raise ValueError("warmup must be >= 0")

    _set_thread_limits()

    rows: list[dict[str, object]] = []
    for atoms in args.atoms:
        qubits = 2 * int(atoms)
        sector_dim = _sector_dimension(qubits, int(atoms))
        if atoms % 2:
            note = "current active-space helper uses a closed-shell RHF reference; odd H chains are skipped"
            for batch_size in args.batch_sizes:
                rows.append(
                    {
                        "atoms": atoms,
                        "qubits": qubits,
                        "sector_dimension": sector_dim,
                        "nnz": "",
                        "sparse_matrix_gib": "",
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
        if sector_dim > int(args.max_sector_dim):
            note = f"sector_dimension exceeds max_sector_dim={int(args.max_sector_dim)}"
            for batch_size in args.batch_sizes:
                rows.append(
                    {
                        "atoms": atoms,
                        "qubits": qubits,
                        "sector_dimension": sector_dim,
                        "nnz": "",
                        "sparse_matrix_gib": "",
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

        H, details = _build_sector_sparse(atoms, args.spacing)
        dim = int(H.shape[0])
        sparse_gib = _sparse_gib(H)
        for batch_size in args.batch_sizes:
            states = _state_batch(dim, int(batch_size), seed=2000 + int(atoms) * 100 + int(batch_size))
            cpu_median, cpu_std = _bench_cpu(H, states, args.repeats, args.warmup)
            gpu_median, gpu_std = _bench_gpu(H, states, args.repeats, args.warmup)
            rows.append(
                {
                    "atoms": atoms,
                    "qubits": qubits,
                    "sector_dimension": dim,
                    "nnz": int(details["hamiltonian_nnz"]),
                    "sparse_matrix_gib": round(sparse_gib, 6),
                    "batch_size": int(batch_size),
                    "cpu_median_seconds": round(cpu_median, 6),
                    "cpu_std_seconds": round(cpu_std, 6),
                    "gpu_median_seconds": round(gpu_median, 6),
                    "gpu_std_seconds": round(gpu_std, 6),
                    "speedup_vs_cpu": round(float(cpu_median / gpu_median), 6),
                    "status": "timed",
                    "note": "fixed-electron sparse Hamiltonian matmul after CPU chemistry setup",
                }
            )

    csv_path = args.outdir / "hchain_sector_gpu_capability.csv"
    speedup_path = args.outdir / "hchain_sector_gpu_capability_speedup.png"
    write_csv(rows, csv_path)
    write_speedup_plot(rows, speedup_path)
    print(f"CSV: {csv_path}")
    print(f"Speedup plot: {speedup_path}")


if __name__ == "__main__":
    main()
