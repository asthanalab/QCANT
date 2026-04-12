"""Benchmark default frozen-K g-uCJ-GCIM thread scaling on linear H chains.

Usage
-----
python scripts/plot_gucj_gcim_default_h4_thread_scaling.py
python scripts/plot_gucj_gcim_default_h4_thread_scaling.py --csv path/to/results.csv
python scripts/plot_gucj_gcim_default_h4_thread_scaling.py --n-atoms 6 --bond-length 1.5
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import time
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import QCANT


OUTDIR = Path("benchmarks/gucj_gcim_stretched_h2_h4")
DEFAULT_THREAD_COUNTS = tuple(range(1, int(os.cpu_count() or 1) + 1))
REPEATS = 2


def _linear_h_chain_geometry(n_atoms: int, bond_length: float) -> np.ndarray:
    return np.asarray([[0.0, 0.0, float(bond_length) * idx] for idx in range(int(n_atoms))], dtype=float)


def _run_default_h_chain(*, n_atoms: int, bond_length: float, num_threads: int) -> tuple[float, float]:
    start = time.perf_counter()
    _k_history, _labels, energies, details = QCANT.gucj_gcim(
        symbols=["H"] * int(n_atoms),
        geometry=_linear_h_chain_geometry(int(n_atoms), float(bond_length)),
        basis="sto-3g",
        charge=0,
        spin=0,
        active_electrons=int(n_atoms),
        active_orbitals=int(n_atoms),
        method_variant="frozen_K",
        subset_rank_max=2,
        kappa=math.pi / 4.0,
        num_threads=int(num_threads),
        return_details=True,
    )
    runtime = time.perf_counter() - start
    return runtime, float(details["iteration_records"][-1]["abs_error_fci"])


def _write_csv(rows: list[dict[str, float | int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "threads",
                "repeat",
                "runtime_seconds",
                "abs_error_hartree",
                "best_runtime_seconds",
                "speedup_vs_1_thread",
                "parallel_efficiency",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _aggregate_rows(rows: list[dict[str, str | float | int]]) -> list[dict[str, float]]:
    grouped: dict[int, list[dict[str, str | float | int]]] = {}
    for row in rows:
        threads = int(row["threads"])
        grouped.setdefault(threads, []).append(row)

    baseline = min(
        float(item["best_runtime_seconds"])
        for item in grouped[min(grouped)].copy()
    )
    summary = []
    for threads in sorted(grouped):
        best_runtime = min(float(item["runtime_seconds"]) for item in grouped[threads])
        abs_error = min(float(item["abs_error_hartree"]) for item in grouped[threads])
        speedup = baseline / best_runtime
        summary.append(
            {
                "threads": float(threads),
                "best_runtime_seconds": float(best_runtime),
                "abs_error_hartree": float(abs_error),
                "speedup_vs_1_thread": float(speedup),
                "parallel_efficiency": float(speedup / float(threads)),
            }
        )
    return summary


def _make_plot(summary_rows: list[dict[str, float]], png_path: Path, pdf_path: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.labelsize": 18,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
        }
    )

    threads = np.asarray([row["threads"] for row in summary_rows], dtype=float)
    speedup = np.asarray([row["speedup_vs_1_thread"] for row in summary_rows], dtype=float)
    ideal = threads / threads[0]

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot(
        threads,
        speedup,
        color="#0072B2",
        marker="o",
        linewidth=2.4,
        markersize=6.0,
        markerfacecolor="white",
        markeredgewidth=1.2,
        label="Measured",
    )
    ax.plot(
        threads,
        ideal,
        color="#7A7A7A",
        linestyle="--",
        linewidth=1.8,
        label="Ideal",
    )

    ax.set_xlabel("BLAS/LAPACK threads")
    ax.set_ylabel("Speedup vs 1 thread")
    ax.set_xticks(threads)
    ax.grid(True, which="major", alpha=0.24, linewidth=0.8)
    ax.grid(True, which="minor", alpha=0.12, linewidth=0.5)
    ax.tick_params(axis="both", which="major", direction="in", length=6, width=1.1)
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)
    ax.legend(frameon=False, loc="upper left")

    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def _benchmark(*, n_atoms: int, bond_length: float) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []

    # Warm the Python module graph before timing the thread-scaling sweep.
    print("Warm-up run...", flush=True)
    _run_default_h_chain(
        n_atoms=int(n_atoms),
        bond_length=float(bond_length),
        num_threads=max(DEFAULT_THREAD_COUNTS),
    )

    baseline_best = None
    for threads in DEFAULT_THREAD_COUNTS:
        runtimes = []
        print(f"Benchmarking {threads} thread(s)...", flush=True)
        for repeat in range(REPEATS):
            runtime, abs_error = _run_default_h_chain(
                n_atoms=int(n_atoms),
                bond_length=float(bond_length),
                num_threads=threads,
            )
            runtimes.append(runtime)
            if threads == 1:
                baseline_best = runtime if baseline_best is None else min(baseline_best, runtime)
            best_runtime = min(runtimes)
            speedup = (
                1.0
                if baseline_best is None
                else float(baseline_best / best_runtime)
            )
            rows.append(
                {
                    "threads": int(threads),
                    "repeat": int(repeat),
                    "runtime_seconds": float(runtime),
                    "abs_error_hartree": float(abs_error),
                    "best_runtime_seconds": float(best_runtime),
                    "speedup_vs_1_thread": float(speedup),
                    "parallel_efficiency": float(speedup / float(threads)),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-atoms",
        type=int,
        default=4,
        help="Number of atoms in the linear hydrogen chain. Default: 4.",
    )
    parser.add_argument(
        "--bond-length",
        type=float,
        default=1.5,
        help="Nearest-neighbor spacing in Angstrom. Default: 1.5.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Existing CSV to plot without rerunning the timing benchmark.",
    )
    args = parser.parse_args()

    stem = f"gucj_gcim_default_h{int(args.n_atoms)}_linear_{str(args.bond_length).replace('.', 'p')}A_thread_scaling"
    csv_path = OUTDIR / f"{stem}.csv"
    png_path = OUTDIR / f"{stem}.png"
    pdf_path = OUTDIR / f"{stem}.pdf"

    if args.csv is None:
        rows = _benchmark(n_atoms=int(args.n_atoms), bond_length=float(args.bond_length))
        _write_csv(rows, csv_path)
        source_rows = rows
    else:
        csv_path = args.csv
        source_rows = _read_csv(csv_path)

    summary_rows = _aggregate_rows(source_rows)
    _make_plot(summary_rows, png_path, pdf_path)

    best_row = max(summary_rows, key=lambda row: row["speedup_vs_1_thread"])
    print(
        {
            "best_threads": int(best_row["threads"]),
            "best_runtime_seconds": best_row["best_runtime_seconds"],
            "best_speedup_vs_1_thread": best_row["speedup_vs_1_thread"],
            "best_parallel_efficiency": best_row["parallel_efficiency"],
        }
    )
    print(f"Wrote CSV to {csv_path}")
    print(f"Wrote PNG to {png_path}")
    print(f"Wrote PDF to {pdf_path}")


if __name__ == "__main__":
    main()
