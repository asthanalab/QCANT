"""Benchmark BRG-qscEOM errors on linear H4 and generate publication-quality plots.

Usage
-----
python scripts/plot_qsceom_brg_h4.py
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import QCANT


TOLERANCES = [10.0 ** (-k) for k in range(2, 11)]
STATE_LABELS = {0: "GS", 1: "E1", 2: "E2", 3: "E3", 4: "E4"}
OUTDIR = Path("benchmarks/qsceom_brg_h4_1p5A")


def _h4_geometry_bohr() -> np.ndarray:
    geometry_angstrom = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.5],
            [0.0, 0.0, 3.0],
            [0.0, 0.0, 4.5],
        ],
        dtype=float,
    )
    return QCANT.geometry_to_bohr(geometry_angstrom)


def _write_csv(rows: Iterable[dict[str, float | int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "tolerance",
        "state_index",
        "exact_energy_hartree",
        "brg_energy_hartree",
        "abs_error_hartree",
        "brg_rank",
        "brg_truncation_value",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _make_plot(rows: list[dict[str, float | int]], png_path: Path, pdf_path: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.labelsize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 13,
        }
    )

    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
    markers = ["o", "s", "^", "D", "v"]

    fig, ax = plt.subplots(figsize=(8.0, 5.6))
    x_values = np.asarray(sorted(TOLERANCES), dtype=float)

    for state_index in range(5):
        state_rows = sorted(
            (row for row in rows if int(row["state_index"]) == state_index),
            key=lambda row: float(row["tolerance"]),
        )
        y_values = np.asarray([max(float(row["abs_error_hartree"]), 1e-16) for row in state_rows], dtype=float)
        ax.plot(
            x_values,
            y_values,
            color=colors[state_index],
            marker=markers[state_index],
            linewidth=2.4,
            markersize=7.5,
            markerfacecolor="white",
            markeredgewidth=1.5,
            label=STATE_LABELS[state_index],
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("BRG tolerance")
    ax.set_ylabel("Absolute energy error (Ha)")
    ax.grid(True, which="major", alpha=0.24, linewidth=0.8)
    ax.grid(True, which="minor", alpha=0.12, linewidth=0.5)
    ax.legend(frameon=False, ncol=3, loc="upper left", handlelength=2.4, columnspacing=1.2)
    ax.set_xlim(float(x_values.min()), float(x_values.max()))
    ax.tick_params(axis="both", which="major", direction="in", length=6, width=1.1)
    ax.tick_params(axis="both", which="minor", direction="in", length=3, width=0.9)
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)

    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    symbols = ["H", "H", "H", "H"]
    geometry = _h4_geometry_bohr()

    print("Running ADAPT-VQE ground-state preparation...", flush=True)
    params, ash_excitation, energies = QCANT.adapt_vqe(
        symbols=symbols,
        geometry=geometry,
        adapt_it=12,
        basis="sto-3g",
        charge=0,
        spin=0,
        active_electrons=4,
        active_orbitals=4,
        device_name="default.qubit",
        optimizer_method="BFGS",
        optimizer_maxiter=40,
        pool_type="fermionic_sd",
        hamiltonian_source="molecular",
    )

    print("Computing exact qscEOM baseline...", flush=True)
    exact_values = QCANT.qscEOM(
        symbols=symbols,
        geometry=geometry,
        active_electrons=4,
        active_orbitals=4,
        charge=0,
        ansatz=(params, ash_excitation, energies),
        basis="sto-3g",
        method="pyscf",
        shots=0,
        projector_backend="sparse_number_preserving",
        parallel_matrix=True,
        max_workers=4,
    )
    exact_roots = np.asarray(exact_values[0], dtype=float)

    rows: list[dict[str, float | int]] = []
    for tolerance in TOLERANCES:
        print(f"Computing BRG qscEOM for tolerance={tolerance:.0e}...", flush=True)
        brg_values, brg_details = QCANT.qscEOM(
            symbols=symbols,
            geometry=geometry,
            active_electrons=4,
            active_orbitals=4,
            charge=0,
            ansatz=(params, ash_excitation, energies),
            basis="sto-3g",
            method="pyscf",
            shots=0,
            brg_tolerance=tolerance,
            projector_backend="sparse_number_preserving",
            parallel_matrix=True,
            max_workers=4,
            return_details=True,
        )
        brg_roots = np.asarray(brg_values[0], dtype=float)
        for state_index in range(5):
            rows.append(
                {
                    "tolerance": float(tolerance),
                    "state_index": int(state_index),
                    "exact_energy_hartree": float(exact_roots[state_index]),
                    "brg_energy_hartree": float(brg_roots[state_index]),
                    "abs_error_hartree": float(abs(brg_roots[state_index] - exact_roots[state_index])),
                    "brg_rank": int(brg_details["brg_rank"]),
                    "brg_truncation_value": float(brg_details["brg_truncation_value"]),
                }
            )

    OUTDIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTDIR / "qsceom_brg_h4_1p5A_errors.csv"
    png_path = OUTDIR / "qsceom_brg_h4_1p5A_errors.png"
    pdf_path = OUTDIR / "qsceom_brg_h4_1p5A_errors.pdf"

    _write_csv(rows, csv_path)
    _make_plot(rows, png_path, pdf_path)

    print(f"Wrote {csv_path}", flush=True)
    print(f"Wrote {png_path}", flush=True)
    print(f"Wrote {pdf_path}", flush=True)


if __name__ == "__main__":
    main()
