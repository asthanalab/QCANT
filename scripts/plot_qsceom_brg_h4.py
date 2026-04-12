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
CSV_FILENAME = "qsceom_brg_h4_1p5A_errors.csv"
PNG_FILENAME = "qsceom_brg_h4_1p5A_errors.png"
PDF_FILENAME = "qsceom_brg_h4_1p5A_errors.pdf"
PLOT_WIDTH_INCHES = 6.8
PLOT_HEIGHT_INCHES = 4.8
PNG_DPI = 600


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


def _read_csv(path: Path) -> list[dict[str, str]]:
    fieldnames = [
        "tolerance",
        "state_index",
        "exact_energy_hartree",
        "brg_energy_hartree",
        "abs_error_hartree",
        "brg_rank",
        "brg_truncation_value",
    ]
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if list(reader.fieldnames or []) != fieldnames:
            raise ValueError(f"{path} has unexpected CSV schema.")
        return [dict(row) for row in reader]


def _make_plot(rows: list[dict[str, float | int]], png_path: Path, pdf_path: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.labelsize": 13,
            "axes.titlesize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10.5,
        }
    )

    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
    markers = ["o", "s", "^", "D", "v"]

    fig, ax = plt.subplots(figsize=(PLOT_WIDTH_INCHES, PLOT_HEIGHT_INCHES))
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
            linewidth=2.6,
            markersize=7.8,
            markerfacecolor="white",
            markeredgewidth=1.6,
            label=STATE_LABELS[state_index],
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("BRG tolerance")
    ax.set_ylabel("Absolute energy error (Ha)")
    ax.grid(True, which="major", alpha=0.22, linewidth=0.85)
    ax.grid(True, which="minor", alpha=0.12, linewidth=0.55)
    ax.legend(
        frameon=False,
        ncol=5,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        handlelength=2.5,
        columnspacing=1.2,
        borderaxespad=0.2,
        handletextpad=0.55,
    )
    ax.set_xlim(float(x_values.min()), float(x_values.max()))
    ax.tick_params(axis="both", which="major", direction="in", length=6, width=1.15, pad=5)
    ax.tick_params(axis="both", which="minor", direction="in", length=3, width=0.95)
    for spine in ax.spines.values():
        spine.set_linewidth(1.15)

    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=PNG_DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTDIR / CSV_FILENAME
    png_path = OUTDIR / PNG_FILENAME
    pdf_path = OUTDIR / PDF_FILENAME

    if csv_path.exists():
        print(f"Using existing benchmark data from {csv_path}...", flush=True)
        csv_rows = _read_csv(csv_path)
        rows = [
            {
                "tolerance": float(row["tolerance"]),
                "state_index": int(row["state_index"]),
                "exact_energy_hartree": float(row["exact_energy_hartree"]),
                "brg_energy_hartree": float(row["brg_energy_hartree"]),
                "abs_error_hartree": float(row["abs_error_hartree"]),
                "brg_rank": int(row["brg_rank"]),
                "brg_truncation_value": float(row["brg_truncation_value"]),
            }
            for row in csv_rows
        ]
        _make_plot(rows, png_path, pdf_path)
        print(f"Wrote {png_path}", flush=True)
        print(f"Wrote {pdf_path}", flush=True)
        return

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

    _write_csv(rows, csv_path)
    _make_plot(rows, png_path, pdf_path)

    print(f"Wrote {csv_path}", flush=True)
    print(f"Wrote {png_path}", flush=True)
    print(f"Wrote {pdf_path}", flush=True)


if __name__ == "__main__":
    main()
