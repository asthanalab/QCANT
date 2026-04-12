"""Run a 10-point linear H4 PES with default frozen-K g-uCJ-GCIM.

Usage
-----
python scripts/plot_gucj_gcim_h4_pes.py
python scripts/plot_gucj_gcim_h4_pes.py --csv path/to/results.csv

This script uses the default algorithm-1 configuration:
- method_variant = "frozen_K"
- default generalized same-spin fixed K
- heuristic seed with kappa = pi/4

The output plot shows the final absolute energy error relative to exact FCI
as a function of the H-H spacing on a logarithmic y-axis.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import QCANT


OUTDIR = Path("benchmarks/gucj_gcim_stretched_h2_h4")
BOND_LENGTHS = np.linspace(0.5, 3.0, 10, dtype=float)


def _linear_h4_geometry(spacing_angstrom: float) -> np.ndarray:
    return np.asarray([[0.0, 0.0, spacing_angstrom * idx] for idx in range(4)], dtype=float)


def _write_csv(rows: Iterable[dict[str, float | int | str]], path: Path) -> None:
    fieldnames = [
        "bond_length_angstrom",
        "iteration_count",
        "final_iteration",
        "final_energy_hartree",
        "fci_energy_hartree",
        "abs_error_hartree",
        "k_param_norm",
        "k_param_shape",
        "selection_mode",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _run_benchmark() -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for bond_length in BOND_LENGTHS:
        print(f"Running frozen_K for H4 at {bond_length:.6f} A...", flush=True)
        _k_history, _labels, _energies, details = QCANT.gucj_gcim(
            symbols=["H"] * 4,
            geometry=_linear_h4_geometry(float(bond_length)),
            basis="sto-3g",
            charge=0,
            spin=0,
            active_electrons=4,
            active_orbitals=4,
            method_variant="frozen_K",
            subset_rank_max=2,
            kappa=math.pi / 4.0,
            selection_mode="adaptive_commutator",
            include_iteration_matrices=False,
            return_details=True,
        )
        final_record = dict(details["iteration_records"][-1])
        rows.append(
            {
                "bond_length_angstrom": float(bond_length),
                "iteration_count": int(len(details["iteration_records"])),
                "final_iteration": int(final_record["iteration"]),
                "final_energy_hartree": float(final_record["energy"]),
                "fci_energy_hartree": float(details["fci_energy"]),
                "abs_error_hartree": float(final_record["abs_error_fci"]),
                "k_param_norm": float(final_record["k_param_norm"]),
                "k_param_shape": str(details["k_param_shape"]),
                "selection_mode": str(details["selection_mode"]),
            }
        )
    return rows


def _make_plot(rows: list[dict[str, float | int | str]], png_path: Path, pdf_path: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.labelsize": 18,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
        }
    )

    sorted_rows = sorted(rows, key=lambda row: float(row["bond_length_angstrom"]))
    x_values = np.asarray([float(row["bond_length_angstrom"]) for row in sorted_rows], dtype=float)
    y_values = np.asarray(
        [max(float(row["abs_error_hartree"]), 1e-16) for row in sorted_rows],
        dtype=float,
    )

    fig, axis = plt.subplots(1, 1, figsize=(6.4, 4.8))
    axis.plot(
        x_values,
        y_values,
        color="#0072B2",
        marker="o",
        linewidth=2.2,
        markersize=5.8,
        markerfacecolor="white",
        markeredgewidth=1.2,
    )
    axis.set_xlabel(r"H-H spacing ($\AA$)")
    axis.set_ylabel("Final absolute energy error (Ha)")
    axis.set_yscale("log")
    axis.grid(True, which="major", alpha=0.24, linewidth=0.8)
    axis.grid(True, which="minor", alpha=0.12, linewidth=0.5)
    axis.tick_params(axis="both", which="major", direction="in", length=6, width=1.1)
    axis.tick_params(axis="both", which="minor", direction="in", length=3, width=0.9)
    for spine in axis.spines.values():
        spine.set_linewidth(1.1)

    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Existing CSV to plot without rerunning the chemistry benchmark.",
    )
    args = parser.parse_args()

    csv_path = OUTDIR / "gucj_gcim_h4_pes_10pt.csv"
    png_path = OUTDIR / "gucj_gcim_h4_pes_10pt.png"
    pdf_path = OUTDIR / "gucj_gcim_h4_pes_10pt.pdf"

    if args.csv is not None:
        rows = _read_csv(Path(args.csv))
    else:
        rows = _run_benchmark()
        _write_csv(rows, csv_path)
        print(f"Wrote {csv_path}", flush=True)

    _make_plot(rows, png_path, pdf_path)
    print(f"Wrote {png_path}", flush=True)
    print(f"Wrote {pdf_path}", flush=True)


if __name__ == "__main__":
    main()
