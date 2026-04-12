"""Benchmark stretched H2/H4 g-uCJ-GRIM/GCIM convergence against exact FCI.

Usage
-----
python scripts/plot_gucj_gcim_stretched_h2_h4.py
python scripts/plot_gucj_gcim_stretched_h2_h4.py --csv path/to/results.csv

This driver currently benchmarks only algorithm 1. The frozen-K branch uses
the production defaults: generalized same-spin one-body K with the
deterministic heuristic seed set by ``kappa = pi/4``.
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
VARIANTS = [
    "frozen_K",
]
VARIANT_LABELS = {
    "frozen_K": "Frozen K",
}
MOLECULES = [
    ("H2", ["H", "H"], 1.5),
    ("H4", ["H"] * 4, 1.5),
]


def _linear_h_chain_geometry(n_atoms: int, spacing_angstrom: float) -> np.ndarray:
    return np.asarray([[0.0, 0.0, spacing_angstrom * idx] for idx in range(n_atoms)], dtype=float)


def _write_csv(rows: Iterable[dict[str, float | int | str]], path: Path) -> None:
    fieldnames = [
        "molecule",
        "n_atoms",
        "geometry_parameter_angstrom",
        "method_variant",
        "iteration",
        "basis_dimension_before_screening",
        "basis_dimension_after_zero_screening",
        "basis_dimension_after_screening",
        "kappa",
        "k_param_norm",
        "k_param_shape",
        "k_params",
        "ritz_energy_hartree",
        "fci_energy_hartree",
        "abs_error_hartree",
        "overlap_lambda_min_raw",
        "overlap_lambda_min_retained",
        "overlap_lambda_max_retained",
        "overlap_condition_number",
        "selection_mode",
        "selected_projector_index",
        "selected_projector",
        "winning_selection_score",
        "added_subset_label",
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

    for molecule, symbols, bond_distance in MOLECULES:
        geometry = _linear_h_chain_geometry(len(symbols), bond_distance)
        for variant in VARIANTS:
            print(f"Running {variant} for {molecule}...", flush=True)
            _k_history, _labels, _energies, details = QCANT.gucj_gcim(
                symbols=symbols,
                geometry=geometry,
                basis="sto-3g",
                charge=0,
                spin=0,
                active_electrons=len(symbols),
                active_orbitals=len(symbols),
                method_variant=variant,
                subset_rank_max=2,
                kappa=math.pi / 4.0,
                selection_mode="adaptive_commutator",
                include_iteration_matrices=False,
                return_details=True,
            )
            for record in details["iteration_records"]:
                rows.append(
                    {
                        "molecule": molecule,
                        "n_atoms": int(len(symbols)),
                        "geometry_parameter_angstrom": float(bond_distance),
                        "method_variant": variant,
                        "iteration": int(record["iteration"]),
                        "basis_dimension_before_screening": int(record["basis_dimension_before_screening"]),
                        "basis_dimension_after_zero_screening": int(record["basis_dimension_after_zero_screening"]),
                        "basis_dimension_after_screening": int(record["basis_dimension_after_screening"]),
                        "kappa": float(record["kappa"]),
                        "k_param_norm": float(record["k_param_norm"]),
                        "k_param_shape": str(details["k_param_shape"]),
                        "k_params": " ".join(f"{float(value):+.16e}" for value in record["k_params"]),
                        "ritz_energy_hartree": float(record["energy"]),
                        "fci_energy_hartree": float(details["fci_energy"]),
                        "abs_error_hartree": float(record["abs_error_fci"]),
                        "overlap_lambda_min_raw": float(record["overlap_lambda_min_raw"]),
                        "overlap_lambda_min_retained": float(record["overlap_lambda_min_retained"]),
                        "overlap_lambda_max_retained": float(record["overlap_lambda_max_retained"]),
                        "overlap_condition_number": float(record["overlap_condition_number"]),
                        "selection_mode": str(details["selection_mode"]),
                        "selected_projector_index": (
                            ""
                            if record.get("selected_projector_index") is None
                            else int(record["selected_projector_index"])
                        ),
                        "selected_projector": (
                            ""
                            if record.get("selected_projector") is None
                            else str(record["selected_projector"])
                        ),
                        "winning_selection_score": (
                            ""
                            if record.get("winning_selection_score") is None
                            else float(record["winning_selection_score"])
                        ),
                        "added_subset_label": str(record["added_subset_label"]),
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
            "axes.titlesize": 18,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
        }
    )

    colors = {
        "frozen_K": "#0072B2",
    }
    markers = {
        "frozen_K": "o",
    }

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), sharey=True)
    molecule_to_axis = {"H2": axes[0], "H4": axes[1]}
    molecule_to_title = {"H2": r"H$_2$", "H4": r"H$_4$"}

    for molecule, axis in molecule_to_axis.items():
        molecule_rows = [row for row in rows if str(row["molecule"]) == molecule]
        for variant in VARIANTS:
            variant_rows = sorted(
                (row for row in molecule_rows if str(row["method_variant"]) == variant),
                key=lambda row: int(row["iteration"]),
            )
            if not variant_rows:
                continue
            x_values = np.asarray([int(row["iteration"]) for row in variant_rows], dtype=float)
            y_values = np.asarray(
                [max(float(row["abs_error_hartree"]), 1e-16) for row in variant_rows],
                dtype=float,
            )
            axis.plot(
                x_values,
                y_values,
                color=colors[variant],
                marker=markers[variant],
                linewidth=2.2,
                markersize=5.6,
                markerfacecolor="white",
                markeredgewidth=1.2,
                label=VARIANT_LABELS[variant],
            )

        axis.set_title(molecule_to_title[molecule])
        axis.set_xlabel("GCIM operator-selection iteration")
        axis.set_yscale("log")
        axis.grid(True, which="major", alpha=0.24, linewidth=0.8)
        axis.grid(True, which="minor", alpha=0.12, linewidth=0.5)
        axis.tick_params(axis="both", which="major", direction="in", length=6, width=1.1)
        axis.tick_params(axis="both", which="minor", direction="in", length=3, width=0.9)
        for spine in axis.spines.values():
            spine.set_linewidth(1.1)

    axes[0].set_ylabel("Absolute energy error (Ha)")
    axes[1].legend(frameon=False, loc="upper right")

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

    csv_path = OUTDIR / "gucj_gcim_stretched_h2_h4.csv"
    png_path = OUTDIR / "gucj_gcim_stretched_h2_h4.png"
    pdf_path = OUTDIR / "gucj_gcim_stretched_h2_h4.pdf"

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
