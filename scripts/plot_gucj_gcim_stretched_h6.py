"""Plot default frozen-K g-uCJ-GCIM convergence for stretched linear H6.

Usage
-----
python scripts/plot_gucj_gcim_stretched_h6.py
python scripts/plot_gucj_gcim_stretched_h6.py --bond-length 2.0
python scripts/plot_gucj_gcim_stretched_h6.py --csv path/to/results.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import QCANT


OUTDIR = Path("benchmarks/gucj_gcim_stretched_h2_h4")
MOLECULE = "H6"
N_ATOMS = 6
DEFAULT_BOND_LENGTH = 2.0


def _linear_h_chain_geometry(n_atoms: int, spacing_angstrom: float) -> np.ndarray:
    return np.asarray([[0.0, 0.0, spacing_angstrom * idx] for idx in range(n_atoms)], dtype=float)


def _write_csv(rows: Iterable[dict[str, float | int | str]], path: Path) -> None:
    fieldnames = [
        "molecule",
        "n_atoms",
        "geometry_parameter_angstrom",
        "method_variant",
        "num_threads",
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


def _run_benchmark(*, bond_length: float, num_threads: int) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    geometry = _linear_h_chain_geometry(N_ATOMS, float(bond_length))

    print(
        f"Running default frozen-K guCJ-GCIM for {MOLECULE} at {bond_length:.2f} A with {num_threads} threads...",
        flush=True,
    )
    _k_history, _labels, _energies, details = QCANT.gucj_gcim(
        symbols=["H"] * N_ATOMS,
        geometry=geometry,
        basis="sto-3g",
        charge=0,
        spin=0,
        active_electrons=N_ATOMS,
        active_orbitals=N_ATOMS,
        method_variant="frozen_K",
        subset_rank_max=2,
        kappa=math.pi / 4.0,
        selection_mode="adaptive_commutator",
        num_threads=int(num_threads),
        include_iteration_matrices=False,
        return_details=True,
    )
    fci_energy = float(details["fci_energy"])
    gucj_iteration_count = len(details["iteration_records"])

    for record in details["iteration_records"]:
        rows.append(
            {
                "molecule": MOLECULE,
                "n_atoms": int(N_ATOMS),
                "geometry_parameter_angstrom": float(bond_length),
                "method_variant": "gucj_gcim",
                "num_threads": int(num_threads),
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
                    "" if record.get("selected_projector_index") is None else int(record["selected_projector_index"])
                ),
                "selected_projector": (
                    "" if record.get("selected_projector") is None else str(record["selected_projector"])
                ),
                "winning_selection_score": (
                    "" if record.get("winning_selection_score") is None else float(record["winning_selection_score"])
                ),
                "added_subset_label": str(record["added_subset_label"]),
            }
        )

    print(
        f"Running traditional GCIM for {MOLECULE} at {bond_length:.2f} A for {gucj_iteration_count} iterations...",
        flush=True,
    )
    _params, _selected_excitations, legacy_energies = QCANT.gcim(
        symbols=["H"] * N_ATOMS,
        geometry=geometry,
        adapt_it=gucj_iteration_count,
        max_iterations=gucj_iteration_count,
        basis="sto-3g",
        charge=0,
        spin=0,
        active_electrons=N_ATOMS,
        active_orbitals=N_ATOMS,
        pool_type="sd",
        theta=math.pi / 4.0,
        print_progress=False,
        return_details=False,
    )
    for iteration, energy in enumerate(legacy_energies):
        rows.append(
            {
                "molecule": MOLECULE,
                "n_atoms": int(N_ATOMS),
                "geometry_parameter_angstrom": float(bond_length),
                "method_variant": "traditional_gcim",
                "num_threads": "",
                "iteration": int(iteration),
                "basis_dimension_before_screening": "",
                "basis_dimension_after_zero_screening": "",
                "basis_dimension_after_screening": "",
                "kappa": float(math.pi / 4.0),
                "k_param_norm": "",
                "k_param_shape": "legacy_theta",
                "k_params": "",
                "ritz_energy_hartree": float(energy),
                "fci_energy_hartree": float(fci_energy),
                "abs_error_hartree": float(abs(float(energy) - float(fci_energy))),
                "overlap_lambda_min_raw": "",
                "overlap_lambda_min_retained": "",
                "overlap_lambda_max_retained": "",
                "overlap_condition_number": "",
                "selection_mode": "legacy_gcim",
                "selected_projector_index": "",
                "selected_projector": "",
                "winning_selection_score": "",
                "added_subset_label": "",
            }
        )
    return rows


def _make_plot(rows: list[dict[str, float | int | str]], *, bond_length: float, png_path: Path, pdf_path: Path) -> None:
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

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    colors = {
        "gucj_gcim": "#0072B2",
        "traditional_gcim": "#D55E00",
    }
    markers = {
        "gucj_gcim": "o",
        "traditional_gcim": "s",
    }
    labels = {
        "gucj_gcim": rf"Default g-uCJ-GCIM, H$_6$ ({bond_length:.1f} $\AA$)",
        "traditional_gcim": "Traditional GCIM",
    }
    for method_variant in ("gucj_gcim", "traditional_gcim"):
        ordered_rows = sorted(
            (row for row in rows if str(row["method_variant"]) == method_variant),
            key=lambda row: int(row["iteration"]),
        )
        if not ordered_rows:
            continue
        x_values = np.asarray([int(row["iteration"]) for row in ordered_rows], dtype=float)
        y_values = np.asarray(
            [max(float(row["abs_error_hartree"]), 1e-16) for row in ordered_rows],
            dtype=float,
        )
        ax.plot(
            x_values,
            y_values,
            color=colors[method_variant],
            marker=markers[method_variant],
            linewidth=2.4,
            markersize=5.8,
            markerfacecolor="white",
            markeredgewidth=1.2,
            label=labels[method_variant],
        )
    ax.set_xlabel("GCIM operator-selection iteration")
    ax.set_ylabel("Absolute energy error (Ha)")
    ax.set_yscale("log")
    ax.grid(True, which="major", alpha=0.24, linewidth=0.8)
    ax.grid(True, which="minor", alpha=0.12, linewidth=0.5)
    ax.tick_params(axis="both", which="major", direction="in", length=6, width=1.1)
    ax.tick_params(axis="both", which="minor", direction="in", length=3, width=0.9)
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)
    ax.legend(frameon=False, loc="upper right")

    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bond-length",
        type=float,
        default=DEFAULT_BOND_LENGTH,
        help="Nearest-neighbor spacing for linear H6 in Angstrom. Default: 2.0.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=int(os.cpu_count() or 1),
        help="BLAS/LAPACK thread count. Default: all available cores.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Existing CSV to plot without rerunning the chemistry benchmark.",
    )
    args = parser.parse_args()

    bond_tag = str(float(args.bond_length)).replace(".", "p")
    stem = f"gucj_gcim_stretched_h6_{bond_tag}A"
    csv_path = OUTDIR / f"{stem}.csv"
    png_path = OUTDIR / f"{stem}.png"
    pdf_path = OUTDIR / f"{stem}.pdf"

    if args.csv is not None:
        rows = _read_csv(Path(args.csv))
    else:
        rows = _run_benchmark(bond_length=float(args.bond_length), num_threads=int(args.num_threads))
        _write_csv(rows, csv_path)

    _make_plot(rows, bond_length=float(args.bond_length), png_path=png_path, pdf_path=pdf_path)

    final_error = float(rows[-1]["abs_error_hartree"])
    print(
        {
            "num_threads": int(args.num_threads),
            "bond_length_angstrom": float(args.bond_length),
            "iterations": len(rows),
            "final_abs_error_hartree": final_error,
        }
    )
    print(f"Wrote CSV to {csv_path}")
    print(f"Wrote PNG to {png_path}")
    print(f"Wrote PDF to {pdf_path}")


if __name__ == "__main__":
    main()
