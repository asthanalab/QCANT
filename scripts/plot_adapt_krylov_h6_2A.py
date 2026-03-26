"""Run ADAPT+Krylov on linear H6 at 2.0 A and plot energy error by iteration."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import QCANT


OUTDIR = Path("benchmarks/adapt_krylov_h6_2A")
CSV_FILENAME = "adapt_krylov_h6_2A_history.csv"
PNG_FILENAME = "adapt_krylov_h6_2A_energy_error.png"
PDF_FILENAME = "adapt_krylov_h6_2A_energy_error.pdf"
PLOT_WIDTH_INCHES = 6.8
PLOT_HEIGHT_INCHES = 4.6
PNG_DPI = 400


def _h6_geometry_angstrom() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 4.0],
            [0.0, 0.0, 6.0],
            [0.0, 0.0, 8.0],
            [0.0, 0.0, 10.0],
        ],
        dtype=float,
    )


def _write_csv(rows: list[dict[str, float | int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "iteration",
        "adapt_energy_hartree",
        "adapt_error_hartree",
        "adapt_krylov_order1_energy_hartree",
        "adapt_krylov_order1_error_hartree",
        "adapt_krylov_order2_energy_hartree",
        "adapt_krylov_order2_error_hartree",
        "exact_ground_energy_hartree",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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

    iterations = np.asarray([int(row["iteration"]) for row in rows], dtype=int)
    adapt_error = np.asarray([max(float(row["adapt_error_hartree"]), 1e-14) for row in rows], dtype=float)
    order1_error = np.asarray(
        [max(float(row["adapt_krylov_order1_error_hartree"]), 1e-14) for row in rows],
        dtype=float,
    )
    order2_error = np.asarray(
        [max(float(row["adapt_krylov_order2_error_hartree"]), 1e-14) for row in rows],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(PLOT_WIDTH_INCHES, PLOT_HEIGHT_INCHES))
    ax.plot(
        iterations,
        adapt_error,
        color="#0072B2",
        marker="o",
        linewidth=2.5,
        markersize=7.0,
        markerfacecolor="white",
        markeredgewidth=1.5,
        label="ADAPT energy",
    )
    ax.plot(
        iterations,
        order1_error,
        color="#D55E00",
        marker="s",
        linewidth=2.5,
        markersize=6.8,
        markerfacecolor="white",
        markeredgewidth=1.5,
        label="ADAPTKrylov order 1",
    )
    ax.plot(
        iterations,
        order2_error,
        color="#009E73",
        marker="^",
        linewidth=2.5,
        markersize=7.2,
        markerfacecolor="white",
        markeredgewidth=1.5,
        label="ADAPTKrylov order 2",
    )

    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Energy error (Ha)")
    ax.set_title("Linear H6 at 2.0 A")
    ax.grid(True, which="major", alpha=0.22, linewidth=0.85)
    ax.grid(True, which="minor", alpha=0.12, linewidth=0.55)
    ax.legend(frameon=False, loc="upper right")
    ax.set_xticks(iterations)
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
    symbols = ["H", "H", "H", "H", "H", "H"]
    geometry = _h6_geometry_angstrom()

    params, ash_excitation, adapt_energies, details = QCANT.adaptKrylov(
        symbols=symbols,
        geometry=geometry,
        adapt_it=12,
        basis="sto-3g",
        charge=0,
        spin=0,
        active_electrons=6,
        active_orbitals=6,
        optimizer_method="BFGS",
        optimizer_maxiter=40,
        pool_type="fermionic_sd",
        hamiltonian_source="casci",
        backend="qulacs",
        parallel_gradients=True,
        max_workers=4,
        parallel_postprocessing=True,
        postprocess_workers=4,
    )

    exact_ground = float(details["exact_ground_energy"])
    order1 = np.asarray(details["krylov_order1_energies"], dtype=float)
    order2 = np.asarray(details["krylov_order2_energies"], dtype=float)
    adapt = np.asarray(adapt_energies, dtype=float)
    iterations = np.arange(1, adapt.size + 1, dtype=int)

    rows: list[dict[str, float | int]] = []
    for idx, adapt_energy, order1_energy, order2_energy in zip(iterations, adapt, order1, order2):
        rows.append(
            {
                "iteration": int(idx),
                "adapt_energy_hartree": float(adapt_energy),
                "adapt_error_hartree": float(abs(adapt_energy - exact_ground)),
                "adapt_krylov_order1_energy_hartree": float(order1_energy),
                "adapt_krylov_order1_error_hartree": float(abs(order1_energy - exact_ground)),
                "adapt_krylov_order2_energy_hartree": float(order2_energy),
                "adapt_krylov_order2_error_hartree": float(abs(order2_energy - exact_ground)),
                "exact_ground_energy_hartree": float(exact_ground),
            }
        )

    csv_path = OUTDIR / CSV_FILENAME
    png_path = OUTDIR / PNG_FILENAME
    pdf_path = OUTDIR / PDF_FILENAME

    _write_csv(rows, csv_path)
    _make_plot(rows, png_path, pdf_path)

    print(f"Final ADAPT energy: {adapt[-1]:.12f}", flush=True)
    print(f"Final order-1 Krylov energy: {order1[-1]:.12f}", flush=True)
    print(f"Final order-2 Krylov energy: {order2[-1]:.12f}", flush=True)
    print(f"Exact ground energy: {exact_ground:.12f}", flush=True)
    print(f"ADAPT ansatz length: {len(params)}", flush=True)
    print(f"Excitations selected: {len(ash_excitation)}", flush=True)
    print(f"Wrote {csv_path}", flush=True)
    print(f"Wrote {png_path}", flush=True)
    print(f"Wrote {pdf_path}", flush=True)


if __name__ == "__main__":
    main()
