"""Benchmark fixed-tolerance BRG qscEOM errors versus H-chain size.

Usage
-----
python scripts/plot_qsceom_brg_size_scaling.py
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


BRG_TOLERANCE = 1e-4
BOND_DISTANCE_ANGSTROM = 1.5
MOLECULE_SIZES = [2, 4, 6, 8]
STATE_LABELS = {0: "GS", 1: "E1", 2: "E2", 3: "E3", 4: "E4"}
OUTDIR = Path("benchmarks/qsceom_brg_size_scaling_1p5A_tol1e-4")


def _linear_h_chain_geometry_angstrom(n_atoms: int, spacing_angstrom: float) -> np.ndarray:
    return np.asarray([[0.0, 0.0, spacing_angstrom * idx] for idx in range(n_atoms)], dtype=float)


def _write_csv(rows: Iterable[dict[str, float | int | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "molecule",
        "n_atoms",
        "n_qubits",
        "bond_distance_angstrom",
        "brg_tolerance",
        "gs_prep_method",
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


def _prepare_adapt_ansatz(symbols: list[str], geometry_bohr: np.ndarray, n_atoms: int):
    adapt_iterations = {2: 1, 4: 4}[n_atoms]
    params, excitations, energies = QCANT.adapt_vqe(
        symbols=symbols,
        geometry=geometry_bohr,
        adapt_it=adapt_iterations,
        basis="sto-3g",
        charge=0,
        spin=0,
        active_electrons=n_atoms,
        active_orbitals=n_atoms,
        device_name="default.qubit",
        optimizer_method="BFGS",
        optimizer_maxiter=25,
        pool_type="fermionic_sd",
        hamiltonian_source="molecular",
    )
    return (params, excitations, energies), "adapt_vqe"


def _prepare_sampled_qe_ansatz(symbols: list[str], geometry_bohr: np.ndarray, n_atoms: int):
    adapt_runner = QCANT.adapt_vqe_qulacs if hasattr(QCANT, "adapt_vqe_qulacs") else QCANT.adapt_vqe
    params, excitations, energies = adapt_runner(
        symbols=symbols,
        geometry=geometry_bohr,
        adapt_it=1,
        basis="sto-3g",
        charge=0,
        spin=0,
        active_electrons=n_atoms,
        active_orbitals=n_atoms,
        optimizer_method="BFGS",
        optimizer_maxiter=5,
        pool_type="qe",
        pool_sample_size=8,
        pool_seed=7,
        hamiltonian_source="molecular",
    )
    method = "adapt_qe_sampled_qulacs" if adapt_runner is QCANT.adapt_vqe_qulacs else "adapt_qe_sampled"
    return (params, excitations, energies), method


def _prepare_ground_state(symbols: list[str], geometry_bohr: np.ndarray):
    n_atoms = len(symbols)
    if n_atoms >= 6:
        return _prepare_sampled_qe_ansatz(symbols, geometry_bohr, n_atoms)
    return _prepare_adapt_ansatz(symbols, geometry_bohr, n_atoms)


def _run_qsceom(symbols: list[str], geometry_bohr: np.ndarray, ansatz, *, brg_tolerance: float | None = None):
    kwargs = dict(
        symbols=symbols,
        geometry=geometry_bohr,
        active_electrons=len(symbols),
        active_orbitals=len(symbols),
        charge=0,
        ansatz=ansatz,
        basis="sto-3g",
        method="pyscf",
        shots=0,
        projector_backend="sparse_number_preserving",
        parallel_matrix=True,
        max_workers=4,
    )
    if brg_tolerance is None:
        return QCANT.qscEOM(**kwargs), None
    return QCANT.qscEOM(**kwargs, brg_tolerance=brg_tolerance, return_details=True)


def _make_plot(rows: list[dict[str, float | int | str]], png_path: Path, pdf_path: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.labelsize": 18,
            "xtick.labelsize": 15,
            "ytick.labelsize": 14,
            "legend.fontsize": 13,
        }
    )

    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
    markers = ["o", "s", "^", "D", "v"]
    molecules = [rf"H$_{size}$" for size in MOLECULE_SIZES]
    size_to_position = {size: idx for idx, size in enumerate(MOLECULE_SIZES)}
    x_positions = np.arange(len(molecules), dtype=float)

    fig, ax = plt.subplots(figsize=(8.0, 5.6))

    for state_index in range(5):
        state_rows = sorted(
            (row for row in rows if int(row["state_index"]) == state_index),
            key=lambda row: int(row["n_atoms"]),
        )
        if not state_rows:
            continue
        x_values = np.asarray(
            [float(size_to_position[int(row["n_atoms"])]) for row in state_rows],
            dtype=float,
        )
        y_values = np.asarray(
            [max(float(row["abs_error_hartree"]), 1e-16) for row in state_rows],
            dtype=float,
        )
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

    ax.set_yscale("log")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(molecules)
    ax.set_xlabel("Molecule")
    ax.set_ylabel("Absolute energy error (Ha)")
    ax.grid(True, which="major", alpha=0.24, linewidth=0.8)
    ax.grid(True, which="minor", alpha=0.12, linewidth=0.5)
    ax.legend(frameon=False, ncol=3, loc="upper left", handlelength=2.4, columnspacing=1.2)
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
    rows: list[dict[str, float | int | str]] = []

    for n_atoms in MOLECULE_SIZES:
        symbols = ["H"] * n_atoms
        geometry_angstrom = _linear_h_chain_geometry_angstrom(n_atoms, BOND_DISTANCE_ANGSTROM)
        geometry_bohr = QCANT.geometry_to_bohr(geometry_angstrom)
        molecule = f"H{n_atoms}"

        print(f"Preparing {molecule} ground state...", flush=True)
        ansatz, gs_prep_method = _prepare_ground_state(symbols, geometry_bohr)

        print(f"Computing exact qscEOM roots for {molecule}...", flush=True)
        exact_values, _ = _run_qsceom(symbols, geometry_bohr, ansatz, brg_tolerance=None)
        exact_roots = np.asarray(exact_values[0], dtype=float)

        print(f"Computing BRG qscEOM roots for {molecule} at tolerance={BRG_TOLERANCE:.0e}...", flush=True)
        brg_values, brg_details = _run_qsceom(
            symbols,
            geometry_bohr,
            ansatz,
            brg_tolerance=BRG_TOLERANCE,
        )
        brg_roots = np.asarray(brg_values[0], dtype=float)

        n_states = min(5, exact_roots.size, brg_roots.size)
        for state_index in range(n_states):
            rows.append(
                {
                    "molecule": molecule,
                    "n_atoms": int(n_atoms),
                    "n_qubits": int(2 * n_atoms),
                    "bond_distance_angstrom": float(BOND_DISTANCE_ANGSTROM),
                    "brg_tolerance": float(BRG_TOLERANCE),
                    "gs_prep_method": gs_prep_method,
                    "state_index": int(state_index),
                    "exact_energy_hartree": float(exact_roots[state_index]),
                    "brg_energy_hartree": float(brg_roots[state_index]),
                    "abs_error_hartree": float(abs(brg_roots[state_index] - exact_roots[state_index])),
                    "brg_rank": int(brg_details["brg_rank"]),
                    "brg_truncation_value": float(brg_details["brg_truncation_value"]),
                }
            )

    OUTDIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTDIR / "qsceom_brg_size_scaling_1p5A_tol1e-4.csv"
    png_path = OUTDIR / "qsceom_brg_size_scaling_1p5A_tol1e-4.png"
    pdf_path = OUTDIR / "qsceom_brg_size_scaling_1p5A_tol1e-4.pdf"

    _write_csv(rows, csv_path)
    _make_plot(rows, png_path, pdf_path)

    print(f"Wrote {csv_path}", flush=True)
    print(f"Wrote {png_path}", flush=True)
    print(f"Wrote {pdf_path}", flush=True)


if __name__ == "__main__":
    main()
