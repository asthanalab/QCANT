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
BASE_MOLECULE_SIZES = [2, 4, 6, 8]
H10_OUTDIR = Path("benchmarks/qsceom_brg_h10_1p5A_tol1e-4")
STATE_LABELS = {0: "GS", 1: "E1", 2: "E2", 3: "E3", 4: "E4"}
OUTDIR = Path("benchmarks/qsceom_brg_size_scaling_1p5A_tol1e-4")
CSV_FILENAME = "qsceom_brg_size_scaling_1p5A_tol1e-4.csv"
PNG_FILENAME = "qsceom_brg_size_scaling_1p5A_tol1e-4.png"
PDF_FILENAME = "qsceom_brg_size_scaling_1p5A_tol1e-4.pdf"
H10_CSV_FILENAME = "qsceom_brg_h10_1p5A_tol1e-4.csv"
PLOT_WIDTH_INCHES = 6.8
PLOT_HEIGHT_INCHES = 4.8
PNG_DPI = 600


def _fieldnames() -> list[str]:
    return [
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


def _linear_h_chain_geometry_angstrom(n_atoms: int, spacing_angstrom: float) -> np.ndarray:
    return np.asarray([[0.0, 0.0, spacing_angstrom * idx] for idx in range(n_atoms)], dtype=float)


def _write_csv(rows: Iterable[dict[str, float | int | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_fieldnames())
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        expected = _fieldnames()
        if fieldnames != expected:
            raise ValueError(
                f"{path} has schema {fieldnames!r}; expected {expected!r}."
            )
        return [dict(row) for row in reader]


def _validate_rows(rows: list[dict[str, str]], *, source: Path) -> None:
    for row in rows:
        bond_distance = float(row["bond_distance_angstrom"])
        tolerance = float(row["brg_tolerance"])
        if abs(bond_distance - BOND_DISTANCE_ANGSTROM) > 1e-12:
            raise ValueError(
                f"{source} contains bond_distance_angstrom={bond_distance}; "
                f"expected {BOND_DISTANCE_ANGSTROM}."
            )
        if abs(tolerance - BRG_TOLERANCE) > 1e-16:
            raise ValueError(
                f"{source} contains brg_tolerance={tolerance}; expected {BRG_TOLERANCE}."
            )


def _merge_h10_rows(base_rows: list[dict[str, str]], h10_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    merged = [row for row in base_rows if int(row["n_atoms"]) != 10]
    merged.extend(h10_rows)
    return sorted(merged, key=lambda row: (int(row["n_atoms"]), int(row["state_index"])))


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
    sizes = sorted({int(row["n_atoms"]) for row in rows})
    molecules = [rf"H$_{{{size}}}$" for size in sizes]
    size_to_position = {size: idx for idx, size in enumerate(sizes)}
    x_positions = np.arange(len(molecules), dtype=float)

    fig, ax = plt.subplots(figsize=(PLOT_WIDTH_INCHES, PLOT_HEIGHT_INCHES))

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
            linewidth=2.6,
            markersize=7.8,
            markerfacecolor="white",
            markeredgewidth=1.6,
            label=STATE_LABELS[state_index],
        )

    ax.set_yscale("log")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(molecules)
    ax.set_xlabel("Molecule")
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
    h10_csv_path = H10_OUTDIR / H10_CSV_FILENAME

    if csv_path.exists():
        print(f"Using existing size-scaling dataset from {csv_path}...", flush=True)
        base_rows = _read_csv(csv_path)
    else:
        rows: list[dict[str, float | int | str]] = []

        for n_atoms in BASE_MOLECULE_SIZES:
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

        _write_csv(rows, csv_path)
        base_rows = _read_csv(csv_path)

    _validate_rows(base_rows, source=csv_path)

    final_rows = list(base_rows)
    if h10_csv_path.exists():
        print(f"Merging standalone H10 dataset from {h10_csv_path}...", flush=True)
        h10_rows = _read_csv(h10_csv_path)
        _validate_rows(h10_rows, source=h10_csv_path)
        if any(int(row["n_atoms"]) != 10 for row in h10_rows):
            raise ValueError(f"{h10_csv_path} contains non-H10 rows.")
        final_rows = _merge_h10_rows(base_rows, h10_rows)
    else:
        print(f"No standalone H10 dataset found at {h10_csv_path}; plotting current size range only.", flush=True)

    _write_csv(final_rows, csv_path)
    _make_plot(final_rows, png_path, pdf_path)

    print(f"Wrote {csv_path}", flush=True)
    print(f"Wrote {png_path}", flush=True)
    print(f"Wrote {pdf_path}", flush=True)


if __name__ == "__main__":
    main()
