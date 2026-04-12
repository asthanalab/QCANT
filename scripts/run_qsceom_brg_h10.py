"""Run exact and BRG qscEOM for linear H10 in STO-3G.

Usage
-----
python scripts/run_qsceom_brg_h10.py
"""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path

import numpy as np

import QCANT
from QCANT.qsceom.excitations import inite


N_ATOMS = 10
BOND_DISTANCE_ANGSTROM = 1.5
BRG_TOLERANCE = 1e-4
N_PLOTTED_STATES = 5
OUTDIR = Path("benchmarks/qsceom_brg_h10_1p5A_tol1e-4")


def _linear_h_chain_geometry_angstrom(n_atoms: int, spacing_angstrom: float) -> np.ndarray:
    return np.asarray([[0.0, 0.0, spacing_angstrom * idx] for idx in range(n_atoms)], dtype=float)


def _estimate_memory_gib(n_atoms: int) -> dict[str, float | int]:
    qubits = 2 * int(n_atoms)
    qsceom_states = 1 + len(inite(n_atoms, qubits))
    full_dim = 2**qubits
    sector_dim = math.comb(qubits, n_atoms)

    full_basis_bytes = full_dim * qsceom_states * np.dtype(np.complex128).itemsize
    sector_basis_bytes = sector_dim * qsceom_states * np.dtype(np.complex128).itemsize

    return {
        "n_atoms": int(n_atoms),
        "n_qubits": int(qubits),
        "qsceom_basis_states": int(qsceom_states),
        "full_hilbert_dimension": int(full_dim),
        "sector_dimension": int(sector_dim),
        "full_basis_stack_gib": float(full_basis_bytes / 2**30),
        "sector_basis_stack_gib": float(sector_basis_bytes / 2**30),
    }


def _prepare_ground_state(symbols: list[str], geometry_bohr: np.ndarray):
    adapt_runner = QCANT.adapt_vqe_qulacs if hasattr(QCANT, "adapt_vqe_qulacs") else QCANT.adapt_vqe
    start = time.time()
    kwargs = dict(
        symbols=symbols,
        geometry=geometry_bohr,
        adapt_it=1,
        basis="sto-3g",
        charge=0,
        spin=0,
        active_electrons=len(symbols),
        active_orbitals=len(symbols),
        optimizer_method="BFGS",
        optimizer_maxiter=5,
        pool_type="qe",
        pool_sample_size=8,
        pool_seed=7,
        hamiltonian_source="molecular",
    )
    if adapt_runner is QCANT.adapt_vqe:
        kwargs["device_name"] = "default.qubit"

    params, excitations, energies = adapt_runner(**kwargs)
    elapsed = time.time() - start
    method = "adapt_qe_sampled_qulacs" if adapt_runner is QCANT.adapt_vqe_qulacs else "adapt_qe_sampled"
    return (params, excitations, energies), method, float(elapsed)


def _run_qsceom(
    symbols: list[str],
    geometry_bohr: np.ndarray,
    ansatz,
    *,
    brg_tolerance: float | None,
):
    start = time.time()
    values, details = QCANT.qscEOM(
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
        parallel_matrix=False,
        brg_tolerance=brg_tolerance,
        return_details=True,
    )
    elapsed = time.time() - start
    return np.asarray(values[0], dtype=float), details, float(elapsed)


def _write_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
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


def _write_summary(path: Path, lines: list[str]) -> None:
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    symbols = ["H"] * N_ATOMS
    geometry_angstrom = _linear_h_chain_geometry_angstrom(N_ATOMS, BOND_DISTANCE_ANGSTROM)
    geometry_bohr = QCANT.geometry_to_bohr(geometry_angstrom)
    molecule = f"H{N_ATOMS}"

    memory_estimate = _estimate_memory_gib(N_ATOMS)
    print("Estimated qscEOM basis memory:", flush=True)
    for key, value in memory_estimate.items():
        print(f"  {key}: {value}", flush=True)

    print(f"Preparing {molecule} ground state...", flush=True)
    ansatz, gs_prep_method, gs_prep_time = _prepare_ground_state(symbols, geometry_bohr)
    print(f"Ground-state prep time: {gs_prep_time:.2f} s", flush=True)

    print("Running exact sparse qscEOM...", flush=True)
    exact_roots, exact_details, exact_time = _run_qsceom(
        symbols,
        geometry_bohr,
        ansatz,
        brg_tolerance=None,
    )
    print(f"Exact qscEOM time: {exact_time:.2f} s", flush=True)

    print(f"Running BRG sparse qscEOM at tolerance={BRG_TOLERANCE:.0e}...", flush=True)
    brg_roots, brg_details, brg_time = _run_qsceom(
        symbols,
        geometry_bohr,
        ansatz,
        brg_tolerance=BRG_TOLERANCE,
    )
    print(f"BRG qscEOM time: {brg_time:.2f} s", flush=True)

    rows: list[dict[str, float | int | str]] = []
    n_states = min(N_PLOTTED_STATES, exact_roots.size, brg_roots.size)
    for state_index in range(n_states):
        rows.append(
            {
                "molecule": molecule,
                "n_atoms": int(N_ATOMS),
                "n_qubits": int(2 * N_ATOMS),
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

    csv_path = OUTDIR / "qsceom_brg_h10_1p5A_tol1e-4.csv"
    summary_path = OUTDIR / "qsceom_brg_h10_1p5A_tol1e-4_summary.txt"

    _write_csv(rows, csv_path)
    _write_summary(
        summary_path,
        [
            f"molecule: {molecule}",
            f"bond_distance_angstrom: {BOND_DISTANCE_ANGSTROM}",
            f"brg_tolerance: {BRG_TOLERANCE}",
            f"gs_prep_method: {gs_prep_method}",
            f"gs_prep_time_s: {gs_prep_time:.6f}",
            f"exact_qsceom_time_s: {exact_time:.6f}",
            f"brg_qsceom_time_s: {brg_time:.6f}",
            f"projector_backend: {brg_details['projector_backend']}",
            f"sector_dimension: {brg_details['sector_dimension']}",
            f"hamiltonian_nnz: {brg_details['hamiltonian_nnz']}",
            f"brg_rank: {brg_details['brg_rank']}",
            f"brg_truncation_value: {brg_details['brg_truncation_value']}",
            f"estimated_full_basis_stack_gib: {memory_estimate['full_basis_stack_gib']}",
            f"estimated_sector_basis_stack_gib: {memory_estimate['sector_basis_stack_gib']}",
        ],
    )

    print(f"Wrote {csv_path}", flush=True)
    print(f"Wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
