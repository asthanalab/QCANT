"""Plot exact-selection CVQE energies for H2 and H4.

Usage
-----
python scripts/plot_cvqe_h2_h4_exact.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import QCANT


def _run_case(name: str):
    if name == "h2":
        symbols = ["H", "H"]
        geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]], dtype=float)
        kwargs = dict(
            active_electrons=2,
            active_orbitals=2,
            adapt_it=3,
            optimizer_maxiter=15,
        )
    elif name == "h4":
        symbols = ["H", "H", "H", "H"]
        geometry = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.5],
                [0.0, 0.0, 3.0],
                [0.0, 0.0, 4.5],
            ],
            dtype=float,
        )
        kwargs = dict(
            active_electrons=4,
            active_orbitals=4,
            adapt_it=3,
            optimizer_maxiter=15,
        )
    else:  # pragma: no cover
        raise ValueError(f"Unsupported case: {name}")

    params, determinants, energies, details = QCANT.cvqe(
        symbols=symbols,
        geometry=geometry,
        basis="sto-3g",
        charge=0,
        spin=0,
        shots=0,
        optimizer_method="BFGS",
        selection_seed=0,
        print_progress=False,
        return_details=True,
        **kwargs,
    )
    return {
        "params": params,
        "determinants": determinants,
        "energies": energies,
        "details": details,
    }


def write_outputs(outdir: Path) -> None:
    """Run the exact-selection H2/H4 workloads and write CSV + PNG outputs."""
    outdir.mkdir(parents=True, exist_ok=True)

    results = {"H2": _run_case("h2"), "H4": _run_case("h4")}

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    rows = ["system,iteration,energy\n"]
    for label, payload in results.items():
        iterations = np.arange(1, len(payload["energies"]) + 1)
        ax.plot(iterations, payload["energies"], marker="o", linewidth=2, label=label)
        for iteration, energy in zip(iterations, payload["energies"]):
            rows.append(f"{label.lower()},{int(iteration)},{float(energy):.12f}\n")

    ax.set_xlabel("CVQE iteration")
    ax.set_ylabel("Energy (Ha)")
    ax.set_title("CVQE exact-selection energy by iteration")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()

    csv_path = outdir / "cvqe_h2_h4_exact.csv"
    png_path = outdir / "cvqe_h2_h4_exact.png"
    csv_path.write_text("".join(rows), encoding="utf-8")
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    print(f"Wrote {csv_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    write_outputs(Path("benchmarks/cvqe_exact_h2_h4"))
