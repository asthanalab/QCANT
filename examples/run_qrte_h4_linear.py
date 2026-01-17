"""Run QRTE for linear H4 and plot energies vs iteration.

This script runs QCANT.qrte on a linear H4 chain for multiple timestep values
(delta_t) and plots the lowest projected-basis energy at each iteration.

Usage
-----
From the repo root:

    python examples/run_qrte_h4_linear.py

Notes
-----
- Requires the full QCANT runtime deps (NumPy, SciPy, PennyLane, PySCF, autoray).
- The QRTE implementation currently constructs a dense Hamiltonian matrix, so this
  will only be practical for small active spaces.
"""

from __future__ import annotations

import numpy as np

import QCANT


def make_linear_h4_geometry(bond_length: float = 1.5) -> np.ndarray:
    """Return a linear H4 geometry with uniform spacing (Angstrom)."""
    z = np.array([0.0, bond_length, 2.0 * bond_length, 3.0 * bond_length], dtype=float)
    return np.stack([np.zeros_like(z), np.zeros_like(z), z], axis=1)


def run_for_delta_t(delta_t: float, n_steps: int) -> np.ndarray:
    """Run QRTE and collect the lowest energy after each iteration (1..n_steps)."""
    symbols = ["H", "H", "H", "H"]
    geometry = make_linear_h4_geometry(bond_length=1.5)

    # Choose a small active space for practicality.
    # For H4 sto-3g, 4 electrons / 4 orbitals is a common minimal choice.
    active_electrons = 4
    active_orbitals = 4

    energies_over_time = []
    for k in range(1, n_steps + 1):
        energies, _basis_states, _times = QCANT.qrte(
            symbols=symbols,
            geometry=geometry,
            delta_t=delta_t,
            n_steps=k,
            active_electrons=active_electrons,
            active_orbitals=active_orbitals,
            basis="sto-3g",
            charge=0,
            spin=0,
            device_name="default.qubit",
            trotter_steps=1,
        )
        energies_over_time.append(float(np.min(energies)))

    return np.array(energies_over_time, dtype=float)


def main() -> None:
    import matplotlib.pyplot as plt

    delta_ts = [0.1, 0.5, 1.0]
    n_steps = 50

    plt.figure(figsize=(9, 5))

    iterations = np.arange(1, n_steps + 1)
    for dt in delta_ts:
        e = run_for_delta_t(dt, n_steps=n_steps)
        plt.plot(iterations, e, label=f"Δt = {dt}")

    plt.xlabel("Iteration")
    plt.ylabel("Lowest projected-basis energy")
    plt.title("QRTE on linear H4 (sto-3g)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
