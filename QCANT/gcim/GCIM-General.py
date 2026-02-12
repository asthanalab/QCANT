"""Legacy GCIM runner kept for backwards compatibility.

This script wraps :func:`QCANT.gcim` and exposes a simple CLI interface.
Prefer importing :func:`QCANT.gcim` directly in new code.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from QCANT.gcim import gcim


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapt_it", type=int, default=27)
    parser.add_argument("--basis", type=str, default="sto-6g")
    parser.add_argument("--bond_length", type=float, default=3.0)
    parser.add_argument("--theta", type=float, default=float(np.pi / 4.0))
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--spin", type=int, default=0)
    parser.add_argument("--active_electrons", type=int, default=4)
    parser.add_argument("--active_orbitals", type=int, default=4)
    parser.add_argument("--device_name", type=str, default="default.qubit")
    parser.add_argument("--pool_seed", type=int, default=1)
    parser.add_argument(
        "--pool_type",
        type=str,
        default="sd",
        choices=["sd", "singlet_sd", "gsd"],
    )
    parser.add_argument("--allow_repeated_operators", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the legacy GCIM entry-point."""
    args = parse_args()

    symbols = ["H", "H", "H", "H"]
    z = float(args.bond_length)
    geometry = np.array(
        [
            [0.0, 0.0, 1.0 * z],
            [0.0, 0.0, 2.0 * z],
            [0.0, 0.0, 3.0 * z],
            [0.0, 0.0, 4.0 * z],
        ],
        dtype=float,
    )

    params, excitations, energies = gcim(
        symbols=symbols,
        geometry=geometry,
        adapt_it=int(args.adapt_it),
        basis=str(args.basis),
        charge=int(args.charge),
        spin=int(args.spin),
        active_electrons=int(args.active_electrons),
        active_orbitals=int(args.active_orbitals),
        device_name=str(args.device_name),
        pool_seed=int(args.pool_seed),
        pool_type=str(args.pool_type),
        theta=float(args.theta),
        allow_repeated_operators=bool(args.allow_repeated_operators),
        print_progress=True,
    )

    print("================================== Calculation complete ===================================")
    print(f"Selected operators: {len(excitations)}")
    print(f"Final energy: {energies[-1] if energies else float('nan')}")
    print(f"Parameter vector length: {len(params)}")


if __name__ == "__main__":
    main()
