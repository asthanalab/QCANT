"""Run a minimal H2 ADAPT-VQE calculation."""

from __future__ import annotations

import argparse

from _common import Timer, add_common_output_arg, h2_geometry, require_modules, write_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bond-length", type=float, default=1.5, help="H-H bond length in Angstrom.")
    parser.add_argument("--adapt-it", type=int, default=1, help="Number of ADAPT iterations.")
    parser.add_argument("--optimizer-maxiter", type=int, default=25, help="SciPy optimizer iteration limit.")
    parser.add_argument("--device-name", type=str, default="default.qubit", help="PennyLane device name.")
    add_common_output_arg(parser, "adapt_vqe_h2")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not require_modules("pennylane", "pyscf", "scipy"):
        return 0

    import QCANT

    symbols, geometry = h2_geometry(args.bond_length)
    with Timer() as timer:
        params, excitations, energies = QCANT.adapt_vqe(
            symbols=symbols,
            geometry=geometry,
            adapt_it=args.adapt_it,
            basis="sto-3g",
            charge=0,
            spin=0,
            active_electrons=2,
            active_orbitals=2,
            device_name=args.device_name,
            optimizer_maxiter=args.optimizer_maxiter,
        )

    print(f"ADAPT-VQE selected {len(excitations)} excitation(s).")
    print(f"Final energy: {float(energies[-1]): .12f} Ha")
    write_summary(
        args.output_dir,
        {
            "algorithm": "adapt_vqe",
            "symbols": symbols,
            "geometry": geometry,
            "device_name": args.device_name,
            "params": params,
            "excitations": excitations,
            "energies": energies,
            "elapsed_s": timer.elapsed_s,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
