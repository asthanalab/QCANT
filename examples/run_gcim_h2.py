"""Run a minimal H2 GCIM calculation."""

from __future__ import annotations

import argparse

from _common import Timer, add_common_output_arg, h2_geometry, require_modules, write_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bond-length", type=float, default=1.5, help="H-H bond length in Angstrom.")
    parser.add_argument("--pool-type", type=str, default="sd", help="GCIM pool type: sd, singlet_sd, or gsd.")
    parser.add_argument("--device-name", type=str, default="default.qubit", help="PennyLane device name.")
    add_common_output_arg(parser, "gcim_h2")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not require_modules("pennylane", "pyscf", "scipy"):
        return 0

    import QCANT

    symbols, geometry = h2_geometry(args.bond_length)
    with Timer() as timer:
        params, excitations, energies, details = QCANT.gcim(
            symbols=symbols,
            geometry=geometry,
            adapt_it=1,
            basis="sto-3g",
            charge=0,
            spin=0,
            active_electrons=2,
            active_orbitals=2,
            pool_type=args.pool_type,
            device_name=args.device_name,
            print_progress=False,
            return_details=True,
        )

    print(f"GCIM selected {len(excitations)} operator(s).")
    print(f"Final projected energy: {float(energies[-1]): .12f} Ha")
    write_summary(
        args.output_dir,
        {
            "algorithm": "gcim",
            "symbols": symbols,
            "geometry": geometry,
            "pool_type": args.pool_type,
            "device_name": args.device_name,
            "params": params,
            "excitations": excitations,
            "energies": energies,
            "basis_size": len(details.get("basis_states", [])),
            "elapsed_s": timer.elapsed_s,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
