"""Run a minimal H2 CVQE calculation."""

from __future__ import annotations

import argparse

from _common import Timer, add_common_output_arg, h2_geometry, require_modules, write_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bond-length", type=float, default=3.0, help="H-H bond length in Angstrom.")
    parser.add_argument("--adapt-it", type=int, default=1, help="Number of CVQE cycles.")
    parser.add_argument("--optimizer-maxiter", type=int, default=10, help="SciPy optimizer iteration limit.")
    parser.add_argument("--ansatz", choices=["lucj", "uccsd"], default="lucj")
    parser.add_argument("--array-backend", choices=["auto", "numpy", "cupy"], default="numpy")
    add_common_output_arg(parser, "cvqe_h2")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not require_modules("pennylane", "pyscf", "scipy"):
        return 0

    import QCANT

    symbols, geometry = h2_geometry(args.bond_length)
    with Timer() as timer:
        params, determinants, energies, details = QCANT.cvqe(
            symbols=symbols,
            geometry=geometry,
            adapt_it=args.adapt_it,
            ansatz=args.ansatz,
            basis="sto-3g",
            charge=0,
            spin=0,
            active_electrons=2,
            active_orbitals=2,
            shots=0,
            optimizer_maxiter=args.optimizer_maxiter,
            selection_seed=11,
            print_progress=False,
            array_backend=args.array_backend,
            return_details=True,
        )

    print(f"CVQE selected {len(determinants)} determinant(s).")
    print(f"Final energy: {float(energies[-1]): .12f} Ha")
    write_summary(
        args.output_dir,
        {
            "algorithm": "cvqe",
            "symbols": symbols,
            "geometry": geometry,
            "ansatz": args.ansatz,
            "array_backend": details.get("array_backend", args.array_backend),
            "params": params,
            "determinants": determinants,
            "energies": energies,
            "history": details.get("history"),
            "elapsed_s": timer.elapsed_s,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
