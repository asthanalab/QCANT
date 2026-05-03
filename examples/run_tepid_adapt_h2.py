"""Run a minimal H2 TEPID-ADAPT calculation."""

from __future__ import annotations

import argparse

from _common import Timer, add_common_output_arg, h2_geometry, require_modules, write_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bond-length", type=float, default=0.74, help="H-H bond length in Angstrom.")
    parser.add_argument("--adapt-it", type=int, default=1, help="Number of TEPID iterations.")
    parser.add_argument("--beta", type=float, default=2.0, help="Inverse temperature.")
    parser.add_argument("--optimizer-maxiter", type=int, default=25, help="SciPy optimizer iteration limit.")
    parser.add_argument("--array-backend", choices=["auto", "numpy", "cupy"], default="numpy")
    add_common_output_arg(parser, "tepid_adapt_h2")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not require_modules("pennylane", "pyscf", "scipy"):
        return 0

    import QCANT

    symbols, geometry = h2_geometry(args.bond_length)
    with Timer() as timer:
        params, excitations, free_energies, details = QCANT.tepid_adapt(
            symbols=symbols,
            geometry=geometry,
            adapt_it=args.adapt_it,
            beta=args.beta,
            basis="sto-3g",
            charge=0,
            spin=0,
            active_electrons=2,
            active_orbitals=2,
            optimizer_maxiter=args.optimizer_maxiter,
            array_backend=args.array_backend,
            return_details=True,
        )

    print(f"TEPID selected {len(excitations)} excitation(s).")
    print(f"Final free energy: {float(free_energies[-1]): .12f} Ha")
    write_summary(
        args.output_dir,
        {
            "algorithm": "tepid_adapt",
            "symbols": symbols,
            "geometry": geometry,
            "array_backend": details.get("array_backend", args.array_backend),
            "params": params,
            "excitations": excitations,
            "free_energies": free_energies,
            "thermal_weights": details.get("final_thermal_weights"),
            "basis_energies": details.get("final_basis_energies"),
            "elapsed_s": timer.elapsed_s,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
