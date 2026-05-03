"""Run ADAPT-Krylov for H2."""

from __future__ import annotations

import argparse

from _common import Timer, add_common_output_arg, h2_geometry, require_modules, write_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bond-length", type=float, default=1.5, help="H-H bond length in Angstrom.")
    parser.add_argument("--adapt-it", type=int, default=2, help="Number of ADAPT iterations.")
    parser.add_argument("--optimizer-maxiter", type=int, default=25, help="SciPy optimizer iteration limit.")
    parser.add_argument("--backend", choices=["auto", "pennylane", "qulacs"], default="pennylane")
    add_common_output_arg(parser, "adapt_krylov_h2")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not require_modules("pennylane", "pyscf", "scipy"):
        return 0

    import QCANT

    symbols, geometry = h2_geometry(args.bond_length)
    with Timer() as timer:
        params, excitations, adapt_energies, details = QCANT.adaptKrylov(
            symbols=symbols,
            geometry=geometry,
            adapt_it=args.adapt_it,
            basis="sto-3g",
            charge=0,
            spin=0,
            active_electrons=2,
            active_orbitals=2,
            device_name=None if args.backend in {"auto", "qulacs"} else "default.qubit",
            optimizer_maxiter=args.optimizer_maxiter,
            hamiltonian_source="casci",
            backend=args.backend,
        )

    print("ADAPT energies:", adapt_energies)
    print("Krylov order-2 energies:", details["krylov_order2_energies"])
    write_summary(
        args.output_dir,
        {
            "algorithm": "adapt_krylov",
            "symbols": symbols,
            "geometry": geometry,
            "backend": details.get("backend", args.backend),
            "params": params,
            "excitations": excitations,
            "adapt_energies": adapt_energies,
            "krylov_order1_energies": details.get("krylov_order1_energies"),
            "krylov_order2_energies": details.get("krylov_order2_energies"),
            "elapsed_s": timer.elapsed_s,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
