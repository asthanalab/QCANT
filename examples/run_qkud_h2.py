"""Run QKUD for H2."""

from __future__ import annotations

import argparse

from _common import Timer, add_common_output_arg, h2_geometry, require_modules, write_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bond-length", type=float, default=1.5, help="H-H bond length in Angstrom.")
    parser.add_argument("--epsilon", type=float, default=0.1, help="QKUD recurrence epsilon.")
    parser.add_argument("--n-steps", type=int, default=2, help="Number of QKUD steps.")
    parser.add_argument("--array-backend", choices=["auto", "numpy", "cupy"], default="numpy")
    add_common_output_arg(parser, "qkud_h2")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not require_modules("pennylane", "pyscf", "scipy"):
        return 0

    import QCANT

    symbols, geometry = h2_geometry(args.bond_length)
    with Timer() as timer:
        energies, basis_states, min_history = QCANT.qkud(
            symbols=symbols,
            geometry=geometry,
            n_steps=args.n_steps,
            epsilon=args.epsilon,
            active_electrons=2,
            active_orbitals=2,
            basis="sto-3g",
            charge=0,
            spin=0,
            array_backend=args.array_backend,
            return_min_energy_history=True,
        )

    print("QKUD eigenvalues:", energies)
    write_summary(
        args.output_dir,
        {
            "algorithm": "qkud",
            "symbols": symbols,
            "geometry": geometry,
            "array_backend": args.array_backend,
            "energies": energies,
            "basis_shape": basis_states.shape,
            "min_energy_history": min_history,
            "elapsed_s": timer.elapsed_s,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
