"""Run qscEOM for H2 from the Hartree-Fock reference."""

from __future__ import annotations

import argparse

import numpy as np

from _common import Timer, add_common_output_arg, h2_geometry, require_modules, write_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bond-length", type=float, default=1.5, help="H-H bond length in Angstrom.")
    parser.add_argument("--device-name", type=str, default="default.qubit", help="PennyLane device name.")
    parser.add_argument("--projector-backend", type=str, default="dense", help="qscEOM projector backend.")
    add_common_output_arg(parser, "qsceom_h2")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not require_modules("pennylane", "pyscf", "scipy"):
        return 0

    import QCANT

    symbols, geometry = h2_geometry(args.bond_length)
    with Timer() as timer:
        values, details = QCANT.qscEOM(
            symbols=symbols,
            geometry=geometry,
            active_electrons=2,
            active_orbitals=2,
            charge=0,
            params=np.zeros(0),
            ash_excitation=[],
            basis="sto-3g",
            shots=0,
            device_name=args.device_name,
            projector_backend=args.projector_backend,
            return_details=True,
        )

    roots = np.asarray(values[0], dtype=float)
    print("qscEOM roots:", roots)
    write_summary(
        args.output_dir,
        {
            "algorithm": "qscEOM",
            "symbols": symbols,
            "geometry": geometry,
            "device_name": args.device_name,
            "projector_backend": args.projector_backend,
            "roots": roots,
            "details": {
                "projector_backend": details.get("projector_backend"),
                "basis_size": len(details.get("basis_states", [])),
            },
            "elapsed_s": timer.elapsed_s,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
