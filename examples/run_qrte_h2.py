"""Run QRTE and QRTE-PMTE for H2."""

from __future__ import annotations

import argparse

from _common import Timer, add_common_output_arg, h2_geometry, require_modules, write_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bond-length", type=float, default=1.5, help="H-H bond length in Angstrom.")
    parser.add_argument("--delta-t", type=float, default=0.1, help="Real-time evolution step.")
    parser.add_argument("--n-steps", type=int, default=2, help="Number of evolution steps.")
    parser.add_argument("--device-name", type=str, default="default.qubit", help="PennyLane device name.")
    add_common_output_arg(parser, "qrte_h2")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not require_modules("pennylane", "pyscf", "scipy"):
        return 0

    import QCANT

    symbols, geometry = h2_geometry(args.bond_length)
    common = dict(
        symbols=symbols,
        geometry=geometry,
        delta_t=args.delta_t,
        n_steps=args.n_steps,
        active_electrons=2,
        active_orbitals=2,
        basis="sto-3g",
        charge=0,
        spin=0,
        device_name=args.device_name,
        trotter_steps=1,
    )
    with Timer() as timer:
        qrte_energies, qrte_basis, qrte_times = QCANT.qrte(**common)
        pmte_energies, pmte_basis, pmte_times = QCANT.qrte_pmte(**common)

    print("QRTE minimum energy:", float(qrte_energies[0]))
    print("QRTE-PMTE minimum energy:", float(pmte_energies[0]))
    write_summary(
        args.output_dir,
        {
            "algorithm": "qrte_family",
            "symbols": symbols,
            "geometry": geometry,
            "device_name": args.device_name,
            "qrte": {"energies": qrte_energies, "basis_shape": qrte_basis.shape, "times": qrte_times},
            "qrte_pmte": {"energies": pmte_energies, "basis_shape": pmte_basis.shape, "times": pmte_times},
            "elapsed_s": timer.elapsed_s,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
