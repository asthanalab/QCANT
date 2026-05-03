"""Run QRTE, QRTE-PMTE, QKUD, and exact Krylov for H2."""

from __future__ import annotations

import argparse

from _common import Timer, add_common_output_arg, h2_geometry, require_modules, write_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bond-length", type=float, default=1.5, help="H-H bond length in Angstrom.")
    parser.add_argument("--n-steps", type=int, default=2, help="Number of basis-growth steps.")
    parser.add_argument("--delta-t", type=float, default=0.1, help="QRTE time step.")
    parser.add_argument("--epsilon", type=float, default=0.1, help="QKUD epsilon.")
    add_common_output_arg(parser, "krylov_family_h2")
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
        active_electrons=2,
        active_orbitals=2,
        basis="sto-3g",
        charge=0,
        spin=0,
    )
    with Timer() as timer:
        qrte_energies, _qrte_basis, qrte_times = QCANT.qrte(
            **common,
            delta_t=args.delta_t,
            n_steps=args.n_steps,
            device_name="default.qubit",
            trotter_steps=1,
        )
        pmte_energies, _pmte_basis, pmte_times = QCANT.qrte_pmte(
            **common,
            delta_t=args.delta_t,
            n_steps=args.n_steps,
            device_name="default.qubit",
            trotter_steps=1,
        )
        qkud_energies, _qkud_basis, qkud_history = QCANT.qkud(
            **common,
            n_steps=args.n_steps,
            epsilon=args.epsilon,
            array_backend="numpy",
            return_min_energy_history=True,
        )
        krylov_energies, _krylov_basis, krylov_history = QCANT.exact_krylov(
            **common,
            n_steps=args.n_steps,
            array_backend="numpy",
            return_min_energy_history=True,
        )

    print("QRTE min:", float(qrte_energies[0]))
    print("QRTE-PMTE min:", float(pmte_energies[0]))
    print("QKUD min:", float(qkud_energies[0]))
    print("Exact Krylov min:", float(krylov_energies[0]))
    write_summary(
        args.output_dir,
        {
            "algorithm": "krylov_family",
            "symbols": symbols,
            "geometry": geometry,
            "qrte": {"energies": qrte_energies, "times": qrte_times},
            "qrte_pmte": {"energies": pmte_energies, "times": pmte_times},
            "qkud": {"energies": qkud_energies, "min_energy_history": qkud_history},
            "exact_krylov": {"energies": krylov_energies, "min_energy_history": krylov_history},
            "elapsed_s": timer.elapsed_s,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
