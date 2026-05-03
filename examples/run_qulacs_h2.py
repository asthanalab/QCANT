"""Run optional Qulacs CPU-accelerated H2 examples."""

from __future__ import annotations

import argparse

from _common import Timer, add_common_output_arg, h2_geometry, require_modules, write_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bond-length", type=float, default=0.74, help="H-H bond length in Angstrom.")
    parser.add_argument("--optimizer-maxiter", type=int, default=10, help="SciPy optimizer iteration limit.")
    add_common_output_arg(parser, "qulacs_h2")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not require_modules("pennylane", "pyscf", "scipy", "qulacs"):
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
        qkud_energies, _qkud_basis = QCANT.qkud_qulacs(**common, n_steps=1, epsilon=0.1)
        qrte_energies, _qrte_basis, qrte_times = QCANT.qrte_qulacs(**common, delta_t=0.1, n_steps=1)
        params, excitations, adapt_energies = QCANT.adapt_vqe_qulacs(
            **common,
            adapt_it=1,
            optimizer_maxiter=args.optimizer_maxiter,
            parallel_gradients=True,
            max_workers=2,
        )

    print("Qulacs QKUD min:", float(qkud_energies[0]))
    print("Qulacs QRTE min:", float(qrte_energies[0]))
    print("Qulacs ADAPT final:", float(adapt_energies[-1]))
    write_summary(
        args.output_dir,
        {
            "algorithm": "qulacs_cpu_examples",
            "symbols": symbols,
            "geometry": geometry,
            "qkud_energies": qkud_energies,
            "qrte_energies": qrte_energies,
            "qrte_times": qrte_times,
            "adapt_params": params,
            "adapt_excitations": excitations,
            "adapt_energies": adapt_energies,
            "elapsed_s": timer.elapsed_s,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
