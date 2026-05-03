"""Run an optional dense-GPU H-chain Krylov example."""

from __future__ import annotations

import argparse

from _common import Timer, add_common_output_arg, h_chain_geometry, require_modules, write_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atoms", type=int, default=4, help="Number of hydrogen atoms in the linear chain.")
    parser.add_argument("--bond-length", type=float, default=1.5, help="H-H spacing in Angstrom.")
    parser.add_argument("--n-steps", type=int, default=1, help="Number of Krylov steps.")
    parser.add_argument("--array-backend", choices=["auto", "numpy", "cupy"], default="cupy")
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Return a nonzero exit code instead of skipping when CuPy/GPU is unavailable.",
    )
    add_common_output_arg(parser, "gpu_dense_hchain")
    return parser


def _cupy_available() -> tuple[bool, str]:
    try:
        import cupy as cp
    except ImportError:
        return False, "CuPy is not installed"
    try:
        count = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:
        return False, f"CuPy could not query CUDA devices: {exc}"
    if count < 1:
        return False, "no CUDA-visible GPU found"
    return True, f"{count} CUDA-visible GPU(s)"


def main() -> int:
    args = build_parser().parse_args()
    if not require_modules("pennylane", "pyscf", "scipy"):
        return 0
    if args.array_backend == "cupy":
        ok, message = _cupy_available()
        if not ok:
            print(f"Skipping GPU example: {message}")
            return 2 if args.require_gpu else 0
        print(f"GPU check passed: {message}")

    import QCANT

    symbols, geometry = h_chain_geometry(args.atoms, args.bond_length)
    with Timer() as timer:
        energies, basis_states, min_history = QCANT.exact_krylov(
            symbols=symbols,
            geometry=geometry,
            n_steps=args.n_steps,
            active_electrons=args.atoms,
            active_orbitals=args.atoms,
            basis="sto-3g",
            charge=0,
            spin=0,
            array_backend=args.array_backend,
            return_min_energy_history=True,
        )

    print(f"H{args.atoms} exact Krylov min energy:", float(energies[0]))
    write_summary(
        args.output_dir,
        {
            "algorithm": "gpu_dense_hchain_exact_krylov",
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
