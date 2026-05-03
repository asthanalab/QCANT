"""Run the standalone TEPID/qscEOM workflow for H2."""

from __future__ import annotations

import argparse
import json

from _common import REPO_ROOT, Timer, add_common_output_arg, require_modules, write_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapt-it", type=int, default=1, help="Number of standalone TEPID iterations.")
    parser.add_argument("--beta", type=float, default=2.0, help="Inverse temperature.")
    parser.add_argument("--optimizer-maxiter", type=int, default=25, help="SciPy optimizer iteration limit.")
    parser.add_argument("--array-backend", choices=["auto", "numpy", "cupy"], default="auto")
    add_common_output_arg(parser, "standalone_tepid_qsceom_h2")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not require_modules("pennylane", "pyscf", "scipy"):
        return 0

    from standalone.tepid_qsceom.core import load_config, run_workflow

    config_path = REPO_ROOT / "standalone" / "tepid_qsceom" / "configs" / "h2_sto3g.json"
    config = load_config(config_path)
    config["array_backend"] = args.array_backend
    config["tepid"]["adapt_it"] = args.adapt_it
    config["tepid"]["beta"] = args.beta
    config["tepid"]["optimizer_maxiter"] = args.optimizer_maxiter
    config["qsceom"]["max_roots"] = 0

    with Timer() as timer:
        summary = run_workflow(config, mode="tepid_qsceom", output_dir_override=args.output_dir)

    print(json.dumps(summary, indent=2, sort_keys=True))
    write_summary(
        args.output_dir,
        {
            "algorithm": "standalone_tepid_qsceom",
            "config": str(config_path),
            "array_backend": args.array_backend,
            "workflow_summary": summary,
            "elapsed_s": timer.elapsed_s,
        },
        filename="example_summary.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
