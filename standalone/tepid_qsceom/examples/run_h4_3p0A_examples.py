"""Generate standalone H4 3.0 A example outputs for tepid, qsceom, and tepid_qsceom."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from standalone.tepid_qsceom.core import run_workflow
from standalone.tepid_qsceom.plotting import (
    plot_qsceom_spectrum_levels,
    plot_tepid_basis_by_iteration,
    plot_tepid_qsceom_by_iteration,
)


ROOT = Path(__file__).resolve().parent
PKG_ROOT = ROOT.parent
CONFIG_PATH = PKG_ROOT / "configs" / "h4_linear_3p0A.json"
OUTPUT_ROOT = ROOT / "outputs" / "h4_3p0A_examples"


def load_base_config() -> dict:
    """Load the base H4 config."""
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["_config_path"] = str(CONFIG_PATH.resolve())
    return payload


def reset_output_dir(path: Path) -> None:
    """Remove and recreate an output directory so examples are fresh."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_tepid_summary_from_combined(combined_dir: Path, tepid_dir: Path) -> None:
    """Extract the TEPID-only summary fields from a combined-run summary."""
    combined_summary = json.loads((combined_dir / "summary.json").read_text(encoding="utf-8"))
    tepid_summary = {
        "mode": "tepid",
        "final_free_energy_hartree": combined_summary["tepid_final_free_energy_hartree"],
        "final_entropy": combined_summary["tepid_final_entropy"],
        "selected_operators": combined_summary["tepid_selected_operators"],
    }
    with (tepid_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(tepid_summary, handle, indent=2)
        handle.write("\n")


def main() -> None:
    """Run standalone H4 examples and generate plots."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    base = load_base_config()

    qsceom_config = dict(base)
    qsceom_config["tepid"] = dict(base["tepid"])
    qsceom_config["qsceom"] = dict(base["qsceom"])
    qsceom_config["qsceom"]["max_roots"] = 0

    combined_config = dict(base)
    combined_config["tepid"] = dict(base["tepid"])
    combined_config["qsceom"] = dict(base["qsceom"])
    combined_config["tepid"]["adapt_it"] = 20
    combined_config["tepid"]["optimizer_maxiter"] = 10
    combined_config["qsceom"]["each_iteration"] = True
    combined_config["qsceom"]["max_roots"] = 0

    tepid_dir = OUTPUT_ROOT / "tepid"
    qsceom_dir = OUTPUT_ROOT / "qsceom"
    combined_dir = OUTPUT_ROOT / "tepid_qsceom"

    reset_output_dir(tepid_dir)
    reset_output_dir(qsceom_dir)
    reset_output_dir(combined_dir)

    print("Running standalone H4 tepid+qsceom example...", flush=True)
    run_workflow(combined_config, mode="tepid_qsceom", output_dir_override=combined_dir)
    plot_tepid_qsceom_by_iteration(
        tepid_qsceom_csv=combined_dir / "tepid_qsceom_by_iteration.csv",
        exact_csv=combined_dir / "exact_sector_fci_roots.csv",
        out_path=combined_dir / "h4_3p0A_tepid_qsceom_spectrum.png",
        title="Standalone H4 TEPID + qscEOM spectrum by iteration",
    )

    print("Preparing standalone H4 tepid example from combined outputs...", flush=True)
    for filename in [
        "exact_sector_fci_roots.csv",
        "resolved_config.json",
        "tepid_ansatz.json",
        "tepid_basis_states.csv",
        "tepid_history.csv",
    ]:
        shutil.copy2(combined_dir / filename, tepid_dir / filename)
    write_tepid_summary_from_combined(combined_dir, tepid_dir)
    plot_tepid_basis_by_iteration(
        tepid_basis_csv=tepid_dir / "tepid_basis_states.csv",
        exact_csv=tepid_dir / "exact_sector_fci_roots.csv",
        out_path=tepid_dir / "h4_3p0A_tepid_basis_window.png",
        title="Standalone H4 TEPID basis energies by iteration",
    )

    print("Running standalone H4 qsceom example...", flush=True)
    run_workflow(qsceom_config, mode="qsceom", output_dir_override=qsceom_dir)
    plot_qsceom_spectrum_levels(
        qsceom_csv=qsceom_dir / "qsceom_spectrum.csv",
        exact_csv=qsceom_dir / "exact_sector_fci_roots.csv",
        out_path=qsceom_dir / "h4_3p0A_qsceom_spectrum.png",
        title="Standalone H4 qscEOM spectrum",
    )

    print(f"Wrote example outputs to {OUTPUT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
