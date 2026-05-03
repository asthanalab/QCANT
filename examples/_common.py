"""Shared utilities for QCANT runnable examples."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"


def h2_geometry(bond_length: float = 1.5):
    """Return an H2 geometry in Angstrom."""
    return ["H", "H"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, bond_length]], dtype=float)


def h_chain_geometry(n_atoms: int, bond_length: float = 1.5):
    """Return a linear H-chain geometry in Angstrom."""
    if n_atoms < 2:
        raise ValueError("n_atoms must be >= 2")
    symbols = ["H"] * int(n_atoms)
    geometry = np.array([[0.0, 0.0, bond_length * i] for i in range(n_atoms)], dtype=float)
    return symbols, geometry


def add_common_output_arg(parser: argparse.ArgumentParser, example_name: str) -> None:
    """Add the standard output directory argument to an example parser."""
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_ROOT / example_name,
        help="Directory for generated JSON/CSV outputs.",
    )


def require_modules(*module_names: str) -> bool:
    """Return False and print a clear skip message if optional modules are missing."""
    missing = []
    for module_name in module_names:
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing.append(module_name)
    if missing:
        print("Skipping example because optional dependencies are missing:", ", ".join(missing))
        return False
    return True


def to_plain_data(value: Any):
    """Convert nested numerical objects into JSON-serializable data."""
    if isinstance(value, dict):
        return {str(key): to_plain_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain_data(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    try:
        import cupy as cp
    except ImportError:
        cp = None
    if cp is not None and isinstance(value, cp.ndarray):
        return cp.asnumpy(value).tolist()
    if cp is not None and isinstance(value, cp.generic):
        return value.item()
    return value


def write_summary(output_dir: Path, payload: dict[str, Any], *, filename: str = "summary.json") -> Path:
    """Write an example summary JSON file and return the path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    path.write_text(json.dumps(to_plain_data(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote summary: {path}")
    return path


class Timer:
    """Tiny context manager for human-readable example timings."""

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *_exc_info):
        self.elapsed_s = time.perf_counter() - self.start
