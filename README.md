# QCANT

[![CI](https://github.com/asthanalab/QCANT/actions/workflows/CI.yaml/badge.svg?branch=main)](https://github.com/asthanalab/QCANT/actions/workflows/CI.yaml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/asthanalab/QCANT/branch/main/graph/badge.svg)](https://codecov.io/gh/asthanalab/QCANT/branch/main)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://asthanalab.github.io/QCANT/)

QCANT is a research software package for near-term quantum chemistry algorithms,
statevector workflows, and accelerator studies. It combines PySCF/PennyLane
chemistry setup with ADAPT-style ansatz construction, qscEOM projected spectra,
Krylov dynamics, finite-temperature TEPID workflows, optional Qulacs CPU
acceleration, and explicit single-GPU execution paths.

The project is being prepared as an API-compatible release candidate. Existing
CPU behavior remains the default; GPU acceleration is opt-in.

## What QCANT Provides

| Area | Public entry points | Notes |
| --- | --- | --- |
| Adaptive ansatzes | `QCANT.adapt_vqe`, `QCANT.gcim`, `QCANT.tepid_adapt`, `QCANT.cvqe` | Active-space chemistry workflows for ground-state and finite-temperature studies. |
| Spectroscopy | `QCANT.qscEOM` | Projected qscEOM spectra from ADAPT/TEPID ansatzes or explicit excitations. |
| Krylov dynamics | `QCANT.qrte`, `QCANT.qrte_pmte`, `QCANT.qkud`, `QCANT.exact_krylov`, `QCANT.adaptKrylov` | Real-time, unitary-decomposition, exact Krylov, and ADAPT+Krylov workflows. |
| Accelerators | `*_qulacs`, `device_name="lightning.gpu"`, `array_backend="cupy"` | Qulacs is CPU-only; GPU support is single-GPU and explicit. |
| Standalone workflow | `python -m standalone.tepid_qsceom` | Self-contained TEPID/qscEOM CLI and configs. |

## Install

QCANT supports Python 3.11 and 3.12.

For users:

```bash
python -m pip install QCANT
```

For development from this repository:

```bash
conda env create -f devtools/conda-envs/qcant.yaml
conda activate qcant
python -m pip install -e . --no-deps
```

Optional CPU simulator acceleration:

```bash
python -m pip install -e ".[qulacs]"
```

Optional single-GPU acceleration:

```bash
python -m pip install -e ".[gpu]"
```

For Talon GPU nodes, start from the CUDA 12 environment:

```bash
conda env create -f devtools/conda-envs/qcant-gpu.yaml
conda activate qcant-gpu
python -m pip install -e . --no-deps
```

## Quickstart

```python
import numpy as np
import QCANT

symbols = ["H", "H"]
geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]])

params, excitations, energies = QCANT.adapt_vqe(
    symbols=symbols,
    geometry=geometry,
    adapt_it=1,
    basis="sto-3g",
    charge=0,
    spin=0,
    active_electrons=2,
    active_orbitals=2,
    device_name="default.qubit",
    optimizer_maxiter=25,
)

roots = QCANT.qscEOM(
    symbols=symbols,
    geometry=geometry,
    active_electrons=2,
    active_orbitals=2,
    charge=0,
    ansatz=(params, excitations, energies),
    basis="sto-3g",
    shots=0,
)

print("ADAPT energy:", energies[-1])
print("qscEOM roots:", roots[0])
```

More runnable inputs are in `examples/`:

```bash
python examples/run_adapt_vqe_h2.py
python examples/run_qsceom_h2.py
python examples/run_tepid_adapt_h2.py
python examples/run_gpu_dense_hchain.py --atoms 4
python examples/run_standalone_tepid_qsceom_h2.py
```

Generated example outputs are written under `examples/outputs/` and are ignored
by git.

## GPU Acceleration

QCANT has two explicit GPU paths:

- PennyLane device acceleration with `device_name="lightning.gpu"` for
  `adapt_vqe`, `qscEOM`, `gcim`, `qrte`, and `qrte_pmte`.
- CuPy dense linear algebra with `array_backend="cupy"` for `cvqe`,
  `tepid_adapt`, `qkud`, `exact_krylov`, and the standalone `tepid_qsceom`
  workflow.

CPU behavior is preserved when no GPU option is passed. Sparse paths remain
CPU-only in this release:

- `qscEOM(projector_backend="sparse_number_preserving")`
- `qkud(..., use_sparse=True)`
- `exact_krylov(..., use_sparse=True)`
- all `*_qulacs` entry points

One GPU per Slurm task is the supported model. On Talon, use `talon32`,
`talon33`, or `talon35`; do not use `talon04` for GPU work.

```bash
PARTITION=talon-gpu32 NODE=talon32 GPUS=1 \
  scripts/run_on_talon_gpu.sh \
  'python scripts/benchmark_gpu_speedups.py --profile h6 --outdir benchmarks/gpubenchmark --gpu-device lightning.gpu'
```

See `docs/gpu_acceleration.rst` for install notes, fallback behavior, and
benchmarking guidance.

## Benchmarks

QCANT benchmark scripts report both raw runtime and speedup relative to the CPU
baseline:

```bash
python scripts/benchmark_gpu_speedups.py --profile h6 --repeats 2 --outdir benchmarks/gpubenchmark
python scripts/benchmark_gpu_speedups.py --profile small --repeats 2 --outdir benchmarks/gpu_speedups_small
python scripts/benchmark_hchain_gpu_capability.py --atoms 4 6
python scripts/benchmark_hchain_sector_gpu_capability.py --atoms 6 8
python scripts/benchmark_hchain_hybrid_capability.py --atoms 6
```

The release H6 benchmark writes `benchmarks/gpubenchmark/h6_speedups.csv`,
`benchmarks/gpubenchmark/h6_speedup_summary.png`, and one speedup plot per code
path under `benchmarks/gpubenchmark/plots/`. Qulacs rows are labeled
`qulacs_cpu` because they are CPU accelerator measurements, not GPU runs.

Important interpretation:

- End-to-end small chemistry jobs may not be faster on GPU because PySCF setup,
  device construction, and host/device transfers dominate.
- Dense full-space H8 in STO-3G is not a practical V100 target because the full
  dense Hamiltonian is too large.
- The useful GPU capability appears in the dense or sector linear-algebra hot
  loops once the Hamiltonian/batches are already built.

Curated benchmark references live under `benchmarks/reference/`. Timestamped
benchmark output directories are ignored.

## Documentation

Hosted docs:

- https://asthanalab.github.io/QCANT/

Build locally:

```bash
cd docs
make html
```

The generated site is written to `docs/_build/html`.

## Release Checklist

Before tagging:

```bash
python -m pytest -q QCANT/tests
python -m compileall QCANT standalone scripts examples
cd docs && make html
python -m build
python -m twine check dist/*
```

This repository publishes to PyPI from GitHub Actions through trusted
publishing when a version tag is pushed.

## Acknowledgements

QCANT's repository structure began from the Computational Molecular Science
Python Cookiecutter. The scientific implementation, acceleration work, and
release documentation are maintained by Asthana Lab contributors.
