# QCANT Development Tools

This directory contains environment files and helper scripts used for local
development, CI, docs builds, and accelerator smoke testing.

## Environments

- `conda-envs/qcant.yaml`: standard Python 3.11/3.12 CPU development stack.
- `conda-envs/qcant-gpu.yaml`: CUDA 12 oriented GPU stack for Talon GPU nodes.
- `conda-envs/test_env.yaml`: lightweight CI/test environment seed.

Recommended local setup:

```bash
conda env create -f devtools/conda-envs/qcant.yaml
conda activate qcant
python -m pip install -e . --no-deps
```

GPU setup:

```bash
conda env create -f devtools/conda-envs/qcant-gpu.yaml
conda activate qcant-gpu
python -m pip install -e . --no-deps
```

## Helper Scripts

- `scripts/create_conda_env.py`: creates conda environments from starter
  specifications and command-line Python/env-name choices.

## Release Checks

Before opening a release PR or tagging a release candidate, run:

```bash
python -m pytest -q QCANT/tests
python -m compileall -q QCANT standalone scripts examples
cd docs && make html
python -m build
python -m twine check dist/*
```

Generated example outputs, standalone workflow outputs, and timestamped
benchmark directories should not be committed. Curated benchmark references
belong under `benchmarks/reference/`.
