QCANT
==============================
[//]: # (Badges)
[![CI](https://github.com/asthanalab/QCANT/actions/workflows/CI.yaml/badge.svg?branch=main)](https://github.com/asthanalab/QCANT/actions/workflows/CI.yaml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/asthanalab/QCANT/branch/main/graph/badge.svg)](https://codecov.io/gh/asthanalab/QCANT/branch/main)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://asthanalab.github.io/QCANT/)


Utilities for near-term applications of quantum computing in chemistry and materials science.

This repository currently contains a lightweight, template-derived QCANT package. The public API is small
and intended to grow as project modules are added.

## Install

You can install QCANT from PyPI:

```bash
pip install QCANT
```

QCANT requires scientific Python dependencies (installed automatically when you `pip install QCANT`):

- `numpy<2`, `scipy<2`
- `pennylane`
- `pyscf`
- `autoray<0.7`

For development (recommended: conda env for the full stack):

```bash
conda env create -f devtools/conda-envs/qcant.yaml
conda activate qcant
pip install -e . --no-deps
```

For development (pip/venv):

```bash
pip install -e .
```

For users (once QCANT is published to PyPI):

```bash
pip install QCANT
```

## Release (PyPI)

This repo is configured to publish to PyPI from GitHub Actions using Trusted Publishing.

1. Create the project on PyPI and enable "Trusted Publishing" for:
	- Owner: `srivathsanps-quantum`
	- Repo: `QCANT`
	- Workflow: `publish-pypi.yml`
2. Tag a release (tag must start with a digit, e.g. `1.0.0`) and push the tag:

```bash
git tag 1.0.0
git push origin 1.0.0
```

That tag push triggers the publish workflow.

## Quickstart

```python
import QCANT

print(QCANT.canvas())
```

## Documentation

Hosted documentation:

- https://asthanalab.github.io/QCANT/

The documentation lives in `docs/` and is built with Sphinx:

```bash
cd docs
make html
```

The output will be in `docs/_build/html`.

### Copyright

Copyright (c) 2025, Asthana Lab


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.11.
