QCANT
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/QCANT/workflows/CI/badge.svg)](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/QCANT/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/QCANT/branch/main/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/QCANT/branch/main)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://srivathsanps-quantum.github.io/QCANT/)


Utilities for near-term applications of quantum computing in chemistry and materials science.

This repository currently contains a lightweight, template-derived QCANT package. The public API is small
and intended to grow as project modules are added.

## Install

For development:

```bash
pip install -e .
```

## Quickstart

```python
import QCANT

print(QCANT.canvas())
```

## Documentation

Hosted documentation:

- https://srivathsanps-quantum.github.io/QCANT/

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
