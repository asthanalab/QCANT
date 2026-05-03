# Building QCANT Documentation

QCANT docs are built with Sphinx and the PyData Sphinx theme.

## Conda

```bash
conda env create -f docs/requirements.yaml
conda activate docs_QCANT
make html
```

## Existing Development Environment

```bash
python -m pip install -e ".[docs]"
cd docs
make html
```

The generated site is written to `docs/_build/html`.

## Release Checks

Before a release candidate is tagged, build the docs from a clean environment
and confirm that these pages render correctly:

- landing page and getting started guide
- examples gallery
- GPU acceleration and benchmarking pages
- API reference/autosummary output
