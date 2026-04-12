# Package Structure

This standalone code is organized as one small package plus local data,
documentation, and example drivers.

## Package Files

- `__init__.py`: public imports for the standalone API
- `__main__.py`: CLI entrypoint for `python -m standalone.tepid_qsceom`
- `core.py`: chemistry setup, basis construction, pools, TEPID-ADAPT, qscEOM,
  config loading, ansatz save/load, and CLI dispatch
- `plotting.py`: reusable plotting helpers for TEPID, qscEOM, and
  TEPID+qscEOM outputs
- `requirements.txt`: external dependencies for the standalone workflows

## Data And Docs

- `configs/`: example JSON inputs for different molecules
- `docs/`: standalone documentation for installation, configuration, modes,
  outputs, API, and implementation notes
- `examples/`: runnable example scripts and generated example outputs
- `outputs/`: smoke-test or user-generated outputs outside the example bundle

## Recommended Entry Points

CLI:

```bash
python -m standalone.tepid_qsceom --config ... --mode ...
```

Bundled example:

```bash
python -m standalone.tepid_qsceom.examples.run_h4_3p0A_examples
```

Python API:

```python
from standalone.tepid_qsceom import tepid_adapt, qsceom, tepid_qsceom
```

## Output Conventions

TEPID runs write:

- `tepid_history.csv`
- `tepid_basis_states.csv`
- `tepid_ansatz.json`
- `summary.json`

qscEOM runs write:

- `qsceom_spectrum.csv`
- `summary.json`

Combined runs write:

- all TEPID outputs
- `qsceom_spectrum.csv`
- `tepid_qsceom_by_iteration.csv` when `qsceom_each_iteration` is enabled

Exact conserved-sector reference roots are written to:

- `exact_sector_fci_roots.csv`
