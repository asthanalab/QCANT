# Standalone TEPID/qscEOM

This folder is intentionally self-contained. It does not import from `QCANT`.
It contains enough local code to run:

- TEPID-ADAPT
- analytic qscEOM
- TEPID-ADAPT followed by qscEOM

The standalone workflow mirrors the package-level dense backend policy:
`array_backend="auto"` preserves CPU behavior, while `"cupy"` requests opt-in
GPU dense linear algebra when CuPy is installed and a CUDA-visible GPU exists.

## Quick start

Install the external dependencies:

```bash
python -m pip install -r standalone/tepid_qsceom/requirements.txt
```

Run one of the three supported modes:

```bash
python -m standalone.tepid_qsceom --config standalone/tepid_qsceom/configs/h4_linear_3p0A.json --mode tepid
python -m standalone.tepid_qsceom --config standalone/tepid_qsceom/configs/h4_linear_3p0A.json --mode qsceom
python -m standalone.tepid_qsceom --config standalone/tepid_qsceom/configs/h4_linear_3p0A.json --mode tepid_qsceom
```

Useful overrides:

```bash
python -m standalone.tepid_qsceom \
  --config standalone/tepid_qsceom/configs/h4_linear_3p0A.json \
  --mode tepid_qsceom \
  --pool-type qe \
  --beta 3.0 \
  --adapt-it 5 \
  --qsceom-each-iteration
```

## Documentation

All standalone documentation lives in this folder:

- [Install And Run](./docs/INSTALL.md)
- [Collaborator Handoff](./docs/COLLABORATOR.md)
- [Overview](./docs/OVERVIEW.md)
- [Configuration](./docs/CONFIG.md)
- [Run Modes](./docs/MODES.md)
- [Outputs](./docs/OUTPUTS.md)
- [API Notes](./docs/API.md)
- [Implementation Notes](./docs/NOTES.md)
- [Package Structure](./docs/STRUCTURE.md)
- [Examples](./examples/README.md)

## H4 Example

Generate the bundled H4 3.0 A examples for all three modes:

```bash
python -m standalone.tepid_qsceom.examples.run_h4_3p0A_examples
```

This writes:

- `standalone/tepid_qsceom/examples/outputs/h4_3p0A_examples/tepid/h4_3p0A_tepid_basis_window.png`
- `standalone/tepid_qsceom/examples/outputs/h4_3p0A_examples/qsceom/h4_3p0A_qsceom_spectrum.png`
- `standalone/tepid_qsceom/examples/outputs/h4_3p0A_examples/tepid_qsceom/h4_3p0A_tepid_qsceom_spectrum.png`

## File layout

- `core.py`: self-contained implementation
- `__main__.py`: CLI entrypoint
- `configs/`: example input files
- `examples/`: runnable example scripts and generated example outputs
- `docs/`: standalone documentation
- `outputs/`: smoke-test outputs and user runs
  (generated outputs are ignored by git)

Recommended install/run details are in [docs/INSTALL.md](./docs/INSTALL.md).

## Scope

This standalone package is analytic-only:

- TEPID supports `shots=0` only
- qscEOM uses exact projected-matrix diagonalization from statevectors

The truncated computational basis defaults to the Hartree-Fock reference plus
all single and double excitations, matching the qscEOM basis construction used
elsewhere in this repo.
