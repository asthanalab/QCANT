# Collaborator Handoff

This standalone bundle is meant to be sent without the rest of the `QCANT`
repository.

## What To Send

Send the zip archive built from the standalone package. After unzip, the
collaborator will have a top-level folder that contains:

- `README.md`
- `standalone/`

Inside `standalone/tepid_qsceom` they will find:

- the standalone code
- configs
- docs
- the bundled H4 example driver
- the bundled H4 example outputs

## Install

From the unzipped folder:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r standalone/tepid_qsceom/requirements.txt
```

## Run The Example

From the unzipped folder:

```bash
python -m standalone.tepid_qsceom.examples.run_h4_3p0A_examples
```

This regenerates the bundled H4 example set.

## Run The CLI Directly

TEPID only:

```bash
python -m standalone.tepid_qsceom \
  --config standalone/tepid_qsceom/configs/h4_linear_3p0A.json \
  --mode tepid
```

qscEOM only:

```bash
python -m standalone.tepid_qsceom \
  --config standalone/tepid_qsceom/configs/h4_linear_3p0A.json \
  --mode qsceom
```

TEPID followed by qscEOM:

```bash
python -m standalone.tepid_qsceom \
  --config standalone/tepid_qsceom/configs/h4_linear_3p0A.json \
  --mode tepid_qsceom \
  --adapt-it 20 \
  --beta 3.0 \
  --qsceom-each-iteration
```

## Expected Example Outputs

After the bundled H4 example finishes, these plots should exist:

- `standalone/tepid_qsceom/examples/outputs/h4_3p0A_examples/tepid/h4_3p0A_tepid_basis_window.png`
- `standalone/tepid_qsceom/examples/outputs/h4_3p0A_examples/qsceom/h4_3p0A_qsceom_spectrum.png`
- `standalone/tepid_qsceom/examples/outputs/h4_3p0A_examples/tepid_qsceom/h4_3p0A_tepid_qsceom_spectrum.png`

## Notes

- The workflows are analytic-only.
- The default truncated computational basis is the HF determinant plus single
  and double excitations.
- The bundled H4 example uses linear `H4` at `3.0 A`, `sto-3g`, `4e,4o`.
