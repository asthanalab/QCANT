# Install And Run

This standalone folder can be used in two ways.

## Option 1: Run In This Repo

Use this when you are already inside the `QCANT` checkout and just want to run
the standalone workflows.

Create and activate an environment, then install the external dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r standalone/tepid_qsceom/requirements.txt
```

Run the CLI:

```bash
python -m standalone.tepid_qsceom \
  --config standalone/tepid_qsceom/configs/h4_linear_3p0A.json \
  --mode tepid_qsceom
```

Run the bundled H4 example set:

```bash
python -m standalone.tepid_qsceom.examples.run_h4_3p0A_examples
```

## Option 2: Install QCANT In Editable Mode

Use this when you want the `standalone.tepid_qsceom` module importable after
installation.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

After that, the same commands work:

```bash
python -m standalone.tepid_qsceom --config standalone/tepid_qsceom/configs/h2_sto3g.json --mode tepid
python -m standalone.tepid_qsceom --config standalone/tepid_qsceom/configs/h2_sto3g.json --mode qsceom
python -m standalone.tepid_qsceom --config standalone/tepid_qsceom/configs/h2_sto3g.json --mode tepid_qsceom
```

## Common Commands

TEPID only:

```bash
python -m standalone.tepid_qsceom \
  --config standalone/tepid_qsceom/configs/h4_linear_3p0A.json \
  --mode tepid \
  --adapt-it 20 \
  --beta 3.0
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

Using the qubit-excitation pool:

```bash
python -m standalone.tepid_qsceom \
  --config standalone/tepid_qsceom/configs/h4_linear_3p0A.json \
  --mode tepid_qsceom \
  --pool-type qe
```

Using a saved ansatz for qscEOM:

```bash
python -m standalone.tepid_qsceom \
  --config standalone/tepid_qsceom/configs/h4_linear_3p0A.json \
  --mode qsceom \
  --ansatz-file standalone/tepid_qsceom/examples/outputs/h4_3p0A_examples/tepid_qsceom/tepid_ansatz.json
```

## Output Locations

Default outputs go to the `output_dir` field in the config file.

Override the destination with:

```bash
python -m standalone.tepid_qsceom \
  --config standalone/tepid_qsceom/configs/h4_linear_3p0A.json \
  --mode tepid_qsceom \
  --output-dir standalone/tepid_qsceom/outputs/my_h4_run
```

## Notes

- This standalone code path is analytic-only.
- TEPID uses a truncated computational basis built from HF plus single and
  double excitations unless `basis_occupations` is provided explicitly.
- The bundled H4 example script currently uses `20` TEPID iterations for the
  `tepid` and `tepid_qsceom` examples.
