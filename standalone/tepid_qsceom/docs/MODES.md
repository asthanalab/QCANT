# Run Modes

## `tepid`

Runs only TEPID-ADAPT.

Command:

```bash
python -m standalone.tepid_qsceom \
  --config standalone/tepid_qsceom/configs/h4_linear_3p0A.json \
  --mode tepid
```

Writes:

- resolved config
- exact reference roots if requested
- TEPID history
- TEPID basis-state energies and weights
- saved ansatz
- summary

## `qsceom`

Runs analytic qscEOM from:

- an explicit ansatz file, or
- the identity ansatz if no ansatz file is provided

Command:

```bash
python -m standalone.tepid_qsceom \
  --config standalone/tepid_qsceom/configs/h4_linear_3p0A.json \
  --mode qsceom \
  --ansatz-file standalone/tepid_qsceom/outputs/some_run/tepid_ansatz.json
```

Writes:

- resolved config
- exact reference roots if requested
- qscEOM spectrum
- summary

## `tepid_qsceom`

Runs TEPID-ADAPT and then qscEOM on the final TEPID ansatz.

Command:

```bash
python -m standalone.tepid_qsceom \
  --config standalone/tepid_qsceom/configs/h4_linear_3p0A.json \
  --mode tepid_qsceom
```

Optional per-iteration qscEOM:

```bash
python -m standalone.tepid_qsceom \
  --config standalone/tepid_qsceom/configs/h4_linear_3p0A.json \
  --mode tepid_qsceom \
  --qsceom-each-iteration
```

Writes:

- all TEPID outputs
- final qscEOM spectrum
- optional qscEOM-by-iteration table
- summary

## Useful CLI overrides

- `--adapt-it`
- `--beta`
- `--temperature`
- `--pool-type`
- `--optimizer-maxiter`
- `--qsceom-each-iteration`
- `--qsceom-max-roots`
- `--output-dir`
- `--ansatz-file`
