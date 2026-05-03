# Standalone Examples

This folder contains runnable examples built on top of the standalone package.

## H4 3.0 A example

Run:

```bash
python -m standalone.tepid_qsceom.examples.run_h4_3p0A_examples
```

You can still run it as a file path from the repo root if needed:

```bash
python standalone/tepid_qsceom/examples/run_h4_3p0A_examples.py
```

This generates three example output folders under:

`standalone/tepid_qsceom/examples/outputs/h4_3p0A_examples`

The output directory is generated at runtime and ignored by git.

- `tepid`
- `qsceom`
- `tepid_qsceom`

And writes one representative PNG for each mode:

- `tepid/h4_3p0A_tepid_basis_window.png`
- `qsceom/h4_3p0A_qsceom_spectrum.png`
- `tepid_qsceom/h4_3p0A_tepid_qsceom_spectrum.png`

All three use the same molecule:

- linear `H4`
- spacing `3.0 Å`
- basis `sto-3g`
- charge `0`
- active space `4e, 4o`

The plots overlay exact fixed-sector FCI roots as light grey dashed lines.

The bundled `tepid` and `tepid_qsceom` examples use `20` TEPID iterations.
