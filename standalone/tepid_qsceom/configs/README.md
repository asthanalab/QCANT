# Config Files

This folder contains example JSON configs for the standalone workflows.

- `h2_sto3g.json`: small H2 smoke-test configuration
- `h4_linear_3p0A.json`: linear H4 with 3.0 A spacing

Each config defines:

- `symbols`
- `geometry`
- `basis`
- `charge`
- `active_electrons`
- `active_orbitals`
- `method`
- `output_dir`
- `tepid`
- `qsceom`

Use them directly with:

```bash
python -m standalone.tepid_qsceom --config standalone/tepid_qsceom/configs/h4_linear_3p0A.json --mode tepid_qsceom
```
