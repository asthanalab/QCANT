# Configuration

## Format

The program reads a single JSON config file.

Top-level keys:

- `symbols`
- `geometry`
- `basis`
- `charge`
- `spin`
- `active_electrons`
- `active_orbitals`
- `output_dir`
- `tepid`
- `qsceom`

## Minimal example

```json
{
  "symbols": ["H", "H"],
  "geometry": [
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.74]
  ],
  "basis": "sto-3g",
  "charge": 0,
  "spin": 0,
  "active_electrons": 2,
  "active_orbitals": 2,
  "output_dir": "../outputs/h2_sto3g",
  "tepid": {
    "adapt_it": 4,
    "beta": 3.0,
    "pool_type": "fermionic_sd",
    "include_identity": true,
    "optimizer_method": "BFGS",
    "optimizer_maxiter": 50
  },
  "qsceom": {
    "include_identity": true,
    "max_roots": 0,
    "compute_exact_sector": true,
    "each_iteration": false
  }
}
```

## Molecular fields

- `symbols`
  Atomic symbols, for example `["H", "H", "H", "H"]`.

- `geometry`
  Cartesian coordinates in Angstrom, shaped like `[[x, y, z], ...]`.

- `basis`
  Basis set name understood by PennyLane/PySCF.

- `charge`
  Total molecular charge.

- `spin`
  Stored in the config for clarity. The current standalone workflow uses the
  closed-shell active-space path and does not explicitly thread `spin` into the
  Hamiltonian builder.

- `active_electrons`
  Number of active electrons.

- `active_orbitals`
  Number of active spatial orbitals.

## `tepid` section

Supported keys:

- `adapt_it`
- `beta`
- `temperature`
- `pool_type`
- `include_identity`
- `basis_occupations`
- `pool_sample_size`
- `pool_seed`
- `gradient_eps`
- `gradient_tol`
- `optimizer_method`
- `optimizer_maxiter`

Notes:

- Provide exactly one of `beta` or `temperature`.
- `pool_type` supports `fermionic_sd`, `sd`, `fermionic`, `qubit_excitation`,
  `qe`, and `qubit`.
- `basis_occupations` is optional. If omitted, the computational basis defaults
  to HF plus singles and doubles.

## `qsceom` section

Supported keys:

- `include_identity`
- `basis_occupations`
- `max_roots`
- `compute_exact_sector`
- `each_iteration`
- `ansatz_file`
- `ansatz_type`

Notes:

- `max_roots <= 0` means keep the full qscEOM spectrum.
- `ansatz_file` is used only in `qsceom` mode when you want to load a saved
  standalone ansatz.
- `each_iteration` matters only in `tepid_qsceom` mode.

## Path resolution

- `output_dir` inside the config is resolved relative to the config file.
- CLI `--output-dir` is resolved relative to the current working directory.
- Config `ansatz_file` is resolved relative to the config file.
- CLI `--ansatz-file` is resolved relative to the current working directory.
