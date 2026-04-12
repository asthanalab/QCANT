# Outputs

## Core files

### `resolved_config.json`

The final config after CLI overrides are applied.

### `summary.json`

High-level numeric summary of the selected mode.

## TEPID outputs

### `tepid_history.csv`

One row per TEPID iteration.

Fields include:

- `iteration`
- `free_energy_hartree`
- `entropy`
- `gradient_abs`
- `basis_e0_hartree`
- `basis_e1_hartree`
- `basis_e2_hartree`
- `basis_ee1_hartree`
- `basis_ee2_hartree`
- `selected_pool_index`
- `selected_excitation`
- optimizer status fields

### `tepid_basis_states.csv`

One row per basis state per iteration.

Fields include:

- `iteration`
- `state_index`
- `energy_hartree`
- `thermal_weight`
- `occupation`

### `tepid_ansatz.json`

Saved ansatz file that can be reused later in standalone `qsceom` mode.

Fields include:

- `params`
- `ash_excitation`
- `ansatz_type`
- `pool_type`
- `metadata`

## qscEOM outputs

### `qsceom_spectrum.csv`

One row per qscEOM root.

Fields include:

- `state_index`
- `energy_hartree`
- `excitation_energy_hartree`

### `tepid_qsceom_by_iteration.csv`

Written only when `qsceom.each_iteration=true` or `--qsceom-each-iteration`
is used in `tepid_qsceom` mode.

Fields include:

- `iteration`
- `state_index`
- `energy_hartree`
- `excitation_energy_hartree`
- `tepid_free_energy_hartree`
- `tepid_entropy`

## Exact reference output

### `exact_sector_fci_roots.csv`

Exact roots in the fixed `N_alpha = N_beta = active_electrons / 2` sector.

This is the reference used in the standalone summaries for low-state error
reporting.
