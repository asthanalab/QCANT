# API Notes

The standalone package exposes the following imports from
`standalone.tepid_qsceom`.

## `load_config(path)`

Loads a JSON config file and stores the resolved config path internally for
relative-path handling.

## `save_ansatz(path, ...)`

Writes a standalone ansatz JSON file that can be reused in later runs.

## `load_ansatz(path)`

Reads a standalone ansatz JSON file and reconstructs:

- `params`
- `ash_excitation`
- ansatz metadata

## `tepid_adapt(...)`

Runs standalone ancilla-free TEPID-ADAPT.

Returns:

- `params`
- `ash_excitation`
- `free_energies`
- optional `details`

## `qsceom(...)`

Runs standalone analytic qscEOM.

It accepts either:

- `params` plus `ash_excitation`
- `ansatz=(params, ash_excitation, anything)`

Returns:

- qscEOM eigenvalues
- optional projected-matrix details

## `tepid_qsceom(...)`

Runs TEPID first, then qscEOM on the resulting ansatz.

Returns a dictionary with:

- `tepid`
- `qsceom`
- optional `per_iteration`

## `run_workflow(config, mode, ...)`

High-level function used by the CLI.

It:

- resolves config overrides
- runs the selected workflow
- writes outputs to disk
- returns a summary dictionary

## `main()`

CLI entrypoint used by:

```bash
python -m standalone.tepid_qsceom
```
