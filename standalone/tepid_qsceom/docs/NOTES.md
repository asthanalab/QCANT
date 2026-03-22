# Implementation Notes

## Computational basis

The default truncated computational basis is:

- Hartree-Fock reference
- all single excitations
- all double excitations

That is built locally with `inite(...)` plus the Hartree-Fock occupation.

## Pool versus basis

The adaptive pool and the computational basis are separate choices.

Examples:

- `pool_type = fermionic_sd`
  Uses fermionic single/double excitation gates in the ansatz.

- `pool_type = qe`
  Uses qubit `SingleExcitation` and `DoubleExcitation` gates in the ansatz.

In both cases, the computational basis can still remain HF plus singles and
doubles.

## What the TEPID energies mean

`tepid_basis_states.csv` stores sorted transformed-basis energy expectations:

```text
<psi_i(theta)| H |psi_i(theta)>
```

These are not the same object as:

- exact FCI eigenvalues
- qscEOM projected eigenvalues

This distinction matters when interpreting TEPID-only excited-state quality.

## What qscEOM does here

The standalone qscEOM implementation is analytic and projected-subspace based:

1. build the selected computational basis
2. apply the ansatz to each basis vector
3. form the projected Hamiltonian matrix exactly from statevectors
4. diagonalize that matrix

That is why the standalone qscEOM code is much shorter than the original
package implementation.

## Exact-sector references

The exact reference roots are computed in the closed-shell fixed-particle
sector:

- `N_alpha = active_electrons / 2`
- `N_beta = active_electrons / 2`

This is currently the supported reference path in the standalone code.

## Current limitations

- analytic-only
- closed-shell exact-sector helper
- no plotting helpers
- no shot-noise support
- no multiprocessing path for qscEOM

## Why one big `core.py`

This is intentional:

- easier to move to another repo
- easier to audit for unwanted `QCANT` dependencies
- easier to debug from a single file
