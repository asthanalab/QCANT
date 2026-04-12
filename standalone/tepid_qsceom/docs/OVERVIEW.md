# Overview

## Purpose

`standalone/tepid_qsceom` is a self-contained mini-package for:

- running TEPID-ADAPT
- running analytic qscEOM
- running TEPID-ADAPT and then qscEOM on top of the resulting ansatz

It is designed so you can copy this folder elsewhere without needing the
larger `QCANT` package internals.

## What is included

- active-space molecular Hamiltonian construction through PennyLane/PySCF
- Hartree-Fock plus single/double excitation computational-basis construction
- fermionic and qubit-excitation (`qe`) adaptive pools
- ancilla-free TEPID-ADAPT free-energy optimization
- analytic projected-matrix qscEOM
- standalone ansatz save/load
- exact fixed-particle-sector reference roots for comparison

## What is not included

- shot-based TEPID
- hardware execution paths
- noisy simulation support
- plotting scripts
- any import from `QCANT`

## Main files

- `core.py`
  Contains the algorithms, config loading, output writing, and CLI plumbing.

- `__main__.py`
  Exposes `python -m standalone.tepid_qsceom`.

- `configs/`
  Example JSON input files.

- `requirements.txt`
  External dependencies only.

## Design choices

- One core file instead of many modules.
  This keeps the package easy to copy, inspect, and audit.

- Analytic qscEOM only.
  The projected matrix is built from exact statevectors, which keeps the
  implementation smaller and easier to validate.

- Default computational basis is `HF + singles + doubles`.
  That matches the basis choice used elsewhere in this repo for qscEOM-style
  workflows.
