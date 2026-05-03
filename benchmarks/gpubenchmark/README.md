# H6 GPU And Qulacs Benchmark

This folder contains practical H6/STO-3G smoke benchmark artifacts for QCANT.
Speedups are normalized to the matching one-thread CPU baseline for each code path.

## Problem Profile

- Profile: `h6`
- Atoms: `HHHHHH`
- Basis: `sto-3g`
- H-chain spacing: `0.9` Angstrom
- Active space: `6` electrons in `6` orbitals
- Qubits: `12`

## Run Metadata

- Host: `talon32.cm.cluster`
- Command: `scripts/benchmark_gpu_speedups.py --profile h6 --repeats 2 --warmup 0 --timeout 180 --cpu-threads 1 --outdir benchmarks/gpubenchmark --require-gpu`
- CPU baseline threads: `1`
- CPU PennyLane device: `default.qubit`
- GPU PennyLane device: `lightning.gpu`
- CUDA_VISIBLE_DEVICES: `0`

## Outputs

- `h6_speedups.csv`: runtime and speedup rows for CPU, GPU, and Qulacs backends.
- `h6_speedup_summary.png`: combined accelerator speedup plot.
- `plots/*_speedup.png`: one speedup plot per benchmarked code path.

## Interpretation

Best observed accelerator row: `qkud_qulacs` on `qulacs_cpu` at 17.14x versus CPU.
Rows labeled `qulacs_cpu` are CPU accelerator measurements, not GPU measurements.
Rows labeled `skipped`, `timeout`, or `error` are retained in the CSV so the benchmark is reproducible.
No public GPU entrypoint row completed in this H6 smoke pass; those rows are preserved as skipped rows in the CSV and plots.

## Measured Accelerator Speedups

| Code path | Accelerator | CPU median (s) | Accelerator median (s) | Speedup |
| --- | --- | ---: | ---: | ---: |
| `adapt_vqe_qulacs` | `qulacs_cpu` | 32.656 | 3.315 | 9.85x |
| `cvqe_qulacs` | `qulacs_cpu` | 48.337 | 37.740 | 1.28x |
| `qkud_qulacs` | `qulacs_cpu` | 47.792 | 2.789 | 17.14x |
| `adaptKrylov_qulacs` | `qulacs_cpu` | 108.974 | 21.185 | 5.14x |

`qrte_qulacs` and `qrte_pmte_qulacs` completed in about 2.8 seconds on Qulacs, but their one-thread CPU baselines hit the 180 second smoke timeout, so no finite speedup ratio is reported.

## Skipped Or Failed Rows

- `adapt_vqe` / `gpu`: skipped
- `qscEOM` / `gpu`: skipped
- `gcim` / `cpu`: timeout
- `gcim` / `gpu`: skipped
- `qrte` / `cpu`: timeout
- `qrte` / `gpu`: skipped
- `qrte_pmte` / `cpu`: timeout
- `qrte_pmte` / `gpu`: skipped
- `cvqe` / `gpu`: skipped
- `tepid_adapt` / `cpu`: timeout
- `tepid_adapt` / `gpu`: skipped
- `qkud` / `gpu`: skipped
- `exact_krylov` / `gpu`: skipped
- `adaptKrylov` / `gpu`: skipped
- `standalone_tepid_adapt` / `cpu`: timeout
- `standalone_tepid_adapt` / `gpu`: skipped
- `standalone_qsceom` / `gpu`: skipped
- `standalone_tepid_qsceom` / `cpu`: timeout
- `standalone_tepid_qsceom` / `gpu`: skipped
- `qrte_qulacs` / `cpu`: timeout
- `qrte_pmte_qulacs` / `cpu`: timeout
