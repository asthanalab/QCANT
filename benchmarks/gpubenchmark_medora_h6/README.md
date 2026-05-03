# H6 Medora GPU Benchmark

This folder contains practical H6/STO-3G CPU-vs-GPU smoke benchmark artifacts
measured on a Medora H200/GH200 GPU node. Speedups are normalized to the
matching one-thread CPU baseline for each code path.

## Problem Profile

- Profile: `h6`
- Atoms: `HHHHHH`
- Basis: `sto-3g`
- H-chain spacing: `0.9` Angstrom
- Active space: `6` electrons in `6` orbitals
- Qubits: `12`

## Run Metadata

- Host: `medora01.cm.cluster`
- GPU: `NVIDIA GH200 480GB`
- Command: `scripts/benchmark_gpu_speedups.py --profile h6 --algorithms qkud exact_krylov cvqe standalone_qsceom --backends cpu gpu --repeats 2 --warmup 0 --timeout 900 --cpu-threads 1 --outdir benchmarks/gpubenchmark_medora_h6 --require-gpu`
- CPU baseline threads: `1`
- CPU PennyLane device: `default.qubit`
- GPU PennyLane device: `lightning.gpu`
- CUDA_VISIBLE_DEVICES: `0`
- Qulacs: not included in this Medora run because Qulacs was not available in
  the ARM-native H200 Python environment.

## Outputs

- `h6_speedups.csv`: runtime and speedup rows for CPU and GPU backends.
- `h6_speedup_summary.png`: combined accelerator speedup plot.
- `plots/*_speedup.png`: one speedup plot per benchmarked code path.

## Interpretation

This H6 end-to-end smoke benchmark does not show GPU speedup. The GPU rows run
successfully on Medora H200, but the measured workflows are still dominated by
CPU-side chemistry setup, dense Hamiltonian construction, transfer overhead, and
device/context overhead at this size.

| Code path | CPU median (s) | GPU median (s) | Speedup |
| --- | ---: | ---: | ---: |
| `qkud` | 41.359 | 46.479 | 0.89x |
| `exact_krylov` | 41.258 | 47.956 | 0.86x |
| `cvqe` | 42.952 | 43.024 | 1.00x |
| `standalone_qsceom` | 39.686 | 41.429 | 0.96x |

The companion Medora H-chain capability artifacts in `benchmarks/reference/`
isolate the dense batched Hamiltonian multiply after CPU chemistry setup. That
kernel-level study shows the expected H200 benefit for H6 once enough work is
batched, reaching about 233x at batch size 1024.

Rows labeled `skipped`, `timeout`, or `error` are retained in the CSV so the benchmark is reproducible.

## Skipped Or Failed Rows

- No skipped or failed rows.
