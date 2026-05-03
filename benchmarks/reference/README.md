# QCANT Curated Benchmark References

This directory contains small benchmark artifacts retained for release
documentation. Timestamped benchmark output directories are generated artifacts
and are ignored by git.

## Included Artifacts

| File | Meaning |
| --- | --- |
| `hchain_gpu_capability_talon33_20260502.csv` | Dense H-chain hot-loop CPU vs GPU timings measured on `talon33`. |
| `hchain_gpu_capability_talon33_20260502_speedup.png` | Speedup plot for the dense H-chain hot-loop study. |
| `hchain_dense_memory_wall_talon33_20260502.png` | Dense full-space memory wall for H chains. |
| `hchain_gpu_capability_medora_h200_20260502.csv` | Dense H-chain hot-loop CPU vs GPU timings measured on Medora H200/GH200. |
| `hchain_gpu_capability_medora_h200_20260502_speedup.png` | Speedup plot for the Medora H200/GH200 dense H-chain hot-loop study. |
| `hchain_dense_memory_wall_medora_h200_20260502.png` | Dense full-space memory wall for the Medora H200/GH200 H-chain study. |
| `h8_sector_projection_talon33_20260502.csv` | H8 fixed-electron sector CPU measurements and projected GPU/hybrid speedups. |
| `h8_sector_projection_talon33_20260502_speedup.png` | Speedup plot for the H8 sector projection study. |

## Interpretation

The H-chain dense study separates the dense linear-algebra hot loop from CPU
chemistry setup. H6 shows GPU speedups in the hot loop; H8 and larger are dense
memory-wall cases on a V100. The H8 sector projection CSV documents the
number-preserving sector size and projected speedups when the GPU hot-loop
timing is applied to the measured H8 CPU sector workload.

The Medora H200/GH200 H-chain study confirms that the GPU path is functional on
modern Medora GPU nodes. End-to-end H6 public workflows remain overhead-bound,
but the isolated dense H6 Hamiltonian multiply reaches 12.74x at batch size 1,
117.46x at batch size 32, and 232.96x at batch size 1024 versus a one-thread
CPU baseline.
