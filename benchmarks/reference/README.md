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
| `h8_sector_projection_talon33_20260502.csv` | H8 fixed-electron sector CPU measurements and projected GPU/hybrid speedups. |
| `h8_sector_projection_talon33_20260502_speedup.png` | Speedup plot for the H8 sector projection study. |

## Interpretation

The H-chain dense study separates the dense linear-algebra hot loop from CPU
chemistry setup. H6 shows GPU speedups in the hot loop; H8 and larger are dense
memory-wall cases on a V100. The H8 sector projection CSV documents the
number-preserving sector size and projected speedups when the GPU hot-loop
timing is applied to the measured H8 CPU sector workload.
