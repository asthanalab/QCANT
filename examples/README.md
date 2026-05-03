# QCANT Runnable Examples

These scripts are calculation input files for the public QCANT workflows. Run
them from the repository root:

```bash
python examples/run_adapt_vqe_h2.py
```

Most examples use small H2/H4 STO-3G systems so they can double as smoke tests.
Generated outputs are written under `examples/outputs/` and ignored by git.

## Core Workflow Examples

| Script | Demonstrates |
| --- | --- |
| `run_adapt_vqe_h2.py` | ADAPT-VQE ansatz growth. |
| `run_qsceom_h2.py` | qscEOM projected spectrum from a reference ansatz. |
| `run_adapt_vqe_qsceom_h2.py` | ADAPT-VQE to qscEOM handoff. |
| `run_gcim_h2.py` | GCIM projected basis workflow. |
| `run_cvqe_h2.py` | CVQE determinant growth. |
| `run_tepid_adapt_h2.py` | Finite-temperature TEPID-ADAPT. |
| `run_adapt_krylov_h2.py` | ADAPT-Krylov post-processing. |
| `run_krylov_family_h2.py` | QRTE, QRTE-PMTE, QKUD, and exact Krylov side by side. |
| `run_qulacs_h2.py` | Optional Qulacs CPU acceleration. |
| `run_gpu_dense_hchain.py` | Optional CuPy dense GPU execution for H chains. |
| `run_standalone_tepid_qsceom_h2.py` | Standalone TEPID/qscEOM CLI-style workflow. |

## Useful Options

Most scripts accept `--output-dir`. Algorithm-specific scripts also expose
small knobs such as `--bond-length`, `--adapt-it`, `--optimizer-maxiter`,
`--atoms`, and `--array-backend`.

GPU example:

```bash
python examples/run_gpu_dense_hchain.py --atoms 4 --array-backend cupy
```

Qulacs example:

```bash
python examples/run_qulacs_h2.py
```

If optional dependencies are not installed, optional examples print a skip
message instead of failing noisily.
