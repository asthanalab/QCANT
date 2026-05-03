GPU Acceleration
================

QCANT supports opt-in single-GPU acceleration in two complementary ways.

- PennyLane device acceleration for QNode-driven algorithms
- CuPy dense linear algebra acceleration for exact-state matrix workflows

Installation
------------

For a local Python environment:

.. code-block:: bash

    pip install -e ".[gpu]"

For Talon GPU nodes, QCANT also ships a CUDA 12 oriented environment file:

.. code-block:: bash

    conda env create -f devtools/conda-envs/qcant-gpu.yaml
    conda activate qcant-gpu
    pip install -e . --no-deps

QCANT's GPU extra installs:

- ``pennylane-lightning-gpu``
- ``cupy-cuda12x``

If CuPy reports missing CUDA component libraries in a fresh pip-only
environment, install the CUDA component wheel extra supported by CuPy:

.. code-block:: bash

    python -m pip install "cupy-cuda12x[ctk]<14"

The official installation references are:

- https://pennylane.ai/devices/lightning-gpu
- https://docs.pennylane.ai/projects/lightning/en/stable/lightning_gpu/installation.html
- https://docs.cupy.dev/en/latest/install.html

Choosing A GPU Path
-------------------

Use ``device_name="lightning.gpu"`` for PennyLane-device algorithms:

- :func:`QCANT.adapt_vqe`
- :func:`QCANT.qscEOM`
- :func:`QCANT.gcim`
- :func:`QCANT.qrte`
- :func:`QCANT.qrte_pmte`

Use ``array_backend="cupy"`` for dense linear algebra algorithms:

- :func:`QCANT.cvqe`
- :func:`QCANT.tepid_adapt`
- :func:`QCANT.qkud`
- :func:`QCANT.exact_krylov`
- ``standalone.tepid_qsceom.tepid_adapt``
- ``standalone.tepid_qsceom.qsceom``
- ``standalone.tepid_qsceom.tepid_qsceom``

If you do not pass these options, QCANT keeps the existing CPU behavior.

Current CPU-Only Limits
-----------------------

The following paths intentionally remain CPU-only in v1:

- ``QCANT.qscEOM(projector_backend="sparse_number_preserving")``
- ``QCANT.qkud(..., use_sparse=True)``
- ``QCANT.exact_krylov(..., use_sparse=True)``
- all ``*_qulacs`` entry points

For ``adapt_vqe`` and dense ``qscEOM``, QCANT also forces single-worker GPU execution by downgrading process pools to thread mode and clamping ``max_workers`` to ``1`` on GPU-backed devices.

Talon Slurm Wrapper
-------------------

Use the repo-local wrapper for one-GPU Slurm jobs:

.. code-block:: bash

    PARTITION=talon-gpu32 NODE=talon32 GPUS=1 \
      scripts/run_on_talon_gpu.sh \
      'python scripts/benchmark_gpu_speedups.py --profile h6 --outdir benchmarks/gpubenchmark --gpu-device lightning.gpu'

Supported target nodes for this pass are:

- ``talon32``
- ``talon33``
- ``talon35``

``talon04`` is CPU-only and should not be used for QCANT GPU runs.

Benchmarking
------------

QCANT includes a GPU benchmark driver that writes both runtime and speedup outputs:

.. code-block:: bash

    python scripts/benchmark_gpu_speedups.py --profile h6 --repeats 2 --outdir benchmarks/gpubenchmark
    python scripts/benchmark_gpu_speedups.py --profile small --repeats 2 --outdir benchmarks/gpu_speedups_small

Outputs:

- ``h6_speedups.csv``
- ``h6_speedup_summary.png``
- ``h6_runtime_summary.png``
- ``plots/<algorithm>_speedup.png``
- ``README.md``

The CSV includes:

- algorithm name
- backend label
- accelerator kind
- problem profile
- active space and qubit count
- median runtime
- standard deviation
- speedup versus the CPU baseline
- status and notes for skipped or timed-out rows

``qulacs_cpu`` rows are CPU accelerator measurements and should not be
interpreted as GPU speedups.

For larger H-chain capability studies, use:

.. code-block:: bash

    python scripts/benchmark_hchain_gpu_capability.py --atoms 4 6 8 10
    python scripts/benchmark_hchain_sector_gpu_capability.py --atoms 6 8 10
    python scripts/benchmark_hchain_hybrid_capability.py --atoms 6

The first script benchmarks dense full-space Hamiltonian matmul after CPU
chemistry setup and records the full dense memory wall. The second script
benchmarks the fixed-electron number-preserving sparse sector, which is the
more realistic path for H8 and larger active spaces.
The third script splits state batches between CPU and GPU so you can measure
whether hybrid execution improves on GPU-only execution for a given batch size.

Curated reference outputs live under ``benchmarks/reference/``. Timestamped
benchmark directories are generated artifacts and are ignored by git.
