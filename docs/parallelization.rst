Parallelization Guide
=====================

This guide explains how QCANT parallelizes two expensive algorithm stages:

- ADAPT-VQE operator-pool commutator/gradient scoring
- qscEOM matrix-element construction

What Is Parallelized
--------------------

ADAPT-VQE

- Parallelized: independent commutator evaluations across candidate operators.
- Serial: ADAPT iteration loop and parameter optimization.

qscEOM

- Parallelized: diagonal and off-diagonal matrix-element evaluations.
- Serial: final matrix assembly and eigenvalue solve.

How To Enable It
----------------

ADAPT-VQE parallel gradients:

.. code-block:: python

   params, excitations, energies = QCANT.adapt_vqe(
       symbols=symbols,
       geometry=geometry,
       adapt_it=3,
       basis="sto-3g",
       charge=0,
       spin=1,
       active_electrons=5,
       active_orbitals=5,
       device_name="default.qubit",
       parallel_gradients=True,
       parallel_backend="process",   # process | thread | auto
       max_workers=8,
       gradient_chunk_size=2,
   )

qscEOM parallel matrix construction:

.. code-block:: python

   values = QCANT.qscEOM(
       symbols=symbols,
       geometry=geometry,
       active_electrons=6,
       active_orbitals=6,
       charge=0,
       params=params,
       ash_excitation=ash_excitation,
       basis="sto-3g",
       method="pyscf",
       shots=0,
       symmetric=True,
       parallel_matrix=True,
       parallel_backend="process",   # process | thread | auto
       max_workers=8,
       matrix_chunk_size=20,
   )

Backend Selection
-----------------

- ``parallel_backend="process"``: preferred for CPU-bound QNode-heavy workloads.
- ``parallel_backend="thread"``: useful where process creation is restricted.
- ``parallel_backend="auto"``:
  uses process backend on POSIX and thread backend on Windows.

If process pools cannot be created (restricted environment), QCANT falls back
to thread backend automatically.

On GPU-backed PennyLane devices, QCANT keeps the run single-GPU safe by:

- downgrading ``parallel_backend="process"`` to thread mode
- clamping ``max_workers`` to ``1``

Tuning Parameters
-----------------

- ``max_workers``:
  worker count for the selected backend.
- ``gradient_chunk_size``:
  number of ADAPT candidates per submitted task.
- ``matrix_chunk_size``:
  number of qscEOM matrix entries per submitted task.

Practical defaults:

- Start with ``max_workers`` in ``[2, 4, 8]``.
- For ADAPT gradients, start ``gradient_chunk_size=2``.
- For larger qscEOM matrices, use smaller chunks (for example ``20``) to
  improve load balance.

Benchmarking
------------

QCANT includes a benchmark script:

.. code-block:: bash

   python scripts/benchmark_parallel_adapt_qsceom.py --profile small --repeats 1
   python scripts/benchmark_parallel_adapt_qsceom.py --profile large --repeats 1

Options:

- ``--profile small|large``: workload size.
- ``--workers 1 2 4 8``: worker counts to test.
- ``--repeats N`` and ``--warmup N``: timing controls.
- ``--outdir <path>``: output directory for CSV and plot.

Outputs:

- ``benchmark_parallel_adapt_qsceom.csv``
- ``benchmark_parallel_adapt_qsceom_speedup.png``

QCANT also includes a CPU-vs-GPU benchmark script for the opt-in GPU paths:

.. code-block:: bash

   python scripts/benchmark_gpu_speedups.py --profile h6 --repeats 2 --outdir benchmarks/gpubenchmark
   python scripts/benchmark_gpu_speedups.py --profile small --repeats 2 --outdir benchmarks/gpu_speedups_small

GPU benchmark outputs:

- ``h6_speedups.csv``
- ``h6_speedup_summary.png``
- ``h6_runtime_summary.png``
- ``plots/<algorithm>_speedup.png``

Rows labeled ``qulacs_cpu`` are CPU accelerator measurements, not GPU
measurements.

For H-chain capability sweeps that separate CPU chemistry setup from the GPU
linear-algebra hot loop:

.. code-block:: bash

   python scripts/benchmark_hchain_gpu_capability.py --atoms 4 6 8 10
   python scripts/benchmark_hchain_sector_gpu_capability.py --atoms 6 8 10
   python scripts/benchmark_hchain_hybrid_capability.py --atoms 6

Notes
-----

- Results depend on backend/device availability and CPU topology.
- Set BLAS/OpenMP thread limits to avoid oversubscription when benchmarking.
- Expect diminishing returns once synchronization and scheduling overhead
  approaches per-task compute cost.
