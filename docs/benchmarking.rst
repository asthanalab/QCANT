Benchmarking And Speedups
=========================

QCANT benchmark scripts write both raw runtimes and speedup ratios. Speedup is
reported relative to the matching CPU baseline for the same algorithm/profile.

Main GPU Benchmark
------------------

.. code-block:: bash

   python scripts/benchmark_gpu_speedups.py --profile h6 --repeats 2 --outdir benchmarks/gpubenchmark
   python scripts/benchmark_gpu_speedups.py --profile small --repeats 2 --outdir benchmarks/gpu_speedups_small
   python scripts/benchmark_gpu_speedups.py --profile large --repeats 2 --outdir benchmarks/gpu_speedups_large

Outputs:

- ``h6_speedups.csv``
- ``h6_speedup_summary.png``
- ``h6_runtime_summary.png``
- ``plots/<algorithm>_speedup.png``
- ``README.md``

The CSV includes algorithm, backend/device, accelerator kind, problem profile,
basis, active space, qubit count, median runtime, standard deviation, speedup
versus CPU, CPU thread count, ``CUDA_VISIBLE_DEVICES``, status, and notes.
Rows labeled ``qulacs_cpu`` are CPU accelerator measurements, not GPU
measurements.

H-Chain Capability Studies
--------------------------

Dense full-space study:

.. code-block:: bash

   python scripts/benchmark_hchain_gpu_capability.py --atoms 4 6 8

Fixed-electron sector study:

.. code-block:: bash

   python scripts/benchmark_hchain_sector_gpu_capability.py --atoms 6 8

Hybrid CPU/GPU batch study:

.. code-block:: bash

   python scripts/benchmark_hchain_hybrid_capability.py --atoms 6

Interpretation
--------------

Small end-to-end calculations are often dominated by CPU preprocessing,
PennyLane device construction, and host/device transfer. Those workloads may be
slower on GPU even when the GPU hot loop is faster.

For H8 in STO-3G, the full dense Hamiltonian is not a practical V100 target
because the full-space dense matrix is too large. The number-preserving sector
is the meaningful capability target for larger H chains.

The Talon study from May 2, 2026 measured useful GPU capability in the dense
and fixed-sector linear-algebra hot loops after Hamiltonian construction. The
curated reference artifacts under ``benchmarks/reference/`` record the measured
or projected speedup values used in the release documentation.

Parallel CPU Benchmark
----------------------

.. code-block:: bash

   python scripts/benchmark_parallel_adapt_qsceom.py --profile small --workers 1 2 4 8

This script measures CPU worker scaling for ADAPT commutator scoring and qscEOM
matrix construction. It is separate from the GPU benchmark because single-GPU
execution intentionally clamps GPU worker pools to one worker.
