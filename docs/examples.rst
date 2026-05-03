Examples Gallery
================

The ``examples/`` directory contains runnable calculation input files. Each
script is designed to be launched directly from the repository root:

.. code-block:: bash

   python examples/run_adapt_vqe_h2.py

Most examples use H2 or H4 in STO-3G so they are suitable for smoke tests,
documentation checks, and onboarding. Heavier scripts expose command-line
controls for atom count, optimizer iterations, and backend selection.

Core Examples
-------------

.. list-table::
   :header-rows: 1

   * - Script
     - Demonstrates
   * - ``run_adapt_vqe_h2.py``
     - ADAPT-VQE ansatz growth and energy history.
   * - ``run_qsceom_h2.py``
     - qscEOM projected spectrum from the Hartree-Fock reference.
   * - ``run_adapt_vqe_qsceom_h2.py``
     - ADAPT-VQE ansatz handoff into qscEOM.
   * - ``run_gcim_h2.py``
     - GCIM projected basis construction.
   * - ``run_cvqe_h2.py``
     - CVQE determinant growth with checkpoint-style output.
   * - ``run_tepid_adapt_h2.py``
     - Finite-temperature TEPID-ADAPT and thermal weights.
   * - ``run_adapt_krylov_h2.py``
     - ADAPT-VQE with order-1/order-2 Krylov post-processing.
   * - ``run_qrte_h2.py``
     - QRTE and QRTE-PMTE real-time basis growth.
   * - ``run_qkud_h2.py``
     - QKUD unitary-decomposition basis growth.
   * - ``run_exact_krylov_h2.py``
     - Exact and Lanczos Krylov projections.
   * - ``run_krylov_family_h2.py``
     - QRTE, QRTE-PMTE, QKUD, and exact Krylov side by side.
   * - ``run_qulacs_h2.py``
     - Optional Qulacs CPU-accelerated routines with graceful dependency skip.
   * - ``run_gpu_dense_hchain.py``
     - Optional CuPy dense GPU path with graceful GPU skip.
   * - ``run_standalone_tepid_qsceom_h2.py``
     - Self-contained standalone TEPID/qscEOM workflow.

Outputs
-------

By default, examples write outputs under ``examples/outputs/<script-name>/``.
Those outputs are intentionally ignored by git. Pass ``--output-dir`` when you
want a different location.

Example:

.. code-block:: bash

   python examples/run_cvqe_h2.py --output-dir /tmp/qcant-cvqe-h2

GPU Notes
---------

``run_gpu_dense_hchain.py`` requests ``array_backend="cupy"``. If CuPy or a
CUDA-visible GPU is unavailable, the script prints a clear skip message and
exits successfully. Use ``--require-gpu`` in automation when missing GPU support
should be treated as a failure.
