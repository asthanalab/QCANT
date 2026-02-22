ADAPT-VQE
=========

QCANT provides an ADAPT-style VQE routine exposed as :func:`QCANT.adapt_vqe`.

What it does
------------
The implementation in QCANT is an experimental/script-style implementation of an ADAPT loop:

- builds an electronic Hamiltonian for a small example system,
- iteratively selects an operator from a pool based on commutator magnitude,
- optimizes the ansatz parameters each iteration, and
- returns the optimized parameters, chosen excitations, and energies.

Dependencies
------------
This function requires scientific Python dependencies that are installed with QCANT.

If you are developing from source and see :class:`ImportError`, install the required
dependencies (see below).

.. note::

   The entry point is named ``adapt_vqe`` (instead of just ``adapt``) to avoid
   shadowing the :mod:`QCANT.adapt` subpackage.

Minimum expected dependencies:

.. code-block:: bash

   pip install numpy scipy
   pip install pennylane
   pip install pyscf
   pip install basis_set_exchange

Depending on your PennyLane backend, you may also need:

.. code-block:: bash

   pip install pennylane-lightning

Basic usage
-----------
.. code-block:: python

   import numpy as np
   import QCANT

   symbols = ["H", "H", "H", "H"]
   geometry = np.array(
      [
         [0.0, 0.0, 0.0],
         [0.0, 0.0, 3.0],
         [0.0, 0.0, 6.0],
         [0.0, 0.0, 9.0],
      ]
   )

   params, excitations, energies = QCANT.adapt_vqe(
      symbols=symbols,
      geometry=geometry,
      adapt_it=5,
      basis="sto-6g",
      charge=0,
      spin=0,
      active_electrons=4,
      active_orbitals=4,
   )

   print("Final energy:", energies[-1])
   print("Number of selected excitations:", len(excitations))

Parallel commutator evaluation
------------------------------
The most expensive ADAPT step is usually evaluating commutators (operator
selection gradients) over the pool. QCANT now supports optional concurrent
evaluation for this stage:

.. code-block:: python

   params, excitations, energies = QCANT.adapt_vqe(
      symbols=symbols,
      geometry=geometry,
      adapt_it=5,
      basis="sto-6g",
      charge=0,
      spin=0,
      active_electrons=4,
      active_orbitals=4,
      parallel_gradients=True,
      parallel_backend="process",  # process|thread|auto
      max_workers=4,          # optional; defaults to os.cpu_count()
      gradient_chunk_size=8,  # optional task granularity control
   )

Notes:

- ADAPT iterations and optimizer steps remain serial by design.
- Only independent commutator evaluations are parallelized.
- Selection tie-breaking remains deterministic (matches serial order).
- No additional package is required for this parallel mode; it uses Python's
  standard-library ``concurrent.futures``.
- In restricted environments where process pools are unavailable, QCANT
  automatically falls back to thread-based execution.
- For tuning guidance, see :doc:`parallelization`.

Outputs
-------
The function returns ``(params, ash_excitation, energies)``:

- ``params``: optimized parameter vector (final)
- ``ash_excitation``: list of excitations chosen over iterations
- ``energies``: list of energies after each iteration

Notes
-----
- The molecular geometry is user-provided via ``symbols`` and ``geometry``.
- Runtime and convergence depend strongly on the chosen backend/device and optimization settings.
- Treat this as research code; verify results and units for your specific use case.
