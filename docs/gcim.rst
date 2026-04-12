GCIM (Generalized Configuration Interaction Method)
===================================================

QCANT provides GCIM via :func:`QCANT.gcim.gcim`.

What it does
------------
The GCIM routine:

- builds the active-space electronic Hamiltonian from PySCF/CASCI integrals,
- constructs an anti-Hermitian operator pool (``sd``, ``singlet_sd``, or ``gsd``),
- selects one operator per iteration using commutator magnitude,
- builds a non-orthogonal state basis from cumulative and individual states, and
- solves a projected generalized eigenvalue problem to estimate the ground-state
  energy at each iteration.

The public API is intentionally aligned with :func:`QCANT.adapt_vqe` so users
can switch workflows with minimal code changes.

Basic usage
-----------

.. code-block:: python

   import numpy as np
   from QCANT.gcim import gcim

   symbols = ["H", "H"]
   geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]])

   params, excitations, energies = gcim(
      symbols=symbols,
      geometry=geometry,
      adapt_it=50,  # kept for API compatibility; ignored in GCIM
      basis="sto-3g",
      charge=0,
      spin=0,
      active_electrons=2,
      active_orbitals=2,
      pool_type="sd",
      device_name="default.qubit",
   )

   print(len(excitations))
   print(energies[-1])

Key options
-----------

- ``theta``: fixed generator angle per selected operator (default ``pi/4``).
- ``pool_sample_size``/``pool_seed``: random operator-pool subsampling.
- ``pool_type``: operator pool construction strategy:

  - ``"sd"``: reference-based singles+doubles pool,
  - ``"singlet_sd"``: singlet-adapted paired singles+doubles pool (closed-shell active spaces),
  - ``"gsd"``: generalized spin-conserving singles+doubles pool.

- ``regularization``: overlap-matrix diagonal regularization before solving
  ``M c = S c E``.
- ``return_details``: return debug artifacts (state basis, projected matrices).

Iteration policy
----------------

GCIM iteration count is fixed to the full size of the selected operator pool:

- ``n_iterations = len(pool)``

This behavior is hardcoded. Therefore:

- ``adapt_it`` is accepted only for API compatibility with :func:`QCANT.adapt_vqe`, and
- ``allow_repeated_operators`` is accepted only for API compatibility.

GCIM always enforces unique operator selection and runs until the chosen pool is exhausted.

Returns
-------

The function returns ``(params, ash_excitation, energies)``:

- ``params``: selected parameter vector (all values equal to ``theta``),
- ``ash_excitation``: selected excitation list in ADAPT-style format,
- ``energies``: per-iteration minimum projected energy.

If ``return_details=True``, a fourth dictionary is returned.
