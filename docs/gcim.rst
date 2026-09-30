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

- ``theta``: fixed angle in ``exp(theta * A / 2)`` (default ``pi/4``).
- ``pool_sample_size``/``pool_seed``: random operator-pool subsampling.
- ``pool_type``: operator pool construction strategy:

  - ``"sd"``: reference-based singles+doubles pool,
  - ``"singlet_sd"``: complete spin-adapted singles+doubles pool (closed-shell active spaces),
  - ``"gsd"``: generalized spin-conserving singles+doubles pool.

- ``regularization``: overlap-matrix diagonal regularization before solving
  ``M c = S c E``.
- ``return_details``: return debug artifacts (state basis, projected matrices).

Complete SingletSD pool
-----------------------

For ``o`` occupied and ``v`` virtual spatial orbitals, ``singlet_sd`` contains
``o*v`` singles and ``C(o+1,2)*C(v+1,2) + C(o,2)*C(v,2)`` doubles. Both double
channels commute with total spin; the triplet-pair contraction is a spin scalar,
not a triplet-state ansatz. CAS(6e,6o) therefore has 9 singles + 45 doubles = 54
operators. CAS(4e,4o), CAS(8e,6o), and CAS(10e,8o) have 14, 44, and 135.

Every generator is normalized to unit Euclidean norm of its normal-ordered
fermionic coefficients, following the original SingletSD GCIM construction.
This corrects the former paired-only implementation (18 operators for
CAS(6e,6o)), which also left its coefficients unnormalized. Consequently old
``singlet_sd`` trajectories are not preserved at the same numerical ``theta``.
The angle convention, overlap regularization, unique-selection policy, and
other pool types are unchanged. This pool correction alone does not establish
reproduction of any historical convergence figure.

Returned SingletSD excitation labels use spatial indices: ``[i,a]`` for singles
and ``[i,j,a,b,pair_spin]`` for doubles, with ``pair_spin`` 0 or 1. This replaces
the paired-only implementation's spin-orbital labels. ``return_details=True``
also reports ``pool_type``, ``pool_size``, and all ``pool_excitations``.

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
