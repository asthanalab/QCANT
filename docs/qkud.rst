QKUD (Quantum Krylov using Unitary Decomposition)
=================================================

QCANT provides a QKUD routine exposed as :func:`QCANT.qkud`.

What it does
------------
The QKUD routine constructs a Krylov basis using the unitary-decomposition
recurrence:

.. math::

   |psi_n> = \frac{X + X^\dagger}{2 \epsilon} |psi_{n-1}>, \quad
   X = i e^{-i \epsilon H}

It then projects the Hamiltonian into that basis and diagonalizes it.

Basic usage
-----------

.. code-block:: python

   import numpy as np
   import QCANT

   symbols = ["H", "H"]
   geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]])

   energies, basis_states, min_history = QCANT.qkud(
      symbols=symbols,
      geometry=geometry,
      n_steps=3,
      epsilon=0.1,
      active_electrons=2,
      active_orbitals=2,
      basis="sto-3g",
      charge=0,
      spin=0,
      return_min_energy_history=True,
   )

   print(energies)
   print(basis_states.shape)
   print(min_history.shape)

Options
-------

- ``use_sparse``: use a sparse Hamiltonian representation for state updates.
- ``basis_threshold``: drop amplitudes below this threshold after each basis update and
  re-normalize the state (use 0.0 to disable).

Outputs
-------

The function returns ``(energies, basis_states)``:

- ``energies``: eigenvalues obtained by diagonalizing the Hamiltonian in the generated basis
- ``basis_states``: statevectors after each step, shape ``(n_steps+1, 2**n_qubits)``

If ``return_min_energy_history=True``, the function returns
``(energies, basis_states, min_energy_history)`` where ``min_energy_history`` has
shape ``(n_steps,)``.

Notes
-----

- This implementation uses matrix-based time evolution (no circuit execution).
- ``epsilon`` must be positive; smaller values can lead to nearly dependent
  basis vectors.
