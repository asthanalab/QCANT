Exact Krylov
============

QCANT provides a Krylov subspace routine exposed as :func:`QCANT.exact_krylov`.

What it does
------------
The exact Krylov routine:

- builds an electronic Hamiltonian using PennyLane quantum chemistry tooling,
- prepares a reference state (Hartree-Fock by default or a provided statevector),
- generates a Krylov basis from successive applications of the Hamiltonian, and
- diagonalizes the Hamiltonian projected into that basis.

The basis construction can use a power basis (``krylov_method="exact"``) or the
Lanczos recurrence (``krylov_method="lanczos"``), which orthonormalizes the
basis during construction.

Basic usage
-----------

.. code-block:: python

   import numpy as np
   import QCANT

   symbols = ["H", "H"]
   geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]])

   energies, basis_states, min_history = QCANT.exact_krylov(
      symbols=symbols,
      geometry=geometry,
      n_steps=3,
      active_electrons=2,
      active_orbitals=2,
      basis="sto-3g",
      charge=0,
      spin=0,
      krylov_method="exact",
      return_min_energy_history=True,
   )

   print(energies)
   print(basis_states.shape)
   print(min_history.shape)

Options
-------

- ``krylov_method``: ``"exact"`` (power basis) or ``"lanczos"`` (orthonormal recurrence).
- ``use_sparse``: use a sparse Hamiltonian representation for state updates.
- ``return_min_energy_history``: return the minimum energy after each basis step.

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

- ``krylov_method="exact"`` does not orthogonalize during basis construction;
  orthogonalization is performed via the overlap matrix during projection.
- ``krylov_method="lanczos"`` builds an orthonormal basis by construction.
