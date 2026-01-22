QRTE (Quantum Real-Time Evolution)
=================================

QCANT provides a real-time evolution routine exposed as :func:`QCANT.qrte`.

What it does
------------
The QRTE routine:

- builds an electronic Hamiltonian using PennyLane quantum chemistry tooling,
- prepares the Hartree–Fock (HF) state for the requested active space,
- applies time evolution in fixed increments of ``delta_t``, and
- returns the sequence of states, which can be treated as a (non-orthogonal) basis.

Dependencies
------------
This function requires scientific Python dependencies that are installed with QCANT.

If you are developing from source, the recommended setup is:

.. code-block:: bash

   conda env create -f devtools/conda-envs/qcant.yaml
   conda activate qcant
   pip install -e . --no-deps

Basic usage
-----------

.. code-block:: python

   import numpy as np
   import QCANT

   symbols = ["H", "H"]
   geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]])

   energies, basis_states, times = QCANT.qrte(
      symbols=symbols,
      geometry=geometry,
      delta_t=0.1,
      n_steps=3,
      active_electrons=2,
      active_orbitals=2,
      basis="sto-3g",
      charge=0,
      spin=0,
      device_name="default.qubit",
   )

   print(times)
   print(energies)
   print(basis_states.shape)

Options
-------

- ``use_sparse``: use a sparse Hamiltonian representation for projections.
- ``basis_threshold``: drop amplitudes below this threshold after each basis update and
  re-normalize the state (use 0.0 to disable).

Outputs
-------

The function returns ``(energies, basis_states, times)``:

- ``energies``: eigenvalues obtained by diagonalizing the Hamiltonian in the generated basis
- ``basis_states``: statevectors after each step, shape ``(n_steps+1, 2**n_qubits)``
- ``times``: times associated with each vector, shape ``(n_steps+1,)``

Notes
-----

- This implementation uses analytic statevector access (``qml.state()``).
- The returned basis is generally not orthonormal.
