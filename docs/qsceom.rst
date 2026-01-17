qscEOM
======

QCANT provides a qscEOM routine exposed as :func:`QCANT.qscEOM`.

What it does
------------
At a high level, the routine:

- builds a molecular Hamiltonian using PennyLane's quantum chemistry tooling,
- evaluates diagonal and off-diagonal matrix elements for a set of configurations,
- forms an effective matrix, and
- returns its eigenvalues (sorted).

Dependencies
------------
This function requires scientific Python dependencies that are installed with QCANT.

If you are developing from source and see :class:`ImportError`, install the required
dependencies (see below).

Minimum expected dependencies:

.. code-block:: bash

   pip install numpy
   pip install pennylane

PennyLane's quantum chemistry features often require additional packages (e.g. PySCF).
If you see runtime errors during Hamiltonian construction, install:

.. code-block:: bash

   pip install pyscf

Basic usage
-----------
.. code-block:: python

   import numpy as np
   import QCANT

   symbols = ["H", "H"]
   geometry = np.array(
       [
           [0.0, 0.0, 0.0],
           [0.0, 0.0, 0.74],
       ]
   )

   active_electrons = 2
   active_orbitals = 2
   charge = 0

   # These are typically obtained from a separate ansatz/VQE routine.
   params = np.array([0.0])
   ash_excitation = [[0, 1]]

   values = QCANT.qscEOM(
       symbols=symbols,
       geometry=geometry,
       active_electrons=active_electrons,
       active_orbitals=active_orbitals,
       charge=charge,
       params=params,
       ash_excitation=ash_excitation,
       basis="sto-3g",
       method="pyscf",
       shots=0,
   )

   print(values)

Inputs
------
A few practical notes:

- ``geometry`` is expected to be array-like with shape ``(n_atoms, 3)``.
- ``params`` and ``ash_excitation`` must be consistent with each other:
  the number of parameters must match the number of excitations.

Notes
-----
- This implementation is currently targeted at research experimentation.
- For reproducible results, keep ``shots=0`` (analytic mode) where possible.
