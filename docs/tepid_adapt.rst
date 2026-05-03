TEPID-ADAPT
===========

QCANT provides an ancilla-free TEPID-ADAPT routine exposed as
:func:`QCANT.tepid_adapt`.

What it does
------------
The implementation in QCANT follows the truncated finite-temperature workflow
from the TEPID paper, adapted to the existing QCANT chemistry stack:

- build an active-space molecular Hamiltonian,
- construct a truncated computational basis from the Hartree-Fock reference
  plus single and double excitations by default,
- adaptively grow an excitation ansatz that lowers the truncated free energy,
- analytically update Boltzmann weights from the transformed-basis energies at
  the user-selected temperature, and
- return an ansatz tuple that can be replayed directly by :func:`QCANT.qscEOM`.

Dependencies
------------
This function requires the same scientific Python stack as the chemistry-heavy
QCANT routines:

.. code-block:: bash

   pip install numpy scipy
   pip install pennylane
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

   params, excitations, free_energies, details = QCANT.tepid_adapt(
       symbols=symbols,
       geometry=geometry,
       adapt_it=5,
       beta=2.0,
       basis="sto-3g",
       charge=0,
       spin=0,
       active_electrons=2,
       active_orbitals=2,
       return_details=True,
   )

   print("Final free energy:", free_energies[-1])
   print("Final transformed-basis energies:", details["sorted_basis_energies"])

Temperature control
-------------------
Provide exactly one of:

- ``beta`` for inverse temperature, or
- ``temperature`` for direct temperature.

QCANT assumes ``k_B = 1``, so the temperature uses the same units as the
Hamiltonian energies.

Truncated basis
---------------
By default, ``tepid_adapt`` uses the same chemistry basis pattern as qscEOM:

- Hartree-Fock reference (unless ``include_identity=False``), plus
- all single and double excitation occupations.

You can override this with ``basis_occupations=...`` if you want a custom
truncated basis while keeping the same adaptive free-energy optimization loop.

qscEOM handoff
--------------
The return value is intentionally compatible with qscEOM's ``ansatz=...`` path:

.. code-block:: python

   values = QCANT.qscEOM(
       symbols=symbols,
       geometry=geometry,
       active_electrons=2,
       active_orbitals=2,
       charge=0,
       ansatz=(params, excitations, free_energies),
       basis="sto-3g",
       method="pyscf",
       shots=0,
   )

Boltzmann reweighting
---------------------
If you already have a truncated spectrum and just want new thermal weights for
another temperature, use :func:`QCANT.tepid_boltzmann_weights`.

Backend Control
---------------

Use ``array_backend="numpy"`` for explicit CPU dense arrays or
``array_backend="cupy"`` for opt-in GPU dense arrays after CPU chemistry setup.
``array_backend="auto"`` preserves the default CPU behavior.

Notes
-----
- The current QCANT implementation is analytic/statevector-only and requires
  ``shots=0``.
- The basis states remain orthonormal because the ansatz is unitary, so the
  transformed-basis energies and thermal weights are easy to inspect in the
  returned ``details`` payload.
