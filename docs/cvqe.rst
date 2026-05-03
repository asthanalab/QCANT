CVQE
====

QCANT provides cyclic VQE via :func:`QCANT.cvqe`.

What It Does
------------

CVQE grows a selected determinant reference space and optimizes a compact
ansatz between determinant additions. The implementation supports LUCJ and
UCCSD-style ansatz updates, exact or sampled determinant selection, optional
checkpoint/resume payloads, and opt-in CuPy dense linear algebra.

Basic Usage
-----------

.. code-block:: python

   import numpy as np
   import QCANT

   symbols = ["H", "H"]
   geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]])

   params, determinants, energies, details = QCANT.cvqe(
       symbols=symbols,
       geometry=geometry,
       adapt_it=1,
       basis="sto-3g",
       charge=0,
       spin=0,
       active_electrons=2,
       active_orbitals=2,
       shots=0,
       optimizer_maxiter=10,
       array_backend="numpy",
       print_progress=False,
       return_details=True,
   )

   print("Selected determinants:", determinants)
   print("Final energy:", energies[-1])

Key Options
-----------

- ``ansatz="lucj"`` or ``"uccsd"`` selects the compact entangler.
- ``selection_method="probability"`` or ``"pt2"`` controls determinant
  admission.
- ``shots=0`` uses exact determinant probabilities; positive shots use sampled
  frequencies.
- ``checkpoint_path`` and ``resume_state`` support restartable runs.
- ``array_backend="cupy"`` requests GPU dense linear algebra after CPU
  chemistry setup.

GPU Notes
---------

The CuPy path accelerates dense state/reference-space operations. PySCF and
CCSD setup remain CPU-bound, and small problems may not show end-to-end
speedup.
