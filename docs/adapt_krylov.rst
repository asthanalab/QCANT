ADAPT-Krylov
============

QCANT provides ADAPT-Krylov post-processing via :func:`QCANT.adaptKrylov` and
the alias :func:`QCANT.adapt_krylov`.

What It Does
------------

ADAPT-Krylov runs the ADAPT-VQE selection/optimization loop and, after each
iteration, projects the optimized state into order-1 and order-2 Krylov
subspaces. This reports whether low-order Krylov post-processing improves the
current variational energy.

Basic Usage
-----------

.. code-block:: python

   import numpy as np
   import QCANT

   symbols = ["H", "H"]
   geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]])

   params, excitations, adapt_energies, details = QCANT.adaptKrylov(
       symbols=symbols,
       geometry=geometry,
       adapt_it=2,
       basis="sto-3g",
       charge=0,
       spin=0,
       active_electrons=2,
       active_orbitals=2,
       device_name="default.qubit",
       optimizer_maxiter=25,
       backend="pennylane",
   )

   print("ADAPT:", adapt_energies)
   print("Order-1 Krylov:", details["krylov_order1_energies"])
   print("Order-2 Krylov:", details["krylov_order2_energies"])

Backends
--------

``backend="auto"`` prefers the Qulacs CPU backend when the request is
compatible with analytic Qulacs execution. Use ``backend="pennylane"`` when you
want the reference PennyLane implementation or when a specific PennyLane device
is requested.
