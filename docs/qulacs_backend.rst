Qulacs Backend
==============

QCANT includes an optional Qulacs-backed exact-state backend for the
simulator-heavy algorithms:

- :func:`QCANT.qkud_qulacs`
- :func:`QCANT.qrte_qulacs`
- :func:`QCANT.qrte_pmte_qulacs`
- :func:`QCANT.adapt_vqe_qulacs`
- :func:`QCANT.cvqe_qulacs`

These Qulacs entry points remain CPU-only in the current GPU acceleration
pass. They reduce simulator overhead on CPU nodes; they are not the GPU path.

Installation
------------

Install QCANT with the optional Qulacs extra:

.. code-block:: bash

    pip install -e ".[qulacs]"

or

.. code-block:: bash

    pip install "QCANT[qulacs]"

What Is Accelerated
-------------------

The Qulacs path is not a full chemistry-stack replacement. QCANT still uses
PySCF and PennyLane for molecule construction, active-space setup, and
Hamiltonian generation. Qulacs is used as the simulator backend in the hot
loops:

- state evolution
- expectation values
- transition amplitudes
- native parametric-circuit updates
- native gradient backprop where available

Efficiency Changes
------------------

The current implementation in :mod:`QCANT.qulacs_accel` focuses on reducing
avoidable simulator overhead.

Compiled parametric circuits
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PennyLane chemistry gates are decomposed once into primitive rotations and
Clifford gates, then compiled into reusable Qulacs
``ParametricQuantumCircuit`` objects. The backend updates only scalar
parameters during optimization rather than rebuilding dense gate matrices on
every cost-function call.

This is the main structural speedup for:

- ADAPT-VQE excitation ansatz growth
- CVQE LUCJ and UCCSD ansatz application

Native ADAPT-VQE gradients
^^^^^^^^^^^^^^^^^^^^^^^^^^

``adapt_vqe_qulacs`` uses Qulacs backprop on the compiled circuit instead of
SciPy numerical gradients over repeatedly rebuilt state-preparation logic.

That reduces the inner optimization cost substantially compared with the older
matrix-based simulator path.

Reduced CVQE inner optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``cvqe_qulacs`` no longer treats determinant coefficients and ansatz parameters
as one large nonlinear optimization problem.

Instead, after each determinant is added:

1. the selected-determinant reference space is projected exactly
2. the optimal reference amplitudes in that selected space are solved from the
   projected Hamiltonian
3. only the ansatz parameters are optimized in the reduced outer objective

This keeps the Qulacs CVQE loop closer to a selected-CI style update and
avoids pushing all determinant coefficients through numerical BFGS.

Threaded projection work
^^^^^^^^^^^^^^^^^^^^^^^^

The projected-Hamiltonian assembly used by QKUD and QRTE can be parallelized
over rows with ``max_workers``. ADAPT-VQE also parallelizes the commutator
screening step.

Practical Guidance
------------------

For larger exact-state runs, prefer:

- ``evolution_mode="trotter"`` for QKUD/QRTE
- exact-state Qulacs backends over the dense PennyLane matrix path

Do not expect ``evolution_mode="sparse"`` to scale well to the 16-20 qubit
range, because it still materializes the Hamiltonian matrix.

Current Limits
--------------

The backend is materially faster in the simulator hot path, but the overall
cost can still be dominated by the algorithm itself.

In particular:

- H6 CVQE with long outer loops is still expensive
- poor determinant-selection signals can stall CVQE even when the simulator
  backend is faster
- PySCF SCF/CCSD setup time is unchanged

So the Qulacs backend removes a large class of simulator overhead, but it does
not by itself fix algorithmic plateaus.
