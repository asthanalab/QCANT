Getting Started
===============

QCANT runs active-space quantum chemistry workflows through a Python API and a
small set of runnable example scripts. The default path is CPU-only and uses
NumPy/SciPy, PennyLane, and PySCF.

Install
-------

QCANT supports Python 3.11 and 3.12.

For development from source, use the conda environment:

.. code-block:: bash

   conda env create -f devtools/conda-envs/qcant.yaml
   conda activate qcant
   python -m pip install -e . --no-deps

Optional Qulacs CPU acceleration:

.. code-block:: bash

   python -m pip install -e ".[qulacs]"

Optional single-GPU acceleration:

.. code-block:: bash

   python -m pip install -e ".[gpu]"

On Talon GPU nodes, use the CUDA 12 environment file:

.. code-block:: bash

   conda env create -f devtools/conda-envs/qcant-gpu.yaml
   conda activate qcant-gpu
   python -m pip install -e . --no-deps

Run Your First Calculation
--------------------------

.. code-block:: python

   import numpy as np
   import QCANT

   symbols = ["H", "H"]
   geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]])

   params, excitations, energies = QCANT.adapt_vqe(
       symbols=symbols,
       geometry=geometry,
       adapt_it=1,
       basis="sto-3g",
       charge=0,
       spin=0,
       active_electrons=2,
       active_orbitals=2,
       device_name="default.qubit",
       optimizer_maxiter=25,
   )

   print("Final ADAPT energy:", energies[-1])

The same ansatz can be passed directly into qscEOM:

.. code-block:: python

   roots = QCANT.qscEOM(
       symbols=symbols,
       geometry=geometry,
       active_electrons=2,
       active_orbitals=2,
       charge=0,
       ansatz=(params, excitations, energies),
       basis="sto-3g",
       shots=0,
   )

   print("qscEOM roots:", roots[0])

Use The Example Inputs
----------------------

Every script in ``examples/`` is runnable as ``python examples/<name>.py``.

.. code-block:: bash

   python examples/run_adapt_vqe_h2.py
   python examples/run_qsceom_h2.py
   python examples/run_qrte_h2.py
   python examples/run_gpu_dense_hchain.py --atoms 4
   python examples/run_standalone_tepid_qsceom_h2.py

Generated JSON/CSV outputs are written under ``examples/outputs/`` and are not
tracked by git.

Choose A Backend
----------------

Default CPU behavior:

.. code-block:: python

   QCANT.qkud(..., array_backend="numpy")

PennyLane GPU device path:

.. code-block:: python

   QCANT.adapt_vqe(..., device_name="lightning.gpu")

CuPy dense linear algebra path:

.. code-block:: python

   QCANT.exact_krylov(..., array_backend="cupy")

For details on supported GPU paths and CPU-only limits, see
:doc:`gpu_acceleration`.
