QRTE and QRTE-PMTE
==================

QCANT provides two real-time evolution routines:

- :func:`QCANT.qrte` for forward-only time-step basis growth, and
- :func:`QCANT.qrte_pmte` for symmetric ``+/-`` time-step basis growth.

What it does
------------
Both routines:

- builds an electronic Hamiltonian using PennyLane quantum chemistry tooling,
- prepares the Hartree–Fock (HF) state for the requested active space,
- applies real-time evolution in fixed increments of ``delta_t``, and
- returns the sequence of states, which can be treated as a (non-orthogonal) basis.

The basis growth differs between the two:

- ``qrte`` basis size after ``n_steps``: ``1 + n_steps``
- ``qrte_pmte`` basis size after ``n_steps``: ``1 + 2*n_steps``

``qrte_pmte`` uses the symmetric basis

.. math::

   \{|\psi_0\rangle,\ e^{-i\Delta t H}|\psi_0\rangle,\ e^{+i\Delta t H}|\psi_0\rangle,\ldots\}

and solves the projected generalized eigenproblem

.. math::

   M c = S c E.

Dependencies
------------
This function requires scientific Python dependencies that are installed with QCANT.

If you are developing from source, the recommended setup is:

.. code-block:: bash

   conda env create -f devtools/conda-envs/qcant.yaml
   conda activate qcant
   pip install -e . --no-deps

Basic usage (QRTE)
------------------

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

Basic usage (QRTE-PMTE)
-----------------------

.. code-block:: python

   import numpy as np
   import QCANT

   symbols = ["H", "H"]
   geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]])

   energies, basis_states, times = QCANT.qrte_pmte(
      symbols=symbols,
      geometry=geometry,
      delta_t=1e-6,
      n_steps=20,
      active_electrons=2,
      active_orbitals=2,
      basis="sto-3g",
      charge=0,
      spin=0,
      device_name="default.qubit",
   )

   print(times[:5])          # [0, +dt, -dt, +2dt, -2dt]
   print(energies)
   print(basis_states.shape) # (1 + 2*n_steps, 2**n_qubits)

Options
-------

- ``use_sparse``: use a sparse Hamiltonian representation for projections.
- ``basis_threshold``: drop amplitudes below this threshold after each basis update and
  re-normalize the state (use 0.0 to disable).
- ``device_kwargs``: optional PennyLane device constructor options, used for
  CPU or GPU devices.

Outputs
-------

Both functions return ``(energies, basis_states, times)``:

- ``energies``: eigenvalues obtained by diagonalizing the Hamiltonian in the generated basis
- ``basis_states``:

  - ``qrte`` shape ``(n_steps+1, 2**n_qubits)``
  - ``qrte_pmte`` shape ``(1 + 2*n_steps, 2**n_qubits)``

- ``times``:

  - ``qrte``: ``[0, dt, 2dt, ...]``
  - ``qrte_pmte``: ``[0, +dt, -dt, +2dt, -2dt, ...]``

If ``return_min_energy_history=True`` is passed, both return a fourth array
with cycle-wise minimum projected energy.

Notes
-----

- This implementation uses analytic statevector access (``qml.state()``).
- The returned basis is generally not orthonormal.
- Use ``device_name="lightning.gpu"`` to request the PennyLane GPU device.
  PySCF chemistry setup remains CPU-bound.

Comparison script
-----------------

To compare ``qrte``, ``qrte_pmte``, and ``qkud`` side-by-side for H2:

.. code-block:: bash

   python examples/run_qrte_pmte_h2.py --delta_t 1e-6 --epsilon 1e-3 --n_steps 20

This prints cycle-wise energies and errors with respect to active-space exact
diagonalization. Add ``--table_csv <path>`` to save the table.
