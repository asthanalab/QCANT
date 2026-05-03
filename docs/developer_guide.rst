Developer Guide
===============

This page captures the release-quality development workflow for QCANT.

Local Environment
-----------------

Use the conda environment for the full CPU stack:

.. code-block:: bash

   conda env create -f devtools/conda-envs/qcant.yaml
   conda activate qcant
   python -m pip install -e . --no-deps

Use the GPU environment on CUDA 12 systems:

.. code-block:: bash

   conda env create -f devtools/conda-envs/qcant-gpu.yaml
   conda activate qcant-gpu
   python -m pip install -e . --no-deps

Test And Build
--------------

Run the CPU test suite:

.. code-block:: bash

   python -m pytest -q QCANT/tests

Compile public source and examples:

.. code-block:: bash

   python -m compileall QCANT standalone scripts examples

Build docs:

.. code-block:: bash

   cd docs
   make html

Build the package:

.. code-block:: bash

   python -m build
   python -m twine check dist/*

Generated Artifacts
-------------------

Do not commit timestamped benchmark outputs or example run outputs. Keep only
source scripts, docs, and curated benchmark references under
``benchmarks/reference/``.

Compatibility Policy
--------------------

The next release is API-compatible and supports Python 3.11+. GPU acceleration
is opt-in. Sparse paths and Qulacs remain CPU-only unless explicitly changed in
a future release plan.
