Developer Guide
===============

This page describes basic development workflows for QCANT.

Local development
-----------------
QCANT has required runtime dependencies (NumPy/SciPy/PennyLane/PySCF). For the most reliable setup,
use the conda environment file and then do an editable install without pulling deps from pip:

.. code-block:: bash

	conda env create -f devtools/conda-envs/qcant.yaml
	conda activate qcant
	pip install -e . --no-deps

Running tests
-------------
QCANT uses ``pytest``.

.. code-block:: bash

	python -m pytest

Building documentation
----------------------
The documentation is built with Sphinx.

From the ``docs`` directory:

.. code-block:: bash

	make html

The generated site will be under ``docs/_build/html``.
