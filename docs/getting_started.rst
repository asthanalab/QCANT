Getting Started
===============

QCANT currently ships with a small template-derived API.

Installation
------------
QCANT requires scientific Python dependencies (installed automatically when you install QCANT):

- ``numpy<2`` and ``scipy<2``
- ``pennylane``
- ``pyscf``
- ``autoray<0.7``

For development (recommended: conda env for the full stack):

.. code-block:: bash

    conda env create -f devtools/conda-envs/qcant.yaml
    conda activate qcant
    pip install -e . --no-deps

For development (pip/venv):

.. code-block:: bash

    pip install -e .

To enable the optional Qulacs backend:

.. code-block:: bash

    pip install -e ".[qulacs]"

Quickstart
----------
.. code-block:: python

    import QCANT

    print(QCANT.canvas())

You should see a short quote printed.
