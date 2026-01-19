User Guide
===============

This page describes the current user-facing API.

Overview
--------
QCANT is currently a lightweight scaffold created from a standard scientific
Python project template. The package provides a minimal example function
(:func:`QCANT.canvas`) to verify imports and documentation wiring.

If you are extending QCANT for your research code, treat :mod:`QCANT` as the
stable entry point: add new modules for functionality and re-export the
supported functions/classes at the package level.

Discovering the API
-------------------
In Python you can inspect what QCANT exposes:

.. code-block:: python

	import QCANT

	help(QCANT)

Or jump directly to the API reference in these docs.

Algorithms
----------

.. toctree::
	:maxdepth: 1

	adapt_vqe
	krylov
	qkud
	qrte
	qsceom
