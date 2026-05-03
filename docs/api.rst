API Reference
=============

This page lists the supported package-level API exported from :mod:`QCANT`.
The examples and user guide use these imports directly.

Adaptive and projected methods
------------------------------

.. autosummary::
   :toctree: autosummary

   QCANT.adapt_vqe
   QCANT.adaptKrylov
   QCANT.adapt_krylov
   QCANT.gcim
   QCANT.cvqe
   QCANT.tepid_adapt
   QCANT.tepid_boltzmann_weights
   QCANT.qscEOM

Krylov and time-evolution methods
---------------------------------

.. autosummary::
   :toctree: autosummary

   QCANT.qrte
   QCANT.qrte_pmte
   QCANT.exact_krylov
   QCANT.qkud

Qulacs CPU accelerators
-----------------------

.. autosummary::
   :toctree: autosummary

   QCANT.adapt_vqe_qulacs
   QCANT.cvqe_qulacs
   QCANT.qkud_qulacs
   QCANT.qrte_qulacs
   QCANT.qrte_pmte_qulacs

Geometry helpers and metadata
-----------------------------

.. autosummary::
   :toctree: autosummary

   QCANT.geometry_for_pennylane
   QCANT.geometry_to_bohr
   QCANT.canvas
   QCANT.__version__
