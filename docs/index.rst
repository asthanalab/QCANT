QCANT Documentation
===================

.. raw:: html

   <div class="qcant-hero">
     <h2>Quantum chemistry algorithms, accelerator paths, and reproducible examples.</h2>
     <p>
       QCANT brings together PySCF/PennyLane chemistry setup, ADAPT-style
       ansatz construction, qscEOM spectra, Krylov dynamics, finite-temperature
       workflows, CPU Qulacs acceleration, and explicit single-GPU execution.
     </p>
   </div>

QCANT is built for near-term quantum computing research where the same molecule
often needs to be explored across algorithm families and hardware backends. The
CPU path is the default; GPU acceleration is opt-in and documented explicitly.

.. grid:: 1 1 2 2

   .. grid-item-card:: Start Running
      :margin: 0 3 0 0

      Install QCANT, run H2/H4 examples, and understand the expected outputs.

      .. button-link:: ./getting_started.html
         :color: primary
         :outline:
         :expand:

         Getting Started

   .. grid-item-card:: Algorithm Map
      :margin: 0 3 0 0

      Choose between ADAPT-VQE, qscEOM, GCIM, Krylov, CVQE, and TEPID workflows.

      .. button-link:: ./user_guide.html
         :color: primary
         :outline:
         :expand:

         User Guide

   .. grid-item-card:: GPU And Talon
      :margin: 0 3 0 0

      Use ``lightning.gpu`` or ``array_backend="cupy"`` safely on one GPU.

      .. button-link:: ./gpu_acceleration.html
         :color: primary
         :outline:
         :expand:

         GPU Guide

   .. grid-item-card:: API Reference
      :margin: 0 3 0 0

      Browse the public package API and docstring-generated reference pages.

      .. button-link:: ./api.html
         :color: primary
         :outline:
         :expand:

         API Reference

.. toctree::
   :maxdepth: 2
   :hidden:
   :titlesonly:

   getting_started
   user_guide
   examples
   gpu_acceleration
   benchmarking
   parallelization
   qulacs_backend
   api
   developer_guide

.. toctree::
   :maxdepth: 1
   :caption: Algorithms
   :hidden:

   adapt_vqe
   adapt_krylov
   qsceom
   gcim
   cvqe
   tepid_adapt
   krylov
   qkud
   qrte
