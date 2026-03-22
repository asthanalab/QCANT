"""QCANT: quantum computing utilities (chemistry/materials science).

This package is currently a lightweight scaffold created from the MolSSI
cookiecutter template. The public API is intentionally small and stable.

Public API
----------
- :func:`QCANT.canvas` – small example function used by the template.
- :data:`QCANT.__version__` – package version string.
"""

from .QCANT import canvas
from .adapt import adapt_vqe
from .qrte import qrte, qrte_pmte
from .krylov import exact_krylov
from .qkud import qkud
from .gcim import gcim
from .cvqe import cvqe
from .qsceom import qscEOM
from .tepid_adapt import tepid_adapt, tepid_boltzmann_weights
from .qchem_units import geometry_for_pennylane, geometry_to_bohr
from .qulacs_accel import (
	adapt_vqe_qulacs,
	cvqe_qulacs,
	qkud_qulacs,
	qrte_pmte_qulacs,
	qrte_qulacs,
)
from ._version import __version__

__all__ = [
	"adapt_vqe",
	"qrte",
	"qrte_pmte",
	"exact_krylov",
	"qkud",
	"gcim",
	"cvqe",
	"qscEOM",
	"tepid_adapt",
	"tepid_boltzmann_weights",
	"canvas",
	"adapt_vqe_qulacs",
	"cvqe_qulacs",
	"qkud_qulacs",
	"qrte_qulacs",
	"qrte_pmte_qulacs",
	"geometry_for_pennylane",
	"geometry_to_bohr",
	"__version__",
]
