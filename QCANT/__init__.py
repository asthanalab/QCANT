"""QCANT public API.

QCANT collects active-space quantum chemistry workflows, projected-spectrum
methods, Krylov dynamics, finite-temperature routines, and optional accelerator
paths behind package-level imports. CPU execution remains the default; GPU and
Qulacs accelerators are explicitly selected by users.
"""

from .QCANT import canvas
from .adapt import adapt_vqe
from .adaptkrylov import adaptKrylov, adapt_krylov
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
	"adaptKrylov",
	"adapt_krylov",
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
