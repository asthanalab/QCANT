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
from .qrte import qrte
from .qsceom import qscEOM
from ._version import __version__

__all__ = [
	"adapt_vqe",
	"qrte",
	"qscEOM",
	"canvas",
	"__version__",
]
