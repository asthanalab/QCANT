"""Cyclic VQE utilities."""

from __future__ import annotations

import sys
from types import ModuleType

from .cvqe import cvqe


class _CallableModule(ModuleType):
    """Allow ``QCANT.cvqe(...)`` even after the ``QCANT.cvqe`` submodule is imported."""

    def __call__(self, *args, **kwargs):
        return cvqe(*args, **kwargs)


sys.modules[__name__].__class__ = _CallableModule

__all__ = [
    "cvqe",
]
