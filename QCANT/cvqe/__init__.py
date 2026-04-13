"""Cyclic VQE utilities."""

from __future__ import annotations

import sys

from .cvqe import cvqe


_parent_pkg = sys.modules.get("QCANT")
if _parent_pkg is not None:
    setattr(_parent_pkg, "cvqe", cvqe)

__all__ = [
    "cvqe",
]
