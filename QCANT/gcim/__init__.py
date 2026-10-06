"""Generalized Configuration Interaction Method (GCIM) utilities."""

from .gcim import gcim
from ._matrix import gcim_from_matrices

__all__ = [
    "gcim",
    "gcim_from_matrices",
]
