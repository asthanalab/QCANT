"""Geometry unit helpers for quantum chemistry backends.

This module provides constants and functions for converting molecular geometries
between Angstrom and Bohr units, which are commonly used in quantum chemistry.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

ANGSTROM_PER_BOHR: float = 0.529177210903
"""Conversion factor from Bohr to Angstrom."""

BOHR_PER_ANGSTROM: float = 1.0 / ANGSTROM_PER_BOHR
"""Conversion factor from Angstrom to Bohr."""


def geometry_to_bohr(geometry: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return a copy of geometry converted from Angstrom to Bohr.

    Parameters
    ----------
    geometry
        NumPy array representing the molecular geometry in Angstrom.

    Returns
    -------
    numpy.ndarray
        Geometry converted to Bohr.
    """
    return np.asarray(geometry, dtype=float) * BOHR_PER_ANGSTROM


def geometry_for_pennylane(
    geometry: NDArray[np.float64], *, method: str = "pyscf"
) -> NDArray[np.float64]:
    """Return geometry in units expected by PennyLane's qchem backend.

    Parameters
    ----------
    geometry
        NumPy array representing the molecular geometry in Angstrom.
    method
        Quantum chemistry backend method used by PennyLane. Defaults to
        ``"pyscf"``.

    Returns
    -------
    numpy.ndarray
        Geometry converted to the units expected by the selected backend.
    """

    if method == "pyscf":
        return geometry_to_bohr(geometry)

    return np.asarray(geometry, dtype=float)
