"""
Unit tests for the qchem_units module.
"""

import numpy as np
import pytest

from QCANT.qchem_units import (
    geometry_to_bohr,
    geometry_for_pennylane,
    ANGSTROM_PER_BOHR,
    BOHR_PER_ANGSTROM,
)


def test_geometry_to_bohr():
    """Test the geometry_to_bohr function."""
    geom_angstrom = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]])
    geom_bohr = geometry_to_bohr(geom_angstrom)
    assert np.allclose(geom_bohr, geom_angstrom * BOHR_PER_ANGSTROM)


def test_geometry_for_pennylane_pyscf():
    """Test the geometry_for_pennylane function with the pyscf method."""
    geom_angstrom = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]])
    geom_bohr = geometry_for_pennylane(geom_angstrom, method="pyscf")
    assert np.allclose(geom_bohr, geom_angstrom * BOHR_PER_ANGSTROM)


def test_geometry_for_pennylane_other():
    """Test the geometry_for_pennylane function with another method."""
    geom_angstrom = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]])
    geom_other = geometry_for_pennylane(geom_angstrom, method="other")
    assert np.allclose(geom_other, geom_angstrom)


def test_constants():
    """Test the conversion constants."""
    assert np.isclose(ANGSTROM_PER_BOHR, 1.0 / BOHR_PER_ANGSTROM)
