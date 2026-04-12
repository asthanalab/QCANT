"""
Unit and regression test for the QCANT package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import QCANT


def test_QCANT_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "QCANT" in sys.modules


def test_canvas_without_attribution():
    """Test canvas function without attribution."""
    quote = QCANT.canvas(with_attribution=False)
    assert quote == "The code is but a canvas to our imagination."


def test_canvas_with_attribution():
    """Test canvas function with attribution."""
    quote = QCANT.canvas(with_attribution=True)
    assert quote == "The code is but a canvas to our imagination.\n\t- Adapted from Henry David Thoreau"
