"""Core public API for QCANT.

At the moment QCANT is a small, template-derived package with a minimal API.
The intent is for project-specific functionality to live in dedicated modules
and be re-exported via :mod:`QCANT`.
"""

from __future__ import annotations


def canvas(with_attribution: bool = True) -> str:
    """Return a short quote used as a template smoke-test.

    This function is intentionally simple and exists primarily to verify that
    the package imports correctly and that the documentation build is wired up.

    Parameters
    ----------
    with_attribution : bool
        If ``True``, append a short attribution line.

    Returns
    -------
    str
        The quote, optionally with attribution.
    """

    quote: str = "The code is but a canvas to our imagination."
    if with_attribution:
        quote += "\n\t- Adapted from Henry David Thoreau"
    return quote


if __name__ == "__main__":
    # Do something if this file is invoked on its own
    print(canvas())

