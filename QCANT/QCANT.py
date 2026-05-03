"""Small compatibility helpers re-exported by :mod:`QCANT`.

The scientific workflows live in dedicated subpackages and are imported from
the package root. ``canvas`` is kept as a lightweight import smoke test for
existing users and tests.
"""

from __future__ import annotations


def canvas(with_attribution: bool = True) -> str:
    """Return a short quote used as an import smoke test.

    This function is intentionally simple and remains part of the public API for
    backward compatibility with early QCANT releases.

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
