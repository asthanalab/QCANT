"""QCANT: quantum computing utilities (chemistry/materials science).

This package exposes a stable public API while loading heavy scientific
submodules lazily on first access.
"""

from __future__ import annotations

from importlib import import_module

from ._version import __version__


_LAZY_EXPORTS = {
    "canvas": (".QCANT", "canvas"),
    "adapt_vqe": (".adapt", "adapt_vqe"),
    "adaptKrylov": (".adaptkrylov", "adaptKrylov"),
    "adapt_krylov": (".adaptkrylov", "adapt_krylov"),
    "qrte": (".qrte", "qrte"),
    "qrte_pmte": (".qrte", "qrte_pmte"),
    "exact_krylov": (".krylov", "exact_krylov"),
    "qkud": (".qkud", "qkud"),
    "gcim": (".gcim", "gcim"),
    "gucj_gcim": (".gucj_gcim", "gucj_gcim"),
    "cvqe": (".cvqe", "cvqe"),
    "qscEOM": (".qsceom", "qscEOM"),
    "tepid_adapt": (".tepid_adapt", "tepid_adapt"),
    "tepid_boltzmann_weights": (".tepid_adapt", "tepid_boltzmann_weights"),
    "geometry_for_pennylane": (".qchem_units", "geometry_for_pennylane"),
    "geometry_to_bohr": (".qchem_units", "geometry_to_bohr"),
    "adapt_vqe_qulacs": (".qulacs_accel", "adapt_vqe_qulacs"),
    "cvqe_qulacs": (".qulacs_accel", "cvqe_qulacs"),
    "qkud_qulacs": (".qulacs_accel", "qkud_qulacs"),
    "qrte_qulacs": (".qulacs_accel", "qrte_qulacs"),
    "qrte_pmte_qulacs": (".qulacs_accel", "qrte_pmte_qulacs"),
}


def __getattr__(name: str):
    """Lazily resolve heavy public exports on first access."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


__all__ = [
    *list(_LAZY_EXPORTS),
    "__version__",
]
