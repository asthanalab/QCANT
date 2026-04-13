"""Focused unit tests for ADAPT-Krylov helpers and branch selection."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix
import scipy.sparse.linalg as spla

from QCANT.adaptkrylov import adaptkrylov as adaptkrylov_module
from QCANT.adaptkrylov.adaptkrylov import (
    _basis_rank_from_vectors,
    _compute_exact_ground_energy_qulacs,
    _make_device,
    _project_ground_energy,
    _qulacs_compatible_kwargs,
    _solve_krylov_orders_dense,
    adaptKrylov,
)


def test_make_device_prefers_requested_backend_then_falls_back():
    calls = []

    def fake_device(name, wires):
        calls.append((name, wires))
        if name in {"broken", "lightning.qubit"}:
            raise RuntimeError("device unavailable")
        return {"name": name, "wires": wires}

    fake_qml = SimpleNamespace(device=fake_device)

    assert _make_device(fake_qml, "default.qubit", 2) == {"name": "default.qubit", "wires": 2}
    assert _make_device(fake_qml, "broken", 3) == {"name": "default.qubit", "wires": 3}
    assert _make_device(fake_qml, None, 4) == {"name": "default.qubit", "wires": 4}
    assert calls[-2:] == [("lightning.qubit", 4), ("default.qubit", 4)]


def test_krylov_linear_algebra_helpers_cover_success_and_failure_cases(monkeypatch):
    hamiltonian = np.diag([1.0, 2.0]).astype(complex)
    basis = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]

    energy, rank = _project_ground_energy(np, basis, hamiltonian, overlap_tol=1e-12)
    assert energy == pytest.approx(1.0)
    assert rank == 2

    with pytest.raises(ValueError, match="Krylov basis collapsed numerically"):
        _project_ground_energy(np, [np.array([0.0, 0.0])], hamiltonian, overlap_tol=1e-12)

    psi = np.array([1.0, 0.0], dtype=complex)
    order1, order2, rank1, rank2 = _solve_krylov_orders_dense(np, psi, hamiltonian, overlap_tol=1e-12)
    assert order1 == pytest.approx(1.0)
    assert order2 == pytest.approx(1.0)
    assert rank1 == rank2 == 1

    assert _basis_rank_from_vectors(np, basis, overlap_tol=1e-12) == 2
    with pytest.raises(ValueError, match="Krylov basis collapsed numerically"):
        _basis_rank_from_vectors(np, [np.array([0.0, 0.0])], overlap_tol=1e-12)

    fake_h = SimpleNamespace(get_matrix=lambda: csr_matrix(np.diag([2.0, 1.0])))
    assert _compute_exact_ground_energy_qulacs(fake_h, np) == pytest.approx(1.0)

    monkeypatch.setattr(spla, "eigsh", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    assert _compute_exact_ground_energy_qulacs(fake_h, np) == pytest.approx(1.0)


def test_qulacs_compatibility_flags_are_checked():
    assert _qulacs_compatible_kwargs(
        device_name=None,
        shots=0,
        commutator_shots=0,
        commutator_mode="ansatz",
        commutator_debug=False,
        device_kwargs=None,
    )
    assert not _qulacs_compatible_kwargs(
        device_name="default.qubit",
        shots=0,
        commutator_shots=0,
        commutator_mode="ansatz",
        commutator_debug=False,
        device_kwargs=None,
    )
    assert not _qulacs_compatible_kwargs(
        device_name=None,
        shots=100,
        commutator_shots=0,
        commutator_mode="ansatz",
        commutator_debug=False,
        device_kwargs=None,
    )


def test_adaptkrylov_backend_selection_and_fallback(monkeypatch):
    captured = {}

    def fake_qulacs(*args, **kwargs):
        captured["qulacs"] = kwargs
        return "qulacs-path"

    def fake_pennylane(*args, **kwargs):
        captured["pennylane"] = kwargs
        return "pennylane-path"

    monkeypatch.setattr(adaptkrylov_module, "_run_adapt_krylov_qulacs", fake_qulacs)
    monkeypatch.setattr(adaptkrylov_module, "_run_adapt_krylov_pennylane", fake_pennylane)

    result = adaptKrylov(
        symbols=["H"],
        geometry=[[0.0, 0.0, 0.0]],
        adapt_it=0,
        active_electrons=1,
        active_orbitals=1,
    )
    assert result == "qulacs-path"
    assert captured["qulacs"]["parallel_gradients"] is True

    result = adaptKrylov(
        symbols=["H"],
        geometry=[[0.0, 0.0, 0.0]],
        adapt_it=0,
        active_electrons=1,
        active_orbitals=1,
        backend="pennylane",
    )
    assert result == "pennylane-path"
    assert captured["pennylane"]["parallel_gradients"] is False

    monkeypatch.setattr(
        adaptkrylov_module,
        "_run_adapt_krylov_qulacs",
        lambda *args, **kwargs: (_ for _ in ()).throw(ImportError("no qulacs")),
    )
    result = adaptKrylov(
        symbols=["H"],
        geometry=[[0.0, 0.0, 0.0]],
        adapt_it=0,
        active_electrons=1,
        active_orbitals=1,
    )
    assert result == "pennylane-path"


def test_adaptkrylov_backend_validation_rejects_bad_requests():
    with pytest.raises(ValueError, match="backend must be one of"):
        adaptKrylov(
            symbols=["H"],
            geometry=[[0.0, 0.0, 0.0]],
            adapt_it=0,
            active_electrons=1,
            active_orbitals=1,
            backend="bad",
        )

    with pytest.raises(ValueError, match="backend='qulacs' currently supports only analytic ADAPT runs"):
        adaptKrylov(
            symbols=["H"],
            geometry=[[0.0, 0.0, 0.0]],
            adapt_it=0,
            active_electrons=1,
            active_orbitals=1,
            backend="qulacs",
            device_name="default.qubit",
        )
