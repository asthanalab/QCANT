"""Focused unit tests for qscEOM helpers and fast validation branches."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import QCANT
from QCANT.qsceom import qsceom as qsceom_module
from QCANT.qsceom.qsceom import (
    _apply_excitation_gate,
    _build_pyscf_molecular_integrals,
    _expand_spatial_integrals_to_spin_orbital,
    _iter_chunks,
    _jw_number_sector_indices,
    _normalize_ansatz_type,
    _normalize_projector_backend,
    _resolve_chunk_size,
    _resolve_parallel_backend,
    _resolve_worker_count,
    _restrict_basis_states_to_number_sector,
)


class _GateRecorder:
    def __init__(self):
        self.calls = []

    def FermionicDoubleExcitation(self, **kwargs):
        self.calls.append(("fermionic_double", kwargs))

    def FermionicSingleExcitation(self, **kwargs):
        self.calls.append(("fermionic_single", kwargs))

    def DoubleExcitation(self, weight, wires):
        self.calls.append(("qubit_double", {"weight": weight, "wires": wires}))

    def SingleExcitation(self, weight, wires):
        self.calls.append(("qubit_single", {"weight": weight, "wires": wires}))


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        (None, "fermionic"),
        ("fermionic", "fermionic"),
        ("fermionic_sd", "fermionic"),
        ("sd", "fermionic"),
        ("qubit_excitation", "qubit_excitation"),
        ("qe", "qubit_excitation"),
        ("qubit", "qubit_excitation"),
    ],
)
def test_qsceom_ansatz_aliases_normalize(alias, canonical):
    assert _normalize_ansatz_type(alias) == canonical


@pytest.mark.parametrize("backend", ["auto", "dense", "sparse_number_preserving"])
def test_qsceom_projector_backend_aliases_normalize(backend):
    assert _normalize_projector_backend(backend) == backend


def test_qsceom_normalizers_reject_invalid_values():
    with pytest.raises(ValueError, match="ansatz_type must be one of"):
        _normalize_ansatz_type("bad")

    with pytest.raises(ValueError, match="projector_backend must be one of"):
        _normalize_projector_backend("bad")


def test_qsceom_apply_excitation_gate_routes_correctly():
    recorder = _GateRecorder()

    _apply_excitation_gate(recorder, [0, 2], 0.5, ansatz_type="fermionic")
    _apply_excitation_gate(recorder, [0, 1, 2, 3], 0.7, ansatz_type="fermionic")
    _apply_excitation_gate(recorder, [0, 2], 0.9, ansatz_type="qubit_excitation")
    _apply_excitation_gate(recorder, [0, 1, 2, 3], 1.1, ansatz_type="qubit_excitation")

    assert recorder.calls == [
        ("fermionic_single", {"weight": 0.5, "wires": [0, 1, 2]}),
        ("fermionic_double", {"weight": 0.7, "wires1": [0, 1], "wires2": [2, 3]}),
        ("qubit_single", {"weight": 0.9, "wires": [0, 2]}),
        ("qubit_double", {"weight": 1.1, "wires": [0, 1, 2, 3]}),
    ]

    with pytest.raises(ValueError, match="Each excitation must have length 2"):
        _apply_excitation_gate(recorder, [0, 1, 2], 0.0, ansatz_type="fermionic")


def test_integral_expansion_and_sector_restriction_helpers():
    one_spin, two_spin = _expand_spatial_integrals_to_spin_orbital(
        np.array([[1.5]]),
        np.array([[[[2.0]]]]),
    )

    np.testing.assert_allclose(one_spin, np.diag([1.5, 1.5]))
    assert two_spin.shape == (2, 2, 2, 2)
    assert two_spin[0, 0, 0, 0] == pytest.approx(1.0)
    assert two_spin[0, 1, 1, 0] == pytest.approx(1.0)
    assert two_spin[1, 0, 0, 1] == pytest.approx(1.0)
    assert two_spin[1, 1, 1, 1] == pytest.approx(1.0)

    sector_indices = _jw_number_sector_indices(n_electrons=2, n_qubits=4)
    assert sector_indices == [12, 6, 5, 10, 9, 3]

    basis_states = np.arange(32).reshape(16, 2)
    restricted, selected = _restrict_basis_states_to_number_sector(
        basis_states,
        active_electrons=2,
        qubits=4,
    )
    np.testing.assert_array_equal(selected, np.array(sector_indices))
    np.testing.assert_array_equal(restricted, basis_states[selected, :])


def test_pyscf_integral_wrapper_returns_copies(monkeypatch):
    core = np.array([1.0])
    one = np.array([[2.0]])
    two = np.array([[[[3.0]]]])

    monkeypatch.setattr(
        qsceom_module,
        "_build_pyscf_molecular_integrals_cached",
        lambda **_: (core, one, two),
    )

    out_core, out_one, out_two = _build_pyscf_molecular_integrals(
        symbols=["H"],
        geometry=[[0.0, 0.0, 0.0]],
        basis="sto-3g",
        charge=0,
        active_electrons=1,
        active_orbitals=1,
    )

    out_core[0] = 10.0
    out_one[0, 0] = 20.0
    out_two[0, 0, 0, 0] = 30.0

    assert core[0] == 1.0
    assert one[0, 0] == 2.0
    assert two[0, 0, 0, 0] == 3.0


def test_qsceom_parallel_resolution_helpers(monkeypatch):
    monkeypatch.setattr(qsceom_module.os, "cpu_count", lambda: 5)
    assert _resolve_worker_count(None) == 5
    assert _resolve_worker_count(2) == 2

    assert _resolve_chunk_size(total_items=0, worker_count=4, user_chunk_size=None) == 1
    assert _resolve_chunk_size(total_items=10, worker_count=4, user_chunk_size=None) == 3
    assert _resolve_chunk_size(total_items=10, worker_count=4, user_chunk_size=2) == 2
    assert list(_iter_chunks(list(range(5)), 2)) == [[0, 1], [2, 3], [4]]

    monkeypatch.setattr(qsceom_module.os, "name", "posix", raising=False)
    assert _resolve_parallel_backend("auto") == "process"
    monkeypatch.setattr(qsceom_module.os, "name", "nt", raising=False)
    assert _resolve_parallel_backend("auto") == "thread"
    assert _resolve_parallel_backend("thread") == "thread"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_states": 1}, "max_states-based truncation has been removed"),
        ({"max_workers": 0}, "max_workers must be > 0"),
        ({"matrix_chunk_size": 0}, "matrix_chunk_size must be > 0"),
        ({"brg_tolerance": 0.0}, "brg_tolerance must be > 0"),
        ({"parallel_backend": "bad"}, "parallel_backend must be one of"),
        (
            {"pauli_grouping": True, "grouping_type": "bad"},
            "grouping_type must be one of",
        ),
        ({"brg_tolerance": 1e-6, "shots": 10}, "brg_tolerance requires shots=0"),
        (
            {"brg_tolerance": 1e-6, "method": "dhf"},
            "brg_tolerance requires method='pyscf'",
        ),
        (
            {"projector_backend": "sparse_number_preserving", "shots": 10},
            "projector_backend='sparse_number_preserving' requires shots=0",
        ),
        (
            {"projector_backend": "sparse_number_preserving", "method": "dhf"},
            "projector_backend='sparse_number_preserving' requires method='pyscf'",
        ),
        ({"params": [0.0], "ash_excitation": []}, "params and ash_excitation must have the same length"),
    ],
)
def test_qsceom_fast_validation_branches(kwargs, message):
    base_kwargs = {
        "symbols": ["H", "H"],
        "geometry": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
        "active_electrons": 2,
        "active_orbitals": 2,
        "charge": 0,
        "params": [],
        "ash_excitation": [],
    }
    base_kwargs.update(kwargs)

    with pytest.raises(ValueError, match=message):
        QCANT.qscEOM(**base_kwargs)
