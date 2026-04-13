"""Focused unit tests for ADAPT-VQE helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from QCANT.adapt import adaptvqe as adapt_module
from QCANT.adapt.adaptvqe import (
    _ExcitationList,
    _ansatz_type_from_pool_type,
    _apply_excitation_gate,
    _build_operator_pool,
    _iter_chunks,
    _normalize_pool_type,
    _resolve_chunk_size,
    _resolve_parallel_backend,
    _resolve_worker_count,
    _validate_inputs,
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


def test_excitation_list_preserves_ansatz_metadata():
    excitations = _ExcitationList([[0, 2]], ansatz_type="qubit_excitation", pool_type="qe")

    assert excitations == [[0, 2]]
    assert excitations.ansatz_type == "qubit_excitation"
    assert excitations.pool_type == "qe"


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("fermionic_sd", "fermionic_sd"),
        ("sd", "fermionic_sd"),
        ("fermionic", "fermionic_sd"),
        ("qubit_excitation", "qubit_excitation"),
        ("qe", "qubit_excitation"),
        ("qubit", "qubit_excitation"),
    ],
)
def test_pool_type_aliases_normalize(alias, canonical):
    assert _normalize_pool_type(alias) == canonical
    assert _ansatz_type_from_pool_type(canonical) in {"fermionic", "qubit_excitation"}


def test_pool_type_helpers_reject_invalid_values():
    with pytest.raises(ValueError, match="pool_type must be one of"):
        _normalize_pool_type("bad")

    with pytest.raises(ValueError, match="Unsupported canonical pool_type"):
        _ansatz_type_from_pool_type("bad")


def test_apply_excitation_gate_routes_to_expected_qml_operator():
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


def test_build_operator_pool_supports_fermionic_and_qubit_variants():
    class FakeSingleExcitation:
        def __init__(self, _weight, wires):
            self.wires = tuple(wires)

        def generator(self):
            return 2.0

    class FakeDoubleExcitation:
        def __init__(self, _weight, wires):
            self.wires = tuple(wires)

        def generator(self):
            return 3.0

    fake_qml = SimpleNamespace(
        qchem=SimpleNamespace(excitations=lambda active_electrons, qubits: ([[0, 2]], [[0, 1, 2, 3]])),
        fermi=SimpleNamespace(
            FermiWord=lambda ops: ("fermion_word", tuple(sorted(ops.items()))),
            jordan_wigner=lambda op: ("jw", op),
        ),
        SingleExcitation=FakeSingleExcitation,
        DoubleExcitation=FakeDoubleExcitation,
    )

    excitations, fermionic_ops = _build_operator_pool(
        fake_qml,
        active_electrons=2,
        qubits=4,
        pool_type="fermionic_sd",
    )
    assert excitations == [[0, 2], [0, 1, 2, 3]]
    assert fermionic_ops[0][0] == "jw"
    assert fermionic_ops[1][0] == "jw"

    excitations_qe, qubit_ops = _build_operator_pool(
        fake_qml,
        active_electrons=2,
        qubits=4,
        pool_type="qubit_excitation",
    )
    assert excitations_qe == excitations
    assert qubit_ops == [2j, 3j]

    with pytest.raises(ValueError, match="Unsupported canonical pool_type"):
        _build_operator_pool(fake_qml, active_electrons=2, qubits=4, pool_type="bad")


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"symbols": []}, "symbols must be non-empty"),
        ({"adapt_it": -1}, "adapt_it must be >= 0"),
        ({"shots": -1}, "shots must be >= 0"),
        ({"commutator_shots": -1}, "commutator_shots must be >= 0"),
        ({"commutator_mode": "bad"}, "commutator_mode must be 'ansatz' or 'statevec'"),
        ({"hamiltonian_cutoff": -1.0}, "hamiltonian_cutoff must be >= 0"),
        ({"hamiltonian_source": "bad"}, "hamiltonian_source must be one of"),
        ({"pool_type": "bad"}, "pool_type must be one of"),
        ({"pool_sample_size": 0}, "pool_sample_size must be > 0"),
        ({"max_workers": 0}, "max_workers must be > 0"),
        ({"gradient_chunk_size": 0}, "gradient_chunk_size must be > 0"),
        ({"parallel_backend": "bad"}, "parallel_backend must be one of"),
        (
            {"pauli_grouping": True, "grouping_type": "bad"},
            "grouping_type must be one of",
        ),
        ({"geometry": [[0.0, 0.0, 0.0]]}, "geometry must have the same length as symbols"),
    ],
)
def test_validate_inputs_rejects_invalid_inputs(updates, message):
    kwargs = {
        "symbols": ["H", "H"],
        "geometry": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
        "adapt_it": 1,
        "shots": 0,
        "commutator_shots": 0,
        "commutator_mode": "ansatz",
        "hamiltonian_cutoff": 0.0,
        "hamiltonian_source": "casci",
        "pool_type": "fermionic_sd",
        "pool_sample_size": None,
        "max_workers": None,
        "gradient_chunk_size": None,
        "parallel_backend": "auto",
        "pauli_grouping": False,
        "grouping_type": "qwc",
    }
    kwargs.update(updates)

    with pytest.raises(ValueError, match=message):
        _validate_inputs(**kwargs)


def test_parallel_resolution_helpers_cover_default_paths(monkeypatch):
    monkeypatch.setattr(adapt_module.os, "cpu_count", lambda: 6)
    assert _resolve_worker_count(None) == 6
    assert _resolve_worker_count(3) == 3

    assert _resolve_chunk_size(total_items=0, worker_count=4, user_chunk_size=None) == 1
    assert _resolve_chunk_size(total_items=10, worker_count=4, user_chunk_size=None) == 3
    assert _resolve_chunk_size(total_items=10, worker_count=4, user_chunk_size=2) == 2
    assert list(_iter_chunks(list(range(5)), 2)) == [[0, 1], [2, 3], [4]]

    monkeypatch.setattr(adapt_module.os, "name", "posix", raising=False)
    assert _resolve_parallel_backend("auto") == "process"
    monkeypatch.setattr(adapt_module.os, "name", "nt", raising=False)
    assert _resolve_parallel_backend("auto") == "thread"
    assert _resolve_parallel_backend("thread") == "thread"
