"""Focused unit tests for CVQE helper utilities."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

import QCANT
from QCANT.cvqe.cvqe import (
    _apply_local_gate,
    _apply_pair_diagonal_phase,
    _bits_to_index,
    _build_reference_state,
    _build_uccsd_excitation_metadata,
    _extract_active_ccsd_amplitudes,
    _index_to_bits,
    _initialize_lucj_params,
    _initialize_uccsd_params,
    _pack_params,
    _pack_uccsd_params,
    _parameter_slices,
    _parameter_slices_uccsd,
    _restore_history,
    _select_new_determinant,
    _to_plain_data,
    _unpack_params,
    _unpack_uccsd_params,
    _validate_inputs,
    _write_json_payload,
    cvqe as internal_cvqe,
)


def test_public_cvqe_export_remains_callable_after_submodule_import():
    """Top-level QCANT.cvqe should still resolve to the callable function."""
    assert callable(QCANT.cvqe)
    assert QCANT.cvqe.cvqe is internal_cvqe


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"symbols": []}, "symbols must be non-empty"),
        ({"adapt_it": -1}, "adapt_it must be >= 0"),
        ({"shots": -1}, "shots must be >= 0"),
        ({"active_electrons": 0}, "active_electrons must be > 0"),
        ({"active_orbitals": 0}, "active_orbitals must be > 0"),
        ({"selection_topk": 0}, "selection_topk must be > 0"),
        ({"active_electrons": 5, "active_orbitals": 2}, "active_electrons cannot exceed 2 \\* active_orbitals"),
        ({"spin": 2}, "spin=0"),
        ({"active_electrons": 3}, "even number of active_electrons"),
        ({"geometry": np.zeros((2, 3))}, "geometry must have the same length as symbols"),
    ],
)
def test_validate_inputs_rejects_invalid_values(updates, message):
    kwargs = {
        "symbols": ["H"],
        "geometry": np.zeros((1, 3)),
        "adapt_it": 1,
        "shots": 0,
        "active_electrons": 2,
        "active_orbitals": 2,
        "spin": 0,
        "selection_topk": 1,
    }
    kwargs.update(updates)

    with pytest.raises(ValueError, match=message):
        _validate_inputs(**kwargs)


def test_bit_reference_and_parameter_helpers_round_trip():
    bits = [1, 0, 1, 0]
    qubits = 4

    index = _bits_to_index(bits)
    assert index == 10
    assert _index_to_bits(index, qubits) == bits

    reference = _build_reference_state(
        hf_bits=[1, 1, 0, 0],
        added_determinants=[bits],
        coeff_params=np.array([3.0]),
        qubits=qubits,
        np=np,
    )
    assert np.isclose(np.linalg.norm(reference), 1.0)
    np.testing.assert_allclose(
        np.abs(reference[[12, 10]]),
        np.array([1.0, 3.0]) / np.sqrt(10.0),
        atol=1e-12,
    )

    with pytest.raises(ValueError, match="reference superposition became numerically zero"):
        _build_reference_state(
            hf_bits=bits,
            added_determinants=[bits],
            coeff_params=np.array([-1.0]),
            qubits=qubits,
            np=np,
        )

    packed = _pack_params(
        det_coeffs=np.array([0.2, -0.3]),
        orbital_angles=np.array([1.0, 2.0]),
        same_spin=np.array([3.0, 4.0]),
        opposite_spin=np.array([5.0, 6.0]),
        np=np,
    )
    assert _parameter_slices(n_det_params=2, n_orb_pairs=2) == {
        "determinant_coeffs": (0, 2),
        "lucj_orbital": (2, 4),
        "lucj_same_spin": (4, 6),
        "lucj_opposite_spin": (6, 8),
    }
    unpacked = _unpack_params(packed, n_det_params=2, n_orb_pairs=2, np=np)
    np.testing.assert_allclose(unpacked[0], np.array([0.2, -0.3]))
    np.testing.assert_allclose(unpacked[1], np.array([1.0, 2.0]))
    np.testing.assert_allclose(unpacked[2], np.array([3.0, 4.0]))
    np.testing.assert_allclose(unpacked[3], np.array([5.0, 6.0]))

    packed_uccsd = _pack_uccsd_params(np.array([0.7]), np.array([1.5, -2.5]), np=np)
    assert _parameter_slices_uccsd(n_det_params=1, n_uccsd_params=2) == {
        "determinant_coeffs": (0, 1),
        "uccsd": (1, 3),
    }
    det_coeffs, uccsd_weights = _unpack_uccsd_params(
        packed_uccsd,
        n_det_params=1,
        n_uccsd_params=2,
        np=np,
    )
    np.testing.assert_allclose(det_coeffs, np.array([0.7]))
    np.testing.assert_allclose(uccsd_weights, np.array([1.5, -2.5]))


def test_extract_active_ccsd_amplitudes_and_lucj_initialization():
    mycc = SimpleNamespace(
        nocc=3,
        t1=np.arange(9, dtype=float).reshape(3, 3),
        t2=np.arange(81, dtype=float).reshape(3, 3, 3, 3),
    )

    t1_active, t2_active = _extract_active_ccsd_amplitudes(
        mycc,
        active_electrons=4,
        active_orbitals=4,
        np=np,
    )
    np.testing.assert_allclose(t1_active, np.array([[3.0, 4.0], [6.0, 7.0]]))
    np.testing.assert_allclose(t2_active, mycc.t2[1:3, 1:3, :2, :2])

    with pytest.raises(ValueError, match="active_orbitals must be at least active_electrons // 2"):
        _extract_active_ccsd_amplitudes(
            mycc,
            active_electrons=4,
            active_orbitals=1,
            np=np,
        )

    orbital_angles, same_spin, opposite_spin = _initialize_lucj_params(
        t1=np.array([[0.4, 0.0]], dtype=float),
        t2=np.zeros((1, 1, 2, 2), dtype=float),
        active_orbitals=3,
        active_electrons=2,
        np=np,
    )
    np.testing.assert_allclose(orbital_angles, np.array([-0.4, 0.0]))
    np.testing.assert_allclose(same_spin, np.zeros(2))
    np.testing.assert_allclose(opposite_spin, np.zeros(2))

    zero_pair_payload = _initialize_lucj_params(
        t1=np.zeros((1, 0), dtype=float),
        t2=np.zeros((1, 1, 0, 0), dtype=float),
        active_orbitals=1,
        active_electrons=2,
        np=np,
    )
    assert all(block.size == 0 for block in zero_pair_payload)


def test_uccsd_metadata_and_weight_initialization():
    class FakeQChem:
        @staticmethod
        def excitations(active_electrons, qubits):
            assert active_electrons == 2
            assert qubits == 4
            return [[0, 2]], [[0, 1, 2, 3]]

    fake_qml = SimpleNamespace(qchem=FakeQChem())

    metadata = _build_uccsd_excitation_metadata(
        active_electrons=2,
        qubits=4,
        qml=fake_qml,
    )
    assert metadata == [
        {
            "kind": "single",
            "excitation": [0, 2],
            "target_wires": (0, 1, 2),
            "wires": [0, 1, 2],
        },
        {
            "kind": "double",
            "excitation": [0, 1, 2, 3],
            "target_wires": (0, 1, 2, 3),
            "wires1": [0, 1],
            "wires2": [2, 3],
        },
    ]

    weights = _initialize_uccsd_params(
        t1=np.array([[0.25]], dtype=float),
        t2=np.array([[[[-0.75]]]], dtype=float),
        excitation_metadata=metadata,
        active_electrons=2,
        np=np,
    )
    np.testing.assert_allclose(weights, np.array([0.25, -0.75]))


def test_gate_phase_and_selection_helpers_cover_key_branches():
    state = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
    flip_all = np.zeros((4, 4), dtype=complex)
    flip_all[3, 0] = 1.0
    transformed = _apply_local_gate(state, flip_all, target_wires=(0, 1), qubits=2, np=np)
    np.testing.assert_allclose(transformed, np.array([0.0, 0.0, 0.0, 1.0], dtype=complex))

    occupancy_table = np.asarray([_index_to_bits(idx, 4) for idx in range(16)], dtype=float)
    phased = _apply_pair_diagonal_phase(
        np.eye(16, dtype=complex)[_bits_to_index([1, 0, 1, 0])],
        occupancy_table,
        alpha_p=0,
        beta_p=1,
        alpha_q=2,
        beta_q=3,
        gamma_same=np.pi,
        gamma_opposite=0.0,
        np=np,
    )
    assert phased[_bits_to_index([1, 0, 1, 0])] == pytest.approx(-1.0 + 0.0j, abs=1e-12)

    exact_choice = _select_new_determinant(
        probabilities=np.array([0.1, 0.6, 0.3], dtype=float),
        selected_indices={1},
        candidate_indices=np.array([0, 1, 2], dtype=int),
        shots=0,
        selection_method="probability",
        selection_topk=3,
        current_state=np.sqrt(np.array([0.1, 0.6, 0.3], dtype=float)),
        current_energy=0.5,
        h_matrix=np.diag([0.0, 1.0, 3.0]),
        rng=np.random.default_rng(1),
        np=np,
    )
    assert exact_choice[:4] == (2, pytest.approx(0.3), pytest.approx(0.3), None)
    assert exact_choice[4] == {"selection_strategy": "probability", "selection_topk": 1}

    sampled_choice = _select_new_determinant(
        probabilities=np.array([0.05, 0.05, 0.9], dtype=float),
        selected_indices=set(),
        candidate_indices=np.array([0, 1, 2], dtype=int),
        shots=20,
        selection_method="probability",
        selection_topk=3,
        current_state=np.sqrt(np.array([0.05, 0.05, 0.9], dtype=float)),
        current_energy=0.5,
        h_matrix=np.diag([0.0, 1.0, 3.0]),
        rng=np.random.default_rng(0),
        np=np,
    )
    assert sampled_choice[0] == 2
    assert sampled_choice[2] == pytest.approx(0.9)
    assert int(np.sum(sampled_choice[3])) == 20

    pt2_choice = _select_new_determinant(
        probabilities=np.array([0.1, 0.4, 0.5], dtype=float),
        selected_indices=set(),
        candidate_indices=np.array([0, 1, 2], dtype=int),
        shots=0,
        selection_method="pt2",
        selection_topk=2,
        current_state=np.sqrt(np.array([0.1, 0.4, 0.5], dtype=float)),
        current_energy=0.5,
        h_matrix=np.diag([0.0, 1.0, 3.0]),
        rng=np.random.default_rng(2),
        np=np,
    )
    assert pt2_choice[0] == 2
    assert pt2_choice[1] == pytest.approx(1.25)
    assert pt2_choice[4]["selection_strategy"] == "pt2"
    assert pt2_choice[4]["screened_candidate_count"] == 2

    no_candidates = _select_new_determinant(
        probabilities=np.array([1.0], dtype=float),
        selected_indices=set(),
        candidate_indices=np.array([], dtype=int),
        shots=0,
        selection_method="probability",
        selection_topk=1,
        current_state=np.array([1.0], dtype=complex),
        current_energy=0.0,
        h_matrix=np.array([[0.0]], dtype=float),
        rng=np.random.default_rng(3),
        np=np,
    )
    assert no_candidates == (None, 0.0, 0.0, None, {"selection_strategy": "probability"})

    with pytest.raises(ValueError, match="selection_method='pt2' currently requires exact selection with shots=0"):
        _select_new_determinant(
            probabilities=np.array([1.0], dtype=float),
            selected_indices=set(),
            candidate_indices=np.array([0], dtype=int),
            shots=5,
            selection_method="pt2",
            selection_topk=1,
            current_state=np.array([1.0], dtype=complex),
            current_energy=0.0,
            h_matrix=np.array([[0.0]], dtype=float),
            rng=np.random.default_rng(4),
            np=np,
        )


def test_serialization_helpers_round_trip(tmp_path):
    plain = _to_plain_data(
        {
            "array": np.array([1, 2, 3]),
            "scalar": np.int64(7),
            "nested": (np.array([1.5]), {"value": np.float64(2.5)}),
        },
        np=np,
    )
    assert plain == {
        "array": [1, 2, 3],
        "scalar": 7,
        "nested": [[1.5], {"value": 2.5}],
    }

    history = _restore_history(
        [
            {
                "determinant_coeffs": [0.2, -0.1],
                "lucj_orbital": [0.3],
                "lucj_same_spin": [0.4],
                "lucj_opposite_spin": [0.5],
                "uccsd_weights": [0.6, 0.7],
                "sample_counts": [1, 2, 3],
                "selected_determinant": [1, 0, 1, 0],
            }
        ],
        np=np,
    )
    np.testing.assert_allclose(history[0]["determinant_coeffs"], np.array([0.2, -0.1]))
    np.testing.assert_allclose(history[0]["uccsd_weights"], np.array([0.6, 0.7]))
    np.testing.assert_array_equal(history[0]["sample_counts"], np.array([1, 2, 3]))
    assert history[0]["selected_determinant"] == [1, 0, 1, 0]

    payload_path = tmp_path / "nested" / "checkpoint.json"
    _write_json_payload(payload_path, plain)
    assert payload_path.exists()
    assert not payload_path.with_name("checkpoint.json.tmp").exists()
    assert json.loads(payload_path.read_text(encoding="utf-8")) == plain
