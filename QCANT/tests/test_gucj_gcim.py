"""Tests for the g-uCJ-GRIM/GCIM implementation."""

from __future__ import annotations

import importlib.util
from functools import lru_cache
import os
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("pennylane")
pytest.importorskip("pyscf")

import QCANT


_IMPL_PATH = Path(__file__).resolve().parents[1] / "gucj_gcim" / "gucj_gcim.py"
_IMPL_SPEC = importlib.util.spec_from_file_location("gucj_gcim_impl", _IMPL_PATH)
_IMPL_MODULE = importlib.util.module_from_spec(_IMPL_SPEC)
assert _IMPL_SPEC is not None and _IMPL_SPEC.loader is not None
sys.modules[_IMPL_SPEC.name] = _IMPL_MODULE
_IMPL_SPEC.loader.exec_module(_IMPL_MODULE)
_build_system_data = _IMPL_MODULE._build_system_data
_default_k_param_shape_for_method = _IMPL_MODULE._default_k_param_shape_for_method
_evaluate_current_basis_for_selection = _IMPL_MODULE._evaluate_current_basis_for_selection
_evaluate_subspace = _IMPL_MODULE._evaluate_subspace
_projector_commutator_score = _IMPL_MODULE._projector_commutator_score
_resolve_initial_k_params = _IMPL_MODULE._resolve_initial_k_params
_resolve_k_params_and_source = _IMPL_MODULE._resolve_k_params_and_source

_RUN_LEGACY_TESTS = os.getenv("QCANT_RUN_GUCJ_GCIM_LEGACY") == "1"


def _geometry(symbol_count: int, bond_length: float) -> np.ndarray:
    return np.asarray([[0.0, 0.0, bond_length * idx] for idx in range(symbol_count)], dtype=float)


def _stretched_h2_kwargs():
    return dict(
        symbols=["H", "H"],
        geometry=_geometry(2, 1.5),
        basis="sto-3g",
        charge=0,
        spin=0,
        active_electrons=2,
        active_orbitals=2,
    )


def _stretched_h4_kwargs():
    return dict(
        symbols=["H"] * 4,
        geometry=_geometry(4, 1.5),
        basis="sto-3g",
        charge=0,
        spin=0,
        active_electrons=4,
        active_orbitals=4,
    )


@lru_cache(maxsize=None)
def _cached_h4_gucj_result(
    method_variant: str,
    k_param_shape=None,
    k_source=None,
):
    kwargs = dict(
        **_stretched_h4_kwargs(),
        method_variant=method_variant,
        subset_rank_max=2,
        kappa=np.pi / 4.0,
        outer_opt_maxiter=10,
        alternating_opt_maxiter=4,
        return_details=True,
    )
    if k_source is not None:
        kwargs["k_source"] = k_source
    if k_param_shape is not None:
        kwargs["k_param_shape"] = k_param_shape
    return QCANT.gucj_gcim(**kwargs)


@lru_cache(maxsize=None)
def _cached_h4_legacy_gcim_result():
    return QCANT.gcim(
        **_stretched_h4_kwargs(),
        adapt_it=0,
        pool_type="sd",
        theta=np.pi / 4.0,
        print_progress=False,
        return_details=True,
    )


def test_k_generator_is_antihermitian_h2():
    system = _build_system_data(
        **_stretched_h2_kwargs(),
        hamiltonian_cutoff=1e-20,
        subset_rank_max=2,
        k_param_shape="generalized_same_spin_vector",
    )

    for generator in np.asarray(system.k_generator_matrices, dtype=complex):
        np.testing.assert_allclose(
            np.asarray(generator).conj().T,
            -np.asarray(generator),
            atol=1e-10,
            rtol=0.0,
        )


def test_method_specific_default_k_shape_resolution():
    assert _default_k_param_shape_for_method("frozen_k") == "generalized_same_spin_vector"
    assert _default_k_param_shape_for_method("optimized_k_outer_loop") == "ov_same_spin_vector"
    assert _default_k_param_shape_for_method("alternating_k_and_subspace") == "ov_same_spin_vector"


def test_ccsd_t1_frozen_k_mapping_matches_h2_same_spin_order():
    system = _build_system_data(
        **_stretched_h2_kwargs(),
        hamiltonian_cutoff=1e-20,
        subset_rank_max=2,
        k_param_shape="ov_same_spin_vector",
        compute_ccsd_t1=True,
    )

    assert tuple(system.k_generator_pairs) == ((0, 2), (1, 3))
    assert system.ccsd_t1_k_params is not None
    ccsd_params = np.asarray(system.ccsd_t1_k_params, dtype=float)
    assert ccsd_params.shape == (2,)
    np.testing.assert_allclose(ccsd_params[0], ccsd_params[1], atol=1e-10, rtol=0.0)


def test_projector_masks_are_idempotent_and_commuting_h2():
    system = _build_system_data(
        **_stretched_h2_kwargs(),
        hamiltonian_cutoff=1e-20,
        subset_rank_max=2,
        k_param_shape="generalized_same_spin_vector",
    )

    rank1_count = len(system.projector_pairs)
    rank1_masks = np.asarray(system.subset_masks[1 : 1 + rank1_count], dtype=bool)

    for mask in rank1_masks:
        assert np.array_equal(mask & mask, mask)
    for idx in range(rank1_masks.shape[0]):
        for jdx in range(rank1_masks.shape[0]):
            assert np.array_equal(rank1_masks[idx] & rank1_masks[jdx], rank1_masks[jdx] & rank1_masks[idx])


def test_adaptive_selector_picks_max_score_h2():
    system = _build_system_data(
        **_stretched_h2_kwargs(),
        hamiltonian_cutoff=1e-20,
        subset_rank_max=2,
        k_param_shape="generalized_same_spin_vector",
    )
    k_params = _resolve_initial_k_params(system, kappa=np.pi / 4.0, k_params=None, np=np)
    _labels, _masks, rotation_artifacts, record = _evaluate_current_basis_for_selection(
        system,
        selected_projector_indices=tuple(),
        subset_rank_max=2,
        current_k_params=k_params,
        overlap_tol=1e-10,
        regularization=1e-10,
        np=np,
    )

    scores = []
    for projector_index in range(len(system.projector_pairs)):
        scores.append(
            _projector_commutator_score(
                system,
                projector_index=projector_index,
                ritz_state=record["_ritz_state_vector"],
                hbar_k=rotation_artifacts[1],
                np=np,
            )
        )

    assert int(np.argmax(scores)) == 0
    assert scores[0] == pytest.approx(max(scores), abs=1e-12)


def test_mask_score_matches_dense_commutator_reference_h2():
    system = _build_system_data(
        **_stretched_h2_kwargs(),
        hamiltonian_cutoff=1e-20,
        subset_rank_max=2,
        k_param_shape="generalized_same_spin_vector",
    )
    k_params = _resolve_initial_k_params(system, kappa=np.pi / 4.0, k_params=None, np=np)
    _labels, _masks, rotation_artifacts, record = _evaluate_current_basis_for_selection(
        system,
        selected_projector_indices=tuple(),
        subset_rank_max=2,
        current_k_params=k_params,
        overlap_tol=1e-10,
        regularization=1e-10,
        np=np,
    )

    projector_index = 1
    mask = np.asarray(system.projector_pair_masks[projector_index], dtype=float)
    projector_matrix = np.diag(mask.astype(complex))
    a_matrix = 1j * projector_matrix
    psi = np.asarray(record["_ritz_state_vector"], dtype=complex)
    hbar_k = np.asarray(rotation_artifacts[1], dtype=complex)
    comm_psi = hbar_k @ (a_matrix @ psi) - a_matrix @ (hbar_k @ psi)
    dense_score = abs(2.0 * float(np.real_if_close(np.vdot(psi, comm_psi))))

    helper_score = _projector_commutator_score(
        system,
        projector_index=projector_index,
        ritz_state=psi,
        hbar_k=hbar_k,
        np=np,
    )
    assert helper_score == pytest.approx(dense_score, abs=1e-10)


def test_subset_rank_zero_matches_hf_energy_h2():
    k_history, subset_labels, energies, details = QCANT.gucj_gcim(
        **_stretched_h2_kwargs(),
        method_variant="frozen_K",
        subset_rank_max=0,
        kappa=np.pi / 4.0,
        return_details=True,
    )

    assert len(k_history) == len(subset_labels) == len(energies) == 1
    assert subset_labels[0] == tuple()
    assert abs(float(energies[0]) - float(details["hf_energy"])) < 1e-10
    assert details["k_source"] == "heuristic"
    assert details["k_param_shape"] == "generalized_same_spin_vector"


def test_default_frozen_k_runs_stretched_h2():
    k_history, subset_labels, energies, details = QCANT.gucj_gcim(
        **_stretched_h2_kwargs(),
        method_variant="frozen_K",
        subset_rank_max=2,
        kappa=np.pi / 4.0,
        num_threads=1,
        return_details=True,
    )

    assert len(k_history) == len(subset_labels) == len(energies) == len(details["iteration_records"])
    assert len(energies) > 0
    assert np.all(np.isfinite(np.asarray(energies, dtype=float)))
    assert float(np.min(np.asarray(energies, dtype=float))) >= float(details["fci_energy"]) - 1e-6
    assert details["selection_mode"] == "adaptive_commutator"
    assert details["variant_status"] == "default"
    assert len(details["selected_projector_order"]) == len(energies)
    assert len({tuple(label) for label in details["selected_projector_order"]}) == len(details["selected_projector_order"])
    assert details["k_source"] == "heuristic"
    assert details["k_param_shape"] == "generalized_same_spin_vector"
    assert float(details["iteration_records"][-1]["abs_error_fci"]) < 1e-10

    retained_dims = [int(record["basis_dimension_after_screening"]) for record in details["iteration_records"]]
    assert all(next_dim >= cur_dim for cur_dim, next_dim in zip(retained_dims, retained_dims[1:]))
    assert details["num_threads"] == 1


def test_overlap_and_projected_h_are_hermitian_h2():
    _k_history, _subset_labels, _energies, details = QCANT.gucj_gcim(
        **_stretched_h2_kwargs(),
        method_variant="frozen_K",
        subset_rank_max=2,
        kappa=np.pi / 4.0,
        include_iteration_matrices=True,
        return_details=True,
    )

    last_record = details["iteration_records"][-1]
    overlap = np.asarray(last_record["overlap_matrix"], dtype=complex)
    projected_h = np.asarray(last_record["projected_hamiltonian"], dtype=complex)

    np.testing.assert_allclose(overlap, overlap.conj().T, atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(projected_h, projected_h.conj().T, atol=1e-10, rtol=0.0)
    assert float(np.min(np.linalg.eigvalsh(overlap))) > -1e-8


def test_manual_k_source_requires_explicit_k_params():
    with pytest.raises(ValueError, match="requires explicit k_params"):
        QCANT.gucj_gcim(
            **_stretched_h2_kwargs(),
            method_variant="frozen_K",
            k_source="manual",
            subset_rank_max=2,
        )


@pytest.mark.parametrize("symbol_count", [2, 4])
def test_exact_fci_reference_builds_for_h2_and_h4(symbol_count: int):
    system = _build_system_data(
        symbols=["H"] * symbol_count,
        geometry=_geometry(symbol_count, 1.5),
        basis="sto-3g",
        charge=0,
        spin=0,
        active_electrons=symbol_count,
        active_orbitals=symbol_count,
        hamiltonian_cutoff=1e-20,
        subset_rank_max=2,
        k_param_shape="generalized_same_spin_vector",
    )

    assert np.isfinite(float(system.fci_energy))
    assert float(system.fci_energy) <= float(system.hf_energy) + 1e-8


def test_scalar_control_is_non_exact_but_generalized_k_is_exact_on_stretched_h4():
    scalar_system = _build_system_data(
        **_stretched_h4_kwargs(),
        hamiltonian_cutoff=1e-20,
        subset_rank_max=2,
        k_param_shape="uniform_ov_same_spin",
    )
    vector_system = _build_system_data(
        **_stretched_h4_kwargs(),
        hamiltonian_cutoff=1e-20,
        subset_rank_max=2,
        k_param_shape="generalized_same_spin_vector",
    )
    scalar_params = _resolve_initial_k_params(
        scalar_system,
        kappa=np.pi / 4.0,
        k_params=None,
        np=np,
    )
    vector_params = _resolve_initial_k_params(
        vector_system,
        kappa=np.pi / 4.0,
        k_params=None,
        np=np,
    )
    scalar_record = _evaluate_subspace(
        scalar_system,
        num_basis_states=len(scalar_system.subset_labels),
        k_params=scalar_params,
        overlap_tol=1e-10,
        regularization=1e-10,
        include_iteration_matrices=False,
    )
    vector_record = _evaluate_subspace(
        vector_system,
        num_basis_states=len(vector_system.subset_labels),
        k_params=vector_params,
        overlap_tol=1e-10,
        regularization=1e-10,
        include_iteration_matrices=False,
    )

    assert float(scalar_record["abs_error_fci"]) > 1e-4
    assert float(vector_record["abs_error_fci"]) < 1e-10


def test_frozen_k_uses_one_constant_vector_on_stretched_h4():
    _k_history, _subset_labels, _energies, details = _cached_h4_gucj_result("frozen_K")

    frozen = np.asarray(details["frozen_k_params"], dtype=float)
    assert frozen.size > 1
    assert details["k_source"] == "heuristic"
    for record in details["iteration_records"]:
        np.testing.assert_allclose(np.asarray(record["k_params"], dtype=float), frozen, atol=1e-10, rtol=0.0)


def test_default_frozen_k_reaches_numerical_accuracy_on_stretched_h4():
    _k_history, _subset_labels, _energies, details = _cached_h4_gucj_result("frozen_K")

    assert details["variant_status"] == "default"
    assert float(details["iteration_records"][-1]["abs_error_fci"]) < 1e-8


@pytest.mark.skipif(
    not _RUN_LEGACY_TESTS,
    reason="Legacy algorithms 2 and 3 are excluded from the default regression suite.",
)
@pytest.mark.parametrize(
    "method_variant",
    ["optimized_K_outer_loop", "alternating_K_and_subspace"],
)
def test_legacy_variants_reach_numerical_accuracy_on_stretched_h4(method_variant: str):
    _k_history, _subset_labels, _energies, details = _cached_h4_gucj_result(method_variant)

    assert details["variant_status"] == "legacy"
    assert float(details["iteration_records"][-1]["abs_error_fci"]) < 1e-8


def test_ccsd_t1_frozen_k_runs_on_stretched_h4_and_keeps_finite_error():
    _k_history, _subset_labels, energies, details = _cached_h4_gucj_result(
        "frozen_K",
        k_param_shape="ov_same_spin_vector",
        k_source="ccsd_t1",
    )

    assert np.isfinite(float(energies[-1]))
    assert np.isfinite(float(details["iteration_records"][-1]["abs_error_fci"]))
    assert details["k_source"] == "ccsd_t1"
    assert details["selection_mode"] == "adaptive_commutator"


def test_plot_script_renders_from_saved_csv(tmp_path: Path):
    script_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "plot_gucj_gcim_stretched_h2_h4.py"
    )
    spec = importlib.util.spec_from_file_location("plot_gucj_gcim_stretched_h2_h4", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)

    rows = [
        {
            "molecule": "H2",
            "n_atoms": 2,
            "geometry_parameter_angstrom": 1.5,
            "method_variant": "frozen_K",
            "iteration": 0,
            "basis_dimension_before_screening": 1,
            "basis_dimension_after_zero_screening": 1,
            "basis_dimension_after_screening": 1,
            "kappa": float(np.pi / 4.0),
            "ritz_energy_hartree": -1.0,
            "fci_energy_hartree": -1.1,
            "abs_error_hartree": 1e-2,
            "overlap_lambda_min_raw": 1.0,
            "overlap_lambda_min_retained": 1.0,
            "overlap_lambda_max_retained": 1.0,
            "overlap_condition_number": 1.0,
            "added_subset_label": "()",
        },
        {
            "molecule": "H4",
            "n_atoms": 4,
            "geometry_parameter_angstrom": 1.5,
            "method_variant": "frozen_K",
            "iteration": 0,
            "basis_dimension_before_screening": 1,
            "basis_dimension_after_zero_screening": 1,
            "basis_dimension_after_screening": 1,
            "kappa": float(np.pi / 4.0),
            "ritz_energy_hartree": -2.0,
            "fci_energy_hartree": -2.2,
            "abs_error_hartree": 2e-2,
            "overlap_lambda_min_raw": 1.0,
            "overlap_lambda_min_retained": 1.0,
            "overlap_lambda_max_retained": 1.0,
            "overlap_condition_number": 1.0,
            "added_subset_label": "()",
        },
    ]

    csv_path = tmp_path / "synthetic.csv"
    png_path = tmp_path / "synthetic.png"
    pdf_path = tmp_path / "synthetic.pdf"

    module._write_csv(rows, csv_path)
    parsed_rows = module._read_csv(csv_path)
    module._make_plot(parsed_rows, png_path, pdf_path)

    assert csv_path.exists()
    assert png_path.exists()
    assert pdf_path.exists()
