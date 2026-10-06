"""Noiseless GCIM spectrum, basis construction, and angle regressions."""
import numpy as np
import pytest
from scipy.linalg import expm
from QCANT.gcim import gcim_from_matrices
from QCANT.gcim._matrix import solve_ritz


def test_direct_angle_two_state_growth_and_reuse():
    h = np.diag([0., 1., 2., 3.])
    h[0, 1] = h[1, 0] = -.2
    h[1, 2] = h[2, 1] = .13
    generators = []
    for i, j in [(0,1), (1,2), (0,3)]:
        a = np.zeros((4,4))
        a[i,j], a[j,i] = 1., -1.
        generators.append(a)
    ref = np.eye(4)[0]
    result = gcim_from_matrices(h, generators, ref, record_history=True)
    product = ref.copy()
    for k, idx in enumerate(result["selected_operator_indices"]):
        product = expm(np.pi/4*generators[idx])@product
        basis = result["basis_history"][k]
        assert basis.shape[1] == 2*(k+1)
        assert np.allclose(basis[:,-1], product)
        if k:
            assert np.allclose(basis[:,-2], expm(np.pi/4*generators[idx])@ref)
    for policy in ("cycle", "repeat_after_first", "unrestricted"):
        repeated = gcim_from_matrices(h, generators, ref, policy=policy, max_iterations=8)
        assert len(repeated["trajectory"]) == 8
        if policy != "unrestricted":
            assert repeated["selected_operator_indices"][:3] == result["selected_operator_indices"]


def test_complex_rank_truncation_and_constant_energy_origin():
    rng = np.random.default_rng(54)
    v = rng.normal(size=(8,3))+1j*rng.normal(size=(8,3))
    v /= np.linalg.norm(v, axis=0)
    v = np.column_stack((v, v[:,0]+1e-9*v[:,1]))
    h = rng.normal(size=(8,8))+1j*rng.normal(size=(8,8))
    h += h.conj().T.copy()
    a = solve_ritz(h, v)
    b = solve_ritz(h-55985*np.eye(8), v)
    assert a["rank"] == b["rank"] == 3
    assert abs(a["energy"]-b["energy"]-55985) < 1e-9
    assert a["rayleigh_gap"] < 1e-12
    assert abs(np.linalg.norm(a["state"])-1) < 1e-12


def test_bad_reference_rejected():
    with pytest.raises(ValueError, match="unit norm"):
        gcim_from_matrices(np.eye(2), [np.array([[0.,1.],[-1.,0.]])], [2.,0.])


@pytest.mark.parametrize("policy", ["cycle", "repeat_after_first", "unrestricted"])
def test_pool_reuse_never_restarts_reference_product_or_basis(policy):
    rng = np.random.default_rng(54)
    h = rng.normal(size=(9, 9))
    h = (h + h.T) / 2
    generators = []
    for _ in range(3):
        a = rng.normal(size=(9, 9))
        generators.append((a - a.T) / 4)
    ref = np.eye(9)[0]
    result = gcim_from_matrices(h, generators, ref, policy=policy,
                               max_iterations=8, record_history=True)
    product = ref.copy()
    for k, idx in enumerate(result["selected_operator_indices"]):
        basis = result["basis_history"][k]
        product = expm(np.pi/4 * generators[idx]) @ product
        assert basis.shape == (9, 2*(k+1))
        np.testing.assert_allclose(basis[:, 0], ref, atol=1e-14)
        np.testing.assert_allclose(basis[:, -1], product, atol=1e-13)
        if k:
            old = result["basis_history"][k-1]
            np.testing.assert_array_equal(basis[:, :old.shape[1]], old)
            np.testing.assert_allclose(basis[:, -2],
                                       expm(np.pi/4*generators[idx]) @ ref,
                                       atol=1e-13)
    if policy == "cycle":
        assert len(set(result["selected_operator_indices"][:3])) == 3
        assert len(set(result["selected_operator_indices"][3:6])) == 3
    assert result["reference_restarts"] == 0
    assert result["retains_accumulated_basis"]


@pytest.mark.parametrize("policy", ["cycle", "repeat_after_first"])
def test_molecular_api_preserves_full_product_and_basis(policy):
    import pennylane as qml
    from QCANT.gcim import gcim
    from QCANT.gcim._singlet_sd import build_singlet_sd_pool
    _, _, energy, details = gcim(
        ["H", "H"], [[0, 0, 0], [0, 0, .74]], adapt_it=5,
        basis="sto-3g", active_electrons=2, active_orbitals=2,
        pool_type="singlet_sd", selection_policy=policy, max_iterations=5,
        device_name="default.qubit", return_details=True, print_progress=False)
    _, pool = build_singlet_sd_pool((1, 1), 2)
    matrices = [qml.matrix(qml.jordan_wigner(g), wire_order=range(4)) for g in pool]
    reference = details["hf_state"]
    product = reference.copy()
    assert len(energy) == 5 > len(pool)
    for k, idx in enumerate(details["selected_operator_indices"]):
        basis = details["state_basis_history"][k]
        product = expm(np.pi/4 * matrices[idx]) @ product
        np.testing.assert_allclose(basis[-1], product, atol=1e-12)
        np.testing.assert_array_equal(basis[0], reference)
        assert basis.shape[0] == 2*(k+1)
        if k:
            old = details["state_basis_history"][k-1]
            np.testing.assert_array_equal(basis[:len(old)], old)
    assert details["reference_restarts"] == 0
