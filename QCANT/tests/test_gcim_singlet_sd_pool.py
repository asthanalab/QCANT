"""Algebraic and end-to-end regressions for the complete GCIM SingletSD pool."""

from math import comb, sqrt

import numpy as np
import pytest
from openfermion import (
    FermionOperator, commutator, hermitian_conjugated, normal_ordered,
    number_operator, s_squared_operator, sz_operator,
)

from QCANT.gcim import gcim
from QCANT.gcim._singlet_sd import build_singlet_sd_pool


def as_openfermion(sentence):
    """Independent conversion for symbolic checks; no dense Fock matrices."""
    result = FermionOperator()
    for word, coefficient in sentence.items():
        term = tuple((mode, int(action == "+"))
                     for (_, mode), action in sorted(word.items()))
        result += FermionOperator(term, coefficient)
    return normal_ordered(result)


def assert_zero(operator):
    operator = normal_ordered(operator)
    assert max((abs(c) for c in operator.terms.values()), default=0.0) < 1e-12


def pair_creation(p, q, spin):
    """Independent Clebsch-Gordan pair tensors (not expanded SD formulas)."""
    def create(left, right):
        return FermionOperator(((left, 1), (right, 1)))

    alpha_beta = create(2*p, 2*q+1)
    beta_alpha = create(2*p+1, 2*q)
    if spin == 0:
        return [(alpha_beta - beta_alpha) / sqrt(2)]
    return [create(2*p, 2*q), (alpha_beta + beta_alpha) / sqrt(2),
            create(2*p+1, 2*q+1)]


@pytest.mark.parametrize("electrons,orbitals,expected", [
    (2, 2, 2), (4, 4, 14), (6, 6, 54), (8, 6, 44), (10, 8, 135),
])
def test_complete_pool_counts(electrons, orbitals, expected):
    occupied, virtual = electrons // 2, orbitals - electrons // 2
    labels, pool = build_singlet_sd_pool((occupied, occupied), orbitals)
    assert len(pool) == len(labels) == expected
    assert sum(len(label) == 2 for label in labels) == occupied * virtual
    assert sum(len(label) == 5 and label[-1] == 0 for label in labels) == (
        comb(occupied + 1, 2) * comb(virtual + 1, 2))
    assert sum(len(label) == 5 and label[-1] == 1 for label in labels) == (
        comb(occupied, 2) * comb(virtual, 2))
    assert len({tuple(label) for label in labels}) == expected


@pytest.mark.parametrize("occupied,orbitals", [(2, 4), (3, 6), (4, 6)])
def test_spin_particle_symmetry_normalization_and_independence(occupied, orbitals):
    _, pool = build_singlet_sd_pool((occupied, occupied), orbitals)
    operators = [as_openfermion(op) for op in pool]
    symmetries = [number_operator(2 * orbitals), sz_operator(orbitals),
                  s_squared_operator(orbitals)]
    for generator in operators:
        assert_zero(generator + hermitian_conjugated(generator))
        assert sum(abs(c)**2 for c in generator.terms.values()) == pytest.approx(1.0)
        for symmetry in symmetries:
            assert_zero(commutator(generator, symmetry))
    terms = sorted({term for op in operators for term in op.terms})
    coefficients = np.array([[op.terms.get(term, 0) for term in terms] for op in operators])
    assert np.linalg.matrix_rank(coefficients, tol=1e-12) == len(pool)


def test_doubles_match_independent_spin_pair_contraction():
    labels, pool = build_singlet_sd_pool((3, 3), 6)
    for label, sentence in zip(labels, pool):
        if len(label) == 2:
            i, a = label
            expected = (FermionOperator(((2*a, 1), (2*i, 0)))
                        + FermionOperator(((2*a+1, 1), (2*i+1, 0))))
        else:
            i, j, a, b, spin = label
            expected = FermionOperator()
            for source, target in zip(pair_creation(i, j, spin), pair_creation(a, b, spin)):
                # The original GCIM convention annihilates i before j.
                expected -= target * hermitian_conjugated(source)
        expected = normal_ordered(expected - hermitian_conjugated(expected))
        expected /= sqrt(sum(abs(c)**2 for c in expected.terms.values()))
        assert_zero(as_openfermion(sentence) - expected)


@pytest.mark.parametrize("electrons,orbitals", [((3, 2), 6), ((-1, -1), 6),
                                                   ((4, 4), 3), ((1.5, 1.5), 3)])
def test_invalid_occupations_rejected(electrons, orbitals):
    with pytest.raises(ValueError):
        build_singlet_sd_pool(electrons, orbitals)


def test_public_gcim_dispatches_to_full_pool(monkeypatch):
    """Exercise the real molecular path, but stop before expensive 12-qubit gates."""
    import pennylane as qml

    class PoolBuilt(Exception):
        pass

    seen = []
    real_jordan_wigner = qml.jordan_wigner

    def record_conversion(operator, *args, **kwargs):
        seen.append(operator)
        return real_jordan_wigner(operator, *args, **kwargs)

    def stop_at_device(*args, **kwargs):
        # One Hamiltonian conversion, then every member of the pool.
        assert len(seen) == 1 + 54
        raise PoolBuilt

    monkeypatch.setattr(qml, "jordan_wigner", record_conversion)
    monkeypatch.setattr(qml, "device", stop_at_device)
    with pytest.raises(PoolBuilt):
        gcim(["H"] * 6, np.array([[0, 0, 1.4*k] for k in range(6)]),
             adapt_it=1, active_electrons=6, active_orbitals=6, basis="sto-3g",
             pool_type="singlet_sd", print_progress=False, max_iterations=1)


def test_singlet_h2_reaches_casci_and_reports_pool():
    from pyscf import fci, gto, scf

    geometry = np.array([[0, 0, 0], [0, 0, 1.5]])
    mol = gto.M(atom=[("H", tuple(row)) for row in geometry], basis="sto-3g",
                unit="Angstrom", verbose=0)
    mean_field = scf.RHF(mol).run()
    exact = fci.FCI(mean_field).kernel()[0]
    _, labels, energies, details = gcim(
        ["H", "H"], geometry, adapt_it=1, active_electrons=2,
        active_orbitals=2, basis="sto-3g", pool_type="singlet_sd",
        device_name="default.qubit", regularization=0, print_progress=False,
        return_details=True)
    assert details["pool_size"] == 2
    assert details["pool_type"] == "singlet_sd"
    assert len(labels) == len(energies) == 2
    assert abs(energies[-1] - exact) < 1e-9
    assert len(set(details["selected_operator_indices"])) == 2
