"""Complete reference-relative, spin-scalar singles and doubles for GCIM."""

from math import sqrt
from numbers import Integral


def build_singlet_sd_pool(nelecas, n_spatial):
    """Return labels and PennyLane FermiSentences for a closed-shell SD pool.

    Spatial orbitals are occupied first; spin orbitals are alpha/beta
    interleaved. Labels are spatial ``[i, a]`` for singles and
    ``[i, j, a, b, pair_spin]`` for doubles. Both double channels are spin
    scalars: pair_spin=1 contracts triplet pairs to a scalar and does not
    change the total spin of the reference.

    Each anti-Hermitian generator has unit Euclidean norm of its
    normal-ordered fermionic coefficients. The two double channels use the
    construction and ordering of the original SingletSD GCIM scripts:
    triplet-pair contraction first, then singlet-pair transfer, omitting
    identically zero channels. No orbital-phase transformation is applied.
    """
    if len(nelecas) != 2 or any(not isinstance(n, Integral) for n in nelecas):
        raise ValueError("nelecas must contain integer alpha and beta electron counts")
    n_alpha, n_beta = nelecas
    if n_alpha != n_beta:
        raise ValueError("pool_type='singlet_sd' requires a closed-shell active space")
    if not isinstance(n_spatial, Integral) or not 0 <= n_alpha <= n_spatial:
        raise ValueError("n_spatial must be an integer >= the occupied orbital count")

    from pennylane.fermi import FermiSentence, FermiWord
    from openfermion import FermionOperator, hermitian_conjugated, normal_ordered

    labels, pool = [], []

    def append_generator(label, excitation):
        generator = normal_ordered(excitation - hermitian_conjugated(excitation))
        generator.compress(abs_tol=1e-14)
        norm = sqrt(sum(abs(c) ** 2 for c in generator.terms.values()))
        if norm <= 1e-14:
            return
        labels.append(label)
        pool.append(FermiSentence({
            FermiWord({(position, mode): "+" if action else "-"
                           for position, (mode, action) in enumerate(term)}): c / norm
            for term, c in generator.terms.items()
        }))

    occ = range(n_alpha)
    virt = range(n_alpha, n_spatial)
    for i in occ:
        for a in virt:
            single = FermionOperator(((2*a, 1), (2*i, 0)), 1 / sqrt(2))
            single += FermionOperator(((2*a+1, 1), (2*i+1, 0)), 1 / sqrt(2))
            append_generator([i, a], single)

    for i in occ:
        ia, ib = 2*i, 2*i+1
        for j in range(i, n_alpha):
            ja, jb = 2*j, 2*j+1
            for a in virt:
                aa, ab = 2*a, 2*a+1
                for b in range(a, n_spatial):
                    ba, bb = 2*b, 2*b+1

                    # Contract all three triplet components (vanishes if i=j or a=b).
                    triplet = FermionOperator(((aa, 1), (ba, 1), (ia, 0), (ja, 0)), 2 / sqrt(12))
                    triplet += FermionOperator(((ab, 1), (bb, 1), (ib, 0), (jb, 0)), 2 / sqrt(12))
                    triplet += FermionOperator(((aa, 1), (bb, 1), (ia, 0), (jb, 0)), 1 / sqrt(12))
                    triplet += FermionOperator(((ab, 1), (ba, 1), (ib, 0), (ja, 0)), 1 / sqrt(12))
                    triplet += FermionOperator(((aa, 1), (bb, 1), (ib, 0), (ja, 0)), 1 / sqrt(12))
                    triplet += FermionOperator(((ab, 1), (ba, 1), (ia, 0), (jb, 0)), 1 / sqrt(12))
                    append_generator([i, j, a, b, 1], triplet)

                    # Singlet pairs include both paired and non-paired excitations.
                    singlet = FermionOperator(((aa, 1), (bb, 1), (ia, 0), (jb, 0)), 0.5)
                    singlet += FermionOperator(((ab, 1), (ba, 1), (ib, 0), (ja, 0)), 0.5)
                    singlet -= FermionOperator(((aa, 1), (bb, 1), (ib, 0), (ja, 0)), 0.5)
                    singlet -= FermionOperator(((ab, 1), (ba, 1), (ia, 0), (jb, 0)), 0.5)
                    append_generator([i, j, a, b, 0], singlet)

    return labels, pool
