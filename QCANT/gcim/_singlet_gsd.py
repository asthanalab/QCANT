"""Spin-scalar GSD algebra matching the operator pool released with QuGCM.

Reference: Zheng et al., npj Quantum Information 10, 127 (2024);
Firepanda415/adapt-vqe-for-gcim, operator_pools.singlet_GSD.
Signs, order, and normal-ordered coefficient normalization are retained.
"""
from math import sqrt


def build_singlet_gsd_pool(n_spatial):
    from openfermion import FermionOperator as F, hermitian_conjugated, normal_ordered
    from pennylane.fermi import FermiSentence, FermiWord
    labels, pool = [], []

    def append(label, term):
        term = normal_ordered(term - hermitian_conjugated(term))
        term.compress(abs_tol=1e-14)
        norm = sqrt(sum(abs(c)**2 for c in term.terms.values()))
        if norm < 1e-14:
            return
        labels.append(label)
        pool.append(FermiSentence({
            FermiWord({(j, mode): "+" if action else "-"
                       for j, (mode, action) in enumerate(word)}): c/norm
            for word, c in term.terms.items()}))

    for p in range(n_spatial):
        for q in range(p, n_spatial):
            append([p, q], F(((2*p, 1), (2*q, 0))) +
                   F(((2*p+1, 1), (2*q+1, 0))))
    pairs = [(p, q) for p in range(n_spatial) for q in range(p, n_spatial)]
    for i, (p, q) in enumerate(pairs):
        pa, pb, qa, qb = 2*p, 2*p+1, 2*q, 2*q+1
        for r, s in pairs[i:]:
            ra, rb, sa, sb = 2*r, 2*r+1, 2*s, 2*s+1
            a = F(((ra, 1), (pa, 0), (sa, 1), (qa, 0)), 2/sqrt(12))
            a += F(((rb, 1), (pb, 0), (sb, 1), (qb, 0)), 2/sqrt(12))
            a += F(((ra, 1), (pa, 0), (sb, 1), (qb, 0)), 1/sqrt(12))
            a += F(((rb, 1), (pb, 0), (sa, 1), (qa, 0)), 1/sqrt(12))
            a += F(((ra, 1), (pb, 0), (sb, 1), (qa, 0)), 1/sqrt(12))
            a += F(((rb, 1), (pa, 0), (sa, 1), (qb, 0)), 1/sqrt(12))
            b = F(((ra, 1), (pa, 0), (sb, 1), (qb, 0)), .5)
            b += F(((rb, 1), (pb, 0), (sa, 1), (qa, 0)), .5)
            b -= F(((ra, 1), (pb, 0), (sb, 1), (qa, 0)), .5)
            b -= F(((rb, 1), (pa, 0), (sa, 1), (qb, 0)), .5)
            append([p, q, r, s, 1], a)
            append([p, q, r, s, 2], b)
    return labels, pool
