"""Exact GCIM for an already supplied Hamiltonian and orthonormal sector basis.

Same product-state ranking and individual/product subspace as the molecular
API; avoids rebuilding deposited Hamiltonians or materializing full Fock
matrices. No orbital rotation, phase change, fitted shift, or gate splitting.
"""
import numpy as np
from scipy import linalg, sparse
from scipy.sparse.linalg import expm_multiply
from ._selection import POLICIES, eligible_indices


def solve_ritz(hamiltonian, columns, overlap_tol=1e-13):
    """Canonical overlap truncation via SVD: keep sigma**2 > overlap_tol.

SVD avoids squaring the condition number before identifying the physical
directions. A scalar energy origin is removed and restored analytically.
"""
    v = np.asarray(columns)
    if not np.all(np.isfinite(v)):
        raise ValueError("Non-finite GCIM basis")
    try:
        u, singular, _ = linalg.svd(v, full_matrices=False, check_finite=False)
    except linalg.LinAlgError:
        # LAPACK divide-and-conquer can fail on highly redundant complex bases.
        # QR-iteration SVD computes the same spectrum, without changing cutoff.
        u, singular, _ = linalg.svd(v, full_matrices=False, check_finite=False,
                                   lapack_driver="gesvd")
    keep = singular**2 > overlap_tol
    if not np.any(keep):
        raise ValueError("No overlap directions retained")
    q = u[:, keep]
    origin = float(np.vdot(v[:, 0], hamiltonian @ v[:, 0]).real /
                   np.vdot(v[:, 0], v[:, 0]).real)
    projected = q.conj().T @ (hamiltonian @ q - origin*q)
    values, coeff = linalg.eigh((projected+projected.conj().T)/2,
                               subset_by_index=(0, 0), check_finite=False)
    state = q @ coeff[:, 0]
    state /= np.linalg.norm(state)
    energy = float(values[0]+origin)
    physical = float(np.vdot(state, hamiltonian @ state).real)
    return dict(energy=energy, state=state, physical_energy=physical,
                rank=int(keep.sum()), min_overlap=float((singular[keep]**2).min()),
                rayleigh_gap=abs(energy-physical))


def gcim_from_matrices(hamiltonian, generators, reference, *,
                       theta=np.pi/4, overlap_tol=1e-13, max_iterations=None,
                       policy="unique", record_history=False, callback=None):
    """Run GCIM on fixed input matrices, returning compact trajectory and states.

Policies: unique (published first pass); cycle (refresh eligibility every pool,
retain the reference and subspace); repeat_after_first (unique first pool,
then arbitrary reuse); unrestricted (reuse from step one). The latter three
are extensions, not claimed to be in the Zheng et al. paper.
Full-pool/explicit iteration endpoints deliberately disable early stopping.
"""
    h = sparse.csr_matrix(hamiltonian)
    a = [sparse.csr_matrix(g) for g in generators]
    ref = np.asarray(reference).reshape(-1).copy()
    if not np.isfinite(theta) or theta <= 0 or overlap_tol <= 0:
        raise ValueError("theta and overlap_tol must be positive and finite")
    if not a or h.shape != (len(ref), len(ref)):
        raise ValueError("Incompatible or empty input")
    if abs(np.linalg.norm(ref)-1) > 1e-10:
        raise ValueError("Reference must have unit norm")
    if sparse.linalg.norm(h-h.conj().T) > 1e-9:
        raise ValueError("Hamiltonian must be Hermitian")
    for generator in a:
        if generator.shape != h.shape or sparse.linalg.norm(generator+generator.conj().T) > 1e-10:
            raise ValueError("Generators must be anti-Hermitian with matching shape")
    if policy not in POLICIES:
        raise ValueError("Unknown repetition policy")
    count = len(a) if max_iterations is None else int(max_iterations)
    if policy == "unique":
        count = min(count, len(a))
    if count < 1:
        raise ValueError("max_iterations must be positive")
    stack = sparse.vstack(a, format="csr")
    product = ref.copy()
    basis, used, selections, rows, history = [ref.copy()], set(), [], [], []
    checkpoints = {}
    for k in range(count):
        # Only eligibility changes: ref, product and every old column persist.
        candidates = eligible_indices(len(a), k, used, policy)
        acts = (stack @ product).reshape(len(a), len(ref))
        # d/dt <exp(tA)psi|H|exp(tA)psi> at t=0. Product state, not Ritz.
        scores = np.abs(2*np.real(acts.conj() @ (h @ product)))
        selected = candidates[int(np.argmax(scores[candidates]))]
        used.add(selected)
        selections.append(selected)
        product = expm_multiply(theta*a[selected], product, traceA=0)
        product /= np.linalg.norm(product)
        if k:
            individual = expm_multiply(theta*a[selected], ref, traceA=0)
            basis.append(individual/np.linalg.norm(individual))
        basis.append(product.copy())
        columns = np.column_stack(basis)
        result = solve_ritz(h, columns, overlap_tol)
        if result["rayleigh_gap"] > 1e-8:
            raise ArithmeticError("Ritz eigenvalue and normalized physical energy disagree")
        row = dict(K=k+1, operator=selected, gradient=float(scores[selected]),
                   energy=result["energy"], rank=result["rank"],
                   basis_size=len(basis), min_overlap=result["min_overlap"],
                   rayleigh_gap=result["rayleigh_gap"])
        rows.append(row)
        if (k+1) % len(a) == 0 or k+1 == count:
            checkpoints[k+1] = result["state"].copy()
        if record_history:
            history.append(columns.copy())
        if callback:
            callback(row, result["state"])
    return dict(trajectory=rows, selected_operator_indices=selections,
                final_ritz_state=result["state"], final_product_state=product,
                final_basis=columns, checkpoints=checkpoints, basis_history=history,
                theta=float(theta), overlap_tol=float(overlap_tol), policy=policy,
                reference_restarts=0, retains_accumulated_basis=True)
