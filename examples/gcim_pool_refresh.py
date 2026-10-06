"""Reproduce retained-subspace U2 SD passes from supplied sector matrices.

No molecular rebuild or orbital phase change is performed. The input states
archive contains indices, hf, exact and occupations in the supplied basis.
Run with --policy cycle or --policy repeat_after_first; both use one continuous
GCIM call for 270 selections. Output is a new directory, never overwritten.
"""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pennylane as qml
from scipy import sparse, linalg
from QCANT.gcim import gcim_from_matrices
from QCANT.gcim._singlet_sd import build_singlet_sd_pool


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hamiltonian", type=Path, required=True)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--spin", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", choices=("cycle", "repeat_after_first"), required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.time()
    h = sparse.load_npz(args.hamiltonian)
    s2 = sparse.load_npz(args.spin)
    with np.load(args.states, allow_pickle=False) as z:
        indices, hf, exact, occupations = (z[k] for k in ("indices", "hf", "exact", "occupations"))
    assert np.max(np.abs(h.data.imag)) < 1e-14
    h = h.real
    labels, pool = build_singlet_sd_pool((3, 3), 6)
    assert len(pool) == 54
    generators = [qml.jordan_wigner(g).sparse_matrix(wire_order=range(12)).tocsr()[indices][:, indices].real
                  for g in pool]
    assert all(sparse.linalg.norm(g @ s2 - s2 @ g) < 1e-10 for g in generators)
    target = float(np.vdot(exact, h @ exact).real)
    assert np.linalg.norm(h @ exact - target * exact) < 1e-8
    rows = []
    def progress(row, state):
        spin = float(np.vdot(state, s2 @ state).real)
        error = row["energy"] - target
        assert error > -1e-8 and abs(spin) < 1e-7
        assert row["basis_size"] == 2 * row["K"]
        rows.append(dict(row, error_mEh=1000*error, spin_squared=spin))
    result = gcim_from_matrices(h, generators, hf, policy=args.policy,
                               max_iterations=270, callback=progress)
    # Independent short calls verify the precise pool-refresh boundary.
    first = gcim_from_matrices(h, generators, hf, max_iterations=54)
    boundary = gcim_from_matrices(h, generators, hf, policy=args.policy, max_iterations=55)
    np.testing.assert_array_equal(boundary["final_basis"][:, :108], first["final_basis"])
    np.testing.assert_array_equal(result["final_basis"][:, :110], boundary["final_basis"])
    np.testing.assert_array_equal(result["final_basis"][:, 0], hf)
    assert result["selected_operator_indices"][:54] == first["selected_operator_indices"]
    assert result["selected_operator_indices"][:55] == boundary["selected_operator_indices"]
    assert result["reference_restarts"] == 0 and result["final_basis"].shape[1] == 540
    if args.policy == "cycle":
        assert all(len(set(result["selected_operator_indices"][k:k+54])) == 54
                   for k in range(0, 270, 54))
    # Independent projected diagonalization in the retained span.
    u, sigma, _ = linalg.svd(result["final_basis"], full_matrices=False, lapack_driver="gesvd")
    q, _ = linalg.qr(u[:, sigma**2 > 1e-13], mode="economic")
    hp = q.conj().T @ (h @ q)
    check = float(linalg.eigh((hp + hp.conj().T)/2, eigvals_only=True)[0])
    assert abs(check-rows[-1]["energy"]) < 1e-9
    occupied = occupations[np.argmax(abs(hf))].astype(bool)
    ranks = occupations[:, ~occupied].sum(axis=1).astype(int)
    def probs(state):
        return np.bincount(ranks, weights=abs(state)**2, minlength=7).tolist()
    summary = dict(status="complete", policy=args.policy, pool="SD", pool_size=54,
                   R=2.6, theta=float(np.pi/4), overlap_tol=1e-13, diagonal_shift=0,
                   reference_restarts=0, retains_accumulated_basis=True,
                   manual_phase_changes=False, exact_shifted=target,
                   first_crossing=next((r["K"] for r in rows if r["error_mEh"] <= 1.5936), None),
                   endpoint_error_mEh=rows[-1]["error_mEh"],
                   maximum_rayleigh_gap=max(r["rayleigh_gap"] for r in rows),
                   exact_rank_probabilities=probs(exact),
                   checkpoints={k: probs(v) for k, v in result["checkpoints"].items()},
                   input_sha256=hashlib.sha256(args.hamiltonian.read_bytes()).hexdigest(),
                   states_sha256=hashlib.sha256(args.states.read_bytes()).hexdigest(),
                   elapsed_seconds=time.time()-start,
                   boundary_checks={"K54_columns": 108, "K55_columns": 110,
                                    "K270_columns": 540, "old_columns_identical": True,
                                    "first_pass_identical": True, "original_reference_retained": True})
    for name, obj in (("complete.json", summary),
                      ("result.json", dict(summary=summary, trajectory=rows, labels=labels))):
        (args.output/name).write_text(json.dumps(obj, indent=2, allow_nan=False)+"\n")
    np.savez_compressed(args.output/"final_states.npz", **{"K"+str(k): v for k, v in result["checkpoints"].items()},
                        final_basis=result["final_basis"], hf=hf, exact=exact)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
