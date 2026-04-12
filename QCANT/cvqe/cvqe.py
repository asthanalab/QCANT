"""Simplified cyclic VQE with determinant growth and a compact entangler.

The implementation in this module intentionally keeps the paper idea lightweight:

1. Start from a Hartree-Fock reference determinant.
2. Apply a compact entangler initialized from CCSD ``t1``/``t2``.
3. Select a *new* determinant from the output state, either by highest
   probability or by a simple amplitude-screened second-order energy estimate.
4. Add that determinant to the reference superposition.
5. Re-optimize all active parameters: determinant coefficients and entangler
   parameters.

This keeps the algorithm close to the CVQE feedback loop while remaining easy
to run in small-molecule regression tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional, Sequence


def _validate_inputs(
    symbols: Sequence[str],
    geometry,
    adapt_it: int,
    shots: Optional[int],
    active_electrons: int,
    active_orbitals: int,
    spin: int,
    selection_topk: int,
) -> int:
    """Validate user inputs and return the number of atoms."""
    if len(symbols) == 0:
        raise ValueError("symbols must be non-empty")
    if adapt_it < 0:
        raise ValueError("adapt_it must be >= 0")
    if shots is not None and shots < 0:
        raise ValueError("shots must be >= 0")
    if active_electrons <= 0:
        raise ValueError("active_electrons must be > 0")
    if active_orbitals <= 0:
        raise ValueError("active_orbitals must be > 0")
    if selection_topk <= 0:
        raise ValueError("selection_topk must be > 0")
    if active_electrons > (2 * active_orbitals):
        raise ValueError("active_electrons cannot exceed 2 * active_orbitals")
    if spin != 0:
        raise ValueError("cvqe currently supports only closed-shell systems with spin=0")
    if (active_electrons % 2) != 0:
        raise ValueError("cvqe currently requires an even number of active_electrons")

    try:
        n_atoms = len(symbols)
        if len(geometry) != n_atoms:
            raise ValueError
    except Exception as exc:
        raise ValueError("geometry must have the same length as symbols") from exc

    return n_atoms


def _bits_to_index(bits) -> int:
    """Convert an occupation list into the matching computational-basis index."""
    return int("".join(str(int(bit)) for bit in bits), 2)


def _index_to_bits(index: int, qubits: int) -> list[int]:
    """Convert a computational-basis index into an occupation list."""
    return [int(bit) for bit in format(int(index), f"0{int(qubits)}b")]


def _build_basis_vector(bits, qubits: int, np):
    """Return a computational-basis vector for one determinant."""
    vec = np.zeros(2**int(qubits), dtype=complex)
    vec[_bits_to_index(bits)] = 1.0
    return vec


def _build_reference_state(hf_bits, added_determinants, coeff_params, qubits: int, np):
    """Construct the normalized reference superposition with HF coefficient fixed to one."""
    vec = _build_basis_vector(hf_bits, qubits, np)
    for coeff, determinant in zip(coeff_params, added_determinants):
        vec = vec + float(coeff) * _build_basis_vector(determinant, qubits, np)
    norm = np.linalg.norm(vec)
    if norm < 1e-14:
        raise ValueError("reference superposition became numerically zero")
    return vec / norm


def _pack_params(det_coeffs, orbital_angles, same_spin, opposite_spin, np):
    """Pack parameter blocks into one real vector."""
    return np.concatenate(
        [
            np.asarray(det_coeffs, dtype=float),
            np.asarray(orbital_angles, dtype=float),
            np.asarray(same_spin, dtype=float),
            np.asarray(opposite_spin, dtype=float),
        ]
    )


def _parameter_slices(n_det_params: int, n_orb_pairs: int) -> dict[str, tuple[int, int]]:
    """Return slice bounds for the flattened parameter vector."""
    start = 0
    slices = {}
    slices["determinant_coeffs"] = (start, start + int(n_det_params))
    start += int(n_det_params)
    slices["lucj_orbital"] = (start, start + int(n_orb_pairs))
    start += int(n_orb_pairs)
    slices["lucj_same_spin"] = (start, start + int(n_orb_pairs))
    start += int(n_orb_pairs)
    slices["lucj_opposite_spin"] = (start, start + int(n_orb_pairs))
    return slices


def _unpack_params(params, n_det_params: int, n_orb_pairs: int, np):
    """Unpack the flattened real vector into determinant and entangler blocks."""
    slices = _parameter_slices(n_det_params=n_det_params, n_orb_pairs=n_orb_pairs)
    params = np.asarray(params, dtype=float)
    det_start, det_stop = slices["determinant_coeffs"]
    orb_start, orb_stop = slices["lucj_orbital"]
    same_start, same_stop = slices["lucj_same_spin"]
    opp_start, opp_stop = slices["lucj_opposite_spin"]
    return (
        params[det_start:det_stop],
        params[orb_start:orb_stop],
        params[same_start:same_stop],
        params[opp_start:opp_stop],
    )


def _pack_uccsd_params(det_coeffs, uccsd_weights, np):
    """Pack determinant coefficients and UCCSD amplitudes into one vector."""
    return np.concatenate(
        [
            np.asarray(det_coeffs, dtype=float),
            np.asarray(uccsd_weights, dtype=float),
        ]
    )


def _unpack_uccsd_params(params, n_det_params: int, n_uccsd_params: int, np):
    """Unpack determinant coefficients and UCCSD amplitudes from one vector."""
    params = np.asarray(params, dtype=float)
    det_stop = int(n_det_params)
    uccsd_stop = det_stop + int(n_uccsd_params)
    return params[:det_stop], params[det_stop:uccsd_stop]


def _parameter_slices_uccsd(n_det_params: int, n_uccsd_params: int) -> dict[str, tuple[int, int]]:
    """Return slice bounds for determinant coefficients and UCCSD amplitudes."""
    return {
        "determinant_coeffs": (0, int(n_det_params)),
        "uccsd": (int(n_det_params), int(n_det_params) + int(n_uccsd_params)),
    }


def _extract_active_ccsd_amplitudes(mycc, active_electrons: int, active_orbitals: int, np):
    """Slice RHF-CCSD amplitudes into the same occupied/virtual window as the active space."""
    active_occ = int(active_electrons // 2)
    active_virt = int(active_orbitals - active_occ)
    if active_virt < 0:
        raise ValueError("active_orbitals must be at least active_electrons // 2")
    nocc_total = int(mycc.nocc)
    occ_start = max(0, nocc_total - active_occ)
    occ_stop = nocc_total
    virt_stop = active_virt

    t1 = np.asarray(mycc.t1[occ_start:occ_stop, :virt_stop], dtype=float)
    t2 = np.asarray(
        mycc.t2[occ_start:occ_stop, occ_start:occ_stop, :virt_stop, :virt_stop],
        dtype=float,
    )
    return t1, t2


def _initialize_lucj_params(t1, t2, active_orbitals: int, active_electrons: int, np):
    """Build a compact local-UCJ parameter guess from CCSD amplitudes.

    The mapping here is intentionally simple:

    - ``t1`` initializes nearest-neighbour orbital rotations.
    - ``t2`` is aggregated into a symmetric occupied/virtual pair-coupling matrix,
      then projected onto nearest-neighbour same-spin and opposite-spin phases.
    """
    norb = int(active_orbitals)
    n_pairs = max(0, norb - 1)
    active_occ = int(active_electrons // 2)

    orbital_angles = np.zeros(n_pairs, dtype=float)
    same_spin = np.zeros(n_pairs, dtype=float)
    opposite_spin = np.zeros(n_pairs, dtype=float)

    if n_pairs == 0:
        return orbital_angles, same_spin, opposite_spin

    coupling = np.zeros((norb, norb), dtype=float)
    for i in range(t1.shape[0]):
        for a in range(t1.shape[1]):
            p = int(i)
            q = int(active_occ + a)
            value = float(np.real(t1[i, a]))
            coupling[p, q] += value
            coupling[q, p] -= value

    pair_strength = np.zeros((norb, norb), dtype=float)
    for i in range(t2.shape[0]):
        for j in range(t2.shape[1]):
            for a in range(t2.shape[2]):
                for b in range(t2.shape[3]):
                    value = float(np.real(t2[i, j, a, b]))
                    p_i = int(i)
                    p_j = int(j)
                    q_a = int(active_occ + a)
                    q_b = int(active_occ + b)
                    for p, q in (
                        (p_i, q_a),
                        (p_i, q_b),
                        (p_j, q_a),
                        (p_j, q_b),
                    ):
                        pair_strength[p, q] += 0.25 * value
                        pair_strength[q, p] += 0.25 * value

    for p in range(n_pairs):
        orbital_angles[p] = float(coupling[p + 1, p] + pair_strength[p, p + 1])
        opposite_spin[p] = float(pair_strength[p, p + 1])
        same_spin[p] = 0.5 * float(pair_strength[p, p + 1])

    return orbital_angles, same_spin, opposite_spin


def _build_uccsd_excitation_metadata(active_electrons: int, qubits: int, qml):
    """Return excitation metadata for applying PennyLane UCCSD-style gates."""
    singles, doubles = qml.qchem.excitations(int(active_electrons), int(qubits))
    metadata = []
    for excitation in singles + doubles:
        exc = [int(idx) for idx in excitation]
        if len(exc) == 2:
            wires = list(range(exc[0], exc[1] + 1))
            metadata.append(
                {
                    "kind": "single",
                    "excitation": exc,
                    "target_wires": tuple(wires),
                    "wires": wires,
                }
            )
        else:
            wires1 = list(range(exc[0], exc[1] + 1))
            wires2 = list(range(exc[2], exc[3] + 1))
            metadata.append(
                {
                    "kind": "double",
                    "excitation": exc,
                    "target_wires": tuple(wires1 + wires2),
                    "wires1": wires1,
                    "wires2": wires2,
                }
            )
    return metadata


def _initialize_uccsd_params(t1, t2, excitation_metadata, active_electrons: int, np):
    """Map active-space CCSD amplitudes onto a UCCSD parameter seed."""
    active_occ = int(active_electrons // 2)
    weights = np.zeros(len(excitation_metadata), dtype=float)

    for idx, meta in enumerate(excitation_metadata):
        excitation = meta["excitation"]
        if meta["kind"] == "single":
            p, q = (int(excitation[0]), int(excitation[1]))
            occ_idx = int(p // 2)
            virt_idx = int(q // 2) - active_occ
            if 0 <= occ_idx < t1.shape[0] and 0 <= virt_idx < t1.shape[1]:
                weights[idx] = float(np.real(t1[occ_idx, virt_idx]))
            continue

        p, q, r, s = (int(excitation[0]), int(excitation[1]), int(excitation[2]), int(excitation[3]))
        occ_i = int(p // 2)
        occ_j = int(q // 2)
        virt_a = int(r // 2) - active_occ
        virt_b = int(s // 2) - active_occ

        candidates = []
        for i_idx, j_idx in ((occ_i, occ_j), (occ_j, occ_i)):
            for a_idx, b_idx in ((virt_a, virt_b), (virt_b, virt_a)):
                if (
                    0 <= i_idx < t2.shape[0]
                    and 0 <= j_idx < t2.shape[1]
                    and 0 <= a_idx < t2.shape[2]
                    and 0 <= b_idx < t2.shape[3]
                ):
                    candidates.append(float(np.real(t2[i_idx, j_idx, a_idx, b_idx])))
        if candidates:
            weights[idx] = max(candidates, key=abs)

    return weights


def _apply_pair_diagonal_phase(
    vec,
    occupancy_table,
    alpha_p: int,
    beta_p: int,
    alpha_q: int,
    beta_q: int,
    gamma_same: float,
    gamma_opposite: float,
    np,
):
    """Apply one local spin-balanced diagonal Jastrow factor in-place."""
    if abs(float(gamma_same)) < 1e-15 and abs(float(gamma_opposite)) < 1e-15:
        return vec

    phase_argument = (
        float(gamma_same)
        * (
            occupancy_table[:, alpha_p] * occupancy_table[:, alpha_q]
            + occupancy_table[:, beta_p] * occupancy_table[:, beta_q]
        )
        + float(gamma_opposite)
        * (
            occupancy_table[:, alpha_p] * occupancy_table[:, beta_q]
            + occupancy_table[:, beta_p] * occupancy_table[:, alpha_q]
        )
    )
    return np.exp(-1j * phase_argument) * vec


def _apply_local_gate(vec, gate_matrix, target_wires, *, qubits: int, np):
    """Apply a small gate matrix to a subset of wires without building a full matrix."""
    targets = tuple(int(wire) for wire in target_wires)
    n_targets = len(targets)
    state_tensor = np.asarray(vec, dtype=complex).reshape((2,) * int(qubits))
    remaining_wires = tuple(wire for wire in range(int(qubits)) if wire not in targets)
    permutation = targets + remaining_wires
    inverse_permutation = np.argsort(permutation)

    permuted = np.transpose(state_tensor, permutation)
    flat_state = permuted.reshape(2**n_targets, -1)
    updated = np.asarray(gate_matrix, dtype=complex) @ flat_state
    restored = updated.reshape((2,) * int(qubits))
    return np.transpose(restored, inverse_permutation).reshape(-1)


def _apply_lucj(
    vec,
    orbital_angles,
    same_spin,
    opposite_spin,
    *,
    qubits: int,
    occupancy_table,
    qml,
    np,
):
    """Apply the fixed one-layer local UCJ/LUCJ-style entangler to a statevector."""
    current = np.asarray(vec, dtype=complex)
    local_wires = (0, 1, 2, 3)
    for pair_idx, angle in enumerate(orbital_angles):
        alpha_p = 2 * int(pair_idx)
        beta_p = alpha_p + 1
        alpha_q = alpha_p + 2
        beta_q = alpha_p + 3
        pair_wires = (alpha_p, beta_p, alpha_q, beta_q)

        if abs(float(angle)) > 1e-15:
            inverse_rotation = np.asarray(
                qml.matrix(qml.OrbitalRotation(-float(angle), wires=local_wires)),
                dtype=complex,
            )
            current = _apply_local_gate(
                current,
                inverse_rotation,
                pair_wires,
                qubits=qubits,
                np=np,
            )

        current = _apply_pair_diagonal_phase(
            current,
            occupancy_table,
            alpha_p=alpha_p,
            beta_p=beta_p,
            alpha_q=alpha_q,
            beta_q=beta_q,
            gamma_same=float(same_spin[pair_idx]),
            gamma_opposite=float(opposite_spin[pair_idx]),
            np=np,
        )

        if abs(float(angle)) > 1e-15:
            forward_rotation = np.asarray(
                qml.matrix(qml.OrbitalRotation(float(angle), wires=local_wires)),
                dtype=complex,
            )
            current = _apply_local_gate(
                current,
                forward_rotation,
                pair_wires,
                qubits=qubits,
                np=np,
            )

    return current


def _apply_uccsd(
    vec,
    weights,
    excitation_metadata,
    *,
    qubits: int,
    qml,
    np,
):
    """Apply a UCCSD-style unitary to an arbitrary reference statevector."""
    current = np.asarray(vec, dtype=complex)
    for weight, meta in zip(weights, excitation_metadata):
        angle = float(weight)
        if abs(angle) < 1e-15:
            continue

        if meta["kind"] == "single":
            op = qml.FermionicSingleExcitation(angle, wires=meta["wires"])
        else:
            op = qml.FermionicDoubleExcitation(
                angle,
                wires1=meta["wires1"],
                wires2=meta["wires2"],
            )

        gate_matrix = np.asarray(qml.matrix(op), dtype=complex)
        current = _apply_local_gate(
            current,
            gate_matrix,
            meta["target_wires"],
            qubits=qubits,
            np=np,
        )

    return current


def _select_new_determinant(
    probabilities,
    selected_indices,
    *,
    candidate_indices,
    shots: Optional[int],
    selection_method: str,
    selection_topk: int,
    current_state,
    current_energy: float,
    h_matrix,
    rng,
    np,
):
    """Choose a new determinant from exact or sampled output statistics."""
    probs = np.asarray(probabilities, dtype=float)
    candidates = np.asarray(candidate_indices, dtype=int)
    if candidates.size == 0:
        return None, 0.0, 0.0, None, {"selection_strategy": str(selection_method)}

    if selection_method == "pt2":
        if shots is not None and int(shots) != 0:
            raise ValueError("selection_method='pt2' currently requires exact selection with shots=0")

        ranking = candidates[np.argsort(-probs[candidates], kind="stable")]
        screened = []
        for idx in ranking:
            idx_int = int(idx)
            if idx_int in selected_indices:
                continue
            screened.append(idx_int)
            if len(screened) >= int(selection_topk):
                break

        if len(screened) == 0:
            return None, 0.0, 0.0, None, {"selection_strategy": "pt2"}

        state = np.asarray(current_state, dtype=complex)
        energy = float(current_energy)
        h_psi = np.asarray(h_matrix @ state, dtype=complex)
        residual = h_psi - energy * state
        pt2_eps = 1e-12

        best_index = None
        best_improvement = -1.0
        best_details = None
        candidate_details = []

        for idx_int in screened:
            numerator = float(np.abs(residual[idx_int]) ** 2)
            denominator = float(np.real(h_matrix[idx_int, idx_int]) - energy)
            denominator_scale = max(abs(denominator), pt2_eps)
            improvement = numerator / denominator_scale
            correction = -improvement
            candidate_details.append(
                {
                    "index": int(idx_int),
                    "probability": float(probs[idx_int]),
                    "amplitude_abs": float(np.abs(state[idx_int])),
                    "pt2_improvement": float(improvement),
                    "pt2_correction": float(correction),
                    "pt2_numerator": float(numerator),
                    "pt2_denominator": float(denominator),
                }
            )
            if improvement > best_improvement:
                best_index = int(idx_int)
                best_improvement = float(improvement)
                best_details = candidate_details[-1]

        return (
            best_index,
            float(best_improvement),
            float(probs[best_index]),
            None,
            {
                "selection_strategy": "pt2",
                "selection_topk": int(selection_topk),
                "screened_candidate_count": int(len(screened)),
                "screened_candidates": candidate_details,
                "selected_pt2_improvement": float(best_details["pt2_improvement"]),
                "selected_pt2_correction": float(best_details["pt2_correction"]),
                "selected_pt2_numerator": float(best_details["pt2_numerator"]),
                "selected_pt2_denominator": float(best_details["pt2_denominator"]),
            },
        )

    if shots is None or int(shots) == 0:
        ranking = candidates[np.argsort(-probs[candidates], kind="stable")]
        for idx in ranking:
            idx_int = int(idx)
            if idx_int in selected_indices:
                continue
            return (
                idx_int,
                float(probs[idx_int]),
                float(probs[idx_int]),
                None,
                {"selection_strategy": "probability", "selection_topk": 1},
            )
        return None, 0.0, 0.0, None, {"selection_strategy": "probability", "selection_topk": 1}

    counts = np.bincount(
        rng.choice(len(probs), size=int(shots), p=probs),
        minlength=len(probs),
    )
    freqs = counts.astype(float) / float(shots)
    ranking = candidates[np.argsort(-freqs[candidates], kind="stable")]
    for idx in ranking:
        idx_int = int(idx)
        if idx_int in selected_indices:
            continue
        return (
            idx_int,
            float(freqs[idx_int]),
            float(probs[idx_int]),
            counts,
            {"selection_strategy": "probability", "selection_topk": 1},
        )
    return None, 0.0, 0.0, counts, {"selection_strategy": "probability", "selection_topk": 1}


def _to_plain_data(value, np):
    """Convert NumPy-heavy nested data into JSON-serializable Python objects."""
    if isinstance(value, dict):
        return {str(key): _to_plain_data(item, np) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain_data(item, np) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _restore_history(history_payload, np):
    """Restore NumPy arrays inside serialized history entries."""
    restored = []
    for entry in history_payload:
        item = dict(entry)
        for key in (
            "determinant_coeffs",
            "lucj_orbital",
            "lucj_same_spin",
            "lucj_opposite_spin",
            "uccsd_weights",
        ):
            if key in item and item[key] is not None:
                item[key] = np.asarray(item[key], dtype=float)
        if item.get("sample_counts") is not None:
            item["sample_counts"] = np.asarray(item["sample_counts"], dtype=int)
        if item.get("selected_determinant") is not None:
            item["selected_determinant"] = [int(bit) for bit in item["selected_determinant"]]
        restored.append(item)
    return restored


def _write_json_payload(path_like, payload) -> None:
    """Atomically write a JSON payload to disk."""
    path = Path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def cvqe(
    symbols: Sequence[str],
    geometry,
    *,
    adapt_it: int,
    ansatz: str = "lucj",
    basis: str = "sto-3g",
    charge: int = 0,
    spin: int = 0,
    active_electrons: int,
    active_orbitals: int,
    shots: Optional[int] = 0,
    optimizer_method: str = "BFGS",
    optimizer_maxiter: int = 100,
    hamiltonian_cutoff: float = 1e-20,
    selection_method: str = "probability",
    selection_topk: int = 10,
    selection_seed: Optional[int] = None,
    print_progress: bool = True,
    return_details: bool = False,
    resume_state: Optional[dict[str, object]] = None,
    checkpoint_path: Optional[str | Path] = None,
    iteration_callback: Optional[Callable[[dict[str, object]], None]] = None,
):
    """Run a simplified cyclic VQE loop.

    Parameters
    ----------
    symbols
        Atomic symbols, e.g. ``["H", "H"]``.
    geometry
        Nuclear coordinates, array-like with shape ``(n_atoms, 3)`` in Angstrom.
    adapt_it
        Number of CVQE determinant-growth cycles to attempt.
    basis
        Basis set name understood by PySCF.
    ansatz
        Compact entangler used between determinant-growth steps. Supported
        values are ``"lucj"`` and ``"uccsd"``.
    charge
        Total molecular charge.
    spin
        Spin parameter used by PySCF as ``2S``. Only ``spin=0`` is currently
        supported.
    active_electrons
        Number of active electrons in the CASCI reference.
    active_orbitals
        Number of active orbitals in the CASCI reference.
    shots
        Determinant-selection shots. ``None`` or ``0`` uses exact output-state
        probabilities; a positive integer uses sampled frequencies.
    optimizer_method
        SciPy optimizer name used after each determinant addition.
    optimizer_maxiter
        Maximum SciPy iterations per CVQE cycle.
    hamiltonian_cutoff
        Drop Hamiltonian terms below this absolute coefficient threshold.
    selection_method
        Determinant-admission strategy. ``"probability"`` adds the highest-weight
        unselected determinant from the output state. ``"pt2"`` screens the
        highest-amplitude unselected determinants and selects the one with the
        largest estimated second-order energy lowering.
    selection_topk
        Number of highest-amplitude unselected determinants screened by the
        ``"pt2"`` selector at each CVQE iteration.
    selection_seed
        RNG seed for shot-based determinant selection.
    print_progress
        If True, print iteration and energy updates.
    return_details
        If True, return an additional details dictionary.
    resume_state
        Optional checkpoint payload returned by an earlier CVQE run. When
        provided, CVQE resumes from that determinant list, parameter vector,
        and energy history and continues until ``adapt_it`` total iterations.
    checkpoint_path
        Optional JSON checkpoint path written atomically after each completed
        CVQE iteration.
    iteration_callback
        Optional callable invoked after each completed CVQE iteration with the
        same plain-Python checkpoint payload written to ``checkpoint_path``.

    Returns
    -------
    tuple
        ``(params, determinants, energies)`` where ``params`` is a flattened
        parameter vector whose leading block contains the determinant
        coefficients for the returned ``determinants`` list, followed by the
        optimized entangler parameters for the selected ansatz.

        If ``return_details=True``, returns
        ``(params, determinants, energies, details)``.
    """
    # pylint: disable=too-many-locals, too-many-statements, too-many-branches, too-many-arguments
    n_atoms = _validate_inputs(
        symbols=symbols,
        geometry=geometry,
        adapt_it=adapt_it,
        shots=shots,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
        spin=spin,
        selection_topk=selection_topk,
    )
    ansatz = str(ansatz).strip().lower()
    if ansatz not in {"lucj", "uccsd"}:
        raise ValueError("ansatz must be one of {'lucj', 'uccsd'}")
    selection_method = str(selection_method).strip().lower()
    if selection_method not in {"probability", "pt2"}:
        raise ValueError("selection_method must be one of {'probability', 'pt2'}")
    if selection_method == "pt2" and shots not in (None, 0):
        raise ValueError("selection_method='pt2' currently requires shots=0")

    try:
        import numpy as np
        import pennylane as qml
        import pyscf
        from pyscf import cc, gto, mcscf, scf
        from scipy.optimize import minimize
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "cvqe requires dependencies. Install at least: "
            "`pip install numpy scipy pennylane pyscf`."
        ) from exc

    atom = [(symbols[i], tuple(float(x) for x in geometry[i])) for i in range(n_atoms)]
    mol_ref = gto.Mole()
    mol_ref.atom = atom
    mol_ref.unit = "Angstrom"
    mol_ref.basis = basis
    mol_ref.charge = charge
    mol_ref.spin = spin
    mol_ref.symmetry = False
    mol_ref.build()

    mf_ref = scf.RHF(mol_ref)
    mf_ref.level_shift = 0.5
    mf_ref.diis_space = 12
    mf_ref.max_cycle = 100
    mf_ref.kernel()
    if not mf_ref.converged:
        mf_ref = scf.newton(mf_ref).run()

    mycc = cc.CCSD(mf_ref)
    mycc.kernel()

    mycas_ref = mcscf.CASCI(mf_ref, active_orbitals, active_electrons)
    h1ecas, ecore = mycas_ref.get_h1eff(mf_ref.mo_coeff)
    h2ecas = mycas_ref.get_h2eff(mf_ref.mo_coeff)

    ncas = int(mycas_ref.ncas)
    two_mo = pyscf.ao2mo.restore("1", h2ecas, norb=ncas)
    two_mo = np.swapaxes(two_mo, 1, 3)
    one_mo = h1ecas
    core_constant = np.array([ecore])
    h_fermionic = qml.qchem.fermionic_observable(
        core_constant,
        one_mo,
        two_mo,
        cutoff=hamiltonian_cutoff,
    )
    h_qubit = qml.jordan_wigner(h_fermionic)

    if hasattr(h_qubit, "terms"):
        coeffs, ops = h_qubit.terms()
    else:
        coeffs, ops = getattr(h_qubit, "coeffs", []), getattr(h_qubit, "ops", [])
    coeffs = np.asarray(coeffs, dtype=complex)
    if coeffs.size > 0 and (np.any(np.abs(coeffs.imag) > 1e-12) or coeffs.dtype.kind == "c"):
        h_qubit = qml.Hamiltonian(coeffs.real.astype(float), ops)

    qubits = 2 * ncas
    wires = tuple(range(qubits))
    h_matrix = np.asarray(qml.matrix(h_qubit, wire_order=wires), dtype=complex)

    hf_bits = np.asarray(qml.qchem.hf_state(active_electrons, qubits), dtype=int)
    hf_bits_list = [int(bit) for bit in hf_bits]
    hf_index = _bits_to_index(hf_bits_list)
    occupancy_table = np.asarray(
        [_index_to_bits(index, qubits) for index in range(2**qubits)],
        dtype=float,
    )
    fixed_electron_indices = np.where(
        np.sum(occupancy_table, axis=1).astype(int) == int(active_electrons)
    )[0]

    t1_active, t2_active = _extract_active_ccsd_amplitudes(
        mycc,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
        np=np,
    )
    n_lucj_pairs = max(0, active_orbitals - 1)
    uccsd_metadata = None
    if ansatz == "lucj":
        current_ansatz_payload = _initialize_lucj_params(
            t1_active,
            t2_active,
            active_orbitals=active_orbitals,
            active_electrons=active_electrons,
            np=np,
        )
    else:
        uccsd_metadata = _build_uccsd_excitation_metadata(
            active_electrons=active_electrons,
            qubits=qubits,
            qml=qml,
        )
        current_ansatz_payload = _initialize_uccsd_params(
            t1_active,
            t2_active,
            uccsd_metadata,
            active_electrons=active_electrons,
            np=np,
        )

    added_determinants: list[list[int]] = []
    selected_indices = {int(hf_index)}
    rng = np.random.default_rng(selection_seed)
    energies: list[float] = []
    history: list[dict[str, object]] = []

    def _pack_all_params(det_coeffs, ansatz_payload):
        if ansatz == "lucj":
            orbital_params, same_params, opposite_params = ansatz_payload
            return _pack_params(
                det_coeffs,
                orbital_params,
                same_params,
                opposite_params,
                np=np,
            )
        return _pack_uccsd_params(det_coeffs, ansatz_payload, np=np)

    def _unpack_all_params(flat_params, n_det_params: int):
        if ansatz == "lucj":
            det_coeffs, orbital_params, same_params, opposite_params = _unpack_params(
                flat_params,
                n_det_params=n_det_params,
                n_orb_pairs=n_lucj_pairs,
                np=np,
            )
            return det_coeffs, (orbital_params, same_params, opposite_params)

        det_coeffs, uccsd_weights = _unpack_uccsd_params(
            flat_params,
            n_det_params=n_det_params,
            n_uccsd_params=len(uccsd_metadata),
            np=np,
        )
        return det_coeffs, uccsd_weights

    def _state_from_blocks(det_coeffs, ansatz_payload):
        reference = _build_reference_state(
            hf_bits_list,
            added_determinants,
            det_coeffs,
            qubits=qubits,
            np=np,
        )
        if ansatz == "lucj":
            orbital_params, same_params, opposite_params = ansatz_payload
            state = _apply_lucj(
                reference,
                orbital_params,
                same_params,
                opposite_params,
                qubits=qubits,
                occupancy_table=occupancy_table,
                qml=qml,
                np=np,
            )
        else:
            state = _apply_uccsd(
                reference,
                ansatz_payload,
                uccsd_metadata,
                qubits=qubits,
                qml=qml,
                np=np,
            )
        norm = np.linalg.norm(state)
        if norm < 1e-14:
            raise ValueError("trial state became numerically zero")
        return state / norm

    def _cost(flat_params):
        det_coeffs, ansatz_payload = _unpack_all_params(
            flat_params,
            n_det_params=len(added_determinants),
        )
        state = _state_from_blocks(det_coeffs, ansatz_payload)
        return float(np.real(state.conj().T @ (h_matrix @ state)))

    def _parameter_slices_for_count(n_det_params: int) -> dict[str, tuple[int, int]]:
        if ansatz == "lucj":
            return _parameter_slices(
                n_det_params=n_det_params,
                n_orb_pairs=n_lucj_pairs,
            )
        return _parameter_slices_uccsd(
            n_det_params=n_det_params,
            n_uccsd_params=len(uccsd_metadata),
        )

    def _build_details_payload(*, plain: bool):
        details = {
            "ansatz": str(ansatz),
            "initial_energy": float(initial_energy),
            "hf_determinant": [int(bit) for bit in hf_bits_list],
            "reference_determinants": [[int(bit) for bit in hf_bits_list]]
            + [[int(bit) for bit in determinant] for determinant in added_determinants],
            "fixed_electron_sector_size": int(len(fixed_electron_indices)),
            "parameter_slices": _parameter_slices_for_count(len(added_determinants)),
            "history": history,
        }
        if ansatz == "uccsd":
            details["uccsd_excitations"] = [
                [int(idx) for idx in meta["excitation"]] for meta in uccsd_metadata
            ]
        if plain:
            return _to_plain_data(details, np)
        return details

    def _build_checkpoint_payload():
        return {
            "version": 1,
            "config": {
                "symbols": [str(symbol) for symbol in symbols],
                "geometry": _to_plain_data(geometry, np),
                "adapt_it": int(adapt_it),
                "ansatz": str(ansatz),
                "basis": str(basis),
                "charge": int(charge),
                "spin": int(spin),
                "active_electrons": int(active_electrons),
                "active_orbitals": int(active_orbitals),
                "shots": None if shots is None else int(shots),
                "optimizer_method": str(optimizer_method),
                "optimizer_maxiter": int(optimizer_maxiter),
                "hamiltonian_cutoff": float(hamiltonian_cutoff),
                "selection_method": str(selection_method),
                "selection_topk": int(selection_topk),
                "selection_seed": None if selection_seed is None else int(selection_seed),
            },
            "ansatz": str(ansatz),
            "completed_iterations": int(len(energies)),
            "current_params": _to_plain_data(np.asarray(current_params, dtype=float), np),
            "added_determinants": [[int(bit) for bit in determinant] for determinant in added_determinants],
            "energies": [float(energy) for energy in energies],
            "details": _build_details_payload(plain=True),
        }

    def _emit_progress() -> None:
        payload = _build_checkpoint_payload()
        if checkpoint_path is not None:
            _write_json_payload(checkpoint_path, payload)
        if iteration_callback is not None:
            iteration_callback(payload)

    current_det_coeffs = np.zeros(0, dtype=float)
    current_params = _pack_all_params(current_det_coeffs, current_ansatz_payload)
    current_state = _state_from_blocks(
        current_det_coeffs,
        current_ansatz_payload,
    )
    initial_energy = float(np.real(current_state.conj().T @ (h_matrix @ current_state)))

    if resume_state is not None:
        resume_details = dict(resume_state.get("details", {}))
        resume_config = dict(resume_state.get("config", {}))
        resume_ansatz = str(resume_state.get("ansatz", resume_details.get("ansatz", ansatz))).strip().lower()
        if resume_ansatz != ansatz:
            raise ValueError("resume_state ansatz does not match the requested ansatz")

        if resume_config:
            expected_symbols = [str(symbol) for symbol in symbols]
            if "symbols" in resume_config and list(resume_config["symbols"]) != expected_symbols:
                raise ValueError("resume_state symbols do not match the requested system")
            if "geometry" in resume_config:
                resume_geometry = np.asarray(resume_config["geometry"], dtype=float)
                requested_geometry = np.asarray(geometry, dtype=float)
                if resume_geometry.shape != requested_geometry.shape or not np.allclose(
                    resume_geometry,
                    requested_geometry,
                    atol=1e-12,
                    rtol=0.0,
                ):
                    raise ValueError("resume_state geometry does not match the requested system")
            for key, expected in (
                ("basis", str(basis)),
                ("charge", int(charge)),
                ("spin", int(spin)),
                ("active_electrons", int(active_electrons)),
                ("active_orbitals", int(active_orbitals)),
                ("optimizer_method", str(optimizer_method)),
                ("optimizer_maxiter", int(optimizer_maxiter)),
                ("shots", None if shots is None else int(shots)),
                ("selection_method", str(selection_method)),
                ("selection_topk", int(selection_topk)),
                ("selection_seed", None if selection_seed is None else int(selection_seed)),
            ):
                if key in resume_config and resume_config[key] != expected:
                    raise ValueError(f"resume_state {key} does not match the requested run")

        resume_hf = resume_details.get("hf_determinant", hf_bits_list)
        resume_hf = [int(bit) for bit in resume_hf]
        if resume_hf != hf_bits_list:
            raise ValueError("resume_state does not match the Hartree-Fock determinant")

        added_determinants = [
            [int(bit) for bit in determinant]
            for determinant in resume_state.get("added_determinants", [])
        ]
        energies = [float(energy) for energy in resume_state.get("energies", [])]
        history_payload = resume_details.get("history", resume_state.get("history", []))
        history = _restore_history(history_payload, np)

        if len(added_determinants) != len(energies) or len(history) != len(energies):
            raise ValueError(
                "resume_state is inconsistent: determinant, energy, and history lengths must match"
            )

        current_params_payload = resume_state.get("current_params")
        if current_params_payload is None:
            raise ValueError("resume_state must include current_params")

        current_params = np.asarray(current_params_payload, dtype=float)
        selected_indices = {int(hf_index)}
        for determinant in added_determinants:
            selected_indices.add(_bits_to_index(determinant))

        current_det_coeffs, current_ansatz_payload = _unpack_all_params(
            current_params,
            n_det_params=len(added_determinants),
        )
        current_state = _state_from_blocks(
            current_det_coeffs,
            current_ansatz_payload,
        )
        initial_energy = float(resume_details.get("initial_energy", initial_energy))

    if print_progress:
        if len(energies) == 0:
            print(f"Initial CVQE energy: {initial_energy}", flush=True)
        else:
            print(
                f"Resuming CVQE from iteration {len(energies)} with current energy {energies[-1]}",
                flush=True,
            )

    for iteration in range(len(energies), int(adapt_it)):
        probabilities = np.abs(current_state) ** 2
        current_energy = float(np.real(current_state.conj().T @ (h_matrix @ current_state)))
        selected_index, selected_metric, selected_exact_prob, counts, selection_details = _select_new_determinant(
            probabilities,
            selected_indices,
            candidate_indices=fixed_electron_indices,
            shots=shots,
            selection_method=selection_method,
            selection_topk=selection_topk,
            current_state=current_state,
            current_energy=current_energy,
            h_matrix=h_matrix,
            rng=rng,
            np=np,
        )

        if selected_index is None:
            if print_progress:
                print("No additional determinant found; stopping early.", flush=True)
            break

        selected_bits = _index_to_bits(int(selected_index), qubits)
        added_determinants.append(selected_bits)
        selected_indices.add(int(selected_index))

        selected_amplitude = complex(current_state[int(selected_index)])
        amplitude_guess = float(np.real(selected_amplitude))
        if abs(amplitude_guess) < 1e-8:
            sign = -1.0 if np.real(selected_amplitude) < 0 else 1.0
            amplitude_guess = sign * float(np.sqrt(max(selected_exact_prob, selected_metric, 1e-6)))

        prev_det_count = len(added_determinants) - 1
        prev_det_coeffs, prev_ansatz_payload = _unpack_all_params(
            current_params,
            n_det_params=prev_det_count,
        )
        x0 = _pack_all_params(
            np.concatenate([prev_det_coeffs, np.array([amplitude_guess])]),
            prev_ansatz_payload,
        )

        if print_progress:
            mode = "exact" if shots is None or int(shots) == 0 else f"{int(shots)} shots"
            print(
                f"CVQE iteration {iteration + 1}: selected determinant {selected_bits} "
                f"with selection weight {selected_metric:.6e} "
                f"({mode}, strategy={selection_method})",
                flush=True,
            )

        result = minimize(
            _cost,
            x0,
            method=optimizer_method,
            tol=1e-12,
            options={"disp": False, "maxiter": int(optimizer_maxiter)},
        )

        current_params = np.asarray(result.x, dtype=float)
        current_det_coeffs, current_ansatz_payload = _unpack_all_params(
            current_params,
            n_det_params=len(added_determinants),
        )
        current_state = _state_from_blocks(
            current_det_coeffs,
            current_ansatz_payload,
        )
        energy = float(result.fun)
        energies.append(energy)

        history_entry = {
            "iteration": int(iteration + 1),
            "ansatz": str(ansatz),
            "selected_determinant": [int(bit) for bit in selected_bits],
            "selected_index": int(selected_index),
            "selection_mode": "exact" if shots is None or int(shots) == 0 else "sampled",
            "selection_shots": int(0 if shots is None else shots),
            "selection_metric": float(selected_metric),
            "selected_exact_probability": float(selected_exact_prob),
            "energy": float(energy),
            "optimizer_method": str(optimizer_method),
            "optimizer_maxiter": int(optimizer_maxiter),
            "optimizer_nit": None if getattr(result, "nit", None) is None else int(result.nit),
            "optimizer_success": bool(getattr(result, "success", False)),
            "determinant_coeffs": np.asarray(current_det_coeffs, dtype=float).copy(),
            "sample_counts": None if counts is None else np.asarray(counts, dtype=int),
        }
        if selection_details is not None:
            history_entry.update(_to_plain_data(selection_details, np))
        if ansatz == "lucj":
            orbital_angles, same_spin, opposite_spin = current_ansatz_payload
            history_entry.update(
                {
                    "lucj_orbital": np.asarray(orbital_angles, dtype=float).copy(),
                    "lucj_same_spin": np.asarray(same_spin, dtype=float).copy(),
                    "lucj_opposite_spin": np.asarray(opposite_spin, dtype=float).copy(),
                }
            )
        else:
            history_entry["uccsd_weights"] = np.asarray(current_ansatz_payload, dtype=float).copy()
        history.append(history_entry)
        _emit_progress()

        if print_progress:
            print("Energies are", energies, flush=True)

    if len(energies) > 0 and print_progress:
        print("energies:", energies[-1], flush=True)

    if return_details:
        details = _build_details_payload(plain=False)
        return current_params, added_determinants, energies, details

    return current_params, added_determinants, energies
