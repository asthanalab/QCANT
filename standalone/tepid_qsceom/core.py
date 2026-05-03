"""Self-contained TEPID/qscEOM workflows.

This module deliberately does not import from ``QCANT``. It contains the
minimal chemistry, basis, pool, ansatz, TEPID-ADAPT, and analytic qscEOM logic
needed to run finite-temperature and excited-state workflows from one folder.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence


_FERMIONIC_POOL_ALIASES = {"fermionic_sd", "sd", "fermionic"}
_QUBIT_POOL_ALIASES = {"qubit_excitation", "qe", "qubit"}


class ExcitationList(list):
    """List-like excitation container that preserves ansatz metadata."""

    def __init__(
        self,
        *args,
        ansatz_type: str = "fermionic",
        pool_type: str = "fermionic_sd",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.ansatz_type = str(ansatz_type)
        self.pool_type = str(pool_type)


def _normalize_array_backend(array_backend: Optional[str]) -> str:
    """Normalize dense linear algebra backend selection."""
    normalized = str(array_backend or "auto").strip().lower()
    if normalized not in {"auto", "numpy", "cupy"}:
        raise ValueError("array_backend must be one of {'auto', 'numpy', 'cupy'}")
    return normalized


def _resolve_array_module(array_backend: Optional[str], *, context: str):
    """Resolve NumPy vs CuPy for dense linear algebra."""
    np, _qml, _minimize = _require_dependencies()
    normalized = _normalize_array_backend(array_backend)
    if normalized == "cupy":
        try:
            import cupy as cp
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                f"{context} requires CuPy for array_backend='cupy'. Install `cupy-cuda12x`."
            ) from exc
        return cp, "cupy", True
    return np, "numpy", False


def _to_host_array(value, *, dtype=None):
    """Convert NumPy/CuPy arrays into NumPy host arrays."""
    np, _qml, _minimize = _require_dependencies()
    try:
        import cupy as cp
    except ImportError:  # pragma: no cover
        cp = None

    if cp is not None and isinstance(value, cp.ndarray):
        arr = cp.asnumpy(value)
        return np.asarray(arr, dtype=dtype) if dtype is not None else np.asarray(arr)
    if hasattr(value, "get") and cp is not None:
        arr = cp.asnumpy(cp.asarray(value))
        return np.asarray(arr, dtype=dtype) if dtype is not None else np.asarray(arr)
    return np.asarray(value, dtype=dtype) if dtype is not None else np.asarray(value)


def _require_dependencies():
    """Import required scientific Python dependencies lazily."""
    try:
        import numpy as np
        import pennylane as qml
        from scipy.optimize import minimize
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "This standalone package requires numpy, scipy, pennylane, and pyscf."
        ) from exc
    return np, qml, minimize


def _normalize_geometry(symbols: Sequence[str], geometry):
    """Normalize geometry to a real-valued array-like structure."""
    np, _qml, _minimize = _require_dependencies()
    arr = np.asarray(geometry, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] != len(symbols):
        raise ValueError("geometry must have shape (len(symbols), 3)")
    return arr


def _normalize_pool_type(pool_type: str) -> str:
    """Normalize user pool-type aliases into a canonical value."""
    normalized = str(pool_type).strip().lower()
    if normalized in _FERMIONIC_POOL_ALIASES:
        return "fermionic_sd"
    if normalized in _QUBIT_POOL_ALIASES:
        return "qubit_excitation"
    raise ValueError(
        "pool_type must be one of {'fermionic_sd', 'sd', 'fermionic', "
        "'qubit_excitation', 'qe', 'qubit'}"
    )


def _normalize_ansatz_type(ansatz_type: Optional[str]) -> str:
    """Normalize ansatz-type aliases into a canonical value."""
    if ansatz_type is None:
        return "fermionic"
    normalized = str(ansatz_type).strip().lower()
    if normalized in _FERMIONIC_POOL_ALIASES:
        return "fermionic"
    if normalized in _QUBIT_POOL_ALIASES:
        return "qubit_excitation"
    raise ValueError(
        "ansatz_type must be one of {'fermionic', 'fermionic_sd', 'sd', "
        "'qubit_excitation', 'qe', 'qubit'}"
    )


def _ansatz_type_from_pool_type(pool_type: str) -> str:
    """Resolve the ansatz operator family implied by the ADAPT pool."""
    if pool_type == "fermionic_sd":
        return "fermionic"
    if pool_type == "qubit_excitation":
        return "qubit_excitation"
    raise ValueError(f"Unsupported canonical pool_type: {pool_type}")


def inite(elec: int, orb: int) -> list[list[int]]:
    """Generate HF-preserving single and double excitation occupations."""
    config: list[int] = []
    out: list[list[int]] = []

    for x in range(elec):
        count = orb - elec
        while count < orb:
            for e in range(elec):
                if x == e:
                    if x % 2 == 0:
                        config.append(count)
                        count += 2
                    else:
                        config.append(count + 1)
                        count += 2
                else:
                    config.append(e)
            out.append(config)
            config = []

    for x in range(elec):
        for y in range(x + 1, elec):
            for count1 in range(elec, orb, 2):
                for count2 in range(elec, orb, 2):
                    keep = 0
                    if count1 == count2:
                        if (x % 2) != (y % 2):
                            keep = 1
                    else:
                        keep = 1
                    if (x % 2) == (y % 2) and count2 < count1:
                        keep = 0

                    if keep == 1:
                        for e in range(elec):
                            if x == e:
                                config.append(count1 if x % 2 == 0 else count1 + 1)
                            elif y == e:
                                config.append(count2 if y % 2 == 0 else count2 + 1)
                            else:
                                config.append(e)
                        out.append(config)
                        config = []
    return out


def _validate_basis_occupations(
    basis_occupations,
    *,
    qubits: Optional[int],
    active_electrons: int,
):
    """Validate and normalize user-provided occupation configurations."""
    if basis_occupations is None:
        return None

    normalized = []
    for occ in basis_occupations:
        try:
            occ_list = sorted(int(idx) for idx in occ)
        except Exception as exc:
            raise ValueError("basis_occupations must be an iterable of occupied-index iterables") from exc

        if len(occ_list) != int(active_electrons):
            raise ValueError("each basis occupation must contain active_electrons occupied indices")
        if len(set(occ_list)) != len(occ_list):
            raise ValueError("basis occupations cannot contain repeated occupied indices")
        if qubits is not None and any(idx < 0 or idx >= int(qubits) for idx in occ_list):
            raise ValueError("basis occupation indices must lie in [0, qubits)")
        normalized.append(occ_list)
    return normalized


def _basis_index_from_occupations(occ, qubits: int) -> int:
    """Convert an occupied-orbital list into a computational-basis index."""
    bits = [0] * int(qubits)
    for idx in occ:
        bits[int(idx)] = 1
    return int("".join(str(int(bit)) for bit in bits), 2)


def _build_basis_columns(occupations, *, qubits: int, np):
    """Build a column-stacked basis matrix for the chosen occupations."""
    dim = 2 ** int(qubits)
    basis = np.zeros((dim, len(occupations)), dtype=complex)
    for col, occ in enumerate(occupations):
        basis[_basis_index_from_occupations(occ, qubits), col] = 1.0
    return basis


def _apply_local_operator_batch(states, op_matrix, target_wires, *, qubits: int, np):
    """Apply a local operator matrix to one or more statevectors."""
    arr = np.asarray(states, dtype=complex)
    squeeze = False
    if arr.ndim == 1:
        arr = arr[:, None]
        squeeze = True

    batch = int(arr.shape[1])
    targets = tuple(int(wire) for wire in target_wires)
    remaining = tuple(wire for wire in range(int(qubits)) if wire not in targets)

    state_tensor = arr.reshape((2,) * int(qubits) + (batch,))
    permutation = targets + remaining + (int(qubits),)
    inverse_permutation = tuple(sorted(range(len(permutation)), key=permutation.__getitem__))

    permuted = np.transpose(state_tensor, permutation)
    flat_state = permuted.reshape(2 ** len(targets), -1)
    updated = np.asarray(op_matrix, dtype=complex) @ flat_state
    restored = updated.reshape((2,) * int(qubits) + (batch,))
    out = np.transpose(restored, inverse_permutation).reshape(2 ** int(qubits), batch)

    if squeeze:
        return out[:, 0]
    return out


def _build_excitation_operation(qml, excitation, weight, ansatz_type: str):
    """Build a PennyLane excitation operation and return it with its wires."""
    if ansatz_type == "fermionic":
        if len(excitation) == 2:
            op = qml.FermionicSingleExcitation(
                weight=float(weight),
                wires=list(range(int(excitation[0]), int(excitation[1]) + 1)),
            )
            return op, tuple(int(wire) for wire in op.wires)
        if len(excitation) == 4:
            op = qml.FermionicDoubleExcitation(
                weight=float(weight),
                wires1=list(range(int(excitation[0]), int(excitation[1]) + 1)),
                wires2=list(range(int(excitation[2]), int(excitation[3]) + 1)),
            )
            return op, tuple(int(wire) for wire in op.wires)
    elif ansatz_type == "qubit_excitation":
        if len(excitation) == 2:
            op = qml.SingleExcitation(float(weight), wires=[int(excitation[0]), int(excitation[1])])
            return op, tuple(int(wire) for wire in op.wires)
        if len(excitation) == 4:
            op = qml.DoubleExcitation(
                float(weight),
                wires=[
                    int(excitation[0]),
                    int(excitation[1]),
                    int(excitation[2]),
                    int(excitation[3]),
                ],
            )
            return op, tuple(int(wire) for wire in op.wires)

    raise ValueError(
        "Each excitation must have length 2 (single) or 4 (double); "
        f"received {excitation!r} for ansatz_type='{ansatz_type}'."
    )


def _apply_ansatz_batch(
    states,
    params,
    ash_excitation,
    *,
    ansatz_type: str,
    qubits: int,
    qml,
    np,
):
    """Apply the current ansatz to a batch of basis states."""
    current = np.asarray(states, dtype=complex)
    for weight, excitation in zip(params, ash_excitation):
        op, target_wires = _build_excitation_operation(qml, excitation, weight, ansatz_type)
        current = _apply_local_operator_batch(
            current,
            qml.matrix(op),
            target_wires,
            qubits=qubits,
            np=np,
        )
    return current


def _build_operator_pool(qml, active_electrons: int, qubits: int):
    """Build the excitation pool labels for TEPID-ADAPT."""
    singles, doubles = qml.qchem.excitations(int(active_electrons), int(qubits))
    pool_excitations = [list(map(int, excitation)) for excitation in singles]
    pool_excitations.extend(list(map(int, excitation)) for excitation in doubles)
    return pool_excitations


def _build_derivative_metadata(
    qml,
    pool_excitations,
    *,
    ansatz_type: str,
    gradient_eps: float,
    np,
):
    """Pre-compute local derivative matrices for candidate excitation gates."""
    metadata = []
    eps = float(gradient_eps)
    for excitation in pool_excitations:
        op_plus, target_wires = _build_excitation_operation(qml, excitation, +eps, ansatz_type)
        op_minus, _ = _build_excitation_operation(qml, excitation, -eps, ansatz_type)
        derivative = (
            np.asarray(qml.matrix(op_plus), dtype=complex)
            - np.asarray(qml.matrix(op_minus), dtype=complex)
        ) / (2.0 * eps)
        metadata.append(
            {
                "excitation": [int(v) for v in excitation],
                "target_wires": tuple(int(v) for v in target_wires),
                "derivative": derivative,
            }
        )
    return metadata


def _thermal_summary(energies, *, beta: float, np):
    """Compute Boltzmann weights, entropy, and free energy."""
    energy_values = np.asarray(energies, dtype=float)
    shifted = -float(beta) * energy_values
    shift = float(np.max(shifted))
    unnormalized = np.exp(shifted - shift)
    partition = float(np.sum(unnormalized))
    weights = unnormalized / partition
    safe_weights = np.clip(weights, 1e-300, None)
    entropy = float(-np.sum(weights * np.log(safe_weights)))
    log_partition = float(shift + np.log(partition))
    free_energy = float(-log_partition / float(beta))
    return weights, entropy, free_energy, log_partition


def tepid_boltzmann_weights(
    energies,
    *,
    temperature: Optional[float] = None,
    beta: Optional[float] = None,
):
    """Return normalized Boltzmann weights from a list of energies."""
    np, _qml, _minimize = _require_dependencies()
    if (temperature is None) == (beta is None):
        raise ValueError("Provide exactly one of temperature or beta")
    beta_value = float(beta) if beta is not None else 1.0 / float(temperature)
    if beta_value <= 0:
        raise ValueError("beta must be > 0")
    energy_values = np.asarray(energies, dtype=float)
    if energy_values.ndim != 1 or energy_values.size == 0:
        raise ValueError("energies must be a non-empty one-dimensional array")
    shifted = -beta_value * energy_values
    shift = float(np.max(shifted))
    unnormalized = np.exp(shifted - shift)
    return unnormalized / float(np.sum(unnormalized))


def _build_molecular_hamiltonian(
    symbols: Sequence[str],
    geometry,
    *,
    basis: str,
    charge: int,
    active_electrons: int,
    active_orbitals: int,
    method: str,
):
    """Build the active-space molecular Hamiltonian and its dense matrix."""
    np, qml, _minimize = _require_dependencies()
    geometry_array = _normalize_geometry(symbols, geometry)
    H, qubits = qml.qchem.molecular_hamiltonian(
        symbols,
        geometry_array,
        basis=basis,
        method=method,
        active_electrons=int(active_electrons),
        active_orbitals=int(active_orbitals),
        charge=int(charge),
    )
    H_dense = np.asarray(qml.matrix(H, wire_order=range(qubits)), dtype=complex)
    H_dense = 0.5 * (H_dense + H_dense.conj().T)
    return H, H_dense, int(qubits), geometry_array


def _build_truncated_basis_occupations(
    *,
    active_electrons: int,
    qubits: int,
    include_identity: bool = True,
    basis_occupations=None,
):
    """Build the HF + singles + doubles truncated basis."""
    np, qml, _minimize = _require_dependencies()
    basis_occupations = _validate_basis_occupations(
        basis_occupations,
        qubits=qubits,
        active_electrons=int(active_electrons),
    )
    hf_state = qml.qchem.hf_state(int(active_electrons), int(qubits))
    hf_occ = [int(idx) for idx, bit in enumerate(np.asarray(hf_state, dtype=int)) if int(bit) == 1]

    if basis_occupations is None:
        sd_occupations = [list(int(v) for v in occ) for occ in inite(int(active_electrons), int(qubits))]
    else:
        sd_occupations = [list(int(v) for v in occ) for occ in basis_occupations]

    if include_identity:
        basis_occ = [hf_occ] + [occ for occ in sd_occupations if occ != hf_occ]
    else:
        basis_occ = [occ for occ in sd_occupations if occ != hf_occ]

    seen = set()
    unique_basis_occ = []
    for occ in basis_occ:
        key = tuple(int(v) for v in occ)
        if key in seen:
            continue
        seen.add(key)
        unique_basis_occ.append(list(key))
    return unique_basis_occ, hf_occ


def compute_exact_sector_roots(
    symbols: Sequence[str],
    geometry,
    *,
    basis: str,
    charge: int,
    active_electrons: int,
    active_orbitals: int,
    method: str = "pyscf",
):
    """Compute exact roots in the closed-shell fixed-N_alpha, fixed-N_beta sector."""
    np, _qml, _minimize = _require_dependencies()
    _H, H_dense, qubits, _geometry_array = _build_molecular_hamiltonian(
        symbols,
        geometry,
        basis=basis,
        charge=charge,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
        method=method,
    )
    if (int(active_electrons) % 2) != 0:
        raise ValueError("compute_exact_sector_roots currently assumes an even number of active electrons")

    target_alpha = int(active_electrons // 2)
    target_beta = int(active_electrons // 2)
    sector_indices = []
    for idx in range(2**int(qubits)):
        bits = format(int(idx), f"0{int(qubits)}b")
        alpha_count = sum(int(bits[pos]) for pos in range(0, int(qubits), 2))
        beta_count = sum(int(bits[pos]) for pos in range(1, int(qubits), 2))
        if alpha_count == target_alpha and beta_count == target_beta:
            sector_indices.append(int(idx))
    if not sector_indices:
        raise RuntimeError("No basis states found in the requested N_alpha/N_beta sector")

    sector_hamiltonian = H_dense[np.ix_(sector_indices, sector_indices)]
    eigvals = np.sort(np.real(np.linalg.eigvalsh(sector_hamiltonian)))
    return np.asarray(eigvals, dtype=float)


def tepid_adapt(
    symbols: Sequence[str],
    geometry,
    *,
    adapt_it: int,
    temperature: Optional[float] = None,
    beta: Optional[float] = None,
    basis: str = "sto-3g",
    charge: int = 0,
    active_electrons: int,
    active_orbitals: int,
    shots: int = 0,
    method: str = "pyscf",
    pool_type: str = "fermionic_sd",
    include_identity: bool = True,
    basis_occupations=None,
    pool_sample_size: Optional[int] = None,
    pool_seed: Optional[int] = None,
    array_backend: str = "auto",
    gradient_eps: float = 1e-7,
    gradient_tol: Optional[float] = None,
    optimizer_method: str = "BFGS",
    optimizer_maxiter: int = 200,
    return_details: bool = False,
    iteration_callback: Optional[Callable[[Mapping[str, Any]], None]] = None,
):
    """Run ancilla-free TEPID-ADAPT on a molecular active space."""
    np, qml, minimize = _require_dependencies()
    xp, array_backend_name, using_gpu = _resolve_array_module(array_backend, context="standalone tepid_adapt")

    if len(symbols) == 0:
        raise ValueError("symbols must be non-empty")
    if adapt_it < 0:
        raise ValueError("adapt_it must be >= 0")
    if active_electrons <= 0 or active_orbitals <= 0:
        raise ValueError("active_electrons and active_orbitals must be > 0")
    if active_electrons > 2 * active_orbitals:
        raise ValueError("active_electrons cannot exceed 2 * active_orbitals")
    if shots != 0:
        raise ValueError("standalone tepid_adapt currently supports only analytic mode with shots=0")
    if gradient_eps <= 0:
        raise ValueError("gradient_eps must be > 0")
    if gradient_tol is not None and gradient_tol < 0:
        raise ValueError("gradient_tol must be >= 0")
    if (temperature is None) == (beta is None):
        raise ValueError("Provide exactly one of temperature or beta")
    if temperature is not None and temperature <= 0:
        raise ValueError("temperature must be > 0")
    if beta is not None and beta <= 0:
        raise ValueError("beta must be > 0")
    if pool_sample_size is not None and pool_sample_size <= 0:
        raise ValueError("pool_sample_size must be > 0")

    pool_type_canonical = _normalize_pool_type(pool_type)
    ansatz_type = _ansatz_type_from_pool_type(pool_type_canonical)
    beta_value = float(beta) if beta is not None else 1.0 / float(temperature)
    temperature_value = float(temperature) if temperature is not None else 1.0 / float(beta)

    _H, H_dense_host, qubits, _geometry_array = _build_molecular_hamiltonian(
        symbols,
        geometry,
        basis=basis,
        charge=charge,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
        method=method,
    )
    basis_occ, hf_occ = _build_truncated_basis_occupations(
        active_electrons=int(active_electrons),
        qubits=int(qubits),
        include_identity=bool(include_identity),
        basis_occupations=basis_occupations,
    )
    H_dense = xp.asarray(H_dense_host, dtype=complex)
    basis_columns = _build_basis_columns(basis_occ, qubits=int(qubits), np=xp)
    pool_excitations = _build_operator_pool(qml, int(active_electrons), int(qubits))
    derivative_metadata = _build_derivative_metadata(
        qml,
        pool_excitations,
        ansatz_type=ansatz_type,
        gradient_eps=float(gradient_eps),
        np=xp,
    )

    ash_excitation = ExcitationList(ansatz_type=ansatz_type, pool_type=pool_type_canonical)
    params = np.zeros(0, dtype=float)
    rng = np.random.default_rng(pool_seed)
    history = []
    free_energies = []

    def _evaluate_snapshot(params_eval, excitations_eval):
        states = _apply_ansatz_batch(
            basis_columns,
            params_eval,
            excitations_eval,
            ansatz_type=ansatz_type,
            qubits=int(qubits),
            qml=qml,
            np=xp,
        )
        h_states = H_dense @ states
        basis_energies = xp.real(xp.sum(xp.conj(states) * h_states, axis=0))
        weights, entropy, free_energy, log_partition = _thermal_summary(
            basis_energies,
            beta=beta_value,
            np=xp,
        )
        return {
            "states": xp.asarray(states, dtype=complex),
            "basis_energies": xp.asarray(basis_energies, dtype=float),
            "thermal_weights": xp.asarray(weights, dtype=float),
            "entropy": float(entropy),
            "free_energy": float(free_energy),
            "log_partition": float(log_partition),
        }

    def _objective(params_eval):
        snapshot = _evaluate_snapshot(params_eval, ash_excitation)
        return float(snapshot["free_energy"])

    initial_snapshot = _evaluate_snapshot(params, ash_excitation)
    current_snapshot = initial_snapshot

    for iteration in range(int(adapt_it)):
        if pool_sample_size is None or pool_sample_size >= len(pool_excitations):
            candidate_indices = list(range(len(pool_excitations)))
        else:
            candidate_indices = [
                int(idx)
                for idx in rng.choice(len(pool_excitations), size=int(pool_sample_size), replace=False)
            ]

        max_score = float("-inf")
        max_gradient = 0.0
        max_operator_index = None
        max_operator_excitation = None

        for idx in candidate_indices:
            meta = derivative_metadata[int(idx)]
            d_states = _apply_local_operator_batch(
                current_snapshot["states"],
                meta["derivative"],
                meta["target_wires"],
                qubits=int(qubits),
                np=xp,
            )
            hd_states = H_dense @ d_states
            deriv_energies = 2.0 * xp.real(
                xp.sum(xp.conj(current_snapshot["states"]) * hd_states, axis=0)
            )
            weighted_gradient = float(
                _to_host_array(
                    xp.dot(current_snapshot["thermal_weights"], xp.asarray(deriv_energies, dtype=float))
                ).item()
            )
            score = abs(weighted_gradient)
            if score > max_score:
                max_score = float(score)
                max_gradient = float(weighted_gradient)
                max_operator_index = int(idx)
                max_operator_excitation = [int(v) for v in meta["excitation"]]

        if max_operator_excitation is None or max_operator_index is None:
            raise RuntimeError("No operator selected during TEPID-ADAPT iteration.")

        if gradient_tol is not None and max_score < float(gradient_tol):
            break

        ash_excitation.append(max_operator_excitation)
        params = np.append(np.asarray(params, dtype=float), 0.0)

        result = minimize(
            _objective,
            params,
            method=optimizer_method,
            tol=1e-12,
            options={"disp": False, "maxiter": int(optimizer_maxiter)},
        )

        params = np.asarray(result.x, dtype=float)
        current_snapshot = _evaluate_snapshot(params, ash_excitation)
        free_energies.append(float(current_snapshot["free_energy"]))

        basis_energies_host = _to_host_array(current_snapshot["basis_energies"], dtype=float)
        thermal_weights_host = _to_host_array(current_snapshot["thermal_weights"], dtype=float)
        energy_order = np.argsort(basis_energies_host, kind="stable")
        sorted_basis_occ = [basis_occ[int(idx)] for idx in energy_order]
        iteration_snapshot = {
            "iteration": int(iteration + 1),
            "selected_pool_index": int(max_operator_index),
            "selected_excitation": [int(v) for v in max_operator_excitation],
            "tepid_gradient": float(max_gradient),
            "tepid_gradient_abs": float(max_score),
            "free_energy": float(current_snapshot["free_energy"]),
            "entropy": float(current_snapshot["entropy"]),
            "log_partition": float(current_snapshot["log_partition"]),
            "basis_energies": basis_energies_host.copy(),
            "thermal_weights": thermal_weights_host.copy(),
            "sorted_basis_energies": basis_energies_host[energy_order].copy(),
            "sorted_thermal_weights": thermal_weights_host[energy_order].copy(),
            "sorted_basis_occupations": [list(int(v) for v in occ) for occ in sorted_basis_occ],
            "params": np.asarray(params, dtype=float).copy(),
            "ash_excitation": [list(int(v) for v in excitation) for excitation in ash_excitation],
            "optimizer_success": bool(result.success),
            "optimizer_status": int(result.status),
            "optimizer_message": str(result.message),
            "optimizer_nit": getattr(result, "nit", None),
            "optimizer_nfev": getattr(result, "nfev", None),
        }
        history.append(iteration_snapshot)
        if iteration_callback is not None:
            iteration_callback(iteration_snapshot)

    initial_basis_energies = _to_host_array(initial_snapshot["basis_energies"], dtype=float)
    initial_thermal_weights = _to_host_array(initial_snapshot["thermal_weights"], dtype=float)
    final_basis_energies = _to_host_array(current_snapshot["basis_energies"], dtype=float)
    final_thermal_weights = _to_host_array(current_snapshot["thermal_weights"], dtype=float)
    final_order = np.argsort(final_basis_energies, kind="stable")
    details = {
        "beta": float(beta_value),
        "temperature": float(temperature_value),
        "array_backend": str(array_backend_name),
        "basis_occupations": [[int(v) for v in occ] for occ in basis_occ],
        "hf_occupation": [int(v) for v in hf_occ],
        "include_identity": bool(include_identity),
        "pool_type": pool_type_canonical,
        "ansatz_type": ansatz_type,
        "initial_free_energy": float(initial_snapshot["free_energy"]),
        "initial_entropy": float(initial_snapshot["entropy"]),
        "initial_basis_energies": initial_basis_energies.copy(),
        "initial_thermal_weights": initial_thermal_weights.copy(),
        "final_free_energy": float(current_snapshot["free_energy"]),
        "final_entropy": float(current_snapshot["entropy"]),
        "final_log_partition": float(current_snapshot["log_partition"]),
        "final_basis_energies": final_basis_energies.copy(),
        "final_thermal_weights": final_thermal_weights.copy(),
        "sorted_basis_energies": final_basis_energies[final_order].copy(),
        "sorted_basis_occupations": [basis_occ[int(idx)] for idx in final_order],
        "sorted_thermal_weights": final_thermal_weights[final_order].copy(),
        "history": history,
    }

    if return_details:
        return params, ash_excitation, free_energies, details
    return params, ash_excitation, free_energies


def _qsceom_projected_spectrum(
    *,
    H_dense,
    qubits: int,
    active_electrons: int,
    params,
    ash_excitation,
    ansatz_type: str,
    include_identity: bool,
    basis_occupations,
    max_roots: int,
    array_backend: str,
):
    """Compute the analytic projected qscEOM spectrum for a fixed Hamiltonian."""
    np, qml, _minimize = _require_dependencies()
    xp, array_backend_name, _using_gpu = _resolve_array_module(
        array_backend,
        context="standalone qsceom",
    )
    basis_occ, _hf_occ = _build_truncated_basis_occupations(
        active_electrons=int(active_electrons),
        qubits=int(qubits),
        include_identity=bool(include_identity),
        basis_occupations=basis_occupations,
    )
    basis_columns = _build_basis_columns(basis_occ, qubits=int(qubits), np=xp)
    basis_states = _apply_ansatz_batch(
        basis_columns,
        params,
        ash_excitation,
        ansatz_type=ansatz_type,
        qubits=int(qubits),
        qml=qml,
        np=xp,
    )
    h_dense_backend = xp.asarray(H_dense, dtype=complex)
    projected = basis_states.conj().T @ (h_dense_backend @ basis_states)
    projected = 0.5 * (projected + projected.conj().T)
    eigvals, eigvecs = xp.linalg.eigh(projected)
    order = xp.argsort(xp.real(eigvals))
    eigvals_sorted = _to_host_array(xp.real(eigvals[order]), dtype=float)
    eigvecs_sorted = _to_host_array(eigvecs[:, order], dtype=complex)
    if max_roots > 0:
        eigvals_sorted = eigvals_sorted[: int(max_roots)]
        eigvecs_sorted = eigvecs_sorted[:, : int(max_roots)]
    details = {
        "array_backend": str(array_backend_name),
        "projected_matrix": _to_host_array(projected, dtype=complex),
        "basis_states": _to_host_array(basis_states, dtype=complex),
        "eigenvectors": eigvecs_sorted,
        "basis_occupations": [list(int(v) for v in occ) for occ in basis_occ],
    }
    return eigvals_sorted, details


def qsceom(
    symbols: Sequence[str],
    geometry,
    active_electrons: int,
    active_orbitals: int,
    charge: int,
    params=None,
    ash_excitation=None,
    *,
    ansatz=None,
    ansatz_type: Optional[str] = None,
    basis: str = "sto-3g",
    method: str = "pyscf",
    include_identity: bool = True,
    basis_occupations=None,
    max_roots: int = 0,
    array_backend: str = "auto",
    return_details: bool = False,
):
    """Compute analytic qscEOM eigenvalues from an ansatz state."""
    np, _qml, _minimize = _require_dependencies()

    inferred_ansatz_type = None
    if ansatz is not None:
        try:
            params_from_ansatz, ash_excitation_from_ansatz, _history = ansatz
        except Exception as exc:
            raise ValueError(
                "ansatz must be a tuple like (params, ash_excitation, history_or_energies)"
            ) from exc
        params = params_from_ansatz
        ash_excitation = ash_excitation_from_ansatz
        inferred_ansatz_type = getattr(ash_excitation_from_ansatz, "ansatz_type", None)

    if params is None:
        params = np.zeros(0, dtype=float)
    else:
        params = np.asarray(params, dtype=float)
    if ash_excitation is None:
        ash_excitation = ExcitationList(ansatz_type=_normalize_ansatz_type(ansatz_type), pool_type="external")
    else:
        ash_excitation = [list(int(v) for v in excitation) for excitation in ash_excitation]

    if ansatz_type is None:
        ansatz_type = inferred_ansatz_type
    ansatz_type = _normalize_ansatz_type(ansatz_type)

    if len(params) != len(ash_excitation):
        raise ValueError("params and ash_excitation must have the same length")

    _H, H_dense, qubits, _geometry_array = _build_molecular_hamiltonian(
        symbols,
        geometry,
        basis=basis,
        charge=charge,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
        method=method,
    )
    values, details = _qsceom_projected_spectrum(
        H_dense=H_dense,
        qubits=int(qubits),
        active_electrons=int(active_electrons),
        params=params,
        ash_excitation=ash_excitation,
        ansatz_type=ansatz_type,
        include_identity=bool(include_identity),
        basis_occupations=basis_occupations,
        max_roots=int(max_roots),
        array_backend=array_backend,
    )
    if not return_details:
        return values
    return values, details


def tepid_qsceom(
    symbols: Sequence[str],
    geometry,
    *,
    adapt_it: int,
    temperature: Optional[float] = None,
    beta: Optional[float] = None,
    basis: str = "sto-3g",
    charge: int = 0,
    active_electrons: int,
    active_orbitals: int,
    method: str = "pyscf",
    pool_type: str = "fermionic_sd",
    include_identity: bool = True,
    basis_occupations=None,
    gradient_eps: float = 1e-7,
    gradient_tol: Optional[float] = None,
    optimizer_method: str = "BFGS",
    optimizer_maxiter: int = 200,
    qsceom_include_identity: bool = True,
    qsceom_each_iteration: bool = False,
    qsceom_max_roots: int = 0,
    array_backend: str = "auto",
):
    """Run TEPID-ADAPT and then analytic qscEOM on the final ansatz."""
    np, _qml, _minimize = _require_dependencies()

    tepid_params, tepid_excitation, free_energies, tepid_details = tepid_adapt(
        symbols,
        geometry,
        adapt_it=adapt_it,
        temperature=temperature,
        beta=beta,
        basis=basis,
        charge=charge,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
        method=method,
        pool_type=pool_type,
        include_identity=include_identity,
        basis_occupations=basis_occupations,
        gradient_eps=gradient_eps,
        gradient_tol=gradient_tol,
        optimizer_method=optimizer_method,
        optimizer_maxiter=optimizer_maxiter,
        array_backend=array_backend,
        return_details=True,
    )

    _H, H_dense, qubits, _geometry_array = _build_molecular_hamiltonian(
        symbols,
        geometry,
        basis=basis,
        charge=charge,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
        method=method,
    )
    final_values, final_details = _qsceom_projected_spectrum(
        H_dense=H_dense,
        qubits=int(qubits),
        active_electrons=int(active_electrons),
        params=np.asarray(tepid_params, dtype=float),
        ash_excitation=tepid_excitation,
        ansatz_type=str(tepid_details["ansatz_type"]),
        include_identity=bool(qsceom_include_identity),
        basis_occupations=basis_occupations,
        max_roots=int(qsceom_max_roots),
        array_backend=array_backend,
    )

    per_iteration = []
    if qsceom_each_iteration:
        for snapshot in tepid_details["history"]:
            values, _details = _qsceom_projected_spectrum(
                H_dense=H_dense,
                qubits=int(qubits),
                active_electrons=int(active_electrons),
                params=np.asarray(snapshot["params"], dtype=float),
                ash_excitation=snapshot["ash_excitation"],
                ansatz_type=str(tepid_details["ansatz_type"]),
                include_identity=bool(qsceom_include_identity),
                basis_occupations=basis_occupations,
                max_roots=int(qsceom_max_roots),
                array_backend=array_backend,
            )
            per_iteration.append(
                {
                    "iteration": int(snapshot["iteration"]),
                    "free_energy": float(snapshot["free_energy"]),
                    "entropy": float(snapshot["entropy"]),
                    "values": np.asarray(values, dtype=float),
                }
            )

    return {
        "tepid": {
            "params": np.asarray(tepid_params, dtype=float),
            "ash_excitation": tepid_excitation,
            "free_energies": [float(v) for v in free_energies],
            "details": tepid_details,
        },
        "qsceom": {
            "values": np.asarray(final_values, dtype=float),
            "details": final_details,
        },
        "per_iteration": per_iteration,
    }


def _json_default(value):
    """Convert scientific-python scalars to JSON-safe values."""
    np, _qml, _minimize = _require_dependencies()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, tuple)):
        return list(value)
    if isinstance(value, ExcitationList):
        return list(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a JSON file with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)
        handle.write("\n")


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write CSV rows to disk."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a workflow JSON config."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Config must be a JSON object")
    payload["_config_path"] = str(config_path.resolve())
    return payload


def save_ansatz(
    path: str | Path,
    *,
    params,
    ash_excitation,
    ansatz_type: str,
    pool_type: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> None:
    """Save a standalone ansatz file."""
    payload = {
        "params": [float(v) for v in params],
        "ash_excitation": [[int(x) for x in excitation] for excitation in ash_excitation],
        "ansatz_type": str(ansatz_type),
        "pool_type": None if pool_type is None else str(pool_type),
        "metadata": dict(metadata or {}),
    }
    write_json(Path(path), payload)


def load_ansatz(path: str | Path):
    """Load a standalone ansatz file."""
    ansatz_path = Path(path)
    with ansatz_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    params = [float(v) for v in payload["params"]]
    ansatz_type = _normalize_ansatz_type(payload.get("ansatz_type"))
    pool_type = payload.get("pool_type") or ("fermionic_sd" if ansatz_type == "fermionic" else "qubit_excitation")
    excitations = ExcitationList(
        [[int(x) for x in excitation] for excitation in payload["ash_excitation"]],
        ansatz_type=ansatz_type,
        pool_type=str(pool_type),
    )
    return params, excitations, dict(payload.get("metadata") or {})


def _resolve_output_dir(config: Mapping[str, Any], override: Optional[str | Path]) -> Path:
    """Resolve the output directory relative to the config file."""
    config_path = Path(str(config["_config_path"]))
    if override is not None:
        output_dir = Path(override).expanduser()
        if not output_dir.is_absolute():
            output_dir = output_dir.resolve()
    else:
        output_dir = Path(str(config.get("output_dir", "outputs")))
        if not output_dir.is_absolute():
            output_dir = (config_path.parent / output_dir).resolve()
    return output_dir


def _resolve_input_path(config: Mapping[str, Any], path_value: str | Path) -> Path:
    """Resolve an input path relative to the config file."""
    path_obj = Path(path_value)
    if path_obj.is_absolute():
        return path_obj
    return (Path(str(config["_config_path"])).parent / path_obj).resolve()


def _resolve_override_path(path_value: str | Path) -> Path:
    """Resolve a CLI override path relative to the current working directory."""
    path_obj = Path(path_value).expanduser()
    if path_obj.is_absolute():
        return path_obj
    return path_obj.resolve()


def _exact_rows(exact_roots) -> list[dict[str, Any]]:
    """Convert exact roots into CSV rows."""
    return [
        {"state_index": int(idx), "energy_hartree": float(energy)}
        for idx, energy in enumerate(exact_roots)
    ]


def _tepid_history_rows(details: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Flatten TEPID history into CSV rows."""
    rows = []
    for snapshot in details["history"]:
        sorted_energies = snapshot["sorted_basis_energies"]
        e0 = float(sorted_energies[0]) if len(sorted_energies) > 0 else float("nan")
        e1 = float(sorted_energies[1]) if len(sorted_energies) > 1 else float("nan")
        e2 = float(sorted_energies[2]) if len(sorted_energies) > 2 else float("nan")
        rows.append(
            {
                "iteration": int(snapshot["iteration"]),
                "free_energy_hartree": float(snapshot["free_energy"]),
                "entropy": float(snapshot["entropy"]),
                "gradient_abs": float(snapshot["tepid_gradient_abs"]),
                "basis_e0_hartree": e0,
                "basis_e1_hartree": e1,
                "basis_e2_hartree": e2,
                "basis_ee1_hartree": float(e1 - e0),
                "basis_ee2_hartree": float(e2 - e0),
                "selected_pool_index": int(snapshot["selected_pool_index"]),
                "selected_excitation": " ".join(str(int(v)) for v in snapshot["selected_excitation"]),
                "optimizer_success": bool(snapshot["optimizer_success"]),
                "optimizer_status": int(snapshot["optimizer_status"]),
                "optimizer_nit": "" if snapshot["optimizer_nit"] is None else int(snapshot["optimizer_nit"]),
                "optimizer_nfev": "" if snapshot["optimizer_nfev"] is None else int(snapshot["optimizer_nfev"]),
            }
        )
    return rows


def _tepid_basis_rows(details: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Flatten per-iteration TEPID basis energies into CSV rows."""
    rows = []
    for snapshot in details["history"]:
        for state_index, (energy, weight, occ) in enumerate(
            zip(
                snapshot["sorted_basis_energies"],
                snapshot["sorted_thermal_weights"],
                snapshot["sorted_basis_occupations"],
            )
        ):
            rows.append(
                {
                    "iteration": int(snapshot["iteration"]),
                    "state_index": int(state_index),
                    "energy_hartree": float(energy),
                    "thermal_weight": float(weight),
                    "occupation": " ".join(str(int(v)) for v in occ),
                }
            )
    return rows


def _qsceom_rows(values) -> list[dict[str, Any]]:
    """Convert a qscEOM spectrum into CSV rows."""
    rows = []
    ground = float(values[0])
    for idx, energy in enumerate(values):
        rows.append(
            {
                "state_index": int(idx),
                "energy_hartree": float(energy),
                "excitation_energy_hartree": float(energy - ground),
            }
        )
    return rows


def _qsceom_iteration_rows(per_iteration) -> list[dict[str, Any]]:
    """Convert per-iteration qscEOM spectra into CSV rows."""
    rows = []
    for payload in per_iteration:
        values = payload["values"]
        ground = float(values[0])
        for idx, energy in enumerate(values):
            rows.append(
                {
                    "iteration": int(payload["iteration"]),
                    "state_index": int(idx),
                    "energy_hartree": float(energy),
                    "excitation_energy_hartree": float(energy - ground),
                    "tepid_free_energy_hartree": float(payload["free_energy"]),
                    "tepid_entropy": float(payload["entropy"]),
                }
            )
    return rows


def _error_summary(approx_values, exact_roots) -> dict[str, Any]:
    """Compute low-state and excitation-energy errors against exact roots."""
    if exact_roots is None or len(exact_roots) == 0 or len(approx_values) == 0:
        return {}
    n = min(3, len(approx_values), len(exact_roots))
    summary: dict[str, Any] = {
        "n_compared": int(n),
        "ground_state_error_hartree": float(approx_values[0] - exact_roots[0]),
    }
    if n > 1:
        summary["first_excited_error_hartree"] = float(approx_values[1] - exact_roots[1])
        summary["first_excitation_error_hartree"] = float(
            (approx_values[1] - approx_values[0]) - (exact_roots[1] - exact_roots[0])
        )
    if n > 2:
        summary["second_excited_error_hartree"] = float(approx_values[2] - exact_roots[2])
        summary["second_excitation_error_hartree"] = float(
            (approx_values[2] - approx_values[0]) - (exact_roots[2] - exact_roots[0])
        )
    return summary


def run_workflow(
    config: Mapping[str, Any],
    *,
    mode: str,
    output_dir_override: Optional[str | Path] = None,
    ansatz_file_override: Optional[str | Path] = None,
):
    """Run a standalone workflow and write outputs to disk."""
    np, _qml, _minimize = _require_dependencies()

    if mode not in {"tepid", "qsceom", "tepid_qsceom"}:
        raise ValueError("mode must be one of {'tepid', 'qsceom', 'tepid_qsceom'}")

    symbols = config["symbols"]
    geometry = config["geometry"]
    basis = str(config.get("basis", "sto-3g"))
    charge = int(config.get("charge", 0))
    active_electrons = int(config["active_electrons"])
    active_orbitals = int(config["active_orbitals"])
    method = str(config.get("method", "pyscf"))
    array_backend = str(config.get("array_backend", "auto"))
    output_dir = _resolve_output_dir(config, output_dir_override)
    output_dir.mkdir(parents=True, exist_ok=True)

    tepid_cfg = dict(config.get("tepid") or {})
    qsceom_cfg = dict(config.get("qsceom") or {})

    write_json(output_dir / "resolved_config.json", dict(config))

    exact_roots = None
    if bool(qsceom_cfg.get("compute_exact_sector", True)):
        exact_roots = compute_exact_sector_roots(
            symbols,
            geometry,
            basis=basis,
            charge=charge,
            active_electrons=active_electrons,
            active_orbitals=active_orbitals,
            method=method,
        )
        write_csv_rows(output_dir / "exact_sector_fci_roots.csv", _exact_rows(exact_roots))

    if mode == "tepid":
        params, ash_excitation, free_energies, details = tepid_adapt(
            symbols,
            geometry,
            adapt_it=int(tepid_cfg.get("adapt_it", 10)),
            temperature=tepid_cfg.get("temperature"),
            beta=tepid_cfg.get("beta"),
            basis=basis,
            charge=charge,
            active_electrons=active_electrons,
            active_orbitals=active_orbitals,
            method=method,
            pool_type=str(tepid_cfg.get("pool_type", "fermionic_sd")),
            include_identity=bool(tepid_cfg.get("include_identity", True)),
            basis_occupations=tepid_cfg.get("basis_occupations"),
            pool_sample_size=tepid_cfg.get("pool_sample_size"),
            pool_seed=tepid_cfg.get("pool_seed"),
            gradient_eps=float(tepid_cfg.get("gradient_eps", 1e-7)),
            gradient_tol=tepid_cfg.get("gradient_tol"),
            optimizer_method=str(tepid_cfg.get("optimizer_method", "BFGS")),
            optimizer_maxiter=int(tepid_cfg.get("optimizer_maxiter", 200)),
            array_backend=array_backend,
            return_details=True,
        )
        write_csv_rows(output_dir / "tepid_history.csv", _tepid_history_rows(details))
        write_csv_rows(output_dir / "tepid_basis_states.csv", _tepid_basis_rows(details))
        save_ansatz(
            output_dir / "tepid_ansatz.json",
            params=params,
            ash_excitation=ash_excitation,
            ansatz_type=str(details["ansatz_type"]),
            pool_type=str(details["pool_type"]),
            metadata={
                "free_energies": [float(v) for v in free_energies],
                "basis": basis,
                "charge": charge,
                "active_electrons": active_electrons,
                "active_orbitals": active_orbitals,
            },
        )
        summary = {
            "mode": mode,
            "final_free_energy_hartree": float(details["final_free_energy"]),
            "final_entropy": float(details["final_entropy"]),
            "selected_operators": int(len(ash_excitation)),
            "exact_reference": None if exact_roots is None else _error_summary(details["sorted_basis_energies"], exact_roots),
        }
        write_json(output_dir / "summary.json", summary)
        return summary

    if mode == "qsceom":
        params = np.zeros(0, dtype=float)
        ash_excitation = ExcitationList(ansatz_type="fermionic", pool_type="external")
        ansatz_type = _normalize_ansatz_type(qsceom_cfg.get("ansatz_type"))
        if ansatz_file_override is not None:
            params, ash_excitation, _metadata = load_ansatz(_resolve_override_path(ansatz_file_override))
            ansatz_type = getattr(ash_excitation, "ansatz_type", ansatz_type)
        elif qsceom_cfg.get("ansatz_file"):
            params, ash_excitation, _metadata = load_ansatz(_resolve_input_path(config, qsceom_cfg["ansatz_file"]))
            ansatz_type = getattr(ash_excitation, "ansatz_type", ansatz_type)

        values, details = qsceom(
            symbols,
            geometry,
            active_electrons,
            active_orbitals,
            charge,
            params=params,
            ash_excitation=ash_excitation,
            ansatz_type=ansatz_type,
            basis=basis,
            method=method,
            include_identity=bool(qsceom_cfg.get("include_identity", True)),
            basis_occupations=qsceom_cfg.get("basis_occupations"),
            max_roots=int(qsceom_cfg.get("max_roots", 0)),
            array_backend=array_backend,
            return_details=True,
        )
        write_csv_rows(output_dir / "qsceom_spectrum.csv", _qsceom_rows(values))
        summary = {
            "mode": mode,
            "ansatz_type": ansatz_type,
            "n_roots": int(len(values)),
            "ground_state_energy_hartree": float(values[0]),
            "exact_reference": None if exact_roots is None else _error_summary(values, exact_roots),
        }
        write_json(output_dir / "summary.json", summary)
        return summary

    combined = tepid_qsceom(
        symbols,
        geometry,
        adapt_it=int(tepid_cfg.get("adapt_it", 10)),
        temperature=tepid_cfg.get("temperature"),
        beta=tepid_cfg.get("beta"),
        basis=basis,
        charge=charge,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
        method=method,
        pool_type=str(tepid_cfg.get("pool_type", "fermionic_sd")),
        include_identity=bool(tepid_cfg.get("include_identity", True)),
        basis_occupations=tepid_cfg.get("basis_occupations"),
        gradient_eps=float(tepid_cfg.get("gradient_eps", 1e-7)),
        gradient_tol=tepid_cfg.get("gradient_tol"),
        optimizer_method=str(tepid_cfg.get("optimizer_method", "BFGS")),
        optimizer_maxiter=int(tepid_cfg.get("optimizer_maxiter", 200)),
        array_backend=array_backend,
        qsceom_include_identity=bool(qsceom_cfg.get("include_identity", True)),
        qsceom_each_iteration=bool(qsceom_cfg.get("each_iteration", False)),
        qsceom_max_roots=int(qsceom_cfg.get("max_roots", 0)),
    )

    tepid_payload = combined["tepid"]
    qsceom_payload = combined["qsceom"]
    write_csv_rows(output_dir / "tepid_history.csv", _tepid_history_rows(tepid_payload["details"]))
    write_csv_rows(output_dir / "tepid_basis_states.csv", _tepid_basis_rows(tepid_payload["details"]))
    save_ansatz(
        output_dir / "tepid_ansatz.json",
        params=tepid_payload["params"],
        ash_excitation=tepid_payload["ash_excitation"],
        ansatz_type=str(tepid_payload["details"]["ansatz_type"]),
        pool_type=str(tepid_payload["details"]["pool_type"]),
        metadata={
            "free_energies": tepid_payload["free_energies"],
            "basis": basis,
            "charge": charge,
            "active_electrons": active_electrons,
            "active_orbitals": active_orbitals,
        },
    )
    write_csv_rows(output_dir / "qsceom_spectrum.csv", _qsceom_rows(qsceom_payload["values"]))
    if combined["per_iteration"]:
        write_csv_rows(output_dir / "tepid_qsceom_by_iteration.csv", _qsceom_iteration_rows(combined["per_iteration"]))

    summary = {
        "mode": mode,
        "tepid_final_free_energy_hartree": float(tepid_payload["details"]["final_free_energy"]),
        "tepid_final_entropy": float(tepid_payload["details"]["final_entropy"]),
        "tepid_selected_operators": int(len(tepid_payload["ash_excitation"])),
        "qsceom_ground_state_energy_hartree": float(qsceom_payload["values"][0]),
        "qsceom_exact_reference": None if exact_roots is None else _error_summary(qsceom_payload["values"], exact_roots),
        "qsceom_each_iteration": bool(qsceom_cfg.get("each_iteration", False)),
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def _apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Apply simple CLI overrides on top of a loaded config."""
    updated = dict(config)
    tepid_cfg = dict(updated.get("tepid") or {})
    qsceom_cfg = dict(updated.get("qsceom") or {})

    if args.output_dir is not None:
        updated["output_dir"] = str(args.output_dir)
    if args.adapt_it is not None:
        tepid_cfg["adapt_it"] = int(args.adapt_it)
    if args.beta is not None:
        tepid_cfg["beta"] = float(args.beta)
        tepid_cfg.pop("temperature", None)
    if args.temperature is not None:
        tepid_cfg["temperature"] = float(args.temperature)
        tepid_cfg.pop("beta", None)
    if args.pool_type is not None:
        tepid_cfg["pool_type"] = str(args.pool_type)
    if args.optimizer_maxiter is not None:
        tepid_cfg["optimizer_maxiter"] = int(args.optimizer_maxiter)
    if args.qsceom_each_iteration:
        qsceom_cfg["each_iteration"] = True
    if args.qsceom_max_roots is not None:
        qsceom_cfg["max_roots"] = int(args.qsceom_max_roots)
    if args.array_backend is not None:
        updated["array_backend"] = str(args.array_backend)

    updated["tepid"] = tepid_cfg
    updated["qsceom"] = qsceom_cfg
    return updated


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Standalone TEPID/qscEOM workflows")
    parser.add_argument("--config", type=Path, required=True, help="Path to workflow JSON config")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["tepid", "qsceom", "tepid_qsceom"],
        help="Workflow mode",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional output directory override")
    parser.add_argument("--ansatz-file", type=Path, default=None, help="Optional ansatz file for qsceom mode")
    parser.add_argument("--adapt-it", type=int, default=None, help="Override TEPID iteration count")
    parser.add_argument("--beta", type=float, default=None, help="Override inverse temperature")
    parser.add_argument("--temperature", type=float, default=None, help="Override temperature")
    parser.add_argument("--pool-type", type=str, default=None, help="Override TEPID pool type")
    parser.add_argument(
        "--optimizer-maxiter",
        type=int,
        default=None,
        help="Override per-iteration optimizer maximum iterations",
    )
    parser.add_argument(
        "--qsceom-each-iteration",
        action="store_true",
        help="When mode=tepid_qsceom, compute qscEOM after every TEPID iteration",
    )
    parser.add_argument(
        "--qsceom-max-roots",
        type=int,
        default=None,
        help="Optional qscEOM root truncation; <=0 keeps the full spectrum",
    )
    parser.add_argument(
        "--array-backend",
        type=str,
        default=None,
        choices=["auto", "numpy", "cupy"],
        help="Dense array backend for standalone TEPID/qscEOM workflows",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entrypoint."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    config = load_config(args.config)
    config = _apply_cli_overrides(config, args)
    summary = run_workflow(
        config,
        mode=str(args.mode),
        output_dir_override=args.output_dir,
        ansatz_file_override=args.ansatz_file,
    )

    output_dir = _resolve_output_dir(config, args.output_dir)
    print(f"Standalone workflow finished: mode={args.mode}")
    print(f"Output directory: {output_dir}")
    for key, value in summary.items():
        print(f"{key}: {value}")
