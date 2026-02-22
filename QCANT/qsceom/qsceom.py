"""qscEOM implementation.

This module was previously stored under ``QCANT/tests`` as an experiment/script.
It has been promoted into the package so it can be imported and documented.

Notes
-----
This code depends on optional scientific Python packages (e.g. PennyLane).
Imports are intentionally performed inside functions so that importing QCANT
does not require these optional dependencies.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
from typing import Any, Optional, Sequence, Tuple

from .excitations import inite


def _resolve_worker_count(max_workers: Optional[int]) -> int:
    """Resolve worker thread count for optional parallel sections."""
    if max_workers is not None:
        return int(max_workers)
    detected = os.cpu_count()
    return int(detected if detected is not None else 1)


def _resolve_chunk_size(
    *,
    total_items: int,
    worker_count: int,
    user_chunk_size: Optional[int],
) -> int:
    """Resolve a chunk size for batched worker submission."""
    if total_items <= 0:
        return 1
    if user_chunk_size is not None:
        return int(user_chunk_size)
    return max(1, (int(total_items) + int(worker_count) - 1) // int(worker_count))


def _iter_chunks(items, chunk_size: int):
    """Yield fixed-size chunks from a sequence-like container."""
    for start in range(0, len(items), int(chunk_size)):
        yield items[start : start + int(chunk_size)]


def _resolve_parallel_backend(parallel_backend: str) -> str:
    """Resolve optional parallel backend selection."""
    if parallel_backend == "auto":
        if os.name == "nt":
            return "thread"
        return "process"
    return parallel_backend


_QSCEOM_WORKER_STATE = {}


def _qsceom_worker_init(payload):
    """Initialize persistent worker-local state for qscEOM matrix evaluation."""
    import numpy as np
    import pennylane as qml

    def _make_device(name: Optional[str], wires: int, shots: int):
        kwargs = {}
        if shots > 0:
            kwargs["shots"] = shots
        if name is not None:
            return qml.device(name, wires=wires, **kwargs)
        try:
            return qml.device("lightning.qubit", wires=wires, **kwargs)
        except Exception:
            return qml.device("default.qubit", wires=wires, **kwargs)

    qubits = int(payload["qubits"])
    shots = int(payload["shots"])
    H = payload["H"]
    params = payload["params"]
    ash_excitation = payload["ash_excitation"]
    list1 = payload["list1"]
    null_state = payload["null_state"]
    device_name = payload["device_name"]

    dev = _make_device(device_name, qubits, shots)

    def _apply_ansatz_local(params_local, ash_local):
        for i, excitations in enumerate(ash_local):
            if len(excitations) == 4:
                qml.FermionicDoubleExcitation(
                    weight=params_local[i],
                    wires1=list(range(excitations[0], excitations[1] + 1)),
                    wires2=list(range(excitations[2], excitations[3] + 1)),
                )
            elif len(excitations) == 2:
                qml.FermionicSingleExcitation(
                    weight=params_local[i],
                    wires=list(range(excitations[0], excitations[1] + 1)),
                )

    @qml.qnode(dev)
    def _diag_by_index(idx):
        occ = list1[int(idx)]
        qml.BasisState(null_state, wires=range(qubits))
        for w in occ:
            qml.X(wires=w)
        _apply_ansatz_local(params, ash_excitation)
        return qml.expval(H)

    @qml.qnode(dev)
    def _offdiag_by_pair(i, j):
        occ1 = list1[int(i)]
        occ2 = list1[int(j)]
        qml.BasisState(null_state, wires=range(qubits))
        for w in occ1:
            qml.X(wires=w)
        first = -1
        for v in occ2:
            if v not in occ1:
                if first == -1:
                    first = v
                    qml.Hadamard(wires=v)
                else:
                    qml.CNOT(wires=[first, v])
        for v in occ1:
            if v not in occ2:
                if first == -1:
                    first = v
                    qml.Hadamard(wires=v)
                else:
                    qml.CNOT(wires=[first, v])
        _apply_ansatz_local(params, ash_excitation)
        return qml.expval(H)

    _QSCEOM_WORKER_STATE.clear()
    _QSCEOM_WORKER_STATE.update(
        {
            "np": np,
            "diag_fn": _diag_by_index,
            "offdiag_fn": _offdiag_by_pair,
        }
    )


def _qsceom_worker_diagonal(chunk_indices):
    """Evaluate diagonal matrix entries for a chunk of state indices."""
    np = _QSCEOM_WORKER_STATE["np"]
    diag_fn = _QSCEOM_WORKER_STATE["diag_fn"]
    out = {}
    for idx in chunk_indices:
        value = diag_fn(idx)
        out[idx] = float(np.real(np.asarray(value).item()))
    return out


def _qsceom_worker_offdiagonal(chunk_pairs):
    """Evaluate raw off-diagonal expectation values for a chunk of pairs."""
    np = _QSCEOM_WORKER_STATE["np"]
    offdiag_fn = _QSCEOM_WORKER_STATE["offdiag_fn"]
    out = {}
    for i, j in chunk_pairs:
        value = offdiag_fn(i, j)
        out[(i, j)] = float(np.real(np.asarray(value).item()))
    return out


def qscEOM(
    symbols: Sequence[str],
    geometry,
    active_electrons: int,
    active_orbitals: int,
    charge: int,
    params=None,
    ash_excitation=None,
    *,
    ansatz: Optional[Tuple[Any, Any, Any]] = None,
    basis: str = "sto-3g",
    method: str = "pyscf",
    shots: int = 0,
    device_name: Optional[str] = None,
    max_states: Optional[int] = None,
    state_seed: Optional[int] = None,
    symmetric: bool = True,
    parallel_matrix: bool = False,
    parallel_backend: str = "auto",
    max_workers: Optional[int] = None,
    matrix_chunk_size: Optional[int] = None,
):
    """Compute qscEOM eigenvalues from an ansatz state.

    Parameters
    ----------
    symbols
        Atomic symbols.
    geometry
        Nuclear coordinates (as an array-like object).
    active_electrons
        Number of active electrons.
    active_orbitals
        Number of active orbitals.
    charge
        Total molecular charge.
    params
        Ansatz parameters.
    ash_excitation
        Excitation list describing the ansatz.
    shots
        If 0, run in analytic mode; otherwise use shot-based estimation.
    device_name
        Optional PennyLane device name (e.g. ``"lightning.qubit"``).
    max_states
        If provided, limit the number of occupation configurations used to
        build the effective matrix.
    state_seed
        Seed for selecting a random subset when ``max_states`` is used.
    symmetric
        If True, compute only the upper-triangular off-diagonal elements and
        mirror them to reduce circuit evaluations.
    parallel_matrix
        If True, evaluate independent qscEOM matrix elements concurrently.
    parallel_backend
        Parallel backend used when ``parallel_matrix=True``:
        ``"process"``, ``"thread"``, or ``"auto"`` (default). ``"auto"``
        selects processes on POSIX and threads on Windows.
    max_workers
        Maximum number of worker threads used when ``parallel_matrix=True``.
        If omitted, ``os.cpu_count()`` is used.
    matrix_chunk_size
        Number of matrix entries per submitted worker task. If omitted, a
        balanced chunk size based on ``max_workers`` is used.

    Returns
    -------
    list
        Sorted eigenvalues for the constructed effective matrix.
    """

    if ansatz is not None:
        try:
            params_from_adapt, ash_excitation_from_adapt, _energies = ansatz
        except Exception as exc:
            raise ValueError(
                "ansatz must be a 3-tuple like (params, ash_excitation, energies) "
                "as returned by QCANT.adapt_vqe"
            ) from exc

        params = params_from_adapt
        ash_excitation = ash_excitation_from_adapt

    if params is None or ash_excitation is None:
        raise TypeError(
            "qscEOM requires either (params, ash_excitation) or ansatz=(params, ash_excitation, energies)."
        )
    if max_states is not None and max_states <= 0:
        raise ValueError("max_states must be > 0")
    if max_workers is not None and max_workers <= 0:
        raise ValueError("max_workers must be > 0")
    if matrix_chunk_size is not None and matrix_chunk_size <= 0:
        raise ValueError("matrix_chunk_size must be > 0")
    if parallel_backend not in {"auto", "thread", "process"}:
        raise ValueError("parallel_backend must be one of {'auto', 'thread', 'process'}")

    try:
        if len(params) != len(ash_excitation):
            raise ValueError
    except Exception as exc:
        raise ValueError("params and ash_excitation must have the same length") from exc

    try:
        import numpy as np
        import pennylane as qml
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "qscEOM requires dependencies. Install at least: "
            "`pip install numpy pennylane`."
        ) from exc

    H, qubits = qml.qchem.molecular_hamiltonian(
        symbols,
        geometry,
        basis=basis,
        method=method,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
        charge=charge,
    )

    hf_state = qml.qchem.hf_state(active_electrons, qubits)
    singles, doubles = qml.qchem.excitations(active_electrons, qubits)
    s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)
    wires = range(qubits)

    null_state = np.zeros(qubits, int)
    list1 = inite(active_electrons, qubits)
    if max_states is not None and max_states < len(list1):
        rng = np.random.default_rng(state_seed)
        indices = rng.choice(len(list1), size=max_states, replace=False)
        list1 = [list1[idx] for idx in sorted(indices)]
    values = []

    def _make_device(name: Optional[str], wires: int):
        kwargs = {}
        if shots > 0:
            kwargs["shots"] = shots
        if name is not None:
            return qml.device(name, wires=wires, **kwargs)
        try:
            return qml.device("lightning.qubit", wires=wires, **kwargs)
        except Exception:
            return qml.device("default.qubit", wires=wires, **kwargs)

    def _to_real_scalar(value):
        """Convert PennyLane/numpy scalar-like values to python float."""
        arr = np.asarray(value)
        return float(np.real(arr).item())

    def _build_circuit_d(dev):
        @qml.qnode(dev)
        def circuit_d_local(params_local, occ, wires, s_wires, d_wires, hf_state_local, ash_local):
            qml.BasisState(hf_state_local, wires=range(qubits))
            for w in occ:
                qml.X(wires=w)
            for i, excitations in enumerate(ash_local):
                if len(excitations) == 4:
                    qml.FermionicDoubleExcitation(
                        weight=params_local[i],
                        wires1=list(range(excitations[0], excitations[1] + 1)),
                        wires2=list(range(excitations[2], excitations[3] + 1)),
                    )
                elif len(excitations) == 2:
                    qml.FermionicSingleExcitation(
                        weight=params_local[i],
                        wires=list(range(excitations[0], excitations[1] + 1)),
                    )
            return qml.expval(H)

        return circuit_d_local

    def _build_circuit_od(dev):
        @qml.qnode(dev)
        def circuit_od_local(params_local, occ1, occ2, wires, s_wires, d_wires, hf_state_local, ash_local):
            qml.BasisState(hf_state_local, wires=range(qubits))
            for w in occ1:
                qml.X(wires=w)
            first = -1
            for v in occ2:
                if v not in occ1:
                    if first == -1:
                        first = v
                        qml.Hadamard(wires=v)
                    else:
                        qml.CNOT(wires=[first, v])
            for v in occ1:
                if v not in occ2:
                    if first == -1:
                        first = v
                        qml.Hadamard(wires=v)
                    else:
                        qml.CNOT(wires=[first, v])
            for i, excitations in enumerate(ash_local):
                if len(excitations) == 4:
                    qml.FermionicDoubleExcitation(
                        weight=params_local[i],
                        wires1=list(range(excitations[0], excitations[1] + 1)),
                        wires2=list(range(excitations[2], excitations[3] + 1)),
                    )
                elif len(excitations) == 2:
                    qml.FermionicSingleExcitation(
                        weight=params_local[i],
                        wires=list(range(excitations[0], excitations[1] + 1)),
                    )
            return qml.expval(H)

        return circuit_od_local

    n_states = len(list1)
    M = np.zeros((n_states, n_states), dtype=float)
    worker_count = _resolve_worker_count(max_workers)
    backend = _resolve_parallel_backend(parallel_backend)
    executor = None
    if parallel_matrix and worker_count > 1:
        if backend == "process":
            payload = {
                "qubits": int(qubits),
                "shots": int(shots),
                "H": H,
                "params": np.asarray(params),
                "ash_excitation": tuple(tuple(int(v) for v in exc) for exc in ash_excitation),
                "list1": tuple(tuple(int(v) for v in occ) for occ in list1),
                "null_state": np.asarray(null_state),
                "device_name": device_name,
            }
            try:
                mp_context = mp.get_context("fork")
            except ValueError:
                mp_context = mp.get_context()
            try:
                executor = ProcessPoolExecutor(
                    max_workers=worker_count,
                    mp_context=mp_context,
                    initializer=_qsceom_worker_init,
                    initargs=(payload,),
                )
            except (PermissionError, OSError, NotImplementedError):
                backend = "thread"
                executor = ThreadPoolExecutor(max_workers=worker_count)
        else:
            executor = ThreadPoolExecutor(max_workers=worker_count)

    def _diagonal_chunk(chunk_indices):
        local_dev = _make_device(device_name, qubits)
        circuit_d_local = _build_circuit_d(local_dev)
        out = {}
        for idx in chunk_indices:
            value = circuit_d_local(params, list1[idx], wires, s_wires, d_wires, null_state, ash_excitation)
            out[idx] = _to_real_scalar(value)
        return out

    def _off_diagonal_chunk(chunk_pairs, diagonal_values):
        local_dev = _make_device(device_name, qubits)
        circuit_od_local = _build_circuit_od(local_dev)
        out = {}
        for i, j in chunk_pairs:
            mtmp = circuit_od_local(
                params,
                list1[i],
                list1[j],
                wires,
                s_wires,
                d_wires,
                null_state,
                ash_excitation,
            )
            value = _to_real_scalar(mtmp) - diagonal_values[i] / 2.0 - diagonal_values[j] / 2.0
            out[(i, j)] = float(value)
        return out

    try:
        diagonal_indices = list(range(n_states))
        if executor is not None and n_states > 1:
            diag_chunk_size = _resolve_chunk_size(
                total_items=n_states,
                worker_count=worker_count,
                user_chunk_size=matrix_chunk_size,
            )
            diagonal_map = {}
            if backend == "process":
                futures = [
                    executor.submit(_qsceom_worker_diagonal, chunk)
                    for chunk in _iter_chunks(diagonal_indices, diag_chunk_size)
                ]
            else:
                futures = [
                    executor.submit(_diagonal_chunk, chunk)
                    for chunk in _iter_chunks(diagonal_indices, diag_chunk_size)
                ]
            for future in as_completed(futures):
                diagonal_map.update(future.result())
            for i in diagonal_indices:
                M[i, i] = diagonal_map[i]
        else:
            dev = _make_device(device_name, qubits)
            circuit_d = _build_circuit_d(dev)
            for i in diagonal_indices:
                M[i, i] = _to_real_scalar(
                    circuit_d(params, list1[i], wires, s_wires, d_wires, null_state, ash_excitation)
                )

        if symmetric:
            pair_indices = [(i, j) for i in range(n_states) for j in range(i + 1, n_states)]
        else:
            pair_indices = [(i, j) for i in range(n_states) for j in range(n_states) if i != j]

        if executor is not None and len(pair_indices) > 1:
            pair_chunk_size = _resolve_chunk_size(
                total_items=len(pair_indices),
                worker_count=worker_count,
                user_chunk_size=matrix_chunk_size,
            )
            pair_map = {}
            if backend == "process":
                futures = [
                    executor.submit(_qsceom_worker_offdiagonal, chunk)
                    for chunk in _iter_chunks(pair_indices, pair_chunk_size)
                ]
            else:
                futures = [
                    executor.submit(_off_diagonal_chunk, chunk, M.diagonal().copy())
                    for chunk in _iter_chunks(pair_indices, pair_chunk_size)
                ]
            for future in as_completed(futures):
                pair_map.update(future.result())

            for i, j in pair_indices:
                if backend == "process":
                    value = pair_map[(i, j)] - M[i, i] / 2.0 - M[j, j] / 2.0
                else:
                    value = pair_map[(i, j)]
                M[i, j] = value
                if symmetric:
                    M[j, i] = value
        else:
            dev = _make_device(device_name, qubits)
            circuit_od = _build_circuit_od(dev)
            if symmetric:
                for i in range(n_states):
                    for j in range(i + 1, n_states):
                        mtmp = circuit_od(
                            params,
                            list1[i],
                            list1[j],
                            wires,
                            s_wires,
                            d_wires,
                            null_state,
                            ash_excitation,
                        )
                        value = _to_real_scalar(mtmp) - M[i, i] / 2.0 - M[j, j] / 2.0
                        M[i, j] = value
                        M[j, i] = value
            else:
                for i in range(n_states):
                    for j in range(n_states):
                        if i != j:
                            mtmp = circuit_od(
                                params,
                                list1[i],
                                list1[j],
                                wires,
                                s_wires,
                                d_wires,
                                null_state,
                                ash_excitation,
                            )
                            M[i, j] = _to_real_scalar(mtmp) - M[i, i] / 2.0 - M[j, j] / 2.0
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    eig, _ = np.linalg.eig(M)
    values.append(np.sort(eig))

    return values
