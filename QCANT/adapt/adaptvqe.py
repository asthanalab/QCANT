"""ADAPT-VQE implementation.

This module was previously stored under ``QCANT/tests`` as an experiment/script.
It has been promoted into the package so it can be imported and documented.

Notes
-----
This code uses optional heavy dependencies (PySCF, PennyLane, SciPy, etc.).
Imports are performed inside the main function so that importing QCANT does not
require these dependencies.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
from typing import Optional, Sequence


def _validate_inputs(
    symbols: Sequence[str],
    geometry,
    shots: Optional[int],
    commutator_shots: Optional[int],
    commutator_mode: str,
    hamiltonian_cutoff: float,
    pool_sample_size: Optional[int],
    max_workers: Optional[int],
    gradient_chunk_size: Optional[int],
    parallel_backend: str,
) -> int:
    """Validate user-provided inputs for the ADAPT-VQE algorithm.

    Returns
    -------
    int
        The number of atoms.

    Raises
    ------
    ValueError
        If any of the input parameters are invalid.
    """
    if len(symbols) == 0:
        raise ValueError("symbols must be non-empty")
    if shots is not None and shots < 0:
        raise ValueError("shots must be >= 0")
    if commutator_shots is not None and commutator_shots < 0:
        raise ValueError("commutator_shots must be >= 0")
    if commutator_mode not in {"ansatz", "statevec"}:
        raise ValueError("commutator_mode must be 'ansatz' or 'statevec'")
    if hamiltonian_cutoff < 0:
        raise ValueError("hamiltonian_cutoff must be >= 0")
    if pool_sample_size is not None and pool_sample_size <= 0:
        raise ValueError("pool_sample_size must be > 0")
    if max_workers is not None and max_workers <= 0:
        raise ValueError("max_workers must be > 0")
    if gradient_chunk_size is not None and gradient_chunk_size <= 0:
        raise ValueError("gradient_chunk_size must be > 0")
    if parallel_backend not in {"auto", "thread", "process"}:
        raise ValueError("parallel_backend must be one of {'auto', 'thread', 'process'}")

    try:
        n_atoms = len(symbols)
        if len(geometry) != n_atoms:
            raise ValueError
    except Exception as exc:
        raise ValueError("geometry must have the same length as symbols") from exc

    return n_atoms


def _resolve_worker_count(max_workers: Optional[int]) -> int:
    """Resolve the number of worker threads for optional parallel sections."""
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
    """Resolve chunk size for batched worker submission."""
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


_ADAPT_WORKER_STATE = {}


def _adapt_worker_init(payload):
    """Initialize persistent worker-local state for process-based ADAPT scoring."""
    import numpy as np
    import pennylane as qml

    def _make_device(name: Optional[str], wires: int, device_shots: Optional[int]):
        kwargs = {}
        if device_shots is not None and device_shots > 0:
            kwargs["shots"] = device_shots
        if name is not None:
            return qml.device(name, wires=wires, **kwargs)
        try:
            return qml.device("lightning.qubit", wires=wires, **kwargs)
        except Exception:
            return qml.device("default.qubit", wires=wires, **kwargs)

    qubits = int(payload["qubits"])
    H = payload["H"]
    hf_state = payload["hf_state"]
    operator_pool_ops = payload["operator_pool_ops"]
    comm_shots = payload["comm_shots"]
    commutator_mode = payload["commutator_mode"]
    commutator_debug = bool(payload["commutator_debug"])
    device_name = payload["device_name"]

    def _apply_ansatz_local(hf_state_local, ash_excitation_local, params_local):
        qml.BasisState(hf_state_local, wires=range(qubits))
        for i, excitation in enumerate(ash_excitation_local):
            if len(excitation) == 4:
                qml.FermionicDoubleExcitation(
                    weight=params_local[i],
                    wires1=list(range(excitation[0], excitation[1] + 1)),
                    wires2=list(range(excitation[2], excitation[3] + 1)),
                )
            elif len(excitation) == 2:
                qml.FermionicSingleExcitation(
                    weight=params_local[i],
                    wires=list(range(excitation[0], excitation[1] + 1)),
                )

    comm_from_ansatz = None
    comm_from_state = None

    if commutator_mode == "statevec" or commutator_debug:
        dev_state = _make_device(device_name, qubits, None)

        @qml.qnode(dev_state)
        def comm_from_state(state, w_local):
            qml.StatePrep(state, wires=range(qubits))
            return qml.expval(qml.commutator(H, w_local))

    if commutator_mode == "ansatz" or commutator_debug:
        dev_ansatz = _make_device(device_name, qubits, comm_shots)

        @qml.qnode(dev_ansatz)
        def comm_from_ansatz(params_local, ash_local, hf_state_local, w_local):
            _apply_ansatz_local(hf_state_local, ash_local, params_local)
            return qml.expval(qml.commutator(H, w_local))

    _ADAPT_WORKER_STATE.clear()
    _ADAPT_WORKER_STATE.update(
        {
            "np": np,
            "commutator_mode": commutator_mode,
            "commutator_debug": commutator_debug,
            "operator_pool_ops": operator_pool_ops,
            "hf_state": hf_state,
            "comm_from_ansatz": comm_from_ansatz,
            "comm_from_state": comm_from_state,
        }
    )


def _adapt_worker_eval_chunk(chunk_positions, params_eval, ash_eval, state_eval):
    """Evaluate commutator scores for a candidate chunk in a process worker."""
    np = _ADAPT_WORKER_STATE["np"]
    commutator_mode = _ADAPT_WORKER_STATE["commutator_mode"]
    commutator_debug = _ADAPT_WORKER_STATE["commutator_debug"]
    operator_pool_ops = _ADAPT_WORKER_STATE["operator_pool_ops"]
    hf_state = _ADAPT_WORKER_STATE["hf_state"]
    comm_from_ansatz = _ADAPT_WORKER_STATE["comm_from_ansatz"]
    comm_from_state = _ADAPT_WORKER_STATE["comm_from_state"]

    scores = {}
    diffs = {}
    for position, idx in chunk_positions:
        w = operator_pool_ops[idx]
        if commutator_mode == "statevec":
            exp_used = comm_from_state(state_eval, w)
        else:
            exp_used = comm_from_ansatz(params_eval, ash_eval, hf_state, w)
        scores[position] = float(np.abs(2.0 * exp_used))

        if commutator_debug:
            if commutator_mode == "statevec":
                exp_other = comm_from_ansatz(params_eval, ash_eval, hf_state, w)
            else:
                exp_other = comm_from_state(state_eval, w)
            diffs[position] = float(np.abs(exp_used - exp_other))

    return scores, diffs


def adapt_vqe(
    symbols: Sequence[str],
    geometry,
    *,
    adapt_it: int,
    basis: str = "sto-6g",
    charge: int = 0,
    spin: int = 0,
    active_electrons: int,
    active_orbitals: int,
    device_name: Optional[str] = None,
    shots: Optional[int] = None,
    commutator_shots: Optional[int] = None,
    commutator_mode: str = "ansatz",
    commutator_debug: bool = False,
    hamiltonian_cutoff: float = 1e-20,
    pool_sample_size: Optional[int] = None,
    pool_seed: Optional[int] = None,
    parallel_gradients: bool = False,
    parallel_backend: str = "auto",
    max_workers: Optional[int] = None,
    gradient_chunk_size: Optional[int] = None,
    optimizer_method: str = "BFGS",
    optimizer_maxiter: int = 100_000_000,
):
    """Run an ADAPT-style VQE loop for a user-specified molecular geometry.

    The core ADAPT loop selects operators from a singles+doubles pool based on
    commutator magnitude, then optimizes the ansatz parameters at each
    iteration.

    Parameters
    ----------
    symbols
        Atomic symbols, e.g. ``["H", "H"]``.
    geometry
        Nuclear coordinates, array-like with shape ``(n_atoms, 3)`` in Angstrom.
    adapt_it
        Number of ADAPT iterations.
    basis
        Basis set name understood by PySCF (e.g. ``"sto-3g"``, ``"sto-6g"``).
    charge
        Total molecular charge.
    spin
        Spin multiplicity parameter used by PySCF as ``2S`` (e.g. 0 for singlet).
    active_electrons
        Number of active electrons in the CASCI reference.
    active_orbitals
        Number of active orbitals in the CASCI reference.
    shots
        If provided and > 0, run with shot-based sampling on the chosen device.
    commutator_shots
        If provided, override the shot count for commutator evaluations.
    commutator_mode
        ``"ansatz"`` uses the ansatz circuit to evaluate commutators; ``"statevec"``
        prepares the current statevector via ``qml.StatePrep`` before measuring.
    commutator_debug
        If True, compute both commutator modes per operator and report the
        maximum absolute difference per ADAPT iteration.
    hamiltonian_cutoff
        Drop Hamiltonian terms with absolute value below this cutoff when
        building the fermionic operator.
    pool_sample_size
        If provided, randomly sample this many operators from the pool per
        ADAPT iteration to reduce commutator evaluations.
    pool_seed
        Seed for the operator-pool sampler.
    parallel_gradients
        If True, evaluate candidate ADAPT commutators concurrently.
    parallel_backend
        Parallel backend used when ``parallel_gradients=True``:
        ``"process"``, ``"thread"``, or ``"auto"`` (default). ``"auto"``
        selects processes on POSIX and threads on Windows.
    max_workers
        Maximum number of worker threads used when ``parallel_gradients=True``.
        If omitted, ``os.cpu_count()`` is used.
    gradient_chunk_size
        Number of candidate operators per submitted worker task. If omitted, a
        balanced chunk size based on ``max_workers`` is used.
    optimizer_method
        SciPy optimization method (e.g. ``"BFGS"``, ``"COBYLA"``, ``"Nelder-Mead"``).

    Returns
    -------
    tuple
        ``(params, ash_excitation, energies)`` as produced by the optimization.

    Raises
    ------
    ValueError
        If ``symbols``/``geometry`` sizes are inconsistent.
    ImportError
        If required optional dependencies are not installed.
    """
    # pylint: disable=too-many-locals, too-many-statements, too-many-branches

    # --------------------------------------------------------------------------
    # 1. Input validation and dependency imports.
    # --------------------------------------------------------------------------
    n_atoms = _validate_inputs(
        symbols,
        geometry,
        shots,
        commutator_shots,
        commutator_mode,
        hamiltonian_cutoff,
        pool_sample_size,
        max_workers,
        gradient_chunk_size,
        parallel_backend,
    )

    try:
        import re
        import warnings

        import numpy as np
        import pennylane as qml
        import pyscf
        from pennylane import numpy as pnp
        from pyscf import gto, mcscf, scf
        from scipy.optimize import minimize

        warnings.filterwarnings("ignore")
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "adapt_vqe requires dependencies. Install at least: "
            "`pip install numpy scipy pennylane pyscf` "
            "(and optionally a faster PennyLane device backend, e.g. `pip install pennylane-lightning`)."
        ) from exc

    # --------------------------------------------------------------------------
    # 2. Device setup.
    # --------------------------------------------------------------------------
    def _make_device(name: Optional[str], wires: int, device_shots: Optional[int]):
        """Create a PennyLane device with optional shot-based execution."""
        kwargs = {}
        if device_shots is not None and device_shots > 0:
            kwargs["shots"] = device_shots
        if name is not None:
            return qml.device(name, wires=wires, **kwargs)
        # Backwards-compatible preference for lightning if available.
        try:
            return qml.device("lightning.qubit", wires=wires, **kwargs)
        except Exception:
            return qml.device("default.qubit", wires=wires, **kwargs)

    # Build the molecule from user-provided symbols/geometry.
    # PySCF accepts either a multiline string or a list of (symbol, (x,y,z)).
    atom = [(symbols[i], tuple(float(x) for x in geometry[i])) for i in range(n_atoms)]

    # --------------------------------------------------------------------------
    # 3. Reference CASCI calculation using PySCF.
    # --------------------------------------------------------------------------
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

    mycas_ref = mcscf.CASCI(mf_ref, active_orbitals, active_electrons)
    h1ecas, ecore = mycas_ref.get_h1eff(mf_ref.mo_coeff)
    h2ecas = mycas_ref.get_h2eff(mf_ref.mo_coeff)

    en = mycas_ref.kernel()
    print("Ref.CASCI energy:", en[0])

    # --------------------------------------------------------------------------
    # 4. Hamiltonian construction.
    # --------------------------------------------------------------------------
    ncas = int(mycas_ref.ncas)
    two_mo = pyscf.ao2mo.restore("1", h2ecas, norb=ncas)
    two_mo = np.swapaxes(two_mo, 1, 3)

    one_mo = h1ecas
    core_constant = np.array([ecore])

    H_fermionic = qml.qchem.fermionic_observable(
        core_constant, one_mo, two_mo, cutoff=hamiltonian_cutoff
    )
    H = qml.jordan_wigner(H_fermionic)

    qubits = 2 * ncas
    active_electrons = sum(mycas_ref.nelecas)

    energies = []
    ash_excitation = []

    hf_state = qml.qchem.hf_state(active_electrons, qubits)

    # --------------------------------------------------------------------------
    # 5. Quantum circuit and cost function setup.
    # --------------------------------------------------------------------------
    comm_shots = shots if commutator_shots is None else commutator_shots
    if commutator_mode == "statevec" and comm_shots is not None and comm_shots > 0:
        raise ValueError("commutator_mode='statevec' requires analytic commutator_shots")
    if commutator_debug and comm_shots is not None and comm_shots > 0:
        raise ValueError("commutator_debug requires analytic commutator_shots")
    dev_comm = _make_device(device_name, qubits, comm_shots)
    dev = _make_device(device_name, qubits, shots)
    dev_state = None
    dev_comm_state = None

    def _apply_ansatz(hf_state, ash_excitation, params):
        """Apply the current ansatz to the Hartree-Fock state."""
        qml.BasisState(hf_state, wires=range(qubits))
        for i, excitation in enumerate(ash_excitation):
            if len(ash_excitation[i]) == 4:
                qml.FermionicDoubleExcitation(
                    weight=params[i],
                    wires1=list(range(ash_excitation[i][0], ash_excitation[i][1] + 1)),
                    wires2=list(range(ash_excitation[i][2], ash_excitation[i][3] + 1)),
                )
            elif len(ash_excitation[i]) == 2:
                qml.FermionicSingleExcitation(
                    weight=params[i],
                    wires=list(range(ash_excitation[i][0], ash_excitation[i][1] + 1)),
                )

    @qml.qnode(dev_comm)
    def commutator_expectation(params, ash_excitation, hf_state, H, w):
        """Compute the expectation value of a commutator."""
        _apply_ansatz(hf_state, ash_excitation, params)
        res = qml.commutator(H, w)
        return qml.expval(res)

    if commutator_mode == "statevec" or commutator_debug:
        dev_state = _make_device(device_name, qubits, None)
        dev_comm_state = _make_device(device_name, qubits, None)

        @qml.qnode(dev_state)
        def current_state(params, ash_excitation, hf_state):
            """Return the current statevector."""
            _apply_ansatz(hf_state, ash_excitation, params)
            return qml.state()

        @qml.qnode(dev_comm_state)
        def commutator_expectation_state(state, H, w):
            """Compute the expectation value of a commutator from a statevector."""
            qml.StatePrep(state, wires=range(qubits))
            res = qml.commutator(H, w)
            return qml.expval(res)

    @qml.qnode(dev)
    def ash(params, ash_excitation, hf_state, H):
        """Compute the expectation value of the Hamiltonian."""
        _apply_ansatz(hf_state, ash_excitation, params)
        return qml.expval(H)

    def cost(params):
        """Cost function for the optimizer."""
        return float(np.real(ash(params, ash_excitation, hf_state, H)))

    # --------------------------------------------------------------------------
    # 6. ADAPT-VQE loop.
    # --------------------------------------------------------------------------
    singles, doubles = qml.qchem.excitations(active_electrons, qubits)
    op1 = [qml.fermi.FermiWord({(0, x[0]): "+", (1, x[1]): "-"}) for x in singles]
    op2 = [
        qml.fermi.FermiWord({(0, x[0]): "+", (1, x[1]): "+", (2, x[2]): "-", (3, x[3]): "-"})
        for x in doubles
    ]
    operator_pool = op1 + op2
    operator_pool_ops = [qml.fermi.jordan_wigner(op) for op in operator_pool]
    params = pnp.zeros(len(ash_excitation), requires_grad=True)
    rng = np.random.default_rng(pool_seed)

    worker_count = _resolve_worker_count(max_workers)
    backend = _resolve_parallel_backend(parallel_backend)
    executor = None
    if parallel_gradients and worker_count > 1:
        if backend == "process":
            worker_payload = {
                "qubits": int(qubits),
                "H": H,
                "hf_state": np.asarray(hf_state),
                "operator_pool_ops": tuple(operator_pool_ops),
                "comm_shots": comm_shots,
                "commutator_mode": commutator_mode,
                "commutator_debug": bool(commutator_debug),
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
                    initializer=_adapt_worker_init,
                    initargs=(worker_payload,),
                )
            except (PermissionError, OSError, NotImplementedError):
                backend = "thread"
                executor = ThreadPoolExecutor(max_workers=worker_count)
        else:
            executor = ThreadPoolExecutor(max_workers=worker_count)

    def _evaluate_commutator_chunk(chunk_positions, params_eval, ash_eval, state_eval):
        """Evaluate commutators for a chunk of candidate operator positions."""
        scores = {}
        diffs = {}
        comm_from_state = None
        comm_from_ansatz = None

        if commutator_mode == "statevec":
            local_dev_state = _make_device(device_name, qubits, None)

            @qml.qnode(local_dev_state)
            def comm_from_state(state, H_local, w_local):
                qml.StatePrep(state, wires=range(qubits))
                return qml.expval(qml.commutator(H_local, w_local))

            if commutator_debug:
                local_dev_ansatz = _make_device(device_name, qubits, comm_shots)

                @qml.qnode(local_dev_ansatz)
                def comm_from_ansatz(params_local, ash_local, hf_state_local, H_local, w_local):
                    _apply_ansatz(hf_state_local, ash_local, params_local)
                    return qml.expval(qml.commutator(H_local, w_local))
        else:
            local_dev_ansatz = _make_device(device_name, qubits, comm_shots)

            @qml.qnode(local_dev_ansatz)
            def comm_from_ansatz(params_local, ash_local, hf_state_local, H_local, w_local):
                _apply_ansatz(hf_state_local, ash_local, params_local)
                return qml.expval(qml.commutator(H_local, w_local))

            if commutator_debug:
                local_dev_state = _make_device(device_name, qubits, None)

                @qml.qnode(local_dev_state)
                def comm_from_state(state, H_local, w_local):
                    qml.StatePrep(state, wires=range(qubits))
                    return qml.expval(qml.commutator(H_local, w_local))

        for position, idx in chunk_positions:
            w = operator_pool_ops[idx]
            if commutator_mode == "statevec":
                exp_used = comm_from_state(state_eval, H, w)
            else:
                exp_used = comm_from_ansatz(params_eval, ash_eval, hf_state, H, w)

            scores[position] = float(np.abs(2.0 * exp_used))

            if commutator_debug:
                if commutator_mode == "statevec":
                    exp_other = comm_from_ansatz(params_eval, ash_eval, hf_state, H, w)
                else:
                    exp_other = comm_from_state(state_eval, H, w)
                diffs[position] = float(np.abs(exp_used - exp_other))

        return scores, diffs

    try:
        for j in range(adapt_it):
            print("The adapt iteration now is", j, flush=True)

            # ------------------------------------------------------------------
            # 6a. Select the next operator for the ansatz.
            # ------------------------------------------------------------------
            max_value = float("-inf")
            max_operator = None
            max_diff = 0.0
            state_for_comm = None
            if commutator_mode == "statevec" or commutator_debug:
                state_for_comm = current_state(params, ash_excitation, hf_state)

            if pool_sample_size is None or pool_sample_size >= len(operator_pool_ops):
                candidate_indices = list(range(len(operator_pool_ops)))
            else:
                candidate_indices = [int(idx) for idx in rng.choice(
                    len(operator_pool_ops), size=pool_sample_size, replace=False
                )]

            candidate_positions = list(enumerate(candidate_indices))
            score_by_position = {}
            diff_by_position = {}

            if executor is not None and len(candidate_positions) > 1:
                chunk_size = _resolve_chunk_size(
                    total_items=len(candidate_positions),
                    worker_count=worker_count,
                    user_chunk_size=gradient_chunk_size,
                )
                params_eval = np.asarray(params)
                ash_eval = tuple(tuple(int(v) for v in exc) for exc in ash_excitation)
                if backend == "process":
                    futures = [
                        executor.submit(
                            _adapt_worker_eval_chunk,
                            chunk,
                            params_eval,
                            ash_eval,
                            state_for_comm,
                        )
                        for chunk in _iter_chunks(candidate_positions, chunk_size)
                    ]
                else:
                    futures = [
                        executor.submit(
                            _evaluate_commutator_chunk,
                            chunk,
                            params_eval,
                            ash_eval,
                            state_for_comm,
                        )
                        for chunk in _iter_chunks(candidate_positions, chunk_size)
                    ]
                for future in as_completed(futures):
                    chunk_scores, chunk_diffs = future.result()
                    score_by_position.update(chunk_scores)
                    diff_by_position.update(chunk_diffs)
            else:
                for position, idx in candidate_positions:
                    w = operator_pool_ops[idx]
                    if commutator_mode == "statevec":
                        exp_used = commutator_expectation_state(state_for_comm, H, w)
                    else:
                        exp_used = commutator_expectation(params, ash_excitation, hf_state, H, w)
                    score_by_position[position] = float(np.abs(2.0 * exp_used))
                    if commutator_debug:
                        if commutator_mode == "statevec":
                            exp_other = commutator_expectation(params, ash_excitation, hf_state, H, w)
                        else:
                            exp_other = commutator_expectation_state(state_for_comm, H, w)
                        diff_by_position[position] = float(np.abs(exp_used - exp_other))

            # Preserve exact serial tie-breaking by reducing in candidate order.
            for position, idx in candidate_positions:
                current_value = score_by_position[position]
                if current_value > max_value:
                    max_value = current_value
                    max_operator = operator_pool[idx]
                if commutator_debug:
                    max_diff = max(max_diff, diff_by_position[position])

            indices_str = re.findall(r"\d+", str(max_operator))
            excitations = [int(index) for index in indices_str]
            ash_excitation.append(excitations)

            # ------------------------------------------------------------------
            # 6b. Optimize the ansatz parameters.
            # ------------------------------------------------------------------
            params = np.append(np.asarray(params), 0.0)
            result = minimize(
                cost,
                params,
                method=optimizer_method,
                tol=1e-12,
                options={"disp": False, "maxiter": int(optimizer_maxiter)},
            )

            energies.append(result.fun)
            params = result.x
            print("Energies are", energies, flush=True)
            if commutator_debug:
                print(f"Max commutator diff: {max_diff:.6e}", flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    print("energies:", energies[-1])
    return params, ash_excitation, energies
