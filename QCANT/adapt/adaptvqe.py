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

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any, Callable, Mapping, Optional, Sequence


_FERMIONIC_POOL_ALIASES = {"fermionic_sd", "sd", "fermionic"}
_QUBIT_POOL_ALIASES = {"qubit_excitation", "qe", "qubit"}


class _ExcitationList(list):
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


def _ansatz_type_from_pool_type(pool_type: str) -> str:
    """Resolve the ansatz operator family implied by the ADAPT pool."""
    if pool_type == "fermionic_sd":
        return "fermionic"
    if pool_type == "qubit_excitation":
        return "qubit_excitation"
    raise ValueError(f"Unsupported canonical pool_type: {pool_type}")


def _apply_excitation_gate(qml, excitation, weight, ansatz_type: str) -> None:
    """Apply one excitation gate in the selected ansatz family."""
    if ansatz_type == "fermionic":
        if len(excitation) == 4:
            qml.FermionicDoubleExcitation(
                weight=weight,
                wires1=list(range(excitation[0], excitation[1] + 1)),
                wires2=list(range(excitation[2], excitation[3] + 1)),
            )
            return
        if len(excitation) == 2:
            qml.FermionicSingleExcitation(
                weight=weight,
                wires=list(range(excitation[0], excitation[1] + 1)),
            )
            return
    elif ansatz_type == "qubit_excitation":
        if len(excitation) == 4:
            qml.DoubleExcitation(
                weight,
                wires=[int(excitation[0]), int(excitation[1]), int(excitation[2]), int(excitation[3])],
            )
            return
        if len(excitation) == 2:
            qml.SingleExcitation(
                weight,
                wires=[int(excitation[0]), int(excitation[1])],
            )
            return

    raise ValueError(
        "Each excitation must have length 2 (single) or 4 (double); "
        f"received {excitation!r} for ansatz_type='{ansatz_type}'."
    )


def _build_operator_pool(qml, active_electrons: int, qubits: int, pool_type: str):
    """Build ADAPT operator pool labels and corresponding qubit operators."""
    singles, doubles = qml.qchem.excitations(active_electrons, qubits)
    pool_excitations = [list(map(int, ex)) for ex in singles] + [list(map(int, ex)) for ex in doubles]

    if pool_type == "fermionic_sd":
        op1 = [qml.fermi.FermiWord({(0, x[0]): "+", (1, x[1]): "-"}) for x in singles]
        op2 = [
            qml.fermi.FermiWord({(0, x[0]): "+", (1, x[1]): "+", (2, x[2]): "-", (3, x[3]): "-"})
            for x in doubles
        ]
        operator_pool_ops = [qml.fermi.jordan_wigner(op) for op in (op1 + op2)]
    elif pool_type == "qubit_excitation":
        operator_pool_ops = []
        for excitation in pool_excitations:
            if len(excitation) == 2:
                gen = qml.SingleExcitation(0.0, wires=[excitation[0], excitation[1]]).generator()
            elif len(excitation) == 4:
                gen = qml.DoubleExcitation(
                    0.0,
                    wires=[excitation[0], excitation[1], excitation[2], excitation[3]],
                ).generator()
            else:
                raise ValueError(f"Unexpected excitation shape in pool: {excitation!r}")
            # Use anti-Hermitian excitation operators for commutator scoring,
            # consistent with ADAPT's |2<[H, A_i]>| selection metric.
            operator_pool_ops.append(1j * gen)
    else:
        raise ValueError(f"Unsupported canonical pool_type: {pool_type}")

    return pool_excitations, operator_pool_ops


def _validate_inputs(
    symbols: Sequence[str],
    geometry,
    adapt_it: int,
    shots: Optional[int],
    commutator_shots: Optional[int],
    commutator_mode: str,
    hamiltonian_cutoff: float,
    hamiltonian_source: str,
    pool_type: str,
    pool_sample_size: Optional[int],
    max_workers: Optional[int],
    gradient_chunk_size: Optional[int],
    parallel_backend: str,
    pauli_grouping: bool,
    grouping_type: str,
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
    if adapt_it < 0:
        raise ValueError("adapt_it must be >= 0")
    if shots is not None and shots < 0:
        raise ValueError("shots must be >= 0")
    if commutator_shots is not None and commutator_shots < 0:
        raise ValueError("commutator_shots must be >= 0")
    if commutator_mode not in {"ansatz", "statevec"}:
        raise ValueError("commutator_mode must be 'ansatz' or 'statevec'")
    if hamiltonian_cutoff < 0:
        raise ValueError("hamiltonian_cutoff must be >= 0")
    if hamiltonian_source not in {"casci", "molecular"}:
        raise ValueError("hamiltonian_source must be one of {'casci', 'molecular'}")
    _normalize_pool_type(pool_type)
    if pool_sample_size is not None and pool_sample_size <= 0:
        raise ValueError("pool_sample_size must be > 0")
    if max_workers is not None and max_workers <= 0:
        raise ValueError("max_workers must be > 0")
    if gradient_chunk_size is not None and gradient_chunk_size <= 0:
        raise ValueError("gradient_chunk_size must be > 0")
    if parallel_backend not in {"auto", "thread", "process"}:
        raise ValueError("parallel_backend must be one of {'auto', 'thread', 'process'}")
    if pauli_grouping and grouping_type not in {"qwc", "commuting", "anticommuting"}:
        raise ValueError(
            "grouping_type must be one of {'qwc', 'commuting', 'anticommuting'} "
            "when pauli_grouping=True"
        )

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

    device_kwargs = dict(payload.get("device_kwargs") or {})

    def _make_device(
        name: Optional[str],
        wires: int,
        device_shots: Optional[int],
        extra_kwargs: Optional[dict[str, Any]] = None,
    ):
        kwargs = dict(extra_kwargs or {})
        kwargs.pop("wires", None)
        kwargs.pop("shots", None)
        if device_shots is not None and device_shots > 0:
            kwargs["shots"] = device_shots
        if name is not None:
            return qml.device(name, wires=wires, **kwargs)
        try:
            return qml.device("lightning.qubit", wires=wires, **kwargs)
        except Exception:
            return qml.device("default.qubit", wires=wires, **kwargs)

    qubits = int(payload["qubits"])
    hf_state = payload["hf_state"]
    commutator_pool_ops = payload["commutator_pool_ops"]
    comm_shots = payload["comm_shots"]
    commutator_mode = payload["commutator_mode"]
    commutator_debug = bool(payload["commutator_debug"])
    ansatz_type = str(payload["ansatz_type"])
    device_name = payload["device_name"]

    def _apply_ansatz_local(hf_state_local, ash_excitation_local, params_local):
        qml.BasisState(hf_state_local, wires=range(qubits))
        for i, excitation in enumerate(ash_excitation_local):
            _apply_excitation_gate(qml, excitation, params_local[i], ansatz_type)

    comm_from_ansatz = None
    comm_from_state = None

    if commutator_mode == "statevec" or commutator_debug:
        dev_state = _make_device(device_name, qubits, None, device_kwargs)

        @qml.qnode(dev_state)
        def comm_from_state(state, comm_op_local):
            qml.StatePrep(state, wires=range(qubits))
            return qml.expval(comm_op_local)

    if commutator_mode == "ansatz" or commutator_debug:
        dev_ansatz = _make_device(device_name, qubits, comm_shots, device_kwargs)

        @qml.qnode(dev_ansatz)
        def comm_from_ansatz(params_local, ash_local, hf_state_local, comm_op_local):
            _apply_ansatz_local(hf_state_local, ash_local, params_local)
            return qml.expval(comm_op_local)

    _ADAPT_WORKER_STATE.clear()
    _ADAPT_WORKER_STATE.update(
        {
            "np": np,
            "commutator_mode": commutator_mode,
            "commutator_debug": commutator_debug,
            "commutator_pool_ops": commutator_pool_ops,
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
    commutator_pool_ops = _ADAPT_WORKER_STATE["commutator_pool_ops"]
    hf_state = _ADAPT_WORKER_STATE["hf_state"]
    comm_from_ansatz = _ADAPT_WORKER_STATE["comm_from_ansatz"]
    comm_from_state = _ADAPT_WORKER_STATE["comm_from_state"]

    scores = {}
    diffs = {}
    for position, idx in chunk_positions:
        comm_op = commutator_pool_ops[idx]
        if commutator_mode == "statevec":
            exp_used = comm_from_state(state_eval, comm_op)
        else:
            exp_used = comm_from_ansatz(params_eval, ash_eval, hf_state, comm_op)
        scores[position] = float(np.abs(2.0 * exp_used))

        if commutator_debug:
            if commutator_mode == "statevec":
                exp_other = comm_from_ansatz(params_eval, ash_eval, hf_state, comm_op)
            else:
                exp_other = comm_from_state(state_eval, comm_op)
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
    hamiltonian_source: str = "casci",
    pool_type: str = "fermionic_sd",
    pool_sample_size: Optional[int] = None,
    pool_seed: Optional[int] = None,
    parallel_gradients: bool = False,
    parallel_backend: str = "auto",
    max_workers: Optional[int] = None,
    gradient_chunk_size: Optional[int] = None,
    optimizer_method: str = "BFGS",
    optimizer_maxiter: int = 100_000_000,
    pauli_grouping: bool = False,
    grouping_type: str = "qwc",
    device_kwargs: Optional[Mapping[str, Any]] = None,
    return_history: bool = False,
    iteration_callback: Optional[Callable[[Mapping[str, Any]], None]] = None,
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
    hamiltonian_source
        Source Hamiltonian used for ADAPT optimization:

        - ``"casci"`` (default): CASCI-effective active-space Hamiltonian.
        - ``"molecular"``: PennyLane molecular Hamiltonian from
          ``qml.qchem.molecular_hamiltonian`` using the same
          active-space arguments as qscEOM.
    pool_type
        Operator-pool family used by ADAPT:

        - ``"fermionic_sd"``/``"sd"``/``"fermionic"``: reference SD fermionic pool
          with ``FermionicSingle/DoubleExcitation`` ansatz updates.
        - ``"qubit_excitation"``/``"qe"``/``"qubit"``: qubit excitation pool with
          ``Single/DoubleExcitation`` ansatz updates.
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
    pauli_grouping
        If True, pre-compute Pauli grouping metadata (e.g. QWC) for Hamiltonian
        and commutator observables before shot-based measurements.
    grouping_type
        Grouping strategy passed to ``compute_grouping`` when
        ``pauli_grouping=True``.
    device_kwargs
        Optional keyword arguments forwarded to ``qml.device``. This is useful
        when selecting hardware/noise-model specific backends that require
        extra constructor parameters.
    return_history
        If True, also return a per-iteration history payload containing ADAPT
        selection/optimization snapshots.
    iteration_callback
        Optional callback invoked after each ADAPT iteration with a snapshot
        dictionary containing the selected operator index, max gradient, current
        energy, parameters, and excitations.

    Returns
    -------
    tuple
        ``(params, ash_excitation, energies)`` as produced by the optimization.
        If ``return_history=True``, a fourth element is appended:
        ``history`` (list of iteration snapshots).

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
        adapt_it,
        shots,
        commutator_shots,
        commutator_mode,
        hamiltonian_cutoff,
        hamiltonian_source,
        pool_type,
        pool_sample_size,
        max_workers,
        gradient_chunk_size,
        parallel_backend,
        pauli_grouping,
        grouping_type,
    )

    try:
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

    pool_type_canonical = _normalize_pool_type(pool_type)
    ansatz_type = _ansatz_type_from_pool_type(pool_type_canonical)

    # --------------------------------------------------------------------------
    # 2. Device setup.
    # --------------------------------------------------------------------------
    device_kwargs_local = dict(device_kwargs or {})

    def _make_device(
        name: Optional[str],
        wires: int,
        device_shots: Optional[int],
        extra_kwargs: Optional[dict[str, Any]] = None,
    ):
        """Create a PennyLane device with optional shot-based execution."""
        kwargs = dict(extra_kwargs or {})
        kwargs.pop("wires", None)
        kwargs.pop("shots", None)
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
    # 3. Hamiltonian construction.
    # --------------------------------------------------------------------------
    hamiltonian_source = str(hamiltonian_source).strip().lower()
    if hamiltonian_source == "casci":
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
        active_electrons = int(sum(mycas_ref.nelecas))
    else:
        H, qubits = qml.qchem.molecular_hamiltonian(
            symbols,
            geometry,
            basis=basis,
            method="pyscf",
            active_electrons=int(active_electrons),
            active_orbitals=int(active_orbitals),
            charge=int(charge),
        )
        active_electrons = int(active_electrons)
        print("Reference Hamiltonian: qml.qchem.molecular_hamiltonian", flush=True)

    if pauli_grouping and hasattr(H, "compute_grouping"):
        H.compute_grouping(grouping_type=grouping_type)

    energies = []
    history = []
    ash_excitation = _ExcitationList(ansatz_type=ansatz_type, pool_type=pool_type_canonical)

    hf_state = qml.qchem.hf_state(active_electrons, qubits)

    # --------------------------------------------------------------------------
    # 5. Quantum circuit and cost function setup.
    # --------------------------------------------------------------------------
    comm_shots = shots if commutator_shots is None else commutator_shots
    if commutator_mode == "statevec" and comm_shots is not None and comm_shots > 0:
        raise ValueError("commutator_mode='statevec' requires analytic commutator_shots")
    if commutator_debug and comm_shots is not None and comm_shots > 0:
        raise ValueError("commutator_debug requires analytic commutator_shots")
    dev_comm = _make_device(device_name, qubits, comm_shots, device_kwargs_local)
    dev = _make_device(device_name, qubits, shots, device_kwargs_local)
    dev_state = None
    dev_comm_state = None

    def _apply_ansatz(hf_state, ash_excitation, params):
        """Apply the current ansatz to the Hartree-Fock state."""
        qml.BasisState(hf_state, wires=range(qubits))
        for i, excitation in enumerate(ash_excitation):
            _apply_excitation_gate(qml, excitation, params[i], ansatz_type)

    @qml.qnode(dev_comm)
    def commutator_expectation(params, ash_excitation, hf_state, comm_op):
        """Compute the expectation value of a commutator observable."""
        _apply_ansatz(hf_state, ash_excitation, params)
        return qml.expval(comm_op)

    if commutator_mode == "statevec" or commutator_debug:
        dev_state = _make_device(device_name, qubits, None, device_kwargs_local)
        dev_comm_state = _make_device(device_name, qubits, None, device_kwargs_local)

        @qml.qnode(dev_state)
        def current_state(params, ash_excitation, hf_state):
            """Return the current statevector."""
            _apply_ansatz(hf_state, ash_excitation, params)
            return qml.state()

        @qml.qnode(dev_comm_state)
        def commutator_expectation_state(state, comm_op):
            """Compute the expectation value of a commutator from a statevector."""
            qml.StatePrep(state, wires=range(qubits))
            return qml.expval(comm_op)

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
    pool_excitations, operator_pool_ops = _build_operator_pool(
        qml,
        active_electrons,
        qubits,
        pool_type_canonical,
    )
    commutator_pool_ops = [qml.commutator(H, op) for op in operator_pool_ops]
    if pauli_grouping:
        for comm_op in commutator_pool_ops:
            if hasattr(comm_op, "compute_grouping"):
                comm_op.compute_grouping(grouping_type=grouping_type)
    params = pnp.zeros(len(ash_excitation), requires_grad=True)
    rng = np.random.default_rng(pool_seed)

    worker_count = _resolve_worker_count(max_workers)
    backend = _resolve_parallel_backend(parallel_backend)
    executor = None
    if parallel_gradients and worker_count > 1:
        if backend == "process":
            worker_payload = {
                "qubits": int(qubits),
                "hf_state": np.asarray(hf_state),
                "commutator_pool_ops": tuple(commutator_pool_ops),
                "comm_shots": comm_shots,
                "commutator_mode": commutator_mode,
                "commutator_debug": bool(commutator_debug),
                "ansatz_type": ansatz_type,
                "device_name": device_name,
                "device_kwargs": device_kwargs_local,
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
                # Threaded commutator evaluation can be unstable for certain
                # operator templates/device stacks; fallback to safe serial mode.
                backend = "serial"
                executor = None
        else:
            executor = ThreadPoolExecutor(max_workers=worker_count)

    def _evaluate_commutator_chunk(chunk_positions, params_eval, ash_eval, state_eval):
        """Evaluate commutators for a chunk of candidate operator positions."""
        scores = {}
        diffs = {}
        comm_from_state = None
        comm_from_ansatz = None

        if commutator_mode == "statevec":
            local_dev_state = _make_device(device_name, qubits, None, device_kwargs_local)

            @qml.qnode(local_dev_state)
            def comm_from_state(state, comm_op_local):
                qml.StatePrep(state, wires=range(qubits))
                return qml.expval(comm_op_local)

            if commutator_debug:
                local_dev_ansatz = _make_device(
                    device_name,
                    qubits,
                    comm_shots,
                    device_kwargs_local,
                )

                @qml.qnode(local_dev_ansatz)
                def comm_from_ansatz(params_local, ash_local, hf_state_local, comm_op_local):
                    _apply_ansatz(hf_state_local, ash_local, params_local)
                    return qml.expval(comm_op_local)
        else:
            local_dev_ansatz = _make_device(
                device_name,
                qubits,
                comm_shots,
                device_kwargs_local,
            )

            @qml.qnode(local_dev_ansatz)
            def comm_from_ansatz(params_local, ash_local, hf_state_local, comm_op_local):
                _apply_ansatz(hf_state_local, ash_local, params_local)
                return qml.expval(comm_op_local)

            if commutator_debug:
                local_dev_state = _make_device(
                    device_name,
                    qubits,
                    None,
                    device_kwargs_local,
                )

                @qml.qnode(local_dev_state)
                def comm_from_state(state, comm_op_local):
                    qml.StatePrep(state, wires=range(qubits))
                    return qml.expval(comm_op_local)

        for position, idx in chunk_positions:
            comm_op = commutator_pool_ops[idx]
            if commutator_mode == "statevec":
                exp_used = comm_from_state(state_eval, comm_op)
            else:
                exp_used = comm_from_ansatz(params_eval, ash_eval, hf_state, comm_op)

            scores[position] = float(np.abs(2.0 * exp_used))

            if commutator_debug:
                if commutator_mode == "statevec":
                    exp_other = comm_from_ansatz(params_eval, ash_eval, hf_state, comm_op)
                else:
                    exp_other = comm_from_state(state_eval, comm_op)
                diffs[position] = float(np.abs(exp_used - exp_other))

        return scores, diffs

    try:
        for j in range(adapt_it):
            print("The adapt iteration now is", j, flush=True)

            # ------------------------------------------------------------------
            # 6a. Select the next operator for the ansatz.
            # ------------------------------------------------------------------
            max_value = float("-inf")
            max_operator_excitation = None
            max_operator_index = None
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
                    comm_op = commutator_pool_ops[idx]
                    if commutator_mode == "statevec":
                        exp_used = commutator_expectation_state(state_for_comm, comm_op)
                    else:
                        exp_used = commutator_expectation(params, ash_excitation, hf_state, comm_op)
                    score_by_position[position] = float(np.abs(2.0 * exp_used))
                    if commutator_debug:
                        if commutator_mode == "statevec":
                            exp_other = commutator_expectation(
                                params, ash_excitation, hf_state, comm_op
                            )
                        else:
                            exp_other = commutator_expectation_state(state_for_comm, comm_op)
                        diff_by_position[position] = float(np.abs(exp_used - exp_other))

            # Preserve exact serial tie-breaking by reducing in candidate order.
            for position, idx in candidate_positions:
                current_value = score_by_position[position]
                if current_value > max_value:
                    max_value = current_value
                    max_operator_excitation = pool_excitations[idx]
                    max_operator_index = int(idx)
                if commutator_debug:
                    max_diff = max(max_diff, diff_by_position[position])

            if max_operator_excitation is None or max_operator_index is None:
                raise RuntimeError("No operator selected during ADAPT iteration.")
            ash_excitation.append([int(index) for index in max_operator_excitation])

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
            iteration_snapshot = {
                "iteration": int(j + 1),
                "selected_pool_index": int(max_operator_index),
                "selected_excitation": [int(v) for v in max_operator_excitation],
                "adapt_max_gradient": float(max_value),
                "energy": float(result.fun),
                "params": np.asarray(params, dtype=float).copy(),
                "ash_excitation": [[int(v) for v in excitation] for excitation in ash_excitation],
            }
            if return_history:
                history.append(iteration_snapshot)
            if iteration_callback is not None:
                iteration_callback(iteration_snapshot)
            print("Energies are", energies, flush=True)
            if commutator_debug:
                print(f"Max commutator diff: {max_diff:.6e}", flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    print("energies:", energies[-1])
    if return_history:
        return params, ash_excitation, energies, history
    return params, ash_excitation, energies
