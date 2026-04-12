def test_exact_krylov_sparse_and_basis_threshold():
    import numpy as np
    import scipy  # noqa: F401
    import pennylane  # noqa: F401
    import pyscf  # noqa: F401

    import QCANT

    symbols = ["H", "H"]
    geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]])
    threshold = 1e-3

    energies, basis_states, min_history = QCANT.exact_krylov(
        symbols=symbols,
        geometry=geometry,
        n_steps=1,
        active_electrons=2,
        active_orbitals=2,
        basis="sto-3g",
        charge=0,
        spin=0,
        use_sparse=True,
        basis_threshold=threshold,
        return_min_energy_history=True,
    )

    assert basis_states.shape[0] == 2
    assert min_history.shape == (1,)

    norms = np.linalg.norm(basis_states, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-7)

    assert energies.ndim == 1
    assert energies.size >= 1
    assert np.all(np.isfinite(energies))

    mask = np.abs(basis_states) < threshold
    assert np.all(basis_states[mask] == 0.0)


def test_qkud_sparse_and_basis_threshold():
    import numpy as np
    import scipy  # noqa: F401
    import pennylane  # noqa: F401
    import pyscf  # noqa: F401

    import QCANT

    symbols = ["H", "H"]
    geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]])
    threshold = 1e-3

    energies, basis_states, min_history = QCANT.qkud(
        symbols=symbols,
        geometry=geometry,
        n_steps=1,
        epsilon=0.1,
        active_electrons=2,
        active_orbitals=2,
        basis="sto-3g",
        charge=0,
        spin=0,
        use_sparse=True,
        basis_threshold=threshold,
        return_min_energy_history=True,
    )

    assert basis_states.shape[0] == 2
    assert min_history.shape == (1,)

    norms = np.linalg.norm(basis_states, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-7)

    assert energies.ndim == 1
    assert energies.size >= 1
    assert np.all(np.isfinite(energies))

    mask = np.abs(basis_states) < threshold
    assert np.all(basis_states[mask] == 0.0)


def test_qrte_sparse_and_basis_threshold():
    import numpy as np
    import scipy  # noqa: F401
    import pennylane  # noqa: F401
    import pyscf  # noqa: F401

    import QCANT

    symbols = ["H", "H"]
    geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]])
    threshold = 1e-3

    energies, basis_states, times = QCANT.qrte(
        symbols=symbols,
        geometry=geometry,
        delta_t=0.1,
        n_steps=1,
        active_electrons=2,
        active_orbitals=2,
        basis="sto-3g",
        charge=0,
        spin=0,
        device_name="default.qubit",
        trotter_steps=1,
        use_sparse=True,
        basis_threshold=threshold,
    )

    assert basis_states.shape[0] == 2
    assert times.shape == (2,)

    norms = np.linalg.norm(basis_states, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-7)

    assert energies.ndim == 1
    assert energies.size >= 1
    assert np.all(np.isfinite(energies))

    mask = np.abs(basis_states) < threshold
    assert np.all(basis_states[mask] == 0.0)
