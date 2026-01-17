def test_qrte_generates_basis_h2_sto3g():
    import numpy as np
    import scipy  # noqa: F401
    import pennylane  # noqa: F401
    import pyscf  # noqa: F401

    import QCANT

    symbols = ["H", "H"]
    geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]])

    energies, basis_states, times = QCANT.qrte(
        symbols=symbols,
        geometry=geometry,
        delta_t=0.1,
        n_steps=2,
        active_electrons=2,
        active_orbitals=2,
        basis="sto-3g",
        charge=0,
        spin=0,
        device_name="default.qubit",
        trotter_steps=1,
    )
    assert basis_states.shape[0] == 3
    assert times.shape == (3,)
    assert abs(times[0] - 0.0) < 1e-12
    assert abs(times[-1] - 0.2) < 1e-12

    norms = np.linalg.norm(basis_states, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-7)

    assert np.linalg.norm(basis_states[1] - basis_states[0]) > 1e-8

    assert energies.ndim == 1
    assert energies.size >= 1
    assert np.all(np.isfinite(energies))