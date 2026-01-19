def test_qkud_generates_basis_h2_sto3g():
    import numpy as np
    import scipy  # noqa: F401
    import pennylane  # noqa: F401
    import pyscf  # noqa: F401

    import QCANT

    symbols = ["H", "H"]
    geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]])

    energies, basis_states, min_energy_history = QCANT.qkud(
        symbols=symbols,
        geometry=geometry,
        n_steps=2,
        epsilon=0.1,
        active_electrons=2,
        active_orbitals=2,
        basis="sto-3g",
        charge=0,
        spin=0,
        return_min_energy_history=True,
    )

    assert basis_states.shape[0] == 3

    norms = np.linalg.norm(basis_states, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-7)

    assert energies.ndim == 1
    assert energies.size >= 1
    assert np.all(np.isfinite(energies))

    assert min_energy_history.shape == (2,)
    assert np.isfinite(min_energy_history).all()
