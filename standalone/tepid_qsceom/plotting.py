"""Plot helpers for standalone TEPID/qscEOM example outputs."""

from __future__ import annotations

from pathlib import Path


def _require_plotting_dependencies():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    return np, pd, plt


def plot_tepid_basis_by_iteration(
    *,
    tepid_basis_csv: str | Path,
    exact_csv: str | Path,
    out_path: str | Path,
    y_min: float = -2.0,
    y_max: float = -1.5,
    title: str = "TEPID basis energies by iteration",
) -> None:
    """Plot all TEPID basis-state traces that enter the chosen window."""
    np, pd, plt = _require_plotting_dependencies()
    basis = pd.read_csv(tepid_basis_csv)
    exact = pd.read_csv(exact_csv)

    iterations = sorted(int(v) for v in basis["iteration"].unique())
    n_states = int(basis["state_index"].max()) + 1
    energy_matrix = np.vstack(
        [
            basis[basis["iteration"] == it]
            .sort_values("state_index")["energy_hartree"]
            .to_numpy(dtype=float)
            for it in iterations
        ]
    )

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    label_added = False
    for energy in exact["energy_hartree"].to_numpy(dtype=float):
        if y_min <= float(energy) <= y_max:
            ax.axhline(
                float(energy),
                linestyle="--",
                linewidth=1.2,
                alpha=0.8,
                color="#c7c7c7",
                label="FCI roots" if not label_added else None,
                zorder=0,
            )
            label_added = True

    cmap = plt.get_cmap("tab20", max(n_states, 1))
    tepid_label_added = False
    for state_idx in range(n_states):
        series = energy_matrix[:, state_idx]
        if not np.any((series >= y_min) & (series <= y_max)):
            continue
        ax.plot(
            iterations,
            series,
            marker="o",
            linewidth=1.8,
            color=cmap(state_idx),
            alpha=0.95,
            label="TEPID basis states" if not tepid_label_added else None,
            zorder=2,
        )
        tepid_label_added = True

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Energy (Ha)")
    ax.set_title(title)
    ax.set_xlim(float(iterations[0]) - 0.1, float(iterations[-1]) + 0.9)
    ax.set_ylim(y_min, y_max)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_qsceom_spectrum_levels(
    *,
    qsceom_csv: str | Path,
    exact_csv: str | Path,
    out_path: str | Path,
    y_min: float = -2.0,
    y_max: float = -1.5,
    title: str = "qscEOM spectrum",
) -> None:
    """Plot a final qscEOM spectrum as horizontal levels with exact dashed lines."""
    _np, pd, plt = _require_plotting_dependencies()
    spec = pd.read_csv(qsceom_csv)
    exact = pd.read_csv(exact_csv)

    windowed = spec[(spec["energy_hartree"] >= y_min) & (spec["energy_hartree"] <= y_max)].copy()
    fig, ax = plt.subplots(figsize=(8.4, 4.8))

    label_added = False
    for energy in exact["energy_hartree"].to_numpy(dtype=float):
        if y_min <= float(energy) <= y_max:
            ax.axhline(
                float(energy),
                linestyle="--",
                linewidth=1.2,
                alpha=0.8,
                color="#c7c7c7",
                label="FCI roots" if not label_added else None,
                zorder=0,
            )
            label_added = True

    if len(windowed) > 0:
        ax.scatter(
            windowed["state_index"],
            windowed["energy_hartree"],
            s=48,
            color="#0b5d8f",
            label="qscEOM roots",
            zorder=3,
        )
        for _, row in windowed.iterrows():
            ax.hlines(
                float(row["energy_hartree"]),
                float(row["state_index"]) - 0.28,
                float(row["state_index"]) + 0.28,
                color="#0b5d8f",
                linewidth=2.0,
                zorder=2,
            )
        ax.set_xlim(float(windowed["state_index"].min()) - 0.8, float(windowed["state_index"].max()) + 0.8)

    ax.set_xlabel("State index")
    ax.set_ylabel("Energy (Ha)")
    ax.set_title(title)
    ax.set_ylim(y_min, y_max)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_tepid_qsceom_by_iteration(
    *,
    tepid_qsceom_csv: str | Path,
    exact_csv: str | Path,
    out_path: str | Path,
    y_min: float = -2.0,
    y_max: float = -1.5,
    title: str = "TEPID + qscEOM spectrum by iteration",
) -> None:
    """Plot qscEOM roots by TEPID iteration with exact dashed references."""
    np, pd, plt = _require_plotting_dependencies()
    spec = pd.read_csv(tepid_qsceom_csv)
    exact = pd.read_csv(exact_csv)

    iterations = sorted(int(v) for v in spec["iteration"].unique())
    n_states = int(spec["state_index"].max()) + 1
    energy_matrix = np.vstack(
        [
            spec[spec["iteration"] == it]
            .sort_values("state_index")["energy_hartree"]
            .to_numpy(dtype=float)
            for it in iterations
        ]
    )

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    label_added = False
    for energy in exact["energy_hartree"].to_numpy(dtype=float):
        if y_min <= float(energy) <= y_max:
            ax.axhline(
                float(energy),
                linestyle="--",
                linewidth=1.2,
                alpha=0.8,
                color="#c7c7c7",
                label="FCI roots" if not label_added else None,
                zorder=0,
            )
            label_added = True

    cmap = plt.get_cmap("tab20", max(n_states, 1))
    qsceom_label_added = False
    for state_idx in range(n_states):
        series = energy_matrix[:, state_idx]
        if not np.any((series >= y_min) & (series <= y_max)):
            continue
        ax.plot(
            iterations,
            series,
            marker="o",
            linewidth=1.8,
            color=cmap(state_idx),
            alpha=0.95,
            label="qscEOM roots" if not qsceom_label_added else None,
            zorder=2,
        )
        qsceom_label_added = True

    ax.set_xlabel("TEPID iteration")
    ax.set_ylabel("Energy (Ha)")
    ax.set_title(title)
    ax.set_xlim(float(iterations[0]) - 0.1, float(iterations[-1]) + 0.9)
    ax.set_ylim(y_min, y_max)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
