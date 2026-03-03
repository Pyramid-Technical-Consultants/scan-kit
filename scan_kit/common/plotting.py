"""Plotting utilities for scan-kit analysis scripts."""

import pandas as pd

# Default colors for multi-session plots
DEFAULT_SESSION_COLORS = [
    "skyblue",
    "lightcoral",
    "limegreen",
    "orange",
    "purple",
    "brown",
    "pink",
    "gray",
]


def plot_boxplots_for_column(
    ax,
    session_data,
    column_name,
    energies,
    colors=None,
    showfliers=False,
    position_offset=0.35,
    width=0.5,
):
    """Plot box plots for a specific column across all sessions, grouped by energy.

    Args:
        ax: Matplotlib axes to plot on.
        session_data: Dict mapping session_id -> data dict (must have column_name and energy).
        column_name: Name of the column to plot.
        energies: Sorted list of energy values for x-axis.
        colors: List of colors for each session. Defaults to DEFAULT_SESSION_COLORS.
        showfliers: Whether to show outliers. Default False.
        position_offset: Horizontal offset between sessions at same energy. Default 0.35.
        width: Width of each box. Default 0.5.
    """
    if colors is None:
        colors = DEFAULT_SESSION_COLORS[: len(session_data)]

    for i, (session_id, data) in enumerate(session_data.items()):
        if column_name not in data:
            continue

        df = pd.DataFrame(
            {column_name: data[column_name], "energy": data["energy"]}
        )

        column_data = []
        positions = []
        for j, energy in enumerate(energies):
            energy_data = df[df["energy"] == energy][column_name].values
            column_data.append(energy_data)
            positions.append(j + (i - 0.5) * position_offset)

        ax.boxplot(
            column_data,
            positions=positions,
            patch_artist=True,
            showfliers=showfliers,
            showcaps=False,
            widths=width,
            boxprops=dict(facecolor=colors[i], alpha=0.7),
            medianprops=dict(color="black", linewidth=1.5),
            whiskerprops=dict(color="black"),
            capprops=dict(color="black"),
        )


def plot_scatter_energy(ax, x, y, energy, **kwargs):
    """Scatter plot with energy as color.

    Args:
        ax: Matplotlib axes.
        x, y: Data arrays.
        energy: Energy values for colormap.
        **kwargs: Passed to ax.scatter (e.g., alpha, s, marker, label).
    """
    defaults = {"c": energy, "cmap": "viridis", "alpha": 0.3, "s": 50}
    defaults.update(kwargs)
    return ax.scatter(x, y, **defaults)


def add_energy_colorbar(fig_or_ax, energies=None, vmin=None, vmax=None):
    """Add a colorbar for energy to a figure or axes.

    Args:
        fig_or_ax: Figure or axes to attach colorbar to.
        energies: Optional array of energy values to determine range.
        vmin, vmax: Optional explicit range (overrides energies if provided).
    """
    import matplotlib.pyplot as plt

    if vmin is None and energies is not None:
        vmin = min(energies)
    if vmax is None and energies is not None:
        vmax = max(energies)
    if vmin is None:
        vmin = 0
    if vmax is None:
        vmax = 250

    sm = plt.cm.ScalarMappable(
        cmap="viridis", norm=plt.Normalize(vmin=vmin, vmax=vmax)
    )
    sm.set_array([])
    return plt.colorbar(sm, ax=fig_or_ax, label="Energy (MeV)")
