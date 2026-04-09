"""Plotting utilities for scan-kit analysis scripts."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Shared style constants — import these in views for a consistent look
# ---------------------------------------------------------------------------

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

FIG_SIZE_2x2 = (15, 8)
FIG_SIZE_1x2 = (15, 6)
FIG_SIZE_SINGLE = (14, 8)

SUPTITLE_KW = dict(fontsize=13, fontweight="bold")
GRID_KW = dict(visible=True, alpha=0.3)
REFLINE_KW = dict(color="gray", linestyle="--", linewidth=1, alpha=0.6)

SCATTER_ALPHA = 0.45
SCATTER_SIZE = 18

SLOPE_LABEL_KW = dict(
    fontsize=10,
    color="black",
    fontweight="normal",
    va="top",
    ha="left",
)
SLOPE_LABEL_BOX = dict(
    boxstyle="round,pad=0.3",
    facecolor="white",
    edgecolor="lightgray",
    alpha=0.9,
    linewidth=0.8,
)


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


def annotate_slopes(ax, labels_and_colors, *, x_anchor=0.03, y_top=0.97):
    """Stack slope annotations inside the axes with a consistent project style.

    Args:
        ax: Matplotlib axes.
        labels_and_colors: List of (text, color) tuples.
        x_anchor: Horizontal position in axes fraction.
        y_top: Top of the first label in axes fraction.
    """
    dy = min(0.09, 0.88 / max(len(labels_and_colors), 1))
    for k, (txt, _color) in enumerate(labels_and_colors):
        ax.text(
            x_anchor,
            y_top - k * dy,
            txt,
            transform=ax.transAxes,
            zorder=6,
            bbox=SLOPE_LABEL_BOX,
            **SLOPE_LABEL_KW,
        )


def make_session_legend(ax, session_ids, colors, **kwargs):
    """Add a rectangle-patch legend for sessions on *ax*.

    Args:
        ax: Matplotlib axes.
        session_ids: List of session ID strings.
        colors: Matching list of face colors.
        **kwargs: Forwarded to ``ax.legend()``.
    """
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colors[i], alpha=0.7,
                       label=f"Session {sid}")
        for i, sid in enumerate(session_ids)
    ]
    defaults = dict(loc="upper right")
    defaults.update(kwargs)
    ax.legend(handles=handles, **defaults)


def style_energy_axes(ax, energies, ylabel=None):
    """Apply the standard energy x-axis and grid to *ax*.

    Args:
        ax: Matplotlib axes.
        energies: Sorted energy list for the x-ticks.
        ylabel: Optional y-axis label.
    """
    ax.set_xlabel("Energy (MeV)")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(**GRID_KW)
    ax.set_xticks(np.arange(len(energies)))
    ax.set_xticklabels([f"{e:g}" for e in energies], rotation=90)


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
