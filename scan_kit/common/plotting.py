"""Plotting utilities for scan-kit analysis scripts."""

import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
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


# ---------------------------------------------------------------------------
# Interactive energy filtering: SpanSelector linking boxplots to histograms
# ---------------------------------------------------------------------------


def _plot_histogram(ax, session_data, col, loaded_ids, colors, *,
                    energy_mask=None, n_bins=101, ylabel="Probability (%)",
                    xlabel=None, title=None, ref_val=None):
    """Draw a probability-weighted histogram on *ax*.

    Args:
        ax: Matplotlib axes (will be cleared first).
        session_data: Dict mapping session_id -> data dict.
        col: Column name to histogram.
        loaded_ids: Ordered session id list.
        colors: Matching color list.
        energy_mask: Optional dict[sid -> bool array] to filter spots.
        n_bins: Number of bins (edges = n_bins).
        ylabel: Y-axis label.
        xlabel: X-axis label.
        title: Axes title.
        ref_val: Optional reference line value (vertical).
    """
    ax.clear()

    all_chunks = []
    for sid in loaded_ids:
        vals = np.asarray(session_data[sid][col], dtype=float)
        if energy_mask is not None and sid in energy_mask:
            vals = vals[energy_mask[sid]]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            all_chunks.append(vals)

    if not all_chunks:
        ax.set_visible(False)
        return

    all_finite = np.concatenate(all_chunks)
    bin_edges = np.linspace(all_finite.min(), all_finite.max(), n_bins)

    for sid, color in zip(loaded_ids, colors):
        vals = np.asarray(session_data[sid][col], dtype=float)
        if energy_mask is not None and sid in energy_mask:
            vals = vals[energy_mask[sid]]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        weights = np.full_like(vals, 100.0 / vals.size)
        ax.hist(vals, bins=bin_edges, alpha=0.5, color=color,
                label=sid, edgecolor="none", weights=weights)

    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ref_val is not None:
        ax.axvline(x=ref_val, **REFLINE_KW)
    ax.set_visible(True)


def link_boxplot_to_histogram(
    box_axes,
    hist_axes,
    session_data,
    energies,
    columns,
    colors,
    loaded_ids,
    *,
    n_bins=101,
    hist_ylabel="Probability (%)",
    hist_xlabels=None,
    hist_titles=None,
    hist_refs=None,
):
    """Install SpanSelectors on boxplot axes that interactively filter histograms.

    Args:
        box_axes: Single axis or list of boxplot axes.
        hist_axes: Matching single axis or list of histogram axes.
        session_data: Dict mapping session_id -> data dict with ``"energy"`` key.
        energies: Sorted list of energy values used for boxplot x-ticks.
        columns: Single column name or list of column names (one per axis pair).
        colors: Session color list.
        loaded_ids: Ordered session id list.
        n_bins: Number of histogram bins.
        hist_ylabel: Y-axis label for histograms.
        hist_xlabels: Single or list of x-axis labels for histograms.
        hist_titles: Single or list of titles for histograms.
        hist_refs: Single or list of reference-line values (vertical) for histograms.

    Returns:
        List of SpanSelector objects. **The caller must keep a reference** to
        prevent garbage collection.
    """
    if not isinstance(box_axes, (list, np.ndarray)):
        box_axes = [box_axes]
        hist_axes = [hist_axes]
    if isinstance(columns, str):
        columns = [columns]
    if isinstance(hist_xlabels, str) or hist_xlabels is None:
        hist_xlabels = [hist_xlabels] * len(columns)
    if isinstance(hist_titles, str) or hist_titles is None:
        hist_titles = [hist_titles] * len(columns)
    if not isinstance(hist_refs, (list, np.ndarray, type(None))):
        hist_refs = [hist_refs] * len(columns)
    if hist_refs is None:
        hist_refs = [None] * len(columns)

    energy_arr = np.array(energies, dtype=float)
    selectors = []

    for box_ax, hist_ax, col, xlabel, title, ref in zip(
        box_axes, hist_axes, columns, hist_xlabels, hist_titles, hist_refs
    ):
        _plot_histogram(
            hist_ax, session_data, col, loaded_ids, colors,
            n_bins=n_bins, ylabel=hist_ylabel, xlabel=xlabel,
            title=title, ref_val=ref,
        )

        highlight = box_ax.axvspan(0, 0, alpha=0.15, color="gold", visible=False, zorder=0)

        def _make_callback(_hist_ax, _col, _xlabel, _title, _ref, _hl, _span_ref):
            def _on_select(xmin, xmax):
                idx_lo = max(0, int(np.floor(xmin + 0.5)))
                idx_hi = min(len(energy_arr) - 1, int(np.floor(xmax + 0.5)))
                sel_energies = set(energy_arr[idx_lo:idx_hi + 1])

                mask = {}
                for sid in loaded_ids:
                    e = np.asarray(session_data[sid]["energy"], dtype=float)
                    mask[sid] = np.isin(e, list(sel_energies))

                _plot_histogram(
                    _hist_ax, session_data, _col, loaded_ids, colors,
                    energy_mask=mask, n_bins=n_bins, ylabel=hist_ylabel,
                    xlabel=_xlabel, title=_title, ref_val=_ref,
                )
                make_session_legend(_hist_ax, loaded_ids, colors)

                _hl.set_x(idx_lo - 0.5)
                _hl.set_width(idx_hi - idx_lo + 1)
                _hl.set_visible(True)
                _hist_ax.figure.canvas.draw_idle()

            def _on_dblclick(event, _box_ax=box_ax):
                if not event.dblclick or event.inaxes is not _box_ax:
                    return
                _plot_histogram(
                    _hist_ax, session_data, _col, loaded_ids, colors,
                    n_bins=n_bins, ylabel=hist_ylabel,
                    xlabel=_xlabel, title=_title, ref_val=_ref,
                )
                make_session_legend(_hist_ax, loaded_ids, colors)
                _hl.set_visible(False)
                if _span_ref:
                    _span_ref[0].clear()
                _hist_ax.figure.canvas.draw_idle()

            return _on_select, _on_dblclick

        span_ref = []
        on_select, on_dblclick = _make_callback(
            hist_ax, col, xlabel, title, ref, highlight, span_ref,
        )

        span = SpanSelector(
            box_ax, on_select, "horizontal",
            useblit=True, interactive=True,
            props=dict(alpha=0.2, facecolor="gold", zorder=0),
        )
        span_ref.append(span)
        box_ax.figure.canvas.mpl_connect("button_press_event", on_dblclick)
        selectors.append(span)

    # Match initial y-limits across histogram axes
    visible_axes = [ax for ax in hist_axes if ax.get_visible()]
    if visible_axes:
        y_max = max(ax.get_ylim()[1] for ax in visible_axes)
        for ax in visible_axes:
            ax.set_ylim(0, y_max)

    return selectors
