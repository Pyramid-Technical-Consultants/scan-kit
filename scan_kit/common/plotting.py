"""Plotting utilities for scan-kit analysis scripts."""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.legend_handler import HandlerPatch
from matplotlib.widgets import SpanSelector
import numpy as np
import pandas as pd
from scipy import stats

from .plot_colors import DEFAULT_SESSION_COLORS
from .session_notes import load_notes

# ---------------------------------------------------------------------------
# Shared style constants — import these in views for a consistent look
# ---------------------------------------------------------------------------

FIG_SIZE_2x2 = (15, 8)
FIG_SIZE_1x2 = (15, 6)
FIG_SIZE_SINGLE = (14, 8)

# Standard per-cell figure sizing for analytic grid views. Figure size is
# derived as (ncols * CELL_W, nrows * CELL_H + HEADER_H) so every view shares
# consistent panel proportions and header spacing. Use :func:`view_grid`.
# Views that need square panels (scatter / position space) can pass
# ``cell_w == cell_h`` (see :data:`CELL_SQUARE`).
CELL_W = 7.0
CELL_H = 4.2
CELL_SQUARE = 4.6
HEADER_H = 0.9  # vertical inches reserved for the view header (title + legend)

SUPTITLE_KW = dict(fontsize=13, fontweight="bold")

# Shared view header — title and session legend centered as one group.
VIEW_HEADER_HEIGHT = 0.055
VIEW_HEADER_TOP = 0.985
VIEW_HEADER_SUBPLOT_TOP = 1.0 - VIEW_HEADER_HEIGHT
VIEW_HEADER_PAD_PT = 4.0  # gap between header content and subplot area
VIEW_HEADER_TITLE_LEGEND_GAP_PT = 14.0  # gap between title and legend in the group
VIEW_HEADER_LEGEND_KW = dict(frameon=False, borderaxespad=0, handlelength=1.2, handleheight=0.9)
SESSION_LEGEND_HANDLE_KW = dict(
    boxstyle="round,pad=0.05,rounding_size=0.35",
    edgecolor="0.30",
    linewidth=1.0,
    alpha=0.88,
)


class _HandlerRoundedPatch(HandlerPatch):
    """Preserve rounded, bordered swatches when drawn in a legend."""

    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans,
    ):
        return [
            FancyBboxPatch(
                (xdescent, ydescent),
                width,
                height,
                boxstyle=SESSION_LEGEND_HANDLE_KW["boxstyle"],
                facecolor=orig_handle.get_facecolor(),
                edgecolor=SESSION_LEGEND_HANDLE_KW["edgecolor"],
                linewidth=SESSION_LEGEND_HANDLE_KW["linewidth"],
                alpha=SESSION_LEGEND_HANDLE_KW["alpha"],
                transform=trans,
            )
        ]


SESSION_LEGEND_HANDLER_MAP = {FancyBboxPatch: _HandlerRoundedPatch()}
GRID_KW = dict(visible=True, alpha=0.3)
REFLINE_KW = dict(color="gray", linestyle="--", linewidth=1, alpha=0.6)

# Symmetric ± limit lines — shared styling for violin/scatter and histogram panels.
POSITION_MM_TOLERANCE_LEVELS = (
    (1.0, "green", "\u00b11 mm"),
    (2.0, "orange", "\u00b12 mm"),
    (3.0, "red", "\u00b13 mm"),
)
LIMIT_LINE_KW = dict(linestyle="--", linewidth=1.3, alpha=0.3, zorder=6)

HIST_PERCENTILE_CLIP = 100  # percentile for histogram outlier filtering (e.g. 99 → 1st–99th)

SCATTER_ALPHA = 0.45
SCATTER_SIZE = 18

TIGHT_LAYOUT_PAD = 1.08  # matplotlib default; matches the figure-window "Tight layout" button

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


def plot_violins_for_column(
    ax,
    session_data,
    column_name,
    energies,
    colors=None,
    position_offset=0.0,
    width=0.65,
    alpha=0.35,
):
    """Plot overlapping violin plots for a column across sessions, grouped by energy.

    Sessions share the same x position at each energy so violins stack; use
    *alpha* to keep overlapping distributions readable.
    """
    if colors is None:
        colors = DEFAULT_SESSION_COLORS[: len(session_data)]

    for i, (_session_id, data) in enumerate(session_data.items()):
        if column_name not in data:
            continue

        df = pd.DataFrame(
            {column_name: data[column_name], "energy": data["energy"]}
        )

        column_data = []
        positions = []
        for j, energy in enumerate(energies):
            energy_data = df[df["energy"] == energy][column_name].values
            energy_data = energy_data[np.isfinite(energy_data)]
            if energy_data.size == 0:
                continue
            column_data.append(energy_data)
            positions.append(j + (i - 0.5) * position_offset)

        if not column_data:
            continue

        vp = ax.violinplot(
            column_data,
            positions=positions,
            widths=width,
            showmeans=False,
            showmedians=True,
            showextrema=True,
        )
        color = colors[i]
        for body in vp["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(alpha)
            body.set_linewidth(0.8)
        if "cmedians" in vp:
            vp["cmedians"].set_color(color)
            vp["cmedians"].set_linewidth(1.2)
            vp["cmedians"].set_alpha(min(1.0, alpha + 0.45))
        for key in ("cbars", "cmins", "cmaxes"):
            if key in vp:
                vp[key].set_color(color)
                vp[key].set_linewidth(0.8)
                vp[key].set_alpha(min(1.0, alpha + 0.25))


def annotate_slopes(ax, labels_and_colors, *, x_anchor=0.03, y_top=0.97,
                    line_pitch=None):
    """Stack slope annotations inside the axes with a consistent project style.

    Lines are spaced by a font-relative pitch (in points) so they sit as a
    tight column regardless of the axes size.

    Args:
        ax: Matplotlib axes.
        labels_and_colors: List of (text, color) tuples.
        x_anchor: Horizontal position in axes fraction.
        y_top: Top of the first label in axes fraction.
        line_pitch: Vertical step between lines, in points. Defaults to a tight
            multiple of the label font size.
    """
    pitch = line_pitch if line_pitch is not None else SLOPE_LABEL_KW["fontsize"] * 1.6
    for k, (txt, _color) in enumerate(labels_and_colors):
        ax.annotate(
            txt,
            xy=(x_anchor, y_top),
            xycoords="axes fraction",
            xytext=(0, -k * pitch),
            textcoords="offset points",
            zorder=6,
            bbox=SLOPE_LABEL_BOX,
            **SLOPE_LABEL_KW,
        )


def view_header_layout_rect() -> list[float]:
    """Fallback ``rect`` for ``tight_layout`` before the header has been measured."""
    return [0.0, 0.0, 1.0, VIEW_HEADER_SUBPLOT_TOP]


def _figure_pad_fraction(fig, pts: float) -> float:
    height_in = fig.get_figheight()
    if height_in <= 0:
        return 0.01
    return (pts / 72.0) / height_in


def measure_view_header_rect(fig, renderer) -> list[float] | None:
    """Return a ``tight_layout`` *rect* sized to the actual header content."""
    title = getattr(fig, "_scan_kit_title", None)
    if title is None:
        return None

    bboxes = [title.get_window_extent(renderer)]
    legend = getattr(fig, "_scan_kit_legend", None)
    if legend is not None:
        bboxes.append(legend.get_window_extent(renderer))

    header_bottom = min(bb.y0 for bb in bboxes)
    _, y_fig = fig.transFigure.inverted().transform((0.0, header_bottom))
    pad = _figure_pad_fraction(fig, VIEW_HEADER_PAD_PT)
    top = max(y_fig - pad, 0.1)
    return [0.0, 0.0, 1.0, top]


def refresh_view_header_rect(fig, *, renderer=None) -> list[float] | None:
    """Re-center the header group, measure its extent, and cache the layout *rect*."""
    if getattr(fig, "_scan_kit_title", None) is None:
        return None
    if renderer is None:
        fig.draw_without_rendering()
        renderer = fig.canvas.get_renderer()
    _layout_view_header_group(fig, renderer)
    fig.draw_without_rendering()
    renderer = fig.canvas.get_renderer()
    rect = measure_view_header_rect(fig, renderer)
    if rect is not None:
        fig._scan_kit_header_rect = rect
    return rect


def _header_layout_rect(fig) -> list[float] | None:
    return getattr(fig, "_scan_kit_header_rect", None)


def format_session_legend_label(session_id: str, notes: dict[str, str] | None = None) -> str:
    """Build a session legend label, appending the note when present."""
    note = (notes or {}).get(session_id, "").strip()
    if note:
        return f"{session_id} — {note}"
    return session_id


def session_legend_handles(session_ids, colors):
    """Rounded, bordered patch handles for a session color legend."""
    return [
        FancyBboxPatch(
            (0, 0),
            1,
            1,
            facecolor=colors[i],
            **SESSION_LEGEND_HANDLE_KW,
        )
        for i in range(len(session_ids))
    ]


def _clear_view_header(fig) -> None:
    for attr in ("_scan_kit_title", "_scan_kit_legend"):
        artist = getattr(fig, attr, None)
        if artist is not None:
            try:
                artist.remove()
            except Exception:
                pass
            setattr(fig, attr, None)


def _layout_view_header_group(fig, renderer) -> None:
    """Center the title and session legend as one inline group (title on the left)."""
    title = getattr(fig, "_scan_kit_title", None)
    if title is None:
        return

    legend = getattr(fig, "_scan_kit_legend", None)
    fig_bb = fig.get_window_extent(renderer)
    inv = fig.transFigure.inverted()

    title_bb = title.get_window_extent(renderer)
    title_h_fig = title_bb.height / fig_bb.height

    if legend is None:
        row_cy_fig = VIEW_HEADER_TOP - title_h_fig / 2.0
        title.set_ha("center")
        title.set_va("center")
        title.set_position((0.5, row_cy_fig))
        return

    legend_bb = legend.get_window_extent(renderer)
    legend_h_fig = legend_bb.height / fig_bb.height
    row_h_fig = max(title_h_fig, legend_h_fig)
    row_cy_fig = VIEW_HEADER_TOP - row_h_fig / 2.0

    gap = VIEW_HEADER_TITLE_LEGEND_GAP_PT * fig.dpi / 72.0
    total_w = title_bb.width + gap + legend_bb.width
    group_left = fig_bb.x0 + (fig_bb.width - total_w) / 2.0

    title_x_fig, _ = inv.transform((group_left, 0.0))
    title.set_ha("left")
    title.set_va("center")
    title.set_position((title_x_fig, row_cy_fig))

    legend_left = group_left + title_bb.width + gap
    legend_x_fig, _ = inv.transform((legend_left, 0.0))
    legend.set_loc("center left")
    legend.set_bbox_to_anchor((legend_x_fig, row_cy_fig), transform=fig.transFigure)


def set_view_header(
    fig,
    title: str,
    session_ids,
    colors,
    *,
    base_dir: str | None = None,
    notes: dict[str, str] | None = None,
) -> None:
    """Place the view title and session legend centered as one inline group.

    The title sits immediately to the left of the session legend.  The legend
    shows session color, ID, and optional note from ``session_notes.json`` when
    *base_dir* is provided.
    """
    _clear_view_header(fig)

    if notes is None and base_dir:
        notes = load_notes(base_dir)
    else:
        notes = notes or {}

    fig._scan_kit_header_rect = None
    fig._scan_kit_title = fig.text(
        0.0,
        VIEW_HEADER_TOP,
        title,
        ha="left",
        va="center",
        transform=fig.transFigure,
        **SUPTITLE_KW,
    )

    if not session_ids:
        fig._scan_kit_legend = None
    else:
        labels = [format_session_legend_label(sid, notes) for sid in session_ids]
        n = len(session_ids)
        fontsize = 9 if n <= 3 else (8 if n <= 5 else 7)
        ncol = n if n <= 4 else min(n, 3)

        fig._scan_kit_legend = fig.legend(
            handles=session_legend_handles(session_ids, colors),
            labels=labels,
            loc="center left",
            bbox_to_anchor=(0.0, VIEW_HEADER_TOP),
            bbox_transform=fig.transFigure,
            fontsize=fontsize,
            ncol=ncol,
            handler_map=SESSION_LEGEND_HANDLER_MAP,
            **VIEW_HEADER_LEGEND_KW,
        )

    fig.draw_without_rendering()
    _layout_view_header_group(fig, fig.canvas.get_renderer())


def apply_tight_layout(fig=None, *, pad=TIGHT_LAYOUT_PAD, h_pad=None, w_pad=None,
                       rect=None, measure=True):
    """Apply tight layout to *fig*, matching the matplotlib window button.

    When *measure* is True (default), runs a draw pass first so label/legend/
    suptitle bboxes are measured before ``tight_layout``. Set *measure* to False
    to match the interactive toolbar button exactly (used after the window is
    shown at its final size).
    """
    fig = fig if fig is not None else plt.gcf()
    if rect is None:
        if measure:
            fig.draw_without_rendering()
            renderer = fig.canvas.get_renderer()
            rect = refresh_view_header_rect(fig, renderer=renderer)
        if rect is None:
            rect = _header_layout_rect(fig)
        if rect is None and getattr(fig, "_scan_kit_title", None) is not None:
            rect = view_header_layout_rect()
    elif measure:
        fig.draw_without_rendering()
    fig.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad, rect=rect)


def view_grid(
    nrows: int = 1,
    ncols: int = 1,
    *,
    cell_w: float = CELL_W,
    cell_h: float = CELL_H,
    header: bool = True,
    squeeze: bool = False,
    **subplots_kwargs,
):
    """Create a standard analytic-view subplot grid sized by per-cell dimensions.

    Figure size is ``(ncols * cell_w, nrows * cell_h + header_height)`` so panel
    proportions and header spacing stay consistent across views. Pass
    ``header=False`` to omit the reserved header band (views without a header).

    Returns the ``(fig, axes)`` tuple from :func:`matplotlib.pyplot.subplots`.
    With the default ``squeeze=False``, *axes* is always a 2-D array.
    """
    fig_w = ncols * cell_w
    fig_h = nrows * cell_h + (HEADER_H if header else 0.0)
    return plt.subplots(
        nrows, ncols, figsize=(fig_w, fig_h), squeeze=squeeze, **subplots_kwargs
    )


def finish_view(
    fig,
    title: str,
    session_ids,
    colors,
    *,
    base_dir: str | None = None,
    notes: dict | None = None,
    show: bool = True,
    **layout_kwargs,
) -> None:
    """Apply the standard view header then tight layout.

    Wraps the ``set_view_header(...)`` + ``apply_tight_layout(...)`` tail shared
    by analytic views. Extra keyword args are forwarded to
    :func:`apply_tight_layout`.

    When *show* is true (default), calls ``plt.show()`` so views render in the
    interactive launcher and during PDF report capture (Agg or patched show).
    """
    set_view_header(fig, title, session_ids, colors, base_dir=base_dir, notes=notes)
    apply_tight_layout(fig, **layout_kwargs)
    if show:
        plt.show()


def apply_toolbar_tight_layout(fig=None) -> None:
    """Re-run tight layout, reserving measured space for the view header."""
    fig = fig if fig is not None else plt.gcf()
    fig.draw_without_rendering()
    renderer = fig.canvas.get_renderer()
    rect = refresh_view_header_rect(fig, renderer=renderer)
    if rect is None:
        rect = _header_layout_rect(fig)
    if rect is None and getattr(fig, "_scan_kit_title", None) is not None:
        rect = view_header_layout_rect()
    if rect is not None:
        fig.tight_layout(rect=rect)
    else:
        fig.tight_layout()
    fig.canvas.draw_idle()


def make_session_legend(
    ax,
    session_ids,
    colors,
    *,
    base_dir: str | None = None,
    notes: dict[str, str] | None = None,
    **kwargs,
):
    """Add a rectangle-patch legend for sessions on *ax*.

    Prefer :func:`set_view_header` for new views so title and legend share one
    header row. This helper remains for axes-local legends (e.g. dynamic
    histogram refresh) and includes session notes when *base_dir* is set.

    Args:
        ax: Matplotlib axes.
        session_ids: List of session ID strings.
        colors: Matching list of face colors.
        base_dir: Data directory for ``session_notes.json``.
        notes: Pre-loaded notes dict; overrides *base_dir* when given.
        **kwargs: Forwarded to ``ax.legend()``.
    """
    if notes is None and base_dir:
        notes = load_notes(base_dir)
    else:
        notes = notes or {}

    labels = [format_session_legend_label(sid, notes) for sid in session_ids]
    defaults = dict(loc="upper right")
    defaults.update(kwargs)
    ax.legend(
        handles=session_legend_handles(session_ids, colors),
        labels=labels,
        handler_map=SESSION_LEGEND_HANDLER_MAP,
        **defaults,
    )


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


def scatter_with_trend(
    ax,
    x,
    y,
    *,
    color,
    label,
    alpha=SCATTER_ALPHA,
    size=SCATTER_SIZE,
    line_width=2,
    zorder=5,
):
    """Scatter ``y`` vs ``x`` and overlay a linear trend line.

    Returns:
        Fitted slope, or ``None`` when fewer than two finite points exist.
    """
    ax.scatter(
        x,
        y,
        c=color,
        alpha=alpha,
        s=size,
        edgecolors="none",
        label=label,
    )

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    xf = x_arr[mask]
    yf = y_arr[mask]
    if xf.size < 2:
        return None

    slope, intercept = np.polyfit(xf, yf, 1)
    x_range = np.array([xf.min(), xf.max()])
    ax.plot(
        x_range,
        slope * x_range + intercept,
        color=color,
        linewidth=line_width,
        linestyle="-",
        zorder=zorder,
    )
    return slope


# ---------------------------------------------------------------------------
# Unified trend lines
#
# A single core for every trend line drawn across the views. ``fit_trend``
# does the math (with optional robust MAD outlier rejection), ``trend_line_color``
# / ``TREND_LINE_KW`` give a consistent look, and ``format_trend_label`` produces
# consistent annotation text. The high-level helpers (``add_scatter_trend``,
# ``add_energy_trend``, ``add_correlation_scatter``) cover the three shapes the
# views actually need.
# ---------------------------------------------------------------------------

_MAD_SCALE = 1.4826  # median(|x - m|) * factor matches std for a normal dist
TREND_DARKEN = 0.55  # multiply a face color by this for its trend-line color

TREND_LINE_KW = dict(linewidth=2.0, linestyle="-", solid_capstyle="round")
REJECTED_KW = dict(edgecolors="red", linewidths=0.8, alpha=0.25, zorder=2)


def trend_line_color(face_color):
    """Darker variant of *face_color* used for trend lines.

    Keeps trend lines visually tied to their session/series color while staying
    readable on top of the lighter scatter/box markers.
    """
    try:
        rgb = mcolors.to_rgb(face_color)
    except (ValueError, TypeError):
        rgb = mcolors.to_rgb("C0")
    return tuple(max(0.0, c * TREND_DARKEN) for c in rgb)


def make_trend_legend(ax, trend_entries, *, loc="upper right", fontsize=9, **kwargs):
    """Add a line-style legend for trend fits on *ax*.

    Uses ``add_artist`` so the legend coexists with other legends on the same
    axes (session scatter patches, gate lines, etc.).

    Args:
        ax: Matplotlib axes.
        trend_entries: List of ``(label_text, line_color)`` tuples as returned
            by :func:`add_scatter_trend`, :func:`add_energy_trend`, or
            :func:`add_correlation_scatter`.
        loc: Legend location. Default ``"upper right"``.
        fontsize: Legend font size.
        **kwargs: Forwarded to ``ax.legend()``.

    Returns:
        The legend artist, or ``None`` when *trend_entries* is empty.
    """
    if not trend_entries:
        return None
    handles = [
        Line2D([0], [0], color=color, label=text, **TREND_LINE_KW)
        for text, color in trend_entries
    ]
    defaults = dict(loc=loc, fontsize=fontsize, framealpha=0.9)
    defaults.update(kwargs)
    legend = ax.legend(handles=handles, **defaults)
    ax.add_artist(legend)
    return legend


@dataclass
class TrendFit:
    """Result of a linear trend fit (see :func:`fit_trend`)."""

    slope: float
    intercept: float
    x: np.ndarray  # finite x values used for the fit
    y: np.ndarray  # finite y values used for the fit
    keep: np.ndarray  # bool mask over x/y of inliers kept by robust rejection
    r2: float
    n: int  # number of inliers

    def eval(self, xs):
        """Fitted ``y`` at the given ``xs``."""
        return self.slope * np.asarray(xs, dtype=float) + self.intercept

    @property
    def x_span(self) -> float:
        return float(self.x.max() - self.x.min()) if self.x.size else float("nan")

    @property
    def delta(self) -> float:
        """Change in fitted ``y`` across the observed x-range."""
        return self.slope * self.x_span


def fit_trend(x, y, *, robust=False, outlier_sigma=2.0, outlier_iterations=3):
    """Least-squares linear fit with optional iterative MAD outlier rejection.

    Args:
        x, y: Data arrays (non-finite pairs are dropped before fitting).
        robust: When True, iteratively reject points more than ``outlier_sigma``
            robust standard deviations (MAD-based) from the fit.
        outlier_sigma: Rejection threshold in robust sigmas.
        outlier_iterations: Maximum re-weighting passes.

    Returns:
        A :class:`TrendFit` (``keep`` flags robust inliers), or ``None`` when
        fewer than two finite points are available.
    """
    xf = np.asarray(x, dtype=float)
    yf = np.asarray(y, dtype=float)
    finite = np.isfinite(xf) & np.isfinite(yf)
    xf, yf = xf[finite], yf[finite]
    if xf.size < 2:
        return None

    keep = np.ones(xf.size, dtype=bool)
    if robust:
        for _ in range(outlier_iterations):
            if keep.sum() < 3:
                break
            s, b = np.polyfit(xf[keep], yf[keep], 1)
            resid = yf - (s * xf + b)
            med = np.median(resid[keep])
            sigma = np.median(np.abs(resid[keep] - med)) * _MAD_SCALE
            if sigma < 1e-12:
                break
            keep = np.abs(resid - med) <= outlier_sigma * sigma

    slope, intercept = np.polyfit(xf[keep], yf[keep], 1)
    fitted = slope * xf[keep] + intercept
    ss_res = float(np.sum((yf[keep] - fitted) ** 2))
    ss_tot = float(np.sum((yf[keep] - yf[keep].mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return TrendFit(
        slope=float(slope), intercept=float(intercept),
        x=xf, y=yf, keep=keep, r2=r2, n=int(keep.sum()),
    )


def _value_unit(unit):
    """The quantity's own unit, derived from a slope unit (``"%/MeV"`` -> ``"%"``)."""
    return unit.split("/")[0].strip() if unit else ""


def _fmt_value(value, fmt, value_unit):
    """Format a value with its unit (``%`` attaches directly, others get a space)."""
    if value_unit == "%":
        return f"{value:{fmt}}%"
    if value_unit:
        return f"{value:{fmt}} {value_unit}"
    return f"{value:{fmt}}"


def trend_session_prefix(session_id, *, n_sessions=1) -> str:
    """Short session tag for trend legend labels (first 3 ID characters)."""
    if n_sessions <= 1:
        return ""
    return f"{str(session_id)[:3]}: "


def format_trend_label(*, prefix="", slope=None, unit=None, mean=None,
                       delta=None, r2=None, ccc=None):
    """Build a consistent trend annotation string.

    Pieces appear in a fixed order (slope, mean, Δ, R², CCC); pass only what
    applies. ``mean`` and ``\u0394`` are shown in the quantity's own unit, derived
    from ``unit`` (e.g. ``"%/MeV"`` -> ``"%"``, ``"ms/MeV"`` -> ``"ms"``).
    Example: ``"S1: +0.12 %/MeV   \u03bc+0.34%   \u0394+1.2%"`` (all values 2 s.f.).
    """
    value_unit = _value_unit(unit)
    parts = []
    if slope is not None and unit is not None:
        parts.append(f"{slope:+.2g} {unit}")
    if mean is not None:
        parts.append("\u03bc" + _fmt_value(mean, "+.2g", value_unit))
    if delta is not None:
        parts.append("\u0394" + _fmt_value(delta, "+.2g", value_unit))
    if r2 is not None:
        parts.append(f"R\u00b2={r2:.2g}")
    if ccc is not None:
        parts.append(f"CCC={ccc:.2g}")
    return prefix + "   ".join(parts)


def add_scatter_trend(
    ax, x, y, *, color, unit, prefix="",
    scatter=True, label=None,
    alpha=SCATTER_ALPHA, size=SCATTER_SIZE, scatter_zorder=3,
    robust=False, outlier_sigma=2.0, outlier_iterations=3,
    highlight_rejected=False, show_mean=True, show_delta=True, show_r2=False,
    line_zorder=5,
):
    """Scatter ``y`` vs a continuous ``x`` and overlay a linear trend line.

    Combines the simple and robust scatter-trend variants used across the views.
    When ``robust`` is set, outliers are rejected (MAD-based) and optionally
    re-drawn with a red edge via ``highlight_rejected``. The annotation shows
    slope, mean and Δ-over-range by default (``show_mean`` / ``show_delta``).

    Returns:
        ``(label_text, line_color)`` suitable for :func:`make_trend_legend`, or
        ``None`` when fewer than two finite points exist.
    """
    if scatter:
        ax.scatter(x, y, c=color, alpha=alpha, s=size, edgecolors="none",
                   zorder=scatter_zorder, label=label)

    fit = fit_trend(x, y, robust=robust, outlier_sigma=outlier_sigma,
                    outlier_iterations=outlier_iterations)
    if fit is None:
        return None

    if highlight_rejected and robust:
        rejected = ~fit.keep
        if rejected.any():
            ax.scatter(fit.x[rejected], fit.y[rejected], c=color, s=size,
                       **REJECTED_KW)

    line_color = trend_line_color(color)
    x_range = np.array([fit.x.min(), fit.x.max()])
    ax.plot(x_range, fit.eval(x_range), color=line_color, zorder=line_zorder,
            **TREND_LINE_KW)

    mean_val = float(np.mean(fit.y)) if fit.y.size else float("nan")
    label_text = format_trend_label(
        prefix=prefix, slope=fit.slope, unit=unit,
        mean=mean_val if show_mean else None,
        delta=fit.delta if show_delta else None,
        r2=fit.r2 if show_r2 else None,
    )
    return label_text, line_color


def add_energy_trend(
    ax, session_data, column, energies, colors, *,
    agg="median", unit="%/MeV", position_offset=0.0,
    show_mean=True, show_delta=True, prefix_with_session=True, line_zorder=5,
):
    """Per-session trend through per-energy aggregates on a categorical x-axis.

    For box-plot / per-energy-scatter panels where the x-axis is energy index
    (0..N-1). Each session's ``column`` is aggregated per energy (``"median"``
    or ``"mean"``), a line is fit vs energy in MeV, and drawn across the
    categorical positions (offset per session by ``position_offset``). The
    annotation shows slope, mean and Δ-over-range by default. A trend legend is
    added via :func:`make_trend_legend`.

    Returns the list of ``(label, color)`` tuples that were annotated.
    """
    agg_fn = np.median if agg == "median" else np.mean
    n_sessions = len(session_data)
    energies_f = np.asarray(energies, dtype=float)
    labels: list[tuple[str, tuple]] = []

    for i, (sid, data) in enumerate(session_data.items()):
        if column not in data:
            continue
        col = np.asarray(data[column], dtype=float)
        e_all = np.asarray(data["energy"], dtype=float)

        mean_err = None
        if show_mean:
            finite = col[np.isfinite(col)]
            mean_err = float(np.mean(finite)) if finite.size else float("nan")

        e_mev, y_agg = [], []
        for energy in energies:
            vals = col[e_all == energy]
            vals = vals[np.isfinite(vals)]
            if vals.size:
                e_mev.append(float(energy))
                y_agg.append(float(agg_fn(vals)))
        if len(e_mev) < 2:
            continue

        slope, intercept = np.polyfit(np.array(e_mev), np.array(y_agg), 1)
        line_color = trend_line_color(colors[i])
        xs = [j + (i - 0.5) * position_offset for j in range(len(energies))]
        ys = slope * energies_f + intercept
        ax.plot(xs, ys, color=line_color, zorder=line_zorder, clip_on=True,
                **TREND_LINE_KW)

        delta = slope * (max(e_mev) - min(e_mev)) if show_delta else None
        prefix = trend_session_prefix(sid, n_sessions=n_sessions) if prefix_with_session else ""
        labels.append((
            format_trend_label(prefix=prefix, slope=slope, unit=unit,
                               mean=mean_err, delta=delta),
            line_color,
        ))

    if labels:
        make_trend_legend(ax, labels)
    return labels


def add_correlation_scatter(
    ax, session_data, col_x, col_y, loaded_ids, colors, *,
    xlabel=None, ylabel=None, percentile_clip=None, equal_aspect=False,
    alpha=SCATTER_ALPHA, size=SCATTER_SIZE,
):
    """Scatter ``col_x`` vs ``col_y`` per session with fit, R\u00b2/CCC and a y=x line.

    Each session gets a linear fit; the annotation reports Pearson R\u00b2 and the
    concordance correlation coefficient (CCC). ``percentile_clip`` (e.g. 99.9)
    trims shared outliers before fitting. The axis is hidden when no session has
    both columns.

    Returns the list of ``(label, color)`` tuples that were annotated.
    """
    raw_pairs = []
    for sid in loaded_ids:
        data = session_data[sid]
        if col_x not in data or col_y not in data:
            continue
        x = np.asarray(data[col_x], dtype=float)
        y = np.asarray(data[col_y], dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.any():
            raw_pairs.append((sid, x[m], y[m]))

    if not raw_pairs:
        ax.set_visible(False)
        return []

    if percentile_clip is not None and percentile_clip < 100:
        allv = np.concatenate([np.concatenate([p[1], p[2]]) for p in raw_pairs])
        lo, hi = np.percentile(allv, [100 - percentile_clip, percentile_clip])
    else:
        lo, hi = -np.inf, np.inf

    sid_index = {sid: i for i, sid in enumerate(loaded_ids)}
    labels: list[tuple[str, tuple]] = []
    plotted = False

    for sid, x_raw, y_raw in raw_pairs:
        keep = (x_raw >= lo) & (x_raw <= hi) & (y_raw >= lo) & (y_raw <= hi)
        x, y = x_raw[keep], y_raw[keep]
        if x.size < 2:
            continue
        i = sid_index[sid]
        ax.scatter(x, y, c=colors[i], alpha=alpha, s=size, edgecolors="none")
        plotted = True

        r, _ = stats.pearsonr(x, y)
        sx, sy, mx, my = x.std(), y.std(), x.mean(), y.mean()
        ccc = (2 * r * sx * sy) / (sx**2 + sy**2 + (mx - my) ** 2)

        line_color = trend_line_color(colors[i])
        fit = fit_trend(x, y)
        x_range = np.array([x.min(), x.max()])
        ax.plot(x_range, fit.eval(x_range), color=line_color, linewidth=1.8,
                linestyle="-", zorder=5)

        prefix = trend_session_prefix(sid, n_sessions=len(loaded_ids))
        labels.append((format_trend_label(prefix=prefix, r2=r**2, ccc=ccc),
                       line_color))

    if plotted:
        ax.autoscale_view()
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        ref_lo, ref_hi = max(xlim[0], ylim[0]), min(xlim[1], ylim[1])
        if ref_lo < ref_hi:
            ax.plot([ref_lo, ref_hi], [ref_lo, ref_hi], **REFLINE_KW)
        if equal_aspect:
            ax.set_aspect("equal", adjustable="datalim")

    if labels:
        make_trend_legend(ax, labels)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(visible=True, alpha=0.3)
    return labels


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
# Symmetric ± limit lines
# ---------------------------------------------------------------------------


def draw_symmetric_limit_lines(
    ax,
    levels,
    *,
    orientation="horizontal",
    line_kw=None,
    legend=False,
):
    """Draw fixed ±limit lines on *ax*.

    Args:
        ax: Matplotlib axes.
        levels: Sequence of ``(value, color, label)`` tuples.
        orientation: ``"horizontal"`` (``axhline``) or ``"vertical"`` (``axvline``).
        line_kw: Line style kwargs; defaults to :data:`LIMIT_LINE_KW`.
        legend: When True, label only the positive line at each level for legends.

    Returns:
        List of positive-side line artists (one per level), or empty when
        *levels* is empty.
    """
    if not levels:
        return []

    kw = dict(line_kw or LIMIT_LINE_KW)
    draw = ax.axhline if orientation == "horizontal" else ax.axvline
    handles = []
    for val, color, label in levels:
        label_kw = {"label": label} if legend else {}
        line = draw(val, color=color, **label_kw, **kw)
        draw(-val, color=color, **kw)
        handles.append(line)
    return handles


def pad_limits_for_symmetric_limits(lo, hi, levels):
    """Expand numeric *lo* / *hi* to include every symmetric limit magnitude."""
    for val, _, _ in levels:
        lo = min(lo, -val)
        hi = max(hi, val)
    return lo, hi


def apply_shared_block_labels(
    axes,
    *,
    column_titles=None,
    row_ylabels,
    xlabel,
    bottom_row=-1,
    ylabel_col=0,
):
    """Apply shared column titles and per-row y-axis labels to a 2-D axes block."""
    grid = np.asarray(axes)
    n_rows, n_cols = grid.shape

    if column_titles is not None:
        for col_idx, title in enumerate(column_titles):
            grid[0, col_idx].set_title(title)

    for row_idx in range(n_rows):
        grid[row_idx, ylabel_col].set_ylabel(row_ylabels[row_idx])
        for col_idx in range(n_cols):
            if col_idx != ylabel_col:
                grid[row_idx, col_idx].set_ylabel("")
            if row_idx != bottom_row:
                grid[row_idx, col_idx].set_xlabel("")

    for col_idx in range(n_cols):
        grid[bottom_row, col_idx].set_xlabel(xlabel)

    return grid


# ---------------------------------------------------------------------------
# Interactive energy filtering: SpanSelector linking boxplots to histograms
# ---------------------------------------------------------------------------


def _plot_histogram(ax, session_data, col, loaded_ids, colors, *,
                    energy_mask=None, n_bins=101, ylabel="Probability (%)",
                    xlabel=None, title=None, ref_val=None,
                    bin_range=None):
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
        bin_range: Optional (lo, hi) tuple for shared bin edges across panels.
    """
    ax.clear()

    all_chunks = []
    for sid in loaded_ids:
        if col not in session_data[sid]:
            continue
        vals = np.asarray(session_data[sid][col], dtype=float)
        if energy_mask is not None and sid in energy_mask:
            vals = vals[energy_mask[sid]]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            all_chunks.append(vals)

    if not all_chunks:
        ax.set_visible(False)
        return

    if bin_range is not None:
        lo, hi = bin_range
    else:
        all_finite = np.concatenate(all_chunks)
        pclip = HIST_PERCENTILE_CLIP
        lo, hi = np.percentile(all_finite, [100 - pclip, pclip])
    bin_edges = np.linspace(lo, hi, n_bins)

    for sid, color in zip(loaded_ids, colors):
        if col not in session_data[sid]:
            continue
        vals = np.asarray(session_data[sid][col], dtype=float)
        if energy_mask is not None and sid in energy_mask:
            vals = vals[energy_mask[sid]]
        vals = vals[np.isfinite(vals)]
        vals = vals[(vals >= lo) & (vals <= hi)]
        if vals.size == 0:
            continue
        weights = np.full_like(vals, 100.0 / vals.size)
        ax.hist(vals, bins=bin_edges, alpha=0.5, color=color,
                label=sid, edgecolor="none", weights=weights)

    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
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
    hist_percentile_clip=None,
    tolerance_levels=None,
    tolerance_line_kw=None,
    n_columns=None,
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
        hist_ylabel: Y-axis label(s) for histograms — a string, or one label per
            row when *n_columns* is set.
        hist_xlabels: Single or list of x-axis labels for histograms.
        hist_titles: Single or list of titles for histograms.
        hist_refs: Single or list of reference-line values (vertical) for histograms.
        tolerance_levels: Optional sequence of ``(value, color, label)`` tuples;
            draws vertical ``±value`` lines on each histogram after every redraw.
        tolerance_line_kw: Matplotlib kwargs for tolerance lines; defaults to
            :data:`LIMIT_LINE_KW`.
        n_columns: When set, only the first column in each row receives the
            row's *hist_ylabel* entry.

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

    if isinstance(hist_ylabel, (list, tuple)):
        hist_ylabels = list(hist_ylabel)
    else:
        hist_ylabels = [hist_ylabel] * len(columns)

    tol_kw = tolerance_line_kw or LIMIT_LINE_KW
    energy_arr = np.array(energies, dtype=float)
    selectors = []
    pclip = hist_percentile_clip if hist_percentile_clip is not None else HIST_PERCENTILE_CLIP

    def _after_histogram(ax):
        if tolerance_levels:
            draw_symmetric_limit_lines(
                ax,
                tolerance_levels,
                orientation="vertical",
                line_kw=tol_kw,
            )

    def _ylabel_for_index(idx, *, show_ylabel):
        if not show_ylabel:
            return None
        row = idx // n_columns if n_columns else idx
        if row >= len(hist_ylabels):
            return hist_ylabels[0]
        return hist_ylabels[row]

    def _redraw_histogram(hist_ax, col, xlabel, title, ref, energy_mask, *, show_ylabel, axis_idx):
        _plot_histogram(
            hist_ax, session_data, col, loaded_ids, colors,
            energy_mask=energy_mask, n_bins=n_bins,
            ylabel=_ylabel_for_index(axis_idx, show_ylabel=show_ylabel),
            xlabel=xlabel, title=title, ref_val=ref, bin_range=shared_bin_range,
        )
        _after_histogram(hist_ax)

    # Compute a shared bin range across all columns
    all_hist_vals = []
    for col in columns:
        for sid in loaded_ids:
            if col in session_data[sid]:
                v = np.asarray(session_data[sid][col], dtype=float)
                v = v[np.isfinite(v)]
                if v.size:
                    all_hist_vals.append(v)
    if all_hist_vals:
        combined = np.concatenate(all_hist_vals)
        shared_bin_range = tuple(np.percentile(combined, [100 - pclip, pclip]))
    else:
        shared_bin_range = None

    for idx, (box_ax, hist_ax, col, xlabel, title, ref) in enumerate(zip(
        box_axes, hist_axes, columns, hist_xlabels, hist_titles, hist_refs
    )):
        show_ylabel = n_columns is None or idx % n_columns == 0
        _redraw_histogram(
            hist_ax, col, xlabel, title, ref, energy_mask=None,
            show_ylabel=show_ylabel, axis_idx=idx,
        )

        highlight = box_ax.axvspan(0, 0, alpha=0.15, color="gold", visible=False, zorder=0)

        def _make_callback(_hist_ax, _col, _xlabel, _title, _ref, _hl, _span_ref,
                           _show_ylabel, _axis_idx, _bin_range=shared_bin_range):
            def _on_select(xmin, xmax):
                idx_lo = max(0, int(np.floor(xmin + 0.5)))
                idx_hi = min(len(energy_arr) - 1, int(np.floor(xmax + 0.5)))
                sel_energies = set(energy_arr[idx_lo:idx_hi + 1])

                mask = {}
                for sid in loaded_ids:
                    e = np.asarray(session_data[sid]["energy"], dtype=float)
                    mask[sid] = np.isin(e, list(sel_energies))

                _redraw_histogram(
                    _hist_ax, _col, _xlabel, _title, _ref, energy_mask=mask,
                    show_ylabel=_show_ylabel, axis_idx=_axis_idx,
                )
                _hl.set_x(idx_lo - 0.5)
                _hl.set_width(idx_hi - idx_lo + 1)
                _hl.set_visible(True)
                _hist_ax.figure.canvas.draw_idle()

            def _on_dblclick(event, _box_ax=box_ax):
                if not event.dblclick or event.inaxes is not _box_ax:
                    return
                _redraw_histogram(
                    _hist_ax, _col, _xlabel, _title, _ref, energy_mask=None,
                    show_ylabel=_show_ylabel, axis_idx=_axis_idx,
                )
                _hl.set_visible(False)
                if _span_ref:
                    _span_ref[0].clear()
                _hist_ax.figure.canvas.draw_idle()

            return _on_select, _on_dblclick

        span_ref = []
        on_select, on_dblclick = _make_callback(
            hist_ax, col, xlabel, title, ref, highlight, span_ref, show_ylabel, idx,
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


def link_scatter_to_histogram(
    scatter_axes,
    hist_axes,
    session_data,
    columns,
    colors,
    loaded_ids,
    *,
    n_bins=101,
    hist_ylabel="Probability (%)",
    hist_xlabels=None,
    hist_titles=None,
    hist_refs=None,
    hist_percentile_clip=None,
    tolerance_levels=None,
    tolerance_line_kw=None,
):
    """Install SpanSelectors on scatter axes that filter linked histograms by energy.

    Drag horizontally on a scatter panel (x = energy in MeV) to restrict the
    matching histogram to spots in that energy range.  Double-click the scatter
    panel to reset.

    Args:
        scatter_axes: Single axis or list of scatter axes (x = energy).
        hist_axes: Matching single axis or list of histogram axes.
        session_data: Dict mapping session_id -> data dict with ``"energy"`` key.
        columns: Single column name or list of column names (one per axis pair).
        colors: Session color list.
        loaded_ids: Ordered session id list.
        n_bins: Number of histogram bins.
        hist_ylabel: Y-axis label for histograms.
        hist_xlabels: Single or list of x-axis labels for histograms.
        hist_titles: Single or list of titles for histograms.
        hist_refs: Single or list of reference-line values (vertical) for histograms.
        hist_percentile_clip: Percentile clip for shared bin edges.
        tolerance_levels: Optional sequence of ``(value, color, label)`` tuples;
            draws vertical ``±value`` lines on each histogram after every redraw.
        tolerance_line_kw: Matplotlib kwargs for tolerance lines; defaults to
            :data:`LIMIT_LINE_KW`.

    Returns:
        List of SpanSelector objects. **The caller must keep a reference** to
        prevent garbage collection.
    """
    if not isinstance(scatter_axes, (list, np.ndarray)):
        scatter_axes = [scatter_axes]
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

    tol_kw = tolerance_line_kw or LIMIT_LINE_KW
    selectors = []
    pclip = hist_percentile_clip if hist_percentile_clip is not None else HIST_PERCENTILE_CLIP

    all_hist_vals = []
    for col in columns:
        for sid in loaded_ids:
            if col in session_data[sid]:
                v = np.asarray(session_data[sid][col], dtype=float)
                v = v[np.isfinite(v)]
                if v.size:
                    all_hist_vals.append(v)
    if all_hist_vals:
        combined = np.concatenate(all_hist_vals)
        shared_bin_range = tuple(np.percentile(combined, [100 - pclip, pclip]))
    else:
        shared_bin_range = None

    def _after_histogram(ax):
        if tolerance_levels:
            draw_symmetric_limit_lines(
                ax,
                tolerance_levels,
                orientation="vertical",
                line_kw=tol_kw,
            )

    def _redraw_histogram(hist_ax, col, xlabel, title, ref, energy_mask):
        _plot_histogram(
            hist_ax, session_data, col, loaded_ids, colors,
            energy_mask=energy_mask, n_bins=n_bins, ylabel=hist_ylabel,
            xlabel=xlabel, title=title, ref_val=ref, bin_range=shared_bin_range,
        )
        _after_histogram(hist_ax)

    for scatter_ax, hist_ax, col, xlabel, title, ref in zip(
        scatter_axes, hist_axes, columns, hist_xlabels, hist_titles, hist_refs
    ):
        _redraw_histogram(hist_ax, col, xlabel, title, ref, energy_mask=None)

        highlight = scatter_ax.axvspan(0, 0, alpha=0.15, color="gold", visible=False, zorder=0)

        def _make_callback(_scatter_ax, _hist_ax, _col, _xlabel, _title, _ref, _hl, _span_ref):
            def _on_select(xmin, xmax):
                lo, hi = float(min(xmin, xmax)), float(max(xmin, xmax))
                mask = {}
                for sid in loaded_ids:
                    e = np.asarray(session_data[sid]["energy"], dtype=float)
                    mask[sid] = (e >= lo) & (e <= hi)

                _redraw_histogram(_hist_ax, _col, _xlabel, _title, _ref, energy_mask=mask)
                _hl.set_x(lo)
                _hl.set_width(hi - lo)
                _hl.set_visible(True)
                _hist_ax.figure.canvas.draw_idle()

            def _on_dblclick(event, _sax=_scatter_ax):
                if not event.dblclick or event.inaxes is not _sax:
                    return
                _redraw_histogram(_hist_ax, _col, _xlabel, _title, _ref, energy_mask=None)
                _hl.set_visible(False)
                if _span_ref:
                    _span_ref[0].clear()
                _hist_ax.figure.canvas.draw_idle()

            return _on_select, _on_dblclick

        span_ref = []
        on_select, on_dblclick = _make_callback(
            scatter_ax, hist_ax, col, xlabel, title, ref, highlight, span_ref,
        )

        span = SpanSelector(
            scatter_ax, on_select, "horizontal",
            useblit=True, interactive=True,
            props=dict(alpha=0.2, facecolor="gold", zorder=0),
        )
        span_ref.append(span)
        scatter_ax.figure.canvas.mpl_connect("button_press_event", on_dblclick)
        selectors.append(span)

    visible_axes = [ax for ax in hist_axes if ax.get_visible()]
    if visible_axes:
        y_max = max(ax.get_ylim()[1] for ax in visible_axes)
        for ax in visible_axes:
            ax.set_ylim(0, y_max)

    return selectors
