"""Shared constants and helpers for timeslice replay views."""

from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.widgets import SpanSelector

from ..common import (
    C_ENERGY,
    C_LAYER_ID,
    GRID_KW,
    resolve_concept_column,
)
from ..common.session_source import (
    SessionSource,
    load_session_csv,
    resolve_session_source,
)

TIMELINE_BINS = 4000
DETAIL_MAX_POINTS = 80_000
DETAIL_MARKER_MAX_POINTS = 500
MS_PER_SLICE = 1.0
ZOOM_FACTOR = 0.8
INITIAL_WINDOW_MS = 2000.0
MIN_ZOOM_SPAN_MS = 50.0

DIGITAL_COLUMN_DEFS: tuple[tuple[str, str], ...] = (
    ("rci_in_trigger", "RCI Trigger"),
    ("r_beamOk", "Beam OK"),
)

DIGITAL_LANE_COLORS = ("#e67e22", "#3498db", "#2ecc71", "#9b59b6", "#e74c3c")


def detail_line_style(n_points: int) -> dict:
    """Marker kwargs for detail traces when few enough points are drawn."""
    if n_points <= DETAIL_MARKER_MAX_POINTS:
        return dict(marker=".", markersize=6.0, markeredgewidth=0)
    return {}


def time_axis(ax: plt.Axes, lo_ms: float, hi_ms: float) -> None:
    """Configure *ax* X-axis to display time with adaptive units and dense ticks."""
    span_ms = hi_ms - lo_ms
    if span_ms <= 0:
        return

    if span_ms > 10_000:
        scale = 1e-3
        unit = "s"
    else:
        scale = 1.0
        unit = "ms"

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v * scale:.6g}"
    ))
    ax.set_xlabel(f"Time ({unit})")
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.tick_params(axis="x", which="minor", length=3)


def compress_minmax(signal: np.ndarray, n_bins: int = TIMELINE_BINS):
    """Compress *signal* into a min/max envelope for fast timeline rendering."""
    n = len(signal)
    if n <= n_bins * 2:
        x = np.arange(n, dtype=float) * MS_PER_SLICE
        return x, signal.copy(), signal.copy()

    bin_edges = np.linspace(0, n, n_bins + 1, dtype=int)
    starts = bin_edges[:-1]
    x = (starts + bin_edges[1:]) * 0.5 * MS_PER_SLICE

    safe = np.where(np.isnan(signal), np.inf, signal)
    y_min = np.minimum.reduceat(safe, starts)

    safe_max = np.where(np.isnan(signal), -np.inf, signal)
    y_max = np.maximum.reduceat(safe_max, starts)

    return x, y_min, y_max


def window_xy(
    signal: np.ndarray,
    lo: int,
    hi: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return decimated (time_ms, values) for sample indices [*lo*, *hi*)."""
    n = len(signal)
    a_lo = min(lo, n)
    a_hi = min(hi, n)
    if a_hi <= a_lo:
        return None
    window = signal[a_lo:a_hi]
    w_len = a_hi - a_lo
    if w_len > DETAIL_MAX_POINTS:
        step = max(1, w_len // DETAIL_MAX_POINTS)
        return np.arange(a_lo, a_hi, step) * MS_PER_SLICE, window[::step]
    return np.arange(a_lo, a_hi) * MS_PER_SLICE, window


def resolve_col(columns: Iterable[str], concept: str) -> str | None:
    return resolve_concept_column(columns, concept)


def build_energy_lookups(input_map) -> tuple[dict | None, dict[int, float]] | None:
    """Return (layer_id → energy, layer_idx → energy) lookups from input_map.

    G3 sessions map each spot layer to a distinct ``layer_id``; G2 sessions often
    reuse one ``layer_id`` and rely on frame ``_layer_idx`` instead.
    """
    col_layer = resolve_col(input_map.columns, C_LAYER_ID)
    col_energy = resolve_col(input_map.columns, C_ENERGY)
    if col_energy is None:
        return None

    ordered_energies = list(dict.fromkeys(input_map[col_energy].values))
    energy_by_idx = {i: float(e) for i, e in enumerate(ordered_energies)}

    energy_by_layer: dict | None = None
    if col_layer is not None and input_map[col_layer].nunique() >= 2:
        energy_by_layer = input_map.groupby(col_layer)[col_energy].first().to_dict()

    return energy_by_layer, energy_by_idx


def resolve_frame_energy(
    df,
    frame_idx: int,
    *,
    energy_by_layer: dict | None,
    energy_by_idx: dict[int, float],
    layer_col: str,
) -> float | None:
    """Resolve the beam energy for one timeslice frame."""
    energy = None
    if energy_by_layer is not None and layer_col in df.columns:
        lid = df[layer_col].iloc[0]
        energy = energy_by_layer.get(lid)
    if energy is None and "_layer_idx" in df.columns:
        idx = int(df["_layer_idx"].iloc[0])
        energy = energy_by_idx.get(idx)
    if energy is None:
        energy = energy_by_idx.get(frame_idx)
    if energy is None:
        return None
    return float(energy)


def load_energy_by_layer(
    session_id: str,
    base_dir: str,
) -> tuple[SessionSource, dict] | None:
    """Return session source and layer_id → energy map from input_map."""
    src = resolve_session_source(session_id, base_dir)
    if src is None:
        return None

    input_map = load_session_csv(src, "input_map.csv")
    if input_map is None:
        return None

    col_layer = resolve_col(input_map.columns, C_LAYER_ID)
    col_energy = resolve_col(input_map.columns, C_ENERGY)
    if col_layer is None or col_energy is None:
        return None

    energy_by_layer = input_map.groupby(col_layer)[col_energy].first().to_dict()
    return src, energy_by_layer


def load_energy_lookups(
    session_id: str,
    base_dir: str,
) -> tuple[SessionSource, dict | None, dict[int, float]] | None:
    """Return session source and both layer_id / layer_idx energy lookups."""
    src = resolve_session_source(session_id, base_dir)
    if src is None:
        return None

    input_map = load_session_csv(src, "input_map.csv")
    if input_map is None:
        return None

    lookups = build_energy_lookups(input_map)
    if lookups is None:
        return None

    energy_by_layer, energy_by_idx = lookups
    return src, energy_by_layer, energy_by_idx


def detect_digital_columns(columns: Iterable[str]) -> list[tuple[str, str]]:
    """Return (column_name, display_label) pairs present in *columns*."""
    col_set = set(columns)
    return [(raw, label) for raw, label in DIGITAL_COLUMN_DEFS if raw in col_set]


def build_digital_signals(
    digital_parts: dict[str, list[np.ndarray]],
    digital_cols: list[tuple[str, str]],
) -> dict[str, tuple[np.ndarray, str]]:
    out: dict[str, tuple[np.ndarray, str]] = {}
    for col, label in digital_cols:
        if digital_parts[col]:
            out[col] = (np.concatenate(digital_parts[col]), label)
    return out


def draw_timeline_layer_markers(
    ax: plt.Axes,
    layer_boundaries: list[tuple[int, float]],
    *,
    linewidth: float = 0.3,
    alpha: float = 0.25,
) -> None:
    for offset, _energy in layer_boundaries:
        if offset > 0:
            ax.axvline(offset * MS_PER_SLICE, color="gray", linewidth=linewidth, alpha=alpha)


def draw_detail_layer_markers(
    ax: plt.Axes,
    layer_boundaries: list[tuple[int, float]],
    lo_t: float,
    hi_t: float,
    *,
    annotate: bool = False,
) -> None:
    for offset, energy in layer_boundaries:
        offset_t = offset * MS_PER_SLICE
        if lo_t < offset_t < hi_t:
            ax.axvline(offset_t, color="gray", linewidth=0.5, alpha=0.35)
            if annotate:
                ax.text(
                    offset_t, 1.0, f" {energy:g} MeV",
                    transform=ax.get_xaxis_transform(),
                    fontsize=7, va="top", ha="left",
                    color="gray", alpha=0.7,
                )


def draw_beam_off_edges(
    ax: plt.Axes,
    edges: np.ndarray | None,
    lo: int,
    hi: int,
) -> None:
    if edges is None or not len(edges):
        return
    vis = edges[(edges >= lo) & (edges < hi)]
    for ei in vis:
        ax.axvline(
            ei * MS_PER_SLICE, color="red",
            linewidth=0.6, alpha=0.55, zorder=1,
        )


def plot_beam_on_twin(
    ax: plt.Axes,
    ax_b: plt.Axes,
    beam: np.ndarray,
    lo: int,
    hi: int,
    *,
    color: str,
    label: str | None,
) -> None:
    xy = window_xy(beam, lo, hi)
    if xy is None:
        return
    bx, by = xy
    ax_b.plot(
        bx, by,
        color=color, linewidth=0.4, linestyle=(0, (4, 3)),
        label=label,
    )


def align_beam_zero_to_primary(ax: plt.Axes, ax_b: plt.Axes) -> None:
    """Align the beam twin y=0 with the primary axis y=0."""
    ic_lo, ic_hi = ax.get_ylim()
    b_lo, b_hi = ax_b.get_ylim()
    if ic_hi == ic_lo or b_hi == b_lo:
        return
    ic_frac = -ic_lo / (ic_hi - ic_lo)
    b_frac = -b_lo / (b_hi - b_lo)
    if abs(ic_frac - b_frac) <= 1e-6:
        return
    if ic_frac > b_frac:
        new_b_lo = -ic_frac * (b_hi - b_lo) / (1 - ic_frac) if ic_frac < 1 else b_lo
        ax_b.set_ylim(new_b_lo, b_hi)
    else:
        new_b_hi = -b_lo * (1 - ic_frac) / ic_frac if ic_frac > 0 else b_hi
        ax_b.set_ylim(b_lo, new_b_hi)


def draw_digital_row(
    ax: plt.Axes,
    session_data: dict[str, dict],
    lo: int,
    hi: int,
    lo_t: float,
    hi_t: float,
    *,
    multi: bool,
) -> None:
    ax.clear()

    all_dig_keys: list[str] = []
    for data in session_data.values():
        for k in data.get("digital", {}):
            if k not in all_dig_keys:
                all_dig_keys.append(k)

    lane_height = 1.15
    lane_gap = 0.25
    dig_handles: list = []
    dig_labels_leg: list[str] = []
    for lane_i, col_key in enumerate(all_dig_keys):
        y_base = lane_i * (lane_height + lane_gap)
        color = DIGITAL_LANE_COLORS[lane_i % len(DIGITAL_LANE_COLORS)]
        for si, (sid, data) in enumerate(session_data.items()):
            entry = data.get("digital", {}).get(col_key)
            if entry is None:
                continue
            arr, label = entry
            xy = window_xy(arr, lo, hi)
            if xy is None:
                continue
            px, py = xy
            sig_max = np.nanmax(np.abs(py)) if len(py) else 1.0
            if sig_max > 0:
                py = py / sig_max
            ax.fill_between(
                px, y_base, y_base + py * lane_height,
                step="post", alpha=0.55, color=color,
                linewidth=0,
            )
            if lane_i < len(dig_labels_leg):
                continue
            sig_label = f"{label} ({sid})" if multi else label
            dig_labels_leg.append(sig_label)
            dig_handles.append(plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.55))

    n_lanes = len(all_dig_keys) or 1
    ax.set_xlim(lo_t, hi_t)
    ax.set_ylim(-0.1, n_lanes * (lane_height + lane_gap))
    ax.set_yticks([])
    ax.set_ylabel("Digital", fontsize=8)
    ax.grid(**GRID_KW, which="major")
    if dig_handles:
        ax.legend(
            dig_handles, dig_labels_leg,
            loc="upper right", fontsize=7, ncol=len(dig_labels_leg),
        )
    plt.setp(ax.get_xticklabels(), visible=False)


def connect_timeline_interaction(
    fig: plt.Figure,
    ax_timeline: plt.Axes,
    draw_detail: Callable[[float, float], None],
    max_t: float,
) -> None:
    """Attach span brush, scroll zoom, and initial detail draw."""
    initial_end_t = min(max_t, max(INITIAL_WINDOW_MS * MS_PER_SLICE, max_t / 10))
    draw_detail(0, initial_end_t)

    def _on_select(xmin: float, xmax: float) -> None:
        draw_detail(xmin, xmax)

    span = SpanSelector(
        ax_timeline, _on_select, "horizontal",
        useblit=True, interactive=True,
        props=dict(alpha=0.25, facecolor="gold"),
    )
    fig._scan_kit_span = span  # type: ignore[attr-defined]

    def _on_scroll(event):
        if event.inaxes is not ax_timeline:
            return
        cur_lo, cur_hi = ax_timeline.get_xlim()
        rng = cur_hi - cur_lo
        cx = event.xdata if event.xdata is not None else (cur_lo + cur_hi) / 2
        if event.button == "up":
            new_rng = rng * ZOOM_FACTOR
        elif event.button == "down":
            new_rng = rng / ZOOM_FACTOR
        else:
            return
        new_rng = max(new_rng, MIN_ZOOM_SPAN_MS * MS_PER_SLICE)
        new_rng = min(new_rng, max_t)
        frac = (cx - cur_lo) / rng if rng > 0 else 0.5
        new_lo = cx - frac * new_rng
        new_hi = cx + (1 - frac) * new_rng
        new_lo = max(0, new_lo)
        new_hi = min(max_t, new_hi)
        ax_timeline.set_xlim(new_lo, new_hi)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("scroll_event", _on_scroll)
