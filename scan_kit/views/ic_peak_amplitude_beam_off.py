"""G3 beam-off peak amplitude histograms (IC1/IC2 X/Y, nA)."""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from ..common import (
    DEFAULT_SESSION_COLORS,
    GRID_KW,
    IC_PEAK_AMPLITUDE_COLUMNS,
    finish_view,
    resolve_concept_column,
    view_grid,
)
from ..common.processing import _detect_beam_off_mask
from ..common.session_source import (
    load_session_timeslice_device_units,
    resolve_session_source,
)

_log = logging.getLogger(__name__)

VIEW_TITLE = "IC Peak Amplitude — Beam-Off (G3)"

_PANELS = tuple(zip(
    IC_PEAK_AMPLITUDE_COLUMNS,
    ("IC1 X", "IC1 Y", "IC2 X", "IC2 Y"),
))

_BINS = 101
_X_HI_PERCENTILE = 99.95
_RCI_TRIGGER = "rci_in_trigger"


def _dedupe_columns(df):
    return df.loc[:, ~df.columns.duplicated()]


def _load_beam_off_peaks(session_id: str, base_dir: str) -> dict[str, np.ndarray] | None:
    src = resolve_session_source(session_id, base_dir)
    if src is None:
        return None

    frames = load_session_timeslice_device_units(
        src,
        usecols=[*IC_PEAK_AMPLITUDE_COLUMNS, _RCI_TRIGGER],
    )
    if not frames:
        return None

    head = _dedupe_columns(frames[0])
    if _detect_beam_off_mask(head) is None:
        _log.debug("Session %s: missing %s", session_id, _RCI_TRIGGER)
        return None

    physical = {
        col: resolve_concept_column(head.columns, col)
        for col in IC_PEAK_AMPLITUDE_COLUMNS
    }
    if any(v is None for v in physical.values()):
        return None

    chunks: dict[str, list[np.ndarray]] = {col: [] for col in IC_PEAK_AMPLITUDE_COLUMNS}
    for frame in frames:
        df = _dedupe_columns(frame)
        beam_off = _detect_beam_off_mask(df)
        if beam_off is None or not np.any(beam_off):
            continue
        for col in IC_PEAK_AMPLITUDE_COLUMNS:
            vals = df[physical[col]].to_numpy(dtype=float, na_value=np.nan)
            chunks[col].append(vals[beam_off])

    out: dict[str, np.ndarray] = {}
    for col in IC_PEAK_AMPLITUDE_COLUMNS:
        if not chunks[col]:
            return None
        vals = np.concatenate(chunks[col])
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return None
        out[col] = vals
    return out


def _panel_x_limits(data: dict[str, dict[str, np.ndarray]], col: str) -> tuple[float, float]:
    vals = np.concatenate([data[sid][col] for sid in data])
    lo = float(np.min(vals))
    hi = float(np.percentile(vals, _X_HI_PERCENTILE))
    return (lo, lo + 1.0) if lo == hi else (lo, hi)


def _labeled_ticks(lo: float, hi: float, target: int) -> tuple[np.ndarray, mticker.Formatter]:
    span = hi - lo
    if span <= 0:
        step = 1.0
    else:
        raw = span / target
        mag = 10.0 ** np.floor(np.log10(raw))
        step = next(
            (mult * mag for mult in (1.0, 2.0, 2.5, 5.0, 10.0) if span / (mult * mag) <= target * 1.25),
            10.0 * mag,
        )
    start = np.ceil(lo / step - 1e-9) * step
    ticks = np.arange(start, hi + step * 0.01, step)
    ticks = ticks[(ticks >= lo - step * 0.01) & (ticks <= hi + step * 0.01)]
    if step >= 10:
        fmt: mticker.Formatter = mticker.FormatStrFormatter("%.0f")
    elif step >= 1:
        fmt = mticker.FormatStrFormatter("%.1f")
    elif step >= 0.1:
        fmt = mticker.FormatStrFormatter("%.2f")
    elif step >= 0.01:
        fmt = mticker.FormatStrFormatter("%.3f")
    else:
        fmt = mticker.FormatStrFormatter("%.4f")
    return ticks, fmt


def _style_panel(ax, x_lo: float, x_hi: float, y_hi: float) -> None:
    for axis, lo, hi, target in (
        (ax.xaxis, x_lo, x_hi, 14),
        (ax.yaxis, 0.0, y_hi, 10),
    ):
        ticks, fmt = _labeled_ticks(lo, hi, target)
        axis.set_ticks(ticks)
        axis.set_major_formatter(fmt)
    ax.grid(axis="x", which="major", **GRID_KW)
    ax.tick_params(axis="both", which="major", labelsize=9)


def _plot_panel(
    ax,
    data: dict[str, dict[str, np.ndarray]],
    col: str,
    session_ids: list[str],
    colors: list[str],
    *,
    x_lo: float,
    x_hi: float,
    title: str,
) -> None:
    edges = np.linspace(x_lo, x_hi, _BINS)
    ymax = 0.0

    for sid, color in zip(session_ids, colors):
        vals = data[sid][col]
        vals = vals[(vals >= x_lo) & (vals <= x_hi)]
        if vals.size == 0:
            continue
        weights = np.full(vals.shape, 100.0 / vals.size)
        counts, _ = np.histogram(vals, bins=edges, weights=weights)
        ymax = max(ymax, float(counts.max()) if counts.size else 0.0)
        ax.hist(vals, bins=edges, weights=weights, alpha=0.5, color=color,
                label=sid, edgecolor="none")

    y_hi = ymax * 1.06 if ymax > 0 else 1.0
    ax.set(xlim=(x_lo, x_hi), ylim=(0.0, y_hi), title=title,
           xlabel="Peak amplitude (nA)", ylabel="Probability (%)")
    _style_panel(ax, x_lo, x_hi, y_hi)


def run(session_ids: list[str], base_dir: str = "test_data", *, settings=None) -> None:
    del settings

    if not session_ids:
        return

    data = {
        sid: peaks
        for sid in session_ids
        if (peaks := _load_beam_off_peaks(sid, base_dir)) is not None
    }
    if not data:
        _log.debug("No G3 beam-off peak amplitude data for selected sessions")
        return

    loaded_ids = list(data.keys())
    colors = DEFAULT_SESSION_COLORS[: len(loaded_ids)]

    fig, axes = view_grid(2, 2)
    for ax, (col, title) in zip(axes.flat, _PANELS):
        x_lo, x_hi = _panel_x_limits(data, col)
        _plot_panel(ax, data, col, loaded_ids, colors, x_lo=x_lo, x_hi=x_hi, title=title)

    finish_view(fig, VIEW_TITLE, loaded_ids, colors, base_dir=base_dir)
    plt.show()
