"""Spot delivery time analysis: total, beam-on, and overhead time."""

import numpy as np
import matplotlib.pyplot as plt

from ..common import (
    C_IC1_CURRENT,
    C_LAYER_ID,
    C_SPOT_NO,
    C_TIMESTAMP,
    C_TIME_S,
    C_TIME_NS,
    resolve_concept_column,
    add_spot_delivery_time,
    filter_data_rows,
    process_position_data,
    plot_boxplots_for_column,
    set_view_header,
    style_energy_axes,
    add_energy_trend,
    link_boxplot_to_histogram,
    DEFAULT_SESSION_COLORS,
    FIG_SIZE_2x2,
    apply_tight_layout,
    try_load_position_data,
)
from ..common.session_source import (
    load_session_timeslice_device_units,
    resolve_session_source,
)

import logging

_log = logging.getLogger(__name__)

EXTRA_SPOT = ["timestamp", "layer_id", "spot_no", "time_s", "time_ns"]

MAX_SPOT_TIME_MS = 100

ON_FRAC = 0.10


def _ensure_timestamp(data: dict) -> dict:
    """Synthesize a ``timestamp`` key (ms) from ``time_s`` + ``time_ns`` when missing."""
    if "timestamp" in data:
        return data
    if "time_s" not in data or "time_ns" not in data:
        return data
    result = dict(data)
    ts = np.asarray(result["time_s"], dtype=float) * 1000.0 + np.asarray(result["time_ns"], dtype=float) / 1e6
    result["timestamp"] = ts
    return result


def _process_session(session_id: str, position_key: str, base_dir: str):
    data = process_position_data(
        session_id, position_key,
        extra_spot_columns=EXTRA_SPOT,
        base_dir=base_dir,
    )
    if data is None:
        return None
    data = _ensure_timestamp(data)
    return add_spot_delivery_time(data, max_spot_time_ms=MAX_SPOT_TIME_MS)


def _resolve_ts_timestamp(df):
    """Return a float timestamp array (ms) from a timeslice frame.

    Prefers a direct ``timestamp`` column; falls back to ``time_s * 1000 + time_ns / 1e6``.
    Returns None when neither is available.
    """
    col_ts = resolve_concept_column(df.columns, C_TIMESTAMP)
    if col_ts is not None:
        return df[col_ts].values.astype(float)

    col_s = resolve_concept_column(df.columns, C_TIME_S)
    col_ns = resolve_concept_column(df.columns, C_TIME_NS)
    if col_s is not None and col_ns is not None:
        return df[col_s].values.astype(float) * 1000.0 + df[col_ns].values.astype(float) / 1e6

    return None


def _compute_beam_on_times(session_id: str, base_dir: str, *, bg_subtract: bool = False):
    """Compute beam-on time per spot from timeslice IC1 thresholding.

    Loads ALL layers (just like beam_on_off_current) and returns a dict
    mapping (layer_id, spot_no) -> beam_on_time_ms, or None on failure.
    """
    src = resolve_session_source(session_id, base_dir)
    if src is None:
        return None

    frames = load_session_timeslice_device_units(src)
    if not frames:
        return None
    if bg_subtract:
        from ..common import subtract_background_frames
        subtract_background_frames(frames)

    df0 = frames[0].loc[:, ~frames[0].columns.duplicated()]
    col_layer = resolve_concept_column(df0.columns, C_LAYER_ID)
    col_spot = resolve_concept_column(df0.columns, C_SPOT_NO)
    col_ic1 = resolve_concept_column(df0.columns, C_IC1_CURRENT)
    if not all([col_layer, col_spot, col_ic1]):
        return None

    result: dict[tuple, float] = {}

    for df in frames:
        df = df.loc[:, ~df.columns.duplicated()]

        df_clean = df.dropna(subset=[col_layer, col_spot, col_ic1])
        if df_clean.empty:
            continue

        timestamps = _resolve_ts_timestamp(df_clean)
        if timestamps is None:
            continue

        layer_id = int(df_clean[col_layer].iloc[0])
        ic1 = df_clean[col_ic1].values.astype(float)

        bg = np.nanpercentile(ic1, 25)
        pk = np.nanpercentile(ic1, 99)
        dyn = pk - bg
        if pk == 0 or abs(dyn / pk) < 0.05:
            continue
        threshold = bg + ON_FRAC * dyn

        beam_on = ic1 > threshold
        ts = timestamps
        spot_nos = df_clean[col_spot].values.astype(int)

        unique_spots = np.unique(spot_nos)
        for sno in unique_spots:
            mask = spot_nos == sno
            spot_on = beam_on[mask]
            spot_ts = ts[mask]
            n_on = spot_on.sum()
            if n_on == 0:
                result[(layer_id, sno)] = 0.0
                continue
            on_ts = spot_ts[spot_on]
            if len(on_ts) >= 2:
                result[(layer_id, sno)] = float(on_ts[-1] - on_ts[0])
            elif len(spot_ts) >= 2:
                dt = np.median(np.diff(spot_ts))
                result[(layer_id, sno)] = float(dt * n_on)
            else:
                result[(layer_id, sno)] = float(n_on)

    return result if result else None


def _merge_beam_on_times(data: dict, beam_on_lookup: dict) -> dict:
    """Add beam_on_time and overhead_time arrays to session data."""
    layer_ids = np.asarray(data["layer_id"], dtype=int)
    spot_nos = np.asarray(data["spot_no"], dtype=int)

    beam_on_arr = np.full(len(layer_ids), np.nan)

    for i in range(len(layer_ids)):
        key = (layer_ids[i], spot_nos[i])
        if key in beam_on_lookup:
            beam_on_arr[i] = beam_on_lookup[key]

    valid = np.isfinite(beam_on_arr)
    if not valid.any():
        return data

    delivery = data["spot_time"]
    overhead = delivery - beam_on_arr

    keep = valid & (overhead >= 0) & (delivery <= MAX_SPOT_TIME_MS)
    filtered = filter_data_rows(data, keep)
    filtered["beam_on_time"] = beam_on_arr[keep]
    filtered["overhead_time"] = overhead[keep]
    filtered["spot_time"] = delivery[keep]
    return filtered


def run(session_ids: list[str], base_dir: str = "test_data", *, settings=None) -> None:
    """Plot spot delivery time, beam-on time, and overhead by energy."""
    if not session_ids:
        _log.debug("No sessions selected")
        return

    session_data: dict = {}
    for sid in session_ids:
        d = try_load_position_data(sid, base_dir, _process_session)
        if d is None:
            continue

        bg = settings.bg_subtract if settings else False
        beam_on = _compute_beam_on_times(sid, base_dir, bg_subtract=bg)
        if beam_on is not None:
            d = _merge_beam_on_times(d, beam_on)
        session_data[sid] = d

    if not session_data:
        _log.debug("No valid spot time data found for any session")
        return

    all_energies: set = set()
    for d in session_data.values():
        all_energies.update(np.unique(d["energy"]))
    energies = sorted(all_energies)

    loaded_ids = list(session_data.keys())
    colors = DEFAULT_SESSION_COLORS[: len(loaded_ids)]

    columns = [
        ("spot_time", "Total Delivery Time", "Spot Time (ms)"),
        ("beam_on_time", "Beam-On Time", "Beam-On Time (ms)"),
        ("overhead_time", "Overhead (Dead) Time", "Overhead Time (ms)"),
    ]
    has_beam_on = any("beam_on_time" in d for d in session_data.values())
    if not has_beam_on:
        columns = columns[:1]

    n_cols = len(columns)
    fig, axes = plt.subplots(
        2, n_cols,
        figsize=(max(12, 6 * n_cols), FIG_SIZE_2x2[1] * 2),
        squeeze=False,
    )

    all_selectors = []
    for col_idx, (col_key, title, xlabel) in enumerate(columns):
        ax_box = axes[0, col_idx]
        ax_hist = axes[1, col_idx]

        col_data = {
            sid: d for sid, d in session_data.items()
            if col_key in d
        }
        col_colors = [colors[loaded_ids.index(sid)] for sid in col_data]
        col_ids = list(col_data.keys())

        if col_data:
            plot_boxplots_for_column(
                ax_box, col_data, col_key, energies, col_colors, width=0.3,
            )
            add_energy_trend(
                ax_box, col_data, col_key, energies, col_colors,
                unit="ms/MeV", position_offset=0.35,
            )

        ax_box.set_title(f"{title} vs Energy")
        style_energy_axes(ax_box, energies, ylabel=xlabel)

        if col_data:
            sels = link_boxplot_to_histogram(
                ax_box, ax_hist,
                col_data, energies, col_key, col_colors, col_ids,
                hist_xlabels=xlabel,
                hist_titles=f"{title} distribution (all energies)",
            )
            all_selectors.extend(sels)

    set_view_header(fig, "Spot Delivery Time Analysis", loaded_ids, colors, base_dir=base_dir)

    apply_tight_layout()
    plt.show()
