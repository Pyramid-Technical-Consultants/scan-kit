"""Spot delivery time analysis: total, beam-on, and overhead time."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..common import (
    process_position_data,
    plot_boxplots_for_column,
    make_session_legend,
    style_energy_axes,
    link_boxplot_to_histogram,
    DEFAULT_SESSION_COLORS,
    FIG_SIZE_2x2,
    SUPTITLE_KW,
)
from ..common.session_source import (
    load_session_timeslice_device_units,
    resolve_session_source,
)
from .dose_ratios import _add_median_trend_lines

POSITION_KEY_G2 = "spot_raw"
POSITION_KEY_G3 = "spot_position_raw"

EXTRA_SPOT = ["timestamp", "layer_id", "spot_no"]

MAX_SPOT_TIME_MS = 100

ON_FRAC = 0.10

_NEEDED_TS_COLS = {"spot_no", "layer_id", "timestamp", "ic1_primary_channel"}


def _process_session(session_id: str, position_key: str, base_dir: str):
    data = process_position_data(
        session_id, position_key,
        extra_spot_columns=EXTRA_SPOT,
        base_dir=base_dir,
    )
    if data is None:
        return None
    if "timestamp" not in data or "layer_id" not in data:
        return None

    data = dict(data)
    df = pd.DataFrame({
        "timestamp": np.asarray(data["timestamp"], dtype=float),
        "layer_id": np.asarray(data["layer_id"]),
    })
    spot_time = df.groupby("layer_id")["timestamp"].diff()
    first_mask = spot_time.isna()
    spot_time.loc[first_mask] = df.loc[first_mask, "timestamp"]
    st = spot_time.values

    keep = st <= MAX_SPOT_TIME_MS
    for key in list(data):
        if key == "session_id":
            continue
        val = data[key]
        if isinstance(val, np.ndarray):
            data[key] = val[keep]
        else:
            data[key] = val.iloc[keep.nonzero()[0]]
    data["spot_time"] = st[keep]
    return data


def _compute_beam_on_times(session_id: str, base_dir: str):
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

    result: dict[tuple, float] = {}

    for df in frames:
        df = df.loc[:, ~df.columns.duplicated()]
        if not _NEEDED_TS_COLS.issubset(df.columns):
            continue

        df = df.dropna(subset=["layer_id", "spot_no", "ic1_primary_channel"])
        if df.empty:
            continue

        layer_id = int(df["layer_id"].iloc[0])
        ic1 = df["ic1_primary_channel"].values.astype(float)

        bg = np.nanpercentile(ic1, 25)
        pk = np.nanpercentile(ic1, 99)
        dyn = pk - bg
        if dyn < 1.0:
            continue
        threshold = bg + ON_FRAC * dyn

        beam_on = ic1 > threshold
        timestamps = df["timestamp"].values.astype(float)
        spot_nos = df["spot_no"].values.astype(int)

        unique_spots = np.unique(spot_nos)
        for sno in unique_spots:
            mask = spot_nos == sno
            spot_on = beam_on[mask]
            spot_ts = timestamps[mask]
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
    for key in list(data):
        if key == "session_id":
            continue
        val = data[key]
        if isinstance(val, np.ndarray):
            data[key] = val[keep]
        else:
            data[key] = val.iloc[keep.nonzero()[0]]

    data["beam_on_time"] = beam_on_arr[keep]
    data["overhead_time"] = overhead[keep]
    data["spot_time"] = delivery[keep]
    return data


def run(session_ids: list[str], base_dir: str = "test_data") -> None:
    """Plot spot delivery time, beam-on time, and overhead by energy."""
    if not session_ids:
        print("No sessions selected")
        return

    session_data: dict = {}
    for sid in session_ids:
        d = _process_session(sid, POSITION_KEY_G3, base_dir)
        if d is None:
            d = _process_session(sid, POSITION_KEY_G2, base_dir)
        if d is None:
            continue

        beam_on = _compute_beam_on_times(sid, base_dir)
        if beam_on is not None:
            d = _merge_beam_on_times(d, beam_on)
        session_data[sid] = d

    if not session_data:
        print("No valid spot time data found for any session")
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
    fig.suptitle("Spot Delivery Time Analysis", **SUPTITLE_KW)

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
            _add_median_trend_lines(
                ax_box, col_data, col_key, energies, col_colors,
                position_offset=0.35,
            )

        ax_box.set_title(f"{title} vs Energy")
        style_energy_axes(ax_box, energies, ylabel=xlabel)

        if col_idx == 0:
            make_session_legend(ax_box, loaded_ids, colors)

        if col_data:
            sels = link_boxplot_to_histogram(
                ax_box, ax_hist,
                col_data, energies, col_key, col_colors, col_ids,
                hist_xlabels=xlabel,
                hist_titles=f"{title} distribution (all energies)",
            )
            all_selectors.extend(sels)
            make_session_legend(ax_hist, col_ids, col_colors)

    plt.tight_layout()
    fig.subplots_adjust(top=0.92, hspace=0.35)
    plt.show()
