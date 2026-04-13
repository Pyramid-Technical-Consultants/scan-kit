"""Spot delivery time analysis: boxplots by energy and overall histogram."""

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
    REFLINE_KW,
)
from .dose_ratios import _add_median_trend_lines

POSITION_KEY_G2 = "spot_raw"
POSITION_KEY_G3 = "spot_position_raw"

EXTRA_SPOT = ["timestamp", "layer_id"]

MAX_SPOT_TIME_MS = 100


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


def run(session_ids: list[str], base_dir: str = "test_data") -> None:
    """Plot spot delivery time by energy (boxplots) and overall distribution."""
    if not session_ids:
        print("No sessions selected")
        return

    session_data: dict = {}
    for sid in session_ids:
        d = _process_session(sid, POSITION_KEY_G3, base_dir)
        if d is None:
            d = _process_session(sid, POSITION_KEY_G2, base_dir)
        if d is not None:
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

    fig, (ax_box, ax_hist) = plt.subplots(
        2, 1, figsize=(max(12, FIG_SIZE_2x2[0]), FIG_SIZE_2x2[1] * 2)
    )
    fig.suptitle("Spot Delivery Time Analysis", **SUPTITLE_KW)

    # Top: boxplots by energy
    plot_boxplots_for_column(
        ax_box, session_data, "spot_time", energies, colors, width=0.3,
    )
    _add_median_trend_lines(
        ax_box, session_data, "spot_time", energies, colors, position_offset=0.35,
    )
    ax_box.set_title("Spot delivery time vs energy")
    style_energy_axes(ax_box, energies, ylabel="Spot Time (ms)")

    make_session_legend(ax_box, loaded_ids, colors)

    # Bottom: interactive histogram linked to boxplot via SpanSelector
    _selectors = link_boxplot_to_histogram(
        ax_box, ax_hist,
        session_data, energies, "spot_time", colors, loaded_ids,
        hist_xlabels="Spot Time (ms)",
        hist_titles="Spot delivery time distribution (all energies)",
    )
    make_session_legend(ax_hist, loaded_ids, colors)

    plt.tight_layout()
    fig.subplots_adjust(top=0.92, hspace=0.35)
    plt.show()
