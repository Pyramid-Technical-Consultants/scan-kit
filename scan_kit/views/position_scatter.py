"""Spot position XY scatter plots (auto G3->G2 fallback).

One row of three columns — Planned | IC1 | IC2 — with every selected session
overlaid and colored by its session color.
"""

import logging

import matplotlib.pyplot as plt

from ..common import (
    C_X_POSITION,
    C_Y_POSITION,
    POSITION_KEY_G3,
    process_position_data,
    try_load_position_data,
    DEFAULT_SESSION_COLORS,
    set_view_header,
    FIG_SIZE_SINGLE,
    GRID_KW,
    REFLINE_KW,
    SCATTER_ALPHA,
    SCATTER_SIZE,
)

_log = logging.getLogger(__name__)


def _process_session(session_id: str, position_key: str, base_dir: str):
    data = process_position_data(
        session_id,
        position_key,
        extra_input_columns=[C_X_POSITION, C_Y_POSITION],
        base_dir=base_dir,
    )
    if data is None:
        return None
    out = dict(data)
    out["position_key"] = position_key
    return out


def run(session_ids: list[str], base_dir: str = "test_data", *, settings=None) -> None:
    """Run spot position scatter analysis and show matplotlib window."""
    if not session_ids:
        _log.debug("No sessions selected")
        return

    all_session_data = []
    for sid in session_ids:
        data = try_load_position_data(sid, base_dir, _process_session, raw=False)
        if data is not None:
            all_session_data.append(data)

    if not all_session_data:
        _log.debug("No valid session data found!")
        return

    n_g3 = sum(d.get("position_key") == POSITION_KEY_G3 for d in all_session_data)
    mode_label = (
        "G3" if n_g3 == len(all_session_data) else ("G2" if n_g3 == 0 else "mixed G3/G2")
    )
    loaded_ids = [d["session_id"] for d in all_session_data]
    colors = DEFAULT_SESSION_COLORS[: len(loaded_ids)]
    color_by_sid = dict(zip(loaded_ids, colors))

    h = FIG_SIZE_SINGLE[1]
    fig, (ax_plan, ax_ic1, ax_ic2) = plt.subplots(
        1, 3, figsize=(h * 3.2, h), sharex=True, sharey=True,
    )
    row_axes = [ax_plan, ax_ic1, ax_ic2]

    set_view_header(
        fig,
        f"Spot Positions ({mode_label})",
        loaded_ids,
        colors,
        base_dir=base_dir,
    )

    for ax in row_axes:
        ax.axhline(y=0, **REFLINE_KW)
        ax.axvline(x=0, **REFLINE_KW)

    def _scatter(ax, xs, ys, color):
        ax.scatter(
            xs,
            ys,
            color=color,
            alpha=SCATTER_ALPHA,
            marker="o",
            s=SCATTER_SIZE,
            edgecolors="none",
        )

    for session_data in all_session_data:
        sid = session_data["session_id"]
        color = color_by_sid[sid]

        if C_X_POSITION in session_data and C_Y_POSITION in session_data:
            _scatter(ax_plan, session_data[C_X_POSITION], session_data[C_Y_POSITION], color)
        _scatter(ax_ic1, session_data["ic1_x"], session_data["ic1_y"], color)
        _scatter(ax_ic2, session_data["ic2_x"], session_data["ic2_y"], color)

    for ax, title in zip(row_axes, ["Planned", "IC1", "IC2"]):
        ax.set_title(title)
        ax.set_xlabel("X Position (mm)")
        ax.set_ylabel("Y Position (mm)")
        ax.grid(**GRID_KW)
        ax.margins(0.05)
        ax.set_aspect("equal", adjustable="box")

    # Shared square limits across plan/IC1/IC2.
    for ax in row_axes:
        ax.autoscale_view()
    lo = min(min(ax.get_xlim()[0], ax.get_ylim()[0]) for ax in row_axes)
    hi = max(max(ax.get_xlim()[1], ax.get_ylim()[1]) for ax in row_axes)
    ax_plan.set_xlim(lo, hi)
    ax_plan.set_ylim(lo, hi)

    plt.show()
