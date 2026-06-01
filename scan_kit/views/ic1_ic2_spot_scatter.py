"""IC1/IC2 spot position scatter plots (auto G3->G2 fallback).

Each session gets its own row: Planned | IC1 | IC2 | colorbar.
"""

import logging

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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
    """Run IC1/IC2 spot scatter analysis and show matplotlib window."""
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

    n_rows = len(all_session_data)
    row_h = FIG_SIZE_SINGLE[1]
    fig = plt.figure(figsize=(row_h * 3.4, row_h * n_rows))
    gs = fig.add_gridspec(
        n_rows, 4, width_ratios=[1, 1, 1, 0.04], wspace=0.4, hspace=0.45,
    )

    n_g3 = sum(d.get("position_key") == POSITION_KEY_G3 for d in all_session_data)
    mode_label = (
        "G3" if n_g3 == len(all_session_data) else ("G2" if n_g3 == 0 else "mixed G3/G2")
    )
    loaded_ids = [d["session_id"] for d in all_session_data]
    colors = DEFAULT_SESSION_COLORS[: len(loaded_ids)]
    set_view_header(
        fig,
        f"IC1/IC2 Spot Positions ({mode_label})",
        loaded_ids,
        colors,
        base_dir=base_dir,
    )

    for row, session_data in enumerate(all_session_data):
        ax_plan = fig.add_subplot(gs[row, 0])
        ax_ic1 = fig.add_subplot(gs[row, 1], sharex=ax_plan, sharey=ax_plan)
        ax_ic2 = fig.add_subplot(gs[row, 2], sharex=ax_plan, sharey=ax_plan)
        cax = fig.add_subplot(gs[row, 3])
        sid = session_data["session_id"]
        energy = session_data["energy"]
        row_axes = [ax_plan, ax_ic1, ax_ic2]

        norm = mcolors.Normalize(vmin=float(energy.min()), vmax=float(energy.max()))

        for ax in row_axes:
            ax.axhline(y=0, **REFLINE_KW)
            ax.axvline(x=0, **REFLINE_KW)

        has_plan = C_X_POSITION in session_data and C_Y_POSITION in session_data
        if has_plan:
            ax_plan.scatter(
                session_data[C_X_POSITION],
                session_data[C_Y_POSITION],
                c=energy,
                cmap="viridis",
                norm=norm,
                alpha=SCATTER_ALPHA,
                marker="o",
                s=SCATTER_SIZE,
                edgecolors="none",
            )
        ax_plan.set_title(f"Planned — {sid}")

        ax_ic1.scatter(
            session_data["ic1_x"],
            session_data["ic1_y"],
            c=energy,
            cmap="viridis",
            norm=norm,
            alpha=SCATTER_ALPHA,
            marker="o",
            s=SCATTER_SIZE,
            edgecolors="none",
        )
        ax_ic1.set_title(f"IC1 — {sid}")

        ax_ic2.scatter(
            session_data["ic2_x"],
            session_data["ic2_y"],
            c=energy,
            cmap="viridis",
            norm=norm,
            alpha=SCATTER_ALPHA,
            marker="o",
            s=SCATTER_SIZE,
            edgecolors="none",
        )
        ax_ic2.set_title(f"IC2 — {sid}")

        sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
        sm.set_array([])
        fig.colorbar(sm, cax=cax, label="Energy (MeV)")

        for ax in row_axes:
            ax.set_xlabel("X Position (mm)")
            ax.set_ylabel("Y Position (mm)")
            ax.grid(**GRID_KW)
            ax.margins(0.05)
            ax.set_aspect("equal", adjustable="box")

        # Per-session shared square limits across plan/IC1/IC2
        for ax in row_axes:
            ax.autoscale_view()
        lo = min(min(ax.get_xlim()[0], ax.get_ylim()[0]) for ax in row_axes)
        hi = max(max(ax.get_xlim()[1], ax.get_ylim()[1]) for ax in row_axes)
        ax_plan.set_xlim(lo, hi)
        ax_plan.set_ylim(lo, hi)

    plt.show()
