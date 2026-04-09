"""IC1 spot position scatter plots (G2)."""

import matplotlib.pyplot as plt

from ..common import (
    process_position_data,
    FIG_SIZE_SINGLE,
    SUPTITLE_KW,
    GRID_KW,
    REFLINE_KW,
    SCATTER_ALPHA,
    SCATTER_SIZE,
)


def run(session_ids: list[str], base_dir: str = "test_data") -> None:
    """Run IC1 spot scatter (G2) analysis and show matplotlib window."""
    if not session_ids:
        print("No sessions selected")
        return

    key = "spot_raw"

    all_session_data = []
    for sid in session_ids:
        data = process_position_data(sid, key, base_dir=base_dir)
        if data is not None:
            all_session_data.append(data)

    if not all_session_data:
        print("No valid session data found!")
        return

    fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE)
    sids = ", ".join(d["session_id"] for d in all_session_data)
    fig.suptitle(f"IC1 Spot Positions (G2) — Sessions: {sids}", **SUPTITLE_KW)

    ax.axhline(y=0, **REFLINE_KW, label="Center")
    ax.axvline(x=0, **REFLINE_KW)

    for session_data in all_session_data:
        session_id = session_data["session_id"]
        scatter1 = ax.scatter(
            session_data["ic1_x"],
            session_data["ic1_y"],
            c=session_data["energy"],
            cmap="viridis",
            alpha=SCATTER_ALPHA,
            label=f"IC1 (Session {session_id})",
            marker="o",
            s=SCATTER_SIZE,
            edgecolors="none",
        )

    plt.colorbar(scatter1, ax=ax, label="Energy (MeV)")
    ax.set_xlabel("X Position (mm)")
    ax.set_ylabel("Y Position (mm)")
    ax.legend(loc="upper left")
    ax.grid(**GRID_KW)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()
