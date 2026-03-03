"""IC1/IC2 spot position scatter plots (G3)."""

import matplotlib.pyplot as plt

from ..common import process_position_data


def run(session_ids: list[str], base_dir: str = "test_data") -> None:
    """Run IC1/IC2 spot scatter (G3) analysis and show matplotlib window."""
    if not session_ids:
        print("No sessions selected")
        return

    key = "spot_position_raw"

    all_session_data = []
    for sid in session_ids:
        data = process_position_data(sid, key, base_dir=base_dir)
        if data is not None:
            all_session_data.append(data)

    if not all_session_data:
        print("No valid session data found!")
        return

    plt.figure(figsize=(12, 10))

    center = 0
    plt.axhline(
        center,
        linestyle="--",
        color="gray",
        alpha=0.5,
        linewidth=1,
        label=f"Center ({center})",
    )
    plt.axvline(
        center,
        linestyle="--",
        color="gray",
        alpha=0.5,
        linewidth=1,
    )

    for session_data in all_session_data:
        session_id = session_data["session_id"]
        ic1_x = session_data["ic1_x"]
        ic1_y = session_data["ic1_y"]
        energy = session_data["energy"]

        scatter1 = plt.scatter(
            ic1_x,
            ic1_y,
            c=energy,
            cmap="viridis",
            alpha=0.3,
            label=f"IC1 (Session {session_id})",
            marker="o",
            s=50,
            edgecolors="black",
            linewidth=0,
        )

    plt.colorbar(scatter1, label="Energy (MeV)")
    plt.xlabel("X Position (mm)")
    plt.ylabel("Y Position (mm)")
    plt.title(
        f"IC1 and IC2 Spot Positions\nSessions: {', '.join([d['session_id'] for d in all_session_data])}"
    )
    plt.legend(bbox_to_anchor=(0, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.tight_layout()
    plt.show()
