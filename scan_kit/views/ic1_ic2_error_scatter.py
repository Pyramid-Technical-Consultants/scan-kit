"""IC1 vs IC2 position error scatter plots."""

import numpy as np
import matplotlib.pyplot as plt

from ..common import process_position_data


def run(session_ids: list[str], base_dir: str = "test_data") -> None:
    """Run IC1 vs IC2 error scatter analysis and show matplotlib window."""
    if not session_ids:
        print("No sessions selected")
        return

    key = "spot_position_raw"

    session_data = {}
    for sid in session_ids:
        data = process_position_data(sid, key, base_dir=base_dir)
        if data is not None:
            data = data.copy()
            data["ic_x_error"] = data["ic1_x"] - data["ic2_x"]
            data["ic_y_error"] = data["ic1_y"] - data["ic2_y"]
            session_data[sid] = data

    if not session_data:
        print("No valid data found for any session")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    all_energies = set()
    for data in session_data.values():
        all_energies.update(data["energy"].unique())
    energies = sorted(all_energies)

    all_ic1_x_data = np.concatenate([data["ic1_x"] for data in session_data.values()])
    x_min = all_ic1_x_data.min()
    x_max = all_ic1_x_data.max()
    x_padding = (x_max - x_min) * 0.05
    x_min -= x_padding
    x_max += x_padding

    all_x_error = np.concatenate([data["ic_x_error"] for data in session_data.values()])
    all_y_error = np.concatenate([data["ic_y_error"] for data in session_data.values()])
    error_min = min(all_x_error.min(), all_y_error.min())
    error_max = max(all_x_error.max(), all_y_error.max())
    error_padding = (error_max - error_min) * 0.05
    error_min -= error_padding
    error_max += error_padding

    def plot_scatter_for_errors(ax, session_data, position_column, error_column):
        for session, (sid, data) in enumerate(session_data.items()):
            marker = "o" if session == 0 else "s"
            ax.scatter(
                data[position_column],
                data[error_column],
                c=data["energy"],
                cmap="viridis",
                alpha=0.2,
                s=30,
                marker=marker,
            )

    plot_scatter_for_errors(ax1, session_data, "ic1_x", "ic_x_error")
    plot_scatter_for_errors(ax2, session_data, "ic1_y", "ic_y_error")

    for ax, title, ylabel in zip(
        [ax1, ax2],
        ["IC X Delta vs IC1 X Position", "IC Y Delta vs IC1 Y Position"],
        ["IC X Delta (mm)", "IC Y Delta (mm)"],
    ):
        ax.set_title(title)
        ax.set_xlabel("IC1 Position (mm)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(error_min, error_max)
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.8)
        ax.axvline(x=0, color="gray", linestyle="--", linewidth=1, alpha=0.8)

    loaded_ids = list(session_data.keys())
    session_legend_elements = [
        plt.Line2D(
            [0], [0],
            marker="o" if i == 0 else "s",
            color="gray",
            markersize=8,
            label=f"Session {loaded_ids[i]}",
        )
        for i in range(len(loaded_ids))
    ]
    ax2.legend(handles=session_legend_elements, loc="upper right", title="Session")

    energy_min = min(all_energies)
    energy_max = max(all_energies)
    sm = plt.cm.ScalarMappable(
        cmap="viridis", norm=plt.Normalize(vmin=energy_min, vmax=energy_max)
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax2, location="right")
    cbar.set_label("Energy (MeV)", fontsize=12)

    plt.suptitle("Position Difference Between IC1 and IC2")
    plt.tight_layout()
    plt.show()
