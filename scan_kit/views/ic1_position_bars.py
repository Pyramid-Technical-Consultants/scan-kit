"""IC1 X/Y Position box plots for multiple sessions."""

import numpy as np
import matplotlib.pyplot as plt

from ..common import (
    process_position_data,
    plot_boxplots_for_column,
    make_session_legend,
    style_energy_axes,
    DEFAULT_SESSION_COLORS,
    FIG_SIZE_1x2,
    SUPTITLE_KW,
    REFLINE_KW,
)


def run(session_ids: list[str], base_dir: str = "test_data") -> None:
    """Run IC1 X/Y position bars analysis and show matplotlib window."""
    if not session_ids:
        print("No sessions selected")
        return

    key = "spot_position_raw"
    plot_x = "ic1_x"
    plot_y = "ic1_y"

    session_data = {}
    for sid in session_ids:
        data = process_position_data(sid, key, base_dir=base_dir)
        if data is not None:
            session_data[sid] = data

    if not session_data:
        print("No valid data found for any session")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_SIZE_1x2)
    fig.suptitle("IC1 X/Y Position Distribution by Energy", **SUPTITLE_KW)

    all_energies = set()
    for data in session_data.values():
        all_energies.update(data["energy"].unique())
    energies = sorted(all_energies)

    all_x_data = np.concatenate([data[plot_x] for data in session_data.values()])
    all_y_data = np.concatenate([data[plot_y] for data in session_data.values()])
    min_val = min(all_x_data.min(), all_y_data.min())
    max_val = max(all_x_data.max(), all_y_data.max())
    padding = (max_val - min_val) * 0.05
    y_min = min_val - padding
    y_max = max_val + padding

    colors = DEFAULT_SESSION_COLORS[: len(session_ids)]

    plot_boxplots_for_column(ax1, session_data, plot_x, energies, colors)
    plot_boxplots_for_column(ax2, session_data, plot_y, energies, colors)

    for ax, col in zip([ax1, ax2], [plot_x, plot_y]):
        ax.set_title(f"{col} Position Distribution")
        style_energy_axes(ax, energies, ylabel=f"{col} Position (mm)")
        ax.set_ylim(y_min, y_max)
        ax.axhline(y=0, **REFLINE_KW)

    make_session_legend(ax1, session_ids, colors)

    plt.tight_layout()
    plt.show()
