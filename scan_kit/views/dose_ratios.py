"""Dose ratio box plots (IC2/IC1, IC3/IC1, IC3/IC2) for G2 and G3 sessions."""

import numpy as np
import matplotlib.pyplot as plt

from ..common import (
    process_position_data,
    plot_boxplots_for_column,
    DEFAULT_SESSION_COLORS,
)

POSITION_KEY_G2 = "spot_raw"
POSITION_KEY_G3 = "spot_position_raw"

EXTRA_SPOT_G3 = [
    "ic1_total_dose_spot_raw",
    "ic2_total_dose_spot_raw",
    "r_ic3_total_dose_spot_raw",
]
EXTRA_SPOT_G2 = [
    "ic1_total_dose_spot_raw",
    "ic2_total_dose_spot_raw",
]


def _process_ratios_session(session_id: str, position_key: str, base_dir: str):
    """Process session and compute dose ratios."""
    extra = EXTRA_SPOT_G3 if position_key == POSITION_KEY_G3 else EXTRA_SPOT_G2
    data = process_position_data(
        session_id, position_key, extra_spot_columns=extra, base_dir=base_dir
    )
    if data is None:
        return None

    ic1_dose = data["ic1_total_dose_spot_raw"]
    ic2_dose = data["ic2_total_dose_spot_raw"]
    data = data.copy()
    data["ic1_dose"] = ic1_dose
    data["ic2_dose"] = ic2_dose
    data["ic21_ratio"] = ((ic2_dose / ic1_dose) - 1.0) * 100.0
    data["ic1_dist"] = np.sqrt(data["ic1_x"] ** 2 + data["ic1_y"] ** 2)
    data["ic2_dist"] = np.sqrt(data["ic2_x"] ** 2 + data["ic2_y"] ** 2)

    if position_key == POSITION_KEY_G3:
        ic3_dose = data["r_ic3_total_dose_spot_raw"]
        data["ic3_dose"] = ic3_dose
        data["ic31_ratio"] = ((ic3_dose / ic1_dose) - 1.0) * 100.0
        data["ic32_ratio"] = ((ic3_dose / ic2_dose) - 1.0) * 100.0

    return data


def run(session_ids: list[str], base_dir: str = "test_data") -> None:
    """Run dose ratios analysis and show matplotlib window."""
    if not session_ids:
        print("No sessions selected")
        return

    session_data = {}
    for sid in session_ids:
        data = _process_ratios_session(sid, POSITION_KEY_G3, base_dir)
        if data is None:
            data = _process_ratios_session(sid, POSITION_KEY_G2, base_dir)
        if data is not None:
            session_data[sid] = data

    if not session_data:
        print("No valid data found for any session")
        return

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2, 2, figsize=(15, 6), sharex=False, sharey=False
    )

    all_energies = set()
    for data in session_data.values():
        all_energies.update(data["energy"].unique())
    energies = sorted(all_energies)

    all_values = np.concatenate([data["ic21_ratio"] for data in session_data.values()])
    min_val = all_values.min()
    max_val = all_values.max()
    padding = (max_val - min_val) * 0.05
    y_min = min_val - padding
    y_max = max_val + padding

    loaded_ids = list(session_data.keys())
    colors = DEFAULT_SESSION_COLORS[: len(loaded_ids)]
    session_labels = [f"Session {sid}" for sid in loaded_ids]

    session_data_g3 = {k: v for k, v in session_data.items() if "ic31_ratio" in v}
    colors_g3 = [colors[loaded_ids.index(sid)] for sid in session_data_g3]

    if session_data_g3:
        plot_boxplots_for_column(
            ax1, session_data_g3, "ic31_ratio", energies, colors_g3, width=0.3
        )
    ax1.set_title("IC3/IC1 Ratio Difference (%)")
    ax1.set_xlabel("Energy (MeV)")
    ax1.set_ylabel("IC3/IC1 Ratio Difference (%)")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(y_min, y_max)
    ax1.set_xticks(np.arange(len(energies)))
    ax1.set_xticklabels([f"{e}" for e in energies], rotation=90)

    plot_boxplots_for_column(
        ax2, session_data, "ic21_ratio", energies, colors, width=0.3
    )
    ax2.set_title("IC2/IC1 Ratio Difference (%)")
    ax2.set_xlabel("Energy (MeV)")
    ax2.set_ylabel("IC2/IC1 Ratio Difference (%)")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(y_min, y_max)
    ax2.set_xticks(np.arange(len(energies)))
    ax2.set_xticklabels([f"{e}" for e in energies], rotation=90)

    if session_data_g3:
        plot_boxplots_for_column(
            ax3, session_data_g3, "ic32_ratio", energies, colors_g3, width=0.3
        )
    ax3.set_title("IC3/IC2 Ratio Difference (%)")
    ax3.set_xlabel("Energy (MeV)")
    ax3.set_ylabel("IC3/IC2 Ratio Difference (%)")
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(y_min, y_max)
    ax3.set_xticks(np.arange(len(energies)))
    ax3.set_xticklabels([f"{e}" for e in energies], rotation=90)

    ax4.axis("off")

    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colors[i], alpha=0.7, label=session_labels[i])
        for i in range(len(loaded_ids))
    ]
    ax1.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    plt.show()
