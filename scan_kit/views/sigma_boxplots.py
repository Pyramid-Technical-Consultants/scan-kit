"""Sigma X/Y box plots by energy."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from ..common import load_session_raw, create_valid_mask

POS_KEY = "spot_position_raw"
SIG_KEY = "spot_sigma_raw"


def _process_session_data(session_id: str, base_dir: str):
    """Process data for a single session with position and sigma columns."""
    input_map, spot_data = load_session_raw(session_id, base_dir=base_dir)
    if input_map is None or spot_data is None:
        return None

    spot_data_columns = [
        f"r_ic1_x_{POS_KEY}",
        f"r_ic1_y_{POS_KEY}",
        f"r_ic2_x_{POS_KEY}",
        f"r_ic2_y_{POS_KEY}",
        f"r_ic1_x_{SIG_KEY}",
        f"r_ic1_y_{SIG_KEY}",
        f"r_ic2_x_{SIG_KEY}",
        f"r_ic2_y_{SIG_KEY}",
    ]
    for col in spot_data_columns:
        if col not in spot_data.columns:
            return None

    spot_data = spot_data[spot_data_columns].copy().join(input_map["ENERGY"])
    spot_data = spot_data.apply(pd.to_numeric, errors="coerce")

    valid_mask = create_valid_mask(spot_data)
    spot_data_clean = spot_data[valid_mask]

    ic1_sig_x = spot_data_clean[f"r_ic1_x_{SIG_KEY}"] * 2
    ic1_sig_y = spot_data_clean[f"r_ic1_y_{SIG_KEY}"] * 2
    ic2_sig_x = spot_data_clean[f"r_ic2_x_{SIG_KEY}"] * 2
    ic2_sig_y = spot_data_clean[f"r_ic2_y_{SIG_KEY}"] * 2

    return {
        "session_id": session_id,
        "energy": spot_data_clean["ENERGY"],
        "ic1_sig_x": ic1_sig_x,
        "ic1_sig_y": ic1_sig_y,
        "ic2_sig_x": ic2_sig_x,
        "ic2_sig_y": ic2_sig_y,
    }


def run(session_ids: list[str], base_dir: str = "test_data") -> None:
    """Run sigma box plots analysis and show matplotlib window."""
    if not session_ids:
        print("No sessions selected")
        return

    session_data = {}
    for sid in session_ids:
        data = _process_session_data(sid, base_dir)
        if data is not None:
            session_data[sid] = data

    if not session_data:
        print("No valid data found for any session")
        return

    combined_data = []
    for session_id, data in session_data.items():
        for i in range(len(data["energy"])):
            for sigma_type, sigma_key in [
                ("ic1_sig_x", "ic1_sig_x"),
                ("ic1_sig_y", "ic1_sig_y"),
                ("ic2_sig_x", "ic2_sig_x"),
                ("ic2_sig_y", "ic2_sig_y"),
            ]:
                combined_data.append({
                    "energy": data["energy"].iloc[i],
                    "sigma_value": data[sigma_key].iloc[i],
                    "sigma_type": sigma_type,
                    "session_id": session_id,
                })

    plot_df = pd.DataFrame(combined_data)

    fig, ax = plt.subplots(figsize=(14, 8))

    unique_energies = sorted(plot_df["energy"].unique())
    unique_sigma_types = ["ic1_sig_x", "ic1_sig_y", "ic2_sig_x", "ic2_sig_y"]

    box_data_by_energy = {}
    for energy in unique_energies:
        box_data_by_energy[energy] = {}
        for sigma_type in unique_sigma_types:
            subset = plot_df[
                (plot_df["energy"] == energy) & (plot_df["sigma_type"] == sigma_type)
            ]
            if not subset.empty:
                box_data_by_energy[energy][sigma_type] = subset["sigma_value"].values

    width = 0.2
    x_positions = np.arange(len(unique_energies))
    colors = ["skyblue", "lightcoral", "limegreen", "orange"]

    for i, sigma_type in enumerate(unique_sigma_types):
        data_for_type = []
        positions = []
        for j, energy in enumerate(unique_energies):
            if energy in box_data_by_energy and sigma_type in box_data_by_energy[energy]:
                data_for_type.append(box_data_by_energy[energy][sigma_type])
                positions.append(x_positions[j] + (i - 1.5) * width)

        if data_for_type:
            ax.boxplot(
                data_for_type,
                positions=positions,
                widths=width * 0.8,
                patch_artist=True,
                showfliers=False,
                boxprops=dict(facecolor=colors[i], alpha=0.7),
                medianprops=dict(color="black", linewidth=1.5),
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{e}" for e in unique_energies], rotation=90)
    ax.set_xlabel("Energy (MeV)", fontsize=12)
    ax.set_ylabel("Sigma (mm)", fontsize=12)
    ax.set_title("Sigma X and Y Grouped by Energy", fontsize=14)
    ax.grid(True, alpha=0.3)

    legend_elements = [
        Patch(facecolor=colors[i], alpha=0.7, label=unique_sigma_types[i])
        for i in range(len(unique_sigma_types))
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    plt.show()
