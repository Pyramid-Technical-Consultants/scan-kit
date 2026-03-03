import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from common import load_session_raw, remap, create_valid_mask

pos_key = "spot_position_raw"
sig_key = "spot_sigma_raw"

# session_ids = ["2078693092", "337175845"]
# session_ids = ["337175845", "2078693092", "1307573499", "1835420917"]
session_ids = ["684740627"]


def process_session_data(session_id):
    """Process data for a single session with position and sigma columns."""
    input_map, spot_data = load_session_raw(session_id)
    if input_map is None or spot_data is None:
        return None

    spot_data_columns = [
        f"r_ic1_x_{pos_key}",
        f"r_ic1_y_{pos_key}",
        f"r_ic2_x_{pos_key}",
        f"r_ic2_y_{pos_key}",
        f"r_ic1_x_{sig_key}",
        f"r_ic1_y_{sig_key}",
        f"r_ic2_x_{sig_key}",
        f"r_ic2_y_{sig_key}",
    ]
    spot_data = spot_data[spot_data_columns].copy().join(input_map["ENERGY"])
    spot_data = spot_data.apply(pd.to_numeric, errors="coerce")

    valid_mask = create_valid_mask(spot_data)
    spot_data_clean = spot_data[valid_mask]

    ic1_x = remap(spot_data_clean[f"r_ic1_x_{pos_key}"], 1, 128, -128, 128)
    ic1_y = remap(spot_data_clean[f"r_ic1_y_{pos_key}"], 1, 128, 128, -128)
    ic2_x = remap(spot_data_clean[f"r_ic2_x_{pos_key}"], 1, 128, 128, -128)
    ic2_y = remap(spot_data_clean[f"r_ic2_y_{pos_key}"], 1, 128, -128, 128)

    ic1_sig_x = spot_data_clean[f"r_ic1_x_{sig_key}"] * 2
    ic1_sig_y = spot_data_clean[f"r_ic1_y_{sig_key}"] * 2
    ic2_sig_x = spot_data_clean[f"r_ic2_x_{sig_key}"] * 2
    ic2_sig_y = spot_data_clean[f"r_ic2_y_{sig_key}"] * 2

    return {
        "session_id": session_id,
        "energy": spot_data_clean["ENERGY"],
        "ic1_x": ic1_x,
        "ic1_y": ic1_y,
        "ic2_x": ic2_x,
        "ic2_y": ic2_y,
        "ic1_sig_x": ic1_sig_x,
        "ic1_sig_y": ic1_sig_y,
        "ic2_sig_x": ic2_sig_x,
        "ic2_sig_y": ic2_sig_y,
    }


# Process both sessions
session_data = {}
for session_id in session_ids:
    session_data[session_id] = process_session_data(session_id)

# Combine data from both sessions for plotting
combined_data = []
for session_id, data in session_data.items():
    if data is not None:
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

# Convert to DataFrame for easier plotting
plot_df = pd.DataFrame(combined_data)

# Create the grouped box plot
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
ax.set_xticklabels([f"{energy}" for energy in unique_energies], rotation=90)
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
