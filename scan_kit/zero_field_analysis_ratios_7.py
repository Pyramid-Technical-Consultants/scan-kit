import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib.pyplot as plt

from common import process_position_data, plot_boxplots_for_column, DEFAULT_SESSION_COLORS

position_key_g2 = "spot_raw"
position_key_g3 = "spot_position_raw"

session_ids_g2 = []
# session_ids_g2 = ["590658542"]
# session_ids_g3 = ["1968011512"]
session_ids_g3 = ["1022244633"]
# session_ids_g3 = ["619692735", "1503314831", "737490228"]
session_ids = session_ids_g3 + session_ids_g2

# Extra columns for G3 (includes IC3 dose)
extra_spot_g3 = [
    "ic1_total_dose_spot_raw",
    "ic2_total_dose_spot_raw",
    "r_ic3_total_dose_spot_raw",
]
# Extra columns for G2 (no IC3)
extra_spot_g2 = [
    "ic1_total_dose_spot_raw",
    "ic2_total_dose_spot_raw",
]


def process_ratios_session(session_id, position_key):
    """Process session and compute dose ratios."""
    extra = extra_spot_g3 if position_key == position_key_g3 else extra_spot_g2
    data = process_position_data(session_id, position_key, extra_spot_columns=extra)
    if data is None:
        return None

    ic1_dose = data["ic1_total_dose_spot_raw"]
    ic2_dose = data["ic2_total_dose_spot_raw"]
    data["ic1_dose"] = ic1_dose
    data["ic2_dose"] = ic2_dose
    data["ic21_ratio"] = ((ic2_dose / ic1_dose) - 1.0) * 100.0
    data["ic1_dist"] = np.sqrt(data["ic1_x"] ** 2 + data["ic1_y"] ** 2)
    data["ic2_dist"] = np.sqrt(data["ic2_x"] ** 2 + data["ic2_y"] ** 2)

    if position_key == position_key_g3:
        ic3_dose = data["r_ic3_total_dose_spot_raw"]
        data["ic3_dose"] = ic3_dose
        data["ic31_ratio"] = ((ic3_dose / ic1_dose) - 1.0) * 100.0
        data["ic32_ratio"] = ((ic3_dose / ic2_dose) - 1.0) * 100.0

    return data


# Process both sessions
session_data = {}
for session_id in session_ids_g3:
    data = process_ratios_session(session_id, position_key_g3)
    if data is not None:
        session_data[session_id] = data

for session_id in session_ids_g2:
    data = process_ratios_session(session_id, position_key_g2)
    if data is not None:
        session_data[session_id] = data

if not session_data:
    print("No valid data found for any session")
    exit()

# Create figure with subplots
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

colors = DEFAULT_SESSION_COLORS[: len(session_ids)]
session_labels = [f"Session {sid}" for sid in session_ids]

# Plot box plots
plot_boxplots_for_column(
    ax1, session_data, "ic31_ratio", energies, colors, width=0.3
)
plot_boxplots_for_column(
    ax2, session_data, "ic21_ratio", energies, colors, width=0.3
)
plot_boxplots_for_column(
    ax3, session_data, "ic32_ratio", energies, colors, width=0.3
)

for ax, title in zip(
    [ax1, ax2, ax3],
    [
        "IC3/IC1 Ratio Difference (%)",
        "IC2/IC1 Ratio Difference (%)",
        "IC3/IC2 Ratio Difference (%)",
    ],
):
    ax.set_xlabel("Energy (MeV)")
    ax.set_ylabel(title)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(np.arange(len(energies)))
    ax.set_xticklabels([f"{energy}" for energy in energies], rotation=90)

legend_elements = [
    plt.Rectangle((0, 0), 1, 1, facecolor=colors[i], alpha=0.7, label=session_labels[i])
    for i in range(len(session_ids))
]
ax1.legend(handles=legend_elements, loc="upper right")

plt.tight_layout()
plt.show()
