import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib.pyplot as plt

from common import process_position_data, plot_boxplots_for_column, DEFAULT_SESSION_COLORS

key = "spot_position_raw"

session_ids = ["716678478"]

# Process both sessions
session_data = {}
for session_id in session_ids:
    data = process_position_data(session_id, key)
    if data is not None:
        session_data[session_id] = data

if not session_data:
    print("No valid data found for any session")
    exit()

# Create figure with subplots for IC1 x and y box plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Get all unique energies from all sessions and sort them
all_energies = set()
for data in session_data.values():
    all_energies.update(data["energy"].unique())
energies = sorted(all_energies)

# Determine common axis limits for both plots
all_x_data = np.concatenate([data["ic1_x"] for data in session_data.values()])
all_y_data = np.concatenate([data["ic1_y"] for data in session_data.values()])
min_val = min(all_x_data.min(), all_y_data.min())
max_val = max(all_x_data.max(), all_y_data.max())
padding = (max_val - min_val) * 0.05
y_min = min_val - padding
y_max = max_val + padding

colors = DEFAULT_SESSION_COLORS[: len(session_ids)]
session_labels = [f"Session {sid}" for sid in session_ids]

# Plot box plots for both IC1 X and Y
plot_boxplots_for_column(ax1, session_data, "ic1_x", energies, colors)
plot_boxplots_for_column(ax2, session_data, "ic1_y", energies, colors)

# Configure axes
for ax, title in zip(
    [ax1, ax2],
    ["IC1 X Position Distribution by Energy", "IC1 Y Position Distribution by Energy"],
):
    ax.set_title(title)
    ax.set_xlabel("Energy (MeV)")
    ax.set_ylabel("IC1 X Position (mm)" if ax == ax1 else "IC1 Y Position (mm)")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(y_min, y_max)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.8)
    ax.set_xticks(np.arange(len(energies)))
    ax.set_xticklabels([f"{energy}" for energy in energies], rotation=90)

# Add legend
legend_elements = [
    plt.Rectangle((0, 0), 1, 1, facecolor=colors[i], alpha=0.7, label=session_labels[i])
    for i in range(len(session_ids))
]
ax1.legend(handles=legend_elements, loc="upper right")
ax2.legend(handles=legend_elements, loc="upper right")

# Adjust layout and show
plt.tight_layout()
plt.suptitle(f"IC1 Position Distribution by Energy - Both Sessions", y=1.02)
plt.show()
