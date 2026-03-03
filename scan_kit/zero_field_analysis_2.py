import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import matplotlib.pyplot as plt

from common import process_position_data

key = "spot_position_raw"

session_id = "1835420917"
# session_id = "337175845"
# session_ids = ["337175845", "1307573499", "1835420917"]

data = process_position_data(session_id, key)

if data is None:
    print("No valid data found")
    exit()

# Create figure with subplots for IC1 x and y box plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Prepare data for box plots - group by energy
df = pd.DataFrame(
    {"ic1_x": data["ic1_x"], "ic1_y": data["ic1_y"], "energy": data["energy"]}
)

# Get unique energy values and sort them
energies = sorted(df["energy"].unique())

# Determine common axis limits for both plots
all_x_data = df["ic1_x"].values
all_y_data = df["ic1_y"].values
min_val = min(all_x_data.min(), all_y_data.min())
max_val = max(all_x_data.max(), all_y_data.max())

# Add some padding to the limits
padding = (max_val - min_val) * 0.05
y_min = min_val - padding
y_max = max_val + padding

# IC1 X box plot
ic1_x_data = [df[df["energy"] == energy]["ic1_x"].values for energy in energies]
bp1 = ax1.boxplot(
    ic1_x_data,
    labels=[f"{e:g}" for e in energies],
    patch_artist=False,
    showfliers=False,
)
ax1.set_title("IC1 X Position Distribution by Energy")
ax1.set_xlabel("Energy (MeV)")
ax1.set_ylabel("IC1 X Position")
ax1.grid(True, alpha=0.3)
ax1.set_ylim(y_min, y_max)
ax1.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.8)
ax1.tick_params(axis='x', rotation=90)

# IC1 Y box plot
ic1_y_data = [df[df["energy"] == energy]["ic1_y"].values for energy in energies]
bp2 = ax2.boxplot(
    ic1_y_data,
    labels=[f"{e:g}" for e in energies],
    patch_artist=False,
    showfliers=False,
)
ax2.set_title("IC1 Y Position Distribution by Energy")
ax2.set_xlabel("Energy (MeV)")
ax2.set_ylabel("IC1 Y Position")
ax2.grid(True, alpha=0.3)
ax2.set_ylim(y_min, y_max)
ax2.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.8)
ax2.tick_params(axis='x', rotation=90)

# Adjust layout and show
plt.tight_layout()
plt.suptitle(f"IC1 Position Distribution by Energy - Session {session_id}", y=1.02)
plt.show()
