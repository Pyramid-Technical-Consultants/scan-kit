import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib.pyplot as plt

from common import process_position_data

session_ids = ["1750366935"]

# Mag analysis uses voltage columns and position from input_map
extra_spot = ["r_xV_raw", "r_yV_raw", "c_x_raw", "c_y_raw"]
extra_input = ["X_POSITION", "Y_POSITION"]

# Process all sessions
all_session_data = []
for session_id in session_ids:
    data = process_position_data(
        session_id,
        "spot_position_raw",
        extra_spot_columns=extra_spot,
        extra_input_columns=extra_input,
    )
    if data is not None:
        data["xV"] = data["c_x_raw"]
        data["yV"] = data["c_y_raw"]
        data["x"] = data["X_POSITION"]
        data["y"] = data["Y_POSITION"]
        all_session_data.append(data)

# Get all unique energies from all sessions and sort them
all_energies = set()
for data in all_session_data:
    all_energies.update(data["energy"].unique())
energies = sorted(all_energies)

# Create scatter plot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

ax1.set_xlabel("IC1 X Position (mm)")
ax1.set_ylabel("IC1 Y Position (mm)")
ax1.axis("equal")
ax1.grid(True, alpha=0.3)

ax3.set_xlabel("X Amplifier Voltage Command (V)")
ax3.set_ylabel("IC1 X Position (mm)")
ax3.grid(True, alpha=0.3)

ax4.set_xlabel("Y Amplifier Voltage Command (V)")
ax4.set_ylabel("IC1 Y Position (mm)")
ax4.grid(True, alpha=0.3)

# Plot data for each session
for i, session_data in enumerate(all_session_data):
    session_id = session_data["session_id"]
    ic1_x = session_data["ic1_x"]
    ic1_y = session_data["ic1_y"]
    ic2_x = session_data["ic2_x"]
    ic2_y = session_data["ic2_y"]
    xV = session_data["xV"]
    yV = session_data["yV"]
    energy = session_data["energy"]
    x = session_data["x"]
    y = session_data["y"]

    # Plot the xY position of the IC1
    ax1.scatter(
        ic1_x,
        ic1_y,
        c=energy,
        cmap="viridis",
        alpha=0.3,
        label=f"IC1 (Session {session_id})",
        marker="o",
        s=50,
        edgecolors="black",
        linewidth=1,
    )

    # Polynomial fit of ic1_x vs xV
    ratio_ixe = ic1_x / np.asarray(energy) ** 0.5
    fit_ixe = np.polyfit(ratio_ixe, xV, 1)
    ratio_vxe = xV / np.asarray(energy) ** 0.5
    fit_vxe = np.polyfit(ratio_vxe, ic1_x, 1)

    # Polynomial fit of ic1_y vs yV
    ratio_iye = ic1_y / np.asarray(energy) ** 0.5
    fit_iye = np.polyfit(ratio_iye, yV, 1)
    ratio_vye = yV / np.asarray(energy) ** 0.5
    fit_vye = np.polyfit(ratio_vye, ic1_y, 1)

    # Plot predicted xV vs ic1_x
    ax2.scatter(
        np.polyval(fit_vxe, np.polyval(fit_ixe, x)),
        np.polyval(fit_iye, np.polyval(fit_vye, y)),
        c=energy,
        alpha=0.5,
        cmap="viridis",
    )

    ax3.scatter(
        xV,
        ic1_x,
        c=energy,
        alpha=0.5,
        cmap="viridis",
        marker="x",
    )
    ax3.scatter(
        yV,
        ic1_y,
        c=energy,
        alpha=0.5,
        cmap="viridis",
        marker="o",
    )

# Create a colorbar for the energy
sm = plt.cm.ScalarMappable(
    cmap="viridis",
    norm=plt.Normalize(vmin=min(all_energies), vmax=max(all_energies)),
)
sm.set_array([])
plt.colorbar(sm, ax=ax1, label="Energy (MeV)")

ax1.legend(bbox_to_anchor=(0, 1), loc="upper left")

plt.tight_layout()
plt.show()
