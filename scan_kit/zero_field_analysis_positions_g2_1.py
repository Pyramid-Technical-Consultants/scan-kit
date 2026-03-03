import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import matplotlib.pyplot as plt

from common import process_position_data

key = "spot_raw"

session_ids = ["590658542"]

# Process all sessions
all_session_data = []
for session_id in session_ids:
    session_data = process_position_data(session_id, key)
    if session_data is not None:
        all_session_data.append(session_data)

if not all_session_data:
    print("No valid session data found!")
    exit()

# Print summary of loaded data
for session_data in all_session_data:
    print(
        f"Session {session_data['session_id']}: {len(session_data['ic1_x'])} valid data points"
    )

# Create scatter plot
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

# Plot data for each session
for i, session_data in enumerate(all_session_data):
    session_id = session_data["session_id"]
    ic1_x = session_data["ic1_x"]
    ic1_y = session_data["ic1_y"]
    ic2_x = session_data["ic2_x"]
    ic2_y = session_data["ic2_y"]
    energy = session_data["energy"]

    scatter1 = plt.scatter(
        ic1_x,
        ic1_y,
        c=energy,
        cmap="viridis",
        alpha=0.5,
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
    f"IC1 Spot Positions\nSessions: {', '.join([data['session_id'] for data in all_session_data])}"
)
plt.legend(bbox_to_anchor=(0, 1), loc="upper left")
plt.grid(True, alpha=0.3)
plt.axis("equal")
plt.tight_layout()
plt.show()

# Export the data back to a csv file
for session_data in all_session_data:
    session_id = session_data["session_id"]
    pd.DataFrame(
        {
            "ic1_x": session_data["ic1_x"],
            "ic1_y": session_data["ic1_y"],
            "ic2_x": session_data["ic2_x"],
            "ic2_y": session_data["ic2_y"],
            "energy": session_data["energy"],
        }
    ).to_csv(f"scan_kit/zero_field_analysis_{session_id}.csv", index=False)
