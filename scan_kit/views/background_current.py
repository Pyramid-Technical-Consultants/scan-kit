"""Background current analysis (IC1, IC2, IC3) from timeslice data."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..common import (
    load_csv_from_zip,
    load_timeslice_device_units,
    plot_boxplots_for_column,
    make_session_legend,
    style_energy_axes,
    DEFAULT_SESSION_COLORS,
    FIG_SIZE_2x2,
    SUPTITLE_KW,
)

_TIMESLICE_COLS = [
    "layer_id",
    "rci_in_dose_enable",
    "ic1_primary_channel",
    "ic2_primary_channel",
    "ic3_current_A",
    "ic3_current_B",
    "ic3_current_C",
    "ic3_current_D",
    "timesliceNumber",
]


def _load_background_data(
    session_id: str, base_dir: str
) -> pd.DataFrame | None:
    """Load timeslice data for all layers, filter to beam-off, and map to energy.

    Returns a DataFrame with columns: energy, ic1_bg, ic2_bg, ic3_bg,
    layer_idx, timeslice — or None on failure.
    """
    zip_path = str(Path(base_dir) / f"{session_id}.zip")

    input_map = load_csv_from_zip(zip_path, "input_map.csv", session_id)
    if input_map is None:
        return None

    energy_by_layer = (
        input_map.groupby("layer_id")["ENERGY"].first().to_dict()
    )

    frames = load_timeslice_device_units(
        zip_path, session_id, usecols=_TIMESLICE_COLS
    )
    if not frames:
        return None

    parts: list[pd.DataFrame] = []
    for df in frames:
        bg = df[df["rci_in_dose_enable"] == 0].copy()
        if bg.empty:
            continue
        energy = bg["layer_id"].map(energy_by_layer)
        parts.append(
            pd.DataFrame(
                {
                    "energy": energy,
                    "ic1_bg": bg["ic1_primary_channel"].values,
                    "ic2_bg": bg["ic2_primary_channel"].values,
                    "ic3_bg": (
                        bg["ic3_current_A"]
                        + bg["ic3_current_B"]
                        + bg["ic3_current_C"]
                        + bg["ic3_current_D"]
                    ).values,
                    "layer_idx": bg["_layer_idx"].values,
                    "timeslice": bg["timesliceNumber"].values,
                }
            )
        )

    if not parts:
        return None

    result = pd.concat(parts, ignore_index=True).dropna(subset=["energy"])
    return result if not result.empty else None


def run(session_ids: list[str], base_dir: str = "test_data") -> None:
    """Run background current analysis and show matplotlib window."""
    if not session_ids:
        print("No sessions selected")
        return

    session_data: dict[str, dict] = {}
    for sid in session_ids:
        df = _load_background_data(sid, base_dir)
        if df is not None:
            session_data[sid] = {
                "energy": df["energy"].values,
                "ic1_bg": df["ic1_bg"].values,
                "ic2_bg": df["ic2_bg"].values,
                "ic3_bg": df["ic3_bg"].values,
            }

    if not session_data:
        print("No valid background current data found for any session")
        return

    all_energies: set[float] = set()
    for data in session_data.values():
        all_energies.update(np.unique(data["energy"]))
    energies = sorted(all_energies)

    loaded_ids = list(session_data.keys())
    colors = DEFAULT_SESSION_COLORS[: len(loaded_ids)]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2, 2, figsize=FIG_SIZE_2x2, sharex=False, sharey=False
    )
    fig.suptitle("Background Current (beam off)", **SUPTITLE_KW)

    for ax, col, title in [
        (ax1, "ic1_bg", "IC1 Background Current"),
        (ax2, "ic2_bg", "IC2 Background Current"),
        (ax3, "ic3_bg", "IC3 Background Current (sum A+B+C+D)"),
    ]:
        plot_boxplots_for_column(
            ax, session_data, col, energies, colors, width=0.3
        )
        ax.set_title(title)
        style_energy_axes(ax, energies, ylabel="Current (device units)")

    ax4.axis("off")
    make_session_legend(ax4, loaded_ids, colors, loc="upper left")

    for i, sid in enumerate(loaded_ids):
        d = session_data[sid]
        lines = [
            f"IC1  mean={np.mean(d['ic1_bg']):+.3f}  std={np.std(d['ic1_bg']):.3f}",
            f"IC2  mean={np.mean(d['ic2_bg']):+.3f}  std={np.std(d['ic2_bg']):.3f}",
            f"IC3  mean={np.mean(d['ic3_bg']):+.3f}  std={np.std(d['ic3_bg']):.3f}",
        ]
        y_start = 0.55 - i * 0.35
        ax4.text(
            0.05, y_start, f"Session {sid}",
            transform=ax4.transAxes, fontsize=10, fontweight="bold",
            color=colors[i], va="top",
        )
        ax4.text(
            0.05, y_start - 0.05, "\n".join(lines),
            transform=ax4.transAxes, fontsize=9, fontfamily="monospace",
            va="top",
        )

    plt.tight_layout()
    plt.show()
