"""Sigma X/Y box plots by energy."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..common import (
    load_session_raw,
    create_valid_mask,
    resolve_concept_column,
    C_ENERGY,
    DEFAULT_SESSION_COLORS,
    FIG_SIZE_2x2,
    set_view_header,
    apply_tight_layout,
    GRID_KW,
)

import logging

_log = logging.getLogger(__name__)

_SIG_KEY_VARIANTS = ("spot_sigma_raw", "spot_sigma")

_IC_SIGMA_DEFS = [
    ("ic1_sig_x", "ic1", "x"),
    ("ic1_sig_y", "ic1", "y"),
    ("ic2_sig_x", "ic2", "x"),
    ("ic2_sig_y", "ic2", "y"),
]


def _resolve_sigma_col(columns, ic: str, axis: str) -> str | None:
    """Find a sigma column for a given IC and axis, trying known key variants."""
    for key in _SIG_KEY_VARIANTS:
        for prefix in (f"r_{ic}_{axis}_{key}", f"{ic}_{axis}_{key}"):
            if prefix in columns:
                return prefix
    return None


def _process_session_data(session_id: str, base_dir: str):
    """Process data for a single session, dynamically resolving sigma columns."""
    input_map, spot_data = load_session_raw(session_id, base_dir=base_dir)
    if input_map is None or spot_data is None:
        return None

    energy_col = resolve_concept_column(input_map.columns, C_ENERGY)
    if energy_col is None:
        _log.debug("Session %s: no energy column found", session_id)
        return None

    found: dict[str, str] = {}
    for label, ic, axis in _IC_SIGMA_DEFS:
        col = _resolve_sigma_col(spot_data.columns, ic, axis)
        if col is not None:
            found[label] = col

    if not found:
        _log.debug("Session %s: no sigma columns found", session_id)
        return None

    keep_cols = list(found.values())
    spot_data = spot_data[keep_cols].copy().join(input_map[energy_col])
    spot_data = spot_data.apply(pd.to_numeric, errors="coerce")

    valid_mask = create_valid_mask(spot_data)
    spot_data_clean = spot_data[valid_mask]

    result: dict = {
        "session_id": session_id,
        "energy": spot_data_clean[energy_col],
        "sigma_types": [],
    }
    for label, raw_col in found.items():
        result[label] = spot_data_clean[raw_col] * 2
        result["sigma_types"].append(label)

    return result


_SUBPLOT_LAYOUT = [
    ("ic1_sig_x", "IC1 Sigma X"),
    ("ic1_sig_y", "IC1 Sigma Y"),
    ("ic2_sig_x", "IC2 Sigma X"),
    ("ic2_sig_y", "IC2 Sigma Y"),
]


def run(session_ids: list[str], base_dir: str = "test_data", *, settings=None) -> None:
    """Run sigma box plots analysis and show matplotlib window."""
    if not session_ids:
        _log.debug("No sessions selected")
        return

    session_data = {}
    for sid in session_ids:
        data = _process_session_data(sid, base_dir)
        if data is not None:
            session_data[sid] = data

    if not session_data:
        _log.debug("No valid data found for any session")
        return

    ordered_sids = list(session_data.keys())
    n_sessions = len(ordered_sids)
    colors = DEFAULT_SESSION_COLORS[:n_sessions]

    all_energies: set[float] = set()
    for data in session_data.values():
        all_energies.update(data["energy"].dropna().unique())
    unique_energies = sorted(all_energies)
    x_positions = np.arange(len(unique_energies))

    fig, axes = plt.subplots(2, 2, figsize=FIG_SIZE_2x2, sharex=True)

    width = 0.8 / max(n_sessions, 1)

    for ax, (sig_key, title) in zip(axes.flat, _SUBPLOT_LAYOUT):
        has_data = False
        for s_idx, sid in enumerate(ordered_sids):
            data = session_data[sid]
            if sig_key not in data["sigma_types"]:
                continue

            energy_vals = data["energy"].values
            sigma_vals = data[sig_key].values

            box_data = []
            positions = []
            for j, energy in enumerate(unique_energies):
                mask = energy_vals == energy
                vals = sigma_vals[mask]
                vals = vals[np.isfinite(vals)]
                if len(vals) > 0:
                    box_data.append(vals)
                    positions.append(
                        x_positions[j] + (s_idx - (n_sessions - 1) / 2) * width
                    )

            if box_data:
                has_data = True
                bp = ax.boxplot(
                    box_data,
                    positions=positions,
                    widths=width * 0.8,
                    patch_artist=True,
                    showfliers=False,
                    boxprops=dict(facecolor=colors[s_idx], alpha=0.7),
                    medianprops=dict(color="black", linewidth=1.5),
                )

        ax.set_title(title)
        ax.set_ylabel("Sigma (mm)")
        ax.grid(**GRID_KW)
        if not has_data:
            ax.text(
                0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="gray",
            )

    for ax in axes[1]:
        ax.set_xticks(x_positions)
        ax.set_xticklabels([f"{e:g}" for e in unique_energies], rotation=90)
        ax.set_xlabel("Energy (MeV)")

    set_view_header(fig, "Sigma X / Y by Energy", ordered_sids, colors, base_dir=base_dir)

    apply_tight_layout()
    plt.show()
