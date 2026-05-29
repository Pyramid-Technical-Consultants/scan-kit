"""Dose ratio vs beam position scatter plots (IC2/IC1, IC3/IC1, IC3/IC2)."""

import numpy as np
import matplotlib.pyplot as plt

from ..common import (
    C_IC1_TOTAL_DOSE,
    C_IC2_TOTAL_DOSE,
    C_IC3_TOTAL_DOSE,
    C_CHARGE_REQ,
    POSITION_KEY_G3_RAW,
    ViewSettings,
    apply_auto_calibration,
    apply_calibration_factors,
    add_dose_ratio_columns,
    process_position_data,
    add_scatter_trend,
    try_load_position_data,
    annotate_slopes,
    make_session_legend,
    DEFAULT_SESSION_COLORS,
    FIG_SIZE_2x2,
    SUPTITLE_KW,
    apply_tight_layout,
    GRID_KW,
)

import logging

_log = logging.getLogger(__name__)

EXTRA_SPOT_G3 = [C_IC1_TOTAL_DOSE, C_IC2_TOTAL_DOSE, C_IC3_TOTAL_DOSE]
EXTRA_SPOT_G2 = [C_IC1_TOTAL_DOSE, C_IC2_TOTAL_DOSE]


def _process_session(session_id: str, position_key: str, base_dir: str,
                     settings: ViewSettings | None = None):
    """Load session, compute dose ratios and radial distance."""
    extra = EXTRA_SPOT_G3 if position_key == POSITION_KEY_G3_RAW else EXTRA_SPOT_G2
    extra_input = [C_CHARGE_REQ] if settings and settings.auto_calibrate else None
    data = process_position_data(
        session_id, position_key, extra_spot_columns=extra,
        extra_input_columns=extra_input, base_dir=base_dir,
    )
    if data is None:
        return None

    if settings and settings.auto_calibrate:
        if settings.cal_factors:
            data = apply_calibration_factors(data, extra, settings.cal_factors)
        else:
            data = apply_auto_calibration(data, C_CHARGE_REQ, extra)
    data = add_dose_ratio_columns(data, include_ic3=position_key == POSITION_KEY_G3_RAW)
    if data is None:
        return None
    data["ic1_dist"] = np.sqrt(data["ic1_x"] ** 2 + data["ic1_y"] ** 2)

    return data


def _plot_ratio_vs_distance(ax, session_data, ratio_col, colors, title):
    """Scatter ratio vs IC1 radial distance for each session."""
    slope_labels = []
    for i, (sid, data) in enumerate(session_data.items()):
        if ratio_col not in data:
            continue
        prefix = f"{sid}: " if len(session_data) > 1 else ""
        res = add_scatter_trend(
            ax,
            data["ic1_dist"],
            data[ratio_col],
            color=colors[i],
            unit="%/mm",
            prefix=prefix,
            label=f"Session {sid}",
        )
        if res is not None:
            slope_labels.append(res)

    ax.set_title(title)
    ax.set_xlabel("IC1 Radial Distance (mm)")
    ax.set_ylabel("Ratio Difference (%)")
    ax.grid(**GRID_KW)

    if slope_labels:
        annotate_slopes(ax, slope_labels)


def run(session_ids: list[str], base_dir: str = "test_data",
        *, settings: ViewSettings | None = None) -> None:
    """Run dose-ratio vs position analysis and show matplotlib window."""
    if not session_ids:
        _log.debug("No sessions selected")
        return

    def _loader(sid, pkey, bdir):
        return _process_session(sid, pkey, bdir, settings=settings)

    session_data: dict[str, dict] = {}
    for sid in session_ids:
        data = try_load_position_data(sid, base_dir, _loader)
        if data is not None:
            session_data[sid] = data

    if not session_data:
        _log.debug("No valid data found for any session")
        return

    loaded_ids = list(session_data.keys())
    colors = DEFAULT_SESSION_COLORS[: len(loaded_ids)]

    session_data_g3 = {k: v for k, v in session_data.items() if "ic31_ratio" in v}
    colors_g3 = [colors[loaded_ids.index(sid)] for sid in session_data_g3]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2, 2, figsize=FIG_SIZE_2x2, sharex=False, sharey=False
    )
    fig.suptitle("Dose Ratios vs Beam Position", **SUPTITLE_KW)

    if session_data_g3:
        _plot_ratio_vs_distance(
            ax1, session_data_g3, "ic31_ratio", colors_g3,
            "IC3/IC1 vs Radial Distance",
        )

    _plot_ratio_vs_distance(
        ax2, session_data, "ic21_ratio", colors,
        "IC2/IC1 vs Radial Distance",
    )

    if session_data_g3:
        _plot_ratio_vs_distance(
            ax3, session_data_g3, "ic32_ratio", colors_g3,
            "IC3/IC2 vs Radial Distance",
        )

    ax4.axis("off")
    make_session_legend(ax1, loaded_ids, colors)

    apply_tight_layout()
    plt.show()
