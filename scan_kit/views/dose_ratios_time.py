"""Dose ratio vs spot delivery time scatter plots (IC2/IC1, IC3/IC1, IC3/IC2)."""

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
    add_spot_delivery_time,
    process_position_data,
    scatter_with_trend,
    try_load_position_data,
    annotate_slopes,
    make_session_legend,
    DEFAULT_SESSION_COLORS,
    FIG_SIZE_2x2,
    SUPTITLE_KW,
    GRID_KW,
    SCATTER_ALPHA,
    SCATTER_SIZE,
)

import logging

_log = logging.getLogger(__name__)

EXTRA_SPOT_G3 = [C_IC1_TOTAL_DOSE, C_IC2_TOTAL_DOSE, C_IC3_TOTAL_DOSE, "timestamp", "layer_id"]
EXTRA_SPOT_G2 = [C_IC1_TOTAL_DOSE, C_IC2_TOTAL_DOSE, "timestamp", "layer_id"]


def _process_session(session_id: str, position_key: str, base_dir: str,
                     settings: ViewSettings | None = None):
    """Load session, compute dose ratios and spot delivery time."""
    extra = EXTRA_SPOT_G3 if position_key == POSITION_KEY_G3_RAW else EXTRA_SPOT_G2
    dose_cols = [C_IC1_TOTAL_DOSE, C_IC2_TOTAL_DOSE, C_IC3_TOTAL_DOSE]
    extra_input = [C_CHARGE_REQ] if settings and settings.auto_calibrate else None
    data = process_position_data(
        session_id, position_key, extra_spot_columns=extra,
        extra_input_columns=extra_input, base_dir=base_dir,
    )
    if data is None:
        return None

    if settings and settings.auto_calibrate:
        if settings.cal_factors:
            data = apply_calibration_factors(data, dose_cols, settings.cal_factors)
        else:
            data = apply_auto_calibration(data, C_CHARGE_REQ, dose_cols)
    data = add_dose_ratio_columns(data, include_ic3=position_key == POSITION_KEY_G3_RAW)
    if data is None:
        return None
    return add_spot_delivery_time(data, max_spot_time_ms=100.0)


def _plot_ratio_vs_time(ax, session_data, ratio_col, colors, title):
    """Scatter ratio vs spot delivery time for each session."""
    slope_labels = []
    for i, (sid, data) in enumerate(session_data.items()):
        if ratio_col not in data:
            continue
        slope = scatter_with_trend(
            ax,
            data["spot_time"],
            data[ratio_col],
            color=colors[i],
            label=f"Session {sid}",
            alpha=SCATTER_ALPHA,
            size=SCATTER_SIZE,
        )
        if slope is not None:
            prefix = f"{sid}: " if len(session_data) > 1 else ""
            slope_labels.append((f"{prefix}{slope:+.4g} % per ms", colors[i]))

    ax.set_title(title)
    ax.set_xlabel("Spot Delivery Time (ms)")
    ax.set_ylabel("Ratio Difference (%)")
    ax.grid(**GRID_KW)

    if slope_labels:
        annotate_slopes(ax, slope_labels)


def run(session_ids: list[str], base_dir: str = "test_data",
        *, settings: ViewSettings | None = None) -> None:
    """Run dose-ratio vs spot time analysis and show matplotlib window."""
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
    fig.suptitle("Dose Ratios vs Spot Delivery Time", **SUPTITLE_KW)

    if session_data_g3:
        _plot_ratio_vs_time(
            ax1, session_data_g3, "ic31_ratio", colors_g3,
            "IC3/IC1 vs Spot Time",
        )

    _plot_ratio_vs_time(
        ax2, session_data, "ic21_ratio", colors,
        "IC2/IC1 vs Spot Time",
    )

    if session_data_g3:
        _plot_ratio_vs_time(
            ax3, session_data_g3, "ic32_ratio", colors_g3,
            "IC3/IC2 vs Spot Time",
        )

    ax4.axis("off")
    make_session_legend(ax1, loaded_ids, colors)

    plt.tight_layout()
    plt.show()
