"""Dose ratio box plots (IC2/IC1, IC3/IC1, IC3/IC2) for G2 and G3 sessions."""

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
    try_load_position_data,
    plot_boxplots_for_column,
    add_energy_trend,
    add_correlation_scatter,
    make_session_legend,
    style_energy_axes,
    DEFAULT_SESSION_COLORS,
    FIG_SIZE_2x2,
    SUPTITLE_KW,
    apply_tight_layout,
)

import logging

_log = logging.getLogger(__name__)

EXTRA_SPOT_G3 = [C_IC1_TOTAL_DOSE, C_IC2_TOTAL_DOSE, C_IC3_TOTAL_DOSE]
EXTRA_SPOT_G2 = [C_IC1_TOTAL_DOSE, C_IC2_TOTAL_DOSE]


def _process_ratios_session(session_id: str, position_key: str, base_dir: str,
                            settings: ViewSettings | None = None):
    """Process session and compute dose ratios."""
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
    return add_dose_ratio_columns(data, include_ic3=position_key == POSITION_KEY_G3_RAW)


CORR_PAIRS = [
    (C_IC1_TOTAL_DOSE, C_IC2_TOTAL_DOSE, "IC1 Dose", "IC2 Dose"),
    (C_IC1_TOTAL_DOSE, C_IC3_TOTAL_DOSE, "IC1 Dose", "IC3 Dose"),
    (C_IC2_TOTAL_DOSE, C_IC3_TOTAL_DOSE, "IC2 Dose", "IC3 Dose"),
]


def run(session_ids: list[str], base_dir: str = "test_data",
        *, settings: ViewSettings | None = None) -> None:
    """Run dose ratios analysis and show matplotlib window."""
    if not session_ids:
        _log.debug("No sessions selected")
        return

    def _loader(sid, pkey, bdir):
        return _process_ratios_session(sid, pkey, bdir, settings=settings)

    session_data = {}
    for sid in session_ids:
        data = try_load_position_data(sid, base_dir, _loader)
        if data is not None:
            session_data[sid] = data

    if not session_data:
        _log.debug("No valid data found for any session")
        return

    fig, axes = plt.subplots(
        3, 2, figsize=(FIG_SIZE_2x2[0] + 4, FIG_SIZE_2x2[1] * 1.4),
        gridspec_kw={"width_ratios": [4, 1]},
    )
    fig.suptitle("Dose Ratios vs Energy", **SUPTITLE_KW)

    box_axes = [axes[0, 0], axes[1, 0], axes[2, 0]]
    corr_axes = [axes[0, 1], axes[1, 1], axes[2, 1]]

    box_axes[1].sharex(box_axes[0])
    box_axes[2].sharex(box_axes[0])

    all_energies = set()
    for data in session_data.values():
        all_energies.update(data["energy"].unique())
    energies = sorted(all_energies)

    loaded_ids = list(session_data.keys())
    colors = DEFAULT_SESSION_COLORS[: len(loaded_ids)]

    session_data_g3 = {k: v for k, v in session_data.items() if "ic31_ratio" in v}
    colors_g3 = [colors[loaded_ids.index(sid)] for sid in session_data_g3]

    plot_boxplots_for_column(
        box_axes[0], session_data, "ic21_ratio", energies, colors, width=0.3
    )
    add_energy_trend(
        box_axes[0], session_data, "ic21_ratio", energies, colors, position_offset=0.35
    )
    style_energy_axes(box_axes[0], energies, ylabel="IC2/IC1 Ratio Difference (%)")

    if session_data_g3:
        plot_boxplots_for_column(
            box_axes[1], session_data_g3, "ic31_ratio", energies, colors_g3, width=0.3
        )
        add_energy_trend(
            box_axes[1], session_data_g3, "ic31_ratio", energies, colors_g3, position_offset=0.35
        )
    style_energy_axes(box_axes[1], energies, ylabel="IC3/IC1 Ratio Difference (%)")

    if session_data_g3:
        plot_boxplots_for_column(
            box_axes[2], session_data_g3, "ic32_ratio", energies, colors_g3, width=0.3
        )
        add_energy_trend(
            box_axes[2], session_data_g3, "ic32_ratio", energies, colors_g3, position_offset=0.35
        )
    style_energy_axes(box_axes[2], energies, ylabel="IC3/IC2 Ratio Difference (%)")

    y_lo = min(ax.get_ylim()[0] for ax in box_axes)
    y_hi = max(ax.get_ylim()[1] for ax in box_axes)
    for ax in box_axes:
        ax.set_ylim(y_lo, y_hi)

    # Correlation scatter plots (right column)
    corr_data = [
        (session_data, loaded_ids, colors),
        (session_data_g3, list(session_data_g3.keys()), colors_g3),
        (session_data_g3, list(session_data_g3.keys()), colors_g3),
    ]
    for row, (col_x, col_y, xlabel, ylabel) in enumerate(CORR_PAIRS):
        sdata, sids, scols = corr_data[row]
        if sdata:
            add_correlation_scatter(corr_axes[row], sdata, col_x, col_y, sids, scols,
                                    xlabel=xlabel, ylabel=ylabel,
                                    percentile_clip=99.9, equal_aspect=True)
        else:
            corr_axes[row].set_visible(False)

    make_session_legend(box_axes[0], loaded_ids, colors)

    apply_tight_layout()
    plt.show()
