"""Launch Binned Summary on a fixed preset (Qt interactive or Agg headless)."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt

from ..common import DEFAULT_SESSION_COLORS
from ..common.plotting import finish_view
from ..common.settings import ViewSettings
from .binned_summary_catalog import (
    PRESET_BY_ID,
    PRESET_DOSE_ERROR_ENERGY,
    PRESET_DOSE_ERROR_ENERGY_MEAN,
    PRESET_DOSE_ERROR_MU,
    PRESET_DOSE_RATIO_ENERGY,
    PRESET_DOSE_RATIO_RADIUS,
    PRESET_DOSE_RATIO_SPOT_TIME,
    PRESET_POSITION_ERROR_ENERGY,
    PRESET_SIGMA_ENERGY,
    PRESET_SPOT_TIME_ENERGY,
    BinnedSummaryConfig,
)
from .binned_summary_data import load_sessions_summary
from .binned_summary_ui import render_binned_summary
from .binned_summary_window import run_binned_summary_window

# Legacy launcher module names kept for PDF reports and saved settings.
LEGACY_PRESET_VIEW_MODULES: dict[str, str] = {
    "dose_error_energy": PRESET_DOSE_ERROR_ENERGY,
    "dose_error_energy_mean": PRESET_DOSE_ERROR_ENERGY_MEAN,
    "dose_error_mu": PRESET_DOSE_ERROR_MU,
    "dose_ratios_energy": PRESET_DOSE_RATIO_ENERGY,
    "dose_ratios_position": PRESET_DOSE_RATIO_RADIUS,
    "dose_ratios_spot_time": PRESET_DOSE_RATIO_SPOT_TIME,
    "position_error_energy": PRESET_POSITION_ERROR_ENERGY,
    "sigma_energy": PRESET_SIGMA_ENERGY,
    "spot_delivery_time": PRESET_SPOT_TIME_ENERGY,
}


def _config_for_preset(preset_id: str) -> BinnedSummaryConfig:
    preset = PRESET_BY_ID[preset_id]
    return BinnedSummaryConfig(
        y_group=preset.y_group,
        x_param=preset.x_param,
        glyph=preset.glyph,
        show_trend=preset.show_trend,
        show_hist=preset.show_hist,
        show_corr=preset.show_corr,
    )


def run_preset_matplotlib(
    session_ids: Sequence[str],
    base_dir: str,
    preset_id: str,
    *,
    settings: ViewSettings | None = None,
) -> None:
    """Render one preset as a static matplotlib figure (reports and Agg tests)."""
    session_data = load_sessions_summary(
        list(session_ids),
        base_dir,
        settings=settings,
    )
    if not session_data:
        return

    config = _config_for_preset(preset_id)
    fig = plt.figure(figsize=(16, 9))
    render_binned_summary(fig, config, session_data, base_dir)
    loaded_ids = list(session_data.keys())
    finish_view(
        fig,
        config.title,
        loaded_ids,
        DEFAULT_SESSION_COLORS[: len(loaded_ids)],
        base_dir=base_dir,
    )


def run_preset_view(
    session_ids: Sequence[str],
    base_dir: str,
    preset_id: str,
    *,
    settings: ViewSettings | None = None,
) -> None:
    """Open the Qt Binned Summary shell or render headlessly on Agg."""
    if preset_id not in PRESET_BY_ID:
        raise ValueError(f"Unknown binned summary preset: {preset_id!r}")

    import matplotlib

    if matplotlib.get_backend().lower() == "agg":
        run_preset_matplotlib(
            session_ids,
            base_dir,
            preset_id,
            settings=settings,
        )
        return

    run_binned_summary_window(
        session_ids,
        base_dir,
        settings=settings,
        initial_preset=preset_id,
    )
