"""Analysis view modules for scan-kit."""

from typing import Callable

from . import (
    ic1_position_bars,
    ic1_ic2_error_scatter,
    ic1_ic2_spot_scatter_g3,
    ic1_spot_scatter_g2,
    dose_ratios,
    dose_ratios_position,
    dose_ratios_time,
    dose_error_vs_target,
    spot_delivery_time,
    sigma_boxplots,
    beam_off_rampdown,
    beam_on_off_current,
)

VIEWS: list[tuple[str, str, Callable[[list[str], str], None]]] = [
    ("IC1 X/Y Position Error", "ic1_position_bars", ic1_position_bars.run),
    ("IC1 vs IC2 Error Scatter", "ic1_ic2_error_scatter", ic1_ic2_error_scatter.run),
    ("IC1 Spot Scatter (G3)", "ic1_ic2_spot_scatter_g3", ic1_ic2_spot_scatter_g3.run),
    ("IC1 Spot Scatter (G2)", "ic1_spot_scatter_g2", ic1_spot_scatter_g2.run),
    ("Dose Ratios vs Energy", "dose_ratios", dose_ratios.run),
    ("Dose Ratios vs Position", "dose_ratios_position", dose_ratios_position.run),
    ("Dose Ratios vs Spot Time", "dose_ratios_time", dose_ratios_time.run),
    ("Dose Error vs Target (%)", "dose_error_vs_target", dose_error_vs_target.run),
    ("Spot Delivery Time", "spot_delivery_time", spot_delivery_time.run),
    ("Sigma X/Y Box Plots", "sigma_boxplots", sigma_boxplots.run),
    ("Beam-Off Ramp-Down", "beam_off_rampdown", beam_off_rampdown.run),
    ("Beam-On vs Beam-Off Current", "beam_on_off_current", beam_on_off_current.run),
]
