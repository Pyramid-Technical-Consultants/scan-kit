"""Analysis view modules for scan-kit."""

from typing import Callable

from . import (
    ic1_position_bars,
    ic1_ic2_error_scatter,
    ic1_ic2_spot_scatter_g3,
    ic1_spot_scatter_g2,
    dose_ratios,
    sigma_boxplots,
)

VIEWS: list[tuple[str, str, Callable[[list[str], str], None]]] = [
    ("IC1 X/Y Position Bars", "ic1_position_bars", ic1_position_bars.run),
    ("IC1 vs IC2 Error Scatter", "ic1_ic2_error_scatter", ic1_ic2_error_scatter.run),
    ("IC1/IC2 Spot Scatter (G3)", "ic1_ic2_spot_scatter_g3", ic1_ic2_spot_scatter_g3.run),
    ("IC1 Spot Scatter (G2)", "ic1_spot_scatter_g2", ic1_spot_scatter_g2.run),
    ("Dose Ratios (IC2/IC1, IC3/IC1)", "dose_ratios", dose_ratios.run),
    ("Sigma X/Y Box Plots", "sigma_boxplots", sigma_boxplots.run),
]
