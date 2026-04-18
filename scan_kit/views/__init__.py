"""Analysis view modules for scan-kit."""

from typing import Callable

from . import (
    ic1_position_bars,
    ic1_ic2_error_scatter,
    ic1_ic2_spot_scatter,
    dose_ratios,
    dose_ratios_position,
    dose_ratios_time,
    current_ratios,
    dose_error_vs_target,
    spot_delivery_time,
    sigma_boxplots,
    beam_off_rampdown,
    beam_on_off_current,
    ic_timeslice_replay,
    dose_accumulation,
    ic_fft_analysis,
)

try:
    from . import ic_audio_export
    _HAS_AUDIO = True
except (ImportError, OSError):
    _HAS_AUDIO = False

VIEWS: list[tuple[str, str, Callable[[list[str], str], None]]] = [
    ("IC1 X/Y Position Error", "ic1_position_bars", ic1_position_bars.run),
    ("IC1 vs IC2 Error Scatter", "ic1_ic2_error_scatter", ic1_ic2_error_scatter.run),
    ("IC1/IC2 Spot Scatter", "ic1_ic2_spot_scatter", ic1_ic2_spot_scatter.run),
    ("Dose Ratios vs Energy", "dose_ratios", dose_ratios.run),
    ("Dose Ratios vs Position", "dose_ratios_position", dose_ratios_position.run),
    ("Dose Ratios vs Spot Time", "dose_ratios_time", dose_ratios_time.run),
    ("Current Ratios vs Energy", "current_ratios", current_ratios.run),
    ("Dose Error vs Target (%)", "dose_error_vs_target", dose_error_vs_target.run),
    ("Spot Delivery Time", "spot_delivery_time", spot_delivery_time.run),
    ("Sigma X/Y Box Plots", "sigma_boxplots", sigma_boxplots.run),
    ("Beam-Off Ramp-Down", "beam_off_rampdown", beam_off_rampdown.run),
    ("Beam-On vs Beam-Off Current", "beam_on_off_current", beam_on_off_current.run),
    ("IC Timeslice Replay", "ic_timeslice_replay", ic_timeslice_replay.run),
    ("Dose Accumulation", "dose_accumulation", dose_accumulation.run),
    ("IC Current FFT Analysis", "ic_fft_analysis", ic_fft_analysis.run),
]

if _HAS_AUDIO:
    VIEWS.append(
        ("IC Audio Export (WAV)", "ic_audio_export", ic_audio_export.run),
    )
