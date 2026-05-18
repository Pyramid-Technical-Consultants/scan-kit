"""Analysis view modules for scan-kit.

Launcher metadata avoids importing view modules (heavy matplotlib/pandas stack)
until a view is actually run. Use :data:`VIEW_GROUPS` / :data:`VIEWS` for
(display name, module name) pairs.

Optional **IC Audio Export** is detected via :func:`importlib.util.find_spec`
without importing :mod:`sounddevice`.
"""

from __future__ import annotations

import importlib.util

ViewEntry = tuple[str, str]

# Workflow-oriented groups for the launcher (flat VIEWS is derived below).
VIEW_GROUPS: list[tuple[str, list[ViewEntry]]] = [
    (
        "Spot & position QA",
        [
            ("IC1 X/Y Position Error", "ic1_position_bars"),
            ("IC1 vs IC2 Error Scatter", "ic1_ic2_error_scatter"),
            ("IC1/IC2 Spot Scatter", "ic1_ic2_spot_scatter"),
            ("Sigma X/Y Box Plots", "sigma_boxplots"),
            ("Spot Delivery Time", "spot_delivery_time"),
        ],
    ),
    (
        "Dose vs prescription",
        [
            ("Dose Ratios vs Energy", "dose_ratios"),
            ("Dose Ratios vs Position", "dose_ratios_position"),
            ("Dose Ratios vs Spot Time", "dose_ratios_time"),
            ("Dose Error vs Target (%)", "dose_error_vs_target"),
            (
                "Dose Error vs Target (mean scatter)",
                "dose_error_vs_target_mean_scatter",
            ),
            ("Dose Accumulation", "dose_accumulation"),
        ],
    ),
    (
        "IC currents & beam state",
        [
            ("Current Ratios vs Energy", "current_ratios"),
            ("Beam-On vs Beam-Off Current", "beam_on_off_current"),
        ],
    ),
    (
        "Timeseries & transients",
        [
            ("Beam-Off Ramp-Down", "beam_off_rampdown"),
            ("IC Timeslice Replay", "ic_timeslice_replay"),
            (
                "IC Timeslice Replay (dDose/dt)",
                "ic_timeslice_replay_derived",
            ),
            ("IC Current FFT Analysis", "ic_fft_analysis"),
        ],
    ),
]

_HAS_AUDIO = importlib.util.find_spec("scan_kit.views.ic_audio_export") is not None

if _HAS_AUDIO:
    VIEW_GROUPS = [
        *VIEW_GROUPS,
        (
            "Export",
            [("IC Audio Export (WAV)", "ic_audio_export")],
        ),
    ]

VIEWS: list[ViewEntry] = [
    entry for _title, entries in VIEW_GROUPS for entry in entries
]
