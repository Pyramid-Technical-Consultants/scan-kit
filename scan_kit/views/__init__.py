"""Analysis view modules for scan-kit.

Launcher metadata avoids importing view modules (heavy matplotlib/pandas stack)
until a view is actually run. Use :data:`VIEW_GROUPS` / :data:`VIEWS` for
(display name, module name, description) tuples.

Optional **IC Audio Export** is detected via :func:`importlib.util.find_spec`
without importing :mod:`sounddevice`.
"""

from __future__ import annotations

import importlib.util

ViewEntry = tuple[str, str, str]


def view_module_name(entry: ViewEntry) -> str:
    return entry[1]


def view_description(entry: ViewEntry) -> str:
    return entry[2]


def view_description_for_module(module_name: str) -> str:
    for entry in VIEWS:
        if view_module_name(entry) == module_name:
            return view_description(entry)
    return ""


_HAS_AUDIO = importlib.util.find_spec("scan_kit.views.ic_audio_export") is not None

# Workflow-oriented groups for the launcher (flat VIEWS is derived below).
_VIEW_NOISE: list[ViewEntry] = [
    (
        "IC Current FFT Analysis",
        "ic_fft_analysis",
        "Frequency-domain view of IC1, IC2, and IC3 timeslice current.",
    ),
    (
        "IC Peak Amplitude — Beam-Off (G3)",
        "ic_peak_amplitude_beam_off",
        "G3 beam-off peak current amplitude distributions for IC1/IC2 X and Y.",
    ),
]
if _HAS_AUDIO:
    _VIEW_NOISE.append(
        (
            "IC Audio Export (WAV)",
            "ic_audio_export",
            "Listen to IC current waveforms and export them as WAV audio files.",
        )
    )

VIEW_GROUPS: list[tuple[str, list[ViewEntry]]] = [
    (
        "Beam Distribution Quality",
        [
            (
                "Position Error vs Energy",
                "position_error_energy",
                "IC1 and IC2 position error in X and Y versus beam energy.",
            ),
            (
                "Position Error Distribution (Timeslice)",
                "position_error_distribution_timeslice",
                "Beam-on timeslice IC1/IC2 position error density contours and X/Y histograms.",
            ),
            (
                "Position Error Distribution (Spot)",
                "position_error_distribution_spot",
                "Per-spot IC1/IC2 position error density contours and X/Y histograms.",
            ),
            (
                "Position Error Outliers (Spot)",
                "position_error_outliers_spot",
                "Spots whose X/Y deviation from target is a clear statistical outlier (median/MAD).",
            ),
            (
                "Sigma vs Energy",
                "sigma_energy",
                "IC1 and IC2 spot size (sigma) in X and Y versus beam energy.",
            ),
            (
                "Sigma Distribution (Timeslice)",
                "sigma_distribution_timeslice",
                "Beam-on timeslice IC1/IC2 sigma density contours and X/Y histograms.",
            ),
            (
                "Confidence Correlations (Timeslice)",
                "confidence_correlation_timeslice",
                "Beam-on G3 fit confidence vs peak IC current and primary channel (density contours).",
            ),
            (
                "Gaussian Fit Filter Coverage",
                "gaussian_fit_filter_coverage",
                "Spot retention versus Gaussian fit confidence, peak current, and spot error code (IC1/IC2, combined X/Y).",
            ),
            (
                "Position Scatter",
                "position_scatter",
                "Planned, IC1, and IC2 spot positions overlaid by session.",
            ),
            (
                "IC Beam Trajectory",
                "ic_beam_trajectory",
                "Per-spot IC beam path in X and Y, extended along the beam axis.",
            ),
            (
                "Beam Error Motion vs Energy",
                "beam_motion_energy",
                "Per-energy X/Y position error spill paths from IC1 (solid) and IC2 (dotted).",
            ),
        ],
    ),
    (
        "Dosimetry Quality",
        [
            (
                "Dose Ratios vs Energy",
                "dose_ratios_energy",
                "IC2/IC1, IC3/IC1, and IC3/IC2 dose ratios versus beam energy.",
            ),
            (
                "Dose Ratios vs Position",
                "dose_ratios_position",
                "Inter-IC dose ratio consistency versus beam position.",
            ),
            (
                "Dose Ratios vs Spot Time",
                "dose_ratios_spot_time",
                "Inter-IC dose ratio behavior versus spot delivery time.",
            ),
            (
                "Dose Error vs Energy",
                "dose_error_energy",
                "Percent dose error versus prescribed target by energy (IC1/IC2/IC3).",
            ),
            (
                "Dose Error vs Energy (mean)",
                "dose_error_energy_mean",
                "Mean percent dose error per energy layer (IC1/IC2/IC3).",
            ),
            (
                "Dose Error vs Target MU",
                "dose_error_mu",
                "Per-spot percent dose error versus prescribed target MU.",
            ),
            (
                "Dose Accumulation",
                "dose_accumulation",
                "Expected versus measured cumulative dose for each ion chamber.",
            ),
            (
                "MU Delivery Rate vs Energy",
                "mu_delivery_rate_energy",
                "Effective MU delivery rate versus beam energy (wall-clock time per layer).",
            ),
        ],
    ),
    (
        "Beam Current Quality",
        [
            (
                "Current Ratios vs Energy",
                "current_ratios",
                "Beam-on mean IC current ratios versus beam energy.",
            ),
            (
                "Beam-On vs Beam-Off Current",
                "beam_on_off_current",
                "Beam-on and beam-off current distributions by energy.",
            ),
            (
                "Spot Delivery Time",
                "spot_delivery_time",
                "Total, beam-on, and overhead time per delivered spot.",
            ),
        ],
    ),
    (
        "Timeseries & Transients Quality",
        [
            (
                "Beam-Off Ramp-Down",
                "beam_off_rampdown",
                "Beam-off current ramp-down curves for IC1, IC2, and IC3.",
            ),
            (
                "IC Timeslice Replay",
                "ic_timeslice_replay",
                "Interactive viewer for IC1, IC2, and IC3 timeslice current.",
            ),
            (
                "IC Timeslice Replay (dDose/dt)",
                "ic_timeslice_replay_derived",
                "Interactive viewer for IC current derived from scan-total dose rate.",
            ),
        ],
    ),
    (
        "Magnetic Analysis",
        [
            (
                "Magnetic Field Timeslice Replay",
                "field_timeslice_replay",
                "Interactive Bx and By scan-magnet field viewer with timeline brush.",
            ),
            (
                "Amplifier Command Correlations",
                "amplifier_correlation",
                "Beam-on scatter plots of settled amplifier command vs readback, field, and IC iso position.",
            ),
        ],
    ),
    ("Noise measurement", _VIEW_NOISE),
    (
        "Session Log Analysis",
        [
            (
                "Session Log Compare",
                "session_log_compare",
                "Session log layer timings, errors, and side-by-side event comparison.",
            ),
        ],
    ),
]

VIEWS: list[ViewEntry] = [entry for _title, entries in VIEW_GROUPS for entry in entries]
