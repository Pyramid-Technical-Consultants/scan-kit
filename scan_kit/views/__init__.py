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


_HAS_AUDIO = importlib.util.find_spec("scan_kit.views.ic_audio_export") is not None

# Unified reusable views first, then specialized single-purpose plots (candidates for consolidation).
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
        "Unified Views",
        [
            (
                "Binned Summary",
                "binned_summary",
                "Universal binned summary: dose error, dose ratios, position error, "
                "sigma, and spot time vs energy, target MU, spot time, or beam radius.",
            ),
            (
                "Distribution Explorer",
                "distribution",
                "Density or scatter plots for position, position error, sigma, sigma error, "
                "confidence correlations, and Gaussian filter coverage.",
            ),
            (
                "Timeslice Replay",
                "timeslice_replay",
                "Interactive timeslice viewer with selectable IC, dDose/dt, sigma, and magnetic-field channels.",
            ),
            (
                "Session Log Compare",
                "session_log_compare",
                "Session log layer timings, errors, and side-by-side event comparison.",
            ),
        ],
    ),
    (
        "Beam Distribution Quality",
        [
            (
                "Position Error Outliers (Spot)",
                "position_error_outliers_spot",
                "Spots whose X/Y deviation from target is a clear statistical outlier (median/MAD).",
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
        ],
    ),
    (
        "Timeseries & Transients Quality",
        [
            (
                "Beam-Off Ramp-Down",
                "beam_off_rampdown",
                "Beam-off ramp-down curves for IC1, IC2, and IC3 from scan-total dose.",
            ),
            (
                "IC HV Transient Test",
                "ic_hv_transient",
                "IC high-voltage toggle transients with capacitance re-derived from the nA waveforms (HCC + strips), compared to the firmware result.",
            ),
        ],
    ),
    (
        "Magnetic Analysis",
        [
            (
                "Amplifier Command Correlations",
                "amplifier_correlation",
                "Beam-on scatter plots of settled amplifier command vs readback, field, and IC iso position.",
            ),
        ],
    ),
    ("Noise measurement", _VIEW_NOISE),
]

VIEWS: list[ViewEntry] = [entry for _title, entries in VIEW_GROUPS for entry in entries]
