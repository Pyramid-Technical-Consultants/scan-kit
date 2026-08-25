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
                "Universal binned summary: dose error, dose ratios, dose rate, current ratios, "
                "IC current, position error, sigma, and spot time vs energy, target MU, spot time, or beam radius.",
            ),
            (
                "Distribution Explorer",
                "distribution",
                "Density or scatter plots for position, position error, sigma, sigma error, "
                "confidence correlations, and Gaussian filter coverage.",
            ),
            (
                "FFT Explorer",
                "ic_fft_analysis",
                "Frequency-domain FFT line spectra for selectable timeslice IC currents.",
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
        ],
    ),
    (
        "Beam Current Quality",
        [
            (
                "Beam-Off Ramp-Down",
                "beam_off_rampdown",
                "Beam-off ramp-down curves for IC1, IC2, and IC3 from scan-total dose.",
            ),
        ],
    ),
    (
        "Timeseries & Transients Quality",
        [
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
