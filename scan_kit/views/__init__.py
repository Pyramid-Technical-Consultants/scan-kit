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

_UNIFIED_VIEWS: list[ViewEntry] = [
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
        "IC Beam Trajectory (3D)",
        "trajectory",
        "Interactive 3D per-spot IC beam paths, dipole gaps, iso/IC planes, "
        "and optional plan overlay (visPy).",
    ),
    (
        "Session Log Compare",
        "session_log_compare",
        "Session log layer timings, errors, and side-by-side event comparison.",
    ),
]

_SPECIALIZED_VIEWS: list[ViewEntry] = [
    (
        "Beam Error Motion vs Energy",
        "beam_motion_energy",
        "Per-energy X/Y position error spill paths from IC1 (solid) and IC2 (dotted).",
    ),
    (
        "Dose Accumulation",
        "dose_accumulation",
        "Expected versus measured cumulative dose for each ion chamber.",
    ),
    (
        "Beam-Off Ramp-Down",
        "beam_off_rampdown",
        "Beam-off ramp-down curves for IC1, IC2, and IC3 from scan-total dose.",
    ),
    (
        "IC HV Transient Test",
        "ic_hv_transient",
        "IC high-voltage toggle transients with capacitance re-derived from the nA waveforms "
        "(HCC + strips), compared to the firmware result.",
    ),
    (
        "Amplifier Command Correlations",
        "amplifier_correlation",
        "Beam-on scatter plots of settled amplifier command vs readback, field, and IC iso position.",
    ),
    (
        "IC Peak Amplitude — Beam-Off (G3)",
        "ic_peak_amplitude_beam_off",
        "G3 beam-off peak current amplitude distributions for IC1/IC2 X and Y.",
    ),
]
if _HAS_AUDIO:
    _SPECIALIZED_VIEWS.append(
        (
            "IC Audio Export (WAV)",
            "ic_audio_export",
            "Listen to IC current waveforms and export them as WAV audio files.",
        ),
    )

VIEW_GROUPS: list[tuple[str, list[ViewEntry]]] = [
    ("Unified Views", _UNIFIED_VIEWS),
    ("Specialized Analysis", _SPECIALIZED_VIEWS),
]

VIEWS: list[ViewEntry] = [entry for _title, entries in VIEW_GROUPS for entry in entries]
