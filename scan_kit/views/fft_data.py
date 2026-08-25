"""Timeslice signal loaders and PSD helpers for the FFT Explorer viewer."""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.signal import find_peaks

from ..common.data_filter import (
    FILTER_BEAM_BOTH,
    FILTER_BEAM_OFF,
    filter_mask_from_columns,
)
from ..common.settings import ViewSettings
from .binned_summary_data import load_sessions_ic_current
from .fft_catalog import FFT_SIGNALS, FftConfig, SIGNAL_IC3, SIGNAL_BY_ID

_log = logging.getLogger(__name__)

FS_HZ = 1000.0
SEGMENT_LENGTH = 4096
OVERLAP_FRACTION = 0.5
FREQ_MIN_HZ = 1.0
FREQ_MAX_HZ = 500.0
PEAK_PROMINENCE_FACTOR = 20.0
MAX_PEAKS_PER_IC = 8

_BG_NOISE_FLOOR_NA = 10.0
_BG_GUARD_SAMPLES = 3


def _current_quiet_mask(sig: np.ndarray) -> np.ndarray:
    hot = np.abs(sig) > _BG_NOISE_FLOOR_NA
    if _BG_GUARD_SAMPLES > 0:
        hot = binary_dilation(hot, iterations=_BG_GUARD_SAMPLES)
    return ~hot


def load_sessions_fft(
    session_ids: Sequence[str],
    base_dir: str,
    *,
    settings: ViewSettings | None = None,
) -> dict[str, dict]:
    """Return IC current timeslice payloads for FFT rendering."""
    return load_sessions_ic_current(session_ids, base_dir, settings=settings)


def probe_signal_availability(
    session_ids: Sequence[str],
    base_dir: str,
    *,
    session_data: dict[str, dict] | None = None,
) -> dict[str, bool]:
    """Return which FFT signal sources have data in the selected sessions."""
    data = session_data if session_data is not None else load_sessions_fft(
        session_ids, base_dir,
    )
    availability = {signal.id: False for signal in FFT_SIGNALS}
    if not data:
        return availability
    for signal in FFT_SIGNALS:
        if signal.id == SIGNAL_IC3:
            if not any(d.get("has_ic3", False) for d in data.values()):
                continue
        if any(
            signal.column_key in d and len(np.asarray(d[signal.column_key])) > 0
            for d in data.values()
        ):
            availability[signal.id] = True
    return availability


def default_config(availability: dict[str, bool]) -> FftConfig:
    signals = tuple(s.id for s in FFT_SIGNALS if availability.get(s.id, False))
    if not signals:
        signals = (FFT_SIGNALS[0].id, FFT_SIGNALS[1].id)
    return FftConfig(signals=signals)


def extract_fft_traces(
    session: dict,
    ic_key: str,
    *,
    domain_filter: str,
    beam_state_filter: str,
    filter_column_keys: Sequence[str],
) -> list[tuple[np.ndarray, str]]:
    """Return filtered (signal, linestyle) pairs for one IC channel."""
    signal = SIGNAL_BY_ID.get(ic_key)
    if signal is None:
        return []
    col = signal.column_key
    if col not in session:
        return []

    sig = np.asarray(session[col], dtype=float)
    beam_on = session.get("beam_on")
    domain_mask = filter_mask_from_columns(
        session,
        filter_column_keys,
        domain_filter,
        FILTER_BEAM_BOTH,
    )

    if beam_state_filter == FILTER_BEAM_BOTH:
        if beam_on is None:
            masked = sig[domain_mask]
            return [(masked, "-")] if masked.size else []
        on = np.asarray(beam_on, dtype=bool)
        traces: list[tuple[np.ndarray, str]] = []
        on_mask = domain_mask & on
        if np.any(on_mask):
            traces.append((sig[on_mask], "-"))
        off_mask = domain_mask & ~on
        quiet_off = off_mask & _current_quiet_mask(sig)
        if np.any(quiet_off):
            traces.append((sig[quiet_off], "--"))
        return traces

    mask = filter_mask_from_columns(
        session,
        filter_column_keys,
        domain_filter,
        beam_state_filter,
    )
    if beam_state_filter == FILTER_BEAM_OFF:
        mask = mask & _current_quiet_mask(sig)
    if not np.any(mask):
        return []
    return [(sig[mask], "-")]


def welch_psd(signal: np.ndarray, fs: float, seg_len: int, overlap: float):
    """Estimate power spectral density via Welch's method."""
    sig = np.nan_to_num(signal - np.nanmean(signal))
    n = len(sig)
    if n < seg_len:
        seg_len = max(16, n)

    step = max(1, int(seg_len * (1 - overlap)))
    window = np.hanning(seg_len)
    win_power = np.sum(window ** 2)

    psd_accum: np.ndarray | None = None
    count = 0
    for s in range(0, n - seg_len + 1, step):
        segment = sig[s: s + seg_len] * window
        power = np.abs(np.fft.rfft(segment)) ** 2
        if psd_accum is None:
            psd_accum = power
        else:
            psd_accum += power
        count += 1

    if psd_accum is None or count == 0:
        return None, None

    psd = psd_accum / (count * win_power)
    psd[1:-1] *= 2
    freqs_hz = np.fft.rfftfreq(seg_len, d=1.0 / fs)
    return freqs_hz, psd


def find_peak_indices(psd: np.ndarray) -> np.ndarray:
    if len(psd) < 3:
        return np.array([], dtype=int)
    median_psd = np.median(psd)
    prominence = median_psd * PEAK_PROMINENCE_FACTOR
    indices, props = find_peaks(psd, prominence=prominence)
    if len(indices) == 0:
        return indices
    order = np.argsort(props["prominences"])[::-1][:MAX_PEAKS_PER_IC]
    indices = indices[order]
    return np.sort(indices)
