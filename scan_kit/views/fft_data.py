"""Timeslice signal loaders and PSD helpers for the FFT Explorer viewer."""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation
from scipy.signal import find_peaks

from ..common import (
    C_AMPLIFIER_CMD_X,
    C_AMPLIFIER_CMD_Y,
    C_AMPLIFIER_READBACK_X,
    C_AMPLIFIER_READBACK_Y,
    IC_PEAK_AMPLITUDE_COLUMNS,
    detect_beam_on_mask,
    resolve_concept_column,
)
from ..common.data_filter import (
    FILTER_BEAM_BOTH,
    FILTER_BEAM_OFF,
    filter_mask_from_columns,
)
from ..common.settings import ViewSettings
from ..common.timeslice_table import load_session_timeslice_frames
from ..data.timeline_channels import TIMELINE_CHANNEL_BY_KEY, channel_available
from .fft_catalog import (
    FFT_METRICS,
    FftConfig,
    METRIC_BY_ID,
    METRIC_IC_CURRENT,
)
from .timeslice_replay_channels import (
    load_session_timeline_catalog,
    probe_session_timeline_flags,
)

_log = logging.getLogger(__name__)

FS_HZ = 1000.0
SEGMENT_LENGTH = 4096
OVERLAP_FRACTION = 0.5
FREQ_MIN_HZ = 1.0
FREQ_MAX_HZ = 500.0
PEAK_PROMINENCE_FACTOR = 20.0
MAX_PEAKS_PER_IC = 8

_BG_GUARD_SAMPLES = 3

_PEAK_COLUMN_KEYS = {
    "ic1_x_peak": IC_PEAK_AMPLITUDE_COLUMNS[0],
    "ic1_y_peak": IC_PEAK_AMPLITUDE_COLUMNS[1],
    "ic2_x_peak": IC_PEAK_AMPLITUDE_COLUMNS[2],
    "ic2_y_peak": IC_PEAK_AMPLITUDE_COLUMNS[3],
}

_AMP_CHANNEL_KEYS = frozenset({
    "amp_cmd_x", "amp_cmd_y", "amp_rb_x", "amp_rb_y",
})
_PEAK_CHANNEL_KEYS = frozenset(_PEAK_COLUMN_KEYS.keys())
_CATALOG_CHANNEL_KEYS = frozenset(TIMELINE_CHANNEL_BY_KEY.keys())


def channel_keys_for_metric(metric_id: str) -> frozenset[str]:
    metric = METRIC_BY_ID.get(metric_id)
    if metric is None:
        return frozenset()
    return frozenset(channel.id for channel in metric.channels)


def _session_fft_header_flags(session_id: str, base_dir: str) -> dict[str, bool] | None:
    flags = probe_session_timeline_flags(session_id, base_dir)
    if flags is None:
        return None
    from ..common.session_source import (
        load_session_timeslice_device_units,
        resolve_session_source,
    )

    src = resolve_session_source(session_id, base_dir)
    if src is None:
        return flags
    frames = load_session_timeslice_device_units(src, max_frames=1)
    if not frames:
        return flags
    df0 = frames[0]
    flags = dict(flags)
    flags["has_amp_cmd_x"] = (
        resolve_concept_column(df0.columns, C_AMPLIFIER_CMD_X) is not None
    )
    flags["has_amp_cmd_y"] = (
        resolve_concept_column(df0.columns, C_AMPLIFIER_CMD_Y) is not None
    )
    flags["has_amp_rb_x"] = (
        resolve_concept_column(df0.columns, C_AMPLIFIER_READBACK_X) is not None
    )
    flags["has_amp_rb_y"] = (
        resolve_concept_column(df0.columns, C_AMPLIFIER_READBACK_Y) is not None
    )
    for key, concept in _PEAK_COLUMN_KEYS.items():
        flags[f"has_{key}"] = (
            resolve_concept_column(df0.columns, concept) is not None
        )
    return flags


def _channel_available_from_flags(channel_id: str, flags: dict[str, bool]) -> bool:
    if channel_id in ("ic1", "ic2"):
        return flags.get("has_ic", False)
    if channel_id == "ic3":
        return flags.get("has_ic3", False)
    if channel_id.endswith("_ddose"):
        if channel_id == "ic3_ddose":
            return flags.get("has_ddose3", False)
        return flags.get("has_ddose", False)
    if channel_id.startswith("sigma_"):
        return flags.get("has_sigma", False)
    if channel_id in ("bx", "by", "b_mag"):
        return flags.get("has_field", False)
    if channel_id == "beam":
        return flags.get("has_beam", False)
    if channel_id in ("ic1_x", "ic1_y", "ic2_x", "ic2_y"):
        return flags.get("has_positions", False)
    if channel_id in _AMP_CHANNEL_KEYS:
        return flags.get(f"has_{channel_id}", False)
    if channel_id in _PEAK_CHANNEL_KEYS:
        return flags.get(f"has_{channel_id}", False)
    return False


def probe_fft_metric_availability_headers(
    session_ids: Sequence[str],
    base_dir: str,
) -> dict[str, bool]:
    """Probe FFT metrics from timeslice headers without loading full signals."""
    flags_list: list[dict[str, bool]] = []
    for session_id in session_ids:
        flags = _session_fft_header_flags(session_id, base_dir)
        if flags is not None:
            flags_list.append(flags)
    availability = {metric.id: False for metric in FFT_METRICS}
    if not flags_list:
        return availability
    for metric in FFT_METRICS:
        for flags in flags_list:
            if any(
                _channel_available_from_flags(channel.id, flags)
                for channel in metric.channels
            ):
                availability[metric.id] = True
                break
    return availability


def _current_quiet_mask(sig: np.ndarray, threshold: float) -> np.ndarray:
    hot = np.abs(sig) > threshold
    if _BG_GUARD_SAMPLES > 0:
        hot = binary_dilation(hot, iterations=_BG_GUARD_SAMPLES)
    return ~hot


def _append_frame_channels(
    data: dict,
    frames: Sequence[pd.DataFrame],
    *,
    channel_keys: frozenset[str] | None = None,
) -> dict:
    """Add beam_on, amplifier, and G3 peak-amplitude arrays from pre-loaded frames."""
    if not frames:
        return data

    want_beam_on = True
    want_amp = channel_keys is None or bool(channel_keys & _AMP_CHANNEL_KEYS)
    want_peak = channel_keys is None or bool(channel_keys & _PEAK_CHANNEL_KEYS)

    df0 = frames[0]
    amp_cols: dict[str, str | None] = {}
    if want_amp:
        amp_cols = {
            "amp_cmd_x": resolve_concept_column(df0.columns, C_AMPLIFIER_CMD_X),
            "amp_cmd_y": resolve_concept_column(df0.columns, C_AMPLIFIER_CMD_Y),
            "amp_rb_x": resolve_concept_column(df0.columns, C_AMPLIFIER_READBACK_X),
            "amp_rb_y": resolve_concept_column(df0.columns, C_AMPLIFIER_READBACK_Y),
        }
    peak_cols: dict[str, str | None] = {}
    if want_peak:
        peak_cols = {
            key: resolve_concept_column(df0.columns, concept)
            for key, concept in _PEAK_COLUMN_KEYS.items()
        }

    parts: dict[str, list[np.ndarray]] = {"beam_on": []}
    for key, col in amp_cols.items():
        if col is not None and (channel_keys is None or key in channel_keys):
            parts[key] = []
    for key, col in peak_cols.items():
        if col is not None and (channel_keys is None or key in channel_keys):
            parts[key] = []

    for df in frames:
        if want_beam_on:
            beam_on = detect_beam_on_mask(df)
            if beam_on is None:
                beam_on = np.ones(len(df), dtype=bool)
            parts["beam_on"].append(beam_on.astype(bool))

        for key, col in amp_cols.items():
            if col is None or key not in parts:
                continue
            parts[key].append(df[col].to_numpy(dtype=float))
        for key, col in peak_cols.items():
            if col is None or key not in parts:
                continue
            parts[key].append(df[col].to_numpy(dtype=float))

    for key, chunks in parts.items():
        if chunks:
            data[key] = np.concatenate(chunks)
    return data


def load_session_fft_signals(
    session_id: str,
    base_dir: str,
    *,
    bg_subtract: bool = False,
    channel_keys: frozenset[str] | None = None,
) -> dict | None:
    """Load timeslice channels usable by the FFT Explorer.

    When *channel_keys* is set, only those catalog / frame channels are
  extracted (timeslice frames are still read once per session).
    """
    catalog_keys: frozenset[str] | None = None
    if channel_keys is not None:
        catalog_keys = frozenset(
            key for key in channel_keys if key in _CATALOG_CHANNEL_KEYS
        )

    opened = load_session_timeslice_frames(
        session_id, base_dir, bg_subtract=bg_subtract,
    )
    if opened is None:
        return None

    data: dict | None = None
    if channel_keys is None or catalog_keys:
        data = load_session_timeline_catalog(
            session_id,
            base_dir,
            bg_subtract=bg_subtract,
            opened=opened,
            channel_keys=catalog_keys if channel_keys is not None else None,
        )
    elif channel_keys is not None:
        flags = probe_session_timeline_flags(session_id, base_dir)
        if flags is None:
            return None
        data = {
            "has_ic3": flags.get("has_ic3", False),
            "has_beam": flags.get("has_beam", False),
            "has_positions": flags.get("has_positions", False),
            "has_sigma": flags.get("has_sigma", False),
            "has_field": flags.get("has_field", False),
            "has_ddose": flags.get("has_ddose", False),
        }

    if data is None:
        return None
    data = dict(data)
    data["session_id"] = session_id
    _, frames, *_rest = opened
    return _append_frame_channels(
        data, frames, channel_keys=channel_keys,
    )


def merge_fft_session_channels(
    existing: dict | None,
    session_id: str,
    base_dir: str,
    channel_keys: frozenset[str],
    *,
    bg_subtract: bool = False,
) -> dict | None:
    """Load *channel_keys* for one session and merge into *existing* payload."""
    if not channel_keys:
        return existing
    new = load_session_fft_signals(
        session_id,
        base_dir,
        bg_subtract=bg_subtract,
        channel_keys=channel_keys,
    )
    if new is None:
        return existing
    if existing is None:
        return new
    merged = dict(existing)
    for key, value in new.items():
        merged[key] = value
    return merged


def load_sessions_fft(
    session_ids: Sequence[str],
    base_dir: str,
    *,
    settings: ViewSettings | None = None,
    channel_keys: frozenset[str] | None = None,
) -> dict[str, dict]:
    """Return combined-session timeslice payloads for FFT rendering."""
    bg = settings.bg_subtract if settings else False
    session_data: dict[str, dict] = {}
    for sid in session_ids:
        data = load_session_fft_signals(
            sid, base_dir, bg_subtract=bg, channel_keys=channel_keys,
        )
        if data is not None:
            session_data[sid] = data
    return session_data


def _channel_available(session: dict, channel_id: str) -> bool:
    spec = TIMELINE_CHANNEL_BY_KEY.get(channel_id)
    if spec is None:
        return False
    return channel_available(session, spec)


def probe_metric_availability(
    session_data: dict[str, dict],
) -> dict[str, bool]:
    """Return which FFT metrics have at least one channel in the loaded sessions."""
    availability = {metric.id: False for metric in FFT_METRICS}
    if not session_data:
        return availability
    for metric in FFT_METRICS:
        for channel in metric.channels:
            if any(_channel_available(data, channel.id) for data in session_data.values()):
                availability[metric.id] = True
                break
    return availability


def probe_channel_availability(
    session_data: dict[str, dict],
    metric_id: str,
) -> dict[str, bool]:
    """Return channel availability for one metric across loaded sessions."""
    metric = METRIC_BY_ID.get(metric_id)
    if metric is None or not session_data:
        return {}
    return {
        channel.id: any(_channel_available(data, channel.id) for data in session_data.values())
        for channel in metric.channels
    }


def default_config(
    metric_availability: dict[str, bool],
    session_data: dict[str, dict],
) -> FftConfig:
    metric_id = next(
        (metric.id for metric in FFT_METRICS if metric_availability.get(metric.id, False)),
        METRIC_IC_CURRENT,
    )
    metric = METRIC_BY_ID[metric_id]
    channel_avail = probe_channel_availability(session_data, metric_id)
    channels = tuple(
        ch.id
        for ch in metric.channels
        if channel_avail.get(ch.id, False)
    )
    if not channels:
        channels = metric.default_channel_ids
    return FftConfig(metric_id=metric_id, channels=channels)


def extract_fft_traces(
    session: dict,
    channel_id: str,
    *,
    domain_filter: str,
    beam_state_filter: str,
    filter_column_keys: Sequence[str],
    beam_off_quiet_threshold: float | None = None,
) -> list[tuple[np.ndarray, str]]:
    """Return filtered (signal, linestyle) pairs for one channel."""
    if channel_id not in session:
        return []

    sig = np.asarray(session[channel_id], dtype=float)
    beam_on = session.get("beam_on")
    domain_mask = filter_mask_from_columns(
        session,
        filter_column_keys,
        domain_filter,
        FILTER_BEAM_BOTH,
    )
    quiet_threshold = beam_off_quiet_threshold if beam_off_quiet_threshold is not None else 10.0

    if beam_state_filter == FILTER_BEAM_BOTH:
        if beam_on is None:
            if domain_mask is None:
                return [(sig, "-")] if sig.size else []
            masked = sig[np.asarray(domain_mask, dtype=bool)]
            return [(masked, "-")] if masked.size else []
        on = np.asarray(beam_on, dtype=bool)
        if domain_mask is None:
            domain_mask_arr = np.ones(len(sig), dtype=bool)
        else:
            domain_mask_arr = np.asarray(domain_mask, dtype=bool)
        traces: list[tuple[np.ndarray, str]] = []
        on_mask = domain_mask_arr & on
        if np.any(on_mask):
            traces.append((sig[on_mask], "-"))
        off_mask = domain_mask_arr & ~on
        quiet_off = off_mask & _current_quiet_mask(sig, quiet_threshold)
        if np.any(quiet_off):
            traces.append((sig[quiet_off], "--"))
        return traces

    mask = filter_mask_from_columns(
        session,
        filter_column_keys,
        domain_filter,
        beam_state_filter,
    )
    if mask is None:
        return []
    mask = np.asarray(mask, dtype=bool)
    if beam_state_filter == FILTER_BEAM_OFF:
        mask = mask & _current_quiet_mask(sig, quiet_threshold)
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
