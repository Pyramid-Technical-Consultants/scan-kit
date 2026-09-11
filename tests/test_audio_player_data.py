"""Tests for IC audio player data preparation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import wave

from scan_kit.common.data_filter import FILTER_BEAM_ON
from scan_kit.views.audio_player_data import (
    AudioPlayerConfig,
    beam_subtract,
    build_playback_channels,
    extract_audio_signal,
    normalize,
    prepare_waveform_render,
    write_wav,
)
from scan_kit.views.fft_catalog import METRIC_IC_CURRENT


def test_normalize_scales_to_unit_peak() -> None:
    sig = np.array([0.0, 2.0, -4.0], dtype=np.float64)
    out = normalize(sig)
    assert out.dtype == np.float32
    assert float(np.max(np.abs(out))) == 1.0


def test_unipolar_pulse_line_tracks_peaks_not_floor() -> None:
    from scan_kit.views.fft_catalog import METRIC_DDOSE

    raw = np.tile(np.array([0.0, 0.0, 0.8, 0.0], dtype=float), 2000)
    session = {
        "session_id": "ddose",
        "ic1_ddose": raw,
        "beam_on": np.ones(len(raw), dtype=bool),
    }
    channels = prepare_waveform_render(
        "ddose",
        session,
        AudioPlayerConfig(metric_id=METRIC_DDOSE, channels=("ic1_ddose",)),
        cache={},
    )
    assert len(channels) == 1
    line_y = channels[0].line_pos[:, 1]
    n = len(channels[0].envelope_poly) // 2
    ymin = channels[0].envelope_poly[n:, 1][::-1]
    assert float(np.median(ymin)) == pytest.approx(0.0)
    assert float(line_y.max()) == pytest.approx(1.0)
    assert float(np.median(line_y)) > 0.5


def test_beam_subtract_preserves_length() -> None:
    target = np.sin(np.linspace(0, 20 * np.pi, 500))
    reference = np.sin(np.linspace(0, 5 * np.pi, 500))
    out = beam_subtract(target, reference)
    assert len(out) == 500


def test_beam_subtract_matches_scipy_stft() -> None:
    from scipy.signal import istft, stft

    from scan_kit.views.audio_player_data import (
        _SPECSUB_PASSES,
        _istft_spectrum,
        _stft_spectrum,
    )
    from scan_kit.views.fft_data import FS_HZ

    rng = np.random.default_rng(0)
    x = rng.normal(size=2048)
    for nperseg, noverlap in _SPECSUB_PASSES[:2]:
        _, _, zs = stft(x, fs=FS_HZ, nperseg=nperseg, noverlap=noverlap)
        zn = _stft_spectrum(x, nperseg, noverlap)
        assert np.allclose(zs, zn, atol=1e-12)
        _, ys = istft(zs, fs=FS_HZ, nperseg=nperseg, noverlap=noverlap)
        yn = _istft_spectrum(zn, nperseg, noverlap, len(x))
        assert np.allclose(ys[: len(x)], yn, atol=1e-12)

    target = np.sin(np.linspace(0, 40 * np.pi, 2048))
    reference = np.sin(np.linspace(0, 10 * np.pi, 2048))
    expected = _scipy_beam_subtract(target, reference)
    assert np.allclose(beam_subtract(target, reference), expected, atol=1e-10)


def _scipy_beam_subtract(target: np.ndarray, reference: np.ndarray) -> np.ndarray:
    from scipy.signal import istft, stft

    from scan_kit.views.audio_player_data import (
        _SPECSUB_ALPHA,
        _SPECSUB_BETA,
        _SPECSUB_PASSES,
        _effective_stft_params,
    )
    from scan_kit.views.fft_data import FS_HZ

    t = np.nan_to_num(target).astype(np.float64)
    r = np.nan_to_num(reference).astype(np.float64)
    n = min(len(t), len(r))
    t, r = t[:n], r[:n]
    t_mean = np.mean(t)
    cur = t - t_mean
    r_ac = r - np.mean(r)
    ref_spectra: dict[tuple[int, int], np.ndarray] = {}
    for nperseg, noverlap in _SPECSUB_PASSES:
        nperseg, noverlap = _effective_stft_params(n, nperseg, noverlap)
        zr = ref_spectra.get((nperseg, noverlap))
        if zr is None:
            _, _, zr = stft(r_ac, fs=FS_HZ, nperseg=nperseg, noverlap=noverlap)
            ref_spectra[(nperseg, noverlap)] = zr
        _, _, zt = stft(cur, fs=FS_HZ, nperseg=nperseg, noverlap=noverlap)
        mag_t = np.abs(zt)
        mag_r = np.abs(zr)
        e_t = np.sum(mag_t ** 2, axis=0, keepdims=True)
        e_r = np.sum(mag_r ** 2, axis=0, keepdims=True)
        alpha_frame = np.sqrt(e_t / (e_r + 1e-20))
        beam_estimate = _SPECSUB_ALPHA * alpha_frame * mag_r
        clean_mag = np.maximum(mag_t - beam_estimate, _SPECSUB_BETA * mag_t)
        z_clean = clean_mag * np.exp(1j * np.angle(zt))
        _, out = istft(z_clean, fs=FS_HZ, nperseg=nperseg, noverlap=noverlap)
        cur = out[:n]
    return cur + t_mean


def test_extract_audio_signal_beam_on_filter() -> None:
    session = {
        "ic1": np.array([1.0, 2.0, 3.0, 4.0]),
        "ic2": np.array([1.0, 2.0, 3.0, 4.0]),
        "beam_on": np.array([True, True, False, False]),
    }
    out = extract_audio_signal(
        session,
        "ic1",
        domain_filter="all",
        beam_state_filter=FILTER_BEAM_ON,
        filter_column_keys=["ic1", "ic2"],
    )
    assert out.tolist() == [1.0, 2.0]


def test_beam_subtract_uses_filtered_ic3() -> None:
    n = 80
    t = np.linspace(0, 8 * np.pi, n)
    ic1 = np.sin(t)
    session = {
        "ic1": ic1,
        "ic2": ic1.copy(),
        "ic3": ic1.copy(),
        "beam_on": np.array([False] * 30 + [True] * 50),
        "has_ic3": True,
    }
    config = AudioPlayerConfig(
        metric_id=METRIC_IC_CURRENT,
        channels=("ic1",),
        beam_subtract=True,
        beam_state_filter=FILTER_BEAM_ON,
    )
    channels = build_playback_channels(session, config)
    assert len(channels) == 1
    raw = extract_audio_signal(
        session,
        "ic1",
        domain_filter="all",
        beam_state_filter=FILTER_BEAM_ON,
        filter_column_keys=["ic1", "ic2"],
    )
    ref = extract_audio_signal(
        session,
        "ic3",
        domain_filter="all",
        beam_state_filter=FILTER_BEAM_ON,
        filter_column_keys=["ic1", "ic2"],
    )
    assert len(raw) == 50
    assert not np.allclose(raw, session["ic3"][: len(raw)])
    assert np.allclose(raw, ref)
    expected = normalize(beam_subtract(raw, ref))
    assert np.allclose(channels[0].signal, expected, atol=1e-5)


def test_build_playback_channels_beam_subtract() -> None:
    n = 512
    session = {
        "ic1": np.sin(np.linspace(0, 12 * np.pi, n)),
        "ic2": np.cos(np.linspace(0, 8 * np.pi, n)),
        "ic3": np.sin(np.linspace(0, 4 * np.pi, n)),
        "beam_on": np.ones(n, dtype=bool),
    }
    config = AudioPlayerConfig(
        metric_id=METRIC_IC_CURRENT,
        channels=("ic1", "ic2"),
        beam_subtract=True,
    )
    channels = build_playback_channels(session, config)
    labels = [ch.label for ch in channels]
    assert any("\u2212beam" in label for label in labels)
    assert all(ch.signal.dtype == np.float32 for ch in channels)
    assert all(np.max(np.abs(ch.signal)) <= 1.0 for ch in channels)


def test_prepare_waveform_render_cache_survives_channel_toggle() -> None:
    n = 256
    session = {
        "session_id": "test",
        "ic1": np.sin(np.linspace(0, 8 * np.pi, n)),
        "ic2": np.cos(np.linspace(0, 8 * np.pi, n)),
        "beam_on": np.ones(n, dtype=bool),
    }
    cache: dict = {}
    three = AudioPlayerConfig(
        metric_id=METRIC_IC_CURRENT,
        channels=("ic1", "ic2"),
    )
    two = AudioPlayerConfig(
        metric_id=METRIC_IC_CURRENT,
        channels=("ic1",),
    )
    first = prepare_waveform_render("test", session, three, cache)
    second = prepare_waveform_render("test", session, two, cache)
    assert first[0] is second[0]


def test_prepare_waveform_render_uses_cache() -> None:
    n = 256
    session = {
        "session_id": "test",
        "ic1": np.sin(np.linspace(0, 8 * np.pi, n)),
        "beam_on": np.ones(n, dtype=bool),
    }
    config = AudioPlayerConfig(
        metric_id=METRIC_IC_CURRENT,
        channels=("ic1",),
    )
    cache: dict = {}
    first = prepare_waveform_render("test", session, config, cache)
    second = prepare_waveform_render("test", session, config, cache)
    assert len(first) == 1
    assert first[0] is second[0]
    assert len(cache) == 1


def test_write_wav_round_trip() -> None:
    sig = np.array([0.0, 0.5, -0.5, 1.0], dtype=np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.wav"
        write_wav(path, sig)
        with wave.open(str(path), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 1000
            assert wf.getnframes() == len(sig)
