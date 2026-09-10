"""Tests for IC audio player data preparation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
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


def test_beam_subtract_preserves_length() -> None:
    target = np.sin(np.linspace(0, 20 * np.pi, 500))
    reference = np.sin(np.linspace(0, 5 * np.pi, 500))
    out = beam_subtract(target, reference)
    assert len(out) == 500


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
