"""Tests for FFT Explorer loaders, renderer, and Qt shell."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from scan_kit.views.fft_catalog import (
    FftConfig,
    METRIC_IC_CURRENT,
    METRIC_MAG_FIELD,
    PRESET_ALL_ICS,
    PRESET_BY_ID,
)
from scan_kit.views.fft_data import (
    extract_fft_traces,
    load_sessions_fft,
    probe_channel_availability,
    probe_metric_availability,
    welch_psd,
)
from scan_kit.views.fft_ui import render_fft
from scan_kit.views.fft_window import FftExplorerWindow
from tests.conftest import G3_LARGE_SESSION, G3_SESSION, TEST_DATA


def test_welch_psd_returns_band_limited_frequencies() -> None:
    import numpy as np

    signal = np.sin(np.linspace(0, 40 * np.pi, 8000))
    freqs, psd = welch_psd(signal, 1000.0, 1024, 0.5)
    assert freqs is not None and psd is not None
    assert len(freqs) == len(psd)


def test_load_sessions_fft_g3(g3_fft_data) -> None:
    session_data = g3_fft_data
    if not session_data:
        pytest.skip("timeslice FFT data unavailable in fixture")
    assert G3_SESSION in session_data
    assert "beam_on" in session_data[G3_SESSION]
    metrics = probe_metric_availability(session_data)
    assert metrics[METRIC_IC_CURRENT]
    channels = probe_channel_availability(session_data, METRIC_IC_CURRENT)
    assert channels["ic1"]


def test_render_fft_headless(g3_fft_data) -> None:
    from scan_kit.views.plot_view_shell import new_headless_figure

    session_data = g3_fft_data
    if not session_data:
        pytest.skip("timeslice FFT data unavailable in fixture")
    config = PRESET_BY_ID[PRESET_ALL_ICS]
    fig = new_headless_figure((16, 9))
    render_fft(
        fig,
        FftConfig(
            metric_id=config.metric_id,
            channels=config.channels,
            domain_filter=config.domain_filter,
            beam_state_filter=config.beam_state_filter,
            annotate_peaks=config.annotate_peaks,
        ),
        session_data,
        str(TEST_DATA),
    )
    assert fig.axes
    plt.close(fig)


def test_load_sessions_fft_g3_large(g3_large_fft_data) -> None:
    session_data = g3_large_fft_data
    if not session_data:
        pytest.skip("timeslice FFT data unavailable in fixture")
    assert G3_LARGE_SESSION in session_data
    assert len(session_data[G3_LARGE_SESSION]["ic1"]) > 0
    metrics = probe_metric_availability(session_data)
    assert metrics[METRIC_IC_CURRENT]
    channels = probe_channel_availability(session_data, METRIC_IC_CURRENT)
    assert channels["ic3"]


def test_extract_fft_traces_respects_beam_filter() -> None:
    import numpy as np

    from scan_kit.common.data_filter import FILTER_BEAM_OFF, FILTER_BEAM_ON

    session = {
        "ic1": np.array([1.0, 2.0, 3.0, 4.0]),
        "ic2": np.array([1.0, 2.0, 3.0, 4.0]),
        "beam_on": np.array([True, True, False, False]),
    }
    on_traces = extract_fft_traces(
        session,
        "ic1",
        domain_filter="all",
        beam_state_filter=FILTER_BEAM_ON,
        filter_column_keys=["ic1", "ic2"],
    )
    assert len(on_traces) == 1
    assert on_traces[0][0].tolist() == [1.0, 2.0]

    off_traces = extract_fft_traces(
        session,
        "ic1",
        domain_filter="all",
        beam_state_filter=FILTER_BEAM_OFF,
        filter_column_keys=["ic1", "ic2"],
        beam_off_quiet_threshold=100.0,
    )
    assert len(off_traces) == 1
    assert off_traces[0][0].tolist() == [3.0, 4.0]


def test_probe_metric_availability_includes_field(g3_fft_data) -> None:
    session_data = g3_fft_data
    if not session_data:
        pytest.skip("timeslice FFT data unavailable in fixture")
    metrics = probe_metric_availability(session_data)
    assert METRIC_MAG_FIELD in metrics


@pytest.mark.slow
def test_fft_explorer_window_smoke_large_session(qt_wait) -> None:
    window = FftExplorerWindow([G3_LARGE_SESSION], str(TEST_DATA))
    qt_wait(
        lambda: bool(window._session_data) and bool(window.figure.axes),
        timeout_ms=15000,
    )
    assert window._read_config().channels
    window.close()


@pytest.mark.slow
def test_fft_explorer_window_smoke(qt_wait) -> None:
    window = FftExplorerWindow([G3_SESSION], str(TEST_DATA))
    qt_wait(
        lambda: bool(window._session_data) and bool(window.figure.axes),
        timeout_ms=20000,
    )
    assert window._read_config().channels
    window.close()
