"""Tests for FFT Explorer loaders, renderer, and Qt shell."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from scan_kit.views.fft_catalog import FftConfig, PRESET_ALL_ICS, PRESET_BY_ID, SIGNAL_IC1
from scan_kit.views.fft_data import load_sessions_fft, probe_signal_availability, welch_psd
from scan_kit.views.fft_ui import render_fft
from scan_kit.views.fft_window import FftExplorerWindow

TEST_DATA = Path(__file__).resolve().parents[1] / "test_data"
G3_SESSION = "1091134775"
G3_LARGE_SESSION = "1242721320"


def test_welch_psd_returns_band_limited_frequencies() -> None:
    import numpy as np

    signal = np.sin(np.linspace(0, 40 * np.pi, 8000))
    freqs, psd = welch_psd(signal, 1000.0, 1024, 0.5)
    assert freqs is not None and psd is not None
    assert len(freqs) == len(psd)


def test_load_sessions_fft_g3() -> None:
    session_data = load_sessions_fft([G3_SESSION], str(TEST_DATA))
    if not session_data:
        pytest.skip("timeslice FFT data unavailable in fixture")
    assert G3_SESSION in session_data
    availability = probe_signal_availability(
        [G3_SESSION], str(TEST_DATA), session_data=session_data,
    )
    assert availability[SIGNAL_IC1]


def test_render_fft_headless() -> None:
    from scan_kit.views.plot_view_shell import new_headless_figure

    session_data = load_sessions_fft([G3_SESSION], str(TEST_DATA))
    if not session_data:
        pytest.skip("timeslice FFT data unavailable in fixture")
    config = PRESET_BY_ID[PRESET_ALL_ICS]
    fig = new_headless_figure((16, 9))
    render_fft(
        fig,
        FftConfig(
            signals=config.signals,
            domain_filter=config.domain_filter,
            beam_state_filter=config.beam_state_filter,
            annotate_peaks=config.annotate_peaks,
        ),
        session_data,
        str(TEST_DATA),
    )
    assert fig.axes
    plt.close(fig)


def test_load_sessions_fft_g3_large() -> None:
    session_data = load_sessions_fft([G3_LARGE_SESSION], str(TEST_DATA))
    if not session_data:
        pytest.skip("timeslice FFT data unavailable in fixture")
    assert G3_LARGE_SESSION in session_data
    assert len(session_data[G3_LARGE_SESSION]["ic1_current"]) > 0
    availability = probe_signal_availability(
        [G3_LARGE_SESSION], str(TEST_DATA), session_data=session_data,
    )
    assert availability[SIGNAL_IC1]
    assert availability["ic3"]


def test_extract_fft_traces_respects_beam_filter() -> None:
    import numpy as np

    from scan_kit.common.data_filter import FILTER_BEAM_OFF, FILTER_BEAM_ON
    from scan_kit.views.fft_data import extract_fft_traces

    session = {
        "ic1_current": np.array([1.0, 2.0, 3.0, 4.0]),
        "ic2_current": np.array([1.0, 2.0, 3.0, 4.0]),
        "beam_on": np.array([True, True, False, False]),
    }
    on_traces = extract_fft_traces(
        session,
        "ic1",
        domain_filter="all",
        beam_state_filter=FILTER_BEAM_ON,
        filter_column_keys=["ic1_current", "ic2_current"],
    )
    assert len(on_traces) == 1
    assert on_traces[0][0].tolist() == [1.0, 2.0]

    off_traces = extract_fft_traces(
        session,
        "ic1",
        domain_filter="all",
        beam_state_filter=FILTER_BEAM_OFF,
        filter_column_keys=["ic1_current", "ic2_current"],
    )
    assert len(off_traces) == 1
    assert off_traces[0][0].tolist() == [3.0, 4.0]


def test_fft_explorer_window_smoke_large_session(qapp) -> None:
    import time

    window = FftExplorerWindow([G3_LARGE_SESSION], str(TEST_DATA))
    for _ in range(200):
        qapp.processEvents()
        time.sleep(0.05)
    assert window._session_data
    assert window._read_config().signals
    assert window.figure.axes, "FFT plot should render for large G3 session"
    window.close()


def test_fft_explorer_window_smoke(qapp) -> None:
    import time

    window = FftExplorerWindow([G3_SESSION], str(TEST_DATA))
    for _ in range(200):
        qapp.processEvents()
        time.sleep(0.05)
    assert window._session_data
    assert window._read_config().signals
    assert window.figure.axes, "FFT plot should render after data load"
    window.close()
