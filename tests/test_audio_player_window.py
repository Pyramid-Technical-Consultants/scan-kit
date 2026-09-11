"""Smoke tests for IC audio player Qt shell."""

from __future__ import annotations

import pytest

from scan_kit.views.audio_player_window import AudioPlayerWindow
from tests.conftest import G3_SESSION, TEST_DATA


@pytest.mark.slow
def test_audio_player_window_smoke(qt_wait) -> None:
    window = AudioPlayerWindow([G3_SESSION], str(TEST_DATA))
    qt_wait(
        lambda: bool(window._session_data) and bool(window._channel_radios),
        timeout_ms=20000,
    )
    qt_wait(lambda: bool(window._playback_channels), timeout_ms=20000)
    assert window._read_config().channels
    assert window._read_config().beam_state_filter == "beam_on"
    assert window._play_pause_btn is not None
    assert window._seek_slider is not None
    assert window._fft_window_combo is not None
    assert window._fft_window_combo.currentData() == 1000
    assert window._rewind_btn is not None
    assert "start" in window._rewind_btn.toolTip().lower()
    assert window._volume_slider is not None
    assert window._volume_slider.value() == 100
    window.close()
