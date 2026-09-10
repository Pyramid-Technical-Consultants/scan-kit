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
    window.close()
