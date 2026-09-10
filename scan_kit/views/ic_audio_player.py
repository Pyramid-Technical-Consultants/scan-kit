"""Audio Explorer entry point."""

from __future__ import annotations

from .audio_player_window import run_audio_player_window


def run(session_ids: list[str], base_dir: str = "test_data", *, settings=None) -> None:
    """Launch the unified IC audio player for timeslice signals."""
    run_audio_player_window(session_ids, base_dir, settings=settings)
