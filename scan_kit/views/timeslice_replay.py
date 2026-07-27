"""Unified timeslice replay: Qt shell with selectable data channels."""

from __future__ import annotations

from .timeslice_replay_window import run_timeslice_replay_window


def run(session_ids: list[str], base_dir: str = "test_data", *, settings=None) -> None:
    """Launch the unified timeslice replay viewer."""
    bg = bool(settings.bg_subtract) if settings is not None else False
    run_timeslice_replay_window(session_ids, base_dir, bg_subtract=bg)
