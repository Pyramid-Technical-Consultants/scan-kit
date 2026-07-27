"""Compatibility wrapper: opens unified Timeslice Replay on the sigma preset."""

from __future__ import annotations

from .timeslice_replay_channels import PRESET_SIGMA
from .timeslice_replay_window import run_timeslice_replay_window


def run(session_ids: list[str], base_dir: str = "test_data", *, settings=None) -> None:
    bg = bool(settings.bg_subtract) if settings is not None else False
    run_timeslice_replay_window(
        session_ids,
        base_dir,
        bg_subtract=bg,
        initial_preset=PRESET_SIGMA,
    )
