"""Compatibility wrapper: opens unified Timeslice Replay on the field preset."""

from __future__ import annotations

from .timeslice_replay_channels import PRESET_FIELD
from .timeslice_replay_window import run_timeslice_replay_window


def run(session_ids: list[str], base_dir: str = "test_data", *, settings=None) -> None:
    del settings
    run_timeslice_replay_window(
        session_ids,
        base_dir,
        initial_preset=PRESET_FIELD,
    )
