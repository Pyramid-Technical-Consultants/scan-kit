"""Compatibility wrapper: opens Distribution Explorer on the spot position preset."""

from __future__ import annotations

from .distribution_catalog import PRESET_POSITION_SPOT
from .distribution_window import run_distribution_explorer_window


def run(session_ids: list[str], base_dir: str = "test_data", *, settings=None) -> None:
    run_distribution_explorer_window(
        session_ids,
        base_dir,
        settings=settings,
        initial_preset=PRESET_POSITION_SPOT,
    )
