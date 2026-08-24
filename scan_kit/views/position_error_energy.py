"""Compatibility wrapper: opens Binned Summary on the position error vs energy preset."""

from __future__ import annotations

from .binned_summary_catalog import PRESET_POSITION_ERROR_ENERGY
from .binned_summary_preset import run_preset_view


def run(session_ids: list[str], base_dir: str = "test_data", *, settings=None) -> None:
    run_preset_view(session_ids, base_dir, PRESET_POSITION_ERROR_ENERGY, settings=settings)
