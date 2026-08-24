"""Compatibility wrapper: opens Binned Summary on the dose ratios vs energy preset."""

from __future__ import annotations

from .binned_summary_catalog import PRESET_DOSE_RATIO_ENERGY
from .binned_summary_preset import run_preset_view


def run(session_ids: list[str], base_dir: str = "test_data", *, settings=None) -> None:
    run_preset_view(session_ids, base_dir, PRESET_DOSE_RATIO_ENERGY, settings=settings)
