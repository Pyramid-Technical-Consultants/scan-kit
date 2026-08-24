"""Distribution Explorer entry point."""

from __future__ import annotations

from .distribution_window import run_distribution_explorer_window


def run(session_ids: list[str], base_dir: str = "test_data", *, settings=None) -> None:
    """Launch the configurable distribution and fit-quality viewer."""
    run_distribution_explorer_window(session_ids, base_dir, settings=settings)
