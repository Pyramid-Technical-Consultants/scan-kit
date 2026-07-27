"""Universal binned summary viewer entry point."""

from __future__ import annotations

from .binned_summary_window import run_binned_summary_window


def run(session_ids: list[str], base_dir: str = "test_data", *, settings=None) -> None:
    """Launch the configurable binned summary viewer."""
    run_binned_summary_window(session_ids, base_dir, settings=settings)
