"""FFT Explorer entry point."""

from __future__ import annotations

from .fft_window import run_fft_explorer_window


def run(session_ids: list[str], base_dir: str = "test_data", *, settings=None) -> None:
    """Launch the configurable FFT explorer for timeslice signals."""
    run_fft_explorer_window(session_ids, base_dir, settings=settings)
