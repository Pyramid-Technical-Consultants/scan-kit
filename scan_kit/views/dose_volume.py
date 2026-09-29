"""3D dose-volume viewer (visPy + unified Qt controls)."""

from __future__ import annotations

from ..common import ViewSettings


def run(
    session_ids: list[str],
    base_dir: str = "test_data",
    *,
    settings: ViewSettings | None = None,
) -> None:
    """Open the 3D dose volume for the selected sessions."""
    from .vispy_plot import ensure_gl_plus

    # vispy locks its GL wrapper on first gloo import; request gl+ before the window.
    ensure_gl_plus()
    from .dose_volume_window import run_dose_volume_window

    run_dose_volume_window(session_ids, base_dir, settings=settings)
