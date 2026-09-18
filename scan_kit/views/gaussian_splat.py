"""3D Gaussian splat viewer (visPy + unified Qt controls)."""

from __future__ import annotations

from ..common import ViewSettings


def run(
    session_ids: list[str],
    base_dir: str = "test_data",
    *,
    settings: ViewSettings | None = None,
) -> None:
    """Open the configurable 3D Gaussian splat viewer for the selected sessions."""
    from .vispy_plot import ensure_gl_plus

    # vispy locks its GL wrapper on first gloo import; request gl+ before the window.
    ensure_gl_plus()
    from .gaussian_splat_window import run_gaussian_splat_window

    run_gaussian_splat_window(session_ids, base_dir, settings=settings)
