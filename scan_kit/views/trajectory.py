"""3D IC beam trajectory viewer (visPy + unified Qt controls)."""

from __future__ import annotations

from ..common import ViewSettings
from .trajectory_window import run_trajectory_window


def run(
    session_ids: list[str],
    base_dir: str = "test_data",
    *,
    settings: ViewSettings | None = None,
) -> None:
    """Open the configurable 3D trajectory viewer for the selected sessions."""
    run_trajectory_window(session_ids, base_dir, settings=settings)
