"""Headless view execution for PDF report generation."""

from __future__ import annotations

import logging
from typing import Callable

import matplotlib.pyplot as plt

from .settings import ViewSettings

_log = logging.getLogger(__name__)
_SKIP_NO_DATA = "No data available for selected sessions"


def _prepare_settings(
    session_ids: list[str],
    base_dir: str,
    settings: ViewSettings,
) -> ViewSettings:
    """Apply runtime calibration factors the same way as view_runner."""
    if settings.calibration_mode != "constrained":
        return settings
    from .processing import compute_calibration_factors

    factors = compute_calibration_factors(session_ids, base_dir)
    settings.cal_factors = factors if factors else None
    return settings


def capture_view_figure(
    view_func: Callable,
    session_ids: list[str],
    base_dir: str,
    settings: ViewSettings | None,
) -> tuple[plt.Figure | None, str | None]:
    """Run *view_func* headlessly and return ``(figure, skip_reason)``.

    Patches ``plt.show`` for views that call it, and also collects any figures
    still open after the view returns (covers legacy paths that skipped show on
    Agg). Both paths are needed for reliable PDF report capture.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)

    captured: list[plt.Figure] = []
    real_show = plt.show

    def _patched_show(*args, **kwargs) -> None:
        del args, kwargs
        for num in plt.get_fignums():
            captured.append(plt.figure(num))

    prepared = _prepare_settings(
        session_ids,
        base_dir,
        settings if settings is not None else ViewSettings(),
    )

    plt.show = _patched_show
    try:
        view_func(session_ids, base_dir, settings=prepared)
    except Exception as exc:
        _log.exception("Report view raised an exception")
        return None, f"View error: {exc}"
    finally:
        plt.show = real_show
        for num in plt.get_fignums():
            fig = plt.figure(num)
            if not any(existing.number == fig.number for existing in captured):
                captured.append(fig)
        for num in plt.get_fignums():
            if num not in {fig.number for fig in captured}:
                plt.close(num)

    if not captured:
        _log.warning("Report view produced no figure")
        return None, _SKIP_NO_DATA
    return captured[0], None
