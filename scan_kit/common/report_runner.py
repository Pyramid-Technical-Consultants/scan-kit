"""Headless view execution for PDF report generation."""

from __future__ import annotations

import traceback
from typing import Callable

import matplotlib.pyplot as plt

from .settings import ViewSettings


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
) -> plt.Figure | None:
    """Run *view_func* headlessly and return the figure shown via ``plt.show()``."""
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
    except Exception:
        traceback.print_exc()
        return None
    finally:
        plt.show = real_show
        for num in plt.get_fignums():
            if num not in {fig.number for fig in captured}:
                plt.close(num)

    if not captured:
        return None
    return captured[0]
