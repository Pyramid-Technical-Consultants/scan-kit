"""Live-updating view runner for scan-kit analysis views.

Wraps a view's ``run()`` function so that the matplotlib figure window
stays open and redraws in-place whenever ``settings.json`` changes on
disk.  The launcher saves settings; the subprocess picks up the
change via a 1-second polling timer on the matplotlib event loop.
"""

from __future__ import annotations

import os
import traceback
from typing import Callable

import matplotlib.pyplot as plt

from .settings import ViewSettings

_READY_SENTINEL = "__SCAN_KIT_PLOT_READY__"

_FIG_ONLY_KW = frozenset({
    "figsize", "dpi", "facecolor", "edgecolor", "frameon",
    "FigureClass", "clear", "layout", "num",
})


def run_with_live_settings(
    view_func: Callable,
    session_ids: list[str],
    base_dir: str,
    initial_settings_json: str,
) -> None:
    """Run *view_func* and silently refresh whenever settings change."""

    settings_path = os.path.join(base_dir, "settings.json")

    _state: dict = {
        "rerun": False,
        "existing_figs": [],
        "last_mtime": 0.0,
        "timer": None,
    }

    _orig_figure = plt.figure
    _orig_subplots = plt.subplots
    _real_show = plt.show

    # -- Patched plt.figure: reuse cleared window during re-runs -----------

    def _patched_figure(*args, **kwargs):
        if _state["rerun"] and _state["existing_figs"]:
            fig = _state["existing_figs"].pop(0)
            fig.clf()
            return fig
        return _orig_figure(*args, **kwargs)

    # -- Patched plt.subplots: reuse cleared window during re-runs ---------

    def _patched_subplots(*args, **kwargs):
        if _state["rerun"] and _state["existing_figs"]:
            fig = _state["existing_figs"].pop(0)
            fig.clf()
            sub_kw = {k: v for k, v in kwargs.items() if k not in _FIG_ONLY_KW}
            axes = fig.subplots(*args, **sub_kw)
            return fig, axes
        return _orig_subplots(*args, **kwargs)

    # -- Re-run logic ------------------------------------------------------

    def _do_rerun() -> None:
        _state["rerun"] = True
        _state["existing_figs"] = [_orig_figure(n) for n in plt.get_fignums()]
        try:
            settings = ViewSettings.load(base_dir)
            if settings.calibration_mode == "constrained":
                from .processing import compute_calibration_factors
                factors = compute_calibration_factors(session_ids, base_dir)
                settings.cal_factors = factors if factors else None
            view_func(session_ids, base_dir, settings=settings)
        except Exception:
            traceback.print_exc()
        finally:
            _state["rerun"] = False
            for fig in _state["existing_figs"]:
                plt.close(fig)
            _state["existing_figs"] = []
        for num in plt.get_fignums():
            try:
                _orig_figure(num).canvas.draw_idle()
            except Exception:
                pass

    # -- Timer callback: poll settings.json --------------------------------

    def _check_settings() -> None:
        try:
            mt = os.path.getmtime(settings_path)
        except OSError:
            return
        if mt == _state["last_mtime"]:
            return
        _state["last_mtime"] = mt
        try:
            _do_rerun()
        except Exception:
            traceback.print_exc()

    # -- Patched plt.show: maximize, emit sentinel, install timer ----------

    def _patched_show(*args, **kwargs):
        if _state["rerun"]:
            return

        for num in plt.get_fignums():
            try:
                _orig_figure(num).canvas.manager.window.state("zoomed")
            except Exception:
                try:
                    _orig_figure(num).canvas.manager.window.showMaximized()
                except Exception:
                    pass

        print(_READY_SENTINEL, flush=True)

        figs = plt.get_fignums()
        if figs:
            fig0 = _orig_figure(figs[0])
            timer = fig0.canvas.new_timer(interval=1000)
            timer.add_callback(_check_settings)
            timer.start()
            _state["timer"] = timer

        try:
            _state["last_mtime"] = os.path.getmtime(settings_path)
        except OSError:
            pass

        _real_show(*args, **kwargs)

    # -- Run ---------------------------------------------------------------

    plt.figure = _patched_figure
    plt.subplots = _patched_subplots
    plt.show = _patched_show
    try:
        settings = ViewSettings.from_json(initial_settings_json)
        view_func(session_ids, base_dir, settings=settings)
    finally:
        plt.figure = _orig_figure
        plt.subplots = _orig_subplots
        plt.show = _real_show
