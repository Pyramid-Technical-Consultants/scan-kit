"""Live-updating view runner for scan-kit analysis views.

Wraps a view's ``run()`` function so that the matplotlib figure window
stays open and redraws in-place whenever ``settings.json`` changes on
disk.  The launcher saves settings; the subprocess picks up the
change via a 1-second polling timer on the matplotlib event loop.
"""

from __future__ import annotations

import logging
import os
import sys
import traceback
from typing import Any, Callable

from .app_icon import load_app_icon, prepare_qt_app_identity
from .matplotlib_backend import init_matplotlib_for_views
from .settings import ViewSettings

_READY_SENTINEL = "__SCAN_KIT_PLOT_READY__"
#: Printed once a warm worker has imported the heavy stack and is ready for a command.
WARM_WORKER_SENTINEL = "__SCAN_KIT_WORKER_WARM__"

_FIG_ONLY_KW = frozenset({
    "figsize", "dpi", "facecolor", "edgecolor", "frameon",
    "FigureClass", "clear", "layout", "num",
})

# Delays (ms) after show / resize before re-running toolbar tight layout.
_LAYOUT_DELAYS_MS = (0, 100, 300, 600)
_LAYOUT_HOOK = "_scan_kit_layout_hook"
_VIEW_LOG_FORMAT = "%(levelname)s [%(name)s] %(message)s"


def _configure_view_logging() -> None:
    """Send view-process log records to stderr so the launcher can capture them."""
    logging.basicConfig(level=logging.INFO, format=_VIEW_LOG_FORMAT, force=True)


def _get_pyplot():
    """Import pyplot after the GUI backend is configured in frozen builds."""
    init_matplotlib_for_views()
    import matplotlib.pyplot as plt

    return plt


def warm_worker_main() -> None:
    """Pre-warmed view worker entry point.

    Spawned idle by the launcher's worker pool so the expensive imports
    (matplotlib/scipy/pandas) are already paid for before the user clicks a
    view. After printing :data:`WARM_WORKER_SENTINEL`, it blocks reading a
    single JSON render command from stdin:
    ``{"module", "sessions", "base_dir", "settings"}``.
    """
    import importlib
    import json

    _configure_view_logging()
    init_matplotlib_for_views()
    _get_pyplot()

    for _mod in ("scipy.signal", "scipy.fft", "pandas"):
        try:
            importlib.import_module(_mod)
        except Exception:
            pass

    print(WARM_WORKER_SENTINEL, flush=True)

    line = sys.stdin.readline()
    if not line:
        return
    try:
        command = json.loads(line)
    except Exception:
        return

    mod = importlib.import_module(f"scan_kit.views.{command['module']}")
    run_with_live_settings(
        mod.run,
        command["sessions"],
        command["base_dir"],
        command["settings"],
    )


def run_with_live_settings(
    view_func: Callable,
    session_ids: list[str],
    base_dir: str,
    initial_settings_json: str,
) -> None:
    """Run *view_func* and silently refresh whenever settings change."""
    _configure_view_logging()
    plt = _get_pyplot()
    from .plotting import apply_toolbar_tight_layout

    settings_path = os.path.join(base_dir, "settings.json")

    _state: dict[str, Any] = {
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

    # -- Figure layout after maximize / resize ------------------------------

    def _set_figure_window_icon(fig) -> None:
        try:
            window = fig.canvas.manager.window
            icon = load_app_icon()
            if not icon.isNull() and hasattr(window, "setWindowIcon"):
                window.setWindowIcon(icon)
        except Exception:
            pass

    def _maximize_figure_window(fig) -> None:
        try:
            fig.canvas.manager.window.state("zoomed")
        except Exception:
            try:
                fig.canvas.manager.window.showMaximized()
            except Exception:
                pass

    def _apply_toolbar_layout(fig) -> None:
        try:
            apply_toolbar_tight_layout(fig)
        except Exception:
            pass

    def _queue_delayed_layout(fig, delays_ms) -> None:
        try:
            from matplotlib.backends.qt_compat import QtCore

            for delay_ms in delays_ms:
                QtCore.QTimer.singleShot(
                    delay_ms, lambda f=fig: _apply_toolbar_layout(f),
                )
            return
        except Exception:
            pass

        for delay_ms in delays_ms:
            if delay_ms <= 0:
                continue
            try:
                timer = fig.canvas.new_timer(interval=delay_ms)
                timer.add_callback(lambda f=fig: _apply_toolbar_layout(f))
                timer.start()
            except Exception:
                pass

    def _schedule_toolbar_tight_layout(
        fig, *, delays_ms=_LAYOUT_DELAYS_MS, immediate=False,
    ) -> None:
        """Re-run toolbar tight layout once the canvas has its final size."""
        if immediate:
            _apply_toolbar_layout(fig)

        if getattr(fig, _LAYOUT_HOOK, False):
            _queue_delayed_layout(fig, (100, 300) if immediate else delays_ms)
            return

        setattr(fig, _LAYOUT_HOOK, True)
        fig.canvas.mpl_connect(
            "resize_event", lambda _event, f=fig: _apply_toolbar_layout(f),
        )
        _queue_delayed_layout(fig, delays_ms)

    def _refresh_figure_layout(fig) -> None:
        _schedule_toolbar_tight_layout(fig, immediate=True)

    def _finalize_figure_layout(fig) -> None:
        _maximize_figure_window(fig)
        _set_figure_window_icon(fig)
        _schedule_toolbar_tight_layout(fig)

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
                _refresh_figure_layout(_orig_figure(num))
            except Exception:
                pass

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
                _finalize_figure_layout(_orig_figure(num))
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

    prepare_qt_app_identity()

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
