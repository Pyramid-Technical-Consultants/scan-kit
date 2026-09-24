#!/usr/bin/env python3
"""Capture screenshots for README documentation.

Usage:
    python scripts/capture_doc_screenshots.py
    python scripts/capture_doc_screenshots.py dose_volume   # one job whose label contains this

Requires a local test_data/ folder (not shipped with the repo). Forces the dark
theme for the grab only; it does not persist View > Theme.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_DATA = ROOT / "test_data"
OUT_DIR = ROOT / "docs" / "images"

# Sessions with rich data for representative plots.
SESSION_G3_A = "1943968267"
SESSION_G3_B = "1091134775"
SESSION_G2 = "590658542"
# Adjacent in the date-sorted table so both Use swatches show in the launcher shot.
LAUNCHER_SESSIONS = ["656350661", "1943968267"]

LAUNCHER_SIZE = (1400, 760)
VIEW_SIZE = (1400, 860)
PUBLIC_DATA_DIR = "test_data"
PUBLIC_RCI_HOST = "192.168.100.184"


def _ensure_paths() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not TEST_DATA.is_dir():
        raise SystemExit(f"test_data not found at {TEST_DATA}")


def _qt_app():
    from PySide6.QtWidgets import QApplication

    from scan_kit.common.qt_theme import apply_ui_theme

    app = QApplication.instance() or QApplication(sys.argv)
    apply_ui_theme("dark", app=app)
    return app


def _wait_until(predicate, *, timeout_ms: int = 180_000, interval_ms: int = 200) -> None:
    from PySide6.QtCore import QEventLoop, QTimer

    loop = QEventLoop()
    timer = QTimer()
    timer.setInterval(interval_ms)
    elapsed = {"ms": 0}

    def tick() -> None:
        elapsed["ms"] += interval_ms
        if predicate() or elapsed["ms"] >= timeout_ms:
            loop.quit()

    timer.timeout.connect(tick)
    timer.start()
    loop.exec()
    if not predicate():
        raise TimeoutError("Timed out waiting for UI readiness")


def _grab_widget(widget, path: Path) -> None:
    from PySide6.QtWidgets import QApplication

    path.parent.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance()
    assert app is not None
    for _ in range(8):
        app.processEvents()
    pix = widget.grab()
    if pix.isNull():
        raise RuntimeError(f"grab() returned null pixmap for {path.name}")
    if not pix.save(str(path), "PNG"):
        raise RuntimeError(f"Failed to write {path}")


def _figure_has_axes(window) -> bool:
    fig = getattr(window, "figure", None)
    return fig is not None and bool(fig.axes)


def _prepare_offscreen(window, size: tuple[int, int]) -> None:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    window.resize(*size)
    window.show()
    app = QApplication.instance()
    assert app is not None
    app.processEvents()


def _capture_qt_window(window, output: Path, *, ready, size: tuple[int, int] = VIEW_SIZE) -> None:
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    assert app is not None
    _prepare_offscreen(window, size)
    _wait_until(ready)
    if hasattr(window, "canvas"):
        window.canvas.draw()
    for _ in range(8):
        app.processEvents()
    _grab_widget(window, output)
    window.close()
    app.processEvents()


def _save_figure(fig, output: Path) -> None:
    import matplotlib.pyplot as plt

    fig.canvas.draw()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output,
        dpi=150,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)


def _select_sessions(browser, session_ids: list[str]) -> None:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QAbstractItemView

    from scan_kit.common.session_browser import _COL_SESSION_ID, _COL_USE

    want = set(session_ids)
    table = browser._table
    table.blockSignals(True)
    try:
        for row in range(table.rowCount()):
            sid_item = table.item(row, _COL_SESSION_ID)
            use_item = table.item(row, _COL_USE)
            if sid_item is None or use_item is None:
                continue
            checked = sid_item.text() in want
            use_item.setCheckState(
                Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
            )
    finally:
        table.blockSignals(False)
    browser._check_order = list(session_ids)
    browser._persist_selection()
    browser._schedule_status_refresh()
    if session_ids:
        for row in range(table.rowCount()):
            sid_item = table.item(row, _COL_SESSION_ID)
            if sid_item is not None and sid_item.text() == session_ids[0]:
                table.scrollToItem(
                    sid_item, QAbstractItemView.ScrollHint.PositionAtCenter
                )
                break


def _switch_tab(window, tab_name: str) -> None:
    from PySide6.QtWidgets import QApplication

    tabs = window._main_tabs
    assert tabs is not None
    for i in range(tabs.count()):
        if tabs.tabText(i) == tab_name:
            tabs.setCurrentIndex(i)
            break
    app = QApplication.instance()
    assert app is not None
    app.processEvents()


def _generate_zero_field_preview(panel) -> None:
    from PySide6.QtWidgets import QPushButton

    for btn in panel.findChildren(QPushButton):
        if btn.text() == "10 MeV Steps":
            btn.click()
            break
    panel._on_generate()
    _wait_until(
        lambda: panel._generated is not None and not panel._generating,
        timeout_ms=120_000,
    )
    _wait_until(lambda: panel._preview_table.rowCount() >= 8, timeout_ms=60_000)


def _capture_launcher_screenshots(base_dir: str) -> None:
    from PySide6.QtWidgets import QApplication, QSplitter

    from scan_kit.qt_launcher import (
        ScanKitMainWindow,
        _MAIN_TAB_CONFIG_TUNING,
        _MAIN_TAB_DATA_ANALYSIS,
        _MAIN_TAB_PLAN_RUNNER,
        _MAIN_TAB_PLAN_SYNTHESIS,
    )
    from scan_kit.workflows.config_tuning.auto_tuning.paths import resolve_session_config_dir
    from scan_kit.workflows.plan_synthesis_panel import PlanSynthesisPanel

    app = QApplication.instance()
    assert app is not None

    window = ScanKitMainWindow()
    _prepare_offscreen(window, LAUNCHER_SIZE)

    _wait_until(
        lambda: window._main_tabs is not None and window._main_tabs.count() >= 5
    )
    _wait_until(lambda: window._session_browser is not None)
    browser = window._session_browser
    assert browser is not None

    browser.set_base_dir(base_dir)
    _wait_until(lambda: browser._scan_complete and browser._table.rowCount() >= 2)

    _select_sessions(browser, LAUNCHER_SESSIONS)
    _wait_until(lambda: browser._scan_complete)
    _select_sessions(browser, LAUNCHER_SESSIONS)
    browser._base_dir_input.setText(PUBLIC_DATA_DIR)
    app.processEvents()

    runner = window._plan_runner_panel
    if runner is not None:
        runner._host_edit.setText(PUBLIC_RCI_HOST)

    tabs = window._main_tabs
    assert tabs is not None

    _switch_tab(window, _MAIN_TAB_DATA_ANALYSIS)
    data_page = None
    for i in range(tabs.count()):
        if tabs.tabText(i) == _MAIN_TAB_DATA_ANALYSIS:
            data_page = tabs.widget(i)
            break
    if data_page is not None:
        for splitter in data_page.findChildren(QSplitter):
            splitter.setSizes([720, 680])
            break
    app.processEvents()
    _grab_widget(window, OUT_DIR / "launcher-data-analysis.png")

    _switch_tab(window, _MAIN_TAB_PLAN_SYNTHESIS)
    plan_panel = None
    for i in range(tabs.count()):
        widget = tabs.widget(i)
        if isinstance(widget, PlanSynthesisPanel):
            plan_panel = widget
            break
    if plan_panel is not None:
        try:
            _generate_zero_field_preview(plan_panel)
        except Exception as exc:
            print(f"  (plan preview skipped: {exc})")
        app.processEvents()
    _grab_widget(window, OUT_DIR / "launcher-plan-synthesis.png")

    _switch_tab(window, _MAIN_TAB_PLAN_RUNNER)
    _grab_widget(window, OUT_DIR / "launcher-plan-runner.png")

    config_dir = resolve_session_config_dir(SESSION_G3_A, base_dir)
    panel = window._config_tuning_panel
    if config_dir is not None and panel is not None:
        if panel.open_config_root(config_dir, select_devices_xml=True):
            panel._path_input.setText(f"{PUBLIC_DATA_DIR}/{SESSION_G3_A}/config")
            _switch_tab(window, _MAIN_TAB_CONFIG_TUNING)
            app.processEvents()
            _grab_widget(window, OUT_DIR / "launcher-config-tuning.png")

    window.close()
    app.processEvents()


def _capture_binned_summary(session_ids: list[str], output: Path, *, base_dir: str, preset_id: str) -> None:
    from scan_kit.views.binned_summary_window import BinnedSummaryWindow

    window = BinnedSummaryWindow(session_ids, base_dir, initial_preset=preset_id)
    _capture_qt_window(window, output, ready=lambda: _figure_has_axes(window))


def _capture_distribution(session_ids: list[str], output: Path, *, base_dir: str, preset_id: str) -> None:
    from scan_kit.views.distribution_window import DistributionExplorerWindow

    window = DistributionExplorerWindow(session_ids, base_dir, initial_preset=preset_id)
    _capture_qt_window(window, output, ready=lambda: _figure_has_axes(window))


def _capture_timeslice_replay(session_ids: list[str], output: Path, *, base_dir: str, preset: str) -> None:
    from scan_kit.views.timeslice_replay_window import TimesliceReplayWindow

    window = TimesliceReplayWindow(session_ids, base_dir, initial_preset=preset)
    _capture_qt_window(window, output, ready=lambda: _figure_has_axes(window))


def _capture_fft_explorer(session_ids: list[str], output: Path, *, base_dir: str, preset_id: str) -> None:
    from scan_kit.views.fft_window import FftExplorerWindow

    window = FftExplorerWindow(session_ids, base_dir, initial_preset=preset_id)
    _capture_qt_window(window, output, ready=lambda: _figure_has_axes(window))


def _dose_frame(window):
    """Scene plus the ray march. ``canvas.render`` stops before the march."""
    import numpy as np
    from vispy import gloo

    canvas = window._vispy_canvas
    scene = window._scene
    canvas.set_current()
    scene._place_labels()
    csize = canvas.size
    scale = canvas.pixel_scale
    size = tuple(int(x * scale) for x in csize)
    fbo = gloo.FrameBuffer(
        color=gloo.RenderBuffer(size[::-1]),
        depth=gloo.RenderBuffer(size[::-1]),
    )
    canvas.push_fbo(fbo, (0, 0), csize)
    try:
        canvas._draw_scene()
        if scene._has_volume and not scene._broken:
            scene._box.draw_volume(canvas)
            for line in scene._late:
                line.held = False
                if line.visible:
                    line.draw()
        frame = np.asarray(fbo.read())
    finally:
        canvas.pop_fbo()
    return frame


def _composite_vispy(window, pix):
    """Paste the visPy framebuffer over the blank OpenGL widget in a window grab."""
    import numpy as np
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QImage, QPainter

    frame = _dose_frame(window)
    if frame.ndim != 3 or frame.shape[2] < 3:
        return pix
    frame = np.ascontiguousarray(frame[:, :, :3], dtype=np.uint8)
    height, width, _ = frame.shape
    image = QImage(frame.tobytes(), width, height, 3 * width, QImage.Format.Format_RGB888).copy()
    canvas = window._vispy_canvas
    native = canvas.native
    origin = native.mapTo(window, QPoint(0, 0))
    ratio = window.devicePixelRatioF()
    painter = QPainter(pix)
    painter.drawImage(
        int(origin.x() * ratio),
        int(origin.y() * ratio),
        image.scaled(int(native.width() * ratio), int(native.height() * ratio)),
    )
    painter.end()
    return pix


def _capture_dose_volume(session_ids: list[str], output: Path, *, base_dir: str, preset_id: str) -> None:
    from PySide6.QtWidgets import QApplication

    from scan_kit.views.dose_volume_window import DoseVolumeWindow

    window = DoseVolumeWindow(session_ids, base_dir, initial_preset=preset_id)
    app = QApplication.instance()
    assert app is not None
    # A real GL surface: DontShowOnScreen leaves the visPy canvas blank.
    window.resize(*VIEW_SIZE)
    window.show()
    app.processEvents()
    _wait_until(
        lambda: window._scene._has_volume and "×" in window._grid_label.text(),
        timeout_ms=180_000,
    )
    camera = window._scene._view.camera
    camera.azimuth = 40.0
    camera.elevation = -28.0
    camera.roll = 0.0
    for _ in range(20):
        window._vispy_canvas.update()
        app.processEvents()
    pix = window.grab()
    if pix.isNull():
        raise RuntimeError(f"grab() returned null pixmap for {output.name}")
    pix = _composite_vispy(window, pix)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not pix.save(str(output), "PNG"):
        raise RuntimeError(f"Failed to write {output}")
    window.close()
    app.processEvents()


def _capture_session_log(session_ids: list[str], output: Path, *, base_dir: str) -> None:
    from scan_kit.common.plot_colors import DEFAULT_SESSION_COLORS
    from scan_kit.common.session_log import load_session_log
    from scan_kit.views.session_log_compare import SessionLogCompareWindow

    logs = []
    for sid in session_ids:
        data = load_session_log(sid, base_dir)
        if data is not None and data.entries:
            logs.append(data)
    if not logs:
        raise RuntimeError("No session logs loaded for screenshot")
    colors = [
        DEFAULT_SESSION_COLORS[i % len(DEFAULT_SESSION_COLORS)]
        for i in range(len(logs))
    ]
    window = SessionLogCompareWindow(logs, colors)
    _capture_qt_window(window, output, ready=lambda: True, size=(1400, 820))


def _capture_matplotlib_view(
    module_name: str,
    session_ids: list[str],
    output: Path,
    *,
    base_dir: str,
) -> None:
    import matplotlib.pyplot as plt

    real_show = plt.show

    def _save_show(*_args, **_kwargs) -> None:
        _save_figure(plt.gcf(), output)

    plt.show = _save_show
    try:
        mod = importlib.import_module(f"scan_kit.views.{module_name}")
        mod.run(session_ids, base_dir)
    finally:
        plt.show = real_show

    if not output.is_file():
        raise RuntimeError(f"Failed to capture {module_name} → {output}")


def main(only: list[str] | None = None) -> None:
    sys.path.insert(0, str(ROOT))
    os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.*=false")
    _ensure_paths()
    base_dir = str(TEST_DATA)
    _qt_app()

    if not only:
        print("Capturing launcher screenshots…")
        _capture_launcher_screenshots(base_dir)

    print("Capturing analysis view screenshots…")
    jobs = [
        (
            "distribution/position_spot",
            lambda: _capture_distribution(
                [SESSION_G3_A, SESSION_G3_B],
                OUT_DIR / "view-distribution-explorer.png",
                base_dir=base_dir,
                preset_id="position_spot",
            ),
        ),
        (
            "binned_summary/sigma_energy",
            lambda: _capture_binned_summary(
                [SESSION_G3_A],
                OUT_DIR / "view-sigma-energy.png",
                base_dir=base_dir,
                preset_id="sigma_energy",
            ),
        ),
        (
            "binned_summary/dose_ratio_energy",
            lambda: _capture_binned_summary(
                [SESSION_G3_A, SESSION_G3_B],
                OUT_DIR / "view-dose-ratios-energy.png",
                base_dir=base_dir,
                preset_id="dose_ratio_energy",
            ),
        ),
        (
            "timeslice_replay/ic_current",
            lambda: _capture_timeslice_replay(
                [SESSION_G3_A],
                OUT_DIR / "view-ic-timeslice-replay.png",
                base_dir=base_dir,
                preset="ic_current",
            ),
        ),
        (
            "timeslice_replay/mag_field",
            lambda: _capture_timeslice_replay(
                [SESSION_G3_B],
                OUT_DIR / "view-magnetic-field-replay.png",
                base_dir=base_dir,
                preset="mag_field",
            ),
        ),
        (
            "fft_explorer/all_ics",
            lambda: _capture_fft_explorer(
                [SESSION_G3_A],
                OUT_DIR / "view-fft-explorer.png",
                base_dir=base_dir,
                preset_id="all_ics",
            ),
        ),
        (
            "session_log_compare",
            lambda: _capture_session_log(
                [SESSION_G3_A, SESSION_G3_B],
                OUT_DIR / "view-session-log-compare.png",
                base_dir=base_dir,
            ),
        ),
        (
            "dose_volume/spot_ic1",
            lambda: _capture_dose_volume(
                ["1093436476"],
                OUT_DIR / "view-dose-volume.png",
                base_dir=base_dir,
                preset_id="spot_ic1",
            ),
        ),
        (
            "amplifier_correlation",
            lambda: _capture_matplotlib_view(
                "amplifier_correlation",
                [SESSION_G2],
                OUT_DIR / "view-amplifier-correlation.png",
                base_dir=base_dir,
            ),
        ),
    ]
    for label, job in jobs:
        if only and not any(part in label for part in only):
            continue
        print(f"  {label} ->")
        job()

    stale = OUT_DIR / "view-position-scatter.png"
    if stale.is_file():
        stale.unlink()
        print(f"Removed stale {stale.name}")

    print(f"\nDone: {len(list(OUT_DIR.glob('*.png')))} images in {OUT_DIR}")


if __name__ == "__main__":
    main(sys.argv[1:])
