#!/usr/bin/env python3
"""Capture screenshots for README documentation.

Usage:
    python scripts/capture_doc_screenshots.py
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

# Agg backend before any matplotlib import.
os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[1]
TEST_DATA = ROOT / "test_data"
OUT_DIR = ROOT / "docs" / "images"

# Sessions with rich data for representative plots.
SESSION_G3_A = "1943968267"
SESSION_G3_B = "1091134775"
SESSION_G2 = "590658542"

LAUNCHER_SIZE = (1400, 880)
VIEW_DPI = 150


def _ensure_paths() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not TEST_DATA.is_dir():
        raise SystemExit(f"test_data not found at {TEST_DATA}")


def _save_figure(fig, output: Path) -> None:
    import matplotlib.pyplot as plt

    fig.canvas.draw()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output,
        dpi=VIEW_DPI,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)


def _capture_timeslice_replay_view(
    session_ids: list[str],
    output: Path,
    *,
    base_dir: str,
    preset: str,
) -> None:
    import matplotlib.pyplot as plt

    from scan_kit.views.timeslice_replay_channels import (
        PRESET_CHANNELS,
        available_channel_keys,
        build_replay_config,
        filter_available_keys,
        load_sessions_catalog,
    )
    from scan_kit.views.timeslice_replay_ui import render_timeslice_replay

    session_data = load_sessions_catalog(session_ids, base_dir)
    available = available_channel_keys(session_data)
    keys = filter_available_keys(PRESET_CHANNELS[preset], available)
    config = build_replay_config(keys, session_data, title="Timeslice Replay")
    fig = plt.figure(figsize=config.figsize)
    render_timeslice_replay(fig, config, session_data, base_dir)
    _save_figure(fig, output)


def _capture_binned_summary_preset(
    session_ids: list[str],
    output: Path,
    *,
    base_dir: str,
    preset_id: str,
) -> None:
    import matplotlib.pyplot as plt

    from scan_kit.common import DEFAULT_SESSION_COLORS
    from scan_kit.common.plotting import finish_view
    from scan_kit.views.binned_summary_catalog import PRESET_BY_ID, BinnedSummaryConfig
    from scan_kit.views.binned_summary_data import load_sessions_summary
    from scan_kit.views.binned_summary_ui import render_binned_summary

    preset = PRESET_BY_ID[preset_id]
    session_data = load_sessions_summary(session_ids, base_dir)
    if not session_data:
        raise RuntimeError(f"No data for binned summary preset {preset_id!r}")

    config = BinnedSummaryConfig(
        y_group=preset.y_group,
        x_param=preset.x_param,
        glyph=preset.glyph,
        show_trend=preset.show_trend,
        show_hist=preset.show_hist,
        show_corr=preset.show_corr,
    )
    fig = plt.figure(figsize=(16, 9))
    render_binned_summary(fig, config, session_data, base_dir)
    loaded_ids = list(session_data.keys())
    finish_view(
        fig,
        config.title,
        loaded_ids,
        DEFAULT_SESSION_COLORS[: len(loaded_ids)],
        base_dir=base_dir,
    )
    _save_figure(fig, output)


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


def _select_sessions(browser, session_ids: list[str]) -> None:
    from PySide6.QtCore import Qt

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


def _capture_launcher_screenshots(base_dir: str) -> None:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from scan_kit.qt_launcher import ScanKitMainWindow, _MAIN_TAB_CONFIG_TUNING
    from scan_kit.qt_launcher import _MAIN_TAB_DATA_ANALYSIS, _MAIN_TAB_PLAN_SYNTHESIS
    from scan_kit.workflows.config_tuning.auto_tuning.paths import resolve_session_config_dir

    app = QApplication.instance() or QApplication(sys.argv)

    window = ScanKitMainWindow()
    window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    window.resize(*LAUNCHER_SIZE)
    window.show()

    _wait_until(lambda: window._session_browser is not None)
    browser = window._session_browser
    assert browser is not None

    browser.set_base_dir(base_dir)
    _wait_until(lambda: browser._scan_complete and browser._table.rowCount() >= 2)

    _select_sessions(browser, [SESSION_G3_A, SESSION_G3_B])
    app.processEvents()

    tabs = window._main_tabs
    assert tabs is not None

    def _switch_tab(tab_name: str) -> None:
        for i in range(tabs.count()):
            if tabs.tabText(i) == tab_name:
                tabs.setCurrentIndex(i)
                break
        app.processEvents()

    _switch_tab(_MAIN_TAB_DATA_ANALYSIS)
    _grab_widget(window, OUT_DIR / "launcher-data-analysis.png")

    _switch_tab(_MAIN_TAB_PLAN_SYNTHESIS)
    _grab_widget(window, OUT_DIR / "launcher-plan-synthesis.png")

    config_dir = resolve_session_config_dir(SESSION_G3_A, base_dir)
    if config_dir is not None:
        panel = window._config_tuning_panel
        if panel is not None and panel.open_config_root(config_dir, select_devices_xml=True):
            _switch_tab(_MAIN_TAB_CONFIG_TUNING)
            _grab_widget(window, OUT_DIR / "launcher-config-tuning.png")

    window.close()
    app.processEvents()


def main() -> None:
    sys.path.insert(0, str(ROOT))
    _ensure_paths()
    base_dir = str(TEST_DATA)

    print("Capturing launcher screenshots…")
    _capture_launcher_screenshots(base_dir)

    views = [
        ("position_scatter", [SESSION_G3_A, SESSION_G3_B], "view-position-scatter.png"),
        ("amplifier_correlation", [SESSION_G2], "view-amplifier-correlation.png"),
    ]
    binned_presets = [
        ("dose_ratio_energy", [SESSION_G3_A, SESSION_G3_B], "view-dose-ratios-energy.png"),
        ("sigma_energy", [SESSION_G3_A], "view-sigma-energy.png"),
    ]
    timeslice_views = [
        ("ic_current", [SESSION_G3_A], "view-ic-timeslice-replay.png"),
        ("field", [SESSION_G3_B], "view-magnetic-field-replay.png"),
    ]

    print("Capturing analysis view screenshots…")
    for module_name, sessions, filename in views:
        out = OUT_DIR / filename
        print(f"  {module_name} -> {out.name}")
        _capture_matplotlib_view(module_name, sessions, out, base_dir=base_dir)

    for preset_id, sessions, filename in binned_presets:
        out = OUT_DIR / filename
        print(f"  binned_summary/{preset_id} -> {out.name}")
        _capture_binned_summary_preset(
            sessions, out, base_dir=base_dir, preset_id=preset_id,
        )

    for preset, sessions, filename in timeslice_views:
        out = OUT_DIR / filename
        print(f"  timeslice_replay/{preset} -> {out.name}")
        _capture_timeslice_replay_view(
            sessions, out, base_dir=base_dir, preset=preset,
        )

    print(f"\nDone — {len(list(OUT_DIR.glob('*.png')))} images in {OUT_DIR}")


if __name__ == "__main__":
    main()
