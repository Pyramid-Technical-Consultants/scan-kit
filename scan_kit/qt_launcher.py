"""Qt GUI launcher for scan-kit analysis views."""

from __future__ import annotations

import atexit
import json
import multiprocessing
import os
import signal
import subprocess
import sys
import threading
from functools import partial
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QThread, QTimer, QUrl, Signal, Slot
from PySide6.QtGui import (
    QCloseEvent,
    QDesktopServices,
    QIcon,
    QKeySequence,
    QMoveEvent,
    QResizeEvent,
    QShortcut,
)
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from . import __version__
from .common.app_icon import apply_windows_window_icons, load_app_icon
from .common.app_settings import AppSettings
from .common.session_browser import SessionBrowserWidget
from .common.session_meta import SessionMeta
from .common.settings import ViewSettings, CALIBRATION_MODES
from .common.qt_widgets import make_pane_scroll_area, set_pane_scroll_widget
from .views import VIEW_GROUPS, VIEWS
from .workflows.plan_synthesis_panel import PlanSynthesisPanel
from .workflows.config_tuning.auto_tuning.paths import resolve_session_config_dir
from .workflows.config_tuning_panel import ConfigTuningPanel
from .workflows.report import reportable_module_names
from .workflows.report.generation_worker import ReportGenerationWorker
from .workflows.report.wizard import ReportWizardDialog

MAX_SESSIONS = 5
FROZEN = getattr(sys, "frozen", False)

if FROZEN:
    PROJECT_ROOT = Path(sys.executable).resolve().parent
else:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Analysis launcher button grid (see VIEW_GROUPS for workflow sections).
_VIEW_GRID_COLS = 2

_MAIN_TAB_DATA_ANALYSIS = "Data Analysis"
_MAIN_TAB_PLAN_SYNTHESIS = "Plan Synthesis"
_MAIN_TAB_CONFIG_TUNING = "Configuration Tuning"

_DEFAULT_WINDOW_WIDTH = 1400
_DEFAULT_WINDOW_HEIGHT = 880

_CAL_MODE_LABELS = {
    "off": "Off",
    "per_session": "Per-Session",
    "constrained": "Constrained",
}

_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_READY_SENTINEL = "__SCAN_KIT_PLOT_READY__"
#: Printed by a warm worker once the heavy scientific/plotting stack is imported.
_WORKER_WARM_SENTINEL = "__SCAN_KIT_WORKER_WARM__"
#: Number of pre-warmed view worker processes kept idle and ready to render.
_WARM_POOL_SIZE = 2


class _WarmWorker:
    """A pre-spawned subprocess that has imported the heavy stack and awaits a view command.

    ``module_name`` stays ``None`` while the worker is idle in the pool and is
    set to the view module once the worker is handed a render command.
    """

    __slots__ = ("proc", "module_name", "warm")

    def __init__(self, proc: subprocess.Popen) -> None:
        self.proc = proc
        self.module_name: str | None = None
        self.warm = threading.Event()
_CAL_FACTOR_LABELS = {
    # Canonical calibration keys; must match scan_kit.common.schema dose columns.
    "ic1_total_dose": "IC1",
    "ic2_total_dose": "IC2",
    "ic3_total_dose": "IC3",
}


# Segmented control look: connected checkable buttons, accent-filled when selected.
# Uses palette() roles so it follows the active light/dark theme.
_SEGMENTED_QSS = """
_SegmentedControl QPushButton {
    border: 1px solid palette(mid);
    padding: 5px 14px;
    background: palette(button);
    color: palette(button-text);
}
_SegmentedControl QPushButton[seg="mid"],
_SegmentedControl QPushButton[seg="right"] {
    border-left: none;
}
_SegmentedControl QPushButton[seg="left"] {
    border-top-left-radius: 6px;
    border-bottom-left-radius: 6px;
}
_SegmentedControl QPushButton[seg="right"] {
    border-top-right-radius: 6px;
    border-bottom-right-radius: 6px;
}
_SegmentedControl QPushButton[seg="only"] {
    border-radius: 6px;
}
_SegmentedControl QPushButton:hover:!checked {
    background: palette(midlight);
}
_SegmentedControl QPushButton:checked {
    background: palette(highlight);
    color: palette(highlighted-text);
    border-color: palette(highlight);
}
"""

class _SegmentedControl(QWidget):
    """Compact horizontal selector: connected checkable buttons that act as one radio set.

    Build with ``[(key, label), …]``; emits :attr:`selectionChanged` with the chosen
    key on user interaction only. Use :meth:`set_current` to reflect external state
    without re-emitting.
    """

    selectionChanged = Signal(str)

    def __init__(
        self,
        options: list[tuple[str, str]],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._buttons: dict[str, QPushButton] = {}
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        n = len(options)
        for i, (key, label) in enumerate(options):
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
            if n == 1:
                seg = "only"
            elif i == 0:
                seg = "left"
            elif i == n - 1:
                seg = "right"
            else:
                seg = "mid"
            btn.setProperty("seg", seg)
            self._group.addButton(btn)
            self._buttons[key] = btn
            lay.addWidget(btn)

        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.setStyleSheet(_SEGMENTED_QSS)
        # buttonClicked fires only on user interaction, not programmatic setChecked.
        self._group.buttonClicked.connect(self._on_clicked)

    def _on_clicked(self, button: QPushButton) -> None:
        for key, btn in self._buttons.items():
            if btn is button:
                self.selectionChanged.emit(key)
                return

    def set_current(self, key: str) -> None:
        btn = self._buttons.get(key)
        if btn is not None:
            btn.setChecked(True)


class ScanKitMainWindow(QMainWindow):
    """Scan-kit analysis launcher (Qt)."""

    #: subprocess reported plot window ready (module_name, Popen instance for identity check)
    _sig_plot_window_ready = Signal(str, object)
    #: settings.json loaded off the GUI thread (bootstrap_generation, ViewSettings).
    _sig_settings_ready = Signal(int, object)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"Scan Kit v{__version__}")
        self.setMinimumSize(1050, 650)
        self._app_settings = AppSettings.load()
        self._restore_window_geometry()
        if FROZEN:
            self._initial_base_dir = str(Path.cwd())
        else:
            self._initial_base_dir = str(PROJECT_ROOT / "test_data")
        self._session_browser: SessionBrowserWidget | None = None
        self._worker_threads: list[threading.Thread] = []
        self._child_procs: list[subprocess.Popen] = []
        self._running_views: dict[str, tuple[subprocess.Popen, str]] = {}
        self._open_views: dict[str, subprocess.Popen] = {}
        self._launch_args: dict[str, tuple[list[str], str]] = {}
        self._warm_idle: list[_WarmWorker] = []
        self._spinner_frame: int = 0
        self._poll_timer: QTimer | None = None
        self._settings = ViewSettings()
        self._view_buttons: dict[str, QPushButton] = {}
        self._bootstrap_generation: int = 0
        self._report_generating = False
        self._report_thread: QThread | None = None
        self._report_worker: ReportGenerationWorker | None = None
        self._report_progress: QProgressDialog | None = None
        self._main_tabs: QTabWidget | None = None

        boot = QWidget()
        boot_l = QVBoxLayout(boot)
        _boot_msg = QLabel("Loading…")
        _boot_msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        boot_l.addWidget(_boot_msg, stretch=1)
        self.setCentralWidget(boot)

        QTimer.singleShot(0, self._deferred_finish_init)

    def _deferred_finish_init(self) -> None:
        """Build the full UI on the next event-loop pass so the window can paint first."""
        self._build_ui()
        self._connect_thread_signals()
        QTimer.singleShot(0, self._request_settings_then_scan)
        # Pre-warm view workers in the background so the first click is instant.
        QTimer.singleShot(0, self._refill_warm_pool)

    def _connect_thread_signals(self) -> None:
        self._sig_plot_window_ready.connect(
            self._mark_view_ready, Qt.ConnectionType.QueuedConnection
        )
        self._sig_settings_ready.connect(
            self._on_settings_ready, Qt.ConnectionType.QueuedConnection
        )

    @property
    def _base_dir(self) -> str:
        if self._session_browser is not None:
            return self._session_browser.base_dir()
        return self._initial_base_dir

    def _build_ui(self) -> None:
        tabs = QTabWidget()
        tabs.addTab(self._build_data_analysis_tab(), _MAIN_TAB_DATA_ANALYSIS)
        tabs.addTab(self._build_plan_synthesis_tab(), _MAIN_TAB_PLAN_SYNTHESIS)
        self._config_tuning_panel = ConfigTuningPanel(app_settings=self._app_settings)
        self._config_tuning_panel.set_session_data_dir(self._base_dir)
        tabs.addTab(self._config_tuning_panel, _MAIN_TAB_CONFIG_TUNING)
        self._main_tabs = tabs
        self._restore_main_tab()
        tabs.currentChanged.connect(self._on_main_tab_changed)
        self.setCentralWidget(tabs)

        for seq in ("Esc", "Ctrl+Q"):
            QShortcut(QKeySequence(seq), self, activated=self.close)

    def _restore_window_geometry(self) -> None:
        width = self._app_settings.window_width
        height = self._app_settings.window_height
        if (
            width is not None
            and height is not None
            and width >= self.minimumWidth()
            and height >= self.minimumHeight()
        ):
            self.resize(width, height)
        else:
            self.resize(_DEFAULT_WINDOW_WIDTH, _DEFAULT_WINDOW_HEIGHT)

        x = self._app_settings.window_x
        y = self._app_settings.window_y
        if x is not None and y is not None:
            self.move(x, y)

    def _remember_window_geometry(self) -> None:
        if self.isMaximized() or self.isFullScreen():
            return
        geo = self.geometry()
        self._app_settings.window_width = geo.width()
        self._app_settings.window_height = geo.height()
        self._app_settings.window_x = geo.x()
        self._app_settings.window_y = geo.y()

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._remember_window_geometry()

    def moveEvent(self, event: QMoveEvent) -> None:
        super().moveEvent(event)
        self._remember_window_geometry()

    def _restore_main_tab(self) -> None:
        tabs = self._main_tabs
        saved = self._app_settings.last_main_tab
        if tabs is None or not saved:
            return
        for i in range(tabs.count()):
            if tabs.tabText(i) == saved:
                tabs.setCurrentIndex(i)
                return

    def _persist_main_tab(self) -> None:
        tabs = self._main_tabs
        if tabs is None:
            return
        index = tabs.currentIndex()
        if index < 0:
            return
        self._app_settings.last_main_tab = tabs.tabText(index)
        try:
            self._app_settings.save()
        except Exception:
            pass

    def _on_main_tab_changed(self, _index: int) -> None:
        self._persist_main_tab()

    def _switch_to_main_tab(self, tab_name: str) -> None:
        tabs = self._main_tabs
        if tabs is None:
            return
        for i in range(tabs.count()):
            if tabs.tabText(i) == tab_name:
                tabs.setCurrentIndex(i)
                return

    def _populate_session_context_menu(self, sid: str, menu: QMenu) -> None:
        menu.addAction(
            "Open in Config Tuning…",
            lambda checked=False, session_id=sid: self._open_session_configuration(
                session_id
            ),
        )

    def _open_session_configuration(self, sid: str) -> None:
        config_dir = resolve_session_config_dir(sid, self._base_dir)
        if config_dir is None:
            self._notify(
                f"No on-disk configuration folder found for session {sid}.",
                error=True,
            )
            return

        panel = getattr(self, "_config_tuning_panel", None)
        if panel is None:
            return
        if not panel.open_config_root(config_dir, select_devices_xml=True):
            return
        self._switch_to_main_tab(_MAIN_TAB_CONFIG_TUNING)

    def _build_data_analysis_tab(self) -> QWidget:
        tab = QWidget()
        outer = QHBoxLayout(tab)
        outer.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(splitter)

        # --- Left panel ---
        self._session_browser = SessionBrowserWidget(
            project_root=PROJECT_ROOT,
            initial_base_dir=self._initial_base_dir,
            max_selections=MAX_SESSIONS,
            parent=self,
        )
        self._session_browser.set_selection_persistence(self._persist_selected_sessions)
        self._session_browser.base_dir_changed.connect(self._on_session_base_dir_changed)
        self._session_browser.populate_context_menu.connect(
            self._populate_session_context_menu,
        )
        left = self._session_browser

        # --- Right panel ---
        right = QWidget()
        right_outer = QVBoxLayout(right)
        right_outer.setContentsMargins(0, 0, 0, 0)

        right_scroll = make_pane_scroll_area()
        right_inner = QWidget()
        right_l = QVBoxLayout(right_inner)
        right_l.setContentsMargins(4, 4, 4, 4)
        right_l.setSpacing(8)

        self._bg_segmented = _SegmentedControl([("off", "Off"), ("on", "On")])
        self._bg_segmented.selectionChanged.connect(self._on_bg_segment_changed)
        self._cal_segmented = _SegmentedControl(
            [(mode, _CAL_MODE_LABELS[mode]) for mode in CALIBRATION_MODES]
        )
        self._cal_segmented.selectionChanged.connect(self._on_cal_segment_changed)

        settings_host = QWidget()
        settings_host.setSizePolicy(
            QSizePolicy.Policy.Minimum,
            QSizePolicy.Policy.Minimum,
        )
        settings_row = QHBoxLayout(settings_host)
        settings_row.setContentsMargins(0, 0, 0, 0)
        settings_row.setSpacing(8)
        bg_label = QLabel("BG Subtraction")
        bg_label.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        cal_label = QLabel("Calibration")
        cal_label.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        settings_row.addWidget(bg_label)
        settings_row.addWidget(self._bg_segmented)
        settings_row.addSpacing(24)
        settings_row.addWidget(cal_label)
        settings_row.addWidget(self._cal_segmented)
        settings_row.addStretch(1)
        right_l.addWidget(settings_host)

        self.cal_factors_label = QLabel("")
        self.cal_factors_label.setWordWrap(True)
        self.cal_factors_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Minimum,
        )
        self.cal_factors_label.hide()
        right_l.addWidget(self.cal_factors_label)

        for group_title, entries in VIEW_GROUPS:
            view_box = QGroupBox(group_title)
            view_box.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Minimum,
            )
            view_box.setMinimumWidth(0)
            view_inner = QVBoxLayout(view_box)
            view_inner.setContentsMargins(8, 8, 8, 8)
            grid = QGridLayout()
            grid.setHorizontalSpacing(10)
            grid.setVerticalSpacing(8)
            for col in range(_VIEW_GRID_COLS):
                grid.setColumnStretch(col, 1)
            for i, (display_name, module_name, description) in enumerate(entries):
                btn = QPushButton(display_name)
                btn.setToolTip(description)
                btn.setMinimumHeight(30)
                btn.setSizePolicy(
                    QSizePolicy.Policy.Expanding,
                    QSizePolicy.Policy.Fixed,
                )
                btn.clicked.connect(partial(self._on_view_clicked, module_name))
                self._view_buttons[module_name] = btn
                row, col = divmod(i, _VIEW_GRID_COLS)
                grid.addWidget(btn, row, col)
            view_inner.addLayout(grid)
            right_l.addWidget(view_box)

        right_l.addStretch(1)

        set_pane_scroll_widget(right_scroll, right_inner)
        right_outer.addWidget(right_scroll, stretch=1)

        report_row = QHBoxLayout()
        report_row.setContentsMargins(4, 4, 4, 4)
        self._report_btn = QPushButton("Generate Report…")
        self._report_btn.setAutoDefault(True)
        self._report_btn.setDefault(True)
        self._report_btn.setToolTip(
            "Build a PDF report from selected sessions and analysis views"
        )
        self._report_btn.clicked.connect(self._on_generate_report)
        report_row.addStretch(1)
        report_row.addWidget(self._report_btn)
        right_outer.addLayout(report_row)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([640, 760])
        return tab

    def _build_plan_synthesis_tab(self) -> QWidget:
        return PlanSynthesisPanel(app_settings=self._app_settings)

    def _track_worker(self, thread: threading.Thread) -> None:
        self._worker_threads = [t for t in self._worker_threads if t.is_alive()]
        self._worker_threads.append(thread)
        thread.start()

    def _request_settings_then_scan(self) -> None:
        """Load settings.json on a worker thread, then start session discovery."""
        self._bootstrap_generation += 1
        gen = self._bootstrap_generation
        base_dir = self._base_dir
        self._track_worker(
            threading.Thread(
                target=self._settings_io_worker,
                args=(gen, base_dir),
                daemon=True,
            )
        )

    def _settings_io_worker(self, bootstrap_gen: int, base_dir: str) -> None:
        try:
            settings = ViewSettings.load(base_dir)
        except Exception:
            settings = ViewSettings()
        self._sig_settings_ready.emit(bootstrap_gen, settings)

    @Slot(int, object)
    def _on_settings_ready(self, bootstrap_gen: int, settings_obj: object) -> None:
        if bootstrap_gen != self._bootstrap_generation:
            return
        if isinstance(settings_obj, ViewSettings):
            self._settings = settings_obj
        self._sync_bg_buttons()
        self._sync_cal_buttons()
        self._update_cal_factors_display()
        self._refresh_sessions()

    def _on_session_base_dir_changed(self, path: str) -> None:
        panel = getattr(self, "_config_tuning_panel", None)
        if panel is not None:
            panel.set_session_data_dir(path)
        self._request_settings_then_scan()

    def _on_bg_segment_changed(self, key: str) -> None:
        self._set_bg_subtract(key == "on")

    def _on_cal_segment_changed(self, key: str) -> None:
        self._set_calibration_mode(key)

    def _sync_bg_buttons(self) -> None:
        self._bg_segmented.set_current("on" if self._settings.bg_subtract else "off")

    def _set_bg_subtract(self, on: bool) -> None:
        self._settings.bg_subtract = on
        self._settings.save(self._base_dir)
        self._sync_bg_buttons()

    def _sync_cal_buttons(self) -> None:
        self._cal_segmented.set_current(self._settings.calibration_mode)

    def _set_calibration_mode(self, mode: str) -> None:
        self._settings.calibration_mode = mode
        self._settings.cal_factors = None
        self._settings.save(self._base_dir)
        self._sync_cal_buttons()
        self._update_cal_factors_display()

    def _update_cal_factors_display(self) -> None:
        factors = self._settings.cal_factors
        if not factors or self._settings.calibration_mode == "off":
            self.cal_factors_label.setText("")
            self.cal_factors_label.hide()
            return
        parts = []
        for col, label in _CAL_FACTOR_LABELS.items():
            if col in factors:
                parts.append(f"{label}: {factors[col]:.4f}")
        text = "  ".join(parts) if parts else ""
        self.cal_factors_label.setText(text)
        if text:
            self.cal_factors_label.show()
        else:
            self.cal_factors_label.hide()

    def _refresh_sessions(self) -> None:
        if self._session_browser is None:
            return
        self._session_browser.refresh(
            restored_selection=list(self._settings.selected_sessions or [])[:MAX_SESSIONS],
        )

    def _selected_sids_in_order(self) -> list[str]:
        if self._session_browser is None:
            return []
        return self._session_browser.selected_session_ids()

    def _persist_selected_sessions(self, session_ids: list[str] | None = None) -> None:
        """Save the current session selection into the persistent settings file."""
        if session_ids is None:
            session_ids = self._selected_sids_in_order()
        self._settings.selected_sessions = session_ids
        try:
            self._settings.save(self._base_dir)
        except Exception:
            pass

    def _session_meta_by_sid(self) -> dict[str, SessionMeta | None]:
        if self._session_browser is None:
            return {}
        return self._session_browser.session_meta_by_id()

    def _notify(self, message: str, *, error: bool = False) -> None:
        box = QMessageBox(self)
        box.setText(message)
        box.setIcon(QMessageBox.Icon.Critical if error else QMessageBox.Icon.Warning)
        box.exec()

    def _on_generate_report(self) -> None:
        if self._report_generating:
            self._notify("Report generation already in progress")
            return

        session_ids = self._selected_sids_in_order()
        if not session_ids:
            self._notify(
                f"Select 1-{MAX_SESSIONS} sessions first (use the checkboxes)"
            )
            return

        reportable = reportable_module_names()
        saved_views = [
            module_name
            for module_name in self._app_settings.last_report_views
            if module_name in reportable
        ]

        wizard = ReportWizardDialog(
            session_ids=session_ids,
            base_dir=self._base_dir,
            settings=self._settings,
            session_meta=self._session_meta_by_sid(),
            notes=self._session_browser.notes() if self._session_browser else {},
            last_report_dir=self._app_settings.last_report_dir,
            last_report_author=self._app_settings.last_report_author,
            last_report_organization=self._app_settings.last_report_organization,
            last_report_views=saved_views,
            parent=self,
        )
        if wizard.exec() != QDialog.DialogCode.Accepted:
            return

        config = wizard.config
        if config is None:
            return

        self._app_settings.last_report_views = [
            entry[1] for entry in config.views
        ]
        self._app_settings.last_report_author = config.author or None
        self._app_settings.last_report_organization = config.organization or None
        try:
            self._app_settings.save()
        except Exception:
            pass

        self._report_generating = True
        self._report_btn.setEnabled(False)

        progress = QProgressDialog("Preparing report…", None, 0, 100, self)
        progress.setWindowTitle("Generate Report")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(True)
        progress.setAutoReset(True)
        progress.setCancelButton(None)
        progress.setValue(0)
        self._report_progress = progress

        thread = QThread(self)
        worker = ReportGenerationWorker(config)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_report_progress)
        worker.finished.connect(self._on_report_finished)
        worker.failed.connect(self._on_report_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(progress.close)
        worker.failed.connect(progress.close)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_report_thread)
        self._report_thread = thread
        self._report_worker = worker
        thread.start()
        progress.show()

    def _on_report_progress(self, value: int, message: str) -> None:
        if self._report_progress is not None:
            self._report_progress.setLabelText(message)
            self._report_progress.setValue(value)

    def _release_report_state(self) -> None:
        self._report_generating = False
        self._report_btn.setEnabled(True)
        self._report_progress = None

    def _on_report_finished(self, output_path: str) -> None:
        self._release_report_state()
        try:
            self._app_settings.last_report_dir = str(Path(output_path).parent)
            self._app_settings.save()
        except Exception:
            pass
        box = QMessageBox(self)
        box.setWindowTitle("Report complete")
        box.setIcon(QMessageBox.Icon.Information)
        box.setText("PDF report saved successfully.")
        box.setInformativeText(output_path)
        open_btn = box.addButton("Open Report", QMessageBox.ButtonRole.ActionRole)
        box.addButton(QMessageBox.StandardButton.Ok)
        box.exec()
        if box.clickedButton() is open_btn:
            opened = QDesktopServices.openUrl(
                QUrl.fromLocalFile(str(Path(output_path).resolve()))
            )
            if not opened:
                QMessageBox.warning(
                    self,
                    "Could not open report",
                    f"No application is available to open:\n{output_path}",
                )

    def _on_report_failed(self, message: str) -> None:
        self._release_report_state()
        QMessageBox.critical(self, "Report failed", message)

    def _clear_report_thread(self) -> None:
        self._report_thread = None
        self._report_worker = None

    def _on_view_clicked(self, module_name: str) -> None:
        if module_name in self._running_views:
            self._notify("Already running")
            return

        session_ids = self._selected_sids_in_order()
        if not session_ids:
            self._notify(
                f"Select 1-{MAX_SESSIONS} sessions first (use the checkboxes)"
            )
            return

        self._launch_view(module_name, session_ids, self._base_dir)

    def _launch_view(
        self, module_name: str, session_ids: list[str], base_dir: str,
    ) -> None:
        if module_name in self._open_views:
            try:
                self._open_views.pop(module_name).kill()
            except Exception:
                pass
        if module_name in self._running_views:
            try:
                self._running_views.pop(module_name)[0].kill()
            except Exception:
                pass

        self._launch_args[module_name] = (session_ids, base_dir)

        if self._settings.calibration_mode == "constrained":
            from .common.processing import compute_calibration_factors
            factors = compute_calibration_factors(session_ids, base_dir)
            self._settings.cal_factors = factors if factors else None
            self._update_cal_factors_display()
        else:
            self._settings.cal_factors = None

        settings_json = self._settings.to_json()

        # Fast path: hand the command to an already-warm pooled worker. If none
        # is ready (or the handoff fails), fall back to the proven direct launch
        # so behavior is never worse than spawning a fresh process per click.
        proc: subprocess.Popen | None = None
        worker = self._take_ready_worker(module_name)
        if worker is not None and worker.proc.stdin is not None:
            command = json.dumps(
                {
                    "module": module_name,
                    "sessions": session_ids,
                    "base_dir": base_dir,
                    "settings": settings_json,
                }
            )
            try:
                worker.proc.stdin.write((command + "\n").encode())
                worker.proc.stdin.flush()
                proc = worker.proc
            except Exception:
                proc = None

        if proc is None:
            proc = self._spawn_direct_view(
                module_name, session_ids, base_dir, settings_json
            )
            if proc is None:
                self._notify("Failed to run analysis", error=True)
                return
            threading.Thread(
                target=self._watch_subprocess_ready,
                args=(proc, module_name),
                daemon=True,
            ).start()

        self._child_procs.append(proc)
        self._reap_children()

        original_label = next(
            (name for name, mod, _desc in VIEWS if mod == module_name),
            module_name,
        )
        self._running_views[module_name] = (proc, original_label)
        btn = self._view_buttons.get(module_name)
        if btn is not None:
            btn.setText(original_label)
        self._update_spinner()
        if self._poll_timer is None:
            self._poll_timer = QTimer(self)
            self._poll_timer.timeout.connect(self._poll_running_views)
            self._poll_timer.start(120)

    def _update_spinner(self) -> None:
        frame = _SPINNER[self._spinner_frame % len(_SPINNER)]
        for module_name, (_proc, original_label) in self._running_views.items():
            btn = self._view_buttons.get(module_name)
            if btn is not None:
                btn.setText(f"{frame} {original_label}")

    def _warm_worker_popen(self) -> subprocess.Popen:
        """Spawn a worker process that preloads the heavy stack and waits on stdin."""
        env = os.environ.copy()
        if FROZEN:
            cmd = [sys.executable, "--warm-worker"]
        else:
            code = (
                "from scan_kit.common.view_runner import warm_worker_main;"
                " warm_worker_main()"
            )
            env["PYTHONPATH"] = str(PROJECT_ROOT) + (
                os.pathsep + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else ""
            )
            cmd = [sys.executable, "-c", code]

        popen_kwargs: dict[str, Any] = dict(
            cwd=PROJECT_ROOT,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if sys.platform == "win32":
            popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        return subprocess.Popen(cmd, **popen_kwargs)

    def _spawn_warm_worker(self) -> _WarmWorker | None:
        try:
            proc = self._warm_worker_popen()
        except Exception as e:
            self._notify(f"Failed to start view worker: {e}", error=True)
            return None
        worker = _WarmWorker(proc)
        self._warm_idle.append(worker)
        threading.Thread(
            target=self._watch_worker, args=(worker,), daemon=True
        ).start()
        return worker

    def _refill_warm_pool(self) -> None:
        """Top up the idle pool to ``_WARM_POOL_SIZE``, pruning any dead workers."""
        self._warm_idle = [w for w in self._warm_idle if w.proc.poll() is None]
        while len(self._warm_idle) < _WARM_POOL_SIZE:
            if self._spawn_warm_worker() is None:
                break

    def _take_ready_worker(self, module_name: str) -> _WarmWorker | None:
        """Claim an already-warm idle worker, or ``None`` if none is ready yet.

        Only fully-warmed workers are returned so the click is genuinely instant;
        callers fall back to a direct launch otherwise. The pool is topped up
        regardless so subsequent clicks find a warm worker.
        """
        self._warm_idle = [w for w in self._warm_idle if w.proc.poll() is None]
        worker: _WarmWorker | None = None
        for i, candidate in enumerate(self._warm_idle):
            if candidate.warm.is_set():
                worker = self._warm_idle.pop(i)
                break
        if worker is not None:
            worker.module_name = module_name
        self._refill_warm_pool()
        return worker

    def _spawn_direct_view(
        self,
        module_name: str,
        session_ids: list[str],
        base_dir: str,
        settings_json: str,
    ) -> subprocess.Popen | None:
        """Original per-click launch: spawn a fresh process to run a single view."""
        env = os.environ.copy()
        if FROZEN:
            cmd = [
                sys.executable,
                "--run-view", module_name,
                "--sessions", ",".join(session_ids),
                "--base-dir", base_dir,
                "--settings", settings_json,
            ]
        else:
            code = (
                "from scan_kit.common.view_runner import run_with_live_settings;"
                f"from scan_kit.views.{module_name} import run;"
                f"run_with_live_settings(run, {session_ids!r}, {base_dir!r}, {settings_json!r})"
            )
            env["PYTHONPATH"] = str(PROJECT_ROOT) + (
                os.pathsep + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else ""
            )
            cmd = [sys.executable, "-c", code]

        popen_kwargs: dict[str, Any] = dict(
            cwd=PROJECT_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if sys.platform == "win32":
            popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        try:
            return subprocess.Popen(cmd, **popen_kwargs)
        except Exception as e:
            self._notify(f"Failed to run analysis: {e}", error=True)
            return None

    def _watch_subprocess_ready(
        self, proc: subprocess.Popen, module_name: str
    ) -> None:
        try:
            for line in proc.stdout:
                if _READY_SENTINEL.encode() in line:
                    self._sig_plot_window_ready.emit(module_name, proc)
                    break
        except Exception:
            pass
        try:
            for _ in proc.stdout:
                pass
        except Exception:
            pass

    def _watch_worker(self, worker: _WarmWorker) -> None:
        """Read a worker's stdout for the warm and plot-ready sentinels (and drain)."""
        stdout = worker.proc.stdout
        if stdout is None:
            return
        try:
            for line in stdout:
                if _WORKER_WARM_SENTINEL.encode() in line:
                    worker.warm.set()
                    continue
                if worker.module_name and _READY_SENTINEL.encode() in line:
                    self._sig_plot_window_ready.emit(worker.module_name, worker.proc)
        except Exception:
            pass

    @Slot(str, object)
    def _mark_view_ready(self, module_name: str, proc: object) -> None:
        if not isinstance(proc, subprocess.Popen):
            return
        if module_name not in self._running_views:
            return
        running_proc, original_label = self._running_views[module_name]
        if running_proc is not proc:
            return
        self._running_views.pop(module_name)
        self._open_views[module_name] = proc
        btn = self._view_buttons.get(module_name)
        if btn is not None:
            btn.setText(original_label)

    def _poll_running_views(self) -> None:
        self._spinner_frame += 1
        finished = [
            name for name, (proc, _) in self._running_views.items()
            if proc.poll() is not None
        ]
        for module_name in finished:
            proc, original_label = self._running_views.pop(module_name)
            btn = self._view_buttons.get(module_name)
            if btn is not None:
                btn.setText(original_label)
            if proc.returncode != 0:
                detail = ""
                if proc.stderr:
                    try:
                        raw = proc.stderr.read()
                        lines = raw.decode(errors="replace").strip().splitlines()
                        detail = f": {lines[-1]}" if lines else ""
                    except Exception:
                        pass
                self._notify(f"{module_name} failed{detail}", error=True)

        closed = [
            name for name, proc in self._open_views.items()
            if proc.poll() is not None
        ]
        for module_name in closed:
            self._open_views.pop(module_name)
            self._launch_args.pop(module_name, None)

        if self._running_views:
            self._update_spinner()
        self._maybe_stop_timers()
        self._reap_children()

    def _maybe_stop_timers(self) -> None:
        if not self._running_views and not self._open_views:
            if self._poll_timer is not None:
                self._poll_timer.stop()
                self._poll_timer = None

    def _reap_children(self) -> None:
        self._child_procs = [p for p in self._child_procs if p.poll() is None]

    def closeEvent(self, event: QCloseEvent) -> None:
        panel = getattr(self, "_config_tuning_panel", None)
        if panel is not None and not panel.confirm_discard_if_dirty():
            event.ignore()
            return
        self._remember_window_geometry()
        self._persist_main_tab()
        self._shutdown_children()
        super().closeEvent(event)

    def _shutdown_children(self) -> None:
        if self._session_browser is not None:
            self._session_browser.shutdown()
        panel = getattr(self, "_config_tuning_panel", None)
        if panel is not None:
            panel.shutdown()
        for t in self._worker_threads:
            if t.is_alive():
                t.join(timeout=2.0)
        self._worker_threads.clear()
        if self._poll_timer is not None:
            self._poll_timer.stop()
            self._poll_timer = None
        self._running_views.clear()
        self._open_views.clear()
        self._launch_args.clear()
        for worker in self._warm_idle:
            try:
                worker.proc.kill()
            except Exception:
                pass
        self._warm_idle.clear()
        for proc in self._child_procs:
            try:
                proc.kill()
            except Exception:
                pass
        self._child_procs.clear()


def _run_view_subprocess(
    module_name: str,
    session_ids: list[str],
    base_dir: str,
    settings: ViewSettings | None = None,
) -> None:
    """Execute a single analysis view (used by frozen exe in --run-view mode)."""
    import importlib
    from .common.view_runner import run_with_live_settings

    mod = importlib.import_module(f"scan_kit.views.{module_name}")
    if settings is None:
        settings = ViewSettings()
    run_with_live_settings(mod.run, session_ids, base_dir, settings.to_json())


def main() -> None:
    """Entry point for the scan-kit GUI."""
    multiprocessing.freeze_support()

    if "--version" in sys.argv or "-V" in sys.argv:
        print(f"scan-kit {__version__}")
        return

    if "--warm-worker" in sys.argv:
        from .common.view_runner import warm_worker_main
        warm_worker_main()
        return

    if "--run-view" in sys.argv:
        idx = sys.argv.index("--run-view")
        module_name = sys.argv[idx + 1]
        sessions_idx = sys.argv.index("--sessions")
        session_ids = sys.argv[sessions_idx + 1].split(",")
        base_idx = sys.argv.index("--base-dir")
        base_dir = sys.argv[base_idx + 1]
        settings = None
        if "--settings" in sys.argv:
            settings_idx = sys.argv.index("--settings")
            settings = ViewSettings.from_json(sys.argv[settings_idx + 1])
        _run_view_subprocess(module_name, session_ids, base_dir, settings=settings)
        return

    app = QApplication(sys.argv)
    app_icon = load_app_icon()
    if not app_icon.isNull():
        app.setWindowIcon(app_icon)
    win = ScanKitMainWindow()
    if not app_icon.isNull():
        win.setWindowIcon(app_icon)

    def _force_exit(*_args) -> None:
        win._shutdown_children()
        inst = QApplication.instance()
        if inst is not None:
            inst.quit()
        os._exit(0)

    # Ensure worker threads are joined before QApplication is destroyed (avoids
    # QThreadStorage warnings on quit paths that don't go through closeEvent).
    app.aboutToQuit.connect(win._shutdown_children)

    signal.signal(signal.SIGINT, _force_exit)
    atexit.register(_force_exit)

    # Allow SIGINT (Ctrl+C) to interrupt when run from a terminal.
    if hasattr(signal, "SIGINT"):
        _sig_timer = QTimer(app)
        _sig_timer.start(200)
        _sig_timer.timeout.connect(lambda: None)

    win.show()
    QApplication.processEvents()
    apply_windows_window_icons(win, app_icon)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
