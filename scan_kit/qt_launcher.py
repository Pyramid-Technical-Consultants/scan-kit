"""Qt GUI launcher for scan-kit analysis views."""

from __future__ import annotations

import atexit
import multiprocessing
import os
import signal
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

from PySide6.QtCore import QRectF, Qt, QTimer, Signal, Slot, QSize
from PySide6.QtGui import (
    QCloseEvent,
    QColor,
    QIcon,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
    QShortcut,
)
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from . import __version__
from .common.session_meta import SessionMeta
from .common.session_notes import load_notes, save_note
from .common.settings import ViewSettings, CALIBRATION_MODES
from .common.plot_colors import DEFAULT_SESSION_COLORS
from .views import VIEW_GROUPS, VIEWS

MAX_SESSIONS = 5
FROZEN = getattr(sys, "frozen", False)

if FROZEN:
    PROJECT_ROOT = Path(sys.executable).resolve().parent
else:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Analysis launcher button grid (see VIEW_GROUPS for workflow sections).
_VIEW_GRID_COLS = 2

# Batched session rows emitted from discovery thread (keep small so each UI slot stays short).
_SESSION_ROW_BATCH = 24

_CAL_MODE_LABELS = {
    "off": "Off",
    "per_session": "Per-Session",
    "constrained": "Constrained",
}

_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_READY_SENTINEL = "__SCAN_KIT_PLOT_READY__"
_SESSION_ROLE = Qt.ItemDataRole.UserRole
_SWATCH_STATE_ROLE = Qt.ItemDataRole.UserRole + 1
#: Real (datetime/float/int) value used for chronological/numeric column sorting.
_SORT_VALUE_ROLE = Qt.ItemDataRole.UserRole + 2

# session_table column indices
_COL_USE = 0
_COL_SESSION_ID = 1
_COL_DATE = 2
_COL_MU = 3
_COL_TIME = 4
_COL_NOTE = 5

_SWATCH_PX = 14
_UNCHECKED_SWATCH = QColor("#d0d0d0")
_SWATCH_LINE = QColor("#6a6a6a")


def _session_plot_swatch_icon(*, active: bool, plot_index: int | None) -> QIcon:
    """Square swatch matching visualization session colors (inactive = neutral gray)."""
    if active and plot_index is not None:
        name = DEFAULT_SESSION_COLORS[plot_index % len(DEFAULT_SESSION_COLORS)]
        fill = QColor(name)
    else:
        fill = _UNCHECKED_SWATCH
    pix = QPixmap(_SWATCH_PX, _SWATCH_PX)
    pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setBrush(fill)
    p.setPen(QPen(_SWATCH_LINE, 1))
    inset = 1.0
    r = float(_SWATCH_PX) - 2 * inset
    p.drawRoundedRect(
        QRectF(inset, inset, r, r),
        2.5,
        2.5,
    )
    p.end()
    return QIcon(pix)


_SWATCH_ICON_CACHE: dict[tuple[str, int | None], QIcon] = {}


def _cached_swatch_icon(*, active: bool, plot_index: int | None) -> QIcon:
    """Reuse icons; painting swatches for every row on each click is expensive."""
    if active and plot_index is not None:
        key = ("on", plot_index % len(DEFAULT_SESSION_COLORS))
    else:
        key = ("off", None)
    icon = _SWATCH_ICON_CACHE.get(key)
    if icon is None:
        icon = _session_plot_swatch_icon(
            active=bool(active and plot_index is not None),
            plot_index=plot_index,
        )
        _SWATCH_ICON_CACHE[key] = icon
    return icon

_CAL_FACTOR_LABELS = {
    # Canonical calibration keys; must match scan_kit.common.schema dose columns.
    "ic1_total_dose": "IC1",
    "ic2_total_dose": "IC2",
    "ic3_total_dose": "IC3",
}


class _SortableItem(QTableWidgetItem):
    """Table item that sorts on the value stored in :data:`_SORT_VALUE_ROLE`.

    Lets the Date / MU / Time columns sort chronologically/numerically while the
    cell still displays a formatted string (and a ``—``/``?`` placeholder when the
    metadata is missing). Missing values (``None``) always sort as the smallest, so
    they land at the bottom under the default descending-by-date order.
    """

    def __lt__(self, other: QTableWidgetItem) -> bool:
        a = self.data(_SORT_VALUE_ROLE)
        b = other.data(_SORT_VALUE_ROLE) if isinstance(other, QTableWidgetItem) else None
        if a is None or b is None:
            return a is None and b is not None
        try:
            return a < b
        except TypeError:
            return str(a) < str(b)


def _meta_column_texts(meta: SessionMeta | None) -> tuple[str, str, str]:
    """Display strings for Date / MU / Time table columns."""
    if meta is None:
        return "—", "—", "—"
    return meta.short_date, meta.short_mu, meta.short_time


def _meta_sort_values(
    meta: SessionMeta | None,
) -> tuple[datetime | None, float | None, int | None]:
    """Real sort values for the Date / MU / Time columns (``None`` when unknown)."""
    if meta is None:
        return (None, None, None)
    return (meta.date, meta.primary_mu, meta.treatment_time_s)


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
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
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

    #: (hydrate_generation, session_id) — emitted from worker threads; handled on GUI thread.
    _sig_session_extracting = Signal(int, str)
    #: (hydrate_generation, session_id, path_str, meta_or_none)
    _sig_session_metadata = Signal(int, str, str, object)
    #: subprocess reported plot window ready (module_name, Popen instance for identity check)
    _sig_plot_window_ready = Signal(str, object)
    #: Notes JSON loaded on worker; GUI applies before session rows stream in.
    _sig_notes_loaded = Signal(int, object)
    #: Chunk of (session_id, path_str) rows discovered for this generation.
    _sig_session_rows_batch = Signal(int, object)
    #: Folder scan and row batch emits are done for this generation.
    _sig_scan_finished = Signal(int)
    #: Full folder listing for an incremental rescan (hydrate_generation, rows).
    _sig_incremental_rescan = Signal(int, object)
    #: settings.json loaded off the GUI thread (bootstrap_generation, ViewSettings).
    _sig_settings_ready = Signal(int, object)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"scan-kit v{__version__}")
        self.resize(1400, 880)
        self.setMinimumSize(1050, 650)
        if FROZEN:
            self._base_dir = str(Path.cwd())
        else:
            self._base_dir = str(PROJECT_ROOT / "test_data")
        self._sessions: list[str] = []
        self._discovered: list[tuple[str, str, SessionMeta | None]] = []
        self._hydrate_generation: int = 0
        self._hydrate_received: int = 0
        self._scan_complete: bool = False
        self._meta_pool: ThreadPoolExecutor | None = None
        self._worker_threads: list[threading.Thread] = []
        self._child_procs: list[subprocess.Popen] = []
        self._running_views: dict[str, tuple[subprocess.Popen, str]] = {}
        self._open_views: dict[str, subprocess.Popen] = {}
        self._launch_args: dict[str, tuple[list[str], str]] = {}
        self._spinner_frame: int = 0
        self._poll_timer: QTimer | None = None
        self._notes: dict[str, str] = {}
        self._highlighted_sid: str | None = None
        self._settings = ViewSettings()
        self._check_order: list[str] = []
        self._view_buttons: dict[str, QPushButton] = {}
        self._bootstrap_generation: int = 0
        self._meta_col_resize_timer: QTimer | None = None
        self._status_refresh_timer: QTimer | None = None
        self._row_by_sid: dict[str, int] = {}

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
        self._meta_col_resize_timer = QTimer(self)
        self._meta_col_resize_timer.setSingleShot(True)
        self._meta_col_resize_timer.timeout.connect(self._resize_session_meta_columns)
        self._status_refresh_timer = QTimer(self)
        self._status_refresh_timer.setSingleShot(True)
        self._status_refresh_timer.timeout.connect(self._update_status)
        self._connect_thread_signals()
        QTimer.singleShot(0, self._request_settings_then_scan)

    def _connect_thread_signals(self) -> None:
        self._sig_session_extracting.connect(
            self._show_extracting_status, Qt.ConnectionType.QueuedConnection
        )
        self._sig_session_metadata.connect(
            self._on_session_metadata_ready, Qt.ConnectionType.QueuedConnection
        )
        self._sig_plot_window_ready.connect(
            self._mark_view_ready, Qt.ConnectionType.QueuedConnection
        )
        self._sig_notes_loaded.connect(
            self._on_notes_loaded, Qt.ConnectionType.QueuedConnection
        )
        self._sig_session_rows_batch.connect(
            self._on_session_rows_batch, Qt.ConnectionType.QueuedConnection
        )
        self._sig_scan_finished.connect(
            self._on_scan_finished, Qt.ConnectionType.QueuedConnection
        )
        self._sig_incremental_rescan.connect(
            self._on_incremental_rescan, Qt.ConnectionType.QueuedConnection
        )
        self._sig_settings_ready.connect(
            self._on_settings_ready, Qt.ConnectionType.QueuedConnection
        )

    def _schedule_meta_column_resize(self) -> None:
        if self._meta_col_resize_timer is not None:
            self._meta_col_resize_timer.start(60)

    def _schedule_status_refresh(self) -> None:
        if self._status_refresh_timer is not None:
            self._status_refresh_timer.start(0)
        else:
            self._update_status()

    def _rebuild_session_row_index(self) -> None:
        self._row_by_sid.clear()
        for r in range(self.session_table.rowCount()):
            sid = self._session_row_sid(r)
            if sid is not None:
                self._row_by_sid[sid] = r

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        outer = QHBoxLayout(central)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(splitter)

        # --- Left panel ---
        left = QWidget()
        left_l = QVBoxLayout(left)
        left_l.setContentsMargins(4, 4, 4, 4)

        data_dir_row = QHBoxLayout()
        clear_btn = QPushButton("✕")
        clear_btn.setFixedWidth(28)
        clear_btn.setToolTip("Clear session selection")
        clear_btn.clicked.connect(self._clear_selection)
        data_dir_row.addWidget(clear_btn)
        self.base_dir_input = QLineEdit()
        self.base_dir_input.setPlaceholderText("Path to session ZIPs…")
        self.base_dir_input.setText(self._base_dir)
        self.base_dir_input.editingFinished.connect(self._on_base_dir_finished)
        self.base_dir_input.returnPressed.connect(self._on_base_dir_finished)
        data_dir_row.addWidget(self.base_dir_input, stretch=1)
        browse_dir_btn = QPushButton("Browse…")
        browse_dir_btn.setToolTip("Choose folder containing session archives or folders")
        browse_dir_btn.setFixedWidth(96)
        browse_dir_btn.clicked.connect(self._on_browse_data_dir)
        data_dir_row.addWidget(browse_dir_btn)
        refresh_btn = QPushButton("↻")
        refresh_btn.setFixedWidth(28)
        refresh_btn.setToolTip("Refresh session list from folder")
        refresh_btn.clicked.connect(self._incremental_refresh_sessions)
        data_dir_row.addWidget(refresh_btn)
        left_l.addLayout(data_dir_row)

        self.session_table = QTableWidget()
        self.session_table.setColumnCount(6)
        self.session_table.setHorizontalHeaderLabels(
            ["Use", "Session ID", "Date", "MU", "Time (s)", "Note"]
        )
        self.session_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.session_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        # Native column sorting: click a header to sort; rows stay intact because every
        # cell is a QTableWidgetItem (no setCellWidget). Date/MU/Time use _SortableItem
        # so they order by real value rather than the formatted display string.
        self.session_table.setSortingEnabled(True)
        self.session_table.setAlternatingRowColors(True)
        # Avoid "..." clipping in cells; use horizontal scroll when row is wider than the view.
        self.session_table.setTextElideMode(Qt.TextElideMode.ElideNone)
        self.session_table.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        hh = self.session_table.horizontalHeader()
        hh.setMinimumSectionSize(72)
        hh.setSectionResizeMode(_COL_USE, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(_COL_SESSION_ID, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(_COL_DATE, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(_COL_MU, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(_COL_TIME, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(_COL_NOTE, QHeaderView.ResizeMode.Stretch)
        hh.resizeSection(_COL_SESSION_ID, 220)
        hh.setSortIndicatorShown(True)
        # Default ordering: newest session first.
        self.session_table.sortByColumn(_COL_DATE, Qt.SortOrder.DescendingOrder)
        self.session_table.verticalHeader().setVisible(False)
        self.session_table.setIconSize(QSize(_SWATCH_PX, _SWATCH_PX))
        self.session_table.itemChanged.connect(self._on_session_table_item_changed)
        self.session_table.currentCellChanged.connect(self._on_session_current_cell_changed)
        left_l.addWidget(self.session_table, stretch=1)

        # --- Right panel ---
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_inner = QWidget()
        right_l = QVBoxLayout(right_inner)
        right_l.setContentsMargins(4, 4, 4, 4)

        right_l.addWidget(QLabel("SETTINGS"))

        self._bg_segmented = _SegmentedControl([("off", "Off"), ("on", "On")])
        self._bg_segmented.selectionChanged.connect(self._on_bg_segment_changed)
        self._cal_segmented = _SegmentedControl(
            [(mode, _CAL_MODE_LABELS[mode]) for mode in CALIBRATION_MODES]
        )
        self._cal_segmented.selectionChanged.connect(self._on_cal_segment_changed)

        settings_row = QHBoxLayout()
        settings_row.addWidget(QLabel("BG Subtraction"))
        settings_row.addWidget(self._bg_segmented)
        settings_row.addSpacing(16)
        settings_row.addWidget(QLabel("Calibration"))
        settings_row.addWidget(self._cal_segmented, stretch=1)
        right_l.addLayout(settings_row)

        self.cal_factors_label = QLabel("")
        self.cal_factors_label.setWordWrap(True)
        right_l.addWidget(self.cal_factors_label)

        right_l.addWidget(QLabel("RUN ANALYSIS"))
        for group_title, entries in VIEW_GROUPS:
            view_box = QGroupBox(group_title)
            view_inner = QVBoxLayout(view_box)
            grid = QGridLayout()
            grid.setHorizontalSpacing(8)
            grid.setVerticalSpacing(6)
            for col in range(_VIEW_GRID_COLS):
                grid.setColumnStretch(col, 1)
            for i, (display_name, module_name) in enumerate(entries):
                btn = QPushButton(display_name)
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

        right_scroll.setWidget(right_inner)
        splitter.addWidget(left)
        splitter.addWidget(right_scroll)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([640, 760])

        version_label = QLabel(f"scan-kit v{__version__}")
        self.statusBar().addPermanentWidget(version_label)

        for seq in ("Esc", "Ctrl+Q"):
            QShortcut(QKeySequence(seq), self, activated=self.close)

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

    def _on_base_dir_finished(self) -> None:
        path = self.base_dir_input.text().strip()
        if path:
            self._base_dir = path
            self._request_settings_then_scan()

    def _on_browse_data_dir(self) -> None:
        start = self.base_dir_input.text().strip() or self._base_dir
        p = Path(start).expanduser()
        initial = str(p.resolve()) if p.is_dir() else str(Path.home())

        chosen = QFileDialog.getExistingDirectory(
            self,
            "Select session data folder",
            initial,
        )
        if not chosen:
            return
        self.base_dir_input.setText(chosen)
        self._on_base_dir_finished()

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
            return
        parts = []
        for col, label in _CAL_FACTOR_LABELS.items():
            if col in factors:
                parts.append(f"{label}: {factors[col]:.4f}")
        self.cal_factors_label.setText("  ".join(parts) if parts else "")

    def _refresh_sessions(self) -> None:
        self._hydrate_generation += 1
        gen = self._hydrate_generation
        self._shutdown_meta_pool()
        # Cap concurrency: many parallel archive/FS reads slow down on typical disks.
        workers = max(4, min(12, (os.cpu_count() or 4) * 2))
        self._meta_pool = ThreadPoolExecutor(max_workers=workers)
        self._discovered = []
        self._sessions = []
        self._hydrate_received = 0
        self._scan_complete = False
        self.session_table.setRowCount(0)
        self._row_by_sid.clear()
        # Restore the previously persisted selection so rows get re-checked as
        # they stream in (reconciled against actually-present sessions on finish).
        self._check_order = list(self._settings.selected_sessions or [])[:MAX_SESSIONS]
        self._track_worker(
            threading.Thread(
                target=self._discover_sessions_worker,
                args=(gen, self._base_dir),
                daemon=True,
            )
        )
        self._schedule_status_refresh()

    def _incremental_refresh_sessions(self) -> None:
        """Rescan the data folder and merge new/removed sessions without clearing the table."""
        if not self._scan_complete:
            return
        self._hydrate_generation += 1
        gen = self._hydrate_generation
        self._shutdown_meta_pool()
        workers = max(4, min(12, (os.cpu_count() or 4) * 2))
        self._meta_pool = ThreadPoolExecutor(max_workers=workers)
        self._track_worker(
            threading.Thread(
                target=self._incremental_rescan_worker,
                args=(gen, self._base_dir),
                daemon=True,
            )
        )

    def _incremental_rescan_worker(self, gen: int, base_dir: str) -> None:
        from .common.sessions import discover_sessions

        found: list[tuple[str, str]] = []
        try:
            try:
                notes = load_notes(base_dir)
            except Exception:
                notes = {}
            if gen != self._hydrate_generation:
                return
            self._sig_notes_loaded.emit(gen, notes)
            try:
                rows = discover_sessions(
                    base_dirs=(base_dir,),
                    project_root=PROJECT_ROOT,
                )
            except Exception:
                rows = []
            for sid, path_str, _ in rows:
                if gen != self._hydrate_generation:
                    return
                found.append((sid, path_str))
        finally:
            if gen == self._hydrate_generation:
                self._sig_incremental_rescan.emit(gen, found)

    @Slot(int, object)
    def _on_incremental_rescan(self, gen: int, rows_obj: object) -> None:
        if gen != self._hydrate_generation:
            return
        if not isinstance(rows_obj, list):
            return
        found: list[tuple[str, str]] = []
        for item in rows_obj:
            if isinstance(item, tuple) and len(item) >= 2:
                found.append((str(item[0]), str(item[1])))
        found_map = dict(found)
        found_sids = set(found_map.keys())
        present_sids = set(self._row_by_sid.keys())

        removed = present_sids - found_sids
        if removed:
            self.session_table.blockSignals(True)
            self.session_table.setSortingEnabled(False)
            try:
                for row in sorted(
                    (self._find_row_for_sid(sid) for sid in removed),
                    reverse=True,
                ):
                    if row >= 0:
                        self.session_table.removeRow(row)
                self._discovered = [
                    entry for entry in self._discovered if entry[0] in found_sids
                ]
                if any(s in removed for s in self._check_order):
                    self._check_order = [s for s in self._check_order if s in found_sids]
                    self._persist_selected_sessions()
            finally:
                self.session_table.setSortingEnabled(True)
                self.session_table.blockSignals(False)
            self._rebuild_session_row_index()

        for i, (sid, path_str, meta) in enumerate(self._discovered):
            updated_path = found_map.get(sid)
            if updated_path is not None and updated_path != path_str:
                self._discovered[i] = (sid, updated_path, meta)

        new_sids = found_sids - present_sids
        if new_sids:
            to_add = [(sid, found_map[sid]) for sid in sorted(new_sids)]
            self.session_table.blockSignals(True)
            self.session_table.setSortingEnabled(False)
            try:
                first_row = self.session_table.rowCount()
                self.session_table.setRowCount(first_row + len(to_add))
                restored = set(self._check_order)
                for i, (sid, path_str) in enumerate(to_add):
                    row = first_row + i
                    self._discovered.append((sid, path_str, None))
                    self._set_session_row_widgets(
                        row, sid, None, use_checked=sid in restored
                    )
                    self._schedule_meta_hydrate(gen, sid, path_str)
            finally:
                self.session_table.setSortingEnabled(True)
                self.session_table.blockSignals(False)
            self._rebuild_session_row_index()

        self._sync_note_cells_from_store(found_sids)
        self._sessions = [s for s, _, _ in self._discovered]
        self._schedule_status_refresh()

    def _sync_note_cells_from_store(self, sids: set[str] | None = None) -> None:
        """Apply in-memory notes to table cells (e.g. after reloading notes.json)."""
        target = sids if sids is not None else set(self._row_by_sid.keys())
        self.session_table.blockSignals(True)
        try:
            for sid in target:
                row = self._find_row_for_sid(sid)
                if row < 0:
                    continue
                note_cell = self.session_table.item(row, _COL_NOTE)
                if note_cell is None:
                    continue
                note = self._notes.get(sid, "")
                if note_cell.text() != note:
                    note_cell.setText(note)
                note_cell.setToolTip(note if len(note) > 120 else "")
        finally:
            self.session_table.blockSignals(False)

    def _discover_sessions_worker(self, gen: int, base_dir: str) -> None:
        from .common.sessions import discover_sessions

        try:
            try:
                notes = load_notes(base_dir)
            except Exception:
                notes = {}
            if gen != self._hydrate_generation:
                return
            self._sig_notes_loaded.emit(gen, notes)
            try:
                rows = discover_sessions(
                    base_dirs=(base_dir,),
                    project_root=PROJECT_ROOT,
                )
            except Exception:
                rows = []
            batch: list[tuple[str, str]] = []
            for sid, path_str, _ in rows:
                if gen != self._hydrate_generation:
                    return
                batch.append((sid, path_str))
                if len(batch) >= _SESSION_ROW_BATCH:
                    self._sig_session_rows_batch.emit(gen, batch)
                    batch = []
            if gen != self._hydrate_generation:
                return
            if batch:
                self._sig_session_rows_batch.emit(gen, batch)
        finally:
            if gen == self._hydrate_generation:
                self._sig_scan_finished.emit(gen)

    def _track_worker(self, thread: threading.Thread) -> None:
        """Start *thread* and keep a handle so it can be joined on shutdown.

        Worker threads emit Qt signals, so Qt allocates thread-local storage for
        each one. They must finish before QApplication is destroyed, otherwise Qt
        prints "QThreadStorage: entry N destroyed before end of thread" on exit.
        """
        self._worker_threads = [t for t in self._worker_threads if t.is_alive()]
        self._worker_threads.append(thread)
        thread.start()

    def _shutdown_meta_pool(self, *, wait: bool = False) -> None:
        pool = self._meta_pool
        if pool is not None:
            self._meta_pool = None
            pool.shutdown(wait=wait, cancel_futures=True)

    def _schedule_meta_hydrate(self, gen: int, sid: str, path_str: str) -> None:
        pool = self._meta_pool
        if pool is None:
            return
        base_dir = self._base_dir

        def job() -> tuple[str, str, SessionMeta | None]:
            from .common.session_source import (
                load_session_termination_summary,
                resolve_session_source,
            )

            def _on_extracting(session_id: str) -> None:
                self._sig_session_extracting.emit(gen, session_id)

            base = Path(base_dir)
            src = resolve_session_source(sid, base, on_extracting=_on_extracting)
            meta = load_session_termination_summary(src) if src else None
            return sid, path_str, meta

        def _done(fut: Any) -> None:
            if gen != self._hydrate_generation:
                return
            try:
                sid_r, path_str_r, meta = fut.result()
            except Exception:
                sid_r, path_str_r, meta = sid, path_str, None
            self._sig_session_metadata.emit(gen, sid_r, path_str_r, meta)

        fut = pool.submit(job)
        fut.add_done_callback(_done)

    @Slot(int, object)
    def _on_notes_loaded(self, gen: int, notes: object) -> None:
        if gen != self._hydrate_generation:
            return
        if isinstance(notes, dict):
            self._notes = {str(k): str(v) for k, v in notes.items()}
        else:
            self._notes = {}

    @Slot(int, object)
    def _on_session_rows_batch(self, gen: int, batch: object) -> None:
        if gen != self._hydrate_generation:
            return
        if not isinstance(batch, list):
            return
        clean: list[tuple[str, str]] = []
        for item in batch:
            if isinstance(item, tuple) and len(item) >= 2:
                clean.append((str(item[0]), str(item[1])))
        if not clean:
            return
        # Disable sorting while inserting: with sorting on, each setItem would
        # re-sort and shift row indices mid-build. Re-enabling re-sorts once.
        self.session_table.blockSignals(True)
        self.session_table.setSortingEnabled(False)
        try:
            first_row = self.session_table.rowCount()
            self.session_table.setRowCount(first_row + len(clean))
            restored = set(self._check_order)
            for i, (sid, path_str) in enumerate(clean):
                row = first_row + i
                self._discovered.append((sid, path_str, None))
                self._set_session_row_widgets(
                    row, sid, None, use_checked=sid in restored
                )
                self._schedule_meta_hydrate(gen, sid, path_str)
        finally:
            self.session_table.setSortingEnabled(True)
            self.session_table.blockSignals(False)
        self._rebuild_session_row_index()
        self._sessions = [s for s, _, _ in self._discovered]
        self._schedule_status_refresh()

    @Slot(int)
    def _on_scan_finished(self, gen: int) -> None:
        if gen != self._hydrate_generation:
            return
        self._scan_complete = True
        self._sessions = [s for s, _, _ in self._discovered]
        # Drop any persisted selection that no longer maps to a present session.
        present = set(self._sessions)
        reconciled = [s for s in self._check_order if s in present]
        if reconciled != self._check_order:
            self._check_order = reconciled
            self._persist_selected_sessions()
        if self._hydrate_received >= len(self._discovered):
            self._schedule_meta_column_resize()
        self._schedule_status_refresh()

    def _resize_session_meta_columns(self) -> None:
        for c in (_COL_USE, _COL_SESSION_ID, _COL_DATE, _COL_MU, _COL_TIME):
            self.session_table.resizeColumnToContents(c)

    def _fill_meta_columns(self, row: int, meta: SessionMeta | None) -> None:
        """Set Date / MU / Time cells for one row (display text + real sort value)."""
        texts = _meta_column_texts(meta)
        sort_vals = _meta_sort_values(meta)
        align = Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft
        for col_offset, (text, sval) in enumerate(zip(texts, sort_vals)):
            col = _COL_DATE + col_offset
            it = self.session_table.item(row, col)
            if it is None:
                cell = _SortableItem(text)
                cell.setFlags(cell.flags() & ~Qt.ItemFlag.ItemIsEditable)
                cell.setTextAlignment(align)
                cell.setData(_SORT_VALUE_ROLE, sval)
                self.session_table.setItem(row, col, cell)
            else:
                it.setText(text)
                it.setData(_SORT_VALUE_ROLE, sval)

    def _set_session_row_widgets(
        self,
        row: int,
        sid: str,
        meta: SessionMeta | None,
        *,
        use_checked: bool,
    ) -> None:
        """Create or replace widgets for one session row (Use / Id / meta / note)."""
        use = QTableWidgetItem()
        use.setFlags(
            (use.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            & ~Qt.ItemFlag.ItemIsEditable
        )
        use.setData(_SESSION_ROLE, sid)
        use.setCheckState(
            Qt.CheckState.Checked if use_checked else Qt.CheckState.Unchecked
        )
        self.session_table.setItem(row, _COL_USE, use)

        id_cell = QTableWidgetItem(sid)
        id_cell.setFlags(id_cell.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.session_table.setItem(row, _COL_SESSION_ID, id_cell)

        self._fill_meta_columns(row, meta)

        note = self._notes.get(sid, "")
        note_cell = QTableWidgetItem(note)
        note_cell.setFlags(
            (note_cell.flags() | Qt.ItemFlag.ItemIsEditable | Qt.ItemFlag.ItemIsSelectable)
            & ~Qt.ItemFlag.ItemIsUserCheckable
        )
        note_cell.setTextAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        if len(note) > 120:
            note_cell.setToolTip(note)
        else:
            note_cell.setToolTip("")
        self.session_table.setItem(row, _COL_NOTE, note_cell)

    @Slot(int, str, str, object)
    def _on_session_metadata_ready(
        self,
        gen: int,
        sid: str,
        path_str: str,
        meta: object,
    ) -> None:
        if gen != self._hydrate_generation:
            return
        if isinstance(meta, SessionMeta):
            meta_obj = meta
        elif meta is None:
            meta_obj = None
        else:
            meta_obj = None
        for i, (s, p, _) in enumerate(self._discovered):
            if s == sid:
                self._discovered[i] = (sid, path_str, meta_obj)
                break
        self._hydrate_received += 1
        # Patch this row's Date/MU/Time values; native sorting reorders the row as the
        # data arrives (fine — rows are expected to settle as metadata streams in).
        self._patch_session_metadata_cells(sid, meta_obj)
        if self._scan_complete and self._hydrate_received >= len(self._discovered):
            self._schedule_meta_column_resize()
        self._schedule_status_refresh()

    def _patch_session_metadata_cells(self, sid: str, meta: SessionMeta | None) -> None:
        """Update Date/MU/Time cells for one row, then let the table re-sort once."""
        r = self._find_row_for_sid(sid)
        if r < 0:
            return
        # Toggle sorting off so all three cells update before a single re-sort; a row
        # whose values change may move, so refresh the sid→row index afterwards.
        self.session_table.blockSignals(True)
        self.session_table.setSortingEnabled(False)
        try:
            self._fill_meta_columns(r, meta)
        finally:
            self.session_table.setSortingEnabled(True)
            self.session_table.blockSignals(False)
        self._rebuild_session_row_index()

    @Slot(int, str)
    def _show_extracting_status(self, gen: int, session_id: str) -> None:
        pass

    def _session_row_sid(self, row: int) -> str | None:
        if row < 0 or row >= self.session_table.rowCount():
            return None
        it = self.session_table.item(row, _COL_USE)
        if it is None:
            return None
        sid = it.data(_SESSION_ROLE)
        return str(sid) if sid is not None else None

    def _find_row_for_sid(self, sid: str) -> int:
        r = self._row_by_sid.get(sid, -1)
        if 0 <= r < self.session_table.rowCount() and self._session_row_sid(r) == sid:
            return r
        for r2 in range(self.session_table.rowCount()):
            if self._session_row_sid(r2) == sid:
                self._row_by_sid[sid] = r2
                return r2
        return -1

    def _set_row_checked(self, sid: str, checked: bool) -> None:
        r = self._find_row_for_sid(sid)
        if r < 0:
            return
        it = self.session_table.item(r, _COL_USE)
        if it is None:
            return
        self.session_table.blockSignals(True)
        try:
            it.setCheckState(
                Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
            )
        finally:
            self.session_table.blockSignals(False)

    def _checked_sids_ordered(self) -> list[str]:
        """Top-to-bottom list of session IDs with Use checked (visual row order)."""
        out: list[str] = []
        for r in range(self.session_table.rowCount()):
            it0 = self.session_table.item(r, _COL_USE)
            if (
                it0 is not None
                and it0.checkState() == Qt.CheckState.Checked
            ):
                sid = it0.data(_SESSION_ROLE)
                if sid is not None:
                    out.append(str(sid))
        return out

    def _selected_sids_in_order(self) -> list[str]:
        """Checked sessions in the order the user selected them.

        This — not the visual row order — drives plot color assignment: color
        index = position in this list. Using selection order keeps each session's
        color fixed when the table is re-sorted or rows stream in, and assigns
        colors in the sequence the user checked boxes (``_check_order``).
        """
        checked = set(self._checked_sids_ordered())
        ordered = [s for s in self._check_order if s in checked]
        # Safety net: include any checked sid not tracked in _check_order.
        for s in self._checked_sids_ordered():
            if s not in ordered:
                ordered.append(s)
        return ordered[:MAX_SESSIONS]

    def _on_session_table_item_changed(self, item: QTableWidgetItem) -> None:
        col = item.column()
        if col == _COL_USE:
            sid_raw = item.data(_SESSION_ROLE)
            if sid_raw is None:
                return
            sid = str(sid_raw)
            if item.checkState() == Qt.CheckState.Checked:
                if sid not in self._check_order:
                    self._check_order.append(sid)
                while len(self._check_order) > MAX_SESSIONS:
                    drop = self._check_order.pop(0)
                    self._set_row_checked(drop, False)
            else:
                self._check_order = [s for s in self._check_order if s != sid]
            selected_set = set(self._checked_sids_ordered())
            self._check_order = [s for s in self._check_order if s in selected_set]
            self._persist_selected_sessions()
            self._schedule_status_refresh()
            return
        if col == _COL_NOTE:
            self._persist_note_from_cell(item)

    def _persist_note_from_cell(self, item: QTableWidgetItem) -> None:
        row = item.row()
        sid = self._session_row_sid(row)
        if sid is None:
            return
        text = item.text()
        current = self._notes.get(sid, "")
        if text == current:
            return
        if text.strip():
            self._notes[sid] = text
        else:
            self._notes.pop(sid, None)
        save_note(self._base_dir, sid, text)
        tip = text if len(text) > 120 else ""
        self.session_table.blockSignals(True)
        try:
            item.setToolTip(tip)
        finally:
            self.session_table.blockSignals(False)

    def _on_session_current_cell_changed(
        self,
        current_row: int,
        _current_col: int,
        _prev_row: int,
        _prev_col: int,
    ) -> None:
        if current_row < 0:
            self._highlighted_sid = None
            return
        sid = self._session_row_sid(current_row)
        if sid is None:
            return
        self._highlighted_sid = sid

    def _refresh_use_column_swatches(self, selected: list[str] | None = None) -> None:
        """Set Use-column icons to match visualization colors for checked sessions."""
        if selected is None:
            selected = self._selected_sids_in_order()
        n_selected = len(selected)
        idx_by_sid = {sid: i for i, sid in enumerate(selected)}
        self.session_table.blockSignals(True)
        try:
            for r in range(self.session_table.rowCount()):
                it = self.session_table.item(r, _COL_USE)
                if it is None:
                    continue
                sid_raw = it.data(_SESSION_ROLE)
                if sid_raw is None:
                    continue
                sid = str(sid_raw)
                active = it.checkState() == Qt.CheckState.Checked
                plot_i = idx_by_sid.get(sid) if active else None
                if active and plot_i is not None:
                    state_key = (True, plot_i, n_selected)
                elif active:
                    state_key = (True, -1, n_selected)
                else:
                    state_key = (False, -1, 0)
                if it.data(_SWATCH_STATE_ROLE) == state_key:
                    continue
                it.setData(_SWATCH_STATE_ROLE, state_key)
                it.setIcon(_cached_swatch_icon(active=active, plot_index=plot_i))
                if active and plot_i is not None:
                    cname = DEFAULT_SESSION_COLORS[
                        plot_i % len(DEFAULT_SESSION_COLORS)
                    ]
                    it.setToolTip(
                        f"Session color in plots: {cname} ({plot_i + 1} of {n_selected})"
                    )
                elif active:
                    it.setToolTip("Selected for analysis")
                else:
                    it.setToolTip(
                        "Not used in plots — check to assign a plot color by order"
                    )
        finally:
            self.session_table.blockSignals(False)

    def _update_status(self) -> None:
        selected = self._selected_sids_in_order()
        self._refresh_use_column_swatches(selected)

    def _clear_selection(self) -> None:
        self.session_table.blockSignals(True)
        try:
            for r in range(self.session_table.rowCount()):
                it = self.session_table.item(r, _COL_USE)
                if it is not None:
                    it.setCheckState(Qt.CheckState.Unchecked)
        finally:
            self.session_table.blockSignals(False)
        self._check_order.clear()
        self._persist_selected_sessions()
        self._schedule_status_refresh()

    def _persist_selected_sessions(self) -> None:
        """Save the current session selection into the persistent settings file."""
        self._settings.selected_sessions = self._selected_sids_in_order()
        try:
            self._settings.save(self._base_dir)
        except Exception:
            pass

    def _notify(self, message: str, *, error: bool = False) -> None:
        box = QMessageBox(self)
        box.setText(message)
        box.setIcon(QMessageBox.Icon.Critical if error else QMessageBox.Icon.Warning)
        box.exec()

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
            proc = subprocess.Popen(cmd, **popen_kwargs)
            self._child_procs.append(proc)
            self._reap_children()
        except Exception as e:
            self._notify(f"Failed to run analysis: {e}", error=True)
            return

        original_label = next(
            (name for name, mod in VIEWS if mod == module_name),
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

        threading.Thread(
            target=self._watch_subprocess_ready,
            args=(proc, module_name),
            daemon=True,
        ).start()

    def _update_spinner(self) -> None:
        frame = _SPINNER[self._spinner_frame % len(_SPINNER)]
        for module_name, (_proc, original_label) in self._running_views.items():
            btn = self._view_buttons.get(module_name)
            if btn is not None:
                btn.setText(f"{frame} {original_label}")

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
        self._shutdown_children()
        super().closeEvent(event)

    def _shutdown_children(self) -> None:
        # Bump generation first so in-flight workers bail at their next check, then
        # wait for the metadata pool and worker threads to finish. Letting these
        # signal-emitting threads outlive QApplication causes QThreadStorage warnings.
        self._hydrate_generation += 1
        self._shutdown_meta_pool(wait=True)
        for t in self._worker_threads:
            if t.is_alive():
                t.join(timeout=2.0)
        self._worker_threads.clear()
        if self._meta_col_resize_timer is not None:
            self._meta_col_resize_timer.stop()
        if self._status_refresh_timer is not None:
            self._status_refresh_timer.stop()
        if self._poll_timer is not None:
            self._poll_timer.stop()
            self._poll_timer = None
        self._running_views.clear()
        self._open_views.clear()
        self._launch_args.clear()
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
    win = ScanKitMainWindow()

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
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
