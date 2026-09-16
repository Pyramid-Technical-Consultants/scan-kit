"""Reusable session folder browser with discovery, metadata, and selection."""

from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QFileSystemWatcher, QRectF, QUrl, Qt, QTimer, Signal, QSize, Slot
from PySide6.QtGui import (
    QAction,
    QColor,
    QFontMetrics,
    QGuiApplication,
    QIcon,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
    QUndoCommand,
    QUndoStack,
)
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from scan_kit.common.data_location import (
    canonical_location,
    drop_cached_fs,
    format_location_error,
    is_auth_error,
    is_remote_location,
    location_uses_password,
    remember_password,
)
from scan_kit.common.plot_colors import DEFAULT_SESSION_COLORS
from scan_kit.common.recycle import move_to_trash
from scan_kit.common.session_meta import SessionMeta
from scan_kit.common.session_source import list_session_storage_paths
from scan_kit.common.user_store import (
    delete_session,
    record_session_meta,
    set_selected_sessions,
    set_session_note,
    snapshot_library,
)

_log = logging.getLogger(__name__)

_SESSION_ROW_BATCH = 24
_SESSION_ROLE = Qt.ItemDataRole.UserRole
_SWATCH_STATE_ROLE = Qt.ItemDataRole.UserRole + 1
_SORT_VALUE_ROLE = Qt.ItemDataRole.UserRole + 2

_COL_USE = 0
_COL_SESSION_ID = 1
_COL_DATE = 2
_COL_MU = 3
_COL_TIME = 4
_COL_ROOM = 5
_COL_CONFIG = 6
_COL_NOTE = 7

_COMPACT_META_COLS = (_COL_MU, _COL_TIME, _COL_ROOM)
_META_COLS = (_COL_DATE, _COL_MU, _COL_TIME, _COL_ROOM, _COL_CONFIG)

_SWATCH_PX = 14
_UNCHECKED_SWATCH = QColor("#d0d0d0")
_SWATCH_LINE = QColor("#6a6a6a")

# Declared so native/portal dialogs can offer URI locations (GTK GVfs, KDE
# KIO, xdg-desktop-portal). Windows still browses UNC via Network.
_DATA_DIR_SCHEMES = (
    "sftp",
    "ssh",
    "scp",
    "smb",
    "ftp",
    "ftps",
    "http",
    "https",
)


def directory_url_for_dialog(spec: str) -> QUrl:
    """Starting URL for the session-folder picker."""
    text = spec.strip()
    if not text:
        return QUrl.fromLocalFile(str(Path.home()))
    if is_remote_location(text):
        return QUrl(text)
    return QUrl.fromLocalFile(str(Path(text).expanduser()))


def location_from_dialog_url(url: QUrl) -> str:
    """Turn a picker result into a data-source string (local path or URL)."""
    if url.isEmpty() or not url.isValid():
        return ""
    if url.scheme().lower() in {"clsid", "shell"}:
        return ""
    if url.isLocalFile():
        return url.toLocalFile()
    return url.toString()


def prompt_remote_password(parent: QWidget | None, spec: str) -> bool:
    """Ask for a remote password and remember it in process memory."""
    label = canonical_location(spec) or spec
    text, ok = QInputDialog.getText(
        parent,
        "Remote password",
        f"Password for {label}\n(kept in memory until Scan Kit exits):",
        QLineEdit.EchoMode.Password,
    )
    if not ok or not text:
        return False
    remember_password(spec, text)
    return True


def default_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _session_plot_swatch_icon(*, active: bool, plot_index: int | None) -> QIcon:
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
    p.drawRoundedRect(QRectF(inset, inset, r, r), 2.5, 2.5)
    p.end()
    return QIcon(pix)


_SWATCH_ICON_CACHE: dict[tuple[str, int | None], QIcon] = {}


def _cached_swatch_icon(*, active: bool, plot_index: int | None) -> QIcon:
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


class _SortableItem(QTableWidgetItem):
    def __lt__(self, other: QTableWidgetItem) -> bool:
        a = self.data(_SORT_VALUE_ROLE)
        b = other.data(_SORT_VALUE_ROLE) if isinstance(other, QTableWidgetItem) else None
        if a is None or b is None:
            return a is None and b is not None
        try:
            return a < b
        except TypeError:
            return str(a) < str(b)


def _meta_column_texts(meta: SessionMeta | None) -> tuple[str, str, str, str, str]:
    if meta is None:
        return "—", "—", "—", "?", "—"
    return (
        meta.short_date,
        meta.short_mu,
        meta.short_time,
        meta.short_room,
        meta.short_config,
    )


def _meta_sort_values(
    meta: SessionMeta | None,
) -> tuple[datetime | None, float | None, int | None, int | None, str | None]:
    if meta is None:
        return (None, None, None, None, None)
    config = (meta.config_name or "").strip() or None
    return (meta.date, meta.primary_mu, meta.treatment_time_s, meta.room_number, config)


def _compact_meta_column_widths(fm: QFontMetrics) -> dict[int, int]:
    """Tight fixed widths for MU / Time / RM; global header min size would otherwise clamp them."""
    pad = 10  # cell padding + sort indicator slack
    return {
        _COL_MU: fm.horizontalAdvance("999.9") + pad,
        _COL_TIME: fm.horizontalAdvance("99:59") + pad,
        _COL_ROOM: max(fm.horizontalAdvance("RM"), fm.horizontalAdvance("99")) + pad,
    }


def _use_column_width(table: QTableWidget) -> int:
    """Checkbox + plot swatch + header; fixed so refresh does not twitch the column."""
    fm = QFontMetrics(table.font())
    check = table.style().pixelMetric(
        QStyle.PixelMetric.PM_IndicatorWidth, None, table
    )
    header = fm.horizontalAdvance("Use") + 16
    return max(header, check + table.iconSize().width() + 16)


def _config_column_width(fm: QFontMetrics) -> int:
    """About half a typical configuration name; leftover pane width is shared with Note."""
    return fm.horizontalAdvance("working_hvtt_new_") + 10


class _NoteEditCommand(QUndoCommand):
    """Undoable change to a single session's note.

    Each committed cell edit is its own command; we deliberately do not merge
    consecutive edits (``id()`` stays the default ``-1``) because a table cell
    commits one discrete value per edit, so per-edit granularity is what users
    expect from undo here.
    """

    def __init__(
        self,
        browser: "SessionBrowserWidget",
        sid: str,
        old_text: str,
        new_text: str,
    ) -> None:
        super().__init__(f"edit note for {sid}")
        self._browser = browser
        self._sid = sid
        self._old_text = old_text
        self._new_text = new_text

    def redo(self) -> None:  # also runs once when first pushed
        self._browser._apply_note(self._sid, self._new_text)

    def undo(self) -> None:
        self._browser._apply_note(self._sid, self._old_text)


class SessionBrowserWidget(QWidget):
    """Browse session archives/folders, inspect metadata, and select sessions."""

    base_dir_changed = Signal(str)
    selection_changed = Signal(list)
    highlighted_session_changed = Signal(object)
    populate_context_menu = Signal(str, object)

    _sig_session_metadata = Signal(int, str, str, object)
    _sig_notes_loaded = Signal(int, object)
    _sig_session_rows_batch = Signal(int, object)
    _sig_scan_finished = Signal(int)
    _sig_incremental_rescan = Signal(int, object)
    _sig_scan_error = Signal(int, str, bool)

    def __init__(
        self,
        *,
        project_root: Path | None = None,
        initial_base_dir: str,
        max_selections: int = 5,
        editable_notes: bool = True,
        show_plot_swatches: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._project_root = project_root or default_project_root()
        self._base_dir = initial_base_dir
        self._max_selections = max(1, max_selections)
        self._editable_notes = editable_notes
        self._show_plot_swatches = show_plot_swatches
        self._selection_persist: Callable[[list[str]], None] | None = None

        self._discovered: list[tuple[str, str, SessionMeta | None]] = []
        self._hydrate_generation = 0
        self._hydrate_received = 0
        self._scan_complete = False
        self._meta_pool: ThreadPoolExecutor | None = None
        self._worker_threads: list[threading.Thread] = []
        self._notes: dict[str, str] = {}
        self._undo_stack = QUndoStack(self)
        self._undo_action: QAction | None = None
        self._redo_action: QAction | None = None
        self._highlighted_sid: str | None = None
        self._check_order: list[str] = []
        self._row_by_sid: dict[str, int] = {}
        self._restored_selection_override: list[str] | None = None
        self._sorting_held = False
        self._trash = move_to_trash
        self._confirm_recycle = self._confirm_recycle_dialog

        self._meta_col_resize_timer = QTimer(self)
        self._meta_col_resize_timer.setSingleShot(True)
        self._meta_col_resize_timer.timeout.connect(self._resize_session_meta_columns)
        self._status_refresh_timer = QTimer(self)
        self._status_refresh_timer.setSingleShot(True)
        self._status_refresh_timer.timeout.connect(self._update_status)
        self._fs_debounce = QTimer(self)
        self._fs_debounce.setSingleShot(True)
        self._fs_debounce.timeout.connect(self.incremental_refresh)
        self._fs_watcher = QFileSystemWatcher(self)
        self._fs_watcher.directoryChanged.connect(self._on_data_dir_changed)
        self._auth_prompted: set[str] = set()
        self._scan_error_message: str | None = None

        self._connect_worker_signals()
        self._build_ui()
        self._watch_base_dir()

    def _connect_worker_signals(self) -> None:
        self._sig_session_metadata.connect(
            self._on_session_metadata_ready,
            Qt.ConnectionType.QueuedConnection,
        )
        self._sig_notes_loaded.connect(
            self._on_notes_loaded,
            Qt.ConnectionType.QueuedConnection,
        )
        self._sig_session_rows_batch.connect(
            self._on_session_rows_batch,
            Qt.ConnectionType.QueuedConnection,
        )
        self._sig_scan_finished.connect(
            self._on_scan_finished,
            Qt.ConnectionType.QueuedConnection,
        )
        self._sig_incremental_rescan.connect(
            self._on_incremental_rescan,
            Qt.ConnectionType.QueuedConnection,
        )
        self._sig_scan_error.connect(
            self._on_scan_error,
            Qt.ConnectionType.QueuedConnection,
        )

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        data_dir_row = QHBoxLayout()
        clear_btn = QPushButton("✕")
        clear_btn.setFixedWidth(28)
        clear_btn.setToolTip("Clear session selection")
        clear_btn.clicked.connect(self.clear_selection)
        data_dir_row.addWidget(clear_btn)

        self._base_dir_input = QLineEdit()
        self._base_dir_input.setPlaceholderText(
            "Folder, UNC, or sftp://user@host/path"
        )
        self._base_dir_input.setText(self._base_dir)
        self._base_dir_input.editingFinished.connect(self._on_base_dir_finished)
        self._base_dir_input.returnPressed.connect(self._on_base_dir_finished)
        data_dir_row.addWidget(self._base_dir_input, stretch=1)

        browse_dir_btn = QPushButton("Browse…")
        browse_dir_btn.setToolTip(
            "Choose a local, UNC, or network folder. SFTP URLs can also be pasted."
        )
        browse_dir_btn.setFixedWidth(96)
        browse_dir_btn.clicked.connect(self._on_browse_data_dir)
        data_dir_row.addWidget(browse_dir_btn)

        refresh_btn = QPushButton("↻")
        refresh_btn.setFixedWidth(28)
        refresh_btn.setToolTip("Refresh session list")
        refresh_btn.clicked.connect(self.incremental_refresh)
        data_dir_row.addWidget(refresh_btn)
        root.addLayout(data_dir_row)

        self._path_status = QLabel("")
        self._path_status.setWordWrap(True)
        self._path_status.setVisible(False)
        root.addWidget(self._path_status)

        self._table = QTableWidget()
        self._table.setColumnCount(8)
        self._table.setHorizontalHeaderLabels(
            ["Use", "Session ID", "Date", "MU", "Time", "RM", "Config", "Note"]
        )
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.setSortingEnabled(True)
        self._table.setAlternatingRowColors(True)
        self._table.setTextElideMode(Qt.TextElideMode.ElideRight)
        self._table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._table.setIconSize(QSize(_SWATCH_PX, _SWATCH_PX))
        hh = self._table.horizontalHeader()
        hh.setMinimumSectionSize(16)
        hh.setSectionResizeMode(_COL_USE, QHeaderView.ResizeMode.Fixed)
        hh.resizeSection(_COL_USE, _use_column_width(self._table))
        hh.setSectionResizeMode(_COL_SESSION_ID, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(_COL_DATE, QHeaderView.ResizeMode.ResizeToContents)
        fm = QFontMetrics(self._table.font())
        compact_widths = _compact_meta_column_widths(fm)
        for col in _COMPACT_META_COLS:
            hh.setSectionResizeMode(col, QHeaderView.ResizeMode.Fixed)
            hh.resizeSection(col, compact_widths[col])
        hh.setSectionResizeMode(_COL_CONFIG, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(_COL_NOTE, QHeaderView.ResizeMode.Stretch)
        hh.resizeSection(_COL_SESSION_ID, 220)
        hh.resizeSection(_COL_CONFIG, _config_column_width(fm))
        hh.setSortIndicatorShown(True)
        self._table.sortByColumn(_COL_DATE, Qt.SortOrder.DescendingOrder)
        self._table.verticalHeader().setVisible(False)
        self._table.itemChanged.connect(self._on_table_item_changed)
        self._table.currentCellChanged.connect(self._on_current_cell_changed)
        self._table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._table.customContextMenuRequested.connect(self._on_context_menu)
        root.addWidget(self._table, stretch=1)

        if self._editable_notes:
            self._install_undo_actions()

    def _install_undo_actions(self) -> None:
        """Create Qt-managed undo/redo actions (auto text + enabled state).

        These are exposed via :meth:`undo_action`/:meth:`redo_action` so a host
        window can drop them straight into an Edit menu; in the meantime they
        carry the standard shortcuts scoped to this widget and its children.
        """
        undo = self._undo_stack.createUndoAction(self, "Undo")
        undo.setShortcuts(QKeySequence.StandardKey.Undo)
        undo.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)

        redo = self._undo_stack.createRedoAction(self, "Redo")
        redo.setShortcuts(
            [QKeySequence(QKeySequence.StandardKey.Redo), QKeySequence("Ctrl+Shift+Z")]
        )
        redo.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)

        self.addAction(undo)
        self.addAction(redo)
        self._undo_action = undo
        self._redo_action = redo

    def undo_action(self) -> QAction | None:
        """The Qt undo action (for embedding in a menu/toolbar), if notes are editable."""
        return self._undo_action

    def redo_action(self) -> QAction | None:
        """The Qt redo action (for embedding in a menu/toolbar), if notes are editable."""
        return self._redo_action

    def base_dir(self) -> str:
        return self._base_dir

    def browse_for_base_dir(self) -> None:
        """Open the folder picker to choose a new session data directory."""
        self._on_browse_data_dir()

    def set_base_dir(self, path: str, *, refresh: bool = True) -> None:
        text = path.strip()
        if not text:
            return
        if canonical_location(text) != canonical_location(self._base_dir):
            self._auth_prompted.clear()
        self._base_dir = text
        self._base_dir_input.setText(text)
        if refresh:
            self.refresh()

    def set_selection_persistence(
        self,
        handler: Callable[[list[str]], None] | None,
    ) -> None:
        self._selection_persist = handler

    def notes(self) -> dict[str, str]:
        return dict(self._notes)

    def session_meta_by_id(self) -> dict[str, SessionMeta | None]:
        return {sid: meta for sid, _, meta in self._discovered}

    def discovered_sessions(self) -> list[tuple[str, str, SessionMeta | None]]:
        return list(self._discovered)

    def highlighted_session_id(self) -> str | None:
        return self._highlighted_sid

    def selected_session_ids(self) -> list[str]:
        checked = set(self._checked_sids_ordered())
        ordered = [s for s in self._check_order if s in checked]
        for sid in self._checked_sids_ordered():
            if sid not in ordered:
                ordered.append(sid)
        return ordered[: self._max_selections]

    def refresh(
        self,
        *,
        restored_selection: list[str] | None = None,
        retrying_auth: bool = False,
    ) -> None:
        self._hydrate_generation += 1
        gen = self._hydrate_generation
        if not retrying_auth:
            self._auth_prompted.discard(canonical_location(self._base_dir))
        self._scan_error_message = None
        self._undo_stack.clear()
        self._shutdown_meta_pool()
        workers = max(4, min(12, (os.cpu_count() or 4) * 2))
        self._meta_pool = ThreadPoolExecutor(max_workers=workers)
        self._discovered = []
        self._hydrate_received = 0
        self._scan_complete = False
        self._table.setRowCount(0)
        self._row_by_sid.clear()
        if restored_selection is not None:
            self._check_order = list(restored_selection)[: self._max_selections]
        self._restored_selection_override = restored_selection
        self._hold_sorting()
        self._watch_base_dir()
        if is_remote_location(self._base_dir):
            self._set_path_status(
                f"Listing {canonical_location(self._base_dir)}…"
            )
        else:
            self._set_path_status("")
        self._track_worker(
            threading.Thread(
                target=self._discover_sessions_worker,
                args=(gen, self._base_dir),
                daemon=True,
            )
        )
        self._schedule_status_refresh()

    def incremental_refresh(self) -> None:
        if not self._scan_complete:
            return
        self._hydrate_generation += 1
        gen = self._hydrate_generation
        self._auth_prompted.discard(canonical_location(self._base_dir))
        self._scan_error_message = None
        self._shutdown_meta_pool()
        workers = max(4, min(12, (os.cpu_count() or 4) * 2))
        self._meta_pool = ThreadPoolExecutor(max_workers=workers)
        self._hold_sorting()
        if is_remote_location(self._base_dir):
            self._set_path_status(
                f"Listing {canonical_location(self._base_dir)}…"
            )
        self._track_worker(
            threading.Thread(
                target=self._incremental_rescan_worker,
                args=(gen, self._base_dir),
                daemon=True,
            )
        )

    def clear_selection(self) -> None:
        self._table.blockSignals(True)
        try:
            for row in range(self._table.rowCount()):
                use = self._table.item(row, _COL_USE)
                if use is not None:
                    use.setCheckState(Qt.CheckState.Unchecked)
        finally:
            self._table.blockSignals(False)
        self._check_order.clear()
        self._persist_selection()
        self._schedule_status_refresh()

    def shutdown(self) -> None:
        self._hydrate_generation += 1
        self._shutdown_meta_pool(wait=True)
        for thread in self._worker_threads:
            if thread.is_alive():
                thread.join(timeout=2.0)
        self._worker_threads.clear()
        if self._meta_col_resize_timer is not None:
            self._meta_col_resize_timer.stop()
        if self._status_refresh_timer is not None:
            self._status_refresh_timer.stop()
        if self._fs_debounce is not None:
            self._fs_debounce.stop()

    def _on_base_dir_finished(self) -> None:
        path = self._base_dir_input.text().strip()
        if not path:
            return
        if canonical_location(path) != canonical_location(self._base_dir):
            self._auth_prompted.clear()
        self._base_dir = path
        self.base_dir_changed.emit(self._base_dir)
        self.refresh()

    def _on_browse_data_dir(self) -> None:
        start = self._base_dir_input.text().strip() or self._base_dir
        chosen = QFileDialog.getExistingDirectoryUrl(
            self,
            "Select session data folder",
            directory_url_for_dialog(start),
            QFileDialog.Option.ShowDirsOnly,
            list(_DATA_DIR_SCHEMES),
        )
        location = location_from_dialog_url(chosen)
        if not location:
            return
        self._base_dir_input.setText(location)
        self._on_base_dir_finished()

    def _on_context_menu(self, pos) -> None:
        row = self._table.rowAt(pos.y())
        if row < 0:
            return
        sid = self._session_row_sid(row)
        if sid is None:
            return
        menu = QMenu(self)
        menu.addAction(
            "Copy Session ID",
            lambda checked=False, session_id=sid: self._copy_session_id(session_id),
        )
        if is_remote_location(self._base_dir):
            recycle_label = "Delete from remote host…"
        else:
            recycle_label = "Move to Recycle Bin…"
        menu.addAction(
            recycle_label,
            lambda checked=False, session_id=sid: self._recycle_session(session_id),
        )
        self.populate_context_menu.emit(sid, menu)
        if menu.isEmpty():
            return
        menu.exec(self._table.viewport().mapToGlobal(pos))

    def _copy_session_id(self, sid: str) -> None:
        QGuiApplication.clipboard().setText(sid)

    def _watch_base_dir(self) -> None:
        for current in list(self._fs_watcher.directories()):
            self._fs_watcher.removePath(current)
        if is_remote_location(self._base_dir):
            return
        folder = Path(self._base_dir)
        if folder.is_dir():
            self._fs_watcher.addPath(str(folder))

    def _on_data_dir_changed(self, _path: str) -> None:
        self._fs_debounce.start(500)

    def _hold_sorting(self) -> None:
        if not self._sorting_held:
            self._table.setSortingEnabled(False)
            self._sorting_held = True

    def _release_sorting(self) -> None:
        if not self._sorting_held:
            return
        self._table.setSortingEnabled(True)
        self._sorting_held = False
        self._rebuild_session_row_index()

    def _confirm_recycle_dialog(self, sid: str, paths: list[str]) -> bool:
        lines = "\n".join(f"  {p}" for p in paths)
        selected = sid in set(self.selected_session_ids())
        extra = ""
        if selected:
            extra = (
                "\n\nThis session is currently selected for analysis. "
                "Open plot windows may fail after the files are removed."
            )
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        if is_remote_location(self._base_dir):
            box.setWindowTitle("Delete remote session")
            box.setText(
                f"Permanently delete session {sid} from the remote host?\n\n"
                f"These items will be removed:\n{lines}\n\n"
                "They are not sent to the Recycle Bin."
                f"{extra}"
            )
        else:
            box.setWindowTitle("Move session to Recycle Bin")
            box.setText(
                f"Move session {sid} to the Recycle Bin?\n\n"
                f"These items will be removed:\n{lines}\n\n"
                "You can restore them from the Recycle Bin if needed."
                f"{extra}"
            )
        box.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel
        )
        box.setDefaultButton(QMessageBox.StandardButton.Cancel)
        return box.exec() == QMessageBox.StandardButton.Yes

    def _recycle_session(self, sid: str) -> None:
        paths = list_session_storage_paths(self._base_dir, sid)
        if not paths:
            return
        if not self._confirm_recycle(sid, paths):
            return
        if sid in set(self.selected_session_ids()):
            self._set_row_checked(sid, False)
            self._check_order = [s for s in self._check_order if s != sid]
            self._persist_selection()
        self._fs_debounce.stop()
        errors: list[str] = []
        for path in paths:
            try:
                self._trash(path)
            except OSError as exc:
                errors.append(f"{path}: {exc}")
        try:
            delete_session(self._base_dir, sid)
        except Exception:
            pass
        self._notes.pop(sid, None)
        row = self._find_row_for_sid(sid)
        if row >= 0:
            self._table.removeRow(row)
            self._rebuild_session_row_index()
        self._discovered = [entry for entry in self._discovered if entry[0] != sid]
        self._schedule_status_refresh()
        if errors:
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Critical)
            box.setText("Could not move some session files to the Recycle Bin:\n" + "\n".join(errors))
            box.exec()

    def _schedule_status_refresh(self) -> None:
        self._status_refresh_timer.start(0)

    def _schedule_meta_column_resize(self) -> None:
        self._meta_col_resize_timer.start(60)

    def _rebuild_session_row_index(self) -> None:
        self._row_by_sid.clear()
        for row in range(self._table.rowCount()):
            sid = self._session_row_sid(row)
            if sid is not None:
                self._row_by_sid[sid] = row

    def _track_worker(self, thread: threading.Thread) -> None:
        self._worker_threads = [t for t in self._worker_threads if t.is_alive()]
        self._worker_threads.append(thread)
        thread.start()

    def _shutdown_meta_pool(self, *, wait: bool = False) -> None:
        pool = self._meta_pool
        if pool is not None:
            self._meta_pool = None
            pool.shutdown(wait=wait, cancel_futures=True)

    def _discover_sessions_worker(self, gen: int, base_dir: str) -> None:
        try:
            result = self._snapshot_library_or_error(gen, base_dir)
            if result is None:
                return
            notes, selected, rows = result
            if gen != self._hydrate_generation:
                return
            self._sig_notes_loaded.emit(gen, {"notes": notes, "selected": selected})
            batch: list[tuple[str, str, SessionMeta | None]] = []
            for sid, path_str, meta in rows:
                if gen != self._hydrate_generation:
                    return
                batch.append((sid, path_str, meta))
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

    def _incremental_rescan_worker(self, gen: int, base_dir: str) -> None:
        found: list[tuple[str, str, SessionMeta | None]] | None = []
        try:
            result = self._snapshot_library_or_error(gen, base_dir)
            if result is None:
                found = None
                return
            notes, selected, rows = result
            if gen != self._hydrate_generation:
                return
            self._sig_notes_loaded.emit(gen, {"notes": notes, "selected": selected})
            found = []
            for sid, path_str, meta in rows:
                if gen != self._hydrate_generation:
                    return
                found.append((sid, path_str, meta))
        finally:
            if gen == self._hydrate_generation:
                self._sig_incremental_rescan.emit(gen, found)

    def _snapshot_library_or_error(
        self, gen: int, base_dir: str
    ) -> tuple[dict[str, str], list[str], list[tuple[str, str, SessionMeta | None]]] | None:
        try:
            return snapshot_library(base_dir, project_root=self._project_root)
        except Exception as exc:
            _log.exception("Could not list sessions in %s", canonical_location(base_dir))
            if is_auth_error(exc):
                drop_cached_fs(base_dir)
            needs_pw = is_auth_error(exc) and location_uses_password(base_dir)
            self._sig_scan_error.emit(
                gen, format_location_error(base_dir, exc), needs_pw
            )
            return None

    @Slot(int, object)
    def _on_notes_loaded(self, gen: int, notes: object) -> None:
        if gen != self._hydrate_generation:
            return
        selected: list[str] = []
        payload = notes
        if isinstance(notes, dict) and "notes" in notes:
            payload = notes.get("notes")
            raw_sel = notes.get("selected")
            if isinstance(raw_sel, list):
                selected = [str(s) for s in raw_sel]
        if isinstance(payload, dict):
            self._notes = {str(k): str(v) for k, v in payload.items()}
        else:
            self._notes = {}
        if self._restored_selection_override is not None:
            self._check_order = list(self._restored_selection_override)[
                : self._max_selections
            ]
        elif isinstance(notes, dict) and "selected" in notes:
            self._check_order = selected[: self._max_selections]

    @Slot(int, object)
    def _on_session_rows_batch(self, gen: int, batch: object) -> None:
        if gen != self._hydrate_generation:
            return
        if not isinstance(batch, list):
            return
        clean: list[tuple[str, str, SessionMeta | None]] = []
        for item in batch:
            if isinstance(item, tuple) and len(item) >= 2:
                meta = item[2] if len(item) >= 3 and isinstance(item[2], SessionMeta) else None
                clean.append((str(item[0]), str(item[1]), meta))
        if not clean:
            return
        self._table.blockSignals(True)
        self._hold_sorting()
        try:
            first_row = self._table.rowCount()
            self._table.setRowCount(first_row + len(clean))
            restored = set(self._check_order)
            for i, (sid, path_str, meta) in enumerate(clean):
                row = first_row + i
                self._discovered.append((sid, path_str, meta))
                self._set_session_row_widgets(
                    row,
                    sid,
                    meta,
                    use_checked=sid in restored,
                )
                if meta is None:
                    self._schedule_meta_hydrate(gen, sid, path_str)
                else:
                    self._hydrate_received += 1
        finally:
            self._table.blockSignals(False)
        self._rebuild_session_row_index()
        self._maybe_finish_hydrate()
        self._schedule_status_refresh()

    @Slot(int)
    def _on_scan_finished(self, gen: int) -> None:
        if gen != self._hydrate_generation:
            return
        self._scan_complete = True
        present = {sid for sid, _, _ in self._discovered}
        reconciled = [s for s in self._check_order if s in present]
        if reconciled != self._check_order:
            self._check_order = reconciled
            self._persist_selection()
        if self._hydrate_received >= len(self._discovered):
            self._maybe_finish_hydrate()
        self._schedule_status_refresh()
        self._refresh_path_status_after_scan()

    @Slot(int, str, bool)
    def _on_scan_error(self, gen: int, message: str, needs_password: bool) -> None:
        if gen != self._hydrate_generation:
            return
        self._scan_error_message = message
        self._set_path_status(message, error=True)
        if needs_password:
            QTimer.singleShot(0, self._prompt_and_retry_password)

    def _prompt_and_retry_password(self) -> None:
        key = canonical_location(self._base_dir)
        if key in self._auth_prompted:
            return
        self._auth_prompted.add(key)
        if not prompt_remote_password(self, self._base_dir):
            return
        self.refresh(retrying_auth=True)

    def _refresh_path_status_after_scan(self) -> None:
        if self._scan_error_message:
            self._set_path_status(self._scan_error_message, error=True)
            return
        if is_remote_location(self._base_dir):
            self._set_path_status(f"Remote  ·  {canonical_location(self._base_dir)}")
        else:
            self._set_path_status("")

    def _set_path_status(self, text: str, *, error: bool = False) -> None:
        self._path_status.setText(text)
        self._path_status.setVisible(bool(text))
        if error:
            self._path_status.setStyleSheet("color: #c62828;")
        else:
            self._path_status.setStyleSheet("color: palette(mid);")

    @Slot(int, object)
    def _on_incremental_rescan(self, gen: int, rows_obj: object) -> None:
        if gen != self._hydrate_generation:
            return
        if not isinstance(rows_obj, list):
            self._release_sorting()
            self._schedule_status_refresh()
            return
        found: list[tuple[str, str, SessionMeta | None]] = []
        for item in rows_obj:
            if isinstance(item, tuple) and len(item) >= 2:
                meta = item[2] if len(item) >= 3 and isinstance(item[2], SessionMeta) else None
                found.append((str(item[0]), str(item[1]), meta))
        found_map = {sid: (path_str, meta) for sid, path_str, meta in found}
        found_sids = set(found_map.keys())
        present_sids = set(self._row_by_sid.keys())

        removed = present_sids - found_sids
        if removed:
            self._table.blockSignals(True)
            self._hold_sorting()
            try:
                for row in sorted(
                    (self._find_row_for_sid(sid) for sid in removed),
                    reverse=True,
                ):
                    if row >= 0:
                        self._table.removeRow(row)
                self._discovered = [
                    entry for entry in self._discovered if entry[0] in found_sids
                ]
                if any(s in removed for s in self._check_order):
                    self._check_order = [s for s in self._check_order if s in found_sids]
                    self._persist_selection()
            finally:
                self._table.blockSignals(False)
            self._rebuild_session_row_index()

        for i, (sid, path_str, meta) in enumerate(self._discovered):
            updated = found_map.get(sid)
            if updated is None:
                continue
            updated_path, found_meta = updated
            if updated_path != path_str or found_meta is not None:
                self._discovered[i] = (sid, updated_path, found_meta or meta)
            if found_meta is not None:
                self._patch_session_metadata_cells(sid, found_meta)
            elif updated_path != path_str:
                self._schedule_meta_hydrate(gen, sid, updated_path)

        new_sids = found_sids - present_sids
        if new_sids:
            to_add = [(sid, found_map[sid][0], found_map[sid][1]) for sid in sorted(new_sids)]
            self._table.blockSignals(True)
            self._hold_sorting()
            try:
                first_row = self._table.rowCount()
                self._table.setRowCount(first_row + len(to_add))
                restored = set(self._check_order)
                for i, (sid, path_str, meta) in enumerate(to_add):
                    row = first_row + i
                    self._discovered.append((sid, path_str, meta))
                    self._set_session_row_widgets(
                        row,
                        sid,
                        meta,
                        use_checked=sid in restored,
                    )
                    if meta is None:
                        self._schedule_meta_hydrate(gen, sid, path_str)
                    else:
                        self._hydrate_received += 1
            finally:
                self._table.blockSignals(False)
            self._rebuild_session_row_index()

        self._sync_note_cells_from_store(found_sids)
        self._maybe_finish_hydrate()
        self._schedule_status_refresh()
        self._refresh_path_status_after_scan()

    def _sync_note_cells_from_store(self, sids: set[str] | None = None) -> None:
        target = sids if sids is not None else set(self._row_by_sid.keys())
        self._table.blockSignals(True)
        try:
            for sid in target:
                row = self._find_row_for_sid(sid)
                if row < 0:
                    continue
                note_cell = self._table.item(row, _COL_NOTE)
                if note_cell is None:
                    continue
                note = self._notes.get(sid, "")
                if note_cell.text() != note:
                    note_cell.setText(note)
                note_cell.setToolTip(note if len(note) > 120 else "")
        finally:
            self._table.blockSignals(False)

    def _schedule_meta_hydrate(self, gen: int, sid: str, path_str: str) -> None:
        pool = self._meta_pool
        if pool is None:
            return
        base_dir = self._base_dir

        def job() -> tuple[str, str, SessionMeta | None]:
            from scan_kit.common.session_source import load_termination_summary_cached

            meta = load_termination_summary_cached(sid, path_str)
            try:
                record_session_meta(base_dir, sid, path_str, meta)
            except Exception:
                pass
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
        meta_obj = meta if isinstance(meta, SessionMeta) else None
        for i, (s, p, _) in enumerate(self._discovered):
            if s == sid:
                self._discovered[i] = (sid, path_str, meta_obj)
                break
        self._hydrate_received += 1
        self._patch_session_metadata_cells(sid, meta_obj)
        self._maybe_finish_hydrate()
        self._schedule_status_refresh()

    def _maybe_finish_hydrate(self) -> None:
        if self._scan_complete and self._hydrate_received >= len(self._discovered):
            self._release_sorting()
            self._schedule_meta_column_resize()

    def _patch_session_metadata_cells(self, sid: str, meta: SessionMeta | None) -> None:
        row = self._find_row_for_sid(sid)
        if row < 0:
            return
        self._table.blockSignals(True)
        try:
            self._fill_meta_columns(row, meta)
        finally:
            self._table.blockSignals(False)

    def _resize_session_meta_columns(self) -> None:
        for col in (_COL_SESSION_ID, _COL_DATE):
            self._table.resizeColumnToContents(col)

    def _fill_meta_columns(self, row: int, meta: SessionMeta | None) -> None:
        texts = _meta_column_texts(meta)
        sort_vals = _meta_sort_values(meta)
        full_config = (meta.config_name or "").strip() if meta is not None else ""
        align = Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft
        for col, text, sval in zip(_META_COLS, texts, sort_vals):
            item = self._table.item(row, col)
            tip = full_config if col == _COL_CONFIG and full_config else ""
            if item is None:
                cell = _SortableItem(text)
                cell.setFlags(cell.flags() & ~Qt.ItemFlag.ItemIsEditable)
                cell.setTextAlignment(align)
                cell.setData(_SORT_VALUE_ROLE, sval)
                cell.setToolTip(tip)
                self._table.setItem(row, col, cell)
            else:
                item.setText(text)
                item.setData(_SORT_VALUE_ROLE, sval)
                item.setToolTip(tip)

    def _set_session_row_widgets(
        self,
        row: int,
        sid: str,
        meta: SessionMeta | None,
        *,
        use_checked: bool,
    ) -> None:
        use = QTableWidgetItem()
        use.setFlags(
            (use.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            & ~Qt.ItemFlag.ItemIsEditable
        )
        use.setData(_SESSION_ROLE, sid)
        use.setCheckState(
            Qt.CheckState.Checked if use_checked else Qt.CheckState.Unchecked
        )
        self._table.setItem(row, _COL_USE, use)

        id_cell = QTableWidgetItem(sid)
        id_cell.setFlags(id_cell.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self._table.setItem(row, _COL_SESSION_ID, id_cell)

        self._fill_meta_columns(row, meta)

        note = self._notes.get(sid, "")
        note_cell = QTableWidgetItem(note)
        if self._editable_notes:
            note_cell.setFlags(
                (note_cell.flags() | Qt.ItemFlag.ItemIsEditable | Qt.ItemFlag.ItemIsSelectable)
                & ~Qt.ItemFlag.ItemIsUserCheckable
            )
        else:
            note_cell.setFlags(note_cell.flags() & ~Qt.ItemFlag.ItemIsEditable)
        note_cell.setTextAlignment(
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft
        )
        note_cell.setToolTip(note if len(note) > 120 else "")
        self._table.setItem(row, _COL_NOTE, note_cell)

    def _session_row_sid(self, row: int) -> str | None:
        if row < 0 or row >= self._table.rowCount():
            return None
        use = self._table.item(row, _COL_USE)
        if use is None:
            return None
        sid = use.data(_SESSION_ROLE)
        return str(sid) if sid is not None else None

    def _find_row_for_sid(self, sid: str) -> int:
        row = self._row_by_sid.get(sid, -1)
        if 0 <= row < self._table.rowCount() and self._session_row_sid(row) == sid:
            return row
        for candidate in range(self._table.rowCount()):
            if self._session_row_sid(candidate) == sid:
                self._row_by_sid[sid] = candidate
                return candidate
        return -1

    def _set_row_checked(self, sid: str, checked: bool) -> None:
        row = self._find_row_for_sid(sid)
        if row < 0:
            return
        use = self._table.item(row, _COL_USE)
        if use is None:
            return
        self._table.blockSignals(True)
        try:
            use.setCheckState(
                Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
            )
        finally:
            self._table.blockSignals(False)

    def _checked_sids_ordered(self) -> list[str]:
        out: list[str] = []
        for row in range(self._table.rowCount()):
            use = self._table.item(row, _COL_USE)
            if use is not None and use.checkState() == Qt.CheckState.Checked:
                sid = use.data(_SESSION_ROLE)
                if sid is not None:
                    out.append(str(sid))
        return out

    def _on_table_item_changed(self, item: QTableWidgetItem) -> None:
        col = item.column()
        if col == _COL_USE:
            sid_raw = item.data(_SESSION_ROLE)
            if sid_raw is None:
                return
            sid = str(sid_raw)
            if item.checkState() == Qt.CheckState.Checked:
                if sid not in self._check_order:
                    self._check_order.append(sid)
                while len(self._check_order) > self._max_selections:
                    drop = self._check_order.pop(0)
                    self._set_row_checked(drop, False)
            else:
                self._check_order = [s for s in self._check_order if s != sid]
            selected_set = set(self._checked_sids_ordered())
            self._check_order = [s for s in self._check_order if s in selected_set]
            self._persist_selection()
            self._schedule_status_refresh()
            return
        if col == _COL_NOTE and self._editable_notes:
            self._persist_note_from_cell(item)

    def _persist_note_from_cell(self, item: QTableWidgetItem) -> None:
        sid = self._session_row_sid(item.row())
        if sid is None:
            return
        text = item.text()
        old = self._notes.get(sid, "")
        if text == old:
            return
        # push() runs the command's redo(), which performs the actual edit.
        self._undo_stack.push(_NoteEditCommand(self, sid, old, text))

    def _apply_note(self, sid: str, text: str) -> None:
        """Set a note in the store, on disk, and in its table cell."""
        if text.strip():
            self._notes[sid] = text
        else:
            self._notes.pop(sid, None)
        try:
            set_session_note(self._base_dir, sid, text)
        except Exception:
            pass
        self._set_note_cell_text(sid, text)

    def _set_note_cell_text(self, sid: str, text: str) -> None:
        row = self._find_row_for_sid(sid)
        if row < 0:
            return
        item = self._table.item(row, _COL_NOTE)
        if item is None:
            return
        self._table.blockSignals(True)
        try:
            if item.text() != text:
                item.setText(text)
            item.setToolTip(text if len(text) > 120 else "")
        finally:
            self._table.blockSignals(False)

    def undo(self) -> bool:
        """Reverse the most recent note edit. Returns ``False`` if none remain."""
        if not self._undo_stack.canUndo():
            return False
        self._undo_stack.undo()
        return True

    def redo(self) -> bool:
        """Reapply the most recently undone note edit."""
        if not self._undo_stack.canRedo():
            return False
        self._undo_stack.redo()
        return True

    def _on_current_cell_changed(
        self,
        current_row: int,
        _current_col: int,
        _prev_row: int,
        _prev_col: int,
    ) -> None:
        if current_row < 0:
            self._highlighted_sid = None
            self.highlighted_session_changed.emit(None)
            return
        sid = self._session_row_sid(current_row)
        if sid is None:
            return
        self._highlighted_sid = sid
        self.highlighted_session_changed.emit(sid)

    def _refresh_use_column_swatches(self, selected: list[str] | None = None) -> None:
        if not self._show_plot_swatches:
            return
        if selected is None:
            selected = self.selected_session_ids()
        n_selected = len(selected)
        idx_by_sid = {sid: i for i, sid in enumerate(selected)}
        self._table.blockSignals(True)
        try:
            for row in range(self._table.rowCount()):
                use = self._table.item(row, _COL_USE)
                if use is None:
                    continue
                sid_raw = use.data(_SESSION_ROLE)
                if sid_raw is None:
                    continue
                sid = str(sid_raw)
                active = use.checkState() == Qt.CheckState.Checked
                plot_i = idx_by_sid.get(sid) if active else None
                if active and plot_i is not None:
                    state_key = (True, plot_i, n_selected)
                elif active:
                    state_key = (True, -1, n_selected)
                else:
                    state_key = (False, -1, 0)
                if use.data(_SWATCH_STATE_ROLE) == state_key:
                    continue
                use.setData(_SWATCH_STATE_ROLE, state_key)
                use.setIcon(_cached_swatch_icon(active=active, plot_index=plot_i))
                if active and plot_i is not None:
                    cname = DEFAULT_SESSION_COLORS[
                        plot_i % len(DEFAULT_SESSION_COLORS)
                    ]
                    use.setToolTip(
                        f"Session color in plots: {cname} ({plot_i + 1} of {n_selected})"
                    )
                elif active:
                    use.setToolTip("Selected for analysis")
                else:
                    use.setToolTip(
                        "Not used in plots — check to assign a plot color by order"
                    )
        finally:
            self._table.blockSignals(False)

    def _update_status(self) -> None:
        selected = self.selected_session_ids()
        self._refresh_use_column_swatches(selected)
        self.selection_changed.emit(selected)

    def _persist_selection(self) -> None:
        ids = self.selected_session_ids()
        try:
            set_selected_sessions(self._base_dir, ids)
        except Exception:
            pass
        if self._selection_persist is not None:
            self._selection_persist(ids)
