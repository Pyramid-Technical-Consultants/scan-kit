"""Interactive session log compare and explorer (Qt).

Select one or two sessions in the launcher. With two sessions, the view
highlights timeline differences, error templates, and message-count deltas.
"""

from __future__ import annotations

import sys
from collections import Counter
from datetime import datetime

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..common.app_icon import apply_qt_application_branding, prepare_qt_app_identity
from ..common.plot_colors import DEFAULT_SESSION_COLORS
from ..common.session_log import (
    SessionLogData,
    compare_template_counts,
    is_noise_message,
    load_session_log,
    merged_layer_ids,
)
from ..common.view_runner import _READY_SENTINEL

_LEVEL_ORDER = ("ERROR", "WARNING", "WARN", "INFO", "DEBUG")
_LEVEL_FILTER = ("All levels", "ERROR", "WARNING+ (no DEBUG)", "Notable (no DEBUG, no noise)")


def _fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {sec}s"
    return f"{minutes}m {sec}s"


def _fmt_ts(ts: datetime | None) -> str:
    if ts is None:
        return "—"
    return ts.strftime("%Y-%m-%d %H:%M:%S")


class _SummaryPanel(QGroupBox):
    def __init__(self, title: str, color: str, parent: QWidget | None = None) -> None:
        super().__init__(title, parent)
        self._color = color
        grid = QGridLayout(self)
        self._labels: dict[str, QLabel] = {}
        fields = (
            ("Lines", "lines"),
            ("Span", "span"),
            ("Duration", "duration"),
            ("Errors", "errors"),
            ("Layers scanned", "layers"),
            ("WDT mismatches", "wdt"),
        )
        for row, (label, key) in enumerate(fields):
            name = QLabel(label + ":")
            name.setStyleSheet("color: #555;")
            value = QLabel("—")
            value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            grid.addWidget(name, row, 0)
            grid.addWidget(value, row, 1)
            self._labels[key] = value
        accent = QLabel()
        accent.setFixedHeight(4)
        accent.setStyleSheet(f"background: {color}; border-radius: 2px;")
        outer = QVBoxLayout()
        outer.addWidget(accent)
        wrap = QWidget()
        wrap.setLayout(grid)
        outer.addWidget(wrap)
        self.setLayout(outer)

    def set_data(self, log: SessionLogData | None) -> None:
        if log is None:
            for lbl in self._labels.values():
                lbl.setText("—")
            return
        wdt_total = sum(log.wdt_mismatches.values())
        span = f"{_fmt_ts(log.start_time)} → {_fmt_ts(log.end_time)}"
        self._labels["lines"].setText(f"{len(log.entries):,}")
        self._labels["span"].setText(span)
        self._labels["duration"].setText(_fmt_duration(log.duration_s))
        self._labels["errors"].setText(f"{log.error_count:,}")
        self._labels["layers"].setText(str(log.layers_scanned))
        self._labels["wdt"].setText(f"{wdt_total:,}" if wdt_total else "0")


def _make_table(columns: list[str], parent: QWidget | None = None) -> QTableWidget:
    table = QTableWidget(0, len(columns), parent)
    table.setHorizontalHeaderLabels(columns)
    table.setAlternatingRowColors(True)
    table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    table.verticalHeader().setVisible(False)
    header = table.horizontalHeader()
    header.setStretchLastSection(True)
    header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
    return table


def _set_cell(table: QTableWidget, row: int, col: int, text: str, *, color: str | None = None) -> None:
    item = QTableWidgetItem(text)
    if color:
        item.setForeground(QColor(color))
    table.setItem(row, col, item)


class SessionLogCompareWindow(QMainWindow):
    """Side-by-side session log analysis."""

    def __init__(
        self,
        logs: list[SessionLogData],
        colors: list[str],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._logs = logs
        self._colors = colors
        compare = len(logs) == 2
        title = "Session Log Compare" if compare else "Session Log Explorer"
        self.setWindowTitle(title)
        self.resize(1180, 780)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        summary_row = QHBoxLayout()
        self._summary_panels: list[_SummaryPanel] = []
        for idx, log in enumerate(logs):
            panel = _SummaryPanel(log.session_id, colors[idx % len(colors)])
            panel.set_data(log)
            summary_row.addWidget(panel)
            self._summary_panels.append(panel)
        if not compare and len(logs) == 1:
            placeholder = _SummaryPanel("(select a second session to compare)", "#cccccc")
            placeholder.setEnabled(False)
            summary_row.addWidget(placeholder)
        root.addLayout(summary_row)

        tabs = QTabWidget()
        root.addWidget(tabs, stretch=1)

        tabs.addTab(self._build_overview_tab(), "Overview")
        tabs.addTab(self._build_layer_tab(), "Layer timeline")
        tabs.addTab(self._build_issues_tab(), "Issues")
        tabs.addTab(self._build_explorer_tab(), "Event browser")
        if compare:
            tabs.addTab(self._build_diff_tab(), "Message diff")

        hint = QLabel(
            "Tip: hide noise to focus on map loads, layer timings, and errors. "
            "Use Message diff to see templates that differ between sessions."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #666; font-size: 11px; padding: 4px 0;")
        root.addWidget(hint)

    def _build_overview_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        self._level_table = _make_table(["Level"] + [log.session_id for log in self._logs])
        layout.addWidget(QLabel("Log level counts"))
        layout.addWidget(self._level_table)

        if len(self._logs) == 2:
            hdr = ["Message template", self._logs[0].session_id, self._logs[1].session_id]
        else:
            hdr = ["Message template", "Count"]
        self._top_msg_table = _make_table(hdr)
        layout.addWidget(QLabel("Top message templates (noise excluded)"))
        layout.addWidget(self._top_msg_table, stretch=1)

        self._fill_overview()
        return page

    def _fill_overview(self) -> None:
        levels_seen: list[str] = []
        for log in self._logs:
            for level in log.level_counts:
                if level not in levels_seen:
                    levels_seen.append(level)
        ordered = [lv for lv in _LEVEL_ORDER if lv in levels_seen]
        ordered += sorted(lv for lv in levels_seen if lv not in _LEVEL_ORDER)

        self._level_table.setRowCount(len(ordered))
        for row, level in enumerate(ordered):
            _set_cell(self._level_table, row, 0, level)
            for col, log in enumerate(self._logs, start=1):
                _set_cell(self._level_table, row, col, str(log.level_counts.get(level, 0)))
        self._level_table.resizeColumnsToContents()

        if len(self._logs) == 2:
            rows = compare_template_counts(self._logs[0], self._logs[1])[:40]
            self._top_msg_table.setRowCount(len(rows))
            for row, (template, ca, cb, _delta) in enumerate(rows):
                _set_cell(self._top_msg_table, row, 0, template)
                _set_cell(self._top_msg_table, row, 1, str(ca))
                _set_cell(self._top_msg_table, row, 2, str(cb))
        else:
            log = self._logs[0]
            top = log.template_counts.most_common(40)
            self._top_msg_table.setRowCount(len(top))
            for row, (template, count) in enumerate(top):
                _set_cell(self._top_msg_table, row, 0, template)
                _set_cell(self._top_msg_table, row, 1, str(count))
        self._top_msg_table.resizeColumnsToContents()

    def _build_layer_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        cols = ["Layer"]
        if len(self._logs) == 2:
            a, b = self._logs
            cols += [
                f"{a.session_id} start map (s)",
                f"{a.session_id} scan (s)",
                f"{b.session_id} start map (s)",
                f"{b.session_id} scan (s)",
                "Δ scan (s)",
            ]
        else:
            log = self._logs[0]
            cols += [
                f"{log.session_id} start map (s)",
                f"{log.session_id} scan (s)",
            ]
        self._layer_table = _make_table(cols)
        layout.addWidget(self._layer_table)
        self._fill_layer_table()
        return page

    def _fill_layer_table(self) -> None:
        layers = merged_layer_ids(*self._logs)
        self._layer_table.setRowCount(len(layers))
        for row, layer in enumerate(layers):
            _set_cell(self._layer_table, row, 0, str(layer))
            if len(self._logs) == 2:
                a_row = self._logs[0].layer_timeline.get(layer)
                b_row = self._logs[1].layer_timeline.get(layer)
                a_start = a_row.start_map_s if a_row else None
                a_scan = a_row.scan_execute_s if a_row else None
                b_start = b_row.start_map_s if b_row else None
                b_scan = b_row.scan_execute_s if b_row else None
                _set_cell(
                    self._layer_table, row, 1,
                    f"{a_start:.3f}" if a_start is not None else "—",
                )
                _set_cell(
                    self._layer_table, row, 2,
                    f"{a_scan:.3f}" if a_scan is not None else "—",
                )
                _set_cell(
                    self._layer_table, row, 3,
                    f"{b_start:.3f}" if b_start is not None else "—",
                )
                _set_cell(
                    self._layer_table, row, 4,
                    f"{b_scan:.3f}" if b_scan is not None else "—",
                )
                if a_scan is not None and b_scan is not None:
                    delta = a_scan - b_scan
                    color = "#b03a2e" if abs(delta) > 0.5 else None
                    _set_cell(self._layer_table, row, 5, f"{delta:+.3f}", color=color)
                else:
                    _set_cell(self._layer_table, row, 5, "—")
            else:
                lrow = self._logs[0].layer_timeline.get(layer)
                _set_cell(
                    self._layer_table, row, 1,
                    f"{lrow.start_map_s:.3f}" if lrow and lrow.start_map_s is not None else "—",
                )
                _set_cell(
                    self._layer_table, row, 2,
                    f"{lrow.scan_execute_s:.3f}" if lrow and lrow.scan_execute_s is not None else "—",
                )
        self._layer_table.resizeColumnsToContents()

    def _build_issues_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        cols = ["Message template"]
        for log in self._logs:
            cols.append(log.session_id)
        if len(self._logs) == 2:
            cols.append("Δ count")
        self._issues_table = _make_table(cols)
        layout.addWidget(QLabel("ERROR-level messages grouped by template"))
        layout.addWidget(self._issues_table, stretch=1)

        self._wdt_table = _make_table(["Device"] + [log.session_id for log in self._logs])
        layout.addWidget(QLabel("Watchdog counter mismatches by device"))
        layout.addWidget(self._wdt_table)
        self._fill_issues()
        return page

    def _fill_issues(self) -> None:
        error_templates: Counter[str] = Counter()
        per_log_errors: list[Counter[str]] = []
        for log in self._logs:
            counts: Counter[str] = Counter()
            for entry in log.entries:
                if entry.level != "ERROR":
                    continue
                counts[entry.template] += 1
            per_log_errors.append(counts)
            error_templates.update(counts.keys())

        templates = sorted(
            error_templates.keys(),
            key=lambda t: sum(c.get(t, 0) for c in per_log_errors),
            reverse=True,
        )[:200]
        self._issues_table.setRowCount(len(templates))
        for row, template in enumerate(templates):
            _set_cell(self._issues_table, row, 0, template)
            counts = [c.get(template, 0) for c in per_log_errors]
            for col, count in enumerate(counts, start=1):
                _set_cell(self._issues_table, row, col, str(count))
            if len(self._logs) == 2:
                delta = counts[0] - counts[1]
                color = "#b03a2e" if delta != 0 else None
                _set_cell(self._issues_table, row, 3, f"{delta:+d}", color=color)
        self._issues_table.resizeColumnsToContents()

        devices: set[str] = set()
        for log in self._logs:
            devices.update(log.wdt_mismatches)
        dev_list = sorted(devices)
        self._wdt_table.setRowCount(len(dev_list))
        for row, device in enumerate(dev_list):
            _set_cell(self._wdt_table, row, 0, device)
            for col, log in enumerate(self._logs, start=1):
                _set_cell(self._wdt_table, row, col, str(log.wdt_mismatches.get(device, 0)))
        self._wdt_table.resizeColumnsToContents()

    def _build_explorer_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        controls = QHBoxLayout()
        self._search = QLineEdit()
        self._search.setPlaceholderText("Filter messages (substring match)…")
        self._level_filter = QComboBox()
        self._level_filter.addItems(_LEVEL_FILTER)
        self._hide_noise = QCheckBox("Hide noise (ACK / command polling)")
        self._hide_noise.setChecked(True)
        self._session_filter = QComboBox()
        self._session_filter.addItem("All sessions")
        for log in self._logs:
            self._session_filter.addItem(log.session_id)
        controls.addWidget(QLabel("Search:"))
        controls.addWidget(self._search, stretch=2)
        controls.addWidget(QLabel("Level:"))
        controls.addWidget(self._level_filter)
        controls.addWidget(self._session_filter)
        controls.addWidget(self._hide_noise)
        layout.addLayout(controls)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self._explorer_tables: list[QTableWidget] = []
        for idx, log in enumerate(self._logs):
            box = QGroupBox(log.session_id)
            box_layout = QVBoxLayout(box)
            table = _make_table(["Time", "Level", "Message"])
            mono = QFont("Consolas")
            mono.setStyleHint(QFont.StyleHint.Monospace)
            table.setFont(mono)
            box_layout.addWidget(table)
            splitter.addWidget(box)
            self._explorer_tables.append(table)
        layout.addWidget(splitter, stretch=1)

        self._search.textChanged.connect(self._refresh_explorer)
        self._level_filter.currentIndexChanged.connect(self._refresh_explorer)
        self._hide_noise.toggled.connect(self._refresh_explorer)
        self._session_filter.currentIndexChanged.connect(self._refresh_explorer_filter_visibility)
        self._refresh_explorer_filter_visibility()
        self._refresh_explorer()
        return page

    def _refresh_explorer_filter_visibility(self) -> None:
        show_all = self._session_filter.currentIndex() == 0
        for idx, table in enumerate(self._explorer_tables):
            parent = table.parentWidget()
            if parent is not None:
                parent.setVisible(show_all or self._session_filter.currentIndex() - 1 == idx)

    def _entry_passes_filter(self, entry, query: str) -> bool:
        if self._hide_noise.isChecked() and is_noise_message(entry.message):
            return False
        mode = self._level_filter.currentText()
        if mode == "ERROR" and entry.level != "ERROR":
            return False
        if mode == "WARNING+ (no DEBUG)" and entry.level == "DEBUG":
            return False
        if mode == "Notable (no DEBUG, no noise)" and entry.level == "DEBUG":
            return False
        if query and query.lower() not in entry.message.lower():
            return False
        return True

    def _refresh_explorer(self) -> None:
        query = self._search.text().strip().lower()
        max_rows = 5000
        for log, table in zip(self._logs, self._explorer_tables):
            rows = [e for e in log.entries if self._entry_passes_filter(e, query)]
            if len(rows) > max_rows:
                rows = rows[:max_rows]
            table.setRowCount(len(rows))
            for row, entry in enumerate(rows):
                _set_cell(table, row, 0, entry.timestamp.strftime("%H:%M:%S.%f")[:-3])
                color = None
                if entry.level == "ERROR":
                    color = "#b03a2e"
                elif entry.level in ("WARN", "WARNING"):
                    color = "#d68910"
                _set_cell(table, row, 1, entry.level, color=color)
                _set_cell(table, row, 2, entry.message, color=color)
            table.resizeColumnsToContents()

    def _build_diff_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        a, b = self._logs
        self._diff_table = _make_table(
            ["Message template", a.session_id, b.session_id, "Δ (A−B)"]
        )
        layout.addWidget(
            QLabel("Templates sorted by largest count difference (noise excluded)")
        )
        layout.addWidget(self._diff_table, stretch=1)
        rows = compare_template_counts(a, b)
        self._diff_table.setRowCount(min(len(rows), 300))
        for row, (template, ca, cb, delta) in enumerate(rows[:300]):
            _set_cell(self._diff_table, row, 0, template)
            _set_cell(self._diff_table, row, 1, str(ca))
            _set_cell(self._diff_table, row, 2, str(cb))
            color = "#b03a2e" if delta != 0 else None
            _set_cell(self._diff_table, row, 3, f"{delta:+d}", color=color)
        self._diff_table.resizeColumnsToContents()
        return page


def run(session_ids: list[str], base_dir: str = "test_data", *, settings=None) -> None:
    """Open the session log compare / explorer window."""
    del settings  # log view does not use calibration settings yet
    if not session_ids:
        print("No sessions selected")
        return

    sids = session_ids[:2]
    logs: list[SessionLogData] = []
    for sid in sids:
        data = load_session_log(sid, base_dir)
        if data is None or not data.entries:
            print(f"  {sid}: SessionLogFile.log not found or empty, skipping")
            continue
        logs.append(data)

    if not logs:
        print("No session logs could be loaded")
        return

    prepare_qt_app_identity()
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    app_icon = apply_qt_application_branding(app)

    colors = [DEFAULT_SESSION_COLORS[i % len(DEFAULT_SESSION_COLORS)] for i in range(len(logs))]
    window = SessionLogCompareWindow(logs, colors)
    if not app_icon.isNull():
        window.setWindowIcon(app_icon)
    window.show()
    print(_READY_SENTINEL, flush=True)
    app.exec()
