"""Multi-step wizard dialog for PDF report generation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from scan_kit.common.qt_widgets import make_pane_scroll_area, set_pane_scroll_widget
from scan_kit.common.session_meta import SessionMeta
from scan_kit.common.settings import ViewSettings

from . import (
    default_report_subtitle,
    report_view_groups,
)
from .naming import suggest_report_filename, suggest_report_title
from .paths import resolve_report_save_dir
from .types import ReportConfig

_CAL_LABELS = {
    "off": "Off",
    "per_session": "Per-Session",
    "constrained": "Constrained",
}

_STEP_LABELS = ("Report details", "Select views", "Output")


def _compact_button(label: str) -> QPushButton:
    button = QPushButton(label)
    button.setStyleSheet("QPushButton { padding: 2px 6px; min-width: 0px; }")
    return button


class ReportWizardDialog(QDialog):
    """Walk the user through report title, view selection, and output path."""

    def __init__(
        self,
        *,
        session_ids: list[str],
        base_dir: str,
        settings: ViewSettings,
        session_meta: dict[str, SessionMeta | None],
        notes: dict[str, str],
        last_report_dir: str | None = None,
        last_report_author: str | None = None,
        last_report_organization: str | None = None,
        last_report_views: list[str] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._session_ids = list(session_ids)
        self._base_dir = base_dir
        self._settings = settings
        self._session_meta = session_meta
        self._notes = notes
        self._last_report_dir = last_report_dir
        self._last_report_author = last_report_author or ""
        self._last_report_organization = last_report_organization or ""
        self._last_report_views = set(last_report_views or [])
        self._config: ReportConfig | None = None
        self._view_checks: dict[str, QCheckBox] = {}
        self._path_input: QLineEdit | None = None
        self._title_input: QLineEdit | None = None
        self._last_auto_title = ""

        self.setWindowTitle("Generate Report")
        self.setModal(True)
        self.resize(720, 560)
        self._build_ui()

    @property
    def config(self) -> ReportConfig | None:
        return self._config

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        self._step_label = QLabel()
        self._step_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        root.addWidget(self._step_label)

        self._stack = QStackedWidget()
        self._stack.addWidget(self._build_details_page())
        self._stack.addWidget(self._build_views_page())
        self._stack.addWidget(self._build_output_page())
        root.addWidget(self._stack, stretch=1)

        nav = QHBoxLayout()
        self._back_btn = QPushButton("Back")
        self._back_btn.clicked.connect(self._on_back)
        nav.addWidget(self._back_btn)
        nav.addStretch(1)

        self._button_box = QDialogButtonBox()
        self._cancel_btn = self._button_box.addButton(
            QDialogButtonBox.StandardButton.Cancel,
        )
        self._next_btn = self._button_box.addButton(
            "Next",
            QDialogButtonBox.ButtonRole.AcceptRole,
        )
        self._cancel_btn.clicked.connect(self.reject)
        self._next_btn.clicked.connect(self._on_next)
        nav.addWidget(self._button_box)
        root.addLayout(nav)

        self._refresh_default_title()
        self._update_step_ui()

    def _refresh_default_title(self) -> None:
        if self._title_input is None:
            return
        auto_title = suggest_report_title(
            self._session_ids,
            self._notes,
            self._selected_views(),
        )
        current = self._title_input.text().strip()
        if not current or current == self._last_auto_title:
            self._title_input.setText(auto_title)
        self._last_auto_title = auto_title

    def _build_details_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        sessions_box = QGroupBox("Selected sessions")
        sessions_layout = QVBoxLayout(sessions_box)
        for sid in self._session_ids:
            meta = self._session_meta.get(sid)
            if meta is None:
                text = sid
            else:
                text = (
                    f"{sid}  —  {meta.short_date}, "
                    f"{meta.short_mu} MU, {meta.short_time}"
                )
            note = self._notes.get(sid, "").strip()
            if note:
                text += f"  ({note})"
            sessions_layout.addWidget(QLabel(text))
        layout.addWidget(sessions_box)

        settings_box = QGroupBox("Analysis settings")
        settings_layout = QVBoxLayout(settings_box)
        bg = "On" if self._settings.bg_subtract else "Off"
        cal = _CAL_LABELS.get(
            self._settings.calibration_mode,
            self._settings.calibration_mode,
        )
        settings_layout.addWidget(QLabel(f"BG subtraction: {bg}"))
        settings_layout.addWidget(QLabel(f"Calibration: {cal}"))
        layout.addWidget(settings_box)

        form_box = QGroupBox("Report information")
        form = QFormLayout(form_box)
        self._title_input = QLineEdit()
        self._subtitle_input = QLineEdit(
            default_report_subtitle(self._session_ids),
        )
        self._author_input = QLineEdit(self._last_report_author)
        self._author_input.setPlaceholderText("Optional author name")
        self._organization_input = QLineEdit(self._last_report_organization)
        self._organization_input.setPlaceholderText("Optional organization")
        form.addRow("Title", self._title_input)
        form.addRow("Subtitle", self._subtitle_input)
        form.addRow("Author", self._author_input)
        form.addRow("Organization", self._organization_input)
        layout.addWidget(form_box)
        layout.addStretch(1)
        return page

    def _build_views_page(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)

        scroll = make_pane_scroll_area()
        inner = QWidget()
        inner_layout = QVBoxLayout(inner)

        for group_title, entries in report_view_groups():
            box = QGroupBox(group_title)
            box_layout = QVBoxLayout(box)

            action_row = QHBoxLayout()
            select_all = _compact_button("Select All")
            clear_all = _compact_button("Clear")
            checks: list[QCheckBox] = []

            def _select_all_handler(group_checks: list[QCheckBox]) -> None:
                for cb in group_checks:
                    cb.setChecked(True)

            def _clear_all_handler(group_checks: list[QCheckBox]) -> None:
                for cb in group_checks:
                    cb.setChecked(False)

            select_all.clicked.connect(
                lambda _checked=False, c=checks: _select_all_handler(c),
            )
            clear_all.clicked.connect(
                lambda _checked=False, c=checks: _clear_all_handler(c),
            )
            action_row.addWidget(select_all)
            action_row.addWidget(clear_all)
            action_row.addStretch(1)
            box_layout.addLayout(action_row)

            for display_name, module_name, _description in entries:
                cb = QCheckBox(display_name)
                cb.setChecked(module_name in self._last_report_views)
                cb.setToolTip(_description)
                checks.append(cb)
                self._view_checks[module_name] = cb
                box_layout.addWidget(cb)

            inner_layout.addWidget(box)

        inner_layout.addStretch(1)
        set_pane_scroll_widget(scroll, inner)
        outer.addWidget(scroll)
        return page

    def _default_output_path(self) -> Path:
        directory = resolve_report_save_dir(self._last_report_dir)
        filename = suggest_report_filename(
            self._session_ids,
            self._notes,
            self._selected_views(),
        )
        return directory / filename

    def _refresh_default_output_path(self) -> None:
        if self._path_input is not None:
            self._path_input.setText(str(self._default_output_path()))

    def _build_output_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        path_row = QHBoxLayout()
        self._path_input = QLineEdit()
        browse_btn = QPushButton("Browse…")
        browse_btn.setFixedWidth(96)
        browse_btn.clicked.connect(self._on_browse_output)
        path_row.addWidget(self._path_input, stretch=1)
        path_row.addWidget(browse_btn)
        layout.addLayout(path_row)

        self._summary_label = QLabel()
        self._summary_label.setWordWrap(True)
        layout.addWidget(self._summary_label)
        layout.addStretch(1)
        return page

    def _on_browse_output(self) -> None:
        current = self._path_input.text().strip()
        if current:
            start = current
        else:
            start = str(self._default_output_path())
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save report as…",
            start,
            "PDF files (*.pdf)",
        )
        if path:
            if not path.lower().endswith(".pdf"):
                path += ".pdf"
            self._path_input.setText(path)

    def _selected_views(self) -> list[tuple[str, str, str]]:
        selected: list[tuple[str, str, str]] = []
        for group_title, entries in report_view_groups():
            del group_title
            for display_name, module_name, description in entries:
                cb = self._view_checks.get(module_name)
                if cb is not None and cb.isChecked():
                    selected.append((display_name, module_name, description))
        return selected

    def _refresh_output_summary(self) -> None:
        views = self._selected_views()
        self._summary_label.setText(
            f"{len(self._session_ids)} session(s), {len(views)} view(s)\n"
            f"Title: {self._title_input.text().strip()}\n"
            f"Output: {self._path_input.text().strip()}"
        )

    def _update_step_ui(self) -> None:
        index = self._stack.currentIndex()
        self._step_label.setText(
            f"Step {index + 1} of {len(_STEP_LABELS)} — {_STEP_LABELS[index]}"
        )
        self._back_btn.setEnabled(index > 0)
        if index == 0:
            self._refresh_default_title()
        if index < len(_STEP_LABELS) - 1:
            self._next_btn.setText("Next")
        else:
            self._next_btn.setText("Generate")
            self._refresh_default_title()
            self._refresh_default_output_path()
            self._refresh_output_summary()

    def _on_back(self) -> None:
        index = self._stack.currentIndex()
        if index > 0:
            self._stack.setCurrentIndex(index - 1)
            self._update_step_ui()

    def _validate_current_step(self) -> bool:
        index = self._stack.currentIndex()
        if index == 0:
            if not self._title_input.text().strip():
                QMessageBox.warning(self, "Missing title", "Enter a report title.")
                return False
        elif index == 1:
            if not self._selected_views():
                QMessageBox.warning(
                    self,
                    "No views selected",
                    "Select at least one view to include in the report.",
                )
                return False
        elif index == 2:
            path_text = self._path_input.text().strip()
            if not path_text:
                QMessageBox.warning(
                    self,
                    "Missing output path",
                    "Choose where to save the PDF report.",
                )
                return False
            if not path_text.lower().endswith(".pdf"):
                QMessageBox.warning(
                    self,
                    "Invalid file type",
                    "The report must be saved as a .pdf file.",
                )
                return False
        return True

    def _on_next(self) -> None:
        if not self._validate_current_step():
            return

        index = self._stack.currentIndex()
        if index < len(_STEP_LABELS) - 1:
            self._stack.setCurrentIndex(index + 1)
            self._update_step_ui()
            return

        self._config = ReportConfig(
            title=self._title_input.text().strip()
            or suggest_report_title(
                self._session_ids,
                self._notes,
                self._selected_views(),
            ),
            subtitle=self._subtitle_input.text().strip(),
            author=self._author_input.text().strip(),
            organization=self._organization_input.text().strip(),
            output_path=Path(self._path_input.text().strip()),
            session_ids=list(self._session_ids),
            base_dir=self._base_dir,
            settings=self._settings,
            views=self._selected_views(),
            session_meta=dict(self._session_meta),
            notes=dict(self._notes),
            generated_at=datetime.now(),
        )
        self.accept()
