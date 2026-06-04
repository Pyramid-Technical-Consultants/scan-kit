"""Configuration Tuning panel — browse and edit XML config files."""

from __future__ import annotations

import html
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QLineEdit,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from scan_kit.common.app_settings import AppSettings

from scan_kit.common.file_integrity import is_sidecar_path

from .file_tree import XmlFileTreeWidget
from scan_kit.common.file_integrity import verify_file_integrity

from .integrity_view import (
    FileIntegrityWidget,
    build_sidecar_only_report,
    integrity_badge_markup,
)
from .xml_document import XmlDocument, XmlParseError
from .xml_form import XmlFormWidget


class ConfigTuningPanel(QWidget):
    """Browse a config folder and edit XML files with an auto-generated form."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._settings = AppSettings.load()
        self._config_root: Path | None = None
        self._document: XmlDocument | None = None
        self._current_path: Path | None = None
        self._form: XmlFormWidget | None = None

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.setSpacing(6)

        path_row = QHBoxLayout()
        self._path_input = QLineEdit()
        self._path_input.setPlaceholderText("Config folder…")
        if self._settings.config_dir:
            self._path_input.setText(self._settings.config_dir)
        self._path_input.editingFinished.connect(self._on_path_finished)
        self._path_input.returnPressed.connect(self._on_path_finished)
        path_row.addWidget(self._path_input, stretch=1)

        browse_btn = QPushButton("Browse…")
        browse_btn.setFixedWidth(96)
        browse_btn.clicked.connect(self._on_browse_config_dir)
        path_row.addWidget(browse_btn)
        left_layout.addLayout(path_row)

        self._file_tree = XmlFileTreeWidget()
        self._file_tree.file_selected.connect(self._on_file_selected)
        left_layout.addWidget(self._file_tree, stretch=1)

        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 4, 4, 4)

        header_row = QHBoxLayout()
        self._header_base_title = "Select an XML file"
        self._header_label = QLabel(self._header_base_title)
        self._header_label.setWordWrap(True)
        self._header_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        header_row.addWidget(self._header_label, 1)

        self._revert_btn = QPushButton("Revert")
        self._revert_btn.setEnabled(False)
        self._revert_btn.clicked.connect(self._on_revert)
        header_row.addWidget(self._revert_btn, 0, Qt.AlignmentFlag.AlignRight)

        self._save_btn = QPushButton("Save")
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._on_save)
        header_row.addWidget(self._save_btn, 0, Qt.AlignmentFlag.AlignRight)

        right_layout.addLayout(header_row)

        self._form_host = QWidget()
        self._form_layout = QVBoxLayout(self._form_host)
        self._form_layout.setContentsMargins(0, 0, 0, 0)
        self._integrity_details = FileIntegrityWidget()
        self._form_layout.addWidget(
            self._integrity_details,
            0,
            Qt.AlignmentFlag.AlignTop,
        )
        right_layout.addWidget(self._form_host, stretch=1)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 35)
        splitter.setStretchFactor(1, 65)
        splitter.setSizes([320, 680])
        root_layout.addWidget(splitter, stretch=1)

        if self._settings.config_dir:
            self._apply_config_root(Path(self._settings.config_dir))

    def confirm_discard_if_dirty(self) -> bool:
        """Return True if it is safe to proceed (no dirty doc or user discards)."""
        if self._document is None or not self._document.dirty:
            return True
        answer = QMessageBox.question(
            self,
            "Unsaved changes",
            "Discard unsaved changes?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _on_browse_config_dir(self) -> None:
        start = self._path_input.text().strip()
        initial = str(Path(start).expanduser().resolve()) if start else str(Path.home())
        chosen = QFileDialog.getExistingDirectory(
            self,
            "Select config folder",
            initial,
        )
        if not chosen:
            return
        self._path_input.setText(chosen)
        self._on_path_finished()

    def _on_path_finished(self) -> None:
        text = self._path_input.text().strip()
        if not text:
            return
        path = Path(text).expanduser()
        if not path.is_dir():
            QMessageBox.warning(self, "Invalid folder", f"Not a directory:\n{path}")
            return
        if not self.confirm_discard_if_dirty():
            if self._config_root is not None:
                self._path_input.setText(str(self._config_root))
            return
        self._apply_config_root(path.resolve())

    def _apply_config_root(self, path: Path) -> None:
        self._config_root = path
        self._path_input.setText(str(path))
        self._file_tree.set_root(path)
        self._settings.config_dir = str(path)
        self._settings.save()
        self._clear_editor()

        if self._settings.last_opened_xml:
            rel = self._settings.last_opened_xml
            candidate = path / rel
            if candidate.is_file():
                self._file_tree.select_relative_path(rel)
                self._open_file(candidate)

    def _on_file_selected(self, file_path: str) -> None:
        target = Path(file_path).resolve()
        if self._current_path == target:
            return
        if not self.confirm_discard_if_dirty():
            if self._current_path is not None:
                rel = self._relative_path(self._current_path)
                if rel is not None:
                    self._file_tree.select_relative_path(rel)
            return
        if is_sidecar_path(target):
            self._open_sidecar(target)
        else:
            self._open_file(target)

    def _open_sidecar(self, path: Path) -> None:
        """Show parsed sidecar fields and verify against the paired XML file."""
        self._document = None
        self._current_path = path.resolve()
        self._clear_form_widget()
        self._save_btn.setEnabled(False)
        self._revert_btn.setEnabled(False)

        rel = self._relative_path(self._current_path)
        self._set_header_title(f"{rel or path.name}")
        report = build_sidecar_only_report(path)
        self._integrity_details.set_report(report)
        self._ensure_sidecar_tail_spacer()

        if self._config_root is not None:
            self._settings.last_opened_xml = rel
            self._settings.save()

    def _open_file(self, path: Path) -> None:
        try:
            document = XmlDocument.load(path)
        except XmlParseError as exc:
            QMessageBox.critical(self, "Parse error", f"Could not parse XML:\n{exc}")
            return

        self._document = document
        self._current_path = path.resolve()
        rel = self._relative_path(self._current_path)
        if rel is not None:
            self._settings.last_opened_xml = rel
            self._settings.save()

        self._set_header_title(str(rel or self._current_path.name))
        self._set_form(document)
        self._update_integrity_badge()

    def _set_header_title(self, title: str) -> None:
        self._header_base_title = title
        self._render_header_label()

    def _update_integrity_badge(self) -> None:
        self._render_header_label()

    def _render_header_label(self) -> None:
        title = self._header_base_title
        if self._current_path is None or is_sidecar_path(self._current_path):
            self._header_label.setTextFormat(Qt.TextFormat.PlainText)
            self._header_label.setToolTip("")
            self._header_label.setText(title)
            return
        result = verify_file_integrity(self._current_path)
        glyph, color, tooltip = integrity_badge_markup(result.status)
        safe_title = html.escape(title)
        self._header_label.setTextFormat(Qt.TextFormat.RichText)
        self._header_label.setText(
            f'{safe_title}&nbsp;<span style="color:{color}; '
            f'font-weight:700; font-size:110%;">{glyph}</span>'
        )
        self._header_label.setToolTip(tooltip)

    def _set_form(self, document: XmlDocument) -> None:
        self._clear_form_widget()
        self._remove_sidecar_tail_spacer()
        self._integrity_details.hide()
        self._form = XmlFormWidget(document.root, on_change=self._on_form_changed)
        self._form.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self._form_layout.addWidget(self._form, stretch=1)
        document.mark_clean()
        self._update_action_state()

    def _clear_form_widget(self) -> None:
        if self._form is not None:
            self._form.setParent(None)
            self._form.deleteLater()
            self._form = None

    def _has_sidecar_tail_spacer(self) -> bool:
        count = self._form_layout.count()
        if count == 0:
            return False
        last = self._form_layout.itemAt(count - 1)
        return last is not None and last.spacerItem() is not None

    def _ensure_sidecar_tail_spacer(self) -> None:
        if not self._has_sidecar_tail_spacer():
            self._form_layout.addStretch(1)

    def _remove_sidecar_tail_spacer(self) -> None:
        if self._has_sidecar_tail_spacer():
            self._form_layout.takeAt(self._form_layout.count() - 1)

    def _clear_editor(self) -> None:
        self._document = None
        self._current_path = None
        self._clear_form_widget()
        self._remove_sidecar_tail_spacer()
        self._set_header_title("Select an XML file")
        self._integrity_details.set_report(None)
        self._update_action_state()

    def _relative_path(self, path: Path) -> str | None:
        if self._config_root is None:
            return None
        try:
            return path.resolve().relative_to(self._config_root.resolve()).as_posix()
        except ValueError:
            return None

    def _on_form_changed(self) -> None:
        if self._document is not None:
            self._document.mark_dirty()
        self._update_action_state()

    def _update_action_state(self) -> None:
        dirty = bool(self._document and self._document.dirty)
        has_file = self._document is not None
        self._save_btn.setEnabled(has_file and dirty)
        self._revert_btn.setEnabled(has_file)

    def _on_revert(self) -> None:
        if self._document is None:
            return
        if self._document.dirty:
            answer = QMessageBox.question(
                self,
                "Revert changes",
                "Reload this file from disk and discard edits?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        path = self._document.path
        self._document.revert()
        self._set_form(self._document)
        self._current_path = path
        self._update_integrity_badge()

    def _on_save(self) -> None:
        if self._document is None or self._form is None:
            return
        try:
            self._form.apply_to_dom()
            self._document.save()
        except OSError as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        self._update_action_state()
        self._update_integrity_badge()
        self._set_header_title(
            f"{self._relative_path(self._document.path) or self._document.path.name}  (saved)"
        )
