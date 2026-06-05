"""Configuration Tuning panel — browse and edit XML config files."""

from __future__ import annotations

import html
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QLineEdit,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from scan_kit.common.app_settings import AppSettings

from scan_kit.common.file_integrity import is_sidecar_path

from .auto_tune_panel import AutoTuneDetailWidget, AutoTuneListWidget
from .auto_tuning.base import AutoTuneRunResult, AutoTuneWorkflow
from .auto_tuning.paths import resolve_devices_xml_path
from .file_tree import XmlFileTreeWidget
from scan_kit.common.file_integrity import verify_file_integrity

from .integrity_view import (
    FileIntegrityWidget,
    build_sidecar_only_report,
    integrity_badge_markup,
)
from .config_folder_io import save_config_folder
from .save_folder_dialog import SaveConfigFolderDialog
from .map2map_attr_registry import is_map2map_config_path
from .xml_document import XmlDocument, XmlParseError
from .xml_form import XmlFormWidget

_UNSAVED_MARKUP = (
    '<span style="color:#e6a700; font-weight:700; font-size:110%;">●</span>'
)
_UNSAVED_TOOLTIP = "Unsaved changes in this configuration"

_VIEW_XML = 0
_VIEW_WORKFLOW = 1


class ConfigTuningPanel(QWidget):
    """Browse a config folder and edit XML files with an auto-generated form."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._settings = AppSettings.load()
        self._config_root: Path | None = None
        self._document: XmlDocument | None = None
        self._current_path: Path | None = None
        self._form: XmlFormWidget | None = None
        self._open_documents: dict[Path, XmlDocument] = {}
        self._current_workflow: AutoTuneWorkflow | None = None

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

        self._hide_unused_cb = QCheckBox("Hide unused map2map XML")
        self._hide_unused_cb.setToolTip(
            "Hides attributes and elements that the Pyramid map2map library never reads "
            "(e.g. e0, m2, zero_offset_mm)."
        )
        self._hide_unused_cb.setChecked(self._settings.hide_unused_map2map_xml)
        self._hide_unused_cb.toggled.connect(self._on_hide_unused_toggled)
        left_layout.addWidget(self._hide_unused_cb)

        left_splitter = QSplitter(Qt.Orientation.Vertical)

        file_tree_host = QWidget()
        file_tree_layout = QVBoxLayout(file_tree_host)
        file_tree_layout.setContentsMargins(0, 0, 0, 0)
        file_tree_layout.setSpacing(4)
        files_heading = QLabel("Configuration files")
        files_heading.setStyleSheet("font-weight: 600;")
        file_tree_layout.addWidget(files_heading)
        self._file_tree = XmlFileTreeWidget()
        self._file_tree.file_selected.connect(self._on_file_selected)
        file_tree_layout.addWidget(self._file_tree)
        left_splitter.addWidget(file_tree_host)

        self._auto_tune_list = AutoTuneListWidget()
        self._auto_tune_list.workflow_selected.connect(self._on_workflow_selected)
        left_splitter.addWidget(self._auto_tune_list)

        left_splitter.setStretchFactor(0, 3)
        left_splitter.setStretchFactor(1, 2)
        left_splitter.setSizes([280, 160])

        left_layout.addWidget(left_splitter, stretch=1)

        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 4, 4, 4)

        header_row = QHBoxLayout()
        self._header_base_title = "Select a configuration file or auto-tuning workflow"
        self._header_label = QLabel(self._header_base_title)
        self._header_label.setWordWrap(True)
        self._header_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        header_row.addWidget(self._header_label, 1)

        self._unsaved_indicator = QLabel()
        self._unsaved_indicator.setTextFormat(Qt.TextFormat.RichText)
        self._unsaved_indicator.setToolTip(_UNSAVED_TOOLTIP)
        self._unsaved_indicator.hide()
        header_row.addWidget(self._unsaved_indicator, 0, Qt.AlignmentFlag.AlignRight)

        self._revert_btn = QPushButton("Revert")
        self._revert_btn.setEnabled(False)
        self._revert_btn.setToolTip("Discard all unsaved edits and reload from disk")
        self._revert_btn.clicked.connect(self._on_revert)
        header_row.addWidget(self._revert_btn, 0, Qt.AlignmentFlag.AlignRight)

        self._save_as_btn = QPushButton("Save As…")
        self._save_as_btn.setEnabled(False)
        self._save_as_btn.setToolTip(
            "Save the whole configuration folder to a new or existing folder. "
            "Type a new folder name or use New folder in the dialog. "
            "Select the same folder as the open config to overwrite on disk."
        )
        self._save_as_btn.clicked.connect(self._on_save_as)
        header_row.addWidget(self._save_as_btn, 0, Qt.AlignmentFlag.AlignRight)

        right_layout.addLayout(header_row)

        self._right_stack = QStackedWidget()

        self._xml_page = QWidget()
        xml_page_layout = QVBoxLayout(self._xml_page)
        xml_page_layout.setContentsMargins(0, 0, 0, 0)
        self._form_host = QWidget()
        self._form_layout = QVBoxLayout(self._form_host)
        self._form_layout.setContentsMargins(0, 0, 0, 0)
        self._integrity_details = FileIntegrityWidget()
        self._form_layout.addWidget(
            self._integrity_details,
            0,
            Qt.AlignmentFlag.AlignTop,
        )
        xml_page_layout.addWidget(self._form_host, stretch=1)
        self._right_stack.addWidget(self._xml_page)

        self._auto_tune_detail = AutoTuneDetailWidget()
        self._auto_tune_detail.set_apply_handler(self._on_auto_tune_apply)
        self._right_stack.addWidget(self._auto_tune_detail)

        right_layout.addWidget(self._right_stack, stretch=1)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 35)
        splitter.setStretchFactor(1, 65)
        splitter.setSizes([320, 680])
        root_layout.addWidget(splitter, stretch=1)

        self._auto_tune_list.select_first()

        if self._settings.config_dir:
            self._apply_config_root(Path(self._settings.config_dir))

    def set_session_data_dir(self, path: str) -> None:
        """Default folder for session-based auto-tuning workflows."""
        self._auto_tune_detail.set_default_data_dir(path)

    def _show_xml_view(self) -> None:
        self._right_stack.setCurrentIndex(_VIEW_XML)
        self._hide_unused_cb.setEnabled(True)

    def _show_workflow_view(self, workflow: AutoTuneWorkflow) -> None:
        self._current_workflow = workflow
        self._auto_tune_detail.set_workflow(workflow)
        self._right_stack.setCurrentIndex(_VIEW_WORKFLOW)
        self._hide_unused_cb.setEnabled(False)
        self._set_header_title(workflow.name)
        self._header_label.setTextFormat(Qt.TextFormat.PlainText)
        self._header_label.setToolTip(workflow.description)
        self._update_action_state()

    def _on_workflow_selected(self, workflow: object) -> None:
        if not isinstance(workflow, AutoTuneWorkflow):
            self._current_workflow = None
            self._auto_tune_detail.set_workflow(None)
            if self._current_path is None:
                self._set_header_title("Select a configuration file or auto-tuning workflow")
            return
        self._flush_current_editor()
        self._file_tree.clear_selection()
        self._show_workflow_view(workflow)

    def _on_auto_tune_apply(
        self,
        workflow: AutoTuneWorkflow,
        params: dict,
    ) -> AutoTuneRunResult | None:
        if self._config_root is None:
            QMessageBox.warning(
                self,
                "Auto tuning",
                "Open a configuration folder first.",
            )
            return None

        devices_path = resolve_devices_xml_path(self._config_root)
        if devices_path is None:
            QMessageBox.warning(
                self,
                "Auto tuning",
                "Could not find map2map/devices.xml in this configuration folder.",
            )
            return None

        resolved = devices_path.resolve()
        self._flush_current_editor()
        document = self._open_documents.get(resolved)
        if document is None:
            try:
                document = XmlDocument.load(resolved)
            except XmlParseError as exc:
                QMessageBox.critical(
                    self,
                    "Parse error",
                    f"Could not parse devices.xml:\n{exc}",
                )
                return None
            self._open_documents[resolved] = document

        result = workflow.apply_to_root(document.root, params)
        if result.success:
            document.mark_dirty()
            self._auto_tune_list.clear_selection()
            self._open_file(resolved)
            self._update_action_state()
        return result

    def confirm_discard_if_dirty(self) -> bool:
        """Return True if it is safe to proceed (no unsaved config edits or user discards)."""
        if not self._has_unsaved_changes():
            return True
        answer = QMessageBox.question(
            self,
            "Unsaved changes",
            "Discard unsaved changes to this configuration?",
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
        self._open_documents.clear()
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
                return

        if self._current_workflow is None:
            self._auto_tune_list.select_first()

    def _on_file_selected(self, file_path: str) -> None:
        target = Path(file_path).resolve()
        if self._current_path == target:
            return
        self._auto_tune_list.clear_selection()
        self._current_workflow = None
        self._auto_tune_detail.set_workflow(None)
        self._flush_current_editor()
        if is_sidecar_path(target):
            self._open_sidecar(target)
        else:
            self._open_file(target)

    def _open_sidecar(self, path: Path) -> None:
        """Show parsed sidecar fields and verify against the paired XML file."""
        self._document = None
        self._current_path = path.resolve()
        self._show_xml_view()
        self._clear_form_widget()
        self._update_action_state()

        rel = self._relative_path(self._current_path)
        self._set_header_title(f"{rel or path.name}")
        report = build_sidecar_only_report(path)
        self._integrity_details.set_report(report)
        self._ensure_sidecar_tail_spacer()

        if self._config_root is not None:
            self._settings.last_opened_xml = rel
            self._settings.save()

    def _open_file(self, path: Path) -> None:
        resolved = path.resolve()
        document = self._open_documents.get(resolved)
        if document is None:
            try:
                document = XmlDocument.load(path)
            except XmlParseError as exc:
                QMessageBox.critical(self, "Parse error", f"Could not parse XML:\n{exc}")
                return
            self._open_documents[resolved] = document

        self._document = document
        self._current_path = path.resolve()
        self._current_workflow = None
        self._auto_tune_detail.set_workflow(None)
        self._show_xml_view()
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
        if self._right_stack.currentIndex() == _VIEW_WORKFLOW:
            return
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

    def _hide_unused_map2map_enabled(self) -> bool:
        if not self._hide_unused_cb.isChecked():
            return False
        if self._current_path is None:
            return True
        return is_map2map_config_path(str(self._current_path))

    def _on_hide_unused_toggled(self, checked: bool) -> None:
        self._settings.hide_unused_map2map_xml = checked
        self._settings.save()
        if self._document is not None and self._current_path is not None:
            self._set_form(self._document)

    def _set_form(self, document: XmlDocument) -> None:
        self._clear_form_widget()
        self._remove_sidecar_tail_spacer()
        self._integrity_details.hide()
        hide_unused = self._hide_unused_map2map_enabled()
        self._form = XmlFormWidget(
            document.root,
            on_change=self._on_form_changed,
            hide_unused_map2map=hide_unused,
        )
        self._form.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self._form_layout.addWidget(self._form, stretch=1)
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
        self._current_workflow = None
        self._auto_tune_detail.set_workflow(None)
        self._clear_form_widget()
        self._remove_sidecar_tail_spacer()
        self._set_header_title("Select a configuration file or auto-tuning workflow")
        self._integrity_details.set_report(None)
        self._show_xml_view()
        self._update_action_state()

    def _relative_path(self, path: Path) -> str | None:
        if self._config_root is None:
            return None
        try:
            return path.resolve().relative_to(self._config_root.resolve()).as_posix()
        except ValueError:
            return None

    def _flush_current_editor(self) -> None:
        if self._form is None or self._document is None:
            return
        self._form.apply_to_dom()

    def _has_unsaved_changes(self) -> bool:
        return any(doc.dirty for doc in self._open_documents.values())

    def _on_form_changed(self) -> None:
        if self._document is not None:
            self._document.mark_dirty()
        self._update_action_state()

    def _update_unsaved_indicator(self) -> None:
        if self._has_unsaved_changes():
            self._unsaved_indicator.setText(_UNSAVED_MARKUP)
            self._unsaved_indicator.show()
        else:
            self._unsaved_indicator.hide()

    def _update_action_state(self) -> None:
        has_config = self._config_root is not None
        has_xml = self._document is not None
        self._save_as_btn.setEnabled(has_config)
        self._revert_btn.setEnabled(has_config and (has_xml or self._has_unsaved_changes()))
        self._update_unsaved_indicator()

    def _on_revert(self) -> None:
        if self._config_root is None:
            return
        if not self._has_unsaved_changes():
            if self._document is None:
                return
            self._document.revert()
            self._set_form(self._document)
            self._update_integrity_badge()
            return
        answer = QMessageBox.question(
            self,
            "Revert changes",
            "Discard all unsaved changes in this configuration and reload from disk?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        for doc in self._open_documents.values():
            if doc.dirty:
                doc.revert()
        if self._document is not None:
            self._set_form(self._document)
            self._update_integrity_badge()

    def _on_save_as(self) -> None:
        if self._config_root is None:
            return
        self._flush_current_editor()

        dest = self._pick_save_config_directory()
        if dest is None:
            return
        source = self._config_root.resolve()
        if dest != source and dest.exists() and any(dest.iterdir()):
            answer = QMessageBox.question(
                self,
                "Save configuration",
                f"Folder already exists:\n{dest}\n\n"
                "Merge your edits into this folder?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return

        try:
            save_config_folder(
                source,
                dest,
                dirty_by_path=self._open_documents,
            )
        except OSError as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return

        if dest != source:
            self._remap_open_documents(source, dest)
            self._config_root = dest
            self._path_input.setText(str(dest))
            self._file_tree.set_root(dest)
            self._settings.config_dir = str(dest)
            self._settings.save()

        self._update_action_state()
        self._update_integrity_badge()
        if self._document is not None:
            rel = self._relative_path(self._document.path)
            self._set_header_title(f"{rel or self._document.path.name}  (saved)")

    def _pick_save_config_directory(self) -> Path | None:
        """Let the user name a new config folder or choose an existing one."""
        assert self._config_root is not None
        source = self._config_root.resolve()
        parent_dir = source.parent
        suggested = f"{source.name}_copy"

        dialog = SaveConfigFolderDialog(
            self,
            caption="Save configuration folder as…",
            start_dir=parent_dir,
            suggested_name=suggested,
        )
        if dialog.exec() != QFileDialog.DialogCode.Accepted:
            return None
        dest = dialog.selected_directory()
        if dest is None:
            return None
        if not dest.exists():
            dest.mkdir(parents=True, exist_ok=True)
        return dest

    def _remap_open_documents(self, old_root: Path, new_root: Path) -> None:
        remapped: dict[Path, XmlDocument] = {}
        for path, doc in self._open_documents.items():
            try:
                rel = path.relative_to(old_root)
            except ValueError:
                continue
            new_path = (new_root / rel).resolve()
            doc.path = new_path
            remapped[new_path] = doc
        self._open_documents = remapped
        if self._current_path is not None:
            try:
                rel = self._current_path.relative_to(old_root)
                self._current_path = (new_root / rel).resolve()
                self._document = self._open_documents.get(self._current_path)
            except ValueError:
                pass
