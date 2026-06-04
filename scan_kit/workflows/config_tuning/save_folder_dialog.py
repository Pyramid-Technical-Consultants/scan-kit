"""Directory save dialog that accepts new folder names (not only existing paths)."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QFileInfo
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QFileDialog, QLineEdit


class SaveConfigFolderDialog(QFileDialog):
    """Save-style folder picker: existing directories or a new name to create."""

    def __init__(
        self,
        parent=None,
        *,
        caption: str = "Save configuration folder as…",
        start_dir: str | Path = "",
        suggested_name: str = "",
    ) -> None:
        super().__init__(parent)
        self.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        self.setFileMode(QFileDialog.FileMode.Directory)
        self.setOption(QFileDialog.Option.ShowDirsOnly, True)
        self.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        self.setWindowTitle(caption)
        self.setLabelText(QFileDialog.DialogLabel.Accept, "Save")

        start = Path(start_dir).expanduser() if start_dir else Path.cwd()
        self.setDirectory(str(start))
        if suggested_name:
            self.selectFile(suggested_name)

        self._file_name_edit = self.findChild(QLineEdit, "fileNameEdit")
        button_box = self.findChild(QDialogButtonBox)
        self._ok_button = (
            button_box.button(QDialogButtonBox.StandardButton.Save) if button_box else None
        )
        if self._file_name_edit is not None:
            self._file_name_edit.textChanged.connect(self._sync_ok_button)

    def _sync_ok_button(self) -> None:
        if self._ok_button is None or self._file_name_edit is None:
            return
        if self._ok_button.isEnabled():
            return
        if self._path_is_acceptable(self._file_name_edit.text()):
            self._ok_button.setEnabled(True)

    def _path_is_acceptable(self, raw: str) -> bool:
        text = raw.strip()
        if not text:
            return False
        info = QFileInfo(self._resolve_input(text))
        if info.isDir():
            return True
        return not info.exists()

    def _resolve_input(self, raw: str) -> str:
        text = raw.strip()
        path = Path(text)
        if path.is_absolute():
            return str(path)
        return str(Path(self.directory().absolutePath()) / path)

    def accept(self) -> None:
        files = self.selectedFiles()
        if not files:
            super().accept()
            return
        resolved = self._resolve_input(files[0])
        info = QFileInfo(resolved)
        if info.isDir() or not info.exists():
            QDialog.accept(self)
            return
        self.reject()

    def selected_directory(self) -> Path | None:
        files = self.selectedFiles()
        if not files:
            return None
        return Path(self._resolve_input(files[0])).expanduser().resolve()
