"""File tree widget for browsing a config folder."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QDir, QModelIndex, Signal
from PySide6.QtWidgets import QFileSystemModel, QHeaderView, QTreeView, QVBoxLayout, QWidget


class ConfigFileTreeWidget(QWidget):
    """Directory tree showing all files under a config root."""

    file_selected = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._root: Path | None = None

        self._model = QFileSystemModel()
        self._model.setFilter(QDir.Filter.AllDirs | QDir.Filter.Files | QDir.Filter.NoDotAndDotDot)

        self._tree = QTreeView()
        self._tree.setModel(self._model)
        self._tree.setHeaderHidden(True)
        self._tree.setRootIsDecorated(True)
        header = self._tree.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for col in range(1, self._model.columnCount()):
            self._tree.hideColumn(col)
        self._tree.clicked.connect(self._on_clicked)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._tree)

    def set_root(self, root: str | Path | None) -> None:
        path = Path(root).resolve() if root else None
        self._root = path
        if path is None or not path.is_dir():
            self._tree.setRootIndex(QModelIndex())
            return
        index = self._model.setRootPath(str(path))
        self._tree.setRootIndex(index)
        self._tree.expandToDepth(1)

    def select_relative_path(self, rel_path: str) -> None:
        if self._root is None:
            return
        target = (self._root / rel_path).resolve()
        if not target.is_file():
            return
        index = self._model.index(str(target))
        if index.isValid():
            self._tree.setCurrentIndex(index)
            self._tree.scrollTo(index)

    def _on_clicked(self, index: QModelIndex) -> None:
        path = self._model.filePath(index)
        if self._model.isDir(index):
            return
        lower = path.lower()
        if lower.endswith(".xml") or lower.endswith(".xml.md5"):
            self.file_selected.emit(path)


# Backward-compatible alias
XmlFileTreeWidget = ConfigFileTreeWidget
