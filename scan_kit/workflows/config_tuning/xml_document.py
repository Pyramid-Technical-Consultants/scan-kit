"""Load, mutate, and save XML configuration files."""

from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from scan_kit.common.file_integrity import commit_file_integrity


class XmlParseError(Exception):
    """Raised when an XML file cannot be parsed."""


@dataclass
class XmlDocument:
    """In-memory XML document with explicit save/revert."""

    path: Path
    tree: ET.ElementTree
    dirty: bool = False
    _backup_created: bool = field(default=False, repr=False)

    @property
    def root(self) -> ET.Element:
        return self.tree.getroot()

    @classmethod
    def load(cls, path: str | Path) -> XmlDocument:
        p = Path(path).resolve()
        try:
            tree = ET.parse(p)
        except ET.ParseError as exc:
            raise XmlParseError(str(exc)) from exc
        except OSError as exc:
            raise XmlParseError(str(exc)) from exc
        return cls(path=p, tree=tree)

    def revert(self) -> None:
        """Reload the document from disk and discard unsaved edits."""
        reloaded = type(self).load(self.path)
        self.tree = reloaded.tree
        self.dirty = False
        self._backup_created = False

    def mark_dirty(self) -> None:
        self.dirty = True

    def mark_clean(self) -> None:
        self.dirty = False

    def save(self) -> None:
        """Write the document to disk, creating a ``.bak`` backup once per session."""
        if self.path.exists() and not self._backup_created:
            backup = self.path.with_suffix(self.path.suffix + ".bak")
            shutil.copy2(self.path, backup)
            self._backup_created = True
        self.tree.write(
            self.path,
            encoding="utf-8",
            xml_declaration=True,
        )
        commit_file_integrity(self.path)
        self.dirty = False


def write_element_tree(tree: ET.ElementTree, path: Path) -> None:
    """Write *tree* to *path* with a UTF-8 XML declaration."""
    tree.write(path, encoding="utf-8", xml_declaration=True)
