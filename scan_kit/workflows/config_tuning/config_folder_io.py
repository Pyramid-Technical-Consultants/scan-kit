"""Save and verify whole configuration folders (map2map XML + ``.md5`` sidecars)."""

from __future__ import annotations

import shutil
from pathlib import Path

from scan_kit.common.file_integrity import commit_file_integrity

from .xml_document import XmlDocument, write_element_tree


def iter_xml_data_files(root: Path):
    """Yield map2map-style XML data files under *root* (not ``.xml.md5`` sidecars)."""
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        if name.endswith(".xml") and not name.endswith(".xml.md5"):
            yield path


def commit_config_integrity(root: Path) -> None:
    """Rewrite ``.md5`` sidecars for every XML data file under *root*."""
    for xml_path in iter_xml_data_files(root):
        commit_file_integrity(xml_path)


def save_config_folder(
    source_root: Path,
    dest_root: Path,
    *,
    dirty_by_path: dict[Path, XmlDocument],
) -> None:
    """Copy a config folder to *dest_root* and apply in-memory edits.

    When *dest_root* equals *source_root*, only dirty XML files are written
    (explicit overwrite). Otherwise the full tree is copied first, then dirty
    files are overlaid. All XML data files under the destination get fresh
    ``.md5`` sidecars.
    """
    source_root = source_root.resolve()
    dest_root = dest_root.resolve()
    in_place = dest_root == source_root

    if not in_place:
        shutil.copytree(source_root, dest_root, dirs_exist_ok=True)

    for src_path, doc in dirty_by_path.items():
        if not doc.dirty:
            continue
        try:
            rel = src_path.resolve().relative_to(source_root)
        except ValueError:
            continue
        target = dest_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        write_element_tree(doc.tree, target)

    commit_config_integrity(dest_root)

    for doc in dirty_by_path.values():
        doc.mark_clean()
