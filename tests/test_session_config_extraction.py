"""Tests for opening session configuration from archived sessions."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from scan_kit.common.session_source import ensure_session_on_disk, resolve_session_source
from scan_kit.workflows.config_tuning.auto_tuning.paths import resolve_session_config_dir


def _write_minimal_session_tree(root: Path, session_id: str) -> None:
    session_root = root / session_id
    config_dir = session_root / "config" / "map2map"
    config_dir.mkdir(parents=True)
    (session_root / "input_map.csv").write_text("SPOT_ID\n1\n", encoding="utf-8")
    (config_dir / "devices.xml").write_text(
        '<?xml version="1.0"?><root></root>',
        encoding="utf-8",
    )


def _zip_session_tree(session_root: Path, zip_path: Path) -> None:
    session_id = session_root.name
    with zipfile.ZipFile(zip_path, "w") as zf:
        for path in session_root.rglob("*"):
            if path.is_file():
                rel = Path(session_id) / path.relative_to(session_root)
                zf.write(path, rel.as_posix())


def test_resolve_session_config_dir_from_zip(tmp_path: Path) -> None:
    sid = "1262268206"
    staging = tmp_path / "staging"
    _write_minimal_session_tree(staging, sid)

    zip_path = tmp_path / f"{sid}.zip"
    _zip_session_tree(staging / sid, zip_path)
    shutil.rmtree(staging)

    config_dir = resolve_session_config_dir(sid, tmp_path)
    assert config_dir is not None
    assert config_dir.name == "config"
    assert (config_dir / "map2map" / "devices.xml").is_file()

    source = resolve_session_source(sid, tmp_path)
    assert source is not None
    assert source.kind == "directory"
    assert (source.path / "input_map.csv").is_file()


def test_ensure_session_on_disk_skips_repeat_extraction(tmp_path: Path) -> None:
    sid = "999000111"
    _write_minimal_session_tree(tmp_path, sid)

    first = ensure_session_on_disk(sid, tmp_path)
    assert first is not None
    assert (first / "input_map.csv").is_file()

    second = ensure_session_on_disk(sid, tmp_path)
    assert second == first
