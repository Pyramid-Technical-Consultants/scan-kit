"""Session list hydrate must not extract archives."""

from __future__ import annotations

import io
import tarfile
import zipfile
from pathlib import Path

from scan_kit.common.session_source import (
    clear_termination_summary_cache,
    hydrate_session_metadata,
    list_session_storage_paths,
    load_termination_summary_cached,
    peek_session_source,
    peek_session_source_from_path,
)

_SUMMARY = """\
Date: Thu Sep 10 21:07:41 2026
Primary total dose: 12.5 MU
Treatment time: 10 seconds
Room number: 2
Configuration name: facility_map2map
"""


def _write_zip(folder: Path, sid: str) -> Path:
    archive = folder / f"{sid}.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(f"{sid}/input_map.csv", "a,b\n1,2\n")
        zf.writestr(f"{sid}/termination_summary.txt", _SUMMARY)
    return archive


def test_peek_zip_does_not_extract(tmp_path: Path) -> None:
    clear_termination_summary_cache()
    sid = "424242"
    archive = _write_zip(tmp_path, sid)
    src = peek_session_source(sid, tmp_path)
    assert src is not None
    assert src.kind == "zip"
    meta = load_termination_summary_cached(sid, archive)
    assert meta is not None
    assert meta.config_name == "facility_map2map"
    assert meta.primary_mu == 12.5
    assert not (tmp_path / sid).exists()


def test_hydrate_zip_does_not_extract(tmp_path: Path) -> None:
    clear_termination_summary_cache()
    sid = "424243"
    archive = _write_zip(tmp_path, sid)
    rows = hydrate_session_metadata([(sid, str(archive), None)], tmp_path)
    assert rows[0][2] is not None
    assert rows[0][2].short_config == "facility_map2map"
    assert not (tmp_path / sid).exists()


def test_peek_tar_uses_named_member(tmp_path: Path) -> None:
    clear_termination_summary_cache()
    sid = "555"
    archive = tmp_path / f"{sid}.tar"
    with tarfile.open(archive, "w") as tf:
        info = tarfile.TarInfo(f"{sid}/input_map.csv")
        data = b"a,b\n1,2\n"
        info.size = len(data)
        tf.addfile(info, fileobj=io.BytesIO(data))
        info2 = tarfile.TarInfo(f"{sid}/termination_summary.txt")
        raw = _SUMMARY.encode("utf-8")
        info2.size = len(raw)
        tf.addfile(info2, fileobj=io.BytesIO(raw))
    meta = load_termination_summary_cached(sid, archive)
    assert meta is not None
    assert meta.config_name == "facility_map2map"
    assert not (tmp_path / sid).is_dir()


def test_list_session_storage_paths_includes_folder_and_zip(tmp_path: Path) -> None:
    sid = "99"
    (tmp_path / sid).mkdir()
    (tmp_path / sid / "input_map.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    zip_path = _write_zip(tmp_path, sid)
    paths = list_session_storage_paths(tmp_path, sid)
    assert tmp_path / sid in paths
    assert zip_path in paths


def test_peek_from_unpacked_directory(tmp_path: Path) -> None:
    sid = "77"
    root = tmp_path / sid
    root.mkdir()
    (root / "input_map.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (root / "termination_summary.txt").write_text(_SUMMARY, encoding="utf-8")
    src = peek_session_source_from_path(root, sid)
    assert src is not None
    assert src.kind == "directory"
    meta = load_termination_summary_cached(sid, root)
    assert meta is not None
    assert meta.room_number == 2
