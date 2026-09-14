"""Recycle-bin helper and session-list recycle action."""

from __future__ import annotations

import shutil
from pathlib import Path

from scan_kit.common.recycle import move_to_trash
from scan_kit.common.session_browser import SessionBrowserWidget
from scan_kit.common.session_source import list_session_storage_paths
from scan_kit.common.user_store import notes_for_library, set_session_note


def test_move_to_trash_uses_platform_helper(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "gone.txt"
    target.write_text("x", encoding="utf-8")
    sent: list[Path] = []

    def fake(path: Path) -> None:
        sent.append(path)
        path.unlink()

    monkeypatch.setattr("scan_kit.common.recycle._windows_recycle", fake)
    monkeypatch.setattr("scan_kit.common.recycle._darwin_recycle", fake)
    monkeypatch.setattr("scan_kit.common.recycle._posix_recycle", fake)
    move_to_trash(target)
    assert sent == [target]
    assert not target.exists()


def test_recycle_session_trashes_folder_and_zip(qapp, tmp_path: Path) -> None:
    sid = "1010"
    folder = tmp_path / sid
    folder.mkdir()
    (folder / "input_map.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    zip_path = tmp_path / f"{sid}.zip"
    zip_path.write_bytes(b"PK\x05\x06" + b"\x00" * 18)

    widget = SessionBrowserWidget(initial_base_dir=str(tmp_path), editable_notes=True)
    try:
        widget._table.setRowCount(1)
        widget._set_session_row_widgets(0, sid, None, use_checked=True)
        widget._rebuild_session_row_index()
        widget._check_order = [sid]
        set_session_note(tmp_path, sid, "keep me")

        trashed: list[Path] = []

        def fake_trash(path) -> None:
            p = Path(path)
            trashed.append(p)
            if p.is_dir():
                shutil.rmtree(p)
            elif p.is_file():
                p.unlink()

        widget._trash = fake_trash
        widget._confirm_recycle = lambda session_id, paths: True
        widget._recycle_session(sid)

        assert tmp_path / sid in trashed
        assert zip_path in trashed
        assert not folder.exists()
        assert not zip_path.exists()
        assert widget._table.rowCount() == 0
        assert sid not in notes_for_library(tmp_path)
        assert list_session_storage_paths(tmp_path, sid) == []
    finally:
        widget.shutdown()


def test_recycle_cancelled_leaves_files(qapp, tmp_path: Path) -> None:
    sid = "2020"
    folder = tmp_path / sid
    folder.mkdir()
    (folder / "input_map.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    widget = SessionBrowserWidget(initial_base_dir=str(tmp_path))
    try:
        widget._table.setRowCount(1)
        widget._set_session_row_widgets(0, sid, None, use_checked=False)
        widget._rebuild_session_row_index()
        widget._confirm_recycle = lambda session_id, paths: False
        widget._trash = lambda path: (_ for _ in ()).throw(
            AssertionError("should not trash")
        )
        widget._recycle_session(sid)
        assert folder.exists()
        assert widget._table.rowCount() == 1
    finally:
        widget.shutdown()
