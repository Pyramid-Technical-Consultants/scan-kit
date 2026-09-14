"""Machine-local SQLite app store (prefs + session index)."""

from __future__ import annotations

import shutil
from pathlib import Path

from scan_kit.common.session_meta import parse_termination_summary_text
from scan_kit.common.session_source import clear_termination_summary_cache
from scan_kit.common.user_store import (
    PREF_LAST_DATA_DIR,
    db_path,
    delete_session,
    notes_for_library,
    prefs_get,
    prefs_set,
    record_session_meta,
    selected_sessions,
    set_selected_sessions,
    set_session_note,
    snapshot_library,
)
from scan_kit.common import user_store as user_store_mod

_SUMMARY = """\
Date: Thu Sep 10 21:07:41 2026
Primary total dose: 5 MU
Treatment time: 8 seconds
Room number: 1
Configuration name: alpha
"""


def _session_dir(folder: Path, sid: str, summary: str = _SUMMARY) -> Path:
    root = folder / sid
    root.mkdir()
    (root / "input_map.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (root / "termination_summary.txt").write_text(summary, encoding="utf-8")
    return root


def test_prefs_round_trip() -> None:
    assert prefs_get("missing") is None
    prefs_set("session.last_data_dir", "C:/data")
    assert prefs_get(PREF_LAST_DATA_DIR) == "C:/data"
    prefs_set("ui.theme", "dark")
    assert prefs_get("ui.theme") == "dark"
    assert db_path().parent == user_store_mod.user_data_dir()
    assert db_path().name == "scan-kit.sqlite"


def test_import_json_once(tmp_path: Path) -> None:
    sid = "111"
    _session_dir(tmp_path, sid)
    (tmp_path / "session_notes.json").write_text(
        '{"111": "from json"}\n', encoding="utf-8"
    )
    (tmp_path / "settings.json").write_text(
        '{"selected_sessions": ["111"]}\n', encoding="utf-8"
    )
    notes, selected, rows = snapshot_library(tmp_path)
    assert notes[sid] == "from json"
    assert selected == [sid]
    assert len(rows) == 1

    (tmp_path / "session_notes.json").write_text(
        '{"111": "should not clobber"}\n', encoding="utf-8"
    )
    set_session_note(tmp_path, sid, "from db")
    notes2, selected2, _ = snapshot_library(tmp_path)
    assert notes2[sid] == "from db"
    assert selected2 == [sid]


def test_cache_hit_skips_reparse(tmp_path: Path) -> None:
    clear_termination_summary_cache()
    sid = "222"
    root = _session_dir(tmp_path, sid)
    notes, _, rows = snapshot_library(tmp_path)
    assert rows[0][2] is None
    meta = parse_termination_summary_text(_SUMMARY)
    record_session_meta(tmp_path, sid, str(root), meta)
    _, _, rows2 = snapshot_library(tmp_path)
    cached = rows2[0][2]
    assert cached is not None
    assert cached.config_name == "alpha"
    assert cached.primary_mu == 5.0

    (root / "termination_summary.txt").write_text(
        _SUMMARY.replace("alpha", "beta"), encoding="utf-8"
    )
    _, _, rows3 = snapshot_library(tmp_path)
    assert rows3[0][2] is None


def test_missing_library_dir_does_not_wipe_notes(tmp_path: Path) -> None:
    lib = tmp_path / "lib"
    lib.mkdir()
    sid = "444"
    _session_dir(lib, sid)
    snapshot_library(lib)
    set_session_note(lib, sid, "keep")
    shutil.rmtree(lib)
    notes, _, rows = snapshot_library(lib)
    assert notes.get(sid) == "keep"
    assert rows == []


def test_delete_session_drops_note_and_selection(tmp_path: Path) -> None:
    sid = "333"
    _session_dir(tmp_path, sid)
    snapshot_library(tmp_path)
    set_session_note(tmp_path, sid, "bye")
    set_selected_sessions(tmp_path, [sid])
    delete_session(tmp_path, sid)
    assert sid not in notes_for_library(tmp_path)
    assert selected_sessions(tmp_path) == []
