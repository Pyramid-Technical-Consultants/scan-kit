"""Machine-local SQLite app store (prefs, view settings, session index)."""

from __future__ import annotations

import shutil
from pathlib import Path

from scan_kit.common.session_meta import parse_termination_summary_text
from scan_kit.common.session_source import clear_termination_summary_cache
from scan_kit.common.settings import ViewSettings
from scan_kit.common.user_store import (
    PREF_LAST_DATA_DIR,
    db_path,
    delete_session,
    notes_for_library,
    prefs_get,
    prefs_set,
    record_session_meta,
    reset_connection,
    selected_sessions,
    set_selected_sessions,
    set_session_note,
    snapshot_library,
    view_settings_rev,
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
        '{"selected_sessions": ["111"], "bg_subtract": true,'
        ' "calibration_mode": "constrained"}\n',
        encoding="utf-8",
    )
    notes, selected, rows = snapshot_library(tmp_path)
    assert notes[sid] == "from json"
    assert selected == [sid]
    assert len(rows) == 1

    vs = ViewSettings.load(tmp_path)
    assert vs.bg_subtract is True
    assert vs.calibration_mode == "constrained"

    (tmp_path / "session_notes.json").write_text(
        '{"111": "should not clobber"}\n', encoding="utf-8"
    )
    (tmp_path / "settings.json").write_text(
        '{"bg_subtract": false, "selected_sessions": []}\n', encoding="utf-8"
    )
    set_session_note(tmp_path, sid, "from db")
    notes2, selected2, _ = snapshot_library(tmp_path)
    assert notes2[sid] == "from db"
    assert selected2 == [sid]
    vs2 = ViewSettings.load(tmp_path)
    assert vs2.bg_subtract is True
    assert vs2.calibration_mode == "constrained"


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


def test_saves_do_not_write_json(tmp_path: Path) -> None:
    from scan_kit.common.app_settings import AppSettings
    from scan_kit.common.session_notes import load_notes, save_note

    AppSettings(config_dir="/cfg").save()
    ViewSettings(bg_subtract=True, contour_cutoff_percentile=8.0).save(tmp_path)
    save_note(tmp_path, "111", "hello")

    assert not (tmp_path / "settings.json").exists()
    assert not (tmp_path / "session_notes.json").exists()
    assert not (user_store_mod.user_data_dir() / "app_settings.json").exists()
    assert ViewSettings.load(tmp_path).bg_subtract is True
    assert ViewSettings.load(tmp_path).contour_cutoff_percentile == 8.0
    assert load_notes(tmp_path)["111"] == "hello"
    assert AppSettings.load().config_dir == "/cfg"


def test_view_settings_rev_increments(tmp_path: Path) -> None:
    assert view_settings_rev(tmp_path) == 0
    ViewSettings(bg_subtract=True).save(tmp_path)
    assert view_settings_rev(tmp_path) == 1
    ViewSettings(bg_subtract=False).save(tmp_path)
    assert view_settings_rev(tmp_path) == 2


def test_app_settings_json_import_once() -> None:
    import json

    from scan_kit.common.app_settings import AppSettings

    path = user_store_mod.user_data_dir() / "app_settings.json"
    path.write_text(
        json.dumps({"config_dir": "/from-json", "window_width": 800}),
        encoding="utf-8",
    )
    loaded = AppSettings.load()
    assert loaded.config_dir == "/from-json"
    assert loaded.window_width == 800
    path.write_text(json.dumps({"config_dir": "/ignored"}), encoding="utf-8")
    assert AppSettings.load().config_dir == "/from-json"
    AppSettings(config_dir="/saved").save()
    assert json.loads(path.read_text(encoding="utf-8"))["config_dir"] == "/ignored"
    assert AppSettings.load().config_dir == "/saved"


def test_v1_schema_migrates_and_imports_view_settings(tmp_path: Path) -> None:
    import sqlite3

    from scan_kit.common.session_notes import load_notes

    sid = "111"
    _session_dir(tmp_path, sid)
    root = str(tmp_path.resolve())
    user_store_mod.user_data_dir().mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path()))
    conn.executescript(
        """
        CREATE TABLE prefs (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE libraries (
            id INTEGER PRIMARY KEY,
            root_path TEXT NOT NULL UNIQUE,
            selected_sessions TEXT NOT NULL DEFAULT '[]',
            notes_imported INTEGER NOT NULL DEFAULT 0,
            last_scan_at REAL
        );
        CREATE TABLE sessions (
            id INTEGER PRIMARY KEY,
            library_id INTEGER NOT NULL REFERENCES libraries(id) ON DELETE CASCADE,
            session_id TEXT NOT NULL,
            storage_path TEXT,
            kind TEXT,
            size INTEGER,
            mtime_ns INTEGER,
            date TEXT,
            primary_mu REAL,
            treatment_time_s INTEGER,
            room_number INTEGER,
            config_name TEXT,
            note TEXT NOT NULL DEFAULT '',
            UNIQUE(library_id, session_id)
        );
        """
    )
    conn.execute("PRAGMA user_version = 1")
    conn.execute(
        "INSERT INTO libraries(root_path, selected_sessions, notes_imported)"
        " VALUES (?, ?, 1)",
        (root, '["111"]'),
    )
    lib_id = conn.execute("SELECT id FROM libraries").fetchone()[0]
    conn.execute(
        "INSERT INTO sessions(library_id, session_id, note) VALUES (?, ?, ?)",
        (lib_id, sid, "keep-me"),
    )
    conn.commit()
    conn.close()

    (tmp_path / "settings.json").write_text(
        '{"bg_subtract": true, "calibration_mode": "per_session",'
        ' "contour_cutoff_percentile": 12}\n',
        encoding="utf-8",
    )
    reset_connection()
    loaded = ViewSettings.load(tmp_path)
    assert loaded.bg_subtract is True
    assert loaded.calibration_mode == "per_session"
    assert loaded.contour_cutoff_percentile == 12.0
    assert loaded.selected_sessions == [sid]
    assert load_notes(tmp_path)[sid] == "keep-me"

    probe = sqlite3.connect(str(db_path()))
    assert probe.execute("PRAGMA user_version").fetchone()[0] == 3
    probe.close()


def test_cache_round_trips_extent_and_layers(tmp_path: Path) -> None:
    clear_termination_summary_cache()
    sid = "555"
    root = _session_dir(
        tmp_path,
        sid,
        summary=_SUMMARY
        + "Layer delivery: 27/27\n"
        + "Spot extent width: 118.75 mm\n"
        + "Spot extent height: 110.592 mm\n",
    )
    snapshot_library(tmp_path)
    meta = parse_termination_summary_text(
        (root / "termination_summary.txt").read_text(encoding="utf-8")
    )
    assert meta.map_extent_mm == 118.75
    assert meta.layer_count == 27
    record_session_meta(tmp_path, sid, str(root), meta)
    _, _, rows = snapshot_library(tmp_path)
    cached = rows[0][2]
    assert cached is not None
    assert cached.map_extent_mm == 118.75
    assert cached.layer_count == 27
    assert cached.short_extent == "119"


def test_unchecked_geom_cache_misses(tmp_path: Path) -> None:
    import sqlite3

    clear_termination_summary_cache()
    sid = "666"
    root = _session_dir(tmp_path, sid)
    snapshot_library(tmp_path)
    record_session_meta(tmp_path, sid, str(root), parse_termination_summary_text(_SUMMARY))
    conn = sqlite3.connect(str(db_path()))
    conn.execute("UPDATE sessions SET map_geom_checked = 0")
    conn.commit()
    conn.close()
    reset_connection()
    _, _, rows = snapshot_library(tmp_path)
    assert rows[0][2] is None


def test_v2_schema_migrates_geom_columns(tmp_path: Path) -> None:
    import sqlite3

    sid = "111"
    _session_dir(tmp_path, sid)
    root = str(tmp_path.resolve())
    user_store_mod.user_data_dir().mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path()))
    conn.executescript(
        """
        CREATE TABLE prefs (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE libraries (
            id INTEGER PRIMARY KEY,
            root_path TEXT NOT NULL UNIQUE,
            selected_sessions TEXT NOT NULL DEFAULT '[]',
            notes_imported INTEGER NOT NULL DEFAULT 0,
            last_scan_at REAL,
            bg_subtract INTEGER NOT NULL DEFAULT 0,
            calibration_mode TEXT NOT NULL DEFAULT 'off',
            contour_cutoff_percentile REAL NOT NULL DEFAULT 5.0,
            view_settings_imported INTEGER NOT NULL DEFAULT 0,
            view_settings_rev INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE sessions (
            id INTEGER PRIMARY KEY,
            library_id INTEGER NOT NULL REFERENCES libraries(id) ON DELETE CASCADE,
            session_id TEXT NOT NULL,
            storage_path TEXT,
            kind TEXT,
            size INTEGER,
            mtime_ns INTEGER,
            date TEXT,
            primary_mu REAL,
            treatment_time_s INTEGER,
            room_number INTEGER,
            config_name TEXT,
            note TEXT NOT NULL DEFAULT '',
            UNIQUE(library_id, session_id)
        );
        """
    )
    conn.execute("PRAGMA user_version = 2")
    conn.execute("INSERT INTO libraries(root_path) VALUES (?)", (root,))
    conn.commit()
    conn.close()

    reset_connection()
    snapshot_library(tmp_path)
    probe = sqlite3.connect(str(db_path()))
    assert probe.execute("PRAGMA user_version").fetchone()[0] == 3
    cols = {row[1] for row in probe.execute("PRAGMA table_info(sessions)")}
    assert "map_extent_mm" in cols
    assert "layer_count" in cols
    assert "map_geom_checked" in cols
    probe.close()
