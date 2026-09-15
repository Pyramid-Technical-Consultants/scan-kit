"""Machine-local SQLite store for app prefs, view settings, session index, and notes.

The database lives under :func:`user_data_dir` so it is independent of the
install path and of any particular data folder. Filesystem remains the source
of truth for session archives. JSON files that used to hold app/view/note
state are imported once and then left unread.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .session_meta import SessionMeta
from .session_notes import load_notes_json
from .session_source import (
    peek_session_source_from_path,
    storage_fingerprint,
)
from .sessions import discover_sessions
from .settings import ViewSettings

_SCHEMA_VERSION = 2
_DB_NAME = "scan-kit.sqlite"
PREF_LAST_DATA_DIR = "session.last_data_dir"
PREF_APP_SETTINGS = "app.settings"
PREF_APP_SETTINGS_IMPORTED = "app.settings_imported"
_LEGACY_APP_SETTINGS = "app_settings.json"

_lock = threading.RLock()
_conn: sqlite3.Connection | None = None


def user_data_dir() -> Path:
    """Directory for machine-local app state (``~/.scan-kit``)."""
    return Path.home() / ".scan-kit"


def db_path() -> Path:
    return user_data_dir() / _DB_NAME


def reset_connection() -> None:
    """Close the cached connection (tests redirect ``user_data_dir``)."""
    global _conn
    with _lock:
        if _conn is not None:
            _conn.close()
            _conn = None


def prefs_get(key: str, default: Any = None) -> Any:
    conn = _connection()
    with _lock:
        row = conn.execute("SELECT value FROM prefs WHERE key = ?", (key,)).fetchone()
    if row is None:
        return default
    try:
        return json.loads(row[0])
    except (TypeError, json.JSONDecodeError):
        return default


def prefs_set(key: str, value: Any) -> None:
    raw = json.dumps(value)
    conn = _connection()
    with _lock:
        conn.execute(
            "INSERT INTO prefs(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, raw),
        )
        conn.commit()


def snapshot_library(
    base_dir: str | Path,
    *,
    project_root: Path | None = None,
) -> tuple[dict[str, str], list[str], list[tuple[str, str, SessionMeta | None]]]:
    """Discover *base_dir*, sync the index, and return notes, selection, rows.

    Cached metadata (matching storage fingerprint) is filled in; misses have
    ``meta is None`` so the caller can hydrate without extracting archives.
    """
    root = _canonical_library_path(base_dir)
    discovered = discover_sessions(
        base_dirs=(str(base_dir),),
        project_root=project_root,
    )
    conn = _connection()
    with _lock:
        lib_id = _ensure_library(conn, root)
        _import_if_needed(conn, lib_id, base_dir)
        found: dict[str, str] = {sid: path for sid, path, _ in discovered}
        if Path(root).is_dir():
            _delete_missing(conn, lib_id, set(found))
        out: list[tuple[str, str, SessionMeta | None]] = []
        for sid, path_str, _ in discovered:
            meta = _upsert_discovered(conn, lib_id, sid, path_str)
            out.append((sid, path_str, meta))
        notes = _notes_map(conn, lib_id)
        selected = _selected_list(conn, lib_id)
        conn.commit()
    return notes, selected, out


def record_session_meta(
    base_dir: str | Path,
    session_id: str,
    storage_path: str,
    meta: SessionMeta | None,
) -> None:
    """Write hydrated termination-summary fields for one session."""
    root = _canonical_library_path(base_dir)
    fp = storage_fingerprint(storage_path, session_id)
    src = peek_session_source_from_path(storage_path, session_id)
    kind = src.kind if src is not None else None
    mtime_ns = fp[1] if fp else None
    size = fp[2] if fp else None
    conn = _connection()
    with _lock:
        lib_id = _ensure_library(conn, root)
        conn.execute(
            """
            INSERT INTO sessions(
                library_id, session_id, storage_path, kind, size, mtime_ns,
                date, primary_mu, treatment_time_s, room_number, config_name, note
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '')
            ON CONFLICT(library_id, session_id) DO UPDATE SET
                storage_path = excluded.storage_path,
                kind = excluded.kind,
                size = excluded.size,
                mtime_ns = excluded.mtime_ns,
                date = excluded.date,
                primary_mu = excluded.primary_mu,
                treatment_time_s = excluded.treatment_time_s,
                room_number = excluded.room_number,
                config_name = excluded.config_name
            """,
            (
                lib_id,
                session_id,
                storage_path,
                kind,
                size,
                mtime_ns,
                _date_to_iso(meta),
                meta.primary_mu if meta else None,
                meta.treatment_time_s if meta else None,
                meta.room_number if meta else None,
                meta.config_name if meta else None,
            ),
        )
        conn.commit()


def set_session_note(base_dir: str | Path, session_id: str, text: str) -> None:
    root = _canonical_library_path(base_dir)
    note = text if text.strip() else ""
    conn = _connection()
    with _lock:
        lib_id = _ensure_library(conn, root)
        _import_if_needed(conn, lib_id, base_dir)
        conn.execute(
            """
            INSERT INTO sessions(library_id, session_id, note)
            VALUES (?, ?, ?)
            ON CONFLICT(library_id, session_id) DO UPDATE SET note = excluded.note
            """,
            (lib_id, session_id, note),
        )
        conn.commit()


def set_selected_sessions(base_dir: str | Path, session_ids: list[str]) -> None:
    root = _canonical_library_path(base_dir)
    conn = _connection()
    with _lock:
        lib_id = _ensure_library(conn, root)
        _import_if_needed(conn, lib_id, base_dir)
        conn.execute(
            "UPDATE libraries SET selected_sessions = ? WHERE id = ?",
            (json.dumps(list(session_ids)), lib_id),
        )
        conn.commit()


def selected_sessions(base_dir: str | Path) -> list[str]:
    root = _canonical_library_path(base_dir)
    conn = _connection()
    with _lock:
        lib_id = _ensure_library(conn, root)
        _import_if_needed(conn, lib_id, base_dir)
        selected = _selected_list(conn, lib_id)
        conn.commit()
    return selected


def notes_for_library(base_dir: str | Path) -> dict[str, str]:
    root = _canonical_library_path(base_dir)
    conn = _connection()
    with _lock:
        lib_id = _ensure_library(conn, root)
        _import_if_needed(conn, lib_id, base_dir)
        notes = _notes_map(conn, lib_id)
        conn.commit()
    return notes


def load_view_settings(base_dir: str | Path) -> ViewSettings:
    root = _canonical_library_path(base_dir)
    conn = _connection()
    with _lock:
        lib_id = _ensure_library(conn, root)
        _import_if_needed(conn, lib_id, base_dir)
        row = conn.execute(
            """
            SELECT bg_subtract, calibration_mode, contour_cutoff_percentile,
                   selected_sessions
            FROM libraries WHERE id = ?
            """,
            (lib_id,),
        ).fetchone()
        conn.commit()
    if row is None:
        return ViewSettings()
    cleaned = ViewSettings._clean(
        {
            "bg_subtract": bool(row["bg_subtract"]),
            "calibration_mode": row["calibration_mode"],
            "contour_cutoff_percentile": row["contour_cutoff_percentile"],
            "selected_sessions": _parse_selected(row["selected_sessions"]),
        }
    )
    return ViewSettings(**cleaned)


def save_view_settings(base_dir: str | Path, settings: ViewSettings) -> None:
    cleaned = ViewSettings._clean(
        {
            "bg_subtract": settings.bg_subtract,
            "calibration_mode": settings.calibration_mode,
            "contour_cutoff_percentile": settings.contour_cutoff_percentile,
        }
    )
    root = _canonical_library_path(base_dir)
    conn = _connection()
    with _lock:
        lib_id = _ensure_library(conn, root)
        conn.execute(
            """
            UPDATE libraries SET
                bg_subtract = ?,
                calibration_mode = ?,
                contour_cutoff_percentile = ?,
                view_settings_imported = 1,
                view_settings_rev = view_settings_rev + 1
            WHERE id = ?
            """,
            (
                1 if cleaned.get("bg_subtract") else 0,
                cleaned.get("calibration_mode", "off"),
                cleaned.get("contour_cutoff_percentile", 5.0),
                lib_id,
            ),
        )
        conn.commit()


def view_settings_rev(base_dir: str | Path) -> int:
    """Revision counter for live view refresh (WAL file mtime is not a signal)."""
    root = _canonical_library_path(base_dir)
    conn = _connection()
    with _lock:
        row = conn.execute(
            "SELECT view_settings_rev FROM libraries WHERE root_path = ?",
            (root,),
        ).fetchone()
    return int(row[0]) if row else 0


def load_app_settings():
    from .app_settings import AppSettings

    _import_app_settings_json_once()
    raw = prefs_get(PREF_APP_SETTINGS)
    if not isinstance(raw, dict):
        return AppSettings()
    return AppSettings.from_mapping(raw)


def save_app_settings(settings) -> None:
    prefs_set(PREF_APP_SETTINGS, asdict(settings))
    prefs_set(PREF_APP_SETTINGS_IMPORTED, True)


def delete_session(base_dir: str | Path, session_id: str) -> None:
    root = _canonical_library_path(base_dir)
    conn = _connection()
    with _lock:
        lib_id = _ensure_library(conn, root)
        conn.execute(
            "DELETE FROM sessions WHERE library_id = ? AND session_id = ?",
            (lib_id, session_id),
        )
        selected = [s for s in _selected_list(conn, lib_id) if s != session_id]
        conn.execute(
            "UPDATE libraries SET selected_sessions = ? WHERE id = ?",
            (json.dumps(selected), lib_id),
        )
        conn.commit()


def _connection() -> sqlite3.Connection:
    global _conn
    with _lock:
        if _conn is None:
            folder = user_data_dir()
            folder.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(
                str(db_path()),
                check_same_thread=False,
                isolation_level=None,
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("PRAGMA foreign_keys=ON")
            _migrate(conn)
            _conn = conn
        return _conn


def _migrate(conn: sqlite3.Connection) -> None:
    version = int(conn.execute("PRAGMA user_version").fetchone()[0])
    if version < 1:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS prefs (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS libraries (
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
            CREATE TABLE IF NOT EXISTS sessions (
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
        conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
        conn.commit()
        return
    if version < 2:
        conn.executescript(
            """
            ALTER TABLE libraries ADD COLUMN bg_subtract INTEGER NOT NULL DEFAULT 0;
            ALTER TABLE libraries ADD COLUMN calibration_mode TEXT NOT NULL DEFAULT 'off';
            ALTER TABLE libraries ADD COLUMN contour_cutoff_percentile REAL NOT NULL DEFAULT 5.0;
            ALTER TABLE libraries ADD COLUMN view_settings_imported INTEGER NOT NULL DEFAULT 0;
            ALTER TABLE libraries ADD COLUMN view_settings_rev INTEGER NOT NULL DEFAULT 0;
            """
        )
        conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
        conn.commit()


def _canonical_library_path(base_dir: str | Path) -> str:
    path = Path(base_dir).expanduser()
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


def _ensure_library(conn: sqlite3.Connection, root: str) -> int:
    conn.execute("INSERT OR IGNORE INTO libraries(root_path) VALUES (?)", (root,))
    row = conn.execute("SELECT id FROM libraries WHERE root_path = ?", (root,)).fetchone()
    assert row is not None
    return int(row[0])


def _import_if_needed(
    conn: sqlite3.Connection, lib_id: int, base_dir: str | Path
) -> None:
    row = conn.execute(
        "SELECT notes_imported, view_settings_imported FROM libraries WHERE id = ?",
        (lib_id,),
    ).fetchone()
    notes_done = bool(row and row[0])
    views_done = bool(row and row[1])
    if notes_done and views_done:
        return
    legacy = ViewSettings.load_legacy_json(base_dir)
    if not notes_done:
        notes = load_notes_json(base_dir)
        for sid, text in notes.items():
            if not str(text).strip():
                continue
            conn.execute(
                """
                INSERT INTO sessions(library_id, session_id, note)
                VALUES (?, ?, ?)
                ON CONFLICT(library_id, session_id) DO UPDATE SET note = excluded.note
                """,
                (lib_id, str(sid), str(text)),
            )
        selected = [str(s) for s in (legacy.selected_sessions or [])]
        conn.execute(
            "UPDATE libraries SET selected_sessions = ?, notes_imported = 1 WHERE id = ?",
            (json.dumps(selected), lib_id),
        )
    if not views_done:
        cleaned = ViewSettings._clean(
            {
                "bg_subtract": legacy.bg_subtract,
                "calibration_mode": legacy.calibration_mode,
                "contour_cutoff_percentile": legacy.contour_cutoff_percentile,
            }
        )
        conn.execute(
            """
            UPDATE libraries SET
                bg_subtract = ?,
                calibration_mode = ?,
                contour_cutoff_percentile = ?,
                view_settings_imported = 1
            WHERE id = ?
            """,
            (
                1 if cleaned.get("bg_subtract") else 0,
                cleaned.get("calibration_mode", "off"),
                cleaned.get("contour_cutoff_percentile", 5.0),
                lib_id,
            ),
        )


def _import_app_settings_json_once() -> None:
    if prefs_get(PREF_APP_SETTINGS_IMPORTED):
        return
    path = user_data_dir() / _LEGACY_APP_SETTINGS
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError, TypeError):
        data = None
    if isinstance(data, dict):
        prefs_set(PREF_APP_SETTINGS, data)
    prefs_set(PREF_APP_SETTINGS_IMPORTED, True)


def _delete_missing(
    conn: sqlite3.Connection, lib_id: int, found: set[str]
) -> None:
    rows = conn.execute(
        "SELECT session_id FROM sessions WHERE library_id = ?", (lib_id,)
    ).fetchall()
    gone = [str(r[0]) for r in rows if str(r[0]) not in found]
    if not gone:
        return
    conn.executemany(
        "DELETE FROM sessions WHERE library_id = ? AND session_id = ?",
        [(lib_id, sid) for sid in gone],
    )
    selected = [s for s in _selected_list(conn, lib_id) if s in found]
    conn.execute(
        "UPDATE libraries SET selected_sessions = ?, last_scan_at = ? WHERE id = ?",
        (json.dumps(selected), datetime.now().timestamp(), lib_id),
    )


def _upsert_discovered(
    conn: sqlite3.Connection,
    lib_id: int,
    sid: str,
    path_str: str,
) -> SessionMeta | None:
    fp = storage_fingerprint(path_str, sid)
    src = peek_session_source_from_path(path_str, sid)
    kind = src.kind if src is not None else None
    mtime_ns = fp[1] if fp else None
    size = fp[2] if fp else None
    existing = conn.execute(
        """
        SELECT size, mtime_ns, date, primary_mu, treatment_time_s, room_number,
               config_name
        FROM sessions WHERE library_id = ? AND session_id = ?
        """,
        (lib_id, sid),
    ).fetchone()
    if (
        existing is not None
        and fp is not None
        and existing["size"] == size
        and existing["mtime_ns"] == mtime_ns
    ):
        has_meta = any(
            existing[col] is not None
            for col in (
                "date",
                "primary_mu",
                "treatment_time_s",
                "room_number",
                "config_name",
            )
        )
        conn.execute(
            """
            UPDATE sessions SET storage_path = ?, kind = ?
            WHERE library_id = ? AND session_id = ?
            """,
            (path_str, kind, lib_id, sid),
        )
        return _meta_from_row(existing) if has_meta else None

    conn.execute(
        """
        INSERT INTO sessions(
            library_id, session_id, storage_path, kind, size, mtime_ns,
            date, primary_mu, treatment_time_s, room_number, config_name
        ) VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL)
        ON CONFLICT(library_id, session_id) DO UPDATE SET
            storage_path = excluded.storage_path,
            kind = excluded.kind,
            size = excluded.size,
            mtime_ns = excluded.mtime_ns,
            date = NULL,
            primary_mu = NULL,
            treatment_time_s = NULL,
            room_number = NULL,
            config_name = NULL
        """,
        (lib_id, sid, path_str, kind, size, mtime_ns),
    )
    return None


def _notes_map(conn: sqlite3.Connection, lib_id: int) -> dict[str, str]:
    rows = conn.execute(
        "SELECT session_id, note FROM sessions WHERE library_id = ? AND note != ''",
        (lib_id,),
    ).fetchall()
    return {str(r[0]): str(r[1]) for r in rows}


def _selected_list(conn: sqlite3.Connection, lib_id: int) -> list[str]:
    row = conn.execute(
        "SELECT selected_sessions FROM libraries WHERE id = ?", (lib_id,)
    ).fetchone()
    if row is None:
        return []
    return _parse_selected(row[0])


def _parse_selected(raw: Any) -> list[str]:
    try:
        parsed = json.loads(raw) if not isinstance(raw, list) else raw
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(parsed, list):
        return []
    return [str(s) for s in parsed]


def _date_to_iso(meta: SessionMeta | None) -> str | None:
    if meta is None or meta.date is None:
        return None
    return meta.date.isoformat()


def _meta_from_row(row: sqlite3.Row) -> SessionMeta:
    date = None
    raw = row["date"]
    if raw:
        try:
            date = datetime.fromisoformat(str(raw))
        except ValueError:
            date = None
    return SessionMeta(
        date=date,
        primary_mu=row["primary_mu"],
        treatment_time_s=row["treatment_time_s"],
        room_number=row["room_number"],
        config_name=row["config_name"],
    )
