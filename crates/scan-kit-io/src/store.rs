//! Sqlite store compatible with `scan_kit/common/user_store.py` (`user_version` 3).
//!
//! ponytail: one process-wide lock serializes every store call. A pool can
//! replace it if command latency shows up.

use std::collections::HashSet;
use std::path::{Component, Path, PathBuf};
use std::sync::Mutex;

use rusqlite::{params, Connection, OptionalExtension};
use scan_kit_core::{SessionMeta, SummaryDate};
use serde_json::{json, Value};

use super::discover::{self, Discovered};

const SCHEMA_VERSION: i32 = 3;
const PREF_LAST_DATA_DIR: &str = "session.last_data_dir";
const PREF_APP_SETTINGS: &str = "app.settings";
const PREF_APP_SETTINGS_IMPORTED: &str = "app.settings_imported";

static STORE_LOCK: Mutex<()> = Mutex::new(());

pub fn default_db_path() -> PathBuf {
    let home = std::env::var_os("USERPROFILE")
        .or_else(|| std::env::var_os("HOME"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    home.join(".scan-kit").join("scan-kit.sqlite")
}

pub fn with_store<T>(
    path: &Path,
    body: impl FnOnce(&mut Connection) -> Result<T, String>,
) -> Result<T, String> {
    let _guard = STORE_LOCK
        .lock()
        .map_err(|_| "the session store lock was poisoned".to_owned())?;
    let mut conn = open_connection(path)?;
    body(&mut conn)
}

pub fn open_connection(path: &Path) -> Result<Connection, String> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|err| err.to_string())?;
        }
    }
    let conn = Connection::open(path).map_err(|err| err.to_string())?;
    conn.pragma_update(None, "journal_mode", "WAL")
        .map_err(|err| err.to_string())?;
    conn.pragma_update(None, "busy_timeout", 5000)
        .map_err(|err| err.to_string())?;
    conn.pragma_update(None, "foreign_keys", "ON")
        .map_err(|err| err.to_string())?;
    migrate(&conn)?;
    import_app_settings_once(&conn)?;
    Ok(conn)
}

pub fn user_version(conn: &Connection) -> Result<i32, String> {
    conn.query_row("PRAGMA user_version", [], |row| row.get(0))
        .map_err(|err| err.to_string())
}

pub fn canonical_local(path: &Path) -> String {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(path)
    };
    let resolved = std::fs::canonicalize(&absolute).unwrap_or(normalize_lexically(&absolute));
    strip_verbatim(&resolved)
}

pub fn last_data_dir(conn: &Connection) -> Result<Option<String>, String> {
    Ok(prefs_get(conn, PREF_LAST_DATA_DIR)?.and_then(|value| value.as_str().map(str::to_owned)))
}

pub fn window_geometry(conn: &Connection) -> Result<Value, String> {
    let settings = prefs_get(conn, PREF_APP_SETTINGS)?.unwrap_or_else(|| json!({}));
    let object = settings.as_object();
    let number =
        |key: &str| -> Option<i64> { object.and_then(|map| map.get(key)).and_then(Value::as_i64) };
    Ok(json!({
        "width": number("window_width"),
        "height": number("window_height"),
        "x": number("window_x"),
        "y": number("window_y"),
    }))
}

pub fn set_window_geometry(
    conn: &Connection,
    width: i32,
    height: i32,
    x: i32,
    y: i32,
) -> Result<(), String> {
    let mut settings = prefs_get(conn, PREF_APP_SETTINGS)?
        .and_then(|value| value.as_object().cloned())
        .unwrap_or_default();
    settings.insert("window_width".to_owned(), json!(width));
    settings.insert("window_height".to_owned(), json!(height));
    settings.insert("window_x".to_owned(), json!(x));
    settings.insert("window_y".to_owned(), json!(y));
    prefs_set(conn, PREF_APP_SETTINGS, &Value::Object(settings))
}

pub fn last_main_tab(conn: &Connection) -> Result<Option<String>, String> {
    let settings = prefs_get(conn, PREF_APP_SETTINGS)?.unwrap_or_else(|| json!({}));
    Ok(settings
        .get("last_main_tab")
        .and_then(Value::as_str)
        .map(str::to_owned))
}

pub fn set_last_main_tab(conn: &Connection, tab: &str) -> Result<(), String> {
    let mut settings = prefs_get(conn, PREF_APP_SETTINGS)?
        .and_then(|value| value.as_object().cloned())
        .unwrap_or_default();
    settings.insert("last_main_tab".to_owned(), json!(tab));
    prefs_set(conn, PREF_APP_SETTINGS, &Value::Object(settings))
}

pub fn set_note(conn: &Connection, root: &str, session_id: &str, text: &str) -> Result<(), String> {
    let lib_id = ensure_library(conn, root)?;
    import_legacy(conn, lib_id, Path::new(root))?;
    let note = if text.trim().is_empty() { "" } else { text };
    conn.execute(
        "INSERT INTO sessions(library_id, session_id, note) VALUES (?1, ?2, ?3)
         ON CONFLICT(library_id, session_id) DO UPDATE SET note = excluded.note",
        params![lib_id, session_id, note],
    )
    .map_err(|err| err.to_string())?;
    Ok(())
}

pub fn set_selected(conn: &Connection, root: &str, session_ids: &[String]) -> Result<(), String> {
    let mut unique = Vec::new();
    for session_id in session_ids {
        if !unique.contains(session_id) {
            unique.push(session_id.clone());
        }
    }
    if unique.len() > 5 {
        return Err("at most five sessions can be selected".to_owned());
    }
    let lib_id = ensure_library(conn, root)?;
    import_legacy(conn, lib_id, Path::new(root))?;
    let encoded = serde_json::to_string(&unique).map_err(|err| err.to_string())?;
    conn.execute(
        "UPDATE libraries SET selected_sessions = ?1 WHERE id = ?2",
        params![encoded, lib_id],
    )
    .map_err(|err| err.to_string())?;
    Ok(())
}

pub enum Cache {
    Ready,
    Hydrate,
}

pub fn sync_entry(conn: &Connection, lib_id: i64, entry: &Discovered) -> Result<Cache, String> {
    let fingerprint = discover::storage_fingerprint(&entry.storage_path, &entry.session_id);
    let (mtime_ns, size) = match fingerprint {
        Some((mtime, size)) => (Some(mtime), Some(size)),
        None => (None, None),
    };
    let existing = conn
        .query_row(
            "SELECT size, mtime_ns, map_geom_checked
             FROM sessions WHERE library_id = ?1 AND session_id = ?2",
            params![lib_id, entry.session_id],
            |row| {
                Ok(Existing {
                    size: row.get(0)?,
                    mtime_ns: row.get(1)?,
                    map_geom_checked: row.get(2)?,
                })
            },
        )
        .optional()
        .map_err(|err| err.to_string())?;

    let path = entry.storage_path.to_string_lossy().to_string();
    if let Some(existing) = existing {
        if fingerprint.is_some() && existing.size == size && existing.mtime_ns == mtime_ns {
            conn.execute(
                "UPDATE sessions SET storage_path = ?1, kind = ?2
                 WHERE library_id = ?3 AND session_id = ?4",
                params![path, entry.kind, lib_id, entry.session_id],
            )
            .map_err(|err| err.to_string())?;
            if existing.map_geom_checked != 0 {
                return Ok(Cache::Ready);
            }
            return Ok(Cache::Hydrate);
        }
    }

    conn.execute(
        "INSERT INTO sessions(
            library_id, session_id, storage_path, kind, size, mtime_ns,
            date, primary_mu, treatment_time_s, room_number, config_name,
            map_extent_mm, layer_count, map_geom_checked
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 0)
         ON CONFLICT(library_id, session_id) DO UPDATE SET
            storage_path = excluded.storage_path,
            kind = excluded.kind,
            size = excluded.size,
            mtime_ns = excluded.mtime_ns,
            date = NULL,
            primary_mu = NULL,
            treatment_time_s = NULL,
            room_number = NULL,
            config_name = NULL,
            map_extent_mm = NULL,
            layer_count = NULL,
            map_geom_checked = 0",
        params![lib_id, entry.session_id, path, entry.kind, size, mtime_ns],
    )
    .map_err(|err| err.to_string())?;
    Ok(Cache::Hydrate)
}

pub fn record_meta(
    conn: &Connection,
    lib_id: i64,
    entry: &Discovered,
    meta: Option<&SessionMeta>,
) -> Result<(), String> {
    let fingerprint = discover::storage_fingerprint(&entry.storage_path, &entry.session_id);
    let (mtime_ns, size) = match fingerprint {
        Some((mtime, size)) => (Some(mtime), Some(size)),
        None => (None, None),
    };
    let path = entry.storage_path.to_string_lossy().to_string();
    let date = meta.and_then(SessionMeta::date_iso);
    conn.execute(
        "INSERT INTO sessions(
            library_id, session_id, storage_path, kind, size, mtime_ns,
            date, primary_mu, treatment_time_s, room_number, config_name,
            map_extent_mm, layer_count, map_geom_checked, note
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, 1, '')
         ON CONFLICT(library_id, session_id) DO UPDATE SET
            storage_path = excluded.storage_path,
            kind = excluded.kind,
            size = excluded.size,
            mtime_ns = excluded.mtime_ns,
            date = excluded.date,
            primary_mu = excluded.primary_mu,
            treatment_time_s = excluded.treatment_time_s,
            room_number = excluded.room_number,
            config_name = excluded.config_name,
            map_extent_mm = excluded.map_extent_mm,
            layer_count = excluded.layer_count,
            map_geom_checked = 1",
        params![
            lib_id,
            entry.session_id,
            path,
            entry.kind,
            size,
            mtime_ns,
            date,
            meta.and_then(|item| item.primary_mu),
            meta.and_then(|item| item.treatment_time_s),
            meta.and_then(|item| item.room_number),
            meta.and_then(|item| item.config_name.clone()),
            meta.and_then(|item| item.map_extent_mm),
            meta.and_then(|item| item.layer_count),
        ],
    )
    .map_err(|err| err.to_string())?;
    Ok(())
}

/// Copy `session_notes.json` into blank SQL notes.
///
/// ponytail: runs only when this library has no notes at all, so deleting one
/// note does not revive the JSON. Clearing every note would.
pub fn backfill_blank_notes(conn: &Connection, lib_id: i64, root: &Path) -> Result<(), String> {
    let filled: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM sessions WHERE library_id = ?1 AND note != ''",
            params![lib_id],
            |row| row.get(0),
        )
        .map_err(|err| err.to_string())?;
    if filled > 0 {
        return Ok(());
    }
    let Some(notes) =
        read_json(root.join("session_notes.json")).and_then(|value| value.as_object().cloned())
    else {
        return Ok(());
    };
    for (session_id, text) in notes {
        let Some(text) = text.as_str() else {
            continue;
        };
        if text.trim().is_empty() {
            continue;
        }
        conn.execute(
            "UPDATE sessions SET note = ?1
             WHERE library_id = ?2 AND session_id = ?3 AND note = ''",
            params![text, lib_id, session_id],
        )
        .map_err(|err| err.to_string())?;
    }
    Ok(())
}

pub fn prepare_library(conn: &Connection, root: &str) -> Result<i64, String> {
    let lib_id = ensure_library(conn, root)?;
    import_legacy(conn, lib_id, Path::new(root))?;
    let found = discover::discover_entries(Path::new(root))?
        .into_iter()
        .map(|entry| entry.session_id)
        .collect::<HashSet<_>>();
    delete_missing(conn, lib_id, &found)?;
    Ok(lib_id)
}

pub fn library_rows(conn: &Connection, lib_id: i64) -> Result<Vec<Value>, String> {
    let selected = selected_list(conn, lib_id)?;
    let mut statement = conn
        .prepare(
            "SELECT session_id, storage_path, date, primary_mu, treatment_time_s,
                    room_number, config_name, map_extent_mm, layer_count, note
             FROM sessions WHERE library_id = ?1 ORDER BY session_id",
        )
        .map_err(|err| err.to_string())?;
    let rows = statement
        .query_map(params![lib_id], |row| {
            Ok(StoredRow {
                session_id: row.get(0)?,
                storage_path: row.get::<_, Option<String>>(1)?.unwrap_or_default(),
                date: row.get(2)?,
                primary_mu: row.get(3)?,
                treatment_time_s: row.get(4)?,
                room_number: row.get(5)?,
                config_name: row.get(6)?,
                map_extent_mm: row.get(7)?,
                layer_count: row.get(8)?,
                note: row.get::<_, Option<String>>(9)?.unwrap_or_default(),
            })
        })
        .map_err(|err| err.to_string())?;
    let mut out = Vec::new();
    for row in rows {
        let row = row.map_err(|err| err.to_string())?;
        let meta = meta_from_stored(&row);
        out.push(json!({
            "session_id": row.session_id,
            "storage_path": row.storage_path,
            "selected": selected.iter().any(|id| id == &row.session_id),
            "note": row.note,
            "date": meta.short_date(),
            "date_iso": meta.date_iso(),
            "mu": meta.short_mu(),
            "mu_value": meta.primary_mu,
            "extent": meta.short_extent(),
            "extent_value": meta.map_extent_mm,
            "layers": meta.short_layers(),
            "layers_value": meta.layer_count,
            "time": meta.short_time(),
            "time_value": meta.treatment_time_s,
            "room": meta.short_room(),
            "room_value": meta.room_number,
            "config": meta.short_config(),
        }));
    }
    Ok(out)
}

pub fn remember_data_dir(conn: &Connection, root: &str) -> Result<(), String> {
    prefs_set(conn, PREF_LAST_DATA_DIR, &json!(root))
}

fn migrate(conn: &Connection) -> Result<(), String> {
    let version = user_version(conn)?;
    if version < 1 {
        conn.execute_batch(
            "
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
                map_extent_mm REAL,
                layer_count INTEGER,
                map_geom_checked INTEGER NOT NULL DEFAULT 0,
                note TEXT NOT NULL DEFAULT '',
                UNIQUE(library_id, session_id)
            );
            ",
        )
        .map_err(|err| err.to_string())?;
        conn.pragma_update(None, "user_version", SCHEMA_VERSION)
            .map_err(|err| err.to_string())?;
        return Ok(());
    }
    if version < 2 {
        conn.execute_batch(
            "
            ALTER TABLE libraries ADD COLUMN bg_subtract INTEGER NOT NULL DEFAULT 0;
            ALTER TABLE libraries ADD COLUMN calibration_mode TEXT NOT NULL DEFAULT 'off';
            ALTER TABLE libraries ADD COLUMN contour_cutoff_percentile REAL NOT NULL DEFAULT 5.0;
            ALTER TABLE libraries ADD COLUMN view_settings_imported INTEGER NOT NULL DEFAULT 0;
            ALTER TABLE libraries ADD COLUMN view_settings_rev INTEGER NOT NULL DEFAULT 0;
            ",
        )
        .map_err(|err| err.to_string())?;
    }
    if version < 3 {
        conn.execute_batch(
            "
            ALTER TABLE sessions ADD COLUMN map_extent_mm REAL;
            ALTER TABLE sessions ADD COLUMN layer_count INTEGER;
            ALTER TABLE sessions ADD COLUMN map_geom_checked INTEGER NOT NULL DEFAULT 0;
            ",
        )
        .map_err(|err| err.to_string())?;
    }
    if version < SCHEMA_VERSION {
        conn.pragma_update(None, "user_version", SCHEMA_VERSION)
            .map_err(|err| err.to_string())?;
    }
    Ok(())
}

fn ensure_library(conn: &Connection, root: &str) -> Result<i64, String> {
    conn.execute(
        "INSERT OR IGNORE INTO libraries(root_path) VALUES (?1)",
        params![root],
    )
    .map_err(|err| err.to_string())?;
    conn.query_row(
        "SELECT id FROM libraries WHERE root_path = ?1",
        params![root],
        |row| row.get(0),
    )
    .map_err(|err| err.to_string())
}

fn import_legacy(conn: &Connection, lib_id: i64, base: &Path) -> Result<(), String> {
    let (notes_done, views_done) = conn
        .query_row(
            "SELECT notes_imported, view_settings_imported FROM libraries WHERE id = ?1",
            params![lib_id],
            |row| Ok((row.get::<_, i64>(0)? != 0, row.get::<_, i64>(1)? != 0)),
        )
        .map_err(|err| err.to_string())?;
    if notes_done && views_done {
        return Ok(());
    }
    let settings = read_json(base.join("settings.json")).unwrap_or_else(|| json!({}));
    if !notes_done {
        if let Some(notes) =
            read_json(base.join("session_notes.json")).and_then(|value| value.as_object().cloned())
        {
            for (session_id, text) in notes {
                let Some(text) = text.as_str() else {
                    continue;
                };
                if text.trim().is_empty() {
                    continue;
                }
                conn.execute(
                    "INSERT INTO sessions(library_id, session_id, note) VALUES (?1, ?2, ?3)
                     ON CONFLICT(library_id, session_id) DO UPDATE SET note = excluded.note",
                    params![lib_id, session_id, text],
                )
                .map_err(|err| err.to_string())?;
            }
        }
        let selected = settings
            .get("selected_sessions")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(|item| item.as_str().map(str::to_owned))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let encoded = serde_json::to_string(&selected).map_err(|err| err.to_string())?;
        conn.execute(
            "UPDATE libraries SET selected_sessions = ?1, notes_imported = 1 WHERE id = ?2",
            params![encoded, lib_id],
        )
        .map_err(|err| err.to_string())?;
    }
    if !views_done {
        let bg_subtract = settings
            .get("bg_subtract")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let mut mode = settings
            .get("calibration_mode")
            .and_then(Value::as_str)
            .unwrap_or("off")
            .to_owned();
        if settings.get("calibration_mode").is_none() {
            if let Some(auto) = settings.get("auto_calibrate").and_then(Value::as_bool) {
                mode = if auto {
                    "per_session".to_owned()
                } else {
                    "off".to_owned()
                };
            }
        }
        if !matches!(mode.as_str(), "off" | "per_session" | "constrained") {
            mode = "off".to_owned();
        }
        let cutoff = settings
            .get("contour_cutoff_percentile")
            .and_then(Value::as_f64)
            .map(|value| value.clamp(0.0, 90.0))
            .unwrap_or(5.0);
        conn.execute(
            "UPDATE libraries SET bg_subtract = ?1, calibration_mode = ?2,
                    contour_cutoff_percentile = ?3, view_settings_imported = 1
             WHERE id = ?4",
            params![i64::from(bg_subtract), mode, cutoff, lib_id],
        )
        .map_err(|err| err.to_string())?;
    }
    Ok(())
}

fn import_app_settings_once(conn: &Connection) -> Result<(), String> {
    if prefs_get(conn, PREF_APP_SETTINGS_IMPORTED)?.is_some() {
        return Ok(());
    }
    let path = default_db_path()
        .parent()
        .unwrap_or(Path::new("."))
        .join("app_settings.json");
    if let Some(value) = read_json(path).filter(|value| value.is_object()) {
        prefs_set(conn, PREF_APP_SETTINGS, &value)?;
    }
    prefs_set(conn, PREF_APP_SETTINGS_IMPORTED, &json!(true))
}

fn delete_missing(conn: &Connection, lib_id: i64, found: &HashSet<String>) -> Result<(), String> {
    let mut statement = conn
        .prepare("SELECT session_id FROM sessions WHERE library_id = ?1")
        .map_err(|err| err.to_string())?;
    let ids = statement
        .query_map(params![lib_id], |row| row.get::<_, String>(0))
        .map_err(|err| err.to_string())?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| err.to_string())?;
    let gone: Vec<_> = ids.into_iter().filter(|id| !found.contains(id)).collect();
    if gone.is_empty() {
        return Ok(());
    }
    for session_id in &gone {
        conn.execute(
            "DELETE FROM sessions WHERE library_id = ?1 AND session_id = ?2",
            params![lib_id, session_id],
        )
        .map_err(|err| err.to_string())?;
    }
    let selected = selected_list(conn, lib_id)?
        .into_iter()
        .filter(|id| found.contains(id))
        .collect::<Vec<_>>();
    let encoded = serde_json::to_string(&selected).map_err(|err| err.to_string())?;
    let scanned = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs_f64())
        .unwrap_or(0.0);
    conn.execute(
        "UPDATE libraries SET selected_sessions = ?1, last_scan_at = ?2 WHERE id = ?3",
        params![encoded, scanned, lib_id],
    )
    .map_err(|err| err.to_string())?;
    Ok(())
}

pub(crate) fn selected_session_ids(conn: &Connection, lib_id: i64) -> Result<Vec<String>, String> {
    selected_list(conn, lib_id)
}

fn selected_list(conn: &Connection, lib_id: i64) -> Result<Vec<String>, String> {
    let raw: String = conn
        .query_row(
            "SELECT selected_sessions FROM libraries WHERE id = ?1",
            params![lib_id],
            |row| row.get(0),
        )
        .map_err(|err| err.to_string())?;
    Ok(parse_string_list(&raw))
}

fn prefs_get(conn: &Connection, key: &str) -> Result<Option<Value>, String> {
    let raw = conn
        .query_row(
            "SELECT value FROM prefs WHERE key = ?1",
            params![key],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(|err| err.to_string())?;
    Ok(raw.and_then(|text| serde_json::from_str(&text).ok()))
}

fn prefs_set(conn: &Connection, key: &str, value: &Value) -> Result<(), String> {
    let raw = serde_json::to_string(value).map_err(|err| err.to_string())?;
    conn.execute(
        "INSERT INTO prefs(key, value) VALUES (?1, ?2)
         ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        params![key, raw],
    )
    .map_err(|err| err.to_string())?;
    Ok(())
}

fn read_json(path: PathBuf) -> Option<Value> {
    let text = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&text).ok()
}

fn parse_string_list(raw: &str) -> Vec<String> {
    let Ok(Value::Array(items)) = serde_json::from_str(raw) else {
        return Vec::new();
    };
    items
        .into_iter()
        .filter_map(|item| match item {
            Value::String(text) => Some(text),
            Value::Number(number) => Some(number.to_string()),
            _ => None,
        })
        .collect()
}

struct Existing {
    size: Option<i64>,
    mtime_ns: Option<i64>,
    map_geom_checked: i64,
}

struct StoredRow {
    session_id: String,
    storage_path: String,
    date: Option<String>,
    primary_mu: Option<f64>,
    treatment_time_s: Option<i64>,
    room_number: Option<i64>,
    config_name: Option<String>,
    map_extent_mm: Option<f64>,
    layer_count: Option<i64>,
    note: String,
}

fn meta_from_stored(row: &StoredRow) -> SessionMeta {
    session_meta(
        row.date.clone(),
        row.primary_mu,
        row.treatment_time_s,
        row.room_number,
        row.config_name.clone(),
        row.map_extent_mm,
        row.layer_count,
    )
}

fn session_meta(
    date: Option<String>,
    primary_mu: Option<f64>,
    treatment_time_s: Option<i64>,
    room_number: Option<i64>,
    config_name: Option<String>,
    map_extent_mm: Option<f64>,
    layer_count: Option<i64>,
) -> SessionMeta {
    SessionMeta {
        date: date.as_deref().and_then(SummaryDate::parse_iso),
        primary_mu,
        treatment_time_s: treatment_time_s.and_then(|value| i32::try_from(value).ok()),
        room_number: room_number.and_then(|value| i32::try_from(value).ok()),
        config_name,
        map_extent_mm,
        layer_count: layer_count.and_then(|value| i32::try_from(value).ok()),
    }
}

fn strip_verbatim(path: &Path) -> String {
    let text = path.to_string_lossy();
    #[cfg(windows)]
    {
        if let Some(rest) = text.strip_prefix(r"\\?\UNC\") {
            return format!(r"\\{rest}");
        }
        if let Some(rest) = text.strip_prefix(r"\\?\") {
            return rest.to_owned();
        }
    }
    text.into_owned()
}

fn normalize_lexically(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}
