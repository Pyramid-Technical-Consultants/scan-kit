//! Sqlite store. Rust writes `user_version` 4. Python stays at 3 and only
//! migrates a file when its version is older, so it leaves the exams table.
//!
//! ponytail: one process-wide lock serializes every store call. A pool can
//! replace it if command latency shows up.

use std::collections::HashSet;
use std::path::{Component, Path, PathBuf};
use std::sync::Mutex;

use rusqlite::{params, Connection, OptionalExtension};
use scan_kit_core::{SessionMeta, SummaryDate};
use serde_json::{json, Value};

use super::discover::{Discovered, ExamHit};

const SCHEMA_VERSION: i32 = 4;
const PREF_LAST_DATA_DIR: &str = "session.last_data_dir";
const PREF_DATA_DIRS: &str = "session.data_dirs";
const EXAMS_TABLE: &str = "
CREATE TABLE IF NOT EXISTS exams (
    id INTEGER PRIMARY KEY,
    library_id INTEGER NOT NULL REFERENCES libraries(id) ON DELETE CASCADE,
    study_uid TEXT NOT NULL,
    folder_name TEXT NOT NULL,
    storage_path TEXT,
    patient_name TEXT NOT NULL DEFAULT '',
    patient_id TEXT NOT NULL DEFAULT '',
    study_date TEXT,
    study_description TEXT NOT NULL DEFAULT '',
    file_count INTEGER NOT NULL DEFAULT 0,
    UNIQUE(library_id, study_uid)
);
";
const PREF_LAST_DICOM_DIR: &str = "dicom.last_study_dir";
const PREF_APP_SETTINGS: &str = "app.settings";
const PREF_APP_SETTINGS_IMPORTED: &str = "app.settings_imported";

static STORE_LOCK: Mutex<()> = Mutex::new(());

pub fn default_db_path() -> PathBuf {
    let home = std::env::var_os("USERPROFILE")
        .filter(|value| !value.is_empty())
        .or_else(|| std::env::var_os("HOME").filter(|value| !value.is_empty()))
        .map(PathBuf::from)
        // ponytail: env-bare hosts get a CWD-relative store — a service
        // account could mint a db under its launch dir; erroring instead
        // would brick every tool on such hosts.
        .unwrap_or_else(|| PathBuf::from("."));
    home.join(".scan-kit").join("scan-kit.sqlite")
}

pub fn with_store<T>(
    path: &Path,
    body: impl FnOnce(&mut Connection) -> Result<T, String>,
) -> Result<T, String> {
    // The lock guards no data — a poisoned mutex (a panic inside a callback)
    // must not wedge every store call for the rest of the process.
    let _guard = STORE_LOCK.lock().unwrap_or_else(|err| err.into_inner());
    let mut conn = open_connection(path)?;
    body(&mut conn)
}

/// Short STORE_LOCK sections on an already-open connection — file and
/// network IO must run outside the lock, not inside a fresh with_store
/// per item.
pub(crate) fn locked<T>(body: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    let _guard = STORE_LOCK.lock().unwrap_or_else(|err| err.into_inner());
    body()
}

pub fn open_connection(path: &Path) -> Result<Connection, String> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|err| err.to_string())?;
        }
    }
    let conn = Connection::open(path).map_err(|err| err.to_string())?;
    // busy_timeout first: journal_mode needs an exclusive lock on a
    // non-WAL file, and without the timeout armed a busy peer fails it
    // instantly.
    conn.pragma_update(None, "busy_timeout", 5000)
        .map_err(|err| err.to_string())?;
    conn.pragma_update(None, "journal_mode", "WAL")
        .map_err(|err| err.to_string())?;
    conn.pragma_update(None, "foreign_keys", "ON")
        .map_err(|err| err.to_string())?;
    migrate(&conn)?;
    import_app_settings_once(&conn, path)?;
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
    Ok(prefs_get(conn, PREF_LAST_DATA_DIR)?
        .and_then(|value| value.as_str().map(str::to_owned))
        .filter(|dir| !dir.is_empty()))
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
    // Untrusted invoke args — don't persist dimensions or positions a
    // platform can't show (an i32::MIN origin restores fully off-screen).
    if !(1..=32768).contains(&width) || !(1..=32768).contains(&height) {
        return Err(format!("invalid window size {width}x{height}"));
    }
    if !(-32768..=32768).contains(&x) || !(-32768..=32768).contains(&y) {
        return Err(format!("invalid window position {x},{y}"));
    }
    prefs_patch(
        conn,
        PREF_APP_SETTINGS,
        &json!({
            "window_width": width,
            "window_height": height,
            "window_x": x,
            "window_y": y,
        }),
    )
}

pub fn last_main_tab(conn: &Connection) -> Result<Option<String>, String> {
    let settings = prefs_get(conn, PREF_APP_SETTINGS)?.unwrap_or_else(|| json!({}));
    Ok(settings
        .get("last_main_tab")
        .and_then(Value::as_str)
        .map(str::to_owned))
}

pub fn set_last_main_tab(conn: &Connection, tab: &str) -> Result<(), String> {
    prefs_patch(conn, PREF_APP_SETTINGS, &json!({ "last_main_tab": tab }))
}

pub fn last_plan_save_dir(conn: &Connection) -> Result<Option<String>, String> {
    let settings = prefs_get(conn, PREF_APP_SETTINGS)?.unwrap_or_else(|| json!({}));
    Ok(settings
        .get("last_plan_synthesis_save_dir")
        .and_then(Value::as_str)
        .filter(|dir| !dir.is_empty())
        .map(str::to_owned))
}

pub fn config_settings(conn: &Connection) -> Result<(Option<String>, bool), String> {
    let settings = prefs_get(conn, PREF_APP_SETTINGS)?.unwrap_or_else(|| json!({}));
    let dir = settings
        .get("config_dir")
        .and_then(Value::as_str)
        .filter(|dir| !dir.is_empty())
        .map(str::to_owned);
    let hide = settings
        .get("hide_unused_map2map_xml")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    Ok((dir, hide))
}

pub fn set_config_settings(
    conn: &Connection,
    dir: Option<&str>,
    hide: Option<bool>,
) -> Result<(), String> {
    let mut patch = serde_json::Map::new();
    if let Some(dir) = dir {
        patch.insert("config_dir".to_owned(), json!(dir));
    }
    if let Some(hide) = hide {
        patch.insert("hide_unused_map2map_xml".to_owned(), json!(hide));
    }
    prefs_patch(conn, PREF_APP_SETTINGS, &Value::Object(patch))
}

pub fn last_dicom_dir(conn: &Connection) -> Result<Option<String>, String> {
    Ok(prefs_get(conn, PREF_LAST_DICOM_DIR)?
        .and_then(|value| value.as_str().map(str::to_owned))
        .filter(|dir| !dir.is_empty()))
}

pub fn set_last_dicom_dir(conn: &Connection, dir: &str) -> Result<(), String> {
    prefs_set(conn, PREF_LAST_DICOM_DIR, &json!(dir))
}

pub fn set_last_plan_save_dir(conn: &Connection, dir: &str) -> Result<(), String> {
    prefs_patch(
        conn,
        PREF_APP_SETTINGS,
        &json!({ "last_plan_synthesis_save_dir": dir }),
    )
}

pub fn runner_settings(conn: &Connection) -> Result<(Option<String>, Option<String>), String> {
    let settings = prefs_get(conn, PREF_APP_SETTINGS)?.unwrap_or_else(|| json!({}));
    let text = |key: &str| {
        settings
            .get(key)
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
            .map(str::to_owned)
    };
    Ok((text("last_rci_host"), text("last_plan_runner_file_dir")))
}

pub fn set_runner_host(conn: &Connection, host: &str) -> Result<(), String> {
    set_runner_pref(conn, "last_rci_host", host)
}

pub fn set_runner_file_dir(conn: &Connection, dir: &str) -> Result<(), String> {
    set_runner_pref(conn, "last_plan_runner_file_dir", dir)
}

/// The plan CSV last uploaded to the controller; the session capture reuses it
/// as the packaged `input_map.csv`.
pub fn runner_plan_path(conn: &Connection) -> Result<Option<String>, String> {
    let settings = prefs_get(conn, PREF_APP_SETTINGS)?.unwrap_or_else(|| json!({}));
    Ok(settings
        .get("last_plan_runner_plan_path")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_owned))
}

fn set_runner_pref(conn: &Connection, key: &str, value: &str) -> Result<(), String> {
    prefs_patch(conn, PREF_APP_SETTINGS, &json!({ key: value }))
}

/// Both upload prefs in one statement - two autocommits could commit the
/// new dir while keeping a previous upload's plan CSV.
pub fn set_runner_upload(conn: &Connection, dir: &str, plan: &str) -> Result<(), String> {
    prefs_patch(
        conn,
        PREF_APP_SETTINGS,
        &json!({"last_plan_runner_file_dir": dir, "last_plan_runner_plan_path": plan}),
    )
}

/// User-registered MCsquare scanner folders, `[{label, folder}]`.
pub fn scanners(conn: &Connection) -> Result<Vec<(String, String)>, String> {
    let settings = prefs_get(conn, PREF_APP_SETTINGS)?.unwrap_or_else(|| json!({}));
    Ok(settings
        .get("scanners")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|entry| {
            let label = entry.get("label").and_then(Value::as_str)?;
            let folder = entry.get("folder").and_then(Value::as_str)?;
            Some((label.to_owned(), folder.to_owned()))
        })
        .collect())
}

pub fn set_scanners(conn: &Connection, scanners: &[(String, String)]) -> Result<(), String> {
    let list: Vec<Value> = scanners
        .iter()
        .map(|(label, folder)| json!({ "label": label, "folder": folder }))
        .collect();
    prefs_patch(conn, PREF_APP_SETTINGS, &json!({ "scanners": list }))
}

pub fn set_note(
    conn: &Connection,
    root: &str,
    session_id: &str,
    text: &str,
    legacy: Option<&LegacyFiles>,
) -> Result<(), String> {
    if !scan_kit_core::valid_session_id(session_id) {
        return Err("invalid session id".to_owned());
    }
    let lib_id = ensure_library(conn, root)?;
    if !crate::location::is_remote_location(root) {
        import_legacy(conn, lib_id, Path::new(root), legacy)?;
    }
    // Store the trimmed text — the command response already reports the
    // trimmed form, so the two must agree about what persisted.
    let note = text.trim();
    // Upsert is deliberate — a user can note a session before a scan makes
    // its row (tests pin the row surviving trash/re-import).
    // Upsert and ghost cleanup commit together — a failure between them
    // leaves an empty-note NULL-storage row delete_missing never collects,
    // a phantom catalog entry nothing can remove. BEGIN IMMEDIATE, the
    // same discipline delete_session uses.
    conn.execute_batch("BEGIN IMMEDIATE")
        .map_err(|err| err.to_string())?;
    let result = (|| {
        conn.execute(
            "INSERT INTO sessions(library_id, session_id, note) VALUES (?1, ?2, ?3)
             ON CONFLICT(library_id, session_id) DO UPDATE SET note = excluded.note",
            params![lib_id, session_id, note],
        )
        .map_err(|err| err.to_string())?;
        if note.is_empty() {
            // A whitespace note on an unscanned session creates a row with no
            // note and no storage — delete_missing keeps NULL-storage rows, so
            // it would list forever. Only truly-empty ghosts go.
            conn.execute(
                "DELETE FROM sessions WHERE library_id = ?1 AND session_id = ?2
                 AND storage_path IS NULL AND note = ''",
                params![lib_id, session_id],
            )
            .map_err(|err| err.to_string())?;
        }
        Ok(())
    })();
    if result.is_err() {
        let _ = conn.execute_batch("ROLLBACK");
        return result;
    }
    conn.execute_batch("COMMIT").map_err(|err| err.to_string())
}

pub fn set_selected(
    conn: &Connection,
    root: &str,
    session_ids: &[String],
    legacy: Option<&LegacyFiles>,
) -> Result<Vec<String>, String> {
    let mut unique = Vec::new();
    for session_id in session_ids {
        if !scan_kit_core::valid_session_id(session_id) {
            return Err(format!("invalid session id: {session_id}"));
        }
        if !unique.contains(session_id) {
            unique.push(session_id.clone());
            if unique.len() > 5 {
                return Err("at most five sessions can be selected".to_owned());
            }
        }
    }
    let lib_id = ensure_library(conn, root)?;
    if !crate::location::is_remote_location(root) {
        import_legacy(conn, lib_id, Path::new(root), legacy)?;
    }
    let encoded = serde_json::to_string(&unique).map_err(|err| err.to_string())?;
    conn.execute(
        "UPDATE libraries SET selected_sessions = ?1 WHERE id = ?2",
        params![encoded, lib_id],
    )
    .map_err(|err| err.to_string())?;
    Ok(unique)
}

pub enum Cache {
    Ready,
    Hydrate,
}

/// `fingerprint` comes from `discover::storage_fingerprint` — the caller
/// computes it BEFORE taking STORE_LOCK because a remote stat is a fresh
/// connection.
pub fn sync_entry(
    conn: &Connection,
    lib_id: i64,
    entry: &Discovered,
    fingerprint: Option<(i64, i64)>,
) -> Result<Cache, String> {
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

    // ponytail: the fingerprint was stat'ed before STORE_LOCK while the
    // caller walked storage — a file replaced in that window gets the new
    // path with the old stamp. The ceiling is one scan interval; the next
    // scan's fingerprint mismatch rehydrates. Closing it would need the
    // stat inside the lock, which puts remote connects under STORE_LOCK.
    let path = entry.storage_path.to_string_lossy().to_string();
    if let Some(existing) = existing {
        if fingerprint.is_none() {
            // A transient stat failure (UNC hiccup, locked marker) must not
            // NULL out the hydrated metadata — keep it and retry next scan.
            conn.execute(
                "UPDATE sessions SET storage_path = ?1, kind = ?2
                 WHERE library_id = ?3 AND session_id = ?4",
                params![path, entry.kind, lib_id, entry.session_id],
            )
            .map_err(|err| err.to_string())?;
            // A never-hydrated row still needs hydrate() - the stat failure
            // above is transient, not a reason to stay blank another scan.
            return Ok(if existing.map_geom_checked != 0 {
                Cache::Ready
            } else {
                Cache::Hydrate
            });
        }
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

/// `fingerprint` comes from `discover::storage_fingerprint` — the caller
/// computes it before STORE_LOCK; a remote stat is a fresh connection.
pub fn record_meta(
    conn: &Connection,
    lib_id: i64,
    entry: &Discovered,
    meta: Option<&SessionMeta>,
    checked: bool,
    fingerprint: Option<(i64, i64)>,
) -> Result<(), String> {
    // The UPDATE below leaves size/mtime_ns alone: sync_entry stamps them
    // before hydrate reads any bytes, and restatting now would certify
    // content the row does not hold — a stored fingerprint newer than the
    // hydrated bytes pins stale meta instead of re-reading it. The WHERE
    // clause binds the fingerprint hydrate certified: a peer that restamped
    // the row to newer bytes takes 0 rows and stays retryable.
    let (fp_mtime, fp_size) = fingerprint.unzip();
    let path = entry.storage_path.to_string_lossy().to_string();
    let date = meta.and_then(SessionMeta::date_iso);
    // A meta=None row only seals (checked=1) when hydrate found the storage
    // readable — otherwise the NULLs were a transient failure and the row
    // must stay retryable on the next scan.
    let checked = i32::from(checked);
    // UPDATE only — the row exists by the time meta lands (sync_entry made
    // it inside the committed tx). An INSERT would resurrect a row deleted
    // in the autocommit window with hydrated meta and a stale fingerprint.
    if checked == 0 {
        // Meta read from newer bytes than the stamp certifies must not be
        // written under the old fingerprint — keep the path current and
        // leave the row retryable for a certifying hydrate.
        conn.execute(
            "UPDATE sessions SET storage_path = ?3, kind = ?4, map_geom_checked = 0
             WHERE library_id = ?1 AND session_id = ?2
               AND size IS ?5 AND mtime_ns IS ?6",
            params![
                lib_id,
                entry.session_id,
                path,
                entry.kind,
                fp_size,
                fp_mtime
            ],
        )
        .map_err(|err| err.to_string())?;
        return Ok(());
    }
    conn.execute(
        "UPDATE sessions SET
            storage_path = ?3,
            kind = ?4,
            date = ?5,
            primary_mu = ?6,
            treatment_time_s = ?7,
            room_number = ?8,
            config_name = ?9,
            map_extent_mm = ?10,
            layer_count = ?11,
            map_geom_checked = ?12
         WHERE library_id = ?1 AND session_id = ?2
           AND size IS ?13 AND mtime_ns IS ?14",
        params![
            lib_id,
            entry.session_id,
            path,
            entry.kind,
            date,
            meta.and_then(|item| item.primary_mu),
            meta.and_then(|item| item.treatment_time_s),
            meta.and_then(|item| item.room_number),
            meta.and_then(|item| item.config_name.clone()),
            meta.and_then(|item| item.map_extent_mm),
            meta.and_then(|item| item.layer_count),
            checked,
            fp_size,
            fp_mtime,
        ],
    )
    .map_err(|err| err.to_string())?;
    Ok(())
}

/// Copy `session_notes.json` into blank SQL notes.
///
/// ponytail: runs only when this library has no notes at all, so deleting one
/// note does not revive the JSON. Clearing every note would.
pub fn backfill_blank_notes(
    conn: &Connection,
    lib_id: i64,
    notes_raw: Option<&Value>,
) -> Result<(), String> {
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
    let Some(notes) = notes_raw.and_then(|value| value.as_object()) else {
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
            params![text.trim(), lib_id, session_id],
        )
        .map_err(|err| err.to_string())?;
    }
    Ok(())
}

pub fn prepare_library(
    conn: &Connection,
    root: &str,
    legacy: Option<&LegacyFiles>,
) -> Result<i64, String> {
    let lib_id = ensure_library(conn, root)?;
    if !crate::location::is_remote_location(root) {
        import_legacy(conn, lib_id, Path::new(root), legacy)?;
    }
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

/// Read-modify-write — callers must hold a transaction (the index tx or a
/// `BEGIN IMMEDIATE`), like `forget_data_dir`, or a concurrent add is lost.
pub fn remember_data_dir(conn: &Connection, root: &str) -> Result<(), String> {
    let mut dirs = data_dirs(conn)?;
    if !dirs.iter().any(|dir| dir == root) {
        dirs.push(root.to_owned());
    }
    prefs_set(conn, PREF_DATA_DIRS, &json!(dirs))?;
    prefs_set(conn, PREF_LAST_DATA_DIR, &json!(root))
}

/// Saved roots in order. The first read of an older store turns the last folder into a one-item list.
pub fn data_dirs(conn: &Connection) -> Result<Vec<String>, String> {
    // A valid-but-wrong-shape value must not mask the legacy fallback.
    if let Some(value) = prefs_get(conn, PREF_DATA_DIRS)?.filter(|value| value.is_array()) {
        return Ok(string_list(&value));
    }
    let Some(last) = last_data_dir(conn)?.filter(|dir| !dir.is_empty()) else {
        return Ok(Vec::new());
    };
    let dirs = vec![last];
    // Fill only while the pref is still absent - an unconditional write
    // could resurrect a directory a peer just forgot mid-scan.
    let raw = serde_json::to_string(&json!(dirs)).map_err(|err| err.to_string())?;
    // The fill is a one-time migration — a pure-read caller must not fail
    // on a busy store for it; the fallback is returned and the fill lands
    // on the next uncontended call.
    if conn
        .execute(
            "INSERT INTO prefs(key, value) VALUES (?1, ?2) ON CONFLICT(key) DO NOTHING",
            params![PREF_DATA_DIRS, raw],
        )
        .is_err()
    {
        return Ok(dirs);
    }
    // A peer may have committed the real list between our read and the
    // insert — return what is stored, not our one-item fallback.
    if let Some(value) = prefs_get(conn, PREF_DATA_DIRS)?.filter(|value| value.is_array()) {
        return Ok(string_list(&value));
    }
    Ok(dirs)
}

pub fn forget_data_dir(conn: &Connection, root: &str) -> Result<(), String> {
    // The read-modify-write must hold a write lock across processes — another
    // process's commit between the read and the prefs_set would be silently
    // overwritten (a library vanishing from the catalog).
    conn.execute_batch("BEGIN IMMEDIATE")
        .map_err(|err| err.to_string())?;
    let result = (|| {
        let mut dirs = data_dirs(conn)?;
        dirs.retain(|dir| dir != root);
        prefs_set(conn, PREF_DATA_DIRS, &json!(dirs))?;
        if last_data_dir(conn)?.as_deref() == Some(root) {
            // Most recently added survivor — first() would reopen the
            // stalest location.
            prefs_set(
                conn,
                PREF_LAST_DATA_DIR,
                &json!(dirs.last().cloned().unwrap_or_default()),
            )?;
        }
        Ok(())
    })();
    if result.is_err() {
        let _ = conn.execute_batch("ROLLBACK");
        return result;
    }
    conn.execute_batch("COMMIT").map_err(|err| err.to_string())
}

fn string_list(value: &Value) -> Vec<String> {
    value
        .as_array()
        .map(|items| {
            items
                .iter()
                .filter_map(|item| {
                    item.as_str()
                        .filter(|text| !text.is_empty())
                        .map(str::to_owned)
                })
                .collect()
        })
        .unwrap_or_default()
}

pub(crate) struct StoredExam {
    pub folder_name: String,
    pub storage_path: String,
    pub patient_name: String,
    pub patient_id: String,
    pub study_date: String,
    pub study_description: String,
    pub file_count: i64,
}

pub fn library_id(conn: &Connection, root: &str) -> Result<Option<i64>, String> {
    conn.query_row(
        "SELECT id FROM libraries WHERE root_path = ?1",
        params![root],
        |row| row.get(0),
    )
    .optional()
    .map_err(|err| err.to_string())
}

pub fn upsert_exam(conn: &Connection, lib_id: i64, exam: &ExamHit) -> Result<(), String> {
    let study_date = if exam.study_date.is_empty() {
        None
    } else {
        Some(exam.study_date.as_str())
    };
    conn.execute(
        "INSERT INTO exams(
            library_id, study_uid, folder_name, storage_path, patient_name, patient_id,
            study_date, study_description, file_count
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
         ON CONFLICT(library_id, study_uid) DO UPDATE SET
            folder_name = excluded.folder_name,
            storage_path = excluded.storage_path,
            patient_name = excluded.patient_name,
            patient_id = excluded.patient_id,
            study_date = excluded.study_date,
            study_description = excluded.study_description,
            file_count = excluded.file_count",
        params![
            lib_id,
            exam.study_uid,
            exam.folder_name,
            exam.storage_path.to_string_lossy().to_string(),
            exam.patient_name,
            exam.patient_id,
            study_date,
            exam.study_description,
            exam.file_count,
        ],
    )
    .map_err(|err| err.to_string())?;
    Ok(())
}

pub fn exam_rows(conn: &Connection, lib_id: i64) -> Result<Vec<StoredExam>, String> {
    let mut statement = conn
        .prepare(
            "SELECT folder_name, storage_path, patient_name, patient_id, study_date,
                    study_description, file_count
             FROM exams WHERE library_id = ?1 ORDER BY folder_name",
        )
        .map_err(|err| err.to_string())?;
    let rows = statement
        .query_map(params![lib_id], |row| {
            Ok(StoredExam {
                folder_name: row.get(0)?,
                storage_path: row.get::<_, Option<String>>(1)?.unwrap_or_default(),
                patient_name: row.get(2)?,
                patient_id: row.get(3)?,
                study_date: row.get::<_, Option<String>>(4)?.unwrap_or_default(),
                study_description: row.get(5)?,
                file_count: row.get(6)?,
            })
        })
        .map_err(|err| err.to_string())?;
    rows.collect::<Result<Vec<_>, _>>()
        .map_err(|err| err.to_string())
}

pub(crate) fn delete_missing_exams(
    conn: &Connection,
    lib_id: i64,
    found: &HashSet<String>,
) -> Result<(), String> {
    let mut statement = conn
        .prepare("SELECT study_uid FROM exams WHERE library_id = ?1")
        .map_err(|err| err.to_string())?;
    let ids = statement
        .query_map(params![lib_id], |row| row.get::<_, String>(0))
        .map_err(|err| err.to_string())?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| err.to_string())?;
    for study_uid in ids.into_iter().filter(|id| !found.contains(id)) {
        conn.execute(
            "DELETE FROM exams WHERE library_id = ?1 AND study_uid = ?2",
            params![lib_id, study_uid],
        )
        .map_err(|err| err.to_string())?;
    }
    Ok(())
}

fn migrate(conn: &Connection) -> Result<(), String> {
    let version = user_version(conn)?;
    if version > SCHEMA_VERSION {
        return Err(format!(
            "this library was written by a newer Scan Kit (schema v{version}); upgrade to open it"
        ));
    }
    // The CREATEs heal every version — a versioned file missing a table
    // (torn creation, foreign partial write) self-repairs like an unstamped
    // one instead of failing every later open.
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
    conn.execute_batch(EXAMS_TABLE)
        .map_err(|err| err.to_string())?;
    if version < 1 {
        // Tables that predate version stamping keep their older shape —
        // heal the later columns the same way a versioned DB does.
        migrate_columns(
            conn,
            "libraries",
            &[
                ("selected_sessions", "TEXT NOT NULL DEFAULT '[]'"),
                ("notes_imported", "INTEGER NOT NULL DEFAULT 0"),
                ("last_scan_at", "REAL"),
                ("bg_subtract", "INTEGER NOT NULL DEFAULT 0"),
                ("calibration_mode", "TEXT NOT NULL DEFAULT 'off'"),
                ("contour_cutoff_percentile", "REAL NOT NULL DEFAULT 5.0"),
                ("view_settings_imported", "INTEGER NOT NULL DEFAULT 0"),
                ("view_settings_rev", "INTEGER NOT NULL DEFAULT 0"),
            ],
        )?;
        migrate_columns(
            conn,
            "sessions",
            &[
                ("storage_path", "TEXT"),
                ("kind", "TEXT"),
                ("size", "INTEGER"),
                ("mtime_ns", "INTEGER"),
                ("date", "TEXT"),
                ("primary_mu", "REAL"),
                ("treatment_time_s", "INTEGER"),
                ("room_number", "INTEGER"),
                ("config_name", "TEXT"),
                ("map_extent_mm", "REAL"),
                ("layer_count", "INTEGER"),
                ("map_geom_checked", "INTEGER NOT NULL DEFAULT 0"),
                ("note", "TEXT NOT NULL DEFAULT ''"),
            ],
        )?;
        // Tables that predate version stamping may lack the UNIQUE
        // constraints the ON CONFLICT upserts resolve against — dedupe
        // then index or every session/exam write fails on such a DB.
        // Rows dangling on a deleted/never-written libraries row get NULL
        // from the repoint subquery and fail NOT NULL — sweep them first.
        conn.execute_batch(
            "DELETE FROM sessions WHERE library_id NOT IN (SELECT id FROM libraries);
             DELETE FROM exams WHERE library_id NOT IN (SELECT id FROM libraries);
             UPDATE sessions SET library_id = (
                SELECT keep.id FROM libraries l1
                JOIN libraries keep
                    ON keep.root_path = l1.root_path
                WHERE l1.id = sessions.library_id
                ORDER BY keep.id LIMIT 1);
             UPDATE exams SET library_id = (
                SELECT keep.id FROM libraries l1
                JOIN libraries keep
                    ON keep.root_path = l1.root_path
                WHERE l1.id = exams.library_id
                ORDER BY keep.id LIMIT 1);
             DELETE FROM sessions WHERE rowid NOT IN (
                SELECT MIN(rowid) FROM sessions GROUP BY library_id, session_id);
             DELETE FROM exams WHERE rowid NOT IN (
                SELECT MIN(rowid) FROM exams GROUP BY library_id, study_uid);
             DELETE FROM libraries WHERE id NOT IN (
                SELECT MIN(id) FROM libraries GROUP BY root_path);
             CREATE UNIQUE INDEX IF NOT EXISTS sessions_unique
                ON sessions(library_id, session_id);
             CREATE UNIQUE INDEX IF NOT EXISTS exams_unique
                ON exams(library_id, study_uid);
             CREATE UNIQUE INDEX IF NOT EXISTS libraries_root_unique
                ON libraries(root_path);",
        )
        .map_err(|err| err.to_string())?;
        conn.pragma_update(None, "user_version", SCHEMA_VERSION)
            .map_err(|err| err.to_string())?;
        return Ok(());
    }
    // Steps run inside one transaction, and each column is added only when
    // absent — a mid-upgrade kill (or a foreign tool's partial migration)
    // used to leave a versioned-but-half-altered DB that bricked every
    // later open with duplicate-column errors.
    {
        migrate_columns(
            conn,
            "libraries",
            &[
                ("bg_subtract", "INTEGER NOT NULL DEFAULT 0"),
                ("calibration_mode", "TEXT NOT NULL DEFAULT 'off'"),
                ("contour_cutoff_percentile", "REAL NOT NULL DEFAULT 5.0"),
                ("view_settings_imported", "INTEGER NOT NULL DEFAULT 0"),
                ("view_settings_rev", "INTEGER NOT NULL DEFAULT 0"),
            ],
        )?;
    }
    {
        migrate_columns(
            conn,
            "sessions",
            &[
                ("map_extent_mm", "REAL"),
                ("layer_count", "INTEGER"),
                ("map_geom_checked", "INTEGER NOT NULL DEFAULT 0"),
            ],
        )?;
    }
    {
        conn.execute_batch(EXAMS_TABLE)
            .map_err(|err| err.to_string())?;
        // An exams table that exists with a stale column set (partial
        // creation at the same version) heals like sessions/libraries do.
        migrate_columns(
            conn,
            "exams",
            &[
                ("folder_name", "TEXT NOT NULL DEFAULT ''"),
                ("storage_path", "TEXT"),
                ("patient_name", "TEXT NOT NULL DEFAULT ''"),
                ("patient_id", "TEXT NOT NULL DEFAULT ''"),
                ("study_date", "TEXT"),
                ("study_description", "TEXT NOT NULL DEFAULT ''"),
                ("file_count", "INTEGER NOT NULL DEFAULT 0"),
            ],
        )?;
    }
    if version < SCHEMA_VERSION {
        conn.pragma_update(None, "user_version", SCHEMA_VERSION)
            .map_err(|err| err.to_string())?;
    }
    Ok(())
}

/// Table and column names are compile-time constants — never interpolated
/// input — so the ALTER text can be formatted safely.
fn migrate_columns(conn: &Connection, table: &str, columns: &[(&str, &str)]) -> Result<(), String> {
    conn.execute_batch("BEGIN IMMEDIATE")
        .map_err(|err| err.to_string())?;
    let result = (|| -> Result<(), String> {
        let existing: HashSet<String> = conn
            .prepare(&format!("PRAGMA table_info({table})"))
            .map_err(|err| err.to_string())?
            .query_map([], |row| row.get::<_, String>(1))
            .map_err(|err| err.to_string())?
            .collect::<Result<_, _>>()
            .map_err(|err| err.to_string())?;
        for (name, def) in columns {
            if !existing.contains(*name) {
                conn.execute_batch(&format!("ALTER TABLE {table} ADD COLUMN {name} {def}"))
                    .map_err(|err| err.to_string())?;
            }
        }
        Ok(())
    })();
    if result.is_err() {
        let _ = conn.execute_batch("ROLLBACK");
        return result;
    }
    conn.execute_batch("COMMIT").map_err(|err| err.to_string())
}

pub(crate) fn ensure_library(conn: &Connection, root: &str) -> Result<i64, String> {
    // SELECT first: INSERT OR IGNORE only ignores when the UNIQUE
    // constraint exists — a table that predates version stamping may lack
    // it, and then every call leaks a duplicate row.
    if let Some(id) = conn
        .query_row(
            "SELECT id FROM libraries WHERE root_path = ?1",
            params![root],
            |row| row.get(0),
        )
        .optional()
        .map_err(|err| err.to_string())?
    {
        return Ok(id);
    }
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

/// The legacy JSON lives on the library root — for a UNC share these reads
/// can stall for the OS timeout, so lock-sensitive callers prefetch before
/// STORE_LOCK and pass the files in.
#[derive(Clone)]
pub(crate) struct LegacyFiles {
    settings_ok: bool,
    notes_ok: bool,
    settings: Value,
    notes_raw: Option<Value>,
}

impl LegacyFiles {
    /// The prefetched session_notes.json, parsed — lets the backfill reuse
    /// it instead of re-reading inside STORE_LOCK.
    pub(crate) fn notes(&self) -> Option<&Value> {
        self.notes_raw.as_ref()
    }
}

pub fn legacy_files(base: &Path) -> LegacyFiles {
    let settings_path = base.join("settings.json");
    let settings_raw = read_json(settings_path.clone());
    // A corrupt legacy file is not an empty one — retry the import on the
    // next open instead of stamping the flag and losing the data forever.
    let settings_ok = settings_raw
        .as_ref()
        .map(|value| value.is_object())
        .unwrap_or(!settings_path.exists());
    let notes_path = base.join("session_notes.json");
    let notes_raw = read_json(notes_path.clone());
    // A parsed-but-wrong-shape file (an array, or non-string values) is a
    // corrupt one — stamping notes_imported would discard it forever.
    let notes_ok = notes_raw
        .as_ref()
        .map(|value| {
            value
                .as_object()
                .map(|map| map.values().all(Value::is_string))
                .unwrap_or(false)
        })
        .unwrap_or(!notes_path.exists());
    LegacyFiles {
        settings_ok,
        notes_ok,
        settings: settings_raw.unwrap_or_else(|| json!({})),
        notes_raw,
    }
}

pub(crate) fn import_legacy(
    conn: &Connection,
    lib_id: i64,
    base: &Path,
    legacy: Option<&LegacyFiles>,
) -> Result<(), String> {
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
    let owned;
    let files = match legacy {
        Some(files) => files,
        None => {
            owned = legacy_files(base);
            &owned
        }
    };
    let LegacyFiles {
        settings_ok,
        notes_ok,
        settings,
        notes_raw,
    } = files;
    // The notes import does not depend on a healthy settings.json — gating
    // on settings_ok used to wedge a good notes file behind a corrupt one.
    if !notes_done && *notes_ok {
        if let Some(notes) = notes_raw.as_ref().and_then(|value| value.as_object()) {
            for (session_id, text) in notes {
                let Some(text) = text.as_str() else {
                    continue;
                };
                let text = text.trim();
                if text.is_empty() || !scan_kit_core::valid_session_id(session_id) {
                    continue;
                }
                // Update only - an absent session keeps no row, so a stale
                // legacy key cannot resurrect a ghost catalog entry.
                conn.execute(
                    "UPDATE sessions SET note = ?3
                     WHERE library_id = ?1 AND session_id = ?2 AND note = ''",
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
        // Don't clobber a live selection when the legacy file was absent.
        conn.execute(
            "UPDATE libraries SET
                selected_sessions = CASE WHEN selected_sessions IS NULL OR selected_sessions = '[]'
                    THEN ?1 ELSE selected_sessions END,
                notes_imported = 1
             WHERE id = ?2",
            params![encoded, lib_id],
        )
        .map_err(|err| err.to_string())?;
    }
    if !views_done && *settings_ok {
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

fn import_app_settings_once(conn: &Connection, db_path: &Path) -> Result<(), String> {
    // Check-then-act must hold the write lock across processes — a peer that
    // read "unmarked" before our commit would otherwise replay the wholesale
    // prefs_set over keys the user wrote in between (forget_data_dir pattern).
    // Read the marker before taking the write lock — a pure read (data_dirs,
    // window_geometry) must not wait on a busy writer. Read the raw value —
    // prefs_get maps corrupt JSON to None, which would re-import the legacy
    // file and revert every setting on a bad row.
    let marked: Option<String> = conn
        .query_row(
            "SELECT value FROM prefs WHERE key = ?1",
            params![PREF_APP_SETTINGS_IMPORTED],
            |row| row.get(0),
        )
        .optional()
        .map_err(|err| err.to_string())?;
    if marked.is_some() {
        return Ok(());
    }
    // The legacy file sits beside the database — a custom db imports its
    // own profile, not the default one's. Read before the write tx: a
    // stalled share must not sit inside BEGIN IMMEDIATE.
    let path = db_path
        .parent()
        .unwrap_or(Path::new("."))
        .join("app_settings.json");
    let exists = path.exists();
    let value = exists.then(|| read_json(path)).flatten();
    conn.execute_batch("BEGIN IMMEDIATE")
        .map_err(|err| err.to_string())?;
    let result = import_app_settings_body(conn, exists, value);
    if result.is_err() {
        let _ = conn.execute_batch("ROLLBACK");
        return result;
    }
    conn.execute_batch("COMMIT").map_err(|err| err.to_string())
}

fn import_app_settings_body(
    conn: &Connection,
    exists: bool,
    value: Option<Value>,
) -> Result<(), String> {
    if exists {
        let Some(value) = value.filter(|value| value.is_object()) else {
            // A corrupt file is not an empty one — retry the import on the
            // next open instead of stamping the flag and losing the data
            // forever (same rule as import_legacy).
            return Ok(());
        };
        // The import fills an empty store — replacing live settings would
        // silently revert window geometry, scanners, and runner prefs to the
        // stale file on any DB that predates the marker.
        let settings_empty = conn
            .query_row(
                "SELECT value IS NULL FROM prefs WHERE key = ?1",
                params![PREF_APP_SETTINGS],
                |row| row.get::<_, bool>(0),
            )
            .optional()
            .map_err(|err| err.to_string())?
            .unwrap_or(true);
        if settings_empty {
            prefs_set(conn, PREF_APP_SETTINGS, &value)?;
        }
    }
    prefs_set(conn, PREF_APP_SETTINGS_IMPORTED, &json!(true))
}

pub(crate) fn delete_missing(
    conn: &Connection,
    lib_id: i64,
    found: &HashSet<String>,
) -> Result<(), String> {
    let mut statement = conn
        .prepare("SELECT session_id, storage_path IS NOT NULL FROM sessions WHERE library_id = ?1")
        .map_err(|err| err.to_string())?;
    let rows = statement
        .query_map(params![lib_id], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, bool>(1)?))
        })
        .map_err(|err| err.to_string())?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| err.to_string())?;
    let gone: Vec<_> = rows
        .into_iter()
        .filter(|(id, _)| !found.contains(id))
        .collect();
    for (session_id, had_storage) in &gone {
        // A case-only rename on a case-insensitive filesystem yields a new
        // id for the same file - rekey the row instead of losing the note.
        // Two same-case-insensitive ids can legitimately coexist on a
        // case-sensitive filesystem — rekeying on an arbitrary match could
        // attach this row's note to the wrong survivor.
        let mut matches = found
            .iter()
            .filter(|id| id.eq_ignore_ascii_case(session_id));
        let renamed = match (matches.next(), matches.next()) {
            (Some(only), None) => Some(only.clone()),
            _ => None,
        };
        if let Some(renamed) = renamed {
            let survives: bool = conn
                .query_row(
                    "SELECT EXISTS(
                        SELECT 1 FROM sessions
                        WHERE library_id = ?1 AND session_id = ?2)",
                    params![lib_id, renamed],
                    |row| row.get(0),
                )
                .map_err(|err| err.to_string())?;
            if survives {
                // The new casing already has a row - carry the note over a
                // blank one and drop the gone row; a blind rekey would hit
                // UNIQUE(library_id, session_id) and abort the whole sync.
                conn.execute(
                    "UPDATE sessions SET note = (
                        SELECT note FROM sessions
                        WHERE library_id = ?1 AND session_id = ?2)
                     WHERE library_id = ?1 AND session_id = ?3 AND note = ''",
                    params![lib_id, session_id, renamed],
                )
                .map_err(|err| err.to_string())?;
                conn.execute(
                    "DELETE FROM sessions WHERE library_id = ?1 AND session_id = ?2",
                    params![lib_id, session_id],
                )
                .map_err(|err| err.to_string())?;
            } else {
                conn.execute(
                    "UPDATE sessions SET session_id = ?1
                     WHERE library_id = ?2 AND session_id = ?3",
                    params![renamed, lib_id, session_id],
                )
                .map_err(|err| err.to_string())?;
            }
            continue;
        }
        // A note-before-scan row has no storage_path — it was never an
        // indexed session, so a scan that misses the files must not GC the
        // user's note. Rows that did index a file GC as before.
        if *had_storage {
            conn.execute(
                "DELETE FROM sessions WHERE library_id = ?1 AND session_id = ?2",
                params![lib_id, session_id],
            )
            .map_err(|err| err.to_string())?;
        }
    }
    // last_scan_at marks the last scan, not the last scan that deleted a row.
    let selected = selected_list(conn, lib_id)?.into_iter().filter_map(|id| {
        // Case-renamed ids adopt the on-disk casing like the rows above.
        if found.contains(&id) {
            Some(id)
        } else {
            // Same rule as the row rekey above — an ambiguous case match
            // must not hand the selection to an arbitrary survivor.
            let mut matches = found.iter().filter(|f| f.eq_ignore_ascii_case(&id));
            match (matches.next(), matches.next()) {
                (Some(only), None) => Some(only.clone()),
                _ => None,
            }
        }
    });
    let mut seen = std::collections::HashSet::new();
    let selected: Vec<String> = selected
        .into_iter()
        .filter(|id| seen.insert(id.clone()))
        .collect();
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

/// Remove a session row and drop it from the selection (session files are
/// removed by the caller — trash or remote delete).
pub fn delete_session(conn: &Connection, lib_id: i64, session_id: &str) -> Result<(), String> {
    // The session's files are already gone when this runs — a mid-call
    // failure must not leave a deleted id inside selected_sessions.
    // BEGIN IMMEDIATE — the delete-then-update read-modify-write must not
    // reopen the cross-process window a deferred BEGIN would.
    conn.execute_batch("BEGIN IMMEDIATE")
        .map_err(|err| err.to_string())?;
    let result = (|| {
        conn.execute(
            "DELETE FROM sessions WHERE library_id = ?1 AND session_id = ?2",
            params![lib_id, session_id],
        )
        .map_err(|err| err.to_string())?;
        let selected: Vec<String> = selected_list(conn, lib_id)?
            .into_iter()
            .filter(|id| id != session_id)
            .collect();
        let encoded = serde_json::to_string(&selected).map_err(|err| err.to_string())?;
        conn.execute(
            "UPDATE libraries SET selected_sessions = ?1 WHERE id = ?2",
            params![encoded, lib_id],
        )
        .map_err(|err| err.to_string())?;
        Ok(())
    })();
    if result.is_err() {
        let _ = conn.execute_batch("ROLLBACK");
        return result;
    }
    conn.execute_batch("COMMIT").map_err(|err| err.to_string())
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

/// Merge keys into a JSON pref inside one statement — a read-modify-write
/// outside the row loses sibling keys another writer set in between.
fn prefs_patch(conn: &Connection, key: &str, patch: &Value) -> Result<(), String> {
    let raw = serde_json::to_string(patch).map_err(|err| err.to_string())?;
    conn.execute(
        "INSERT INTO prefs(key, value) VALUES (?1, ?2)
         ON CONFLICT(key) DO UPDATE SET value =
            json_patch(CASE WHEN json_valid(value) THEN value ELSE '{}' END, ?2)",
        params![key, raw],
    )
    .map_err(|err| err.to_string())?;
    Ok(())
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

/// `explorer.exe` and shells cannot parse `\?\` verbatim paths — strip
/// the prefix canonicalize adds on Windows.
pub fn strip_verbatim(path: &Path) -> String {
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
                // Only a Normal component pops — popping RootDir/Prefix
                // would yield a relative path stored as canonical.
                if matches!(out.components().next_back(), Some(Component::Normal(_))) {
                    out.pop();
                }
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_replayed_legacy_import_keeps_a_live_note() {
        let path = temp_db("legacy_note");
        let conn = open_connection(&path).unwrap();
        let base = std::env::temp_dir().join(format!("sk_lib_{}", std::process::id()));
        std::fs::create_dir_all(&base).unwrap();
        std::fs::write(base.join("settings.json"), "{}").unwrap();
        std::fs::write(base.join("session_notes.json"), r#"{"s1": "legacy note"}"#).unwrap();
        let root = base.to_string_lossy().to_string();
        let lib_id = prepare_library(&conn, &root, None).unwrap();
        set_note(&conn, &root, "s1", "user note", None).unwrap();
        // A crash before the flag commit replays the import — the upsert must
        // fill blanks only, never revert a live note to the legacy text.
        conn.execute(
            "UPDATE libraries SET notes_imported = 0 WHERE id = ?1",
            params![lib_id],
        )
        .unwrap();
        set_note(&conn, &root, "s2", "other", None).unwrap();
        let note: String = conn
            .query_row(
                "SELECT note FROM sessions WHERE library_id = ?1 AND session_id = 's1'",
                params![lib_id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(note, "user note");
        let _ = std::fs::remove_dir_all(&base);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn record_meta_does_not_resurrect_a_deleted_row() {
        let path = temp_db("no_resurrect");
        let conn = open_connection(&path).unwrap();
        let root = std::env::temp_dir()
            .join("sk_meta_root")
            .to_string_lossy()
            .to_string();
        let lib_id = ensure_library(&conn, &root).unwrap();
        let entry = Discovered {
            session_id: "gone".into(),
            storage_path: PathBuf::from("gone.zip"),
            kind: "zip",
        };
        // A delete racing the hydrate used to see the row re-INSERTed with
        // hydrated meta — meta may only update a row sync_entry already made.
        record_meta(&conn, lib_id, &entry, None, true, None).unwrap();
        let rows: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sessions WHERE library_id = ?1 AND session_id = 'gone'",
                params![lib_id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(rows, 0);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_case_only_rename_rekeys_the_row_and_keeps_the_note() {
        // NTFS case-folds: sess.zip -> SESS.zip is a new id for the same
        // file, and the stored note must follow the rename.
        let path = temp_db("case_rename");
        let conn = open_connection(&path).unwrap();
        let root = std::env::temp_dir()
            .join("sk_case_root")
            .to_string_lossy()
            .to_string();
        let lib_id = ensure_library(&conn, &root).unwrap();
        conn.execute(
            "INSERT INTO sessions(library_id, session_id, note) VALUES (?1, 'sess', 'keep me')",
            params![lib_id],
        )
        .unwrap();
        conn.execute(
            "UPDATE libraries SET selected_sessions = '[\"sess\"]' WHERE id = ?1",
            params![lib_id],
        )
        .unwrap();
        let found: HashSet<String> = ["SESS".to_owned()].into_iter().collect();
        delete_missing(&conn, lib_id, &found).unwrap();
        let note: String = conn
            .query_row(
                "SELECT note FROM sessions WHERE library_id = ?1 AND session_id = 'SESS'",
                params![lib_id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(note, "keep me");
        let selected: String = conn
            .query_row(
                "SELECT selected_sessions FROM libraries WHERE id = ?1",
                params![lib_id],
                |row| row.get(0),
            )
            .unwrap();
        assert!(selected.contains("SESS"), "selected={selected}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn set_note_stores_the_trimmed_text() {
        let path = temp_db("trim_note");
        let conn = open_connection(&path).unwrap();
        let root = std::env::temp_dir()
            .join("sk_note_root")
            .to_string_lossy()
            .to_string();
        // The command reports the trimmed text — the row must agree with it.
        set_note(&conn, &root, "s1", "  padded note  ", None).unwrap();
        let note: String = conn
            .query_row(
                "SELECT note FROM sessions WHERE session_id = 's1'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(note, "padded note");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_live_setting_survives_a_late_import() {
        let dir = std::env::temp_dir().join(format!("sk_late_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("store.db");
        let conn = open_connection(&path).unwrap();
        prefs_set(&conn, PREF_APP_SETTINGS, &json!({"window": "live"})).unwrap();
        // (The file body is what import prefetches — passed directly.)
        // A DB that predates the import marker must not have its live
        // settings reverted by the stale legacy file.
        import_app_settings_body(&conn, true, Some(json!({"window": "stale"}))).unwrap();
        assert_eq!(
            prefs_get(&conn, PREF_APP_SETTINGS).unwrap(),
            Some(json!({"window": "live"}))
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_array_shaped_notes_file_stays_retryable() {
        let path = temp_db("notes_shape");
        let conn = open_connection(&path).unwrap();
        let base = std::env::temp_dir().join(format!("sk_lib_shape_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();
        std::fs::write(
            base.join("session_notes.json"),
            r#"["not", "an", "object"]"#,
        )
        .unwrap();
        let root = base.to_string_lossy().to_string();
        let lib_id = ensure_library(&conn, &root).unwrap();
        // Parsed-but-wrong-shape is corrupt — stamping notes_imported would
        // discard it forever instead of retrying a fixed file.
        import_legacy(&conn, lib_id, Path::new(&root), None).unwrap();
        let stamped: i64 = conn
            .query_row(
                "SELECT notes_imported FROM libraries WHERE id = ?1",
                params![lib_id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(stamped, 0);
        let _ = std::fs::remove_dir_all(&base);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_version_stamped_db_missing_columns_heals_on_open() {
        let path = temp_db("heal");
        {
            let conn = open_connection(&path).unwrap();
            // The partial migration the header comment describes: the column
            // is gone but the version stamp survived, so the versioned gates
            // would never re-add it.
            conn.execute_batch("ALTER TABLE libraries DROP COLUMN bg_subtract")
                .unwrap();
            conn.execute_batch("ALTER TABLE sessions DROP COLUMN map_extent_mm")
                .unwrap();
            conn.pragma_update(None, "user_version", 2).unwrap();
        }
        let conn = open_connection(&path).unwrap();
        for (table, column) in [("libraries", "bg_subtract"), ("sessions", "map_extent_mm")] {
            let count: i64 = conn
                .query_row(
                    &format!(
                        "SELECT COUNT(*) FROM pragma_table_info('{table}') WHERE name = '{column}'"
                    ),
                    [],
                    |row| row.get(0),
                )
                .unwrap();
            assert_eq!(count, 1, "{table}.{column} did not heal");
        }
    }

    fn temp_db(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("sk_store_{tag}_{}", std::process::id()));
        // Windows recycles pids fast enough that a fresh test binary can
        // inherit a killed run's fixture dir — a stamped store.db or a
        // leftover settings file then corrupts the first assertion.
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir.join("store.db")
    }

    #[test]
    fn a_settings_patch_merges_instead_of_replacing() {
        let path = temp_db("patch");
        let conn = open_connection(&path).unwrap();
        set_scanners(&conn, &[("label".into(), "folder".into())]).unwrap();
        set_window_geometry(&conn, 800, 600, 1, 2).unwrap();
        // The second write must not drop the first write's key.
        assert_eq!(
            scanners(&conn).unwrap(),
            vec![("label".to_owned(), "folder".to_owned())]
        );
        let geo = window_geometry(&conn).unwrap();
        assert_eq!(geo["width"], json!(800));
        set_last_main_tab(&conn, "library").unwrap();
        assert_eq!(scanners(&conn).unwrap().len(), 1);
        assert_eq!(geo["height"], json!(600));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_non_array_data_dirs_value_falls_back_to_last_dir() {
        let path = temp_db("dirs");
        let conn = open_connection(&path).unwrap();
        prefs_set(&conn, PREF_LAST_DATA_DIR, &json!("C:\\data")).unwrap();
        // A valid-but-wrong-shape value must not mask the legacy fallback.
        prefs_set(&conn, PREF_DATA_DIRS, &json!("not-an-array")).unwrap();
        assert_eq!(data_dirs(&conn).unwrap(), vec!["C:\\data".to_owned()]);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_newer_schema_is_refused() {
        let path = temp_db("future");
        let conn = open_connection(&path).unwrap();
        conn.pragma_update(None, "user_version", SCHEMA_VERSION + 1)
            .unwrap();
        drop(conn);
        let err = open_connection(&path).unwrap_err();
        assert!(err.contains("newer Scan Kit"), "{err}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_partly_migrated_schema_recovers() {
        let path = temp_db("migrate");
        let conn = open_connection(&path).unwrap();
        // Simulate a kill that left version 1 with one v2 column applied.
        conn.execute_batch(
            "ALTER TABLE libraries DROP COLUMN view_settings_rev;
             PRAGMA user_version = 1;",
        )
        .unwrap();
        drop(conn);
        let conn = open_connection(&path).unwrap();
        let version: i64 = conn
            .query_row("PRAGMA user_version", [], |row| row.get(0))
            .unwrap();
        assert_eq!(version, SCHEMA_VERSION as i64);
        let count: i64 = conn
            .query_row(
                "SELECT count(*) FROM pragma_table_info('libraries')
                 WHERE name = 'view_settings_rev'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 1);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_corrupt_import_flag_does_not_reimport() {
        let path = temp_db("import");
        let conn = open_connection(&path).unwrap();
        // A row exists but holds non-JSON — prefs_get reads None, the raw
        // check must still see it.
        conn.execute(
            "INSERT OR REPLACE INTO prefs(key, value) VALUES (?1, 'not json')",
            params![PREF_APP_SETTINGS_IMPORTED],
        )
        .unwrap();
        std::fs::write(
            path.parent().unwrap().join("app_settings.json"),
            r#"{"config_dir":"legacy"}"#,
        )
        .unwrap();
        import_app_settings_once(&conn, &path).unwrap();
        assert!(prefs_get(&conn, PREF_APP_SETTINGS).unwrap().is_none());
        let _ = std::fs::remove_file(path.parent().unwrap().join("app_settings.json"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_nonpositive_window_size_is_rejected() {
        let conn = open_connection(&temp_db("geometry")).unwrap();
        assert!(set_window_geometry(&conn, 0, 600, 0, 0).is_err());
        assert!(set_window_geometry(&conn, 800, -1, 0, 0).is_err());
        assert!(set_window_geometry(&conn, 800, 600, 0, 0).is_ok());
    }

    #[test]
    fn a_corrupt_settings_file_is_retried_not_stamped() {
        let path = temp_db("corrupt_settings");
        let settings = path.parent().unwrap().join("app_settings.json");
        std::fs::write(&settings, "{not json").unwrap();
        let conn = open_connection(&path).unwrap();
        // The corrupt file must not stamp the import flag — fixing the file
        // should import on the next pass.
        assert!(prefs_get(&conn, PREF_APP_SETTINGS_IMPORTED)
            .unwrap()
            .is_none());
        std::fs::write(&settings, r#"{"config_dir":"legacy"}"#).unwrap();
        import_app_settings_once(&conn, &path).unwrap();
        assert_eq!(
            prefs_get(&conn, PREF_APP_SETTINGS).unwrap(),
            Some(json!({"config_dir": "legacy"}))
        );
        let _ = std::fs::remove_file(&settings);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_failed_hydrate_stays_retryable() {
        let path = temp_db("meta_checked");
        let root = path.with_file_name("lib");
        std::fs::create_dir_all(&root).unwrap();
        let conn = open_connection(&path).unwrap();
        let lib_id = prepare_library(&conn, &root.to_string_lossy(), None).unwrap();
        let entry = Discovered {
            session_id: "s1".to_owned(),
            storage_path: path.with_file_name("no-such-session.zip"),
            kind: "zip",
        };
        // checked=false: the storage was unreachable, so NULL meta is a
        // transient failure, not an absence — the next scan retries.
        sync_entry(&conn, lib_id, &entry, None).unwrap();
        record_meta(&conn, lib_id, &entry, None, false, None).unwrap();
        let checked: i64 = conn
            .query_row(
                "SELECT map_geom_checked FROM sessions
                 WHERE library_id = ?1 AND session_id = 's1'",
                params![lib_id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(checked, 0);
        // checked=true: readable storage without meta — seal so scans skip it.
        sync_entry(&conn, lib_id, &entry, None).unwrap();
        record_meta(&conn, lib_id, &entry, None, true, None).unwrap();
        let checked: i64 = conn
            .query_row(
                "SELECT map_geom_checked FROM sessions
                 WHERE library_id = ?1 AND session_id = 's1'",
                params![lib_id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(checked, 1);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn delete_session_drops_the_selection_entry() {
        let path = temp_db("del_session");
        let root = path.with_file_name("lib");
        std::fs::create_dir_all(&root).unwrap();
        let conn = open_connection(&path).unwrap();
        let lib_id = prepare_library(&conn, &root.to_string_lossy(), None).unwrap();
        let entry = Discovered {
            session_id: "keep".to_owned(),
            storage_path: root.join("keep.zip"),
            kind: "zip",
        };
        sync_entry(&conn, lib_id, &entry, None).unwrap();
        record_meta(&conn, lib_id, &entry, None, true, None).unwrap();
        let entry = Discovered {
            session_id: "drop".to_owned(),
            storage_path: root.join("drop.zip"),
            kind: "zip",
        };
        sync_entry(&conn, lib_id, &entry, None).unwrap();
        record_meta(&conn, lib_id, &entry, None, true, None).unwrap();
        conn.execute(
            "UPDATE libraries SET selected_sessions = ?1 WHERE id = ?2",
            params![r#"["keep","drop"]"#, lib_id],
        )
        .unwrap();
        delete_session(&conn, lib_id, "drop").unwrap();
        assert_eq!(
            selected_list(&conn, lib_id).unwrap(),
            vec!["keep".to_owned()]
        );
        let count: i64 = conn
            .query_row(
                "SELECT count(*) FROM sessions WHERE library_id = ?1 AND session_id = 'drop'",
                params![lib_id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 0);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn forget_data_dir_commits_the_whole_update() {
        // The read-modify-write runs under BEGIN IMMEDIATE — the catalog list
        // and the last-dir pointer must move together.
        let path = temp_db("forget");
        let conn = open_connection(&path).unwrap();
        remember_data_dir(&conn, "/one").unwrap();
        remember_data_dir(&conn, "/two").unwrap();
        forget_data_dir(&conn, "/two").unwrap();
        assert_eq!(data_dirs(&conn).unwrap(), vec!["/one".to_owned()]);
        assert_eq!(last_data_dir(&conn).unwrap().as_deref(), Some("/one"));
        forget_data_dir(&conn, "/one").unwrap();
        assert!(data_dirs(&conn).unwrap().is_empty());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn an_off_screen_window_origin_is_rejected() {
        let path = temp_db("geompos");
        let conn = open_connection(&path).unwrap();
        assert!(set_window_geometry(&conn, 800, 600, i32::MIN, 0).is_err());
        assert!(set_window_geometry(&conn, 800, 600, 0, i32::MAX).is_err());
        set_window_geometry(&conn, 800, 600, -100, 100).unwrap();
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }
}
