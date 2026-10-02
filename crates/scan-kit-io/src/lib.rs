//! Session discovery, the sqlite store, and columnar loads.
//!
//! The desktop shell and the MCP server call these functions. They do not
//! reimplement them.

mod analysis;
mod beam;
mod binned;
mod columns;
mod discover;
mod dose_view;
mod mc_tables;
mod patient_view;
mod store;

pub use analysis::{analysis_scene, channel_catalog, load_timeslice_columns};
pub use beam::{beam_record, protons_per_mu, spot_record};
pub use dose_view::dose_volume;
pub use mc_tables::mc_tables;

use std::path::{Path, PathBuf};

use scan_kit_core::{
    merge_session_geom, parse_termination_summary_text, SessionMeta, ToolKind, ToolSpec,
};
use serde_json::{json, Value};

use columns::map_geom_from_spots;
use discover::Discovered;
use store::{
    backfill_blank_notes, canonical_local, default_db_path, last_main_tab, library_rows,
    prepare_library, record_meta, remember_data_dir, set_last_main_tab, set_note, set_selected,
    sync_entry, window_geometry, with_store, Cache,
};

const TOOLS: &[ToolSpec] = &[
    ToolSpec {
        name: "scan_kit_open_library",
        summary: "Discover a data folder, sync the session index, and return the rows.",
        kind: ToolKind::Workflow,
    },
    ToolSpec {
        name: "scan_kit_set_note",
        summary: "Store one session note in the sqlite index.",
        kind: ToolKind::Granular,
    },
    ToolSpec {
        name: "scan_kit_select_sessions",
        summary: "Store the selected session ids, at most five.",
        kind: ToolKind::Granular,
    },
    ToolSpec {
        name: "scan_kit_load_columns",
        summary: "Load named session columns after alias resolution and the G2 current scale.",
        kind: ToolKind::Granular,
    },
    ToolSpec {
        name: "scan_kit_load_timeslice",
        summary: "List device-unit timeslice columns and the input_map energy lookup length.",
        kind: ToolKind::Granular,
    },
    ToolSpec {
        name: "scan_kit_channel_catalog",
        summary: "List timeslice channels present for Replay, FFT, and Audio.",
        kind: ToolKind::Granular,
    },
    ToolSpec {
        name: "scan_kit_analysis_scene",
        summary: "Build one analysis view scene from the selected sessions.",
        kind: ToolKind::Workflow,
    },
];

pub fn tools() -> &'static [ToolSpec] {
    TOOLS
}

pub fn tool_input_schema(name: &str) -> Value {
    match name {
        "scan_kit_open_library" => json!({
            "type": "object",
            "properties": {
                "path": { "type": "string" },
                "db_path": { "type": "string" }
            },
            "required": ["path"],
            "additionalProperties": false
        }),
        "scan_kit_set_note" => json!({
            "type": "object",
            "properties": {
                "path": { "type": "string" },
                "session_id": { "type": "string" },
                "note": { "type": "string" },
                "db_path": { "type": "string" }
            },
            "required": ["path", "session_id", "note"],
            "additionalProperties": false
        }),
        "scan_kit_select_sessions" => json!({
            "type": "object",
            "properties": {
                "path": { "type": "string" },
                "session_ids": { "type": "array", "items": { "type": "string" } },
                "db_path": { "type": "string" }
            },
            "required": ["path", "session_ids"],
            "additionalProperties": false
        }),
        "scan_kit_load_columns" => json!({
            "type": "object",
            "properties": {
                "storage_path": { "type": "string" },
                "session_id": { "type": "string" },
                "columns": { "type": "array", "items": { "type": "string" } }
            },
            "required": ["storage_path", "session_id", "columns"],
            "additionalProperties": false
        }),
        "scan_kit_load_timeslice" | "scan_kit_channel_catalog" => json!({
            "type": "object",
            "properties": {
                "path": { "type": "string" },
                "session_id": { "type": "string" }
            },
            "required": ["path", "session_id"],
            "additionalProperties": false
        }),
        "scan_kit_analysis_scene" => json!({
            "type": "object",
            "properties": {
                "view": { "type": "string" },
                "path": { "type": "string" },
                "session_ids": { "type": "array", "items": { "type": "string" } },
                "options": { "type": "object" }
            },
            "required": ["view", "path", "session_ids"],
            "additionalProperties": false
        }),
        _ => json!({ "type": "object", "additionalProperties": false }),
    }
}

#[derive(Debug)]
pub enum InvokeError {
    UnknownTool { name: String },
    Message(String),
}

impl std::fmt::Display for InvokeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownTool { name } => write!(f, "unknown tool {name}"),
            Self::Message(message) => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for InvokeError {}

pub fn invoke(name: &str, input: &Value) -> Result<Value, InvokeError> {
    match name {
        "scan_kit_open_library" => {
            let path = required_str(input, "path")?;
            let db = database_path(input)?;
            with_store(&db, |conn| open_library(conn, Path::new(path)))
                .map_err(InvokeError::Message)
        }
        "scan_kit_set_note" => {
            let path = required_str(input, "path")?;
            let session_id = required_str(input, "session_id")?;
            let note = required_str(input, "note")?;
            let db = database_path(input)?;
            with_store(&db, |conn| {
                let root = library_root(path)?;
                set_note(conn, &root, session_id, note)
            })
            .map_err(InvokeError::Message)?;
            Ok(json!({ "session_id": session_id, "note": note.trim() }))
        }
        "scan_kit_select_sessions" => {
            let path = required_str(input, "path")?;
            let session_ids = required_strings(input, "session_ids")?;
            let db = database_path(input)?;
            with_store(&db, |conn| {
                let root = library_root(path)?;
                set_selected(conn, &root, &session_ids)?;
                Ok(json!({ "session_ids": session_ids }))
            })
            .map_err(InvokeError::Message)
        }
        "scan_kit_load_columns" => {
            let storage = required_str(input, "storage_path")?;
            let session_id = required_str(input, "session_id")?;
            let columns = required_strings(input, "columns")?;
            columns::load_columns(Path::new(storage), session_id, &columns)
                .map_err(InvokeError::Message)
        }
        "scan_kit_load_timeslice" => {
            let path = required_str(input, "path")?;
            let session_id = required_str(input, "session_id")?;
            Ok(load_timeslice_columns(Path::new(path), session_id))
        }
        "scan_kit_channel_catalog" => {
            let path = required_str(input, "path")?;
            let session_id = required_str(input, "session_id")?;
            Ok(json!({ "channels": channel_catalog(Path::new(path), session_id) }))
        }
        "scan_kit_analysis_scene" => {
            let view = required_str(input, "view")?;
            let path = required_str(input, "path")?;
            let session_ids = required_strings(input, "session_ids")?;
            let options = input.get("options").cloned().unwrap_or_else(|| json!({}));
            let scene = analysis_scene(view, Path::new(path), &session_ids, &options)
                .map_err(InvokeError::Message)?;
            serde_json::to_value(scene).map_err(|err| InvokeError::Message(err.to_string()))
        }
        _ => Err(InvokeError::UnknownTool {
            name: name.to_owned(),
        }),
    }
}

pub fn read_last_data_dir() -> Result<Option<String>, String> {
    with_store(&default_db_path(), |conn| store::last_data_dir(conn))
}

pub fn read_window_geometry() -> Result<Value, String> {
    with_store(&default_db_path(), |conn| window_geometry(conn))
}

pub fn write_window_geometry(width: i32, height: i32, x: i32, y: i32) -> Result<(), String> {
    with_store(&default_db_path(), |conn| {
        store::set_window_geometry(conn, width, height, x, y)
    })
}

pub fn read_last_main_tab() -> Result<Option<String>, String> {
    with_store(&default_db_path(), |conn| last_main_tab(conn))
}

pub fn write_last_main_tab(tab: &str) -> Result<(), String> {
    with_store(&default_db_path(), |conn| set_last_main_tab(conn, tab))
}

/// Workflow `scan_kit_open_library`. Discovers, syncs the index, and returns rows.
pub fn open_library(conn: &mut rusqlite::Connection, data_dir: &Path) -> Result<Value, String> {
    if !data_dir.is_dir() {
        return Err(format!("{} is not a directory", data_dir.display()));
    }
    let root = canonical_local(data_dir);
    let tx = conn.transaction().map_err(|err| err.to_string())?;
    let lib_id = prepare_library(&tx, &root)?;
    for entry in discover::discover_entries(Path::new(&root))? {
        if let Cache::Hydrate = sync_entry(&tx, lib_id, &entry)? {
            let meta = hydrate(&entry);
            record_meta(&tx, lib_id, &entry, meta.as_ref())?;
        }
    }
    remember_data_dir(&tx, &root)?;
    backfill_blank_notes(&tx, lib_id, Path::new(&root))?;
    let rows = library_rows(&tx, lib_id)?;
    let selected = store::selected_session_ids(&tx, lib_id)?;
    tx.commit().map_err(|err| err.to_string())?;
    Ok(json!({ "root": root, "rows": rows, "selected": selected }))
}

fn hydrate(entry: &Discovered) -> Option<SessionMeta> {
    let mut meta = discover::read_session_file(
        &entry.storage_path,
        &entry.session_id,
        "termination_summary.txt",
    )
    .map(|bytes| {
        let text = String::from_utf8_lossy(&bytes);
        parse_termination_summary_text(&text)
    });
    if geom_incomplete(meta.as_ref()) {
        let (extent, layers) = map_geom_from_spots(entry);
        meta = merge_session_geom(meta, extent, layers);
    }
    meta
}

fn geom_incomplete(meta: Option<&SessionMeta>) -> bool {
    match meta {
        None => true,
        Some(meta) => meta.map_extent_mm.is_none() || meta.layer_count.is_none(),
    }
}

fn library_root(path: &str) -> Result<String, String> {
    let path = Path::new(path);
    if !path.is_dir() {
        return Err(format!("{} is not a directory", path.display()));
    }
    Ok(canonical_local(path))
}

fn database_path(input: &Value) -> Result<PathBuf, InvokeError> {
    match input.get("db_path").and_then(Value::as_str) {
        Some(path) => Ok(PathBuf::from(path)),
        None => Ok(default_db_path()),
    }
}

fn required_str<'a>(input: &'a Value, key: &str) -> Result<&'a str, InvokeError> {
    input
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| InvokeError::Message(format!("{key} is required")))
}

fn required_strings(input: &Value, key: &str) -> Result<Vec<String>, InvokeError> {
    let items = input
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| InvokeError::Message(format!("{key} is required")))?;
    items
        .iter()
        .map(|item| {
            item.as_str()
                .map(str::to_owned)
                .ok_or_else(|| InvokeError::Message(format!("{key} must be an array of strings")))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::io::Write;
    use std::path::PathBuf;

    use super::*;

    fn workspace(name: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "scan-kit-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        path
    }

    fn write_session(root: &std::path::Path, name: &str, summary: &str, map: &str) {
        let dir = root.join(name);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("termination_summary.txt"), summary).unwrap();
        fs::write(dir.join("input_map.csv"), map).unwrap();
    }

    const KEPT_SUMMARY: &str = "\
Date: Thu Sep 10 21:07:41 2026
Primary total dose: 38.0223 MU
Treatment time: 251.604 seconds
Room number: 4
Configuration name: kept-config
Spot extent width: 118.75 mm
Spot extent height: 110.592 mm
Layer delivery: 27/27
";

    #[test]
    fn sk_req_004_open_library_reuses_fingerprint() {
        let root = workspace("library");
        let db = root.join("test.sqlite");
        write_session(
            &root,
            "kept",
            KEPT_SUMMARY,
            "energy,X_POSITION,Y_POSITION\n1,0,0\n2,1,1\n",
        );
        write_session(
            &root,
            "maponly",
            "Date: Wed Jan 15 12:00:00 2025\nConfiguration name: from-spots\n",
            "energy,X_POSITION,Y_POSITION\n70,0,0\n90,10,4\n",
        );
        discover::write_zip(
            &root.join("kept.zip"),
            &[(
                "kept/termination_summary.txt",
                b"Date: Mon Jan 01 00:00:00 2001\nConfiguration name: from-zip\n",
            )],
        )
        .unwrap();
        discover::write_zip(
            &root.join("archived.zip"),
            &[(
                "archived/termination_summary.txt",
                b"Date: Tue Jan 02 03:04:05 2024\nPrimary total dose: 1.0 MU\nConfiguration name: zipped\n",
            )],
        )
        .unwrap();

        let root_arg = root.to_string_lossy().into_owned();
        let db_arg = db.to_string_lossy().into_owned();
        let opened = invoke(
            "scan_kit_open_library",
            &json!({ "path": &root_arg, "db_path": &db_arg }),
        )
        .unwrap();
        let rows = opened["rows"].as_array().unwrap();
        assert!(rows.iter().any(|row| row["session_id"] == "kept"));
        assert!(rows.iter().any(|row| row["session_id"] == "archived"));
        assert!(!rows.iter().any(|row| row["session_id"] == "kept.zip"));
        let kept = rows.iter().find(|row| row["session_id"] == "kept").unwrap();
        assert_eq!(kept["date"], "09/10/26");
        assert_eq!(kept["extent"], "119");
        assert_eq!(kept["config"], "kept-config");
        assert!(!kept["storage_path"].as_str().unwrap().ends_with(".zip"));
        let maponly = rows
            .iter()
            .find(|row| row["session_id"] == "maponly")
            .unwrap();
        assert_eq!(maponly["extent"], "10");
        assert_eq!(maponly["layers"], "2");
        let archived = rows
            .iter()
            .find(|row| row["session_id"] == "archived")
            .unwrap();
        assert_eq!(archived["date"], "01/02/24");
        assert_eq!(archived["config"], "zipped");
        assert!(!root.join("archived").exists());
        assert!(!root.join("kept").join("kept").exists());

        let conn = rusqlite::Connection::open(&db).unwrap();
        conn.execute(
            "UPDATE sessions SET date = '1999-01-01T00:00:00' WHERE session_id = 'kept'",
            [],
        )
        .unwrap();
        drop(conn);

        let again = invoke(
            "scan_kit_open_library",
            &json!({ "path": &root_arg, "db_path": &db_arg }),
        )
        .unwrap();
        let kept = again["rows"]
            .as_array()
            .unwrap()
            .iter()
            .find(|row| row["session_id"] == "kept")
            .unwrap();
        assert_eq!(kept["date"], "01/01/99");

        let mut summary = fs::OpenOptions::new()
            .append(true)
            .open(root.join("kept").join("termination_summary.txt"))
            .unwrap();
        writeln!(summary).unwrap();
        drop(summary);

        let refreshed = invoke(
            "scan_kit_open_library",
            &json!({ "path": &root_arg, "db_path": &db_arg }),
        )
        .unwrap();
        let kept = refreshed["rows"]
            .as_array()
            .unwrap()
            .iter()
            .find(|row| row["session_id"] == "kept")
            .unwrap();
        assert_eq!(kept["date"], "09/10/26");
        assert_eq!(kept["extent"], "119");
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn sk_req_005_notes_and_selection_round_trip() {
        let root = workspace("notes");
        let db = root.join("test.sqlite");
        write_session(
            &root,
            "alpha",
            KEPT_SUMMARY,
            "energy,X_POSITION,Y_POSITION\n1,0,0\n",
        );
        fs::write(root.join("session_notes.json"), r#"{"alpha":"from json"}"#).unwrap();
        let root_arg = root.to_string_lossy().into_owned();
        let db_arg = db.to_string_lossy().into_owned();
        let opened = invoke(
            "scan_kit_open_library",
            &json!({ "path": &root_arg, "db_path": &db_arg }),
        )
        .unwrap();
        let alpha = opened["rows"]
            .as_array()
            .unwrap()
            .iter()
            .find(|row| row["session_id"] == "alpha")
            .unwrap();
        assert_eq!(alpha["note"], "from json");

        invoke(
            "scan_kit_set_note",
            &json!({ "path": &root_arg, "db_path": &db_arg, "session_id": "alpha", "note": "edited" }),
        )
        .unwrap();
        invoke(
            "scan_kit_select_sessions",
            &json!({ "path": &root_arg, "db_path": &db_arg, "session_ids": ["alpha"] }),
        )
        .unwrap();
        let again = invoke(
            "scan_kit_open_library",
            &json!({ "path": &root_arg, "db_path": &db_arg }),
        )
        .unwrap();
        let alpha = again["rows"]
            .as_array()
            .unwrap()
            .iter()
            .find(|row| row["session_id"] == "alpha")
            .unwrap();
        assert_eq!(alpha["note"], "edited");
        assert_eq!(alpha["selected"], true);
        assert_eq!(again["selected"], json!(["alpha"]));

        let conn = rusqlite::Connection::open(&db).unwrap();
        conn.execute("UPDATE sessions SET note = ''", []).unwrap();
        drop(conn);
        let restored = invoke(
            "scan_kit_open_library",
            &json!({ "path": &root_arg, "db_path": &db_arg }),
        )
        .unwrap();
        let alpha = restored["rows"]
            .as_array()
            .unwrap()
            .iter()
            .find(|row| row["session_id"] == "alpha")
            .unwrap();
        assert_eq!(alpha["note"], "from json");

        let too_many = invoke(
            "scan_kit_select_sessions",
            &json!({
                "path": &root_arg,
                "db_path": &db_arg,
                "session_ids": ["a", "b", "c", "d", "e", "f"]
            }),
        );
        assert!(too_many.is_err());

        let conn = rusqlite::Connection::open(&db).unwrap();
        assert_eq!(store::user_version(&conn).unwrap(), 3);
        drop(conn);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn sk_req_005_migrates_user_version_1() {
        let root = workspace("migrate");
        let db = root.join("old.sqlite");
        let conn = rusqlite::Connection::open(&db).unwrap();
        conn.execute_batch(
            "
            CREATE TABLE prefs (key TEXT PRIMARY KEY, value TEXT NOT NULL);
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
            PRAGMA user_version = 1;
            ",
        )
        .unwrap();
        drop(conn);

        let conn = store::open_connection(&db).unwrap();
        assert_eq!(store::user_version(&conn).unwrap(), 3);
        let extent: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM pragma_table_info('sessions') WHERE name = 'map_extent_mm'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        let mode: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM pragma_table_info('libraries') WHERE name = 'calibration_mode'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(extent, 1);
        assert_eq!(mode, 1);
        drop(conn);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn sk_req_006_quoted_csv_projects_two_columns_and_scales_g2() {
        let root = workspace("columns");
        let session = root.join("sess");
        fs::create_dir_all(&session).unwrap();
        fs::write(
            session.join("timeslice_data_device_units.csv"),
            "Energy,r_ic1_current_dose,note\n\"70.5\",\"1.5e-9\",\"a,b\"\n\"80.0\",\"2.5e-9\",\"c\"\n",
        )
        .unwrap();
        let loaded = invoke(
            "scan_kit_load_columns",
            &json!({
                "storage_path": session.to_string_lossy(),
                "session_id": "sess",
                "columns": ["energy", "ic1_current"]
            }),
        )
        .unwrap();
        assert_eq!(loaded["rows"], 2);
        assert!(loaded["missing"].as_array().unwrap().is_empty());
        let columns = loaded["columns"].as_array().unwrap();
        assert_eq!(columns[0]["name"], "energy");
        assert_eq!(columns[0]["dtype"], "f32");
        assert!((columns[0]["values"][0].as_f64().unwrap() - 70.5).abs() < 1e-3);
        assert!((columns[0]["values"][1].as_f64().unwrap() - 80.0).abs() < 1e-3);
        let first = columns[1]["values"][0].as_f64().unwrap();
        let second = columns[1]["values"][1].as_f64().unwrap();
        assert!((first - 1500.0).abs() < 1e-2, "scaled current {first}");
        assert!((second - 2500.0).abs() < 1e-2, "scaled current {second}");
        let _ = fs::remove_dir_all(&root);
    }
}
