//! Session discovery, the sqlite store, and columnar loads.
//!
//! The desktop shell and the MCP server call these functions. They do not
//! reimplement them.

mod analysis;
mod beam;
mod bins;
mod columns;
mod config;
mod discover;
mod histogram;
mod location;
mod marks;
mod mc_tables;
mod patient_view;
mod phantom;
mod remote;
mod runner;
mod source;
mod store;
mod synthesis;
mod tables;
pub mod volumetric;

pub use analysis::{analysis_scene, channel_catalog, load_timeslice_columns};
pub use beam::{beam_record, protons_per_mu, spot_record};
pub use mc_tables::mc_tables;
pub use tables::{
    bind_slice, clear_slice, session_pieces, slice_tail_done, SessionPieces, SliceTake, SLICE_ROWS,
};
pub use volumetric::volumetric;

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use scan_kit_core::{
    merge_session_geom, parse_termination_summary_text, SessionMeta, ToolKind, ToolSpec,
};
use serde_json::{json, Value};

use columns::map_geom_from_spots;
use discover::Discovered;
use store::{
    backfill_blank_notes, data_dirs, default_db_path, delete_missing, delete_missing_exams,
    exam_rows, forget_data_dir, last_main_tab, library_id, library_rows, prepare_library,
    record_meta, remember_data_dir, set_last_main_tab, set_note, set_selected, sync_entry,
    upsert_exam, window_geometry, with_store, Cache,
};

const TOOLS: &[ToolSpec] = &[
    ToolSpec {
        name: "scan_kit_open_library",
        summary: "Discover a data folder or URL, sync the session index, and return the rows.",
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
        name: "scan_kit_data_dirs",
        summary: "List saved data locations in order.",
        kind: ToolKind::Granular,
    },
    ToolSpec {
        name: "scan_kit_read_catalog",
        summary: "Return the combined session and exam catalog.",
        kind: ToolKind::Granular,
    },
    ToolSpec {
        name: "scan_kit_forget_data_dir",
        summary: "Remove one saved data location from the catalog.",
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
        summary: "List timeslice channels present for Replay.",
        kind: ToolKind::Granular,
    },
    ToolSpec {
        name: "scan_kit_analysis_scene",
        summary: "Build one analysis view scene from the selected sessions.",
        kind: ToolKind::Workflow,
    },
    ToolSpec {
        name: "scan_kit_plan_catalog",
        summary: "List plan templates and the parameter specs for each.",
        kind: ToolKind::Granular,
    },
    ToolSpec {
        name: "scan_kit_synthesize_plan",
        summary: "Build an input map CSV from a plan template.",
        kind: ToolKind::Workflow,
    },
    ToolSpec {
        name: "scan_kit_config_catalog",
        summary: "List configuration tuning workflows and the remembered config folder.",
        kind: ToolKind::Granular,
    },
    ToolSpec {
        name: "scan_kit_config_open",
        summary: "List XML files in a configuration folder.",
        kind: ToolKind::Granular,
    },
    ToolSpec {
        name: "scan_kit_config_form",
        summary: "Turn one XML file into an editable form.",
        kind: ToolKind::Granular,
    },
    ToolSpec {
        name: "scan_kit_config_apply",
        summary: "Write form edits back into XML text.",
        kind: ToolKind::Granular,
    },
    ToolSpec {
        name: "scan_kit_config_save",
        summary: "Save a configuration folder and refresh .md5 sidecars.",
        kind: ToolKind::Workflow,
    },
    ToolSpec {
        name: "scan_kit_config_tune",
        summary: "Apply one devices.xml tuner to the selected sessions.",
        kind: ToolKind::Workflow,
    },
    ToolSpec {
        name: "scan_kit_config_integrity",
        summary: "Check one XML file against its .md5 sidecar.",
        kind: ToolKind::Granular,
    },
    ToolSpec {
        name: "scan_kit_config_hide",
        summary: "Remember whether unused map2map fields are hidden.",
        kind: ToolKind::Granular,
    },
    ToolSpec {
        name: "scan_kit_phantom_catalog",
        summary: "List phantom presets, positions, and the remembered DICOM folder.",
        kind: ToolKind::Granular,
    },
    ToolSpec {
        name: "scan_kit_phantom_preview",
        summary: "Describe the CT size, layers, spots, and MU of a phantom study.",
        kind: ToolKind::Granular,
    },
    ToolSpec {
        name: "scan_kit_write_phantom",
        summary: "Write a synthetic CT, RTSTRUCT, and RT Ion plan into a new folder.",
        kind: ToolKind::Workflow,
    },
    ToolSpec {
        name: "scan_kit_runner_catalog",
        summary: "Return the remembered RCI host and an idle Plan Runner view.",
        kind: ToolKind::Granular,
    },
    ToolSpec {
        name: "scan_kit_runner_connect",
        summary: "Open an mpack session to an RCI and read its status.",
        kind: ToolKind::Workflow,
    },
    ToolSpec {
        name: "scan_kit_runner_disconnect",
        summary: "Close the Plan Runner session.",
        kind: ToolKind::Granular,
    },
    ToolSpec {
        name: "scan_kit_runner_status",
        summary: "Poll the connected RCI and return the operator view.",
        kind: ToolKind::Granular,
    },
    ToolSpec {
        name: "scan_kit_runner_upload",
        summary: "Upload an input_map.csv to the connected RCI.",
        kind: ToolKind::Workflow,
    },
    ToolSpec {
        name: "scan_kit_runner_control",
        summary: "Press Start, Pause, Stop, or Reset on the connected RCI.",
        kind: ToolKind::Workflow,
    },
    ToolSpec {
        name: "scan_kit_runner_download",
        summary: "Download the latest session folder from the RCI into a zip.",
        kind: ToolKind::Workflow,
    },
    ToolSpec {
        name: "scan_kit_runner_remember",
        summary: "Remember the folder used to browse plan and session files.",
        kind: ToolKind::Granular,
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
        "scan_kit_data_dirs" | "scan_kit_read_catalog" => json!({
            "type": "object",
            "properties": {
                "db_path": { "type": "string" }
            },
            "additionalProperties": false
        }),
        "scan_kit_forget_data_dir" => json!({
            "type": "object",
            "properties": {
                "path": { "type": "string" },
                "db_path": { "type": "string" }
            },
            "required": ["path"],
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
        "scan_kit_plan_catalog" => json!({
            "type": "object",
            "properties": { "db_path": { "type": "string" } },
            "additionalProperties": false
        }),
        "scan_kit_synthesize_plan" => json!({
            "type": "object",
            "properties": {
                "template": { "type": "string" },
                "params": { "type": "object" },
                "path": { "type": "string" },
                "csv": { "type": "string" },
                "db_path": { "type": "string" }
            },
            "required": ["template", "params"],
            "additionalProperties": false
        }),
        "scan_kit_config_catalog" | "scan_kit_config_open" => json!({
            "type": "object",
            "properties": {
                "path": { "type": "string" },
                "db_path": { "type": "string" }
            },
            "additionalProperties": false
        }),
        "scan_kit_config_form" | "scan_kit_config_integrity" => json!({
            "type": "object",
            "properties": { "path": { "type": "string" } },
            "required": ["path"],
            "additionalProperties": false
        }),
        "scan_kit_config_apply" => json!({
            "type": "object",
            "properties": {
                "xml": { "type": "string" },
                "form": { "type": "object" }
            },
            "required": ["xml", "form"],
            "additionalProperties": false
        }),
        "scan_kit_config_save" => json!({
            "type": "object",
            "properties": {
                "source": { "type": "string" },
                "dest": { "type": "string" },
                "files": { "type": "array" },
                "hide_unused": { "type": "boolean" },
                "db_path": { "type": "string" }
            },
            "required": ["source", "dest", "files"],
            "additionalProperties": false
        }),
        "scan_kit_config_tune" => json!({
            "type": "object",
            "properties": {
                "workflow": { "type": "string" },
                "xml": { "type": "string" },
                "form": { "type": "object" },
                "data_dir": { "type": "string" },
                "session_ids": { "type": "array", "items": { "type": "string" } },
                "params": { "type": "object" },
                "db_path": { "type": "string" }
            },
            "required": ["workflow", "xml", "data_dir", "session_ids"],
            "additionalProperties": false
        }),
        "scan_kit_config_hide" => json!({
            "type": "object",
            "properties": {
                "hide_unused": { "type": "boolean" },
                "db_path": { "type": "string" }
            },
            "required": ["hide_unused"],
            "additionalProperties": false
        }),
        "scan_kit_phantom_catalog" => json!({
            "type": "object",
            "properties": { "db_path": { "type": "string" } },
            "additionalProperties": false
        }),
        "scan_kit_phantom_preview" => json!({
            "type": "object",
            "properties": { "params": { "type": "object" } },
            "required": ["params"],
            "additionalProperties": false
        }),
        "scan_kit_write_phantom" => json!({
            "type": "object",
            "properties": {
                "parent": { "type": "string" },
                "params": { "type": "object" },
                "db_path": { "type": "string" }
            },
            "required": ["parent", "params"],
            "additionalProperties": false
        }),
        "scan_kit_runner_catalog" | "scan_kit_runner_disconnect" => json!({
            "type": "object",
            "properties": { "db_path": { "type": "string" } },
            "additionalProperties": false
        }),
        "scan_kit_runner_connect" => json!({
            "type": "object",
            "properties": {
                "host": { "type": "string" },
                "db_path": { "type": "string" }
            },
            "required": ["host"],
            "additionalProperties": false
        }),
        "scan_kit_runner_status" => json!({
            "type": "object",
            "properties": {
                "has_plan": { "type": "boolean" },
                "dest": { "type": "string" },
                "db_path": { "type": "string" }
            },
            "additionalProperties": false
        }),
        "scan_kit_runner_upload" => json!({
            "type": "object",
            "properties": {
                "path": { "type": "string" },
                "db_path": { "type": "string" }
            },
            "required": ["path"],
            "additionalProperties": false
        }),
        "scan_kit_runner_control" => json!({
            "type": "object",
            "properties": {
                "action": { "type": "string" },
                "db_path": { "type": "string" }
            },
            "required": ["action"],
            "additionalProperties": false
        }),
        "scan_kit_runner_download" => json!({
            "type": "object",
            "properties": {
                "dest": { "type": "string" },
                "db_path": { "type": "string" }
            },
            "required": ["dest"],
            "additionalProperties": false
        }),
        "scan_kit_runner_remember" => json!({
            "type": "object",
            "properties": {
                "file_dir": { "type": "string" },
                "db_path": { "type": "string" }
            },
            "required": ["file_dir"],
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
        "scan_kit_data_dirs" => {
            let db = database_path(input)?;
            with_store(&db, |conn| data_dirs(conn))
                .map(|dirs| json!(dirs))
                .map_err(InvokeError::Message)
        }
        "scan_kit_read_catalog" => {
            let db = database_path(input)?;
            with_store(&db, |conn| publish_catalog(conn)).map_err(InvokeError::Message)
        }
        "scan_kit_forget_data_dir" => {
            let path = required_str(input, "path")?;
            let db = database_path(input)?;
            with_store(&db, |conn| {
                let root = library_root(path)?;
                forget_data_dir(conn, &root)?;
                publish_catalog(conn)
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
        "scan_kit_plan_catalog" => {
            let db = database_path(input)?;
            Ok(synthesis::catalog(&db))
        }
        "scan_kit_synthesize_plan" => {
            let db = database_path(input)?;
            synthesis::synthesize(input, &db).map_err(InvokeError::Message)
        }
        "scan_kit_config_catalog" => {
            let db = database_path(input)?;
            Ok(config::catalog(&db))
        }
        "scan_kit_config_open" => {
            let db = database_path(input)?;
            let path = input.get("path").and_then(Value::as_str);
            let data_dir = input.get("data_dir").and_then(Value::as_str);
            let session_id = input.get("session_id").and_then(Value::as_str);
            config::open_folder(&db, path, data_dir, session_id).map_err(InvokeError::Message)
        }
        "scan_kit_config_form" => {
            let path = required_str(input, "path")?;
            config::read_form(Path::new(path)).map_err(InvokeError::Message)
        }
        "scan_kit_config_apply" => {
            let xml = required_str(input, "xml")?;
            let form = input.get("form").cloned().unwrap_or_else(|| json!({}));
            config::apply(xml, &form).map_err(InvokeError::Message)
        }
        "scan_kit_config_save" => {
            let db = database_path(input)?;
            let source = required_str(input, "source")?;
            let dest = required_str(input, "dest")?;
            let files = input.get("files").cloned().unwrap_or_else(|| json!([]));
            let hide = input.get("hide_unused").and_then(Value::as_bool);
            config::save_folder(&db, Path::new(source), Path::new(dest), &files, hide)
                .map_err(InvokeError::Message)
        }
        "scan_kit_config_tune" => {
            let db = database_path(input)?;
            config::tune(input, &db).map_err(InvokeError::Message)
        }
        "scan_kit_config_integrity" => {
            let path = required_str(input, "path")?;
            Ok(config::integrity(Path::new(path)))
        }
        "scan_kit_config_hide" => {
            let db = database_path(input)?;
            let hide = input
                .get("hide_unused")
                .and_then(Value::as_bool)
                .ok_or_else(|| InvokeError::Message("hide_unused is required".into()))?;
            config::set_hide_unused(&db, hide).map_err(InvokeError::Message)
        }
        "scan_kit_phantom_catalog" => {
            let db = database_path(input)?;
            Ok(phantom::catalog(&db))
        }
        "scan_kit_phantom_preview" => {
            let params = input.get("params").cloned().unwrap_or_else(|| json!({}));
            phantom::preview(&params).map_err(InvokeError::Message)
        }
        "scan_kit_write_phantom" => {
            let db = database_path(input)?;
            let parent = required_str(input, "parent")?;
            let params = input.get("params").cloned().unwrap_or_else(|| json!({}));
            phantom::write(Path::new(parent), &params, &db).map_err(InvokeError::Message)
        }
        "scan_kit_runner_catalog" => {
            let db = database_path(input)?;
            Ok(runner::catalog(&db))
        }
        "scan_kit_runner_connect" => {
            let db = database_path(input)?;
            let host = required_str(input, "host")?;
            runner::connect(&db, host).map_err(InvokeError::Message)
        }
        "scan_kit_runner_disconnect" => {
            let db = database_path(input)?;
            runner::disconnect();
            Ok(runner::catalog(&db))
        }
        "scan_kit_runner_status" => {
            let has_plan = input
                .get("has_plan")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let dest = input.get("dest").and_then(Value::as_str).unwrap_or("");
            runner::read_status(has_plan, dest).map_err(InvokeError::Message)
        }
        "scan_kit_runner_upload" => {
            let db = database_path(input)?;
            let path = required_str(input, "path")?;
            runner::upload(&db, Path::new(path)).map_err(InvokeError::Message)
        }
        "scan_kit_runner_control" => {
            let action = required_str(input, "action")?;
            runner::control(action).map_err(InvokeError::Message)
        }
        "scan_kit_runner_download" => {
            let db = database_path(input)?;
            let dest = required_str(input, "dest")?;
            runner::download(&db, dest).map_err(InvokeError::Message)
        }
        "scan_kit_runner_remember" => {
            let db = database_path(input)?;
            let dir = required_str(input, "file_dir")?;
            runner::remember_dir(&db, dir).map_err(InvokeError::Message)
        }
        _ => Err(InvokeError::UnknownTool {
            name: name.to_owned(),
        }),
    }
}

/// Copy the connected session into `dest`, calling `keep` after each file.
pub fn copy_runner(dest: &str, keep: &mut dyn FnMut(u64, u64) -> bool) -> Result<Value, String> {
    runner::download_with(&default_db_path(), dest, keep)
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
    open_library_keep(conn, data_dir, &mut |_, _, _| true)
}

/// Same index as [`open_library`], publishing `done`, `total`, and the rows so far.
///
/// `keep` returns false to roll the index back. The final value is still the committed rows.
pub fn index_library(
    data_dir: &Path,
    keep: &mut dyn FnMut(u64, u64, Option<&Value>) -> bool,
) -> Result<Value, String> {
    with_store(&default_db_path(), |conn| {
        open_library_keep(conn, data_dir, keep)
    })
}

fn open_library_keep(
    conn: &mut rusqlite::Connection,
    data_dir: &Path,
    keep: &mut dyn FnMut(u64, u64, Option<&Value>) -> bool,
) -> Result<Value, String> {
    let spec = data_dir.to_string_lossy();
    let (root, opened) = location::open_root(&spec)?;
    let found = match &opened {
        location::Opened::Local(path) => discover::discover(path)?,
        location::Opened::Remote(url) => location::discover_remote(url)?,
    };
    let places = found
        .entries
        .iter()
        .map(|entry| {
            (
                entry.session_id.clone(),
                entry.storage_path.to_string_lossy().into_owned(),
            )
        })
        .collect::<Vec<_>>();
    let exams = found.exams;
    let skipped = found.skipped;
    let tx = conn.transaction().map_err(|err| err.to_string())?;
    let lib_id = prepare_library(&tx, &root)?;
    let ids = places
        .iter()
        .map(|(session_id, _)| session_id.clone())
        .collect::<HashSet<_>>();
    delete_missing(&tx, lib_id, &ids)?;
    let total = found.entries.len() as u64;
    if !keep(0, total, None) {
        return Err("cancelled".into());
    }
    for (index, entry) in found.entries.into_iter().enumerate() {
        if let Cache::Hydrate = sync_entry(&tx, lib_id, &entry)? {
            let meta = hydrate(&entry);
            record_meta(&tx, lib_id, &entry, meta.as_ref())?;
        }
        let done = (index as u64) + 1;
        let preview = json!({
            "root": root,
            "rows": library_rows(&tx, lib_id)?,
            "selected": [],
        });
        if !keep(done, total, Some(&preview)) {
            return Err("cancelled".into());
        }
    }
    let exam_ids = exams
        .iter()
        .map(|exam| exam.study_uid.clone())
        .collect::<HashSet<_>>();
    delete_missing_exams(&tx, lib_id, &exam_ids)?;
    for exam in &exams {
        upsert_exam(&tx, lib_id, exam)?;
    }
    remember_data_dir(&tx, &root)?;
    backfill_blank_notes(&tx, lib_id, Path::new(&root))?;
    let rows = library_rows(&tx, lib_id)?;
    let selected = store::selected_session_ids(&tx, lib_id)?;
    tx.commit().map_err(|err| err.to_string())?;
    location::replace_sessions(&root, &places);
    publish_catalog(conn)?;
    Ok(json!({ "root": root, "rows": rows, "selected": selected, "skipped": skipped }))
}

fn publish_catalog(conn: &rusqlite::Connection) -> Result<Value, String> {
    let locations = data_dirs(conn)?;
    let mut used_sessions = HashSet::new();
    let mut used_exams = HashSet::new();
    let mut rows = Vec::new();
    let mut places = Vec::new();
    let mut selected = Vec::new();
    let mut exams = Vec::new();
    for root in &locations {
        let Some(lib_id) = library_id(conn, root)? else {
            continue;
        };
        let mut batch = Vec::new();
        for mut row in library_rows(conn, lib_id)? {
            let folder_id = row["session_id"].as_str().unwrap_or("").to_owned();
            let place = row["storage_path"].as_str().unwrap_or("").to_owned();
            let key = catalog_key(&folder_id, &place, &mut used_sessions);
            row["folder_id"] = json!(&folder_id);
            row["library"] = json!(root);
            row["session_id"] = json!(&key);
            places.push((key.clone(), root.clone(), place, folder_id.clone()));
            batch.push((folder_id, key, row));
        }
        for id in store::selected_session_ids(conn, lib_id)? {
            if let Some((_, key, _)) = batch.iter().find(|(folder, _, _)| folder == &id) {
                selected.push(key.clone());
            }
        }
        for (_, _, row) in batch {
            rows.push(row);
        }
        for exam in exam_rows(conn, lib_id)? {
            let key = catalog_key(&exam.folder_name, &exam.storage_path, &mut used_exams);
            exams.push(json!({
                "exam": key,
                "patient": exam.patient_name,
                "patient_id": exam.patient_id,
                "date": short_study_date(&exam.study_date),
                "description": exam.study_description,
                "files": exam.file_count,
                "library": root,
            }));
        }
    }
    location::replace_catalog(&places);
    Ok(json!({
        "locations": locations,
        "rows": rows,
        "exams": exams,
        "selected": selected,
    }))
}

fn catalog_key(name: &str, storage: &str, used: &mut HashSet<String>) -> String {
    if used.insert(name.to_owned()) {
        return name.to_owned();
    }
    let parent = parent_folder(storage);
    let mut key = format!("{name} · {parent}");
    let mut extra = 2u32;
    while !used.insert(key.clone()) {
        key = format!("{name} · {parent} ({extra})");
        extra += 1;
    }
    key
}

fn parent_folder(storage: &str) -> String {
    let trimmed = storage.trim_end_matches(['/', '\\']);
    trimmed
        .rsplit(['/', '\\'])
        .nth(1)
        .filter(|name| !name.is_empty())
        .unwrap_or("location")
        .to_owned()
}

fn short_study_date(raw: &str) -> String {
    let bytes = raw.as_bytes();
    if bytes.len() >= 8 && bytes[..8].iter().all(u8::is_ascii_digit) {
        format!("{}-{}-{}", &raw[..4], &raw[4..6], &raw[6..8])
    } else {
        raw.to_owned()
    }
}

/// Remember a password for a remote library. It stays in process memory.
pub fn remember_password(spec: &str, password: &str) {
    location::remember_password(spec, password);
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
    if location::is_remote_location(path) {
        return location::canonical_root(path);
    }
    let path = Path::new(path);
    if !path.is_dir() {
        return Err(format!("{} is not a directory", path.display()));
    }
    location::canonical_root(&path.to_string_lossy())
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
        assert_eq!(store::user_version(&conn).unwrap(), 4);
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
        assert_eq!(store::user_version(&conn).unwrap(), 4);
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
        let exams: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'exams'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(exams, 1);
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

    #[test]
    fn a_nested_session_is_found_and_a_duplicate_is_named() {
        let root = workspace("nested");
        let db = root.join("test.sqlite");
        let year = root.join("site").join("year");
        let kept = year.join("kept");
        fs::create_dir_all(&kept).unwrap();
        fs::write(kept.join("termination_summary.txt"), KEPT_SUMMARY).unwrap();
        fs::write(
            kept.join("input_map.csv"),
            "energy,X_POSITION,Y_POSITION\n1,0,0\n",
        )
        .unwrap();
        let notes = year.join("notes");
        fs::create_dir_all(&notes).unwrap();
        fs::write(notes.join("readme.txt"), b"not a session").unwrap();
        discover::write_zip(
            &year.join("bundle.zip"),
            &[
                ("input_map.csv", b"energy,X_POSITION,Y_POSITION\n1,0,0\n"),
                ("termination_summary.txt", KEPT_SUMMARY.as_bytes()),
            ],
        )
        .unwrap();
        let duplicate = root.join("site").join("z-dup").join("kept");
        fs::create_dir_all(&duplicate).unwrap();
        fs::write(duplicate.join("input_map.csv"), b"energy\n1\n").unwrap();
        fs::write(
            duplicate.join("termination_summary.txt"),
            b"Configuration name: dup\n",
        )
        .unwrap();

        let before = discover::session_directory(&root, "kept");
        assert!(!before.join("input_map.csv").is_file());

        let opened = invoke(
            "scan_kit_open_library",
            &json!({ "path": root.to_string_lossy(), "db_path": db.to_string_lossy() }),
        )
        .unwrap();
        let ids = opened["rows"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| row["session_id"].as_str().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(ids, vec!["bundle", "kept"]);
        let skipped = opened["skipped"].as_array().unwrap();
        assert!(
            skipped
                .iter()
                .any(|item| item.as_str().unwrap().contains("z-dup")),
            "{skipped:?}"
        );

        let found =
            discover::session_directory(Path::new(opened["root"].as_str().unwrap()), "kept");
        assert!(found.join("input_map.csv").is_file());
        assert!(found
            .to_string_lossy()
            .replace('\\', "/")
            .contains("site/year/kept"));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_remote_listing_is_cached_when_the_view_opens() {
        let root = workspace("remote");
        let db = root.join("test.sqlite");
        let url = format!(
            "sftp://scan-kit@example.invalid/data/{}/{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let canonical = location::install_fixture(
            &url,
            &[
                (
                    "site/year/sess/input_map.csv",
                    b"energy,X_POSITION,Y_POSITION\n1,0,0\n",
                ),
                (
                    "site/year/sess/termination_summary.txt",
                    KEPT_SUMMARY.as_bytes(),
                ),
                ("site/year/notes/readme.txt", b"not a session"),
            ],
            false,
        )
        .unwrap();
        let opened = invoke(
            "scan_kit_open_library",
            &json!({ "path": &url, "db_path": db.to_string_lossy() }),
        )
        .unwrap();
        assert_eq!(opened["root"], canonical);
        let row = opened["rows"]
            .as_array()
            .unwrap()
            .iter()
            .find(|row| row["session_id"] == "sess")
            .unwrap();
        assert_eq!(row["config"], "kept-config");
        assert_eq!(location::fixture_copies(&canonical), 0);

        let dir = discover::session_directory(Path::new(&canonical), "sess");
        let text = fs::read_to_string(dir.join("input_map.csv")).unwrap();
        assert!(text.contains("X_POSITION"));
        assert_eq!(location::fixture_copies(&canonical), 1);
        let _ = discover::session_directory(Path::new(&canonical), "sess");
        assert_eq!(location::fixture_copies(&canonical), 1);
        let _ = fs::remove_dir_all(location::cache_directory(&canonical));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_remote_open_asks_again_when_the_password_is_missing() {
        let root = workspace("remote-auth");
        let db = root.join("test.sqlite");
        let url = format!(
            "sftp://scan-kit@example.invalid/locked/{}/{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        location::install_fixture(
            &url,
            &[
                ("sess/input_map.csv", b"energy\n1\n"),
                (
                    "sess/termination_summary.txt",
                    b"Configuration name: locked\n",
                ),
            ],
            true,
        )
        .unwrap();
        let err = invoke(
            "scan_kit_open_library",
            &json!({ "path": &url, "db_path": db.to_string_lossy() }),
        )
        .unwrap_err();
        assert!(
            err.to_string().starts_with("authentication required"),
            "{err}"
        );
        remember_password(&url, "secret");
        let opened = invoke(
            "scan_kit_open_library",
            &json!({ "path": &url, "db_path": db.to_string_lossy() }),
        )
        .unwrap();
        assert!(!opened["root"].as_str().unwrap().contains("secret"));
        assert!(opened["rows"]
            .as_array()
            .unwrap()
            .iter()
            .any(|row| row["session_id"] == "sess"));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn saved_locations_share_one_session_list_and_exams_stay_separate() {
        let root = workspace("catalog");
        let db = root.join("test.sqlite");
        let db_arg = db.to_string_lossy().into_owned();
        let site_a = root.join("site-a");
        let site_b = root.join("site-b");
        write_session(
            &site_a,
            "kept",
            KEPT_SUMMARY,
            "energy,X_POSITION,Y_POSITION\n1,0,0\n",
        );
        write_session(
            &site_a,
            "other",
            KEPT_SUMMARY,
            "energy,X_POSITION,Y_POSITION\n1,0,0\n",
        );
        write_session(
            &site_b,
            "kept",
            KEPT_SUMMARY,
            "energy,X_POSITION,Y_POSITION\n1,1,1\n",
        );
        let exam = site_a.join("study");
        fs::create_dir_all(&exam).unwrap();
        let mut stub = vec![0u8; 132];
        stub[128..132].copy_from_slice(b"DICM");
        fs::write(exam.join("1.2.3"), &stub).unwrap();
        let notes = site_a.join("notes").join("deep");
        fs::create_dir_all(&notes).unwrap();
        fs::write(notes.join("readme.txt"), b"not dicom").unwrap();

        invoke(
            "scan_kit_open_library",
            &json!({ "path": site_a.to_string_lossy(), "db_path": &db_arg }),
        )
        .unwrap();
        invoke(
            "scan_kit_open_library",
            &json!({ "path": site_b.to_string_lossy(), "db_path": &db_arg }),
        )
        .unwrap();
        let catalog = invoke("scan_kit_read_catalog", &json!({ "db_path": &db_arg })).unwrap();
        let locations = catalog["locations"].as_array().unwrap();
        assert_eq!(locations.len(), 2);
        let ids: Vec<&str> = catalog["rows"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| row["session_id"].as_str().unwrap())
            .collect();
        assert!(ids.contains(&"kept"), "{ids:?}");
        assert!(ids.contains(&"other"), "{ids:?}");
        assert!(ids.contains(&"kept · site-b"), "{ids:?}");
        let exams = catalog["exams"].as_array().unwrap();
        assert_eq!(exams.len(), 1);
        assert_eq!(exams[0]["exam"], "study");
        assert_eq!(exams[0]["files"], 1);
        let found =
            discover::session_directory(Path::new(locations[0].as_str().unwrap()), "kept · site-b");
        let found_text = found.to_string_lossy().replace('\\', "/");
        assert!(found_text.contains("site-b/kept"), "{found_text}");
        assert!(found.join("input_map.csv").is_file());

        invoke(
            "scan_kit_select_sessions",
            &json!({
                "path": locations[1],
                "db_path": &db_arg,
                "session_ids": ["kept"]
            }),
        )
        .unwrap();
        let selected = invoke("scan_kit_read_catalog", &json!({ "db_path": &db_arg })).unwrap();
        assert!(selected["selected"]
            .as_array()
            .unwrap()
            .iter()
            .any(|id| id == "kept · site-b"));

        let forgotten = invoke(
            "scan_kit_forget_data_dir",
            &json!({ "path": locations[0], "db_path": &db_arg }),
        )
        .unwrap();
        let left: Vec<&str> = forgotten["rows"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| row["session_id"].as_str().unwrap())
            .collect();
        assert_eq!(left, vec!["kept"]);
        assert!(forgotten["exams"].as_array().unwrap().is_empty());
        let conn = rusqlite::Connection::open(&db).unwrap();
        let still: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sessions s JOIN libraries l ON l.id = s.library_id WHERE l.root_path = ?1",
                [locations[0].as_str().unwrap()],
                |row| row.get(0),
            )
            .unwrap();
        assert!(
            still >= 1,
            "the library row stays after the location is removed"
        );
        assert_eq!(store::user_version(&conn).unwrap(), 4);
        drop(conn);
        let _ = fs::remove_dir_all(&root);
    }
}
