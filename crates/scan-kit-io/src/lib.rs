//! Session discovery, the sqlite store, and columnar loads.
//!
//! The desktop shell and the MCP server call these functions. They do not
//! reimplement them.

mod analysis;
mod beam;
mod bins;
pub mod calibration;
mod columns;
mod config;
mod device_scan;
mod discover;
mod histogram;
mod location;
mod marks;
mod mc_tables;
pub mod npz;
pub mod patient_view;
mod phantom;
mod remote;
pub mod report;
mod runner;
mod session_capture;
mod source;
mod store;
mod synthesis;
mod tables;
pub mod volumetric;
mod workspace;

pub use analysis::{analysis_scene, channel_catalog, load_timeslice_columns};
pub use beam::{beam_record, protons_per_mu, spot_record, BeamModel};
pub use discover::session_directory;
pub use location::{file_uri_path, is_remote_location};
pub use marks::normalize_options;
pub use mc_tables::{csda_range_mm, mc_tables, medium_rho};
pub use store::default_db_path;
pub use store::strip_verbatim;
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
    backfill_blank_notes, data_dirs, delete_missing, delete_missing_exams, ensure_library,
    exam_rows, forget_data_dir, import_legacy, last_main_tab, library_id, library_rows,
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
        name: "scan_kit_config_diff",
        summary: "Diff the loaded config tree against each selected session's config/.",
        kind: ToolKind::Granular,
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
    ToolSpec {
        name: "scan_kit_discover_subnet",
        summary: "Scan a CIDR subnet for reachable IGX devices.",
        kind: ToolKind::Granular,
    },
    ToolSpec {
        name: "scan_kit_scanners",
        summary: "List, register, or remove MCsquare CT-calibration scanner folders.",
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
        "scan_kit_config_catalog" => json!({
            "type": "object",
            "properties": {
                "db_path": { "type": "string" }
            },
            "additionalProperties": false
        }),
        "scan_kit_config_open" => json!({
            "type": "object",
            "properties": {
                "path": { "type": "string" },
                "db_path": { "type": "string" },
                "data_dir": { "type": "string" },
                "session_id": { "type": "string" }
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
                "params": { "type": "object" }
            },
            "required": ["workflow", "xml", "data_dir", "session_ids"],
            "additionalProperties": false
        }),
        "scan_kit_config_diff" => json!({
            "type": "object",
            "properties": {
                "config_dir": { "type": "string" },
                "file": { "type": "string" },
                "xml": { "type": "string" },
                "form": { "type": "object" },
                "data_dir": { "type": "string" },
                "session_ids": { "type": "array", "items": { "type": "string" } }
            },
            "required": ["config_dir", "data_dir", "session_ids"],
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
                "dest": { "type": "string" }
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
                "action": { "type": "string" }
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
        "scan_kit_discover_subnet" => json!({
            "type": "object",
            "properties": {
                "subnet": { "type": "string" }
            },
            "required": ["subnet"],
            "additionalProperties": false
        }),
        "scan_kit_scanners" => json!({
            "type": "object",
            "properties": {
                "db_path": { "type": "string" },
                "action": { "type": "string", "enum": ["list", "add", "remove"] },
                "folder": { "type": "string" }
            },
            "required": ["action"],
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
            open_library(&db, Path::new(path)).map_err(InvokeError::Message)
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
        "scan_kit_session_storage_paths" => {
            let path = required_str(input, "path")?;
            let session_id = required_str(input, "session_id")?;
            let root = library_root(path).map_err(InvokeError::Message)?;
            let paths =
                location::session_storage_paths(&root, session_id).map_err(InvokeError::Message)?;
            Ok(json!({ "paths": paths }))
        }
        "scan_kit_trash_session" => {
            let path = required_str(input, "path")?;
            let session_id = required_str(input, "session_id")?;
            let db = database_path(input)?;
            // Remote storage IO runs outside STORE_LOCK — the lock covers
            // only the row removal when the delete succeeds.
            let root = library_root(path).map_err(InvokeError::Message)?;
            let paths =
                location::session_storage_paths(&root, session_id).map_err(InvokeError::Message)?;
            let errors = location::delete_storage(&root, &paths).map_err(InvokeError::Message)?;
            // A session whose files are still on disk keeps its row —
            // note/selection/hidden survive the failed delete. Nothing
            // found at all is not a successful delete either.
            if errors.is_empty() && !paths.is_empty() {
                with_store(&db, |conn| {
                    let lib_id = store::prepare_library(conn, &root)?;
                    store::delete_session(conn, lib_id, session_id)
                })
                .map_err(InvokeError::Message)?;
            }
            Ok(json!({ "removed": paths.len() - errors.len(), "errors": errors }))
                .map_err(InvokeError::Message)
        }
        "scan_kit_select_sessions" => {
            let path = required_str(input, "path")?;
            let session_ids = required_strings(input, "session_ids")?;
            let db = database_path(input)?;
            with_store(&db, |conn| {
                let root = library_root(path)?;
                let stored = set_selected(conn, &root, &session_ids)?;
                Ok(json!({ "session_ids": stored }))
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
                // canonical_root, not library_root: forgetting a dead path is the point.
                let root = location::canonical_root(path)?;
                forget_data_dir(conn, &root)?;
                location::drop_sessions(&root);
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
            let storage = Path::new(path);
            // Match load_columns - a bad path or session id is an error,
            // not a successful empty answer.
            if !storage.exists() {
                return Err(InvokeError::Message(format!(
                    "{} does not exist",
                    storage.display()
                )));
            }
            if !discover::session_directory(storage, session_id).is_dir() {
                return Err(InvokeError::Message(format!(
                    "no session {session_id} under {path}"
                )));
            }
            Ok(load_timeslice_columns(storage, session_id))
        }
        "scan_kit_channel_catalog" => {
            let path = required_str(input, "path")?;
            let session_id = required_str(input, "session_id")?;
            let storage = Path::new(path);
            if !storage.exists() {
                return Err(InvokeError::Message(format!(
                    "{} does not exist",
                    storage.display()
                )));
            }
            if !discover::session_directory(storage, session_id).is_dir() {
                return Err(InvokeError::Message(format!(
                    "no session {session_id} under {path}"
                )));
            }
            Ok(json!({ "channels": channel_catalog(storage, session_id) }))
        }
        "scan_kit_analysis_scene" => {
            let view = required_str(input, "view")?;
            let path = required_str(input, "path")?;
            let session_ids = required_strings(input, "session_ids")?;
            let mut options = match input.get("options") {
                // null means "defaults" to JSON callers; a scalar does not.
                Some(value) if value.is_null() => json!({}),
                Some(value) if !value.is_object() => {
                    return Err(InvokeError::Message("options must be an object".into()));
                }
                Some(value) => value.clone(),
                None => json!({}),
            };
            normalize_options(&mut options);
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
            let files = input
                .get("files")
                .cloned()
                .ok_or_else(|| InvokeError::Message("files is required".into()))?;
            let hide = input.get("hide_unused").and_then(Value::as_bool);
            config::save_folder(&db, Path::new(source), Path::new(dest), &files, hide)
                .map_err(InvokeError::Message)
        }
        "scan_kit_config_tune" => {
            let db = database_path(input)?;
            config::tune(input, &db).map_err(InvokeError::Message)
        }
        "scan_kit_config_diff" => config::diff(input).map_err(InvokeError::Message),
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
        "scan_kit_discover_subnet" => {
            let subnet = required_str(input, "subnet")?;
            device_scan::scan_subnet(subnet).map_err(InvokeError::Message)
        }
        "scan_kit_scanners" => {
            let db = database_path(input)?;
            let action = required_str(input, "action")?;
            scanners(&db, action, input.get("folder").and_then(Value::as_str))
                .map_err(InvokeError::Message)
        }
        _ => Err(InvokeError::UnknownTool {
            name: name.to_owned(),
        }),
    }
}

fn scanners(db: &Path, action: &str, folder: Option<&str>) -> Result<Value, String> {
    match action {
        "add" => {
            let folder = folder.ok_or("scanner folder required")?;
            // File IO outside STORE_LOCK — resolve before taking it.
            let calibration = calibration::CtCalibration::resolve(folder)?;
            store::with_store(db, |conn| {
                conn.execute_batch("BEGIN IMMEDIATE")
                    .map_err(|err| err.to_string())?;
                let result = (|| {
                    let mut list = store::scanners(conn)?;
                    if !list.iter().any(|(_, existing)| existing == folder) {
                        list.push((calibration.name.clone(), folder.to_owned()));
                    }
                    store::set_scanners(conn, &list)
                })();
                if result.is_err() {
                    let _ = conn.execute_batch("ROLLBACK");
                    return result;
                }
                conn.execute_batch("COMMIT").map_err(|err| err.to_string())
            })?;
        }
        "remove" => {
            let folder = folder.ok_or("scanner folder required")?;
            store::with_store(db, |conn| {
                conn.execute_batch("BEGIN IMMEDIATE")
                    .map_err(|err| err.to_string())?;
                let result = (|| {
                    let list: Vec<(String, String)> = store::scanners(conn)?
                        .into_iter()
                        .filter(|(_, existing)| existing != folder)
                        .collect();
                    store::set_scanners(conn, &list)
                })();
                if result.is_err() {
                    let _ = conn.execute_batch("ROLLBACK");
                    return result;
                }
                conn.execute_batch("COMMIT").map_err(|err| err.to_string())
            })?;
        }
        "list" => {}
        other => return Err(format!("unknown scanners action {other}")),
    }
    let stored = store::with_store(db, |conn| store::scanners(conn))?;
    let mut entries = Vec::new();
    for (label, folder) in stored {
        // Digest reads files — outside the lock.
        let digest = calibration::CtCalibration::resolve(&folder)
            .map(|calibration| calibration.digest())
            .unwrap_or_default();
        entries.push(json!({ "label": label, "folder": folder, "digest": digest }));
    }
    Ok(json!({ "scanners": entries }))
}

/// Copy the connected session into `dest`, calling `keep` after each file.
pub fn copy_runner(dest: &str, keep: &mut dyn FnMut(u64, u64) -> bool) -> Result<Value, String> {
    runner::download_with(&default_db_path(), dest, keep)
}

/// Where the desktop shell appends panics, beside the store.
pub fn crash_log_path() -> PathBuf {
    default_db_path().with_file_name("crash.log")
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
pub fn open_library(db: &Path, data_dir: &Path) -> Result<Value, String> {
    index_library(db, data_dir, &mut |_, _, _| true)
}

/// Same index as [`open_library`], publishing `done`, `total`, and the rows so far.
///
/// `keep` returns false to roll the index back. The final value is still the committed rows.
pub fn index_library(
    db: &Path,
    data_dir: &Path,
    keep: &mut dyn FnMut(u64, u64, Option<&Value>) -> bool,
) -> Result<Value, String> {
    // File and network IO run outside STORE_LOCK — a remote library can take
    // minutes and must not stall every other store call behind it. The lock
    // covers the short DB sections only.
    let spec = data_dir.to_string_lossy();
    let (root, opened) = location::open_root(&spec)?;
    let found = match &opened {
        location::Opened::Local(path) => {
            discover::discover(path, &mut |done, total| keep(done, total, None))?
        }
        location::Opened::Remote(url) => {
            location::discover_remote(url, &mut |done, total| keep(done, total, None))?
        }
    };
    // Remote fingerprints stat each session over a fresh connection —
    // compute them before STORE_LOCK so the write tx below stays short.
    let total = found.entries.len() as u64;
    let mut fingerprints = Vec::with_capacity(found.entries.len());
    for entry in &found.entries {
        fingerprints.push(discover::storage_fingerprint(
            &entry.storage_path,
            &entry.session_id,
        ));
        // A remote fingerprint opens a connection per session - cancel must
        // land between them, not only after the whole phase.
        if !keep(fingerprints.len() as u64, total, None) {
            return Err("cancelled".into());
        }
    }
    let conn = &mut store::open_connection(db)?;
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
    let walk_failed = found.walk_failed;
    let (lib_id, pending) = store::locked(|| {
        // Two phases: the write transaction covers the row-sync decisions only.
        // hydrate does file/network IO — inside the transaction a remote
        // library could hold the write lock for minutes and every other
        // process's write (a note, window geometry) would hit busy_timeout.
        // record_meta afterwards autocommits per entry, and a checked=0 row
        // left by a cancel rehydrates on the next sync.
        // Immediate like the scanner paths — a future SELECT-first edit must
        // not reopen the read-modify-write window this transaction closes.
        let tx = conn
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
            .map_err(|err| err.to_string())?;
        // ensure_library only — import_legacy/backfill read legacy JSON off
        // disk and must not hold the write lock while a share stalls.
        let lib_id = ensure_library(&tx, &root)?;
        let ids = places
            .iter()
            .map(|(session_id, _)| session_id.clone())
            .collect::<HashSet<_>>();
        if !walk_failed {
            // A partial listing must not look like every session was deleted.
            delete_missing(&tx, lib_id, &ids)?;
        }
        let exam_ids = exams
            .iter()
            .map(|exam| exam.study_uid.clone())
            .collect::<HashSet<_>>();
        if !walk_failed {
            delete_missing_exams(&tx, lib_id, &exam_ids)?;
        }
        for exam in &exams {
            upsert_exam(&tx, lib_id, exam)?;
        }
        remember_data_dir(&tx, &root)?;
        let mut pending = HashSet::new();
        for (index, entry) in found.entries.iter().enumerate() {
            if let Cache::Hydrate = sync_entry(&tx, lib_id, entry, fingerprints[index])? {
                pending.insert(index);
            }
        }
        tx.commit().map_err(|err| err.to_string())?;
        // Both read legacy JSON off disk — autocommit after commit so a
        // stalled share can't sit inside the write lock. Fill-blanks only,
        // safe to replay if the next scan finds them half-done.
        import_legacy(conn, lib_id, Path::new(&root))?;
        backfill_blank_notes(conn, lib_id, Path::new(&root))?;
        Ok((lib_id, pending))
    })?;
    // Refresh the recorded bindings now, not only after the hydrate loop -
    // a cancel or error below used to leave new storage_paths committed
    // while the in-memory maps still resolved the old locations.
    location::replace_sessions(&root, &places);
    store::locked(|| publish_catalog(conn))?;
    if !keep(0, total, None) {
        return Err("cancelled".into());
    }
    for (index, entry) in found.entries.iter().enumerate() {
        if pending.contains(&index) {
            // hydrate does file/network IO — outside the lock; the write is
            // a short autocommit on the shared connection.
            let (meta, readable) = hydrate(entry, fingerprints[index]);
            store::locked(|| {
                record_meta(
                    conn,
                    lib_id,
                    entry,
                    meta.as_ref(),
                    readable,
                    fingerprints[index],
                )
            })?;
        }
        let done = (index as u64) + 1;
        // library_rows lists everything indexed so far — per entry that is
        // O(n²). Refresh the preview on a stride; progress still ticks every
        // entry and the final result carries the full row set.
        let preview = if index % 16 == 0 {
            Some(store::locked(|| {
                Ok(json!({
                    "root": root,
                    "rows": library_rows(conn, lib_id)?,
                    "selected": [],
                }))
            })?)
        } else {
            None
        };
        if !keep(done, total, preview.as_ref()) {
            return Err("cancelled".into());
        }
    }
    let (rows, selected) = store::locked(|| {
        Ok((
            library_rows(conn, lib_id)?,
            store::selected_session_ids(conn, lib_id)?,
        ))
    })?;
    // Selection echoes only ids that actually resolved - a ghost id must
    // not read back as if the session still exists.
    let present: std::collections::HashSet<&str> = rows
        .iter()
        .filter_map(|row| row.get("session_id").and_then(Value::as_str))
        .collect();
    let selected: Vec<&String> = selected
        .iter()
        .filter(|id| present.contains(id.as_str()))
        .collect();
    Ok(
        json!({ "root": root, "rows": rows, "selected": selected, "skipped": skipped, "walk_failed": walk_failed }),
    )
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

pub(crate) fn parent_folder(storage: &str) -> String {
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

fn hydrate(entry: &Discovered, stamped: Option<(i64, i64)>) -> (Option<SessionMeta>, bool) {
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
    // A None meta is ambiguous: a reachable storage path without a summary is
    // a legitimate absence, but an unreachable one is a transient failure —
    // the fingerprint stat is the same probe sync_entry already pays for.
    let fresh = discover::storage_fingerprint(&entry.storage_path, &entry.session_id);
    // Meta hydrated after the fingerprint stat can describe newer bytes than
    // the row certifies — a moved file stays retryable instead of sealing
    // stale meta under the old fingerprint.
    let readable = (meta.is_some() || fresh.is_some()) && fresh == stamped;
    (meta, readable)
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
        Some(path) if !path.trim().is_empty() => Ok(PathBuf::from(path)),
        _ => Ok(default_db_path()),
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
        // A real StudyInstanceUID: a bare DICM preamble is correctly skipped
        // by the exam probe now.
        scan_kit_dicom::write_header(&exam.join("1.2.3"), "1.2", "CT", "1.2.3", "1.2.4").unwrap();
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

    #[test]
    fn a_deleted_data_dir_can_still_be_forgotten() {
        let root = workspace("forget-dead");
        let db = root.join("test.sqlite");
        let db_arg = db.to_string_lossy().into_owned();
        let site = root.join("site");
        write_session(
            &site,
            "kept",
            KEPT_SUMMARY,
            "energy,X_POSITION,Y_POSITION\n1,0,0\n",
        );
        invoke(
            "scan_kit_open_library",
            &json!({ "path": site.to_string_lossy(), "db_path": &db_arg }),
        )
        .unwrap();
        let catalog = invoke("scan_kit_read_catalog", &json!({ "db_path": &db_arg })).unwrap();
        let stored = catalog["locations"][0].as_str().unwrap().to_owned();
        fs::remove_dir_all(&site).unwrap();
        let forgotten = invoke(
            "scan_kit_forget_data_dir",
            &json!({ "path": stored, "db_path": &db_arg }),
        )
        .unwrap();
        assert!(forgotten["locations"].as_array().unwrap().is_empty());
        let dirs = invoke("scan_kit_data_dirs", &json!({ "db_path": &db_arg })).unwrap();
        assert!(dirs.as_array().unwrap().is_empty());
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn an_unreachable_remote_data_dir_can_still_be_forgotten() {
        let root = workspace("forget-remote");
        let db = root.join("test.sqlite");
        let db_arg = db.to_string_lossy().into_owned();
        let url = "sftp://scan-kit@example.invalid/dead-lib";
        let canonical = location::canonical_root(url).unwrap();
        with_store(&db, |conn| store::remember_data_dir(conn, &canonical)).unwrap();
        let forgotten = invoke(
            "scan_kit_forget_data_dir",
            &json!({ "path": url, "db_path": &db_arg }),
        )
        .unwrap();
        assert!(forgotten["locations"].as_array().unwrap().is_empty());
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_missing_data_dir_errors_instead_of_indexing_nothing() {
        let root = workspace("open-dead");
        let db = root.join("test.sqlite");
        let db_arg = db.to_string_lossy().into_owned();
        let gone = root.join("gone");
        // Opening a dead path must stay an error: a silent success would let
        // delete_missing wipe the recorded sessions for that root.
        let err = invoke(
            "scan_kit_open_library",
            &json!({ "path": gone.to_string_lossy(), "db_path": &db_arg }),
        )
        .unwrap_err();
        assert!(err.to_string().contains("not a directory"), "{err}");
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_local_session_moves_to_the_trash() {
        let root = workspace("trash");
        let db = root.join("test.sqlite");
        fs::create_dir_all(root.join("sess")).unwrap();
        fs::write(root.join("sess/input_map.csv"), b"energy\n1\n").unwrap();
        fs::write(root.join("sess.zip"), b"zip").unwrap();
        let listed = invoke(
            "scan_kit_session_storage_paths",
            &json!({ "path": root.to_string_lossy(), "session_id": "sess" }),
        )
        .unwrap();
        assert_eq!(listed["paths"].as_array().unwrap().len(), 2);
        let result = invoke(
            "scan_kit_trash_session",
            &json!({
                "path": root.to_string_lossy(),
                "session_id": "sess",
                "db_path": db.to_string_lossy(),
            }),
        )
        .unwrap();
        assert_eq!(result["errors"].as_array().unwrap().len(), 0);
        assert!(!root.join("sess").exists());
        assert!(!root.join("sess.zip").exists());
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn analysis_scene_rejects_a_non_object_options_value() {
        // A scalar options bag used to be silently replaced with defaults —
        // a typo'd caller could not tell its settings never applied.
        let error = invoke(
            "scan_kit_analysis_scene",
            &json!({
                "view": "distribution",
                "path": "/tmp",
                "session_ids": ["sess"],
                "options": "wide",
            }),
        )
        .unwrap_err();
        assert!(error.to_string().contains("options"), "{error}");
        // An explicit null still means "defaults", not an error.
        assert!(invoke(
            "scan_kit_analysis_scene",
            &json!({
                "view": "distribution",
                "path": "/tmp",
                "session_ids": ["sess"],
                "options": null,
            }),
        )
        .is_ok());
    }

    #[test]
    fn config_save_requires_a_files_list() {
        // Missing `files` defaulted to [] — a dest-pruning save that writes
        // nothing, with no signal the argument was dropped.
        let error = invoke(
            "scan_kit_config_save",
            &json!({
                "db_path": "/tmp/db.sqlite",
                "source": "/tmp/a",
                "dest": "/tmp/b",
            }),
        )
        .unwrap_err();
        assert!(error.to_string().contains("files"), "{error}");
    }

    #[test]
    fn a_session_with_no_files_keeps_its_row_and_note() {
        // Trashing a session whose files are already gone must not erase the
        // index row — the note/selection would vanish with it.
        let root = workspace("trash-missing");
        let db = root.join("test.sqlite");
        let root_str = root.to_string_lossy().into_owned();
        store::with_store(&db, |conn| {
            store::set_note(conn, &root_str, "sess", "keep me")
        })
        .unwrap();
        let result = invoke(
            "scan_kit_trash_session",
            &json!({
                "path": &root_str,
                "session_id": "sess",
                "db_path": db.to_string_lossy(),
            }),
        )
        .unwrap();
        assert_eq!(result["errors"].as_array().unwrap().len(), 0);
        let remaining = store::with_store(&db, |conn| {
            conn.query_row(
                "SELECT note FROM sessions WHERE session_id = 'sess'",
                [],
                |row| row.get::<_, String>(0),
            )
            .map_err(|err| err.to_string())
        })
        .unwrap();
        assert_eq!(remaining, "keep me");
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_remote_session_is_deleted_in_place() {
        let root = workspace("remote-trash");
        let db = root.join("test.sqlite");
        let url = format!(
            "sftp://scan-kit@example.invalid/trash/{}/{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let canonical = location::install_fixture(
            &url,
            &[
                ("sess/input_map.csv", b"energy\n1\n"),
                (
                    "sess/termination_summary.txt",
                    b"Configuration name: trashme\n",
                ),
                ("sess.zip", b"zip"),
                ("keep.txt", b"keep"),
            ],
            false,
        )
        .unwrap();
        let listed = invoke(
            "scan_kit_session_storage_paths",
            &json!({ "path": &url, "session_id": "sess" }),
        )
        .unwrap();
        let paths = listed["paths"].as_array().unwrap();
        assert_eq!(paths.len(), 2);
        assert!(paths
            .iter()
            .all(|p| p.as_str().unwrap().starts_with(&canonical)));
        let result = invoke(
            "scan_kit_trash_session",
            &json!({ "path": &url, "session_id": "sess", "db_path": db.to_string_lossy() }),
        )
        .unwrap();
        assert_eq!(result["errors"].as_array().unwrap().len(), 0);
        let remote = location::require_remote(&canonical).unwrap();
        let mut fs = remote::connect(&remote, "").unwrap();
        let base = remote.path.trim_end_matches('/');
        assert!(fs.stat(&format!("{base}/sess")).unwrap().is_none());
        assert!(fs.stat(&format!("{base}/sess.zip")).unwrap().is_none());
        assert!(fs.stat(&format!("{base}/keep.txt")).unwrap().is_some());
        let _ = fs::remove_dir_all(&root);
        let _ = fs::remove_dir_all(location::cache_directory(&canonical));
    }

    #[test]
    fn a_blank_db_path_means_the_default_store() {
        let path = database_path(&json!({"db_path": ""})).unwrap();
        assert_eq!(path, default_db_path());
        let path = database_path(&json!({"db_path": "  "})).unwrap();
        assert_eq!(path, default_db_path());
    }

    #[test]
    fn a_missing_library_path_is_an_error_not_an_empty_answer() {
        for tool in ["scan_kit_load_timeslice", "scan_kit_channel_catalog"] {
            let result = invoke(
                tool,
                &json!({"path": "C:/definitely/not/here", "session_id": "x"}),
            );
            assert!(result.is_err(), "{tool} reported success on a bad path");
        }
    }

    #[test]
    fn a_wrong_session_id_is_an_error_not_an_empty_answer() {
        let root = std::env::temp_dir().join(format!("scan-kit-sess-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        for tool in ["scan_kit_load_timeslice", "scan_kit_channel_catalog"] {
            let result = invoke(
                tool,
                &json!({"path": root.display().to_string(), "session_id": "ghost"}),
            );
            assert!(result.is_err(), "{tool} reported success on a bad session");
        }
        let _ = fs::remove_dir_all(&root);
    }
}
