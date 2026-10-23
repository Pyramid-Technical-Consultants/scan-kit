//! Configuration folder, Pyramid `.md5` sidecars, and the four tuners.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use chrono::{Local, TimeZone};
use md5::{Digest, Md5};
use scan_kit_core::{apply_form, config_form, run_tune, tune_catalog, TuneSpots};
use serde_json::{json, Value};
use similar::{ChangeTag, TextDiff};

use crate::discover;
use crate::store::{self, with_store};
use crate::tables::{slice_table, tune_spot_table};

const DOSE: [&str; 3] = ["ic1_dose", "ic2_dose", "ic3_dose"];
const MEASURED: [&str; 4] = ["ic1_x", "ic1_y", "ic2_x", "ic2_y"];
const ERRORS: [&str; 4] = ["ic1_x_err", "ic1_y_err", "ic2_x_err", "ic2_y_err"];
const SIGMA: [&str; 4] = ["ic1_sig_x", "ic1_sig_y", "ic2_sig_x", "ic2_sig_y"];

pub fn catalog(db: &Path) -> Value {
    let mut body = tune_catalog();
    let (dir, hide) = with_store(db, |conn| store::config_settings(conn))
        .ok()
        .unwrap_or((None, false));
    body["config_dir"] = json!(dir);
    body["hide_unused"] = json!(hide);
    if let Some(workflows) = body["workflows"].as_array_mut() {
        workflows.push(json!({
            "id": "config_diff",
            "name": "Config Diff",
            "description": "Diff all config files against selected sessions",
            "sessions": true,
            "diff": true,
            "params": [],
        }));
    }
    body
}

pub fn open_folder(
    db: &Path,
    path: Option<&str>,
    data_dir: Option<&str>,
    session_id: Option<&str>,
) -> Result<Value, String> {
    let remembered = with_store(db, |conn| store::config_settings(conn)).ok();
    let (remembered_dir, hide) = remembered.unwrap_or((None, false));
    let explicit = path
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .map(PathBuf::from);
    let from_session = match (data_dir, session_id) {
        (Some(dir), Some(id)) => session_config_dir(dir, id),
        _ => None,
    };
    let chosen = explicit
        .or(from_session)
        .or_else(|| remembered_dir.map(PathBuf::from));
    let Some(folder) = chosen else {
        return Ok(json!({ "path": Value::Null, "files": [], "hide_unused": hide }));
    };
    if !folder.is_dir() {
        return Err(format!(
            "Configuration folder is not a directory: {}",
            folder.display()
        ));
    }
    let files = list_xml(&folder)?;
    let text = folder.to_string_lossy().to_string();
    let _ = with_store(db, |conn| {
        store::set_config_settings(conn, Some(&text), None)
    });
    Ok(json!({ "path": text, "files": files, "hide_unused": hide }))
}

pub fn set_hide_unused(db: &Path, hide: bool) -> Result<Value, String> {
    with_store(db, |conn| {
        store::set_config_settings(conn, None, Some(hide))
    })?;
    Ok(json!({ "hide_unused": hide }))
}

pub fn read_form(path: &Path) -> Result<Value, String> {
    let xml = fs::read_to_string(path)
        .map_err(|err| format!("Could not read {}: {err}", path.display()))?;
    let form = config_form(&xml)?;
    Ok(json!({
        "path": path.to_string_lossy(),
        "xml": xml,
        "form": form,
        "integrity": integrity_value(path),
    }))
}

pub fn apply(xml: &str, form: &Value) -> Result<Value, String> {
    let xml = apply_form(xml, form)?;
    let form = config_form(&xml)?;
    Ok(json!({ "xml": xml, "form": form }))
}

pub fn save_folder(
    db: &Path,
    source: &Path,
    dest: &Path,
    files: &Value,
    hide: Option<bool>,
) -> Result<Value, String> {
    if !source.is_dir() {
        return Err(format!(
            "Configuration folder is not a directory: {}",
            source.display()
        ));
    }
    let source_key = folder_key(source);
    let dest_key = folder_key(dest);
    if source_key != dest_key {
        if dest_key.starts_with(&format!("{source_key}/")) {
            return Err("Save the configuration beside the source folder, not inside it.".into());
        }
        copy_tree(source, dest, 0)?;
    } else if !dest.exists() {
        fs::create_dir_all(dest).map_err(|err| err.to_string())?;
    }
    let written = files.as_array().ok_or("files must be a list")?;
    for file in written {
        let rel = file
            .get("rel")
            .and_then(Value::as_str)
            .ok_or("each file needs a relative path")?;
        let xml = file
            .get("xml")
            .and_then(Value::as_str)
            .ok_or("each file needs xml text")?;
        if !xml.trim_start().starts_with('<') {
            return Err(format!("{rel} is not XML"));
        }
        let target = under(dest, rel)?;
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent).map_err(|err| err.to_string())?;
        }
        fs::write(&target, xml)
            .map_err(|err| format!("Could not write {}: {err}", target.display()))?;
    }
    let refreshed = commit_integrity(dest)?;
    let dest_text = dest.to_string_lossy().to_string();
    let _ = with_store(db, |conn| {
        store::set_config_settings(conn, Some(&dest_text), hide)
    });
    Ok(json!({
        "path": dest_text,
        "files": list_xml(dest)?,
        "sidecars": refreshed,
    }))
}

pub fn tune(input: &Value, _db: &Path) -> Result<Value, String> {
    let workflow = text(input, "workflow")?;
    let mut xml = text(input, "xml")?.to_owned();
    if let Some(form) = input.get("form") {
        xml = apply_form(&xml, form)?;
    }
    let data_dir = text(input, "data_dir").unwrap_or("").trim().to_owned();
    let session_ids = strings(input, "session_ids");
    if session_ids.is_empty() {
        return Err("Select at least one session.".into());
    }
    if data_dir.is_empty() {
        return Err("Enter the folder containing session data.".into());
    }
    let root = Path::new(&data_dir);
    if !root.is_dir() {
        return Err("Session data folder is not a directory.".into());
    }
    let mut found = Vec::new();
    let mut warnings = Vec::new();
    for session_id in &session_ids {
        if session_present(root, session_id) {
            found.push(session_id.clone());
        } else {
            warnings.push(format!(
                "Session {session_id} was not found under {data_dir}."
            ));
        }
    }
    if found.is_empty() {
        return Err(format!("No selected sessions were found under {data_dir}."));
    }
    let params = input.get("params").cloned().unwrap_or_else(|| json!({}));
    let timeslice = workflow == "position_offset_tuning"
        && params.get("data_source").and_then(Value::as_str) == Some("timeslice");
    let spots = load_spots(root, &found, timeslice, &mut warnings);
    let label = if found.len() == 1 {
        found[0].clone()
    } else {
        format!("{} sessions", found.len())
    };
    let mut result = run_tune(workflow, &xml, &spots, &params, &label)?;
    if let Some(list) = result.get_mut("warnings").and_then(Value::as_array_mut) {
        for warning in warnings {
            list.insert(0, json!(warning));
        }
    }
    Ok(result)
}

/// Read-only whole-tree diff: the loaded config folder vs each selected
/// session's `config/` tree (map2map, dcs, nozzle), one column per session.
pub fn diff(input: &Value) -> Result<Value, String> {
    let config_dir = text(input, "config_dir").unwrap_or("").trim();
    if config_dir.is_empty() {
        return Err("Open a configuration folder.".into());
    }
    let loaded_dir = Path::new(config_dir);
    if !loaded_dir.is_dir() {
        return Err(format!(
            "Configuration folder is not a directory: {}",
            loaded_dir.display()
        ));
    }
    let data_dir = text(input, "data_dir").unwrap_or("").trim().to_owned();
    let session_ids = strings(input, "session_ids");
    if session_ids.is_empty() {
        return Err("Select at least one session.".into());
    }
    let root = Path::new(&data_dir);
    if data_dir.is_empty() || !root.is_dir() {
        return Err("Session data folder is not a directory.".into());
    }
    let mut warnings = Vec::new();
    let mut found = Vec::new();
    for session_id in &session_ids {
        if session_present(root, session_id) {
            found.push(session_id.clone());
        } else {
            warnings.push(format!(
                "Session {session_id} was not found under {data_dir}."
            ));
        }
    }
    if found.is_empty() {
        return Err(format!("No selected sessions were found under {data_dir}."));
    }
    let base_root = diff_root(loaded_dir);
    let mut base_files = walk_files(&base_root)?;
    // The open file diffs its in-memory text (form edits applied) so pending
    // changes are visible; every other file diffs on disk.
    let mut edited = String::new();
    let file_rel = text(input, "file").unwrap_or("").trim();
    if !file_rel.is_empty() {
        if let Some(xml) = input
            .get("xml")
            .and_then(Value::as_str)
            .filter(|xml| !xml.is_empty())
        {
            let mut text = xml.to_owned();
            if let Some(form) = input.get("form") {
                text = apply_form(&text, form)?;
            }
            let rel = match loaded_dir.strip_prefix(&base_root) {
                Ok(sub) if !sub.as_os_str().is_empty() => {
                    format!("{}/{file_rel}", relative(&base_root, loaded_dir))
                }
                _ => file_rel.to_owned(),
            };
            let bytes = text.into_bytes();
            if base_files.get(&rel) != Some(&bytes) {
                base_files.insert(rel.clone(), bytes);
                edited = rel;
            }
        }
    }
    let base_key = folder_key(&base_root);
    let mut sessions = Vec::new();
    let mut summary_parts = Vec::new();
    for session_id in &found {
        let session_root = discover::session_directory(root, session_id).join("config");
        // With pending edits the self column still shows the unsaved changes.
        if edited.is_empty() && session_root.is_dir() && folder_key(&session_root) == base_key {
            sessions.push(json!({
                "id": session_id,
                "self": true,
                "files": [],
            }));
            summary_parts.push(format!("{session_id}: its own config"));
            continue;
        }
        if !session_root.is_dir() {
            sessions.push(json!({
                "id": session_id,
                "warning": format!("No config/ tree in session {session_id}."),
                "files": [],
            }));
            summary_parts.push(format!("{session_id}: no config"));
            continue;
        }
        let session_files = walk_files(&session_root)?;
        let files = diff_trees(&base_files, &session_files, &edited);
        let changed = files.iter().filter(|file| file["status"] != "same").count();
        let total = files.len();
        summary_parts.push(format!("{session_id}: {changed} of {total} differ"));
        sessions.push(json!({
            "id": session_id,
            "root": session_root.to_string_lossy(),
            "files": files,
        }));
    }
    Ok(json!({
        "base": base_root.to_string_lossy(),
        "edited": edited,
        "summary": summary_parts.join(" · "),
        "warnings": warnings,
        "sessions": sessions,
        "changed": false,
        "columns": [],
        "rows": [],
    }))
}

/// The innermost `config/` ancestor of the loaded folder, so a loaded
/// `config/map2map` diffs the whole `config/` tree. A folder outside any
/// `config/` tree diffs as-is (a plain copy still path-matches session trees).
fn diff_root(dir: &Path) -> PathBuf {
    for ancestor in dir.ancestors().skip(1) {
        if ancestor.file_name().is_some_and(|name| name == "config") {
            return ancestor.to_path_buf();
        }
    }
    dir.to_path_buf()
}

fn walk_files(root: &Path) -> Result<BTreeMap<String, Vec<u8>>, String> {
    let mut files = BTreeMap::new();
    walk_files_at(root, root, 0, &mut files)?;
    Ok(files)
}

fn walk_files_at(
    root: &Path,
    dir: &Path,
    depth: usize,
    files: &mut BTreeMap<String, Vec<u8>>,
) -> Result<(), String> {
    if depth > MAX_XML_WALK_DEPTH {
        return Err(format!(
            "config tree is deeper than {} levels under {}",
            MAX_XML_WALK_DEPTH,
            root.display()
        ));
    }
    for entry in fs::read_dir(dir).map_err(|err| err.to_string())? {
        let entry = entry.map_err(|err| err.to_string())?;
        let path = entry.path();
        if path.is_dir() {
            walk_files_at(root, &path, depth + 1, files)?;
            continue;
        }
        let name = entry.file_name().to_string_lossy().to_ascii_lowercase();
        if name.ends_with(".md5") {
            continue;
        }
        if let Ok(bytes) = fs::read(&path) {
            files.insert(relative(root, &path), bytes);
        }
    }
    Ok(())
}

const DIFF_MAX_BYTES: usize = 1_500_000;
const DIFF_MAX_LINES: usize = 2_000;
const DIFF_CONTEXT: usize = 3;

/// Flat line rows for the diff view: "context" | "add" | "del" carry `old_no`
/// and `new_no` (1-based) plus `text`; "gap" carries `count` of folded lines.
fn diff_trees(
    base: &BTreeMap<String, Vec<u8>>,
    theirs: &BTreeMap<String, Vec<u8>>,
    edited: &str,
) -> Vec<Value> {
    let paths: std::collections::BTreeSet<&String> = base.keys().chain(theirs.keys()).collect();
    let mut files: Vec<(u8, Value)> = Vec::new();
    for rel in paths {
        let ours = base.get(rel);
        let other = theirs.get(rel);
        let (status, lines) = match (ours, other) {
            (None, Some(bytes)) => ("added", whole_file(bytes, "add")),
            (Some(bytes), None) => ("removed", whole_file(bytes, "del")),
            (Some(ours), Some(other)) if ours == other => ("same", Vec::new()),
            (Some(ours), Some(other)) => match diff_lines(ours, other) {
                Some(emitted) => ("changed", emitted),
                None if ours.len() > DIFF_MAX_BYTES || other.len() > DIFF_MAX_BYTES => {
                    ("large", Vec::new())
                }
                None => ("binary", Vec::new()),
            },
            (None, None) => continue,
        };
        let rank = match status {
            "changed" => 0,
            "added" => 1,
            "removed" => 2,
            "same" => 4,
            _ => 3,
        };
        files.push((
            rank,
            json!({
                "rel": rel,
                "status": status,
                "edited": rel == edited,
                "adds": lines.iter().filter(|line| line["kind"] == "add").count(),
                "dels": lines.iter().filter(|line| line["kind"] == "del").count(),
                "lines": lines,
            }),
        ));
    }
    files.sort_by(|(a, left), (b, right)| {
        a.cmp(b).then(
            left["rel"]
                .as_str()
                .unwrap_or("")
                .cmp(right["rel"].as_str().unwrap_or("")),
        )
    });
    files.into_iter().map(|(_, file)| file).collect()
}

fn text_lines(bytes: &[u8]) -> Option<String> {
    if bytes.len() > DIFF_MAX_BYTES || bytes[..bytes.len().min(8192)].contains(&0) {
        return None;
    }
    Some(String::from_utf8_lossy(bytes).into_owned())
}

fn whole_file(bytes: &[u8], kind: &str) -> Vec<Value> {
    let Some(text) = text_lines(bytes) else {
        return Vec::new();
    };
    let mut lines = Vec::new();
    for (index, text) in text.lines().enumerate() {
        if index >= DIFF_MAX_LINES {
            return Vec::new();
        }
        let no = index + 1;
        lines.push(if kind == "add" {
            json!({ "kind": kind, "new_no": no, "text": text })
        } else {
            json!({ "kind": kind, "old_no": no, "text": text })
        });
    }
    lines
}

fn diff_lines(ours: &[u8], theirs: &[u8]) -> Option<Vec<Value>> {
    let old = text_lines(ours)?;
    let new = text_lines(theirs)?;
    let diff = TextDiff::from_lines(&old, &new);
    let mut lines = Vec::new();
    let mut previous_old = 0_usize;
    for (hunk_index, hunk) in diff.grouped_ops(DIFF_CONTEXT).iter().enumerate() {
        if hunk_index > 0 {
            if let Some(first) = hunk.first() {
                let gap = first.as_tag_tuple().1.start.saturating_sub(previous_old);
                if gap > 0 {
                    lines.push(json!({ "kind": "gap", "count": gap }));
                }
            }
        }
        for op in hunk {
            for change in diff.iter_changes(op) {
                let text = change.value().trim_end_matches(['\n', '\r']);
                match change.tag() {
                    ChangeTag::Equal => lines.push(json!({
                        "kind": "context",
                        "old_no": change.old_index().map(|n| n + 1),
                        "new_no": change.new_index().map(|n| n + 1),
                        "text": text,
                    })),
                    ChangeTag::Delete => lines.push(json!({
                        "kind": "del",
                        "old_no": change.old_index().map(|n| n + 1),
                        "text": text,
                    })),
                    ChangeTag::Insert => lines.push(json!({
                        "kind": "add",
                        "new_no": change.new_index().map(|n| n + 1),
                        "text": text,
                    })),
                }
                if lines.len() > DIFF_MAX_LINES {
                    return None;
                }
            }
        }
        if let Some(last) = hunk.last() {
            previous_old = last.as_tag_tuple().1.end;
        }
    }
    Some(lines)
}

pub fn integrity(path: &Path) -> Value {
    integrity_value(path)
}

fn load_spots(
    root: &Path,
    sessions: &[String],
    timeslice: bool,
    warnings: &mut Vec<String>,
) -> TuneSpots {
    let mut spots = TuneSpots {
        energy: Vec::new(),
        weight: Vec::new(),
        plan_x: Vec::new(),
        plan_y: Vec::new(),
        measured: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
        error: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
        sigma: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
        mu: [Vec::new(), Vec::new(), Vec::new()],
        charge: Vec::new(),
    };
    for session_id in sessions {
        let table = if timeslice {
            slice_table(root, session_id)
        } else {
            tune_spot_table(root, session_id)
        };
        let n = table.get("energy").map(Vec::len).unwrap_or(0);
        if n == 0 {
            warnings.push(format!(
                "No {} data in {session_id}.",
                if timeslice { "timeslice" } else { "spot" }
            ));
            continue;
        }
        extend_column(&mut spots.energy, &table, "energy", n);
        extend_column(&mut spots.weight, &table, "target_mu", n);
        extend_column(&mut spots.charge, &table, "target_mu", n);
        extend_column(&mut spots.plan_x, &table, "plan_x", n);
        extend_column(&mut spots.plan_y, &table, "plan_y", n);
        for index in 0..4 {
            extend_column(&mut spots.measured[index], &table, MEASURED[index], n);
            extend_named(&mut spots.error[index], &table, ERRORS[index], n);
            let raw = format!("{}_raw", SIGMA[index]);
            let key = if table
                .get(&raw)
                .is_some_and(|values| values.iter().any(|value| value.is_finite()))
            {
                raw
            } else {
                SIGMA[index].to_string()
            };
            extend_column(&mut spots.sigma[index], &table, &key, n);
        }
        for index in 0..3 {
            extend_named(&mut spots.mu[index], &table, DOSE[index], n);
        }
    }
    spots
}

fn extend_column(dest: &mut Vec<f64>, table: &BTreeMap<String, Vec<f32>>, key: &str, n: usize) {
    let values = table.get(key).map(Vec::as_slice).unwrap_or(&[]);
    dest.extend((0..n).map(|index| finite(values.get(index).copied())));
}

fn extend_named(dest: &mut Vec<f64>, table: &BTreeMap<String, Vec<f32>>, key: &str, n: usize) {
    if table.contains_key(key) {
        extend_column(dest, table, key, n);
    } else {
        dest.extend(std::iter::repeat_n(f64::NAN, n));
    }
}

fn finite(value: Option<f32>) -> f64 {
    value
        .map(|value| value as f64)
        .filter(|value| value.is_finite())
        .unwrap_or(f64::NAN)
}

fn session_config_dir(data_dir: &str, session_id: &str) -> Option<PathBuf> {
    let data_dir = data_dir.trim();
    let session_id = session_id.trim();
    if data_dir.is_empty() || session_id.is_empty() {
        return None;
    }
    let dir = discover::session_directory(Path::new(data_dir), session_id)
        .join("config")
        .join("map2map");
    dir.is_dir().then_some(dir)
}

fn session_present(root: &Path, session_id: &str) -> bool {
    discover::read_session_file(root, session_id, "input_map.csv").is_some()
        || discover::read_session_file(root, session_id, "spot_data.csv").is_some()
        || discover::session_directory(root, session_id)
            .join("timeslice_data_device_units.csv")
            .is_file()
}

const MAX_XML_WALK_DEPTH: usize = 32;

fn list_xml(root: &Path) -> Result<Vec<String>, String> {
    let mut files = Vec::new();
    walk_xml(root, root, 0, &mut files)?;
    files.sort();
    Ok(files)
}

fn walk_xml(root: &Path, dir: &Path, depth: usize, files: &mut Vec<String>) -> Result<(), String> {
    if depth > MAX_XML_WALK_DEPTH {
        // is_dir() follows symlinks — a config tree could carry a loop.
        return Err(format!(
            "config tree is deeper than {} levels under {}",
            MAX_XML_WALK_DEPTH,
            root.display()
        ));
    }
    for entry in fs::read_dir(dir).map_err(|err| err.to_string())? {
        let entry = entry.map_err(|err| err.to_string())?;
        let path = entry.path();
        if path.is_dir() {
            walk_xml(root, &path, depth + 1, files)?;
            continue;
        }
        let name = entry.file_name().to_string_lossy().to_ascii_lowercase();
        if name.ends_with(".xml") && !name.ends_with(".xml.md5") {
            files.push(relative(root, &path));
        }
    }
    Ok(())
}

fn commit_integrity(root: &Path) -> Result<usize, String> {
    let files = list_xml(root)?;
    for rel in &files {
        commit_file(&under(root, rel)?)?;
    }
    Ok(files.len())
}

fn commit_file(path: &Path) -> Result<(), String> {
    let bytes = fs::read(path).map_err(|err| err.to_string())?;
    let salt = random_salt();
    let digest = hex_digest(&bytes, salt);
    let mtime = pyramid_mtime(path)?;
    let sidecar = sidecar_path(path);
    fs::write(&sidecar, pack_sidecar(&digest, salt, mtime)?)
        .map_err(|err| format!("Could not write {}: {err}", sidecar.display()))
}

fn integrity_value(path: &Path) -> Value {
    let (status, label) = verify_file(path);
    json!({ "status": status, "label": label })
}

fn verify_file(path: &Path) -> (&'static str, String) {
    if !path.is_file() {
        return ("SOURCE_FILE_NOT_EXIST", "Data file missing".into());
    }
    let sidecar = sidecar_path(path);
    if !sidecar.is_file() {
        return ("HASH_FILE_NOT_EXIST", "No .md5 sidecar".into());
    }
    let bytes = match fs::read(&sidecar) {
        Ok(bytes) => bytes,
        Err(_) => return ("READ_ERR", "Sidecar unreadable or corrupt".into()),
    };
    let (digest, salt, stored) = match parse_sidecar(&bytes) {
        Ok(parsed) => parsed,
        Err(_) => return ("READ_ERR", "Sidecar unreadable or corrupt".into()),
    };
    let file = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(_) => return ("OPEN_ERR", "Open error".into()),
    };
    if hex_digest(&file, salt) != digest {
        return ("HASH_ERR", "Digest mismatch".into());
    }
    match pyramid_mtime(path) {
        Ok(actual) if actual == stored => ("OK", "OK — sidecar matches file".into()),
        Ok(_) => ("TIMESTAMP_ERR", "Timestamp mismatch".into()),
        Err(_) => ("TIMESTAMP_ERR", "Timestamp mismatch".into()),
    }
}

fn hex_digest(bytes: &[u8], salt: i32) -> String {
    let mut hasher = Md5::new();
    hasher.update(bytes);
    hasher.update(salt.to_string().as_bytes());
    format!("{:x}", hasher.finalize())
}

fn pack_sidecar(digest: &str, salt: i32, mtime: i64) -> Result<Vec<u8>, String> {
    if digest.len() != 32 {
        return Err(format!("digest must be 32 hex chars, got {}", digest.len()));
    }
    let mut out = Vec::with_capacity(52);
    out.extend_from_slice(&32u64.to_le_bytes());
    out.extend_from_slice(&(salt as u32).to_le_bytes());
    out.extend_from_slice(digest.as_bytes());
    out.extend_from_slice(&mtime.to_le_bytes());
    Ok(out)
}

fn parse_sidecar(data: &[u8]) -> Result<(String, i32, i64), String> {
    if data.len() < 52 {
        return Err(format!("sidecar too short ({} bytes)", data.len()));
    }
    let hex_len = u64::from_le_bytes(data[0..8].try_into().unwrap());
    if hex_len > 1024 {
        return Err(format!("implausible digest length {hex_len}"));
    }
    let salt = i32::from_le_bytes(data[8..12].try_into().unwrap());
    let end = 12 + hex_len as usize;
    if data.len() < end + 8 {
        return Err("sidecar truncated".into());
    }
    let digest = std::str::from_utf8(&data[12..end])
        .map_err(|_| "digest is not ASCII".to_owned())?
        .to_owned();
    let mtime = i64::from_le_bytes(data[end..end + 8].try_into().unwrap());
    Ok((digest, salt, mtime))
}

fn pyramid_mtime(path: &Path) -> Result<i64, String> {
    let modified = fs::metadata(path)
        .and_then(|meta| meta.modified())
        .map_err(|err| err.to_string())?;
    let unix = modified
        .duration_since(UNIX_EPOCH)
        .map_err(|err| err.to_string())?
        .as_secs() as i64;
    let naive = chrono::DateTime::from_timestamp(unix, 0)
        .ok_or_else(|| "mtime is out of range".to_owned())?
        .naive_utc();
    match Local.from_local_datetime(&naive) {
        chrono::LocalResult::Single(local) => Ok(local.timestamp()),
        chrono::LocalResult::Ambiguous(earliest, _) => Ok(earliest.timestamp()),
        chrono::LocalResult::None => Ok(unix),
    }
}

fn random_salt() -> i32 {
    static TICK: AtomicU64 = AtomicU64::new(1);
    let tick = TICK.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos() as u64)
        .unwrap_or(0);
    let mixed = nanos ^ tick.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (mixed as u32 & 0x7fff_ffff) as i32
}

fn sidecar_path(path: &Path) -> PathBuf {
    PathBuf::from(format!("{}{}", path.display(), ".md5"))
}

fn relative(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .components()
        .map(|component| component.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/")
}

fn under(root: &Path, rel: &str) -> Result<PathBuf, String> {
    let path = Path::new(rel);
    if path.is_absolute()
        || path
            .components()
            .any(|component| matches!(component, Component::ParentDir))
    {
        return Err(format!("config path escapes the folder: {rel}"));
    }
    Ok(root.join(path))
}

fn folder_key(path: &Path) -> String {
    let normalized = if path.exists() {
        path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
    } else if let (Some(parent), Some(name)) = (path.parent(), path.file_name()) {
        parent
            .canonicalize()
            .unwrap_or_else(|_| parent.to_path_buf())
            .join(name)
    } else {
        path.to_path_buf()
    };
    normalized
        .to_string_lossy()
        .replace('\\', "/")
        .trim_end_matches('/')
        .to_owned()
}

fn copy_tree(source: &Path, dest: &Path, depth: usize) -> Result<(), String> {
    if depth > MAX_XML_WALK_DEPTH {
        // is_dir() follows symlinks — a config tree could carry a loop.
        return Err(format!(
            "config tree is deeper than {} levels under {}",
            MAX_XML_WALK_DEPTH,
            source.display()
        ));
    }
    fs::create_dir_all(dest).map_err(|err| err.to_string())?;
    for entry in fs::read_dir(source).map_err(|err| err.to_string())? {
        let entry = entry.map_err(|err| err.to_string())?;
        let from = entry.path();
        let to = dest.join(entry.file_name());
        if from.is_dir() {
            copy_tree(&from, &to, depth + 1)?;
        } else {
            fs::copy(&from, &to)
                .map_err(|err| format!("Could not copy {}: {err}", from.display()))?;
        }
    }
    Ok(())
}

fn text<'a>(input: &'a Value, key: &str) -> Result<&'a str, String> {
    input
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{key} is required"))
}

fn strings(input: &Value, key: &str) -> Vec<String> {
    input
        .get(key)
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(str::trim)
                .filter(|item| !item.is_empty())
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_sidecar_round_trip_is_52_bytes() {
        let packed = pack_sidecar(&"a".repeat(32), 12345, 0x6A1F_D124).unwrap();
        assert_eq!(packed.len(), 52);
        let (digest, salt, mtime) = parse_sidecar(&packed).unwrap();
        assert_eq!(digest, "a".repeat(32));
        assert_eq!(salt, 12345);
        assert_eq!(mtime, 0x6A1F_D124);
    }

    #[test]
    fn commit_then_verify_matches_the_file() {
        let dir = std::env::temp_dir().join(format!("scan-kit-md5-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let xml = dir.join("devices.xml");
        fs::write(&xml, b"<devices/>\n").unwrap();
        commit_file(&xml).unwrap();
        assert_eq!(verify_file(&xml).0, "OK");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn saving_a_folder_copies_the_tree_and_refreshes_sidecars() {
        let root = std::env::temp_dir().join(format!("scan-kit-cfg-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let source = root.join("source");
        let dest = root.join("dest");
        fs::create_dir_all(source.join("map2map")).unwrap();
        fs::write(
            source.join("map2map/devices.xml"),
            b"<devices><keep/></devices>\n",
        )
        .unwrap();
        fs::write(source.join("map2map/other.xml"), b"<other/>\n").unwrap();
        let db = root.join("test.sqlite");
        let saved = save_folder(
            &db,
            &source,
            &dest,
            &json!([{ "rel": "map2map/devices.xml", "xml": "<devices><edited/></devices>\n" }]),
            Some(true),
        )
        .unwrap();
        let written = fs::read_to_string(dest.join("map2map/devices.xml")).unwrap();
        assert!(written.contains("<edited"));
        assert!(dest.join("map2map/other.xml").is_file());
        assert!(dest.join("map2map/devices.xml.md5").is_file());
        assert!(dest.join("map2map/other.xml.md5").is_file());
        assert_eq!(verify_file(&dest.join("map2map/devices.xml")).0, "OK");
        assert_eq!(saved["sidecars"], 2);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_sigma_tune_uses_spot_rows_from_the_session_folder() {
        let root = std::env::temp_dir().join(format!("scan-kit-tune-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let session = root.join("sess");
        fs::create_dir_all(&session).unwrap();
        fs::write(
            session.join("input_map.csv"),
            "energy,charge_req,x_position,y_position\n100,0.02,0,0\n100,0.02,0,0\n100,0.02,0,0\n",
        )
        .unwrap();
        fs::write(
            session.join("spot_data.csv"),
            "ic1_total_dose_spot,r_ic1_x_spot_sigma\n0.02,2\n0.02,2.5\n0.02,3\n",
        )
        .unwrap();
        let xml = r#"<devices><ion_chamber><device name="IC_1_X"/><beam_sigma_conversions in_units="MEV" out_units="mm" min_energy="70" max_energy="250" K0="1" K1="0" K2="0" K3="0"/></ion_chamber></devices>"#;
        let db = root.join("unused.sqlite");
        let result = tune(
            &json!({
                "workflow": "sigma_tuning",
                "xml": xml,
                "data_dir": root.to_string_lossy(),
                "session_ids": ["sess"],
                "params": {}
            }),
            &db,
        )
        .unwrap();
        assert_eq!(result["changed"], true);
        assert!(result["summary"].as_str().unwrap().contains("from sess"));
        assert!(!result["xml"].as_str().unwrap().contains("K0=\"1\""));
        let cells = preview_row(&result, "IC_1", "70").as_array().unwrap();
        // K0 rewrites (change); the new band holds 2 of 3 samples (the largest
        // escapes past the top edge), up from 0 of 3 — improved but below
        // 100%, so warn.
        assert_eq!(cells[4]["tone"], "change");
        assert_eq!(cells[5]["text"], "0.0%");
        assert_eq!(cells[5]["tone"], "bad");
        assert_eq!(cells[6]["text"], "66.7%");
        assert_eq!(cells[6]["tone"], "warn");
        assert!(cells[3].get("tone").is_none());
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn sigma_tuning_prefers_raw_spot_sigma() {
        let root = std::env::temp_dir().join(format!("scan-kit-tune-raw-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let session = root.join("sess");
        fs::create_dir_all(&session).unwrap();
        fs::write(
            session.join("input_map.csv"),
            "energy,charge_req,x_position,y_position\n100,0.02,0,0\n100,0.02,10,0\n100,0.02,-10,0\n",
        )
        .unwrap();
        fs::write(
            session.join("spot_data.csv"),
            "ic1_total_dose_spot,r_ic1_x_spot_sigma,r_ic1_x_spot_sigma_raw\n0.02,4,1.4\n0.02,5,1.5\n0.02,6,1.6\n",
        )
        .unwrap();
        let xml = r#"<devices><ion_chamber><device name="IC_1_X"/><beam_sigma_conversions in_units="MEV" out_units="mm" min_energy="70" max_energy="250" K0="1" K1="0" K2="0" K3="0"/></ion_chamber></devices>"#;
        let result = tune(
            &json!({
                "workflow": "sigma_tuning",
                "xml": xml,
                "data_dir": root.to_string_lossy(),
                "session_ids": ["sess"],
                "params": {}
            }),
            &root.join("unused.sqlite"),
        )
        .unwrap();
        let after = preview_number(&result, "IC_1", "70", 4);
        // raw 1.4 mm ×2 = 2.8 min; 1% headroom below it, then the ±20% band.
        assert!(
            (after - 3.465).abs() < 0.02,
            "K0 {after} followed the processed sigma column"
        );
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn session_1093436476_tunes_its_map2map_devices() {
        let logs = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_data/logs");
        let devices = logs.join("1093436476/1093436476/config/map2map/devices.xml");
        if !devices.is_file() {
            return;
        }
        let db = std::env::temp_dir().join(format!(
            "scan-kit-tune-1093436476-{}.sqlite",
            std::process::id()
        ));
        let _ = fs::remove_file(&db);
        let opened =
            open_folder(&db, None, Some(&logs.to_string_lossy()), Some("1093436476")).unwrap();
        let path = opened["path"].as_str().unwrap().replace('\\', "/");
        assert!(
            path.ends_with("1093436476/1093436476/config/map2map"),
            "{path}"
        );
        assert!(opened["files"]
            .as_array()
            .unwrap()
            .iter()
            .any(|file| file.as_str() == Some("devices.xml")));

        let xml = fs::read_to_string(&devices).unwrap();
        let form = config_form(&xml).unwrap();
        assert!(form_has(&form, "IC 1 X", true));

        let sigma = tune(
            &json!({
                "workflow": "sigma_tuning",
                "xml": &xml,
                "data_dir": logs.to_string_lossy(),
                "session_ids": ["1093436476"],
                "params": {}
            }),
            &db,
        )
        .unwrap();
        assert_eq!(sigma["changed"], true, "{sigma}");
        let k0 = preview_number(&sigma, "IC_1", "179.5", 4);
        assert!(
            (2.0..6.0).contains(&k0),
            "180 MeV K0 {k0} is outside the raw-sigma band"
        );

        for workflow in ["position_offset_tuning", "ic_distance_tuning", "kmu_tuning"] {
            let result = tune(
                &json!({
                    "workflow": workflow,
                    "xml": &xml,
                    "data_dir": logs.to_string_lossy(),
                    "session_ids": ["1093436476"],
                    "params": {}
                }),
                &db,
            )
            .unwrap_or_else(|err| panic!("{workflow}: {err}"));
            assert_eq!(result["changed"], true, "{workflow}: {}", result["summary"]);
        }
        let _ = fs::remove_file(&db);
    }

    #[test]
    fn known_fixture_digests_match_when_the_files_are_present() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../test_data/1943968267/1943968267/config/map2map");
        if !root.is_dir() {
            return;
        }
        let known = [
            (
                "tolerances.xml",
                "590c408d80858321baae5d070f384f85",
                0x0425_5B44_i32,
            ),
            (
                "Input.xml",
                "7c132b67956e13f28ab75081d22982a2",
                0xDD0C_1658_u32 as i32,
            ),
            (
                "Database.xml",
                "e0f6de7f3cc46c2a194f5bc7b8a0b63d",
                0xDA29_3505_u32 as i32,
            ),
            (
                "devices.xml",
                "0d15ddb0df181a9696b1c3a8233c8eff",
                0xC70E_F424_u32 as i32,
            ),
        ];
        for (name, digest, salt) in known {
            let path = root.join(name);
            if !path.is_file() {
                continue;
            }
            let bytes = fs::read(sidecar_path(&path)).unwrap();
            let (found, found_salt, _) = parse_sidecar(&bytes).unwrap();
            assert_eq!(found, digest, "{name}");
            assert_eq!(found_salt, salt, "{name}");
            let file = fs::read(&path).unwrap();
            assert_eq!(hex_digest(&file, found_salt), digest, "{name}");
        }
    }

    #[test]
    fn a_config_diff_covers_the_whole_config_tree() {
        let root = std::env::temp_dir().join(format!("scan-kit-diff-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let loaded = root.join("loaded/config/map2map");
        fs::create_dir_all(&loaded).unwrap();
        fs::write(loaded.join("devices.xml"), "<devices><a/></devices>\n").unwrap();
        fs::write(loaded.join("devices.xml.md5"), b"sidecar").unwrap();
        fs::write(root.join("loaded/config/extra.xml"), "<extra/>\n").unwrap();
        fs::create_dir_all(root.join("loaded/config/nozzle")).unwrap();
        fs::write(root.join("loaded/config/nozzle/n.json"), "{\"a\":1}\n").unwrap();
        let session = root.join("sess");
        fs::create_dir_all(session.join("config/dcs")).unwrap();
        fs::create_dir_all(session.join("config/map2map")).unwrap();
        fs::create_dir_all(session.join("config/nozzle")).unwrap();
        fs::write(session.join("input_map.csv"), "energy\n100\n").unwrap();
        fs::write(
            session.join("config/map2map/devices.xml"),
            "<devices><b/></devices>\n",
        )
        .unwrap();
        fs::write(session.join("config/dcs/x.xml"), "<x/>\n").unwrap();
        fs::write(session.join("config/nozzle/n.json"), "{\"a\":1}\n").unwrap();

        let result = diff(&json!({
            "config_dir": loaded.to_string_lossy(),
            "file": "devices.xml",
            "xml": "<devices><edited/></devices>\n",
            "data_dir": root.to_string_lossy(),
            "session_ids": ["sess"],
        }))
        .unwrap();

        assert!(result["base"]
            .as_str()
            .unwrap()
            .replace('\\', "/")
            .ends_with("loaded/config"));
        assert_eq!(result["edited"], "map2map/devices.xml");
        let files = result["sessions"][0]["files"].as_array().unwrap();
        let status = |rel: &str| {
            files
                .iter()
                .find(|file| file["rel"] == rel)
                .unwrap_or_else(|| panic!("no {rel} in {files:?}"))
        };
        assert_eq!(status("map2map/devices.xml")["status"], "changed");
        assert_eq!(status("map2map/devices.xml")["edited"], true);
        assert!(status("map2map/devices.xml")["lines"]
            .as_array()
            .unwrap()
            .iter()
            .any(|line| line["kind"] == "del" && line["text"] == "<devices><edited/></devices>"));
        assert_eq!(status("dcs/x.xml")["status"], "added");
        assert_eq!(status("extra.xml")["status"], "removed");
        assert_eq!(status("nozzle/n.json")["status"], "same");
        assert!(files
            .iter()
            .all(|file| file["rel"] != "map2map/devices.xml.md5"));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_missing_session_config_is_a_warning_not_an_error() {
        let root = std::env::temp_dir().join(format!("scan-kit-diff-miss-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let loaded = root.join("loaded/map2map");
        fs::create_dir_all(&loaded).unwrap();
        fs::write(loaded.join("devices.xml"), "<devices/>\n").unwrap();
        let session = root.join("sess");
        fs::create_dir_all(&session).unwrap();
        fs::write(session.join("input_map.csv"), "energy\n100\n").unwrap();

        let result = diff(&json!({
            "config_dir": loaded.to_string_lossy(),
            "file": "",
            "data_dir": root.to_string_lossy(),
            "session_ids": ["sess"],
        }))
        .unwrap();
        assert!(result["sessions"][0]["warning"]
            .as_str()
            .unwrap()
            .contains("No config/"));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_diff_against_the_owning_session_is_flagged_self() {
        let root = std::env::temp_dir().join(format!("scan-kit-diff-self-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let loaded = root.join("sess/config/map2map");
        fs::create_dir_all(&loaded).unwrap();
        fs::write(loaded.join("devices.xml"), "<devices/>\n").unwrap();
        fs::write(root.join("sess/input_map.csv"), "energy\n100\n").unwrap();

        // The desktop always sends the open file's text; identical content
        // must not defeat the self check.
        let result = diff(&json!({
            "config_dir": loaded.to_string_lossy(),
            "file": "devices.xml",
            "xml": "<devices/>\n",
            "data_dir": root.to_string_lossy(),
            "session_ids": ["sess"],
        }))
        .unwrap();
        assert_eq!(result["sessions"][0]["self"], true);
        assert!(result["sessions"][0]["files"]
            .as_array()
            .unwrap()
            .is_empty());
        let _ = fs::remove_dir_all(&root);
    }

    fn preview_row<'a>(result: &'a Value, device: &str, energy: &str) -> &'a Value {
        let rows = result["rows"].as_array().expect("preview rows");
        rows.iter()
            .find(|row| {
                let cells = row.as_array().expect("row");
                cells.first().and_then(|cell| cell["text"].as_str()) == Some(device)
                    && cells
                        .get(1)
                        .and_then(|cell| cell["text"].as_str())
                        .is_some_and(|text| text.starts_with(energy))
            })
            .unwrap_or_else(|| panic!("no {device} row starting {energy} in {result}"))
    }

    fn preview_number(result: &Value, device: &str, energy: &str, column: usize) -> f64 {
        preview_row(result, device, energy).as_array().unwrap()[column]["text"]
            .as_str()
            .unwrap()
            .split(' ')
            .next()
            .unwrap()
            .parse()
            .unwrap()
    }

    fn form_has(node: &Value, title: &str, collapsible: bool) -> bool {
        let titled = node.get("title").and_then(Value::as_str) == Some(title);
        let folds = node.get("collapsible").and_then(Value::as_bool) == Some(true);
        if titled && (!collapsible || folds) {
            return true;
        }
        node.get("nodes")
            .and_then(Value::as_array)
            .is_some_and(|nodes| {
                nodes
                    .iter()
                    .any(|child| form_has(child, title, collapsible))
            })
    }
}
