//! Capture a live IGX run into a G3-compatible session directory.
//!
//! Current firmware keeps the run in two tables per device, both served as CSV
//! by `GET /io/<dev>/control_point_table/<name>/data.csv`:
//!
//! - `table` — "Control Points": one row per spot with plan limits and measured
//!   results. Becomes `layer-N/run-M/<DEV>_spot_data.csv`; the RCI copy is also
//!   `RCI_map.csv`.
//! - `timeslice_table` — per-timeslice channels. Becomes the wide
//!   `layer-N/run-M/timeslice_data_device_units.csv` joined on `timesliceNumber`.
//!
//! There are no on-disk session folders to enumerate; the session id is read
//! from `timeslice_table/file_header_meta` (`"<id>|…"`, shared by all devices).
//! Sub-devices are discovered from `rci/map_manager/device_*_client` nodes and
//! fetched over their own HTTP addresses.
//!
//! Emitted headers use bare G3 names — the device `(unit)` suffixes are
//! display metadata, not part of the column name (`r_ic1_x(ch)` is written as
//! `r_ic1_x_position`, `ic3_current_A(nA)` as `ic3_current_A`). Flag cells come
//! out as `1`/`0`, never `true`/`false`.
//!
//! ponytail: `timeslice_data.csv` and the iso-mm / converted `spot_data.csv`
//! columns are emitted raw only — this firmware has no map2map XML on-device.
//! Upgrade path: accept a local `devices.xml`/`scan_dose_system.xml` bundle, or
//! port `load_map2map_geometry` + the affine fallback from the Python
//! `g3_session` module.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Duration;

use md5::{Digest, Md5};
use serde_json::{json, Map, Value};

use scan_kit_core::{device_file_url, io_number, io_url, CONTROL_POINT_COUNT, TIME_ELAPSED};

use crate::{runner, tables};

/// Per-device meta columns that lead every G3 table block.
const META: [&str; 4] = ["spot_no", "layer_id", "timesliceNumber", "timestamp"];

/// `/root/config/*.json` storage files copied into `config/nozzle/<DEV>/`.
const NOZZLE_FILES: [&str; 8] = [
    "accumulation.json",
    "admin.json",
    "config.json",
    "hourmeter.json",
    "interlocks.json",
    "network.json",
    "prescription.json",
    "system_admin.json",
];

/// A device that holds session tables: the RCI plus each map-manager client.
pub struct CaptureDevice {
    /// Uppercase file prefix, e.g. `IX256_1`, `FX4`, `RCI`.
    pub name: String,
    /// IO subtree on the device, e.g. `ix256`, `rci`.
    pub io_root: String,
    /// HTTP host[:port] that serves this device.
    pub host: String,
    /// Channel-map XML on the RCI, e.g. `/root/config/control/IX256_1.xml`.
    pub config_target: String,
}

/// One parsed `data.csv` export: header cells in order plus rows.
#[derive(Clone)]
pub struct Sheet {
    pub header: Vec<String>,
    pub rows: Vec<Vec<String>>,
    /// A mid-file CSV error ended the parse — the kept prefix is usable but
    /// the table is incomplete, so callers should surface it.
    pub truncated: bool,
}

impl Sheet {
    /// `data.csv` headers carry `(unit)` suffixes (`timestamp(ms)`); look up by
    /// the bare column name.
    fn column(&self, name: &str) -> Option<usize> {
        self.header.iter().position(|cell| base_name(cell) == name)
    }
}

fn base_name(name: &str) -> &str {
    name.split('(').next().unwrap_or(name).trim_end()
}

/// A contiguous run of equal `layer_id` values inside one table.
struct Block {
    layer: usize,
    run: usize,
    start: usize,
    end: usize,
}

#[derive(Debug)]
pub struct CaptureResult {
    pub files: usize,
    pub warnings: Vec<String>,
    pub session_id: String,
}

/// First `|` field of `file_header_meta` — device text, so strip control
/// characters before it lands in termination_summary where a newline would
/// forge a record line.
fn meta_id(meta: &str) -> String {
    meta.split('|')
        .next()
        .unwrap_or("")
        .chars()
        .filter(|ch| !ch.is_control())
        .take(64)
        .collect::<String>()
        .trim()
        .to_owned()
}

/// Read the current session id from the RCI's timeslice table metadata.
/// `None` when the device has never run a session.
/// `live_session_id` on a bounded deadline — for the status poll, where an
/// unresponsive file server must not freeze the health check behind it.
pub fn live_session_id_timed(host: &str, timeout: Duration) -> Option<String> {
    runner::http_json_timed(
        host,
        "rci/control_point_table/timeslice_table/file_header_meta",
        timeout,
    )
    .and_then(|value| value.as_str().map(str::to_owned))
    .map(|meta| meta_id(&meta))
    .filter(|id| !id.is_empty())
}

pub fn live_session_id(host: &str) -> Option<String> {
    table_io(host, "rci", "timeslice_table", "file_header_meta")
        .and_then(|value| value.as_str().map(str::to_owned))
        .map(|meta| meta_id(&meta))
        .filter(|id| !id.is_empty())
}

/// Enumerate the RCI plus every map-manager device client that has an address.
fn devices(
    rci_host: &str,
    warnings: &mut Vec<String>,
    keep: &mut dyn FnMut(u64, u64) -> bool,
) -> Result<Vec<CaptureDevice>, String> {
    let mut out = vec![CaptureDevice {
        name: "RCI".into(),
        io_root: "rci".into(),
        host: rci_host.to_owned(),
        config_target: "/root/config/control/RCI.xml".into(),
    }];
    let Some(children) = runner::http_json_file(rci_host, "rci/map_manager", "children.json")
    else {
        warnings.push("map_manager did not list device clients".into());
        return Ok(out);
    };
    let mut names: Vec<&str> = children
        .as_object()
        .map(|map| {
            map.keys()
                .map(String::as_str)
                .filter(|key| key.starts_with("device_") && key.ends_with("_client"))
                .collect()
        })
        .unwrap_or_default();
    names.sort_unstable();
    // The device list is remote-supplied and each client costs ~13 HTTP
    // requests — cap it so a malfunctioning RCI can't turn a download into
    // hours of sequential fetches. A real controller has a handful.
    const MAX_DEVICES: usize = 64;
    if names.len() > MAX_DEVICES {
        warnings.push(format!(
            "map_manager listed {} device clients — capturing the first {MAX_DEVICES}",
            names.len()
        ));
        names.truncate(MAX_DEVICES);
    }
    for client in names {
        let base = format!("rci/map_manager/{client}");
        let field = |leaf: &str| {
            runner::http_json_file(rci_host, &format!("{base}/{leaf}"), "value.json")
                .and_then(|value| value.as_str().map(str::to_owned))
                .unwrap_or_default()
        };
        let io_root = field("device_path").trim_matches('/').to_owned();
        let host = field("websocket_client/server_ip_address");
        tick(0, keep)?;
        if io_root.is_empty() || host.trim().is_empty() {
            warnings.push(format!("{client} has no device address — skipped"));
            continue;
        }
        // The device map is remote-supplied: its host drives real fetches
        // and its path lands in the request line. An IP literal only, and
        // no .. or query steering — a hostile map must not proxy internal
        // reads into the session zip.
        let host_ip = host
            .trim()
            .rsplit_once(':')
            .map(|(head, _)| head)
            .unwrap_or_else(|| host.trim());
        // A literal still permits steering reads at hosts that can never
        // be a device file server: link-local (the 169.254.169.254-style
        // metadata endpoint), unspecified, or multicast. Loopback is left
        // alone — co-located device software serves from it legitimately.
        let steering = host_ip
            .parse::<std::net::IpAddr>()
            .map(|ip| {
                ip.is_unspecified()
                    || ip.is_multicast()
                    || match ip {
                        std::net::IpAddr::V4(ip) => ip.is_link_local(),
                        std::net::IpAddr::V6(ip) => (ip.segments()[0] & 0xffc0) == 0xfe80,
                    }
            })
            .unwrap_or(true);
        if steering
            || io_root.split('/').any(|seg| seg == "..")
            || io_root.contains(['?', '#'])
            || io_root.contains('%')
        {
            warnings.push(format!(
                "{client} has an untrusted device address — skipped"
            ));
            continue;
        }
        // The name lands in capture file paths (x_spot_data.csv, x.log, the
        // nozzle folder) — a device client key is untrusted text, so keep it
        // to filename characters only.
        let name = client_name(client);
        if name.is_empty() {
            warnings.push(format!("{client} has no usable device name — skipped"));
            continue;
        }
        if out.iter().any(|device: &CaptureDevice| device.name == name) {
            // A second device writing the same filenames would clobber the
            // first's captures — skip it rather than lose the data silently.
            warnings.push(format!(
                "{client} collides on capture name '{name}' — skipped"
            ));
            continue;
        }
        // config_target is fetched from the RCI itself — a poisoned
        // target just loses that one file, not the whole device.
        let config_target = field("configuration_upload/target");
        let config_target = if config_target.split('/').any(|seg| seg == "..")
            || config_target.contains(['?', '#'])
            || config_target.contains('%')
        {
            String::new()
        } else {
            config_target
        };
        out.push(CaptureDevice {
            name,
            io_root,
            host,
            config_target,
        });
    }
    Ok(out)
}

/// Device client key → filename-safe name (`device_ic1_client` → `IC1`).
fn dos_reserved(name: &str) -> bool {
    matches!(
        name.split('.')
            .next()
            .unwrap_or("")
            .to_ascii_uppercase()
            .as_str(),
        "CON"
            | "PRN"
            | "AUX"
            | "NUL"
            | "COM1"
            | "COM2"
            | "COM3"
            | "COM4"
            | "COM5"
            | "COM6"
            | "COM7"
            | "COM8"
            | "COM9"
            | "LPT1"
            | "LPT2"
            | "LPT3"
            | "LPT4"
            | "LPT5"
            | "LPT6"
            | "LPT7"
            | "LPT8"
            | "LPT9"
    )
}

fn client_name(client: &str) -> String {
    let stripped = client.strip_prefix("device_").unwrap_or(client);
    let name: String = stripped
        .strip_suffix("_client")
        .unwrap_or(stripped)
        .to_ascii_uppercase()
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric() || *ch == '_' || *ch == '-')
        .take(64)
        .collect();
    // A bare DOS device stem (`CON.log`, `NUL_spot_data.csv`) fails every
    // write on Windows — the charset filter alone lets it through.
    if dos_reserved(&name) {
        format!("DEVICE_{name}")
    } else {
        name
    }
}

fn table_io(host: &str, io_root: &str, table: &str, io: &str) -> Option<Value> {
    runner::http_json_file(
        host,
        &format!("{io_root}/control_point_table/{table}/{io}"),
        "value.json",
    )
}

/// Fetch one table as CSV and parse it. Missing/empty tables are `Ok(None)`;
/// a transport or URL failure is `Err` so callers can say "could not fetch"
/// instead of reporting data as absent.
fn fetch_sheet(host: &str, io_root: &str, table: &str) -> Result<Option<Sheet>, String> {
    let url = io_url(
        host,
        &format!("{io_root}/control_point_table/{table}"),
        "data.csv",
    )?;
    let bytes = runner::http(&url, "GET", None)?;
    Ok(parse_sheet(&bytes))
}

fn parse_sheet(bytes: &[u8]) -> Option<Sheet> {
    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .comment(Some(b'#'))
        .from_reader(bytes);
    let header: Vec<String> = reader
        .headers()
        .ok()?
        .iter()
        .enumerate()
        .map(|(index, name)| {
            // A UTF-8 BOM is not whitespace — a BOM'd first header would
            // be unreachable by name everywhere downstream.
            let name = name.trim_start_matches('\u{feff}').trim();
            if name.is_empty() {
                format!("unnamed_{index}")
            } else {
                name.to_owned()
            }
        })
        .collect();
    let mut rows = Vec::new();
    // A close-delimited reply has no length to check - a body that does not
    // end in a newline was cut mid-record even though the parse is clean.
    let mut truncated = !bytes.is_empty() && !matches!(bytes.last(), Some(&10) | Some(&13));
    for record in reader.records() {
        let Ok(record) = record else {
            truncated = true;
            break;
        };
        let cells = record
            .iter()
            .map(|cell| cell.trim().to_owned())
            .collect::<Vec<_>>();
        // A repeated header line is a banner, not a data row — keeping it
        // keys phantom "layer_id"/"spot_no" rows in every downstream join.
        let repeated = cells.len() == header.len()
            && cells
                .iter()
                .zip(header.iter())
                .all(|(cell, name)| cell.eq_ignore_ascii_case(name));
        if !repeated {
            rows.push(cells);
        }
    }
    if rows.is_empty() {
        None
    } else {
        Some(Sheet {
            header,
            rows,
            truncated,
        })
    }
}

/// Split a table into contiguous `layer_id` blocks. `layer` is the first-seen
/// order of each distinct id (delivery order); `run` counts how many times the
/// same id has been seen so far.
///
/// ponytail: an interleaved layer sequence becomes extra runs — real
/// deliveries keep layers contiguous.
// `layer_of` is shared across every sheet in one capture: the layer_id ->
// folder ordinal map must be global, or a device whose table is missing the
// first layer mis-files its rows into layer-0.
fn blocks(sheet: &Sheet, layer_of: &mut HashMap<String, usize>) -> Vec<Block> {
    let Some(column) = sheet.column("layer_id") else {
        return Vec::new();
    };
    let mut runs_of: HashMap<String, usize> = HashMap::new();
    let mut out = Vec::new();
    let mut start = 0usize;
    // prev_eff is the open run's effective id — a ragged row inherits it
    // rather than the previous row's raw (possibly absent) cell. Ids are
    // join_key-normalized like every other layer key: "1" and "1.0" across
    // differently-formatted tables are one physical layer, not two.
    let mut prev_eff: Option<String> = None;
    for row in 0..=sheet.rows.len() {
        let cell = sheet
            .rows
            .get(row)
            .and_then(|cells| cells.get(column))
            .map(String::as_str);
        let id = if row < sheet.rows.len()
            && cell.is_none_or(|id| {
                // A repeated literal header line mid-file is not a layer —
                // inherit the open run like a ragged cell.
                let id = clean(id);
                id.is_empty() || id.eq_ignore_ascii_case("layer_id")
            }) {
            prev_eff.clone()
        } else {
            cell.map(|id| join_key(clean(id)))
        };
        if row > 0
            && (row == sheet.rows.len() || id != prev_eff)
            && (prev_eff.is_some() || row == sheet.rows.len())
        {
            // A leading ragged run has no previous id — it belongs to the
            // layer that opens it, not a phantom "" layer.
            let prev_id = prev_eff.clone().or_else(|| id.clone()).unwrap_or_default();
            let next = layer_of.len();
            let layer = *layer_of.entry(prev_id.clone()).or_insert(next);
            let run = runs_of.entry(prev_id).or_insert(0);
            out.push(Block {
                layer,
                run: *run,
                start,
                end: row,
            });
            *run += 1;
            start = row;
        }
        prev_eff = id;
    }
    out
}

fn slice(sheet: &Sheet, block: &Block) -> Sheet {
    Sheet {
        header: sheet.header.clone(),
        rows: sheet.rows[block.start..block.end].to_vec(),
        truncated: sheet.truncated,
    }
}

fn csv_text(sheet: &Sheet) -> String {
    let mut out = join_cells(sheet.header.iter().map(String::as_str));
    out.push('\n');
    for row in &sheet.rows {
        // Pad/truncate to the header width — parse_sheet keeps ragged rows
        // and writing them back verbatim misaligns every strict reader.
        let cells = (0..sheet.header.len())
            .map(|index| clean(row.get(index).map(String::as_str).unwrap_or("")))
            .collect::<Vec<_>>();
        out.push_str(&join_cells(cells.iter().copied()));
        out.push('\n');
    }
    out
}

/// Join keys arrive as raw cell text — "1" and "1.0" are the same physical
/// row across differently-formatted device tables. Numeric keys normalize;
/// anything else joins on its text.
fn join_key(text: &str) -> String {
    // Integer ids normalize exactly - f64 rounds past 2^53 and would
    // collapse two distinct ids into one key.
    if let Ok(value) = text.parse::<i64>() {
        return value.to_string();
    }
    match text.parse::<f64>() {
        // "-0" and "0"/"0.0" must join; a non-finite cell keeps its raw
        // text rather than collapsing every NaN into one key.
        Ok(value) if value.is_finite() => {
            if value == 0.0 {
                "0".to_owned()
            } else {
                value.to_string()
            }
        }
        _ => text.to_owned(),
    }
}

fn join_cells<'a>(cells: impl Iterator<Item = &'a str>) -> String {
    cells
        .map(|cell| {
            // The fast table reader is line-based — a quoted newline is valid
            // RFC CSV to it, just split into a phantom row. Flatten instead.
            let flat = cell.replace(['\r', '\n'], " ");
            if flat.contains([',', '"']) {
                format!("\"{}\"", flat.replace('"', "\"\""))
            } else {
                flat
            }
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn write_text(path: &Path, text: &str) -> Result<(), String> {
    write_bytes(path, text.as_bytes())
}

fn write_bytes(path: &Path, bytes: &[u8]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|err| err.to_string())?;
    }
    std::fs::write(path, bytes).map_err(|err| err.to_string())
}

/// The device's `ic` prefix (`ic1`, `ic3`, …) detected from its own column
/// names — the channel-map XML names the columns, so no device-name table is
/// needed. Only dose chambers carry `total_dose`, which keeps ICSIM's
/// `ic3_current_cmd` command column from mis-detecting it as IC3.
fn ic_prefix(sheet: &Sheet) -> Option<String> {
    sheet.column("total_dose")?;
    sheet.header.iter().find_map(|name| {
        let rest = name.strip_prefix("ic")?;
        let digits: String = rest.chars().take_while(|ch| ch.is_ascii_digit()).collect();
        if digits.is_empty() || !rest[digits.len()..].starts_with('_') {
            return None;
        }
        Some(format!("ic{digits}"))
    })
}

/// Download the live session into `dir` in the G3 layout.
pub fn capture(
    rci_host: &str,
    status: &Map<String, Value>,
    plan_path: Option<&Path>,
    dir: &Path,
    keep: &mut dyn FnMut(u64, u64) -> bool,
) -> Result<CaptureResult, String> {
    let mut warnings = Vec::new();
    let mut count = 0usize;

    let devices = devices(rci_host, &mut warnings, keep)?;
    let session_id = session_id(&devices, &mut warnings, keep)?
        .ok_or("no session on the controller — the devices' tables are empty")?;

    let mut spots: Vec<(&CaptureDevice, Sheet)> = Vec::new();
    let mut slices: Vec<(&CaptureDevice, Sheet)> = Vec::new();
    for device in &devices {
        match fetch_sheet(&device.host, &device.io_root, "table") {
            Ok(Some(sheet)) => {
                if sheet.truncated {
                    warnings.push(format!(
                        "{}: control-point table truncated mid-file",
                        device.name
                    ));
                }
                spots.push((device, sheet));
            }
            Ok(None) => warnings.push(format!("{}: no control-point table", device.name)),
            Err(err) => warnings.push(format!(
                "{}: could not fetch the control-point table: {err}",
                device.name
            )),
        }
        tick(count, keep)?;
        match fetch_sheet(&device.host, &device.io_root, "timeslice_table") {
            Ok(Some(sheet)) => {
                if sheet.truncated {
                    warnings.push(format!(
                        "{}: timeslice table truncated mid-file",
                        device.name
                    ));
                }
                slices.push((device, sheet));
            }
            Ok(None) => warnings.push(format!("{}: no timeslice table", device.name)),
            Err(err) => warnings.push(format!(
                "{}: could not fetch the timeslice table: {err}",
                device.name
            )),
        }
        tick(count, keep)?;
    }
    if spots.is_empty() && slices.is_empty() {
        return Err(format!(
            "no session tables found on {} for session {session_id}",
            rci_host
        ));
    }
    // A run boundary between the meta read and the table fetches would
    // label session B's CSVs with session A's id — re-check.
    let mut scratch = Vec::new();
    match self::session_id(&devices, &mut scratch, keep) {
        Ok(Some(later)) if later != session_id => warnings.push(format!(
            "session rolled from {session_id} to {later} mid-capture — the tables may be mixed"
        )),
        Ok(_) => {}
        Err(err) => return Err(err),
    }

    // Per layer/run folder: each device's spot rows (RCI's also as RCI_map.csv)
    // plus the wide timeslice join.
    let mut hits: BTreeMap<
        (usize, usize),
        (Vec<(&CaptureDevice, Sheet)>, Vec<(&CaptureDevice, Sheet)>),
    > = BTreeMap::new();
    let mut layer_of: HashMap<String, usize> = HashMap::new();
    for (device, sheet) in &spots {
        for block in blocks(sheet, &mut layer_of) {
            hits.entry((block.layer, block.run))
                .or_default()
                .0
                .push((*device, slice(sheet, &block)));
        }
    }
    for (device, sheet) in &slices {
        for block in blocks(sheet, &mut layer_of) {
            hits.entry((block.layer, block.run))
                .or_default()
                .1
                .push((*device, slice(sheet, &block)));
        }
    }
    for ((layer, run), (spot_parts, ts_parts)) in &hits {
        let run_dir = dir
            .join(format!("layer-{layer}"))
            .join(format!("run-{run}"));
        for (device, sheet) in spot_parts {
            write_text(
                &run_dir.join(format!("{}_spot_data.csv", device.name)),
                &csv_text(sheet),
            )?;
            if device.name == "RCI" {
                write_text(&run_dir.join("RCI_map.csv"), &csv_text(sheet))?;
                count += 1;
            }
            count += 1;
        }
        if let Some(text) = join_timeslices(ts_parts, &mut warnings) {
            write_text(&run_dir.join("timeslice_data_device_units.csv"), &text)?;
            count += 1;
        }
        tick(count, keep)?;
    }

    if let Some(text) = join_spot_data(&spots, &mut warnings) {
        write_text(&dir.join("spot_data.csv"), &text)?;
        count += 1;
        tick(count, keep)?;
    }
    // The first fetched sheet is only the RCI's when its fetch succeeded —
    // a failed RCI table must not substitute another device's columns.
    let rci_spots = spots.iter().find(|(d, _)| d.name == "RCI").map(|(_, s)| s);
    count += write_input_map(rci_host, plan_path, rci_spots, dir, &mut warnings)?;
    tick(count, keep)?;

    let last = hits.keys().last().copied().unwrap_or((0, 0));
    let last_layer = spots
        .iter()
        .chain(slices.iter())
        .filter_map(|(_, sheet)| sheet.column("layer_id").map(|c| (sheet, c)))
        .find_map(|(sheet, c)| {
            sheet
                .rows
                .last()
                .and_then(|row| row.get(c))
                .map(|cell| join_key(clean(cell)))
                .filter(|cell| !cell.is_empty() && cell != "layer_id")
        })
        .unwrap_or_else(|| "0".into());
    let state = match status
        .get(scan_kit_core::COMBINED_STATE)
        .map(scan_kit_core::io_text)
        .unwrap_or_default()
        .as_str()
    {
        "" => "FnStateMapComplete".into(),
        text => format!("FnStateMap{}", title_case(text)),
    };
    write_text(
        &dir.join("session_info.json"),
        &format!(
            "{:#}\n",
            json!({
                "DCSState": state,
                "LayerID": last_layer,
                "Run": last.1.to_string(),
                "SessionID": session_id,
            })
        ),
    )?;
    count += 1;
    count += write_termination(dir, &session_id, status, &spots, &slices)?;
    count += write_session_meta(dir, &session_id, status, &devices)?;
    count += fetch_aux(rci_host, &devices, dir, keep)?;
    // The aux phase runs ~13 GETs per device after the last recheck - the
    // logs it lands are the worst mixed-session artifact.
    let mut scratch = Vec::new();
    match self::session_id(&devices, &mut scratch, keep) {
        Ok(Some(later)) if later != session_id => warnings.push(format!(
            "session rolled from {session_id} to {later} during aux fetches — logs may be mixed"
        )),
        Ok(_) => {}
        Err(err) => return Err(err),
    }
    Ok(CaptureResult {
        files: count,
        warnings,
        session_id,
    })
}

/// The shared session id: first `|` field of each `file_header_meta`.
/// The RCI's id wins; any disagreement is reported as a warning.
fn session_id(
    devices: &[CaptureDevice],
    warnings: &mut Vec<String>,
    keep: &mut dyn FnMut(u64, u64) -> bool,
) -> Result<Option<String>, String> {
    let mut ids = Vec::new();
    let mut rci = None;
    for (index, device) in devices.iter().enumerate() {
        tick(0, keep)?;
        let id = table_io(
            &device.host,
            &device.io_root,
            "timeslice_table",
            "file_header_meta",
        )
        .and_then(|value| value.as_str().map(str::to_owned))
        .map(|meta| meta_id(&meta))
        .unwrap_or_default();
        if id.is_empty() {
            continue;
        }
        if index == 0 {
            rci = Some(id.clone());
        }
        ids.push(id);
    }
    let unique: HashSet<&String> = ids.iter().collect();
    if unique.len() > 1 {
        let mut list: Vec<&String> = unique.into_iter().collect();
        list.sort();
        warnings.push(format!(
            "devices disagree on the session id: {}",
            list.iter()
                .map(|id| id.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    Ok(rci.or_else(|| ids.into_iter().next()))
}

fn tick(count: usize, keep: &mut dyn FnMut(u64, u64) -> bool) -> Result<(), String> {
    if keep(count as u64, 0) {
        Ok(())
    } else {
        Err("cancelled".into())
    }
}

fn title_case(text: &str) -> String {
    let mut out = String::new();
    let mut up = true;
    for ch in text.chars() {
        if ch.is_ascii_alphabetic() {
            out.push(if up { ch.to_ascii_uppercase() } else { ch });
            up = false;
        } else {
            up = true;
            out.push(ch);
        }
    }
    out
}

/// `timeslice_data_device_units.csv`: outer join of each device's slice on
/// `timesliceNumber`. G3 layout repeats `spot_no,layer_id,timesliceNumber,
/// timestamp` per block and ends each block with `_state`.
fn join_timeslices(
    parts: &[(&CaptureDevice, Sheet)],
    warnings: &mut Vec<String>,
) -> Option<String> {
    struct Part {
        header: Vec<String>,
        rows: HashMap<(String, String, String), Vec<String>>,
        order: Vec<(String, String, String)>,
    }
    let mut parts_out: Vec<(String, Part)> = Vec::new();
    for (device, sheet) in parts {
        let Some(ts_i) = sheet.column("timesliceNumber") else {
            warnings.push(format!(
                "{}: timeslice table has no timesliceNumber column",
                device.name
            ));
            continue;
        };
        let meta_idx: Vec<Option<usize>> = META.iter().map(|name| sheet.column(name)).collect();
        // An all-empty META shadow sorts ahead of another device's real
        // column - only emit what this sheet carries (spot_columns' fix).
        let meta_cols: Vec<usize> = (0..META.len())
            .filter(|index| meta_idx[*index].is_some())
            .collect();
        let keep: Vec<usize> = (0..sheet.header.len())
            .filter(|index| !META.contains(&base_name(&sheet.header[*index])))
            .collect();
        let spot_i = sheet.column("spot_no");
        let layer_i = sheet.column("layer_id");
        let mut part = Part {
            header: meta_cols
                .iter()
                .map(|index| META[*index].to_owned())
                .chain(keep.iter().map(|i| dest_name(&sheet.header[*i])))
                .collect(),
            rows: HashMap::new(),
            order: Vec::new(),
        };
        // Ragged meta cells inherit the running value like blocks() and
        // join_spot_data() — an empty layer or spot cell must not file the
        // row under "" and split it from its siblings.
        let mut eff_layer = sheet
            .rows
            .iter()
            .find_map(|row| {
                layer_i.and_then(|i| row.get(i)).and_then(|cell| {
                    let raw = clean(cell);
                    (!raw.is_empty()).then(|| raw.to_owned())
                })
            })
            .unwrap_or_default();
        let mut eff_spot = String::new();
        for row in &sheet.rows {
            let Some(raw) = row.get(ts_i) else { continue };
            let key = clean(raw);
            if key.is_empty() {
                continue;
            }
            // Key on (layer, spot, timeslice): a counter scoped narrower
            // than the whole delivery — per spot or per layer — would
            // overwrite earlier rows under a timesliceNumber-only key.
            // Missing meta columns degrade to the old single key.
            let raw_layer = layer_i
                .and_then(|i| row.get(i))
                .map(|cell| clean(cell))
                .unwrap_or_default();
            if !raw_layer.is_empty() && !raw_layer.eq_ignore_ascii_case("layer_id") {
                eff_layer = raw_layer.to_owned();
            }
            let raw_spot = spot_i
                .and_then(|i| row.get(i))
                .map(|cell| clean(cell))
                .unwrap_or_default();
            if !raw_spot.is_empty() && !raw_spot.eq_ignore_ascii_case("spot_no") {
                eff_spot = raw_spot.to_owned();
            }
            let key = (join_key(&eff_layer), join_key(&eff_spot), join_key(key));
            if !part.rows.contains_key(&key) {
                part.order.push(key.clone());
            }
            part.rows.insert(
                key.clone(),
                meta_cols
                    .iter()
                    .map(|index| {
                        meta_idx[*index]
                            .and_then(|i| row.get(i))
                            .map(|cell| clean(cell))
                            .unwrap_or("")
                            .to_owned()
                    })
                    .chain(
                        keep.iter()
                            .map(|i| row.get(*i).map(|cell| clean(cell)).unwrap_or("").to_owned()),
                    )
                    .collect(),
            );
        }
        parts_out.push((device.name.clone(), part));
    }
    if parts_out.is_empty() {
        return None;
    }
    parts_out.sort_by(|a, b| a.0.cmp(&b.0));
    let parts: Vec<Part> = parts_out.into_iter().map(|(_, part)| part).collect();

    // Timeslice keys joined on numeric value; a row missing in one device
    // leaves that block empty.
    let mut keys: Vec<(String, String, String)> = Vec::new();
    let mut seen: HashSet<(String, String, String)> = HashSet::new();
    for part in &parts {
        for key in &part.order {
            if seen.insert(key.clone()) {
                keys.push(key.clone());
            }
        }
    }
    // Numeric first, then lexicographic, per key component — the old
    // mixed-compare path was not a strict weak order (3 < 10 < "2x" < 3)
    // and total_cmp keeps NaN/inf text keys ordered deterministically.
    let cell_cmp = |a: &str, b: &str| match (a.parse::<f64>().ok(), b.parse::<f64>().ok()) {
        (Some(x), Some(y)) => x.total_cmp(&y),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => a.cmp(b),
    };
    keys.sort_by(|a, b| {
        cell_cmp(&a.0, &b.0)
            .then_with(|| cell_cmp(&a.1, &b.1))
            .then_with(|| cell_cmp(&a.2, &b.2))
    });

    let mut header = Vec::new();
    for part in &parts {
        header.extend(part.header.iter().cloned());
        header.push("_state".into());
    }
    let mut out = join_cells(header.iter().map(String::as_str));
    out.push('\n');
    for key in &keys {
        let mut cells: Vec<String> = Vec::new();
        for part in &parts {
            match part.rows.get(key) {
                Some(row) => cells.extend(row.iter().map(|cell| quoted(cell))),
                None => cells.extend(std::iter::repeat_n(String::new(), part.header.len())),
            }
            cells.push("0".into());
        }
        out.push_str(&cells.join(","));
        out.push('\n');
    }
    Some(out)
}

/// Devices write `null` for absent values and `true`/`false` for flags; the G3
/// readers want an empty cell and numeric `1`/`0`.
fn clean(cell: &str) -> &str {
    match cell {
        "null" => "",
        "true" => "1",
        "false" => "0",
        _ => cell,
    }
}

/// Emitted header name for a joined device column: drop the `(unit)` suffix
/// the device writes — G3 headers are bare names — and give the bare
/// `r_icN_{x|y}` continuous position its G3 name `r_icN_{x|y}_position`.
fn dest_name(name: &str) -> String {
    let base = base_name(name);
    if let Some(rest) = base.strip_prefix("r_ic") {
        if let Some((digits, axis)) = rest.split_once('_') {
            if !digits.is_empty()
                && digits.chars().all(|ch| ch.is_ascii_digit())
                && matches!(axis, "x" | "y")
            {
                return format!("r_ic{digits}_{axis}_position");
            }
        }
    }
    base.to_owned()
}

fn quoted(cell: &str) -> String {
    let flat = cell.replace(['\r', '\n'], " ");
    if flat.contains([',', '"']) {
        format!("\"{}\"", flat.replace('"', "\"\""))
    } else {
        flat
    }
}

/// `spot_data.csv`: outer join of every device's per-spot table on
/// `(layer_id, spot_no)` in RCI spot order — G3 block layout with repeated meta
/// columns, renamed measured columns, `_state` separators and `datetime`,
/// `state` trailers.
fn join_spot_data(spots: &[(&CaptureDevice, Sheet)], warnings: &mut Vec<String>) -> Option<String> {
    struct Part {
        dest: Vec<String>,
        rows: BTreeMap<(String, String, usize), Vec<String>>,
    }
    let mut parts = Vec::new();
    let mut order: Vec<(String, String, usize)> = Vec::new();
    let mut seen_order: HashSet<(String, String, usize)> = HashSet::new();
    let mut states: HashMap<(String, String, usize), String> = HashMap::new();
    for (device, sheet) in spots {
        let Some(spot_i) = sheet.column("spot_no") else {
            warnings.push(format!("{}: spot table has no spot_no column", device.name));
            continue;
        };
        let Some(layer_i) = sheet.column("layer_id") else {
            warnings.push(format!(
                "{}: spot table has no layer_id column",
                device.name
            ));
            continue;
        };
        let state_i = sheet.column("state");
        let (dest, sources) = spot_columns(device, sheet);
        let src_idx: Vec<Option<usize>> = sources.iter().map(|name| sheet.column(name)).collect();
        let mut part = Part {
            dest,
            rows: BTreeMap::new(),
        };
        // A re-delivered layer repeats a (layer, spot) pair in the same sheet;
        // occurrence keeps deliveries distinct instead of the last row
        // silently overwriting the first while its state survives.
        let mut occurrences: HashMap<(String, String), usize> = HashMap::new();
        // A leading ragged row files under the NEXT layer in blocks() - seed
        // the same way or it joins under "" and doubles the spot.
        let mut eff_layer = sheet
            .rows
            .iter()
            .find_map(|row| {
                let raw = clean(row.get(layer_i).map(String::as_str).unwrap_or_default());
                if raw.is_empty() {
                    None
                } else {
                    Some(raw.to_owned())
                }
            })
            .unwrap_or_default();
        for row in &sheet.rows {
            // A ragged layer cell inherits the running layer like blocks()
            // does - foldering and joining must agree on the same layer.
            let raw_layer = clean(row.get(layer_i).map(String::as_str).unwrap_or_default());
            if !raw_layer.is_empty() && !raw_layer.eq_ignore_ascii_case("layer_id") {
                eff_layer = raw_layer.to_owned();
            }
            let layer = join_key(&eff_layer);
            let spot = join_key(clean(
                row.get(spot_i).map(String::as_str).unwrap_or_default(),
            ));
            if spot.is_empty() {
                continue;
            }
            let base = (layer.clone(), spot.clone());
            let n = occurrences.entry(base).or_insert(0);
            let key = (layer, spot, *n);
            *n += 1;
            // Spot order follows the RCI table first; a device-only spot still
            // joins in encounter order rather than being dropped.
            if seen_order.insert(key.clone()) {
                order.push(key.clone());
            }
            if let Some(state) = state_i
                .and_then(|i| row.get(i))
                .map(|text| clean(text))
                .filter(|text| !text.is_empty())
            {
                states
                    .entry(key.clone())
                    .or_insert_with(|| state.to_owned());
            }
            part.rows.insert(
                key,
                src_idx
                    .iter()
                    .map(|idx| {
                        idx.and_then(|i| row.get(i))
                            .map(|c| clean(c))
                            .unwrap_or("")
                            .to_owned()
                    })
                    .collect(),
            );
        }
        parts.push(part);
    }
    if order.is_empty() {
        return None;
    }
    let mut header = Vec::new();
    for part in &parts {
        header.extend(part.dest.iter().cloned());
        header.push("_state".into());
    }
    header.push("datetime".into());
    header.push("state".into());
    let mut out = join_cells(header.iter().map(String::as_str));
    out.push('\n');
    for key in &order {
        let mut cells = Vec::new();
        for part in &parts {
            match part.rows.get(key) {
                Some(row) => cells.extend(row.iter().map(|cell| quoted(cell))),
                None => cells.extend(std::iter::repeat_n(String::new(), part.dest.len())),
            }
            cells.push(if part.rows.contains_key(key) {
                "1".into()
            } else {
                String::new()
            });
        }
        cells.push(String::new()); // datetime — the device tables carry none
        cells.push(quoted(states.get(key).map(String::as_str).unwrap_or("")));
        out.push_str(&cells.join(","));
        out.push('\n');
    }
    Some(out)
}

/// (G3 destination, device source) column pairs for a device's spot block.
/// Measured columns land on their `*_raw` names (raw-only mode).
fn spot_columns(device: &CaptureDevice, sheet: &Sheet) -> (Vec<String>, Vec<String>) {
    let mut pairs: Vec<(String, String)> = vec![];
    // A meta column the device lacks lands as an all-empty shadow that
    // renames the real column in a later block to "timestamp.1" — only emit
    // what the sheet actually carries.
    for (dest, src) in [
        ("spot_no", "spot_no"),
        ("layer_id", "layer_id"),
        ("timesliceNumber", "timeslice_number"),
        ("timestamp", "timeslice_timestamp"),
        ("point_time", "point_time"),
    ] {
        if sheet.column(src).is_some() {
            pairs.push((dest.into(), src.into()));
        }
    }
    if device.name == "RCI" {
        pairs.push(("c_current_rci".into(), "beam_current_command".into()));
    } else if let Some(ic) = ic_prefix(sheet) {
        // Python's G3 map: ic3 dose keeps the r_ prefix, ic1/ic2 do not;
        // strip axis B is X and axis A is Y. `point_dose` is the per-spot dose
        // (G3 `icN_total_dose_spot`); `total_dose` is the running scan total.
        let per_spot = if sheet.column("point_dose").is_some() {
            "point_dose"
        } else {
            "total_dose"
        };
        let dose = if ic == "ic3" {
            format!("r_{ic}_total_dose_spot_raw")
        } else {
            format!("{ic}_total_dose_spot_raw")
        };
        pairs.push((dose, per_spot.into()));
        if per_spot == "point_dose" {
            pairs.push((format!("r_{ic}_scan_total_dose_raw"), "total_dose".into()));
        }
        let target = format!("{ic}_dose_target");
        if sheet.column(&target).is_some() {
            pairs.push((target.clone(), target));
        }
        for (strip, axis) in [("b", "x"), ("a", "y")] {
            pairs.push((
                format!("r_{ic}_{axis}_spot_position_raw"),
                format!("{ic}_position_measured_{strip}"),
            ));
            pairs.push((
                format!("r_{ic}_{axis}_spot_sigma_raw"),
                format!("{ic}_sigma_measured_{strip}"),
            ));
        }
    } else {
        // Non-IC devices (TX2 amplifiers, probes, scan coils): carry every
        // remaining data column under its raw name, matching the `_raw`
        // convention the IC blocks use.
        let meta = [
            "spot_no",
            "layer_id",
            "timeslice_number",
            "timeslice_timestamp",
            "point_time",
            "state",
            "beam_off",
            "energy",
        ];
        for name in &sheet.header {
            if meta.contains(&base_name(name)) {
                continue;
            }
            pairs.push((
                format!("{}_raw", dest_name(name)),
                base_name(name).to_owned(),
            ));
        }
    }
    (
        pairs.iter().map(|(dst, _)| dst.clone()).collect(),
        pairs.into_iter().map(|(_, src)| src).collect(),
    )
}

/// `input_map.csv`: prefer the locally-uploaded plan file (plan-synthesis
/// schema, headers normalized to bare names plus `spot_no,layer_id,beam_off`
/// appended by row order). Fall back to the on-device `control_points.csv`,
/// then to the plan columns of the RCI table.
fn write_input_map(
    rci_host: &str,
    plan_path: Option<&Path>,
    rci_spot: Option<&Sheet>,
    dir: &Path,
    warnings: &mut Vec<String>,
) -> Result<usize, String> {
    let dest = dir.join("input_map.csv");
    if let Some(path) = plan_path.filter(|path| path.is_file()) {
        let bytes = std::fs::File::open(path)
            .map_err(|err| err.to_string())
            .and_then(|mut file| {
                crate::discover::read_capped(
                    &mut file,
                    "plan file",
                    crate::discover::MAX_MEMBER_BYTES,
                )
            })
            .unwrap_or_else(|err| {
                warnings.push(format!("{}: {err}", path.display()));
                Vec::new()
            });
        if let Some(mut sheet) = parse_sheet(&bytes).filter(|_| !bytes.is_empty()) {
            // "Truncated" also fires when an authored plan lacks a trailing
            // newline - warn on real truncation but still prefer it over
            // the device fallbacks.
            if sheet.truncated {
                warnings.push(format!("{}: plan table truncated mid-file", path.display()));
            }
            sheet.header = sheet
                .header
                .iter()
                .map(|name| name.split('(').next().unwrap_or(name).trim().to_owned())
                .collect();
            let extra: Vec<usize> = rci_spot
                .map(|spot| {
                    ["spot_no", "layer_id", "beam_off"]
                        .iter()
                        .filter_map(|name| spot.column(name))
                        .collect()
                })
                .unwrap_or_default();
            // The join is positional — a shorter RCI table means the device
            // filtered rows, so every pair from the divergence on is a wrong
            // label. Emit without the extra columns rather than mislabel.
            // A longer RCI table still joins the first N rows: re-delivered
            // spots land at the tail, but a mid-table insert would misalign
            // the same way — there is no key to compare against, so `>=` is
            // the reachable middle ground.
            if let Some(spot) =
                rci_spot.filter(|spot| !extra.is_empty() && spot.rows.len() >= sheet.rows.len())
            {
                // Same normalization the other export paths apply — a suffixed
                // header would write `spot_no(x)` for a column readers look up
                // as `spot_no`.
                let names: Vec<String> = extra
                    .iter()
                    .map(|i| base_name(&spot.header[*i]).to_owned())
                    .collect();
                // Pad short rows first — flexible parsing leaves them under the
                // header length, and pushing the new cells there would land them
                // in the wrong columns. Truncate over-long rows the same way.
                let columns = sheet.header.len();
                sheet.header.extend(names);
                for (index, row) in sheet.rows.iter_mut().enumerate() {
                    row.truncate(columns);
                    row.resize(columns, String::new());
                    for col in &extra {
                        row.push(
                            spot.rows
                                .get(index)
                                .and_then(|cells| cells.get(*col))
                                .map(|cell| clean(cell).to_owned())
                                .unwrap_or_default(),
                        );
                    }
                }
            }
            write_text(&dest, &csv_text(&sheet))?;
            return Ok(1);
        }
    }
    if let Ok(url) = device_file_url(rci_host, "/root/config/control/control_points.csv") {
        if let Ok(bytes) = runner::http(&url, "GET", None) {
            if !bytes.is_empty() {
                write_bytes(&dest, &bytes)?;
                return Ok(1);
            }
        }
    }
    if let Some(sheet) = rci_spot {
        let keep: Vec<usize> = ["spot_no", "layer_id", "beam_off", "energy"]
            .iter()
            .filter_map(|name| sheet.column(name))
            .collect();
        if !keep.is_empty() {
            // Same normalization the plan path applies — a suffixed header
            // would write `spot_no(x)` for a column readers look up as
            // `spot_no`.
            let mut text = join_cells(keep.iter().map(|i| {
                sheet.header[*i]
                    .split('(')
                    .next()
                    .unwrap_or(sheet.header[*i].as_str())
                    .trim()
            }));
            text.push('\n');
            for row in &sheet.rows {
                text.push_str(&join_cells(
                    keep.iter()
                        .map(|i| row.get(*i).map(|cell| clean(cell)).unwrap_or("")),
                ));
                text.push('\n');
            }
            write_text(&dest, &text)?;
            return Ok(1);
        }
    }
    Err("No plan data on the device or locally - upload a plan to the device and retry.".into())
}

/// `session_meta.json` + `termination_summary.txt`(+`.md5`), best effort.
fn write_session_meta(
    dir: &Path,
    session_id: &str,
    status: &Map<String, Value>,
    devices: &[CaptureDevice],
) -> Result<usize, String> {
    let meta = json!({
        "SessionID": session_id,
        "CombinedState": status.get(scan_kit_core::COMBINED_STATE).map(scan_kit_core::io_text).unwrap_or_default(),
        "Devices": devices.iter().map(|device| json!({
            "name": device.name, "io": device.io_root, "host": device.host,
        })).collect::<Vec<_>>(),
    });
    write_text(&dir.join("session_meta.json"), &format!("{meta:#}\n"))?;
    Ok(1)
}

/// Number of seconds, trimmed like the G3 writer (`319.494`, not `319.494000`).
fn num_text(value: f64) -> String {
    if !value.is_finite() {
        return "n/a".into();
    }
    let text = format!("{value:.6}");
    let trimmed = text.trim_end_matches('0').trim_end_matches('.');
    if trimmed.is_empty() || trimmed == "-" || trimmed == "-0" {
        "0".into()
    } else {
        trimmed.to_owned()
    }
}

/// Parse the numeric cells of one column.
fn column_values(sheet: &Sheet, name: &str) -> Vec<f64> {
    sheet
        .column(name)
        .map(|index| {
            sheet
                .rows
                .iter()
                .filter_map(|row| row.get(index)?.trim().parse().ok())
                .filter(|value: &f64| value.is_finite())
                .collect()
        })
        .unwrap_or_default()
}

/// `input_map.csv` headers come from the plan CSV (`ENERGY`) or the device
/// control table (`energy`); match either.
fn column_icase(sheet: &Sheet, name: &str) -> Vec<f64> {
    let index = sheet
        .header
        .iter()
        .position(|cell| base_name(cell.trim()).eq_ignore_ascii_case(name));
    index
        .map(|i| {
            sheet
                .rows
                .iter()
                .filter_map(|row| row.get(i)?.trim().parse().ok())
                .filter(|value: &f64| value.is_finite())
                .collect()
        })
        .unwrap_or_default()
}

/// The G3 termination summary. Anything the tables or the runner status can
/// back up is emitted; fields with no honest source (room number, tolerance
/// table, pause accounting, config name) are left out rather than faked.
fn write_termination(
    dir: &Path,
    session_id: &str,
    status: &Map<String, Value>,
    spots: &[(&CaptureDevice, Sheet)],
    slices: &[(&CaptureDevice, Sheet)],
) -> Result<usize, String> {
    let ic_sheet = |ic: &str| -> Option<&Sheet> {
        spots
            .iter()
            .find(|(_, sheet)| ic_prefix(sheet).as_deref() == Some(ic))
            .map(|(_, sheet)| sheet)
    };
    let last_dose = |ic: &str| -> Option<String> {
        let sheet = ic_sheet(ic)?;
        let index = sheet.column("total_dose")?;
        let dose: f64 = sheet.rows.last()?.get(index)?.trim().parse().ok()?;
        // Summary convention is measured units — "Total dose target" below
        // scales its C targets INTO nC to sit beside this raw nC total.
        Some(num_text(dose))
    };
    let input_map = std::fs::read(dir.join("input_map.csv"))
        .ok()
        .and_then(|bytes| parse_sheet(&bytes));
    let rci_slice = slices
        .iter()
        .find(|(device, _)| device.name == "RCI")
        .map(|(_, sheet)| sheet);
    let rci_spot = spots
        .iter()
        .find(|(device, _)| device.name == "RCI")
        .map(|(_, sheet)| sheet);

    // Wall clock: the controller's elapsed seconds wins; the timeslice clock
    // (`timestamp`, ms since start) is the fallback.
    let ticks = rci_slice
        .map(|sheet| column_values(sheet, "timestamp"))
        .unwrap_or_default();
    let span_s = (|| {
        let (first, last) = (ticks.first()?, ticks.last()?);
        (ticks.len() > 1).then(|| (last - first) / 1000.0)
    })();
    let elapsed_s = status
        .get(TIME_ELAPSED)
        .and_then(io_number)
        .or(span_s)
        .filter(|seconds| *seconds > 0.0);

    // Beam-on: the ms of the timeslice clock where commanded beam current
    // (`c_beamI`) is nonzero.
    let beam_on_s = rci_slice.and_then(|sheet| {
        let ti = sheet.column("timestamp")?;
        let bi = sheet.column("c_beamI")?;
        let mut ms = 0.0;
        let mut prev = None;
        for row in &sheet.rows {
            let t = row.get(ti).and_then(|cell| cell.parse::<f64>().ok());
            let on = row
                .get(bi)
                .and_then(|cell| cell.parse::<f64>().ok())
                .unwrap_or(0.0)
                > 0.0;
            if let (Some(t), Some(p)) = (t, prev) {
                if on && t > p {
                    ms += t - p;
                }
            }
            if t.is_some_and(|t| t.is_finite()) {
                prev = t;
            }
        }
        Some(ms / 1000.0)
    });

    // Total dose target: the plan's CHARGE_REQ sum, else the primary channel's
    // `icN_dose_target` scaled from C into the measured units.
    let dose_target = input_map
        .as_ref()
        .map(|sheet| column_icase(sheet, "charge_req"))
        .filter(|values| !values.is_empty())
        .map(|values| values.iter().sum::<f64>())
        .or_else(|| {
            let sheet = ["ic3", "ic1", "ic2"].iter().find_map(|ic| ic_sheet(ic))?;
            let ic = ic_prefix(sheet)?;
            let measured: Vec<f32> = column_values(sheet, "point_dose")
                .iter()
                .map(|v| *v as f32)
                .collect();
            let targets: Vec<f32> = column_values(sheet, &format!("{ic}_dose_target"))
                .iter()
                .map(|v| *v as f32)
                .collect();
            if targets.is_empty() {
                return None;
            }
            let scale = tables::dose_unit_scale(&measured, &targets) as f64;
            Some(targets.iter().map(|v| *v as f64).sum::<f64>() * scale)
        });

    let layers_of = |sheet: &Sheet| -> usize {
        sheet
            .column("layer_id")
            .map(|index| {
                sheet
                    .rows
                    .iter()
                    .filter_map(|row| row.get(index))
                    // join_key normalizes "1"/"1.0"/"01" — counting raw
                    // spellings reports phantom extra layers. Empty/"null"
                    // cells mean "inherit" everywhere else; counting them
                    // adds a phantom layer too.
                    .filter_map(|cell| {
                        let cell = clean(cell);
                        (!cell.is_empty()).then(|| join_key(cell))
                    })
                    .collect::<HashSet<_>>()
                    .len()
            })
            .unwrap_or_default()
    };
    let delivered_layers = rci_spot.map(layers_of).unwrap_or_default();
    let planned_layers = input_map
        .as_ref()
        .filter(|sheet| sheet.column("layer_id").is_some())
        .map(layers_of)
        .filter(|count| *count > 0);
    let delivered_spots = rci_spot.map(|sheet| sheet.rows.len()).unwrap_or_default();
    let planned_spots = status
        .get(CONTROL_POINT_COUNT)
        .and_then(io_number)
        // 1e20 as usize saturates — a nonsense count must not print a
        // 19-digit "planned" total in the summary.
        .filter(|count| count.is_finite() && *count >= 0.0 && count.fract() == 0.0 && *count <= 1e6)
        .map(|count| count as usize)
        .or_else(|| input_map.as_ref().map(|sheet| sheet.rows.len()))
        .filter(|count| *count > 0);

    let energies = input_map
        .as_ref()
        .map(|sheet| column_icase(sheet, "energy"))
        .unwrap_or_default();
    let extent = input_map.as_ref().and_then(|sheet| {
        let span = |name: &str| {
            let values = column_icase(sheet, name);
            values
                .iter()
                .copied()
                .reduce(f64::min)
                .zip(values.iter().copied().reduce(f64::max))
                .map(|(lo, hi)| hi - lo)
        };
        let width = span("x_position")?;
        let height = span("y_position")?;
        Some((width, height))
    });

    let state = status
        .get(scan_kit_core::STATE)
        .map(scan_kit_core::io_text)
        .unwrap_or_default()
        .chars()
        .filter(|ch| !ch.is_control())
        .collect::<String>();
    let mut text = format!(
        "Termination status: {}\nDate: {}\nSession ID: {session_id}\nError code: 0\n",
        if state.is_empty() {
            "eNORMAL".into()
        } else {
            state
        },
        chrono::Local::now().format("%a %b %e %T %Y"),
    );
    // These are raw nC readings, not MU — a "total dose" label would be
    // parsed as MU by the summary reader and displayed as ~1e8 MU.
    for (label, ic) in [
        ("Primary total charge (nC)", "ic3"),
        ("Secondary total charge (nC)", "ic1"),
        ("Tertiary total charge (nC)", "ic2"),
    ] {
        if let Some(dose) = last_dose(ic) {
            // Device text lands raw in this file — a newline would forge a
            // record line, same as `meta_id` guards above.
            let dose: String = dose.chars().filter(|ch| !ch.is_control()).collect();
            text.push_str(&format!("{label}: {dose}\n"));
        }
    }
    if let Some(target) = dose_target {
        text.push_str(&format!("Total dose target: {}\n", num_text(target)));
    }
    if let Some(elapsed) = elapsed_s {
        text.push_str(&format!(
            "Session delivery time: {} seconds\nTreatment time: {} seconds\n",
            num_text(elapsed),
            num_text(elapsed)
        ));
    }
    if let Some(beam) = beam_on_s {
        text.push_str(&format!("Beam on time: {} seconds\n", num_text(beam)));
    }
    if delivered_layers > 0 {
        let planned = planned_layers.unwrap_or(delivered_layers);
        text.push_str(&format!("Layer delivery: {delivered_layers}/{planned}\n"));
    }
    if delivered_spots > 0 {
        let planned = planned_spots.unwrap_or(delivered_spots);
        text.push_str(&format!("Spot delivery: {delivered_spots}/{planned}\n"));
    }
    if let (Some(lo), Some(hi)) = (
        energies.iter().copied().reduce(f64::min),
        energies.iter().copied().reduce(f64::max),
    ) {
        text.push_str(&format!(
            "Energy min: {}\nEnergy max: {}\n",
            num_text(lo),
            num_text(hi)
        ));
    }
    if let Some((width, height)) = extent {
        text.push_str(&format!(
            "Spot extent width: {} mm\nSpot extent height: {} mm\n",
            num_text(width),
            num_text(height)
        ));
    }
    write_text(&dir.join("termination_summary.txt"), &text)?;
    let mut hasher = Md5::new();
    hasher.update(text.as_bytes());
    let digest = hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>()
        + "\n";
    write_text(&dir.join("termination_summary.txt.md5"), &digest)?;
    Ok(2)
}

/// Auxiliary per-device files — each is one HTTP GET; misses are skipped.
fn fetch_aux(
    rci_host: &str,
    devices: &[CaptureDevice],
    dir: &Path,
    keep: &mut dyn FnMut(u64, u64) -> bool,
) -> Result<usize, String> {
    let mut count = 0;
    // dcs channel-map names come from the target basename — two devices can
    // share it; the second write would replace the first's XML silently.
    let mut dcs_names = std::collections::HashSet::new();
    for device in devices {
        let nozzle = dir.join("config").join("nozzle").join(&device.name);
        for file in NOZZLE_FILES {
            tick(count, keep)?;
            let remote = format!("/root/config/{file}");
            if let Ok(url) = device_file_url(&device.host, &remote) {
                if let Ok(bytes) = runner::http(&url, "GET", None) {
                    if !bytes.is_empty() {
                        write_bytes(&nozzle.join(file), &bytes)?;
                        count += 1;
                    }
                }
            }
        }
        tick(count, keep)?;
        if let Ok(url) = device_file_url(&device.host, "/root/config/calibration.json") {
            if let Ok(bytes) = runner::http(&url, "GET", None) {
                if !bytes.is_empty() {
                    write_bytes(
                        &dir.join("calibration")
                            .join("devices_calibration")
                            .join(format!("{}_calibration.json", device.name)),
                        &bytes,
                    )?;
                    count += 1;
                }
            }
        }
        tick(count, keep)?;
        if let Ok(url) = io_url(&device.host, "log", "data.txt") {
            if let Ok(bytes) = runner::http(&url, "GET", None) {
                if !bytes.is_empty() {
                    if device.name == "RCI" {
                        write_bytes(&dir.join("SessionLogFile.log"), &bytes)?;
                        count += 1;
                    }
                    write_bytes(
                        &dir.join("log").join(format!("{}.log", device.name)),
                        &bytes,
                    )?;
                    count += 1;
                }
            }
        }
        // Channel-map XMLs live on the RCI under /root/config/control/.
        if !device.config_target.is_empty() {
            tick(count, keep)?;
            if let Ok(url) = device_file_url(rci_host, &device.config_target) {
                if let Ok(bytes) = runner::http(&url, "GET", None) {
                    if !bytes.is_empty() {
                        let base = PathBuf::from(&device.config_target)
                            .file_name()
                            .and_then(|name| name.to_str())
                            .filter(|name| *name != "." && *name != "..")
                            .unwrap_or(&device.name)
                            .to_owned();
                        // Remote-supplied text becomes a filename — same
                        // whitelist as client_name or characters Windows
                        // rejects (and a trailing "."/" ") sink the whole
                        // capture at write_bytes.
                        let base: String = base
                            .chars()
                            .filter(|ch| {
                                ch.is_ascii_alphanumeric() || *ch == '_' || *ch == '-' || *ch == '.'
                            })
                            .take(64)
                            .collect();
                        let base = if base.is_empty() || base.ends_with('.') || dos_reserved(&base)
                        {
                            format!("_{base}")
                        } else {
                            base
                        };
                        // The target filesystem is case-insensitive - key the
                        // dedup on the lowercase form or MAP.XML replaces
                        // map.xml while count still credits both writes.
                        let name = if dcs_names.insert(base.to_lowercase()) {
                            base
                        } else {
                            let mut name = format!("{}_{base}", device.name);
                            let mut n = 2;
                            while !dcs_names.insert(name.to_lowercase()) {
                                name = format!("{}_{n}_{base}", device.name);
                                n += 1;
                            }
                            name
                        };
                        write_bytes(&dir.join("config").join("dcs").join(name), &bytes)?;
                        count += 1;
                    }
                }
            }
        }
        tick(count, keep)?;
    }
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    fn sheet(text: &str) -> Sheet {
        parse_sheet(text.as_bytes()).unwrap()
    }

    fn device(name: &str) -> CaptureDevice {
        CaptureDevice {
            name: name.into(),
            io_root: name.to_ascii_lowercase(),
            host: String::new(),
            config_target: String::new(),
        }
    }

    #[test]
    fn meta_columns_a_device_lacks_do_not_shadow_later_blocks() {
        // First sheet carries no timeslice_timestamp/point_time — emitting
        // all-empty destination columns would rename a later block's real
        // "timestamp" to "timestamp.1".
        let dev = device("IC");
        let sheet = sheet("spot_no,total_dose\n1,2.5\n");
        let (dest, _src) = spot_columns(&dev, &sheet);
        assert!(!dest.iter().any(|d| d == "timestamp" || d == "point_time"));
        assert!(dest.iter().any(|d| d == "spot_no"));
    }

    #[test]
    fn a_header_with_a_unit_suffix_still_matches_its_base_name() {
        let sheet = sheet("ENERGY(MeV),total_dose\n70.0,1.0\n");
        assert_eq!(column_icase(&sheet, "energy"), vec![70.0]);
    }

    #[test]
    fn the_termination_summary_keeps_dose_lines_in_measured_units() {
        // Device logs nC and the summary's target line is scaled INTO nC —
        // the primary/secondary totals must emit the same unit.
        let dir = std::env::temp_dir().join(format!("sk_r21_term_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let dev = device("IC");
        let sheet = sheet(
            "point_dose,total_dose,ic1_total_dose,ic1_dose_target\n1.0e8,1.0e8,1.0e8,0.1\n2.0e8,2.0e8,2.0e8,0.2\n",
        );
        let spots = vec![(&dev, sheet)];
        let status = Map::new();
        write_termination(&dir, "sess", &status, &spots, &[]).unwrap();
        let text = std::fs::read_to_string(dir.join("termination_summary.txt")).unwrap();
        let dose_line = text
            .lines()
            .find(|line| line.starts_with("Secondary total charge"))
            .unwrap_or("");
        let target_line = text
            .lines()
            .find(|line| line.starts_with("Total dose target"))
            .unwrap_or("");
        // Both in nC: the raw 2.0e8 total and 0.3 C of targets x 1e9.
        assert!(dose_line.contains("200000000"), "dose line: {dose_line}");
        assert!(
            target_line.contains("3000000"),
            "target line: {target_line}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    fn refs(parts: &[(CaptureDevice, Sheet)]) -> Vec<(&CaptureDevice, Sheet)> {
        parts
            .iter()
            .map(|(device, sheet)| (device, (*sheet).clone()))
            .collect::<Vec<(&CaptureDevice, Sheet)>>()
    }

    #[test]
    fn client_names_cannot_escape_the_capture_dir() {
        assert_eq!(client_name("device_ic1_client"), "IC1");
        assert_eq!(client_name("device_../../evil_client"), "EVIL");
        assert_eq!(client_name("device_.._client"), "");
        assert_eq!(client_name("odd"), "ODD");
    }

    #[test]
    fn blocks_group_contiguous_layers_into_runs() {
        let table = sheet("spot_no,layer_id\n1,7\n2,7\n3,9\n4,7\n");
        let got = blocks(&table, &mut HashMap::new());
        assert_eq!(
            got.iter()
                .map(|block| (block.layer, block.run, block.start, block.end))
                .collect::<Vec<_>>(),
            vec![(0, 0, 0, 2), (1, 0, 2, 3), (0, 1, 3, 4)]
        );
        assert!(blocks(&sheet("spot_no\n1\n"), &mut HashMap::new()).is_empty());
    }

    #[test]
    fn blocks_join_numeric_layer_ids_across_formats() {
        // "1" and "1.0" are one layer — joining must happen on the
        // normalized key or a format difference splits one run in two.
        let table = sheet(
            "spot_no,layer_id
1,1
2,1.0
3,2
",
        );
        let got = blocks(&table, &mut HashMap::new());
        assert_eq!(
            got.iter()
                .map(|block| (block.layer, block.run, block.start, block.end))
                .collect::<Vec<_>>(),
            vec![(0, 0, 0, 2), (1, 0, 2, 3)]
        );
    }

    #[test]
    fn a_ragged_row_stays_in_its_run_instead_of_shifting_later_layers() {
        // Row `2` has no layer_id cell — it must not open a phantom "" block
        // that consumes an ordinal and mis-tags every later layer.
        let table = sheet("spot_no,layer_id\n1,7\n2\n3,9\n4,9\n");
        let got = blocks(&table, &mut HashMap::new());
        assert_eq!(
            got.iter()
                .map(|block| (block.layer, block.run, block.start, block.end))
                .collect::<Vec<_>>(),
            vec![(0, 0, 0, 2), (1, 0, 2, 4)]
        );
    }

    #[test]
    fn unit_suffixed_headers_still_resolve_columns() {
        let table = sheet("spot_no,timestamp(ms),total_dose(nC)\n1,0.5,3\n");
        assert_eq!(table.column("timestamp"), Some(1));
        assert_eq!(table.column("total_dose"), Some(2));
        assert_eq!(
            base_name("ic1_sigma_measured_a(mm)"),
            "ic1_sigma_measured_a"
        );
        assert_eq!(ic_prefix(&table), None);
        let ic = sheet("spot_no,ic1_dose_target,total_dose(nC)\n1,2,3\n");
        assert_eq!(ic_prefix(&ic).as_deref(), Some("ic1"));
        // A command column alone is not a dose chamber (ICSIM's ic3_current_cmd).
        let icsim = sheet("spot_no,ic3_current_cmd\n1,2\n");
        assert_eq!(ic_prefix(&icsim), None);
        let unnamed = parse_sheet(b"spot_no,,x\n1,2,3\n").unwrap();
        assert_eq!(unnamed.header[1], "unnamed_1");
        assert!(parse_sheet(b"a,b\n").is_none());
        // Emitted names are bare G3 names; the continuous strip position takes
        // the canonical `_position` name.
        assert_eq!(dest_name("r_ic1_x(ch)"), "r_ic1_x_position");
        assert_eq!(dest_name("r_ic1_y(ch)"), "r_ic1_y_position");
        assert_eq!(dest_name("r_ic12_x(ch)"), "r_ic12_x_position");
        assert_eq!(dest_name("r_ic1_x_strip_sum"), "r_ic1_x_strip_sum");
        assert_eq!(
            dest_name("r_ic1_x_spot_position(ch)"),
            "r_ic1_x_spot_position"
        );
        assert_eq!(dest_name("ic3_current_A(nA)"), "ic3_current_A");
        assert_eq!(dest_name("timestamp(ms)"), "timestamp");
        assert_eq!(clean("true"), "1");
        assert_eq!(clean("false"), "0");
        assert_eq!(clean("null"), "");
        assert_eq!(clean("1.5"), "1.5");
    }

    #[test]
    fn timeslices_join_on_timeslice_number() {
        let rci = sheet(
            "spot_no,layer_id,timesliceNumber,timestamp(ms),r_beamOk\n1,0,1,10,true\n1,0,2,20,false\n",
        );
        let ic = sheet(
            "spot_no,layer_id,timesliceNumber,timestamp(ms),ic1_total_dose_spot(nC),r_ic1_x(ch)\n1,0,1,10,3.5,64.5\n1,0,10,30,4.0,62\n",
        );
        let parts = vec![(device("RCI"), rci), (device("IX256_1"), ic)];
        let text = join_timeslices(&refs(&parts), &mut Vec::new()).unwrap();
        let mut lines = text.lines();
        let header = lines.next().unwrap();
        // Meta repeats per device, in device-name order, each block ends _state.
        // Emitted names are bare: unit suffixes drop, `r_ic1_x` is G3's
        // `r_ic1_x_position`, and flag cells come out numeric.
        assert_eq!(
            header,
            "spot_no,layer_id,timesliceNumber,timestamp,ic1_total_dose_spot,r_ic1_x_position,_state,spot_no,layer_id,timesliceNumber,timestamp,r_beamOk,_state"
        );
        let rows: Vec<&str> = lines.collect();
        assert_eq!(rows.len(), 3); // union of keys 1, 2, 10, numerically sorted
        assert_eq!(rows[0], "1,0,1,10,3.5,64.5,0,1,0,1,10,1,0");
        assert_eq!(rows[1], ",,,,,,0,1,0,2,20,0,0");
        assert_eq!(rows[2], "1,0,10,30,4.0,62,0,,,,,,0");
    }

    #[test]
    fn spot_data_join_maps_measured_columns_to_raw_names() {
        let rci = sheet(
            "spot_no,layer_id,timeslice_number,timeslice_timestamp(ms),beam_current_command\n1,0,1,10,9\n2,0,2,20,8\n",
        );
        let ic = sheet(
            "spot_no,layer_id,timeslice_number,timeslice_timestamp(ms),point_time(ms),point_dose(nC),total_dose(nC),ic1_dose_target,ic1_position_measured_b,ic1_position_measured_a,ic1_sigma_measured_b(mm),ic1_sigma_measured_a(mm)\n1,0,1,10,5,0.4,3.5,4e-10,11,12,0.1,0.2\n",
        );
        let tx2 = sheet(
            "spot_no,layer_id,timeslice_number,timeslice_timestamp(ms),point_time(ms),c_x(V),r_xV(V),state\n1,0,1,10,5,-1.8,-1.9,7\n",
        );
        let parts = vec![
            (device("RCI"), rci),
            (device("IX256_1"), ic),
            (device("TX2"), tx2),
        ];
        let text = join_spot_data(&refs(&parts), &mut Vec::new()).unwrap();
        let mut lines = text.lines();
        let header = lines.next().unwrap();
        assert!(header.contains("c_current_rci"));
        // `point_dose` is the per-spot dose; `total_dose` is the running scan.
        assert!(header.contains("ic1_total_dose_spot_raw"));
        assert!(header.contains("r_ic1_scan_total_dose_raw"));
        assert!(header.contains("ic1_dose_target"));
        assert!(header.contains("r_ic1_x_spot_position_raw"));
        assert!(header.contains("r_ic1_y_spot_sigma_raw"));
        // Non-IC devices carry their data columns under raw names.
        assert!(header.contains("c_x_raw"));
        assert!(header.contains("r_xV_raw"));
        assert!(header.ends_with("datetime,state"));
        let rows: Vec<&str> = lines.collect();
        assert_eq!(rows.len(), 2); // RCI order; spot 2 has no IC row
        assert_eq!(
            rows[0],
            "1,0,1,10,9,1,1,0,1,10,5,0.4,3.5,4e-10,11,0.1,12,0.2,1,1,0,1,10,5,-1.8,-1.9,1,,7"
        );
        // Spot 2: IC and TX2 block cells are empty.
        assert_eq!(rows[1], "2,0,2,20,8,1,,,,,,,,,,,,,,,,,,,,,,,");
    }

    #[test]
    fn capture_writes_a_g3_layout_from_two_fake_devices() {
        // `__HOST__` in a route body is replaced with this server's host:port,
        // which lets the fake RCI point its device client at itself.
        let host = serve(HashMap::from([
            (
                "/io/rci/map_manager/children.json",
                r#"{"name":"map_manager","device_ix256_1_client":{},"version_io":{}}"#,
            ),
            (
                "/io/rci/map_manager/device_ix256_1_client/device_path/value.json",
                r#""ix256""#,
            ),
            (
                "/io/rci/map_manager/device_ix256_1_client/websocket_client/server_ip_address/value.json",
                r#""__HOST__""#,
            ),
            (
                "/io/rci/map_manager/device_ix256_1_client/configuration_upload/target/value.json",
                r#""/root/config/control/IX256_1.xml""#,
            ),
            (
                "/io/rci/control_point_table/timeslice_table/file_header_meta/value.json",
                r#""4242|0|0|0""#,
            ),
            (
                "/io/ix256/control_point_table/timeslice_table/file_header_meta/value.json",
                r#""4242|0|0|3""#,
            ),
            (
                "/io/rci/control_point_table/table/data.csv",
                "spot_no,layer_id,timeslice_number,timeslice_timestamp(ms),beam_current_command,energy\n1,0,1,10,9,70\n2,0,2,20,8,70\n",
            ),
            (
                "/io/ix256/control_point_table/table/data.csv",
                "spot_no,layer_id,timeslice_number,timeslice_timestamp(ms),point_time(ms),total_dose(nC),ic1_dose_target\n1,0,1,10,5,3.5,3.5\n2,0,2,20,5,7.0,3.5\n",
            ),
            (
                "/io/rci/control_point_table/timeslice_table/data.csv",
                "spot_no,layer_id,timesliceNumber,timestamp(ms),r_beamOk\n1,0,1,10,1\n",
            ),
            (
                "/io/ix256/control_point_table/timeslice_table/data.csv",
                "spot_no,layer_id,timesliceNumber,timestamp(ms),ic1_total_dose_spot(nC)\n1,0,1,10,3.5\n",
            ),
            (
                "/root/config/control/control_points.csv",
                "spot_no,layer_id,energy\n1,0,70\n",
            ),
            ("/root/config/control/IX256_1.xml", "<dev/>"),
            ("/root/config/control/RCI.xml", "<rci/>"),
            ("/root/config/config.json", "{}"),
            ("/root/config/calibration.json", "{\"k\":1}"),
            ("/io/log/data.txt", "log lines\n"),
        ]));
        let dir = std::env::temp_dir().join(format!("scan-kit-capture-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let status = Map::new();
        let report = capture(&host, &status, None, &dir, &mut |_, _| true).unwrap();
        assert_eq!(report.session_id, "4242");
        let run = dir.join("layer-0").join("run-0");
        assert!(run.join("RCI_spot_data.csv").is_file());
        assert!(run.join("RCI_map.csv").is_file());
        assert!(run.join("IX256_1_spot_data.csv").is_file());
        let timeslice =
            std::fs::read_to_string(run.join("timeslice_data_device_units.csv")).unwrap();
        assert!(timeslice.starts_with(
            "spot_no,layer_id,timesliceNumber,timestamp,ic1_total_dose_spot,_state,spot_no,layer_id,timesliceNumber,timestamp,r_beamOk,_state\n"
        ));
        let spot = std::fs::read_to_string(dir.join("spot_data.csv")).unwrap();
        assert!(spot.contains("c_current_rci"));
        assert!(spot.contains("ic1_total_dose_spot_raw"));
        assert_eq!(
            std::fs::read_to_string(dir.join("input_map.csv")).unwrap(),
            "spot_no,layer_id,energy\n1,0,70\n"
        );
        let info: Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("session_info.json")).unwrap())
                .unwrap();
        assert_eq!(info["SessionID"], "4242");
        let summary = std::fs::read_to_string(dir.join("termination_summary.txt")).unwrap();
        assert!(summary.contains("Layer delivery: 1/1"), "{summary}");
        assert!(summary.contains("Spot delivery: 2/1"), "{summary}");
        assert!(
            summary.contains("Secondary total charge (nC): 7"),
            "{summary}"
        );
        assert!(
            summary.contains("Energy min: 70\nEnergy max: 70"),
            "{summary}"
        );
        assert!(dir.join("termination_summary.txt.md5").is_file());
        assert!(dir.join("session_meta.json").is_file());
        assert!(dir.join("config/dcs/IX256_1.xml").is_file());
        assert!(dir.join("config/dcs/RCI.xml").is_file());
        assert!(dir.join("config/nozzle/IX256_1/config.json").is_file());
        assert!(dir
            .join("calibration/devices_calibration/IX256_1_calibration.json")
            .is_file());
        assert!(dir.join("SessionLogFile.log").is_file());
        assert!(dir.join("log/IX256_1.log").is_file());
        assert!(report.files > 10);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Live smoke check: `SCAN_KIT_RCI=192.168.100.184 cargo test -p
    /// scan-kit-io capture_live -- --ignored`.
    #[test]
    #[ignore]
    fn capture_live_rci() {
        let host = std::env::var("SCAN_KIT_RCI").expect("SCAN_KIT_RCI=<rci host>");
        let dir = std::env::temp_dir().join(format!("scan-kit-live-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let report = capture(&host, &Map::new(), None, &dir, &mut |done, _| {
            eprintln!("captured {done} files");
            true
        })
        .unwrap();
        eprintln!("session {} ({} files)", report.session_id, report.files);
        for warning in &report.warnings {
            eprintln!("warning: {warning}");
        }
        assert!(dir.join("spot_data.csv").is_file());
        assert!(dir.join("input_map.csv").is_file());
        assert!(dir.join("session_info.json").is_file());
        assert!(dir.join("layer-0").join("run-0").is_dir());
        eprintln!("kept capture at {}", dir.display());
    }

    #[test]
    fn capture_reports_an_empty_device() {
        let host = serve(HashMap::new());
        let dir =
            std::env::temp_dir().join(format!("scan-kit-capture-empty-{}", std::process::id()));
        let err = capture(&host, &Map::new(), None, &dir, &mut |_, _| true).unwrap_err();
        assert!(err.contains("no session"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Serve a route table on a loopback port. A `__HOST__` token in a body is
    /// replaced with the server's own `host:port` so one listener can stand in
    /// for the RCI and every device client it advertises.
    fn serve(routes: HashMap<&str, &'static str>) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let host = format!("127.0.0.1:{port}");
        let routes: HashMap<String, String> = routes
            .into_iter()
            .map(|(path, body)| (path.to_owned(), body.replace("__HOST__", &host)))
            .collect();
        std::thread::spawn(move || {
            for _ in 0..120 {
                let Ok((mut sock, _)) = listener.accept() else {
                    break;
                };
                let mut buf = [0u8; 8192];
                let count = sock.read(&mut buf).unwrap_or(0);
                let request = String::from_utf8_lossy(&buf[..count]).to_string();
                let path = request.split_whitespace().nth(1).unwrap_or("/").to_string();
                let body = routes.get(&path).cloned().unwrap_or_default();
                let status = if body.is_empty() { "404 NO" } else { "200 OK" };
                let head = format!(
                    "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = sock.write_all(head.as_bytes());
                let _ = sock.write_all(body.as_bytes());
            }
        });
        host
    }

    #[test]
    fn spot_join_keeps_each_delivery_of_a_repeated_spot() {
        let rci = sheet(
            "spot_no,layer_id,timeslice_number,timeslice_timestamp(ms),beam_current_command\n1,0,1,10,9\n1,0,2,30,8\n",
        );
        let ic = sheet(
            "spot_no,layer_id,timeslice_number,timeslice_timestamp(ms),point_time(ms),point_dose(nC),total_dose(nC),ic1_dose_target,ic1_position_measured_b,ic1_position_measured_a,ic1_sigma_measured_b(mm),ic1_sigma_measured_a(mm)\n1,0,1,10,5,0.4,3.5,4e-10,11,12,0.1,0.2\n1,0,2,30,6,0.5,7.0,4e-10,21,22,0.3,0.4\n",
        );
        let parts = vec![(device("RCI"), rci), (device("IX256_1"), ic)];
        let text = join_spot_data(&refs(&parts), &mut Vec::new()).unwrap();
        let rows: Vec<&str> = text.lines().skip(1).collect();
        assert_eq!(rows.len(), 2, "both deliveries survive the join");
        assert_ne!(rows[0], rows[1]);
    }

    #[test]
    fn a_sheet_flags_a_truncated_parse() {
        let good = parse_sheet(b"a,b\n1,2\n").unwrap();
        assert!(!good.truncated);
        let bad = parse_sheet(b"a,b\n1,2\n\xff\xfe,9\n").unwrap();
        assert!(bad.truncated);
        assert_eq!(bad.rows.len(), 1);
    }
    #[test]
    fn a_session_id_drops_control_characters() {
        // Device text lands raw in termination_summary — a newline in it
        // would forge a record line on re-ingest.
        assert_eq!(meta_id("x\nError code: 9|rest"), "xError code: 9");
        assert_eq!(meta_id("a\rb"), "ab");
    }

    #[test]
    fn a_leading_ragged_row_joins_the_next_layer() {
        // Row 0 has no layer_id cell — it folds into the run its real
        // layer opens, not a separate one-row run of its own.
        let table = sheet("spot_no,layer_id\n1\n2,9\n3,9\n");
        let mut layers = HashMap::new();
        let got = blocks(&table, &mut layers);
        assert!(!layers.contains_key(""), "{layers:?}");
        assert_eq!(
            got.iter()
                .map(|block| (block.layer, block.run, block.start, block.end))
                .collect::<Vec<_>>(),
            vec![(0, 0, 0, 3)]
        );
    }

    #[test]
    fn a_join_key_normalizes_numeric_spellings() {
        assert_eq!(join_key("1"), "1");
        assert_eq!(join_key("1.0"), "1");
        assert_eq!(join_key("03"), "3");
        assert_eq!(join_key("run-2"), "run-2");
        // Past 2^53 two distinct integer ids must not collapse into one key.
        assert_ne!(join_key("9007199254740992"), join_key("9007199254740993"));
    }
    #[test]
    fn embedded_newlines_flatten_instead_of_writing_multiline_cells() {
        let line = join_cells(
            ["a\nb".to_owned(), "c\rd".to_owned()]
                .iter()
                .map(String::as_str),
        );
        assert_eq!(line, "a b,c d");
        // The fast reader is line-based — an embedded line break would read
        // back as an extra row.
        assert_eq!(line.lines().count(), 1);
        assert_eq!(quoted("x\ny"), "x y");
    }

    #[test]
    fn num_text_refuses_non_finite_values() {
        // A computed NaN/inf (divided doses, empty windows) writes "n/a",
        // not a value the CSV reader parses back as a number.
        assert_eq!(num_text(f64::NAN), "n/a");
        assert_eq!(num_text(f64::INFINITY), "n/a");
        assert_eq!(num_text(f64::NEG_INFINITY), "n/a");
    }

    #[test]
    fn num_text_never_prints_negative_zero() {
        // -0.0000004 formats as -0.000000 and trims to -0 — a backward
        // device timestamp must not write "-0 seconds" into the summary.
        assert_eq!(num_text(-1e-7), "0");
        assert_eq!(num_text(319.494), "319.494");
    }

    #[test]
    fn an_unterminated_sheet_tail_is_flagged_truncated() {
        // A close-delimited reply cut mid-record parses cleanly - the
        // missing final newline is the only signal it was cut.
        let clean = parse_sheet(
            b"a,b
1,2
",
        )
        .unwrap();
        assert!(!clean.truncated);
        let cut = parse_sheet(
            b"a,b
1,2
3,",
        )
        .unwrap();
        assert!(cut.truncated);
    }
}
