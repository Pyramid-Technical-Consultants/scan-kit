//! Plan Runner status: what the operator may press, and how live IO is shown.
//!
//! The WebSocket session that talks to an RCI lives in scan-kit-io.

use serde_json::{json, Value};

pub const STATE: &str = "rci/rci_controller/state";
pub const PROGRESS: &str = "rci/rci_controller/progress";
pub const POINT_PROGRESS: &str = "rci/rci_controller/point_progress";
pub const CONTROL_POINT_INDEX: &str = "rci/rci_controller/control_point_index";
pub const CONTROL_POINT_COUNT: &str = "rci/rci_controller/control_point_count";
pub const POINT_ENERGY: &str = "rci/rci_controller/point_energy";
pub const POINT_LAYER_ID: &str = "rci/rci_controller/point_layer_id";
pub const TIME_ELAPSED: &str = "rci/rci_controller/time_elapsed";
pub const TREATMENT_ACTIVE: &str = "rci/rci_controller/treatment_active";
pub const READY_PERMIT: &str = "rci/rci_controller/ready_permit";
pub const READY_PERMIT_REASON: &str = "rci/rci_controller/ready_permit/revoke_reason";
pub const POINTS_VALID: &str = "rci/rci_controller/points_valid";
pub const POINTS_LOAD_STATE: &str = "rci/control_point_table/points_load_state";
pub const POINTS_UPLOAD: &str = "rci/control_point_table/points_upload";
pub const POINTS_UPLOAD_TARGET: &str = "rci/control_point_table/points_upload/target";
pub const START_BUTTON: &str = "rci/rci_controller/start_button";
pub const PAUSE_BUTTON: &str = "rci/rci_controller/pause_button";
pub const STOP_BUTTON: &str = "rci/rci_controller/stop_button";
pub const RESET_BUTTON: &str = "rci/rci_controller/reset_button";
pub const COMBINED_STATE: &str = "rci/map_manager/combined_state";
pub const COMBINED_POINTS_OK: &str = "rci/map_manager/combined_points_ok";
pub const COMBINED_START_PERMIT: &str = "rci/map_manager/combined_start_permit";
pub const COMBINED_STOP_PERMIT: &str = "rci/map_manager/combined_stop_permit";
pub const SESSION_DIRECTORY: &str = "admin/storage/session/directory";
pub const DEFAULT_CONTROL_POINTS: &str = "/root/config/control/control_points.csv";
pub const DEFAULT_SESSION_ROOT: &str = "/root/reports/session";
pub const LOAD_STATE_SUCCESS: i64 = 2;

pub const STATUS_PATHS: &[&str] = &[
    STATE,
    PROGRESS,
    POINT_PROGRESS,
    CONTROL_POINT_INDEX,
    CONTROL_POINT_COUNT,
    POINT_ENERGY,
    POINT_LAYER_ID,
    TIME_ELAPSED,
    TREATMENT_ACTIVE,
    READY_PERMIT,
    READY_PERMIT_REASON,
    POINTS_VALID,
    POINTS_LOAD_STATE,
    COMBINED_START_PERMIT,
    COMBINED_STOP_PERMIT,
    COMBINED_STATE,
    COMBINED_POINTS_OK,
];

pub fn field_subscribe_key(io_path: &str) -> String {
    let seg = io_path.trim_matches('/');
    if seg.is_empty() {
        return "/value".into();
    }
    if seg.ends_with("/value") {
        format!("/{seg}")
    } else {
        format!("/{seg}/value")
    }
}

pub fn parse_host(host: &str) -> Result<String, String> {
    let text = host.trim();
    if text.is_empty() {
        return Err("host is required".into());
    }
    let authority = if let Some((_, rest)) = text.split_once("://") {
        let authority = rest.split(['/', '?', '#']).next().unwrap_or("");
        if authority.is_empty() {
            return Err(format!("invalid host: {host}"));
        }
        authority.rsplit('@').next().unwrap_or(authority)
    } else {
        text.split('/').next().unwrap_or(text)
    };
    if authority.is_empty() {
        return Err(format!("invalid host: {host}"));
    }
    Ok(authority.to_owned())
}

pub fn normalize_host(host: &str) -> Result<String, String> {
    let host = parse_host(host)?;
    if host.contains(':') {
        Ok(host)
    } else {
        Ok(format!("{host}:80"))
    }
}

pub fn io_url(host: &str, path: &str, file_name: &str) -> Result<String, String> {
    let base = format!("http://{}/io", normalize_host(host)?);
    let path = path.trim_matches('/');
    if path.is_empty() {
        Ok(format!("{base}/{file_name}"))
    } else {
        Ok(format!("{base}/{path}/{file_name}"))
    }
}

pub fn device_file_url(host: &str, device_path: &str) -> Result<String, String> {
    let path = if device_path.starts_with('/') {
        device_path.to_owned()
    } else {
        format!("/{device_path}")
    };
    Ok(format!("http://{}{path}", normalize_host(host)?))
}

pub fn unwrap_io(value: &Value) -> &Value {
    match value {
        Value::Array(items) if !items.is_empty() => unwrap_io(&items[0]),
        other => other,
    }
}

pub fn io_bool(value: &Value) -> bool {
    match unwrap_io(value) {
        Value::Bool(flag) => *flag,
        Value::Number(number) => number.as_f64().unwrap_or(0.0) != 0.0,
        Value::String(text) => matches!(
            text.trim().to_ascii_lowercase().as_str(),
            "true" | "1" | "yes" | "granted"
        ),
        _ => false,
    }
}

pub fn io_number(value: &Value) -> Option<f64> {
    match unwrap_io(value) {
        Value::Number(number) => number.as_f64(),
        Value::String(text) if !text.trim().is_empty() => text.trim().parse().ok(),
        _ => None,
    }
}

pub fn io_text(value: &Value) -> String {
    match unwrap_io(value) {
        Value::Null => String::new(),
        Value::String(text) => text.trim().to_owned(),
        Value::Number(number) => number.to_string(),
        Value::Bool(flag) => flag.to_string(),
        other => other.to_string(),
    }
}

pub fn progress_percent(value: &Value) -> Option<f64> {
    let number = io_number(value)?;
    if (0.0..=1.0).contains(&number) {
        Some(number * 100.0)
    } else {
        Some(number)
    }
}

pub fn control_enables(status: &Value, connected: bool) -> Value {
    if !connected {
        return json!({"start": false, "pause": false, "stop": false, "reset": false});
    }
    let state = io_text(&status[STATE]).to_ascii_lowercase();
    let running = matches!(state.as_str(), "dosing" | "active" | "running");
    let paused = state == "paused";
    json!({
        "start": io_bool(&status[READY_PERMIT]) && !running,
        "pause": running,
        "stop": running || paused,
        "reset": !running,
    })
}

pub fn point_fraction(status: &Value) -> String {
    match (
        display_count(&status[CONTROL_POINT_INDEX]),
        display_count(&status[CONTROL_POINT_COUNT]),
    ) {
        (Some(index), Some(count)) => format!("{index} / {count}"),
        _ => "—".into(),
    }
}

pub fn format_energy(value: &Value) -> String {
    match io_number(value) {
        Some(number) => format!("{number:.2} MeV"),
        None => {
            let raw = unwrap_io(value);
            if raw.is_null() || io_text(raw).is_empty() {
                "—".into()
            } else {
                io_text(raw)
            }
        }
    }
}

pub fn format_elapsed(value: &Value) -> String {
    let Some(mut seconds) = io_number(value) else {
        let raw = unwrap_io(value);
        return if raw.is_null() || io_text(raw).is_empty() {
            "—".into()
        } else {
            io_text(raw)
        };
    };
    if seconds < 0.0 {
        seconds = 0.0;
    }
    if seconds < 60.0 {
        return format!("{seconds:.1} s");
    }
    let total = seconds as u64;
    let hours = total / 3600;
    let minutes = (total % 3600) / 60;
    let secs = total % 60;
    if hours > 0 {
        format!("{hours}:{minutes:02}:{secs:02}")
    } else {
        format!("{minutes}:{secs:02}")
    }
}

pub fn coach_message(connected: bool, has_plan: bool, status: &Value) -> String {
    if !connected {
        return "Enter the RCI IP and click Connect.".into();
    }
    let state = io_text(&status[STATE]).to_ascii_lowercase();
    if matches!(state.as_str(), "dosing" | "active" | "running") {
        return "Running — Pause or Stop if you need to halt.".into();
    }
    if state == "paused" {
        return "Paused. Start to resume, or Stop.".into();
    }
    if matches!(state.as_str(), "fault" | "error") {
        return "Controller fault. Check the RCI, then Reset.".into();
    }
    if state == "completed" {
        return "Run complete. Browse a zip name, then Download.".into();
    }
    if !has_plan {
        return "Connected. Browse to an input_map.csv, then upload it.".into();
    }
    if !io_bool(&status[POINTS_VALID]) && !io_bool(&status[COMBINED_POINTS_OK]) {
        return "CSV selected. Click Upload to RCI to load the plan.".into();
    }
    if io_bool(&status[READY_PERMIT]) {
        return "Start is enabled. Press Start when you are ready to run.".into();
    }
    let reason = io_text(&status[READY_PERMIT_REASON]);
    if !reason.is_empty() {
        return format!("Waiting for ready permit: {reason}");
    }
    "Plan is on the controller. Waiting for the RCI ready permit.".into()
}

pub fn runner_view(status: &Value, connected: bool, has_plan: bool) -> Value {
    let state = io_text(&status[STATE]);
    let combined = unwrap_io(&status[COMBINED_STATE]);
    let mut parts = Vec::new();
    if !combined.is_null() && !io_text(combined).is_empty() && combined != &Value::Bool(false) {
        parts.push(format!("Map manager: {}", io_text(combined)));
    }
    if io_bool(&status[TREATMENT_ACTIVE]) {
        parts.push("Treatment active".into());
    }
    let subtitle = if !connected {
        "Connect, upload a plan, then press Start"
    } else if parts.is_empty() {
        "Waiting for live status"
    } else {
        ""
    };
    let live = status.as_object().is_some_and(|map| !map.is_empty());
    let points = if !live {
        "—"
    } else if io_bool(&status[POINTS_VALID]) || io_bool(&status[COMBINED_POINTS_OK]) {
        "Valid"
    } else {
        "Invalid"
    };
    let permit = if !live {
        "—"
    } else if io_bool(&status[READY_PERMIT]) {
        "Granted"
    } else {
        "Held"
    };
    json!({
        "connected": connected,
        "state": if state.is_empty() { "—".into() } else { state.to_ascii_uppercase() },
        "subtitle": if subtitle.is_empty() { parts.join("  ·  ") } else { subtitle.into() },
        "progress": progress_percent(&status[PROGRESS]),
        "point_progress": progress_percent(&status[POINT_PROGRESS]),
        "point": point_fraction(status),
        "energy": format_energy(&status[POINT_ENERGY]),
        "layer": display_count(&status[POINT_LAYER_ID]).unwrap_or_else(|| "—".into()),
        "elapsed": format_elapsed(&status[TIME_ELAPSED]),
        "permit": permit,
        "points": points,
        "enables": control_enables(status, connected),
        "coach": coach_message(connected, has_plan, status),
    })
}

pub fn resolve_session_download(dest: &str, remote_root: &str) -> Option<(String, String)> {
    let dest = dest.trim();
    if dest.is_empty() || dest.starts_with("/root/") || !dest.to_ascii_lowercase().ends_with(".zip")
    {
        return None;
    }
    let root = remote_root.trim_end_matches('/');
    Some((format!("{root}/{}", zip_stem(dest)), dest.to_owned()))
}

pub fn default_session_zip_path(current: &str, last_folder: &str) -> String {
    let current = current.trim();
    let folder = last_folder.trim();
    if current.starts_with("/root/") {
        let name = format!("{}.zip", file_name(current));
        return join_folder(folder, &name);
    }
    if current.to_ascii_lowercase().ends_with(".zip") {
        return current.to_owned();
    }
    if !current.is_empty() {
        return join_folder(current, "session.zip");
    }
    join_folder(folder, "session.zip")
}

pub fn session_download_hint(dest: &str, remote_root: &str) -> String {
    let dest = dest.trim();
    let root = remote_root.trim_end_matches('/');
    if dest.to_ascii_lowercase().ends_with(".zip") {
        format!("Downloads {root}/{}", zip_stem(dest))
    } else if dest.starts_with("/root/") {
        format!("Remote folder {}", dest.trim_end_matches('/'))
    } else {
        format!("Zip name is the folder under {root}/")
    }
}

pub fn control_button(action: &str) -> Result<&'static str, String> {
    match action {
        "start" => Ok(START_BUTTON),
        "pause" => Ok(PAUSE_BUTTON),
        "stop" => Ok(STOP_BUTTON),
        "reset" => Ok(RESET_BUTTON),
        _ => Err(format!("unknown run action {action}")),
    }
}

fn display_count(value: &Value) -> Option<String> {
    let raw = unwrap_io(value);
    if raw.is_null() || io_text(raw).is_empty() {
        return None;
    }
    match io_number(raw) {
        Some(number) if number.fract() == 0.0 && number.abs() < 1e15 => {
            Some(format!("{}", number as i64))
        }
        Some(_) => Some(io_text(raw)),
        None => Some(io_text(raw)),
    }
}

fn zip_stem(path: &str) -> String {
    let name = file_name(path);
    if let Some(dot) = name.rfind('.') {
        if name[dot..].eq_ignore_ascii_case(".zip") {
            return name[..dot].to_owned();
        }
    }
    name
}

fn file_name(path: &str) -> String {
    path.trim_end_matches(['/', '\\'])
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or(path)
        .to_owned()
}

fn join_folder(folder: &str, name: &str) -> String {
    if folder.is_empty() {
        return name.to_owned();
    }
    let folder = folder.trim_end_matches(['/', '\\']);
    if folder.contains('\\') {
        format!("{folder}\\{name}")
    } else {
        format!("{folder}/{name}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hosts_drop_the_browser_path_and_gain_port_80() {
        assert_eq!(parse_host("192.168.100.184").unwrap(), "192.168.100.184");
        assert_eq!(
            parse_host("http://192.168.100.184/io/").unwrap(),
            "192.168.100.184"
        );
        assert_eq!(
            parse_host("http://192.168.100.184:8080/io/").unwrap(),
            "192.168.100.184:8080"
        );
        assert_eq!(normalize_host("192.168.1.1").unwrap(), "192.168.1.1:80");
        assert_eq!(
            io_url("10.0.0.1", "admin/version", "value.json").unwrap(),
            "http://10.0.0.1:80/io/admin/version/value.json"
        );
        assert_eq!(
            device_file_url("10.0.0.1", "/root/config/control_points.csv").unwrap(),
            "http://10.0.0.1:80/root/config/control_points.csv"
        );
        assert_eq!(field_subscribe_key("admin/version"), "/admin/version/value");
    }

    #[test]
    fn pairs_unwrap_and_progress_scales_a_fraction() {
        assert_eq!(unwrap_io(&json!([[true, 1.0], [false, 2.0]])), &json!(true));
        assert_eq!(progress_percent(&json!(0.25)), Some(25.0));
        assert_eq!(progress_percent(&json!(45.0)), Some(45.0));
        assert_eq!(progress_percent(&json!([0.4, 99.0])), Some(40.0));
        assert_eq!(point_fraction(&json!({})), "—");
        assert_eq!(
            point_fraction(&json!({CONTROL_POINT_INDEX: 3, CONTROL_POINT_COUNT: 7600})),
            "3 / 7600"
        );
        assert!(io_bool(&json!("granted")));
        assert!(!io_bool(&json!("false")));
        assert_eq!(io_number(&json!([226.5, 1.0])), Some(226.5));
        assert_eq!(format_energy(&json!([226.5, 1.0])), "226.50 MeV");
        assert_eq!(format_elapsed(&json!(12.3)), "12.3 s");
        assert_eq!(format_elapsed(&json!([75.2, 1.0])), "1:15");
        assert_eq!(format_elapsed(&json!(3725)), "1:02:05");
    }

    #[test]
    fn start_follows_the_permit_and_a_run_locks_reset() {
        let ready = json!({READY_PERMIT: true});
        assert_eq!(control_enables(&ready, true)["start"], true);
        let dosing = json!({STATE: "dosing", READY_PERMIT: true, POINTS_VALID: true});
        let enables = control_enables(&dosing, true);
        assert_eq!(enables["start"], false);
        assert_eq!(enables["pause"], true);
        assert_eq!(enables["reset"], false);
        let paused = json!({STATE: ["paused", 1.0], READY_PERMIT: [true, 1.0]});
        let enables = control_enables(&paused, true);
        assert_eq!(enables["start"], true);
        assert_eq!(enables["stop"], true);
        assert!(control_enables(&json!({}), false)["start"] == false);
    }

    #[test]
    fn the_coach_names_the_next_action() {
        assert!(coach_message(false, false, &json!({})).contains("Connect"));
        assert!(coach_message(true, false, &json!({})).contains("Browse"));
        assert!(coach_message(true, true, &json!({POINTS_VALID: false})).contains("Upload"));
        assert!(
            coach_message(true, true, &json!({POINTS_VALID: true, READY_PERMIT: true}))
                .contains("Start is enabled")
        );
        assert!(coach_message(true, true, &json!({STATE: "dosing"})).starts_with("Running"));
        assert!(coach_message(true, true, &json!({STATE: "completed"})).contains("Download"));
        assert!(coach_message(
            true,
            true,
            &json!({POINTS_VALID: true, READY_PERMIT: false, READY_PERMIT_REASON: ["Door open", 1.0]})
        )
        .contains("Door open"));
        let remote =
            resolve_session_download(r"C:\data\abc.zip", "/root/reports/session/").unwrap();
        assert_eq!(remote.0, "/root/reports/session/abc");
        assert!(
            resolve_session_download("/root/reports/session/abc", "/root/reports/session")
                .is_none()
        );
        assert_eq!(default_session_zip_path("", ""), "session.zip");
        assert!(
            session_download_hint(r"C:\data\abc.zip", "/root/reports/session").contains("/abc")
        );
    }
}
