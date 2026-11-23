//! Plan Runner session: one mpack WebSocket plus HTTP file transfer.
//!
//! ponytail: the HTTP client understands Content-Length and chunked bodies, and
//! stops after 64 MiB. A device that streams a larger session file needs a
//! ranged or streaming reader.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant};

use scan_kit_core::{
    control_button, device_file_url, field_subscribe_key, io_bool, io_number, io_url,
    normalize_host, parse_host, runner_view, session_zip_dest, DEFAULT_CONTROL_POINTS,
    DEFAULT_SESSION_ROOT, LOAD_STATE_SUCCESS, POINTS_LOAD_STATE, POINTS_UPLOAD,
    POINTS_UPLOAD_TARGET, POINTS_VALID, SESSION_DIRECTORY, STATUS_PATHS,
};
use serde_json::{json, Map, Value};
use tungstenite::client::{client, IntoClientRequest};
use tungstenite::http::HeaderValue;
use tungstenite::{Message, WebSocket};

use crate::session_capture;
use crate::store::{self, with_store};

const MAX_BODY: usize = 64 * 1024 * 1024;
const MAX_ZIP_WALK_DEPTH: usize = 32;

struct Live {
    /// Bumped per connect — an upload window that re-locks LIVE after a
    /// disconnect+connect must fail instead of commanding the new device.
    epoch: u64,
    host: String,
    socket: WebSocket<DeadlineStream>,
    status: Map<String, Value>,
    version: Value,
    device_type: Value,
    session_directory: String,
    session_id: String,
    plan_path: Option<PathBuf>,
}

static LIVE: Mutex<Option<Live>> = Mutex::new(None);
static EPOCH: AtomicU64 = AtomicU64::new(0);
/// One upload at a time — two concurrent uploads PUT to the same device path
/// and the loser's set_field can command the winner's file.
static UPLOAD_BUSY: AtomicBool = AtomicBool::new(false);
/// One connect at a time, and any disconnect during the open phase must land —
/// the open phase runs unlocked for seconds, so a no-op disconnect that the
/// install then overwrites leaves a session nobody asked for.
static CONNECT_BUSY: AtomicBool = AtomicBool::new(false);
static DISCONNECTS: AtomicU64 = AtomicU64::new(0);

pub fn catalog(db: &Path) -> Value {
    let (host, file_dir) = with_store(db, |conn| store::runner_settings(conn))
        .ok()
        .unwrap_or((None, None));
    let session_id = lock()
        .as_ref()
        .map(|live| live.session_id.clone())
        .unwrap_or_default();
    let connected = live();
    let mut body = json!({
        "host": host,
        "file_dir": file_dir,
        "dest": scan_kit_core::default_session_zip_path(
            "",
            file_dir.as_deref().unwrap_or(""),
            &session_id,
        ),
        "connected": connected,
        "view": runner_view(&json!({}), false, false),
    });
    if connected {
        match read_status(false, "") {
            Ok(view) => {
                body["view"] = view;
                body["connected"] = json!(true);
            }
            // A dead socket is not "connected" even while live() still lists it.
            Err(_) => body["connected"] = json!(false),
        }
    }
    body
}

/// Releases a busy flag on drop, including the panic path.
struct BusyFlag(&'static AtomicBool);

impl Drop for BusyFlag {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

fn acquire(flag: &'static AtomicBool, busy: &str) -> Result<BusyFlag, String> {
    if flag.swap(true, Ordering::Acquire) {
        return Err(busy.into());
    }
    Ok(BusyFlag(flag))
}

pub fn connect(db: &Path, host: &str) -> Result<Value, String> {
    let _connect = acquire(&CONNECT_BUSY, "a connect is already in progress")?;
    let host = parse_host(host)?;
    disconnect();
    let generation = DISCONNECTS.load(Ordering::Acquire);
    let mut socket = open_socket(&host)?;
    send(&mut socket, "config", Some(config_body()))?;
    let status = read_paths(&mut socket, STATUS_PATHS, Duration::from_secs(8))?;
    // Each phase can stall on a trickling peer for a full read deadline —
    // a disconnect between phases must still be able to abort the connect.
    if DISCONNECTS.load(Ordering::Acquire) != generation {
        return Err("connection changed".into());
    }
    let version = http_json(&host, "admin/version").unwrap_or(Value::Null);
    let device_type = http_json(&host, "admin/device_type").unwrap_or(Value::Null);
    let session_directory = http_json(&host, SESSION_DIRECTORY)
        .and_then(|value| value.as_str().map(str::to_owned))
        .unwrap_or_else(|| DEFAULT_SESSION_ROOT.into());
    let session_id = session_capture::live_session_id(&host).unwrap_or_default();
    if DISCONNECTS.load(Ordering::Acquire) != generation {
        return Err("connection changed".into());
    }
    let plan_path = with_store(db, |conn| store::runner_plan_path(conn))
        .ok()
        .flatten()
        .map(PathBuf::from)
        .filter(|path| path.is_file());
    {
        let mut guard = lock();
        // A disconnect (or a queued connect's own disconnect) ran during the
        // open phase — it must not be overwritten by this install.
        if DISCONNECTS.load(Ordering::Acquire) != generation {
            return Err("connection changed".into());
        }
        *guard = Some(Live {
            epoch: EPOCH.fetch_add(1, Ordering::Relaxed) + 1,
            host: host.clone(),
            socket,
            status,
            version: version.clone(),
            device_type: device_type.clone(),
            session_directory: session_directory.clone(),
            session_id: session_id.clone(),
            plan_path,
        });
    }
    let _ = with_store(db, |conn| store::set_runner_host(conn, &host));
    Ok(connected_body(
        &host,
        &version,
        &device_type,
        &session_directory,
        &session_id,
    ))
}

pub fn disconnect() {
    DISCONNECTS.fetch_add(1, Ordering::AcqRel);
    // The close frame can block on a half-dead peer for the write timeout —
    // drop the LIVE guard first or every serialized call stalls behind it.
    let live = lock().take();
    if let Some(mut live) = live {
        let _ = live.socket.close(None);
    }
}

pub fn read_status(has_plan: bool, dest: &str) -> Result<Value, String> {
    // The session-id fetch is a plain HTTP GET — it must not run under LIVE,
    // which serializes disconnect/control/upload. A stalled file server would
    // otherwise block even disconnecting for the full request timeout.
    let host = {
        let guard = lock();
        guard.as_ref().ok_or("not connected")?.host.clone()
    };
    let session_id = session_capture::live_session_id(&host);
    let mut guard = lock();
    let live = guard.as_mut().ok_or("not connected")?;
    // A disconnect/reconnect between the two locks could change the device.
    if live.host == host {
        if let Some(id) = session_id {
            live.session_id = id;
        }
    }
    refresh(live, Duration::from_millis(500))?;
    Ok(view_of(live, has_plan, dest))
}

pub fn upload(db: &Path, csv_path: &Path) -> Result<Value, String> {
    let _upload = acquire(&UPLOAD_BUSY, "an upload is already in progress")?;
    if !csv_path.is_file() {
        return Err(format!("CSV not found: {}", csv_path.display()));
    }
    // The caller names the file — cap it like the response body so a stray
    // multi-GB path cannot exhaust memory or land on the device as a plan.
    let mut file = std::fs::File::open(csv_path).map_err(|err| err.to_string())?;
    let mut bytes = Vec::new();
    std::io::Read::take(&mut file, MAX_BODY as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|err| err.to_string())?;
    if bytes.len() > MAX_BODY {
        return Err(format!("CSV is too large: {} bytes", bytes.len()));
    }
    // Each socket exchange holds LIVE; the HTTP put and the poll sleeps do
    // not need it. Holding it across the whole upload froze status reads and
    // control actions for up to ~40s.
    let (host, epoch, target) = {
        let mut guard = lock();
        let live = guard.as_mut().ok_or("not connected")?;
        let epoch = live.epoch;
        set_field(
            &mut live.socket,
            POINTS_UPLOAD_TARGET,
            &json!(DEFAULT_CONTROL_POINTS),
        )?;
        // The settle wait does not need the socket — release LIVE so status
        // reads and control actions are not frozen behind it.
        drop(guard);
        thread::sleep(Duration::from_millis(200));
        let mut guard = lock();
        let live = guard
            .as_mut()
            .filter(|live| live.epoch == epoch)
            .ok_or("connection changed")?;
        let read = read_one(
            &mut live.socket,
            POINTS_UPLOAD_TARGET,
            Duration::from_secs(4),
        )?;
        let target = read
            .as_str()
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .unwrap_or(DEFAULT_CONTROL_POINTS)
            .to_owned();
        (live.host.clone(), live.epoch, target)
    };
    http_put(&host, &target, &bytes)?;
    // A disconnect+connect between windows swaps the Live — commanding the
    // new device with the old one's session is worse than failing.
    {
        let mut guard = lock();
        let live = guard
            .as_mut()
            .filter(|live| live.epoch == epoch)
            .ok_or("connection changed")?;
        set_field(&mut live.socket, POINTS_UPLOAD, &json!(true))?;
    }
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let code = {
            let mut guard = lock();
            let live = guard
                .as_mut()
                .filter(|live| live.epoch == epoch)
                .ok_or("connection changed")?;
            let state = read_one(&mut live.socket, POINTS_LOAD_STATE, Duration::from_secs(2))?;
            io_number(&state).map(|value| value as i64)
        };
        if code == Some(LOAD_STATE_SUCCESS) {
            break;
        }
        if let Some(code) = code {
            if code != 0 && code != 1 {
                return Err(format!("points_load_state error: {code}"));
            }
        }
        if Instant::now() >= deadline {
            return Err("timed out waiting for points_load_state success".into());
        }
        thread::sleep(Duration::from_millis(200));
    }
    {
        let mut guard = lock();
        let live = guard
            .as_mut()
            .filter(|live| live.epoch == epoch)
            .ok_or("connection changed")?;
        let valid = read_one(&mut live.socket, POINTS_VALID, Duration::from_secs(8))?;
        subscribe_status(&mut live.socket)?;
        if !io_bool(&valid) {
            return Err("points_valid is false after upload".into());
        }
        live.status.insert(POINTS_VALID.to_owned(), valid);
        live.plan_path = Some(csv_path.to_owned());
    }
    if let Some(parent) = csv_path.parent() {
        let _ = with_store(db, |conn| {
            store::set_runner_upload(
                conn,
                &parent.display().to_string(),
                &csv_path.display().to_string(),
            )
        });
    }
    let mut body = read_status(true, "")?;
    // A reconnect during read_status would attach "Plan uploaded" to the new
    // device's view — the epoch check must run after the final HTTP window.
    if lock().as_ref().map(|live| live.epoch) != Some(epoch) {
        return Err("connection changed".into());
    }
    body["target"] = json!(target);
    body["message"] = json!(format!("Plan uploaded to {target}."));
    Ok(body)
}

pub fn control(action: &str) -> Result<Value, String> {
    let button = control_button(action)?;
    {
        let mut guard = lock();
        let live = guard.as_mut().ok_or("not connected")?;
        set_field(&mut live.socket, button, &json!(true))?;
    }
    read_status(true, "")
}

pub fn remember_dir(db: &Path, dir: &str) -> Result<Value, String> {
    with_store(db, |conn| store::set_runner_file_dir(conn, dir))?;
    Ok(json!({ "file_dir": dir }))
}

pub fn download(db: &Path, dest: &str) -> Result<Value, String> {
    download_with(db, dest, &mut |_, _| true)
}

pub fn download_with(
    db: &Path,
    dest: &str,
    keep: &mut dyn FnMut(u64, u64) -> bool,
) -> Result<Value, String> {
    let (host, status, plan_path) = {
        let guard = lock();
        let live = guard.as_ref().ok_or("not connected")?;
        (
            live.host.clone(),
            live.status.clone(),
            live.plan_path.clone(),
        )
    };
    let zip_path = session_zip_dest(dest).ok_or("Choose a local .zip path for the session.")?;
    // Unique per call — two overlapping downloads to the same folder must not
    // delete each other's staging tree mid-capture.
    static NEXT_STAGING: AtomicU64 = AtomicU64::new(0);
    let staging = PathBuf::from(&zip_path)
        .parent()
        .unwrap_or(Path::new("."))
        .join(format!(
            ".session_staging-{}-{}",
            std::process::id(),
            NEXT_STAGING.fetch_add(1, Ordering::Relaxed)
        ));
    let session_dir = staging.join("session");
    let _ = std::fs::remove_dir_all(&staging);
    std::fs::create_dir_all(&session_dir).map_err(|err| err.to_string())?;
    let result = (|| {
        let report =
            session_capture::capture(&host, &status, plan_path.as_deref(), &session_dir, keep)?;
        if !keep(report.files as u64, report.files as u64) {
            return Err("cancelled".into());
        }
        package_zip(&session_dir, Path::new(&zip_path), &report.session_id, keep)?;
        if !keep(report.files as u64, report.files as u64) {
            let _ = std::fs::remove_file(&zip_path);
            return Err("cancelled".into());
        }
        Ok::<(String, session_capture::CaptureResult), String>((zip_path.clone(), report))
    })();
    let _ = std::fs::remove_dir_all(&staging);
    let (zip_path, report) = result?;
    if let Some(parent) = Path::new(&zip_path).parent() {
        let _ = with_store(db, |conn| {
            store::set_runner_file_dir(conn, &parent.display().to_string())
        });
    }
    let detail = if report.warnings.is_empty() {
        String::new()
    } else {
        format!(
            " ({} warnings: {})",
            report.warnings.len(),
            report.warnings.join("; ")
        )
    };
    Ok(json!({
        "path": zip_path,
        "session_id": report.session_id,
        "files": report.files,
        "warnings": report.warnings,
        "message": format!("Session {} saved to {zip_path}{detail}", report.session_id),
    }))
}

fn connected_body(
    host: &str,
    version: &Value,
    device_type: &Value,
    session_directory: &str,
    session_id: &str,
) -> Value {
    let guard = lock();
    let status = guard
        .as_ref()
        .map(|live| Value::Object(live.status.clone()))
        .unwrap_or_else(|| json!({}));
    let mut view = runner_view(&status, true, false);
    view["host"] = json!(host);
    view["version"] = json!(display_io(version));
    view["device_type"] = json!(display_io(device_type));
    view["session_directory"] = json!(session_directory);
    view["session_id"] = json!(session_id);
    view
}

fn view_of(live: &Live, has_plan: bool, dest: &str) -> Value {
    let mut view = runner_view(&Value::Object(live.status.clone()), true, has_plan);
    view["host"] = json!(live.host);
    view["version"] = json!(display_io(&live.version));
    view["device_type"] = json!(display_io(&live.device_type));
    view["session_directory"] = json!(live.session_directory);
    view["session_id"] = json!(live.session_id);
    view["dest"] = json!(session_zip_dest(dest));
    view["hint"] = json!(scan_kit_core::session_download_hint(&live.session_id));
    view
}

fn display_io(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Number(number) => number.to_string(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

fn refresh(live: &mut Live, timeout: Duration) -> Result<(), String> {
    subscribe_status(&mut live.socket)?;
    let updates = poll_keys(
        &mut live.socket,
        &STATUS_PATHS
            .iter()
            .map(|path| field_subscribe_key(path))
            .collect::<Vec<_>>(),
        timeout,
    )?;
    for path in STATUS_PATHS {
        let key = field_subscribe_key(path);
        if let Some(value) = updates.get(&key) {
            live.status.insert((*path).to_owned(), value.clone());
        }
    }
    Ok(())
}

fn read_paths(
    socket: &mut WebSocket<DeadlineStream>,
    paths: &[&str],
    timeout: Duration,
) -> Result<Map<String, Value>, String> {
    subscribe(socket, paths)?;
    let updates = poll_keys(
        socket,
        &paths
            .iter()
            .map(|path| field_subscribe_key(path))
            .collect::<Vec<_>>(),
        timeout,
    )?;
    let mut status = Map::new();
    for path in paths {
        if let Some(value) = updates.get(&field_subscribe_key(path)) {
            status.insert((*path).to_owned(), value.clone());
        }
    }
    Ok(status)
}

fn read_one(
    socket: &mut WebSocket<DeadlineStream>,
    path: &str,
    timeout: Duration,
) -> Result<Value, String> {
    let values = read_paths(socket, &[path], timeout)?;
    values.get(path).cloned().ok_or_else(|| {
        format!(
            "no value for '{}': no update within timeout",
            field_subscribe_key(path)
        )
    })
}

fn subscribe_status(socket: &mut WebSocket<DeadlineStream>) -> Result<(), String> {
    subscribe(socket, STATUS_PATHS)
}

fn subscribe(socket: &mut WebSocket<DeadlineStream>, paths: &[&str]) -> Result<(), String> {
    let mut pairs = Vec::new();
    for path in paths {
        pairs.push((
            rmpv::Value::from(field_subscribe_key(path)),
            rmpv::Value::Boolean(false),
        ));
    }
    send(socket, "subscribe", Some(rmpv::Value::Map(pairs)))
}

fn set_field(
    socket: &mut WebSocket<DeadlineStream>,
    path: &str,
    value: &Value,
) -> Result<(), String> {
    let payload = rmpv::Value::Map(vec![(
        rmpv::Value::from(field_subscribe_key(path)),
        rmp_from_json(value),
    )]);
    send(socket, "set", Some(payload))
}

fn poll_keys(
    socket: &mut WebSocket<DeadlineStream>,
    keys: &[String],
    timeout: Duration,
) -> Result<HashMap<String, Value>, String> {
    let wanted: std::collections::HashSet<&str> = keys.iter().map(String::as_str).collect();
    let deadline = Instant::now() + timeout;
    let mut got = HashMap::new();
    while Instant::now() < deadline && got.len() < wanted.len() {
        send(socket, "get", None)?;
        let remaining = deadline.saturating_duration_since(Instant::now());
        let Some(message) = recv(socket, remaining.min(Duration::from_millis(800)))? else {
            continue;
        };
        if message_event(&message) != "update" {
            continue;
        }
        if let Some(rmpv::Value::Map(pairs)) = message_data(&message) {
            for (key, value) in pairs {
                let Some(key) = key.as_str() else { continue };
                if wanted.contains(key) {
                    got.insert(key.to_owned(), json_of(&unwrap_update(value)));
                }
            }
        }
    }
    Ok(got)
}

fn open_socket(host: &str) -> Result<WebSocket<DeadlineStream>, String> {
    let endpoint = normalize_host(host)?;
    let mut last = "connection failed".to_owned();
    for attempt in 0..2 {
        if attempt > 0 {
            thread::sleep(Duration::from_millis(500));
        }
        match dial(&endpoint) {
            Ok(socket) => return Ok(socket),
            Err(err) => last = err,
        }
    }
    Err(last)
}

/// A TcpStream whose reads fail once an absolute deadline passes. The
/// per-syscall read timeout bounds each frame, but tungstenite assembles a
/// fragmented message inside one `read()` — a peer trickling continuation
/// frames one timeout apart would otherwise hold LIVE forever.
#[derive(Debug)]
struct DeadlineStream {
    stream: TcpStream,
    deadline: Instant,
}

impl DeadlineStream {
    fn new(stream: TcpStream, deadline: Instant) -> DeadlineStream {
        DeadlineStream { stream, deadline }
    }

    fn set_deadline(&mut self, deadline: Instant) {
        self.deadline = deadline;
    }
}

impl std::io::Read for DeadlineStream {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let Some(remaining) = self.deadline.checked_duration_since(Instant::now()) else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "websocket message deadline",
            ));
        };
        // Without the cap a fragment arriving just before the deadline would
        // let the last syscall block a full read-timeout past it. The floor
        // keeps a sub-millisecond remainder from erroring the socket call.
        self.stream
            .set_read_timeout(Some(remaining.max(Duration::from_millis(1))))?;
        self.stream.read(buf)
    }
}

impl std::io::Write for DeadlineStream {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        // socket.read() writes internally (auto-pong) — the armed deadline
        // bounds those writes too.
        let Some(remaining) = self.deadline.checked_duration_since(Instant::now()) else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "websocket message deadline",
            ));
        };
        self.stream
            .set_write_timeout(Some(remaining.max(Duration::from_millis(1))))?;
        self.stream.write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.stream.flush()
    }
}

fn dial(endpoint: &str) -> Result<WebSocket<DeadlineStream>, String> {
    let addrs: Vec<_> = endpoint
        .to_socket_addrs()
        .map_err(|err| err.to_string())?
        .collect();
    if addrs.is_empty() {
        return Err(format!("invalid host: {endpoint}"));
    }
    // Every record gets a shot — the first address can be dead or accept
    // TCP but never complete the upgrade while a sibling record answers.
    let mut last_err = String::new();
    for addr in addrs {
        let stream = match TcpStream::connect_timeout(&addr, Duration::from_secs(10)) {
            Ok(stream) => stream,
            Err(err) => {
                last_err = err.to_string();
                continue;
            }
        };
        stream.set_nodelay(true).ok();
        // The upgrade handshake reads the 101 response on the bare stream —
        // without a read deadline a peer that accepts but never upgrades
        // wedges connect() forever. recv() re-arms per call afterward.
        let _ = stream.set_read_timeout(Some(Duration::from_secs(10)));
        let _ = stream.set_write_timeout(Some(Duration::from_secs(10)));
        let stream = DeadlineStream::new(stream, Instant::now() + Duration::from_secs(10));
        let mut request = format!("ws://{endpoint}")
            .into_client_request()
            .map_err(|err| err.to_string())?;
        const OFFERED: &str = "mpack.v2, mpack.v1, mpack";
        request
            .headers_mut()
            .insert("Sec-WebSocket-Protocol", HeaderValue::from_static(OFFERED));
        match client(request, stream) {
            Ok((socket, response)) => {
                let protocol = response
                    .headers()
                    .get("sec-websocket-protocol")
                    .and_then(|value| value.to_str().ok())
                    .unwrap_or("");
                // The negotiated token must be one we offered — a substring
                // match would accept a foreign `notmpack` protocol.
                if !protocol.split(',').any(|token| {
                    OFFERED
                        .split(',')
                        .any(|offered| token.trim().eq_ignore_ascii_case(offered.trim()))
                }) {
                    return Err(format!(
                        "device did not accept mpack subprotocol (got '{protocol}')"
                    ));
                }
                // No extension was requested — a negotiated one would
                // produce frames this client cannot parse.
                if response.headers().get("sec-websocket-extensions").is_some() {
                    return Err("device negotiated an unrequested websocket extension".into());
                }
                return Ok(socket);
            }
            Err(err) => last_err = err.to_string(),
        }
    }
    Err(if last_err.is_empty() {
        format!("could not connect to {endpoint}")
    } else {
        last_err
    })
}

fn config_body() -> rmpv::Value {
    rmpv::Value::Map(vec![
        (
            rmpv::Value::from("use_short_id"),
            rmpv::Value::Boolean(false),
        ),
        (
            rmpv::Value::from("partial_row_update"),
            rmpv::Value::Boolean(false),
        ),
        (
            rmpv::Value::from("update_arrays_as_blobs"),
            rmpv::Value::Boolean(true),
        ),
        (
            rmpv::Value::from("update_buffered_as_blobs"),
            rmpv::Value::Boolean(true),
        ),
    ])
}

fn send(
    socket: &mut WebSocket<DeadlineStream>,
    event: &str,
    data: Option<rmpv::Value>,
) -> Result<(), String> {
    let mut pairs = vec![(rmpv::Value::from("event"), rmpv::Value::from(event))];
    if let Some(data) = data {
        pairs.push((rmpv::Value::from("data"), data));
    }
    let mut bytes = Vec::new();
    rmpv::encode::write_value(&mut bytes, &rmpv::Value::Map(pairs))
        .map_err(|err| err.to_string())?;
    // recv() is the only other place that arms the deadline — a quiet
    // device leaves it lapsed and every later send fails permanently.
    socket
        .get_mut()
        .set_deadline(Instant::now() + Duration::from_secs(10));
    socket
        .send(Message::Binary(bytes.into()))
        .map_err(|err| err.to_string())
}

fn recv(
    socket: &mut WebSocket<DeadlineStream>,
    timeout: Duration,
) -> Result<Option<rmpv::Value>, String> {
    let stream = socket.get_mut();
    // The deadline bounds the whole assembled message — each syscall inside
    // it is capped to whatever time remains.
    stream.set_deadline(Instant::now() + timeout.max(Duration::from_millis(20)));
    match socket.read() {
        Ok(Message::Binary(bytes)) => {
            let mut cursor = std::io::Cursor::new(bytes.as_ref());
            let value = rmpv::decode::read_value(&mut cursor).map_err(|err| err.to_string())?;
            // One message per frame — trailing bytes or a second value
            // silently desynchronize the protocol.
            if cursor.position() != bytes.len() as u64 {
                return Err("malformed frame: trailing bytes".into());
            }
            Ok(Some(value))
        }
        Ok(Message::Close(_)) => Err("connection closed".into()),
        Ok(_) => Ok(None),
        Err(tungstenite::Error::Io(err))
            if err.kind() == std::io::ErrorKind::TimedOut
                || err.kind() == std::io::ErrorKind::WouldBlock =>
        {
            Ok(None)
        }
        Err(err) => Err(err.to_string()),
    }
}

fn message_event(value: &rmpv::Value) -> &str {
    map_get(value, "event")
        .and_then(rmpv::Value::as_str)
        .unwrap_or("")
}

fn message_data(value: &rmpv::Value) -> Option<&rmpv::Value> {
    map_get(value, "data")
}

fn map_get<'a>(value: &'a rmpv::Value, key: &str) -> Option<&'a rmpv::Value> {
    let rmpv::Value::Map(pairs) = value else {
        return None;
    };
    pairs
        .iter()
        .find(|(name, _)| name.as_str() == Some(key))
        .map(|(_, item)| item)
}

fn unwrap_update(value: &rmpv::Value) -> rmpv::Value {
    if let Some(last) = history_last(value) {
        return last;
    }
    // A {$t,$d} map decodes to the full array — it IS the value. Letting it
    // fall to the [v, ts] unwrap below collapses every array field to its
    // first element.
    if let rmpv::Value::Map(pairs) = value {
        if pairs.iter().any(|(key, _)| key.as_str() == Some("$d")) {
            // A history map whose $ts blob is too short fell through
            // history_last — the raw array's first element is the OLDEST
            // sample, not the current one.
            if pairs.iter().any(|(key, _)| key.as_str() == Some("$ts")) {
                return rmpv::Value::Nil;
            }
            return decode_blobs(value);
        }
    }
    let decoded = decode_blobs(value);
    let rmpv::Value::Array(items) = &decoded else {
        return decoded;
    };
    if items.is_empty() {
        return decoded;
    }
    if let rmpv::Value::Array(inner) = &items[0] {
        let Some(rmpv::Value::Array(last)) = items.last() else {
            return decoded;
        };
        if let Some(first) = last.first() {
            return first.clone();
        }
        let _ = inner;
        return decoded;
    }
    if items.len() >= 2 {
        return items[0].clone();
    }
    decoded
}

fn history_last(value: &rmpv::Value) -> Option<rmpv::Value> {
    let pairs = match value {
        rmpv::Value::Map(pairs) => pairs,
        _ => return None,
    };
    if !pairs.iter().any(|(key, _)| key.as_str() == Some("$ts")) {
        return None;
    }
    let dtype = pairs
        .iter()
        .find(|(key, _)| key.as_str() == Some("$t"))
        .map(|(_, item)| item);
    let data = pairs
        .iter()
        .find(|(key, _)| key.as_str() == Some("$d"))
        .and_then(|(_, item)| item.as_slice());
    let times = pairs
        .iter()
        .find(|(key, _)| key.as_str() == Some("$ts"))
        .and_then(|(_, item)| item.as_slice());
    let values = unpack_blob(dtype.and_then(rmpv::Value::as_i64), data?)?;
    let times = times?;
    // A $ts blob that is not exactly values*8 bytes is corrupt, not "long
    // enough" — a trailing partial timestamp must not pass as a whole one.
    if times.len() != values.len() * 8 || values.is_empty() {
        return None;
    }
    values.last().copied().map(rmpv::Value::from)
}

fn decode_blobs(value: &rmpv::Value) -> rmpv::Value {
    match value {
        rmpv::Value::Map(pairs) => {
            let dtype = pairs
                .iter()
                .find(|(key, _)| key.as_str() == Some("$t"))
                .and_then(|(_, item)| item.as_i64());
            let data = pairs
                .iter()
                .find(|(key, _)| key.as_str() == Some("$d"))
                .and_then(|(_, item)| item.as_slice());
            if let Some(numbers) = unpack_blob(dtype, data.unwrap_or(&[])) {
                if pairs.iter().any(|(key, _)| key.as_str() == Some("$t")) && data.is_some() {
                    return rmpv::Value::Array(
                        numbers.into_iter().map(rmpv::Value::from).collect(),
                    );
                }
            }
            rmpv::Value::Map(
                pairs
                    .iter()
                    .map(|(key, item)| (key.clone(), decode_blobs(item)))
                    .collect(),
            )
        }
        rmpv::Value::Array(items) => rmpv::Value::Array(items.iter().map(decode_blobs).collect()),
        other => other.clone(),
    }
}

fn unpack_blob(dtype: Option<i64>, data: &[u8]) -> Option<Vec<f64>> {
    // A partial trailing element means a corrupt frame — refuse rather than
    // silently drop it and hand a plausible-but-short array downstream.
    match dtype? {
        1 => {
            let (chunks, tail) = data.as_chunks::<4>();
            if !tail.is_empty() {
                return None;
            }
            Some(
                chunks
                    .iter()
                    .map(|chunk| f64::from(f32::from_le_bytes(*chunk)))
                    .collect(),
            )
        }
        2 => {
            let (chunks, tail) = data.as_chunks::<8>();
            if !tail.is_empty() {
                return None;
            }
            Some(
                chunks
                    .iter()
                    .map(|chunk| f64::from_le_bytes(*chunk))
                    .collect(),
            )
        }
        _ => None,
    }
}

fn json_of(value: &rmpv::Value) -> Value {
    match value {
        rmpv::Value::Nil => Value::Null,
        rmpv::Value::Boolean(flag) => json!(flag),
        rmpv::Value::Integer(number) => {
            if let Some(value) = number.as_i64() {
                json!(value)
            } else if let Some(value) = number.as_u64() {
                json!(value)
            } else {
                json!(number.as_f64())
            }
        }
        rmpv::Value::F32(number) => json!(number),
        rmpv::Value::F64(number) => json!(number),
        rmpv::Value::String(text) => json!(text.as_str().unwrap_or("")),
        rmpv::Value::Array(items) => Value::Array(items.iter().map(json_of).collect()),
        rmpv::Value::Map(pairs) => {
            let mut object = Map::new();
            for (key, item) in pairs {
                if let Some(key) = key.as_str() {
                    object.insert(key.to_owned(), json_of(item));
                }
            }
            Value::Object(object)
        }
        rmpv::Value::Binary(bytes) => json!(bytes),
        rmpv::Value::Ext(_, bytes) => json!(bytes),
    }
}

fn rmp_from_json(value: &Value) -> rmpv::Value {
    match value {
        Value::Null => rmpv::Value::Nil,
        Value::Bool(flag) => rmpv::Value::Boolean(*flag),
        Value::Number(number) => {
            if let Some(value) = number.as_i64() {
                rmpv::Value::from(value)
            } else if let Some(value) = number.as_u64() {
                rmpv::Value::from(value)
            } else {
                rmpv::Value::from(number.as_f64().unwrap_or(0.0))
            }
        }
        Value::String(text) => rmpv::Value::from(text.as_str()),
        Value::Array(items) => rmpv::Value::Array(items.iter().map(rmp_from_json).collect()),
        Value::Object(map) => rmpv::Value::Map(
            map.iter()
                .map(|(key, item)| (rmpv::Value::from(key.as_str()), rmp_from_json(item)))
                .collect(),
        ),
    }
}

fn http_json(host: &str, path: &str) -> Option<Value> {
    http_json_file(host, path, "value.json")
}

/// JSON read on a short deadline, for subnet discovery probes.
pub(crate) fn http_json_timed(host: &str, path: &str, timeout: Duration) -> Option<Value> {
    let url = io_url(host, path, "value.json").ok()?;
    let body = http_tuned(&url, "GET", None, timeout, timeout).ok()?;
    serde_json::from_slice(&body).ok()
}

pub(crate) fn http_json_file(host: &str, path: &str, file_name: &str) -> Option<Value> {
    let url = io_url(host, path, file_name).ok()?;
    let body = http(&url, "GET", None).ok()?;
    serde_json::from_slice(&body).ok()
}

fn http_put(host: &str, device_path: &str, body: &[u8]) -> Result<(), String> {
    let url = device_file_url(host, device_path)?;
    let _ = http(&url, "PUT", Some(body))?;
    Ok(())
}

pub(crate) fn http(url: &str, method: &str, body: Option<&[u8]>) -> Result<Vec<u8>, String> {
    http_tuned(
        url,
        method,
        body,
        Duration::from_secs(3),
        Duration::from_secs(30),
    )
}

/// `http` with caller-set connect and read deadlines, for fast probes.
pub(crate) fn http_tuned(
    url: &str,
    method: &str,
    body: Option<&[u8]>,
    connect_timeout: Duration,
    read_timeout: Duration,
) -> Result<Vec<u8>, String> {
    let rest = url.split_once("://").map(|(_, rest)| rest).unwrap_or(url);
    let (authority, path) = rest.split_once('/').unwrap_or((rest, ""));
    let path = format!("/{path}");
    // Try every resolved record like dial does — a dead first address
    // (stale v6, split-horizon DNS) must not fail the whole fetch.
    let mut last_err = format!("invalid host: {authority}");
    let mut stream = None;
    for addr in authority.to_socket_addrs().map_err(|err| err.to_string())? {
        match TcpStream::connect_timeout(&addr, connect_timeout) {
            Ok(tcp) => {
                stream = Some(tcp);
                break;
            }
            Err(err) => last_err = err.to_string(),
        }
    }
    let mut stream = stream.ok_or(last_err)?;
    stream.set_read_timeout(Some(read_timeout)).ok();
    stream.set_write_timeout(Some(read_timeout)).ok();
    let mut head = format!(
        "{method} {path} HTTP/1.1\r\nHost: {authority}\r\nConnection: close\r\nAccept: */*\r\n"
    );
    if let Some(body) = body {
        head.push_str(&format!(
            "Content-Length: {}\r\nContent-Type: application/octet-stream\r\n",
            body.len()
        ));
    }
    head.push_str("\r\n");
    // The deadline bounds the whole exchange — the per-syscall timeout
    // alone lets a peer trickling one byte per interval stay open forever.
    let deadline = Instant::now() + read_timeout * 4;
    for bytes in [head.as_bytes(), body.unwrap_or_default()] {
        let mut rest = bytes;
        while !rest.is_empty() {
            let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
                return Err("the device write deadline elapsed".into());
            };
            stream
                .set_write_timeout(Some(remaining.max(Duration::from_millis(1))))
                .ok();
            match stream.write(rest) {
                Ok(n) => rest = &rest[n..],
                Err(err) => return Err(err.to_string()),
            }
        }
    }
    let mut raw = Vec::new();
    let mut chunk = [0u8; 16 * 1024];
    // (body offset, verified chunked bytes) — see chunked_complete.
    let mut parsed = (0usize, 0usize);
    loop {
        if raw.len() > MAX_BODY + 4096 {
            return Err("the device response is larger than 64 MiB".into());
        }
        if Instant::now() >= deadline {
            return Err("the device response did not complete in time".into());
        }
        // Cap the next read's syscall timeout to the remaining budget —
        // a read issued just before the deadline would otherwise block a
        // full read_timeout past it.
        if let Some(remaining) = deadline.checked_duration_since(Instant::now()) {
            stream
                .set_read_timeout(Some(remaining.max(Duration::from_millis(1))))
                .ok();
        }
        match stream.read(&mut chunk) {
            Ok(0) => break,
            Ok(count) => {
                raw.extend_from_slice(&chunk[..count]);
                // A keep-alive peer may hold the socket open past a complete
                // response — waiting for EOF would read a full timeout and
                // then error on a perfectly good answer.
                if response_complete(&raw, &mut parsed) {
                    break;
                }
            }
            // A stall mid-body is truncation, not EOF — surface it rather than
            // handing a partial response to the parser.
            Err(err) if err.kind() == std::io::ErrorKind::TimedOut => {
                return Err("the device stopped responding mid-response".into());
            }
            Err(err) => return Err(err.to_string()),
        }
    }
    let (status, head_at, body_at) =
        final_response(&raw).ok_or("the device response has no HTTP header")?;
    let header = String::from_utf8_lossy(&raw[head_at..body_at - 4]).to_string();
    if !(200..300).contains(&status) {
        return Err(format!("HTTP {status} for {path}"));
    }
    // 204 (and any 2xx the grammar forbids a body for) is header-complete.
    if status == 204 {
        return Ok(Vec::new());
    }
    let body = &raw[body_at..];
    if transfer_chunked(&header) {
        decode_chunks(body)
    } else if let Some(length) = content_length(&header)? {
        if body.len() < length {
            return Err(format!(
                "the device closed {path} after {} of {length} bytes",
                body.len()
            ));
        }
        Ok(body[..length].to_vec())
    } else {
        Ok(body.to_vec())
    }
}

/// True once the headers and the declared body are all in — Content-Length
/// reached, or a chunked body carrying its empty terminal chunk.
fn response_complete(raw: &[u8], parsed: &mut (usize, usize)) -> bool {
    let Some((status, head_at, body_at)) = final_response(raw) else {
        return false;
    };
    // The parsed cursor is relative to this response's body — a different
    // final response (e.g. after a 1xx preamble) restarts it.
    if parsed.0 != body_at {
        *parsed = (body_at, 0);
    }
    // 1xx/204/304 responses carry no body by definition.
    if matches!(status, 204 | 304) {
        return true;
    }
    let header = String::from_utf8_lossy(&raw[head_at..body_at - 4]);
    let body = &raw[body_at..];
    if transfer_chunked(&header) {
        chunked_complete(body, &mut parsed.1)
    } else if let Ok(Some(length)) = content_length(&header) {
        body.len() >= length
    } else {
        // Close-delimited body — the peer's EOF is the terminator.
        false
    }
}

/// True only when the buffered chunks end exactly on the terminal 0-size
/// chunk — a byte-suffix match can fire early on data ending "0\r\n".
fn chunked_complete(data: &[u8], verified: &mut usize) -> bool {
    // Each verified chunk is self-delimiting — resume at the cursor instead
    // of re-walking the whole body after every socket read (O(body²) CPU
    // inside the read deadline otherwise).
    let mut data = &data[*verified..];
    loop {
        let Some(end) = data.windows(2).position(|window| window == b"\r\n") else {
            return false;
        };
        let line = std::str::from_utf8(&data[..end]).unwrap_or("");
        let Ok(size) = usize::from_str_radix(line.split(';').next().unwrap_or("").trim(), 16)
        else {
            return false;
        };
        if size == 0 {
            // Complete only when the trailer section ends — an immediate
            // blank line, or trailers closed by one. Bare "0\r\n" is a
            // truncated response, not a finished one.
            let rest = &data[end + 2..];
            return rest.starts_with(b"\r\n")
                || rest.windows(4).any(|window| window == b"\r\n\r\n");
        }
        let Some(next) = end
            .checked_add(2)
            .and_then(|at| at.checked_add(size))
            .and_then(|at| at.checked_add(2))
        else {
            return false;
        };
        if data.len() < next {
            return false;
        }
        *verified += next;
        data = &data[next..];
    }
}

/// The declared Content-Length, or None when absent. A malformed or
/// conflicting field is an error — silently dropping it would reframe the
/// response as close-delimited or truncated.
fn content_length(header: &str) -> Result<Option<usize>, String> {
    let mut found = None;
    for line in header.lines() {
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        if name.trim().eq_ignore_ascii_case("content-length") {
            let length: usize = value
                .trim()
                .parse()
                .map_err(|_| "malformed Content-Length".to_owned())?;
            if let Some(prev) = found.replace(length) {
                if prev != length {
                    return Err("conflicting Content-Length fields".into());
                }
            }
        }
    }
    Ok(found)
}

/// True only when a Transfer-Encoding field's final coding is chunked —
/// a raw substring match also fires on X-Transfer-Encoding and misses
/// "gzip, chunked".
fn transfer_chunked(header: &str) -> bool {
    header.lines().any(|line| {
        line.split_once(':').is_some_and(|(name, value)| {
            name.trim().eq_ignore_ascii_case("transfer-encoding")
                && value
                    .split(',')
                    .next_back()
                    .is_some_and(|token| token.trim().eq_ignore_ascii_case("chunked"))
        })
    })
}

/// The final response's status and body offset — interim 1xx blocks each
/// end at their own blank line and carry no body.
fn final_response(raw: &[u8]) -> Option<(u16, usize, usize)> {
    let mut at = 0;
    loop {
        let split = raw.get(at..)?.windows(4).position(|w| w == b"\r\n\r\n")? + at;
        let status: u16 = String::from_utf8_lossy(&raw[at..split])
            .split_whitespace()
            .nth(1)
            .and_then(|code| code.parse().ok())
            .unwrap_or(0);
        if (100..200).contains(&status) {
            at = split + 4;
            continue;
        }
        return Some((status, at, split + 4));
    }
}

fn decode_chunks(mut data: &[u8]) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    loop {
        let end = data
            .windows(2)
            .position(|window| window == b"\r\n")
            .ok_or("truncated chunk")?;
        let line = std::str::from_utf8(&data[..end]).map_err(|_| "truncated chunk".to_string())?;
        let size = usize::from_str_radix(line.split(';').next().unwrap_or("0").trim(), 16)
            .map_err(|err| err.to_string())?;
        data = data.get(end + 2..).unwrap_or(&[]);
        if size == 0 {
            // The trailer section must end — bare EOF after "0\r\n" is
            // truncation, and junk bytes are not a body.
            if !(data.starts_with(b"\r\n") || data.windows(4).any(|window| window == b"\r\n\r\n")) {
                return Err("truncated chunk".into());
            }
            break;
        }
        // size is attacker-controlled hex — the +2 can overflow usize.
        let Some(need) = size.checked_add(2) else {
            return Err("truncated chunk".into());
        };
        if data.len() < need {
            return Err("truncated chunk".into());
        }
        if &data[size..need] != b"\r\n" {
            return Err("malformed chunk".into());
        }
        out.extend_from_slice(&data[..size]);
        data = &data[size + 2..];
        if out.len() > MAX_BODY {
            return Err("the device response is larger than 64 MiB".into());
        }
    }
    Ok(out)
}

fn package_zip(
    dir: &Path,
    zip_path: &Path,
    session_id: &str,
    keep: &mut dyn FnMut(u64, u64) -> bool,
) -> Result<(), String> {
    // The id comes from the device's file_header_meta — a hostile or broken
    // table must not write `../` members into the archive.
    let session_id = if crate::discover::valid_session_id(session_id) {
        session_id
    } else {
        "session"
    };
    if let Some(parent) = zip_path.parent() {
        std::fs::create_dir_all(parent).map_err(|err| err.to_string())?;
    }
    // Write beside the destination and rename at the end — a mid-write
    // failure must not truncate a previous good download at zip_path.
    // Unique per call — two overlapping downloads to the same dest must not
    // interleave bytes in one staging file (same reasoning as the staging
    // dir uniquifier in `download_with`).
    static NEXT_PART: AtomicU64 = AtomicU64::new(0);
    let part = zip_path.with_extension(format!(
        "part-{}-{}",
        std::process::id(),
        NEXT_PART.fetch_add(1, Ordering::Relaxed)
    ));
    let result = (|| {
        let file = std::fs::File::create(&part).map_err(|err| err.to_string())?;
        let mut zip = zip::ZipWriter::new(file);
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated);
        let mut files = Vec::new();
        collect_files(dir, 0, &mut files)?;
        files.sort();
        for path in files {
            let rel = path.strip_prefix(dir).map_err(|err| err.to_string())?;
            let name = format!("{session_id}/{}", rel.to_string_lossy().replace('\\', "/"));
            zip.start_file(name, options)
                .map_err(|err| err.to_string())?;
            let bytes = std::fs::read(&path).map_err(|err| err.to_string())?;
            std::io::Write::write_all(&mut zip, &bytes).map_err(|err| err.to_string())?;
        }
        zip.finish().map_err(|err| err.to_string())?;
        // A cancel landing inside the write must not publish the archive a
        // cancelled task already reported it skipped.
        if !keep(0, 0) {
            return Err("cancelled".into());
        }
        Ok::<(), String>(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&part);
    }
    result?;
    // A dest held open by another process fails the rename on Windows — the
    // finished part file must not be left behind on that path too.
    if let Err(err) = std::fs::rename(&part, zip_path) {
        let _ = std::fs::remove_file(&part);
        return Err(err.to_string());
    }
    Ok(())
}

fn collect_files(dir: &Path, depth: usize, out: &mut Vec<PathBuf>) -> Result<(), String> {
    if depth > MAX_ZIP_WALK_DEPTH {
        // is_dir() follows symlinks — a captured tree could carry a loop.
        return Err(format!(
            "session tree is deeper than {} levels under {}",
            MAX_ZIP_WALK_DEPTH,
            dir.display()
        ));
    }
    for item in std::fs::read_dir(dir).map_err(|err| err.to_string())? {
        let path = item.map_err(|err| err.to_string())?.path();
        // is_dir() follows symlinks — skip them entirely so a link inside
        // the captured tree can't pull outside files into the zip.
        if path.is_symlink() {
            continue;
        }
        if path.is_dir() {
            collect_files(&path, depth + 1, out)?;
        } else if path.is_file() {
            out.push(path);
        }
    }
    Ok(())
}

fn lock() -> std::sync::MutexGuard<'static, Option<Live>> {
    LIVE.lock().unwrap_or_else(|poison| poison.into_inner())
}

fn live() -> bool {
    lock().is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    // The busy flags are process statics — tests that acquire them (directly
    // or via upload()) must not overlap.
    static FLAG_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn http_get_reads_a_content_length_body() {
        use std::net::TcpListener;
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        thread::spawn(move || {
            let (mut sock, _) = listener.accept().unwrap();
            let mut buf = [0u8; 512];
            let _ = sock.read(&mut buf);
            let _ = sock.write_all(
                b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\nConnection: close\r\n\r\nhello",
            );
        });
        let body = http(
            &format!("http://127.0.0.1:{port}/io/admin/version/value.json"),
            "GET",
            None,
        )
        .unwrap();
        assert_eq!(body, b"hello");
    }

    #[test]
    fn an_elapsed_deadline_stops_the_read_without_a_syscall_wait() {
        use std::net::TcpListener;
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        // The peer sends nothing — the socket read itself would block.
        let _server = listener.accept().unwrap();
        let mut stream = DeadlineStream::new(client, Instant::now() - Duration::from_millis(1));
        let mut buf = [0u8; 8];
        let err = stream.read(&mut buf).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::TimedOut);
    }

    #[test]
    fn a_pair_update_keeps_the_latest_scalar() {
        let value = rmpv::Value::Array(vec![rmpv::Value::from(42), rmpv::Value::from(1.0)]);
        assert_eq!(json_of(&unwrap_update(&value)), json!(42));
    }

    #[test]
    fn the_catalog_remembers_a_host_without_connecting() {
        let dir = std::env::temp_dir().join(format!("scan-kit-runner-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let db = dir.join("scan-kit.sqlite");
        with_store(&db, |conn| store::set_runner_host(conn, "192.168.100.184")).unwrap();
        let listed = catalog(&db);
        assert_eq!(listed["host"], "192.168.100.184");
        assert_eq!(listed["connected"], false);
        assert!(listed["view"]["coach"]
            .as_str()
            .unwrap()
            .contains("Connect"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_session_dir_is_zipped_under_the_session_id() {
        let dir = std::env::temp_dir().join(format!("scan-kit-spots-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let run = dir.join("layer-0").join("run-0");
        std::fs::create_dir_all(&run).unwrap();
        std::fs::write(
            run.join("IX256_1_spot_data.csv"),
            "spot_no,layer_id,total_dose(nC)\n1,0,3.5\n",
        )
        .unwrap();
        std::fs::write(dir.join("session_info.json"), "{}\n").unwrap();
        let zip_path =
            std::env::temp_dir().join(format!("scan-kit-spots-{}.zip", std::process::id()));
        package_zip(&dir, &zip_path, "sess", &mut |_, _| true).unwrap();
        let file = std::fs::File::open(&zip_path).unwrap();
        let mut archive = zip::ZipArchive::new(file).unwrap();
        assert!(archive
            .by_name("sess/layer-0/run-0/IX256_1_spot_data.csv")
            .is_ok());
        assert!(archive.by_name("sess/session_info.json").is_ok());
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_file(&zip_path);
    }

    #[test]
    fn codecs_round_trip_scalars_blobs_and_chunked_bodies() {
        assert_eq!(display_io(&json!("RCI")), "RCI");
        assert_eq!(display_io(&json!(3)), "3");
        assert_eq!(display_io(&Value::Null), "");
        assert!(display_io(&json!({"a": 1})).contains('a'));
        let body = connected_body("192.168.1.2", &json!(7), &Value::Null, "/sessions", "42");
        assert_eq!(body["host"], "192.168.1.2");
        assert_eq!(body["version"], "7");
        assert_eq!(body["device_type"], "");
        assert!(matches!(config_body(), rmpv::Value::Map(_)));

        let packed = rmp_from_json(&json!({
            "flag": true,
            "n": -2,
            "f": 1.5,
            "s": "spot",
            "a": [null],
            "o": {"k": 1}
        }));
        let restored = json_of(&packed);
        assert_eq!(restored["flag"], true);
        assert_eq!(restored["n"], -2);
        assert_eq!(restored["s"], "spot");
        assert_eq!(json_of(&rmpv::Value::F32(1.25)), json!(1.25));
        let wide = json_of(&rmpv::Value::from(u64::MAX));
        assert!(wide.as_u64().is_some() || wide.as_f64().is_some());
        assert_eq!(json_of(&rmpv::Value::Binary(vec![9, 8])), json!([9, 8]));
        assert_eq!(json_of(&rmpv::Value::Ext(-1, vec![4])), json!([4]));
        assert_eq!(message_event(&rmpv::Value::Nil), "");
        assert!(message_data(&rmpv::Value::Nil).is_none());
        assert!(map_get(&rmpv::Value::Map(vec![]), "event").is_none());

        let f32_blob = 1.5f32.to_le_bytes().to_vec();
        let f64_blob = 2.5f64.to_le_bytes().to_vec();
        let times = 0u64.to_le_bytes().to_vec();
        let history = rmpv::Value::Map(vec![
            (rmpv::Value::from("$t"), rmpv::Value::from(1i64)),
            (
                rmpv::Value::from("$d"),
                rmpv::Value::Binary(f32_blob.clone()),
            ),
            (rmpv::Value::from("$ts"), rmpv::Value::Binary(times.clone())),
        ]);
        assert_eq!(json_of(&unwrap_update(&history)), json!(1.5));
        let short = rmpv::Value::Map(vec![
            (rmpv::Value::from("$t"), rmpv::Value::from(2i64)),
            (
                rmpv::Value::from("$d"),
                rmpv::Value::Binary(f64_blob.clone()),
            ),
            (rmpv::Value::from("$ts"), rmpv::Value::Binary(vec![0])),
        ]);
        assert!(history_last(&short).is_none());
        assert!(history_last(&rmpv::Value::Nil).is_none());
        let decoded = decode_blobs(&rmpv::Value::Map(vec![
            (rmpv::Value::from("$t"), rmpv::Value::from(2i64)),
            (rmpv::Value::from("$d"), rmpv::Value::Binary(f64_blob)),
            (rmpv::Value::from("nested"), rmpv::Value::from("x")),
        ]));
        assert!(matches!(decoded, rmpv::Value::Array(_)));
        let nested = decode_blobs(&rmpv::Value::Array(vec![rmpv::Value::from(3)]));
        assert!(matches!(nested, rmpv::Value::Array(_)));
        let walked = decode_blobs(&rmpv::Value::Map(vec![(
            rmpv::Value::from("child"),
            rmpv::Value::from(1),
        )]));
        assert!(matches!(walked, rmpv::Value::Map(_)));
        assert!(unpack_blob(None, &[]).is_none());
        assert!(unpack_blob(Some(9), &[0, 0, 0, 0]).is_none());
        assert_eq!(
            unwrap_update(&rmpv::Value::Array(vec![])),
            rmpv::Value::Array(vec![])
        );
        let inner = rmpv::Value::Array(vec![
            rmpv::Value::Array(vec![rmpv::Value::from(1)]),
            rmpv::Value::Array(vec![rmpv::Value::from(4)]),
        ]);
        assert_eq!(json_of(&unwrap_update(&inner)), json!(4));
        assert_eq!(
            json_of(&unwrap_update(&rmpv::Value::from("plain"))),
            json!("plain")
        );
        // A $ts that cannot hold even one timestamp makes the blob invalid —
        // the update must be dropped, not unwrapped.
        let bad_ts = rmpv::Value::Map(vec![
            (rmpv::Value::from("$t"), rmpv::Value::from(1i64)),
            (
                rmpv::Value::from("$d"),
                rmpv::Value::Binary(f32_blob.clone()),
            ),
            (rmpv::Value::from("$ts"), rmpv::Value::Binary(vec![0])),
        ]);
        assert_eq!(unwrap_update(&bad_ts), rmpv::Value::Nil);

        assert_eq!(
            content_length("Host: x\r\nContent-Length: 4\r\n"),
            Ok(Some(4))
        );
        assert_eq!(content_length("Host: x"), Ok(None));
        // Malformed and conflicting fields are errors, not a silent reframe.
        assert!(content_length("Host: x\r\nContent-Length: junk\r\n").is_err());
        assert!(content_length("Host: x\r\nContent-Length: 4\r\nContent-Length: 9\r\n").is_err());
        assert_eq!(decode_chunks(b"5\r\nhello\r\n0\r\n\r\n").unwrap(), b"hello");
        assert!(decode_chunks(b"zz").is_err());
        // Bare "0\r\n" and junk after it are truncation, not a body — a
        // trailer section must close with its own blank line.
        assert!(decode_chunks(b"1\r\nx\r\n0\r\n").is_err());
        assert!(decode_chunks(b"1\r\nx\r\n0\r\nX").is_err());
        assert_eq!(
            decode_chunks(b"1\r\nx\r\n0\r\nDigest: a\r\n\r\n").unwrap(),
            b"x"
        );
        // Field-name matching: X-Transfer-Encoding is not TE, and the
        // coding list checks its final token.
        assert!(!transfer_chunked("X-Transfer-Encoding: chunked"));
        assert!(transfer_chunked("Transfer-Encoding:  chunked"));
        assert!(transfer_chunked("transfer-encoding: gzip, chunked"));
        assert!(!transfer_chunked("Transfer-Encoding: chunked, gzip"));
        // An interim 1xx block is not the final response.
        let raw = b"HTTP/1.1 103 Early Hints\r\nLink: x\r\n\r\nHTTP/1.1 200 OK\r\nContent-Length: 1\r\n\r\nx";
        let (status, _, body_at) = final_response(raw).unwrap();
        assert_eq!(status, 200);
        assert_eq!(&raw[body_at..], b"x");
        assert!(decode_chunks(b"5\r\nhi").is_err());
        assert!(decode_chunks(b"5\r\nhello\r\n\xff\r\n").is_err());
        let port = serve_http(|request| {
            if request.starts_with("PUT") {
                return http_response(204, "NO", b"");
            }
            if request.to_ascii_lowercase().contains("chunked-path") {
                return b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n5\r\nhello\r\n0\r\n\r\n".to_vec();
            }
            if request.contains("missing-header") {
                return b"not-http".to_vec();
            }
            if request.contains("/io/") {
                return http_response(200, "OK", b"{\"n\":1}");
            }
            if request.contains("fail") {
                return http_response(500, "NO", b"");
            }
            if request.contains("short-body") {
                return b"HTTP/1.1 200 OK\r\nContent-Length: 4\r\nConnection: close\r\n\r\nxy"
                    .to_vec();
            }
            http_response(404, "NO", b"")
        });
        let host = format!("127.0.0.1:{port}");
        let value = http_json(&host, "admin/version").unwrap();
        assert_eq!(value["n"], 1);
        let chunked = http(&format!("http://{host}/chunked-path"), "GET", None).unwrap();
        assert_eq!(chunked, b"hello");
        assert!(http(&format!("http://{host}/fail"), "GET", None).is_err());
        assert!(http(&format!("http://{host}/missing-header"), "GET", None).is_err());
        let short = http(&format!("http://{host}/short-body"), "GET", None).unwrap_err();
        assert!(short.contains("2 of 4"));
        assert!(http("http://not a host/x", "GET", None).is_err());
        let refused = open_socket("127.0.0.1:1").unwrap_err();
        assert!(!refused.is_empty());
    }

    #[test]
    fn a_local_mpack_socket_delivers_one_update() {
        use std::net::TcpListener;
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let mut socket = accept_mpack(stream);
            for _ in 0..8 {
                let Ok(Message::Binary(_)) = socket.read() else {
                    break;
                };
                let reply = rmpv::Value::Map(vec![
                    (rmpv::Value::from("event"), rmpv::Value::from("update")),
                    (
                        rmpv::Value::from("data"),
                        rmpv::Value::Map(vec![
                            (rmpv::Value::from(1), rmpv::Value::from(0)),
                            (
                                rmpv::Value::from("/status/value"),
                                rmpv::Value::Array(vec![
                                    rmpv::Value::from("ready"),
                                    rmpv::Value::from(1),
                                ]),
                            ),
                        ]),
                    ),
                ]);
                let mut bytes = Vec::new();
                rmpv::encode::write_value(&mut bytes, &reply).unwrap();
                if socket.send(Message::Binary(bytes.into())).is_err() {
                    break;
                }
            }
        });
        let mut socket = dial(&format!("127.0.0.1:{port}")).unwrap();
        send(&mut socket, "config", Some(config_body())).unwrap();
        set_field(&mut socket, "status", &json!(true)).unwrap();
        let value = read_one(&mut socket, "status", Duration::from_secs(2)).unwrap();
        assert_eq!(value, json!("ready"));
        let _ = recv(&mut socket, Duration::from_millis(50));
    }

    fn accept_mpack(mut stream: TcpStream) -> WebSocket<DeadlineStream> {
        let mut buf = Vec::new();
        let mut tmp = [0u8; 1024];
        loop {
            let count = stream.read(&mut tmp).unwrap();
            if count == 0 {
                break;
            }
            buf.extend_from_slice(&tmp[..count]);
            if buf.windows(4).any(|window| window == b"\r\n\r\n") {
                break;
            }
        }
        let header = String::from_utf8_lossy(&buf);
        let key = header
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("sec-websocket-key")
                    .then(|| value.trim())
            })
            .expect("websocket key");
        let accept = tungstenite::handshake::derive_accept_key(key.as_bytes());
        let response = format!(
            "HTTP/1.1 101 Switching Protocols\r\n\
             Upgrade: websocket\r\n\
             Connection: Upgrade\r\n\
             Sec-WebSocket-Accept: {accept}\r\n\
             Sec-WebSocket-Protocol: mpack.v2\r\n\
             \r\n"
        );
        stream.write_all(response.as_bytes()).unwrap();
        let stream = DeadlineStream::new(stream, Instant::now() + Duration::from_secs(10));
        WebSocket::from_raw_socket(stream, tungstenite::protocol::Role::Server, None)
    }

    fn http_response(status: u16, text: &str, body: &[u8]) -> Vec<u8> {
        let mut bytes = format!(
            "HTTP/1.1 {status} {text}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        )
        .into_bytes();
        bytes.extend_from_slice(body);
        bytes
    }

    fn serve_http(handler: impl Fn(&str) -> Vec<u8> + Send + 'static) -> u16 {
        use std::net::TcpListener;
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        thread::spawn(move || {
            for _ in 0..80 {
                let Ok((mut sock, _)) = listener.accept() else {
                    break;
                };
                let mut buf = [0u8; 4096];
                let count = sock.read(&mut buf).unwrap_or(0);
                let request = String::from_utf8_lossy(&buf[..count]).to_string();
                let _ = sock.write_all(&handler(&request));
            }
        });
        port
    }

    #[test]
    fn a_blob_array_update_keeps_every_element() {
        let blob: Vec<u8> = [1.0f32, 2.0, 3.0]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let update = rmpv::Value::Map(vec![
            (rmpv::Value::from("$t"), rmpv::Value::from(1i64)),
            (rmpv::Value::from("$d"), rmpv::Value::Binary(blob)),
        ]);
        let rmpv::Value::Array(items) = unwrap_update(&update) else {
            panic!("a blob decodes to the whole array");
        };
        assert_eq!(items.len(), 3);
    }

    #[test]
    fn a_malformed_chunk_terminator_errors_instead_of_desyncing() {
        // Two non-CRLF bytes after the payload must not be skipped over.
        assert!(decode_chunks(b"3\r\nabcXY0\r\n\r\n").is_err());
        assert_eq!(decode_chunks(b"3\r\nabc\r\n0\r\n\r\n").unwrap(), b"abc");
        // usize::MAX chunk size — the +2 length check must not wrap or panic.
        assert!(decode_chunks(b"ffffffffffffffff\r\n").is_err());
    }

    #[test]
    fn a_blob_with_a_partial_tail_decodes_to_nothing() {
        // 10 bytes is not a whole number of f32s or f64s.
        assert_eq!(unpack_blob(Some(1), &[0u8; 10]), None);
        assert_eq!(unpack_blob(Some(2), &[0u8; 10]), None);
        assert_eq!(unpack_blob(Some(1), &[0, 0, 128, 63]), Some(vec![1.0]));
    }

    #[test]
    fn busy_flags_reject_a_second_connect_or_upload() {
        // Two concurrent uploads would PUT to the same device path and both
        // report success for different files — the flags must single-flight.
        let _guard = FLAG_LOCK.lock().unwrap();
        let upload = acquire(&UPLOAD_BUSY, "an upload is already in progress").unwrap();
        assert!(acquire(&UPLOAD_BUSY, "an upload is already in progress").is_err());
        drop(upload);
        let connect = acquire(&CONNECT_BUSY, "a connect is already in progress").unwrap();
        assert!(acquire(&CONNECT_BUSY, "a connect is already in progress").is_err());
        drop(connect);
    }

    #[test]
    fn a_disconnect_marks_a_connect_in_flight() {
        // A disconnect during the open phase must land — the generation the
        // connect snapshotted at entry can no longer match at install time.
        let before = DISCONNECTS.load(Ordering::Acquire);
        disconnect();
        assert!(DISCONNECTS.load(Ordering::Acquire) > before);
    }
    #[test]
    fn a_response_is_only_complete_with_its_body_or_terminal_chunk() {
        let header = |head: &str, body: &[u8]| {
            let mut raw = head.replace("\n", "\r\n").into_bytes();
            raw.extend_from_slice(body);
            raw
        };
        // Content-Length: full body buffered.
        let full = header("HTTP/1.1 200 OK\nContent-Length: 3\n\n", b"abc");
        assert!(response_complete(&full, &mut (0, 0)));
        assert!(!response_complete(&full[..full.len() - 1], &mut (0, 0)));
        // Chunked: only the terminal zero-length chunk closes the stream.
        let open = header(
            "HTTP/1.1 200 OK\nTransfer-Encoding: chunked\n\n",
            b"3\r\nabc\r\n",
        );
        assert!(!response_complete(&open, &mut (0, 0)));
        let closed = header(
            "HTTP/1.1 200 OK\nTransfer-Encoding: chunked\n\n",
            b"3\r\nabc\r\n0\r\n\r\n",
        );
        assert!(response_complete(&closed, &mut (0, 0)));
        // A chunk whose data ends "0\r\n" produces the same byte suffix
        // as the terminal chunk — the framing scan must not fire early.
        let trap = header(
            "HTTP/1.1 200 OK\nTransfer-Encoding: chunked\n\n",
            b"4\r\na0\r\n\r\n",
        );
        assert!(!response_complete(&trap, &mut (0, 0)));
        // Close-delimited bodies need EOF, so a bare header is incomplete —
        // but a no-body status is complete at the header boundary itself.
        let bare = header("HTTP/1.1 200 OK\n\n", b"");
        assert!(!response_complete(&bare, &mut (0, 0)));
        let no_body = header("HTTP/1.1 204 No Content\n\n", b"");
        assert!(response_complete(&no_body, &mut (0, 0)));
    }

    #[test]
    fn an_oversized_upload_is_rejected_before_any_io() {
        let _guard = FLAG_LOCK.lock().unwrap();
        let dir = std::env::temp_dir().join(format!("sk_r21_up_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let csv = dir.join("big.csv");
        // Sparse file: the length is set without writing the bytes.
        std::fs::File::create(&csv)
            .unwrap()
            .set_len(MAX_BODY as u64 + 1)
            .unwrap();
        let db = dir.join("store.db");
        let err = upload(&db, &csv).unwrap_err();
        assert!(err.contains("too large"), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_chunked_parse_resumes_at_the_verified_cursor() {
        // Each socket read used to re-walk the whole body — verified chunks
        // are skipped on the next pass.
        let mut verified = 0usize;
        assert!(!chunked_complete(b"5\r\nhello\r\n", &mut verified));
        assert_eq!(verified, 10);
        let body = b"5\r\nhello\r\n0\r\n\r\n";
        assert!(chunked_complete(body, &mut verified));
    }
}
