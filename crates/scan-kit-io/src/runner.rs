//! Plan Runner session: one mpack WebSocket plus HTTP file transfer.
//!
//! ponytail: the HTTP client understands Content-Length and chunked bodies, and
//! stops after 64 MiB. A device that streams a larger session file needs a
//! ranged or streaming reader.

use std::collections::{BTreeMap, HashMap};
use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant};

use scan_kit_core::{
    control_button, device_file_url, field_subscribe_key, io_bool, io_number, io_url,
    normalize_host, parse_host, resolve_session_download, runner_view, DEFAULT_CONTROL_POINTS,
    DEFAULT_SESSION_ROOT, LOAD_STATE_SUCCESS, POINTS_LOAD_STATE, POINTS_UPLOAD,
    POINTS_UPLOAD_TARGET, POINTS_VALID, SESSION_DIRECTORY, STATUS_PATHS,
};
use serde_json::{json, Map, Value};
use tungstenite::client::{client, IntoClientRequest};
use tungstenite::http::HeaderValue;
use tungstenite::{Message, WebSocket};

use crate::store::{self, with_store};

const MAX_BODY: usize = 64 * 1024 * 1024;
const MAX_LAYER: usize = 128;
const MAX_RUN: usize = 4;

struct Live {
    host: String,
    socket: WebSocket<TcpStream>,
    status: Map<String, Value>,
    version: Value,
    device_type: Value,
    session_directory: String,
}

static LIVE: Mutex<Option<Live>> = Mutex::new(None);

pub fn catalog(db: &Path) -> Value {
    let (host, file_dir) = with_store(db, |conn| store::runner_settings(conn))
        .ok()
        .unwrap_or((None, None));
    let connected = live();
    let mut body = json!({
        "host": host,
        "file_dir": file_dir,
        "dest": scan_kit_core::default_session_zip_path("", file_dir.as_deref().unwrap_or("")),
        "connected": connected,
        "view": runner_view(&json!({}), false, false),
    });
    if connected {
        if let Ok(view) = read_status(false, "") {
            body["view"] = view;
            body["connected"] = json!(true);
        }
    }
    body
}

pub fn connect(db: &Path, host: &str) -> Result<Value, String> {
    let host = parse_host(host)?;
    disconnect();
    let mut socket = open_socket(&host)?;
    send(&mut socket, "config", Some(config_body()))?;
    let status = read_paths(&mut socket, STATUS_PATHS, Duration::from_secs(8))?;
    let version = http_json(&host, "admin/version").unwrap_or(Value::Null);
    let device_type = http_json(&host, "admin/device_type").unwrap_or(Value::Null);
    let session_directory = http_json(&host, SESSION_DIRECTORY)
        .and_then(|value| value.as_str().map(str::to_owned))
        .unwrap_or_else(|| DEFAULT_SESSION_ROOT.into());
    {
        let mut guard = lock();
        *guard = Some(Live {
            host: host.clone(),
            socket,
            status,
            version: version.clone(),
            device_type: device_type.clone(),
            session_directory: session_directory.clone(),
        });
    }
    let _ = with_store(db, |conn| store::set_runner_host(conn, &host));
    Ok(connected_body(
        &host,
        &version,
        &device_type,
        &session_directory,
    ))
}

pub fn disconnect() {
    if let Some(mut live) = lock().take() {
        let _ = live.socket.close(None);
    }
}

pub fn read_status(has_plan: bool, dest: &str) -> Result<Value, String> {
    let mut guard = lock();
    let live = guard.as_mut().ok_or("not connected")?;
    refresh(live, Duration::from_millis(500))?;
    Ok(view_of(live, has_plan, dest))
}

pub fn upload(db: &Path, csv_path: &Path) -> Result<Value, String> {
    if !csv_path.is_file() {
        return Err(format!("CSV not found: {}", csv_path.display()));
    }
    let bytes = std::fs::read(csv_path).map_err(|err| err.to_string())?;
    let target = {
        let mut guard = lock();
        let live = guard.as_mut().ok_or("not connected")?;
        set_field(
            &mut live.socket,
            POINTS_UPLOAD_TARGET,
            &json!(DEFAULT_CONTROL_POINTS),
        )?;
        thread::sleep(Duration::from_millis(200));
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
        http_put(&live.host, &target, &bytes)?;
        set_field(&mut live.socket, POINTS_UPLOAD, &json!(true))?;
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            let state = read_one(&mut live.socket, POINTS_LOAD_STATE, Duration::from_secs(2))?;
            let code = io_number(&state).map(|value| value as i64);
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
        let valid = read_one(&mut live.socket, POINTS_VALID, Duration::from_secs(8))?;
        subscribe_status(&mut live.socket)?;
        if !io_bool(&valid) {
            return Err("points_valid is false after upload".into());
        }
        live.status.insert(POINTS_VALID.to_owned(), valid);
        target
    };
    if let Some(parent) = csv_path.parent() {
        let _ = with_store(db, |conn| {
            store::set_runner_file_dir(conn, &parent.display().to_string())
        });
    }
    let mut body = read_status(true, "")?;
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
    let (host, root) = {
        let guard = lock();
        let live = guard.as_ref().ok_or("not connected")?;
        (live.host.clone(), live.session_directory.clone())
    };
    let (remote, zip_path) =
        resolve_session_download(dest, &root).ok_or("Choose a local .zip path for the session.")?;
    let staging = PathBuf::from(&zip_path)
        .parent()
        .unwrap_or(Path::new("."))
        .join(format!(
            ".{}_staging",
            Path::new(&remote)
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("session")
        ));
    let session_name = Path::new(&remote)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("session");
    let session_dir = staging.join(session_name);
    let _ = std::fs::remove_dir_all(&staging);
    std::fs::create_dir_all(&session_dir).map_err(|err| err.to_string())?;
    let result = (|| {
        let count = download_files(&host, &remote, &session_dir, keep)?;
        if !keep(count as u64, count as u64) {
            return Err("cancelled".into());
        }
        if count == 0 {
            return Err(format!("no session files found under {remote}"));
        }
        ensure_spot_data(&session_dir)?;
        ensure_session_info(&session_dir, session_name)?;
        package_zip(&session_dir, Path::new(&zip_path), session_name)?;
        Ok(zip_path.clone())
    })();
    let _ = std::fs::remove_dir_all(&staging);
    let zip_path = result?;
    if let Some(parent) = Path::new(&zip_path).parent() {
        let _ = with_store(db, |conn| {
            store::set_runner_file_dir(conn, &parent.display().to_string())
        });
    }
    Ok(json!({
        "path": zip_path,
        "message": format!("Session saved to {zip_path}"),
    }))
}

fn connected_body(
    host: &str,
    version: &Value,
    device_type: &Value,
    session_directory: &str,
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
    view
}

fn view_of(live: &Live, has_plan: bool, dest: &str) -> Value {
    let mut view = runner_view(&Value::Object(live.status.clone()), true, has_plan);
    view["host"] = json!(live.host);
    view["version"] = json!(display_io(&live.version));
    view["device_type"] = json!(display_io(&live.device_type));
    view["session_directory"] = json!(live.session_directory);
    view["hint"] = json!(scan_kit_core::session_download_hint(
        dest,
        &live.session_directory
    ));
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
    socket: &mut WebSocket<TcpStream>,
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
    socket: &mut WebSocket<TcpStream>,
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

fn subscribe_status(socket: &mut WebSocket<TcpStream>) -> Result<(), String> {
    subscribe(socket, STATUS_PATHS)
}

fn subscribe(socket: &mut WebSocket<TcpStream>, paths: &[&str]) -> Result<(), String> {
    let mut pairs = Vec::new();
    for path in paths {
        pairs.push((
            rmpv::Value::from(field_subscribe_key(path)),
            rmpv::Value::Boolean(false),
        ));
    }
    send(socket, "subscribe", Some(rmpv::Value::Map(pairs)))
}

fn set_field(socket: &mut WebSocket<TcpStream>, path: &str, value: &Value) -> Result<(), String> {
    let payload = rmpv::Value::Map(vec![(
        rmpv::Value::from(field_subscribe_key(path)),
        rmp_from_json(value),
    )]);
    send(socket, "set", Some(payload))
}

fn poll_keys(
    socket: &mut WebSocket<TcpStream>,
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

fn open_socket(host: &str) -> Result<WebSocket<TcpStream>, String> {
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

fn dial(endpoint: &str) -> Result<WebSocket<TcpStream>, String> {
    let addr = endpoint
        .to_socket_addrs()
        .map_err(|err| err.to_string())?
        .next()
        .ok_or_else(|| format!("invalid host: {endpoint}"))?;
    let stream = TcpStream::connect_timeout(&addr, Duration::from_secs(10))
        .map_err(|err| err.to_string())?;
    stream.set_nodelay(true).ok();
    let _ = stream.set_write_timeout(Some(Duration::from_secs(10)));
    let mut request = format!("ws://{endpoint}")
        .into_client_request()
        .map_err(|err| err.to_string())?;
    request.headers_mut().insert(
        "Sec-WebSocket-Protocol",
        HeaderValue::from_static("mpack.v2, mpack.v1, mpack"),
    );
    let (socket, response) = client(request, stream).map_err(|err| err.to_string())?;
    let protocol = response
        .headers()
        .get("sec-websocket-protocol")
        .and_then(|value| value.to_str().ok())
        .unwrap_or("");
    if !protocol.contains("mpack") {
        return Err(format!(
            "device did not accept mpack subprotocol (got '{protocol}')"
        ));
    }
    Ok(socket)
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
    socket: &mut WebSocket<TcpStream>,
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
    socket
        .send(Message::Binary(bytes.into()))
        .map_err(|err| err.to_string())
}

fn recv(
    socket: &mut WebSocket<TcpStream>,
    timeout: Duration,
) -> Result<Option<rmpv::Value>, String> {
    let stream = socket.get_mut();
    stream
        .set_read_timeout(Some(timeout.max(Duration::from_millis(20))))
        .ok();
    match socket.read() {
        Ok(Message::Binary(bytes)) => {
            let mut cursor = std::io::Cursor::new(bytes.as_ref());
            rmpv::decode::read_value(&mut cursor)
                .map(Some)
                .map_err(|err| err.to_string())
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
    if times.len() < values.len() * 8 || values.is_empty() {
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
    match dtype? {
        1 => Some(
            data.as_chunks::<4>()
                .0
                .iter()
                .map(|chunk| f64::from(f32::from_le_bytes(*chunk)))
                .collect(),
        ),
        2 => Some(
            data.as_chunks::<8>()
                .0
                .iter()
                .map(|chunk| f64::from_le_bytes(*chunk))
                .collect(),
        ),
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
    let url = io_url(host, path, "value.json").ok()?;
    let body = http(&url, "GET", None).ok()?;
    serde_json::from_slice(&body).ok()
}

fn http_put(host: &str, device_path: &str, body: &[u8]) -> Result<(), String> {
    let url = device_file_url(host, device_path)?;
    let _ = http(&url, "PUT", Some(body))?;
    Ok(())
}

fn http(url: &str, method: &str, body: Option<&[u8]>) -> Result<Vec<u8>, String> {
    let rest = url.split_once("://").map(|(_, rest)| rest).unwrap_or(url);
    let (authority, path) = rest.split_once('/').unwrap_or((rest, ""));
    let path = format!("/{path}");
    let addr = authority
        .to_socket_addrs()
        .map_err(|err| err.to_string())?
        .next()
        .ok_or_else(|| format!("invalid host: {authority}"))?;
    let mut stream =
        TcpStream::connect_timeout(&addr, Duration::from_secs(3)).map_err(|err| err.to_string())?;
    stream.set_read_timeout(Some(Duration::from_secs(30))).ok();
    stream.set_write_timeout(Some(Duration::from_secs(30))).ok();
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
    stream
        .write_all(head.as_bytes())
        .map_err(|err| err.to_string())?;
    if let Some(body) = body {
        stream.write_all(body).map_err(|err| err.to_string())?;
    }
    let mut raw = Vec::new();
    let mut chunk = [0u8; 16 * 1024];
    loop {
        if raw.len() > MAX_BODY + 4096 {
            return Err("the device response is larger than 64 MiB".into());
        }
        match stream.read(&mut chunk) {
            Ok(0) => break,
            Ok(count) => raw.extend_from_slice(&chunk[..count]),
            Err(err) if err.kind() == std::io::ErrorKind::TimedOut && !raw.is_empty() => break,
            Err(err) => return Err(err.to_string()),
        }
    }
    let split = raw
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .ok_or("the device response has no HTTP header")?;
    let header = String::from_utf8_lossy(&raw[..split]).to_string();
    let status: u16 = header
        .split_whitespace()
        .nth(1)
        .and_then(|code| code.parse().ok())
        .unwrap_or(0);
    if !(200..300).contains(&status) {
        return Err(format!("HTTP {status} for {path}"));
    }
    let body = &raw[split + 4..];
    if header
        .to_ascii_lowercase()
        .contains("transfer-encoding: chunked")
    {
        decode_chunks(body)
    } else if let Some(length) = content_length(&header) {
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

fn content_length(header: &str) -> Option<usize> {
    header.lines().find_map(|line| {
        let (name, value) = line.split_once(':')?;
        if name.eq_ignore_ascii_case("content-length") {
            value.trim().parse().ok()
        } else {
            None
        }
    })
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
            break;
        }
        if data.len() < size {
            return Err("truncated chunk".into());
        }
        out.extend_from_slice(&data[..size]);
        data = data.get(size + 2..).unwrap_or(&[]);
        if out.len() > MAX_BODY {
            return Err("the device response is larger than 64 MiB".into());
        }
    }
    Ok(out)
}

fn download_files(
    host: &str,
    remote: &str,
    dest: &Path,
    keep: &mut dyn FnMut(u64, u64) -> bool,
) -> Result<usize, String> {
    let mut count = 0;
    let mut tick = |count: usize| -> Result<(), String> {
        if keep(count as u64, 0) {
            Ok(())
        } else {
            Err("cancelled".into())
        }
    };
    tick(count)?;
    for name in [
        "input_map.csv",
        "spot_data.csv",
        "termination_summary.txt",
        "session_info.json",
        "session_meta.json",
        "SessionLogFile.log",
    ] {
        if fetch(host, &format!("{remote}/{name}"), &dest.join(name))? {
            count += 1;
            tick(count)?;
        }
    }
    for rel in [
        "config/map2map/devices.xml",
        "config/map2map/Output.xml",
        "config/map2map/Input.xml",
        "config/map2map/Database.xml",
        "config/map2map/Report.xml",
        "config/map2map/tolerances.xml",
        "config/map2map/scan_dose_system.xml",
    ] {
        if fetch(host, &format!("{remote}/{rel}"), &dest.join(rel))? {
            count += 1;
            tick(count)?;
        }
    }
    let mut empty_layers = 0;
    for layer in 0..MAX_LAYER {
        tick(count)?;
        let mut layer_hits = 0;
        for run in 0..MAX_RUN {
            let mut run_hits = 0;
            let prefix = format!("layer-{layer}/run-{run}");
            for name in [
                "timeslice_data_device_units.csv",
                "timeslice_data.csv",
                "FX4_spot_data.csv",
                "IX256_1_spot_data.csv",
                "IX256_2_spot_data.csv",
                "RCI_spot_data.csv",
                "RCI_map.csv",
                "TX2_spot_data.csv",
            ] {
                if fetch(
                    host,
                    &format!("{remote}/{prefix}/{name}"),
                    &dest.join(&prefix).join(name),
                )? {
                    count += 1;
                    tick(count)?;
                    run_hits += 1;
                    layer_hits += 1;
                }
            }
            if run_hits == 0 && run > 0 {
                break;
            }
        }
        if layer_hits == 0 {
            empty_layers += 1;
            if empty_layers >= 2 {
                break;
            }
        } else {
            empty_layers = 0;
        }
    }
    Ok(count)
}

fn fetch(host: &str, remote: &str, dest: &Path) -> Result<bool, String> {
    let url = match device_file_url(host, remote) {
        Ok(url) => url,
        Err(_) => return Ok(false),
    };
    let Ok(bytes) = http(&url, "GET", None) else {
        return Ok(false);
    };
    if bytes.is_empty() {
        return Ok(false);
    }
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent).map_err(|err| err.to_string())?;
    }
    std::fs::write(dest, bytes).map_err(|err| err.to_string())?;
    Ok(true)
}

fn ensure_session_info(dir: &Path, session_id: &str) -> Result<(), String> {
    let dest = dir.join("session_info.json");
    if dest.is_file() && dest.metadata().map(|meta| meta.len() > 0).unwrap_or(false) {
        return Ok(());
    }
    let payload = json!({
        "DCSState": "FnStateMapComplete",
        "LayerID": "0",
        "Run": "0",
        "SessionID": session_id,
    });
    std::fs::write(dest, format!("{payload:#}\n")).map_err(|err| err.to_string())
}

fn ensure_spot_data(dir: &Path) -> Result<(), String> {
    let dest = dir.join("spot_data.csv");
    if dest.is_file() && dest.metadata().map(|meta| meta.len() > 0).unwrap_or(false) {
        return Ok(());
    }
    // ponytail: joins the device spot files when the RCI omitted spot_data.csv.
    // Iso conversion from devices.xml stays with the Python session builder.
    let Some(table) = merge_spot_files(dir) else {
        return Ok(());
    };
    std::fs::write(dest, table).map_err(|err| err.to_string())
}

fn merge_spot_files(dir: &Path) -> Option<String> {
    let sources = [
        ("IX256_1_spot_data.csv", IX1),
        ("IX256_2_spot_data.csv", IX2),
        ("FX4_spot_data.csv", FX4),
        ("RCI_spot_data.csv", RCI),
    ];
    let mut rows: BTreeMap<(String, String), BTreeMap<String, String>> = BTreeMap::new();
    let mut columns = Vec::new();
    for layer in 0..MAX_LAYER {
        for run in 0..MAX_RUN {
            let run_dir = dir
                .join(format!("layer-{layer}"))
                .join(format!("run-{run}"));
            if !run_dir.is_dir() {
                continue;
            }
            for (name, rename) in sources {
                let Some(table) = read_csv(&run_dir.join(name)) else {
                    continue;
                };
                for record in table {
                    let spot = record.get("spot_no")?.clone();
                    let layer_id = record.get("layer_id")?.clone();
                    let entry = rows.entry((layer_id.clone(), spot.clone())).or_default();
                    entry.entry("layer_id".into()).or_insert(layer_id);
                    entry.entry("spot_no".into()).or_insert(spot);
                    for (src, dst) in rename {
                        let Some(value) = record.get(*src) else {
                            continue;
                        };
                        if !columns.contains(&(*dst).to_owned())
                            && *dst != "spot_no"
                            && *dst != "layer_id"
                        {
                            columns.push((*dst).to_owned());
                        }
                        entry
                            .entry((*dst).to_owned())
                            .or_insert_with(|| value.clone());
                    }
                }
            }
        }
    }
    if rows.is_empty() {
        return None;
    }
    let mut header = vec!["layer_id".to_owned(), "spot_no".to_owned()];
    header.extend(columns);
    let mut out = header.join(",");
    out.push('\n');
    for row in rows.values() {
        let cells: Vec<String> = header
            .iter()
            .map(|name| csv_cell(row.get(name).map(String::as_str).unwrap_or("")))
            .collect();
        out.push_str(&cells.join(","));
        out.push('\n');
    }
    Some(out)
}

fn read_csv(path: &Path) -> Option<Vec<HashMap<String, String>>> {
    let mut reader = csv::ReaderBuilder::new()
        .trim(csv::Trim::All)
        .from_path(path)
        .ok()?;
    let headers = reader.headers().ok()?.clone();
    let mut rows = Vec::new();
    for record in reader.records() {
        let record = record.ok()?;
        let mut row = HashMap::new();
        for (index, name) in headers.iter().enumerate() {
            row.insert(name.to_owned(), record.get(index).unwrap_or("").to_owned());
        }
        rows.push(row);
    }
    if rows.is_empty() {
        None
    } else {
        Some(rows)
    }
}

fn csv_cell(value: &str) -> String {
    if value.contains([',', '"', '\n']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_owned()
    }
}

const IX1: &[(&str, &str)] = &[
    ("spot_no", "spot_no"),
    ("layer_id", "layer_id"),
    ("timeslice_number", "timesliceNumber"),
    ("timeslice_timestamp(ms)", "timestamp"),
    ("point_time(ms)", "point_time(ms)"),
    ("ic1_position_measured_b", "r_ic1_x_spot_position_raw"),
    ("ic1_position_measured_a", "r_ic1_y_spot_position_raw"),
    ("ic1_sigma_measured_b(mm)", "r_ic1_x_spot_sigma_raw"),
    ("ic1_sigma_measured_a(mm)", "r_ic1_y_spot_sigma_raw"),
    ("total_dose(nC)", "ic1_total_dose_spot_raw"),
];
const IX2: &[(&str, &str)] = &[
    ("spot_no", "spot_no"),
    ("layer_id", "layer_id"),
    ("timeslice_number", "timesliceNumber"),
    ("timeslice_timestamp(ms)", "timestamp"),
    ("point_time(ms)", "point_time(ms)"),
    ("ic2_position_measured_b", "r_ic2_x_spot_position_raw"),
    ("ic2_position_measured_a", "r_ic2_y_spot_position_raw"),
    ("ic2_sigma_measured_b(mm)", "r_ic2_x_spot_sigma_raw"),
    ("ic2_sigma_measured_a(mm)", "r_ic2_y_spot_sigma_raw"),
    ("total_dose(nC)", "ic2_total_dose_spot_raw"),
];
const FX4: &[(&str, &str)] = &[
    ("spot_no", "spot_no"),
    ("layer_id", "layer_id"),
    ("timeslice_number", "timesliceNumber"),
    ("timeslice_timestamp(ms)", "timestamp"),
    ("point_time(ms)", "point_time(ms)"),
    ("total_dose(nC)", "r_ic3_total_dose_spot_raw"),
];
const RCI: &[(&str, &str)] = &[
    ("spot_no", "spot_no"),
    ("layer_id", "layer_id"),
    ("timeslice_number", "timesliceNumber"),
    ("timeslice_timestamp(ms)", "timestamp"),
    ("point_time(ms)", "point_time(ms)"),
    ("beam_current_command", "c_current_rci"),
];

fn package_zip(dir: &Path, zip_path: &Path, session_id: &str) -> Result<(), String> {
    if let Some(parent) = zip_path.parent() {
        std::fs::create_dir_all(parent).map_err(|err| err.to_string())?;
    }
    let file = std::fs::File::create(zip_path).map_err(|err| err.to_string())?;
    let mut zip = zip::ZipWriter::new(file);
    let options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);
    let mut files = Vec::new();
    collect_files(dir, &mut files)?;
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
    Ok(())
}

fn collect_files(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    for item in std::fs::read_dir(dir).map_err(|err| err.to_string())? {
        let path = item.map_err(|err| err.to_string())?.path();
        if path.is_dir() {
            collect_files(&path, out)?;
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
    fn a_missing_spot_file_is_joined_and_zipped() {
        let dir = std::env::temp_dir().join(format!("scan-kit-spots-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let run = dir.join("layer-0").join("run-0");
        std::fs::create_dir_all(&run).unwrap();
        std::fs::write(
            run.join("IX256_1_spot_data.csv"),
            "spot_no,layer_id,total_dose(nC)\n1,0,3.5\n",
        )
        .unwrap();
        std::fs::write(
            run.join("RCI_spot_data.csv"),
            "spot_no,layer_id,beam_current_command\n1,0,9\n",
        )
        .unwrap();
        ensure_spot_data(&dir).unwrap();
        ensure_session_info(&dir, "sess").unwrap();
        let text = std::fs::read_to_string(dir.join("spot_data.csv")).unwrap();
        assert!(text.contains("ic1_total_dose_spot_raw"));
        assert!(text.contains("c_current_rci"));
        assert!(text.contains("3.5"));
        let zip_path =
            std::env::temp_dir().join(format!("scan-kit-spots-{}.zip", std::process::id()));
        package_zip(&dir, &zip_path, "sess").unwrap();
        let file = std::fs::File::open(&zip_path).unwrap();
        let mut archive = zip::ZipArchive::new(file).unwrap();
        assert!(archive.by_name("sess/spot_data.csv").is_ok());
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
        let body = connected_body("192.168.1.2", &json!(7), &Value::Null, "/sessions");
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

        assert_eq!(content_length("Host: x\r\nContent-Length: 4\r\n"), Some(4));
        assert_eq!(content_length("Host: x"), None);
        assert_eq!(decode_chunks(b"5\r\nhello\r\n0\r\n\r\n").unwrap(), b"hello");
        assert!(decode_chunks(b"zz").is_err());
        assert!(decode_chunks(b"5\r\nhi").is_err());
        assert!(decode_chunks(b"5\r\nhello\r\n\xff\r\n").is_err());
        assert_eq!(csv_cell("a,b"), "\"a,b\"");
        assert_eq!(csv_cell("say \"hi\""), "\"say \"\"hi\"\"\"");
        assert_eq!(csv_cell("plain"), "plain");
        assert!(read_csv(Path::new("missing-scan-kit.csv")).is_none());

        let dir = std::env::temp_dir().join(format!("scan-kit-codec-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("empty.csv"), "spot_no\n").unwrap();
        assert!(read_csv(&dir.join("empty.csv")).is_none());
        ensure_session_info(&dir, "sess").unwrap();
        ensure_session_info(&dir, "sess").unwrap();
        ensure_spot_data(&dir).unwrap();
        let _ = std::fs::remove_dir_all(&dir);

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
            if request.contains("input_map.csv") {
                return http_response(200, "OK", b"x");
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

        let files = std::env::temp_dir().join(format!("scan-kit-dl-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&files);
        let err = download_files(&host, "session", &files, &mut |_, _| false).unwrap_err();
        assert!(err.contains("cancelled"));
        let count = download_files(&host, "session", &files, &mut |_, _| true).unwrap();
        assert!(count >= 1);
        assert!(files.join("input_map.csv").is_file());
        let _ = std::fs::remove_dir_all(&files);
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

    fn accept_mpack(mut stream: TcpStream) -> WebSocket<TcpStream> {
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
}
