//! A library root: a local path, a UNC share, or a remote URL.
//!
//! Passwords stay in a process map. They are never written to sqlite or prefs.
//! `ssh://` and `scp://` open as SFTP. A one-letter scheme is a drive letter.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use sha2::{Digest, Sha256};

use crate::discover::{self, Discovered, Found};
use crate::store::canonical_local;

pub const AUTH_REQUIRED: &str = "authentication required";

/// ponytail: a session eight folders below the library root is still found.
/// Raise this to walk a deeper library.
pub(crate) const NEST_DEPTH: u32 = 8;

const CONNECT_TIMEOUT: Duration = Duration::from_secs(20);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Scheme {
    Sftp,
    Ftp,
    Ftps,
    Smb,
    Http,
    Https,
}

#[derive(Clone, Debug)]
pub(crate) struct Remote {
    pub canonical: String,
    pub scheme: Scheme,
    pub username: String,
    pub password: Option<String>,
    pub host: String,
    pub port: Option<u16>,
    pub path: String,
}

enum Kind {
    Local(PathBuf),
    Remote(Remote),
}

pub(crate) enum Opened {
    Local(PathBuf),
    Remote(String),
}

pub fn remember_password(spec: &str, password: &str) {
    let Ok(Kind::Remote(remote)) = classify(spec.trim()) else {
        return;
    };
    store_password(&remote, password);
}

pub fn is_remote_location(spec: &str) -> bool {
    let text = spec.trim();
    let Some((scheme, _)) = split_scheme(text) else {
        return false;
    };
    scheme.len() > 1 && !scheme.eq_ignore_ascii_case("file")
}

pub(crate) fn canonical_root(spec: &str) -> Result<String, String> {
    let spec = spec.trim();
    if spec.is_empty() {
        return Err("choose a folder or URL".into());
    }
    match classify(spec)? {
        Kind::Local(path) => Ok(canonical_local(&path)),
        Kind::Remote(remote) => {
            if let Some(password) = remote.password.clone() {
                store_password(&remote, &password);
            }
            Ok(remote.canonical)
        }
    }
}

pub(crate) fn library_key(library: &Path) -> String {
    let text = library.to_string_lossy();
    canonical_root(&text).unwrap_or_else(|_| text.into_owned())
}

/// Local directory to walk, or the canonical URL when the listing is remote.
pub(crate) fn open_root(spec: &str) -> Result<(String, Opened), String> {
    let spec = spec.trim();
    if spec.is_empty() {
        return Err("choose a folder or URL".into());
    }
    match classify(spec)? {
        Kind::Local(path) => {
            let root = canonical_local(&path);
            let absolute = PathBuf::from(&root);
            if !absolute.is_dir() {
                return Err(format!("{root} is not a directory"));
            }
            Ok((root, Opened::Local(absolute)))
        }
        Kind::Remote(remote) if remote.scheme == Scheme::Smb => {
            if let Some(password) = remote.password.clone() {
                store_password(&remote, &password);
            }
            let unc = connect_smb(&remote)?;
            Ok((remote.canonical, Opened::Local(unc)))
        }
        Kind::Remote(remote) => {
            if let Some(password) = remote.password.clone() {
                store_password(&remote, &password);
            }
            Ok((remote.canonical.clone(), Opened::Remote(remote.canonical)))
        }
    }
}

pub(crate) fn discover_remote(canonical: &str) -> Result<Found, String> {
    let remote = require_remote(canonical)?;
    if matches!(remote.scheme, Scheme::Http | Scheme::Https) {
        return http_archive(&remote);
    }
    let mut fs = crate::remote::connect(&remote, &password_of(&remote))?;
    let mut entries = Vec::new();
    let mut skipped = Vec::new();
    let mut archives = Vec::new();
    let mut exams = Vec::new();
    let root_name = remote
        .path
        .trim_end_matches('/')
        .rsplit('/')
        .next()
        .filter(|name| !name.is_empty())
        .unwrap_or("library");
    if is_remote_session(&mut *fs, &remote.path, root_name)? {
        discover::add_directory(
            &mut entries,
            &mut skipped,
            Discovered {
                session_id: root_name.to_owned(),
                storage_path: PathBuf::from(canonical),
                kind: "directory",
            },
            root_name,
        );
    } else if let Some(exam) = remote_exam(&mut *fs, &remote.path, root_name, canonical)? {
        discover::add_exam(&mut exams, &mut skipped, exam, root_name);
    } else {
        walk_remote(
            &mut *fs,
            canonical,
            &remote.path,
            "",
            0,
            &mut entries,
            &mut skipped,
            &mut archives,
            &mut exams,
        )?;
    }
    discover::add_archives(&mut entries, &mut skipped, archives);
    exams.sort_by(|left, right| left.folder_name.cmp(&right.folder_name));
    Ok(Found {
        entries,
        skipped,
        exams,
    })
}

pub(crate) fn read_remote_session_file(
    session_url: &str,
    session_id: &str,
    filename: &str,
) -> Option<Vec<u8>> {
    let remote = require_remote(session_url).ok()?;
    if archive_name(&remote.path) {
        let local = materialize(session_url, session_url).ok()?;
        return discover::read_session_file(&local, session_id, filename);
    }
    let mut fs = crate::remote::connect(&remote, &password_of(&remote)).ok()?;
    let base = remote.path.trim_end_matches('/');
    for path in [
        format!("{base}/{filename}"),
        format!("{base}/{session_id}/{filename}"),
    ] {
        if let Ok(Some(bytes)) = fs.read(&path) {
            return Some(bytes);
        }
    }
    None
}

pub(crate) fn remote_fingerprint(session_url: &str, session_id: &str) -> Option<(i64, i64)> {
    let remote = require_remote(session_url).ok()?;
    let mut fs = crate::remote::connect(&remote, &password_of(&remote)).ok()?;
    let base = remote.path.trim_end_matches('/');
    let mut candidates = vec![base.to_owned()];
    if !archive_name(&remote.path) {
        candidates.insert(0, format!("{base}/termination_summary.txt"));
        candidates.insert(1, format!("{base}/{session_id}/termination_summary.txt"));
    }
    for path in candidates {
        let stat = fs.stat(&path).ok()??;
        if archive_name(&remote.path) || !stat.dir {
            return Some((stat.mtime_ns, i64::try_from(stat.size).ok()?));
        }
    }
    None
}

pub(crate) fn recorded_directory(library: &Path, session_id: &str) -> Option<PathBuf> {
    let key = library_key(library);
    let place = recorded(&key, session_id)?;
    let local = if is_remote_location(&place) {
        materialize(&key, &place).ok()?
    } else {
        let path = PathBuf::from(&place);
        if !path.exists() {
            return None;
        }
        path
    };
    Some(resolve_container(&local, session_id))
}

pub(crate) fn catalog_directory(library: &Path, session_id: &str) -> Option<PathBuf> {
    let key = library_key(library);
    let hit = {
        let map = catalog().lock().unwrap_or_else(|err| err.into_inner());
        if !map.values().any(|row| row.library == key) {
            return None;
        }
        map.get(session_id).cloned()
    }?;
    let local = if is_remote_location(&hit.place) {
        materialize(&hit.library, &hit.place).ok()?
    } else {
        let path = PathBuf::from(&hit.place);
        if !path.exists() {
            return None;
        }
        path
    };
    Some(resolve_container(&local, &hit.session_id))
}

pub(crate) fn replace_catalog(entries: &[(String, String, String, String)]) {
    let mut map = catalog().lock().unwrap_or_else(|err| err.into_inner());
    map.clear();
    for (key, library, place, session_id) in entries {
        if key.is_empty() || place.is_empty() {
            continue;
        }
        map.insert(
            key.clone(),
            CatalogHit {
                library: library.clone(),
                place: place.clone(),
                session_id: session_id.clone(),
            },
        );
    }
}

pub(crate) fn replace_sessions(library: &str, places: &[(String, String)]) {
    let mut map = sessions().lock().unwrap_or_else(|err| err.into_inner());
    map.retain(|(root, _), _| root != library);
    for (session_id, place) in places {
        map.insert((library.to_owned(), session_id.clone()), place.clone());
    }
}

pub(crate) fn cache_directory(library_canonical: &str) -> PathBuf {
    home_dir()
        .join(".scan-kit")
        .join("remote-cache")
        .join(hex_sha256(library_canonical))
}

#[cfg(test)]
pub(crate) fn install_fixture(
    spec: &str,
    files: &[(&str, &[u8])],
    auth: bool,
) -> Result<String, String> {
    let remote = require_remote(spec)?;
    crate::remote::install_fixture(&remote.canonical, &remote.path, files, auth);
    Ok(remote.canonical)
}

#[cfg(test)]
pub(crate) fn fixture_copies(spec: &str) -> u32 {
    let Ok(remote) = require_remote(spec) else {
        return 0;
    };
    crate::remote::fixture_copies(&remote.canonical)
}

fn materialize(library_canonical: &str, session_url: &str) -> Result<PathBuf, String> {
    let session = require_remote(session_url)?;
    let relative = relative_remote(library_canonical, &session);
    let dest = cache_directory(library_canonical).join(&relative);
    let stamp_text = remote_stamp(&session)?;
    let stamp_path = stamp_file(&dest);
    if dest.exists() && fs::read_to_string(&stamp_path).ok().as_deref() == Some(stamp_text.as_str())
    {
        return Ok(dest);
    }
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).map_err(|err| err.to_string())?;
    }
    let mut fs = crate::remote::connect(&session, &password_of(&session))?;
    fs.copy_to(&session.path, &dest)?;
    if let Some(parent) = stamp_path.parent() {
        fs::create_dir_all(parent).map_err(|err| err.to_string())?;
    }
    fs::write(&stamp_path, &stamp_text).map_err(|err| err.to_string())?;
    Ok(dest)
}

fn remote_stamp(session: &Remote) -> Result<String, String> {
    let mut fs = crate::remote::connect(session, &password_of(session))?;
    let stat = fs.stat(&session.path)?.unwrap_or(crate::remote::Stat {
        dir: true,
        size: 0,
        mtime_ns: 0,
    });
    Ok(format!("{}\n{}", stat.mtime_ns, stat.size))
}

fn stamp_file(dest: &Path) -> PathBuf {
    let name = dest.file_name().unwrap_or_default().to_string_lossy();
    dest.parent().unwrap_or(dest).join(format!(".{name}.stamp"))
}

fn relative_remote(library_canonical: &str, session: &Remote) -> String {
    let lib_path = require_remote(library_canonical)
        .map(|remote| remote.path)
        .unwrap_or_default();
    let path = session.path.trim_end_matches('/');
    let lib = lib_path.trim_end_matches('/');
    let rel = path
        .strip_prefix(lib)
        .unwrap_or(path)
        .trim_start_matches('/');
    if rel.is_empty() {
        path.rsplit('/')
            .next()
            .filter(|name| !name.is_empty())
            .unwrap_or("session")
            .to_owned()
    } else {
        rel.to_owned()
    }
}

fn resolve_container(container: &Path, session_id: &str) -> PathBuf {
    let inner = container.join(session_id);
    if inner.join("input_map.csv").is_file() {
        return inner;
    }
    container.to_path_buf()
}

fn http_archive(remote: &Remote) -> Result<Found, String> {
    let name = remote
        .path
        .rsplit('/')
        .next()
        .filter(|name| !name.is_empty())
        .unwrap_or("");
    let Some((session_id, kind)) = discover::archive_kind(name) else {
        return Err("an http location is one archive, not a folder".into());
    };
    Ok(Found {
        entries: vec![Discovered {
            session_id,
            storage_path: PathBuf::from(&remote.canonical),
            kind,
        }],
        skipped: Vec::new(),
        exams: Vec::new(),
    })
}

fn walk_remote(
    fs: &mut dyn crate::remote::Fs,
    library_url: &str,
    path: &str,
    rel: &str,
    depth: u32,
    entries: &mut Vec<Discovered>,
    skipped: &mut Vec<String>,
    archives: &mut Vec<(Discovered, String)>,
    exams: &mut Vec<discover::ExamHit>,
) -> Result<(), String> {
    if depth >= NEST_DEPTH {
        return Ok(());
    }
    let mut dirs = Vec::new();
    let mut files = Vec::new();
    for item in fs.list(path)? {
        if item.name.is_empty() || item.name == "." || item.name == ".." {
            continue;
        }
        if item.dir {
            dirs.push(item.name);
        } else {
            files.push(item.name);
        }
    }
    dirs.sort();
    files.sort();
    let mut sessions = Vec::new();
    for name in &dirs {
        let child = join_path(path, name);
        let child_rel = join_rel(rel, name);
        if is_remote_session(fs, &child, name)? {
            let url = join_rel(library_url, &child_rel);
            discover::add_directory(
                entries,
                skipped,
                Discovered {
                    session_id: name.clone(),
                    storage_path: PathBuf::from(&url),
                    kind: "directory",
                },
                &child_rel,
            );
            sessions.push(name.clone());
        }
    }
    for name in &files {
        let Some((session_id, kind)) = discover::archive_kind(name) else {
            continue;
        };
        let child_rel = join_rel(rel, name);
        let url = join_rel(library_url, &child_rel);
        archives.push((
            Discovered {
                session_id,
                storage_path: PathBuf::from(url),
                kind,
            },
            child_rel,
        ));
    }
    for name in &dirs {
        if sessions.iter().any(|session| session == name) {
            continue;
        }
        let child = join_path(path, name);
        let child_rel = join_rel(rel, name);
        let url = join_rel(library_url, &child_rel);
        if let Some(exam) = remote_exam(fs, &child, name, &url)? {
            discover::add_exam(exams, skipped, exam, &child_rel);
            continue;
        }
        walk_remote(
            fs,
            library_url,
            &child,
            &child_rel,
            depth + 1,
            entries,
            skipped,
            archives,
            exams,
        )?;
    }
    Ok(())
}

fn remote_exam(
    fs: &mut dyn crate::remote::Fs,
    path: &str,
    folder_name: &str,
    storage_url: &str,
) -> Result<Option<discover::ExamHit>, String> {
    let mut files = Vec::new();
    let mut count = 0i64;
    for item in fs.list(path)? {
        if item.name.is_empty() || item.name == "." || item.name == ".." {
            continue;
        }
        count += 1;
        if !item.dir {
            files.push(item.name);
        }
    }
    files.sort();
    let mut sample = None;
    for name in &files {
        let child = join_path(path, name);
        if dicom_name(name) || prefix_dicom(fs, &child)? {
            sample = Some(child);
            break;
        }
    }
    let Some(sample) = sample else {
        return Ok(None);
    };
    let bytes = fs
        .read_prefix(&sample, 4 * 1024 * 1024)?
        .unwrap_or_default();
    let peek = scan_kit_dicom::peek_study(&bytes);
    let study_uid = if peek.study_uid.is_empty() {
        folder_name.to_owned()
    } else {
        peek.study_uid
    };
    Ok(Some(discover::ExamHit {
        study_uid,
        folder_name: folder_name.to_owned(),
        storage_path: PathBuf::from(storage_url),
        patient_name: peek.patient_name,
        patient_id: peek.patient_id,
        study_date: peek.study_date,
        study_description: peek.study_description,
        file_count: count,
    }))
}

fn dicom_name(name: &str) -> bool {
    name.to_ascii_lowercase().ends_with(".dcm")
}

fn prefix_dicom(fs: &mut dyn crate::remote::Fs, path: &str) -> Result<bool, String> {
    let Some(bytes) = fs.read_prefix(path, 132)? else {
        return Ok(false);
    };
    Ok(scan_kit_dicom::looks_like_dicom(&bytes))
}

fn is_remote_session(
    fs: &mut dyn crate::remote::Fs,
    path: &str,
    name: &str,
) -> Result<bool, String> {
    let base = path.trim_end_matches('/');
    Ok(is_file(fs, &format!("{base}/input_map.csv"))?
        || is_file(fs, &format!("{base}/{name}/input_map.csv"))?)
}

fn is_file(fs: &mut dyn crate::remote::Fs, path: &str) -> Result<bool, String> {
    Ok(fs.stat(path)?.is_some_and(|stat| !stat.dir))
}

fn join_path(dir: &str, name: &str) -> String {
    format!("{}/{}", dir.trim_end_matches('/'), name.trim_matches('/'))
}

fn join_rel(dir: &str, name: &str) -> String {
    if dir.is_empty() {
        name.to_owned()
    } else {
        format!("{}/{}", dir.trim_end_matches('/'), name.trim_matches('/'))
    }
}

fn archive_name(path: &str) -> bool {
    let name = path.rsplit('/').next().unwrap_or(path);
    discover::archive_kind(name).is_some()
}

fn require_remote(spec: &str) -> Result<Remote, String> {
    match classify(spec.trim())? {
        Kind::Remote(remote) => Ok(remote),
        Kind::Local(path) => Err(format!("{} is not a remote URL", path.display())),
    }
}

fn classify(spec: &str) -> Result<Kind, String> {
    let Some((scheme, rest)) = split_scheme(spec) else {
        return Ok(Kind::Local(expand_user(spec)));
    };
    if scheme.len() == 1 {
        return Ok(Kind::Local(PathBuf::from(format!("{scheme}:{rest}"))));
    }
    if scheme.eq_ignore_ascii_case("file") {
        return Ok(Kind::Local(file_uri_path(rest)));
    }
    let scheme = match scheme.to_ascii_lowercase().as_str() {
        "ssh" | "scp" | "sftp" => Scheme::Sftp,
        "ftp" => Scheme::Ftp,
        "ftps" => Scheme::Ftps,
        "smb" => Scheme::Smb,
        "http" => Scheme::Http,
        "https" => Scheme::Https,
        other => {
            return Err(format!(
                "unsupported location scheme {other}. Use a folder, UNC path, or sftp://, smb://, ftp://, or http:// URL"
            ));
        }
    };
    Ok(Kind::Remote(parse_remote(scheme, rest)?))
}

fn parse_remote(scheme: Scheme, rest: &str) -> Result<Remote, String> {
    let rest = rest.split('#').next().unwrap_or(rest);
    let (rest, query) = rest.split_once('?').unwrap_or((rest, ""));
    let (authority, raw_path) = match rest.find('/') {
        Some(index) => (&rest[..index], &rest[index..]),
        None => (rest, "/"),
    };
    if authority.is_empty() {
        return Err("the URL needs a host".into());
    }
    let (userinfo, hostport) = match authority.rfind('@') {
        Some(index) => (Some(&authority[..index]), &authority[index + 1..]),
        None => (None, authority),
    };
    let (username, password) = match userinfo {
        Some(info) => {
            let info = percent_decode(info);
            match info.split_once(':') {
                Some((user, password)) => (user.to_owned(), Some(password.to_owned())),
                None => (info, None),
            }
        }
        None => (String::new(), None),
    };
    let (host, port) = split_host_port(hostport)?;
    if host.is_empty() {
        return Err("the URL needs a host".into());
    }
    let host = host.to_ascii_lowercase();
    let mut path = percent_decode(raw_path);
    if !path.starts_with('/') {
        path.insert(0, '/');
    }
    if path.len() > 1 {
        path = path.trim_end_matches('/').to_owned();
    }
    let mut remote = Remote {
        canonical: String::new(),
        scheme,
        username,
        password,
        host,
        port,
        path,
    };
    remote.canonical = canonical_url(&remote, query);
    Ok(remote)
}

fn canonical_url(remote: &Remote, query: &str) -> String {
    let mut url = format!(
        "{}://{}{}{}",
        scheme_name(remote.scheme),
        user_at(&remote.username),
        host_text(&remote.host),
        port_text(remote.port)
    );
    url.push_str(&remote.path);
    if !query.is_empty() {
        url.push('?');
        url.push_str(query);
    }
    url
}

fn scheme_name(scheme: Scheme) -> &'static str {
    match scheme {
        Scheme::Sftp => "sftp",
        Scheme::Ftp => "ftp",
        Scheme::Ftps => "ftps",
        Scheme::Smb => "smb",
        Scheme::Http => "http",
        Scheme::Https => "https",
    }
}

fn user_at(username: &str) -> String {
    if username.is_empty() {
        String::new()
    } else {
        format!("{}@", encode_user(username))
    }
}

fn host_text(host: &str) -> String {
    if host.contains(':') && !host.starts_with('[') {
        format!("[{host}]")
    } else {
        host.to_owned()
    }
}

fn port_text(port: Option<u16>) -> String {
    port.map(|port| format!(":{port}")).unwrap_or_default()
}

fn host_key(remote: &Remote) -> String {
    format!(
        "{}://{}{}{}",
        scheme_name(remote.scheme),
        user_at(&remote.username),
        host_text(&remote.host),
        port_text(remote.port)
    )
}

fn split_host_port(hostport: &str) -> Result<(String, Option<u16>), String> {
    if let Some(rest) = hostport.strip_prefix('[') {
        let end = rest
            .find(']')
            .ok_or_else(|| "the URL host is missing a closing bracket".to_owned())?;
        let host = rest[..end].to_owned();
        let after = &rest[end + 1..];
        if after.is_empty() {
            return Ok((host, None));
        }
        let port = after
            .strip_prefix(':')
            .ok_or_else(|| "the URL port is not a number".to_owned())?;
        return Ok((host, Some(parse_port(port)?)));
    }
    if let Some((host, port)) = hostport.rsplit_once(':') {
        if !host.contains(':') && !port.is_empty() && port.bytes().all(|byte| byte.is_ascii_digit())
        {
            return Ok((host.to_owned(), Some(parse_port(port)?)));
        }
    }
    Ok((hostport.to_owned(), None))
}

fn parse_port(text: &str) -> Result<u16, String> {
    text.parse::<u16>()
        .map_err(|_| "the URL port is not a number".to_owned())
}

fn split_scheme(text: &str) -> Option<(&str, &str)> {
    let bytes = text.as_bytes();
    if !bytes.first().is_some_and(|byte| byte.is_ascii_alphabetic()) {
        return None;
    }
    let mut index = 1;
    while index < bytes.len()
        && (bytes[index].is_ascii_alphanumeric()
            || bytes[index] == b'+'
            || bytes[index] == b'.'
            || bytes[index] == b'-')
    {
        index += 1;
    }
    if bytes.get(index..index + 3) != Some(b"://") {
        return None;
    }
    Some((&text[..index], &text[index + 3..]))
}

fn file_uri_path(rest: &str) -> PathBuf {
    let rest = percent_decode(rest);
    if let Some(path) = rest.strip_prefix('/') {
        if path.len() >= 2 && path.as_bytes().get(1) == Some(&b':') {
            return PathBuf::from(path);
        }
        return PathBuf::from(format!("/{path}"));
    }
    match rest.split_once('/') {
        Some((host, path)) if !host.is_empty() => {
            PathBuf::from(format!("\\\\{host}\\{}", path.replace('/', "\\")))
        }
        _ => PathBuf::from(rest),
    }
}

fn expand_user(text: &str) -> PathBuf {
    if text == "~" {
        return home_dir();
    }
    if let Some(rest) = text.strip_prefix("~/").or_else(|| text.strip_prefix("~\\")) {
        return home_dir().join(rest);
    }
    PathBuf::from(text)
}

fn home_dir() -> PathBuf {
    for key in ["USERPROFILE", "HOME"] {
        if let Ok(path) = std::env::var(key) {
            if !path.is_empty() {
                return PathBuf::from(path);
            }
        }
    }
    PathBuf::from(".")
}

fn percent_decode(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' && index + 2 < bytes.len() {
            if let Ok(value) = u8::from_str_radix(
                std::str::from_utf8(&bytes[index + 1..index + 3]).unwrap_or(""),
                16,
            ) {
                out.push(value);
                index += 3;
                continue;
            }
        }
        out.push(bytes[index]);
        index += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn encode_user(text: &str) -> String {
    let mut out = String::new();
    for byte in text.bytes() {
        match byte {
            b'@' | b':' | b'/' | b'%' | b'?' | b'#' | b' ' => {
                out.push_str(&format!("%{byte:02X}"));
            }
            _ => out.push(byte as char),
        }
    }
    out
}

fn hex_sha256(text: &str) -> String {
    let digest = Sha256::digest(text.as_bytes());
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

pub(crate) fn auth_error(detail: &str) -> String {
    format!("{AUTH_REQUIRED}: {detail}")
}

fn passwords() -> &'static Mutex<HashMap<String, String>> {
    static PASSWORDS: OnceLock<Mutex<HashMap<String, String>>> = OnceLock::new();
    PASSWORDS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn sessions() -> &'static Mutex<HashMap<(String, String), String>> {
    static SESSIONS: OnceLock<Mutex<HashMap<(String, String), String>>> = OnceLock::new();
    SESSIONS.get_or_init(|| Mutex::new(HashMap::new()))
}

#[derive(Clone)]
struct CatalogHit {
    library: String,
    place: String,
    session_id: String,
}

fn catalog() -> &'static Mutex<HashMap<String, CatalogHit>> {
    static CATALOG: OnceLock<Mutex<HashMap<String, CatalogHit>>> = OnceLock::new();
    CATALOG.get_or_init(|| Mutex::new(HashMap::new()))
}

pub(crate) fn password_of(remote: &Remote) -> String {
    let map = passwords().lock().unwrap_or_else(|err| err.into_inner());
    if let Some(password) = map.get(&remote.canonical) {
        return password.clone();
    }
    map.get(&host_key(remote)).cloned().unwrap_or_default()
}

fn store_password(remote: &Remote, password: &str) {
    if password.is_empty() {
        return;
    }
    let mut map = passwords().lock().unwrap_or_else(|err| err.into_inner());
    map.insert(remote.canonical.clone(), password.to_owned());
    map.insert(host_key(remote), password.to_owned());
}

fn recorded(library: &str, session_id: &str) -> Option<String> {
    sessions()
        .lock()
        .unwrap_or_else(|err| err.into_inner())
        .get(&(library.to_owned(), session_id.to_owned()))
        .cloned()
}

#[cfg(not(windows))]
fn connect_smb(remote: &Remote) -> Result<PathBuf, String> {
    let _ = remote;
    Err("an smb URL opens as a Windows UNC path".into())
}

#[cfg(windows)]
fn connect_smb(remote: &Remote) -> Result<PathBuf, String> {
    let unc = unc_path(remote)?;
    let password = password_of(remote);
    if !password.is_empty() {
        add_windows_connection(&share_root(remote)?, &remote.username, &password)?;
    }
    match fs::read_dir(&unc) {
        Ok(_) => Ok(unc),
        Err(err) if err.kind() == std::io::ErrorKind::PermissionDenied => {
            Err(auth_error("the share refused the password"))
        }
        Err(_) => Err(format!("{} is not a directory", unc.display())),
    }
}

#[cfg(windows)]
fn unc_path(remote: &Remote) -> Result<PathBuf, String> {
    let path = remote.path.trim_start_matches('/').replace('/', "\\");
    if remote.host.is_empty() || path.is_empty() {
        return Err("the smb URL needs a host and a share".into());
    }
    Ok(PathBuf::from(format!("\\\\{}\\{path}", remote.host)))
}

#[cfg(windows)]
fn share_root(remote: &Remote) -> Result<String, String> {
    let share = remote
        .path
        .trim_start_matches('/')
        .split('/')
        .next()
        .filter(|share| !share.is_empty())
        .ok_or_else(|| "the smb URL needs a share".to_owned())?;
    Ok(format!("\\\\{}\\{share}", remote.host))
}

#[cfg(windows)]
fn add_windows_connection(share: &str, username: &str, password: &str) -> Result<(), String> {
    use std::ffi::OsStr;
    use std::os::windows::ffi::OsStrExt;

    #[repr(C)]
    struct NetResourceW {
        scope: u32,
        kind: u32,
        display: u32,
        usage: u32,
        local: *mut u16,
        remote: *mut u16,
        comment: *mut u16,
        provider: *mut u16,
    }

    #[link(name = "mpr")]
    extern "system" {
        fn WNetAddConnection2W(
            resource: *const NetResourceW,
            password: *const u16,
            username: *const u16,
            flags: u32,
        ) -> u32;
    }

    fn wide(text: &str) -> Vec<u16> {
        OsStr::new(text)
            .encode_wide()
            .chain(std::iter::once(0))
            .collect()
    }

    let mut remote = wide(share);
    let user = wide(username);
    let secret = wide(password);
    let resource = NetResourceW {
        scope: 0,
        kind: 1,
        display: 0,
        usage: 0,
        local: std::ptr::null_mut(),
        remote: remote.as_mut_ptr(),
        comment: std::ptr::null_mut(),
        provider: std::ptr::null_mut(),
    };
    let code = unsafe {
        WNetAddConnection2W(
            &resource,
            secret.as_ptr(),
            if username.is_empty() {
                std::ptr::null()
            } else {
                user.as_ptr()
            },
            0x4,
        )
    };
    if code == 0 || code == 1219 {
        return Ok(());
    }
    if matches!(code, 5 | 86 | 1323 | 1326 | 2202) {
        return Err(auth_error("the share refused the password"));
    }
    Err(format!("could not connect to {share} (error {code})"))
}

pub(crate) fn connect_timeout() -> Duration {
    CONNECT_TIMEOUT
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn urls_classify_and_drop_the_password() {
        let canonical =
            canonical_root("sftp://pyramid:secret@192.168.101.206/var/log/ptc_ex").unwrap();
        assert_eq!(canonical, "sftp://pyramid@192.168.101.206/var/log/ptc_ex");
        assert!(!canonical.contains("secret"));
        let remote = match classify(&canonical).unwrap() {
            Kind::Remote(remote) => remote,
            Kind::Local(_) => panic!("sftp is remote"),
        };
        assert_eq!(password_of(&remote), "secret");

        assert_eq!(
            canonical_root("ssh://user@scan-kit.invalid/data").unwrap(),
            "sftp://user@scan-kit.invalid/data"
        );
        assert_eq!(
            canonical_root("scp://user@scan-kit.invalid/data").unwrap(),
            "sftp://user@scan-kit.invalid/data"
        );
        assert_eq!(
            canonical_root("sftp://user@scan-kit.invalid:2222/data/").unwrap(),
            "sftp://user@scan-kit.invalid:2222/data"
        );
        assert_eq!(
            canonical_root("smb://user:pw@filer/share/ptc_ex").unwrap(),
            "smb://user@filer/share/ptc_ex"
        );
        assert!(is_remote_location("ftp://host.invalid/data"));
        assert!(is_remote_location("ftps://host.invalid/data"));
        assert!(is_remote_location("http://host.invalid/session.zip"));
        assert!(is_remote_location("https://host.invalid/session.zip"));
        assert!(!is_remote_location(r"\\192.168.101.206\share\ptc_ex"));
        assert!(!is_remote_location(r"C:\data"));
        assert!(!is_remote_location("C://data"));
        assert!(!is_remote_location("file:///C:/data"));
        assert!(canonical_root("nfs://host/data").is_err());
        assert_eq!(
            Path::new("sftp://pyramid@192.168.101.206/var/log/ptc_ex").to_string_lossy(),
            "sftp://pyramid@192.168.101.206/var/log/ptc_ex"
        );
    }
}
