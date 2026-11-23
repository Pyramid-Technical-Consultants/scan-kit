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

#[derive(Clone)]
pub(crate) struct Remote {
    pub canonical: String,
    /// Wire-only query (e.g. a presigned-URL signature). It stays out of
    /// `canonical` so nothing persisted or echoed carries a live credential;
    /// `connect` reattaches it via `request_url`/`query_of`.
    pub query: String,
    pub scheme: Scheme,
    pub username: String,
    pub password: Option<String>,
    pub host: String,
    pub port: Option<u16>,
    pub path: String,
}

// The derived Debug would dump the password into any future {remote:?} error.
impl std::fmt::Debug for Remote {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Remote")
            .field("canonical", &self.canonical)
            .field("query", &if self.query.is_empty() { "" } else { "…" })
            .field("scheme", &self.scheme)
            .field("username", &self.username)
            .field("password", &self.password.as_ref().map(|_| "…"))
            .field("host", &self.host)
            .field("port", &self.port)
            .field("path", &self.path)
            .finish()
    }
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

pub(crate) fn discover_remote(
    canonical: &str,
    keep: &mut dyn FnMut(u64, u64) -> bool,
) -> Result<Found, String> {
    let remote = require_remote(canonical)?;
    if matches!(remote.scheme, Scheme::Http | Scheme::Https) {
        return http_archive(&remote);
    }
    let mut fs = crate::remote::connect(&remote, &password_of(&remote))?;
    let mut entries = Vec::new();
    let mut skipped = Vec::new();
    let mut archives = Vec::new();
    let mut exams = Vec::new();
    let mut walk_failed = false;
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
            &mut walk_failed,
            keep,
        )?;
    }
    discover::add_archives(&mut entries, &mut skipped, archives);
    exams.sort_by(|left, right| left.folder_name.cmp(&right.folder_name));
    Ok(Found {
        entries,
        skipped,
        exams,
        walk_failed,
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
    let mut candidates = vec![format!("{base}/{filename}")];
    if discover::valid_session_id(session_id) {
        candidates.push(format!("{base}/{session_id}/{filename}"));
    }
    for path in candidates {
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
        if discover::valid_session_id(session_id) {
            candidates.insert(1, format!("{base}/{session_id}/termination_summary.txt"));
            // Still-capturing sessions have no summary and their inner dir's
            // mtime freezes once the layer/run dirs exist — input_map.csv
            // moves on every row write (same probes as remote_stamp).
            candidates.insert(2, format!("{base}/input_map.csv"));
            candidates.insert(3, format!("{base}/{session_id}/input_map.csv"));
            candidates.insert(4, format!("{base}/{session_id}"));
        }
    }
    let mut dir_stat = None;
    for path in candidates {
        // A miss on one candidate must not end the search — the nested
        // session layout stores the marker under {base}/{id}/{id}/.
        let Ok(Some(stat)) = fs.stat(&path) else {
            continue;
        };
        if archive_name(&remote.path) || !stat.dir {
            return Some((stat.mtime_ns, i64::try_from(stat.size).ok()?));
        }
        // First dir hit wins: the inner session dir tracks a live capture's
        // growth; the container's mtime froze when the session dir was made.
        if dir_stat.is_none() {
            dir_stat = Some(stat);
        }
    }
    // A still-capturing session has no summary yet — its own directory's
    // metadata still distinguishes it from nothing.
    let stat = dir_stat?;
    Some((stat.mtime_ns, i64::try_from(stat.size).ok()?))
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
        // Catalog keys are display keys — a disambiguated "name · parent" row
        // resolves through ITS library's place no matter which open library
        // the caller passes; the `any` check only requires an open library.
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

/// Drop every recorded session binding for a forgotten library — otherwise a
/// "forgotten" root still resolves and re-downloads remote sessions.
pub(crate) fn drop_sessions(library: &str) {
    sessions()
        .lock()
        .unwrap_or_else(|err| err.into_inner())
        .retain(|(root, _), _| root != library);
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
    // ponytail: one global lock serializes every remote copy — copies are rare
    // and a per-dest keyed guard is not worth the bookkeeping. Without it two
    // copies of one session can interleave: the first writes the stamp, the
    // second overwrites files under a matching stamp, poisoning the cache.
    static COPY: OnceLock<Mutex<()>> = OnceLock::new();
    let _copy = COPY
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    let session = require_remote(session_url)?;
    let relative = relative_remote(library_canonical, &session);
    // Only plain relative segments may join the cache — a `..` escapes it, and
    // on Windows `join("C:/evil")` discards the base entirely. A name that can
    // never exist locally (illegal char, DOS stem) fails the copy every time.
    if !Path::new(&relative).components().all(|part| {
        matches!(part, std::path::Component::Normal(name)
            if !crate::remote::unsafe_local_name(&name.to_string_lossy()))
    }) {
        return Err("remote session path escapes the cache directory".into());
    }
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
    // Files the remote tree lost must not survive in the cache — replace the
    // copy, never merge over it. A failed clear must error out here: merging
    // over a surviving tree and then stamping it would pin stale files in
    // the cache permanently.
    if let Err(err) = fs::remove_dir_all(&dest) {
        // A first materialize has nothing to clear; a removal that left any
        // of the tree behind must not be stamped over.
        if dest.exists() {
            return Err(err.to_string());
        }
    }
    fs.copy_to(&session.path, &dest)?;
    if let Some(parent) = stamp_path.parent() {
        fs::create_dir_all(parent).map_err(|err| err.to_string())?;
    }
    fs::write(&stamp_path, &stamp_text).map_err(|err| err.to_string())?;
    Ok(dest)
}

fn remote_stamp(session: &Remote) -> Result<String, String> {
    let mut fs = crate::remote::connect(session, &password_of(session))?;
    // The directory's mtime only moves on entry add/remove — a file rewritten
    // in place needs the marker's stat or the cache never refreshes. A nested
    // session keeps its markers under {base}/{id}/; probe both layouts, inner
    // dir before container — mirroring remote_fingerprint's candidates.
    let base = session.path.trim_end_matches('/');
    let leaf = base.rsplit('/').next().unwrap_or(base);
    let mut dir_stat = None;
    // Flat markers first, mirroring remote_fingerprint and the local
    // storage_fingerprint — disagreeing orders would record one stat at
    // fingerprint time and another at materialize time, a perpetual re-copy.
    for marker in [
        format!("{base}/termination_summary.txt"),
        format!("{base}/{leaf}/termination_summary.txt"),
        format!("{base}/input_map.csv"),
        format!("{base}/{leaf}/input_map.csv"),
        format!("{base}/{leaf}"),
        base.to_owned(),
    ] {
        let Some(stat) = fs.stat(&marker)? else {
            continue;
        };
        if !stat.dir {
            return Ok(format!("{}\n{}", stat.mtime_ns, stat.size));
        }
        if dir_stat.is_none() {
            dir_stat = Some(stat);
        }
    }
    let stat = dir_stat.unwrap_or(crate::remote::Stat {
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
    // `strip_prefix` is a string match: `/a/bcd` starts with `/a/b` but is not
    // inside it. Only a segment boundary counts as the library prefix.
    let child = path
        .strip_prefix(lib)
        .filter(|rest| rest.is_empty() || rest.starts_with('/'));
    let rel = child.unwrap_or(path).trim_start_matches('/');
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
    if !discover::valid_session_id(session_id) {
        return container.to_path_buf();
    }
    let inner = container.join(session_id);
    if discover::unlinked_dir(&inner) && discover::marker_file(&inner.join("input_map.csv")) {
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
        walk_failed: false,
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
    walk_failed: &mut bool,
    keep: &mut dyn FnMut(u64, u64) -> bool,
) -> Result<(), String> {
    if !keep(0, 0) {
        return Err("cancelled".into());
    }
    if depth >= NEST_DEPTH {
        return Ok(());
    }
    let mut dirs = Vec::new();
    let mut files = Vec::new();
    // The library root itself must list; an unreadable child matches the local
    // walk — skip it instead of failing the whole discovery.
    let items = match fs.list(path) {
        Ok(items) => items,
        Err(err) if depth == 0 => return Err(err),
        Err(_) => {
            *walk_failed = true;
            skipped.push(rel.to_owned());
            return Ok(());
        }
    };
    for item in items {
        // Same guard as copy_sftp_dir: a hostile or buggy server can return
        // a name holding separators — splicing one in walks and later
        // deletes outside the configured root.
        if item.name.is_empty()
            || item.name == "."
            || item.name == ".."
            || item.name.contains(['/', '\\'])
        {
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
        match is_remote_session(fs, &child, name) {
            Ok(true) => {
                let url = join_rel(library_url, &encode_path(&child_rel));
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
            Ok(false) => {}
            Err(_) => {
                *walk_failed = true;
                skipped.push(child_rel.clone());
            }
        }
    }
    for name in &files {
        let Some((session_id, kind)) = discover::archive_kind(name) else {
            continue;
        };
        let child_rel = join_rel(rel, name);
        let url = join_rel(library_url, &encode_path(&child_rel));
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
        let url = join_rel(library_url, &encode_path(&child_rel));
        match remote_exam(fs, &child, name, &url) {
            Ok(Some(exam)) => {
                discover::add_exam(exams, skipped, exam, &child_rel);
                continue;
            }
            Ok(None) => {}
            Err(_) => {
                *walk_failed = true;
                skipped.push(child_rel);
                continue;
            }
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
            walk_failed,
            keep,
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
    let mut dirs = Vec::new();
    let mut count = 0i64;
    for item in fs.list(path)? {
        if item.name.is_empty()
            || item.name == "."
            || item.name == ".."
            || item.name.contains(['/', '\\'])
        {
            continue;
        }
        if item.dir {
            dirs.push(item.name);
        } else {
            count += 1;
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
    // A stray .dcm beside session folders does not make this an exam — the
    // walk must keep descending or the nested sessions disappear.
    for name in &dirs {
        let child = join_path(path, name);
        if remote_holds_session(fs, &child, 0)? {
            return Ok(None);
        }
    }
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

/// Mirror of discover::holds_session — a session anywhere below `path`
/// within the discovery depth makes the folder a container, not an exam.
fn remote_holds_session(
    fs: &mut dyn crate::remote::Fs,
    path: &str,
    depth: u32,
) -> Result<bool, String> {
    if depth >= NEST_DEPTH {
        return Ok(false);
    }
    if is_file(fs, &join_path(path, "input_map.csv"))? {
        return Ok(true);
    }
    for item in fs.list(path)? {
        if item.dir
            && !item.name.is_empty()
            && item.name != "."
            && item.name != ".."
            && !item.name.contains(['/', '\\'])
        {
            let child = join_path(path, &item.name);
            if remote_holds_session(fs, &child, depth + 1)? {
                return Ok(true);
            }
        }
    }
    Ok(false)
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

pub(crate) fn require_remote(spec: &str) -> Result<Remote, String> {
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
    // Split on the raw ':' first — decoding first lets an encoded ':' in
    // the username slide the password boundary.
    let (username, password) = match userinfo {
        Some(info) => match info.split_once(':') {
            Some((user, password)) => (percent_decode(user), Some(percent_decode(password))),
            None => (percent_decode(info), None),
        },
        None => (String::new(), None),
    };
    let (host, port) = split_host_port(hostport)?;
    // An explicit default port is the same server — normalize so the
    // canonical and host keys match the unadorned spelling.
    let port = port.filter(|port| Some(*port) != default_port(scheme));
    if host.is_empty() {
        return Err("the URL needs a host".into());
    }
    let host = host.to_ascii_lowercase();
    let mut path = percent_decode(raw_path);
    // Decoded control chars would land verbatim in the canonical key, which
    // is persisted and logged — CR/LF forges fake log lines.
    if path.bytes().any(|byte| byte.is_ascii_control())
        || host.bytes().any(|byte| byte.is_ascii_control())
        || username.bytes().any(|byte| byte.is_ascii_control())
        || password
            .as_deref()
            .is_some_and(|pass| pass.bytes().any(|byte| byte.is_ascii_control()))
        || query.bytes().any(|byte| byte.is_ascii_control())
    {
        return Err("the URL holds control characters".into());
    }
    if !path.starts_with('/') {
        path.insert(0, '/');
    }
    if path.len() > 1 {
        path = path.trim_end_matches('/').to_owned();
    }
    let mut remote = Remote {
        canonical: String::new(),
        query: query.to_owned(),
        scheme,
        username,
        password,
        host,
        port,
        path,
    };
    remote.canonical = canonical_url(&remote);
    if remote.query.is_empty() {
        // A persisted spec lost the query (it never carries one) — recover
        // the wire token the library root stored for this path or an
        // ancestor.
        remote.query = query_of(&remote);
    } else {
        store_query(&remote);
    }
    Ok(remote)
}

fn canonical_url(remote: &Remote) -> String {
    let mut url = format!(
        "{}://{}{}{}",
        scheme_name(remote.scheme),
        user_at(&remote.username),
        host_text(&remote.host),
        port_text(remote.port)
    );
    url.push_str(&encode_path(&remote.path));
    url
}

/// The URL requests go to: canonical plus the wire-only query.
pub(crate) fn request_url(remote: &Remote) -> String {
    if remote.query.is_empty() {
        remote.canonical.clone()
    } else {
        format!("{}?{}", remote.canonical, remote.query)
    }
}

fn default_port(scheme: Scheme) -> Option<u16> {
    Some(match scheme {
        Scheme::Sftp => 22,
        Scheme::Ftp => 21,
        // The client only does explicit FTPS (AUTH TLS on the cleartext
        // control channel) — implicit-mode :990 must stay an explicit port.
        Scheme::Ftps => 21,
        Scheme::Smb => 445,
        Scheme::Http => 80,
        Scheme::Https => 443,
    })
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
        if !host.contains(':') {
            // One colon means host:port — a non-numeric or empty tail is a
            // mistyped port, not an IPv6 literal (those carry many colons).
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

pub fn file_uri_path(rest: &str) -> PathBuf {
    let rest = percent_decode(rest);
    if let Some(path) = rest.strip_prefix('/') {
        if path.len() >= 2 && path.as_bytes().get(1) == Some(&b':') {
            return PathBuf::from(path);
        }
        return PathBuf::from(format!("/{path}"));
    }
    match rest.split_once('/') {
        // file://localhost/... names this machine (RFC 8089), not a share.
        Some((host, path))
            if host.eq_ignore_ascii_case("localhost") || host == "127.0.0.1" || host == "[::1]" =>
        {
            file_uri_path(&format!("/{path}"))
        }
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

pub(crate) fn home_dir() -> PathBuf {
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

// The decoded path carries real `?`/`#`/`%` bytes that would reparse into a
// different URL — the canonical form must be a faithful encoding.
fn encode_path(text: &str) -> String {
    let mut out = String::new();
    for byte in text.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'/' | b'-' | b'_' | b'.' | b'~' => {
                out.push(byte as char)
            }
            _ => out.push_str(&format!("%{byte:02X}")),
        }
    }
    out
}

fn encode_user(text: &str) -> String {
    let mut out = String::new();
    for byte in text.bytes() {
        match byte {
            // `byte as char` on a UTF-8 byte above 0x7F emits mojibake — encode
            // non-ASCII so the reparsed username is byte-identical.
            b'@' | b':' | b'/' | b'%' | b'?' | b'#' | b' ' | 0x80.. => {
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

fn queries() -> &'static Mutex<HashMap<String, String>> {
    static QUERIES: OnceLock<Mutex<HashMap<String, String>>> = OnceLock::new();
    QUERIES.get_or_init(|| Mutex::new(HashMap::new()))
}

/// A session spec under a queried root inherits the root's wire query,
/// the same way `password_of` inherits the host credential.
fn query_of(remote: &Remote) -> String {
    let map = queries().lock().unwrap_or_else(|err| err.into_inner());
    if let Some(query) = map.get(&remote.canonical) {
        return query.clone();
    }
    // Longest ancestor wins — HashMap order is arbitrary, and a nested
    // stored root must not lose to a shallower one.
    map.iter()
        .filter(|(key, _)| under_root(&remote.canonical, key))
        .max_by_key(|(key, _)| key.len())
        .map(|(_, query)| query.clone())
        .unwrap_or_default()
}

fn store_query(remote: &Remote) {
    queries()
        .lock()
        .unwrap_or_else(|err| err.into_inner())
        .insert(remote.canonical.clone(), remote.query.clone());
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

/// `canonical` sits under `key`: a root-path key already ends in '/', so
/// `key/` matching would look for a doubled slash that never appears.
pub(crate) fn under_root(canonical: &str, key: &str) -> bool {
    if key.ends_with('/') {
        canonical.starts_with(key)
    } else {
        canonical.starts_with(&format!("{key}/"))
    }
}

pub(crate) fn password_of(remote: &Remote) -> String {
    let map = passwords().lock().unwrap_or_else(|err| err.into_inner());
    if let Some(password) = map.get(&remote.canonical) {
        return password.clone();
    }
    // Session URLs inherit the longest matching library prefix — the host
    // slot alone would hand sibling libraries each other's credentials.
    map.iter()
        .filter(|(key, _)| key.as_str() != host_key(remote).as_str())
        .filter(|(key, _)| under_root(&remote.canonical, key))
        .max_by_key(|(key, _)| key.len())
        .map(|(_, password)| password.clone())
        .or_else(|| map.get(&host_key(remote)).cloned())
        .unwrap_or_default()
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
    if code == 0 {
        return Ok(());
    }
    if code == 1219 {
        return Err(auth_error(
            "a Windows session to that server already exists under different credentials",
        ));
    }
    if matches!(code, 5 | 86 | 1323 | 1326 | 2202) {
        return Err(auth_error("the share refused the password"));
    }
    Err(format!("could not connect to {share} (error {code})"))
}

pub(crate) fn connect_timeout() -> Duration {
    CONNECT_TIMEOUT
}

/// Folder and leftover archives for `session_id` under the opened library
/// `root` — remote specs for a remote root, local paths otherwise.
pub(crate) fn session_storage_paths(root: &str, session_id: &str) -> Result<Vec<String>, String> {
    if !discover::valid_session_id(session_id) {
        return Err(format!("invalid session id {session_id:?}"));
    }
    let mut found = Vec::new();
    // recorded() is in-memory state from the last scan — a session moved
    // externally since then leaves a dead spec whose delete error would
    // keep the row. Local paths are verified here; remote specs are
    // verified by the stat probes below.
    let mut recorded_parent = None;
    if let Some(place) = recorded(root, session_id) {
        if !is_remote_location(&place) {
            recorded_parent = Path::new(&place).parent().map(|dir| dir.to_path_buf());
            if Path::new(&place).exists() {
                found.push(place);
            }
        }
    }
    let mut names = vec![session_id.to_string()];
    for (suffix, _) in discover::ARCHIVE_SUFFIXES {
        names.push(format!("{session_id}{suffix}"));
    }
    let root = match require_remote(root).ok() {
        // smb opens as a UNC path — remote::connect errors on it, so storage
        // paths and deletes resolve locally instead.
        Some(remote) if remote.scheme == Scheme::Smb => {
            let unc = connect_smb(&remote)?;
            // A recorded smb spec maps onto the UNC tree the same way the
            // delete path strips it — verify it and probe its parent for
            // sibling archives like other remotes do.
            if let Some(place) = recorded(root, session_id) {
                if let Ok(child) = require_remote(&place) {
                    let child_path = child.path.trim_end_matches('/');
                    let root_path = remote.path.trim_end_matches('/');
                    let rel = child_path
                        .strip_prefix(root_path)
                        .filter(|rest| rest.is_empty() || rest.starts_with('/'));
                    if let Some(rel) = rel {
                        let local = unc.join(rel.trim_start_matches('/'));
                        recorded_parent = local.parent().map(|dir| dir.to_path_buf());
                        if local.exists() {
                            found.push(local.to_string_lossy().into_owned());
                        }
                    }
                }
            }
            unc.to_string_lossy().into_owned()
        }
        Some(remote) => {
            let mut fs = crate::remote::connect(&remote, &password_of(&remote))?;
            // Verify a recorded remote spec too — same dead-spec hazard.
            if let Some(place) = recorded(root, session_id) {
                if let Some(child) = require_remote(&place).ok().and_then(|child| {
                    fs.stat(child.path.trim_end_matches('/'))
                        .ok()
                        .flatten()
                        .map(|_| place.clone())
                }) {
                    found.push(child);
                }
            }
            // Probe the library root and the recorded spec's parent —
            // a nested session's archive sibling lives next to it. The
            // parent must stay inside the root: a recorded spec pointing
            // elsewhere is stale state, not a delete candidate.
            let root_path = remote.path.trim_end_matches('/');
            let mut bases = vec![root_path.to_string()];
            if let Some(place) = recorded(root, session_id) {
                if let Ok(child) = require_remote(&place) {
                    let parent_path = child.path.trim_end_matches('/');
                    if let Some((parent, _)) = parent_path.rsplit_once('/') {
                        let inside = parent == root_path
                            || parent
                                .strip_prefix(root_path)
                                .is_some_and(|rest| rest.starts_with('/'));
                        if inside && !bases.contains(&parent.to_string()) {
                            bases.push(parent.to_string());
                        }
                    }
                }
            }
            for base in &bases {
                for name in &names {
                    if fs.stat(&format!("{base}/{name}"))?.is_some() {
                        let rel = match base.strip_prefix(root_path) {
                            Some(rest) if rest.is_empty() || rest.starts_with('/') => {
                                format!("{rest}/{name}")
                            }
                            _ => continue,
                        };
                        let place = format!(
                            "{}/{}",
                            remote.canonical.trim_end_matches('/'),
                            encode_path(rel.trim_start_matches('/'))
                        );
                        if !found.contains(&place) {
                            found.push(place);
                        }
                    }
                }
            }
            return Ok(found);
        }
        None => root.to_owned(),
    };
    let mut bases = vec![Path::new(&root).to_path_buf()];
    if let Some(parent) = recorded_parent {
        if parent != bases[0] && parent.starts_with(&bases[0]) {
            bases.push(parent);
        }
    }
    for base in &bases {
        for name in &names {
            let path = base.join(name);
            let hit = if *name == session_id {
                // A stray file carrying the id is not the session folder.
                path.is_dir()
            } else if name.ends_with(".zip") {
                // A readable zip's identity is the folder it wraps — one
                // wrapping another session is not this session's storage,
                // and trashing must not destroy it. An unreadable file
                // carries no wrapped id to contradict its name.
                path.is_file()
                    && match discover::archive_identity(&path) {
                        Some((id, _)) => id == session_id,
                        None => true,
                    }
            } else {
                // Tars cannot peek cheaply — their filename stem IS the id.
                path.is_file()
            };
            let place = path.to_string_lossy().into_owned();
            if hit && !found.contains(&place) {
                found.push(place);
            }
        }
        // A zip's session id comes from the folder wrapped inside it, not its
        // filename — `a.zip` holding `b/` re-discovers as `b` after `b` is
        // trashed, so probe archives by identity too. Tars cannot peek
        // cheaply; their identity is the filename stem already covered above.
        if let Ok(read) = fs::read_dir(base) {
            for entry in read.flatten() {
                let path = entry.path();
                if !path.is_file() {
                    continue;
                }
                let matches =
                    discover::archive_identity(&path).is_some_and(|(id, _)| id == session_id);
                let place = path.to_string_lossy().into_owned();
                if matches && !found.contains(&place) {
                    found.push(place);
                }
            }
        }
    }
    Ok(found)
}

/// Remove `paths` — OS trash locally, permanent recursive delete on remote
/// hosts. Returns per-path errors; empty means everything was removed.
pub(crate) fn delete_storage(root: &str, paths: &[String]) -> Result<Vec<String>, String> {
    // Deepest first — a parent delete removes its children, whose own
    // delete would then error "not found" and keep the row.
    let mut paths: Vec<String> = paths.to_vec();
    paths.sort_by_key(|spec| {
        std::cmp::Reverse(spec.chars().filter(|c| *c == '/' || *c == '\\').count())
    });
    let mut errors = Vec::new();
    if is_remote_location(root) {
        let remote = require_remote(root)?;
        if remote.scheme == Scheme::Smb {
            // smb opens as a UNC path — remote::connect errors on it.
            let unc = connect_smb(&remote)?;
            for spec in &paths {
                // The remainder must start at a segment boundary —
                // `\filer\shareX\y` must not strip `\filer\share`.
                let unc_str = unc.to_string_lossy();
                let relative = spec
                    .strip_prefix(unc_str.as_ref())
                    .and_then(|rel| rel.strip_prefix('\\').or_else(|| rel.strip_prefix('/')))
                    .unwrap_or_default();
                // Same escape guard as materialize: only plain segments join.
                let safe = !relative.is_empty()
                    && Path::new(&relative)
                        .components()
                        .all(|part| matches!(part, std::path::Component::Normal(_)));
                if !safe {
                    errors.push(format!("{spec}: not under the share root"));
                    continue;
                }
                let local = unc.join(relative);
                if let Err(err) = trash_path(&local.to_string_lossy()) {
                    errors.push(format!("{spec}: {err}"));
                }
            }
            return Ok(errors);
        }
        let mut fs = crate::remote::connect(&remote, &password_of(&remote))?;
        for spec in &paths {
            match require_remote(spec) {
                Ok(child) => {
                    // Only remove inside the connected root — a stale or
                    // corrupt recorded spec must not delete elsewhere on
                    // the server.
                    let root_path = remote.path.trim_end_matches('/');
                    if !root_path.is_empty()
                        && !(child.path == root_path
                            || child.path.starts_with(&format!("{root_path}/")))
                    {
                        errors.push(format!("{spec}: outside the connected root"));
                        continue;
                    }
                    if let Err(err) = fs.remove(&child.path) {
                        errors.push(format!("{spec}: {err}"));
                    } else {
                        // Materialized copies outlive the remote session —
                        // drop the local cache subtree with it. Same escape
                        // guard as materialize: only plain segments join.
                        let relative = relative_remote(root, &child);
                        let safe = !relative.is_empty()
                            && Path::new(&relative)
                                .components()
                                .all(|part| matches!(part, std::path::Component::Normal(_)));
                        if safe {
                            let cached = cache_directory(root).join(&relative);
                            let _ = std::fs::remove_dir_all(&cached);
                            let _ = std::fs::remove_file(stamp_file(&cached));
                        }
                    }
                }
                Err(err) => errors.push(format!("{spec}: {err}")),
            }
        }
        return Ok(errors);
    }
    for path in &paths {
        if let Err(err) = trash_path(path) {
            errors.push(format!("{path}: {err}"));
        }
    }
    Ok(errors)
}

#[cfg(not(target_arch = "wasm32"))]
fn trash_path(path: &str) -> Result<(), String> {
    match trash::delete(path) {
        Ok(()) => Ok(()),
        // There is no recycle bin for network shares — a UNC delete falls back
        // to a permanent removal, the same as the remote drivers perform.
        Err(err) if path.starts_with("\\\\") => {
            let target = Path::new(path);
            let meta = fs::symlink_metadata(target).map_err(|_| err.to_string())?;
            let result = if meta.is_dir() && !meta.file_type().is_symlink() {
                fs::remove_dir_all(target)
            } else {
                fs::remove_file(target)
            };
            result.map_err(|e| format!("{err}; permanent delete also failed: {e}"))
        }
        Err(err) => Err(err.to_string()),
    }
}

#[cfg(target_arch = "wasm32")]
fn trash_path(_path: &str) -> Result<(), String> {
    Err("the system trash is not available in this build".into())
}

#[cfg(test)]
mod tests {

    #[test]
    fn an_explicit_ftps_990_port_is_kept() {
        // FTPS upgrades to TLS on 21 — a bare :990 must survive parsing instead
        // of being stripped as the "default" and then connected on 21.
        let remote = super::parse_remote(super::Scheme::Ftps, "archive.example:990/dose")
            .expect("ftps with an explicit port parses");
        assert_eq!(remote.port, Some(990));
    }

    #[test]
    fn root_keys_match_as_prefixes() {
        assert!(super::under_root("sftp://h/root/dir", "sftp://h/root/"));
        assert!(super::under_root("sftp://h/root/", "sftp://h/root/"));
        assert!(!super::under_root("sftp://h/rootless", "sftp://h/root/"));
        assert!(super::under_root("a/b", "a/"));
        assert!(!super::under_root("ab/c", "a/"));
    }

    #[test]
    fn a_localhost_file_uri_is_not_a_share() {
        let local = file_uri_path("localhost/C:/data");
        assert_eq!(local, std::path::PathBuf::from("C:/data"));
        let remote = file_uri_path("nas/share/dose");
        assert!(remote.to_string_lossy().starts_with("\\"), "{remote:?}");
    }

    use super::*;

    #[test]
    fn storage_paths_finds_a_nested_recorded_session_and_sibling_archive() {
        // A session recorded under a subdirectory keeps its archive next to
        // it — probing only the library root misses both.
        let root = std::env::temp_dir().join(format!(
            "scan-kit-nested-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let nested = root.join("group").join("sess");
        fs::create_dir_all(&nested).unwrap();
        fs::write(root.join("group").join("sess.zip"), b"zip").unwrap();
        let root_str = root.to_string_lossy().into_owned();
        sessions().lock().unwrap().insert(
            (root_str.clone(), "sess".to_string()),
            nested.to_string_lossy().into_owned(),
        );
        let paths = session_storage_paths(&root_str, "sess").unwrap();
        sessions()
            .lock()
            .unwrap()
            .remove(&(root_str.clone(), "sess".to_string()));
        assert_eq!(
            paths.len(),
            2,
            "expected the dir and its sibling archive: {paths:?}"
        );
        assert!(paths.iter().any(|p| p.ends_with("sess")));
        assert!(paths.iter().any(|p| p.ends_with("sess.zip")));

        // A stale recorded spec points nowhere — it is ignored, not deleted.
        sessions().lock().unwrap().insert(
            (root_str.clone(), "gone".to_string()),
            root.join("missing").to_string_lossy().into_owned(),
        );
        let paths = session_storage_paths(&root_str, "gone").unwrap();
        sessions()
            .lock()
            .unwrap()
            .remove(&(root_str, "gone".to_string()));
        assert!(paths.is_empty());
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn storage_paths_finds_a_zip_whose_wrapped_id_differs_from_its_name() {
        // a.zip wrapping b/ re-discovers as session b after b is trashed —
        // trash must probe archives by inside identity too.
        let root = std::env::temp_dir().join(format!("scan-kit-zipid-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        discover::write_zip(&root.join("a.zip"), &[("b/input_map.csv", b"energy\n70\n")]).unwrap();
        let paths = session_storage_paths(&root.to_string_lossy(), "b").unwrap();
        assert!(
            paths.iter().any(|p| p.ends_with("a.zip")),
            "expected a.zip in {paths:?}"
        );
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_name_matched_zip_wrapping_another_session_is_not_trashed() {
        // sess.zip wrapping other/ belongs to session "other" — trashing
        // "sess" must leave it alone.
        let root = std::env::temp_dir().join(format!(
            "scan-kit-foreignzip-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(root.join("sess")).unwrap();
        fs::write(
            root.join("sess").join("input_map.csv"),
            b"energy
70
",
        )
        .unwrap();
        discover::write_zip(
            &root.join("sess.zip"),
            &[(
                "other/input_map.csv",
                b"energy
70
",
            )],
        )
        .unwrap();
        let root_str = root.to_string_lossy().into_owned();
        let paths = session_storage_paths(&root_str, "sess").unwrap();
        assert!(
            paths.iter().all(|p| !p.ends_with("sess.zip")),
            "foreign zip trashed: {paths:?}"
        );
        assert!(paths.iter().any(|p| p.ends_with("sess")));
        let _ = fs::remove_dir_all(&root);
    }

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

        let remote = match classify("sftp://pyramid:secret@192.168.101.206/data").unwrap() {
            Kind::Remote(remote) => remote,
            Kind::Local(_) => panic!("sftp is remote"),
        };
        let debug = format!("{remote:?}");
        assert!(!debug.contains("secret"), "{debug}");
        assert!(!is_remote_location(r"\\192.168.101.206\share\ptc_ex"));
        assert!(!is_remote_location(r"C:\data"));
        assert!(!is_remote_location("C://data"));
        assert!(!is_remote_location("file:///C:/data"));
        assert!(canonical_root("nfs://host/data").is_err());
        for bad in [
            "sftp://user@host/data%0aforged",
            "sftp://user@host/data%0d",
            "sftp://u%09ser@host/data",
            "sftp://user@host/data?\tforged",
        ] {
            assert!(canonical_root(bad).is_err(), "{bad}");
        }
        assert_eq!(
            Path::new("sftp://pyramid@192.168.101.206/var/log/ptc_ex").to_string_lossy(),
            "sftp://pyramid@192.168.101.206/var/log/ptc_ex"
        );
    }

    #[test]
    fn the_canonical_url_reparses_to_the_same_path_and_user() {
        // A decoded `?`/`%`/`#` in the path used to reparse as structure —
        // the canonical must round-trip byte-identically.
        let canonical = canonical_root("sftp://host.invalid/data%3Fsub/x%25y").unwrap();
        let remote = match classify(&canonical).unwrap() {
            Kind::Remote(remote) => remote,
            Kind::Local(_) => panic!("sftp is remote"),
        };
        assert_eq!(remote.path, "/data?sub/x%y");
        assert_eq!(remote.canonical, canonical);

        let canonical = canonical_root("sftp://jos%C3%A9@host.invalid/lib").unwrap();
        let remote = match classify(&canonical).unwrap() {
            Kind::Remote(remote) => remote,
            Kind::Local(_) => panic!("sftp is remote"),
        };
        assert_eq!(remote.username, "josé");
        assert_eq!(remote.canonical, canonical);
    }

    #[test]
    fn a_session_cache_path_stays_inside_the_library() {
        let library = require_remote("sftp://u@h/a/b").unwrap();
        let child = require_remote("sftp://u@h/a/b/c/d").unwrap();
        assert_eq!(relative_remote(&library.canonical, &child), "c/d");
        // `/a/bcd` shares the `/a/b` string prefix without being inside it.
        let sibling = require_remote("sftp://u@h/a/bcd/x").unwrap();
        assert_eq!(relative_remote(&library.canonical, &sibling), "a/bcd/x");
        let climbing = require_remote("sftp://u@h/a/b/../evil").unwrap();
        let relative = relative_remote(&library.canonical, &climbing);
        assert!(
            relative.split(['/', '\\']).any(|part| part == ".."),
            "{relative}"
        );
    }
    #[test]
    fn forgetting_a_library_drops_its_recorded_sessions() {
        replace_sessions("a-test-lib", &[("sess".to_owned(), "place".to_owned())]);
        assert!(recorded("a-test-lib", "sess").is_some());
        drop_sessions("a-test-lib");
        assert!(recorded("a-test-lib", "sess").is_none());
    }

    #[test]
    fn catalog_keys_resolve_through_their_own_library() {
        let root = std::env::temp_dir().join(format!(
            "scan-kit-cat-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&root);
        let lib_a = root.join("a");
        let lib_b = root.join("b");
        std::fs::create_dir_all(lib_a.join("s")).unwrap();
        std::fs::create_dir_all(lib_b.join("s")).unwrap();
        replace_catalog(&[
            (
                "s".to_owned(),
                library_key(&lib_a),
                lib_a.join("s").to_string_lossy().into_owned(),
                "s".to_owned(),
            ),
            (
                "s · p".to_owned(),
                library_key(&lib_b),
                lib_b.join("s").to_string_lossy().into_owned(),
                "s".to_owned(),
            ),
        ]);
        // Display keys are global: "s" names a's row and "s · p" names b's —
        // either open library resolves both.
        assert_eq!(catalog_directory(&lib_b, "s"), Some(lib_a.join("s")));
        assert_eq!(catalog_directory(&lib_b, "s · p"), Some(lib_b.join("s")));
        replace_catalog(&[]);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_url_query_stays_off_the_canonical_but_reaches_the_wire() {
        // A presigned-URL signature is a live credential — it must not land
        // in the persisted canonical, prefs, or session storage paths.
        let canonical = canonical_root("https://u:p@files.example/dose.zip?sig=abc&exp=9").unwrap();
        assert_eq!(canonical, "https://u@files.example/dose.zip");
        assert!(!canonical.contains("sig"));
        // The wire form reattaches the query for the same spec — and for a
        // session URL parked under the root — through the in-memory map.
        let remote = match classify(&canonical).unwrap() {
            Kind::Remote(remote) => remote,
            Kind::Local(_) => panic!("https is remote"),
        };
        assert_eq!(
            request_url(&remote),
            "https://u@files.example/dose.zip?sig=abc&exp=9"
        );
        let child = match classify(&format!("{canonical}/nested")).unwrap() {
            Kind::Remote(remote) => remote,
            Kind::Local(_) => panic!("https is remote"),
        };
        assert_eq!(
            request_url(&child),
            "https://u@files.example/dose.zip/nested?sig=abc&exp=9"
        );
    }

    #[test]
    fn a_default_port_is_normalized_out_of_the_canonical() {
        assert_eq!(
            canonical_root("sftp://u@h.example:22/data").unwrap(),
            "sftp://u@h.example/data"
        );
        assert_eq!(
            canonical_root("https://files.example:443/dose.zip").unwrap(),
            "https://files.example/dose.zip"
        );
        // A non-default port stays.
        assert_eq!(
            canonical_root("sftp://u@h.example:2222/data").unwrap(),
            "sftp://u@h.example:2222/data"
        );
    }

    #[test]
    fn an_encoded_colon_stays_inside_the_username() {
        let remote = match classify("sftp://u%3Ax:pw@h.example/data").unwrap() {
            Kind::Remote(remote) => remote,
            Kind::Local(_) => panic!("sftp is remote"),
        };
        assert_eq!(remote.username, "u:x");
        assert_eq!(remote.password.as_deref(), Some("pw"));
    }

    #[test]
    fn a_non_numeric_port_is_an_error_not_an_ipv6_host() {
        assert_eq!(
            split_host_port("host:2222").unwrap(),
            ("host".to_owned(), Some(2222))
        );
        assert!(split_host_port("host:abc").is_err());
        assert!(split_host_port("host:").is_err());
        // A multi-colon authority is unbracketed IPv6 — not a port error.
        assert_eq!(
            split_host_port("fe80::1").unwrap(),
            ("fe80::1".to_owned(), None)
        );
    }

    #[test]
    fn a_nested_remote_session_fingerprints_the_inner_marker() {
        // Nested layout: the session's files live under {base}/sess/. The
        // container dir stat (size 0) must not overwrite the marker's.
        let url = format!(
            "sftp://u@host.invalid/nest/{}/{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let canonical = install_fixture(
            &format!("{url}/sess"),
            &[
                ("sess/input_map.csv", b"energy\n70\n"),
                ("sess/termination_summary.txt", b"Configuration name: x\n"),
            ],
            false,
        )
        .unwrap();
        let (mtime, size) = remote_fingerprint(&canonical, "sess").unwrap();
        assert_eq!(mtime, 1_000_000_000);
        assert_eq!(size, "Configuration name: x\n".len() as i64);

        let remote = require_remote(&canonical).unwrap();
        let stamp = remote_stamp(&remote).unwrap();
        assert_eq!(
            stamp,
            format!("1000000000\n{}", "Configuration name: x\n".len())
        );
    }

    #[test]
    fn flat_and_nested_markers_stamp_the_same_one_they_fingerprint() {
        // Both markers present: fingerprint and stamp must pick the same one
        // (flat first, like storage_fingerprint) or every scan re-copies.
        let url = format!(
            "sftp://u@host.invalid/both/{}/{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let canonical = install_fixture(
            &format!("{url}/sess"),
            &[
                ("input_map.csv", b"energy\n70\n"),
                ("termination_summary.txt", b"flat\n"),
                ("sess/input_map.csv", b"energy\n70\n"),
                ("sess/termination_summary.txt", b"nested-marker\n"),
            ],
            false,
        )
        .unwrap();
        let (mtime, size) = remote_fingerprint(&canonical, "sess").unwrap();
        assert_eq!((mtime, size), (1_000_000_000, 5)); // "flat\n"
        let remote = require_remote(&canonical).unwrap();
        assert_eq!(remote_stamp(&remote).unwrap(), "1000000000\n5");
    }

    #[test]
    fn a_password_matches_the_longest_remote_prefix() {
        // Passwords are keyed by remote spec; a session nested under a
        // stored root must pick the deepest matching key, not a sibling.
        let tag = format!("h{}", std::process::id());
        // Stored last, so the host slot holds "root-pw" — a naive host-key
        // lookup would hand it to /a/sess1; the longest prefix must win.
        remember_password(&format!("sftp://u@{tag}/a"), "a-pw");
        remember_password(&format!("sftp://u@{tag}"), "root-pw");
        let nested = require_remote(&format!("sftp://u@{tag}/a/sess1")).unwrap();
        assert_eq!(password_of(&nested), "a-pw");
        let sibling = require_remote(&format!("sftp://u@{tag}/b/sess1")).unwrap();
        assert_eq!(password_of(&sibling), "root-pw");
    }
}
