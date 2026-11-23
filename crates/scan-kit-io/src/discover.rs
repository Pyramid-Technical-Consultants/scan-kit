//! Session discovery. Archives are read in memory and not unpacked.
//!
//! A library walk classifies each directory once. A session or a DICOM exam is
//! not walked further. Anything else is a container, up to eight levels.
//! Symlinks are not followed. A file walk inside one session is separate.
//!
//! ponytail: directory walks inside a session stop at depth 6, which covers
//! `layer-N/run-M`. A session laid out deeper than that needs a longer walk.

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

pub(crate) const ARCHIVE_SUFFIXES: &[(&str, &str)] = &[
    (".tar.gz", "tar"),
    (".tar.bz2", "tar"),
    (".tar.xz", "tar"),
    (".tgz", "tar"),
    (".tar", "tar"),
    (".zip", "zip"),
];

#[derive(Clone, Debug)]
pub struct Discovered {
    pub session_id: String,
    pub storage_path: PathBuf,
    pub kind: &'static str,
}

#[derive(Clone, Debug)]
pub(crate) struct ExamHit {
    pub study_uid: String,
    pub folder_name: String,
    pub storage_path: PathBuf,
    pub patient_name: String,
    pub patient_id: String,
    pub study_date: String,
    pub study_description: String,
    pub file_count: i64,
}

pub(crate) struct Found {
    pub entries: Vec<Discovered>,
    pub skipped: Vec<String>,
    pub exams: Vec<ExamHit>,
    /// Some directories could not be listed — the index must not treat the
    /// missing entries as deleted.
    pub walk_failed: bool,
}

pub(crate) fn discover(
    root: &Path,
    keep: &mut dyn FnMut(u64, u64) -> bool,
) -> Result<Found, String> {
    let text = root.to_string_lossy();
    if crate::location::is_remote_location(&text) {
        return crate::location::discover_remote(&text, keep);
    }
    let mut entries = Vec::new();
    let mut skipped = Vec::new();
    let mut archives = Vec::new();
    let mut exams = Vec::new();
    let mut walk_failed = false;
    if is_unpacked_session(root, &mut walk_failed) {
        if let Some(session_id) = file_name(root) {
            add_directory(
                &mut entries,
                &mut skipped,
                Discovered {
                    session_id,
                    storage_path: root.to_path_buf(),
                    kind: "directory",
                },
                "",
            );
        }
    } else if let Some(exam) = exam_in(root, &mut walk_failed) {
        add_exam(&mut exams, &mut skipped, exam, "");
    } else {
        walk_local(
            root,
            root,
            0,
            &mut entries,
            &mut skipped,
            &mut archives,
            &mut exams,
            &mut walk_failed,
            keep,
        )?;
    }
    add_archives(&mut entries, &mut skipped, archives);
    exams.sort_by(|left, right| left.folder_name.cmp(&right.folder_name));
    Ok(Found {
        entries,
        skipped,
        exams,
        walk_failed,
    })
}

fn walk_local(
    library: &Path,
    dir: &Path,
    depth: u32,
    entries: &mut Vec<Discovered>,
    skipped: &mut Vec<String>,
    archives: &mut Vec<(Discovered, String)>,
    exams: &mut Vec<ExamHit>,
    walk_failed: &mut bool,
    keep: &mut dyn FnMut(u64, u64) -> bool,
) -> Result<(), String> {
    if depth >= crate::location::NEST_DEPTH {
        return Ok(());
    }
    // The remote walk honours keep at every recursion — a slow local share
    // needs the same cancel path. Returning Err matters: a partial Found
    // would look like mass deletion to delete_missing.
    if !keep(0, 0) {
        return Err("cancelled".into());
    }
    let Ok(read) = fs::read_dir(dir) else {
        // An unreadable dir is not an empty one — a partial listing must not
        // look like "every session was deleted" to delete_missing.
        *walk_failed = true;
        skipped.push(relative(library, dir));
        return Ok(());
    };
    // An errored entry or failed file_type() must not look like a deleted
    // session — delete_missing prunes rows for ids absent from the walk.
    let mut children = Vec::new();
    for entry in read {
        match entry {
            Ok(entry) => children.push(entry),
            Err(_) => *walk_failed = true,
        }
    }
    children.sort_by_key(|entry| entry.file_name());
    let mut sessions = Vec::new();
    for child in &children {
        let Ok(kind) = child.file_type() else {
            *walk_failed = true;
            continue;
        };
        if kind.is_symlink() || !kind.is_dir() {
            continue;
        }
        let path = child.path();
        if !is_unpacked_session(&path, walk_failed) {
            continue;
        }
        let Some(session_id) = file_name(&path) else {
            continue;
        };
        let rel = relative(library, &path);
        add_directory(
            entries,
            skipped,
            Discovered {
                session_id,
                storage_path: path,
                kind: "directory",
            },
            &rel,
        );
        sessions.push(child.file_name());
    }
    for child in &children {
        let Ok(kind) = child.file_type() else {
            continue;
        };
        if kind.is_symlink() || !kind.is_file() {
            continue;
        }
        let path = child.path();
        let Some((session_id, kind)) = archive_identity(&path) else {
            continue;
        };
        archives.push((
            Discovered {
                session_id,
                storage_path: path,
                kind,
            },
            relative(library, &child.path()),
        ));
    }
    for child in &children {
        let Ok(kind) = child.file_type() else {
            continue;
        };
        // A leftover download staging dir holds a bare "session/" tree that
        // would catalog as a ghost session — it is not real storage.
        if kind.is_symlink()
            || !kind.is_dir()
            || sessions.iter().any(|name| name == &child.file_name())
            || child
                .file_name()
                .to_string_lossy()
                .starts_with(".session_staging-")
        {
            continue;
        }
        let path = child.path();
        if let Some(exam) = exam_in(&path, walk_failed) {
            add_exam(exams, skipped, exam, &relative(library, &path));
            continue;
        }
        walk_local(
            library,
            &path,
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

/// True when an unpacked session lives anywhere below `dir` within the
/// discovery depth — such a directory is a container, never an exam.
fn holds_session(dir: &Path, depth: u32, walk_failed: &mut bool) -> bool {
    if depth >= crate::location::NEST_DEPTH {
        return false;
    }
    if is_unpacked_session(dir, walk_failed) {
        return true;
    }
    let Ok(read) = fs::read_dir(dir) else {
        // An unlistable dir might hold a session — never an exam candidate.
        *walk_failed = true;
        return true;
    };
    for entry in read {
        let Ok(entry) = entry else {
            *walk_failed = true;
            return true;
        };
        let Ok(kind) = entry.file_type() else {
            *walk_failed = true;
            return true;
        };
        if kind.is_dir()
            && !kind.is_symlink()
            && holds_session(&entry.path(), depth + 1, walk_failed)
        {
            return true;
        }
    }
    false
}

fn exam_in(dir: &Path, walk_failed: &mut bool) -> Option<ExamHit> {
    // A stray .dcm beside session folders does not make this an exam — the
    // walk must keep descending or the nested sessions disappear.
    if holds_session(dir, 0, walk_failed) {
        return None;
    }
    let folder_name = file_name(dir)?;
    let mut files = Vec::new();
    let mut count = 0i64;
    let Ok(read) = fs::read_dir(dir) else {
        *walk_failed = true;
        return None;
    };
    for entry in read {
        let Ok(entry) = entry else {
            *walk_failed = true;
            return None;
        };
        let Ok(kind) = entry.file_type() else {
            *walk_failed = true;
            return None;
        };
        if kind.is_symlink() {
            continue;
        }
        if kind.is_file() {
            count += 1;
            files.push(entry.path());
        }
    }
    files.sort();
    let sample = files
        .into_iter()
        .find(|path| scan_kit_dicom::file_is_dicom(path))?;
    let peek = scan_kit_dicom::peek_file(&sample);
    // An unreadable or non-DICOM sample is not an exam - a folder-name uid
    // fabricates a phantom record, and classifying here stops the walk
    // from ever reaching archives nested under the directory.
    if peek.study_uid.is_empty() {
        return None;
    }
    let study_uid = peek.study_uid;
    Some(ExamHit {
        study_uid,
        folder_name,
        storage_path: dir.to_path_buf(),
        patient_name: peek.patient_name,
        patient_id: peek.patient_id,
        study_date: peek.study_date,
        study_description: peek.study_description,
        file_count: count,
    })
}

pub(crate) fn add_exam(
    exams: &mut Vec<ExamHit>,
    skipped: &mut Vec<String>,
    exam: ExamHit,
    rel: &str,
) {
    if exams.iter().any(|have| have.study_uid == exam.study_uid) {
        skipped.push(format!("{} ({rel})", exam.study_uid));
        return;
    }
    exams.push(exam);
}

pub(crate) fn add_directory(
    entries: &mut Vec<Discovered>,
    skipped: &mut Vec<String>,
    entry: Discovered,
    rel: &str,
) {
    if !valid_session_id(&entry.session_id) {
        skipped.push(format!("{} ({rel})", entry.session_id));
        return;
    }
    if entries
        .iter()
        .any(|have| have.session_id == entry.session_id)
    {
        skipped.push(format!("{} ({rel})", entry.session_id));
        return;
    }
    entries.push(entry);
}

pub(crate) fn add_archives(
    entries: &mut Vec<Discovered>,
    skipped: &mut Vec<String>,
    archives: Vec<(Discovered, String)>,
) {
    for (entry, rel) in archives {
        if entries
            .iter()
            .any(|have| have.session_id == entry.session_id)
        {
            skipped.push(format!("{} ({rel})", entry.session_id));
            continue;
        }
        entries.push(entry);
    }
    entries.sort_by(|left, right| left.session_id.cmp(&right.session_id));
    skipped.sort();
}

fn relative(library: &Path, path: &Path) -> String {
    path.strip_prefix(library)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

pub fn storage_fingerprint(storage: &Path, session_id: &str) -> Option<(i64, i64)> {
    let text = storage.to_string_lossy();
    if crate::location::is_remote_location(&text) {
        return crate::location::remote_fingerprint(&text, session_id);
    }
    let target = if storage.is_dir() {
        let outer = storage.join("termination_summary.txt");
        let inner = if valid_session_id(session_id) {
            storage.join(session_id).join("termination_summary.txt")
        } else {
            storage.join("termination_summary.txt")
        };
        // The inner dir's mtime freezes once the layer/run dirs exist —
        // input_map.csv moves on every row write, so probe it too.
        let outer_rows = storage.join("input_map.csv");
        let inner_rows = if valid_session_id(session_id) {
            storage.join(session_id).join("input_map.csv")
        } else {
            storage.join("input_map.csv")
        };
        if outer.is_file() {
            outer
        } else if inner.is_file() {
            inner
        } else if outer_rows.is_file() {
            outer_rows
        } else if inner_rows.is_file() {
            inner_rows
        } else {
            // Still capturing: the inner session dir changes, the outer does not.
            session_directory(storage, session_id)
        }
    } else {
        storage.to_path_buf()
    };
    let meta = fs::metadata(&target).ok()?;
    let modified = meta.modified().ok()?;
    let nanos = modified
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_nanos();
    Some((i64::try_from(nanos).ok()?, i64::try_from(meta.len()).ok()?))
}

pub fn read_session_file(storage: &Path, session_id: &str, filename: &str) -> Option<Vec<u8>> {
    let text = storage.to_string_lossy();
    if crate::location::is_remote_location(&text) {
        return crate::location::read_remote_session_file(&text, session_id, filename);
    }
    if storage.is_dir() {
        let root = directory_session_root(storage, session_id);
        return fs::File::open(root.join(filename))
            .ok()
            .and_then(|mut file| read_capped(&mut file, filename, MAX_MEMBER_BYTES).ok());
    }
    let primary = format!("{session_id}/{filename}");
    let suffix = format!("/{filename}");
    let members = archive_members(storage, &|name| {
        let name = normalize_member(name);
        name == primary || name == filename || name.ends_with(&suffix)
    })
    .ok()?;
    for want in [&primary, &filename.to_owned()] {
        if let Some((_, bytes)) = members
            .iter()
            .find(|(name, _)| normalize_member(name) == **want)
        {
            return Some(bytes.clone());
        }
    }
    // A wrapped archive whose folder name was not picked up as the session id
    // (tars keep the zip filename) still resolves by the inner path suffix —
    // but only when the archive holds one session tree, else the suffix could
    // serve a sibling session's file under this id.
    let roots: std::collections::BTreeSet<String> = members
        .iter()
        .filter_map(|(name, _)| {
            normalize_member(name)
                .split('/')
                .next()
                .filter(|root| !root.is_empty())
                .map(str::to_owned)
        })
        .collect();
    if roots.len() != 1 {
        return None;
    }
    members
        .into_iter()
        .find(|(name, _)| normalize_member(name).ends_with(&suffix))
        .map(|(_, bytes)| bytes)
}

pub fn read_timeslices(storage: &Path) -> Vec<Vec<u8>> {
    read_timeslice_frames(storage)
        .into_iter()
        .map(|(_, bytes)| bytes)
        .collect()
}

/// Timeslice files in layer/run order. The index is the `layer-N` folder, or
/// `-1` when the path has no layer folder.
pub fn read_timeslice_frames(storage: &Path) -> Vec<(i64, Vec<u8>)> {
    let mut members = if storage.is_dir() {
        let mut found = Vec::new();
        walk_files(storage, storage, 0, &is_timeslice, &mut found);
        found
    } else {
        archive_members(storage, &|name| is_timeslice(&normalize_member(name))).unwrap_or_default()
    };
    members.sort_by_key(|member| timeslice_key(&member.0));
    members
        .into_iter()
        .map(|(name, bytes)| {
            let index = if name.contains("layer-") {
                number_after(&name, "layer-")
            } else {
                -1
            };
            (index, bytes)
        })
        .collect()
}

/// `is_file` without following links — the walk refuses symlinks, so a
/// `input_map.csv -> elsewhere` must not admit a fake session either.
pub(crate) fn marker_file(path: &Path) -> bool {
    path.symlink_metadata().is_ok_and(|meta| meta.is_file())
}

/// `is_dir` without following links — a junction'd session dir resolves to
/// files outside the library root, so it is not a session dir.
pub(crate) fn unlinked_dir(path: &Path) -> bool {
    path.symlink_metadata()
        .is_ok_and(|meta| meta.is_dir() && !meta.file_type().is_symlink())
}

/// Metadata for a walk probe. NotFound is a definitive absence; any other
/// error flags the walk so delete_missing cannot prune on an unprobed path.
fn probe_meta(path: &Path, failed: &mut bool) -> Option<std::fs::Metadata> {
    match path.symlink_metadata() {
        Ok(meta) => Some(meta),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => None,
        Err(_) => {
            *failed = true;
            None
        }
    }
}

fn is_unpacked_session(folder: &Path, stat_failed: &mut bool) -> bool {
    let Some(name) = folder.file_name() else {
        return false;
    };
    let nested = folder.join(name);
    let nested_session = probe_meta(&nested, stat_failed)
        .is_some_and(|meta| meta.is_dir() && !meta.file_type().is_symlink())
        && probe_meta(&nested.join("input_map.csv"), stat_failed)
            .is_some_and(|meta| meta.is_file());
    nested_session
        || probe_meta(&folder.join("input_map.csv"), stat_failed).is_some_and(|meta| meta.is_file())
}

/// Directory that holds one session's files.
///
/// Unpacked logs use either `library/<id>/input_map.csv` or
/// `library/<id>/<id>/input_map.csv`. A miss stays on that session folder.
/// It does not fall back to the library, which would read every other session.
pub fn session_directory(library: &Path, session_id: &str) -> PathBuf {
    if !valid_session_id(session_id) {
        // Never resolve: the library root would read every other session.
        return library.join("\0invalid");
    }
    let flat = library.join(session_id);
    // A junction at `flat` escapes the library — its leaf metadata is fine, so
    // both levels are checked without following links.
    let flat_dir = unlinked_dir(&flat);
    let nested = flat.join(session_id);
    if flat_dir && unlinked_dir(&nested) && marker_file(&nested.join("input_map.csv")) {
        return nested;
    }
    if flat_dir && marker_file(&flat.join("input_map.csv")) {
        return flat;
    }
    if let Some(recorded) = crate::location::recorded_directory(library, session_id) {
        return recorded;
    }
    if let Some(catalog) = crate::location::catalog_directory(library, session_id) {
        return catalog;
    }
    // The root resolves only for the session it actually is — a data dir
    // that is itself a session must not answer every session id with its
    // files (a foreign id reaching here reads the wrong session's map).
    if library.file_name().is_some_and(|name| name == session_id)
        && marker_file(&library.join("input_map.csv"))
    {
        return library.to_path_buf();
    }
    flat
}

fn directory_session_root(folder: &Path, session_id: &str) -> PathBuf {
    session_directory(folder, session_id)
}

pub(crate) fn archive_kind(name: &str) -> Option<(String, &'static str)> {
    let lower = name.to_ascii_lowercase();
    for (suffix, kind) in ARCHIVE_SUFFIXES {
        if lower.ends_with(suffix) {
            let session_id = name[..name.len() - suffix.len()].to_owned();
            if !valid_session_id(&session_id) {
                return None;
            }
            return Some((session_id, *kind));
        }
    }
    None
}

pub(crate) fn archive_identity(path: &Path) -> Option<(String, &'static str)> {
    let (stem, kind) = archive_kind(&file_name(path)?)?;
    // The zip filename is whatever the user typed (`session.zip`); the real
    // session id is the folder wrapped inside. Peeking at a zip's central
    // directory is cheap — a tar would stream the whole archive, so tars
    // keep the filename stem.
    if kind == "zip" {
        if let Some(id) = zip_session_dir(path) {
            return Some((id, kind));
        }
    }
    Some((stem, kind))
}

/// The session folder wrapped inside a zip: the single top-level directory,
/// or the one holding a session marker when extra entries are present.
fn zip_session_dir(path: &Path) -> Option<String> {
    let file = File::open(path).ok()?;
    let mut archive = zip::ZipArchive::new(file).ok()?;
    let mut tops = std::collections::HashSet::new();
    let mut markers = std::collections::HashSet::new();
    for index in 0..archive.len() {
        let Ok(entry) = archive.by_index(index) else {
            continue;
        };
        let Ok(name) = entry.name() else {
            continue;
        };
        let name = normalize_member(&name);
        let Some((top, _)) = name.split_once('/') else {
            continue;
        };
        if !valid_session_id(top) {
            continue;
        }
        tops.insert(top.to_owned());
        if name.ends_with("/input_map.csv") || name.ends_with("/termination_summary.txt") {
            markers.insert(top.to_owned());
        }
    }
    if markers.len() == 1 {
        return markers.into_iter().next();
    }
    if tops.len() == 1 {
        return tops.into_iter().next();
    }
    None
}

fn file_name(path: &Path) -> Option<String> {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
}

pub(crate) use scan_kit_core::valid_session_id;

fn is_timeslice(name: &str) -> bool {
    normalize_member(name).ends_with("timeslice_data_device_units.csv")
}

fn timeslice_key(name: &str) -> (i64, i64, String) {
    (
        number_after(name, "layer-"),
        number_after(name, "run-"),
        normalize_member(name),
    )
}

pub(crate) fn number_after(path: &str, marker: &str) -> i64 {
    let Some(index) = path.rfind(marker) else {
        return 0;
    };
    let digits: String = path[index + marker.len()..]
        .chars()
        .take_while(|ch| ch.is_ascii_digit())
        .collect();
    digits.parse().unwrap_or(0)
}

fn normalize_member(name: &str) -> String {
    let name = name.replace('\\', "/");
    let name = name.trim_start_matches("./");
    // A mid-path `..` escapes the tree — the member must not bind to a file
    // it only reaches by traversal.
    if name.split('/').any(|part| part == "..") {
        return String::new();
    }
    name.to_owned()
}

fn walk_files(
    root: &Path,
    dir: &Path,
    depth: u8,
    want: &dyn Fn(&str) -> bool,
    out: &mut Vec<(String, Vec<u8>)>,
) {
    // The archive branch caps members and the aggregate; a local session tree
    // is the same trust boundary — bound both before the bytes land.
    // ponytail: a tree over the cap drops the rest silently (the archive path
    // errors instead) — a tail-truncated session still parses as complete.
    let mut total = 0u64;
    visit_files(
        root,
        dir,
        depth,
        want,
        &mut |name, path| {
            if total >= MAX_MEMBER_BYTES {
                return;
            }
            let Ok(mut file) = fs::File::open(path) else {
                return;
            };
            if let Ok(bytes) = read_capped(&mut file, name, MAX_MEMBER_BYTES - total) {
                total += bytes.len() as u64;
                out.push((name.to_string(), bytes));
            }
        },
        &mut |_| {},
    );
}

/// Timeslice files in layer/run order, without reading them.
///
/// The ordered list stays with the stamp, so a later poll does not walk the
/// session again.
pub(crate) fn list_timeslice_paths(storage: &Path) -> Vec<(i64, PathBuf)> {
    if !storage.is_dir() {
        return Vec::new();
    }
    let _ = refresh_slice(storage);
    slice_watches()
        .lock()
        .unwrap_or_else(|err| err.into_inner())
        .get(storage)
        .map(|hit| hit.paths.clone())
        .unwrap_or_default()
}

struct SliceWatch {
    dirs: Vec<(PathBuf, u128)>,
    files: Vec<(PathBuf, u128)>,
    paths: Vec<(i64, PathBuf)>,
    stamp: u128,
}

fn slice_watches() -> &'static Mutex<HashMap<PathBuf, SliceWatch>> {
    static CACHE: OnceLock<Mutex<HashMap<PathBuf, SliceWatch>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Length and mtime of each timeslice, or of the archive itself.
///
/// An unchanged session restats the files and layer directories from the last
/// walk. A changed layer directory, or a changed file, walks again.
///
/// ponytail: metadata only, so a same-size rewrite in the same timestamp tick
/// is missed. Hash a prefix if that shows up. 32 sessions stay remembered.
pub fn timeslice_stamp(storage: &Path) -> u128 {
    refresh_slice(storage)
}

fn refresh_slice(storage: &Path) -> u128 {
    if !storage.is_dir() {
        return meta_stamp(storage);
    }
    let key = storage.to_path_buf();
    if let Some(hit) = slice_watches()
        .lock()
        .unwrap_or_else(|err| err.into_inner())
        .get(&key)
    {
        if hit
            .dirs
            .iter()
            .all(|(path, stamp)| meta_stamp(path) == *stamp)
            && hit
                .files
                .iter()
                .all(|(path, stamp)| meta_stamp(path) == *stamp)
        {
            return hit.stamp;
        }
    }
    let mut found = Vec::new();
    let mut dirs = vec![(storage.to_path_buf(), meta_stamp(storage))];
    // Stamp every directory the walk visits, not just ancestors of matches —
    // a layer dir created after a poll must itself bump the stamp or a
    // timeslice written inside it is never noticed.
    visit_files(
        storage,
        storage,
        0,
        &is_timeslice,
        &mut |name, path| {
            found.push((name.to_string(), path.to_path_buf(), meta_stamp(path)));
        },
        &mut |dir| {
            if dirs.iter().all(|(have, _)| have != dir) {
                dirs.push((dir.to_path_buf(), meta_stamp(dir)));
            }
        },
    );
    found.sort_by_key(|item| timeslice_key(&item.0));
    let paths = found
        .iter()
        .map(|(name, path, _)| {
            let index = if name.contains("layer-") {
                number_after(name, "layer-")
            } else {
                -1
            };
            (index, path.clone())
        })
        .collect();
    let files = found
        .into_iter()
        .map(|(_, path, stamp)| (path, stamp))
        .collect::<Vec<_>>();
    let mut stamp = 0x9e3779b97f4a7c15u128;
    let mut count = 0u128;
    for (_, file_stamp) in &files {
        count += 1;
        stamp ^= file_stamp.wrapping_mul(count | 1);
    }
    stamp ^= count.rotate_left(17);
    let mut cache = slice_watches()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    if cache.len() >= 32 {
        if let Some(drop) = cache.keys().next().cloned() {
            cache.remove(&drop);
        }
    }
    cache.insert(
        key,
        SliceWatch {
            dirs,
            files,
            paths,
            stamp,
        },
    );
    stamp
}

pub(crate) fn meta_stamp(path: &Path) -> u128 {
    let Ok(meta) = fs::metadata(path) else {
        return 0;
    };
    let ns = meta
        .modified()
        .ok()
        .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|time| time.as_nanos())
        .unwrap_or(0);
    ns ^ u128::from(meta.len())
}

fn visit_files(
    root: &Path,
    dir: &Path,
    depth: u8,
    want: &dyn Fn(&str) -> bool,
    visit: &mut dyn FnMut(&str, &Path),
    dirs: &mut dyn FnMut(&Path),
) {
    if depth > 6 {
        return;
    }
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    let mut children: Vec<_> = entries.filter_map(|entry| entry.ok()).collect();
    children.sort_by_key(|entry| entry.file_name());
    for child in children {
        let path = child.path();
        // `is_dir`/`is_file` follow links; the library walk does not. A session
        // symlink must not pull timeslice files in from outside the session.
        let Ok(kind) = child.file_type() else {
            continue;
        };
        if kind.is_symlink() {
            continue;
        }
        if kind.is_dir() {
            dirs(&path);
            visit_files(root, &path, depth + 1, want, visit, dirs);
        } else if kind.is_file() {
            let name = path
                .strip_prefix(root)
                .unwrap_or(&path)
                .to_string_lossy()
                .replace('\\', "/");
            if want(&name) {
                visit(&name, &path);
            }
        }
    }
}

fn archive_members(
    path: &Path,
    want: &dyn Fn(&str) -> bool,
) -> Result<Vec<(String, Vec<u8>)>, String> {
    let name = file_name(path).unwrap_or_default().to_ascii_lowercase();
    if name.ends_with(".zip") {
        read_zip(path, want)
    } else {
        read_tar(path, want)
    }
}

fn read_zip(path: &Path, want: &dyn Fn(&str) -> bool) -> Result<Vec<(String, Vec<u8>)>, String> {
    let file = File::open(path).map_err(|err| err.to_string())?;
    let mut archive = zip::ZipArchive::new(file).map_err(|err| err.to_string())?;
    let mut out = Vec::new();
    let mut total = 0u64;
    for index in 0..archive.len() {
        // One unsupported-compression or undecodable member must not brick
        // every read — the discovery peek skips them the same way.
        let Ok(mut entry) = archive.by_index(index) else {
            continue;
        };
        if entry.is_dir() {
            continue;
        }
        let Ok(name) = entry.name().map(|name| name.into_owned()) else {
            continue;
        };
        if !want(&name) {
            continue;
        }
        // Bound the allocation by the remaining aggregate budget — reading a
        // full member first would spend the cap twice over.
        let bytes = read_capped(&mut entry, &name, MAX_MEMBER_BYTES.saturating_sub(total))?;
        total = total.saturating_add(bytes.len() as u64);
        out.push((name.clone(), bytes));
    }
    Ok(out)
}

fn read_tar(path: &Path, want: &dyn Fn(&str) -> bool) -> Result<Vec<(String, Vec<u8>)>, String> {
    let file = File::open(path).map_err(|err| err.to_string())?;
    let name = file_name(path).unwrap_or_default().to_ascii_lowercase();
    let decoder: Box<dyn Read> = if name.ends_with(".tar.gz") || name.ends_with(".tgz") {
        Box::new(flate2::read::GzDecoder::new(file))
    } else if name.ends_with(".tar.bz2") {
        Box::new(bzip2::read::BzDecoder::new(file))
    } else if name.ends_with(".tar.xz") {
        Box::new(xz2::read::XzDecoder::new(file))
    } else {
        Box::new(file)
    };
    let mut archive = tar::Archive::new(decoder);
    let mut out = Vec::new();
    let mut total = 0u64;
    let entries = archive.entries().map_err(|err| err.to_string())?;
    for entry in entries {
        // Same rule as the zip reader — one undecodable member must not
        // brick every read. A failed entry may leave the stream desynced,
        // so keep what was read and stop; a bad path skips just the member.
        let Ok(mut entry) = entry else {
            break;
        };
        if !entry.header().entry_type().is_file() {
            continue;
        }
        let Ok(path) = entry.path() else {
            continue;
        };
        let name = path.to_string_lossy().into_owned();
        if !want(&name) {
            continue;
        }
        let bytes = read_capped(&mut entry, &name, MAX_MEMBER_BYTES.saturating_sub(total))?;
        total = total.saturating_add(bytes.len() as u64);
        out.push((name.clone(), bytes));
    }
    Ok(out)
}

/// Session members are CSV/XML/JSON — none legitimately approach this. A
/// crafted archive inflating past it is a decompression bomb, not a session.
pub(crate) const MAX_MEMBER_BYTES: u64 = 4 << 30;

pub(crate) fn read_capped(member: &mut impl Read, name: &str, max: u64) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::new();
    member
        .take(max + 1)
        .read_to_end(&mut bytes)
        .map_err(|err| err.to_string())?;
    if bytes.len() as u64 > max {
        return Err(format!("{name} expands past {max} bytes"));
    }
    Ok(bytes)
}

#[cfg(test)]
pub fn write_zip(path: &Path, members: &[(&str, &[u8])]) -> Result<(), String> {
    use std::io::Write;

    let file = File::create(path).map_err(|err| err.to_string())?;
    let mut zip = zip::ZipWriter::new(file);
    let options =
        zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
    for (name, bytes) in members {
        zip.start_file(*name, options)
            .map_err(|err| err.to_string())?;
        zip.write_all(bytes).map_err(|err| err.to_string())?;
    }
    zip.finish().map_err(|err| err.to_string())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_member_with_a_mid_path_traversal_binds_to_nothing() {
        // "sess/../evil/input_map.csv" must not satisfy the single-tree
        // suffix fallback — it escapes the session tree it claims.
        assert_eq!(super::normalize_member("sess/../evil/input_map.csv"), "");
        assert_eq!(super::normalize_member("a/b/../c"), "");
        assert_eq!(
            super::normalize_member("sess/input_map.csv"),
            "sess/input_map.csv"
        );
    }

    use super::*;

    #[test]
    fn a_directory_with_an_invalid_session_id_is_skipped() {
        // An id that fails valid_session_id could never be trashed or deleted
        // — browsing it would trap the row in the catalog.
        let mut entries = Vec::new();
        let mut skipped = Vec::new();
        add_directory(
            &mut entries,
            &mut skipped,
            Discovered {
                session_id: "not:a:session".into(),
                storage_path: PathBuf::from("/lib/not:a:session"),
                kind: "directory",
            },
            "not:a:session",
        );
        assert!(entries.is_empty());
        assert_eq!(skipped.len(), 1);
    }

    #[test]
    fn an_over_cap_member_errors_instead_of_eating_memory() {
        let payload = vec![b'x'; 16];
        let mut reader = std::io::Cursor::new(payload.clone());
        assert_eq!(read_capped(&mut reader, "m.csv", 16).unwrap(), payload);
        let mut reader = std::io::Cursor::new(payload);
        let err = read_capped(&mut reader, "m.csv", 15).unwrap_err();
        assert!(err.contains("expands past"), "{err}");
    }

    #[test]
    fn a_walk_stops_at_eight_levels() {
        let root = std::env::temp_dir().join(format!(
            "scan-kit-nest-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = fs::remove_dir_all(&root);
        let mut seen = root.clone();
        for index in 0..7 {
            seen.push(format!("n{index}"));
        }
        seen.push("seen");
        fs::create_dir_all(&seen).unwrap();
        fs::write(seen.join("input_map.csv"), b"energy\n1\n").unwrap();
        let mut miss = root.clone();
        for index in 0..8 {
            miss.push(format!("n{index}"));
        }
        miss.push("miss");
        fs::create_dir_all(&miss).unwrap();
        fs::write(miss.join("input_map.csv"), b"energy\n1\n").unwrap();

        let ids = discover(&root, &mut |_, _| true)
            .unwrap()
            .entries
            .into_iter()
            .map(|entry| entry.session_id)
            .collect::<Vec<_>>();
        assert!(ids.contains(&"seen".to_owned()), "{ids:?}");
        assert!(!ids.contains(&"miss".to_owned()), "{ids:?}");
        let _ = fs::remove_dir_all(&root);
    }
    #[test]
    fn a_staging_dir_does_not_catalog_as_a_ghost_session() {
        // A leftover download staging dir holds a bare "session/" tree —
        // without the skip it catalogs a ghost id that can then be trashed.
        let root = std::env::temp_dir().join(format!(
            "scan-kit-staging-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = fs::remove_dir_all(&root);
        let ghost = root.join(".session_staging-123/session");
        fs::create_dir_all(&ghost).unwrap();
        fs::write(
            ghost.join("input_map.csv"),
            b"energy
1
",
        )
        .unwrap();
        fs::create_dir_all(root.join("real")).unwrap();
        fs::write(
            root.join("real/input_map.csv"),
            b"energy
1
",
        )
        .unwrap();
        let ids = discover(&root, &mut |_, _| true)
            .unwrap()
            .entries
            .into_iter()
            .map(|entry| entry.session_id)
            .collect::<Vec<_>>();
        assert_eq!(ids, ["real"], "{ids:?}");
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_session_library_root_answers_only_its_own_id() {
        // A data dir that is itself a session must not resolve a foreign
        // valid id to its own input_map.csv.
        let root = std::env::temp_dir().join(format!(
            "scan-kit-rootid-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let library = root.join("sess");
        fs::create_dir_all(&library).unwrap();
        fs::write(
            library.join("input_map.csv"),
            b"energy
1
",
        )
        .unwrap();
        assert_eq!(session_directory(&library, "sess"), library);
        let foreign = session_directory(&library, "other");
        assert!(
            foreign != library,
            "foreign id resolved to the root session"
        );
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_zip_named_session_takes_the_wrapped_session_id() {
        let root = std::env::temp_dir().join(format!(
            "scan-kit-zipid-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        write_zip(
            &root.join("session.zip"),
            &[
                ("2131261153/input_map.csv", b"energy\n70\n"),
                (
                    "2131261153/termination_summary.txt",
                    b"Primary total dose: 5.0\n",
                ),
            ],
        )
        .unwrap();

        let found = discover(&root, &mut |_, _| true).unwrap();
        assert_eq!(found.entries.len(), 1);
        let entry = &found.entries[0];
        assert_eq!(entry.session_id, "2131261153");
        let summary = read_session_file(
            &entry.storage_path,
            &entry.session_id,
            "termination_summary.txt",
        )
        .unwrap();
        assert!(summary.starts_with(b"Primary total dose"));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn traversal_members_do_not_become_the_session_id() {
        let root = std::env::temp_dir().join(format!(
            "scan-kit-ziptrav-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        write_zip(
            &root.join("bundle.zip"),
            &[("../../evil/input_map.csv", b"energy\n70\n")],
        )
        .unwrap();

        let found = discover(&root, &mut |_, _| true).unwrap();
        assert_eq!(found.entries.len(), 1);
        assert_eq!(found.entries[0].session_id, "bundle");
        assert!(crate::location::session_storage_paths(&root.to_string_lossy(), "..").is_err());
        // Invalid ids must never resolve to the library root — that would
        // read every other session.
        for id in ["..", ".", "a/b", "a\\b"] {
            let dir = session_directory(&root, id);
            assert_ne!(dir, root);
            assert!(!dir.starts_with(&root) || !dir.is_dir());
        }
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_stray_inner_dir_does_not_steal_a_flat_session() {
        // A flattened session keeps a stray <id>/<id>/ dir — the storage-dir
        // lookup must not resolve into it and miss input_map.csv.
        let root = std::env::temp_dir().join(format!("scan-kit-stray-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let sess = root.join("sess");
        fs::create_dir_all(sess.join("sess")).unwrap();
        fs::write(
            sess.join("input_map.csv"),
            "energy
70
",
        )
        .unwrap();
        assert_eq!(session_directory(&sess, "sess"), sess);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_dot_prefixed_archive_is_not_a_session() {
        assert_eq!(archive_kind("..zip"), None);
        assert_eq!(archive_kind(".zip"), None);
        assert_eq!(archive_kind("..tar.bz2"), None);
        assert_eq!(archive_kind("logs.zip"), Some(("logs".to_owned(), "zip")));
    }

    #[test]
    fn a_session_walk_does_not_follow_symlinked_dirs() {
        let root = std::env::temp_dir().join(format!(
            "scan-kit-symlink-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = fs::remove_dir_all(&root);
        let session = root.join("sess");
        let outside = root.join("outside");
        fs::create_dir_all(session.join("layer-0")).unwrap();
        fs::create_dir_all(outside.join("layer-9")).unwrap();
        fs::write(
            session.join("layer-0/timeslice_data_device_units.csv"),
            b"a\n",
        )
        .unwrap();
        fs::write(
            outside.join("layer-9/timeslice_data_device_units.csv"),
            b"b\n",
        )
        .unwrap();
        // Symlink creation needs extra privilege on Windows; skip when refused.
        #[cfg(unix)]
        let linked = std::os::unix::fs::symlink(&outside, session.join("linked"));
        #[cfg(windows)]
        let linked = std::os::windows::fs::symlink_dir(&outside, session.join("linked"));
        #[cfg(not(any(unix, windows)))]
        let linked: std::io::Result<()> = Ok(());
        let Ok(()) = linked else {
            let _ = fs::remove_dir_all(&root);
            return;
        };

        let frames = read_timeslice_frames(&session);
        assert_eq!(frames.len(), 1, "{frames:?}");
        assert_eq!(frames[0].0, 0);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_write_inside_a_previously_empty_layer_dir_bumps_the_stamp() {
        let root = std::env::temp_dir().join(format!(
            "scan-kit-slice-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = fs::remove_dir_all(&root);
        let storage = root.join("sess");
        fs::create_dir_all(storage.join("layer-0")).unwrap();
        let first = timeslice_stamp(&storage);
        // The watch is mtime-based — outlast coarse-granularity filesystems so
        // the new run directory's timestamp visibly differs from `first`.
        std::thread::sleep(std::time::Duration::from_millis(1100));
        fs::create_dir_all(storage.join("layer-0/run-0")).unwrap();
        fs::write(
            storage.join("layer-0/run-0/timeslice_data_device_units.csv"),
            b"a,b\n1,2\n",
        )
        .unwrap();
        assert_ne!(timeslice_stamp(&storage), first);
        assert_eq!(list_timeslice_paths(&storage).len(), 1);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_cancelled_local_walk_errors_instead_of_returning_a_partial_listing() {
        // A partial Found looks like mass deletion to delete_missing —
        // cancellation must surface as an error, like the remote walk.
        let root =
            std::env::temp_dir().join(format!("scan-kit-discover-cancel-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("child")).unwrap();
        std::fs::write(
            root.join("child").join("keep.csv"),
            "a
",
        )
        .unwrap();
        let err = discover(&root, &mut |_, _| false).err().unwrap();
        assert!(err.contains("cancelled"), "{err}");
        let _ = std::fs::remove_dir_all(&root);
    }
}
