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

const ARCHIVE_SUFFIXES: &[(&str, &str)] = &[
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
}

pub(crate) fn discover(root: &Path) -> Result<Found, String> {
    let text = root.to_string_lossy();
    if crate::location::is_remote_location(&text) {
        return crate::location::discover_remote(&text);
    }
    let mut entries = Vec::new();
    let mut skipped = Vec::new();
    let mut archives = Vec::new();
    let mut exams = Vec::new();
    if is_unpacked_session(root) {
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
    } else if let Some(exam) = exam_in(root) {
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
        );
    }
    add_archives(&mut entries, &mut skipped, archives);
    exams.sort_by(|left, right| left.folder_name.cmp(&right.folder_name));
    Ok(Found {
        entries,
        skipped,
        exams,
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
) {
    if depth >= crate::location::NEST_DEPTH {
        return;
    }
    let Ok(read) = fs::read_dir(dir) else {
        return;
    };
    let mut children: Vec<_> = read.filter_map(|entry| entry.ok()).collect();
    children.sort_by_key(|entry| entry.file_name());
    let mut sessions = Vec::new();
    for child in &children {
        let Ok(kind) = child.file_type() else {
            continue;
        };
        if kind.is_symlink() || !kind.is_dir() {
            continue;
        }
        let path = child.path();
        if !is_unpacked_session(&path) {
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
        if kind.is_symlink()
            || !kind.is_dir()
            || sessions.iter().any(|name| name == &child.file_name())
        {
            continue;
        }
        let path = child.path();
        if let Some(exam) = exam_in(&path) {
            add_exam(exams, skipped, exam, &relative(library, &path));
            continue;
        }
        walk_local(library, &path, depth + 1, entries, skipped, archives, exams);
    }
}

fn exam_in(dir: &Path) -> Option<ExamHit> {
    let folder_name = file_name(dir)?;
    let mut files = Vec::new();
    let mut count = 0i64;
    for entry in fs::read_dir(dir).ok()?.flatten() {
        let Ok(kind) = entry.file_type() else {
            continue;
        };
        if kind.is_symlink() {
            continue;
        }
        count += 1;
        if kind.is_file() {
            files.push(entry.path());
        }
    }
    files.sort();
    let sample = files
        .into_iter()
        .find(|path| scan_kit_dicom::file_is_dicom(path))?;
    let peek = scan_kit_dicom::peek_file(&sample);
    let study_uid = if peek.study_uid.is_empty() {
        folder_name.clone()
    } else {
        peek.study_uid
    };
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
        let inner = storage.join(session_id).join("termination_summary.txt");
        if outer.is_file() {
            outer
        } else if inner.is_file() {
            inner
        } else {
            storage.to_path_buf()
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
        return fs::read(root.join(filename)).ok();
    }
    let primary = format!("{session_id}/{filename}");
    let members = archive_members(storage, &|name| {
        let name = normalize_member(name);
        name == primary || name == filename
    })
    .ok()?;
    if let Some((_, bytes)) = members
        .iter()
        .find(|(name, _)| normalize_member(name) == primary)
    {
        return Some(bytes.clone());
    }
    members
        .into_iter()
        .find(|(name, _)| normalize_member(name) == filename)
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

fn is_unpacked_session(folder: &Path) -> bool {
    let Some(name) = folder.file_name() else {
        return false;
    };
    folder.join(name).join("input_map.csv").is_file() || folder.join("input_map.csv").is_file()
}

/// Directory that holds one session's files.
///
/// Unpacked logs use either `library/<id>/input_map.csv` or
/// `library/<id>/<id>/input_map.csv`. A miss stays on that session folder.
/// It does not fall back to the library, which would read every other session.
pub fn session_directory(library: &Path, session_id: &str) -> PathBuf {
    let nested = library.join(session_id).join(session_id);
    if nested.join("input_map.csv").is_file() {
        return nested;
    }
    let flat = library.join(session_id);
    if flat.join("input_map.csv").is_file() || flat.is_dir() {
        return flat;
    }
    if library.join("input_map.csv").is_file() {
        return library.to_path_buf();
    }
    if let Some(recorded) = crate::location::recorded_directory(library, session_id) {
        return recorded;
    }
    if let Some(catalog) = crate::location::catalog_directory(library, session_id) {
        return catalog;
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
            if session_id.is_empty() {
                return None;
            }
            return Some((session_id, *kind));
        }
    }
    None
}

fn archive_identity(path: &Path) -> Option<(String, &'static str)> {
    archive_kind(&file_name(path)?)
}

fn file_name(path: &Path) -> Option<String> {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
}

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

fn number_after(path: &str, marker: &str) -> i64 {
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
    name.trim_start_matches("./").to_owned()
}

fn walk_files(
    root: &Path,
    dir: &Path,
    depth: u8,
    want: &dyn Fn(&str) -> bool,
    out: &mut Vec<(String, Vec<u8>)>,
) {
    visit_files(root, dir, depth, want, &mut |name, path| {
        if let Ok(bytes) = fs::read(path) {
            out.push((name.to_string(), bytes));
        }
    });
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
    visit_files(storage, storage, 0, &is_timeslice, &mut |name, path| {
        if let Some(parent) = path.parent() {
            let parent = parent.to_path_buf();
            if dirs.iter().all(|(have, _)| have != &parent) {
                dirs.push((parent.clone(), meta_stamp(&parent)));
            }
        }
        found.push((name.to_string(), path.to_path_buf(), meta_stamp(path)));
    });
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
        if path.is_dir() {
            visit_files(root, &path, depth + 1, want, visit);
        } else if path.is_file() {
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
    for index in 0..archive.len() {
        let mut entry = archive.by_index(index).map_err(|err| err.to_string())?;
        if entry.is_dir() {
            continue;
        }
        let name = entry.name().to_owned();
        if !want(&name) {
            continue;
        }
        let mut bytes = Vec::new();
        entry
            .read_to_end(&mut bytes)
            .map_err(|err| err.to_string())?;
        out.push((name, bytes));
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
    let entries = archive.entries().map_err(|err| err.to_string())?;
    for entry in entries {
        let mut entry = entry.map_err(|err| err.to_string())?;
        if !entry.header().entry_type().is_file() {
            continue;
        }
        let path = entry.path().map_err(|err| err.to_string())?;
        let name = path.to_string_lossy().into_owned();
        if !want(&name) {
            continue;
        }
        let mut bytes = Vec::new();
        entry
            .read_to_end(&mut bytes)
            .map_err(|err| err.to_string())?;
        out.push((name, bytes));
    }
    Ok(out)
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
    use super::*;

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

        let ids = discover(&root)
            .unwrap()
            .entries
            .into_iter()
            .map(|entry| entry.session_id)
            .collect::<Vec<_>>();
        assert!(ids.contains(&"seen".to_owned()), "{ids:?}");
        assert!(!ids.contains(&"miss".to_owned()), "{ids:?}");
        let _ = fs::remove_dir_all(&root);
    }
}
