//! Local session discovery. Archives are read in memory and not unpacked.
//!
//! ponytail: directory walks stop at depth 6, which covers `layer-N/run-M`.
//! A session laid out deeper than that needs a longer walk.

use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};

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

pub fn discover_entries(root: &Path) -> Result<Vec<Discovered>, String> {
    let mut children = match fs::read_dir(root) {
        Ok(entries) => entries.filter_map(|entry| entry.ok()).collect::<Vec<_>>(),
        Err(_) => return Ok(Vec::new()),
    };
    children.sort_by_key(|entry| entry.file_name());

    let mut seen: Vec<Discovered> = Vec::new();
    for entry in &children {
        let path = entry.path();
        if path.is_dir() && is_unpacked_session(&path) {
            let Some(session_id) = file_name(&path) else {
                continue;
            };
            seen.push(Discovered {
                session_id,
                storage_path: path,
                kind: "directory",
            });
        }
    }
    for entry in &children {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some((session_id, kind)) = archive_identity(&path) else {
            continue;
        };
        if seen.iter().any(|item| item.session_id == session_id) {
            continue;
        }
        seen.push(Discovered {
            session_id,
            storage_path: path,
            kind,
        });
    }
    seen.sort_by(|left, right| left.session_id.cmp(&right.session_id));
    Ok(seen)
}

pub fn storage_fingerprint(storage: &Path, session_id: &str) -> Option<(i64, i64)> {
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
    let mut members = if storage.is_dir() {
        let mut found = Vec::new();
        walk_files(storage, storage, 0, &is_timeslice, &mut found);
        found
    } else {
        archive_members(storage, &|name| is_timeslice(&normalize_member(name))).unwrap_or_default()
    };
    members.sort_by_key(|member| timeslice_key(&member.0));
    members.into_iter().map(|(_, bytes)| bytes).collect()
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
    flat
}

fn directory_session_root(folder: &Path, session_id: &str) -> PathBuf {
    session_directory(folder, session_id)
}

fn archive_identity(path: &Path) -> Option<(String, &'static str)> {
    let name = file_name(path)?;
    let lower = name.to_ascii_lowercase();
    for (suffix, kind) in ARCHIVE_SUFFIXES {
        if lower.ends_with(suffix) {
            let session_id = name[..name.len() - suffix.len()].to_owned();
            if session_id.is_empty() {
                return None;
            }
            return Some((session_id, kind));
        }
    }
    None
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
            walk_files(root, &path, depth + 1, want, out);
        } else if path.is_file() {
            let name = path
                .strip_prefix(root)
                .unwrap_or(&path)
                .to_string_lossy()
                .replace('\\', "/");
            if want(&name) {
                if let Ok(bytes) = fs::read(&path) {
                    out.push((name, bytes));
                }
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
