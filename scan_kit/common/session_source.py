"""Resolve session data from directories, ZIP, or tar-based archives (tgz, tar.gz, …)."""

from __future__ import annotations

import os
import re
import shutil
import tarfile
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
import zipfile
from dataclasses import dataclass
from pathlib import Path

import logging

import pandas as pd

from .session_meta import SessionMeta, parse_termination_summary_text
from .schema import (
    canonical_column_aliases,
    canonicalize_dataframe_columns,
    resolve_column_name,
    resolve_concept_column,
    resolve_requested_column,
)
from .data_location import (
    canonical_location,
    is_remote_location,
    join_location,
    list_location_entries,
    location_exists,
    location_is_dir,
    location_is_file,
    location_stat,
    materialize_session,
    open_location_binary,
    read_location_bytes,
)

_log = logging.getLogger(__name__)

_TIMESLICE_RE = re.compile(
    r"^[^/]+/layer-(\d+)/run-(\d+)/timeslice_data_device_units\.csv$"
)
_SPOT_LAYER_RE = re.compile(
    r"^[^/]+/layer-(\d+)/run-(\d+)/([^/]+_spot_data\.csv)$"
)
_POINT_TIME_COLUMN_ALIASES = ("point_time(ms)", "point_time_ms", "point_time")
_PREFERRED_POINT_TIME_SPOT_FILES = (
    "FX4_spot_data.csv",
    "IX256_1_spot_data.csv",
    "IX256_2_spot_data.csv",
    "RCI_spot_data.csv",
)

# Longest suffix first so e.g. .tar.gz wins over .gz
_ARCHIVE_SUFFIXES: tuple[str, ...] = (
    ".tar.gz",
    ".tar.bz2",
    ".tar.xz",
    ".tgz",
    ".tar",
    ".zip",
)
_DEFAULT_CANONICAL_ALIASES = canonical_column_aliases()


def _resolve_raw_csv_usecols(
    header_columns: list[str],
    canonical_usecols: list[str],
) -> list[str]:
    """Map requested post-canonical column names to raw CSV header names."""
    raw: list[str] = []
    seen: set[str] = set()
    header_list = [str(c).strip() for c in header_columns]

    for requested in canonical_usecols:
        resolved = None
        if requested in header_list:
            resolved = requested
        else:
            resolved = resolve_concept_column(header_list, requested)
        if resolved is None:
            resolved = resolve_requested_column(header_list, requested)
        if resolved is None:
            resolved = resolve_column_name(header_list, requested)
        if resolved is None or resolved in seen:
            continue
        seen.add(resolved)
        raw.append(resolved)
    return raw


def _read_csv_header_columns(source) -> list[str] | None:
    """Read CSV column names without loading row data."""
    try:
        if isinstance(source, (str, Path, os.PathLike)):
            return pd.read_csv(
                source, nrows=0, index_col=False, skipinitialspace=True
            ).columns.tolist()
        if hasattr(source, "seekable") and source.seekable():
            pos = source.tell()
            header = pd.read_csv(
                source, nrows=0, index_col=False, skipinitialspace=True
            ).columns.tolist()
            source.seek(pos)
            return header
    except Exception:
        return None
    return None


def _read_csv_robust(
    source,
    *,
    usecols: list[str] | None = None,
    raw_usecols: list[str] | None = None,
    aliases: dict[str, tuple[str, ...]] | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Read a CSV and tolerate schema drift in column naming.

    When *raw_usecols* is supplied (typically resolved once per session from the
    first layer header), header re-read and column resolution are skipped.
    """
    alias_map = aliases or _DEFAULT_CANONICAL_ALIASES
    read_usecols = raw_usecols
    if usecols is not None and read_usecols is None:
        header = _read_csv_header_columns(source)
        if header is not None:
            read_usecols = _resolve_raw_csv_usecols(header, usecols)
            if not read_usecols:
                read_usecols = None

    read_kwargs: dict = {
        "index_col": False,
        "skipinitialspace": True,
        "usecols": read_usecols,
    }
    if nrows is not None:
        read_kwargs["nrows"] = nrows
    df = pd.read_csv(source, **read_kwargs)
    df = canonicalize_dataframe_columns(df, aliases=alias_map, copy=False)
    if usecols is None:
        return df
    seen: set[str] = set()
    keep: list[str] = []
    for col in usecols:
        if col in df.columns and col not in seen:
            seen.add(col)
            keep.append(col)
    if not keep:
        return df.iloc[:, 0:0]
    out = df[keep]
    if out.columns.duplicated().any():
        out = out.loc[:, ~out.columns.duplicated()]
    return out


def _resolve_timeslice_raw_usecols(
    source,
    usecols: list[str],
) -> list[str] | None:
    """Resolve raw CSV column names for *usecols* from one timeslice file."""
    header = _read_csv_header_columns(source)
    if header is None:
        return None
    raw = _resolve_raw_csv_usecols(header, usecols)
    return raw or None


def _strip_archive_suffix(filename: str) -> str | None:
    """Return session id stem if *filename* matches a known archive suffix."""
    lower = filename.lower()
    for suf in _ARCHIVE_SUFFIXES:
        if lower.endswith(suf):
            return filename[: -len(suf)]
    return None


def session_source_from_archive(path: Path) -> SessionSource | None:
    """Build a :class:`SessionSource` from a single archive file path."""
    stem = _strip_archive_suffix(path.name)
    if stem is None:
        return None
    sid = stem
    lower = path.name.lower()
    if lower.endswith(".zip"):
        return SessionSource("zip", path, sid)
    if lower.endswith((".tgz", ".tar.gz", ".tar.bz2", ".tar.xz", ".tar")):
        return SessionSource("tar", path, sid)
    return None


def _directory_session_root(folder: Path, session_id: str) -> Path | None:
    inner = folder / session_id
    if (inner / "input_map.csv").is_file():
        return inner
    if (folder / "input_map.csv").is_file():
        return folder
    return None


def peek_session_source_from_path(
    storage_path: str | Path,
    session_id: str,
) -> SessionSource | None:
    """Build a source from a discovered path without extracting archives."""
    spec = str(storage_path)
    if is_remote_location(spec):
        return None
    path = Path(storage_path)
    if path.is_dir():
        root = _directory_session_root(path, session_id)
        if root is None:
            return None
        return SessionSource("directory", root, session_id)
    if path.is_file():
        src = session_source_from_archive(path)
        if src is not None and src.session_id == session_id:
            return src
    return None


def peek_session_source(
    session_id: str,
    base_dir: str | Path,
) -> SessionSource | None:
    """Find a session under *base_dir* without extracting archives.

    Preference matches discovery: unpacked directory, then zip, then tar.
    """
    spec = str(base_dir)
    if is_remote_location(spec):
        return None
    base = Path(base_dir)
    folder = base / session_id
    if folder.is_dir():
        src = peek_session_source_from_path(folder, session_id)
        if src is not None:
            return src
    zp = base / f"{session_id}.zip"
    if zp.is_file():
        return SessionSource("zip", zp, session_id)
    for suf in _ARCHIVE_SUFFIXES:
        if suf == ".zip":
            continue
        ap = base / f"{session_id}{suf}"
        if ap.is_file():
            return SessionSource("tar", ap, session_id)
    return None


def list_session_storage_paths(base_dir: str | Path, session_id: str) -> list[str]:
    """Folder and leftover archives for *session_id* under *base_dir*."""
    spec = str(base_dir)
    names = [session_id]
    names.extend(f"{session_id}{suf}" for suf in _ARCHIVE_SUFFIXES)
    found: list[str] = []
    if is_remote_location(spec):
        for name in names:
            child = join_location(spec, name)
            if location_exists(child):
                found.append(child)
        return found
    base = Path(base_dir)
    for name in names:
        path = base / name
        if name == session_id:
            if path.exists():
                found.append(str(path))
        elif path.is_file():
            found.append(str(path))
    return found


def storage_fingerprint(
    storage_path: str | Path,
    session_id: str,
) -> tuple[str, int, int] | None:
    """``(resolved_path, mtime_ns, size)`` for cache keys.

    Directories fingerprint ``termination_summary.txt`` when present so a
    metadata edit is visible; archives use the archive file itself.
    """
    spec = str(storage_path)
    if is_remote_location(spec):
        target = spec
        if location_is_dir(spec):
            for rel in (
                "termination_summary.txt",
                f"{session_id}/termination_summary.txt",
            ):
                child = join_location(spec, rel)
                if location_is_file(child):
                    target = child
                    break
        st = location_stat(target)
        if st is None:
            return None
        return (canonical_location(target), st[0], st[1])
    path = Path(storage_path)
    target = path
    if path.is_dir():
        for candidate in (
            path / "termination_summary.txt",
            path / session_id / "termination_summary.txt",
        ):
            if candidate.is_file():
                target = candidate
                break
    try:
        st = target.stat()
        return (str(target.resolve()), st.st_mtime_ns, st.st_size)
    except OSError:
        return None


_META_CACHE: dict[tuple[str, int, int], SessionMeta] = {}
_META_CACHE_LOCK = threading.Lock()


def clear_termination_summary_cache() -> None:
    with _META_CACHE_LOCK:
        _META_CACHE.clear()


def load_termination_summary_cached(
    session_id: str,
    storage_path: str | Path,
) -> SessionMeta | None:
    """Parse ``termination_summary.txt`` without extracting; cache by fingerprint."""
    fp = storage_fingerprint(storage_path, session_id)
    if fp is not None:
        with _META_CACHE_LOCK:
            hit = _META_CACHE.get(fp)
        if hit is not None:
            return hit
    spec = str(storage_path)
    if is_remote_location(spec):
        text = _remote_termination_summary_text(spec, session_id)
        meta = parse_termination_summary_text(text) if text else None
    else:
        src = peek_session_source_from_path(storage_path, session_id)
        meta = load_session_termination_summary(src) if src else None
    if meta is not None and fp is not None:
        with _META_CACHE_LOCK:
            _META_CACHE[fp] = meta
    return meta


@dataclass(frozen=True)
class SessionSource:
    """Where to read session CSVs for one session."""

    kind: str  # "directory" | "zip" | "tar"
    path: Path  # inner folder with input_map.csv, or path to .zip / .tgz
    session_id: str


def _is_unpacked_session_directory(child: Path) -> bool:
    """True if *child* is ``{base}/{session_id}/`` with session CSVs (no archive I/O)."""
    if not child.is_dir():
        return False
    sid = child.name
    if (child / sid / "input_map.csv").is_file():
        return True
    if (child / "input_map.csv").is_file():
        return True
    return False


_EXTRACTING_SUFFIX = "._extracting"


def _safe_extractall(tf: tarfile.TarFile, dest: Path) -> None:
    """Extract with ``filter="data"`` (Python 3.12+), falling back for older versions."""
    try:
        tf.extractall(path=dest, filter="data")
    except TypeError:
        tf.extractall(path=dest)


def _existing_session_root(base: Path, session_id: str) -> Path | None:
    dest = base / session_id
    for candidate in (dest / session_id, dest):
        if (candidate / "input_map.csv").is_file():
            return candidate
    return None


def _finalize_staged_extraction(
    base: Path,
    session_id: str,
    staging: Path,
) -> Path | None:
    dest = base / session_id
    try:
        if not dest.exists():
            staging.rename(dest)
        else:
            shutil.rmtree(staging, ignore_errors=True)
    except OSError:
        shutil.rmtree(staging, ignore_errors=True)
    return _existing_session_root(base, session_id)


def _ensure_zip_extracted(
    archive_path: Path,
    session_id: str,
    on_extracting: Callable[[str], None] | None = None,
) -> Path | None:
    """Extract a ZIP archive alongside it, returning the session root.

    Uses a staging directory for atomicity.  Skips extraction if the session
    directory already contains the expected files.  Returns ``None`` when
    extraction fails (the caller should fall back to streaming ZIP I/O).
    """
    base = archive_path.parent
    existing = _existing_session_root(base, session_id)
    if existing is not None:
        return existing

    staging = base / f"{session_id}{_EXTRACTING_SUFFIX}"
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)

    if on_extracting:
        on_extracting(session_id)

    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(staging)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        return None

    return _finalize_staged_extraction(base, session_id, staging)


def _ensure_tar_extracted(
    archive_path: Path,
    session_id: str,
    on_extracting: Callable[[str], None] | None = None,
) -> Path | None:
    """Extract a tar archive to a directory alongside it, returning the session root.

    Uses a staging directory for atomicity.  Skips extraction if the session
    directory already contains the expected files.  Returns ``None`` only when
    extraction fails (the caller should fall back to streaming tar I/O).
    """
    base = archive_path.parent
    existing = _existing_session_root(base, session_id)
    if existing is not None:
        return existing

    staging = base / f"{session_id}{_EXTRACTING_SUFFIX}"
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)

    if on_extracting:
        on_extracting(session_id)

    try:
        with tarfile.open(archive_path, "r:*") as tf:
            _safe_extractall(tf, staging)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        return None

    return _finalize_staged_extraction(base, session_id, staging)


def ensure_session_on_disk(
    session_id: str,
    base_dir: str | Path,
    *,
    on_extracting: Callable[[str], None] | None = None,
) -> Path | None:
    """Return an on-disk session root, extracting archives when needed."""
    source = resolve_session_source(session_id, base_dir, on_extracting=on_extracting)
    if source is None:
        return None
    if source.kind == "directory":
        return source.path
    if source.kind == "zip":
        return _ensure_zip_extracted(source.path, session_id, on_extracting)
    if source.kind == "tar":
        return _ensure_tar_extracted(source.path, session_id, on_extracting)
    return None


def resolve_session_source(
    session_id: str,
    base_dir: str | Path,
    *,
    on_extracting: Callable[[str], None] | None = None,
) -> SessionSource | None:
    """Find session data under *base_dir* (folder, zip, or tar archive).

    Preference: extracted directory over any archive with the same id.
    Remote URLs are copied into ``~/.scan-kit/remote-cache`` first.
    """
    spec = str(base_dir)
    if is_remote_location(spec):
        local = materialize_session(spec, session_id)
        if local is None:
            return None
        return resolve_session_source(
            session_id, local, on_extracting=on_extracting
        )
    base = Path(base_dir)
    inner = base / session_id / session_id
    if (inner / "input_map.csv").is_file():
        return SessionSource("directory", inner, session_id)
    flat = base / session_id
    if (flat / "input_map.csv").is_file():
        return SessionSource("directory", flat, session_id)

    zp = base / f"{session_id}.zip"
    if zp.is_file():
        extracted = _ensure_zip_extracted(zp, session_id, on_extracting)
        if extracted is not None:
            return SessionSource("directory", extracted, session_id)
        return SessionSource("zip", zp, session_id)

    for suf in (".tgz", ".tar.gz", ".tar.bz2", ".tar.xz", ".tar"):
        ap = base / f"{session_id}{suf}"
        if ap.is_file():
            extracted = _ensure_tar_extracted(ap, session_id, on_extracting)
            if extracted is not None:
                return SessionSource("directory", extracted, session_id)
            return SessionSource("tar", ap, session_id)
    return None


def load_session_csv(source: SessionSource, csv_name: str) -> pd.DataFrame | None:
    """Load ``csv_name`` from the session (paths inside archives use ``session_id/``)."""
    sid = source.session_id
    try:
        if source.kind == "directory":
            p = source.path / csv_name
            if not p.is_file():
                return None
            return _read_csv_robust(p)

        if source.kind == "zip":
            with zipfile.ZipFile(source.path, "r") as zf:
                with zf.open(f"{sid}/{csv_name}") as f:
                    return _read_csv_robust(f)

        if source.kind == "tar":
            with tarfile.open(source.path, "r:*") as tf:
                member = f"{sid}/{csv_name}"
                try:
                    info = tf.getmember(member)
                except KeyError:
                    return None
                raw = tf.extractfile(info)
                if raw is None:
                    return None
                return _read_csv_robust(raw)
    except Exception as e:
        _log.debug("Error loading %s from session %s: %s", csv_name, sid, e)
        return None


def read_session_csv_columns(source: SessionSource, csv_name: str) -> list[str] | None:
    """Return column names from a session CSV without reading row data."""
    sid = source.session_id
    try:
        if source.kind == "directory":
            p = source.path / csv_name
            if not p.is_file():
                return None
            return _read_csv_header_columns(p)

        if source.kind == "zip":
            with zipfile.ZipFile(source.path, "r") as zf:
                with zf.open(f"{sid}/{csv_name}") as f:
                    return _read_csv_header_columns(f)

        if source.kind == "tar":
            with tarfile.open(source.path, "r:*") as tf:
                member = f"{sid}/{csv_name}"
                try:
                    info = tf.getmember(member)
                except KeyError:
                    return None
                raw = tf.extractfile(info)
                if raw is None:
                    return None
                return _read_csv_header_columns(raw)
    except Exception as e:
        _log.debug(
            "Error reading header for %s from session %s: %s",
            csv_name,
            sid,
            e,
        )
        return None


def read_first_timeslice_columns(source: SessionSource) -> list[str] | None:
    """Return column names from the first timeslice frame without reading rows."""
    sid = source.session_id
    try:
        if source.kind == "directory":
            root = source.path
            for layer_dir in sorted(root.glob("layer-*")):
                if not layer_dir.is_dir():
                    continue
                for run_dir in layer_dir.glob("run-*"):
                    path = run_dir / "timeslice_data_device_units.csv"
                    if path.is_file():
                        return _read_csv_header_columns(path)
            return None

        if source.kind == "zip":
            with zipfile.ZipFile(source.path, "r") as zf:
                matches: list[tuple[int, str]] = []
                for entry in zf.namelist():
                    if not entry.startswith(f"{sid}/"):
                        continue
                    m = _TIMESLICE_RE.match(entry)
                    if m:
                        matches.append((int(m.group(1)), entry))
                if not matches:
                    return None
                matches.sort(key=lambda t: t[0])
                with zf.open(matches[0][1]) as f:
                    return _read_csv_header_columns(f)

        if source.kind == "tar":
            with tarfile.open(source.path, "r:*") as tf:
                matches: list[tuple[int, tarfile.TarInfo]] = []
                for info in tf.getmembers():
                    if not info.isfile():
                        continue
                    m = _TIMESLICE_RE.match(info.name)
                    if m:
                        matches.append((int(m.group(1)), info))
                if not matches:
                    return None
                matches.sort(key=lambda t: t[0])
                raw = tf.extractfile(matches[0][1])
                if raw is None:
                    return None
                return _read_csv_header_columns(raw)
    except Exception as e:
        _log.debug(
            "Error reading first timeslice header for session %s: %s",
            sid,
            e,
        )
        return None


def _point_time_column_name(columns: list[str]) -> str | None:
    for alias in _POINT_TIME_COLUMN_ALIASES:
        resolved = resolve_requested_column(columns, alias)
        if resolved is not None:
            return resolved
    for col in columns:
        if str(col).strip().lower() in _POINT_TIME_COLUMN_ALIASES:
            return str(col).strip()
    return None


def _pick_point_time_spot_member(run_members: list[str]) -> str | None:
    for preferred in _PREFERRED_POINT_TIME_SPOT_FILES:
        if preferred in run_members:
            return preferred
    for name in sorted(run_members):
        if name == "TX2_spot_data.csv":
            continue
        return name
    return None


def _read_point_time_spot_frame(
    source,
    *,
    sid: str | None = None,
    member: str | None = None,
) -> pd.DataFrame | None:
    try:
        if isinstance(source, (str, Path, os.PathLike)):
            df = _read_csv_robust(source)
        elif member is None:
            return None
        else:
            sid = sid or source.session_id
            if source.kind == "directory":
                df = _read_csv_robust(source.path / member)
            elif source.kind == "zip":
                with zipfile.ZipFile(source.path, "r") as zf:
                    with zf.open(f"{sid}/{member}") as handle:
                        df = _read_csv_robust(handle)
            elif source.kind == "tar":
                with tarfile.open(source.path, "r:*") as tf:
                    info = tf.getmember(f"{sid}/{member}")
                    raw = tf.extractfile(info)
                    if raw is None:
                        return None
                    df = _read_csv_robust(raw)
            else:
                return None
    except Exception:
        return None
    if df is None or df.empty:
        return None

    point_col = _point_time_column_name(df.columns.tolist())
    spot_col = resolve_requested_column(df.columns, "spot_no") or resolve_concept_column(
        df.columns, "spot_no",
    )
    layer_col = resolve_requested_column(df.columns, "layer_id") or resolve_concept_column(
        df.columns, "layer_id",
    )
    if point_col is None or spot_col is None or layer_col is None:
        return None

    out = pd.DataFrame(
        {
            "spot_no": pd.to_numeric(df[spot_col], errors="coerce"),
            "layer_id": pd.to_numeric(df[layer_col], errors="coerce"),
            "point_time_ms": pd.to_numeric(df[point_col], errors="coerce"),
        }
    )
    out = out.dropna(subset=["spot_no", "layer_id"])
    if out.empty:
        return None
    return out


def _pick_point_time_spot_path(run_dir: Path) -> Path | None:
    for name in _PREFERRED_POINT_TIME_SPOT_FILES:
        path = run_dir / name
        if path.is_file():
            cols = _read_csv_header_columns(path)
            if cols and _point_time_column_name(cols):
                return path
    for path in sorted(run_dir.glob("*_spot_data.csv")):
        if path.name == "TX2_spot_data.csv":
            continue
        cols = _read_csv_header_columns(path)
        if cols and _point_time_column_name(cols):
            return path
    return None


def load_session_point_time_table(source: SessionSource) -> pd.DataFrame | None:
    """Per-spot beam-on interval (ms) from layer run spot_data when absent from merged spot_data.csv."""
    sid = source.session_id
    parts: list[pd.DataFrame] = []
    try:
        if source.kind == "directory":
            root = source.path
            for layer_dir in sorted(root.glob("layer-*")):
                if not layer_dir.is_dir():
                    continue
                try:
                    layer_idx = int(layer_dir.name.split("-", 1)[1])
                except (IndexError, ValueError):
                    continue
                for run_dir in sorted(layer_dir.glob("run-*")):
                    spot_path = _pick_point_time_spot_path(run_dir)
                    if spot_path is None:
                        continue
                    frame = _read_point_time_spot_frame(spot_path)
                    if frame is not None:
                        parts.append(frame)
                    break

        elif source.kind == "zip":
            with zipfile.ZipFile(source.path, "r") as zf:
                by_layer: dict[int, dict[int, list[str]]] = {}
                for entry in zf.namelist():
                    if not entry.startswith(f"{sid}/"):
                        continue
                    m = _SPOT_LAYER_RE.match(entry)
                    if not m:
                        continue
                    layer_idx = int(m.group(1))
                    run_idx = int(m.group(2))
                    by_layer.setdefault(layer_idx, {}).setdefault(run_idx, []).append(
                        m.group(3),
                    )
                for layer_idx in sorted(by_layer):
                    run_members = by_layer[layer_idx].get(0, [])
                    member_name = _pick_point_time_spot_member(run_members)
                    if member_name is None:
                        continue
                    member = f"layer-{layer_idx}/run-0/{member_name}"
                    frame = _read_point_time_spot_frame(
                        source, sid=sid, member=member,
                    )
                    if frame is not None:
                        parts.append(frame)

        elif source.kind == "tar":
            with tarfile.open(source.path, "r:*") as tf:
                by_layer: dict[int, dict[int, list[str]]] = {}
                for info in tf.getmembers():
                    if not info.isfile():
                        continue
                    m = _SPOT_LAYER_RE.match(info.name)
                    if not m:
                        continue
                    layer_idx = int(m.group(1))
                    run_idx = int(m.group(2))
                    by_layer.setdefault(layer_idx, {}).setdefault(run_idx, []).append(
                        m.group(3),
                    )
                for layer_idx in sorted(by_layer):
                    run_members = by_layer[layer_idx].get(0, [])
                    member_name = _pick_point_time_spot_member(run_members)
                    if member_name is None:
                        continue
                    member = f"layer-{layer_idx}/run-0/{member_name}"
                    frame = _read_point_time_spot_frame(
                        source, sid=sid, member=member,
                    )
                    if frame is not None:
                        parts.append(frame)
    except Exception as e:
        _log.debug("Error loading point_time for session %s: %s", sid, e)
        return None

    if not parts:
        return None
    return pd.concat(parts, ignore_index=True)


def _timeslice_io_workers(n_files: int) -> int:
    return max(1, min(24, n_files, (os.cpu_count() or 4) * 3))


def load_session_timeslice_device_units(
    source: SessionSource,
    usecols: list[str] | None = None,
    *,
    max_frames: int | None = None,
    nrows: int | None = None,
) -> list[pd.DataFrame]:
    """Load per-layer timeslice_data_device_units CSVs.

    G2 IC current columns are automatically converted from coulombs to nA
    by the canonicalization step inside ``_read_csv_robust``.

    When *max_frames* is set, stop after that many layer files (for cheap probes).
    When *nrows* is set, only that many data rows are read from each file
    (``0`` is a header-only probe).
    """
    sid = source.session_id
    try:
        if source.kind == "directory":
            root = source.path
            matches: list[tuple[int, Path]] = []
            if max_frames == 1:
                first = root / "layer-0" / "run-0" / "timeslice_data_device_units.csv"
                if first.is_file():
                    matches = [(0, first)]
            if not matches:
                for layer_dir in sorted(root.glob("layer-*")):
                    if not layer_dir.is_dir():
                        continue
                    try:
                        layer_idx = int(layer_dir.name.split("-", 1)[1])
                    except (IndexError, ValueError):
                        continue
                    for run_dir in layer_dir.glob("run-*"):
                        p = run_dir / "timeslice_data_device_units.csv"
                        if p.is_file():
                            matches.append((layer_idx, p))
                            break
            matches.sort(key=lambda t: t[0])
            if max_frames is not None:
                matches = matches[:max_frames]
            if not matches:
                return []
            raw_usecols: list[str] | None = None
            if usecols is not None:
                raw_usecols = _resolve_timeslice_raw_usecols(matches[0][1], usecols)

            def _load_one(item: tuple[int, Path]) -> tuple[int, pd.DataFrame]:
                layer_idx, p = item
                df = _read_csv_robust(
                    p, usecols=usecols, raw_usecols=raw_usecols, nrows=nrows,
                )
                df["_layer_idx"] = layer_idx
                return layer_idx, df

            if len(matches) == 1:
                return [_load_one(matches[0])[1]]
            with ThreadPoolExecutor(max_workers=_timeslice_io_workers(len(matches))) as pool:
                loaded = list(pool.map(_load_one, matches))
            loaded.sort(key=lambda t: t[0])
            return [df for _, df in loaded]

        if source.kind == "zip":
            with zipfile.ZipFile(source.path, "r") as zf:
                return _timeslices_from_zip(
                    zf, sid, usecols, max_frames=max_frames, nrows=nrows,
                )

        if source.kind == "tar":
            with tarfile.open(source.path, "r:*") as tf:
                return _timeslices_from_tar(
                    tf, sid, usecols, max_frames=max_frames, nrows=nrows,
                )
    except Exception as e:
        _log.debug("Error loading timeslice data from session %s: %s", sid, e)
        return []


def _timeslices_from_zip(
    zf: zipfile.ZipFile,
    session_id: str,
    usecols: list[str] | None,
    *,
    max_frames: int | None = None,
    nrows: int | None = None,
) -> list[pd.DataFrame]:
    matches: list[tuple[int, str]] = []
    for entry in zf.namelist():
        if not entry.startswith(f"{session_id}/"):
            continue
        m = _TIMESLICE_RE.match(entry)
        if m:
            matches.append((int(m.group(1)), entry))
    matches.sort(key=lambda t: t[0])
    raw_usecols: list[str] | None = None
    frames = []
    for layer_idx, path in matches:
        with zf.open(path) as f:
            if usecols is not None and raw_usecols is None:
                raw_usecols = _resolve_timeslice_raw_usecols(f, usecols)
            df = _read_csv_robust(
                f, usecols=usecols, raw_usecols=raw_usecols, nrows=nrows,
            )
        df["_layer_idx"] = layer_idx
        frames.append(df)
        if max_frames is not None and len(frames) >= max_frames:
            break
    return frames


def _timeslices_from_tar(
    tf: tarfile.TarFile,
    session_id: str,
    usecols: list[str] | None,
    *,
    max_frames: int | None = None,
    nrows: int | None = None,
) -> list[pd.DataFrame]:
    matches: list[tuple[int, tarfile.TarInfo]] = []
    for info in tf.getmembers():
        if not info.isfile():
            continue
        m = _TIMESLICE_RE.match(info.name)
        if m:
            matches.append((int(m.group(1)), info))
    matches.sort(key=lambda t: t[0])
    raw_usecols: list[str] | None = None
    frames = []
    for layer_idx, info in matches:
        raw = tf.extractfile(info)
        if raw is None:
            continue
        if usecols is not None and raw_usecols is None:
            raw_usecols = _resolve_timeslice_raw_usecols(raw, usecols)
        df = _read_csv_robust(
            raw, usecols=usecols, raw_usecols=raw_usecols, nrows=nrows,
        )
        df["_layer_idx"] = layer_idx
        frames.append(df)
        if max_frames is not None and len(frames) >= max_frames:
            break
    return frames


def _tar_read_text(archive: Path, member: str) -> str | None:
    with tarfile.open(archive, "r:*") as tf:
        try:
            info = tf.getmember(member)
        except KeyError:
            return None
        raw = tf.extractfile(info)
        if raw is None:
            return None
        return raw.read().decode("utf-8", errors="replace")


def load_session_text(source: SessionSource, filename: str) -> str | None:
    """Read a text file from the session (directory, zip, or tar)."""
    sid = source.session_id
    try:
        if source.kind == "directory":
            p = source.path / filename
            if not p.is_file():
                return None
            return p.read_text(encoding="utf-8", errors="replace")

        if source.kind == "zip":
            with zipfile.ZipFile(source.path, "r") as zf:
                with zf.open(f"{sid}/{filename}") as f:
                    return f.read().decode("utf-8", errors="replace")

        if source.kind == "tar":
            return _tar_read_text(source.path, f"{sid}/{filename}")
    except Exception as e:
        _log.debug("Error loading %s from session %s: %s", filename, sid, e)
        return None
    return None


def load_session_termination_summary(source: SessionSource) -> SessionMeta | None:
    """Parse ``termination_summary.txt`` for TUI metadata."""
    text = load_session_text(source, "termination_summary.txt")
    if text is None:
        return None
    return parse_termination_summary_text(text)


def hydrate_session_metadata(
    snapshot: list[tuple[str, str, SessionMeta | None]],
    base_dir: str | Path,
    *,
    max_workers: int | None = None,
) -> list[tuple[str, str, SessionMeta | None]]:
    """Load ``termination_summary`` metadata for each row in *snapshot* (parallel I/O).

    Preserves order. Safe for large trees: uses a thread pool so zip/tar opens overlap.
    """
    if not snapshot:
        return []
    n = len(snapshot)
    if max_workers is None:
        # I/O-bound: oversubscribe modestly
        max_workers = max(4, min(24, n, (os.cpu_count() or 4) * 3))

    def _one(row: tuple[str, str, SessionMeta | None]) -> tuple[str, str, SessionMeta | None]:
        sid, path_str, _ = row
        storage: str | Path
        if is_remote_location(path_str):
            storage = path_str
        else:
            storage = Path(path_str)
            if not storage.is_absolute():
                storage = Path(base_dir) / path_str
        meta = load_termination_summary_cached(sid, storage)
        return (sid, path_str, meta)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(_one, snapshot))


def discover_session_entries(
    base_path: str | Path,
) -> list[tuple[str, str, SessionMeta | None]]:
    """List sessions under *base_path*: folders first, then archives not overridden.

    Returns ``(session_id, storage_path_for_display, meta)`` with *meta* always
    ``None`` here (fast scan). Call :func:`hydrate_session_metadata` or
    :func:`load_session_termination_summary` after :func:`resolve_session_source`
    to fill metadata for the TUI or scripts.
    """
    spec = str(base_path)
    if is_remote_location(spec):
        return _discover_remote_entries(spec)
    path = Path(base_path)
    seen: dict[str, tuple[Path, SessionMeta | None]] = {}

    # Single ``iterdir`` pass: unpacked dirs first (same sort order as before)
    try:
        children = sorted(path.iterdir(), key=lambda p: p.name)
    except OSError:
        return []

    for child in children:
        if child.is_dir() and _is_unpacked_session_directory(child):
            seen[child.name] = (child, None)

    for child in children:
        if not child.is_file():
            continue
        src = session_source_from_archive(child)
        if src is None:
            continue
        sid = src.session_id
        if sid not in seen:
            seen[sid] = (child, None)

    return sorted(
        ((sid, str(path_obj), meta) for sid, (path_obj, meta) in seen.items()),
        key=lambda t: t[0],
    )


def _discover_remote_entries(
    spec: str,
) -> list[tuple[str, str, SessionMeta | None]]:
    children = list_location_entries(spec)
    seen: dict[str, str] = {}
    for name, is_dir in children:
        if not is_dir:
            continue
        folder = join_location(spec, name)
        if _remote_is_unpacked_session(folder, name):
            seen[name] = folder
    for name, is_dir in children:
        if is_dir:
            continue
        stem = _strip_archive_suffix(name)
        if stem is None or stem in seen:
            continue
        seen[stem] = join_location(spec, name)
    return sorted(((sid, path, None) for sid, path in seen.items()), key=lambda t: t[0])


def _remote_is_unpacked_session(folder: str, session_id: str) -> bool:
    if location_is_file(join_location(folder, "input_map.csv")):
        return True
    return location_is_file(join_location(folder, f"{session_id}/input_map.csv"))


def _tar_open_mode(filename: str) -> str:
    lower = filename.lower()
    if lower.endswith((".tgz", ".tar.gz")):
        return "r:gz"
    if lower.endswith(".tar.bz2"):
        return "r:bz2"
    if lower.endswith(".tar.xz"):
        return "r:xz"
    return "r:"


def _remote_termination_summary_text(spec: str, session_id: str) -> str | None:
    if location_is_dir(spec):
        for rel in (
            f"{session_id}/termination_summary.txt",
            "termination_summary.txt",
        ):
            child = join_location(spec, rel)
            if location_is_file(child):
                return read_location_bytes(child).decode("utf-8", errors="replace")
        return None
    name = spec.rstrip("/").rsplit("/", 1)[-1].lower()
    member = f"{session_id}/termination_summary.txt"
    try:
        fh = open_location_binary(spec)
    except Exception:
        _log.debug("Could not open %s", spec, exc_info=True)
        return None
    try:
        if name.endswith(".zip"):
            with zipfile.ZipFile(fh) as zf:
                with zf.open(member) as raw:
                    return raw.read().decode("utf-8", errors="replace")
        if any(name.endswith(suf) for suf in _ARCHIVE_SUFFIXES):
            with tarfile.open(fileobj=fh, mode=_tar_open_mode(name)) as tf:
                try:
                    info = tf.getmember(member)
                except KeyError:
                    return None
                raw = tf.extractfile(info)
                if raw is None:
                    return None
                return raw.read().decode("utf-8", errors="replace")
    except Exception:
        _log.debug("Could not read termination_summary from %s", spec, exc_info=True)
        return None
    finally:
        fh.close()
    return None
