"""Resolve session data from directories, ZIP, or tar-based archives (tgz, tar.gz, …)."""

from __future__ import annotations

import os
import re
import shutil
import tarfile
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

_log = logging.getLogger(__name__)

_TIMESLICE_RE = re.compile(
    r"^[^/]+/layer-(\d+)/run-(\d+)/timeslice_data_device_units\.csv$"
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

    df = pd.read_csv(
        source,
        index_col=False,
        skipinitialspace=True,
        usecols=read_usecols,
    )
    df = canonicalize_dataframe_columns(df, aliases=alias_map)
    if usecols is None:
        return df
    keep = [c for c in usecols if c in df.columns]
    if not keep:
        return df.iloc[:, 0:0]
    return df[keep]


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
    """
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


def load_session_timeslice_device_units(
    source: SessionSource,
    usecols: list[str] | None = None,
) -> list[pd.DataFrame]:
    """Load all per-layer timeslice_data_device_units CSVs.

    G2 IC current columns are automatically converted from coulombs to nA
    by the canonicalization step inside ``_read_csv_robust``.
    """
    sid = source.session_id
    try:
        if source.kind == "directory":
            root = source.path
            matches: list[tuple[int, Path]] = []
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
            raw_usecols: list[str] | None = None
            frames = []
            for layer_idx, p in matches:
                if usecols is not None and raw_usecols is None:
                    raw_usecols = _resolve_timeslice_raw_usecols(p, usecols)
                df = _read_csv_robust(p, usecols=usecols, raw_usecols=raw_usecols)
                df["_layer_idx"] = layer_idx
                frames.append(df)
            return frames

        if source.kind == "zip":
            with zipfile.ZipFile(source.path, "r") as zf:
                return _timeslices_from_zip(zf, sid, usecols)

        if source.kind == "tar":
            with tarfile.open(source.path, "r:*") as tf:
                return _timeslices_from_tar(tf, sid, usecols)
    except Exception as e:
        _log.debug("Error loading timeslice data from session %s: %s", sid, e)
        return []


def _timeslices_from_zip(
    zf: zipfile.ZipFile, session_id: str, usecols: list[str] | None
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
            df = _read_csv_robust(f, usecols=usecols, raw_usecols=raw_usecols)
        df["_layer_idx"] = layer_idx
        frames.append(df)
    return frames


def _timeslices_from_tar(
    tf: tarfile.TarFile, session_id: str, usecols: list[str] | None
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
        df = _read_csv_robust(raw, usecols=usecols, raw_usecols=raw_usecols)
        df["_layer_idx"] = layer_idx
        frames.append(df)
    return frames


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
            target = f"{sid}/{filename}"
            with tarfile.open(source.path, "r:*") as tf:
                for info in tf:
                    if info.name == target:
                        raw = tf.extractfile(info)
                        if raw is None:
                            return None
                        return raw.read().decode("utf-8", errors="replace")
            return None
    except Exception as e:
        _log.debug("Error loading %s from session %s: %s", filename, sid, e)
        return None


def load_session_termination_summary(source: SessionSource) -> SessionMeta | None:
    """Parse ``termination_summary.txt`` for TUI metadata."""
    sid = source.session_id
    try:
        if source.kind == "directory":
            p = source.path / "termination_summary.txt"
            if not p.is_file():
                return None
            text = p.read_text(encoding="utf-8", errors="replace")
            return parse_termination_summary_text(text)

        if source.kind == "zip":
            with zipfile.ZipFile(source.path, "r") as zf:
                with zf.open(f"{sid}/termination_summary.txt") as f:
                    text = f.read().decode("utf-8", errors="replace")
            return parse_termination_summary_text(text)

        if source.kind == "tar":
            target = f"{sid}/termination_summary.txt"
            with tarfile.open(source.path, "r:*") as tf:
                for info in tf:
                    if info.name == target:
                        raw = tf.extractfile(info)
                        if raw is None:
                            return None
                        text = raw.read().decode("utf-8", errors="replace")
                        return parse_termination_summary_text(text)
            return None
    except Exception:
        return None


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
    base = Path(base_dir)
    n = len(snapshot)
    if max_workers is None:
        # I/O-bound: oversubscribe modestly
        max_workers = max(4, min(24, n, (os.cpu_count() or 4) * 3))

    def _one(row: tuple[str, str, SessionMeta | None]) -> tuple[str, str, SessionMeta | None]:
        sid, path_str, _ = row
        src = resolve_session_source(sid, base)
        meta = load_session_termination_summary(src) if src else None
        return (sid, path_str, meta)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(_one, snapshot))


def discover_session_entries(base_path: Path) -> list[tuple[str, str, SessionMeta | None]]:
    """List sessions under *base_path*: folders first, then archives not overridden.

    Returns ``(session_id, storage_path_for_display, meta)`` with *meta* always
    ``None`` here (fast scan). Call :func:`hydrate_session_metadata` or
    :func:`load_session_termination_summary` after :func:`resolve_session_source`
    to fill metadata for the TUI or scripts.
    """
    seen: dict[str, tuple[Path, SessionMeta | None]] = {}

    # Single ``iterdir`` pass: unpacked dirs first (same sort order as before)
    try:
        children = sorted(base_path.iterdir(), key=lambda p: p.name)
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
