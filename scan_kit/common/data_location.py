"""Session-library locations: local folders, UNC shares, and fsspec URLs.

Local paths (including Windows ``\\\\server\\share`` UNC) stay ``pathlib.Path``.
URLs such as ``sftp://user@host/var/log/ptc_ex`` go through fsspec so SFTP,
SMB, FTP, HTTP, and similar protocols work without a custom client. Analysis
still reads local files: a session is copied into ``~/.scan-kit/remote-cache``
the first time it is opened.
"""

from __future__ import annotations

import hashlib
import logging
import posixpath
import re
import shutil
import threading
from collections.abc import Callable
from pathlib import Path
from urllib.parse import unquote, urlparse, urlunparse

_log = logging.getLogger(__name__)

_URI_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")
_SCHEME_ALIASES = {"ssh": "sftp", "scp": "sftp"}
_ARCHIVE_SUFFIXES: tuple[str, ...] = (
    ".tar.gz",
    ".tar.bz2",
    ".tar.xz",
    ".tgz",
    ".tar",
    ".zip",
)
_CONNECT_TIMEOUT_S = 20
_PASSWORD_SCHEMES = frozenset({"sftp", "smb", "ftp", "ftps"})
_TIMEOUT_SCHEMES = frozenset({"sftp", "smb", "ftp", "ftps"})
_SSH_SCHEMES = frozenset({"sftp"})
_AUTH_NEEDLES = (
    "authentication",
    "auth fail",
    "autherror",
    "password required",
    "password",
    "permission denied",
    "login incorrect",
    "not authenticated",
    "unauthorized",
    "credentials",
    "access denied",
    "no existing session",
)

# In-memory only — never written to sqlite or prefs.
_passwords: dict[str, str] = {}
_passwords_lock = threading.Lock()
_fs_cache_lock = threading.Lock()


class _FsEntry:
    __slots__ = ("fs", "lock")

    def __init__(self, fs: object) -> None:
        self.fs = fs
        self.lock = threading.RLock()


_fs_cache: dict[tuple[str, str, int, str, str], _FsEntry] = {}


class _LockedBinary:
    """Hold the per-connection lock until the file object is closed."""

    def __init__(self, fh, lock: threading.RLock) -> None:
        self._fh = fh
        self._lock = lock
        self._held = True

    def close(self) -> None:
        try:
            self._fh.close()
        finally:
            if self._held:
                self._held = False
                self._lock.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __iter__(self):
        return iter(self._fh)

    def __getattr__(self, name: str):
        return getattr(self._fh, name)


def strip_location(spec: str | Path) -> str:
    return str(spec).strip()


def is_uri(spec: str | Path) -> bool:
    return bool(_URI_RE.match(strip_location(spec)))


def is_remote_location(spec: str | Path) -> bool:
    """True for fsspec URLs. UNC, drive letters, and ``file://`` stay local."""
    text = strip_location(spec)
    if not is_uri(text):
        return False
    scheme = urlparse(text).scheme.lower()
    # ``C://...`` would match the URI regex; a one-letter scheme is a drive.
    return scheme != "file" and len(scheme) > 1


def local_path(spec: str | Path) -> Path | None:
    """``Path`` for a local spec, or None when *spec* is a remote URL."""
    text = strip_location(spec)
    if not text or is_remote_location(text):
        return None
    if is_uri(text) and urlparse(text).scheme.lower() == "file":
        return _file_uri_to_path(text)
    return Path(text).expanduser()


def is_usable_data_location(spec: str | Path) -> bool:
    """Whether *spec* can be restored as the last session library.

    Remote URLs are accepted without a network round-trip. Local paths must
    already be directories.
    """
    text = strip_location(spec)
    if not text:
        return False
    if is_remote_location(text):
        return True
    path = local_path(text)
    if path is None:
        return False
    try:
        return path.is_dir()
    except OSError:
        return False


def canonical_location(spec: str | Path) -> str:
    """Stable identity for prefs/sqlite. Passwords are stripped from URLs."""
    text = strip_location(spec)
    if not text:
        return text
    if is_uri(text):
        parsed = urlparse(text)
        scheme = parsed.scheme.lower()
        if scheme == "file" or len(scheme) == 1:
            local = local_path(text)
            if local is not None:
                return _canonical_local(local)
            return text
        scheme = _SCHEME_ALIASES.get(scheme, scheme)
        host = parsed.hostname or ""
        if host.count(":") and not host.startswith("["):
            host = f"[{host}]"
        if parsed.port:
            host = f"{host}:{parsed.port}"
        netloc = host
        if parsed.username:
            netloc = f"{unquote(parsed.username)}@{host}"
        path = parsed.path or "/"
        if path != "/":
            path = path.rstrip("/")
        return urlunparse((scheme, netloc, path, "", parsed.query, ""))
    return _canonical_local(Path(text).expanduser())


def join_location(base: str | Path, name: str) -> str:
    """Join a child name onto a local path or URL."""
    child = str(name).lstrip("/").replace("\\", "/")
    text = strip_location(base)
    if is_remote_location(text):
        return text.rstrip("/") + "/" + child
    path = local_path(text)
    if path is None:
        return text.rstrip("/") + "/" + child
    return str(path / name)


def location_uses_password(spec: str | Path) -> bool:
    """True when a GUI password prompt can help this URL."""
    if not is_remote_location(spec):
        return False
    return _scheme_of(spec) in _PASSWORD_SCHEMES


def remember_password(spec: str | Path, password: str) -> None:
    """Keep *password* in process memory for *spec* (never sqlite/prefs)."""
    key = canonical_location(spec)
    with _passwords_lock:
        _passwords[key] = password
    drop_cached_fs(spec)


def forget_passwords(spec: str | Path | None = None) -> None:
    with _passwords_lock:
        if spec is None:
            _passwords.clear()
        else:
            _passwords.pop(canonical_location(spec), None)
    drop_cached_fs(spec)


def password_for(spec: str | Path) -> str | None:
    text = strip_location(spec)
    parsed = urlparse(_fsspec_url(text))
    if parsed.password:
        return unquote(parsed.password)
    with _passwords_lock:
        return _passwords.get(canonical_location(text))


def drop_cached_fs(spec: str | Path | None = None) -> None:
    """Drop reused fsspec clients (all, or those matching *spec*)."""
    with _fs_cache_lock:
        if spec is None:
            _fs_cache.clear()
            return
        identity = _fs_identity(spec)
        for key in [k for k in _fs_cache if k[:4] == identity]:
            _fs_cache.pop(key, None)


def is_auth_error(exc: BaseException) -> bool:
    """True when *exc* looks like a failed login rather than a missing path."""
    parts: list[str] = []
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        parts.append(type(cur).__name__)
        parts.append(str(cur))
        nxt = cur.__cause__ or cur.__context__
        cur = nxt if nxt is not cur else None
    blob = " ".join(parts).lower()
    if "timed out" in blob or "timeout" in blob:
        if "authentication" not in blob and "password" not in blob:
            return False
    return any(needle in blob for needle in _AUTH_NEEDLES)


def format_location_error(spec: str | Path, exc: BaseException) -> str:
    """One-line status text; URL passwords are stripped."""
    where = canonical_location(spec) or strip_location(spec)
    msg = " ".join(str(exc).split()) or type(exc).__name__
    if len(msg) > 180:
        msg = msg[:177] + "..."
    return f"Could not list {where}: {msg}"


def open_fs(spec: str | Path):
    """Return ``(fs, path)`` for *spec*, reusing one client per host/user."""
    fs, path, _lock = _cached_fs(spec)
    return fs, path


def list_location_entries(spec: str | Path) -> list[tuple[str, bool]]:
    """``(name, is_dir)`` children of *spec*. Empty on local errors; remote errors raise."""
    text = strip_location(spec)
    if not is_remote_location(text):
        folder = local_path(text)
        if folder is None:
            return []
        try:
            children = sorted(folder.iterdir(), key=lambda p: p.name)
        except OSError:
            return []
        return [(child.name, child.is_dir()) for child in children]
    try:
        fs, path, lock = _cached_fs(text)
        with lock:
            listing = fs.ls(path, detail=True)
    except Exception as exc:
        _log.exception("Could not list %s", canonical_location(text))
        if is_auth_error(exc):
            drop_cached_fs(text)
        raise
    out: list[tuple[str, bool]] = []
    for item in listing:
        if isinstance(item, str):
            name = posixpath.basename(item.rstrip("/"))
            is_dir = False
        else:
            raw_name = str(item.get("name") or "")
            name = posixpath.basename(raw_name.rstrip("/"))
            kind = str(item.get("type") or "file").lower()
            is_dir = kind in {"directory", "dir"}
        if name in {"", ".", ".."}:
            continue
        out.append((name, is_dir))
    out.sort(key=lambda row: row[0])
    return out


def location_exists(spec: str | Path) -> bool:
    text = strip_location(spec)
    if not is_remote_location(text):
        path = local_path(text)
        if path is None:
            return False
        try:
            return path.exists()
        except OSError:
            return False
    try:
        fs, path, lock = _cached_fs(text)
        with lock:
            return bool(fs.exists(path))
    except Exception:
        return False


def location_is_dir(spec: str | Path) -> bool:
    text = strip_location(spec)
    if not is_remote_location(text):
        path = local_path(text)
        if path is None:
            return False
        try:
            return path.is_dir()
        except OSError:
            return False
    try:
        fs, path, lock = _cached_fs(text)
        with lock:
            return bool(fs.isdir(path))
    except Exception:
        return False


def location_is_file(spec: str | Path) -> bool:
    text = strip_location(spec)
    if not is_remote_location(text):
        path = local_path(text)
        if path is None:
            return False
        try:
            return path.is_file()
        except OSError:
            return False
    try:
        fs, path, lock = _cached_fs(text)
        with lock:
            return bool(fs.isfile(path))
    except Exception:
        return False


def read_location_bytes(spec: str | Path) -> bytes:
    text = strip_location(spec)
    if not is_remote_location(text):
        path = local_path(text)
        if path is None:
            raise FileNotFoundError(text)
        return path.read_bytes()
    fs, path, lock = _cached_fs(text)
    with lock:
        data = fs.cat_file(path)
    return data if isinstance(data, bytes) else bytes(data)


def open_location_binary(spec: str | Path):
    """Readable binary file object (caller closes)."""
    text = strip_location(spec)
    if not is_remote_location(text):
        path = local_path(text)
        if path is None:
            raise FileNotFoundError(text)
        return path.open("rb")
    fs, path, lock = _cached_fs(text)
    lock.acquire()
    try:
        fh = fs.open(path, "rb")
    except Exception:
        lock.release()
        raise
    return _LockedBinary(fh, lock)


def location_stat(spec: str | Path) -> tuple[int, int] | None:
    """``(mtime_ns, size)`` or None."""
    text = strip_location(spec)
    if not is_remote_location(text):
        path = local_path(text)
        if path is None:
            return None
        try:
            st = path.stat()
            return (int(st.st_mtime_ns), int(st.st_size))
        except OSError:
            return None
    try:
        fs, path, lock = _cached_fs(text)
        with lock:
            info = fs.info(path)
    except Exception:
        return None
    size = int(info.get("size") or 0)
    mtime = info.get("mtime") or info.get("updated") or 0
    try:
        mtime_ns = int(float(mtime) * 1_000_000_000)
    except (TypeError, ValueError):
        mtime_ns = 0
    return mtime_ns, size


def remove_location(spec: str | Path) -> None:
    """Delete a remote object via fsspec."""
    text = strip_location(spec)
    if not is_remote_location(text):
        raise ValueError(f"remove_location is for remote URLs, got {text!r}")
    fs, path, lock = _cached_fs(text)
    with lock:
        fs.rm(path, recursive=True)


def materialize_session(
    base_spec: str | Path,
    session_id: str,
    *,
    on_extracting: Callable[[str], None] | None = None,
) -> Path | None:
    """Copy one session into the local cache and return the cache library folder."""
    spec = strip_location(base_spec)
    if not is_remote_location(spec):
        path = local_path(spec)
        return path if path is not None and path.is_dir() else None
    local_lib = cache_root_for(spec)
    local_lib.mkdir(parents=True, exist_ok=True)
    names = [session_id, *(f"{session_id}{suf}" for suf in _ARCHIVE_SUFFIXES)]
    fs, rpath, lock = _cached_fs(spec)
    root = (rpath or "").rstrip("/")
    with lock:
        for name in names:
            remote = posixpath.join(root, name) if root else name
            try:
                exists = fs.exists(remote)
            except Exception:
                exists = False
            if not exists:
                continue
            dest = local_lib / name
            stamp = local_lib / f".{name}.stamp"
            stamp_text = _remote_stamp(fs, remote)
            if (
                dest.exists()
                and stamp.is_file()
                and stamp.read_text(encoding="utf-8") == stamp_text
            ):
                return local_lib
            if on_extracting:
                on_extracting(session_id)
            _log.info(
                "Copying session %s from %s",
                session_id,
                canonical_location(spec),
            )
            if dest.exists():
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            dest.parent.mkdir(parents=True, exist_ok=True)
            recursive = bool(fs.isdir(remote))
            fs.get(remote, str(dest), recursive=recursive)
            stamp.write_text(stamp_text, encoding="utf-8")
            return local_lib
    return None


def cache_root_for(spec: str | Path) -> Path:
    key = canonical_location(spec)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return _remote_cache_home() / digest


def remote_cache_dir() -> Path:
    return _remote_cache_home()


def remote_cache_size_bytes() -> int:
    root = _remote_cache_home()
    if not root.is_dir():
        return 0
    total = 0
    for path in root.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except OSError:
            continue
    return total


def format_byte_size(n: int) -> str:
    size = float(max(0, int(n)))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            if unit == "B":
                return f"{int(size)} B"
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def clear_remote_cache() -> None:
    """Delete ``~/.scan-kit/remote-cache`` and drop live remote clients."""
    root = _remote_cache_home()
    shutil.rmtree(root, ignore_errors=True)
    drop_cached_fs()


def _remote_cache_home() -> Path:
    from .user_store import user_data_dir

    return user_data_dir() / "remote-cache"


def _remote_stamp(fs, remote: str) -> str:
    try:
        info = fs.info(remote)
    except Exception:
        return "0\n0"
    mtime = info.get("mtime") or info.get("updated") or 0
    size = info.get("size") or 0
    return f"{mtime}\n{size}"


def _canonical_local(path: Path) -> str:
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


def _file_uri_to_path(spec: str) -> Path | None:
    parsed = urlparse(spec)
    if parsed.scheme.lower() != "file":
        return None
    path = unquote(parsed.path or "")
    if parsed.netloc and parsed.netloc not in {"localhost", "127.0.0.1"}:
        return Path(f"//{parsed.netloc}{path}")
    if os_name_is_windows() and re.match(r"^/[A-Za-z]:", path):
        path = path[1:]
    return Path(path) if path else None


def os_name_is_windows() -> bool:
    import sys

    return sys.platform == "win32"


def _fsspec_url(spec: str | Path) -> str:
    text = strip_location(spec)
    parsed = urlparse(text)
    scheme = parsed.scheme.lower()
    alias = _SCHEME_ALIASES.get(scheme)
    if alias is None:
        return text
    return urlunparse(
        (alias, parsed.netloc, parsed.path, parsed.params, parsed.query, parsed.fragment)
    )


def _scheme_of(spec: str | Path) -> str:
    parsed = urlparse(_fsspec_url(spec))
    return _SCHEME_ALIASES.get(parsed.scheme.lower(), parsed.scheme.lower())


def _fs_identity(spec: str | Path) -> tuple[str, str, int, str]:
    parsed = urlparse(_fsspec_url(spec))
    scheme = _SCHEME_ALIASES.get(parsed.scheme.lower(), parsed.scheme.lower())
    host = (parsed.hostname or "").lower()
    port = int(parsed.port or 0)
    user = unquote(parsed.username or "")
    return (scheme, host, port, user)


def _fs_cache_key(spec: str | Path) -> tuple[str, str, int, str, str]:
    pw = password_for(spec) or ""
    token = hashlib.sha256(pw.encode("utf-8")).hexdigest()[:12] if pw else ""
    return _fs_identity(spec) + (token,)


def _connect_kwargs(spec: str | Path) -> dict:
    parsed = urlparse(_fsspec_url(spec))
    scheme = _SCHEME_ALIASES.get(parsed.scheme.lower(), parsed.scheme.lower())
    kw: dict = {}
    pw = password_for(spec)
    if pw and not parsed.password:
        kw["password"] = pw
    if scheme in _TIMEOUT_SCHEMES:
        kw["timeout"] = _CONNECT_TIMEOUT_S
    if scheme in _SSH_SCHEMES:
        kw["banner_timeout"] = _CONNECT_TIMEOUT_S
        kw["auth_timeout"] = _CONNECT_TIMEOUT_S
    return kw


def _cached_fs(spec: str | Path):
    from fsspec.core import url_to_fs

    url = _fsspec_url(spec)
    key = _fs_cache_key(spec)
    with _fs_cache_lock:
        entry = _fs_cache.get(key)
        if entry is not None:
            path = entry.fs._strip_protocol(url)
            return entry.fs, path, entry.lock
        kwargs = _connect_kwargs(spec)
        try:
            fs, path = url_to_fs(url, **kwargs)
        except TypeError:
            kwargs = {k: v for k, v in kwargs.items() if k == "password"}
            fs, path = url_to_fs(url, **kwargs)
        entry = _FsEntry(fs)
        _fs_cache[key] = entry
        return fs, path, entry.lock
