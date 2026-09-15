"""Local, UNC, and fsspec session-library locations."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from scan_kit.common.data_location import (
    canonical_location,
    is_remote_location,
    is_uri,
    is_usable_data_location,
    join_location,
    list_location_entries,
    local_path,
    location_exists,
    materialize_session,
    remove_location,
)
from scan_kit.common.recycle import move_to_trash
from scan_kit.common.session_source import (
    clear_termination_summary_cache,
    hydrate_session_metadata,
    list_session_storage_paths,
    load_termination_summary_cached,
    resolve_session_source,
)
from scan_kit.common.sessions import discover_sessions
from scan_kit.common.user_store import snapshot_library

_SUMMARY = """\
Date: Thu Sep 10 21:07:41 2026
Primary total dose: 12.5 MU
Treatment time: 10 seconds
Room number: 2
Configuration name: facility_map2map
"""


def test_drive_letter_is_not_a_url() -> None:
    assert not is_uri("C:/data/ptc_ex")
    assert not is_remote_location("C:/data/ptc_ex")
    assert not is_remote_location(r"C:\data\ptc_ex")


def test_unc_is_local() -> None:
    unc = r"\\192.168.101.206\share\ptc_ex"
    assert not is_uri(unc)
    assert not is_remote_location(unc)


def test_sftp_url_is_remote_without_network() -> None:
    url = "sftp://pyramid@192.168.101.206/var/log/ptc_ex"
    assert is_uri(url)
    assert is_remote_location(url)
    assert is_usable_data_location(url)
    assert not Path(url).is_dir()
    assert is_remote_location("ssh://pyramid@192.168.101.206/var/log/ptc_ex")
    assert is_remote_location("smb://filer/share/ptc_ex")
    assert is_remote_location("ftp://host/pub/sessions")
    assert is_usable_data_location("smb://user@filer/data")


def test_canonical_strips_password_and_aliases_ssh() -> None:
    raw = "sftp://pyramid:secret@192.168.101.206/var/log/ptc_ex/"
    assert canonical_location(raw) == "sftp://pyramid@192.168.101.206/var/log/ptc_ex"
    assert canonical_location("ssh://pyramid@host/data") == "sftp://pyramid@host/data"
    assert "secret" not in canonical_location(raw)


def test_join_location_keeps_url_slashes() -> None:
    base = "sftp://pyramid@192.168.101.206/var/log/ptc_ex"
    assert join_location(base, "99") == f"{base}/99"
    assert join_location(base, "99.zip") == f"{base}/99.zip"


def test_file_uri_round_trips_to_local_dir(tmp_path: Path) -> None:
    uri = tmp_path.as_uri()
    assert not is_remote_location(uri)
    got = local_path(uri)
    assert got is not None
    assert got.resolve() == tmp_path.resolve()
    assert is_usable_data_location(uri)


def test_is_usable_rejects_missing_local(tmp_path: Path) -> None:
    assert not is_usable_data_location(tmp_path / "missing")
    assert not is_usable_data_location("")


def _reset_memory_fs() -> None:
    from fsspec.implementations.memory import MemoryFileSystem

    MemoryFileSystem.store.clear()
    dirs = getattr(MemoryFileSystem, "pseudo_dirs", None)
    if dirs is not None:
        dirs.clear()


def _memory_lib(sid: str = "99") -> str:
    import fsspec

    _reset_memory_fs()
    fs = fsspec.filesystem("memory")
    fs.makedirs(f"ptc_ex/{sid}", exist_ok=True)
    with fs.open(f"ptc_ex/{sid}/input_map.csv", "wb") as fh:
        fh.write(b"a,b\n1,2\n")
    with fs.open(f"ptc_ex/{sid}/termination_summary.txt", "wb") as fh:
        fh.write(_SUMMARY.encode("utf-8"))
    return "memory:///ptc_ex"


def test_memory_url_lists_and_hydrates() -> None:
    pytest.importorskip("fsspec")
    clear_termination_summary_cache()
    url = _memory_lib("4242")
    names = {name for name, _ in list_location_entries(url)}
    assert "4242" in names
    rows = discover_sessions(base_dirs=(url,))
    assert [sid for sid, _, _ in rows] == ["4242"]
    storage = rows[0][1]
    assert storage.startswith("memory://")
    hydrated = hydrate_session_metadata(rows, url)
    meta = hydrated[0][2]
    assert meta is not None
    assert meta.config_name == "facility_map2map"
    assert meta.primary_mu == 12.5
    cached = load_termination_summary_cached("4242", storage)
    assert cached is not None
    assert cached.room_number == 2


def test_memory_zip_hydrates_without_copy() -> None:
    pytest.importorskip("fsspec")
    import fsspec

    clear_termination_summary_cache()
    _reset_memory_fs()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("88/input_map.csv", "a,b\n1,2\n")
        zf.writestr("88/termination_summary.txt", _SUMMARY)
    fs = fsspec.filesystem("memory")
    fs.makedirs("ptc_ex", exist_ok=True)
    with fs.open("ptc_ex/88.zip", "wb") as fh:
        fh.write(buf.getvalue())
    url = "memory:///ptc_ex"
    rows = discover_sessions(base_dirs=(url,))
    assert rows[0][0] == "88"
    meta = load_termination_summary_cached("88", rows[0][1])
    assert meta is not None
    assert meta.short_config == "facility_map2map"


def test_resolve_materializes_into_local_cache() -> None:
    pytest.importorskip("fsspec")
    url = _memory_lib("77")
    src = resolve_session_source("77", url)
    assert src is not None
    assert src.kind == "directory"
    assert src.path.is_dir()
    assert (src.path / "input_map.csv").is_file()
    assert "remote-cache" in src.path.parts
    local = materialize_session(url, "77")
    assert local is not None
    again = resolve_session_source("77", url)
    assert again is not None
    assert again.path == src.path


def test_memory_list_and_delete_session() -> None:
    pytest.importorskip("fsspec")
    url = _memory_lib("1010")
    paths = list_session_storage_paths(url, "1010")
    assert len(paths) == 1
    assert paths[0].rstrip("/").endswith("1010")
    move_to_trash(paths[0])
    assert not location_exists(paths[0])
    assert list_session_storage_paths(url, "1010") == []
    with pytest.raises(ValueError):
        remove_location(Path.cwd())


def test_snapshot_library_indexes_memory_url() -> None:
    pytest.importorskip("fsspec")
    url = _memory_lib("333")
    notes, selected, rows = snapshot_library(url)
    assert selected == []
    assert notes == {}
    assert len(rows) == 1
    assert rows[0][0] == "333"
    assert rows[0][1].startswith("memory://")
