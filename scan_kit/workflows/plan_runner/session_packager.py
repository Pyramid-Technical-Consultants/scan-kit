"""Download RCI session folders and package G3-compatible scan-kit archives."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from ...igx.http import get_bytes
from .g3_session import build_g3_spot_data, write_session_info

# ponytail: probe layer/run until a streak of empty layers; upgrade when IGX lists dirs.
_MAX_LAYER_PROBE = 128
_MAX_RUN_PROBE = 4
_EMPTY_LAYER_STOP = 2

_SESSION_ROOT_FILES = (
    "input_map.csv",
    "spot_data.csv",
    "termination_summary.txt",
    "session_info.json",
    "session_meta.json",
    "SessionLogFile.log",
)

# Known G3 config files under the session tree (small; needed for strip→iso).
_SESSION_CONFIG_FILES = (
    "config/map2map/devices.xml",
    "config/map2map/Output.xml",
    "config/map2map/Input.xml",
    "config/map2map/Database.xml",
    "config/map2map/Report.xml",
    "config/map2map/tolerances.xml",
    "config/map2map/scan_dose_system.xml",
)

_LAYER_RUN_FILES = (
    "timeslice_data_device_units.csv",
    "timeslice_data.csv",
    "FX4_spot_data.csv",
    "IX256_1_spot_data.csv",
    "IX256_2_spot_data.csv",
    "RCI_spot_data.csv",
    "RCI_map.csv",
    "TX2_spot_data.csv",
)


def _try_download(host: str, remote: str, dest: Path) -> Path | None:
    try:
        data = get_bytes(host, remote)
    except Exception:
        return None
    if not data:
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return dest


def download_session_files(
    host: str,
    remote_session_path: str,
    dest_dir: Path,
) -> list[Path]:
    """Download known G3 session files from the device into *dest_dir*."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    base = remote_session_path.rstrip("/")
    downloaded: list[Path] = []

    for name in _SESSION_ROOT_FILES:
        path = _try_download(host, f"{base}/{name}", dest_dir / name)
        if path is not None:
            downloaded.append(path)

    for rel in _SESSION_CONFIG_FILES:
        path = _try_download(host, f"{base}/{rel}", dest_dir / rel)
        if path is not None:
            downloaded.append(path)

    empty_layers = 0
    for layer in range(_MAX_LAYER_PROBE):
        layer_hits = 0
        for run in range(_MAX_RUN_PROBE):
            run_hits = 0
            prefix = f"layer-{layer}/run-{run}"
            for name in _LAYER_RUN_FILES:
                path = _try_download(
                    host, f"{base}/{prefix}/{name}", dest_dir / prefix / name
                )
                if path is None:
                    continue
                downloaded.append(path)
                run_hits += 1
                layer_hits += 1
            if run_hits == 0 and run > 0:
                break
        if layer_hits == 0:
            empty_layers += 1
            if empty_layers >= _EMPTY_LAYER_STOP:
                break
        else:
            empty_layers = 0

    return downloaded


def package_session_zip(session_dir: Path, zip_path: Path) -> Path:
    """Zip a session directory as ``<session_id>/…`` for scan-kit discovery."""
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    session_id = session_dir.name

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(session_dir.rglob("*")):
            if not path.is_file():
                continue
            arcname = f"{session_id}/{path.relative_to(session_dir).as_posix()}"
            zf.write(path, arcname)
    return zip_path


def download_session_zip(
    host: str,
    remote_session_path: str,
    zip_path: Path,
) -> Path:
    """Download a device session and write a G3-layout zip analysis tools can open."""
    remote_session_path = remote_session_path.rstrip("/")
    session_name = Path(remote_session_path).name or "session"
    zip_path = Path(zip_path)
    staging_root = zip_path.parent / f".{session_name}_staging"
    session_dir = staging_root / session_name
    if staging_root.exists():
        shutil.rmtree(staging_root, ignore_errors=True)
    session_dir.mkdir(parents=True, exist_ok=True)

    try:
        downloaded = download_session_files(host, remote_session_path, session_dir)
        if not downloaded:
            raise FileNotFoundError(
                f"no session files found under {remote_session_path}"
            )
        build_g3_spot_data(session_dir)
        write_session_info(session_dir, session_name)
        return package_session_zip(session_dir, zip_path)
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
