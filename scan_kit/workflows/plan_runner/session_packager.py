"""Download RCI session folders and package scan-kit-compatible archives."""

from __future__ import annotations

import zipfile
from pathlib import Path

from ...igx.http import get_bytes

# ponytail: naive layer/run probe (0..max_layer, 0..max_run); upgrade when IGX exposes dir listing.
_MAX_LAYER_PROBE = 32
_MAX_RUN_PROBE = 8

_SESSION_ROOT_FILES = (
    "input_map.csv",
    "termination_summary.txt",
    "session_meta.json",
)

_LAYER_RUN_FILES = (
    "timeslice_data_device_units.csv",
    "FX4_spot_data.csv",
    "IX256_1_spot_data.csv",
    "IX256_2_spot_data.csv",
    "RCI_spot_data.csv",
)


def _session_relative_paths() -> list[str]:
    paths = list(_SESSION_ROOT_FILES)
    for layer in range(_MAX_LAYER_PROBE):
        for run in range(_MAX_RUN_PROBE):
            prefix = f"layer-{layer}/run-{run}"
            for name in _LAYER_RUN_FILES:
                paths.append(f"{prefix}/{name}")
    return paths


def download_session_files(
    host: str,
    remote_session_path: str,
    dest_dir: Path,
) -> list[Path]:
    """Download known session files from the device into *dest_dir*.

    *remote_session_path* is the session folder on the device
    (e.g. ``/root/reports/session/my_session_id``).
    Returns paths of files that were downloaded successfully.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    base = remote_session_path.rstrip("/")
    downloaded: list[Path] = []

    for rel in _session_relative_paths():
        remote = f"{base}/{rel}"
        try:
            data = get_bytes(host, remote)
        except Exception:
            continue
        if not data:
            continue
        local = dest_dir / rel
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(data)
        downloaded.append(local)

    return downloaded


def package_session_zip(session_dir: Path, zip_path: Path) -> Path:
    """Zip a local session directory tree for scan-kit session discovery."""
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
    """Download session files from device and write a scan-kit-compatible zip."""
    remote_session_path = remote_session_path.rstrip("/")
    session_name = Path(remote_session_path).name or "session"
    staging = Path(zip_path).parent / f".{session_name}_staging"
    if staging.exists():
        import shutil

        shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)

    downloaded = download_session_files(host, remote_session_path, staging)
    if not downloaded:
        raise FileNotFoundError(
            f"no session files found under {remote_session_path}"
        )

    return package_session_zip(staging, zip_path)
