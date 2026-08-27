"""Format and gate Plan Runner live status for the operator UI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...igx.rci_paths import (
    COMBINED_POINTS_OK,
    CONTROL_POINT_COUNT,
    CONTROL_POINT_INDEX,
    POINTS_VALID,
    READY_PERMIT,
    READY_PERMIT_REASON,
    STATE,
)


def unwrap_io(value: Any) -> Any:
    """Unwrap IGX ``[value, timestamp]`` pairs (and one extra nest)."""
    if isinstance(value, (list, tuple)) and value:
        return unwrap_io(value[0])
    return value


def io_bool(value: Any) -> bool:
    """Coerce an IGX digital/permit value to bool (unwrap [value, time] pairs)."""
    value = unwrap_io(value)
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "granted"}
    return False


def io_number(value: Any) -> float | None:
    """Unwrap and parse a numeric IO value."""
    raw = unwrap_io(value)
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def io_text(value: Any) -> str:
    """Unwrap an IO value to a display string; empty if missing."""
    raw = unwrap_io(value)
    if raw is None:
        return ""
    return str(raw).strip()


def progress_percent(value: Any) -> float | None:
    """ProgressIO is 0..1; treat values above 1 as already-percent."""
    number = io_number(value)
    if number is None:
        return None
    if 0.0 <= number <= 1.0:
        return number * 100.0
    return number


def control_enables(status: dict[str, Any], *, connected: bool) -> dict[str, bool]:
    """Which run buttons are legal for the current controller state."""
    if not connected:
        return {"start": False, "pause": False, "stop": False, "reset": False}

    state = io_text(status.get(STATE)).lower()
    start_permit = io_bool(status.get(READY_PERMIT))
    running = state in {"dosing", "active", "running"}
    paused = state == "paused"
    return {
        "start": start_permit and not running,
        "pause": running,
        "stop": running or paused,
        "reset": not running,
    }


def _display_count(value: Any) -> str | None:
    raw = unwrap_io(value)
    if raw is None or raw == "":
        return None
    number = io_number(raw)
    if number is not None and number.is_integer():
        return str(int(number))
    return str(raw)


def point_fraction(status: dict[str, Any]) -> str:
    idx = _display_count(status.get(CONTROL_POINT_INDEX))
    count = _display_count(status.get(CONTROL_POINT_COUNT))
    if idx is None or count is None:
        return "—"
    return f"{idx} / {count}"


def format_energy(value: Any) -> str:
    number = io_number(value)
    if number is None:
        raw = unwrap_io(value)
        return "—" if raw in (None, "") else str(raw)
    return f"{number:.2f} MeV"


def format_elapsed(value: Any) -> str:
    """Short runs stay in seconds; longer runs use m:ss or h:mm:ss."""
    seconds = io_number(value)
    if seconds is None:
        raw = unwrap_io(value)
        return "—" if raw in (None, "") else str(raw)
    if seconds < 0:
        seconds = 0.0
    if seconds < 60:
        return f"{seconds:.1f} s"
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def coach_message(*, connected: bool, has_plan: bool, status: dict[str, Any]) -> str:
    """Footer line that names the next operator action."""
    if not connected:
        return "Enter the RCI IP and click Connect."
    state = io_text(status.get(STATE)).lower()
    if state in {"dosing", "active", "running"}:
        return "Running — Pause or Stop if you need to halt."
    if state == "paused":
        return "Paused. Start to resume, or Stop."
    if state in {"fault", "error"}:
        return "Controller fault. Check the RCI, then Reset."
    if state == "completed":
        return "Run complete. Browse a zip name, then Download."
    if not has_plan:
        return "Connected. Browse to an input_map.csv, then upload it."
    if not io_bool(status.get(POINTS_VALID)) and not io_bool(
        status.get(COMBINED_POINTS_OK)
    ):
        return "CSV selected. Click Upload to RCI to load the plan."
    if io_bool(status.get(READY_PERMIT)):
        return "Start is enabled. Press Start when you are ready to run."
    reason = io_text(status.get(READY_PERMIT_REASON))
    if reason:
        return f"Waiting for ready permit: {reason}"
    return "Plan is on the controller. Waiting for the RCI ready permit."


def resolve_session_download(
    dest: str, remote_root: str
) -> tuple[str, str] | None:
    """Return ``(remote_dir, zip_path)`` when *dest* is enough to skip a dialog."""
    dest = dest.strip()
    if not dest or dest.startswith("/root/") or not dest.lower().endswith(".zip"):
        return None
    root = remote_root.rstrip("/")
    return f"{root}/{Path(dest).stem}", dest


def default_session_zip_path(current: str, last_folder: str) -> str:
    """Starting path for the session save dialog."""
    current = current.strip()
    folder = last_folder.strip()
    if current.startswith("/root/"):
        name = f"{Path(current).name}.zip"
        return str(Path(folder) / name) if folder else name
    if current.lower().endswith(".zip"):
        return current
    if current:
        return str(Path(current) / "session.zip")
    if folder:
        return str(Path(folder) / "session.zip")
    return "session.zip"


def session_download_hint(dest: str, remote_root: str) -> str:
    """One-line explanation of what Download will pull."""
    dest = dest.strip()
    root = remote_root.rstrip("/")
    if dest.lower().endswith(".zip"):
        return f"Downloads {root}/{Path(dest).stem}"
    if dest.startswith("/root/"):
        return f"Remote folder {dest.rstrip('/')}"
    return f"Zip name is the folder under {root}/"
