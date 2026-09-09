"""Upload control-point CSV files to an IGX device."""

from __future__ import annotations

import time
from pathlib import Path

from .http import put_bytes
from .mpack import MpackSession, MpackSessionError
from .rci_paths import (
    DEFAULT_CONTROL_POINTS_PATH,
    LOAD_STATE_SUCCESS,
    POINTS_LOAD_STATE,
    POINTS_UPLOAD,
    POINTS_UPLOAD_TARGET,
)


def upload_control_points_csv(
    session: MpackSession,
    host: str,
    csv_path: Path,
    *,
    upload_io: str = POINTS_UPLOAD,
    default_target: str = DEFAULT_CONTROL_POINTS_PATH,
    timeout_s: float = 30.0,
) -> str:
    """HTTP PUT CSV then press points_upload; wait until load succeeds.

    Returns the device path that received the file.
    """
    session.set_field(POINTS_UPLOAD_TARGET, default_target)
    time.sleep(0.2)
    target = session.read_field(POINTS_UPLOAD_TARGET, timeout_s=4.0)
    if not target or not str(target).strip():
        target = default_target
    else:
        target = str(target).strip()

    data = csv_path.read_bytes()
    put_bytes(host, target, data)

    session.press_button(upload_io)

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        state = session.read_field(POINTS_LOAD_STATE, timeout_s=2.0)
        if state == LOAD_STATE_SUCCESS:
            return target
        if state not in (0, 1, LOAD_STATE_SUCCESS):
            raise MpackSessionError(f"points_load_state error: {state}")
        time.sleep(0.2)

    raise MpackSessionError("timed out waiting for points_load_state success")
