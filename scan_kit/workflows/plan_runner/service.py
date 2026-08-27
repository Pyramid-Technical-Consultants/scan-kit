"""Plan runner orchestration against a live RCI controller."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ...igx.keys import field_subscribe_key
from ...igx.mpack import MpackSession, MpackSessionError
from ...igx.rci_paths import (
    PAUSE_BUTTON,
    POINTS_VALID,
    RESET_BUTTON,
    SESSION_DIRECTORY,
    START_BUTTON,
    STATUS_IO_PATHS,
    STOP_BUTTON,
)
from ...igx.upload import upload_control_points_csv


class PlanRunnerService:
    """Connect to RCI, upload plans, run controls, and poll progress."""

    def __init__(self) -> None:
        self._session: MpackSession | None = None
        self._host: str = ""
        self._status_keys: dict[str, str] = {}

    @property
    def connected(self) -> bool:
        return self._session is not None and self._session.ws is not None

    @property
    def host(self) -> str:
        return self._host

    def connect(self, host: str) -> dict[str, Any]:
        """Open mpack session and subscribe to status fields."""
        host = host.strip()
        if not host:
            raise ValueError("host is required")

        self.disconnect()
        session = MpackSession(host)
        session.connect()

        version = session.read_field("admin/version", timeout_s=8.0)
        device_type = None
        try:
            device_type = session.read_field("admin/device_type", timeout_s=4.0)
        except MpackSessionError:
            pass

        key_for_path = {
            field_subscribe_key(p, "value"): p for p in STATUS_IO_PATHS
        }
        session.subscribe_fields({k: False for k in key_for_path})

        self._session = session
        self._host = host
        self._status_keys = key_for_path

        session_dir = "/root/reports/session"
        try:
            session_dir = str(session.read_field(SESSION_DIRECTORY, timeout_s=4.0))
        except MpackSessionError:
            pass

        return {
            "host": host,
            "version": version,
            "device_type": device_type,
            "session_directory": session_dir,
        }

    def disconnect(self) -> None:
        if self._session is not None:
            self._session.close()
        self._session = None
        self._host = ""
        self._status_keys = {}

    def upload_plan(self, csv_path: Path, timeout_s: float = 30.0) -> str:
        """Upload input_map CSV and wait until points load succeeds."""
        session = self._require_session()
        path = Path(csv_path)
        if not path.is_file():
            raise FileNotFoundError(f"CSV not found: {path}")
        target = upload_control_points_csv(
            session,
            self._host,
            path,
            timeout_s=timeout_s,
        )
        valid = session.read_field(POINTS_VALID, timeout_s=8.0)
        if not valid:
            raise MpackSessionError("points_valid is false after upload")
        return target

    def start(self) -> None:
        self._press(START_BUTTON)

    def pause(self) -> None:
        self._press(PAUSE_BUTTON)

    def stop(self) -> None:
        self._press(STOP_BUTTON)

    def reset(self) -> None:
        self._press(RESET_BUTTON)

    def read_status(self) -> dict[str, Any]:
        """Poll subscribed status fields once."""
        session = self._require_session()
        raw = session.recv_subscribed_updates(timeout_s=0.5)
        out: dict[str, Any] = {"host": self._host}
        for key, path in self._status_keys.items():
            if key in raw:
                out[path] = raw[key]
        return out

    def wait_for_state(
        self,
        state_path: str,
        expected: str | set[str],
        timeout_s: float = 60.0,
    ) -> bool:
        """Poll until *state_path* value is in *expected*."""
        if isinstance(expected, str):
            expected_set = {expected}
        else:
            expected_set = set(expected)
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            status = self.read_status()
            value = status.get(state_path)
            if value in expected_set:
                return True
            time.sleep(0.2)
        return False

    def _press(self, button_path: str) -> None:
        self._require_session().press_button(button_path)

    def _require_session(self) -> MpackSession:
        if self._session is None or self._session.ws is None:
            raise MpackSessionError("not connected")
        return self._session
