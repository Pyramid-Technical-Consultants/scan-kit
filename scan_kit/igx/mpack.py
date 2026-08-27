"""MessagePack WebSocket session (production path: mpack.v2 + array blobs)."""

from __future__ import annotations

import time
from typing import Any

import websocket

from .blobs import unwrap_field_update
from .http import normalize_host
from .keys import field_subscribe_key

try:
    import msgpack as _msgpack
except ImportError:
    _msgpack = None

MPACK_SUBPROTOCOLS = ["mpack.v2", "mpack.v1", "mpack"]

CONNECT_TIMEOUT_S = 10
DEFAULT_POLL_TIMEOUT_S = 8.0
DEFAULT_SESSION_CONFIG = {
    "use_short_id": False,
    "partial_row_update": False,
    "update_arrays_as_blobs": True,
    "update_buffered_as_blobs": True,
}


class MpackSessionError(Exception):
    """WebSocket / msgpack session failure."""


class MpackSession:
    """One mpack WebSocket to a device; configure once, then subscribe/get/set."""

    def __init__(self, host: str):
        self.host = host
        self.ws: websocket.WebSocket | None = None
        self._configured = False

    def connect(self) -> None:
        if _msgpack is None:
            raise MpackSessionError("msgpack package not installed; pip install msgpack")
        last_err: Exception | None = None
        for _ in range(2):
            try:
                self.ws = websocket.create_connection(
                    f"ws://{normalize_host(self.host)}",
                    timeout=CONNECT_TIMEOUT_S,
                    subprotocols=MPACK_SUBPROTOCOLS,
                )
                break
            except Exception as e:
                last_err = e
                time.sleep(0.5)
        if self.ws is None:
            raise MpackSessionError(str(last_err or "connection failed"))
        sub = self.ws.subprotocol or ""
        if "mpack" not in sub:
            raise MpackSessionError(
                f"device did not accept mpack subprotocol (got '{sub}')"
            )
        self._send("config", DEFAULT_SESSION_CONFIG)
        self._configured = True

    def close(self) -> None:
        if self.ws:
            try:
                self.ws.close()
            except Exception:
                pass
            self.ws = None
        self._configured = False

    def __enter__(self) -> MpackSession:
        self.connect()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def _send(self, event: str, data: Any = None) -> None:
        if not self.ws:
            raise MpackSessionError("not connected")
        msg = {"event": event, "data": data} if data is not None else {"event": event}
        self.ws.send_binary(_msgpack.packb(msg, use_bin_type=True))

    def _recv(self, timeout: float = 0.8) -> dict | None:
        if not self.ws:
            return None
        self.ws.settimeout(timeout)
        try:
            raw = self.ws.recv()
        except Exception:
            return None
        if not isinstance(raw, (bytes, bytearray)):
            return None
        msg = _msgpack.unpackb(raw, raw=False)
        return msg if isinstance(msg, dict) else None

    def subscribe_fields(self, subscribe_map: dict[str, bool]) -> None:
        """Subscribe to field keys. Value True = buffered (all samples since last get)."""
        if not subscribe_map:
            return
        self._send("subscribe", subscribe_map)

    def poll_field_updates(
        self,
        keys: set[str],
        timeout_s: float = DEFAULT_POLL_TIMEOUT_S,
        poll_interval_s: float = 0.05,
    ) -> dict[str, Any]:
        """Poll 'get' until all keys arrive or timeout."""
        got: dict[str, Any] = {}
        deadline = time.time() + timeout_s
        while time.time() < deadline and keys - set(got.keys()):
            self._send("get")
            msg = self._recv(timeout=min(0.8, max(0.05, deadline - time.time())))
            if msg and msg.get("event") == "update" and isinstance(msg.get("data"), dict):
                for key, value in msg["data"].items():
                    if key in keys:
                        got[key] = unwrap_field_update(value)
            if poll_interval_s > 0:
                time.sleep(poll_interval_s)
        return got

    def read_fields(
        self,
        io_paths: list[str],
        field: str = "value",
        timeout_s: float = DEFAULT_POLL_TIMEOUT_S,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        """Batch read fields over one connection."""
        key_for_path = {field_subscribe_key(p, field): p for p in io_paths}
        keys = set(key_for_path.keys())
        self.subscribe_fields({k: False for k in keys})
        raw = self.poll_field_updates(keys, timeout_s=timeout_s)
        values: dict[str, Any] = {}
        errors: dict[str, str] = {}
        for key, path in key_for_path.items():
            if key in raw:
                values[path] = raw[key]
            else:
                errors[path] = "no update within timeout"
        return values, errors

    def read_field(
        self,
        io_path: str,
        field: str = "value",
        timeout_s: float = DEFAULT_POLL_TIMEOUT_S,
    ) -> Any:
        """Read a single field; raises MpackSessionError if missing."""
        values, errors = self.read_fields([io_path], field=field, timeout_s=timeout_s)
        if io_path in values:
            return values[io_path]
        detail = errors.get(io_path, "no update")
        raise MpackSessionError(
            f"no value for '{field_subscribe_key(io_path, field)}': {detail}"
        )

    def set_fields(self, field_values: dict[str, Any]) -> None:
        """Write fields. Keys are long ids or io paths (value field)."""
        payload: dict[str, Any] = {}
        for key, val in field_values.items():
            k = (
                key
                if key.startswith("/") and key.count("/") >= 2
                else field_subscribe_key(key, "value")
            )
            payload[k] = val
        self._send("set", payload)

    def set_field(self, io_path: str, value: Any, field: str = "value") -> None:
        self.set_fields({field_subscribe_key(io_path, field): value})

    def press_button(self, io_path: str) -> None:
        """Press a button IO (set value True)."""
        self.set_field(io_path, True)

    def recv_subscribed_updates(self, timeout_s: float = 0.35) -> dict[str, Any]:
        """Poll get and return unwrapped field updates (ignore non-update frames)."""
        got: dict[str, Any] = {}
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            self._send("get")
            msg = self._recv(timeout=min(0.25, max(0.02, deadline - time.time())))
            if msg and msg.get("event") == "update" and isinstance(msg.get("data"), dict):
                for key, value in msg["data"].items():
                    got[key] = unwrap_field_update(value)
                if got:
                    break
        return got
