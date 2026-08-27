"""VTree path helpers for WebSocket subscription and set messages."""


def field_subscribe_key(io_path: str, field: str = "value") -> str:
    """Long subscription id for a field: /<io_path>/<field> (leading slash, /value suffix)."""
    seg = io_path.strip("/")
    if not seg:
        return f"/{field}"
    if seg.endswith(f"/{field}"):
        return f"/{seg}"
    return f"/{seg}/{field}"
