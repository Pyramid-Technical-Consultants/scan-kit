"""Decode MessagePack blob wrappers to Python values (mirrors web/js/lib/typedArray.js)."""

import struct
from typing import Any

DTYPE_FLOAT32 = 1
DTYPE_FLOAT64 = 2


def summarize_array(value: Any) -> Any:
    """Compact large numeric arrays to {len, head, min, max} so responses stay small."""
    if isinstance(value, list) and len(value) > 16 and all(
        isinstance(x, (int, float)) for x in value
    ):
        nums = [x for x in value if isinstance(x, (int, float))]
        return {
            "len": len(value),
            "head": value[:8],
            "min": min(nums) if nums else None,
            "max": max(nums) if nums else None,
        }
    return value


def _unpack_numeric_blob(dtype: Any, data: Any) -> list[float] | None:
    if dtype not in (DTYPE_FLOAT32, DTYPE_FLOAT64) or not isinstance(
        data, (bytes, bytearray)
    ):
        return None
    if dtype == DTYPE_FLOAT32:
        count = len(data) // 4
        return list(struct.unpack(f"<{count}f", data[: count * 4]))
    count = len(data) // 8
    return list(struct.unpack(f"<{count}d", data[: count * 8]))


def expand_history_blob(obj: Any) -> list[tuple[Any, float]] | None:
    """Decode { $t, $d, $ts } buffered history, or None if this is not that shape."""
    if not isinstance(obj, dict) or "$ts" not in obj:
        return None
    values = _unpack_numeric_blob(obj.get("$t"), obj.get("$d"))
    times = obj.get("$ts")
    if values is None or not isinstance(times, (bytes, bytearray)):
        return None
    if len(times) < len(values) * 8:
        return None
    stamps = struct.unpack(f"<{len(values)}Q", times[: len(values) * 8])
    return list(zip(values, (float(ts) for ts in stamps)))


def decode_mpack_value(obj: Any) -> Any:
    """Recursively convert blob wrappers {$t, $d} to lists; summarize large arrays."""
    if isinstance(obj, dict):
        if "$ts" in obj:
            history = expand_history_blob(obj)
            if history is not None:
                return history
        dtype = obj.get("$t")
        data = obj.get("$d")
        unpacked = _unpack_numeric_blob(dtype, data)
        if unpacked is not None:
            return summarize_array(unpacked)
        return {k: decode_mpack_value(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [decode_mpack_value(v) for v in obj]
    return obj


def unwrap_field_update(obj: Any) -> Any:
    """Extract the latest scalar from an mpack field update."""
    history = expand_history_blob(obj)
    if history:
        return history[-1][0]
    value = decode_mpack_value(obj)
    if not isinstance(value, list) or not value:
        return value
    if isinstance(value[0], list):
        last = value[-1]
        if isinstance(last, list) and last:
            return last[0]
        return value
    if len(value) >= 2 and not isinstance(value[0], list):
        return value[0]
    return value
