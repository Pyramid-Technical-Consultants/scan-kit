"""Session metadata types and ``termination_summary.txt`` parsing."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from datetime import datetime


@dataclass(frozen=True)
class SessionMeta:
    """Lightweight metadata extracted from termination_summary.txt."""

    date: datetime | None
    primary_mu: float | None
    treatment_time_s: int | None
    room_number: int | None
    config_name: str | None = None
    map_extent_mm: float | None = None
    layer_count: int | None = None

    @property
    def short_date(self) -> str:
        if self.date is None:
            return "?"
        return self.date.strftime("%m/%d/%y")

    @property
    def short_mu(self) -> str:
        if self.primary_mu is None:
            return "?"
        return f"{self.primary_mu:.1f}"

    @property
    def short_extent(self) -> str:
        if self.map_extent_mm is None:
            return "?"
        return str(int(round(self.map_extent_mm)))

    @property
    def short_layers(self) -> str:
        if self.layer_count is None:
            return "?"
        return str(self.layer_count)

    @property
    def short_time(self) -> str:
        if self.treatment_time_s is None:
            return "?"
        minutes, seconds = divmod(self.treatment_time_s, 60)
        return f"{minutes}:{seconds:02d}"

    @property
    def short_room(self) -> str:
        if self.room_number is None:
            return "?"
        return str(self.room_number)

    @property
    def short_config(self) -> str:
        name = (self.config_name or "").strip()
        return name if name else "?"


_DATE_FMT = "%a %b %d %H:%M:%S %Y"  # e.g. "Thu Dec 11 21:36:55 2025"

# Termination summaries may be bare numbers ("152.153") or include units
# ("37.3123 MU", "195.602 seconds").
_NUMERIC_VALUE_RE = re.compile(r"^([+-]?(?:\d+\.?\d*|\.\d+))")


def _parse_labeled_numeric(line: str, label: str) -> float | None:
    """Extract a numeric value from ``Label: <value> [<unit>]`` lines."""
    prefix = f"{label}:"
    if not line.startswith(prefix):
        return None
    value_part = line[len(prefix) :].strip()
    match = _NUMERIC_VALUE_RE.match(value_part)
    if match is None:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _parse_layer_delivery(line: str) -> int | None:
    """``Layer delivery: 56/76`` → planned 76; a lone number is used as-is."""
    prefix = "Layer delivery:"
    if not line.startswith(prefix):
        return None
    rest = line[len(prefix) :].strip()
    if not rest:
        return None
    if "/" in rest:
        rest = rest.split("/", 1)[1].strip()
    match = _NUMERIC_VALUE_RE.match(rest)
    if match is None:
        return None
    try:
        return int(float(match.group(1)))
    except ValueError:
        return None


def merge_session_geom(
    meta: SessionMeta | None,
    *,
    map_extent_mm: float | None = None,
    layer_count: int | None = None,
) -> SessionMeta | None:
    """Fill missing extent/layer fields; do not overwrite values already set."""
    if map_extent_mm is None and layer_count is None:
        return meta
    base = meta or SessionMeta(
        date=None,
        primary_mu=None,
        treatment_time_s=None,
        room_number=None,
    )
    return replace(
        base,
        map_extent_mm=(
            base.map_extent_mm if base.map_extent_mm is not None else map_extent_mm
        ),
        layer_count=(
            base.layer_count if base.layer_count is not None else layer_count
        ),
    )


def parse_termination_summary_text(text: str) -> SessionMeta:
    """Parse ``termination_summary.txt`` body into :class:`SessionMeta`."""
    date: datetime | None = None
    primary_mu: float | None = None
    treatment_s: int | None = None
    room_number: int | None = None
    config_name: str | None = None
    extent_w: float | None = None
    extent_h: float | None = None
    layer_count: int | None = None

    for line in text.splitlines():
        line = line.strip()
        if line.startswith("Date:"):
            try:
                date = datetime.strptime(line.split(":", 1)[1].strip(), _DATE_FMT)
            except ValueError:
                pass
        elif line.startswith("Primary total dose:"):
            parsed = _parse_labeled_numeric(line, "Primary total dose")
            if parsed is not None:
                primary_mu = parsed
        elif line.startswith("Treatment time:"):
            parsed = _parse_labeled_numeric(line, "Treatment time")
            if parsed is not None:
                treatment_s = int(parsed)
        elif line.startswith("Room number:"):
            parsed = _parse_labeled_numeric(line, "Room number")
            if parsed is not None:
                room_number = int(parsed)
        elif line.startswith("Configuration name:"):
            name = line.split(":", 1)[1].strip()
            config_name = name or None
        elif line.startswith("Spot extent width:"):
            parsed = _parse_labeled_numeric(line, "Spot extent width")
            if parsed is not None:
                extent_w = parsed
        elif line.startswith("Spot extent height:"):
            parsed = _parse_labeled_numeric(line, "Spot extent height")
            if parsed is not None:
                extent_h = parsed
        elif line.startswith("Layer delivery:"):
            parsed = _parse_layer_delivery(line)
            if parsed is not None:
                layer_count = parsed

    map_extent_mm = None
    if extent_w is not None or extent_h is not None:
        map_extent_mm = max(v for v in (extent_w, extent_h) if v is not None)

    return SessionMeta(
        date=date,
        primary_mu=primary_mu,
        treatment_time_s=treatment_s,
        room_number=room_number,
        config_name=config_name,
        map_extent_mm=map_extent_mm,
        layer_count=layer_count,
    )
