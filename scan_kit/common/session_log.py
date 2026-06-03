"""Parse and summarize DCS ``SessionLogFile.log`` session logs."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .session_source import load_session_text, resolve_session_source

SESSION_LOG_FILENAME = "SessionLogFile.log"

_LINE_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3}) "
    r"(\w+) \[(\w+)\] (.*)$"
)

# Repetitive low-signal traffic (ACK storms, command polling).
_NOISE_SUBSTRINGS = (
    "Got ACK to ",
    "Received command:",
)

_TIMELINE_RE = re.compile(
    r"^(?P<kind>START MAP|SCAN EXECUTING|LOAD 2D MAP|LOAD 3D MAP)"
    r"(?: FOR (?P<scope>LAYER|SESSION): (?P<layer>-?\d+))?"
    r" - TIMELINE\((?P<phase>[^)]+)\): T=(?P<seconds>[\d.]+)s$"
)

_WDT_RE = re.compile(
    r"^(?P<device>\S+) wdt read cnt: (?P<read>\d+) - write counter: (?P<write>\d+)$"
)


def is_noise_message(message: str) -> bool:
    """True for high-volume, low-information log lines."""
    return any(token in message for token in _NOISE_SUBSTRINGS)


def message_template(message: str) -> str:
    """Normalize a log message for grouping and comparison."""
    text = re.sub(r"\d+\.?\d*", "#", message)
    text = re.sub(r"/var/log/[^\s\"']+", "<path>", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@dataclass(frozen=True)
class LogEntry:
    """One parsed line from a session log."""

    line_no: int
    timestamp: datetime
    level: str
    logger: str
    message: str

    @property
    def template(self) -> str:
        return message_template(self.message)


@dataclass
class LayerTimelineRow:
    """Per-layer timing extracted from TIMELINE log lines."""

    layer: int
    start_map_s: float | None = None
    scan_execute_s: float | None = None
    load_map_s: float | None = None

    @property
    def total_reported_s(self) -> float | None:
        parts = [self.start_map_s, self.scan_execute_s, self.load_map_s]
        finite = [p for p in parts if p is not None]
        if not finite:
            return None
        return sum(finite)


@dataclass
class SessionLogData:
    """Parsed session log with derived summaries."""

    session_id: str
    path_label: str
    entries: list[LogEntry] = field(default_factory=list)
    level_counts: Counter[str] = field(default_factory=Counter)
    template_counts: Counter[str] = field(default_factory=Counter)
    layer_timeline: dict[int, LayerTimelineRow] = field(default_factory=dict)
    wdt_mismatches: Counter[str] = field(default_factory=Counter)

    @property
    def start_time(self) -> datetime | None:
        return self.entries[0].timestamp if self.entries else None

    @property
    def end_time(self) -> datetime | None:
        return self.entries[-1].timestamp if self.entries else None

    @property
    def duration_s(self) -> float | None:
        if self.start_time is None or self.end_time is None:
            return None
        return (self.end_time - self.start_time).total_seconds()

    @property
    def error_count(self) -> int:
        return self.level_counts.get("ERROR", 0)

    @property
    def layers_scanned(self) -> int:
        return sum(1 for row in self.layer_timeline.values() if row.scan_execute_s is not None)

    def notable_entries(self, *, include_debug: bool = False) -> list[LogEntry]:
        """Entries excluding noise; optionally drop DEBUG."""
        out: list[LogEntry] = []
        for entry in self.entries:
            if is_noise_message(entry.message):
                continue
            if not include_debug and entry.level == "DEBUG":
                continue
            out.append(entry)
        return out


def parse_session_log_text(text: str, *, session_id: str = "", path_label: str = "") -> SessionLogData:
    """Parse raw log text into :class:`SessionLogData`."""
    data = SessionLogData(session_id=session_id, path_label=path_label)
    for line_no, raw in enumerate(text.splitlines(), start=1):
        raw = raw.strip()
        if not raw:
            continue
        match = _LINE_RE.match(raw)
        if match is None:
            continue
        ts_s, ms_s, level, logger, message = match.groups()
        try:
            ts = datetime.strptime(f"{ts_s},{ms_s}", "%Y-%m-%d %H:%M:%S,%f")
        except ValueError:
            continue
        entry = LogEntry(line_no=line_no, timestamp=ts, level=level, logger=logger, message=message)
        data.entries.append(entry)
        data.level_counts[level] += 1
        if not is_noise_message(message):
            data.template_counts[entry.template] += 1
        _ingest_special_lines(data, message)
    return data


def load_session_log(session_id: str, base_dir: str | Path) -> SessionLogData | None:
    """Load ``SessionLogFile.log`` for *session_id* under *base_dir*."""
    src = resolve_session_source(session_id, base_dir)
    if src is None:
        return None
    text = load_session_text(src, SESSION_LOG_FILENAME)
    if text is None:
        return None
    label = str(src.path)
    return parse_session_log_text(text, session_id=session_id, path_label=label)


def compare_template_counts(
    a: SessionLogData,
    b: SessionLogData,
    *,
    min_count: int = 1,
) -> list[tuple[str, int, int, int]]:
    """Return ``(template, count_a, count_b, delta)`` sorted by largest |delta|."""
    keys = set(a.template_counts) | set(b.template_counts)
    rows: list[tuple[str, int, int, int]] = []
    for key in keys:
        ca = a.template_counts.get(key, 0)
        cb = b.template_counts.get(key, 0)
        if ca < min_count and cb < min_count:
            continue
        rows.append((key, ca, cb, ca - cb))
    rows.sort(key=lambda row: (abs(row[3]), max(row[1], row[2])), reverse=True)
    return rows


def _ingest_special_lines(data: SessionLogData, message: str) -> None:
    timeline = _TIMELINE_RE.match(message)
    if timeline:
        kind = timeline.group("kind")
        layer_s = timeline.group("layer")
        seconds = float(timeline.group("seconds"))
        if layer_s is None:
            return
        layer = int(layer_s)
        if kind == "LOAD 3D MAP":
            return
        row = data.layer_timeline.setdefault(layer, LayerTimelineRow(layer=layer))
        if kind == "START MAP":
            row.start_map_s = seconds
        elif kind == "SCAN EXECUTING":
            row.scan_execute_s = seconds
        elif kind == "LOAD 2D MAP" and layer == -1:
            pass
        elif kind == "LOAD 2D MAP":
            row.load_map_s = seconds
        return

    wdt = _WDT_RE.match(message)
    if wdt:
        device = wdt.group("device")
        data.wdt_mismatches[device] += 1


def merged_layer_ids(*logs: SessionLogData) -> list[int]:
    """Sorted layer indices present in any log's timeline."""
    layers: set[int] = set()
    for log in logs:
        layers.update(log.layer_timeline)
    return sorted(n for n in layers if n >= 0)
