"""Session metadata types and ``termination_summary.txt`` parsing."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class SessionMeta:
    """Lightweight metadata extracted from termination_summary.txt."""

    date: datetime | None
    primary_mu: float | None
    treatment_time_s: int | None

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
    def short_time(self) -> str:
        if self.treatment_time_s is None:
            return "?"
        minutes, seconds = divmod(self.treatment_time_s, 60)
        return f"{minutes}:{seconds:02d}"


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


def parse_termination_summary_text(text: str) -> SessionMeta:
    """Parse ``termination_summary.txt`` body into :class:`SessionMeta`."""
    date: datetime | None = None
    primary_mu: float | None = None
    treatment_s: int | None = None

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

    return SessionMeta(date=date, primary_mu=primary_mu, treatment_time_s=treatment_s)
