"""Session metadata types and ``termination_summary.txt`` parsing."""

from __future__ import annotations

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
        return str(self.treatment_time_s)


_DATE_FMT = "%a %b %d %H:%M:%S %Y"  # e.g. "Thu Dec 11 21:36:55 2025"


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
            try:
                primary_mu = float(line.split(":")[1].strip())
            except ValueError:
                pass
        elif line.startswith("Treatment time:"):
            try:
                treatment_s = int(float(line.split(":")[1].strip()))
            except ValueError:
                pass

    return SessionMeta(date=date, primary_mu=primary_mu, treatment_time_s=treatment_s)
