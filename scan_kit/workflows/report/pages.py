"""Landscape title and conclusion pages for PDF reports."""

from __future__ import annotations

from datetime import datetime

import matplotlib.pyplot as plt

from scan_kit import __version__
from scan_kit.common.session_meta import SessionMeta
from scan_kit.common.settings import ViewSettings

from scan_kit.views import ViewEntry

from . import GITHUB_URL
from .types import ViewRenderResult

_LANDSCAPE_FIGSIZE = (16.0, 9.0)
_DPI = 100

_INK = "#1B2838"
_BODY = "#2D3748"
_MUTED = "#5A6578"
_RULE = "#C5CCD6"
_WARN = "#9B2C2C"

_CAL_LABELS = {
    "off": "Off",
    "per_session": "Per-session",
    "constrained": "Constrained",
}


def _new_landscape_figure() -> tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=_LANDSCAPE_FIGSIZE, dpi=_DPI)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_facecolor("white")
    return fig, ax


def _section_heading(ax, y: float, text: str) -> float:
    ax.text(
        0.05,
        y,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
        color=_INK,
    )
    return y - 0.04


def _add_table(
    ax,
    bbox: tuple[float, float, float, float],
    *,
    col_labels: list[str],
    rows: list[list[str]],
    col_widths: list[float] | None = None,
) -> None:
    x, y, w, h = bbox
    if not rows:
        ax.text(
            x,
            y + h * 0.5,
            "No data available.",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=9,
            color=_MUTED,
        )
        return

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        colWidths=col_widths,
        cellLoc="left",
        loc="upper left",
        bbox=[x, y, w, h],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(_RULE)
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_facecolor("#F7F7F7")
            cell.set_text_props(fontweight="bold", color=_INK)
        else:
            cell.set_facecolor("white")
            cell.set_text_props(color=_BODY)


def _prepared_by(author: str, organization: str) -> str | None:
    author = author.strip()
    organization = organization.strip()
    if author and organization:
        return f"{author}, {organization}"
    return author or organization or None


def _session_rows(
    session_ids: list[str],
    session_meta: dict[str, SessionMeta | None],
    notes: dict[str, str],
) -> list[list[str]]:
    rows: list[list[str]] = []
    for sid in session_ids:
        meta = session_meta.get(sid)
        note = notes.get(sid, "").strip() or "—"
        if meta is None:
            rows.append([sid, "—", "—", "—", note])
        else:
            rows.append([sid, meta.short_date, meta.short_mu, meta.short_time, note])
    return rows


def render_title_page(
    *,
    title: str,
    subtitle: str,
    author: str,
    organization: str,
    generated_at: datetime,
    session_ids: list[str],
    session_meta: dict[str, SessionMeta | None],
    notes: dict[str, str],
    settings: ViewSettings,
) -> plt.Figure:
    """Build the landscape report title page."""
    fig, ax = _new_landscape_figure()
    stamp = generated_at.strftime("%Y-%m-%d %H:%M")

    ax.text(
        0.05,
        0.92,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=22,
        fontweight="bold",
        color=_INK,
    )

    y = 0.84
    if subtitle.strip():
        ax.text(0.05, y, subtitle, transform=ax.transAxes, ha="left", va="top", fontsize=11, color=_BODY)
        y -= 0.04

    prepared = _prepared_by(author, organization)
    if prepared:
        ax.text(0.05, y, f"Prepared by {prepared}", transform=ax.transAxes, ha="left", va="top", fontsize=10, color=_MUTED)
        y -= 0.035

    ax.text(0.95, 0.92, stamp, transform=ax.transAxes, ha="right", va="top", fontsize=9, color=_MUTED)

    y = _section_heading(ax, 0.72, "Selected sessions")
    session_rows = _session_rows(session_ids, session_meta, notes)
    table_height = min(0.42, 0.06 + 0.045 * max(len(session_rows), 1))
    _add_table(
        ax,
        (0.05, 0.72 - table_height - 0.04, 0.90, table_height),
        col_labels=["Session ID", "Date", "MU", "Time", "Note"],
        rows=session_rows,
        col_widths=[0.28, 0.10, 0.08, 0.08, 0.36],
    )

    y = 0.72 - table_height - 0.10
    y = _section_heading(ax, y, "Analysis settings")
    settings_lines = [
        f"Background subtraction: {'On' if settings.bg_subtract else 'Off'}",
        f"Calibration: {_CAL_LABELS.get(settings.calibration_mode, settings.calibration_mode)}",
    ]
    if settings.cal_factors:
        factors = ", ".join(
            f"{sid}={factor:.4g}"
            for sid, factor in sorted(settings.cal_factors.items())
        )
        settings_lines.append(f"Calibration factors: {factors}")

    for i, line in enumerate(settings_lines):
        ax.text(
            0.05,
            y - i * 0.03,
            line,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.5,
            color=_BODY,
        )

    ax.text(
        0.05,
        0.04,
        f"Generated with Scan Kit v{__version__}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color=_MUTED,
    )
    return fig


def render_conclusion_page(
    *,
    title: str,
    author: str,
    organization: str,
    generated_at: datetime,
    session_ids: list[str],
    settings: ViewSettings,
    rendered: list[ViewRenderResult],
    views: list[ViewEntry],
) -> plt.Figure:
    """Build the landscape report conclusion page."""
    fig, ax = _new_landscape_figure()
    stamp = generated_at.strftime("%Y-%m-%d %H:%M")
    success_count = sum(1 for r in rendered if r.success)
    skipped = [r for r in rendered if not r.success]

    ax.text(
        0.05,
        0.92,
        "Report summary",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=22,
        fontweight="bold",
        color=_INK,
    )
    ax.text(0.95, 0.92, stamp, transform=ax.transAxes, ha="right", va="top", fontsize=9, color=_MUTED)

    y = 0.84
    ax.text(0.05, y, title, transform=ax.transAxes, ha="left", va="top", fontsize=11, color=_BODY)
    y -= 0.04

    prepared = _prepared_by(author, organization)
    summary_lines = [
        f"Views rendered: {success_count} of {len(rendered)}",
        f"Sessions: {len(session_ids)}",
        f"Background subtraction: {'On' if settings.bg_subtract else 'Off'}",
        f"Calibration: {_CAL_LABELS.get(settings.calibration_mode, settings.calibration_mode)}",
    ]
    if prepared:
        summary_lines.insert(0, f"Prepared by: {prepared}")

    for i, line in enumerate(summary_lines):
        ax.text(
            0.05,
            y - i * 0.03,
            line,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.5,
            color=_MUTED,
        )

    y = 0.62
    y = _section_heading(ax, y, "Analysis views")
    view_rows = []
    rendered_by_name = {r.display_name: r for r in rendered}
    for display_name, _module_name, description in views:
        result = rendered_by_name.get(display_name)
        if result is None:
            status = "—"
        elif result.success:
            status = "Rendered"
        else:
            status = "Skipped"
        view_rows.append([display_name, description, status])

    view_table_height = min(0.40, 0.06 + 0.038 * max(len(view_rows), 1))
    _add_table(
        ax,
        (0.05, y - view_table_height - 0.02, 0.90, view_table_height),
        col_labels=["View", "Description", "Status"],
        rows=view_rows,
        col_widths=[0.24, 0.58, 0.10],
    )

    if skipped:
        y_right = 0.62
        y_right = _section_heading(ax, y_right, "Skipped views")
        for i, result in enumerate(skipped[:6]):
            reason = result.skip_reason or "No figure produced"
            ax.text(
                0.52,
                y_right - i * 0.035,
                f"{result.display_name}: {reason}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8.5,
                color=_WARN,
            )
        if len(skipped) > 6:
            ax.text(
                0.52,
                y_right - 6 * 0.035,
                f"… and {len(skipped) - 6} additional skipped view(s).",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8.5,
                color=_WARN,
            )

    ax.text(
        0.05,
        0.12,
        f"Scan Kit v{__version__}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        color=_BODY,
    )
    ax.text(
        0.05,
        0.08,
        "Open-source proton pencil beam scanning analysis toolkit.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color=_MUTED,
    )
    ax.text(
        0.05,
        0.05,
        GITHUB_URL,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color=_BODY,
    )
    return fig
