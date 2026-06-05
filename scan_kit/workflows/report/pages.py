"""Landscape title and conclusion pages for PDF reports."""

from __future__ import annotations

from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

from scan_kit import __version__
from scan_kit.common.session_meta import SessionMeta
from scan_kit.common.settings import ViewSettings

from . import GITHUB_URL
from .types import ViewRenderResult

_LANDSCAPE_FIGSIZE = (16.0, 9.0)
_DPI = 100

# Technical report palette — restrained, print-friendly.
_INK = "#1B2838"
_BODY = "#2D3748"
_MUTED = "#5A6578"
_RULE = "#C5CCD6"
_PANEL_FILL = "#F4F6F9"
_PANEL_EDGE = "#D1D9E6"
_HEADER = "#1B365D"
_ACCENT = "#2B6CB0"
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


def _draw_header(ax, *, left: str, right: str) -> None:
    ax.add_patch(Rectangle((0, 0.93), 1, 0.07, transform=ax.transAxes, facecolor=_HEADER, edgecolor="none", zorder=1))
    ax.text(0.04, 0.965, left, transform=ax.transAxes, ha="left", va="center", fontsize=9, color="white", fontweight="bold", zorder=2)
    ax.text(0.96, 0.965, right, transform=ax.transAxes, ha="right", va="center", fontsize=8.5, color="#D6E4F0", zorder=2)


def _draw_footer(ax, *, left: str, right: str) -> None:
    ax.plot([0.04, 0.96], [0.06, 0.06], color=_RULE, linewidth=0.8, transform=ax.transAxes, zorder=1)
    ax.text(0.04, 0.035, left, transform=ax.transAxes, ha="left", va="center", fontsize=7.5, color=_MUTED, zorder=2)
    ax.text(0.96, 0.035, right, transform=ax.transAxes, ha="right", va="center", fontsize=7.5, color=_MUTED, zorder=2)


def _draw_panel(ax, bbox: tuple[float, float, float, float], *, title: str) -> None:
    x, y, w, h = bbox
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        transform=ax.transAxes,
        facecolor=_PANEL_FILL,
        edgecolor=_PANEL_EDGE,
        linewidth=0.9,
        zorder=1,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.015,
        y + h - 0.028,
        title.upper(),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        fontweight="bold",
        color=_ACCENT,
        zorder=2,
    )


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
            x + 0.015,
            y + h * 0.5,
            "No data available.",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=8.5,
            color=_MUTED,
            zorder=2,
        )
        return

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        colWidths=col_widths,
        cellLoc="left",
        loc="upper left",
        bbox=[x + 0.012, y + 0.012, w - 0.024, h - 0.05],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1, 1.35)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(_RULE)
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_facecolor("#E8EDF4")
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


def render_title_page(
    *,
    title: str,
    subtitle: str,
    author: str,
    organization: str,
    generated_at: datetime,
    base_dir: str,
    session_ids: list[str],
    session_meta: dict[str, SessionMeta | None],
    notes: dict[str, str],
    settings: ViewSettings,
) -> plt.Figure:
    """Build the landscape report title page."""
    fig, ax = _new_landscape_figure()
    stamp = generated_at.strftime("%Y-%m-%d %H:%M")
    _draw_header(ax, left="SCAN KIT ANALYSIS REPORT", right=stamp)

    ax.text(
        0.04,
        0.86,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=24,
        fontweight="bold",
        color=_INK,
        linespacing=1.15,
    )

    y = 0.80
    if subtitle.strip():
        ax.text(0.04, y, subtitle, transform=ax.transAxes, ha="left", va="top", fontsize=12, color=_BODY)
        y -= 0.045

    prepared = _prepared_by(author, organization)
    if prepared:
        ax.text(0.04, y, f"Prepared by {prepared}", transform=ax.transAxes, ha="left", va="top", fontsize=10, color=_MUTED)
        y -= 0.04

    ax.text(
        0.04,
        y,
        "Proton pencil-beam scanning session quality assessment",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        color=_MUTED,
        style="italic",
    )

    _draw_panel(ax, (0.04, 0.50, 0.58, 0.28), title="Sessions analyzed")
    session_rows: list[list[str]] = []
    for sid in session_ids:
        meta = session_meta.get(sid)
        if meta is None:
            session_rows.append([sid, "—", "—", "—", notes.get(sid, "").strip() or "—"])
        else:
            session_rows.append([
                sid,
                meta.short_date,
                meta.short_mu,
                meta.short_time,
                notes.get(sid, "").strip() or "—",
            ])
    _add_table(
        ax,
        (0.04, 0.50, 0.58, 0.28),
        col_labels=["Session ID", "Date", "MU", "Time", "Note"],
        rows=session_rows,
        col_widths=[0.24, 0.12, 0.10, 0.10, 0.28],
    )

    _draw_panel(ax, (0.66, 0.50, 0.30, 0.28), title="Analysis parameters")
    param_lines = [
        f"Background subtraction: {'Enabled' if settings.bg_subtract else 'Disabled'}",
        f"Calibration: {_CAL_LABELS.get(settings.calibration_mode, settings.calibration_mode)}",
    ]
    if settings.cal_factors:
        factors = ", ".join(
            f"{sid}={factor:.4g}"
            for sid, factor in sorted(settings.cal_factors.items())
        )
        param_lines.append(f"Calibration factors: {factors}")
    param_lines.append(f"Sessions in report: {len(session_ids)}")

    for i, line in enumerate(param_lines):
        ax.text(
            0.675,
            0.72 - i * 0.038,
            line,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            color=_BODY,
            zorder=2,
        )

    _draw_panel(ax, (0.04, 0.12, 0.92, 0.14), title="Data source")
    ax.text(
        0.055,
        0.20,
        base_dir,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color=_BODY,
        family="monospace",
        zorder=2,
    )
    ax.text(
        0.055,
        0.145,
        "Absolute path to the session archive directory used for this report.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color=_MUTED,
        zorder=2,
    )

    _draw_footer(
        ax,
        left=f"Generated with Scan Kit v{__version__}",
        right="Page 1 — Title",
    )
    return fig


def render_conclusion_page(
    *,
    title: str,
    author: str,
    organization: str,
    generated_at: datetime,
    base_dir: str,
    session_ids: list[str],
    settings: ViewSettings,
    rendered: list[ViewRenderResult],
    view_display_names: list[str],
) -> plt.Figure:
    """Build the landscape report conclusion / metadata page."""
    fig, ax = _new_landscape_figure()
    stamp = generated_at.strftime("%Y-%m-%d %H:%M")
    _draw_header(ax, left="REPORT SUMMARY & PROVENANCE", right=stamp)

    success_count = sum(1 for r in rendered if r.success)
    skipped = [r for r in rendered if not r.success]

    ax.text(
        0.04,
        0.86,
        "Document Control",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=18,
        fontweight="bold",
        color=_INK,
    )
    ax.text(
        0.04,
        0.805,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        color=_BODY,
    )

    prepared = _prepared_by(author, organization)
    summary_lines = [
        f"Completion: {stamp}",
        f"Views rendered: {success_count} of {len(rendered)}",
        f"Sessions: {len(session_ids)}",
    ]
    if prepared:
        summary_lines.insert(0, f"Prepared by: {prepared}")
    for i, line in enumerate(summary_lines):
        ax.text(
            0.04,
            0.76 - i * 0.032,
            line,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color=_MUTED,
        )

    _draw_panel(ax, (0.04, 0.12, 0.42, 0.52), title="Analysis views")
    view_rows = []
    rendered_by_name = {r.display_name: r for r in rendered}
    for name in view_display_names:
        result = rendered_by_name.get(name)
        if result is None:
            status = "—"
        elif result.success:
            status = "Rendered"
        else:
            status = "Skipped"
        view_rows.append([name, status])

    _add_table(
        ax,
        (0.04, 0.12, 0.42, 0.52),
        col_labels=["View", "Status"],
        rows=view_rows,
        col_widths=[0.72, 0.18],
    )

    _draw_panel(ax, (0.50, 0.40, 0.46, 0.24), title="Processing settings")
    settings_lines = [
        f"Background subtraction: {'Enabled' if settings.bg_subtract else 'Disabled'}",
        f"Calibration: {_CAL_LABELS.get(settings.calibration_mode, settings.calibration_mode)}",
        f"Data directory: {base_dir}",
    ]
    for i, line in enumerate(settings_lines):
        ax.text(
            0.515,
            0.58 - i * 0.045,
            line,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color=_BODY,
            wrap=True,
            zorder=2,
        )

    if skipped:
        _draw_panel(ax, (0.50, 0.12, 0.46, 0.22), title="Skipped views")
        for i, result in enumerate(skipped[:4]):
            reason = result.skip_reason or "No figure produced"
            ax.text(
                0.515,
                0.28 - i * 0.042,
                f"{result.display_name}: {reason}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7.5,
                color=_WARN,
                zorder=2,
            )
        if len(skipped) > 4:
            ax.text(
                0.515,
                0.14,
                f"… and {len(skipped) - 4} additional skipped view(s).",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7.5,
                color=_WARN,
                zorder=2,
            )

    _draw_panel(ax, (0.50, 0.66, 0.46, 0.16), title="Software provenance")
    ax.text(
        0.515,
        0.76,
        f"Scan Kit v{__version__}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        color=_ACCENT,
        zorder=2,
    )
    ax.text(
        0.515,
        0.72,
        "Open-source proton pencil beam scanning analysis toolkit.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color=_MUTED,
        zorder=2,
    )
    ax.text(
        0.515,
        0.685,
        GITHUB_URL,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color=_ACCENT,
        zorder=2,
    )

    _draw_footer(
        ax,
        left="End of report",
        right=f"Scan Kit v{__version__}",
    )
    return fig
