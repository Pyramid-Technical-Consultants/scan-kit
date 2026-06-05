"""Matplotlib-rendered title and conclusion pages for PDF reports."""

from __future__ import annotations

from datetime import datetime

import matplotlib.pyplot as plt

from scan_kit import __version__
from scan_kit.common.session_meta import SessionMeta
from scan_kit.common.settings import ViewSettings

from . import GITHUB_URL
from .types import ViewRenderResult

_PORTRAIT_FIGSIZE = (8.5, 11.0)
_TITLE_COLOR = "#1a1a2e"
_MUTED_COLOR = "#4a4a68"
_ACCENT_COLOR = "#2c5282"

_CAL_LABELS = {
    "off": "Off",
    "per_session": "Per-Session",
    "constrained": "Constrained",
}


def _new_portrait_figure() -> plt.Figure:
    fig = plt.figure(figsize=_PORTRAIT_FIGSIZE, dpi=150)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_facecolor("white")
    return fig


def _add_text_block(
    ax,
    lines: list[tuple[str, dict]],
    *,
    start_y: float = 0.92,
    line_height: float = 0.028,
) -> float:
    y = start_y
    for text, kwargs in lines:
        if not text:
            y -= line_height * 0.5
            continue
        ax.text(0.08, y, text, transform=ax.transAxes, va="top", **kwargs)
        y -= line_height
    return y


def render_title_page(
    *,
    title: str,
    subtitle: str,
    author: str,
    generated_at: datetime,
    base_dir: str,
    session_ids: list[str],
    session_meta: dict[str, SessionMeta | None],
    notes: dict[str, str],
    settings: ViewSettings,
) -> plt.Figure:
    """Build the report title page as a matplotlib figure."""
    fig = _new_portrait_figure()
    ax = fig.axes[0]

    header_lines: list[tuple[str, dict]] = [
        (title, {"fontsize": 22, "fontweight": "bold", "color": _TITLE_COLOR}),
    ]
    if subtitle.strip():
        header_lines.append(
            (subtitle, {"fontsize": 12, "color": _MUTED_COLOR}),
        )
    if author.strip():
        header_lines.append(
            (author, {"fontsize": 11, "color": _MUTED_COLOR}),
        )
    header_lines.append(
        (
            generated_at.strftime("Generated %B %d, %Y at %H:%M"),
            {"fontsize": 10, "color": _MUTED_COLOR},
        ),
    )
    y = _add_text_block(ax, header_lines, start_y=0.90, line_height=0.035)

    y -= 0.02
    ax.plot([0.08, 0.92], [y, y], color="#d0d0d8", linewidth=0.8, transform=ax.transAxes)
    y -= 0.04

    body_lines: list[tuple[str, dict]] = [
        ("Data directory", {"fontsize": 11, "fontweight": "bold", "color": _TITLE_COLOR}),
        (base_dir, {"fontsize": 9, "color": _MUTED_COLOR, "family": "monospace"}),
        ("", {}),
        ("Analysis settings", {"fontsize": 11, "fontweight": "bold", "color": _TITLE_COLOR}),
        (
            f"BG subtraction: {'On' if settings.bg_subtract else 'Off'}",
            {"fontsize": 10, "color": _MUTED_COLOR},
        ),
        (
            f"Calibration: {_CAL_LABELS.get(settings.calibration_mode, settings.calibration_mode)}",
            {"fontsize": 10, "color": _MUTED_COLOR},
        ),
    ]
    if settings.cal_factors:
        factors_text = ", ".join(
            f"{sid}: {factor:.4g}"
            for sid, factor in sorted(settings.cal_factors.items())
        )
        body_lines.append(
            (f"Cal factors: {factors_text}", {"fontsize": 9, "color": _MUTED_COLOR}),
        )

    body_lines.extend([
        ("", {}),
        ("Sessions", {"fontsize": 11, "fontweight": "bold", "color": _TITLE_COLOR}),
    ])
    for sid in session_ids:
        meta = session_meta.get(sid)
        if meta is None:
            row = f"  {sid}"
        else:
            row = (
                f"  {sid}  |  {meta.short_date}  |  "
                f"{meta.short_mu} MU  |  {meta.short_time}"
            )
        note = notes.get(sid, "").strip()
        if note:
            row += f"  |  {note}"
        body_lines.append((row, {"fontsize": 9, "color": _MUTED_COLOR}))

    _add_text_block(ax, body_lines, start_y=y, line_height=0.026)
    return fig


def render_conclusion_page(
    *,
    title: str,
    generated_at: datetime,
    base_dir: str,
    session_ids: list[str],
    settings: ViewSettings,
    rendered: list[ViewRenderResult],
    view_display_names: list[str],
) -> plt.Figure:
    """Build the report conclusion page as a matplotlib figure."""
    fig = _new_portrait_figure()
    ax = fig.axes[0]

    success_count = sum(1 for r in rendered if r.success)
    skipped = [r for r in rendered if not r.success]

    lines: list[tuple[str, dict]] = [
        ("Conclusion", {"fontsize": 20, "fontweight": "bold", "color": _TITLE_COLOR}),
        (
            f'Report "{title}" completed on '
            f"{generated_at.strftime('%B %d, %Y at %H:%M')}.",
            {"fontsize": 10, "color": _MUTED_COLOR},
        ),
        ("", {}),
        ("Summary", {"fontsize": 12, "fontweight": "bold", "color": _TITLE_COLOR}),
        (
            f"{success_count} of {len(rendered)} views rendered successfully.",
            {"fontsize": 10, "color": _MUTED_COLOR},
        ),
        (
            f"{len(session_ids)} session(s), data directory: {base_dir}",
            {"fontsize": 9, "color": _MUTED_COLOR},
        ),
        (
            (
                f"BG subtraction: {'On' if settings.bg_subtract else 'Off'}; "
                f"Calibration: {_CAL_LABELS.get(settings.calibration_mode, settings.calibration_mode)}"
            ),
            {"fontsize": 9, "color": _MUTED_COLOR},
        ),
        ("", {}),
        ("Views included", {"fontsize": 12, "fontweight": "bold", "color": _TITLE_COLOR}),
    ]
    for name in view_display_names:
        lines.append((f"  • {name}", {"fontsize": 9, "color": _MUTED_COLOR}))

    if skipped:
        lines.extend([
            ("", {}),
            ("Skipped views", {"fontsize": 12, "fontweight": "bold", "color": _TITLE_COLOR}),
        ])
        for result in skipped:
            reason = result.skip_reason or "No figure produced"
            lines.append(
                (f"  • {result.display_name}: {reason}", {"fontsize": 9, "color": _MUTED_COLOR}),
            )

    lines.extend([
        ("", {}),
        ("About Scan Kit", {"fontsize": 12, "fontweight": "bold", "color": _TITLE_COLOR}),
        (
            f"Generated with Scan Kit v{__version__}",
            {"fontsize": 10, "color": _ACCENT_COLOR, "fontweight": "bold"},
        ),
        (
            "Open-source proton pencil beam scanning session analysis toolkit.",
            {"fontsize": 9, "color": _MUTED_COLOR},
        ),
        (GITHUB_URL, {"fontsize": 9, "color": _ACCENT_COLOR}),
        (
            "Download releases and documentation on GitHub.",
            {"fontsize": 9, "color": _MUTED_COLOR},
        ),
    ])

    _add_text_block(ax, lines, start_y=0.90, line_height=0.026)
    return fig
