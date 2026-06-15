"""Assemble multi-page PDF reports from analysis views."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from scan_kit import __version__
from scan_kit.common.plotting import relayout_view_for_pdf_export
from scan_kit.common.report_runner import capture_view_figure

from .pages import render_conclusion_page, render_title_page
from .types import ReportConfig, ViewRenderResult

_LANDSCAPE_PAGE = (16.0, 9.0)
# Vector PDF at print resolution.
_OUTPUT_DPI = 300
_SKIP_NO_DATA = "No data available for selected sessions"
_log = logging.getLogger(__name__)


def _log_skipped_view(result: ViewRenderResult) -> None:
    reason = result.skip_reason or "skipped"
    _log.warning(
        "Report skipped view %s (%s): %s",
        result.display_name,
        result.module_name,
        reason,
    )


def _prepare_view_figure(fig: plt.Figure) -> None:
    """Resize for the PDF page and re-layout the view header at the new size."""
    relayout_view_for_pdf_export(fig, dpi=_OUTPUT_DPI, page_inches=_LANDSCAPE_PAGE)


def _save_figure_page(fig: plt.Figure, pdf: PdfPages) -> None:
    """Append a matplotlib figure to the PDF as vector artwork."""
    fig.savefig(
        pdf,
        format="pdf",
        dpi=_OUTPUT_DPI,
        facecolor=fig.get_facecolor(),
        edgecolor="none",
    )


def _save_figure_to_pdf(fig: plt.Figure, pdf: PdfPages, *, landscape: bool) -> None:
    if not landscape:
        raise ValueError("Report view pages are landscape only")
    _prepare_view_figure(fig)
    _save_figure_page(fig, pdf)


def _save_landscape_page(fig: plt.Figure, pdf: PdfPages) -> None:
    """Save a landscape document page (title, conclusion) directly to the PDF."""
    fig.set_size_inches(*_LANDSCAPE_PAGE, forward=True)
    fig.set_dpi(_OUTPUT_DPI)
    fig.savefig(
        pdf,
        format="pdf",
        dpi=_OUTPUT_DPI,
        facecolor=fig.get_facecolor(),
        edgecolor="none",
    )


def build_report_pdf(
    config: ReportConfig,
    *,
    progress: Callable[[int, str], None] | None = None,
) -> tuple[Path, list[ViewRenderResult]]:
    """Generate a PDF report and return the output path and per-view results."""
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rendered: list[ViewRenderResult] = []
    total_steps = len(config.views) + 2
    step = 0

    def _emit(message: str) -> None:
        nonlocal step
        if progress is not None:
            pct = int(100 * step / max(total_steps, 1))
            progress(pct, message)

    with PdfPages(output_path) as pdf:
        info = pdf.infodict()
        info["Title"] = config.title
        author_parts = [
            part.strip()
            for part in (config.author, config.organization)
            if part.strip()
        ]
        info["Author"] = ", ".join(author_parts) or "Scan Kit"
        info["Creator"] = f"Scan Kit {__version__}"
        info["Subject"] = config.subtitle

        _emit("Rendering title page…")
        title_fig = render_title_page(
            title=config.title,
            subtitle=config.subtitle,
            author=config.author,
            organization=config.organization,
            generated_at=config.generated_at,
            session_ids=config.session_ids,
            session_meta=config.session_meta,
            notes=config.notes,
            settings=config.settings,
        )
        _save_landscape_page(title_fig, pdf)
        plt.close(title_fig)
        step += 1

        for display_name, module_name, _description in config.views:
            _emit(f"Rendering {display_name}…")
            result = ViewRenderResult(
                display_name=display_name,
                module_name=module_name,
                success=False,
            )
            try:
                module = importlib.import_module(f"scan_kit.views.{module_name}")
                view_func = module.run
            except Exception as exc:
                result.skip_reason = f"Failed to load view: {exc}"
                rendered.append(result)
                _log_skipped_view(result)
                step += 1
                continue

            fig, capture_skip = capture_view_figure(
                view_func,
                config.session_ids,
                config.base_dir,
                config.settings,
            )
            if fig is None:
                result.skip_reason = capture_skip or _SKIP_NO_DATA
                rendered.append(result)
                _log_skipped_view(result)
                step += 1
                continue

            try:
                _save_figure_to_pdf(fig, pdf, landscape=True)
                result.success = True
            except Exception as exc:
                result.skip_reason = f"Failed to save figure: {exc}"
                _log.exception(
                    "Report failed saving view %s (%s)",
                    display_name,
                    module_name,
                )
            finally:
                plt.close(fig)
            rendered.append(result)
            if not result.success:
                _log_skipped_view(result)
            step += 1

        _emit("Rendering conclusion page…")
        conclusion_fig = render_conclusion_page(
            title=config.title,
            author=config.author,
            organization=config.organization,
            generated_at=config.generated_at,
            session_ids=config.session_ids,
            settings=config.settings,
            rendered=rendered,
            views=config.views,
        )
        _save_landscape_page(conclusion_fig, pdf)
        plt.close(conclusion_fig)
        step += 1

    if progress is not None:
        progress(100, "Complete")

    return output_path, rendered
