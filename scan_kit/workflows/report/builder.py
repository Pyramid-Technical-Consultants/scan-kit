"""Assemble multi-page PDF reports from analysis views."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from scan_kit import __version__
from scan_kit.common.plotting import apply_tight_layout
from scan_kit.common.report_runner import capture_view_figure

from .pages import render_conclusion_page, render_title_page
from .types import ReportConfig, ViewRenderResult

_VIEW_WIDTH_PX = 1920
_VIEW_HEIGHT_PX = 1080
_VIEW_DPI = 100
_VIEW_FIGSIZE = (_VIEW_WIDTH_PX / _VIEW_DPI, _VIEW_HEIGHT_PX / _VIEW_DPI)
_LANDSCAPE_PAGE = (16.0, 9.0)
_SKIP_NO_DATA = "No data available for selected sessions"


def _prepare_view_figure(fig: plt.Figure) -> None:
    """Resize to a 1920×1080 (16:9) canvas and apply tight layout."""
    fig.set_size_inches(*_VIEW_FIGSIZE, forward=True)
    apply_tight_layout(fig)


def _rasterize_figure(fig: plt.Figure) -> np.ndarray:
    """Render *fig* to a 1920×1080 PNG array for PDF embedding."""
    _prepare_view_figure(fig)
    buf = BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=_VIEW_DPI,
        facecolor=fig.get_facecolor(),
        edgecolor="none",
        pil_kwargs={"compress_level": 3},
    )
    buf.seek(0)
    image = plt.imread(buf)
    if image.shape[1] != _VIEW_WIDTH_PX or image.shape[0] != _VIEW_HEIGHT_PX:
        raise RuntimeError(
            f"Expected {_VIEW_WIDTH_PX}×{_VIEW_HEIGHT_PX} plot image, "
            f"got {image.shape[1]}×{image.shape[0]}"
        )
    return image


def _save_raster_page(
    image: np.ndarray,
    pdf: PdfPages,
    *,
    landscape: bool,
) -> None:
    """Place a raster image on a PDF page (avoids vectorizing dense plots)."""
    page_size = _LANDSCAPE_PAGE if landscape else _PORTRAIT_PAGE
    page_fig = plt.figure(figsize=page_size, dpi=150)
    page_fig.patch.set_facecolor("white")
    ax = page_fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.imshow(image, interpolation="lanczos", resample=True, aspect="auto")
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)
    page_fig.savefig(
        pdf,
        format="pdf",
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(page_fig)


def _save_figure_to_pdf(fig: plt.Figure, pdf: PdfPages, *, landscape: bool) -> None:
    image = _rasterize_figure(fig)
    _save_raster_page(image, pdf, landscape=landscape)


def _save_landscape_page(fig: plt.Figure, pdf: PdfPages) -> None:
    """Save a landscape document page (title, conclusion) directly to the PDF."""
    fig.set_size_inches(*_LANDSCAPE_PAGE, forward=True)
    fig.savefig(
        pdf,
        format="pdf",
        dpi=150,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
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
                step += 1
                continue

            fig = capture_view_figure(
                view_func,
                config.session_ids,
                config.base_dir,
                config.settings,
            )
            if fig is None:
                result.skip_reason = _SKIP_NO_DATA
                rendered.append(result)
                step += 1
                continue

            try:
                _save_figure_to_pdf(fig, pdf, landscape=True)
                result.success = True
            except Exception as exc:
                result.skip_reason = f"Failed to save figure: {exc}"
            finally:
                plt.close(fig)
            rendered.append(result)
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
