"""Reusable Qt shells for analysis views with optional side controls.

Matplotlib views subclass :class:`PlotViewWindow`. visPy views subclass
:class:`VispyViewWindow` and add canvases with :meth:`VispyViewWindow.add_vispy_canvas`.
2D visPy drawing primitives live in :mod:`scan_kit.views.vispy_plot`.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QMenu,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..common.app_icon import apply_qt_application_branding, prepare_qt_app_identity
from ..common.qt_theme import apply_saved_ui_theme
from ..common.view_runner import _READY_SENTINEL

_DEFAULT_SIDE_MIN = 220
_DEFAULT_SIDE_WIDTH = 280


def new_headless_figure(figsize: tuple[float, float]) -> Figure:
    """Matplotlib figure with an Agg canvas for off-thread rendering."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig = Figure(figsize=figsize, layout="none")
    FigureCanvasAgg(fig)
    return fig


class SidePanelWindow(QMainWindow):
    """Plot host + optional right-hand controls (and optional footer bar)."""

    def __init__(
        self,
        *,
        title: str,
        plot_host: QWidget,
        side_panel_min_width: int = _DEFAULT_SIDE_MIN,
        side_panel_default_width: int = _DEFAULT_SIDE_WIDTH,
        parent: QWidget | None = None,
        with_footer: bool = False,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1400, 900)
        self._side_min_width = side_panel_min_width
        self._side_default_width = side_panel_default_width
        self._mount_side_shell(plot_host, with_footer=with_footer)

    def _mount_side_shell(self, plot_host: QWidget, *, with_footer: bool) -> None:
        self._side_scroll = QScrollArea()
        self._side_scroll.setWidgetResizable(True)
        self._side_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
        )
        self._side_scroll.setMinimumWidth(self._side_min_width)
        self._side_scroll.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Expanding,
        )
        self._side_scroll.hide()

        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter.setChildrenCollapsible(False)
        self._splitter.setHandleWidth(6)
        self._splitter.addWidget(plot_host)
        self._splitter.addWidget(self._side_scroll)
        self._splitter.setStretchFactor(0, 1)
        self._splitter.setStretchFactor(1, 0)

        if not with_footer:
            self._footer_host = None
            self.setCentralWidget(self._splitter)
            return

        self._footer_host = QWidget()
        footer_layout = QVBoxLayout(self._footer_host)
        footer_layout.setContentsMargins(0, 0, 0, 0)
        footer_layout.setSpacing(0)
        self._footer_host.hide()
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        root_layout.addWidget(self._splitter, 1)
        root_layout.addWidget(self._footer_host)
        self.setCentralWidget(root)

    @property
    def side_panel(self) -> QWidget | None:
        return self._side_scroll.widget()

    def set_side_panel(self, widget: QWidget | None) -> None:
        """Attach or clear the right-hand controls panel (splitter-resizable)."""
        existing = self._side_scroll.takeWidget()
        if existing is not None:
            existing.setParent(None)
            existing.deleteLater()

        if widget is None:
            self._side_scroll.hide()
            return

        widget.setMinimumWidth(self._side_min_width)
        self._side_scroll.setWidget(widget)
        self._side_scroll.show()
        total = max(self.width(), 1000)
        side = min(
            max(self._side_default_width, self._side_min_width),
            max(self._side_min_width, total // 4),
        )
        self._splitter.setSizes([max(total - side, 400), side])

    def set_footer(self, widget: QWidget | None) -> None:
        """Attach or clear a full-width bar under the plot (transport, status)."""
        if self._footer_host is None:
            raise RuntimeError("this window was created without a footer slot")
        layout = self._footer_host.layout()
        while layout.count():
            item = layout.takeAt(0)
            child = item.widget()
            if child is not None:
                child.setParent(None)
        if widget is None:
            self._footer_host.hide()
            return
        layout.addWidget(widget)
        self._footer_host.show()


class PlotViewWindow(SidePanelWindow):
    """Matplotlib canvas + toolbar with an optional draggable side panel."""

    def __init__(
        self,
        *,
        title: str,
        figsize: tuple[float, float] = (16, 9),
        side_panel_min_width: int = _DEFAULT_SIDE_MIN,
        side_panel_default_width: int = _DEFAULT_SIDE_WIDTH,
        parent: QWidget | None = None,
    ) -> None:
        self.figure = Figure(figsize=figsize, layout="none")
        self.canvas = FigureCanvasQTAgg(self.figure)
        # QMainWindow is not ready yet; toolbar parent is set after super().
        plot_host = QWidget()
        plot_layout = QVBoxLayout(plot_host)
        plot_layout.setContentsMargins(6, 6, 0, 6)
        plot_layout.setSpacing(2)
        super().__init__(
            title=title,
            plot_host=plot_host,
            side_panel_min_width=side_panel_min_width,
            side_panel_default_width=side_panel_default_width,
            parent=parent,
        )
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas, stretch=1)

    def draw_idle(self) -> None:
        self.canvas.draw_idle()


class VispyViewWindow(SidePanelWindow):
    """visPy canvas host with the same side-panel splitter as PlotViewWindow.

    Subclass, then :meth:`add_vispy_canvas` for each plot (stacked 2D panes or
    a single 3D view). Attach controls with :meth:`set_side_panel`.
    """

    def __init__(
        self,
        *,
        title: str,
        side_panel_min_width: int = _DEFAULT_SIDE_MIN,
        side_panel_default_width: int = _DEFAULT_SIDE_WIDTH,
        parent: QWidget | None = None,
    ) -> None:
        self._plot_host = QWidget()
        self._plot_layout = QVBoxLayout(self._plot_host)
        self._plot_layout.setContentsMargins(6, 6, 0, 0)
        self._plot_layout.setSpacing(4)
        super().__init__(
            title=title,
            plot_host=self._plot_host,
            side_panel_min_width=side_panel_min_width,
            side_panel_default_width=side_panel_default_width,
            parent=parent,
            with_footer=True,
        )

    def add_vispy_canvas(
        self,
        *,
        keys=None,
        size: tuple[int, int] = (1200, 800),
        bgcolor: str | None = None,
        stretch: int = 1,
        min_height: int | None = None,
        max_height: int | None = None,
        block_wheel: bool = False,
    ):
        from .vispy_plot import BG, block_canvas_navigation, make_scene_canvas

        canvas = make_scene_canvas(
            keys=keys,
            bgcolor=BG if bgcolor is None else bgcolor,
            size=size,
        )
        native = canvas.native
        if min_height is not None:
            native.setMinimumHeight(min_height)
        if max_height is not None:
            native.setMaximumHeight(max_height)
        self._plot_layout.addWidget(native, stretch)
        if block_wheel:
            block_canvas_navigation(canvas)
        return canvas


def make_side_panel_column(*, margins: tuple[int, int, int, int] = (8, 8, 8, 8)) -> tuple[QWidget, QVBoxLayout]:
    """Return an empty vertical column suitable for :meth:`PlotViewWindow.set_side_panel`."""
    panel = QWidget()
    layout = QVBoxLayout(panel)
    layout.setContentsMargins(*margins)
    layout.setSpacing(8)
    return panel, layout


def make_presets_menu_button(
    presets: Sequence[tuple[str, str, bool]],
    on_selected: Callable[[str], None],
    *,
    text: str = "Presets",
    parent: QWidget | None = None,
) -> QToolButton:
    """Build a full-width menu button listing presets with readable labels.

    *presets* entries are ``(preset_id, label, enabled)``.
    """
    button = QToolButton(parent)
    button.setText(text)
    button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
    button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
    button.setSizePolicy(
        QSizePolicy.Policy.Expanding,
        QSizePolicy.Policy.Fixed,
    )
    menu = QMenu(button)
    for preset_id, label, enabled in presets:
        enabled_bool = bool(enabled)
        action = QAction(str(label), menu)
        action.setEnabled(enabled_bool)
        action.triggered.connect(
            lambda _checked=False, pid=str(preset_id): on_selected(pid)
        )
        menu.addAction(action)
    button.setMenu(menu)
    button.setEnabled(any(bool(e) for _pid, _label, e in presets))
    return button


def run_view_window(
    build_window: Callable[[], QMainWindow],
    *,
    maximize: bool = True,
) -> None:
    """Create a view window, print the ready sentinel, and run the Qt event loop.

    *build_window* is called after ``QApplication`` exists so constructors can
    safely create Qt widgets.
    """
    prepare_qt_app_identity()
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    app_icon = apply_qt_application_branding(app)
    apply_saved_ui_theme(app=app)

    window = build_window()
    if not app_icon.isNull():
        window.setWindowIcon(app_icon)
    if maximize:
        window.showMaximized()
    else:
        window.show()
    print(_READY_SENTINEL, flush=True)
    app.exec()
