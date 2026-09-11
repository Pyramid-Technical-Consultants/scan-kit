"""Light / dark / system color scheme for Qt windows."""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QAction, QActionGroup, QIcon, QPainter, QPalette, QPixmap
from PySide6.QtWidgets import QApplication, QMenu, QStyle, QWidget

from .app_settings import AppSettings, normalize_ui_theme

UI_THEME_LABELS: tuple[tuple[str, str], ...] = (
    ("system", "System"),
    ("light", "Light"),
    ("dark", "Dark"),
)

_SCHEME = {
    "light": Qt.ColorScheme.Light,
    "dark": Qt.ColorScheme.Dark,
    "system": Qt.ColorScheme.Unknown,
}


def apply_ui_theme(theme: str, *, app: QApplication | None = None) -> None:
    """Apply *theme* to the running Qt application."""
    instance = app or QApplication.instance()
    if instance is None:
        return
    scheme = _SCHEME[normalize_ui_theme(theme)]
    instance.styleHints().setColorScheme(scheme)


def apply_saved_ui_theme(*, app: QApplication | None = None) -> None:
    apply_ui_theme(AppSettings.load().ui_theme, app=app)


def persist_ui_theme(theme: str, *, settings: AppSettings | None = None) -> None:
    """Save *theme* and apply it immediately."""
    theme = normalize_ui_theme(theme)
    loaded = settings
    if loaded is None:
        loaded = AppSettings.load()
    changed = loaded.ui_theme != theme
    loaded.ui_theme = theme
    if changed:
        loaded.save()
    apply_ui_theme(theme)


def add_theme_menu(
    window: QWidget,
    menu: QMenu,
    *,
    settings: AppSettings | None = None,
    on_applied: Callable[[], None] | None = None,
) -> QActionGroup:
    """Fill *menu* with exclusive System / Light / Dark actions."""
    group = QActionGroup(window)
    group.setExclusive(True)
    current = normalize_ui_theme(
        (settings or AppSettings.load()).ui_theme
    )
    for key, label in UI_THEME_LABELS:
        action = QAction(label, window)
        action.setCheckable(True)
        action.setData(key)
        action.setChecked(key == current)
        action.setStatusTip(f"Use the {label.lower()} color scheme")
        group.addAction(action)
        menu.addAction(action)
    # QActionGroup.triggered fires once for the clicked action. Connecting each
    # QAction.triggered would also run for the action that was unchecked.
    group.triggered.connect(
        lambda action: _choose_theme(
            str(action.data()), settings=settings, on_applied=on_applied,
        )
    )
    return group


def tinted_standard_icon(
    widget: QWidget,
    pixmap: QStyle.StandardPixmap,
) -> QIcon:
    """Standard pixmap recolored to the widget's button text (dark-theme safe)."""
    style = widget.style()
    size = style.pixelMetric(QStyle.PixelMetric.PM_ButtonIconSize, None, widget)
    if size < 8:
        size = 16
    ratio = max(1.0, float(widget.devicePixelRatioF()))
    pixel = max(1, int(round(size * ratio)))
    src = style.standardIcon(pixmap, None, widget).pixmap(QSize(pixel, pixel))
    if src.isNull():
        return style.standardIcon(pixmap, None, widget)
    src.setDevicePixelRatio(ratio)
    color = widget.palette().color(QPalette.ColorRole.ButtonText)
    tinted = QPixmap(src.size())
    tinted.setDevicePixelRatio(ratio)
    tinted.fill(Qt.GlobalColor.transparent)
    painter = QPainter(tinted)
    painter.drawPixmap(0, 0, src)
    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
    painter.fillRect(tinted.rect(), color)
    painter.end()
    icon = QIcon()
    icon.addPixmap(tinted)
    return icon


def _choose_theme(
    theme: str,
    *,
    settings: AppSettings | None,
    on_applied: Callable[[], None] | None,
) -> None:
    persist_ui_theme(theme, settings=settings)
    if on_applied is not None:
        on_applied()
