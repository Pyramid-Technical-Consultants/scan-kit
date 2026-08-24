"""Reusable side-panel controls for unified analysis views."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGroupBox,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..common.segmented_control import SegmentedControl
from .unified_catalog import (
    DATA_SOURCE_SPOT,
    DATA_SOURCE_TIMESLICE,
    DataSourceKind,
    UnifiedViewOption,
    default_option_id,
    default_source,
    is_option_available,
    option_for,
    options_for_source,
    source_has_available_options,
)

_SOURCE_OPTIONS = [
    (DATA_SOURCE_SPOT, "Spot"),
    (DATA_SOURCE_TIMESLICE, "Timeslice"),
]


class DataSourceOptionPanel(QWidget):
    """Spot/timeslice source toggle and option list in one fieldset."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        on_selection_changed: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self._on_selection_changed = on_selection_changed
        self._options: tuple[UnifiedViewOption, ...] = ()
        self._availability: dict[str, bool] = {}
        self._updating = False

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._group_box = QGroupBox("Analysis")
        group_layout = QVBoxLayout(self._group_box)
        group_layout.setSpacing(8)

        self._source_segmented = SegmentedControl(_SOURCE_OPTIONS)
        self._source_segmented.selectionChanged.connect(self._on_source_segment_changed)
        group_layout.addWidget(self._source_segmented)

        self._option_list = QListWidget()
        self._option_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self._option_list.currentItemChanged.connect(self._on_option_changed)
        group_layout.addWidget(self._option_list)

        root.addWidget(self._group_box)

    def configure(
        self,
        options: Sequence[UnifiedViewOption],
        availability: dict[str, bool],
        *,
        group_title: str = "Analysis",
        preferred_source: DataSourceKind = DATA_SOURCE_SPOT,
    ) -> None:
        self._options = tuple(options)
        self._availability = dict(availability)
        self._group_box.setTitle(group_title)

        spot_ok = source_has_available_options(
            self._options, self._availability, DATA_SOURCE_SPOT,
        )
        timeslice_ok = source_has_available_options(
            self._options, self._availability, DATA_SOURCE_TIMESLICE,
        )
        self._source_segmented.set_option_enabled(DATA_SOURCE_SPOT, spot_ok)
        self._source_segmented.set_option_enabled(DATA_SOURCE_TIMESLICE, timeslice_ok)
        self._source_segmented.setVisible(spot_ok and timeslice_ok)

        if source_has_available_options(
            self._options, self._availability, preferred_source,
        ):
            source = preferred_source
        else:
            source = default_source(self._options, self._availability)
        self._set_source(source, refresh_list=False)
        first_id = default_option_id(
            self._options,
            self._availability,
            source=source,
        )
        self._refresh_option_list(select_id=first_id)

    def selected_source(self) -> DataSourceKind:
        key = self._source_segmented.current_key()
        if key == DATA_SOURCE_TIMESLICE:
            return DATA_SOURCE_TIMESLICE
        return DATA_SOURCE_SPOT

    def selected_id(self) -> str | None:
        item = self._option_list.currentItem()
        if item is None:
            return None
        return str(item.data(Qt.ItemDataRole.UserRole))

    def select_id(self, option_id: str, *, source: DataSourceKind | None = None) -> None:
        src = source
        if src is None:
            match = option_for(self._options, option_id, source=self.selected_source())
            if match is None:
                match = next(
                    (opt for opt in self._options if opt.id == option_id),
                    None,
                )
            if match is None:
                return
            src = match.source
        elif option_for(self._options, option_id, source=src) is None:
            return
        self._set_source(src, refresh_list=False)
        self._refresh_option_list(select_id=option_id)

    def _set_source(self, source: DataSourceKind, *, refresh_list: bool) -> None:
        self._updating = True
        try:
            self._source_segmented.set_current(source)
        finally:
            self._updating = False
        if refresh_list:
            current = self._selected_id()
            self._refresh_option_list(select_id=current)

    def _selected_id(self) -> str | None:
        return self.selected_id()

    def _refresh_option_list(self, *, select_id: str | None) -> None:
        self._updating = True
        try:
            self._option_list.clear()
            source = self.selected_source()
            visible = options_for_source(self._options, source)
            selected_row = -1
            for row, opt in enumerate(visible):
                item = QListWidgetItem(opt.label)
                item.setData(Qt.ItemDataRole.UserRole, opt.id)
                enabled = is_option_available(self._availability, opt)
                if enabled:
                    item.setFlags(
                        Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled,
                    )
                else:
                    item.setFlags(Qt.ItemFlag.NoItemFlags)
                self._option_list.addItem(item)
                if select_id is not None and opt.id == select_id and enabled:
                    selected_row = row

            if selected_row < 0:
                for row, opt in enumerate(visible):
                    if is_option_available(self._availability, opt):
                        selected_row = row
                        break

            if selected_row >= 0:
                self._option_list.setCurrentRow(selected_row)
            self._option_list.setEnabled(bool(visible))
        finally:
            self._updating = False

    def _on_source_segment_changed(self, _key: str) -> None:
        if self._updating:
            return
        preserve_id = self.selected_id()
        self._refresh_option_list(select_id=preserve_id)
        self._emit_selection_changed()

    def _on_option_changed(self, _current, _previous) -> None:
        if self._updating:
            return
        self._emit_selection_changed()

    def _emit_selection_changed(self) -> None:
        if self._on_selection_changed is not None:
            self._on_selection_changed()
