"""Reusable side-panel controls for unified analysis views."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..common.data_filter import (
    BEAM_STATE_FILTERS,
    DOMAIN_FILTERS,
    FILTER_ALL,
    FILTER_BEAM_BOTH,
    FILTER_BEAM_ON,
    DataFilterSelection,
    default_data_filter_selection,
)
from ..common.segmented_control import SegmentedControl
from .unified_catalog import (
    DATA_SOURCE_SPOT,
    DATA_SOURCE_TIMESLICE,
    REFERENCE_CHAMBER,
    REFERENCE_ISO,
    REFERENCE_OPTIONS,
    DataSourceKind,
    ReferenceFrameKind,
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

PlotStyleChoice = tuple[str, str]


class PlotStylePanel(QWidget):
    """Plot-style segmented control with style-specific options in one fieldset."""

    def __init__(
        self,
        styles: Sequence[PlotStyleChoice],
        parent: QWidget | None = None,
        *,
        current: str | None = None,
        on_selection_changed: Callable[[], None] | None = None,
        group_title: str = "Plot Style",
    ) -> None:
        super().__init__(parent)
        self._on_selection_changed = on_selection_changed
        self._styles = tuple(styles)
        self._checkboxes: dict[str, QCheckBox] = {}
        self._spin_rows: dict[str, tuple[QWidget, QSpinBox]] = {}
        self._spin_debounce_timers: dict[str, QTimer] = {}
        self._spin_debounce_ms = 250

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._group_box = QGroupBox(group_title)
        self._layout = QVBoxLayout(self._group_box)
        self._segmented = SegmentedControl(list(self._styles))
        self._segmented.selectionChanged.connect(self._on_segment_changed)
        self._layout.addWidget(self._segmented)
        root.addWidget(self._group_box)

        keys = [key for key, _label in self._styles]
        pick = current if current in keys else (keys[0] if keys else None)
        if pick is not None:
            self._segmented.set_current(pick)

    def selected_key(self) -> str | None:
        key = self._segmented.current_key()
        return key or None

    def set_current(self, key: str) -> None:
        self._segmented.set_current(key)

    def set_enabled(self, enabled: bool) -> None:
        self._group_box.setEnabled(enabled)

    def add_checkbox(
        self,
        option_id: str,
        label: str,
        *,
        checked: bool = False,
    ) -> QCheckBox:
        box = QCheckBox(label)
        box.setChecked(checked)
        box.toggled.connect(self._emit_changed)
        self._layout.addWidget(box)
        self._checkboxes[option_id] = box
        return box

    def add_percent_spinbox(
        self,
        option_id: str,
        label: str,
        *,
        value: int = 5,
        minimum: int = 0,
        maximum: int = 90,
    ) -> QSpinBox:
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        label_widget = QLabel(label)
        label_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        row_layout.addWidget(label_widget, stretch=1)
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSuffix("%")
        spin.setValue(value)
        timer = QTimer(self)
        timer.setSingleShot(True)
        timer.setInterval(self._spin_debounce_ms)
        timer.timeout.connect(self._emit_changed)
        spin.valueChanged.connect(lambda _value, t=timer: t.start())
        row_layout.addWidget(spin)
        self._layout.addWidget(row)
        self._spin_rows[option_id] = (row, spin)
        self._spin_debounce_timers[option_id] = timer
        return spin

    def is_checked(self, option_id: str) -> bool:
        box = self._checkboxes.get(option_id)
        return box.isChecked() if box is not None else False

    def set_checked(self, option_id: str, checked: bool) -> None:
        box = self._checkboxes.get(option_id)
        if box is not None:
            box.setChecked(checked)

    def spin_value(self, option_id: str) -> int | None:
        row = self._spin_rows.get(option_id)
        return row[1].value() if row is not None else None

    def set_spin_value(self, option_id: str, value: int) -> None:
        row = self._spin_rows.get(option_id)
        if row is not None:
            spin = row[1]
            spin.blockSignals(True)
            try:
                spin.setValue(value)
            finally:
                spin.blockSignals(False)
            timer = self._spin_debounce_timers.get(option_id)
            if timer is not None:
                timer.stop()

    def set_option_visible(self, option_id: str, visible: bool) -> None:
        box = self._checkboxes.get(option_id)
        if box is not None:
            box.setVisible(visible)
        row = self._spin_rows.get(option_id)
        if row is not None:
            row[0].setVisible(visible)

    def _on_segment_changed(self, _key: str) -> None:
        self._emit_changed()

    def _emit_changed(self, *_args) -> None:
        if self._on_selection_changed is not None:
            self._on_selection_changed()


class DataFilterPanel(QWidget):
    """Domain and beam-state filter selectors that combine independently."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        domain_current: str | None = None,
        beam_current: str | None = None,
        on_selection_changed: Callable[[], None] | None = None,
        group_title: str = "Filter Data",
    ) -> None:
        super().__init__(parent)
        self._on_selection_changed = on_selection_changed
        self._domain_keys = {key for key, _label in DOMAIN_FILTERS}
        self._beam_keys = {key for key, _label in BEAM_STATE_FILTERS}

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._group_box = QGroupBox(group_title)
        group_layout = QVBoxLayout(self._group_box)

        domain_row = QHBoxLayout()
        domain_row.addWidget(QLabel("Domain"))
        self._domain_combo = QComboBox()
        for key, label in DOMAIN_FILTERS:
            self._domain_combo.addItem(label, key)
        self._domain_combo.currentIndexChanged.connect(self._on_index_changed)
        domain_row.addWidget(self._domain_combo, 1)
        group_layout.addLayout(domain_row)

        beam_row = QHBoxLayout()
        self._beam_label = QLabel("Beam")
        beam_row.addWidget(self._beam_label)
        self._beam_combo = QComboBox()
        for key, label in BEAM_STATE_FILTERS:
            self._beam_combo.addItem(label, key)
        self._beam_combo.currentIndexChanged.connect(self._on_index_changed)
        beam_row.addWidget(self._beam_combo, 1)
        group_layout.addLayout(beam_row)

        root.addWidget(self._group_box)

        domain_pick = (
            domain_current
            if domain_current in self._domain_keys
            else DOMAIN_FILTERS[0][0]
        )
        beam_pick = (
            beam_current
            if beam_current in self._beam_keys
            else BEAM_STATE_FILTERS[0][0]
        )
        self.set_domain(domain_pick)
        self.set_beam_state(beam_pick)

    def selection(self) -> DataFilterSelection:
        return DataFilterSelection(
            domain_filter=self.selected_domain() or FILTER_ALL,
            beam_state_filter=self.selected_beam_state() or FILTER_BEAM_BOTH,
        )

    def selected_domain(self) -> str | None:
        data = self._domain_combo.currentData()
        return str(data) if data is not None else None

    def selected_beam_state(self) -> str | None:
        data = self._beam_combo.currentData()
        return str(data) if data is not None else None

    def selected_key(self) -> str | None:
        """Legacy alias: returns domain filter only."""
        return self.selected_domain()

    def set_domain(self, key: str) -> None:
        idx = self._domain_combo.findData(key)
        if idx >= 0:
            self._domain_combo.setCurrentIndex(idx)

    def set_beam_state(self, key: str) -> None:
        idx = self._beam_combo.findData(key)
        if idx >= 0:
            self._beam_combo.setCurrentIndex(idx)

    def set_current(self, key: str) -> None:
        """Legacy alias: sets domain filter only."""
        self.set_domain(key)

    def set_enabled(self, enabled: bool) -> None:
        self._group_box.setEnabled(enabled)

    def set_beam_enabled(self, enabled: bool) -> None:
        self._beam_label.setEnabled(enabled)
        self._beam_combo.setEnabled(enabled)

    def _on_index_changed(self, _index: int) -> None:
        if self._on_selection_changed is not None:
            self._on_selection_changed()


class ReferenceFramePanel(QWidget):
    """Isocenter vs raw chamber-plane reference for position and sigma metrics."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        current: ReferenceFrameKind = REFERENCE_ISO,
        on_selection_changed: Callable[[], None] | None = None,
        group_title: str = "Reference Frame",
    ) -> None:
        super().__init__(parent)
        self._on_selection_changed = on_selection_changed

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._group_box = QGroupBox(group_title)
        layout = QVBoxLayout(self._group_box)
        self._segmented = SegmentedControl(list(REFERENCE_OPTIONS))
        self._segmented.selectionChanged.connect(self._on_segment_changed)
        layout.addWidget(self._segmented)
        root.addWidget(self._group_box)

        self.set_current(current)

    def selected_key(self) -> ReferenceFrameKind:
        key = self._segmented.current_key()
        return key if key in (REFERENCE_ISO, REFERENCE_CHAMBER) else REFERENCE_ISO  # type: ignore[return-value]

    def set_current(self, key: ReferenceFrameKind) -> None:
        self._segmented.set_current(key)

    def set_enabled(self, enabled: bool) -> None:
        self._group_box.setEnabled(enabled)

    def _on_segment_changed(self, _key: str) -> None:
        if self._on_selection_changed is not None:
            self._on_selection_changed()


def sync_data_filter_panel(
    panel: DataFilterPanel | None,
    *,
    supports_filter: bool,
    has_beam_state: bool,
    reset_defaults: bool = False,
) -> None:
    """Update filter panel enabled state; optionally reset domain/beam to defaults."""
    if panel is None:
        return
    panel.set_enabled(supports_filter)
    panel.set_beam_enabled(supports_filter and has_beam_state)
    if reset_defaults:
        defaults = default_data_filter_selection(has_beam_state=has_beam_state)
        panel.set_domain(defaults.domain_filter)
        panel.set_beam_state(defaults.beam_state_filter)


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

    def update_availability(self, availability: dict[str, bool]) -> None:
        """Refresh enabled options without changing the current source/selection."""
        self._availability = dict(availability)
        spot_ok = source_has_available_options(
            self._options, self._availability, DATA_SOURCE_SPOT,
        )
        timeslice_ok = source_has_available_options(
            self._options, self._availability, DATA_SOURCE_TIMESLICE,
        )
        self._source_segmented.set_option_enabled(DATA_SOURCE_SPOT, spot_ok)
        self._source_segmented.set_option_enabled(DATA_SOURCE_TIMESLICE, timeslice_ok)
        self._source_segmented.setVisible(spot_ok and timeslice_ok)
        current = self._selected_id()
        self._refresh_option_list(select_id=current)

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
