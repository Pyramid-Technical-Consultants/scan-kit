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
    CoarseDataSourceKind,
    DATA_SOURCE_SPOT_ISO,
    DataSourceKind,
    GRANULARITY_SOURCES,
    UnifiedViewOption,
    coarse_data_source,
    coarse_has_available_options,
    default_coarse_source,
    is_option_available,
    option_for,
    option_from_list_key,
    option_list_key,
    options_for_coarse,
)

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


class HistogramPanel(QWidget):
    """Side-panel controls for optional histogram panels."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        on_selection_changed: Callable[[], None] | None = None,
        group_title: str = "Histogram",
        default_bin_count: int = 30,
        min_bin_count: int = 5,
        max_bin_count: int = 200,
    ) -> None:
        super().__init__(parent)
        self._on_selection_changed = on_selection_changed
        self._bin_debounce = QTimer(self)
        self._bin_debounce.setSingleShot(True)
        self._bin_debounce.setInterval(250)
        self._bin_debounce.timeout.connect(self._emit_changed)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._group_box = QGroupBox(group_title)
        group_layout = QVBoxLayout(self._group_box)

        self._enabled = QCheckBox("Show panel")
        self._enabled.toggled.connect(self._on_enabled_changed)
        group_layout.addWidget(self._enabled)

        self._bins_row = QWidget()
        bins_layout = QHBoxLayout(self._bins_row)
        bins_layout.setContentsMargins(0, 0, 0, 0)
        bins_layout.addWidget(QLabel("Bins"))
        self._bins_spin = QSpinBox()
        self._bins_spin.setRange(min_bin_count, max_bin_count)
        self._bins_spin.setValue(default_bin_count)
        self._bins_spin.valueChanged.connect(
            lambda _value: self._bin_debounce.start(),
        )
        bins_layout.addWidget(self._bins_spin, stretch=1)
        group_layout.addWidget(self._bins_row)

        self._shared_bins = QCheckBox("Share bin edges across rows")
        self._shared_bins.toggled.connect(self._emit_changed)
        group_layout.addWidget(self._shared_bins)

        root.addWidget(self._group_box)
        self._sync_suboptions()

    def is_enabled(self) -> bool:
        return self._enabled.isChecked()

    def set_enabled(self, enabled: bool) -> None:
        self._enabled.setChecked(enabled)

    def bin_count(self) -> int:
        return self._bins_spin.value()

    def set_bin_count(self, value: int) -> None:
        self._bins_spin.blockSignals(True)
        try:
            self._bins_spin.setValue(value)
        finally:
            self._bins_spin.blockSignals(False)
        self._bin_debounce.stop()

    def shared_bins(self) -> bool:
        return self._shared_bins.isChecked()

    def set_shared_bins(self, shared: bool) -> None:
        self._shared_bins.setChecked(shared)

    def set_from_config(
        self,
        *,
        show_hist: bool,
        hist_bin_count: int,
        hist_shared_bins: bool,
    ) -> None:
        self._enabled.setChecked(show_hist)
        self.set_bin_count(hist_bin_count)
        self.set_shared_bins(hist_shared_bins)
        self._sync_suboptions()

    def _on_enabled_changed(self, _checked: bool) -> None:
        self._sync_suboptions()
        self._emit_changed()

    def _sync_suboptions(self) -> None:
        enabled = self._enabled.isChecked()
        self._bins_row.setEnabled(enabled)
        self._shared_bins.setEnabled(enabled)

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
    """Spot/timeslice toggle with a metric list (iso/chamber variants are separate rows)."""

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

        self._granularity_segmented = SegmentedControl(list(GRANULARITY_SOURCES))
        self._granularity_segmented.selectionChanged.connect(
            self._on_granularity_segment_changed,
        )
        group_layout.addWidget(self._granularity_segmented)

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
        preferred_source: DataSourceKind = DATA_SOURCE_SPOT_ISO,
    ) -> None:
        self._options = tuple(options)
        self._availability = dict(availability)
        self._group_box.setTitle(group_title)

        self._update_granularity_enabled()
        preferred_coarse = coarse_data_source(preferred_source)
        if not coarse_has_available_options(
            self._options, self._availability, preferred_coarse,
        ):
            preferred_coarse = default_coarse_source(self._options, self._availability)
        self._set_granularity(preferred_coarse, refresh_list=False)

        first_key = self._default_list_key(coarse=preferred_coarse)
        self._refresh_option_list(select_key=first_key)

    def update_availability(self, availability: dict[str, bool]) -> None:
        """Refresh enabled options without changing the current source/selection."""
        self._availability = dict(availability)
        self._update_granularity_enabled()
        current_key = self._selected_list_key()
        self._refresh_option_list(select_key=current_key)

    def selected_source(self) -> DataSourceKind:
        opt = self._selected_option()
        if opt is not None:
            return opt.source
        coarse = self._selected_granularity()
        for opt in options_for_coarse(self._options, coarse):
            if is_option_available(self._availability, opt):
                return opt.source
        return DATA_SOURCE_SPOT_ISO

    def selected_id(self) -> str | None:
        opt = self._selected_option()
        return opt.id if opt is not None else None

    def select_id(self, option_id: str, *, source: DataSourceKind | None = None) -> None:
        if source is not None:
            match = option_for(self._options, option_id, source=source)
        else:
            match = option_for(
                self._options,
                option_id,
                source=self.selected_source(),
            )
            if match is None:
                match = next(
                    (opt for opt in self._options if opt.id == option_id),
                    None,
                )
        if match is None:
            return
        self._set_granularity(coarse_data_source(match.source), refresh_list=False)
        self._refresh_option_list(select_key=option_list_key(match))

    def _selected_granularity(self) -> CoarseDataSourceKind:
        key = self._granularity_segmented.current_key()
        if key in dict(GRANULARITY_SOURCES):
            return key  # type: ignore[return-value]
        return GRANULARITY_SOURCES[0][0]

    def _set_source(self, source: DataSourceKind, *, refresh_list: bool) -> None:
        coarse = coarse_data_source(source)
        self._set_granularity(coarse, refresh_list=False)
        if refresh_list:
            preserve_id = self.selected_id()
            self._refresh_option_list(
                select_id=preserve_id,
                select_source=source,
            )

    def _set_granularity(
        self,
        coarse: CoarseDataSourceKind,
        *,
        refresh_list: bool,
    ) -> None:
        self._updating = True
        try:
            self._granularity_segmented.set_current(coarse)
        finally:
            self._updating = False
        if refresh_list:
            preserve_id = self.selected_id()
            self._refresh_option_list(select_id=preserve_id)

    def _update_granularity_enabled(self) -> None:
        enabled_count = 0
        for coarse_key, _label in GRANULARITY_SOURCES:
            ok = coarse_has_available_options(
                self._options, self._availability, coarse_key,
            )
            self._granularity_segmented.set_option_enabled(coarse_key, ok)
            if ok:
                enabled_count += 1
        self._granularity_segmented.setVisible(enabled_count > 1)

        current = self._selected_granularity()
        if not coarse_has_available_options(
            self._options, self._availability, current,
        ):
            self._set_granularity(
                default_coarse_source(self._options, self._availability),
                refresh_list=False,
            )

    def _selected_list_key(self) -> str | None:
        item = self._option_list.currentItem()
        if item is None:
            return None
        data = item.data(Qt.ItemDataRole.UserRole)
        return str(data) if data is not None else None

    def _selected_option(self) -> UnifiedViewOption | None:
        key = self._selected_list_key()
        if key is None:
            return None
        return option_from_list_key(self._options, key)

    def _default_list_key(self, *, coarse: CoarseDataSourceKind) -> str | None:
        for opt in options_for_coarse(self._options, coarse):
            if is_option_available(self._availability, opt):
                return option_list_key(opt)
        return None

    def _refresh_option_list(
        self,
        *,
        select_key: str | None = None,
        select_id: str | None = None,
        select_source: DataSourceKind | None = None,
    ) -> None:
        self._updating = True
        try:
            self._option_list.clear()
            coarse = self._selected_granularity()
            visible = options_for_coarse(self._options, coarse)
            selected_row = -1
            for row, opt in enumerate(visible):
                item = QListWidgetItem(opt.label)
                item.setData(Qt.ItemDataRole.UserRole, option_list_key(opt))
                enabled = is_option_available(self._availability, opt)
                if enabled:
                    item.setFlags(
                        Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled,
                    )
                else:
                    item.setFlags(Qt.ItemFlag.NoItemFlags)
                self._option_list.addItem(item)
                if select_key is not None and option_list_key(opt) == select_key and enabled:
                    selected_row = row
                elif (
                    select_key is None
                    and select_id is not None
                    and opt.id == select_id
                    and enabled
                    and (
                        select_source is None
                        or opt.source == select_source
                    )
                ):
                    selected_row = row

            if selected_row < 0 and select_id is not None:
                for row, opt in enumerate(visible):
                    if opt.id == select_id and is_option_available(self._availability, opt):
                        selected_row = row
                        break

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

    def _on_granularity_segment_changed(self, _key: str) -> None:
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
