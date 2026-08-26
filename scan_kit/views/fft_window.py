"""Qt shell for the FFT Explorer viewer."""

from __future__ import annotations

import logging
from typing import Sequence

from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..common.data_filter import FILTER_ALL, FILTER_BEAM_BOTH, FILTER_BEAM_ON
from ..common.settings import ViewSettings
from .async_refresh import DebouncedBackgroundTask
from .fft_catalog import (
    FFT_METRICS,
    METRIC_BY_ID,
    PRESET_BY_ID,
    PRESETS,
    FftConfig,
)
from .fft_data import (
    channel_keys_for_metric,
    default_config,
    load_sessions_fft,
    merge_fft_session_channels,
    probe_channel_availability,
    probe_fft_metric_availability_headers,
)
from .fft_ui import render_fft
from .plot_view_shell import (
    PlotViewWindow,
    make_presets_menu_button,
    make_side_panel_column,
    new_headless_figure,
    run_view_window,
)
from .unified_view_controls import DataFilterPanel, sync_data_filter_panel

_log = logging.getLogger(__name__)


class FftExplorerWindow(PlotViewWindow):
    """Configurable frequency-domain explorer for timeslice signals."""

    def __init__(
        self,
        session_ids: Sequence[str],
        base_dir: str,
        *,
        settings: ViewSettings | None = None,
        initial_preset: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(
            title="FFT Explorer",
            figsize=(16, 9),
            side_panel_min_width=240,
            side_panel_default_width=300,
            parent=parent,
        )
        self._session_ids = list(session_ids)
        self._base_dir = base_dir
        self._settings = settings
        self._session_data: dict[str, dict] = {}
        self._metric_availability: dict[str, bool] = {}
        self._channel_availability: dict[str, bool] = {}
        self._metric_list: QListWidget | None = None
        self._channel_checks: dict[str, QCheckBox] = {}
        self._channel_group: QGroupBox | None = None
        self._channel_layout: QVBoxLayout | None = None
        self._filter_panel: DataFilterPanel | None = None
        self._peaks_box: QCheckBox | None = None
        self._refresh_generation = 0
        self._updating = False
        self._pending_preset = initial_preset

        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(80)
        self._refresh_timer.timeout.connect(self._start_refresh)

        self._load_task = DebouncedBackgroundTask(debounce_ms=0, parent=self)
        self._load_task.finished.connect(self._on_load_finished)

        self._metric_load_task = DebouncedBackgroundTask(debounce_ms=0, parent=self)
        self._metric_load_task.finished.connect(self._on_metric_channels_loaded)

        self._render_task = DebouncedBackgroundTask(debounce_ms=50, parent=self)
        self._render_task.finished.connect(self._on_render_finished)

        self.set_side_panel(self._build_controls())
        self._show_status_message("Loading timeslice data…")
        self._start_initial_load()

    def _show_status_message(self, message: str) -> None:
        self.figure.clear()
        self.figure.text(0.5, 0.5, message, ha="center", va="center")
        self.draw_idle()

    def _build_controls(self) -> QWidget:
        panel, layout = make_side_panel_column()

        layout.addWidget(
            make_presets_menu_button(
                [(preset.id, preset.label, True) for preset in PRESETS],
                self._apply_preset,
            )
        )

        metric_group = QGroupBox("Signal Source")
        metric_layout = QVBoxLayout(metric_group)
        self._metric_list = QListWidget()
        self._metric_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        for metric in FFT_METRICS:
            item = QListWidgetItem(metric.label)
            item.setData(256, metric.id)
            self._metric_list.addItem(item)
        self._metric_list.currentItemChanged.connect(self._on_metric_changed)
        metric_layout.addWidget(self._metric_list)
        layout.addWidget(metric_group)

        self._channel_group = QGroupBox("Channels")
        self._channel_layout = QVBoxLayout(self._channel_group)
        layout.addWidget(self._channel_group)

        self._filter_panel = DataFilterPanel(
            on_selection_changed=self._schedule_refresh,
            domain_current=FILTER_ALL,
            beam_current=FILTER_BEAM_ON,
        )
        layout.addWidget(self._filter_panel)

        options_group = QGroupBox("Options")
        options_layout = QVBoxLayout(options_group)
        self._peaks_box = QCheckBox("Annotate Peaks")
        self._peaks_box.setChecked(True)
        self._peaks_box.toggled.connect(self._schedule_refresh)
        options_layout.addWidget(self._peaks_box)
        layout.addWidget(options_group)

        layout.addStretch(1)
        return panel

    def _current_metric_id(self) -> str | None:
        if self._metric_list is None:
            return None
        item = self._metric_list.currentItem()
        if item is None:
            return None
        metric_id = item.data(256)
        return str(metric_id) if metric_id is not None else None

    def _rebuild_channel_checks(self) -> None:
        if self._channel_layout is None or self._channel_group is None:
            return
        while self._channel_layout.count():
            child = self._channel_layout.takeAt(0)
            if child.widget() is not None:
                child.widget().deleteLater()
        self._channel_checks.clear()

        metric_id = self._current_metric_id()
        metric = METRIC_BY_ID.get(metric_id) if metric_id else None
        if metric is None:
            self._channel_group.setEnabled(False)
            return
        self._channel_group.setEnabled(True)
        self._channel_group.setTitle(f"Channels — {metric.label}")

        for channel in metric.channels:
            box = QCheckBox(channel.label)
            available = bool(self._channel_availability.get(channel.id, False))
            box.setEnabled(available)
            box.setChecked(available)
            box.toggled.connect(self._schedule_refresh)
            self._channel_layout.addWidget(box)
            self._channel_checks[channel.id] = box

    def _sync_metric_list(self) -> None:
        if self._metric_list is None:
            return
        self._updating = True
        try:
            current = self._current_metric_id()
            for row in range(self._metric_list.count()):
                item = self._metric_list.item(row)
                metric_id = str(item.data(256))
                available = self._metric_availability.get(metric_id, False)
                item.setFlags(
                    item.flags() | Qt.ItemFlag.ItemIsEnabled
                    if available
                    else item.flags() & ~Qt.ItemFlag.ItemIsEnabled
                )
            if current and self._metric_availability.get(current, False):
                return
            for row in range(self._metric_list.count()):
                item = self._metric_list.item(row)
                metric_id = str(item.data(256))
                if self._metric_availability.get(metric_id, False):
                    self._metric_list.setCurrentRow(row)
                    break
        finally:
            self._updating = False

    def _start_initial_load(self) -> None:
        session_ids = list(self._session_ids)
        base_dir = self._base_dir
        settings = self._settings
        pending_preset = self._pending_preset

        def loader() -> tuple[dict[str, dict], dict[str, bool]]:
            header_avail = probe_fft_metric_availability_headers(
                session_ids, base_dir,
            )
            if pending_preset and pending_preset in PRESET_BY_ID:
                metric_id = PRESET_BY_ID[pending_preset].metric_id
            else:
                metric_id = next(
                    (
                        metric.id
                        for metric in FFT_METRICS
                        if header_avail.get(metric.id, False)
                    ),
                    FFT_METRICS[0].id,
                )
            channel_keys = channel_keys_for_metric(metric_id)
            session_data = load_sessions_fft(
                session_ids,
                base_dir,
                settings=settings,
                channel_keys=channel_keys,
            )
            return session_data, header_avail

        self._load_task.schedule(loader)

    def _apply_preset(self, preset_id: str) -> None:
        preset = PRESET_BY_ID[preset_id]
        self._set_config(
            FftConfig(
                metric_id=preset.metric_id,
                channels=preset.channels,
                domain_filter=preset.domain_filter,
                beam_state_filter=preset.beam_state_filter,
                annotate_peaks=preset.annotate_peaks,
            )
        )
        self._schedule_refresh()

    def _set_config(self, config: FftConfig) -> None:
        self._updating = True
        try:
            if self._metric_list is not None:
                for row in range(self._metric_list.count()):
                    item = self._metric_list.item(row)
                    if str(item.data(256)) == config.metric_id:
                        self._metric_list.setCurrentRow(row)
                        break
            self._channel_availability = probe_channel_availability(
                self._session_data, config.metric_id,
            )
            self._rebuild_channel_checks()
            for channel_id, box in self._channel_checks.items():
                available = bool(self._channel_availability.get(channel_id, False))
                if not available and self._session_data:
                    available = any(
                        len(data.get(channel_id, ())) > 0
                        for data in self._session_data.values()
                    )
                    box.setEnabled(bool(available))
                box.setChecked(available and channel_id in config.channels)
            if self._filter_panel is not None:
                self._filter_panel.set_domain(config.domain_filter)
                self._filter_panel.set_beam_state(config.beam_state_filter)
            if self._peaks_box is not None:
                self._peaks_box.setChecked(config.annotate_peaks)
        finally:
            self._updating = False

    def _read_config(self) -> FftConfig:
        metric_id = self._current_metric_id() or FFT_METRICS[0].id
        channels = tuple(
            channel_id
            for channel_id, box in self._channel_checks.items()
            if box.isChecked()
        )
        return FftConfig(
            metric_id=metric_id,
            channels=channels,
            domain_filter=(
                self._filter_panel.selected_domain()
                if self._filter_panel is not None
                else FILTER_ALL
            ) or FILTER_ALL,
            beam_state_filter=(
                self._filter_panel.selected_beam_state()
                if self._filter_panel is not None
                else FILTER_BEAM_BOTH
            ) or FILTER_BEAM_BOTH,
            annotate_peaks=(
                self._peaks_box.isChecked() if self._peaks_box is not None else True
            ),
        )

    def _on_metric_changed(self, _current, _previous) -> None:
        if self._updating:
            return
        metric_id = self._current_metric_id()
        if metric_id is None:
            return
        needed_keys = channel_keys_for_metric(metric_id)
        missing = frozenset(
            key
            for key in needed_keys
            if not any(
                data.get(key) is not None and len(data[key]) > 0
                for data in self._session_data.values()
            )
        )
        if missing and self._metric_availability.get(metric_id, False):
            self._load_metric_channels(metric_id, missing)
            return
        self._finish_metric_change(metric_id)

    def _load_metric_channels(
        self,
        metric_id: str,
        channel_keys: frozenset[str],
    ) -> None:
        session_ids = list(self._session_ids)
        base_dir = self._base_dir
        bg = self._settings.bg_subtract if self._settings else False
        existing = dict(self._session_data)

        def loader() -> tuple[str, dict[str, dict]]:
            updated = dict(existing)
            for sid in session_ids:
                merged = merge_fft_session_channels(
                    updated.get(sid),
                    sid,
                    base_dir,
                    channel_keys,
                    bg_subtract=bg,
                )
                if merged is not None:
                    updated[sid] = merged
            return metric_id, updated

        self._show_status_message("Loading signal data…")
        self._metric_load_task.schedule(loader)

    @Slot(int, object)
    def _on_metric_channels_loaded(self, _task_generation: int, result: object) -> None:
        if not isinstance(result, tuple) or len(result) != 2:
            return
        metric_id, session_data = result
        self._session_data = session_data
        if metric_id == self._current_metric_id():
            self._finish_metric_change(metric_id)

    def _finish_metric_change(self, metric_id: str) -> None:
        self._channel_availability = probe_channel_availability(
            self._session_data, metric_id,
        )
        metric = METRIC_BY_ID[metric_id]
        self._updating = True
        try:
            self._rebuild_channel_checks()
            for channel in metric.channels:
                box = self._channel_checks.get(channel.id)
                if box is not None and box.isEnabled():
                    box.setChecked(channel.id in metric.default_channel_ids)
        finally:
            self._updating = False
        self._schedule_refresh()

    def _schedule_refresh(self) -> None:
        if self._updating:
            return
        self._refresh_generation += 1
        self._refresh_timer.start()

    def _start_refresh(self) -> None:
        gen = self._refresh_generation
        config = self._read_config()
        self.setWindowTitle(config.title)
        self._schedule_render(gen, config)

    def _schedule_render(self, gen: int, config: FftConfig) -> None:
        session_data = self._session_data
        base_dir = self._base_dir
        figsize = tuple(float(v) for v in self.figure.get_size_inches())

        def render_fn() -> tuple[int, Figure | None]:
            fig = new_headless_figure(figsize)
            try:
                render_fft(fig, config, session_data, base_dir)
            except Exception:
                _log.exception("FFT Explorer render failed")
                return gen, None
            return gen, fig

        self._render_task.schedule(render_fn)

    @Slot(int, object)
    def _on_load_finished(self, _task_generation: int, result: object) -> None:
        if not isinstance(result, tuple) or len(result) != 2:
            self._show_status_message("Failed to load timeslice data")
            _log.error("FFT load returned invalid result: %r", result)
            return
        session_data, availability = result
        self._session_data = session_data
        self._metric_availability = availability

        if not session_data:
            self._show_status_message("No timeslice signal data for selected sessions")
            self._set_config(FftConfig(channels=()))
            return

        sync_data_filter_panel(
            self._filter_panel,
            supports_filter=True,
            has_beam_state=True,
            reset_defaults=True,
        )
        self._sync_metric_list()

        if self._pending_preset and self._pending_preset in PRESET_BY_ID:
            preset_id = self._pending_preset
            self._pending_preset = None
            self._apply_preset(preset_id)
        else:
            self._set_config(default_config(availability, session_data))
            self._schedule_refresh()

    @Slot(int, object)
    def _on_render_finished(self, _task_generation: int, result: object) -> None:
        if not isinstance(result, tuple) or len(result) != 2:
            return
        gen, fig = result
        if gen != self._refresh_generation:
            if isinstance(fig, Figure):
                import matplotlib.pyplot as plt
                plt.close(fig)
            return
        if not isinstance(fig, Figure):
            self._show_status_message("FFT render failed — see log for details")
            return
        previous = self.figure
        self.figure = fig
        self.canvas.figure = fig
        self.draw_idle()
        if previous is not fig:
            import matplotlib.pyplot as plt
            plt.close(previous)


def run_fft_explorer_window(
    session_ids: Sequence[str],
    base_dir: str = "test_data",
    *,
    settings: ViewSettings | None = None,
    initial_preset: str | None = None,
) -> None:
    if not session_ids:
        print("No sessions selected")
        return

    run_view_window(
        lambda: FftExplorerWindow(
            session_ids,
            base_dir,
            settings=settings,
            initial_preset=initial_preset,
        ),
    )
