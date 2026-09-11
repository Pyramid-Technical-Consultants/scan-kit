"""Qt shell hosting Matplotlib timeslice replay with channel controls."""

from __future__ import annotations

from typing import Sequence

from PySide6.QtCore import QTimer, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QVBoxLayout,
    QWidget,
)

from ..data.types import DATA_SOURCE_TIMESLICE_ISO, DataSourceKind
from .async_refresh import DebouncedBackgroundTask
from .plot_view_shell import (
    PlotViewWindow,
    make_presets_menu_button,
    make_side_panel_column,
    run_view_window,
)
from .timeslice_replay_catalog import (
    METRIC_BY_ID,
    PRESET_BY_ID,
    PRESETS,
    VIEW_OPTIONS,
    metric_for_option,
)
from .timeslice_replay_channels import (
    available_channel_keys,
    build_replay_config,
    default_metric_selection,
    filter_available_keys,
    load_sessions_catalog,
    probe_replay_option_availability,
)
from .timeslice_replay_ui import render_timeslice_replay
from .unified_view_controls import DataSourceOptionPanel


class TimesliceReplayWindow(PlotViewWindow):
    """Interactive timeslice replay with unified data-source controls."""

    def __init__(
        self,
        session_ids: Sequence[str],
        base_dir: str,
        *,
        initial_bg_subtract: bool = False,
        initial_preset: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(
            title="Timeslice Replay",
            figsize=(16, 9),
            side_panel_min_width=240,
            side_panel_default_width=300,
            parent=parent,
        )
        self._session_ids = list(session_ids)
        self._base_dir = base_dir
        self._bg_subtract = initial_bg_subtract
        self._session_data: dict[str, dict] = {}
        self._option_availability: dict[str, bool] = {}
        self._channel_checks: dict[str, QCheckBox] = {}
        self._option_panel: DataSourceOptionPanel | None = None
        self._channel_group: QGroupBox | None = None
        self._channel_layout: QVBoxLayout | None = None
        self._bg_check: QCheckBox | None = None
        self._peer_check: QCheckBox | None = None
        self._edges_check: QCheckBox | None = None
        self._digital_check: QCheckBox | None = None
        self._beam_check: QCheckBox | None = None
        self._pending_preset = initial_preset
        self._updating = False

        self._load_task = DebouncedBackgroundTask(debounce_ms=0, parent=self)
        self._load_task.finished.connect(self._on_load_finished)

        self._probe_task = DebouncedBackgroundTask(debounce_ms=0, parent=self)
        self._probe_task.finished.connect(self._on_probe_finished)

        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(30)
        self._refresh_timer.timeout.connect(self._refresh_plot)

        self.set_side_panel(self._build_controls())
        self._show_status_message("Loading timeslice options…")
        self._start_probe()

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

        self._option_panel = DataSourceOptionPanel(
            on_selection_changed=self._on_metric_selection_changed,
            show_granularity=False,
        )
        self._option_panel.configure(
            VIEW_OPTIONS,
            {},
            group_title="Signal Source",
            preferred_source=DATA_SOURCE_TIMESLICE_ISO,
        )
        layout.addWidget(self._option_panel)

        self._channel_group = QGroupBox("Channels")
        self._channel_layout = QVBoxLayout(self._channel_group)
        layout.addWidget(self._channel_group)

        options = QGroupBox("Options")
        opt_layout = QVBoxLayout(options)
        self._bg_check = QCheckBox("Background Subtract")
        self._bg_check.setChecked(self._bg_subtract)
        self._bg_check.toggled.connect(self._on_bg_subtract_toggled)
        opt_layout.addWidget(self._bg_check)

        self._peer_check = QCheckBox("Peer Overlay (Single Session)")
        self._peer_check.setChecked(False)
        self._peer_check.toggled.connect(self._schedule_refresh)
        opt_layout.addWidget(self._peer_check)

        self._edges_check = QCheckBox("Beam-Off Edges")
        self._edges_check.setChecked(True)
        self._edges_check.toggled.connect(self._schedule_refresh)
        opt_layout.addWidget(self._edges_check)

        self._digital_check = QCheckBox("Digital Lanes")
        self._digital_check.setChecked(True)
        self._digital_check.toggled.connect(self._schedule_refresh)
        opt_layout.addWidget(self._digital_check)

        self._beam_check = QCheckBox("Source Beam Current Twin Axis")
        self._beam_check.setChecked(True)
        self._beam_check.toggled.connect(self._schedule_refresh)
        opt_layout.addWidget(self._beam_check)
        layout.addWidget(options)

        layout.addStretch(1)
        return panel

    def _current_metric_id(self) -> str | None:
        if self._option_panel is None:
            return None
        return self._option_panel.selected_id()

    def _current_source(self) -> DataSourceKind:
        if self._option_panel is None:
            return DATA_SOURCE_TIMESLICE_ISO
        return self._option_panel.selected_source()

    def _selected_channel_keys(self) -> list[str]:
        return [
            key for key, check in self._channel_checks.items() if check.isChecked()
        ]

    def _start_probe(self) -> None:
        session_ids = list(self._session_ids)
        base_dir = self._base_dir

        def loader() -> dict[str, bool]:
            return probe_replay_option_availability(session_ids, base_dir)

        self._probe_task.schedule(loader)

    @Slot(int, object)
    def _on_probe_finished(self, _task_generation: int, result: object) -> None:
        if not isinstance(result, dict):
            self._show_status_message("Failed to load timeslice options")
            return
        self._option_availability = result
        if self._option_panel is not None:
            self._option_panel.configure(
                VIEW_OPTIONS,
                self._option_availability,
                group_title="Signal Source",
                preferred_source=DATA_SOURCE_TIMESLICE_ISO,
            )

        preset = self._pending_preset
        self._pending_preset = None
        if preset and preset in PRESET_BY_ID:
            self._apply_preset(preset)
        else:
            metric_id, source, channels = default_metric_selection(
                self._option_availability,
            )
            if self._option_panel is not None:
                self._option_panel.select_id(metric_id, source=source)
            self._set_peer_overlay_default(metric_id, source)
            self._pending_channels = channels
            self._start_load(metric_id, source)

    def _set_peer_overlay_default(self, metric_id: str, source: DataSourceKind) -> None:
        metric = metric_for_option(metric_id, source)
        if self._peer_check is None or metric is None:
            return
        self._peer_check.setChecked(
            metric.peer_overlay_default and len(self._session_ids) <= 1,
        )

    def _apply_preset(self, preset_id: str) -> None:
        preset = PRESET_BY_ID[preset_id]
        if self._option_panel is not None:
            self._option_panel.select_id(preset.metric_id, source=preset.source)
        self._set_peer_overlay_default(preset.metric_id, preset.source)
        self._pending_channels = preset.channels
        self._start_load(preset.metric_id, preset.source)

    def _on_metric_selection_changed(self) -> None:
        if self._updating:
            return
        metric_id = self._current_metric_id()
        source = self._current_source()
        if metric_id is None:
            return
        metric = metric_for_option(metric_id, source)
        if metric is None:
            return
        self._set_peer_overlay_default(metric_id, source)
        self._pending_channels = metric.default_channel_keys
        self._start_load(metric_id, source)

    def _start_load(self, metric_id: str, source: DataSourceKind) -> None:
        session_ids = list(self._session_ids)
        base_dir = self._base_dir
        bg_subtract = self._bg_subtract
        load_metric = metric_id
        load_source = source

        def loader() -> tuple[str, DataSourceKind, dict[str, dict]]:
            return (
                load_metric,
                load_source,
                load_sessions_catalog(
                    session_ids,
                    base_dir,
                    bg_subtract=bg_subtract,
                    metric_id=load_metric,
                    data_source=load_source,
                ),
            )

        self._show_status_message("Loading timeslice data…")
        self._load_task.schedule(loader)

    @Slot(int, object)
    def _on_load_finished(self, task_generation: int, result: object) -> None:
        if task_generation != self._load_task.generation:
            return
        if not isinstance(result, tuple) or len(result) != 3:
            self._show_status_message("Failed to load timeslice data")
            return
        metric_id, source, session_data = result
        if (
            metric_id != self._current_metric_id()
            or source != self._current_source()
        ):
            return
        if not isinstance(session_data, dict):
            self._show_status_message("Failed to load timeslice data")
            return
        self._session_data = session_data
        self._rebuild_channel_checks()
        self._refresh_plot()

    def _rebuild_channel_checks(self) -> None:
        if self._channel_layout is None or self._channel_group is None:
            return
        while self._channel_layout.count():
            child = self._channel_layout.takeAt(0)
            if child.widget() is not None:
                child.widget().deleteLater()
        self._channel_checks.clear()

        metric_id = self._current_metric_id()
        source = self._current_source()
        metric = (
            metric_for_option(metric_id, source)
            if metric_id is not None
            else None
        )
        if metric is None:
            self._channel_group.setEnabled(False)
            return

        available = available_channel_keys(self._session_data)
        pending = getattr(self, "_pending_channels", metric.default_channel_keys)
        self._pending_channels = None

        self._channel_group.setEnabled(True)
        self._channel_group.setTitle(f"Channels — {metric.label}")

        from .timeslice_replay_channels import CHANNEL_BY_KEY

        for channel_key in metric.channel_keys:
            channel = CHANNEL_BY_KEY.get(channel_key)
            if channel is None:
                continue
            box = QCheckBox(channel.label)
            enabled = channel_key in available
            box.setEnabled(enabled)
            box.setChecked(enabled and channel_key in pending)
            box.toggled.connect(self._schedule_refresh)
            self._channel_layout.addWidget(box)
            self._channel_checks[channel_key] = box

    def _on_bg_subtract_toggled(self, checked: bool) -> None:
        if self._updating:
            return
        self._bg_subtract = checked
        metric_id = self._current_metric_id()
        if metric_id is None:
            return
        self._pending_channels = tuple(self._selected_channel_keys())
        self._start_load(metric_id, self._current_source())

    def _schedule_refresh(self, *_args) -> None:
        if self._updating:
            return
        self._refresh_timer.start()

    def _refresh_plot(self) -> None:
        metric_id = self._current_metric_id()
        source = self._current_source()
        metric = (
            metric_for_option(metric_id, source)
            if metric_id is not None
            else None
        )
        selected = filter_available_keys(
            self._selected_channel_keys(),
            available_channel_keys(self._session_data),
        )
        if metric is None:
            self._show_status_message("Select a signal source to plot")
            return
        if not self._session_data:
            self._show_status_message("No timeslice data found for the selected sessions.")
            return
        if not selected:
            self._show_status_message("Select one or more channels to plot")
            return

        title = f"Timeslice Replay — {metric.label}"
        config = build_replay_config(
            selected,
            self._session_data,
            peer_overlay=bool(self._peer_check and self._peer_check.isChecked()),
            show_digital=bool(self._digital_check and self._digital_check.isChecked()),
            show_beam_twin=bool(self._beam_check and self._beam_check.isChecked()),
            beam_off_edges=bool(self._edges_check and self._edges_check.isChecked()),
            title=title,
        )
        render_timeslice_replay(
            self.figure, config, self._session_data, self._base_dir,
        )
        self.draw_idle()


def run_timeslice_replay_window(
    session_ids: Sequence[str],
    base_dir: str = "test_data",
    *,
    bg_subtract: bool = False,
    initial_preset: str | None = None,
) -> None:
    """Create the Qt timeslice replay window and run its event loop."""
    if not session_ids:
        print("No sessions selected")
        return

    run_view_window(
        lambda: TimesliceReplayWindow(
            session_ids,
            base_dir,
            initial_bg_subtract=bg_subtract,
            initial_preset=initial_preset,
        ),
    )
