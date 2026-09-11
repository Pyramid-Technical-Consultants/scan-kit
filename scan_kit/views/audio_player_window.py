"""Qt shell for the IC audio player (visPy waveforms + sounddevice playback)."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import sounddevice as sd
from PySide6.QtCore import QEvent, Qt, QTimer, Slot
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from ..common.data_filter import FILTER_ALL, FILTER_BEAM_ON
from ..common.settings import ViewSettings
from .async_refresh import DebouncedBackgroundTask
from .audio_player_data import (
    AudioPlayerConfig,
    CacheKey,
    DEFAULT_LIVE_FFT_WINDOW_MS,
    FS_HZ,
    LIVE_FFT_WINDOW_MS,
    WaveformRenderChannel,
    format_play_time,
    live_spectrum,
    prepare_waveform_render,
    session_has_ic3,
    write_wav,
)
from .audio_player_vispy import AudioSpectrumScene, AudioWaveformScene
from .fft_catalog import (
    FFT_METRICS,
    METRIC_BY_ID,
    METRIC_IC_CURRENT,
    PRESET_BY_ID,
)
from .fft_data import (
    channel_keys_for_metric,
    default_config,
    load_sessions_fft,
    merge_fft_session_channels,
    probe_channel_availability,
    probe_fft_metric_availability_headers,
)
from .plot_view_shell import (
    make_presets_menu_button,
    make_side_panel_column,
    run_view_window,
)

_log = logging.getLogger(__name__)

_CURSOR_UPDATE_MS = 40

AUDIO_PRESET_IC_CURRENT_ALL = "ic_current_all"
AUDIO_PRESET_IC_BEAM_SUB = "ic_current_beam_sub"

_AUDIO_PRESETS: tuple[tuple[str, str, AudioPlayerConfig], ...] = (
    (
        AUDIO_PRESET_IC_CURRENT_ALL,
        "IC1 current",
        AudioPlayerConfig(
            metric_id=METRIC_IC_CURRENT,
            channels=("ic1",),
            beam_state_filter=FILTER_BEAM_ON,
        ),
    ),
    (
        AUDIO_PRESET_IC_BEAM_SUB,
        "IC1 beam subtracted",
        AudioPlayerConfig(
            metric_id=METRIC_IC_CURRENT,
            channels=("ic1",),
            beam_state_filter=FILTER_BEAM_ON,
            beam_subtract=True,
        ),
    ),
)

_AUDIO_PRESET_BY_ID = {preset_id: config for preset_id, _label, config in _AUDIO_PRESETS}


class AudioPlayerWindow(QMainWindow):
    """Interactive IC timeslice audio player with FFT-style signal selection."""

    def __init__(
        self,
        session_ids: Sequence[str],
        base_dir: str,
        *,
        settings: ViewSettings | None = None,
        initial_preset: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Audio Explorer")
        self.resize(1400, 900)

        self._session_ids = list(session_ids)
        self._base_dir = base_dir
        self._settings = settings
        self._pending_preset = initial_preset
        self._session_data: dict[str, dict] = {}
        self._metric_availability: dict[str, bool] = {}
        self._channel_availability: dict[str, bool] = {}
        self._playback_channels: list[WaveformRenderChannel] = []
        self._channel_map: dict[str, np.ndarray] = {}
        self._waveform_cache: dict[CacheKey, WaveformRenderChannel] = {}
        self._selected_index = 0
        self._cursor_pos = 0.0
        self._playing_label: str | None = None
        self._play_start_time = 0.0
        self._wall_origin = 0.0
        self._play_n_samples = 0
        self._dragging = False
        self._slider_dragging = False
        self._was_playing: str | None = None
        self._refresh_generation = 0
        self._updating = False
        self._fft_window_combo: QComboBox | None = None
        self._play_pause_btn: QPushButton | None = None
        self._rewind_btn: QPushButton | None = None
        self._play_icon = None
        self._pause_icon = None
        self._seek_slider: QSlider | None = None
        self._time_label: QLabel | None = None
        self._duration_label: QLabel | None = None
        self._status_label: QLabel | None = None

        from vispy import scene
        from vispy.app import use_app

        use_app("pyside6")
        self._vispy_canvas = scene.SceneCanvas(
            keys=None,
            bgcolor="#1a1a1a",
            size=(1200, 620),
            show=False,
        )
        self._scene = AudioWaveformScene(self._vispy_canvas)
        self._spectrum_canvas = scene.SceneCanvas(
            keys=None,
            bgcolor="#1a1a1a",
            size=(1200, 180),
            show=False,
        )
        self._spectrum = AudioSpectrumScene(self._spectrum_canvas)

        plot_host = QWidget()
        plot_layout = QVBoxLayout(plot_host)
        plot_layout.setContentsMargins(6, 6, 0, 0)
        plot_layout.setSpacing(4)
        plot_layout.addWidget(self._vispy_canvas.native, 3)
        plot_layout.addWidget(self._spectrum_canvas.native, 1)
        self._spectrum_canvas.native.setMinimumHeight(180)
        self._spectrum_canvas.native.setMaximumHeight(280)

        self._channel_radios: dict[str, QRadioButton] = {}
        self._channel_button_group: QButtonGroup | None = None
        side_panel = self._build_controls()
        side_panel.setMinimumWidth(240)

        side_scroll = QScrollArea()
        side_scroll.setWidgetResizable(True)
        side_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
        )
        side_scroll.setMinimumWidth(240)
        side_scroll.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Expanding,
        )
        side_scroll.setWidget(side_panel)

        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter.setChildrenCollapsible(False)
        self._splitter.setHandleWidth(6)
        self._splitter.addWidget(plot_host)
        self._splitter.addWidget(side_scroll)
        self._splitter.setStretchFactor(0, 1)
        self._splitter.setStretchFactor(1, 0)
        self._splitter.setSizes([1100, 300])

        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        root_layout.addWidget(self._splitter, 1)
        root_layout.addWidget(self._build_transport())
        self.setCentralWidget(root)
        self._build_menu_bar()

        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(80)
        self._refresh_timer.timeout.connect(self._start_refresh)

        self._cursor_timer = QTimer(self)
        self._cursor_timer.setInterval(_CURSOR_UPDATE_MS)
        self._cursor_timer.timeout.connect(self._tick_cursor)

        self._load_task = DebouncedBackgroundTask(debounce_ms=0, parent=self)
        self._load_task.finished.connect(self._on_load_finished)

        self._metric_load_task = DebouncedBackgroundTask(debounce_ms=0, parent=self)
        self._metric_load_task.finished.connect(self._on_metric_channels_loaded)

        self._render_task = DebouncedBackgroundTask(debounce_ms=50, parent=self)
        self._render_task.finished.connect(self._on_render_finished)

        self._vispy_canvas.events.mouse_press.connect(self._on_mouse_press)
        self._vispy_canvas.events.mouse_move.connect(self._on_mouse_move)
        self._vispy_canvas.events.mouse_release.connect(self._on_mouse_release)
        self._vispy_canvas.events.mouse_wheel.connect(self._block_vispy_navigation)
        self._spectrum_canvas.events.mouse_wheel.connect(self._block_vispy_navigation)
        native = self._vispy_canvas.native
        native.setMouseTracking(True)
        native.setFocusPolicy(Qt.FocusPolicy.ClickFocus)

        play_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Space), self)
        play_shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
        play_shortcut.setAutoRepeat(False)
        play_shortcut.activated.connect(self._toggle_play_pause)

        self._scene.show_status("Loading timeslice data…")
        self._start_initial_load()

    def closeEvent(self, event) -> None:
        self._stop_all()
        super().closeEvent(event)

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.Type.PaletteChange:
            self._refresh_transport_icons()

    def _refresh_transport_icons(self) -> None:
        if self._play_pause_btn is None or self._rewind_btn is None:
            return
        from ..common.qt_theme import tinted_standard_icon

        self._play_icon = tinted_standard_icon(
            self, QStyle.StandardPixmap.SP_MediaPlay,
        )
        self._pause_icon = tinted_standard_icon(
            self, QStyle.StandardPixmap.SP_MediaPause,
        )
        self._rewind_btn.setIcon(
            tinted_standard_icon(
                self, QStyle.StandardPixmap.SP_MediaSkipBackward,
            )
        )
        self._sync_transport()

    def _clear_waveform_cache(self) -> None:
        self._waveform_cache.clear()

    def _build_controls(self) -> QWidget:
        panel, layout = make_side_panel_column()

        self._session_group = QGroupBox("Sessions")
        session_layout = QVBoxLayout(self._session_group)
        self._session_list = QListWidget()
        self._session_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        for sid in self._session_ids:
            self._session_list.addItem(QListWidgetItem(sid))
        if self._session_ids:
            self._session_list.setCurrentRow(0)
        self._session_list.currentItemChanged.connect(self._on_session_changed)
        session_layout.addWidget(self._session_list)
        self._session_group.setVisible(len(self._session_ids) > 1)
        layout.addWidget(self._session_group)

        layout.addWidget(
            make_presets_menu_button(
                [(preset_id, label, True) for preset_id, label, _ in _AUDIO_PRESETS],
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

        from .unified_view_controls import DataFilterPanel, sync_data_filter_panel

        self._filter_panel = DataFilterPanel(
            on_selection_changed=self._schedule_refresh,
            domain_current=FILTER_ALL,
            beam_current=FILTER_BEAM_ON,
        )
        layout.addWidget(self._filter_panel)
        self._sync_filter_panel = sync_data_filter_panel

        options_group = QGroupBox("Audio Options")
        options_layout = QVBoxLayout(options_group)
        self._beam_subtract_box = QCheckBox("Beam subtract IC1/IC2")
        self._beam_subtract_box.setChecked(True)
        self._beam_subtract_box.toggled.connect(self._schedule_refresh)
        options_layout.addWidget(self._beam_subtract_box)
        fft_row = QHBoxLayout()
        fft_row.addWidget(QLabel("FFT window"))
        self._fft_window_combo = QComboBox()
        for ms in LIVE_FFT_WINDOW_MS:
            self._fft_window_combo.addItem(f"{ms} ms", ms)
        self._fft_window_combo.blockSignals(True)
        idx = self._fft_window_combo.findData(DEFAULT_LIVE_FFT_WINDOW_MS)
        if idx >= 0:
            self._fft_window_combo.setCurrentIndex(idx)
        self._fft_window_combo.blockSignals(False)
        self._fft_window_combo.currentIndexChanged.connect(self._update_live_spectrum)
        fft_row.addWidget(self._fft_window_combo, 1)
        options_layout.addLayout(fft_row)
        layout.addWidget(options_group)

        save_btn = QPushButton("Save WAV")
        save_btn.clicked.connect(self._save_selected)
        layout.addWidget(save_btn)

        layout.addStretch(1)
        return panel

    def _build_menu_bar(self) -> None:
        from ..common.qt_theme import add_theme_menu

        view_menu = self.menuBar().addMenu("&View")
        theme_menu = view_menu.addMenu("Theme")
        self._menus = [view_menu, theme_menu]
        self._theme_menu_group = add_theme_menu(
            self, theme_menu, on_applied=self._refresh_transport_icons,
        )

    def _build_transport(self) -> QWidget:
        bar = QFrame()
        bar.setFrameShape(QFrame.Shape.StyledPanel)
        bar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(8)

        self._play_pause_btn = QPushButton()
        self._play_pause_btn.setToolTip("Play / Pause (Space)")
        self._play_pause_btn.setFixedSize(36, 28)
        self._play_pause_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._play_pause_btn.clicked.connect(self._toggle_play_pause)

        self._rewind_btn = QPushButton()
        self._rewind_btn.setToolTip("Go to start")
        self._rewind_btn.setFixedSize(36, 28)
        self._rewind_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._rewind_btn.clicked.connect(self._stop_reset)
        self._refresh_transport_icons()

        self._time_label = QLabel(format_play_time(0.0))
        self._time_label.setMinimumWidth(52)

        self._seek_slider = QSlider(Qt.Orientation.Horizontal)
        self._seek_slider.setRange(0, 10000)
        self._seek_slider.setValue(0)
        self._seek_slider.setToolTip("Seek")
        self._slider_dragging = False
        self._seek_slider.sliderPressed.connect(self._on_slider_pressed)
        self._seek_slider.sliderMoved.connect(self._on_slider_moved)
        self._seek_slider.sliderReleased.connect(self._on_slider_released)

        self._duration_label = QLabel(format_play_time(0.0))
        self._duration_label.setMinimumWidth(52)

        self._status_label = QLabel("Click waveform to seek")
        self._status_label.setMinimumWidth(180)

        layout.addWidget(self._play_pause_btn)
        layout.addWidget(self._rewind_btn)
        layout.addWidget(self._time_label)
        layout.addWidget(self._seek_slider, 1)
        layout.addWidget(self._duration_label)
        layout.addWidget(self._status_label, 1)
        return bar

    def _current_session_id(self) -> str | None:
        if self._session_list is None:
            return self._session_ids[0] if self._session_ids else None
        item = self._session_list.currentItem()
        return item.text() if item is not None else None

    def _current_metric_id(self) -> str | None:
        if self._metric_list is None:
            return None
        item = self._metric_list.currentItem()
        if item is None:
            return None
        metric_id = item.data(256)
        return str(metric_id) if metric_id is not None else None

    def _rebuild_channel_radios(self, metric_id: str | None = None) -> None:
        if self._channel_layout is None or self._channel_group is None:
            return
        while self._channel_layout.count():
            child = self._channel_layout.takeAt(0)
            if child.widget() is not None:
                child.widget().deleteLater()
        self._channel_radios.clear()
        self._channel_button_group = QButtonGroup(self)
        self._channel_button_group.setExclusive(True)

        metric_id = metric_id or self._current_metric_id()
        metric = METRIC_BY_ID.get(metric_id) if metric_id else None
        if metric is None:
            self._channel_group.setEnabled(False)
            return
        self._channel_group.setEnabled(True)
        self._channel_group.setTitle(f"Channel — {metric.label}")

        for channel in metric.channels:
            radio = QRadioButton(channel.label)
            available = bool(self._channel_availability.get(channel.id, False))
            radio.setEnabled(available)
            radio.toggled.connect(self._on_channel_radio_toggled)
            self._channel_button_group.addButton(radio)
            self._channel_layout.addWidget(radio)
            self._channel_radios[channel.id] = radio

    def _on_channel_radio_toggled(self, checked: bool) -> None:
        if not checked or self._updating:
            return
        self._selected_index = 0
        self._schedule_refresh()

    def _selected_channel_id(self) -> str | None:
        for channel_id, radio in self._channel_radios.items():
            if radio.isChecked():
                return channel_id
        return None

    def _pick_channel_id(
        self,
        preferred: Sequence[str],
    ) -> str | None:
        for channel_id in preferred:
            radio = self._channel_radios.get(channel_id)
            if radio is not None and radio.isEnabled():
                return channel_id
        for channel_id, radio in self._channel_radios.items():
            if radio.isEnabled():
                return channel_id
        return None

    def _select_channel_radio(self, channel_id: str | None) -> None:
        if channel_id is None:
            return
        radio = self._channel_radios.get(channel_id)
        if radio is not None and radio.isEnabled():
            radio.setChecked(True)

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
            if self._metric_list is not None:
                self._metric_list.blockSignals(True)
                try:
                    for row in range(self._metric_list.count()):
                        item = self._metric_list.item(row)
                        metric_id = str(item.data(256))
                        if self._metric_availability.get(metric_id, False):
                            self._metric_list.setCurrentRow(row)
                            break
                finally:
                    self._metric_list.blockSignals(False)
        finally:
            self._updating = False

    def _update_beam_subtract_box(self) -> None:
        if self._beam_subtract_box is None:
            return
        session_id = self._current_session_id()
        session = self._session_data.get(session_id or "", {})
        enabled = (
            self._current_metric_id() == METRIC_IC_CURRENT
            and session_has_ic3(session)
        )
        self._beam_subtract_box.setEnabled(enabled)
        if not enabled:
            self._beam_subtract_box.setChecked(False)

    def _read_config(self) -> AudioPlayerConfig:
        metric_id = self._current_metric_id() or FFT_METRICS[0].id
        selected = self._selected_channel_id()
        channels = (selected,) if selected is not None else ()
        return AudioPlayerConfig(
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
                else FILTER_BEAM_ON
            ) or FILTER_BEAM_ON,
            annotate_peaks=False,
            beam_subtract=(
                self._beam_subtract_box.isChecked()
                if self._beam_subtract_box is not None
                else False
            ),
        )

    def _set_config(self, config: AudioPlayerConfig) -> None:
        self._updating = True
        try:
            if self._metric_list is not None:
                self._metric_list.blockSignals(True)
                try:
                    for row in range(self._metric_list.count()):
                        item = self._metric_list.item(row)
                        if str(item.data(256)) == config.metric_id:
                            self._metric_list.setCurrentRow(row)
                            break
                finally:
                    self._metric_list.blockSignals(False)
            self._channel_availability = probe_channel_availability(
                self._session_data, config.metric_id,
            )
            self._rebuild_channel_radios(config.metric_id)
            for channel_id, radio in self._channel_radios.items():
                available = bool(self._channel_availability.get(channel_id, False))
                if not available and self._session_data:
                    available = any(
                        len(data.get(channel_id, ())) > 0
                        for data in self._session_data.values()
                    )
                    radio.setEnabled(bool(available))
            self._select_channel_radio(self._pick_channel_id(config.channels))
            if self._filter_panel is not None:
                self._filter_panel.set_domain(config.domain_filter)
                self._filter_panel.set_beam_state(config.beam_state_filter)
            if self._beam_subtract_box is not None:
                self._beam_subtract_box.setChecked(config.beam_subtract)
            self._update_beam_subtract_box()
        finally:
            self._updating = False

    def _apply_preset(self, preset_id: str) -> None:
        if preset_id in _AUDIO_PRESET_BY_ID:
            self._set_config(_AUDIO_PRESET_BY_ID[preset_id])
        elif preset_id in PRESET_BY_ID:
            preset = PRESET_BY_ID[preset_id]
            self._set_config(
                AudioPlayerConfig(
                    metric_id=preset.metric_id,
                    channels=preset.channels,
                    domain_filter=preset.domain_filter,
                    beam_state_filter=preset.beam_state_filter,
                    annotate_peaks=False,
                )
            )
        self._schedule_refresh()

    def _start_initial_load(self) -> None:
        session_ids = list(self._session_ids)
        base_dir = self._base_dir
        settings = self._settings
        pending_preset = self._pending_preset

        def loader() -> tuple[dict[str, dict], dict[str, bool]]:
            header_avail = probe_fft_metric_availability_headers(
                session_ids, base_dir,
            )
            if pending_preset and pending_preset in _AUDIO_PRESET_BY_ID:
                metric_id = _AUDIO_PRESET_BY_ID[pending_preset].metric_id
            elif pending_preset and pending_preset in PRESET_BY_ID:
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

    @Slot(int, object)
    def _on_load_finished(self, _task_generation: int, result: object) -> None:
        if not isinstance(result, tuple) or len(result) != 2:
            self._scene.show_status("Failed to load timeslice data")
            _log.error("Audio player load returned invalid result: %r", result)
            return
        session_data, availability = result
        self._session_data = session_data
        self._metric_availability = availability
        self._clear_waveform_cache()

        if not session_data:
            self._scene.show_status("No timeslice signal data for selected sessions")
            self._set_config(AudioPlayerConfig(channels=()))
            return

        self._sync_filter_panel(
            self._filter_panel,
            supports_filter=True,
            has_beam_state=True,
            reset_defaults=True,
        )
        self._sync_metric_list()

        if self._pending_preset and (
            self._pending_preset in _AUDIO_PRESET_BY_ID
            or self._pending_preset in PRESET_BY_ID
        ):
            preset_id = self._pending_preset
            self._pending_preset = None
            self._apply_preset(preset_id)
        else:
            base = default_config(availability, session_data)
            self._set_config(
                AudioPlayerConfig(
                    metric_id=base.metric_id,
                    channels=base.channels,
                    domain_filter=base.domain_filter,
                    beam_state_filter=base.beam_state_filter,
                    annotate_peaks=False,
                )
            )
            self._schedule_refresh()

    def _on_session_changed(self, _current, _previous) -> None:
        if self._updating:
            return
        self._selected_index = 0
        self._clear_waveform_cache()
        self._update_beam_subtract_box()
        self._schedule_refresh()

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

        if self._status_label is not None:
            self._status_label.setText("Loading signal data…")
        self._metric_load_task.schedule(loader)

    @Slot(int, object)
    def _on_metric_channels_loaded(self, _task_generation: int, result: object) -> None:
        if not isinstance(result, tuple) or len(result) != 2:
            return
        metric_id, session_data = result
        self._session_data = session_data
        self._clear_waveform_cache()
        if metric_id == self._current_metric_id():
            self._finish_metric_change(metric_id)

    def _finish_metric_change(self, metric_id: str) -> None:
        self._channel_availability = probe_channel_availability(
            self._session_data, metric_id,
        )
        metric = METRIC_BY_ID[metric_id]
        self._updating = True
        try:
            self._rebuild_channel_radios()
            preferred = tuple(
                channel.id
                for channel in metric.channels
                if channel.id in metric.default_channel_ids
            )
            self._select_channel_radio(self._pick_channel_id(preferred))
            self._update_beam_subtract_box()
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
        session_id = self._current_session_id() or ""
        session = self._session_data.get(session_id, {})
        self.setWindowTitle(f"Audio Explorer — {session_id or '—'}")

        if not config.channels or not session:
            self._stop_all()
            self._playback_channels = []
            self._channel_map = {}
            self._scene.show_status("No channel selected")
            self._update_live_spectrum()
            if self._status_label is not None:
                self._status_label.setText("Select an available channel")
            return

        if self._status_label is not None:
            self._status_label.setText("Preparing waveform…")

        snapshot = dict(self._waveform_cache)

        def render_fn() -> tuple[int, list[WaveformRenderChannel] | None, dict]:
            try:
                channels = prepare_waveform_render(
                    session_id, session, config, snapshot,
                )
            except Exception:
                _log.exception("Audio Explorer waveform prepare failed")
                return gen, None, {}
            return gen, channels, snapshot

        self._schedule_render(gen, render_fn)

    def _schedule_render(
        self,
        gen: int,
        render_fn,
    ) -> None:
        self._render_task.schedule(render_fn)

    @Slot(int, object)
    def _on_render_finished(self, _task_generation: int, result: object) -> None:
        if not isinstance(result, tuple) or len(result) != 3:
            return
        gen, channels, snapshot = result
        if gen != self._refresh_generation:
            return
        if channels is None:
            self._scene.show_status("Waveform render failed — see log for details")
            if self._status_label is not None:
                self._status_label.setText("Render failed")
            return
        self._waveform_cache.update(snapshot)

        self._playback_channels = channels
        self._channel_map = {ch.label: ch.signal for ch in channels}

        if not channels:
            self._stop_all()
            self._scene.show_status("No channel selected")
            self._update_live_spectrum()
            if self._status_label is not None:
                self._status_label.setText("Select an available channel")
            return

        resume_playback = self._playing_label is not None
        self._selected_index = 0
        duration = max(len(ch.signal) for ch in channels) / FS_HZ
        self._cursor_pos = min(self._cursor_pos, duration)

        try:
            self._scene.set_render_channels(
                channels,
                selected_index=self._selected_index,
                cursor_time=self._cursor_pos,
            )
        except Exception:
            _log.exception("Audio Explorer waveform display failed")
            self._scene.show_status("Waveform display failed — see log for details")
            if self._status_label is not None:
                self._status_label.setText("Display failed")
            return
        self._update_time_label()
        self._update_live_spectrum()
        if resume_playback:
            self._play_selected()
            return
        if self._status_label is not None:
            self._status_label.setText(
                f"{format_play_time(duration)} @ {int(FS_HZ)} Hz"
            )

    def _selected_label(self) -> str | None:
        if not self._playback_channels:
            return None
        idx = max(0, min(self._selected_index, len(self._playback_channels) - 1))
        return self._playback_channels[idx].label

    def _fft_window_ms(self) -> int:
        if self._fft_window_combo is None:
            return DEFAULT_LIVE_FFT_WINDOW_MS
        data = self._fft_window_combo.currentData()
        return int(data) if data is not None else DEFAULT_LIVE_FFT_WINDOW_MS

    def _selected_signal(self) -> np.ndarray | None:
        label = self._selected_label()
        if label is None:
            return None
        return self._channel_map.get(label)

    def _update_live_spectrum(self, *_unused) -> None:
        sig = self._selected_signal()
        if sig is None or len(sig) == 0:
            empty_f, empty_db = live_spectrum(np.array([]), 0.0, self._fft_window_ms())
            self._spectrum.set_spectrum(empty_f, empty_db)
            return
        color = "#c9d1d9"
        if self._playback_channels:
            color = self._playback_channels[0].color
        freqs, db = live_spectrum(sig, self._cursor_pos, self._fft_window_ms())
        self._spectrum.set_spectrum(freqs, db, color=color)

    def _update_time_label(self) -> None:
        duration = self._scene.duration
        if self._time_label is not None:
            self._time_label.setText(format_play_time(self._cursor_pos))
        if self._duration_label is not None:
            self._duration_label.setText(format_play_time(duration))
        if (
            self._seek_slider is not None
            and not self._slider_dragging
            and duration > 0.0
        ):
            self._seek_slider.blockSignals(True)
            self._seek_slider.setValue(
                int(round(10000.0 * self._cursor_pos / duration))
            )
            self._seek_slider.blockSignals(False)

    def _sync_transport(self) -> None:
        if self._play_pause_btn is None or self._play_icon is None:
            return
        if self._playing_label is not None:
            self._play_pause_btn.setIcon(self._pause_icon)
            self._play_pause_btn.setToolTip("Pause (Space)")
        else:
            self._play_pause_btn.setIcon(self._play_icon)
            self._play_pause_btn.setToolTip("Play (Space)")

    def _set_cursor(self, time_s: float) -> None:
        self._cursor_pos = max(0.0, min(time_s, self._scene.duration))
        self._scene.set_cursor(self._cursor_pos)
        self._update_time_label()
        self._update_live_spectrum()

    def _time_from_slider(self, value: int) -> float:
        duration = self._scene.duration
        if duration <= 0.0:
            return 0.0
        return duration * max(0, min(int(value), 10000)) / 10000.0

    def _on_slider_pressed(self) -> None:
        self._slider_dragging = True
        self._was_playing = self._playing_label
        if self._was_playing:
            self._stop_all(silent=True)

    def _on_slider_moved(self, value: int) -> None:
        self._set_cursor(self._time_from_slider(value))

    def _on_slider_released(self) -> None:
        self._slider_dragging = False
        if self._seek_slider is not None:
            self._set_cursor(self._time_from_slider(self._seek_slider.value()))
        if self._was_playing:
            self._play_selected()
        self._was_playing = None

    def _block_vispy_navigation(self, event) -> None:
        event.handled = True

    def _time_from_mouse(self, event) -> float | None:
        pos = getattr(event, "pos", None)
        if pos is None:
            return None
        return self._scene.time_at_canvas_pos((float(pos[0]), float(pos[1])))

    def _on_mouse_press(self, event) -> None:
        if getattr(event, "button", None) != 1:
            return
        time_s = self._time_from_mouse(event)
        if time_s is None:
            return
        event.handled = True
        self._dragging = True
        self._was_playing = self._playing_label
        if self._was_playing:
            self._stop_all()
        self._set_cursor(time_s)

    def _on_mouse_move(self, event) -> None:
        if not self._dragging:
            return
        time_s = self._time_from_mouse(event)
        if time_s is None:
            return
        event.handled = True
        self._set_cursor(time_s)

    def _on_mouse_release(self, event) -> None:
        if not self._dragging:
            return
        self._dragging = False
        time_s = self._time_from_mouse(event)
        if time_s is not None:
            event.handled = True
            self._set_cursor(time_s)
        if self._was_playing:
            label = self._selected_label()
            if label:
                self._play(label, self._cursor_pos)
        self._was_playing = None

    def _toggle_play_pause(self) -> None:
        if self._playing_label is not None:
            self._pause()
            return
        self._play_selected()

    def _pause(self) -> None:
        self._stop_all(silent=True)
        if self._status_label is not None:
            self._status_label.setText("Paused")

    def _stop_reset(self) -> None:
        self._stop_all(silent=True)
        self._set_cursor(0.0)
        if self._status_label is not None:
            self._status_label.setText("Stopped")

    def _play_selected(self) -> None:
        label = self._selected_label()
        if label:
            self._play(label, self._cursor_pos)

    def _play(self, label: str, start_time: float) -> None:
        self._stop_all(silent=True)
        sig = self._channel_map.get(label)
        if sig is None or len(sig) == 0:
            return

        duration = len(sig) / FS_HZ
        if start_time >= duration - 1e-6:
            start_time = 0.0
            self._cursor_pos = 0.0

        start_sample = int(start_time * FS_HZ)
        start_sample = max(0, min(start_sample, len(sig) - 1))
        remaining = sig[start_sample:]
        if len(remaining) == 0:
            return

        self._playing_label = label
        self._play_start_time = start_time
        self._play_n_samples = len(remaining)
        sd.play(remaining, samplerate=int(FS_HZ))
        self._wall_origin = time.monotonic()
        if self._status_label is not None:
            self._status_label.setText(f"Playing {label}")
        self._sync_transport()
        self._cursor_timer.start()

    def _tick_cursor(self) -> None:
        if self._playing_label is None:
            self._cursor_timer.stop()
            return
        elapsed = time.monotonic() - self._wall_origin
        play_duration = self._play_n_samples / FS_HZ
        if elapsed >= play_duration:
            self._set_cursor(self._play_start_time + play_duration)
            self._on_playback_done()
            return
        self._set_cursor(self._play_start_time + elapsed)

    def _on_playback_done(self) -> None:
        self._playing_label = None
        self._cursor_timer.stop()
        self._sync_transport()
        if self._status_label is not None:
            self._status_label.setText("Stopped")

    def _stop_all(self, *, silent: bool = False) -> None:
        self._playing_label = None
        self._cursor_timer.stop()
        sd.stop()
        self._sync_transport()
        if not silent and self._status_label is not None and not self._dragging:
            self._status_label.setText("Stopped")

    def _save_selected(self) -> None:
        label = self._selected_label()
        sig = self._channel_map.get(label or "")
        if sig is None or label is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save WAV",
            f"{label}.wav",
            "WAV files (*.wav)",
        )
        if path:
            write_wav(Path(path), sig)
            if self._status_label is not None:
                self._status_label.setText(f"Saved {path}")


def run_audio_player_window(
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
        lambda: AudioPlayerWindow(
            session_ids,
            base_dir,
            settings=settings,
            initial_preset=initial_preset,
        ),
    )
