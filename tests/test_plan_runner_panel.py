"""Plan Runner panel chrome, status apply, and download gating."""

from __future__ import annotations

from scan_kit.common.app_settings import AppSettings
from scan_kit.igx.rci_paths import (
    COMBINED_STATE,
    CONTROL_POINT_COUNT,
    CONTROL_POINT_INDEX,
    POINT_ENERGY,
    POINT_PROGRESS,
    POINTS_VALID,
    PROGRESS,
    READY_PERMIT,
    STATE,
    TIME_ELAPSED,
    TREATMENT_ACTIVE,
)
from scan_kit.workflows.plan_runner_panel import PlanRunnerPanel, _PercentBar


def _silent_settings() -> AppSettings:
    settings = AppSettings(last_rci_host="192.168.100.184")
    settings.save = lambda: None  # type: ignore[method-assign]
    return settings


def _panel() -> PlanRunnerPanel:
    return PlanRunnerPanel(app_settings=_silent_settings())


def test_panel_starts_disconnected(qapp) -> None:
    panel = _panel()
    try:
        assert panel._connect_btn.text() == "Connect"
        assert panel._upload_btn.isEnabled() is False
        assert panel._start_btn.isEnabled() is False
        assert panel._download_btn.isEnabled() is False
        assert panel._metric_permit._value._full == "—"
        assert "Connect" in panel._footer._full
        assert panel._host_edit.text() == "192.168.100.184"
    finally:
        panel.shutdown()


def test_percent_bar_tooltip(qapp) -> None:
    bar = _PercentBar("overall")
    bar.set_percent(None)
    assert "unknown" in bar.toolTip()
    bar.set_percent(12.5)
    assert bar._pct == 12.5
    assert "12.5%" in bar.toolTip()


def test_apply_status_dosing_unwraps_pairs_and_coaches_run(qapp) -> None:
    panel = _panel()
    try:
        panel._connected = True
        panel._plan_path_edit.setText(r"C:\plans\input_map.csv")
        panel._apply_status(
            {
                STATE: ["dosing", 1.0],
                READY_PERMIT: [True, 1.0],
                POINTS_VALID: [True, 1.0],
                PROGRESS: [0.4, 1.0],
                POINT_PROGRESS: [0.5, 1.0],
                CONTROL_POINT_INDEX: [3.0, 1.0],
                CONTROL_POINT_COUNT: [10.0, 1.0],
                POINT_ENERGY: [226.5, 1.0],
                TIME_ELAPSED: [75.2, 1.0],
                TREATMENT_ACTIVE: [False, 1.0],
                COMBINED_STATE: ["dosing", 1.0],
            }
        )
        assert panel._start_btn.isEnabled() is False
        assert panel._pause_btn.isEnabled() is True
        assert panel._stop_btn.isEnabled() is True
        assert panel._reset_btn.isEnabled() is False
        assert panel._progress_bar._pct == 40.0
        assert panel._point_bar._pct == 50.0
        assert panel._metric_point._value._full == "3 / 10"
        assert panel._metric_energy._value._full == "226.50 MeV"
        assert panel._metric_elapsed._value._full == "1:15"
        assert panel._metric_permit._value._full == "Granted"
        assert "Treatment active" not in panel._state_sub._full
        assert panel._state_label.text() == "DOSING"
        assert panel._footer._full.startswith("Running")
        assert "10 control points" in panel._plan_hint._full
        assert panel._download_btn.isEnabled() is True
    finally:
        panel.shutdown()


def test_apply_status_false_treatment_pair_is_not_active(qapp) -> None:
    panel = _panel()
    try:
        panel._connected = True
        panel._apply_status({TREATMENT_ACTIVE: [False, 1.0], COMBINED_STATE: ""})
        assert "Treatment active" not in panel._state_sub._full
    finally:
        panel.shutdown()


def test_download_stays_disabled_while_busy(qapp) -> None:
    panel = _panel()
    try:
        panel._connected = True
        panel._refresh_download_enabled()
        assert panel._download_btn.isEnabled() is True
        panel._set_footer("Downloading session…", busy=True)
        assert panel._download_btn.isEnabled() is False
        panel._apply_status({STATE: "completed", READY_PERMIT: True})
        assert panel._download_btn.isEnabled() is False
        panel._hold_footer = True
        panel._set_footer(r"Session saved to C:\out\run.zip", busy=False)
        assert panel._download_btn.isEnabled() is True
        panel._apply_status({STATE: "completed"})
        assert "Session saved" in panel._footer._full
    finally:
        panel.shutdown()


def test_connected_info_sets_session_root_and_enables(qapp) -> None:
    panel = _panel()
    try:
        panel._on_connected(
            {
                "host": "192.168.100.184",
                "version": "26.08.051704",
                "device_type": "RCI",
                "session_directory": "/root/reports/session/",
                "status": {STATE: "locked", READY_PERMIT: False},
            }
        )
        assert panel._connected is True
        assert panel._connect_btn.text() == "Disconnect"
        assert panel._conn_identity.text() == "RCI"
        assert panel._session_remote_root == "/root/reports/session"
        assert panel._download_btn.isEnabled() is True
        assert panel._upload_btn.isEnabled() is False
        panel._plan_path_edit.setText(r"C:\plans\input_map.csv")
        assert panel._upload_btn.isEnabled() is True
    finally:
        panel.shutdown()


def test_session_hint_follows_zip_name(qapp) -> None:
    panel = _panel()
    try:
        panel._session_path_edit.setText(r"C:\out\abc.zip")
        assert "abc" in panel._session_hint._full
        assert "/root/reports/session/abc" in panel._session_hint._full
    finally:
        panel.shutdown()
