"""Plan Runner status formatting and control gating."""

from pathlib import Path

from scan_kit.igx.rci_paths import (
    COMBINED_POINTS_OK,
    CONTROL_POINT_COUNT,
    CONTROL_POINT_INDEX,
    POINTS_VALID,
    READY_PERMIT,
    READY_PERMIT_REASON,
    STATE,
)
from scan_kit.workflows.plan_runner.status import (
    coach_message,
    control_enables,
    default_session_zip_path,
    format_elapsed,
    format_energy,
    io_bool,
    io_number,
    io_text,
    point_fraction,
    progress_percent,
    resolve_session_download,
    session_download_hint,
    unwrap_io,
)


def test_unwrap_io_pairs() -> None:
    assert unwrap_io([0.25, 1710000000.0]) == 0.25
    assert unwrap_io([[True, 1.0], [False, 2.0]]) is True
    assert unwrap_io(None) is None


def test_progress_percent_fraction_and_already_percent() -> None:
    assert progress_percent(0.25) == 25.0
    assert progress_percent(0.0) == 0.0
    assert progress_percent(1.0) == 100.0
    assert progress_percent(45.0) == 45.0
    assert progress_percent(None) is None
    assert progress_percent("nope") is None
    assert progress_percent([0.4, 99.0]) == 40.0


def test_point_fraction() -> None:
    assert point_fraction({}) == "—"
    assert (
        point_fraction({CONTROL_POINT_INDEX: 3, CONTROL_POINT_COUNT: 7600}) == "3 / 7600"
    )
    assert (
        point_fraction(
            {CONTROL_POINT_INDEX: [3.0, 1.0], CONTROL_POINT_COUNT: [7600.0, 1.0]}
        )
        == "3 / 7600"
    )


def test_io_bool_unwraps_pairs_and_strings() -> None:
    assert io_bool(True) is True
    assert io_bool(False) is False
    assert io_bool([True, 1.0]) is True
    assert io_bool([False, 1.0]) is False
    assert io_bool("false") is False
    assert io_bool("granted") is True
    assert io_bool(None) is False


def test_io_number_and_text() -> None:
    assert io_number([226.5, 1.0]) == 226.5
    assert io_number("") is None
    assert io_text(["locked", 1.0]) == "locked"
    assert io_text(None) == ""


def test_format_energy_and_elapsed() -> None:
    assert format_energy(None) == "—"
    assert format_energy([226.5, 1.0]) == "226.50 MeV"
    assert format_elapsed(12.3) == "12.3 s"
    assert format_elapsed([75.2, 1.0]) == "1:15"
    assert format_elapsed(3725) == "1:02:05"
    assert format_elapsed(None) == "—"


def test_control_enables_start_follows_permit_not_state_name() -> None:
    status = {READY_PERMIT: True}
    enables = control_enables(status, connected=True)
    assert enables["start"] is True
    assert enables["reset"] is True


def test_control_enables_locked_with_permit() -> None:
    status = {
        STATE: "locked",
        POINTS_VALID: True,
        READY_PERMIT: True,
    }
    enables = control_enables(status, connected=True)
    assert enables["start"] is True
    assert enables["pause"] is False
    assert enables["stop"] is False
    assert enables["reset"] is True


def test_control_enables_dosing() -> None:
    status = {
        STATE: "dosing",
        POINTS_VALID: True,
        READY_PERMIT: True,
    }
    enables = control_enables(status, connected=True)
    assert enables["start"] is False
    assert enables["pause"] is True
    assert enables["stop"] is True
    assert enables["reset"] is False


def test_control_enables_paused_can_resume() -> None:
    status = {STATE: ["paused", 1.0], READY_PERMIT: [True, 1.0]}
    enables = control_enables(status, connected=True)
    assert enables["start"] is True
    assert enables["pause"] is False
    assert enables["stop"] is True
    assert enables["reset"] is True


def test_control_enables_ready_without_permit() -> None:
    status = {
        STATE: "ready",
        POINTS_VALID: True,
        READY_PERMIT: False,
    }
    enables = control_enables(status, connected=True)
    assert enables["start"] is False


def test_control_enables_disconnected() -> None:
    enables = control_enables({STATE: "ready", POINTS_VALID: True}, connected=False)
    assert enables == {"start": False, "pause": False, "stop": False, "reset": False}


def test_coach_message_follows_run_state() -> None:
    assert "Connect" in coach_message(connected=False, has_plan=False, status={})
    assert "Browse" in coach_message(connected=True, has_plan=False, status={})
    assert "Upload" in coach_message(
        connected=True, has_plan=True, status={POINTS_VALID: False}
    )
    assert "Start is enabled" in coach_message(
        connected=True,
        has_plan=True,
        status={POINTS_VALID: True, READY_PERMIT: True},
    )
    assert coach_message(
        connected=True,
        has_plan=True,
        status={STATE: "dosing", READY_PERMIT: True, POINTS_VALID: True},
    ).startswith("Running")
    assert coach_message(
        connected=True, has_plan=True, status={STATE: "paused"}
    ).startswith("Paused")
    assert "Download" in coach_message(
        connected=True, has_plan=True, status={STATE: "completed"}
    )
    assert "fault" in coach_message(
        connected=True, has_plan=True, status={STATE: "fault"}
    ).lower()
    assert "Door open" in coach_message(
        connected=True,
        has_plan=True,
        status={
            POINTS_VALID: True,
            READY_PERMIT: False,
            READY_PERMIT_REASON: ["Door open", 1.0],
        },
    )
    assert "Upload" not in coach_message(
        connected=True, has_plan=True, status={COMBINED_POINTS_OK: True}
    )


def test_resolve_session_download() -> None:
    assert resolve_session_download("", "/root/reports/session") is None
    assert resolve_session_download("/root/reports/session/abc", "/root/reports/session") is None
    assert resolve_session_download(r"C:\data", "/root/reports/session") is None
    remote, zip_path = resolve_session_download(
        r"C:\data\abc.zip", "/root/reports/session/"
    )
    assert remote == "/root/reports/session/abc"
    assert zip_path == r"C:\data\abc.zip"
    remote, zip_path = resolve_session_download(
        r"C:\data\ABC.ZIP", "/root/reports/session"
    )
    assert remote.endswith("/ABC")


def test_default_session_zip_path() -> None:
    assert default_session_zip_path("", "") == "session.zip"
    assert default_session_zip_path("", r"C:\out") == str(Path(r"C:\out") / "session.zip")
    assert default_session_zip_path(r"C:\out\run.zip", "") == r"C:\out\run.zip"
    assert default_session_zip_path(r"C:\out", "") == str(Path(r"C:\out") / "session.zip")
    assert default_session_zip_path(
        "/root/reports/session/abc", r"C:\out"
    ) == str(Path(r"C:\out") / "abc.zip")


def test_session_download_hint() -> None:
    assert "abc" in session_download_hint(r"C:\out\abc.zip", "/root/reports/session")
    assert "Remote folder" in session_download_hint(
        "/root/reports/session/abc", "/root/reports/session"
    )
    assert "under" in session_download_hint("", "/root/reports/session")
