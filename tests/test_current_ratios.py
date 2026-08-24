"""Tests for current ratios view loading and rendering."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from scan_kit.common.report_runner import capture_view_figure
from scan_kit.common.settings import ViewSettings
from scan_kit.views import current_ratios

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_TEST_DATA = _PROJECT_ROOT / "test_data"


def _first_session_id() -> str | None:
    if not _TEST_DATA.is_dir():
        return None
    for child in sorted(_TEST_DATA.iterdir()):
        if child.is_dir() and not child.name.startswith("."):
            return child.name
    return None


def test_load_current_ratios_skips_empty_timeslice_frames(monkeypatch) -> None:
    from scan_kit.common.session_source import SessionSource

    src = SessionSource(kind="directory", path=Path("/fake"), session_id="sess")
    input_map = pd.DataFrame({"energy": [200.0], "layer_id": [1]})
    empty = pd.DataFrame({"_layer_idx": pd.Series([], dtype=int)})
    nonempty = pd.DataFrame(
        {
            "ic1_current": [50.0] * 20,
            "ic2_current": [50.0] * 20,
        }
    )
    nonempty["_layer_idx"] = 0

    monkeypatch.setattr(
        "scan_kit.views.current_ratios.resolve_session_source",
        lambda sid, base: src,
    )
    monkeypatch.setattr(
        "scan_kit.views.current_ratios.load_session_csv",
        lambda s, name: input_map if name == "input_map.csv" else None,
    )
    monkeypatch.setattr(
        "scan_kit.views.current_ratios.load_session_timeslice_device_units",
        lambda s: [empty, nonempty],
    )

    result = current_ratios._load_current_ratios("sess", "/fake")
    assert result is not None
    assert len(result["energy"]) == 1
    assert float(result["energy"].iloc[0]) == 200.0


def _synthetic_session_data(*, with_ic3: bool = False) -> dict:
    energies = np.array([220.0, 200.0, 180.0], dtype=float)
    n = len(energies)
    samples = [np.linspace(90.0, 110.0, 25) for _ in range(n)]
    disp = np.array([float(np.mean(s)) for s in samples], dtype=float)

    data = {
        "energy": pd.Series(energies, dtype=float),
        "ic1_disp": disp,
        "ic2_disp": disp * 1.02,
        "ic1_disp_filt": disp,
        "ic2_disp_filt": disp * 1.02,
        "ic1_beam_samples": samples,
        "ic2_beam_samples": [s * 1.02 for s in samples],
    }
    data["ic21_raw"] = current_ratios._sym_pct(data["ic2_disp"], data["ic1_disp"])
    data["ic21_filt"] = current_ratios._sym_pct(
        data["ic2_disp_filt"],
        data["ic1_disp_filt"],
    )

    if with_ic3:
        data["ic3_disp"] = disp * 0.98
        data["ic3_disp_filt"] = disp * 0.98
        data["ic3_beam_samples"] = [s * 0.98 for s in samples]
        data["ic31_raw"] = current_ratios._sym_pct(data["ic3_disp"], data["ic1_disp"])
        data["ic31_filt"] = current_ratios._sym_pct(
            data["ic3_disp_filt"],
            data["ic1_disp_filt"],
        )
        data["ic32_raw"] = current_ratios._sym_pct(data["ic3_disp"], data["ic2_disp"])
        data["ic32_filt"] = current_ratios._sym_pct(
            data["ic3_disp_filt"],
            data["ic2_disp_filt"],
        )

    return data


@pytest.mark.parametrize("with_ic3", [False, True])
def test_run_renders_finish_view_with_heatmaps(with_ic3: bool, monkeypatch) -> None:
    """Exercise the full plot + colorbar + finish_view path used by the launcher."""

    def _fake_load(session_id: str, base_dir: str, *, bg_subtract: bool = False) -> dict:
        del session_id, base_dir, bg_subtract
        return _synthetic_session_data(with_ic3=with_ic3)

    monkeypatch.setattr(current_ratios, "_load_current_ratios", _fake_load)

    fig, skip = capture_view_figure(
        current_ratios.run,
        ["synthetic"],
        str(_TEST_DATA),
        ViewSettings(),
    )
    assert skip is None, skip
    assert fig is not None
    assert len(fig.get_axes()) > 0
    plt.close(fig)


def test_run_captures_figure_from_test_data() -> None:
    session_id = _first_session_id()
    if session_id is None:
        pytest.skip("test_data session folder not available")

    fig, skip = capture_view_figure(
        current_ratios.run,
        [session_id],
        str(_TEST_DATA),
        ViewSettings(),
    )
    assert skip is None, skip
    assert fig is not None
    plt.close(fig)
