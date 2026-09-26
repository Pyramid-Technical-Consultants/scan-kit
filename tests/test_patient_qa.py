"""Patient QA window on a synthetic study: plan recalc, logged fractions, gamma, goals and report."""

from __future__ import annotations

import json
import time

import numpy as np
import pydicom
import pytest

from scan_kit.dicom import StudyIndex, write_dose
from scan_kit.dicom.calibration import CtCalibration
from scan_kit.dicom.synthetic import write_phantom
from scan_kit.qa import BeamModel, Delivery, patient_run, plan_spots


def _settle(qapp, window, timeout_s: float = 120.0) -> None:
    end = time.monotonic() + timeout_s
    while not (window._runs and window._export_button.isEnabled()) and time.monotonic() < end:
        qapp.processEvents()
    assert window._export_button.isEnabled(), window._mc_label.text() or window._study_label.text()


def test_patient_qa_window_runs_logged_fractions_and_exports(qapp, gpu, tmp_path, monkeypatch) -> None:
    from scan_kit.views import patient_qa_window as pqw

    ph = write_phantom(tmp_path / "study")
    case = StudyIndex.scan(ph.folder).load()
    plan = case.plan()
    tps, grid = patient_run(case.ct, CtCalibration.mcsquare("default"), BeamModel.read(pqw.DEFAULT_BDL), plan,
                            plan_spots(plan), histories=1_000_000, seed=1)
    tps.step()
    write_dose(ph.folder / "RD_tps.dcm", tps.dose * plan.fractions, grid, case.ct, plan_uid=plan.sop_uid)
    s = plan_spots(plan)
    monkeypatch.setattr(pqw, "delivery_from_session", lambda sid, _base: Delivery(s.x, s.y, s.energy, s.mu, sid))

    w = pqw.PatientQaWindow(["fx1", "fx2"], str(tmp_path))
    try:
        w._histories.set_current("1000000")
        w.load(str(ph.folder))
        _settle(qapp, w)
        # The one beam logged twice is two fractions, plus their sum.
        assert w._fraction_combo.count() == 3 and w._n_fractions == 1
        assert np.array_equal(w._doses[pqw.DELIVERED], w._doses[pqw.PLANNED])
        assert w._gamma is not None and w._gamma.rate > 0.95
        # The synthetic spots fall far short of the 2 Gy prescription, so D95% fails, as it should.
        goal, value, ok = w._goal_results[0]
        assert goal.text == "PTV: D95% >= 95%" and not ok
        assert value == pytest.approx(100.0 * w._dvh[pqw.DELIVERED]["PTV"].dose_at(0.95) / plan.prescription)
        for key in (pqw.PLANNED, pqw.DELIVERED, pqw.DIFF, pqw.GAMMA):
            w._show.set_current(key)
            w._on_show_changed()
            assert w._shown() is not None
        one = float(w._doses[pqw.DELIVERED].sum())

        w._fraction_combo.setCurrentIndex(2)
        _settle(qapp, w)
        assert w._n_fractions == 2
        assert float(w._doses[pqw.DELIVERED].sum()) == pytest.approx(2.0 * one, rel=0.03)

        html = w.export_report(tmp_path / "out")
        text = html.read_text(encoding="utf-8")
        assert "Gamma vs TPS" in text and "Clinical goals" in text and "data:image/png" in text
        assert html.with_name(html.stem + "_dvh.csv").is_file()
        doses = sorted((tmp_path / "out").glob("*.dcm"))
        assert len(doses) == 2
        prov = json.loads(pydicom.dcmread(doses[0]).ImageComments)
        assert prov["fractions_summed"] == 2 and prov["inputs"]["ct_images"] == len(case.ct.sop_uids)
        assert pydicom.dcmread(doses[0]).DoseSummationType == "RECORD"

        # A study without a plan drops the last one's plan and runs instead of reusing them.
        bare = tmp_path / "bare"
        bare.mkdir()
        for f in ph.ct_files:
            (bare / f.name).write_bytes(f.read_bytes())
        w.load(str(bare))
        end = time.monotonic() + 60.0
        while "No RT Ion Plan" not in w._study_label.text() and time.monotonic() < end:
            qapp.processEvents()
        assert w._plan is None and not w._runs and not w._export_button.isEnabled()
        w._restart()
        assert not w._runs
    finally:
        w.close()
        tps.close()
