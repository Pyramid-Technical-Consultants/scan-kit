"""Dose Volume on a synthetic DICOM study: plan recalc, logged fractions, gamma, goals and report."""

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
from scan_kit.qa.rbe import MCNAMARA


def _settle(qapp, study, timeout_s: float = 120.0) -> None:
    end = time.monotonic() + timeout_s
    while not (study._runs and study._export_button.isEnabled()) and time.monotonic() < end:
        qapp.processEvents()
    assert study._export_button.isEnabled(), study._mc_label.text() or study._study_label.text()


def test_dose_view_runs_a_study_with_logged_fractions_and_exports(qapp, gpu, tmp_path, monkeypatch) -> None:
    from scan_kit.views import dose_volume_window as dvw
    from scan_kit.views import study_source as ss

    ph = write_phantom(tmp_path / "study")
    case = StudyIndex.scan(ph.folder).load()
    plan = case.plan()
    tps, grid = patient_run(case.ct, CtCalibration.mcsquare("default"), BeamModel.read(ss.DEFAULT_BDL), plan,
                            plan_spots(plan), histories=1_000_000, seed=1)
    tps.step()
    write_dose(ph.folder / "RD_tps.dcm", tps.dose * plan.fractions, grid, case.ct, plan_uid=plan.sop_uid)
    s = plan_spots(plan)
    monkeypatch.setattr(ss, "delivery_from_session", lambda sid, _base: Delivery(s.x, s.y, s.energy, s.mu, sid))

    w = dvw.DoseVolumeWindow(["fx1", "fx2"], str(tmp_path), study=str(ph.folder))
    st = w._study
    try:
        assert w._dicom() and st.panel.isVisibleTo(w) and not w._session_box.isVisibleTo(w)
        st._histories.set_current("1000000")
        st._restart()
        _settle(qapp, st)
        # The one beam logged twice is two fractions, plus their sum.
        assert st._fraction_combo.count() == 3 and st._n_fractions == 1
        assert np.array_equal(st.doses[ss.DELIVERED], st.doses[ss.PLANNED])
        assert st.gamma is not None and st.gamma.rate > 0.95
        # The synthetic spots fall far short of the 2 Gy prescription, so D95% fails, as it should.
        goal, value, ok = st._goal_results[0]
        assert goal.text == "PTV: D95% >= 95%" and not ok
        assert value == pytest.approx(100.0 * st.dvh[ss.DELIVERED]["PTV"].dose_at(0.95) / plan.prescription)
        for show in ("dose", "difference", "gamma"):
            w._show_combo.set_current(show)
            w._on_show_changed()
            w._start_refresh()
            assert w._content is not None and w._scene.frame is not None
        assert w._scene.gamma and w._content[1]["gamma"] and "pass" in w._color_axis._title
        w._show_combo.set_current("difference")
        w._on_show_changed()
        w._start_refresh()
        assert w._scene.difference and w._content[0]

        # Axial PTV outline runs along voxel edges in mm; a click lands on the voxel under it.
        dx, dy, _dz = st._grid.spacing
        ws = w._workspace
        k, r = int(ws.cursor[2]), [roi.name for roi in case.structures.rois].index("PTV")
        cols = np.flatnonzero(((st.frame().bits[0][k] >> np.uint32(r)) & 1).any(axis=0))
        segs = ws._outline(0, k, r, (dx, dy))
        assert segs[:, 0].min() == pytest.approx(cols[0] * dx) and segs[:, 0].max() == pytest.approx((cols[-1] + 1) * dx)
        ws._pick(0, 5.5 * dx, 7.2 * dy)
        assert tuple(ws.cursor[:2]) == (5, 7) and ws.cursor[2] == k
        one = float(st.doses[ss.DELIVERED].sum())

        st._fraction_combo.setCurrentIndex(2)
        _settle(qapp, st)
        assert st._n_fractions == 2
        assert float(st.doses[ss.DELIVERED].sum()) == pytest.approx(2.0 * one, rel=0.03)

        # A variable RBE reweighs the finished doses without transporting again; gamma stays dose to dose.
        rate, ptv_1p1 = st.gamma.rate, st.dvh[ss.DELIVERED]["PTV"].mean
        st._rbe_combo.setCurrentIndex(st._rbe_combo.findData(MCNAMARA))
        assert st._alpha_beta.isEnabled() and st.done
        letd, weight = st.let_stats[ss.DELIVERED]["PTV"]
        assert 1.0 < letd < 10.0 and 1.0 < weight < 1.6
        assert st.dvh[ss.DELIVERED]["PTV"].mean == pytest.approx(ptv_1p1 * weight / 1.1, rel=0.02)
        assert st.gamma.rate == rate
        st._alpha_beta.setValue(10.0)
        assert st.let_stats[ss.DELIVERED]["PTV"][1] < weight

        html = st.export_report(tmp_path / "out")
        text = html.read_text(encoding="utf-8")
        assert "Gamma vs TPS" in text and "Clinical goals" in text and "data:image/png" in text
        assert "McNamara 2015" in text and "LETd (keV/µm)" in text
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
        st.load(str(bare))
        end = time.monotonic() + 60.0
        while "No RT Ion Plan" not in st._study_label.text() and time.monotonic() < end:
            qapp.processEvents()
        assert st._plan is None and not st._runs and not st._export_button.isEnabled()
        st._restart()
        assert not st._runs

        # Back to sessions: their controls return and the study's slices clear.
        w._set_source(dvw.SOURCE_SESSIONS)
        assert w._session_box.isVisibleTo(w) and not st.panel.isVisibleTo(w) and ws.frame is None
    finally:
        w.close()
        tps.close()
