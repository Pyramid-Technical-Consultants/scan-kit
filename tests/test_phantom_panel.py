"""Phantom Synthesis tab: phantom presets, the panel's background write, and the reference RTDOSE."""

from __future__ import annotations

import time

import numpy as np
import pytest

from scan_kit.dicom import StudyIndex
from scan_kit.dicom.synthetic import PHANTOMS
from scan_kit.workflows.phantom_panel import SPECS, ct_size, write_study

SMALL = {s.key: s.default for s in SPECS} | {"pixel": 4.0, "slice": 4.0, "reference": False}


@pytest.mark.parametrize("phantom", list(PHANTOMS))
def test_presets_write_their_inserts(tmp_path, phantom) -> None:
    write_study(tmp_path, SMALL | {"phantom": phantom, "fractions": 3})
    case = StudyIndex.scan(tmp_path).load()
    inserts = PHANTOMS[phantom][1]
    assert [r.name for r in case.structures.rois] == ["BODY", "PTV", "RING", *inserts]
    assert case.ct.hu.shape[::-1] == ct_size(4.0, 4.0) and case.plan().fractions == 3
    assert (case.ct.hu > 500).any() == ("BONE" in inserts) and (case.ct.hu == -750).any() == ("LUNG" in inserts)


def test_panel_writes_in_the_background_and_points_patient_qa_at_it(qapp, tmp_path) -> None:
    from scan_kit.common.user_store import PREF_LAST_DICOM_DIR, prefs_get
    from scan_kit.workflows.phantom_panel import PhantomSynthesisPanel

    panel = PhantomSynthesisPanel()
    assert "CT 80×80×70" in panel._summary.text()  # the defaults are write_phantom's
    panel.write(tmp_path / "study", SMALL | {"energies": []})
    end = time.monotonic() + 30
    while panel._busy and time.monotonic() < end:
        qapp.processEvents()
    assert "at least one energy" in panel._status.text() and prefs_get(PREF_LAST_DICOM_DIR) is None
    panel.write(tmp_path / "study", SMALL)
    end = time.monotonic() + 30
    while panel._busy and time.monotonic() < end:
        qapp.processEvents()
    assert prefs_get(PREF_LAST_DICOM_DIR) == str(tmp_path / "study"), panel._status.text()
    assert StudyIndex.scan(tmp_path / "study").load().plan().beams


def test_reference_dose_is_the_plans_course_dose(gpu, tmp_path) -> None:
    note = write_study(tmp_path, SMALL | {"reference": True, "histories": 1_000_000, "fractions": 2})
    assert "reference RTDOSE" in note
    case = StudyIndex.scan(tmp_path).load()
    (dose,) = case.doses_for(case.plan())
    ptv = case.structures.roi("PTV")
    assert dose.summation.upper() == "PLAN" and float(np.max(dose.dose)) > 0.0 and ptv is not None
