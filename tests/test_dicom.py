"""Patient DICOM ingest on a synthetic phantom written by pydicom."""

from __future__ import annotations

import json

import numpy as np
import pytest

from scan_kit.dicom import DicomError, StudyIndex, gantry_to_patient, load_dose, mask_bits, rasterize, write_dose
from scan_kit.dicom.synthetic import BONE, LUNG, phantom_hu, write_phantom


@pytest.fixture(scope="module")
def hfs(tmp_path_factory):
    return write_phantom(tmp_path_factory.mktemp("hfs"))


def test_study_loads_ct_structures_and_plan(hfs) -> None:
    case = StudyIndex.scan(hfs.folder).load()
    ct = case.ct
    assert ct.hu.shape == (70, 80, 80) and ct.patient_position == "HFS"
    # Shuffled slices come back in ascending z with HU rescaled.
    z = ct.grid.to_patient(np.column_stack([np.zeros(70), np.zeros(70), np.arange(70)]))[:, 2]
    assert np.all(np.diff(z) > 0.0) and ct.hu.min() == -1000.0
    points = ct.grid.to_patient(np.stack(np.meshgrid(np.arange(80), np.arange(80), np.arange(70), indexing="ij"), -1))
    expect = phantom_hu(points[..., 0], points[..., 1], points[..., 2]).transpose(2, 1, 0)
    assert np.array_equal(ct.hu, expect.astype(np.float32))
    assert [r.name for r in case.structures.rois] == ["BODY", "PTV", "RING"]
    plan = case.plan()
    beam = plan.beams[0]
    assert plan.prescription == 2.0
    assert [layer.energy for layer in beam.layers] == [120.0, 130.0, 140.0]
    assert sum(float(layer.mu.sum()) for layer in beam.layers) == pytest.approx(beam.meterset)
    assert beam.isocenter == pytest.approx([0.0, 35.0, 0.0]) and beam.patient_position == "HFS"


def test_prone_ct_axes_flip(tmp_path) -> None:
    ph = write_phantom(tmp_path, position="HFP", size=(40, 40, 30), spacing=(3.0, 3.0, 4.0))
    ct = StudyIndex.scan(ph.folder).load().ct
    assert np.allclose(np.diag(ct.grid.axes), [-1.0, -1.0, 1.0])
    # A point in the bone slab lands on a bone voxel through the flipped axes.
    p = np.array([10.0, np.mean(BONE[1]), 0.0])
    i, j, k = np.rint(ct.grid.to_index(p)).astype(int)
    assert ct.hu[k, j, i] == 700.0
    p = np.array([10.0, np.mean(LUNG[1]), 0.0])
    i, j, k = np.rint(ct.grid.to_index(p)).astype(int)
    assert ct.hu[k, j, i] == -750.0


def test_ring_has_a_hole(hfs) -> None:
    case = StudyIndex.scan(hfs.folder).load()
    grid = case.ct.grid
    ring = rasterize(case.structures.roi("RING"), grid)
    k = int(np.rint(grid.to_index([0.0, 0.0, 0.0])[2]))

    def at(x, y):
        i, j, _ = np.rint(grid.to_index([x, y, 0.0])).astype(int)
        return bool(ring[k, j, i])

    assert at(-45.0, 25.0) and at(-15.0, 55.0) and not at(-30.0, 40.0) and not at(0.0, 40.0)
    # 40 mm square minus a 20 mm hole on 2 mm pixels.
    assert ring[k].sum() == (40 * 40 - 20 * 20) // 4
    bits = mask_bits(case.structures.rois, grid)
    assert np.array_equal((bits >> 2) & 1, ring.astype(np.uint32))
    assert np.all(bits[ring] & 1)  # the ring is inside BODY


def test_gantry_to_patient_follows_iec_61217() -> None:
    down = np.array([0.0, 0.0, -1.0])  # beam direction in the gantry frame
    assert gantry_to_patient(0.0, 0.0, "HFS") @ down == pytest.approx([0.0, 1.0, 0.0])  # anterior to posterior
    assert gantry_to_patient(90.0, 0.0, "HFS") @ down == pytest.approx([-1.0, 0.0, 0.0])  # from patient left
    assert gantry_to_patient(0.0, 0.0, "HFP") @ down == pytest.approx([0.0, -1.0, 0.0])  # prone: P to A
    # Couch 90 at gantry 90 turns the lateral beam along the patient axis.
    assert np.abs(gantry_to_patient(90.0, 90.0, "HFS") @ down) == pytest.approx([0.0, 0.0, 1.0])
    for pos in ("HFS", "HFP", "FFS", "FFP"):
        r = gantry_to_patient(37.0, 15.0, pos)
        assert r.T @ r == pytest.approx(np.eye(3)) and np.linalg.det(r) == pytest.approx(1.0)


def test_rtdose_round_trip(hfs, tmp_path) -> None:
    case = StudyIndex.scan(hfs.folder).load()
    rng = np.random.default_rng(1)
    dose = rng.uniform(0.0, 2.0, case.ct.grid.shape_zyx).astype(np.float32)
    path = write_dose(tmp_path / "RD.dcm", dose, case.ct.grid, case.ct, plan_uid=hfs.plan_uid,
                      provenance={"seed": 3, "histories": 10})
    back = load_dose(path)
    assert back.plan_uid == hfs.plan_uid and back.grid.frame_uid == hfs.frame_uid
    assert np.allclose(back.grid.origin, case.ct.grid.origin) and np.allclose(back.grid.spacing, case.ct.grid.spacing)
    np.testing.assert_allclose(back.dose, dose, rtol=1e-6, atol=1e-9)
    assert "not for clinical use" in back.comment
    import pydicom

    assert json.loads(pydicom.dcmread(str(path)).ImageComments) == {"histories": 10, "seed": 3}
    # The written dose is found beside the plan it references.
    (tmp_path / "study").mkdir()
    for f in (*hfs.ct_files, hfs.struct_file, hfs.plan_file, path):
        (tmp_path / "study" / f.name).write_bytes(f.read_bytes())
    again = StudyIndex.scan(tmp_path / "study").load()
    assert len(again.doses_for(again.plan())) == 1


def test_mismatched_frame_is_refused(hfs, tmp_path) -> None:
    import pydicom
    from pydicom.uid import generate_uid

    for f in (*hfs.ct_files, hfs.struct_file):
        (tmp_path / f.name).write_bytes(f.read_bytes())
    plan = pydicom.dcmread(str(hfs.plan_file))
    plan.FrameOfReferenceUID = generate_uid()
    plan.save_as(str(tmp_path / "RP.dcm"))
    index = StudyIndex.scan(tmp_path)
    case = index.load()
    assert case.plans == []  # the plan sits on its own frame, not this CT's
    ds = pydicom.dcmread(str(hfs.struct_file))
    for roi in ds.StructureSetROISequence:
        roi.ReferencedFrameOfReferenceUID = plan.FrameOfReferenceUID
    ds.ReferencedFrameOfReferenceSequence[0].FrameOfReferenceUID = plan.FrameOfReferenceUID
    ds.save_as(str(tmp_path / hfs.struct_file.name))
    with pytest.raises(DicomError):
        StudyIndex.scan(tmp_path).load(plan.FrameOfReferenceUID)


def test_missing_slice_is_refused(hfs, tmp_path) -> None:
    import pydicom

    files = sorted(hfs.ct_files, key=lambda f: float(pydicom.dcmread(str(f)).ImagePositionPatient[2]))
    for f in files[:10] + files[11:]:
        (tmp_path / f.name).write_bytes(f.read_bytes())
    with pytest.raises(DicomError, match="not uniform"):
        StudyIndex.scan(tmp_path).load()


def test_planning_ct_is_the_contoured_series(hfs, tmp_path) -> None:
    import pydicom
    from pydicom.uid import generate_uid

    other = generate_uid()
    for f in hfs.ct_files:
        (tmp_path / f.name).write_bytes(f.read_bytes())
        ds = pydicom.dcmread(str(f))
        ds.SeriesInstanceUID, ds.SOPInstanceUID = other, generate_uid()
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        ds.save_as(str(tmp_path / f"copy_{f.name}"))
    (tmp_path / hfs.struct_file.name).write_bytes(hfs.struct_file.read_bytes())
    wanted = pydicom.dcmread(str(hfs.ct_files[0])).SeriesInstanceUID
    assert StudyIndex.scan(tmp_path).load().ct.series_uid == wanted
    (tmp_path / hfs.struct_file.name).unlink()
    with pytest.raises(DicomError, match="no RTSTRUCT names one"):
        StudyIndex.scan(tmp_path).load()


def _plan_with(hfs, edit):
    import pydicom

    from scan_kit.dicom.plan import plan_from_dataset

    ds = pydicom.dcmread(str(hfs.plan_file))
    edit(ds, ds.IonBeamSequence[0])
    return plan_from_dataset(ds)


def _shifter(ds, number, setting, wet=None):
    from pydicom import Dataset

    s = Dataset()
    s.ReferencedRangeShifterNumber, s.RangeShifterSetting = number, setting
    if wet is not None:
        s.RangeShifterWaterEquivalentThickness = wet
    return s


def test_plan_range_shifters_and_metersets(hfs) -> None:
    from pydicom import Dataset

    def shifters(*numbers):
        def edit(ds, beam):
            beam.RangeShifterSequence = []
            for n in numbers:
                r = Dataset()
                r.RangeShifterNumber, r.RangeShifterID = n, f"RS{n}"
                beam.RangeShifterSequence.append(r)
        return edit

    def settings(*items):
        def edit(ds, beam):
            shifters(1, 2)(ds, beam)
            beam.IonControlPointSequence[0].RangeShifterSettingsSequence = [_shifter(ds, *i) for i in items]
        return edit

    # IN without a WET is NaN (the beam model's nominal WET), not silently OUT.
    layer = _plan_with(hfs, settings((1, "IN"))).beams[0].layers[0]
    assert layer.range_shifter == "RS1" and np.isnan(layer.range_shifter_wet)
    # Another shifter going OUT leaves the first IN.
    layer = _plan_with(hfs, settings((1, "IN", 40.0), (2, "OUT"))).beams[0].layers[0]
    assert layer.range_shifter == "RS1" and layer.range_shifter_wet == 40.0
    with pytest.raises(DicomError, match="more than one"):
        _plan_with(hfs, settings((1, "IN", 40.0), (2, "IN", 20.0)))
    with pytest.raises(DicomError, match="not in its RangeShifterSequence"):
        _plan_with(hfs, settings((3, "IN", 40.0)))
    with pytest.raises(DicomError, match="BeamMeterset"):
        _plan_with(hfs, lambda ds, _b: delattr(ds.FractionGroupSequence[0].ReferencedBeamSequence[0], "BeamMeterset"))
    # A beam the fraction group doesn't deliver is left out.
    assert _plan_with(hfs, lambda ds, b: setattr(b, "BeamNumber", 7)).beams == ()


def test_contours_off_the_grid_spacing_neither_cancel_nor_gap(hfs) -> None:
    from scan_kit.dicom.grid import VolumeGrid
    from scan_kit.dicom.structures import Roi

    grid = VolumeGrid((-20.0, -20.0, -20.0), (2.0, 2.0, 3.0), (21, 21, 14))
    sq = lambda z: np.array([[-10, -10, z], [10, -10, z], [10, 10, z], [-10, 10, z]], float)  # noqa: E731
    fine = rasterize(Roi(1, "fine", (0, 0, 0), "", tuple(sq(z) for z in np.arange(-12.0, 12.1, 2.0))), grid)
    coarse = rasterize(Roi(2, "coarse", (0, 0, 0), "", tuple(sq(z) for z in np.arange(-12.0, 12.1, 6.0))), grid)
    for mask in (fine, coarse):
        per = mask.sum(axis=(1, 2))
        on = np.flatnonzero(per)
        assert set(per[on]) == {100} and np.all(np.diff(on) == 1) and on.size >= 9


def test_ct_calibration_follows_mcsquare_lookup() -> None:
    from scan_kit.dicom.calibration import CtCalibration
    from scan_kit.views.dose_mc import load_tables

    cal = CtCalibration.mcsquare("default")
    names = [str(n) for n in load_tables()["mcsquare_names"]]
    # A threshold HU still belongs to the material below it; one HU above switches.
    assert names[cal.material(19)] == "Schneider_AT_AG_SI5"
    assert names[cal.material(19.5)] == "Schneider_SoftTissus"
    assert names[cal.material(-5000)] == "Schneider_Air" and names[cal.material(9000)] == "Schneider_Marrow_Bone15"
    assert np.allclose(cal.density(cal.hu_points[1:-1]), cal.densities[1:-1])
    assert cal.density(-1e6) == pytest.approx(0.001) and cal.density(1e5) > cal.densities[-1]
    hu = np.array([-700.0, -100.0, 0.0, 40.0, 250.0, 1250.0])
    mat, dens = cal.from_spr(cal.spr(hu))
    assert np.array_equal(mat, cal.material(hu)) and np.allclose(dens, cal.density(hu), rtol=1e-4)
    mat, dens = cal.voxels(np.full((2, 3, 4), 40.0))
    assert mat.dtype == np.uint8 and dens.dtype == np.float32 and mat.shape == (2, 3, 4)
    for hu in (np.arange(-1024.0, 3072.0).reshape(64, 64), np.linspace(-1024.0, 3071.0, 999)):
        mat, dens = cal.voxels(hu)
        assert np.array_equal(mat, cal.material(hu)) and np.array_equal(dens, cal.density(hu))
    assert len(cal.digest) == 16 and cal.digest != CtCalibration.mcsquare("Water_Phantom").digest
