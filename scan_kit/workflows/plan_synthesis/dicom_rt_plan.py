"""Load RT Ion DICOM plans and convert them to input_map rows."""



from __future__ import annotations



import math

from pathlib import Path

from typing import Any



import pandas as pd



from .input_map import DEFAULT_BEAM_SIZE, DEFAULT_CURRENT_A

from .layouts.rectangular_field import FAST_AXIS_X

from .plan_import import ImportSpot, PlanImportError, import_spots_to_input_map

from .spot_order import SPOT_ORDER_PLAN



try:

    import pydicom

except ImportError:  # pragma: no cover - exercised when optional dep missing

    pydicom = None  # type: ignore[assignment]





class DicomPlanError(PlanImportError):

    """Raised when an RT Ion DICOM plan cannot be parsed."""





def _require_pydicom() -> Any:

    if pydicom is None:

        raise DicomPlanError(

            "pydicom is required for DICOM RT plan import. "

            "Install with: pip install pydicom"

        )

    return pydicom





def is_rt_ion_plan_file(path: Path) -> bool:

    """Return True when *path* looks like an RT Ion plan DICOM file."""

    return path.is_file() and path.suffix.lower() == ".dcm"





def rt_plan_label_from_path(path: Path) -> str:

    """Return the RTPlanLabel from a DICOM file, or the file stem."""

    module = _require_pydicom()

    ds = module.dcmread(path, stop_before_pixels=True, force=True)

    label = str(ds.get("RTPlanLabel", "") or "").strip()

    return label or path.stem





def _flatten_ds_list(seq: Any) -> list[Any]:

    if seq is None:

        return []

    try:

        return [item for item in seq if item is not None]

    except TypeError:

        return []





def _scanning_spot_size_xy_mm_from_cp(cp: Any) -> tuple[float, float] | None:

    raw = cp.get("ScanningSpotSize", None)

    if raw is None:

        return None

    try:

        seq = list(raw)

        if len(seq) >= 2:

            fx, fy = float(seq[0]), float(seq[1])

            if (

                math.isfinite(fx)

                and math.isfinite(fy)

                and fx > 0.0

                and fy > 0.0

                and fx < 500.0

                and fy < 500.0

            ):

                return fx, fy

    except (TypeError, ValueError):

        pass

    return None





def _beam_size_mm(

    fwhm_xy: tuple[float, float] | None,

    *,

    use_dicom_beam_size: bool,

    default_beam_size: float,

) -> float:

    if use_dicom_beam_size and fwhm_xy is not None:

        return float((fwhm_xy[0] + fwhm_xy[1]) / 2.0)

    return default_beam_size





def _iter_planned_spot_slots_from_dataset(ds: Any):

    for beam in _flatten_ds_list(ds.get("IonBeamSequence")):

        for cp in _flatten_ds_list(beam.get("IonControlPointSequence")):

            n = int(cp.get("NumberOfScanSpotPositions", 0) or 0)

            sm = cp.get("ScanSpotPositionMap", None)

            if not n or sm is None:

                continue

            energy = float(cp.get("NominalBeamEnergy", 0.0) or 0.0)

            coords = list(sm)

            limit = min(len(coords), 2 * n)

            weights: list[float] | None

            raw_w = cp.get("ScanSpotMetersetWeights", None)

            if raw_w is None:

                weights = None

            else:

                try:

                    weights = [float(x) for x in raw_w]

                except (TypeError, ValueError):

                    weights = None

            fwhm_cp = _scanning_spot_size_xy_mm_from_cp(cp)

            for i in range(0, limit, 2):

                si = i // 2

                drop = False

                mu = float("nan")

                if weights is not None and si < len(weights):

                    mu = float(weights[si])

                    if not math.isfinite(mu) or mu <= 0.0:

                        drop = True

                x = float(coords[i])

                y = float(coords[i + 1])

                yield drop, x, y, energy, fwhm_cp, mu





def _spots_from_dataset(

    ds: Any,

    *,

    use_dicom_beam_size: bool,

    default_beam_size: float,

) -> list[ImportSpot]:

    spots: list[ImportSpot] = []

    for plan_index, (drop, x, y, energy, fwhm_cp, mu) in enumerate(

        _iter_planned_spot_slots_from_dataset(ds)

    ):

        if drop:

            continue

        spots.append(

            ImportSpot(

                x=x,

                y=y,

                energy=energy,

                charge=mu,

                beam_size=_beam_size_mm(

                    fwhm_cp,

                    use_dicom_beam_size=use_dicom_beam_size,

                    default_beam_size=default_beam_size,

                ),

                plan_index=plan_index,

            )

        )

    return spots





def dicom_to_input_map(

    path: str | Path,

    *,

    use_dicom_beam_size: bool = True,

    default_beam_size: float = DEFAULT_BEAM_SIZE,

    default_current: float = DEFAULT_CURRENT_A,

    spot_order: str = SPOT_ORDER_PLAN,

    fast_axis: str = FAST_AXIS_X,

) -> pd.DataFrame:

    """Parse an RT Ion DICOM plan and return an input_map DataFrame."""

    module = _require_pydicom()

    plan_path = Path(path)

    if not plan_path.is_file():

        raise DicomPlanError(f"DICOM plan file not found: {plan_path}")



    ds = module.dcmread(plan_path, stop_before_pixels=True, force=True)

    spots = _spots_from_dataset(

        ds,

        use_dicom_beam_size=use_dicom_beam_size,

        default_beam_size=default_beam_size,

    )

    if not spots:

        raise DicomPlanError("No planned spots with positive MU found in DICOM plan")



    return import_spots_to_input_map(

        spots,

        default_current=default_current,

        spot_order=spot_order,

        fast_axis=fast_axis,

    )





def validate_dicom_plan_path(path: Any) -> list[str]:

    text = str(path or "").strip()

    if not text:

        return ["Select an RT Ion DICOM plan file (.dcm)."]

    plan_path = Path(text)

    if not plan_path.is_file():

        return [f"DICOM plan file not found: {plan_path}"]

    if plan_path.suffix.lower() != ".dcm":

        return ["Plan file must have a .dcm extension."]

    try:

        _require_pydicom()

    except DicomPlanError as exc:

        return [str(exc)]

    return []


