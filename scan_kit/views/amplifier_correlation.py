"""Amplifier command vs readback, field, and IC beam deflection angle (timeslice)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..common import (
    C_AMPLIFIER_CMD_X,
    C_AMPLIFIER_CMD_Y,
    C_AMPLIFIER_READBACK_X,
    C_AMPLIFIER_READBACK_Y,
    C_LAYER_ID,
    C_MAG_FIELD_X,
    C_MAG_FIELD_Y,
    DEFAULT_SESSION_COLORS,
    G2_AMPLIFIER_A_PER_V,
    GRID_KW,
    REFLINE_KW,
    TIMESLICE_AMPLIFIER_FIELD_COLS,
    detect_beam_on_mask,
    finish_view,
    fit_trend,
    make_trend_legend,
    resolve_concept_column,
    subtract_background_frames,
    trend_line_color,
    trend_session_prefix,
    view_grid,
)
from ..common.amplifier_settling import (
    amplifier_readback_tracks_command,
    amplifier_readback_tracks_command_axis,
    amplifier_settled_mask,
)
from ..common.ic_trajectory import (
    IcFanConvergence,
    aligned_beam_angles_mrad,
    ic_alignment_offsets,
    ic_fan_convergence,
)
from ..common.session_source import (
    SessionSource,
    load_session_csv,
    load_session_timeslice_device_units,
    resolve_session_source,
)
from ..common.timeslice_position_error import (
    TIMESLICE_POSITION_ERROR_COLS,
    frame_timeslice_chamber_position_arrays,
    resolve_session_timeslice_chamber_position_source,
)
from .timeslice_replay_common import (
    build_energy_lookups,
    resolve_col,
    resolve_frame_energy,
)

_SCATTER_ALPHA = 0.5
_SCATTER_SIZE = 4
_HEXBIN_THRESHOLD = 5000

# Pooled "super fit" across all selected sessions (drawn/labelled in black).
_COMBINED_COLOR = "k"
_COMBINED_PREFIX = "All: "

# Proton rest energy (MeV). A dipole deflects by an angle B*L/(B*rho), so
# normalising field to a reference momentum collapses every energy layer.
_PROTON_REST_ENERGY_MEV = 938.272
_REFERENCE_ENERGY_MEV = 250.0


def _momentum_mev(energy_mev: np.ndarray | float) -> np.ndarray:
    """Proton momentum (MeV/c) from kinetic energy (MeV)."""
    energy = np.asarray(energy_mev, dtype=float)
    return np.sqrt(energy * energy + 2.0 * energy * _PROTON_REST_ENERGY_MEV)


_REFERENCE_MOMENTUM_MEV = float(_momentum_mev(_REFERENCE_ENERGY_MEV))

_AMPLIFIER_VIEW_USECOLS = list(
    dict.fromkeys([*TIMESLICE_AMPLIFIER_FIELD_COLS, *TIMESLICE_POSITION_ERROR_COLS])
)

_DEFAULT_G2_CMD_REGISTER_TO_VOLTS = 31.444597756497412


def _first_g2_timeslice_paths(src: SessionSource) -> tuple[Path, Path] | None:
    """Return (device_units, raw) timeslice paths for the first G2 layer."""
    if src.kind != "directory":
        return None
    for layer_dir in sorted(src.path.glob("layer-*")):
        run_dir = layer_dir / "run-0"
        dev_path = run_dir / "timeslice_data_device_units.csv"
        raw_path = run_dir / "timeslice_data.csv"
        if dev_path.is_file() and raw_path.is_file():
            cols = pd.read_csv(dev_path, nrows=0).columns
            if resolve_concept_column(cols, C_MAG_FIELD_X) == "field_c_x":
                return dev_path, raw_path
    return None


def _g2_cmd_register_to_volts_scale(src: SessionSource) -> float:
    """Median raw c_x / device-units c_x for the first G2 correcting-coil layer."""
    paths = _first_g2_timeslice_paths(src)
    if paths is None:
        return _DEFAULT_G2_CMD_REGISTER_TO_VOLTS
    dev_path, raw_path = paths
    dev_c = pd.read_csv(dev_path, usecols=["c_x"], nrows=20_000)["c_x"].astype(float)
    raw_c = pd.read_csv(raw_path, usecols=["c_x"], nrows=20_000)["c_x"].astype(float)
    n = min(len(dev_c), len(raw_c))
    ok = (
        np.isfinite(dev_c[:n].values)
        & np.isfinite(raw_c[:n].values)
        & (np.abs(dev_c[:n].values) > 0.05)
    )
    if not ok.any():
        return _DEFAULT_G2_CMD_REGISTER_TO_VOLTS
    return float(np.median(raw_c[:n].values[ok] / dev_c[:n].values[ok]))


def _g2_field_physical_scale(src: SessionSource) -> float:
    """Scale G2 field_c setpoint gauss to coil field gauss for cmd-field analysis."""
    if _first_g2_timeslice_paths(src) is None:
        return 1.0
    return G2_AMPLIFIER_A_PER_V * _g2_cmd_register_to_volts_scale(src)


@dataclass(frozen=True)
class AmplifierCorrelationSamples:
    cmd_x: np.ndarray
    cmd_y: np.ndarray
    readback_x: np.ndarray
    readback_y: np.ndarray
    field_x: np.ndarray
    field_y: np.ndarray
    angle_x_mrad: np.ndarray
    angle_y_mrad: np.ndarray
    momentum: np.ndarray
    align_x: "_AxisAlignment" = field(default_factory=lambda: _AxisAlignment())
    align_y: "_AxisAlignment" = field(default_factory=lambda: _AxisAlignment())


@dataclass(frozen=True)
class _AxisAlignment:
    """Per-axis IC alignment: robust per-IC offsets plus the fan convergence.

    The offsets are the measured chamber alignment errors (subtracting them
    centres each IC's cloud on-axis without changing slope/angle).  The
    *difference* IC2−IC1 is the relative IC-to-IC misalignment; removing each
    independently cancels the fake common tilt it would otherwise inject.
    ``convergence`` reports where the alignment-corrected back-projected ray fan
    crosses the axis (≈ 0 mm) as a sanity check.
    """

    ic2_offset: float = 0.0
    ic1_offset: float = 0.0
    convergence: IcFanConvergence = field(
        default_factory=lambda: IcFanConvergence(float("nan"), float("nan"))
    )

    @property
    def offset_difference(self) -> float:
        """IC2 − IC1 offset (mm) = relative chamber misalignment."""
        return self.ic2_offset - self.ic1_offset


def _axis_alignment(ic2: np.ndarray, ic1: np.ndarray) -> _AxisAlignment:
    off2, off1 = ic_alignment_offsets(ic2, ic1)
    convergence = ic_fan_convergence(
        np.asarray(ic2, dtype=float) - off2,
        np.asarray(ic1, dtype=float) - off1,
    )
    return _AxisAlignment(off2, off1, convergence)


def _session_alignment_offsets(
    frames: list,
    chamber_source,
) -> tuple[_AxisAlignment, _AxisAlignment]:
    """Session-level IC alignment for X and Y (robust per-IC median offsets)."""
    ic2_x_parts: list[np.ndarray] = []
    ic1_x_parts: list[np.ndarray] = []
    ic2_y_parts: list[np.ndarray] = []
    ic1_y_parts: list[np.ndarray] = []

    for df in frames:
        beam_on = detect_beam_on_mask(df)
        if beam_on is None:
            continue
        chamber = frame_timeslice_chamber_position_arrays(df, chamber_source)
        if chamber is None:
            continue
        ic1_x, ic1_y, ic2_x, ic2_y = chamber
        ic2_x_parts.append(ic2_x[beam_on])
        ic1_x_parts.append(ic1_x[beam_on])
        ic2_y_parts.append(ic2_y[beam_on])
        ic1_y_parts.append(ic1_y[beam_on])

    if not ic2_x_parts:
        return _AxisAlignment(), _AxisAlignment()

    return (
        _axis_alignment(np.concatenate(ic2_x_parts), np.concatenate(ic1_x_parts)),
        _axis_alignment(np.concatenate(ic2_y_parts), np.concatenate(ic1_y_parts)),
    )


def _session_readback_field_drive(
    frames: list[pd.DataFrame],
    *,
    col_cmd_x: str,
    col_cmd_y: str,
    col_rb_x: str,
    col_rb_y: str,
) -> bool:
    """True when readback tracks command for field-correlation settling."""
    cmd_x_parts: list[np.ndarray] = []
    cmd_y_parts: list[np.ndarray] = []
    rb_x_parts: list[np.ndarray] = []
    rb_y_parts: list[np.ndarray] = []
    for df in frames:
        beam_on = detect_beam_on_mask(df)
        if beam_on is None or not beam_on.any():
            continue
        cmd_x_parts.append(df[col_cmd_x].values.astype(float)[beam_on])
        cmd_y_parts.append(df[col_cmd_y].values.astype(float)[beam_on])
        rb_x_parts.append(df[col_rb_x].values.astype(float)[beam_on])
        rb_y_parts.append(df[col_rb_y].values.astype(float)[beam_on])
    if not cmd_x_parts:
        return False
    return amplifier_readback_tracks_command(
        np.concatenate(cmd_x_parts),
        np.concatenate(cmd_y_parts),
        np.concatenate(rb_x_parts),
        np.concatenate(rb_y_parts),
    )


def _load_session_samples(
    session_id: str,
    base_dir: str,
    *,
    bg_subtract: bool = False,
) -> AmplifierCorrelationSamples | None:
    src = resolve_session_source(session_id, base_dir)
    if src is None:
        return None

    frames = load_session_timeslice_device_units(
        src, usecols=_AMPLIFIER_VIEW_USECOLS
    )
    if not frames:
        return None
    if bg_subtract:
        subtract_background_frames(frames)

    df0 = frames[0]
    col_cmd_x = resolve_col(df0.columns, C_AMPLIFIER_CMD_X)
    col_cmd_y = resolve_col(df0.columns, C_AMPLIFIER_CMD_Y)
    col_rb_x = resolve_col(df0.columns, C_AMPLIFIER_READBACK_X)
    col_rb_y = resolve_col(df0.columns, C_AMPLIFIER_READBACK_Y)
    col_bx = resolve_col(df0.columns, C_MAG_FIELD_X)
    col_by = resolve_col(df0.columns, C_MAG_FIELD_Y)
    if not all([col_cmd_x, col_cmd_y, col_rb_x, col_rb_y, col_bx, col_by]):
        return None

    field_scale = _g2_field_physical_scale(src)

    chamber_source = resolve_session_timeslice_chamber_position_source(src, frames)
    if chamber_source is None:
        return None

    align_x, align_y = _session_alignment_offsets(frames, chamber_source)
    off2_x, off1_x = align_x.ic2_offset, align_x.ic1_offset
    off2_y, off1_y = align_y.ic2_offset, align_y.ic1_offset

    readback_field_drive = _session_readback_field_drive(
        frames,
        col_cmd_x=col_cmd_x,
        col_cmd_y=col_cmd_y,
        col_rb_x=col_rb_x,
        col_rb_y=col_rb_y,
    )

    layer_col = resolve_col(df0.columns, C_LAYER_ID)
    input_map = load_session_csv(src, "input_map.csv")
    energy_lookups = build_energy_lookups(input_map) if input_map is not None else None
    energy_by_layer, energy_by_idx = energy_lookups or (None, {})

    cmd_x_parts: list[np.ndarray] = []
    cmd_y_parts: list[np.ndarray] = []
    rb_x_parts: list[np.ndarray] = []
    rb_y_parts: list[np.ndarray] = []
    bx_parts: list[np.ndarray] = []
    by_parts: list[np.ndarray] = []
    angle_x_parts: list[np.ndarray] = []
    angle_y_parts: list[np.ndarray] = []
    momentum_parts: list[np.ndarray] = []

    for frame_idx, df in enumerate(frames):
        beam_on = detect_beam_on_mask(df)
        if beam_on is None:
            continue

        chamber = frame_timeslice_chamber_position_arrays(df, chamber_source)
        if chamber is None:
            continue
        ic1_x, ic1_y, ic2_x, ic2_y = chamber

        cmd_x = df[col_cmd_x].values.astype(float)
        cmd_y = df[col_cmd_y].values.astype(float)
        rb_x = df[col_rb_x].values.astype(float)
        rb_y = df[col_rb_y].values.astype(float)
        field_x = df[col_bx].values.astype(float)
        field_y = df[col_by].values.astype(float)
        if field_scale != 1.0:
            field_x = field_x * field_scale
            field_y = field_y * field_scale

        angle_x = aligned_beam_angles_mrad(
            ic2_x, ic1_x, ic2_offset=off2_x, ic1_offset=off1_x
        )
        angle_y = aligned_beam_angles_mrad(
            ic2_y, ic1_y, ic2_offset=off2_y, ic1_offset=off1_y
        )

        sample_mask = beam_on & amplifier_settled_mask(
            cmd_x,
            cmd_y,
            rb_x,
            rb_y,
            field_x=field_x,
            field_y=field_y,
            readback_field_drive=readback_field_drive,
        )
        sample_mask &= np.isfinite(angle_x) & np.isfinite(angle_y)
        if not sample_mask.any():
            continue

        energy = resolve_frame_energy(
            df,
            frame_idx,
            energy_by_layer=energy_by_layer,
            energy_by_idx=energy_by_idx,
            layer_col=layer_col,
        )
        momentum = (
            float(_momentum_mev(energy)) if energy is not None else float("nan")
        )

        n_kept = int(sample_mask.sum())
        bx_parts.append(field_x[sample_mask])
        by_parts.append(field_y[sample_mask])
        cmd_x_parts.append(cmd_x[sample_mask])
        cmd_y_parts.append(cmd_y[sample_mask])
        rb_x_parts.append(rb_x[sample_mask])
        rb_y_parts.append(rb_y[sample_mask])
        angle_x_parts.append(angle_x[sample_mask])
        angle_y_parts.append(angle_y[sample_mask])
        momentum_parts.append(np.full(n_kept, momentum, dtype=float))

    if not cmd_x_parts:
        return None

    return AmplifierCorrelationSamples(
        cmd_x=np.concatenate(cmd_x_parts),
        cmd_y=np.concatenate(cmd_y_parts),
        readback_x=np.concatenate(rb_x_parts),
        readback_y=np.concatenate(rb_y_parts),
        field_x=np.concatenate(bx_parts),
        field_y=np.concatenate(by_parts),
        angle_x_mrad=np.concatenate(angle_x_parts),
        angle_y_mrad=np.concatenate(angle_y_parts),
        momentum=np.concatenate(momentum_parts),
        align_x=align_x,
        align_y=align_y,
    )


def _finite_pair(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def _scatter_or_hexbin(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    *,
    color: str,
    use_hexbin: bool = True,
) -> None:
    x, y = _finite_pair(x, y)
    if x.size == 0:
        return
    if use_hexbin and x.size >= _HEXBIN_THRESHOLD and color == "#1f77b4":
        ax.hexbin(x, y, gridsize=45, cmap="Blues", mincnt=1, linewidths=0)
    else:
        ax.scatter(x, y, s=_SCATTER_SIZE, alpha=_SCATTER_ALPHA, color=color, edgecolors="none")


def _linear_residuals(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float, np.ndarray] | None:
    """Linear fit y ≈ slope * x + intercept and per-sample residuals."""
    fit = fit_trend(x, y)
    if fit is None:
        return None
    x_f = x.astype(float)
    y_f = y.astype(float)
    finite = np.isfinite(x_f) & np.isfinite(y_f)
    residual = np.full(x_f.shape, np.nan, dtype=float)
    residual[finite] = y_f[finite] - fit.eval(x_f[finite])
    return fit.slope, fit.intercept, residual


_MIN_CUBIC_FIT_SAMPLES = 12


@dataclass(frozen=True)
class _CubicFit:
    """Cubic fit y ≈ c3·x³ + c2·x² + c1·x + c0 (coefficients in the data domain)."""

    c0: float  # offset (mrad)
    c1: float  # near-zero-field slope / gain (mrad per field unit)
    c2: float  # quadratic (mrad per field unit²)
    c3: float  # cubic (mrad per field unit³)
    residual: np.ndarray
    poly: "np.polynomial.Polynomial"


def _cubic_fit(x: np.ndarray, y: np.ndarray) -> _CubicFit | None:
    """Cubic fit capturing the large-angle curvature a straight line cannot follow.

    The polynomial is domain-scaled internally for conditioning, so it can be
    evaluated directly; coefficients are returned in the unscaled data domain.
    """
    x_f = np.asarray(x, dtype=float)
    y_f = np.asarray(y, dtype=float)
    finite = np.isfinite(x_f) & np.isfinite(y_f)
    if int(finite.sum()) < _MIN_CUBIC_FIT_SAMPLES:
        return None
    if float(np.ptp(x_f[finite])) <= 0.0:
        return None

    poly = np.polynomial.Polynomial.fit(x_f[finite], y_f[finite], 3)
    coef = poly.convert().coef  # ascending powers in the data domain
    c = [float(coef[i]) if i < len(coef) else 0.0 for i in range(4)]

    residual = np.full(x_f.shape, np.nan, dtype=float)
    residual[finite] = y_f[finite] - poly(x_f[finite])
    return _CubicFit(c[0], c[1], c[2], c[3], residual, poly)


@dataclass(frozen=True)
class _ArcFit:
    """Circular-arc beam-angle model: ``sin θ`` (not ``θ``) is linear in field.

    Our deflection dipoles are straight rectangular magnets with no pole-face
    rotation, so a beam entering on-axis follows a circular arc of radius
    ``ρ = p/(qB)`` and exits the (axial) field length ``L_eff`` at the angle

        ``sin θ = L_eff / ρ = (q·L_eff / p)·B``.

    The field-linear quantity is therefore ``sin θ``, and the deflection curve is
    ``θ = arcsin(k·B)``.  We fit a cubic of ``1000·sin θ`` (≈ ``θ`` in mrad for
    small angles, so the coefficients keep their mrad / mrad-per-kG meaning) vs
    energy-corrected field, then evaluate the model through ``arcsin``.  ``c1`` is
    the small-signal magnetic gain; ``c3`` absorbs any residual (non-geometric)
    field nonlinearity that the arc itself does not explain.
    """

    sin_fit: _CubicFit
    residual: np.ndarray

    @property
    def c0(self) -> float:
        return self.sin_fit.c0

    @property
    def c1(self) -> float:
        return self.sin_fit.c1

    @property
    def c3(self) -> float:
        return self.sin_fit.c3

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Beam angle (mrad) for energy-corrected field *x* via the arc model."""
        sin_theta = np.clip(self.sin_fit.poly(np.asarray(x, dtype=float)) / 1000.0, -1.0, 1.0)
        return np.arcsin(sin_theta) * 1000.0


def _arc_fit(field: np.ndarray, angle_mrad: np.ndarray) -> _ArcFit | None:
    """Circular-arc fit of beam angle vs field (``θ = arcsin(c₁·B + c₃·B³ + c₀)``).

    A straight deflection dipole is antisymmetric in field (reversing ``B``
    reverses ``θ``), so the field response carries only *odd* powers; we fit the
    physical odd form ``sin θ = c₁·B + c₃·B³`` plus a constant ``c₀`` for the
    zero-field alignment/remnant offset.  No quadratic term is included because
    it has no physical basis (it just fits noise).  The sin values are scaled by
    1000 so the coefficients stay in mrad-like units (matching the small-angle
    limit and the rest of the view).
    """
    field_f = np.asarray(field, dtype=float)
    angle_f = np.asarray(angle_mrad, dtype=float)
    sin_scaled = np.sin(angle_f / 1000.0) * 1000.0
    finite = np.isfinite(field_f) & np.isfinite(sin_scaled)
    if int(finite.sum()) < _MIN_CUBIC_FIT_SAMPLES:
        return None
    b = field_f[finite]
    if float(np.ptp(b)) <= 0.0:
        return None

    basis = np.vstack([np.ones_like(b), b, b**3]).T  # [1, B, B³] — odd + offset
    coef, *_ = np.linalg.lstsq(basis, sin_scaled[finite], rcond=None)
    c0, c1, c3 = float(coef[0]), float(coef[1]), float(coef[2])
    poly = np.polynomial.Polynomial([c0, c1, 0.0, c3])

    residual_sin = np.full(angle_f.shape, np.nan, dtype=float)
    residual_sin[finite] = sin_scaled[finite] - poly(b)
    sin_fit = _CubicFit(c0, c1, 0.0, c3, residual_sin, poly)

    residual = np.full(angle_f.shape, np.nan, dtype=float)
    predicted = np.arcsin(np.clip(poly(b) / 1000.0, -1.0, 1.0)) * 1000.0
    residual[finite] = angle_f[finite] - predicted
    return _ArcFit(sin_fit, residual)


def _format_legend_number(
    value: float, *, sig_figs: int = 3, signed: bool = True
) -> str:
    """Fixed-point legend value with about *sig_figs* significant figures.

    *signed* prefixes a leading ``+``/``-``; set it ``False`` for magnitudes
    such as a median absolute residual where a sign would be misleading.
    """
    if not np.isfinite(value):
        return "nan"
    sign = ("+" if value >= 0 else "-") if signed else ("-" if value < 0 else "")
    magnitude = abs(value)
    if magnitude == 0:
        return "+0" if signed else "0"
    exp = int(np.floor(np.log10(magnitude)))
    decimals = max(0, sig_figs - 1 - exp)
    text = f"{magnitude:.{decimals}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return f"{sign}{text}"


def _format_readback_fit_label(
    gain: float,
    offset_v: float,
    *,
    prefix: str = "",
) -> str:
    return (
        f"{prefix}gain {_format_legend_number(gain)} V/V   "
        f"offset {_format_legend_number(offset_v)} V"
    )


def _format_field_fit_label(
    gain: float,
    offset_g: float,
    *,
    prefix: str = "",
) -> str:
    return (
        f"{prefix}gain {_format_legend_number(gain)} G/V   "
        f"offset {_format_legend_number(offset_g)} G"
    )


def _energy_corrected_field(
    field: np.ndarray,
    momentum: np.ndarray,
) -> np.ndarray:
    """Field (G) normalised to the reference momentum."""
    momentum = np.asarray(momentum, dtype=float)
    scale = np.divide(
        _REFERENCE_MOMENTUM_MEV,
        momentum,
        out=np.full(momentum.shape, np.nan),
        where=np.isfinite(momentum) & (momentum > 0),
    )
    return np.asarray(field, dtype=float) * scale


# Energy-corrected field is reported in kilogauss (1 kG = 1000 G) so the gain
# reads as mrad/kG; the much smaller cubic is reported in µrad/kG³.
_GAUSS_PER_KILOGAUSS = 1.0e3
_CUBIC_MRAD_TO_MICRORAD = 1.0e3  # mrad/kG³ → µrad/kG³


def _energy_corrected_field_kg(
    field: np.ndarray,
    momentum: np.ndarray,
) -> np.ndarray:
    """Energy-corrected field in kilogauss, the x-domain for the beam-angle fits."""
    return _energy_corrected_field(field, momentum) / _GAUSS_PER_KILOGAUSS


# Full deflection model shown on the angle panels: straight-dipole circular arc,
# antisymmetric in field (only odd powers) plus a zero-field alignment offset.
_ANGLE_MODEL_EQUATION = (
    "\u03b8 = arcsin(gain\u00b7B + cubic\u00b7B\u00b3 + offset)\nstraight-dipole circular arc"
)


def _annotate_angle_model(ax) -> None:
    """Annotate the beam-angle model equation in the lower-left of *ax*."""
    ax.text(
        0.03,
        0.03,
        _ANGLE_MODEL_EQUATION,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        zorder=8,
        bbox=dict(
            boxstyle="round", facecolor="white", alpha=0.8, edgecolor="0.7"
        ),
    )


def _format_angle_fit_label(fit: "_ArcFit", *, prefix: str = "") -> str:
    # Circular-arc model θ = arcsin(cubic(B)) fitted in the kilogauss domain:
    # gain (small-signal dθ/dB) reads as mrad/kG; the residual cubic is tiny, so
    # report it in µrad/kG\u00b3.
    cubic_microrad = fit.c3 * _CUBIC_MRAD_TO_MICRORAD
    return (
        f"{prefix}gain {_format_legend_number(fit.c1)} mrad/kG   "
        f"cubic {_format_legend_number(cubic_microrad)} \u00b5rad/kG\u00b3   "
        f"offset {_format_legend_number(fit.c0)} mrad"
    )


def _format_alignment_label(align: "_AxisAlignment") -> str:
    """Per-IC alignment offsets, their difference, and the fan convergence.

    IC1/IC2 are the subtracted chamber offsets (mm); ``\u0394`` is the relative
    IC-to-IC misalignment; ``converge`` is where the alignment-corrected
    back-projected ray fan crosses the axis (≈ 0 mm sanity check).
    """
    conv = align.convergence
    conv_txt = (
        f"{_format_legend_number(conv.position_mm)} mm"
        if conv.is_valid
        else "n/a"
    )
    return (
        f"   align IC1 {_format_legend_number(align.ic1_offset)} "
        f"IC2 {_format_legend_number(align.ic2_offset)} "
        f"(\u0394 {_format_legend_number(align.offset_difference)}) mm   "
        f"converge {conv_txt}"
    )


def _plot_ecfield_angle(
    ax,
    samples: AmplifierCorrelationSamples,
    field_key: str,
    angle_key: str,
    *,
    color: str,
    session_id: str,
    n_sessions: int,
    align: "_AxisAlignment | None" = None,
    use_hexbin: bool = True,
    draw_fit: bool = True,
) -> tuple[str, tuple] | None:
    """Beam deflection angle vs energy-corrected field (circular-arc model).

    Companion to the residual panel.  When *draw_fit* is set the per-session
    arc curve (θ = arcsin(cubic(B))) is drawn; otherwise only the scatter is
    shown (e.g. when the pooled super-fit curve is overlaid instead).  *align*
    appends the per-IC alignment offsets and fan-convergence sanity check to the
    legend.
    """
    ecfield = _energy_corrected_field_kg(
        getattr(samples, field_key), samples.momentum
    )
    angle = getattr(samples, angle_key)
    x, _ = _finite_pair(ecfield, angle)
    if x.size == 0:
        return None
    fit = _arc_fit(ecfield, angle)
    if fit is None:
        return None
    _scatter_or_hexbin(ax, ecfield, angle, color=color, use_hexbin=use_hexbin)
    line_color = trend_line_color(color)
    if draw_fit:
        xs = np.linspace(float(np.min(x)), float(np.max(x)), 200)
        ax.plot(xs, fit.predict(xs), color=line_color, linewidth=1.2, zorder=6)
    prefix = trend_session_prefix(session_id, n_sessions=n_sessions)
    label = _format_angle_fit_label(fit, prefix=prefix)
    if align is not None:
        label = f"{label}\n{_format_alignment_label(align)}"
    return label, line_color


def _style_axis(ax, xlabel: str, ylabel: str, *, title: str | None = None) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    ax.grid(**GRID_KW)


@dataclass
class _PooledRow:
    """Accumulates one axis-row's data across sessions for the combined fit."""

    cmd: list[np.ndarray] = field(default_factory=list)
    readback: list[np.ndarray] = field(default_factory=list)
    drive: list[np.ndarray] = field(default_factory=list)
    field_g: list[np.ndarray] = field(default_factory=list)
    ecfield: list[np.ndarray] = field(default_factory=list)
    angle: list[np.ndarray] = field(default_factory=list)

    def add(
        self,
        samples: "AmplifierCorrelationSamples",
        cmd_key: str,
        rb_key: str,
        field_key: str,
        angle_key: str,
    ) -> None:
        cmd = getattr(samples, cmd_key)
        rb = getattr(samples, rb_key)
        field_g = getattr(samples, field_key)
        self.cmd.append(cmd)
        self.readback.append(rb)
        self.drive.append(rb if amplifier_readback_tracks_command_axis(cmd, rb) else cmd)
        self.field_g.append(field_g)
        self.ecfield.append(_energy_corrected_field_kg(field_g, samples.momentum))
        self.angle.append(getattr(samples, angle_key))

    def _cat(self, parts: list[np.ndarray]) -> np.ndarray | None:
        return np.concatenate(parts) if parts else None

    def readback_fit(self):
        """Pooled linear cmd→readback fit (the super fit for column 1)."""
        cmd = self._cat(self.cmd)
        rb = self._cat(self.readback)
        return fit_trend(cmd, rb) if cmd is not None else None

    def field_fit(self):
        """Pooled linear drive→field fit (the super fit for column 2)."""
        drive = self._cat(self.drive)
        field_g = self._cat(self.field_g)
        return fit_trend(drive, field_g) if drive is not None else None

    def angle_fit(self) -> "_ArcFit | None":
        """Pooled circular-arc e-corr-field→angle fit (super fit for cols 3/4)."""
        ec = self._cat(self.ecfield)
        ang = self._cat(self.angle)
        return _arc_fit(ec, ang) if ec is not None else None


def _format_residual_median_label(prefix: str, median_abs: float, unit: str) -> str:
    return f"{prefix}med|res| {_format_legend_number(median_abs, signed=False)} {unit}"


def _plot_residual_about(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    predicted: np.ndarray,
    *,
    color: str,
    session_id: str,
    n_sessions: int,
    unit: str,
    use_hexbin: bool = True,
) -> tuple[str, tuple] | None:
    """Scatter residual (y − super-fit prediction) vs x; label its median |res|."""
    residual = np.asarray(y, dtype=float) - np.asarray(predicted, dtype=float)
    x_f, res_f = _finite_pair(np.asarray(x, dtype=float), residual)
    if x_f.size == 0:
        return None
    _scatter_or_hexbin(ax, x, residual, color=color, use_hexbin=use_hexbin)
    prefix = trend_session_prefix(session_id, n_sessions=n_sessions)
    median_abs = float(np.median(np.abs(res_f)))
    return (
        _format_residual_median_label(prefix, median_abs, unit),
        trend_line_color(color),
    )


def run(session_ids: list[str], base_dir: str = "test_data", *, settings=None) -> None:
    """Plot amplifier command relationships with readback, field, and beam angle."""
    if not session_ids:
        print("No sessions selected")
        return

    bg_subtract = settings.bg_subtract if settings else False
    session_data: dict[str, AmplifierCorrelationSamples] = {}
    for sid in session_ids:
        samples = _load_session_samples(sid, base_dir, bg_subtract=bg_subtract)
        if samples is not None:
            session_data[sid] = samples

    if not session_data:
        print("No valid amplifier/field/trajectory timeslice data found for any session")
        return

    loaded_ids = list(session_data.keys())
    colors = DEFAULT_SESSION_COLORS[: len(loaded_ids)]

    fig, axes = view_grid(2, 4, sharex=False, sharey=False)
    row_specs = (
        ("X", "cmd_x", "readback_x", "field_x", "angle_x_mrad"),
        ("Y", "cmd_y", "readback_y", "field_y", "angle_y_mrad"),
    )
    ref_e = f"{_REFERENCE_ENERGY_MEV:g} MeV"
    col_titles = (
        "Readback Residual vs Cmd (super fit)",
        "Field Residual vs Drive (super fit)",
        "Beam Angle Residual vs E-corr Field (super fit)",
        "Beam Angle vs E-corr Field",
    )

    for row, (axis_label, cmd_k, rb_k, field_k, angle_k) in enumerate(row_specs):
        readback_trends: list[tuple[str, tuple]] = []
        field_trends: list[tuple[str, tuple]] = []
        angle_trends: list[tuple[str, tuple]] = []
        angle_raw_trends: list[tuple[str, tuple]] = []
        field_xlabels: list[str] = []

        # Build the super fit first; residuals are then measured against it.
        pooled = _PooledRow()
        for sid in loaded_ids:
            pooled.add(session_data[sid], cmd_k, rb_k, field_k, angle_k)
        rb_fit = pooled.readback_fit()
        field_super_fit = pooled.field_fit()
        angle_super_fit = pooled.angle_fit()

        combined_prefix = _COMBINED_PREFIX if len(loaded_ids) > 1 else ""
        if rb_fit is not None:
            readback_trends.append((
                _format_readback_fit_label(
                    rb_fit.slope, rb_fit.intercept, prefix=combined_prefix
                ),
                _COMBINED_COLOR,
            ))
        if field_super_fit is not None:
            field_trends.append((
                _format_field_fit_label(
                    field_super_fit.slope, field_super_fit.intercept,
                    prefix=combined_prefix,
                ),
                _COMBINED_COLOR,
            ))
        if angle_super_fit is not None:
            angle_label = _format_angle_fit_label(
                angle_super_fit, prefix=combined_prefix
            )
            angle_trends.append((angle_label, _COMBINED_COLOR))
            angle_raw_trends.append((angle_label, _COMBINED_COLOR))

        for sid, color in zip(loaded_ids, colors):
            samples = session_data[sid]
            cmd = getattr(samples, cmd_k)
            rb = getattr(samples, rb_k)
            field_g = getattr(samples, field_k)
            ecfield = _energy_corrected_field_kg(field_g, samples.momentum)
            angle = getattr(samples, angle_k)

            if rb_fit is not None:
                trend = _plot_residual_about(
                    axes[row, 0], cmd, rb, rb_fit.eval(cmd),
                    color=color, session_id=sid, n_sessions=len(loaded_ids),
                    unit="V",
                )
                if trend is not None:
                    readback_trends.append(trend)

            if field_super_fit is not None:
                use_rb = amplifier_readback_tracks_command_axis(cmd, rb)
                drive = rb if use_rb else cmd
                trend = _plot_residual_about(
                    axes[row, 1], drive, field_g, field_super_fit.eval(drive),
                    color=color, session_id=sid, n_sessions=len(loaded_ids),
                    unit="G",
                )
                if trend is not None:
                    field_trends.append(trend)
                    field_xlabels.append(
                        f"{'Readback' if use_rb else 'Cmd'} {axis_label} (V)"
                    )

            if angle_super_fit is not None:
                trend = _plot_residual_about(
                    axes[row, 2], ecfield, angle, angle_super_fit.predict(ecfield),
                    color=color, session_id=sid, n_sessions=len(loaded_ids),
                    unit="mrad",
                )
                if trend is not None:
                    angle_trends.append(trend)

            angle_raw_trend = _plot_ecfield_angle(
                axes[row, 3],
                samples,
                field_k,
                angle_k,
                color=color,
                session_id=sid,
                n_sessions=len(loaded_ids),
                align=samples.align_x if angle_k == "angle_x_mrad" else samples.align_y,
                draw_fit=False,
            )
            if angle_raw_trend is not None:
                angle_raw_trends.append(angle_raw_trend)

        # Draw the super-fit arc curve over the raw-angle panel (col 4).
        if angle_super_fit is not None:
            ec_all = pooled._cat(pooled.ecfield)
            ang_all = pooled._cat(pooled.angle)
            x_all, _ = _finite_pair(ec_all, ang_all)
            if x_all.size:
                xs = np.linspace(float(np.min(x_all)), float(np.max(x_all)), 200)
                axes[row, 3].plot(
                    xs, angle_super_fit.predict(xs),
                    color=_COMBINED_COLOR, linestyle="--", linewidth=1.6, zorder=7,
                )

        axes[row, 0].axhline(0, **REFLINE_KW)
        make_trend_legend(axes[row, 0], readback_trends)
        axes[row, 1].axhline(0, **REFLINE_KW)
        make_trend_legend(axes[row, 1], field_trends)
        axes[row, 2].axhline(0, **REFLINE_KW)
        make_trend_legend(axes[row, 2], angle_trends)
        make_trend_legend(axes[row, 3], angle_raw_trends)
        _annotate_angle_model(axes[row, 3])

        _style_axis(
            axes[row, 0],
            f"Cmd {axis_label} (V)",
            f"Readback {axis_label} − super fit (V)",
        )
        _style_axis(
            axes[row, 1],
            field_xlabels[0] if field_xlabels else f"Drive {axis_label} (V)",
            f"B{axis_label.lower()} − super fit (G)",
        )
        _style_axis(
            axes[row, 2],
            f"B{axis_label.lower()} energy-corr (kG @ {ref_e})",
            f"Beam {axis_label} angle − super fit (mrad)",
        )
        _style_axis(
            axes[row, 3],
            f"B{axis_label.lower()} energy-corr (kG @ {ref_e})",
            f"Beam {axis_label} angle (mrad)",
        )

    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title)

    finish_view(
        fig,
        "Amplifier Command Correlations (settled)",
        loaded_ids,
        colors,
        base_dir=base_dir,
    )
    plt.show()
