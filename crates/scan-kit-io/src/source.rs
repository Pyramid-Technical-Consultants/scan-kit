//! Shared Spot / Timeslice menu.
//!
//! Callers pass column names they already have. This module does not open a
//! file, remap strips, or scale a column. Those happen when a view loads the
//! one quantity the user picked.

use scan_kit_core::{
    concept_column_candidates, normalize_column_name, Choice, Control, POSITION_KEY_G2_RAW,
    POSITION_KEY_G3_RAW,
};
use serde_json::Value;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Shape {
    YAndX,
    Xy,
    Y,
    /// Spot / Timeslice only. Dose volume uses this.
    Source,
}

pub(crate) struct SessionCols<'a> {
    pub name: &'a str,
    pub columns: &'a [String],
}

pub(crate) struct Picked {
    pub grain: &'static str,
    pub y: String,
    pub frame: &'static str,
    pub x: &'static str,
    pub xy: &'static str,
    pub controls: Vec<Control>,
}

struct YQty {
    id: &'static str,
    label: &'static str,
    icon: &'static str,
    detail: &'static str,
    /// 0 hidden, 1 isocenter, 2 isocenter and chamber.
    spot: u8,
    slice: u8,
    concepts: &'static [&'static str],
    keys: &'static [&'static str],
    energy_only: bool,
    geometry: bool,
}

struct XQty {
    id: &'static str,
    label: &'static str,
    icon: &'static str,
    detail: &'static str,
    concepts: &'static [&'static str],
    keys: &'static [&'static str],
}

struct XyMode {
    id: &'static str,
    label: &'static str,
    icon: &'static str,
    detail: &'static str,
    slice_only: bool,
    concepts: &'static [&'static str],
    keys: &'static [&'static str],
    geometry: bool,
}

const Y_QTY: &[YQty] = &[
    YQty {
        id: "dose_ratio",
        label: "Dose Ratios",
        icon: "dose_ratio",
        detail: "Chambers over each other",
        spot: 1,
        slice: 0,
        concepts: &["ic1_total_dose", "ic2_total_dose"],
        keys: &["ic1_dose", "ic2_dose"],
        energy_only: false,
        geometry: false,
    },
    YQty {
        id: "dose_error",
        label: "Dose Error (%)",
        icon: "dose_error",
        detail: "Measured against target",
        spot: 1,
        slice: 0,
        concepts: &["ic1_total_dose", "ic2_total_dose"],
        keys: &["ic1_dose", "ic1_dose_err_pct"],
        energy_only: false,
        geometry: false,
    },
    YQty {
        id: "dose_per_mu",
        label: "Dose per MU",
        icon: "dose_per_mu",
        detail: "Delivered over requested MU",
        spot: 1,
        slice: 0,
        concepts: &["ic1_total_dose", "ic2_total_dose"],
        keys: &["ic1_dose", "ic1_dose_per_mu"],
        energy_only: false,
        geometry: false,
    },
    YQty {
        id: "position_error",
        label: "Position Error (mm)",
        icon: "position_error",
        detail: "Measured minus plan",
        spot: 2,
        slice: 1,
        concepts: &["ic1_x_pos_raw", "ic1_y_pos_raw", "ic1_x_pos", "ic2_x_pos"],
        keys: &["position_error_x", "ic1_x_err", "r_ic1_x_position"],
        energy_only: true,
        geometry: true,
    },
    YQty {
        id: "position_error_rel",
        label: "Relative Position Error (mm)",
        icon: "position_error_rel",
        detail: "Nozzle offset removed",
        spot: 2,
        slice: 1,
        concepts: &["ic1_x_pos_raw", "ic1_y_pos_raw", "ic1_x_pos", "ic2_x_pos"],
        keys: &["position_error_x", "ic1_x_err", "r_ic1_x_position"],
        energy_only: true,
        geometry: true,
    },
    YQty {
        id: "position_radius",
        label: "Position Radius (mm)",
        icon: "position_radius",
        detail: "Distance from the plan",
        spot: 2,
        slice: 1,
        concepts: &["ic1_x_pos_raw", "ic1_y_pos_raw", "ic1_x_pos", "ic2_x_pos"],
        keys: &["position_error_x", "ic1_x_err", "r_ic1_x_position"],
        energy_only: true,
        geometry: true,
    },
    YQty {
        id: "position_radius_rel",
        label: "Relative Position Radius (mm)",
        icon: "position_radius_rel",
        detail: "Distance after the nozzle offset",
        spot: 2,
        slice: 1,
        concepts: &["ic1_x_pos_raw", "ic1_y_pos_raw", "ic1_x_pos", "ic2_x_pos"],
        keys: &["position_error_x", "ic1_x_err", "r_ic1_x_position"],
        energy_only: true,
        geometry: true,
    },
    YQty {
        id: "sigma",
        label: "Sigma (mm)",
        icon: "sigma",
        detail: "Beam width",
        spot: 2,
        slice: 1,
        concepts: &[],
        keys: &["sigma", "ic1_sig_x", "spot_sigma", "ic1_sigma_x"],
        energy_only: true,
        geometry: true,
    },
    YQty {
        id: "sigma_error",
        label: "Sigma Error (mm)",
        icon: "sigma_error",
        detail: "Against expected",
        spot: 2,
        slice: 1,
        concepts: &[],
        keys: &["sigma_error", "ic1_sig_x"],
        energy_only: true,
        geometry: true,
    },
    YQty {
        id: "sigma_error_pct",
        label: "Sigma Error (%)",
        icon: "sigma_error_pct",
        detail: "Percent of expected width",
        spot: 2,
        slice: 1,
        concepts: &[],
        keys: &["sigma_error", "ic1_sig_x"],
        energy_only: true,
        geometry: true,
    },
    YQty {
        id: "ic_current",
        label: "IC Current (nA)",
        icon: "ic_current",
        detail: "Primary channel",
        spot: 0,
        slice: 1,
        concepts: &["ic1_current", "ic2_current", "ic3_current_a"],
        keys: &["ic1_current", "ic2_current", "ic3_current"],
        energy_only: false,
        geometry: false,
    },
    YQty {
        id: "current_ratio",
        label: "Current Ratios (%)",
        icon: "current_ratio",
        detail: "IC2 over IC1",
        spot: 0,
        slice: 1,
        concepts: &["ic1_current", "ic2_current"],
        keys: &["ic1_current", "ic2_current"],
        energy_only: false,
        geometry: false,
    },
    YQty {
        id: "dose_rate",
        label: "Dose Rate (MU/s)",
        icon: "dose_rate",
        detail: "Delivery rate",
        spot: 1,
        slice: 0,
        concepts: &["ic1_total_dose"],
        keys: &["ic1_dose"],
        energy_only: false,
        geometry: false,
    },
    YQty {
        id: "ic12_pos_diff",
        label: "IC2-IC1 Position (mm)",
        icon: "ic12_pos_diff",
        detail: "IC2 minus IC1",
        spot: 2,
        slice: 2,
        concepts: &["ic1_x_pos_raw", "ic2_x_pos_raw", "ic1_x_pos", "ic2_x_pos"],
        keys: &["ic12_x_diff", "r_ic1_x_position", "r_ic2_x_position"],
        energy_only: true,
        geometry: true,
    },
    YQty {
        id: "amplifier_error",
        label: "Amplifier Error (V)",
        icon: "amplifier_error",
        detail: "Command minus readback",
        spot: 0,
        slice: 1,
        concepts: &["amplifier_cmd_x", "amplifier_readback_x"],
        keys: &["amp_x", "amp_cmd_x"],
        energy_only: true,
        geometry: false,
    },
    YQty {
        id: "fit_confidence",
        label: "Fit Confidence",
        icon: "fit_confidence",
        detail: "Strip fit",
        spot: 0,
        slice: 1,
        concepts: &[],
        keys: &["ic1_x_confidence", "r_ic1_x_confidence", "ic1_y_confidence"],
        energy_only: true,
        geometry: false,
    },
    YQty {
        id: "peak_amplitude",
        label: "Peak Amplitude",
        icon: "peak_amplitude",
        detail: "Strip peak",
        spot: 0,
        slice: 1,
        concepts: &["ic1_x_peak_amplitude", "ic1_y_peak_amplitude"],
        keys: &["ic1_x_peak", "ic1_peak_amplitude_x"],
        energy_only: true,
        geometry: false,
    },
    YQty {
        id: "probe_field",
        label: "Probe Field (G)",
        icon: "probe_field",
        detail: "Hall probe",
        spot: 0,
        slice: 1,
        concepts: &["mag_field_x", "mag_field_y"],
        keys: &["field_x", "field_y"],
        energy_only: true,
        geometry: false,
    },
    YQty {
        id: "spot_time",
        label: "Spot Delivery Time (ms)",
        icon: "spot_time",
        detail: "Layer dwell",
        spot: 1,
        slice: 0,
        concepts: &["timestamp", "time_s", "time_ns"],
        keys: &["spot_time"],
        energy_only: false,
        geometry: false,
    },
];

const X_QTY: &[XQty] = &[
    XQty {
        id: "energy",
        label: "Energy (MeV)",
        icon: "energy",
        detail: "Layer energy",
        concepts: &["energy"],
        keys: &["energy"],
    },
    XQty {
        id: "target_mu",
        label: "Target MU",
        icon: "target_mu",
        detail: "Requested charge",
        concepts: &["charge_req"],
        keys: &["target_mu", "charge_req"],
    },
    XQty {
        id: "spot_time",
        label: "Spot time (ms)",
        icon: "spot_time",
        detail: "Delivery time",
        concepts: &["timestamp", "time_s", "time_ns"],
        keys: &["spot_time"],
    },
    XQty {
        id: "radius",
        label: "Radius (mm)",
        icon: "radius",
        detail: "Plan radius",
        concepts: &["x_position", "y_position"],
        keys: &["radius", "position_x", "plan_x"],
    },
];

const XY_MODE: &[XyMode] = &[
    XyMode {
        id: "position",
        label: "Position (mm)",
        icon: "position",
        detail: "Chamber or plan",
        slice_only: false,
        concepts: &["ic1_x_pos_raw", "ic1_x_pos", "x_position"],
        keys: &["position_x", "ic1_x", "plan_x"],
        geometry: true,
    },
    XyMode {
        id: "position_error",
        label: "Position Error (mm)",
        icon: "position_error",
        detail: "Against the plan",
        slice_only: false,
        concepts: &["ic1_x_pos_raw", "ic1_x_pos"],
        keys: &["position_error_x", "ic1_x_err"],
        geometry: true,
    },
    XyMode {
        id: "position_error_rel",
        label: "Relative Position Error (mm)",
        icon: "position_error_rel",
        detail: "Nozzle offset removed",
        slice_only: false,
        concepts: &["ic1_x_pos_raw", "ic1_x_pos"],
        keys: &["position_error_x", "ic1_x_err"],
        geometry: true,
    },
    XyMode {
        id: "sigma",
        label: "Sigma (mm)",
        icon: "sigma",
        detail: "Beam width",
        slice_only: false,
        concepts: &[],
        keys: &["sigma", "ic1_sig_x", "spot_sigma"],
        geometry: true,
    },
    XyMode {
        id: "amplifier_voltage",
        label: "Amplifier (V)",
        icon: "amplifier",
        detail: "Command and readback",
        slice_only: true,
        concepts: &["amplifier_cmd_x", "amplifier_readback_x"],
        keys: &["amp_cmd_x", "amp_read_x", "c_x", "r_xV"],
        geometry: false,
    },
    XyMode {
        id: "amplifier",
        label: "Amplifier Error (V)",
        icon: "amplifier_error",
        detail: "Readback minus command",
        slice_only: true,
        concepts: &["amplifier_cmd_x", "amplifier_readback_x"],
        keys: &["amp_x", "c_x", "r_xV"],
        geometry: false,
    },
    XyMode {
        id: "probe",
        label: "Probe (G)",
        icon: "probe",
        detail: "Hall probe",
        slice_only: true,
        concepts: &["mag_field_x", "mag_field_y"],
        keys: &["field_x", "field_y"],
        geometry: false,
    },
    XyMode {
        id: "confidence",
        label: "Confidence",
        icon: "confidence",
        detail: "Strip fit",
        slice_only: true,
        concepts: &[],
        keys: &["ic1_x_confidence", "r_ic1_x_confidence"],
        geometry: false,
    },
    XyMode {
        id: "coverage",
        label: "Coverage (%)",
        icon: "coverage",
        detail: "Confident samples",
        slice_only: true,
        concepts: &[],
        keys: &["ic1_x_confidence", "r_ic1_x_confidence"],
        geometry: false,
    },
];

pub(crate) fn wants_timeslice(shape: Shape, options: &Value) -> bool {
    grain_of(shape, true, true, options) == "timeslice"
}

pub(crate) fn select(
    shape: Shape,
    allow_spot: bool,
    allow_slice: bool,
    sessions: &[SessionCols<'_>],
    options: &Value,
) -> Picked {
    let grain = grain_of(shape, allow_spot, allow_slice, options);
    let sessions = prepare(sessions);
    let geometry = geometry_note(&sessions);
    let mut controls = Vec::new();
    let mut y = String::new();
    let mut frame = "iso";
    let mut x = "energy";
    let mut xy = "position";

    let show_source = allow_spot
        && allow_slice
        && shape != Shape::Y
        && !(shape == Shape::Xy && xy_slice_only(options));
    if show_source {
        let choices = vec![
            Choice::full("spot", "Spot", "One row per spot", "spot"),
            Choice::full(
                "timeslice",
                "Timeslice",
                "One row per millisecond",
                "timeslice",
            ),
        ];
        let selected = select_label(&choices, option_str(options, &["source", "grain"]))
            .unwrap_or(if grain == "timeslice" {
                "Timeslice"
            } else {
                "Spot"
            })
            .to_string();
        controls.push(menu("source", "Source", choices, &selected));
    }

    match shape {
        Shape::YAndX => {
            let rows = y_rows(grain, &sessions);
            let choices = y_menu(&rows, &geometry);
            let frame_hint = option_str(options, &["source", "grain"]).and_then(frame_from_text);
            let picked = pick_menu(&choices, option_str(options, &["y", "metric"]), frame_hint)
                .map(|choice| {
                    (
                        metric_of(&choice.id).to_string(),
                        frame_of_id(&choice.id),
                        choice.label.clone(),
                    )
                });
            if let Some((metric, picked_frame, label)) = picked {
                y = metric;
                frame = picked_frame;
                let energy_only = grain == "timeslice" && energy_only_metric(&y);
                controls.push(menu("y", "Y", choices, &label));
                let x_choices = x_menu(energy_only, &sessions);
                let picked_x = pick_menu(&x_choices, option_str(options, &["x"]), None)
                    .map(|choice| (x_id_of(choice.id.as_str()), choice.label.clone()));
                if let Some((id, x_label)) = picked_x {
                    x = id;
                    controls.push(menu("x", "X", x_choices, &x_label));
                }
            }
        }
        Shape::Xy => {
            let choices = xy_menu(grain, &sessions, &geometry);
            let picked = pick_menu(&choices, option_str(options, &["xy", "mode"]), None)
                .map(|choice| (xy_id_of(choice.id.as_str()), choice.label.clone()));
            if let Some((id, label)) = picked {
                xy = id;
                controls.push(menu("xy", "XY", choices, &label));
            }
        }
        Shape::Y => {
            let choices = channel_menu(&sessions);
            let raw = option_str(options, &["y", "channel"]);
            let picked = if raw.is_none() {
                choices
                    .iter()
                    .find(|choice| choice.id == "ic1_current")
                    .or_else(|| choices.first())
            } else {
                pick_menu(&choices, raw, None)
            }
            .map(|choice| (choice.id.clone(), choice.label.clone()));
            if let Some((id, label)) = picked {
                y = id;
                controls.push(menu("y", "Y", choices, &label));
            }
        }
        Shape::Source => {}
    }

    if y.is_empty() {
        y = if grain == "timeslice" {
            "current_ratio".to_string()
        } else {
            "dose_error".to_string()
        };
    }
    Picked {
        grain,
        y,
        frame,
        x,
        xy,
        controls,
    }
}

/// Display name for a timeslice channel, with the unit when one is known.
pub(crate) fn channel_text(id: &str) -> String {
    let name = channel_name(id);
    match channel_unit(id) {
        Some(unit) if !name.contains('(') => format!("{name} ({unit})"),
        _ => name,
    }
}

fn menu(id: &str, label: &str, options: Vec<Choice>, value: &str) -> Control {
    Control {
        id: id.to_string(),
        label: label.to_string(),
        options,
        value: value.to_string(),
        group: "Data Source".to_string(),
        kind: String::new(),
    }
}

fn grain_of(shape: Shape, allow_spot: bool, allow_slice: bool, options: &Value) -> &'static str {
    if shape == Shape::Y || !allow_spot {
        return "timeslice";
    }
    if !allow_slice {
        return "spot";
    }
    if shape == Shape::Xy && xy_slice_only(options) {
        return "timeslice";
    }
    if let Some(grain) = requested_grain(options) {
        return grain;
    }
    if shape == Shape::YAndX {
        if let Some(raw) = option_str(options, &["y", "metric"]) {
            if let Some(qty) = find_y(raw) {
                if qty.spot == 0 {
                    return "timeslice";
                }
            }
        }
    }
    "spot"
}

fn requested_grain(options: &Value) -> Option<&'static str> {
    let raw = option_str(options, &["source", "grain"])?;
    let lower = raw.to_ascii_lowercase();
    if lower == "spot" || lower.starts_with("spot ") || lower == "iso" || lower == "chamber" {
        Some("spot")
    } else if lower.contains("time") {
        Some("timeslice")
    } else {
        None
    }
}

fn xy_slice_only(options: &Value) -> bool {
    option_str(options, &["xy", "mode"])
        .and_then(find_xy)
        .is_some_and(|mode| mode.slice_only)
}

fn find_y(raw: &str) -> Option<&'static YQty> {
    Y_QTY.iter().find(|qty| qty_matches(qty, raw))
}

fn find_xy(raw: &str) -> Option<&'static XyMode> {
    XY_MODE
        .iter()
        .find(|mode| mode.id == raw || mode.label == raw || without_unit(mode.label) == raw)
}

fn qty_matches(qty: &YQty, raw: &str) -> bool {
    qty.id == raw
        || qty.label == raw
        || without_unit(qty.label) == raw
        || strip_frame(raw) == qty.label
        || strip_frame(raw) == without_unit(qty.label)
}

fn option_str<'a>(options: &'a Value, keys: &[&str]) -> Option<&'a str> {
    keys.iter()
        .find_map(|key| options.get(*key).and_then(Value::as_str))
}

fn frames(qty: &YQty, grain: &str) -> Vec<&'static str> {
    let code = if grain == "timeslice" {
        qty.slice
    } else {
        qty.spot
    };
    match code {
        2 => vec!["iso", "chamber"],
        1 => vec!["iso"],
        _ => Vec::new(),
    }
}

struct YRow {
    index: usize,
    frame: &'static str,
    gap: Gap,
}

#[derive(Clone, Default)]
struct Gap {
    count: usize,
    names: Vec<String>,
}

struct Prepared<'a> {
    name: &'a str,
    raw: &'a [String],
    /// Each header normalized once, so later concept checks do not scan and
    /// allocate again.
    norm: Vec<String>,
}

fn prepare<'a>(sessions: &'a [SessionCols<'a>]) -> Vec<Prepared<'a>> {
    sessions
        .iter()
        .map(|session| Prepared {
            name: session.name,
            raw: session.columns,
            norm: session
                .columns
                .iter()
                .map(|name| normalize_column_name(name))
                .collect(),
        })
        .collect()
}

fn y_rows(grain: &str, sessions: &[Prepared<'_>]) -> Vec<YRow> {
    let filtering = sessions.iter().any(|session| !session.raw.is_empty());
    let mut all = Vec::new();
    let mut kept = Vec::new();
    for (index, qty) in Y_QTY.iter().enumerate() {
        let frames = frames(qty, grain);
        if frames.is_empty() {
            continue;
        }
        for frame in frames.iter().copied() {
            all.push(YRow {
                index,
                frame,
                gap: Gap::default(),
            });
        }
        if !filtering {
            continue;
        }
        let (hit, gap) = column_hits(qty.id, qty.concepts, qty.keys, sessions);
        if hit {
            for frame in frames.iter().copied() {
                kept.push(YRow {
                    index,
                    frame,
                    gap: gap.clone(),
                });
            }
        }
    }
    // ponytail: an unrecognized header keeps the static list instead of a blank menu.
    if !filtering || kept.is_empty() {
        all
    } else {
        kept
    }
}

fn y_menu(rows: &[YRow], geometry: &str) -> Vec<Choice> {
    rows.iter()
        .map(|row| y_choice(row, rows, geometry))
        .collect()
}

fn y_choice(row: &YRow, rows: &[YRow], geometry: &str) -> Choice {
    let qty = &Y_QTY[row.index];
    let qualify = rows.iter().filter(|other| other.index == row.index).count() > 1;
    let label = if qualify {
        format!(
            "{} ({})",
            qty.label,
            if row.frame == "chamber" {
                "Chamber"
            } else {
                "Isocenter"
            }
        )
    } else {
        qty.label.to_string()
    };
    let id = if qualify {
        format!("{}:{}", qty.id, row.frame)
    } else {
        qty.id.to_string()
    };
    Choice::full(&id, &label, y_detail(qty, geometry, &row.gap), qty.icon)
}

fn y_detail(qty: &YQty, geometry: &str, gap: &Gap) -> String {
    let mut parts = vec![qty.detail.to_string()];
    if qty.geometry && !geometry.is_empty() {
        parts.push(geometry.to_string());
    }
    if let Some(note) = missing_note(gap) {
        parts.push(note);
    }
    parts.join(", ")
}

fn x_menu(energy_only: bool, sessions: &[Prepared<'_>]) -> Vec<Choice> {
    let filtering = sessions.iter().any(|session| !session.raw.is_empty());
    let mut choices = Vec::new();
    for qty in X_QTY {
        if energy_only && qty.id != "energy" {
            continue;
        }
        if filtering {
            let (hit, _) = column_hits(qty.id, qty.concepts, qty.keys, sessions);
            if !hit && qty.id != "energy" {
                continue;
            }
        }
        let detail = if energy_only {
            "Only axis for this signal".to_string()
        } else {
            qty.detail.to_string()
        };
        choices.push(Choice::full(qty.id, qty.label, detail, qty.icon));
    }
    if choices.is_empty() {
        choices.push(Choice::full(
            "energy",
            "Energy (MeV)",
            "Layer energy",
            "energy",
        ));
    }
    choices
}

fn xy_menu(grain: &str, sessions: &[Prepared<'_>], geometry: &str) -> Vec<Choice> {
    let filtering = sessions.iter().any(|session| !session.raw.is_empty());
    let none = Gap::default();
    if !filtering {
        return XY_MODE
            .iter()
            .map(|mode| xy_choice(mode, geometry, &none))
            .collect();
    }
    let mut kept = Vec::new();
    for mode in XY_MODE {
        if mode.slice_only && grain != "timeslice" {
            kept.push(xy_choice(mode, geometry, &none));
            continue;
        }
        let (hit, gap) = column_hits(mode.id, mode.concepts, mode.keys, sessions);
        if hit {
            kept.push(xy_choice(mode, geometry, &gap));
        }
    }
    if kept.is_empty() {
        XY_MODE
            .iter()
            .map(|mode| xy_choice(mode, geometry, &none))
            .collect()
    } else {
        kept
    }
}

fn xy_choice(mode: &XyMode, geometry: &str, gap: &Gap) -> Choice {
    let mut parts = vec![mode.detail.to_string()];
    if mode.geometry && !geometry.is_empty() {
        parts.push(geometry.to_string());
    }
    if let Some(note) = missing_note(gap) {
        parts.push(note);
    }
    Choice::full(mode.id, mode.label, parts.join(", "), mode.icon)
}

fn channel_menu(sessions: &[Prepared<'_>]) -> Vec<Choice> {
    let mut ids: Vec<String> = Vec::new();
    for session in sessions {
        for name in session.raw {
            if channel_key(name) && !ids.iter().any(|have| have == name) {
                ids.push(name.clone());
            }
        }
    }
    ids.sort();
    if ids.is_empty() {
        ids.push("ic1_current".to_string());
    }
    let geometry = geometry_note(sessions);
    ids.into_iter()
        .map(|id| {
            let (_, gap) = column_hits("", &[], &[id.as_str()], sessions);
            let mut detail = String::new();
            if channel_geometry(&id) && !geometry.is_empty() {
                detail = geometry.clone();
            }
            if let Some(note) = missing_note(&gap) {
                if !detail.is_empty() {
                    detail.push_str(", ");
                }
                detail.push_str(&note);
            }
            Choice::full(&id, &channel_text(&id), detail, channel_icon(&id))
        })
        .collect()
}

fn column_hits(
    id: &str,
    concepts: &[&str],
    keys: &[&str],
    sessions: &[Prepared<'_>],
) -> (bool, Gap) {
    let needles = name_needles(id);
    if concepts.is_empty() && keys.is_empty() && needles.is_empty() {
        return (true, Gap::default());
    }
    let mut hit = false;
    let mut gap = Gap::default();
    for session in sessions {
        if concept_list_hit(concepts, keys, session) || needle_hit(needles, session) {
            hit = true;
        } else if !session.name.is_empty() {
            gap.count += 1;
            if gap.names.len() < 2 {
                gap.names.push(session.name.to_string());
            }
        }
    }
    (hit, gap)
}

fn name_needles(id: &str) -> &'static [&'static str] {
    match id {
        // Measured columns use several spellings. Plan `position_x` is not one of them.
        "sigma" | "sigma_error" | "sigma_error_pct" => &["sigma"],
        "position_error" | "position_error_rel" | "position_radius" | "position_radius_rel" => {
            &["spot_position", "spot_raw", "_x_position", "_y_position"]
        }
        _ => &[],
    }
}

fn needle_hit(needles: &[&str], session: &Prepared<'_>) -> bool {
    session
        .norm
        .iter()
        .any(|name| needles.iter().any(|needle| name.contains(needle)))
}

fn concept_list_hit(concepts: &[&str], keys: &[&str], session: &Prepared<'_>) -> bool {
    concepts.iter().any(|concept| concept_hit(session, concept))
        || keys.iter().any(|key| has_name(session, key))
}

fn concept_hit(session: &Prepared<'_>, concept: &str) -> bool {
    if has_name(session, concept) {
        return true;
    }
    let aliases = concept_column_candidates(concept, None);
    if !aliases.is_empty() {
        return aliases.iter().any(|name| has_name(session, name));
    }
    [POSITION_KEY_G2_RAW, POSITION_KEY_G3_RAW]
        .into_iter()
        .any(|key| {
            concept_column_candidates(concept, Some(key))
                .iter()
                .any(|name| has_name(session, name))
        })
}

fn has_name(session: &Prepared<'_>, requested: &str) -> bool {
    session.raw.iter().any(|name| name == requested)
        || session
            .norm
            .iter()
            .any(|name| name == &normalize_column_name(requested))
}

fn geometry_note(sessions: &[Prepared<'_>]) -> String {
    let mut g2 = false;
    let mut g3 = false;
    for session in sessions {
        match generation(&session.norm) {
            Some("G2") => g2 = true,
            Some("G3") => g3 = true,
            Some(_) => {
                g2 = true;
                g3 = true;
            }
            None => {}
        }
    }
    match (g2, g3) {
        (true, true) => "mixed G2/G3".to_string(),
        (true, false) => "G2".to_string(),
        (false, true) => "G3".to_string(),
        _ => String::new(),
    }
}

fn generation(columns: &[String]) -> Option<&'static str> {
    let mut g2 = false;
    let mut g3 = false;
    for name in columns {
        let lower = name.to_ascii_lowercase();
        if lower.contains("spot_position") {
            g3 = true;
        } else if lower.contains("spot_raw") {
            g2 = true;
        }
    }
    match (g2, g3) {
        (true, true) => Some("mixed G2/G3"),
        (true, false) => Some("G2"),
        (false, true) => Some("G3"),
        _ => None,
    }
}

fn missing_note(gap: &Gap) -> Option<String> {
    if gap.count == 0 {
        return None;
    }
    let head = gap.names.join(", ");
    if gap.count > gap.names.len() {
        Some(format!(
            "missing in {head} +{}",
            gap.count - gap.names.len()
        ))
    } else {
        Some(format!("missing in {head}"))
    }
}

fn pick_menu<'a>(
    choices: &'a [Choice],
    raw: Option<&str>,
    frame: Option<&str>,
) -> Option<&'a Choice> {
    let Some(raw) = raw.filter(|value| !value.is_empty()) else {
        return frame
            .and_then(|frame| choices.iter().find(|choice| frame_matches(choice, frame)))
            .or_else(|| choices.first());
    };
    if let Some(hit) = choices
        .iter()
        .find(|choice| choice.id == raw || choice.label == raw)
    {
        return Some(hit);
    }
    let bare = strip_frame(raw);
    let frame = frame_from_text(raw).or(frame);
    let hits: Vec<&Choice> = choices
        .iter()
        .filter(|choice| {
            metric_of(&choice.id) == raw
                || metric_of(&choice.id) == bare
                || without_unit(&choice.label) == raw
                || without_unit(&choice.label) == bare
                || strip_frame(&choice.label) == raw
                || strip_frame(&choice.label) == bare
        })
        .collect();
    frame
        .and_then(|frame| {
            hits.iter()
                .copied()
                .find(|choice| frame_matches(choice, frame))
        })
        .or_else(|| hits.first().copied())
        .or_else(|| choices.first())
}

fn frame_matches(choice: &Choice, frame: &str) -> bool {
    let (suffix, marker) = if frame == "chamber" {
        (":chamber", "(Chamber)")
    } else {
        (":iso", "(Isocenter)")
    };
    choice.id.ends_with(suffix) || choice.label.contains(marker)
}

fn frame_from_text(raw: &str) -> Option<&'static str> {
    let lower = raw.to_ascii_lowercase();
    if lower.ends_with("chamber") || lower.ends_with("(chamber)") {
        Some("chamber")
    } else if lower.ends_with("_iso")
        || lower.ends_with("isocenter")
        || lower.ends_with("(isocenter)")
    {
        Some("iso")
    } else {
        None
    }
}

fn select_label<'a>(choices: &'a [Choice], raw: Option<&str>) -> Option<&'a str> {
    let raw = raw.filter(|value| !value.is_empty())?;
    if let Some(hit) = choices
        .iter()
        .find(|choice| choice.id == raw || choice.label == raw)
    {
        return Some(hit.label.as_str());
    }
    let lower = raw.to_ascii_lowercase();
    let id = if lower.contains("time") {
        "timeslice"
    } else if lower == "spot" || lower == "iso" || lower == "chamber" || lower.starts_with("spot") {
        "spot"
    } else {
        return pick_menu(choices, Some(raw), None).map(|choice| choice.label.as_str());
    };
    choices
        .iter()
        .find(|choice| choice.id == id)
        .map(|choice| choice.label.as_str())
}

fn metric_of(id: &str) -> &str {
    id.split(':').next().unwrap_or(id)
}

fn frame_of_id(id: &str) -> &'static str {
    if id.ends_with(":chamber") {
        "chamber"
    } else {
        "iso"
    }
}

fn energy_only_metric(metric: &str) -> bool {
    Y_QTY
        .iter()
        .find(|qty| qty.id == metric)
        .is_some_and(|qty| qty.energy_only)
}

fn x_id_of(id: &str) -> &'static str {
    X_QTY
        .iter()
        .find(|qty| qty.id == id)
        .map(|qty| qty.id)
        .unwrap_or("energy")
}

fn xy_id_of(id: &str) -> &'static str {
    XY_MODE
        .iter()
        .find(|mode| mode.id == id)
        .map(|mode| mode.id)
        .unwrap_or("position")
}

fn strip_frame(label: &str) -> &str {
    label
        .strip_suffix(" (Chamber)")
        .or_else(|| label.strip_suffix(" (Isocenter)"))
        .unwrap_or(label)
}

fn without_unit(label: &str) -> &str {
    let bare = strip_frame(label);
    let Some(start) = bare.rfind(" (") else {
        return bare;
    };
    if bare.ends_with(')') {
        &bare[..start]
    } else {
        bare
    }
}

fn channel_key(name: &str) -> bool {
    !matches!(name, "energy" | "beam_on" | "beam_on_time" | "spot_time")
}

fn channel_geometry(id: &str) -> bool {
    length_mm(&id.to_ascii_lowercase())
}

fn length_mm(lower: &str) -> bool {
    matches!(lower, "ic1_x" | "ic1_y" | "ic2_x" | "ic2_y")
        || lower.contains("sig")
        || lower.contains("err")
        || lower.contains("pos")
        || lower.contains("diff")
}

fn command_volt(lower: &str) -> bool {
    lower.contains("amp") || matches!(lower, "c_x" | "c_y" | "r_xv" | "r_yv")
}

fn channel_unit(id: &str) -> Option<&'static str> {
    let lower = id.to_ascii_lowercase();
    if lower.contains("confidence") || lower.contains("peak") {
        None
    } else if lower.contains("current") {
        Some("nA")
    } else if lower.contains("field") {
        Some("G")
    } else if command_volt(&lower) {
        Some("V")
    } else if length_mm(&lower) {
        Some("mm")
    } else {
        None
    }
}

fn channel_icon(id: &str) -> &'static str {
    let lower = id.to_ascii_lowercase();
    if lower.contains("current") {
        "ic_current"
    } else if lower.contains("confidence") {
        "fit_confidence"
    } else if lower.contains("peak") {
        "peak_amplitude"
    } else if lower.contains("field") {
        "probe_field"
    } else if command_volt(&lower) {
        "amplifier"
    } else if lower.contains("sig") {
        "sigma"
    } else if lower.contains("err") || lower.contains("diff") || lower.contains("pos") {
        "position_error"
    } else if lower.ends_with("_y") {
        "move_vertical"
    } else {
        "move_horizontal"
    }
}

fn channel_name(id: &str) -> String {
    match id {
        "ic1_current" => "IC1 Current",
        "ic2_current" => "IC2 Current",
        "ic3_current" => "IC3 Current",
        "ic1_x" => "IC1 X",
        "ic1_y" => "IC1 Y",
        "ic2_x" => "IC2 X",
        "ic2_y" => "IC2 Y",
        "ic1_x_err" => "IC1 X Error",
        "ic1_y_err" => "IC1 Y Error",
        "ic2_x_err" => "IC2 X Error",
        "ic2_y_err" => "IC2 Y Error",
        "ic1_sig_x" => "IC1 Sigma X",
        "ic1_sig_y" => "IC1 Sigma Y",
        "ic2_sig_x" => "IC2 Sigma X",
        "ic2_sig_y" => "IC2 Sigma Y",
        "ic12_x_diff" => "IC2-IC1 X",
        "ic12_y_diff" => "IC2-IC1 Y",
        "field_x" => "Field X",
        "field_y" => "Field Y",
        "ic1_x_confidence" => "IC1 X Confidence",
        "ic1_y_confidence" => "IC1 Y Confidence",
        "ic2_x_confidence" => "IC2 X Confidence",
        "ic2_y_confidence" => "IC2 Y Confidence",
        "ic1_x_peak" => "IC1 X Peak",
        "ic1_y_peak" => "IC1 Y Peak",
        "ic2_x_peak" => "IC2 X Peak",
        "ic2_y_peak" => "IC2 Y Peak",
        "amp_x" => "Amplifier X",
        "amp_y" => "Amplifier Y",
        "amp_cmd_x" => "Amplifier Command X",
        "amp_read_x" => "Amplifier Readback X",
        "amp_cmd_y" => "Amplifier Command Y",
        "amp_read_y" => "Amplifier Readback Y",
        _ => {
            return id
                .split('_')
                .map(|part| {
                    let mut chars = part.chars();
                    match chars.next() {
                        None => String::new(),
                        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
                    }
                })
                .collect::<Vec<_>>()
                .join(" ")
        }
    }
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn cols<'a>(name: &'a str, columns: &'a [String]) -> SessionCols<'a> {
        SessionCols { name, columns }
    }

    fn names(columns: &[&str]) -> Vec<String> {
        columns.iter().map(|name| (*name).to_string()).collect()
    }

    #[test]
    fn spot_only_and_timeslice_only_omit_source() {
        let spot = select(Shape::YAndX, true, false, &[], &json!({}));
        assert!(spot.controls.iter().all(|control| control.id != "source"));
        assert_eq!(spot.grain, "spot");
        let slice = select(Shape::YAndX, false, true, &[], &json!({}));
        assert!(slice.controls.iter().all(|control| control.id != "source"));
        assert_eq!(slice.grain, "timeslice");
    }

    #[test]
    fn both_grains_keep_disjoint_y_lists_and_snap() {
        let spot = select(Shape::YAndX, true, true, &[], &json!({}));
        let source = spot
            .controls
            .iter()
            .find(|control| control.id == "source")
            .unwrap();
        assert_eq!(source.labels(), vec!["Spot", "Timeslice"]);
        let y = spot
            .controls
            .iter()
            .find(|control| control.id == "y")
            .unwrap();
        assert_eq!(
            y.labels(),
            vec![
                "Dose Ratios",
                "Dose Error (%)",
                "Dose per MU",
                "Position Error (mm) (Isocenter)",
                "Position Error (mm) (Chamber)",
                "Relative Position Error (mm) (Isocenter)",
                "Relative Position Error (mm) (Chamber)",
                "Position Radius (mm) (Isocenter)",
                "Position Radius (mm) (Chamber)",
                "Relative Position Radius (mm) (Isocenter)",
                "Relative Position Radius (mm) (Chamber)",
                "Sigma (mm) (Isocenter)",
                "Sigma (mm) (Chamber)",
                "Sigma Error (mm) (Isocenter)",
                "Sigma Error (mm) (Chamber)",
                "Sigma Error (%) (Isocenter)",
                "Sigma Error (%) (Chamber)",
                "Dose Rate (MU/s)",
                "IC2-IC1 Position (mm) (Isocenter)",
                "IC2-IC1 Position (mm) (Chamber)",
                "Spot Delivery Time (ms)",
            ]
        );
        assert!(y
            .options
            .iter()
            .any(|option| option == "Position Error (mm) (Isocenter)"));
        assert!(y.options.iter().all(|option| option != "IC Current (nA)"));
        let switched = select(
            Shape::YAndX,
            true,
            true,
            &[],
            &json!({"y": "Dose Error (%)", "source": "Timeslice"}),
        );
        let y = switched
            .controls
            .iter()
            .find(|control| control.id == "y")
            .unwrap();
        assert_eq!(y.value, "Position Error (mm)");
        assert_eq!(
            y.labels(),
            vec![
                "Position Error (mm)",
                "Relative Position Error (mm)",
                "Position Radius (mm)",
                "Relative Position Radius (mm)",
                "Sigma (mm)",
                "Sigma Error (mm)",
                "Sigma Error (%)",
                "IC Current (nA)",
                "Current Ratios (%)",
                "IC2-IC1 Position (mm) (Isocenter)",
                "IC2-IC1 Position (mm) (Chamber)",
                "Amplifier Error (V)",
                "Fit Confidence",
                "Peak Amplitude",
                "Probe Field (G)",
            ]
        );
        assert!(y.options.iter().all(|option| option != "Dose Error (%)"));
        let fit = select(
            Shape::YAndX,
            true,
            true,
            &[],
            &json!({"metric": "Fit Confidence", "source": "Timeslice", "x": "Target MU"}),
        );
        let x = fit
            .controls
            .iter()
            .find(|control| control.id == "x")
            .unwrap();
        assert_eq!(x.labels(), vec!["Energy (MeV)"]);
        assert_eq!(x.value, "Energy (MeV)");
        assert_eq!(fit.x, "energy");
    }

    #[test]
    fn timeslice_only_xy_hides_source() {
        let picked = select(Shape::Xy, true, true, &[], &json!({"mode": "amplifier"}));
        assert!(picked.controls.iter().all(|control| control.id != "source"));
        assert_eq!(picked.grain, "timeslice");
        assert_eq!(picked.xy, "amplifier");
        let xy = picked
            .controls
            .iter()
            .find(|control| control.id == "xy")
            .unwrap();
        assert_eq!(xy.value, "Amplifier Error (V)");
        assert_eq!(
            xy.labels(),
            vec![
                "Position (mm)",
                "Position Error (mm)",
                "Relative Position Error (mm)",
                "Sigma (mm)",
                "Amplifier (V)",
                "Amplifier Error (V)",
                "Probe (G)",
                "Confidence",
                "Coverage (%)",
            ]
        );
    }

    #[test]
    fn y_only_lists_finite_channels_with_units() {
        let columns = names(&["ic1_current", "energy", "field_x"]);
        let sessions = [cols("sess", &columns)];
        let picked = select(Shape::Y, false, true, &sessions, &json!({}));
        assert!(picked.controls.iter().all(|control| control.id != "source"));
        let y = picked
            .controls
            .iter()
            .find(|control| control.id == "y")
            .unwrap();
        assert_eq!(picked.y, "ic1_current");
        assert!(y
            .options
            .iter()
            .any(|option| option.label == "IC1 Current (nA)"));
        assert!(y.options.iter().any(|option| option.label == "Field X (G)"));
        assert!(y.options.iter().all(|option| option != "Energy"));
        assert!(y.options.iter().all(|option| !option.icon.is_empty()));
    }

    #[test]
    fn g2_and_g3_share_position_and_a_mixed_note() {
        let g2 = names(&["r_ic1_x_spot_raw", "r_ic1_y_spot_raw"]);
        let g3 = names(&["r_ic1_x_spot_position_raw", "r_ic1_y_spot_position_raw"]);
        let one = [cols("g2", &g2)];
        let spot = select(
            Shape::YAndX,
            true,
            true,
            &one,
            &json!({"y": "position_error"}),
        );
        let y = spot
            .controls
            .iter()
            .find(|control| control.id == "y")
            .unwrap();
        let position = y
            .options
            .iter()
            .find(|option| option.label.contains("Position Error") && option.label.contains("(mm)"))
            .unwrap();
        assert!(position.detail.contains("G2"));
        let mixed = [cols("a", &g2), cols("b", &g3)];
        let both = select(
            Shape::YAndX,
            true,
            true,
            &mixed,
            &json!({"metric": "position_error"}),
        );
        let y = both
            .controls
            .iter()
            .find(|control| control.id == "y")
            .unwrap();
        let position = y
            .options
            .iter()
            .find(|option| metric_of(&option.id) == "position_error")
            .unwrap();
        assert!(
            position.detail.contains("mixed G2/G3"),
            "{}",
            position.detail
        );
    }

    #[test]
    fn a_session_missing_the_column_stays_named() {
        let full = names(&[
            "ic1_total_dose_spot",
            "r_ic1_x_spot_raw",
            "r_ic1_y_spot_raw",
        ]);
        let bare = names(&["ic1_total_dose_spot"]);
        let sessions = [cols("full", &full), cols("bare", &bare)];
        let picked = select(Shape::YAndX, true, true, &sessions, &json!({}));
        let y = picked
            .controls
            .iter()
            .find(|control| control.id == "y")
            .unwrap();
        let position = y
            .options
            .iter()
            .find(|option| metric_of(&option.id) == "position_error")
            .unwrap();
        assert!(
            position.detail.contains("missing in bare"),
            "{}",
            position.detail
        );
        assert!(y
            .options
            .iter()
            .any(|option| option.label.contains("Dose Error (%)")));
        assert!(y
            .options
            .iter()
            .all(|option| metric_of(&option.id) != "sigma"));
    }

    #[test]
    fn sigma_follows_a_sigma_column_and_not_position() {
        let position = names(&["r_ic1_x_spot_position", "r_ic1_y_spot_position"]);
        let sessions = [cols("sess", &position)];
        let picked = select(Shape::YAndX, true, true, &sessions, &json!({}));
        let y = picked
            .controls
            .iter()
            .find(|control| control.id == "y")
            .unwrap();
        assert!(y
            .options
            .iter()
            .any(|option| metric_of(&option.id) == "position_error"));
        assert!(y
            .options
            .iter()
            .all(|option| metric_of(&option.id) != "sigma"));
        let sigma = names(&["r_ic1_x_spot_sigma"]);
        let sessions = [cols("sess", &sigma)];
        let picked = select(
            Shape::YAndX,
            true,
            true,
            &sessions,
            &json!({"metric": "Sigma (mm)"}),
        );
        assert_eq!(picked.y, "sigma");
        let y = picked
            .controls
            .iter()
            .find(|control| control.id == "y")
            .unwrap();
        assert!(y
            .options
            .iter()
            .any(|option| option.label.contains("Sigma (mm)")));
        assert!(y
            .options
            .iter()
            .any(|option| metric_of(&option.id) == "sigma_error"));
    }

    #[test]
    fn command_voltage_is_not_labelled_as_millimetres() {
        assert_eq!(channel_text("c_x"), "C X (V)");
        assert_eq!(channel_text("ic1_x"), "IC1 X (mm)");
        assert_eq!(channel_text("field_x"), "Field X (G)");
        assert_eq!(channel_icon("c_x"), "amplifier");
    }

    #[test]
    fn chamber_request_keeps_that_frame() {
        let chamber = select(
            Shape::YAndX,
            true,
            true,
            &[],
            &json!({"metric": "Position Error (mm) (Chamber)", "source": "Spot"}),
        );
        assert_eq!(chamber.y, "position_error");
        assert_eq!(chamber.frame, "chamber");
        assert_eq!(chamber.grain, "spot");
        let legacy = select(
            Shape::YAndX,
            true,
            true,
            &[],
            &json!({"metric": "ic12_pos_diff", "source": "timeslice_chamber"}),
        );
        assert_eq!(legacy.y, "ic12_pos_diff");
        assert_eq!(legacy.frame, "chamber");
        assert_eq!(legacy.grain, "timeslice");
        let source = legacy
            .controls
            .iter()
            .find(|control| control.id == "source")
            .unwrap();
        assert_eq!(source.value, "Timeslice");
    }

    #[test]
    fn catalog_options_carry_an_icon_and_a_unit() {
        let picked = select(Shape::YAndX, true, true, &[], &json!({}));
        for control in &picked.controls {
            assert!(control.options.iter().all(|option| !option.icon.is_empty()));
        }
        let y = picked
            .controls
            .iter()
            .find(|control| control.id == "y")
            .unwrap();
        assert!(y.options.iter().any(|option| option.label.contains("(%)")));
        assert!(y.options.iter().any(|option| option.label.contains("(mm)")));
        let x = picked
            .controls
            .iter()
            .find(|control| control.id == "x")
            .unwrap();
        assert!(x
            .options
            .iter()
            .any(|option| option.label == "Energy (MeV)"));
    }

    #[test]
    fn a_legacy_header_still_parses_string_options() {
        let control: Control =
            serde_json::from_str(r#"{"id":"x","label":"X","options":["Energy"],"value":"Energy"}"#)
                .unwrap();
        assert_eq!(control.options[0].label, "Energy");
        assert_eq!(control.group, "");
        assert_eq!(control.kind, "");
    }
}
