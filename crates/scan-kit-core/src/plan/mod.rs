//! Test-plan rows for the four Plan Synthesis templates.
//!
//! The desktop and MCP call [`build_plan`]. They do not assemble the CSV themselves.

use serde_json::{json, Value};

mod field;
mod name;
mod pld;
mod weight;

use field::{order_within_layers, rectangular_field, sort_rows, zero_field};
use name::suggest_filename;
pub use pld::parse_pld;
use weight::spot_weights;

pub const PREVIEW_CAP: usize = 5_000;
pub(super) const DEFAULT_CURRENT_A: f64 = 1e-9;
pub(super) const DEFAULT_BEAM_SIZE_MM: f64 = 3.61;
pub(super) const DELIVERY_MU_PER_S: f64 = 0.4;
pub(super) const WEIGHT_SCALE: f64 = 10_000.0;

const EXPORT_COLUMNS: [&str; 8] = [
    "#NO",
    "ENERGY(MeV)",
    "CURRENT(A)",
    "BEAM_SIZE(mm)",
    "X_POSITION(mm)",
    "Y_POSITION(mm)",
    "CHARGE_REQ(MU)",
    "VELOCITY(mm/s)",
];

/// 76 energies from the reference session map, 70–250 MeV.
pub fn standard_energies() -> &'static [f64] {
    &STANDARD_ENERGIES_MEV
}

pub(super) const STANDARD_ENERGIES_MEV: [f64; 76] = [
    70.0, 72.0, 74.0, 76.0, 78.0, 80.0, 82.0, 84.0, 86.0, 88.0, 90.0, 92.0, 94.0, 96.0, 98.0,
    100.0, 102.5, 105.0, 107.5, 110.0, 112.5, 115.0, 117.5, 120.0, 122.5, 125.0, 127.5, 130.0,
    132.5, 135.0, 137.5, 140.0, 142.5, 145.0, 147.5, 150.0, 152.5, 155.0, 157.5, 160.0, 162.5,
    165.0, 167.5, 170.0, 172.5, 175.0, 177.5, 180.0, 182.5, 185.0, 187.5, 190.0, 192.5, 195.0,
    197.5, 200.0, 202.5, 205.0, 207.5, 210.0, 212.5, 215.0, 217.5, 220.0, 222.5, 225.0, 227.5,
    230.0, 232.5, 235.0, 237.5, 240.0, 242.5, 245.0, 247.5, 250.0,
];

#[derive(Clone, Debug)]
pub(super) struct Row {
    energy: f64,
    x: f64,
    y: f64,
    charge: f64,
    current: f64,
    beam_size: f64,
    velocity: f64,
}

/// One spot taken from a DICOM plan or an IBA PLD file.
#[derive(Clone, Debug)]
pub struct ImportSpot {
    pub x: f64,
    pub y: f64,
    pub energy: f64,
    pub charge: f64,
    pub beam_size: f64,
    pub plan_index: i32,
}

/// Where the spots come from. Generated templates ignore the import.
pub enum PlanSource<'a> {
    Generate,
    Pld(&'a str),
    Ion(&'a [ImportSpot]),
}

#[derive(Clone, Debug)]
pub struct PlanDocument {
    pub summary: String,
    pub filename: String,
    pub columns: Vec<String>,
    pub preview: Vec<Vec<String>>,
    pub csv: String,
}

/// Template list and parameter specs. Defaults live here.
pub fn plan_catalog() -> Value {
    let energies = Value::Array(
        STANDARD_ENERGIES_MEV
            .iter()
            .copied()
            .map(json_num)
            .collect(),
    );
    let whole = Value::Array(
        STANDARD_ENERGIES_MEV
            .iter()
            .copied()
            .filter(|energy| energy.fract().abs() < 1e-9)
            .map(json_num)
            .collect(),
    );
    let ten = Value::Array(
        STANDARD_ENERGIES_MEV
            .iter()
            .copied()
            .filter(|energy| energy.rem_euclid(10.0) < 1e-6)
            .map(json_num)
            .collect(),
    );
    json!({
        "templates": [
            template("zero_field", "Zero Field", "Every spot at (0, 0) for each energy layer.", json!([
                energy_spec(&energies, &whole, &ten),
                spec_int("spots_per_layer", "Spots per Layer (spots)", 100, 1, 100_000, "geometry")
            ])),
            template("rectangular_field", "Rectangular Field", "Even spot grid per layer with configurable field size.", json!([
                energy_spec(&energies, &whole, &ten),
                spec_float("center_x_mm", "Field Center (mm)", "X", 0.0, -500.0, 500.0, "geometry", json!([{"label": "0,0", "values": {"center_x_mm": 0.0, "center_y_mm": 0.0}}])),
                spec_partner("center_y_mm", "Field Center Y (mm)", "Y", "float", json!(0.0), Some(-500.0), Some(500.0), "center_x_mm", "geometry"),
                spec_float("field_width_mm", "Field Size (mm)", "W", 100.0, 0.001, 1000.0, "geometry", json!([
                    {"label": "100", "values": {"field_width_mm": 100.0, "field_height_mm": 100.0}},
                    {"label": "200", "values": {"field_width_mm": 200.0, "field_height_mm": 200.0}},
                    {"label": "250", "values": {"field_width_mm": 250.0, "field_height_mm": 250.0}},
                    {"label": "300", "values": {"field_width_mm": 300.0, "field_height_mm": 300.0}}
                ])),
                spec_partner("field_height_mm", "Field Size H (mm)", "H", "float", json!(100.0), Some(0.001), Some(1000.0), "field_width_mm", "geometry"),
                spec_int_quick("spots_x", "Spot Grid (spots)", "X", 33, 1, 1000, "geometry", json!([
                    {"label": "3", "values": {"spots_x": 3, "spots_y": 3}},
                    {"label": "7", "values": {"spots_x": 7, "spots_y": 7}},
                    {"label": "11", "values": {"spots_x": 11, "spots_y": 11}},
                    {"label": "33", "values": {"spots_x": 33, "spots_y": 33}}
                ])),
                spec_partner("spots_y", "Spot Grid Y", "Y", "int", json!(33), Some(1.0), Some(1000.0), "spots_x", "geometry"),
                spec_choices("fast_axis", "Fast Axis", "button_group", "x", "geometry", json!([
                    {"value": "x", "label": "X"},
                    {"value": "y", "label": "Y"}
                ]), Value::Null),
                spec_choices("start_corner", "Start Corner", "button_group", "top_left", "geometry", json!([
                    {"value": "top_left", "label": "Top Left"},
                    {"value": "top_right", "label": "Top Right"},
                    {"value": "bottom_left", "label": "Bottom Left"},
                    {"value": "bottom_right", "label": "Bottom Right"}
                ]), Value::Null),
                spec_choices("layer_transition", "Layer Transition", "button_group", "reset", "geometry", json!([
                    {"value": "reset", "label": "Reset to Corner"},
                    {"value": "continue", "label": "Continue from End"}
                ]), Value::Null)
            ])),
            template("dicom_rt_plan", "DICOM RT Plan", "Convert an RT Ion therapy plan (.dcm) into an input map CSV.", json!([
                spec_file("dicom_path", "RT Plan File", "source"),
                spec_bool("use_dicom_beam_size", "Use DICOM Scanning Spot Size", true, "geometry", Value::Null),
                {
                    "key": "beam_size_override_mm",
                    "label": "Beam Size",
                    "kind": "float",
                    "default": DEFAULT_BEAM_SIZE_MM,
                    "minimum": 0.001,
                    "maximum": 500.0,
                    "decimals": 3,
                    "step": 0.1,
                    "suffix": "mm",
                    "field_set": "geometry",
                    "visible_when": {"use_dicom_beam_size": [false]}
                },
                spec_choices("spot_order", "Spot Order", "button_group", "plan_order", "geometry", spot_order_choices(), Value::Null),
                spec_choices("fast_axis", "Fast Axis", "button_group", "x", "geometry", json!([
                    {"value": "x", "label": "X"},
                    {"value": "y", "label": "Y"}
                ]), json!({"spot_order": ["minimize_travel"]}))
            ])),
            template("iba_pld_plan", "IBA PLD Plan", "Convert an IBA PBS plan (.pld) into an input map CSV.", json!([
                spec_file("pld_path", "PLD Plan File", "source"),
                {
                    "key": "beam_size_mm",
                    "label": "Beam Size",
                    "kind": "float",
                    "default": DEFAULT_BEAM_SIZE_MM,
                    "minimum": 0.001,
                    "maximum": 500.0,
                    "decimals": 3,
                    "step": 0.1,
                    "suffix": "mm",
                    "field_set": "geometry"
                },
                spec_choices("spot_order", "Spot Order", "button_group", "plan_order", "geometry", spot_order_choices(), Value::Null),
                spec_choices("fast_axis", "Fast Axis", "button_group", "x", "geometry", json!([
                    {"value": "x", "label": "X"},
                    {"value": "y", "label": "Y"}
                ]), json!({"spot_order": ["minimize_travel"]}))
            ]))
        ]
    })
}

/// Validate, build rows, and return the export document.
pub fn build_plan(
    template: &str,
    params: &Value,
    source: PlanSource<'_>,
    dicom_label: Option<&str>,
) -> Result<PlanDocument, String> {
    let errors = validate_plan(template, params);
    if !errors.is_empty() {
        return Err(errors.join("\n"));
    }
    let rows = match template {
        "zero_field" => zero_field(params)?,
        "rectangular_field" => rectangular_field(params)?,
        "dicom_rt_plan" => match source {
            PlanSource::Ion(spots) => rows_from_import(spots, params)?,
            _ => return Err("DICOM plan file not found.".into()),
        },
        "iba_pld_plan" => match source {
            PlanSource::Pld(text) => {
                let beam = number(params, "beam_size_mm", DEFAULT_BEAM_SIZE_MM);
                let spots = parse_pld(text, beam)?;
                rows_from_import(&spots, params)?
            }
            _ => return Err("PLD plan file not found.".into()),
        },
        _ => return Err("Unknown plan template.".into()),
    };
    Ok(document(template, params, dicom_label, rows))
}

pub fn validate_plan(template: &str, params: &Value) -> Vec<String> {
    match template {
        "zero_field" => {
            let mut errors = validate_energies(params);
            errors.extend(validate_positive_int(
                params.get("spots_per_layer"),
                "Spots per Layer (spots)",
                100_000,
            ));
            errors.extend(validate_weights(params));
            errors
        }
        "rectangular_field" => {
            let mut errors = validate_energies(params);
            errors.extend(validate_positive_float(
                params.get("field_width_mm"),
                "Field Size (mm) W",
            ));
            errors.extend(validate_positive_float(
                params.get("field_height_mm"),
                "Field Size (mm) H",
            ));
            errors.extend(validate_positive_int(
                params.get("spots_x"),
                "Spot Grid (spots) X",
                1000,
            ));
            errors.extend(validate_positive_int(
                params.get("spots_y"),
                "Spot Grid (spots) Y",
                1000,
            ));
            let axis = text(params, "fast_axis", "x");
            if axis != "x" && axis != "y" {
                errors.push("Fast axis must be X or Y.".into());
            }
            let corner = text(params, "start_corner", "top_left");
            if !matches!(
                corner.as_str(),
                "top_left" | "top_right" | "bottom_left" | "bottom_right"
            ) {
                errors.push(
                    "Start corner must be Top Left, Top Right, Bottom Left, or Bottom Right."
                        .into(),
                );
            }
            let transition = text(params, "layer_transition", "reset");
            if transition != "reset" && transition != "continue" {
                errors
                    .push("Layer transition must be Reset to Corner or Continue from End.".into());
            }
            errors.extend(validate_weights(params));
            errors
        }
        "dicom_rt_plan" => {
            let mut errors = Vec::new();
            if !bool_param(params, "use_dicom_beam_size", true) {
                errors.extend(validate_positive_float(
                    params.get("beam_size_override_mm"),
                    "Beam Size (mm)",
                ));
            }
            errors.extend(validate_spot_order(params));
            errors
        }
        "iba_pld_plan" => {
            let mut errors = validate_positive_float(params.get("beam_size_mm"), "Beam Size (mm)");
            errors.extend(validate_spot_order(params));
            errors
        }
        _ => vec!["Unknown plan template.".into()],
    }
}

/// Average of a DICOM scanning-spot size pair, when both values are usable.
pub fn dicom_beam_size(size_x: f64, size_y: f64, fallback: f64, use_dicom: bool) -> f64 {
    if use_dicom
        && size_x.is_finite()
        && size_y.is_finite()
        && size_x > 0.0
        && size_y > 0.0
        && size_x < 500.0
        && size_y < 500.0
    {
        (size_x + size_y) / 2.0
    } else {
        fallback
    }
}

pub(super) struct Draft {
    energy: f64,
    layer: i32,
    x: f64,
    y: f64,
}

pub(super) fn finish_generated(draft: Vec<Draft>, params: &Value) -> Result<Vec<Row>, String> {
    let charges = spot_weights(&draft, params)?;
    let current = DEFAULT_CURRENT_A;
    let beam = DEFAULT_BEAM_SIZE_MM;
    let mut rows: Vec<Row> = draft
        .into_iter()
        .zip(charges)
        .map(|(spot, charge)| Row {
            energy: spot.energy,
            x: spot.x,
            y: spot.y,
            charge,
            current,
            beam_size: beam,
            velocity: 0.0,
        })
        .collect();
    sort_rows(&mut rows);
    Ok(rows)
}

fn rows_from_import(spots: &[ImportSpot], params: &Value) -> Result<Vec<Row>, String> {
    if spots.is_empty() {
        return Err("No planned spots with positive MU found in plan file".into());
    }
    let order = text(params, "spot_order", "plan_order");
    let axis = text(params, "fast_axis", "x");
    let ordered = order_within_layers(spots, &order, &axis);
    let current = DEFAULT_CURRENT_A;
    let mut rows: Vec<Row> = ordered
        .into_iter()
        .map(|spot| Row {
            energy: spot.energy,
            x: spot.x,
            y: spot.y,
            charge: spot.charge,
            current,
            beam_size: spot.beam_size,
            velocity: 0.0,
        })
        .collect();
    sort_rows(&mut rows);
    Ok(rows)
}

fn document(
    template: &str,
    params: &Value,
    dicom_label: Option<&str>,
    rows: Vec<Row>,
) -> PlanDocument {
    let summary = format_summary(&rows, PREVIEW_CAP);
    let filename = suggest_filename(template, params, dicom_label, 128);
    let columns = EXPORT_COLUMNS
        .iter()
        .map(|name| (*name).to_owned())
        .collect();
    let preview = rows
        .iter()
        .take(PREVIEW_CAP)
        .enumerate()
        .map(|(index, row)| preview_row(index, row))
        .collect();
    let csv = to_csv(&rows);
    PlanDocument {
        summary,
        filename,
        columns,
        preview,
        csv,
    }
}

fn preview_row(index: usize, row: &Row) -> Vec<String> {
    vec![
        (index + 1).to_string(),
        trim_float(row.energy),
        trim_float(row.current),
        trim_float(row.beam_size),
        trim_float(row.x),
        trim_float(row.y),
        format!("{:.4}", row.charge),
        trim_float(row.velocity),
    ]
}

fn to_csv(rows: &[Row]) -> String {
    let mut out = EXPORT_COLUMNS.join(",");
    out.push('\n');
    for (index, row) in rows.iter().enumerate() {
        out.push_str(&format!(
            "{},{},{},{},{},{},{},{}\n",
            index + 1,
            csv_num(row.energy),
            csv_num(row.current),
            csv_num(row.beam_size),
            csv_num(row.x),
            csv_num(row.y),
            csv_num(row.charge),
            csv_num(row.velocity),
        ));
    }
    out
}

fn csv_num(value: f64) -> String {
    if value == 0.0 {
        return "0".into();
    }
    trim_zeros(&format!("{value:.12}"))
}

fn trim_float(value: f64) -> String {
    if value == 0.0 {
        return "0".into();
    }
    trim_zeros(&format!("{value:.6}"))
}

pub(super) fn trim_zeros(text: &str) -> String {
    if !text.contains('.') {
        return text.to_owned();
    }
    let trimmed = text.trim_end_matches('0').trim_end_matches('.');
    if trimmed.is_empty() || trimmed == "-" {
        "0".into()
    } else {
        trimmed.to_owned()
    }
}

fn format_summary(rows: &[Row], cap: usize) -> String {
    if rows.is_empty() {
        return "No plan generated yet.".into();
    }
    let layers = unique_energies(rows);
    let spots = rows.len();
    let total: f64 = rows.iter().map(|row| row.charge).sum();
    let layer_word = if layers == 1 { "layer" } else { "layers" };
    let spot_word = if spots == 1 { "spot" } else { "spots" };
    let mut message = format!(
        "{layers} {layer_word} · {spots} {spot_word} · {} MU total · est. {} delivery",
        format_sig(total, 3),
        format_delivery(total / DELIVERY_MU_PER_S)
    );
    if spots > cap {
        message.push_str(&format!(" · preview shows first {} rows", with_commas(cap)));
    }
    message
}

fn unique_energies(rows: &[Row]) -> usize {
    let mut seen = Vec::new();
    for row in rows {
        if !seen.contains(&row.energy) {
            seen.push(row.energy);
        }
    }
    seen.len()
}

fn format_delivery(seconds: f64) -> String {
    if seconds < 60.0 {
        return format!("{seconds:.1} s");
    }
    let total_minutes = (seconds / 60.0).floor() as i64;
    if total_minutes < 60 {
        let rem = (seconds % 60.0).round() as i64;
        if rem == 0 {
            return format!("{total_minutes} min");
        }
        return format!("{total_minutes} min {rem} s");
    }
    let hours = total_minutes / 60;
    let minutes = total_minutes % 60;
    if minutes == 0 {
        format!("{hours} h")
    } else {
        format!("{hours} h {minutes} min")
    }
}

fn format_sig(value: f64, digits: usize) -> String {
    if value == 0.0 || !value.is_finite() {
        return "0".into();
    }
    let sign = if value < 0.0 { "-" } else { "" };
    let abs = value.abs();
    let mut exp = abs.log10().floor() as i32;
    let mut rounded = round_sig(abs, digits, exp);
    let exp2 = rounded.log10().floor() as i32;
    if exp2 != exp {
        exp = exp2;
        rounded = round_sig(abs, digits, exp);
    }
    if exp < -4 || exp >= digits as i32 {
        let mantissa = rounded / 10f64.powi(exp);
        let prec = digits.saturating_sub(1);
        let text = trim_zeros(&format!("{mantissa:.prec$}"));
        format!("{sign}{text}e{exp:+03}")
    } else {
        let prec = (digits as i32 - 1 - exp).max(0) as usize;
        format!("{sign}{}", trim_zeros(&format!("{rounded:.prec$}")))
    }
}

fn round_sig(abs: f64, digits: usize, exp: i32) -> f64 {
    let scale = 10f64.powi(digits as i32 - 1 - exp);
    (abs * scale).round() / scale
}

fn with_commas(value: usize) -> String {
    let text = value.to_string();
    let mut out = String::new();
    for (index, ch) in text.chars().rev().enumerate() {
        if index > 0 && index % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    out.chars().rev().collect()
}

pub(super) fn selected_energies(params: &Value) -> Vec<f64> {
    let Some(list) = params.get("selected_energies").and_then(Value::as_array) else {
        return Vec::new();
    };
    let selected: Vec<f64> = list.iter().filter_map(Value::as_f64).collect();
    STANDARD_ENERGIES_MEV
        .iter()
        .rev()
        .copied()
        .filter(|energy| selected.contains(energy))
        .collect()
}

fn validate_energies(params: &Value) -> Vec<String> {
    let list = params.get("selected_energies").and_then(Value::as_array);
    if list.map(|items| items.is_empty()).unwrap_or(true) {
        return vec!["Select at least one energy layer.".into()];
    }
    if selected_energies(params).is_empty() {
        return vec!["Select at least one energy layer.".into()];
    }
    Vec::new()
}

fn validate_weights(params: &Value) -> Vec<String> {
    let method = text(params, "spot_weight_method", "fixed");
    match method.as_str() {
        "fixed" => validate_positive_float(params.get("spot_weight_mu"), "Spot Weight (MU)"),
        "even_total" => validate_positive_float(
            params.get("spot_weight_total_mu"),
            "Target Total Weight (MU)",
        ),
        "random_total_variance" => {
            let mut errors = validate_positive_float(
                params.get("spot_weight_total_mu"),
                "Target Total Weight (MU)",
            );
            if !errors.is_empty() {
                return errors;
            }
            errors.extend(validate_variance(params.get("spot_weight_variance_pct")));
            errors
        }
        "random_range" | "layer_even_range" => validate_weight_range(params),
        _ => vec!["Select a spot weight method.".into()],
    }
}

fn validate_weight_range(params: &Value) -> Vec<String> {
    let mut errors =
        validate_positive_float(params.get("spot_weight_min_mu"), "Minimum Weight (MU)");
    errors.extend(validate_positive_float(
        params.get("spot_weight_max_mu"),
        "Maximum Weight (MU)",
    ));
    if !errors.is_empty() {
        return errors;
    }
    if number(params, "spot_weight_max_mu", 0.0) < number(params, "spot_weight_min_mu", 0.0) {
        return vec![
            "Maximum Weight (MU) must be greater than or equal to Minimum Weight (MU).".into(),
        ];
    }
    Vec::new()
}

fn validate_variance(value: Option<&Value>) -> Vec<String> {
    let Some(number) = value.and_then(Value::as_f64) else {
        return vec!["Spot Variance (%) must be a number.".into()];
    };
    if !number.is_finite() {
        return vec!["Spot Variance (%) must be a number.".into()];
    }
    if number < 0.0 {
        return vec!["Spot Variance (%) must be zero or greater.".into()];
    }
    if number > 100.0 {
        return vec!["Spot Variance (%) must be at most 100.".into()];
    }
    Vec::new()
}

fn validate_spot_order(params: &Value) -> Vec<String> {
    let mut errors = Vec::new();
    let order = text(params, "spot_order", "plan_order");
    if order != "plan_order" && order != "minimize_travel" {
        errors.push("Spot order must be Plan Order or Minimize Travel.".into());
    }
    if order == "minimize_travel" {
        let axis = text(params, "fast_axis", "x");
        if axis != "x" && axis != "y" {
            errors.push("Fast axis must be X or Y.".into());
        }
    }
    errors
}

fn validate_positive_float(value: Option<&Value>, label: &str) -> Vec<String> {
    let Some(number) = value.and_then(Value::as_f64) else {
        return vec![format!("{label} must be a number.")];
    };
    if !number.is_finite() {
        return vec![format!("{label} must be a finite number.")];
    }
    if number <= 0.0 {
        return vec![format!("{label} must be greater than zero.")];
    }
    Vec::new()
}

fn validate_positive_int(value: Option<&Value>, label: &str, max: i64) -> Vec<String> {
    let Some(number) = value.and_then(as_int) else {
        return vec![format!("{label} must be an integer.")];
    };
    if number < 1 {
        return vec![format!("{label} must be at least 1.")];
    }
    if number > max {
        return vec![format!("{label} must be at most {max}.")];
    }
    Vec::new()
}

fn as_int(value: &Value) -> Option<i64> {
    if let Some(number) = value.as_i64() {
        return Some(number);
    }
    let number = value.as_f64()?;
    if number.is_finite() && number.fract() == 0.0 {
        Some(number as i64)
    } else {
        None
    }
}

pub(super) fn number(params: &Value, key: &str, fallback: f64) -> f64 {
    params
        .get(key)
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite())
        .unwrap_or(fallback)
}

pub(super) fn int_param(params: &Value, key: &str, fallback: i64) -> i64 {
    params.get(key).and_then(as_int).unwrap_or(fallback)
}

pub(super) fn text(params: &Value, key: &str, fallback: &str) -> String {
    params
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or(fallback)
        .to_owned()
}

pub(super) fn bool_param(params: &Value, key: &str, fallback: bool) -> bool {
    params.get(key).and_then(Value::as_bool).unwrap_or(fallback)
}

fn json_num(value: f64) -> Value {
    serde_json::Number::from_f64(value)
        .map(Value::Number)
        .unwrap_or(Value::Null)
}

fn template(id: &str, name: &str, description: &str, mut params: Value) -> Value {
    if id == "zero_field" || id == "rectangular_field" {
        if let Some(list) = params.as_array_mut() {
            list.extend(weight_specs());
        }
    }
    json!({ "id": id, "name": name, "description": description, "params": params })
}

fn energy_spec(all: &Value, whole: &Value, ten: &Value) -> Value {
    json!({
        "key": "selected_energies",
        "label": "Energy Layers (MeV)",
        "kind": "energy_multiselect",
        "default": all,
        "field_set": "energy",
        "presets": [
            {"label": "Select All", "energies": all},
            {"label": "Whole MeV Steps", "energies": whole},
            {"label": "10 MeV Steps", "energies": ten},
            {"label": "Clear All", "energies": []}
        ]
    })
}

fn weight_specs() -> Vec<Value> {
    vec![
        spec_choices(
            "spot_weight_method",
            "Spot Weight Method",
            "choice",
            "fixed",
            "weight",
            json!([
                {"value": "fixed", "label": "Fixed"},
                {"value": "random_range", "label": "Random Range"},
                {"value": "layer_even_range", "label": "Even per Layer"},
                {"value": "even_total", "label": "Even Total"},
                {"value": "random_total_variance", "label": "Random Total"}
            ]),
            Value::Null,
        ),
        spec_visible_float(
            "spot_weight_mu",
            "Spot Weight (MU)",
            0.02,
            0.0001,
            json!({"spot_weight_method": ["fixed"]}),
        ),
        spec_visible_float(
            "spot_weight_total_mu",
            "Target Total Weight (MU)",
            1.0,
            0.0001,
            json!({"spot_weight_method": ["even_total", "random_total_variance"]}),
        ),
        json!({
            "key": "spot_weight_variance_pct",
            "label": "Spot Variance (%)",
            "kind": "float",
            "default": 10.0,
            "minimum": 0.0,
            "maximum": 100.0,
            "decimals": 1,
            "step": 1.0,
            "field_set": "weight",
            "visible_when": {"spot_weight_method": ["random_total_variance"]}
        }),
        json!({
            "key": "spot_weight_min_mu",
            "label": "Weight Range (MU)",
            "sub_label": "Min",
            "kind": "float",
            "default": 0.002,
            "minimum": 0.0001,
            "decimals": 4,
            "step": 0.001,
            "field_set": "weight",
            "visible_when": {"spot_weight_method": ["random_range", "layer_even_range"]}
        }),
        json!({
            "key": "spot_weight_max_mu",
            "label": "Weight Range (MU)",
            "sub_label": "Max",
            "row_partner": "spot_weight_min_mu",
            "kind": "float",
            "default": 0.1,
            "minimum": 0.0001,
            "decimals": 4,
            "step": 0.001,
            "field_set": "weight",
            "visible_when": {"spot_weight_method": ["random_range", "layer_even_range"]}
        }),
        spec_bool(
            "spot_weight_layer_shuffle",
            "Shuffle Order per Layer",
            false,
            "weight",
            json!({"spot_weight_method": ["layer_even_range"]}),
        ),
    ]
}

fn spec_int(key: &str, label: &str, default: i64, min: i64, max: i64, field: &str) -> Value {
    json!({
        "key": key,
        "label": label,
        "kind": "int",
        "default": default,
        "minimum": min,
        "maximum": max,
        "step": 1,
        "field_set": field
    })
}

fn spec_int_quick(
    key: &str,
    label: &str,
    sub: &str,
    default: i64,
    min: i64,
    max: i64,
    field: &str,
    quick: Value,
) -> Value {
    json!({
        "key": key,
        "label": label,
        "sub_label": sub,
        "kind": "int",
        "default": default,
        "minimum": min,
        "maximum": max,
        "step": 1,
        "field_set": field,
        "quick_sets": quick
    })
}

fn spec_float(
    key: &str,
    label: &str,
    sub: &str,
    default: f64,
    min: f64,
    max: f64,
    field: &str,
    quick: Value,
) -> Value {
    json!({
        "key": key,
        "label": label,
        "sub_label": sub,
        "kind": "float",
        "default": default,
        "minimum": min,
        "maximum": max,
        "decimals": 3,
        "step": 0.1,
        "field_set": field,
        "quick_sets": quick
    })
}

fn spec_partner(
    key: &str,
    label: &str,
    sub: &str,
    kind: &str,
    default: Value,
    min: Option<f64>,
    max: Option<f64>,
    partner: &str,
    field: &str,
) -> Value {
    json!({
        "key": key,
        "label": label,
        "sub_label": sub,
        "kind": kind,
        "default": default,
        "minimum": min,
        "maximum": max,
        "decimals": 3,
        "step": if kind == "int" { json!(1) } else { json!(0.1) },
        "row_partner": partner,
        "field_set": field
    })
}

fn spec_choices(
    key: &str,
    label: &str,
    kind: &str,
    default: &str,
    field: &str,
    choices: Value,
    visible: Value,
) -> Value {
    let mut value = json!({
        "key": key,
        "label": label,
        "kind": kind,
        "default": default,
        "field_set": field,
        "choices": choices
    });
    if !visible.is_null() {
        value["visible_when"] = visible;
    }
    value
}

fn spec_file(key: &str, label: &str, field: &str) -> Value {
    json!({
        "key": key,
        "label": label,
        "kind": "file_path",
        "default": "",
        "field_set": field
    })
}

fn spec_bool(key: &str, label: &str, default: bool, field: &str, visible: Value) -> Value {
    let mut value = json!({
        "key": key,
        "label": label,
        "kind": "bool",
        "default": default,
        "field_set": field
    });
    if !visible.is_null() {
        value["visible_when"] = visible;
    }
    value
}

fn spec_visible_float(key: &str, label: &str, default: f64, min: f64, visible: Value) -> Value {
    json!({
        "key": key,
        "label": label,
        "kind": "float",
        "default": default,
        "minimum": min,
        "decimals": 4,
        "step": 0.01,
        "field_set": "weight",
        "visible_when": visible
    })
}

fn spot_order_choices() -> Value {
    json!([
        {"value": "plan_order", "label": "Plan Order"},
        {"value": "minimize_travel", "label": "Minimize Travel"}
    ])
}

#[cfg(test)]
mod tests {
    use super::field::grid_positions;
    use super::weight::{even_total, layer_even, random_total};
    use super::*;

    fn params(value: Value) -> Value {
        value
    }

    #[test]
    fn zero_field_stays_at_the_origin_and_sorts_high_energy_first() {
        let built = build_plan(
            "zero_field",
            &params(json!({
                "selected_energies": [245.0, 250.0, 247.5],
                "spots_per_layer": 10,
                "spot_weight_method": "fixed",
                "spot_weight_mu": 0.02
            })),
            PlanSource::Generate,
            None,
        )
        .unwrap();
        assert!(built.summary.starts_with("3 layers · 30 spots"));
        assert!(built.filename.starts_with("ZeroField_"));
        assert!(built.filename.contains("E250-245"));
        assert!(built.filename.contains("Sp10"));
        assert!(built.filename.contains("Wfix0.02"));
        let mut lines = built.csv.lines();
        assert_eq!(
            lines.next().unwrap(),
            "#NO,ENERGY(MeV),CURRENT(A),BEAM_SIZE(mm),X_POSITION(mm),Y_POSITION(mm),CHARGE_REQ(MU),VELOCITY(mm/s)"
        );
        let first = lines.next().unwrap();
        assert!(first.starts_with("1,250,"));
        assert!(first.contains(",0,0,"));
        let last = built.csv.lines().next_back().unwrap();
        assert!(last.contains(",245,"));
        assert_eq!(built.csv.lines().count(), 31);
    }

    #[test]
    fn empty_energies_are_rejected() {
        let errors = validate_plan(
            "zero_field",
            &json!({"selected_energies": [], "spots_per_layer": 1, "spot_weight_method": "fixed", "spot_weight_mu": 0.02}),
        );
        assert!(errors.iter().any(|error| error.contains("energy")));
    }

    #[test]
    fn rectangular_serpentine_matches_the_top_left_fast_x_grid() {
        let positions = grid_positions(0.0, 0.0, 20.0, 20.0, 3, 3, "x", "top_left");
        assert_eq!(
            positions,
            vec![
                (-10.0, 10.0),
                (0.0, 10.0),
                (10.0, 10.0),
                (10.0, 0.0),
                (0.0, 0.0),
                (-10.0, 0.0),
                (-10.0, -10.0),
                (0.0, -10.0),
                (10.0, -10.0),
            ]
        );
    }

    #[test]
    fn continue_reverses_every_other_layer() {
        let built = build_plan(
            "rectangular_field",
            &json!({
                "selected_energies": [250.0, 70.0],
                "center_x_mm": 0.0,
                "center_y_mm": 0.0,
                "field_width_mm": 20.0,
                "field_height_mm": 20.0,
                "spots_x": 2,
                "spots_y": 1,
                "fast_axis": "x",
                "start_corner": "top_left",
                "layer_transition": "continue",
                "spot_weight_method": "fixed",
                "spot_weight_mu": 0.02
            }),
            PlanSource::Generate,
            None,
        )
        .unwrap();
        let rows: Vec<&str> = built.csv.lines().skip(1).collect();
        assert!(rows[0].contains(",250,"));
        assert!(rows[0].contains(",-10,"));
        assert!(rows[1].contains(",10,"));
        assert!(rows[2].contains(",70,"));
        assert!(rows[2].contains(",10,"));
        assert!(rows[3].contains(",-10,"));
    }

    #[test]
    fn even_total_puts_the_remainder_on_the_last_spot() {
        let weights = even_total(3, 1.0).unwrap();
        let sum: f64 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
        assert!(weights.iter().all(|weight| *weight > 0.0));
        assert!((weights[2] - weights[0] - 0.0001).abs() < 1e-9);
    }

    #[test]
    fn layer_even_range_is_linear_per_layer() {
        let rows = vec![
            Draft {
                energy: 250.0,
                layer: 10,
                x: 0.0,
                y: 0.0,
            },
            Draft {
                energy: 250.0,
                layer: 10,
                x: 1.0,
                y: 0.0,
            },
            Draft {
                energy: 250.0,
                layer: 10,
                x: 2.0,
                y: 0.0,
            },
            Draft {
                energy: 247.5,
                layer: 11,
                x: 0.0,
                y: 0.0,
            },
            Draft {
                energy: 247.5,
                layer: 11,
                x: 1.0,
                y: 0.0,
            },
        ];
        let weights = layer_even(&rows, 0.01, 0.03, false);
        assert!((weights[0] - 0.01).abs() < 1e-9);
        assert!((weights[1] - 0.02).abs() < 1e-9);
        assert!((weights[2] - 0.03).abs() < 1e-9);
        assert!((weights[3] - 0.01).abs() < 1e-9);
        assert!((weights[4] - 0.03).abs() < 1e-9);
    }

    #[test]
    fn zero_variance_matches_even_total() {
        let even = even_total(4, 1.0).unwrap();
        let random = random_total(4, 1.0, 0.0).unwrap();
        assert_eq!(even, random);
    }

    #[test]
    fn variance_above_100_is_rejected() {
        let errors = validate_plan(
            "zero_field",
            &json!({
                "selected_energies": [250.0],
                "spots_per_layer": 2,
                "spot_weight_method": "random_total_variance",
                "spot_weight_total_mu": 1.0,
                "spot_weight_variance_pct": 150.0
            }),
        );
        assert!(errors.iter().any(|error| error.contains("Spot Variance")));
    }

    #[test]
    fn inverted_weight_range_is_rejected() {
        let errors = validate_plan(
            "zero_field",
            &json!({
                "selected_energies": [250.0],
                "spots_per_layer": 2,
                "spot_weight_method": "random_range",
                "spot_weight_min_mu": 0.2,
                "spot_weight_max_mu": 0.1
            }),
        );
        assert!(errors
            .iter()
            .any(|error| error.contains("greater than or equal")));
    }

    #[test]
    fn summary_uses_three_sig_figs_and_a_delivery_estimate() {
        let rows = vec![
            Row {
                energy: 200.0,
                x: 0.0,
                y: 0.0,
                charge: 0.4,
                current: 0.0,
                beam_size: 1.0,
                velocity: 0.0,
            },
            Row {
                energy: 200.0,
                x: 0.0,
                y: 0.0,
                charge: 0.4,
                current: 0.0,
                beam_size: 1.0,
                velocity: 0.0,
            },
        ];
        let summary = format_summary(&rows, 10);
        assert!(summary.contains("0.8 MU total"));
        assert!(summary.contains("est. 2.0 s delivery"));
        assert!(!summary.contains("preview"));
        let wide = vec![
            Row {
                energy: 1.0,
                x: 0.0,
                y: 0.0,
                charge: 1234.567,
                current: 0.0,
                beam_size: 1.0,
                velocity: 0.0
            };
            2
        ];
        assert!(format_summary(&wide, 10).contains("2.47e+03 MU total"));
        assert!(format_summary(&rows, 1).contains("preview shows first 1 rows"));
        assert_eq!(format_delivery(125.0), "2 min 5 s");
        assert_eq!(format_delivery(3660.0), "1 h 1 min");
    }

    #[test]
    fn stable_energy_sort_keeps_spot_order_inside_a_layer() {
        let mut rows = vec![
            Row {
                energy: 200.0,
                x: 3.0,
                y: 0.0,
                charge: 0.01,
                current: 0.0,
                beam_size: 3.61,
                velocity: 0.0,
            },
            Row {
                energy: 100.0,
                x: 1.0,
                y: 0.0,
                charge: 0.01,
                current: 0.0,
                beam_size: 3.61,
                velocity: 0.0,
            },
            Row {
                energy: 100.0,
                x: 2.0,
                y: 0.0,
                charge: 0.01,
                current: 0.0,
                beam_size: 3.61,
                velocity: 0.0,
            },
            Row {
                energy: 200.0,
                x: 4.0,
                y: 0.0,
                charge: 0.01,
                current: 0.0,
                beam_size: 3.61,
                velocity: 0.0,
            },
        ];
        sort_rows(&mut rows);
        assert_eq!(rows[0].x, 3.0);
        assert_eq!(rows[1].x, 4.0);
        assert_eq!(rows[2].x, 1.0);
        let csv = to_csv(&rows);
        assert!(csv.lines().nth(1).unwrap().starts_with("1,200,"));
        assert!(csv.lines().nth(3).unwrap().starts_with("3,100,"));
    }

    #[test]
    fn filename_covers_a_rectangle_and_the_full_catalog() {
        let name = suggest_filename(
            "rectangular_field",
            &json!({
                "selected_energies": [200.0],
                "center_x_mm": 10.0,
                "center_y_mm": -5.0,
                "field_width_mm": 100.0,
                "field_height_mm": 80.0,
                "spots_x": 33,
                "spots_y": 33,
                "spot_weight_method": "even_total",
                "spot_weight_total_mu": 2.0
            }),
            None,
            128,
        );
        assert!(name.starts_with("RectField_"));
        assert!(name.contains("E200"));
        assert!(name.contains("C10x-5"));
        assert!(name.contains("100x80mm"));
        assert!(name.contains("G33x33"));
        assert!(name.contains("Wtot2"));
        let all: Vec<f64> = STANDARD_ENERGIES_MEV.to_vec();
        let full = suggest_filename(
            "zero_field",
            &json!({"selected_energies": all, "spots_per_layer": 100, "spot_weight_method": "fixed", "spot_weight_mu": 0.02}),
            None,
            128,
        );
        assert!(full.contains("E250-70"));
        assert!(!full.contains("76L"));
        let many = suggest_filename(
            "zero_field",
            &json!({"selected_energies": [210.0, 250.0, 230.0, 240.0, 220.0], "spots_per_layer": 1, "spot_weight_method": "fixed", "spot_weight_mu": 0.02}),
            None,
            128,
        );
        assert!(many.contains("E5L_250-210"));
        let short = suggest_filename(
            "zero_field",
            &json!({"selected_energies": STANDARD_ENERGIES_MEV.to_vec(), "spots_per_layer": 100000, "spot_weight_method": "fixed", "spot_weight_mu": 0.02}),
            None,
            40,
        );
        assert!(short.len() <= 40);
        assert!(short.ends_with(".csv"));
    }

    #[test]
    fn minimal_pld_keeps_the_positive_spots() {
        let text = "\
Beam,Patient ID,Patient Name,Patient Initial,Patient Firstname,TestPlan,Beam1,10,10,1
Layer,Spot1,100,10,4
Element,0,0,0,0.0
Element,0,0,5,0.0
Element,10,0,0,0.0
Element,10,0,5,0.0
";
        let spots = parse_pld(text, 4.0).unwrap();
        assert_eq!(spots.len(), 2);
        assert_eq!(spots[0].energy, 100.0);
        assert_eq!(spots[0].x, 0.0);
        assert_eq!(spots[1].x, 10.0);
        assert!((spots[0].charge - 5.0).abs() < 1e-9);
        assert!((spots[1].charge - 5.0).abs() < 1e-9);
        assert_eq!(spots[0].beam_size, 4.0);
        let built = build_plan(
            "iba_pld_plan",
            &json!({
                "pld_path": "minimal.pld",
                "beam_size_mm": 4.0,
                "spot_order": "plan_order",
                "fast_axis": "x"
            }),
            PlanSource::Pld(text),
            None,
        )
        .unwrap();
        assert_eq!(built.filename, "minimal.csv");
        assert!(built.csv.lines().nth(1).unwrap().contains(",100,"));
    }

    #[test]
    fn minimize_travel_orders_a_scrambled_layer() {
        let spots = vec![
            ImportSpot {
                x: 20.0,
                y: 0.0,
                energy: 100.0,
                charge: 0.5,
                beam_size: 3.61,
                plan_index: 0,
            },
            ImportSpot {
                x: 0.0,
                y: 0.0,
                energy: 100.0,
                charge: 0.5,
                beam_size: 3.61,
                plan_index: 1,
            },
            ImportSpot {
                x: 20.0,
                y: 10.0,
                energy: 100.0,
                charge: 0.5,
                beam_size: 3.61,
                plan_index: 2,
            },
            ImportSpot {
                x: 0.0,
                y: 10.0,
                energy: 100.0,
                charge: 0.5,
                beam_size: 3.61,
                plan_index: 3,
            },
        ];
        let built = build_plan(
            "dicom_rt_plan",
            &json!({
                "use_dicom_beam_size": false,
                "beam_size_override_mm": 3.61,
                "spot_order": "minimize_travel",
                "fast_axis": "x"
            }),
            PlanSource::Ion(&spots),
            Some("TESTPLAN"),
        )
        .unwrap();
        let xs: Vec<f64> = built
            .csv
            .lines()
            .skip(1)
            .map(|line| line.split(',').nth(4).unwrap().parse().unwrap())
            .collect();
        let ys: Vec<f64> = built
            .csv
            .lines()
            .skip(1)
            .map(|line| line.split(',').nth(5).unwrap().parse().unwrap())
            .collect();
        assert_eq!(xs, vec![0.0, 20.0, 20.0, 0.0]);
        assert_eq!(ys, vec![0.0, 0.0, 10.0, 10.0]);
        assert!(built.filename.starts_with("DicomPlan_"));
        assert!(built.filename.contains("TESTPLAN"));
    }

    #[test]
    fn plan_order_keeps_the_file_sequence() {
        let spots = vec![
            ImportSpot {
                x: 30.0,
                y: 0.0,
                energy: 100.0,
                charge: 0.5,
                beam_size: 7.5,
                plan_index: 0,
            },
            ImportSpot {
                x: 10.0,
                y: 0.0,
                energy: 100.0,
                charge: 0.5,
                beam_size: 7.5,
                plan_index: 1,
            },
            ImportSpot {
                x: 20.0,
                y: 0.0,
                energy: 100.0,
                charge: 0.5,
                beam_size: 7.5,
                plan_index: 2,
            },
        ];
        let built = build_plan(
            "dicom_rt_plan",
            &json!({
                "use_dicom_beam_size": false,
                "beam_size_override_mm": 7.5,
                "spot_order": "plan_order"
            }),
            PlanSource::Ion(&spots),
            None,
        )
        .unwrap();
        let xs: Vec<&str> = built
            .csv
            .lines()
            .skip(1)
            .map(|line| line.split(',').nth(4).unwrap())
            .collect();
        assert_eq!(xs, vec!["30", "10", "20"]);
        assert!(built.csv.contains(",7.5,"));
    }

    #[test]
    fn catalog_lists_the_four_templates() {
        let catalog = plan_catalog();
        let names: Vec<&str> = catalog["templates"]
            .as_array()
            .unwrap()
            .iter()
            .map(|item| item["id"].as_str().unwrap())
            .collect();
        assert_eq!(
            names,
            vec![
                "zero_field",
                "rectangular_field",
                "dicom_rt_plan",
                "iba_pld_plan"
            ]
        );
        assert!(catalog["templates"][0]["params"]
            .as_array()
            .unwrap()
            .iter()
            .any(|spec| spec["key"] == "spot_weight_method"));
    }
}
