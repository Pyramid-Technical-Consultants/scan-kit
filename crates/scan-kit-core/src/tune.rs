//! The four devices.xml tuners. Session columns are loaded by scan-kit-io.

use serde_json::{json, Value};

use crate::config::config_form;
use crate::xml_dom::{parse_xml, write_xml, Elem};

const DEVICES: [&str; 4] = ["IC_1_X", "IC_1_Y", "IC_2_X", "IC_2_Y"];

pub struct TuneSpots {
    pub energy: Vec<f64>,
    pub weight: Vec<f64>,
    pub plan_x: Vec<f64>,
    pub plan_y: Vec<f64>,
    pub measured: [Vec<f64>; 4],
    pub error: [Vec<f64>; 4],
    pub sigma: [Vec<f64>; 4],
    pub mu: [Vec<f64>; 3],
    pub charge: Vec<f64>,
}

pub fn tune_catalog() -> Value {
    json!({
        "workflows": [
            {
                "id": "sigma_tuning",
                "name": "Sigma Tuning",
                "description": "IC1/IC2 σ K0 to fit ±tolerance band",
                "params": [
                    spec_float("sigma_tolerance_percent", "Sigma Tolerance (%)", 20.0, 0.0, 0.1),
                    spec_float("sigma_lower_headroom_percent", "Lower Headroom (%)", 1.0, 0.0, 0.1)
                ]
            },
            {
                "id": "position_offset_tuning",
                "name": "Position Offset Tuning",
                "description": "IC1/IC2 zero offset at iso mm from session(s)",
                "params": [
                    spec_choice("optimize_method", "Optimize Using", "median", json!([
                        {"value": "median", "label": "Median"},
                        {"value": "weighted_average", "label": "Weighted Average"},
                        {"value": "min_max_midpoint", "label": "Min-Max Midpoint"}
                    ])),
                    spec_choice("data_source", "Data Source", "spot", json!([
                        {"value": "spot", "label": "Spot"},
                        {"value": "timeslice", "label": "Timeslice"}
                    ]))
                ]
            },
            {
                "id": "ic_distance_tuning",
                "name": "IC Distance Tuning",
                "description": "IC1/IC2 source to device distance + zero offset from session(s)",
                "params": []
            },
            {
                "id": "kmu_tuning",
                "name": "Dose Calibration",
                "description": "Secondary IC K_MU to match primary (optional primary rescale)",
                "params": [
                    spec_choice("primary_ic", "Primary IC", "ic1", json!([
                        {"value": "ic1", "label": "IC1"},
                        {"value": "ic2", "label": "IC2"},
                        {"value": "ic3", "label": "IC3"}
                    ])),
                    spec_choice("primary_mode", "Primary K_MU", "unchanged", json!([
                        {"value": "unchanged", "label": "Leave Unchanged"},
                        {"value": "known_mu", "label": "Known MU"},
                        {"value": "percent", "label": "Adjust by Percent"}
                    ])),
                    spec_float_when("known_mu", "Known MU", 0.0, 0.0, 0.1, "primary_mode", "known_mu"),
                    spec_float_when("percent", "Percent (%)", 0.0, -99.0, 0.1, "primary_mode", "percent")
                ]
            }
        ]
    })
}

pub fn run_tune(
    workflow: &str,
    xml: &str,
    spots: &TuneSpots,
    params: &Value,
    session_label: &str,
) -> Result<Value, String> {
    let mut root = parse_xml(xml)?;
    let (summary, warnings, columns, rows, changed) = match workflow {
        "sigma_tuning" => tune_sigma(&mut root, spots, params, session_label)?,
        "position_offset_tuning" => tune_offset(&mut root, spots, params, session_label)?,
        "ic_distance_tuning" => tune_distance(&mut root, spots, session_label)?,
        "kmu_tuning" => tune_kmu(&mut root, spots, params, session_label)?,
        _ => return Err(format!("unknown tuning workflow {workflow}")),
    };
    let xml = if changed {
        write_xml(&root)
    } else {
        xml.to_owned()
    };
    let form = config_form(&xml)?;
    Ok(json!({
        "xml": xml,
        "form": form,
        "summary": summary,
        "warnings": warnings,
        "columns": columns,
        "rows": rows,
        "changed": changed,
    }))
}

fn spec_float(key: &str, label: &str, default: f64, min: f64, step: f64) -> Value {
    json!({
        "key": key,
        "label": label,
        "kind": "float",
        "default": default,
        "minimum": min,
        "step": step,
    })
}

fn spec_float_when(
    key: &str,
    label: &str,
    default: f64,
    min: f64,
    step: f64,
    when_key: &str,
    when_value: &str,
) -> Value {
    let mut spec = spec_float(key, label, default, min, step);
    spec["visible_when"] = json!({ when_key: [when_value] });
    spec
}

fn spec_choice(key: &str, label: &str, default: &str, choices: Value) -> Value {
    json!({
        "key": key,
        "label": label,
        "kind": "choice",
        "default": default,
        "choices": choices,
    })
}

fn tune_sigma(
    root: &mut Elem,
    spots: &TuneSpots,
    params: &Value,
    session_label: &str,
) -> Result<(String, Vec<String>, Vec<String>, Vec<Vec<String>>, bool), String> {
    let tolerance = number(params, "sigma_tolerance_percent", 20.0).max(0.0);
    let headroom = number(params, "sigma_lower_headroom_percent", 1.0).max(0.0);
    let mut warnings = Vec::new();
    let mut rows = Vec::new();
    let mut updates = 0;
    for (index, device) in DEVICES.iter().enumerate() {
        let Some(chamber) = find_chamber(root, device) else {
            continue;
        };
        let sigmas = &spots.sigma[index];
        if sigmas.is_empty() {
            warnings.push(format!("No session sigma data for {device}."));
            continue;
        }
        let bands: Vec<usize> = chamber
            .children
            .iter()
            .enumerate()
            .filter(|(_, child)| is_sigma_band(child))
            .map(|(index, _)| index)
            .collect();
        for band_index in bands {
            let band = &chamber.children[band_index];
            let (min_e, max_e, k1, k2, k3, old) = match band_coeffs(band) {
                Some(values) => values,
                None => continue,
            };
            if k1.abs() > 1e-12 || k2.abs() > 1e-12 || k3.abs() > 1e-12 {
                continue;
            }
            let samples = paired(&spots.energy, sigmas, min_e, max_e);
            if samples.is_empty() {
                continue;
            }
            let new_k0 = band_k0(&samples, tolerance, headroom);
            if !new_k0.is_finite() {
                continue;
            }
            let (extreme, observed, kind) = excursion(&samples, new_k0, tolerance);
            chamber.children[band_index].set_attr("K0", &format_k0(new_k0));
            updates += 1;
            rows.push(vec![
                (*device).to_owned(),
                format!("{min_e}–{max_e}"),
                samples.len().to_string(),
                format_k0(old),
                format_k0(new_k0),
                format!("{:.4}", variance(&samples)),
                format!("{extreme:.2} {kind} ({observed:.3} mm)"),
            ]);
        }
    }
    if updates == 0 && warnings.is_empty() {
        warnings.push("No beam_sigma_conversions bands matched session energies.".into());
    }
    let summary = if updates == 0 {
        "No sigma bands were updated.".to_owned()
    } else {
        format!("Updated {updates} beam_sigma band(s) from {session_label}. Save the configuration to write devices.xml.")
    };
    Ok((
        summary,
        warnings,
        vec![
            "Device".into(),
            "Energy (MeV)".into(),
            "Spots".into(),
            "K0 Before".into(),
            "K0 After".into(),
            "Variance".into(),
            "Extreme".into(),
        ],
        rows,
        updates > 0,
    ))
}

fn tune_offset(
    root: &mut Elem,
    spots: &TuneSpots,
    params: &Value,
    session_label: &str,
) -> Result<(String, Vec<String>, Vec<String>, Vec<Vec<String>>, bool), String> {
    let mode = text(params, "optimize_method", "median");
    let mut warnings = Vec::new();
    let mut rows = Vec::new();
    let mut updates = 0;
    for (index, device) in DEVICES.iter().enumerate() {
        let Some(chamber) = find_chamber(root, device) else {
            warnings.push(format!("No zero_offset_at_iso_mm found for {device}."));
            continue;
        };
        let Some(offset_index) = chamber
            .children
            .iter()
            .position(|child| child.name == "zero_offset_at_iso_mm")
        else {
            warnings.push(format!("No zero_offset_at_iso_mm found for {device}."));
            continue;
        };
        let errors = axis_errors(spots, index);
        if errors.is_empty() {
            warnings.push(format!("No session position data for {device}."));
            continue;
        }
        let correction = reduce(&errors, &spots.weight, &mode);
        if !correction.is_finite() {
            warnings.push(format!("Could not compute offset correction for {device}."));
            continue;
        }
        let old = chamber.children[offset_index]
            .text
            .trim()
            .parse::<f64>()
            .unwrap_or(0.0);
        let new_offset = old - correction;
        chamber.children[offset_index].text = format_k0(new_offset);
        updates += 1;
        rows.push(vec![
            (*device).to_owned(),
            errors.len().to_string(),
            format_k0(old),
            format_k0(new_offset),
            format!("{correction:.4}"),
            format!("{:.4}", variance(&errors)),
        ]);
    }
    if updates == 0 && warnings.is_empty() {
        warnings.push("No zero_offset_at_iso_mm elements matched session data.".into());
    }
    let summary = if updates == 0 {
        "No position offsets were updated.".into()
    } else {
        format!("Updated {updates} zero_offset_at_iso_mm value(s) from {session_label}. Save the configuration to write devices.xml.")
    };
    Ok((
        summary,
        warnings,
        vec![
            "Device".into(),
            "Samples".into(),
            "Offset Before".into(),
            "Offset After".into(),
            "Correction".into(),
            "Variance".into(),
        ],
        rows,
        updates > 0,
    ))
}

fn tune_distance(
    root: &mut Elem,
    spots: &TuneSpots,
    session_label: &str,
) -> Result<(String, Vec<String>, Vec<String>, Vec<Vec<String>>, bool), String> {
    let mut warnings = Vec::new();
    let mut rows = Vec::new();
    let mut updates = 0;
    let mut largest: (String, f64, f64) = (String::new(), 0.0, 0.0);
    let mut best_removed = (String::new(), 0.0);
    for (index, device) in DEVICES.iter().enumerate() {
        let Some(chamber) = find_chamber(root, device) else {
            continue;
        };
        let Some(sdd_index) = chamber
            .children
            .iter()
            .position(|child| child.name == "source_to_device_distance_mm")
        else {
            warnings.push(format!(
                "No source_to_device_distance_mm and zero_offset_at_iso_mm pair for {device}."
            ));
            continue;
        };
        let Some(offset_index) = chamber
            .children
            .iter()
            .position(|child| child.name == "zero_offset_at_iso_mm")
        else {
            warnings.push(format!(
                "No source_to_device_distance_mm and zero_offset_at_iso_mm pair for {device}."
            ));
            continue;
        };
        let (plan, measured, weights) = axis_plan_measured(spots, index);
        if plan.len() < 8 {
            warnings.push(format!(
                "{device}: not enough spread to fit a distance (needs 8+ spots spanning 20 mm)."
            ));
            continue;
        }
        let Some(fit) = fit_scale(&plan, &measured, &weights) else {
            warnings.push(format!(
                "{device}: not enough spread to fit a distance (needs 8+ spots spanning 20 mm)."
            ));
            continue;
        };
        if (fit.gain - 1.0).abs() > 0.25 {
            warnings.push(format!(
                "{device}: fitted gain {:.4} is too far from 1 to be a distance calibration; left unchanged.",
                fit.gain
            ));
            continue;
        }
        let old_sdd = text_f64(&chamber.children[sdd_index].text);
        let old_offset = text_f64(&chamber.children[offset_index].text);
        let new_sdd = old_sdd * fit.gain;
        let new_offset = (old_offset - fit.bias) / fit.gain;
        let delta_mm = new_sdd - old_sdd;
        let delta_pct = if old_sdd == 0.0 {
            f64::NAN
        } else {
            100.0 * delta_mm / old_sdd
        };
        let significance = if fit.gain_stderr <= 0.0 {
            f64::INFINITY
        } else {
            (fit.gain - 1.0).abs() / fit.gain_stderr
        };
        if significance < 3.0 {
            warnings.push(format!(
                "{device}: distance change {delta_mm:+.2} mm is within the fit's own uncertainty."
            ));
        }
        if delta_pct.abs() > 1.0 {
            warnings.push(format!(
                "{device}: source_to_device_distance_mm moves {delta_mm:+.2} mm ({delta_pct:+.2}%). This is a surveyed distance."
            ));
        }
        chamber.children[sdd_index].text = format_distance(new_sdd);
        chamber.children[offset_index].text = format_k0(new_offset);
        updates += 1;
        if delta_pct.abs() > largest.2.abs() {
            largest = ((*device).to_owned(), delta_mm, delta_pct);
        }
        if fit.systematic > best_removed.1 {
            best_removed = ((*device).to_owned(), fit.systematic);
        }
        rows.push(vec![
            (*device).to_owned(),
            fit.n_samples.to_string(),
            format_distance(old_sdd),
            format_distance(new_sdd),
            format_k0(old_offset),
            format_k0(new_offset),
            format!("{:.4}", fit.gain),
            format!("{:.3}", fit.systematic),
        ]);
    }
    if updates > 0 {
        warnings.push(
            "Changing source_to_device_distance_mm rescales isocenter sigma by the same factor. Re-run Sigma Tuning afterwards."
                .into(),
        );
    }
    if updates == 0 && warnings.is_empty() {
        warnings.push("No ion chamber matched the session data.".into());
    }
    let summary = if updates == 0 {
        "No IC distances were updated.".into()
    } else {
        format!(
            "Updated {updates} IC distance(s) and zero offset(s) from {session_label}; largest distance change {:+.2} mm ({:+.2}%) on {}, removing up to {:.3} mm of position error at the field edge ({}). Save the configuration to write devices.xml.",
            largest.1, largest.2, largest.0, best_removed.1, best_removed.0
        )
    };
    Ok((
        summary,
        warnings,
        vec![
            "Device".into(),
            "Samples".into(),
            "SDD Before".into(),
            "SDD After".into(),
            "Offset Before".into(),
            "Offset After".into(),
            "Gain".into(),
            "Systematic (mm)".into(),
        ],
        rows,
        updates > 0,
    ))
}

fn tune_kmu(
    root: &mut Elem,
    spots: &TuneSpots,
    params: &Value,
    session_label: &str,
) -> Result<(String, Vec<String>, Vec<String>, Vec<Vec<String>>, bool), String> {
    let primary = match text(params, "primary_ic", "ic1").as_str() {
        "ic2" => 1,
        "ic3" => 2,
        _ => 0,
    };
    let mode = text(params, "primary_mode", "unchanged");
    let known = number(params, "known_mu", f64::NAN);
    let percent = number(params, "percent", 0.0);
    let mut warnings = Vec::new();
    let primary_mu = &spots.mu[primary];
    if primary_mu
        .iter()
        .filter(|value| **value > 0.0 && value.is_finite())
        .count()
        < 8
    {
        warnings.push(format!(
            "Primary {} has too few valid spots; need 8.",
            ["IC1", "IC2", "IC3"][primary]
        ));
    }
    let sum_pri = sum_positive(primary_mu);
    if !(sum_pri.is_finite() && sum_pri > 0.0) {
        warnings.push("Primary IC has no positive MU in the selected sessions.".into());
        return empty_kmu(warnings);
    }
    let target = match mode.as_str() {
        "known_mu" => {
            if !(known.is_finite() && known > 0.0) {
                warnings.push("Enter a known MU at isocenter greater than 0.".into());
                return empty_kmu(warnings);
            }
            let mismatch = pct(known, sum_pri).abs();
            if mismatch > 20.0 {
                warnings.push(format!(
                    "Known MU {known} is {mismatch:.1}% from primary {sum_pri:.4} MU. Check the isocenter reading."
                ));
            }
            known
        }
        "percent" => {
            if percent <= -100.0 {
                warnings.push("Percent adjustment must be greater than -100.".into());
                return empty_kmu(warnings);
            }
            sum_pri * (1.0 + percent / 100.0)
        }
        _ => sum_pri,
    };
    let mut scales = [f64::NAN; 3];
    let mut write = [false; 3];
    for family in 0..3 {
        let sum_ic = sum_positive(&spots.mu[family]);
        let n = spots.mu[family]
            .iter()
            .filter(|value| value.is_finite() && **value > 0.0)
            .count();
        let scale = if target > 0.0 {
            sum_ic / target
        } else {
            f64::NAN
        };
        scales[family] = scale;
        let skip_primary = family == primary && mode == "unchanged";
        if n < 8 {
            warnings.push(format!(
                "{}: {n} valid spots; need 8 to write K_MU.",
                ["IC1", "IC2", "IC3"][family]
            ));
        } else if !(scale.is_finite() && scale > 0.0) {
            warnings.push(format!(
                "{}: could not compute a K_MU scale.",
                ["IC1", "IC2", "IC3"][family]
            ));
        } else if (scale - 1.0).abs() * 100.0 > 5.0 && !skip_primary {
            warnings.push(format!(
                "{}: K_MU scale {scale:.4} ({:+.2}%). Review before applying.",
                ["IC1", "IC2", "IC3"][family],
                (scale - 1.0) * 100.0
            ));
        }
        write[family] = !skip_primary && n >= 8 && scale.is_finite() && scale > 0.0;
    }
    let mut rows = Vec::new();
    let mut updates = 0;
    let mut found = Vec::new();
    collect_kmu(root, &mut found);
    for (path, attr, old, family) in found {
        let scale = scales[family];
        let skip_primary = family == primary && mode == "unchanged";
        let (new_kmu, applied) = if skip_primary || !scale.is_finite() {
            (old, 1.0)
        } else {
            (old * scale, scale)
        };
        if write[family] {
            if let Some(elem) = root.at_mut(&path) {
                elem.set_attr(&attr, &format_k0(new_kmu));
                updates += 1;
            }
        }
        let device = device_on_path(root, &path);
        rows.push(vec![
            device,
            if family == primary {
                "primary".into()
            } else {
                "secondary".into()
            },
            format_k0(old),
            format_k0(new_kmu),
            format!("{applied:.4}"),
            if write[family] { "yes" } else { "no" }.into(),
        ]);
    }
    if rows.is_empty() && warnings.is_empty() {
        warnings.push("No ion chamber MU gain_conversion matched the session data.".into());
    }
    let summary = if updates == 0 {
        "No K_MU values were updated.".into()
    } else {
        format!("Updated {updates} K_MU value(s) from {session_label}. Save the configuration to write devices.xml.")
    };
    Ok((
        summary,
        warnings,
        vec![
            "Device".into(),
            "Role".into(),
            "K_MU Before".into(),
            "K_MU After".into(),
            "Scale".into(),
            "Write".into(),
        ],
        rows,
        updates > 0,
    ))
}

fn empty_kmu(
    warnings: Vec<String>,
) -> Result<(String, Vec<String>, Vec<String>, Vec<Vec<String>>, bool), String> {
    Ok((
        "No K_MU values were updated.".into(),
        warnings,
        vec![
            "Device".into(),
            "Role".into(),
            "K_MU Before".into(),
            "K_MU After".into(),
            "Scale".into(),
            "Write".into(),
        ],
        Vec::new(),
        false,
    ))
}

fn collect_kmu(elem: &Elem, found: &mut Vec<(Vec<usize>, String, f64, usize)>) {
    collect_kmu_at(elem, &[], found);
}

fn collect_kmu_at(elem: &Elem, path: &[usize], found: &mut Vec<(Vec<usize>, String, f64, usize)>) {
    if elem.name == "ion_chamber" {
        if let Some(name) = elem.child("device").and_then(|device| device.attr("name")) {
            if let Some(family) = family_of(name) {
                if let Some((relative, attr, old)) = mu_gain(elem) {
                    let mut gain_path = path.to_vec();
                    gain_path.extend(relative);
                    found.push((gain_path, attr, old, family));
                }
            }
        }
    }
    for (index, child) in elem.children.iter().enumerate() {
        let mut next = path.to_vec();
        next.push(index);
        collect_kmu_at(child, &next, found);
    }
}

fn mu_gain(chamber: &Elem) -> Option<(Vec<usize>, String, f64)> {
    fn walk(elem: &Elem, path: &[usize]) -> Option<(Vec<usize>, String, f64)> {
        if elem.name == "gain_conversion"
            && elem
                .attr("in_units")
                .is_some_and(|units| units.eq_ignore_ascii_case("MU"))
        {
            for key in ["K_MU", "k_mu"] {
                if let Some(raw) = elem.attr(key) {
                    if let Ok(old) = raw.trim().parse::<f64>() {
                        if old.is_finite() && old != 0.0 {
                            return Some((path.to_vec(), key.to_owned(), old));
                        }
                    }
                }
            }
        }
        for (index, child) in elem.children.iter().enumerate() {
            let mut next = path.to_vec();
            next.push(index);
            if let Some(found) = walk(child, &next) {
                return Some(found);
            }
        }
        None
    }
    for (index, child) in chamber.children.iter().enumerate() {
        if let Some(found) = walk(child, &[index]) {
            return Some(found);
        }
    }
    None
}

fn family_of(name: &str) -> Option<usize> {
    let upper = name.trim().to_ascii_uppercase();
    if upper.starts_with("IC_1") {
        Some(0)
    } else if upper.starts_with("IC_2") {
        Some(1)
    } else if upper.starts_with("IC_3") {
        Some(2)
    } else {
        None
    }
}

fn device_on_path(root: &Elem, path: &[usize]) -> String {
    let mut node = root;
    let mut last = String::new();
    for index in path {
        let Some(child) = node.children.get(*index) else {
            break;
        };
        node = child;
        if node.name == "ion_chamber" {
            if let Some(name) = node.child("device").and_then(|device| device.attr("name")) {
                last = name.to_owned();
            }
        }
    }
    if last.is_empty() {
        "chamber".into()
    } else {
        last
    }
}

fn find_chamber<'a>(root: &'a mut Elem, device: &str) -> Option<&'a mut Elem> {
    fn walk<'a>(elem: &'a mut Elem, device: &str) -> Option<&'a mut Elem> {
        if elem.name == "ion_chamber"
            && elem.child("device").and_then(|child| child.attr("name")) == Some(device)
        {
            return Some(elem);
        }
        for child in &mut elem.children {
            if let Some(found) = walk(child, device) {
                return Some(found);
            }
        }
        None
    }
    walk(root, device)
}

fn is_sigma_band(elem: &Elem) -> bool {
    elem.name == "beam_sigma_conversions"
        && elem
            .attr("in_units")
            .is_some_and(|units| units.eq_ignore_ascii_case("MEV"))
        && elem
            .attr("out_units")
            .is_some_and(|units| units.eq_ignore_ascii_case("mm"))
}

fn band_coeffs(elem: &Elem) -> Option<(f64, f64, f64, f64, f64, f64)> {
    let min_e = elem.attr("min_energy")?.parse::<f64>().ok()?;
    let max_e = elem.attr("max_energy")?.parse::<f64>().ok()?;
    if !min_e.is_finite() || !max_e.is_finite() {
        return None;
    }
    let coeff = |name: &str| {
        elem.attr(name)
            .and_then(|value| value.parse().ok())
            .unwrap_or(0.0)
    };
    Some((
        min_e,
        max_e,
        coeff("K1"),
        coeff("K2"),
        coeff("K3"),
        coeff("K0"),
    ))
}

fn paired(energy: &[f64], values: &[f64], min_e: f64, max_e: f64) -> Vec<f64> {
    energy
        .iter()
        .zip(values)
        .filter(|(energy, value)| {
            energy.is_finite() && value.is_finite() && **energy >= min_e && **energy <= max_e
        })
        .map(|(_, value)| *value)
        .collect()
}

fn band_k0(sigmas: &[f64], tolerance: f64, headroom: f64) -> f64 {
    let min = sigmas
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .fold(f64::INFINITY, f64::min);
    if !min.is_finite() {
        return f64::NAN;
    }
    let adjusted = min * (1.0 + headroom / 100.0);
    let fraction = tolerance / 100.0;
    if fraction <= 0.0 || fraction >= 1.0 {
        adjusted
    } else {
        adjusted / (1.0 - fraction)
    }
}

fn excursion(sigmas: &[f64], k0: f64, tolerance: f64) -> (f64, f64, &'static str) {
    if !k0.is_finite() || k0 == 0.0 {
        return (0.0, 0.0, "inside");
    }
    let lower = (k0 * (1.0 - tolerance / 100.0)).max(0.0);
    let upper = (k0 * (1.0 + tolerance / 100.0)).max(0.0);
    let mut worst = 0.0;
    let mut observed = 0.0;
    let mut kind = "inside";
    for sigma in sigmas {
        if !sigma.is_finite() {
            continue;
        }
        let (pct, which) = if *sigma < lower {
            (100.0 * (lower - sigma) / k0.abs(), "below")
        } else if *sigma > upper {
            (100.0 * (sigma - upper) / k0.abs(), "above")
        } else {
            continue;
        };
        if pct > worst {
            worst = pct;
            observed = *sigma;
            kind = which;
        }
    }
    (worst, observed, kind)
}

fn axis_errors(spots: &TuneSpots, index: usize) -> Vec<f64> {
    if !spots.error[index].is_empty() {
        return spots.error[index]
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .collect();
    }
    let plan = if index.is_multiple_of(2) {
        &spots.plan_x
    } else {
        &spots.plan_y
    };
    spots.measured[index]
        .iter()
        .zip(plan)
        .filter(|(measured, plan)| measured.is_finite() && plan.is_finite())
        .map(|(measured, plan)| measured - plan)
        .collect()
}

fn axis_plan_measured(spots: &TuneSpots, index: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let plan = if index.is_multiple_of(2) {
        &spots.plan_x
    } else {
        &spots.plan_y
    };
    let mut plans = Vec::new();
    let mut measured = Vec::new();
    let mut weights = Vec::new();
    for (offset, (plan, value)) in plan.iter().zip(&spots.measured[index]).enumerate() {
        if plan.is_finite() && value.is_finite() {
            plans.push(*plan);
            measured.push(*value);
            let weight = spots.weight.get(offset).copied().unwrap_or(1.0);
            weights.push(if weight.is_finite() && weight > 0.0 {
                weight
            } else {
                1.0
            });
        }
    }
    (plans, measured, weights)
}

fn reduce(values: &[f64], weights: &[f64], mode: &str) -> f64 {
    let finite: Vec<f64> = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    if finite.is_empty() {
        return f64::NAN;
    }
    match mode {
        "min_max_midpoint" => {
            let min = finite.iter().copied().fold(f64::INFINITY, f64::min);
            let max = finite.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            (min + max) / 2.0
        }
        "weighted_average" => {
            let mut num = 0.0;
            let mut den = 0.0;
            for (value, weight) in values.iter().zip(weights) {
                if value.is_finite() && weight.is_finite() && *weight > 0.0 {
                    num += value * weight;
                    den += weight;
                }
            }
            if den > 0.0 {
                num / den
            } else {
                finite.iter().sum::<f64>() / finite.len() as f64
            }
        }
        _ => median(&finite),
    }
}

struct ScaleFit {
    gain: f64,
    bias: f64,
    gain_stderr: f64,
    n_samples: usize,
    systematic: f64,
}

fn fit_scale(plan: &[f64], measured: &[f64], weights: &[f64]) -> Option<ScaleFit> {
    if plan.len() < 8 || plan.len() != measured.len() {
        return None;
    }
    let span = plan.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - plan.iter().copied().fold(f64::INFINITY, f64::min);
    if span < 20.0 {
        return None;
    }
    let sample_weights: Vec<f64> = if weights.len() == plan.len() {
        weights
            .iter()
            .map(|weight| {
                if weight.is_finite() && *weight > 0.0 {
                    *weight
                } else {
                    1.0
                }
            })
            .collect()
    } else {
        vec![1.0; plan.len()]
    };
    let mut fit = line_fit(plan, measured, &sample_weights)?;
    let mut n_rejected = 0;
    let residual: Vec<f64> = measured
        .iter()
        .zip(plan)
        .map(|(measured, plan)| measured - (fit.0 * plan + fit.1))
        .collect();
    if let Some(keep) = outlier_mask(&residual) {
        let kept_plan: Vec<f64> = plan
            .iter()
            .zip(&keep)
            .filter(|(_, keep)| **keep)
            .map(|(value, _)| *value)
            .collect();
        let kept_measured: Vec<f64> = measured
            .iter()
            .zip(&keep)
            .filter(|(_, keep)| **keep)
            .map(|(value, _)| *value)
            .collect();
        let kept_weights: Vec<f64> = sample_weights
            .iter()
            .zip(&keep)
            .filter(|(_, keep)| **keep)
            .map(|(value, _)| *value)
            .collect();
        let kept_span = kept_plan.iter().copied().fold(f64::NEG_INFINITY, f64::max)
            - kept_plan.iter().copied().fold(f64::INFINITY, f64::min);
        if kept_plan.len() >= 8 && kept_span >= 20.0 {
            if let Some(refit) = line_fit(&kept_plan, &kept_measured, &kept_weights) {
                fit = refit;
                n_rejected = keep.iter().filter(|keep| !**keep).count();
            }
        }
    }
    let (gain, bias, stderr) = fit;
    let edges = [
        (gain - 1.0) * plan.iter().copied().fold(f64::INFINITY, f64::min) + bias,
        (gain - 1.0) * plan.iter().copied().fold(f64::NEG_INFINITY, f64::max) + bias,
    ];
    Some(ScaleFit {
        gain,
        bias,
        gain_stderr: stderr,
        n_samples: plan.len() - n_rejected,
        systematic: edges[0].abs().max(edges[1].abs()),
    })
}

fn line_fit(plan: &[f64], measured: &[f64], weights: &[f64]) -> Option<(f64, f64, f64)> {
    let total: f64 = weights.iter().sum();
    if total <= 0.0 || plan.len() < 3 {
        return None;
    }
    let plan_mean = weighted_mean(plan, weights, total);
    let measured_mean = weighted_mean(measured, weights, total);
    let mut s_pp = 0.0;
    let mut s_pm = 0.0;
    for ((plan, measured), weight) in plan.iter().zip(measured).zip(weights) {
        let centered = plan - plan_mean;
        s_pp += weight * centered * centered;
        s_pm += weight * centered * (measured - measured_mean);
    }
    if s_pp <= 0.0 {
        return None;
    }
    let gain = s_pm / s_pp;
    let bias = measured_mean - gain * plan_mean;
    if !gain.is_finite() || !bias.is_finite() || gain.abs() < 1e-9 {
        return None;
    }
    let mut residual_sum = 0.0;
    for ((plan, measured), weight) in plan.iter().zip(measured).zip(weights) {
        let residual = measured - (gain * plan + bias);
        residual_sum += weight * residual * residual;
    }
    let variance = residual_sum / (plan.len() - 2) as f64;
    let stderr = (variance.max(0.0) / s_pp).sqrt();
    Some((gain, bias, stderr))
}

fn weighted_mean(values: &[f64], weights: &[f64], total: f64) -> f64 {
    values
        .iter()
        .zip(weights)
        .map(|(value, weight)| value * weight)
        .sum::<f64>()
        / total
}

fn outlier_mask(residual: &[f64]) -> Option<Vec<bool>> {
    let centre = median(residual);
    let abs: Vec<f64> = residual
        .iter()
        .map(|value| (value - centre).abs())
        .collect();
    let mad = median(&abs);
    if mad <= 0.0 {
        return None;
    }
    let sigma = 1.4826 * mad;
    let keep: Vec<bool> = residual
        .iter()
        .map(|value| (value - centre).abs() <= 5.0 * sigma)
        .collect();
    if keep.iter().all(|keep| *keep) {
        None
    } else {
        Some(keep)
    }
}

fn median(values: &[f64]) -> f64 {
    let mut sorted: Vec<f64> = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    if sorted.is_empty() {
        return f64::NAN;
    }
    sorted.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        sorted[mid]
    } else {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    }
}

fn variance(values: &[f64]) -> f64 {
    let finite: Vec<f64> = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    if finite.len() < 2 {
        return 0.0;
    }
    let mean = finite.iter().sum::<f64>() / finite.len() as f64;
    finite
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / (finite.len() - 1) as f64
}

fn sum_positive(values: &[f64]) -> f64 {
    values
        .iter()
        .filter(|value| value.is_finite() && **value > 0.0)
        .sum()
}

fn pct(value: f64, reference: f64) -> f64 {
    if !value.is_finite() || !reference.is_finite() || reference.abs() < 1e-15 {
        f64::NAN
    } else {
        100.0 * (value - reference) / reference
    }
}

fn text_f64(text: &str) -> f64 {
    text.trim().parse().unwrap_or(f64::NAN)
}

fn format_k0(value: f64) -> String {
    if !value.is_finite() {
        return "0".into();
    }
    if value.abs() >= 1000.0 || (value.abs() > 0.0 && value.abs() < 1e-4) {
        return format!("{value:.6E}");
    }
    let rounded = (value * 1000.0).round() / 1000.0;
    if (rounded - value).abs() < 1e-9 {
        let text = format!("{rounded:.3}");
        return text.trim_end_matches('0').trim_end_matches('.').to_owned();
    }
    format!("{value:.6}")
}

fn format_distance(value: f64) -> String {
    let text = format!("{value:.4}");
    let trimmed = text.trim_end_matches('0').trim_end_matches('.');
    if trimmed.is_empty() {
        "0".into()
    } else {
        trimmed.to_owned()
    }
}

fn number(params: &Value, key: &str, fallback: f64) -> f64 {
    params
        .get(key)
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite())
        .unwrap_or(fallback)
}

fn text(params: &Value, key: &str, fallback: &str) -> String {
    params
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or(fallback)
        .to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigma_k0_sits_above_the_smallest_spot() {
        let xml = r#"
            <devices><ion_chamber>
              <device name="IC_1_X"/>
              <beam_sigma_conversions in_units="MEV" out_units="mm" min_energy="70" max_energy="250" K0="1" K1="0" K2="0" K3="0"/>
            </ion_chamber></devices>
        "#;
        let spots = TuneSpots {
            energy: vec![100.0, 100.0, 100.0],
            weight: vec![],
            plan_x: vec![],
            plan_y: vec![],
            measured: [vec![], vec![], vec![], vec![]],
            error: [vec![], vec![], vec![], vec![]],
            sigma: [vec![4.0, 5.0, 6.0], vec![], vec![], vec![]],
            mu: [vec![], vec![], vec![]],
            charge: vec![],
        };
        let result = run_tune("sigma_tuning", xml, &spots, &json!({}), "one session").unwrap();
        assert_eq!(result["changed"], true);
        let xml = result["xml"].as_str().unwrap();
        assert!(xml.contains("K0=\""));
        assert!(!xml.contains("K0=\"1\""));
    }

    #[test]
    fn an_even_offset_moves_by_the_median_error() {
        let xml = r#"
            <devices><ion_chamber>
              <device name="IC_1_Y"/>
              <zero_offset_at_iso_mm>1.0</zero_offset_at_iso_mm>
            </ion_chamber></devices>
        "#;
        let spots = TuneSpots {
            energy: vec![100.0, 100.0, 100.0],
            weight: vec![],
            plan_x: vec![],
            plan_y: vec![0.0, 0.0, 0.0],
            measured: [vec![], vec![0.2, 0.4, 0.6], vec![], vec![]],
            error: [vec![], vec![], vec![], vec![]],
            sigma: [vec![], vec![], vec![], vec![]],
            mu: [vec![], vec![], vec![]],
            charge: vec![],
        };
        let result = run_tune(
            "position_offset_tuning",
            xml,
            &spots,
            &json!({}),
            "one session",
        )
        .unwrap();
        assert!(result["xml"].as_str().unwrap().contains("0.6"));
    }
}
