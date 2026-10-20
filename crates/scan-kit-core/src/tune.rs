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
                "description": "Fit spot-size factors to session data",
                "sessions": true,
                "params": [
                    hint(
                        spec_float("sigma_tolerance_percent", "Sigma Tolerance", 20.0, 0.0, 0.1),
                        "%",
                        "Symmetric ±% band around each proposed K0."
                    ),
                    hint(
                        spec_float("sigma_lower_headroom_percent", "Lower Headroom", 1.0, 0.0, 0.1),
                        "%",
                        "Extra margin below the smallest observed sigma, covering hypothetical smaller spots the sessions did not sample."
                    )
                ]
            },
            {
                "id": "position_offset_tuning",
                "name": "Position Offset Tuning",
                "description": "Correct chamber zero offsets from position error",
                "sessions": true,
                "params": [
                    spec_choice("optimize_method", "Optimize Using", "median", json!([
                        choice("median", "Median", "Robust median of position errors across all samples."),
                        choice("weighted_average", "Weighted", "Mean weighted by charge per spot. Falls back to an unweighted mean."),
                        choice("min_max_midpoint", "Midpoint", "Midpoint between the smallest and largest observed error.")
                    ])),
                    spec_choice("data_source", "Data Source", "spot", json!([
                        choice("spot", "Spot", "One sample per delivered spot from spot_data.csv."),
                        choice("timeslice", "Timeslice", "Beam-on timeslice samples from timeslice_data_device_units.csv.")
                    ]))
                ]
            },
            {
                "id": "ic_distance_tuning",
                "name": "IC Distance Tuning",
                "description": "Fit source distance per chamber — re-run Sigma Tuning after",
                "sessions": true,
                "params": []
            },
            {
                "id": "kmu_tuning",
                "name": "Dose Calibration",
                "description": "Match secondary K_MU to the primary IC",
                "sessions": true,
                "params": [
                    spec_choice("primary_ic", "Primary IC", "ic1", json!([
                        choice("ic1", "IC1", "Beam-terminator chamber. Left unchanged unless you rescale it below."),
                        choice("ic2", "IC2", "Use IC2 as the primary reference chamber."),
                        choice("ic3", "IC3", "Use IC3 as the primary reference chamber.")
                    ])),
                    spec_choice("primary_mode", "Primary K_MU", "unchanged", json!([
                        choice("unchanged", "Unchanged", "Do not write primary K_MU. Secondaries are scaled to agree with it."),
                        choice("known_mu", "Known MU", "Independent Faraday or iso-chamber total for the selected sessions combined. Changing primary K_MU changes future delivered charge for the same CHARGE_REQ."),
                        choice("percent", "Percent", "Positive means more reported MU for the same prescription, so K_MU decreases. +2% scales by 1/1.02.")
                    ])),
                    hint(
                        spec_float_when("known_mu", "Known MU", 0.0, 0.0, 0.1, "primary_mode", "known_mu"),
                        "MU",
                        "Total MU at isocenter for the selected sessions combined. Defaults to the primary IC total."
                    ),
                    hint(
                        spec_float_when("percent", "Percent", 0.0, -99.0, 0.1, "primary_mode", "percent"),
                        "%",
                        "Positive means more reported MU for the same prescription, so K_MU decreases. +2% scales by 1/1.02."
                    )
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
    let mut body = json!({
        "xml": xml,
        "form": form,
        "summary": summary,
        "warnings": warnings,
        "columns": columns,
        "rows": rows,
        "changed": changed,
    });
    if workflow == "kmu_tuning" {
        body["mu_totals"] = json!(spots
            .mu
            .iter()
            .map(|mu| sum_positive(mu))
            .collect::<Vec<_>>());
    }
    Ok(body)
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

fn choice(value: &str, label: &str, tooltip: &str) -> Value {
    json!({"value": value, "label": label, "tooltip": tooltip})
}

fn hint(mut spec: Value, suffix: &str, tooltip: &str) -> Value {
    if !suffix.is_empty() {
        spec["suffix"] = json!(suffix);
    }
    spec["tooltip"] = json!(tooltip);
    spec
}

/// Preview cell. `tone` is the verdict the grid color-codes: "good" improves or
/// holds tolerance, "warn" needs review, "bad" is a downgrade, "change" marks a
/// value that will be rewritten.
fn cell(text: impl Into<String>, tone: Option<&'static str>) -> Value {
    match tone {
        Some(tone) => json!({ "text": text.into(), "tone": tone }),
        None => json!({ "text": text.into() }),
    }
}

fn tune_sigma(
    root: &mut Elem,
    spots: &TuneSpots,
    params: &Value,
    session_label: &str,
) -> Result<(String, Vec<String>, Vec<String>, Vec<Vec<Value>>, bool), String> {
    let tolerance = number(params, "sigma_tolerance_percent", 20.0).max(0.0);
    let headroom = number(params, "sigma_lower_headroom_percent", 1.0).max(0.0);
    let mut warnings = Vec::new();
    let mut rows = Vec::new();
    let mut updates = 0;
    let mut processed = 0;
    let mut min_inside = f64::INFINITY;
    for (ic, axis_indexes) in [("IC_1", [0_usize, 1_usize]), ("IC_2", [2_usize, 3_usize])] {
        // X and Y bands for the same chamber share a row; each side is
        // collected separately, then merged on (min_e, max_e).
        let mut sides: [Vec<((f64, f64), SigmaBand)>; 2] = [Vec::new(), Vec::new()];
        for (side, index) in axis_indexes.into_iter().enumerate() {
            let device = DEVICES[index];
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
                let inside_after = inside_fraction(&samples, new_k0, tolerance);
                let inside_before = inside_fraction(&samples, old, tolerance);
                if inside_after < min_inside {
                    min_inside = inside_after;
                }
                let before = format_k0(old);
                let after = format_k0(new_k0);
                let differs = after != before;
                processed += 1;
                if differs {
                    chamber.children[band_index].set_attr("K0", &after);
                    updates += 1;
                }
                // inside_fraction() returns NaN when there is no valid old K0
                // baseline; NaN comparisons are all false, so the After cell
                // lands on the warn/neutral branches instead of good/bad.
                // NaN comparisons are false, so a missing old-K0 baseline
                // stays neutral.
                let before_tone = if inside_before < 0.90 {
                    "bad"
                } else if inside_before < 1.0 {
                    "warn"
                } else {
                    ""
                };
                let after_tone = if inside_after >= 1.0 {
                    "good"
                } else if inside_after < inside_before {
                    "bad"
                } else if inside_after > inside_before {
                    "warn"
                } else {
                    ""
                };
                let after_text = if differs && old.is_finite() && old != 0.0 {
                    format!("{after} ({:+.1}%)", 100.0 * (new_k0 - old) / old.abs())
                } else {
                    after.clone()
                };
                sides[side].push((
                    (min_e, max_e),
                    SigmaBand {
                        samples: samples.len(),
                        before,
                        after_text,
                        differs,
                        variance: variance(&samples),
                        inside_before,
                        inside_after,
                        before_tone,
                        after_tone,
                    },
                ));
            }
        }
        let mut keys: Vec<(f64, f64)> = Vec::new();
        for side in &sides {
            for (key, _) in side {
                if !keys.contains(key) {
                    keys.push(*key);
                }
            }
        }
        for (min_e, max_e) in keys {
            let x = band_at(&sides[0], min_e, max_e);
            let y = band_at(&sides[1], min_e, max_e);
            rows.push(sigma_row(ic, min_e, max_e, x, y));
        }
    }
    if processed == 0 && warnings.is_empty() {
        warnings.push("No beam_sigma_conversions bands matched session energies.".into());
    }
    let summary = if updates == 0 {
        "No sigma bands were updated.".to_owned()
    } else {
        format!(
            "{updates} energy band(s) will be updated from {session_label}. Min inside tolerance: {:.0}%.",
            min_inside * 100.0
        )
    };
    Ok((
        summary,
        warnings,
        vec![
            "IC".into(),
            "Energy (MeV)".into(),
            "Spots".into(),
            "X K0 Before".into(),
            "X K0 After".into(),
            "X Coverage Before (%)".into(),
            "X Coverage After (%)".into(),
            "Y K0 Before".into(),
            "Y K0 After".into(),
            "Y Coverage Before (%)".into(),
            "Y Coverage After (%)".into(),
            "Variance (mm²)".into(),
        ],
        rows,
        updates > 0,
    ))
}

struct SigmaBand {
    samples: usize,
    before: String,
    after_text: String,
    differs: bool,
    variance: f64,
    inside_before: f64,
    inside_after: f64,
    before_tone: &'static str,
    after_tone: &'static str,
}

fn band_at(side: &[((f64, f64), SigmaBand)], min_e: f64, max_e: f64) -> Option<&SigmaBand> {
    side.iter()
        .find(|(key, _)| *key == (min_e, max_e))
        .map(|(_, band)| band)
}

// One preview row per IC and energy band: X and Y columns side by side, "—"
// for an axis with no matching band.
fn sigma_row(
    ic: &str,
    min_e: f64,
    max_e: f64,
    x: Option<&SigmaBand>,
    y: Option<&SigmaBand>,
) -> Vec<Value> {
    let spots = match (x, y) {
        (Some(x), Some(y)) => format!("{} / {}", x.samples, y.samples),
        (Some(x), None) => format!("{} / —", x.samples),
        (None, Some(y)) => format!("— / {}", y.samples),
        (None, None) => "—".into(),
    };
    let spread = match (x, y) {
        (Some(x), Some(y)) => format!("{:.4} / {:.4}", x.variance, y.variance),
        (Some(x), None) => format!("{:.4} / —", x.variance),
        (None, Some(y)) => format!("— / {:.4}", y.variance),
        (None, None) => "—".into(),
    };
    let inside_text = |fraction: f64| -> String {
        if fraction.is_finite() {
            format!("{:.1}%", fraction * 100.0)
        } else {
            "—".into()
        }
    };
    let side = |band: Option<&SigmaBand>| -> Vec<Value> {
        match band {
            Some(band) => vec![
                cell(band.before.clone(), None),
                cell(
                    band.after_text.clone(),
                    if band.differs { Some("change") } else { None },
                ),
                cell(
                    inside_text(band.inside_before),
                    if band.before_tone.is_empty() {
                        None
                    } else {
                        Some(band.before_tone)
                    },
                ),
                cell(
                    inside_text(band.inside_after),
                    if band.after_tone.is_empty() {
                        None
                    } else {
                        Some(band.after_tone)
                    },
                ),
            ],
            None => vec![cell("—", None); 4],
        }
    };
    let mut row = vec![
        cell(ic.to_owned(), None),
        cell(format!("{min_e}–{max_e}"), None),
        cell(spots, None),
    ];
    row.extend(side(x));
    row.extend(side(y));
    row.push(cell(spread, None));
    row
}

fn tune_offset(
    root: &mut Elem,
    spots: &TuneSpots,
    params: &Value,
    session_label: &str,
) -> Result<(String, Vec<String>, Vec<String>, Vec<Vec<Value>>, bool), String> {
    let mode = text(params, "optimize_method", "median");
    let mut warnings = Vec::new();
    let mut rows = Vec::new();
    let mut updates = 0;
    let mut processed = 0;
    let mut max_correction = 0.0_f64;
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
        let samples = errors.iter().filter(|value| value.is_finite()).count();
        if samples == 0 {
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
        if correction.abs() > max_correction {
            max_correction = correction.abs();
        }
        let before = format_k0(old);
        let after = format_k0(new_offset);
        let differs = after != before;
        processed += 1;
        if differs {
            chamber.children[offset_index].text = after.clone();
            updates += 1;
        }
        let correction_tone = if correction.abs() > 1.0 {
            Some("warn")
        } else if correction.abs() >= 0.05 {
            Some("good")
        } else {
            None
        };
        rows.push(vec![
            cell((*device).to_owned(), None),
            cell(samples.to_string(), None),
            cell(before, None),
            cell(after, if differs { Some("change") } else { None }),
            cell(format!("{correction:.4}"), correction_tone),
            cell(format!("{:.4}", variance(&errors)), None),
        ]);
    }
    if processed == 0 && warnings.is_empty() {
        warnings.push("No zero_offset_at_iso_mm elements matched session data.".into());
    }
    let summary = if updates == 0 {
        "No position offsets were updated.".into()
    } else {
        format!(
            "{updates} zero offset(s) will be updated from {session_label}. Max correction: {max_correction:.3} mm."
        )
    };
    Ok((
        summary,
        warnings,
        vec![
            "Device".into(),
            "Samples".into(),
            "Offset Before (mm)".into(),
            "Offset After (mm)".into(),
            "Correction (mm)".into(),
            "Variance (mm²)".into(),
        ],
        rows,
        updates > 0,
    ))
}

fn tune_distance(
    root: &mut Elem,
    spots: &TuneSpots,
    session_label: &str,
) -> Result<(String, Vec<String>, Vec<String>, Vec<Vec<Value>>, bool), String> {
    let mut warnings = Vec::new();
    let mut rows = Vec::new();
    let mut updates = 0;
    let mut processed = 0;
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
        if significance.is_nan() || significance < 3.0 {
            warnings.push(format!(
                "{device}: distance change {delta_mm:+.2} mm is within the fit's own uncertainty."
            ));
        }
        if delta_pct.abs() > 1.0 {
            warnings.push(format!(
                "{device}: source_to_device_distance_mm moves {delta_mm:+.2} mm ({delta_pct:+.2}%). This is a surveyed distance."
            ));
        }
        let sdd_before = format_distance(old_sdd);
        let sdd_after = format_distance(new_sdd);
        let off_before = format_k0(old_offset);
        let off_after = format_k0(new_offset);
        let off_text =
            if off_after != off_before && old_offset.is_finite() && new_offset.is_finite() {
                format!("{off_after} ({:+.3})", new_offset - old_offset)
            } else {
                off_after.clone()
            };
        processed += 1;
        if sdd_after != sdd_before || off_after != off_before {
            chamber.children[sdd_index].text = sdd_after.clone();
            chamber.children[offset_index].text = off_after.clone();
            updates += 1;
        }
        if delta_pct.abs() > largest.2.abs() {
            largest = ((*device).to_owned(), delta_mm, delta_pct);
        }
        if fit.systematic > best_removed.1 {
            best_removed = ((*device).to_owned(), fit.systematic);
        }
        let sdd_tone = if delta_pct.abs() > 1.0 {
            Some("warn")
        } else if sdd_after != sdd_before {
            Some("change")
        } else {
            None
        };
        let off_tone = if off_after != off_before {
            Some("change")
        } else {
            None
        };
        let gain_tone = if significance.is_nan() || significance < 3.0 {
            Some("warn")
        } else {
            Some("good")
        };
        let sig_text = if significance.is_infinite() {
            "∞".into()
        } else if significance.is_nan() {
            "—".into()
        } else {
            format!("{significance:.1}σ")
        };
        let spots = if fit.n_samples < fit.n_total {
            format!("{} / {}", fit.n_samples, fit.n_total)
        } else {
            fit.n_samples.to_string()
        };
        let delta = if delta_pct.is_finite() {
            format!("{delta_mm:+.2} ({delta_pct:+.1}%)")
        } else {
            format!("{delta_mm:+.2} (—%)")
        };
        rows.push(vec![
            cell((*device).to_owned(), None),
            cell(spots, None),
            cell(sdd_before, None),
            cell(sdd_after, sdd_tone),
            cell(
                delta,
                if delta_pct.abs() > 1.0 {
                    Some("warn")
                } else {
                    None
                },
            ),
            cell(off_before, None),
            cell(off_text, off_tone),
            cell(
                format!("{:.4} ±{:.4} ({sig_text})", fit.gain, fit.gain_stderr),
                gain_tone,
            ),
            cell(format!("{:.3}", fit.rms_before), None),
            cell(
                format!("{:.3}", fit.rms_after),
                if fit.rms_after + 1e-9 < fit.rms_before {
                    Some("good")
                } else {
                    None
                },
            ),
        ]);
    }
    if updates > 0 {
        warnings.push(
            "Changing source_to_device_distance_mm rescales isocenter sigma by the same factor. Re-run Sigma Tuning afterwards."
                .into(),
        );
    }
    if processed == 0 && warnings.is_empty() {
        warnings.push("No ion chamber matched the session data.".into());
    }
    let summary = if updates == 0 {
        "No IC distances were updated.".into()
    } else {
        format!(
            "{updates} chamber(s): distance and zero offset will be updated from {session_label}. Removes up to {:.3} mm at the field edge. Max distance change: {:.2}%.",
            best_removed.1,
            largest.2.abs()
        )
    };
    Ok((
        summary,
        warnings,
        vec![
            "Device".into(),
            "Spots".into(),
            "SDD Before (mm)".into(),
            "SDD After (mm)".into(),
            "Δ SDD (mm %)".into(),
            "Offset Before (mm)".into(),
            "Offset After (mm)".into(),
            "Gain".into(),
            "Err Before (mm)".into(),
            "Err After (mm)".into(),
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
) -> Result<(String, Vec<String>, Vec<String>, Vec<Vec<Value>>, bool), String> {
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
            // 0 or empty means "use the primary IC total", which the desktop
            // pre-fills from the preview's mu_totals once sessions are loaded.
            let known = if known.is_finite() && known > 0.0 {
                known
            } else {
                sum_pri
            };
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
    let sums = [
        sum_positive(&spots.mu[0]),
        sum_positive(&spots.mu[1]),
        sum_positive(&spots.mu[2]),
    ];
    let mut scales = [f64::NAN; 3];
    let mut write = [false; 3];
    for family in 0..3 {
        let sum_ic = sums[family];
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
    let mut max_delta = 0.0_f64;
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
        let before = format_k0(old);
        let after = format_k0(new_kmu);
        let written = write[family] && after != before;
        if written {
            let delta = (applied - 1.0).abs() * 100.0;
            if delta > max_delta {
                max_delta = delta;
            }
            if let Some(elem) = root.at_mut(&path) {
                elem.set_attr(&attr, &after);
                updates += 1;
            }
        }
        let device = device_on_path(root, &path);
        let scale_pct = (applied - 1.0).abs() * 100.0;
        let scale_tone = if applied == 1.0 {
            None
        } else if scale_pct > 5.0 {
            Some("warn")
        } else if written {
            Some("change")
        } else {
            None
        };
        let sum_ic = sums[family];
        // The primary ends at `target` in every mode: written to it, or left
        // at sum_pri == target when the mode is "unchanged".
        let agree_before = sum_ic / sum_pri * 100.0;
        let agree_after = if applied.is_finite() && applied != 0.0 {
            sum_ic / applied / target * 100.0
        } else {
            f64::NAN
        };
        let agree_tone = |agree: f64| {
            if !agree.is_finite() {
                None
            } else if (agree - 100.0).abs() <= 1.0 {
                Some("good")
            } else if (agree - 100.0).abs() <= 5.0 {
                Some("warn")
            } else {
                Some("bad")
            }
        };
        let improve_tone = if agree_before.is_finite()
            && agree_after.is_finite()
            && (agree_after - 100.0).abs() < (agree_before - 100.0).abs() - 1e-9
        {
            Some("good")
        } else {
            None
        };
        let fmt_agree = |agree: f64| {
            if agree.is_finite() {
                format!("{agree:.1}%")
            } else {
                "—".into()
            }
        };
        rows.push(vec![
            cell(device, None),
            cell(
                if family == primary {
                    "primary"
                } else {
                    "secondary"
                },
                None,
            ),
            cell(before, None),
            cell(after, if written { Some("change") } else { None }),
            cell(format!("{:+.2}%", (applied - 1.0) * 100.0), scale_tone),
            cell(format!("{sum_ic:.2}"), None),
            cell(fmt_agree(agree_before), None),
            cell(fmt_agree(agree_after), agree_tone(agree_after)),
            cell(
                if agree_before.is_finite() && agree_after.is_finite() {
                    format!("{:+.1}%", agree_after - agree_before)
                } else {
                    "—".into()
                },
                improve_tone,
            ),
            cell(
                if written {
                    "yes"
                } else if write[family] {
                    "same"
                } else {
                    "no"
                },
                None,
            ),
        ]);
    }
    if rows.is_empty() && warnings.is_empty() {
        warnings.push("No ion chamber MU gain_conversion matched the session data.".into());
    }
    let summary = if updates == 0 {
        "No K_MU values were updated.".into()
    } else {
        format!(
            "{updates} K_MU value(s) will be updated from {session_label}. Max |Δ K_MU|: {max_delta:.2}%. Σ primary {sum_pri:.4} MU."
        )
    };
    Ok((
        summary,
        warnings,
        vec![
            "Device".into(),
            "Role".into(),
            "K_MU Before".into(),
            "K_MU After".into(),
            "Δ K_MU (%)".into(),
            "Σ MU".into(),
            "Agree Before (%)".into(),
            "Agree After (%)".into(),
            "Δ Agree (%)".into(),
            "Write".into(),
        ],
        rows,
        updates > 0,
    ))
}

fn empty_kmu(
    warnings: Vec<String>,
) -> Result<(String, Vec<String>, Vec<String>, Vec<Vec<Value>>, bool), String> {
    Ok((
        "No K_MU values were updated.".into(),
        warnings,
        vec![
            "Device".into(),
            "Role".into(),
            "K_MU Before".into(),
            "K_MU After".into(),
            "Δ K_MU (%)".into(),
            "Σ MU".into(),
            "Agree Before (%)".into(),
            "Agree After (%)".into(),
            "Δ Agree (%)".into(),
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
    // The band floor sits `headroom`% below the smallest observed sigma, so
    // hypothetical smaller spots the sessions did not sample still fit; the
    // remaining tolerance is spent covering the large end.
    let adjusted = min * (1.0 - headroom / 100.0).max(0.01);
    let fraction = tolerance / 100.0;
    if fraction <= 0.0 || fraction >= 1.0 {
        adjusted
    } else {
        adjusted / (1.0 - fraction)
    }
}

/// Fraction of finite sigmas inside the K0 tolerance band, 0.0–1.0. NaN when
/// there is no usable K0 or no samples, so before/after comparisons fall
/// through to the neutral branches.
fn inside_fraction(sigmas: &[f64], k0: f64, tolerance: f64) -> f64 {
    if !k0.is_finite() || k0 == 0.0 {
        return f64::NAN;
    }
    let lower = (k0 * (1.0 - tolerance / 100.0)).max(0.0);
    let upper = (k0 * (1.0 + tolerance / 100.0)).max(0.0);
    let mut inside = 0usize;
    let mut total = 0usize;
    for sigma in sigmas {
        if !sigma.is_finite() {
            continue;
        }
        total += 1;
        if *sigma >= lower && *sigma <= upper {
            inside += 1;
        }
    }
    if total == 0 {
        f64::NAN
    } else {
        inside as f64 / total as f64
    }
}

// Keep NaN slots: the reduce() weighted path zips these against per-spot
// weights, so indices must stay aligned with the spot rows.
fn axis_errors(spots: &TuneSpots, index: usize) -> Vec<f64> {
    if spots.error[index].iter().any(|value| value.is_finite()) {
        return spots.error[index].clone();
    }
    let plan = if index.is_multiple_of(2) {
        &spots.plan_x
    } else {
        &spots.plan_y
    };
    spots.measured[index]
        .iter()
        .zip(plan)
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
    n_total: usize,
    systematic: f64,
    /// RMS of (measured − plan) over kept samples: the raw position error.
    rms_before: f64,
    /// RMS of the fit residual over kept samples: error left after correction.
    rms_after: f64,
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
    let mut keep = vec![true; plan.len()];
    let residual: Vec<f64> = measured
        .iter()
        .zip(plan)
        .map(|(measured, plan)| measured - (fit.0 * plan + fit.1))
        .collect();
    if let Some(mask) = outlier_mask(&residual) {
        let kept_plan: Vec<f64> = plan
            .iter()
            .zip(&mask)
            .filter(|(_, keep)| **keep)
            .map(|(value, _)| *value)
            .collect();
        let kept_measured: Vec<f64> = measured
            .iter()
            .zip(&mask)
            .filter(|(_, keep)| **keep)
            .map(|(value, _)| *value)
            .collect();
        let kept_weights: Vec<f64> = sample_weights
            .iter()
            .zip(&mask)
            .filter(|(_, keep)| **keep)
            .map(|(value, _)| *value)
            .collect();
        let kept_span = kept_plan.iter().copied().fold(f64::NEG_INFINITY, f64::max)
            - kept_plan.iter().copied().fold(f64::INFINITY, f64::min);
        if kept_plan.len() >= 8 && kept_span >= 20.0 {
            if let Some(refit) = line_fit(&kept_plan, &kept_measured, &kept_weights) {
                fit = refit;
                keep = mask;
            }
        }
    }
    let n_rejected = keep.iter().filter(|keep| !**keep).count();
    let (gain, bias, stderr) = fit;
    let plan_min = plan.iter().copied().fold(f64::INFINITY, f64::min);
    let plan_max = plan.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let edges = [
        (gain - 1.0) * plan_min + bias,
        (gain - 1.0) * plan_max + bias,
    ];
    let mut total_weight = 0.0;
    let mut sq_before = 0.0;
    let mut sq_after = 0.0;
    for (((plan, measured), weight), kept) in
        plan.iter().zip(measured).zip(&sample_weights).zip(&keep)
    {
        if !kept {
            continue;
        }
        let residual = measured - (gain * plan + bias);
        total_weight += weight;
        sq_before += weight * (measured - plan).powi(2);
        sq_after += weight * residual * residual;
    }
    let rms = |sq: f64| {
        if total_weight > 0.0 {
            (sq / total_weight).sqrt()
        } else {
            f64::NAN
        }
    };
    Some(ScaleFit {
        gain,
        bias,
        gain_stderr: stderr,
        n_samples: plan.len() - n_rejected,
        n_total: plan.len(),
        systematic: edges[0].abs().max(edges[1].abs()),
        rms_before: rms(sq_before),
        rms_after: rms(sq_after),
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
    if !value.is_finite() {
        return "0".into();
    }
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
        let summary = result["summary"].as_str().unwrap();
        assert!(summary.contains("from one session"));
        assert!(summary.contains("Min inside tolerance"));
        let xml = result["xml"].as_str().unwrap();
        assert!(xml.contains("K0=\""));
        assert!(!xml.contains("K0=\"1\""));
    }

    #[test]
    fn sigma_noop_does_not_report_changes() {
        // Sigmas sit at 4.0 = K0 * (1 - 20%), so band_k0 reproduces the
        // existing 5: nothing should count as a pending write.
        let xml = r#"
            <devices><ion_chamber>
              <device name="IC_1_X"/>
              <beam_sigma_conversions in_units="MEV" out_units="mm" min_energy="70" max_energy="250" K0="5" K1="0" K2="0" K3="0"/>
            </ion_chamber></devices>
        "#;
        let spots = TuneSpots {
            energy: vec![100.0, 100.0],
            weight: vec![],
            plan_x: vec![],
            plan_y: vec![],
            measured: [vec![], vec![], vec![], vec![]],
            error: [vec![], vec![], vec![], vec![]],
            sigma: [vec![4.0, 4.0], vec![], vec![], vec![]],
            mu: [vec![], vec![], vec![]],
            charge: vec![],
        };
        let result = run_tune(
            "sigma_tuning",
            xml,
            &spots,
            &json!({
                "sigma_tolerance_percent": 20.0,
                "sigma_lower_headroom_percent": 0.0
            }),
            "one session",
        )
        .unwrap();
        assert_eq!(result["changed"], false);
        assert_eq!(
            result["summary"].as_str().unwrap(),
            "No sigma bands were updated."
        );
        // The row still renders, but K0 After carries no change tone.
        let row = &result["rows"][0];
        assert_eq!(row[3]["text"], "5");
        assert_eq!(row[4]["text"], "5");
        assert!(row[4].get("tone").is_none());
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
        let summary = result["summary"].as_str().unwrap();
        assert!(summary.contains("Max correction: 0.400 mm"));
        assert!(result["xml"].as_str().unwrap().contains("0.6"));
    }
}
