use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use scan_kit_core::{
    beam_on_mask, calibration_factor, cumsum, histogram, scale_column, spill_segments, PlotScene,
    Series, MIN_SPILL_GAP_MS,
};
use serde_json::Value;

use super::{
    apply_filter, col, discover, drew_line, energy_lookup, finite_col, guide, labeled, load_csv,
    load_timeslice, panel, pick, placed, scene, slice_table, spot_table, stroke, timeslice_metric,
    LINKED, MARK,
};

const CALIBRATE_CHOICES: &[(&str, &str)] = &[("off", "Off"), ("on", "On")];

pub(super) fn peak_amplitude(root: &Path, session_ids: &[String]) -> PlotScene {
    let mut panels = Vec::new();
    for (title, key) in [
        ("IC1 X", "ic1_x_peak"),
        ("IC1 Y", "ic1_y_peak"),
        ("IC2 X", "ic2_x_peak"),
        ("IC2 Y", "ic2_y_peak"),
    ] {
        let mut series = Vec::new();
        let mut lo = f32::MAX;
        let mut hi = f32::MIN;
        let mut top = 1.0f32;
        for session in session_ids {
            let mut table = timeslice_metric(root, session, "peak_amplitude");
            apply_filter(&mut table, &[key], "all", "beam_off");
            let values = table.get(key).cloned().unwrap_or_default();
            if !values.iter().any(|value| value.is_finite()) {
                continue;
            }
            let (edges, counts) = histogram(&values, 16);
            lo = lo.min(edges.first().copied().unwrap_or(0.0));
            hi = hi.max(edges.last().copied().unwrap_or(1.0));
            top = top.max(counts.iter().copied().fold(0.0, f32::max));
            series.push(Series::Bars {
                edges,
                counts,
                color: MARK,
            });
        }
        if series.is_empty() {
            continue;
        }
        panels.push(panel(title.to_owned(), lo, hi, 0.0, top, series));
    }
    scene("IC Peak Amplitude — Beam-Off", panels, Vec::new())
}

pub(super) fn dose_accumulation(root: &Path, session_ids: &[String], options: &Value) -> PlotScene {
    let calibrate = pick(options, "calibrate", "off", CALIBRATE_CHOICES) == "on";
    let mut panels = Vec::new();
    for (label, dose_key, current_key) in [
        ("IC1", "ic1_dose", "ic1_current"),
        ("IC2", "ic2_dose", "ic2_current"),
        ("IC3", "ic3_dose", "ic3_current"),
    ] {
        let mut cumulative = Vec::new();
        let mut error = Vec::new();
        let mut current_sum = Vec::new();
        for session in session_ids {
            let table = spot_table(root, session);
            let Some(target) = col(&table, "target_mu") else {
                continue;
            };
            let Some(dose) = col(&table, dose_key) else {
                continue;
            };
            let n = target.len().min(dose.len());
            if n == 0 {
                continue;
            }
            let mut dose = dose[..n].to_vec();
            if calibrate {
                if let Some(factor) = calibration_factor(&target[..n], &dose) {
                    dose = scale_column(&dose, factor);
                }
            }
            let cum_t = cumsum(&target[..n]);
            let cum_d = cumsum(&dose);
            let xs: Vec<f32> = (1..=n).map(|i| i as f32).collect();
            let err: Vec<f32> = cum_d.iter().zip(&cum_t).map(|(d, t)| d - t).collect();
            cumulative.push(stroke(xs.clone(), cum_d, false));
            cumulative.push(guide(xs.clone(), cum_t));
            error.push(stroke(xs, err, false));
            let currents = timeslice_metric(root, session, "ic_current");
            if let Some(samples) = col(&currents, current_key).filter(|values| values.len() == n) {
                current_sum.push(stroke(
                    (1..=n).map(|i| i as f32).collect(),
                    cumsum(samples),
                    false,
                ));
            }
        }
        if drew_line(&cumulative) {
            panels.push(placed(format!("{label} Cumulative"), cumulative));
            panels.push(placed(format!("{label} Error"), error));
        }
        if drew_line(&current_sum) {
            panels.push(placed(format!("{label} Current Sum"), current_sum));
        }
    }
    scene(
        "Dose Accumulation",
        panels,
        vec![labeled(
            "calibrate",
            "Calibrate",
            CALIBRATE_CHOICES,
            if calibrate { "on" } else { "off" },
        )],
    )
}

pub(super) fn beam_motion(root: &Path, session_ids: &[String]) -> PlotScene {
    let mut groups: BTreeMap<i32, Vec<Series>> = BTreeMap::new();
    for session in session_ids {
        let Some(motion) = motion_columns(root, session) else {
            continue;
        };
        let mut segments = spill_segments(&motion.on, MIN_SPILL_GAP_MS, 2);
        if segments.is_empty() {
            segments.push((0, motion.x.len()));
        }
        let mut by_energy: BTreeMap<i32, Vec<(usize, usize)>> = BTreeMap::new();
        for (start, end) in segments {
            let sample = start.min(motion.energy.len().saturating_sub(1));
            let key = (motion.energy.get(sample).copied().unwrap_or(0.0) * 10.0).round() as i32;
            by_energy.entry(key).or_default().push((start, end));
        }
        for (key, spills) in by_energy {
            let mut sx = Vec::new();
            let mut sy = Vec::new();
            let mut tx = Vec::new();
            let mut ty = Vec::new();
            for (start, end) in spills {
                push_spill(&mut sx, &mut sy, &motion.x, &motion.y, start, end);
                if !motion.x2.is_empty() {
                    push_spill(&mut tx, &mut ty, &motion.x2, &motion.y2, start, end);
                }
            }
            if !sx.iter().any(|value| value.is_finite()) {
                continue;
            }
            if motion.y.len() >= motion.x.len() {
                push_circle(&mut sx, &mut sy, 1.0);
            }
            let series = groups.entry(key).or_default();
            series.push(Series::Polyline {
                xs: sx,
                ys: sy,
                color: MARK,
                thickness: 1.2,
            });
            if tx.iter().any(|value| value.is_finite()) {
                series.push(Series::Polyline {
                    xs: tx,
                    ys: ty,
                    color: LINKED,
                    thickness: 1.2,
                });
            }
        }
    }
    let panels = groups
        .into_iter()
        .take(40)
        .map(|(key, series)| placed(format!("{:.0} MeV", key as f32 / 10.0), series))
        .collect();
    scene("Beam Error Motion vs Energy", panels, Vec::new())
}

pub(super) fn session_dir(root: &Path, session_id: &str) -> PathBuf {
    discover::session_directory(root, session_id)
}

struct Motion {
    x: Vec<f32>,
    y: Vec<f32>,
    x2: Vec<f32>,
    y2: Vec<f32>,
    energy: Vec<f32>,
    on: Vec<bool>,
}

fn motion_columns(root: &Path, session: &str) -> Option<Motion> {
    let slice = slice_table(root, session);
    if finite_col(&slice, "ic1_x_err").is_some() {
        let x = col(&slice, "ic1_x_err")?.to_vec();
        let y = finite_col(&slice, "ic1_y_err").unwrap_or(&[]).to_vec();
        let x2 = finite_col(&slice, "ic2_x_err").unwrap_or(&[]).to_vec();
        let y2 = finite_col(&slice, "ic2_y_err").unwrap_or(&[]).to_vec();
        let energy = col(&slice, "energy").unwrap_or(&[]).to_vec();
        let on = beam_flags(col(&slice, "beam_on"), x.len());
        return Some(Motion {
            x,
            y,
            x2,
            y2,
            energy,
            on,
        });
    }
    let raw = load_timeslice(root, session);
    let x = col(&raw, "position_error_x")?.to_vec();
    if x.is_empty() {
        return None;
    }
    let y = finite_col(&raw, "position_error_y").unwrap_or(&[]).to_vec();
    let x2 = finite_col(&raw, "position_error_x2")
        .unwrap_or(&[])
        .to_vec();
    let y2 = finite_col(&raw, "position_error_y2")
        .unwrap_or(&[])
        .to_vec();
    let energy = sample_energy(root, session, &raw, x.len());
    let gate = col(&raw, "rci_in_trigger").or_else(|| col(&raw, "r_beamOk"));
    let on = gate
        .map(beam_on_mask)
        .unwrap_or_else(|| vec![true; x.len()]);
    Some(Motion {
        x,
        y,
        x2,
        y2,
        energy,
        on,
    })
}

fn beam_flags(gate: Option<&[f32]>, n: usize) -> Vec<bool> {
    match gate {
        Some(values) if values.len() == n => values.iter().map(|value| *value > 0.5).collect(),
        _ => vec![true; n],
    }
}

fn sample_energy(
    root: &Path,
    session: &str,
    columns: &BTreeMap<String, Vec<f32>>,
    n: usize,
) -> Vec<f32> {
    let map = load_csv(root, session, "input_map.csv");
    let mut by_layer = BTreeMap::new();
    if let (Some(layers), Some(energy)) = (col(&map, "layer_id"), col(&map, "energy")) {
        for (layer, mev) in layers.iter().zip(energy) {
            if layer.is_finite() && mev.is_finite() {
                by_layer.insert(layer.round() as i32, *mev);
            }
        }
    }
    let fallback = energy_lookup(root, session).first().copied().unwrap_or(0.0);
    if let Some(layers) = col(columns, "layer_id") {
        return (0..n)
            .map(|i| {
                layers
                    .get(i)
                    .and_then(|layer| by_layer.get(&(layer.round() as i32)).copied())
                    .unwrap_or(fallback)
            })
            .collect();
    }
    vec![fallback; n]
}

fn push_spill(
    xs: &mut Vec<f32>,
    ys: &mut Vec<f32>,
    x: &[f32],
    y: &[f32],
    start: usize,
    end: usize,
) {
    if xs.iter().any(|value| value.is_finite()) {
        xs.push(f32::NAN);
        ys.push(f32::NAN);
    }
    for i in start..end.min(x.len()) {
        let yv = if y.is_empty() {
            (i - start) as f32
        } else {
            y.get(i).copied().unwrap_or(f32::NAN)
        };
        if x[i].is_finite() && yv.is_finite() {
            xs.push(x[i]);
            ys.push(yv);
        }
    }
}

fn push_circle(xs: &mut Vec<f32>, ys: &mut Vec<f32>, radius: f32) {
    xs.push(f32::NAN);
    ys.push(f32::NAN);
    for step in 0..=24 {
        let angle = step as f32 / 24.0 * std::f32::consts::TAU;
        xs.push(radius * angle.cos());
        ys.push(radius * angle.sin());
    }
}
