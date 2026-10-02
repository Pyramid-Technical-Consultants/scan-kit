use std::collections::BTreeMap;
use std::path::Path;

use scan_kit_core::{
    beam_on_mask, coverage_percent, BeamState, Family, Panel, PlotScene, Series, SESSION,
};
use serde_json::Value;

use super::{
    apply_filter, col, contour_bands, drew_line, finite_col, guide, labeled, panel,
    percentile_sorted, pick, placed, scene, slice_table, spot_table, stroke, timeslice_metric,
    BEAM_CHOICES, MARK,
};

const MODE_CHOICES: &[(&str, &str)] = &[
    ("position", "Position"),
    ("position_error", "Position Error"),
    ("sigma", "Sigma"),
    ("confidence", "Confidence"),
    ("coverage", "Coverage"),
];
const GRAIN_CHOICES: &[(&str, &str)] = &[("spot", "Spot"), ("timeslice", "Timeslice")];
const DRAW_CHOICES: &[(&str, &str)] = &[
    ("scatter", "Scatter"),
    ("contour", "Contour"),
    ("density", "Density"),
];
const CUTOFF_CHOICES: &[(&str, &str)] = &[("0", "0"), ("5", "5"), ("10", "10"), ("20", "20")];
const DENSITY_BINS: usize = 80;

pub(super) fn distribution(root: &Path, session_ids: &[String], options: &Value) -> PlotScene {
    let mode = pick(options, "mode", "position", MODE_CHOICES);
    let grain = pick(options, "grain", "spot", GRAIN_CHOICES);
    let beam = pick(
        options,
        "beam",
        if grain == "timeslice" {
            "beam_on"
        } else {
            "beam_both"
        },
        BEAM_CHOICES,
    );
    let draw = pick(options, "draw", "scatter", DRAW_CHOICES);
    let ramp = pick(
        options,
        "ramp",
        "turbo",
        scan_kit_core::choices(Family::Sequential),
    );
    let cutoff_id = pick(options, "cutoff", "5", CUTOFF_CHOICES);
    let cutoff = cutoff_id.parse::<f32>().unwrap_or(5.0);
    let xy = matches!(mode, "position" | "position_error" | "sigma");
    let (panels, columns) = if mode == "confidence" {
        (
            confidence_scene(root, session_ids, beam, draw, ramp, cutoff),
            0,
        )
    } else if mode == "coverage" {
        (coverage_scene(root, session_ids, beam), 0)
    } else {
        column_scene(root, session_ids, mode, grain, beam, draw, ramp, cutoff)
    };
    let mut controls = Vec::new();
    if xy {
        controls.push(labeled("grain", "Source", GRAIN_CHOICES, grain));
    }
    controls.push(labeled("mode", "Signal", MODE_CHOICES, mode));
    if mode != "coverage" {
        controls.push(labeled("draw", "Style", DRAW_CHOICES, draw));
        if draw == "density" && session_ids.len() == 1 {
            controls.push(labeled(
                "ramp",
                "Ramp",
                scan_kit_core::choices(Family::Sequential),
                ramp,
            ));
        }
        if draw == "contour" {
            controls.push(labeled(
                "cutoff",
                "Contour Cutoff",
                CUTOFF_CHOICES,
                cutoff_id,
            ));
        }
    }
    controls.push(labeled("beam", "Beam", BEAM_CHOICES, beam));
    let mut scene = scene("Distribution Explorer", panels, controls);
    scene.columns = columns;
    if columns > 0 && scene.panels.len() == columns as usize * 3 {
        scene.row_weights = vec![2.0, 1.0, 1.0];
    }
    scene
}

fn finite_pairs(xs: &[f32], ys: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let mut ox = Vec::new();
    let mut oy = Vec::new();
    for (x, y) in xs.iter().zip(ys) {
        if x.is_finite() && y.is_finite() {
            ox.push(*x);
            oy.push(*y);
        }
    }
    (ox, oy)
}

fn kept_pairs(
    table: &BTreeMap<String, Vec<f32>>,
    x_key: &str,
    y_key: &str,
    beam: &str,
) -> (Vec<f32>, Vec<f32>) {
    let mut copy = table.clone();
    apply_filter(&mut copy, &[x_key, y_key], "all", beam);
    finite_pairs(
        copy.get(x_key).map(Vec::as_slice).unwrap_or(&[]),
        copy.get(y_key).map(Vec::as_slice).unwrap_or(&[]),
    )
}

const HIST_BINS: usize = 101;
/// Histogram bars stay translucent so overlaid sessions both stay visible.
const HIST: [f32; 4] = [0.0, 0.0, 0.0, 0.55];

pub(super) fn distribution_limits(mode: &str, samples: &[f32]) -> (f32, f32) {
    let (lo, hi) = if mode == "sigma" {
        let mut positive: Vec<f32> = samples
            .iter()
            .copied()
            .filter(|value| *value > 0.0)
            .collect();
        positive.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let hi = percentile_sorted(&positive, 0.9995).max(1.0);
        (0.0, hi)
    } else if mode == "position_error" {
        let mut abs: Vec<f32> = samples
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .map(f32::abs)
            .collect();
        abs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let bound = percentile_sorted(&abs, 0.9995).max(1.0);
        (-bound, bound)
    } else {
        let mut finite: Vec<f32> = samples
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .collect();
        finite.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        if finite.is_empty() {
            (-1.0, 1.0)
        } else {
            let lo = percentile_sorted(&finite, 0.0005);
            let hi = percentile_sorted(&finite, 0.9995);
            let mid = 0.5 * (lo + hi);
            let half = (mid - lo).max(hi - mid).max(0.5);
            (mid - half, mid + half)
        }
    };
    // A little past the spots so they are not drawn on the frame.
    let pad = ((hi - lo) * 0.06).max(1.0e-4);
    (lo - pad, hi + pad)
}

fn probability_panel(title: &str, lo: f32, hi: f32, columns: &[&[f32]]) -> Panel {
    let mut edges = Vec::with_capacity(HIST_BINS + 1);
    for step in 0..=HIST_BINS {
        edges.push(lo + (hi - lo) * step as f32 / HIST_BINS as f32);
    }
    let mut series = Vec::new();
    let mut top = 1.0f32;
    for values in columns {
        let counts = probability_counts(values, &edges);
        top = top.max(counts.iter().copied().fold(0.0, f32::max));
        series.push(Series::Bars {
            edges: edges.clone(),
            counts,
            color: HIST,
        });
    }
    let mut panel = panel(title.to_owned(), lo, hi, 0.0, top, series);
    panel.y_label = "Probability (%)".into();
    panel
}

fn probability_counts(values: &[f32], edges: &[f32]) -> Vec<f32> {
    let bins = edges.len().saturating_sub(1).max(1);
    let mut counts = vec![0.0f32; bins];
    let lo = edges[0];
    let hi = edges[edges.len() - 1];
    let width = hi - lo;
    let mut total = 0.0f32;
    for value in values.iter().copied() {
        if !value.is_finite() || value < lo || value > hi || width <= 0.0 {
            continue;
        }
        let index = (((value - lo) / width) * bins as f32) as usize;
        counts[index.min(bins - 1)] += 1.0;
        total += 1.0;
    }
    if total > 0.0 {
        for count in &mut counts {
            *count = 100.0 * *count / total;
        }
    }
    counts
}

fn grain_table(root: &Path, session: &str, grain: &str) -> BTreeMap<String, Vec<f32>> {
    if grain == "timeslice" {
        slice_table(root, session)
    } else {
        spot_table(root, session)
    }
}

/// Several sessions fade from transparent to their own color. One session uses the chosen ramp.
fn heat_ramp(sessions: usize, ramp: &str) -> u8 {
    if sessions > 1 {
        SESSION
    } else {
        scan_kit_core::resolve(ramp, Family::Sequential)
    }
}

fn density_map(values: Vec<f32>, ramp: u8) -> Series {
    let bins = DENSITY_BINS as u32;
    Series::Heatmap {
        values,
        cols: bins,
        rows: bins,
        ramp,
        color: if scan_kit_core::is_session(ramp) {
            MARK
        } else {
            [1.0, 1.0, 1.0, 1.0]
        },
        lo: 0.0,
        hi: 0.0,
    }
}

fn density_grid(xs: &[f32], ys: &[f32], x0: f32, x1: f32, y0: f32, y1: f32) -> Vec<f32> {
    let bins = DENSITY_BINS;
    let mut counts = vec![0.0f32; bins * bins];
    let dx = (x1 - x0).max(1e-6);
    let dy = (y1 - y0).max(1e-6);
    for (x, y) in xs.iter().zip(ys) {
        if !x.is_finite() || !y.is_finite() || *x < x0 || *x > x1 || *y < y0 || *y > y1 {
            continue;
        }
        let ix = (((x - x0) / dx) * bins as f32) as usize;
        let iy = (((y - y0) / dy) * bins as f32) as usize;
        counts[ix.min(bins - 1) + bins * iy.min(bins - 1)] += 1.0;
    }
    counts
}

pub(super) fn reference_ring() -> Series {
    let steps = 64;
    let mut xs = Vec::with_capacity(steps + 1);
    let mut ys = Vec::with_capacity(steps + 1);
    for step in 0..=steps {
        let angle = step as f32 / steps as f32 * std::f32::consts::TAU;
        xs.push(angle.cos());
        ys.push(angle.sin());
    }
    guide(xs, ys)
}

fn padded_span(values: &[f32]) -> (f32, f32) {
    let mut lo = f32::MAX;
    let mut hi = f32::MIN;
    for value in values {
        if value.is_finite() {
            lo = lo.min(*value);
            hi = hi.max(*value);
        }
    }
    if !lo.is_finite() || hi <= lo {
        return (0.0, 1.0);
    }
    let pad = ((hi - lo) * 0.04).max(1e-4);
    (lo - pad, hi + pad)
}

fn column_scene(
    root: &Path,
    session_ids: &[String],
    mode: &str,
    grain: &str,
    beam: &str,
    draw: &str,
    ramp: &str,
    cutoff: f32,
) -> (Vec<Panel>, u32) {
    let mut pairs: Vec<(&str, &str, &str)> = match mode {
        "position_error" => vec![
            ("IC1", "ic1_x_err", "ic1_y_err"),
            ("IC2", "ic2_x_err", "ic2_y_err"),
        ],
        "sigma" => vec![
            ("IC1", "ic1_sig_x", "ic1_sig_y"),
            ("IC2", "ic2_sig_x", "ic2_sig_y"),
        ],
        _ => vec![
            ("IC1", "ic1_x", "ic1_y"),
            ("IC2", "ic2_x", "ic2_y"),
            ("Plan", "plan_x", "plan_y"),
        ],
    };
    let tables: Vec<_> = session_ids
        .iter()
        .map(|session| grain_table(root, session, grain))
        .collect();
    if mode == "position" {
        let measured = tables
            .iter()
            .any(|table| finite_col(table, "ic1_x").is_some());
        if measured {
            pairs.retain(|(_, x_key, _)| *x_key != "plan_x");
        } else {
            pairs.retain(|(_, x_key, _)| *x_key == "plan_x");
        }
    }
    let (x_label, y_label) = match mode {
        "position_error" => ("X Error (mm)", "Y Error (mm)"),
        "sigma" => ("X Sigma (mm)", "Y Sigma (mm)"),
        _ => ("X Position (mm)", "Y Position (mm)"),
    };
    let mut tops = Vec::new();
    let mut x_hists = Vec::new();
    let mut y_hists = Vec::new();
    for (title, x_key, y_key) in pairs {
        let mut clouds = Vec::new();
        for table in &tables {
            let (xs, ys) = kept_pairs(table, x_key, y_key, beam);
            if !xs.is_empty() {
                clouds.push((xs, ys));
            }
        }
        if clouds.is_empty() {
            continue;
        }
        let mut samples = Vec::new();
        for (xs, ys) in &clouds {
            samples.extend(xs.iter().copied());
            samples.extend(ys.iter().copied());
        }
        let (lo, hi) = distribution_limits(mode, &samples);
        let mut series = Vec::new();
        if mode == "position_error" {
            series.push(guide(vec![lo, hi], vec![0.0, 0.0]));
            series.push(guide(vec![0.0, 0.0], vec![lo, hi]));
            series.push(reference_ring());
        }
        let ramp_id = heat_ramp(session_ids.len(), ramp);
        match draw {
            "density" => {
                for (xs, ys) in &clouds {
                    series.push(density_map(density_grid(xs, ys, lo, hi, lo, hi), ramp_id));
                }
            }
            "contour" => {
                for (xs, ys) in &clouds {
                    series.extend(contour_bands(xs, ys, cutoff));
                }
            }
            _ => {
                for (xs, ys) in &clouds {
                    series.push(Series::Points {
                        xs: xs.clone(),
                        ys: ys.clone(),
                        color: MARK,
                        radius: 2.0,
                    });
                }
            }
        }
        let mut top = panel(title.to_owned(), lo, hi, lo, hi, series);
        top.equal = true;
        tops.push(top);
        let xs: Vec<&[f32]> = clouds.iter().map(|(xs, _)| xs.as_slice()).collect();
        let ys: Vec<&[f32]> = clouds.iter().map(|(_, ys)| ys.as_slice()).collect();
        x_hists.push(probability_panel(x_label, lo, hi, &xs));
        y_hists.push(probability_panel(y_label, lo, hi, &ys));
    }
    let columns = tops.len() as u32;
    tops.extend(x_hists);
    tops.extend(y_hists);
    (tops, columns)
}

fn confidence_scene(
    root: &Path,
    session_ids: &[String],
    beam: &str,
    draw: &str,
    ramp: &str,
    cutoff: f32,
) -> Vec<Panel> {
    let axes = [
        ("IC1 X", "ic1_x_peak", "ic1_x_confidence"),
        ("IC1 Y", "ic1_y_peak", "ic1_y_confidence"),
        ("IC2 X", "ic2_x_peak", "ic2_x_confidence"),
        ("IC2 Y", "ic2_y_peak", "ic2_y_confidence"),
    ];
    let mut panels = Vec::new();
    for (title, peak_key, conf_key) in axes {
        let mut clouds = Vec::new();
        for session in session_ids {
            let mut peaks = timeslice_metric(root, session, "peak_amplitude");
            let mut confidence = timeslice_metric(root, session, "fit_confidence");
            apply_filter(&mut peaks, &[peak_key], "all", beam);
            apply_filter(&mut confidence, &[conf_key], "all", beam);
            let (xs, ys) = finite_pairs(
                peaks.get(peak_key).map(Vec::as_slice).unwrap_or(&[]),
                confidence.get(conf_key).map(Vec::as_slice).unwrap_or(&[]),
            );
            if !xs.is_empty() {
                clouds.push((xs, ys));
            }
        }
        if clouds.is_empty() {
            continue;
        }
        if draw == "density" || draw == "contour" {
            let mut all_x = Vec::new();
            let mut all_y = Vec::new();
            for (xs, ys) in &clouds {
                all_x.extend(xs.iter().copied());
                all_y.extend(ys.iter().copied());
            }
            let (x0, x1) = padded_span(&all_x);
            let (y0, y1) = padded_span(&all_y);
            let series = if draw == "density" {
                let ramp_id = heat_ramp(session_ids.len(), ramp);
                clouds
                    .iter()
                    .map(|(xs, ys)| density_map(density_grid(xs, ys, x0, x1, y0, y1), ramp_id))
                    .collect()
            } else {
                let mut series = Vec::new();
                for (xs, ys) in &clouds {
                    series.extend(contour_bands(xs, ys, cutoff));
                }
                series
            };
            if !series.is_empty() {
                panels.push(panel(title.to_owned(), x0, x1, y0, y1, series));
            }
            continue;
        }
        let series = clouds
            .into_iter()
            .map(|(xs, ys)| Series::Points {
                xs,
                ys,
                color: MARK,
                radius: 2.0,
            })
            .collect();
        panels.push(placed(title.to_owned(), series));
    }
    panels
}

fn coverage_scene(root: &Path, session_ids: &[String], beam: &str) -> Vec<Panel> {
    let thresholds: Vec<f32> = (0..=400).map(|step| step as f32 * 0.25).collect();
    let mut panels = Vec::new();
    for (title, x_key, y_key) in [
        ("IC1 Coverage", "ic1_x_confidence", "ic1_y_confidence"),
        ("IC2 Coverage", "ic2_x_confidence", "ic2_y_confidence"),
    ] {
        let mut series = Vec::new();
        for session in session_ids {
            let mut table = timeslice_metric(root, session, "fit_confidence");
            apply_filter(&mut table, &[x_key, y_key], "all", beam);
            let metrics = spot_coverage_metrics(
                None,
                col(&table, x_key),
                col(&table, y_key),
                None,
                BeamState::All,
            );
            if metrics.is_empty() {
                continue;
            }
            series.push(stroke(
                thresholds.clone(),
                coverage_percent(&metrics, &thresholds),
                false,
            ));
        }
        if drew_line(&series) {
            panels.push(placed(title.to_owned(), series));
        }
    }
    panels
}

fn spot_coverage_metrics(
    spot: Option<&[f32]>,
    x_conf: Option<&[f32]>,
    y_conf: Option<&[f32]>,
    gate: Option<&[f32]>,
    state: BeamState,
) -> Vec<f32> {
    let n = x_conf
        .map(|values| values.len())
        .unwrap_or(0)
        .max(y_conf.map(|values| values.len()).unwrap_or(0));
    if n == 0 {
        return Vec::new();
    }
    let on = gate.filter(|values| values.len() == n).map(beam_on_mask);
    let mut order = BTreeMap::<i32, usize>::new();
    let mut max_x = Vec::new();
    let mut max_y = Vec::new();
    for i in 0..n {
        let keep = match state {
            BeamState::All => true,
            BeamState::On => on
                .as_ref()
                .and_then(|mask| mask.get(i))
                .copied()
                .unwrap_or(true),
            BeamState::Off => !on
                .as_ref()
                .and_then(|mask| mask.get(i))
                .copied()
                .unwrap_or(false),
        };
        if !keep {
            continue;
        }
        let id = if let Some(values) = spot {
            let Some(value) = values.get(i).copied() else {
                continue;
            };
            if !value.is_finite() {
                continue;
            }
            value.round() as i32
        } else {
            i as i32
        };
        let slot = if let Some(slot) = order.get(&id) {
            *slot
        } else {
            let slot = max_x.len();
            order.insert(id, slot);
            max_x.push(f32::NEG_INFINITY);
            max_y.push(f32::NEG_INFINITY);
            slot
        };
        if let Some(value) = x_conf.and_then(|values| values.get(i)).copied() {
            if value.is_finite() {
                max_x[slot] = max_x[slot].max(value);
            }
        }
        if let Some(value) = y_conf.and_then(|values| values.get(i)).copied() {
            if value.is_finite() {
                max_y[slot] = max_y[slot].max(value);
            }
        }
    }
    max_x
        .into_iter()
        .zip(max_y)
        .map(|(x, y)| x.min(y))
        .filter(|value| value.is_finite())
        .collect()
}
