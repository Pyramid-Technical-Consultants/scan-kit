use std::collections::BTreeMap;
use std::path::Path;

use scan_kit_core::{
    beam_on_mask, coverage_percent, BeamState, Family, Panel, PlotScene, Series, SESSION,
};
use serde_json::Value;

use crate::histogram::{hist_bin_count, histogram_panel, HIST_BIN_CHOICES};

use super::{
    apply_filter, col, contour_bands, control, drew_line, finite_col, flag, guide, labeled, panel,
    percentile_sorted, pick, placed, scene, slice_table, spot_table, stroke, text,
    timeslice_metric, BEAM_CHOICES, MARK,
};

const DRAW_CHOICES: &[(&str, &str)] = &[
    ("scatter", "Scatter"),
    ("contour", "Contour"),
    ("density", "Density"),
];
const CUTOFF_CHOICES: &[(&str, &str)] = &[("0", "0"), ("5", "5"), ("10", "10"), ("20", "20")];
const DENSITY_BINS: usize = 80;

pub(super) fn distribution(root: &Path, session_ids: &[String], options: &Value) -> PlotScene {
    let timeslice = crate::source::wants_timeslice(crate::source::Shape::Xy, options);
    let owned: Vec<(String, Vec<String>)> = session_ids
        .iter()
        .map(|id| {
            (
                id.clone(),
                crate::tables::grain_columns(root, id, timeslice),
            )
        })
        .collect();
    let headers: Vec<crate::source::SessionCols<'_>> = owned
        .iter()
        .map(|(name, columns)| crate::source::SessionCols { name, columns })
        .collect();
    let picked = crate::source::select(crate::source::Shape::Xy, true, true, &headers, options);
    let mode = picked.xy;
    let grain = picked.grain;
    let beam = pick(
        options,
        "beam",
        if grain == "timeslice" || matches!(mode, "amplifier" | "probe") {
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
    let chambers = matches!(mode, "position" | "position_error" | "sigma");
    let cloud = chambers || matches!(mode, "amplifier" | "probe");
    let show_ic1 = flag(options, "ic1", true);
    let show_ic2 = flag(options, "ic2", true);
    let show_plan = flag(options, "plan", true);
    let bins = hist_bin_count(text(options, "hist_bins", "30"));
    let (panels, columns, has_plan) = if mode == "confidence" {
        (
            confidence_scene(root, session_ids, beam, draw, ramp, cutoff),
            0,
            false,
        )
    } else if mode == "coverage" {
        (coverage_scene(root, session_ids, beam), 0, false)
    } else {
        column_scene(
            root,
            session_ids,
            mode,
            grain,
            beam,
            draw,
            ramp,
            cutoff,
            show_ic1,
            show_ic2,
            show_plan,
            bins,
        )
    };
    let mut controls = picked.controls;
    if chambers {
        controls.push(
            control("ic1", "IC1", &["Off", "On"], on_off(show_ic1))
                .grouped("Data Source")
                .checked(),
        );
        controls.push(
            control("ic2", "IC2", &["Off", "On"], on_off(show_ic2))
                .grouped("Data Source")
                .checked(),
        );
        if mode == "position" && has_plan {
            controls.push(
                control("plan", "Plan", &["Off", "On"], on_off(show_plan))
                    .grouped("Data Source")
                    .checked(),
            );
        }
    }
    if mode != "coverage" {
        controls.push(labeled("draw", "Style", DRAW_CHOICES, draw).grouped("Plot Style"));
        if draw == "density" && session_ids.len() == 1 {
            controls.push(
                labeled(
                    "ramp",
                    "Ramp",
                    scan_kit_core::choices(Family::Sequential),
                    ramp,
                )
                .grouped("Plot Style"),
            );
        }
        if draw == "contour" {
            controls.push(
                labeled("cutoff", "Contour Cutoff", CUTOFF_CHOICES, cutoff_id)
                    .grouped("Plot Style"),
            );
        }
    }
    if cloud {
        controls.push(
            control("hist_bins", "Bins", HIST_BIN_CHOICES, &bins.to_string()).grouped("Histogram"),
        );
    }
    controls.push(labeled("beam", "Beam", BEAM_CHOICES, beam).grouped("Filter Data"));
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

fn on_off(on: bool) -> &'static str {
    if on {
        "On"
    } else {
        "Off"
    }
}

fn axis_name(column: &str, axis: &str, mode: &str) -> String {
    match mode {
        "amplifier" => format!("{axis} Error (V)"),
        "probe" => format!("{axis} (G)"),
        "position_error" => format!("{column} {axis} Error (mm)"),
        "sigma" => format!("{column} {axis} Sigma (mm)"),
        _ => format!("{column} {axis} (mm)"),
    }
}

fn limit_kind(mode: &str) -> &'static str {
    match mode {
        "sigma" => "sigma",
        "position_error" | "amplifier" => "position_error",
        _ => "position",
    }
}

fn grain_table(root: &Path, session: &str, grain: &str) -> BTreeMap<String, Vec<f32>> {
    if grain == "timeslice" {
        slice_table(root, session)
    } else {
        spot_table(root, session)
    }
}

fn session_table(
    root: &Path,
    session: &str,
    mode: &str,
    grain: &str,
) -> BTreeMap<String, Vec<f32>> {
    match mode {
        "amplifier" => timeslice_metric(root, session, "amplifier_error"),
        "probe" => timeslice_metric(root, session, "probe_field"),
        _ => grain_table(root, session, grain),
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

struct DrawnColumn {
    name: &'static str,
    clouds: Vec<(Vec<f32>, Vec<f32>)>,
}

fn column_specs(
    mode: &str,
    ic1: bool,
    ic2: bool,
    plan: bool,
) -> Vec<(&'static str, &'static str, &'static str)> {
    let mut pairs = Vec::new();
    match mode {
        "position_error" => {
            if ic1 {
                pairs.push(("IC1", "ic1_x_err", "ic1_y_err"));
            }
            if ic2 {
                pairs.push(("IC2", "ic2_x_err", "ic2_y_err"));
            }
        }
        "sigma" => {
            if ic1 {
                pairs.push(("IC1", "ic1_sig_x", "ic1_sig_y"));
            }
            if ic2 {
                pairs.push(("IC2", "ic2_sig_x", "ic2_sig_y"));
            }
        }
        "amplifier" => pairs.push(("Amplifier", "amp_x", "amp_y")),
        "probe" => pairs.push(("Probe", "field_x", "field_y")),
        _ => {
            if plan {
                pairs.push(("Plan", "plan_x", "plan_y"));
            }
            if ic1 {
                pairs.push(("IC1", "ic1_x", "ic1_y"));
            }
            if ic2 {
                pairs.push(("IC2", "ic2_x", "ic2_y"));
            }
        }
    }
    pairs
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
    ic1: bool,
    ic2: bool,
    plan: bool,
    bins: usize,
) -> (Vec<Panel>, u32, bool) {
    let tables: Vec<_> = session_ids
        .iter()
        .map(|session| session_table(root, session, mode, grain))
        .collect();
    let has_plan = tables
        .iter()
        .any(|table| finite_col(table, "plan_x").is_some());
    let drawn_plan = mode == "position" && plan && has_plan;
    let mut columns = Vec::new();
    for (name, x_key, y_key) in column_specs(mode, ic1, ic2, drawn_plan) {
        let mut clouds = Vec::new();
        for table in &tables {
            let (xs, ys) = kept_pairs(table, x_key, y_key, beam);
            if !xs.is_empty() {
                clouds.push((xs, ys));
            }
        }
        if !clouds.is_empty() {
            columns.push(DrawnColumn { name, clouds });
        }
    }
    if columns.is_empty() {
        return (
            vec![panel(
                "No columns selected".into(),
                0.0,
                1.0,
                0.0,
                1.0,
                Vec::new(),
            )],
            0,
            has_plan,
        );
    }
    let mut samples = Vec::new();
    for column in &columns {
        for (xs, ys) in &column.clouds {
            samples.extend(xs.iter().copied());
            samples.extend(ys.iter().copied());
        }
    }
    let (lo, hi) = distribution_limits(limit_kind(mode), &samples);
    let hist_guides: Vec<f32> = if mode == "position_error" {
        vec![0.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0]
    } else {
        Vec::new()
    };
    let mut tops = Vec::new();
    let mut x_hists = Vec::new();
    let mut y_hists = Vec::new();
    let ramp_id = heat_ramp(session_ids.len(), ramp);
    for column in &columns {
        let mut series = Vec::new();
        if matches!(mode, "position" | "position_error" | "amplifier" | "probe") {
            series.push(guide(vec![lo, hi], vec![0.0, 0.0]));
            series.push(guide(vec![0.0, 0.0], vec![lo, hi]));
        }
        if mode == "position_error" && column.name != "Plan" {
            series.push(reference_ring());
        }
        if mode == "sigma" {
            series.push(guide(vec![lo, hi], vec![lo, hi]));
        }
        match draw {
            "density" => {
                for (xs, ys) in &column.clouds {
                    series.push(density_map(density_grid(xs, ys, lo, hi, lo, hi), ramp_id));
                }
            }
            "contour" => {
                for (xs, ys) in &column.clouds {
                    series.extend(contour_bands(xs, ys, cutoff));
                }
            }
            _ => {
                for (xs, ys) in &column.clouds {
                    series.push(Series::Points {
                        xs: xs.clone(),
                        ys: ys.clone(),
                        color: MARK,
                        radius: 2.0,
                    });
                }
            }
        }
        let mut top = panel(String::new(), lo, hi, lo, hi, series);
        top.equal = true;
        top.x_label = axis_name(column.name, "X", mode);
        top.y_label = axis_name(column.name, "Y", mode);
        tops.push(top);
        let xs: Vec<&[f32]> = column.clouds.iter().map(|(xs, _)| xs.as_slice()).collect();
        let ys: Vec<&[f32]> = column.clouds.iter().map(|(_, ys)| ys.as_slice()).collect();
        x_hists.push(histogram_panel(
            &axis_name(column.name, "X", mode),
            &xs,
            bins,
            true,
            Some((lo, hi)),
            &hist_guides,
        ));
        y_hists.push(histogram_panel(
            &axis_name(column.name, "Y", mode),
            &ys,
            bins,
            true,
            Some((lo, hi)),
            &hist_guides,
        ));
    }
    let count = tops.len() as u32;
    tops.extend(x_hists);
    tops.extend(y_hists);
    (tops, count, has_plan)
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
