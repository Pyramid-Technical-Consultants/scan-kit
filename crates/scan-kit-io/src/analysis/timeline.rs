use std::path::Path;

use scan_kit_core::{
    arc_fit, arc_predict, beam_angle_mrad, beam_off_edges, beam_on_mask, density_counts, fit_decay,
    hv_capacitance_pf, hv_delta_v, hv_expected_pf, hv_firmware_flags, hv_step_window, linear_fit,
    settled_after_step, Panel, PlotScene, Series,
};
use serde_json::Value;

use super::{
    channel_pairs_of, choice_control, choose, col, drew_line, finite_col, load_csv, panel,
    percentile_sorted, placed, scene, session_text, span, spot_table, stroke, timeline,
    timeslice_metric, MARK,
};

/// Timeslice rows are 1 ms apart.
const SAMPLE_S: f32 = 0.001;

pub(super) fn replay(root: &Path, session_ids: &[String], options: &Value) -> PlotScene {
    let tables = timeline(root, session_ids);
    let pairs = channel_pairs_of(&tables);
    let channel = choose(options, "channel", "ic1_current", &pairs);
    let label = pairs
        .iter()
        .find(|(id, _)| id == &channel)
        .map(|(_, label)| label.clone())
        .unwrap_or_else(|| channel.clone());
    let mut overview = Vec::new();
    let mut detail = Vec::new();
    let mut xmax = SAMPLE_S;
    let mut ymin = f32::MAX;
    let mut ymax = f32::MIN;
    for columns in &tables {
        let Some(samples) = columns.get(&channel) else {
            continue;
        };
        if samples.is_empty() {
            continue;
        }
        xmax = xmax.max(samples.len().saturating_sub(1) as f32 * SAMPLE_S);
        if let Some((lo, hi)) = robust_span(samples) {
            ymin = ymin.min(lo);
            ymax = ymax.max(hi);
        }
        let (xs, ys) = envelope(samples, 480);
        overview.push(stroke(xs, ys, false));
        let (xs, ys) = indexed(samples, 4000);
        detail.push(stroke(xs, ys, false));
    }
    let mut panels = Vec::new();
    if drew_line(&overview) {
        if ymin > ymax {
            ymin = 0.0;
            ymax = 1.0;
        }
        panels.push(time_panel("Overview", overview, xmax, ymin, ymax, &label));
        panels.push(time_panel("Detail", detail, xmax, ymin, ymax, &label));
    }
    let mut scene = scene(
        "Timeslice Replay",
        panels,
        vec![choice_control("channel", "Channel", &pairs, &channel)],
    );
    scene.columns = 1;
    scene
}

pub(super) fn rampdown(root: &Path, session_ids: &[String]) -> PlotScene {
    let mut panels = Vec::new();
    for (label, key) in [
        ("IC1", "ic1_current"),
        ("IC2", "ic2_current"),
        ("IC3", "ic3_current"),
    ] {
        let mut series = Vec::new();
        let mut windows_panels = Vec::new();
        for session in session_ids {
            let table = timeslice_metric(root, session, "ic_current");
            let Some(samples) = col(&table, key) else {
                continue;
            };
            let mut edges = beam_off_edges(samples);
            if edges.is_empty() {
                let on = col(&table, "beam_on")
                    .map(beam_on_mask)
                    .unwrap_or_else(|| vec![true; samples.len()]);
                if let Some(end) = on.iter().rposition(|flag| *flag) {
                    edges.push(end);
                }
            }
            let mut windows = Vec::new();
            for edge in edges {
                if let Some(window) = ramp_window(samples, edge) {
                    windows.push(window);
                }
            }
            if windows.is_empty() {
                continue;
            }
            let width = windows.iter().map(Vec::len).max().unwrap_or(0);
            let mut xs = Vec::new();
            let mut ys = Vec::new();
            let mut heat = vec![0.0f32; width * windows.len()];
            for (row, window) in windows.iter().enumerate() {
                if !xs.is_empty() {
                    xs.push(f32::NAN);
                    ys.push(f32::NAN);
                }
                for (i, value) in window.iter().enumerate() {
                    xs.push(i as f32);
                    ys.push(*value);
                    heat[i + width * row] = *value;
                }
            }
            series.push(stroke(xs, ys, false));
            let time: Vec<f32> = (0..width).map(|i| i as f32).collect();
            let mean = mean_window(&windows, width);
            if let Some((amp, tau)) = fit_decay(&time, &mean) {
                let fit: Vec<f32> = time.iter().map(|t| amp * (-t / tau).exp()).collect();
                series.push(stroke(time, fit, true));
            }
            windows_panels.push(panel(
                format!("{session} {label} Windows"),
                0.0,
                width as f32,
                0.0,
                windows.len() as f32,
                vec![Series::heatmap(heat, width as u32, windows.len() as u32)],
            ));
        }
        if drew_line(&series) {
            panels.push(placed(label.into(), series));
        }
        panels.extend(windows_panels);
    }
    scene("Beam-Off Ramp-Down", panels, Vec::new())
}

pub(super) fn amplifier(root: &Path, session_ids: &[String]) -> PlotScene {
    let mut panels = Vec::new();
    for (title, cmd_key, read_key) in [
        ("X", "amp_cmd_x", "amp_read_x"),
        ("Y", "amp_cmd_y", "amp_read_y"),
    ] {
        let mut series = Vec::new();
        let mut arcs = Vec::new();
        for session in session_ids {
            let table = timeslice_metric(root, session, "amplifier_error");
            let Some(cmd) = col(&table, cmd_key) else {
                continue;
            };
            let Some(readback) = col(&table, read_key) else {
                continue;
            };
            let n = cmd.len().min(readback.len()).min(2000);
            if n == 0 {
                continue;
            }
            let mask = settled_after_step(&cmd[..n], 3, 0.05);
            let mut xs = Vec::new();
            let mut ys = Vec::new();
            for i in 0..n {
                if mask[i] && cmd[i].is_finite() && readback[i].is_finite() {
                    xs.push(cmd[i]);
                    ys.push(readback[i]);
                }
            }
            if xs.len() < 2 {
                xs.clear();
                ys.clear();
                for i in 0..n {
                    if cmd[i].is_finite() && readback[i].is_finite() {
                        xs.push(cmd[i]);
                        ys.push(readback[i]);
                    }
                }
            }
            if xs.len() < 2 {
                continue;
            }
            let (xmin, xmax) = span(&xs);
            series.push(Series::Points {
                xs: xs.clone(),
                ys: ys.clone(),
                color: MARK,
                radius: 1.5,
            });
            if let Some((slope, intercept)) = linear_fit(&xs, &ys) {
                series.push(stroke(
                    vec![xmin, xmax],
                    vec![intercept + slope * xmin, intercept + slope * xmax],
                    true,
                ));
            }
            if let Some((counts, x0, x1, y0, y1)) = density_counts(&xs, &ys, 24) {
                panels.push(panel(
                    format!("{session} {title} Density"),
                    x0,
                    x1,
                    y0,
                    y1,
                    vec![Series::heatmap(counts, 24, 24)],
                ));
            }
            let spots = spot_table(root, session);
            let ic1 = finite_col(&spots, "ic1_x");
            let ic2 = finite_col(&spots, "ic2_x");
            if let (Some(ic1), Some(ic2)) = (ic1, ic2) {
                let n = ic1.len().min(ic2.len()).min(cmd.len());
                let angle: Vec<f32> = (0..n).map(|i| beam_angle_mrad(ic1[i], ic2[i])).collect();
                if let Some(fit) = arc_fit(&cmd[..n], &angle) {
                    let mut curve_x = Vec::new();
                    let mut curve_y = Vec::new();
                    for step in 0..40 {
                        let field = xmin + (xmax - xmin) * step as f32 / 39.0;
                        curve_x.push(field);
                        curve_y.push(arc_predict(fit, field));
                    }
                    arcs.push(stroke(curve_x, curve_y, false));
                }
            }
        }
        if !series.is_empty() {
            panels.insert(0, placed(title.into(), series));
        }
        if drew_line(&arcs) {
            panels.push(placed(format!("{title} Arc"), arcs));
        }
    }
    scene("Amplifier Command Correlations", panels, Vec::new())
}

pub(super) fn hv_transient(root: &Path, session_ids: &[String]) -> PlotScene {
    let mut panels = Vec::new();
    for session in session_ids {
        for device in ["IC1", "IC2", "IC3"] {
            let path = format!("ic_hv_toggle/{device}_HCC.csv");
            let columns = load_csv(root, session, &path);
            let Some((_, current)) = columns
                .iter()
                .find(|(name, _)| name.to_ascii_lowercase().contains("current"))
            else {
                continue;
            };
            let time = col(&columns, "time")
                .map(|values| values.to_vec())
                .unwrap_or_else(|| (0..current.len()).map(|i| i as f32).collect());
            let (t0, t1) = hv_step_window(&time, current).unwrap_or((
                time.first().copied().unwrap_or(0.0),
                time.last().copied().unwrap_or(0.0),
            ));
            let config = hv_config_text(root, session, device);
            let delta_v = config.as_deref().and_then(hv_delta_v).unwrap_or(80.0);
            let measured = hv_capacitance_pf(&time, current, t0, t1, delta_v).unwrap_or(0.0);
            let expected = config.as_deref().and_then(hv_expected_pf);
            let result = session_text(root, session, &format!("ic_hv_toggle/{device}_result.json"));
            let (overall, flags) = hv_firmware_flags(&result);
            let grade = overall.unwrap_or_else(|| "ungraded".into());
            let fails = flags.iter().filter(|flag| flag.as_str() == "fail").count();
            panels.push(line_panel(
                format!("{device} {grade}"),
                &time,
                current,
                MARK,
            ));
            let mut counts = vec![measured];
            let mut edges = vec![0.0, 1.0];
            if let Some(expected) = expected {
                counts.push(expected);
                edges.push(2.0);
            }
            let ymax = counts.iter().copied().fold(1.0f32, f32::max);
            panels.push(panel(
                format!("{device} {measured:.1} pF fails {fails}"),
                0.0,
                (counts.len()) as f32,
                0.0,
                ymax,
                vec![Series::Bars {
                    edges,
                    counts,
                    color: MARK,
                }],
            ));
        }
    }
    scene("IC HV Transient Test", panels, Vec::new())
}

fn line_panel(title: String, xs: &[f32], ys: &[f32], color: [f32; 4]) -> Panel {
    let (xmin, xmax) = span(xs);
    let (ymin, ymax) = span(ys);
    panel(
        title,
        xmin,
        xmax,
        ymin,
        ymax,
        vec![Series::Polyline {
            xs: xs.to_vec(),
            ys: ys.to_vec(),
            color,
            thickness: 1.5,
        }],
    )
}

fn time_panel(
    title: &str,
    series: Vec<Series>,
    xmax: f32,
    ymin: f32,
    ymax: f32,
    y_label: &str,
) -> Panel {
    let mut panel = panel(title.into(), 0.0, xmax, ymin, ymax, series);
    panel.y_label = y_label.to_owned();
    panel
}

fn indexed(samples: &[f32], target: usize) -> (Vec<f32>, Vec<f32>) {
    let step = (samples.len() / target.max(1)).max(1);
    (
        (0..samples.len())
            .step_by(step)
            .map(|i| i as f32 * SAMPLE_S)
            .collect(),
        (0..samples.len())
            .step_by(step)
            .map(|i| samples[i])
            .collect(),
    )
}

/// Min and max of each bucket, so a pulse narrower than the stride still draws.
pub(super) fn envelope(samples: &[f32], buckets: usize) -> (Vec<f32>, Vec<f32>) {
    let n = samples.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    let buckets = buckets.max(1).min(n);
    let mut xs = Vec::with_capacity(buckets * 2);
    let mut ys = Vec::with_capacity(buckets * 2);
    for bucket in 0..buckets {
        let start = bucket * n / buckets;
        let end = ((bucket + 1) * n / buckets).max(start + 1).min(n);
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for value in &samples[start..end] {
            if value.is_finite() {
                lo = lo.min(*value);
                hi = hi.max(*value);
            }
        }
        if !lo.is_finite() {
            continue;
        }
        let x = start as f32 * SAMPLE_S;
        xs.push(x);
        ys.push(lo);
        if hi > lo {
            xs.push(x);
            ys.push(hi);
        }
    }
    (xs, ys)
}

/// 0.5% tails. One ADC spike was setting the axis and the trace sat on the frame.
pub(super) fn robust_span(samples: &[f32]) -> Option<(f32, f32)> {
    let mut values: Vec<f32> = samples
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    if values.len() < 2 {
        return None;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let lo = percentile_sorted(&values, 0.005);
    let hi = percentile_sorted(&values, 0.995);
    let pad = ((hi - lo) * 0.06).max(1.0e-4);
    Some((lo - pad, hi + pad))
}

fn ramp_window(samples: &[f32], edge: usize) -> Option<Vec<f32>> {
    let start = edge.saturating_sub(2);
    let stop = (edge + 9).min(samples.len());
    if stop <= start {
        return None;
    }
    let window = &samples[start..stop];
    let peak = window
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .fold(0.0f32, f32::max);
    if peak < 1e-6 {
        return None;
    }
    Some(
        window
            .iter()
            .map(|value| if value.is_finite() { value / peak } else { 0.0 })
            .collect(),
    )
}

fn mean_window(windows: &[Vec<f32>], width: usize) -> Vec<f32> {
    let mut mean = vec![0.0f32; width];
    let mut count = vec![0.0f32; width];
    for window in windows {
        for (i, value) in window.iter().enumerate() {
            if value.is_finite() {
                mean[i] += value;
                count[i] += 1.0;
            }
        }
    }
    for i in 0..width {
        if count[i] > 0.0 {
            mean[i] /= count[i];
        }
    }
    mean
}

fn hv_config_text(root: &Path, session: &str, device: &str) -> Option<String> {
    let relative = format!("config/nozzle/{device}/config.json");
    let from_session = session_text(root, session, &relative);
    if !from_session.is_empty() {
        return Some(from_session);
    }
    std::fs::read_to_string(
        root.join("config")
            .join("nozzle")
            .join(device)
            .join("config.json"),
    )
    .ok()
}
