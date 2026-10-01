//! Analysis view workflows. Each one loads columns and returns a plot scene.
//! Pixels are rendered later by `scan-kit-compute`.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use scan_kit_core::{
    arc_fit, arc_predict, beam_angle_mrad, beam_off_edges, beam_on_mask, calibration_factor,
    compare_templates, coverage_percent, cumsum, density_counts, fit_decay, fit_iso_plane,
    histogram, hv_capacitance_pf, hv_delta_v, hv_expected_pf, hv_firmware_flags, hv_step_window,
    linear_fit, magnet_pivot_z, parse_session_log, resolve_concept_column, scale_column,
    settled_after_step, spill_segments, welch_psd, BeamState, Control, DataTable, Panel, PlotScene,
    Series, IC1_Z_MM, IC2_Z_MM, MIN_SPILL_GAP_MS,
};
use serde_json::{json, Value};

use super::binned::{
    apply_filter, contour_bands, labeled, pick, slice_table, spot_table, timeslice_metric,
    timeslice_signals, BEAM_CHOICES,
};
use super::discover;

/// A session series. `apply_palette` replaces the RGB and keeps this alpha.
const MARK: [f32; 4] = [0.0, 0.0, 0.0, 1.0];
/// A companion of the previous series. Alpha 0 copies that session's color.
const LINKED: [f32; 4] = [0.0, 0.0, 0.0, 0.0];
const GUIDE_COLOR: [f32; 4] = [0.62, 0.62, 0.62, 0.9];

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
const RAMP_CHOICES: &[(&str, &str)] = &[("turbo", "Turbo"), ("viridis", "Viridis")];
const CUTOFF_CHOICES: &[(&str, &str)] = &[("0", "0"), ("5", "5"), ("10", "10"), ("20", "20")];
const DENSITY_BINS: usize = 80;
const CALIBRATE_CHOICES: &[(&str, &str)] = &[("off", "Off"), ("on", "On")];

pub fn analysis_scene(
    view: &str,
    root: &Path,
    session_ids: &[String],
    options: &Value,
) -> Result<PlotScene, String> {
    if session_ids.is_empty() {
        return Err("select a session first".to_owned());
    }
    if session_ids.len() > 5 {
        return Err("at most five sessions can be selected".to_owned());
    }
    match view {
        "dose_accumulation" => Ok(dose_accumulation(root, session_ids, options)),
        "ic_peak_amplitude_beam_off" => Ok(peak_amplitude(root, session_ids)),
        "beam_motion_energy" => Ok(beam_motion(root, session_ids)),
        "distribution" => Ok(distribution(root, session_ids, options)),
        "binned_summary" => Ok(binned_summary(root, session_ids, options)),
        "timeslice_replay" => Ok(replay(root, session_ids, options)),
        "ic_fft_analysis" => Ok(fft_view(root, session_ids, options)),
        "ic_audio_player" => Ok(audio_view(root, session_ids, options)),
        "beam_off_rampdown" => Ok(rampdown(root, session_ids)),
        "amplifier_correlation" => Ok(amplifier(root, session_ids)),
        "ic_hv_transient" => Ok(hv_transient(root, session_ids)),
        "session_log_compare" => Ok(session_log(root, session_ids)),
        "trajectory" => Ok(trajectory(root, session_ids, options)),
        "dose_volume" => Ok(crate::dose_view::dose_volume(
            root,
            session_ids,
            options,
            None,
        )),
        _ => Err(format!("unknown view {view}")),
    }
}

pub fn channel_catalog(root: &Path, session_id: &str) -> Vec<String> {
    session_channels(root, session_id)
        .into_iter()
        .filter(|(name, values)| channel_key(name) && values.iter().any(|value| value.is_finite()))
        .map(|(name, _)| name)
        .collect()
}

pub fn load_timeslice_columns(root: &Path, session_id: &str) -> Value {
    let columns = load_timeslice(root, session_id);
    let mut listed = Vec::new();
    for (name, values) in &columns {
        listed.push(json!({ "name": name, "len": values.len() }));
    }
    let energy = energy_lookup(root, session_id);
    json!({
        "session_id": session_id,
        "columns": listed,
        "energy_layers": energy.len(),
    })
}

fn dose_accumulation(root: &Path, session_ids: &[String], options: &Value) -> PlotScene {
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

fn peak_amplitude(root: &Path, session_ids: &[String]) -> PlotScene {
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

fn beam_motion(root: &Path, session_ids: &[String]) -> PlotScene {
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

fn distribution(root: &Path, session_ids: &[String], options: &Value) -> PlotScene {
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
    let ramp = pick(options, "ramp", "turbo", RAMP_CHOICES);
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
            controls.push(labeled("ramp", "Ramp", RAMP_CHOICES, ramp));
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
    scene
}

fn binned_summary(root: &Path, session_ids: &[String], options: &Value) -> PlotScene {
    super::binned::binned_summary(root, session_ids, options)
}

/// Timeslice rows are 1 ms apart.
const SAMPLE_S: f32 = 0.001;

fn replay(root: &Path, session_ids: &[String], options: &Value) -> PlotScene {
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

fn fft_view(root: &Path, session_ids: &[String], options: &Value) -> PlotScene {
    spectrum(root, session_ids, options, false)
}

fn audio_view(root: &Path, session_ids: &[String], options: &Value) -> PlotScene {
    spectrum(root, session_ids, options, true)
}

fn spectrum(root: &Path, session_ids: &[String], options: &Value, audio: bool) -> PlotScene {
    let tables = timeline(root, session_ids);
    let pairs = channel_pairs_of(&tables);
    let channel = choose(options, "channel", "ic1_current", &pairs);
    let mut series = Vec::new();
    let mut played = Vec::new();
    for (index, columns) in tables.iter().enumerate() {
        let Some(samples) = columns.get(&channel) else {
            continue;
        };
        let (freqs, psd) = welch_psd(samples, 1000.0, 4096, 0.5);
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for (freq, power) in freqs.iter().zip(&psd) {
            if *freq >= 1.0 && *freq <= 500.0 {
                xs.push(*freq);
                // A linear PSD sits on the axis. Decades match the Python explorer.
                ys.push(if power.is_finite() && *power > 0.0 {
                    power.log10()
                } else {
                    f32::NAN
                });
            }
        }
        if ys.iter().any(|value| value.is_finite()) {
            series.push(stroke(xs, ys, false));
        }
        if audio && index == 0 {
            played = audible(samples);
        }
    }
    let mut panels = if series.is_empty() {
        Vec::new()
    } else {
        vec![placed("Spectrum".into(), series)]
    };
    if let Some(panel) = panels.first_mut() {
        panel.y_label = "log10 PSD".into();
    }
    let mut scene = scene(
        if audio {
            "Audio Explorer"
        } else {
            "FFT Explorer"
        },
        panels,
        vec![choice_control("channel", "Channel", &pairs, &channel)],
    );
    scene.samples = played;
    scene
}

fn audible(samples: &[f32]) -> Vec<f32> {
    let peak = samples
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .fold(0.0f32, |max, value| max.max(value.abs()))
        .max(1e-6);
    samples
        .iter()
        .take(8000)
        .map(|value| (value / peak).clamp(-1.0, 1.0))
        .collect()
}

fn rampdown(root: &Path, session_ids: &[String]) -> PlotScene {
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

fn amplifier(root: &Path, session_ids: &[String]) -> PlotScene {
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

fn hv_transient(root: &Path, session_ids: &[String]) -> PlotScene {
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

fn session_log(root: &Path, session_ids: &[String]) -> PlotScene {
    let mut rows = Vec::new();
    let mut parsed = Vec::new();
    for session in session_ids {
        let text = session_text(root, session, "SessionLogFile.log");
        let log = parse_session_log(&text);
        rows.push(vec![
            session.clone(),
            "Overview".into(),
            format!("{} lines", log.lines),
        ]);
        for (level, count) in &log.level_counts {
            rows.push(vec![
                session.clone(),
                "Overview".into(),
                format!("{level} {count}"),
            ]);
        }
        for event in &log.timeline {
            rows.push(vec![
                session.clone(),
                "Timeline".into(),
                format!("layer {} {} {:.3}s", event.layer, event.kind, event.seconds),
            ]);
        }
        for issue in log.issues.iter().take(40) {
            rows.push(vec![session.clone(), "Errors".into(), issue.clone()]);
        }
        for (device, count) in &log.wdt {
            rows.push(vec![
                session.clone(),
                "Watchdog".into(),
                format!("{device} {count}"),
            ]);
        }
        parsed.push((session.clone(), log));
    }
    if parsed.len() == 2 {
        for (template, a, b, delta) in compare_templates(&parsed[0].1, &parsed[1].1)
            .into_iter()
            .take(40)
        {
            rows.push(vec![
                format!("{} vs {}", parsed[0].0, parsed[1].0),
                "Diff".into(),
                format!("{template} {a} {b} {delta}"),
            ]);
        }
    }
    let mut scene = scene("Session Log Compare", Vec::new(), Vec::new());
    scene.table = Some(DataTable {
        columns: vec!["Session".into(), "Section".into(), "Detail".into()],
        rows,
    });
    scene
}

fn trajectory(root: &Path, session_ids: &[String], options: &Value) -> PlotScene {
    let azimuth = number_option(options, "azimuth", 0.4);
    let orbit = ["0.2", "0.4", "0.8", "1.2"]
        .into_iter()
        .find(|label| *label == format!("{azimuth:.1}"))
        .unwrap_or("0.4");
    let azimuth: f32 = orbit.parse().unwrap_or(0.4);
    let mut paths = Vec::new();
    let mut iso = Vec::new();
    let mut pivot = 0.0f32;
    let mut have_pivot = false;
    for session in session_ids {
        let table = spot_table(root, session);
        let plan_x = col(&table, "plan_x").unwrap_or(&[]);
        let plan_y = col(&table, "plan_y").unwrap_or(&[]);
        let ic1_x = finite_col(&table, "ic1_x").unwrap_or(plan_x);
        let ic1_y = finite_col(&table, "ic1_y").unwrap_or(plan_y);
        let ic2_x = finite_col(&table, "ic2_x").unwrap_or(plan_x);
        let ic2_y = finite_col(&table, "ic2_y").unwrap_or(plan_y);
        let energy = col(&table, "energy").unwrap_or(&[]);
        let n = ic1_x
            .len()
            .min(ic2_x.len())
            .min(ic1_y.len())
            .min(ic2_y.len());
        let mut sx = Vec::new();
        let mut sy = Vec::new();
        for i in 0..n.min(400) {
            let (px, py) = project(ic2_x[i], ic2_y[i], IC2_Z_MM, azimuth);
            let (qx, qy) = project(ic1_x[i], ic1_y[i], IC1_Z_MM, azimuth);
            sx.extend([px, qx, f32::NAN]);
            sy.extend([py, qy, f32::NAN]);
        }
        if sx.iter().any(|value| value.is_finite()) {
            if !have_pivot {
                pivot = magnet_pivot_z(
                    ic2_x.first().copied().unwrap_or(0.0),
                    ic1_x.first().copied().unwrap_or(0.0),
                );
                have_pivot = true;
            }
            paths.push(stroke(sx, sy, false));
        }
        let fit_x = if finite_col(&table, "ic1_x").is_some() {
            ic1_x
        } else {
            plan_x
        };
        if let Some((intercept, slope)) = fit_iso_plane(energy, fit_x) {
            let (e0, e1) = span(energy);
            iso.push(stroke(
                vec![e0, e1],
                vec![intercept + slope * e0, intercept + slope * e1],
                false,
            ));
        }
    }
    if have_pivot {
        paths.push(plane_guide(
            (-40.0, 0.0, IC2_Z_MM),
            (40.0, 0.0, IC2_Z_MM),
            azimuth,
        ));
        paths.push(plane_guide(
            (-40.0, 0.0, IC1_Z_MM),
            (40.0, 0.0, IC1_Z_MM),
            azimuth,
        ));
        paths.push(plane_guide(
            (-20.0, -4.0, pivot),
            (20.0, -4.0, pivot),
            azimuth,
        ));
        paths.push(plane_guide(
            (-20.0, 4.0, pivot),
            (20.0, 4.0, pivot),
            azimuth,
        ));
    }
    let mut panels = Vec::new();
    if !paths.is_empty() {
        panels.push(placed(format!("Planes Pivot {pivot:.0} mm"), paths));
    }
    if drew_line(&iso) {
        panels.push(placed("Iso".into(), iso));
    }
    scene(
        "IC Beam Trajectory",
        panels,
        vec![control(
            "azimuth",
            "Orbit",
            &["0.2", "0.4", "0.8", "1.2"],
            orbit,
        )],
    )
}

fn scene(title: &str, panels: Vec<Panel>, controls: Vec<Control>) -> PlotScene {
    PlotScene {
        title: title.into(),
        panels,
        controls,
        table: None,
        samples: Vec::new(),
        columns: 0,
        column_weights: Vec::new(),
    }
}

fn panel(title: String, xmin: f32, xmax: f32, ymin: f32, ymax: f32, series: Vec<Series>) -> Panel {
    let xmax = if (xmax - xmin).abs() < 1e-3 {
        xmin + 1.0
    } else {
        xmax
    };
    let ymax = if (ymax - ymin).abs() < 1e-3 {
        ymin + 1.0
    } else {
        ymax
    };
    Panel {
        title,
        y_label: String::new(),
        xmin,
        xmax,
        ymin,
        ymax,
        series,
        x_labels: Vec::new(),
        equal: false,
    }
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

fn control(id: &str, label: &str, options: &[&str], value: &str) -> Control {
    Control {
        id: id.into(),
        label: label.into(),
        options: options.iter().map(|option| (*option).to_owned()).collect(),
        value: value.into(),
    }
}

fn stroke(xs: Vec<f32>, ys: Vec<f32>, linked: bool) -> Series {
    Series::Polyline {
        xs,
        ys,
        color: if linked { LINKED } else { MARK },
        thickness: 1.5,
    }
}

fn guide(xs: Vec<f32>, ys: Vec<f32>) -> Series {
    Series::Guide {
        xs,
        ys,
        color: GUIDE_COLOR,
        thickness: 1.0,
    }
}

fn placed(title: String, series: Vec<Series>) -> Panel {
    let (xmin, xmax, ymin, ymax) = series_span(&series);
    panel(title, xmin, xmax, ymin, ymax, series)
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

fn drew_line(series: &[Series]) -> bool {
    series.iter().any(|item| {
        matches!(item, Series::Polyline { ys, .. } if ys.iter().any(|value| value.is_finite()))
    })
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

fn finite_col<'a>(table: &'a BTreeMap<String, Vec<f32>>, key: &str) -> Option<&'a [f32]> {
    col(table, key).filter(|values| values.iter().any(|value| value.is_finite()))
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

fn plane_guide(from: (f32, f32, f32), to: (f32, f32, f32), azimuth: f32) -> Series {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    push_projected(&mut xs, &mut ys, from, to, azimuth);
    guide(xs, ys)
}

fn span(values: &[f32]) -> (f32, f32) {
    let mut lo = f32::MAX;
    let mut hi = f32::MIN;
    for value in values.iter().copied().filter(|v| v.is_finite()) {
        lo = lo.min(value);
        hi = hi.max(value);
    }
    if !lo.is_finite() {
        (0.0, 1.0)
    } else if (hi - lo).abs() < 1e-4 {
        (lo - 0.5, hi + 0.5)
    } else {
        (lo, hi)
    }
}

fn project(x: f32, y: f32, z: f32, azimuth: f32) -> (f32, f32) {
    let px = x * azimuth.cos() - z * azimuth.sin();
    let py = y + z * 0.15;
    (px, py)
}

fn number_option(options: &Value, key: &str, default: f32) -> f32 {
    options
        .get(key)
        .and_then(|value| {
            value
                .as_f64()
                .or_else(|| value.as_str().and_then(|text| text.parse().ok()))
        })
        .map(|value| value as f32)
        .unwrap_or(default)
}

fn session_dir(root: &Path, session_id: &str) -> PathBuf {
    discover::session_directory(root, session_id)
}

fn load_csv(root: &Path, session_id: &str, name: &str) -> BTreeMap<String, Vec<f32>> {
    discover::read_session_file(root, session_id, name)
        .and_then(|bytes| numeric_columns(&bytes).ok())
        .unwrap_or_default()
}

fn load_timeslice(root: &Path, session_id: &str) -> BTreeMap<String, Vec<f32>> {
    let mut merged: BTreeMap<String, Vec<f32>> = BTreeMap::new();
    for bytes in discover::read_timeslices(&session_dir(root, session_id)) {
        let Ok(frame) = numeric_columns(&bytes) else {
            continue;
        };
        for (name, values) in frame {
            merged.entry(name).or_default().extend(values);
        }
    }
    merged
}

fn energy_lookup(root: &Path, session_id: &str) -> Vec<f32> {
    col(&load_csv(root, session_id, "input_map.csv"), "energy")
        .unwrap_or(&[])
        .to_vec()
}

fn col<'a>(columns: &'a BTreeMap<String, Vec<f32>>, concept: &str) -> Option<&'a [f32]> {
    let names: Vec<String> = columns.keys().cloned().collect();
    let name = resolve_concept_column(&names, concept).or_else(|| {
        names
            .iter()
            .find(|name| name.eq_ignore_ascii_case(concept))
            .map(String::as_str)
    })?;
    columns.get(name).map(Vec::as_slice)
}

fn numeric_columns(bytes: &[u8]) -> Result<BTreeMap<String, Vec<f32>>, String> {
    let mut reader = csv::ReaderBuilder::new().flexible(true).from_reader(bytes);
    let headers = reader.headers().map_err(|err| err.to_string())?.clone();
    let mut columns: BTreeMap<String, Vec<f32>> = headers
        .iter()
        .map(|name| (name.to_owned(), Vec::new()))
        .collect();
    for record in reader.records() {
        let record = record.map_err(|err| err.to_string())?;
        for (index, name) in headers.iter().enumerate() {
            let value = record
                .get(index)
                .and_then(|text| text.trim().parse().ok())
                .unwrap_or(f32::NAN);
            if let Some(column) = columns.get_mut(name) {
                column.push(value);
            }
        }
    }
    Ok(columns)
}

fn session_text(root: &Path, session_id: &str, name: &str) -> String {
    discover::read_session_file(root, session_id, name)
        .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
        .unwrap_or_default()
}

fn session_channels(root: &Path, session: &str) -> BTreeMap<String, Vec<f32>> {
    timeslice_signals(root, session)
}

fn timeline(root: &Path, session_ids: &[String]) -> Vec<BTreeMap<String, Vec<f32>>> {
    session_ids
        .iter()
        .map(|session| session_channels(root, session))
        .collect()
}

fn channel_key(name: &str) -> bool {
    !matches!(name, "energy" | "beam_on" | "beam_on_time" | "spot_time")
}

fn channel_label(id: &str) -> String {
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
    .to_owned()
}

fn channel_pairs_of(tables: &[BTreeMap<String, Vec<f32>>]) -> Vec<(String, String)> {
    let mut ids = BTreeSet::new();
    for table in tables {
        for (name, values) in table {
            if channel_key(name) && values.iter().any(|value| value.is_finite()) {
                ids.insert(name.clone());
            }
        }
    }
    if ids.is_empty() {
        ids.insert("ic1_current".to_owned());
    }
    ids.into_iter()
        .map(|id| {
            let label = channel_label(&id);
            (id, label)
        })
        .collect()
}

fn choose(options: &Value, key: &str, default_id: &str, pairs: &[(String, String)]) -> String {
    let raw = options
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or(default_id);
    if let Some((id, _)) = pairs.iter().find(|(id, label)| id == raw || label == raw) {
        return id.clone();
    }
    if pairs.iter().any(|(id, _)| id == default_id) {
        default_id.to_owned()
    } else {
        pairs
            .first()
            .map(|(id, _)| id.clone())
            .unwrap_or_else(|| default_id.to_owned())
    }
}

fn choice_control(id: &str, label: &str, pairs: &[(String, String)], current: &str) -> Control {
    let value = pairs
        .iter()
        .find(|(key, _)| key == current)
        .map(|(_, label)| label.clone())
        .unwrap_or_else(|| current.to_owned());
    Control {
        id: id.to_owned(),
        label: label.to_owned(),
        options: pairs.iter().map(|(_, label)| label.clone()).collect(),
        value,
    }
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
fn envelope(samples: &[f32], buckets: usize) -> (Vec<f32>, Vec<f32>) {
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
fn robust_span(samples: &[f32]) -> Option<(f32, f32)> {
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

const HIST_BINS: usize = 101;
/// Histogram bars stay translucent so overlaid sessions both stay visible.
const HIST: [f32; 4] = [0.0, 0.0, 0.0, 0.55];

fn distribution_limits(mode: &str, samples: &[f32]) -> (f32, f32) {
    if mode == "sigma" {
        let mut positive: Vec<f32> = samples
            .iter()
            .copied()
            .filter(|value| *value > 0.0)
            .collect();
        positive.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let hi = percentile_sorted(&positive, 0.9995).max(1.0);
        return (0.0, hi);
    }
    if mode == "position_error" {
        let mut abs: Vec<f32> = samples
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .map(f32::abs)
            .collect();
        abs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let bound = percentile_sorted(&abs, 0.9995).max(1.0);
        return (-bound, bound);
    }
    let mut finite: Vec<f32> = samples
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    finite.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if finite.is_empty() {
        return (-1.0, 1.0);
    }
    let lo = percentile_sorted(&finite, 0.0005);
    let hi = percentile_sorted(&finite, 0.9995);
    let mid = 0.5 * (lo + hi);
    let half = (mid - lo).max(hi - mid).max(0.5);
    (mid - half, mid + half)
}

fn percentile_sorted(values: &[f32], p: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let index = ((values.len() - 1) as f32 * p).round() as usize;
    values[index.min(values.len() - 1)]
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
        2
    } else if ramp == "viridis" {
        0
    } else {
        1
    }
}

fn density_map(values: Vec<f32>, ramp: u8) -> Series {
    let bins = DENSITY_BINS as u32;
    Series::Heatmap {
        values,
        cols: bins,
        rows: bins,
        ramp,
        color: if ramp == 2 {
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

fn reference_ring() -> Series {
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

fn series_span(series: &[Series]) -> (f32, f32, f32, f32) {
    let mut xmin = f32::MAX;
    let mut xmax = f32::MIN;
    let mut ymin = f32::MAX;
    let mut ymax = f32::MIN;
    for item in series {
        let (xs, ys) = match item {
            Series::Polyline { xs, ys, .. }
            | Series::Points { xs, ys, .. }
            | Series::Guide { xs, ys, .. } => (xs, ys),
            _ => continue,
        };
        for (x, y) in xs.iter().zip(ys) {
            if x.is_finite() && y.is_finite() {
                xmin = xmin.min(*x);
                xmax = xmax.max(*x);
                ymin = ymin.min(*y);
                ymax = ymax.max(*y);
            }
        }
    }
    if xmin > xmax {
        (0.0, 1.0, 0.0, 1.0)
    } else {
        (xmin, xmax, ymin, ymax)
    }
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

fn push_projected(
    xs: &mut Vec<f32>,
    ys: &mut Vec<f32>,
    from: (f32, f32, f32),
    to: (f32, f32, f32),
    azimuth: f32,
) {
    if xs.iter().any(|value| value.is_finite()) {
        xs.push(f32::NAN);
        ys.push(f32::NAN);
    }
    let (a, b) = project(from.0, from.1, from.2, azimuth);
    let (c, d) = project(to.0, to.1, to.2, azimuth);
    xs.extend([a, c]);
    ys.extend([b, d]);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn write_session(root: &Path) {
        let session = root.join("sess");
        std::fs::create_dir_all(&session).unwrap();
        std::fs::write(
            session.join("input_map.csv"),
            "energy,charge_req,position_x,position_y\n70,1,0,0\n90,2,5,1\n110,3,10,2\n",
        )
        .unwrap();
        std::fs::write(
            session.join("spot_data.csv"),
            "ic1_total_dose,ic2_total_dose,position_x,position_y\n1.1,1.0,0,0\n1.8,2.1,5,1\n3.2,2.7,10,2\n",
        )
        .unwrap();
        let mut timeslice = String::from(
            "r_ic1_current_dose,rci_in_trigger,ic1_peak_amplitude_x,ic1_peak_amplitude_y,ic2_peak_amplitude_x,ic2_peak_amplitude_y,position_error_x,field_x,field_y,c_x,r_xV\n",
        );
        for i in 0..32 {
            let on = if (8..24).contains(&i) { 1 } else { 0 };
            let current = if on == 1 {
                2.0
            } else {
                (24 - i).max(0) as f32 * 0.05
            };
            let cmd = i as f32 * 0.1;
            timeslice.push_str(&format!(
                "{current},{on},{current},{current},1,1,{err},0.{i},0.2,{cmd},{read}\n",
                err = (i as f32) * 0.01,
                read = cmd + 0.05
            ));
        }
        std::fs::write(
            session.join("000_timeslice_data_device_units.csv"),
            timeslice,
        )
        .unwrap();
        let hv = session.join("ic_hv_toggle");
        std::fs::create_dir_all(&hv).unwrap();
        std::fs::write(
            hv.join("IC1_HCC.csv"),
            "time,current\n0,0\n1,10\n2,2\n3,0\n",
        )
        .unwrap();
        std::fs::write(
            session.join("SessionLogFile.log"),
            "2026-01-01 00:00:00,000 INFO [dcs] START MAP FOR LAYER: 1 - TIMELINE(begin): T=0.1s\n2026-01-01 00:00:01,000 ERROR [dcs] fault\n",
        )
        .unwrap();
    }

    #[test]
    fn sk_req_012_dose_accumulation_scene_has_cumulative_lines() {
        let root = std::env::temp_dir().join(format!("scan-kit-view-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        write_session(&root);
        let scene =
            analysis_scene("dose_accumulation", &root, &["sess".into()], &json!({})).unwrap();
        assert!(scene.panels.iter().any(|panel| panel.series.iter().any(|series| matches!(series, Series::Polyline { ys, .. } if ys.iter().any(|y| y.is_finite())))));
        assert!(scene
            .controls
            .iter()
            .any(|control| control.id == "calibrate"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn sk_req_011_timeslice_load_reports_columns() {
        let root = std::env::temp_dir().join(format!("scan-kit-ts-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        write_session(&root);
        let loaded = load_timeslice_columns(&root, "sess");
        assert!(loaded["columns"].as_array().unwrap().len() >= 2);
        assert!(!channel_catalog(&root, "sess").is_empty());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn sk_req_024_session_log_table() {
        let root = std::env::temp_dir().join(format!("scan-kit-log-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        write_session(&root);
        let scene =
            analysis_scene("session_log_compare", &root, &["sess".into()], &json!({})).unwrap();
        let table = scene.table.unwrap();
        assert!(table.rows.iter().any(|row| row[1] == "Errors"));
        assert!(table.rows.iter().any(|row| row[1] == "Timeline"));
        let _ = std::fs::remove_dir_all(&root);
    }

    fn scene_of(view: &str) -> PlotScene {
        let root = std::env::temp_dir().join(format!("scan-kit-{view}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        write_session(&root);
        let scene = analysis_scene(view, &root, &["sess".into()], &json!({})).unwrap();
        let _ = std::fs::remove_dir_all(&root);
        scene
    }

    fn has_kind(scene: &PlotScene, kind: &str) -> bool {
        scene.panels.iter().any(|panel| {
            panel.series.iter().any(|series| {
                matches!(
                    (kind, series),
                    ("line", Series::Polyline { .. })
                        | ("bars", Series::Bars { .. })
                        | ("points", Series::Points { .. })
                        | ("rects", Series::Rects { .. })
                        | ("triangles", Series::Triangles { .. })
                        | ("heat", Series::Heatmap { .. })
                )
            })
        })
    }

    #[test]
    fn sk_req_013_peak_amplitude_is_beam_off_bars() {
        assert!(has_kind(&scene_of("ic_peak_amplitude_beam_off"), "bars"));
    }

    #[test]
    fn sk_req_014_beam_motion_draws_spill_paths() {
        assert!(has_kind(&scene_of("beam_motion_energy"), "line"));
    }

    #[test]
    fn sk_req_015_distribution_exposes_spot_modes() {
        let scene = scene_of("distribution");
        assert!(has_kind(&scene, "points"));
        assert!(has_kind(&scene, "bars"));
        assert!(scene
            .panels
            .iter()
            .any(|panel| panel.title == "X Position (mm)"));
        assert!(scene
            .panels
            .iter()
            .any(|panel| panel.title == "Y Position (mm)"));
        assert!(scene
            .controls
            .iter()
            .any(|control| control.id == "mode" && control.value == "Position"));
        assert!(scene
            .controls
            .iter()
            .any(|control| control.id == "draw" && control.value == "Scatter"));
        assert!(scene
            .controls
            .iter()
            .any(|control| control.id == "beam" && control.value == "Both"));
        assert!(scene
            .controls
            .iter()
            .any(|control| control.id == "grain" && control.value == "Spot"));
        assert!(scene.controls.iter().any(|control| control.id == "draw"
            && control.options.iter().any(|option| option == "Contour")));
        assert!(scene
            .panels
            .iter()
            .filter(|panel| panel.y_label.is_empty())
            .all(|panel| panel.equal));
    }

    #[test]
    fn position_error_ring_has_radius_one_mm() {
        let Series::Guide { xs, ys, .. } = reference_ring() else {
            panic!("ring");
        };
        assert!(xs.len() >= 32);
        for (x, y) in xs.iter().zip(ys) {
            let radius = (x * x + y * y).sqrt();
            assert!((radius - 1.0).abs() < 1.0e-4, "{radius}");
        }
    }

    #[test]
    fn distribution_density_overlays_sessions_on_one_column() {
        let root = std::env::temp_dir().join(format!("scan-kit-density-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        write_session(&root);
        let from = root.join("sess");
        let to = root.join("sess-b");
        std::fs::create_dir_all(&to).unwrap();
        for name in ["input_map.csv", "spot_data.csv"] {
            std::fs::copy(from.join(name), to.join(name)).unwrap();
        }
        let scene = analysis_scene(
            "distribution",
            &root,
            &["sess".into(), "sess-b".into()],
            &json!({"draw": "Density"}),
        )
        .unwrap();
        let tops: Vec<_> = scene
            .panels
            .iter()
            .filter(|panel| panel.y_label.is_empty())
            .collect();
        assert_eq!(scene.columns as usize, tops.len());
        assert!(tops.iter().all(|panel| panel.equal));
        assert!(tops.iter().all(|panel| !panel.title.contains("sess")));
        assert!(tops.iter().any(|panel| {
            panel
                .series
                .iter()
                .filter(|series| matches!(series, Series::Heatmap { ramp: 2, .. }))
                .count()
                == 2
        }));
        assert_eq!(
            scene
                .panels
                .iter()
                .filter(|panel| panel.title == "X Position (mm)")
                .count(),
            1
        );
        assert!(scene.controls.iter().all(|control| control.id != "ramp"));
        let one = analysis_scene(
            "distribution",
            &root,
            &["sess".into()],
            &json!({"draw": "Density"}),
        )
        .unwrap();
        assert!(one
            .controls
            .iter()
            .any(|control| control.id == "ramp" && control.value == "Turbo"));
        assert!(one.panels.iter().any(|panel| {
            panel
                .series
                .iter()
                .any(|series| matches!(series, Series::Heatmap { ramp: 1, .. }))
        }));
        let contour = analysis_scene(
            "distribution",
            &root,
            &["sess".into()],
            &json!({"draw": "Contour"}),
        )
        .unwrap();
        assert!(contour
            .controls
            .iter()
            .any(|control| control.id == "cutoff" && control.value == "5"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn sk_req_017_binned_summary_and_replay_share_the_session() {
        let binned = scene_of("binned_summary");
        assert!(has_kind(&binned, "triangles"));
        assert!(binned
            .controls
            .iter()
            .any(|control| control.id == "glyph" && control.value == "Violin"));
        assert!(binned
            .controls
            .iter()
            .any(|control| control.id == "metric" && control.value == "Dose Error (%)"));
        let ic1 = binned
            .panels
            .iter()
            .find(|panel| panel.title.starts_with("IC1"))
            .unwrap();
        assert!(ic1.series.iter().any(|series| match series {
            Series::Triangles { ys, .. } => ys.iter().any(|value| (*value - 10.0).abs() < 1e-3),
            _ => false,
        }));
        assert!(ic1.x_labels.iter().any(|label| label == "70"));
        let replay = scene_of("timeslice_replay");
        assert!(has_kind(&replay, "line"));
        assert_eq!(replay.columns, 1);
        assert!(replay.panels.iter().any(|panel| panel.title == "Overview"));
        assert!(replay.panels.iter().any(|panel| panel.title == "Detail"));
        assert!(replay.panels.iter().all(|panel| panel.xmax < 1.0));
        assert!(replay
            .panels
            .iter()
            .any(|panel| panel.y_label == "IC1 Current"));
        assert!(replay.controls.iter().all(|control| control.id != "scrub"));
    }

    #[test]
    fn replay_axis_ignores_a_single_spike() {
        let mut samples = vec![1.0f32; 400];
        samples[3] = 10_000.0;
        let (lo, hi) = robust_span(&samples).unwrap();
        assert!(hi < 10.0, "{hi}");
        assert!(lo < 2.0);
    }

    #[test]
    fn replay_overview_keeps_a_narrow_pulse() {
        let mut samples = vec![0.0f32; 10_000];
        samples[5000] = 40.0;
        let (_, ys) = envelope(&samples, 480);
        assert!(ys.iter().copied().any(|value| value > 30.0));
    }

    #[test]
    fn sk_req_018_fft_and_audio_share_the_spectrum() {
        let fft = scene_of("ic_fft_analysis");
        assert!(has_kind(&fft, "line"));
        assert!(fft.panels.iter().any(|panel| panel.y_label == "log10 PSD"));
        let audio = scene_of("ic_audio_player");
        assert!(!audio.samples.is_empty());
        assert_eq!(audio.title, "Audio Explorer");
    }

    #[test]
    fn sk_req_022_rampdown_amplifier_and_hv() {
        assert!(has_kind(&scene_of("beam_off_rampdown"), "line"));
        assert!(has_kind(&scene_of("amplifier_correlation"), "points"));
        assert!(has_kind(&scene_of("ic_hv_transient"), "line"));
    }

    #[test]
    fn sk_req_026_trajectory_projects_a_path() {
        assert!(has_kind(&scene_of("trajectory"), "line"));
    }

    #[test]
    fn sk_req_028_dose_volume_has_slices_dvh_and_gamma() {
        let scene = scene_of("dose_volume");
        assert!(has_kind(&scene, "heat"));
        assert!(scene.panels.iter().any(|panel| panel.title.contains("DVH")));
        assert!(scene
            .panels
            .iter()
            .any(|panel| panel.title.contains("gamma")));
        assert!(scene
            .panels
            .iter()
            .any(|panel| panel.title.contains("sagittal")));
        assert!(scene
            .panels
            .iter()
            .any(|panel| panel.title.contains("coronal")));
        assert!(scene.panels.iter().any(|panel| {
            panel.series.iter().any(|series| {
                matches!(
                    series,
                    Series::Heatmap {
                        ramp: 1,
                        lo: 0.0,
                        ..
                    }
                )
            })
        }));
        assert!(scene.controls.iter().any(|control| control.id == "scale"));
    }

    #[test]
    fn dose_volume_difference_drops_a_sequential_scale() {
        let root = std::env::temp_dir().join(format!("scan-kit-dose-diff-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        write_session(&root);
        let scene = analysis_scene(
            "dose_volume",
            &root,
            &["sess".into()],
            &json!({"compare": "Difference", "scale": "Turbo"}),
        )
        .unwrap();
        let _ = std::fs::remove_dir_all(&root);
        assert!(scene
            .controls
            .iter()
            .any(|control| { control.id == "scale" && control.value == "Managua" }));
        assert!(scene.panels.iter().any(|panel| {
            panel
                .series
                .iter()
                .any(|series| matches!(series, Series::Heatmap { ramp: 10, .. }))
        }));
        assert!(scene
            .panels
            .iter()
            .any(|panel| panel.title.contains("gamma") && panel.title.contains('%')));
    }

    #[test]
    fn patient_ct_adds_dvh_gamma_and_a_fraction_control() {
        let root =
            std::env::temp_dir().join(format!("scan-kit-patient-view-{}", std::process::id()));
        let study = root.join("study");
        let _ = std::fs::remove_dir_all(&root);
        write_session(&root);
        scan_kit_dicom::write_water_study(&study).unwrap();
        let scene = crate::dose_volume(
            &root,
            &["sess".into()],
            &serde_json::json!({"study": study.display().to_string(), "model": "Monte Carlo"}),
            Some(&|job| {
                let scan_kit_core::McJob::Patient(request) = job else {
                    return Err("expected a patient job".into());
                };
                assert!(request.dose_to_water);
                assert_eq!(request.seed, 1);
                assert_eq!(request.material.len(), 32);
                assert!(request.protons.iter().any(|weight| *weight > 0.0));
                Ok(scan_kit_core::McResult {
                    volume: scan_kit_core::Volume {
                        origin: request.origin_mm,
                        shape: request.shape,
                        voxel: request.spacing_mm[0],
                        values: vec![1.0; request.material.len()],
                    },
                    uncertainty: 0.0,
                    ledger: [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                })
            }),
        );
        let _ = std::fs::remove_dir_all(&root);
        assert!(scene
            .controls
            .iter()
            .any(|control| control.id == "fraction"));
        assert!(scene.panels.iter().any(|panel| panel.title.contains("DVH")));
        assert!(scene
            .panels
            .iter()
            .any(|panel| panel.title.contains("gamma") && panel.title.contains('%')));
        assert!(scene.table.as_ref().is_some_and(|table| {
            table.rows.iter().any(|row| {
                row.iter()
                    .any(|cell| cell.contains("Gamma") || cell.contains("PTV"))
            })
        }));
        let plain = scene_of("dose_volume");
        assert!(plain
            .controls
            .iter()
            .all(|control| control.id != "fraction"));
    }

    #[test]
    fn nested_session_reads_its_own_folder() {
        let root = std::env::temp_dir().join(format!("scan-kit-nested-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let nested = root.join("a").join("a");
        std::fs::create_dir_all(nested.join("layer-0").join("run-0")).unwrap();
        std::fs::write(nested.join("input_map.csv"), "energy,charge_req\n70,1\n").unwrap();
        std::fs::write(
            nested
                .join("layer-0")
                .join("run-0")
                .join("timeslice_data_device_units.csv"),
            "r_ic1_current_dose\n1\n",
        )
        .unwrap();
        let sibling = root.join("b");
        std::fs::create_dir_all(sibling.join("layer-0").join("run-0")).unwrap();
        std::fs::write(sibling.join("input_map.csv"), "energy,charge_req\n10,1\n").unwrap();
        std::fs::write(
            sibling
                .join("layer-0")
                .join("run-0")
                .join("timeslice_data_device_units.csv"),
            "only_sibling\n9\n",
        )
        .unwrap();
        let loaded = load_timeslice_columns(&root, "a");
        let names: Vec<&str> = loaded["columns"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|column| column["name"].as_str())
            .collect();
        assert!(names.contains(&"r_ic1_current_dose"));
        assert!(!names.contains(&"only_sibling"));
        assert_eq!(loaded["energy_layers"], 1);
        let scene = analysis_scene("binned_summary", &root, &["a".into()], &json!({})).unwrap();
        assert!(scene
            .panels
            .iter()
            .any(|panel| panel.title.contains("No finite values")));
        let _ = std::fs::remove_dir_all(&root);
    }
}
