//! Analysis view workflows. Each one loads columns and returns a plot scene.
//! Pixels are rendered later by `scan-kit-compute`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use scan_kit_core::{
    arc_fit, arc_predict, beam_angle_mrad, beam_off_edges, beam_on_mask, calibration_factor,
    compare_templates, coverage_percent, cumsum, density_counts, dose_error_pct, dvh,
    filter_beam_state, fit_decay, fit_iso_plane, gamma_index, histogram, hv_capacitance_pf,
    hv_delta_v, hv_expected_pf, hv_firmware_flags, hv_step_window, linear_fit, magnet_pivot_z,
    mip_xy, parse_session_log, quantile_edges, resample_nearest, resolve_concept_column,
    scale_column, settled_after_step, spill_segments, splat_gaussians, sums_by_spot_id, welch_psd,
    BeamState, Control, DataTable, Panel, PlotScene, Series, IC1_Z_MM, IC2_Z_MM, MIN_SPILL_GAP_MS,
};
use serde_json::{json, Value};

use super::discover;

const BLUE: [f32; 4] = [0.35, 0.55, 0.95, 1.0];
const ORANGE: [f32; 4] = [0.95, 0.55, 0.3, 1.0];
const GREEN: [f32; 4] = [0.4, 0.75, 0.45, 1.0];

const CHANNELS: &[&str] = &[
    "ic1_current",
    "ic2_current",
    "ic3_current",
    "dose_rate",
    "sigma_x",
    "sigma_y",
    "position_x",
    "position_y",
    "position_error_x",
    "position_error_y",
    "field_x",
    "field_y",
    "spot_no",
    "layer_id",
    "r_ic1_x_confidence",
    "r_ic1_y_confidence",
    "r_ic2_x_confidence",
    "r_ic2_y_confidence",
];

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
        "dose_volume" => Ok(dose_volume(root, session_ids)),
        _ => Err(format!("unknown view {view}")),
    }
}

pub fn channel_catalog(root: &Path, session_id: &str) -> Vec<String> {
    let columns = load_timeslice(root, session_id);
    let names: Vec<String> = columns.keys().cloned().collect();
    CHANNELS
        .iter()
        .filter(|concept| resolve_concept_column(&names, concept).is_some())
        .map(|concept| (*concept).to_owned())
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
    let calibrate = text_option(options, "calibrate", "off") == "on";
    let mut panels = Vec::new();
    for (label, concept, current_name) in [
        ("IC1", "ic1_total_dose", "ic1_current"),
        ("IC2", "ic2_total_dose", "ic2_current"),
        ("IC3", "ic3_total_dose", "ic3_current"),
    ] {
        let mut expected = empty_line(ORANGE);
        let mut measured = empty_line(BLUE);
        let mut error = empty_line(GREEN);
        let mut current_sum = empty_line(BLUE);
        let mut drew = false;
        for session in session_ids {
            let map = load_csv(root, session, "input_map.csv");
            let spots = load_csv(root, session, "spot_data.csv");
            let Some(target) = col(&map, "charge_req") else {
                continue;
            };
            let Some(dose) = col(&spots, concept) else {
                continue;
            };
            let n = target.len().min(dose.len());
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
            extend_line(&mut expected, &xs, &cum_t);
            extend_line(&mut measured, &xs, &cum_d);
            extend_line(&mut error, &xs, &err);
            drew = true;
            if let Some(sums) = timeslice_spot_sums(root, session, current_name, n) {
                let cum = cumsum(&sums);
                extend_line(&mut current_sum, &xs[..cum.len().min(xs.len())], &cum);
            }
        }
        if !drew {
            continue;
        }
        let ymax = line_max(&expected).max(line_max(&measured)).max(1.0);
        panels.push(panel(
            format!("{label} cumulative"),
            1.0,
            xs_end(&expected),
            0.0,
            ymax,
            vec![expected, measured],
        ));
        let emin = line_min(&error);
        let emax = line_max(&error).max(emin + 1.0);
        panels.push(panel(
            format!("{label} error"),
            1.0,
            xs_end(&error),
            emin,
            emax,
            vec![error],
        ));
        if line_has_finite(&current_sum) {
            panels.push(panel(
                format!("{label} current sum"),
                1.0,
                xs_end(&current_sum),
                0.0,
                line_max(&current_sum).max(1.0),
                vec![current_sum],
            ));
        }
    }
    scene(
        "Dose Accumulation",
        panels,
        vec![control(
            "calibrate",
            "Calibrate",
            &["off", "on"],
            text_option(options, "calibrate", "off"),
        )],
    )
}

fn peak_amplitude(root: &Path, session_ids: &[String]) -> PlotScene {
    let mut panels = Vec::new();
    for concept in [
        "ic1_x_peak_amplitude",
        "ic1_y_peak_amplitude",
        "ic2_x_peak_amplitude",
        "ic2_y_peak_amplitude",
    ] {
        let mut values = Vec::new();
        for session in session_ids {
            let columns = load_timeslice(root, session);
            let names: Vec<String> = columns.keys().cloned().collect();
            let Some(name) = resolve_concept_column(&names, concept) else {
                continue;
            };
            let gate = col(&columns, "rci_in_trigger").or_else(|| col(&columns, "r_beamOk"));
            let samples = columns.get(name).map(Vec::as_slice).unwrap_or(&[]);
            let on = gate
                .map(beam_on_mask)
                .unwrap_or_else(|| vec![false; samples.len()]);
            values.extend(
                filter_beam_state(samples, &on, BeamState::Off)
                    .into_iter()
                    .filter(|v| v.is_finite()),
            );
        }
        let (edges, counts) = histogram(&values, 16);
        let xmax = edges.last().copied().unwrap_or(1.0);
        let ymax = counts.iter().copied().fold(0.0, f32::max).max(1.0);
        panels.push(panel(
            concept.to_owned(),
            edges.first().copied().unwrap_or(0.0),
            xmax,
            0.0,
            ymax,
            vec![Series::Bars {
                edges,
                counts,
                color: BLUE,
            }],
        ));
    }
    scene("IC Peak Amplitude — Beam-Off", panels, Vec::new())
}

fn beam_motion(root: &Path, session_ids: &[String]) -> PlotScene {
    let mut panels = Vec::new();
    for session in session_ids {
        let columns = load_timeslice(root, session);
        let x = col(&columns, "position_error_x").unwrap_or(&[]);
        if x.is_empty() {
            continue;
        }
        let y = col(&columns, "position_error_y").unwrap_or(&[]);
        let x2 = col(&columns, "position_error_x2").unwrap_or(&[]);
        let y2 = col(&columns, "position_error_y2").unwrap_or(&[]);
        let gate = col(&columns, "rci_in_trigger").or_else(|| col(&columns, "r_beamOk"));
        let on = gate
            .map(beam_on_mask)
            .unwrap_or_else(|| vec![true; x.len()]);
        let mut segments = spill_segments(&on, MIN_SPILL_GAP_MS, 2);
        if segments.is_empty() {
            segments.push((0, x.len()));
        }
        let energies = sample_energy(root, session, &columns, x.len());
        let mut by_energy: BTreeMap<i32, Vec<(usize, usize)>> = BTreeMap::new();
        for (start, end) in segments {
            let sample = start.min(energies.len().saturating_sub(1));
            let key = (energies.get(sample).copied().unwrap_or(0.0) * 10.0).round() as i32;
            by_energy.entry(key).or_default().push((start, end));
        }
        for (key, spills) in by_energy.iter().take(40) {
            let mut sx = Vec::new();
            let mut sy = Vec::new();
            let mut tx = Vec::new();
            let mut ty = Vec::new();
            for (start, end) in spills {
                push_spill(&mut sx, &mut sy, x, y, *start, *end);
                if !x2.is_empty() {
                    push_spill(&mut tx, &mut ty, x2, y2, *start, *end);
                }
            }
            if y.len() >= x.len() && sx.iter().any(|v| v.is_finite()) {
                push_circle(&mut sx, &mut sy, 1.0);
            }
            let mut series = vec![Series::Polyline {
                xs: sx,
                ys: sy,
                color: BLUE,
                thickness: 1.2,
            }];
            if tx.iter().any(|v| v.is_finite()) {
                series.push(Series::Polyline {
                    xs: tx,
                    ys: ty,
                    color: ORANGE,
                    thickness: 1.2,
                });
            }
            let (xmin, xmax, ymin, ymax) = series_span(&series);
            panels.push(panel(
                format!("{session} {:.0} MeV", *key as f32 / 10.0),
                xmin,
                xmax,
                ymin,
                ymax,
                series,
            ));
        }
    }
    scene("Beam Error Motion vs Energy", panels, Vec::new())
}

fn distribution(root: &Path, session_ids: &[String], options: &Value) -> PlotScene {
    let mode = text_option(options, "mode", "position");
    let grain = text_option(options, "grain", "spot");
    let beam = text_option(options, "beam", "all");
    let draw = text_option(options, "draw", "scatter");
    let (x_concept, y_concept) = match mode {
        "sigma" => ("sigma_x", "sigma_y"),
        "position_error" => ("position_error_x", "position_error_y"),
        "confidence" => ("r_ic1_x_confidence", "r_ic1_y_confidence"),
        _ => ("position_x", "position_y"),
    };
    let state = match beam {
        "on" => BeamState::On,
        "off" => BeamState::Off,
        "all" if mode == "confidence" || mode == "coverage" => BeamState::On,
        _ => BeamState::All,
    };
    let mut panels = Vec::new();
    for session in session_ids {
        let timeslice = load_timeslice(root, session);
        if mode == "confidence" {
            panels.extend(confidence_panels(session, &timeslice, state));
            continue;
        }
        if mode == "coverage" {
            panels.extend(coverage_panels(session, &timeslice, state));
            continue;
        }
        let spots = load_csv(root, session, "spot_data.csv");
        let map = load_csv(root, session, "input_map.csv");
        let source = if grain == "timeslice" && col(&timeslice, x_concept).is_some() {
            &timeslice
        } else if col(&spots, x_concept).is_some() {
            &spots
        } else {
            &map
        };
        let Some(xs) = col(source, x_concept) else {
            continue;
        };
        let Some(ys) = col(source, y_concept) else {
            continue;
        };
        let gate = col(&timeslice, "rci_in_trigger").or_else(|| col(&timeslice, "r_beamOk"));
        let (xs, ys) = select_pairs(xs, ys, gate, state);
        if xs.is_empty() {
            continue;
        }
        let (xmin, xmax) = span(&xs);
        let (ymin, ymax) = span(&ys);
        let series = if draw == "density" {
            match density_counts(&xs, &ys, 32) {
                Some((values, x0, x1, y0, y1)) => {
                    panels.push(panel(
                        session.clone(),
                        x0,
                        x1,
                        y0,
                        y1,
                        vec![Series::Heatmap {
                            values,
                            cols: 32,
                            rows: 32,
                        }],
                    ));
                    continue;
                }
                None => Vec::new(),
            }
        } else {
            vec![Series::Points {
                xs,
                ys,
                color: BLUE,
                radius: 2.0,
            }]
        };
        if series.is_empty() {
            continue;
        }
        panels.push(panel(session.clone(), xmin, xmax, ymin, ymax, series));
    }
    scene(
        "Distribution Explorer",
        panels,
        vec![
            control(
                "mode",
                "Signal",
                &[
                    "position",
                    "position_error",
                    "sigma",
                    "confidence",
                    "coverage",
                ],
                mode,
            ),
            control("grain", "Grain", &["spot", "timeslice"], grain),
            control("beam", "Beam", &["all", "on", "off"], beam),
            control("draw", "Draw", &["scatter", "density"], draw),
        ],
    )
}

fn binned_summary(root: &Path, session_ids: &[String], options: &Value) -> PlotScene {
    let metric = text_option(options, "metric", "dose_error");
    let grain = text_option(options, "grain", "spot");
    let axis = text_option(options, "axis", "energy");
    let style = text_option(options, "style", "mean");
    let mut mean_x = Vec::new();
    let mut mean_y = Vec::new();
    let mut raw_x = Vec::new();
    let mut raw_y = Vec::new();
    let mut box_x = Vec::new();
    let mut box_lo = Vec::new();
    let mut box_hi = Vec::new();
    let mut box_mid = Vec::new();
    for session in session_ids {
        let map = load_csv(root, session, "input_map.csv");
        let spots = load_csv(root, session, "spot_data.csv");
        let timeslice = if grain == "timeslice" || metric == "current" {
            load_timeslice(root, session)
        } else {
            BTreeMap::new()
        };
        let (x_axis, values) = if grain == "timeslice" {
            let samples = col(&timeslice, "ic1_current").unwrap_or(&[]);
            let index: Vec<f32> = (0..samples.len()).map(|i| i as f32).collect();
            (index, samples.to_vec())
        } else {
            let values = match metric {
                "dose" => col(&spots, "ic1_total_dose").unwrap_or(&[]).to_vec(),
                "current" => col(&timeslice, "ic1_current").unwrap_or(&[]).to_vec(),
                _ => {
                    let target = col(&map, "charge_req").unwrap_or(&[]);
                    let dose = col(&spots, "ic1_total_dose").unwrap_or(&[]);
                    dose_error_pct(dose, target)
                }
            };
            let axis_values = match axis {
                "mu" => col(&map, "charge_req").unwrap_or(&[]).to_vec(),
                "time" => (0..values.len()).map(|i| i as f32).collect(),
                "radius" => {
                    let x = col(&spots, "position_x")
                        .or_else(|| col(&map, "position_x"))
                        .unwrap_or(&[]);
                    let y = col(&spots, "position_y")
                        .or_else(|| col(&map, "position_y"))
                        .unwrap_or(&[]);
                    x.iter().zip(y).map(|(px, py)| px.hypot(*py)).collect()
                }
                _ => col(&map, "energy").unwrap_or(&[]).to_vec(),
            };
            (axis_values, values)
        };
        let edges = quantile_edges(&x_axis, 6);
        if edges.len() < 2 {
            continue;
        }
        for (bin_i, window) in edges.windows(2).enumerate() {
            let last = bin_i + 2 == edges.len();
            let mut bin = Vec::new();
            let mut bin_x = Vec::new();
            for (e, v) in x_axis.iter().zip(&values) {
                let inside = *e >= window[0] && (*e < window[1] || (last && *e <= window[1]));
                if e.is_finite() && v.is_finite() && inside {
                    bin.push(*v);
                    bin_x.push(*e);
                }
            }
            if bin.is_empty() {
                continue;
            }
            let center = window[0].midpoint(window[1]);
            let mean = bin.iter().sum::<f32>() / bin.len() as f32;
            mean_x.push(center);
            mean_y.push(mean);
            if raw_x.len() < 1500 {
                raw_x.extend(bin_x);
                raw_y.extend_from_slice(&bin);
            }
            bin.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            box_x.push(center);
            box_lo.push(quantile_at(&bin, 0.25));
            box_hi.push(quantile_at(&bin, 0.75));
            box_mid.push(quantile_at(&bin, 0.5));
        }
    }
    let (xs, ys, title) = match style {
        "scatter" => (raw_x, raw_y, "scatter"),
        "box" => (box_x.clone(), box_mid.clone(), "box"),
        _ => (mean_x.clone(), mean_y.clone(), "mean"),
    };
    let (xmin, xmax) = span(&xs);
    let (ymin, ymax) = span(&ys);
    let mut series = vec![Series::Points {
        xs: xs.clone(),
        ys: ys.clone(),
        color: BLUE,
        radius: 3.0,
    }];
    if style == "mean" {
        series.push(Series::Polyline {
            xs: mean_x,
            ys: mean_y,
            color: ORANGE,
            thickness: 1.5,
        });
    }
    if style == "box" {
        let mut vx = Vec::new();
        let mut vy = Vec::new();
        for i in 0..box_x.len() {
            vx.extend([box_x[i], box_x[i], f32::NAN]);
            vy.extend([box_lo[i], box_hi[i], f32::NAN]);
        }
        series.push(Series::Polyline {
            xs: vx,
            ys: vy,
            color: ORANGE,
            thickness: 4.0,
        });
    }
    scene(
        "Binned Summary",
        vec![panel(
            format!("{title} vs {axis}"),
            xmin,
            xmax,
            ymin,
            ymax,
            series,
        )],
        vec![
            control(
                "metric",
                "Metric",
                &["dose_error", "dose", "current"],
                metric,
            ),
            control("axis", "Axis", &["energy", "mu", "time", "radius"], axis),
            control("style", "Style", &["mean", "scatter", "box"], style),
            control("grain", "Grain", &["spot", "timeslice"], grain),
        ],
    )
}

fn replay(root: &Path, session_ids: &[String], options: &Value) -> PlotScene {
    let channel = text_option(options, "channel", "ic1_current");
    let scrub = number_option(options, "scrub", 0.0).clamp(0.0, 1.0);
    let mut panels = Vec::new();
    let mut choices = Vec::new();
    for session in session_ids {
        let catalog = channel_catalog(root, session);
        if choices.is_empty() {
            choices = catalog.clone();
        }
        let columns = load_timeslice(root, session);
        let Some(samples) = col(&columns, channel) else {
            continue;
        };
        let step = (samples.len() / 400).max(1);
        let xs: Vec<f32> = (0..samples.len()).step_by(step).map(|i| i as f32).collect();
        let ys: Vec<f32> = (0..samples.len())
            .step_by(step)
            .map(|i| samples[i])
            .collect();
        let start = ((samples.len() as f32) * scrub) as usize;
        let end = (start + 200).min(samples.len());
        let detail_x: Vec<f32> = (start..end).map(|i| i as f32).collect();
        let detail_y = samples[start..end].to_vec();
        panels.push(line_panel(format!("{session} overview"), &xs, &ys, BLUE));
        panels.push(line_panel(
            format!("{session} detail"),
            &detail_x,
            &detail_y,
            ORANGE,
        ));
    }
    if choices.is_empty() {
        choices = CHANNELS.iter().map(|name| (*name).to_owned()).collect();
    }
    let choice_refs: Vec<&str> = choices.iter().map(String::as_str).collect();
    scene(
        "Timeslice Replay",
        panels,
        vec![
            control("channel", "Channel", &choice_refs, channel),
            control(
                "scrub",
                "Scrub",
                &["0", "0.25", "0.5", "0.75"],
                &scrub_label(scrub),
            ),
        ],
    )
}

fn fft_view(root: &Path, session_ids: &[String], options: &Value) -> PlotScene {
    let channel = text_option(options, "channel", "ic1_current");
    let mut panels = Vec::new();
    for session in session_ids {
        let columns = load_timeslice(root, session);
        let Some(samples) = col(&columns, channel) else {
            continue;
        };
        let (freqs, psd) = welch_psd(samples, 1000.0, 4096, 0.5);
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for (freq, power) in freqs.iter().zip(&psd) {
            if *freq >= 1.0 && *freq <= 500.0 {
                xs.push(*freq);
                ys.push(*power);
            }
        }
        panels.push(line_panel(session.clone(), &xs, &ys, BLUE));
    }
    scene(
        "FFT Explorer",
        panels,
        vec![control("channel", "Channel", CHANNELS, channel)],
    )
}

fn audio_view(root: &Path, session_ids: &[String], options: &Value) -> PlotScene {
    let mut scene = fft_view(root, session_ids, options);
    scene.title = "Audio Explorer".into();
    if let Some(session) = session_ids.first() {
        let columns = load_timeslice(root, session);
        let channel = text_option(options, "channel", "ic1_current");
        if let Some(samples) = col(&columns, channel) {
            let peak = samples
                .iter()
                .copied()
                .filter(|v| v.is_finite())
                .fold(0.0f32, |m, v| m.max(v.abs()))
                .max(1e-6);
            scene.samples = samples
                .iter()
                .take(8000)
                .map(|v| (v / peak).clamp(-1.0, 1.0))
                .collect();
        }
    }
    scene
}

fn rampdown(root: &Path, session_ids: &[String]) -> PlotScene {
    let mut panels = Vec::new();
    for session in session_ids {
        let columns = load_timeslice(root, session);
        for (label, concept) in [
            ("IC1", "ic1_current"),
            ("IC2", "ic2_current"),
            ("IC3", "ic3_current"),
        ] {
            let Some(samples) = col(&columns, concept).or_else(|| {
                if concept == "ic1_current" {
                    col(&columns, "ic1_scan_dose")
                } else {
                    None
                }
            }) else {
                continue;
            };
            let mut edges = beam_off_edges(samples);
            if edges.is_empty() {
                let gate = col(&columns, "rci_in_trigger").or_else(|| col(&columns, "r_beamOk"));
                let on = gate
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
            let mut series = vec![Series::Polyline {
                xs: xs.clone(),
                ys: ys.clone(),
                color: BLUE,
                thickness: 1.5,
            }];
            let time: Vec<f32> = (0..width).map(|i| i as f32).collect();
            let mean = mean_window(&windows, width);
            if let Some((amp, tau)) = fit_decay(&time, &mean) {
                let fit: Vec<f32> = time.iter().map(|t| amp * (-t / tau).exp()).collect();
                series.push(Series::Polyline {
                    xs: time,
                    ys: fit,
                    color: ORANGE,
                    thickness: 1.5,
                });
            }
            panels.push(panel(
                format!("{session} {label}"),
                0.0,
                width as f32,
                0.0,
                1.05,
                series,
            ));
            panels.push(panel(
                format!("{label} windows"),
                0.0,
                width as f32,
                0.0,
                windows.len() as f32,
                vec![Series::Heatmap {
                    values: heat,
                    cols: width as u32,
                    rows: windows.len() as u32,
                }],
            ));
        }
    }
    scene("Beam-Off Ramp-Down", panels, Vec::new())
}

fn amplifier(root: &Path, session_ids: &[String]) -> PlotScene {
    let mut panels = Vec::new();
    for session in session_ids {
        let columns = load_timeslice(root, session);
        let Some(cmd) = col(&columns, "amplifier_command_x").or_else(|| col(&columns, "field_x"))
        else {
            continue;
        };
        let Some(readback) =
            col(&columns, "amplifier_readback_x").or_else(|| col(&columns, "field_y"))
        else {
            continue;
        };
        let n = cmd.len().min(readback.len()).min(2000);
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
            xs = cmd[..n].to_vec();
            ys = readback[..n].to_vec();
        }
        let (xmin, xmax) = span(&xs);
        let (ymin, ymax) = span(&ys);
        let mut series = vec![Series::Points {
            xs: xs.clone(),
            ys: ys.clone(),
            color: BLUE,
            radius: 1.5,
        }];
        if let Some((slope, intercept)) = linear_fit(&xs, &ys) {
            series.push(Series::Polyline {
                xs: vec![xmin, xmax],
                ys: vec![intercept + slope * xmin, intercept + slope * xmax],
                color: ORANGE,
                thickness: 1.5,
            });
        }
        panels.push(panel(session.clone(), xmin, xmax, ymin, ymax, series));
        if let Some((counts, x0, x1, y0, y1)) = density_counts(&xs, &ys, 24) {
            panels.push(panel(
                format!("{session} density"),
                x0,
                x1,
                y0,
                y1,
                vec![Series::Heatmap {
                    values: counts,
                    cols: 24,
                    rows: 24,
                }],
            ));
        }
        let spots = load_csv(root, session, "spot_data.csv");
        let x1 = col(&spots, "ic1_position_x").or_else(|| col(&columns, "position_x"));
        let x2 = col(&spots, "position_x").or_else(|| col(&columns, "ic2_position_x"));
        if let (Some(ic1), Some(ic2)) = (x1, x2) {
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
                panels.push(line_panel(
                    format!("{session} arc"),
                    &curve_x,
                    &curve_y,
                    ORANGE,
                ));
            }
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
                BLUE,
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
                    color: ORANGE,
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
            "OVERVIEW".into(),
            format!("{} lines", log.lines),
        ]);
        for (level, count) in &log.level_counts {
            rows.push(vec![session.clone(), level.clone(), count.to_string()]);
        }
        for event in &log.timeline {
            rows.push(vec![
                session.clone(),
                "TIMELINE".into(),
                format!("layer {} {} {:.3}s", event.layer, event.kind, event.seconds),
            ]);
        }
        for issue in log.issues.iter().take(40) {
            rows.push(vec![session.clone(), "ERROR".into(), issue.clone()]);
        }
        for (device, count) in &log.wdt {
            rows.push(vec![
                session.clone(),
                "WDT".into(),
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
                "DIFF".into(),
                format!("{template} {a} {b} {delta}"),
            ]);
        }
    }
    let mut scene = scene("Session Log Compare", Vec::new(), Vec::new());
    scene.table = Some(DataTable {
        columns: vec!["Session".into(), "Level".into(), "Detail".into()],
        rows,
    });
    scene
}

fn trajectory(root: &Path, session_ids: &[String], options: &Value) -> PlotScene {
    let azimuth = number_option(options, "azimuth", 0.4);
    let mut panels = Vec::new();
    for session in session_ids {
        let spots = load_csv(root, session, "spot_data.csv");
        let map = load_csv(root, session, "input_map.csv");
        let x2 = col(&spots, "position_x")
            .or_else(|| col(&map, "position_x"))
            .unwrap_or(&[]);
        let y2 = col(&spots, "position_y")
            .or_else(|| col(&map, "position_y"))
            .unwrap_or(&[]);
        let x1 = col(&spots, "ic1_position_x").unwrap_or(x2);
        let energy = col(&map, "energy").unwrap_or(&[]);
        let mut sx = Vec::new();
        let mut sy = Vec::new();
        for i in 0..x2.len().min(y2.len()).min(400) {
            let (px, py) = project(x2[i], y2.get(i).copied().unwrap_or(0.0), IC2_Z_MM, azimuth);
            let (qx, qy) = project(
                x1.get(i).copied().unwrap_or(x2[i]),
                y2.get(i).copied().unwrap_or(0.0),
                IC1_Z_MM,
                azimuth,
            );
            sx.extend([px, qx, f32::NAN]);
            sy.extend([py, qy, f32::NAN]);
        }
        push_projected(
            &mut sx,
            &mut sy,
            (-40.0, 0.0, IC2_Z_MM),
            (40.0, 0.0, IC2_Z_MM),
            azimuth,
        );
        push_projected(
            &mut sx,
            &mut sy,
            (-40.0, 0.0, IC1_Z_MM),
            (40.0, 0.0, IC1_Z_MM),
            azimuth,
        );
        let pivot = magnet_pivot_z(
            x2.first().copied().unwrap_or(0.0),
            x1.first().copied().unwrap_or(0.0),
        );
        push_projected(
            &mut sx,
            &mut sy,
            (-20.0, -4.0, pivot),
            (20.0, -4.0, pivot),
            azimuth,
        );
        push_projected(
            &mut sx,
            &mut sy,
            (-20.0, 4.0, pivot),
            (20.0, 4.0, pivot),
            azimuth,
        );
        let (xmin, xmax) = span(&sx);
        let (ymin, ymax) = span(&sy);
        panels.push(panel(
            format!("{session} planes pivot {pivot:.0} mm"),
            xmin,
            xmax,
            ymin,
            ymax,
            vec![Series::Polyline {
                xs: sx,
                ys: sy,
                color: BLUE,
                thickness: 1.2,
            }],
        ));
        if let Some((intercept, slope)) = fit_iso_plane(energy, x2) {
            let (e0, e1) = span(energy);
            panels.push(line_panel(
                format!("{session} iso"),
                &[e0, e1],
                &[intercept + slope * e0, intercept + slope * e1],
                GREEN,
            ));
        }
    }
    scene(
        "IC Beam Trajectory",
        panels,
        vec![control(
            "azimuth",
            "Orbit",
            &["0.2", "0.4", "0.8", "1.2"],
            &format!("{azimuth:.1}"),
        )],
    )
}

fn dose_volume(root: &Path, session_ids: &[String]) -> PlotScene {
    let mut panels = Vec::new();
    for session in session_ids {
        let map = load_csv(root, session, "input_map.csv");
        let spots = load_csv(root, session, "spot_data.csv");
        let x = col(&map, "position_x")
            .or_else(|| col(&spots, "position_x"))
            .unwrap_or(&[]);
        let y = col(&map, "position_y")
            .or_else(|| col(&spots, "position_y"))
            .unwrap_or(&[]);
        let dose = col(&spots, "ic1_total_dose")
            .or_else(|| col(&map, "charge_req"))
            .unwrap_or(&[]);
        let n = x.len().min(y.len()).min(dose.len()).min(2000);
        if n == 0 {
            continue;
        }
        let mut spots_in = Vec::with_capacity(n);
        let mut lo_x = x[0];
        let mut hi_x = x[0];
        let mut lo_y = y[0];
        let mut hi_y = y[0];
        for i in 0..n {
            lo_x = lo_x.min(x[i]);
            hi_x = hi_x.max(x[i]);
            lo_y = lo_y.min(y[i]);
            hi_y = hi_y.max(y[i]);
            spots_in.push([x[i], y[i], 0.0, dose[i].max(0.0), 4.0]);
        }
        let shape = [32usize, 32, 8];
        let spacing = [
            ((hi_x - lo_x) / 31.0).max(1.0),
            ((hi_y - lo_y) / 31.0).max(1.0),
            5.0,
        ];
        let grid = splat_gaussians(&spots_in, [lo_x, lo_y, 0.0], spacing, shape);
        let image = mip_xy(&grid, shape);
        let lateral: Vec<f32> = (0..shape[0])
            .map(|x| (0..shape[1]).map(|y| image[x + shape[0] * y]).sum())
            .collect();
        let depth: Vec<f32> = (0..shape[2])
            .map(|z| {
                let start = shape[0] * shape[1] * z;
                grid[start..start + shape[0] * shape[1]].iter().sum()
            })
            .collect();
        let depth_x: Vec<f32> = (0..depth.len()).map(|i| i as f32 * spacing[2]).collect();
        panels.push(panel(
            format!("{session} MIP"),
            0.0,
            32.0,
            0.0,
            32.0,
            vec![Series::Heatmap {
                values: image,
                cols: 32,
                rows: 32,
            }],
        ));
        panels.push(line_panel(
            format!("{session} depth"),
            &depth_x,
            &depth,
            ORANGE,
        ));
        let lateral_x: Vec<f32> = (0..lateral.len()).map(|i| i as f32).collect();
        panels.push(line_panel(
            format!("{session} lateral"),
            &lateral_x,
            &lateral,
            GREEN,
        ));
        let mut sagittal = vec![0.0f32; shape[1] * shape[2]];
        let xmid = shape[0] / 2;
        for z in 0..shape[2] {
            for y in 0..shape[1] {
                sagittal[y + shape[1] * z] = grid[xmid + shape[0] * (y + shape[1] * z)];
            }
        }
        panels.push(panel(
            format!("{session} sagittal"),
            0.0,
            shape[1] as f32,
            0.0,
            shape[2] as f32,
            vec![Series::Heatmap {
                values: sagittal,
                cols: shape[1] as u32,
                rows: shape[2] as u32,
            }],
        ));
        let mut coronal = vec![0.0f32; shape[0] * shape[2]];
        let ymid = shape[1] / 2;
        for z in 0..shape[2] {
            for x in 0..shape[0] {
                coronal[x + shape[0] * z] = grid[x + shape[0] * (ymid + shape[1] * z)];
            }
        }
        panels.push(panel(
            format!("{session} coronal"),
            0.0,
            shape[0] as f32,
            0.0,
            shape[2] as f32,
            vec![Series::Heatmap {
                values: coronal,
                cols: shape[0] as u32,
                rows: shape[2] as u32,
            }],
        ));
        let mask = vec![true; grid.len()];
        let (edges, curve) = dvh(&grid, &mask, 12);
        let dvh_x: Vec<f32> = edges.iter().take(curve.len()).copied().collect();
        panels.push(line_panel(format!("{session} DVH"), &dvh_x, &curve, BLUE));
        let charge = col(&map, "charge_req").unwrap_or(&[]);
        let plan_in: Vec<[f32; 5]> = (0..n)
            .map(|i| {
                [
                    x[i],
                    y[i],
                    0.0,
                    charge.get(i).copied().unwrap_or(dose[i]).max(0.0),
                    4.0,
                ]
            })
            .collect();
        let plan = splat_gaussians(&plan_in, [lo_x, lo_y, 0.0], spacing, shape);
        let (gamma, _, _) = gamma_index(&plan, &grid, shape, 3.0, 2.0, spacing, 0.1);
        panels.push(panel(
            format!("{session} gamma"),
            0.0,
            32.0,
            0.0,
            32.0,
            vec![Series::Heatmap {
                values: mip_xy(&gamma, shape),
                cols: 32,
                rows: 32,
            }],
        ));
        let coarse = resample_nearest(&grid, shape, [16, 16, 4]);
        panels.push(panel(
            format!("{session} resample"),
            0.0,
            16.0,
            0.0,
            16.0,
            vec![Series::Heatmap {
                values: mip_xy(&coarse, [16, 16, 4]),
                cols: 16,
                rows: 16,
            }],
        ));
    }
    scene("Dose Volume", panels, Vec::new())
}

fn scene(title: &str, panels: Vec<Panel>, controls: Vec<Control>) -> PlotScene {
    PlotScene {
        title: title.into(),
        panels,
        controls,
        table: None,
        samples: Vec::new(),
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
        xmin,
        xmax,
        ymin,
        ymax,
        series,
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

fn extend_line(series: &mut Series, xs: &[f32], ys: &[f32]) {
    if let Series::Polyline { xs: dx, ys: dy, .. } = series {
        if !dx.is_empty() {
            dx.push(f32::NAN);
            dy.push(f32::NAN);
        }
        dx.extend_from_slice(xs);
        dy.extend_from_slice(ys);
    }
}

fn line_max(series: &Series) -> f32 {
    match series {
        Series::Polyline { ys, .. } => ys
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .fold(0.0, f32::max),
        _ => 0.0,
    }
}

fn line_min(series: &Series) -> f32 {
    match series {
        Series::Polyline { ys, .. } => ys
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .fold(0.0, f32::min),
        _ => 0.0,
    }
}

fn xs_end(series: &Series) -> f32 {
    match series {
        Series::Polyline { xs, .. } => xs
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .fold(1.0, f32::max),
        _ => 1.0,
    }
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

fn scrub_label(scrub: f32) -> String {
    if scrub < 0.125 {
        "0".into()
    } else if scrub < 0.375 {
        "0.25".into()
    } else if scrub < 0.625 {
        "0.5".into()
    } else {
        "0.75".into()
    }
}

fn text_option<'a>(options: &'a Value, key: &str, default: &'a str) -> &'a str {
    options.get(key).and_then(Value::as_str).unwrap_or(default)
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
                .and_then(|text| text.parse().ok())
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

fn empty_line(color: [f32; 4]) -> Series {
    Series::Polyline {
        xs: Vec::new(),
        ys: Vec::new(),
        color,
        thickness: 1.5,
    }
}

fn line_has_finite(series: &Series) -> bool {
    match series {
        Series::Polyline { ys, .. } => ys.iter().any(|value| value.is_finite()),
        _ => false,
    }
}

fn timeslice_spot_sums(root: &Path, session: &str, concept: &str, n: usize) -> Option<Vec<f32>> {
    let mut sums = Vec::new();
    let mut found = false;
    for bytes in discover::read_timeslices(&session_dir(root, session)) {
        let Ok(frame) = numeric_columns(&bytes) else {
            continue;
        };
        let Some(part) = frame_spot_sums(&frame, concept) else {
            continue;
        };
        found = true;
        sums.extend(part);
    }
    if !found {
        return None;
    }
    sums.truncate(n);
    while sums.len() < n {
        sums.push(f32::NAN);
    }
    Some(sums)
}

fn frame_spot_sums(columns: &BTreeMap<String, Vec<f32>>, concept: &str) -> Option<Vec<f32>> {
    let spot = col(columns, "spot_no")?;
    let current = if let Some(found) = col(columns, concept) {
        found.to_vec()
    } else if concept == "ic3_current" {
        sum_ic3_quads(columns)?
    } else {
        return None;
    };
    let sums = sums_by_spot_id(spot, &current);
    (!sums.is_empty()).then_some(sums)
}

fn confidence_panels(
    session: &str,
    columns: &BTreeMap<String, Vec<f32>>,
    state: BeamState,
) -> Vec<Panel> {
    let gate = col(columns, "rci_in_trigger").or_else(|| col(columns, "r_beamOk"));
    let mut panels = Vec::new();
    for (label, confidence, peak) in [
        ("IC1 X", "r_ic1_x_confidence", "ic1_x_peak_amplitude"),
        ("IC1 Y", "r_ic1_y_confidence", "ic1_y_peak_amplitude"),
        ("IC2 X", "r_ic2_x_confidence", "ic2_x_peak_amplitude"),
        ("IC2 Y", "r_ic2_y_confidence", "ic2_y_peak_amplitude"),
    ] {
        let Some(conf) = col(columns, confidence) else {
            continue;
        };
        let index: Vec<f32> = (0..conf.len()).map(|i| i as f32).collect();
        let xs = col(columns, peak).unwrap_or(&index);
        let (xs, ys) = select_pairs(xs, conf, gate, state);
        if xs.is_empty() {
            continue;
        }
        let (xmin, xmax) = span(&xs);
        let (ymin, ymax) = span(&ys);
        panels.push(panel(
            format!("{session} {label}"),
            xmin,
            xmax,
            ymin,
            ymax,
            vec![Series::Points {
                xs,
                ys,
                color: BLUE,
                radius: 2.0,
            }],
        ));
    }
    panels
}

fn coverage_panels(
    session: &str,
    columns: &BTreeMap<String, Vec<f32>>,
    state: BeamState,
) -> Vec<Panel> {
    let thresholds: Vec<f32> = (0..=400).map(|step| step as f32 * 0.25).collect();
    let gate = col(columns, "rci_in_trigger").or_else(|| col(columns, "r_beamOk"));
    let mut panels = Vec::new();
    for (label, x_name, y_name) in [
        ("IC1", "r_ic1_x_confidence", "r_ic1_y_confidence"),
        ("IC2", "r_ic2_x_confidence", "r_ic2_y_confidence"),
    ] {
        let metrics = spot_coverage_metrics(
            col(columns, "spot_no"),
            col(columns, x_name),
            col(columns, y_name),
            gate,
            state,
        );
        if metrics.is_empty() {
            continue;
        }
        let percent = coverage_percent(&metrics, &thresholds);
        panels.push(line_panel(
            format!("{session} {label} coverage"),
            &thresholds,
            &percent,
            BLUE,
        ));
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

fn sum_ic3_quads(columns: &BTreeMap<String, Vec<f32>>) -> Option<Vec<f32>> {
    let parts: Vec<&[f32]> = [
        "ic3_current_a",
        "ic3_current_b",
        "ic3_current_c",
        "ic3_current_d",
    ]
    .into_iter()
    .filter_map(|name| col(columns, name))
    .collect();
    if parts.is_empty() {
        return None;
    }
    let n = parts.iter().map(|part| part.len()).max().unwrap_or(0);
    let mut out = vec![0.0f32; n];
    for part in parts {
        for (i, value) in part.iter().enumerate() {
            if value.is_finite() {
                out[i] += value;
            }
        }
    }
    Some(out)
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
        let Series::Polyline { xs, ys, .. } = item else {
            continue;
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

fn select_pairs(
    xs: &[f32],
    ys: &[f32],
    gate: Option<&[f32]>,
    state: BeamState,
) -> (Vec<f32>, Vec<f32>) {
    let n = xs.len().min(ys.len());
    let on = gate.filter(|values| values.len() == n).map(beam_on_mask);
    let mut kept_x = Vec::new();
    let mut kept_y = Vec::new();
    for i in 0..n {
        if !xs[i].is_finite() || !ys[i].is_finite() {
            continue;
        }
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
        if keep {
            kept_x.push(xs[i]);
            kept_y.push(ys[i]);
        }
    }
    (kept_x, kept_y)
}

fn quantile_at(sorted: &[f32], p: f32) -> f32 {
    if sorted.is_empty() {
        return f32::NAN;
    }
    let index = ((sorted.len() - 1) as f32 * p).round() as usize;
    sorted[index.min(sorted.len() - 1)]
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
            "r_ic1_current_dose,rci_in_trigger,ic1_peak_amplitude_x,ic1_peak_amplitude_y,ic2_peak_amplitude_x,ic2_peak_amplitude_y,position_error_x,field_x,field_y\n",
        );
        for i in 0..32 {
            let on = if (8..24).contains(&i) { 1 } else { 0 };
            let current = if on == 1 {
                2.0
            } else {
                (24 - i).max(0) as f32 * 0.05
            };
            timeslice.push_str(&format!(
                "{current},{on},{current},{current},1,1,{err},0.{i},0.2\n",
                err = (i as f32) * 0.01
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
        assert!(table.rows.iter().any(|row| row[1] == "ERROR"));
        assert!(table.rows.iter().any(|row| row[1] == "TIMELINE"));
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
            panel.series.iter().any(|series| match (kind, series) {
                ("line", Series::Polyline { .. }) => true,
                ("bars", Series::Bars { .. }) => true,
                ("points", Series::Points { .. }) => true,
                ("heat", Series::Heatmap { .. }) => true,
                _ => false,
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
        assert!(scene.controls.iter().any(|control| control.id == "mode"));
        assert!(scene.controls.iter().any(|control| control.id == "draw"));
        assert!(scene.controls.iter().any(|control| control.id == "beam"));
    }

    #[test]
    fn sk_req_017_binned_summary_and_replay_share_the_session() {
        let binned = scene_of("binned_summary");
        assert!(has_kind(&binned, "points"));
        assert!(binned.controls.iter().any(|control| control.id == "axis"));
        let replay = scene_of("timeslice_replay");
        assert!(has_kind(&replay, "line"));
        assert!(replay.controls.iter().any(|control| control.id == "scrub"));
    }

    #[test]
    fn sk_req_018_fft_and_audio_share_the_spectrum() {
        assert!(has_kind(&scene_of("ic_fft_analysis"), "line"));
        let audio = scene_of("ic_audio_player");
        assert!(!audio.samples.is_empty());
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
    }

    #[test]
    fn nested_session_reads_its_own_folder() {
        let root = std::env::temp_dir().join(format!("scan-kit-nested-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let nested = root.join("a").join("a");
        std::fs::create_dir_all(nested.join("layer-0").join("run-0")).unwrap();
        std::fs::write(nested.join("input_map.csv"), "energy,charge_req\n70,1\n").unwrap();
        std::fs::write(
            nested.join("layer-0").join("run-0").join("timeslice_data_device_units.csv"),
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
        assert!(scene.panels.iter().any(|panel| panel.title.contains("energy")));
        let _ = std::fs::remove_dir_all(&root);
    }
}
