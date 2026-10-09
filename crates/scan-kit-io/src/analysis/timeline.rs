use std::path::Path;

use scan_kit_core::{
    hv_capacitance_pf, hv_delta_v, hv_expected_pf, hv_firmware_flags, hv_step_window,
    robust_limits, row_mask, scrub_control, scrub_limits, segments_control, segments_from,
    time_end, time_window, welch_psd, BeamGate, Panel, PlotScene, Rank, Segment, Series,
};
use serde_json::Value;

use super::{
    col, control, drew_line, flag, load_csv, panel, placed, scene, session_text, span, stroke, MARK,
};

/// Timeslice rows are 1 ms apart.
const SAMPLE_S: f32 = 0.001;

pub(super) fn timeline(root: &Path, session_ids: &[String], options: &Value) -> PlotScene {
    let options = default_metric(options);
    let owned: Vec<(String, Vec<String>)> = session_ids
        .iter()
        .map(|id| (id.clone(), crate::tables::grain_columns(root, id, true)))
        .collect();
    let headers: Vec<crate::source::SessionCols<'_>> = owned
        .iter()
        .map(|(name, columns)| crate::source::SessionCols { name, columns })
        .collect();
    let picked =
        crate::source::select(crate::source::Shape::YAndX, false, true, &headers, &options);
    let (quantity, channels) = crate::bins::channels_for(&picked.y);
    let grain = replay_grain(&picked.y, picked.frame);
    let keys: Vec<&str> = channels.iter().map(|series| series.key).collect();
    let tables = crate::tables::map_sessions(session_ids, |session| {
        crate::tables::session_columns(root, session, grain, &keys)
    });
    let segments = segments_from(
        &options,
        &[
            Segment::Beam {
                state: BeamGate::Both,
            },
            Segment::Rank { which: Rank::All },
        ],
    );
    // Time traces keep every beam-gated sample. Playback slides a camera across
    // them and draws the playhead slice. Scatter, contour, and density keep
    // those samples too. Scatter draws a slice of the point buffer. Contour and
    // density rebin that same slice. Confidence, coverage, and the spectrum
    // still use the playhead, and catch up once it settles.
    let trace_segments: Vec<Segment> = segments
        .iter()
        .filter(|item| !matches!(item, Segment::Range { column, .. } if column == "time_s"))
        .cloned()
        .collect();
    let window = scrub_limits(&options);
    let end = time_end(
        tables
            .iter()
            .filter_map(|table| table.get("time_s").map(Vec::as_slice)),
    );
    let masks: Vec<Vec<bool>> = tables
        .iter()
        .map(|table| row_mask(table, &trace_segments, &keys))
        .collect();
    let show_fft = flag(&options, "fft", false);
    let fft_masks: Vec<Vec<bool>> = if show_fft {
        tables
            .iter()
            .map(|table| row_mask(table, &segments, &keys))
            .collect()
    } else {
        Vec::new()
    };
    let mut x_lo = f32::MAX;
    let mut x_hi = f32::MIN;
    let mut panels = Vec::new();
    for series in channels {
        if !flag(&options, &channel_id(series.key), true) {
            continue;
        }
        let mut drawn = Vec::new();
        let mut ymin = f32::MAX;
        let mut ymax = f32::MIN;
        for (table, keep) in tables.iter().zip(&masks) {
            let Some(samples) = table.get(series.key) else {
                continue;
            };
            let samples = gated(samples, keep);
            if let Some((lo, hi)) = robust_span(&axis_samples(&samples, window)) {
                ymin = ymin.min(lo);
                ymax = ymax.max(hi);
            }
            if let Some((lo, hi)) = finite_window(&samples, window) {
                x_lo = x_lo.min(lo);
                x_hi = x_hi.max(hi);
            }
            let (xs, ys) = trace(&samples);
            drawn.push(stroke(xs, ys, false));
        }
        if !drew_line(&drawn) {
            continue;
        }
        if ymin > ymax {
            ymin = 0.0;
            ymax = 1.0;
        }
        panels.push(time_panel(
            drawn,
            ymin,
            ymax,
            &crate::bins::axis_label(series.label, quantity),
        ));
        if show_fft {
            let columns: Vec<Vec<f32>> = tables
                .iter()
                .zip(&fft_masks)
                .filter_map(|(table, keep)| {
                    table.get(series.key).map(|samples| gated(samples, keep))
                })
                .collect();
            panels.push(spectrum_panel(&columns));
        }
    }
    let time_count = if show_fft {
        panels.len() / 2
    } else {
        panels.len()
    };
    fit_time_axis(&mut panels, x_lo, x_hi, show_fft);
    let layers = crate::tables::timeline_layers(root, session_ids, true);
    let show_scatter = flag(&options, "scatter", false);
    if show_scatter {
        let query = scatter_query(&options);
        let scatter_owned: Vec<(String, Vec<String>)> = session_ids
            .iter()
            .map(|id| (id.clone(), crate::tables::grain_columns(root, id, true)))
            .collect();
        let scatter_headers: Vec<crate::source::SessionCols<'_>> = scatter_owned
            .iter()
            .map(|(name, columns)| crate::source::SessionCols { name, columns })
            .collect();
        let scatter = crate::source::select(
            crate::source::Shape::Xy,
            false,
            true,
            &scatter_headers,
            &query,
        );
        let windowed = super::distribution::scatter_uses_playhead_window(scatter.xy, &options);
        let filter = if windowed { &segments } else { &trace_segments };
        let (side, side_controls) = super::distribution::scatter_panels(
            root,
            session_ids,
            scatter.xy,
            scatter.grain,
            filter,
            &options,
            !windowed,
        );
        panels.extend(side);
        let mut controls = time_controls(
            &picked.controls,
            channels,
            &options,
            &segments,
            end,
            &layers,
        );
        controls.push(scatter_toggle(true));
        controls.extend(scatter.controls.into_iter().map(scatter_control));
        controls.extend(side_controls);
        return finish(panels, controls, time_count, show_fft);
    }
    let mut controls = time_controls(
        &picked.controls,
        channels,
        &options,
        &segments,
        end,
        &layers,
    );
    controls.push(scatter_toggle(false));
    finish(panels, controls, time_count, show_fft)
}

fn finish(
    panels: Vec<Panel>,
    controls: Vec<scan_kit_core::Control>,
    time_count: usize,
    fft: bool,
) -> PlotScene {
    let rows = if fft {
        time_count.saturating_mul(2)
    } else {
        time_count
    };
    let side = panels.len().saturating_sub(rows);
    let mut scene = scene("Timeline", panels, controls);
    if fft && time_count > 0 && side > 0 {
        scene.columns = 2;
        scene.column_weights = vec![3.0, 1.4, 1.4];
        scene.side = side as u32;
    } else if fft && time_count > 0 {
        scene.columns = 2;
        scene.column_weights = vec![3.0, 1.4];
    } else if side > 0 && time_count > 0 {
        scene.columns = 2;
        scene.column_weights = vec![3.0, 1.4];
        scene.side = side as u32;
    } else {
        scene.columns = 1;
    }
    scene
}

fn time_controls(
    source: &[scan_kit_core::Control],
    channels: &[crate::bins::YSeries],
    options: &Value,
    segments: &[Segment],
    end: f32,
    layers: &[f32],
) -> Vec<scan_kit_core::Control> {
    let mut controls: Vec<_> = source
        .iter()
        .filter(|item| item.id != "x")
        .cloned()
        .collect();
    for series in channels {
        let on = flag(options, &channel_id(series.key), true);
        controls.push(
            control(
                &channel_id(series.key),
                series.label,
                &["Off", "On"],
                on_off(on),
            )
            .grouped("Data Source")
            .checked(),
        );
    }
    controls.push(segments_control(
        segments,
        &[("beam", "Beam"), ("rank", "Rank")],
    ));
    controls.push(fft_toggle(flag(options, "fft", false)));
    controls.push(scrub_control(options, end, layers));
    controls
}

fn fft_toggle(on: bool) -> scan_kit_core::Control {
    control("fft", "FFT", &["Off", "On"], on_off(on))
        .grouped("FFT")
        .checked()
}

fn scatter_toggle(on: bool) -> scan_kit_core::Control {
    control("scatter", "Distribution", &["Off", "On"], on_off(on))
        .grouped("Distribution")
        .checked()
}

fn scatter_control(mut control: scan_kit_core::Control) -> scan_kit_core::Control {
    if control.id == "xy" {
        control.id = "scatter_xy".to_string();
    }
    control.group = "Distribution".to_string();
    control
}

fn scatter_query(options: &Value) -> Value {
    let mut query = serde_json::Map::new();
    if let Some(value) = options.get("scatter_xy") {
        query.insert("xy".into(), value.clone());
    }
    Value::Object(query)
}

fn default_metric(options: &Value) -> Value {
    let mut copy = options.clone();
    let Some(object) = copy.as_object_mut() else {
        return serde_json::json!({"y": "ic_current"});
    };
    let missing = object.get("y").and_then(Value::as_str).is_none()
        && object.get("metric").and_then(Value::as_str).is_none();
    if missing {
        object.insert("y".into(), Value::String("ic_current".into()));
    }
    copy
}

fn replay_grain(metric: &str, frame: &str) -> crate::tables::Grain {
    if metric == "current_ratio" {
        crate::tables::Grain::Layer
    } else if frame == "chamber" {
        crate::tables::Grain::SampleChamber
    } else {
        crate::tables::Grain::Sample
    }
}

fn channel_id(key: &str) -> String {
    format!("ch_{key}")
}

fn on_off(on: bool) -> &'static str {
    if on {
        "On"
    } else {
        "Off"
    }
}

fn gated(samples: &[f32], keep: &[bool]) -> Vec<f32> {
    samples
        .iter()
        .enumerate()
        .map(|(index, value)| {
            if keep.get(index).copied().unwrap_or(true) {
                *value
            } else {
                f32::NAN
            }
        })
        .collect()
}

pub(super) fn hv_transient(root: &Path, session_ids: &[String]) -> PlotScene {
    let groups = crate::tables::map_sessions(session_ids, |session| {
        let mut panels = Vec::new();
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
        panels
    });
    let mut panels = Vec::new();
    for group in groups {
        panels.extend(group);
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

/// Welch spectrum of the samples this row draws, 1 Hz to 500 Hz.
///
/// ponytail: a beam-off gap is dropped, not written as a zero, so the frequency
/// axis is the 1 ms clock of the kept samples. Zero-fill at that clock if a
/// notch from the gate starts to matter.
fn spectrum_panel(columns: &[Vec<f32>]) -> Panel {
    let mut series = Vec::new();
    for samples in columns {
        let kept: Vec<f32> = samples
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .collect();
        if kept.len() < 16 {
            continue;
        }
        let (freqs, psd) = welch_psd(&kept, 1.0 / SAMPLE_S, 4096, 0.5);
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for (freq, power) in freqs.iter().zip(&psd) {
            if (1.0..=500.0).contains(freq) {
                xs.push(*freq);
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
    }
    let mut panel = if series.is_empty() {
        panel(String::new(), 1.0, 500.0, 0.0, 1.0, series)
    } else {
        let mut panel = placed(String::new(), series);
        panel.xmin = 1.0;
        panel.xmax = 500.0;
        panel
    };
    panel.x_label = "Hz".into();
    panel.y_label = "log10 PSD".into();
    panel
}

fn time_panel(series: Vec<Series>, ymin: f32, ymax: f32, y_label: &str) -> Panel {
    let mut panel = panel(String::new(), 0.0, 1.0, ymin, ymax, series);
    // `panel` turns a tiny span into +1. A flat trace should stay near its value.
    panel.ymin = ymin;
    panel.ymax = if ymax > ymin { ymax } else { ymin + 1.0e-6 };
    panel.y_label = y_label.to_owned();
    panel
}

/// Shared time window of the finite samples, with a small margin.
fn fit_time_axis(panels: &mut [Panel], x_lo: f32, x_hi: f32, fft: bool) {
    if x_lo > x_hi {
        return;
    }
    let (xmin, xmax) = time_window(x_lo, x_hi);
    let step = if fft { 2 } else { 1 };
    for item in panels.iter_mut().step_by(step) {
        item.xmin = xmin;
        item.xmax = xmax;
    }
}

fn axis_samples(samples: &[f32], window: Option<(f32, f32)>) -> Vec<f32> {
    samples
        .iter()
        .enumerate()
        .filter_map(|(index, value)| {
            if !value.is_finite() || !inside(index, window) {
                return None;
            }
            Some(*value)
        })
        .collect()
}

fn finite_window(samples: &[f32], window: Option<(f32, f32)>) -> Option<(f32, f32)> {
    let mut lo = f32::MAX;
    let mut hi = f32::MIN;
    for (index, value) in samples.iter().enumerate() {
        if !value.is_finite() || !inside(index, window) {
            continue;
        }
        let t = index as f32 * SAMPLE_S;
        lo = lo.min(t);
        hi = hi.max(t);
    }
    (lo <= hi).then_some((lo, hi))
}

fn inside(index: usize, window: Option<(f32, f32)>) -> bool {
    let Some((lo, hi)) = window else {
        return true;
    };
    let t = index as f32 * SAMPLE_S;
    t >= lo && t <= hi
}

/// One point per sample. Time is the row index, so a gap stays where the file has one.
pub(super) fn trace(samples: &[f32]) -> (Vec<f32>, Vec<f32>) {
    (
        (0..samples.len())
            .map(|index| index as f32 * SAMPLE_S)
            .collect(),
        samples.to_vec(),
    )
}

/// 0.5% tails, then a slim margin. One ADC spike must not set the axis, and a
/// flat trace stays near its value instead of gaining an empty unit of range.
pub(super) fn robust_span(samples: &[f32]) -> Option<(f32, f32)> {
    robust_limits(samples)
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
