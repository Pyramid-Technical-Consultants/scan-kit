use std::collections::BTreeMap;
use std::path::Path;

use scan_kit_core::{
    cloud_series, coverage_percent, percentile_nearest, row_mask, scrub_control, segments_control,
    segments_from, time_end, BeamGate, CloudDraw, Family, Panel, PlotScene, Segment, Series,
    SESSION,
};
use serde_json::Value;

use crate::histogram::{bin_button, hist_bin_count, histogram_panel, BIN_CHOICES};

use super::{
    col, control, drew_line, finite_col, flag, guide, labeled, panel, pick, placed, scene,
    slice_table, spot_table, stroke, text, timeslice_metric, MARK,
};

const DRAW_CHOICES: &[(&str, &str)] = &[
    ("scatter", "Scatter"),
    ("contour", "Contour"),
    ("density", "Density"),
];
const CUTOFF_CHOICES: &[(&str, &str)] = &[("0", "0"), ("5", "5"), ("10", "10"), ("20", "20")];

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
    let segments = segments_from(
        options,
        &[Segment::Beam {
            state: BeamGate::Both,
        }],
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
    let chambers = matches!(
        mode,
        "position" | "position_error" | "position_error_rel" | "sigma"
    );
    let cloud = chambers || matches!(mode, "amplifier" | "amplifier_voltage" | "probe");
    let show_ic1 = flag(options, "ic1", true);
    let show_ic2 = flag(options, "ic2", true);
    let show_plan = flag(options, "plan", true);
    let hist_raw = text(options, "hist_bins", "Auto");
    let bins = hist_bin_count(hist_raw);
    let timeslice_clock = grain == "timeslice"
        || matches!(
            mode,
            "amplifier" | "amplifier_voltage" | "probe" | "confidence" | "coverage"
        );
    let (panels, columns, has_plan) = if mode == "confidence" {
        (
            confidence_scene(root, session_ids, &segments, draw, ramp, cutoff),
            0,
            false,
        )
    } else if mode == "coverage" {
        (coverage_scene(root, session_ids, &segments), 0, false)
    } else {
        column_scene(
            root,
            session_ids,
            mode,
            grain,
            &segments,
            draw,
            ramp,
            cutoff,
            show_ic1,
            show_ic2,
            show_plan,
            bins,
            true,
            timeslice_clock,
        )
    };
    let mut controls = picked.controls;
    if chambers {
        controls.extend(chamber_controls(
            mode,
            has_plan,
            show_plan,
            show_ic1,
            show_ic2,
            "Data Source",
        ));
    }
    if mode != "coverage" {
        controls.extend(style_controls(
            options,
            session_ids.len(),
            draw,
            "Plot Style",
        ));
    }
    if cloud {
        controls.push(
            control("hist_bins", "Bins", BIN_CHOICES, &bin_button(hist_raw)).grouped("Histogram"),
        );
    }
    controls.push(segments_control(&segments, &[("beam", "Beam")]));
    controls.push(scrub_control(
        options,
        clock_end(root, session_ids, mode, grain),
        &crate::tables::timeline_layers(root, session_ids, timeslice_clock),
    ));
    let mut scene = scene("Distribution", panels, controls);
    scene.columns = columns;
    if columns > 0 && scene.panels.len() == columns as usize * 3 {
        scene.row_weights = vec![2.0, 1.0, 1.0];
    }
    scene
}

fn masked_pairs(xs: &[f32], ys: &[f32], keep: &[bool]) -> (Vec<f32>, Vec<f32>) {
    let mut ox = Vec::new();
    let mut oy = Vec::new();
    for (index, (x, y)) in xs.iter().zip(ys).enumerate() {
        if keep.get(index).copied().unwrap_or(true) && x.is_finite() && y.is_finite() {
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
    segments: &[Segment],
    live: bool,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let keep = row_mask(table, segments, &[x_key, y_key]);
    let xs = table.get(x_key).map(Vec::as_slice).unwrap_or(&[]);
    let ys = table.get(y_key).map(Vec::as_slice).unwrap_or(&[]);
    let clock = table.get("time_s").map(Vec::as_slice).unwrap_or(&[]);
    let timed = live && !clock.is_empty();
    let mut ox = Vec::new();
    let mut oy = Vec::new();
    let mut ot = Vec::new();
    for (index, (x, y)) in xs.iter().zip(ys).enumerate() {
        if !keep.get(index).copied().unwrap_or(true) || !x.is_finite() || !y.is_finite() {
            continue;
        }
        if timed {
            let time = clock.get(index).copied().unwrap_or(f32::NAN);
            if !time.is_finite() {
                continue;
            }
            ot.push(time);
        }
        ox.push(*x);
        oy.push(*y);
    }
    (ox, oy, ot)
}

/// Nearest-rank percentile without sorting the whole cloud.
fn percentile_at(values: &mut [f32], portion: f32) -> f32 {
    percentile_nearest(values, portion)
}

pub(super) fn distribution_limits(mode: &str, samples: &[f32]) -> (f32, f32) {
    let (lo, hi) = if mode == "sigma" {
        let mut positive: Vec<f32> = samples
            .iter()
            .copied()
            .filter(|value| *value > 0.0)
            .collect();
        let hi = percentile_at(&mut positive, 0.9995).max(1.0);
        (0.0, hi)
    } else if mode == "position_error" || mode == "position_error_rel" {
        let mut abs: Vec<f32> = samples
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .map(f32::abs)
            .collect();
        let bound = percentile_at(&mut abs, 0.9995).max(1.0);
        (-bound, bound)
    } else {
        let mut finite: Vec<f32> = samples
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .collect();
        if finite.is_empty() {
            (-1.0, 1.0)
        } else {
            let lo = percentile_at(&mut finite, 0.0005);
            let hi = percentile_at(&mut finite, 0.9995);
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
        "amplifier_voltage" => format!("{column} {axis} (V)"),
        "probe" => format!("{axis} (G)"),
        "position_error" | "position_error_rel" => format!("{column} {axis} Error (mm)"),
        "sigma" => format!("{column} {axis} Sigma (mm)"),
        _ => format!("{column} {axis} (mm)"),
    }
}

fn limit_kind(mode: &str) -> &'static str {
    match mode {
        "sigma" => "sigma",
        "position_error" | "position_error_rel" | "amplifier" => "position_error",
        _ => "position",
    }
}

fn grain_table(root: &Path, session: &str, grain: &str) -> super::super::tables::Table {
    if grain == "timeslice" {
        slice_table(root, session)
    } else {
        spot_table(root, session)
    }
}

fn clock_end(root: &Path, session_ids: &[String], mode: &str, grain: &str) -> f32 {
    let tables = if mode == "confidence" {
        crate::tables::map_sessions(session_ids, |session| {
            timeslice_metric(root, session, "peak_amplitude")
        })
    } else if mode == "coverage" {
        crate::tables::map_sessions(session_ids, |session| {
            timeslice_metric(root, session, "fit_confidence")
        })
    } else {
        crate::tables::map_sessions(session_ids, |session| {
            session_table(root, session, mode, grain)
        })
    };
    time_end(
        tables
            .iter()
            .filter_map(|table| table.get("time_s").map(Vec::as_slice)),
    )
}

fn session_table(
    root: &Path,
    session: &str,
    mode: &str,
    grain: &str,
) -> super::super::tables::Table {
    match mode {
        "amplifier" | "amplifier_voltage" => timeslice_metric(root, session, "amplifier_error"),
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

fn heat_color(ramp: u8) -> [f32; 4] {
    if scan_kit_core::is_session(ramp) {
        MARK
    } else {
        [1.0, 1.0, 1.0, 1.0]
    }
}

/// Beam filters stay. The playhead window does not, so a live cloud can slice it.
fn without_playhead(segments: &[Segment]) -> Vec<Segment> {
    segments
        .iter()
        .filter(|item| !matches!(item, Segment::Range { column, .. } if column == "time_s"))
        .cloned()
        .collect()
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
    clouds: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)>,
    /// Marginal histograms of the playhead window. Empty uses `clouds`.
    windowed: Vec<(Vec<f32>, Vec<f32>)>,
}

fn column_specs(
    mode: &str,
    ic1: bool,
    ic2: bool,
    plan: bool,
) -> Vec<(&'static str, &'static str, &'static str)> {
    let mut pairs = Vec::new();
    match mode {
        "position_error" | "position_error_rel" => {
            let (x1, y1, x2, y2) = if mode == "position_error_rel" {
                (
                    "ic1_x_err_rel",
                    "ic1_y_err_rel",
                    "ic2_x_err_rel",
                    "ic2_y_err_rel",
                )
            } else {
                ("ic1_x_err", "ic1_y_err", "ic2_x_err", "ic2_y_err")
            };
            if ic1 {
                pairs.push(("IC1", x1, y1));
            }
            if ic2 {
                pairs.push(("IC2", x2, y2));
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
        "amplifier_voltage" => {
            pairs.push(("Command", "amp_cmd_x", "amp_cmd_y"));
            pairs.push(("Readback", "amp_read_x", "amp_read_y"));
        }
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

/// Timeline's side column is the distribution cloud: the same columns and
/// scatter / contour / density style, without the marginal histograms.
/// A live cloud keeps a time on each sample. Scatter draws a slice of that
/// buffer. Contour and density rebin the same slice.
pub(super) fn scatter_panels(
    root: &Path,
    session_ids: &[String],
    mode: &str,
    grain: &str,
    segments: &[Segment],
    options: &Value,
    live: bool,
) -> (Vec<Panel>, Vec<scan_kit_core::Control>) {
    let draw = pick(options, "draw", "scatter", DRAW_CHOICES);
    let ramp = pick(
        options,
        "ramp",
        "turbo",
        scan_kit_core::choices(Family::Sequential),
    );
    let cutoff = pick(options, "cutoff", "5", CUTOFF_CHOICES)
        .parse::<f32>()
        .unwrap_or(5.0);
    let chambers = matches!(
        mode,
        "position" | "position_error" | "position_error_rel" | "sigma"
    );
    let show_ic1 = flag(options, "ic1", true);
    let show_ic2 = flag(options, "ic2", true);
    let show_plan = flag(options, "plan", true);
    let mut controls = Vec::new();
    let panels = if mode == "confidence" {
        confidence_scene(root, session_ids, segments, draw, ramp, cutoff)
    } else if mode == "coverage" {
        coverage_scene(root, session_ids, segments)
    } else {
        let (panels, _, has_plan) = column_scene(
            root,
            session_ids,
            mode,
            grain,
            segments,
            draw,
            ramp,
            cutoff,
            show_ic1,
            show_ic2,
            show_plan,
            1,
            false,
            live,
        );
        if chambers {
            controls.extend(chamber_controls(
                mode,
                has_plan,
                show_plan,
                show_ic1,
                show_ic2,
                "Distribution",
            ));
        }
        panels
    };
    if mode != "coverage" {
        controls.extend(style_controls(
            options,
            session_ids.len(),
            draw,
            "Distribution",
        ));
    }
    (panels, controls)
}

/// Confidence and coverage are aggregates of the playhead window.
/// Scatter, contour, and density carry the full timed cloud and follow in the plot.
pub(super) fn scatter_uses_playhead_window(mode: &str, _options: &Value) -> bool {
    matches!(mode, "confidence" | "coverage")
}

fn style_controls(
    options: &Value,
    sessions: usize,
    draw: &str,
    group: &str,
) -> Vec<scan_kit_core::Control> {
    let mut controls = vec![labeled("draw", "Style", DRAW_CHOICES, draw).grouped(group)];
    if draw == "density" && sessions == 1 {
        let ramp = pick(
            options,
            "ramp",
            "turbo",
            scan_kit_core::choices(Family::Sequential),
        );
        controls.push(
            labeled(
                "ramp",
                "Ramp",
                scan_kit_core::choices(Family::Sequential),
                ramp,
            )
            .grouped(group),
        );
    }
    if draw == "contour" {
        let cutoff_id = pick(options, "cutoff", "5", CUTOFF_CHOICES);
        controls
            .push(labeled("cutoff", "Contour Cutoff", CUTOFF_CHOICES, cutoff_id).grouped(group));
    }
    controls
}

fn chamber_controls(
    mode: &str,
    has_plan: bool,
    show_plan: bool,
    show_ic1: bool,
    show_ic2: bool,
    group: &str,
) -> Vec<scan_kit_core::Control> {
    let mut controls = Vec::new();
    if mode == "position" && has_plan {
        controls.push(
            control("plan", "Plan", &["Off", "On"], on_off(show_plan))
                .grouped(group)
                .checked(),
        );
    }
    controls.push(
        control("ic1", "IC1", &["Off", "On"], on_off(show_ic1))
            .grouped(group)
            .checked(),
    );
    controls.push(
        control("ic2", "IC2", &["Off", "On"], on_off(show_ic2))
            .grouped(group)
            .checked(),
    );
    controls
}

fn column_scene(
    root: &Path,
    session_ids: &[String],
    mode: &str,
    grain: &str,
    segments: &[Segment],
    draw: &str,
    ramp: &str,
    cutoff: f32,
    ic1: bool,
    ic2: bool,
    plan: bool,
    bins: usize,
    histograms: bool,
    live: bool,
) -> (Vec<Panel>, u32, bool) {
    let tables = crate::tables::map_sessions(session_ids, |session| {
        session_table(root, session, mode, grain)
    });
    let has_plan = tables
        .iter()
        .any(|table| finite_col(table, "plan_x").is_some());
    let drawn_plan = mode == "position" && plan && has_plan;
    let stripped = live.then(|| without_playhead(segments));
    let mut columns = Vec::new();
    for (name, x_key, y_key) in column_specs(mode, ic1, ic2, drawn_plan) {
        let mut clouds = Vec::new();
        let mut windowed = Vec::new();
        for table in &tables {
            let (xs, ys, times) = if let Some(full) = stripped.as_deref() {
                let timed = kept_pairs(table, x_key, y_key, full, true);
                if timed.2.is_empty() {
                    kept_pairs(table, x_key, y_key, segments, false)
                } else {
                    timed
                }
            } else {
                kept_pairs(table, x_key, y_key, segments, false)
            };
            if xs.is_empty() {
                continue;
            }
            if histograms && !times.is_empty() {
                let (hx, hy, _) = kept_pairs(table, x_key, y_key, segments, false);
                if !hx.is_empty() {
                    windowed.push((hx, hy));
                }
            }
            clouds.push((xs, ys, times));
        }
        if !clouds.is_empty() {
            columns.push(DrawnColumn {
                name,
                clouds,
                windowed,
            });
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
        for (xs, ys, _) in &column.clouds {
            samples.extend(xs.iter().copied());
            samples.extend(ys.iter().copied());
        }
    }
    let (lo, hi) = distribution_limits(limit_kind(mode), &samples);
    let hist_guides: Vec<f32> = if mode == "position_error" || mode == "position_error_rel" {
        vec![0.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0]
    } else {
        Vec::new()
    };
    let mut tops = Vec::new();
    let mut x_hists = Vec::new();
    let mut y_hists = Vec::new();
    let ramp_id = heat_ramp(session_ids.len(), ramp);
    let kind = match draw {
        "density" => CloudDraw::Density {
            ramp: ramp_id,
            color: heat_color(ramp_id),
            x0: lo,
            x1: hi,
            y0: lo,
            y1: hi,
        },
        "contour" => CloudDraw::Contour { cutoff },
        _ => CloudDraw::Scatter {
            color: MARK,
            radius: 2.0,
        },
    };
    for column in &columns {
        let mut series = Vec::new();
        if matches!(
            mode,
            "position"
                | "position_error"
                | "position_error_rel"
                | "amplifier"
                | "amplifier_voltage"
                | "probe"
        ) {
            series.push(guide(vec![lo, hi], vec![0.0, 0.0]));
            series.push(guide(vec![0.0, 0.0], vec![lo, hi]));
        }
        if (mode == "position_error" || mode == "position_error_rel") && column.name != "Plan" {
            series.push(reference_ring());
        }
        if mode == "sigma" {
            series.push(guide(vec![lo, hi], vec![lo, hi]));
        }
        for (xs, ys, times) in &column.clouds {
            series.extend(cloud_series(xs, ys, times, kind));
        }
        let mut top = panel(String::new(), lo, hi, lo, hi, series);
        top.equal = true;
        top.x_label = axis_name(column.name, "X", mode);
        top.y_label = axis_name(column.name, "Y", mode);
        tops.push(top);
        let xs: Vec<&[f32]> = if column.windowed.is_empty() {
            column
                .clouds
                .iter()
                .map(|(xs, _, _)| xs.as_slice())
                .collect()
        } else {
            column
                .windowed
                .iter()
                .map(|(xs, _)| xs.as_slice())
                .collect()
        };
        let ys: Vec<&[f32]> = if column.windowed.is_empty() {
            column
                .clouds
                .iter()
                .map(|(_, ys, _)| ys.as_slice())
                .collect()
        } else {
            column
                .windowed
                .iter()
                .map(|(_, ys)| ys.as_slice())
                .collect()
        };
        if histograms {
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
    }
    let count = tops.len() as u32;
    if histograms {
        tops.extend(x_hists);
        tops.extend(y_hists);
    }
    (tops, count, has_plan)
}

fn confidence_scene(
    root: &Path,
    session_ids: &[String],
    segments: &[Segment],
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
    let peaks = crate::tables::map_sessions(session_ids, |session| {
        timeslice_metric(root, session, "peak_amplitude")
    });
    let confidence = crate::tables::map_sessions(session_ids, |session| {
        timeslice_metric(root, session, "fit_confidence")
    });
    let mut panels = Vec::new();
    for (title, peak_key, conf_key) in axes {
        let mut clouds = Vec::new();
        for (peaks, confidence) in peaks.iter().zip(&confidence) {
            let mut keep = row_mask(peaks, segments, &[peak_key]);
            let other = row_mask(confidence, segments, &[conf_key]);
            for (slot, pass) in keep.iter_mut().zip(&other) {
                *slot = *slot && *pass;
            }
            let (xs, ys) = masked_pairs(
                peaks.get(peak_key).map(Vec::as_slice).unwrap_or(&[]),
                confidence.get(conf_key).map(Vec::as_slice).unwrap_or(&[]),
                &keep,
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
            let ramp_id = heat_ramp(session_ids.len(), ramp);
            let kind = if draw == "density" {
                CloudDraw::Density {
                    ramp: ramp_id,
                    color: heat_color(ramp_id),
                    x0,
                    x1,
                    y0,
                    y1,
                }
            } else {
                CloudDraw::Contour { cutoff }
            };
            let series = clouds
                .iter()
                .flat_map(|(xs, ys)| cloud_series(xs, ys, &[], kind))
                .collect::<Vec<_>>();
            if !series.is_empty() {
                panels.push(panel(title.to_owned(), x0, x1, y0, y1, series));
            }
            continue;
        }
        let series = clouds
            .iter()
            .flat_map(|(xs, ys)| {
                cloud_series(
                    xs,
                    ys,
                    &[],
                    CloudDraw::Scatter {
                        color: MARK,
                        radius: 2.0,
                    },
                )
            })
            .collect();
        panels.push(placed(title.to_owned(), series));
    }
    panels
}

fn coverage_scene(root: &Path, session_ids: &[String], segments: &[Segment]) -> Vec<Panel> {
    let thresholds: Vec<f32> = (0..=400).map(|step| step as f32 * 0.25).collect();
    let mut panels = Vec::new();
    for (title, x_key, y_key) in [
        ("IC1 Coverage", "ic1_x_confidence", "ic1_y_confidence"),
        ("IC2 Coverage", "ic2_x_confidence", "ic2_y_confidence"),
    ] {
        let loaded = crate::tables::map_sessions(session_ids, |session| {
            timeslice_metric(root, session, "fit_confidence")
        });
        let mut series = Vec::new();
        for table in loaded {
            let keep = row_mask(&table, segments, &[x_key, y_key]);
            let metrics =
                spot_coverage_metrics(None, col(&table, x_key), col(&table, y_key), &keep);
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
    keep: &[bool],
) -> Vec<f32> {
    let n = x_conf
        .map(|values| values.len())
        .unwrap_or(0)
        .max(y_conf.map(|values| values.len()).unwrap_or(0));
    if n == 0 {
        return Vec::new();
    }
    let mut order = BTreeMap::<i32, usize>::new();
    let mut max_x = Vec::new();
    let mut max_y = Vec::new();
    for i in 0..n {
        if !keep.get(i).copied().unwrap_or(true) {
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
