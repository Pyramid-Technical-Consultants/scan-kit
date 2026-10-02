//! Binned Summary, matching the 1.8 metric groups, binning, and glyphs.
//!
//! ponytail: sigma expected values on the spot frame come from a string scan of
//! devices.xml. Timeslice isocenter position uses the spot-file strip-to-mm fit,
//! and a duplicate `spot_no` keeps pandas' `.1` suffix.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use scan_kit_core::{
    assign_bin_centers, box_stats, column_scale_factor, dose_error_pct, dose_ratio_pct, g2_ic2_mm,
    histogram, linear_fit, quantile_edges, remap, remap_g2_raw, remap_g3_raw,
    resolve_concept_column, scale_column, Control, Panel, PlotScene, Series,
};
use serde_json::Value;

use super::discover;

const OFFSET: f32 = 0.35;
const BOX_WIDTH: f32 = 0.3;
const VIOLIN_WIDTH: f32 = 0.65;
const GATE_ABS_MU: f64 = 0.002;
const MOD_Z: f32 = 3.5;

struct YSeries {
    key: &'static str,
    label: &'static str,
}

struct YGroup {
    id: &'static str,
    label: &'static str,
    series: &'static [YSeries],
    /// Include 0 in the y range so a gridline lands on zero.
    zero: bool,
    /// Beam and domain filters apply.
    filter: bool,
    timeslice: bool,
}

const DOSE_ERROR: &[YSeries] = &[
    YSeries {
        key: "ic1_dose_err_pct",
        label: "IC1 (%)",
    },
    YSeries {
        key: "ic2_dose_err_pct",
        label: "IC2 (%)",
    },
    YSeries {
        key: "ic3_dose_err_pct",
        label: "IC3 (%)",
    },
];
const DOSE_RATIO: &[YSeries] = &[
    YSeries {
        key: "ic21_ratio",
        label: "IC2/IC1 (%)",
    },
    YSeries {
        key: "ic31_ratio",
        label: "IC3/IC1 (%)",
    },
    YSeries {
        key: "ic32_ratio",
        label: "IC3/IC2 (%)",
    },
];
const DOSE_RATE: &[YSeries] = &[YSeries {
    key: "mu_rate",
    label: "MU/S",
}];
const CURRENT: &[YSeries] = &[
    YSeries {
        key: "ic1_current",
        label: "IC1",
    },
    YSeries {
        key: "ic2_current",
        label: "IC2",
    },
    YSeries {
        key: "ic3_current",
        label: "IC3",
    },
];
const POSITION: &[YSeries] = &[
    YSeries {
        key: "ic1_x_err",
        label: "IC1 X",
    },
    YSeries {
        key: "ic1_y_err",
        label: "IC1 Y",
    },
    YSeries {
        key: "ic2_x_err",
        label: "IC2 X",
    },
    YSeries {
        key: "ic2_y_err",
        label: "IC2 Y",
    },
];
const SIGMA: &[YSeries] = &[
    YSeries {
        key: "ic1_sig_x",
        label: "IC1 SX",
    },
    YSeries {
        key: "ic1_sig_y",
        label: "IC1 SY",
    },
    YSeries {
        key: "ic2_sig_x",
        label: "IC2 SX",
    },
    YSeries {
        key: "ic2_sig_y",
        label: "IC2 SY",
    },
];
const SIGMA_ERR: &[YSeries] = &[
    YSeries {
        key: "ic1_sig_x_err",
        label: "IC1 SX ERR",
    },
    YSeries {
        key: "ic1_sig_y_err",
        label: "IC1 SY ERR",
    },
    YSeries {
        key: "ic2_sig_x_err",
        label: "IC2 SX ERR",
    },
    YSeries {
        key: "ic2_sig_y_err",
        label: "IC2 SY ERR",
    },
];
const IC12: &[YSeries] = &[
    YSeries {
        key: "ic12_x_diff",
        label: "DX",
    },
    YSeries {
        key: "ic12_y_diff",
        label: "DY",
    },
];
const CONFIDENCE: &[YSeries] = &[
    YSeries {
        key: "ic1_x_confidence",
        label: "IC1 X",
    },
    YSeries {
        key: "ic1_y_confidence",
        label: "IC1 Y",
    },
    YSeries {
        key: "ic2_x_confidence",
        label: "IC2 X",
    },
    YSeries {
        key: "ic2_y_confidence",
        label: "IC2 Y",
    },
];
const PEAK: &[YSeries] = &[
    YSeries {
        key: "ic1_x_peak",
        label: "IC1 X",
    },
    YSeries {
        key: "ic1_y_peak",
        label: "IC1 Y",
    },
    YSeries {
        key: "ic2_x_peak",
        label: "IC2 X",
    },
    YSeries {
        key: "ic2_y_peak",
        label: "IC2 Y",
    },
];
const AMPLIFIER: &[YSeries] = &[
    YSeries {
        key: "amp_x",
        label: "X",
    },
    YSeries {
        key: "amp_y",
        label: "Y",
    },
];
const PROBE: &[YSeries] = &[
    YSeries {
        key: "field_x",
        label: "X",
    },
    YSeries {
        key: "field_y",
        label: "Y",
    },
];
const SPOT_TIME: &[YSeries] = &[
    YSeries {
        key: "spot_time",
        label: "TOTAL MS",
    },
    YSeries {
        key: "beam_on_time",
        label: "BEAM ON",
    },
    YSeries {
        key: "overhead_time",
        label: "OVERHEAD",
    },
];

const GROUPS: &[YGroup] = &[
    YGroup {
        id: "dose_error",
        label: "Dose Error (%)",
        series: DOSE_ERROR,
        zero: true,
        filter: true,
        timeslice: false,
    },
    YGroup {
        id: "dose_ratio",
        label: "Dose Ratios",
        series: DOSE_RATIO,
        zero: true,
        filter: true,
        timeslice: false,
    },
    YGroup {
        id: "dose_rate",
        label: "Dose Rate (MU/s)",
        series: DOSE_RATE,
        zero: false,
        filter: false,
        timeslice: false,
    },
    YGroup {
        id: "current_ratio",
        label: "Current Ratios (%)",
        series: DOSE_RATIO,
        zero: true,
        filter: true,
        timeslice: true,
    },
    YGroup {
        id: "ic_current",
        label: "IC Current (nA)",
        series: CURRENT,
        zero: true,
        filter: true,
        timeslice: true,
    },
    YGroup {
        id: "fit_confidence",
        label: "Fit Confidence",
        series: CONFIDENCE,
        zero: true,
        filter: true,
        timeslice: true,
    },
    YGroup {
        id: "peak_amplitude",
        label: "Peak Amplitude",
        series: PEAK,
        zero: true,
        filter: true,
        timeslice: true,
    },
    YGroup {
        id: "amplifier_error",
        label: "Amplifier Error (V)",
        series: AMPLIFIER,
        zero: true,
        filter: true,
        timeslice: true,
    },
    YGroup {
        id: "probe_field",
        label: "Probe Field (G)",
        series: PROBE,
        zero: true,
        filter: true,
        timeslice: true,
    },
    YGroup {
        id: "position_error",
        label: "Position Error (mm)",
        series: POSITION,
        zero: true,
        filter: true,
        timeslice: false,
    },
    YGroup {
        id: "sigma",
        label: "Sigma (mm)",
        series: SIGMA,
        zero: true,
        filter: true,
        timeslice: false,
    },
    YGroup {
        id: "sigma_error",
        label: "Sigma Error (mm)",
        series: SIGMA_ERR,
        zero: true,
        filter: true,
        timeslice: false,
    },
    YGroup {
        id: "ic12_pos_diff",
        label: "IC2-IC1 Position (mm)",
        series: IC12,
        zero: true,
        filter: true,
        timeslice: false,
    },
    YGroup {
        id: "spot_time",
        label: "Spot Delivery Time",
        series: SPOT_TIME,
        zero: true,
        filter: true,
        timeslice: false,
    },
];

struct Sheet {
    num: BTreeMap<String, Vec<f32>>,
    wide: BTreeMap<String, Vec<f64>>,
}

const METRIC_CHOICES: &[(&str, &str)] = &[
    ("dose_error", "Dose Error (%)"),
    ("dose_ratio", "Dose Ratios"),
    ("dose_rate", "Dose Rate (MU/s)"),
    ("current_ratio", "Current Ratios (%)"),
    ("ic_current", "IC Current (nA)"),
    ("fit_confidence", "Fit Confidence"),
    ("peak_amplitude", "Peak Amplitude"),
    ("amplifier_error", "Amplifier Error (V)"),
    ("probe_field", "Probe Field (G)"),
    ("position_error", "Position Error (mm)"),
    ("sigma", "Sigma (mm)"),
    ("sigma_error", "Sigma Error (mm)"),
    ("ic12_pos_diff", "IC2-IC1 Position (mm)"),
    ("spot_time", "Spot Delivery Time"),
];
const X_CHOICES: &[(&str, &str)] = &[
    ("energy", "Energy"),
    ("target_mu", "Target MU"),
    ("spot_time", "Spot time"),
    ("radius", "Radius"),
];
const GLYPH_CHOICES: &[(&str, &str)] = &[
    ("violin", "Violin"),
    ("box", "Box"),
    ("mean", "Mean"),
    ("scatter", "Scatter"),
    ("contour", "Contour"),
];
const SOURCE_CHOICES: &[(&str, &str)] = &[
    ("iso", "Spot — Isocenter"),
    ("chamber", "Spot — Chamber"),
    ("timeslice_iso", "Timeslice — Isocenter"),
    ("timeslice_chamber", "Timeslice — Chamber"),
];
/// Spot vs timeslice, then isocenter vs chamber when a metric has both.
/// Same split as the Python granularity control (`GRANULARITY_SOURCES`).
const COARSE_CHOICES: &[(&str, &str)] = &[("spot", "Spot"), ("timeslice", "Timeslice")];
const FRAME_CHOICES: &[(&str, &str)] = &[("iso", "Isocenter"), ("chamber", "Chamber")];

fn sources_for(metric: &str) -> &'static [(&'static str, &'static str)] {
    match metric {
        "current_ratio" | "ic_current" | "fit_confidence" | "peak_amplitude"
        | "amplifier_error" | "probe_field" => &SOURCE_CHOICES[2..3],
        "position_error" | "sigma" | "sigma_error" => &SOURCE_CHOICES[..3],
        "ic12_pos_diff" => SOURCE_CHOICES,
        _ => &SOURCE_CHOICES[..1],
    }
}

fn coarse_of(source: &str) -> &'static str {
    if source.starts_with("timeslice") {
        "timeslice"
    } else {
        "spot"
    }
}

fn frame_of(source: &str) -> &'static str {
    if source == "chamber" || source.ends_with("_chamber") {
        "chamber"
    } else {
        "iso"
    }
}

struct YChoice {
    metric: &'static str,
    frame: &'static str,
    label: String,
}

/// One Y row per metric. Isocenter and chamber are extra rows when both exist.
fn y_choices(coarse: &str) -> Vec<YChoice> {
    let mut choices = Vec::new();
    for &(metric, base) in METRIC_CHOICES {
        let frames = frames_for(metric, coarse);
        if frames.is_empty() {
            continue;
        }
        let qualify = frames.len() > 1;
        for (frame, frame_label) in frames {
            let label = if qualify {
                format!("{base} ({frame_label})")
            } else {
                base.to_string()
            };
            choices.push(YChoice {
                metric,
                frame,
                label,
            });
        }
    }
    choices
}

fn split_frame_suffix(raw: &str) -> (&str, Option<&'static str>) {
    if let Some(bare) = raw.strip_suffix(" (Chamber)") {
        (bare, Some("chamber"))
    } else if let Some(bare) = raw.strip_suffix(" (Isocenter)") {
        (bare, Some("iso"))
    } else {
        (raw, None)
    }
}

fn metric_label(id: &str) -> &str {
    METRIC_CHOICES
        .iter()
        .find(|(key, _)| *key == id)
        .map(|(_, label)| *label)
        .unwrap_or(id)
}

fn resolve_y(
    choices: &[YChoice],
    options: &Value,
    concrete: Option<&str>,
) -> (&'static str, &'static str) {
    let Some(raw) = options.get("metric").and_then(Value::as_str) else {
        return choices
            .iter()
            .find(|choice| choice.metric == "dose_error")
            .or_else(|| choices.first())
            .map(|choice| (choice.metric, choice.frame))
            .unwrap_or(("dose_error", "iso"));
    };
    if let Some(exact) = choices.iter().find(|choice| choice.label == raw) {
        return (exact.metric, exact.frame);
    }
    let (bare, suffix_frame) = split_frame_suffix(raw);
    let frame_hint = suffix_frame
        .or_else(|| pick_in(options, "frame", FRAME_CHOICES))
        .or_else(|| concrete.map(frame_of));
    let hits: Vec<_> = choices
        .iter()
        .filter(|choice| {
            choice.metric == raw || choice.metric == bare || metric_label(choice.metric) == bare
        })
        .collect();
    if let Some(frame) = frame_hint {
        if let Some(hit) = hits.iter().find(|choice| choice.frame == frame) {
            return (hit.metric, hit.frame);
        }
    }
    hits.first()
        .map(|choice| (choice.metric, choice.frame))
        .unwrap_or((
            choices
                .first()
                .map(|choice| choice.metric)
                .unwrap_or("dose_error"),
            choices.first().map(|choice| choice.frame).unwrap_or("iso"),
        ))
}

fn frames_for(metric: &str, coarse: &str) -> Vec<(&'static str, &'static str)> {
    FRAME_CHOICES
        .iter()
        .copied()
        .filter(|(frame, _)| {
            sources_for(metric)
                .iter()
                .any(|(source, _)| coarse_of(source) == coarse && frame_of(source) == *frame)
        })
        .collect()
}
pub(crate) const BEAM_CHOICES: &[(&str, &str)] = &[
    ("beam_on", "Beam On"),
    ("beam_off", "Beam Off"),
    ("beam_both", "Both"),
];
const DOMAIN_CHOICES: &[(&str, &str)] = &[
    ("all", "All"),
    ("lower_95", "Lower 95%"),
    ("upper_95", "Upper 5%"),
    ("mad_outliers", "MAD Outliers"),
];

#[derive(Clone, Copy, PartialEq, Eq)]
enum Trend {
    Off,
    Linear,
    Polynomial,
}

fn trend_mode(options: &Value) -> Trend {
    match text(options, "trend", "Off") {
        "Linear" | "linear" | "On" | "on" | "true" | "1" => Trend::Linear,
        "Polynomial" | "polynomial" => Trend::Polynomial,
        _ => Trend::Off,
    }
}

fn trend_value(trend: Trend) -> &'static str {
    match trend {
        Trend::Off => "Off",
        Trend::Linear => "Linear",
        Trend::Polynomial => "Polynomial",
    }
}

/// Glyph marks for one data identity. Trend and interlock are drawn on top.
///
/// ponytail: one slot, keyed by the data options plus spot/map fingerprints and
/// the session directory mtime. A timeslice frame that changes without touching
/// those stays stale until the directory mtime moves. Upgrade path: hash the
/// frame list.
#[derive(Clone)]
struct Prepared {
    key: String,
    tables: Vec<BTreeMap<String, Vec<f32>>>,
    categories: Vec<f32>,
    xmin: f32,
    xmax: f32,
    x_labels: Vec<String>,
    glyph: String,
    x_column: String,
    raw_x: bool,
    series: Vec<PreparedSeries>,
}

#[derive(Clone)]
struct PreparedSeries {
    key: String,
    label: String,
    y_label: String,
    glyphs: Vec<Vec<Series>>,
    contour: Vec<Series>,
    ymin: f32,
    ymax: f32,
}

fn prepared_cache() -> &'static std::sync::Mutex<Option<Prepared>> {
    static CACHE: std::sync::OnceLock<std::sync::Mutex<Option<Prepared>>> =
        std::sync::OnceLock::new();
    CACHE.get_or_init(|| std::sync::Mutex::new(None))
}

fn cached_prepared(key: String, build: impl FnOnce() -> Prepared) -> Prepared {
    if let Some(hit) = prepared_cache()
        .lock()
        .unwrap_or_else(|err| err.into_inner())
        .as_ref()
        .filter(|item| item.key == key)
        .cloned()
    {
        return hit;
    }
    let mut built = build();
    built.key = key;
    *prepared_cache()
        .lock()
        .unwrap_or_else(|err| err.into_inner()) = Some(built.clone());
    built
}

#[allow(clippy::too_many_arguments)]
fn prepared_key(
    root: &Path,
    session_ids: &[String],
    metric: &str,
    source: &str,
    x_column: &str,
    bins: &BinChoice,
    glyph: &str,
    domain: &str,
    beam: &str,
    cutoff: f32,
) -> String {
    let mut stamp = String::new();
    for session in session_ids {
        stamp.push_str(session);
        stamp.push(':');
        stamp.push_str(&session_stamp(root, session).to_string());
        stamp.push(';');
    }
    format!(
        "{}|{stamp}|{metric}|{source}|{x_column}|{}|{glyph}|{domain}|{beam}|{cutoff}",
        root.display(),
        bin_label(bins),
    )
}

fn session_stamp(root: &Path, session: &str) -> u128 {
    let dir = discover::session_directory(root, session);
    let mut stamp = mtime_ns(&dir);
    for name in ["spot_data.csv", "input_map.csv"] {
        if let Some((len, ns, head)) = file_fingerprint(&dir.join(name)) {
            stamp ^= ns ^ u128::from(len) ^ u128::from(head);
        }
    }
    stamp
}

fn mtime_ns(path: &Path) -> u128 {
    std::fs::metadata(path)
        .and_then(|meta| meta.modified())
        .ok()
        .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|time| time.as_nanos())
        .unwrap_or(0)
}

pub(crate) fn spot_table(root: &Path, session: &str) -> BTreeMap<String, Vec<f32>> {
    load_spot(root, session, false, false)
}

pub(crate) fn slice_table(root: &Path, session: &str) -> BTreeMap<String, Vec<f32>> {
    load_slice_metric(root, session, "position_error", false)
}

pub(crate) fn timeslice_metric(
    root: &Path,
    session: &str,
    metric: &str,
) -> BTreeMap<String, Vec<f32>> {
    load_timeslice(root, session, metric)
}

/// Current, confidence, peak, field, and amplifier columns from one timeslice read.
pub(crate) fn timeslice_signals(root: &Path, session: &str) -> BTreeMap<String, Vec<f32>> {
    let Some((files, energies, layers)) = timeslice_inputs(root, session) else {
        return BTreeMap::new();
    };
    let sheets = read_sheets(&files, timeline_column);
    let mut out = current_from(&sheets, &energies, &layers);
    for metric in [
        "fit_confidence",
        "peak_amplitude",
        "amplifier_error",
        "probe_field",
    ] {
        for (key, values) in signal_from(&sheets, &energies, &layers, metric) {
            out.entry(key).or_insert(values);
        }
    }
    out
}

pub(crate) fn binned_summary(root: &Path, session_ids: &[String], options: &Value) -> PlotScene {
    let metric_hint = pick(options, "metric", "dose_error", METRIC_CHOICES);
    let concrete = options
        .get("source")
        .and_then(Value::as_str)
        .and_then(|raw| {
            SOURCE_CHOICES
                .iter()
                .find(|(id, label)| *id == raw || *label == raw)
                .map(|(id, _)| *id)
        });
    let coarse = pick_in(options, "source", COARSE_CHOICES)
        .or_else(|| concrete.map(coarse_of))
        .unwrap_or_else(|| coarse_of(sources_for(metric_hint)[0].0));
    let y_opts = y_choices(coarse);
    let (metric, frame) = resolve_y(&y_opts, options, concrete);
    let group = GROUPS
        .iter()
        .find(|group| group.id == metric)
        .unwrap_or(&GROUPS[0]);
    let x_id = pick(options, "x", "energy", X_CHOICES);
    let glyph = pick(options, "glyph", "violin", GLYPH_CHOICES);
    let beam_default = if coarse == "timeslice" {
        "beam_on"
    } else {
        "beam_both"
    };
    let beam = pick(options, "beam", beam_default, BEAM_CHOICES);
    let domain = pick(options, "domain", "all", DOMAIN_CHOICES);
    let trend = trend_mode(options);
    let hist = flag(options, "hist", false);
    let corr = flag(options, "corr", false);
    let mut interlock = flag(options, "interlock", false);
    let bins = bin_choice(options);
    let hist_bins = text(options, "hist_bins", "30")
        .parse::<usize>()
        .unwrap_or(30)
        .clamp(5, 80);
    let cutoff = text(options, "cutoff", "5")
        .parse::<f32>()
        .unwrap_or(5.0)
        .clamp(0.0, 90.0);
    let source = sources_for(group.id)
        .iter()
        .find(|(id, _)| coarse_of(id) == coarse && frame_of(id) == frame)
        .map(|(id, _)| *id)
        .unwrap_or_else(|| {
            sources_for(group.id)
                .iter()
                .find(|(id, _)| coarse_of(id) == coarse)
                .map(|(id, _)| *id)
                .unwrap_or(sources_for(group.id)[0].0)
        });
    let shared_bins = flag(options, "shared", false);
    let geometry = matches!(source, "timeslice_iso" | "timeslice_chamber")
        && matches!(
            group.id,
            "position_error" | "sigma" | "sigma_error" | "ic12_pos_diff"
        );
    let energy_only = geometry || timeslice_energy_only(group.id);
    let chamber = source == "chamber"
        && matches!(
            group.id,
            "position_error" | "sigma" | "sigma_error" | "ic12_pos_diff"
        );
    let x_id = if energy_only { "energy" } else { x_id };
    if !interlock_ok(group.id, x_id, glyph) {
        interlock = false;
    }
    let x_column = match x_id {
        "target_mu" => "target_mu",
        "spot_time" => "spot_time",
        "radius" => "radius",
        _ => "energy",
    };
    let raw_x = glyph == "scatter" || glyph == "contour";

    // Trend and interlock only add guides. The glyph geometry stays cached.
    let key = prepared_key(
        root,
        session_ids,
        group.id,
        source,
        x_column,
        &bins,
        glyph,
        domain,
        beam,
        cutoff,
    );
    let prepared = cached_prepared(key, || {
        let load_one = |session: &String| {
            let mut table = if geometry {
                load_slice_metric(root, session, group.id, source == "timeslice_chamber")
            } else if group.timeslice {
                load_timeslice(root, session, group.id)
            } else if group.id == "dose_rate" {
                dose_rate_table(&load_spot(root, session, false, false))
            } else {
                load_spot(root, session, chamber, group.id == "spot_time")
            };
            if group.filter {
                let keys: Vec<&str> = group.series.iter().map(|series| series.key).collect();
                apply_filter(&mut table, &keys, domain, beam);
            }
            table
        };
        // A handful of sessions, one thread each. The spot cache covers a repeat open.
        let mut tables = if session_ids.len() < 2 {
            session_ids.iter().map(load_one).collect::<Vec<_>>()
        } else {
            let mut tables = Vec::with_capacity(session_ids.len());
            std::thread::scope(|scope| {
                let mut joins = Vec::with_capacity(session_ids.len());
                for session in session_ids {
                    joins.push(scope.spawn(|| load_one(session)));
                }
                for join in joins {
                    tables.push(join.join().unwrap());
                }
            });
            tables
        };
        let categories = if raw_x {
            Vec::new()
        } else {
            assign_x(&mut tables, x_column, &bins)
        };
        let present: Vec<&YSeries> = group
            .series
            .iter()
            .filter(|series| tables.iter().any(|table| has_finite(table.get(series.key))))
            .collect();
        let (xmin, xmax) = x_span(&tables, &categories, raw_x, x_column, present.len());
        let x_labels = if raw_x {
            Vec::new()
        } else {
            categories
                .iter()
                .copied()
                .map(scan_kit_core::format_tick)
                .collect()
        };
        let series = present
            .iter()
            .map(|series| {
                let mut ys = Vec::new();
                let glyphs = tables
                    .iter()
                    .enumerate()
                    .map(|(index, table)| {
                        let Some(y) = table.get(series.key) else {
                            return Vec::new();
                        };
                        if !has_finite(Some(y)) {
                            return Vec::new();
                        }
                        ys.extend(y.iter().copied().filter(|value| value.is_finite()));
                        let color = [
                            0.8,
                            0.8,
                            0.8,
                            if glyph == "violin" {
                                0.55
                            } else if glyph == "scatter" {
                                0.4
                            } else {
                                1.0
                            },
                        ];
                        match glyph {
                            "mean" => mean_series(table, y, &categories, index, color),
                            "scatter" => vec![scatter_series(table, y, x_column, color)],
                            "contour" => Vec::new(),
                            "box" => box_series(table, y, &categories, index, color),
                            _ => violin_series(table, y, &categories, index, color),
                        }
                    })
                    .collect();
                if group.zero {
                    ys.push(0.0);
                }
                let (ymin, ymax) = span(&ys);
                let contour = if glyph == "contour" {
                    contour_series(&tables, series.key, x_column, cutoff)
                } else {
                    Vec::new()
                };
                PreparedSeries {
                    key: series.key.to_string(),
                    label: series.label.to_string(),
                    y_label: axis_label(series.label, group.label),
                    glyphs,
                    contour,
                    ymin,
                    ymax,
                }
            })
            .collect();
        Prepared {
            key: String::new(),
            tables,
            categories,
            xmin,
            xmax,
            x_labels,
            glyph: glyph.to_string(),
            x_column: x_column.to_string(),
            raw_x,
            series,
        }
    });

    let mut panels = Vec::new();
    if prepared.series.is_empty() {
        panels.push(note_panel("No finite values for this metric"));
    } else {
        for (index, item) in prepared.series.iter().enumerate() {
            panels.push(assemble_panel(&prepared, index, trend, interlock, group));
            let series = group
                .series
                .iter()
                .find(|series| series.key == item.key)
                .unwrap();
            if hist {
                panels.push(hist_panel(
                    series,
                    &prepared.tables,
                    hist_bins,
                    shared_bins,
                    interlock && group.id == "position_error",
                ));
            }
            if corr {
                let other = &prepared.series[(index + 1) % prepared.series.len()];
                let other = group
                    .series
                    .iter()
                    .find(|series| series.key == other.key)
                    .unwrap();
                panels.push(corr_panel(series, other, &prepared.tables, group.label));
            }
        }
    }

    let side = u32::from(hist) + u32::from(corr);
    let mut weights = vec![8.0];
    weights.extend(std::iter::repeat_n(1.65, side as usize));
    PlotScene {
        title: format!("{} vs {}", group.label, x_label(x_column)),
        panels,
        controls: controls(
            group.id,
            x_id,
            glyph,
            source,
            beam,
            domain,
            trend,
            hist,
            corr,
            interlock,
            &bin_label(&bins),
            hist_bins,
            shared_bins,
            cutoff,
        ),
        table: None,
        samples: Vec::new(),
        columns: 1 + side,
        column_weights: weights,
        row_weights: Vec::new(),
    }
}

fn assemble_panel(
    prepared: &Prepared,
    index: usize,
    trend: Trend,
    interlock: bool,
    group: &YGroup,
) -> Panel {
    let item = &prepared.series[index];
    let glyph = prepared.glyph.as_str();
    let x_column = prepared.x_column.as_str();
    let mut drawn = Vec::new();
    for (session, table) in prepared.tables.iter().enumerate() {
        drawn.extend(item.glyphs[session].clone());
        let Some(y) = table.get(&item.key) else {
            continue;
        };
        if !has_finite(Some(y)) {
            continue;
        }
        if trend != Trend::Off && glyph != "contour" && group.id != "dose_rate" && !prepared.raw_x {
            if let Some(guide) = binned_trend(
                table,
                y,
                &prepared.categories,
                session,
                prepared.xmin,
                prepared.xmax,
                trend,
            ) {
                drawn.push(guide);
            }
        }
        if trend != Trend::Off && group.id == "dose_rate" {
            if let Some(rate) = table
                .get("session_avg_rate")
                .and_then(|v| v.first().copied())
            {
                if rate.is_finite() {
                    drawn.push(hline(
                        prepared.xmin,
                        prepared.xmax,
                        rate,
                        [0.9, 0.75, 0.3, 1.0],
                    ));
                }
            }
        }
        if trend != Trend::Off && glyph == "scatter" {
            if let Some(guide) = scatter_trend(
                table,
                &item.key,
                x_column,
                prepared.xmin,
                prepared.xmax,
                trend,
            ) {
                drawn.push(guide);
            }
        }
    }
    drawn.extend(item.contour.clone());
    if trend != Trend::Off && glyph == "contour" {
        for table in &prepared.tables {
            if let Some(guide) = scatter_trend(
                table,
                &item.key,
                x_column,
                prepared.xmin,
                prepared.xmax,
                trend,
            ) {
                drawn.push(guide);
            }
        }
    }
    if interlock {
        drawn.extend(interlock_guides(
            &prepared.tables,
            &prepared.categories,
            group.id,
            x_column,
            glyph,
            prepared.xmin,
            prepared.xmax,
        ));
    }
    Panel {
        title: item.label.clone(),
        y_label: item.y_label.clone(),
        xmin: prepared.xmin,
        xmax: prepared.xmax,
        ymin: item.ymin,
        ymax: item.ymax,
        series: drawn,
        x_labels: prepared.x_labels.clone(),
        equal: false,
    }
}

fn axis_label(series: &str, group: &str) -> String {
    let Some(unit) = group
        .rfind('(')
        .zip(group.rfind(')'))
        .filter(|(start, end)| end > start)
        .map(|(start, end)| group[start + 1..end].trim())
        .filter(|unit| !unit.is_empty())
    else {
        return series.to_string();
    };
    if series.contains('(') || series.eq_ignore_ascii_case(unit) {
        series.to_string()
    } else {
        format!("{series} ({unit})")
    }
}

fn violin_series(
    table: &BTreeMap<String, Vec<f32>>,
    y: &[f32],
    categories: &[f32],
    session: usize,
    color: [f32; 4],
) -> Vec<Series> {
    let bins = table.get("_bin");
    let groups = group_samples(bins, y, categories);
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut outline_x = Vec::new();
    let mut outline_y = Vec::new();
    for (index, samples) in groups.iter().enumerate() {
        let center = index as f32;
        let shape = kde(samples, VIOLIN_WIDTH * 0.5);
        if shape.len() < 2 {
            if let Some((y, half)) = shape.first() {
                xs.extend([center - half, center + half, center]);
                ys.extend([*y, *y, *y]);
                outline_x.extend([center - half, center + half, f32::NAN]);
                outline_y.extend([*y, *y, f32::NAN]);
            }
            continue;
        }
        for pair in shape.windows(2) {
            let (y0, h0) = pair[0];
            let (y1, h1) = pair[1];
            let (l0, r0) = (center - h0, center + h0);
            let (l1, r1) = (center - h1, center + h1);
            xs.extend([l0, r0, r1, l0, r1, l1]);
            ys.extend([y0, y0, y1, y0, y1, y1]);
        }
        for (y, half) in &shape {
            outline_x.push(center - half);
            outline_y.push(*y);
        }
        // The top sample is a short cap. Both corners have to be on the stroke,
        // or the line cuts across the fill's peak.
        for (y, half) in shape.iter().rev() {
            outline_x.push(center + half);
            outline_y.push(*y);
        }
        outline_x.push(center - shape[0].1);
        outline_y.push(shape[0].0);
        outline_x.push(f32::NAN);
        outline_y.push(f32::NAN);
        let _ = session;
    }
    vec![
        Series::Triangles { xs, ys, color },
        Series::Polyline {
            xs: outline_x,
            ys: outline_y,
            color: [color[0], color[1], color[2], 0.0],
            thickness: 1.0,
        },
    ]
}

fn box_series(
    table: &BTreeMap<String, Vec<f32>>,
    y: &[f32],
    categories: &[f32],
    session: usize,
    color: [f32; 4],
) -> Vec<Series> {
    let bins = table.get("_bin");
    let groups = group_samples(bins, y, categories);
    let mut x = Vec::new();
    let mut bottom = Vec::new();
    let mut w = Vec::new();
    let mut h = Vec::new();
    let mut med_x = Vec::new();
    let mut med_y = Vec::new();
    for (index, samples) in groups.iter().enumerate() {
        let Some(stats) = box_stats(samples) else {
            continue;
        };
        let center = index as f32 + (session as f32 - 0.5) * OFFSET;
        let left = center - BOX_WIDTH * 0.5;
        x.push(left);
        bottom.push(stats.q1);
        w.push(BOX_WIDTH);
        h.push((stats.q3 - stats.q1).max(0.0));
        x.push(center - 0.01);
        bottom.push(stats.whisker_lo);
        w.push(0.02);
        h.push((stats.q1 - stats.whisker_lo).max(0.0));
        x.push(center - 0.01);
        bottom.push(stats.q3);
        w.push(0.02);
        h.push((stats.whisker_hi - stats.q3).max(0.0));
        med_x.extend([left, left + BOX_WIDTH]);
        med_y.extend([stats.median, stats.median]);
        med_x.push(f32::NAN);
        med_y.push(f32::NAN);
    }
    vec![
        Series::Rects {
            x,
            y: bottom,
            w,
            h,
            color,
        },
        Series::Guide {
            xs: med_x,
            ys: med_y,
            color: [0.95, 0.95, 0.95, 1.0],
            thickness: 1.5,
        },
    ]
}

fn mean_series(
    table: &BTreeMap<String, Vec<f32>>,
    y: &[f32],
    categories: &[f32],
    session: usize,
    color: [f32; 4],
) -> Vec<Series> {
    let bins = table.get("_bin");
    let groups = group_samples(bins, y, categories);
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for (index, samples) in groups.iter().enumerate() {
        if samples.is_empty() {
            continue;
        }
        xs.push(index as f32 + (session as f32 - 0.5) * OFFSET);
        ys.push(samples.iter().sum::<f32>() / samples.len() as f32);
    }
    let curve = pchip(&xs, &ys);
    vec![
        Series::Points {
            xs: xs.clone(),
            ys: ys.clone(),
            color,
            radius: 3.5,
        },
        Series::Polyline {
            xs: curve.0,
            ys: curve.1,
            color: [color[0], color[1], color[2], 0.0],
            thickness: 2.0,
        },
    ]
}

fn scatter_series(
    table: &BTreeMap<String, Vec<f32>>,
    y: &[f32],
    x_column: &str,
    color: [f32; 4],
) -> Series {
    let xs_in = table.get(x_column).map(Vec::as_slice).unwrap_or(&[]);
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    // ponytail: scatter past 20k points is strided. Draw every point if a tail matters.
    let stride = (xs_in.len() / 20_000).max(1);
    for (index, (x, y)) in xs_in.iter().zip(y).enumerate() {
        if index % stride == 0 && x.is_finite() && y.is_finite() {
            xs.push(*x);
            ys.push(*y);
        }
    }
    Series::Points {
        xs,
        ys,
        color,
        radius: 2.5,
    }
}

fn contour_series(
    tables: &[BTreeMap<String, Vec<f32>>],
    key: &str,
    x_column: &str,
    cutoff_pct: f32,
) -> Vec<Series> {
    let mut drawn = Vec::new();
    for table in tables {
        let Some(xs) = table.get(x_column) else {
            continue;
        };
        let Some(ys) = table.get(key) else { continue };
        drawn.extend(contour_bands(xs, ys, cutoff_pct));
    }
    drawn
}

/// Nested density fills plus the isolines around each band.
pub(crate) fn contour_bands(xs: &[f32], ys: &[f32], cutoff_pct: f32) -> Vec<Series> {
    let mut pairs: Vec<(f32, f32)> = xs
        .iter()
        .zip(ys)
        .filter(|(x, y)| x.is_finite() && y.is_finite())
        .map(|(x, y)| (*x, *y))
        .collect();
    if pairs.len() > 8000 {
        let step = (pairs.len() / 8000).max(1);
        pairs = pairs.into_iter().step_by(step).collect();
    }
    if pairs.len() < 20 {
        return Vec::new();
    }
    let xs: Vec<f32> = pairs.iter().map(|pair| pair.0).collect();
    let ys: Vec<f32> = pairs.iter().map(|pair| pair.1).collect();
    let (x0, x1) = density_range(&xs);
    let (y0, y1) = density_range(&ys);
    pairs.retain(|(x, y)| *x >= x0 && *x <= x1 && *y >= y0 && *y <= y1);
    if pairs.len() < 20 || x1 <= x0 || y1 <= y0 {
        return Vec::new();
    }
    let bins = 80usize;
    let mut counts = vec![0.0f32; bins * bins];
    let dx = (x1 - x0) / bins as f32;
    let dy = (y1 - y0) / bins as f32;
    for (x, y) in &pairs {
        let ix = (((x - x0) / (x1 - x0)) * bins as f32) as usize;
        let iy = (((y - y0) / (y1 - y0)) * bins as f32) as usize;
        counts[ix.min(bins - 1) + bins * iy.min(bins - 1)] += 1.0;
    }
    let positive: Vec<f32> = counts
        .iter()
        .copied()
        .filter(|value| *value > 0.0)
        .collect();
    let z_max = positive.iter().copied().fold(0.0, f32::max);
    if z_max <= 0.0 {
        return Vec::new();
    }
    let lo = cutoff_pct.clamp(0.0, 90.0).min(97.0);
    let mut levels: Vec<f32> = if lo >= 97.0 {
        vec![97.0]
    } else {
        (0..6)
            .map(|step| lo + (97.0 - lo) * step as f32 / 5.0)
            .collect()
    };
    levels = levels
        .into_iter()
        .map(|level| percentile(&positive, f64::from(level) / 100.0))
        .filter(|level| level.is_finite() && *level > 0.0 && *level < z_max)
        .collect();
    levels.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    levels.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
    if levels.is_empty() {
        return Vec::new();
    }
    let (mesh_x, mesh_y) = contour_mesh(&counts, bins, &levels, x0, y0, dx, dy);
    let mut drawn = Vec::new();
    if mesh_x.len() >= 3 {
        drawn.push(Series::Triangles {
            xs: mesh_x,
            ys: mesh_y,
            color: [0.8, 0.8, 0.8, 0.13],
        });
    }
    let (line_x, line_y) = isolines(&counts, bins, &levels, x0, y0, dx, dy);
    if line_x.len() >= 2 {
        drawn.push(Series::Polyline {
            xs: line_x,
            ys: line_y,
            color: [0.8, 0.8, 0.8, 0.0],
            thickness: 1.0,
        });
    }
    drawn
}

/// The part of each cell on or above a level. Corners and edge crossings match `isolines`.
fn contour_mesh(
    counts: &[f32],
    bins: usize,
    levels: &[f32],
    x0: f32,
    y0: f32,
    dx: f32,
    dy: f32,
) -> (Vec<f32>, Vec<f32>) {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let at = |ix: usize, iy: usize| counts[ix + bins * iy];
    for level in levels {
        for iy in 0..bins.saturating_sub(1) {
            for ix in 0..bins.saturating_sub(1) {
                let c00 = at(ix, iy);
                let c10 = at(ix + 1, iy);
                let c11 = at(ix + 1, iy + 1);
                let c01 = at(ix, iy + 1);
                let case = u8::from(c00 >= *level)
                    | (u8::from(c10 >= *level) << 1)
                    | (u8::from(c11 >= *level) << 2)
                    | (u8::from(c01 >= *level) << 3);
                if case == 0 {
                    continue;
                }
                let x = x0 + (ix as f32 + 0.5) * dx;
                let y = y0 + (iy as f32 + 0.5) * dy;
                let p00 = (x, y);
                let p10 = (x + dx, y);
                let p11 = (x + dx, y + dy);
                let p01 = (x, y + dy);
                let cross =
                    |edge| edge_point(ix, iy, edge, c00, c10, c11, c01, *level, x0, y0, dx, dy);
                let e0 = cross(0);
                let e1 = cross(1);
                let e2 = cross(2);
                let e3 = cross(3);
                match case {
                    1 => fan(&mut xs, &mut ys, &[p00, e0, e3]),
                    2 => fan(&mut xs, &mut ys, &[p10, e1, e0]),
                    3 => fan(&mut xs, &mut ys, &[p00, p10, e1, e3]),
                    4 => fan(&mut xs, &mut ys, &[p11, e2, e1]),
                    5 => {
                        fan(&mut xs, &mut ys, &[p00, e0, e3]);
                        fan(&mut xs, &mut ys, &[p11, e2, e1]);
                    }
                    6 => fan(&mut xs, &mut ys, &[e0, p10, p11, e2]),
                    7 => fan(&mut xs, &mut ys, &[p00, p10, p11, e2, e3]),
                    8 => fan(&mut xs, &mut ys, &[p01, e3, e2]),
                    9 => fan(&mut xs, &mut ys, &[p00, e0, e2, p01]),
                    10 => {
                        fan(&mut xs, &mut ys, &[p10, e1, e0]);
                        fan(&mut xs, &mut ys, &[p01, e3, e2]);
                    }
                    11 => fan(&mut xs, &mut ys, &[p00, p10, e1, e2, p01]),
                    12 => fan(&mut xs, &mut ys, &[e3, e1, p11, p01]),
                    13 => fan(&mut xs, &mut ys, &[p00, e0, e1, p11, p01]),
                    14 => fan(&mut xs, &mut ys, &[e0, p10, p11, p01, e3]),
                    _ => fan(&mut xs, &mut ys, &[p00, p10, p11, p01]),
                }
            }
        }
    }
    (xs, ys)
}

fn fan(xs: &mut Vec<f32>, ys: &mut Vec<f32>, poly: &[(f32, f32)]) {
    if poly.len() < 3 {
        return;
    }
    for index in 1..poly.len() - 1 {
        xs.extend([poly[0].0, poly[index].0, poly[index + 1].0]);
        ys.extend([poly[0].1, poly[index].1, poly[index + 1].1]);
    }
}

fn isolines(
    counts: &[f32],
    bins: usize,
    levels: &[f32],
    x0: f32,
    y0: f32,
    dx: f32,
    dy: f32,
) -> (Vec<f32>, Vec<f32>) {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let at = |ix: usize, iy: usize| counts[ix + bins * iy];
    for level in levels {
        for iy in 0..bins.saturating_sub(1) {
            for ix in 0..bins.saturating_sub(1) {
                let c00 = at(ix, iy);
                let c10 = at(ix + 1, iy);
                let c11 = at(ix + 1, iy + 1);
                let c01 = at(ix, iy + 1);
                let case = u8::from(c00 >= *level)
                    | (u8::from(c10 >= *level) << 1)
                    | (u8::from(c11 >= *level) << 2)
                    | (u8::from(c01 >= *level) << 3);
                let segments: &[(u8, u8)] = match case {
                    1 | 14 => &[(3, 0)],
                    2 | 13 => &[(0, 1)],
                    3 | 12 => &[(3, 1)],
                    4 | 11 => &[(1, 2)],
                    6 | 9 => &[(0, 2)],
                    7 | 8 => &[(3, 2)],
                    5 => &[(3, 0), (1, 2)],
                    10 => &[(0, 1), (2, 3)],
                    _ => &[],
                };
                for (a, b) in segments {
                    let (ax, ay) =
                        edge_point(ix, iy, *a, c00, c10, c11, c01, *level, x0, y0, dx, dy);
                    let (bx, by) =
                        edge_point(ix, iy, *b, c00, c10, c11, c01, *level, x0, y0, dx, dy);
                    xs.push(ax);
                    ys.push(ay);
                    xs.push(bx);
                    ys.push(by);
                    xs.push(f32::NAN);
                    ys.push(f32::NAN);
                }
            }
        }
    }
    (xs, ys)
}

fn edge_point(
    ix: usize,
    iy: usize,
    edge: u8,
    c00: f32,
    c10: f32,
    c11: f32,
    c01: f32,
    level: f32,
    x0: f32,
    y0: f32,
    dx: f32,
    dy: f32,
) -> (f32, f32) {
    let x = x0 + (ix as f32 + 0.5) * dx;
    let y = y0 + (iy as f32 + 0.5) * dy;
    let frac = |a: f32, b: f32| {
        if (b - a).abs() < 1e-12 {
            0.5
        } else {
            ((level - a) / (b - a)).clamp(0.0, 1.0)
        }
    };
    match edge {
        0 => (x + frac(c00, c10) * dx, y),
        1 => (x + dx, y + frac(c10, c11) * dy),
        2 => (x + frac(c01, c11) * dx, y + dy),
        _ => (x, y + frac(c00, c01) * dy),
    }
}

fn density_range(values: &[f32]) -> (f32, f32) {
    let lo = percentile(values, 0.0005);
    let hi = percentile(values, 0.9995);
    if lo.is_finite() && hi.is_finite() && hi > lo {
        return (lo, hi);
    }
    let mut lo = f32::MAX;
    let mut hi = f32::MIN;
    for value in values {
        lo = lo.min(*value);
        hi = hi.max(*value);
    }
    if hi <= lo {
        (lo - 0.5, hi + 0.5)
    } else {
        let mid = (lo + hi) / 2.0;
        let half = (hi - lo) / 2.0;
        (mid - half, mid + half)
    }
}

fn binned_trend(
    table: &BTreeMap<String, Vec<f32>>,
    y: &[f32],
    categories: &[f32],
    session: usize,
    xmin: f32,
    xmax: f32,
    trend: Trend,
) -> Option<Series> {
    let bins = table.get("_bin")?;
    if categories.len() < 2 {
        return None;
    }
    let groups = group_samples(Some(bins), y, categories);
    let mut phys = Vec::new();
    let mut medians = Vec::new();
    for (category, samples) in categories.iter().zip(&groups) {
        if samples.is_empty() {
            continue;
        }
        phys.push(*category);
        medians.push(median(samples));
    }
    let coef = fit_trend(&phys, &medians, trend == Trend::Polynomial)?;
    let knots: Vec<f32> = (0..categories.len())
        .map(|index| index as f32 + (session as f32 - 0.5) * OFFSET)
        .collect();
    curve_across(xmin, xmax, &knots, coef[2] != 0.0, |plot_x| {
        eval_trend(coef, phys_at_plot_x(plot_x, categories, session))
    })
}

fn scatter_trend(
    table: &BTreeMap<String, Vec<f32>>,
    key: &str,
    x_column: &str,
    xmin: f32,
    xmax: f32,
    trend: Trend,
) -> Option<Series> {
    let x = table.get(x_column)?;
    let y = table.get(key)?;
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for (x, y) in x.iter().zip(y) {
        if x.is_finite() && y.is_finite() {
            xs.push(*x);
            ys.push(*y);
        }
    }
    let coef = fit_trend(&xs, &ys, trend == Trend::Polynomial)?;
    curve_across(xmin, xmax, &[], coef[2] != 0.0, |plot_x| {
        eval_trend(coef, plot_x)
    })
}

/// `y = a + b x + c x^2`, centered so a wide energy axis stays stable.
/// A polynomial with too few points falls back to a line.
fn fit_trend(xs: &[f32], ys: &[f32], quadratic: bool) -> Option<[f64; 3]> {
    let pairs: Vec<(f64, f64)> = xs
        .iter()
        .zip(ys)
        .filter(|(x, y)| x.is_finite() && y.is_finite())
        .map(|(x, y)| (f64::from(*x), f64::from(*y)))
        .collect();
    if pairs.len() < 2 {
        return None;
    }
    if !quadratic || pairs.len() < 3 {
        let x32: Vec<f32> = pairs.iter().map(|(x, _)| *x as f32).collect();
        let y32: Vec<f32> = pairs.iter().map(|(_, y)| *y as f32).collect();
        let (slope, intercept) = linear_fit(&x32, &y32)?;
        return Some([f64::from(intercept), f64::from(slope), 0.0]);
    }
    let mean = pairs.iter().map(|(x, _)| *x).sum::<f64>() / pairs.len() as f64;
    let mut n = 0.0;
    let mut s1 = 0.0;
    let mut s2 = 0.0;
    let mut s3 = 0.0;
    let mut s4 = 0.0;
    let mut sy = 0.0;
    let mut szy = 0.0;
    let mut sz2y = 0.0;
    for (x, y) in &pairs {
        let z = x - mean;
        let z2 = z * z;
        n += 1.0;
        s1 += z;
        s2 += z2;
        s3 += z2 * z;
        s4 += z2 * z2;
        sy += y;
        szy += z * y;
        sz2y += z2 * y;
    }
    let solved = solve3([n, s1, s2, sy], [s1, s2, s3, szy], [s2, s3, s4, sz2y]);
    let Some([a, b, c]) = solved else {
        let x32: Vec<f32> = pairs.iter().map(|(x, _)| *x as f32).collect();
        let y32: Vec<f32> = pairs.iter().map(|(_, y)| *y as f32).collect();
        let (slope, intercept) = linear_fit(&x32, &y32)?;
        return Some([f64::from(intercept), f64::from(slope), 0.0]);
    };
    let m = mean;
    Some([a - b * m + c * m * m, b - 2.0 * c * m, c])
}

fn solve3(r0: [f64; 4], r1: [f64; 4], r2: [f64; 4]) -> Option<[f64; 3]> {
    let mut rows = [r0, r1, r2];
    for col in 0..3 {
        let mut pivot = col;
        for row in (col + 1)..3 {
            if rows[row][col].abs() > rows[pivot][col].abs() {
                pivot = row;
            }
        }
        if rows[pivot][col].abs() < 1e-12 {
            return None;
        }
        rows.swap(col, pivot);
        let div = rows[col][col];
        for k in col..4 {
            rows[col][k] /= div;
        }
        for row in 0..3 {
            if row == col {
                continue;
            }
            let factor = rows[row][col];
            for k in col..4 {
                rows[row][k] -= factor * rows[col][k];
            }
        }
    }
    Some([rows[0][3], rows[1][3], rows[2][3]])
}

fn eval_trend(coef: [f64; 3], x: f32) -> f32 {
    let x = f64::from(x);
    (coef[0] + coef[1] * x + coef[2] * x * x) as f32
}

fn phys_at_plot_x(plot_x: f32, categories: &[f32], session: usize) -> f32 {
    let index = plot_x - (session as f32 - 0.5) * OFFSET;
    let n = categories.len();
    if n == 0 {
        return index;
    }
    if n == 1 {
        return categories[0];
    }
    if index <= 0.0 {
        return categories[0] + index * (categories[1] - categories[0]);
    }
    let last = (n - 1) as f32;
    if index >= last {
        return categories[n - 1] + (index - last) * (categories[n - 1] - categories[n - 2]);
    }
    let i = (index.floor() as usize).min(n - 2);
    let t = index - i as f32;
    categories[i] + t * (categories[i + 1] - categories[i])
}

fn curve_across(
    xmin: f32,
    xmax: f32,
    knots: &[f32],
    smooth: bool,
    at: impl Fn(f32) -> f32,
) -> Option<Series> {
    let mut samples = vec![xmin];
    for knot in knots {
        if *knot > xmin && *knot < xmax {
            samples.push(*knot);
        }
    }
    samples.push(xmax);
    if smooth {
        let knots = samples;
        samples = Vec::with_capacity(knots.len() * 4);
        for pair in knots.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            samples.push(a);
            samples.push(a + (b - a) * 0.25);
            samples.push(a + (b - a) * 0.5);
            samples.push(a + (b - a) * 0.75);
        }
        samples.push(xmax);
    }
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for x in samples {
        let y = at(x);
        if x.is_finite() && y.is_finite() {
            xs.push(x);
            ys.push(y);
        }
    }
    if xs.len() < 2 {
        return None;
    }
    // Alpha 0 keeps the previous series color, so the line stays with its session.
    Some(Series::Polyline {
        xs,
        ys,
        color: [0.85, 0.85, 0.85, 0.0],
        thickness: 1.5,
    })
}

fn hist_panel(
    series: &YSeries,
    tables: &[BTreeMap<String, Vec<f32>>],
    bins: usize,
    shared: bool,
    position_guides: bool,
) -> Panel {
    let columns: Vec<&[f32]> = tables
        .iter()
        .filter_map(|table| table.get(series.key).map(Vec::as_slice))
        .filter(|column| column.iter().any(|value| value.is_finite()))
        .collect();
    let shared_edges = shared.then(|| {
        let mut lo = f32::MAX;
        let mut hi = f32::MIN;
        for column in &columns {
            for value in *column {
                if value.is_finite() {
                    lo = lo.min(*value);
                    hi = hi.max(*value);
                }
            }
        }
        uniform_edges(lo, hi, bins)
    });
    let mut drawn = Vec::new();
    let mut peak = 1.0f32;
    let mut lo = f32::MAX;
    let mut hi = f32::MIN;
    for column in columns {
        let (edges, counts) = if let Some(edges) = &shared_edges {
            (edges.clone(), count_bins(edges, column))
        } else {
            histogram(column, bins)
        };
        lo = lo.min(edges.first().copied().unwrap_or(0.0));
        hi = hi.max(edges.last().copied().unwrap_or(1.0));
        let total = counts.iter().sum::<f32>().max(1.0);
        let percent: Vec<f32> = counts.iter().map(|count| count / total * 100.0).collect();
        peak = peak.max(percent.iter().copied().fold(0.0, f32::max));
        let outline = bar_outline(&edges, &percent);
        drawn.push(Series::Bars {
            edges,
            counts: percent,
            color: [0.8, 0.8, 0.8, 0.45],
        });
        drawn.push(outline);
    }
    if position_guides
        && matches!(
            series.key,
            "ic1_x_err" | "ic1_y_err" | "ic2_x_err" | "ic2_y_err"
        )
    {
        for level in [1.0, 2.0, 3.0, -1.0, -2.0, -3.0] {
            drawn.push(Series::Guide {
                xs: vec![level, level],
                ys: vec![0.0, peak],
                color: [0.45, 0.7, 0.4, 0.7],
                thickness: 1.0,
            });
        }
    }
    if !lo.is_finite() || !hi.is_finite() || hi <= lo {
        lo = 0.0;
        hi = 1.0;
    }
    Panel {
        title: String::new(),
        y_label: "Probability (%)".into(),
        xmin: lo,
        xmax: hi,
        ymin: 0.0,
        ymax: peak * 1.1,
        series: drawn,
        x_labels: Vec::new(),
        equal: false,
    }
}

fn corr_panel(
    a: &YSeries,
    b: &YSeries,
    tables: &[BTreeMap<String, Vec<f32>>],
    group_label: &str,
) -> Panel {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for table in tables {
        let (Some(left), Some(right)) = (table.get(a.key), table.get(b.key)) else {
            continue;
        };
        for (left, right) in left.iter().zip(right) {
            if left.is_finite() && right.is_finite() {
                xs.push(*left);
                ys.push(*right);
            }
        }
    }
    let (xmin, xmax) = span(&xs);
    let (ymin, ymax) = span(&ys);
    let mut series = vec![Series::Points {
        xs: xs.clone(),
        ys: ys.clone(),
        color: [0.8, 0.8, 0.8, 0.45],
        radius: 2.0,
    }];
    let lo = xmin.min(ymin);
    let hi = xmax.max(ymax);
    series.push(Series::Guide {
        xs: vec![lo, hi],
        ys: vec![lo, hi],
        color: [0.6, 0.6, 0.6, 0.8],
        thickness: 1.0,
    });
    if let Some((slope, intercept)) = linear_fit(&xs, &ys) {
        series.push(Series::Guide {
            xs: vec![xmin, xmax],
            ys: vec![slope * xmin + intercept, slope * xmax + intercept],
            color: [0.9, 0.75, 0.3, 0.9],
            thickness: 1.5,
        });
    }
    Panel {
        title: String::new(),
        y_label: axis_label(b.label, group_label),
        xmin,
        xmax,
        ymin,
        ymax,
        series,
        x_labels: Vec::new(),
        equal: false,
    }
}

fn interlock_guides(
    tables: &[BTreeMap<String, Vec<f32>>],
    categories: &[f32],
    metric: &str,
    x_column: &str,
    glyph: &str,
    xmin: f32,
    xmax: f32,
) -> Vec<Series> {
    let mut guides = Vec::new();
    if metric == "position_error" || metric == "ic12_pos_diff" {
        for (level, color) in [
            (1.0, [0.35, 0.75, 0.4, 0.85]),
            (2.0, [0.9, 0.6, 0.2, 0.85]),
            (3.0, [0.85, 0.3, 0.25, 0.85]),
        ] {
            guides.push(hline(xmin, xmax, level, color));
            guides.push(hline(xmin, xmax, -level, color));
        }
    }
    if metric == "dose_error" && x_column == "target_mu" {
        guides.extend(dose_gates(categories, glyph, xmin, xmax));
    }
    if (metric == "sigma" || metric == "sigma_error")
        && x_column == "energy"
        && glyph != "scatter"
        && glyph != "contour"
    {
        if let Some(expected) = tables
            .iter()
            .find_map(|table| table.get("expected_sigma").cloned())
        {
            guides.extend(sigma_bands(&expected, categories, metric == "sigma"));
        }
    }
    guides
}

fn dose_gates(categories: &[f32], glyph: &str, xmin: f32, xmax: f32) -> Vec<Series> {
    let samples: Vec<f32> = if glyph == "scatter" || glyph == "contour" {
        let mut values = Vec::new();
        let mut mu = xmin.max(1e-4);
        while mu <= xmax {
            values.push(mu);
            mu += (xmax - xmin).max(1e-3) / 40.0;
        }
        values
    } else {
        categories.to_vec()
    };
    let mut guides = Vec::new();
    for (frac, color) in [
        (0.01, [0.35, 0.75, 0.4, 0.85]),
        (0.02, [0.9, 0.6, 0.2, 0.85]),
        (0.03, [0.85, 0.3, 0.25, 0.85]),
    ] {
        let mut xs = Vec::new();
        let mut hi = Vec::new();
        let mut lo = Vec::new();
        for (index, mu) in samples.iter().enumerate() {
            let x = if glyph == "scatter" || glyph == "contour" {
                *mu
            } else {
                index as f32
            };
            let gate = ((GATE_ABS_MU + frac * f64::from(*mu)) / f64::from(*mu) * 100.0) as f32;
            xs.push(x);
            hi.push(gate);
            lo.push(-gate);
        }
        guides.push(Series::Guide {
            xs: xs.clone(),
            ys: hi,
            color,
            thickness: 1.0,
        });
        guides.push(Series::Guide {
            xs,
            ys: lo,
            color,
            thickness: 1.0,
        });
    }
    guides
}

fn sigma_bands(expected: &[f32], categories: &[f32], absolute: bool) -> Vec<Series> {
    let mut guides = Vec::new();
    for (frac, color) in [(0.2, [0.35, 0.75, 0.4, 0.7]), (0.4, [0.9, 0.6, 0.2, 0.7])] {
        let mut xs = Vec::new();
        let mut hi = Vec::new();
        let mut lo = Vec::new();
        for (index, energy) in categories.iter().enumerate() {
            let Some(mm) = expected.iter().copied().find(|value| same(*value, *energy)) else {
                // expected is stored as parallel (energy, mm) pairs: energy, mm, energy, mm
                continue;
            };
            let _ = mm;
            xs.push(index as f32);
            let band = expected_at(expected, *energy).unwrap_or(0.0) * frac;
            if absolute {
                let center = expected_at(expected, *energy).unwrap_or(0.0);
                hi.push(center + band);
                lo.push((center - band).max(0.0));
            } else {
                hi.push(band);
                lo.push(-band);
            }
        }
        if xs.len() >= 2 {
            guides.push(Series::Guide {
                xs: xs.clone(),
                ys: hi,
                color,
                thickness: 1.0,
            });
            guides.push(Series::Guide {
                xs,
                ys: lo,
                color,
                thickness: 1.0,
            });
        }
    }
    guides
}

fn expected_at(pairs: &[f32], energy: f32) -> Option<f32> {
    pairs
        .chunks(2)
        .find(|pair| pair.len() == 2 && same(pair[0], energy))
        .map(|pair| pair[1])
}

fn hline(xmin: f32, xmax: f32, y: f32, color: [f32; 4]) -> Series {
    Series::Guide {
        xs: vec![xmin, xmax],
        ys: vec![y, y],
        color,
        thickness: 1.0,
    }
}

fn note_panel(title: &str) -> Panel {
    Panel {
        title: title.to_string(),
        y_label: String::new(),
        xmin: 0.0,
        xmax: 1.0,
        ymin: 0.0,
        ymax: 1.0,
        series: Vec::new(),
        x_labels: Vec::new(),
        equal: false,
    }
}

fn x_span(
    tables: &[BTreeMap<String, Vec<f32>>],
    categories: &[f32],
    raw_x: bool,
    x_column: &str,
    _series: usize,
) -> (f32, f32) {
    if raw_x {
        let values: Vec<f32> = tables
            .iter()
            .flat_map(|table| table.get(x_column).into_iter().flatten().copied())
            .filter(|value| value.is_finite())
            .collect();
        let (lo, hi) = span(&values);
        return (
            lo - (hi - lo).max(1.0) * 0.05,
            hi + (hi - lo).max(1.0) * 0.05,
        );
    }
    if categories.is_empty() {
        return (-0.5, 0.5);
    }
    (-0.6, categories.len() as f32 - 0.4)
}

fn x_label(column: &str) -> &'static str {
    match column {
        "target_mu" => "Target MU",
        "spot_time" => "Spot time (ms)",
        "radius" => "Radius (mm)",
        _ => "Energy (MeV)",
    }
}

#[allow(clippy::too_many_arguments)]
fn controls(
    metric: &str,
    x: &str,
    glyph: &str,
    source: &str,
    beam: &str,
    domain: &str,
    trend: Trend,
    hist: bool,
    corr: bool,
    interlock: bool,
    bins: &str,
    hist_bins: usize,
    shared: bool,
    cutoff: f32,
) -> Vec<Control> {
    let on = |value: bool| if value { "On" } else { "Off" };
    let coarse = coarse_of(source);
    let frame = frame_of(source);
    let y_opts = y_choices(coarse);
    let y_labels: Vec<&str> = y_opts.iter().map(|choice| choice.label.as_str()).collect();
    let y_value = y_opts
        .iter()
        .find(|choice| choice.metric == metric && choice.frame == frame)
        .map(|choice| choice.label.as_str())
        .unwrap_or(y_labels.first().copied().unwrap_or(metric));
    let slice_x = matches!(source, "timeslice_iso" | "timeslice_chamber")
        && (timeslice_energy_only(metric)
            || matches!(
                metric,
                "position_error" | "sigma" | "sigma_error" | "ic12_pos_diff"
            ));
    let x_choices: &[(&str, &str)] = if slice_x { &X_CHOICES[..1] } else { X_CHOICES };
    let mut controls = vec![
        labeled("source", "Source", COARSE_CHOICES, coarse),
        control("metric", "Y", &y_labels, y_value),
        labeled("x", "X", x_choices, x),
        labeled("glyph", "Glyph", GLYPH_CHOICES, glyph),
        labeled("beam", "Beam", BEAM_CHOICES, beam),
        labeled("domain", "Domain", DOMAIN_CHOICES, domain),
        control(
            "trend",
            "Trend",
            &["Off", "Linear", "Polynomial"],
            trend_value(trend),
        ),
        control("hist", "Histogram", &["Off", "On"], on(hist)),
        control("corr", "Correlation", &["Off", "On"], on(corr)),
        control(
            "interlock",
            "Interlock Thresholds",
            &["Off", "On"],
            on(interlock),
        ),
        control("bins", "Bins", &["Automatic", "8", "16", "32", "64"], bins),
        control(
            "hist_bins",
            "Histogram Bins",
            &["10", "20", "30", "50"],
            &hist_bins.to_string(),
        ),
        control("shared", "Shared Bins", &["Off", "On"], on(shared)),
        control(
            "cutoff",
            "Contour Cutoff",
            &["0", "5", "10", "20"],
            &cutoff.round().to_string(),
        ),
    ];
    let filters = GROUPS
        .iter()
        .any(|group| group.id == metric && group.filter);
    controls.retain(|item| match item.id.as_str() {
        "cutoff" => glyph == "contour",
        "interlock" => interlock_ok(metric, x, glyph),
        "beam" | "domain" => filters,
        _ => true,
    });
    controls
}

fn interlock_ok(metric: &str, x: &str, glyph: &str) -> bool {
    match metric {
        "position_error" => true,
        "dose_error" => x == "target_mu",
        "sigma" | "sigma_error" => x == "energy" && !matches!(glyph, "scatter" | "contour"),
        _ => false,
    }
}

pub(crate) fn labeled(id: &str, label: &str, pairs: &[(&str, &str)], current: &str) -> Control {
    let value = pairs
        .iter()
        .find(|(key, _)| *key == current)
        .map(|(_, label)| *label)
        .unwrap_or(current);
    control(
        id,
        label,
        &pairs.iter().map(|(_, label)| *label).collect::<Vec<_>>(),
        value,
    )
}

fn control(id: &str, label: &str, options: &[&str], value: &str) -> Control {
    Control {
        id: id.to_string(),
        label: label.to_string(),
        options: options.iter().map(|option| (*option).to_string()).collect(),
        value: value.to_string(),
    }
}

// ponytail: keyed by spot/map length, mtime, and the first 4KB. devices.xml and
// per-layer point-time files stay stale until the spot csv changes. Past 16
// entries the map is cleared.
const SPOT_CACHE_CAP: usize = 16;

#[derive(Hash, PartialEq, Eq)]
struct SpotCacheKey {
    dir: PathBuf,
    chamber: bool,
    spot_len: u64,
    spot_ns: u128,
    spot_head: u64,
    map_len: u64,
    map_ns: u128,
    map_head: u64,
    points: bool,
}

fn spot_cache() -> &'static std::sync::Mutex<HashMap<SpotCacheKey, BTreeMap<String, Vec<f32>>>> {
    static CACHE: std::sync::OnceLock<
        std::sync::Mutex<HashMap<SpotCacheKey, BTreeMap<String, Vec<f32>>>>,
    > = std::sync::OnceLock::new();
    CACHE.get_or_init(|| std::sync::Mutex::new(HashMap::new()))
}

fn file_fingerprint(path: &Path) -> Option<(u64, u128, u64)> {
    let meta = std::fs::metadata(path).ok()?;
    let ns = meta
        .modified()
        .ok()?
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_nanos();
    let mut file = std::fs::File::open(path).ok()?;
    let mut buf = [0u8; 4096];
    let n = std::io::Read::read(&mut file, &mut buf).ok()?;
    Some((meta.len(), ns, fnv64(&buf[..n])))
}

fn fnv64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn load_spot(
    root: &Path,
    session: &str,
    chamber: bool,
    points: bool,
) -> BTreeMap<String, Vec<f32>> {
    let dir = discover::session_directory(root, session);
    let key = match (
        file_fingerprint(&dir.join("spot_data.csv")),
        file_fingerprint(&dir.join("input_map.csv")),
    ) {
        (Some((spot_len, spot_ns, spot_head)), Some((map_len, map_ns, map_head))) => {
            Some(SpotCacheKey {
                dir,
                chamber,
                spot_len,
                spot_ns,
                spot_head,
                map_len,
                map_ns,
                map_head,
                points,
            })
        }
        _ => None,
    };
    if let Some(key) = &key {
        let cache = spot_cache().lock().unwrap_or_else(|err| err.into_inner());
        if let Some(hit) = cache.get(key) {
            return hit.clone();
        }
    }
    let table = build_spot(root, session, chamber, points);
    if let Some(key) = key {
        let mut cache = spot_cache().lock().unwrap_or_else(|err| err.into_inner());
        if cache.len() >= SPOT_CACHE_CAP {
            cache.clear();
        }
        cache.insert(key, table.clone());
    }
    table
}

fn build_spot(
    root: &Path,
    session: &str,
    chamber: bool,
    points: bool,
) -> BTreeMap<String, Vec<f32>> {
    let Some(spot_bytes) = discover::read_session_file(root, session, "spot_data.csv") else {
        return BTreeMap::new();
    };
    let Some(map_bytes) = discover::read_session_file(root, session, "input_map.csv") else {
        return BTreeMap::new();
    };
    let spot = read_sheet(&spot_bytes);
    let map = read_sheet(&map_bytes);
    let energy = column(&map, "energy", &[]);
    let dose = column(
        &spot,
        "ic1_total_dose",
        &["ic1_total_dose_spot", "ic1_dose"],
    );
    let n = energy.len().min(if dose.is_empty() {
        energy.len()
    } else {
        dose.len()
    });
    if n == 0 {
        return BTreeMap::new();
    }
    let target = column(&map, "charge_req", &["target_mu", "MU", "mu"]);
    let plan_x = column(&map, "x_position", &["position_x", "X_POSITION", "x_pos"]);
    let plan_y = column(&map, "y_position", &["position_y", "Y_POSITION", "y_pos"]);
    let mut doses = BTreeMap::new();
    for (concept, key) in [
        ("ic1_total_dose", "ic1_dose"),
        ("ic2_total_dose", "ic2_dose"),
        ("ic3_total_dose", "ic3_dose"),
    ] {
        let values = column(&spot, concept, &[key]);
        if values.len() >= n {
            doses.insert(key, values);
        }
    }
    let (mask_pos, mm_pos) = spot_positions(&spot, chamber, n);
    let sigma = spot_sigma(&spot, chamber, n);
    let mut mask_cols: Vec<&[f32]> = vec![&energy];
    if target.len() >= n {
        mask_cols.push(&target);
    }
    if plan_x.len() >= n {
        mask_cols.push(&plan_x);
    }
    if plan_y.len() >= n {
        mask_cols.push(&plan_y);
    }
    for values in doses.values() {
        mask_cols.push(values);
    }
    for values in &mask_pos {
        mask_cols.push(values);
    }
    let keep = keep_mask(&mask_cols, n);
    if !keep.iter().any(|row| *row) {
        return BTreeMap::new();
    }
    let mut table = BTreeMap::new();
    table.insert("energy".to_string(), take_kept(&energy, &keep));
    if target.len() >= n {
        table.insert("target_mu".to_string(), take_kept(&target, &keep));
    }
    if plan_x.len() >= n && plan_y.len() >= n {
        let x = take_kept(&plan_x, &keep);
        let y = take_kept(&plan_y, &keep);
        table.insert(
            "radius".to_string(),
            x.iter()
                .zip(&y)
                .map(|(x, y)| (x * x + y * y).sqrt())
                .collect(),
        );
        table.insert("plan_x".to_string(), x);
        table.insert("plan_y".to_string(), y);
    }
    for (key, values) in &doses {
        table.insert((*key).to_string(), take_kept(values, &keep));
    }
    if let (Some(target), Some(ic1)) = (
        table.get("target_mu").cloned(),
        table.get("ic1_dose").cloned(),
    ) {
        table.insert(
            "ic1_dose_err_pct".to_string(),
            dose_error_pct(&ic1, &target),
        );
    }
    if let (Some(target), Some(ic2)) = (
        table.get("target_mu").cloned(),
        table.get("ic2_dose").cloned(),
    ) {
        table.insert(
            "ic2_dose_err_pct".to_string(),
            dose_error_pct(&ic2, &target),
        );
    }
    if let (Some(target), Some(ic3)) = (
        table.get("target_mu").cloned(),
        table.get("ic3_dose").cloned(),
    ) {
        table.insert(
            "ic3_dose_err_pct".to_string(),
            dose_error_pct(&ic3, &target),
        );
    }
    if let (Some(ic1), Some(ic2)) = (
        table.get("ic1_dose").cloned(),
        table.get("ic2_dose").cloned(),
    ) {
        table.insert("ic21_ratio".to_string(), dose_ratio_pct(&ic2, &ic1));
        if let Some(ic3) = table.get("ic3_dose").cloned() {
            table.insert("ic31_ratio".to_string(), dose_ratio_pct(&ic3, &ic1));
            table.insert("ic32_ratio".to_string(), dose_ratio_pct(&ic3, &ic2));
        }
    }
    let mm = mm_pos.map(|values| values.map(|column| take_kept(&column, &keep)));
    store_positions(&mut table, mm);
    for (key, values) in sigma {
        table.insert(key.to_string(), take_kept(&values, &keep));
    }
    let dir = discover::session_directory(root, session);
    if let Ok(xml) = std::fs::read_to_string(dir.join("config/map2map/devices.xml")) {
        add_sigma_error(&mut table, &xml);
    }
    let layer = spot.num.get("layer_id").or_else(|| map.num.get("layer_id"));
    let spot_no = spot.num.get("spot_no").filter(|values| values.len() >= n);
    let time_ms = spot_time_ms(&spot, n);
    if let (Some(time), Some(layer)) = (time_ms.as_ref(), layer.filter(|v| v.len() >= n)) {
        let ms = spot_delivery_ms(&take_kept(time, &keep), &take_kept(layer, &keep));
        table.insert("spot_time".to_string(), ms);
        let point = ["point_time", "point_time_ms", "point_time(ms)"]
            .iter()
            .find_map(|name| spot.num.get(*name))
            .filter(|values| values.len() >= n)
            .map(|values| take_kept(values, &keep))
            .or_else(|| {
                if !points {
                    return None;
                }
                let spots = spot_no?;
                let layers = layer;
                Some(lookup_point_time(
                    &dir,
                    &take_kept(spots, &keep),
                    &take_kept(layers, &keep),
                ))
            });
        if let Some(beam) = point.filter(|values| values.iter().any(|value| value.is_finite())) {
            let overhead = beam
                .iter()
                .zip(&table["spot_time"])
                .map(|(beam, total)| {
                    if beam.is_finite() && total.is_finite() {
                        (total - beam).max(0.0)
                    } else {
                        f32::NAN
                    }
                })
                .collect();
            table.insert("beam_on_time".to_string(), beam);
            table.insert("overhead_time".to_string(), overhead);
        }
    }
    if table.contains_key("energy") && table.contains_key("target_mu") {
        if let Some(time) = wall_time(&spot, n) {
            add_dose_rate(&mut table, &take_kept(&time, &keep));
        }
    }
    if let Some(beam) = column_any(&spot, &["beam_on", "rci_in_trigger", "r_beamOk"]) {
        if beam.len() >= n {
            table.insert("beam_on".to_string(), take_kept(&beam, &keep));
        }
    }
    table
}

fn spot_time_ms(spot: &Sheet, n: usize) -> Option<Vec<f64>> {
    if let Some(time) = spot.wide.get("timestamp") {
        if time.len() >= n && time.iter().take(n).any(|value| value.is_finite()) {
            return Some(time[..n].to_vec());
        }
    }
    let seconds = spot.wide.get("time_s")?;
    let nanos = spot.wide.get("time_ns")?;
    if seconds.len() < n || nanos.len() < n {
        return None;
    }
    let time: Vec<f64> = seconds
        .iter()
        .zip(nanos)
        .take(n)
        .map(|(seconds, nanos)| seconds * 1000.0 + nanos / 1e6)
        .collect();
    time.iter().any(|value| value.is_finite()).then_some(time)
}

fn row_ok(value: f32) -> bool {
    value.is_finite() && value != -1.0 && value != -10000.0
}

fn keep_mask(columns: &[&[f32]], n: usize) -> Vec<bool> {
    let mut keep = vec![true; n];
    for column in columns {
        if column.len() < n {
            continue;
        }
        for (slot, value) in keep.iter_mut().zip(*column) {
            if !row_ok(*value) {
                *slot = false;
            }
        }
    }
    keep
}

fn take_kept<T: Copy>(values: &[T], keep: &[bool]) -> Vec<T> {
    values
        .iter()
        .zip(keep)
        .filter(|(_, keep)| **keep)
        .map(|(value, _)| *value)
        .collect()
}

fn store_positions(table: &mut BTreeMap<String, Vec<f32>>, mm: [Option<Vec<f32>>; 4]) {
    let plans = [
        table.get("plan_x").cloned(),
        table.get("plan_y").cloned(),
        table.get("plan_x").cloned(),
        table.get("plan_y").cloned(),
    ];
    let keys = [
        ("ic1_x", "ic1_x_err"),
        ("ic1_y", "ic1_y_err"),
        ("ic2_x", "ic2_x_err"),
        ("ic2_y", "ic2_y_err"),
    ];
    for ((pos_key, err_key), (measured, plan)) in keys.into_iter().zip(mm.into_iter().zip(plans)) {
        let Some(measured) = measured else {
            continue;
        };
        if let Some(plan) = plan {
            if plan.len() == measured.len() {
                table.insert(
                    err_key.to_string(),
                    measured
                        .iter()
                        .zip(&plan)
                        .map(|(measured, plan)| measured - plan)
                        .collect(),
                );
            }
        }
        table.insert(pos_key.to_string(), measured);
    }
    if let (Some(a), Some(b)) = (table.get("ic1_x").cloned(), table.get("ic2_x").cloned()) {
        table.insert(
            "ic12_x_diff".to_string(),
            b.iter().zip(&a).map(|(b, a)| b - a).collect(),
        );
    }
    if let (Some(a), Some(b)) = (table.get("ic1_y").cloned(), table.get("ic2_y").cloned()) {
        table.insert(
            "ic12_y_diff".to_string(),
            b.iter().zip(&a).map(|(b, a)| b - a).collect(),
        );
    }
}

/// Columns that participate in the row mask, plus mm positions when the frame resolves.
fn spot_positions(spot: &Sheet, chamber: bool, n: usize) -> (Vec<Vec<f32>>, [Option<Vec<f32>>; 4]) {
    if chamber {
        if let Some(raw) = four_columns(spot, "spot_position_raw", n) {
            let mm = [
                Some(g3_mm(&raw[0], false)),
                Some(g3_mm(&raw[1], true)),
                Some(g3_mm(&raw[2], true)),
                Some(g3_mm(&raw[3], false)),
            ];
            return (raw.to_vec(), mm);
        }
        if let Some(raw) = four_columns(spot, "spot_raw", n) {
            let ic1_x = raw[0].iter().copied().map(remap_g2_raw).collect::<Vec<_>>();
            let ic1_y = raw[1].iter().copied().map(remap_g2_raw).collect::<Vec<_>>();
            let ic2_x = g2_ic2_mm(&ic1_x, &raw[2]);
            let ic2_y = g2_ic2_mm(&ic1_y, &raw[3]);
            return (
                raw.to_vec(),
                [Some(ic1_x), Some(ic1_y), Some(ic2_x), Some(ic2_y)],
            );
        }
    }
    let mut mask = Vec::new();
    let mut mm = [None, None, None, None];
    for (index, (ic, axis)) in [("ic1", "x"), ("ic1", "y"), ("ic2", "x"), ("ic2", "y")]
        .into_iter()
        .enumerate()
    {
        let Some(values) = column_suffix(spot, ic, axis, &["spot_position", "spot"]) else {
            continue;
        };
        if values.len() < n {
            continue;
        }
        let values = values[..n].to_vec();
        mask.push(values.clone());
        mm[index] = Some(values);
    }
    (mask, mm)
}

fn g3_mm(raw: &[f32], reversed: bool) -> Vec<f32> {
    raw.iter()
        .copied()
        .map(|value| {
            if reversed {
                remap(value, 1.0, 128.0, 128.0, -128.0)
            } else {
                remap(value, 1.0, 128.0, -128.0, 128.0)
            }
        })
        .collect()
}

fn spot_sigma(spot: &Sheet, chamber: bool, n: usize) -> Vec<(&'static str, Vec<f32>)> {
    let order: &[&str] = if chamber {
        &["spot_sigma_raw", "spot_sigma"]
    } else {
        &["spot_sigma", "spot_sigma_raw"]
    };
    let mut out = Vec::new();
    for (ic, axis, key) in [
        ("ic1", "x", "ic1_sig_x"),
        ("ic1", "y", "ic1_sig_y"),
        ("ic2", "x", "ic2_sig_x"),
        ("ic2", "y", "ic2_sig_y"),
    ] {
        let Some(values) = column_suffix(spot, ic, axis, order) else {
            continue;
        };
        if values.len() >= n {
            out.push((key, values[..n].iter().map(|value| value * 2.0).collect()));
        }
    }
    out
}

fn four_columns(spot: &Sheet, suffix: &str, n: usize) -> Option<[Vec<f32>; 4]> {
    let mut out = Vec::new();
    for (ic, axis) in [("ic1", "x"), ("ic1", "y"), ("ic2", "x"), ("ic2", "y")] {
        let values = column_suffix(spot, ic, axis, &[suffix])?;
        if values.len() < n {
            return None;
        }
        out.push(values[..n].to_vec());
    }
    Some([
        out[0].clone(),
        out[1].clone(),
        out[2].clone(),
        out[3].clone(),
    ])
}

fn column_suffix<'a>(
    spot: &'a Sheet,
    ic: &str,
    axis: &str,
    suffixes: &[&str],
) -> Option<&'a [f32]> {
    for suffix in suffixes {
        let with_r = format!("r_{ic}_{axis}_{suffix}");
        let plain = format!("{ic}_{axis}_{suffix}");
        if let Some(values) = spot.num.get(&with_r).or_else(|| spot.num.get(&plain)) {
            return Some(values);
        }
    }
    None
}

fn lookup_point_time(dir: &Path, spot_no: &[f32], layer: &[f32]) -> Vec<f32> {
    let found = layer_point_times(dir);
    spot_no
        .iter()
        .zip(layer)
        .map(|(spot, layer)| {
            found
                .get(&(spot.to_bits(), layer.to_bits()))
                .copied()
                .unwrap_or(f32::NAN)
        })
        .collect()
}

fn layer_point_times(dir: &Path) -> HashMap<(u32, u32), f32> {
    let mut found = HashMap::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return found;
    };
    let mut layers: Vec<PathBuf> = entries
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("layer-"))
        })
        .collect();
    layers.sort();
    for layer_dir in layers {
        let Ok(entries) = std::fs::read_dir(&layer_dir) else {
            continue;
        };
        let mut runs: Vec<PathBuf> = entries
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with("run-"))
            })
            .collect();
        runs.sort();
        for run in runs {
            if let Some(path) = point_time_file(&run) {
                if let Ok(bytes) = std::fs::read(&path) {
                    insert_point_times(&mut found, &read_sheet(&bytes));
                }
                break;
            }
        }
    }
    found
}

fn point_time_file(run: &Path) -> Option<PathBuf> {
    for name in [
        "FX4_spot_data.csv",
        "IX256_1_spot_data.csv",
        "IX256_2_spot_data.csv",
        "RCI_spot_data.csv",
    ] {
        let path = run.join(name);
        if path.is_file() {
            return Some(path);
        }
    }
    let Ok(entries) = std::fs::read_dir(run) else {
        return None;
    };
    let mut others: Vec<PathBuf> = entries
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.ends_with("_spot_data.csv") && name != "TX2_spot_data.csv")
        })
        .collect();
    others.sort();
    others.into_iter().next()
}

fn insert_point_times(found: &mut HashMap<(u32, u32), f32>, sheet: &Sheet) {
    let Some(spots) = sheet.num.get("spot_no") else {
        return;
    };
    let Some(layers) = sheet.num.get("layer_id") else {
        return;
    };
    let Some(point) = ["point_time(ms)", "point_time_ms", "point_time"]
        .iter()
        .find_map(|name| sheet.num.get(*name))
    else {
        return;
    };
    for ((spot, layer), point) in spots.iter().zip(layers).zip(point) {
        if spot.is_finite() && layer.is_finite() && point.is_finite() {
            found.insert((spot.to_bits(), layer.to_bits()), *point);
        }
    }
}

fn add_sigma_error(table: &mut BTreeMap<String, Vec<f32>>, xml: &str) {
    let bands = parse_sigma_bands(xml);
    let Some(energy) = table.get("energy").cloned() else {
        return;
    };
    let mut pairs = Vec::new();
    for (key, device) in [
        ("ic1_sig_x", "IC_1_X"),
        ("ic1_sig_y", "IC_1_Y"),
        ("ic2_sig_x", "IC_2_X"),
        ("ic2_sig_y", "IC_2_Y"),
    ] {
        let Some(measured) = table.get(key).cloned() else {
            continue;
        };
        let mut error = Vec::with_capacity(measured.len());
        for (sample, energy) in measured.iter().zip(&energy) {
            match expected_mm(&bands, device, f64::from(*energy)) {
                Some(expected) if sample.is_finite() => {
                    pairs.push(*energy);
                    pairs.push(expected as f32);
                    error.push(sample - expected as f32);
                }
                _ => error.push(f32::NAN),
            }
        }
        table.insert(format!("{key}_err"), error);
    }
    if !pairs.is_empty() {
        table.insert("expected_sigma".to_string(), pairs);
    }
}

fn add_dose_rate(table: &mut BTreeMap<String, Vec<f32>>, time: &[f64]) {
    let energy = &table["energy"];
    let charge = &table["target_mu"];
    let n = energy.len().min(charge.len()).min(time.len());
    if n == 0 {
        return;
    }
    let mu_total: f64 = charge
        .iter()
        .take(n)
        .filter(|value| value.is_finite())
        .map(|value| f64::from(*value))
        .sum();
    let mut t_min = f64::MAX;
    let mut t_max = f64::MIN;
    let mut t_count = 0usize;
    for stamp in time.iter().take(n) {
        if stamp.is_finite() {
            t_min = t_min.min(*stamp);
            t_max = t_max.max(*stamp);
            t_count += 1;
        }
    }
    let span = t_max - t_min;
    if t_count < 2 || span < 0.05 || mu_total <= 0.0 {
        return;
    }
    let mut groups: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for index in 0..n {
        if energy[index].is_finite() {
            groups
                .entry(energy[index].to_bits())
                .or_default()
                .push(index);
        }
    }
    let mut rows = Vec::new();
    for (bits, indices) in groups {
        let mut mu = 0.0f64;
        let mut lo = f64::MAX;
        let mut hi = f64::MIN;
        let mut count = 0usize;
        for index in indices {
            if charge[index].is_finite() {
                mu += f64::from(charge[index]);
            }
            if time[index].is_finite() {
                lo = lo.min(time[index]);
                hi = hi.max(time[index]);
                count += 1;
            }
        }
        let duration = hi - lo;
        if count >= 2 && duration >= 0.05 && mu > 0.0 {
            rows.push((f32::from_bits(bits), (mu / duration) as f32));
        }
    }
    if rows.is_empty() {
        return;
    }
    rows.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    table.insert(
        "rate_energy".to_string(),
        rows.iter().map(|row| row.0).collect(),
    );
    table.insert(
        "mu_rate".to_string(),
        rows.iter().map(|row| row.1).collect(),
    );
    table.insert(
        "session_avg_rate".to_string(),
        vec![(mu_total / span) as f32],
    );
}

fn wall_time(spot: &Sheet, n: usize) -> Option<Vec<f64>> {
    if let Some(stamp) = spot.wide.get("datetime") {
        if stamp.iter().take(n).any(|value| value.is_finite()) {
            return Some(stamp.iter().take(n).copied().collect());
        }
    }
    if let (Some(seconds), Some(nanos)) = (spot.wide.get("time_s"), spot.wide.get("time_ns")) {
        let n = n.min(seconds.len()).min(nanos.len());
        return Some(
            (0..n)
                .map(|index| seconds[index] + nanos[index] * 1e-9)
                .collect(),
        );
    }
    let stamp = spot.wide.get("timestamp")?;
    Some(stamp.iter().take(n).map(|value| value / 1000.0).collect())
}

fn load_slice_metric(
    root: &Path,
    session: &str,
    _metric: &str,
    chamber: bool,
) -> BTreeMap<String, Vec<f32>> {
    let Some(map_bytes) = discover::read_session_file(root, session, "input_map.csv") else {
        return BTreeMap::new();
    };
    let map = read_sheet(&map_bytes);
    let spot =
        discover::read_session_file(root, session, "spot_data.csv").map(|bytes| read_sheet(&bytes));
    let frames = discover::read_timeslice_frames(&discover::session_directory(root, session));
    let layer_of: Vec<i64> = frames.iter().map(|(index, _)| *index).collect();
    let files: Vec<Vec<u8>> = frames.into_iter().map(|(_, bytes)| bytes).collect();
    let sheets = read_sheets(&files, slice_column);
    let energies = unique_seen(&column(&map, "energy", &[]));
    let plan = plan_xy(&map);
    let (axes, spot_shift) = iso_axes(spot.as_ref(), &plan, &sheets);
    let mut energy = Vec::new();
    let mut beam = Vec::new();
    let mut ic1_x = Vec::new();
    let mut ic1_y = Vec::new();
    let mut ic2_x = Vec::new();
    let mut ic2_y = Vec::new();
    let mut ic1_x_mm = Vec::new();
    let mut ic1_y_mm = Vec::new();
    let mut ic2_x_mm = Vec::new();
    let mut ic2_y_mm = Vec::new();
    let mut sig = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    let mut sig_err = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    for (file_index, sheet) in sheets.iter().enumerate() {
        let tagged = frame_energy(
            &energies,
            layer_of.get(file_index).copied().unwrap_or(-1),
            file_index,
        );
        let Some(trigger) = sheet
            .num
            .get("rci_in_trigger")
            .or_else(|| sheet.num.get("r_beamOk"))
        else {
            continue;
        };
        let n = trigger.len();
        for row in 0..n {
            let on = trigger[row] > 0.5;
            beam.push(if on { 1.0 } else { 0.0 });
            let layer = sheet
                .num
                .get("layer_id")
                .and_then(|values| values.get(row))
                .copied()
                .unwrap_or(f32::NAN);
            energy.push(tagged);
            let spot_ic1 = spot_index(sheet, &["spot_no.1", "spot_no"], row) + spot_shift;
            let spot_ic2 =
                spot_index(sheet, &["spot_no.2", "spot_no.1", "spot_no"], row) + spot_shift;
            let measured = [
                ("ic1", "x", "ic1_x", &axes[0], spot_ic1, 0usize),
                ("ic1", "y", "ic1_y", &axes[1], spot_ic1, 1),
                ("ic2", "x", "ic2_x", &axes[2], spot_ic2, 0),
                ("ic2", "y", "ic2_y", &axes[3], spot_ic2, 1),
            ];
            let mut mm = [f32::NAN; 4];
            for (index, (ic, axis, axis_name, fit, spot_no, plan_axis)) in
                measured.into_iter().enumerate()
            {
                let raw = slice_position(sheet, ic, axis, row);
                let accepted = axis_ok(sheet, axis_name, row);
                mm[index] = if !accepted {
                    f32::NAN
                } else if chamber {
                    if raw.is_finite() && raw >= 0.0 {
                        remap_g3_raw(raw)
                    } else {
                        f32::NAN
                    }
                } else if let Some((slope, intercept)) = *fit {
                    if raw.is_finite() && raw >= 0.0 {
                        slope * raw + intercept
                    } else {
                        f32::NAN
                    }
                } else {
                    f32::NAN
                };
                let planned = plan
                    .get(&(layer.to_bits(), spot_no.to_bits()))
                    .map(|xy| if plan_axis == 0 { xy.0 } else { xy.1 })
                    .unwrap_or(f32::NAN);
                let err = if mm[index].is_finite() && planned.is_finite() {
                    mm[index] - planned
                } else {
                    f32::NAN
                };
                match index {
                    0 => {
                        ic1_x.push(err);
                        ic1_x_mm.push(mm[index]);
                    }
                    1 => {
                        ic1_y.push(err);
                        ic1_y_mm.push(mm[index]);
                    }
                    2 => {
                        ic2_x.push(err);
                        ic2_x_mm.push(mm[index]);
                    }
                    _ => {
                        ic2_y.push(err);
                        ic2_y_mm.push(mm[index]);
                    }
                }
            }
            for (index, (ic, axis)) in [("ic1", "x"), ("ic1", "y"), ("ic2", "x"), ("ic2", "y")]
                .into_iter()
                .enumerate()
            {
                let value = slice_sigma(sheet, ic, axis, row);
                sig[index].push(value);
                let target = cell(sheet, &format!("{ic}_sigma_{axis}_target"), row);
                sig_err[index].push(if value.is_finite() && target.is_finite() && target > 0.0 {
                    value - target
                } else {
                    f32::NAN
                });
            }
        }
    }
    if chamber {
        if ic2_closer_reversed(&ic1_x_mm, &ic2_x_mm) {
            flip_chamber_error(&mut ic2_x, &ic2_x_mm);
            negate_finite(&mut ic2_x_mm);
        }
        if ic2_closer_reversed(&ic1_y_mm, &ic2_y_mm) {
            flip_chamber_error(&mut ic2_y, &ic2_y_mm);
            negate_finite(&mut ic2_y_mm);
        }
    }
    let mut table = BTreeMap::new();
    if energy.is_empty() {
        return table;
    }
    // ponytail: a timeslice past 80k samples is strided. Raise the cap if a tail matters.
    let step = (energy.len() / 80_000).max(1);
    let take = |values: Vec<f32>| values.into_iter().step_by(step).collect::<Vec<_>>();
    table.insert("energy".to_string(), take(energy));
    table.insert("beam_on".to_string(), take(beam));
    table.insert("ic1_x_err".to_string(), take(ic1_x));
    table.insert("ic1_y_err".to_string(), take(ic1_y));
    table.insert("ic2_x_err".to_string(), take(ic2_x));
    table.insert("ic2_y_err".to_string(), take(ic2_y));
    for (key, values) in [
        ("ic1_sig_x", sig[0].clone()),
        ("ic1_sig_y", sig[1].clone()),
        ("ic2_sig_x", sig[2].clone()),
        ("ic2_sig_y", sig[3].clone()),
        ("ic1_sig_x_err", sig_err[0].clone()),
        ("ic1_sig_y_err", sig_err[1].clone()),
        ("ic2_sig_x_err", sig_err[2].clone()),
        ("ic2_sig_y_err", sig_err[3].clone()),
    ] {
        table.insert(key.to_string(), take(values));
    }
    let gap = |left: &[f32], right: &[f32]| {
        left.iter()
            .zip(right)
            .map(|(ic2, ic1)| ic2 - ic1)
            .collect::<Vec<_>>()
    };
    table.insert("ic1_x".to_string(), take(ic1_x_mm.clone()));
    table.insert("ic1_y".to_string(), take(ic1_y_mm.clone()));
    table.insert("ic2_x".to_string(), take(ic2_x_mm.clone()));
    table.insert("ic2_y".to_string(), take(ic2_y_mm.clone()));
    table.insert("ic12_x_diff".to_string(), take(gap(&ic2_x_mm, &ic1_x_mm)));
    table.insert("ic12_y_diff".to_string(), take(gap(&ic2_y_mm, &ic1_y_mm)));
    table
}

fn cell(sheet: &Sheet, name: &str, row: usize) -> f32 {
    sheet
        .num
        .get(name)
        .and_then(|values| values.get(row))
        .copied()
        .unwrap_or(f32::NAN)
}

/// Prefer `r_{ic}_{axis}_position`, then the device-units `r_{ic}_{axis}_spot_position`.
fn slice_position(sheet: &Sheet, ic: &str, axis: &str, row: usize) -> f32 {
    for name in [
        format!("r_{ic}_{axis}_position"),
        format!("r_{ic}_{axis}_spot_position"),
    ] {
        if sheet.num.contains_key(&name) {
            return cell(sheet, &name, row);
        }
    }
    f32::NAN
}

fn slice_sigma(sheet: &Sheet, ic: &str, axis: &str, row: usize) -> f32 {
    let value = cell(sheet, &format!("r_{ic}_{axis}_sigma"), row);
    if !value.is_finite() || value <= 0.0 || value > 20.0 {
        return f32::NAN;
    }
    if !axis_ok(sheet, &format!("{ic}_{axis}"), row) {
        return f32::NAN;
    }
    value
}

fn spot_index(sheet: &Sheet, names: &[&str], row: usize) -> f32 {
    for name in names {
        if sheet.num.contains_key(*name) {
            return cell(sheet, name, row);
        }
    }
    f32::NAN
}

/// Fit gate, then confidence and error-code. A missing column is ignored.
/// The first present fit column wins: `{axis}_fit_ok`, then
/// `r_{axis}_spot_position_ok`, then `r_{axis}_position_ok`.
fn axis_ok(sheet: &Sheet, axis: &str, row: usize) -> bool {
    let gates = [
        format!("{axis}_fit_ok"),
        format!("r_{axis}_spot_position_ok"),
        format!("r_{axis}_position_ok"),
    ];
    if let Some(name) = gates
        .iter()
        .find(|name| sheet.num.contains_key(name.as_str()))
    {
        let value = cell(sheet, name, row);
        if !value.is_finite() || value == 0.0 {
            return false;
        }
    }
    let confidence = format!("r_{axis}_confidence");
    if sheet.num.contains_key(&confidence) {
        let value = cell(sheet, &confidence, row);
        if !value.is_finite() || value < 80.0 {
            return false;
        }
    }
    let code = format!("r_{axis}_spot_error_code");
    if sheet.num.contains_key(&code) {
        let value = cell(sheet, &code, row);
        if !value.is_finite() || value != 0.0 {
            return false;
        }
    }
    true
}

fn ic2_closer_reversed(ic1_mm: &[f32], ic2_mm: &[f32]) -> bool {
    let mut forward = Vec::new();
    let mut reversed = Vec::new();
    for (ic1, ic2) in ic1_mm.iter().zip(ic2_mm) {
        if ic1.is_finite() && ic2.is_finite() {
            forward.push((ic1 - ic2).abs());
            reversed.push((ic1 + ic2).abs());
        }
    }
    if forward.is_empty() {
        return false;
    }
    median(&reversed) < median(&forward)
}

fn negate_finite(values: &mut [f32]) {
    for value in values {
        if value.is_finite() {
            *value = -*value;
        }
    }
}

fn flip_chamber_error(err: &mut [f32], forward_mm: &[f32]) {
    for (err, mm) in err.iter_mut().zip(forward_mm) {
        if err.is_finite() && mm.is_finite() {
            let planned = *mm - *err;
            *err = -*mm - planned;
        }
    }
}

fn plan_xy(map: &Sheet) -> HashMap<(u32, u32), (f32, f32)> {
    let mut out = HashMap::new();
    let layer = column(map, "layer_id", &["layer_id"]);
    let spot = column(map, "spot_no", &["spot_no"]);
    let x = column(map, "x_position", &["position_x", "X_POSITION", "x_pos"]);
    let y = column(map, "y_position", &["position_y", "Y_POSITION", "y_pos"]);
    for (((layer, spot), x), y) in layer.iter().zip(&spot).zip(&x).zip(&y) {
        out.insert((layer.to_bits(), spot.to_bits()), (*x, *y));
    }
    out
}

fn iso_axes(
    spot: Option<&Sheet>,
    plan: &HashMap<(u32, u32), (f32, f32)>,
    sheets: &[Sheet],
) -> ([Option<(f32, f32)>; 4], f32) {
    if let Some(fitted) = plan_target_axes(plan, sheets) {
        return fitted;
    }
    (
        spot.map(strip_axes).unwrap_or([None, None, None, None]),
        0.0,
    )
}

fn plan_spans(plan: &HashMap<(u32, u32), (f32, f32)>) -> bool {
    let mut x_lo = f32::MAX;
    let mut x_hi = f32::MIN;
    let mut y_lo = f32::MAX;
    let mut y_hi = f32::MIN;
    for (x, y) in plan.values() {
        if x.is_finite() {
            x_lo = x_lo.min(*x);
            x_hi = x_hi.max(*x);
        }
        if y.is_finite() {
            y_lo = y_lo.min(*y);
            y_hi = y_hi.max(*y);
        }
    }
    x_hi - x_lo >= 0.1 || y_hi - y_lo >= 0.1
}

/// Device-target vs plan affine when the map has position span.
/// ponytail: the first finite target per spot stands in for the per-spot median.
fn plan_target_axes(
    plan: &HashMap<(u32, u32), (f32, f32)>,
    sheets: &[Sheet],
) -> Option<([Option<(f32, f32)>; 4], f32)> {
    if !plan_spans(plan) {
        return None;
    }
    let mut targets: HashMap<(u32, u32), [f32; 4]> = HashMap::new();
    for sheet in sheets {
        let names = [
            "ic1_position_x_target",
            "ic1_position_y_target",
            "ic2_position_x_target",
            "ic2_position_y_target",
        ];
        if names.iter().any(|name| !sheet.num.contains_key(*name)) {
            continue;
        }
        let n = sheet.num.get("layer_id").map(Vec::len).unwrap_or(0);
        for row in 0..n {
            let layer = cell(sheet, "layer_id", row);
            let spot = spot_index(sheet, &["spot_no.1", "spot_no"], row);
            if !layer.is_finite() || !spot.is_finite() {
                continue;
            }
            let sample = [
                cell(sheet, names[0], row),
                cell(sheet, names[1], row),
                cell(sheet, names[2], row),
                cell(sheet, names[3], row),
            ];
            if sample.iter().any(|value| !value.is_finite()) {
                continue;
            }
            targets
                .entry((layer.to_bits(), spot.to_bits()))
                .or_insert(sample);
        }
    }
    if targets.len() < 10 {
        return None;
    }
    let mut best: Option<(f32, i32, i32, [(f32, f32); 4])> = None;
    for shift in -2i32..=2 {
        let mut strips = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        let mut isos = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for ((layer, spot), sample) in &targets {
            let shifted = f32::from_bits(*spot) + shift as f32;
            let Some(planned) = plan.get(&(*layer, shifted.to_bits())) else {
                continue;
            };
            for index in 0..4 {
                strips[index].push(sample[index]);
                isos[index].push(if index % 2 == 0 { planned.0 } else { planned.1 });
            }
        }
        let matches = strips[0].len();
        if matches < 10 {
            continue;
        }
        let mut axes = [(0.0f32, 0.0f32); 4];
        let mut ok = true;
        for index in 0..4 {
            match fit_strip_iso(&strips[index], &isos[index]) {
                Some(axis) => axes[index] = axis,
                None => ok = false,
            }
        }
        if !ok {
            continue;
        }
        let mut square = 0.0f32;
        let mut count = 0.0f32;
        for ((layer, spot), sample) in &targets {
            let Some(planned) = plan.get(&(*layer, *spot)) else {
                continue;
            };
            let actual = [planned.0, planned.1, planned.0, planned.1];
            for index in 0..4 {
                let pred = axes[index].0 * sample[index] + axes[index].1;
                let err = actual[index] - pred;
                square += err * err;
                count += 1.0;
            }
        }
        if count == 0.0 {
            continue;
        }
        let verify = (square / count).sqrt();
        let key = (verify, -(matches as i32), shift.abs());
        if best
            .as_ref()
            .is_none_or(|have| key < (have.0, have.1, have.2))
        {
            best = Some((verify, -(matches as i32), shift.abs(), axes));
        }
    }
    let (_, _, _, axes) = best?;
    // The lookup shift is the one with the most plan matches, same as Python.
    let mut chosen = 0i32;
    let mut most = 0usize;
    for shift in -2i32..=2 {
        let matches = targets
            .keys()
            .filter(|(layer, spot)| {
                let shifted = f32::from_bits(*spot) + shift as f32;
                plan.contains_key(&(*layer, shifted.to_bits()))
            })
            .count();
        if matches > most {
            most = matches;
            chosen = shift;
        }
    }
    if most < 10 {
        return None;
    }
    Some((axes.map(Some), chosen as f32))
}

fn strip_axes(spot: &Sheet) -> [Option<(f32, f32)>; 4] {
    let mut axes = [None, None, None, None];
    for (index, (ic, axis)) in [("ic1", "x"), ("ic1", "y"), ("ic2", "x"), ("ic2", "y")]
        .into_iter()
        .enumerate()
    {
        let Some(strip) = column_suffix(spot, ic, axis, &["spot_position_raw"]) else {
            continue;
        };
        let Some(iso) = column_suffix(spot, ic, axis, &["spot_position"]) else {
            continue;
        };
        axes[index] = fit_strip_iso(strip, iso);
    }
    axes
}

fn fit_strip_iso(strip: &[f32], iso: &[f32]) -> Option<(f32, f32)> {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for (strip, iso) in strip.iter().zip(iso) {
        if strip.is_finite() && iso.is_finite() && *strip > 1.0 && *strip < 128.0 {
            xs.push(*strip);
            ys.push(*iso);
        }
    }
    if xs.len() < 10 {
        return None;
    }
    let (slope, intercept) = linear_fit(&xs, &ys)?;
    if slope.abs() < 0.01 {
        return None;
    }
    let span =
        ys.iter().copied().fold(f32::MIN, f32::max) - ys.iter().copied().fold(f32::MAX, f32::min);
    if span < 0.1 {
        return None;
    }
    let mut resid = 0.0f32;
    for (x, y) in xs.iter().zip(&ys) {
        let err = y - (slope * x + intercept);
        resid += err * err;
    }
    if (resid / xs.len() as f32).sqrt() > 0.05 {
        return None;
    }
    Some((slope, intercept))
}

fn timeslice_inputs(root: &Path, session: &str) -> Option<(Vec<Vec<u8>>, Vec<f32>, Vec<i64>)> {
    let map_bytes = discover::read_session_file(root, session, "input_map.csv")?;
    let map = read_sheet(&map_bytes);
    let energies = unique_seen(&column(&map, "energy", &[]));
    let frames = discover::read_timeslice_frames(&discover::session_directory(root, session));
    if frames.is_empty() {
        return None;
    }
    let layers = frames.iter().map(|(index, _)| *index).collect();
    let files = frames.into_iter().map(|(_, bytes)| bytes).collect();
    Some((files, energies, layers))
}

fn load_timeslice(root: &Path, session: &str, metric: &str) -> BTreeMap<String, Vec<f32>> {
    let Some((files, energies, layer_of)) = timeslice_inputs(root, session) else {
        return BTreeMap::new();
    };
    if metric == "current_ratio" {
        return current_ratio_table(&files, &energies, &layer_of);
    }
    if matches!(
        metric,
        "fit_confidence" | "peak_amplitude" | "amplifier_error" | "probe_field"
    ) {
        return signal_table(&files, &energies, &layer_of, metric);
    }
    current_table(&files, &energies, &layer_of)
}

fn timeslice_energy_only(metric: &str) -> bool {
    matches!(
        metric,
        "fit_confidence" | "peak_amplitude" | "amplifier_error" | "probe_field"
    )
}

/// Folder `layer-N` picks the Nth energy in input-map order. A path with no
/// layer folder uses the file's order instead.
fn frame_energy(energies: &[f32], layer_idx: i64, file_index: usize) -> f32 {
    let index = if layer_idx >= 0 {
        layer_idx as usize
    } else {
        file_index
    };
    energies.get(index).copied().unwrap_or(f32::NAN)
}

fn signal_table(
    files: &[Vec<u8>],
    energies: &[f32],
    layers: &[i64],
    metric: &str,
) -> BTreeMap<String, Vec<f32>> {
    signal_from(&read_sheets(files, signal_column), energies, layers, metric)
}

fn signal_from(
    sheets: &[Sheet],
    energies: &[f32],
    layers: &[i64],
    metric: &str,
) -> BTreeMap<String, Vec<f32>> {
    let spec: &[(&str, &str)] = match metric {
        "fit_confidence" => &[
            ("ic1_x_confidence", "r_ic1_x_confidence"),
            ("ic1_y_confidence", "r_ic1_y_confidence"),
            ("ic2_x_confidence", "r_ic2_x_confidence"),
            ("ic2_y_confidence", "r_ic2_y_confidence"),
        ],
        "peak_amplitude" => &[
            ("ic1_x_peak", "ic1_x_peak_amplitude"),
            ("ic1_y_peak", "ic1_y_peak_amplitude"),
            ("ic2_x_peak", "ic2_x_peak_amplitude"),
            ("ic2_y_peak", "ic2_y_peak_amplitude"),
        ],
        "probe_field" => &[("field_x", "mag_field_x"), ("field_y", "mag_field_y")],
        _ => &[],
    };
    let mut energy = Vec::new();
    let mut beam = Vec::new();
    if metric == "amplifier_error" {
        let mut stored = vec![Vec::new(); 6];
        for (index, sheet) in sheets.iter().enumerate() {
            let tag = frame_energy(energies, layers.get(index).copied().unwrap_or(-1), index);
            let columns = [
                column_diff(sheet, "amplifier_cmd_x", "amplifier_readback_x"),
                column_diff(sheet, "amplifier_cmd_y", "amplifier_readback_y"),
                resolved_column(sheet, "amplifier_cmd_x"),
                resolved_column(sheet, "amplifier_readback_x"),
                resolved_column(sheet, "amplifier_cmd_y"),
                resolved_column(sheet, "amplifier_readback_y"),
            ];
            append_samples(sheet, tag, &columns, &mut energy, &mut beam, &mut stored);
        }
        return signal_rows(
            energy,
            beam,
            &[
                ("amp_x", &stored[0]),
                ("amp_y", &stored[1]),
                ("amp_cmd_x", &stored[2]),
                ("amp_read_x", &stored[3]),
                ("amp_cmd_y", &stored[4]),
                ("amp_read_y", &stored[5]),
            ],
        );
    }
    let mut stored = vec![Vec::new(); spec.len()];
    for (index, sheet) in sheets.iter().enumerate() {
        let tag = frame_energy(energies, layers.get(index).copied().unwrap_or(-1), index);
        let columns: Vec<Vec<f32>> = spec
            .iter()
            .map(|(_, concept)| resolved_column(sheet, concept))
            .collect();
        append_samples(sheet, tag, &columns, &mut energy, &mut beam, &mut stored);
    }
    let pairs: Vec<(&str, &Vec<f32>)> = spec
        .iter()
        .zip(&stored)
        .map(|((key, _), values)| (*key, values))
        .collect();
    signal_rows(energy, beam, &pairs)
}

fn resolved_column(sheet: &Sheet, concept: &str) -> Vec<f32> {
    let keys: Vec<String> = sheet.num.keys().cloned().collect();
    let Some(name) = resolve_concept_column(&keys, concept) else {
        return Vec::new();
    };
    sheet.num.get(name).cloned().unwrap_or_default()
}

/// Readback minus command. An axis is absent when either column is missing.
fn column_diff(sheet: &Sheet, command: &str, readback: &str) -> Vec<f32> {
    let command = resolved_column(sheet, command);
    let readback = resolved_column(sheet, readback);
    if command.is_empty() || readback.is_empty() {
        return Vec::new();
    }
    command
        .into_iter()
        .zip(readback)
        .map(|(command, readback)| {
            if command.is_finite() && readback.is_finite() {
                readback - command
            } else {
                f32::NAN
            }
        })
        .collect()
}

fn append_samples(
    sheet: &Sheet,
    tag: f32,
    columns: &[Vec<f32>],
    energy: &mut Vec<f32>,
    beam_out: &mut Vec<f32>,
    series_out: &mut [Vec<f32>],
) {
    let n = columns.iter().map(Vec::len).max().unwrap_or(0);
    if n == 0 {
        return;
    }
    let gate = column_any(sheet, &["rci_in_trigger", "r_beamOk", "beam_on"]);
    // ponytail: one sample per stride past 80k rows. The plot is a distribution,
    // so a uniform subsample keeps the shape; raise the cap if a tail matters.
    let stride = (n / 80_000).max(1);
    for sample in (0..n).step_by(stride) {
        let any = columns
            .iter()
            .any(|column| column.get(sample).copied().unwrap_or(f32::NAN).is_finite());
        if !any {
            continue;
        }
        energy.push(tag);
        for (out, column) in series_out.iter_mut().zip(columns) {
            out.push(column.get(sample).copied().unwrap_or(f32::NAN));
        }
        beam_out.push(
            gate.as_ref()
                .and_then(|values| values.get(sample).copied())
                .unwrap_or(1.0),
        );
    }
}

fn signal_rows(
    energy: Vec<f32>,
    beam: Vec<f32>,
    pairs: &[(&str, &Vec<f32>)],
) -> BTreeMap<String, Vec<f32>> {
    let mut table = BTreeMap::new();
    if energy.is_empty() {
        return table;
    }
    table.insert("energy".to_string(), energy);
    table.insert("beam_on".to_string(), beam);
    for (key, values) in pairs {
        if values.iter().any(|value| value.is_finite()) {
            table.insert((*key).to_string(), (*values).clone());
        }
    }
    table
}

fn current_table(
    files: &[Vec<u8>],
    energies: &[f32],
    layers: &[i64],
) -> BTreeMap<String, Vec<f32>> {
    current_from(&read_sheets(files, current_column), energies, layers)
}

fn current_from(sheets: &[Sheet], energies: &[f32], layers: &[i64]) -> BTreeMap<String, Vec<f32>> {
    let mut out_energy = Vec::new();
    let mut ic1 = Vec::new();
    let mut ic2 = Vec::new();
    let mut ic3 = Vec::new();
    let mut beam = Vec::new();
    let mut any_ic3 = false;
    for (index, sheet) in sheets.iter().enumerate() {
        let tag = frame_energy(energies, layers.get(index).copied().unwrap_or(-1), index);
        let a = scaled_current(sheet, "ic1");
        let b = scaled_current(sheet, "ic2");
        let c = ic3_current(sheet);
        let gate = column_any(sheet, &["rci_in_trigger", "r_beamOk", "beam_on"]);
        let n = a.len().max(b.len()).max(c.len());
        // ponytail: a timeslice past 80k samples is strided. Raise the cap if a
        // chamber view needs every sample.
        let stride = (n / 80_000).max(1);
        for sample in (0..n).step_by(stride) {
            out_energy.push(tag);
            ic1.push(*a.get(sample).unwrap_or(&f32::NAN));
            ic2.push(*b.get(sample).unwrap_or(&f32::NAN));
            let part = *c.get(sample).unwrap_or(&f32::NAN);
            if part.is_finite() {
                any_ic3 = true;
            }
            ic3.push(part);
            beam.push(
                gate.as_ref()
                    .and_then(|v| v.get(sample).copied())
                    .unwrap_or(1.0),
            );
        }
    }
    let mut table = BTreeMap::new();
    if !out_energy.is_empty() {
        table.insert("energy".to_string(), out_energy);
        table.insert("ic1_current".to_string(), ic1);
        table.insert("ic2_current".to_string(), ic2);
        if any_ic3 {
            table.insert("ic3_current".to_string(), ic3);
        }
        table.insert("beam_on".to_string(), beam);
    }
    table
}

fn current_ratio_table(
    files: &[Vec<u8>],
    energies: &[f32],
    layers: &[i64],
) -> BTreeMap<String, Vec<f32>> {
    let mut out_energy = Vec::new();
    let mut ic21 = Vec::new();
    let mut ic31 = Vec::new();
    let mut ic32 = Vec::new();
    let mut any_ic3 = false;
    let sheets = read_sheets(files, current_column);
    for (index, sheet) in sheets.iter().enumerate() {
        let tag = frame_energy(energies, layers.get(index).copied().unwrap_or(-1), index);
        let a = scaled_current(sheet, "ic1");
        let b = scaled_current(sheet, "ic2");
        let c = ic3_current(sheet);
        let gate = column_any(sheet, &["rci_in_trigger", "r_beamOk", "beam_on"]);
        let plateau = |values: &[f32]| plateau_mean(values, gate.as_deref());
        let left = plateau(&a);
        let right = plateau(&b);
        let third = plateau(&c);
        out_energy.push(tag);
        ic21.push(sym_pct(right, left));
        if third.is_finite() {
            any_ic3 = true;
        }
        ic31.push(sym_pct(third, left));
        ic32.push(sym_pct(third, right));
    }
    let mut table = BTreeMap::new();
    if !out_energy.is_empty() {
        let rows = out_energy.len();
        table.insert("energy".to_string(), out_energy);
        table.insert("beam_on".to_string(), vec![1.0; rows]);
        table.insert("ic21_ratio".to_string(), ic21);
        if any_ic3 {
            table.insert("ic31_ratio".to_string(), ic31);
            table.insert("ic32_ratio".to_string(), ic32);
        }
    }
    table
}

fn plateau_mean(values: &[f32], gate: Option<&[f32]>) -> f32 {
    let on: Vec<f32> = values
        .iter()
        .enumerate()
        .filter_map(|(index, value)| {
            let passed = gate
                .map(|gate| gate.get(index).copied().unwrap_or(0.0) > 0.5)
                .unwrap_or(true);
            (passed && value.is_finite()).then_some(*value)
        })
        .collect();
    if on.len() < 10 {
        return f32::NAN;
    }
    let mut sorted = on.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p95 = percentile(&sorted, 0.95);
    let floor = 5.0f32.max(0.75 * p95);
    let kept: Vec<f32> = on.into_iter().filter(|value| *value >= floor).collect();
    if kept.len() < 3 {
        f32::NAN
    } else {
        kept.iter().sum::<f32>() / kept.len() as f32
    }
}

fn sym_pct(a: f32, b: f32) -> f32 {
    if !a.is_finite() || !b.is_finite() || (a + b).abs() < 1e-12 {
        f32::NAN
    } else {
        (a - b) / ((a + b) / 2.0) * 100.0
    }
}

fn scaled_current(sheet: &Sheet, ic: &str) -> Vec<f32> {
    let fallback = [
        format!("r_{ic}_current_dose"),
        format!("{ic}_current_dose"),
        format!("r_{ic}_current"),
        format!("{ic}_current"),
    ];
    let keys: Vec<String> = sheet.num.keys().cloned().collect();
    let header = resolve_concept_column(&keys, &format!("{ic}_current"))
        .map(str::to_owned)
        .or_else(|| {
            fallback
                .iter()
                .find(|name| sheet.num.contains_key(name.as_str()))
                .cloned()
        });
    let Some(header) = header else {
        return Vec::new();
    };
    let Some(values) = sheet.num.get(&header) else {
        return Vec::new();
    };
    let factor = column_scale_factor(&header).unwrap_or(1.0) as f32;
    scale_column(values, factor)
}

fn ic3_current(sheet: &Sheet) -> Vec<f32> {
    let keys: Vec<String> = sheet.num.keys().cloned().collect();
    let mut sum = Vec::new();
    for (concept, part) in [
        ("ic3_current_a", "a"),
        ("ic3_current_b", "b"),
        ("ic3_current_c", "c"),
        ("ic3_current_d", "d"),
    ] {
        let values = if let Some(header) = resolve_concept_column(&keys, concept) {
            let factor = column_scale_factor(header).unwrap_or(1.0) as f32;
            sheet
                .num
                .get(header)
                .map(|values| scale_column(values, factor))
                .unwrap_or_default()
        } else {
            scaled_current(sheet, &format!("ic3_{part}"))
        };
        if values.is_empty() {
            continue;
        }
        if sum.is_empty() {
            sum = values;
        } else {
            for (slot, value) in sum.iter_mut().zip(values) {
                if value.is_finite() {
                    *slot = if slot.is_finite() {
                        *slot + value
                    } else {
                        value
                    };
                }
            }
        }
    }
    if sum.is_empty() {
        scaled_current(sheet, "ic3")
    } else {
        sum
    }
}

fn column(sheet: &Sheet, concept: &str, extra: &[&str]) -> Vec<f32> {
    let names: Vec<String> = sheet.num.keys().cloned().collect();
    if let Some(found) = resolve_concept_column(&names, concept) {
        if let Some(values) = sheet.num.get(found) {
            return values.clone();
        }
    }
    extra
        .iter()
        .find_map(|name| sheet.num.get(*name).cloned())
        .unwrap_or_default()
}

fn column_any(sheet: &Sheet, names: &[&str]) -> Option<Vec<f32>> {
    names.iter().find_map(|name| sheet.num.get(*name).cloned())
}

pub(crate) fn apply_filter(
    table: &mut BTreeMap<String, Vec<f32>>,
    keys: &[&str],
    domain: &str,
    beam: &str,
) {
    let n = table.values().map(Vec::len).max().unwrap_or(0);
    if n == 0 {
        return;
    }
    let mut keep = vec![true; n];
    if let Some(gate) = table.get("beam_on") {
        for (slot, value) in keep.iter_mut().zip(gate) {
            let on = !value.is_finite() || *value > 0.5;
            *slot = match beam {
                "beam_off" => !on,
                "beam_both" => true,
                _ => on,
            };
        }
    }
    if domain != "all" {
        let columns: Vec<Vec<f32>> = keys
            .iter()
            .filter_map(|key| table.get(*key).cloned())
            .collect();
        if !columns.is_empty() {
            let severity: Vec<f32> = (0..n)
                .map(|row| {
                    columns
                        .iter()
                        .filter_map(|column| column.get(row).copied())
                        .filter(|v| v.is_finite())
                        .map(f32::abs)
                        .fold(None, |acc: Option<f32>, v| {
                            Some(acc.map(|a| a.max(v)).unwrap_or(v))
                        })
                        .unwrap_or(f32::NAN)
                })
                .collect();
            let valid: Vec<f32> = severity.iter().copied().filter(|v| v.is_finite()).collect();
            if domain == "mad_outliers" {
                let z: Vec<Vec<f32>> = columns.iter().map(|column| modified_z(column)).collect();
                for row in 0..n {
                    let outlier = z
                        .iter()
                        .any(|axis| axis.get(row).copied().unwrap_or(0.0).abs() > MOD_Z);
                    keep[row] &= outlier;
                }
            } else if !valid.is_empty() {
                let cutoff = percentile(&valid, 0.95);
                for (row, value) in severity.iter().enumerate() {
                    let pass = value.is_finite()
                        && if domain == "upper_95" {
                            *value > cutoff
                        } else {
                            *value <= cutoff
                        };
                    keep[row] &= pass;
                }
            }
        }
    }
    for (key, values) in table.iter_mut() {
        if key == "beam_on" || key == "expected_sigma" || key == "session_avg_rate" {
            continue;
        }
        for (value, keep) in values.iter_mut().zip(&keep) {
            if !keep {
                *value = f32::NAN;
            }
        }
    }
}

const AUTO_LEVELS: usize = 100;
const AUTO_QUANTILES: usize = 32;

enum BinChoice {
    Automatic,
    Fixed(usize),
}

fn bin_choice(options: &Value) -> BinChoice {
    let raw = text(options, "bins", "Automatic");
    if raw.eq_ignore_ascii_case("automatic") {
        BinChoice::Automatic
    } else {
        BinChoice::Fixed(raw.parse::<usize>().unwrap_or(AUTO_QUANTILES).clamp(2, 64))
    }
}

fn bin_label(choice: &BinChoice) -> String {
    match choice {
        BinChoice::Automatic => "Automatic".to_string(),
        BinChoice::Fixed(count) => count.to_string(),
    }
}

/// Sorted levels when fewer than `limit` exist. `None` means the column is continuous.
fn quantized_levels(
    tables: &[BTreeMap<String, Vec<f32>>],
    key: &str,
    limit: usize,
) -> Option<Vec<f32>> {
    let mut levels = Vec::new();
    for table in tables {
        let Some(column) = table.get(key) else {
            continue;
        };
        for value in column {
            if !value.is_finite() || levels.iter().any(|have| same(*have, *value)) {
                continue;
            }
            if levels.len() + 1 >= limit {
                return None;
            }
            levels.push(*value);
        }
    }
    levels.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Some(levels)
}

/// One X rule for every column. Automatic keeps a short quantized axis and
/// otherwise uses 32 quantile bins. A fixed count always regroups.
fn assign_x(
    tables: &mut [BTreeMap<String, Vec<f32>>],
    column: &str,
    choice: &BinChoice,
) -> Vec<f32> {
    let quantiles = match choice {
        BinChoice::Fixed(count) => *count,
        BinChoice::Automatic => match quantized_levels(tables, column, AUTO_LEVELS) {
            Some(levels) => {
                for table in tables.iter_mut() {
                    let values = table.get(column).cloned().unwrap_or_default();
                    table.insert("_bin".to_string(), values);
                }
                return levels;
            }
            None => AUTO_QUANTILES,
        },
    };
    let values: Vec<f32> = tables
        .iter()
        .flat_map(|table| table.get(column).into_iter().flatten().copied())
        .collect();
    let edges = quantile_edges(&values, quantiles);
    for table in tables.iter_mut() {
        let centers =
            assign_bin_centers(table.get(column).map(Vec::as_slice).unwrap_or(&[]), &edges);
        table.insert("_bin".to_string(), centers);
    }
    unique_values(tables, "_bin")
}

fn unique_values(tables: &[BTreeMap<String, Vec<f32>>], key: &str) -> Vec<f32> {
    let mut values = Vec::new();
    for table in tables {
        if let Some(column) = table.get(key) {
            for value in column {
                if value.is_finite() && !values.iter().any(|have| same(*have, *value)) {
                    values.push(*value);
                }
            }
        }
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    values
}

fn dose_rate_table(table: &BTreeMap<String, Vec<f32>>) -> BTreeMap<String, Vec<f32>> {
    let mut out = BTreeMap::new();
    if let (Some(energy), Some(rate)) = (table.get("rate_energy"), table.get("mu_rate")) {
        out.insert("energy".to_string(), energy.clone());
        out.insert("mu_rate".to_string(), rate.clone());
    }
    if let Some(avg) = table.get("session_avg_rate") {
        out.insert("session_avg_rate".to_string(), avg.clone());
    }
    out
}

fn group_samples(bins: Option<&Vec<f32>>, y: &[f32], categories: &[f32]) -> Vec<Vec<f32>> {
    let mut groups = vec![Vec::new(); categories.len()];
    let Some(bins) = bins else {
        return groups;
    };
    if categories.is_empty() {
        return groups;
    }
    for (bin, value) in bins.iter().zip(y) {
        if !bin.is_finite() || !value.is_finite() {
            continue;
        }
        let mut index = categories.partition_point(|category| *category < *bin);
        if index == categories.len() || !same(categories[index], *bin) {
            if index > 0 && same(categories[index - 1], *bin) {
                index -= 1;
            } else {
                continue;
            }
        }
        groups[index].push(*value);
    }
    groups
}

fn kde(values: &[f32], half: f32) -> Vec<(f32, f32)> {
    if values.is_empty() {
        return Vec::new();
    }
    if values.len() == 1 || values.iter().all(|value| same(*value, values[0])) {
        return vec![(values[0], half * 0.7)];
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f32>() as f64 / n;
    let var = values
        .iter()
        .map(|value| (f64::from(*value) - mean).powi(2))
        .sum::<f64>()
        / (n - 1.0);
    let bw = n.powf(-0.2) * var.sqrt();
    if bw < 1e-9 {
        return vec![(values[0], half * 0.7)];
    }
    let lo = values.iter().copied().fold(f32::MAX, f32::min);
    let hi = values.iter().copied().fold(f32::MIN, f32::max);
    let steps = 48;
    let mut density = Vec::with_capacity(steps);
    for step in 0..steps {
        let y = lo + (hi - lo) * step as f32 / (steps - 1) as f32;
        let mut total = 0.0f64;
        for sample in values {
            let z = (f64::from(y) - f64::from(*sample)) / bw;
            total += (-0.5 * z * z).exp();
        }
        density.push((y, total));
    }
    let peak = density
        .iter()
        .map(|(_, value)| *value)
        .fold(0.0, f64::max)
        .max(1e-12);
    density
        .into_iter()
        .map(|(y, value)| (y, half * (value / peak) as f32))
        .collect()
}

fn pchip(xs: &[f32], ys: &[f32]) -> (Vec<f32>, Vec<f32>) {
    if xs.len() < 2 {
        return (xs.to_vec(), ys.to_vec());
    }
    let n = xs.len();
    let mut h = vec![0.0f64; n - 1];
    let mut delta = vec![0.0f64; n - 1];
    for i in 0..n - 1 {
        h[i] = f64::from(xs[i + 1] - xs[i]).max(1e-9);
        delta[i] = f64::from(ys[i + 1] - ys[i]) / h[i];
    }
    let mut slope = vec![0.0f64; n];
    slope[0] = delta[0];
    slope[n - 1] = delta[n - 2];
    for i in 1..n - 1 {
        slope[i] = if delta[i - 1] * delta[i] <= 0.0 {
            0.0
        } else {
            (h[i] + h[i - 1]) / (h[i] / delta[i - 1] + h[i - 1] / delta[i])
        };
    }
    let mut out_x = Vec::new();
    let mut out_y = Vec::new();
    for i in 0..n - 1 {
        for step in 0..8 {
            let t = step as f64 / 8.0;
            let t2 = t * t;
            let t3 = t2 * t;
            let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
            let h10 = t3 - 2.0 * t2 + t;
            let h01 = -2.0 * t3 + 3.0 * t2;
            let h11 = t3 - t2;
            out_x.push(xs[i] + (t as f32) * (xs[i + 1] - xs[i]));
            out_y.push(
                (h00 * f64::from(ys[i])
                    + h10 * h[i] * slope[i]
                    + h01 * f64::from(ys[i + 1])
                    + h11 * h[i] * slope[i + 1]) as f32,
            );
        }
    }
    out_x.push(*xs.last().unwrap_or(&0.0));
    out_y.push(*ys.last().unwrap_or(&0.0));
    (out_x, out_y)
}

fn uniform_edges(lo: f32, hi: f32, bins: usize) -> Vec<f32> {
    let bins = bins.max(1);
    let (lo, hi) = if !lo.is_finite() || !hi.is_finite() || (hi - lo).abs() < 1e-12 {
        (lo - 0.5, (lo - 0.5) + 1.0)
    } else {
        (lo, hi)
    };
    (0..=bins)
        .map(|step| lo + (hi - lo) * step as f32 / bins as f32)
        .collect()
}

fn count_bins(edges: &[f32], values: &[f32]) -> Vec<f32> {
    let bins = edges.len().saturating_sub(1);
    let mut counts = vec![0.0f32; bins];
    if bins == 0 {
        return counts;
    }
    let lo = edges[0];
    let hi = edges[bins];
    let span = hi - lo;
    if !span.is_finite() || span <= 0.0 {
        return counts;
    }
    let scale = bins as f32 / span;
    for value in values {
        if !value.is_finite() || *value < lo || *value > hi {
            continue;
        }
        let mut index = ((*value - lo) * scale) as usize;
        if index >= bins {
            index = bins - 1;
        }
        counts[index] += 1.0;
    }
    counts
}

fn bar_outline(edges: &[f32], counts: &[f32]) -> Series {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for (index, count) in counts.iter().copied().enumerate() {
        if !count.is_finite() || count <= 0.0 {
            continue;
        }
        let (Some(left), Some(right)) = (edges.get(index).copied(), edges.get(index + 1).copied())
        else {
            continue;
        };
        if !left.is_finite() || !right.is_finite() {
            continue;
        }
        xs.extend([left, right, right, left, left, f32::NAN]);
        ys.extend([0.0, 0.0, count, count, 0.0, f32::NAN]);
    }
    Series::Polyline {
        xs,
        ys,
        color: [0.85, 0.85, 0.85, 0.0],
        thickness: 1.0,
    }
}

fn has_finite(values: Option<&Vec<f32>>) -> bool {
    values.is_some_and(|values| values.iter().any(|value| value.is_finite()))
}

fn same(a: f32, b: f32) -> bool {
    (a - b).abs() <= 1e-4 * (1.0 + a.abs().max(b.abs()))
}

fn span(values: &[f32]) -> (f32, f32) {
    let finite: Vec<f32> = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    if finite.is_empty() {
        return (0.0, 1.0);
    }
    let lo = finite.iter().copied().fold(f32::MAX, f32::min);
    let hi = finite.iter().copied().fold(f32::MIN, f32::max);
    if (hi - lo).abs() < 1e-6 {
        return (lo - 1.0, hi + 1.0);
    }
    let pad = (hi - lo) * 0.05;
    (lo - pad, hi + pad)
}

fn median(values: &[f32]) -> f32 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[mid - 1] + sorted[mid]) * 0.5
    } else {
        sorted[mid]
    }
}

fn percentile(sorted_or_not: &[f32], q: f64) -> f32 {
    let mut sorted = sorted_or_not.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if sorted.is_empty() {
        return f32::NAN;
    }
    let pos = q * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil().min((sorted.len() - 1) as f64) as usize;
    let frac = (pos - lo as f64) as f32;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

fn modified_z(values: &[f32]) -> Vec<f32> {
    let finite: Vec<f32> = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    let mut out = vec![0.0; values.len()];
    if finite.is_empty() {
        return out;
    }
    let med = median(&finite);
    let devs: Vec<f32> = finite.iter().map(|value| (value - med).abs()).collect();
    let mad = median(&devs);
    let mean = finite.iter().sum::<f32>() / finite.len() as f32;
    let mean_ad = devs.iter().sum::<f32>() / devs.len() as f32;
    for (slot, value) in out.iter_mut().zip(values) {
        if !value.is_finite() {
            continue;
        }
        *slot = if mad > 1e-12 {
            0.6745 * (value - med) / mad
        } else if mean_ad > 1e-12 {
            (value - mean) / (1.253314 * mean_ad)
        } else {
            0.0
        };
    }
    out
}

fn unique_seen(values: &[f32]) -> Vec<f32> {
    let mut out = Vec::new();
    for value in values {
        if value.is_finite() && !out.iter().any(|have: &f32| same(*have, *value)) {
            out.push(*value);
        }
    }
    out
}

fn spot_delivery_ms(timestamp: &[f64], layer: &[f32]) -> Vec<f32> {
    let n = timestamp.len().min(layer.len());
    let mut out = vec![f32::NAN; n];
    let mut groups: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
    for index in 0..n {
        if timestamp[index].is_finite() && layer[index].is_finite() {
            groups
                .entry(layer[index].round() as i64)
                .or_default()
                .push(index);
        }
    }
    for indices in groups.values_mut() {
        indices.sort_by(|&a, &b| {
            timestamp[a]
                .partial_cmp(&timestamp[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if indices.is_empty() {
            continue;
        }
        out[indices[0]] = timestamp[indices[0]] as f32;
        for pair in indices.windows(2) {
            out[pair[1]] = (timestamp[pair[1]] - timestamp[pair[0]]) as f32;
        }
    }
    out
}

struct Band {
    device: String,
    min: f64,
    max: f64,
    k: [f64; 4],
}

fn parse_sigma_bands(xml: &str) -> Vec<Band> {
    let mut bands = Vec::new();
    let mut device = String::new();
    let mut rest = xml;
    while let Some(start) = rest.find('<') {
        rest = &rest[start + 1..];
        let tag = rest.split('>').next().unwrap_or("");
        if tag.starts_with("device") {
            if let Some(name) = xml_attr(tag, "name") {
                device = name;
            }
        } else if tag.starts_with("beam_sigma_conversions")
            && xml_attr(tag, "in_units")
                .unwrap_or_default()
                .eq_ignore_ascii_case("MEV")
            && xml_attr(tag, "out_units")
                .unwrap_or_default()
                .eq_ignore_ascii_case("mm")
        {
            bands.push(Band {
                device: device.clone(),
                min: xml_attr(tag, "min_energy")
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(f64::MIN),
                max: xml_attr(tag, "max_energy")
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(f64::MAX),
                k: [
                    xml_attr(tag, "K0")
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(0.0),
                    xml_attr(tag, "K1")
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(0.0),
                    xml_attr(tag, "K2")
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(0.0),
                    xml_attr(tag, "K3")
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(0.0),
                ],
            });
        }
    }
    bands
}

pub(crate) fn interlock_sigma_mm(xml: &str, device: &str, energy: f64) -> Option<f64> {
    expected_mm(&parse_sigma_bands(xml), device, energy)
}

fn expected_mm(bands: &[Band], device: &str, energy: f64) -> Option<f64> {
    let band = bands
        .iter()
        .find(|band| band.device == device && energy >= band.min && energy <= band.max)?;
    let e = energy;
    Some(band.k[0] + band.k[1] * e + band.k[2] * e * e + band.k[3] * e * e * e)
}

fn xml_attr(tag: &str, name: &str) -> Option<String> {
    let key = format!("{name}=");
    let at = tag.find(&key)? + key.len();
    let rest = tag[at..].trim_start();
    let quote = rest.chars().next()?;
    if quote != '"' && quote != '\'' {
        return None;
    }
    let body = &rest[1..];
    Some(body.split(quote).next()?.to_string())
}

fn header_base(name: &str) -> &str {
    name.split('.').next().unwrap_or(name)
}

fn timeline_column(name: &str) -> bool {
    current_column(name) || signal_column(name)
}

fn current_column(name: &str) -> bool {
    let lower = header_base(name).to_ascii_lowercase();
    lower == "layer_id"
        || lower == "rci_in_trigger"
        || lower == "r_beamok"
        || lower == "beam_on"
        || lower.contains("current")
        || lower.contains("primary_channel")
}

fn signal_column(name: &str) -> bool {
    let lower = header_base(name).to_ascii_lowercase();
    if lower == "layer_id"
        || lower == "rci_in_trigger"
        || lower == "r_beamok"
        || lower == "beam_on"
        || lower == "c_x"
        || lower == "c_y"
        || lower.contains("confidence")
        || lower.contains("peak")
    {
        return true;
    }
    const NAMES: &[&str] = &[
        "amplifier_x_target",
        "amplifier_cmd_x",
        "amplifier_y_target",
        "amplifier_cmd_y",
        "amplifier_x_readback",
        "r_xi",
        "r_xv",
        "amplifier_readback_x",
        "amplifier_y_readback",
        "r_yi",
        "r_yv",
        "amplifier_readback_y",
        "r_tx2_probe_x",
        "field_c_x",
        "r_xb",
        "mag_field_x",
        "field_x",
        "b_field_x",
        "r_tx2_probe_y",
        "field_c_y",
        "r_yb",
        "mag_field_y",
        "field_y",
        "b_field_y",
    ];
    NAMES.contains(&lower.as_str())
}

fn slice_column(name: &str) -> bool {
    let lower = header_base(name).to_ascii_lowercase();
    lower == "layer_id"
        || lower == "spot_no"
        || lower == "rci_in_trigger"
        || lower == "r_beamok"
        || lower == "beam_on"
        || lower.contains("position")
        || lower.contains("sigma")
        || lower.contains("fit_ok")
        || lower.contains("confidence")
        || lower.contains("error_code")
}

// ponytail: a few chunks, not a pool. Tens of timeslice files, not thousands.
fn read_sheets(files: &[Vec<u8>], keep: fn(&str) -> bool) -> Vec<Sheet> {
    if files.len() < 2 {
        return files
            .iter()
            .map(|bytes| read_sheet_where(bytes, keep))
            .collect();
    }
    let workers = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1)
        .clamp(1, files.len());
    let chunk = files.len().div_ceil(workers);
    let mut sheets = Vec::with_capacity(files.len());
    std::thread::scope(|scope| {
        let mut joins = Vec::new();
        for piece in files.chunks(chunk) {
            joins.push(scope.spawn(move || {
                piece
                    .iter()
                    .map(|bytes| read_sheet_where(bytes, keep))
                    .collect::<Vec<_>>()
            }));
        }
        for join in joins {
            sheets.extend(join.join().unwrap());
        }
    });
    sheets
}

enum Slot {
    Skip,
    Num(Vec<f32>),
    Wide(Vec<f64>),
    Clock(Vec<f64>),
}

enum Num {
    Nan,
    Value(f64),
    Std,
}

fn read_sheet(bytes: &[u8]) -> Sheet {
    read_sheet_where(bytes, |_| true)
}

/// Parse a device CSV without a `String` per cell.
///
/// ponytail: the decimal scanner handles the padded fixed-point form these logs
/// use. Scientific notation and non-finite tokens fall back to `str::parse`.
/// `keep` drops columns a caller will not read. Strip columns stay dropped.
fn read_sheet_where(bytes: &[u8], keep: impl Fn(&str) -> bool) -> Sheet {
    let text = String::from_utf8_lossy(bytes);
    let mut lines = text.lines();
    let Some(header) = lines.next() else {
        return Sheet {
            num: BTreeMap::new(),
            wide: BTreeMap::new(),
        };
    };
    let rows = bytes.iter().filter(|byte| **byte == b'\n').count();
    let mut used = HashMap::<String, ()>::new();
    let mut names = Vec::new();
    let mut slots = Vec::new();
    let mut cursor = 0usize;
    while let Some((start, end, next)) = next_cell(header, cursor) {
        let name = cell_owned(header, start, end);
        let base = header_base(&name);
        // ponytail: strip columns are not a binned metric. Stop skipping them if one is.
        if name.is_empty() || base.to_ascii_lowercase().contains("strip") || !keep(&name) {
            names.push(None);
            slots.push(Slot::Skip);
        } else {
            let key = unique_header(&name, &mut used);
            let slot = if base == "datetime" {
                Slot::Clock(Vec::with_capacity(rows))
            } else if matches!(base, "timestamp" | "time_s" | "time_ns") {
                Slot::Wide(Vec::with_capacity(rows))
            } else {
                Slot::Num(Vec::with_capacity(rows))
            };
            names.push(Some(key));
            slots.push(slot);
        }
        if next <= cursor {
            break;
        }
        cursor = next;
    }
    let mut seen = vec![0u32; slots.len()];
    let mut row = 1u32;
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let mut cursor = 0usize;
        let mut index = 0usize;
        while index < slots.len() {
            let Some((start, end, next)) = next_cell(line, cursor) else {
                break;
            };
            push_cell(&mut slots[index], line, start, end);
            seen[index] = row;
            index += 1;
            if next <= cursor {
                break;
            }
            cursor = next;
        }
        for (index, slot) in slots.iter_mut().enumerate() {
            if seen[index] != row {
                push_nan(slot);
            }
        }
        row = row.wrapping_add(1);
        if row == 0 {
            row = 1;
            seen.fill(0);
        }
    }
    let mut num = BTreeMap::new();
    let mut wide = BTreeMap::new();
    for (name, slot) in names.into_iter().zip(slots) {
        let Some(name) = name else {
            continue;
        };
        match slot {
            Slot::Skip => {}
            Slot::Num(values) => {
                num.insert(name, values);
            }
            Slot::Wide(values) | Slot::Clock(values) => {
                wide.insert(name, values);
            }
        }
    }
    Sheet { num, wide }
}

fn unique_header(name: &str, used: &mut HashMap<String, ()>) -> String {
    if !used.contains_key(name) {
        used.insert(name.to_string(), ());
        return name.to_string();
    }
    let mut n = 1;
    loop {
        let candidate = format!("{name}.{n}");
        if !used.contains_key(&candidate) {
            used.insert(candidate.clone(), ());
            return candidate;
        }
        n += 1;
    }
}

fn next_cell(line: &str, start: usize) -> Option<(usize, usize, usize)> {
    let bytes = line.as_bytes();
    if start > bytes.len() {
        return None;
    }
    if start == bytes.len() {
        if start > 0 && bytes[start - 1] == b',' {
            return Some((start, start, start + 1));
        }
        return None;
    }
    if bytes[start] == b'"' {
        let mut i = start + 1;
        while i < bytes.len() {
            if bytes[i] == b'"' {
                if i + 1 < bytes.len() && bytes[i + 1] == b'"' {
                    i += 2;
                    continue;
                }
                i += 1;
                break;
            }
            i += 1;
        }
        let end = i;
        let next = if i < bytes.len() && bytes[i] == b',' {
            i + 1
        } else {
            i
        };
        return Some((start, end, next));
    }
    let mut i = start;
    while i < bytes.len() && bytes[i] != b',' {
        i += 1;
    }
    let next = if i < bytes.len() { i + 1 } else { i };
    Some((start, i, next))
}

fn cell_owned(line: &str, start: usize, end: usize) -> String {
    let cell = &line[start..end];
    if cell.as_bytes().first() == Some(&b'"') {
        strip_quotes(cell)
    } else {
        cell.to_string()
    }
}

fn strip_quotes(cell: &str) -> String {
    cell.replace('"', "")
}

fn push_cell(slot: &mut Slot, line: &str, start: usize, end: usize) {
    match slot {
        Slot::Skip => {}
        Slot::Num(values) => values.push(cell_f32(line, start, end)),
        Slot::Wide(values) => values.push(cell_f64(line, start, end)),
        Slot::Clock(values) => values.push(cell_datetime(line, start, end)),
    }
}

fn push_nan(slot: &mut Slot) {
    match slot {
        Slot::Skip => {}
        Slot::Num(values) => values.push(f32::NAN),
        Slot::Wide(values) | Slot::Clock(values) => values.push(f64::NAN),
    }
}

fn cell_f32(line: &str, start: usize, end: usize) -> f32 {
    let cell = &line[start..end];
    if cell.as_bytes().first() == Some(&b'"') {
        let stripped = strip_quotes(cell);
        return f32_bytes(stripped.as_bytes());
    }
    f32_bytes(cell.as_bytes())
}

fn f32_bytes(bytes: &[u8]) -> f32 {
    match parse_decimal(bytes) {
        Num::Nan => f32::NAN,
        Num::Value(value) => value as f32,
        Num::Std => std::str::from_utf8(bytes)
            .ok()
            .and_then(|text| text.trim().parse().ok())
            .unwrap_or(f32::NAN),
    }
}

fn cell_f64(line: &str, start: usize, end: usize) -> f64 {
    let cell = &line[start..end];
    if cell.as_bytes().first() == Some(&b'"') {
        return strip_quotes(cell).trim().parse().unwrap_or(f64::NAN);
    }
    cell.trim().parse().unwrap_or(f64::NAN)
}

fn cell_datetime(line: &str, start: usize, end: usize) -> f64 {
    let cell = &line[start..end];
    if cell.as_bytes().first() == Some(&b'"') {
        return parse_datetime(&strip_quotes(cell)).unwrap_or(f64::NAN);
    }
    parse_datetime(cell).unwrap_or(f64::NAN)
}

fn parse_decimal(bytes: &[u8]) -> Num {
    let n = bytes.len();
    let mut i = 0;
    while i < n && bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    if i >= n {
        return Num::Nan;
    }
    if bytes[i] != b'+' && bytes[i] != b'-' && bytes[i] != b'.' && !bytes[i].is_ascii_digit() {
        return Num::Std;
    }
    let neg = if bytes[i] == b'+' || bytes[i] == b'-' {
        let minus = bytes[i] == b'-';
        i += 1;
        minus
    } else {
        false
    };
    let mut value = 0u64;
    let mut saw = false;
    while i < n && bytes[i].is_ascii_digit() {
        let digit = u64::from(bytes[i] - b'0');
        if value > (u64::MAX - digit) / 10 {
            return Num::Std;
        }
        value = value * 10 + digit;
        saw = true;
        i += 1;
    }
    let mut scale = 1u64;
    if i < n && bytes[i] == b'.' {
        i += 1;
        while i < n && bytes[i].is_ascii_digit() {
            let digit = u64::from(bytes[i] - b'0');
            if scale > u64::MAX / 10 || value > (u64::MAX - digit) / 10 {
                return Num::Std;
            }
            value = value * 10 + digit;
            scale *= 10;
            saw = true;
            i += 1;
        }
    }
    if !saw {
        return Num::Nan;
    }
    if i < n && (bytes[i] == b'e' || bytes[i] == b'E') {
        return Num::Std;
    }
    while i < n && bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    if i != n {
        return Num::Std;
    }
    let mut out = value as f64 / scale as f64;
    if neg {
        out = -out;
    }
    Num::Value(out)
}

fn parse_datetime(text: &str) -> Option<f64> {
    let text = text.trim();
    if text.is_empty() {
        return None;
    }
    let (date, time) = text.split_once(['T', ' ']).unwrap_or((text, "00:00:00"));
    let mut date_parts = date.split('-');
    let year: i32 = date_parts.next()?.parse().ok()?;
    let month: u32 = date_parts.next()?.parse().ok()?;
    let day: u32 = date_parts.next()?.parse().ok()?;
    let mut clock = time.split(':');
    let hour: u32 = clock.next().unwrap_or("0").parse().unwrap_or(0);
    let minute: u32 = clock.next().unwrap_or("0").parse().unwrap_or(0);
    let second: f64 = clock.next().unwrap_or("0").parse().unwrap_or(0.0);
    let days = days_from_civil(year, month, day)?;
    Some(days as f64 * 86400.0 + f64::from(hour) * 3600.0 + f64::from(minute) * 60.0 + second)
}

fn days_from_civil(mut year: i32, month: u32, day: u32) -> Option<i64> {
    if !(1..=12).contains(&month) || day == 0 || day > 31 {
        return None;
    }
    if month <= 2 {
        year -= 1;
    }
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let yoe = (year - era * 400) as u32;
    let month_prime = if month > 2 { month - 3 } else { month + 9 };
    let doy = (153 * month_prime + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    Some(i64::from(era) * 146097 + i64::from(doe) - 719468)
}

fn text<'a>(options: &'a Value, key: &str, default: &'a str) -> &'a str {
    options.get(key).and_then(Value::as_str).unwrap_or(default)
}

pub(crate) fn pick(
    options: &Value,
    key: &str,
    default_id: &'static str,
    pairs: &[(&'static str, &'static str)],
) -> &'static str {
    pick_in(options, key, pairs).unwrap_or(default_id)
}

fn pick_in<'a>(options: &Value, key: &str, pairs: &[(&'a str, &'a str)]) -> Option<&'a str> {
    let raw = options.get(key).and_then(Value::as_str)?;
    pairs
        .iter()
        .find(|(id, label)| *id == raw || *label == raw)
        .map(|(id, _)| *id)
}

fn flag(options: &Value, key: &str, default_on: bool) -> bool {
    match text(options, key, if default_on { "On" } else { "Off" }) {
        "off" | "Off" | "false" | "0" => false,
        "on" | "On" | "true" | "1" => true,
        _ => default_on,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use scan_kit_core::Series;

    use super::{
        apply_filter, assign_x, axis_label, binned_summary, binned_trend, contour_bands,
        days_from_civil, parse_datetime, read_sheet, signal_table, spot_delivery_ms, violin_series,
        BinChoice, Trend,
    };

    #[test]
    fn civil_1970_is_unix_zero() {
        assert_eq!(days_from_civil(1970, 1, 1), Some(0));
        assert!(parse_datetime("1970-01-01 00:00:00").unwrap().abs() < 1e-6);
    }

    #[test]
    fn first_spot_in_a_layer_keeps_its_timestamp() {
        let ms = spot_delivery_ms(&[1000.0, 1004.0, 1010.0], &[1.0, 1.0, 1.0]);
        assert_eq!(ms, vec![1000.0, 4.0, 6.0]);
    }

    #[test]
    fn duplicate_headers_keep_the_first_column() {
        let sheet = read_sheet(b"energy,energy,charge_req\n70,999,1\n");
        assert_eq!(sheet.num["energy"], vec![70.0]);
        assert_eq!(sheet.num["charge_req"], vec![1.0]);
    }

    #[test]
    fn sheet_cells_keep_alignment() {
        let sheet = read_sheet(
            b"energy,ic1_strip_sum,energy,dose,timestamp,datetime\n\"70.5\",9,1.5e-3,2,826.0233764648437500,1970-01-01 00:00:00\n3\n",
        );
        assert!((sheet.num["energy"][0] - 70.5).abs() < 1e-4);
        assert!((sheet.num["energy.1"][0] - 0.0015).abs() < 1e-6);
        assert!((sheet.num["dose"][0] - 2.0).abs() < 1e-6);
        assert!((sheet.num["energy"][1] - 3.0).abs() < 1e-6);
        assert!(sheet.num["energy.1"][1].is_nan());
        assert!(sheet.num["dose"][1].is_nan());
        assert!(!sheet.num.contains_key("ic1_strip_sum"));
        let expected: f64 = "826.0233764648437500".parse().unwrap();
        assert!((sheet.wide["timestamp"][0] - expected).abs() <= 1e-6);
        assert!(sheet.wide["timestamp"][1].is_nan());
        assert!(sheet.wide["datetime"][0].abs() < 1e-6);
        assert!(sheet.wide["datetime"][1].is_nan());
    }

    #[test]
    fn first_log_on_disk_has_finite_dose_error() {
        let root =
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../test_data/logs");
        let Ok(entries) = std::fs::read_dir(&root) else {
            return;
        };
        let Some(session) = entries.filter_map(|entry| entry.ok()).find_map(|entry| {
            let id = entry.file_name().to_string_lossy().into_owned();
            crate::discover::session_directory(&root, &id)
                .join("spot_data.csv")
                .is_file()
                .then_some(id)
        }) else {
            return;
        };
        let scene = binned_summary(&root, &[session], &serde_json::json!({}));
        assert!(
            scene
                .panels
                .iter()
                .any(|panel| panel.title.starts_with("IC1")),
            "dose error should plot when the log has padded dose numbers",
        );
    }

    #[test]
    fn padded_log_numbers_stay_finite() {
        let sheet = read_sheet(
            b"ENERGY,CHARGE_REQ,ic1_total_dose_spot\n250.0,0.005,     0.0055517933338228\n",
        );
        assert!((sheet.num["ENERGY"][0] - 250.0).abs() < 1e-3);
        assert!((sheet.num["ic1_total_dose_spot"][0] - 0.005551793).abs() < 1e-6);
    }

    #[test]
    fn scatter_position_and_dose_rate_follow_the_python_rules() {
        let root = std::env::temp_dir().join(format!("scan-kit-binned-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let session = root.join("sess");
        std::fs::create_dir_all(&session).unwrap();
        std::fs::write(
            session.join("input_map.csv"),
            "energy,charge_req,position_x,position_y,layer_id\n70,1,0,0,1\n70,1,0,0,1\n90,2,10,0,2\n",
        )
        .unwrap();
        std::fs::write(
            session.join("spot_data.csv"),
            "ic1_total_dose,r_ic1_x_spot_position,r_ic1_y_spot_position,layer_id,timestamp,timestamp\n1.1,1,0,1,0,999\n1.1,3,0,1,1000,999\n2.2,12,-1,2,2000,999\n",
        )
        .unwrap();
        let ids = ["sess".to_string()];
        let scatter = binned_summary(&root, &ids, &serde_json::json!({"glyph": "Scatter"}));
        let dose = scatter
            .panels
            .iter()
            .find(|panel| panel.title.starts_with("IC1"))
            .unwrap();
        assert!(dose.xmin > 50.0, "scatter x is energy in MeV");
        let position = binned_summary(
            &root,
            &ids,
            &serde_json::json!({"metric": "Position Error (mm)"}),
        );
        let xerr = position
            .panels
            .iter()
            .find(|panel| panel.title == "IC1 X")
            .unwrap();
        let marks = rect_ys(xerr);
        assert!(marks.iter().any(|value| (*value - 1.0).abs() < 1e-3));
        let rate = binned_summary(
            &root,
            &ids,
            &serde_json::json!({"metric": "Dose Rate (MU/s)"}),
        );
        let mu = rate
            .panels
            .iter()
            .find(|panel| panel.title == "MU/S")
            .unwrap();
        assert!(rect_ys(mu).iter().any(|value| (*value - 2.0).abs() < 1e-3));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn sigma_is_twice_the_logged_column() {
        let root = std::env::temp_dir().join("scan-kit-binned-sigma");
        let _ = std::fs::remove_dir_all(&root);
        let session = root.join("sess");
        std::fs::create_dir_all(&session).unwrap();
        std::fs::write(session.join("input_map.csv"), "energy,charge_req\n250,1\n").unwrap();
        std::fs::write(
            session.join("spot_data.csv"),
            "ic1_total_dose_spot,r_ic1_x_spot_sigma\n1,3\n",
        )
        .unwrap();
        let scene = binned_summary(
            &root,
            &["sess".to_string()],
            &serde_json::json!({"metric": "Sigma (mm)"}),
        );
        let panel = scene
            .panels
            .iter()
            .find(|panel| panel.title == "IC1 SX")
            .unwrap();
        assert!(rect_ys(panel)
            .iter()
            .any(|value| (*value - 6.0).abs() < 1e-3));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn chamber_raw_maps_strip_64_5_to_zero() {
        let root = std::env::temp_dir().join("scan-kit-binned-chamber");
        let _ = std::fs::remove_dir_all(&root);
        let session = root.join("sess");
        std::fs::create_dir_all(&session).unwrap();
        std::fs::write(
            session.join("input_map.csv"),
            "energy,charge_req,position_x,position_y\n250,1,0,0\n",
        )
        .unwrap();
        std::fs::write(
            session.join("spot_data.csv"),
            "ic1_total_dose_spot,r_ic1_x_spot_position,r_ic1_y_spot_position,r_ic2_x_spot_position,r_ic2_y_spot_position,r_ic1_x_spot_position_raw,r_ic1_y_spot_position_raw,r_ic2_x_spot_position_raw,r_ic2_y_spot_position_raw\n1,5,0,0,0,64.5,64.5,64.5,64.5\n",
        )
        .unwrap();
        let ids = ["sess".to_string()];
        let iso = binned_summary(
            &root,
            &ids,
            &serde_json::json!({"metric": "Position Error (mm)"}),
        );
        let iso_x = iso
            .panels
            .iter()
            .find(|panel| panel.title == "IC1 X")
            .unwrap();
        assert!(rect_ys(iso_x)
            .iter()
            .any(|value| (*value - 5.0).abs() < 1e-3));
        let chamber = binned_summary(
            &root,
            &ids,
            &serde_json::json!({"metric": "Position Error (mm)", "source": "chamber"}),
        );
        let chamber_x = chamber
            .panels
            .iter()
            .find(|panel| panel.title == "IC1 X")
            .unwrap();
        assert!(rect_ys(chamber_x).iter().any(|value| value.abs() < 1e-2));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn spot_time_uses_time_s_and_time_ns_when_timestamp_is_absent() {
        let root = std::env::temp_dir().join("scan-kit-binned-times");
        let _ = std::fs::remove_dir_all(&root);
        let session = root.join("sess");
        std::fs::create_dir_all(&session).unwrap();
        std::fs::write(
            session.join("input_map.csv"),
            "energy,charge_req,layer_id\n70,1,1\n",
        )
        .unwrap();
        std::fs::write(
            session.join("spot_data.csv"),
            "layer_id,ic1_total_dose_spot,time_s,time_ns\n1,1,1,0\n",
        )
        .unwrap();
        let scene = binned_summary(
            &root,
            &["sess".to_string()],
            &serde_json::json!({"metric": "Spot Delivery Time"}),
        );
        let total = scene
            .panels
            .iter()
            .find(|panel| panel.title == "TOTAL MS")
            .unwrap();
        assert!(rect_ys(total)
            .iter()
            .any(|value| (*value - 1000.0).abs() < 1e-2));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn beam_on_time_comes_from_the_layer_run() {
        let root = std::env::temp_dir().join("scan-kit-binned-point");
        let _ = std::fs::remove_dir_all(&root);
        let session = root.join("sess");
        std::fs::create_dir_all(session.join("layer-0/run-0")).unwrap();
        std::fs::write(
            session.join("input_map.csv"),
            "energy,charge_req,layer_id\n250,1,7\n",
        )
        .unwrap();
        std::fs::write(
            session.join("spot_data.csv"),
            "spot_no,layer_id,ic1_total_dose_spot,timestamp\n4,7,1,1000\n",
        )
        .unwrap();
        std::fs::write(
            session.join("layer-0/run-0/FX4_spot_data.csv"),
            "spot_no,layer_id,point_time(ms)\n4,7,12.5\n",
        )
        .unwrap();
        let scene = binned_summary(
            &root,
            &["sess".to_string()],
            &serde_json::json!({"metric": "Spot Delivery Time"}),
        );
        let beam = scene
            .panels
            .iter()
            .find(|panel| panel.title == "BEAM ON")
            .unwrap();
        assert!(rect_ys(beam)
            .iter()
            .any(|value| (*value - 12.5).abs() < 1e-3));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn timeslice_sigma_stays_in_millimetres() {
        let root = std::env::temp_dir().join("scan-kit-binned-slice-sigma");
        let _ = std::fs::remove_dir_all(&root);
        let session = root.join("sess");
        std::fs::create_dir_all(session.join("layer-0/run-0")).unwrap();
        std::fs::write(session.join("input_map.csv"), "energy,layer_id\n70,1\n").unwrap();
        std::fs::write(session.join("spot_data.csv"), "ic1_total_dose_spot\n1\n").unwrap();
        std::fs::write(
            session.join("layer-0/run-0/timeslice_data_device_units.csv"),
            "layer_id,rci_in_trigger,r_ic1_x_sigma,r_ic1_x_spot_error_code\n1,1,4,0\n1,1,30,0\n",
        )
        .unwrap();
        let scene = binned_summary(
            &root,
            &["sess".to_string()],
            &serde_json::json!({"metric": "Sigma (mm)", "source": "timeslice_iso"}),
        );
        let panel = scene
            .panels
            .iter()
            .find(|panel| panel.title == "IC1 SX")
            .unwrap();
        let marks = rect_ys(panel);
        assert!(marks.iter().any(|value| (*value - 4.0).abs() < 1e-2));
        assert!(marks.iter().all(|value| *value < 20.0));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn timeslice_isocenter_error_uses_the_strip_fit() {
        let root = std::env::temp_dir().join("scan-kit-binned-slice-pos");
        let _ = std::fs::remove_dir_all(&root);
        let session = root.join("sess");
        std::fs::create_dir_all(session.join("layer-0/run-0")).unwrap();
        let mut spot = String::from(
            "spot_no,layer_id,r_ic1_x_spot_position_raw,r_ic1_x_spot_position,r_ic1_y_spot_position_raw,r_ic1_y_spot_position,r_ic2_x_spot_position_raw,r_ic2_x_spot_position,r_ic2_y_spot_position_raw,r_ic2_y_spot_position\n",
        );
        let mut map = String::from("energy,layer_id,spot_no,position_x,position_y\n");
        for i in 0..12 {
            let strip = 20.0 + i as f32;
            let iso = strip - 64.5;
            spot.push_str(&format!(
                "{i},1,{strip},{iso},{strip},{iso},{strip},{iso},{strip},{iso}\n"
            ));
            map.push_str(&format!("70,1,{i},{iso},0\n"));
        }
        std::fs::write(session.join("spot_data.csv"), spot).unwrap();
        std::fs::write(session.join("input_map.csv"), map).unwrap();
        std::fs::write(
            session.join("layer-0/run-0/timeslice_data_device_units.csv"),
            "layer_id,spot_no,rci_in_trigger,r_ic1_x_position\n1,5,1,25\n",
        )
        .unwrap();
        let scene = binned_summary(
            &root,
            &["sess".to_string()],
            &serde_json::json!({"metric": "Position Error (mm)", "source": "Timeslice — Isocenter"}),
        );
        let panel = scene
            .panels
            .iter()
            .find(|panel| panel.title == "IC1 X")
            .unwrap();
        assert!(rect_ys(panel).iter().any(|value| value.abs() < 1e-2));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn timeslice_isocenter_error_reads_spot_position() {
        let root = std::env::temp_dir().join("scan-kit-binned-slice-spot-pos");
        let _ = std::fs::remove_dir_all(&root);
        let session = root.join("sess");
        std::fs::create_dir_all(session.join("layer-0/run-0")).unwrap();
        let mut spot = String::from(
            "spot_no,layer_id,r_ic1_x_spot_position_raw,r_ic1_x_spot_position,r_ic1_y_spot_position_raw,r_ic1_y_spot_position,r_ic2_x_spot_position_raw,r_ic2_x_spot_position,r_ic2_y_spot_position_raw,r_ic2_y_spot_position\n",
        );
        let mut map = String::from("energy,layer_id,spot_no,position_x,position_y\n");
        for i in 0..12 {
            let strip = 20.0 + i as f32;
            let iso = strip - 64.5;
            spot.push_str(&format!(
                "{i},1,{strip},{iso},{strip},{iso},{strip},{iso},{strip},{iso}\n"
            ));
            map.push_str(&format!("70,1,{i},{iso},0\n"));
        }
        std::fs::write(session.join("spot_data.csv"), spot).unwrap();
        std::fs::write(session.join("input_map.csv"), map).unwrap();
        std::fs::write(
            session.join("layer-0/run-0/timeslice_data_device_units.csv"),
            "layer_id,spot_no,rci_in_trigger,r_ic1_x_spot_position\n1,5,1,25\n",
        )
        .unwrap();
        let scene = binned_summary(
            &root,
            &["sess".to_string()],
            &serde_json::json!({"metric": "Position Error (mm)", "source": "Timeslice — Isocenter"}),
        );
        let panel = scene
            .panels
            .iter()
            .find(|panel| panel.title == "IC1 X")
            .unwrap();
        assert!(rect_ys(panel).iter().any(|value| value.abs() < 1e-2));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn timeslice_primary_channel_is_ic_current() {
        let root = std::env::temp_dir().join("scan-kit-binned-slice-current");
        let _ = std::fs::remove_dir_all(&root);
        let session = root.join("sess");
        std::fs::create_dir_all(session.join("layer-0/run-0")).unwrap();
        std::fs::write(session.join("input_map.csv"), "energy,layer_id\n70,1\n").unwrap();
        std::fs::write(
            session.join("layer-0/run-0/timeslice_data_device_units.csv"),
            "layer_id,ic1_primary_channel,ic3_current_A\n1,12,3\n",
        )
        .unwrap();
        let scene = binned_summary(
            &root,
            &["sess".to_string()],
            &serde_json::json!({"metric": "IC Current (nA)"}),
        );
        let ic1 = scene
            .panels
            .iter()
            .find(|panel| panel.title == "IC1")
            .unwrap();
        let ic3 = scene
            .panels
            .iter()
            .find(|panel| panel.title == "IC3")
            .unwrap();
        assert!(rect_ys(ic1).iter().any(|value| (value - 12.0).abs() < 1e-2));
        assert!(rect_ys(ic3).iter().any(|value| (value - 3.0).abs() < 1e-2));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn timeslice_energy_follows_the_layer_folder() {
        let root = std::env::temp_dir().join("scan-kit-binned-slice-energy");
        let _ = std::fs::remove_dir_all(&root);
        let session = root.join("sess");
        std::fs::create_dir_all(session.join("layer-0/run-0")).unwrap();
        std::fs::create_dir_all(session.join("layer-1/run-0")).unwrap();
        std::fs::write(
            session.join("input_map.csv"),
            "energy,layer_id\n180,9\n70,9\n",
        )
        .unwrap();
        std::fs::write(
            session.join("layer-0/run-0/timeslice_data_device_units.csv"),
            "layer_id,ic1_primary_channel\n9,11\n",
        )
        .unwrap();
        std::fs::write(
            session.join("layer-1/run-0/timeslice_data_device_units.csv"),
            "layer_id,ic1_primary_channel\n9,22\n",
        )
        .unwrap();
        let scene = binned_summary(
            &root,
            &["sess".to_string()],
            &serde_json::json!({"metric": "IC Current (nA)", "glyph": "Scatter"}),
        );
        let panel = scene
            .panels
            .iter()
            .find(|panel| panel.title == "IC1")
            .unwrap();
        let points = panel.series.iter().find_map(|series| match series {
            Series::Points { xs, ys, .. } => {
                Some(xs.iter().zip(ys).map(|(x, y)| (*x, *y)).collect::<Vec<_>>())
            }
            _ => None,
        });
        let points = points.unwrap();
        assert!(points
            .iter()
            .any(|(x, y)| (x - 180.0).abs() < 1.0 && (y - 11.0).abs() < 1e-2));
        assert!(points
            .iter()
            .any(|(x, y)| (x - 70.0).abs() < 1.0 && (y - 22.0).abs() < 1e-2));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn timeslice_ic2_uses_its_own_spot_number() {
        let root = std::env::temp_dir().join("scan-kit-binned-slice-ic2");
        let _ = std::fs::remove_dir_all(&root);
        let session = root.join("sess");
        std::fs::create_dir_all(session.join("layer-0/run-0")).unwrap();
        let mut spot = String::from(
            "spot_no,layer_id,r_ic1_x_spot_position_raw,r_ic1_x_spot_position,r_ic1_y_spot_position_raw,r_ic1_y_spot_position,r_ic2_x_spot_position_raw,r_ic2_x_spot_position,r_ic2_y_spot_position_raw,r_ic2_y_spot_position\n",
        );
        let mut map = String::from("energy,layer_id,spot_no,position_x,position_y\n");
        for i in 0..12 {
            let strip = 20.0 + i as f32;
            let iso = strip - 64.5;
            spot.push_str(&format!(
                "{i},1,{strip},{iso},{strip},{iso},{strip},{iso},{strip},{iso}\n"
            ));
            map.push_str(&format!("70,1,{i},{iso},0\n"));
        }
        std::fs::write(session.join("spot_data.csv"), spot).unwrap();
        std::fs::write(session.join("input_map.csv"), map).unwrap();
        std::fs::write(
            session.join("layer-0/run-0/timeslice_data_device_units.csv"),
            "layer_id,spot_no,spot_no,spot_no,rci_in_trigger,r_ic2_x_position\n1,0,1,5,1,25\n",
        )
        .unwrap();
        let scene = binned_summary(
            &root,
            &["sess".to_string()],
            &serde_json::json!({"metric": "Position Error (mm)", "source": "timeslice_iso"}),
        );
        let panel = scene
            .panels
            .iter()
            .find(|panel| panel.title == "IC2 X")
            .unwrap();
        assert!(rect_ys(panel).iter().any(|value| value.abs() < 1e-2));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn timeslice_sigma_rejects_low_confidence() {
        let root = std::env::temp_dir().join("scan-kit-binned-slice-conf");
        let _ = std::fs::remove_dir_all(&root);
        let session = root.join("sess");
        std::fs::create_dir_all(session.join("layer-0/run-0")).unwrap();
        std::fs::write(session.join("input_map.csv"), "energy,layer_id\n70,1\n").unwrap();
        std::fs::write(session.join("spot_data.csv"), "ic1_total_dose_spot\n1\n").unwrap();
        std::fs::write(
            session.join("layer-0/run-0/timeslice_data_device_units.csv"),
            "layer_id,rci_in_trigger,r_ic1_x_sigma,r_ic1_x_confidence,ic1_x_fit_ok,r_ic1_x_spot_error_code\n1,1,4,50,1,0\n",
        )
        .unwrap();
        let scene = binned_summary(
            &root,
            &["sess".to_string()],
            &serde_json::json!({"metric": "sigma", "source": "timeslice_iso"}),
        );
        assert!(scene.panels.iter().all(|panel| panel.title != "IC1 SX"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn timeslice_chamber_reverses_ic2_when_closer() {
        let root = std::env::temp_dir().join("scan-kit-binned-slice-chamber");
        let _ = std::fs::remove_dir_all(&root);
        let session = root.join("sess");
        std::fs::create_dir_all(session.join("layer-0/run-0")).unwrap();
        std::fs::write(
            session.join("input_map.csv"),
            "energy,layer_id,spot_no,position_x,position_y\n70,1,1,0,0\n",
        )
        .unwrap();
        std::fs::write(session.join("spot_data.csv"), "ic1_total_dose_spot\n1\n").unwrap();
        std::fs::write(
            session.join("layer-0/run-0/timeslice_data_device_units.csv"),
            "layer_id,spot_no,rci_in_trigger,r_ic1_x_position,r_ic2_x_position\n1,1,1,70,60\n",
        )
        .unwrap();
        let scene = binned_summary(
            &root,
            &["sess".to_string()],
            &serde_json::json!({"metric": "ic12_pos_diff", "source": "timeslice_chamber"}),
        );
        let panel = scene
            .panels
            .iter()
            .find(|panel| panel.title == "DX")
            .unwrap();
        assert!(rect_ys(panel)
            .iter()
            .any(|value| (*value + 2.0).abs() < 1e-2));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn spot_timeslice_switch_follows_the_metric_list() {
        let spot = binned_summary(
            std::path::Path::new("."),
            &[],
            &serde_json::json!({"metric": "Dose Error (%)"}),
        );
        let source = spot
            .controls
            .iter()
            .find(|control| control.id == "source")
            .unwrap();
        assert_eq!(
            source.options,
            vec!["Spot".to_string(), "Timeslice".to_string()]
        );
        assert_eq!(source.value, "Spot");
        assert!(spot.controls.iter().all(|control| control.id != "frame"));
        let metric = spot
            .controls
            .iter()
            .find(|control| control.id == "metric")
            .unwrap();
        assert_eq!(metric.value, "Dose Error (%)");
        assert!(metric
            .options
            .iter()
            .any(|option| option == "Position Error (mm) (Isocenter)"));
        assert!(metric
            .options
            .iter()
            .any(|option| option == "Position Error (mm) (Chamber)"));
        assert!(metric
            .options
            .iter()
            .all(|option| option != "IC Current (nA)"));
        assert!(metric
            .options
            .iter()
            .all(|option| option != "Fit Confidence"));
        let bins = spot
            .controls
            .iter()
            .find(|control| control.id == "bins")
            .unwrap();
        assert_eq!(bins.value, "Automatic");
        assert_eq!(
            bins.options,
            vec![
                "Automatic".to_string(),
                "8".to_string(),
                "16".to_string(),
                "32".to_string(),
                "64".to_string()
            ]
        );
        let trend = spot
            .controls
            .iter()
            .find(|control| control.id == "trend")
            .unwrap();
        assert_eq!(trend.value, "Off");
        assert_eq!(
            trend.options,
            vec![
                "Off".to_string(),
                "Linear".to_string(),
                "Polynomial".to_string()
            ]
        );
        assert!(spot.controls.iter().all(|control| control.id != "fliers"));
        assert!(spot.controls.iter().all(|control| control.id != "cutoff"));
        let position = binned_summary(
            std::path::Path::new("."),
            &[],
            &serde_json::json!({"metric": "position_error"}),
        );
        let interlock = position
            .controls
            .iter()
            .find(|control| control.id == "interlock")
            .unwrap();
        assert_eq!(interlock.label, "Interlock Thresholds");
        assert_eq!(interlock.value, "Off");
        assert_eq!(interlock.options, vec!["Off".to_string(), "On".to_string()]);

        let switched = binned_summary(
            std::path::Path::new("."),
            &[],
            &serde_json::json!({"metric": "Dose Error (%)", "source": "Timeslice"}),
        );
        let metric = switched
            .controls
            .iter()
            .find(|control| control.id == "metric")
            .unwrap();
        assert_eq!(metric.value, "Current Ratios (%)");
        for label in [
            "Fit Confidence",
            "Peak Amplitude",
            "Amplifier Error (V)",
            "Probe Field (G)",
            "Position Error (mm)",
        ] {
            assert!(
                metric.options.iter().any(|option| option == label),
                "{label}"
            );
        }
        let fit = binned_summary(
            std::path::Path::new("."),
            &[],
            &serde_json::json!({"metric": "Fit Confidence", "source": "Timeslice", "x": "Target MU"}),
        );
        let x = fit
            .controls
            .iter()
            .find(|control| control.id == "x")
            .unwrap();
        assert_eq!(x.value, "Energy");
        assert_eq!(x.options, vec!["Energy".to_string()]);
        assert!(metric
            .options
            .iter()
            .any(|option| option == "IC2-IC1 Position (mm) (Chamber)"));
        assert!(metric
            .options
            .iter()
            .all(|option| option != "Dose Error (%)"));
        assert!(metric
            .options
            .iter()
            .all(|option| option != "Position Error (mm) (Chamber)"));

        let position = binned_summary(
            std::path::Path::new("."),
            &[],
            &serde_json::json!({"metric": "Position Error (mm) (Chamber)", "source": "Spot"}),
        );
        assert_eq!(
            position
                .controls
                .iter()
                .find(|control| control.id == "metric")
                .unwrap()
                .value,
            "Position Error (mm) (Chamber)"
        );
        assert!(position
            .controls
            .iter()
            .all(|control| control.id != "frame"));
        let slice = binned_summary(
            std::path::Path::new("."),
            &[],
            &serde_json::json!({"metric": "Position Error (mm)", "source": "Timeslice", "frame": "Chamber"}),
        );
        assert_eq!(
            slice
                .controls
                .iter()
                .find(|control| control.id == "metric")
                .unwrap()
                .value,
            "Position Error (mm)"
        );
        assert!(slice.controls.iter().any(|control| control.id == "bins"));
        assert!(slice.controls.iter().all(|control| control.id != "frame"));

        let current = binned_summary(
            std::path::Path::new("."),
            &[],
            &serde_json::json!({"metric": "IC Current (nA)"}),
        );
        assert_eq!(
            current
                .controls
                .iter()
                .find(|control| control.id == "source")
                .unwrap()
                .value,
            "Timeslice"
        );
    }

    #[test]
    fn histograms_keep_each_session_range() {
        let root = std::env::temp_dir().join("scan-kit-binned-hist");
        let _ = std::fs::remove_dir_all(&root);
        for (id, dose) in [("a", "1"), ("b", "2")] {
            let session = root.join(id);
            std::fs::create_dir_all(&session).unwrap();
            std::fs::write(session.join("input_map.csv"), "energy,charge_req\n70,1\n").unwrap();
            std::fs::write(
                session.join("spot_data.csv"),
                format!("ic1_total_dose_spot\n{dose}\n"),
            )
            .unwrap();
        }
        let ids = ["a".to_string(), "b".to_string()];
        let own = binned_summary(&root, &ids, &serde_json::json!({"hist": "On"}));
        let shared = binned_summary(
            &root,
            &ids,
            &serde_json::json!({"hist": "On", "shared": "On"}),
        );
        let hist = own
            .panels
            .iter()
            .find(|panel| panel.y_label == "Probability (%)")
            .unwrap();
        assert!(hist.title.is_empty());
        assert!(hist
            .series
            .iter()
            .any(|series| matches!(series, Series::Polyline { .. })));
        let own_edges = hist_edges(&own);
        let shared_edges = hist_edges(&shared);
        assert!(own_edges[0].last().unwrap() < &10.0);
        assert!(own_edges[1].last().unwrap() > &50.0);
        assert!(shared_edges[0].last().unwrap() > &50.0);
        assert!(shared_edges[1].last().unwrap() > &50.0);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn automatic_bins_keep_a_short_axis_and_a_fixed_count_regroups() {
        let column = |values: &[f32]| {
            let mut table = BTreeMap::new();
            table.insert("x".to_string(), values.to_vec());
            vec![table]
        };
        let short = column(&[1.0, 1.0, 2.0, 5.0]);
        let mut tables = short;
        let levels = assign_x(&mut tables, "x", &BinChoice::Automatic);
        assert_eq!(levels, vec![1.0, 2.0, 5.0]);
        assert_eq!(tables[0]["_bin"], vec![1.0, 1.0, 2.0, 5.0]);

        let many: Vec<f32> = (0..100).map(|value| value as f32).collect();
        let mut tables = column(&many);
        let grouped = assign_x(&mut tables, "x", &BinChoice::Automatic);
        assert!(
            grouped.len() <= 32,
            "a long axis falls back to quantile bins"
        );
        assert!(grouped.len() > 1);

        let mut tables = column(&[10.0, 20.0, 30.0, 40.0, 50.0]);
        let fixed = assign_x(&mut tables, "x", &BinChoice::Fixed(2));
        assert!(fixed.len() <= 2);
    }

    #[test]
    fn trend_line_crosses_the_axis_and_a_polynomial_bends() {
        let mut table = BTreeMap::new();
        table.insert("_bin".to_string(), vec![20.0, 20.0, 30.0, 30.0]);
        table.insert("y".to_string(), vec![2.0, 2.0, 4.0, 4.0]);
        let line = binned_trend(
            &table,
            &table["y"],
            &[10.0, 20.0, 30.0],
            0,
            -0.6,
            2.6,
            Trend::Linear,
        )
        .unwrap();
        let Series::Polyline { xs, .. } = &line else {
            panic!("trend is a polyline");
        };
        assert!((xs[0] + 0.6).abs() < 1e-4, "left edge, got {}", xs[0]);
        assert!((xs.last().unwrap() - 2.6).abs() < 1e-4, "right edge");

        let mut curved = BTreeMap::new();
        curved.insert("_bin".to_string(), vec![0.0, 1.0, 2.0]);
        curved.insert("y".to_string(), vec![0.0, 1.0, 0.0]);
        let y_at_middle = |mode: Trend| {
            let line =
                binned_trend(&curved, &curved["y"], &[0.0, 1.0, 2.0], 0, -0.6, 2.6, mode).unwrap();
            let Series::Polyline { xs, ys, .. } = line else {
                panic!("trend is a polyline");
            };
            let middle = 1.0 + (0.0 - 0.5) * super::OFFSET;
            let (mut best, mut y) = (f32::MAX, 0.0);
            for (x, value) in xs.iter().zip(&ys) {
                let distance = (x - middle).abs();
                if distance < best {
                    best = distance;
                    y = *value;
                }
            }
            y
        };
        assert!(y_at_middle(Trend::Linear) < 0.5);
        assert!(y_at_middle(Trend::Polynomial) > 0.5);
    }

    #[test]
    fn filters_keep_the_rows_their_names_describe() {
        let values: Vec<f32> = (0..20).map(|value| value as f32).chain([1000.0]).collect();
        let mut beam = vec![1.0; values.len()];
        beam[0] = 0.0;
        let fresh = || {
            let mut table = BTreeMap::new();
            table.insert("y".to_string(), values.clone());
            table.insert("beam_on".to_string(), beam.clone());
            table
        };
        let finite = |table: &BTreeMap<String, Vec<f32>>| {
            table["y"]
                .iter()
                .copied()
                .filter(|value| value.is_finite())
                .collect::<Vec<_>>()
        };

        let mut upper = fresh();
        apply_filter(&mut upper, &["y"], "upper_95", "beam_both");
        assert_eq!(finite(&upper), vec![1000.0]);

        let mut lower = fresh();
        apply_filter(&mut lower, &["y"], "lower_95", "beam_both");
        let kept = finite(&lower);
        assert!(kept.contains(&1.0));
        assert!(!kept.contains(&1000.0));

        let mut mad = fresh();
        apply_filter(&mut mad, &["y"], "mad_outliers", "beam_both");
        assert_eq!(finite(&mad), vec![1000.0]);

        let mut off = fresh();
        apply_filter(&mut off, &["y"], "all", "beam_off");
        assert_eq!(finite(&off), vec![0.0]);

        let mut on = fresh();
        apply_filter(&mut on, &["y"], "all", "beam_on");
        let kept = finite(&on);
        assert!(!kept.contains(&0.0));
        assert!(kept.contains(&1000.0));

        let dose = binned_summary(
            std::path::Path::new("."),
            &[],
            &serde_json::json!({"metric": "dose_error"}),
        );
        let beam = dose
            .controls
            .iter()
            .find(|control| control.id == "beam")
            .unwrap();
        assert_eq!(beam.value, "Both");
        assert_eq!(
            beam.options,
            vec![
                "Beam On".to_string(),
                "Beam Off".to_string(),
                "Both".to_string()
            ]
        );
        let domain = dose
            .controls
            .iter()
            .find(|control| control.id == "domain")
            .unwrap();
        assert_eq!(
            domain.options,
            vec![
                "All".to_string(),
                "Lower 95%".to_string(),
                "Upper 5%".to_string(),
                "MAD Outliers".to_string()
            ]
        );
        let current = binned_summary(
            std::path::Path::new("."),
            &[],
            &serde_json::json!({"metric": "ic_current"}),
        );
        let current_beam = current
            .controls
            .iter()
            .find(|control| control.id == "beam")
            .unwrap();
        assert_eq!(current_beam.value, "Beam On");
        let rate = binned_summary(
            std::path::Path::new("."),
            &[],
            &serde_json::json!({"metric": "dose_rate"}),
        );
        assert!(rate
            .controls
            .iter()
            .all(|control| control.id != "beam" && control.id != "domain"));
    }

    fn hist_edges(scene: &scan_kit_core::PlotScene) -> Vec<Vec<f32>> {
        scene
            .panels
            .iter()
            .find(|panel| panel.y_label == "Probability (%)")
            .unwrap()
            .series
            .iter()
            .filter_map(|series| match series {
                Series::Bars { edges, .. } => Some(edges.clone()),
                _ => None,
            })
            .collect()
    }

    fn rect_ys(panel: &scan_kit_core::Panel) -> Vec<f32> {
        panel
            .series
            .iter()
            .find_map(|series| match series {
                Series::Rects { y, .. } => Some(y.clone()),
                Series::Triangles { ys, .. } => Some(ys.clone()),
                _ => None,
            })
            .unwrap_or_default()
    }

    #[test]
    fn timeslice_signals_keep_low_confidence_and_tracking_error() {
        let device = b"\
rci_in_trigger,r_ic1_x_confidence,r_ic1_x_peak_amplitude,c_x,r_xV,r_tx2_probe_x
1,90,4,1.0,1.2,30
0,10,1,1.0,1.5,-10
1,40,2,2.0,2.0,5
";
        let other = b"rci_in_trigger,r_ic1_current\n1,3\n";
        let files = vec![device.to_vec(), other.to_vec()];
        let energies = [150.0];
        let layers = [0, 1];
        let close = |got: &[f32], want: &[f32]| {
            assert_eq!(got.len(), want.len());
            for (got, want) in got.iter().zip(want) {
                assert!((got - want).abs() < 1e-4, "{got} vs {want}");
            }
        };

        let confidence = signal_table(&files, &energies, &layers, "fit_confidence");
        close(&confidence["ic1_x_confidence"], &[90.0, 10.0, 40.0]);
        close(&confidence["energy"], &[150.0, 150.0, 150.0]);
        close(&confidence["beam_on"], &[1.0, 0.0, 1.0]);
        assert!(!confidence.contains_key("ic1_y_confidence"));

        let mut filtered = confidence;
        apply_filter(&mut filtered, &["ic1_x_confidence"], "all", "beam_on");
        assert!(filtered["ic1_x_confidence"][0].is_finite());
        assert!(filtered["ic1_x_confidence"][1].is_nan());
        assert!((filtered["ic1_x_confidence"][2] - 40.0).abs() < 1e-4);

        let peak = signal_table(&files, &energies, &layers, "peak_amplitude");
        close(&peak["ic1_x_peak"], &[4.0, 1.0, 2.0]);
        assert!(!peak.contains_key("ic1_y_peak"));

        let amplifier = signal_table(&files, &energies, &layers, "amplifier_error");
        close(&amplifier["amp_x"], &[0.2, 0.5, 0.0]);
        assert!(!amplifier.contains_key("amp_y"));

        let field = signal_table(&files, &energies, &layers, "probe_field");
        close(&field["field_x"], &[30.0, -10.0, 5.0]);
        assert!(!field.contains_key("field_y"));
    }

    #[test]
    fn axis_label_includes_units_and_side_plots_are_wider() {
        assert_eq!(axis_label("IC1", "IC Current (nA)"), "IC1 (nA)");
        assert_eq!(axis_label("IC1 (%)", "Dose Error (%)"), "IC1 (%)");
        assert_eq!(axis_label("MU/S", "Dose Rate (MU/s)"), "MU/S");
        assert_eq!(axis_label("TOTAL MS", "Spot Delivery Time"), "TOTAL MS");
        assert_eq!(axis_label("IC1 X", "Fit Confidence"), "IC1 X");
        assert_eq!(axis_label("X", "Amplifier Error (V)"), "X (V)");
        assert_eq!(axis_label("Y", "Probe Field (G)"), "Y (G)");
        let scene = binned_summary(
            std::path::Path::new("."),
            &[],
            &serde_json::json!({"hist": "On", "corr": "On"}),
        );
        assert_eq!(scene.column_weights, vec![8.0, 1.65, 1.65]);
    }

    #[test]
    fn contour_fill_uses_the_isoline_vertices() {
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for i in 0..50 {
            for j in 0..50 {
                let x = (i as f32 - 25.0) * 0.15;
                let y = (j as f32 - 25.0) * 0.15;
                let weight = (-0.5 * (x * x + y * y) / 0.8).exp();
                for _ in 0..(weight * 8.0) as i32 {
                    xs.push(x);
                    ys.push(y);
                }
            }
        }
        let drawn = contour_bands(&xs, &ys, 5.0);
        let Series::Triangles {
            xs: mesh_x,
            ys: mesh_y,
            color,
        } = &drawn[0]
        else {
            panic!("contour fill should be a triangle mesh");
        };
        assert!((color[3] - 0.13).abs() < 1e-6);
        assert_eq!(mesh_x.len() % 3, 0);
        assert!(mesh_x.len() > 12);
        let Series::Polyline {
            xs: line_x,
            ys: line_y,
            ..
        } = &drawn[1]
        else {
            panic!("contour should keep its isolines");
        };
        let mut shared = 0;
        for (x, y) in line_x.iter().zip(line_y) {
            if !x.is_finite() {
                continue;
            }
            assert!(
                mesh_x
                    .iter()
                    .zip(mesh_y)
                    .any(|(mx, my)| { (mx - x).abs() < 1e-4 && (my - y).abs() < 1e-4 }),
                "isoline point {x},{y} is missing from the fill"
            );
            shared += 1;
        }
        assert!(shared > 8);
    }

    #[test]
    fn violin_is_a_filled_curve_with_an_outline() {
        let mut table = BTreeMap::new();
        let y: Vec<f32> = (0..40).map(|i| (i as f32 - 20.0) * 0.2).collect();
        table.insert("_bin".to_string(), vec![1.0; y.len()]);
        let drawn = violin_series(&table, &y, &[1.0], 0, [0.2, 0.4, 0.8, 0.55]);
        let Series::Triangles { xs, ys, color } = &drawn[0] else {
            panic!("violin fill should be triangles");
        };
        assert!((color[3] - 0.55).abs() < 1e-6);
        assert_eq!(xs.len() % 3, 0);
        assert_eq!(xs.len(), ys.len());
        assert!(xs.len() > 12);
        let Series::Polyline {
            xs: outline,
            ys: outline_y,
            color: edge,
            thickness,
        } = &drawn[1]
        else {
            panic!("violin should keep a solid outline");
        };
        assert!((*thickness - 1.0).abs() < 1e-6);
        assert_eq!(edge[3], 0.0);
        let top = outline_y
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .fold(f32::MIN, f32::max);
        let mut cap: Vec<f32> = outline
            .iter()
            .zip(outline_y)
            .filter(|(x, y)| x.is_finite() && (**y - top).abs() < 1e-4)
            .map(|(x, _)| *x)
            .collect();
        cap.sort_by(|a, b| a.partial_cmp(b).unwrap());
        cap.dedup_by(|a, b| (*a - *b).abs() < 1e-4);
        assert!(
            cap.len() >= 2,
            "the outline should reach both corners of the top cap"
        );
        let mut widths: Vec<f32> = outline
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .map(f32::abs)
            .collect();
        widths.sort_by(|a, b| a.partial_cmp(b).unwrap());
        widths.dedup_by(|a, b| (*a - *b).abs() < 1e-4);
        assert!(
            widths.len() > 4,
            "outline widths should follow the density, not a rectangle"
        );
    }
}
