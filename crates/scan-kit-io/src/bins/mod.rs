//! Bins, matching the 1.8 metric groups, binning, and glyphs.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;

use scan_kit_core::{
    assign_bin_centers, quantile_edges, row_mask, scrub_control, segments_control, segments_from,
    segments_json, time_end, BeamGate, Control, Panel, PlotScene, Rank, Segment, Series,
};
use serde_json::Value;

use super::discover;
use super::histogram::{bin_button, bin_share, hist_bin_count, share_key, BinShare, BIN_CHOICES};
use super::marks::{contour_bands, control, flag, labeled, pick, text};
mod glyphs;

use super::tables::{median, same, session_columns, span, timeslice_energy_only, Grain};
use glyphs::{
    begin_grown, binned_trend, box_series, contour_series, corr_panel, grown_violin, hist_panel,
    hline, interlock_guides, mean_series, note_panel, scatter_series, scatter_trend, violin_series,
};

const OFFSET: f32 = 0.35;
const BOX_WIDTH: f32 = 0.3;
const VIOLIN_WIDTH: f32 = 0.65;
const GATE_ABS_MU: f64 = 0.002;

pub(crate) struct YSeries {
    pub key: &'static str,
    pub label: &'static str,
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
const POSITION_REL: &[YSeries] = &[
    YSeries {
        key: "ic1_x_err_rel",
        label: "IC1 X",
    },
    YSeries {
        key: "ic1_y_err_rel",
        label: "IC1 Y",
    },
    YSeries {
        key: "ic2_x_err_rel",
        label: "IC2 X",
    },
    YSeries {
        key: "ic2_y_err_rel",
        label: "IC2 Y",
    },
];
const POSITION_R: &[YSeries] = &[
    YSeries {
        key: "ic1_r_err",
        label: "IC1",
    },
    YSeries {
        key: "ic2_r_err",
        label: "IC2",
    },
];
const POSITION_R_REL: &[YSeries] = &[
    YSeries {
        key: "ic1_r_err_rel",
        label: "IC1",
    },
    YSeries {
        key: "ic2_r_err_rel",
        label: "IC2",
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
const SIGMA_ERR_PCT: &[YSeries] = &[
    YSeries {
        key: "ic1_sig_x_err_pct",
        label: "IC1 SX",
    },
    YSeries {
        key: "ic1_sig_y_err_pct",
        label: "IC1 SY",
    },
    YSeries {
        key: "ic2_sig_x_err_pct",
        label: "IC2 SX",
    },
    YSeries {
        key: "ic2_sig_y_err_pct",
        label: "IC2 SY",
    },
];
const DOSE_PER_MU: &[YSeries] = &[
    YSeries {
        key: "ic1_dose_per_mu",
        label: "IC1",
    },
    YSeries {
        key: "ic2_dose_per_mu",
        label: "IC2",
    },
    YSeries {
        key: "ic3_dose_per_mu",
        label: "IC3",
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
        id: "dose_per_mu",
        label: "Dose per MU",
        series: DOSE_PER_MU,
        zero: false,
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
        id: "position_error_rel",
        label: "Relative Position Error (mm)",
        series: POSITION_REL,
        zero: true,
        filter: true,
        timeslice: false,
    },
    YGroup {
        id: "position_radius",
        label: "Position Radius (mm)",
        series: POSITION_R,
        zero: true,
        filter: true,
        timeslice: false,
    },
    YGroup {
        id: "position_radius_rel",
        label: "Relative Position Radius (mm)",
        series: POSITION_R_REL,
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
        id: "sigma_error_pct",
        label: "Sigma Error (%)",
        series: SIGMA_ERR_PCT,
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
/// Series of one Y quantity, in the same order Bins draws them.
pub(crate) fn channels_for(metric: &str) -> (&'static str, &'static [YSeries]) {
    match GROUPS.iter().find(|group| group.id == metric) {
        Some(group) => (group.label, group.series),
        None => ("", &[]),
    }
}

fn sources_for(metric: &str) -> &'static [(&'static str, &'static str)] {
    match metric {
        "current_ratio" | "ic_current" | "fit_confidence" | "peak_amplitude"
        | "amplifier_error" | "probe_field" => &SOURCE_CHOICES[2..3],
        "position_error"
        | "position_error_rel"
        | "position_radius"
        | "position_radius_rel"
        | "sigma"
        | "sigma_error"
        | "sigma_error_pct" => &SOURCE_CHOICES[..3],
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
    end: f32,
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
    // A partial slice must not be stored as the finished session.
    if crate::tables::slice_is_bound() {
        let mut built = build();
        built.key = key;
        return built;
    }
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
    segments: &str,
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
        "{}|{stamp}|{metric}|{source}|{x_column}|{}|{glyph}|{segments}|{cutoff}",
        root.display(),
        bin_label(bins),
    )
}

fn session_stamp(root: &Path, session: &str) -> u128 {
    let dir = discover::session_directory(root, session);
    let mut stamp = mtime_ns(&dir);
    for name in ["spot_data.csv", "input_map.csv"] {
        stamp ^= discover::meta_stamp(&dir.join(name));
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

pub(crate) fn bins(root: &Path, session_ids: &[String], options: &Value) -> PlotScene {
    let timeslice = crate::source::wants_timeslice(crate::source::Shape::YAndX, options);
    let owned: Vec<(String, Vec<String>)> = session_ids
        .iter()
        .map(|id| {
            (
                id.clone(),
                super::tables::grain_columns(root, id, timeslice),
            )
        })
        .collect();
    let headers: Vec<crate::source::SessionCols<'_>> = owned
        .iter()
        .map(|(name, columns)| crate::source::SessionCols { name, columns })
        .collect();
    let picked = crate::source::select(crate::source::Shape::YAndX, true, true, &headers, options);
    let crate::source::Picked {
        y,
        frame,
        x: x_id,
        grain: coarse,
        controls: source_controls,
        ..
    } = picked;
    let metric = y.as_str();
    let group = GROUPS
        .iter()
        .find(|group| group.id == metric)
        .unwrap_or(&GROUPS[0]);
    let glyph = pick(options, "glyph", "violin", GLYPH_CHOICES);
    let segments = segments_from(
        options,
        &[
            Segment::Beam {
                state: BeamGate::Both,
            },
            Segment::Rank { which: Rank::All },
        ],
    );
    let trend = trend_mode(options);
    let hist = flag(options, "hist", false);
    let corr = flag(options, "corr", false);
    let mut interlock = flag(options, "interlock", false);
    let bins = bin_choice(options);
    let hist_label = bin_button(text(options, "hist_bins", "Auto"));
    let hist_bins = hist_bin_count(&hist_label);
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
    let share = bin_share(text(options, "share", ""), flag(options, "shared", false));
    let geometry = matches!(source, "timeslice_iso" | "timeslice_chamber")
        && matches!(
            group.id,
            "position_error"
                | "position_error_rel"
                | "position_radius"
                | "position_radius_rel"
                | "sigma"
                | "sigma_error"
                | "sigma_error_pct"
                | "ic12_pos_diff"
        );
    let energy_only = geometry || timeslice_energy_only(group.id);
    let chamber = source == "chamber"
        && matches!(
            group.id,
            "position_error"
                | "position_error_rel"
                | "position_radius"
                | "position_radius_rel"
                | "sigma"
                | "sigma_error"
                | "sigma_error_pct"
                | "ic12_pos_diff"
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
        &segments_json(&segments),
        cutoff,
    );
    let grow_key = key.clone();
    let grow_ok = segments.iter().all(|item| {
        !matches!(
            item,
            Segment::Rank {
                which: Rank::Lower95 | Rank::Upper95 | Rank::Mad
            }
        )
    });
    let prepared = cached_prepared(key, || {
        let load_one = |session: &String| {
            let grain = if geometry {
                if source == "timeslice_chamber" {
                    Grain::SampleChamber
                } else {
                    Grain::Sample
                }
            } else if group.id == "current_ratio" {
                Grain::Layer
            } else if group.timeslice {
                Grain::Sample
            } else if chamber {
                Grain::SpotChamber
            } else {
                Grain::Spot
            };
            let mut names: Vec<&str> = group.series.iter().map(|series| series.key).collect();
            names.push(x_column);
            let loaded = session_columns(root, session, grain, &names);
            let clock = time_end(loaded.get("time_s").map(Vec::as_slice));
            let loaded = if group.id == "dose_rate" {
                std::sync::Arc::new(dose_rate_table(loaded.as_ref()))
            } else {
                loaded
            };
            (
                plotted_columns(loaded.as_ref(), &names, &segments, group.filter),
                clock,
            )
        };
        // A handful of sessions, one thread each. The spot cache covers a repeat open.
        let pairs = if session_ids.len() < 2 {
            session_ids.iter().map(load_one).collect::<Vec<_>>()
        } else {
            std::thread::scope(|scope| {
                let mut joins = Vec::with_capacity(session_ids.len());
                for session in session_ids {
                    joins.push(scope.spawn(|| load_one(session)));
                }
                joins
                    .into_iter()
                    .map(|join| join.join().unwrap())
                    .collect::<Vec<_>>()
            })
        };
        let mut end = 0.0f32;
        let mut tables = Vec::with_capacity(pairs.len());
        for (table, clock) in pairs {
            end = end.max(clock);
            tables.push(table);
        }
        let (categories, stable_bins) = if raw_x {
            (Vec::new(), false)
        } else {
            assign_x(&mut tables, x_column, &bins)
        };
        let grow = grow_ok && stable_bins && crate::tables::slice_is_bound();
        if grow {
            begin_grown(&grow_key);
        }
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
                let mut lo = f32::MAX;
                let mut hi = f32::MIN;
                let mut any = false;
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
                        for value in y {
                            if value.is_finite() {
                                any = true;
                                lo = lo.min(*value);
                                hi = hi.max(*value);
                            }
                        }
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
                            _ if grow => grown_violin(
                                &format!("{grow_key}|{}|{index}", series.key),
                                table.get("_bin").map(Vec::as_slice).unwrap_or(&[]),
                                y,
                                &categories,
                                color,
                            ),
                            _ => violin_series(table, y, &categories, index, color),
                        }
                    })
                    .collect();
                if group.zero {
                    any = true;
                    lo = lo.min(0.0);
                    hi = hi.max(0.0);
                }
                let (ymin, ymax) = if any { span(&[lo, hi]) } else { span(&[]) };
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
            end,
        }
    });

    let pairs = if corr && !prepared.series.is_empty() {
        let shown: Vec<&YSeries> = prepared
            .series
            .iter()
            .filter_map(|item| group.series.iter().find(|series| series.key == item.key))
            .collect();
        correlation_slots(group.id, &shown)
    } else {
        Vec::new()
    };
    let mut panels = Vec::new();
    if prepared.series.is_empty() {
        panels.push(note_panel("No finite values for this metric"));
    } else {
        let page = if share == BinShare::Page {
            let keys: Vec<&str> = prepared
                .series
                .iter()
                .map(|item| item.key.as_str())
                .collect();
            page_span(&prepared.tables, &keys)
        } else {
            None
        };
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
                    share == BinShare::Plot,
                    page,
                    interlock
                        && matches!(
                            group.id,
                            "position_error"
                                | "position_error_rel"
                                | "position_radius"
                                | "position_radius_rel"
                        ),
                ));
            }
            if let Some((left, right)) = pairs.get(index) {
                panels.push(corr_panel(left, right, &prepared.tables, group.label));
            }
        }
    }

    let draw_corr = corr && (prepared.series.is_empty() || !pairs.is_empty());
    let side = u32::from(hist) + u32::from(draw_corr);
    let mut weights = vec![8.0];
    weights.extend(std::iter::repeat_n(1.65, side as usize));
    PlotScene {
        title: format!("{} vs {}", group.label, x_label(x_column)),
        panels,
        controls: {
            let mut controls = controls(
                source_controls,
                group.id,
                x_id,
                glyph,
                &segments,
                trend,
                hist,
                corr,
                interlock,
                &bin_label(&bins),
                &hist_label,
                share,
                cutoff,
            );
            controls.push(scrub_control(
                options,
                prepared.end,
                &crate::tables::timeline_layers(root, session_ids, coarse == "timeslice"),
            ));
            controls
        },
        table: None,
        columns: 1 + side,
        column_weights: weights,
        row_weights: Vec::new(),
        side: 0,
    }
}

/// Pairs for the correlation column, one per plotted series.
///
/// A channel quad (IC1/IC2 by X/Y) asks four questions: the two chambers on X,
/// the two chambers on Y, then X against Y inside each chamber. A two-axis
/// source asks that one question. Anything else pairs each series with the next.
/// A source with one series has nothing to correlate. Missing channels drop the
/// pairs that need them, and the remaining pairs fill the rows.
fn correlation_slots<'a>(metric: &str, present: &[&'a YSeries]) -> Vec<(&'a YSeries, &'a YSeries)> {
    let preferred = match metric {
        "position_error" => &[
            ("ic1_x_err", "ic2_x_err"),
            ("ic1_y_err", "ic2_y_err"),
            ("ic1_x_err", "ic1_y_err"),
            ("ic2_x_err", "ic2_y_err"),
        ][..],
        "position_error_rel" => &[
            ("ic1_x_err_rel", "ic2_x_err_rel"),
            ("ic1_y_err_rel", "ic2_y_err_rel"),
            ("ic1_x_err_rel", "ic1_y_err_rel"),
            ("ic2_x_err_rel", "ic2_y_err_rel"),
        ][..],
        "sigma" => &[
            ("ic1_sig_x", "ic2_sig_x"),
            ("ic1_sig_y", "ic2_sig_y"),
            ("ic1_sig_x", "ic1_sig_y"),
            ("ic2_sig_x", "ic2_sig_y"),
        ][..],
        "sigma_error" => &[
            ("ic1_sig_x_err", "ic2_sig_x_err"),
            ("ic1_sig_y_err", "ic2_sig_y_err"),
            ("ic1_sig_x_err", "ic1_sig_y_err"),
            ("ic2_sig_x_err", "ic2_sig_y_err"),
        ][..],
        "sigma_error_pct" => &[
            ("ic1_sig_x_err_pct", "ic2_sig_x_err_pct"),
            ("ic1_sig_y_err_pct", "ic2_sig_y_err_pct"),
            ("ic1_sig_x_err_pct", "ic1_sig_y_err_pct"),
            ("ic2_sig_x_err_pct", "ic2_sig_y_err_pct"),
        ][..],
        "fit_confidence" => &[
            ("ic1_x_confidence", "ic2_x_confidence"),
            ("ic1_y_confidence", "ic2_y_confidence"),
            ("ic1_x_confidence", "ic1_y_confidence"),
            ("ic2_x_confidence", "ic2_y_confidence"),
        ][..],
        "peak_amplitude" => &[
            ("ic1_x_peak", "ic2_x_peak"),
            ("ic1_y_peak", "ic2_y_peak"),
            ("ic1_x_peak", "ic1_y_peak"),
            ("ic2_x_peak", "ic2_y_peak"),
        ][..],
        "amplifier_error" => &[("amp_x", "amp_y")][..],
        "probe_field" => &[("field_x", "field_y")][..],
        "ic12_pos_diff" => &[("ic12_x_diff", "ic12_y_diff")][..],
        _ => &[][..],
    };
    let mut available = Vec::new();
    for (left, right) in preferred {
        let Some(a) = present.iter().copied().find(|series| series.key == *left) else {
            continue;
        };
        let Some(b) = present.iter().copied().find(|series| series.key == *right) else {
            continue;
        };
        available.push((a, b));
    }
    if available.is_empty() && present.len() >= 2 {
        for (index, series) in present.iter().copied().enumerate() {
            let other = present[(index + 1) % present.len()];
            available.push((series, other));
        }
    }
    if available.is_empty() {
        return Vec::new();
    }
    (0..present.len())
        .map(|index| available[index % available.len()])
        .collect()
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
        x_label: String::new(),
        xmin: prepared.xmin,
        xmax: prepared.xmax,
        ymin: item.ymin,
        ymax: item.ymax,
        series: drawn,
        x_labels: prepared.x_labels.clone(),
        equal: false,
    }
}

pub(crate) fn axis_label(series: &str, group: &str) -> String {
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
    mut controls: Vec<Control>,
    metric: &str,
    x: &str,
    glyph: &str,
    segments: &[Segment],
    trend: Trend,
    hist: bool,
    corr: bool,
    interlock: bool,
    bins: &str,
    hist_bins: &str,
    share: BinShare,
    cutoff: f32,
) -> Vec<Control> {
    let on = |value: bool| if value { "On" } else { "Off" };
    controls.push(control("bins", "Bins", BIN_CHOICES, bins).grouped("Data Source"));
    controls.push(labeled("glyph", "Glyph", GLYPH_CHOICES, glyph).grouped("Plot Style"));
    controls.push(
        control(
            "trend",
            "Trend",
            &["Off", "Linear", "Polynomial"],
            trend_value(trend),
        )
        .grouped("Plot Style"),
    );
    controls.push(
        control(
            "interlock",
            "Interlock Thresholds",
            &["Off", "On"],
            on(interlock),
        )
        .grouped("Plot Style"),
    );
    controls.push(
        control(
            "cutoff",
            "Contour Cutoff",
            &["0", "5", "10", "20"],
            &cutoff.round().to_string(),
        )
        .grouped("Plot Style"),
    );
    controls.push(
        control("hist", "Show Panel", &["Off", "On"], on(hist))
            .grouped("Histogram")
            .checked(),
    );
    controls.push(control("hist_bins", "Bins", BIN_CHOICES, hist_bins).grouped("Histogram"));
    controls.push(
        labeled(
            "share",
            "Share",
            &[("own", "Own"), ("plot", "Plot"), ("page", "Page")],
            share_key(share),
        )
        .grouped("Histogram"),
    );
    controls.push(
        control("corr", "Show Panel", &["Off", "On"], on(corr))
            .grouped("Correlation")
            .checked(),
    );
    let filters = GROUPS
        .iter()
        .any(|group| group.id == metric && group.filter);
    if filters {
        controls.push(segments_control(
            segments,
            &[("beam", "Beam"), ("rank", "Rank")],
        ));
    }
    controls.retain(|item| match item.id.as_str() {
        "cutoff" => glyph == "contour",
        "interlock" => interlock_ok(metric, x, glyph),
        _ => true,
    });
    controls
}

fn interlock_ok(metric: &str, x: &str, glyph: &str) -> bool {
    match metric {
        "position_error" | "position_error_rel" | "position_radius" | "position_radius_rel" => true,
        "dose_error" => x == "target_mu",
        "sigma" | "sigma_error" | "sigma_error_pct" => {
            x == "energy" && !matches!(glyph, "scatter" | "contour")
        }
        _ => false,
    }
}

const AUTO_LEVELS: usize = 100;
const AUTO_QUANTILES: usize = 32;

enum BinChoice {
    Automatic,
    Fixed(usize),
}

fn bin_choice(options: &Value) -> BinChoice {
    match crate::histogram::count_choice(text(options, "bins", "Auto")) {
        None => BinChoice::Automatic,
        Some(count) => BinChoice::Fixed(count),
    }
}

fn bin_label(choice: &BinChoice) -> String {
    match choice {
        BinChoice::Automatic => "Auto".to_string(),
        BinChoice::Fixed(count) => count.to_string(),
    }
}

fn page_span(tables: &[BTreeMap<String, Vec<f32>>], keys: &[&str]) -> Option<(f32, f32)> {
    let mut lo = f32::MAX;
    let mut hi = f32::MIN;
    for key in keys {
        for table in tables {
            let Some(column) = table.get(*key) else {
                continue;
            };
            for value in column {
                if value.is_finite() {
                    lo = lo.min(*value);
                    hi = hi.max(*value);
                }
            }
        }
    }
    (lo <= hi).then_some((lo, hi))
}

/// Sorted levels when fewer than `limit` exist. `None` means the column is continuous.
fn quantized_levels(
    tables: &[BTreeMap<String, Vec<f32>>],
    key: &str,
    limit: usize,
) -> Option<Vec<f32>> {
    let mut levels = Vec::new();
    // Exact repeats are the common case (one energy copied down a layer).
    // `same` still merges a value that is only a tolerance away.
    let mut seen = HashSet::new();
    for table in tables {
        let Some(column) = table.get(key) else {
            continue;
        };
        for value in column {
            if !value.is_finite() || !seen.insert(value.to_bits()) {
                continue;
            }
            if levels.iter().any(|have| same(*have, *value)) {
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

/// Levels, and whether each row's bin is that row's own x value.
///
/// A quantized axis stays put as later rows arrive. Quantile edges move, so a
/// growing picture cannot reuse the previous groups.
fn assign_x(
    tables: &mut [BTreeMap<String, Vec<f32>>],
    column: &str,
    choice: &BinChoice,
) -> (Vec<f32>, bool) {
    let quantiles = match choice {
        BinChoice::Fixed(count) => *count,
        BinChoice::Automatic => match quantized_levels(tables, column, AUTO_LEVELS) {
            Some(levels) => {
                for table in tables.iter_mut() {
                    let values = table.get(column).cloned().unwrap_or_default();
                    table.insert("_bin".to_string(), values);
                }
                return (levels, true);
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
    (unique_values(tables, "_bin"), false)
}

fn unique_values(tables: &[BTreeMap<String, Vec<f32>>], key: &str) -> Vec<f32> {
    let mut values = Vec::new();
    let mut seen = HashSet::new();
    for table in tables {
        if let Some(column) = table.get(key) {
            for value in column {
                if !value.is_finite() || !seen.insert(value.to_bits()) {
                    continue;
                }
                if values.iter().any(|have| same(*have, *value)) {
                    continue;
                }
                values.push(*value);
            }
        }
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    values
}

/// Copy the columns this glyph reads and apply the mask there.
///
/// The session table stays shared. `beam_on`, `expected_sigma`, and
/// `session_avg_rate` stay finite, matching `apply_mask`.
fn plotted_columns(
    loaded: &BTreeMap<String, Vec<f32>>,
    names: &[&str],
    segments: &[Segment],
    filter: bool,
) -> BTreeMap<String, Vec<f32>> {
    let keys: Vec<&str> = names
        .iter()
        .copied()
        .filter(|name| loaded.contains_key(*name))
        .collect();
    let time_only: Vec<Segment> = segments
        .iter()
        .filter(|item| matches!(item, Segment::Range { column, .. } if column == "time_s"))
        .cloned()
        .collect();
    let mask = if filter {
        Some(row_mask(loaded, segments, &keys))
    } else if time_only.is_empty() {
        None
    } else {
        Some(row_mask(loaded, &time_only, &keys))
    };
    let mut want: Vec<&str> = names.to_vec();
    for item in segments {
        match item {
            Segment::Range { column, .. } | Segment::Compare { column, .. } => {
                want.push(column.as_str());
            }
            Segment::Beam { .. } | Segment::Rank { .. } => {}
        }
    }
    want.extend([
        "expected_sigma",
        "session_avg_rate",
        "rate_energy",
        "mu_rate",
        "energy",
        "beam_on",
    ]);
    let mut table = BTreeMap::new();
    for name in want {
        let Some(values) = loaded.get(name) else {
            continue;
        };
        if table.contains_key(name) {
            continue;
        }
        let mut values = values.clone();
        let protect = matches!(name, "beam_on" | "expected_sigma" | "session_avg_rate");
        if let Some(mask) = &mask {
            if !protect {
                for (value, keep) in values.iter_mut().zip(mask.iter()) {
                    if !*keep {
                        *value = f32::NAN;
                    }
                }
            }
        }
        table.insert(name.to_string(), values);
    }
    table
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
    if let Some(time) = table.get("time_s") {
        out.insert("time_s".to_string(), time.clone());
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
    let mut index_of = HashMap::with_capacity(categories.len());
    for (index, category) in categories.iter().enumerate() {
        index_of.insert(category.to_bits(), index);
    }
    for (bin, value) in bins.iter().zip(y) {
        if !bin.is_finite() || !value.is_finite() {
            continue;
        }
        let index = if let Some(index) = index_of.get(&bin.to_bits()).copied() {
            index
        } else {
            let mut index = categories.partition_point(|category| *category < *bin);
            if index == categories.len() || !same(categories[index], *bin) {
                if index > 0 && same(categories[index - 1], *bin) {
                    index -= 1;
                } else {
                    continue;
                }
            }
            index
        };
        groups[index].push(*value);
    }
    groups
}

fn has_finite(values: Option<&Vec<f32>>) -> bool {
    values.is_some_and(|values| values.iter().any(|value| value.is_finite()))
}
#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use scan_kit_core::Series;

    use scan_kit_core::{apply_mask, parse_segments, BeamGate, Rank, Segment};

    use super::{
        assign_x, axis_label, binned_trend, bins, scatter_series, violin_series, BinChoice, Trend,
    };

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
        let scene = bins(
            &root,
            &[session],
            &serde_json::json!({"metric": "Dose Error (%)"}),
        );
        assert!(
            scene
                .panels
                .iter()
                .any(|panel| panel.title.starts_with("IC1")),
            "dose error should plot when the log has padded dose numbers",
        );
    }

    #[test]
    fn scatter_keeps_every_finite_point() {
        let n = 20_001;
        let mut table = BTreeMap::new();
        table.insert("x".into(), (0..n).map(|index| index as f32).collect());
        let y: Vec<f32> = (0..n)
            .map(|index| if index == n - 1 { 7.0 } else { 1.0 })
            .collect();
        let Series::Points { xs, ys, .. } = scatter_series(&table, &y, "x", [1.0; 4]) else {
            panic!("scatter should be points");
        };
        assert_eq!(xs.len(), n);
        assert_eq!(ys[n - 1], 7.0);
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
        let scatter = bins(
            &root,
            &ids,
            &serde_json::json!({"glyph": "Scatter", "metric": "Dose Error (%)"}),
        );
        let dose = scatter
            .panels
            .iter()
            .find(|panel| panel.title.starts_with("IC1"))
            .unwrap();
        assert!(dose.xmin > 50.0, "scatter x is energy in MeV");
        let position = bins(
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
        let rate = bins(
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
        let scene = bins(
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
        let iso = bins(
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
        let chamber = bins(
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
        let scene = bins(
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
        let scene = bins(
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
        let scene = bins(
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
        let scene = bins(
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
        let scene = bins(
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
        let scene = bins(
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
        let scene = bins(
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
        let scene = bins(
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
        let scene = bins(
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
        let scene = bins(
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
        let spot = bins(
            std::path::Path::new("."),
            &[],
            &serde_json::json!({"metric": "Dose Error (%)"}),
        );
        let source = spot
            .controls
            .iter()
            .find(|control| control.id == "source")
            .unwrap();
        assert_eq!(source.labels(), vec!["Spot", "Timeslice"]);
        assert_eq!(source.value, "Spot");
        assert!(spot.controls.iter().all(|control| control.id != "frame"));
        let metric = spot
            .controls
            .iter()
            .find(|control| control.id == "y")
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
        let count = spot
            .controls
            .iter()
            .find(|control| control.id == "bins")
            .unwrap();
        assert_eq!(count.value, "Auto");
        assert_eq!(count.labels(), vec!["Auto", "8", "16", "32", "64"]);
        let hist = spot
            .controls
            .iter()
            .find(|control| control.id == "hist_bins")
            .unwrap();
        assert_eq!(hist.value, "Auto");
        assert_eq!(hist.labels(), vec!["Auto", "8", "16", "32", "64"]);
        let share = spot
            .controls
            .iter()
            .find(|control| control.id == "share")
            .unwrap();
        assert_eq!(share.labels(), vec!["Own", "Plot", "Page"]);
        let trend = spot
            .controls
            .iter()
            .find(|control| control.id == "trend")
            .unwrap();
        assert_eq!(trend.value, "Off");
        assert_eq!(trend.labels(), vec!["Off", "Linear", "Polynomial"]);
        assert!(spot.controls.iter().all(|control| control.id != "fliers"));
        assert!(spot.controls.iter().all(|control| control.id != "cutoff"));
        let position = bins(
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
        assert_eq!(interlock.labels(), vec!["Off", "On"]);

        let switched = bins(
            std::path::Path::new("."),
            &[],
            &serde_json::json!({"metric": "Dose Error (%)", "source": "Timeslice"}),
        );
        let metric = switched
            .controls
            .iter()
            .find(|control| control.id == "y")
            .unwrap();
        assert_eq!(metric.value, "Position Error (mm)");
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
        let fit = bins(
            std::path::Path::new("."),
            &[],
            &serde_json::json!({"metric": "Fit Confidence", "source": "Timeslice", "x": "Target MU"}),
        );
        let x = fit
            .controls
            .iter()
            .find(|control| control.id == "x")
            .unwrap();
        assert_eq!(x.value, "Energy (MeV)");
        assert_eq!(x.labels(), vec!["Energy (MeV)"]);
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

        let position = bins(
            std::path::Path::new("."),
            &[],
            &serde_json::json!({"metric": "Position Error (mm) (Chamber)", "source": "Spot"}),
        );
        assert_eq!(
            position
                .controls
                .iter()
                .find(|control| control.id == "y")
                .unwrap()
                .value,
            "Position Error (mm) (Chamber)"
        );
        assert!(position
            .controls
            .iter()
            .all(|control| control.id != "frame"));
        let slice = bins(
            std::path::Path::new("."),
            &[],
            &serde_json::json!({"metric": "Position Error (mm)", "source": "Timeslice", "frame": "Chamber"}),
        );
        assert_eq!(
            slice
                .controls
                .iter()
                .find(|control| control.id == "y")
                .unwrap()
                .value,
            "Position Error (mm)"
        );
        assert!(slice.controls.iter().any(|control| control.id == "bins"));
        assert!(slice.controls.iter().all(|control| control.id != "frame"));

        let current = bins(
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
        let own = bins(
            &root,
            &ids,
            &serde_json::json!({"hist": "On", "metric": "Dose Error (%)"}),
        );
        let shared = bins(
            &root,
            &ids,
            &serde_json::json!({"hist": "On", "shared": "On", "metric": "Dose Error (%)"}),
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
    fn page_share_uses_one_range_for_every_histogram() {
        let root = std::env::temp_dir().join("scan-kit-binned-page");
        let _ = std::fs::remove_dir_all(&root);
        let session = root.join("a");
        std::fs::create_dir_all(&session).unwrap();
        std::fs::write(session.join("input_map.csv"), "energy,charge_req\n70,1\n").unwrap();
        std::fs::write(
            session.join("spot_data.csv"),
            "ic1_total_dose_spot,ic2_total_dose_spot\n1,50\n",
        )
        .unwrap();
        let ids = ["a".to_string()];
        let plot = bins(
            &root,
            &ids,
            &serde_json::json!({"hist": "On", "share": "Plot", "metric": "Dose Error (%)"}),
        );
        let page = bins(
            &root,
            &ids,
            &serde_json::json!({"hist": "On", "share": "Page", "metric": "Dose Error (%)"}),
        );
        let plot_spans = probability_spans(&plot);
        let page_spans = probability_spans(&page);
        assert_eq!(plot_spans.len(), 2);
        assert!(plot_spans[0] < 10.0);
        assert!(plot_spans[1] > 100.0);
        assert!(page_spans.iter().all(|span| *span > 100.0));
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
        let (levels, stable) = assign_x(&mut tables, "x", &BinChoice::Automatic);
        assert!(stable);
        assert_eq!(levels, vec![1.0, 2.0, 5.0]);
        assert_eq!(tables[0]["_bin"], vec![1.0, 1.0, 2.0, 5.0]);

        let many: Vec<f32> = (0..100).map(|value| value as f32).collect();
        let mut tables = column(&many);
        let (grouped, stable) = assign_x(&mut tables, "x", &BinChoice::Automatic);
        assert!(
            !stable && grouped.len() <= 32,
            "a long axis falls back to quantile bins"
        );
        assert!(grouped.len() > 1);

        let mut tables = column(&[10.0, 20.0, 30.0, 40.0, 50.0]);
        let (fixed, stable) = assign_x(&mut tables, "x", &BinChoice::Fixed(2));
        assert!(!stable);
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
        apply_mask(
            &mut upper,
            &[
                Segment::Beam {
                    state: BeamGate::Both,
                },
                Segment::Rank {
                    which: Rank::Upper95,
                },
            ],
            &["y"],
        );
        assert_eq!(finite(&upper), vec![1000.0]);

        let mut lower = fresh();
        apply_mask(
            &mut lower,
            &[
                Segment::Beam {
                    state: BeamGate::Both,
                },
                Segment::Rank {
                    which: Rank::Lower95,
                },
            ],
            &["y"],
        );
        let kept = finite(&lower);
        assert!(kept.contains(&1.0));
        assert!(!kept.contains(&1000.0));

        let mut mad = fresh();
        apply_mask(
            &mut mad,
            &[
                Segment::Beam {
                    state: BeamGate::Both,
                },
                Segment::Rank { which: Rank::Mad },
            ],
            &["y"],
        );
        assert_eq!(finite(&mad), vec![1000.0]);

        let mut off = fresh();
        apply_mask(
            &mut off,
            &[
                Segment::Beam {
                    state: BeamGate::Off,
                },
                Segment::Rank { which: Rank::All },
            ],
            &["y"],
        );
        assert_eq!(finite(&off), vec![0.0]);

        let mut on = fresh();
        apply_mask(
            &mut on,
            &[
                Segment::Beam {
                    state: BeamGate::On,
                },
                Segment::Rank { which: Rank::All },
            ],
            &["y"],
        );
        let kept = finite(&on);
        assert!(!kept.contains(&0.0));
        assert!(kept.contains(&1000.0));

        let dose = bins(
            std::path::Path::new("."),
            &[],
            &serde_json::json!({"metric": "dose_error"}),
        );
        let listed = dose
            .controls
            .iter()
            .find(|control| control.id == "segments")
            .unwrap();
        assert_eq!(
            parse_segments(&listed.value).unwrap(),
            vec![
                Segment::Beam {
                    state: BeamGate::Both
                },
                Segment::Rank { which: Rank::All },
            ]
        );
        assert_eq!(listed.labels(), vec!["Beam", "Rank"]);
        let current = bins(
            std::path::Path::new("."),
            &[],
            &serde_json::json!({"metric": "ic_current"}),
        );
        let current_list = parse_segments(
            &current
                .controls
                .iter()
                .find(|control| control.id == "segments")
                .unwrap()
                .value,
        )
        .unwrap();
        assert!(current_list.iter().any(|item| matches!(
            item,
            Segment::Beam {
                state: BeamGate::Both
            }
        )));
        let rate = bins(
            std::path::Path::new("."),
            &[],
            &serde_json::json!({"metric": "dose_rate"}),
        );
        assert!(rate.controls.iter().all(|control| control.id != "segments"));
    }

    fn probability_spans(scene: &scan_kit_core::PlotScene) -> Vec<f32> {
        scene
            .panels
            .iter()
            .filter(|panel| panel.y_label == "Probability (%)")
            .filter_map(|panel| {
                panel.series.iter().find_map(|series| match series {
                    Series::Bars { edges, .. } => edges.last().copied(),
                    _ => None,
                })
            })
            .collect()
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
    fn axis_label_includes_units_and_side_plots_are_wider() {
        assert_eq!(axis_label("IC1", "IC Current (nA)"), "IC1 (nA)");
        assert_eq!(axis_label("IC1 (%)", "Dose Error (%)"), "IC1 (%)");
        assert_eq!(axis_label("MU/S", "Dose Rate (MU/s)"), "MU/S");
        assert_eq!(axis_label("TOTAL MS", "Spot Delivery Time"), "TOTAL MS");
        assert_eq!(axis_label("IC1 X", "Fit Confidence"), "IC1 X");
        assert_eq!(axis_label("X", "Amplifier Error (V)"), "X (V)");
        assert_eq!(axis_label("Y", "Probe Field (G)"), "Y (G)");
        let scene = bins(
            std::path::Path::new("."),
            &[],
            &serde_json::json!({"hist": "On", "corr": "On"}),
        );
        assert_eq!(scene.column_weights, vec![8.0, 1.65, 1.65]);
    }

    #[test]
    fn a_growing_violin_matches_one_pass() {
        let bins: Vec<f32> = (0..900)
            .map(|index| if index < 400 { 1.0 } else { 2.0 })
            .collect();
        let y: Vec<f32> = (0..900)
            .map(|index| ((index * 17) % 50) as f32 * 0.05)
            .collect();
        let categories = [1.0, 2.0];
        let color = [0.2, 0.4, 0.8, 0.55];
        let mut table = BTreeMap::new();
        table.insert("_bin".to_string(), bins.clone());
        let once = super::glyphs::violin_series(&table, &y, &categories, 0, color);
        let id = "grow-violin-one-pass";
        super::glyphs::begin_grown(id);
        let _ = super::glyphs::grown_violin(
            &format!("{id}|y|0"),
            &bins[..400],
            &y[..400],
            &categories,
            color,
        );
        let grown =
            super::glyphs::grown_violin(&format!("{id}|y|0"), &bins, &y, &categories, color);
        let ys = |series: &[scan_kit_core::Series]| match &series[0] {
            scan_kit_core::Series::Triangles { ys, .. } => ys.clone(),
            _ => panic!("violin fill should be triangles"),
        };
        assert_eq!(ys(&once), ys(&grown));
    }

    #[test]
    fn a_long_violin_peaks_on_the_dense_value() {
        let mut values = vec![0.0f32; 1_500];
        values.extend((0..500).map(|i| (i as f32 - 250.0) * 0.02));
        let shape = super::glyphs::kde(&values, 0.3);
        assert_eq!(shape.len(), 48);
        let (peak, _) = shape
            .iter()
            .max_by(|left, right| left.1.partial_cmp(&right.1).unwrap())
            .copied()
            .unwrap();
        assert!(peak.abs() < 0.25, "{peak}");
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

    fn pair_keys(metric: &str, series: &[super::YSeries]) -> Vec<(&'static str, &'static str)> {
        let present: Vec<_> = series.iter().collect();
        super::correlation_slots(metric, &present)
            .into_iter()
            .map(|(left, right)| (left.key, right.key))
            .collect()
    }

    #[test]
    fn correlation_pairs_follow_the_selected_source() {
        assert_eq!(
            pair_keys("position_error", super::POSITION),
            vec![
                ("ic1_x_err", "ic2_x_err"),
                ("ic1_y_err", "ic2_y_err"),
                ("ic1_x_err", "ic1_y_err"),
                ("ic2_x_err", "ic2_y_err"),
            ]
        );
        assert_eq!(
            pair_keys("position_error", &super::POSITION[..2]),
            vec![("ic1_x_err", "ic1_y_err"), ("ic1_x_err", "ic1_y_err")]
        );
        assert_eq!(
            pair_keys("sigma", super::SIGMA),
            vec![
                ("ic1_sig_x", "ic2_sig_x"),
                ("ic1_sig_y", "ic2_sig_y"),
                ("ic1_sig_x", "ic1_sig_y"),
                ("ic2_sig_x", "ic2_sig_y"),
            ]
        );
        assert_eq!(
            pair_keys("amplifier_error", super::AMPLIFIER),
            vec![("amp_x", "amp_y"), ("amp_x", "amp_y")]
        );
        assert!(pair_keys("dose_rate", super::DOSE_RATE).is_empty());
        assert_eq!(
            pair_keys("dose_error", super::DOSE_ERROR),
            vec![
                ("ic1_dose_err_pct", "ic2_dose_err_pct"),
                ("ic2_dose_err_pct", "ic3_dose_err_pct"),
                ("ic3_dose_err_pct", "ic1_dose_err_pct"),
            ]
        );
    }

    fn write_position_session(root: &std::path::Path, name: &str, rows: &str) {
        let session = root.join(name);
        std::fs::create_dir_all(&session).unwrap();
        std::fs::write(
            session.join("input_map.csv"),
            "energy,charge_req,position_x,position_y\n70,1,0,0\n90,1,4,1\n",
        )
        .unwrap();
        std::fs::write(session.join("spot_data.csv"), rows).unwrap();
    }

    #[test]
    fn position_error_correlations_use_session_colors_and_the_four_questions() {
        let root = std::env::temp_dir().join("scan-kit-binned-corr");
        let _ = std::fs::remove_dir_all(&root);
        let header = "ic1_total_dose_spot,r_ic1_x_spot_position,r_ic1_y_spot_position,r_ic2_x_spot_position,r_ic2_y_spot_position\n";
        write_position_session(&root, "a", &format!("{header}1,1,0,2,1\n1,2,3,6,4\n"));
        write_position_session(&root, "b", &format!("{header}1,3,1,0,2\n1,5,2,2,3\n"));
        let scene = bins(
            &root,
            &["a".to_string(), "b".to_string()],
            &serde_json::json!({"metric": "Position Error (mm)", "corr": "On"}),
        );
        let labels: Vec<_> = scene
            .panels
            .iter()
            .filter(|panel| panel.title.is_empty() && !panel.x_label.is_empty())
            .map(|panel| (panel.x_label.as_str(), panel.y_label.as_str()))
            .collect();
        assert_eq!(
            labels,
            vec![
                ("IC1 X (mm)", "IC2 X (mm)"),
                ("IC1 Y (mm)", "IC2 Y (mm)"),
                ("IC1 X (mm)", "IC1 Y (mm)"),
                ("IC2 X (mm)", "IC2 Y (mm)"),
            ]
        );
        for panel in scene
            .panels
            .iter()
            .filter(|panel| panel.title.is_empty() && !panel.x_label.is_empty())
        {
            let clouds = panel
                .series
                .iter()
                .filter(|series| matches!(series, Series::Points { .. }))
                .count();
            assert_eq!(clouds, 2, "{} vs {}", panel.x_label, panel.y_label);
            assert!(panel.series.iter().any(|series| matches!(
                series,
                Series::Polyline { color, .. } if color[3] == 0.0
            )));
        }
        let _ = std::fs::remove_dir_all(&root);
    }
}
