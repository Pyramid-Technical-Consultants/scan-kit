//! Outlined histograms shared by Distribution and Bins.
//!
//! Each session is a translucent bar series plus a polyline around those bars.
//! The outline's alpha is 0 so palette assignment keeps the previous series' color.

use scan_kit_core::{histogram, Control, Panel, Series};

use crate::marks::control;

pub(crate) const BIN_CHOICES: &[&str] = &["Auto", "8", "16", "32", "64"];

/// The bins picker: `Auto` renders icon-only with a tooltip so the row stays
/// compact; the counts keep their labels.
pub(crate) fn bins_control(id: &str, value: &str) -> Control {
    let mut control = control(id, "Bins", BIN_CHOICES, value);
    let auto = &mut control.options[0];
    auto.detail = "Pick a bin count automatically from the data".to_owned();
    auto.icon = "auto".to_owned();
    auto.icon_only = true;
    control
}

const BIN_STEPS: [usize; 4] = [8, 16, 32, 64];

const FILL: [f32; 4] = [0.8, 0.8, 0.8, 0.45];
const OUTLINE: [f32; 4] = [0.85, 0.85, 0.85, 0.0];
const GUIDE: [f32; 4] = [0.45, 0.7, 0.4, 0.7];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BinShare {
    Own,
    Plot,
    Page,
}

/// `Auto` and `Automatic` stay automatic. Any other count snaps to 8, 16, 32, or 64.
pub(crate) fn count_choice(raw: &str) -> Option<usize> {
    let raw = raw.trim();
    if raw.is_empty() || raw.eq_ignore_ascii_case("auto") || raw.eq_ignore_ascii_case("automatic") {
        None
    } else {
        // JSON callers send 16.0 for 16 â€” parse as float too instead of
        // silently snapping to 32.
        let parsed = raw
            .parse::<usize>()
            .ok()
            .or_else(|| {
                raw.parse::<f64>()
                    .ok()
                    .filter(|value| value.is_finite() && *value >= 0.0)
                    .map(|value| value.round() as usize)
            })
            .unwrap_or(32);
        Some(
            BIN_STEPS
                .into_iter()
                .min_by_key(|step| step.abs_diff(parsed))
                .unwrap_or(32),
        )
    }
}

pub(crate) fn bin_button(raw: &str) -> String {
    match count_choice(raw) {
        None => "Auto".to_string(),
        Some(count) => count.to_string(),
    }
}

/// Equal-width histogram count. Auto is 32, the same width the grouped axis uses.
pub(crate) fn hist_bin_count(raw: &str) -> usize {
    count_choice(raw).unwrap_or(32)
}

pub(crate) fn bin_share(raw: &str, legacy_on: bool) -> BinShare {
    match raw.trim().to_ascii_lowercase().as_str() {
        "plot" | "shared" => BinShare::Plot,
        "page" | "global" => BinShare::Page,
        "own" | "none" => BinShare::Own,
        "" => {
            if legacy_on {
                BinShare::Plot
            } else {
                BinShare::Own
            }
        }
        _ => BinShare::Own,
    }
}

pub(crate) fn share_key(share: BinShare) -> &'static str {
    match share {
        BinShare::Own => "own",
        BinShare::Plot => "plot",
        BinShare::Page => "page",
    }
}

/// Filled bars and their outline. `range` forces one edge set: the distribution
/// marginals use the cloud limits, and a page share uses one range for every
/// histogram. Otherwise `shared` uses one range for the columns on this plot,
/// and each column keeps its own. `guides` are vertical lines at those x values.
pub(crate) fn histogram_panel(
    x_label: &str,
    columns: &[&[f32]],
    bins: usize,
    shared: bool,
    range: Option<(f32, f32)>,
    guides: &[f32],
) -> Panel {
    let shared_edges = if let Some((lo, hi)) = range {
        Some(uniform_edges(lo, hi, bins))
    } else if shared {
        let mut lo = f32::MAX;
        let mut hi = f32::MIN;
        for column in columns {
            for value in *column {
                if value.is_finite() {
                    lo = lo.min(*value);
                    hi = hi.max(*value);
                }
            }
        }
        Some(uniform_edges(lo, hi, bins))
    } else {
        None
    };
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
        // A placeholder column is all-zero counts — its edges must not
        // stretch the axis to the 0..1 default.
        if !column.is_empty() {
            lo = lo.min(edges.first().copied().unwrap_or(0.0));
            hi = hi.max(edges.last().copied().unwrap_or(1.0));
        }
        let total = counts.iter().sum::<f32>().max(1.0);
        let percent: Vec<f32> = counts.iter().map(|count| count / total * 100.0).collect();
        peak = peak.max(percent.iter().copied().fold(0.0, f32::max));
        drawn.push(Series::Bars {
            edges: edges.clone(),
            counts: percent.clone(),
            color: FILL,
        });
        drawn.push(bar_outline(&edges, &percent));
    }
    for level in guides {
        drawn.push(Series::Guide {
            xs: vec![*level, *level],
            ys: vec![0.0, peak],
            color: GUIDE,
            thickness: 1.0,
        });
    }
    if !lo.is_finite() || !hi.is_finite() || hi <= lo {
        lo = 0.0;
        hi = 1.0;
    }
    Panel {
        title: String::new(),
        y_label: "Probability (%)".into(),
        x_label: x_label.to_string(),
        xmin: lo,
        xmax: hi,
        ymin: 0.0,
        ymax: peak * 1.1,
        series: drawn,
        x_labels: Vec::new(),
        equal: false,
    }
}

fn uniform_edges(lo: f32, hi: f32, bins: usize) -> Vec<f32> {
    let bins = bins.max(1);
    // The span subtracts in f64 — a finite range wider than f32::MAX would
    // otherwise overflow to +inf and NaN the first edge.
    let span = f64::from(hi) - f64::from(lo);
    // Relative flat test like scan_kit_core::histogram — an absolute 1e-12
    // calls a real 1e-13-wide column flat and bins it onto a ±0.5 axis.
    let scale = f64::from(hi.abs().max(lo.abs())).max(1e-30);
    let (lo, hi) = if !lo.is_finite() || !hi.is_finite() || span.abs() <= 1e-12 * scale {
        // A fixed 0.5 pad is unrepresentable above |lo| ~= 2^23 (ulp > 1) —
        // scale the pad to the magnitude so the edges stay distinct.
        let pad = (lo.abs() * f32::EPSILON).max(0.5);
        ((lo - pad).max(-f32::MAX), (lo + pad).min(f32::MAX))
    } else {
        (lo, hi)
    };
    let span = f64::from(hi) - f64::from(lo);
    (0..=bins)
        .map(|step| (f64::from(lo) + span * step as f64 / bins as f64) as f32)
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
    let span = f64::from(hi) - f64::from(lo);
    if !span.is_finite() || span <= 0.0 {
        return counts;
    }
    let scale = bins as f64 / span;
    for value in values {
        if !value.is_finite() || *value < lo || *value > hi {
            continue;
        }
        let mut index = ((f64::from(*value) - f64::from(lo)) * scale) as usize;
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
        color: OUTLINE,
        thickness: 1.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_range_wider_than_f32_still_bins() {
        let edges = uniform_edges(-2.0e38, 2.0e38, 4);
        assert!(edges.iter().all(|edge| edge.is_finite()), "{edges:?}");
        assert!(edges.windows(2).all(|w| w[0] <= w[1]), "{edges:?}");
        let counts = count_bins(&edges, &[-2.0e38, 0.0, 2.0e38]);
        assert_eq!(counts.iter().sum::<f32>(), 3.0, "{counts:?}");
    }

    #[test]
    fn outlined_histogram_strokes_each_session_on_shared_edges() {
        let panel = histogram_panel(
            "X (mm)",
            &[&[0.0, 0.0, 1.0], &[1.0, 1.0, 1.0]],
            2,
            true,
            None,
            &[0.0],
        );
        assert!(panel.title.is_empty());
        assert_eq!(panel.x_label, "X (mm)");
        assert_eq!(panel.y_label, "Probability (%)");
        let bars: Vec<_> = panel
            .series
            .iter()
            .filter_map(|series| match series {
                Series::Bars { edges, .. } => Some(edges.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(bars.len(), 2);
        assert_eq!(bars[0], bars[1]);
        assert_eq!(
            panel
                .series
                .iter()
                .filter(|series| matches!(series, Series::Polyline { .. }))
                .count(),
            2
        );
        assert!(panel.series.iter().any(|series| matches!(
            series,
            Series::Guide { xs, .. } if xs.first() == Some(&0.0)
        )));
    }

    #[test]
    fn bin_counts_snap_to_the_button_row() {
        assert_eq!(bin_button("Automatic"), "Auto");
        assert_eq!(bin_button("30"), "32");
        assert_eq!(bin_button("10"), "8");
        assert_eq!(bin_button("20"), "16");
        assert_eq!(bin_button("50"), "64");
        assert_eq!(hist_bin_count("Auto"), 32);
        assert_eq!(bin_share("Plot", false), BinShare::Plot);
        assert_eq!(bin_share("", true), BinShare::Plot);
        assert_eq!(bin_share("", false), BinShare::Own);
        assert_eq!(share_key(BinShare::Page), "page");
    }

    #[test]
    fn a_json_float_count_still_snaps_to_the_nearest_step() {
        // MCP callers send 16.0 for 16 — it must not fall back to 32.
        assert_eq!(super::count_choice("16.0"), Some(16));
        assert_eq!(super::count_choice("20.0"), Some(16));
        assert_eq!(super::count_choice("64"), Some(64));
    }
}
