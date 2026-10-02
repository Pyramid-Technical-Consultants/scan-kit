//! Shared plot helpers: option picking, row filters, and contour bands.

use std::collections::BTreeMap;

use scan_kit_core::{Control, Series};
use serde_json::Value;

use super::tables::{modified_z, percentile};

const MOD_Z: f32 = 3.5;
pub(crate) const BEAM_CHOICES: &[(&str, &str)] = &[
    ("beam_on", "Beam On"),
    ("beam_off", "Beam Off"),
    ("beam_both", "Both"),
];

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

pub(crate) fn control(id: &str, label: &str, options: &[&str], value: &str) -> Control {
    Control {
        id: id.to_string(),
        label: label.to_string(),
        options: options.iter().map(|option| (*option).to_string()).collect(),
        value: value.to_string(),
    }
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

pub(crate) fn text<'a>(options: &'a Value, key: &str, default: &'a str) -> &'a str {
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

pub(crate) fn pick_in<'a>(
    options: &Value,
    key: &str,
    pairs: &[(&'a str, &'a str)],
) -> Option<&'a str> {
    let raw = options.get(key).and_then(Value::as_str)?;
    pairs
        .iter()
        .find(|(id, label)| *id == raw || *label == raw)
        .map(|(id, _)| *id)
}

pub(crate) fn flag(options: &Value, key: &str, default_on: bool) -> bool {
    match text(options, key, if default_on { "On" } else { "Off" }) {
        "off" | "Off" | "false" | "0" => false,
        "on" | "On" | "true" | "1" => true,
        _ => default_on,
    }
}

#[cfg(test)]
mod tests {
    use scan_kit_core::Series;

    use super::super::tables::signal_table;
    use super::*;

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
}
