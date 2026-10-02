use std::collections::BTreeMap;

use scan_kit_core::{box_stats, linear_fit, Panel, Series};

use crate::histogram::histogram_panel;

use super::*;

pub(super) fn violin_series(
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

pub(super) fn box_series(
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

pub(super) fn mean_series(
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

pub(super) fn scatter_series(
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

pub(super) fn contour_series(
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

pub(super) fn binned_trend(
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

pub(super) fn scatter_trend(
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
pub(super) fn fit_trend(xs: &[f32], ys: &[f32], quadratic: bool) -> Option<[f64; 3]> {
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

pub(super) fn solve3(r0: [f64; 4], r1: [f64; 4], r2: [f64; 4]) -> Option<[f64; 3]> {
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

pub(super) fn eval_trend(coef: [f64; 3], x: f32) -> f32 {
    let x = f64::from(x);
    (coef[0] + coef[1] * x + coef[2] * x * x) as f32
}

pub(super) fn phys_at_plot_x(plot_x: f32, categories: &[f32], session: usize) -> f32 {
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

pub(super) fn curve_across(
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

pub(super) fn hist_panel(
    series: &YSeries,
    tables: &[BTreeMap<String, Vec<f32>>],
    bins: usize,
    shared: bool,
    range: Option<(f32, f32)>,
    position_guides: bool,
) -> Panel {
    let columns: Vec<&[f32]> = tables
        .iter()
        .filter_map(|table| table.get(series.key).map(Vec::as_slice))
        .filter(|column| column.iter().any(|value| value.is_finite()))
        .collect();
    let guides: Vec<f32> = if position_guides
        && matches!(
            series.key,
            "ic1_x_err" | "ic1_y_err" | "ic2_x_err" | "ic2_y_err"
        ) {
        vec![1.0, 2.0, 3.0, -1.0, -2.0, -3.0]
    } else {
        Vec::new()
    };
    histogram_panel("", &columns, bins, shared, range, &guides)
}

pub(super) fn corr_panel(
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
        x_label: axis_label(a.label, group_label),
        xmin,
        xmax,
        ymin,
        ymax,
        series,
        x_labels: Vec::new(),
        equal: false,
    }
}

pub(super) fn interlock_guides(
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

pub(super) fn dose_gates(categories: &[f32], glyph: &str, xmin: f32, xmax: f32) -> Vec<Series> {
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

pub(super) fn sigma_bands(expected: &[f32], categories: &[f32], absolute: bool) -> Vec<Series> {
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

pub(super) fn expected_at(pairs: &[f32], energy: f32) -> Option<f32> {
    pairs
        .chunks(2)
        .find(|pair| pair.len() == 2 && same(pair[0], energy))
        .map(|pair| pair[1])
}

pub(super) fn hline(xmin: f32, xmax: f32, y: f32, color: [f32; 4]) -> Series {
    Series::Guide {
        xs: vec![xmin, xmax],
        ys: vec![y, y],
        color,
        thickness: 1.0,
    }
}

pub(super) fn note_panel(title: &str) -> Panel {
    Panel {
        title: title.to_string(),
        y_label: String::new(),
        x_label: String::new(),
        xmin: 0.0,
        xmax: 1.0,
        ymin: 0.0,
        ymax: 1.0,
        series: Vec::new(),
        x_labels: Vec::new(),
        equal: false,
    }
}

pub(super) fn kde(values: &[f32], half: f32) -> Vec<(f32, f32)> {
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

pub(super) fn pchip(xs: &[f32], ys: &[f32]) -> (Vec<f32>, Vec<f32>) {
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
