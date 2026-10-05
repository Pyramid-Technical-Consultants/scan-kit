//! Shared numeric meaning for the analysis views.
//!
//! File IO stays in `scan-kit-io`. These functions take columns they already hold.

/// G2 strip register 64.5 maps to 0 mm. Pitch matches the 1–128 → −128–128 mm map.
pub const G2_STRIP_CENTER: f32 = 64.5;
pub const G2_MM_PER_STRIP: f32 = 256.0 / 127.0;

/// G3 strip channel 64.5 maps to 0 mm at a 2 mm pitch.
pub const G3_STRIP_CENTER: f32 = 64.5;
pub const G3_STRIP_PITCH_MM: f32 = 2.0;
/// Linear remap. `in_max == in_min` returns `out_min`.
pub fn remap(x: f32, in_min: f32, in_max: f32, out_min: f32, out_max: f32) -> f32 {
    let span = in_max - in_min;
    if span.abs() < 1e-12 {
        return out_min;
    }
    (x - in_min) * (out_max - out_min) / span + out_min
}

pub fn remap_g2_raw(x: f32) -> f32 {
    (x - G2_STRIP_CENTER) * G2_MM_PER_STRIP
}

pub fn remap_g2_raw_reversed(x: f32) -> f32 {
    (G2_STRIP_CENTER - x) * G2_MM_PER_STRIP
}

pub fn remap_g3_raw(x: f32) -> f32 {
    (x - G3_STRIP_CENTER) * G3_STRIP_PITCH_MM
}

pub fn remap_g3_raw_reversed(x: f32) -> f32 {
    (G3_STRIP_CENTER - x) * G3_STRIP_PITCH_MM
}

/// Pick the IC2 strip direction that sits closer to IC1. IC1 is never flipped.
pub fn g2_ic2_mm(ic1_mm: &[f32], raw_ic2: &[f32]) -> Vec<f32> {
    let n = ic1_mm.len().min(raw_ic2.len());
    let fwd: Vec<f32> = raw_ic2.iter().take(n).copied().map(remap_g2_raw).collect();
    let err = |pred: &[f32], flip: bool| -> f32 {
        let mut vals: Vec<f32> = ic1_mm
            .iter()
            .zip(pred)
            .filter_map(|(a, b)| {
                let d = if flip { a + b } else { a - b };
                d.abs().is_finite().then_some(d.abs())
            })
            .collect();
        if vals.is_empty() {
            return f32::MAX;
        }
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        vals[vals.len() / 2]
    };
    if err(&fwd, true) < err(&fwd, false) {
        raw_ic2.iter().copied().map(remap_g2_raw_reversed).collect()
    } else {
        raw_ic2.iter().copied().map(remap_g2_raw).collect()
    }
}

/// `sum(target) / sum(delivered)` over finite positive pairs. Missing data is `None`.
pub fn calibration_factor(target: &[f32], delivered: &[f32]) -> Option<f32> {
    let mut sum_t = 0.0f64;
    let mut sum_d = 0.0f64;
    for (t, d) in target.iter().zip(delivered) {
        if t.is_finite() && d.is_finite() && *t > 0.0 && *d > 0.0 {
            sum_t += f64::from(*t);
            sum_d += f64::from(*d);
        }
    }
    (sum_d > 0.0).then_some((sum_t / sum_d) as f32)
}

pub fn scale_column(values: &[f32], factor: f32) -> Vec<f32> {
    values.iter().map(|value| value * factor).collect()
}

/// `(delivered / reference - 1) * 100`. Non-finite or zero reference becomes NaN.
pub fn dose_ratio_pct(delivered: &[f32], reference: &[f32]) -> Vec<f32> {
    delivered
        .iter()
        .zip(reference)
        .map(|(d, r)| {
            if d.is_finite() && r.is_finite() && r.abs() > 1e-15 {
                (d / r - 1.0) * 100.0
            } else {
                f32::NAN
            }
        })
        .collect()
}

/// `(delivered - target) / target * 100`.
pub fn dose_error_pct(delivered: &[f32], target: &[f32]) -> Vec<f32> {
    delivered
        .iter()
        .zip(target)
        .map(|(d, t)| {
            if d.is_finite() && t.is_finite() && t.abs() > 1e-15 {
                (d - t) / t * 100.0
            } else {
                f32::NAN
            }
        })
        .collect()
}
/// Gate is on when the column is non-zero. An empty column is all off.
pub fn beam_on_mask(gate: &[f32]) -> Vec<bool> {
    gate.iter()
        .map(|value| *value != 0.0 && value.is_finite())
        .collect()
}
/// Quantile bin edges. A single finite value becomes a unit-width bin around it.
pub fn quantile_edges(values: &[f32], n_bins: usize) -> Vec<f32> {
    let mut finite: Vec<f32> = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    if finite.is_empty() {
        return vec![0.0, 1.0];
    }
    finite.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n_bins = n_bins.max(1);
    if finite.first() == finite.last() {
        let lo = finite[0];
        return vec![lo - 0.5, lo + 0.5];
    }
    let mut edges = Vec::with_capacity(n_bins + 1);
    for i in 0..=n_bins {
        let q = i as f32 / n_bins as f32;
        edges.push(quantile_sorted(&finite, q));
    }
    edges.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
    if edges.len() < 2 {
        let lo = finite[0];
        return vec![lo - 0.5, lo + 0.5];
    }
    edges
}

fn quantile_sorted(sorted: &[f32], q: f32) -> f32 {
    if sorted.is_empty() {
        return f32::NAN;
    }
    let pos = q.clamp(0.0, 1.0) * (sorted.len() - 1) as f32;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let t = pos - lo as f32;
    sorted[lo] * (1.0 - t) + sorted[hi] * t
}

/// Map each value to the center of its quantile bin. Out of range is NaN.
///
/// Matches `numpy.digitize` on the interior edges with the last bin closed
/// on the right, which is what the 1.8 binned summary uses.
pub fn assign_bin_centers(values: &[f32], edges: &[f32]) -> Vec<f32> {
    let mut out = vec![f32::NAN; values.len()];
    if edges.len() < 2 {
        return out;
    }
    let centers: Vec<f32> = edges
        .windows(2)
        .map(|pair| pair[0].midpoint(pair[1]))
        .collect();
    let interior = &edges[1..edges.len() - 1];
    for (slot, value) in out.iter_mut().zip(values) {
        if !value.is_finite() || *value < edges[0] || *value > edges[edges.len() - 1] {
            continue;
        }
        let mut index = 0usize;
        for edge in interior {
            if *value >= *edge {
                index += 1;
            }
        }
        *slot = centers[index.min(centers.len() - 1)];
    }
    out
}

/// Matplotlib boxplot stats. Whiskers are the extreme points inside 1.5 IQR.
pub struct BoxStats {
    pub q1: f32,
    pub median: f32,
    pub q3: f32,
    pub whisker_lo: f32,
    pub whisker_hi: f32,
    pub fliers: Vec<f32>,
}

pub fn box_stats(values: &[f32]) -> Option<BoxStats> {
    let mut sorted: Vec<f32> = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    if sorted.is_empty() {
        return None;
    }
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let q1 = quantile_sorted(&sorted, 0.25);
    let median = quantile_sorted(&sorted, 0.5);
    let q3 = quantile_sorted(&sorted, 0.75);
    let iqr = q3 - q1;
    let fence_lo = q1 - 1.5 * iqr;
    let fence_hi = q3 + 1.5 * iqr;
    let whisker_lo = sorted
        .iter()
        .copied()
        .find(|value| *value >= fence_lo)
        .unwrap_or(q1);
    let whisker_hi = sorted
        .iter()
        .rev()
        .copied()
        .find(|value| *value <= fence_hi)
        .unwrap_or(q3);
    let fliers = sorted
        .into_iter()
        .filter(|value| *value < fence_lo || *value > fence_hi)
        .collect();
    Some(BoxStats {
        q1,
        median,
        q3,
        whisker_lo,
        whisker_hi,
        fliers,
    })
}

/// Histogram counts and edges covering the finite values.
pub fn histogram(values: &[f32], bins: usize) -> (Vec<f32>, Vec<f32>) {
    let finite: Vec<f32> = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    let bins = bins.max(1);
    if finite.is_empty() {
        return (vec![0.0, 1.0], vec![0.0]);
    }
    let mut lo = finite[0];
    let mut hi = finite[0];
    for value in &finite {
        lo = lo.min(*value);
        hi = hi.max(*value);
    }
    if (hi - lo).abs() < 1e-12 {
        lo -= 0.5;
        hi += 0.5;
    }
    let mut edges = Vec::with_capacity(bins + 1);
    for i in 0..=bins {
        edges.push(lo + (hi - lo) * i as f32 / bins as f32);
    }
    let mut counts = vec![0.0f32; bins];
    for value in finite {
        let mut index = ((value - lo) / (hi - lo) * bins as f32) as usize;
        if index >= bins {
            index = bins - 1;
        }
        counts[index] += 1.0;
    }
    (edges, counts)
}

/// Welch PSD. `seg_len` 4096, overlap 0.5, and `fs` 1000 match FFT Explorer.
pub fn welch_psd(
    signal: &[f32],
    fs: f32,
    mut seg_len: usize,
    overlap: f32,
) -> (Vec<f32>, Vec<f32>) {
    let mean = finite_mean(signal);
    let centered: Vec<f32> = signal
        .iter()
        .map(|value| if value.is_finite() { value - mean } else { 0.0 })
        .collect();
    if centered.len() < seg_len {
        seg_len = centered.len().max(16);
    }
    let step = ((seg_len as f32) * (1.0 - overlap)).round().max(1.0) as usize;
    let window = hanning(seg_len);
    let win_power: f32 = window.iter().map(|w| w * w).sum();
    let mut accum: Option<Vec<f32>> = None;
    let mut count = 0usize;
    let mut start = 0usize;
    while start + seg_len <= centered.len() {
        let mut segment: Vec<f32> = centered[start..start + seg_len]
            .iter()
            .zip(&window)
            .map(|(sample, w)| sample * w)
            .collect();
        let power = rfft_power(&mut segment);
        match &mut accum {
            None => accum = Some(power),
            Some(sum) => {
                for (bin, value) in sum.iter_mut().zip(power) {
                    *bin += value;
                }
            }
        }
        count += 1;
        start += step;
    }
    let Some(mut psd) = accum else {
        return (Vec::new(), Vec::new());
    };
    let scale = (count as f32) * win_power.max(1e-12);
    let last = psd.len().saturating_sub(1);
    for (i, bin) in psd.iter_mut().enumerate() {
        *bin /= scale;
        if i > 0 && i < last {
            *bin *= 2.0;
        }
    }
    let freqs: Vec<f32> = (0..psd.len())
        .map(|i| i as f32 * fs / seg_len as f32)
        .collect();
    (freqs, psd)
}

fn finite_mean(values: &[f32]) -> f32 {
    let mut sum = 0.0f64;
    let mut n = 0.0f64;
    for value in values {
        if value.is_finite() {
            sum += f64::from(*value);
            n += 1.0;
        }
    }
    if n == 0.0 {
        0.0
    } else {
        (sum / n) as f32
    }
}

fn hanning(n: usize) -> Vec<f32> {
    if n <= 1 {
        return vec![1.0; n];
    }
    (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (n - 1) as f32).cos()))
        .collect()
}

/// Real FFT power `|rfft|^2`. Length may be any n ≥ 1.
fn rfft_power(segment: &mut [f32]) -> Vec<f32> {
    let n = segment.len();
    (0..n / 2 + 1)
        .map(|k| {
            let mut re = 0.0f32;
            let mut im = 0.0f32;
            let angle = -2.0 * std::f32::consts::PI * k as f32 / n as f32;
            for (t, sample) in segment.iter().copied().enumerate() {
                let phase = angle * t as f32;
                re += sample * phase.cos();
                im += sample * phase.sin();
            }
            re * re + im * im
        })
        .collect()
}
/// Deposit isotropic Gaussians onto a regular grid. `spots` are `(x, y, z, weight, sigma)`.
pub fn splat_gaussians(
    spots: &[[f32; 5]],
    origin: [f32; 3],
    spacing: [f32; 3],
    shape: [usize; 3],
) -> Vec<f32> {
    let [nx, ny, nz] = shape;
    let mut grid = vec![0.0f32; nx * ny * nz];
    for spot in spots {
        let [x, y, z, weight, sigma] = *spot;
        if !sigma.is_finite() || sigma <= 0.0 || !weight.is_finite() {
            continue;
        }
        let reach = (3.0 * sigma).ceil() as i32;
        let cx = ((x - origin[0]) / spacing[0]).round() as i32;
        let cy = ((y - origin[1]) / spacing[1]).round() as i32;
        let cz = ((z - origin[2]) / spacing[2]).round() as i32;
        let inv = 1.0 / (2.0 * sigma * sigma);
        for dz in -reach..=reach {
            for dy in -reach..=reach {
                for dx in -reach..=reach {
                    let ix = cx + dx;
                    let iy = cy + dy;
                    let iz = cz + dz;
                    if ix < 0
                        || iy < 0
                        || iz < 0
                        || ix >= nx as i32
                        || iy >= ny as i32
                        || iz >= nz as i32
                    {
                        continue;
                    }
                    let px = origin[0] + ix as f32 * spacing[0];
                    let py = origin[1] + iy as f32 * spacing[1];
                    let pz = origin[2] + iz as f32 * spacing[2];
                    let r2 = (px - x).powi(2) + (py - y).powi(2) + (pz - z).powi(2);
                    let value = weight * (-r2 * inv).exp();
                    let index = ix as usize + nx * (iy as usize + ny * iz as usize);
                    grid[index] += value;
                }
            }
        }
    }
    grid
}

/// Maximum-intensity projection along Z.
pub fn mip_xy(grid: &[f32], shape: [usize; 3]) -> Vec<f32> {
    let [nx, ny, nz] = shape;
    let mut image = vec![0.0f32; nx * ny];
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let value = grid[x + nx * (y + ny * z)];
                let pixel = x + nx * y;
                if value > image[pixel] {
                    image[pixel] = value;
                }
            }
        }
    }
    image
}

/// Global 3D gamma. `cutoff_pct` is a percent of the maximum of `evaluated`
/// (TG-218), not an absolute dose. The search is sub-voxel and stops at γ = 2.
/// Returns `(gamma, passed, evaluated)`.
pub fn gamma_index(
    reference: &[f32],
    evaluated: &[f32],
    shape: [usize; 3],
    dose_percent: f32,
    distance_mm: f32,
    spacing: [f32; 3],
    cutoff_pct: f32,
) -> (Vec<f32>, u32, u32) {
    crate::dose::gamma_index(
        reference,
        evaluated,
        shape,
        dose_percent,
        distance_mm,
        spacing,
        cutoff_pct,
    )
}
/// Least-squares line. `None` when fewer than two finite points share an x span.
pub fn linear_fit(xs: &[f32], ys: &[f32]) -> Option<(f32, f32)> {
    let mut n = 0.0f32;
    let mut sx = 0.0f32;
    let mut sy = 0.0f32;
    let mut sxx = 0.0f32;
    let mut sxy = 0.0f32;
    for (x, y) in xs.iter().zip(ys) {
        if x.is_finite() && y.is_finite() {
            n += 1.0;
            sx += x;
            sy += y;
            sxx += x * x;
            sxy += x * y;
        }
    }
    let den = n * sxx - sx * sx;
    if n < 2.0 || den.abs() < 1e-12 {
        return None;
    }
    let slope = (n * sxy - sx * sy) / den;
    let intercept = (sy - slope * sx) / n;
    Some((slope, intercept))
}

/// Dose-volume histogram as cumulative volume fraction above each dose edge.
pub fn dvh(dose: &[f32], mask: &[bool], bins: usize) -> (Vec<f32>, Vec<f32>) {
    let selected: Vec<f32> = dose
        .iter()
        .zip(mask.iter().chain(std::iter::repeat(&true)))
        .filter_map(|(value, keep)| (*keep && value.is_finite() && *value >= 0.0).then_some(*value))
        .collect();
    let (edges, counts) = histogram(&selected, bins);
    let total = counts.iter().sum::<f32>().max(1.0);
    let mut cumulative = Vec::with_capacity(counts.len());
    let mut above = total;
    for count in &counts {
        cumulative.push(above / total);
        above -= count;
    }
    (edges, cumulative)
}

/// Sum `current` for each spot id, in the order each id first appears.
pub fn sums_by_spot_id(spot: &[f32], current: &[f32]) -> Vec<f32> {
    let mut index = std::collections::BTreeMap::<i32, usize>::new();
    let mut sums = Vec::new();
    for (spot_id, value) in spot.iter().zip(current) {
        if !spot_id.is_finite() || !value.is_finite() {
            continue;
        }
        let id = spot_id.round() as i32;
        if let Some(slot) = index.get(&id) {
            sums[*slot] += value;
        } else {
            index.insert(id, sums.len());
            sums.push(*value);
        }
    }
    sums
}

/// Sum `current` over each contiguous run of the same spot id.
pub fn sums_by_spot_run(spot: &[f32], current: &[f32]) -> Vec<f32> {
    let n = spot.len().min(current.len());
    let mut out = Vec::new();
    let mut acc = 0.0f32;
    let mut have = false;
    let mut last = 0i32;
    for i in 0..n {
        if !spot[i].is_finite() || !current[i].is_finite() {
            continue;
        }
        let id = spot[i].round() as i32;
        if have && id != last {
            out.push(acc);
            acc = 0.0;
        }
        acc += current[i];
        last = id;
        have = true;
    }
    if have {
        out.push(acc);
    }
    out
}

pub fn median_finite(values: &[f32]) -> Option<f32> {
    let mut kept: Vec<f32> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if kept.is_empty() {
        return None;
    }
    kept.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = kept.len();
    if n % 2 == 1 {
        Some(kept[n / 2])
    } else {
        Some((kept[n / 2 - 1] + kept[n / 2]) * 0.5)
    }
}

pub fn trapz(time: &[f32], values: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for i in 1..time.len().min(values.len()) {
        let dt = time[i] - time[i - 1];
        if dt.is_finite() && values[i].is_finite() && values[i - 1].is_finite() {
            acc += 0.5 * (values[i] + values[i - 1]) * dt;
        }
    }
    acc
}

/// Time window around the positive HV step, padded by 40 ms.
pub fn hv_step_window(time: &[f32], current: &[f32]) -> Option<(f32, f32)> {
    let n = time.len().min(current.len());
    if n == 0 {
        return None;
    }
    let base = median_finite(&current[..n])?;
    let mut peak_i = 0usize;
    let mut peak = f32::MIN;
    for (i, value) in current[..n].iter().enumerate() {
        let delta = value - base;
        if delta.is_finite() && delta > peak {
            peak = delta;
            peak_i = i;
        }
    }
    if peak <= 0.0 {
        return None;
    }
    let thr = peak * 0.10;
    let mut lo = peak_i;
    while lo > 0 && current[lo - 1] - base > thr {
        lo -= 1;
    }
    let mut hi = peak_i;
    while hi + 1 < n && current[hi + 1] - base > thr {
        hi += 1;
    }
    Some((time[lo] - 40.0, time[hi] + 40.0))
}

/// Capacitance in pF. Current is nA and time is ms, so nA·ms / V = pF.
pub fn hv_capacitance_pf(
    time: &[f32],
    current: &[f32],
    t0: f32,
    t1: f32,
    delta_v: f32,
) -> Option<f32> {
    if !delta_v.is_finite() || delta_v == 0.0 {
        return None;
    }
    let n = time.len().min(current.len());
    let mut outside = Vec::new();
    let mut tt = Vec::new();
    let mut yy = Vec::new();
    for i in 0..n {
        if !time[i].is_finite() || !current[i].is_finite() {
            continue;
        }
        if time[i] >= t0 && time[i] <= t1 {
            tt.push(time[i]);
            yy.push(current[i]);
        } else {
            outside.push(current[i]);
        }
    }
    let base = if outside.len() > 5 {
        median_finite(&outside)?
    } else {
        median_finite(&current[..n])?
    };
    for value in &mut yy {
        *value -= base;
    }
    if tt.len() < 2 {
        return None;
    }
    Some(trapz(&tt, &yy) / delta_v)
}
/// Percent of finite metrics at or above each threshold.
pub fn coverage_percent(metrics: &[f32], thresholds: &[f32]) -> Vec<f32> {
    let total = metrics.len();
    thresholds
        .iter()
        .map(|threshold| {
            if total == 0 {
                return 0.0;
            }
            let covered = metrics
                .iter()
                .filter(|value| value.is_finite() && **value >= *threshold)
                .count();
            100.0 * covered as f32 / total as f32
        })
        .collect()
}
/// `(overall pass/fail, per-channel flags)` from a firmware result JSON.
pub fn hv_firmware_flags(text: &str) -> (Option<String>, Vec<String>) {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(text) else {
        return (None, Vec::new());
    };
    let result = value.get("result").unwrap_or(&value);
    let mut scalars = Vec::new();
    let mut channels = Vec::new();
    walk_flags(result, &mut scalars, &mut channels);
    let mut all = scalars.iter().chain(&channels);
    if all.next().is_none() {
        return (None, channels);
    }
    let failed = scalars.iter().chain(&channels).any(|flag| flag == "fail");
    (
        Some(if failed { "fail" } else { "pass" }.to_owned()),
        channels,
    )
}

fn walk_flags(node: &serde_json::Value, scalars: &mut Vec<String>, channels: &mut Vec<String>) {
    match node {
        serde_json::Value::Object(map) => {
            for value in map.values() {
                walk_flags(value, scalars, channels);
            }
        }
        serde_json::Value::Array(items) => {
            let flags: Vec<String> = items
                .iter()
                .filter_map(|item| item.as_str())
                .map(|item| item.to_ascii_lowercase())
                .filter(|item| item == "pass" || item == "fail")
                .collect();
            if flags.len() == items.len() && flags.len() > channels.len() {
                *channels = flags;
            } else {
                for item in items {
                    walk_flags(item, scalars, channels);
                }
            }
        }
        serde_json::Value::String(text) => {
            let flag = text.to_ascii_lowercase();
            if flag == "pass" || flag == "fail" {
                scalars.push(flag);
            }
        }
        _ => {}
    }
}

/// Applied HV step from nozzle config, when both voltages are present.
pub fn hv_delta_v(text: &str) -> Option<f32> {
    let value: serde_json::Value = serde_json::from_str(text).ok()?;
    let mut start = None;
    let mut end = None;
    walk_delta(&value, &mut start, &mut end);
    match (start, end) {
        (Some(a), Some(b)) if (b - a).abs() > 0.0 => Some((b - a).abs()),
        _ => None,
    }
}

fn walk_delta(node: &serde_json::Value, start: &mut Option<f32>, end: &mut Option<f32>) {
    let serde_json::Value::Object(map) = node else {
        return;
    };
    for (key, child) in map {
        if key.ends_with("hv_transient_test/starting_voltage") {
            *start = child.as_f64().map(|value| value as f32);
        }
        if key.ends_with("hv_transient_test/ending_voltage") {
            *end = child.as_f64().map(|value| value as f32);
        }
        walk_delta(child, start, end);
    }
}

/// Expected primary-channel capacitance in pF, when the config records one.
pub fn hv_expected_pf(text: &str) -> Option<f32> {
    let value: serde_json::Value = serde_json::from_str(text).ok()?;
    find_suffix(
        &value,
        "hv_transient_primary_channel_test/expected_capacitance",
    )
    .or_else(|| find_suffix(&value, "integrator_result/expected_capacitance"))
}

fn find_suffix(node: &serde_json::Value, suffix: &str) -> Option<f32> {
    let serde_json::Value::Object(map) = node else {
        return None;
    };
    for (key, child) in map {
        if key.ends_with(suffix) {
            if let Some(value) = child.as_f64() {
                return Some(value as f32);
            }
        }
        if let Some(found) = find_suffix(child, suffix) {
            return Some(found);
        }
    }
    None
}

/// Nearest-neighbor resample of a row-major grid onto a new shape.
pub fn resample_nearest(source: &[f32], shape: [usize; 3], out_shape: [usize; 3]) -> Vec<f32> {
    let [nx, ny, nz] = shape;
    let [ox, oy, oz] = out_shape;
    let mut out = vec![0.0f32; ox * oy * oz];
    if nx == 0 || ny == 0 || nz == 0 {
        return out;
    }
    for z in 0..oz {
        for y in 0..oy {
            for x in 0..ox {
                let sx = (x * nx / ox).min(nx - 1);
                let sy = (y * ny / oy).min(ny - 1);
                let sz = (z * nz / oz).min(nz - 1);
                let from = sx + nx * (sy + ny * sz);
                let to = x + ox * (y + oy * z);
                if let Some(value) = source.get(from) {
                    out[to] = *value;
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sk_req_010_remap_calibration_and_beam_mask() {
        assert!((remap(64.5, 1.0, 128.0, -128.0, 128.0)).abs() < 1.0);
        assert!((remap_g2_raw(64.5)).abs() < 1e-3);
        assert!((remap_g3_raw(64.5)).abs() < 1e-3);
        let ic2 = g2_ic2_mm(&[0.0, 10.0], &[64.5, 64.5 + 10.0 / G2_MM_PER_STRIP]);
        assert!(ic2[1] > 5.0);
        let factor = calibration_factor(&[2.0, 4.0], &[1.0, 2.0]).unwrap();
        assert!((factor - 2.0).abs() < 1e-5);
        let err = dose_error_pct(&[110.0], &[100.0]);
        assert!((err[0] - 10.0).abs() < 1e-4);
        let ratio = dose_ratio_pct(&[1.1], &[1.0]);
        assert!((ratio[0] - 10.0).abs() < 1e-3);
        assert_eq!(
            beam_on_mask(&[0.0, 1.0, 0.0, 1.0, 1.0]),
            vec![false, true, false, true, true]
        );
        let mut filtered = std::collections::BTreeMap::from([
            ("y".to_owned(), vec![1.0, 2.0]),
            ("beam_on".to_owned(), vec![1.0, 0.0]),
        ]);
        crate::apply_mask(
            &mut filtered,
            &[crate::Segment::Beam {
                state: crate::BeamGate::On,
            }],
            &["y"],
        );
        assert!(filtered["y"][1].is_nan());
    }

    #[test]
    fn sk_req_023_line_fit_returns_slope_and_intercept() {
        let (slope, intercept) = linear_fit(&[0.0, 1.0, 2.0], &[1.0, 3.0, 5.0]).unwrap();
        assert!((slope - 2.0).abs() < 1e-4);
        assert!((intercept - 1.0).abs() < 1e-4);
    }

    #[test]
    fn sk_req_016_quantile_edges_and_histogram() {
        let edges = quantile_edges(&[1.0, 2.0, 3.0, 4.0], 2);
        assert!(edges.len() >= 2);
        assert!(edges[0] <= 1.0 + 1e-3);
        let (bins, counts) = histogram(&[0.0, 0.1, 0.9, 1.0], 2);
        assert_eq!(bins.len(), 3);
        assert_eq!(counts.iter().sum::<f32>(), 4.0);
    }

    #[test]
    fn sk_req_019_welch_sees_a_tone() {
        let fs = 1000.0f32;
        let signal: Vec<f32> = (0..256)
            .map(|i| (2.0 * std::f32::consts::PI * 50.0 * i as f32 / fs).sin())
            .collect();
        let (freqs, psd) = welch_psd(&signal, fs, 128, 0.5);
        let peak = psd
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert!((freqs[peak] - 50.0).abs() < 12.0);
    }

    #[test]
    fn sk_req_027_splat_peaks_on_the_spot() {
        let grid = splat_gaussians(
            &[[1.0, 1.0, 0.0, 1.0, 1.0]],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [3, 3, 1],
        );
        let center = grid[1 + 3];
        assert!(center > grid[0]);
        let image = mip_xy(&grid, [3, 3, 1]);
        assert_eq!(image[1 + 3], center);
    }

    #[test]
    fn sk_req_030_gamma_dvh_and_resample() {
        let shape = [2usize, 2, 1];
        let reference = vec![1.0, 1.0, 1.0, 0.0];
        let evaluated = vec![1.0, 1.0, 1.0, 0.0];
        let (_gamma, passed, scored) = gamma_index(
            &reference,
            &evaluated,
            shape,
            3.0,
            2.0,
            [1.0, 1.0, 1.0],
            0.1,
        );
        assert_eq!(scored, 3);
        assert_eq!(passed, 3);
        let shape = [9usize, 1, 1];
        let mut reference = vec![0.0f32; 9];
        reference[4] = 1.0;
        let mut near = vec![0.0f32; 9];
        near[5] = 1.0;
        let mut far = vec![0.0f32; 9];
        far[7] = 1.0;
        let (_g, passed, scored) =
            gamma_index(&reference, &near, shape, 3.0, 2.0, [1.0, 1.0, 1.0], 10.0);
        assert_eq!((scored, passed), (1, 1));
        let (gamma, passed, _) =
            gamma_index(&reference, &far, shape, 3.0, 2.0, [1.0, 1.0, 1.0], 10.0);
        assert_eq!(passed, 0);
        assert!((gamma[4] - 1.5).abs() < 1e-3, "{}", gamma[4]);
        let (edges, curve) = dvh(&[0.0, 1.0, 2.0], &[true, true, true], 2);
        assert!(edges.len() >= 2);
        assert!((curve[0] - 1.0).abs() < 1e-4);
        let out = resample_nearest(&[1.0, 2.0, 3.0, 4.0], [2, 2, 1], [1, 1, 1]);
        assert_eq!(out, vec![1.0]);
    }

    #[test]
    fn sk_req_033_spot_sums_hv_and_coverage() {
        assert_eq!(
            sums_by_spot_run(&[1.0, 1.0, 2.0, 2.0], &[1.0, 2.0, 3.0, 4.0]),
            vec![3.0, 7.0]
        );
        let time = [0.0, 1.0, 2.0];
        let current = [0.0, 2.0, 0.0];
        let pf = hv_capacitance_pf(&time, &current, -1.0, 3.0, 2.0).unwrap();
        assert!((pf - 1.0).abs() < 1e-3);
        let window = hv_step_window(&time, &current).unwrap();
        assert!(window.0 < 1.0 && window.1 > 1.0);
        assert_eq!(
            sums_by_spot_id(&[1.0, 2.0, 1.0], &[1.0, 10.0, 2.0]),
            vec![3.0, 10.0]
        );
        let covered = coverage_percent(&[10.0, 50.0, 90.0], &[0.0, 40.0, 80.0, 100.0]);
        assert!((covered[0] - 100.0).abs() < 1e-3);
        assert!((covered[1] - 200.0 / 3.0).abs() < 1e-3);
        assert!((covered[3] - 0.0).abs() < 1e-3);

        let (overall, channels) =
            hv_firmware_flags(r#"{"result":{"grade":"pass","strips":["pass","fail"]}}"#);
        assert_eq!(overall.as_deref(), Some("fail"));
        assert_eq!(channels, vec!["pass".to_owned(), "fail".to_owned()]);
        let cfg = r#"{"dose_controller/safety_test/hv_transient_test/starting_voltage":0,"dose_controller/safety_test/hv_transient_test/ending_voltage":80,"dose_controller/safety_test/hv_transient_primary_channel_test/expected_capacitance":12.5}"#;
        assert!((hv_delta_v(cfg).unwrap() - 80.0).abs() < 1e-3);
        assert!((hv_expected_pf(cfg).unwrap() - 12.5).abs() < 1e-3);
    }

    #[test]
    fn binned_box_and_quantile_centers_match_the_python_rules() {
        let stats = box_stats(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]).unwrap();
        assert!((stats.q1 - 3.25).abs() < 1e-4);
        assert!((stats.q3 - 7.75).abs() < 1e-4);
        assert!((stats.whisker_lo - 1.0).abs() < 1e-4);
        assert!((stats.whisker_hi - 10.0).abs() < 1e-4);
        assert!(stats.fliers.is_empty());
        let centers = assign_bin_centers(&[0.0, 9.9, 10.0, 20.0, -1.0, 21.0], &[0.0, 10.0, 20.0]);
        assert!((centers[0] - 5.0).abs() < 1e-4);
        assert!((centers[1] - 5.0).abs() < 1e-4);
        assert!((centers[2] - 15.0).abs() < 1e-4);
        assert!((centers[3] - 15.0).abs() < 1e-4);
        assert!(centers[4].is_nan());
        assert!(centers[5].is_nan());
    }
}
