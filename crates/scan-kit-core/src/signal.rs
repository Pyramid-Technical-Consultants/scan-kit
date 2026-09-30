//! Shared numeric meaning for the analysis views.
//!
//! File IO stays in `scan-kit-io`. These functions take columns they already hold.

/// G2 strip register 64.5 maps to 0 mm. Pitch matches the 1–128 → −128–128 mm map.
pub const G2_STRIP_CENTER: f32 = 64.5;
pub const G2_MM_PER_STRIP: f32 = 256.0 / 127.0;

/// G3 strip channel 64.5 maps to 0 mm at a 2 mm pitch.
pub const G3_STRIP_CENTER: f32 = 64.5;
pub const G3_STRIP_PITCH_MM: f32 = 2.0;

pub const MS_PER_SLICE: f32 = 1.0;
pub const MIN_SPILL_GAP_MS: f32 = 500.0;

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

pub fn cumsum(values: &[f32]) -> Vec<f32> {
    let mut total = 0.0f32;
    values
        .iter()
        .map(|value| {
            if value.is_finite() {
                total += value;
            }
            total
        })
        .collect()
}

/// Gate is on when the column is non-zero. An empty column is all off.
pub fn beam_on_mask(gate: &[f32]) -> Vec<bool> {
    gate.iter()
        .map(|value| *value != 0.0 && value.is_finite())
        .collect()
}

/// Half-open spill ranges. Off gaps shorter than `gap_ms` stay inside the spill.
pub fn spill_segments(beam_on: &[bool], gap_ms: f32, min_on_slices: usize) -> Vec<(usize, usize)> {
    if beam_on.is_empty() {
        return Vec::new();
    }
    let gap = ((gap_ms / MS_PER_SLICE).round() as usize).max(1);
    let mut segments = Vec::new();
    let mut in_spill = false;
    let mut start = 0usize;
    let mut off_count = 0usize;
    for (i, is_on) in beam_on.iter().copied().enumerate() {
        if is_on {
            if !in_spill {
                start = i;
                in_spill = true;
            }
            off_count = 0;
        } else if in_spill {
            off_count += 1;
            if off_count >= gap {
                let end = i - off_count + 1;
                if end - start >= min_on_slices {
                    segments.push((start, end));
                }
                in_spill = false;
                off_count = 0;
            }
        }
    }
    if in_spill && beam_on.len() - start >= min_on_slices {
        segments.push((start, beam_on.len()));
    }
    segments
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BeamState {
    All,
    On,
    Off,
}

/// Keep samples that match the beam state. Filtered samples become NaN.
pub fn filter_beam_state(values: &[f32], beam_on: &[bool], state: BeamState) -> Vec<f32> {
    values
        .iter()
        .enumerate()
        .map(|(i, value)| {
            let on = beam_on.get(i).copied().unwrap_or(false);
            let keep = match state {
                BeamState::All => true,
                BeamState::On => on,
                BeamState::Off => !on,
            };
            if keep {
                *value
            } else {
                f32::NAN
            }
        })
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

/// Single exponential `a * exp(-t / tau)` fit on positive samples. Returns `(a, tau)`.
pub fn fit_decay(time: &[f32], values: &[f32]) -> Option<(f32, f32)> {
    let mut n = 0.0f64;
    let mut sx = 0.0f64;
    let mut sy = 0.0f64;
    let mut sxx = 0.0f64;
    let mut sxy = 0.0f64;
    for (t, y) in time.iter().zip(values) {
        if t.is_finite() && y.is_finite() && *y > 0.0 {
            let x = f64::from(*t);
            let ly = (f64::from(*y)).ln();
            n += 1.0;
            sx += x;
            sy += ly;
            sxx += x * x;
            sxy += x * ly;
        }
    }
    if n < 2.0 {
        return None;
    }
    let denom = n * sxx - sx * sx;
    if denom.abs() < 1e-18 {
        return None;
    }
    let slope = (n * sxy - sx * sy) / denom;
    let intercept = (sy - slope * sx) / n;
    if slope >= 0.0 {
        return None;
    }
    Some((intercept.exp() as f32, (-1.0 / slope) as f32))
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

/// Global gamma on a coarse offset set. Returns `(gamma, passed, evaluated)`.
pub fn gamma_index(
    reference: &[f32],
    evaluated: &[f32],
    shape: [usize; 3],
    dose_percent: f32,
    distance_mm: f32,
    spacing: [f32; 3],
    cutoff: f32,
) -> (Vec<f32>, u32, u32) {
    let [nx, ny, nz] = shape;
    let n = nx * ny * nz;
    let mut gamma = vec![0.0f32; n.min(reference.len()).min(evaluated.len())];
    let dose_tol = dose_percent.max(1e-6) / 100.0;
    let mut norm = 0.0f32;
    for value in evaluated.iter().take(gamma.len()) {
        if value.is_finite() {
            norm = norm.max(*value);
        }
    }
    let dd = (dose_tol * norm).max(1e-6);
    let reach = [
        (distance_mm / spacing[0]).ceil() as i32,
        (distance_mm / spacing[1]).ceil() as i32,
        (distance_mm / spacing[2]).ceil() as i32,
    ];
    let mut passed = 0u32;
    let mut scored = 0u32;
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let i = x + nx * (y + ny * z);
                if i >= gamma.len() {
                    continue;
                }
                let r = reference[i];
                if !r.is_finite() || r <= 0.0 || r < cutoff {
                    continue;
                }
                scored += 1;
                let mut best = 4.0f32;
                for dz in -reach[2]..=reach[2] {
                    for dy in -reach[1]..=reach[1] {
                        for dx in -reach[0]..=reach[0] {
                            let xx = x as i32 + dx;
                            let yy = y as i32 + dy;
                            let zz = z as i32 + dz;
                            if xx < 0
                                || yy < 0
                                || zz < 0
                                || xx >= nx as i32
                                || yy >= ny as i32
                                || zz >= nz as i32
                            {
                                continue;
                            }
                            let j = xx as usize + nx * (yy as usize + ny * zz as usize);
                            let dist2 = (dx as f32 * spacing[0] / distance_mm).powi(2)
                                + (dy as f32 * spacing[1] / distance_mm).powi(2)
                                + (dz as f32 * spacing[2] / distance_mm).powi(2);
                            if dist2 >= best {
                                continue;
                            }
                            let dose = (evaluated[j] - r) / dd;
                            best = best.min(dist2 + dose * dose);
                        }
                    }
                }
                let g = best.sqrt();
                gamma[i] = g;
                if g <= 1.0 {
                    passed += 1;
                }
            }
        }
    }
    (gamma, passed, scored)
}

/// True once a sample is `settle` steps after the last change larger than `tol`.
pub fn settled_after_step(samples: &[f32], settle: usize, tol: f32) -> Vec<bool> {
    if settle == 0 {
        return vec![true; samples.len()];
    }
    let mut start = 0usize;
    let mut mask = vec![false; samples.len()];
    for index in 0..samples.len() {
        if index > 0 && (samples[index] - samples[index - 1]).abs() > tol {
            start = index;
        }
        mask[index] = index >= start + settle;
    }
    mask
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

/// Counts of finite pairs on a `bins` by `bins` grid. `(counts, xmin, xmax, ymin, ymax)`.
pub fn density_counts(
    xs: &[f32],
    ys: &[f32],
    bins: usize,
) -> Option<(Vec<f32>, f32, f32, f32, f32)> {
    let bins = bins.max(2);
    let mut xmin = f32::MAX;
    let mut xmax = f32::MIN;
    let mut ymin = f32::MAX;
    let mut ymax = f32::MIN;
    let mut pairs = 0usize;
    for (x, y) in xs.iter().zip(ys) {
        if x.is_finite() && y.is_finite() {
            xmin = xmin.min(*x);
            xmax = xmax.max(*x);
            ymin = ymin.min(*y);
            ymax = ymax.max(*y);
            pairs += 1;
        }
    }
    if pairs == 0 {
        return None;
    }
    if (xmax - xmin).abs() < 1e-6 {
        xmax = xmin + 1.0;
    }
    if (ymax - ymin).abs() < 1e-6 {
        ymax = ymin + 1.0;
    }
    let mut counts = vec![0.0f32; bins * bins];
    let dx = xmax - xmin;
    let dy = ymax - ymin;
    for (x, y) in xs.iter().zip(ys) {
        if x.is_finite() && y.is_finite() {
            let ix = (((x - xmin) / dx) * bins as f32) as usize;
            let iy = (((y - ymin) / dy) * bins as f32) as usize;
            let ix = ix.min(bins - 1);
            let iy = iy.min(bins - 1);
            counts[ix + bins * iy] += 1.0;
        }
    }
    Some((counts, xmin, xmax, ymin, ymax))
}

/// Odd circular-arc fit `sin θ = c0 + c1·B + c3·B³`, with θ in mrad.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ArcFit {
    pub c0: f32,
    pub c1: f32,
    pub c3: f32,
}

pub fn arc_fit(field: &[f32], angle_mrad: &[f32]) -> Option<ArcFit> {
    let mut ata = [[0.0f64; 3]; 3];
    let mut aty = [0.0f64; 3];
    let mut n = 0usize;
    let mut lo = f32::MAX;
    let mut hi = f32::MIN;
    for (b, angle) in field.iter().zip(angle_mrad) {
        if !b.is_finite() || !angle.is_finite() {
            continue;
        }
        let y = (angle / 1000.0).sin() * 1000.0;
        if !y.is_finite() {
            continue;
        }
        lo = lo.min(*b);
        hi = hi.max(*b);
        let basis = [1.0, f64::from(*b), f64::from(b.powi(3))];
        for i in 0..3 {
            aty[i] += basis[i] * f64::from(y);
            for j in 0..3 {
                ata[i][j] += basis[i] * basis[j];
            }
        }
        n += 1;
    }
    if n < 12 || hi <= lo {
        return None;
    }
    let coef = solve3(ata, aty)?;
    Some(ArcFit {
        c0: coef[0] as f32,
        c1: coef[1] as f32,
        c3: coef[2] as f32,
    })
}

pub fn arc_predict(fit: ArcFit, field: f32) -> f32 {
    let sin_scaled = fit.c0 + fit.c1 * field + fit.c3 * field.powi(3);
    (sin_scaled / 1000.0).clamp(-1.0, 1.0).asin() * 1000.0
}

fn solve3(mut a: [[f64; 3]; 3], mut b: [f64; 3]) -> Option<[f64; 3]> {
    for col in 0..3 {
        let mut pivot = col;
        for row in col + 1..3 {
            if a[row][col].abs() > a[pivot][col].abs() {
                pivot = row;
            }
        }
        if a[pivot][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        let div = a[col][col];
        for value in &mut a[col][col..] {
            *value /= div;
        }
        b[col] /= div;
        let pivot_row = a[col];
        for row in 0..3 {
            if row == col {
                continue;
            }
            let factor = a[row][col];
            for (value, pivot) in a[row][col..].iter_mut().zip(&pivot_row[col..]) {
                *value -= factor * pivot;
            }
            b[row] -= factor * b[col];
        }
    }
    Some(b)
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

/// Per-sample background from off-beam samples, plus the global background and peak.
///
/// A short or flat trace keeps a constant background. Otherwise the background
/// is a median filter of the off-beam samples, interpolated across the trace.
pub fn sliding_background(
    signal: &[f32],
    threshold_frac: f32,
    rolling_window: usize,
) -> (Vec<f32>, f32, f32) {
    let finite: Vec<f32> = signal
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    if finite.is_empty() {
        return (vec![0.0; signal.len()], 0.0, 0.0);
    }
    let low_edge = percentile(&finite, 0.25);
    let low: Vec<f32> = finite
        .into_iter()
        .filter(|value| *value <= low_edge)
        .collect();
    let bg_global = median_finite(&low).unwrap_or(low_edge);
    let peak = percentile(
        &signal
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .collect::<Vec<_>>(),
        0.99,
    );
    let flat = vec![bg_global; signal.len()];
    if peak - bg_global < 1.0 || rolling_window < 3 {
        return (flat, bg_global, peak);
    }
    let thresh = threshold_frac * (peak - bg_global);
    let mut off_x = Vec::new();
    let mut off_y = Vec::new();
    for (i, value) in signal.iter().enumerate() {
        if value.is_finite() && *value - bg_global <= thresh {
            off_x.push(i as f32);
            off_y.push(*value);
        }
    }
    if off_x.len() < rolling_window {
        return (flat, bg_global, peak);
    }
    let smoothed = median_filter_reflect(&off_y, rolling_window);
    (
        interp_clamped(&off_x, &smoothed, signal.len()),
        bg_global,
        peak,
    )
}

/// Falling-edge indices where the signal stays on for 2 samples and off for 9.
pub fn beam_off_edges(signal: &[f32]) -> Vec<usize> {
    const MIN_ON: usize = 2;
    const POST: usize = 9;
    const MIN_SPAN: f32 = 1e-4;
    let finite: Vec<f32> = signal
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    if finite.len() < MIN_ON + POST {
        return Vec::new();
    }
    let fill = median_finite(&finite).unwrap_or(0.0);
    let clean: Vec<f32> = signal
        .iter()
        .map(|value| if value.is_finite() { *value } else { fill })
        .collect();
    let (bg, bg_global, peak) = sliding_background(&clean, 0.10, 200);
    if peak - bg_global < MIN_SPAN {
        return Vec::new();
    }
    let thresh = 0.10 * (peak - bg_global);
    let on: Vec<bool> = clean
        .iter()
        .zip(&bg)
        .map(|(value, base)| value - base > thresh)
        .collect();
    let mut edges = Vec::new();
    for idx in 1..on.len() {
        if on[idx - 1] && !on[idx] {
            if idx < MIN_ON || idx + POST > on.len() {
                continue;
            }
            if on[idx - MIN_ON..idx].iter().any(|flag| !*flag) {
                continue;
            }
            if on[idx..idx + POST].iter().any(|flag| *flag) {
                continue;
            }
            edges.push(idx);
        }
    }
    edges
}

fn median_filter_reflect(values: &[f32], size: usize) -> Vec<f32> {
    let n = values.len();
    if n == 0 || size <= 1 {
        return values.to_vec();
    }
    let half = size / 2;
    let mut window = Vec::with_capacity(size);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        window.clear();
        let start = i as isize - half as isize;
        for step in 0..size {
            window.push(values[reflect_index(start + step as isize, n)]);
        }
        out.push(median_finite(&window).unwrap_or(0.0));
    }
    out
}

fn reflect_index(mut index: isize, n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    let len = n as isize;
    for _ in 0..8 {
        if index < 0 {
            index = -index;
        } else if index >= len {
            index = 2 * len - 2 - index;
        } else {
            break;
        }
    }
    index.clamp(0, len - 1) as usize
}

fn interp_clamped(xs: &[f32], ys: &[f32], n: usize) -> Vec<f32> {
    if xs.is_empty() || ys.is_empty() {
        return vec![0.0; n];
    }
    let mut out = Vec::with_capacity(n);
    let mut j = 0usize;
    for i in 0..n {
        let x = i as f32;
        if x <= xs[0] {
            out.push(ys[0]);
            continue;
        }
        if x >= xs[xs.len() - 1] {
            out.push(ys[ys.len() - 1]);
            continue;
        }
        while j + 1 < xs.len() && xs[j + 1] < x {
            j += 1;
        }
        let x0 = xs[j];
        let x1 = xs[j + 1];
        let span = (x1 - x0).max(1e-6);
        let t = (x - x0) / span;
        out.push(ys[j] + t * (ys[j + 1] - ys[j]));
    }
    out
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

fn percentile(values: &[f32], p: f32) -> f32 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let index = ((sorted.len() - 1) as f32 * p).round() as usize;
    sorted[index.min(sorted.len() - 1)]
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
        let mask = beam_on_mask(&[0.0, 1.0, 0.0, 1.0, 1.0]);
        assert_eq!(spill_segments(&mask, 1.0, 2), vec![(3, 5)]);
        let filtered = filter_beam_state(&[1.0, 2.0], &[true, false], BeamState::On);
        assert!(filtered[1].is_nan());
    }

    #[test]
    fn sk_req_023_settled_mask_waits_out_a_command_step() {
        let cmd = [0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0];
        assert_eq!(
            settled_after_step(&cmd, 3, 1e-6),
            vec![false, false, false, false, true, false, false, false, true, true, true]
        );
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
    fn sk_req_021_decay_fit_recovers_tau() {
        let time: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let values: Vec<f32> = time.iter().map(|t| 4.0 * (-t / 5.0).exp()).collect();
        let (_a, tau) = fit_decay(&time, &values).unwrap();
        assert!((tau - 5.0).abs() < 0.2);
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
        let (edges, curve) = dvh(&[0.0, 1.0, 2.0], &[true, true, true], 2);
        assert!(edges.len() >= 2);
        assert!((curve[0] - 1.0).abs() < 1e-4);
        let out = resample_nearest(&[1.0, 2.0, 3.0, 4.0], [2, 2, 1], [1, 1, 1]);
        assert_eq!(out, vec![1.0]);
    }

    #[test]
    fn sk_req_033_spot_sums_arc_hv_and_ramp_edges() {
        assert_eq!(
            sums_by_spot_run(&[1.0, 1.0, 2.0, 2.0], &[1.0, 2.0, 3.0, 4.0]),
            vec![3.0, 7.0]
        );
        let field: Vec<f32> = (-6..=6).map(|v| v as f32).collect();
        let angle: Vec<f32> = field
            .iter()
            .map(|b| (b / 1000.0).clamp(-1.0, 1.0).asin() * 1000.0)
            .collect();
        let fit = arc_fit(&field, &angle).unwrap();
        assert!(fit.c0.abs() < 1e-3);
        assert!((fit.c1 - 1.0).abs() < 1e-3);
        assert!(fit.c3.abs() < 1e-4);
        assert!((arc_predict(fit, 2.0) - (0.002f32).asin() * 1000.0).abs() < 1e-2);

        let time = [0.0, 1.0, 2.0];
        let current = [0.0, 2.0, 0.0];
        let pf = hv_capacitance_pf(&time, &current, -1.0, 3.0, 2.0).unwrap();
        assert!((pf - 1.0).abs() < 1e-3);
        let window = hv_step_window(&time, &current).unwrap();
        assert!(window.0 < 1.0 && window.1 > 1.0);

        let mut pulse = vec![1.0f32; 20];
        pulse.extend(std::iter::repeat_n(0.0, 20));
        assert_eq!(beam_off_edges(&pulse), vec![20]);
        let mut drifted: Vec<f32> = (0..800).map(|i| i as f32 * 0.02).collect();
        for sample in &mut drifted[500..530] {
            *sample += 50.0;
        }
        assert!(beam_off_edges(&drifted)
            .iter()
            .any(|edge| (520..545).contains(edge)));
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
