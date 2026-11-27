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
        let mid = vals.len() / 2;
        crate::stats::select_rank(&mut vals, mid)
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
    if sum_d <= 0.0 {
        return None;
    }
    // A denormal-tiny denominator casts to inf — refuse it rather than
    // spraying inf/NaN through the scaled column.
    let factor = (sum_t / sum_d) as f32;
    factor.is_finite().then_some(factor)
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
            if d.is_finite() && r.is_finite() && *r != 0.0 {
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
            if d.is_finite() && t.is_finite() && *t != 0.0 {
                (d - t) / t * 100.0
            } else {
                f32::NAN
            }
        })
        .collect()
}

/// `delivered / target`. A non-finite value or a zero target becomes NaN.
pub fn dose_per_mu(delivered: &[f32], target: &[f32]) -> Vec<f32> {
    delivered
        .iter()
        .zip(target)
        .map(|(d, t)| {
            if d.is_finite() && t.is_finite() && *t != 0.0 {
                d / t
            } else {
                f32::NAN
            }
        })
        .collect()
}

/// Distance from the origin. A non-finite axis becomes NaN.
pub fn radial_mm(x: &[f32], y: &[f32]) -> Vec<f32> {
    x.iter()
        .zip(y)
        .map(|(x, y)| {
            if x.is_finite() && y.is_finite() {
                x.hypot(*y)
            } else {
                f32::NAN
            }
        })
        .collect()
}

/// Subtract the mean of the finite samples. A non-finite sample stays NaN.
///
/// ponytail: each row weighs the same, so a long timeslice spot outweighs a short
/// one. The upgrade is a mean of per-spot centroids.
pub fn remove_mean(values: &[f32]) -> Vec<f32> {
    let mut sum = 0.0f64;
    let mut count = 0u32;
    for value in values {
        if value.is_finite() {
            sum += f64::from(*value);
            count += 1;
        }
    }
    if count == 0 {
        return vec![f32::NAN; values.len()];
    }
    let mean = (sum / f64::from(count)) as f32;
    values
        .iter()
        .map(|value| {
            if value.is_finite() {
                value - mean
            } else {
                f32::NAN
            }
        })
        .collect()
}

/// Error as a percent of the expected width. Expected width is `measured - error`.
pub fn sigma_error_pct(measured: &[f32], error_mm: &[f32]) -> Vec<f32> {
    measured
        .iter()
        .zip(error_mm)
        .map(|(measured, error)| {
            if measured.is_finite() && error.is_finite() {
                let expected = measured - error;
                if expected > 0.0 {
                    error / expected * 100.0
                } else {
                    f32::NAN
                }
            } else {
                f32::NAN
            }
        })
        .collect()
}
/// Gate is on when the column clears half scale, like beam_keep and
/// plateau_mean. A non-finite or fractional-low sample is off.
pub fn beam_on_mask(gate: &[f32]) -> Vec<bool> {
    gate.iter()
        .map(|value| value.is_finite() && *value > 0.5)
        .collect()
}
/// Quantile bin edges. A single finite value becomes a unit-width bin around it.
// Callers hand us bins from tool arguments — an unclamped count is an
// unbounded allocation. 1M bins is already far past any meaningful plot.
const MAX_BINS: usize = 1 << 20;

pub fn quantile_edges(values: &[f32], n_bins: usize) -> Vec<f32> {
    let n_bins = n_bins.min(MAX_BINS);
    let mut finite: Vec<f32> = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    if finite.is_empty() {
        return vec![0.0, 1.0];
    }
    let n_bins = n_bins.max(1);
    if finite.iter().all(|value| *value == finite[0]) {
        let lo = finite[0];
        let pad = (lo.abs() * f32::EPSILON).max(0.5);
        // |lo| within a pad of f32::MAX overflows to +-inf — clamp like
        // padded_span does.
        return vec![(lo - pad).max(-f32::MAX), (lo + pad).min(f32::MAX)];
    }
    let qs: Vec<f32> = (0..=n_bins)
        .map(|step| step as f32 / n_bins as f32)
        .collect();
    let mut edges = crate::stats::quantiles_at(&mut finite, &qs);
    // Relative epsilon — an absolute 1e-6 swallows every edge in a column
    // whose data lives at a smaller scale (charge integrals, ratios).
    edges.dedup_by(|a, b| (*a - *b).abs() <= 1e-9 * a.abs().max(b.abs()).max(1e-30));
    if edges.len() < 2 {
        let lo = edges.first().copied().unwrap_or(0.0);
        let pad = (lo.abs() * f32::EPSILON).max(0.5);
        return vec![(lo - pad).max(-f32::MAX), (lo + pad).min(f32::MAX)];
    }
    edges
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
    let quartiles = crate::stats::quantiles_at(&mut sorted, &[0.25, 0.5, 0.75]);
    let [q1, median, q3] = [quartiles[0], quartiles[1], quartiles[2]];
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
    let bins = bins.min(MAX_BINS);
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
    // "Flat" is relative to the data's scale — an absolute floor calls a
    // 1e-13-wide column flat and bins every distinct value into one bucket.
    if (hi - lo).abs() <= 1e-12 * hi.abs().max(lo.abs()).max(1e-30) {
        // The pad must survive the f32 ulp at the data's magnitude — an
        // absolute 0.5 is unrepresentable above 2^23 and leaves span 0,
        // piling every count into bin 0. Same fix uniform_edges carries.
        let pad = (hi.abs().max(lo.abs()) * f32::EPSILON).max(0.5);
        lo = (lo - pad).max(-f32::MAX);
        hi = (hi + pad).min(f32::MAX);
    }
    // Span math in f64: a column can legitimately span more than f32::MAX,
    // which would turn every edge NaN and pile all counts into bin 0.
    let span = f64::from(hi) - f64::from(lo);
    let mut edges = Vec::with_capacity(bins + 1);
    for i in 0..=bins {
        edges.push((f64::from(lo) + span * i as f64 / bins as f64) as f32);
    }
    // f32 counters freeze at 2^24 — adding 1.0 rounds away. Count exactly
    // in usize and convert once; large volumes cross that threshold.
    let mut counts = vec![0usize; bins];
    for value in finite {
        let mut index = ((f64::from(value) - f64::from(lo)) / span * bins as f64) as usize;
        if index >= bins {
            index = bins - 1;
        }
        counts[index] += 1;
    }
    let counts = counts.into_iter().map(|count| count as f32).collect();
    (edges, counts)
}

/// Welch PSD. Timeline asks for 4096-sample segments, 50% overlap, and `fs` 1000.
///
/// The segment length is a power of two. A shorter trace uses the largest such
/// length that fits, and at least 16 samples.
pub fn welch_psd(
    signal: &[f32],
    fs: f32,
    mut seg_len: usize,
    overlap: f32,
) -> (Vec<f32>, Vec<f32>) {
    if !fs.is_finite() || fs <= 0.0 {
        return (Vec::new(), Vec::new());
    }
    let centered: Vec<f32> = signal
        .iter()
        .map(|value| if value.is_finite() { *value } else { 0.0 })
        .collect();
    if centered.len() < seg_len {
        seg_len = centered.len();
    }
    if !seg_len.is_power_of_two() {
        seg_len = seg_len.next_power_of_two() / 2;
    }
    if seg_len < 16 {
        return (Vec::new(), Vec::new());
    }
    // overlap >= 1 (or NaN) would collapse the step to one sample and turn
    // the loop quadratic — clamp it into the range the API promises.
    let overlap = if overlap.is_finite() {
        overlap.clamp(0.0, 0.99)
    } else {
        0.0
    };
    let step = ((seg_len as f32) * (1.0 - overlap)).round().max(1.0) as usize;
    let window = hanning(seg_len);
    let win_power: f32 = window.iter().map(|w| w * w).sum();
    let mut windowed = vec![0.0f32; seg_len];
    let mut psd = vec![0.0f32; seg_len / 2 + 1];
    let mut count = 0usize;
    let mut start = 0usize;
    while start + seg_len <= centered.len() {
        // Detrend 'constant' per segment like scipy — the global mean lets a
        // baseline step leak broadband into the lowest bins.
        let segment = &centered[start..start + seg_len];
        let seg_mean = finite_mean(segment);
        for (out, (sample, weight)) in windowed.iter_mut().zip(segment.iter().zip(&window)) {
            *out = (sample - seg_mean) * weight;
        }
        for (bin, power) in psd.iter_mut().zip(rfft_power(&windowed)) {
            *bin += power;
        }
        count += 1;
        start += step;
    }
    if count == 0 {
        return (Vec::new(), Vec::new());
    }
    // scipy welch divides by fs * sum(w^2) per segment — without fs every
    // PSD value reads fs times too large.
    let scale = fs * (count as f32) * win_power.max(1e-12);
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

/// One-sided power `|DFT|^2` for a power-of-two length. The sign is the analysis transform.
fn rfft_power(samples: &[f32]) -> Vec<f32> {
    let n = samples.len();
    debug_assert!(n.is_power_of_two() && n >= 2);
    let mut re = samples.to_vec();
    let mut im = vec![0.0f32; n];
    let bits = n.trailing_zeros();
    for i in 0..n {
        let j = i.reverse_bits() >> (usize::BITS - bits);
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut tw_re = vec![0.0f32; n / 2];
    let mut tw_im = vec![0.0f32; n / 2];
    let mut len = 2usize;
    while len <= n {
        let half = len / 2;
        let turn = -2.0 * std::f64::consts::PI / len as f64;
        let step_re = turn.cos();
        let step_im = turn.sin();
        tw_re[0] = 1.0;
        tw_im[0] = 0.0;
        let mut w_re = 1.0f64;
        let mut w_im = 0.0f64;
        for k in 1..half {
            let next_re = w_re * step_re - w_im * step_im;
            w_im = w_re * step_im + w_im * step_re;
            w_re = next_re;
            tw_re[k] = w_re as f32;
            tw_im[k] = w_im as f32;
        }
        let mut i = 0usize;
        while i < n {
            for k in 0..half {
                let v_re = re[i + k + half] * tw_re[k] - im[i + k + half] * tw_im[k];
                let v_im = re[i + k + half] * tw_im[k] + im[i + k + half] * tw_re[k];
                let u_re = re[i + k];
                let u_im = im[i + k];
                re[i + k] = u_re + v_re;
                im[i + k] = u_im + v_im;
                re[i + k + half] = u_re - v_re;
                im[i + k + half] = u_im - v_im;
            }
            i += len;
        }
        len <<= 1;
    }
    (0..n / 2 + 1)
        .map(|k| re[k] * re[k] + im[k] * im[k])
        .collect()
}
/// Deposit isotropic Gaussians onto a regular grid. `spots` are `(x, y, z, weight, sigma)`.
// Lattice contract: voxel `i` is centered at `origin + i*spacing` — the
// DICOM ImagePosition convention, NOT the corner-origin convention of
// dose::Volume. Mixing the two misregisters every deposit by half a voxel.
pub fn splat_gaussians(
    spots: &[[f32; 5]],
    origin: [f32; 3],
    spacing: [f32; 3],
    shape: [usize; 3],
) -> Vec<f32> {
    let [nx, ny, nz] = shape;
    let Some(len) = nx.checked_mul(ny).and_then(|n| n.checked_mul(nz)) else {
        return Vec::new();
    };
    let mut grid = vec![0.0f32; len];
    if spacing.iter().any(|s| !s.is_finite() || *s <= 0.0) {
        return grid;
    }
    for spot in spots {
        let [x, y, z, weight, sigma] = *spot;
        if !x.is_finite()
            || !y.is_finite()
            || !z.is_finite()
            || !sigma.is_finite()
            || sigma <= 0.0
            || !weight.is_finite()
        {
            continue;
        }
        // 3σ in millimetres, converted to voxels per axis — spacing < 1 mm
        // needs more voxels to cover the kernel, and clamping to the axis
        // keeps a huge σ bounded (nothing outside the grid is written).
        let rx = ((3.0 * sigma / spacing[0]).ceil() as i32).min(nx as i32);
        let ry = ((3.0 * sigma / spacing[1]).ceil() as i32).min(ny as i32);
        let rz = ((3.0 * sigma / spacing[2]).ceil() as i32).min(nz as i32);
        let cx = ((x - origin[0]) / spacing[0]).round() as i32;
        let cy = ((y - origin[1]) / spacing[1]).round() as i32;
        let cz = ((z - origin[2]) / spacing[2]).round() as i32;
        let inv = 1.0 / (2.0 * sigma * sigma);
        for dz in -rz..=rz {
            for dy in -ry..=ry {
                for dx in -rx..=rx {
                    let ix = cx.saturating_add(dx);
                    let iy = cy.saturating_add(dy);
                    let iz = cz.saturating_add(dz);
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
    let Some(pixels) = nx.checked_mul(ny) else {
        return Vec::new();
    };
    let mut image = vec![0.0f32; pixels];
    // A grid shorter than shape promises returns zeros, matching the
    // accumulate_z/scan_axes length checks.
    if grid.len() < nx.saturating_mul(ny).saturating_mul(nz) {
        return image;
    }
    // Seed from the z=0 slice — a zero seed would floor an all-negative
    // grid's true maximum.
    if nz > 0 {
        image.copy_from_slice(&grid[..pixels]);
    }
    for z in 1..nz {
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
    // Centered f64 sums — n*sxx - sx*sx in f32 cancels the true denominator
    // at large x offsets, and an absolute epsilon on den rejects any column
    // at coulomb scale.
    let mut n = 0.0f64;
    let mut sx = 0.0f64;
    let mut sy = 0.0f64;
    for (x, y) in xs.iter().zip(ys) {
        if x.is_finite() && y.is_finite() {
            n += 1.0;
            sx += f64::from(*x);
            sy += f64::from(*y);
        }
    }
    if n < 2.0 {
        return None;
    }
    let x_bar = sx / n;
    let y_bar = sy / n;
    let mut sxx = 0.0f64;
    let mut sxy = 0.0f64;
    for (x, y) in xs.iter().zip(ys) {
        if x.is_finite() && y.is_finite() {
            let dx = f64::from(*x) - x_bar;
            sxx += dx * dx;
            sxy += dx * (f64::from(*y) - y_bar);
        }
    }
    if !sxx.is_finite() || sxx <= f64::MIN_POSITIVE {
        return None;
    }
    let slope = sxy / sxx;
    let intercept = y_bar - slope * x_bar;
    let (slope, intercept) = (slope as f32, intercept as f32);
    // A huge y can push the numerator to inf while den stays finite.
    if !slope.is_finite() || !intercept.is_finite() {
        return None;
    }
    Some((slope, intercept))
}

/// Dose-volume histogram as cumulative volume fraction above each dose edge.
pub fn dvh(dose: &[f32], mask: &[bool], bins: usize) -> (Vec<f32>, Vec<f32>) {
    let selected: Vec<f32> = dose
        .iter()
        // A mask shorter than the dose means the tail is not selected —
        // defaulting true silently counted it into the coverage volume.
        .zip(mask.iter().chain(std::iter::repeat(&false)))
        .filter_map(|(value, keep)| (*keep && value.is_finite() && *value >= 0.0).then_some(*value))
        .collect();
    let (edges, counts) = histogram(&selected, bins);
    let total = counts.iter().sum::<f32>();
    let mut cumulative = Vec::with_capacity(counts.len());
    let mut above = total;
    for count in &counts {
        // An empty selection must not read as 100% coverage.
        cumulative.push(if total == 0.0 { 0.0 } else { above / total });
        above -= count;
    }
    (edges, cumulative)
}

/// Sum `current` for each spot id, in the order each id first appears.
pub fn sums_by_spot_id(spot: &[f32], current: &[f32]) -> Vec<f32> {
    // i64: `as i32` saturates, folding every id past ~2.1e9 into one bucket.
    let mut index = std::collections::BTreeMap::<i64, usize>::new();
    let mut sums: Vec<f64> = Vec::new();
    for (spot_id, value) in spot.iter().zip(current) {
        if !spot_id.is_finite() || !value.is_finite() {
            continue;
        }
        let rounded = spot_id.round();
        // A value past i64 range saturates the cast — distinct ids would
        // merge into one bucket (same guard as spot_delivery_ms).
        if rounded.abs() >= 9e18 {
            continue;
        }
        let id = rounded as i64;
        if let Some(slot) = index.get(&id) {
            sums[*slot] += f64::from(*value);
        } else {
            index.insert(id, sums.len());
            sums.push(f64::from(*value));
        }
    }
    sums.into_iter().map(|sum| sum as f32).collect()
}

/// Sum `current` over each contiguous run of the same spot id.
pub fn sums_by_spot_run(spot: &[f32], current: &[f32]) -> Vec<f32> {
    let n = spot.len().min(current.len());
    let mut out = Vec::new();
    let mut acc = 0.0f64;
    let mut have = false;
    let mut last = 0i64;
    for i in 0..n {
        if !spot[i].is_finite() || !current[i].is_finite() {
            continue;
        }
        let rounded = spot[i].round();
        // Past i64 range the cast saturates — a run boundary at such an id
        // would silently merge into the previous run.
        if rounded.abs() >= 9e18 {
            continue;
        }
        let id = rounded as i64;
        if have && id != last {
            out.push(acc as f32);
            acc = 0.0;
        }
        acc += f64::from(current[i]);
        last = id;
        have = true;
    }
    if have {
        out.push(acc as f32);
    }
    out
}

/// 0.5% tails, then a slim margin. One spike does not set the axis, and a flat
/// trace stays near its value.
pub fn robust_limits(samples: &[f32]) -> Option<(f32, f32)> {
    let mut values: Vec<f32> = samples
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    robust_limits_finite(&mut values)
}

/// Same margins as [`robust_limits`]. `values` are already finite, and this reorders them.
pub fn robust_limits_finite(values: &mut [f32]) -> Option<(f32, f32)> {
    if values.is_empty() {
        return None;
    }
    let (lo, hi) = if values.len() == 1 {
        (values[0], values[0])
    } else {
        let last = values.len() - 1;
        let rank = |portion: f32| ((last as f32) * portion).round() as usize;
        crate::stats::select_ranks(values, rank(0.005).min(last), rank(0.995).min(last))
    };
    let scale = hi.abs().max(lo.abs());
    // An absolute 1e-6 floor swallows charge-integral columns that live at
    // nC/pC scale (1e-9..1e-12) — the trace would pin to 0. Scale the floor.
    let pad = ((hi - lo) * 0.02)
        .max(scale * 0.02)
        .max((scale * 1.0e-3).max(1.0e-12));
    Some(((lo - pad).max(-f32::MAX), (hi + pad).min(f32::MAX)))
}

/// Time span plus a small margin. The floor is one 1 ms sample, and the start stays at 0.
pub fn time_window(lo: f32, hi: f32) -> (f32, f32) {
    // The absolute 1 ms floor is below the f32 ulp for |t| past ~2^14 — a
    // flat large timestamp would return a zero-width window.
    let pad = ((hi - lo) * 0.02)
        .max(0.001)
        .max(hi.abs().max(lo.abs()) * f32::EPSILON);
    ((lo - pad).max(0.0), (hi + pad).min(f32::MAX))
}

pub fn median_finite(values: &[f32]) -> Option<f32> {
    let mut kept: Vec<f32> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if kept.is_empty() {
        return None;
    }
    Some(crate::stats::median_unstable(&mut kept))
}

pub fn trapz(time: &[f32], values: &[f32]) -> f32 {
    let mut acc = 0.0f64;
    for i in 1..time.len().min(values.len()) {
        let dt = time[i] - time[i - 1];
        if dt.is_finite() && values[i].is_finite() && values[i - 1].is_finite() {
            acc += 0.5 * (f64::from(values[i]) + f64::from(values[i - 1])) * f64::from(dt);
        }
    }
    acc as f32
}

/// Time window around the largest HV excursion, padded by 40 ms.
pub fn hv_step_window(time: &[f32], current: &[f32]) -> Option<(f32, f32)> {
    let n = time.len().min(current.len());
    if n == 0 {
        return None;
    }
    let base = median_finite(&current[..n])?;
    let mut peak_i = 0usize;
    let mut peak = 0.0f32;
    for (i, value) in current[..n].iter().enumerate() {
        // A down-step is a legal config — its transient is negative.
        let delta = (value - base).abs();
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
    while lo > 0 && (current[lo - 1] - base).abs() > thr {
        lo -= 1;
    }
    let mut hi = peak_i;
    while hi + 1 < n && (current[hi + 1] - base).abs() > thr {
        hi += 1;
    }
    // A NaN/inf edge timestamp silently erases a measurable capacitance -
    // fall back when the excursion edge is not finite. The row end itself
    // can be the non-finite edge, so fall back to the nearest finite
    // timestamp, not the positional end.
    let first_finite = time.iter().copied().find(|v| v.is_finite())?;
    let last_finite = time.iter().copied().rev().find(|v| v.is_finite())?;
    let left = if time[lo].is_finite() {
        time[lo]
    } else {
        first_finite
    };
    let right = if time[hi].is_finite() {
        time[hi]
    } else {
        last_finite
    };
    Some((left - 40.0, right + 40.0))
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
    // delta_v is |b - a| and a down-step integrates a negative transient —
    // capacitance is unsigned.
    Some(trapz(&tt, &yy).abs() / delta_v.abs())
}
/// Percent of finite metrics at or above each threshold.
pub fn coverage_percent(metrics: &[f32], thresholds: &[f32]) -> Vec<f32> {
    let total = metrics.iter().filter(|value| value.is_finite()).count();
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
                // A displaced shorter list must still count — dropping a
                // fail here would read as a pass below.
                scalars.append(channels);
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
        // A non-numeric duplicate key must not erase a reading already
        // found — only a real value replaces.
        if key.ends_with("hv_transient_test/starting_voltage") {
            if let Some(value) = child.as_f64() {
                *start = Some(value as f32);
            }
        }
        if key.ends_with("hv_transient_test/ending_voltage") {
            if let Some(value) = child.as_f64() {
                *end = Some(value as f32);
            }
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
    let Some(cells) = ox.checked_mul(oy).and_then(|n| n.checked_mul(oz)) else {
        return Vec::new();
    };
    let mut out = vec![0.0f32; cells];
    if nx == 0 || ny == 0 || nz == 0 {
        return out;
    }
    for z in 0..oz {
        for y in 0..oy {
            for x in 0..ox {
                // resample_linear's rule: checked index arithmetic — a
                // wrapped flat index could sample the wrong in-bounds voxel.
                let Some(sx) = x.checked_mul(nx).map(|v| (v / ox).min(nx - 1)) else {
                    continue;
                };
                let Some(sy) = y.checked_mul(ny).map(|v| (v / oy).min(ny - 1)) else {
                    continue;
                };
                let Some(sz) = z.checked_mul(nz).map(|v| (v / oz).min(nz - 1)) else {
                    continue;
                };
                let from = sy
                    .checked_add(ny.saturating_mul(sz))
                    .and_then(|v| nx.checked_mul(v))
                    .and_then(|v| v.checked_add(sx));
                let to = x + ox * (y + oy * z);
                if let Some(value) = from.and_then(|from| source.get(from)) {
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
    fn histogram_counts_past_the_f32_exact_integer_ceiling() {
        // 2^24 is where f32 accumulation freezes — a 20M-sample column
        // must count exactly, not pin at 16,777,216.
        let values = vec![1.0f32; 20_000_000];
        let (_, counts) = histogram(&values, 2);
        assert_eq!(counts[1], 20_000_000.0);
    }

    #[test]
    fn hv_step_window_falls_back_to_the_nearest_finite_timestamp() {
        // The excursion reaches the array end and the end itself is NaN —
        // the positional fallback is the same NaN, erasing the measurement.
        let time = [f32::NAN, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let current = [10.0, 10.0, 10.0, 0.0, 0.0, 0.0, 0.0];
        let (left, _) = hv_step_window(&time, &current).unwrap();
        assert!(left.is_finite(), "left fallback must be finite: {left}");
        let time = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, f32::NAN];
        let current = [0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0];
        let (_, right) = hv_step_window(&time, &current).unwrap();
        assert!(right.is_finite(), "right fallback must be finite: {right}");
    }

    #[test]
    fn a_non_finite_edge_timestamp_falls_back_to_the_row_ends() {
        // A NaN inside the excursion must not erase a measurable window.
        let window = hv_step_window(&[f32::NAN, 1.0, 2.0], &[0.0, 10.0, 0.0]).unwrap();
        assert_eq!(window, (-39.0, 41.0));
        // The NaN sits on the excursion edge itself — both fallbacks fire.
        let window = hv_step_window(&[0.0, f32::NAN, 20.0], &[0.0, 10.0, 0.0]).unwrap();
        assert_eq!(window, (-40.0, 60.0));
    }

    #[test]
    fn a_shorter_flag_list_cannot_hide_a_fail() {
        // The longer array used to replace channels outright — a fail in
        // the displaced list read as an overall pass.
        let (grade, _) = hv_firmware_flags(r#"{"result":{"a":["fail"],"z":["pass","pass"]}}"#);
        assert_eq!(grade.as_deref(), Some("fail"));
    }

    #[test]
    fn a_non_numeric_voltage_key_cannot_clobber_a_reading() {
        let delta = hv_delta_v(
            r#"{"a/hv_transient_test/starting_voltage":0,"a/hv_transient_test/ending_voltage":80,"z/hv_transient_test/starting_voltage":"n/a"}"#,
        );
        assert_eq!(delta, Some(80.0));
    }

    #[test]
    fn sk_req_035_relative_position_radius_sigma_percent_and_dose_per_mu() {
        let per_mu = dose_per_mu(&[2.0, 1.0], &[1.0, 0.0]);
        assert!((per_mu[0] - 2.0).abs() < 1e-5);
        assert!(per_mu[1].is_nan());
        let radius = radial_mm(&[3.0, f32::NAN], &[4.0, 1.0]);
        assert!((radius[0] - 5.0).abs() < 1e-5);
        assert!(radius[1].is_nan());
        let residual = remove_mean(&[1.0, 3.0, f32::NAN]);
        assert!((residual[0] + 1.0).abs() < 1e-5);
        assert!((residual[1] - 1.0).abs() < 1e-5);
        assert!(residual[2].is_nan());
        assert!(remove_mean(&[f32::NAN]).iter().all(|value| value.is_nan()));
        let percent = sigma_error_pct(&[11.0, 5.0], &[1.0, f32::NAN]);
        assert!((percent[0] - 10.0).abs() < 1e-4);
        assert!(percent[1].is_nan());
    }

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
    fn a_power_of_two_fft_matches_the_summed_dft() {
        let samples: Vec<f32> = (0..32).map(|i| (i as f32 * 0.37).sin() + 0.2).collect();
        let fast = rfft_power(&samples);
        let n = samples.len();
        for (k, power) in fast.iter().enumerate() {
            let mut re = 0.0f32;
            let mut im = 0.0f32;
            let angle = -2.0 * std::f32::consts::PI * k as f32 / n as f32;
            for (t, sample) in samples.iter().copied().enumerate() {
                let phase = angle * t as f32;
                re += sample * phase.cos();
                im += sample * phase.sin();
            }
            let expected = re * re + im * im;
            let tol = 1e-3 * expected.abs().max(1.0);
            assert!(
                (power - expected).abs() < tol,
                "bin {k}: {power} vs {expected}"
            );
        }
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
    fn splat_reach_is_in_voxels_not_millimetres() {
        // sigma 1 mm on a 0.5 mm grid needs ~13 voxels per axis to cover 3
        // sigma — the old code used the mm reach as a voxel count (~7).
        let spot = [6.0, 6.0, 6.0, 1.0, 1.0];
        let wide = splat_gaussians(&[spot], [0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [25, 25, 25]);
        let nonzero = wide.iter().filter(|v| **v > 0.0).count();
        assert!(
            nonzero > 7 * 7 * 7,
            "3 sigma at 0.5 mm should exceed 7^3: {nonzero}"
        );
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

    #[test]
    fn robust_limits_drop_one_spike_and_non_finite() {
        let mut samples = vec![1.0; 400];
        samples.push(f32::NAN);
        samples.push(100.0);
        let (lo, hi) = robust_limits(&samples).unwrap();
        assert!(hi < 2.0, "{hi}");
        assert!(lo > 0.5 && lo < 1.5, "{lo}");
        let (lo, hi) = robust_limits(&[2.0]).unwrap();
        assert!((lo - 1.96).abs() < 1e-3, "{lo}");
        assert!((hi - 2.04).abs() < 1e-3, "{hi}");
        assert!(robust_limits(&[f32::NAN]).is_none());
    }
    #[test]
    fn a_huge_bin_request_is_capped() {
        // bins arrives from tool arguments — unclamped it is an unbounded
        // allocation. The cap keeps the request finite.
        let (_edges, counts) = histogram(&[1.0, 2.0, 3.0], usize::MAX);
        assert_eq!(counts.len(), MAX_BINS);
        assert!(quantile_edges(&[1.0, 2.0], usize::MAX).len() <= MAX_BINS + 1);
    }

    #[test]
    fn coverage_percent_counts_only_finite_metrics() {
        assert_eq!(coverage_percent(&[10.0, f32::NAN], &[0.0]), vec![100.0]);
        assert_eq!(
            coverage_percent(&[10.0, f32::INFINITY, 50.0], &[25.0]),
            vec![50.0]
        );
    }

    #[test]
    fn an_empty_selection_reports_zero_dvh_coverage() {
        let (_, cumulative) = dvh(&[], &[true], 10);
        assert!(cumulative.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn quantile_edges_follow_the_data_scale() {
        let edges = quantile_edges(&[1e-9, 2e-9, 3e-9], 3);
        assert!(edges.len() >= 2, "{edges:?}");
        assert!(edges.iter().all(|e| e.abs() < 1e-6), "{edges:?}");
    }

    #[test]
    fn histogram_bins_a_column_below_the_absolute_floor() {
        let (edges, counts) = histogram(&[1e-13, 2e-13, 3e-13], 2);
        assert!(edges.iter().all(|e| e.abs() < 1e-10), "{edges:?}");
        assert_eq!(counts.iter().sum::<f32>(), 3.0);
    }

    #[test]
    fn welch_psd_normalizes_by_sample_rate() {
        let signal: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.1).sin()).collect();
        let (_, fast) = welch_psd(&signal, 1000.0, 256, 0.5);
        let (_, slow) = welch_psd(&signal, 100.0, 256, 0.5);
        let peak = |psd: &[f32]| psd.iter().cloned().fold(0.0f32, f32::max);
        assert!((peak(&slow) / peak(&fast) - 10.0).abs() < 0.5);
    }

    #[test]
    fn a_short_mask_does_not_select_the_dvh_tail() {
        let (_, curve) = dvh(&[0.0, 1.0, 2.0], &[true], 4);
        assert_eq!(curve, vec![1.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn zero_spacing_returns_an_empty_grid() {
        let grid = splat_gaussians(
            &[[0.0, 0.0, 0.0, 1.0, 1.0]],
            [0.0; 3],
            [0.0, 1.0, 1.0],
            [4, 4, 4],
        );
        assert!(grid.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn a_nonpositive_sample_rate_returns_no_psd() {
        let (freqs, psd) = welch_psd(&[1.0; 64], 0.0, 16, 0.5);
        assert!(freqs.is_empty() && psd.is_empty());
        let (freqs, psd) = welch_psd(&[1.0; 64], f32::NAN, 16, 0.5);
        assert!(freqs.is_empty() && psd.is_empty());
    }

    #[test]
    fn a_span_wider_than_f32_still_bins() {
        let (edges, counts) = histogram(&[-2.0e38, 0.0, 2.0e38], 3);
        assert!(edges.iter().all(|edge| edge.is_finite()));
        assert_eq!(counts, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn a_nan_fit_denominator_returns_none() {
        // sxx and sx*sx both overflow to inf for 2e38 inputs — den is NaN,
        // which must not slip the near-zero guard into Some((NaN, NaN)).
        assert_eq!(linear_fit(&[2.0e38, 2.0e38], &[1.0, 2.0]), None);
    }

    #[test]
    fn an_infinite_fit_numerator_returns_none() {
        // den stays finite here — only the numerator overflows.
        assert_eq!(linear_fit(&[1.0, 2.0], &[3.4e38, 1.0e-3]), None);
    }

    #[test]
    fn a_fit_over_a_huge_x_offset_keeps_the_true_slope() {
        // n*sxx - sx*sx cancels to rounding noise in f32 — the denominator
        // must be accumulated centered and in f64.
        let (slope, intercept) =
            linear_fit(&[1_000_000.0, 1_000_001.0, 1_000_002.0], &[1.0, 3.0, 5.0]).unwrap();
        assert!((slope - 2.0).abs() < 1e-4, "{slope}");
        assert!((intercept + 1_999_999.0).abs() < 1.0, "{intercept}");
    }

    #[test]
    fn a_fit_over_tiny_x_values_is_not_rejected() {
        // An absolute epsilon on the denominator refused coulomb-scale x data.
        let (slope, _) = linear_fit(&[1e-9, 2e-9, 3e-9], &[0.0, 1.0, 2.0]).unwrap();
        assert!((slope - 1e9).abs() < 1e6, "{slope}");
    }

    #[test]
    fn a_mid_trace_step_does_not_leak_into_the_dc_bin() {
        // scipy detrends 'constant' per segment — a baseline step's mean must
        // not leak into the lowest bins.
        let mut signal = vec![0.0f32; 256];
        signal[128..].fill(1.0);
        let (_, psd) = welch_psd(&signal, 100.0, 64, 0.5);
        assert!(psd[0] < 1e-6, "{}", psd[0]);
    }

    #[test]
    fn a_down_step_is_a_valid_hv_excursion() {
        // ending < starting voltage produces a negative transient — the
        // window must find it like the positive step.
        let time: Vec<f32> = (0..40).map(|i| i as f32).collect();
        let mut current = vec![0.0f32; 40];
        current[20] = -10.0;
        assert!(hv_step_window(&time, &current).is_some());
    }

    #[test]
    fn a_down_step_reports_unsigned_capacitance() {
        let time = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let current = vec![0.0, -10.0, -2.0, 0.0, 0.0, 0.0];
        let pf = hv_capacitance_pf(&time, &current, -1.0, 4.0, 80.0).unwrap();
        assert!(pf > 0.0, "{pf}");
    }

    #[test]
    fn a_short_grid_yields_zero_mip() {
        // A grid shorter than shape promises reads as misses, like
        // accumulate_z's length check.
        assert_eq!(mip_xy(&[1.0, 2.0, 3.0], [2, 2, 2]), vec![0.0; 4]);
    }

    #[test]
    fn a_splat_shape_product_that_overflows_returns_empty() {
        let grid = splat_gaussians(
            &[[0.0, 0.0, 0.0, 1.0, 1.0]],
            [0.0; 3],
            [1.0; 3],
            [usize::MAX, 2, 2],
        );
        assert!(grid.is_empty());
    }

    #[test]
    fn huge_spot_ids_do_not_merge_into_one_bucket() {
        // `as i32` would saturate both to i32::MAX and fold them together.
        assert_eq!(
            sums_by_spot_id(&[3.0e9, 4.0e9], &[1.0, 2.0]),
            vec![1.0, 2.0]
        );
        assert_eq!(
            sums_by_spot_run(&[3.0e9, 4.0e9, 4.0e9], &[1.0, 2.0, 3.0]),
            vec![1.0, 5.0]
        );
    }

    #[test]
    fn a_huge_declared_shape_returns_an_empty_projection() {
        assert!(mip_xy(&[1.0f32], [usize::MAX, 2, 1]).is_empty());
        assert!(resample_nearest(&[], [1, 1, 1], [usize::MAX, 2, 2]).is_empty());
    }

    #[test]
    fn an_out_of_range_overlap_cannot_collapse_the_stride() {
        // overlap >= 1 or NaN would leave step = 1 — a quadratic walk over the
        // whole signal for every segment. A clamped stride produces the same
        // PSD as the healthy overlap; a degenerate one would not complete.
        let signal = vec![1.0f32; 256];
        let (freqs, psd) = welch_psd(&signal, 100.0, 64, 0.5);
        assert!(!freqs.is_empty() && !psd.is_empty());
        let (_, out_of_range) = welch_psd(&signal, 100.0, 64, 1.5);
        let (_, nan_overlap) = welch_psd(&signal, 100.0, 64, f32::NAN);
        assert_eq!(out_of_range, psd);
        assert_eq!(nan_overlap, psd);
    }

    #[test]
    fn extreme_finite_samples_stay_finite() {
        let mut values = vec![-f32::MAX, f32::MAX];
        let window = robust_limits_finite(&mut values).unwrap();
        assert_eq!(window, (-f32::MAX, f32::MAX));
        let window = time_window(-f32::MAX, f32::MAX);
        assert_eq!(window, (0.0, f32::MAX));
    }

    #[test]
    fn a_flat_column_above_the_ulp_pads_representably() {
        // |v| > 2^23: +-0.5 rounds away and leaves span 0 — every count
        // used to pile into bin 0 and the panel domain collapsed.
        let (edges, counts) = histogram(&[1e9f32; 8], 4);
        assert!(edges[0] < edges[edges.len() - 1]);
        assert_eq!(counts.iter().sum::<f32>(), 8.0);
        let (wlo, whi) = time_window(20_000.0, 20_000.0);
        assert!(wlo < whi);
    }

    #[test]
    fn spot_ids_past_i64_range_do_not_merge() {
        // `as i64` saturates at i64::MAX — without the guard 1e19 and 2e19
        // would fold into one bucket and sum across spots.
        let ids = sums_by_spot_id(&[1e19, 2e19, 3.0], &[1.0, 2.0, 4.0]);
        assert_eq!(ids, vec![4.0]);
        let runs = sums_by_spot_run(&[1e19, 2e19, 3.0, 3.0], &[1.0, 2.0, 4.0, 4.0]);
        assert_eq!(runs, vec![8.0]);
    }

    #[test]
    fn calibration_factor_refuses_a_denormal_denominator() {
        // 1e-45 is positive-but-denormal f32 — the ratio casts to inf and
        // used to come back as a usable factor.
        assert_eq!(calibration_factor(&[1.0], &[1e-45]), None);
        assert_eq!(calibration_factor(&[2.0], &[1.0]), Some(2.0));
    }

    #[test]
    fn a_huge_declared_shape_does_not_wrap_the_flat_index() {
        // x * nx used to overflow before the bounds check — a wrapped index
        // could land on a valid voxel and sample the wrong one.
        let out = resample_nearest(
            &[1.0, 2.0, 3.0, 4.0],
            [usize::MAX, usize::MAX, usize::MAX],
            [4, 4, 4],
        );
        assert_eq!(out.len(), 64);
        // A wrapped index could land on a valid voxel — only the first cell
        // may hold a real sample.
        assert_eq!(out[0], 1.0);
        assert!(out[1..].iter().all(|value| *value == 0.0), "{out:?}");
    }
}
