use super::Volume;

/// The 99.9th percentile of the positive finite samples.
pub fn robust_high(values: &[f32]) -> f32 {
    // Positive finite floats sort as their bits. Count the high 16 bits, then
    // select only inside the bin that holds the rank, not a copy of the cube.
    let bin =
        |value: f32| (value.is_finite() && value > 0.0).then(|| (value.to_bits() >> 16) as usize);
    let mut bins = vec![0usize; 1 << 16];
    for at in values.iter().filter_map(|value| bin(*value)) {
        bins[at] += 1;
    }
    let count: usize = bins.iter().sum();
    if count == 0 {
        return 1.0;
    }
    let last = count - 1;
    let index = (((last as f64) * 0.999) as usize).min(last);
    let mut below = 0;
    let mut chosen = 0;
    for (at, n) in bins.iter().enumerate() {
        if below + n > index {
            chosen = at;
            break;
        }
        below += n;
    }
    let mut inside: Vec<f32> = values
        .iter()
        .copied()
        .filter(|value| bin(*value) == Some(chosen))
        .collect();
    crate::stats::select_rank(&mut inside, index - below).max(1e-6)
}

/// Bounding box of samples at or above `threshold`, in the same coordinates as the slice axes.
pub fn field_bounds(
    image: &[f32],
    cols: usize,
    rows: usize,
    origin_x: f32,
    origin_y: f32,
    voxel: f32,
    threshold: f32,
) -> Option<[f32; 4]> {
    let mut x0 = cols;
    let mut x1 = 0;
    let mut y0 = rows;
    let mut y1 = 0;
    for y in 0..rows {
        for x in 0..cols {
            let value = image.get(x + cols * y).copied().unwrap_or(0.0);
            if value >= threshold {
                x0 = x0.min(x);
                x1 = x1.max(x + 1);
                y0 = y0.min(y);
                y1 = y1.max(y + 1);
            }
        }
    }
    if x1 <= x0 || y1 <= y0 {
        return None;
    }
    Some([
        origin_x + x0 as f32 * voxel,
        origin_x + x1 as f32 * voxel,
        origin_y + y0 as f32 * voxel,
        origin_y + y1 as f32 * voxel,
    ])
}

/// Global gamma. `cutoff_pct` is a percent of the maximum of `evaluated`.
/// Search uses sub-voxel steps and stops at γ = 2. Returns `(gamma, passed, evaluated)`.
pub fn gamma_index(
    reference: &[f32],
    evaluated: &[f32],
    shape: [usize; 3],
    dose_percent: f32,
    distance_mm: f32,
    spacing: [f32; 3],
    cutoff_pct: f32,
) -> (Vec<f32>, u32, u32) {
    gamma_index_norm(
        reference,
        evaluated,
        shape,
        dose_percent,
        distance_mm,
        spacing,
        cutoff_pct,
        0.0,
    )
}

/// `gamma_index` with an explicit normalization dose: `dose_percent` and
/// `cutoff_pct` are percents of `norm` (the TPS maximum for a patient-QA
/// comparison). A `norm` <= 0 falls back to the evaluated maximum.
pub fn gamma_index_norm(
    reference: &[f32],
    evaluated: &[f32],
    shape: [usize; 3],
    dose_percent: f32,
    distance_mm: f32,
    spacing: [f32; 3],
    cutoff_pct: f32,
    norm: f32,
) -> (Vec<f32>, u32, u32) {
    let [nx, ny, nz] = shape;
    let n = nx * ny * nz;
    let len = n.min(reference.len()).min(evaluated.len());
    let mut gamma = vec![0.0f32; len];
    let mut norm = norm.max(0.0);
    if norm <= 0.0 {
        for value in evaluated.iter().take(len) {
            if value.is_finite() {
                norm = norm.max(*value);
            }
        }
    }
    if norm <= 0.0 {
        return (gamma, 0, 0);
    }
    let dd = (dose_percent.max(1e-6) / 100.0 * norm).max(1e-12);
    let floor = cutoff_pct / 100.0 * norm;
    let cap = 2.0f32;
    let mut mask = Vec::new();
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let i = x + nx * (y + ny * z);
                if i >= len {
                    continue;
                }
                let r = reference[i];
                if r.is_finite() && r > 0.0 && r >= floor {
                    mask.push((x, y, z, i));
                }
            }
        }
    }
    if mask.is_empty() {
        return (gamma, 0, 0);
    }
    let mut offsets = gamma_offsets(spacing, distance_mm.max(1e-3), cap, shape);
    // A one-voxel axis is a plane. Stepping off it blends in the zeros outside,
    // which a hot evaluated plane would score as a match.
    offsets.retain(|o| (0..3).all(|axis| shape[axis] > 1 || o[axis] == 0.0));
    let cap2 = cap * cap;
    // Most voxels of a smooth pair settle within a millimetre. Only a noisy
    // cube leaves enough behind to pay for the local dose range.
    let split = offsets.partition_point(|o| o[3] <= 0.25);
    let start = vec![cap2; mask.len()];
    let none = vec![0.0; mask.len()];
    let near = &offsets[..split];
    let mut scores = score_mask(reference, evaluated, shape, &mask, &none, &start, near, dd);
    if let Some(next) = offsets.get(split) {
        let left: Vec<usize> = (0..mask.len())
            .filter(|&slot| scores[slot] > next[3])
            .collect();
        if !left.is_empty() {
            let rest: Vec<_> = left.iter().map(|&slot| mask[slot]).collect();
            let from: Vec<f32> = left.iter().map(|&slot| scores[slot]).collect();
            let floors: Vec<f32> = if len == n && left.len() * 20 > mask.len() {
                let radius: [usize; 3] = std::array::from_fn(|axis| {
                    let reach = offsets.iter().fold(0.0f32, |m, o| m.max(o[axis].abs()));
                    reach.ceil() as usize + 1
                });
                let (lo, hi) = local_range(&evaluated[..n], shape, radius);
                rest.iter()
                    .map(|&(_, _, _, i)| dose_floor(reference[i], lo[i], hi[i], dd))
                    .collect()
            } else {
                vec![0.0; rest.len()]
            };
            let far = &offsets[split..];
            let more = score_mask(reference, evaluated, shape, &rest, &floors, &from, far, dd);
            for (slot, g2) in left.into_iter().zip(more) {
                scores[slot] = g2;
            }
        }
    }
    let mut passed = 0u32;
    for (&(_, _, _, i), g2) in mask.iter().zip(scores) {
        let g = g2.sqrt();
        gamma[i] = g;
        if g <= 1.0 {
            passed += 1;
        }
    }
    (gamma, passed, mask.len() as u32)
}

/// 2D gamma of the doses summed through z, y, and x: axial, coronal, sagittal.
/// Each plane is its own global gamma in dose × mm, normalized and cut off at
/// that plane's maximum, with the search kept inside the plane. Returns the γ
/// planes and each plane's pass percent.
pub fn projected_gamma(
    reference: &Volume,
    evaluated: &Volume,
    dose_percent: f32,
    distance_mm: f32,
    cutoff_pct: f32,
) -> [(Vec<f32>, f32); 3] {
    let [nx, ny, nz] = reference.shape;
    let voxel = reference.voxel.max(1e-6);
    let sums =
        |volume: &Volume| super::ray::scan_grid(&volume.values, volume.shape, voxel).integrals;
    let (got, want) = (sums(reference), sums(evaluated));
    let planes = [[nx, ny, 1], [nx, nz, 1], [ny, nz, 1]];
    std::array::from_fn(|plane| {
        let (gamma, passed, scored) = gamma_index(
            &got[plane],
            &want[plane],
            planes[plane],
            dose_percent,
            distance_mm,
            [voxel; 3],
            cutoff_pct,
        );
        let rate = if scored == 0 {
            0.0
        } else {
            100.0 * passed as f32 / scored as f32
        };
        (gamma, rate)
    })
}

/// γ² at each masked voxel. Bands own disjoint voxels, so a large field still
/// searches the whole box instead of three planes.
fn score_mask(
    reference: &[f32],
    evaluated: &[f32],
    shape: [usize; 3],
    mask: &[(usize, usize, usize, usize)],
    floors: &[f32],
    from: &[f32],
    offsets: &[[f32; 4]],
    dd: f32,
) -> Vec<f32> {
    let workers = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1)
        .clamp(1, 8);
    let bands = if mask.len() < 2_048 || workers == 1 {
        1
    } else {
        workers.min(mask.len() / 512).max(1)
    };
    if bands == 1 {
        return score_chunk(reference, evaluated, shape, mask, floors, from, offsets, dd);
    }
    let chunk = mask.len().div_ceil(bands);
    std::thread::scope(|scope| {
        let mut joins = Vec::new();
        let parts = mask
            .chunks(chunk)
            .zip(floors.chunks(chunk))
            .zip(from.chunks(chunk));
        for ((band, floor), start) in parts {
            joins.push(scope.spawn(move || {
                score_chunk(reference, evaluated, shape, band, floor, start, offsets, dd)
            }));
        }
        joins
            .into_iter()
            .flat_map(|join| join.join().expect("gamma band"))
            .collect()
    })
}

fn score_chunk(
    reference: &[f32],
    evaluated: &[f32],
    shape: [usize; 3],
    mask: &[(usize, usize, usize, usize)],
    floors: &[f32],
    from: &[f32],
    offsets: &[[f32; 4]],
    dd: f32,
) -> Vec<f32> {
    let mut best = from.to_vec();
    // Offsets run nearest first, and no sample can beat a voxel's dose floor,
    // so a voxel stops once its γ² is at or below this offset's distance term
    // plus that floor. A noisy voxel no longer searches the whole sphere.
    let mut active: Vec<usize> = (0..mask.len()).collect();
    for offset in offsets {
        active.retain(|&slot| best[slot] > offset[3] + floors[slot]);
        if active.is_empty() {
            break;
        }
        for &slot in &active {
            let (x, y, z, i) = mask[slot];
            let sample = sample_linear(
                evaluated,
                shape,
                x as f32 + offset[0],
                y as f32 + offset[1],
                z as f32 + offset[2],
            );
            let dose = (sample - reference[i]) / dd;
            let g2 = offset[3] + dose * dose;
            if g2 < best[slot] {
                best[slot] = g2;
            }
        }
    }
    best
}

/// Smallest dose term any sample in `[lo, hi]` can give, shaved so float
/// rounding in the trilinear sample cannot dip under it.
fn dose_floor(reference: f32, lo: f32, hi: f32, dd: f32) -> f32 {
    let gap = (lo - reference).max(reference - hi);
    if gap.is_nan() || gap <= 0.0 {
        return 0.0;
    }
    let slack = 1e-5 * reference.abs().max(lo.abs()).max(hi.abs());
    let gap = (gap - slack).max(0.0) / dd;
    gap * gap * (1.0 - 1e-5)
}

/// Min and max of `values` over a box of `radius` voxels on each axis. Samples
/// outside the grid read as zero, as in `sample_linear`.
fn local_range(values: &[f32], shape: [usize; 3], radius: [usize; 3]) -> (Vec<f32>, Vec<f32>) {
    let mut lo = values.to_vec();
    let mut hi = values.to_vec();
    let [nx, ny, nz] = shape;
    let strides = [1, nx, nx * ny];
    for axis in 0..3 {
        let n = shape[axis];
        let r = radius[axis];
        let w = 2 * r + 1;
        let stride = strides[axis];
        let mut line_lo = vec![0.0f32; n + 2 * r];
        let mut line_hi = vec![0.0f32; n + 2 * r];
        let (mut g, mut h) = (vec![0.0f32; n + 2 * r], vec![0.0f32; n + 2 * r]);
        for start in 0..nx * ny * nz {
            if (start / stride) % n != 0 {
                continue;
            }
            for k in 0..n {
                line_lo[r + k] = lo[start + k * stride];
                line_hi[r + k] = hi[start + k * stride];
            }
            window(&line_lo, w, &mut g, &mut h, f32::min, |k, value| {
                lo[start + k * stride] = value;
            });
            window(&line_hi, w, &mut g, &mut h, f32::max, |k, value| {
                hi[start + k * stride] = value;
            });
        }
    }
    (lo, hi)
}

/// van Herk / Gil–Werman running min or max over windows of `w` samples.
fn window(
    line: &[f32],
    w: usize,
    g: &mut [f32],
    h: &mut [f32],
    op: fn(f32, f32) -> f32,
    mut put: impl FnMut(usize, f32),
) {
    let m = line.len();
    for i in 0..m {
        g[i] = if i % w == 0 {
            line[i]
        } else {
            op(g[i - 1], line[i])
        };
    }
    for i in (0..m).rev() {
        h[i] = if i % w == w - 1 || i == m - 1 {
            line[i]
        } else {
            op(h[i + 1], line[i])
        };
    }
    for j in 0..=m - w {
        put(j, op(h[j], g[j + w - 1]));
    }
}

pub(super) fn gamma_offsets(
    spacing: [f32; 3],
    dta: f32,
    cap: f32,
    shape: [usize; 3],
) -> Vec<[f32; 4]> {
    if !dta.is_finite()
        || dta <= 0.0
        || !cap.is_finite()
        || cap <= 0.0
        || spacing.iter().any(|v| !v.is_finite() || *v <= 0.0)
    {
        return Vec::new();
    }
    let reach = cap * dta;
    let mut axes = Vec::new();
    for (axis, &v) in spacing.iter().enumerate() {
        let step = v / (v / (dta / 4.0)).ceil().max(1.0);
        // A shift past the grid's own extent in this axis can never land on
        // another voxel — and it keeps a silly dta from looping forever.
        let limit = shape[axis] as f64 * v as f64 / step as f64;
        let n = ((reach as f64 / step as f64).ceil() as i64).clamp(0, limit.ceil() as i64 + 1);
        let mut axis = Vec::new();
        for i in -n..=n {
            axis.push(i as f32 * step);
        }
        axes.push(axis);
    }
    let mut rows = Vec::new();
    for &gx in &axes[0] {
        for &gy in &axes[1] {
            for &gz in &axes[2] {
                let d2 = (gx * gx + gy * gy + gz * gz) / (dta * dta);
                if d2 <= cap * cap {
                    rows.push([gx / spacing[0], gy / spacing[1], gz / spacing[2], d2]);
                }
            }
        }
    }
    rows.sort_by(|a, b| a[3].partial_cmp(&b[3]).unwrap_or(std::cmp::Ordering::Equal));
    rows
}

pub(super) fn sample_linear(vol: &[f32], shape: [usize; 3], x: f32, y: f32, z: f32) -> f32 {
    let [nx, ny, nz] = shape;
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let z0 = z.floor() as i32;
    let tx = x - x0 as f32;
    let ty = y - y0 as f32;
    let tz = z - z0 as f32;
    let mut acc = 0.0f32;
    for dz in 0..2 {
        for dy in 0..2 {
            for dx in 0..2 {
                let xx = x0 + dx;
                let yy = y0 + dy;
                let zz = z0 + dz;
                let value = if xx < 0
                    || yy < 0
                    || zz < 0
                    || xx >= nx as i32
                    || yy >= ny as i32
                    || zz >= nz as i32
                {
                    0.0
                } else {
                    vol[xx as usize + nx * (yy as usize + ny * zz as usize)]
                };
                let wx = if dx == 0 { 1.0 - tx } else { tx };
                let wy = if dy == 0 { 1.0 - ty } else { ty };
                let wz = if dz == 0 { 1.0 - tz } else { tz };
                acc += value * wx * wy * wz;
            }
        }
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::{gamma_index, gamma_offsets, projected_gamma, robust_high, sample_linear, Volume};

    #[test]
    fn a_noisy_reference_matches_a_full_search() {
        let shape = [12usize, 10, 9];
        let n = shape[0] * shape[1] * shape[2];
        let mut seed = 11u32;
        let mut next = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as f32 / 16_777_216.0
        };
        let evaluated: Vec<f32> = (0..n)
            .map(|i| 1.0 + 0.2 * ((i % 12) as f32 / 12.0) + 0.1 * next())
            .collect();
        // Heavy noise builds the local dose range. Light noise settles near and skips it.
        for spread in [0.6f32, 0.02] {
            let reference: Vec<f32> = evaluated
                .iter()
                .map(|v| v * (1.0 - spread / 2.0 + spread * next()))
                .collect();
            matches_a_full_search(&reference, &evaluated, shape);
        }
    }

    fn matches_a_full_search(reference: &[f32], evaluated: &[f32], shape: [usize; 3]) {
        let n = reference.len();
        let spacing = [1.0, 1.0, 1.0];
        let (gamma, _, scored) = gamma_index(reference, evaluated, shape, 3.0, 2.0, spacing, 10.0);
        assert_eq!(scored as usize, n);
        let offsets = gamma_offsets(spacing, 2.0, 2.0, shape);
        let norm = evaluated.iter().copied().fold(0.0f32, f32::max);
        let dd = 0.03 * norm;
        for z in 0..shape[2] {
            for y in 0..shape[1] {
                for x in 0..shape[0] {
                    let i = x + shape[0] * (y + shape[1] * z);
                    let full = offsets
                        .iter()
                        .map(|o| {
                            let s = sample_linear(
                                evaluated,
                                shape,
                                x as f32 + o[0],
                                y as f32 + o[1],
                                z as f32 + o[2],
                            );
                            o[3] + ((s - reference[i]) / dd).powi(2)
                        })
                        .fold(4.0f32, f32::min)
                        .sqrt();
                    assert_eq!(gamma[i].to_bits(), full.to_bits(), "voxel {i}");
                }
            }
        }
    }

    #[test]
    fn the_robust_high_matches_a_sorted_rank() {
        let sorted = |values: &[f32]| {
            let mut kept: Vec<f32> = values
                .iter()
                .copied()
                .filter(|v| v.is_finite() && *v > 0.0)
                .collect();
            if kept.is_empty() {
                return 1.0;
            }
            kept.sort_by(f32::total_cmp);
            kept[((kept.len() - 1) as f64 * 0.999) as usize].max(1e-6)
        };
        let mut seed = 7u32;
        let mut next = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            seed
        };
        let mut skewed: Vec<f32> = (0..40_000)
            .map(|_| (next() >> 8) as f32 / 65_536.0)
            .map(|v| v * v * v)
            .collect();
        skewed.extend([0.0, -3.0, f32::NAN, f32::INFINITY, 1e-30]);
        let ties = vec![2.5f32; 3000];
        for values in [skewed, ties, vec![0.0, -1.0], vec![4.0], Vec::new()] {
            assert_eq!(robust_high(&values).to_bits(), sorted(&values).to_bits());
        }
    }

    #[test]
    fn a_shift_along_depth_within_dta_passes() {
        let n = 5usize;
        let shape = [n, n, n];
        let mut reference = vec![0.0f32; n * n * n];
        let mut evaluated = vec![0.0f32; n * n * n];
        let center = 2 + n * (2 + n * 2);
        let deeper = 2 + n * (2 + n * 4);
        reference[center] = 1.0;
        evaluated[deeper] = 1.0;
        let (gamma, passed, scored) = gamma_index(
            &reference,
            &evaluated,
            shape,
            3.0,
            2.0,
            [1.0, 1.0, 1.0],
            10.0,
        );
        assert_eq!((scored, passed), (1, 1));
        assert!((gamma[center] - 1.0).abs() < 1e-3, "{}", gamma[center]);
    }

    #[test]
    fn a_wide_match_scores_every_voxel() {
        let shape = [32usize, 32, 4];
        let n = shape[0] * shape[1] * shape[2];
        let dose = vec![1.0f32; n];
        let (gamma, passed, scored) =
            gamma_index(&dose, &dose, shape, 3.0, 2.0, [1.0, 1.0, 1.0], 10.0);
        assert_eq!((scored, passed), (n as u32, n as u32));
        assert!(gamma.iter().all(|value| *value <= 1e-4));
    }

    #[test]
    fn offsets_stay_bounded_by_the_grid() {
        // A 1 km dta asks for shifts far wider than the grid; anything past the
        // extent can never land on a voxel, so the set stays small.
        let offsets = gamma_offsets([1.0, 1.0, 1.0], 1e6, 2.0, [10, 10, 10]);
        assert!(offsets.iter().all(|o| (0..3).all(|a| o[a].abs() <= 12.0)));
        for dta in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert!(gamma_offsets([1.0; 3], dta, 2.0, [10, 10, 10]).is_empty());
        }
        assert!(gamma_offsets([0.0; 3], 2.0, 2.0, [10, 10, 10]).is_empty());
    }

    #[test]
    fn a_plane_does_not_search_off_itself() {
        // 14% hot fails in the plane. Half a millimetre off a 4 mm plane blends
        // in 12.5% of the zeros outside, which would land on the reference.
        let shape = [8usize, 8, 1];
        let reference = vec![1.0f32; 64];
        let evaluated = vec![1.14f32; 64];
        let (gamma, _, scored) =
            gamma_index(&reference, &evaluated, shape, 3.0, 2.0, [4.0; 3], 10.0);
        assert_eq!(scored, 64);
        let inside = (1..7).flat_map(|y| (1..7).map(move |x| x + 8 * y));
        assert!(inside.map(|i| gamma[i]).all(|g| g > 1.0), "{gamma:?}");
    }

    #[test]
    fn projected_gamma_scores_each_summed_plane() {
        let shape = [6usize, 5, 4];
        let values: Vec<f32> = (0..120).map(|i| 1.0 + (i % 7) as f32 * 0.1).collect();
        let volume = Volume {
            origin: [0.0; 3],
            shape,
            voxel: 2.0,
            values,
        };
        let planes = projected_gamma(&volume, &volume, 3.0, 2.0, 10.0);
        let sizes = planes.each_ref().map(|(gamma, _)| gamma.len());
        assert_eq!(sizes, [30, 24, 20]);
        assert!(planes.iter().all(|(_, rate)| (*rate - 100.0).abs() < 1e-4));
        let hot = Volume {
            values: volume.values.iter().map(|value| value * 1.2).collect(),
            ..volume.clone()
        };
        let planes = projected_gamma(&volume, &hot, 3.0, 2.0, 10.0);
        assert!(
            planes.iter().all(|(_, rate)| *rate < 50.0),
            "20% hot passed"
        );
    }
}
