pub fn robust_high(values: &[f32]) -> f32 {
    let mut kept: Vec<f32> = values
        .iter()
        .copied()
        .filter(|v| v.is_finite() && *v > 0.0)
        .collect();
    if kept.is_empty() {
        return 1.0;
    }
    let last = kept.len() - 1;
    let index = ((last as f64) * 0.999) as usize;
    crate::stats::select_rank(&mut kept, index.min(last)).max(1e-6)
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
    let [nx, ny, nz] = shape;
    let n = nx * ny * nz;
    let len = n.min(reference.len()).min(evaluated.len());
    let mut gamma = vec![0.0f32; len];
    let mut norm = 0.0f32;
    for value in evaluated.iter().take(len) {
        if value.is_finite() {
            norm = norm.max(*value);
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
    let offsets = gamma_offsets(spacing, distance_mm.max(1e-3), cap);
    let cap2 = cap * cap;
    let scores = score_mask(reference, evaluated, shape, &mask, &offsets, dd, cap2);
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

/// γ² at each masked voxel. Bands own disjoint voxels, so a large field still
/// searches the whole box instead of three planes.
fn score_mask(
    reference: &[f32],
    evaluated: &[f32],
    shape: [usize; 3],
    mask: &[(usize, usize, usize, usize)],
    offsets: &[[f32; 4]],
    dd: f32,
    cap2: f32,
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
        return score_chunk(reference, evaluated, shape, mask, offsets, dd, cap2);
    }
    let chunk = mask.len().div_ceil(bands);
    std::thread::scope(|scope| {
        let mut joins = Vec::new();
        for band in mask.chunks(chunk) {
            joins.push(
                scope.spawn(move || {
                    score_chunk(reference, evaluated, shape, band, offsets, dd, cap2)
                }),
            );
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
    offsets: &[[f32; 4]],
    dd: f32,
    cap2: f32,
) -> Vec<f32> {
    let mut best = vec![cap2; mask.len()];
    let mut worst = cap2;
    for offset in offsets {
        if offset[3] >= worst {
            break;
        }
        let mut next = 0.0f32;
        for (slot, &(x, y, z, i)) in mask.iter().enumerate() {
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
            next = next.max(best[slot]);
        }
        worst = next;
    }
    best
}

pub(super) fn gamma_offsets(spacing: [f32; 3], dta: f32, cap: f32) -> Vec<[f32; 4]> {
    let reach = cap * dta;
    let mut axes = Vec::new();
    for v in spacing {
        let step = v / (v / (dta / 4.0)).ceil().max(1.0);
        let n = (reach / step).ceil() as i32;
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
    use super::gamma_index;

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
}
