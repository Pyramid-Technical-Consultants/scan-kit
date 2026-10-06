use super::kernel::{
    build_kernel, csda_range_mm, depth_sigma_mm, lateral_mm, layer_of, mass_fraction, mcs_along,
    mcs_at, protons_from_mu, through_wet, LayerKernel,
};
use super::{DoseFrame, Medium, Pencil, Quantity, Volume, MAX_CELLS, MEV_TO_GY_MM3, SIGMA_CUT};

pub(super) fn erf_as(x: f64) -> f64 {
    let ax = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let p = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
        + 0.254829592)
        * t;
    x.signum() * (1.0 - p * (-ax * ax).exp())
}

pub(super) fn normal_mass(mu: f64, sigma: f64, lo: f64, hi: f64) -> f64 {
    let sig = sigma.max(1e-6);
    let z1 = (hi - mu) / (sig * std::f64::consts::SQRT_2);
    let z0 = (lo - mu) / (sig * std::f64::consts::SQRT_2);
    0.5 * (erf_as(z1) - erf_as(z0))
}

pub(super) struct Prepared {
    x: f32,
    y: f32,
    sx: f32,
    sy: f32,
    energy: f32,
    weight: f32,
    z0: f32,
    z1: f32,
    grow: f32,
}

pub(super) fn prepare(
    medium: Medium,
    pencils: &[Pencil],
    quantity: Quantity,
    spread_pct: f64,
    scatter: bool,
    wet_mm: f64,
    k_mu: f64,
    gap_mm: f64,
) -> (LayerKernel, Vec<Prepared>) {
    let mut residual = Vec::with_capacity(pencils.len());
    let mut kept = Vec::with_capacity(pencils.len());
    for pencil in pencils {
        if !pencil.energy.is_finite() || !pencil.amount.is_finite() || pencil.amount == 0.0 {
            residual.push(0.0);
            kept.push(0.0);
            continue;
        }
        let (e, k) = through_wet(f64::from(pencil.energy), wet_mm);
        residual.push(e as f32);
        kept.push(k as f32);
    }
    let kernel = build_kernel(
        medium,
        &residual,
        spread_pct,
        scatter && quantity == Quantity::Dose,
    );
    let mut out = Vec::new();
    for (pencil, (&energy, &factor)) in pencils.iter().zip(residual.iter().zip(&kept)) {
        if energy <= 1.0 || factor <= 0.0 {
            continue;
        }
        let protons = match quantity {
            Quantity::Protons => f64::from(pencil.amount),
            Quantity::Dose | Quantity::Mu => {
                protons_from_mu(f64::from(pencil.amount), f64::from(energy), gap_mm, k_mu)
            }
        } * f64::from(factor);
        if !protons.is_finite() || protons == 0.0 {
            continue;
        }
        let row = layer_of(&kernel, energy);
        let (weight, z0, z1, grow) = if quantity == Quantity::Dose {
            let dep = f64::from(kernel.energy_dep[row]);
            let gy = protons * dep * MEV_TO_GY_MM3 / medium.rho;
            (gy as f32, kernel.zmin[row], 0.0, kernel.mcs_max[row])
        } else {
            let range = csda_range_mm(medium, f64::from(energy)) as f32;
            let sigma = depth_sigma_mm(medium, f64::from(energy), spread_pct) as f32;
            let end = if scatter {
                let (path, var) = mcs_along(medium, f64::from(energy));
                mcs_at(&path, &var, f64::from(range), f64::from(range)) as f32
            } else {
                0.0
            };
            let amount = match quantity {
                Quantity::Mu => pencil.amount * factor,
                _ => protons as f32,
            };
            (
                amount,
                -range - SIGMA_CUT as f32 * sigma.max(0.2),
                -range + SIGMA_CUT as f32 * sigma.max(0.2),
                end,
            )
        };
        out.push(Prepared {
            x: pencil.x,
            y: pencil.y,
            sx: pencil.sx.max(0.2),
            sy: pencil.sy.max(0.2),
            energy,
            weight,
            z0,
            z1,
            grow,
        });
    }
    (kernel, out)
}

pub(super) fn dose_grid(
    spots: &[Prepared],
    voxel: f32,
    z_floor: Option<f32>,
) -> ([f32; 3], [usize; 3]) {
    let v = voxel.clamp(0.25, 10.0);
    if spots.is_empty() {
        return ([0.0, 0.0, 0.0], [1, 1, 1]);
    }
    let mut lo = [f32::MAX; 3];
    let mut hi = [f32::MIN; 3];
    for spot in spots {
        let reach_x = SIGMA_CUT as f32 * hypot(spot.sx, spot.grow);
        let reach_y = SIGMA_CUT as f32 * hypot(spot.sy, spot.grow);
        lo[0] = lo[0].min(spot.x - reach_x);
        hi[0] = hi[0].max(spot.x + reach_x);
        lo[1] = lo[1].min(spot.y - reach_y);
        hi[1] = hi[1].max(spot.y + reach_y);
        lo[2] = lo[2].min(spot.z0);
        hi[2] = hi[2].max(spot.z1);
    }
    let mut lo_i = [
        (lo[0] / v).floor(),
        (lo[1] / v).floor(),
        (lo[2] / v).floor(),
    ];
    let mut hi_i = [(hi[0] / v).ceil(), (hi[1] / v).ceil(), (hi[2] / v).ceil()];
    if let Some(floor) = z_floor {
        lo_i[2] = lo_i[2].max((floor / v).ceil());
        hi_i[2] = hi_i[2].max(lo_i[2] + 1.0);
    }
    let mut shape = [1usize; 3];
    for i in 0..3 {
        shape[i] = (hi_i[i] - lo_i[i]).round().max(1.0) as usize;
        if shape[i] > MAX_CELLS {
            let mid = 0.5 * (lo_i[i] + hi_i[i]);
            lo_i[i] = (mid - MAX_CELLS as f32 / 2.0).floor();
            shape[i] = MAX_CELLS;
        }
    }
    ([lo_i[0] * v, lo_i[1] * v, lo_i[2] * v], shape)
}

pub(super) fn hypot(a: f32, b: f32) -> f32 {
    (a * a + b * b).sqrt()
}

pub(super) fn index_span(
    center: f32,
    sigma: f32,
    origin: f32,
    n: usize,
    voxel: f32,
) -> (usize, usize) {
    let sig = sigma.max(1e-3);
    let i0 = ((center - SIGMA_CUT as f32 * sig - origin) / voxel).floor() as i32;
    let i1 = ((center + SIGMA_CUT as f32 * sig - origin) / voxel).ceil() as i32;
    let a = i0.max(0) as usize;
    let b = i1.max(i0).min(n as i32).max(0) as usize;
    (a.min(n), b.min(n))
}

/// Analytic dose (or stopping fluence) on a 1 mm grid unless `voxel_mm` says otherwise.
/// `phantom_mm` of 0 sizes the grid to the deepest range plus five sigma.
pub fn analytic_volume(
    medium: Medium,
    pencils: &[Pencil],
    quantity: Quantity,
    spread_pct: f32,
    scatter: bool,
    wet_mm: f32,
    phantom_mm: f32,
    voxel_mm: f32,
    k_mu: f32,
    gap_mm: f32,
) -> Volume {
    analytic_on(
        medium, pencils, quantity, spread_pct, scatter, wet_mm, phantom_mm, voxel_mm, k_mu, gap_mm,
        None, None,
    )
}

/// Grid, weighted peak, and how many cell updates a full deposit would do.
pub fn dose_frame(
    medium: Medium,
    pencils: &[Pencil],
    quantity: Quantity,
    spread_pct: f32,
    scatter: bool,
    wet_mm: f32,
    phantom_mm: f32,
    voxel_mm: f32,
    k_mu: f32,
    gap_mm: f32,
    margin_mm: f32,
) -> DoseFrame {
    let (kernel, spots) = prepare(
        medium,
        pencils,
        quantity,
        f64::from(spread_pct),
        scatter,
        f64::from(wet_mm),
        f64::from(k_mu),
        f64::from(gap_mm),
    );
    let dose_mode = quantity == Quantity::Dose;
    let floor = (phantom_mm > 0.0).then_some(-phantom_mm);
    let (mut origin, mut shape) = dose_grid(&spots, voxel_mm, floor);
    let voxel = voxel_mm.clamp(0.25, 10.0);
    if margin_mm > 0.0 {
        let cells = ((margin_mm / voxel).ceil() as usize).min(32);
        origin[0] -= cells as f32 * voxel;
        origin[1] -= cells as f32 * voxel;
        shape[0] = (shape[0] + cells * 2).min(super::MAX_CELLS);
        shape[1] = (shape[1] + cells * 2).min(super::MAX_CELLS);
    }
    let focus = focus_index(&kernel, &spots, origin, shape, voxel, dose_mode);
    let visits = visit_count(&spots, origin, shape, voxel, dose_mode);
    DoseFrame {
        origin,
        shape,
        voxel,
        focus,
        visits,
    }
}

/// Same deposit as [`analytic_volume`]. `grid` is `(origin, shape, voxel)` when several
/// sessions must share one lattice. `planes` fills only the axial, coronal, and sagittal
/// planes through that voxel.
pub fn analytic_on(
    medium: Medium,
    pencils: &[Pencil],
    quantity: Quantity,
    spread_pct: f32,
    scatter: bool,
    wet_mm: f32,
    phantom_mm: f32,
    voxel_mm: f32,
    k_mu: f32,
    gap_mm: f32,
    grid: Option<([f32; 3], [usize; 3], f32)>,
    planes: Option<[usize; 3]>,
) -> Volume {
    let (kernel, spots) = prepare(
        medium,
        pencils,
        quantity,
        f64::from(spread_pct),
        scatter,
        f64::from(wet_mm),
        f64::from(k_mu),
        f64::from(gap_mm),
    );
    let floor = (phantom_mm > 0.0).then_some(-phantom_mm);
    let (origin, shape, v) = if let Some((origin, shape, voxel)) = grid {
        (origin, shape, voxel.clamp(0.25, 10.0))
    } else {
        let (origin, shape) = dose_grid(&spots, voxel_mm, floor);
        (origin, shape, voxel_mm.clamp(0.25, 10.0))
    };
    let [nx, ny, nz] = shape;
    let dose_mode = quantity == Quantity::Dose;
    let mut values = if let Some(focus) = planes {
        let focus = [
            focus[0].min(nx.saturating_sub(1)),
            focus[1].min(ny.saturating_sub(1)),
            focus[2].min(nz.saturating_sub(1)),
        ];
        let (axial, coronal, sagittal) =
            accumulate_planes(&kernel, &spots, origin, shape, v, focus, dose_mode);
        scatter_planes(shape, focus, &axial, &coronal, &sagittal)
    } else {
        let mut values = vec![0.0f32; nx * ny * nz];
        let mut fx = Vec::new();
        let mut fy = Vec::new();
        for spot in &spots {
            if spot.weight == 0.0 {
                continue;
            }
            let Some((ix0, ix1, iy0, iy1, iz0, iz1)) = spot_box(spot, origin, shape, v, dose_mode)
            else {
                continue;
            };
            for iz in iz0..iz1 {
                let z_lo = origin[2] + iz as f32 * v;
                let (fz, sx, sy) = depth_lateral(&kernel, spot, z_lo, v, dose_mode);
                if fz == 0.0 {
                    continue;
                }
                normal_row(spot.x, sx, origin[0], ix0, ix1, v, &mut fx);
                normal_row(spot.y, sy, origin[1], iy0, iy1, v, &mut fy);
                for (ky, iy) in (iy0..iy1).enumerate() {
                    let row = spot.weight * fz * fy[ky];
                    if row == 0.0 {
                        continue;
                    }
                    for (kx, ix) in (ix0..ix1).enumerate() {
                        values[ix + nx * (iy + ny * iz)] += row * fx[kx];
                    }
                }
            }
        }
        values
    };
    if quantity == Quantity::Dose {
        let scale = 1.0 / (v * v * v);
        for value in &mut values {
            *value *= scale;
        }
    }
    Volume {
        origin,
        shape,
        voxel: v,
        values,
    }
}

pub(super) fn spot_box(
    spot: &Prepared,
    origin: [f32; 3],
    shape: [usize; 3],
    voxel: f32,
    dose_mode: bool,
) -> Option<(usize, usize, usize, usize, usize, usize)> {
    let [nx, ny, nz] = shape;
    let (ix0, ix1) = index_span(spot.x, hypot(spot.sx, spot.grow), origin[0], nx, voxel);
    let (iy0, iy1) = index_span(spot.y, hypot(spot.sy, spot.grow), origin[1], ny, voxel);
    let (iz0, iz1) = if dose_mode {
        let a = ((spot.z0 - origin[2]) / voxel).floor() as i32;
        let b = ((spot.z1 - origin[2]) / voxel).ceil() as i32;
        (
            a.max(0).min(nz as i32) as usize,
            b.max(0).min(nz as i32) as usize,
        )
    } else {
        let mid = 0.5 * (spot.z0 + spot.z1);
        let sig = (spot.z1 - spot.z0) / (2.0 * SIGMA_CUT as f32);
        index_span(mid, sig, origin[2], nz, voxel)
    };
    if ix0 >= ix1 || iy0 >= iy1 || iz0 >= iz1 {
        None
    } else {
        Some((ix0, ix1, iy0, iy1, iz0, iz1))
    }
}

pub(super) fn visit_count(
    spots: &[Prepared],
    origin: [f32; 3],
    shape: [usize; 3],
    voxel: f32,
    dose_mode: bool,
) -> u64 {
    spots
        .iter()
        .filter_map(|spot| spot_box(spot, origin, shape, voxel, dose_mode))
        .map(|(ix0, ix1, iy0, iy1, iz0, iz1)| {
            (ix1 - ix0) as u64 * (iy1 - iy0) as u64 * (iz1 - iz0) as u64
        })
        .sum()
}

pub(super) fn focus_index(
    kernel: &LayerKernel,
    spots: &[Prepared],
    origin: [f32; 3],
    shape: [usize; 3],
    voxel: f32,
    dose_mode: bool,
) -> [usize; 3] {
    let mut weight = 0.0f64;
    let mut x = 0.0f64;
    let mut y = 0.0f64;
    let mut z = 0.0f64;
    for spot in spots {
        let w = f64::from(spot.weight.abs());
        if w == 0.0 {
            continue;
        }
        weight += w;
        x += w * f64::from(spot.x);
        y += w * f64::from(spot.y);
        z += w * f64::from(peak_z(kernel, spot, dose_mode));
    }
    if weight == 0.0 {
        return [0, 0, 0];
    }
    [
        axis_index((x / weight) as f32, origin[0], voxel, shape[0]),
        axis_index((y / weight) as f32, origin[1], voxel, shape[1]),
        axis_index((z / weight) as f32, origin[2], voxel, shape[2]),
    ]
}

pub(super) fn peak_z(kernel: &LayerKernel, spot: &Prepared, dose_mode: bool) -> f32 {
    if !dose_mode {
        return 0.5 * (spot.z0 + spot.z1);
    }
    let row = layer_of(kernel, spot.energy);
    let nodes = kernel.nodes;
    let mut best = 1usize;
    let mut rise = 0.0f32;
    for k in 1..nodes {
        let step = kernel.cdf[row * nodes + k] - kernel.cdf[row * nodes + k - 1];
        if step > rise {
            rise = step;
            best = k;
        }
    }
    let t = best as f32 / (nodes - 1) as f32;
    kernel.zmin[row] * (1.0 - t)
}

pub(super) fn axis_index(coord: f32, origin: f32, voxel: f32, n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    let i = ((coord - origin) / voxel).floor() as i32;
    i.clamp(0, n as i32 - 1) as usize
}

pub(super) fn depth_lateral(
    kernel: &LayerKernel,
    spot: &Prepared,
    z_lo: f32,
    voxel: f32,
    dose_mode: bool,
) -> (f32, f32, f32) {
    if dose_mode {
        let fz = mass_fraction(kernel, spot.energy, z_lo, z_lo + voxel);
        let mcs = lateral_mm(kernel, spot.energy, z_lo + 0.5 * voxel);
        (fz, hypot(spot.sx, mcs), hypot(spot.sy, mcs))
    } else {
        let mid = 0.5 * (spot.z0 + spot.z1);
        let sig = f64::from((spot.z1 - spot.z0) / (2.0 * SIGMA_CUT as f32)).max(1e-3);
        let fz = normal_mass(
            f64::from(mid),
            sig,
            f64::from(z_lo),
            f64::from(z_lo + voxel),
        ) as f32;
        (fz, hypot(spot.sx, spot.grow), hypot(spot.sy, spot.grow))
    }
}

pub(super) fn normal_row(
    center: f32,
    sigma: f32,
    origin: f32,
    i0: usize,
    i1: usize,
    voxel: f32,
    out: &mut Vec<f32>,
) {
    out.clear();
    let sig = f64::from(sigma).max(1e-3);
    for i in i0..i1 {
        let lo = f64::from(origin + i as f32 * voxel);
        out.push(normal_mass(f64::from(center), sig, lo, lo + f64::from(voxel)) as f32);
    }
}

pub(super) fn accumulate_planes(
    kernel: &LayerKernel,
    spots: &[Prepared],
    origin: [f32; 3],
    shape: [usize; 3],
    voxel: f32,
    focus: [usize; 3],
    dose_mode: bool,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let [nx, ny, nz] = shape;
    let workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .clamp(1, 8);
    let parts = if spots.len() < 32 || workers == 1 {
        1
    } else {
        workers.min(spots.len())
    };
    let size = spots.len().div_ceil(parts);
    let chunks: Vec<_> = spots.chunks(size.max(1)).collect();
    if chunks.len() <= 1 {
        let mut axial = vec![0.0; nx * ny];
        let mut coronal = vec![0.0; nx * nz];
        let mut sagittal = vec![0.0; ny * nz];
        paint_planes(
            kernel,
            spots,
            origin,
            shape,
            voxel,
            focus,
            dose_mode,
            &mut axial,
            &mut coronal,
            &mut sagittal,
        );
        return (axial, coronal, sagittal);
    }
    std::thread::scope(|scope| {
        let mut joins = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            joins.push(scope.spawn(move || {
                let mut axial = vec![0.0; nx * ny];
                let mut coronal = vec![0.0; nx * nz];
                let mut sagittal = vec![0.0; ny * nz];
                paint_planes(
                    kernel,
                    chunk,
                    origin,
                    shape,
                    voxel,
                    focus,
                    dose_mode,
                    &mut axial,
                    &mut coronal,
                    &mut sagittal,
                );
                (axial, coronal, sagittal)
            }));
        }
        let mut acc = joins.pop().unwrap().join().expect("plane deposit");
        for join in joins {
            let (axial, coronal, sagittal) = join.join().expect("plane deposit");
            for (dst, src) in acc.0.iter_mut().zip(axial) {
                *dst += src;
            }
            for (dst, src) in acc.1.iter_mut().zip(coronal) {
                *dst += src;
            }
            for (dst, src) in acc.2.iter_mut().zip(sagittal) {
                *dst += src;
            }
        }
        acc
    })
}

pub(super) fn paint_planes(
    kernel: &LayerKernel,
    spots: &[Prepared],
    origin: [f32; 3],
    shape: [usize; 3],
    voxel: f32,
    focus: [usize; 3],
    dose_mode: bool,
    axial: &mut [f32],
    coronal: &mut [f32],
    sagittal: &mut [f32],
) {
    let [nx, ny, _] = shape;
    let [fx, fy, fz] = focus;
    let mut gx = Vec::new();
    let mut gy = Vec::new();
    for spot in spots {
        if spot.weight == 0.0 {
            continue;
        }
        let Some((ix0, ix1, iy0, iy1, iz0, iz1)) = spot_box(spot, origin, shape, voxel, dose_mode)
        else {
            continue;
        };
        let hit_x = fx >= ix0 && fx < ix1;
        let hit_y = fy >= iy0 && fy < iy1;
        for iz in iz0..iz1 {
            let on_z = iz == fz;
            if !on_z && !hit_y && !hit_x {
                continue;
            }
            let z_lo = origin[2] + iz as f32 * voxel;
            let (mass, sx, sy) = depth_lateral(kernel, spot, z_lo, voxel, dose_mode);
            if mass == 0.0 {
                continue;
            }
            if on_z || hit_y {
                normal_row(spot.x, sx, origin[0], ix0, ix1, voxel, &mut gx);
            }
            if on_z || hit_x {
                normal_row(spot.y, sy, origin[1], iy0, iy1, voxel, &mut gy);
            }
            if on_z {
                for (ky, iy) in (iy0..iy1).enumerate() {
                    let row = spot.weight * mass * gy[ky];
                    if row == 0.0 {
                        continue;
                    }
                    for (kx, ix) in (ix0..ix1).enumerate() {
                        axial[ix + nx * iy] += row * gx[kx];
                    }
                }
            }
            if hit_y {
                let y_lo = origin[1] + fy as f32 * voxel;
                let fy1 = normal_mass(
                    f64::from(spot.y),
                    f64::from(sy).max(1e-3),
                    f64::from(y_lo),
                    f64::from(y_lo + voxel),
                ) as f32;
                let row = spot.weight * mass * fy1;
                if row != 0.0 {
                    for (kx, ix) in (ix0..ix1).enumerate() {
                        coronal[ix + nx * iz] += row * gx[kx];
                    }
                }
            }
            if hit_x {
                let x_lo = origin[0] + fx as f32 * voxel;
                let fx1 = normal_mass(
                    f64::from(spot.x),
                    f64::from(sx).max(1e-3),
                    f64::from(x_lo),
                    f64::from(x_lo + voxel),
                ) as f32;
                let row = spot.weight * mass * fx1;
                if row != 0.0 {
                    for (ky, iy) in (iy0..iy1).enumerate() {
                        sagittal[iy + ny * iz] += row * gy[ky];
                    }
                }
            }
        }
    }
}

pub(super) fn scatter_planes(
    shape: [usize; 3],
    focus: [usize; 3],
    axial: &[f32],
    coronal: &[f32],
    sagittal: &[f32],
) -> Vec<f32> {
    let [nx, ny, nz] = shape;
    let [fx, fy, fz] = focus;
    let mut values = vec![0.0f32; nx * ny * nz];
    for y in 0..ny {
        for x in 0..nx {
            values[x + nx * (y + ny * fz)] = axial[x + nx * y];
        }
    }
    for z in 0..nz {
        if z == fz {
            continue;
        }
        for x in 0..nx {
            values[x + nx * (fy + ny * z)] = coronal[x + nx * z];
        }
    }
    for z in 0..nz {
        if z == fz {
            continue;
        }
        for y in 0..ny {
            if y == fy {
                continue;
            }
            values[fx + nx * (y + ny * z)] = sagittal[y + ny * z];
        }
    }
    values
}

#[cfg(test)]
mod tests {
    use super::super::{csda_range_mm, Pencil, Quantity, WATER};
    use super::{analytic_on, analytic_volume, dose_frame};

    #[test]
    fn deposited_peak_sits_on_the_spot() {
        let volume = analytic_volume(
            WATER,
            &[Pencil {
                x: 5.0,
                y: -3.0,
                sx: 3.0,
                sy: 3.0,
                energy: 100.0,
                amount: 0.1,
            }],
            Quantity::Dose,
            1.0,
            true,
            0.0,
            0.0,
            1.0,
            2.0e-8,
            10.0,
        );
        let (ix, iy, iz) = volume.peak_index();
        let x = volume.origin[0] + (ix as f32 + 0.5) * volume.voxel;
        let y = volume.origin[1] + (iy as f32 + 0.5) * volume.voxel;
        let depth = -(volume.origin[2] + (iz as f32 + 0.5) * volume.voxel);
        let range = csda_range_mm(WATER, 100.0) as f32;
        assert!((x - 5.0).abs() < 1.5, "{x}");
        assert!((y + 3.0).abs() < 1.5, "{y}");
        assert!((depth - range).abs() < 4.0, "depth {depth} range {range}");
        assert!(volume.get(ix, iy, iz) > 0.0);
        let frame = dose_frame(
            WATER,
            &[Pencil {
                x: 5.0,
                y: -3.0,
                sx: 3.0,
                sy: 3.0,
                energy: 100.0,
                amount: 0.1,
            }],
            Quantity::Dose,
            1.0,
            true,
            0.0,
            0.0,
            1.0,
            2.0e-8,
            10.0,
            0.0,
        );
        assert!((frame.focus[0] as i32 - ix as i32).abs() <= 2);
        assert!((frame.focus[1] as i32 - iy as i32).abs() <= 2);
        assert!(
            (frame.focus[2] as i32 - iz as i32).abs() <= 4,
            "focus z {} peak {iz}",
            frame.focus[2]
        );
    }

    #[test]
    fn planes_match_the_full_deposit_on_the_cross() {
        let pencil = Pencil {
            x: 5.0,
            y: -3.0,
            sx: 3.0,
            sy: 3.0,
            energy: 100.0,
            amount: 0.1,
        };
        let full = analytic_volume(
            WATER,
            &[pencil],
            Quantity::Dose,
            1.0,
            true,
            0.0,
            0.0,
            1.0,
            2.0e-8,
            10.0,
        );
        let (ix, iy, iz) = full.peak_index();
        let grid = Some((full.origin, full.shape, full.voxel));
        let cross = analytic_on(
            WATER,
            &[pencil],
            Quantity::Dose,
            1.0,
            true,
            0.0,
            0.0,
            1.0,
            2.0e-8,
            10.0,
            grid,
            Some([ix, iy, iz]),
        );
        let [nx, ny, nz] = full.shape;
        for y in 0..ny {
            for x in 0..nx {
                let a = full.get(x, y, iz);
                let b = cross.get(x, y, iz);
                assert!((a - b).abs() <= 1e-4 * a.abs().max(1.0), "axial {x},{y}");
            }
        }
        for z in 0..nz {
            for x in 0..nx {
                let a = full.get(x, iy, z);
                let b = cross.get(x, iy, z);
                assert!((a - b).abs() <= 1e-4 * a.abs().max(1.0), "coronal {x},{z}");
            }
        }
        for z in 0..nz {
            for y in 0..ny {
                let a = full.get(ix, y, z);
                let b = cross.get(ix, y, z);
                assert!((a - b).abs() <= 1e-4 * a.abs().max(1.0), "sagittal {y},{z}");
            }
        }
    }

    #[test]
    fn a_wide_spot_map_skips_the_full_lattice() {
        let pencils: Vec<_> = (0..40)
            .map(|i| Pencil {
                x: i as f32 * 5.0,
                y: 0.0,
                sx: 4.0,
                sy: 4.0,
                energy: 180.0,
                amount: 0.05,
            })
            .collect();
        let frame = dose_frame(
            WATER,
            &pencils,
            Quantity::Dose,
            1.0,
            true,
            0.0,
            0.0,
            1.0,
            2.0e-8,
            10.0,
            0.0,
        );
        assert!(frame.planes_only(), "visits {}", frame.visits);
    }
}
