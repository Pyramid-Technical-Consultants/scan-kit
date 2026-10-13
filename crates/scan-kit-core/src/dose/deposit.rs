use super::kernel::{
    build_kernel, csda_range_mm, depth_sigma_mm, layer_of, mcs_along, mcs_at, protons_from_mu,
    through_wet, LayerKernel,
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
    let kernel = build_kernel(medium, &residual, spread_pct);
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
            let (path, var) = mcs_along(medium, f64::from(energy));
            let end = mcs_at(&path, &var, f64::from(range), f64::from(range)) as f32;
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
    wet_mm: f32,
    phantom_mm: f32,
    voxel_mm: f32,
    k_mu: f32,
    gap_mm: f32,
) -> Volume {
    analytic_on(
        medium, pencils, quantity, spread_pct, wet_mm, phantom_mm, voxel_mm, k_mu, gap_mm, None,
        None,
    )
}

/// Deepest range plus `margin_sigma` range-spread widths. Python's Auto phantom.
fn auto_depth_mm(medium: Medium, pencils: &[Pencil], spread_pct: f64, margin_sigma: f64) -> f32 {
    let mut best = 0.0f64;
    for pencil in pencils {
        let energy = f64::from(pencil.energy.max(1.0));
        let reach = super::kernel::csda_range_mm(medium, energy)
            + margin_sigma.max(0.0) * super::kernel::depth_sigma_mm(medium, energy, spread_pct);
        best = best.max(reach);
    }
    best as f32
}

/// Grid, weighted peak, and how many cell updates a full deposit would do.
pub fn dose_frame(
    medium: Medium,
    pencils: &[Pencil],
    quantity: Quantity,
    spread_pct: f32,
    wet_mm: f32,
    phantom_mm: f32,
    voxel_mm: f32,
    k_mu: f32,
    gap_mm: f32,
    margin_sigma: f32,
) -> DoseFrame {
    let (kernel, spots) = prepare(
        medium,
        pencils,
        quantity,
        f64::from(spread_pct),
        f64::from(wet_mm),
        f64::from(k_mu),
        f64::from(gap_mm),
    );
    let dose_mode = quantity == Quantity::Dose;
    let floor = if phantom_mm > 0.0 {
        Some(-phantom_mm)
    } else if margin_sigma > 0.0 {
        let depth = auto_depth_mm(
            medium,
            pencils,
            f64::from(spread_pct),
            f64::from(margin_sigma),
        );
        (depth > 0.0).then_some(-depth)
    } else {
        None
    };
    let (origin, shape) = dose_grid(&spots, voxel_mm, floor);
    let voxel = voxel_mm.clamp(0.25, 10.0);
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
        deposit_lattice(
            &kernel,
            &spots,
            origin,
            shape,
            v,
            dose_mode,
            (quantity == Quantity::Dose).then_some(1.0 / (v * v * v)),
        )
    };
    if quantity == Quantity::Dose && planes.is_some() {
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

fn deposit_lattice(
    kernel: &LayerKernel,
    spots: &[Prepared],
    origin: [f32; 3],
    shape: [usize; 3],
    voxel: f32,
    dose_mode: bool,
    scale: Option<f32>,
) -> Vec<f32> {
    let [nx, ny, nz] = shape;
    let mut values = vec![0.0f32; nx.saturating_mul(ny).saturating_mul(nz)];
    if nx == 0 || ny == 0 || nz == 0 {
        return values;
    }
    let plan = axis_plan(kernel, spots, origin, shape, voxel, dose_mode);
    let workers = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1)
        .clamp(1, 32);
    // Each band owns a contiguous run of z slices, so the adds stay in one
    // order and match a single thread. A short lattice is not worth the spawn.
    let bands = if nz < 32 || spots.len() < 8 || workers == 1 {
        1
    } else {
        workers.min(nz / 8).max(1)
    };
    if bands == 1 {
        paint_band(
            kernel,
            spots,
            origin,
            shape,
            voxel,
            dose_mode,
            &plan,
            0,
            nz,
            &mut values,
            scale,
        );
        return values;
    }
    let stride = nz.div_ceil(bands);
    let plan = &plan;
    std::thread::scope(|scope| {
        let plane = nx * ny;
        let mut rest = values.as_mut_slice();
        let mut z0 = 0usize;
        while z0 < nz {
            let z1 = (z0 + stride).min(nz);
            let take = (z1 - z0) * plane;
            let (band, tail) = rest.split_at_mut(take);
            rest = tail;
            let z_from = z0;
            scope.spawn(move || {
                paint_band(
                    kernel, spots, origin, shape, voxel, dose_mode, plan, z_from, z1, band, scale,
                );
            });
            z0 = z1;
        }
    });
    values
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum AxisKey {
    Dose {
        layer: u32,
        width: u32,
        phase: u64,
    },
    Stopping {
        width: u32,
        grow: u32,
        z0: u32,
        z1: u32,
        phase: u64,
    },
}

struct AxisGroup {
    repr: usize,
    center: f32,
    /// First cell of the shared row, in the representative spot's index.
    rel0: isize,
    rel1: isize,
    at: usize,
    len: usize,
    z0: usize,
    z1: usize,
}

struct SpotUse {
    weight: f32,
    ix0: usize,
    ix1: usize,
    iy0: usize,
    iy1: usize,
    iz0: usize,
    iz1: usize,
    x_group: usize,
    y_group: usize,
    x_at: usize,
    y_at: usize,
}

struct Column {
    x_group: usize,
    x_at: usize,
    ix0: usize,
    ix1: usize,
    z0: usize,
    z1: usize,
    spots: Vec<usize>,
}

struct AxisPlan {
    x_groups: Vec<AxisGroup>,
    y_groups: Vec<AxisGroup>,
    spots: Vec<SpotUse>,
    columns: Vec<Column>,
    x_len: usize,
    y_len: usize,
}

fn axis_phase(center: f32, origin: f32, voxel: f32) -> u64 {
    let step = f64::from(voxel);
    let t = (f64::from(center) - f64::from(origin)) / step;
    let frac = t - t.floor();
    if frac == 0.0 {
        0
    } else {
        frac.to_bits()
    }
}

fn axis_key(
    spot: &Prepared,
    layer: usize,
    width: f32,
    center: f32,
    origin: f32,
    voxel: f32,
    dose_mode: bool,
) -> AxisKey {
    let phase = axis_phase(center, origin, voxel);
    let width = width.to_bits();
    if dose_mode {
        // Sigma at depth comes from the layer and this width. The center's
        // fractional cell is `phase`, so a whole spot map shares one row.
        AxisKey::Dose {
            layer: layer as u32,
            width,
            phase,
        }
    } else {
        AxisKey::Stopping {
            width,
            grow: spot.grow.to_bits(),
            z0: spot.z0.to_bits(),
            z1: spot.z1.to_bits(),
            phase,
        }
    }
}

fn voxel_shift(center: f32, repr: f32, voxel: f32) -> isize {
    let delta = (f64::from(center) - f64::from(repr)) / f64::from(voxel);
    let shift = delta.round();
    debug_assert!((delta - shift).abs() < 1e-8);
    shift as isize
}

fn enroll(
    groups: &mut Vec<AxisGroup>,
    map: &mut std::collections::HashMap<AxisKey, usize>,
    key: AxisKey,
    repr: usize,
    center: f32,
    i0: usize,
    i1: usize,
    iz0: usize,
    iz1: usize,
    voxel: f32,
) -> usize {
    if let Some(&id) = map.get(&key) {
        let shift = voxel_shift(center, groups[id].center, voxel);
        let rel0 = i0 as isize - shift;
        let rel1 = i1 as isize - shift;
        let group = &mut groups[id];
        group.rel0 = group.rel0.min(rel0);
        group.rel1 = group.rel1.max(rel1);
        group.z0 = group.z0.min(iz0);
        group.z1 = group.z1.max(iz1);
        id
    } else {
        let id = groups.len();
        groups.push(AxisGroup {
            repr,
            center,
            rel0: i0 as isize,
            rel1: i1 as isize,
            at: 0,
            len: 0,
            z0: iz0,
            z1: iz1,
        });
        map.insert(key, id);
        id
    }
}

fn pack_groups(groups: &mut [AxisGroup]) -> usize {
    let mut cursor = 0usize;
    for group in groups.iter_mut() {
        group.at = cursor;
        group.len = (group.rel1 - group.rel0).max(0) as usize;
        cursor += group.len;
    }
    cursor
}

fn axis_plan(
    kernel: &LayerKernel,
    spots: &[Prepared],
    origin: [f32; 3],
    shape: [usize; 3],
    voxel: f32,
    dose_mode: bool,
) -> AxisPlan {
    let mut x_groups = Vec::new();
    let mut y_groups = Vec::new();
    let mut x_map = std::collections::HashMap::new();
    let mut y_map = std::collections::HashMap::new();
    let mut raw = Vec::new();
    for (index, spot) in spots.iter().enumerate() {
        if spot.weight == 0.0 {
            continue;
        }
        let Some((ix0, ix1, iy0, iy1, iz0, iz1)) = spot_box(spot, origin, shape, voxel, dose_mode)
        else {
            continue;
        };
        let layer = layer_of(kernel, spot.energy);
        let x_group = enroll(
            &mut x_groups,
            &mut x_map,
            axis_key(spot, layer, spot.sx, spot.x, origin[0], voxel, dose_mode),
            index,
            spot.x,
            ix0,
            ix1,
            iz0,
            iz1,
            voxel,
        );
        let y_group = enroll(
            &mut y_groups,
            &mut y_map,
            axis_key(spot, layer, spot.sy, spot.y, origin[1], voxel, dose_mode),
            index,
            spot.y,
            iy0,
            iy1,
            iz0,
            iz1,
            voxel,
        );
        raw.push((index, ix0, ix1, iy0, iy1, iz0, iz1, x_group, y_group));
    }
    let x_len = pack_groups(&mut x_groups);
    let y_len = pack_groups(&mut y_groups);
    let uses: Vec<SpotUse> = raw
        .into_iter()
        .map(|(index, ix0, ix1, iy0, iy1, iz0, iz1, x_group, y_group)| {
            let spot = &spots[index];
            let shift_x = voxel_shift(spot.x, x_groups[x_group].center, voxel);
            let shift_y = voxel_shift(spot.y, y_groups[y_group].center, voxel);
            let x_at = (ix0 as isize - shift_x - x_groups[x_group].rel0) as usize;
            let y_at = (iy0 as isize - shift_y - y_groups[y_group].rel0) as usize;
            debug_assert!(x_at + ix1 - ix0 <= x_groups[x_group].len);
            debug_assert!(y_at + iy1 - iy0 <= y_groups[y_group].len);
            SpotUse {
                weight: spot.weight,
                ix0,
                ix1,
                iy0,
                iy1,
                iz0,
                iz1,
                x_group,
                y_group,
                x_at,
                y_at,
            }
        })
        .collect();
    let columns = columns_of(&uses);
    AxisPlan {
        x_groups,
        y_groups,
        spots: uses,
        columns,
        x_len,
        y_len,
    }
}

fn columns_of(spots: &[SpotUse]) -> Vec<Column> {
    let mut map = std::collections::HashMap::<(usize, usize, usize, usize), usize>::new();
    let mut columns: Vec<Column> = Vec::new();
    for (index, spot) in spots.iter().enumerate() {
        let key = (spot.x_group, spot.x_at, spot.ix0, spot.ix1);
        if let Some(&id) = map.get(&key) {
            let column = &mut columns[id];
            column.z0 = column.z0.min(spot.iz0);
            column.z1 = column.z1.max(spot.iz1);
            column.spots.push(index);
        } else {
            map.insert(key, columns.len());
            columns.push(Column {
                x_group: spot.x_group,
                x_at: spot.x_at,
                ix0: spot.ix0,
                ix1: spot.ix1,
                z0: spot.iz0,
                z1: spot.iz1,
                spots: vec![index],
            });
        }
    }
    columns
}

fn write_group_rows(
    groups: &[AxisGroup],
    spots: &[Prepared],
    kernel: &LayerKernel,
    rows: &mut [f32],
    mut fz: Option<&mut [f32]>,
    origin_axis: f32,
    z_lo: f32,
    voxel: f32,
    dose_mode: bool,
    iz: usize,
    horizontal: bool,
) {
    for (index, group) in groups.iter().enumerate() {
        if iz < group.z0 || iz >= group.z1 || group.len == 0 {
            continue;
        }
        let (mass, sx, sy) = depth_lateral(kernel, &spots[group.repr], z_lo, voxel, dose_mode);
        if let Some(slot) = fz.as_deref_mut() {
            slot[index] = mass;
        }
        if mass == 0.0 {
            continue;
        }
        fill_normal(
            group.center,
            if horizontal { sx } else { sy },
            origin_axis,
            group.rel0,
            voxel,
            &mut rows[group.at..group.at + group.len],
        );
    }
}

fn paint_band(
    kernel: &LayerKernel,
    spots: &[Prepared],
    origin: [f32; 3],
    shape: [usize; 3],
    voxel: f32,
    dose_mode: bool,
    plan: &AxisPlan,
    z0: usize,
    z1: usize,
    values: &mut [f32],
    dose_scale: Option<f32>,
) {
    let [nx, ny, _] = shape;
    let mut x_rows = vec![0.0f32; plan.x_len];
    let mut y_rows = vec![0.0f32; plan.y_len];
    let mut x_fz = vec![0.0f32; plan.x_groups.len()];
    // One scratch slice, so a column can apply its x profile down the whole
    // depth row in one tight loop. The volume is stored after that.
    let mut col_y = vec![0.0f32; plan.columns.len().saturating_mul(ny)];
    let mut span = vec![(ny, 0usize); plan.columns.len()];
    let mut scratch = vec![0.0f32; nx.saturating_mul(ny)];
    for iz in z0..z1 {
        let z_lo = origin[2] + iz as f32 * voxel;
        let local = iz - z0;
        x_fz.fill(0.0);
        span.fill((ny, 0));
        write_group_rows(
            &plan.x_groups,
            spots,
            kernel,
            &mut x_rows,
            Some(&mut x_fz),
            origin[0],
            z_lo,
            voxel,
            dose_mode,
            iz,
            true,
        );
        write_group_rows(
            &plan.y_groups,
            spots,
            kernel,
            &mut y_rows,
            None,
            origin[1],
            z_lo,
            voxel,
            dose_mode,
            iz,
            false,
        );
        let mut y0 = ny;
        let mut y1 = 0usize;
        for (c, column) in plan.columns.iter().enumerate() {
            if iz < column.z0 || iz >= column.z1 {
                continue;
            }
            let fz = x_fz[column.x_group];
            if fz == 0.0 || column.ix0 >= column.ix1 {
                continue;
            }
            let acc = &mut col_y[c * ny..(c + 1) * ny];
            let (lo, hi) = column_weights(plan, column, &y_rows, iz, fz, acc);
            span[c] = (lo, hi);
            y0 = y0.min(lo);
            y1 = y1.max(hi);
        }
        let mut x_lo = nx;
        let mut x_hi = 0usize;
        for (c, column) in plan.columns.iter().enumerate() {
            let (lo, hi) = span[c];
            if lo >= hi || column.ix0 >= column.ix1 {
                continue;
            }
            let x_at = plan.x_groups[column.x_group].at + column.x_at;
            let fx = &x_rows[x_at..x_at + (column.ix1 - column.ix0)];
            paint_column(
                &mut scratch,
                nx,
                column.ix0,
                fx,
                &mut col_y[c * ny..(c + 1) * ny],
                lo,
                hi,
            );
            x_lo = x_lo.min(column.ix0);
            x_hi = x_hi.max(column.ix1);
        }
        if x_lo >= x_hi {
            continue;
        }
        let width = x_hi - x_lo;
        for iy in y0..y1 {
            let base = nx * (iy + ny * local) + x_lo;
            flush_row(
                &mut values[base..base + width],
                &mut scratch[iy * nx + x_lo..iy * nx + x_hi],
                dose_scale,
            );
        }
    }
}

/// Apply one column's x profile down its y weights. Kept separate so the
/// multiply is a small loop the compiler will vectorize.
#[inline(never)]
fn paint_column(
    scratch: &mut [f32],
    nx: usize,
    ix0: usize,
    fx: &[f32],
    weights: &mut [f32],
    y0: usize,
    y1: usize,
) {
    let width = fx.len();
    for iy in y0..y1 {
        let row = weights[iy];
        weights[iy] = 0.0;
        if row == 0.0 {
            continue;
        }
        let start = iy * nx + ix0;
        for (slot, weight) in scratch[start..start + width].iter_mut().zip(fx) {
            *slot += row * weight;
        }
    }
}

#[inline(never)]
fn flush_row(dst: &mut [f32], src: &mut [f32], dose_scale: Option<f32>) {
    if let Some(scale) = dose_scale {
        for (slot, part) in dst.iter_mut().zip(src.iter()) {
            *slot += *part * scale;
        }
    } else {
        for (slot, part) in dst.iter_mut().zip(src.iter()) {
            *slot += *part;
        }
    }
    src.fill(0.0);
}

/// Sum the y profiles of one column into `acc`. The caller clears what this writes.
fn column_weights(
    plan: &AxisPlan,
    column: &Column,
    y_rows: &[f32],
    iz: usize,
    fz: f32,
    acc: &mut [f32],
) -> (usize, usize) {
    let mut ymin = acc.len();
    let mut ymax = 0usize;
    for &index in &column.spots {
        let spot = &plan.spots[index];
        if iz < spot.iz0 || iz >= spot.iz1 {
            continue;
        }
        let y0 = spot.iy0;
        let y1 = spot.iy1;
        if y0 >= y1 {
            continue;
        }
        let y_at = plan.y_groups[spot.y_group].at + spot.y_at;
        let fy = &y_rows[y_at..y_at + (y1 - y0)];
        let scale = spot.weight * fz;
        for (ky, iy) in (y0..y1).enumerate() {
            acc[iy] += scale * fy[ky];
        }
        ymin = ymin.min(y0);
        ymax = ymax.max(y1);
    }
    (ymin, ymax)
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
        let row = layer_of(kernel, spot.energy);
        let fz = super::kernel::row_at(kernel, row, z_lo + voxel, false)
            - super::kernel::row_at(kernel, row, z_lo, false);
        let mcs = super::kernel::row_at(kernel, row, z_lo + 0.5 * voxel, true).max(0.0);
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
    let count = i1.saturating_sub(i0);
    if count == 0 {
        return;
    }
    out.resize(count, 0.0);
    fill_normal(center, sigma, origin, i0 as isize, voxel, out);
}

/// Cell integrals of one Gaussian. `i0` is the first cell and may sit left of the grid.
fn fill_normal(center: f32, sigma: f32, origin: f32, i0: isize, voxel: f32, out: &mut [f32]) {
    if out.is_empty() {
        return;
    }
    // Adjacent cells share an edge, so one erf covers both sides of that face.
    let sig = f64::from(sigma).max(1e-3) * std::f64::consts::SQRT_2;
    let center = f64::from(center);
    let origin = f64::from(origin);
    let step = f64::from(voxel);
    let mut prev = erf_as((origin + i0 as f64 * step - center) / sig);
    for (i, slot) in out.iter_mut().enumerate() {
        let edge = i0 + i as isize + 1;
        let next = erf_as((origin + edge as f64 * step - center) / sig);
        *slot = (0.5 * (next - prev)) as f32;
        prev = next;
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
        .clamp(1, 32);
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
    use super::super::{csda_range_mm, Pencil, Quantity, Volume, WATER};
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
    fn overlapping_grids_match() {
        let pencil = Pencil {
            x: 40.0,
            y: 0.0,
            sx: 4.0,
            sy: 4.0,
            energy: 160.0,
            amount: 0.05,
        };
        let natural = analytic_volume(
            WATER,
            &[pencil],
            Quantity::Dose,
            1.0,
            0.0,
            0.0,
            1.0,
            2.0e-8,
            10.0,
        );
        let nz = natural.shape[2];
        let z0 = natural.origin[2];
        let nx = 128usize;
        let wide = analytic_on(
            WATER,
            &[pencil],
            Quantity::Dose,
            1.0,
            0.0,
            0.0,
            1.0,
            2.0e-8,
            10.0,
            Some(([0.0, -1024.0, z0], [nx, 1104, nz], 1.0)),
            None,
        );
        let small = analytic_on(
            WATER,
            &[pencil],
            Quantity::Dose,
            1.0,
            0.0,
            0.0,
            1.0,
            2.0e-8,
            10.0,
            Some(([0.0, -64.0, z0], [nx, 128, nz], 1.0)),
            None,
        );
        let shift = 1024 - 64;
        for z in 0..nz {
            for y in 0..128 {
                for x in 0..nx {
                    assert_eq!(small.get(x, y, z), wide.get(x, y + shift, z));
                }
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
            0.0,
            0.0,
            1.0,
            2.0e-8,
            10.0,
            0.0,
        );
        assert!(frame.planes_only(), "visits {}", frame.visits);
    }

    #[test]
    fn split_bands_still_fill_the_depth() {
        let pencils: Vec<_> = (0..8)
            .map(|i| Pencil {
                x: i as f32 * 6.0,
                y: 0.0,
                sx: 3.0,
                sy: 3.0,
                energy: 140.0,
                amount: 0.05,
            })
            .collect();
        let volume = analytic_volume(
            WATER,
            &pencils,
            Quantity::Dose,
            1.0,
            0.0,
            0.0,
            1.0,
            2.0e-8,
            10.0,
        );
        let [nx, ny, nz] = volume.shape;
        let mut first = None;
        let mut last = None;
        for z in 0..nz {
            let hit = (0..ny).any(|y| (0..nx).any(|x| volume.get(x, y, z) > 0.0));
            if hit {
                first.get_or_insert(z);
                last = Some(z);
            }
        }
        let span = last.unwrap_or(0) - first.unwrap_or(0);
        assert!(span > nz / 3, "dose span {span} of {nz}");
    }

    #[test]
    fn split_bands_match_a_single_pass() {
        let pencils: Vec<_> = (0..8)
            .map(|i| Pencil {
                x: i as f32 * 6.0,
                y: (i as f32 - 3.0) * 4.0,
                sx: 3.0,
                sy: 3.0,
                energy: 140.0,
                amount: 0.05,
            })
            .collect();
        let (kernel, spots) = super::prepare(WATER, &pencils, Quantity::Dose, 1.0, 0.0, 2e-8, 10.0);
        let (origin, shape) = super::dose_grid(&spots, 1.0, None);
        let [nx, ny, nz] = shape;
        let plan = super::axis_plan(&kernel, &spots, origin, shape, 1.0, true);
        let mut one = vec![0.0f32; nx * ny * nz];
        super::paint_band(
            &kernel, &spots, origin, shape, 1.0, true, &plan, 0, nz, &mut one, None,
        );
        let mid = nz / 2;
        let mut lo = vec![0.0f32; nx * ny * mid];
        let mut hi = vec![0.0f32; nx * ny * (nz - mid)];
        super::paint_band(
            &kernel, &spots, origin, shape, 1.0, true, &plan, 0, mid, &mut lo, None,
        );
        super::paint_band(
            &kernel, &spots, origin, shape, 1.0, true, &plan, mid, nz, &mut hi, None,
        );
        lo.extend(hi);
        assert_eq!(lo, one);
    }

    #[test]
    fn a_gaussian_row_matches_the_cell_integral() {
        let mut row = Vec::new();
        super::normal_row(1.2, 3.0, -4.0, 2, 9, 1.0, &mut row);
        assert_eq!(row.len(), 7);
        for (i, weight) in row.iter().enumerate() {
            let index = 2 + i;
            let lo = -4.0 + index as f64;
            let cell = super::normal_mass(1.2, 3.0, lo, lo + 1.0) as f32;
            assert!((weight - cell).abs() < 1e-6, "{weight} vs {cell}");
        }
    }

    #[test]
    fn auto_margin_is_sigma_past_the_range() {
        let pencil = Pencil {
            x: 0.0,
            y: 0.0,
            sx: 4.0,
            sy: 4.0,
            energy: 100.0,
            amount: 0.1,
        };
        let bare = dose_frame(
            WATER,
            &[pencil],
            Quantity::Dose,
            1.0,
            0.0,
            0.0,
            1.0,
            2e-8,
            10.0,
            0.0,
        );
        let five = dose_frame(
            WATER,
            &[pencil],
            Quantity::Dose,
            1.0,
            0.0,
            0.0,
            1.0,
            2e-8,
            10.0,
            5.0,
        );
        assert_eq!(bare.shape[0], five.shape[0]);
        assert_eq!(bare.shape[1], five.shape[1]);
        assert!(five.shape[2] <= bare.shape[2]);
        assert!(five.shape[2] + 2 >= bare.shape[2]);
        let tight = dose_frame(
            WATER,
            &[pencil],
            Quantity::Dose,
            1.0,
            0.0,
            0.0,
            1.0,
            2e-8,
            10.0,
            1.0,
        );
        assert!(
            tight.shape[2] < five.shape[2],
            "{} vs {}",
            tight.shape[2],
            five.shape[2]
        );
        let volume = Volume {
            origin: [-1.0, -1.0, -4.0],
            shape: [3, 3, 4],
            voxel: 1.0,
            values: vec![0.0; 36],
        };
        let (depth, _) = volume.depth_profile(1, 1);
        assert!((depth[0] - 0.5).abs() < 1e-4, "{depth:?}");
        let (lateral, _) = volume.lateral_profile(1, 1, 1);
        assert!(lateral[1].abs() < 1e-4, "{lateral:?}");
        assert!(lateral[0] < 0.0 && lateral[2] > 0.0);
        let (along, _) = volume.longitudinal_profile(1, 1, 1);
        assert!((along[0] - 1.0).abs() < 1e-4, "{along:?}");
    }

    #[test]
    fn a_spot_lattice_shares_one_lateral_row() {
        let pencils: Vec<_> = (0..4)
            .flat_map(|iy| {
                (0..4).map(move |ix| Pencil {
                    x: ix as f32 * 5.0,
                    y: iy as f32 * 5.0,
                    sx: 4.0,
                    sy: 4.0,
                    energy: 140.0,
                    amount: 0.02,
                })
            })
            .collect();
        let (kernel, spots) = super::prepare(WATER, &pencils, Quantity::Dose, 1.0, 0.0, 2e-8, 10.0);
        let (origin, shape) = super::dose_grid(&spots, 1.0, None);
        let plan = super::axis_plan(&kernel, &spots, origin, shape, 1.0, true);
        assert_eq!(plan.spots.len(), 16);
        assert_eq!(plan.x_groups.len(), 1);
        assert_eq!(plan.y_groups.len(), 1);
        assert_eq!(plan.columns.len(), 4);
        assert!(plan.columns.iter().all(|column| column.spots.len() == 4));
    }

    #[test]
    fn shared_rows_match_adding_each_spot() {
        let pencils = [
            Pencil {
                x: 0.0,
                y: 0.0,
                sx: 4.0,
                sy: 4.0,
                energy: 120.0,
                amount: 0.02,
            },
            Pencil {
                x: 5.0,
                y: 10.0,
                sx: 4.0,
                sy: 4.0,
                energy: 120.0,
                amount: 0.03,
            },
            Pencil {
                x: 1.7,
                y: -2.4,
                sx: 4.0,
                sy: 6.0,
                energy: 120.0,
                amount: 0.01,
            },
            Pencil {
                x: -8.0,
                y: 4.0,
                sx: 5.0,
                sy: 5.0,
                energy: 160.0,
                amount: 0.04,
            },
        ];
        for (quantity, dose_mode) in [(Quantity::Dose, true), (Quantity::Mu, false)] {
            let (kernel, spots) = super::prepare(WATER, &pencils, quantity, 1.0, 0.0, 2e-8, 10.0);
            let (origin, shape) = super::dose_grid(&spots, 1.0, None);
            let whole =
                super::deposit_lattice(&kernel, &spots, origin, shape, 1.0, dose_mode, None);
            let mut acc = vec![0.0f32; whole.len()];
            for spot in &spots {
                let one = super::deposit_lattice(
                    &kernel,
                    std::slice::from_ref(spot),
                    origin,
                    shape,
                    1.0,
                    dose_mode,
                    None,
                );
                for (sum, part) in acc.iter_mut().zip(&one) {
                    *sum += *part;
                }
            }
            assert_eq!(acc, whole);
        }
    }

    #[test]
    fn a_column_stays_close_to_adding_each_spot() {
        let mut pencils = Vec::new();
        for energy in [110.0f32, 150.0] {
            for iy in 0..8 {
                for ix in 0..2 {
                    pencils.push(Pencil {
                        x: ix as f32 * 8.0,
                        y: iy as f32 * 5.0 - 20.0,
                        sx: 4.0,
                        sy: 4.0,
                        energy,
                        amount: 0.01 * (1.0 + iy as f32),
                    });
                }
            }
        }
        for (quantity, dose_mode) in [(Quantity::Dose, true), (Quantity::Mu, false)] {
            let (kernel, spots) = super::prepare(WATER, &pencils, quantity, 1.0, 0.0, 2e-8, 10.0);
            let (origin, shape) = super::dose_grid(&spots, 2.0, None);
            let whole =
                super::deposit_lattice(&kernel, &spots, origin, shape, 2.0, dose_mode, None);
            let mut acc = vec![0.0f32; whole.len()];
            for spot in &spots {
                let one = super::deposit_lattice(
                    &kernel,
                    std::slice::from_ref(spot),
                    origin,
                    shape,
                    2.0,
                    dose_mode,
                    None,
                );
                for (sum, part) in acc.iter_mut().zip(&one) {
                    *sum += *part;
                }
            }
            let mut peak = 0.0f32;
            let mut worst = 0.0f32;
            for (got, separate) in whole.iter().zip(&acc) {
                peak = peak.max(got.abs()).max(separate.abs());
                worst = worst.max((got - separate).abs());
            }
            assert!(
                worst <= 1e-5 * peak.max(1e-6),
                "worst {worst} peak {peak} mode {dose_mode}"
            );
        }
    }
}
