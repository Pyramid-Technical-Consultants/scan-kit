//! Turntable ray march through a dose box.
//!
//! This is the vispy turntable march from `dose_volume_raycast.py`: one sample
//! per voxel from the front face to the back, Integrate / Maximum / Transparent.
//! `scan-kit-plot` repeats it in WGSL. Empty 8³ bricks are skipped; the brick
//! grid is dilated by one so linear and cubic samples on a live edge still run.

use super::Volume;

/// Voxels on a side of one occupancy brick. The plot shader uses the same size.
pub const BRICK: usize = 8;
const MAX_STEPS: usize = 1024;
/// Film samples used to find the brightest integrate ray. Python reads the max
/// back from the full frame; this probe plus the peak voxel is the same window
/// for a smooth beam. ponytail: a sheet that misses the probe and the peak clips
/// to the top of the ramp. A full-frame reduction would match the readback.
const RAY_PROBE: usize = 16;
/// Optical depth of 1 at gain 1, in millimetres. `VIEW_DEPTH_MM` in the Python fill.
const VIEW_DEPTH_MM: f32 = 8.0;
/// Rays dimmer than this fraction of the color scale are the background.
const DOSE_FLOOR: f32 = 0.02;
/// Vispy `TurntableCamera` field of view. `0` is orthographic.
pub const FOV_Y: f32 = 45.0 * std::f32::consts::PI / 180.0;

/// Orbit and color state for one 3D cell. Angles are radians, vispy turntable.
#[derive(Clone, Copy, Debug)]
pub struct RayView {
    pub azimuth: f32,
    pub elevation: f32,
    pub zoom: f32,
    pub aspect: f32,
    pub mode: u8,
    pub filter: u8,
    pub gain: f32,
    pub opacity: f32,
    pub lo: f32,
    pub hi: f32,
    pub ramp: u8,
    /// Integrate color scale in dose × mm: the brightest ray of this view.
    pub line_scale: f32,
    /// Vertical field of view, radians. `0` is orthographic.
    pub fov: f32,
    /// Look-at offset from the volume center, millimetres.
    pub center: [f32; 3],
    /// Degrees about +X. The lattice stays in beam coordinates; only the view box turns.
    pub gantry: f32,
}

/// One read of the volume: brick occupancy, the integrate window, and the three
/// plane integrals. Built when the dose grid is attached, then reused.
#[derive(Clone, Debug, PartialEq)]
pub struct VolumeScan {
    pub bricks: Vec<f32>,
    pub brick_shape: [usize; 3],
    pub line_scale: f32,
    pub peak: f32,
    /// Max |dose|. The slice atlas stores a signed window around this.
    pub abs_peak: f32,
    pub peak_at: [usize; 3],
    /// Sum through z, y, and x, in dose × mm. Axial, coronal, sagittal.
    pub integrals: [Vec<f32>; 3],
    pub integral_peaks: [f32; 3],
}

impl VolumeScan {
    /// True when this scan is the occupancy and plane sums of a cube with `shape`.
    pub fn covers(&self, shape: [usize; 3]) -> bool {
        let expect = [
            shape[0].div_ceil(BRICK).max(1),
            shape[1].div_ceil(BRICK).max(1),
            shape[2].div_ceil(BRICK).max(1),
        ];
        let bricks = expect[0]
            .saturating_mul(expect[1])
            .saturating_mul(expect[2]);
        self.brick_shape == expect
            && self.bricks.len() == bricks
            && self.integrals[0].len() == shape[0].saturating_mul(shape[1])
            && self.integrals[1].len() == shape[0].saturating_mul(shape[2])
            && self.integrals[2].len() == shape[1].saturating_mul(shape[2])
    }
}

/// Max |dose| in each 8³ brick, dilated by one brick. Zeros are safe to leap.
pub fn brick_grid(volume: &Volume) -> (Vec<f32>, [usize; 3]) {
    let scan = scan_volume(volume);
    (scan.bricks, scan.brick_shape)
}

/// Brightest axis-aligned line integral, in dose × mm.
pub fn line_scale(volume: &Volume) -> f32 {
    scan_volume(volume).line_scale
}

pub fn scan_volume(volume: &Volume) -> VolumeScan {
    scan_axes(&volume.values, volume.shape, volume.spacing)
}

pub fn scan_grid(values: &[f32], shape: [usize; 3], voxel: f32) -> VolumeScan {
    scan_axes(values, shape, [voxel, voxel, voxel])
}

pub fn scan_spacing(values: &[f32], shape: [usize; 3], spacing: [f32; 3]) -> VolumeScan {
    scan_axes(values, shape, spacing)
}

fn scan_axes(values: &[f32], shape: [usize; 3], spacing: [f32; 3]) -> VolumeScan {
    let [nx, ny, nz] = shape;
    let voxels = nx.saturating_mul(ny).saturating_mul(nz);
    let brick_z = nz.div_ceil(BRICK).max(1);
    let workers = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1)
        .clamp(1, 32);
    // A short cube is one pass. Past that, z bands own whole bricks so the
    // occupancy max needs no merge, and the plane sums add in z order.
    let bands = if voxels < 250_000 || workers == 1 || brick_z < 2 {
        1
    } else {
        workers.min(brick_z)
    };
    if bands == 1 {
        return scan_axes_serial(values, shape, spacing);
    }
    let per = brick_z.div_ceil(bands);
    let mut ranges = Vec::new();
    let mut band = 0;
    while band * per * BRICK < nz {
        let z0 = band * per * BRICK;
        let z1 = ((band + 1) * per * BRICK).min(nz);
        ranges.push((z0, z1));
        band += 1;
    }
    let parts = std::thread::scope(|scope| {
        let mut joins = Vec::with_capacity(ranges.len());
        for (z0, z1) in ranges {
            joins.push(scope.spawn(move || accumulate_z(values, shape, z0, z1)));
        }
        joins
            .into_iter()
            .map(|join| join.join().expect("scan band"))
            .collect::<Vec<_>>()
    });
    let mut y_sum = vec![0.0f32; nx.saturating_mul(nz)];
    let mut coronal = vec![0.0f32; nx.saturating_mul(nz)];
    let mut sagittal = vec![0.0f32; ny.saturating_mul(nz)];
    let brick_shape = [
        nx.div_ceil(BRICK).max(1),
        ny.div_ceil(BRICK).max(1),
        brick_z,
    ];
    let mut raw = vec![0.0f32; brick_shape[0] * brick_shape[1] * brick_shape[2]];
    let mut axial = vec![0.0f32; nx.saturating_mul(ny)];
    let mut z_sum = vec![0.0f32; nx.saturating_mul(ny)];
    let mut x_best = 0.0f32;
    let mut peak = 0.0f32;
    let mut abs_peak = 0.0f32;
    let mut peak_at = [0usize; 3];
    let plane = brick_shape[0] * brick_shape[1];
    for part in parts {
        if part.peak > peak {
            peak = part.peak;
            peak_at = part.peak_at;
        }
        abs_peak = abs_peak.max(part.abs_peak);
        x_best = x_best.max(part.x_best);
        for (slot, value) in axial.iter_mut().zip(&part.axial) {
            *slot += *value;
        }
        for (slot, value) in z_sum.iter_mut().zip(&part.z_sum) {
            *slot += *value;
        }
        for (index, value) in part.y_sum.into_iter().enumerate() {
            let z = part.z0 + index / nx.max(1);
            let x = index % nx.max(1);
            if z < nz {
                y_sum[x + nx * z] = value;
            }
        }
        for (index, value) in part.coronal.into_iter().enumerate() {
            let z = part.z0 + index / nx.max(1);
            let x = index % nx.max(1);
            if z < nz {
                coronal[x + nx * z] = value;
            }
        }
        for (index, value) in part.sagittal.into_iter().enumerate() {
            let z = part.z0 + index / ny.max(1);
            let y = index % ny.max(1);
            if z < nz {
                sagittal[y + ny * z] = value;
            }
        }
        let brick0 = part.z0 / BRICK;
        for (index, value) in part.bricks.into_iter().enumerate() {
            let at = brick0 * plane + index;
            if at < raw.len() {
                raw[at] = value;
            }
        }
    }
    finish_scan(
        spacing,
        brick_shape,
        raw,
        axial,
        coronal,
        sagittal,
        x_best,
        y_sum,
        z_sum,
        peak,
        abs_peak,
        peak_at,
    )
}

struct ZPart {
    z0: usize,
    abs_peak: f32,
    peak: f32,
    peak_at: [usize; 3],
    x_best: f32,
    axial: Vec<f32>,
    z_sum: Vec<f32>,
    y_sum: Vec<f32>,
    coronal: Vec<f32>,
    sagittal: Vec<f32>,
    bricks: Vec<f32>,
}

fn accumulate_z(values: &[f32], shape: [usize; 3], z0: usize, z1: usize) -> ZPart {
    let [nx, ny, nz] = shape;
    let brick_shape = [
        nx.div_ceil(BRICK).max(1),
        ny.div_ceil(BRICK).max(1),
        nz.div_ceil(BRICK).max(1),
    ];
    let z_count = z1.saturating_sub(z0);
    let brick0 = z0 / BRICK;
    let brick1 = if z_count == 0 {
        brick0
    } else {
        (z1 - 1) / BRICK + 1
    };
    let mut y_sum = vec![0.0f32; nx.saturating_mul(z_count)];
    let mut coronal = vec![0.0f32; nx.saturating_mul(z_count)];
    let mut sagittal = vec![0.0f32; ny.saturating_mul(z_count)];
    let mut axial = vec![0.0f32; nx.saturating_mul(ny)];
    let mut z_sum = vec![0.0f32; nx.saturating_mul(ny)];
    let mut bricks = vec![0.0f32; brick_shape[0] * brick_shape[1] * brick1.saturating_sub(brick0)];
    let mut x_best = 0.0f32;
    let mut peak = 0.0f32;
    let mut abs_peak = 0.0f32;
    let mut peak_at = [0usize; 3];
    if nx > 0 && ny > 0 && z0 < z1 && values.len() >= nx * ny * nz {
        for z in z0..z1 {
            for y in 0..ny {
                let base = nx * (y + ny * z);
                let mut along_x = 0.0f32;
                let mut signed_x = 0.0f32;
                for x in 0..nx {
                    let value = values[base + x];
                    let mag = value.abs();
                    abs_peak = abs_peak.max(mag);
                    along_x += mag;
                    signed_x += value;
                    y_sum[x + nx * (z - z0)] += mag;
                    z_sum[x + nx * y] += mag;
                    axial[x + nx * y] += value;
                    coronal[x + nx * (z - z0)] += value;
                    if mag > 0.0 {
                        let brick = (x / BRICK)
                            + brick_shape[0]
                                * ((y / BRICK) + brick_shape[1] * (z / BRICK - brick0));
                        bricks[brick] = bricks[brick].max(mag);
                    }
                    if value > peak {
                        peak = value;
                        peak_at = [x, y, z];
                    }
                }
                x_best = x_best.max(along_x);
                sagittal[y + ny * (z - z0)] = signed_x;
            }
        }
    }
    ZPart {
        z0,
        abs_peak,
        peak,
        peak_at,
        x_best,
        axial,
        z_sum,
        y_sum,
        coronal,
        sagittal,
        bricks,
    }
}

fn scan_axes_serial(values: &[f32], shape: [usize; 3], spacing: [f32; 3]) -> VolumeScan {
    let [nx, ny, nz] = shape;
    let brick_shape = [
        nx.div_ceil(BRICK).max(1),
        ny.div_ceil(BRICK).max(1),
        nz.div_ceil(BRICK).max(1),
    ];
    let mut raw = vec![0.0f32; brick_shape[0] * brick_shape[1] * brick_shape[2]];
    let mut y_sum = vec![0.0f32; nx.saturating_mul(nz)];
    let mut z_sum = vec![0.0f32; nx.saturating_mul(ny)];
    let mut axial = vec![0.0f32; nx.saturating_mul(ny)];
    let mut coronal = vec![0.0f32; nx.saturating_mul(nz)];
    let mut sagittal = vec![0.0f32; ny.saturating_mul(nz)];
    let mut x_best = 0.0f32;
    let mut peak = 0.0f32;
    let mut abs_peak = 0.0f32;
    let mut peak_at = [0usize; 3];
    if nx > 0 && ny > 0 && nz > 0 && values.len() >= nx * ny * nz {
        for z in 0..nz {
            for y in 0..ny {
                let base = nx * (y + ny * z);
                let mut along_x = 0.0f32;
                let mut signed_x = 0.0f32;
                for x in 0..nx {
                    let value = values[base + x];
                    let mag = value.abs();
                    abs_peak = abs_peak.max(mag);
                    along_x += mag;
                    signed_x += value;
                    y_sum[x + nx * z] += mag;
                    z_sum[x + nx * y] += mag;
                    axial[x + nx * y] += value;
                    coronal[x + nx * z] += value;
                    if mag > 0.0 {
                        let brick = (x / BRICK)
                            + brick_shape[0] * ((y / BRICK) + brick_shape[1] * (z / BRICK));
                        raw[brick] = raw[brick].max(mag);
                    }
                    if value > peak {
                        peak = value;
                        peak_at = [x, y, z];
                    }
                }
                x_best = x_best.max(along_x);
                sagittal[y + ny * z] = signed_x;
            }
        }
    }
    finish_scan(
        spacing,
        brick_shape,
        raw,
        axial,
        coronal,
        sagittal,
        x_best,
        y_sum,
        z_sum,
        peak,
        abs_peak,
        peak_at,
    )
}

fn finish_scan(
    spacing: [f32; 3],
    brick_shape: [usize; 3],
    raw: Vec<f32>,
    mut axial: Vec<f32>,
    mut coronal: Vec<f32>,
    mut sagittal: Vec<f32>,
    x_best: f32,
    y_sum: Vec<f32>,
    z_sum: Vec<f32>,
    peak: f32,
    abs_peak: f32,
    peak_at: [usize; 3],
) -> VolumeScan {
    let spacing = spacing.map(|mm| mm.max(1e-6));
    let y_best = y_sum.into_iter().fold(0.0f32, f32::max);
    let z_best = z_sum.into_iter().fold(0.0f32, f32::max);
    // Axial sums through Z, coronal through Y, sagittal through X.
    for sample in &mut axial {
        *sample *= spacing[2];
    }
    for sample in &mut coronal {
        *sample *= spacing[1];
    }
    for sample in &mut sagittal {
        *sample *= spacing[0];
    }
    let integral_peaks = [
        axial.iter().copied().fold(0.0f32, f32::max).max(1e-6),
        coronal.iter().copied().fold(0.0f32, f32::max).max(1e-6),
        sagittal.iter().copied().fold(0.0f32, f32::max).max(1e-6),
    ];
    VolumeScan {
        bricks: dilate_bricks(&raw, brick_shape),
        brick_shape,
        line_scale: (x_best * spacing[0])
            .max(y_best * spacing[1])
            .max(z_best * spacing[2])
            .max(1e-6),
        peak: peak.max(0.0),
        abs_peak,
        peak_at,
        integrals: [axial, coronal, sagittal],
        integral_peaks,
    }
}

fn dilate_bricks(raw: &[f32], shape: [usize; 3]) -> Vec<f32> {
    let mut dilated = vec![0.0f32; raw.len()];
    for z in 0..shape[2] {
        for y in 0..shape[1] {
            for x in 0..shape[0] {
                let mut peak = 0.0f32;
                for dz in -1..=1 {
                    for dy in -1..=1 {
                        for dx in -1..=1 {
                            let xx = x as i32 + dx;
                            let yy = y as i32 + dy;
                            let zz = z as i32 + dz;
                            if xx < 0
                                || yy < 0
                                || zz < 0
                                || xx >= shape[0] as i32
                                || yy >= shape[1] as i32
                                || zz >= shape[2] as i32
                            {
                                continue;
                            }
                            let index =
                                xx as usize + shape[0] * (yy as usize + shape[1] * zz as usize);
                            peak = peak.max(raw[index]);
                        }
                    }
                }
                dilated[x + shape[0] * (y + shape[1] * z)] = peak;
            }
        }
    }
    dilated
}

/// Straight RGBA for one film pixel. `uv` is 0..1, y down, over the marched image.
pub fn ray_rgba(
    volume: &Volume,
    bricks: &[f32],
    brick_shape: [usize; 3],
    view: RayView,
    uv: [f32; 2],
) -> [f32; 4] {
    march(volume, bricks, brick_shape, view, uv).1
}

/// `mode`: 0 integrate, 1 maximum, 2 transparent. `filter`: 0 nearest, 1 linear, 2 cubic.
pub fn raymarch(
    volume: &Volume,
    azimuth: f32,
    elevation: f32,
    mode: u8,
    filter: u8,
    cols: usize,
    rows: usize,
) -> Vec<f32> {
    let cols = cols.max(1);
    let rows = rows.max(1);
    let scan = scan_volume(volume);
    let view = plain_view(volume, azimuth, elevation, mode, filter, scan.line_scale);
    let mut image = vec![0.0; cols * rows];
    for row in 0..rows {
        for col in 0..cols {
            let u = (col as f32 + 0.5) / cols as f32;
            let v = (row as f32 + 0.5) / rows as f32;
            image[col + cols * row] = march(volume, &scan.bricks, scan.brick_shape, view, [u, v]).0;
        }
    }
    image
}

pub fn ray_value(
    volume: &Volume,
    azimuth: f32,
    elevation: f32,
    mode: u8,
    filter: u8,
    u: f32,
    v: f32,
) -> f32 {
    let scan = scan_volume(volume);
    let view = plain_view(volume, azimuth, elevation, mode, filter, scan.line_scale);
    march(volume, &scan.bricks, scan.brick_shape, view, [u, v]).0
}

/// Brightest |line integral| on this view, in dose × mm.
///
/// `peak_at` is the voxel `scan_volume` already found. The 16×16 probe can miss
/// that ray, and walking the lattice again to find it costs more than the probe
/// on a sparse beam.
///
/// Python's auto window puts that ray at the top of the colormap and zero at
/// the bottom. The axis integral is a different direction, so a side view of a
/// deep beam stays dark when the window is the axis sum.
pub fn view_ray_scale(
    volume: &Volume,
    bricks: &[f32],
    brick_shape: [usize; 3],
    azimuth: f32,
    elevation: f32,
    zoom: f32,
    aspect: f32,
    filter: u8,
    fov: f32,
    center: [f32; 3],
    gantry: f32,
    peak_at: [usize; 3],
) -> f32 {
    let view = RayView {
        azimuth,
        elevation,
        zoom,
        aspect,
        mode: 0,
        filter,
        gain: 1.0,
        opacity: 1.0,
        lo: 0.0,
        hi: 1.0,
        ramp: 1,
        line_scale: 1.0,
        fov,
        center,
        gantry,
    };
    let mut best = 0.0f32;
    let mut consider = |uv: [f32; 2]| {
        best = best.max(march(volume, bricks, brick_shape, view, uv).0.abs());
    };
    for row in 0..RAY_PROBE {
        for col in 0..RAY_PROBE {
            consider([
                (col as f32 + 0.5) / RAY_PROBE as f32,
                (row as f32 + 0.5) / RAY_PROBE as f32,
            ]);
        }
    }
    let [nx, ny, nz] = volume.shape;
    let [ix, iy, iz] = peak_at;
    if ix < nx && iy < ny && iz < nz {
        let point = [
            volume.mm_of(0, ix),
            volume.mm_of(1, iy),
            volume.mm_of(2, iz),
        ];
        if let Some(uv) = film_uv(volume, view, point) {
            consider(uv);
            let step = 1.0 / nx.max(ny).max(nz) as f32;
            consider([(uv[0] - step).clamp(0.0, 1.0), uv[1]]);
            consider([(uv[0] + step).clamp(0.0, 1.0), uv[1]]);
            consider([uv[0], (uv[1] - step).clamp(0.0, 1.0)]);
            consider([uv[0], (uv[1] + step).clamp(0.0, 1.0)]);
        }
    }
    best.max(1e-6)
}

fn plain_view(
    volume: &Volume,
    azimuth: f32,
    elevation: f32,
    mode: u8,
    filter: u8,
    line_scale: f32,
) -> RayView {
    let peak = volume
        .values
        .iter()
        .copied()
        .fold(0.0f32, f32::max)
        .max(1e-6);
    RayView {
        azimuth,
        elevation,
        zoom: 1.0,
        aspect: 1.0,
        mode,
        filter,
        gain: 1.0,
        opacity: 1.0,
        lo: 0.0,
        hi: peak,
        ramp: 1,
        line_scale,
        fov: FOV_Y,
        center: [0.0; 3],
        gantry: 0.0,
    }
}

/// Vertical size of the fitted view, in millimetres. Perspective and ortho share it.
pub fn film_height(extent: [f32; 3], aspect: f32, zoom: f32) -> f32 {
    let aspect = aspect.max(1e-3);
    let zoom = zoom.max(0.05);
    let pad = |edge: f32| (edge * 0.12).max(1.0);
    let mut rx = extent[0] + 2.0 * pad(extent[0]);
    let mut ry = extent[1] + 2.0 * pad(extent[1]);
    let mut rz = extent[2] + 2.0 * pad(extent[2]);
    if aspect > 1.0 {
        rx /= aspect;
        ry /= aspect;
    } else {
        rz *= aspect;
    }
    let rxs = (rx * rx + ry * ry).sqrt();
    let rys = (rx * rx + ry * ry + rz * rz).sqrt();
    let scale = rxs.max(rys) * 1.04 * zoom;
    if aspect > 1.0 {
        scale
    } else {
        scale / aspect
    }
}

fn march(
    volume: &Volume,
    bricks: &[f32],
    brick_shape: [usize; 3],
    view: RayView,
    uv: [f32; 2],
) -> (f32, [f32; 4]) {
    let Some((eye, dir, t0, t1)) = cast(volume, view, uv) else {
        return (0.0, [0.0; 4]);
    };
    let spacing = [volume.axis(0), volume.axis(1), volume.axis(2)];
    let step = spacing[0].min(spacing[1]).min(spacing[2]);
    let dist = t1 - t0;
    if dist < 1e-3 {
        return (0.0, [0.0; 4]);
    }
    let nstep = ((dist / step).ceil() as usize).clamp(1, MAX_STEPS);
    let entry = add(eye, scale(dir, t0));
    let step_v = scale(dir, step);
    let typical = view.hi.abs().max(view.lo.abs()).max(1e-6);
    let mut integ = 0.0f32;
    let mut peak = 0.0f32;
    let mut trans = 1.0f32;
    let mut col = [0.0f32; 3];
    let mut i = 0usize;
    while i < nstep {
        let p = add(entry, scale(step_v, i as f32 + 0.5));
        let rotated = view.gantry.abs() > 1e-3;
        let q = if rotated {
            lattice_point(p, lattice_center(volume), view.gantry)
        } else {
            p
        };
        let step_dir = if rotated {
            lattice_dir(dir, view.gantry)
        } else {
            dir
        };
        let cell = [
            (q[0] - volume.origin[0]) / spacing[0],
            (q[1] - volume.origin[1]) / spacing[1],
            (q[2] - volume.origin[2]) / spacing[2],
        ];
        let occupancy = brick_at(bricks, brick_shape, cell);
        if occupancy <= 0.0 || (view.mode == 1 && occupancy <= peak) {
            i += steps_to_leave(cell, step_dir, spacing, step).max(1);
            continue;
        }
        let dose = sample_mm(volume, q, view.filter);
        match view.mode {
            1 => peak = peak.max(dose),
            2 => {
                let (rgb, alpha) = fog(view, dose, step, typical);
                for channel in 0..3 {
                    col[channel] += trans * rgb[channel] * alpha;
                }
                trans *= 1.0 - alpha;
                if trans < 0.02 {
                    break;
                }
            }
            _ => integ += dose * step,
        }
        i += 1;
    }
    shade_march(view, integ, peak, trans, col, typical)
}

fn shade_march(
    view: RayView,
    integ: f32,
    peak: f32,
    trans: f32,
    col: [f32; 3],
    typical: f32,
) -> (f32, [f32; 4]) {
    let opacity = view.opacity.clamp(0.0, 1.0);
    if view.mode == 2 {
        let covered = 1.0 - trans;
        if covered < 0.02 {
            return (0.0, [0.0; 4]);
        }
        let alpha = (covered * opacity).clamp(0.0, 1.0);
        let rgb = [col[0] / covered, col[1] / covered, col[2] / covered];
        return (covered, [rgb[0], rgb[1], rgb[2], alpha]);
    }
    let shown = if view.mode == 1 { peak } else { integ };
    let (wlo, whi) = color_window(view, typical);
    let scale = wlo.abs().max(whi.abs()).max(1e-6);
    if shown.abs() < scale * DOSE_FLOOR {
        return (shown, [0.0; 4]);
    }
    let span = (whi - wlo).max(1e-6);
    let t = ((shown - wlo) / span).clamp(0.0, 1.0);
    let rgb = crate::sample(view.ramp, t);
    (shown, [rgb[0], rgb[1], rgb[2], opacity])
}

fn color_window(view: RayView, typical: f32) -> (f32, f32) {
    let gain = view.gain.max(0.01);
    if view.mode == 0 {
        if view.lo < 0.0 {
            let reach = view.line_scale.max(1e-6) / gain;
            return (-reach, reach);
        }
        let peak = typical.max(1e-6);
        let flo = (view.lo / peak).clamp(0.0, 1.0);
        let fhi = (view.hi / peak).max(flo + 1e-4);
        let scale = view.line_scale.max(1e-6) / gain;
        return (flo * scale, fhi * scale);
    }
    // Maximum uses the voxel window, tightened by gain the way the Python limits do.
    if view.lo < 0.0 {
        let reach = typical / gain;
        return (-reach, reach);
    }
    (view.lo / gain, (view.hi.max(view.lo + 1e-6)) / gain)
}

fn fog(view: RayView, dose: f32, step: f32, typical: f32) -> ([f32; 3], f32) {
    let gain = view.gain.clamp(0.0, 1.0);
    let reach = typical.max(1e-6);
    let (rgb, alpha) = if view.lo < 0.0 {
        let n = dose / reach;
        let rgb = crate::sample(
            view.ramp,
            ((dose - view.lo) / (view.hi - view.lo).max(1e-6)).clamp(0.0, 1.0),
        );
        let alpha = n.abs().clamp(0.0, 1.0) * gain * step / VIEW_DEPTH_MM;
        (rgb, alpha)
    } else {
        let rgb = crate::sample(
            view.ramp,
            ((dose - view.lo) / (view.hi - view.lo).max(1e-6)).clamp(0.0, 1.0),
        );
        let tau = gain * dose.max(0.0) / reach * step / VIEW_DEPTH_MM;
        (rgb, 1.0 - (-tau).exp())
    };
    (rgb, alpha.clamp(0.0, 1.0))
}

struct Frame {
    eye: [f32; 3],
    look: [f32; 3],
    right: [f32; 3],
    up: [f32; 3],
    tan_y: f32,
    aspect: f32,
    center: [f32; 3],
    half_y: f32,
    ortho: bool,
}

fn camera(volume: &Volume, view: RayView) -> Frame {
    let lattice = lattice_extent(volume);
    let extent = if view.gantry.abs() > 1e-3 {
        gantry_extent(lattice, view.gantry)
    } else {
        lattice
    };
    let lcenter = lattice_center(volume);
    let center = [
        lcenter[0] + view.center[0],
        lcenter[1] + view.center[1],
        lcenter[2] + view.center[2],
    ];
    let aspect = view.aspect.max(1e-3);
    let zoom = view.zoom.max(0.05);
    let fy = film_height(extent, aspect, zoom);
    let half_y = fy * 0.5;
    let (look, right, up) = turntable(view.azimuth, view.elevation);
    // 179° keeps tan(fov/2) finite. 180° is the vispy clamp and blows the fit up.
    let fov = view.fov.clamp(0.0, 179.0_f32.to_radians());
    let ortho = fov <= 1.0e-4;
    // Ortho sits far enough back that the whole box is in front of the eye.
    // The picture size is still `fy`, the same fit vispy uses at fov 0.
    let distance = if ortho {
        extent[0].hypot(extent[1]).hypot(extent[2]) + half_y
    } else {
        fy / (2.0 * (fov * 0.5).tan())
    };
    let tan_y = if ortho { 0.0 } else { (fov * 0.5).tan() };
    Frame {
        eye: [
            center[0] - look[0] * distance,
            center[1] - look[1] * distance,
            center[2] - look[2] * distance,
        ],
        look,
        right,
        up,
        tan_y,
        aspect,
        center,
        half_y,
        ortho,
    }
}

fn cast(volume: &Volume, view: RayView, uv: [f32; 2]) -> Option<([f32; 3], [f32; 3], f32, f32)> {
    let frame = camera(volume, view);
    let x = uv[0] * 2.0 - 1.0;
    let y = (1.0 - uv[1]) * 2.0 - 1.0;
    let shift = x * frame.aspect * frame.half_y;
    let lift = y * frame.half_y;
    let dir = if frame.ortho {
        frame.look
    } else {
        normalize([
            frame.look[0]
                + frame.right[0] * x * frame.aspect * frame.tan_y
                + frame.up[0] * y * frame.tan_y,
            frame.look[1]
                + frame.right[1] * x * frame.aspect * frame.tan_y
                + frame.up[1] * y * frame.tan_y,
            frame.look[2]
                + frame.right[2] * x * frame.aspect * frame.tan_y
                + frame.up[2] * y * frame.tan_y,
        ])
    };
    let eye = if frame.ortho {
        [
            frame.eye[0] + frame.right[0] * shift + frame.up[0] * lift,
            frame.eye[1] + frame.right[1] * shift + frame.up[1] * lift,
            frame.eye[2] + frame.right[2] * shift + frame.up[2] * lift,
        ]
    } else {
        frame.eye
    };
    let (bmin, bmax) = view_box(volume, view.gantry);
    let (t0, t1) = ray_box(eye, dir, bmin, bmax)?;
    Some((eye, dir, t0, t1))
}

/// Film uv of a lattice point. `None` when it sits behind the camera or off the frame.
pub fn dose_film_uv(volume: &Volume, view: RayView, point: [f32; 3]) -> Option<[f32; 2]> {
    film_uv(volume, view, point)
}

/// Film uv of a world point, including points outside the frame.
/// `None` only when the point sits behind the camera.
pub fn dose_film_open(volume: &Volume, view: RayView, point: [f32; 3]) -> Option<[f32; 2]> {
    let point = if view.gantry.abs() > 1e-3 {
        to_view(point, lattice_center(volume), view.gantry)
    } else {
        point
    };
    let frame = camera(volume, view);
    let (x, y) = if frame.ortho {
        let rel = [
            point[0] - frame.center[0],
            point[1] - frame.center[1],
            point[2] - frame.center[2],
        ];
        (
            dot(rel, frame.right) / (frame.aspect * frame.half_y.max(1e-6)),
            dot(rel, frame.up) / frame.half_y.max(1e-6),
        )
    } else {
        let rel = [
            point[0] - frame.eye[0],
            point[1] - frame.eye[1],
            point[2] - frame.eye[2],
        ];
        let depth = dot(rel, frame.look);
        if depth < 1e-3 {
            return None;
        }
        (
            dot(rel, frame.right) / (depth * frame.aspect * frame.tan_y),
            dot(rel, frame.up) / (depth * frame.tan_y),
        )
    };
    Some([(x + 1.0) * 0.5, (1.0 - y) * 0.5])
}

/// Film uv of a world point. `None` when it sits behind the camera or off the frame.
fn film_uv(volume: &Volume, view: RayView, point: [f32; 3]) -> Option<[f32; 2]> {
    let uv = dose_film_open(volume, view, point)?;
    if !(0.0..=1.0).contains(&uv[0]) || !(0.0..=1.0).contains(&uv[1]) {
        return None;
    }
    Some(uv)
}

/// Vispy turntable, +z up. Azimuth 0 and elevation 0 look along +y.
fn lattice_extent(volume: &Volume) -> [f32; 3] {
    volume.extent()
}

fn lattice_center(volume: &Volume) -> [f32; 3] {
    let extent = lattice_extent(volume);
    [
        volume.origin[0] + extent[0] * 0.5,
        volume.origin[1] + extent[1] * 0.5,
        volume.origin[2] + extent[2] * 0.5,
    ]
}

/// Axis-aligned size of `extent` after a rotation about +X through its center.
/// 90° swaps Y and Z: depth, which is lattice Z, lies along world Y.
pub fn gantry_extent(extent: [f32; 3], degrees: f32) -> [f32; 3] {
    let (s, c) = degrees.to_radians().sin_cos();
    let (c, s) = (c.abs(), s.abs());
    [
        extent[0],
        c * extent[1] + s * extent[2],
        s * extent[1] + c * extent[2],
    ]
}

fn view_box(volume: &Volume, degrees: f32) -> ([f32; 3], [f32; 3]) {
    if degrees.abs() <= 1e-3 {
        let extent = lattice_extent(volume);
        return (
            volume.origin,
            [
                volume.origin[0] + extent[0],
                volume.origin[1] + extent[1],
                volume.origin[2] + extent[2],
            ],
        );
    }
    let extent = gantry_extent(lattice_extent(volume), degrees);
    let center = lattice_center(volume);
    (
        [
            center[0] - extent[0] * 0.5,
            center[1] - extent[1] * 0.5,
            center[2] - extent[2] * 0.5,
        ],
        [
            center[0] + extent[0] * 0.5,
            center[1] + extent[1] * 0.5,
            center[2] + extent[2] * 0.5,
        ],
    )
}

/// `R` about +X. 90° maps `(x, y, z)` to `(x, −z, y)` relative to the center.
fn to_view(point: [f32; 3], center: [f32; 3], degrees: f32) -> [f32; 3] {
    let (s, c) = degrees.to_radians().sin_cos();
    let d = [
        point[0] - center[0],
        point[1] - center[1],
        point[2] - center[2],
    ];
    [
        center[0] + d[0],
        center[1] + c * d[1] - s * d[2],
        center[2] + s * d[1] + c * d[2],
    ]
}

fn lattice_point(point: [f32; 3], center: [f32; 3], degrees: f32) -> [f32; 3] {
    let (s, c) = degrees.to_radians().sin_cos();
    let d = [
        point[0] - center[0],
        point[1] - center[1],
        point[2] - center[2],
    ];
    [
        center[0] + d[0],
        center[1] + c * d[1] + s * d[2],
        center[2] - s * d[1] + c * d[2],
    ]
}

fn lattice_dir(dir: [f32; 3], degrees: f32) -> [f32; 3] {
    let (s, c) = degrees.to_radians().sin_cos();
    [dir[0], c * dir[1] + s * dir[2], -s * dir[1] + c * dir[2]]
}

fn turntable(azimuth: f32, elevation: f32) -> ([f32; 3], [f32; 3], [f32; 3]) {
    let (sa, ca) = azimuth.sin_cos();
    let (se, ce) = elevation.sin_cos();
    let look = [-ce * sa, ce * ca, -se];
    let right = [ca, sa, 0.0];
    let up = cross(right, look);
    (normalize(look), normalize(right), normalize(up))
}

fn ray_box(origin: [f32; 3], dir: [f32; 3], bmin: [f32; 3], bmax: [f32; 3]) -> Option<(f32, f32)> {
    let mut t0 = 0.0f32;
    let mut t1 = f32::MAX;
    for axis in 0..3 {
        if dir[axis].abs() < 1e-8 {
            if origin[axis] < bmin[axis] || origin[axis] > bmax[axis] {
                return None;
            }
            continue;
        }
        let inv = 1.0 / dir[axis];
        let mut near = (bmin[axis] - origin[axis]) * inv;
        let mut far = (bmax[axis] - origin[axis]) * inv;
        if near > far {
            std::mem::swap(&mut near, &mut far);
        }
        t0 = t0.max(near);
        t1 = t1.min(far);
        if t0 > t1 {
            return None;
        }
    }
    if t1 < 0.0 {
        return None;
    }
    Some((t0.max(0.0), t1))
}

fn brick_at(bricks: &[f32], shape: [usize; 3], cell: [f32; 3]) -> f32 {
    if shape[0] == 0 || shape[1] == 0 || shape[2] == 0 {
        return 1.0;
    }
    let x = (cell[0] / BRICK as f32).floor() as i32;
    let y = (cell[1] / BRICK as f32).floor() as i32;
    let z = (cell[2] / BRICK as f32).floor() as i32;
    if x < 0
        || y < 0
        || z < 0
        || x >= shape[0] as i32
        || y >= shape[1] as i32
        || z >= shape[2] as i32
    {
        return 0.0;
    }
    bricks
        .get(x as usize + shape[0] * (y as usize + shape[1] * z as usize))
        .copied()
        .unwrap_or(0.0)
}

fn steps_to_leave(cell: [f32; 3], dir: [f32; 3], spacing: [f32; 3], step: f32) -> usize {
    let mut delta = f32::MAX;
    for axis in 0..3 {
        let dir_i = dir[axis] / spacing[axis];
        if dir_i.abs() < 1e-8 {
            continue;
        }
        let brick = (cell[axis] / BRICK as f32).floor() * BRICK as f32;
        let bound = if dir_i > 0.0 {
            brick + BRICK as f32
        } else {
            brick
        };
        let along = (bound - cell[axis]) / dir_i;
        if along > 1e-4 {
            delta = delta.min(along);
        }
    }
    let steps = (delta / step).ceil();
    // delta can stay f32::MAX when every axis is within tolerance of a brick
    // bound — finite, but saturates `as usize` to usize::MAX and wraps the
    // caller's `i +=`. Leaving a brick never needs more than the march cap.
    if !steps.is_finite() {
        return 1;
    }
    (steps as usize).clamp(1, MAX_STEPS)
}

fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale(v: [f32; 3], s: f32) -> [f32; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub fn sample_mm(volume: &Volume, mm: [f32; 3], filter: u8) -> f32 {
    sample_index(
        volume,
        (mm[0] - volume.origin[0]) / volume.axis(0) - 0.5,
        (mm[1] - volume.origin[1]) / volume.axis(1) - 0.5,
        (mm[2] - volume.origin[2]) / volume.axis(2) - 0.5,
        filter,
    )
}

pub fn sample_index(volume: &Volume, x: f32, y: f32, z: f32, filter: u8) -> f32 {
    match filter {
        2 => cubic(volume, x, y, z),
        1 => linear(volume, x, y, z),
        _ => {
            // Python nearest is floor(cell), and a miss outside the grid is 0.
            // `x` is cell - 0.5, the same coordinate linear interpolation uses.
            let [nx, ny, nz] = volume.shape;
            let ix = (x + 0.5).floor() as i32;
            let iy = (y + 0.5).floor() as i32;
            let iz = (z + 0.5).floor() as i32;
            if ix < 0 || iy < 0 || iz < 0 || ix >= nx as i32 || iy >= ny as i32 || iz >= nz as i32 {
                return 0.0;
            }
            at(volume, ix as usize, iy as usize, iz as usize)
        }
    }
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-6);
    [v[0] / len, v[1] / len, v[2] / len]
}

fn at(volume: &Volume, x: usize, y: usize, z: usize) -> f32 {
    let [nx, ny, nz] = volume.shape;
    if x >= nx || y >= ny || z >= nz {
        return 0.0;
    }
    volume.get(x, y, z)
}

fn linear(volume: &Volume, x: f32, y: f32, z: f32) -> f32 {
    let x0 = x.floor();
    let y0 = y.floor();
    let z0 = z.floor();
    let tx = x - x0;
    let ty = y - y0;
    let tz = z - z0;
    let mut acc = 0.0;
    for dz in 0..2 {
        for dy in 0..2 {
            for dx in 0..2 {
                let w = (if dx == 0 { 1.0 - tx } else { tx })
                    * (if dy == 0 { 1.0 - ty } else { ty })
                    * (if dz == 0 { 1.0 - tz } else { tz });
                acc += w * at(
                    volume,
                    (x0 as i32 + dx) as usize,
                    (y0 as i32 + dy) as usize,
                    (z0 as i32 + dz) as usize,
                );
            }
        }
    }
    acc
}

fn cubic(volume: &Volume, x: f32, y: f32, z: f32) -> f32 {
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let z0 = z.floor() as i32;
    let mut acc = 0.0;
    for dz in -1..3 {
        for dy in -1..3 {
            for dx in -1..3 {
                let w = catmull(x - (x0 + dx) as f32)
                    * catmull(y - (y0 + dy) as f32)
                    * catmull(z - (z0 + dz) as f32);
                acc += w * at(
                    volume,
                    (x0 + dx) as usize,
                    (y0 + dy) as usize,
                    (z0 + dz) as usize,
                );
            }
        }
    }
    acc
}

fn catmull(d: f32) -> f32 {
    let x = d.abs();
    if x >= 2.0 {
        0.0
    } else if x >= 1.0 {
        ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0
    } else {
        (1.5 * x - 2.5) * x * x + 1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn peaked() -> Volume {
        let mut values = vec![0.0; 27];
        values[1 + 3 * 4] = 4.0;
        Volume {
            origin: [0.0, 0.0, 0.0],
            shape: [3, 3, 3],
            spacing: [1.0; 3],
            values,
        }
    }

    fn view_of(volume: &Volume, mode: u8) -> RayView {
        let peak = volume
            .values
            .iter()
            .copied()
            .fold(0.0f32, f32::max)
            .max(1e-6);
        RayView {
            azimuth: 30.0_f32.to_radians(),
            elevation: 25.0_f32.to_radians(),
            zoom: 1.0,
            aspect: 1.0,
            mode,
            filter: 1,
            gain: 1.0,
            opacity: 1.0,
            lo: 0.0,
            hi: peak,
            ramp: 1,
            line_scale: line_scale(volume),
            fov: FOV_Y,
            center: [0.0; 3],
            gantry: 0.0,
        }
    }

    #[test]
    fn integrate_maximum_and_transparent_disagree() {
        let volume = peaked();
        let integrate = ray_value(&volume, 0.4, 0.3, 0, 0, 0.5, 0.5);
        let maximum = ray_value(&volume, 0.4, 0.3, 1, 0, 0.5, 0.5);
        let transparent = ray_value(&volume, 0.4, 0.3, 2, 1, 0.5, 0.5);
        assert!(integrate > 0.0);
        assert!(maximum <= 4.0 + 1e-3);
        assert!(transparent > 0.0);
        assert!(integrate != maximum || transparent != maximum);
    }

    #[test]
    fn the_center_ray_sees_the_hot_voxel_and_the_corner_misses() {
        let volume = peaked();
        let (bricks, shape) = brick_grid(&volume);
        let hit = ray_rgba(&volume, &bricks, shape, view_of(&volume, 1), [0.5, 0.5]);
        let miss = ray_rgba(&volume, &bricks, shape, view_of(&volume, 1), [0.0, 0.0]);
        assert!(hit[3] > 0.5, "{hit:?}");
        assert_eq!(miss[3], 0.0);
    }

    #[test]
    fn empty_bricks_do_not_hide_a_voxel_behind_them() {
        let mut values = vec![0.0; 64 * 64 * 64];
        for z in 30..34 {
            for y in 30..34 {
                for x in 30..34 {
                    values[x + 64 * (y + 64 * z)] = 7.0;
                }
            }
        }
        let volume = Volume {
            origin: [0.0, 0.0, 0.0],
            shape: [64, 64, 64],
            spacing: [1.0; 3],
            values,
        };
        let (bricks, shape) = brick_grid(&volume);
        assert!(bricks.contains(&0.0));
        let hit = ray_rgba(&volume, &bricks, shape, view_of(&volume, 1), [0.5, 0.5]);
        assert!(hit[3] > 0.5, "{hit:?}");
        let shown = ray_value(
            &volume,
            30.0_f32.to_radians(),
            25.0_f32.to_radians(),
            1,
            1,
            0.5,
            0.5,
        );
        assert!(shown > 5.0, "{shown}");
    }

    #[test]
    fn a_uniform_cube_integrates_about_a_chord() {
        let volume = Volume {
            origin: [0.0, 0.0, 0.0],
            shape: [8, 8, 8],
            spacing: [1.0; 3],
            values: vec![1.0; 512],
        };
        let shown = ray_value(
            &volume,
            30.0_f32.to_radians(),
            25.0_f32.to_radians(),
            0,
            0,
            0.5,
            0.5,
        );
        assert!(shown > 8.0 && shown < 8.0 * 1.75, "{shown}");
    }

    #[test]
    fn maximum_of_a_uniform_field_is_the_voxel() {
        let volume = Volume {
            origin: [0.0, 0.0, 0.0],
            shape: [4, 4, 4],
            spacing: [1.0; 3],
            values: vec![3.0; 64],
        };
        let shown = ray_value(
            &volume,
            30.0_f32.to_radians(),
            25.0_f32.to_radians(),
            1,
            0,
            0.5,
            0.5,
        );
        assert!((shown - 3.0).abs() < 1e-3, "{shown}");
    }

    #[test]
    fn one_scan_matches_the_plane_integrals() {
        let mut values = Vec::new();
        for z in 0..2 {
            for y in 0..3 {
                for x in 0..4 {
                    values.push((x + 10 * y + 100 * z) as f32);
                }
            }
        }
        let volume = Volume {
            origin: [0.0, 0.0, 0.0],
            shape: [4, 3, 2],
            spacing: [2.0; 3],
            values,
        };
        let scan = scan_volume(&volume);
        for plane in 0..3 {
            let (image, _, _) = volume.integrated_slice(plane);
            assert_eq!(scan.integrals[plane], image);
        }
        assert_eq!(scan.peak_at, [3, 2, 1]);
        assert!(scan.line_scale > 100.0);
        let (_, dose) = volume.depth_integral(1);
        let ny = volume.shape[1];
        for (index, z) in (0..volume.shape[2]).rev().enumerate() {
            let cached = scan.integrals[2][1 + ny * z] * volume.spacing[1];
            assert!((dose[index] - cached).abs() < 1e-3);
        }
        let z = 1usize;
        let (_, lateral) = volume.lateral_integral(0, z);
        for (x, value) in lateral.iter().enumerate() {
            let cached = scan.integrals[1][x + volume.shape[0] * z] * volume.spacing[2];
            assert!((value - cached).abs() < 1e-3);
        }
    }

    #[test]
    fn linear_sample_sits_between_neighbors() {
        let volume = Volume {
            origin: [0.0, 0.0, 0.0],
            shape: [2, 1, 1],
            spacing: [1.0; 3],
            values: vec![0.0, 4.0],
        };
        let mid = sample_index(&volume, 0.5, 0.0, 0.0, 1);
        assert!((mid - 2.0).abs() < 1e-3, "{mid}");
    }

    #[test]
    fn nearest_uses_the_voxel_that_contains_the_sample() {
        let volume = Volume {
            origin: [0.0, 0.0, 0.0],
            shape: [2, 1, 1],
            spacing: [1.0; 3],
            values: vec![1.0, 4.0],
        };
        // cell 0.2 is inside voxel 0. cell -0.1 is outside.
        assert!((sample_mm(&volume, [0.2, 0.5, 0.5], 0) - 1.0).abs() < 1e-3);
        assert!(sample_mm(&volume, [-0.1, 0.5, 0.5], 0).abs() < 1e-3);
        assert!((sample_mm(&volume, [1.2, 0.5, 0.5], 0) - 4.0).abs() < 1e-3);
    }

    #[test]
    fn ortho_rays_stay_parallel_through_the_box() {
        let n = 8usize;
        let volume = Volume {
            origin: [0.0, 0.0, 0.0],
            shape: [n, n, n],
            spacing: [1.0; 3],
            values: vec![1.0; n * n * n],
        };
        let scan = scan_volume(&volume);
        let mut view = view_of(&volume, 0);
        view.azimuth = 0.0;
        view.elevation = 0.0;
        view.fov = 0.0;
        let center = march(&volume, &scan.bricks, scan.brick_shape, view, [0.5, 0.5]).0;
        let off = march(&volume, &scan.bricks, scan.brick_shape, view, [0.62, 0.5]).0;
        assert!(center > 6.0, "{center}");
        assert!((center - off).abs() < 0.5, "center {center} off {off}");
    }

    #[test]
    fn integrate_window_follows_the_view_ray() {
        let n = 24usize;
        let mut values = vec![0.0f32; n * n * n];
        for z in 0..n {
            let depth = z as f32 / n as f32;
            let dose = 0.35 + 0.65 * (-((depth - 0.8) / 0.08).powi(2)).exp();
            for y in 0..n {
                for x in 0..n {
                    let dx = x as f32 / n as f32 - 0.5;
                    let dy = y as f32 / n as f32 - 0.5;
                    let lateral = (-(dx * dx + dy * dy) / 0.02).exp();
                    values[x + n * (y + n * z)] = dose * lateral;
                }
            }
        }
        let volume = Volume {
            origin: [0.0, 0.0, 0.0],
            shape: [n, n, n],
            spacing: [1.0; 3],
            values,
        };
        let scan = scan_volume(&volume);
        let azimuth = 30.0_f32.to_radians();
        let elevation = 25.0_f32.to_radians();
        let scale = view_ray_scale(
            &volume,
            &scan.bricks,
            scan.brick_shape,
            azimuth,
            elevation,
            1.0,
            1.0,
            1,
            FOV_Y,
            [0.0; 3],
            0.0,
            scan.peak_at,
        );
        let (px, py, pz) = volume.peak_index();
        assert_eq!(scan.peak_at, [px, py, pz]);
        let corner = view_ray_scale(
            &volume,
            &scan.bricks,
            scan.brick_shape,
            azimuth,
            elevation,
            1.0,
            1.0,
            1,
            FOV_Y,
            [0.0; 3],
            0.0,
            [0, 0, 0],
        );
        assert!(scale > corner, "peak ray {scale} corner ray {corner}");
        // Home looks mostly across the beam, so the view ray is shorter than the depth sum.
        assert!(
            scale < scan.line_scale * 0.9,
            "view {scale} axis {}",
            scan.line_scale
        );
        let mut view = view_of(&volume, 0);
        view.line_scale = scale;
        let (ix, iy, iz) = volume.peak_index();
        let point = [(ix as f32 + 0.5), (iy as f32 + 0.5), (iz as f32 + 0.5)];
        let uv = film_uv(&volume, view, point).expect("peak on screen");
        let rgba = ray_rgba(&volume, &scan.bricks, scan.brick_shape, view, uv);
        let top = crate::sample(1, 1.0);
        for channel in 0..3 {
            assert!(
                (rgba[channel] - top[channel]).abs() < 0.08,
                "peak {:?} top {top:?}",
                rgba
            );
        }
        let entrance = film_uv(&volume, view, [n as f32 * 0.5, n as f32 * 0.5, 2.5]).unwrap();
        let peak_ray = march(&volume, &scan.bricks, scan.brick_shape, view, uv).0;
        let entrance_ray = march(&volume, &scan.bricks, scan.brick_shape, view, entrance).0;
        assert!(
            peak_ray > entrance_ray * 1.4,
            "peak ray {peak_ray} entrance {entrance_ray}"
        );
    }

    #[test]
    #[cfg(not(debug_assertions))]
    fn dense_view_scale_stays_under_a_frame() {
        let n = 64usize;
        let mut values = vec![0.0f32; n * n * n];
        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    let dx = x as f32 - n as f32 * 0.5;
                    let dy = y as f32 - n as f32 * 0.5;
                    values[x + n * (y + n * z)] = (-(dx * dx + dy * dy) / 80.0).exp();
                }
            }
        }
        let volume = Volume {
            origin: [0.0, 0.0, 0.0],
            shape: [n, n, n],
            spacing: [1.0; 3],
            values,
        };
        let scan = scan_volume(&volume);
        let start = std::time::Instant::now();
        let scale = view_ray_scale(
            &volume,
            &scan.bricks,
            scan.brick_shape,
            0.5,
            0.4,
            1.0,
            1.2,
            1,
            FOV_Y,
            [0.0; 3],
            0.0,
            scan.peak_at,
        );
        let ms = start.elapsed().as_secs_f32() * 1000.0;
        assert!(scale > 1.0, "{scale}");
        assert!(ms < 8.0, "{ms} ms");
    }

    #[test]
    fn gantry_ninety_lays_depth_along_y() {
        let center = [0.0, 0.0, 0.0];
        let laid = to_view([1.0, 2.0, 3.0], center, 90.0);
        assert!((laid[0] - 1.0).abs() < 1e-5);
        assert!((laid[1] + 3.0).abs() < 1e-5, "{laid:?}");
        assert!((laid[2] - 2.0).abs() < 1e-5, "{laid:?}");
        let back = lattice_point(laid, center, 90.0);
        assert!((back[0] - 1.0).abs() < 1e-4);
        assert!((back[1] - 2.0).abs() < 1e-4);
        assert!((back[2] - 3.0).abs() < 1e-4);
        let volume = Volume {
            origin: [0.0, 0.0, -10.0],
            shape: [2, 2, 10],
            spacing: [1.0; 3],
            values: vec![0.0; 40],
        };
        let (bmin, bmax) = view_box(&volume, 90.0);
        let extent = [bmax[0] - bmin[0], bmax[1] - bmin[1], bmax[2] - bmin[2]];
        assert!((extent[1] - 10.0).abs() < 1e-4, "{extent:?}");
        assert!((extent[2] - 2.0).abs() < 1e-4, "{extent:?}");
    }

    #[test]
    fn a_thick_slice_keeps_its_depth() {
        let mut values = vec![0.0; 8];
        values[4] = 1.0;
        let volume = Volume {
            origin: [0.0, 10.0, 20.0],
            shape: [2, 2, 2],
            spacing: [1.0, 1.0, 2.5],
            values,
        };
        assert!((volume.mm_of(0, 1) - 1.5).abs() < 1e-4);
        assert!((volume.mm_of(2, 1) - 23.75).abs() < 1e-4);
        assert_eq!(volume.index_of(2, 23.75), 1);
        let (depth, dose) = volume.depth_profile(0, 0);
        assert!((depth[0] - 1.25).abs() < 1e-4, "{depth:?}");
        assert!((dose[0] - 1.0).abs() < 1e-6);
        let (axial, _, _) = volume.integrated_slice(0);
        assert!((axial[0] - 2.5).abs() < 1e-4, "{axial:?}");
        let at = sample_mm(&volume, [0.5, 10.5, 23.75], 0);
        assert!((at - 1.0).abs() < 1e-4, "{at}");
        assert!((volume.extent()[2] - 5.0).abs() < 1e-4);
    }

    #[test]
    fn a_large_lattice_matches_one_thread() {
        let (nx, ny, nz) = (48usize, 40usize, 160usize);
        let mut values = vec![0.0f32; nx * ny * nz];
        for z in 0..nz {
            values[7 + nx * (9 + ny * z)] = 0.1 + 0.01 * (z as f32);
            if z % 8 == 3 {
                values[20 + nx * (15 + ny * z)] = -0.4;
            }
        }
        values[4 + nx * (6 + ny * 2)] = 1.25;
        values[30 + nx * (22 + ny * 150)] = 6.5;
        let volume = Volume {
            origin: [0.0; 3],
            shape: [nx, ny, nz],
            spacing: [1.0, 1.5, 2.0],
            values,
        };
        let serial = scan_axes_serial(&volume.values, volume.shape, volume.spacing);
        let scan = scan_volume(&volume);
        assert!(scan.covers(volume.shape));
        assert_eq!(scan.peak_at, serial.peak_at);
        assert_eq!(scan.peak.to_bits(), serial.peak.to_bits());
        assert_eq!(scan.abs_peak.to_bits(), serial.abs_peak.to_bits());
        assert_eq!(scan.bricks, serial.bricks);
        assert_eq!(scan.brick_shape, serial.brick_shape);
        for plane in 0..3 {
            for (got, want) in scan.integrals[plane].iter().zip(&serial.integrals[plane]) {
                let tol = 1.0e-4 * want.abs().max(1.0e-4);
                assert!((got - want).abs() <= tol, "plane {plane} {got} vs {want}");
            }
        }
        let scale = serial.line_scale.max(1.0e-4);
        assert!((scan.line_scale - serial.line_scale).abs() <= 1.0e-4 * scale);
    }
    #[test]
    fn steps_to_leave_stays_bounded_at_a_brick_edge() {
        // Every axis within tolerance of a bound leaves delta = f32::MAX —
        // finite but usize::MAX after the cast. The march cap bounds it.
        let steps = steps_to_leave([8.0 - 6e-5, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0; 3], 1.0);
        assert!(steps <= MAX_STEPS, "{steps}");
        assert!(steps >= 1);
    }
}
