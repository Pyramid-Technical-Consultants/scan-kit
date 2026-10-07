//! Dose workspace sampling. Slices read an atlas uploaded once. The 3D cell
//! marches the float volume, so paging and orbit do not build a new grid.
//!
//! Atlas: z slices in a square of tiles, row 0 of a slice at the bottom of its
//! tile. The last texture row is the color scale, 256 entries. Voxel R is
//! `0.5 + 0.5 * dose/span` (zero at mid-grey). G is the CT window, B is 1 when
//! a CT sample is present.

use scan_kit_core::{index, robust_high, sample, scan_volume, Volume};
#[cfg(not(target_arch = "wasm32"))]
use scan_kit_core::{ray_rgba, RayView};

/// Display state copied from the scene. The plot owns it after the upload.
#[derive(Clone, Debug)]
pub struct DoseGrid {
    pub volume: Volume,
    pub ct: Vec<f32>,
    pub labels: Vec<u8>,
    pub peak: f32,
    /// Max |dose|. Slice texels store `0.5 + 0.5 * dose/span`, so a difference keeps its sign.
    pub span: f32,
    pub ramp: u8,
    pub lo: f32,
    pub hi: f32,
    /// Data window. Gain and a manual window scale this; they do not replace it.
    pub base_lo: f32,
    pub base_hi: f32,
    pub gamma: bool,
    pub difference: bool,
    pub gain: f32,
    pub opacity: f32,
    /// 0 integrate, 1 maximum, 2 transparent.
    pub mode: u8,
    /// 0 nearest, 1 linear, 2 cubic.
    pub filter: u8,
    pub atlas_shape: [usize; 3],
    /// Dilated max-|dose| bricks. A zero is safe to leap in the 3D march.
    pub bricks: Vec<f32>,
    pub brick_shape: [usize; 3],
    /// Brightest axis-aligned line integral, dose × mm.
    pub line_scale: f32,
    pub peak_at: [usize; 3],
    /// Axial, coronal, and sagittal integrals in dose × mm.
    pub integrals: [Vec<f32>; 3],
    pub integral_peaks: [f32; 3],
    /// Degrees about +X. Zero keeps the lattice box.
    pub gantry: f32,
    pub unit: String,
    pub show_phantom: bool,
    pub field: [f32; 6],
    /// Selection order. `None` is `volume`.
    pub sessions: Vec<Option<Volume>>,
    pub session_colors: Vec<[f32; 4]>,
    pub session_focus: usize,
}

#[cfg(not(target_arch = "wasm32"))]
pub struct ViewSample {
    /// 0 axial, 1 coronal, 2 sagittal, 3 volume.
    pub plane: u8,
    pub index: usize,
    pub turns: u8,
    pub integral: bool,
    pub azimuth: f32,
    pub elevation: f32,
    pub zoom: f32,
    pub aspect: f32,
    /// Brightest integrate ray. Zero keeps the axis line scale.
    pub ray_scale: f32,
    pub fov: f32,
    pub center: [f32; 3],
}

pub fn grid_from(
    values: Vec<f32>,
    ct: Vec<f32>,
    labels: Vec<u8>,
    shape: [u32; 3],
    origin: [f32; 3],
    voxel: f32,
    ramp: u8,
    lo: f32,
    hi: f32,
    gain: f32,
    opacity: f32,
    mode: u8,
    filter: u8,
) -> Option<DoseGrid> {
    let shape_us = [shape[0] as usize, shape[1] as usize, shape[2] as usize];
    let n = shape_us[0] * shape_us[1] * shape_us[2];
    if n == 0 || values.len() < n {
        return None;
    }
    let mut values = values;
    values.truncate(n);
    let volume = Volume {
        origin,
        shape: shape_us,
        voxel: voxel.max(1e-3),
        values,
    };
    let scan = scan_volume(&volume);
    let span = volume
        .values
        .iter()
        .copied()
        .fold(0.0f32, |best, value| best.max(value.abs()))
        .max(1e-6);
    Some(DoseGrid {
        atlas_shape: fit_shape(shape_us),
        volume,
        bricks: scan.bricks,
        brick_shape: scan.brick_shape,
        line_scale: scan.line_scale,
        peak_at: scan.peak_at,
        integrals: scan.integrals,
        integral_peaks: scan.integral_peaks,
        ct: if ct.len() >= n {
            ct[..n].to_vec()
        } else {
            Vec::new()
        },
        labels: if labels.len() >= n {
            labels[..n].to_vec()
        } else {
            Vec::new()
        },
        peak: scan.peak.max(1e-6),
        span,
        ramp,
        lo,
        hi,
        base_lo: lo,
        base_hi: hi,
        gamma: false,
        difference: false,
        gain,
        opacity,
        mode,
        filter,
        gantry: 0.0,
        unit: String::new(),
        show_phantom: false,
        field: [0.0; 6],
        sessions: Vec::new(),
        session_colors: Vec::new(),
        session_focus: 0,
    })
}

pub fn fit_shape(shape: [usize; 3]) -> [usize; 3] {
    let mut fitted = shape;
    loop {
        let tiles = tile_count(fitted[2]);
        let tiles_y = fitted[2].div_ceil(tiles.max(1));
        let wide =
            tiles.saturating_mul(fitted[0]) <= 2048 && tiles_y.saturating_mul(fitted[1]) <= 2048;
        if wide || fitted.iter().all(|axis| *axis <= 8) {
            break;
        }
        let axis = (0..3).max_by_key(|axis| fitted[*axis]).unwrap_or(0);
        fitted[axis] = (fitted[axis] / 2).max(1);
    }
    fitted
}

pub fn tile_count(nz: usize) -> usize {
    (nz.max(1) as f32).sqrt().ceil() as usize
}

/// RGBA atlas plus its pixel size. Building this is the volume upload.
pub fn atlas_bytes(grid: &DoseGrid) -> (Vec<u8>, u32, u32) {
    let [nx, ny, nz] = grid.atlas_shape;
    let tiles = tile_count(nz);
    let tiles_y = nz.div_ceil(tiles.max(1));
    let width = tiles.saturating_mul(nx).max(256);
    let height = tiles_y.saturating_mul(ny) + 1;
    let mut pixels = vec![0u8; width * height * 4];
    let [fx, fy, fz] = grid.volume.shape;
    if fx == nx && fy == ny && fz == nz && grid.ct.is_empty() {
        let values = &grid.volume.values;
        let span = grid.span;
        for z in 0..nz {
            let tx = z % tiles;
            let ty = z / tiles;
            let src_z = z * nx * ny;
            for y in 0..ny {
                let dst_row = (ty * ny + (ny - 1 - y)) * width + tx * nx;
                let src_row = src_z + y * nx;
                for x in 0..nx {
                    let dst = (dst_row + x) * 4;
                    pixels[dst] = stored_dose(values[src_row + x], span);
                    pixels[dst + 3] = 255;
                }
            }
        }
    } else {
        fill_atlas(&mut pixels, grid, nx, ny, nz, tiles, width);
    }
    let ramp_row = height - 1;
    for x in 0..256 {
        let rgb = sample(grid.ramp, x as f32 / 255.0);
        let dst = (ramp_row * width + x) * 4;
        pixels[dst] = byte(rgb[0]);
        pixels[dst + 1] = byte(rgb[1]);
        pixels[dst + 2] = byte(rgb[2]);
        pixels[dst + 3] = byte(grid.opacity);
    }
    (pixels, width as u32, height as u32)
}

/// Keep the measured window. A shipped `base_hi` skips a second pass over the cube.
pub fn remember_window(grid: &mut DoseGrid, base_lo: f32, base_hi: f32) {
    let gamma = grid.unit == "γ" || grid.ramp == index("gamma");
    if gamma {
        grid.gamma = true;
        grid.difference = false;
        grid.base_lo = 0.0;
        grid.base_hi = 2.0;
        return;
    }
    grid.gamma = false;
    if base_hi > base_lo || base_lo < 0.0 {
        grid.difference = base_lo < 0.0;
        grid.base_lo = base_lo;
        grid.base_hi = base_hi;
        return;
    }
    let difference = grid.lo < 0.0 || grid.volume.values.iter().any(|value| *value < 0.0);
    grid.difference = difference;
    if difference {
        let flat: Vec<f32> = grid
            .volume
            .values
            .iter()
            .copied()
            .filter(|value| *value != 0.0)
            .map(f32::abs)
            .collect();
        let hi = robust_high(&flat);
        grid.base_lo = -hi;
        grid.base_hi = hi;
    } else {
        grid.base_lo = 0.0;
        grid.base_hi = robust_high(&grid.volume.values);
    }
}

/// Rewrite the 256 color-scale pixels on the last atlas row.
pub fn paint_ramp_row(pixels: &mut [u8], cols: u32, rows: u32, ramp: u8, opacity: f32) {
    let alpha = byte(opacity);
    for x in 0..256 {
        let Some(pixel) = ramp_pixel(pixels, cols, rows, x) else {
            return;
        };
        let rgb = sample(ramp, x as f32 / 255.0);
        pixel[0] = byte(rgb[0]);
        pixel[1] = byte(rgb[1]);
        pixel[2] = byte(rgb[2]);
        pixel[3] = alpha;
    }
}

/// Opacity lives in the ramp-row alpha because the color bar reads that texel.
pub fn paint_ramp_alpha(pixels: &mut [u8], cols: u32, rows: u32, opacity: f32) {
    let alpha = byte(opacity);
    for x in 0..256 {
        let Some(pixel) = ramp_pixel(pixels, cols, rows, x) else {
            return;
        };
        pixel[3] = alpha;
    }
}

fn ramp_pixel(pixels: &mut [u8], cols: u32, rows: u32, x: usize) -> Option<&mut [u8]> {
    if cols < 256 || rows == 0 || x >= 256 {
        return None;
    }
    let dst = ((rows as usize - 1) * cols as usize + x) * 4;
    pixels.get_mut(dst..dst + 4)
}

fn fill_atlas(
    pixels: &mut [u8],
    grid: &DoseGrid,
    nx: usize,
    ny: usize,
    nz: usize,
    tiles: usize,
    width: usize,
) {
    let [fx, fy, fz] = grid.volume.shape;
    let values = &grid.volume.values;
    let span = grid.span;
    let x_of: Vec<usize> = (0..nx)
        .map(|x| {
            if fx == nx {
                x
            } else {
                (x * fx / nx).min(fx - 1)
            }
        })
        .collect();
    let y_of: Vec<usize> = (0..ny)
        .map(|y| {
            if fy == ny {
                y
            } else {
                (y * fy / ny).min(fy - 1)
            }
        })
        .collect();
    let z_of: Vec<usize> = (0..nz)
        .map(|z| {
            if fz == nz {
                z
            } else {
                (z * fz / nz).min(fz - 1)
            }
        })
        .collect();
    if grid.ct.is_empty() {
        for z in 0..nz {
            let tx = z % tiles;
            let ty = z / tiles;
            let src_z = fx * fy * z_of[z];
            for y in 0..ny {
                let src_row = src_z + fx * y_of[y];
                let dst_row = (ty * ny + (ny - 1 - y)) * width + tx * nx;
                for x in 0..nx {
                    let dst = (dst_row + x) * 4;
                    pixels[dst] = stored_dose(values[src_row + x_of[x]], span);
                    pixels[dst + 3] = 255;
                }
            }
        }
        return;
    }
    for z in 0..nz {
        let tx = z % tiles;
        let ty = z / tiles;
        let src_z = fx * fy * z_of[z];
        for y in 0..ny {
            let src_row = src_z + fx * y_of[y];
            let dst_row = (ty * ny + (ny - 1 - y)) * width + tx * nx;
            for x in 0..nx {
                let sx = x_of[x];
                let dose = values[src_row + sx];
                let dst = (dst_row + x) * 4;
                pixels[dst] = stored_dose(dose, span);
                if let Some(&hu) = grid.ct.get(src_row + sx) {
                    pixels[dst + 1] = byte(((hu + 500.0) / 1000.0).clamp(0.0, 1.0));
                    pixels[dst + 2] = 255;
                }
                pixels[dst + 3] = 255;
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub struct PlaneImage {
    pub values: Vec<f32>,
    pub cols: usize,
    pub rows: usize,
    pub own: bool,
}

#[cfg(not(target_arch = "wasm32"))]
pub fn prepare_plane(grid: &DoseGrid, view: &ViewSample) -> Option<PlaneImage> {
    if view.plane == 3 {
        return None;
    }
    let (values, cols, rows, own) = plane_image(grid, view);
    Some(PlaneImage {
        values,
        cols,
        rows,
        own,
    })
}

#[cfg(not(target_arch = "wasm32"))]
pub fn shade_uv(
    grid: &DoseGrid,
    view: &ViewSample,
    plane: Option<&PlaneImage>,
    uv: [f32; 2],
) -> [f32; 4] {
    if view.plane == 3 {
        return shade_ray(grid, view, uv);
    }
    let film = turn_uv(uv, view.turns);
    let Some(image) = plane else {
        return [0.0, 0.0, 0.0, 0.0];
    };
    if image.cols == 0 || image.rows == 0 {
        return [0.0, 0.0, 0.0, 0.0];
    }
    let x = (film[0] * image.cols as f32).clamp(0.0, image.cols as f32 - 0.001);
    let y = ((1.0 - film[1]) * image.rows as f32).clamp(0.0, image.rows as f32 - 0.001);
    // Integrals are nearest on the GPU. A slice cubic is 2D Catmull-Rom: at an
    // integer depth the shader's z weights collapse onto this slice.
    let filter = if view.integral { 0 } else { grid.filter };
    let dose = image_sample(&image.values, image.cols, image.rows, x, y, filter);
    let (lo, hi) = if image.own {
        integral_limits_of(&image.values)
    } else {
        window(grid)
    };
    let t = ((dose * grid.gain - lo) / (hi - lo).max(1e-6)).clamp(0.0, 1.0);
    let rgb = sample(grid.ramp, t);
    let under = ct_under(grid, view, film);
    mix(under, [rgb[0], rgb[1], rgb[2], grid.opacity])
}

pub fn integral_peak(grid: &DoseGrid, plane: u8) -> f32 {
    grid.integral_peaks[(plane as usize).min(2)]
}

/// Color window for one plane integral. A positive projection sits on `0..peak`.
/// A difference keeps its sign, with zero in the middle of the ramp.
pub fn integral_limits(grid: &DoseGrid, plane: u8) -> (f32, f32) {
    integral_limits_of(&grid.integrals[(plane as usize).min(2)])
}

fn integral_limits_of(values: &[f32]) -> (f32, f32) {
    let mut peak = 1e-6f32;
    let mut negative = false;
    for value in values {
        if value.is_finite() {
            peak = peak.max(value.abs());
            negative |= *value < 0.0;
        }
    }
    if negative {
        (-peak, peak)
    } else {
        (0.0, peak)
    }
}

pub fn page(grid: &DoseGrid, plane: u8, index: usize, delta: i32) -> usize {
    let n = match plane {
        1 => grid.volume.shape[1],
        2 => grid.volume.shape[0],
        _ => grid.volume.shape[2],
    };
    (index as i32 + delta).clamp(0, n.saturating_sub(1) as i32) as usize
}

/// Boundary segments of each structure on this slice, in millimetres.
pub fn outlines(grid: &DoseGrid, plane: u8, index: usize) -> Vec<(u8, [f32; 2], [f32; 2])> {
    if grid.labels.is_empty() || plane > 2 {
        return Vec::new();
    }
    let [nx, ny, nz] = grid.volume.shape;
    let (cols, rows) = match plane {
        1 => (nx, nz),
        2 => (ny, nz),
        _ => (nx, ny),
    };
    let label_at = |col: usize, row: usize| -> usize {
        match plane {
            1 => col + nx * (index + ny * row),
            2 => index + nx * (col + ny * row),
            _ => col + nx * (row + ny * index),
        }
    };
    let mut lines = Vec::new();
    let origin = grid.volume.origin;
    let voxel = grid.volume.voxel;
    for row in 0..rows {
        for col in 0..cols {
            let id = grid.labels.get(label_at(col, row)).copied().unwrap_or(0);
            if id == 0 {
                continue;
            }
            let right = if col + 1 < cols {
                grid.labels
                    .get(label_at(col + 1, row))
                    .copied()
                    .unwrap_or(0)
            } else {
                0
            };
            let up = if row + 1 < rows {
                grid.labels
                    .get(label_at(col, row + 1))
                    .copied()
                    .unwrap_or(0)
            } else {
                0
            };
            let (x0, y0) = plane_mm(plane, origin, voxel, col, row);
            if right != id && lines.len() < 4000 {
                lines.push((id, [x0 + voxel, y0], [x0 + voxel, y0 + voxel]));
            }
            if up != id && lines.len() < 4000 {
                lines.push((id, [x0, y0 + voxel], [x0 + voxel, y0 + voxel]));
            }
        }
    }
    lines
}

fn plane_mm(plane: u8, origin: [f32; 3], voxel: f32, col: usize, row: usize) -> (f32, f32) {
    match plane {
        1 => (
            origin[0] + col as f32 * voxel,
            origin[2] + row as f32 * voxel,
        ),
        2 => (
            origin[1] + col as f32 * voxel,
            origin[2] + row as f32 * voxel,
        ),
        _ => (
            origin[0] + col as f32 * voxel,
            origin[1] + row as f32 * voxel,
        ),
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn shade_ray(grid: &DoseGrid, view: &ViewSample, uv: [f32; 2]) -> [f32; 4] {
    let (lo, hi) = window(grid);
    ray_rgba(
        &grid.volume,
        &grid.bricks,
        grid.brick_shape,
        RayView {
            azimuth: view.azimuth,
            elevation: view.elevation,
            zoom: view.zoom,
            aspect: view.aspect,
            mode: grid.mode,
            filter: grid.filter,
            gain: grid.gain,
            opacity: grid.opacity,
            lo,
            hi,
            ramp: grid.ramp,
            line_scale: if view.ray_scale > 0.0 {
                view.ray_scale
            } else {
                grid.line_scale
            },
            fov: view.fov,
            center: view.center,
            gantry: grid.gantry,
        },
        uv,
    )
}

#[cfg(not(target_arch = "wasm32"))]
fn plane_image(grid: &DoseGrid, view: &ViewSample) -> (Vec<f32>, usize, usize, bool) {
    if view.integral {
        let plane = (view.plane as usize).min(2);
        let [nx, ny, nz] = grid.volume.shape;
        let (cols, rows) = match plane {
            1 => (nx, nz),
            2 => (ny, nz),
            _ => (nx, ny),
        };
        return (grid.integrals[plane].clone(), cols, rows, true);
    }
    let n = match view.plane {
        1 => grid.volume.shape[1],
        2 => grid.volume.shape[0],
        _ => grid.volume.shape[2],
    };
    let index = view.index.min(n.saturating_sub(1));
    let (image, cols, rows) = grid.volume.slice_at(view.plane as usize, index);
    (image, cols, rows, false)
}

#[cfg(not(target_arch = "wasm32"))]
fn ct_under(grid: &DoseGrid, view: &ViewSample, film: [f32; 2]) -> Option<[f32; 3]> {
    if grid.ct.is_empty() || view.integral {
        return None;
    }
    let [nx, ny, nz] = grid.volume.shape;
    let (cols, rows) = match view.plane {
        1 => (nx, nz),
        2 => (ny, nz),
        _ => (nx, ny),
    };
    let hu = sample_plane(
        film[0] * cols as f32,
        (1.0 - film[1]) * rows as f32,
        cols,
        rows,
        grid.filter,
        |ix, iy| {
            let index = match view.plane {
                1 => ix.min(nx - 1) + nx * (view.index.min(ny - 1) + ny * iy.min(nz - 1)),
                2 => view.index.min(nx - 1) + nx * (ix.min(ny - 1) + ny * iy.min(nz - 1)),
                _ => ix.min(nx - 1) + nx * (iy.min(ny - 1) + ny * view.index.min(nz - 1)),
            };
            grid.ct.get(index).copied().unwrap_or(0.0)
        },
    );
    let t = ((hu + 500.0) / 1000.0).clamp(0.0, 1.0);
    Some([t, t, t])
}

/// `x` and `y` are cell coordinates: the center of voxel 0 is 0.5, matching
/// `film * n` in the slice shader. The shader then subtracts 0.5 before filtering.
#[cfg(not(target_arch = "wasm32"))]
fn image_sample(image: &[f32], cols: usize, rows: usize, x: f32, y: f32, filter: u8) -> f32 {
    sample_plane(x, y, cols, rows, filter, |ix, iy| {
        *image.get(ix + cols * iy).unwrap_or(&0.0)
    })
}

#[cfg(not(target_arch = "wasm32"))]
fn sample_plane(
    x: f32,
    y: f32,
    cols: usize,
    rows: usize,
    filter: u8,
    tap: impl Fn(usize, usize) -> f32,
) -> f32 {
    if cols == 0 || rows == 0 {
        return 0.0;
    }
    let x = x - 0.5;
    let y = y - 0.5;
    let ix_of = |ix: i32| ix.clamp(0, cols as i32 - 1) as usize;
    let iy_of = |iy: i32| iy.clamp(0, rows as i32 - 1) as usize;
    if filter == 0 {
        return tap(ix_of(x.round() as i32), iy_of(y.round() as i32));
    }
    if filter == 2 {
        let x0 = x.floor();
        let y0 = y.floor();
        let mut acc = 0.0;
        for oy in -1..3 {
            for ox in -1..3 {
                let w = catmull(x - (x0 + ox as f32)) * catmull(y - (y0 + oy as f32));
                acc += w * tap(ix_of(x0 as i32 + ox), iy_of(y0 as i32 + oy));
            }
        }
        return acc;
    }
    let x0 = x.floor();
    let y0 = y.floor();
    let tx = x - x0;
    let ty = y - y0;
    let mut acc = 0.0;
    for dy in 0..2 {
        for dx in 0..2 {
            let w = (if dx == 0 { 1.0 - tx } else { tx }) * (if dy == 0 { 1.0 - ty } else { ty });
            acc += w * tap(ix_of(x0 as i32 + dx), iy_of(y0 as i32 + dy));
        }
    }
    acc
}

#[cfg(not(target_arch = "wasm32"))]
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

#[cfg(not(target_arch = "wasm32"))]
fn window(grid: &DoseGrid) -> (f32, f32) {
    if grid.hi > grid.lo {
        (grid.lo, grid.hi)
    } else {
        (0.0, grid.peak)
    }
}

/// The dose picture fills the quad. The color axis is a column beside it.
pub(crate) const FILM_X: f32 = 1.0;

/// Display UV whose turned lookup is `film`. `film` is the unrotated slice.
fn unturn_uv(film: [f32; 2], turns: u8) -> [f32; 2] {
    match turns % 4 {
        1 => [1.0 - film[1], film[0]],
        2 => [1.0 - film[0], 1.0 - film[1]],
        3 => [film[1], 1.0 - film[0]],
        _ => film,
    }
}

fn turn_uv(uv: [f32; 2], turns: u8) -> [f32; 2] {
    match turns % 4 {
        1 => [uv[1], 1.0 - uv[0]],
        2 => [1.0 - uv[0], 1.0 - uv[1]],
        3 => [1.0 - uv[1], uv[0]],
        _ => uv,
    }
}

/// Where a slice millimetre is drawn. The picture fills the quad.
pub(crate) fn display_mm(x: f32, y: f32, bounds: [f32; 4], turns: u8) -> [f32; 2] {
    let width = (bounds[1] - bounds[0]).max(1e-6);
    let height = (bounds[3] - bounds[2]).max(1e-6);
    let film = [(x - bounds[0]) / width, (bounds[3] - y) / height];
    let uv = unturn_uv(film, turns);
    [
        bounds[0] + uv[0] * FILM_X * width,
        bounds[3] - uv[1] * height,
    ]
}

/// Slice millimetres under a data-space click.
pub(crate) fn source_mm(x: f32, y: f32, bounds: [f32; 4], turns: u8) -> Option<[f32; 2]> {
    let width = (bounds[1] - bounds[0]).max(1e-6);
    let height = (bounds[3] - bounds[2]).max(1e-6);
    let uvx = (x - bounds[0]) / width;
    let uvy = (bounds[3] - y) / height;
    let film = turn_uv([uvx / FILM_X, uvy], turns);
    Some([bounds[0] + film[0] * width, bounds[3] - film[1] * height])
}

#[cfg(not(target_arch = "wasm32"))]
fn mix(under: Option<[f32; 3]>, wash: [f32; 4]) -> [f32; 4] {
    let Some(under) = under else {
        return wash;
    };
    let a = wash[3].clamp(0.0, 1.0);
    [
        under[0] * (1.0 - a) + wash[0] * a,
        under[1] * (1.0 - a) + wash[1] * a,
        under[2] * (1.0 - a) + wash[2] * a,
        1.0,
    ]
}

fn byte(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// `0` is `-span`, `128` is zero, `255` is `+span`.
fn stored_dose(value: f32, span: f32) -> u8 {
    byte(0.5 + 0.5 * value / span.max(1e-6))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_atlas_keeps_a_voxel_and_a_ramp_row() {
        let grid = grid_from(
            vec![0.0, 2.0, 0.0, 0.0],
            Vec::new(),
            Vec::new(),
            [2, 1, 2],
            [0.0, 0.0, 0.0],
            1.0,
            0,
            0.0,
            2.0,
            1.0,
            1.0,
            0,
            0,
        )
        .unwrap();
        let (pixels, width, height) = atlas_bytes(&grid);
        assert!(width >= 256);
        assert!(height >= 2);
        let tiles = tile_count(grid.atlas_shape[2]);
        let nx = grid.atlas_shape[0];
        let ny = grid.atlas_shape[1];
        let x = 1;
        let z = 0;
        let tx = z % tiles;
        let dst = ((ny - 1) * width as usize + tx * nx + x) * 4;
        assert!(pixels[dst] > 200, "the hot voxel stays in the atlas");
        let cold = grid_from(
            vec![-2.0, 1.0],
            Vec::new(),
            Vec::new(),
            [2, 1, 1],
            [0.0, 0.0, 0.0],
            1.0,
            0,
            -2.0,
            2.0,
            1.0,
            1.0,
            0,
            0,
        )
        .unwrap();
        let (signed, signed_width, _) = atlas_bytes(&cold);
        assert!(
            signed[0] < 10,
            "a negative voxel stays below zero in the atlas"
        );
        let positive = signed[4];
        assert!(
            positive > 160 && positive < 220,
            "half the span lands between zero and the top, got {positive}"
        );
        assert!(signed_width >= 2);
        let ramp = ((height as usize - 1) * width as usize) * 4;
        assert_eq!(pixels.len(), width as usize * height as usize * 4);
        assert!(pixels[ramp + 3] > 0);
    }

    #[test]
    fn a_slice_center_reads_that_voxel() {
        let image = [0.0, 10.0];
        assert_eq!(image_sample(&image, 2, 1, 0.5, 0.5, 0), 0.0);
        assert!((image_sample(&image, 2, 1, 0.5, 0.5, 1) - 0.0).abs() < 1e-4);
        assert!((image_sample(&image, 2, 1, 0.5, 0.5, 2) - 0.0).abs() < 1e-3);
        assert_eq!(image_sample(&image, 2, 1, 1.5, 0.5, 0), 10.0);
        assert!((image_sample(&image, 2, 1, 1.0, 0.5, 1) - 5.0).abs() < 1e-4);
        assert!((image_sample(&image, 2, 1, 0.0, 0.5, 1) - 0.0).abs() < 1e-4);
    }

    #[test]
    fn a_turned_slice_is_not_rotated_twice() {
        let grid = grid_from(
            vec![10.0, 0.0],
            Vec::new(),
            Vec::new(),
            [2, 1, 1],
            [0.0, 0.0, 0.0],
            1.0,
            1,
            0.0,
            10.0,
            1.0,
            1.0,
            0,
            0,
        )
        .unwrap();
        let view = ViewSample {
            plane: 0,
            index: 0,
            turns: 1,
            integral: false,
            azimuth: 0.0,
            elevation: 0.0,
            zoom: 1.0,
            aspect: 1.0,
            ray_scale: 0.0,
            fov: 0.0,
            center: [0.0; 3],
        };
        let plane = prepare_plane(&grid, &view).unwrap();
        assert_eq!((plane.cols, plane.rows), (2, 1));
        assert_eq!(plane.values[0], 10.0);
        let shown = unturn_uv([0.25, 0.5], 1);
        let color = shade_uv(&grid, &view, Some(&plane), [shown[0] * FILM_X, shown[1]]);
        let hot = sample(1, 1.0);
        assert!((color[0] - hot[0]).abs() < 1e-3);
        assert!((color[1] - hot[1]).abs() < 1e-3);
        assert!((color[2] - hot[2]).abs() < 1e-3);
    }

    #[test]
    fn the_film_puts_a_voxel_on_its_pixel() {
        let bounds = [0.0, 100.0, 0.0, 50.0];
        let shown = display_mm(50.0, 25.0, bounds, 0);
        assert!((shown[0] - 50.0).abs() < 1e-3);
        assert!((shown[1] - 25.0).abs() < 1e-3);
        let back = source_mm(shown[0], shown[1], bounds, 0).unwrap();
        assert!((back[0] - 50.0).abs() < 1e-3);
        assert!((back[1] - 25.0).abs() < 1e-3);
        let edge = source_mm(96.0, 25.0, bounds, 0).unwrap();
        assert!((edge[0] - 96.0).abs() < 1e-3);
        let turned = display_mm(0.0, 25.0, bounds, 1);
        let back = source_mm(turned[0], turned[1], bounds, 1).unwrap();
        assert!((back[0] - 0.0).abs() < 1e-3);
        assert!((back[1] - 25.0).abs() < 1e-3);
    }

    #[test]
    fn a_negative_integral_keeps_its_sign_on_the_ramp() {
        let grid = grid_from(
            vec![4.0, -1.0, -4.0],
            Vec::new(),
            Vec::new(),
            [3, 1, 1],
            [0.0, 0.0, 0.0],
            1.0,
            1,
            -4.0,
            4.0,
            1.0,
            1.0,
            0,
            0,
        )
        .unwrap();
        let (lo, hi) = integral_limits(&grid, 0);
        assert!((lo + 4.0).abs() < 1e-4);
        assert!((hi - 4.0).abs() < 1e-4);
        let view = ViewSample {
            plane: 0,
            index: 0,
            turns: 0,
            integral: true,
            azimuth: 0.0,
            elevation: 0.0,
            zoom: 1.0,
            aspect: 1.0,
            ray_scale: 0.0,
            fov: 0.0,
            center: [0.0; 3],
        };
        let plane = prepare_plane(&grid, &view).unwrap();
        let color = shade_uv(&grid, &view, Some(&plane), [0.5 * FILM_X, 0.5]);
        let expected = sample(1, 0.375);
        assert!((color[0] - expected[0]).abs() < 1e-3);
        assert!((color[1] - expected[1]).abs() < 1e-3);
        assert!((color[2] - expected[2]).abs() < 1e-3);
        let floor = sample(1, 0.0);
        assert!(
            (color[0] - floor[0]).abs() + (color[1] - floor[1]).abs() + (color[2] - floor[2]).abs()
                > 0.05
        );
    }
}
