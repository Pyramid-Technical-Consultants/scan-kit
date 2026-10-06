//! Dose workspace sampling. The volume is uploaded once as an atlas. Slice and
//! 3D cells read that atlas, so paging, rotation, and orbit do not build a new one.
//!
//! Atlas: z slices in a square of tiles, row 0 of a slice at the bottom of its
//! tile. The last texture row is the color scale, 256 entries. Voxel R is
//! dose/peak, G is the CT window, B is 1 when a CT sample is present.

#[cfg(not(target_arch = "wasm32"))]
use scan_kit_core::ray_value;
use scan_kit_core::{sample, Volume};

/// Display state copied from the scene. The plot owns it after the upload.
#[derive(Clone, Debug)]
pub struct DoseGrid {
    pub volume: Volume,
    pub ct: Vec<f32>,
    pub labels: Vec<u8>,
    pub peak: f32,
    pub ramp: u8,
    pub lo: f32,
    pub hi: f32,
    pub gain: f32,
    pub opacity: f32,
    /// 0 integrate, 1 maximum, 2 transparent.
    pub mode: u8,
    /// 0 nearest, 1 linear, 2 cubic.
    pub filter: u8,
    pub atlas_shape: [usize; 3],
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
    let peak = values
        .iter()
        .take(n)
        .copied()
        .fold(0.0f32, f32::max)
        .max(1e-6);
    let volume = Volume {
        origin,
        shape: shape_us,
        voxel: voxel.max(1e-3),
        values: values[..n].to_vec(),
    };
    Some(DoseGrid {
        atlas_shape: fit_shape(shape_us),
        volume,
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
        peak,
        ramp,
        lo,
        hi,
        gain,
        opacity,
        mode,
        filter,
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
    for z in 0..nz {
        let tx = z % tiles;
        let ty = z / tiles;
        for y in 0..ny {
            for x in 0..nx {
                let sx = if fx == nx { x } else { x * fx / nx };
                let sy = if fy == ny { y } else { y * fy / ny };
                let sz = if fz == nz { z } else { z * fz / nz };
                let dose = grid
                    .volume
                    .get(sx.min(fx - 1), sy.min(fy - 1), sz.min(fz - 1));
                let index = sx + fx * (sy + fy * sz);
                let (ct, has_ct) = grid
                    .ct
                    .get(index)
                    .copied()
                    .map(|hu| ((((hu + 500.0) / 1000.0).clamp(0.0, 1.0)), 255u8))
                    .unwrap_or((0.0, 0));
                let dst = ((ty * ny + (ny - 1 - y)) * width + tx * nx + x) * 4;
                let t = (dose / grid.peak).clamp(0.0, 1.0);
                pixels[dst] = byte(t);
                pixels[dst + 1] = byte(ct);
                pixels[dst + 2] = has_ct;
                pixels[dst + 3] = 255;
            }
        }
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
    if uv[0] > 0.94 {
        let t = (1.0 - uv[1]).clamp(0.0, 1.0);
        let rgb = sample(grid.ramp, t);
        return [rgb[0], rgb[1], rgb[2], grid.opacity];
    }
    let film = turn_uv([uv[0] / 0.94, uv[1]], view.turns);
    if view.plane == 3 {
        return shade_ray(grid, view, film);
    }
    let Some(image) = plane else {
        return [0.0, 0.0, 0.0, 0.0];
    };
    if image.cols == 0 || image.rows == 0 {
        return [0.0, 0.0, 0.0, 0.0];
    }
    let x = (film[0] * image.cols as f32).clamp(0.0, image.cols as f32 - 0.001);
    let y = ((1.0 - film[1]) * image.rows as f32).clamp(0.0, image.rows as f32 - 0.001);
    let dose = image_sample(&image.values, image.cols, image.rows, x, y, grid.filter);
    let (lo, hi) = if image.own {
        let peak = image
            .values
            .iter()
            .copied()
            .fold(0.0f32, f32::max)
            .max(1e-6);
        (0.0, peak)
    } else {
        window(grid)
    };
    let t = ((dose * grid.gain - lo) / (hi - lo).max(1e-6)).clamp(0.0, 1.0);
    let rgb = sample(grid.ramp, t);
    let under = ct_under(grid, view, film);
    mix(under, [rgb[0], rgb[1], rgb[2], grid.opacity])
}

pub fn integral_peak(grid: &DoseGrid, plane: u8) -> f32 {
    let (image, _, _) = grid.volume.integrated_slice(plane as usize);
    image.into_iter().fold(0.0f32, f32::max).max(1e-6)
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
    let u = uv[0] * 2.0 - 1.0;
    let v = (1.0 - uv[1]) * 2.0 - 1.0;
    let dose = ray_value(
        &grid.volume,
        view.azimuth,
        view.elevation,
        grid.mode,
        grid.filter,
        u,
        v,
    );
    let (lo, hi) = window(grid);
    let t = ((dose * grid.gain - lo) / (hi - lo).max(1e-6)).clamp(0.0, 1.0);
    let rgb = sample(grid.ramp, t);
    [rgb[0], rgb[1], rgb[2], grid.opacity]
}

#[cfg(not(target_arch = "wasm32"))]
fn plane_image(grid: &DoseGrid, view: &ViewSample) -> (Vec<f32>, usize, usize, bool) {
    if view.integral {
        let (image, cols, rows) = grid.volume.integrated_slice(view.plane as usize);
        let (turned, cols, rows) = Volume::rotate_plane(&image, cols, rows, view.turns);
        return (turned, cols, rows, true);
    }
    let n = match view.plane {
        1 => grid.volume.shape[1],
        2 => grid.volume.shape[0],
        _ => grid.volume.shape[2],
    };
    let index = view.index.min(n.saturating_sub(1));
    let (image, cols, rows) = grid.volume.slice_at(view.plane as usize, index);
    let (turned, cols, rows) = Volume::rotate_plane(&image, cols, rows, view.turns);
    (turned, cols, rows, false)
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
    let x = (film[0] * cols as f32) as usize;
    let y = ((1.0 - film[1]) * rows as f32) as usize;
    let index = match view.plane {
        1 => x.min(nx - 1) + nx * (view.index.min(ny - 1) + ny * y.min(nz - 1)),
        2 => view.index.min(nx - 1) + nx * (x.min(ny - 1) + ny * y.min(nz - 1)),
        _ => x.min(nx - 1) + nx * (y.min(ny - 1) + ny * view.index.min(nz - 1)),
    };
    grid.ct.get(index).copied().map(|hu| {
        let t = ((hu + 500.0) / 1000.0).clamp(0.0, 1.0);
        [t, t, t]
    })
}

#[cfg(not(target_arch = "wasm32"))]
fn image_sample(image: &[f32], cols: usize, rows: usize, x: f32, y: f32, filter: u8) -> f32 {
    if filter == 0 {
        let ix = x.round() as usize;
        let iy = y.round() as usize;
        return *image
            .get(ix.min(cols - 1) + cols * iy.min(rows - 1))
            .unwrap_or(&0.0);
    }
    let x0 = x.floor();
    let y0 = y.floor();
    let tx = x - x0;
    let ty = y - y0;
    let mut acc = 0.0;
    for dy in 0..2 {
        for dx in 0..2 {
            let ix = (x0 as i32 + dx).clamp(0, cols as i32 - 1) as usize;
            let iy = (y0 as i32 + dy).clamp(0, rows as i32 - 1) as usize;
            let w = (if dx == 0 { 1.0 - tx } else { tx }) * (if dy == 0 { 1.0 - ty } else { ty });
            acc += w * image[ix + cols * iy];
        }
    }
    acc
}

#[cfg(not(target_arch = "wasm32"))]
fn window(grid: &DoseGrid) -> (f32, f32) {
    if grid.hi > grid.lo {
        (grid.lo, grid.hi)
    } else {
        (0.0, grid.peak)
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn turn_uv(uv: [f32; 2], turns: u8) -> [f32; 2] {
    match turns % 4 {
        1 => [uv[1], 1.0 - uv[0]],
        2 => [1.0 - uv[0], 1.0 - uv[1]],
        3 => [1.0 - uv[1], uv[0]],
        _ => uv,
    }
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
        let ramp = ((height as usize - 1) * width as usize) * 4;
        assert_eq!(pixels.len(), width as usize * height as usize * 4);
        assert!(pixels[ramp + 3] > 0);
    }
}
