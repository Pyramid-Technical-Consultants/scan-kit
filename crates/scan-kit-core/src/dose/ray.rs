//! Orthographic ray march through a dose grid.
//!
//! `scan-kit-plot` repeats this sampling in WGSL for the 3D cell. The CPU path
//! is the fallback and the check that Integrate, Maximum, and Transparent disagree
//! on a peaked volume.

use super::Volume;

const STEPS: usize = 48;

/// `mode`: 0 integrate, 1 maximum, 2 transparent.
/// `filter`: 0 nearest, 1 linear, 2 cubic.
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
    let [nx, ny, nz] = volume.shape;
    let center = [
        volume.origin[0] + nx as f32 * volume.voxel * 0.5,
        volume.origin[1] + ny as f32 * volume.voxel * 0.5,
        volume.origin[2] + nz as f32 * volume.voxel * 0.5,
    ];
    let radius = (nx.max(ny).max(nz) as f32) * volume.voxel * 0.5 * 1.732;
    let (dir, right, up) = camera_basis(azimuth, elevation);
    let peak = volume
        .values
        .iter()
        .copied()
        .fold(0.0f32, f32::max)
        .max(1e-6);
    let mut image = vec![0.0; cols * rows];
    for row in 0..rows {
        for col in 0..cols {
            let u = (col as f32 + 0.5) / cols as f32 * 2.0 - 1.0;
            let v = (row as f32 + 0.5) / rows as f32 * 2.0 - 1.0;
            image[col + cols * row] = march_ray(
                volume, dir, right, up, center, radius, peak, mode, filter, u, v,
            );
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
    let [nx, ny, nz] = volume.shape;
    let center = [
        volume.origin[0] + nx as f32 * volume.voxel * 0.5,
        volume.origin[1] + ny as f32 * volume.voxel * 0.5,
        volume.origin[2] + nz as f32 * volume.voxel * 0.5,
    ];
    let radius = (nx.max(ny).max(nz) as f32) * volume.voxel * 0.5 * 1.732;
    let (dir, right, up) = camera_basis(azimuth, elevation);
    let peak = volume
        .values
        .iter()
        .copied()
        .fold(0.0f32, f32::max)
        .max(1e-6);
    march_ray(
        volume, dir, right, up, center, radius, peak, mode, filter, u, v,
    )
}

fn march_ray(
    volume: &Volume,
    dir: [f32; 3],
    right: [f32; 3],
    up: [f32; 3],
    center: [f32; 3],
    radius: f32,
    peak: f32,
    mode: u8,
    filter: u8,
    u: f32,
    v: f32,
) -> f32 {
    let start = [
        center[0] + right[0] * u * radius + up[0] * v * radius - dir[0] * radius,
        center[1] + right[1] * u * radius + up[1] * v * radius - dir[1] * radius,
        center[2] + right[2] * u * radius + up[2] * v * radius - dir[2] * radius,
    ];
    let step = radius * 2.0 / STEPS as f32;
    let mut acc = 0.0f32;
    let mut cover = 0.0f32;
    let mut best = 0.0f32;
    for i in 0..STEPS {
        let t = i as f32 + 0.5;
        let point = [
            start[0] + dir[0] * step * t,
            start[1] + dir[1] * step * t,
            start[2] + dir[2] * step * t,
        ];
        let dose = sample_mm(volume, point, filter);
        match mode {
            1 => best = best.max(dose),
            2 => {
                let alpha = 1.0 - (-dose / peak * 1.5).exp();
                acc += (1.0 - cover) * dose * alpha;
                cover += (1.0 - cover) * alpha;
            }
            _ => acc += dose * step,
        }
    }
    match mode {
        1 => best,
        _ => acc,
    }
}

pub fn sample_mm(volume: &Volume, mm: [f32; 3], filter: u8) -> f32 {
    let voxel = volume.voxel.max(1e-6);
    sample_index(
        volume,
        (mm[0] - volume.origin[0]) / voxel - 0.5,
        (mm[1] - volume.origin[1]) / voxel - 0.5,
        (mm[2] - volume.origin[2]) / voxel - 0.5,
        filter,
    )
}

pub fn sample_index(volume: &Volume, x: f32, y: f32, z: f32, filter: u8) -> f32 {
    match filter {
        2 => cubic(volume, x, y, z),
        1 => linear(volume, x, y, z),
        _ => {
            let [nx, ny, nz] = volume.shape;
            let ix = x.round().clamp(0.0, nx.saturating_sub(1) as f32) as usize;
            let iy = y.round().clamp(0.0, ny.saturating_sub(1) as f32) as usize;
            let iz = z.round().clamp(0.0, nz.saturating_sub(1) as f32) as usize;
            at(volume, ix, iy, iz)
        }
    }
}

fn camera_basis(azimuth: f32, elevation: f32) -> ([f32; 3], [f32; 3], [f32; 3]) {
    let ce = elevation.cos();
    let dir = [ce * azimuth.sin(), ce * azimuth.cos(), elevation.sin()];
    let right = [azimuth.cos(), -azimuth.sin(), 0.0];
    let up = [
        dir[1] * right[2] - dir[2] * right[1],
        dir[2] * right[0] - dir[0] * right[2],
        dir[0] * right[1] - dir[1] * right[0],
    ];
    (normalize(dir), normalize(right), normalize(up))
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
    acc.max(0.0)
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
            voxel: 1.0,
            values,
        }
    }

    #[test]
    fn integrate_maximum_and_transparent_disagree() {
        let volume = peaked();
        let integrate = raymarch(&volume, 0.4, 0.3, 0, 0, 8, 8);
        let maximum = raymarch(&volume, 0.4, 0.3, 1, 0, 8, 8);
        let transparent = raymarch(&volume, 0.4, 0.3, 2, 1, 8, 8);
        let peak = |image: &[f32]| image.iter().copied().fold(0.0f32, f32::max);
        assert!(peak(&integrate) > 0.0);
        assert!(peak(&maximum) <= 4.0 + 1e-3);
        assert!(peak(&transparent) > 0.0);
        assert!(peak(&integrate) != peak(&maximum) || peak(&transparent) != peak(&maximum));
    }

    #[test]
    fn linear_sample_sits_between_neighbors() {
        let volume = Volume {
            origin: [0.0, 0.0, 0.0],
            shape: [2, 1, 1],
            voxel: 1.0,
            values: vec![0.0, 4.0],
        };
        let mid = sample_index(&volume, 0.5, 0.0, 0.0, 1);
        assert!((mid - 2.0).abs() < 1e-3, "{mid}");
    }
}
