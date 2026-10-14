//! One 80-bin cloud for scatter, density, and contour.
//!
//! Density and contour count the same grid. A live cloud also keeps the timed
//! samples so playback can rebin the visible window without walking the rows again.

use crate::plot::{CloudStyle, Series};
use crate::stats::percentile_linear;

/// Cells on each side of a density or contour grid.
pub const CLOUD_BINS: usize = 80;

/// How one cloud is drawn. The axis limits belong to the caller.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CloudDraw {
    Scatter {
        color: [f32; 4],
        radius: f32,
    },
    /// Bins into `[x0, x1]` by `[y0, y1]`, which a distribution panel shares.
    Density {
        ramp: u8,
        color: [f32; 4],
        x0: f32,
        x1: f32,
        y0: f32,
        y1: f32,
    },
    /// Levels sit on this cloud's own percentile extent.
    Contour {
        cutoff: f32,
    },
}

/// Scatter points. `times` is empty when the rows have no clock.
pub fn scatter_cloud(
    xs: Vec<f32>,
    ys: Vec<f32>,
    times: Vec<f32>,
    color: [f32; 4],
    radius: f32,
) -> Series {
    Series::Points {
        xs,
        ys,
        times,
        color,
        radius,
    }
}

/// Marks for one cloud. A clock adds a [`Series::Cloud`] after a density or contour
/// so the plot can rebin the playhead without a new scene.
pub fn cloud_series(xs: &[f32], ys: &[f32], times: &[f32], draw: CloudDraw) -> Vec<Series> {
    match draw {
        CloudDraw::Scatter { color, radius } => {
            let times = if times.len() == xs.len() {
                times.to_vec()
            } else {
                Vec::new()
            };
            vec![scatter_cloud(
                xs.to_vec(),
                ys.to_vec(),
                times,
                color,
                radius,
            )]
        }
        CloudDraw::Density {
            ramp,
            color,
            x0,
            x1,
            y0,
            y1,
        } => {
            let values = count_grid(xs, ys, x0, x1, y0, y1, CLOUD_BINS);
            let mut drawn = vec![Series::Heatmap {
                values,
                cols: CLOUD_BINS as u32,
                rows: CLOUD_BINS as u32,
                ramp,
                color,
                lo: 0.0,
                hi: 0.0,
            }];
            if let Some(source) =
                timed_source(xs, ys, times, x0, x1, y0, y1, CloudStyle::Density, 0.0)
            {
                drawn.push(source);
            }
            drawn
        }
        CloudDraw::Contour { cutoff } => {
            let Some(prep) = contour_prep(xs, ys) else {
                return Vec::new();
            };
            let mut drawn = contour_in_frame(
                &prep.xs, &prep.ys, cutoff, prep.x0, prep.x1, prep.y0, prep.y1,
            );
            if let Some(source) = timed_source(
                xs,
                ys,
                times,
                prep.x0,
                prep.x1,
                prep.y0,
                prep.y1,
                CloudStyle::Contour,
                cutoff,
            ) {
                drawn.push(source);
            }
            drawn
        }
    }
}

/// Nested density fills plus the isolines around each band.
pub fn contour_bands(xs: &[f32], ys: &[f32], cutoff_pct: f32) -> Vec<Series> {
    cloud_series(xs, ys, &[], CloudDraw::Contour { cutoff: cutoff_pct })
}

/// Contour of samples already limited to `x0..x1`, `y0..y1`.
/// Fewer than 20 finite samples inside that frame draws nothing.
pub fn contour_in_frame(
    xs: &[f32],
    ys: &[f32],
    cutoff_pct: f32,
    x0: f32,
    x1: f32,
    y0: f32,
    y1: f32,
) -> Vec<Series> {
    if x1 <= x0 || y1 <= y0 {
        return Vec::new();
    }
    let inside = xs
        .iter()
        .zip(ys)
        .filter(|(x, y)| {
            let (x, y) = (**x, **y);
            x.is_finite() && y.is_finite() && x >= x0 && x <= x1 && y >= y0 && y <= y1
        })
        .count();
    if inside < 20 {
        return Vec::new();
    }
    let counts = count_grid(xs, ys, x0, x1, y0, y1, CLOUD_BINS);
    let dx = (x1 - x0) / CLOUD_BINS as f32;
    let dy = (y1 - y0) / CLOUD_BINS as f32;
    contour_marks(&counts, CLOUD_BINS, cutoff_pct, x0, y0, dx, dy)
}

/// One pass into `bins` by `bins` cells. Samples outside the frame are skipped.
pub fn count_grid(
    xs: &[f32],
    ys: &[f32],
    x0: f32,
    x1: f32,
    y0: f32,
    y1: f32,
    bins: usize,
) -> Vec<f32> {
    let mut counts = vec![0.0f32; bins.saturating_mul(bins)];
    if bins == 0 {
        return counts;
    }
    let dx = x1 - x0;
    let dy = y1 - y0;
    let dx = if dx > 0.0 { dx } else { 1e-6 };
    let dy = if dy > 0.0 { dy } else { 1e-6 };
    for (x, y) in xs.iter().zip(ys) {
        if !x.is_finite() || !y.is_finite() || *x < x0 || *x > x1 || *y < y0 || *y > y1 {
            continue;
        }
        let ix = (((x - x0) / dx) * bins as f32) as usize;
        let iy = (((y - y0) / dy) * bins as f32) as usize;
        counts[ix.min(bins - 1) + bins * iy.min(bins - 1)] += 1.0;
    }
    counts
}

fn timed_source(
    xs: &[f32],
    ys: &[f32],
    times: &[f32],
    x0: f32,
    x1: f32,
    y0: f32,
    y1: f32,
    style: CloudStyle,
    cutoff: f32,
) -> Option<Series> {
    if times.len() != xs.len() || times.is_empty() {
        return None;
    }
    let (xs, ys, times) = sorted_cloud(xs, ys, times);
    Some(Series::Cloud {
        xs,
        ys,
        times,
        x0,
        x1,
        y0,
        y1,
        style,
        cutoff,
    })
}

fn sorted_cloud(xs: &[f32], ys: &[f32], times: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut order: Vec<usize> = (0..xs.len()).collect();
    order.sort_by(|&left, &right| times[left].total_cmp(&times[right]));
    (
        order.iter().map(|&index| xs[index]).collect(),
        order.iter().map(|&index| ys[index]).collect(),
        order.iter().map(|&index| times[index]).collect(),
    )
}

struct ContourPrep {
    x0: f32,
    x1: f32,
    y0: f32,
    y1: f32,
    xs: Vec<f32>,
    ys: Vec<f32>,
}

fn contour_prep(xs: &[f32], ys: &[f32]) -> Option<ContourPrep> {
    let mut pairs: Vec<(f32, f32)> = xs
        .iter()
        .zip(ys)
        .filter(|(x, y)| x.is_finite() && y.is_finite())
        .map(|(x, y)| (*x, *y))
        .collect();
    if pairs.len() < 20 {
        return None;
    }
    let px: Vec<f32> = pairs.iter().map(|pair| pair.0).collect();
    let py: Vec<f32> = pairs.iter().map(|pair| pair.1).collect();
    let (x0, x1) = density_range(&px);
    let (y0, y1) = density_range(&py);
    pairs.retain(|(x, y)| *x >= x0 && *x <= x1 && *y >= y0 && *y <= y1);
    if pairs.len() < 20 || x1 <= x0 || y1 <= y0 {
        return None;
    }
    Some(ContourPrep {
        x0,
        x1,
        y0,
        y1,
        xs: pairs.iter().map(|pair| pair.0).collect(),
        ys: pairs.iter().map(|pair| pair.1).collect(),
    })
}

fn density_range(values: &[f32]) -> (f32, f32) {
    let mut owned = values.to_vec();
    let lo = percentile_linear(&mut owned, 0.0005);
    let mut owned = values.to_vec();
    let hi = percentile_linear(&mut owned, 0.9995);
    if lo.is_finite() && hi.is_finite() && hi > lo {
        return (lo, hi);
    }
    let mut lo = f32::MAX;
    let mut hi = f32::MIN;
    for value in values {
        lo = lo.min(*value);
        hi = hi.max(*value);
    }
    if hi <= lo {
        (lo - 0.5, hi + 0.5)
    } else {
        let mid = (lo + hi) / 2.0;
        let half = (hi - lo) / 2.0;
        (mid - half, mid + half)
    }
}

fn contour_marks(
    counts: &[f32],
    bins: usize,
    cutoff_pct: f32,
    x0: f32,
    y0: f32,
    dx: f32,
    dy: f32,
) -> Vec<Series> {
    let positive: Vec<f32> = counts
        .iter()
        .copied()
        .filter(|value| *value > 0.0)
        .collect();
    let z_max = positive.iter().copied().fold(0.0, f32::max);
    if z_max <= 0.0 {
        return Vec::new();
    }
    let lo = cutoff_pct.clamp(0.0, 90.0).min(97.0);
    let levels: Vec<f32> = if lo >= 97.0 {
        vec![97.0]
    } else {
        (0..6)
            .map(|step| lo + (97.0 - lo) * step as f32 / 5.0)
            .collect()
    };
    let mut levels: Vec<f32> = levels
        .into_iter()
        .map(|level| {
            let mut owned = positive.clone();
            percentile_linear(&mut owned, f64::from(level) / 100.0)
        })
        .filter(|level| level.is_finite() && *level > 0.0 && *level < z_max)
        .collect();
    levels.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    levels.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
    if levels.is_empty() {
        return Vec::new();
    }
    let (mesh_x, mesh_y) = contour_mesh(counts, bins, &levels, x0, y0, dx, dy);
    let mut drawn = Vec::new();
    if mesh_x.len() >= 3 {
        drawn.push(Series::Triangles {
            xs: mesh_x,
            ys: mesh_y,
            color: [0.8, 0.8, 0.8, 0.13],
        });
    }
    let (line_x, line_y) = isolines(counts, bins, &levels, x0, y0, dx, dy);
    if line_x.len() >= 2 {
        drawn.push(Series::Polyline {
            xs: line_x,
            ys: line_y,
            color: [0.8, 0.8, 0.8, 0.0],
            thickness: 1.0,
        });
    }
    drawn
}

/// The part of each cell on or above a level. Corners and edge crossings match `isolines`.
fn contour_mesh(
    counts: &[f32],
    bins: usize,
    levels: &[f32],
    x0: f32,
    y0: f32,
    dx: f32,
    dy: f32,
) -> (Vec<f32>, Vec<f32>) {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let at = |ix: usize, iy: usize| counts[ix + bins * iy];
    for level in levels {
        for iy in 0..bins.saturating_sub(1) {
            for ix in 0..bins.saturating_sub(1) {
                let c00 = at(ix, iy);
                let c10 = at(ix + 1, iy);
                let c11 = at(ix + 1, iy + 1);
                let c01 = at(ix, iy + 1);
                let case = u8::from(c00 >= *level)
                    | (u8::from(c10 >= *level) << 1)
                    | (u8::from(c11 >= *level) << 2)
                    | (u8::from(c01 >= *level) << 3);
                if case == 0 {
                    continue;
                }
                let x = x0 + (ix as f32 + 0.5) * dx;
                let y = y0 + (iy as f32 + 0.5) * dy;
                let p00 = (x, y);
                let p10 = (x + dx, y);
                let p11 = (x + dx, y + dy);
                let p01 = (x, y + dy);
                let cross =
                    |edge| edge_point(ix, iy, edge, c00, c10, c11, c01, *level, x0, y0, dx, dy);
                let e0 = cross(0);
                let e1 = cross(1);
                let e2 = cross(2);
                let e3 = cross(3);
                match case {
                    1 => fan(&mut xs, &mut ys, &[p00, e0, e3]),
                    2 => fan(&mut xs, &mut ys, &[p10, e1, e0]),
                    3 => fan(&mut xs, &mut ys, &[p00, p10, e1, e3]),
                    4 => fan(&mut xs, &mut ys, &[p11, e2, e1]),
                    5 => {
                        fan(&mut xs, &mut ys, &[p00, e0, e3]);
                        fan(&mut xs, &mut ys, &[p11, e2, e1]);
                    }
                    6 => fan(&mut xs, &mut ys, &[e0, p10, p11, e2]),
                    7 => fan(&mut xs, &mut ys, &[p00, p10, p11, e2, e3]),
                    8 => fan(&mut xs, &mut ys, &[p01, e3, e2]),
                    9 => fan(&mut xs, &mut ys, &[p00, e0, e2, p01]),
                    10 => {
                        fan(&mut xs, &mut ys, &[p10, e1, e0]);
                        fan(&mut xs, &mut ys, &[p01, e3, e2]);
                    }
                    11 => fan(&mut xs, &mut ys, &[p00, p10, e1, e2, p01]),
                    12 => fan(&mut xs, &mut ys, &[e3, e1, p11, p01]),
                    13 => fan(&mut xs, &mut ys, &[p00, e0, e1, p11, p01]),
                    14 => fan(&mut xs, &mut ys, &[e0, p10, p11, p01, e3]),
                    _ => fan(&mut xs, &mut ys, &[p00, p10, p11, p01]),
                }
            }
        }
    }
    (xs, ys)
}

fn fan(xs: &mut Vec<f32>, ys: &mut Vec<f32>, poly: &[(f32, f32)]) {
    if poly.len() < 3 {
        return;
    }
    for index in 1..poly.len() - 1 {
        xs.extend([poly[0].0, poly[index].0, poly[index + 1].0]);
        ys.extend([poly[0].1, poly[index].1, poly[index + 1].1]);
    }
}

fn isolines(
    counts: &[f32],
    bins: usize,
    levels: &[f32],
    x0: f32,
    y0: f32,
    dx: f32,
    dy: f32,
) -> (Vec<f32>, Vec<f32>) {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let at = |ix: usize, iy: usize| counts[ix + bins * iy];
    for level in levels {
        for iy in 0..bins.saturating_sub(1) {
            for ix in 0..bins.saturating_sub(1) {
                let c00 = at(ix, iy);
                let c10 = at(ix + 1, iy);
                let c11 = at(ix + 1, iy + 1);
                let c01 = at(ix, iy + 1);
                let case = u8::from(c00 >= *level)
                    | (u8::from(c10 >= *level) << 1)
                    | (u8::from(c11 >= *level) << 2)
                    | (u8::from(c01 >= *level) << 3);
                let segments: &[(u8, u8)] = match case {
                    1 | 14 => &[(3, 0)],
                    2 | 13 => &[(0, 1)],
                    3 | 12 => &[(3, 1)],
                    4 | 11 => &[(1, 2)],
                    6 | 9 => &[(0, 2)],
                    7 | 8 => &[(3, 2)],
                    5 => &[(3, 0), (1, 2)],
                    10 => &[(0, 1), (2, 3)],
                    _ => &[],
                };
                for (a, b) in segments {
                    let (ax, ay) =
                        edge_point(ix, iy, *a, c00, c10, c11, c01, *level, x0, y0, dx, dy);
                    let (bx, by) =
                        edge_point(ix, iy, *b, c00, c10, c11, c01, *level, x0, y0, dx, dy);
                    xs.push(ax);
                    ys.push(ay);
                    xs.push(bx);
                    ys.push(by);
                    xs.push(f32::NAN);
                    ys.push(f32::NAN);
                }
            }
        }
    }
    (xs, ys)
}

fn edge_point(
    ix: usize,
    iy: usize,
    edge: u8,
    c00: f32,
    c10: f32,
    c11: f32,
    c01: f32,
    level: f32,
    x0: f32,
    y0: f32,
    dx: f32,
    dy: f32,
) -> (f32, f32) {
    let x = x0 + (ix as f32 + 0.5) * dx;
    let y = y0 + (iy as f32 + 0.5) * dy;
    let frac = |a: f32, b: f32| {
        if (b - a).abs() < 1e-12 {
            0.5
        } else {
            ((level - a) / (b - a)).clamp(0.0, 1.0)
        }
    };
    match edge {
        0 => (x + frac(c00, c10) * dx, y),
        1 => (x + dx, y + frac(c10, c11) * dy),
        2 => (x + frac(c01, c11) * dx, y + dy),
        _ => (x, y + frac(c00, c01) * dy),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contour_counts_every_sample() {
        let n = 16_001;
        let mut xs = vec![0.0f32; n];
        let mut ys = vec![0.0f32; n];
        for index in (1..n).step_by(2) {
            let turn = index as f32;
            xs[index] = (turn * 0.01).sin();
            ys[index] = (turn * 0.013).cos();
        }
        assert!(!contour_bands(&xs, &ys, 5.0).is_empty());
    }

    #[test]
    fn contour_fill_uses_the_isoline_vertices() {
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for i in 0..50 {
            for j in 0..50 {
                let x = (i as f32 - 25.0) * 0.15;
                let y = (j as f32 - 25.0) * 0.15;
                let weight = (-0.5 * (x * x + y * y) / 0.8).exp();
                for _ in 0..(weight * 8.0) as i32 {
                    xs.push(x);
                    ys.push(y);
                }
            }
        }
        let drawn = contour_bands(&xs, &ys, 5.0);
        let Series::Triangles {
            xs: mesh_x,
            ys: mesh_y,
            color,
        } = &drawn[0]
        else {
            panic!("contour fill should be a triangle mesh");
        };
        assert!((color[3] - 0.13).abs() < 1e-6);
        assert_eq!(mesh_x.len() % 3, 0);
        assert!(mesh_x.len() > 12);
        let Series::Polyline {
            xs: line_x,
            ys: line_y,
            ..
        } = &drawn[1]
        else {
            panic!("contour should keep its isolines");
        };
        let mut shared = 0;
        for (x, y) in line_x.iter().zip(line_y) {
            if !x.is_finite() {
                continue;
            }
            assert!(
                mesh_x
                    .iter()
                    .zip(mesh_y)
                    .any(|(mx, my)| { (mx - x).abs() < 1e-4 && (my - y).abs() < 1e-4 }),
                "isoline point {x},{y} is missing from the fill"
            );
            shared += 1;
        }
        assert!(shared > 8);
    }

    #[test]
    fn density_and_contour_share_one_grid() {
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for index in 0..40 {
            xs.push(index as f32);
            ys.push((index % 5) as f32);
        }
        let grid = count_grid(&xs, &ys, 0.0, 40.0, 0.0, 5.0, 8);
        assert_eq!(grid.iter().sum::<f32>(), 40.0);
        let Series::Heatmap { values, cols, .. } = &cloud_series(
            &xs,
            &ys,
            &[],
            CloudDraw::Density {
                ramp: 0,
                color: [1.0; 4],
                x0: 0.0,
                x1: 40.0,
                y0: 0.0,
                y1: 5.0,
            },
        )[0] else {
            panic!("density is a heatmap");
        };
        assert_eq!(*cols, CLOUD_BINS as u32);
        assert_eq!(values.iter().sum::<f32>(), 40.0);
    }
}
