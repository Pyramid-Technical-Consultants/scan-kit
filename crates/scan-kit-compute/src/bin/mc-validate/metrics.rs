//! Metrics on `(nz, ny, nx)` Gy-per-proton volumes, z index 0 deepest, 1 mm
//! voxels — direct ports of `mcsquare_validate.py`.

use crate::cases::{Case, MEV_TO_J, SIGMA_DEPTHS};

/// Laterally integrated dose, surface first.
pub fn depth_dose(vol: &[f32], nz: usize, ny: usize, nx: usize) -> Vec<f64> {
    let plane = ny * nx;
    (0..nz)
        .map(|i| {
            let z = nz - 1 - i;
            vol[z * plane..(z + 1) * plane]
                .iter()
                .map(|&v| f64::from(v))
                .sum()
        })
        .collect()
}

/// Depth (mm) where `idd` falls through `frac` of its maximum beyond the peak.
pub fn distal_depth(idd: &[f64], frac: f64) -> f64 {
    let peak = idd
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);
    let level = frac * idd[peak];
    for i in peak..idd.len().saturating_sub(1) {
        if idd[i + 1] < level && level <= idd[i] {
            return i as f64 + 0.5 + (idd[i] - level) / (idd[i] - idd[i + 1]);
        }
    }
    f64::NAN
}

/// Gaussian sigma (voxels) from the full width at half maximum.
pub fn fwhm_sigma(profile: &[f64]) -> f64 {
    let peak = profile
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);
    let half = 0.5 * profile[peak];
    let mut lo = peak;
    while lo > 0 && profile[lo - 1] >= half {
        lo -= 1;
    }
    let mut hi = peak;
    while hi < profile.len() - 1 && profile[hi + 1] >= half {
        hi += 1;
    }
    if lo == 0 || hi == profile.len() - 1 {
        return f64::NAN;
    }
    let left = lo as f64 - (profile[lo] - half) / (profile[lo] - profile[lo - 1]);
    let right = hi as f64 + (profile[hi] - half) / (profile[hi] - profile[hi + 1]);
    (right - left) / (2.0 * (2.0 * std::f64::consts::LN_2).sqrt())
}

/// Sigma of the x-lateral profile at each depth (mm, surface first).
pub fn lateral_sigmas(vol: &[f32], nz: usize, ny: usize, nx: usize, depths: &[f64]) -> Vec<f64> {
    let plane = ny * nx;
    depths
        .iter()
        .map(|d| {
            let i = (*d as usize).min(nz - 1);
            let z = nz - 1 - i; // surface first
            let profile: Vec<f64> = (0..nx)
                .map(|x| {
                    (0..ny)
                        .map(|y| f64::from(vol[z * plane + y * nx + x]))
                        .sum()
                })
                .collect();
            fwhm_sigma(&profile)
        })
        .collect()
}

/// Dose-weighted (x, y) in plan mm.
pub fn centroid_mm(vol: &[f32], _nz: usize, ny: usize, nx: usize) -> [f64; 2] {
    let plane = ny * nx;
    let nz = vol.len() / plane;
    let mut x_num = 0.0;
    let mut y_num = 0.0;
    let mut total = 0.0;
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let v = f64::from(vol[z * plane + y * nx + x]);
                x_num += v * (x as f64 + 0.5 - nx as f64 / 2.0);
                y_num += v * (y as f64 + 0.5 - ny as f64 / 2.0);
                total += v;
            }
        }
    }
    [x_num / total, y_num / total]
}

/// Energy deposited in the grid per proton (1 mm voxels).
pub fn energy_mev(vol: &[f32], rho: f64) -> f64 {
    vol.iter().map(|&v| f64::from(v)).sum::<f64>() * rho * 1e-6 / MEV_TO_J
}

pub struct Summary {
    pub idd: Vec<f64>,
    pub r80: f64,
    pub r90: f64,
    pub sigma_depths: Vec<f64>,
    pub sigmas: Vec<f64>,
    pub energy: f64,
    pub centroid: [f64; 2],
}

pub fn summary(vol: &[f32], nz: usize, ny: usize, nx: usize, case: &Case, rho: f64) -> Summary {
    let idd = depth_dose(vol, nz, ny, nx);
    let r80 = distal_depth(&idd, 0.8);
    let r90 = distal_depth(&idd, 0.9);
    let depths: Vec<f64> = SIGMA_DEPTHS.iter().map(|f| f * r80).collect();
    Summary {
        idd,
        r80,
        r90,
        sigma_depths: depths.clone(),
        sigmas: if case.pencil() {
            lateral_sigmas(vol, nz, ny, nx, &depths)
        } else {
            vec![f64::NAN; depths.len()]
        },
        energy: energy_mev(vol, rho),
        centroid: centroid_mm(vol, nz, ny, nx),
    }
}

pub struct Diff {
    pub idd: f64,
    pub r80: f64,
    pub sigma: f64,
    pub energy: f64,
    pub centroid: f64,
}

/// Relative / absolute differences of `gpu` from `ref` summaries.
pub fn compare(ref_: &Summary, gpu: &Summary) -> Diff {
    let upto = ref_.r90 as usize;
    let n = upto.min(gpu.idd.len()).min(ref_.idd.len());
    let idd = (0..n)
        .map(|i| (gpu.idd[i] / ref_.idd[i] - 1.0).abs())
        .fold(0.0f64, f64::max);
    let sig: Vec<f64> = gpu
        .sigmas
        .iter()
        .zip(&ref_.sigmas)
        .map(|(g, r)| g / r - 1.0)
        .collect();
    Diff {
        idd,
        r80: gpu.r80 - ref_.r80,
        sigma: if sig.iter().all(|v| v.is_finite()) {
            sig.iter().copied().map(f64::abs).fold(0.0, f64::max)
        } else {
            f64::NAN
        },
        energy: gpu.energy / ref_.energy - 1.0,
        centroid: gpu
            .centroid
            .iter()
            .zip(&ref_.centroid)
            .map(|(g, r)| (g - r).abs())
            .fold(0.0, f64::max),
    }
}

/// np.percentile (linear interpolation).
pub fn percentile(data: &[f64], q: f64) -> f64 {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if sorted.is_empty() {
        return f64::NAN;
    }
    let rank = q / 100.0 * (sorted.len() - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    sorted[lo] + (rank - lo as f64) * (sorted[hi] - sorted[lo])
}
