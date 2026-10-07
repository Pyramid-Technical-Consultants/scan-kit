//! Analytic proton dose in a uniform phantom, and the TG-218 gamma search.
//!
//! Stopping power is Bethe without shell or density-effect corrections. The
//! depth dose is Bortfeld's curve: a local power law on the CSDA range, smoothed
//! by the parabolic cylinder that convolves that power with range straggling.
//! Lateral width is Fermi–Eyges. These are the QA models in
//! `scan_kit/views/dose_volume_physics.py`.

mod deposit;
mod gamma;
mod kernel;
mod paint;
mod ray;

pub use deposit::{analytic_on, analytic_volume, dose_frame};
pub use gamma::{field_bounds, gamma_index, robust_high};
pub use kernel::{bragg_idd, csda_range_mm, protons_from_mu, through_wet};
pub use paint::{painted_unit, trim_number, wash_of, LevelSpan, PaintChoice, Wash};
pub use ray::{
    brick_grid, dose_film_open, dose_film_uv, film_height, line_scale, ray_rgba, ray_value,
    raymarch, sample_index, sample_mm, scan_volume, view_ray_scale, RayView, VolumeScan, BRICK,
    FOV_Y,
};

pub(super) const K_BETHE: f64 = 0.307075;
pub(super) const ME: f64 = 0.51099895;
pub(super) const MP: f64 = 938.27208816;
pub(super) const NUCLEAR_LOCAL: f64 = 0.6;
pub(super) const E_SCATTER: f64 = 13.0;
pub(super) const MEV_TO_GY_MM3: f64 = 1.602176634e-7;
pub(super) const LAYER_NODES: usize = 512;
pub(super) const LAYER_STEP: f64 = 0.1;
pub(super) const MAX_LAYERS: usize = 1024;
pub(super) const SIGMA_CUT: f64 = 4.0;
pub(super) const MAX_CELLS: usize = 512;
// A full 1 mm deposit past this many cell visits is expensive in a debug build.
// `analytic_on` can still fill only the three focus planes, but the volumetric
// view does not: that lattice raymarches as a cross.
pub(super) const PLANE_VISITS: u64 = 8_000_000;
pub(super) const TABLE_N: usize = 4000;
pub(super) const W_AIR_EV: f64 = 33.97;
pub(super) const E_CHARGE_C: f64 = 1.602176634e-19;
pub(super) const RHO_AIR: f64 = 1.205e-3;
pub(super) const PSTAR_S70: f64 = 9.20;
pub(super) const PSTAR_E: f64 = 70.0;
pub(super) const PSTAR_EXP: f64 = -0.643;

/// What a spot's `amount` means.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quantity {
    /// Bragg dose in Gy, from monitor units.
    Dose,
    /// Monitor units, deposited where the protons stop.
    Mu,
    /// Proton count, deposited where they stop.
    Protons,
}

/// One pencil at the phantom entrance. `z` of the dose is 0 at the surface and
/// negative along the beam.
#[derive(Clone, Copy, Debug)]
pub struct Pencil {
    pub x: f32,
    pub y: f32,
    pub sx: f32,
    pub sy: f32,
    pub energy: f32,
    pub amount: f32,
}

/// Uniform medium. `I`, density, and `Z/A` follow ICRU 49.
#[derive(Clone, Copy, Debug)]
pub struct Medium {
    pub key: &'static str,
    pub z_over_a: f64,
    pub i_ev: f64,
    pub rho: f64,
    pub x0_g_cm2: f64,
    pub nuclear_cm2_g: f64,
}

pub(super) const WATER: Medium = Medium {
    key: "water",
    z_over_a: 0.55509,
    i_ev: 75.0,
    rho: 1.0,
    x0_g_cm2: 36.08,
    nuclear_cm2_g: 0.012,
};
pub(super) const PMMA: Medium = Medium {
    key: "pmma",
    z_over_a: 0.53937,
    i_ev: 74.0,
    rho: 1.190,
    x0_g_cm2: 40.55,
    nuclear_cm2_g: 0.012,
};
pub(super) const POLYSTYRENE: Medium = Medium {
    key: "polystyrene",
    z_over_a: 0.53768,
    i_ev: 68.7,
    rho: 1.060,
    x0_g_cm2: 43.79,
    nuclear_cm2_g: 0.012,
};
pub(super) const POLYETHYLENE: Medium = Medium {
    key: "polyethylene",
    z_over_a: 0.57034,
    i_ev: 57.4,
    rho: 0.940,
    x0_g_cm2: 44.64,
    nuclear_cm2_g: 0.012,
};
pub(super) const A150: Medium = Medium {
    key: "a150",
    z_over_a: 0.54915,
    i_ev: 65.1,
    rho: 1.127,
    x0_g_cm2: 41.9,
    nuclear_cm2_g: 0.012,
};
pub(super) const ALUMINUM: Medium = Medium {
    key: "aluminum",
    z_over_a: 0.48181,
    i_ev: 166.0,
    rho: 2.699,
    x0_g_cm2: 24.01,
    nuclear_cm2_g: 0.008,
};
pub(super) const COPPER: Medium = Medium {
    key: "copper",
    z_over_a: 0.45636,
    i_ev: 322.0,
    rho: 8.96,
    x0_g_cm2: 12.86,
    nuclear_cm2_g: 0.0074,
};

/// Water when `key` is unknown.
pub fn medium(key: &str) -> Medium {
    match key {
        "pmma" => PMMA,
        "polystyrene" => POLYSTYRENE,
        "polyethylene" => POLYETHYLENE,
        "a150" => A150,
        "aluminum" => ALUMINUM,
        "copper" => COPPER,
        _ => WATER,
    }
}

pub fn water() -> Medium {
    WATER
}

/// A uniform-phantom Monte Carlo request. Spots are in scene millimetres and MeV.
#[derive(Clone, Debug)]
pub struct SlabRequest {
    pub medium: String,
    pub x: Vec<f32>,
    pub y: Vec<f32>,
    pub sx: Vec<f32>,
    pub sy: Vec<f32>,
    pub energy: Vec<f32>,
    pub protons: Vec<f32>,
    pub histories: u32,
    pub seed: u32,
    /// Percent, the same knob as the analytic energy spread.
    pub spread_pct: f32,
    pub wet_mm: f32,
    pub depth_mm: f32,
    pub voxel_mm: f32,
    pub origin: [f32; 3],
    pub shape: [usize; 3],
}

/// Patient Monte Carlo. `spots` are kernel records of stride 40. `shape` is `(nx, ny, nz)`.
#[derive(Clone, Debug)]
pub struct PatientRequest {
    pub spots: Vec<f32>,
    pub protons: Vec<f32>,
    pub beams: Vec<f32>,
    pub material: Vec<u8>,
    pub density: Vec<f32>,
    pub spacing_mm: [f32; 3],
    pub origin_mm: [f32; 3],
    pub shape: [usize; 3],
    pub histories: u32,
    pub seed: u32,
    pub dose_to_water: bool,
    /// MCsquare's medium LETd. Off leaves `McResult::let_d` empty.
    pub score_let: bool,
}

#[derive(Clone, Debug)]
pub enum McJob {
    Slab(SlabRequest),
    Patient(PatientRequest),
}

/// Finished Monte Carlo dose. `ledger` is MeV per history:
/// incident, grid, off-grid, leaked, lost, beamline.
/// `let_d` is dose-weighted LET in keV/µm, x-fastest, empty unless the
/// patient job asked for it.
#[derive(Clone, Debug)]
pub struct McResult {
    pub volume: Volume,
    pub uncertainty: f32,
    pub ledger: [f32; 6],
    pub let_d: Vec<f32>,
}

/// Dose grid. Values are x-fastest: `x + nx * (y + ny * z)`.
#[derive(Clone, Debug)]
pub struct Volume {
    pub origin: [f32; 3],
    pub shape: [usize; 3],
    pub voxel: f32,
    pub values: Vec<f32>,
}

impl Volume {
    pub fn get(&self, x: usize, y: usize, z: usize) -> f32 {
        let [nx, ny, _] = self.shape;
        self.values[x + nx * (y + ny * z)]
    }

    pub fn peak_index(&self) -> (usize, usize, usize) {
        let [nx, ny, nz] = self.shape;
        let mut best = 0.0f32;
        let mut at = (0, 0, 0);
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let v = self.get(x, y, z);
                    if v > best {
                        best = v;
                        at = (x, y, z);
                    }
                }
            }
        }
        at
    }

    pub fn axial(&self, z: usize) -> Vec<f32> {
        let [nx, ny, _] = self.shape;
        let mut image = vec![0.0; nx * ny];
        for y in 0..ny {
            for x in 0..nx {
                image[x + nx * y] = self.get(x, y, z);
            }
        }
        image
    }

    pub fn coronal(&self, y: usize) -> Vec<f32> {
        let [nx, _, nz] = self.shape;
        let mut image = vec![0.0; nx * nz];
        for z in 0..nz {
            for x in 0..nx {
                image[x + nx * z] = self.get(x, y, z);
            }
        }
        image
    }

    pub fn sagittal(&self, x: usize) -> Vec<f32> {
        let [_, ny, nz] = self.shape;
        let mut image = vec![0.0; ny * nz];
        for z in 0..nz {
            for y in 0..ny {
                image[y + ny * z] = self.get(x, y, z);
            }
        }
        image
    }

    /// Depth from the entrance face of the grid (mm) and dose, entrance first.
    /// The beam is −Z, so the high-z face is the entrance.
    pub fn depth_profile(&self, x: usize, y: usize) -> (Vec<f32>, Vec<f32>) {
        let nz = self.shape[2];
        let face = self.origin[2] + nz as f32 * self.voxel;
        let mut depth = Vec::with_capacity(nz);
        let mut dose = Vec::with_capacity(nz);
        for z in (0..nz).rev() {
            let z_mm = self.origin[2] + (z as f32 + 0.5) * self.voxel;
            depth.push(face - z_mm);
            dose.push(self.get(x, y, z));
        }
        (depth, dose)
    }

    /// Across the beam (X), millimetres from the crosshair.
    pub fn lateral_profile(&self, x: usize, y: usize, z: usize) -> (Vec<f32>, Vec<f32>) {
        let nx = self.shape[0];
        let x0 = self.origin[0] + (x as f32 + 0.5) * self.voxel;
        let mut xs = Vec::with_capacity(nx);
        let mut dose = Vec::with_capacity(nx);
        for column in 0..nx {
            xs.push(self.origin[0] + (column as f32 + 0.5) * self.voxel - x0);
            dose.push(self.get(column, y, z));
        }
        (xs, dose)
    }

    /// Along the beam, millimetres from the crosshair. Positive is deeper.
    pub fn longitudinal_profile(&self, x: usize, y: usize, z: usize) -> (Vec<f32>, Vec<f32>) {
        let nz = self.shape[2];
        let z0 = self.origin[2] + (z as f32 + 0.5) * self.voxel;
        let mut xs = Vec::with_capacity(nz);
        let mut dose = Vec::with_capacity(nz);
        for depth in 0..nz {
            let z_mm = self.origin[2] + (depth as f32 + 0.5) * self.voxel;
            xs.push(z0 - z_mm);
            dose.push(self.get(x, y, depth));
        }
        (xs, dose)
    }

    /// Sum across the profile's other in-plane axis. The unit is Gy·mm².
    pub fn depth_integral(&self, y: usize) -> (Vec<f32>, Vec<f32>) {
        let [nx, _, nz] = self.shape;
        let area = self.voxel * self.voxel;
        let face = self.origin[2] + nz as f32 * self.voxel;
        let mut depth = Vec::with_capacity(nz);
        let mut dose = Vec::with_capacity(nz);
        for z in (0..nz).rev() {
            let mut sum = 0.0;
            for x in 0..nx {
                sum += self.get(x, y, z);
            }
            depth.push(face - (self.origin[2] + (z as f32 + 0.5) * self.voxel));
            dose.push(sum * area);
        }
        (depth, dose)
    }

    pub fn lateral_integral(&self, x: usize, z: usize) -> (Vec<f32>, Vec<f32>) {
        let [nx, ny, _] = self.shape;
        let area = self.voxel * self.voxel;
        let x0 = self.origin[0] + (x as f32 + 0.5) * self.voxel;
        let mut xs = Vec::with_capacity(nx);
        let mut dose = Vec::with_capacity(nx);
        for column in 0..nx {
            let mut sum = 0.0;
            for y in 0..ny {
                sum += self.get(column, y, z);
            }
            xs.push(self.origin[0] + (column as f32 + 0.5) * self.voxel - x0);
            dose.push(sum * area);
        }
        (xs, dose)
    }

    pub fn longitudinal_integral(&self, y: usize, z: usize) -> (Vec<f32>, Vec<f32>) {
        let [nx, _, nz] = self.shape;
        let area = self.voxel * self.voxel;
        let z0 = self.origin[2] + (z as f32 + 0.5) * self.voxel;
        let mut xs = Vec::with_capacity(nz);
        let mut dose = Vec::with_capacity(nz);
        for depth in 0..nz {
            let mut sum = 0.0;
            for x in 0..nx {
                sum += self.get(x, y, depth);
            }
            let z_mm = self.origin[2] + (depth as f32 + 0.5) * self.voxel;
            xs.push(z0 - z_mm);
            dose.push(sum * area);
        }
        (xs, dose)
    }

    /// Sum through the plane. Values are Gy·mm and use their own range.
    pub fn integrated_slice(&self, plane: usize) -> (Vec<f32>, usize, usize) {
        let [nx, ny, nz] = self.shape;
        match plane {
            1 => {
                let mut image = vec![0.0; nx * nz];
                for z in 0..nz {
                    for x in 0..nx {
                        let mut sum = 0.0;
                        for y in 0..ny {
                            sum += self.get(x, y, z);
                        }
                        image[x + nx * z] = sum * self.voxel;
                    }
                }
                (image, nx, nz)
            }
            2 => {
                let mut image = vec![0.0; ny * nz];
                for z in 0..nz {
                    for y in 0..ny {
                        let mut sum = 0.0;
                        for x in 0..nx {
                            sum += self.get(x, y, z);
                        }
                        image[y + ny * z] = sum * self.voxel;
                    }
                }
                (image, ny, nz)
            }
            _ => {
                let mut image = vec![0.0; nx * ny];
                for y in 0..ny {
                    for x in 0..nx {
                        let mut sum = 0.0;
                        for z in 0..nz {
                            sum += self.get(x, y, z);
                        }
                        image[x + nx * y] = sum * self.voxel;
                    }
                }
                (image, nx, ny)
            }
        }
    }

    pub fn slice_at(&self, plane: usize, index: usize) -> (Vec<f32>, usize, usize) {
        match plane {
            1 => (self.coronal(index), self.shape[0], self.shape[2]),
            2 => (self.sagittal(index), self.shape[1], self.shape[2]),
            _ => (self.axial(index), self.shape[0], self.shape[1]),
        }
    }

    pub fn index_of(&self, axis: usize, mm: f32) -> usize {
        let n = self.shape[axis].max(1);
        let t = ((mm - self.origin[axis]) / self.voxel.max(1e-6)).floor();
        (t as isize).clamp(0, n as isize - 1) as usize
    }

    pub fn mm_of(&self, axis: usize, index: usize) -> f32 {
        self.origin[axis] + (index as f32 + 0.5) * self.voxel
    }
}

/// Lattice a pencil list would fill, without depositing.
#[derive(Clone, Copy, Debug)]
pub struct DoseFrame {
    pub origin: [f32; 3],
    pub shape: [usize; 3],
    pub voxel: f32,
    /// Voxel of the weighted Bragg peak. Plane fills paint through this point.
    pub focus: [usize; 3],
    /// Footprint updates a full deposit of these pencils would perform.
    pub visits: u64,
}

impl DoseFrame {
    pub fn planes_only(&self) -> bool {
        self.visits > PLANE_VISITS
    }
}
