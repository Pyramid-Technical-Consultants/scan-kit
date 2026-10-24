//! Case tables and tolerances, ported from `mcsquare_validate.py`.

use scan_kit_io::{csda_range_mm, report::sig6};

pub const EV_PER_G_TO_GY: f64 = 1.602176e-16;
pub const MEV_TO_J: f64 = 1.602176634e-13;
pub const SPREAD_PCT: f64 = 0.7;
pub const SEED: u32 = 1;
pub const LATERAL_MM: usize = 100;
pub const SIGMA_DEPTHS: [f64; 3] = [0.25, 0.5, 0.9];
pub const CROP_FRAC: f32 = 0.05;
pub const CROP_MARGIN: usize = 3;
pub const GOLDEN_PRIMARIES: f64 = 1e7;
/// g/cm3; MCsquare's Ignore_low_density_voxels threshold — air is not compared.
pub const LOW_DENSITY: f32 = 0.1;
/// DVH metrics over the reference's >= 50 % region.
pub const TARGET_FRAC: f32 = 0.5;

#[derive(Clone, Copy)]
pub struct Tolerances {
    pub histories: f64,
    pub idd: f64,
    pub r80: f64,
    pub sigma: f64,
    pub energy: f64,
    pub centroid: f64,
    pub gamma_pct: f64,
    pub gamma: f64,
}

#[derive(Clone, Copy)]
pub struct PatientTolerances {
    pub histories: f64,
    pub gamma_pct: f64,
    pub gamma_mm: f64,
    pub gamma: f64,
    pub dmean: f64,
    pub d95: f64,
    pub d2: f64,
    pub let_tol: f64,
}

// fast runs 10x fewer GPU histories than the goldens, so its noise allows less.
pub const FAST_TOL: Tolerances = Tolerances {
    histories: 1e6,
    idd: 0.02,
    r80: 0.25,
    sigma: 0.03,
    energy: 0.01,
    centroid: 0.2,
    gamma_pct: 2.0,
    gamma: 0.98,
};

pub const FULL_TOL: Tolerances = Tolerances {
    histories: 1e7,
    idd: 0.01,
    r80: 0.2,
    sigma: 0.02,
    energy: 0.005,
    centroid: 0.1,
    gamma_pct: 1.0,
    gamma: 0.99,
};

pub const PATIENT_TOL: PatientTolerances = PatientTolerances {
    histories: 1e7,
    gamma_pct: 2.0,
    gamma_mm: 2.0,
    gamma: 0.99,
    dmean: 0.01,
    d95: 0.02,
    d2: 0.02,
    let_tol: 0.03,
};

#[derive(Clone)]
pub struct PatientCase {
    pub name: String,
    /// slab in the upstream half of the tissue block.
    pub insert_hu: Option<f32>,
    pub gantry: f64,
    pub couch: f64,
    /// mm water equivalent.
    pub rs_wet: f64,
    pub energies: Vec<f64>,
    /// MCsquare's Sample_input_data CT and PlanPencil.txt instead.
    pub sample: bool,
}

pub fn patient_cases() -> Vec<PatientCase> {
    let base = |name: &str, insert_hu| PatientCase {
        name: name.into(),
        insert_hu,
        gantry: 0.0,
        couch: 0.0,
        rs_wet: 0.0,
        energies: vec![130.0, 140.0],
        sample: false,
    };
    vec![
        base("tissue_bone_slab", Some(1200.0)),
        base("tissue_lung_slab", Some(-700.0)),
        PatientCase {
            rs_wet: 40.0,
            energies: vec![150.0, 160.0],
            ..base("tissue_range_shifter", None)
        },
        PatientCase {
            gantry: 45.0,
            couch: 20.0,
            ..base("bone_slab_g45_c20", Some(1200.0))
        },
        PatientCase {
            sample: true,
            ..base("sample_ct_3field", None)
        },
    ]
}

#[derive(Clone)]
pub struct Case {
    pub name: String,
    pub medium: &'static str,
    /// mm, both axes, at the nozzle.
    pub sigma: f32,
    /// (x mm, y mm, energy MeV, protons) per spot.
    pub spots: Vec<[f64; 4]>,
    /// mm of water in front of the phantom.
    pub wet: usize,
    pub lateral: usize,
    /// x histories; a field spreads them over far more voxels than a pencil.
    pub scale: f64,
    /// also in the fast suite.
    pub fast: bool,
}

impl Case {
    pub fn pencil(&self) -> bool {
        self.spots.len() == 1
    }

    pub fn depth(&self) -> usize {
        let e_max = self.spots.iter().map(|s| s[2]).fold(0.0f64, f64::max);
        let range = csda_range_mm(self.medium, e_max).unwrap_or(f64::NAN);
        (1.1 * range + 10.0).ceil() as usize
    }
}

fn pencil(medium: &'static str, energy: f64, sigma: f32, wet: usize, kw: Pencil) -> Case {
    let Pencil {
        lateral,
        fast,
        name,
    } = kw;
    let name = name.unwrap_or_else(|| {
        let wet_part = if wet > 0 {
            format!("_wet{wet}")
        } else {
            String::new()
        };
        format!(
            "{medium}_{}_s{}{wet_part}",
            sig6(energy),
            sig6(f64::from(sigma))
        )
    });
    Case {
        name,
        medium,
        sigma,
        spots: vec![[0.0, 0.0, energy, 1.0]],
        wet,
        lateral,
        scale: 1.0,
        fast,
    }
}

#[derive(Clone, Default)]
struct Pencil {
    lateral: usize,
    fast: bool,
    name: Option<String>,
}

fn field() -> Case {
    let mut spots = Vec::new();
    for (i, energy) in [130.0, 135.0, 140.0, 145.0, 150.0].iter().enumerate() {
        for yi in 0..7 {
            for xi in 0..7 {
                spots.push([
                    -15.0 + xi as f64 * 5.0,
                    -15.0 + yi as f64 * 5.0,
                    *energy,
                    1.0 + i as f64,
                ]);
            }
        }
    }
    Case {
        name: "water_field_5x7x7".into(),
        medium: "water",
        sigma: 4.0,
        spots,
        wet: 0,
        lateral: 120,
        scale: 4.0,
        fast: false,
    }
}

pub fn cases() -> Vec<Case> {
    let fast = |m: &'static str, e: f64, s: f32, wet: usize, lateral: usize, name: Option<&str>| {
        pencil(
            m,
            e,
            s,
            wet,
            Pencil {
                lateral,
                fast: true,
                name: name.map(str::to_string),
            },
        )
    };
    let slow = |m: &'static str, e: f64, s: f32, wet: usize| {
        pencil(
            m,
            e,
            s,
            wet,
            Pencil {
                lateral: LATERAL_MM,
                fast: false,
                name: None,
            },
        )
    };
    let mut out = vec![
        // fast: small grids, each aimed at a place a transport bug would show first.
        // scatter-dominated width:
        fast("water", 100.0, 1.0, 0, 40, Some("water_100_s1_narrow")),
        // high Z, 6 mm range, ends below the ICRU 7 MeV cut:
        fast("copper", 70.0, 3.0, 0, 30, None),
        // water -> aluminum interface:
        fast("aluminum", 120.0, 3.0, 40, 40, None),
        // nuclear build-up and secondary protons:
        fast("water", 180.0, 3.0, 0, 40, None),
        // off-axis spots, mixed energies and weights: placement, mirroring, weighting:
        Case {
            name: "pmma_offaxis_3spot".into(),
            medium: "pmma",
            sigma: 3.0,
            spots: vec![
                [12.0, -7.0, 100.0, 1.0],
                [-8.0, 9.0, 120.0, 2.0],
                [0.0, 0.0, 110.0, 0.5],
            ],
            wet: 0,
            lateral: 60,
            scale: 1.0,
            fast: true,
        },
    ];
    // full only
    for energy in [70.0, 100.0, 150.0, 200.0, 230.0] {
        for sigma in [3.0, 6.0] {
            out.push(slow("water", energy, sigma, 0));
        }
    }
    out.push(slow("pmma", 150.0, 4.0, 0));
    out.push(slow("polystyrene", 150.0, 4.0, 0));
    out.push(slow("aluminum", 100.0, 4.0, 0));
    out.push(slow("copper", 100.0, 4.0, 0));
    out.push(slow("water", 150.0, 4.0, 30));
    out.push(field());
    out
}
