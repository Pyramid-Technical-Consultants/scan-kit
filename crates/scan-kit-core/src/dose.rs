//! Analytic proton dose in a uniform phantom, and the TG-218 gamma search.
//!
//! Stopping power is Bethe without shell or density-effect corrections. The
//! depth dose is Bortfeld's curve: a local power law on the CSDA range, smoothed
//! by the parabolic cylinder that convolves that power with range straggling.
//! Lateral width is Fermi–Eyges. These are the QA models in
//! `scan_kit/views/dose_volume_physics.py`.

use std::sync::OnceLock;

const K_BETHE: f64 = 0.307075;
const ME: f64 = 0.51099895;
const MP: f64 = 938.27208816;
const NUCLEAR_LOCAL: f64 = 0.6;
const E_SCATTER: f64 = 13.0;
const MEV_TO_GY_MM3: f64 = 1.602176634e-7;
const LAYER_NODES: usize = 512;
const LAYER_STEP: f64 = 0.1;
const MAX_LAYERS: usize = 1024;
const SIGMA_CUT: f64 = 4.0;
const MAX_CELLS: usize = 512;
// ponytail: 8e6 cell visits is the debug-build ceiling for a full 1 mm deposit.
// Above it, only the three planes through the weighted peak are filled.
// Upgrade path: a GPU splat of the same kernel.
const PLANE_VISITS: u64 = 8_000_000;
const TABLE_N: usize = 4000;
const W_AIR_EV: f64 = 33.97;
const E_CHARGE_C: f64 = 1.602176634e-19;
const RHO_AIR: f64 = 1.205e-3;
const PSTAR_S70: f64 = 9.20;
const PSTAR_E: f64 = 70.0;
const PSTAR_EXP: f64 = -0.643;

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

const WATER: Medium = Medium {
    key: "water",
    z_over_a: 0.55509,
    i_ev: 75.0,
    rho: 1.0,
    x0_g_cm2: 36.08,
    nuclear_cm2_g: 0.012,
};
const PMMA: Medium = Medium {
    key: "pmma",
    z_over_a: 0.53937,
    i_ev: 74.0,
    rho: 1.190,
    x0_g_cm2: 40.55,
    nuclear_cm2_g: 0.012,
};
const POLYSTYRENE: Medium = Medium {
    key: "polystyrene",
    z_over_a: 0.53768,
    i_ev: 68.7,
    rho: 1.060,
    x0_g_cm2: 43.79,
    nuclear_cm2_g: 0.012,
};
const POLYETHYLENE: Medium = Medium {
    key: "polyethylene",
    z_over_a: 0.57034,
    i_ev: 57.4,
    rho: 0.940,
    x0_g_cm2: 44.64,
    nuclear_cm2_g: 0.012,
};
const A150: Medium = Medium {
    key: "a150",
    z_over_a: 0.54915,
    i_ev: 65.1,
    rho: 1.127,
    x0_g_cm2: 41.9,
    nuclear_cm2_g: 0.012,
};
const ALUMINUM: Medium = Medium {
    key: "aluminum",
    z_over_a: 0.48181,
    i_ev: 166.0,
    rho: 2.699,
    x0_g_cm2: 24.01,
    nuclear_cm2_g: 0.008,
};
const COPPER: Medium = Medium {
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

struct RangeTable {
    energy: Vec<f64>,
    range_g: Vec<f64>,
    straggle: Vec<f64>,
    exponent: f64,
}

fn range_table(key: &'static str) -> &'static RangeTable {
    macro_rules! slot {
        ($name:ident, $medium:expr) => {{
            static SLOT: OnceLock<RangeTable> = OnceLock::new();
            SLOT.get_or_init(|| build_range_table($medium))
        }};
    }
    match key {
        "pmma" => slot!(pmma, PMMA),
        "polystyrene" => slot!(poly, POLYSTYRENE),
        "polyethylene" => slot!(pe, POLYETHYLENE),
        "a150" => slot!(a150, A150),
        "aluminum" => slot!(al, ALUMINUM),
        "copper" => slot!(cu, COPPER),
        _ => slot!(water, WATER),
    }
}

fn beta2(energy: f64) -> (f64, f64) {
    let g = 1.0 + energy.max(0.5) / MP;
    (1.0 - 1.0 / (g * g), g)
}

fn mass_stopping(medium: Medium, energy: f64) -> f64 {
    let (b2, g) = beta2(energy.max(0.5));
    let ratio = ME / MP;
    let tmax = 2.0 * ME * b2 * g * g / (1.0 + 2.0 * g * ratio + ratio * ratio);
    let i_mev = medium.i_ev * 1e-6;
    let arg = 2.0 * ME * b2 * g * g * tmax / (i_mev * i_mev);
    K_BETHE * medium.z_over_a / b2 * (0.5 * arg.ln() - b2)
}

fn build_range_table(medium: Medium) -> RangeTable {
    let mut energy = Vec::with_capacity(TABLE_N);
    let log_lo = 1.0f64.ln();
    let log_hi = 400.0f64.ln();
    for i in 0..TABLE_N {
        let t = i as f64 / (TABLE_N - 1) as f64;
        energy.push((log_lo + (log_hi - log_lo) * t).exp());
    }
    let stop: Vec<f64> = energy.iter().map(|e| mass_stopping(medium, *e)).collect();
    let r0 = energy[0] / (1.75 * stop[0]);
    let mut range_g = vec![r0];
    let mut straggle = vec![0.0];
    for i in 1..TABLE_N {
        let de = energy[i] - energy[i - 1];
        let inv = 0.5 * (1.0 / stop[i] + 1.0 / stop[i - 1]);
        range_g.push(range_g[i - 1] + inv * de);
        let (b2, _) = beta2(energy[i]);
        let (b2_prev, _) = beta2(energy[i - 1]);
        let omega = |b2: f64| K_BETHE * ME * medium.z_over_a * (1.0 - 0.5 * b2) / (1.0 - b2);
        let dvar = 0.5 * (omega(b2) / stop[i].powi(3) + omega(b2_prev) / stop[i - 1].powi(3));
        straggle.push(straggle[i - 1] + dvar * de);
    }
    let exponent = power_fit(&energy, &range_g);
    RangeTable {
        energy,
        range_g,
        straggle,
        exponent,
    }
}

fn power_fit(energy: &[f64], range_g: &[f64]) -> f64 {
    let mut n = 0.0;
    let mut sx = 0.0;
    let mut sy = 0.0;
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    for (e, r) in energy.iter().zip(range_g) {
        if (20.0..=250.0).contains(e) && *r > 0.0 {
            let x = e.ln();
            let y = r.ln();
            n += 1.0;
            sx += x;
            sy += y;
            sxx += x * x;
            sxy += x * y;
        }
    }
    let den = n * sxx - sx * sx;
    if den.abs() < 1e-18 {
        1.75
    } else {
        (n * sxy - sx * sy) / den
    }
}

fn log_interp(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    let x = x.max(xp[0]);
    if x >= xp[xp.len() - 1] {
        return fp[fp.len() - 1];
    }
    let i = xp.partition_point(|v| *v < x).saturating_sub(1);
    let i = i.min(xp.len() - 2);
    let t = (x.ln() - xp[i].ln()) / (xp[i + 1].ln() - xp[i].ln());
    fp[i] + t * (fp[i + 1] - fp[i])
}

/// CSDA range in millimetres.
pub fn csda_range_mm(medium: Medium, energy_mev: f64) -> f64 {
    let table = range_table(medium.key);
    log_interp(
        energy_mev.max(table.energy[0]),
        &table.energy,
        &table.range_g,
    ) * 10.0
        / medium.rho
}

/// Residual energy of a proton that still has `range_mm` to travel.
pub fn energy_at_range_mm(medium: Medium, range_mm: f64) -> f64 {
    let table = range_table(medium.key);
    let r_g = (range_mm * medium.rho / 10.0).max(1e-9);
    let range = &table.range_g;
    if r_g <= range[0] {
        return table.energy[0];
    }
    if r_g >= range[range.len() - 1] {
        return table.energy[range.len() - 1];
    }
    let i = range.partition_point(|r| *r < r_g) - 1;
    let t = (r_g.ln() - range[i].ln()) / (range[i + 1].ln() - range[i].ln());
    (table.energy[i].ln() + t * (table.energy[i + 1].ln() - table.energy[i].ln())).exp()
}

fn straggle_mm(medium: Medium, energy_mev: f64) -> f64 {
    let table = range_table(medium.key);
    let var = log_interp(
        energy_mev.max(table.energy[0]),
        &table.energy,
        &table.straggle,
    );
    var.max(0.0).sqrt() * 10.0 / medium.rho
}

fn depth_sigma_mm(medium: Medium, energy_mev: f64, spread_pct: f64) -> f64 {
    let e = energy_mev.max(1.0);
    let dr_de = 10.0 / (medium.rho * mass_stopping(medium, e));
    let spread = dr_de * e * spread_pct.max(0.0) / 100.0;
    (straggle_mm(medium, e).powi(2) + spread.powi(2)).sqrt()
}

fn gamma_lanczos(z: f64) -> f64 {
    const P: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if z < 0.5 {
        return std::f64::consts::PI / ((std::f64::consts::PI * z).sin() * gamma_lanczos(1.0 - z));
    }
    let z = z - 1.0;
    let mut x = P[0];
    for (i, p) in P.iter().enumerate().skip(1) {
        x += p / (z + i as f64);
    }
    let t = z + 7.5;
    (2.0 * std::f64::consts::PI).sqrt() * t.powf(z + 0.5) * (-t).exp() * x
}

fn hyp1f1(a: f64, b: f64, x: f64) -> f64 {
    let mut term = 1.0;
    let mut sum = 1.0;
    for k in 1..800 {
        term *= (a + k as f64 - 1.0) / (b + k as f64 - 1.0) * x / k as f64;
        sum += term;
        if term.abs() < 1e-16 * sum.abs().max(1.0) && k > 20 {
            break;
        }
    }
    sum
}

/// Parabolic cylinder `D_v(z)`, DLMF 12.2.5.
fn parabolic_d(v: f64, z: f64) -> f64 {
    let z2 = z * z / 2.0;
    let t1 =
        std::f64::consts::PI.sqrt() / gamma_lanczos((1.0 - v) / 2.0) * hyp1f1(-v / 2.0, 0.5, z2);
    let t2 = (2.0 * std::f64::consts::PI).sqrt() * z / gamma_lanczos(-v / 2.0)
        * hyp1f1((1.0 - v) / 2.0, 1.5, z2);
    2.0f64.powf(v / 2.0) * (-z * z / 4.0).exp() * (t1 - t2)
}

fn smoothed_power(nu: f64, r: f64, sigma: f64) -> f64 {
    let zeta = (r / sigma).max(-40.0);
    if zeta > 30.0 {
        return r.powf(nu);
    }
    if zeta < -8.0 {
        return 0.0;
    }
    let v = -nu - 1.0;
    let d = parabolic_d(v, -zeta);
    sigma.powf(nu) * gamma_lanczos(nu + 1.0) * (-zeta * zeta / 4.0).exp() * d
        / (2.0 * std::f64::consts::PI).sqrt()
}

/// Bortfeld depth dose per proton (MeV/mm), integrated over the beam cross-section.
pub fn bragg_idd(medium: Medium, energy_mev: f64, spread_pct: f64, depth_mm: f64) -> f64 {
    let table = range_table(medium.key);
    let e0 = energy_mev.max(1.0);
    let p = table.exponent;
    let r0 = csda_range_mm(medium, e0) / 10.0;
    let sigma = (depth_sigma_mm(medium, e0, spread_pct) / 10.0).max(1e-4);
    let alpha = r0 / e0.powf(p);
    let beta = medium.nuclear_cm2_g * medium.rho;
    let r = r0 - depth_mm / 10.0;
    let coef = beta * (1.0 + NUCLEAR_LOCAL * p);
    let val = smoothed_power(1.0 / p - 1.0, r, sigma) + coef * smoothed_power(1.0 / p, r, sigma);
    val / (p * alpha.powf(1.0 / p) * (1.0 + beta * r0)) / 10.0
}

fn pv(energy: f64) -> f64 {
    energy * (energy + 2.0 * MP) / (energy + MP)
}

fn mcs_along(medium: Medium, energy_mev: f64) -> (Vec<f64>, Vec<f64>) {
    let e0 = energy_mev.max(1.0);
    let r0 = csda_range_mm(medium, e0);
    let n = 512;
    let mut s = Vec::with_capacity(n);
    let mut t = Vec::with_capacity(n);
    let x0_mm = medium.x0_g_cm2 / medium.rho * 10.0;
    for i in 0..n {
        let z = r0 * i as f64 / (n - 1) as f64;
        s.push(z);
        let e = energy_at_range_mm(medium, r0 - z).max(0.5);
        t.push((E_SCATTER / pv(e)).powi(2) / x0_mm);
    }
    let mut a0 = vec![0.0];
    let mut a1 = vec![0.0];
    let mut a2 = vec![0.0];
    for i in 1..n {
        let ds = s[i] - s[i - 1];
        a0.push(a0[i - 1] + 0.5 * (t[i] + t[i - 1]) * ds);
        a1.push(a1[i - 1] + 0.5 * (t[i] * s[i] + t[i - 1] * s[i - 1]) * ds);
        a2.push(a2[i - 1] + 0.5 * (t[i] * s[i] * s[i] + t[i - 1] * s[i - 1] * s[i - 1]) * ds);
    }
    let var: Vec<f64> = (0..n)
        .map(|i| (s[i] * s[i] * a0[i] - 2.0 * s[i] * a1[i] + a2[i]).max(0.0))
        .collect();
    (s, var)
}

fn mcs_at(path: &[f64], var: &[f64], depth_mm: f64, r0: f64) -> f64 {
    let z = depth_mm.clamp(0.0, r0);
    lerp(z, path, var).max(0.0).sqrt()
}

struct LayerKernel {
    energies: Vec<f32>,
    zmin: Vec<f32>,
    cdf: Vec<f32>,
    mcs: Vec<f32>,
    energy_dep: Vec<f32>,
    mcs_max: Vec<f32>,
    nodes: usize,
}

fn layer_energies(energy: &[f32]) -> Vec<f32> {
    let raw: Vec<f64> = energy
        .iter()
        .copied()
        .filter(|e| e.is_finite() && *e > 1.0)
        .map(f64::from)
        .collect();
    if raw.is_empty() {
        return vec![100.0];
    }
    let mut step = LAYER_STEP;
    loop {
        let mut layers: Vec<i64> = raw.iter().map(|e| (e / step).round() as i64).collect();
        layers.sort_unstable();
        layers.dedup();
        if layers.len() <= MAX_LAYERS || step > 50.0 {
            return layers
                .into_iter()
                .map(|k| (k as f64 * step) as f32)
                .collect();
        }
        step *= 2.0;
    }
}

fn build_kernel(medium: Medium, energy: &[f32], spread_pct: f64, scatter: bool) -> LayerKernel {
    let energies = layer_energies(energy);
    let nodes = LAYER_NODES;
    let n = energies.len();
    let mut cdf = vec![0.0f32; n * nodes];
    let mut mcs = vec![0.0f32; n * nodes];
    let mut zmin = vec![0.0f32; n];
    let mut dep = vec![0.0f32; n];
    let mut mcs_max = vec![0.0f32; n];
    for (i, &e0) in energies.iter().enumerate() {
        let e = f64::from(e0);
        let r0 = csda_range_mm(medium, e);
        let sig = depth_sigma_mm(medium, e, spread_pct);
        let dmax = r0 + 5.0 * sig;
        let mut idd = vec![0.0; nodes];
        for k in 0..nodes {
            let depth = dmax * (nodes - 1 - k) as f64 / (nodes - 1) as f64;
            idd[k] = bragg_idd(medium, e, spread_pct, depth);
        }
        // idd[0] is the deep end, idd[last] is the surface. Integrate from deep to surface
        // so the CDF rises toward z = 0, matching the Python table.
        let step = dmax / (nodes - 1) as f64;
        let mut acc = vec![0.0; nodes];
        for k in 1..nodes {
            acc[k] = acc[k - 1] + 0.5 * (idd[k] + idd[k - 1]) * step;
        }
        let total = acc[nodes - 1].max(1e-30);
        dep[i] = total as f32;
        for k in 0..nodes {
            cdf[i * nodes + k] = (acc[k] / total) as f32;
        }
        zmin[i] = -dmax as f32;
        if scatter {
            let (path, var) = mcs_along(medium, e);
            let mut peak = 0.0f32;
            for k in 0..nodes {
                let depth = dmax * (nodes - 1 - k) as f64 / (nodes - 1) as f64;
                let w = mcs_at(&path, &var, depth, r0) as f32;
                mcs[i * nodes + k] = w;
                peak = peak.max(w);
            }
            mcs_max[i] = peak;
        }
    }
    LayerKernel {
        energies,
        zmin,
        cdf,
        mcs,
        energy_dep: dep,
        mcs_max,
        nodes,
    }
}

fn layer_of(kernel: &LayerKernel, energy: f32) -> usize {
    let energies = &kernel.energies;
    if energies.len() <= 1 {
        return 0;
    }
    let idx = energies
        .partition_point(|e| *e < energy)
        .clamp(1, energies.len() - 1);
    let lo = idx - 1;
    let hi = idx.min(energies.len() - 1);
    if (energies[hi] - energy).abs() < (energies[lo] - energy).abs() {
        hi
    } else {
        lo
    }
}

fn row_at(kernel: &LayerKernel, row: usize, at: f32, mcs: bool) -> f32 {
    let nodes = kernel.nodes;
    let z0 = kernel.zmin[row];
    let t = if z0.abs() < 1e-6 {
        1.0
    } else {
        (1.0 - at / z0).clamp(0.0, 1.0)
    };
    let x = t * (nodes - 1) as f32;
    let k = (x.floor() as usize).min(nodes - 2);
    let f = x - k as f32;
    let sample = |index: usize| {
        if mcs {
            kernel.mcs[row * nodes + index]
        } else {
            kernel.cdf[row * nodes + index]
        }
    };
    sample(k) + f * (sample(k + 1) - sample(k))
}

fn mass_fraction(kernel: &LayerKernel, energy: f32, z_lo: f32, z_hi: f32) -> f32 {
    let row = layer_of(kernel, energy);
    row_at(kernel, row, z_hi, false) - row_at(kernel, row, z_lo, false)
}

fn lateral_mm(kernel: &LayerKernel, energy: f32, z: f32) -> f32 {
    row_at(kernel, layer_of(kernel, energy), z, true).max(0.0)
}

fn lerp(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    if xp.is_empty() {
        return 0.0;
    }
    if x <= xp[0] {
        return fp[0];
    }
    if x >= xp[xp.len() - 1] {
        return fp[fp.len() - 1];
    }
    let i = xp.partition_point(|v| *v < x) - 1;
    let span = xp[i + 1] - xp[i];
    if span.abs() < 1e-18 {
        return fp[i];
    }
    let t = (x - xp[i]) / span;
    fp[i] + t * (fp[i + 1] - fp[i])
}

/// Energy left and primary protons kept after `wet_mm` of water.
pub fn through_wet(energy_mev: f64, wet_mm: f64) -> (f64, f64) {
    let e = energy_mev.max(1.0);
    let r = csda_range_mm(WATER, e);
    let left = r - wet_mm;
    if left <= 0.0 {
        return (0.0, 0.0);
    }
    let e_out = energy_at_range_mm(WATER, left.max(1e-6));
    let beta = WATER.nuclear_cm2_g * WATER.rho / 10.0;
    let kept = (1.0 + beta * left) / (1.0 + beta * r);
    (e_out, kept)
}

pub fn protons_from_mu(mu: f64, energy: f64, gap_mm: f64, k_mu: f64) -> f64 {
    let gap_cm = gap_mm.max(1e-6) / 10.0;
    let stop = PSTAR_S70 * (energy.max(1.0) / PSTAR_E).powf(PSTAR_EXP);
    let de = stop * RHO_AIR * gap_cm;
    let q = (de / (W_AIR_EV * 1e-6)) * E_CHARGE_C;
    mu * k_mu / q.max(1e-40)
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
}

#[derive(Clone, Debug)]
pub enum McJob {
    Slab(SlabRequest),
    Patient(PatientRequest),
}

/// Finished Monte Carlo dose. `ledger` is MeV per history:
/// incident, grid, off-grid, leaked, lost, beamline.
#[derive(Clone, Debug)]
pub struct McResult {
    pub volume: Volume,
    pub uncertainty: f32,
    pub ledger: [f32; 6],
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

    /// Depth from the entrance (mm) and dose, entrance first.
    pub fn depth_profile(&self, x: usize, y: usize) -> (Vec<f32>, Vec<f32>) {
        let nz = self.shape[2];
        let mut depth = Vec::with_capacity(nz);
        let mut dose = Vec::with_capacity(nz);
        for z in (0..nz).rev() {
            let z_mm = self.origin[2] + (z as f32 + 0.5) * self.voxel;
            depth.push(-z_mm);
            dose.push(self.get(x, y, z));
        }
        (depth, dose)
    }

    pub fn lateral_profile(&self, y: usize, z: usize) -> (Vec<f32>, Vec<f32>) {
        let nx = self.shape[0];
        let mut xs = Vec::with_capacity(nx);
        let mut dose = Vec::with_capacity(nx);
        for x in 0..nx {
            xs.push(self.origin[0] + (x as f32 + 0.5) * self.voxel);
            dose.push(self.get(x, y, z));
        }
        (xs, dose)
    }
}

fn erf_as(x: f64) -> f64 {
    let ax = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let p = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
        + 0.254829592)
        * t;
    x.signum() * (1.0 - p * (-ax * ax).exp())
}

fn normal_mass(mu: f64, sigma: f64, lo: f64, hi: f64) -> f64 {
    let sig = sigma.max(1e-6);
    let z1 = (hi - mu) / (sig * std::f64::consts::SQRT_2);
    let z0 = (lo - mu) / (sig * std::f64::consts::SQRT_2);
    0.5 * (erf_as(z1) - erf_as(z0))
}

struct Prepared {
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

fn prepare(
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

fn dose_grid(spots: &[Prepared], voxel: f32, z_floor: Option<f32>) -> ([f32; 3], [usize; 3]) {
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

fn hypot(a: f32, b: f32) -> f32 {
    (a * a + b * b).sqrt()
}

fn index_span(center: f32, sigma: f32, origin: f32, n: usize, voxel: f32) -> (usize, usize) {
    let sig = sigma.max(1e-3);
    let i0 = ((center - SIGMA_CUT as f32 * sig - origin) / voxel).floor() as i32;
    let i1 = ((center + SIGMA_CUT as f32 * sig - origin) / voxel).ceil() as i32;
    let a = i0.max(0) as usize;
    let b = i1.max(i0).min(n as i32).max(0) as usize;
    (a.min(n), b.min(n))
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

fn spot_box(
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

fn visit_count(
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

fn focus_index(
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

fn peak_z(kernel: &LayerKernel, spot: &Prepared, dose_mode: bool) -> f32 {
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

fn axis_index(coord: f32, origin: f32, voxel: f32, n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    let i = ((coord - origin) / voxel).floor() as i32;
    i.clamp(0, n as i32 - 1) as usize
}

fn depth_lateral(
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

fn normal_row(
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

fn accumulate_planes(
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

fn paint_planes(
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

fn scatter_planes(
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

/// 99.9th percentile of the positive samples. A single hot voxel does not set the window.
pub fn robust_high(values: &[f32]) -> f32 {
    let mut kept: Vec<f32> = values
        .iter()
        .copied()
        .filter(|v| v.is_finite() && *v > 0.0)
        .collect();
    if kept.is_empty() {
        return 1.0;
    }
    kept.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let index = ((kept.len() - 1) as f64 * 0.999) as usize;
    kept[index].max(1e-6)
}

/// Bounding box of samples at or above `threshold`, in the same coordinates as the slice axes.
pub fn field_bounds(
    image: &[f32],
    cols: usize,
    rows: usize,
    origin_x: f32,
    origin_y: f32,
    voxel: f32,
    threshold: f32,
) -> Option<[f32; 4]> {
    let mut x0 = cols;
    let mut x1 = 0;
    let mut y0 = rows;
    let mut y1 = 0;
    for y in 0..rows {
        for x in 0..cols {
            let value = image.get(x + cols * y).copied().unwrap_or(0.0);
            if value >= threshold {
                x0 = x0.min(x);
                x1 = x1.max(x + 1);
                y0 = y0.min(y);
                y1 = y1.max(y + 1);
            }
        }
    }
    if x1 <= x0 || y1 <= y0 {
        return None;
    }
    Some([
        origin_x + x0 as f32 * voxel,
        origin_x + x1 as f32 * voxel,
        origin_y + y0 as f32 * voxel,
        origin_y + y1 as f32 * voxel,
    ])
}

/// Global gamma. `cutoff_pct` is a percent of the maximum of `evaluated`.
/// Search uses sub-voxel steps and stops at γ = 2. Returns `(gamma, passed, evaluated)`.
pub fn gamma_index(
    reference: &[f32],
    evaluated: &[f32],
    shape: [usize; 3],
    dose_percent: f32,
    distance_mm: f32,
    spacing: [f32; 3],
    cutoff_pct: f32,
) -> (Vec<f32>, u32, u32) {
    let [nx, ny, nz] = shape;
    let n = nx * ny * nz;
    let len = n.min(reference.len()).min(evaluated.len());
    let mut gamma = vec![0.0f32; len];
    let mut norm = 0.0f32;
    for value in evaluated.iter().take(len) {
        if value.is_finite() {
            norm = norm.max(*value);
        }
    }
    if norm <= 0.0 {
        return (gamma, 0, 0);
    }
    let dd = (dose_percent.max(1e-6) / 100.0 * norm).max(1e-12);
    let floor = cutoff_pct / 100.0 * norm;
    let cap = 2.0f32;
    let mut mask = Vec::new();
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let i = x + nx * (y + ny * z);
                if i >= len {
                    continue;
                }
                let r = reference[i];
                if r.is_finite() && r > 0.0 && r >= floor {
                    mask.push((x, y, z, i));
                }
            }
        }
    }
    if mask.is_empty() {
        return (gamma, 0, 0);
    }
    let offsets = gamma_offsets(spacing, distance_mm.max(1e-3), cap);
    let mut best = vec![cap * cap; len];
    let mut worst = cap * cap;
    for offset in &offsets {
        if offset[3] >= worst {
            break;
        }
        let mut next = 0.0f32;
        for &(x, y, z, i) in &mask {
            let sample = sample_linear(
                evaluated,
                shape,
                x as f32 + offset[0],
                y as f32 + offset[1],
                z as f32 + offset[2],
            );
            let dose = (sample - reference[i]) / dd;
            let g2 = offset[3] + dose * dose;
            if g2 < best[i] {
                best[i] = g2;
            }
            next = next.max(best[i]);
        }
        worst = next;
    }
    let mut passed = 0u32;
    for &(_, _, _, i) in &mask {
        let g = best[i].sqrt();
        gamma[i] = g;
        if g <= 1.0 {
            passed += 1;
        }
    }
    (gamma, passed, mask.len() as u32)
}

fn gamma_offsets(spacing: [f32; 3], dta: f32, cap: f32) -> Vec<[f32; 4]> {
    let reach = cap * dta;
    let mut axes = Vec::new();
    for v in spacing {
        let step = v / (v / (dta / 4.0)).ceil().max(1.0);
        let n = (reach / step).ceil() as i32;
        let mut axis = Vec::new();
        for i in -n..=n {
            axis.push(i as f32 * step);
        }
        axes.push(axis);
    }
    let mut rows = Vec::new();
    for &gx in &axes[0] {
        for &gy in &axes[1] {
            for &gz in &axes[2] {
                let d2 = (gx * gx + gy * gy + gz * gz) / (dta * dta);
                if d2 <= cap * cap {
                    rows.push([gx / spacing[0], gy / spacing[1], gz / spacing[2], d2]);
                }
            }
        }
    }
    rows.sort_by(|a, b| a[3].partial_cmp(&b[3]).unwrap_or(std::cmp::Ordering::Equal));
    rows
}

fn sample_linear(vol: &[f32], shape: [usize; 3], x: f32, y: f32, z: f32) -> f32 {
    let [nx, ny, nz] = shape;
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let z0 = z.floor() as i32;
    let tx = x - x0 as f32;
    let ty = y - y0 as f32;
    let tz = z - z0 as f32;
    let mut acc = 0.0f32;
    for dz in 0..2 {
        for dy in 0..2 {
            for dx in 0..2 {
                let xx = x0 + dx;
                let yy = y0 + dy;
                let zz = z0 + dz;
                let value = if xx < 0
                    || yy < 0
                    || zz < 0
                    || xx >= nx as i32
                    || yy >= ny as i32
                    || zz >= nz as i32
                {
                    0.0
                } else {
                    vol[xx as usize + nx * (yy as usize + ny * zz as usize)]
                };
                let wx = if dx == 0 { 1.0 - tx } else { tx };
                let wy = if dy == 0 { 1.0 - ty } else { ty };
                let wz = if dz == 0 { 1.0 - tz } else { tz };
                acc += value * wx * wy * wz;
            }
        }
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn water_bragg_matches_the_python_table() {
        let range = csda_range_mm(WATER, 70.0);
        assert!((range - 40.742962).abs() < 0.02, "{range}");
        let samples = [
            (70.0, 0.0, 1.00971115),
            (70.0, 40.742962, 3.89316733),
            (150.0, 0.0, 0.62331327),
            (150.0, 157.670695, 1.94015664),
            (230.0, 0.0, 0.509487412),
            (230.0, 329.416942, 1.23637375),
        ];
        for (energy, depth, expect) in samples {
            let got = bragg_idd(WATER, energy, 1.0, depth);
            let rel = (got - expect).abs() / expect;
            assert!(rel < 0.01, "E {energy} d {depth}: {got} vs {expect}");
        }
    }

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
        );
        assert!(frame.planes_only(), "visits {}", frame.visits);
    }
}
