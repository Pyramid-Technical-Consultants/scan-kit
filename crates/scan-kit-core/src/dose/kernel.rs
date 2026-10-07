use std::sync::OnceLock;

use super::{
    Medium, A150, ALUMINUM, COPPER, E_CHARGE_C, E_SCATTER, K_BETHE, LAYER_NODES, LAYER_STEP,
    MAX_LAYERS, ME, MP, NUCLEAR_LOCAL, PMMA, POLYETHYLENE, POLYSTYRENE, PSTAR_E, PSTAR_EXP,
    PSTAR_S70, RHO_AIR, TABLE_N, WATER, W_AIR_EV,
};

pub(super) struct RangeTable {
    energy: Vec<f64>,
    range_g: Vec<f64>,
    straggle: Vec<f64>,
    exponent: f64,
}

pub(super) fn range_table(key: &'static str) -> &'static RangeTable {
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

pub(super) fn beta2(energy: f64) -> (f64, f64) {
    let g = 1.0 + energy.max(0.5) / MP;
    (1.0 - 1.0 / (g * g), g)
}

pub(super) fn mass_stopping(medium: Medium, energy: f64) -> f64 {
    let (b2, g) = beta2(energy.max(0.5));
    let ratio = ME / MP;
    let tmax = 2.0 * ME * b2 * g * g / (1.0 + 2.0 * g * ratio + ratio * ratio);
    let i_mev = medium.i_ev * 1e-6;
    let arg = 2.0 * ME * b2 * g * g * tmax / (i_mev * i_mev);
    K_BETHE * medium.z_over_a / b2 * (0.5 * arg.ln() - b2)
}

pub(super) fn build_range_table(medium: Medium) -> RangeTable {
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

pub(super) fn power_fit(energy: &[f64], range_g: &[f64]) -> f64 {
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

pub(super) fn log_interp(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
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

pub(super) fn straggle_mm(medium: Medium, energy_mev: f64) -> f64 {
    let table = range_table(medium.key);
    let var = log_interp(
        energy_mev.max(table.energy[0]),
        &table.energy,
        &table.straggle,
    );
    var.max(0.0).sqrt() * 10.0 / medium.rho
}

pub(super) fn depth_sigma_mm(medium: Medium, energy_mev: f64, spread_pct: f64) -> f64 {
    let e = energy_mev.max(1.0);
    let dr_de = 10.0 / (medium.rho * mass_stopping(medium, e));
    let spread = dr_de * e * spread_pct.max(0.0) / 100.0;
    (straggle_mm(medium, e).powi(2) + spread.powi(2)).sqrt()
}

pub(super) fn gamma_lanczos(z: f64) -> f64 {
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

pub(super) fn hyp1f1(a: f64, b: f64, x: f64) -> f64 {
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
pub(super) fn parabolic_d(v: f64, z: f64) -> f64 {
    let z2 = z * z / 2.0;
    let t1 =
        std::f64::consts::PI.sqrt() / gamma_lanczos((1.0 - v) / 2.0) * hyp1f1(-v / 2.0, 0.5, z2);
    let t2 = (2.0 * std::f64::consts::PI).sqrt() * z / gamma_lanczos(-v / 2.0)
        * hyp1f1((1.0 - v) / 2.0, 1.5, z2);
    2.0f64.powf(v / 2.0) * (-z * z / 4.0).exp() * (t1 - t2)
}

pub(super) fn smoothed_power(nu: f64, r: f64, sigma: f64) -> f64 {
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

pub(super) fn pv(energy: f64) -> f64 {
    energy * (energy + 2.0 * MP) / (energy + MP)
}

pub(super) fn mcs_along(medium: Medium, energy_mev: f64) -> (Vec<f64>, Vec<f64>) {
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

pub(super) fn mcs_at(path: &[f64], var: &[f64], depth_mm: f64, r0: f64) -> f64 {
    let z = depth_mm.clamp(0.0, r0);
    lerp(z, path, var).max(0.0).sqrt()
}

pub(super) struct LayerKernel {
    pub(super) energies: Vec<f32>,
    pub(super) zmin: Vec<f32>,
    pub(super) cdf: Vec<f32>,
    pub(super) mcs: Vec<f32>,
    pub(super) energy_dep: Vec<f32>,
    pub(super) mcs_max: Vec<f32>,
    pub(super) nodes: usize,
}

pub(super) fn layer_energies(energy: &[f32]) -> Vec<f32> {
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

pub(super) fn build_kernel(medium: Medium, energy: &[f32], spread_pct: f64) -> LayerKernel {
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

pub(super) fn layer_of(kernel: &LayerKernel, energy: f32) -> usize {
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

pub(super) fn row_at(kernel: &LayerKernel, row: usize, at: f32, mcs: bool) -> f32 {
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

pub(super) fn mass_fraction(kernel: &LayerKernel, energy: f32, z_lo: f32, z_hi: f32) -> f32 {
    let row = layer_of(kernel, energy);
    row_at(kernel, row, z_hi, false) - row_at(kernel, row, z_lo, false)
}

pub(super) fn lateral_mm(kernel: &LayerKernel, energy: f32, z: f32) -> f32 {
    row_at(kernel, layer_of(kernel, energy), z, true).max(0.0)
}

pub(super) fn lerp(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
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

#[cfg(test)]
mod tests {
    use super::super::WATER;
    use super::{bragg_idd, csda_range_mm};

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
}
