// Proton transport ported from MCsquare (Universite catholique de Louvain, Apache-2.0;
// see MCsquare_LICENSE.txt). Mirrors compute_hadron.c, compute_EM_interaction.c,
// compute_nuclear_interaction.c, compute_beam_model.c and compute_range_shifter.c with
// MCsquare's default build: Geant4 stopping powers, PSTAR EM method, voxel-interface
// stepping, float32. Units are MCsquare's: eV and cm.
// The host prepends the Params struct and `const` table offsets (dose_mc.py).
//
// Geometry modes: 0 laterally infinite slab phantom behind a water WET slab, scored on a
// grid; 1 voxel CT (material and density per voxel), scored on the CT grid; the range
// shifter is a transient mode 2 slab in the beam frame.

@group(0) @binding(0) var<storage, read> spot: array<f32>;
@group(0) @binding(1) var<storage, read> F: array<f32>;
@group(0) @binding(2) var<storage, read> I: array<i32>;
// Per voxel (lo, hi) of E / (rho SPR) in quanta.
@group(0) @binding(3) var<storage, read_write> tally: array<atomic<u32>>;
// Pairs (lo, hi): incident, grid, off-grid, leaked, lost, beamline. Then overflow count,
// deepest stack, and the dispatch's count of histories taken.
@group(0) @binding(4) var<storage, read_write> ledger: array<atomic<u32>>;
@group(0) @binding(5) var<uniform> P: Params;
// Material index per voxel, four u8 to a word.
@group(0) @binding(6) var<storage, read> material: array<u32>;
@group(0) @binding(7) var<storage, read> density: array<f32>;
// Per voxel (lo, hi) of E x LET (keV/um), then (lo, hi) of E, in quanta.
@group(0) @binding(8) var<storage, read_write> let_tally: array<atomic<u32>>;
// Per beam: gantry-frame to grid rotation (9, row-major), isocenter (3), nozzle, SMX, SMY distances (cm).
@group(0) @binding(9) var<storage, read> beam: array<f32>;

const PI: f32 = 3.14159265358979;
const R_ELEC: f32 = 2.8179403e-13;
const MC2_ELEC: f32 = 0.5109989e6;
const MC2_PRO: f32 = 938.272046e6;
const UAMU: f32 = 931.46e6;
const UMEV: f32 = 1e6;
const ECUT: f32 = 0.5e6;
const TE_MIN: f32 = 0.05e6;
const D_MAX: f32 = 0.2;
const EPS_MAX: f32 = 0.25;
const CONST_MS: f32 = 18.3;
const CONST_DERIV: f32 = 0.99;
const QUANTUM: f32 = 100.0;
const STACK: i32 = 16;
const EXITS: i32 = 8;
const TWO_PI_RE2_MC2: f32 = 2.0 * PI * R_ELEC * R_ELEC * MC2_ELEC;
const SIMPLE_STRIDE: i32 = 8;
const BDL_STRIDE: i32 = 40;
const BEAM_STRIDE: i32 = 16;
const FLAG_DOSE_TO_WATER: u32 = 1u;
const FLAG_LET: u32 = 2u;
const GEOM_SLAB: i32 = 0;
const GEOM_VOXEL: i32 = 1;
const GEOM_SHIFTER: i32 = 2;

struct Particle {
    x: vec3f,
    d: vec3f,
    T: f32,
    M: f32,
    mass: f32,
    charge: f32,
}

// Where a point sits: region (-1 outside), medium, density and scoring index (-1 none).
struct Where {
    reg: i32,
    m: i32,
    rho: f32,
    idx: i32,
}

var<private> g_stack: array<Particle, 16>;
var<private> g_sp: i32 = 0;
var<private> g_exit: array<Particle, 8>;
var<private> g_nexit: i32 = 0;
var<private> g_deepest: i32 = 0;
var<private> g_overflow: u32 = 0u;
var<private> g_geom: i32 = 0;
var<private> g_beam: i32 = 0;
// Range-shifter slab of the current spot: z range in the beam frame, medium, density.
var<private> g_rs_lo: f32 = 0.0;
var<private> g_rs_hi: f32 = 0.0;
var<private> g_rs_m: i32 = 0;
var<private> g_rs_rho: f32 = 1.0;
// Per-history ledger energies (eV), quantized once at the end.
var<private> g_inc: f32 = 0.0;
var<private> g_grid: f32 = 0.0;
var<private> g_off: f32 = 0.0;
var<private> g_leak: f32 = 0.0;
var<private> g_lost: f32 = 0.0;
var<private> g_line: f32 = 0.0;

// ---------------------------------------------------------------- random numbers
// Counter-based (pcg4d), so a history's stream depends only on (seed, history).
var<private> g_key: vec4u;
var<private> g_ctr: u32 = 0u;
var<private> g_buf: vec4u;
var<private> g_left: i32 = 0;

fn pcg4d(v0: vec4u) -> vec4u {
    var v = v0 * 1664525u + 1013904223u;
    v.x += v.y * v.w; v.y += v.z * v.x; v.z += v.x * v.y; v.w += v.y * v.z;
    v = v ^ (v >> vec4u(16u));
    v.x += v.y * v.w; v.y += v.z * v.x; v.z += v.x * v.y; v.w += v.y * v.z;
    return v;
}

// Uniform on (0, 1]: log() below never sees 0.
fn urand() -> f32 {
    if (g_left == 0) {
        g_buf = pcg4d(vec4u(g_key.x, g_key.y, g_ctr, g_key.z));
        g_ctr += 1u;
        g_left = 4;
    }
    g_left -= 1;
    return f32((g_buf[g_left] >> 8u) + 1u) * (1.0 / 16777216.0);
}

fn gauss() -> f32 {
    let a = urand();
    return sqrt(-2.0 * log(a)) * cos(2.0 * PI * urand());
}

fn quanta(e_ev: f32) -> i32 {
    return i32(floor(e_ev / QUANTUM + urand() - 1e-7));
}

fn add_tally(slot: u32, v: i32) {
    let lo = u32(v);
    var hi = select(0u, 0xFFFFFFFFu, v < 0);
    let old = atomicAdd(&tally[slot], lo);
    if (old + lo < old) { hi += 1u; }
    if (hi != 0u) { atomicAdd(&tally[slot + 1u], hi); }
}

fn add_let(slot: u32, v: i32) {
    let lo = u32(v);
    var hi = select(0u, 0xFFFFFFFFu, v < 0);
    let old = atomicAdd(&let_tally[slot], lo);
    if (old + lo < old) { hi += 1u; }
    if (hi != 0u) { atomicAdd(&let_tally[slot + 1u], hi); }
}

fn add_ledger(slot: u32, v: i32) {
    let lo = u32(v);
    var hi = select(0u, 0xFFFFFFFFu, v < 0);
    let old = atomicAdd(&ledger[slot], lo);
    if (old + lo < old) { hi += 1u; }
    if (hi != 0u) { atomicAdd(&ledger[slot + 1u], hi); }
}

// ---------------------------------------------------------------- tables
fn mbase(m: i32) -> i32 { return M_BASE + m * M_STRIDE; }

fn lerp(x: f32, x1: f32, x2: f32, y1: f32, y2: f32) -> f32 {
    return ((y2 - y1) / (x2 - x1)) * (x - x1) + y1;
}

// MCsquare Binary_Search over F[base .. base+n): largest i with F[i] < v, or -1.
fn bsearch(v: f32, base: i32, n: i32) -> i32 {
    if (n <= 1) { return -1; }
    var i = -1;
    var j = n;
    while (j - i > 1) {
        let k = (i + j) / 2;
        if (F[base + k] < v) { i = k; } else { j = k; }
    }
    return i;
}

// ---------------------------------------------------------------- geometry
fn voxel_index(x: vec3f) -> i32 {
    let c = vec3i(floor((x - vec3f(P.ox, P.oy, P.oz)) / vec3f(P.vx, P.vy, P.vz)));
    if (any(c < vec3i(0)) || any(c >= vec3i(P.nx, P.ny, P.nz))) { return -1; }
    return c.x + P.nx * (c.y + P.ny * c.z);
}

fn locate(x: vec3f) -> Where {
    if (g_geom == GEOM_VOXEL) {
        let idx = voxel_index(x);
        if (idx < 0) { return Where(-1, WATER_MEDIUM, 1.0, -1); }
        let m = i32((material[u32(idx) >> 2u] >> ((u32(idx) & 3u) * 8u)) & 0xFFu);
        return Where(1, m, density[idx], idx);
    }
    if (g_geom == GEOM_SHIFTER) {
        // MCsquare's range shifter is semi-infinite upstream of its exit face.
        return Where(select(-1, 1, x.z > g_rs_lo), g_rs_m, g_rs_rho, -1);
    }
    // Slab phantom from z = 0 to -depth, entrance WET slab (water, not scored) above it.
    if (x.z <= 0.0 && x.z >= -P.depth) {
        return Where(1, P.medium, F[mbase(P.medium) + M_RHO], voxel_index(x));
    }
    if (x.z > 0.0 && x.z <= P.wet) {
        return Where(0, WATER_MEDIUM, F[mbase(WATER_MEDIUM) + M_RHO], -1);
    }
    return Where(-1, WATER_MEDIUM, 1.0, -1);
}

// Energy e (eV) deposited at scoring index idx; w is 1 / (rho SPR).
fn deposit(idx: i32, e_ev: f32, w: f32) {
    if (e_ev == 0.0) { return; }
    if (idx >= 0) {
        let q = quanta(e_ev * w);
        if (q != 0) { add_tally(2u * u32(idx), q); }
        g_grid += e_ev;
    } else if (g_geom == GEOM_SHIFTER) {
        g_line += e_ev;
    } else {
        g_off += e_ev;
    }
}

fn lattice_dist(x: f32, o: f32, v: f32, u: f32) -> f32 {
    if (abs(u) < 1e-12) { return 1e30; }
    let f = (x - o) / v;
    return abs(((floor(f) + select(0.0, 1.0, u > 0.0)) * v + o - x) / u);
}

fn plane_dist(z: f32, w: f32, p: f32) -> f32 {
    let d = (p - z) / w;
    return select(1e30, d, d > 0.0);
}

// Dist_To_Interface: next voxel face, or a slab plane.
fn dist_to_interface(x: vec3f, d: vec3f) -> f32 {
    // SemiInfiniteSlab_step: height above the exit face, whatever the direction.
    if (g_geom == GEOM_SHIFTER) { return max(x.z - g_rs_lo + 1e-4, 0.0); }
    var s = min(lattice_dist(x.x, P.ox, P.vx, d.x), min(lattice_dist(x.y, P.oy, P.vy, d.y), lattice_dist(x.z, P.oz, P.vz, d.z)));
    if (g_geom == GEOM_SLAB && abs(d.z) >= 1e-12) {
        s = min(s, plane_dist(x.z, d.z, 0.0));
        s = min(s, plane_dist(x.z, d.z, -P.depth));
        if (P.wet > 0.0) { s = min(s, plane_dist(x.z, d.z, P.wet)); }
    }
    return select(s + 5e-5, s + 2e-4, s < 1e-3);
}

// Update_Hadron: (E, gamma, beta2, Te_max).
fn kin(T: f32, mass: f32) -> vec4f {
    let mM = mass * MC2_PRO;
    let E = T + mM;
    let g = E / mM;
    let b2 = 1.0 - 1.0 / (g * g);
    let te = (2.0 * MC2_ELEC * mM * mM * (g * g - 1.0)) / (mM * mM + 2.0 * MC2_ELEC * g * mM + MC2_ELEC * MC2_ELEC);
    return vec4f(E, g, b2, te);
}

// Total_Stop_Pow: eV cm^2/g at T/mass on 0.5 MeV bins.
fn stop_pow(mb: i32, T: f32, mass: f32) -> f32 {
    let s = T / mass;
    let i = clamp(i32(floor(s / (0.5 * UMEV))), 0, STOP_BINS - 2);
    return lerp(s, f32(i) * 0.5 * UMEV, f32(i + 1) * 0.5 * UMEV, F[mb + M_STOP + i], F[mb + M_STOP + i + 1]);
}

fn sigma_ion(nel: f32, z2: f32, k: vec4f) -> f32 {
    if (k.w <= TE_MIN) { return 0.0; }
    return TWO_PI_RE2_MC2 * nel * z2
        * ((1.0 / TE_MIN - 1.0 / k.w) - (k.z / k.w) * log(k.w / TE_MIN) + (k.w - TE_MIN) / (2.0 * k.x * k.x))
        / k.z;
}

// total_Nuclear_cross_section, 1/cm.
fn sigma_nuc(mb: i32, T: f32, rho: f32) -> f32 {
    var i = i32(floor(T / UMEV));
    let T1 = f32(i) * UMEV;
    let T2 = f32(i + 1) * UMEV;
    // MCsquare quirk: tables stop at 250 MeV and the index wraps to 0 above 249.
    if (i >= 249) { i = 0; }
    return lerp(T, T1, T2, F[mb + M_NUC + i], F[mb + M_NUC + i + 1]) * rho;
}

// sp is stop_pow at the energy k was computed for.
fn compute_L(nel: f32, rho: f32, z2: f32, sp: f32, k: vec4f) -> f32 {
    var m: f32 = 0.0;
    if (k.w > TE_MIN) {
        m = (TWO_PI_RE2_MC2 * nel * z2 / k.z)
            * (log(k.w / TE_MIN) - (k.w - TE_MIN) * k.z / k.w + (k.w * k.w - TE_MIN * TE_MIN) / (4.0 * k.x * k.x));
    }
    return rho * z2 * sp - m;
}

fn compute_dE2(mb: i32, nel: f32, rho: f32, z2: f32, T: f32, mass: f32, sp: f32, k: vec4f, s: f32) -> f32 {
    let L = compute_L(nel, rho, z2, sp, k);
    let dE1 = L * s;
    let tau1 = T / MC2_PRO;
    let e1 = dE1 / T;
    let C = L * k.z;
    let T2 = T * CONST_DERIV;
    let k2 = kin(T2, mass);
    let C2 = compute_L(nel, rho, z2, stop_pow(mb, T2, mass), k2) * k2.z;
    let b = T * ((C2 - C) / (T2 - T)) / C;
    let a1 = (1.0 + tau1) * (2.0 + tau1);
    return dE1 * (1.0 + e1 / a1 + e1 * e1 * (2.0 + 2.0 * tau1 + tau1 * tau1) / (a1 * a1)
                  - b * e1 * (0.5 + 2.0 * e1 / (3.0 * a1) + (1.0 - b) * e1 / 6.0));
}

// Update_direction, verbatim.
fn rotate(d: vec3f, theta: f32, phi: f32) -> vec3f {
    let cT = cos(theta);
    let sT = sin(theta);
    let cP = cos(phi);
    let sP = sin(phi);
    var o: vec3f;
    if (abs(d.z) < 0.999999) {
        let r = sqrt(1.0 - d.z * d.z);
        o.x = d.x * cT + (sT / r) * (d.x * d.z * cP - d.y * sP);
        o.y = d.y * cT + (sT / r) * (d.y * d.z * cP + d.x * sP);
        o.z = d.z * cT - r * sT * cP;
    } else {
        o.y = sT * sP;
        if (d.z > 0.0) { o.x = sT * cP; o.z = cT; } else { o.x = -sT * cP; o.z = -cT; }
    }
    return o / length(o);
}

// Compute_Ionization_Energy.
fn ionization_energy(k: vec4f) -> f32 {
    if (k.w < TE_MIN) { return 0.0; }
    for (var n = 0; n < 1000; n++) {
        let r = urand();
        let te = (TE_MIN * k.w) / ((1.0 - r) * k.w + r * TE_MIN);
        let g = 1.0 - k.z * (te / k.w) + te * te / (2.0 * k.x * k.x);
        if (urand() <= g) { return te; }
    }
    return 0.0;
}

// ---------------------------------------------------------------- nuclear
fn pp_sigma(e_mev: f32) -> f32 {
    return (0.315 * pow(e_mev, -1.126) + 3.78e-6 * e_mev) / 0.1119;
}

fn push(s: Particle) -> bool {
    if (g_sp >= STACK) {
        g_overflow += 1u;
        return false;
    }
    g_stack[g_sp] = s;
    g_sp += 1;
    g_deepest = max(g_deepest, g_sp);
    return true;
}

// Compute_Elastic_PP. Returns energy deposited locally (unweighted).
fn elastic_pp(p: ptr<function, Particle>) -> f32 {
    let c = 2.0 * urand() - 1.0;
    let dE = (*p).T * (1.0 - c) / 2.0;
    let k = 1.0 - (*p).T / ((*p).T + 2.0 * MC2_PRO);
    let theta = acos((c + 1.0) / max(sqrt((c + 1.0) * (c + 1.0) + k * (1.0 - c * c)), 1e-30));
    let phi = 2.0 * PI * urand();
    let s_theta = acos((1.0 - c) / max(sqrt((1.0 - c) * (1.0 - c) + k * (1.0 - c * c)), 1e-30));
    let s_phi = select(phi + PI, phi - PI, phi > PI);
    let s = Particle((*p).x, rotate((*p).d, s_theta, s_phi), dE, (*p).M, 1.0, 1.0);
    (*p).d = rotate((*p).d, theta, phi);
    (*p).T -= dE;
    if (dE < ECUT) { return dE; }
    if (!push(s)) { return dE; }
    return 0.0;
}

// Compute_Elastic_ICRU for element e.
fn elastic_icru(e: i32, p: ptr<function, Particle>) -> f32 {
    let eb = I_EL + 5 * e;
    let base = I[eb + 1];
    let n = I[eb + 2];
    let E = (*p).T / UMEV;
    let idx = bsearch(E, F_ELE + base, n);
    let E1 = F[F_ELE + base + idx];
    let E2 = F[F_ELE + base + idx + 1];
    let c1 = F_ELC + 36 * (base + idx);
    let c2 = c1 + 36;
    let top = lerp(E, E1, E2, F[c1 + 35], F[c2 + 35]);
    let r = urand() * top;
    var i = 0;
    while (i < 36 && r > lerp(E, E1, E2, F[c1 + i], F[c2 + i])) { i++; }
    // Sequential_Search + 1 is the first bin whose CDF reaches r.
    var cos_cm: f32;
    // Bin edges exactly as MCsquare has them, including its bin 0 and bin 35 ranges.
    if (i == 0) {
        cos_cm = cos(5.0 * PI / 180.0) + urand() * (cos(7.5 * PI / 180.0) - cos(5.0 * PI / 180.0));
    } else if (i >= 35) {
        cos_cm = cos(175.5 * PI / 180.0) + urand() * (cos(PI) - cos(177.5 * PI / 180.0));
    } else {
        let lo = (f32(i + 1) * 5.0 - 2.5) * PI / 180.0;
        let hi = (f32(i + 1) * 5.0 + 2.5) * PI / 180.0;
        cos_cm = cos(lo) + urand() * (cos(hi) - cos(lo));
    }
    let A = F[F_EA + e];
    let den = (*p).T + MC2_PRO + A * UAMU;
    let b2 = ((*p).T * ((*p).T + 2.0 * MC2_PRO)) / (den * den);
    let g2 = 1.0 / (1.0 - b2);
    let ratio = MC2_PRO / (A * UAMU);
    let tau = sqrt(ratio * ratio * (1.0 - b2) + b2);
    let theta = acos((cos_cm + tau) / sqrt((cos_cm + tau) * (cos_cm + tau) + (1.0 - cos_cm * cos_cm) / g2));
    (*p).d = rotate((*p).d, theta, 2.0 * PI * urand());
    let dE = g2 * b2 * A * UAMU * (1.0 - cos_cm);
    (*p).T -= dE;
    return dE;
}

// Cumulative secondary-energy CDF at row i, interpolated between incident energies.
fn cum_d(r0: i32, r1: i32, i: i32, E: f32, Ek0: f32, Ek1: f32) -> f32 {
    return lerp(E, Ek0, Ek1, F[F_SD + r0 + i], F[F_SD + r1 + i]);
}

// Compute_Nuclear_Inelastic_{proton,deuteron,alpha} + recoils; kills the primary.
fn inelastic(e: i32, idx: i32, p: ptr<function, Particle>, pushed: ptr<function, f32>) -> f32 {
    let k0 = I[I_EL + 5 * e + 3] + idx;
    let k1 = k0 + 1;
    let E = (*p).T / UMEV;
    let Ek0 = F[F_INE + k0];
    let Ek1 = F[F_INE + k1];
    var dE: f32 = 0.0;
    var ang = array<f32, 13>(0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 90.0, 110.0, 130.0, 150.0, 180.0);
    for (var s = 0; s < 3; s++) {
        let m0 = F[F_INM + 3 * k0 + s];
        let m1 = F[F_INM + 3 * k1 + s];
        if (m0 == 0.0 || m1 == 0.0) { continue; }
        let Ms = (*p).M * lerp(E, Ek0, Ek1, m0, m1);
        let n = I[I_RN + 3 * k0 + s];
        let r0 = I[I_RS + 3 * k0 + s];
        let r1 = I[I_RS + 3 * k1 + s];
        let total = cum_d(r0, r1, n - 1, E, Ek0, Ek1);
        if (!(total > 0.0)) { continue; }
        let r = urand() * total;
        var lo = -1;
        var hi = n;
        while (hi - lo > 1) {
            let mid = (lo + hi) / 2;
            if (cum_d(r0, r1, mid, E, Ek0, Ek1) < r) { lo = mid; } else { hi = mid; }
        }
        let si = clamp(lo, 0, n - 2);
        let ca = cum_d(r0, r1, si, E, Ek0, Ek1);
        let cb = cum_d(r0, r1, si + 1, E, Ek0, Ek1);
        let ea = F[F_SE + r0 + si];
        let eb = F[F_SE + r0 + si + 1];
        var Ts = lerp(r, ca, cb, ea, eb);
        Ts = (E / Ek0) * UMEV * Ts;
        if (!(Ts >= ECUT)) {
            dE += Ms * Ts / (*p).M;
            continue;
        }
        let phi = 2.0 * PI * urand();
        var dd: array<f32, 13>;
        for (var j = 0; j < 13; j++) {
            let d1 = lerp(E, Ek0, Ek1, F[F_SDD + 13 * (r0 + si) + j], F[F_SDD + 13 * (r1 + si) + j]);
            let d2 = lerp(E, Ek0, Ek1, F[F_SDD + 13 * (r0 + si + 1) + j], F[F_SDD + 13 * (r1 + si + 1) + j]);
            // Matches MCsquare: Ts is in eV here while the row energies are in MeV.
            dd[j] = lerp(Ts, ea, eb, d1, d2);
        }
        let ra = urand() * dd[12];
        var alo = -1;
        var ahi = 13;
        while (ahi - alo > 1) {
            let mid = (alo + ahi) / 2;
            if (dd[mid] < ra) { alo = mid; } else { ahi = mid; }
        }
        let ai = alo + 1;
        var theta: f32;
        if (ai == 0) {
            theta = acos(1.0 + urand() * (cos(5.0 * PI / 180.0) - 1.0));
        } else if (ai == 12) {
            theta = acos(cos(165.0 * PI / 180.0) + urand() * (-1.0 - cos(165.0 * PI / 180.0)));
        } else if (ai == 13) {
            // MCsquare reads ICRU_angles[13] and [14] past the array; every build has zero
            // padding there, which makes this bin isotropic over the forward hemisphere.
            theta = acos(urand());
        } else {
            let c1 = cos((ang[ai - 1] + ang[ai]) * PI / 360.0);
            let c2 = cos((ang[ai] + ang[ai + 1]) * PI / 360.0);
            theta = acos(c1 + urand() * (c2 - c1));
        }
        let mass = select(select(4.0, 2.0, s == 1), 1.0, s == 0);
        let charge = select(1.0, 2.0, s == 2);
        if (push(Particle((*p).x, rotate((*p).d, theta, phi), Ts, Ms, mass, charge))) {
            *pushed += Ms * Ts;
        } else {
            dE += Ms * Ts / (*p).M;
        }
    }
    dE += (*p).T * lerp(E, Ek0, Ek1, F[F_INR + k0], F[F_INR + k1]);
    return dE;
}

// Compute_Nuclear_interaction. Sets alive = false on an inelastic event and
// reports the weighted energy that leaves as neutrals and gammas in lost.
fn nuclear(m: i32, p: ptr<function, Particle>, alive: ptr<function, bool>, lost: ptr<function, f32>) -> f32 {
    let mb = mbase(m);
    let mtype = I[I_M + I_MSTRIDE * m];
    let ncomp = I[I_M + I_MSTRIDE * m + 1];
    let E = (*p).T / UMEV;
    if (mtype == TYPE_ICRU) {
        // MCsquare quirk: no ICRU nuclear interactions below 7 or above 249 MeV.
        if (!(E > 7.0 && E < 249.0)) { return 0.0; }
        let e = I[I_M + I_MSTRIDE * m + 2];
        let r = urand();
        let i = i32(floor(E));
        let total = lerp(E, f32(i), f32(i + 1), F[mb + M_NUC + i], F[mb + M_NUC + i + 1]);
        let base = I[I_EL + 5 * e + 1];
        let ie = bsearch(E, F_ELE + base, I[I_EL + 5 * e + 2]);
        let el = lerp(E, F[F_ELE + base + ie], F[F_ELE + base + ie + 1], F[F_ELS + base + ie], F[F_ELS + base + ie + 1]);
        if (r <= el / total) { return elastic_icru(e, p); }
        let ib = I[I_EL + 5 * e + 3];
        let ii = bsearch(E, F_INE + ib, I[I_EL + 5 * e + 4]);
        var pushed: f32 = 0.0;
        let before = (*p).M * (*p).T;
        let dE = inelastic(e, ii, p, &pushed);
        *lost += before - (*p).M * dE - pushed;
        *alive = false;
        return dE;
    }
    let i = min(i32(floor(E)), NUC_BINS - 2);
    let total = lerp(E, f32(i), f32(i + 1), F[mb + M_NUC + i], F[mb + M_NUC + i + 1]);
    let r = urand() * total;
    var cs: f32 = 0.0;
    for (var c = 0; c < ncomp; c++) {
        let e = I[I_M + I_MSTRIDE * m + 2 + c];
        let f = F[mb + M_FRAC + c];
        if (I[I_EL + 5 * e] == TYPE_PP) {
            // MCsquare quirk: no proton-proton elastic below 10 MeV.
            if (E > 10.0) { cs += f * pp_sigma(E); }
            if (r <= cs) { return elastic_pp(p); }
        } else if (E > 7.0 && E < 249.0) {
            let base = I[I_EL + 5 * e + 1];
            let ie = bsearch(E, F_ELE + base, I[I_EL + 5 * e + 2]);
            cs += f * lerp(E, F[F_ELE + base + ie], F[F_ELE + base + ie + 1], F[F_ELS + base + ie], F[F_ELS + base + ie + 1]);
            if (r <= cs) { return elastic_icru(e, p); }
            let ib = I[I_EL + 5 * e + 3];
            let ii = bsearch(E, F_INE + ib, I[I_EL + 5 * e + 4]);
            cs += f * lerp(E, F[F_INE + ib + ii], F[F_INE + ib + ii + 1], F[F_INS + ib + ii], F[F_INS + ib + ii + 1]);
            if (r <= cs) {
                var pushed: f32 = 0.0;
                let before = (*p).M * (*p).T;
                let dE = inelastic(e, ii, p, &pushed);
                *lost += before - (*p).M * dE - pushed;
                *alive = false;
                return dE;
            }
        }
    }
    return 0.0;
}

// ---------------------------------------------------------------- transport
// A particle below the cut deposits where it is instead of stepping; false then.
fn begin(p: Particle) -> bool {
    if (p.T >= ECUT) { return true; }
    let w = locate(p.x);
    deposit(w.idx, p.M * p.T, 1.0 / w.rho);
    return false;
}

// hadron_step from *wp, where the particle is; false once it stops or leaves, else *wp is
// where it ended up. In the range shifter a particle that leaves through the downstream
// face is kept for the patient.
fn hadron_step(pp: ptr<function, Particle>, wp: ptr<function, Where>) -> bool {
    var p = *pp;
    let here = *wp;
    var alive = true;
    let to_water = g_geom == GEOM_VOXEL && (P.flags & FLAG_DOSE_TO_WATER) != 0u;
    let score_let = g_geom == GEOM_VOXEL && (P.flags & FLAG_LET) != 0u;
    let m = here.m;
    let mb = mbase(m);
    let rho = here.rho;
    let nel = F[mb + M_NEL] * rho;
    let z2 = p.charge * p.charge;
    var k = kin(p.T, p.mass);
    let sp = stop_pow(mb, p.T, p.mass);
    let S = rho * z2 * sp;
    let step_max = min(min(dist_to_interface(p.x, p.d), D_MAX), EPS_MAX * p.T / S);
    let dE_max = step_max * S;

    // Total_Hard_Cross_Section
    var sig = sigma_ion(nel, z2, k) + sigma_nuc(mb, p.T, rho);
    var T2 = p.T - dE_max;
    if (T2 <= 0.0) { T2 = p.T; }
    let k2 = kin(T2, p.mass);
    sig = max(sig, sigma_ion(nel, z2, k2) + sigma_nuc(mb, T2, rho));
    sig = (sig + 1e-10) * 1.017;

    // Dose-to-water: medium over water mass stopping power at the start of the step.
    var spr: f32 = 1.0;
    if (to_water) { spr = sp / stop_pow(mbase(WATER_MEDIUM), p.T, p.mass); }

    var step = -log(urand()) / sig;
    if (step > step_max) { step = step_max; }
    let mean_dE = compute_dE2(mb, nel, rho, z2, p.T, p.mass, sp, k, step);
    let strag = sqrt(TWO_PI_RE2_MC2 * nel * z2 * step * min(TE_MIN, k.w) * (1.0 - 0.5 * k.z) / k.z);
    var dE = mean_dE + strag * gauss();
    let X0 = F[mb + M_X0] / rho;
    let theta = (CONST_MS * UMEV * p.charge / (k.z * k.y * MC2_PRO)) * sqrt(step / X0) * gauss();
    let phi = 2.0 * PI * urand();

    let hinge = here.idx;
    p.x += step * p.d;
    if (p.T - dE <= ECUT) {
        dE = p.T;
        alive = false;
    }
    p.T -= dE;
    deposit(hinge, p.M * dE, 1.0 / (rho * spr));
    if (!alive) { return false; }

    let there = locate(p.x);
    p.d = rotate(p.d, theta, phi);
    if (there.reg < 0) {
        if (g_geom == GEOM_SHIFTER && p.x.z < g_rs_lo) {
            if (g_nexit < EXITS) {
                g_exit[g_nexit] = p;
                g_nexit += 1;
            } else {
                g_overflow += 1u;
                g_line += p.M * p.T;
            }
        } else {
            g_leak += p.M * p.T;
        }
        return false;
    }
    k = kin(p.T, p.mass);

    // get_interaction_type, with the start-of-step material and majorant.
    var itype = 0;
    if (step != step_max) {
        let r = urand();
        let si = sigma_ion(nel, z2, k) / sig;
        if (r <= si) { itype = 1; } else if (r <= si + sigma_nuc(mb, p.T, rho) / sig) { itype = 2; }
    }
    var dEh: f32 = 0.0;
    var lost: f32 = 0.0;
    if (itype == 1) {
        dEh = ionization_energy(k);
        p.T -= dEh;
    } else if (itype == 2) {
        dEh = nuclear(m, &p, &alive, &lost);
        if (lost != 0.0) { g_lost += lost; }
    }
    // LET_Scoring, StopPow method: mean of the linear stopping power before and after the step.
    var s_let: f32 = 0.0;
    if (score_let && itype != 2 && hinge >= 0) { s_let = 0.5 * (S + rho * z2 * stop_pow(mb, p.T, p.mass)); }
    if (alive && p.T <= ECUT) {
        dEh += p.T;
        p.T = 0.0;
        alive = false;
    }
    if (s_let > 0.0) {
        let e_let = p.M * (dE + dEh);
        let qn = quanta(e_let * s_let * 1e-7);
        let qd = quanta(e_let);
        if (qn != 0) { add_let(4u * u32(hinge), qn); }
        if (qd != 0) { add_let(4u * u32(hinge) + 2u, qd); }
    }
    // MCsquare applies no SPR to nuclear deposits.
    deposit(there.idx, p.M * dEh, select(1.0 / (there.rho * spr), 1.0 / there.rho, itype == 2));
    *pp = p;
    *wp = there;
    return alive;
}

// ---------------------------------------------------------------- sources
fn pick_spot(stride: i32, cdf_at: i32) -> i32 {
    let u = urand();
    var lo = -1;
    var hi = P.nspots - 1;
    while (hi - lo > 1) {
        let mid = (lo + hi) / 2;
        if (spot[stride * mid + cdf_at] < u) { lo = mid; } else { hi = mid; }
    }
    return stride * hi;
}

fn rot_x(a: f32, v: vec3f) -> vec3f {
    let c = cos(a);
    let s = sin(a);
    return vec3f(v.x, v.y * c - v.z * s, v.y * s + v.z * c);
}

fn rot_y(a: f32, v: vec3f) -> vec3f {
    let c = cos(a);
    let s = sin(a);
    return vec3f(v.x * c + v.z * s, v.y, -v.x * s + v.z * c);
}

// MCsquare deviates: a Marsaglia-polar normal pair scaled by sigmas, rotated by T.
fn deviates(b: i32) -> vec2f {
    var v1: f32;
    var v2: f32;
    var r: f32;
    loop {
        v1 = 2.0 * urand() - 1.0;
        v2 = 2.0 * urand() - 1.0;
        r = v1 * v1 + v2 * v2;
        if (r <= 1.0 && r > 0.0) { break; }
    }
    let fac = sqrt(-2.0 * log(r) / r);
    let a = v1 * fac * spot[b + 4];
    let c = v2 * fac * spot[b + 5];
    return vec2f(spot[b] * a + spot[b + 1] * c, spot[b + 2] * a + spot[b + 3] * c);
}

// Sample_particle (UPenn double-Gaussian beam model) in the beam frame, z toward the source,
// in the BDL's mm and rad (its phase-space covariance is diagonalized in those units).
// Spot record: X, Y at the isocenter plane (mm), mean E, sigma E (MeV), weight-1 share,
// cdf, beam, pad; per Gaussian (x then y): T (4), sigmas (2); range shifter thickness,
// exit distance from the isocenter (cm), medium, density. Returns cm.
fn sample_bdl(sp: i32) -> Particle {
    let b = BEAM_STRIDE * i32(spot[sp + 6]);
    let dn = beam[b + 12];
    let smx = beam[b + 13];
    let smy = beam[b + 14];
    let X = spot[sp];
    let Y = spot[sp + 1];
    let E = spot[sp + 2] + spot[sp + 3] * gauss();
    let g = select(20, 8, urand() < spot[sp + 4]);
    let xt = deviates(sp + g);
    let yp = deviates(sp + g + 6);
    let r0 = PI + atan(Y / smy);
    let r1 = -atan(X / smx);
    var pos = rot_y(r1, rot_x(r0, vec3f(xt.x, yp.x, 0.0)));
    pos += vec3f(X * (smx - dn) / smx, Y * (smy - dn) / smy, dn);
    let dir = rot_y(r1, rot_x(r0, normalize(vec3f(tan(xt.y), tan(yp.y), 1.0))));
    return Particle(pos * 0.1, dir, max(E, 0.0) * UMEV, 1.0, 1.0, 1.0);
}

// Air between the nozzle (or range shifter) and the CT (Transport_to_CT's ICRU fit), eV/cm.
fn air_stop(T: f32, mass: f32, charge: f32) -> f32 {
    let e = T / (UMEV * mass);
    return charge * (9.6139e-9 * pow(e, 4.0) - 7.0508e-6 * pow(e, 3.0) + 2.0028e-3 * e * e - 2.7615e-1 * e + 2.0082e1) * 1.20479e-3 * 1e6;
}

// BEV_to_CT_frame + Transport_to_CT; false when the particle misses or stops before the CT.
fn into_patient(pp: ptr<function, Particle>, b: i32) -> bool {
    var p = *pp;
    let R = mat3x3f(beam[b], beam[b + 3], beam[b + 6], beam[b + 1], beam[b + 4], beam[b + 7], beam[b + 2], beam[b + 5], beam[b + 8]);
    p.x = R * p.x + vec3f(beam[b + 9], beam[b + 10], beam[b + 11]);
    p.d = normalize(R * p.d);
    g_geom = GEOM_VOXEL;
    let lo = vec3f(P.ox, P.oy, P.oz);
    let hi = lo + vec3f(P.vx * f32(P.nx), P.vy * f32(P.ny), P.vz * f32(P.nz));
    if (voxel_index(p.x) < 0) {
        // Slab method: first entry of the ray into the CT box.
        let inv = 1.0 / p.d;
        let t0 = (lo - p.x) * inv;
        let t1 = (hi - p.x) * inv;
        let tn = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z));
        let tf = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));
        if (!(tf > max(tn, 0.0))) {
            g_leak += p.M * p.T;
            return false;
        }
        let t = max(tn, 0.0) + 1e-4;
        let loss = air_stop(p.T, p.mass, p.charge) * t;
        p.x += t * p.d;
        if (p.T - loss < ECUT) {
            g_line += p.M * p.T;
            return false;
        }
        p.T -= loss;
        g_line += p.M * loss;
    }
    *pp = p;
    return true;
}

// Resets the per-history state and samples history h's primary into *pp; false when it
// never reaches the geometry.
fn open_history(h: u32, pp: ptr<function, Particle>) -> bool {
    g_key = vec4u(h, P.seed, 0x9E3779B9u ^ (P.seed * 0x85EBCA6Bu), 0u);
    g_ctr = 0u;
    g_left = 0;
    g_sp = 0;
    g_nexit = 0;
    g_grid = 0.0;
    g_off = 0.0;
    g_leak = 0.0;
    g_lost = 0.0;
    g_line = 0.0;
    if (P.mode == GEOM_SLAB) {
        let sp = pick_spot(SIMPLE_STRIDE, 6);
        let x = spot[sp] + spot[sp + 2] * gauss();
        let y = spot[sp + 1] + spot[sp + 3] * gauss();
        let e_mev = spot[sp + 4] + spot[sp + 5] * gauss();
        *pp = Particle(vec3f(x, y, P.wet), vec3f(0.0, 0.0, -1.0), max(e_mev, 0.0) * UMEV, 1.0, 1.0, 1.0);
        g_inc = (*pp).T;
        g_geom = GEOM_SLAB;
        return true;
    }
    let sp = pick_spot(BDL_STRIDE, 5);
    var p = sample_bdl(sp);
    g_inc = p.T;
    g_beam = BEAM_STRIDE * i32(spot[sp + 6]);
    let rs_t = spot[sp + 32];
    if (rs_t > 0.0) {
        // Transport_to_RangeShifter, then Simulate_RangeShifter as a slab in the beam frame.
        g_rs_lo = spot[sp + 33];
        g_rs_hi = g_rs_lo + rs_t;
        g_rs_m = i32(spot[sp + 34]);
        g_rs_rho = spot[sp + 35];
        p.x += p.d * ((p.x.z - g_rs_hi) / abs(p.d.z));
        p.x.z = g_rs_hi;
        g_geom = GEOM_SHIFTER;
        *pp = p;
        return true;
    }
    *pp = p;
    return into_patient(pp, g_beam);
}

fn close_history() {
    add_ledger(0u, quanta(g_inc));
    add_ledger(2u, quanta(g_grid));
    add_ledger(4u, quanta(g_off));
    add_ledger(6u, quanta(g_leak));
    add_ledger(8u, quanta(g_lost));
    add_ledger(10u, quanta(g_line));
}

// One step per loop iteration, whichever particle the lane holds: when it is done the lane
// takes the next off its stack, then its range-shifter exits, then a new history off
// ledger[14]. So a subgroup's lanes never idle through each other's secondaries, and the
// resident lanes keep taking histories while later workgroups find none left. The order
// within a history, and so every random number, is that of transport-then-drain recursion.
@compute @workgroup_size(64)
fn main() {
    var p: Particle;
    var w: Where;
    var busy = false;
    var open = false;
    var steps = 0;
    var pops = 0;
    var next_exit = 0;
    loop {
        while (!busy) {
            if (g_sp > 0 && pops < 4096) {
                g_sp -= 1;
                pops += 1;
                p = g_stack[g_sp];
                busy = begin(p);
            } else if (next_exit < g_nexit) {
                p = g_exit[next_exit];
                next_exit += 1;
                pops = 0;
                busy = into_patient(&p, g_beam) && begin(p);
            } else {
                if (open) { close_history(); }
                let n = atomicAdd(&ledger[14], 1u);
                open = n < P.nhist;
                if (!open) { break; }
                pops = 0;
                next_exit = 0;
                busy = open_history(P.hist_base + n, &p) && begin(p);
            }
            steps = 0;
        }
        if (!busy) { break; }
        if (steps == 0) { w = locate(p.x); }
        steps += 1;
        busy = hadron_step(&p, &w) && steps < 100000;
    }
    if (g_overflow != 0u) { atomicAdd(&ledger[12], g_overflow); }
    atomicMax(&ledger[13], u32(g_deepest));
}
