//! Pack MCsquare material data into `assets/mc_materials.npz` and copy its
//! scanner calibrations and beam models to `assets/mcsquare`.
//!
//! Reproduces MCsquare's own preprocessing (`data_materials.c`,
//! `data_nuclear.c`, `data_Stop_Pow.c`) for the media the GPU Monte Carlo
//! supports, so the shader reads exactly the numbers MCsquare would. The media
//! are the phantom slabs plus every material an MCsquare scanner calibration or
//! beam-model range shifter refers to. Run from the repo root after
//! `git submodule update --init`:
//!
//! ```text
//! cargo run -p scan-kit-io --bin build-mc-tables
//! ```

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use scan_kit_io::npz;

const N_AVO: f64 = 6.0221415e23;
const INTERP_BINS: usize = 250; // ceil(250 / INTERP_BIN), INTERP_BIN = 1 MeV
const PSTAR_BIN: f64 = 0.5;
const ICRU_ANGLES: [f64; 13] = [
    0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 90.0, 110.0, 130.0, 150.0, 180.0,
];
const SPECIES: [&str; 3] = ["proton", "deuteron", "alpha"];
const TYPE_MIXTURE: i32 = 0;
const TYPE_ICRU: i32 = 1;
const TYPE_PP: i32 = 2;

// scan-kit slab medium key -> MCsquare material name; these keep the first rows.
const SLAB_MEDIA: [(&str, &str); 5] = [
    ("water", "Water"),
    ("pmma", "PMMA"),
    ("polystyrene", "Polystyrene"),
    ("aluminum", "aluminium"),
    ("copper", "copper"),
];

struct Root {
    mcsquare: PathBuf,
    out: PathBuf,
}

impl Root {
    fn find() -> Result<Self, String> {
        // The bin runs from the repo root or any crate dir; walk up to the
        // directory that holds third_party/MCsquare.
        let mut dir = std::env::current_dir().map_err(|e| e.to_string())?;
        loop {
            if dir.join("third_party/MCsquare/Materials").is_dir() {
                return Ok(Self {
                    mcsquare: dir.join("third_party/MCsquare"),
                    out: dir.join("assets/mc_materials.npz"),
                });
            }
            if !dir.pop() {
                return Err(
                    "missing third_party/MCsquare; run: git submodule update --init".into(),
                );
            }
        }
    }
}

fn solid_angle(theta_deg: f64) -> f64 {
    2.0 * std::f64::consts::PI * (1.0 - theta_deg.to_radians().cos())
}

/// MCsquare `Binary_Search`: largest i with values[i] < value, or -1.
fn binary_search(value: f64, values: &[f64]) -> i64 {
    if values.len() <= 1 {
        return -1;
    }
    let (mut i, mut j) = (-1i64, values.len() as i64);
    while j - i > 1 {
        let k = (i + j) / 2;
        if values[k as usize] < value {
            i = k;
        } else {
            j = k;
        }
    }
    i
}

fn lerp(x: f64, x1: f64, x2: f64, y1: f64, y2: f64) -> f64 {
    (y2 - y1) / (x2 - x1) * (x - x1) + y1
}

/// Non-comment tokens per line, `#` stripping trailing comments.
fn tokens(path: &Path) -> Result<Vec<Vec<String>>, String> {
    let text = fs::read_to_string(path)
        .or_else(|_| {
            let bytes = fs::read(path).map_err(|e| format!("{path:?}: {e}"))?;
            Ok::<String, String>(bytes.iter().map(|&b| b as char).collect()) // latin-1
        })
        .map_err(|e| format!("{path:?}: {e}"))?;
    let mut out = Vec::new();
    for line in text.lines() {
        if line.starts_with('#') {
            continue;
        }
        let body = line.split('#').next().unwrap_or("");
        let parts: Vec<String> = body.split_whitespace().map(str::to_string).collect();
        if !parts.is_empty() {
            out.push(parts);
        }
    }
    Ok(out)
}

fn f64v(t: &[String], i: usize) -> f64 {
    t[i].parse().unwrap_or(f64::NAN)
}

/// MCsquare material name -> label, from `Materials/list.dat`.
fn material_ids(materials: &Path) -> Result<BTreeMap<String, i32>, String> {
    let mut out = BTreeMap::new();
    for t in tokens(&materials.join("list.dat"))? {
        if t.len() >= 2 {
            if let Ok(label) = t[0].parse::<i32>() {
                out.insert(t[1].clone(), label);
            }
        }
    }
    Ok(out)
}

/// scan-kit key -> MCsquare name: the slabs, then scanner and range-shifter
/// materials by label (insertion ordered like the Python dict).
fn media(root: &Root, ids: &BTreeMap<String, i32>) -> Result<Vec<(String, String)>, String> {
    let by_id: BTreeMap<i32, &String> = ids.iter().map(|(k, v)| (*v, k)).collect();
    let mut used = BTreeSet::new();
    let scanners = root.mcsquare.join("Scanners");
    if scanners.is_dir() {
        for entry in fs::read_dir(&scanners).map_err(|e| e.to_string())? {
            let f = entry
                .map_err(|e| e.to_string())?
                .path()
                .join("HU_Material_Conversion.txt");
            if f.is_file() {
                for t in tokens(&f)? {
                    if t.len() > 1 {
                        if let Ok(label) = t[1].parse::<i32>() {
                            used.insert(label);
                        }
                    }
                }
            }
        }
    }
    let bdl = root.mcsquare.join("BDL");
    if bdl.is_dir() {
        let mut files: Vec<PathBuf> = fs::read_dir(&bdl)
            .map_err(|e| e.to_string())?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "txt"))
            .collect();
        files.sort();
        for f in files {
            let text = fs::read_to_string(&f).map_err(|e| format!("{f:?}: {e}"))?;
            for line in text.lines() {
                let body = line.split('#').next().unwrap_or("");
                if let Some((key, value)) = body.split_once('=') {
                    if key.trim() == "RS_material" {
                        if let Ok(label) = value.trim().parse::<i32>() {
                            used.insert(label);
                        }
                    }
                }
            }
        }
    }
    let mut out: Vec<(String, String)> = SLAB_MEDIA
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    for label in used {
        let name = by_id
            .get(&label)
            .ok_or_else(|| format!("material label {label} not in Materials/list.dat"))?;
        if !out.iter().any(|(_, v)| v == *name) {
            out.push(((*name).clone(), (*name).clone()));
        }
    }
    Ok(out)
}

#[derive(Default)]
struct Props {
    name: String,
    a: f64,
    density: f64,
    electron_density: f64,
    x0: f64,
    mtype: i32,
    components: Vec<(String, f64)>,
}

fn read_properties(materials: &Path, name: &str) -> Result<Props, String> {
    let mut p = Props {
        name: name.to_string(),
        ..Default::default()
    };
    for t in tokens(&materials.join(name).join("Material_Properties.dat"))? {
        match t[0].as_str() {
            "Atomic_Weight" | "Molecular_Weight" => p.a = f64v(&t, 1),
            "Density" => p.density = f64v(&t, 1),
            "Electron_Density" => p.electron_density = f64v(&t, 1),
            "Radiation_Length" => p.x0 = f64v(&t, 1),
            "Nuclear_Data" => {
                p.mtype = match t[1].as_str() {
                    "Mixture" => TYPE_MIXTURE,
                    "ICRU" => TYPE_ICRU,
                    "proton-proton" => TYPE_PP,
                    other => return Err(format!("{name}: unknown Nuclear_Data {other}")),
                }
            }
            "Mixture_Component" => p.components.push((t[2].clone(), f64v(&t, 3) / 100.0)),
            _ => {}
        }
    }
    Ok(p)
}

fn read_stop_pow(materials: &Path, name: &str) -> Result<Vec<f64>, String> {
    let path = materials.join(name).join("G4_Stop_Pow.dat");
    let text = fs::read_to_string(&path).map_err(|e| format!("{path:?}: {e}"))?;
    let mut rows: Vec<(f64, f64)> = Vec::new();
    for line in text.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 2 {
            rows.push((
                parts[0].parse().unwrap_or(0.0),
                parts[1].parse().unwrap_or(0.0),
            ));
        }
    }
    // MCsquare indexes this table as uniform 0.5 MeV bins from 0.
    for (i, (e, _)) in rows.iter().enumerate() {
        if (e - PSTAR_BIN * i as f64).abs() > 1e-6 {
            return Err(format!(
                "{name}: G4_Stop_Pow.dat is not on a uniform {PSTAR_BIN} MeV grid from 0"
            ));
        }
    }
    Ok(rows.into_iter().map(|(_, s)| s * 1e6).collect()) // eV cm^2 / g
}

fn read_elastic(
    materials: &Path,
    name: &str,
    a: f64,
) -> Result<(Vec<f64>, Vec<f64>, Vec<[f64; 36]>), String> {
    let lines = tokens(&materials.join(name).join("ICRU_Nuclear_elastic.dat"))?;
    let (mut energies, mut sigma, mut cdfs) = (Vec::new(), Vec::new(), Vec::new());
    let mut i = 0;
    while i < lines.len() {
        match lines[i][0].as_str() {
            "Energy" => energies.push(f64v(&lines[i], 1)),
            "Cross_section" => sigma.push(N_AVO * f64v(&lines[i], 1) * 1e-24 / a),
            "Differential_cross_section" => {
                let mut cdf = [0.0f64; 36];
                for k in 0..36 {
                    i += 1;
                    let angle = f64v(&lines[i], 0);
                    let value = f64v(&lines[i], 1);
                    if angle != (k + 1) as f64 * 5.0 {
                        return Err(format!("{name}: elastic angle {angle} at row {k}"));
                    }
                    let w = if k == 0 {
                        solid_angle(7.5) - solid_angle(5.0)
                    } else if k < 35 {
                        solid_angle((k + 1) as f64 * 5.0 + 2.5)
                            - solid_angle((k + 1) as f64 * 5.0 - 2.5)
                    } else {
                        4.0 * std::f64::consts::PI - solid_angle(177.5)
                    };
                    cdf[k] = (if k > 0 { cdf[k - 1] } else { 0.0 }) + value * w;
                }
                let total = cdf[35];
                cdfs.push(cdf.map(|c| c / total));
            }
            _ => {}
        }
        i += 1;
    }
    Ok((energies, sigma, cdfs))
}

fn dd_weights() -> [f64; 13] {
    let a = ICRU_ANGLES;
    let mut w = [0.0f64; 13];
    w[0] = solid_angle(5.0);
    w[12] = 4.0 * std::f64::consts::PI - solid_angle(165.0);
    for j in 1..12 {
        w[j] = solid_angle((a[j] + a[j + 1]) / 2.0) - solid_angle((a[j - 1] + a[j]) / 2.0);
    }
    w
}

/// Per incident energy: E, sigma, mult[3], recoil, and per species rows
/// (secondary energy, cumulative D, cumulative-over-angle DD). Sampling
/// interpolates then accumulates; both are linear so pre-accumulating is exact.
#[derive(Default)]
struct InelasticRecord {
    e: f64,
    sigma: f64,
    mult: [f64; 3],
    recoil: f64,
    rows: [Option<(Vec<f64>, Vec<f64>, Vec<[f64; 13]>)>; 3],
}

fn read_inelastic(materials: &Path, name: &str, a: f64) -> Result<Vec<InelasticRecord>, String> {
    let weights = dd_weights();
    let lines = tokens(&materials.join(name).join("ICRU_Nuclear_inelastic.dat"))?;
    let mut records: Vec<InelasticRecord> = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let t = &lines[i];
        match t[0].as_str() {
            "Energy" => records.push(InelasticRecord {
                e: f64v(t, 1),
                ..Default::default()
            }),
            "Cross_section" => {
                if let Some(r) = records.last_mut() {
                    r.sigma = N_AVO * f64v(t, 1) * 1e-24 / a;
                }
            }
            "Multiplicity" => {
                if let Some(r) = records.last_mut() {
                    if let Some(s) = SPECIES.iter().position(|&x| x == t[1]) {
                        r.mult[s] = f64v(t, 2);
                    }
                }
            }
            "Energy_fraction_recoils" => {
                if let Some(r) = records.last_mut() {
                    r.recoil = f64v(t, 1);
                }
            }
            "Differential_cross_section" => {
                let s = SPECIES.iter().position(|&x| x == t[1]).unwrap_or(0);
                let n: usize = t[2].parse().unwrap_or(0);
                let mut e = Vec::with_capacity(n);
                let mut d = Vec::with_capacity(n);
                let mut dd = Vec::with_capacity(n);
                for r in 0..n {
                    let row = &lines[i + 1 + r];
                    e.push(f64v(row, 0));
                    d.push(f64v(row, 1));
                    let mut accum = [0.0f64; 13];
                    for k in 0..13 {
                        accum[k] = f64v(row, 2 + k) * weights[k];
                    }
                    dd.push(accum);
                }
                i += n;
                // cumulative along each row
                for k in 1..d.len() {
                    d[k] += d[k - 1];
                }
                for row in dd.iter_mut() {
                    for k in 1..13 {
                        row[k] += row[k - 1];
                    }
                }
                if let Some(r) = records.last_mut() {
                    r.rows[s] = Some((e, d, dd));
                }
            }
            _ => {}
        }
        i += 1;
    }
    Ok(records)
}

#[derive(Default)]
struct Element {
    etype: i32,
    a: f64,
    el_e: Vec<f64>,
    el_sigma: Vec<f64>,
    el_cdf: Vec<[f64; 36]>,
    inelastic: Vec<InelasticRecord>,
    in_e: Vec<f64>,
    in_sigma: Vec<f64>,
}

fn pp(e: f64) -> f64 {
    (0.315 * e.powf(-1.126) + 3.78e-6 * e) / 0.1119
}

/// Largest i clamped to [0, len-2] then lerp (ICRU single element).
fn icru_clamped(e_grid: &[f64], s_grid: &[f64], e: f64) -> f64 {
    let k = (binary_search(e, e_grid).max(0) as usize).min(e_grid.len().saturating_sub(2));
    lerp(e, e_grid[k], e_grid[k + 1], s_grid[k], s_grid[k + 1])
}

/// Unclamped binary-search lerp over both elastic and inelastic.
fn icru_mixture(el: &Element, e: f64) -> f64 {
    let mut total = 0.0;
    for (eg, sg) in [(&el.el_e, &el.el_sigma), (&el.in_e, &el.in_sigma)] {
        let k = binary_search(e, eg);
        if k >= 0 && (k as usize) + 1 < eg.len() {
            let k = k as usize;
            total += lerp(e, eg[k], eg[k + 1], sg[k], sg[k + 1]);
        }
    }
    total
}

/// MCsquare `Interp_Nuclear_Cross_section` on 1 MeV bins 0..249.
fn interp_total(props: &Props, elements: &BTreeMap<String, Element>) -> Vec<f64> {
    (0..INTERP_BINS)
        .map(|j| {
            let e = j as f64;
            match props.mtype {
                TYPE_PP => {
                    if e >= 10.0 {
                        pp(e)
                    } else {
                        0.0
                    }
                }
                TYPE_ICRU => {
                    let el = &elements[&props.name];
                    icru_clamped(&el.el_e, &el.el_sigma, e).max(0.0)
                        + icru_clamped(&el.in_e, &el.in_sigma, e)
                }
                _ => {
                    let mut total = 0.0;
                    for (comp, frac) in &props.components {
                        let el = &elements[comp];
                        if el.etype == TYPE_PP {
                            if e >= 10.0 {
                                total += frac * pp(e);
                            }
                        } else if e > 7.0 {
                            total += frac * icru_mixture(el, e);
                        }
                    }
                    total
                }
            }
        })
        .collect()
}

fn copy_tree(src: &Path, dst: &Path) -> Result<(), String> {
    for entry in fs::read_dir(src).map_err(|e| format!("{src:?}: {e}"))? {
        let entry = entry.map_err(|e| e.to_string())?;
        let (s, d) = (entry.path(), dst.join(entry.file_name()));
        if s.is_dir() {
            fs::create_dir_all(&d).map_err(|e| format!("{d:?}: {e}"))?;
            copy_tree(&s, &d)?;
        } else {
            fs::create_dir_all(dst).map_err(|e| format!("{dst:?}: {e}"))?;
            fs::copy(&s, &d).map_err(|e| format!("{s:?} -> {d:?}: {e}"))?;
        }
    }
    Ok(())
}

fn run() -> Result<(), String> {
    let root = Root::find()?;
    let materials = root.mcsquare.join("Materials");
    let ids = material_ids(&materials)?;
    let media = media(&root, &ids)?;

    let media_props: Vec<Props> = media
        .iter()
        .map(|(_, name)| read_properties(&materials, name))
        .collect::<Result<_, _>>()?;

    let mut element_names: Vec<String> = Vec::new();
    for p in &media_props {
        let names: Vec<&String> = if p.mtype == TYPE_MIXTURE {
            p.components.iter().map(|(n, _)| n).collect()
        } else {
            vec![&p.name]
        };
        for n in names {
            if !element_names.contains(n) {
                element_names.push(n.clone());
            }
        }
    }

    let mut elements: BTreeMap<String, Element> = BTreeMap::new();
    for n in &element_names {
        let p = read_properties(&materials, n)?;
        let mut el = Element {
            etype: p.mtype,
            a: p.a,
            ..Default::default()
        };
        if p.mtype == TYPE_ICRU {
            let (e, s, c) = read_elastic(&materials, n, p.a)?;
            el.el_e = e;
            el.el_sigma = s;
            el.el_cdf = c;
            el.inelastic = read_inelastic(&materials, n, p.a)?;
            el.in_e = el.inelastic.iter().map(|r| r.e).collect();
            el.in_sigma = el.inelastic.iter().map(|r| r.sigma).collect();
        }
        elements.insert(n.clone(), el);
    }

    let nm = media.len();
    let ne = element_names.len();
    let mcsquare_ids: Vec<i32> = media.iter().map(|(_, n)| ids[n]).collect();
    let m_props: Vec<f64> = media_props
        .iter()
        .flat_map(|p| [p.density, p.electron_density / p.density, p.x0])
        .collect();
    let m_type: Vec<i32> = media_props.iter().map(|p| p.mtype).collect();
    let width = media_props
        .iter()
        .map(|p| {
            if p.mtype == TYPE_MIXTURE {
                p.components.len()
            } else {
                1
            }
        })
        .max()
        .unwrap_or(1);
    let mut m_comp = vec![-1i32; nm * width];
    let mut m_frac = vec![0.0f64; nm * width];
    for (m, p) in media_props.iter().enumerate() {
        let comps: Vec<(String, f64)> = if p.mtype == TYPE_MIXTURE {
            p.components.clone()
        } else {
            vec![(p.name.clone(), 1.0)]
        };
        for (k, (n, f)) in comps.iter().enumerate() {
            m_comp[m * width + k] = element_names.iter().position(|x| x == n).unwrap() as i32;
            m_frac[m * width + k] = *f;
        }
    }
    let stops: Vec<Vec<f64>> = media
        .iter()
        .map(|(_, n)| read_stop_pow(&materials, n))
        .collect::<Result<_, _>>()?;
    let m_stop: Vec<f64> = stops.iter().flatten().copied().collect();
    let n_stop = stops.first().map(|s| s.len()).unwrap_or(0);
    let m_nuc: Vec<f64> = media_props
        .iter()
        .flat_map(|p| interp_total(p, &elements))
        .collect();

    let e_type: Vec<i32> = element_names.iter().map(|n| elements[n].etype).collect();
    let e_a: Vec<f64> = element_names.iter().map(|n| elements[n].a).collect();

    let (mut el_start, mut el_e, mut el_sigma, mut el_cdf): (
        Vec<i32>,
        Vec<f64>,
        Vec<f64>,
        Vec<[f64; 36]>,
    ) = (vec![0i32], Vec::new(), Vec::new(), Vec::new());
    let (mut in_start, mut in_e, mut in_sigma, mut in_mult, mut in_recoil) =
        (vec![0i32], Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut rows_start, mut rows_n, mut sec_e, mut sec_cd, mut sec_cdd) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut n_rows = 0i32;
    for n in &element_names {
        let el = &elements[n];
        if el.etype == TYPE_ICRU {
            el_e.extend(&el.el_e);
            el_sigma.extend(&el.el_sigma);
            el_cdf.extend(&el.el_cdf);
            for r in &el.inelastic {
                in_e.push(r.e);
                in_sigma.push(r.sigma);
                in_mult.extend(r.mult);
                in_recoil.push(r.recoil);
                let (mut starts, mut counts) = (Vec::new(), Vec::new());
                for rows in r.rows.iter() {
                    match rows {
                        None => {
                            starts.push(n_rows);
                            counts.push(0i32);
                        }
                        Some((e, cd, cdd)) => {
                            starts.push(n_rows);
                            counts.push(e.len() as i32);
                            sec_e.extend(e);
                            sec_cd.extend(cd);
                            sec_cdd.extend(cdd.iter().flatten());
                            n_rows += e.len() as i32;
                        }
                    }
                }
                rows_start.extend(starts);
                rows_n.extend(counts);
            }
        }
        el_start.push(el_e.len() as i32);
        in_start.push(in_e.len() as i32);
    }

    // MCsquare samples rows of incident energy i and i+1 with row count of i.
    for s in 0..3usize {
        for e in 0..ne {
            let (lo, hi) = (in_start[e] as usize, in_start[e + 1] as usize);
            for k in lo..hi.saturating_sub(1) {
                let (a, b) = (rows_n[k * 3 + s], rows_n[(k + 1) * 3 + s]);
                if a != 0 && b != 0 && a != b {
                    return Err(format!(
                        "{}: {} row count changes at {} MeV",
                        element_names[e], SPECIES[s], in_e[k]
                    ));
                }
            }
        }
    }

    let el_cdf_flat: Vec<f64> = el_cdf.iter().flatten().copied().collect();
    let media_keys: Vec<String> = media.iter().map(|(k, _)| k.clone()).collect();
    let mcsquare_names: Vec<String> = media.iter().map(|(_, v)| v.clone()).collect();
    let entries: Vec<(String, npz::Npy)> = vec![
        ("media".into(), npz::npy_unicode(&media_keys)),
        ("mcsquare_names".into(), npz::npy_unicode(&mcsquare_names)),
        (
            "mcsquare_ids".into(),
            npz::npy_i32(&[mcsquare_ids.len()], &mcsquare_ids),
        ),
        ("elements".into(), npz::npy_unicode(&element_names)),
        ("m_props".into(), npz::npy_f64(&[nm, 3], &m_props)),
        ("m_type".into(), npz::npy_i32(&[nm], &m_type)),
        ("m_comp".into(), npz::npy_i32(&[nm, width], &m_comp)),
        ("m_frac".into(), npz::npy_f64(&[nm, width], &m_frac)),
        (
            "m_stop".into(),
            npz::npy_f64(&[stops.len(), n_stop], &m_stop),
        ),
        ("m_nuc".into(), npz::npy_f64(&[nm, INTERP_BINS], &m_nuc)),
        ("e_type".into(), npz::npy_i32(&[ne], &e_type)),
        ("e_A".into(), npz::npy_f64(&[ne], &e_a)),
        (
            "el_start".into(),
            npz::npy_i32(&[el_start.len()], &el_start),
        ),
        ("el_E".into(), npz::npy_f64(&[el_e.len()], &el_e)),
        (
            "el_sigma".into(),
            npz::npy_f64(&[el_sigma.len()], &el_sigma),
        ),
        (
            "el_cdf".into(),
            npz::npy_f64(&[el_cdf.len(), 36], &el_cdf_flat),
        ),
        (
            "in_start".into(),
            npz::npy_i32(&[in_start.len()], &in_start),
        ),
        ("in_E".into(), npz::npy_f64(&[in_e.len()], &in_e)),
        (
            "in_sigma".into(),
            npz::npy_f64(&[in_sigma.len()], &in_sigma),
        ),
        (
            "in_mult".into(),
            npz::npy_f64(&[in_mult.len() / 3, 3], &in_mult),
        ),
        (
            "in_recoil".into(),
            npz::npy_f64(&[in_recoil.len()], &in_recoil),
        ),
        (
            "in_rows_start".into(),
            npz::npy_i32(&[rows_start.len() / 3, 3], &rows_start),
        ),
        (
            "in_rows_n".into(),
            npz::npy_i32(&[rows_n.len() / 3, 3], &rows_n),
        ),
        ("sec_E".into(), npz::npy_f64(&[sec_e.len()], &sec_e)),
        ("sec_cD".into(), npz::npy_f64(&[sec_cd.len()], &sec_cd)),
        (
            "sec_cDD".into(),
            npz::npy_f64(&[sec_cdd.len() / 13, 13], &sec_cdd),
        ),
    ];

    if let Some(parent) = root.out.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    npz::write_npz(&root.out, &entries)?;
    println!(
        "wrote {}: media={media_keys:?} elements={element_names:?}",
        root.out.display()
    );
    for data in ["Scanners", "BDL"] {
        copy_tree(
            &root.mcsquare.join(data),
            &root.out.parent().unwrap().join("mcsquare").join(data),
        )?;
    }
    println!("copied the MCsquare scanner calibrations and beam models to assets/mcsquare");
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
