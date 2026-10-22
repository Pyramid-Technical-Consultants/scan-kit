//! Check the GPU Monte Carlo against MCsquare, the reference engine.
//!
//! Slab checks and the patient suite both run the Rust engine the app uses,
//! in process. The patient suite also checks dose-weighted LETd. MCsquare's
//! dose is cached in `validation/goldens/`; `--write-goldens` needs
//! `MCSQUARE_DIR` pointing at an MCsquare build (folder or executable).
//!
//! ```text
//! mc-validate fast                   # tricky small cases
//! mc-validate full                   # every slab case
//! mc-validate full --case water_150_s3
//! mc-validate patient                # CTs, clinical BDL, gamma + DVH + LETd
//! mc-validate fast --write-goldens   # rerun MCsquare, replace the cache
//! ```
//!
//! Ported from `validation/mcsquare_validate.py`.

mod cases;
mod mcsquare;
mod metrics;
mod patient;

use std::path::PathBuf;
use std::time::Instant;

use scan_kit_core::{gamma_index_norm, McJob, SlabRequest};
use scan_kit_io::medium_rho;
use scan_kit_io::npz::{load_npz, npy_f16, npy_f64, npy_i64, npy_unicode, write_npz, Npy};

use cases::{
    patient_cases, Case, PatientCase, PatientTolerances, Tolerances, CROP_FRAC, CROP_MARGIN,
    FAST_TOL, FULL_TOL, GOLDEN_PRIMARIES, LOW_DENSITY, PATIENT_TOL, SEED, SPREAD_PCT, TARGET_FRAC,
};
use mcsquare::{crop_box, crop_vol, golden_dir, mcsquare_exe, mcsquare_version, TempDir};
use metrics::{compare, percentile, summary, Summary};

fn main() {
    let code = run().unwrap_or_else(|err| {
        eprintln!("{err}");
        2
    });
    std::process::exit(code);
}

struct Args {
    suite: String,
    case: Vec<String>,
    histories: Option<f64>,
    write_goldens: bool,
    golden_primaries: f64,
}

fn parse_args() -> Result<Args, String> {
    let mut suite = None;
    let mut args = Args {
        suite: String::new(),
        case: Vec::new(),
        histories: None,
        write_goldens: false,
        golden_primaries: GOLDEN_PRIMARIES,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--case" => {
                args.case.push(it.next().ok_or("--case needs a value")?);
            }
            "--histories" => {
                args.histories = Some(
                    it.next()
                        .ok_or("--histories needs a value")?
                        .parse()
                        .map_err(|_| "bad --histories")?,
                );
            }
            "--write-goldens" => args.write_goldens = true,
            "--golden-primaries" => {
                args.golden_primaries = it
                    .next()
                    .ok_or("--golden-primaries needs a value")?
                    .parse()
                    .map_err(|_| "bad --golden-primaries")?;
            }
            "-h" | "--help" => {
                println!("{}", std::env!("CARGO_PKG_NAME"));
                println!("mc-validate <fast|full|patient> [--case NAME] [--histories N]");
                println!("              [--write-goldens] [--golden-primaries N]");
                std::process::exit(0);
            }
            other if !other.starts_with('-') && suite.is_none() => suite = Some(other.to_string()),
            other => return Err(format!("unknown argument {other}")),
        }
    }
    args.suite = suite.ok_or("usage: mc-validate <fast|full|patient> [options]")?;
    Ok(args)
}

fn run() -> Result<i32, String> {
    let args = parse_args()?;
    match args.suite.as_str() {
        "patient" => main_patient(&args),
        "fast" | "full" => main_slab(&args),
        other => Err(format!("unknown suite {other}")),
    }
}

// ---- slab goldens: MCsquare summaries plus the dose around the beam

fn golden_path(case: &Case) -> PathBuf {
    golden_dir().join(format!("{}.npz", case.name))
}

fn golden_params(case: &Case) -> Vec<(String, Npy)> {
    let spots: Vec<f64> = case.spots.iter().flat_map(|s| s.iter().copied()).collect();
    vec![
        ("medium".into(), npy_unicode(&[case.medium.to_string()])),
        ("sigma".into(), npy_f64(&[], &[f64::from(case.sigma)])),
        ("spots".into(), npy_f64(&[case.spots.len(), 4], &spots)),
        ("wet".into(), npy_i64(&[], &[case.wet as i64])),
        ("lateral".into(), npy_i64(&[], &[case.lateral as i64])),
        ("depth".into(), npy_i64(&[], &[case.depth() as i64])),
        ("spread_pct".into(), npy_f64(&[], &[SPREAD_PCT])),
    ]
}

fn write_golden(
    case: &Case,
    vol: &[f32],
    nz: usize,
    ny: usize,
    nx: usize,
    primaries: u64,
    version: &str,
) -> Result<PathBuf, String> {
    let (lo, hi) = crop_box(vol, nz, ny, nx, CROP_FRAC, CROP_MARGIN);
    let core = crop_vol(vol, ny, nx, lo, hi);
    let scale = core.iter().copied().fold(0.0f32, f32::max) as f64;
    let shape = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
    let crop: Vec<i64> = lo.iter().chain(hi.iter()).map(|&v| v as i64).collect();
    let rho = medium_rho(case.medium)?;
    let s = summary(vol, nz, ny, nx, case, rho);
    let mut entries = golden_params(case);
    entries.extend([
        ("primaries".into(), npy_i64(&[], &[primaries as i64])),
        ("version".into(), npy_unicode(&[version.to_string()])),
        ("crop".into(), npy_i64(&[2, 3], &crop)),
        ("dose_scale".into(), npy_f64(&[], &[scale])),
        (
            "dose".into(),
            npy_f16(
                &shape,
                &core.iter().map(|&v| v / scale as f32).collect::<Vec<_>>(),
            ),
        ),
        ("idd".into(), npy_f64(&[s.idd.len()], &s.idd)),
        ("r80".into(), npy_f64(&[], &[s.r80])),
        ("r90".into(), npy_f64(&[], &[s.r90])),
        (
            "sigma_depths".into(),
            npy_f64(&[s.sigma_depths.len()], &s.sigma_depths),
        ),
        ("sigmas".into(), npy_f64(&[s.sigmas.len()], &s.sigmas)),
        ("energy".into(), npy_f64(&[], &[s.energy])),
        ("centroid".into(), npy_f64(&[s.centroid.len()], &s.centroid)),
    ]);
    std::fs::create_dir_all(golden_dir()).map_err(|err| err.to_string())?;
    let path = golden_path(case);
    write_npz(&path, &entries)?;
    Ok(path)
}

struct Golden {
    /// Cropped dose core, rescaled to Gy per proton.
    dose: Vec<f32>,
    crop: ([usize; 3], [usize; 3]),
    summary: Summary,
}

/// The cached MCsquare result; fails if it was made for a different case setup.
fn load_golden(case: &Case) -> Result<Golden, String> {
    let path = golden_path(case);
    let files = load_npz(&path).map_err(|err| format!("{}: {err}", case.name))?;
    let get = |k: &str| files.get(k).ok_or_else(|| format!("{}: {k}", case.name));
    // Params must match, exactly like np.array_equal on the Python side.
    let medium = get("medium")?.scalar_string()?;
    if medium != case.medium {
        return Err(format!(
            "{}: golden medium is {medium}, case has {}; rerun --write-goldens",
            case.name, case.medium
        ));
    }
    if get("sigma")?.scalar_f64()? != f64::from(case.sigma)
        || get("wet")?.scalar_i64()? != case.wet as i64
        || get("lateral")?.scalar_i64()? != case.lateral as i64
        || get("depth")?.scalar_i64()? != case.depth() as i64
        || get("spread_pct")?.scalar_f64()? != SPREAD_PCT
    {
        return Err(format!(
            "{}: golden params differ; rerun --write-goldens",
            case.name
        ));
    }
    let spots = get("spots")?;
    let want: Vec<f64> = case.spots.iter().flat_map(|s| s.iter().copied()).collect();
    if spots.f64s()? != want || spots.shape != vec![case.spots.len(), 4] {
        return Err(format!(
            "{}: golden spots differ; rerun --write-goldens",
            case.name
        ));
    }
    let scale = get("dose_scale")?.scalar_f64()? as f32;
    let dose: Vec<f32> = get("dose")?.f16s()?.iter().map(|v| v * scale).collect();
    let crop = get("crop")?.i64s()?;
    if crop.len() != 6 || crop.iter().any(|&v| v < 0) {
        return Err(format!("{}: bad crop", case.name));
    }
    let lo = [crop[0] as usize, crop[1] as usize, crop[2] as usize];
    let hi = [crop[3] as usize, crop[4] as usize, crop[5] as usize];
    if lo.iter().zip(&hi).any(|(lo, hi)| lo > hi) {
        return Err(format!("{}: bad crop", case.name));
    }
    let nan = f64::NAN;
    Ok(Golden {
        dose,
        crop: (lo, hi),
        summary: Summary {
            idd: get("idd")?.f64s()?,
            r80: get("r80")?.scalar_f64()?,
            r90: get("r90")?.scalar_f64()?,
            sigma_depths: get("sigma_depths")?.f64s()?,
            sigmas: get("sigmas")?.f64s().unwrap_or_else(|_| vec![nan; 3]),
            energy: get("energy")?.scalar_f64()?,
            centroid: {
                let c = get("centroid")?.f64s()?;
                [
                    c.first().copied().unwrap_or(nan),
                    c.get(1).copied().unwrap_or(nan),
                ]
            },
        },
    })
}

// ---- GPU slab run

fn run_gpu(case: &Case, histories: u64) -> Result<(Vec<f32>, usize, usize, usize), String> {
    let total: f64 = case.spots.iter().map(|s| s[3]).sum();
    let depth = case.depth();
    let lat = case.lateral;
    let request = McJob::Slab(SlabRequest {
        medium: case.medium.to_string(),
        x: case.spots.iter().map(|s| s[0] as f32).collect(),
        y: case.spots.iter().map(|s| s[1] as f32).collect(),
        sx: vec![case.sigma; case.spots.len()],
        sy: vec![case.sigma; case.spots.len()],
        energy: case.spots.iter().map(|s| s[2] as f32).collect(),
        protons: case.spots.iter().map(|s| (s[3] / total) as f32).collect(),
        histories: histories as u32,
        seed: SEED,
        spread_pct: SPREAD_PCT as f32,
        wet_mm: case.wet as f32,
        depth_mm: depth as f32,
        voxel_mm: 1.0,
        origin: [-(lat as f32) / 2.0, -(lat as f32) / 2.0, -(depth as f32)],
        shape: [lat, lat, depth],
    });
    let result = scan_kit_compute::run_mc(&request).map_err(|err| err.to_string())?;
    Ok((result.volume.values, depth, lat, lat))
}

fn gamma_rate(
    reference: &[f32],
    evaluated: &[f32],
    shape: [usize; 3],
    dose_pct: f32,
    spacing: [f32; 3],
    dta_mm: f32,
    cutoff_pct: f32,
) -> f64 {
    let (_map, passed, scored) = gamma_index_norm(
        reference, evaluated, shape, dose_pct, dta_mm, spacing, cutoff_pct, 0.0,
    );
    if scored == 0 {
        f64::NAN
    } else {
        f64::from(passed) / f64::from(scored)
    }
}

fn passes(diff: &metrics::Diff, gamma: f64, tol: &Tolerances) -> bool {
    let ok = diff.idd.abs() <= tol.idd
        && diff.r80.abs() <= tol.r80
        && diff.energy.abs() <= tol.energy
        && diff.centroid.abs() <= tol.centroid;
    // nan (fields) is not a failure
    let ok = ok && (diff.sigma <= tol.sigma || diff.sigma.is_nan());
    ok && gamma >= tol.gamma
}

// ---- suites

fn main_slab(args: &Args) -> Result<i32, String> {
    let (tol, fast_only) = match args.suite.as_str() {
        "fast" => (FAST_TOL, true),
        _ => (FULL_TOL, false),
    };
    let all = cases::cases();
    let suite_cases: Vec<&Case> = all
        .iter()
        .filter(|c| c.fast || !fast_only)
        .filter(|c| args.case.is_empty() || args.case.contains(&c.name))
        .collect();
    let unknown: Vec<&String> = args
        .case
        .iter()
        .filter(|name| !suite_cases.iter().any(|c| &c.name == *name))
        .collect();
    if !unknown.is_empty() {
        let names: Vec<&str> = all
            .iter()
            .filter(|c| c.fast || !fast_only)
            .map(|c| c.name.as_str())
            .collect();
        return Err(format!("unknown cases {unknown:?}; choose from {names:?}"));
    }
    let histories = args.histories.unwrap_or(tol.histories);

    if args.write_goldens {
        let exe = mcsquare_exe()?;
        let version = mcsquare_version(&exe);
        println!("Writing goldens with MCsquare: {version}");
        for case in &suite_cases {
            let n = (args.golden_primaries * case.scale) as u64;
            let t0 = Instant::now();
            let work = TempDir::new("mcsq_")?;
            let vol = mcsquare::run_mcsquare(case, &exe, n, &work.0)?;
            let depth = case.depth();
            write_golden(case, &vol, depth, case.lateral, case.lateral, n, &version)?;
            println!(
                "  {:24} {:>9.0e} primaries  {:4.0} s",
                case.name,
                n as f64,
                t0.elapsed().as_secs_f64()
            );
        }
        println!();
    }

    println!(
        "{} suite: GPU {:.0e} histories per case against cached MCsquare\n",
        args.suite, histories
    );
    println!(
        "{:24} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7}  GPU s",
        "case", "IDD", "dR80", "sigma", "energy", "xy", "gamma"
    );
    let mut failed = Vec::new();
    for case in &suite_cases {
        if !golden_path(case).is_file() {
            println!("{:24} no golden; run with --write-goldens", case.name);
            failed.push(case.name.clone());
            continue;
        }
        let gold = load_golden(case)?;
        let t0 = Instant::now();
        let (vol, nz, ny, nx) = run_gpu(case, (histories * case.scale) as u64)?;
        let seconds = t0.elapsed().as_secs_f64();
        let rho = medium_rho(case.medium)?;
        let diff = compare(&gold.summary, &summary(&vol, nz, ny, nx, case, rho));
        let (lo, hi) = gold.crop;
        let core = crop_vol(&vol, ny, nx, lo, hi);
        let gamma = gamma_rate(
            &gold.dose,
            &core,
            [hi[2] - lo[2], hi[1] - lo[1], hi[0] - lo[0]],
            tol.gamma_pct as f32,
            [1.0; 3],
            1.0,
            10.0,
        );
        let ok = passes(&diff, gamma, &tol);
        if !ok {
            failed.push(case.name.clone());
        }
        println!(
            "{:24} {:6.2}% {:+6.2}mm {:6.2}% {:+6.2}% {:5.2}mm {:6.2}%  {:4.0}  {}",
            case.name,
            100.0 * diff.idd,
            diff.r80,
            100.0 * diff.sigma,
            100.0 * diff.energy,
            diff.centroid,
            100.0 * gamma,
            seconds,
            if ok { "ok" } else { "FAIL" }
        );
    }
    println!(
        "\nTolerances: IDD {:.0}% to R90, |dR80| {} mm, sigma {:.0}% at {:?} R80, \
         energy {:.1}%, centroid {} mm, gamma {:.0}%/1mm (10% cutoff) >= {:.0}%",
        100.0 * tol.idd,
        tol.r80,
        100.0 * tol.sigma,
        cases::SIGMA_DEPTHS,
        100.0 * tol.energy,
        tol.centroid,
        tol.gamma_pct,
        100.0 * tol.gamma
    );
    if !failed.is_empty() {
        println!("FAILED: {}", failed.join(", "));
        return Ok(1);
    }
    println!("All cases pass.");
    Ok(0)
}

fn main_patient(args: &Args) -> Result<i32, String> {
    let tol = PATIENT_TOL;
    let all = patient_cases();
    let suite_cases: Vec<&PatientCase> = all
        .iter()
        .filter(|c| args.case.is_empty() || args.case.contains(&c.name))
        .collect();
    let unknown: Vec<&String> = args
        .case
        .iter()
        .filter(|name| !suite_cases.iter().any(|c| &c.name == *name))
        .collect();
    if !unknown.is_empty() {
        let names: Vec<&str> = all.iter().map(|c| c.name.as_str()).collect();
        return Err(format!(
            "unknown patient cases {unknown:?}; choose from {names:?}"
        ));
    }
    let histories = args.histories.unwrap_or(tol.histories);

    if args.write_goldens {
        let exe = mcsquare_exe()?;
        let version = mcsquare_version(&exe);
        println!("Writing patient goldens with MCsquare: {version}");
        for case in &suite_cases {
            let n = args.golden_primaries as u64;
            let t0 = Instant::now();
            let work = patient::work_dir()?;
            let (dose, let_d) = patient::run_mcsquare_patient(case, &exe, n, &work.0)?;
            let (dims, _) = mcsquare::read_mhd(&work.0.join("CT.mhd"))?;
            let digest = patient::patient_hash(&work.0)?;
            patient::write_patient_golden(
                &golden_dir(),
                case,
                &dose,
                &let_d,
                dims[2],
                dims[1],
                dims[0],
                n,
                &version,
                &digest,
            )?;
            println!(
                "  {:24} {:>9.0e} primaries  {:4.0} s",
                case.name,
                n as f64,
                t0.elapsed().as_secs_f64()
            );
        }
        println!();
    }

    println!("patient suite: GPU {histories:.0e} histories per case against cached MCsquare\n");
    println!(
        "{:24} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7}  GPU s",
        "case", "gamma", "Dmean", "D95", "D2", "LETd", "closure"
    );
    let mut failed = Vec::new();
    for case in &suite_cases {
        match check_patient(case, histories, &tol) {
            Ok((true, line)) => println!("{:24} {}  ok", case.name, line),
            Ok((false, line)) => {
                failed.push(case.name.clone());
                println!("{:24} {}  FAIL", case.name, line);
            }
            Err(err) => {
                failed.push(case.name.clone());
                println!("{:24} {err}  FAIL", case.name);
            }
        }
    }
    println!(
        "\nTolerances: gamma {:.0}%/{:.0}mm (10% cutoff, density >= {}) >= {:.0}%, \
         Dmean {:.0}%, D95 {:.0}%, D2 {:.0}%, dose-weighted LETd {:.0}% over the \
         >= {:.0}% region",
        tol.gamma_pct,
        tol.gamma_mm,
        LOW_DENSITY,
        100.0 * tol.gamma,
        100.0 * tol.dmean,
        100.0 * tol.d95,
        100.0 * tol.d2,
        100.0 * tol.let_tol,
        100.0 * TARGET_FRAC
    );
    if !failed.is_empty() {
        println!("FAILED: {}", failed.join(", "));
        return Ok(1);
    }
    println!("All cases pass.");
    Ok(0)
}

fn dvh_metrics(dose: &[f32], region: &[bool]) -> (f64, f64, f64) {
    let d: Vec<f64> = dose
        .iter()
        .zip(region)
        .filter(|(_, r)| **r)
        .map(|(&v, _)| f64::from(v))
        .collect();
    (
        d.iter().sum::<f64>() / d.len().max(1) as f64,
        percentile(&d, 5.0),
        percentile(&d, 98.0),
    )
}

fn check_patient(
    case: &PatientCase,
    histories: f64,
    tol: &PatientTolerances,
) -> Result<(bool, String), String> {
    let path = golden_dir().join(format!("patient_{}.npz", case.name));
    if !path.is_file() {
        return Ok((false, "no golden; run with --write-goldens".into()));
    }
    let work = patient::work_dir()?;
    patient::build_patient(case, &work.0)?;
    let digest = patient::patient_hash(&work.0)?;
    let gold = patient::load_patient_golden(&path)?;
    if gold.inputs != digest {
        return Ok((
            false,
            "golden made from different inputs; rerun --write-goldens".into(),
        ));
    }
    let t0 = Instant::now();
    let (dose, let_d, density, ledger) = patient::run_gpu_patient(&work.0, histories as u64)?;
    let seconds = t0.elapsed().as_secs_f64();
    let spacing = mcsquare::mhd_spacing(&work.0.join("CT.mhd"))?;
    let (lo, hi) = gold.crop;
    let (dims, _) = mcsquare::read_mhd(&work.0.join("CT.mhd"))?;
    let (nx, ny) = (dims[0], dims[1]);
    let core_dose = crop_vol(&dose, ny, nx, lo, hi);
    let core_let = crop_vol(&let_d, ny, nx, lo, hi);
    let core_density = crop_vol(&density, ny, nx, lo, hi);
    // Compare only voxels with real material — air is not scored.
    let body: Vec<bool> = core_density.iter().map(|&d| d >= LOW_DENSITY).collect();
    let reference: Vec<f32> = gold
        .dose
        .iter()
        .zip(&body)
        .map(|(&d, &b)| if b { d * gold.dose_scale as f32 } else { 0.0 })
        .collect();
    let evaluated: Vec<f32> = core_dose
        .iter()
        .zip(&body)
        .map(|(&d, &b)| if b { d } else { 0.0 })
        .collect();
    let gamma = gamma_rate(
        &reference,
        &evaluated,
        [gold.shape[2], gold.shape[1], gold.shape[0]],
        tol.gamma_pct as f32,
        [spacing[0] as f32, spacing[1] as f32, spacing[2] as f32],
        tol.gamma_mm as f32,
        10.0,
    );
    let ref_max = reference.iter().copied().fold(0.0f32, f32::max);
    let region: Vec<bool> = reference
        .iter()
        .map(|&d| d >= TARGET_FRAC * ref_max)
        .collect();
    let (a_mean, a_d95, a_d2) = dvh_metrics(&reference, &region);
    let (b_mean, b_d95, b_d2) = dvh_metrics(&evaluated, &region);
    let d_mean = b_mean / a_mean - 1.0;
    let d_d95 = b_d95 / a_d95 - 1.0;
    let d_d2 = b_d2 / a_d2 - 1.0;
    // Dose-weighted LETd over the region.
    let mut wsum = 0.0f64;
    let mut let_gpu = 0.0f64;
    let mut let_ref = 0.0f64;
    for i in 0..core_let.len() {
        if region[i] {
            let w = f64::from(reference[i]);
            wsum += w;
            let_gpu += f64::from(core_let[i]) * w;
            let_ref += f64::from(gold.let_d[i]) * w;
        }
    }
    let d_let = let_gpu / wsum / (let_ref / wsum) - 1.0;
    let incident = f64::from(ledger[0]);
    let closure = if incident == 0.0 {
        f64::NAN
    } else {
        (incident - ledger[1..].iter().map(|&v| f64::from(v)).sum::<f64>()) / incident
    };
    let ok = gamma >= tol.gamma
        && d_mean.abs() <= tol.dmean
        && d_d95.abs() <= tol.d95
        && d_d2.abs() <= tol.d2
        && d_let.abs() <= tol.let_tol;
    let line = format!(
        "{:6.2}% {:+6.2}% {:+6.2}% {:+6.2}% {:+6.2}% {:+.0e} {:5.1}",
        100.0 * gamma,
        100.0 * d_mean,
        100.0 * d_d95,
        100.0 * d_d2,
        100.0 * d_let,
        closure,
        seconds
    );
    Ok((ok, line))
}
