//! MCsquare driver: input files, subprocess, `.mhd` IO — ported from
//! `mcsquare_validate.py`.

use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

use scan_kit_io::medium_rho;

use crate::cases::{Case, EV_PER_G_TO_GY, SPREAD_PCT};
use crate::metrics::centroid_mm;

/// The repo root: `crates/scan-kit-compute` up two levels.
pub fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap_or_else(|_| Path::new(env!("CARGO_MANIFEST_DIR")).join("../.."))
}

pub fn golden_dir() -> PathBuf {
    // MC_VALIDATE_GOLDEN_DIR redirects the cache — for testing --write-goldens
    // without touching the committed files.
    if let Some(dir) = std::env::var_os("MC_VALIDATE_GOLDEN_DIR") {
        return PathBuf::from(dir);
    }
    repo_root().join("validation/goldens")
}

pub fn materials_dir() -> PathBuf {
    repo_root().join("third_party/MCsquare/Materials")
}

/// A unique work dir under the system temp, removed on drop.
pub struct TempDir(pub PathBuf);

impl TempDir {
    pub fn new(prefix: &str) -> Result<TempDir, String> {
        let base = std::env::temp_dir().join(format!(
            "{prefix}_{}_{:x}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(0)
        ));
        fs::create_dir_all(&base).map_err(|err| format!("{base:?}: {err}"))?;
        Ok(TempDir(base))
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        // MC_VALIDATE_KEEP_WORK keeps the MCsquare input dirs for debugging.
        if std::env::var_os("MC_VALIDATE_KEEP_WORK").is_none() {
            let _ = fs::remove_dir_all(&self.0);
        } else {
            eprintln!("kept {:?}", self.0);
        }
    }
}

pub fn mcsquare_exe() -> Result<PathBuf, String> {
    let raw = std::env::var("MCSQUARE_DIR")
        .map_err(|_| "set MCSQUARE_DIR to an MCsquare build folder or executable".to_string())?;
    let path = Path::new(&raw);
    if path.is_file() {
        return Ok(path.to_path_buf());
    }
    let platform = if cfg!(windows) {
        "MCsquare_win.exe"
    } else if cfg!(target_os = "macos") {
        "MCsquare_mac"
    } else {
        "MCsquare_linux"
    };
    for name in [platform, "MCsquare.exe", "MCsquare"] {
        let candidate = path.join(name);
        if candidate.is_file() {
            return Ok(candidate);
        }
    }
    Err(format!("no MCsquare executable in {path:?}"))
}

pub fn mcsquare_version(exe: &Path) -> String {
    let out = Command::new(exe)
        .arg("-v")
        .env("MCsquare_Materials_Dir", materials_dir())
        .output();
    match out {
        Ok(out) => String::from_utf8_lossy(&out.stdout)
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>()
            .join(" | "),
        Err(err) => format!("version check failed: {err}"),
    }
}

fn material_label(name: &str) -> Result<i64, String> {
    let list = fs::read_to_string(materials_dir().join("list.dat"))
        .map_err(|err| format!("list.dat: {err}"))?;
    for line in list.lines() {
        let body = line.split('#').next().unwrap_or("");
        let parts: Vec<&str> = body.split_whitespace().collect();
        if parts.len() >= 2 && parts[1] == name {
            return parts[0]
                .parse()
                .map_err(|_| format!("bad material id in {line:?}"));
        }
    }
    Err(format!("material {name} not in list.dat"))
}

fn mcsquare_name(medium: &str) -> Result<String, String> {
    let tables = scan_kit_io::mc_tables();
    tables
        .media
        .iter()
        .zip(&tables.mcsquare_names)
        .find(|(m, _)| m.as_str() == medium)
        .map(|(_, n)| n.clone())
        .ok_or_else(|| format!("medium {medium}"))
}

/// Write CT.mhd/.raw, HU tables, BDL.txt, Plan.txt, config.txt into `work`.
pub fn write_inputs(case: &Case, work: &Path, primaries: u64) -> Result<(), String> {
    let lat = case.lateral;
    let depth = case.depth();
    let wet = case.wet;
    let total = depth + wet;
    // MHD axes: 0 = plan X, 1 = beam (enters at the top face, travels -1), 2 = plan Y.
    let mut hu = vec![0.0f32; lat * total * lat];
    for i_y in 0..lat {
        for j in depth..total {
            for i_x in 0..lat {
                hu[i_y * total * lat + j * lat + i_x] = -1000.0;
            }
        }
    }
    let raw = bytemuck_f32(&hu);
    fs::write(work.join("CT.raw"), raw).map_err(|err| err.to_string())?;
    write_text(
        &work.join("CT.mhd"),
        &format!(
            "ObjectType = Image\nNDims = 3\nDimSize = {lat} {total} {lat}\n\
             ElementSpacing = 1 1 1\nOffset = 0 0 0\nElementType = MET_FLOAT\n\
             ElementByteOrderMSB = False\nElementDataFile = CT.raw\n"
        ),
    )
    .map_err(|err| err.to_string())?;
    let rho_water = medium_rho("water")?;
    let rho = medium_rho(case.medium)?;
    write_text(
        &work.join("HU_Density.txt"),
        &format!("-1000 {rho_water}\n0 {rho}\n"),
    )
    .map_err(|err| err.to_string())?;
    write_text(
        &work.join("HU_Material.txt"),
        &format!(
            "-1000 {}\n-1 {}\n",
            material_label(&mcsquare_name("water")?)?,
            material_label(&mcsquare_name(case.medium)?)?,
        ),
    )
    .map_err(|err| err.to_string())?;
    // MCsquare's 2x2 eigen-solve divides 0/0 at zero divergence or correlation;
    // 1e-5 rad at 0.1 moves sigma by < 0.01 % over 40 cm.
    let s = case.sigma;
    let beam = format!("{s} 1e-05 0.1 {s} 1e-05 0.1");
    let row = format!("{SPREAD_PCT} 1.0 1.0 {beam} 0.0 {beam}");
    write_text(
        &work.join("BDL.txt"),
        &format!(
            "--UPenn beam model (double gaussian)--\n# scan-kit validation: parallel mono-Gaussian beam\n\n\
             Nozzle exit to Isocenter distance\n0.01\n\n\
             SMX to Isocenter distance\n10000000.0\n\nSMY to Isocenter distance\n10000000.0\n\n\
             Beam parameters\n2 energies\n\n\
             NominalEnergy MeanEnergy EnergySpread ProtonsMU Weight1 SpotSize1x Divergence1x Correlation1x \
             SpotSize1y Divergence1y Correlation1y Weight2 SpotSize2x Divergence2x Correlation2x \
             SpotSize2y Divergence2y Correlation2y\n\
             1.0 1.0 {row}\n400.0 400.0 {row}\n"
        ),
    )
    .map_err(|err| err.to_string())?;
    let mut layers: BTreeMap<String, (f64, Vec<(f64, f64, f64)>)> = BTreeMap::new();
    for &[x, y, e, w] in &case.spots {
        layers
            .entry(format!("{e:e}"))
            .or_insert((e, Vec::new()))
            .1
            .push((x, y, w));
    }
    let weight: f64 = case.spots.iter().map(|s| s[3]).sum();
    let mut lines = vec![
        "#TREATMENT-PLAN-DESCRIPTION".to_string(),
        "#PlanName".into(),
        case.name.clone(),
        "#NumberOfFractions".into(),
        "1".into(),
        "##FractionID".into(),
        "1".into(),
        "##NumberOfFields".into(),
        "1".into(),
        "###FieldsID".into(),
        "1".into(),
        "#TotalMetersetWeightOfAllFields".into(),
        pyfloat(weight),
        "".into(),
        "#FIELD-DESCRIPTION".into(),
        "###FieldID".into(),
        "1".into(),
        "###FinalCumulativeMeterSetWeight".into(),
        pyfloat(weight),
        "###GantryAngle".into(),
        "0".into(),
        "###PatientSupportAngle".into(),
        "0".into(),
        // Isocenter on the entry face; the 0.01 mm nozzle gap keeps air loss negligible.
        "###IsocenterPosition".into(),
        format!(
            "{} {total} {}",
            pyfloat(lat as f64 / 2.0),
            pyfloat(lat as f64 / 2.0)
        ),
        "###NumberOfControlPoints".into(),
        format!("{}", layers.len()),
        "".into(),
        "#SPOTS-DESCRIPTION".into(),
    ];
    let mut cum = 0.0;
    for (k, (_, (e, spots))) in layers.iter().enumerate() {
        cum += spots.iter().map(|s| s.2).sum::<f64>();
        lines.extend([
            "####ControlPointIndex".into(),
            format!("{}", k + 1),
            "####SpotTunnedID".into(),
            "1".into(),
            "####CumulativeMetersetWeight".into(),
            pyfloat(cum),
            "####Energy (MeV)".into(),
            pyfloat(*e),
            "####NbOfScannedSpots".into(),
            format!("{}", spots.len()),
            "####X Y Weight".into(),
        ]);
        for &(x, y, w) in spots {
            lines.push(format!("{} {} {}", pyfloat(x), pyfloat(y), pyfloat(w)));
        }
    }
    write_text(&work.join("Plan.txt"), &(lines.join("\n") + "\n"))?;
    write_text(
        &work.join("config.txt"),
        &format!(
            "Num_Threads 0\nRNG_Seed 0\nNum_Primaries {primaries}\n\
             CT_File CT.mhd\nHU_Density_Conversion_File HU_Density.txt\n\
             HU_Material_Conversion_File HU_Material.txt\n\
             BDL_Machine_Parameter_File BDL.txt\nBDL_Plan_File Plan.txt\n\
             Output_Directory Outputs\nDose_MHD_Output True\nCompute_stat_uncertainty True\n"
        ),
    )
    .map_err(|err| err.to_string())?;
    fs::create_dir_all(work.join("Outputs")).map_err(|err| err.to_string())?;
    Ok(())
}

/// An `.mhd` volume: dims in file (x, y, z) order plus x-fastest data.
pub fn read_mhd(path: &Path) -> Result<([usize; 3], Vec<f64>), String> {
    let text = fs::read_to_string(path).map_err(|err| format!("{path:?}: {err}"))?;
    let mut head = BTreeMap::new();
    for line in text.lines() {
        if let Some((k, v)) = line.split_once('=') {
            head.insert(k.trim().to_string(), v.trim().to_string());
        }
    }
    let dims: Vec<usize> = head
        .get("DimSize")
        .ok_or("DimSize")?
        .split_whitespace()
        .map(|v| v.parse().map_err(|_| "bad DimSize".to_string()))
        .collect::<Result<_, _>>()?;
    if dims.len() != 3 {
        return Err(format!("{path:?}: DimSize {dims:?}"));
    }
    let raw_path = path
        .parent()
        .unwrap_or(path)
        .join(head.get("ElementDataFile").ok_or("ElementDataFile")?);
    let raw = fs::read(&raw_path).map_err(|err| format!("{raw_path:?}: {err}"))?;
    let data = match head.get("ElementType").map(String::as_str) {
        Some("MET_FLOAT") => raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f64::from(f32::from_le_bytes(*c)))
            .collect(),
        Some("MET_DOUBLE") => raw
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| f64::from_le_bytes(*c))
            .collect(),
        other => return Err(format!("{path:?}: ElementType {other:?}")),
    };
    Ok(([dims[0], dims[1], dims[2]], data))
}

/// ElementSpacing (x, y, z) in mm.
pub fn mhd_spacing(path: &Path) -> Result<[f64; 3], String> {
    for line in fs::read_to_string(path)
        .map_err(|err| format!("{path:?}: {err}"))?
        .lines()
    {
        if let Some((k, v)) = line.split_once('=') {
            if k.trim() == "ElementSpacing" {
                let parts: Vec<f64> = v
                    .split_whitespace()
                    .map(|s| s.parse().unwrap_or(f64::NAN))
                    .collect();
                if parts.len() == 3 {
                    return Ok([parts[0], parts[1], parts[2]]);
                }
            }
        }
    }
    Err(format!("{path:?}: no ElementSpacing"))
}

/// `Path.write_text` writes `os.linesep` on Windows — and the patient
/// goldens' input hash covers those bytes. Match it exactly.
pub fn write_text(path: &Path, content: &str) -> Result<(), String> {
    let eol = if cfg!(windows) { "\r\n" } else { "\n" };
    fs::write(path, content.replace('\n', eol)).map_err(|err| format!("{path:?}: {err}"))
}

/// Python `repr` of a float: integral values keep their `.0`. Plan.txt feeds
/// the patient goldens' input hash, so the bytes must match what
/// `mcsquare_validate.py` wrote.
pub fn pyfloat(v: f64) -> String {
    if v.is_finite() && v == v.trunc() && v.abs() < 1e16 {
        format!("{v:.1}")
    } else {
        format!("{v}")
    }
}

/// f32 slice → little-endian bytes.
pub fn bytemuck_f32(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// MCsquare dose on our grid: `(nz, ny, nx)` Gy per proton, z index 0 deepest.
pub fn run_mcsquare(
    case: &Case,
    exe: &Path,
    primaries: u64,
    work: &Path,
) -> Result<Vec<f32>, String> {
    write_inputs(case, work, primaries)?;
    let done = Command::new(exe)
        .arg("config.txt")
        .current_dir(work)
        .env("MCsquare_Materials_Dir", materials_dir())
        .output()
        .map_err(|err| err.to_string())?;
    let dose_path = work.join("Outputs/Dose.mhd");
    if !done.status.success() || !dose_path.is_file() {
        let stdout = String::from_utf8_lossy(&done.stdout);
        let stderr = String::from_utf8_lossy(&done.stderr);
        return Err(format!(
            "MCsquare failed on {}:\n{}\n{}",
            case.name,
            &stdout[stdout.len().saturating_sub(2000)..],
            &stderr[stderr.len().saturating_sub(2000)..]
        ));
    }
    let lat = case.lateral;
    let total = case.depth() + case.wet;
    let (dims, dose) = read_mhd(&dose_path)?;
    if dims != [lat, total, lat] {
        return Err(format!(
            "Dose.mhd dims {dims:?}, expected [{lat} {total} {lat}]"
        ));
    }
    // [plan Y, beam, plan X] -> (z deepest-first, y, x mirrored), Gy per proton.
    let depth = case.depth();
    let mut vol = vec![0.0f32; depth * lat * lat];
    for z in 0..depth {
        for y in 0..lat {
            for x in 0..lat {
                let src = y * total * lat + z * lat + (lat - 1 - x);
                vol[(z * lat + y) * lat + x] = (dose[src] * EV_PER_G_TO_GY) as f32;
            }
        }
    }
    // The beam must land where the plan puts it; each spot deposits roughly
    // protons x energy.
    let mut wx = 0.0;
    let mut wy = 0.0;
    let mut wsum = 0.0;
    for &[x, y, e, w] in &case.spots {
        wx += x * w * e;
        wy += y * w * e;
        wsum += w * e;
    }
    let expect = [wx / wsum, wy / wsum];
    let got = centroid_mm(&vol, depth, lat, lat);
    if (got[0] - expect[0]).abs() > 1.0 || (got[1] - expect[1]).abs() > 1.0 {
        return Err(format!(
            "{}: MCsquare beam centroid {got:?} mm, expected {expect:?}",
            case.name
        ));
    }
    Ok(vol)
}

/// Crop `vol` (z, y, x) to the box holding dose >= `frac` of max, plus `margin`.
pub fn crop_box(
    vol: &[f32],
    nz: usize,
    ny: usize,
    nx: usize,
    frac: f32,
    margin: usize,
) -> ([usize; 3], [usize; 3]) {
    let max = vol.iter().copied().fold(0.0f32, f32::max);
    let level = frac * max;
    let mut lo = [nz, ny, nx];
    let mut hi = [0usize; 3];
    let plane = ny * nx;
    for (i, &v) in vol.iter().enumerate() {
        if v >= level {
            let z = i / plane;
            let y = (i % plane) / nx;
            let x = i % nx;
            lo[0] = lo[0].min(z);
            lo[1] = lo[1].min(y);
            lo[2] = lo[2].min(x);
            hi[0] = hi[0].max(z + 1);
            hi[1] = hi[1].max(y + 1);
            hi[2] = hi[2].max(x + 1);
        }
    }
    for a in 0..3 {
        lo[a] = lo[a].saturating_sub(margin);
        // A volume with no voxel >= level (all NaN) leaves hi at 0 — keep
        // the box empty rather than letting hi - lo underflow downstream.
        hi[a] = (hi[a] + margin).min([nz, ny, nx][a]).max(lo[a]);
    }
    (lo, hi)
}

/// Copy the `lo..hi` box out of a `(nz, ny, nx)` volume.
pub fn crop_vol(vol: &[f32], ny: usize, nx: usize, lo: [usize; 3], hi: [usize; 3]) -> Vec<f32> {
    let plane = ny * nx;
    if plane == 0 {
        return Vec::new();
    }
    let nz = vol.len() / plane;
    // A box from a golden can disagree with this run's volume — clamp
    // rather than index out of bounds.
    let lo = [lo[0].min(nz), lo[1].min(ny), lo[2].min(nx)];
    let hi = [
        hi[0].clamp(lo[0], nz),
        hi[1].clamp(lo[1], ny),
        hi[2].clamp(lo[2], nx),
    ];
    let mut out = Vec::with_capacity((hi[0] - lo[0]) * (hi[1] - lo[1]) * (hi[2] - lo[2]));
    for z in lo[0]..hi[0] {
        for y in lo[1]..hi[1] {
            out.extend_from_slice(&vol[z * plane + y * nx + lo[2]..z * plane + y * nx + hi[2]]);
        }
    }
    out
}

/// Write an `.mhd` + `.raw` pair for a `(nz, ny, nx)` f32 volume (patient CTs).
pub fn write_mhd(
    path: &Path,
    vol: &[f32],
    nz: usize,
    ny: usize,
    nx: usize,
    spacing: [f64; 3],
) -> Result<(), String> {
    let raw_path = path.with_extension("raw");
    let mut file = fs::File::create(&raw_path).map_err(|err| err.to_string())?;
    file.write_all(&bytemuck_f32(vol))
        .map_err(|err| err.to_string())?;
    write_text(
        path,
        &format!(
            "ObjectType = Image\nNDims = 3\nDimSize = {nx} {ny} {nz}\n\
             ElementSpacing = {} {} {}\nOffset = 0 0 0\nElementType = MET_FLOAT\n\
             ElementByteOrderMSB = False\nElementDataFile = {}\n",
            spacing[0],
            spacing[1],
            spacing[2],
            raw_path.file_name().unwrap_or_default().to_string_lossy()
        ),
    )
    .map_err(|err| err.to_string())?;
    Ok(())
}
