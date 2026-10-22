//! Patient suite: voxel CTs through the clinical BDL, oblique beams and a
//! range shifter. Both engines read the same CT.mhd and Plan.txt; dose is
//! dose-to-water (OnlineSPR) in Gy. Ported from `mcsquare_validate.py`.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use scan_kit_core::{McJob, PatientRequest};
use scan_kit_dicom::gantry_to_patient;
use scan_kit_io::npz::{load_npz, npy_f16, npy_f64, npy_i64, npy_unicode, write_npz};
use scan_kit_io::{calibration::CtCalibration, BeamModel};

use crate::cases::{PatientCase, CROP_FRAC, CROP_MARGIN, EV_PER_G_TO_GY, SEED};
use crate::mcsquare::{
    crop_box, crop_vol, materials_dir, mhd_spacing, pyfloat, read_mhd, repo_root, write_mhd,
    write_text, TempDir,
};

pub fn patient_bdl() -> PathBuf {
    repo_root().join("third_party/MCsquare/BDL/BDL_default_DN_RangeShifter.txt")
}

pub fn scanner_dir() -> PathBuf {
    repo_root().join("third_party/MCsquare/Scanners/default")
}

pub fn sample_dir() -> PathBuf {
    repo_root().join("third_party/MCsquare/Sample_input_data")
}

/// One field of an MCsquare Plan.txt.
#[derive(Clone)]
pub struct PlanField {
    pub gantry: f64,
    pub couch: f64,
    pub iso: [f64; 3],
    /// Range-shifter id ("" for none).
    pub rs: String,
    /// (energy, rs_wet applied, rs_dist, spots [(x, y, MU)]).
    pub layers: Vec<(f64, f64, f64, Vec<[f64; 3]>)>,
}

/// CT.mhd and Plan.txt for `case` in `work`.
pub fn build_patient(case: &PatientCase, work: &Path) -> Result<(), String> {
    if case.sample {
        for name in ["CT.mhd", "CT.raw"] {
            fs::copy(sample_dir().join(name), work.join(name)).map_err(|err| err.to_string())?;
        }
        fs::copy(sample_dir().join("PlanPencil.txt"), work.join("Plan.txt"))
            .map_err(|err| err.to_string())?;
        return Ok(());
    }
    let n = 80;
    let spacing = 2.0;
    let mut hu = vec![-1000.0f32; n * n * n];
    for z in 5..75 {
        for y in 5..75 {
            for x in 5..75 {
                hu[(z * n + y) * n + x] = 0.0;
            }
        }
    }
    if let Some(insert) = case.insert_hu {
        // Gantry 0 travels -y through the CT; the slab covers half the field at
        // 40-60 mm depth.
        for z in 5..75 {
            for y in 45..55 {
                for x in 5..40 {
                    hu[(z * n + y) * n + x] = insert;
                }
            }
        }
    }
    write_mhd(&work.join("CT.mhd"), &hu, n, n, n, [spacing; 3])?;
    let mut spots = Vec::new();
    for yi in 0..5 {
        for xi in 0..5 {
            spots.push([-12.0 + xi as f64 * 6.0, -12.0 + yi as f64 * 6.0, 1.0]);
        }
    }
    let iso = [n as f64 * spacing / 2.0; 3];
    write_plan(
        &work.join("Plan.txt"),
        "Plan",
        &[PlanField {
            gantry: case.gantry,
            couch: case.couch,
            iso,
            rs: if case.rs_wet > 0.0 {
                "RS_Block".into()
            } else {
                String::new()
            },
            layers: case
                .energies
                .iter()
                .map(|&e| (e, case.rs_wet, 300.0, spots.clone()))
                .collect(),
        }],
    )
}

/// MCsquare Plan.txt.
pub fn write_plan(path: &Path, name: &str, fields: &[PlanField]) -> Result<(), String> {
    let total: f64 = fields
        .iter()
        .flat_map(|f| f.layers.iter())
        .flat_map(|(_, _, _, spots)| spots.iter())
        .map(|s| s[2])
        .sum();
    let mut lines = vec![
        "#TREATMENT-PLAN-DESCRIPTION".to_string(),
        "#PlanName".into(),
        name.to_string(),
        "#NumberOfFractions".into(),
        "1".into(),
        "##FractionID".into(),
        "1".into(),
        "##NumberOfFields".into(),
        format!("{}", fields.len()),
    ];
    for k in 0..fields.len() {
        lines.extend(["###FieldsID".to_string(), format!("{}", k + 1)]);
    }
    lines.extend([
        "#TotalMetersetWeightOfAllFields".into(),
        pyfloat(total),
        "".into(),
    ]);
    for (k, f) in fields.iter().enumerate() {
        let field_total: f64 = f
            .layers
            .iter()
            .flat_map(|(_, _, _, spots)| spots.iter())
            .map(|s| s[2])
            .sum();
        lines.extend([
            "#FIELD-DESCRIPTION".into(),
            "###FieldID".into(),
            format!("{}", k + 1),
            "###FinalCumulativeMeterSetWeight".into(),
            pyfloat(field_total),
            "###GantryAngle".into(),
            pyfloat(f.gantry),
            "###PatientSupportAngle".into(),
            pyfloat(f.couch),
            "###IsocenterPosition".into(),
            format!(
                "{} {} {}",
                pyfloat(f.iso[0]),
                pyfloat(f.iso[1]),
                pyfloat(f.iso[2])
            ),
        ]);
        if !f.rs.is_empty() {
            lines.extend([
                "###RangeShifterID".into(),
                f.rs.clone(),
                "###RangeShifterType".into(),
                "binary".into(),
            ]);
        }
        lines.extend([
            "###NumberOfControlPoints".into(),
            format!("{}", f.layers.len()),
            "".into(),
            "#SPOTS-DESCRIPTION".into(),
        ]);
        let mut cum = 0.0;
        for (j, (energy, wet, dist, spots)) in f.layers.iter().enumerate() {
            cum += spots.iter().map(|s| s[2]).sum::<f64>();
            lines.extend([
                "####ControlPointIndex".into(),
                format!("{}", j + 1),
                "####SpotTunnedID".into(),
                "1".into(),
                "####CumulativeMetersetWeight".into(),
                pyfloat(cum),
                "####Energy (MeV)".into(),
                pyfloat(*energy),
            ]);
            if !f.rs.is_empty() {
                lines.extend([
                    "####RangeShifterSetting".into(),
                    if *wet > 0.0 {
                        "IN".into()
                    } else {
                        "OUT".into()
                    },
                    "####IsocenterToRangeShifterDistance".into(),
                    pyfloat(*dist),
                    "####RangeShifterWaterEquivalentThickness".into(),
                    pyfloat(*wet),
                ]);
            }
            lines.extend([
                "####NbOfScannedSpots".into(),
                format!("{}", spots.len()),
                "####X Y Weight".into(),
            ]);
            for s in spots {
                lines.push(format!(
                    "{} {} {}",
                    pyfloat(s[0]),
                    pyfloat(s[1]),
                    pyfloat(s[2])
                ));
            }
        }
    }
    write_text(path, &(lines.join("\n") + "\n"))
}

/// Fields of an MCsquare Plan.txt, as [`write_plan`] writes them (rs_wet 0
/// when the shifter is out).
pub fn read_plan(path: &Path) -> Result<Vec<PlanField>, String> {
    let lines: Vec<String> = fs::read_to_string(path)
        .map_err(|err| format!("{path:?}: {err}"))?
        .lines()
        .map(|l| l.trim().to_string())
        .collect();
    let mut fields: Vec<PlanField> = Vec::new();
    let mut field: Option<PlanField> = None;
    let mut layers: Vec<Layer> = Vec::new();
    let mut n_spots = 0usize;
    let mut i = 0;
    while i < lines.len() {
        let key = lines[i].as_str();
        let nxt = lines.get(i + 1).map(String::as_str).unwrap_or("");
        if key == "#FIELD-DESCRIPTION" {
            if let Some(mut f) = field.take() {
                finish_field(&mut f, &mut layers);
                fields.push(f);
            }
            field = Some(PlanField {
                gantry: 0.0,
                couch: 0.0,
                iso: [0.0; 3],
                rs: String::new(),
                layers: Vec::new(),
            });
        } else if key == "###GantryAngle" {
            if let Some(f) = field.as_mut() {
                f.gantry = nxt.parse().unwrap_or(0.0);
            }
        } else if key == "###PatientSupportAngle" {
            if let Some(f) = field.as_mut() {
                f.couch = nxt.parse().unwrap_or(0.0);
            }
        } else if key == "###IsocenterPosition" {
            let v: Vec<f64> = nxt
                .split_whitespace()
                .map(|s| s.parse().unwrap_or(0.0))
                .collect();
            if v.len() == 3 {
                if let Some(f) = field.as_mut() {
                    f.iso = [v[0], v[1], v[2]];
                }
            }
        } else if key == "###RangeShifterID" {
            if let Some(f) = field.as_mut() {
                f.rs = nxt.to_string();
            }
        } else if key == "####ControlPointIndex" {
            layers.push(Layer {
                rs_dist: 400.0,
                ..Layer::default()
            });
        } else if key.starts_with("####Energy") {
            if let Some(ly) = layers.last_mut() {
                ly.energy = nxt.parse().unwrap_or(0.0);
            }
        } else if key == "####RangeShifterSetting" {
            if let Some(ly) = layers.last_mut() {
                ly.rs_in = nxt == "IN";
            }
        } else if key == "####IsocenterToRangeShifterDistance" {
            if let Some(ly) = layers.last_mut() {
                ly.rs_dist = nxt.parse().unwrap_or(400.0);
            }
        } else if key == "####RangeShifterWaterEquivalentThickness" {
            if let Some(ly) = layers.last_mut() {
                ly.rs_wet = nxt.parse().unwrap_or(0.0);
            }
        } else if key == "####NbOfScannedSpots" {
            n_spots = nxt.parse().unwrap_or(0);
        } else if key.starts_with("####X") {
            if let Some(ly) = layers.last_mut() {
                for k in 0..n_spots {
                    if let Some(row) = lines.get(i + 1 + k) {
                        let v: Vec<f64> = row
                            .split_whitespace()
                            .take(3)
                            .map(|s| s.parse().unwrap_or(0.0))
                            .collect();
                        if v.len() == 3 {
                            ly.spots.push([v[0], v[1], v[2]]);
                        }
                    }
                }
            }
            i += n_spots;
        }
        i += 1;
    }
    if let Some(mut f) = field.take() {
        finish_field(&mut f, &mut layers);
        fields.push(f);
    }
    Ok(fields)
}

#[derive(Default)]
struct Layer {
    energy: f64,
    rs_in: bool,
    rs_wet: f64,
    rs_dist: f64,
    spots: Vec<[f64; 3]>,
}

fn finish_field(f: &mut PlanField, layers: &mut Vec<Layer>) {
    f.layers = layers
        .drain(..)
        .map(|ly| {
            let wet = if ly.rs_in && !f.rs.is_empty() {
                ly.rs_wet
            } else {
                0.0
            };
            (ly.energy, wet, ly.rs_dist, ly.spots)
        })
        .collect();
}

pub fn write_patient_inputs(work: &Path, primaries: u64) -> Result<(), String> {
    write_text(
        &work.join("config.txt"),
        &format!(
            "Num_Threads 0\nRNG_Seed 0\nNum_Primaries {primaries}\nCT_File CT.mhd\n\
             HU_Density_Conversion_File {}\n\
             HU_Material_Conversion_File {}\n\
             BDL_Machine_Parameter_File {}\nBDL_Plan_File Plan.txt\n\
             Output_Directory Outputs\nDose_MHD_Output True\nLET_MHD_Output True\n\
             Compute_stat_uncertainty True\nDose_to_Water_conversion OnlineSPR\n",
            scanner_dir().join("HU_Density_Conversion.txt").display(),
            scanner_dir().join("HU_Material_Conversion.txt").display(),
            patient_bdl().display(),
        ),
    )
    .map_err(|err| err.to_string())?;
    fs::create_dir_all(work.join("Outputs")).map_err(|err| err.to_string())?;
    Ok(())
}

fn plan_protons(fields: &[PlanField], model: &BeamModel) -> f64 {
    fields
        .iter()
        .flat_map(|f| f.layers.iter())
        .flat_map(|(e, _, _, spots)| spots.iter().map(move |s| (e, s)))
        .map(|(e, s)| s[2] * f64::from(model.protons_per_mu(*e as f32)))
        .sum()
}

/// MCsquare (dose Gy, LETd keV/um) on the CT grid, `(nz, ny, nx)`.
pub fn run_mcsquare_patient(
    case: &PatientCase,
    exe: &Path,
    primaries: u64,
    work: &Path,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    build_patient(case, work)?;
    write_patient_inputs(work, primaries)?;
    let done = Command::new(exe)
        .arg("config.txt")
        .current_dir(work)
        .env("MCsquare_Materials_Dir", materials_dir())
        .output()
        .map_err(|err| err.to_string())?;
    let out = work.join("Outputs");
    if !done.status.success() || !out.join("Dose.mhd").is_file() {
        let stdout = String::from_utf8_lossy(&done.stdout);
        let stderr = String::from_utf8_lossy(&done.stderr);
        return Err(format!(
            "MCsquare failed on {}:\n{}\n{}",
            case.name,
            &stdout[stdout.len().saturating_sub(2000)..],
            &stderr[stderr.len().saturating_sub(2000)..]
        ));
    }
    let model = BeamModel::read(&patient_bdl())?;
    let fields = read_plan(&work.join("Plan.txt"))?;
    let protons = plan_protons(&fields, &model);
    let (_, dose) = read_mhd(&out.join("Dose.mhd"))?;
    let (_, let_d) = read_mhd(&out.join("LET.mhd"))?;
    Ok((
        dose.iter()
            .map(|d| (d * EV_PER_G_TO_GY * protons) as f32)
            .collect(),
        let_d.iter().map(|v| *v as f32).collect(),
    ))
}

/// The GPU run for the CT/Plan already in `work`: CT through `CtCalibration`,
/// BDL through `BeamModel`, engine in process. Returns (dose Gy, LETd,
/// density, ledger MeV/history).
pub fn run_gpu_patient(
    work: &Path,
    histories: u64,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, [f32; 6]), String> {
    let (dims, hu64) = read_mhd(&work.join("CT.mhd"))?;
    let spacing = mhd_spacing(&work.join("CT.mhd"))?;
    let hu: Vec<f32> = hu64.iter().map(|&v| v as f32).collect();
    let (nx, ny, nz) = (dims[0], dims[1], dims[2]);
    let calib = CtCalibration::resolve(scanner_dir().to_string_lossy().as_ref())?;
    let (material, density) = calib.voxels(&hu);
    let model = BeamModel::read(&patient_bdl())?;
    let fields = read_plan(&work.join("Plan.txt"))?;
    let size = [
        nx as f64 * spacing[0],
        ny as f64 * spacing[1],
        nz as f64 * spacing[2],
    ];
    let mut beams = Vec::new();
    let mut records = Vec::new();
    let mut protons = Vec::new();
    for (k, f) in fields.iter().enumerate() {
        // MCsquare's simulation frame is HFS patient LPS with y flipped, and its
        // x runs against the CT's (x_ct = Lx - x_sim): two mirrors, so
        // diag(-1, -1, 1) from LPS.
        let mut rotation = gantry_to_patient(f.gantry as f32, f.couch as f32, "HFS")?;
        for v in rotation[..6].iter_mut() {
            *v = -*v;
        }
        let mut iso = [f.iso[0] as f32, f.iso[1] as f32, f.iso[2] as f32];
        iso[0] = (size[0] - f.iso[0]) as f32;
        beams.extend_from_slice(&model.beam_record(&rotation, iso));
        for (energy, wet, dist, spots) in &f.layers {
            for s in spots {
                records.extend_from_slice(&model.spot_record(
                    *energy as f32,
                    s[0] as f32,
                    s[1] as f32,
                    k as f32,
                    *wet as f32,
                    *dist as f32,
                )?);
                protons.push((s[2] * f64::from(model.protons_per_mu(*energy as f32))) as f32);
            }
        }
    }
    let request = McJob::Patient(PatientRequest {
        spots: records,
        protons,
        beams,
        material,
        density: density.clone(),
        spacing_mm: [spacing[0] as f32, spacing[1] as f32, spacing[2] as f32],
        origin_mm: [0.0, 0.0, 0.0],
        shape: [nx, ny, nz],
        histories: histories as u32,
        seed: SEED,
        dose_to_water: true,
        score_let: true,
        let_water: false,
    });
    let result = scan_kit_compute::run_mc(&request).map_err(|err| err.to_string())?;
    Ok((result.volume.values, result.let_d, density, result.ledger))
}

/// sha256 of the inputs a golden was made from, first 16 hex chars.
pub fn patient_hash(work: &Path) -> Result<String, String> {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    for p in [
        work.join("CT.raw"),
        work.join("Plan.txt"),
        patient_bdl(),
        scanner_dir().join("HU_Density_Conversion.txt"),
        scanner_dir().join("HU_Material_Conversion.txt"),
    ] {
        h.update(fs::read(&p).map_err(|err| format!("{p:?}: {err}"))?);
    }
    Ok(format!("{:x}", h.finalize())[..16].to_string())
}

/// `(lo, hi)` crop + f16 dose/let + metadata, like `write_patient_golden`.
pub fn write_patient_golden(
    dir: &Path,
    case: &PatientCase,
    dose: &[f32],
    let_d: &[f32],
    nz: usize,
    ny: usize,
    nx: usize,
    primaries: u64,
    version: &str,
    digest: &str,
) -> Result<PathBuf, String> {
    let (lo, hi) = crop_box(dose, nz, ny, nx, CROP_FRAC, CROP_MARGIN);
    let core = crop_vol(dose, ny, nx, lo, hi);
    let let_core = crop_vol(let_d, ny, nx, lo, hi);
    let scale = core.iter().copied().fold(0.0f32, f32::max) as f64;
    let shape = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
    fs::create_dir_all(dir).map_err(|err| err.to_string())?;
    let path = dir.join(format!("patient_{}.npz", case.name));
    let crop: Vec<i64> = lo.iter().chain(hi.iter()).map(|&v| v as i64).collect();
    write_npz(
        &path,
        &[
            ("inputs".into(), npy_unicode(&[digest.to_string()])),
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
            ("let".into(), npy_f16(&shape, &let_core)),
        ],
    )?;
    Ok(path)
}

pub struct PatientGolden {
    pub inputs: String,
    pub crop: ([usize; 3], [usize; 3]),
    pub dose_scale: f64,
    pub shape: [usize; 3],
    pub dose: Vec<f32>,
    pub let_d: Vec<f32>,
}

pub fn load_patient_golden(path: &Path) -> Result<PatientGolden, String> {
    let files = load_npz(path)?;
    let crop = files.get("crop").ok_or("crop")?.i64s()?;
    if crop.len() != 6 {
        return Err("crop".into());
    }
    let lo = [crop[0] as usize, crop[1] as usize, crop[2] as usize];
    let hi = [crop[3] as usize, crop[4] as usize, crop[5] as usize];
    let shape = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
    Ok(PatientGolden {
        inputs: files.get("inputs").ok_or("inputs")?.scalar_string()?,
        crop: (lo, hi),
        dose_scale: files.get("dose_scale").ok_or("dose_scale")?.scalar_f64()?,
        shape,
        dose: files.get("dose").ok_or("dose")?.f16s()?,
        let_d: files.get("let").ok_or("let")?.f16s()?,
    })
}

/// A temp work dir for one case.
pub fn work_dir() -> Result<TempDir, String> {
    TempDir::new("mc_patient_")
}
