//! One Monte Carlo, for `validation/mcsquare_validate.py`.
//!
//! A slab request is JSON. A patient request (`"kind": "patient"`) names raw
//! little-endian files beside that JSON: spots, protons, beams and density as
//! f32, material as u8. The dose (and LETd, when a third path is given) is
//! x-fastest, `x + nx * (y + ny * z)`. A patient run also prints the energy
//! ledger in MeV per history and the dose uncertainty.

use std::env;
use std::fs;
use std::path::Path;
use std::process::ExitCode;

use scan_kit_compute::run_mc;
use scan_kit_core::{McJob, McResult, PatientRequest, SlabRequest};
use serde_json::Value;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let mut args = env::args().skip(1);
    let request_path = args
        .next()
        .ok_or("usage: mc-case request.json dose.f32 [let.f32]")?;
    let out_path = args
        .next()
        .ok_or("usage: mc-case request.json dose.f32 [let.f32]")?;
    let let_path = args.next();
    let text = fs::read_to_string(&request_path).map_err(|err| err.to_string())?;
    let request: Value = serde_json::from_str(&text).map_err(|err| err.to_string())?;
    let patient = request.get("kind").and_then(Value::as_str) == Some("patient");
    let result = if patient {
        let dir = Path::new(&request_path)
            .parent()
            .unwrap_or_else(|| Path::new("."));
        run_mc(&patient_job(&request, dir)?).map_err(|err| err.to_string())?
    } else {
        run_mc(&slab_job(&request)?).map_err(|err| err.to_string())?
    };
    store_f32(Path::new(&out_path), &result.volume.values)?;
    if let Some(path) = &let_path {
        if result.let_d.is_empty() {
            return Err("LET was not scored".into());
        }
        store_f32(Path::new(path), &result.let_d)?;
    }
    report(&result, patient);
    Ok(())
}

fn report(result: &McResult, patient: bool) {
    let [nx, ny, nz] = result.volume.shape;
    println!("{nx} {ny} {nz}");
    if patient {
        let ledger = result.ledger;
        println!(
            "{} {} {} {} {} {}",
            ledger[0], ledger[1], ledger[2], ledger[3], ledger[4], ledger[5]
        );
        println!("{}", result.uncertainty);
    }
}

fn slab_job(request: &Value) -> Result<McJob, String> {
    Ok(McJob::Slab(SlabRequest {
        medium: request
            .get("medium")
            .and_then(Value::as_str)
            .unwrap_or("water")
            .to_string(),
        x: floats(request, "x")?,
        y: floats(request, "y")?,
        sx: floats(request, "sx")?,
        sy: floats(request, "sy")?,
        energy: floats(request, "energy")?,
        protons: floats(request, "protons")?,
        histories: number(request, "histories")? as u32,
        seed: number(request, "seed")? as u32,
        spread_pct: number(request, "spread_pct")?,
        wet_mm: number(request, "wet_mm")?,
        depth_mm: number(request, "depth_mm")?,
        voxel_mm: number(request, "voxel_mm")?,
        origin: vec3(request, "origin")?,
        shape: shape_of(request)?,
    }))
}

fn patient_job(request: &Value, dir: &Path) -> Result<McJob, String> {
    let shape = shape_of(request)?;
    let nvox = shape[0] * shape[1] * shape[2];
    let spots = load_f32(&dir.join(name_of(request, "spots")?))?;
    let protons = load_f32(&dir.join(name_of(request, "protons")?))?;
    let beams = load_f32(&dir.join(name_of(request, "beams")?))?;
    let material =
        fs::read(dir.join(name_of(request, "material")?)).map_err(|err| err.to_string())?;
    let density = load_f32(&dir.join(name_of(request, "density")?))?;
    if material.len() != nvox || density.len() != nvox {
        return Err(format!(
            "material {} and density {} do not match {nvox} voxels",
            material.len(),
            density.len()
        ));
    }
    Ok(McJob::Patient(PatientRequest {
        spots,
        protons,
        beams,
        material,
        density,
        spacing_mm: vec3(request, "spacing_mm")?,
        origin_mm: vec3(request, "origin_mm")?,
        shape,
        histories: number(request, "histories")? as u32,
        seed: number(request, "seed")? as u32,
        dose_to_water: request
            .get("dose_to_water")
            .and_then(Value::as_bool)
            .unwrap_or(true),
        score_let: true,
    }))
}

fn number(request: &Value, key: &str) -> Result<f32, String> {
    request
        .get(key)
        .and_then(Value::as_f64)
        .map(|value| value as f32)
        .ok_or_else(|| format!("missing {key}"))
}

fn floats(request: &Value, key: &str) -> Result<Vec<f32>, String> {
    let items = request
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("missing {key}"))?;
    Ok(items
        .iter()
        .map(|item| item.as_f64().unwrap_or(f64::NAN) as f32)
        .collect())
}

fn shape_of(request: &Value) -> Result<[usize; 3], String> {
    let items = request
        .get("shape")
        .and_then(Value::as_array)
        .ok_or("missing shape")?;
    if items.len() != 3 {
        return Err("shape needs 3 counts".into());
    }
    let mut shape = [0usize; 3];
    for (slot, item) in shape.iter_mut().zip(items) {
        *slot = item.as_u64().ok_or("shape counts must be integers")? as usize;
    }
    Ok(shape)
}

fn vec3(request: &Value, key: &str) -> Result<[f32; 3], String> {
    let items = request
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("missing {key}"))?;
    if items.len() != 3 {
        return Err(format!("{key} needs 3 numbers"));
    }
    let mut out = [0.0f32; 3];
    for (slot, item) in out.iter_mut().zip(items) {
        *slot = item
            .as_f64()
            .ok_or_else(|| format!("{key} must be numeric"))? as f32;
    }
    Ok(out)
}

fn name_of(request: &Value, key: &str) -> Result<String, String> {
    request
        .get(key)
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| format!("missing {key}"))
}

fn load_f32(path: &Path) -> Result<Vec<f32>, String> {
    let bytes = fs::read(path).map_err(|err| format!("{}: {err}", path.display()))?;
    if bytes.len() % 4 != 0 {
        return Err(format!("{}: not a f32 file", path.display()));
    }
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect())
}

fn store_f32(path: &Path, values: &[f32]) -> Result<(), String> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend(value.to_le_bytes());
    }
    fs::write(path, bytes).map_err(|err| err.to_string())
}
