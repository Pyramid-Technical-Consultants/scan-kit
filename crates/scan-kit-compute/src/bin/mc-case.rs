//! One slab Monte Carlo, for `validation/mcsquare_validate.py`.
//!
//! Arguments are a JSON request and the raw little-endian f32 dose to write.
//! The dose is x-fastest, `x + nx * (y + ny * z)`, with z = 0 at the origin.

use std::env;
use std::fs;
use std::process::ExitCode;

use scan_kit_compute::run_mc;
use scan_kit_core::{McJob, SlabRequest};
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
    let request_path = args.next().ok_or("usage: mc-case request.json dose.f32")?;
    let out_path = args.next().ok_or("usage: mc-case request.json dose.f32")?;
    let text = fs::read_to_string(&request_path).map_err(|err| err.to_string())?;
    let request: Value = serde_json::from_str(&text).map_err(|err| err.to_string())?;
    let shape = shape_of(&request)?;
    let origin = origin_of(&request)?;
    let job = McJob::Slab(SlabRequest {
        medium: request
            .get("medium")
            .and_then(Value::as_str)
            .unwrap_or("water")
            .to_string(),
        x: floats(&request, "x")?,
        y: floats(&request, "y")?,
        sx: floats(&request, "sx")?,
        sy: floats(&request, "sy")?,
        energy: floats(&request, "energy")?,
        protons: floats(&request, "protons")?,
        histories: number(&request, "histories")? as u32,
        seed: number(&request, "seed")? as u32,
        spread_pct: number(&request, "spread_pct")?,
        wet_mm: number(&request, "wet_mm")?,
        depth_mm: number(&request, "depth_mm")?,
        voxel_mm: number(&request, "voxel_mm")?,
        origin,
        shape,
    });
    let result = run_mc(&job).map_err(|err| err.to_string())?;
    let [nx, ny, nz] = result.volume.shape;
    let mut bytes = Vec::with_capacity(result.volume.values.len() * 4);
    for value in &result.volume.values {
        bytes.extend(value.to_le_bytes());
    }
    fs::write(&out_path, bytes).map_err(|err| err.to_string())?;
    println!("{nx} {ny} {nz}");
    Ok(())
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

fn origin_of(request: &Value) -> Result<[f32; 3], String> {
    let items = request
        .get("origin")
        .and_then(Value::as_array)
        .ok_or("missing origin")?;
    if items.len() != 3 {
        return Err("origin needs 3 coordinates".into());
    }
    let mut origin = [0.0f32; 3];
    for (slot, item) in origin.iter_mut().zip(items) {
        *slot = item.as_f64().ok_or("origin must be numeric")? as f32;
    }
    Ok(origin)
}
