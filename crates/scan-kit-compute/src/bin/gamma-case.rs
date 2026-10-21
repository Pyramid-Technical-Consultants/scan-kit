//! Global gamma pass rate between two little-endian f32 volumes, for
//! `validation/mcsquare_validate.py`.
//!
//! `gamma-case ref.f32 evl.f32 nx ny nz sx sy sz dose_pct dta_mm cutoff_pct`
//! prints `<passed> <evaluated>`; the dose criterion is a percent of the
//! evaluated maximum like the app's default.

use std::env;
use std::fs;
use std::path::Path;
use std::process::ExitCode;

use scan_kit_core::gamma_index_norm;

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
    let args: Vec<String> = env::args().skip(1).collect();
    let [reference_path, evaluated_path, nx, ny, nz, sx, sy, sz, dose_pct, dta_mm, cutoff_pct] =
        args.as_slice()
    else {
        return Err(
            "usage: gamma-case ref.f32 evl.f32 nx ny nz sx sy sz dose_pct dta_mm cutoff_pct"
                .to_string(),
        );
    };
    let int = |s: &String| s.parse::<usize>().map_err(|e| e.to_string());
    let f32_ = |s: &String| s.parse::<f32>().map_err(|e| e.to_string());
    let shape = [int(nx)?, int(ny)?, int(nz)?];
    let spacing = [f32_(sx)?, f32_(sy)?, f32_(sz)?];
    let reference = load_f32(Path::new(reference_path))?;
    let evaluated = load_f32(Path::new(evaluated_path))?;
    let (_map, passed, scored) = gamma_index_norm(
        &reference,
        &evaluated,
        shape,
        f32_(dose_pct)?,
        f32_(dta_mm)?,
        spacing,
        f32_(cutoff_pct)?,
        0.0,
    );
    println!("{passed} {scored}");
    Ok(())
}

fn load_f32(path: &Path) -> Result<Vec<f32>, String> {
    let bytes = fs::read(path).map_err(|e| format!("{path:?}: {e}"))?;
    let (floats, rest) = bytes.as_chunks::<4>();
    if !rest.is_empty() {
        return Err(format!("{path:?}: not a whole number of f32"));
    }
    Ok(floats.iter().map(|c| f32::from_le_bytes(*c)).collect())
}
