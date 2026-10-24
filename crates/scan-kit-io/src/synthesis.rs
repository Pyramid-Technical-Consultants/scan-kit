//! Plan Synthesis: catalog, generation, and the CSV the user saves.

use std::fs;
use std::path::Path;

use scan_kit_core::{
    build_plan, dicom_beam_size, plan_catalog, ImportSpot, PlanDocument, PlanSource,
};
use scan_kit_dicom::read_ion_plan;
use serde_json::{json, Value};

use crate::store::{self, with_store};

pub fn catalog(db: &Path) -> Value {
    let mut catalog = plan_catalog();
    let save_dir = with_store(db, |conn| store::last_plan_save_dir(conn))
        .ok()
        .flatten();
    catalog["save_dir"] = json!(save_dir);
    catalog
}

pub fn synthesize(input: &Value, db: &Path) -> Result<Value, String> {
    let template = input
        .get("template")
        .and_then(Value::as_str)
        .ok_or("template is required")?;
    let params = input.get("params").cloned().unwrap_or_else(|| json!({}));
    let path = input
        .get("path")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|path| !path.is_empty())
        .map(str::to_owned);
    if let Some(csv) = input.get("csv").and_then(Value::as_str) {
        let path = path.ok_or("path is required")?;
        if !csv.starts_with("#NO,ENERGY(MeV)") {
            return Err("The plan CSV is missing its header.".into());
        }
        write_csv(Path::new(&path), csv, db)?;
        return Ok(json!({ "written": path }));
    }

    let mut errors = scan_kit_core::validate_plan(template, &params);
    if template == "dicom_rt_plan" {
        errors.splice(0..0, dicom_path_errors(&text(&params, "dicom_path")));
    } else if template == "iba_pld_plan" {
        errors.splice(0..0, pld_path_errors(&text(&params, "pld_path")));
    }
    if !errors.is_empty() {
        return Err(errors.join("\n"));
    }

    let document = match template {
        "dicom_rt_plan" => {
            let plan = read_ion_plan(Path::new(&text(&params, "dicom_path")))?;
            if plan.spots.is_empty() {
                return Err("No planned spots with positive MU found in DICOM plan".into());
            }
            let use_dicom = params
                .get("use_dicom_beam_size")
                .and_then(Value::as_bool)
                .unwrap_or(true);
            let fallback = if use_dicom {
                3.61
            } else {
                params
                    .get("beam_size_override_mm")
                    .and_then(Value::as_f64)
                    .unwrap_or(3.61)
            };
            let spots: Vec<ImportSpot> = plan
                .spots
                .iter()
                .map(|spot| ImportSpot {
                    x: f64::from(spot.x),
                    y: f64::from(spot.y),
                    energy: f64::from(spot.energy),
                    charge: f64::from(spot.charge),
                    beam_size: dicom_beam_size(
                        f64::from(spot.size_x),
                        f64::from(spot.size_y),
                        fallback,
                        use_dicom,
                    ),
                    plan_index: spot.plan_index,
                })
                .collect();
            let label = if plan.label.is_empty() {
                None
            } else {
                Some(plan.label)
            };
            build_plan(template, &params, PlanSource::Ion(&spots), label.as_deref())?
        }
        "iba_pld_plan" => {
            let raw = fs::read_to_string(text(&params, "pld_path"))
                .map_err(|err| format!("Could not read PLD plan file: {err}"))?;
            let text = raw.trim_start_matches('\u{feff}');
            build_plan(template, &params, PlanSource::Pld(text), None)?
        }
        _ => build_plan(template, &params, PlanSource::Generate, None)?,
    };

    let written = if let Some(path) = &path {
        write_csv(Path::new(path), &document.csv, db)?;
        Some(path.clone())
    } else {
        None
    };
    Ok(document_json(&document, written))
}

fn document_json(document: &PlanDocument, written: Option<String>) -> Value {
    json!({
        "summary": document.summary,
        "filename": document.filename,
        "columns": document.columns,
        "rows": document.preview,
        "csv": document.csv,
        "written": written,
    })
}

fn write_csv(path: &Path, csv: &str, db: &Path) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|err| err.to_string())?;
        }
    }
    fs::write(path, csv).map_err(|err| err.to_string())?;
    if let Some(parent) = path.parent() {
        let dir = parent.to_string_lossy().to_string();
        if !dir.is_empty() {
            let _ = with_store(db, |conn| store::set_last_plan_save_dir(conn, &dir));
        }
    }
    Ok(())
}

fn dicom_path_errors(path: &str) -> Vec<String> {
    let text = path.trim();
    if text.is_empty() {
        return vec!["Select an RT Ion DICOM plan file (.dcm).".into()];
    }
    let file = Path::new(text);
    if !file.is_file() {
        return vec![format!("DICOM plan file not found: {text}")];
    }
    match file.extension().and_then(|ext| ext.to_str()) {
        Some(ext) if ext.eq_ignore_ascii_case("dcm") => Vec::new(),
        _ => vec!["Plan file must have a .dcm extension.".into()],
    }
}

fn pld_path_errors(path: &str) -> Vec<String> {
    let text = path.trim();
    if text.is_empty() {
        return vec!["Select an IBA PLD plan file (.pld).".into()];
    }
    let file = Path::new(text);
    if !file.is_file() {
        return vec![format!("PLD plan file not found: {text}")];
    }
    match file.extension().and_then(|ext| ext.to_str()) {
        Some(ext) if ext.eq_ignore_ascii_case("pld") => Vec::new(),
        _ => vec!["Plan file must have a .pld extension.".into()],
    }
}

fn text(params: &Value, key: &str) -> String {
    params
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_zero_field_writes_the_export_header_and_remembers_the_folder() {
        let root = std::env::temp_dir().join(format!(
            "scan-kit-plan-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let db = root.join("store.sqlite");
        let out = root.join("out").join("map.csv");
        let missing = synthesize(
            &json!({
                "template": "dicom_rt_plan",
                "params": { "dicom_path": "missing.dcm", "use_dicom_beam_size": true, "spot_order": "plan_order" },
                "db_path": db
            }),
            &db,
        );
        assert!(missing.unwrap_err().contains("not found"));
        let built = synthesize(
            &json!({
                "template": "zero_field",
                "params": {
                    "selected_energies": [250.0, 70.0],
                    "spots_per_layer": 1,
                    "spot_weight_method": "fixed",
                    "spot_weight_mu": 0.02
                },
                "path": out,
                "db_path": db
            }),
            &db,
        )
        .unwrap();
        let csv = fs::read_to_string(&out).unwrap();
        assert!(csv.starts_with(
            "#NO,ENERGY(MeV),CURRENT(A),BEAM_SIZE(mm),X_POSITION(mm),Y_POSITION(mm),CHARGE_REQ(MU),VELOCITY(mm/s)\n"
        ));
        assert!(csv.contains("\n1,250,"));
        assert!(csv.contains("\n2,70,"));
        assert_eq!(built["written"], json!(out.to_string_lossy()));
        let catalog = catalog(&db);
        assert_eq!(
            catalog["save_dir"].as_str().unwrap(),
            out.parent().unwrap().to_string_lossy()
        );
        let _ = fs::remove_dir_all(&root);
    }
}
