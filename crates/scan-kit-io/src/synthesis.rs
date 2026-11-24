//! Plan Synthesis: catalog, generation, and the CSV the user saves.

use std::fs;
use std::io::Read;
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
    // A present-but-blank path is a caller error — silently degrading to
    // preview means the caller thinks a file was written.
    let path = match input.get("path") {
        None => None,
        Some(value) => match value.as_str().map(str::trim) {
            Some("") | None => return Err("path must be a non-empty string".into()),
            Some(text) => Some(text.to_owned()),
        },
    };
    // Every write lands through write_csv — fail a bad extension before
    // the parse, not after it.
    if let Some(path) = &path {
        if !Path::new(path)
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("csv"))
        {
            return Err("the plan path must end in .csv".into());
        }
    }
    if let Some(csv_value) = input.get("csv") {
        let Some(csv) = csv_value.as_str() else {
            return Err("csv must be a string".into());
        };
        let path = path.ok_or("path is required")?;
        if !csv.starts_with("#NO,ENERGY(MeV)") {
            return Err("The plan CSV is missing its header.".into());
        }
        write_csv(Path::new(&path), csv, db)?;
        return Ok(json!({ "written": path }));
    }

    let template = input
        .get("template")
        .and_then(Value::as_str)
        .ok_or("template is required")?;
    let params = input.get("params").cloned().unwrap_or_else(|| json!({}));
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
            // A PLD is a text spot table — nothing legitimately near 64 MiB.
            let mut file = fs::File::open(text(&params, "pld_path"))
                .map_err(|err| format!("Could not read PLD plan file: {err}"))?;
            let mut raw = String::new();
            std::io::Read::take(&mut file, 64 * 1024 * 1024 + 1)
                .read_to_string(&mut raw)
                .map_err(|err| format!("Could not read PLD plan file: {err}"))?;
            if raw.len() > 64 * 1024 * 1024 {
                return Err("PLD plan file exceeds 64 MiB".into());
            }
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
    // The csv+path pair crosses the invoke boundary — confine the write to
    // a .csv name or any text becomes an arbitrary file-write primitive.
    if !path
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("csv"))
    {
        return Err("the plan path must end in .csv".into());
    }
    // A bare "map.csv" lands in the process CWD and then stores CWD as the
    // next save's default — the zip-dest check names the same failure.
    match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => {
            fs::create_dir_all(parent).map_err(|err| err.to_string())?;
        }
        _ => return Err("the plan path must name a directory to write into".into()),
    }
    scan_kit_core::atomic_write(path, csv.as_bytes())?;
    if let Some(parent) = path.parent() {
        // Store the resolved directory - a bare "." or a relative parent is
        // not a usable default location for the next save.
        let dir = crate::store::canonical_local(parent);
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
        .trim()
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
            crate::store::canonical_local(out.parent().unwrap())
        );
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn the_plan_path_must_be_a_csv_name() {
        // csv+path crosses the invoke boundary — without the extension check
        // it is an arbitrary file-write primitive.
        let root = std::env::temp_dir().join(format!("scan-kit-csv-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let db = root.join("store.sqlite");
        for bad in ["map.txt", "map", "map.csv.exe"] {
            let target = root.join(bad);
            let err = synthesize(
                &json!({
                "template": "zero_field",
                "params": {
                "selected_energies": [70.0],
                "spots_per_layer": 1,
                "spot_weight_method": "fixed",
                "spot_weight_mu": 0.02
                },
                "path": target,
                "db_path": db
                }),
                &db,
            )
            .unwrap_err();
            assert!(err.contains(".csv"), "{bad}: {err}");
            assert!(!target.exists(), "{bad} should not have been written");
        }
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_non_string_csv_is_an_error_not_a_generate() {
        let err = synthesize(
            &serde_json::json!({"template": "csv_plan", "csv": 3}),
            Path::new("x"),
        )
        .unwrap_err();
        assert!(err.contains("csv must be a string"), "{err}");
    }
}
