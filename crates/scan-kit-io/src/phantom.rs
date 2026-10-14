//! Phantom Synthesis: a synthetic CT, RTSTRUCT, and RT Ion plan.

use std::path::Path;

use chrono::Local;
use scan_kit_core::standard_energies;
use scan_kit_dicom::{phantom_summary, write_phantom_study};
use serde_json::{json, Value};

use crate::store::{self, with_store};

pub fn catalog(db: &Path) -> Value {
    let save_dir = with_store(db, |conn| store::last_dicom_dir(conn))
        .ok()
        .flatten();
    let energies: Vec<f64> = standard_energies().to_vec();
    json!({
        "phantoms": [
            { "value": "slabs", "label": "Water box, bone and lung slabs" },
            { "value": "bone", "label": "Water box, bone slab" },
            { "value": "lung", "label": "Water box, lung slab" },
            { "value": "water", "label": "Water box" }
        ],
        "positions": ["HFS", "HFP", "FFS", "FFP"],
        "energies": energies,
        "defaults": {
            "phantom": "slabs",
            "position": "HFS",
            "pixel": 2.0,
            "slice": 2.0,
            "energies": [140.0, 130.0, 120.0],
            "gantry": 0.0,
            "couch": 0.0,
            "spot_pitch": 6.0,
            "range_shifter_wet": 0.0,
            "mu_per_spot": 0.02,
            "fractions": 1
        },
        "save_dir": save_dir
    })
}

pub fn preview(params: &Value) -> Result<Value, String> {
    Ok(json!({ "summary": phantom_summary(params)? }))
}

pub fn write(parent: &Path, params: &Value, db: &Path) -> Result<Value, String> {
    if !parent.is_dir() {
        return Err("Choose a folder to write the study into.".into());
    }
    let phantom = params
        .get("phantom")
        .and_then(Value::as_str)
        .unwrap_or("slabs");
    let stamp = Local::now().format("%Y%m%d_%H%M%S");
    let folder = parent.join(format!("synthetic_{phantom}_{stamp}"));
    if folder.exists() {
        return Err(format!("{} already exists", folder.display()));
    }
    let note = write_phantom_study(&folder, params)?;
    let text = folder.display().to_string();
    let _ = with_store(db, |conn| store::set_last_dicom_dir(conn, &text));
    Ok(json!({
        "folder": text,
        "note": note,
        "summary": phantom_summary(params)?,
        "status": format!(
            "Wrote {note} to {text}. Volumetric's Open Study starts here next."
        )
    }))
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    #[test]
    fn a_study_lands_in_a_stamped_folder() {
        let parent =
            std::env::temp_dir().join(format!("scan-kit-phantom-io-{}", std::process::id()));
        let _ = fs::remove_dir_all(&parent);
        fs::create_dir_all(&parent).unwrap();
        let db = parent.join("scan-kit.sqlite");
        let written = write(
            &parent,
            &json!({
                "phantom": "water",
                "pixel": 8.0,
                "slice": 8.0,
                "energies": [140.0],
                "fractions": 1
            }),
            &db,
        )
        .unwrap();
        let folder = written["folder"].as_str().unwrap();
        assert!(folder.contains("synthetic_water_"));
        assert!(Path::new(folder).join("RP_synthetic.dcm").is_file());
        assert!(written["status"].as_str().unwrap().contains("Wrote"));
        let remembered = with_store(&db, |conn| store::last_dicom_dir(conn))
            .unwrap()
            .unwrap();
        assert_eq!(remembered, folder);
        let listed = catalog(&db);
        assert_eq!(listed["save_dir"].as_str(), Some(folder));
        let _ = fs::remove_dir_all(&parent);
    }
}
