//! Phantom Synthesis: a synthetic CT, RTSTRUCT, and RT Ion plan.

use std::fs;
use std::io::ErrorKind;
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
    // The name lands in the folder path, so it is validated before the join
    // rather than relying on write_phantom_study's own check ordering.
    if !matches!(phantom, "slabs" | "bone" | "lung" | "water") {
        return Err(format!(
            "unknown phantom '{phantom}'; choose from slabs, bone, lung, water"
        ));
    }
    let stamp = Local::now().format("%Y%m%d_%H%M%S");
    // Claim the folder atomically — two writers in the same second used to
    // pass the exists() check together and interleave one study's files.
    let mut folder = parent.join(format!("synthetic_{phantom}_{stamp}"));
    let mut claimed = false;
    for suffix in 0..10u32 {
        match fs::create_dir(&folder) {
            Ok(()) => {
                claimed = true;
                break;
            }
            Err(err) if err.kind() == ErrorKind::AlreadyExists => {
                folder = parent.join(format!("synthetic_{phantom}_{stamp}-{suffix}"));
            }
            Err(err) => return Err(err.to_string()),
        }
    }
    // Exhaustion used to fall through and write into the unclaimed -9 —
    // and its rollback could remove_dir_all a folder another writer owns.
    if !claimed {
        return Err(format!(
            "could not claim a synthetic study folder under {}",
            parent.display()
        ));
    }
    let note = match write_phantom_study(&folder, params) {
        Ok(note) => note,
        Err(err) => {
            // Claimed above, so this can only drop this call's partial
            // study — a plausible-looking half-written one must not stay.
            let _ = fs::remove_dir_all(&folder);
            return Err(err);
        }
    };
    let text = store::canonical_local(&folder);
    if !text.is_empty() {
        let _ = with_store(db, |conn| store::set_last_dicom_dir(conn, &text));
    }
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
    fn a_traversal_phantom_name_is_rejected() {
        let parent =
            std::env::temp_dir().join(format!("scan-kit-phantom-traversal-{}", std::process::id()));
        let _ = fs::remove_dir_all(&parent);
        fs::create_dir_all(&parent).unwrap();
        let db = parent.join("scan-kit.sqlite");
        let err = write(&parent, &serde_json::json!({ "phantom": "../escape" }), &db).unwrap_err();
        assert!(err.contains("unknown phantom"));
        // An escaped write lands as a sibling of parent — check that too.
        assert!(!parent.parent().unwrap().join("escape").exists());
        assert!(
            fs::read_dir(&parent).unwrap().next().is_none()
                || fs::read_dir(&parent)
                    .unwrap()
                    .all(|entry| entry.unwrap().file_name() == "scan-kit.sqlite")
        );
        let _ = fs::remove_dir_all(&parent);
    }

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

    #[test]
    fn a_same_second_write_claims_a_suffixed_folder() {
        // Two writers in one second must not interleave one folder — the
        // atomic claim lands the second on a -0 name. Block both the
        // current and next second's names so the suffix path is forced
        // whichever tick the write lands in.
        let parent =
            std::env::temp_dir().join(format!("scan-kit-phantom-suffix-{}", std::process::id()));
        let _ = fs::remove_dir_all(&parent);
        fs::create_dir_all(&parent).unwrap();
        for offset in 0..=1 {
            let stamp = (Local::now() + chrono::Duration::seconds(offset)).format("%Y%m%d_%H%M%S");
            fs::create_dir(parent.join(format!("synthetic_water_{stamp}"))).unwrap();
        }
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
        assert!(folder.contains("-0"), "{folder}");
        let _ = fs::remove_dir_all(&parent);
    }
}
