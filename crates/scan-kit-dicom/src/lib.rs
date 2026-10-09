//! Explicit little-endian DICOM: a study index, and CT, structure, ion-plan, and dose reads.
//!
//! ponytail: implicit VR and undefined-length sequences are rejected. Compressed
//! pixel data is not decoded.

mod patient;
mod phantom;

pub use patient::{
    gantry_to_patient, hu_density, hu_label, inside_structure, load_study, read_ion_plan,
    scanner_density, scanner_label, write_water_study, Beam, CtVolume, IonPlan, IonPlanSpot,
    PatientStudy, Spot, Structure,
};
pub use phantom::{phantom_summary, write_phantom_study};

use std::fs::{self, File};
use std::io::Read;
use std::path::Path;

use scan_kit_core::{dvh, ToolKind, ToolSpec};
use serde::Serialize;
use serde_json::{json, Value};

const CT: &str = "1.2.840.10008.5.1.4.1.1.2";
const RT_STRUCT: &str = "1.2.840.10008.5.1.4.1.1.481.3";
const RT_PLAN: &str = "1.2.840.10008.5.1.4.1.1.481.8";
const RT_DOSE: &str = "1.2.840.10008.5.1.4.1.1.481.2";

const TOOLS: &[ToolSpec] = &[
    ToolSpec {
        name: "scan_kit_open_study",
        summary: "Index a DICOM folder and report modalities, goals, and a text summary.",
        kind: ToolKind::Workflow,
    },
    ToolSpec {
        name: "scan_kit_clinical_goal",
        summary: "Volume fraction of a dose grid at or above a goal dose.",
        kind: ToolKind::Granular,
    },
];

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct DicomEntry {
    pub path: String,
    pub sop_class: String,
    pub modality: String,
    pub study_uid: String,
    pub series_uid: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct GoalResult {
    pub dose: f32,
    pub volume_fraction: f32,
    pub passed: bool,
}

pub fn tools() -> &'static [ToolSpec] {
    TOOLS
}

pub fn tool_input_schema(name: &str) -> Value {
    match name {
        "scan_kit_open_study" => json!({
            "type": "object",
            "properties": { "path": { "type": "string" } },
            "required": ["path"],
            "additionalProperties": false
        }),
        "scan_kit_clinical_goal" => json!({
            "type": "object",
            "properties": {
                "dose": { "type": "array", "items": { "type": "number" } },
                "goal": { "type": "number" },
                "min_volume": { "type": "number" }
            },
            "required": ["dose", "goal"],
            "additionalProperties": false
        }),
        _ => json!({ "type": "object", "additionalProperties": false }),
    }
}

pub fn index_study(path: &Path) -> Result<Vec<DicomEntry>, String> {
    if !path.is_dir() {
        return Err(format!("{} is not a directory", path.display()));
    }
    let mut entries = Vec::new();
    for item in fs::read_dir(path).map_err(|err| err.to_string())? {
        let item = item.map_err(|err| err.to_string())?;
        let file = item.path();
        if file
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("dcm"))
            || fs::read(&file).ok().as_deref().is_some_and(is_dicom)
        {
            if let Some(entry) = read_entry(&file) {
                entries.push(entry);
            }
        }
    }
    entries.sort_by(|a, b| a.modality.cmp(&b.modality));
    Ok(entries)
}

pub fn clinical_goal(dose: &[f32], goal: f32, min_volume: f32) -> GoalResult {
    let mask = vec![true; dose.len()];
    let (_edges, curve) = dvh(dose, &mask, 20);
    let fraction = dose
        .iter()
        .filter(|value| value.is_finite() && **value >= goal)
        .count() as f32
        / (dose.len().max(1) as f32);
    let _ = curve;
    GoalResult {
        dose: goal,
        volume_fraction: fraction,
        passed: fraction + 1e-4 >= min_volume,
    }
}

pub fn report(entries: &[DicomEntry], goals: &[GoalResult]) -> String {
    let mut lines = vec![format!("{} DICOM objects", entries.len())];
    for entry in entries {
        lines.push(format!(
            "{} {} {}",
            entry.modality,
            class_name(&entry.sop_class),
            entry.series_uid
        ));
    }
    for goal in goals {
        let state = if goal.passed { "pass" } else { "fail" };
        lines.push(format!(
            "goal {state}: {:.0}% of volume at >= {:.2}",
            goal.volume_fraction * 100.0,
            goal.dose
        ));
    }
    lines.join("\n")
}

pub fn invoke(name: &str, input: &Value) -> Result<Value, String> {
    match name {
        "scan_kit_open_study" => {
            let path = input
                .get("path")
                .and_then(Value::as_str)
                .ok_or("path is required")?;
            let entries = index_study(Path::new(path))?;
            let text = report(&entries, &[]);
            Ok(json!({ "entries": entries, "report": text }))
        }
        "scan_kit_clinical_goal" => {
            let dose = f32s(input, "dose")?;
            let goal = input.get("goal").and_then(Value::as_f64).unwrap_or(0.0) as f32;
            let min_volume = input
                .get("min_volume")
                .and_then(Value::as_f64)
                .unwrap_or(0.95) as f32;
            let result = clinical_goal(&dose, goal, min_volume);
            Ok(json!(result))
        }
        _ => Err(format!("unknown tool {name}")),
    }
}

fn f32s(input: &Value, key: &str) -> Result<Vec<f32>, String> {
    input
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{key} is required"))?
        .iter()
        .map(|value| {
            value
                .as_f64()
                .map(|n| n as f32)
                .ok_or_else(|| "dose values must be numbers".to_owned())
        })
        .collect()
}

fn class_name(sop: &str) -> &'static str {
    match sop {
        CT => "CT",
        RT_STRUCT => "RTSTRUCT",
        RT_PLAN => "RTPLAN",
        RT_DOSE => "RTDOSE",
        _ => "OTHER",
    }
}

fn is_dicom(bytes: &[u8]) -> bool {
    bytes.len() > 132 && &bytes[128..132] == b"DICM"
}

const PEEK_CAP: usize = 4 * 1024 * 1024;

/// Header fields read from one file, stopping at the pixel-data tag.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StudyPeek {
    pub study_uid: String,
    pub patient_name: String,
    pub patient_id: String,
    pub study_date: String,
    pub study_description: String,
}

pub fn looks_like_dicom(bytes: &[u8]) -> bool {
    bytes.len() >= 132 && &bytes[128..132] == b"DICM"
}

/// Read study tags from the start of an explicit little-endian file.
///
/// Stops at the pixel-data tag or after 4 MiB, so a CT slice is not pulled in.
pub fn peek_study(bytes: &[u8]) -> StudyPeek {
    let mut out = StudyPeek::default();
    if !looks_like_dicom(bytes) {
        return out;
    }
    let limit = bytes.len().min(PEEK_CAP);
    let mut cursor = 132usize;
    while cursor + 8 <= limit {
        let group = u16::from_le_bytes([bytes[cursor], bytes[cursor + 1]]);
        let element = u16::from_le_bytes([bytes[cursor + 2], bytes[cursor + 3]]);
        if group == 0x7FE0 && element == 0x0010 {
            break;
        }
        let vr = &bytes[cursor + 4..cursor + 6];
        let Some((length, header)) = value_length(vr, &bytes[cursor + 6..]) else {
            break;
        };
        let Some(start) = cursor.checked_add(6 + header) else {
            break;
        };
        let Some(end) = start.checked_add(length) else {
            break;
        };
        if end > limit {
            break;
        }
        let text = dicom_text(&bytes[start..end]);
        match (group, element) {
            (0x0020, 0x000D) => out.study_uid = text,
            (0x0010, 0x0010) => out.patient_name = text.replace('^', " "),
            (0x0010, 0x0020) => out.patient_id = text,
            (0x0008, 0x0020) => out.study_date = text,
            (0x0008, 0x1030) => out.study_description = text,
            _ => {}
        }
        cursor = end;
    }
    out
}

pub fn peek_file(path: &Path) -> StudyPeek {
    let mut file = match File::open(path) {
        Ok(file) => file,
        Err(_) => return StudyPeek::default(),
    };
    let mut bytes = vec![0u8; PEEK_CAP];
    let Ok(read) = file.read(&mut bytes) else {
        return StudyPeek::default();
    };
    bytes.truncate(read);
    peek_study(&bytes)
}

/// `.dcm`, or the `DICM` preamble at byte 128. The name is enough for `.dcm`.
pub fn file_is_dicom(path: &Path) -> bool {
    if path
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("dcm"))
    {
        return true;
    }
    let mut file = match File::open(path) {
        Ok(file) => file,
        Err(_) => return false,
    };
    let mut head = [0u8; 132];
    file.read_exact(&mut head).is_ok() && looks_like_dicom(&head)
}

fn dicom_text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes)
        .trim_matches(|ch: char| ch == '\0' || ch.is_whitespace())
        .to_owned()
}

fn read_entry(path: &Path) -> Option<DicomEntry> {
    let bytes = fs::read(path).ok()?;
    if !is_dicom(&bytes) {
        return None;
    }
    let mut cursor = 132usize;
    let mut sop_class = String::new();
    let mut modality = String::new();
    let mut study_uid = String::new();
    let mut series_uid = String::new();
    while cursor + 8 <= bytes.len() {
        let group = u16::from_le_bytes([bytes[cursor], bytes[cursor + 1]]);
        let element = u16::from_le_bytes([bytes[cursor + 2], bytes[cursor + 3]]);
        if group == 0x7FE0 && element == 0x0010 {
            break;
        }
        let vr = &bytes[cursor + 4..cursor + 6];
        let (length, header) = value_length(vr, &bytes[cursor + 6..])?;
        let start = cursor + 6 + header;
        let end = start + length;
        if end > bytes.len() {
            break;
        }
        let text = String::from_utf8_lossy(&bytes[start..end])
            .trim_matches(|c: char| c == '\0' || c.is_whitespace())
            .to_owned();
        match (group, element) {
            (0x0008, 0x0016) => sop_class = text,
            (0x0008, 0x0060) => modality = text,
            (0x0020, 0x000D) => study_uid = text,
            (0x0020, 0x000E) => series_uid = text,
            _ => {}
        }
        cursor = end;
    }
    if sop_class.is_empty() && modality.is_empty() {
        return None;
    }
    Some(DicomEntry {
        path: path.display().to_string(),
        sop_class,
        modality,
        study_uid,
        series_uid,
    })
}

fn value_length(vr: &[u8], rest: &[u8]) -> Option<(usize, usize)> {
    let long = matches!(
        vr,
        b"OB" | b"OW" | b"OF" | b"SQ" | b"UT" | b"UN" | b"OD" | b"OL" | b"OV" | b"UC" | b"UR"
    );
    if long {
        if rest.len() < 6 {
            return None;
        }
        let length = u32::from_le_bytes([rest[2], rest[3], rest[4], rest[5]]) as usize;
        Some((length, 6))
    } else {
        if rest.len() < 2 {
            return None;
        }
        let length = u16::from_le_bytes([rest[0], rest[1]]) as usize;
        Some((length, 2))
    }
}

/// Write one explicit-VR DICOM header for tests and synthetic studies.
pub fn write_header(
    path: &Path,
    sop_class: &str,
    modality: &str,
    study: &str,
    series: &str,
) -> Result<(), String> {
    let mut bytes = vec![0u8; 128];
    bytes.extend_from_slice(b"DICM");
    push_ui(&mut bytes, 0x0008, 0x0016, sop_class);
    push_ui(&mut bytes, 0x0008, 0x0060, modality);
    push_ui(&mut bytes, 0x0020, 0x000D, study);
    push_ui(&mut bytes, 0x0020, 0x000E, series);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| err.to_string())?;
    }
    fs::write(path, bytes).map_err(|err| err.to_string())
}

fn push_ui(bytes: &mut Vec<u8>, group: u16, element: u16, text: &str) {
    let mut value = text.as_bytes().to_vec();
    if value.len() % 2 == 1 {
        value.push(0);
    }
    bytes.extend_from_slice(&group.to_le_bytes());
    bytes.extend_from_slice(&element.to_le_bytes());
    bytes.extend_from_slice(b"UI");
    bytes.extend_from_slice(&(value.len() as u16).to_le_bytes());
    bytes.extend(value);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sk_req_032_study_index_goal_and_report() {
        let root = std::env::temp_dir().join(format!("scan-kit-dicom-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        write_header(&root.join("ct.dcm"), CT, "CT", "1.2.3", "1.2.3.1").unwrap();
        write_header(
            &root.join("dose.dcm"),
            RT_DOSE,
            "RTDOSE",
            "1.2.3",
            "1.2.3.2",
        )
        .unwrap();
        let entries = index_study(&root).unwrap();
        assert_eq!(entries.len(), 2);
        let goal = clinical_goal(&[0.0, 2.0, 2.5, 1.0], 2.0, 0.5);
        assert!(goal.passed);
        let text = report(&entries, &[goal]);
        assert!(text.contains("CT"));
        assert!(text.contains("pass"));
        let opened = invoke(
            "scan_kit_open_study",
            &json!({ "path": root.display().to_string() }),
        )
        .unwrap();
        assert_eq!(opened["entries"].as_array().unwrap().len(), 2);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn peek_stops_at_pixel_data_and_reads_the_study_header() {
        let mut bytes = vec![0u8; 128];
        bytes.extend_from_slice(b"DICM");
        push_tag(&mut bytes, 0x0010, 0x0010, b"PN", "Doe^Jane");
        push_tag(&mut bytes, 0x0010, 0x0020, b"LO", "P1");
        push_tag(&mut bytes, 0x0008, 0x0020, b"DA", "20260102");
        push_tag(&mut bytes, 0x0008, 0x1030, b"LO", "Head");
        push_tag(&mut bytes, 0x0020, 0x000D, b"UI", "1.2.3");
        bytes.extend_from_slice(&0x7FE0u16.to_le_bytes());
        bytes.extend_from_slice(&0x0010u16.to_le_bytes());
        bytes.extend_from_slice(b"OB");
        bytes.extend_from_slice(&0u16.to_le_bytes());
        bytes.extend_from_slice(&u32::MAX.to_le_bytes());
        let peek = peek_study(&bytes);
        assert_eq!(peek.patient_name, "Doe Jane");
        assert_eq!(peek.patient_id, "P1");
        assert_eq!(peek.study_date, "20260102");
        assert_eq!(peek.study_description, "Head");
        assert_eq!(peek.study_uid, "1.2.3");
        assert!(looks_like_dicom(&bytes));
        assert!(!looks_like_dicom(b"nope"));
    }

    fn push_tag(bytes: &mut Vec<u8>, group: u16, element: u16, vr: &[u8], text: &str) {
        let mut value = text.as_bytes().to_vec();
        if value.len() % 2 == 1 {
            value.push(b' ');
        }
        bytes.extend_from_slice(&group.to_le_bytes());
        bytes.extend_from_slice(&element.to_le_bytes());
        bytes.extend_from_slice(vr);
        bytes.extend_from_slice(&(u16::try_from(value.len()).unwrap_or(u16::MAX)).to_le_bytes());
        bytes.extend(value);
    }
}
