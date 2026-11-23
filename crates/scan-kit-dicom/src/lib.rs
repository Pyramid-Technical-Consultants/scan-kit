//! Little-endian DICOM — explicit and implicit VR: a study index, and CT,
//! structure, ion-plan, and dose reads.
//!
//! ponytail: undefined-length sequences are rejected. Compressed
//! pixel data is not decoded.

mod patient;
mod phantom;

pub use patient::{
    gantry_to_patient, hu_density, hu_label, inside_structure, load_study, read_ion_plan,
    scanner_density, scanner_label, write_rtdose, write_water_study, Beam, CtVolume, IonPlan,
    IonPlanSpot, PatientStudy, Spot, Structure,
};
pub use phantom::{phantom_summary, write_phantom_study};

use std::fs::{self, File};
use std::io::Read;
use std::path::Path;

use scan_kit_core::{ToolKind, ToolSpec};
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
    for file in dicom_files(path)? {
        if file
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("dcm"))
            || file_is_dicom(&file)
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
    let fraction = dose
        .iter()
        .filter(|value| value.is_finite() && **value >= goal)
        .count() as f32
        / (dose.len().max(1) as f32);
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
            let goal = input
                .get("goal")
                .and_then(Value::as_f64)
                .map(|n| n as f32)
                .filter(|n| n.is_finite())
                .ok_or("goal must be a finite number")?;
            // Optional — but a mistyped value must not silently default.
            let min_volume = match input.get("min_volume") {
                Some(value) => value
                    .as_f64()
                    .map(|n| n as f32)
                    .filter(|n| n.is_finite())
                    .ok_or("min_volume must be a finite number")?,
                None => 0.95,
            };
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
                .filter(|n| n.is_finite())
                .ok_or_else(|| "dose values must be finite numbers".to_owned())
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
    bytes.len() >= 132 && &bytes[128..132] == b"DICM"
}

/// Implicit VR Little Endian transfer syntax (no VR codes in the dataset).
pub(crate) const IMPLICIT_VR_LE: &str = "1.2.840.10008.1.2";

/// Transfer syntaxes that change the *dataset* encoding, not just the pixel
/// data — deflated explicit LE and big-endian explicit. Encapsulated
/// (JPEG/RLE) syntaxes keep an explicit-LE dataset and parse normally; only
/// their pixel element is unreadable, which fails on its own.
const UNREADABLE_DATASET: [&str; 2] = ["1.2.840.10008.1.2.1.99", "1.2.840.10008.1.2.2"];

/// Where the post-meta dataset starts, and whether it is implicit VR.
///
/// The group-0002 file meta is always explicit VR LE; its (0002,0010)
/// Transfer Syntax UID names the encoding that follows. A file whose first
/// tag is not group 0002 is sniffed instead: two ASCII uppercase letters in
/// the VR slot mean explicit, anything else reads as an implicit length.
pub(crate) fn dataset_offset(bytes: &[u8]) -> (usize, Result<bool, String>) {
    let mut cursor = 132usize;
    let mut syntax_seen = false;
    let mut syntax = String::new();
    let mut implicit = false;
    while cursor + 8 <= bytes.len() {
        let group = u16::from_le_bytes([bytes[cursor], bytes[cursor + 1]]);
        if group != 0x0002 {
            break;
        }
        let element = u16::from_le_bytes([bytes[cursor + 2], bytes[cursor + 3]]);
        let vr = &bytes[cursor + 4..cursor + 6];
        let Some((length, header)) = value_length(vr, &bytes[cursor + 6..]) else {
            break;
        };
        let start = cursor + 6 + header;
        let Some(end) = start.checked_add(length) else {
            break;
        };
        if end > bytes.len() {
            break;
        }
        if element == 0x0010 {
            syntax_seen = true;
            syntax = dicom_text(&bytes[start..end]);
            implicit = syntax == IMPLICIT_VR_LE;
        }
        cursor = end;
    }
    if !syntax_seen && cursor + 8 <= bytes.len() {
        // Explicit iff bytes 4-5 are a real VR code AND the element's
        // declared length fits the file — an implicit u32 length whose low
        // bytes happen to spell a VR ("SQ" = 0x5351) no longer fakes it,
        // because the element end then blows past the buffer.
        implicit = !looks_like_explicit_element(&bytes[cursor..]);
    }
    // A syntax that re-encodes the dataset itself (deflate, big-endian) must
    // not be parsed as explicit LE garbage. Encapsulated-pixel syntaxes keep
    // an explicit-LE dataset — RTPLAN/RTSTRUCT with no pixels read fine, and
    // a pixel element fails on its own undefined-length items.
    if UNREADABLE_DATASET.contains(&syntax.as_str()) {
        return (
            cursor.min(bytes.len()),
            Err(format!("unsupported transfer syntax {syntax}")),
        );
    }
    (cursor.min(bytes.len()), Ok(implicit))
}

/// Media Storage SOP Class UID from the always-explicit group-0002 meta, when
/// the file carries one. Lets `load_study` skip object classes it never reads
/// (treatment records, presentation states) without parsing their datasets.
pub(crate) fn meta_sop_class(bytes: &[u8]) -> Option<String> {
    let mut cursor = 132usize;
    while cursor + 8 <= bytes.len() {
        let group = u16::from_le_bytes([bytes[cursor], bytes[cursor + 1]]);
        if group != 0x0002 {
            return None;
        }
        let element = u16::from_le_bytes([bytes[cursor + 2], bytes[cursor + 3]]);
        let vr = &bytes[cursor + 4..cursor + 6];
        let (length, header) = value_length(vr, &bytes[cursor + 6..])?;
        let start = cursor + 6 + header;
        let end = start.checked_add(length)?;
        if end > bytes.len() {
            return None;
        }
        if element == 0x0002 {
            return Some(dicom_text(&bytes[start..end]));
        }
        cursor = end;
    }
    None
}

/// `.dcm` and `DICM` files under `root`, walking the Study/Series nesting
/// PACS and machine exports use. Symlinks are not followed, and the list is
/// sorted so "last plan read wins" is deterministic.
pub(crate) fn dicom_files(root: &Path) -> Result<Vec<std::path::PathBuf>, String> {
    const MAX_DEPTH: usize = 6;
    fn visit(dir: &Path, depth: usize, out: &mut Vec<std::path::PathBuf>) -> Result<(), String> {
        if depth == 0 {
            return Ok(());
        }
        // An unreadable directory entry must not silently drop slices from a
        // volume — load_study makes per-file read errors loud for the same
        // reason.
        let items = fs::read_dir(dir).map_err(|err| format!("{}: {err}", dir.display()))?;
        for item in items {
            let item = item.map_err(|err| format!("{}: {err}", dir.display()))?;
            let path = item.path();
            let kind = item
                .file_type()
                .map_err(|err| format!("{}: {err}", path.display()))?;
            if kind.is_symlink() {
                continue;
            }
            if kind.is_dir() {
                visit(&path, depth - 1, out)?;
            } else if kind.is_file() {
                out.push(path);
            }
        }
        Ok(())
    }
    let mut out = Vec::new();
    visit(root, MAX_DEPTH, &mut out)?;
    out.sort();
    Ok(out)
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
    let (mut cursor, implicit) = dataset_offset(&bytes[..limit]);
    let Ok(implicit) = implicit else {
        return out;
    };
    while cursor + 8 <= limit {
        let group = u16::from_le_bytes([bytes[cursor], bytes[cursor + 1]]);
        let element = u16::from_le_bytes([bytes[cursor + 2], bytes[cursor + 3]]);
        if group == 0x7FE0 && element == 0x0010 {
            break;
        }
        let Some((start, end)) = element_span(bytes, cursor, implicit) else {
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
    let file = match File::open(path) {
        Ok(file) => file,
        Err(_) => return StudyPeek::default(),
    };
    let mut bytes = Vec::new();
    if file.take(PEEK_CAP as u64).read_to_end(&mut bytes).is_err() {
        return StudyPeek::default();
    }
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
    // The index only scans header elements (stops at pixel data) — a
    // whole-file read would let one huge/crafted file OOM the process.
    let mut bytes = Vec::new();
    std::fs::File::open(path)
        .ok()?
        .take(PEEK_CAP as u64)
        .read_to_end(&mut bytes)
        .ok()?;
    if !is_dicom(&bytes) {
        return None;
    }
    let (mut cursor, implicit) = dataset_offset(&bytes);
    let Ok(implicit) = implicit else {
        return None;
    };
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
        let Some((start, end)) = element_span(&bytes, cursor, implicit) else {
            break;
        };
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

/// Value span of the element at `cursor`, in the file's VR mode. `None` on a
/// truncated header or an undefined length (not read).
fn element_span(bytes: &[u8], cursor: usize, implicit: bool) -> Option<(usize, usize)> {
    if implicit {
        if cursor + 8 > bytes.len() {
            return None;
        }
        let length = u32::from_le_bytes([
            bytes[cursor + 4],
            bytes[cursor + 5],
            bytes[cursor + 6],
            bytes[cursor + 7],
        ]) as usize;
        if length == u32::MAX as usize {
            return None;
        }
        let start = cursor + 8;
        return Some((start, start.checked_add(length)?));
    }
    let vr = bytes.get(cursor + 4..cursor + 6)?;
    let (length, header) = value_length(vr, bytes.get(cursor + 6..)?)?;
    let start = cursor + 6 + header;
    Some((start, start.checked_add(length)?))
}

/// Whether the element at `bytes[0]` reads as explicit VR: a known VR code
/// whose declared length stays inside the buffer. An implicit u32 length
/// whose low bytes spell a VR code ("SQ" = 0x5351) fails the bounds check —
/// the bytes it would read as length are the value's own head.
fn looks_like_explicit_element(bytes: &[u8]) -> bool {
    const LONG_VR: &[&[u8]] = &[
        b"OB", b"OW", b"OF", b"SQ", b"UT", b"UN", b"OD", b"OL", b"OV", b"UC", b"UR",
    ];
    const SHORT_VR: &[&[u8]] = &[
        b"AE", b"AS", b"AT", b"CS", b"DA", b"DS", b"DT", b"FD", b"FL", b"IS", b"LO", b"LT", b"PN",
        b"SH", b"SL", b"SS", b"ST", b"TM", b"UI", b"UL", b"US",
    ];
    let vr = &bytes[4..6];
    if !LONG_VR.contains(&vr) && !SHORT_VR.contains(&vr) {
        return false;
    }
    if LONG_VR.contains(&vr) && bytes[6..8] != [0, 0] {
        return false; // explicit long VRs reserve two zero bytes
    }
    let Some((length, header)) = value_length(vr, &bytes[6..]) else {
        return false;
    };
    6 + header + length <= bytes.len()
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
    push_ui(&mut bytes, 0x0008, 0x0016, sop_class)?;
    push_vr(&mut bytes, 0x0008, 0x0060, b"CS", modality)?;
    push_ui(&mut bytes, 0x0020, 0x000D, study)?;
    push_ui(&mut bytes, 0x0020, 0x000E, series)?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| err.to_string())?;
    }
    scan_kit_core::atomic_write(path, &bytes)
}

fn push_ui(bytes: &mut Vec<u8>, group: u16, element: u16, text: &str) -> Result<(), String> {
    push_vr(bytes, group, element, b"UI", text)
}

fn push_vr(
    bytes: &mut Vec<u8>,
    group: u16,
    element: u16,
    vr: &[u8; 2],
    text: &str,
) -> Result<(), String> {
    let mut value = text.as_bytes().to_vec();
    if value.len() % 2 == 1 {
        // UI pads with NUL; text VRs pad with a space.
        value.push(if vr == b"UI" { 0 } else { b' ' });
    }
    bytes.extend_from_slice(&group.to_le_bytes());
    bytes.extend_from_slice(&element.to_le_bytes());
    bytes.extend_from_slice(vr);
    bytes.extend_from_slice(
        &u16::try_from(value.len())
            .map_err(|_| "value exceeds the 16-bit length field".to_owned())?
            .to_le_bytes(),
    );
    bytes.extend(value);
    Ok(())
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
    fn a_clinical_goal_requires_goal_and_typed_min_volume() {
        // A missing goal silently evaluated every dose against 0% — a
        // guaranteed pass. Missing must error; so must a mistyped override.
        assert!(invoke("scan_kit_clinical_goal", &json!({"dose": [1.0]})).is_err());
        assert!(invoke(
            "scan_kit_clinical_goal",
            &json!({"dose": [1.0], "goal": 50.0, "min_volume": "high"})
        )
        .is_err());
        assert!(invoke(
            "scan_kit_clinical_goal",
            &json!({"dose": [1.0], "goal": 50.0})
        )
        .is_ok());
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

    #[test]
    fn an_unsupported_transfer_syntax_is_not_sniffed_as_data() {
        // Group-0002 meta declaring deflated explicit VR LE — the dataset is
        // re-encoded, so parsing it as explicit LE produces garbage tags.
        let mut bytes = vec![0u8; 128];
        bytes.extend_from_slice(b"DICM");
        push_tag(&mut bytes, 0x0002, 0x0010, b"UI", "1.2.840.10008.1.2.1.99");
        push_tag(
            &mut bytes,
            0x0008,
            0x0016,
            b"UI",
            "1.2.840.10008.5.1.4.1.1.2",
        );
        let (_, implicit) = dataset_offset(&bytes);
        assert!(implicit.unwrap_err().contains("unsupported"));
    }

    #[test]
    fn an_implicit_length_that_spells_a_vr_is_not_explicit() {
        // No file meta — the first dataset element is implicit VR with a u32
        // length whose low bytes read "SQ". The value head then "declares" a
        // length that runs past the file, so explicit fails the bounds check.
        let mut bytes = vec![0u8; 128];
        bytes.extend_from_slice(b"DICM");
        bytes.extend_from_slice(&0x0008u16.to_le_bytes());
        bytes.extend_from_slice(&0x0018u16.to_le_bytes());
        bytes.extend_from_slice(&0x5153u32.to_le_bytes()); // length; low u16 is "SQ"
        bytes.extend_from_slice(&u32::MAX.to_le_bytes()); // the value's head
        bytes.extend_from_slice(&[0u8; 20]);
        let (offset, implicit) = dataset_offset(&bytes);
        assert_eq!(offset, 132);
        assert!(implicit.unwrap());
    }

    #[test]
    fn the_header_writes_modality_as_cs_not_ui() {
        let dir = std::env::temp_dir().join(format!("scan-kit-hdr-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("ct.dcm");
        write_header(&path, "1.2.840", "CT", "1.2.3", "1.2.3.1").unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let pos = bytes
            .windows(4)
            .position(|w| w == [0x08, 0x00, 0x60, 0x00])
            .unwrap();
        assert_eq!(&bytes[pos + 4..pos + 6], b"CS");
        let _ = std::fs::remove_dir_all(&dir);
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
