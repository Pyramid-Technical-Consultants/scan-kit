//! CT, RTSTRUCT, RT Ion Plan, and RTDOSE. One frame of reference, or the load fails.

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

const DENSITY: &str =
    include_str!("../../../assets/mcsquare/Scanners/default/HU_Density_Conversion.txt");
const MATERIALS: &str =
    include_str!("../../../assets/mcsquare/Scanners/default/HU_Material_Conversion.txt");

#[derive(Clone, Debug)]
pub struct CtVolume {
    pub frame: String,
    pub origin: [f32; 3],
    pub spacing: [f32; 3],
    pub shape: [usize; 3],
    pub hu: Vec<f32>,
    pub position: String,
    /// RTDOSE DoseType (3004,0004) — "EFFECTIVE" values are already Gy(RBE).
    /// Empty on a CT volume.
    pub dose_type: String,
    /// RTDOSE DoseSummationType (3004,000A); empty on a CT volume.
    pub summation: String,
    /// Patient/study identity of the source study; RTDOSE written against the
    /// CT repeats it so the dose lands in the same study.
    pub patient_id: String,
    pub patient_name: String,
    pub study_uid: String,
    /// Series UID and per-slice SOP UIDs of the source CT, for provenance.
    pub series_uid: String,
    pub sop_uids: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct Spot {
    pub energy: f32,
    pub x: f32,
    pub y: f32,
    pub mu: f32,
    pub shifter: String,
    pub wet_mm: f32,
    pub shifter_distance_mm: f32,
    /// Mean scanning-spot size in mm, or 0 when the control point does not carry one.
    pub spot_size_mm: f32,
}

#[derive(Clone, Debug)]
pub struct Beam {
    pub number: i32,
    pub name: String,
    pub gantry: f32,
    pub couch: f32,
    pub isocenter: [f32; 3],
    pub position: String,
    pub meterset: f32,
    pub spots: Vec<Spot>,
}

#[derive(Clone, Debug)]
pub struct Structure {
    pub name: String,
    /// One closed contour per planar item. Slices stay separate so a voxel is
    /// tested against the contour on its own plane.
    pub contour: Vec<Vec<[f32; 3]>>,
}

/// One control-point spot from an RT Ion plan, with the raw meterset weight.
#[derive(Clone, Debug)]
pub struct IonPlanSpot {
    pub x: f32,
    pub y: f32,
    pub energy: f32,
    pub charge: f32,
    pub size_x: f32,
    pub size_y: f32,
    pub plan_index: i32,
}

#[derive(Clone, Debug)]
pub struct IonPlan {
    pub label: String,
    pub spots: Vec<IonPlanSpot>,
}

pub struct PatientStudy {
    pub frame: String,
    pub ct: CtVolume,
    pub beams: Vec<Beam>,
    pub fractions: i32,
    pub structures: Vec<Structure>,
    pub tps: Option<CtVolume>,
    pub prescription: f32,
    /// SOP Instance UID and label of the RTPLAN the beams came from, if any.
    pub plan_uid: String,
    pub plan_label: String,
    /// SOP Instance UID of the RTSTRUCT the structures came from, if any.
    pub structures_uid: String,
}

struct Elem {
    tag: u32,
    /// The explicit-VR code, or [0, 0] under implicit VR — the coercion
    /// helpers route on this before guessing at a value's byte shape.
    vr: [u8; 2],
    items: Vec<Vec<Elem>>,
    bytes: Vec<u8>,
}

pub fn load_study(path: &Path) -> Result<PatientStudy, String> {
    let mut ct_by_series: std::collections::BTreeMap<String, (Vec<String>, Vec<(String, Slice)>)> =
        std::collections::BTreeMap::new();
    let mut dose_slices = Vec::new();
    let mut dose_seen = false;
    let mut dose_type = String::new();
    let mut summation = String::new();
    let mut plan_raw = None;
    // The plan's Patient Setup Sequence is optional; when absent, the CT's
    // own Patient Position (0018,5100) is the only orientation source —
    // keyed by series so a merged folder cannot borrow a different
    // patient's orientation.
    let mut ct_positions: std::collections::BTreeMap<String, String> =
        std::collections::BTreeMap::new();
    let mut structures = Vec::new();
    let mut structures_frame = String::new();
    let mut patient_id = String::new();
    let mut patient_name = String::new();
    let mut study_uid = String::new();
    let mut series_uid = String::new();
    let mut sop_uids = Vec::new();
    let mut structures_uid = String::new();
    // Identity is taken from the winning series below, not the first CT file
    // in sorted order — a merged folder can hold more than one patient.
    let mut identities: std::collections::BTreeMap<String, (String, String, String)> =
        std::collections::BTreeMap::new();
    for file in crate::dicom_files(path)? {
        // Unreadable files and oversized files are loud failures — an I/O
        // error that silently skipped a CT slice would drop it from the
        // volume without a trace. Same per-file ceiling as read_ion_plan.
        let mut source =
            fs::File::open(&file).map_err(|err| format!("{}: {err}", file.display()))?;
        let mut bytes = Vec::new();
        std::io::Read::take(&mut source, 256 * 1024 * 1024 + 1)
            .read_to_end(&mut bytes)
            .map_err(|err| format!("{}: {err}", file.display()))?;
        if bytes.len() < 132 || &bytes[128..132] != b"DICM" {
            continue;
        }
        // The cap applies to DICOM payloads — an oversized non-DICOM file in
        // the folder is skipped on the magic check, not a hard failure.
        if bytes.len() > 256 * 1024 * 1024 {
            return Err(format!("{} exceeds 256 MiB", file.display()));
        }
        // A meta that names a class we never read (treatment records, display
        // objects) skips the dataset parse entirely — those datasets can carry
        // encodings this reader deliberately rejects, and skipping is cheaper.
        if let Some(class) = crate::meta_sop_class(&bytes) {
            let wanted = [crate::CT, crate::RT_STRUCT, crate::RT_PLAN, crate::RT_DOSE];
            // Exact match — RTDOSE ...481.2 is a substring of the
            // second-generation RT UIDs (...481.20-481.25), which this
            // reader deliberately rejects inside.
            if !wanted.contains(&class.as_str()) {
                continue;
            }
        }
        let elems = parse_file(&bytes).map_err(|err| format!("{}: {err}", file.display()))?;
        let class = text_tag(&elems, tag(0x0008, 0x0016));
        let frame = text_tag(&elems, tag(0x0020, 0x0052));
        if class == "1.2.840.10008.5.1.4.1.1.2" {
            // A study can carry several CT series over one frame of
            // reference — merging them stacks two reconstructions into one
            // volume. Bucket by SeriesInstanceUID and keep the largest below.
            let uid = text_tag(&elems, tag(0x0020, 0x000E));
            identities.entry(uid.clone()).or_insert_with(|| {
                (
                    text_tag(&elems, tag(0x0010, 0x0020)),
                    text_tag(&elems, tag(0x0010, 0x0010)),
                    text_tag(&elems, tag(0x0020, 0x000D)),
                )
            });
            let position = text_tag(&elems, tag(0x0018, 0x5100));
            if !position.is_empty() {
                ct_positions.entry(uid.clone()).or_insert(position);
            }
            let (sops, bucket) = ct_by_series.entry(uid).or_default();
            sops.push(text_tag(&elems, tag(0x0008, 0x0018)));
            for slice in image_slices(&elems, 1.0)? {
                bucket.push((frame.clone(), slice));
            }
        } else if class == "1.2.840.10008.5.1.4.1.1.481.8" {
            // Parsed after the loop — the CT position fallback is only
            // known once every file has been seen.
            let sop_uid = text_tag(&elems, tag(0x0008, 0x0018));
            plan_raw = Some((elems, frame, sop_uid, file.clone()));
        } else if class == "1.2.840.10008.5.1.4.1.1.481.3" {
            if structures_uid.is_empty() {
                structures_uid = text_tag(&elems, tag(0x0008, 0x0018));
                structures_frame = frame.clone();
                structures =
                    read_structures(&elems).map_err(|err| format!("{}: {err}", file.display()))?;
            }
        } else if class == "1.2.840.10008.5.1.4.1.1.481.2" {
            if dose_seen {
                // A standard export holds per-beam plus fraction doses on the
                // same grid — stacking them either errors on duplicate z or
                // silently sums the wrong object. Keep the first dose object.
                continue;
            }
            dose_seen = true;
            // Type 1 required — absent means raw stored counts would read
            // as Gy, off by the scale's ~1e-5.
            let scale =
                float_tag(&elems, tag(0x3004, 0x000E)).ok_or("dose grid scaling is missing")?;
            // A zero scale decodes the whole grid to all-zero Gy and a
            // negative one inverts it — finite is not enough.
            if !scale.is_finite() || scale <= 0.0 {
                return Err("dose grid scaling is not a positive number".into());
            }
            if dose_type.is_empty() {
                dose_type = text_tag(&elems, tag(0x3004, 0x0004));
                summation = text_tag(&elems, tag(0x3004, 0x000A));
            }
            for slice in image_slices(&elems, scale)? {
                dose_slices.push((frame.clone(), slice));
            }
        }
    }
    // A CT file without a SeriesInstanceUID cannot be isolated — treat it as
    // part of the chosen series so a frame-of-reference conflict still fails.
    let (orphan_sops, orphans) = ct_by_series.remove("").unwrap_or_default();
    let mut ct_slices = Vec::new();
    let mut best = 0usize;
    for (uid, (sops, slices)) in ct_by_series {
        if slices.len() > best {
            best = slices.len();
            series_uid = uid;
            sop_uids = sops;
            ct_slices = slices;
        }
    }
    sop_uids.extend(orphan_sops);
    ct_slices.extend(orphans);
    let ct_position = ct_positions
        .remove(&series_uid)
        .or_else(|| ct_positions.remove(""))
        .unwrap_or_default();
    let mut plan = None;
    if let Some((elems, frame, sop_uid, plan_file)) = plan_raw {
        let mut parsed = read_plan(&elems, frame, &ct_position)
            .map_err(|err| format!("{}: {err}", plan_file.display()))?;
        parsed.sop_uid = sop_uid;
        plan = Some(parsed);
    }
    if let Some((id, name, study)) = identities.get(&series_uid) {
        patient_id = id.clone();
        patient_name = name.clone();
        study_uid = study.clone();
    }
    if ct_slices.is_empty() {
        return Err("the study has no CT".into());
    }
    // Frame of Reference is Type 2 — legitimately empty. Only a real
    // conflict between non-empty frames fails the load.
    let frame = ct_slices
        .iter()
        .map(|(frame, _)| frame.clone())
        .find(|frame| !frame.is_empty())
        .unwrap_or_default();
    if ct_slices
        .iter()
        .any(|(got, _)| !got.is_empty() && got != &frame)
    {
        return Err("CT slices do not share one frame of reference".into());
    }
    let plan = plan.unwrap_or(Plan {
        beams: Vec::new(),
        fractions: 1,
        prescription: f32::NAN,
        frame: frame.clone(),
        sop_uid: String::new(),
        label: String::new(),
    });
    if !plan.frame.is_empty() && plan.frame != frame {
        return Err("the plan frame of reference does not match the CT".into());
    }
    if dose_slices
        .iter()
        .any(|(got, _)| !got.is_empty() && got != &frame)
    {
        return Err("the dose frame of reference does not match the CT".into());
    }
    if !structures_frame.is_empty() && structures_frame != frame {
        return Err("the structures frame of reference does not match the CT".into());
    }
    let mut slices: Vec<_> = ct_slices.into_iter().map(|(_, slice)| slice).collect();
    let mut ct = stack(&mut slices)?;
    ct.position = ct_position.clone();
    ct.frame = frame.clone();
    ct.patient_id = patient_id;
    ct.patient_name = patient_name;
    ct.study_uid = study_uid;
    ct.series_uid = series_uid;
    ct.sop_uids = sop_uids;
    let tps = if dose_slices.is_empty() {
        None
    } else {
        let mut slices: Vec<_> = dose_slices.into_iter().map(|(_, slice)| slice).collect();
        let mut volume = stack(&mut slices)?;
        volume.position = ct_position.clone();
        volume.frame = frame.clone();
        volume.dose_type = dose_type;
        volume.summation = summation;
        Some(volume)
    };
    Ok(PatientStudy {
        frame,
        ct,
        beams: plan.beams,
        fractions: plan.fractions,
        structures,
        tps,
        prescription: plan.prescription,
        plan_uid: plan.sop_uid,
        plan_label: plan.label,
        structures_uid,
    })
}

/// Spots from one RT Ion file. Charge is the raw meterset weight, not the scaled MU.
pub fn read_ion_plan(path: &Path) -> Result<IonPlan, String> {
    // An RT plan is a few MB — bound the read before the Elem tree doubles it.
    let mut file =
        fs::File::open(path).map_err(|err| format!("Could not read DICOM plan file: {err}"))?;
    let mut bytes = Vec::new();
    std::io::Read::take(&mut file, 256 * 1024 * 1024 + 1)
        .read_to_end(&mut bytes)
        .map_err(|err| format!("Could not read DICOM plan file: {err}"))?;
    if bytes.len() > 256 * 1024 * 1024 {
        return Err("DICOM plan file exceeds 256 MiB".into());
    }
    if bytes.len() < 132 || &bytes[128..132] != b"DICM" {
        return Err(format!(
            "Could not read DICOM plan file: {}",
            path.display()
        ));
    }
    let elems = parse_file(&bytes)?;
    // Wrong file kind — a CT or RTDOSE must not read as an empty ion plan.
    let sop = text_tag(&elems, tag(0x0008, 0x0016));
    if !sop.contains("1.2.840.10008.5.1.4.1.1.481.8") {
        return Err("not an RT Ion Plan storage object".into());
    }
    let label = text_tag(&elems, tag(0x300A, 0x0002));
    let mut spots = Vec::new();
    let mut plan_index = 0i32;
    for beam in seq(&elems, tag(0x300A, 0x03A2)) {
        if text_tag(beam, tag(0x300A, 0x00CE)) == "SETUP" {
            continue;
        }
        let mut energy = 0.0f32;
        for cp in seq(beam, tag(0x300A, 0x03A8)) {
            energy = float_tag(cp, tag(0x300A, 0x0114))
                .filter(|value| value.is_finite())
                .unwrap_or(energy);
            let n = int_tag(cp, tag(0x300A, 0x0392)).unwrap_or(0).max(0) as usize;
            let xy = floats_tag(cp, tag(0x300A, 0x0394));
            if n == 0 {
                continue;
            }
            if xy.is_empty() {
                // Declared spots with no usable position map - skipping
                // would deliver the beam without a whole layer.
                return Err("scan spot position map is missing or corrupt".into());
            }
            let weights = floats_tag(cp, tag(0x300A, 0x0396));
            let has_weights = elem(cp, tag(0x300A, 0x0396)).is_some();
            let size = floats_tag(cp, tag(0x300A, 0x0398));
            let limit = n.min(xy.len() / 2);
            // A declared layer whose map holds fewer than two values is the
            // same corrupt-map failure read_plan rejects — skipping it
            // would silently drop the energy layer.
            if limit == 0 {
                return Err("scan spot position map is missing or corrupt".into());
            }
            // The weights tag must hold one entry per spot position — a
            // short array would silently drop the tail of the beam.
            if has_weights && weights.len() < limit {
                return Err("meterset weights do not match the spot count".into());
            }
            // Scan Mode is Type 1C only for MODULATED beams — absent on a
            // MODULATED beam is the loud error read_plan already makes; on a
            // UNIFORM beam every spot gets an equal share. An absent/unknown
            // mode with no weight map cannot pick between them — refuse.
            if !has_weights && limit > 0 {
                match text_tag(beam, tag(0x300A, 0x0308)).as_str() {
                    "UNIFORM" => {}
                    "MODULATED" => {
                        return Err("meterset weights are missing".into());
                    }
                    _ => return Err("a treatment beam has no scan mode".into()),
                }
            }
            for i in 0..limit {
                let mu = if has_weights {
                    weights.get(i).copied().unwrap_or(f32::NAN)
                } else {
                    // UNIFORM — an equal weight share per spot.
                    1.0
                };
                // ScanningSpotSize is 2·n (x,y) pairs in spot order; a lone
                // pair applies to every spot.
                let (size_x, size_y) = if size.len() >= 2 * limit {
                    (size[2 * i], size[2 * i + 1])
                } else if size.len() >= 2 {
                    (size[0], size[1])
                } else {
                    (f32::NAN, f32::NAN)
                };
                let index = plan_index;
                plan_index += 1;
                let (x, y) = (xy[i * 2], xy[i * 2 + 1]);
                // Position gets the same finite check as energy — a NaN spot
                // would reach the transport and the exported plan CSV.
                // A spot-bearing control point before any nominal energy is
                // corrupt input — silently skipping it drops the whole beam.
                if !energy.is_finite() || energy <= 0.0 {
                    return Err("a treatment beam has no nominal energy".into());
                }
                if (has_weights && (!mu.is_finite() || mu <= 0.0))
                    || !x.is_finite()
                    || !y.is_finite()
                {
                    continue;
                }
                spots.push(IonPlanSpot {
                    x,
                    y,
                    energy,
                    charge: mu,
                    size_x,
                    size_y,
                    plan_index: index,
                });
            }
        }
    }
    Ok(IonPlan { label, spots })
}

/// Rotation from the IEC 61217 gantry frame into patient LPS, row-major.
pub fn gantry_to_patient(
    gantry_deg: f32,
    couch_deg: f32,
    position: &str,
) -> Result<[f32; 9], String> {
    let support = match position.to_ascii_uppercase().as_str() {
        "HFS" => [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0],
        "HFP" => [-1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        "FFS" => [-1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 0.0],
        "FFP" => [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0],
        other => {
            return Err(format!(
                "patient position {other} is not HFS, HFP, FFS, or FFP"
            ))
        }
    };
    let couch = rot_z(-couch_deg);
    let gantry = rot_y(gantry_deg);
    Ok(mul(support, mul(couch, gantry)))
}

const WATER_DENSITY: &str =
    include_str!("../../../assets/mcsquare/Scanners/Water_Phantom/HU_Density_Conversion.txt");
const WATER_MATERIAL: &str =
    include_str!("../../../assets/mcsquare/Scanners/Water_Phantom/HU_Material_Conversion.txt");
const SOLID_DENSITY: &str =
    include_str!("../../../assets/mcsquare/Scanners/SolidWater_Phantom/HU_Density_Conversion.txt");
const SOLID_MATERIAL: &str =
    include_str!("../../../assets/mcsquare/Scanners/SolidWater_Phantom/HU_Material_Conversion.txt");

/// Density from a named scanner curve. `default` keeps the built-in table.
pub fn scanner_density(scanner: &str, hu: f32) -> f32 {
    match scanner {
        "water" => density_from(&pairs(WATER_DENSITY), hu),
        "solid" => density_from(&pairs(SOLID_DENSITY), hu),
        _ => hu_density(hu),
    }
}

/// Material id from a named scanner curve. `default` keeps the built-in table.
pub fn scanner_label(scanner: &str, hu: f32) -> i32 {
    match scanner {
        "water" => label_from(&pairs(WATER_MATERIAL), hu),
        "solid" => label_from(&pairs(SOLID_MATERIAL), hu),
        _ => hu_label(hu),
    }
}

fn density_from(table: &(Vec<f32>, Vec<f32>), hu: f32) -> f32 {
    if table.0.len() < 2 {
        return hu_density(hu);
    }
    let i = below(&table.0, hu).clamp(0, table.0.len() - 2);
    let (x0, x1) = (table.0[i], table.0[i + 1]);
    let (y0, y1) = (table.1[i], table.1[i + 1]);
    let density = (y1 - y0) / (x1 - x0).max(1e-6) * (hu - x0) + y0;
    if density <= 0.0 {
        1e-6
    } else {
        density
    }
}

fn label_from(table: &(Vec<f32>, Vec<f32>), hu: f32) -> i32 {
    if table.0.is_empty() {
        return hu_label(hu);
    }
    let i = below(&table.0, hu).clamp(0, table.1.len() - 1);
    table.1[i] as i32
}

pub fn hu_density(hu: f32) -> f32 {
    let table = density_table();
    let i = below(&table.0, hu).clamp(0, table.0.len() - 2);
    let (x0, x1) = (table.0[i], table.0[i + 1]);
    let (y0, y1) = (table.1[i], table.1[i + 1]);
    let density = (y1 - y0) / (x1 - x0) * (hu - x0) + y0;
    if density <= 0.0 {
        1e-6
    } else {
        density
    }
}

pub fn hu_label(hu: f32) -> i32 {
    let table = material_table();
    let i = below(&table.0, hu).clamp(0, table.1.len() - 1);
    table.1[i]
}

/// A 4×4×2 water CT, one HFS beam, a PTV square, and a flat 1 Gy TPS dose.
pub fn write_water_study(dir: &Path) -> Result<(), String> {
    fs::create_dir_all(dir).map_err(|err| err.to_string())?;
    let frame = "1.2.3.4.5";
    for (index, z) in [0.0f32, 1.0].into_iter().enumerate() {
        let mut bytes = preamble();
        let sop = uid();
        file_meta(&mut bytes, "1.2.840.10008.5.1.4.1.1.2", &sop)?;
        ui(&mut bytes, 0x0008, 0x0016, "1.2.840.10008.5.1.4.1.1.2")?;
        ui(&mut bytes, 0x0008, 0x0018, &sop)?;
        cs(&mut bytes, 0x0008, 0x0060, "CT")?;
        ui(&mut bytes, 0x0020, 0x000D, "1.2.3")?;
        ui(&mut bytes, 0x0020, 0x000E, "1.2.3.1")?;
        ds(&mut bytes, 0x0020, 0x0032, &[0.0, 0.0, z])?;
        ds(&mut bytes, 0x0020, 0x0037, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0])?;
        ui(&mut bytes, 0x0020, 0x0052, frame)?;
        us(&mut bytes, 0x0028, 0x0010, 4)?;
        us(&mut bytes, 0x0028, 0x0011, 4)?;
        ds(&mut bytes, 0x0028, 0x0030, &[1.0, 1.0])?;
        ds(&mut bytes, 0x0028, 0x1052, &[0.0])?;
        ds(&mut bytes, 0x0028, 0x1053, &[1.0])?;
        ow(&mut bytes, 0x7FE0, 0x0010, &[0u16; 16])?;
        scan_kit_core::atomic_write(&dir.join(format!("ct{index}.dcm")), &bytes)
            .map_err(|err| err.to_string())?;
    }
    let mut plan = preamble();
    let plan_sop = uid();
    file_meta(&mut plan, RT_ION_PLAN_CLASS, &plan_sop)?;
    ui(&mut plan, 0x0008, 0x0016, RT_ION_PLAN_CLASS)?;
    ui(&mut plan, 0x0008, 0x0018, &plan_sop)?;
    cs(&mut plan, 0x0008, 0x0060, "RTPLAN")?;
    ui(&mut plan, 0x0020, 0x0052, frame)?;
    let setup = {
        let mut item = Vec::new();
        cs(&mut item, 0x0018, 0x5100, "HFS")?;
        is(&mut item, 0x300A, 0x0182, 1)?;
        item
    };
    let referenced = {
        let mut item = Vec::new();
        ds(&mut item, 0x300A, 0x0086, &[1.0])?;
        is(&mut item, 0x300C, 0x0006, 1)?;
        item
    };
    let fraction = {
        let mut item = Vec::new();
        is(&mut item, 0x300A, 0x0078, 1)?;
        sq(&mut item, 0x300C, 0x0004, &[referenced])?;
        item
    };
    sq(&mut plan, 0x300A, 0x0070, &[fraction])?;
    sq(&mut plan, 0x300A, 0x0180, &[setup])?;
    let control = {
        let mut item = Vec::new();
        is(&mut item, 0x300A, 0x0112, 1)?;
        ds(&mut item, 0x300A, 0x0114, &[100.0])?;
        ds(&mut item, 0x300A, 0x011E, &[0.0])?;
        ds(&mut item, 0x300A, 0x0122, &[0.0])?;
        ds(&mut item, 0x300A, 0x012C, &[1.5, 1.5, 0.5])?;
        is(&mut item, 0x300A, 0x0392, 1)?;
        fl(&mut item, 0x300A, 0x0394, &[0.0, 0.0])?;
        fl(&mut item, 0x300A, 0x0396, &[1.0])?;
        item
    };
    let beam = {
        let mut item = Vec::new();
        is(&mut item, 0x300A, 0x00C0, 1)?;
        lo(&mut item, 0x300A, 0x00C2, "B1")?;
        cs(&mut item, 0x300A, 0x00CE, "TREATMENT")?;
        ds(&mut item, 0x300A, 0x010E, &[1.0])?;
        cs(&mut item, 0x300A, 0x0308, "MODULATED")?;
        sq(&mut item, 0x300A, 0x03A8, &[control])?;
        is(&mut item, 0x300C, 0x006A, 1)?;
        item
    };
    sq(&mut plan, 0x300A, 0x03A2, &[beam])?;
    scan_kit_core::atomic_write(&dir.join("plan.dcm"), &plan).map_err(|err| err.to_string())?;
    let roi = {
        let mut item = Vec::new();
        is(&mut item, 0x3006, 0x0022, 1)?;
        lo(&mut item, 0x3006, 0x0026, "PTV")?;
        item
    };
    let contour = {
        let mut item = Vec::new();
        is(&mut item, 0x3006, 0x0046, 4)?;
        ds(
            &mut item,
            0x3006,
            0x0050,
            &[0.0, 0.0, 0.5, 4.0, 0.0, 0.5, 4.0, 4.0, 0.5, 0.0, 4.0, 0.5],
        )?;
        item
    };
    let roi_contour = {
        let mut item = Vec::new();
        sq(&mut item, 0x3006, 0x0040, &[contour])?;
        is(&mut item, 0x3006, 0x0084, 1)?;
        item
    };
    let mut structures = preamble();
    let struct_sop = uid();
    file_meta(
        &mut structures,
        "1.2.840.10008.5.1.4.1.1.481.3",
        &struct_sop,
    )?;
    ui(
        &mut structures,
        0x0008,
        0x0016,
        "1.2.840.10008.5.1.4.1.1.481.3",
    )?;
    ui(&mut structures, 0x0008, 0x0018, &struct_sop)?;
    ui(&mut structures, 0x0020, 0x0052, frame)?;
    let observation = {
        let mut item = Vec::new();
        is(&mut item, 0x3006, 0x0084, 1)?;
        cs(&mut item, 0x3006, 0x00A4, "PTV")?;
        item
    };
    sq(&mut structures, 0x3006, 0x0020, &[roi])?;
    sq(&mut structures, 0x3006, 0x0039, &[roi_contour])?;
    // RT ROI Observations is Type 1 — without it the set has contours but
    // no interpreted semantics.
    sq(&mut structures, 0x3006, 0x0080, &[observation])?;
    scan_kit_core::atomic_write(&dir.join("struct.dcm"), &structures)
        .map_err(|err| err.to_string())?;
    let mut dose = preamble();
    let dose_sop = uid();
    file_meta(&mut dose, RT_DOSE_CLASS, &dose_sop)?;
    ui(&mut dose, 0x0008, 0x0016, RT_DOSE_CLASS)?;
    ui(&mut dose, 0x0008, 0x0018, &dose_sop)?;
    cs(&mut dose, 0x0008, 0x0060, "RTDOSE")?;
    ui(&mut dose, 0x0020, 0x0052, frame)?;
    ds(&mut dose, 0x0020, 0x0032, &[0.0, 0.0, 0.0])?;
    ds(&mut dose, 0x0020, 0x0037, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0])?;
    us(&mut dose, 0x0028, 0x0010, 4)?;
    us(&mut dose, 0x0028, 0x0011, 4)?;
    ds(&mut dose, 0x0028, 0x0030, &[1.0, 1.0])?;
    ds(&mut dose, 0x3004, 0x000C, &[0.0, 1.0])?;
    ds(&mut dose, 0x3004, 0x000E, &[0.01])?;
    ow(&mut dose, 0x7FE0, 0x0010, &[100u16; 32])?;
    scan_kit_core::atomic_write(&dir.join("dose.dcm"), &dose).map_err(|err| err.to_string())?;
    Ok(())
}

struct Slice {
    origin: [f32; 3],
    thickness: f32,
    spacing_x: f32,
    spacing_y: f32,
    cols: usize,
    rows: usize,
    values: Vec<f32>,
}

fn image_slices(elems: &[Elem], scale: f32) -> Result<Vec<Slice>, String> {
    // int_tag can text-parse a 2-byte US payload whose bytes spell an ASCII
    // number (e.g. value 12589 = "-1") — reject non-positive results rather
    // than casting them to usize::MAX.
    let rows = int_tag(elems, tag(0x0028, 0x0010)).unwrap_or(0).max(0) as usize;
    let cols = int_tag(elems, tag(0x0028, 0x0011)).unwrap_or(0).max(0) as usize;
    let spacing = floats_tag(elems, tag(0x0028, 0x0030));
    let origin = floats_tag(elems, tag(0x0020, 0x0032));
    // An absent rescale tag defaults; a present-but-corrupt one is an error,
    // not a silent 1.0.
    let slope = match float_tag(elems, tag(0x0028, 0x1053)) {
        Some(value) => value,
        None if elem(elems, tag(0x0028, 0x1053)).is_none() => 1.0,
        None => return Err("corrupt rescale slope".into()),
    };
    let intercept = match float_tag(elems, tag(0x0028, 0x1052)) {
        Some(value) => value,
        None if elem(elems, tag(0x0028, 0x1052)).is_none() => 0.0,
        None => return Err("corrupt rescale intercept".into()),
    };
    let pixels = elem(elems, tag(0x7FE0, 0x0010)).ok_or("missing pixels")?;
    if rows == 0 || cols == 0 || origin.len() < 3 || spacing.len() < 2 {
        return Err("CT geometry is incomplete".into());
    }
    if !(spacing[0].is_finite() && spacing[1].is_finite() && spacing[0] > 0.0 && spacing[1] > 0.0) {
        return Err("CT pixel spacing is not positive".into());
    }
    if origin.iter().any(|value| !value.is_finite()) {
        return Err("CT image position is not finite".into());
    }
    if !slope.is_finite() || !intercept.is_finite() {
        return Err("CT rescale slope/intercept is not finite".into());
    }
    // Image Orientation is the direction of increasing column, then increasing
    // row. The dose grid is patient +X along columns and +Y along rows, so an
    // axial prone or feet-first image is flipped into that order here.
    let (column_sign, row_sign) = axial_signs(elems)?;
    let normal_sign = column_sign * row_sign;
    let bits = int_tag(elems, tag(0x0028, 0x0100)).unwrap_or(16);
    // Pixel Representation 1 stores two's-complement samples. Reading those as
    // unsigned turns a negative HU into a huge positive one.
    let signed = int_tag(elems, tag(0x0028, 0x0103)).unwrap_or(0) == 1;
    // High Bit marks the sign bit of the stored value — a 12-bit signed
    // sample in a 16-bit slot reads as a large positive without extension.
    // bits - 1 underflows i32 on a corrupt Bits Allocated — clamp first.
    let high_bit = i64::from(
        int_tag(elems, tag(0x0028, 0x0102))
            .unwrap_or_else(|| bits.max(1) - 1)
            .clamp(0, 62),
    );
    let sample_bytes = if bits == 8 {
        1
    } else if bits > 16 {
        4
    } else {
        2
    };
    if !pixels.bytes.len().is_multiple_of(sample_bytes) {
        // A frame ending mid-sample would silently drop the dangling bytes.
        return Err("pixel data ends inside a sample".into());
    }
    let stored: Vec<f32> = if bits == 8 {
        pixels
            .bytes
            .iter()
            .map(|byte| {
                let raw = if signed {
                    sign_extend(i64::from(*byte), high_bit.min(7)) as f32
                } else {
                    *byte as f32
                };
                (raw * slope + intercept) * scale
            })
            .collect()
    } else if bits > 16 {
        pixels
            .bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| {
                let raw = if signed {
                    sign_extend(i64::from(u32::from_le_bytes(*chunk)), high_bit) as f64
                } else {
                    u32::from_le_bytes(*chunk) as f64
                };
                (raw * slope as f64 + intercept as f64) as f32 * scale
            })
            .collect()
    } else {
        pixels
            .bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|chunk| {
                let raw = if signed {
                    sign_extend(i64::from(u16::from_le_bytes(*chunk)), high_bit) as f32
                } else {
                    u16::from_le_bytes(*chunk) as f32
                };
                (raw * slope + intercept) * scale
            })
            .collect()
    };
    if rows == 0 || cols == 0 {
        return Err("missing pixel dimensions".into());
    }
    let Some(plane) = rows.checked_mul(cols).filter(|plane| *plane > 0) else {
        return Err("pixel plane overflows".into());
    };
    if stored.len() < plane {
        return Err("pixel data is short".into());
    }
    // A trailing partial frame is a truncated pixel element, not a shorter
    // volume — error rather than silently dropping it.
    if !stored.len().is_multiple_of(plane) {
        return Err("pixel data ends inside a frame".into());
    }
    let frames = stored.len() / plane;
    // Each frame materializes a Slice + heap Vec — bound the count before
    // the loop so a 1x1-plane crafted file can't amplify ~50x into OOM.
    const MAX_SLICES: usize = 100_000;
    if frames > MAX_SLICES {
        return Err(format!("pixel data claims {frames} frames"));
    }
    let declared = int_tag(elems, tag(0x0028, 0x0008))
        .map(|count| count.max(0) as usize)
        .unwrap_or(frames);
    if declared != frames {
        return Err("pixel data does not match the declared frame count".into());
    }
    let offsets = floats_tag(elems, tag(0x3004, 0x000C));
    if frames > 1 && offsets.is_empty() {
        // SliceThickness is the collimation, not the increment — inventing
        // frame*thickness positions would stack a plausible but wrong volume.
        return Err("multi-frame pixel data has no frame offset vector".into());
    }
    if !offsets.is_empty() && offsets.len() != frames {
        // A short offset vector would mix real positions with fabricated
        // frame*thickness ones.
        return Err("frame offsets do not match the frame count".into());
    }
    // Same absent-vs-corrupt split for Slice Thickness.
    let thickness = match float_tag(elems, tag(0x0018, 0x0050)) {
        Some(value) if value.is_finite() && value > 0.0 => value,
        None if elem(elems, tag(0x0018, 0x0050)).is_none() => 1.0,
        _ => return Err("corrupt slice thickness".into()),
    };
    // Grid Frame Offsets have two legal encodings: relative to the image
    // position (first offset zero), or absolute patient z (first offset
    // equal to the slice origin z with identity orientation). Reading
    // absolute values as relative shifts every frame by the origin z.
    let absolute_z = !offsets.is_empty() && offsets[0] != 0.0;
    if absolute_z && (offsets[0] != origin[2] || column_sign < 0.0 || row_sign < 0.0) {
        return Err("frame offsets are neither relative nor absolute z".into());
    }
    let mut slices = Vec::new();
    for frame in 0..frames {
        let mut along = offsets
            .get(frame)
            .copied()
            .unwrap_or(frame as f32 * thickness);
        if absolute_z {
            along -= origin[2];
        }
        if !along.is_finite() {
            return Err("frame offset is not finite".into());
        }
        let mut values = stored[frame * plane..(frame + 1) * plane].to_vec();
        let mut x = origin[0];
        let mut y = origin[1];
        if column_sign < 0.0 {
            reverse_columns(&mut values, cols, rows);
            x += (cols - 1) as f32 * column_sign * spacing[1];
        }
        if row_sign < 0.0 {
            reverse_rows(&mut values, cols, rows);
            y += (rows - 1) as f32 * row_sign * spacing[0];
        }
        slices.push(Slice {
            origin: [x, y, origin[2] + along * normal_sign],
            thickness,
            spacing_x: spacing[1],
            spacing_y: spacing[0],
            cols,
            rows,
            values,
        });
    }
    Ok(slices)
}

fn stack(slices: &mut [Slice]) -> Result<CtVolume, String> {
    if slices.is_empty() {
        return Err("the study has no slices".into());
    }
    // A malformed position tag ("nan" parses as f32::NAN) must not crash a
    // study load; equal order keeps the row and the dz check reports it.
    slices.sort_by(|a, b| {
        a.origin[2]
            .partial_cmp(&b.origin[2])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let first = &slices[0];
    let dz = if slices.len() == 1 {
        slices[0].thickness
    } else {
        slices[1].origin[2] - slices[0].origin[2]
    };
    // 0.01 mm is tighter than a real slice gap and looser than DICOM DS rounding.
    let uniform = slices.windows(2).all(|pair| {
        let step = pair[1].origin[2] - pair[0].origin[2];
        (step - dz).abs() <= 0.01
    });
    let spacing_matches = slices.iter().all(|slice| {
        (slice.spacing_x - first.spacing_x).abs() <= 0.01
            && (slice.spacing_y - first.spacing_y).abs() <= 0.01
            && (slice.origin[0] - first.origin[0]).abs() <= 0.1
            && (slice.origin[1] - first.origin[1]).abs() <= 0.1
    });
    if dz <= 0.0 || dz.is_nan() || !uniform {
        return Err("CT slice spacing is not uniform".into());
    }
    if !spacing_matches {
        return Err("CT slices differ in pixel spacing or in-plane origin".into());
    }
    // Slices come from the file count — a crafted study dir could ask for a
    // multi-GB buffer. Bound before allocating.
    let Some(voxels) = first
        .cols
        .checked_mul(first.rows)
        .and_then(|plane| plane.checked_mul(slices.len()))
    else {
        return Err("CT volume is too large".into());
    };
    if voxels > (1 << 30) {
        return Err("CT volume exceeds the voxel bound".into());
    }
    let mut hu = vec![0.0; voxels];
    for (z, slice) in slices.iter().enumerate() {
        if slice.cols != first.cols || slice.rows != first.rows {
            return Err("CT slices differ in size".into());
        }
        for y in 0..slice.rows {
            for x in 0..slice.cols {
                hu[x + first.cols * (y + first.rows * z)] = slice.values[x + slice.cols * y];
            }
        }
    }
    Ok(CtVolume {
        frame: String::new(),
        origin: first.origin,
        spacing: [first.spacing_x, first.spacing_y, dz],
        shape: [first.cols, first.rows, slices.len()],
        hu,
        position: "HFS".into(),
        dose_type: String::new(),
        summation: String::new(),
        patient_id: String::new(),
        patient_name: String::new(),
        study_uid: String::new(),
        series_uid: String::new(),
        sop_uids: Vec::new(),
    })
}

struct Plan {
    beams: Vec<Beam>,
    fractions: i32,
    prescription: f32,
    frame: String,
    sop_uid: String,
    label: String,
}

fn read_plan(elems: &[Elem], frame: String, ct_position: &str) -> Result<Plan, String> {
    let mut setups = Vec::new();
    for item in seq(elems, tag(0x300A, 0x0180)) {
        setups.push((
            int_tag(item, tag(0x300A, 0x0182)).unwrap_or(1),
            text_tag(item, tag(0x0018, 0x5100)),
        ));
    }
    let mut fractions = 1;
    let mut metersets = Vec::new();
    let mut populated_groups = 0;
    for group in seq(elems, tag(0x300A, 0x0070)) {
        let beams_in_group = seq(group, tag(0x300C, 0x0004));
        if !beams_in_group.is_empty() {
            // A second populated group's metersets would misattribute via
            // the first-match find below, and its fraction count may differ
            // — refuse rather than pick one silently.
            populated_groups += 1;
            if populated_groups > 1 {
                return Err(
                    "more than one fraction group with beam references is not supported".into(),
                );
            }
            fractions = match int_tag(group, tag(0x300A, 0x0078)) {
                Some(value) if value > 0 => value,
                Some(_) => return Err("a fraction group has a non-positive fraction count".into()),
                None => 1,
            };
        }
        for beam in beams_in_group {
            metersets.push((
                int_tag(beam, tag(0x300C, 0x0006)).unwrap_or(0),
                float_tag(beam, tag(0x300A, 0x0086)).unwrap_or(0.0),
            ));
        }
    }
    let mut beams = Vec::new();
    for beam in seq(elems, tag(0x300A, 0x03A2)) {
        // SETUP beams are imaging or verification fields — they carry no dose
        // and no meterset, so a plan with setup fields would fail its check.
        if text_tag(beam, tag(0x300A, 0x00CE)) == "SETUP" {
            continue;
        }
        let number = int_tag(beam, tag(0x300A, 0x00C0)).unwrap_or(0);
        let meterset = metersets
            .iter()
            .find(|(id, _)| *id == number)
            .map(|(_, mu)| *mu)
            .unwrap_or(0.0);
        let final_weight = float_tag(beam, tag(0x300A, 0x010E)).unwrap_or(0.0);
        // is_finite rejects a NaN meterset, which <= 0.0 would let through.
        if !meterset.is_finite()
            || meterset <= 0.0
            || !final_weight.is_finite()
            || final_weight <= 0.0
        {
            return Err(format!("beam {number} has no meterset"));
        }
        let setup = int_tag(beam, tag(0x300C, 0x006A)).unwrap_or(1);
        // The setup sequence is optional — but an unresolvable or empty
        // position cannot default to "HFS": the support matrix would mirror
        // the beam geometry of an FFS patient. With no setup sequence at
        // all, the CT's own Patient Position is the fallback.
        let position = if setups.is_empty() {
            if ct_position.is_empty() {
                return Err(format!("beam {number} has no patient position"));
            }
            ct_position.to_owned()
        } else {
            setups
                .iter()
                .find(|(id, _)| *id == setup)
                .map(|(_, name)| name.clone())
                .filter(|name| !name.is_empty())
                .ok_or_else(|| format!("beam {number} has no patient position"))?
        };
        let mut shifter_ids = Vec::new();
        for item in seq(beam, tag(0x300A, 0x0314)) {
            let id = int_tag(item, tag(0x300A, 0x0316)).unwrap_or(0);
            let name = text_tag(item, tag(0x300A, 0x0318));
            shifter_ids.push((
                id,
                if name.is_empty() {
                    format!("RS{id}")
                } else {
                    name
                },
            ));
        }
        let mut energy = f32::NAN;
        let mut gantry = 0.0;
        let mut gantry_seen = false;
        let mut first_gantry = 0.0;
        let mut couch = 0.0;
        let mut couch_seen = false;
        let mut first_couch = 0.0;
        let mut iso = [0.0f32; 3];
        let mut iso_seen = false;
        let mut first_iso = [0.0f32; 3];
        let mut in_shifters: Vec<(i32, f32, f32)> = Vec::new();
        let mut spots = Vec::new();
        for cp in seq(beam, tag(0x300A, 0x03A8)) {
            // DS fields can carry "nan" — a non-finite value must not
            // replace the last good reading.
            energy = float_tag(cp, tag(0x300A, 0x0114))
                .filter(|value| value.is_finite())
                .unwrap_or(energy);
            // Spots carry only a beam index — every control point shares
            // this beam's record, so a mid-beam geometry change cannot be
            // represented. Reject it like the multi-shifter case below.
            if let Some(value) = float_tag(cp, tag(0x300A, 0x011E)).filter(|v| v.is_finite()) {
                // Anchor to the first reading so slow drift cannot slip past
                // the tolerance one control point at a time.
                if gantry_seen && (value - first_gantry).abs() > 1e-3 {
                    return Err(format!("beam {number} changes gantry angle mid-beam"));
                }
                if !gantry_seen {
                    first_gantry = value;
                }
                gantry = value;
                gantry_seen = true;
            }
            if let Some(value) = float_tag(cp, tag(0x300A, 0x0122)).filter(|v| v.is_finite()) {
                if couch_seen && (value - first_couch).abs() > 1e-3 {
                    return Err(format!("beam {number} changes couch angle mid-beam"));
                }
                if !couch_seen {
                    first_couch = value;
                }
                couch = value;
                couch_seen = true;
            }
            let iso_raw = floats_tag(cp, tag(0x300A, 0x012C));
            if iso_raw.len() >= 3 && iso_raw[..3].iter().all(|value| value.is_finite()) {
                let next = [iso_raw[0], iso_raw[1], iso_raw[2]];
                if iso_seen
                    && next
                        .iter()
                        .zip(first_iso)
                        .any(|(a, b)| (a - b).abs() > 1e-3)
                {
                    return Err(format!("beam {number} changes isocenter mid-beam"));
                }
                if !iso_seen {
                    first_iso = next;
                }
                iso = next;
                iso_seen = true;
            }
            for setting in seq(cp, tag(0x300A, 0x0360)) {
                let shifter_number = int_tag(setting, tag(0x300C, 0x0100)).unwrap_or(0);
                if !shifter_ids.iter().any(|(id, _)| *id == shifter_number) {
                    return Err(format!(
                        "beam {number}: range shifter {shifter_number} is not in its RangeShifterSequence"
                    ));
                }
                if text_tag(setting, tag(0x300A, 0x0362)) == "IN" {
                    let previous = in_shifters
                        .iter()
                        .find(|(id, _, _)| *id == shifter_number)
                        .copied();
                    // Some(NaN) would defeat the .or(previous) fallback — a
                    // corrupt DS must carry the previous setting, not BDL
                    // defaults downstream.
                    let wet = float_tag(setting, tag(0x300A, 0x0366))
                        .filter(|v| v.is_finite())
                        .or(previous.map(|(_, wet, _)| wet))
                        .unwrap_or(f32::NAN);
                    let distance = float_tag(setting, tag(0x300A, 0x0364))
                        .filter(|v| v.is_finite())
                        .or(previous.map(|(_, _, distance)| distance))
                        .unwrap_or(f32::NAN);
                    in_shifters.retain(|(id, _, _)| *id != shifter_number);
                    in_shifters.push((shifter_number, wet, distance));
                } else {
                    in_shifters.retain(|(id, _, _)| *id != shifter_number);
                }
            }
            // A negative or huge count must not hang the load; spots past the
            // position pairs have no x/y and would pile up at the origin.
            let n = int_tag(cp, tag(0x300A, 0x0392)).unwrap_or(0).max(0) as usize;
            let xy = floats_tag(cp, tag(0x300A, 0x0394));
            let weights = floats_tag(cp, tag(0x300A, 0x0396));
            let has_weights = elem(cp, tag(0x300A, 0x0396)).is_some();
            if n == 0 {
                continue;
            }
            if !energy.is_finite() || energy <= 0.0 {
                // read_ion_plan errors on the same input - a beam's spots
                // must not vanish silently in the study path.
                return Err(format!(
                    "beam {number}: a treatment beam has no nominal energy"
                ));
            }
            let limit = n.min(xy.len() / 2);
            if limit == 0 {
                // n > 0 spots declared but no usable positions - a silent
                // continue would drop the layer from the computed dose.
                return Err(format!(
                    "beam {number}: scan spot position map is missing or corrupt"
                ));
            }
            if has_weights && weights.len() < limit {
                return Err(format!(
                    "beam {number}: meterset weights do not match the spot count"
                ));
            }
            // A modulated beam's weight map is Type 1C — absent, every spot
            // computes mu = 0 and the beam silently empties. UNIFORM means an
            // equal meterset share per spot; a missing mode cannot pick
            // between them, so it errors too.
            if !has_weights && limit > 0 {
                match text_tag(beam, tag(0x300A, 0x0308)).as_str() {
                    "UNIFORM" => {}
                    "MODULATED" => {
                        return Err(format!("beam {number}: meterset weights are missing"));
                    }
                    _ => {
                        return Err(format!("beam {number}: scan mode is missing"));
                    }
                }
            }
            if in_shifters.len() > 1 {
                return Err(format!(
                    "beam {number}: more than one range shifter IN is not supported"
                ));
            }
            let (shifter, wet_mm, shifter_distance_mm) = match in_shifters.first() {
                Some((id, wet, distance)) => (
                    shifter_ids
                        .iter()
                        .find(|(number, _)| number == id)
                        .map(|(_, name)| name.clone())
                        .unwrap_or_default(),
                    *wet,
                    *distance,
                ),
                None => (String::new(), 0.0, f32::NAN),
            };
            for i in 0..limit {
                let mu = if has_weights {
                    weights.get(i).copied().unwrap_or(0.0) * meterset / final_weight
                } else {
                    // UNIFORM — an equal share of the beam meterset.
                    meterset / limit as f32
                };
                let x = xy.get(i * 2).copied().unwrap_or(f32::NAN);
                let y = xy.get(i * 2 + 1).copied().unwrap_or(f32::NAN);
                // A non-finite position or meterset reaches the transport
                // and the plan CSV — reject it like a non-finite energy.
                if mu.is_finite() && mu > 0.0 && x.is_finite() && y.is_finite() {
                    spots.push(Spot {
                        energy,
                        x,
                        y,
                        mu,
                        shifter: shifter.clone(),
                        wet_mm,
                        shifter_distance_mm,
                        spot_size_mm: scanning_spot_mm_at(cp, i, limit),
                    });
                }
            }
        }
        if !gantry_seen || !couch_seen || !iso_seen {
            return Err(format!(
                "beam {number} has no isocenter or gantry/couch angle"
            ));
        }
        beams.push(Beam {
            number,
            name: text_tag(beam, tag(0x300A, 0x00C2)),
            gantry,
            couch,
            isocenter: iso,
            position,
            meterset,
            spots,
        });
    }
    // Target Prescription Dose lives in the Dose Reference Sequence's
    // TARGET item (300A,0026) — the %-of-prescription clinical goals need
    // it. NaN means the plan did not carry one.
    let prescription = seq(elems, tag(0x300A, 0x0010))
        .iter()
        .find(|item| text_tag(item, tag(0x300A, 0x0020)).eq_ignore_ascii_case("target"))
        .and_then(|item| float_tag(item, tag(0x300A, 0x0026)))
        .filter(|dose| dose.is_finite() && *dose > 0.0)
        .unwrap_or(f32::NAN);
    Ok(Plan {
        beams,
        fractions,
        prescription,
        frame,
        sop_uid: String::new(),
        label: text_tag(elems, tag(0x300A, 0x0002)),
    })
}

fn read_structures(elems: &[Elem]) -> Result<Vec<Structure>, String> {
    let mut names = Vec::new();
    for roi in seq(elems, tag(0x3006, 0x0020)) {
        names.push((
            int_tag(roi, tag(0x3006, 0x0022)).unwrap_or(0),
            text_tag(roi, tag(0x3006, 0x0026)),
        ));
    }
    let mut structures = Vec::new();
    for roi in seq(elems, tag(0x3006, 0x0039)) {
        let number = int_tag(roi, tag(0x3006, 0x0084)).unwrap_or(0);
        let name = names
            .iter()
            .find(|(id, _)| *id == number)
            .map(|(_, name)| name.clone())
            .unwrap_or_else(|| format!("ROI {number}"));
        let mut contour = Vec::new();
        for item in seq(roi, tag(0x3006, 0x0040)) {
            // An OPEN_PLANAR item would grow a phantom closing edge in the
            // even-odd test — only closed rings rasterize. A missing tag
            // defaults to closed (Type 3, writers commonly omit it).
            let kind = text_tag(item, tag(0x3006, 0x0042));
            // CLOSEDPLANAR_XOR is retired but still exported — it is a
            // closed ring, and the even-odd fill already implements XOR.
            if !kind.is_empty()
                && kind.trim() != "CLOSED_PLANAR"
                && kind.trim() != "CLOSEDPLANAR_XOR"
            {
                continue;
            }
            let data = floats_tag(item, tag(0x3006, 0x0050));
            // floats_tag is all-or-nothing — one corrupt component in the
            // DS yields empty and the whole contour slice would vanish
            // silently, shrinking the ROI in every downstream metric.
            if data.is_empty() && elem(item, tag(0x3006, 0x0050)).is_some() {
                return Err("corrupt ContourData — one bad value drops the slice".into());
            }
            let plane: Vec<[f32; 3]> = data
                .chunks(3)
                .filter(|point| point.len() == 3)
                .map(|point| [point[0], point[1], point[2]])
                .collect();
            if !plane.is_empty() {
                contour.push(plane);
            }
        }
        if !contour.is_empty() {
            structures.push(Structure { name, contour });
        }
    }
    Ok(structures)
}

/// True when the voxel center is inside the structure's axial contour.
pub fn inside_structure(
    study: &PatientStudy,
    structure: &Structure,
    x: usize,
    y: usize,
    z: usize,
) -> bool {
    let ct = &study.ct;
    let center = [
        ct.origin[0] + x as f32 * ct.spacing[0],
        ct.origin[1] + y as f32 * ct.spacing[1],
        ct.origin[2] + z as f32 * ct.spacing[2],
    ];
    // Even-odd parity only composes within one plane — bucket the ray counts
    // per contour z so distinct planes union instead of XOR-cancelling.
    let mut planes: Vec<(f32, i32)> = Vec::new();
    let mut plane_lo = f32::INFINITY;
    let mut plane_hi = f32::NEG_INFINITY;
    for contour in &structure.contour {
        let Some(z0) = contour.first().map(|point| point[2]) else {
            continue;
        };
        // A contour whose vertices straddle slices can't rasterize on one
        // plane — skip it rather than stamping it onto the wrong slice.
        let (z_lo, z_hi) = contour
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), p| {
                (lo.min(p[2]), hi.max(p[2]))
            });
        if !z0.is_finite() || z_hi - z_lo > ct.spacing[2] * 0.1 {
            continue;
        }
        plane_lo = plane_lo.min(z0);
        plane_hi = plane_hi.max(z0);
        let n = contour.len();
        let mut hits = 0;
        for i in 0..n {
            let a = contour[i];
            let b = contour[(i + 1) % n];
            if (a[1] > center[1]) != (b[1] > center[1]) {
                let t = (center[1] - a[1]) / (b[1] - a[1]);
                if a[0] + t * (b[0] - a[0]) > center[0] {
                    hits += 1;
                }
            }
        }
        match planes
            .iter_mut()
            .find(|(z, _)| (z0 - *z).abs() <= ct.spacing[2] * 0.1)
        {
            Some((_, total)) => *total += hits,
            None => planes.push((z0, hits)),
        }
    }
    // A contour stamps the slice its plane rounds to. A contour lying
    // between slices stamps only its nearest slice — a distance window
    // would admit it on both, painting a phantom slice into the mask.
    if planes.iter().any(|(z0, hits)| {
        ((z0 - ct.origin[2]) / ct.spacing[2]).round() as i64 == z as i64 && hits % 2 == 1
    }) {
        return true;
    }
    // ponytail: a contour pitch coarser than the slice pitch leaves
    // interior slices un-stamped — assign them the nearest plane inside
    // the structure's z extent (nearest-neighbour interpolation, never
    // extrapolation) or the mask stripes and volume_cc under-counts.
    // True polygon interpolation is the upgrade if a study needs it.
    if center[2] >= plane_lo && center[2] <= plane_hi {
        if let Some((_, hits)) = planes
            .iter()
            .min_by(|a, b| (a.0 - center[2]).abs().total_cmp(&(b.0 - center[2]).abs()))
        {
            return hits % 2 == 1;
        }
    }
    false
}

/// Column sign, then row sign, for an axial image. Missing orientation is the
/// identity used by files written before the tag was filled in.
/// Two's-complement sign extension from `high_bit` (zero-indexed). A word
/// whose high bit equals the allocation's top bit extends identically to a
/// plain signed cast, so the callers use this unconditionally when signed.
fn sign_extend(word: i64, high_bit: i64) -> i64 {
    // Bits above High Bit are sender-defined - mask them off first or a
    // sign-replicated writer (0xFFFF for 12-bit -1) reads as a huge positive.
    let word = word
        & if high_bit >= 62 {
            i64::MAX
        } else {
            (1_i64 << (high_bit + 1)) - 1
        };
    if word & (1_i64 << high_bit) != 0 {
        word - (1_i64 << (high_bit + 1))
    } else {
        word
    }
}

fn axial_signs(elems: &[Elem]) -> Result<(f32, f32), String> {
    let orientation = floats_tag(elems, tag(0x0020, 0x0037));
    if orientation.is_empty() {
        // Absent means HFS default; present-but-unparseable is a corrupt
        // orientation - identity would mirror the anatomy silently.
        if elem(elems, tag(0x0020, 0x0037)).is_some() {
            return Err("CT orientation is corrupt".into());
        }
        return Ok((1.0, 1.0));
    }
    if orientation.len() < 6 {
        return Err("CT orientation is not axial".into());
    }
    let column = unit_axis(&orientation[..3])?;
    let row = unit_axis(&orientation[3..6])?;
    if column.0 != 0 || row.0 != 1 {
        return Err("CT orientation is not axial".into());
    }
    Ok((column.1, row.1))
}

fn unit_axis(cosine: &[f32]) -> Result<(usize, f32), String> {
    let mut found = None;
    for (axis, component) in cosine.iter().enumerate() {
        if component.abs() < 1.0e-3 {
            continue;
        }
        if found.is_some() || (component.abs() - 1.0).abs() > 1.0e-3 {
            return Err("CT orientation is not axial".into());
        }
        found = Some((axis, component.signum()));
    }
    found.ok_or_else(|| "CT orientation is not axial".into())
}

fn reverse_columns(values: &mut [f32], cols: usize, rows: usize) {
    for row in 0..rows {
        values[row * cols..(row + 1) * cols].reverse();
    }
}

fn reverse_rows(values: &mut [f32], cols: usize, rows: usize) {
    for row in 0..rows / 2 {
        let low = row * cols;
        let high = (rows - 1 - row) * cols;
        for col in 0..cols {
            values.swap(low + col, high + col);
        }
    }
}

fn density_table() -> (Vec<f32>, Vec<f32>) {
    pairs(DENSITY)
}

fn material_table() -> (Vec<f32>, Vec<i32>) {
    let (hu, label) = pairs(MATERIALS);
    (hu, label.into_iter().map(|value| value as i32).collect())
}

fn pairs(text: &str) -> (Vec<f32>, Vec<f32>) {
    let mut x = Vec::new();
    let mut y = Vec::new();
    for line in text.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        let mut parts = line.split_whitespace();
        let Some(a) = parts.next().and_then(|v| v.parse().ok()) else {
            continue;
        };
        let Some(b) = parts.next().and_then(|v| v.parse().ok()) else {
            continue;
        };
        x.push(a);
        y.push(b);
    }
    (x, y)
}

fn below(values: &[f32], x: f32) -> usize {
    match values.iter().position(|value| *value >= x) {
        Some(0) => 0,
        Some(index) => index - 1,
        None => values.len().saturating_sub(1),
    }
}

fn rot_y(deg: f32) -> [f32; 9] {
    let (s, c) = deg.to_radians().sin_cos();
    [c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c]
}

fn rot_z(deg: f32) -> [f32; 9] {
    let (s, c) = deg.to_radians().sin_cos();
    [c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0]
}

fn mul(a: [f32; 9], b: [f32; 9]) -> [f32; 9] {
    let mut out = [0.0; 9];
    for row in 0..3 {
        for col in 0..3 {
            out[row * 3 + col] =
                a[row * 3] * b[col] + a[row * 3 + 1] * b[3 + col] + a[row * 3 + 2] * b[6 + col];
        }
    }
    out
}

fn tag(group: u16, element: u16) -> u32 {
    (u32::from(group) << 16) | u32::from(element)
}

fn parse_file(bytes: &[u8]) -> Result<Vec<Elem>, String> {
    let (offset, implicit) = crate::dataset_offset(bytes);
    parse_dataset_mode(&bytes[offset..], implicit?, 0)
}

#[cfg(test)]
fn parse_dataset(bytes: &[u8]) -> Result<Vec<Elem>, String> {
    parse_dataset_mode(bytes, false, 0)
}

fn parse_dataset_mode(bytes: &[u8], implicit: bool, depth: u32) -> Result<Vec<Elem>, String> {
    let mut elems = Vec::new();
    let mut cursor = 0;
    while cursor + 8 <= bytes.len() {
        let group = u16::from_le_bytes([bytes[cursor], bytes[cursor + 1]]);
        let element = u16::from_le_bytes([bytes[cursor + 2], bytes[cursor + 3]]);
        // Implicit VR writes a 4-byte length where explicit writes VR+length;
        // sequences are only known by the item marker their value starts with.
        let (length, header, vr) = if implicit {
            (
                u32::from_le_bytes([
                    bytes[cursor + 4],
                    bytes[cursor + 5],
                    bytes[cursor + 6],
                    bytes[cursor + 7],
                ]),
                8usize,
                [0u8; 2],
            )
        } else {
            let vr = [bytes[cursor + 4], bytes[cursor + 5]];
            let long = matches!(
                &vr,
                b"OB"
                    | b"OW"
                    | b"OF"
                    | b"SQ"
                    | b"UT"
                    | b"UN"
                    | b"OD"
                    | b"OL"
                    | b"OV"
                    | b"UC"
                    | b"UR"
            );
            let (length, header) = if long {
                if cursor + 12 > bytes.len() {
                    return Err("the dataset ends inside an element header".into());
                }
                (
                    u32::from_le_bytes([
                        bytes[cursor + 8],
                        bytes[cursor + 9],
                        bytes[cursor + 10],
                        bytes[cursor + 11],
                    ]),
                    12usize,
                )
            } else {
                (
                    u16::from_le_bytes([bytes[cursor + 6], bytes[cursor + 7]]) as u32,
                    8usize,
                )
            };
            (length, header, vr)
        };
        if length == u32::MAX {
            return Err("undefined-length DICOM sequences are not read".into());
        }
        let start = cursor + header;
        // checked_add like element_span — on wasm32 a u32 length could
        // wrap the add and pass the bounds check.
        let end = start
            .checked_add(length as usize)
            .ok_or("the dataset ends inside an element")?;
        if end > bytes.len() {
            return Err("the dataset ends inside an element".into());
        }
        let value = &bytes[start..end];
        // Implicit-VR sequences and explicit UN values are only recognizable
        // by the FFFE,E000 item tag at the head of the value; once the marker
        // is present a failed parse is a corrupt sequence, not plain bytes
        // that happen to share it — demoting would silently empty the SQ.
        let (items, seq) = if !implicit && vr == *b"SQ" {
            (parse_items(value, false, depth + 1)?, true)
        } else if (implicit || vr == *b"UN")
            && tag(group, element) != tag(0x7FE0, 0x0010)
            && looks_like_items(value)
        {
            (parse_items(value, implicit, depth + 1)?, true)
        } else {
            (Vec::new(), false)
        };
        elems.push(Elem {
            tag: tag(group, element),
            vr: if implicit { [0, 0] } else { vr },
            items,
            bytes: if seq { Vec::new() } else { value.to_vec() },
        });
        cursor = end;
    }
    // A cut mid-header leaves trailing bytes the loop never reads — NUL
    // padding is still tolerated, but a truncated tag is not silently dropped.
    if bytes[cursor..].iter().any(|byte| *byte != 0) {
        return Err("the dataset ends inside an element header".into());
    }
    Ok(elems)
}

/// First eight bytes of a defined-length sequence value: the FFFE,E000 Item
/// tag plus its declared length. Under implicit VR a binary element can
/// start with those same four bytes (e.g. two stored values 0xFFFE/0xE000)
/// — requiring the first item's length to fit inside the value shrinks the
/// false-positive window to near zero.
fn looks_like_items(value: &[u8]) -> bool {
    if value.len() < 8 || value[..4] != [0xFE, 0xFF, 0x00, 0xE0] {
        return false;
    }
    let item_len = u32::from_le_bytes([value[4], value[5], value[6], value[7]]) as usize;
    item_len == u32::MAX as usize || item_len <= value.len() - 8
}

fn parse_items(bytes: &[u8], implicit: bool, depth: u32) -> Result<Vec<Vec<Elem>>, String> {
    // Defined-length SQs nest ~20 bytes per level — a hostile file can
    // recurse tens of thousands deep and overflow the stack before any
    // recoverable Err. Same ceiling as xml_dom.
    if depth > 256 {
        return Err("sequence nesting is too deep".into());
    }
    let mut items = Vec::new();
    let mut cursor = 0;
    while cursor + 8 <= bytes.len() {
        let group = u16::from_le_bytes([bytes[cursor], bytes[cursor + 1]]);
        let element = u16::from_le_bytes([bytes[cursor + 2], bytes[cursor + 3]]);
        let length = u32::from_le_bytes([
            bytes[cursor + 4],
            bytes[cursor + 5],
            bytes[cursor + 6],
            bytes[cursor + 7],
        ]);
        if group == 0xFFFE && element == 0xE0DD {
            // A defined-length SQ may still carry a delimiter - consume it
            // or the non-zero tail check below rejects the whole sequence.
            cursor += 8;
            break;
        }
        if group != 0xFFFE || element != 0xE000 {
            return Err("a sequence item is missing its header".into());
        }
        if length == u32::MAX {
            return Err("undefined-length DICOM sequences are not read".into());
        }
        let start = cursor + 8;
        let end = start
            .checked_add(length as usize)
            .ok_or("a sequence item is short")?;
        if end > bytes.len() {
            return Err("a sequence item is short".into());
        }
        items.push(parse_dataset_mode(&bytes[start..end], implicit, depth)?);
        cursor = end;
    }
    // Same tail rule as parse_dataset_mode — a defined-length SQ that ends
    // mid-item-header drops the remainder silently otherwise.
    if bytes[cursor..].iter().any(|byte| *byte != 0) {
        return Err("the sequence ends inside an item header".into());
    }
    Ok(items)
}

fn elem(elems: &[Elem], tag: u32) -> Option<&Elem> {
    elems.iter().find(|elem| elem.tag == tag)
}

fn seq(elems: &[Elem], tag: u32) -> &[Vec<Elem>] {
    elem(elems, tag)
        .map(|elem| elem.items.as_slice())
        .unwrap_or(&[])
}

fn text_of(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes)
        .trim_matches(|c: char| c == '\0' || c.is_whitespace())
        .to_owned()
}

fn text_tag(elems: &[Elem], tag: u32) -> String {
    elem(elems, tag)
        .map(|elem| text_of(&elem.bytes))
        .unwrap_or_default()
}

fn floats_tag(elems: &[Elem], tag: u32) -> Vec<f32> {
    let Some(elem) = elem(elems, tag) else {
        return Vec::new();
    };
    // Known binary VRs decode as little-endian floats on the VR, not on the
    // payload's byte shape — an "AAAA" OF would otherwise read as DS text.
    if matches!(&elem.vr, b"FL" | b"OF") {
        return elem
            .bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect();
    }
    // FD/OD are the doubles — 300A spot position/meterset maps are spec'd
    // FD, so an explicit-VR plan reads f64s, not pairs of garbage f32s.
    if matches!(&elem.vr, b"FD" | b"OD") {
        return elem
            .bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|chunk| f64::from_le_bytes(*chunk) as f32)
            .collect();
    }
    let ds = |bytes: &[u8]| {
        // All-or-nothing: dropping one bad component shifts every later
        // value a slot — treat the corrupted value as absent instead.
        text_of(bytes)
            .split('\\')
            .map(|part| part.trim().parse().ok())
            .collect::<Option<Vec<f32>>>()
            .unwrap_or_default()
    };
    if elem.vr != [0, 0] {
        return ds(&elem.bytes);
    }
    // Implicit VR guesses from shape: OF payloads hold non-text bytes;
    // trailing NULs are DS padding, not a binary marker.
    let mut end = elem.bytes.len();
    while end > 0 && elem.bytes[end - 1] == 0 {
        end -= 1;
    }
    if (end == 0 && !elem.bytes.is_empty())
        || elem.bytes[..end]
            .iter()
            .any(|byte| !(byte.is_ascii_graphic() || *byte == b' '))
    {
        return elem
            .bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect();
    }
    ds(&elem.bytes)
}

fn scanning_spot_mm(cp: &[Elem]) -> f32 {
    let size = floats_tag(cp, tag(0x300A, 0x0398));
    if size.len() >= 2
        && size[0] > 0.0
        && size[1] > 0.0
        && size[0] < 500.0
        && size[1] < 500.0
        && size[0].is_finite()
        && size[1].is_finite()
    {
        (size[0] + size[1]) / 2.0
    } else {
        0.0
    }
}

/// ScanningSpotSize is a 2·n array of (x,y) FWHM pairs in spot order —
/// spot `i`'s pair is `size[2i]`/`size[2i+1]`, a lone pair covers all.
fn scanning_spot_mm_at(cp: &[Elem], i: usize, limit: usize) -> f32 {
    let size = floats_tag(cp, tag(0x300A, 0x0398));
    if size.len() >= 2 * limit {
        let (x, y) = (size[2 * i], size[2 * i + 1]);
        if x > 0.0 && y > 0.0 && x < 500.0 && y < 500.0 && x.is_finite() && y.is_finite() {
            return (x + y) / 2.0;
        }
        0.0
    } else {
        scanning_spot_mm(cp)
    }
}

fn float_tag(elems: &[Elem], tag: u32) -> Option<f32> {
    floats_tag(elems, tag).first().copied()
}

fn int_tag(elems: &[Elem], tag: u32) -> Option<i32> {
    let elem = elem(elems, tag)?;
    // Known integer VRs decode on the VR — a US value of 0x3839 ("98" as
    // bytes) must not ASCII-parse to 98.
    match (&elem.vr, elem.bytes.len()) {
        (b"US", 2) => {
            return Some(i32::from(u16::from_le_bytes([
                elem.bytes[0],
                elem.bytes[1],
            ])));
        }
        (b"SS", 2) => {
            return Some(i32::from(i16::from_le_bytes([
                elem.bytes[0],
                elem.bytes[1],
            ])));
        }
        (b"UL", 4) => {
            return i32::try_from(u32::from_le_bytes(elem.bytes[..4].try_into().ok()?)).ok();
        }
        (b"SL", 4) => {
            return Some(i32::from_le_bytes(elem.bytes[..4].try_into().ok()?));
        }
        _ => {}
    }
    // A NUL cannot appear in an IS value (IS pads with a space) — a 2-byte
    // payload holding one is binary US; value 50 = [0x32, 0x00] reads as
    // the ASCII digit "2" otherwise.
    if elem.bytes.len() == 2 && elem.bytes.contains(&0) {
        return Some(i32::from(u16::from_le_bytes([
            elem.bytes[0],
            elem.bytes[1],
        ])));
    }
    let text = text_of(&elem.bytes);
    if let Ok(value) = text.parse::<i32>() {
        return Some(value);
    }
    if elem.bytes.len() == 2 {
        return Some(i32::from(u16::from_le_bytes([
            elem.bytes[0],
            elem.bytes[1],
        ])));
    }
    None
}

pub(crate) fn preamble() -> Vec<u8> {
    let mut bytes = vec![0u8; 128];
    bytes.extend_from_slice(b"DICM");
    bytes
}

fn even(text: &str) -> Vec<u8> {
    let mut bytes = text.as_bytes().to_vec();
    if bytes.len() % 2 == 1 {
        bytes.push(b' ');
    }
    bytes
}

fn push_elem(
    bytes: &mut Vec<u8>,
    group: u16,
    slot: u16,
    vr: &[u8],
    value: &[u8],
) -> Result<(), String> {
    bytes.extend_from_slice(&group.to_le_bytes());
    bytes.extend_from_slice(&slot.to_le_bytes());
    bytes.extend_from_slice(vr);
    let long = matches!(vr, b"OW" | b"OB" | b"SQ" | b"OD");
    if long {
        bytes.extend_from_slice(&[0, 0]);
        // Exactly 2^32 bytes wraps the VL to 0 — a silent empty element.
        let len = u32::try_from(value.len())
            .map_err(|_| format!("{group:04X},{slot:04X} value over 4 GiB"))?;
        bytes.extend_from_slice(&len.to_le_bytes());
    } else {
        let len = u16::try_from(value.len())
            .map_err(|_| format!("{group:04X},{slot:04X} value over 64 KiB"))?;
        bytes.extend_from_slice(&len.to_le_bytes());
    }
    bytes.extend_from_slice(value);
    Ok(())
}

pub(crate) fn ui(bytes: &mut Vec<u8>, group: u16, element: u16, text: &str) -> Result<(), String> {
    let mut value = text.as_bytes().to_vec();
    if value.len() % 2 == 1 {
        value.push(0);
    }
    push_elem(bytes, group, element, b"UI", &value)
}

pub(crate) fn cs(bytes: &mut Vec<u8>, group: u16, element: u16, text: &str) -> Result<(), String> {
    push_elem(bytes, group, element, b"CS", &even(text))
}

pub(crate) fn lo(bytes: &mut Vec<u8>, group: u16, element: u16, text: &str) -> Result<(), String> {
    push_elem(bytes, group, element, b"LO", &even(text))
}

pub(crate) fn pn(bytes: &mut Vec<u8>, group: u16, element: u16, text: &str) -> Result<(), String> {
    push_elem(bytes, group, element, b"PN", &even(text))
}

pub(crate) fn sh(bytes: &mut Vec<u8>, group: u16, element: u16, text: &str) -> Result<(), String> {
    push_elem(bytes, group, element, b"SH", &even(text))
}

pub(crate) fn ds(
    bytes: &mut Vec<u8>,
    group: u16,
    element: u16,
    values: &[f32],
) -> Result<(), String> {
    let parts: Vec<String> = values
        .iter()
        .map(|value| {
            let plain = format!("{value:.6}");
            // {:.6} collapses small magnitudes to "0.000000" - a declared
            // zero geometry is worse than an exponential spelling.
            if *value != 0.0 && plain.parse::<f64>() == Ok(0.0) {
                format!("{value:.6e}")
            } else {
                plain
            }
        })
        .collect();
    let text = parts.join("\\");
    // A DS component is ≤16 bytes of the decimal grammar — "NaN"/"inf"/
    // "e+20" or an overlong literal is not encodable. The limit is per
    // component, not on the joined string.
    if !values.iter().all(|value| value.is_finite()) || parts.iter().any(|part| part.len() > 16) {
        return Err(format!("{group:04X},{element:04X} is not encodable as DS"));
    }
    push_elem(bytes, group, element, b"DS", &even(&text))
}

/// 300A spot position/meterset and source-distance tags are FL, not DS —
/// strict readers reject a DS encoding there.
pub(crate) fn fl(
    bytes: &mut Vec<u8>,
    group: u16,
    element: u16,
    values: &[f32],
) -> Result<(), String> {
    let mut raw = Vec::with_capacity(values.len() * 4);
    for value in values {
        raw.extend_from_slice(&value.to_le_bytes());
    }
    push_elem(bytes, group, element, b"FL", &raw)
}

pub(crate) fn is(bytes: &mut Vec<u8>, group: u16, element: u16, value: i32) -> Result<(), String> {
    push_elem(bytes, group, element, b"IS", &even(&value.to_string()))
}

pub(crate) fn us(bytes: &mut Vec<u8>, group: u16, element: u16, value: u16) -> Result<(), String> {
    push_elem(bytes, group, element, b"US", &value.to_le_bytes())
}

pub(crate) fn is_list(
    bytes: &mut Vec<u8>,
    group: u16,
    element: u16,
    values: &[u16],
) -> Result<(), String> {
    let text = values
        .iter()
        .map(u16::to_string)
        .collect::<Vec<_>>()
        .join("\\");
    push_elem(bytes, group, element, b"IS", &even(&text))
}

pub(crate) fn ow(
    bytes: &mut Vec<u8>,
    group: u16,
    element: u16,
    values: &[u16],
) -> Result<(), String> {
    let raw: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    push_elem(bytes, group, element, b"OW", &raw)
}

pub(crate) fn sq(
    bytes: &mut Vec<u8>,
    group: u16,
    element: u16,
    items: &[Vec<u8>],
) -> Result<(), String> {
    let mut body = Vec::new();
    for item in items {
        body.extend_from_slice(&0xFFFEu16.to_le_bytes());
        body.extend_from_slice(&0xE000u16.to_le_bytes());
        body.extend_from_slice(
            &u32::try_from(item.len())
                .map_err(|_| "a sequence item exceeds the 4 GiB length limit".to_owned())?
                .to_le_bytes(),
        );
        body.extend_from_slice(item);
    }
    push_elem(bytes, group, element, b"SQ", &body)
}

/// Group-0002 File Meta Information — always explicit VR LE, required for a
/// conformant Part-10 file. `bytes` must hold exactly the 132-byte preamble.
pub(crate) fn file_meta(
    bytes: &mut Vec<u8>,
    sop_class: &str,
    sop_instance: &str,
) -> Result<(), String> {
    let mut meta = Vec::new();
    ob(&mut meta, 0x0002, 0x0001, &[0, 1])?;
    ui(&mut meta, 0x0002, 0x0002, sop_class)?;
    ui(&mut meta, 0x0002, 0x0003, sop_instance)?;
    ui(&mut meta, 0x0002, 0x0010, EXPLICIT_LE)?;
    ui(&mut meta, 0x0002, 0x0012, IMPL_UID)?;
    sh(&mut meta, 0x0002, 0x0013, "SCAN-KIT")?;
    ul(bytes, 0x0002, 0x0000, meta.len() as u32)?;
    bytes.extend_from_slice(&meta);
    Ok(())
}

fn ul(bytes: &mut Vec<u8>, group: u16, element: u16, value: u32) -> Result<(), String> {
    push_elem(bytes, group, element, b"UL", &value.to_le_bytes())
}

fn ob(bytes: &mut Vec<u8>, group: u16, element: u16, data: &[u8]) -> Result<(), String> {
    push_elem(bytes, group, element, b"OB", data)
}

fn lt(bytes: &mut Vec<u8>, group: u16, element: u16, text: &str) -> Result<(), String> {
    push_elem(bytes, group, element, b"LT", &even(text))
}

fn at(
    bytes: &mut Vec<u8>,
    group: u16,
    element: u16,
    at_group: u16,
    at_element: u16,
) -> Result<(), String> {
    let mut value = at_group.to_le_bytes().to_vec();
    value.extend_from_slice(&at_element.to_le_bytes());
    push_elem(bytes, group, element, b"AT", &value)
}

fn ow32(bytes: &mut Vec<u8>, group: u16, element: u16, values: &[u32]) -> Result<(), String> {
    let raw: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    push_elem(bytes, group, element, b"OW", &raw)
}

const RT_DOSE_CLASS: &str = "1.2.840.10008.5.1.4.1.1.481.2";
const RT_ION_PLAN_CLASS: &str = "1.2.840.10008.5.1.4.1.1.481.8";
const EXPLICIT_LE: &str = "1.2.840.10008.1.2.1";
/// scan-kit's implementation root — a fixed uuid5, like pydicom's root.
const IMPL_UID: &str = "2.25.89062982513270078006766796370322763075";
const DOSE_DISCLAIMER: &str = "scan-kit research dose, not for clinical use";

/// A DICOM UID: the 2.25 arc (uuid-as-integer) over clock + pid + counter.
fn uid() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::SystemTime::UNIX_EPOCH)
        .map(|span| span.as_nanos())
        .unwrap_or(0);
    let raw = nanos.wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ ((std::process::id() as u128) << 96)
        ^ NEXT.fetch_add(1, Ordering::Relaxed) as u128;
    format!("2.25.{raw}")
}

/// YYYYMMDD and HHMMSS from the wall clock, for instance/content timestamps.
fn stamp() -> (String, String) {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::SystemTime::UNIX_EPOCH)
        .map(|span| span.as_secs())
        .unwrap_or(0);
    let days = (secs / 86_400) as i64;
    let tod = secs % 86_400;
    // Civil date from day count (Hinnant).
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { y + 1 } else { y };
    (
        format!("{year:04}{month:02}{day:02}"),
        format!("{:02}{:02}{:02}", tod / 3_600, tod % 3_600 / 60, tod % 60),
    )
}

/// Write `dose` (Gy, `(nz, ny, nx)` row-major on the CT grid) as an RTDOSE in
/// the CT's study and frame of reference — `scan_kit.dicom.dose.write_dose`.
/// `comment` carries the provenance JSON; `plan_uid` adds a plan reference.
/// `dose_type` is PHYSICAL or EFFECTIVE — EFFECTIVE marks an RBE-weighted dose.
pub fn write_rtdose(
    path: &Path,
    dose: &[f32],
    ct: &CtVolume,
    plan_uid: &str,
    summation: &str,
    dose_type: &str,
    description: &str,
    comment: &str,
) -> Result<PathBuf, String> {
    let [nx, ny, nz] = ct.shape;
    // Rows/Columns are US fields — a wider frame would write wrapped dims.
    if nx > u16::MAX as usize || ny > u16::MAX as usize {
        return Err(format!(
            "the dose grid {nx}x{ny} does not fit 16-bit rows/columns"
        ));
    }
    let Some(voxels) = nx.checked_mul(ny).and_then(|plane| plane.checked_mul(nz)) else {
        return Err("the dose grid shape overflows".into());
    };
    if nx == 0 || ny == 0 || nz == 0 {
        return Err("the dose grid has an empty axis".into());
    }
    if !ct.spacing.iter().all(|s| s.is_finite() && *s > 0.0) {
        return Err("the dose grid spacing is not positive".into());
    }
    if dose.len() != voxels {
        return Err(format!(
            "dose is {} voxels, the grid wants {}",
            dose.len(),
            voxels
        ));
    }
    if ct.frame.is_empty() {
        return Err("the CT has no frame of reference".into());
    }
    if ct.study_uid.is_empty() {
        // Study Instance UID is Type 1 - a zero-length UI does not mean
        // "unknown" here.
        return Err("the CT has no study instance uid".into());
    }
    let peak = dose.iter().fold(0.0f64, |peak, &v| {
        peak.max(if v.is_nan() { 0.0 } else { v as f64 })
    });
    if !peak.is_finite() || dose.contains(&f32::NEG_INFINITY) || dose.iter().any(|v| v.is_nan()) {
        // An infinite voxel would quantize every value to 0 and write an
        // "inf" DoseGridScaling the reader itself rejects — and -inf never
        // raises the peak, so it needs its own check. Neither does NaN:
        // an unchecked NaN encoded as 0 Gy would turn a broken transport
        // into a valid-looking all-zero export.
        return Err("the dose contains non-finite values".into());
    }
    let scale = if peak > 0.0 { peak / 4.0e9 } else { 1.0 };
    if !(scale as f32).is_normal() {
        // A sub-f32 scale underflows to 0 on read, turning the stored cube
        // silently all-zero — degenerate input, not data to encode.
        return Err("the dose peak is outside the encodable range".into());
    }
    let pixels: Result<Vec<u32>, String> = dose
        .iter()
        .map(|&v| {
            let value = if v.is_nan() { 0.0 } else { v as f64 };
            // Dose is a sum of non-negative contributions — a negative
            // voxel is corrupt input, not something to floor to zero.
            if value < 0.0 {
                return Err("the dose contains a negative value".to_owned());
            }
            Ok((value / scale).round_ties_even() as u32)
        })
        .collect();
    let pixels = pixels?;
    let sop_uid = uid();
    let (date, time) = stamp();

    let mut bytes = preamble();
    file_meta(&mut bytes, RT_DOSE_CLASS, &sop_uid)?;

    // Patient names and ids arrive as UTF-8 from the source CT — declare the
    // repertoire or a non-ASCII name writes undecodable bytes.
    cs(&mut bytes, 0x0008, 0x0005, "ISO_IR 192")?;
    push_elem(&mut bytes, 0x0008, 0x0012, b"DA", &even(&date))?;
    push_elem(&mut bytes, 0x0008, 0x0013, b"TM", &even(&time))?;
    ui(&mut bytes, 0x0008, 0x0016, RT_DOSE_CLASS)?;
    ui(&mut bytes, 0x0008, 0x0018, &sop_uid)?;
    push_elem(&mut bytes, 0x0008, 0x0023, b"DA", &even(&date))?;
    push_elem(&mut bytes, 0x0008, 0x0033, b"TM", &even(&time))?;
    cs(&mut bytes, 0x0008, 0x0060, "RTDOSE")?;
    lo(&mut bytes, 0x0008, 0x0070, "Pyramid Technical Consultants")?;
    // LO holds 64 characters; a byte slice would panic inside a multi-byte char.
    let series_description: String = format!("{description} (research only)")
        .chars()
        .take(64)
        .collect();
    lo(&mut bytes, 0x0008, 0x103E, &series_description)?;
    // PN/LO cap at 64 characters - a malformed source CT must not write an
    // overlong value into the export.
    let patient_name: String = ct.patient_name.chars().take(64).collect();
    let patient_id: String = ct.patient_id.chars().take(64).collect();
    pn(&mut bytes, 0x0010, 0x0010, &patient_name)?;
    lo(&mut bytes, 0x0010, 0x0020, &patient_id)?;
    ds(&mut bytes, 0x0018, 0x0050, &[ct.spacing[2]])?;
    lo(
        &mut bytes,
        0x0018,
        0x1020,
        &format!("scan-kit {}", env!("CARGO_PKG_VERSION")),
    )?;
    ui(&mut bytes, 0x0020, 0x000D, &ct.study_uid)?;
    ui(&mut bytes, 0x0020, 0x000E, &uid())?;
    ds(&mut bytes, 0x0020, 0x0032, &ct.origin)?;
    ds(&mut bytes, 0x0020, 0x0037, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0])?;
    ui(&mut bytes, 0x0020, 0x0052, &ct.frame)?;
    // LT holds 10240 characters and forbids backslashes; the provenance
    // JSON carries UIDs and paths that can exceed both.
    // LT forbids backslashes — swap rather than strip or JSON escapes and
    // Windows paths lose their separators.
    let comment: String = comment
        .chars()
        .take(10240)
        .map(|c| if c == '\\' { '/' } else { c })
        .collect();
    lt(&mut bytes, 0x0020, 0x4000, &comment)?;
    us(&mut bytes, 0x0028, 0x0002, 1)?;
    cs(&mut bytes, 0x0028, 0x0004, "MONOCHROME2")?;
    is(&mut bytes, 0x0028, 0x0008, nz as i32)?;
    at(&mut bytes, 0x0028, 0x0009, 0x3004, 0x000C)?;
    us(&mut bytes, 0x0028, 0x0010, ny as u16)?;
    us(&mut bytes, 0x0028, 0x0011, nx as u16)?;
    ds(&mut bytes, 0x0028, 0x0030, &[ct.spacing[1], ct.spacing[0]])?;
    us(&mut bytes, 0x0028, 0x0100, 32)?;
    // Stored counts reach ~4e9, which needs all 32 bits; High Bit is the
    // zero-based index of the top stored bit.
    us(&mut bytes, 0x0028, 0x0101, 32)?;
    us(&mut bytes, 0x0028, 0x0102, 31)?;
    us(&mut bytes, 0x0028, 0x0103, 0)?;
    cs(&mut bytes, 0x3004, 0x0002, "GY")?;
    // Both are Type 1 CS with defined terms — a caller's free text would
    // land in the file verbatim.
    if !matches!(dose_type, "PHYSICAL" | "EFFECTIVE" | "ERROR") {
        return Err(format!("dose type {dose_type:?} is not a defined term"));
    }
    if !matches!(
        summation,
        "PLAN"
            | "MULTI_PLAN"
            | "FRACTION"
            | "BEAM"
            | "BRACHY"
            | "FRACTION_SESSION"
            | "BEAM_SESSION"
            | "BRACHY_SESSION"
            | "CONTROL_POINT"
    ) {
        return Err(format!(
            "summation type {summation:?} is not a defined term"
        ));
    }
    cs(&mut bytes, 0x3004, 0x0004, dose_type)?;
    lt(&mut bytes, 0x3004, 0x0006, DOSE_DISCLAIMER)?;
    cs(&mut bytes, 0x3004, 0x000A, summation)?;
    let offsets: Vec<f32> = (0..nz).map(|k| k as f32 * ct.spacing[2]).collect();
    ds(&mut bytes, 0x3004, 0x000C, &offsets)?;
    push_elem(
        &mut bytes,
        0x3004,
        0x000E,
        b"DS",
        &even(&format!("{scale:.10e}")),
    )?;
    if !plan_uid.is_empty() {
        let mut item = Vec::new();
        ui(&mut item, 0x0008, 0x1150, RT_ION_PLAN_CLASS)?;
        ui(&mut item, 0x0008, 0x1155, plan_uid)?;
        // Referenced RT Plan Sequence — (300C,0060) is the brachy
        // application setup sequence, a different attribute entirely.
        sq(&mut bytes, 0x300C, 0x0002, &[item])?;
    }
    ow32(&mut bytes, 0x7FE0, 0x0010, &pixels)?;
    scan_kit_core::atomic_write(path, &bytes)?;
    Ok(path.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slice(z: f32) -> Slice {
        Slice {
            origin: [0.0, 0.0, z],
            thickness: 1.0,
            spacing_x: 1.0,
            spacing_y: 1.0,
            cols: 2,
            rows: 2,
            values: vec![0.0; 4],
        }
    }

    #[test]
    fn a_nan_slice_position_errors_instead_of_panicking() {
        // "nan" in a position tag parses as f32::NAN; the sort used to unwrap.
        for slices in [
            vec![slice(0.0), slice(f32::NAN), slice(1.0)],
            vec![slice(f32::NAN), slice(1.0)],
        ] {
            let mut slices = slices;
            assert!(stack(&mut slices).is_err());
        }
    }

    #[test]
    fn uneven_slice_spacing_is_rejected() {
        let mut gap = vec![slice(0.0), slice(1.0), slice(3.0)];
        assert!(stack(&mut gap).is_err());
        let mut mismatched = vec![slice(0.0), slice(1.0)];
        mismatched[1].spacing_x = 2.0;
        assert!(stack(&mut mismatched).is_err());
        let mut even = vec![slice(0.0), slice(2.5), slice(5.0)];
        let volume = stack(&mut even).unwrap();
        assert!((volume.spacing[2] - 2.5).abs() < 1e-6);
    }

    #[test]
    fn a_signed_ct_pixel_stays_negative() {
        let mut bytes = Vec::new();
        us(&mut bytes, 0x0028, 0x0010, 1).unwrap();
        us(&mut bytes, 0x0028, 0x0011, 1).unwrap();
        ds(&mut bytes, 0x0028, 0x0030, &[1.0, 2.0]).unwrap();
        ds(&mut bytes, 0x0020, 0x0032, &[0.0, 0.0, 10.0]).unwrap();
        us(&mut bytes, 0x0028, 0x0100, 16).unwrap();
        us(&mut bytes, 0x0028, 0x0103, 1).unwrap();
        ds(&mut bytes, 0x0028, 0x1052, &[0.0]).unwrap();
        ds(&mut bytes, 0x0028, 0x1053, &[1.0]).unwrap();
        ds(&mut bytes, 0x0018, 0x0050, &[2.5]).unwrap();
        ds(&mut bytes, 0x3004, 0x000C, &[0.0, 2.5]).unwrap();
        ow(&mut bytes, 0x7FE0, 0x0010, &[(-1000i16) as u16, 0]).unwrap();
        let elems = parse_dataset(&bytes).unwrap();
        let slices = image_slices(&elems, 1.0).unwrap();
        assert_eq!(slices.len(), 2);
        assert!((slices[0].values[0] + 1000.0).abs() < 1e-3);
        assert!((slices[0].spacing_x - 2.0).abs() < 1e-4);
        assert!((slices[1].origin[2] - 12.5).abs() < 1e-3);
    }

    #[test]
    fn a_prone_image_is_stored_in_patient_axes() {
        let mut bytes = Vec::new();
        us(&mut bytes, 0x0028, 0x0010, 2).unwrap();
        us(&mut bytes, 0x0028, 0x0011, 2).unwrap();
        ds(&mut bytes, 0x0028, 0x0030, &[2.0, 2.0]).unwrap();
        ds(&mut bytes, 0x0020, 0x0032, &[2.0, 2.0, 0.0]).unwrap();
        ds(
            &mut bytes,
            0x0020,
            0x0037,
            &[-1.0, 0.0, 0.0, 0.0, -1.0, 0.0],
        )
        .unwrap();
        us(&mut bytes, 0x0028, 0x0100, 16).unwrap();
        ds(&mut bytes, 0x0028, 0x1052, &[0.0]).unwrap();
        ds(&mut bytes, 0x0028, 0x1053, &[1.0]).unwrap();
        ow(&mut bytes, 0x7FE0, 0x0010, &[10, 20, 30, 40]).unwrap();
        let slices = image_slices(&parse_dataset(&bytes).unwrap(), 1.0).unwrap();
        assert_eq!(slices.len(), 1);
        assert!((slices[0].origin[0]).abs() < 1e-4);
        assert!((slices[0].origin[1]).abs() < 1e-4);
        assert_eq!(slices[0].values, vec![40.0, 30.0, 20.0, 10.0]);
    }

    #[test]
    fn feet_first_frame_offsets_step_toward_the_feet() {
        let mut bytes = Vec::new();
        us(&mut bytes, 0x0028, 0x0010, 1).unwrap();
        us(&mut bytes, 0x0028, 0x0011, 1).unwrap();
        ds(&mut bytes, 0x0028, 0x0030, &[1.0, 1.0]).unwrap();
        ds(&mut bytes, 0x0020, 0x0032, &[0.0, 0.0, 10.0]).unwrap();
        ds(&mut bytes, 0x0020, 0x0037, &[-1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).unwrap();
        ds(&mut bytes, 0x3004, 0x000C, &[0.0, 2.5]).unwrap();
        us(&mut bytes, 0x0028, 0x0100, 16).unwrap();
        ow(&mut bytes, 0x7FE0, 0x0010, &[1, 2]).unwrap();
        let slices = image_slices(&parse_dataset(&bytes).unwrap(), 1.0).unwrap();
        let mut z: Vec<f32> = slices.iter().map(|slice| slice.origin[2]).collect();
        z.sort_by(f32::total_cmp);
        assert!((z[0] - 7.5).abs() < 1e-3);
        assert!((z[1] - 10.0).abs() < 1e-3);
    }

    #[test]
    fn an_oblique_orientation_is_rejected() {
        let mut bytes = Vec::new();
        us(&mut bytes, 0x0028, 0x0010, 1).unwrap();
        us(&mut bytes, 0x0028, 0x0011, 1).unwrap();
        ds(&mut bytes, 0x0028, 0x0030, &[1.0, 1.0]).unwrap();
        ds(&mut bytes, 0x0020, 0x0032, &[0.0, 0.0, 0.0]).unwrap();
        ds(&mut bytes, 0x0020, 0x0037, &[1.0, 0.0, 0.0, 0.0, 0.7, 0.7]).unwrap();
        ow(&mut bytes, 0x7FE0, 0x0010, &[0]).unwrap();
        match image_slices(&parse_dataset(&bytes).unwrap(), 1.0) {
            Err(err) => assert!(err.contains("not axial")),
            Ok(_) => panic!("an oblique image was accepted"),
        }
    }

    #[test]
    fn each_contour_slice_is_tested_on_its_own_plane() {
        let study = PatientStudy {
            frame: String::new(),
            ct: CtVolume {
                frame: String::new(),
                origin: [0.0, 0.0, 0.0],
                spacing: [1.0, 1.0, 1.0],
                shape: [4, 4, 2],
                hu: vec![0.0; 32],
                position: "HFS".into(),
                dose_type: String::new(),
                summation: String::new(),
                patient_id: String::new(),
                patient_name: String::new(),
                study_uid: String::new(),
                series_uid: String::new(),
                sop_uids: Vec::new(),
            },
            beams: Vec::new(),
            fractions: 1,
            structures: Vec::new(),
            tps: None,
            prescription: f32::NAN,
            plan_uid: String::new(),
            plan_label: String::new(),
            structures_uid: String::new(),
        };
        let structure = Structure {
            name: "PTV".into(),
            contour: vec![
                vec![
                    [0.0, 0.0, 0.0],
                    [4.0, 0.0, 0.0],
                    [4.0, 4.0, 0.0],
                    [0.0, 4.0, 0.0],
                ],
                vec![
                    [2.0, 2.0, 1.0],
                    [4.0, 2.0, 1.0],
                    [4.0, 4.0, 1.0],
                    [2.0, 4.0, 1.0],
                ],
            ],
        };
        assert!(inside_structure(&study, &structure, 1, 1, 0));
        assert!(!inside_structure(&study, &structure, 1, 1, 1));
        assert!(inside_structure(&study, &structure, 3, 3, 1));
        let ring = Structure {
            name: "RING".into(),
            contour: vec![
                vec![
                    [0.0, 0.0, 0.0],
                    [4.0, 0.0, 0.0],
                    [4.0, 4.0, 0.0],
                    [0.0, 4.0, 0.0],
                ],
                vec![
                    [1.2, 1.2, 0.0],
                    [2.8, 1.2, 0.0],
                    [2.8, 2.8, 0.0],
                    [1.2, 2.8, 0.0],
                ],
            ],
        };
        assert!(inside_structure(&study, &ring, 0, 2, 0));
        assert!(!inside_structure(&study, &ring, 2, 2, 0));
    }

    #[test]
    fn sparse_contours_fill_the_slices_between_them() {
        // A contour pitch above the slice pitch used to leave interior
        // slices un-stamped — the mask striped and volume under-counted.
        // Interior slices take the nearest plane; a slice past the
        // structure's z extent still stays empty (no extrapolation).
        let study = PatientStudy {
            frame: String::new(),
            ct: CtVolume {
                frame: String::new(),
                origin: [0.0, 0.0, 0.0],
                spacing: [1.0, 1.0, 1.0],
                shape: [4, 4, 5],
                hu: vec![0.0; 80],
                position: "HFS".into(),
                dose_type: String::new(),
                summation: String::new(),
                patient_id: String::new(),
                patient_name: String::new(),
                study_uid: String::new(),
                series_uid: String::new(),
                sop_uids: Vec::new(),
            },
            beams: Vec::new(),
            fractions: 1,
            structures: Vec::new(),
            tps: None,
            prescription: f32::NAN,
            plan_uid: String::new(),
            plan_label: String::new(),
            structures_uid: String::new(),
        };
        let square = |z: f32| vec![[0.0, 0.0, z], [4.0, 0.0, z], [4.0, 4.0, z], [0.0, 4.0, z]];
        let structure = Structure {
            name: "PTV".into(),
            contour: vec![square(0.0), square(3.0)],
        };
        assert!(inside_structure(&study, &structure, 1, 1, 0));
        assert!(inside_structure(&study, &structure, 1, 1, 1));
        assert!(inside_structure(&study, &structure, 1, 1, 2));
        assert!(inside_structure(&study, &structure, 1, 1, 3));
        assert!(!inside_structure(&study, &structure, 1, 1, 4));
    }

    #[test]
    fn a_zero_pixel_spacing_is_rejected() {
        let mut bytes = Vec::new();
        us(&mut bytes, 0x0028, 0x0010, 1).unwrap();
        us(&mut bytes, 0x0028, 0x0011, 1).unwrap();
        ds(&mut bytes, 0x0028, 0x0030, &[0.0, 1.0]).unwrap();
        ds(&mut bytes, 0x0020, 0x0032, &[0.0, 0.0, 0.0]).unwrap();
        ow(&mut bytes, 0x7FE0, 0x0010, &[0]).unwrap();
        let elems = parse_dataset(&bytes).unwrap();
        assert!(image_slices(&elems, 1.0).is_err());
    }

    #[test]
    fn hfs_gantry_zero_is_the_support_matrix() {
        let got = gantry_to_patient(0.0, 0.0, "HFS").unwrap();
        assert_eq!(got, [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn water_hu_is_unit_density() {
        assert!((hu_density(0.0) - 1.0).abs() < 1e-3);
        assert!((scanner_density("default", 0.0) - hu_density(0.0)).abs() < 1e-6);
        assert!((scanner_density("water", 0.0) - 1.0).abs() < 1e-3);
        assert_eq!(hu_label(0.0), 44);
        assert!(hu_label(-2000.0) != hu_label(0.0));
    }

    fn iv(bytes: &mut Vec<u8>, group: u16, element: u16, value: &[u8]) {
        bytes.extend_from_slice(&group.to_le_bytes());
        bytes.extend_from_slice(&element.to_le_bytes());
        bytes.extend_from_slice(&(value.len() as u32).to_le_bytes());
        bytes.extend_from_slice(value);
    }

    fn iv_sq(bytes: &mut Vec<u8>, group: u16, element: u16, items: &[Vec<u8>]) {
        let mut body = Vec::new();
        for item in items {
            body.extend_from_slice(&0xFFFEu16.to_le_bytes());
            body.extend_from_slice(&0xE000u16.to_le_bytes());
            body.extend_from_slice(&(item.len() as u32).to_le_bytes());
            body.extend_from_slice(item);
        }
        iv(bytes, group, element, &body);
    }

    #[test]
    fn a_nested_study_loads_from_its_root() {
        let root = std::env::temp_dir().join(format!("scan-kit-nested-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        write_water_study(&root.join("study").join("series")).unwrap();
        let study = load_study(&root).unwrap();
        assert_eq!(study.ct.shape, [4, 4, 2]);
        let entries = crate::index_study(&root).unwrap();
        assert!(entries.iter().any(|entry| entry.modality == "CT"));
        assert!(entries.iter().any(|entry| entry.modality == "RTPLAN"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn pixel_data_starting_with_the_item_marker_stays_an_element() {
        // Little-endian FFFE,E000 is a legal first word pair in a pixel
        // payload - it must not be read as an implicit sequence.
        let mut payload = vec![0xFE, 0xFF, 0x00, 0xE0, 0x04, 0x00, 0x00, 0x00];
        payload.extend_from_slice(&[1u8, 2, 3, 4]);
        let mut bytes = preamble();
        ui(&mut bytes, 0x0002, 0x0010, "1.2.840.10008.1.2").unwrap();
        iv(&mut bytes, 0x7FE0, 0x0010, &payload);
        let elems = parse_file(&bytes).unwrap();
        let found = elem(&elems, tag(0x7FE0, 0x0010)).unwrap();
        assert_eq!(found.bytes, payload);
    }

    #[test]
    fn an_implicit_vr_plan_parses_and_setup_beams_are_skipped() {
        let mut ref_beam = Vec::new();
        iv(&mut ref_beam, 0x300C, 0x0006, &even("1"));
        iv(&mut ref_beam, 0x300A, 0x0086, &even("10"));
        let mut group = Vec::new();
        iv(&mut group, 0x300A, 0x0078, &even("1"));
        iv_sq(&mut group, 0x300C, 0x0004, &[ref_beam]);
        let mut cp = Vec::new();
        iv(&mut cp, 0x300A, 0x0114, &even("100"));
        iv(&mut cp, 0x300A, 0x011E, &even("0"));
        iv(&mut cp, 0x300A, 0x0122, &even("0"));
        iv(&mut cp, 0x300A, 0x012C, &even("0\\0\\0"));
        iv(&mut cp, 0x300A, 0x0392, &even("1"));
        // Spot position and weight arrays ride as raw f32 (OF) in real plans.
        let mut positions = Vec::new();
        positions.extend_from_slice(&1.5f32.to_le_bytes());
        positions.extend_from_slice(&(-2.0f32).to_le_bytes());
        iv(&mut cp, 0x300A, 0x0394, &positions);
        let mut weights = Vec::new();
        weights.extend_from_slice(&0.25f32.to_le_bytes());
        iv(&mut cp, 0x300A, 0x0396, &weights);
        let mut treatment = Vec::new();
        iv(&mut treatment, 0x300A, 0x00C0, &even("1"));
        iv(&mut treatment, 0x300A, 0x00CE, &even("TREATMENT"));
        iv(&mut treatment, 0x300A, 0x010E, &even("1"));
        iv(&mut treatment, 0x300C, 0x006A, &even("1"));
        iv_sq(&mut treatment, 0x300A, 0x03A8, &[cp]);
        let mut setup = Vec::new();
        iv(&mut setup, 0x300A, 0x00C0, &even("2"));
        iv(&mut setup, 0x300A, 0x00CE, &even("SETUP"));
        let mut bytes = preamble();
        ui(&mut bytes, 0x0002, 0x0010, "1.2.840.10008.1.2").unwrap();
        iv_sq(&mut bytes, 0x300A, 0x0070, &[group]);
        iv_sq(&mut bytes, 0x300A, 0x03A2, &[treatment, setup]);
        let plan = read_plan(&parse_file(&bytes).unwrap(), "f".into(), "HFS").unwrap();
        assert_eq!(plan.beams.len(), 1);
        assert!((plan.beams[0].meterset - 10.0).abs() < 1e-4);
        let spots = &plan.beams[0].spots;
        assert_eq!(spots.len(), 1);
        assert!((spots[0].x - 1.5).abs() < 1e-4);
        assert!((spots[0].y + 2.0).abs() < 1e-4);
    }

    #[test]
    fn one_frame_loads_and_a_second_frame_fails() {
        let root = std::env::temp_dir().join(format!("scan-kit-patient-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        write_water_study(&root).unwrap();
        let study = load_study(&root).unwrap();
        assert_eq!(study.ct.shape, [4, 4, 2]);
        assert_eq!(study.beams.len(), 1);
        assert_eq!(study.beams[0].spots.len(), 1);
        assert_eq!(study.structures.len(), 1);
        assert!(study.tps.is_some());
        assert_eq!(study.ct.frame, "1.2.3.4.5");
        assert_eq!(study.tps.as_ref().unwrap().frame, "1.2.3.4.5");
        // The fixture contour sits at z=0.5 — midway between slices — and
        // must stamp exactly its nearest slice, not both.
        assert!(!inside_structure(&study, &study.structures[0], 1, 1, 0));
        assert!(inside_structure(&study, &study.structures[0], 1, 1, 1));
        let mut extra = preamble();
        ui(&mut extra, 0x0008, 0x0016, "1.2.840.10008.5.1.4.1.1.2").unwrap();
        ui(&mut extra, 0x0020, 0x0052, "9.9.9").unwrap();
        us(&mut extra, 0x0028, 0x0010, 1).unwrap();
        us(&mut extra, 0x0028, 0x0011, 1).unwrap();
        ds(&mut extra, 0x0028, 0x0030, &[1.0, 1.0]).unwrap();
        ds(&mut extra, 0x0020, 0x0032, &[0.0, 0.0, 5.0]).unwrap();
        ow(&mut extra, 0x7FE0, 0x0010, &[0]).unwrap();
        std::fs::write(root.join("other.dcm"), extra).unwrap();
        assert!(load_study(&root).is_err());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn an_ion_plan_keeps_raw_weights_and_the_scanning_spot_size() {
        let mut cp = Vec::new();
        ds(&mut cp, 0x300A, 0x0114, &[100.0]).unwrap();
        is(&mut cp, 0x300A, 0x0392, 2).unwrap();
        ds(&mut cp, 0x300A, 0x0394, &[1.0, 2.0, 3.0, 4.0]).unwrap();
        ds(&mut cp, 0x300A, 0x0396, &[0.5, 0.25]).unwrap();
        ds(&mut cp, 0x300A, 0x0398, &[4.0, 6.0]).unwrap();
        let mut beam = Vec::new();
        sq(&mut beam, 0x300A, 0x03A8, &[cp]).unwrap();
        let mut bytes = preamble();
        ui(&mut bytes, 0x0008, 0x0016, "1.2.840.10008.5.1.4.1.1.481.8").unwrap();
        lo(&mut bytes, 0x300A, 0x0002, "TESTPLAN").unwrap();
        sq(&mut bytes, 0x300A, 0x03A2, &[beam]).unwrap();
        let path = std::env::temp_dir().join(format!("scan-kit-ion-{}.dcm", std::process::id()));
        std::fs::write(&path, bytes).unwrap();
        let plan = read_ion_plan(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(plan.label, "TESTPLAN");
        assert_eq!(plan.spots.len(), 2);
        assert!((plan.spots[0].charge - 0.5).abs() < 1e-4);
        assert!((plan.spots[0].x - 1.0).abs() < 1e-4);
        assert!((plan.spots[0].y - 2.0).abs() < 1e-4);
        assert!((plan.spots[1].x - 3.0).abs() < 1e-4);
        assert!((plan.spots[1].y - 4.0).abs() < 1e-4);
        assert!((plan.spots[0].size_x - 4.0).abs() < 1e-3);
        assert!((plan.spots[0].size_y - 6.0).abs() < 1e-3);
        let imported: Vec<_> = plan
            .spots
            .iter()
            .map(|spot| scan_kit_core::ImportSpot {
                x: f64::from(spot.x),
                y: f64::from(spot.y),
                energy: f64::from(spot.energy),
                charge: f64::from(spot.charge),
                beam_size: scan_kit_core::dicom_beam_size(
                    f64::from(spot.size_x),
                    f64::from(spot.size_y),
                    3.61,
                    true,
                ),
                plan_index: spot.plan_index,
            })
            .collect();
        let built = scan_kit_core::build_plan(
            "dicom_rt_plan",
            &serde_json::json!({
                "dicom_path": "plan.dcm",
                "use_dicom_beam_size": true,
                "spot_order": "plan_order"
            }),
            scan_kit_core::PlanSource::Ion(&imported),
            Some(&plan.label),
        )
        .unwrap();
        assert_eq!(built.preview[0][3], "5");
        assert_eq!(built.preview[0][6], "0.5000");
        assert_eq!(built.preview[1][6], "0.2500");
    }

    #[test]
    fn an_in_range_shifter_keeps_its_wet_and_a_second_one_fails() {
        let plan = read_plan(
            &plan_with_shifter(&[(1, "IN", Some(40.0))]),
            "f".into(),
            "HFS",
        )
        .unwrap();
        let spot = &plan.beams[0].spots[0];
        assert_eq!(spot.shifter, "RS1");
        assert!((spot.wet_mm - 40.0).abs() < 1e-3);
        assert!((spot.shifter_distance_mm - 300.0).abs() < 1e-3);
        let bare = read_plan(&plan_with_shifter(&[(1, "IN", None)]), "f".into(), "HFS").unwrap();
        assert!(bare.beams[0].spots[0].wet_mm.is_nan());
        assert!(read_plan(
            &plan_with_shifter(&[(1, "IN", Some(40.0)), (2, "IN", Some(20.0))]),
            "f".into(),
            "HFS"
        )
        .is_err());
        assert!(read_plan(
            &plan_with_shifter(&[(3, "IN", Some(40.0))]),
            "f".into(),
            "HFS"
        )
        .is_err());
    }

    #[test]
    fn a_negative_dose_is_an_error_not_a_silent_floor() {
        // Dose is a sum of non-negative contributions — flooring a negative
        // would hide upstream corruption.
        let dir = std::env::temp_dir().join(format!("scan-kit-negdose-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let ct = CtVolume {
            frame: "1.2.3".into(),
            origin: [0.0, 0.0, 0.0],
            spacing: [1.0, 1.0, 1.0],
            shape: [2, 2, 2],
            hu: Vec::new(),
            position: "HFS".into(),
            dose_type: String::new(),
            summation: String::new(),
            patient_id: "P1".into(),
            patient_name: "A^B".into(),
            study_uid: "1.2.3".into(),
            series_uid: "1.2.3.1".into(),
            sop_uids: vec!["1.2.3.1.1".into()],
        };
        let mut dose = vec![1.0f32; 8];
        dose[3] = -0.5;
        let err = write_rtdose(
            &dir.join("dose.dcm"),
            &dose,
            &ct,
            "9.8",
            "FRACTION",
            "PHYSICAL",
            "t",
            "{}",
        )
        .unwrap_err();
        assert!(err.contains("negative"), "{err}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_water_study_stamps_file_meta_and_a_control_point_index() {
        let root = std::env::temp_dir().join(format!("scan-kit-wmeta-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        write_water_study(&root).unwrap();
        for name in ["ct0.dcm", "ct1.dcm", "plan.dcm", "struct.dcm", "dose.dcm"] {
            let bytes = fs::read(root.join(name)).unwrap();
            assert!(
                crate::meta_sop_class(&bytes).is_some(),
                "{name} missing meta"
            );
        }
        let bytes = fs::read(root.join("plan.dcm")).unwrap();
        assert_eq!(
            crate::meta_sop_class(&bytes).as_deref(),
            Some(RT_ION_PLAN_CLASS)
        );
        let elems = parse_file(&bytes).unwrap();
        let beams = seq(&elems, tag(0x300A, 0x03A2));
        let control = seq(&beams[0], tag(0x300A, 0x03A8));
        assert!(!control.is_empty());
        for item in control {
            // Control Point Index is Type 1 on every item.
            assert!(elem(item, tag(0x300A, 0x0112)).is_some());
            let tags: Vec<u32> = item.iter().map(|entry| entry.tag).collect();
            let mut sorted = tags.clone();
            sorted.sort();
            assert_eq!(tags, sorted, "control point tags out of order");
        }
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn rtdose_round_trips_through_the_reader() {
        let dir = std::env::temp_dir().join(format!("scan-kit-rtdose-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let ct = CtVolume {
            frame: "1.2.3.4.5".into(),
            origin: [10.0, 20.0, -5.0],
            spacing: [2.0, 3.0, 4.0],
            shape: [4, 3, 2],
            hu: Vec::new(),
            position: "HFS".into(),
            dose_type: String::new(),
            summation: String::new(),
            patient_id: "P42".into(),
            patient_name: "Doe^Jane".into(),
            study_uid: "1.2.3".into(),
            series_uid: "1.2.3.1".into(),
            sop_uids: vec!["1.2.3.1.1".into(), "1.2.3.1.2".into()],
        };
        let dose: Vec<f32> = (0..24).map(|i| i as f32 * 0.5).collect();
        let path = write_rtdose(
            &dir.join("dose.dcm"),
            &dose,
            &ct,
            "9.8.7.6",
            "FRACTION",
            "PHYSICAL",
            "test dose",
            r#"{"beam":"a"}"#,
        )
        .unwrap();
        let bytes = fs::read(&path).unwrap();
        assert_eq!(&bytes[128..132], b"DICM");
        let elems = parse_dataset(&bytes[132..]).unwrap();
        assert_eq!(text_tag(&elems, tag(0x0008, 0x0016)), RT_DOSE_CLASS);
        assert_eq!(text_tag(&elems, tag(0x0008, 0x0060)), "RTDOSE");
        assert_eq!(text_tag(&elems, tag(0x0010, 0x0020)), "P42");
        assert_eq!(text_tag(&elems, tag(0x0010, 0x0010)), "Doe^Jane");
        assert_eq!(text_tag(&elems, tag(0x0020, 0x000D)), "1.2.3");
        assert_eq!(text_tag(&elems, tag(0x0020, 0x0052)), "1.2.3.4.5");
        assert_eq!(text_tag(&elems, tag(0x3004, 0x0002)), "GY");
        assert_eq!(text_tag(&elems, tag(0x3004, 0x0004)), "PHYSICAL");
        assert_eq!(text_tag(&elems, tag(0x3004, 0x000A)), "FRACTION");
        assert_eq!(text_tag(&elems, tag(0x3004, 0x0006)), DOSE_DISCLAIMER);
        assert_eq!(text_tag(&elems, tag(0x0020, 0x4000)), r#"{"beam":"a"}"#);
        assert_eq!(text_tag(&elems, tag(0x0028, 0x0008)), "2");
        let plan_item = &seq(&elems, tag(0x300C, 0x0002))[0];
        assert_eq!(text_tag(plan_item, tag(0x0008, 0x1150)), RT_ION_PLAN_CLASS);
        assert_eq!(text_tag(plan_item, tag(0x0008, 0x1155)), "9.8.7.6");
        // The dose reader reconstructs Gy from the scaled u32 pixels.
        let scale = float_tag(&elems, tag(0x3004, 0x000E)).unwrap();
        assert!(scale > 0.0);
        let slices = image_slices(&elems, scale).unwrap();
        assert_eq!(slices.len(), 2);
        assert!((slices[1].origin[2] - (-1.0)).abs() < 1e-3);
        for (z, slice) in slices.iter().enumerate() {
            for (i, &got) in slice.values.iter().enumerate() {
                let index = i + 12 * z;
                // One stored count plus the DS/f32 scaling precision.
                assert!(
                    (got - dose[index]).abs() <= dose[index].abs() * 1e-5 + scale,
                    "voxel {index}: {got} vs {}",
                    dose[index]
                );
            }
        }
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn rtdose_keeps_a_non_ascii_description_and_declares_32_stored_bits() {
        let dir = std::env::temp_dir().join(format!("scan-kit-rtdose-desc-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let ct = CtVolume {
            frame: "f".into(),
            origin: [0.0; 3],
            spacing: [1.0; 3],
            shape: [1, 1, 1],
            hu: Vec::new(),
            position: "HFS".into(),
            dose_type: String::new(),
            summation: String::new(),
            patient_id: String::new(),
            patient_name: String::new(),
            study_uid: "1.2.3".into(),
            series_uid: String::new(),
            sop_uids: Vec::new(),
        };
        // "x" + 60 é's is 77 characters and puts a two-byte char across the
        // old 64-byte cut.
        let description = format!("x{}", "é".repeat(60));
        let path = write_rtdose(
            &dir.join("dose.dcm"),
            &[0.5],
            &ct,
            "",
            "PLAN",
            "PHYSICAL",
            &description,
            "",
        )
        .unwrap();
        let elems = parse_dataset(&fs::read(&path).unwrap()[132..]).unwrap();
        let written = text_tag(&elems, tag(0x0008, 0x103E));
        assert_eq!(written.chars().count(), 64);
        assert!(written.starts_with('x'));
        assert_eq!(int_tag(&elems, tag(0x0028, 0x0101)), Some(32));
        assert_eq!(int_tag(&elems, tag(0x0028, 0x0102)), Some(31));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn rtdose_rejects_a_wrong_sized_dose() {
        let ct = CtVolume {
            frame: "f".into(),
            origin: [0.0; 3],
            spacing: [1.0; 3],
            shape: [2, 2, 2],
            hu: Vec::new(),
            position: String::new(),
            dose_type: String::new(),
            summation: String::new(),
            patient_id: String::new(),
            patient_name: String::new(),
            study_uid: "1.2.3".into(),
            series_uid: String::new(),
            sop_uids: Vec::new(),
        };
        assert!(write_rtdose(
            Path::new("x"),
            &[0.0; 4],
            &ct,
            "",
            "PLAN",
            "PHYSICAL",
            "d",
            ""
        )
        .is_err());
        let out = std::env::temp_dir().join(format!("scan-kit-rtdose-bad-{}", std::process::id()));
        assert!(write_rtdose(&out, &[0.0; 8], &ct, "", "PLAN", "PHYSICAL", "d", "").is_ok());
        // NaN is non-finite too — an all-NaN transport must not export as a
        // valid-looking all-zero dose.
        assert!(write_rtdose(&out, &[f32::NAN; 8], &ct, "", "PLAN", "PHYSICAL", "d", "").is_err());
        let _ = fs::remove_file(&out);
    }

    fn plan_with_shifter(settings: &[(i32, &str, Option<f32>)]) -> Vec<Elem> {
        let mut ref_beam = Vec::new();
        is(&mut ref_beam, 0x300C, 0x0006, 1).unwrap();
        ds(&mut ref_beam, 0x300A, 0x0086, &[10.0]).unwrap();
        let mut group = Vec::new();
        is(&mut group, 0x300A, 0x0078, 1).unwrap();
        sq(&mut group, 0x300C, 0x0004, &[ref_beam]).unwrap();
        let mut shifters = Vec::new();
        for number in [1, 2] {
            let mut item = Vec::new();
            is(&mut item, 0x300A, 0x0316, number).unwrap();
            lo(&mut item, 0x300A, 0x0318, &format!("RS{number}")).unwrap();
            shifters.push(item);
        }
        let mut setting_items = Vec::new();
        for (number, mode, wet) in settings {
            let mut item = Vec::new();
            is(&mut item, 0x300C, 0x0100, *number).unwrap();
            cs(&mut item, 0x300A, 0x0362, mode).unwrap();
            ds(&mut item, 0x300A, 0x0364, &[300.0]).unwrap();
            if let Some(wet) = wet {
                ds(&mut item, 0x300A, 0x0366, &[*wet]).unwrap();
            }
            setting_items.push(item);
        }
        let mut cp = Vec::new();
        ds(&mut cp, 0x300A, 0x0114, &[100.0]).unwrap();
        ds(&mut cp, 0x300A, 0x011E, &[0.0]).unwrap();
        ds(&mut cp, 0x300A, 0x0122, &[0.0]).unwrap();
        ds(&mut cp, 0x300A, 0x012C, &[0.0, 0.0, 0.0]).unwrap();
        sq(&mut cp, 0x300A, 0x0360, &setting_items).unwrap();
        is(&mut cp, 0x300A, 0x0392, 1).unwrap();
        ds(&mut cp, 0x300A, 0x0394, &[1.0, -2.0]).unwrap();
        ds(&mut cp, 0x300A, 0x0396, &[1.0]).unwrap();
        let mut beam = Vec::new();
        is(&mut beam, 0x300A, 0x00C0, 1).unwrap();
        ds(&mut beam, 0x300A, 0x010E, &[1.0]).unwrap();
        sq(&mut beam, 0x300A, 0x0314, &shifters).unwrap();
        sq(&mut beam, 0x300A, 0x03A8, &[cp]).unwrap();
        let mut plan = Vec::new();
        sq(&mut plan, 0x300A, 0x0070, &[group]).unwrap();
        sq(&mut plan, 0x300A, 0x03A2, &[beam]).unwrap();
        parse_dataset(&plan).unwrap()
    }

    #[test]
    fn a_short_position_list_does_not_invent_spots_at_the_origin() {
        let plan = read_plan(&plan_with_spot_count(3, 2), "f".into(), "HFS").unwrap();
        let spots = &plan.beams[0].spots;
        assert_eq!(spots.len(), 2);
        assert!((spots[0].x - 1.0).abs() < 1e-4);
        assert!((spots[1].x - 2.0).abs() < 1e-4);
        // A negative count parses as zero spots instead of hanging the load.
        let plan = read_plan(&plan_with_spot_count(-4, 2), "f".into(), "HFS").unwrap();
        assert!(plan.beams[0].spots.is_empty());
    }

    #[test]
    fn a_nan_spot_position_is_dropped() {
        // DS text "nan"/"inf" parses to non-finite f32 — such a spot must not
        // reach the transport or the exported plan CSV.
        let mut ref_beam = Vec::new();
        is(&mut ref_beam, 0x300C, 0x0006, 1).unwrap();
        ds(&mut ref_beam, 0x300A, 0x0086, &[10.0]).unwrap();
        let mut group = Vec::new();
        is(&mut group, 0x300A, 0x0078, 1).unwrap();
        sq(&mut group, 0x300C, 0x0004, &[ref_beam]).unwrap();
        let mut cp = Vec::new();
        ds(&mut cp, 0x300A, 0x0114, &[100.0]).unwrap();
        ds(&mut cp, 0x300A, 0x011E, &[0.0]).unwrap();
        ds(&mut cp, 0x300A, 0x0122, &[0.0]).unwrap();
        ds(&mut cp, 0x300A, 0x012C, &[0.0, 0.0, 0.0]).unwrap();
        is(&mut cp, 0x300A, 0x0392, 2).unwrap();
        // Binary FL carries a literal NaN bit pattern — the corrupt-file case.
        fl(&mut cp, 0x300A, 0x0394, &[f32::NAN, -1.0, 2.0, -2.0]).unwrap();
        ds(&mut cp, 0x300A, 0x0396, &[1.0, 1.0]).unwrap();
        let mut beam = Vec::new();
        is(&mut beam, 0x300A, 0x00C0, 1).unwrap();
        ds(&mut beam, 0x300A, 0x010E, &[1.0]).unwrap();
        is(&mut beam, 0x300C, 0x006A, 1).unwrap();
        sq(&mut beam, 0x300A, 0x03A8, &[cp]).unwrap();
        let mut plan = Vec::new();
        sq(&mut plan, 0x300A, 0x0070, &[group]).unwrap();
        sq(&mut plan, 0x300A, 0x03A2, &[beam]).unwrap();
        let plan = read_plan(&parse_dataset(&plan).unwrap(), "f".into(), "HFS").unwrap();
        let spots = &plan.beams[0].spots;
        assert_eq!(spots.len(), 1);
        assert!((spots[0].x - 2.0).abs() < 1e-4);
    }

    fn plan_with_spot_count(declared: i32, pairs: usize) -> Vec<Elem> {
        plan_with_weights(declared, pairs, declared.max(0) as usize)
    }

    fn plan_with_weights(declared: i32, pairs: usize, nweights: usize) -> Vec<Elem> {
        let mut ref_beam = Vec::new();
        is(&mut ref_beam, 0x300C, 0x0006, 1).unwrap();
        ds(&mut ref_beam, 0x300A, 0x0086, &[10.0]).unwrap();
        let mut group = Vec::new();
        is(&mut group, 0x300A, 0x0078, 1).unwrap();
        sq(&mut group, 0x300C, 0x0004, &[ref_beam]).unwrap();
        let mut cp = Vec::new();
        ds(&mut cp, 0x300A, 0x0114, &[100.0]).unwrap();
        ds(&mut cp, 0x300A, 0x011E, &[0.0]).unwrap();
        ds(&mut cp, 0x300A, 0x0122, &[0.0]).unwrap();
        ds(&mut cp, 0x300A, 0x012C, &[0.0, 0.0, 0.0]).unwrap();
        is(&mut cp, 0x300A, 0x0392, declared).unwrap();
        let xy: Vec<f32> = (0..pairs)
            .flat_map(|i| [(i + 1) as f32, -((i + 1) as f32)])
            .collect();
        ds(&mut cp, 0x300A, 0x0394, &xy).unwrap();
        ds(&mut cp, 0x300A, 0x0396, &vec![1.0; nweights]).unwrap();
        let mut beam = Vec::new();
        is(&mut beam, 0x300A, 0x00C0, 1).unwrap();
        ds(&mut beam, 0x300A, 0x010E, &[1.0]).unwrap();
        is(&mut beam, 0x300C, 0x006A, 1).unwrap();
        sq(&mut beam, 0x300A, 0x03A8, &[cp]).unwrap();
        let mut plan = Vec::new();
        sq(&mut plan, 0x300A, 0x0070, &[group]).unwrap();
        sq(&mut plan, 0x300A, 0x03A2, &[beam]).unwrap();
        parse_dataset(&plan).unwrap()
    }

    #[test]
    fn a_target_prescription_dose_is_parsed() {
        // Clinical goals are %-of-prescription — a plan that carries the
        // Dose Reference TARGET dose must surface it.
        let mut item = Vec::new();
        cs(&mut item, 0x300A, 0x0020, "TARGET").unwrap();
        ds(&mut item, 0x300A, 0x0026, &[70.0]).unwrap();
        let mut other = Vec::new();
        cs(&mut other, 0x300A, 0x0020, "ORGAN_AT_RISK").unwrap();
        ds(&mut other, 0x300A, 0x0026, &[5.0]).unwrap();
        // Rebuild the plan bytes with the dose reference sequence appended —
        // sq() operates on bytes, so regenerate through a tiny plan.
        let mut ref_beam = Vec::new();
        is(&mut ref_beam, 0x300C, 0x0006, 1).unwrap();
        ds(&mut ref_beam, 0x300A, 0x0086, &[10.0]).unwrap();
        let mut group = Vec::new();
        is(&mut group, 0x300A, 0x0078, 1).unwrap();
        sq(&mut group, 0x300C, 0x0004, &[ref_beam]).unwrap();
        let mut cp = Vec::new();
        ds(&mut cp, 0x300A, 0x0114, &[100.0]).unwrap();
        ds(&mut cp, 0x300A, 0x011E, &[0.0]).unwrap();
        ds(&mut cp, 0x300A, 0x0122, &[0.0]).unwrap();
        ds(&mut cp, 0x300A, 0x012C, &[0.0, 0.0, 0.0]).unwrap();
        is(&mut cp, 0x300A, 0x0392, 1).unwrap();
        ds(&mut cp, 0x300A, 0x0394, &[1.0, -1.0]).unwrap();
        ds(&mut cp, 0x300A, 0x0396, &[1.0]).unwrap();
        let mut beam = Vec::new();
        is(&mut beam, 0x300A, 0x00C0, 1).unwrap();
        ds(&mut beam, 0x300A, 0x010E, &[1.0]).unwrap();
        is(&mut beam, 0x300C, 0x006A, 1).unwrap();
        sq(&mut beam, 0x300A, 0x03A8, &[cp]).unwrap();
        let mut plan = Vec::new();
        sq(&mut plan, 0x300A, 0x0070, std::slice::from_ref(&group)).unwrap();
        sq(&mut plan, 0x300A, 0x03A2, &[beam.clone()]).unwrap();
        sq(&mut plan, 0x300A, 0x0010, &[item, other.clone()]).unwrap();
        let elems = parse_dataset(&plan).unwrap();
        let parsed = read_plan(&elems, "f".into(), "HFS").unwrap();
        assert!((parsed.prescription - 70.0).abs() < 1e-4);
        // No TARGET item means no prescription — not "pick the first".
        let mut plan = Vec::new();
        sq(&mut plan, 0x300A, 0x0070, &[group]).unwrap();
        sq(&mut plan, 0x300A, 0x03A2, &[beam.clone()]).unwrap();
        sq(&mut plan, 0x300A, 0x0010, &[other]).unwrap();
        let parsed = read_plan(&parse_dataset(&plan).unwrap(), "f".into(), "HFS").unwrap();
        assert!(parsed.prescription.is_nan());
    }

    #[test]
    fn a_beam_changing_geometry_mid_beam_is_rejected() {
        // Every control point shares the beam record — a mid-beam gantry or
        // isocenter change cannot be transported per-spot and must fail
        // loudly instead of applying the last point's angle to all spots.
        let cp_with = |gantry: f32| {
            let mut cp = Vec::new();
            ds(&mut cp, 0x300A, 0x0114, &[100.0]).unwrap();
            ds(&mut cp, 0x300A, 0x011E, &[gantry]).unwrap();
            ds(&mut cp, 0x300A, 0x0122, &[0.0]).unwrap();
            ds(&mut cp, 0x300A, 0x012C, &[0.0, 0.0, 0.0]).unwrap();
            is(&mut cp, 0x300A, 0x0392, 1).unwrap();
            ds(&mut cp, 0x300A, 0x0394, &[1.0, -1.0]).unwrap();
            ds(&mut cp, 0x300A, 0x0396, &[1.0]).unwrap();
            cp
        };
        let mut ref_beam = Vec::new();
        is(&mut ref_beam, 0x300C, 0x0006, 1).unwrap();
        ds(&mut ref_beam, 0x300A, 0x0086, &[10.0]).unwrap();
        let mut group = Vec::new();
        is(&mut group, 0x300A, 0x0078, 1).unwrap();
        sq(&mut group, 0x300C, 0x0004, &[ref_beam]).unwrap();
        let mut beam = Vec::new();
        is(&mut beam, 0x300A, 0x00C0, 1).unwrap();
        ds(&mut beam, 0x300A, 0x010E, &[1.0]).unwrap();
        is(&mut beam, 0x300C, 0x006A, 1).unwrap();
        sq(&mut beam, 0x300A, 0x03A8, &[cp_with(0.0), cp_with(45.0)]).unwrap();
        let mut plan = Vec::new();
        sq(&mut plan, 0x300A, 0x0070, std::slice::from_ref(&group)).unwrap();
        sq(&mut plan, 0x300A, 0x03A2, &[beam]).unwrap();
        assert!(read_plan(&parse_dataset(&plan).unwrap(), "f".into(), "HFS").is_err());
        // Identical geometry across control points stays legal.
        let mut beam = Vec::new();
        is(&mut beam, 0x300A, 0x00C0, 1).unwrap();
        ds(&mut beam, 0x300A, 0x010E, &[1.0]).unwrap();
        is(&mut beam, 0x300C, 0x006A, 1).unwrap();
        sq(&mut beam, 0x300A, 0x03A8, &[cp_with(0.0), cp_with(0.0)]).unwrap();
        let mut plan = Vec::new();
        sq(&mut plan, 0x300A, 0x0070, &[group]).unwrap();
        sq(&mut plan, 0x300A, 0x03A2, &[beam]).unwrap();
        assert!(read_plan(&parse_dataset(&plan).unwrap(), "f".into(), "HFS").is_ok());
    }

    #[test]
    fn a_plan_without_setups_uses_the_ct_position() {
        // With no Patient Setup Sequence the CT's (0018,5100) is the only
        // orientation source — an FFS patient must not parse as HFS.
        let elems = plan_with_weights(1, 1, 1);
        let parsed = read_plan(&elems, "f".into(), "FFS").unwrap();
        assert_eq!(parsed.beams[0].position, "FFS");
        // And nothing at all is a loud error, not a silent HFS mirror.
        assert!(read_plan(&elems, "f".into(), "").is_err());
    }

    #[test]
    fn an_oversized_short_vr_errors_instead_of_panicking() {
        let mut bytes = Vec::new();
        let huge: Vec<f32> = vec![0.0; 9000]; // ~90 KB of DS text
        assert!(ds(&mut bytes, 0x3004, 0x000C, &huge).is_err());
        assert!(ds(&mut bytes, 0x3004, 0x000C, &[1.0]).is_ok());
    }
    #[test]
    fn a_text_shaped_us_dimension_is_rejected_instead_of_wrapping() {
        // US bytes "-1" text-parse to -1, which used to wrap to usize::MAX
        // and panic in stack() — or pass 0 into a divide-by-zero.
        let elems = vec![
            Elem {
                tag: tag(0x0028, 0x0010),
                vr: [0, 0],
                items: Vec::new(),
                bytes: b"-1".to_vec(),
            },
            Elem {
                tag: tag(0x0028, 0x0011),
                vr: [0, 0],
                items: Vec::new(),
                bytes: b"-1".to_vec(),
            },
        ];
        assert!(image_slices(&elems, 1.0).is_err());
    }

    #[test]
    fn a_binary_us_dimension_beats_its_ascii_digit_shape() {
        // US value 50 encodes as [0x32, 0x00] — the ASCII digit "2" plus a
        // NUL that no legal IS string can hold.
        let elem = Elem {
            tag: tag(0x0028, 0x0010),
            vr: [0, 0],
            items: Vec::new(),
            bytes: vec![0x32, 0x00],
        };
        assert_eq!(int_tag(&[elem], tag(0x0028, 0x0010)), Some(50));
    }

    #[test]
    fn sequence_nesting_has_a_ceiling() {
        // 300 nested defined-length sequences — explicit VR element +
        // item wrapper per level — must Err, not overflow the stack.
        let mut inner: Vec<u8> = Vec::new();
        for _ in 0..300 {
            let mut level = Vec::new();
            level.extend_from_slice(&[0x0A, 0x30, 0xB0, 0x00]); // tag
            level.extend_from_slice(b"SQ\x00\x00"); // VR + reserved
            let item_len = inner.len() as u32;
            level.extend_from_slice(&(8 + item_len).to_le_bytes());
            level.extend_from_slice(&[0xFE, 0xFF, 0x00, 0xE0]); // item tag
            level.extend_from_slice(&item_len.to_le_bytes());
            level.extend_from_slice(&inner);
            inner = level;
        }
        let err = match parse_dataset_mode(&inner, false, 0) {
            Err(err) => err,
            Ok(_) => panic!("nesting must fail past the depth cap"),
        };
        assert!(err.contains("deep"), "{err}");
    }

    #[test]
    fn a_truncated_element_fails_the_whole_dataset() {
        let mut bytes = Vec::new();
        us(&mut bytes, 0x0028, 0x0010, 1).unwrap();
        // A full header declaring 4 bytes, then nothing — the declared end
        // runs past EOF; returning the prefix would silently drop every tag.
        bytes.extend_from_slice(&[0x28, 0x00, 0x11, 0x00, b'U', b'S', 0x04, 0x00]);
        assert!(parse_dataset(&bytes).is_err());
    }

    #[test]
    fn fd_and_od_values_decode_as_doubles() {
        // FD is the spec VR for the 300A spot position/meterset maps — each
        // 8-byte double must not split into two f32s or text-parse to empty.
        let doubles = [1.5f64, -2.25, 3.0];
        let raw: Vec<u8> = doubles
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let mut bytes = Vec::new();
        push_elem(&mut bytes, 0x300A, 0x0394, b"FD", &raw).unwrap();
        push_elem(&mut bytes, 0x3004, 0x000C, b"OD", &raw).unwrap();
        let elems = parse_dataset(&bytes).unwrap();
        assert_eq!(
            floats_tag(&elems, tag(0x300A, 0x0394)),
            vec![1.5, -2.25, 3.0]
        );
        assert_eq!(
            floats_tag(&elems, tag(0x3004, 0x000C)),
            vec![1.5, -2.25, 3.0]
        );
    }

    #[test]
    fn a_us_value_decodes_from_the_vr_not_the_byte_shape() {
        // 0x3231 in US bytes is the ASCII text "12" — a byte-shape guess
        // would read 12 where the VR says binary 12849.
        let mut bytes = Vec::new();
        us(&mut bytes, 0x0028, 0x0010, 0x3231).unwrap();
        let elems = parse_dataset(&bytes).unwrap();
        assert_eq!(int_tag(&elems, tag(0x0028, 0x0010)), Some(0x3231));
    }

    #[test]
    fn an_un_element_with_item_markers_reads_as_a_sequence() {
        let mut item = Vec::new();
        us(&mut item, 0x0028, 0x0010, 7).unwrap();
        let mut body = Vec::new();
        body.extend_from_slice(&0xFFFEu16.to_le_bytes());
        body.extend_from_slice(&0xE000u16.to_le_bytes());
        body.extend_from_slice(&(item.len() as u32).to_le_bytes());
        body.extend_from_slice(&item);
        // UN is a long VR — the header is hand-built: tag, "UN", reserved,
        // u32 length.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0x300Au16.to_le_bytes());
        bytes.extend_from_slice(&0x9999u16.to_le_bytes());
        bytes.extend_from_slice(b"UN");
        bytes.extend_from_slice(&[0, 0]);
        bytes.extend_from_slice(&(body.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&body);
        let elems = parse_dataset(&bytes).unwrap();
        let un = elem(&elems, tag(0x300A, 0x9999)).unwrap();
        assert_eq!(un.items.len(), 1);
        assert_eq!(int_tag(&un.items[0], tag(0x0028, 0x0010)), Some(7));

        // The same tag without the item marker stays opaque bytes.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0x300Au16.to_le_bytes());
        bytes.extend_from_slice(&0x9999u16.to_le_bytes());
        bytes.extend_from_slice(b"UN");
        bytes.extend_from_slice(&[0, 0]);
        bytes.extend_from_slice(&4u32.to_le_bytes());
        bytes.extend_from_slice(&[1, 2, 3, 4]);
        let elems = parse_dataset(&bytes).unwrap();
        let un = elem(&elems, tag(0x300A, 0x9999)).unwrap();
        assert!(un.items.is_empty());
        assert_eq!(un.bytes, vec![1, 2, 3, 4]);
    }

    #[test]
    fn a_multiframe_image_without_frame_offsets_fails() {
        let mut bytes = Vec::new();
        us(&mut bytes, 0x0028, 0x0010, 1).unwrap();
        us(&mut bytes, 0x0028, 0x0011, 1).unwrap();
        ds(&mut bytes, 0x0028, 0x0030, &[1.0, 2.0]).unwrap();
        ds(&mut bytes, 0x0020, 0x0032, &[0.0, 0.0, 10.0]).unwrap();
        us(&mut bytes, 0x0028, 0x0100, 16).unwrap();
        us(&mut bytes, 0x0028, 0x0103, 1).unwrap();
        ds(&mut bytes, 0x0028, 0x1052, &[0.0]).unwrap();
        ds(&mut bytes, 0x0028, 0x1053, &[1.0]).unwrap();
        ds(&mut bytes, 0x0018, 0x0050, &[2.5]).unwrap();
        ow(&mut bytes, 0x7FE0, 0x0010, &[1, 2]).unwrap();
        // Two frames and no GridFrameOffsetVector — the old fallback
        // invented positions at frame * SliceThickness.
        assert!(image_slices(&parse_dataset(&bytes).unwrap(), 1.0).is_err());
    }

    #[test]
    fn a_non_finite_or_unencodable_dose_is_refused() {
        // Shares no temp path with a_rt_dose_round_trips — parallel tests
        // writing the same dose.dcm race on the atomic-write part file.
        let dir =
            std::env::temp_dir().join(format!("scan-kit-rtdose-refused-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let ct = CtVolume {
            frame: "1.2.3.4.5".into(),
            origin: [0.0; 3],
            spacing: [1.0; 3],
            shape: [2, 2, 1],
            hu: Vec::new(),
            position: "HFS".into(),
            dose_type: String::new(),
            summation: String::new(),
            patient_id: "P1".into(),
            patient_name: "A^B".into(),
            study_uid: "1.2.3".into(),
            series_uid: "1.2.3.1".into(),
            sop_uids: vec!["1.2.3.1.1".into()],
        };
        let path = dir.join("dose.dcm");
        // inf used to quantize the whole cube to 0 and write "inf" as the
        // scaling — a file the reader itself rejects.
        assert!(write_rtdose(
            &path,
            &[1.0, f32::INFINITY, 0.0, 0.0],
            &ct,
            "",
            "PLAN",
            "PHYSICAL",
            "d",
            ""
        )
        .is_err());
        // -inf never raises the peak — it needs its own rejection.
        assert!(write_rtdose(
            &path,
            &[1.0, f32::NEG_INFINITY, 0.0, 0.0],
            &ct,
            "",
            "PLAN",
            "PHYSICAL",
            "d",
            ""
        )
        .is_err());
        // A peak this small makes DoseGridScaling underflow to 0 on read.
        assert!(write_rtdose(&path, &[1e-40f32; 4], &ct, "", "PLAN", "PHYSICAL", "d", "").is_err());
        // A zero axis or non-positive spacing writes a file the reader
        // itself refuses.
        let flat = CtVolume {
            shape: [0, 2, 1],
            ..ct.clone()
        };
        assert!(write_rtdose(&path, &[0.0; 4], &flat, "", "PLAN", "PHYSICAL", "d", "").is_err());
        let sunk = CtVolume {
            spacing: [-1.0, 1.0, 1.0],
            ..ct.clone()
        };
        assert!(write_rtdose(&path, &[0.0; 4], &sunk, "", "PLAN", "PHYSICAL", "d", "").is_err());
        // dose_type and summation take defined terms, not free text.
        assert!(write_rtdose(
            &path,
            &[0.0f32, 1.0, 2.0, 3.0],
            &ct,
            "",
            "PLAN",
            "custom",
            "d",
            ""
        )
        .is_err());
        assert!(write_rtdose(
            &path,
            &[0.0f32, 1.0, 2.0, 3.0],
            &ct,
            "",
            "custom",
            "PHYSICAL",
            "d",
            ""
        )
        .is_err());
        // The sane path still writes.
        assert!(write_rtdose(
            &path,
            &[0.0f32, 1.0, 2.0, 3.0],
            &ct,
            "",
            "PLAN",
            "PHYSICAL",
            "d",
            ""
        )
        .is_ok());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn ds_refuses_non_finite_and_overlong_values() {
        let mut bytes = Vec::new();
        assert!(ds(&mut bytes, 0x0018, 0x0050, &[f32::NAN]).is_err());
        assert!(ds(&mut bytes, 0x0018, 0x0050, &[f32::INFINITY]).is_err());
        // 1e10 formats to 18 chars — over the 16-byte DS limit.
        assert!(ds(&mut bytes, 0x0018, 0x0050, &[1e10]).is_err());
        // The limit is per component — one overlong element is not
        // diluted by short siblings.
        assert!(ds(&mut bytes, 0x0018, 0x0050, &[1e10, 1.5]).is_err());
        assert!(ds(&mut bytes, 0x0018, 0x0050, &[1.5, 2.5]).is_ok());
        assert!(ds(&mut bytes, 0x0018, 0x0050, &[1.5]).is_ok());
    }

    #[test]
    fn water_study_tags_ascend_and_modality_is_cs() {
        let dir = std::env::temp_dir().join(format!("scan-kit-water-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        write_water_study(&dir).unwrap();
        let ct = std::fs::read(dir.join("ct0.dcm")).unwrap();
        let find = |hay: &[u8], needle: [u8; 4]| hay.windows(4).position(|w| w == needle).unwrap();
        // (0020,0032) < (0028,0010) and 1052 < 1053, CT written as CS.
        assert!(find(&ct, [0x20, 0x00, 0x32, 0x00]) < find(&ct, [0x28, 0x00, 0x10, 0x00]));
        assert!(find(&ct, [0x28, 0x00, 0x52, 0x10]) < find(&ct, [0x28, 0x00, 0x53, 0x10]));
        let modality = find(&ct, [0x08, 0x00, 0x60, 0x00]);
        assert_eq!(&ct[modality + 4..modality + 6], b"CS");
        let plan = std::fs::read(dir.join("plan.dcm")).unwrap();
        // (300A,0070) < (300A,0180); inside the beam 010E < 0308 < 03A8 < 300C,006A.
        assert!(find(&plan, [0x0A, 0x30, 0x70, 0x00]) < find(&plan, [0x0A, 0x30, 0x80, 0x01]));
        assert!(find(&plan, [0x0A, 0x30, 0x0E, 0x01]) < find(&plan, [0x0A, 0x30, 0x08, 0x03]));
        assert!(find(&plan, [0x0A, 0x30, 0xA8, 0x03]) < find(&plan, [0x0C, 0x30, 0x6A, 0x00]));
        let structures = std::fs::read(dir.join("struct.dcm")).unwrap();
        assert!(
            find(&structures, [0x06, 0x30, 0x40, 0x00])
                < find(&structures, [0x06, 0x30, 0x84, 0x00])
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_weight_map_short_of_the_spot_limit_is_an_error() {
        // 3 declared spots with a single weight would silently zero weights
        // 2 and 3 — the export must be rejected instead.
        assert!(read_plan(&plan_with_weights(3, 3, 1), "f".into(), "HFS").is_err());
        assert!(read_plan(&plan_with_weights(3, 3, 3), "f".into(), "HFS").is_ok());
    }

    #[test]
    fn a_modulated_beam_without_weights_is_an_error() {
        // ScanSpotMetersetWeights is Type 1C under MODULATED — absent, every
        // spot computes mu = 0 and the beam silently empties.
        let mut ref_beam = Vec::new();
        is(&mut ref_beam, 0x300C, 0x0006, 1).unwrap();
        ds(&mut ref_beam, 0x300A, 0x0086, &[10.0]).unwrap();
        let mut group = Vec::new();
        is(&mut group, 0x300A, 0x0078, 1).unwrap();
        sq(&mut group, 0x300C, 0x0004, &[ref_beam]).unwrap();
        let mut cp = Vec::new();
        ds(&mut cp, 0x300A, 0x0114, &[100.0]).unwrap();
        ds(&mut cp, 0x300A, 0x011E, &[0.0]).unwrap();
        ds(&mut cp, 0x300A, 0x0122, &[0.0]).unwrap();
        ds(&mut cp, 0x300A, 0x012C, &[0.0, 0.0, 0.0]).unwrap();
        is(&mut cp, 0x300A, 0x0392, 2).unwrap();
        ds(&mut cp, 0x300A, 0x0394, &[1.0, -1.0, 2.0, -2.0]).unwrap();
        let mut beam = Vec::new();
        is(&mut beam, 0x300A, 0x00C0, 1).unwrap();
        ds(&mut beam, 0x300A, 0x010E, &[1.0]).unwrap();
        cs(&mut beam, 0x300A, 0x0308, "MODULATED").unwrap();
        is(&mut beam, 0x300C, 0x006A, 1).unwrap();
        sq(&mut beam, 0x300A, 0x03A8, &[cp]).unwrap();
        let mut plan = Vec::new();
        sq(&mut plan, 0x300A, 0x0070, &[group]).unwrap();
        sq(&mut plan, 0x300A, 0x03A2, &[beam]).unwrap();
        assert!(read_plan(&parse_dataset(&plan).unwrap(), "f".into(), "HFS").is_err());
    }

    #[test]
    fn an_oversized_item_length_is_not_a_sequence() {
        // A binary value can begin with the FFFE,E000 item bytes — the
        // first item's declared length must still fit inside the value, or
        // a pair of stored 0xFFFE/0xE000 words would parse as phantom items.
        let mut value = vec![0xFE, 0xFF, 0x00, 0xE0];
        value.extend_from_slice(&100u32.to_le_bytes());
        value.extend_from_slice(&[0; 8]);
        assert!(!looks_like_items(&value));
        let mut value = vec![0xFE, 0xFF, 0x00, 0xE0];
        value.extend_from_slice(&0u32.to_le_bytes());
        assert!(looks_like_items(&value));
    }

    #[test]
    fn a_sequence_ending_mid_item_header_is_an_error() {
        let mut bytes = vec![0xFE, 0xFF, 0x00, 0xE0];
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&[0xFE, 0xFF, 0x00]); // half an item header
        assert!(parse_items(&bytes, true, 0).is_err());
        // An all-zero tail stays tolerated padding.
        let mut padded = vec![0xFE, 0xFF, 0x00, 0xE0];
        padded.extend_from_slice(&0u32.to_le_bytes());
        padded.extend_from_slice(&[0, 0, 0]);
        assert!(parse_items(&padded, true, 0).is_ok());
    }

    #[test]
    fn a_partial_pixel_frame_is_an_error() {
        let mut bytes = Vec::new();
        us(&mut bytes, 0x0028, 0x0010, 2).unwrap();
        us(&mut bytes, 0x0028, 0x0011, 2).unwrap();
        ds(&mut bytes, 0x0028, 0x0030, &[1.0, 2.0]).unwrap();
        ds(&mut bytes, 0x0020, 0x0032, &[0.0, 0.0, 10.0]).unwrap();
        us(&mut bytes, 0x0028, 0x0100, 16).unwrap();
        us(&mut bytes, 0x0028, 0x0103, 1).unwrap();
        ds(&mut bytes, 0x0028, 0x1052, &[0.0]).unwrap();
        ds(&mut bytes, 0x0028, 0x1053, &[1.0]).unwrap();
        ds(&mut bytes, 0x0018, 0x0050, &[2.5]).unwrap();
        // A 2x2 plane is 4 words — five ends inside the second frame.
        ow(&mut bytes, 0x7FE0, 0x0010, &[1, 2, 3, 4, 5]).unwrap();
        assert!(image_slices(&parse_dataset(&bytes).unwrap(), 1.0).is_err());
    }

    #[test]
    fn a_beam_without_beam_geometry_is_an_error() {
        // Gantry/couch/isocenter are Type 1 in the control point; a plan that
        // omits all of them must not ship zeroed origin geometry.
        let mut cp = Vec::new();
        ds(&mut cp, 0x300A, 0x0114, &[100.0]).unwrap();
        is(&mut cp, 0x300A, 0x0392, 1).unwrap();
        ds(&mut cp, 0x300A, 0x0394, &[1.0, -2.0]).unwrap();
        ds(&mut cp, 0x300A, 0x0396, &[1.0]).unwrap();
        let mut beam = Vec::new();
        is(&mut beam, 0x300A, 0x00C0, 1).unwrap();
        ds(&mut beam, 0x300A, 0x010E, &[1.0]).unwrap();
        sq(&mut beam, 0x300A, 0x03A8, &[cp]).unwrap();
        let mut plan = Vec::new();
        sq(&mut plan, 0x300A, 0x03A2, &[beam]).unwrap();
        assert!(read_plan(&parse_dataset(&plan).unwrap(), "f".into(), "HFS").is_err());
    }

    #[test]
    fn a_nan_energy_does_not_poison_the_next_control_point() {
        // A corrupt NaN energy must not carry forward and silently drop the
        // following control point's spots.
        let mut cp0 = Vec::new();
        ds(&mut cp0, 0x300A, 0x0114, &[100.0]).unwrap();
        is(&mut cp0, 0x300A, 0x0392, 1).unwrap();
        ds(&mut cp0, 0x300A, 0x0394, &[1.0, 2.0]).unwrap();
        ds(&mut cp0, 0x300A, 0x0396, &[0.5]).unwrap();
        let mut cp1 = Vec::new();
        push_elem(&mut cp1, 0x300A, 0x0114, b"DS", b"nan").unwrap();
        is(&mut cp1, 0x300A, 0x0392, 1).unwrap();
        ds(&mut cp1, 0x300A, 0x0394, &[3.0, 4.0]).unwrap();
        ds(&mut cp1, 0x300A, 0x0396, &[0.25]).unwrap();
        let mut cp2 = Vec::new();
        is(&mut cp2, 0x300A, 0x0392, 1).unwrap();
        ds(&mut cp2, 0x300A, 0x0394, &[5.0, 6.0]).unwrap();
        ds(&mut cp2, 0x300A, 0x0396, &[0.25]).unwrap();
        let mut beam = Vec::new();
        sq(&mut beam, 0x300A, 0x03A8, &[cp0, cp1, cp2]).unwrap();
        let mut bytes = preamble();
        ui(&mut bytes, 0x0008, 0x0016, "1.2.840.10008.5.1.4.1.1.481.8").unwrap();
        sq(&mut bytes, 0x300A, 0x03A2, &[beam]).unwrap();
        let path = std::env::temp_dir().join(format!("scan-kit-nan-{}.dcm", std::process::id()));
        std::fs::write(&path, bytes).unwrap();
        let plan = read_ion_plan(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        // The NaN control point keeps the previous 100 MeV rather than
        // poisoning itself and cp2 with a NaN carry-forward.
        assert_eq!(plan.spots.len(), 3);
    }

    #[test]
    fn a_dose_without_grid_scaling_is_an_error() {
        // DoseGridScaling is Type 1 — without it the raw stored counts would
        // be read as Gy, off by the scale's ~1e-5.
        let root = std::env::temp_dir().join(format!("scan-kit-noscale-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let mut bytes = preamble();
        ui(&mut bytes, 0x0008, 0x0016, "1.2.840.10008.5.1.4.1.1.481.2").unwrap();
        std::fs::write(root.join("dose.dcm"), bytes).unwrap();
        // Assert the guard's own error — a bare is_err() would pass on any
        // later failure (missing pixels, no CT) even with the check removed.
        let err = load_study(&root).err().unwrap();
        assert!(err.contains("dose grid scaling"), "{err}");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_trailing_partial_element_is_an_error() {
        let mut bytes = Vec::new();
        iv(&mut bytes, 0x0008, 0x0016, &even("1.2"));
        let mut junk = bytes.clone();
        junk.extend_from_slice(&[0x10, 0x00, 0x20]);
        assert!(parse_dataset_mode(&junk, true, 0).is_err());
        // Zero padding after the last element is tolerated.
        let mut padded = bytes;
        padded.extend_from_slice(&[0; 5]);
        assert!(parse_dataset_mode(&padded, true, 0).is_ok());
    }

    #[test]
    fn absolute_frame_offsets_are_not_read_as_relative() {
        // Option (b) of the GFOV spec stores absolute patient z - reading it
        // as relative shifted every frame by the origin z.
        let mut bytes = Vec::new();
        us(&mut bytes, 0x0028, 0x0010, 1).unwrap();
        us(&mut bytes, 0x0028, 0x0011, 1).unwrap();
        ds(&mut bytes, 0x0028, 0x0030, &[1.0, 1.0]).unwrap();
        ds(&mut bytes, 0x0020, 0x0032, &[0.0, 0.0, -50.0]).unwrap();
        ds(&mut bytes, 0x0020, 0x0037, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).unwrap();
        ds(&mut bytes, 0x3004, 0x000C, &[-50.0, -48.0]).unwrap();
        us(&mut bytes, 0x0028, 0x0100, 16).unwrap();
        ow(&mut bytes, 0x7FE0, 0x0010, &[1, 2]).unwrap();
        let slices = image_slices(&parse_dataset(&bytes).unwrap(), 1.0).unwrap();
        assert!((slices[0].origin[2] + 50.0).abs() < 1e-3);
        assert!((slices[1].origin[2] + 48.0).abs() < 1e-3);
    }

    #[test]
    fn a_twelve_bit_signed_pixel_is_sign_extended() {
        // High Bit 11 marks the sign of a 12-bit sample in a 16-bit slot -
        // 0x0800 is -2048, not +2048.
        let mut bytes = Vec::new();
        us(&mut bytes, 0x0028, 0x0010, 1).unwrap();
        us(&mut bytes, 0x0028, 0x0011, 1).unwrap();
        ds(&mut bytes, 0x0028, 0x0030, &[1.0, 1.0]).unwrap();
        ds(&mut bytes, 0x0020, 0x0032, &[0.0, 0.0, 0.0]).unwrap();
        us(&mut bytes, 0x0028, 0x0100, 16).unwrap();
        us(&mut bytes, 0x0028, 0x0101, 12).unwrap();
        us(&mut bytes, 0x0028, 0x0102, 11).unwrap();
        us(&mut bytes, 0x0028, 0x0103, 1).unwrap();
        ow(&mut bytes, 0x7FE0, 0x0010, &[0x0800]).unwrap();
        let slices = image_slices(&parse_dataset(&bytes).unwrap(), 1.0).unwrap();
        assert!((slices[0].values[0] + 2048.0).abs() < 1e-3);
    }

    #[test]
    fn a_corrupt_rescale_slope_is_an_error() {
        let mut bytes = Vec::new();
        us(&mut bytes, 0x0028, 0x0010, 1).unwrap();
        us(&mut bytes, 0x0028, 0x0011, 1).unwrap();
        ds(&mut bytes, 0x0028, 0x0030, &[1.0, 1.0]).unwrap();
        ds(&mut bytes, 0x0020, 0x0032, &[0.0, 0.0, 0.0]).unwrap();
        us(&mut bytes, 0x0028, 0x0100, 16).unwrap();
        let mut slope = Vec::new();
        slope.extend_from_slice(b"2ad");
        let header = [0x28u8, 0x00, 0x53, 0x10, b'D', b'S', 0x05, 0x00];
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(b" ");
        bytes.extend_from_slice(&slope);
        ow(&mut bytes, 0x7FE0, 0x0010, &[0]).unwrap();
        assert!(image_slices(&parse_dataset(&bytes).unwrap(), 1.0).is_err());
    }

    fn uniform_beam_group() -> Vec<u8> {
        let mut ref_beam = Vec::new();
        is(&mut ref_beam, 0x300C, 0x0006, 1).unwrap();
        ds(&mut ref_beam, 0x300A, 0x0086, &[10.0]).unwrap();
        let mut group = Vec::new();
        is(&mut group, 0x300A, 0x0078, 1).unwrap();
        sq(&mut group, 0x300C, 0x0004, &[ref_beam]).unwrap();
        group
    }

    #[test]
    fn a_uniform_beam_shares_its_meterset_across_spots() {
        // No Scan Spot Meterset Weights on a UNIFORM scan is not "every spot
        // is zero" — the beam meterset divides evenly over the spot map.
        let group = uniform_beam_group();
        let mut cp = Vec::new();
        ds(&mut cp, 0x300A, 0x0114, &[100.0]).unwrap();
        ds(&mut cp, 0x300A, 0x011E, &[0.0]).unwrap();
        ds(&mut cp, 0x300A, 0x0122, &[0.0]).unwrap();
        ds(&mut cp, 0x300A, 0x012C, &[0.0, 0.0, 0.0]).unwrap();
        is(&mut cp, 0x300A, 0x0392, 2).unwrap();
        ds(&mut cp, 0x300A, 0x0394, &[1.0, -1.0, 2.0, -2.0]).unwrap();
        let mut beam = Vec::new();
        is(&mut beam, 0x300A, 0x00C0, 1).unwrap();
        cs(&mut beam, 0x300A, 0x0308, "UNIFORM").unwrap();
        ds(&mut beam, 0x300A, 0x010E, &[1.0]).unwrap();
        is(&mut beam, 0x300C, 0x006A, 1).unwrap();
        sq(&mut beam, 0x300A, 0x03A8, &[cp]).unwrap();
        let mut plan = Vec::new();
        sq(&mut plan, 0x300A, 0x0070, &[group]).unwrap();
        sq(&mut plan, 0x300A, 0x03A2, &[beam]).unwrap();
        let parsed = read_plan(&parse_dataset(&plan).unwrap(), "f".into(), "HFS").unwrap();
        assert_eq!(parsed.beams[0].spots.len(), 2);
        for spot in &parsed.beams[0].spots {
            assert!((spot.mu - 5.0).abs() < 1e-4, "{}", spot.mu);
        }
    }

    #[test]
    fn a_beam_without_weights_or_scan_mode_is_an_error() {
        // Scan Mode is Type 1C — absent with no weight map the loader cannot
        // pick UNIFORM vs MODULATED and must refuse, not emit a zero beam.
        let group = uniform_beam_group();
        let mut cp = Vec::new();
        ds(&mut cp, 0x300A, 0x0114, &[100.0]).unwrap();
        ds(&mut cp, 0x300A, 0x011E, &[0.0]).unwrap();
        ds(&mut cp, 0x300A, 0x0122, &[0.0]).unwrap();
        ds(&mut cp, 0x300A, 0x012C, &[0.0, 0.0, 0.0]).unwrap();
        is(&mut cp, 0x300A, 0x0392, 1).unwrap();
        ds(&mut cp, 0x300A, 0x0394, &[1.0, -1.0]).unwrap();
        for mode in ["", "MODULATED"] {
            let mut beam = Vec::new();
            is(&mut beam, 0x300A, 0x00C0, 1).unwrap();
            if !mode.is_empty() {
                cs(&mut beam, 0x300A, 0x0308, mode).unwrap();
            }
            ds(&mut beam, 0x300A, 0x010E, &[1.0]).unwrap();
            is(&mut beam, 0x300C, 0x006A, 1).unwrap();
            sq(&mut beam, 0x300A, 0x03A8, &[cp.clone()]).unwrap();
            let mut plan = Vec::new();
            sq(&mut plan, 0x300A, 0x0070, std::slice::from_ref(&group)).unwrap();
            sq(&mut plan, 0x300A, 0x03A2, &[beam]).unwrap();
            let message = read_plan(&parse_dataset(&plan).unwrap(), "f".into(), "HFS")
                .err()
                .unwrap_or_default();
            assert!(
                message.contains("scan mode") || message.contains("weights"),
                "{message}"
            );
        }
    }

    #[test]
    fn a_second_populated_fraction_group_is_rejected() {
        // Metersets pool across groups and first-match wins — a second group
        // with beam refs would misattribute dose silently.
        let group = uniform_beam_group();
        let mut second = Vec::new();
        is(&mut second, 0x300A, 0x0078, 3).unwrap();
        let mut ref_beam = Vec::new();
        is(&mut ref_beam, 0x300C, 0x0006, 9).unwrap();
        ds(&mut ref_beam, 0x300A, 0x0086, &[4.0]).unwrap();
        sq(&mut second, 0x300C, 0x0004, &[ref_beam]).unwrap();
        let mut cp = Vec::new();
        ds(&mut cp, 0x300A, 0x0114, &[100.0]).unwrap();
        ds(&mut cp, 0x300A, 0x011E, &[0.0]).unwrap();
        ds(&mut cp, 0x300A, 0x0122, &[0.0]).unwrap();
        ds(&mut cp, 0x300A, 0x012C, &[0.0, 0.0, 0.0]).unwrap();
        is(&mut cp, 0x300A, 0x0392, 1).unwrap();
        ds(&mut cp, 0x300A, 0x0394, &[1.0, -1.0]).unwrap();
        ds(&mut cp, 0x300A, 0x0396, &[1.0]).unwrap();
        let mut beam = Vec::new();
        is(&mut beam, 0x300A, 0x00C0, 1).unwrap();
        ds(&mut beam, 0x300A, 0x010E, &[1.0]).unwrap();
        is(&mut beam, 0x300C, 0x006A, 1).unwrap();
        sq(&mut beam, 0x300A, 0x03A8, &[cp]).unwrap();
        let mut plan = Vec::new();
        sq(&mut plan, 0x300A, 0x0070, &[group, second]).unwrap();
        sq(&mut plan, 0x300A, 0x03A2, &[beam]).unwrap();
        let message = read_plan(&parse_dataset(&plan).unwrap(), "f".into(), "HFS")
            .err()
            .unwrap_or_default();
        assert!(message.contains("fraction group"), "{message}");
    }

    #[test]
    fn spot_sizes_index_per_spot() {
        // ScanningSpotSize is 2·n pairs in spot order — spot two must not
        // inherit spot one's size.
        let group = uniform_beam_group();
        let mut cp = Vec::new();
        ds(&mut cp, 0x300A, 0x0114, &[100.0]).unwrap();
        ds(&mut cp, 0x300A, 0x011E, &[0.0]).unwrap();
        ds(&mut cp, 0x300A, 0x0122, &[0.0]).unwrap();
        ds(&mut cp, 0x300A, 0x012C, &[0.0, 0.0, 0.0]).unwrap();
        is(&mut cp, 0x300A, 0x0392, 2).unwrap();
        ds(&mut cp, 0x300A, 0x0394, &[1.0, -1.0, 2.0, -2.0]).unwrap();
        ds(&mut cp, 0x300A, 0x0396, &[1.0, 1.0]).unwrap();
        ds(&mut cp, 0x300A, 0x0398, &[4.0, 4.0, 8.0, 8.0]).unwrap();
        let mut beam = Vec::new();
        is(&mut beam, 0x300A, 0x00C0, 1).unwrap();
        ds(&mut beam, 0x300A, 0x010E, &[1.0]).unwrap();
        is(&mut beam, 0x300C, 0x006A, 1).unwrap();
        sq(&mut beam, 0x300A, 0x03A8, &[cp]).unwrap();
        let mut plan = Vec::new();
        sq(&mut plan, 0x300A, 0x0070, &[group]).unwrap();
        sq(&mut plan, 0x300A, 0x03A2, &[beam]).unwrap();
        let parsed = read_plan(&parse_dataset(&plan).unwrap(), "f".into(), "HFS").unwrap();
        assert!((parsed.beams[0].spots[0].spot_size_mm - 4.0).abs() < 1e-4);
        assert!((parsed.beams[0].spots[1].spot_size_mm - 8.0).abs() < 1e-4);
    }
}
