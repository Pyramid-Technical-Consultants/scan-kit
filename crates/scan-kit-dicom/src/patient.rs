//! CT, RTSTRUCT, RT Ion Plan, and RTDOSE. One frame of reference, or the load fails.

use std::fs;
use std::path::{Path, PathBuf};

const DENSITY: &str =
    include_str!("../../../scan_kit/assets/mcsquare/Scanners/default/HU_Density_Conversion.txt");
const MATERIALS: &str =
    include_str!("../../../scan_kit/assets/mcsquare/Scanners/default/HU_Material_Conversion.txt");

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
    pub contour: Vec<[f32; 3]>,
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
    items: Vec<Vec<Elem>>,
    bytes: Vec<u8>,
}

pub fn load_study(path: &Path) -> Result<PatientStudy, String> {
    let mut ct_slices = Vec::new();
    let mut dose_slices = Vec::new();
    let mut dose_type = String::new();
    let mut summation = String::new();
    let mut plan = None;
    let mut structures = Vec::new();
    let mut patient_id = String::new();
    let mut patient_name = String::new();
    let mut study_uid = String::new();
    let mut series_uid = String::new();
    let mut sop_uids = Vec::new();
    let mut structures_uid = String::new();
    for item in fs::read_dir(path).map_err(|err| err.to_string())? {
        let file = item.map_err(|err| err.to_string())?.path();
        let bytes = fs::read(&file).unwrap_or_default();
        if bytes.len() < 132 || &bytes[128..132] != b"DICM" {
            continue;
        }
        let elems = parse_dataset(&bytes[132..])?;
        let class = text_tag(&elems, tag(0x0008, 0x0016));
        let frame = text_tag(&elems, tag(0x0020, 0x0052));
        if class.contains("1.2.840.10008.5.1.4.1.1.2") {
            if patient_id.is_empty() {
                patient_id = text_tag(&elems, tag(0x0010, 0x0020));
                patient_name = text_tag(&elems, tag(0x0010, 0x0010));
                study_uid = text_tag(&elems, tag(0x0020, 0x000D));
                series_uid = text_tag(&elems, tag(0x0020, 0x000E));
            }
            sop_uids.push(text_tag(&elems, tag(0x0008, 0x0018)));
            for slice in image_slices(&elems, 1.0)? {
                ct_slices.push((frame.clone(), slice));
            }
        } else if class.contains("1.2.840.10008.5.1.4.1.1.481.8") {
            let sop_uid = text_tag(&elems, tag(0x0008, 0x0018));
            let mut parsed = read_plan(&elems, frame)?;
            parsed.sop_uid = sop_uid;
            plan = Some(parsed);
        } else if class.contains("1.2.840.10008.5.1.4.1.1.481.3") {
            structures_uid = text_tag(&elems, tag(0x0008, 0x0018));
            structures = read_structures(&elems);
        } else if class.contains("1.2.840.10008.5.1.4.1.1.481.2") {
            let scale = float_tag(&elems, tag(0x3004, 0x000E)).unwrap_or(1.0);
            if dose_type.is_empty() {
                dose_type = text_tag(&elems, tag(0x3004, 0x0004));
                summation = text_tag(&elems, tag(0x3004, 0x000A));
            }
            for slice in image_slices(&elems, scale)? {
                dose_slices.push((frame.clone(), slice));
            }
        }
    }
    if ct_slices.is_empty() {
        return Err("the study has no CT".into());
    }
    let frame = ct_slices[0].0.clone();
    if frame.is_empty() || ct_slices.iter().any(|(got, _)| got != &frame) {
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
    if plan.frame != frame {
        return Err("the plan frame of reference does not match the CT".into());
    }
    if dose_slices.iter().any(|(got, _)| got != &frame) {
        return Err("the dose frame of reference does not match the CT".into());
    }
    let mut slices: Vec<_> = ct_slices.into_iter().map(|(_, slice)| slice).collect();
    let mut ct = stack(&mut slices)?;
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
    let bytes = fs::read(path).map_err(|err| format!("Could not read DICOM plan file: {err}"))?;
    if bytes.len() < 132 || &bytes[128..132] != b"DICM" {
        return Err(format!(
            "Could not read DICOM plan file: {}",
            path.display()
        ));
    }
    let elems = parse_dataset(&bytes[132..])?;
    let label = text_tag(&elems, tag(0x300A, 0x0002));
    let mut spots = Vec::new();
    let mut plan_index = 0i32;
    for beam in seq(&elems, tag(0x300A, 0x03A2)) {
        let mut energy = 0.0f32;
        for cp in seq(beam, tag(0x300A, 0x03A8)) {
            energy = float_tag(cp, tag(0x300A, 0x0114)).unwrap_or(energy);
            let n = int_tag(cp, tag(0x300A, 0x0392)).unwrap_or(0).max(0) as usize;
            let xy = floats_tag(cp, tag(0x300A, 0x0394));
            if n == 0 || xy.is_empty() {
                continue;
            }
            let weights = floats_tag(cp, tag(0x300A, 0x0396));
            let has_weights = elem(cp, tag(0x300A, 0x0396)).is_some();
            let size = floats_tag(cp, tag(0x300A, 0x0398));
            let (size_x, size_y) = if size.len() >= 2 {
                (size[0], size[1])
            } else {
                (f32::NAN, f32::NAN)
            };
            let limit = n.min(xy.len() / 2);
            for i in 0..limit {
                let mu = if has_weights {
                    weights.get(i).copied().unwrap_or(f32::NAN)
                } else {
                    f32::NAN
                };
                let index = plan_index;
                plan_index += 1;
                if has_weights && (!mu.is_finite() || mu <= 0.0) {
                    continue;
                }
                spots.push(IonPlanSpot {
                    x: xy[i * 2],
                    y: xy[i * 2 + 1],
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

const WATER_DENSITY: &str = include_str!(
    "../../../scan_kit/assets/mcsquare/Scanners/Water_Phantom/HU_Density_Conversion.txt"
);
const WATER_MATERIAL: &str = include_str!(
    "../../../scan_kit/assets/mcsquare/Scanners/Water_Phantom/HU_Material_Conversion.txt"
);
const SOLID_DENSITY: &str = include_str!(
    "../../../scan_kit/assets/mcsquare/Scanners/SolidWater_Phantom/HU_Density_Conversion.txt"
);
const SOLID_MATERIAL: &str = include_str!(
    "../../../scan_kit/assets/mcsquare/Scanners/SolidWater_Phantom/HU_Material_Conversion.txt"
);

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
        ui(&mut bytes, 0x0008, 0x0016, "1.2.840.10008.5.1.4.1.1.2");
        ui(&mut bytes, 0x0008, 0x0060, "CT");
        ui(&mut bytes, 0x0020, 0x000D, "1.2.3");
        ui(&mut bytes, 0x0020, 0x000E, "1.2.3.1");
        ui(&mut bytes, 0x0020, 0x0052, frame);
        us(&mut bytes, 0x0028, 0x0010, 4);
        us(&mut bytes, 0x0028, 0x0011, 4);
        ds(&mut bytes, 0x0028, 0x0030, &[1.0, 1.0]);
        ds(&mut bytes, 0x0020, 0x0032, &[0.0, 0.0, z]);
        ds(&mut bytes, 0x0020, 0x0037, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        ds(&mut bytes, 0x0028, 0x1053, &[1.0]);
        ds(&mut bytes, 0x0028, 0x1052, &[0.0]);
        ow(&mut bytes, 0x7FE0, 0x0010, &[0u16; 16]);
        fs::write(dir.join(format!("ct{index}.dcm")), bytes).map_err(|err| err.to_string())?;
    }
    let mut plan = preamble();
    ui(&mut plan, 0x0008, 0x0016, "1.2.840.10008.5.1.4.1.1.481.8");
    ui(&mut plan, 0x0008, 0x0060, "RTPLAN");
    ui(&mut plan, 0x0020, 0x0052, frame);
    let setup = {
        let mut item = Vec::new();
        is(&mut item, 0x300A, 0x0182, 1);
        cs(&mut item, 0x0018, 0x5100, "HFS");
        item
    };
    sq(&mut plan, 0x300A, 0x0180, &[setup]);
    let referenced = {
        let mut item = Vec::new();
        is(&mut item, 0x300A, 0x00C0, 1);
        ds(&mut item, 0x300A, 0x0086, &[1.0]);
        item
    };
    let fraction = {
        let mut item = Vec::new();
        is(&mut item, 0x300A, 0x0078, 1);
        sq(&mut item, 0x300C, 0x0004, &[referenced]);
        item
    };
    sq(&mut plan, 0x300A, 0x0070, &[fraction]);
    let control = {
        let mut item = Vec::new();
        ds(&mut item, 0x300A, 0x0114, &[100.0]);
        ds(&mut item, 0x300A, 0x011E, &[0.0]);
        ds(&mut item, 0x300A, 0x0122, &[0.0]);
        ds(&mut item, 0x300A, 0x012C, &[1.5, 1.5, 0.5]);
        is(&mut item, 0x300A, 0x0392, 1);
        ds(&mut item, 0x300A, 0x0394, &[0.0, 0.0]);
        ds(&mut item, 0x300A, 0x0396, &[1.0]);
        item
    };
    let beam = {
        let mut item = Vec::new();
        is(&mut item, 0x300A, 0x00C0, 1);
        lo(&mut item, 0x300A, 0x00C2, "B1");
        cs(&mut item, 0x300A, 0x00CE, "TREATMENT");
        cs(&mut item, 0x300A, 0x0308, "MODULATED");
        ds(&mut item, 0x300A, 0x010E, &[1.0]);
        is(&mut item, 0x300C, 0x006A, 1);
        sq(&mut item, 0x300A, 0x03A8, &[control]);
        item
    };
    sq(&mut plan, 0x300A, 0x03A2, &[beam]);
    fs::write(dir.join("plan.dcm"), plan).map_err(|err| err.to_string())?;
    let roi = {
        let mut item = Vec::new();
        is(&mut item, 0x3006, 0x0022, 1);
        lo(&mut item, 0x3006, 0x0026, "PTV");
        item
    };
    let contour = {
        let mut item = Vec::new();
        is(&mut item, 0x3006, 0x0046, 4);
        ds(
            &mut item,
            0x3006,
            0x0050,
            &[0.0, 0.0, 0.5, 4.0, 0.0, 0.5, 4.0, 4.0, 0.5, 0.0, 4.0, 0.5],
        );
        item
    };
    let roi_contour = {
        let mut item = Vec::new();
        is(&mut item, 0x3006, 0x0084, 1);
        sq(&mut item, 0x3006, 0x0040, &[contour]);
        item
    };
    let mut structures = preamble();
    ui(
        &mut structures,
        0x0008,
        0x0016,
        "1.2.840.10008.5.1.4.1.1.481.3",
    );
    ui(&mut structures, 0x0020, 0x0052, frame);
    sq(&mut structures, 0x3006, 0x0020, &[roi]);
    sq(&mut structures, 0x3006, 0x0039, &[roi_contour]);
    fs::write(dir.join("struct.dcm"), structures).map_err(|err| err.to_string())?;
    let mut dose = preamble();
    ui(&mut dose, 0x0008, 0x0016, "1.2.840.10008.5.1.4.1.1.481.2");
    ui(&mut dose, 0x0008, 0x0060, "RTDOSE");
    ui(&mut dose, 0x0020, 0x0052, frame);
    us(&mut dose, 0x0028, 0x0010, 4);
    us(&mut dose, 0x0028, 0x0011, 4);
    ds(&mut dose, 0x0028, 0x0030, &[1.0, 1.0]);
    ds(&mut dose, 0x0020, 0x0032, &[0.0, 0.0, 0.0]);
    ds(&mut dose, 0x0020, 0x0037, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    ds(&mut dose, 0x3004, 0x000C, &[0.0, 1.0]);
    ds(&mut dose, 0x3004, 0x000E, &[0.01]);
    ow(&mut dose, 0x7FE0, 0x0010, &[100u16; 32]);
    fs::write(dir.join("dose.dcm"), dose).map_err(|err| err.to_string())?;
    Ok(())
}

struct Slice {
    origin: [f32; 3],
    spacing_x: f32,
    spacing_y: f32,
    cols: usize,
    rows: usize,
    values: Vec<f32>,
}

fn image_slices(elems: &[Elem], scale: f32) -> Result<Vec<Slice>, String> {
    let rows = int_tag(elems, tag(0x0028, 0x0010)).unwrap_or(0) as usize;
    let cols = int_tag(elems, tag(0x0028, 0x0011)).unwrap_or(0) as usize;
    let spacing = floats_tag(elems, tag(0x0028, 0x0030));
    let origin = floats_tag(elems, tag(0x0020, 0x0032));
    let slope = float_tag(elems, tag(0x0028, 0x1053)).unwrap_or(1.0);
    let intercept = float_tag(elems, tag(0x0028, 0x1052)).unwrap_or(0.0);
    let pixels = elem(elems, tag(0x7FE0, 0x0010)).ok_or("missing pixels")?;
    if rows == 0 || cols == 0 || origin.len() < 3 || spacing.len() < 2 {
        return Err("CT geometry is incomplete".into());
    }
    let bits = int_tag(elems, tag(0x0028, 0x0100)).unwrap_or(16);
    let stored: Vec<f32> = if bits > 16 {
        pixels
            .bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| {
                (u32::from_le_bytes(*chunk) as f64 * slope as f64 + intercept as f64) as f32 * scale
            })
            .collect()
    } else {
        pixels
            .bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|chunk| (u16::from_le_bytes(*chunk) as f32 * slope + intercept) * scale)
            .collect()
    };
    let plane = rows * cols;
    if stored.len() < plane {
        return Err("pixel data is short".into());
    }
    let frames = stored.len() / plane;
    let offsets = floats_tag(elems, tag(0x3004, 0x000C));
    let mut slices = Vec::new();
    for frame in 0..frames {
        let z = origin[2] + offsets.get(frame).copied().unwrap_or(frame as f32);
        slices.push(Slice {
            origin: [origin[0], origin[1], z],
            spacing_x: spacing[1],
            spacing_y: spacing[0],
            cols,
            rows,
            values: stored[frame * plane..(frame + 1) * plane].to_vec(),
        });
    }
    Ok(slices)
}

fn stack(slices: &mut [Slice]) -> Result<CtVolume, String> {
    slices.sort_by(|a, b| a.origin[2].partial_cmp(&b.origin[2]).unwrap());
    let first = &slices[0];
    let dz = if slices.len() == 1 {
        1.0
    } else {
        slices[1].origin[2] - slices[0].origin[2]
    };
    if dz <= 0.0 {
        return Err("CT slice spacing is not uniform".into());
    }
    let mut hu = vec![0.0; first.cols * first.rows * slices.len()];
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

fn read_plan(elems: &[Elem], frame: String) -> Result<Plan, String> {
    let mut setups = Vec::new();
    for item in seq(elems, tag(0x300A, 0x0180)) {
        setups.push((
            int_tag(item, tag(0x300A, 0x0182)).unwrap_or(1),
            text_tag(item, tag(0x0018, 0x5100)),
        ));
    }
    let mut fractions = 1;
    let mut metersets = Vec::new();
    if let Some(group) = seq(elems, tag(0x300A, 0x0070)).first() {
        fractions = int_tag(group, tag(0x300A, 0x0078)).unwrap_or(1);
        for beam in seq(group, tag(0x300C, 0x0004)) {
            metersets.push((
                int_tag(beam, tag(0x300A, 0x00C0)).unwrap_or(0),
                float_tag(beam, tag(0x300A, 0x0086)).unwrap_or(0.0),
            ));
        }
    }
    let mut beams = Vec::new();
    for beam in seq(elems, tag(0x300A, 0x03A2)) {
        let number = int_tag(beam, tag(0x300A, 0x00C0)).unwrap_or(0);
        let meterset = metersets
            .iter()
            .find(|(id, _)| *id == number)
            .map(|(_, mu)| *mu)
            .unwrap_or(0.0);
        let final_weight = float_tag(beam, tag(0x300A, 0x010E)).unwrap_or(0.0);
        if meterset <= 0.0 || final_weight <= 0.0 {
            return Err(format!("beam {number} has no meterset"));
        }
        let setup = int_tag(beam, tag(0x300C, 0x006A)).unwrap_or(1);
        let position = setups
            .iter()
            .find(|(id, _)| *id == setup)
            .map(|(_, name)| name.clone())
            .filter(|name| !name.is_empty())
            .unwrap_or_else(|| "HFS".into());
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
        let mut couch = 0.0;
        let mut iso = [0.0f32; 3];
        let mut in_shifters: Vec<(i32, f32, f32)> = Vec::new();
        let mut spots = Vec::new();
        for cp in seq(beam, tag(0x300A, 0x03A8)) {
            energy = float_tag(cp, tag(0x300A, 0x0114)).unwrap_or(energy);
            gantry = float_tag(cp, tag(0x300A, 0x011E)).unwrap_or(gantry);
            couch = float_tag(cp, tag(0x300A, 0x0122)).unwrap_or(couch);
            let iso_raw = floats_tag(cp, tag(0x300A, 0x012C));
            if iso_raw.len() >= 3 {
                iso = [iso_raw[0], iso_raw[1], iso_raw[2]];
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
                    let wet = float_tag(setting, tag(0x300A, 0x0366))
                        .or(previous.map(|(_, wet, _)| wet))
                        .unwrap_or(f32::NAN);
                    let distance = float_tag(setting, tag(0x300A, 0x0364))
                        .or(previous.map(|(_, _, distance)| distance))
                        .unwrap_or(f32::NAN);
                    in_shifters.retain(|(id, _, _)| *id != shifter_number);
                    in_shifters.push((shifter_number, wet, distance));
                } else {
                    in_shifters.retain(|(id, _, _)| *id != shifter_number);
                }
            }
            let n = int_tag(cp, tag(0x300A, 0x0392)).unwrap_or(0) as usize;
            let xy = floats_tag(cp, tag(0x300A, 0x0394));
            let weights = floats_tag(cp, tag(0x300A, 0x0396));
            let spot_size_mm = scanning_spot_mm(cp);
            if n == 0 || !energy.is_finite() {
                continue;
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
            for i in 0..n {
                let mu = weights.get(i).copied().unwrap_or(0.0) * meterset / final_weight;
                if mu > 0.0 {
                    spots.push(Spot {
                        energy,
                        x: xy.get(i * 2).copied().unwrap_or(0.0),
                        y: xy.get(i * 2 + 1).copied().unwrap_or(0.0),
                        mu,
                        shifter: shifter.clone(),
                        wet_mm,
                        shifter_distance_mm,
                        spot_size_mm,
                    });
                }
            }
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
    Ok(Plan {
        beams,
        fractions,
        prescription: f32::NAN,
        frame,
        sop_uid: String::new(),
        label: text_tag(elems, tag(0x300A, 0x0002)),
    })
}

fn read_structures(elems: &[Elem]) -> Vec<Structure> {
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
            let data = floats_tag(item, tag(0x3006, 0x0050));
            for point in data.chunks(3) {
                if point.len() == 3 {
                    contour.push([point[0], point[1], point[2]]);
                }
            }
        }
        if !contour.is_empty() {
            structures.push(Structure { name, contour });
        }
    }
    structures
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
    let z0 = structure.contour[0][2];
    if (center[2] - z0).abs() > ct.spacing[2] * 0.6 {
        return false;
    }
    let mut hits = 0;
    let n = structure.contour.len();
    for i in 0..n {
        let a = structure.contour[i];
        let b = structure.contour[(i + 1) % n];
        if (a[1] > center[1]) != (b[1] > center[1]) {
            let t = (center[1] - a[1]) / (b[1] - a[1]);
            if a[0] + t * (b[0] - a[0]) > center[0] {
                hits += 1;
            }
        }
    }
    hits % 2 == 1
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

fn parse_dataset(bytes: &[u8]) -> Result<Vec<Elem>, String> {
    let mut elems = Vec::new();
    let mut cursor = 0;
    while cursor + 8 <= bytes.len() {
        let group = u16::from_le_bytes([bytes[cursor], bytes[cursor + 1]]);
        let element = u16::from_le_bytes([bytes[cursor + 2], bytes[cursor + 3]]);
        let vr = &bytes[cursor + 4..cursor + 6];
        let long = matches!(
            vr,
            b"OB" | b"OW" | b"OF" | b"SQ" | b"UT" | b"UN" | b"OD" | b"OL" | b"OV"
        );
        let (length, header) = if long {
            if cursor + 12 > bytes.len() {
                break;
            }
            let length = u32::from_le_bytes([
                bytes[cursor + 8],
                bytes[cursor + 9],
                bytes[cursor + 10],
                bytes[cursor + 11],
            ]);
            (length, 12)
        } else {
            let length = u16::from_le_bytes([bytes[cursor + 6], bytes[cursor + 7]]) as u32;
            (length, 8)
        };
        if length == u32::MAX {
            return Err("undefined-length DICOM sequences are not read".into());
        }
        let start = cursor + header;
        let end = start + length as usize;
        if end > bytes.len() {
            break;
        }
        let value = &bytes[start..end];
        let items = if vr == b"SQ" {
            parse_items(value)?
        } else {
            Vec::new()
        };
        elems.push(Elem {
            tag: tag(group, element),
            items,
            bytes: if vr == b"SQ" {
                Vec::new()
            } else {
                value.to_vec()
            },
        });
        cursor = end;
    }
    Ok(elems)
}

fn parse_items(bytes: &[u8]) -> Result<Vec<Vec<Elem>>, String> {
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
            break;
        }
        if group != 0xFFFE || element != 0xE000 {
            return Err("a sequence item is missing its header".into());
        }
        if length == u32::MAX {
            return Err("undefined-length DICOM sequences are not read".into());
        }
        let start = cursor + 8;
        let end = start + length as usize;
        if end > bytes.len() {
            return Err("a sequence item is short".into());
        }
        items.push(parse_dataset(&bytes[start..end])?);
        cursor = end;
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
    text_tag(elems, tag)
        .split('\\')
        .filter_map(|part| part.trim().parse().ok())
        .collect()
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

fn float_tag(elems: &[Elem], tag: u32) -> Option<f32> {
    floats_tag(elems, tag).first().copied()
}

fn int_tag(elems: &[Elem], tag: u32) -> Option<i32> {
    let elem = elem(elems, tag)?;
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

fn push_elem(bytes: &mut Vec<u8>, group: u16, slot: u16, vr: &[u8], value: &[u8]) {
    bytes.extend_from_slice(&group.to_le_bytes());
    bytes.extend_from_slice(&slot.to_le_bytes());
    bytes.extend_from_slice(vr);
    let long = matches!(vr, b"OW" | b"OB" | b"SQ" | b"OD");
    if long {
        bytes.extend_from_slice(&[0, 0]);
        bytes.extend_from_slice(&(value.len() as u32).to_le_bytes());
    } else {
        bytes.extend_from_slice(&(value.len() as u16).to_le_bytes());
    }
    bytes.extend_from_slice(value);
}

pub(crate) fn ui(bytes: &mut Vec<u8>, group: u16, element: u16, text: &str) {
    let mut value = text.as_bytes().to_vec();
    if value.len() % 2 == 1 {
        value.push(0);
    }
    push_elem(bytes, group, element, b"UI", &value);
}

pub(crate) fn cs(bytes: &mut Vec<u8>, group: u16, element: u16, text: &str) {
    push_elem(bytes, group, element, b"CS", &even(text));
}

pub(crate) fn lo(bytes: &mut Vec<u8>, group: u16, element: u16, text: &str) {
    push_elem(bytes, group, element, b"LO", &even(text));
}

pub(crate) fn pn(bytes: &mut Vec<u8>, group: u16, element: u16, text: &str) {
    push_elem(bytes, group, element, b"PN", &even(text));
}

pub(crate) fn sh(bytes: &mut Vec<u8>, group: u16, element: u16, text: &str) {
    push_elem(bytes, group, element, b"SH", &even(text));
}

pub(crate) fn ds(bytes: &mut Vec<u8>, group: u16, element: u16, values: &[f32]) {
    let text = values
        .iter()
        .map(|value| format!("{value:.6}"))
        .collect::<Vec<_>>()
        .join("\\");
    push_elem(bytes, group, element, b"DS", &even(&text));
}

pub(crate) fn is(bytes: &mut Vec<u8>, group: u16, element: u16, value: i32) {
    push_elem(bytes, group, element, b"IS", &even(&value.to_string()));
}

pub(crate) fn us(bytes: &mut Vec<u8>, group: u16, element: u16, value: u16) {
    push_elem(bytes, group, element, b"US", &value.to_le_bytes());
}

pub(crate) fn us_list(bytes: &mut Vec<u8>, group: u16, element: u16, values: &[u16]) {
    let mut raw = Vec::with_capacity(values.len() * 2);
    for value in values {
        raw.extend_from_slice(&value.to_le_bytes());
    }
    push_elem(bytes, group, element, b"US", &raw);
}

pub(crate) fn ow(bytes: &mut Vec<u8>, group: u16, element: u16, values: &[u16]) {
    let raw: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    push_elem(bytes, group, element, b"OW", &raw);
}

pub(crate) fn sq(bytes: &mut Vec<u8>, group: u16, element: u16, items: &[Vec<u8>]) {
    let mut body = Vec::new();
    for item in items {
        body.extend_from_slice(&0xFFFEu16.to_le_bytes());
        body.extend_from_slice(&0xE000u16.to_le_bytes());
        body.extend_from_slice(&(item.len() as u32).to_le_bytes());
        body.extend_from_slice(item);
    }
    push_elem(bytes, group, element, b"SQ", &body);
}

fn ul(bytes: &mut Vec<u8>, group: u16, element: u16, value: u32) {
    push_elem(bytes, group, element, b"UL", &value.to_le_bytes());
}

fn ob(bytes: &mut Vec<u8>, group: u16, element: u16, data: &[u8]) {
    push_elem(bytes, group, element, b"OB", data);
}

fn lt(bytes: &mut Vec<u8>, group: u16, element: u16, text: &str) {
    push_elem(bytes, group, element, b"LT", &even(text));
}

fn at(bytes: &mut Vec<u8>, group: u16, element: u16, at_group: u16, at_element: u16) {
    let mut value = at_group.to_le_bytes().to_vec();
    value.extend_from_slice(&at_element.to_le_bytes());
    push_elem(bytes, group, element, b"AT", &value);
}

fn ow32(bytes: &mut Vec<u8>, group: u16, element: u16, values: &[u32]) {
    let raw: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    push_elem(bytes, group, element, b"OW", &raw);
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
    if dose.len() != nx * ny * nz {
        return Err(format!(
            "dose is {} voxels, the grid wants {}",
            dose.len(),
            nx * ny * nz
        ));
    }
    if ct.frame.is_empty() {
        return Err("the dose grid and CT are in different frames of reference".into());
    }
    let peak = dose.iter().fold(0.0f64, |peak, &v| {
        peak.max(if v.is_nan() { 0.0 } else { v as f64 })
    });
    let scale = if peak > 0.0 { peak / 4.0e9 } else { 1.0 };
    let pixels: Vec<u32> = dose
        .iter()
        .map(|&v| {
            let value = if v.is_nan() { 0.0 } else { v as f64 };
            (value.max(0.0) / scale).round_ties_even() as u32
        })
        .collect();
    let sop_uid = uid();
    let (date, time) = stamp();

    let mut bytes = preamble();
    let mut meta = Vec::new();
    ob(&mut meta, 0x0002, 0x0001, &[0, 1]);
    ui(&mut meta, 0x0002, 0x0002, RT_DOSE_CLASS);
    ui(&mut meta, 0x0002, 0x0003, &sop_uid);
    ui(&mut meta, 0x0002, 0x0010, EXPLICIT_LE);
    ui(&mut meta, 0x0002, 0x0012, IMPL_UID);
    sh(&mut meta, 0x0002, 0x0013, "SCAN-KIT");
    ul(&mut bytes, 0x0002, 0x0000, meta.len() as u32);
    bytes.extend_from_slice(&meta);

    push_elem(&mut bytes, 0x0008, 0x0012, b"DA", &even(&date));
    push_elem(&mut bytes, 0x0008, 0x0013, b"TM", &even(&time));
    ui(&mut bytes, 0x0008, 0x0016, RT_DOSE_CLASS);
    ui(&mut bytes, 0x0008, 0x0018, &sop_uid);
    push_elem(&mut bytes, 0x0008, 0x0023, b"DA", &even(&date));
    push_elem(&mut bytes, 0x0008, 0x0033, b"TM", &even(&time));
    cs(&mut bytes, 0x0008, 0x0060, "RTDOSE");
    lo(&mut bytes, 0x0008, 0x0070, "Pyramid Technical Consultants");
    let series_description = format!("{description} (research only)");
    lo(
        &mut bytes,
        0x0008,
        0x103E,
        &series_description[..series_description.len().min(64)],
    );
    pn(&mut bytes, 0x0010, 0x0010, &ct.patient_name);
    lo(&mut bytes, 0x0010, 0x0020, &ct.patient_id);
    ds(&mut bytes, 0x0018, 0x0050, &[ct.spacing[2]]);
    lo(
        &mut bytes,
        0x0018,
        0x1020,
        &format!("scan-kit {}", env!("CARGO_PKG_VERSION")),
    );
    ui(&mut bytes, 0x0020, 0x000D, &ct.study_uid);
    ui(&mut bytes, 0x0020, 0x000E, &uid());
    ds(&mut bytes, 0x0020, 0x0032, &ct.origin);
    ds(&mut bytes, 0x0020, 0x0037, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    ui(&mut bytes, 0x0020, 0x0052, &ct.frame);
    lt(&mut bytes, 0x0020, 0x4000, comment);
    us(&mut bytes, 0x0028, 0x0002, 1);
    cs(&mut bytes, 0x0028, 0x0004, "MONOCHROME2");
    is(&mut bytes, 0x0028, 0x0008, nz as i32);
    at(&mut bytes, 0x0028, 0x0009, 0x3004, 0x000C);
    us(&mut bytes, 0x0028, 0x0010, ny as u16);
    us(&mut bytes, 0x0028, 0x0011, nx as u16);
    ds(&mut bytes, 0x0028, 0x0030, &[ct.spacing[1], ct.spacing[0]]);
    us(&mut bytes, 0x0028, 0x0100, 32);
    us(&mut bytes, 0x0028, 0x0101, 32);
    us(&mut bytes, 0x0028, 0x0102, 31);
    us(&mut bytes, 0x0028, 0x0103, 0);
    cs(&mut bytes, 0x3004, 0x0002, "GY");
    cs(&mut bytes, 0x3004, 0x0004, dose_type);
    lt(&mut bytes, 0x3004, 0x0006, DOSE_DISCLAIMER);
    cs(&mut bytes, 0x3004, 0x000A, summation);
    let offsets: Vec<f32> = (0..nz).map(|k| k as f32 * ct.spacing[2]).collect();
    ds(&mut bytes, 0x3004, 0x000C, &offsets);
    push_elem(
        &mut bytes,
        0x3004,
        0x000E,
        b"DS",
        &even(&format!("{scale:.10e}")),
    );
    if !plan_uid.is_empty() {
        let mut item = Vec::new();
        ui(&mut item, 0x0008, 0x1150, RT_ION_PLAN_CLASS);
        ui(&mut item, 0x0008, 0x1155, plan_uid);
        sq(&mut bytes, 0x300C, 0x0060, &[item]);
    }
    ow32(&mut bytes, 0x7FE0, 0x0010, &pixels);
    fs::write(path, &bytes).map_err(|err| err.to_string())?;
    Ok(path.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(hu_label(0.0), hu_label(0.0));
        assert!(hu_label(-2000.0) != hu_label(0.0));
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
        assert!(inside_structure(&study, &study.structures[0], 1, 1, 0));
        let mut extra = preamble();
        ui(&mut extra, 0x0008, 0x0016, "1.2.840.10008.5.1.4.1.1.2");
        ui(&mut extra, 0x0020, 0x0052, "9.9.9");
        us(&mut extra, 0x0028, 0x0010, 1);
        us(&mut extra, 0x0028, 0x0011, 1);
        ds(&mut extra, 0x0028, 0x0030, &[1.0, 1.0]);
        ds(&mut extra, 0x0020, 0x0032, &[0.0, 0.0, 5.0]);
        ow(&mut extra, 0x7FE0, 0x0010, &[0]);
        std::fs::write(root.join("other.dcm"), extra).unwrap();
        assert!(load_study(&root).is_err());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn an_ion_plan_keeps_raw_weights_and_the_scanning_spot_size() {
        let mut cp = Vec::new();
        ds(&mut cp, 0x300A, 0x0114, &[100.0]);
        is(&mut cp, 0x300A, 0x0392, 2);
        ds(&mut cp, 0x300A, 0x0394, &[1.0, 2.0, 3.0, 4.0]);
        ds(&mut cp, 0x300A, 0x0396, &[0.5, 0.25]);
        ds(&mut cp, 0x300A, 0x0398, &[4.0, 6.0]);
        let mut beam = Vec::new();
        sq(&mut beam, 0x300A, 0x03A8, &[cp]);
        let mut bytes = preamble();
        ui(&mut bytes, 0x0008, 0x0016, "1.2.840.10008.5.1.4.1.1.481.8");
        lo(&mut bytes, 0x300A, 0x0002, "TESTPLAN");
        sq(&mut bytes, 0x300A, 0x03A2, &[beam]);
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
        let plan = read_plan(&plan_with_shifter(&[(1, "IN", Some(40.0))]), "f".into()).unwrap();
        let spot = &plan.beams[0].spots[0];
        assert_eq!(spot.shifter, "RS1");
        assert!((spot.wet_mm - 40.0).abs() < 1e-3);
        assert!((spot.shifter_distance_mm - 300.0).abs() < 1e-3);
        let bare = read_plan(&plan_with_shifter(&[(1, "IN", None)]), "f".into()).unwrap();
        assert!(bare.beams[0].spots[0].wet_mm.is_nan());
        assert!(read_plan(
            &plan_with_shifter(&[(1, "IN", Some(40.0)), (2, "IN", Some(20.0))]),
            "f".into()
        )
        .is_err());
        assert!(read_plan(&plan_with_shifter(&[(3, "IN", Some(40.0))]), "f".into()).is_err());
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
        let plan_item = &seq(&elems, tag(0x300C, 0x0060))[0];
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
            study_uid: String::new(),
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
        let _ = fs::remove_file(&out);
    }

    fn plan_with_shifter(settings: &[(i32, &str, Option<f32>)]) -> Vec<Elem> {
        let mut ref_beam = Vec::new();
        is(&mut ref_beam, 0x300A, 0x00C0, 1);
        ds(&mut ref_beam, 0x300A, 0x0086, &[10.0]);
        let mut group = Vec::new();
        is(&mut group, 0x300A, 0x0078, 1);
        sq(&mut group, 0x300C, 0x0004, &[ref_beam]);
        let mut shifters = Vec::new();
        for number in [1, 2] {
            let mut item = Vec::new();
            is(&mut item, 0x300A, 0x0316, number);
            lo(&mut item, 0x300A, 0x0318, &format!("RS{number}"));
            shifters.push(item);
        }
        let mut setting_items = Vec::new();
        for (number, mode, wet) in settings {
            let mut item = Vec::new();
            is(&mut item, 0x300C, 0x0100, *number);
            cs(&mut item, 0x300A, 0x0362, mode);
            ds(&mut item, 0x300A, 0x0364, &[300.0]);
            if let Some(wet) = wet {
                ds(&mut item, 0x300A, 0x0366, &[*wet]);
            }
            setting_items.push(item);
        }
        let mut cp = Vec::new();
        ds(&mut cp, 0x300A, 0x0114, &[100.0]);
        ds(&mut cp, 0x300A, 0x012C, &[0.0, 0.0, 0.0]);
        sq(&mut cp, 0x300A, 0x0360, &setting_items);
        is(&mut cp, 0x300A, 0x0392, 1);
        ds(&mut cp, 0x300A, 0x0394, &[1.0, -2.0]);
        ds(&mut cp, 0x300A, 0x0396, &[1.0]);
        let mut beam = Vec::new();
        is(&mut beam, 0x300A, 0x00C0, 1);
        ds(&mut beam, 0x300A, 0x010E, &[1.0]);
        sq(&mut beam, 0x300A, 0x0314, &shifters);
        sq(&mut beam, 0x300A, 0x03A8, &[cp]);
        let mut plan = Vec::new();
        sq(&mut plan, 0x300A, 0x0070, &[group]);
        sq(&mut plan, 0x300A, 0x03A2, &[beam]);
        parse_dataset(&plan).unwrap()
    }
}
