//! CT, RTSTRUCT, RT Ion Plan, and RTDOSE. One frame of reference, or the load fails.

use std::fs;
use std::path::Path;

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
}

struct Elem {
    tag: u32,
    items: Vec<Vec<Elem>>,
    bytes: Vec<u8>,
}

pub fn load_study(path: &Path) -> Result<PatientStudy, String> {
    let mut ct_slices = Vec::new();
    let mut dose_slices = Vec::new();
    let mut plan = None;
    let mut structures = Vec::new();
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
            for slice in image_slices(&elems, 1.0)? {
                ct_slices.push((frame.clone(), slice));
            }
        } else if class.contains("1.2.840.10008.5.1.4.1.1.481.8") {
            plan = Some(read_plan(&elems, frame)?);
        } else if class.contains("1.2.840.10008.5.1.4.1.1.481.3") {
            structures = read_structures(&elems);
        } else if class.contains("1.2.840.10008.5.1.4.1.1.481.2") {
            let scale = float_tag(&elems, tag(0x3004, 0x000E)).unwrap_or(1.0);
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
    let (beams, fractions, prescription, plan_frame) =
        plan.unwrap_or((Vec::new(), 1, f32::NAN, frame.clone()));
    if plan_frame != frame {
        return Err("the plan frame of reference does not match the CT".into());
    }
    if dose_slices.iter().any(|(got, _)| got != &frame) {
        return Err("the dose frame of reference does not match the CT".into());
    }
    let mut slices: Vec<_> = ct_slices.into_iter().map(|(_, slice)| slice).collect();
    let ct = stack(&mut slices)?;
    let tps = if dose_slices.is_empty() {
        None
    } else {
        let mut slices: Vec<_> = dose_slices.into_iter().map(|(_, slice)| slice).collect();
        Some(stack(&mut slices)?)
    };
    Ok(PatientStudy {
        frame,
        ct,
        beams,
        fractions,
        structures,
        tps,
        prescription,
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
    let stored: Vec<f32> = pixels
        .bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|chunk| (u16::from_le_bytes(*chunk) as f32 * slope + intercept) * scale)
        .collect();
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
    })
}

fn read_plan(elems: &[Elem], frame: String) -> Result<(Vec<Beam>, i32, f32, String), String> {
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
    Ok((beams, fractions, f32::NAN, frame))
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
        let (beams, _, _, _) =
            read_plan(&plan_with_shifter(&[(1, "IN", Some(40.0))]), "f".into()).unwrap();
        let spot = &beams[0].spots[0];
        assert_eq!(spot.shifter, "RS1");
        assert!((spot.wet_mm - 40.0).abs() < 1e-3);
        assert!((spot.shifter_distance_mm - 300.0).abs() < 1e-3);
        let bare = read_plan(&plan_with_shifter(&[(1, "IN", None)]), "f".into()).unwrap();
        assert!(bare.0[0].spots[0].wet_mm.is_nan());
        assert!(read_plan(
            &plan_with_shifter(&[(1, "IN", Some(40.0)), (2, "IN", Some(20.0))]),
            "f".into()
        )
        .is_err());
        assert!(read_plan(&plan_with_shifter(&[(3, "IN", Some(40.0))]), "f".into()).is_err());
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
