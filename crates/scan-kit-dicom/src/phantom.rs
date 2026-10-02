//! Synthetic water-box study: CT, RTSTRUCT, and one RT Ion beam.
//!
//! The Monte Carlo reference RTDOSE stays on the Python writer. This file is
//! what Dose Volume loads.

use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::Value;

use crate::patient;

const BOX_MM: f64 = 120.0;
const MARGIN_MM: [f64; 3] = [40.0, 40.0, 20.0];
const CT: &str = "1.2.840.10008.5.1.4.1.1.2";
const RT_STRUCT: &str = "1.2.840.10008.5.1.4.1.1.481.3";
const RT_PLAN: &str = "1.2.840.10008.5.1.4.1.1.481.8";

#[derive(Clone, Copy)]
struct Box {
    x0: f64,
    x1: f64,
    y0: f64,
    y1: f64,
    z0: f64,
    z1: f64,
    hu: f32,
    name: &'static str,
    color_r: u8,
    color_g: u8,
    color_b: u8,
}

const BONE: Box = Box {
    x0: -60.0,
    x1: 60.0,
    y0: -20.0,
    y1: -10.0,
    z0: -30.0,
    z1: 30.0,
    hu: 700.0,
    name: "BONE",
    color_r: 230,
    color_g: 220,
    color_b: 170,
};
const LUNG: Box = Box {
    x0: -60.0,
    x1: 60.0,
    y0: -5.0,
    y1: 10.0,
    z0: -30.0,
    z1: 30.0,
    hu: -750.0,
    name: "LUNG",
    color_r: 120,
    color_g: 180,
    color_b: 255,
};
const PTV: Box = Box {
    x0: -15.0,
    x1: 15.0,
    y0: 20.0,
    y1: 50.0,
    z0: -15.0,
    z1: 15.0,
    hu: 0.0,
    name: "PTV",
    color_r: 255,
    color_g: 0,
    color_b: 0,
};

struct Request {
    phantom: String,
    position: String,
    pixel: f64,
    slice: f64,
    energies: Vec<f64>,
    gantry: f64,
    couch: f64,
    wet: f64,
    mu: f64,
    fractions: i32,
    nx: usize,
    ny: usize,
    nz: usize,
    spots: Vec<(f32, f32)>,
}

/// One-line description of the study the form would write.
pub fn phantom_summary(params: &Value) -> Result<String, String> {
    let request = parse(params)?;
    Ok(summary(&request))
}

/// Write CT slices, `RS_synthetic.dcm`, and `RP_synthetic.dcm` into `dir`.
pub fn write_phantom_study(dir: &Path, params: &Value) -> Result<String, String> {
    let request = parse(params)?;
    fs::create_dir_all(dir).map_err(|err| err.to_string())?;
    let study = uid();
    let frame = uid();
    let ct_series = uid();
    let flip = if request.position.to_ascii_uppercase().ends_with('P') {
        -1.0
    } else {
        1.0
    };
    let dx = request.pixel;
    let dy = request.pixel;
    let dz = request.slice;
    let x0 = -flip * (request.nx as f64 - 1.0) * dx / 2.0;
    let y0 = -flip * (request.ny as f64 - 1.0) * dy / 2.0;
    let inserts = inserts(&request.phantom);
    for k in 0..request.nz {
        let z = (k as f64 - (request.nz as f64 - 1.0) / 2.0) * dz;
        let mut pixels = Vec::with_capacity(request.nx * request.ny);
        for row in 0..request.ny {
            let y = y0 + flip * row as f64 * dy;
            for col in 0..request.nx {
                let x = x0 + flip * col as f64 * dx;
                pixels.push(stored_hu(hu_at(x, y, z, &inserts)));
            }
        }
        let sop = uid();
        let mut bytes = patient::preamble();
        patient::ui(&mut bytes, 0x0008, 0x0016, CT);
        patient::ui(&mut bytes, 0x0008, 0x0018, &sop);
        patient::cs(&mut bytes, 0x0008, 0x0060, "CT");
        patient::pn(&mut bytes, 0x0010, 0x0010, "Phantom^Synthetic");
        patient::lo(&mut bytes, 0x0010, 0x0020, "SYNTH-0001");
        patient::ds(&mut bytes, 0x0018, 0x0050, &[dz as f32]);
        patient::cs(&mut bytes, 0x0018, 0x5100, &request.position);
        patient::ui(&mut bytes, 0x0020, 0x000D, &study);
        patient::ui(&mut bytes, 0x0020, 0x000E, &ct_series);
        patient::is(&mut bytes, 0x0020, 0x0013, k as i32 + 1);
        patient::ds(
            &mut bytes,
            0x0020,
            0x0032,
            &[x0 as f32, y0 as f32, z as f32],
        );
        patient::ds(
            &mut bytes,
            0x0020,
            0x0037,
            &[flip as f32, 0.0, 0.0, 0.0, flip as f32, 0.0],
        );
        patient::ui(&mut bytes, 0x0020, 0x0052, &frame);
        patient::us(&mut bytes, 0x0028, 0x0002, 1);
        patient::cs(&mut bytes, 0x0028, 0x0004, "MONOCHROME2");
        patient::us(&mut bytes, 0x0028, 0x0010, request.ny as u16);
        patient::us(&mut bytes, 0x0028, 0x0011, request.nx as u16);
        patient::ds(&mut bytes, 0x0028, 0x0030, &[dy as f32, dx as f32]);
        patient::us(&mut bytes, 0x0028, 0x0100, 16);
        patient::us(&mut bytes, 0x0028, 0x0101, 16);
        patient::us(&mut bytes, 0x0028, 0x0102, 15);
        patient::us(&mut bytes, 0x0028, 0x0103, 1);
        patient::ds(&mut bytes, 0x0028, 0x1052, &[-1024.0]);
        patient::ds(&mut bytes, 0x0028, 0x1053, &[1.0]);
        patient::ow(&mut bytes, 0x7FE0, 0x0010, &pixels);
        let name = format!("CT_{}.dcm", &sop[sop.len().saturating_sub(12)..]);
        fs::write(dir.join(name), bytes).map_err(|err| err.to_string())?;
    }
    write_struct(dir, &request, &study, &frame, &ct_series)?;
    write_plan(dir, &request, &study, &frame)?;
    Ok(format!(
        "{} CT slices, RTSTRUCT and RT Ion Plan",
        request.nz
    ))
}

fn write_struct(
    dir: &Path,
    request: &Request,
    study: &str,
    frame: &str,
    ct_series: &str,
) -> Result<(), String> {
    let mut rois = Vec::new();
    let mut contours = Vec::new();
    let mut number = 1i32;
    let half = BOX_MM / 2.0;
    push_roi(
        &mut rois,
        &mut contours,
        number,
        frame,
        "BODY",
        "EXTERNAL",
        [0, 255, 0],
        slices_where(
            request,
            |z| z.abs() < half,
            |z| vec![square(-half, half, -half, half, z)],
        ),
    );
    number += 1;
    push_roi(
        &mut rois,
        &mut contours,
        number,
        frame,
        "PTV",
        "PTV",
        [255, 0, 0],
        slices_where(
            request,
            |z| (PTV.z0..PTV.z1).contains(&z),
            |z| vec![square(PTV.x0, PTV.x1, PTV.y0, PTV.y1, z)],
        ),
    );
    number += 1;
    push_roi(
        &mut rois,
        &mut contours,
        number,
        frame,
        "RING",
        "ORGAN",
        [0, 0, 255],
        slices_where(
            request,
            |z| (-10.0..10.0).contains(&z),
            |z| {
                vec![
                    square(-50.0, -10.0, 20.0, 60.0, z),
                    square(-40.0, -20.0, 30.0, 50.0, z),
                ]
            },
        ),
    );
    for insert in inserts(&request.phantom) {
        number += 1;
        let insert = *insert;
        push_roi(
            &mut rois,
            &mut contours,
            number,
            frame,
            insert.name,
            "ORGAN",
            [insert.color_r, insert.color_g, insert.color_b],
            slices_where(
                request,
                move |z| (insert.z0..insert.z1).contains(&z),
                move |z| vec![square(insert.x0, insert.x1, insert.y0, insert.y1, z)],
            ),
        );
    }
    let mut bytes = patient::preamble();
    patient::ui(&mut bytes, 0x0008, 0x0016, RT_STRUCT);
    patient::ui(&mut bytes, 0x0008, 0x0018, &uid());
    patient::cs(&mut bytes, 0x0008, 0x0060, "RTSTRUCT");
    patient::pn(&mut bytes, 0x0010, 0x0010, "Phantom^Synthetic");
    patient::lo(&mut bytes, 0x0010, 0x0020, "SYNTH-0001");
    patient::ui(&mut bytes, 0x0020, 0x000D, study);
    patient::ui(&mut bytes, 0x0020, 0x000E, &uid());
    patient::ui(&mut bytes, 0x0020, 0x0052, frame);
    patient::lo(&mut bytes, 0x3006, 0x0002, "Synthetic");
    let referenced = referenced_frame(frame, study, ct_series);
    patient::sq(&mut bytes, 0x3006, 0x0010, &[referenced]);
    patient::sq(&mut bytes, 0x3006, 0x0020, &rois);
    patient::sq(&mut bytes, 0x3006, 0x0039, &contours);
    fs::write(dir.join("RS_synthetic.dcm"), bytes).map_err(|err| err.to_string())
}

fn referenced_frame(frame: &str, study: &str, ct_series: &str) -> Vec<u8> {
    let mut series = Vec::new();
    patient::ui(&mut series, 0x0020, 0x000E, ct_series);
    let mut study_item = Vec::new();
    patient::ui(&mut study_item, 0x0008, 0x1150, "1.2.840.10008.3.1.2.3.1");
    patient::ui(&mut study_item, 0x0008, 0x1155, study);
    patient::sq(&mut study_item, 0x3006, 0x0014, &[series]);
    let mut item = Vec::new();
    patient::ui(&mut item, 0x0020, 0x0052, frame);
    patient::sq(&mut item, 0x3006, 0x0012, &[study_item]);
    item
}

fn push_roi(
    rois: &mut Vec<Vec<u8>>,
    contours: &mut Vec<Vec<u8>>,
    number: i32,
    frame: &str,
    name: &str,
    kind: &str,
    color: [u8; 3],
    planes: Vec<Vec<Vec<f32>>>,
) {
    if planes.is_empty() {
        return;
    }
    let mut roi = Vec::new();
    patient::is(&mut roi, 0x3006, 0x0022, number);
    patient::ui(&mut roi, 0x3006, 0x0024, frame);
    patient::lo(&mut roi, 0x3006, 0x0026, name);
    patient::cs(&mut roi, 0x3006, 0x00A4, kind);
    rois.push(roi);
    let mut drawn = Vec::new();
    for plane in planes {
        for points in plane {
            let mut item = Vec::new();
            patient::cs(&mut item, 0x3006, 0x0042, "CLOSED_PLANAR");
            patient::is(&mut item, 0x3006, 0x0046, (points.len() / 3) as i32);
            patient::ds(&mut item, 0x3006, 0x0050, &points);
            drawn.push(item);
        }
    }
    let mut contour = Vec::new();
    patient::us_list(&mut contour, 0x3006, 0x002A, &color.map(u16::from));
    patient::is(&mut contour, 0x3006, 0x0084, number);
    patient::sq(&mut contour, 0x3006, 0x0040, &drawn);
    contours.push(contour);
}

fn write_plan(dir: &Path, request: &Request, study: &str, frame: &str) -> Result<(), String> {
    let mut bytes = patient::preamble();
    patient::ui(&mut bytes, 0x0008, 0x0016, RT_PLAN);
    patient::ui(&mut bytes, 0x0008, 0x0018, &uid());
    patient::cs(&mut bytes, 0x0008, 0x0060, "RTPLAN");
    patient::pn(&mut bytes, 0x0010, 0x0010, "Phantom^Synthetic");
    patient::lo(&mut bytes, 0x0010, 0x0020, "SYNTH-0001");
    patient::ui(&mut bytes, 0x0020, 0x000D, study);
    patient::ui(&mut bytes, 0x0020, 0x000E, &uid());
    patient::ui(&mut bytes, 0x0020, 0x0052, frame);
    patient::lo(&mut bytes, 0x300A, 0x0002, "SYNTH");
    patient::cs(&mut bytes, 0x300A, 0x000C, "PATIENT");
    let dose_ref = {
        let mut item = Vec::new();
        patient::is(&mut item, 0x300A, 0x0012, 1);
        patient::cs(&mut item, 0x300A, 0x0014, "SITE");
        patient::cs(&mut item, 0x300A, 0x0020, "TARGET");
        patient::ds(&mut item, 0x300A, 0x0026, &[2.0]);
        item
    };
    patient::sq(&mut bytes, 0x300A, 0x0010, &[dose_ref]);
    let setup = {
        let mut item = Vec::new();
        patient::is(&mut item, 0x300A, 0x0182, 1);
        patient::cs(&mut item, 0x0018, 0x5100, &request.position);
        item
    };
    patient::sq(&mut bytes, 0x300A, 0x0180, &[setup]);
    let n = request.spots.len();
    let cum = (n * request.energies.len()) as f32;
    let referenced = {
        let mut item = Vec::new();
        patient::is(&mut item, 0x300A, 0x00C0, 1);
        patient::ds(&mut item, 0x300A, 0x0086, &[cum * request.mu as f32]);
        item
    };
    let fraction = {
        let mut item = Vec::new();
        patient::is(&mut item, 0x300A, 0x0071, 1);
        patient::is(&mut item, 0x300A, 0x0078, request.fractions);
        patient::is(&mut item, 0x300A, 0x0080, 1);
        patient::sq(&mut item, 0x300C, 0x0004, &[referenced]);
        item
    };
    patient::sq(&mut bytes, 0x300A, 0x0070, &[fraction]);
    let mut cps = Vec::new();
    let mut weight = 0.0f32;
    let mut first = true;
    for energy in &request.energies {
        for last in [false, true] {
            let mut cp = Vec::new();
            patient::ds(&mut cp, 0x300A, 0x0114, &[*energy as f32]);
            if first {
                patient::ds(&mut cp, 0x300A, 0x011E, &[request.gantry as f32]);
                patient::ds(&mut cp, 0x300A, 0x0122, &[request.couch as f32]);
                patient::ds(
                    &mut cp,
                    0x300A,
                    0x012C,
                    &[0.0, ((PTV.y0 + PTV.y1) / 2.0) as f32, 0.0],
                );
                patient::ds(&mut cp, 0x300A, 0x030D, &[300.0]);
                if request.wet > 0.0 {
                    let mut setting = Vec::new();
                    patient::is(&mut setting, 0x300C, 0x0100, 1);
                    patient::cs(&mut setting, 0x300A, 0x0362, "IN");
                    patient::ds(&mut setting, 0x300A, 0x0364, &[300.0]);
                    patient::ds(&mut setting, 0x300A, 0x0366, &[request.wet as f32]);
                    patient::sq(&mut cp, 0x300A, 0x0360, &[setting]);
                }
            }
            patient::ds(&mut cp, 0x300A, 0x0134, &[weight]);
            patient::sh(&mut cp, 0x300A, 0x0390, "3.0");
            patient::is(&mut cp, 0x300A, 0x0392, n as i32);
            let mut xy = Vec::with_capacity(n * 2);
            for (x, y) in &request.spots {
                xy.push(*x);
                xy.push(*y);
            }
            patient::ds(&mut cp, 0x300A, 0x0394, &xy);
            let spot_weight = if last { 0.0 } else { 1.0 };
            patient::ds(&mut cp, 0x300A, 0x0396, &vec![spot_weight; n]);
            if !last {
                weight += n as f32;
            }
            first = false;
            cps.push(cp);
        }
    }
    let beam = {
        let mut item = Vec::new();
        patient::is(&mut item, 0x300A, 0x00C0, 1);
        patient::lo(&mut item, 0x300A, 0x00C2, "B1");
        patient::cs(&mut item, 0x300A, 0x00C6, "PROTON");
        patient::cs(&mut item, 0x300A, 0x00CE, "TREATMENT");
        patient::lo(&mut item, 0x300A, 0x00B2, "SYNTH");
        patient::is(&mut item, 0x300A, 0x0110, cps.len() as i32);
        patient::ds(&mut item, 0x300A, 0x010E, &[weight]);
        patient::cs(&mut item, 0x300A, 0x0308, "MODULATED");
        patient::ds(&mut item, 0x300A, 0x030A, &[2000.0, 2000.0]);
        if request.wet > 0.0 {
            patient::is(&mut item, 0x300A, 0x0312, 1);
            let mut shifter = Vec::new();
            patient::is(&mut shifter, 0x300A, 0x0316, 1);
            patient::sh(&mut shifter, 0x300A, 0x0318, "RS1");
            patient::cs(&mut shifter, 0x300A, 0x0320, "BINARY");
            patient::sq(&mut item, 0x300A, 0x0314, &[shifter]);
        }
        patient::is(&mut item, 0x300C, 0x006A, 1);
        patient::sq(&mut item, 0x300A, 0x03A8, &cps);
        item
    };
    patient::sq(&mut bytes, 0x300A, 0x03A2, &[beam]);
    fs::write(dir.join("RP_synthetic.dcm"), bytes).map_err(|err| err.to_string())
}

fn slices_where(
    request: &Request,
    mut keep: impl FnMut(f64) -> bool,
    mut shape: impl FnMut(f64) -> Vec<Vec<f32>>,
) -> Vec<Vec<Vec<f32>>> {
    let mut planes = Vec::new();
    for k in 0..request.nz {
        let z = (k as f64 - (request.nz as f64 - 1.0) / 2.0) * request.slice;
        if keep(z) {
            planes.push(shape(z));
        }
    }
    planes
}

fn square(x0: f64, x1: f64, y0: f64, y1: f64, z: f64) -> Vec<f32> {
    let (x0, x1, y0, y1, z) = (x0 as f32, x1 as f32, y0 as f32, y1 as f32, z as f32);
    vec![x0, y0, z, x1, y0, z, x1, y1, z, x0, y1, z]
}

fn parse(params: &Value) -> Result<Request, String> {
    let phantom = text(params, "phantom", "slabs");
    if !matches!(phantom.as_str(), "slabs" | "bone" | "lung" | "water") {
        return Err(format!(
            "unknown phantom '{phantom}'; choose from slabs, bone, lung, water"
        ));
    }
    let position = text(params, "position", "HFS");
    if !matches!(position.as_str(), "HFS" | "HFP" | "FFS" | "FFP") {
        return Err(format!("unknown position '{position}'"));
    }
    let pixel = number(params, "pixel", 2.0)?;
    let slice = number(params, "slice", 2.0)?;
    if pixel <= 0.0 || slice <= 0.0 {
        return Err("Pixel spacing and slice thickness must be positive.".into());
    }
    let mut energies = match params.get("energies").and_then(Value::as_array) {
        Some(values) if values.is_empty() => {
            return Err("pick at least one energy".into());
        }
        Some(values) => {
            let mut energies = Vec::new();
            for value in values {
                let energy = value.as_f64().ok_or("energies must be numbers")?;
                if energy.is_finite() && energy > 0.0 {
                    energies.push(energy);
                }
            }
            if energies.is_empty() {
                return Err("pick at least one energy".into());
            }
            energies
        }
        None => vec![140.0, 130.0, 120.0],
    };
    energies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let gantry = number(params, "gantry", 0.0)?;
    let couch = number(params, "couch", 0.0)?;
    let pitch = number(params, "spot_pitch", 6.0)?;
    if !(1.0..=12.0).contains(&pitch) {
        return Err("Spot pitch (mm) must be from 1 to 12.".into());
    }
    let wet = number(params, "range_shifter_wet", 0.0)?;
    if !(0.0..=100.0).contains(&wet) {
        return Err("Range shifter WET (mm) must be from 0 to 100.".into());
    }
    let mu = number(params, "mu_per_spot", 0.02)?;
    if !(0.001..=10.0).contains(&mu) {
        return Err("MU per spot must be from 0.001 to 10.".into());
    }
    let fractions = params
        .get("fractions")
        .and_then(Value::as_i64)
        .or_else(|| {
            params
                .get("fractions")
                .and_then(Value::as_f64)
                .map(|v| v as i64)
        })
        .unwrap_or(1);
    if !(1..=40).contains(&fractions) {
        return Err("Fractions must be from 1 to 40.".into());
    }
    let nx = dim(BOX_MM + MARGIN_MM[0], pixel);
    let ny = dim(BOX_MM + MARGIN_MM[1], pixel);
    let nz = dim(BOX_MM + MARGIN_MM[2], slice);
    if nx == 0 || ny == 0 || nz == 0 || nx > 512 || ny > 512 || nz > 512 {
        return Err("CT is too large. Use a coarser pixel spacing or slice thickness.".into());
    }
    let axis = spot_axis(pitch);
    if axis.is_empty() || axis.len() * axis.len() * energies.len() > 1_000_000 {
        return Err("The plan has too many spots.".into());
    }
    let mut spots = Vec::with_capacity(axis.len() * axis.len());
    for y in &axis {
        for x in &axis {
            spots.push((*x, *y));
        }
    }
    Ok(Request {
        phantom,
        position,
        pixel,
        slice,
        energies,
        gantry,
        couch,
        wet,
        mu,
        fractions: fractions as i32,
        nx,
        ny,
        nz,
        spots,
    })
}

fn summary(request: &Request) -> String {
    let layers = request.energies.len();
    let spots = request.spots.len() * layers;
    let mu = spots as f64 * request.mu;
    format!(
        "CT {}×{}×{} voxels at {}×{}×{} mm; {} {}, {} {}, {} MU per fraction.",
        request.nx,
        request.ny,
        request.nz,
        compact(request.pixel),
        compact(request.pixel),
        compact(request.slice),
        layers,
        if layers == 1 { "layer" } else { "layers" },
        spots,
        if spots == 1 { "spot" } else { "spots" },
        compact_mu(mu)
    )
}

fn compact(value: f64) -> String {
    if value.fract() == 0.0 && value.abs() < 1e12 {
        format!("{}", value as i64)
    } else {
        let text = format!("{value:.4}");
        text.trim_end_matches('0').trim_end_matches('.').to_owned()
    }
}

fn compact_mu(value: f64) -> String {
    if value == 0.0 {
        return "0".into();
    }
    let digits = 3 - value.abs().log10().floor() as i32 - 1;
    let digits = digits.clamp(0, 8) as usize;
    let text = format!("{value:.digits$}");
    let trimmed = text.trim_end_matches('0').trim_end_matches('.');
    if trimmed.contains('.') || trimmed.contains('e') || trimmed.contains('E') {
        trimmed.to_owned()
    } else if value.abs() >= 1000.0 {
        format!("{value:.3e}")
    } else {
        trimmed.to_owned()
    }
}

fn dim(span: f64, spacing: f64) -> usize {
    (span / spacing).ceil() as usize
}

fn spot_axis(pitch: f64) -> Vec<f32> {
    let mut value = -12.0;
    let mut axis = Vec::new();
    while value < 12.01 && axis.len() < 10_000 {
        axis.push(value as f32);
        value += pitch;
    }
    axis
}

fn inserts(phantom: &str) -> Vec<&Box> {
    match phantom {
        "slabs" => vec![&BONE, &LUNG],
        "bone" => vec![&BONE],
        "lung" => vec![&LUNG],
        _ => Vec::new(),
    }
}

fn contains(x: f64, y: f64, z: f64, region: &Box) -> bool {
    (region.x0..region.x1).contains(&x)
        && (region.y0..region.y1).contains(&y)
        && (region.z0..region.z1).contains(&z)
}

fn hu_at(x: f64, y: f64, z: f64, inserts: &[&Box]) -> f32 {
    let half = BOX_MM / 2.0;
    let water = Box {
        x0: -half,
        x1: half,
        y0: -half,
        y1: half,
        z0: -half,
        z1: half,
        hu: 0.0,
        name: "WATER",
        color_r: 0,
        color_g: 0,
        color_b: 0,
    };
    let mut hu = if contains(x, y, z, &water) {
        0.0
    } else {
        -1000.0
    };
    for insert in inserts {
        if contains(x, y, z, insert) {
            hu = insert.hu;
        }
    }
    hu
}

fn stored_hu(hu: f32) -> u16 {
    (hu + 1024.0).round().clamp(0.0, 4095.0) as u16
}

fn text(params: &Value, key: &str, default: &str) -> String {
    params
        .get(key)
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())
        .unwrap_or(default)
        .to_owned()
}

fn number(params: &Value, key: &str, default: f64) -> Result<f64, String> {
    match params.get(key).and_then(Value::as_f64) {
        Some(value) if value.is_finite() => Ok(value),
        Some(_) => Err(format!("{key} must be a finite number")),
        None => Ok(default),
    }
}

fn uid() -> String {
    static TICK: AtomicU32 = AtomicU32::new(1);
    let tick = TICK.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    format!("2.25.{nanos}{tick:04}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::patient::load_study;
    use serde_json::json;

    #[test]
    fn the_default_summary_is_an_80_cube() {
        let text = phantom_summary(&json!({})).unwrap();
        assert_eq!(
            text,
            "CT 80×80×70 voxels at 2×2×2 mm; 3 layers, 75 spots, 1.5 MU per fraction."
        );
    }

    #[test]
    fn an_empty_energy_list_is_rejected() {
        let err = phantom_summary(&json!({"energies": []})).unwrap_err();
        assert_eq!(err, "pick at least one energy");
    }

    #[test]
    fn slabs_round_trip_through_the_study_reader() {
        let dir = std::env::temp_dir().join(format!("scan-kit-phantom-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let note = write_phantom_study(
            &dir,
            &json!({
                "phantom": "slabs",
                "position": "HFS",
                "pixel": 4.0,
                "slice": 4.0,
                "energies": [140.0, 130.0, 120.0],
                "gantry": 90.0,
                "couch": 0.0,
                "spot_pitch": 6.0,
                "range_shifter_wet": 5.0,
                "mu_per_spot": 0.02,
                "fractions": 3
            }),
        )
        .unwrap();
        assert!(note.contains("35 CT slices"));
        let study = load_study(&dir).unwrap();
        assert_eq!(study.fractions, 3);
        assert_eq!(study.ct.shape, [40, 40, 35]);
        let names: Vec<_> = study
            .structures
            .iter()
            .map(|item| item.name.as_str())
            .collect();
        assert_eq!(names, ["BODY", "PTV", "RING", "BONE", "LUNG"]);
        assert!(study.ct.hu.iter().any(|value| *value > 500.0));
        assert!(study.ct.hu.iter().any(|value| (*value + 750.0).abs() < 0.1));
        assert!(study.ct.hu.iter().any(|value| value.abs() < 0.1));
        assert_eq!(study.beams.len(), 1);
        assert_eq!(study.beams[0].spots.len(), 75);
        assert!((study.beams[0].spots[0].energy - 120.0).abs() < 0.01);
        assert!((study.beams[0].spots[0].mu - 0.02).abs() < 1e-4);
        assert!((study.beams[0].spots[0].wet_mm - 5.0).abs() < 0.01);
        assert!((study.beams[0].gantry - 90.0).abs() < 0.01);
        assert_eq!(study.beams[0].spots[0].shifter, "RS1");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn water_has_no_insert() {
        let dir =
            std::env::temp_dir().join(format!("scan-kit-phantom-water-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        write_phantom_study(
            &dir,
            &json!({
                "phantom": "water",
                "pixel": 8.0,
                "slice": 8.0,
                "energies": [140.0],
                "fractions": 1
            }),
        )
        .unwrap();
        let study = load_study(&dir).unwrap();
        let names: Vec<_> = study
            .structures
            .iter()
            .map(|item| item.name.as_str())
            .collect();
        assert_eq!(names, ["BODY", "PTV", "RING"]);
        assert!(study.ct.hu.iter().all(|value| *value <= 0.1));
        assert!(study.ct.hu.iter().all(|value| (*value + 750.0).abs() > 1.0));
        let _ = fs::remove_dir_all(&dir);
    }
}
