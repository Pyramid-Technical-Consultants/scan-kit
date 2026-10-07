//! UPenn double-Gaussian beam model. Records match the transport shader's stride of 40.

use scan_kit_dicom::gantry_to_patient;

use super::mc_tables::mc_tables;

const BDL_UN_RS: &str =
    include_str!("../../../scan_kit/assets/mcsquare/BDL/BDL_default_UN_RangeShifter.txt");
const BDL_UN: &str = include_str!("../../../scan_kit/assets/mcsquare/BDL/BDL_default_UN.txt");
const BDL_DN: &str = include_str!("../../../scan_kit/assets/mcsquare/BDL/BDL_default_DN.txt");
const BDL_DN_RS: &str =
    include_str!("../../../scan_kit/assets/mcsquare/BDL/BDL_default_DN_RangeShifter.txt");

fn bdl_text(name: &str) -> &'static str {
    match name {
        "un" => BDL_UN,
        "dn" => BDL_DN,
        "dn_rs" => BDL_DN_RS,
        _ => BDL_UN_RS,
    }
}

struct Model {
    nozzle: f32,
    smx: f32,
    smy: f32,
    rows: Vec<[f32; 18]>,
    shifter_material: i32,
    shifter_density: f32,
    shifter_wet: f32,
}

pub fn beam_record(
    gantry: f32,
    couch: f32,
    position: &str,
    isocenter_mm: [f32; 3],
) -> Result<[f32; 16], String> {
    beam_record_bdl(gantry, couch, position, isocenter_mm, "un_rs")
}

/// One kernel spot. `wet_mm` of 0 leaves the range shifter out; NaN uses the library thickness.
pub fn spot_record(
    energy: f32,
    x: f32,
    y: f32,
    beam: f32,
    wet_mm: f32,
    distance_mm: f32,
) -> Result<[f32; 40], String> {
    spot_record_bdl(energy, x, y, beam, wet_mm, distance_mm, "un_rs")
}

fn fill_spot(
    model: &Model,
    energy: f32,
    x: f32,
    y: f32,
    beam: f32,
    wet_mm: f32,
    distance_mm: f32,
) -> Result<[f32; 40], String> {
    let mut rec = [0.0f32; 40];
    rec[0] = x;
    rec[1] = y;
    rec[2] = column(&model.rows, 1, energy);
    rec[3] = column(&model.rows, 2, energy) * energy / 100.0;
    let w1 = column(&model.rows, 4, energy);
    let w2 = column(&model.rows, 11, energy);
    rec[4] = w1 / (w1 + w2);
    rec[6] = beam;
    let g1 = phase(
        column(&model.rows, 5, energy),
        column(&model.rows, 6, energy),
        column(&model.rows, 7, energy),
    );
    let g1y = phase(
        column(&model.rows, 8, energy),
        column(&model.rows, 9, energy),
        column(&model.rows, 10, energy),
    );
    let g2 = phase(
        column(&model.rows, 12, energy),
        column(&model.rows, 13, energy),
        column(&model.rows, 14, energy),
    );
    let g2y = phase(
        column(&model.rows, 15, energy),
        column(&model.rows, 16, energy),
        column(&model.rows, 17, energy),
    );
    rec[8..14].copy_from_slice(&g1);
    rec[14..20].copy_from_slice(&g1y);
    rec[20..26].copy_from_slice(&g2);
    rec[26..32].copy_from_slice(&g2y);
    if wet_mm > 0.0 || wet_mm.is_nan() {
        let wet = if wet_mm.is_nan() {
            model.shifter_wet
        } else {
            wet_mm
        };
        let tables = mc_tables();
        let medium = tables
            .ids
            .iter()
            .position(|id| *id == model.shifter_material)
            .ok_or_else(|| {
                format!(
                    "range shifter material {} is not in the tables",
                    model.shifter_material
                )
            })?;
        let spr = model.shifter_density * tables.spr[medium];
        let distance = if distance_mm.is_finite() {
            distance_mm
        } else {
            400.0
        };
        rec[32] = wet / spr / 10.0;
        rec[33] = distance / 10.0;
        rec[34] = medium as f32;
        rec[35] = model.shifter_density;
    }
    Ok(rec)
}

pub fn protons_per_mu(energy: f32) -> f32 {
    protons_per_mu_bdl(energy, "un_rs")
}

pub(crate) fn beam_record_bdl(
    gantry: f32,
    couch: f32,
    position: &str,
    isocenter_mm: [f32; 3],
    bdl: &str,
) -> Result<[f32; 16], String> {
    let model = model_of(bdl);
    let rotation = gantry_to_patient(gantry, couch, position)?;
    let mut rec = [0.0; 16];
    rec[..9].copy_from_slice(&rotation);
    rec[9] = isocenter_mm[0] / 10.0;
    rec[10] = isocenter_mm[1] / 10.0;
    rec[11] = isocenter_mm[2] / 10.0;
    rec[12] = model.nozzle;
    rec[13] = model.smx;
    rec[14] = model.smy;
    Ok(rec)
}

pub(crate) fn spot_record_bdl(
    energy: f32,
    x: f32,
    y: f32,
    beam: f32,
    wet_mm: f32,
    distance_mm: f32,
    bdl: &str,
) -> Result<[f32; 40], String> {
    fill_spot(&model_of(bdl), energy, x, y, beam, wet_mm, distance_mm)
}

pub(crate) fn protons_per_mu_bdl(energy: f32, bdl: &str) -> f32 {
    column(&model_of(bdl).rows, 3, energy)
}

fn model_of(name: &str) -> Model {
    let mut nozzle = 0.0;
    let mut smx = 0.0;
    let mut smy = 0.0;
    let mut shifter_material = 0;
    let mut shifter_density = 0.0;
    let mut shifter_wet = 0.0;
    let mut rows = Vec::new();
    let mut lines = bdl_text(name).lines();
    while let Some(line) = lines.next() {
        let head = line.trim();
        if head == "Nozzle exit to Isocenter distance" {
            nozzle = number(lines.next().unwrap_or(""));
        } else if head == "SMX to Isocenter distance" {
            smx = number(lines.next().unwrap_or(""));
        } else if head == "SMY to Isocenter distance" {
            smy = number(lines.next().unwrap_or(""));
        } else if head == "Range Shifter parameters" {
            for line in lines.by_ref() {
                if !line.contains('=') {
                    break;
                }
                let (key, value) = line.split_once('=').unwrap_or(("", ""));
                let value = value.split('#').next().unwrap_or("").trim();
                match key.trim() {
                    "RS_material" => shifter_material = value.parse().unwrap_or(0),
                    "RS_density" => shifter_density = value.parse().unwrap_or(0.0),
                    "RS_WET" => shifter_wet = value.parse().unwrap_or(0.0),
                    _ => {}
                }
            }
        } else if head == "Beam parameters" {
            let _count = lines.next();
            let _header = lines.next();
            for line in lines.by_ref() {
                let values: Vec<f32> = line
                    .split_whitespace()
                    .filter_map(|v| v.parse().ok())
                    .collect();
                if values.len() == 18 {
                    let mut row = [0.0; 18];
                    row.copy_from_slice(&values);
                    rows.push(row);
                }
            }
        }
    }
    Model {
        nozzle,
        smx,
        smy,
        rows,
        shifter_material,
        shifter_density,
        shifter_wet,
    }
}

fn number(line: &str) -> f32 {
    line.split_whitespace()
        .next()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0)
}

fn column(rows: &[[f32; 18]], col: usize, energy: f32) -> f32 {
    let index = rows
        .partition_point(|row| row[0] < energy)
        .saturating_sub(1);
    let index = index.min(rows.len() - 2);
    let span = rows[index + 1][0] - rows[index][0];
    let t = if span == 0.0 {
        0.0
    } else {
        (energy - rows[index][0]) / span
    };
    rows[index][col] + t * (rows[index + 1][col] - rows[index][col])
}

fn phase(size: f32, divergence: f32, correlation: f32) -> [f32; 6] {
    let a = size * size;
    let c = divergence * divergence;
    let b = correlation * size * divergence;
    let disc = ((a - c) * (a - c) + 4.0 * b * b).sqrt();
    let large = 0.5 * (a + c + disc);
    let small = 0.5 * (a + c - disc);
    let big = eigen(a, b, c, large);
    let little = eigen(a, b, c, small);
    [
        big[0],
        little[0],
        big[1],
        little[1],
        large.max(0.0).sqrt(),
        small.max(0.0).sqrt(),
    ]
}

fn eigen(a: f32, b: f32, c: f32, lambda: f32) -> [f32; 2] {
    if b.abs() < 1e-8 {
        if (lambda - a).abs() <= (lambda - c).abs() {
            [1.0, 0.0]
        } else {
            [0.0, 1.0]
        }
    } else {
        let v = [b, lambda - a];
        let n = (v[0] * v[0] + v[1] * v[1]).sqrt().max(1e-12);
        [v[0] / n, v[1] / n]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_hundred_mev_spot_uses_the_library_and_can_insert_the_shifter() {
        let plain = spot_record(100.0, 1.0, -2.0, 0.0, 0.0, f32::NAN).unwrap();
        assert!((plain[2] - 100.533).abs() < 0.01);
        assert_eq!(plain[32], 0.0);
        let shifted = spot_record(100.0, 0.0, 0.0, 0.0, f32::NAN, f32::NAN).unwrap();
        assert!(shifted[32] > 0.0);
        assert!((shifted[35] - 1.19).abs() < 1e-3);
        assert!(protons_per_mu(100.0) > 1.0e7);
        let downstream = spot_record_bdl(100.0, 0.0, 0.0, 0.0, 0.0, f32::NAN, "dn").unwrap();
        assert_eq!(downstream[32], 0.0);
        assert!(protons_per_mu_bdl(100.0, "dn").is_finite());
    }
}
