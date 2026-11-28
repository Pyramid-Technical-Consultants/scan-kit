//! UPenn double-Gaussian beam model. Records match the transport shader's stride of 40.

use std::path::Path;

use scan_kit_dicom::gantry_to_patient;

use super::mc_tables::mc_tables;

const BDL_UN_RS: &str =
    include_str!("../../../assets/mcsquare/BDL/BDL_default_UN_RangeShifter.txt");
const BDL_UN: &str = include_str!("../../../assets/mcsquare/BDL/BDL_default_UN.txt");
const BDL_DN: &str = include_str!("../../../assets/mcsquare/BDL/BDL_default_DN.txt");
const BDL_DN_RS: &str =
    include_str!("../../../assets/mcsquare/BDL/BDL_default_DN_RangeShifter.txt");

pub(crate) fn bdl_name(name: &str) -> &'static str {
    match name {
        "un" => "BDL_default_UN",
        "dn" => "BDL_default_DN",
        "dn_rs" => "BDL_default_DN_RangeShifter",
        _ => "BDL_default_UN_RangeShifter",
    }
}

pub(crate) fn bdl_text(name: &str) -> &'static str {
    match name {
        "un" => BDL_UN,
        "dn" => BDL_DN,
        "dn_rs" => BDL_DN_RS,
        _ => BDL_UN_RS,
    }
}

/// A parsed BDL machine: nozzle optics, energy rows, range-shifter defaults.
pub struct BeamModel {
    pub nozzle: f32,
    pub smx: f32,
    pub smy: f32,
    pub(crate) rows: Vec<[f32; 18]>,
    shifter_material: i32,
    shifter_density: f32,
    shifter_wet: f32,
}

impl BeamModel {
    /// A BDL `.txt` on disk — the validator's `third_party/MCsquare/BDL` files.
    pub fn read(path: &Path) -> Result<BeamModel, String> {
        let text = std::fs::read_to_string(path).map_err(|err| format!("{path:?}: {err}"))?;
        let model = BeamModel::parse(&text)?;
        if model.rows.len() < 2 {
            return Err(format!(
                "{path:?} has fewer than 2 energy rows — not a usable BDL file"
            ));
        }
        if !model.rows.windows(2).all(|pair| pair[1][0] > pair[0][0]) {
            return Err(format!("{path:?} energy rows must be strictly increasing"));
        }
        if !model.rows.iter().flatten().all(|value| value.is_finite()) {
            return Err(format!("{path:?} energy rows must be finite"));
        }
        // The weight pair divides w1/(w1+w2) and ProtonsMU scales every
        // transport — a 0/0 or non-positive value must fail here, not as a
        // NaN record downstream.
        if model
            .rows
            .iter()
            .any(|row| row[3] <= 0.0 || row[4] < 0.0 || row[11] < 0.0 || row[4] + row[11] <= 0.0)
        {
            return Err(format!(
                "{path:?} has a non-positive ProtonsMU or weight pair"
            ));
        }
        // `<= 0.0` alone misses NaN — a literal "NaN" distance must fail too.
        if [model.nozzle, model.smx, model.smy]
            .iter()
            .any(|distance| !distance.is_finite() || *distance <= 0.0)
        {
            return Err(format!("{path:?} is missing nozzle/SMX/SMY distances"));
        }
        Ok(model)
    }

    pub(crate) fn parse(text: &str) -> Result<BeamModel, String> {
        let mut nozzle = 0.0;
        let mut smx = 0.0;
        let mut smy = 0.0;
        let mut shifter_material = 0;
        let mut shifter_density = 0.0;
        let mut shifter_wet = 0.0;
        let mut rows = Vec::new();
        let mut lines = text.lines();
        let mut pending: Option<&str> = None;
        while let Some(line) = pending.take().or_else(|| lines.next()) {
            let head = line.trim();
            if head == "Nozzle exit to Isocenter distance" {
                nozzle = number(lines.next().unwrap_or(""));
            } else if head == "SMX to Isocenter distance" {
                smx = number(lines.next().unwrap_or(""));
            } else if head == "SMY to Isocenter distance" {
                smy = number(lines.next().unwrap_or(""));
            } else if head == "Range Shifter parameters" {
                for line in lines.by_ref() {
                    let trimmed = line.trim();
                    // A comment inside the block is part of it; a blank
                    // line or a section header ends it  hand the
                    // terminator back to the outer dispatch.
                    if !trimmed.starts_with('#') && !trimmed.contains('=') {
                        pending = Some(line);
                        break;
                    }
                    if trimmed.starts_with('#') {
                        continue;
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
                // The count line is optional — bundled files write it as
                // "26 energies". Consume it when the first token is a bare
                // usize and no later token is numeric; a data row's second
                // column always parses as f32.
                let mut held = "";
                let declared = lines.next().and_then(|line| {
                    let text = line.split('#').next().unwrap_or("").trim();
                    let mut words = text.split_whitespace();
                    let count = match (words.next(), words.next()) {
                        (Some(token), rest)
                            if token.parse::<usize>().is_ok()
                                && rest.is_none_or(|word| word.parse::<f32>().is_err()) =>
                        {
                            token.parse::<usize>().ok()
                        }
                        _ => None,
                    };
                    if count.is_none() {
                        held = text;
                    }
                    count
                });
                for line in std::iter::once(held)
                    .filter(|line| !line.is_empty())
                    .chain(lines.by_ref())
                {
                    let text = line.split('#').next().unwrap_or("").trim();
                    let parsed: Option<Vec<f32>> =
                        text.split_whitespace().map(|v| v.parse().ok()).collect();
                    match parsed {
                        Some(values) if values.len() == 18 => {
                            let mut row = [0.0; 18];
                            row.copy_from_slice(&values);
                            rows.push(row);
                        }
                        _ if text.is_empty() && rows.is_empty() => {}
                        _ if text.is_empty() => break,
                        _ if text
                            .split_whitespace()
                            .next()
                            .is_some_and(|first| first.parse::<f32>().is_ok()) =>
                        {
                            // Starts with a number but isn't a clean
                            // 18-column row - corrupt data, not a header.
                            return Err(format!("malformed energy row in BDL table: {text:?}"));
                        }
                        // Non-numeric junk before the first row.
                        _ if rows.is_empty() => {}
                        // A section header ends the table  hand it back
                        // to the outer dispatch.
                        _ => {
                            pending = Some(line);
                            break;
                        }
                    }
                }
                // The declared count bounds the table  a corrupted row
                // must not silently truncate it.
                if declared.is_some_and(|n| n != rows.len()) {
                    return Err(format!(
                        "BDL declares {declared:?} energies but only {} parsed",
                        rows.len()
                    ));
                }
            }
        }
        Ok(BeamModel {
            nozzle,
            smx,
            smy,
            rows,
            shifter_material,
            shifter_density,
            shifter_wet,
        })
    }

    /// Kernel beam record: *rotation* takes the beam frame to the volume frame,
    /// where the isocenter sits at *isocenter_mm*.
    pub fn beam_record(&self, rotation: &[f32; 9], isocenter_mm: [f32; 3]) -> [f32; 16] {
        let mut rec = [0.0; 16];
        rec[..9].copy_from_slice(rotation);
        rec[9] = isocenter_mm[0] / 10.0;
        rec[10] = isocenter_mm[1] / 10.0;
        rec[11] = isocenter_mm[2] / 10.0;
        rec[12] = self.nozzle;
        rec[13] = self.smx;
        rec[14] = self.smy;
        rec
    }

    /// Kernel spot record through this machine (stride 40).
    pub fn spot_record(
        &self,
        energy: f32,
        x: f32,
        y: f32,
        beam: f32,
        wet_mm: f32,
        distance_mm: f32,
    ) -> Result<[f32; 40], String> {
        fill_spot(self, energy, x, y, beam, wet_mm, distance_mm)
    }

    pub fn protons_per_mu(&self, energy: f32) -> Result<f32, String> {
        check_energy(&self.rows, energy)?;
        Ok(column(&self.rows, 3, energy))
    }
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
    model: &BeamModel,
    energy: f32,
    x: f32,
    y: f32,
    beam: f32,
    wet_mm: f32,
    distance_mm: f32,
) -> Result<[f32; 40], String> {
    check_energy(&model.rows, energy)?;
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
    if wet_mm < 0.0 {
        return Err(format!("negative range shifter thickness {wet_mm}"));
    }
    if wet_mm == f32::INFINITY {
        return Err("infinite range shifter thickness".to_string());
    }
    if wet_mm > 0.0 || wet_mm.is_nan() {
        let wet = if wet_mm.is_nan() {
            model.shifter_wet
        } else {
            wet_mm
        };
        if !wet.is_finite() || wet < 0.0 {
            return Err(format!("range shifter WET {wet} is not usable"));
        }
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
        if !spr.is_finite() || spr <= 0.0 {
            return Err(format!(
                "range shifter density {} gives no stopping power",
                model.shifter_density
            ));
        }
        if distance_mm < 0.0 {
            return Err(format!("negative range shifter distance {distance_mm}"));
        }
        let distance = if distance_mm.is_finite() && distance_mm > 0.0 {
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

pub fn protons_per_mu(energy: f32) -> Result<f32, String> {
    protons_per_mu_bdl(energy, "un_rs")
}

/// Beam record from a parsed model — the parse-once path. Registered and
/// bundled models share it.
pub(crate) fn beam_record_model(
    model: &BeamModel,
    gantry: f32,
    couch: f32,
    position: &str,
    isocenter_mm: [f32; 3],
) -> Result<[f32; 16], String> {
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

pub(crate) fn spot_record_model(
    model: &BeamModel,
    energy: f32,
    x: f32,
    y: f32,
    beam: f32,
    wet_mm: f32,
    distance_mm: f32,
) -> Result<[f32; 40], String> {
    fill_spot(model, energy, x, y, beam, wet_mm, distance_mm)
}

pub(crate) fn protons_per_mu_model(model: &BeamModel, energy: f32) -> Result<f32, String> {
    check_energy(&model.rows, energy)?;
    Ok(column(&model.rows, 3, energy))
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

pub(crate) fn protons_per_mu_bdl(energy: f32, bdl: &str) -> Result<f32, String> {
    let model = model_of(bdl);
    check_energy(&model.rows, energy)?;
    Ok(column(&model.rows, 3, energy))
}

/// Outside the measured table `column` extrapolates with the edge slope.
/// The delivery logs already carry the machine-specific optics this table
/// would approximate, and the energy spectrum — the one unmeasurable term —
/// varies slowly enough that edge extrapolation is the smallest error
/// available. Non-finite or non-positive energies stay corruption.
fn check_energy(rows: &[[f32; 18]], energy: f32) -> Result<(), String> {
    if rows.is_empty() {
        return Err("the beam model has no energy rows".into());
    }
    // Outside the measured table the columns extrapolate with the edge
    // slope. That is approximation — the delivery logs already carry the
    // machine-specific optics this table would otherwise approximate, and
    // the energy spectrum varies slowly enough that edge extrapolation is
    // the smallest-error term. Only non-finite or non-positive energies are
    // corruption: positive-energy physics has no valid reading.
    if !energy.is_finite() || energy <= 0.0 {
        return Err(format!("energy {energy} MeV is not a valid beam energy"));
    }
    Ok(())
}

fn model_of(name: &str) -> BeamModel {
    BeamModel::parse(bdl_text(name)).expect("bundled BDL text parses")
}

fn number(line: &str) -> f32 {
    line.split_whitespace()
        .next()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0)
}

fn column(rows: &[[f32; 18]], col: usize, energy: f32) -> f32 {
    if rows.len() < 2 {
        return rows.first().map(|row| row[col]).unwrap_or(0.0);
    }
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
    // a ~ mm^2 dwarfs c ~ rad^2, so a + c - disc cancels ~6 digits - the
    // determinant (det = product of the eigenvalues) is stable in f32.
    let small = ((a * c - b * b) / large).max(0.0);
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
    fn a_bdl_without_energy_rows_errors_instead_of_panicking() {
        let dir = std::env::temp_dir().join(format!("scan-kit-bdl-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("empty.txt");
        std::fs::write(&file, "Nozzle exit to Isocenter distance\n300\n").unwrap();
        let err = BeamModel::read(&file).err().unwrap();
        assert!(err.contains("energy rows"));
        // column() itself stays safe even on a degenerate model.
        assert!(BeamModel::parse("")
            .expect("empty parses")
            .protons_per_mu(100.0)
            .is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_hundred_mev_spot_uses_the_library_and_can_insert_the_shifter() {
        let plain = spot_record(100.0, 1.0, -2.0, 0.0, 0.0, f32::NAN).unwrap();
        assert!((plain[2] - 100.533).abs() < 0.01);
        assert_eq!(plain[32], 0.0);
        let shifted = spot_record(100.0, 0.0, 0.0, 0.0, f32::NAN, f32::NAN).unwrap();
        assert!(shifted[32] > 0.0);
        assert!((shifted[35] - 1.19).abs() < 1e-3);
        assert!(protons_per_mu(100.0).unwrap() > 1.0e7);
        let downstream = spot_record_bdl(100.0, 0.0, 0.0, 0.0, 0.0, f32::NAN, "dn").unwrap();
        assert_eq!(downstream[32], 0.0);
        assert!(protons_per_mu_bdl(100.0, "dn").unwrap().is_finite());
    }

    #[test]
    fn unsorted_rows_or_missing_geometry_error() {
        // Columns 3/4 must be positive — the parser validates ProtonsMU
        // and the weight pair now.
        let row = |first: &str| format!("{first} 1 1 1 1 {}", "0 ".repeat(13));
        let geometry = "Nozzle exit to Isocenter distance
100
 SMX to Isocenter distance
50
SMY to Isocenter distance
50
";
        let dir = std::env::temp_dir().join(format!("scan-kit-bdl-sort-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let descending = format!(
            "{geometry}Beam parameters
2
h
{}
{}
",
            row("200"),
            row("100")
        );
        let file = dir.join("desc.txt");
        std::fs::write(&file, &descending).unwrap();
        let err = BeamModel::read(&file).err().unwrap();
        assert!(err.contains("increasing"), "{err}");

        let missing_smx = format!(
            "Nozzle exit to Isocenter distance
100
Beam parameters
2
h
{}
{}
",
            row("100"),
            row("200")
        );
        let file = dir.join("nogeo.txt");
        std::fs::write(&file, &missing_smx).unwrap();
        let err = BeamModel::read(&file).err().unwrap();
        assert!(err.contains("SMX"), "{err}");

        let good = format!(
            "{geometry}Beam parameters
2
h
{}
{}
",
            row("100"),
            row("200")
        );
        let file = dir.join("good.txt");
        std::fs::write(&file, &good).unwrap();
        assert!(BeamModel::read(&file).is_ok());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_energy_outside_the_model_range_extrapolates() {
        // The table extrapolates with the edge slope rather than failing:
        // delivery logs already carry the machine-specific optics, and the
        // spectrum is the smallest-error term. 70 MeV must resolve.
        assert!(protons_per_mu(70.0).is_ok());
        assert!(protons_per_mu(94.0).is_ok());
        // Non-finite and non-positive energies stay corruption.
        assert!(protons_per_mu(0.0).is_err());
        assert!(protons_per_mu(f32::NAN).is_err());
        assert!(protons_per_mu(-1.0).is_err());
    }

    #[test]
    fn an_infinite_wet_is_rejected_like_a_negative_one() {
        let model = model_of("un_rs");
        assert!(model
            .spot_record(150.0, 0.0, 0.0, 0.0, f32::NEG_INFINITY, 300.0)
            .is_err());
        assert!(model
            .spot_record(150.0, 0.0, 0.0, 0.0, f32::INFINITY, 300.0)
            .is_err());
    }

    #[test]
    fn a_malformed_energy_row_is_an_error_not_a_truncation() {
        // One corrupt token in a data row must fail loudly rather than
        // silently drop every row below it.
        let text = bdl_text("un_rs").replace("95212461.7", "oops");
        assert!(BeamModel::parse(&text).is_err());
    }

    #[test]
    fn a_header_after_the_shifter_block_is_not_eaten() {
        // No blank line between RS_WET and the next section - the
        // terminator must be handed back to the outer dispatch.
        let row = "100.0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1";
        let row2 = "200.0 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2";
        let text = format!(
            "Range Shifter parameters
RS_WET = 40
# comment inside the block
RS_density = 2.5
Beam parameters
2
h
{row}
{row2}
"
        );
        let model = BeamModel::parse(&text).expect("parses");
        assert_eq!(model.shifter_density, 2.5);
        assert_eq!(model.rows.len(), 2);
    }

    #[test]
    fn a_mixed_scale_phase_matrix_keeps_the_small_axis() {
        // size^2 mm^2 next to divergence^2 rad^2 makes a + c - disc cancel
        // ~6 digits in f32 - the 210 MeV DN row used to land ~30% off.
        let out = phase(11.508698, 0.006344, 0.788183);
        let want = 3.905e-3;
        assert!(
            (out[5] - want).abs() / want < 0.05,
            "sigma_small {} vs {want}",
            out[5]
        );
    }

    #[test]
    fn a_countless_beam_table_keeps_every_row() {
        // With no declared count the header line is junk before row 1 - it
        // must not be consumed as if it were a data row.
        let row = "100.0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1";
        let row2 = "200.0 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2";
        let text = format!(
            "Beam parameters
hdr
{row}
{row2}
"
        );
        let model = BeamModel::parse(&text).expect("parses");
        assert_eq!(model.rows.len(), 2);
    }

    #[test]
    fn a_non_finite_energy_row_is_rejected() {
        // Columns 3/4 must be positive — the parser validates ProtonsMU
        // and the weight pair now.
        let row = |first: &str| format!("{first} 1 1 1 1 {}", "0 ".repeat(13));
        let text = format!(
            "Nozzle exit to Isocenter distance
100
 SMX to Isocenter distance
50
SMY to Isocenter distance
50
Beam parameters
2
h
{}
{}
",
            row("100"),
            row("inf")
        );
        let dir = std::env::temp_dir().join(format!("scan-kit-bdl-inf-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("inf.txt");
        std::fs::write(&file, &text).unwrap();
        let err = BeamModel::read(&file).err().unwrap();
        assert!(err.contains("finite"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_negative_library_wet_is_rejected_not_dropped() {
        // RS_WET <= 0 used to write rec[32] <= 0 - the shader then skips the
        // shifter the plan asked for, silently.
        let row = "100.0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1";
        let row2 = "200.0 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2";
        let text = format!(
            "Nozzle exit to Isocenter distance
100
 SMX to Isocenter distance
50
SMY to Isocenter distance
50
Range Shifter parameters
RS_material = 1
RS_density = 1.0
RS_WET = -5
Beam parameters
2
h
{row}
{row2}
"
        );
        let dir = std::env::temp_dir().join(format!("scan-kit-bdl-wet-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("wet.txt");
        std::fs::write(&file, &text).unwrap();
        let model = BeamModel::read(&file).unwrap();
        let err = model
            .spot_record(100.0, 0.0, 0.0, 0.0, f32::NAN, f32::NAN)
            .err()
            .unwrap();
        assert!(err.contains("WET"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
