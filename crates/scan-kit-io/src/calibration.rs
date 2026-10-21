//! CT calibration: HU to mass density and MCsquare material, from the
//! `Scanners/<name>` table folders MCsquare ships. Ports
//! `scan_kit/dicom/calibration.py` — same interpolation, same material
//! thresholds, same provenance digest.

use std::fs;
use std::path::Path;

use sha2::{Digest, Sha256};

use crate::mc_tables::mc_tables;

const HU_SCAN_LO: f64 = -1100.0;
const HU_SCAN_HI: f64 = 4000.0;
const MIN_DENSITY: f64 = 1e-6;

#[derive(Clone, Debug)]
pub struct CtCalibration {
    pub name: String,
    /// Ascending HU breakpoints of the density curve.
    pub hu_points: Vec<f64>,
    /// Mass density g/cm³ at each `hu_points` breakpoint.
    pub densities: Vec<f64>,
    /// HU at which each material starts.
    pub hu_thresholds: Vec<f64>,
    /// MCsquare material label at each threshold.
    pub labels: Vec<i32>,
    /// Row in the MCsquare media table for each threshold.
    media: Vec<u8>,
}

const BUILTINS: &[(&str, &str, &str, &str)] = &[
    (
        "default",
        "Default",
        include_str!("../../../scan_kit/assets/mcsquare/Scanners/default/HU_Density_Conversion.txt"),
        include_str!("../../../scan_kit/assets/mcsquare/Scanners/default/HU_Material_Conversion.txt"),
    ),
    (
        "water",
        "Water phantom",
        include_str!("../../../scan_kit/assets/mcsquare/Scanners/Water_Phantom/HU_Density_Conversion.txt"),
        include_str!("../../../scan_kit/assets/mcsquare/Scanners/Water_Phantom/HU_Material_Conversion.txt"),
    ),
    (
        "solid",
        "Solid phantom",
        include_str!("../../../scan_kit/assets/mcsquare/Scanners/SolidWater_Phantom/HU_Density_Conversion.txt"),
        include_str!("../../../scan_kit/assets/mcsquare/Scanners/SolidWater_Phantom/HU_Material_Conversion.txt"),
    ),
];

impl CtCalibration {
    /// Resolve a scanner: a builtin name ("default", "water", "solid") or a
    /// folder holding `HU_Density_Conversion.txt` and
    /// `HU_Material_Conversion.txt`.
    pub fn resolve(scanner: &str) -> Result<CtCalibration, String> {
        if let Some((_, name, density, material)) =
            BUILTINS.iter().find(|(id, _, _, _)| *id == scanner)
        {
            return CtCalibration::parse(name, density, material);
        }
        let folder = Path::new(scanner);
        let density = fs::read_to_string(folder.join("HU_Density_Conversion.txt"))
            .map_err(|err| format!("scanner {scanner}: {err}"))?;
        let material = fs::read_to_string(folder.join("HU_Material_Conversion.txt"))
            .map_err(|err| format!("scanner {scanner}: {err}"))?;
        let name = folder
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or(scanner)
            .to_owned();
        CtCalibration::parse(&name, &density, &material)
    }

    /// Parse the two MCsquare conversion tables. Comments are `#`; each row is
    /// `HU <value>`, and the material table's value is an MCsquare material
    /// label (its row in the media table).
    pub fn parse(name: &str, density: &str, material: &str) -> Result<CtCalibration, String> {
        let (hu_points, densities) = pairs(density);
        let (hu_thresholds, label_f) = pairs(material);
        if hu_points.len() < 2 {
            return Err(format!("scanner {name}: HU density table needs two points"));
        }
        if hu_thresholds.is_empty() {
            return Err(format!("scanner {name}: HU material table is empty"));
        }
        if !strictly_increasing(&hu_points) || !strictly_increasing(&hu_thresholds) {
            return Err(format!(
                "scanner {name}: HU columns must be strictly increasing"
            ));
        }
        let labels: Vec<i32> = label_f.iter().map(|value| *value as i32).collect();
        let ids = &mc_tables().ids;
        let mut media = Vec::with_capacity(labels.len());
        let mut missing = Vec::new();
        for label in &labels {
            match ids.iter().position(|id| id == label) {
                Some(index) => media.push(index as u8),
                None => missing.push(label.to_string()),
            }
        }
        if !missing.is_empty() {
            return Err(format!(
                "scanner {name}: material labels not in the MCsquare table: {}",
                missing.join(", ")
            ));
        }
        Ok(CtCalibration {
            name: name.to_owned(),
            hu_points,
            densities,
            hu_thresholds,
            labels,
            media,
        })
    }

    /// Provenance digest: sha256 over the four tables as f64, first 16 hex
    /// characters — identical to `CtCalibration.digest` in Python.
    pub fn digest(&self) -> String {
        let mut hash = Sha256::new();
        for table in [&self.hu_points, &self.densities, &self.hu_thresholds] {
            for value in table {
                hash.update(value.to_le_bytes());
            }
        }
        for label in &self.labels {
            hash.update((*label as f64).to_le_bytes());
        }
        hash.finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
            .chars()
            .take(16)
            .collect()
    }

    /// Mass density in g/cm³ — linear in the density breakpoints, clamped to
    /// the first/last segment, floored at 1e-6.
    pub fn density(&self, hu: f64) -> f64 {
        let i = below(&self.hu_points, hu).clamp(0, self.hu_points.len() as i64 - 2) as usize;
        let (x0, x1) = (self.hu_points[i], self.hu_points[i + 1]);
        let (y0, y1) = (self.densities[i], self.densities[i + 1]);
        let density = (y1 - y0) / (x1 - x0) * (hu - x0) + y0;
        density.max(MIN_DENSITY)
    }

    /// The MCsquare media-table row active at `hu` — the last strictly-below
    /// material threshold.
    pub fn material(&self, hu: f64) -> u8 {
        let i = below(&self.hu_thresholds, hu).clamp(0, self.media.len() as i64 - 1) as usize;
        self.media[i]
    }

    /// Mass stopping power of the material at `hu`, relative to water at the
    /// MCsquare reference energy.
    pub fn relative_stop(&self, hu: f64) -> f64 {
        mc_tables().spr[self.material(hu) as usize] as f64
    }

    /// Stopping-power ratio: density × the material's relative stopping power.
    pub fn spr(&self, hu: f64) -> f64 {
        self.density(hu) * self.relative_stop(hu)
    }

    /// Media index + density for a HU volume, through an integer LUT when the
    /// values are whole HUs in a <2^16 span — the fast path the CT is.
    pub fn voxels(&self, hu: &[f32]) -> (Vec<u8>, Vec<f32>) {
        let integral = hu.iter().all(|value| value.fract() == 0.0);
        let (lo, hi) = hu
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &value| {
                (lo.min(value), hi.max(value))
            });
        if integral && hi - lo < 65536.0 && !hu.is_empty() {
            let lo = lo.floor() as i64;
            let n = (hi.ceil() as i64 - lo + 1) as usize;
            let mut lut_m = Vec::with_capacity(n);
            let mut lut_d = Vec::with_capacity(n);
            for k in 0..n {
                let key = (lo + k as i64) as f64;
                lut_m.push(self.material(key));
                lut_d.push(self.density(key) as f32);
            }
            let media = hu
                .iter()
                .map(|value| lut_m[(*value as i64 - lo) as usize])
                .collect();
            let density = hu
                .iter()
                .map(|value| lut_d[(*value as i64 - lo) as usize])
                .collect();
            return (media, density);
        }
        let media = hu
            .iter()
            .map(|value| self.material(*value as f64))
            .collect();
        let density = hu
            .iter()
            .map(|value| self.density(*value as f64) as f32)
            .collect();
        (media, density)
    }

    /// Media index for a mass density, found on the monotone density curve the
    /// calibration defines over the HU scan; density passes through.
    pub fn from_density(&self, density: f64) -> (u8, f64) {
        let (scan, curve) = self.density_curve();
        let hu = interp(density, &curve, &scan);
        (self.material(hu), density)
    }

    /// Media index + mass density for an SPR value, on the monotone SPR curve.
    pub fn from_spr(&self, spr: f64) -> (u8, f64) {
        let (scan, curve) = self.spr_curve();
        let hu = interp(spr, &curve, &scan);
        let material = self.material(hu);
        (material, spr / self.relative_stop_of(material))
    }

    fn relative_stop_of(&self, material: u8) -> f64 {
        mc_tables().spr[material as usize] as f64
    }

    fn density_curve(&self) -> (Vec<f64>, Vec<f64>) {
        let scan = hu_scan();
        let mut curve = Vec::with_capacity(scan.len());
        let mut max = f64::NEG_INFINITY;
        for &hu in &scan {
            max = max.max(self.density(hu));
            curve.push(max);
        }
        (scan, curve)
    }

    fn spr_curve(&self) -> (Vec<f64>, Vec<f64>) {
        let scan = hu_scan();
        let mut curve = Vec::with_capacity(scan.len());
        let mut max = f64::NEG_INFINITY;
        for &hu in &scan {
            max = max.max(self.spr(hu));
            curve.push(max);
        }
        (scan, curve)
    }
}

fn hu_scan() -> Vec<f64> {
    let n = (HU_SCAN_HI - HU_SCAN_LO) as usize + 1;
    (0..n).map(|k| HU_SCAN_LO + k as f64).collect()
}

/// `searchsorted(values, x, side="left") - 1` — the index of the last value
/// strictly below `x`, or -1 before the first breakpoint.
fn below(xs: &[f64], x: f64) -> i64 {
    xs.partition_point(|&value| value < x) as i64 - 1
}

fn strictly_increasing(xs: &[f64]) -> bool {
    xs.windows(2).all(|pair| pair[1] > pair[0])
}

fn pairs(text: &str) -> (Vec<f64>, Vec<f64>) {
    let mut x = Vec::new();
    let mut y = Vec::new();
    for line in text.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        let mut parts = line.split_whitespace();
        let Some(a) = parts.next().and_then(|value| value.parse::<f64>().ok()) else {
            continue;
        };
        let Some(b) = parts.next().and_then(|value| value.parse::<f64>().ok()) else {
            continue;
        };
        x.push(a);
        y.push(b);
    }
    (x, y)
}

/// `np.interp(x, xp, fp)` on a non-decreasing `xp`: linear inside, clamped to
/// the ends. Flat runs return the left edge, matching numpy.
fn interp(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    if x <= xp[0] {
        return fp[0];
    }
    if x >= xp[xp.len() - 1] {
        return fp[fp.len() - 1];
    }
    let i = (xp.partition_point(|&value| value <= x) - 1).min(xp.len() - 2);
    let (x0, x1) = (xp[i], xp[i + 1]);
    if x1 == x0 {
        return fp[i];
    }
    (fp[i + 1] - fp[i]) / (x1 - x0) * (x - x0) + fp[i]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_calibration() -> CtCalibration {
        CtCalibration::resolve("default").unwrap()
    }

    #[test]
    fn builtin_scanners_parse() {
        for (id, _, _, _) in BUILTINS {
            let calibration = CtCalibration::resolve(id).unwrap();
            assert!(calibration.hu_points.len() >= 2, "{id}");
            assert_eq!(calibration.labels.len(), calibration.media.len(), "{id}");
        }
    }

    #[test]
    fn a_missing_scanner_folder_reports_the_path() {
        let err = CtCalibration::resolve("no/such/scanner").unwrap_err();
        assert!(err.contains("no/such/scanner"), "{err}");
    }

    #[test]
    fn density_interpolates_and_floors() {
        let calibration = default_calibration();
        // -1000 HU is air in the MCsquare table.
        assert!((calibration.density(-1000.0) - 0.001).abs() < 1e-9);
        // Water sits at 0 HU.
        assert!((calibration.density(0.0) - 1.0).abs() < 0.05);
        // A HU far below the table extrapolates but never reaches zero.
        assert!(calibration.density(-20000.0) >= MIN_DENSITY - 1e-12);
    }

    #[test]
    fn material_uses_the_last_threshold_below() {
        let calibration = default_calibration();
        // Below the first threshold the first material still wins.
        assert_eq!(calibration.material(-10_000.0), calibration.media[0]);
        // Python: CtCalibration.mcsquare("default").material(0) == 11
        // (Schneider_AT_AG_SI4 — the Schneider table is tissue, not water).
        assert_eq!(calibration.material(0.0), 11);
        assert_eq!(CtCalibration::resolve("water").unwrap().material(0.0), 0);
        assert_eq!(CtCalibration::resolve("solid").unwrap().material(0.0), 30);
    }

    #[test]
    fn spr_is_density_times_relative_stop() {
        let calibration = default_calibration();
        let hu = 0.0;
        assert!(
            (calibration.spr(hu) - calibration.density(hu) * calibration.relative_stop(hu)).abs()
                < 1e-6
        );
        // Python: spr(0) == 1.0170777620289144, water spr(0) == 1.0,
        // solid spr(0) == 1.031642819533632.
        assert!((calibration.spr(0.0) - 1.0170777620289144).abs() < 1e-6);
        assert_eq!(CtCalibration::resolve("water").unwrap().spr(0.0), 1.0);
        assert!(
            (CtCalibration::resolve("solid").unwrap().spr(0.0) - 1.031642819533632).abs() < 1e-6
        );
    }

    #[test]
    fn the_digest_matches_python_for_every_builtin() {
        // Digests produced by scan_kit.dicom.calibration.CtCalibration.
        let digests = [
            ("default", "b17da9d0afeb26cd"),
            ("water", "9001ab0a57c6499e"),
            ("solid", "949fa193a50293ad"),
        ];
        for (id, digest) in digests {
            assert_eq!(CtCalibration::resolve(id).unwrap().digest(), digest, "{id}");
        }
    }

    #[test]
    fn voxels_matches_the_scalar_path() {
        let calibration = default_calibration();
        let hu = vec![-1000.0f32, -500.0, 0.0, 300.0, 1000.0, 2500.0];
        let (media, density) = calibration.voxels(&hu);
        for (i, &value) in hu.iter().enumerate() {
            assert_eq!(media[i], calibration.material(value as f64));
            assert!((density[i] as f64 - calibration.density(value as f64)).abs() < 1e-5);
        }
        // Non-integral HU skips the LUT but gives identical answers.
        let (media2, density2) = calibration.voxels(&[-999.5f32, 0.4]);
        assert_eq!(media2[0], calibration.material(-999.5));
        assert!((density2[1] as f64 - calibration.density(0.4)).abs() < 1e-5);
    }

    #[test]
    fn from_density_and_from_spr_round_trip() {
        let calibration = default_calibration();
        // Python: from_density(1.0) -> media 11; from_spr(1.0) -> media 9,
        // density 0.97378534.
        let (material, density) = calibration.from_density(1.0);
        assert_eq!(material, 11);
        assert!((density - 1.0).abs() < 1e-9);
        let (material, density) = calibration.from_spr(1.0);
        assert_eq!(material, 9);
        assert!((density - 0.97378534).abs() < 1e-5);
    }
}
