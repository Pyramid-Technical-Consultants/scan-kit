//! Chamber and magnet geometry shared by trajectory and dose.

/// IC2 is the plot origin. IC1 sits 100 mm downstream.
pub const IC2_Z_MM: f32 = 0.0;
pub const IC1_Z_MM: f32 = 100.0;
pub const IC_SEP_MM: f32 = IC1_Z_MM - IC2_Z_MM;

/// A parsed `devices.xml` IC entry. Missing attributes stay at the defaults.
#[derive(Clone, Debug, PartialEq)]
pub struct IcGeometry {
    pub name: String,
    pub z_mm: f32,
}

/// Pull IC z attributes out of a devices.xml snippet. Tags are `name` and `z`.
pub fn parse_ic_geometry(xml: &str) -> Vec<IcGeometry> {
    let mut found = Vec::new();
    for chunk in xml.split("<ic").skip(1) {
        let body = chunk.split('>').next().unwrap_or("");
        let name = attr(body, "name").unwrap_or("IC").to_owned();
        let z_mm = attr(body, "z")
            .and_then(|text| text.parse().ok())
            .unwrap_or(0.0);
        found.push(IcGeometry { name, z_mm });
    }
    found
}

fn attr<'a>(body: &'a str, key: &str) -> Option<&'a str> {
    let needle = format!("{key}=\"");
    let start = body.find(&needle)? + needle.len();
    let rest = &body[start..];
    let end = rest.find('"')?;
    Some(&rest[..end])
}

/// Beam angle in mrad from the IC1−IC2 offset over the chamber separation.
pub fn beam_angle_mrad(ic1_mm: f32, ic2_mm: f32) -> f32 {
    ((ic1_mm - ic2_mm) / IC_SEP_MM).atan() * 1000.0
}

/// Least-squares line `y = a + b x` through finite pairs.
pub fn fit_line(xs: &[f32], ys: &[f32]) -> Option<(f32, f32)> {
    let mut n = 0.0f64;
    let mut sx = 0.0f64;
    let mut sy = 0.0f64;
    let mut sxx = 0.0f64;
    let mut sxy = 0.0f64;
    for (x, y) in xs.iter().zip(ys) {
        if x.is_finite() && y.is_finite() {
            let x = f64::from(*x);
            let y = f64::from(*y);
            n += 1.0;
            sx += x;
            sy += y;
            sxx += x * x;
            sxy += x * y;
        }
    }
    if n < 2.0 {
        return None;
    }
    let denom = n * sxx - sx * sx;
    if denom.abs() < 1e-18 {
        return None;
    }
    let slope = (n * sxy - sx * sy) / denom;
    let intercept = (sy - slope * sx) / n;
    Some((intercept as f32, slope as f32))
}

/// Iso-plane fit: lateral position versus energy. Returns `(offset_mm, mm_per_mev)`.
pub fn fit_iso_plane(energy: &[f32], lateral_mm: &[f32]) -> Option<(f32, f32)> {
    fit_line(energy, lateral_mm)
}

/// Magnet pivot: the z where the fitted beam from two chambers crosses x = 0.
pub fn magnet_pivot_z(ic2_x: f32, ic1_x: f32) -> f32 {
    let slope = (ic1_x - ic2_x) / IC_SEP_MM;
    if slope.abs() < 1e-8 {
        return IC2_Z_MM;
    }
    IC2_Z_MM - ic2_x / slope
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sk_req_025_geometry_parses_and_fits() {
        let xml = r#"<devices><ic name="IC2" z="0"/><ic name="IC1" z="100"/></devices>"#;
        let ics = parse_ic_geometry(xml);
        assert_eq!(ics[1].name, "IC1");
        assert!((ics[1].z_mm - 100.0).abs() < 1e-3);
        assert!(beam_angle_mrad(10.0, 0.0).abs() > 0.0);
        let (offset, slope) = fit_iso_plane(&[70.0, 90.0, 110.0], &[0.0, 2.0, 4.0]).unwrap();
        assert!((slope - 0.1).abs() < 1e-3);
        assert!(offset.abs() < 8.0);
        let pivot = magnet_pivot_z(0.0, 10.0);
        assert!(pivot <= IC2_Z_MM);
    }
}
