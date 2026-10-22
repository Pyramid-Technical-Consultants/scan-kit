//! MCsquare material tables. The transport shader's baked offsets assume this packing.

use std::sync::OnceLock;

use crate::npz::read_npz;

const NPZ: &[u8] = include_bytes!("../../../scan_kit/assets/mc_materials.npz");

#[derive(Clone, Debug)]
pub struct McTables {
    pub floats: Vec<f32>,
    pub ints: Vec<i32>,
    pub media: Vec<String>,
    /// MCsquare material labels, one per medium, in table order.
    pub ids: Vec<i32>,
    /// MCsquare material names (`mcsquare_names`), in table order.
    pub mcsquare_names: Vec<String>,
    /// Mass stopping power over water at 100 MeV.
    pub spr: Vec<f32>,
    /// Physical density (g/cm³), one per medium, in table order.
    pub rho: Vec<f64>,
    /// Mass stopping power rows (MeV·cm²/g over `stop_energies`), per medium.
    pub stop: Vec<Vec<f64>>,
    /// Energy grid the `stop` rows are tabulated on (MeV).
    pub stop_energies: Vec<f64>,
}

pub fn mc_tables() -> &'static McTables {
    static TABLES: OnceLock<McTables> = OnceLock::new();
    TABLES.get_or_init(|| pack().expect("mc material tables"))
}

fn pack() -> Result<McTables, String> {
    let files = read_npz(NPZ)?;
    let media = files.get("media").ok_or("media")?.strings()?;
    let props = files.get("m_props").ok_or("m_props")?.f64s()?;
    let frac = files.get("m_frac").ok_or("m_frac")?.f64s()?;
    let stop = files.get("m_stop").ok_or("m_stop")?.f64s()?;
    let nuc = files.get("m_nuc").ok_or("m_nuc")?.f64s()?;
    let comp = files.get("m_comp").ok_or("m_comp")?.i32s()?;
    let kind = files.get("m_type").ok_or("m_type")?.i32s()?;
    let nm = media.len();
    let width = files.get("m_comp").ok_or("m_comp")?.shape[1];
    let stop_bins = files.get("m_stop").ok_or("m_stop")?.shape[1];
    let nuc_bins = files.get("m_nuc").ok_or("m_nuc")?.shape[1];
    let m_stop = 3 + width;
    let m_nuc = m_stop + stop_bins;
    let stride = m_nuc + nuc_bins;
    let mut media_rows = vec![0.0f64; nm * stride];
    for m in 0..nm {
        for k in 0..3 {
            media_rows[m * stride + k] = props[m * 3 + k];
        }
        for k in 0..width {
            media_rows[m * stride + 3 + k] = frac[m * width + k];
        }
        for k in 0..stop_bins {
            media_rows[m * stride + m_stop + k] = stop[m * stop_bins + k];
        }
        for k in 0..nuc_bins {
            media_rows[m * stride + m_nuc + k] = nuc[m * nuc_bins + k];
        }
    }
    let mut floats = Vec::new();
    push_f64(&mut floats, &media_rows);
    for key in [
        "e_A",
        "el_E",
        "el_sigma",
        "el_cdf",
        "in_E",
        "in_sigma",
        "in_mult",
        "in_recoil",
        "sec_E",
        "sec_cD",
        "sec_cDD",
    ] {
        push_f64(&mut floats, &files.get(key).ok_or(key)?.f64s()?);
    }

    let mut m_int = vec![0i32; nm * (2 + width)];
    for m in 0..nm {
        let base = m * (2 + width);
        m_int[base] = kind[m];
        let mut ncomp = 0;
        for k in 0..width {
            let id = comp[m * width + k];
            m_int[base + 2 + k] = id;
            if id >= 0 {
                ncomp += 1;
            }
        }
        m_int[base + 1] = ncomp;
    }
    let e_type = files.get("e_type").ok_or("e_type")?.i32s()?;
    let el_start = files.get("el_start").ok_or("el_start")?.i32s()?;
    let in_start = files.get("in_start").ok_or("in_start")?.i32s()?;
    let ne = e_type.len();
    let mut el = vec![0i32; ne * 5];
    for e in 0..ne {
        el[e * 5] = e_type[e];
        el[e * 5 + 1] = el_start[e];
        el[e * 5 + 2] = el_start[e + 1] - el_start[e];
        el[e * 5 + 3] = in_start[e];
        el[e * 5 + 4] = in_start[e + 1] - in_start[e];
    }
    let mut ints = Vec::new();
    ints.extend_from_slice(&m_int);
    ints.extend_from_slice(&el);
    ints.extend_from_slice(&files.get("in_rows_start").ok_or("in_rows_start")?.i32s()?);
    ints.extend_from_slice(&files.get("in_rows_n").ok_or("in_rows_n")?.i32s()?);
    let ids = files.get("mcsquare_ids").ok_or("mcsquare_ids")?.i32s()?;
    let mcsquare_names = files
        .get("mcsquare_names")
        .ok_or("mcsquare_names")?
        .strings()?;
    let bin = (100.0f64 / 0.5).round() as usize;
    let water = stop[bin];
    let spr = (0..nm)
        .map(|row| (stop[row * stop_bins + bin] / water) as f32)
        .collect();
    Ok(McTables {
        floats,
        ints,
        media,
        ids,
        mcsquare_names,
        spr,
        rho: (0..nm).map(|m| props[m * 3]).collect(),
        stop: (0..nm)
            .map(|m| stop[m * stop_bins..(m + 1) * stop_bins].to_vec())
            .collect(),
        stop_energies: (0..stop_bins).map(|i| i as f64 * 0.5).collect(),
    })
}

/// Physical density (g/cm³) of a table medium.
pub fn medium_rho(medium: &str) -> Result<f64, String> {
    let tables = mc_tables();
    let m = tables
        .media
        .iter()
        .position(|name| name == medium)
        .ok_or_else(|| format!("medium {medium}"))?;
    Ok(tables.rho[m])
}

/// CSDA range (mm) of a proton in *medium* from MCsquare's stopping powers
/// (0.5 MeV bins from 0, trapezoid on a fine grid like the Python validator).
pub fn csda_range_mm(medium: &str, energy: f64) -> Result<f64, String> {
    let tables = mc_tables();
    let m = tables
        .media
        .iter()
        .position(|name| name == medium)
        .ok_or_else(|| format!("medium {medium}"))?;
    let stop = tables.stop[m]
        .iter()
        .map(|s| s * tables.rho[m] / 1e6)
        .collect::<Vec<_>>(); // MeV/cm
    let energies = &tables.stop_energies;
    let dx = (energy - 0.5) / 1999.0;
    let mut area = 0.0;
    let mut prev = 1.0 / interp_lin(0.5, energies, &stop);
    for i in 1..2000 {
        let e = 0.5 + (energy - 0.5) * i as f64 / 1999.0;
        let inv = 1.0 / interp_lin(e, energies, &stop);
        area += 0.5 * (prev + inv) * dx;
        prev = inv;
    }
    Ok(area * 10.0)
}

fn interp_lin(x: f64, xs: &[f64], ys: &[f64]) -> f64 {
    if x <= xs[0] {
        return ys[0];
    }
    if x >= xs[xs.len() - 1] {
        return ys[ys.len() - 1];
    }
    let i = xs.partition_point(|&v| v < x) - 1;
    let t = (x - xs[i]) / (xs[i + 1] - xs[i]);
    ys[i] + t * (ys[i + 1] - ys[i])
}

fn push_f64(out: &mut Vec<f32>, values: &[f64]) {
    out.extend(values.iter().map(|value| *value as f32));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mc_tables_match_the_baked_shader_offsets() {
        let tables = mc_tables();
        assert_eq!(tables.media.first().map(String::as_str), Some("water"));
        for name in ["pmma", "polystyrene", "aluminum", "copper"] {
            assert!(tables.media.iter().any(|media| media == name), "{name}");
        }
        assert!((tables.floats[0] - 1.0).abs() < 0.05);
        assert_eq!(tables.floats.len(), 562_452);
        assert_eq!(tables.ints.len(), 2_100);
        assert_eq!(tables.ids.len(), tables.media.len());
        assert!((tables.spr[0] - 1.0).abs() < 1e-3);
    }
}
