//! MCsquare material tables. The transport shader's baked offsets assume this packing.

use std::collections::BTreeMap;
use std::io::{Cursor, Read};
use std::sync::OnceLock;

const NPZ: &[u8] = include_bytes!("../../../scan_kit/assets/mc_materials.npz");

#[derive(Clone, Debug)]
pub struct McTables {
    pub floats: Vec<f32>,
    pub ints: Vec<i32>,
    pub media: Vec<String>,
    /// MCsquare material labels, one per medium, in table order.
    pub ids: Vec<i32>,
    /// Mass stopping power over water at 100 MeV.
    pub spr: Vec<f32>,
}

struct Npy {
    descr: String,
    shape: Vec<usize>,
    data: Vec<u8>,
}

pub fn mc_tables() -> &'static McTables {
    static TABLES: OnceLock<McTables> = OnceLock::new();
    TABLES.get_or_init(|| pack().expect("mc material tables"))
}

fn pack() -> Result<McTables, String> {
    let files = read_npz(NPZ)?;
    let media = strings(files.get("media").ok_or("media")?)?;
    let props = f64s(files.get("m_props").ok_or("m_props")?)?;
    let frac = f64s(files.get("m_frac").ok_or("m_frac")?)?;
    let stop = f64s(files.get("m_stop").ok_or("m_stop")?)?;
    let nuc = f64s(files.get("m_nuc").ok_or("m_nuc")?)?;
    let comp = i32s(files.get("m_comp").ok_or("m_comp")?)?;
    let kind = i32s(files.get("m_type").ok_or("m_type")?)?;
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
        push_f64(&mut floats, &f64s(files.get(key).ok_or(key)?)?);
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
    let e_type = i32s(files.get("e_type").ok_or("e_type")?)?;
    let el_start = i32s(files.get("el_start").ok_or("el_start")?)?;
    let in_start = i32s(files.get("in_start").ok_or("in_start")?)?;
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
    ints.extend_from_slice(&i32s(files.get("in_rows_start").ok_or("in_rows_start")?)?);
    ints.extend_from_slice(&i32s(files.get("in_rows_n").ok_or("in_rows_n")?)?);
    let ids = i32s(files.get("mcsquare_ids").ok_or("mcsquare_ids")?)?;
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
        spr,
    })
}

fn push_f64(out: &mut Vec<f32>, values: &[f64]) {
    out.extend(values.iter().map(|value| *value as f32));
}

fn read_npz(bytes: &[u8]) -> Result<BTreeMap<String, Npy>, String> {
    let mut archive = zip::ZipArchive::new(Cursor::new(bytes)).map_err(|err| err.to_string())?;
    let mut files = BTreeMap::new();
    for index in 0..archive.len() {
        let mut file = archive.by_index(index).map_err(|err| err.to_string())?;
        let name = file
            .name()
            .rsplit(['/', '\\'])
            .next()
            .unwrap_or("")
            .trim_end_matches(".npy")
            .to_owned();
        if name.is_empty() {
            continue;
        }
        let mut raw = Vec::new();
        file.read_to_end(&mut raw).map_err(|err| err.to_string())?;
        files.insert(name, parse_npy(&raw)?);
    }
    Ok(files)
}

fn parse_npy(bytes: &[u8]) -> Result<Npy, String> {
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err("not an npy array".into());
    }
    let (header_len, header_at) = if bytes[6] == 1 {
        (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10)
    } else {
        (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12,
        )
    };
    let header = std::str::from_utf8(
        bytes
            .get(header_at..header_at + header_len)
            .ok_or("npy header")?,
    )
    .map_err(|err| err.to_string())?;
    if header.contains("fortran_order': True") {
        return Err("fortran npy".into());
    }
    let descr = quoted(header, "descr")?;
    let shape = shape_of(&quoted_or_tuple(header, "shape")?)?;
    let data = bytes[header_at + header_len..].to_vec();
    Ok(Npy { descr, shape, data })
}

fn quoted(header: &str, key: &str) -> Result<String, String> {
    let needle = format!("'{key}':");
    let rest = header
        .split_once(&needle)
        .ok_or_else(|| format!("npy {key}"))?
        .1
        .trim_start();
    let rest = rest
        .strip_prefix('\'')
        .ok_or_else(|| format!("npy {key}"))?;
    Ok(rest.split('\'').next().unwrap_or("").to_owned())
}

fn quoted_or_tuple(header: &str, key: &str) -> Result<String, String> {
    let needle = format!("'{key}':");
    let rest = header
        .split_once(&needle)
        .ok_or_else(|| format!("npy {key}"))?
        .1
        .trim_start();
    if let Some(inner) = rest.strip_prefix('(') {
        let body = inner.split(')').next().unwrap_or("");
        return Ok(format!("({body})"));
    }
    quoted(header, key)
}

fn shape_of(text: &str) -> Result<Vec<usize>, String> {
    let body = text.trim().trim_matches(|c| c == '(' || c == ')');
    if body.trim().is_empty() {
        return Ok(vec![]);
    }
    body.split(',')
        .filter(|part| !part.trim().is_empty())
        .map(|part| part.trim().parse().map_err(|_| format!("npy shape {text}")))
        .collect()
}

fn itemsize(descr: &str) -> Result<usize, String> {
    if let Some(width) = descr.strip_prefix("<U") {
        return width
            .parse::<usize>()
            .map(|n| n * 4)
            .map_err(|err| err.to_string());
    }
    match descr {
        "<f8" | "<i8" => Ok(8),
        "<f4" | "<i4" | "<u4" => Ok(4),
        other => Err(format!("npy dtype {other}")),
    }
}

fn count(array: &Npy) -> Result<usize, String> {
    let n = array.shape.iter().copied().product::<usize>().max(1);
    let size = itemsize(&array.descr)?;
    if array.data.len() < n * size {
        return Err(format!("npy {} is short", array.descr));
    }
    Ok(n)
}

fn f64s(array: &Npy) -> Result<Vec<f64>, String> {
    let n = count(array)?;
    if array.descr != "<f8" {
        return Err(format!("expected f64, got {}", array.descr));
    }
    Ok((0..n)
        .map(|i| f64::from_le_bytes(array.data[i * 8..i * 8 + 8].try_into().unwrap()))
        .collect())
}

fn i32s(array: &Npy) -> Result<Vec<i32>, String> {
    let n = count(array)?;
    if array.descr != "<i4" {
        return Err(format!("expected i32, got {}", array.descr));
    }
    Ok((0..n)
        .map(|i| i32::from_le_bytes(array.data[i * 4..i * 4 + 4].try_into().unwrap()))
        .collect())
}

fn strings(array: &Npy) -> Result<Vec<String>, String> {
    let n = count(array)?;
    let width = itemsize(&array.descr)?;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let bytes = &array.data[i * width..(i + 1) * width];
        let mut chars = Vec::new();
        for chunk in bytes.chunks(4) {
            let unit = u32::from_le_bytes(chunk.try_into().unwrap());
            if unit == 0 {
                break;
            }
            if let Some(ch) = char::from_u32(unit) {
                chars.push(ch);
            }
        }
        out.push(chars.into_iter().collect());
    }
    Ok(out)
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
