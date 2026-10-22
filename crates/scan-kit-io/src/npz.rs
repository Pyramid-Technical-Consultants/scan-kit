//! .npy/.npz IO: MCsquare material tables and the validator's golden files.

use std::collections::BTreeMap;
use std::fs;
use std::io::{Cursor, Read, Write};
use std::path::Path;

/// One `.npy` array: dtype string, C-order shape, raw little-endian payload.
#[derive(Clone, Debug)]
pub struct Npy {
    pub descr: String,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
}

impl Npy {
    pub fn itemsize(&self) -> Result<usize, String> {
        itemsize(&self.descr)
    }

    pub fn count(&self) -> Result<usize, String> {
        let n = self.shape.iter().copied().product::<usize>().max(1);
        let size = self.itemsize()?;
        if self.data.len() < n * size {
            return Err(format!("npy {} is short", self.descr));
        }
        Ok(n)
    }

    pub fn f64s(&self) -> Result<Vec<f64>, String> {
        let n = self.count()?;
        if self.descr != "<f8" {
            return Err(format!("expected f64, got {}", self.descr));
        }
        Ok((0..n)
            .map(|i| f64::from_le_bytes(self.data[i * 8..i * 8 + 8].try_into().unwrap()))
            .collect())
    }

    pub fn i64s(&self) -> Result<Vec<i64>, String> {
        let n = self.count()?;
        if self.descr != "<i8" {
            return Err(format!("expected i64, got {}", self.descr));
        }
        Ok((0..n)
            .map(|i| i64::from_le_bytes(self.data[i * 8..i * 8 + 8].try_into().unwrap()))
            .collect())
    }

    pub fn i32s(&self) -> Result<Vec<i32>, String> {
        let n = self.count()?;
        if self.descr != "<i4" {
            return Err(format!("expected i32, got {}", self.descr));
        }
        Ok((0..n)
            .map(|i| i32::from_le_bytes(self.data[i * 4..i * 4 + 4].try_into().unwrap()))
            .collect())
    }

    /// float16 payload widened to f32 (`<f2` dose volumes).
    pub fn f16s(&self) -> Result<Vec<f32>, String> {
        let n = self.count()?;
        if self.descr != "<f2" {
            return Err(format!("expected f16, got {}", self.descr));
        }
        Ok((0..n)
            .map(|i| {
                f16_to_f32(u16::from_le_bytes(
                    self.data[i * 2..i * 2 + 2].try_into().unwrap(),
                ))
            })
            .collect())
    }

    /// NumPy `<U…` fixed-width unicode scalars.
    pub fn strings(&self) -> Result<Vec<String>, String> {
        let n = self.count()?;
        let width = self.itemsize()?;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let bytes = &self.data[i * width..(i + 1) * width];
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

    pub fn scalar_f64(&self) -> Result<f64, String> {
        self.f64s()?
            .into_iter()
            .next()
            .ok_or_else(|| "empty".into())
    }

    pub fn scalar_i64(&self) -> Result<i64, String> {
        self.i64s()?
            .into_iter()
            .next()
            .ok_or_else(|| "empty".into())
    }

    pub fn scalar_string(&self) -> Result<String, String> {
        self.strings()?
            .into_iter()
            .next()
            .ok_or_else(|| "empty".into())
    }
}

/// Every array in an `.npz`, keyed by entry name without the `.npy` suffix.
pub fn read_npz(bytes: &[u8]) -> Result<BTreeMap<String, Npy>, String> {
    let mut archive = zip::ZipArchive::new(Cursor::new(bytes)).map_err(|err| err.to_string())?;
    let mut files = BTreeMap::new();
    for index in 0..archive.len() {
        let mut file = archive.by_index(index).map_err(|err| err.to_string())?;
        let name = file.name().map_err(|err| err.to_string())?;
        let name = name
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

pub fn load_npz(path: &Path) -> Result<BTreeMap<String, Npy>, String> {
    read_npz(&fs::read(path).map_err(|err| format!("{path:?}: {err}"))?)
}

/// Write a deflated `.npz` (`np.savez_compressed` compatible); each pair's
/// string becomes the entry name without its `.npy` suffix.
pub fn write_npz(path: &Path, entries: &[(String, Npy)]) -> Result<(), String> {
    let file = fs::File::create(path).map_err(|err| format!("{path:?}: {err}"))?;
    let mut writer = zip::ZipWriter::new(file);
    let options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);
    for (name, entry) in entries {
        writer
            .start_file(format!("{name}.npy"), options)
            .map_err(|err| err.to_string())?;
        writer
            .write_all(&serialize_npy(entry)?)
            .map_err(|err| err.to_string())?;
    }
    writer.finish().map_err(|err| err.to_string())?;
    Ok(())
}

pub fn npy_f64(shape: &[usize], data: &[f64]) -> Npy {
    Npy {
        descr: "<f8".into(),
        shape: shape.to_vec(),
        data: data.iter().flat_map(|v| v.to_le_bytes()).collect(),
    }
}

pub fn npy_i32(shape: &[usize], data: &[i32]) -> Npy {
    Npy {
        descr: "<i4".into(),
        shape: shape.to_vec(),
        data: data.iter().flat_map(|v| v.to_le_bytes()).collect(),
    }
}

pub fn npy_i64(shape: &[usize], data: &[i64]) -> Npy {
    Npy {
        descr: "<i8".into(),
        shape: shape.to_vec(),
        data: data.iter().flat_map(|v| v.to_le_bytes()).collect(),
    }
}

pub fn npy_f16(shape: &[usize], data: &[f32]) -> Npy {
    Npy {
        descr: "<f2".into(),
        shape: shape.to_vec(),
        data: data
            .iter()
            .flat_map(|v| f32_to_f16(*v).to_le_bytes())
            .collect(),
    }
}

/// `<U{max}` fixed-width unicode, one scalar per string.
pub fn npy_unicode(data: &[String]) -> Npy {
    let width = data.iter().map(|s| s.chars().count()).max().unwrap_or(1);
    let mut bytes = Vec::with_capacity(data.len() * width * 4);
    for s in data {
        let mut written = 0;
        for ch in s.chars() {
            bytes.extend_from_slice(&(ch as u32).to_le_bytes());
            written += 1;
        }
        for _ in written..width {
            bytes.extend_from_slice(&0u32.to_le_bytes());
        }
    }
    Npy {
        descr: format!("<U{width}"),
        shape: vec![data.len()],
        data: bytes,
    }
}

fn serialize_npy(entry: &Npy) -> Result<Vec<u8>, String> {
    let shape = match entry.shape.len() {
        0 => "()".to_string(),
        1 => format!("({},)", entry.shape[0]),
        _ => format!(
            "({})",
            entry
                .shape
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ),
    };
    let header = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': {}, }}",
        entry.descr, shape
    );
    // Version 1.0 header: magic(6) + version(2) + u16 len + header, padded to 64.
    let pad = (64 - (10 + header.len() + 1) % 64) % 64;
    let header = format!("{header}{:pad$}\n", "", pad = pad);
    let mut out = Vec::with_capacity(10 + header.len() + entry.data.len());
    out.extend_from_slice(b"\x93NUMPY\x01\x00");
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&entry.data);
    Ok(out)
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
        "<f2" | "<i2" | "<u2" => Ok(2),
        "|u1" | "|i1" | "<u1" | "<i1" => Ok(1),
        other => Err(format!("npy dtype {other}")),
    }
}

/// IEEE 754 half → single, quiet NaNs preserved.
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        // Subnormal: normalize by hand.
        if frac == 0 {
            sign
        } else {
            let mut e = 0u32;
            let mut f = frac;
            while f & 0x400 == 0 {
                f <<= 1;
                e += 1;
            }
            f &= 0x3ff;
            sign | ((113 - e) << 23) | (f << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (frac << 13)
    } else {
        sign | ((exp + 112) << 23) | (frac << 13)
    };
    f32::from_bits(bits)
}

/// Single → IEEE 754 half, round-to-nearest-even, saturating to ±inf.
pub fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let frac = bits & 0x7f_ffff;
    if exp == 0xff {
        return sign | if frac == 0 { 0x7c00 } else { 0x7e00 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        // Subnormal half.
        let m = frac | 0x80_0000;
        let shift = (14 - e) as u32;
        let mut h = m >> shift;
        if (m >> (shift - 1)) & 1 == 1 && (h & 1 == 1 || (m & ((1 << (shift - 1)) - 1)) != 0) {
            h += 1;
        }
        return sign | h as u16;
    }
    let mut h = ((e as u32) << 10) | (frac >> 13);
    if frac & 0x1fff > 0x1000 || (frac & 0x3fff) == 0x3000 {
        h += 1;
    }
    sign | h as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_round_trip() {
        for v in [0.0f32, 1.0, -1.0, 0.5, 65_504.0, 6e-5, 1e-8, f32::INFINITY] {
            let h = f32_to_f16(v);
            let back = f16_to_f32(h);
            if v.is_finite() && v.abs() >= 6e-5 && v.abs() < 65_504.0 {
                assert!((back - v).abs() <= v.abs() * 0.001 + 1e-7, "{v} -> {back}");
            } else {
                assert_eq!(back.is_infinite(), v.is_infinite(), "{v}");
            }
        }
        assert!(f16_to_f32(f32_to_f16(f32::NAN)).is_nan());
    }

    #[test]
    fn npz_round_trip() {
        let dir = std::env::temp_dir().join(format!("npz_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("round.npz");
        let entries = vec![
            (
                "f64s".to_string(),
                npy_f64(&[2, 2], &[1.0, 2.5, -3.0, 4.25]),
            ),
            ("i64s".to_string(), npy_i64(&[3], &[7, -8, 9])),
            ("f16s".to_string(), npy_f16(&[2], &[0.5, 1.5])),
            ("text".to_string(), npy_unicode(&["water".to_string()])),
        ];
        write_npz(&path, &entries).unwrap();
        let files = load_npz(&path).unwrap();
        assert_eq!(files["f64s"].f64s().unwrap(), vec![1.0, 2.5, -3.0, 4.25]);
        assert_eq!(files["f64s"].shape, vec![2, 2]);
        assert_eq!(files["i64s"].i64s().unwrap(), vec![7, -8, 9]);
        assert_eq!(files["f16s"].f16s().unwrap(), vec![0.5, 1.5]);
        assert_eq!(files["text"].strings().unwrap(), vec!["water"]);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// What `np.savez_compressed` wrote, `load_npz` must read back.
    #[test]
    fn reads_a_python_npz() {
        let goldens = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../validation/goldens");
        if !goldens.is_dir() {
            return;
        }
        let path = goldens.join("aluminum_100_s4.npz");
        if !path.is_file() {
            return;
        }
        let files = load_npz(&path).unwrap();
        assert_eq!(files["medium"].scalar_string().unwrap(), "aluminum");
        assert_eq!(files["dose"].shape.len(), 3);
        assert!(files["dose"].f16s().unwrap().iter().all(|v| v.is_finite()));
        assert_eq!(files["crop"].shape, vec![2, 3]);
    }
}
