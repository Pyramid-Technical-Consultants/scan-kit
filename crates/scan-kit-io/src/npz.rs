//! .npy/.npz IO: MCsquare material tables and the validator's golden files.

use std::collections::BTreeMap;
use std::fs;
use std::io::{Cursor, Write};
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
        let n = self
            .shape
            .iter()
            .copied()
            .try_fold(1usize, |acc, dim| acc.checked_mul(dim))
            .ok_or("npy shape overflows")?;
        let need = n
            .checked_mul(self.itemsize()?)
            .ok_or("npy shape overflows")?;
        if self.data.len() < need {
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
        if !self.descr.starts_with("<U") {
            return Err(format!("npz strings need <U, got {}", self.descr));
        }
        if width == 0 {
            return Err(format!("npz strings need <U1 or wider, got {}", self.descr));
        }
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let bytes = &self.data[i * width..(i + 1) * width];
            let mut chars = Vec::new();
            for chunk in bytes.chunks(4) {
                let unit = u32::from_le_bytes(
                    chunk
                        .try_into()
                        .map_err(|_| "npz string width is not a u32 multiple")?,
                );
                if unit == 0 {
                    break;
                }
                // An invalid code unit (surrogate, >0x10FFFF) means a corrupt
                // payload — mark it rather than silently shortening the name.
                chars.push(char::from_u32(unit).unwrap_or(char::REPLACEMENT_CHARACTER));
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
    // Per-member caps still let a crafted archive request N × 4 GiB —
    // bound the decompressed total like the session readers do.
    let mut total = 0u64;
    for index in 0..archive.len() {
        let mut file = archive.by_index(index).map_err(|err| err.to_string())?;
        let name = file.name().map_err(|err| err.to_string())?;
        // Metadata-fork members (__MACOSX/x.npy) flatten to the same basename
        // as the real array — skip them before they collide or win.
        if name.split(['/', '\\']).any(|part| part == "__MACOSX") {
            continue;
        }
        let base = name.rsplit(['/', '\\']).next().unwrap_or("");
        // np.load only exposes .npy members — a stray note.txt or __MACOSX
        // entry must not fail the whole archive.
        let Some(name) = base.strip_suffix(".npy") else {
            continue;
        };
        let name = name.to_owned();
        if name.is_empty() {
            continue;
        }
        let raw =
            crate::discover::read_capped(&mut file, &name, crate::discover::MAX_MEMBER_BYTES)?;
        total = total.saturating_add(raw.len() as u64);
        if total > crate::discover::MAX_MEMBER_BYTES {
            return Err("the npz archive is too large".into());
        }
        if files.insert(name.clone(), parse_npy(&raw)?).is_some() {
            return Err(format!("duplicate npz member {name}"));
        }
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
    // Versions 1-3 are the only published layouts (1 uses a u16 header, 2
    // and 3 share the u32 header) — anything newer must not be assumed.
    if !matches!(bytes[6], 1..=3) {
        return Err(format!("unsupported npy version {}", bytes[6]));
    }
    let (header_len, header_at) = if bytes[6] == 1 {
        (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10)
    } else {
        let header_len = bytes
            .get(8..12)
            .map(|raw| u32::from_le_bytes(raw.try_into().unwrap()))
            .ok_or("npy header")?;
        (header_len as usize, 12)
    };
    let header = std::str::from_utf8(
        bytes
            .get(header_at..header_at + header_len)
            .ok_or("npy header")?,
    )
    .map_err(|err| err.to_string())?;
    // Anchor on the key, not one exact serialization — a non-NumPy writer
    // may emit `"fortran_order": True` or `'fortran_order' : True`, and a
    // Fortran-order array read as C-order is a silent transposition.
    if let Some((_, rest)) = header.split_once("fortran_order") {
        let value = rest.trim_start_matches(['\'', '"', ' ', ':']);
        if value.starts_with("True") || value.starts_with("true") {
            return Err("fortran npy".into());
        }
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
            .ok()
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| format!("npy itemsize {descr}"));
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

    #[test]
    fn an_overflowing_shape_errors_instead_of_panicking() {
        let npy = Npy {
            descr: "<f8".into(),
            shape: vec![usize::MAX / 8 + 2],
            data: vec![0; 8],
        };
        assert!(npy.count().is_err());
        assert!(npy.f64s().is_err());
        let npy = Npy {
            descr: "<f8".into(),
            shape: vec![usize::MAX / 8 + 1],
            data: vec![0; 8],
        };
        // n * size wraps past the short-data check — must not reach the reads.
        assert!(npy.f64s().is_err());
    }

    #[test]
    fn a_truncated_v2_header_errors_instead_of_panicking() {
        // 10 and 11 bytes reach the v2 u32 header-length read short.
        for len in [10usize, 11] {
            let mut bytes = b"\x93NUMPY\x02\x00".to_vec();
            bytes.extend_from_slice(&[0, 0, 0, 0, 0][..len - 8]);
            assert!(parse_npy(&bytes).is_err(), "len {len}");
        }
    }

    /// What `np.savez_compressed` wrote, `load_npz` must read back.
    #[test]
    fn reads_a_python_npz() {
        // The golden is committed — a missing fixture must fail loudly, not
        // quietly report a pass.
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../validation/goldens/aluminum_100_s4.npz");
        let files = load_npz(&path).unwrap();
        assert_eq!(files["medium"].scalar_string().unwrap(), "aluminum");
        assert_eq!(files["dose"].shape.len(), 3);
        assert!(files["dose"].f16s().unwrap().iter().all(|v| v.is_finite()));
        assert_eq!(files["crop"].shape, vec![2, 3]);
    }

    #[test]
    fn a_zero_width_unicode_array_errors_instead_of_allocating() {
        let npy = Npy {
            descr: "<U0".into(),
            shape: vec![4_000_000_000],
            data: Vec::new(),
        };
        assert!(npy.strings().is_err());
    }

    #[test]
    fn stray_zip_members_are_not_arrays() {
        let mut buf = Vec::new();
        {
            let mut writer = zip::ZipWriter::new(std::io::Cursor::new(&mut buf));
            let options = zip::write::SimpleFileOptions::default();
            writer.start_file("note.txt", options).unwrap();
            std::io::Write::write_all(&mut writer, b"not an array").unwrap();
            writer.start_file("ok.npy", options).unwrap();
            std::io::Write::write_all(&mut writer, &serialize_npy(&npy_f64(&[1], &[1.0])).unwrap())
                .unwrap();
            writer.finish().unwrap();
        }
        let files = read_npz(&buf).unwrap();
        assert_eq!(files["ok"].f64s().unwrap(), vec![1.0]);
        assert!(!files.contains_key("note.txt"));
    }

    #[test]
    fn a_nonstandard_fortran_header_still_errors() {
        let make = |header: &str| {
            let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
            bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
            bytes.extend_from_slice(header.as_bytes());
            bytes.extend_from_slice(&[0u8; 16]);
            bytes
        };
        assert!(parse_npy(&make(
            "{'descr': '<f8', 'fortran_order':True, 'shape': (2, 2), }"
        ))
        .is_err());
        assert!(parse_npy(&make(
            "{\"descr\": \"<f8\", \"fortran_order\": true, \"shape\": (2, 2), }"
        ))
        .is_err());
        assert!(parse_npy(&make(
            "{'descr': '<f8', 'fortran_order': False, 'shape': (2, 2), }"
        ))
        .is_ok());
    }

    #[test]
    fn an_unknown_npy_version_is_rejected() {
        // Versions past 3 have no published layout — assuming the v2 header
        // shape on a future file reads its header bytes as data.
        let mut bytes = b"\x93NUMPY".to_vec();
        bytes.extend_from_slice(&[4, 0]);
        bytes.extend_from_slice(&0u32.to_le_bytes());
        let err = parse_npy(&bytes).unwrap_err();
        assert!(err.contains("version"), "{err}");
    }
}
