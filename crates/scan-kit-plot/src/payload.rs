//! One binary payload per opened view.
//!
//! Layout: `u32` little-endian header length, the JSON [`PlotHeader`], then the
//! line, point, and quad instance bytes in GPU layout, then each heatmap's
//! RGBA pixels. The shell reads the header for its controls. The wasm plot uploads
//! the rest without parsing series again.

use scan_kit_core::{Control, DataTable, Panel, PlotScene, VolumeMark};
use serde::{Deserialize, Serialize};

use crate::render::{
    build_marks, build_marks_without_polylines, decode_lines, decode_points, decode_quads,
    encode_lines, encode_points, encode_quads, header_panels, line_stamp, Marks, PanelBatch, Plot,
    LINE_STRIDE, POINT_STRIDE, QUAD_STRIDE,
};

/// `apps/desktop/src/plot-header.ts` mirrors the public fields. Rename both together.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlotHeader {
    pub title: String,
    pub controls: Vec<Control>,
    pub table: Option<DataTable>,
    /// Panel frames. Series are empty. They travel as marks.
    pub panels: Vec<Panel>,
    pub columns: u32,
    pub column_weights: Vec<f32>,
    #[serde(default)]
    pub row_weights: Vec<f32>,
    /// One left/right pair per panel when the grid is two columns.
    #[serde(default)]
    pub row_splits: Vec<f32>,
    /// Panels at the end of `panels` that fill the right column.
    #[serde(default)]
    pub side: u32,
    pub background: [f32; 4],
    pub foreground: [f32; 4],
    /// `partial` while a task is still loading. `final` on the last payload.
    #[serde(default = "final_quality")]
    pub quality: String,
    /// Polyline identity. The shell sends it back so the next picture can omit
    /// trace lines the plot already has.
    #[serde(default)]
    line_token: String,
    /// When set, `lines` counts only the records after `line_prefix`.
    #[serde(default)]
    reuse_lines: bool,
    /// Polyline records already stored from the payload that issued `line_token`.
    #[serde(default)]
    line_prefix: u32,
    batches: Vec<PanelBatch>,
    heatmap_size: Vec<(u32, u32)>,
    lines: u32,
    points: u32,
    quads: u32,
    #[serde(default)]
    volume_shape: [u32; 3],
    #[serde(default)]
    volume_origin: [f32; 3],
    #[serde(default)]
    volume_voxel: f32,
    #[serde(default)]
    volume_ramp: u8,
    #[serde(default)]
    volume_lo: f32,
    #[serde(default)]
    volume_hi: f32,
    /// Data window. Zero means the plot measures the cube it was given.
    #[serde(default)]
    volume_base_lo: f32,
    #[serde(default)]
    volume_base_hi: f32,
    #[serde(default)]
    volume_gain: f32,
    #[serde(default)]
    volume_opacity: f32,
    #[serde(default)]
    volume_mode: u8,
    #[serde(default)]
    volume_filter: u8,
    #[serde(default)]
    volume_gantry: f32,
    #[serde(default)]
    volume_unit: String,
    #[serde(default)]
    volume_phantom: bool,
    #[serde(default)]
    volume_field: [f32; 6],
    #[serde(default)]
    volume_ct: bool,
    #[serde(default)]
    volume_labels: bool,
    #[serde(default)]
    volume_sessions: u32,
    #[serde(default)]
    volume_plans: u32,
    #[serde(default)]
    volume_companions: u32,
    #[serde(default)]
    volume_session_focus: u32,
    #[serde(default)]
    volume_colors: Vec<[f32; 4]>,
}

fn final_quality() -> String {
    "final".into()
}

pub fn encode_plot(
    scene: &PlotScene,
    background: [f32; 4],
    foreground: [f32; 4],
) -> Result<Vec<u8>, String> {
    encode_plot_quality(scene, background, foreground, "final")
}

pub fn encode_plot_quality(
    scene: &PlotScene,
    background: [f32; 4],
    foreground: [f32; 4],
    quality: &str,
) -> Result<Vec<u8>, String> {
    encode_plot_reusing(scene, background, foreground, quality, None)
}

/// `held_lines` is the `line_token` the plot already uploaded. When the polyline
/// samples match, the byte section keeps the guide tail and omits the traces.
pub fn encode_plot_reusing(
    scene: &PlotScene,
    background: [f32; 4],
    foreground: [f32; 4],
    quality: &str,
    held_lines: Option<&str>,
) -> Result<Vec<u8>, String> {
    let stamp = line_stamp(&scene.panels);
    let held = held_lines.and_then(|text| text.parse::<u64>().ok());
    let reuse = stamp.reusable && held == Some(stamp.token);
    let marks = if reuse {
        build_marks_without_polylines(&scene.panels)
    } else {
        build_marks(&scene.panels)
    };
    let header = PlotHeader {
        title: scene.title.clone(),
        controls: scene.controls.clone(),
        table: scene.table.clone(),
        panels: header_panels(&scene.panels),
        columns: scene.columns,
        column_weights: scene.column_weights.clone(),
        row_weights: scene.row_weights.clone(),
        row_splits: scene.row_splits.clone(),
        side: scene.side,
        background,
        foreground,
        quality: quality.into(),
        line_token: stamp.token.to_string(),
        reuse_lines: reuse,
        line_prefix: stamp.prefix,
        batches: marks.panels.clone(),
        heatmap_size: marks.heatmap_size.clone(),
        lines: marks.lines.len() as u32,
        points: marks.points.len() as u32,
        quads: marks.quads.len() as u32,
        volume_shape: scene.volume.shape,
        volume_origin: scene.volume.origin,
        volume_voxel: scene.volume.voxel,
        volume_ramp: scene.volume.ramp,
        volume_lo: scene.volume.lo,
        volume_hi: scene.volume.hi,
        volume_base_lo: scene.volume.base_lo,
        volume_base_hi: scene.volume.base_hi,
        volume_gain: scene.volume.gain,
        volume_opacity: scene.volume.opacity,
        volume_mode: scene.volume.mode,
        volume_filter: scene.volume.filter,
        volume_gantry: scene.volume.gantry,
        volume_unit: scene.volume.unit.clone(),
        volume_phantom: scene.volume.show_phantom,
        volume_field: scene.volume.field,
        volume_ct: !scene.volume.ct.is_empty(),
        volume_labels: !scene.volume.labels.is_empty(),
        volume_sessions: scene.volume.sessions.len() as u32,
        volume_plans: scene.volume.plans.len() as u32,
        volume_companions: scene.volume.companions.len() as u32,
        volume_session_focus: scene.volume.session_focus,
        volume_colors: scene.volume.session_colors.clone(),
    };
    let json = serde_json::to_vec(&header).map_err(|err| err.to_string())?;
    let mut out = Vec::with_capacity(
        4 + json.len()
            + marks.lines.len() * LINE_STRIDE as usize
            + marks.points.len() * POINT_STRIDE as usize
            + marks.quads.len() * QUAD_STRIDE as usize
            + marks.heatmaps.iter().map(Vec::len).sum::<usize>(),
    );
    out.extend_from_slice(&(json.len() as u32).to_le_bytes());
    out.extend_from_slice(&json);
    out.extend_from_slice(&encode_lines(&marks.lines));
    out.extend_from_slice(&encode_points(&marks.points));
    out.extend_from_slice(&encode_quads(&marks.quads));
    for pixels in &marks.heatmaps {
        out.extend_from_slice(pixels);
    }
    let voxels = scene.volume.shape[0] as usize
        * scene.volume.shape[1] as usize
        * scene.volume.shape[2] as usize;
    if voxels > 0 && scene.volume.values.len() >= voxels {
        write_cube(&mut out, &scene.volume.values[..voxels]);
        if !scene.volume.ct.is_empty() {
            write_f32s(
                &mut out,
                &scene.volume.ct[..voxels.min(scene.volume.ct.len())],
            );
        }
        if !scene.volume.labels.is_empty() {
            out.extend_from_slice(&scene.volume.labels[..voxels.min(scene.volume.labels.len())]);
        }
        let shape = [
            scene.volume.shape[0] as usize,
            scene.volume.shape[1] as usize,
            scene.volume.shape[2] as usize,
        ];
        // The view would otherwise scan this cube again on the UI thread.
        write_scan(
            &mut out,
            &scan_kit_core::scan_grid(
                &scene.volume.values[..voxels],
                shape,
                scene.volume.voxel.max(1e-3),
            ),
        );
    }
    for session in scene
        .volume
        .sessions
        .iter()
        .chain(&scene.volume.plans)
        .chain(&scene.volume.companions)
    {
        write_session(&mut out, session);
    }
    Ok(out)
}

fn read_sessions(
    reader: &mut Reader<'_>,
    count: u32,
) -> Result<Vec<scan_kit_core::SessionDose>, String> {
    let mut doses = Vec::new();
    for _ in 0..count {
        let shape = [
            u32::from_le_bytes(reader.take(4)?.try_into().unwrap()),
            u32::from_le_bytes(reader.take(4)?.try_into().unwrap()),
            u32::from_le_bytes(reader.take(4)?.try_into().unwrap()),
        ];
        let origin = [
            f32::from_le_bytes(reader.take(4)?.try_into().unwrap()),
            f32::from_le_bytes(reader.take(4)?.try_into().unwrap()),
            f32::from_le_bytes(reader.take(4)?.try_into().unwrap()),
        ];
        let voxel = f32::from_le_bytes(reader.take(4)?.try_into().unwrap());
        let n = u32::from_le_bytes(reader.take(4)?.try_into().unwrap()) as usize;
        let values = read_cube(reader, n)?;
        doses.push(scan_kit_core::SessionDose {
            values,
            shape,
            origin,
            voxel,
        });
    }
    Ok(doses)
}

fn write_session(out: &mut Vec<u8>, session: &scan_kit_core::SessionDose) {
    for value in session.shape {
        out.extend_from_slice(&value.to_le_bytes());
    }
    for value in session.origin {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out.extend_from_slice(&session.voxel.to_le_bytes());
    let count = session.values.len() as u32;
    out.extend_from_slice(&count.to_le_bytes());
    write_cube(out, &session.values);
}

/// Shortest `+0.0` run that ends a literal block. A block header is 8 bytes.
const ZERO_RUN: usize = 8;

/// A dose cube is a third zeros or more outside the field. It is stored as
/// blocks of `u32` zero count, `u32` literal count, then the literal floats.
/// Only `+0.0` bits count as zero, so every value round-trips exactly.
fn write_cube(out: &mut Vec<u8>, values: &[f32]) {
    let mut at = 0;
    while at < values.len() {
        let zeros = values[at..]
            .iter()
            .take_while(|value| value.to_bits() == 0)
            .count();
        let start = at + zeros;
        let mut end = values.len();
        let mut run = 0;
        for (index, value) in values[start..].iter().enumerate() {
            run = if value.to_bits() == 0 { run + 1 } else { 0 };
            if run == ZERO_RUN {
                end = start + index + 1 - ZERO_RUN;
                break;
            }
        }
        out.extend_from_slice(&(zeros as u32).to_le_bytes());
        out.extend_from_slice(&((end - start) as u32).to_le_bytes());
        write_f32s(out, &values[start..end]);
        at = end;
    }
}

fn read_cube(reader: &mut Reader<'_>, count: usize) -> Result<Vec<f32>, String> {
    // Walk the blocks first, so a bad count fails before the cube is allocated.
    let mut walk = Reader {
        bytes: reader.bytes,
        at: reader.at,
    };
    let mut blocks = Vec::new();
    let mut at = 0;
    while at < count {
        let zeros = read_u32(&mut walk)? as usize;
        let literal = read_u32(&mut walk)? as usize;
        let start = at.saturating_add(zeros);
        let end = start.saturating_add(literal);
        if end > count || end == at {
            return Err("plot payload cube is malformed".into());
        }
        blocks.push((start, end, walk.take(literal * 4)?));
        at = end;
    }
    let mut values = vec![0.0f32; count];
    for (start, end, raw) in blocks {
        copy_le(raw, &mut values[start..end]);
    }
    reader.at = walk.at;
    Ok(values)
}

fn read_f32s(reader: &mut Reader<'_>, count: usize) -> Result<Vec<f32>, String> {
    let bytes = count.checked_mul(4).ok_or("plot payload is truncated")?;
    let mut values = vec![0.0f32; count];
    copy_le(reader.take(bytes)?, &mut values);
    Ok(values)
}

/// `raw` is `values.len() * 4` little-endian bytes.
fn copy_le(raw: &[u8], values: &mut [f32]) {
    assert_eq!(raw.len(), values.len() * 4, "f32 byte length");
    #[cfg(target_endian = "little")]
    // SAFETY: the lengths match above. `f32` is 4 bytes with no padding, and
    // this host is little-endian, which is the order `write_f32s` stored. The
    // JSON header can leave `raw` unaligned, so the bytes are copied into the
    // aligned `values` instead of being cast.
    unsafe {
        std::ptr::copy_nonoverlapping(raw.as_ptr(), values.as_mut_ptr().cast::<u8>(), raw.len());
    }
    #[cfg(target_endian = "big")]
    for (slot, bytes) in values.iter_mut().zip(raw.chunks_exact(4)) {
        *slot = f32::from_le_bytes(bytes.try_into().unwrap());
    }
}

fn write_scan(out: &mut Vec<u8>, scan: &scan_kit_core::VolumeScan) {
    for value in scan.brick_shape {
        out.extend_from_slice(&(value as u32).to_le_bytes());
    }
    for value in scan.peak_at {
        out.extend_from_slice(&(value as u32).to_le_bytes());
    }
    for value in [scan.line_scale, scan.peak, scan.abs_peak] {
        out.extend_from_slice(&value.to_le_bytes());
    }
    for value in scan.integral_peaks {
        out.extend_from_slice(&value.to_le_bytes());
    }
    write_f32s(out, &scan.bricks);
    for plane in &scan.integrals {
        write_f32s(out, plane);
    }
}

fn read_scan(
    reader: &mut Reader<'_>,
    shape: [usize; 3],
) -> Result<scan_kit_core::VolumeScan, String> {
    let brick_shape = [
        read_u32(reader)? as usize,
        read_u32(reader)? as usize,
        read_u32(reader)? as usize,
    ];
    let peak_at = [
        read_u32(reader)? as usize,
        read_u32(reader)? as usize,
        read_u32(reader)? as usize,
    ];
    let line_scale = read_f32(reader)?;
    let peak = read_f32(reader)?;
    let abs_peak = read_f32(reader)?;
    let integral_peaks = [read_f32(reader)?, read_f32(reader)?, read_f32(reader)?];
    let [nx, ny, nz] = shape;
    let bricks = brick_shape[0]
        .saturating_mul(brick_shape[1])
        .saturating_mul(brick_shape[2]);
    Ok(scan_kit_core::VolumeScan {
        bricks: read_f32s(reader, bricks)?,
        brick_shape,
        line_scale,
        peak,
        abs_peak,
        peak_at,
        integrals: [
            read_f32s(reader, nx.saturating_mul(ny))?,
            read_f32s(reader, nx.saturating_mul(nz))?,
            read_f32s(reader, ny.saturating_mul(nz))?,
        ],
        integral_peaks,
    })
}

fn read_u32(reader: &mut Reader<'_>) -> Result<u32, String> {
    Ok(u32::from_le_bytes(reader.take(4)?.try_into().unwrap()))
}

fn read_f32(reader: &mut Reader<'_>) -> Result<f32, String> {
    Ok(f32::from_le_bytes(reader.take(4)?.try_into().unwrap()))
}

fn write_f32s(out: &mut Vec<u8>, values: &[f32]) {
    #[cfg(target_endian = "little")]
    {
        // SAFETY: `f32` is 4 bytes with no padding, and this host stores them
        // little-endian, which is the byte order `to_le_bytes` writes.
        let bytes = unsafe {
            std::slice::from_raw_parts(
                values.as_ptr().cast::<u8>(),
                values.len().checked_mul(4).expect("f32 byte length"),
            )
        };
        out.extend_from_slice(bytes);
    }
    #[cfg(target_endian = "big")]
    {
        for value in values {
            out.extend_from_slice(&value.to_le_bytes());
        }
    }
}

pub fn plot_header(bytes: &[u8]) -> Result<PlotHeader, String> {
    decode_plot(bytes).map(|(header, _, _, _, _)| header)
}

/// GPU bytes already packed by [`encode_plot_quality`]. Uploading these skips a
/// second pass through the mark structs.
#[derive(Clone)]
pub(crate) struct EncodedMarks {
    pub lines: Vec<u8>,
    pub points: Vec<u8>,
    pub quads: Vec<u8>,
}

pub(crate) fn decode_plot(
    bytes: &[u8],
) -> Result<
    (
        PlotHeader,
        Marks,
        EncodedMarks,
        VolumeMark,
        Option<scan_kit_core::VolumeScan>,
    ),
    String,
> {
    let mut reader = Reader { bytes, at: 0 };
    let json_len = u32::from_le_bytes(reader.take(4)?.try_into().unwrap()) as usize;
    let header: PlotHeader =
        serde_json::from_slice(reader.take(json_len)?).map_err(|err| err.to_string())?;
    let line_bytes = reader
        .take(header.lines as usize * LINE_STRIDE as usize)?
        .to_vec();
    let point_bytes = reader
        .take(header.points as usize * POINT_STRIDE as usize)?
        .to_vec();
    let quad_bytes = reader
        .take(header.quads as usize * QUAD_STRIDE as usize)?
        .to_vec();
    let lines = decode_lines(&line_bytes);
    let points = decode_points(&point_bytes);
    let quads = decode_quads(&quad_bytes);
    let heatmaps = header
        .heatmap_size
        .iter()
        .map(|(cols, rows)| reader.take((cols * rows * 4) as usize).map(<[u8]>::to_vec))
        .collect::<Result<Vec<_>, _>>()?;
    let marks = Marks {
        lines,
        points,
        quads,
        heatmaps,
        heatmap_size: header.heatmap_size.clone(),
        panels: header.batches.clone(),
    };
    if marks.panels.len() != header.panels.len() {
        return Err("plot payload panel count mismatch".into());
    }
    let voxels = header
        .volume_shape
        .iter()
        .try_fold(1usize, |total, axis| total.checked_mul(*axis as usize))
        .ok_or("plot payload volume is too large")?;
    let mut volume = VolumeMark {
        shape: header.volume_shape,
        origin: header.volume_origin,
        voxel: header.volume_voxel,
        ramp: header.volume_ramp,
        lo: header.volume_lo,
        hi: header.volume_hi,
        base_lo: header.volume_base_lo,
        base_hi: header.volume_base_hi,
        gain: header.volume_gain,
        opacity: header.volume_opacity,
        mode: header.volume_mode,
        filter: header.volume_filter,
        gantry: header.volume_gantry,
        unit: header.volume_unit.clone(),
        show_phantom: header.volume_phantom,
        field: header.volume_field,
        session_colors: header.volume_colors.clone(),
        session_focus: header.volume_session_focus,
        ..VolumeMark::default()
    };
    let mut scan = None;
    if voxels > 0 {
        volume.values = read_cube(&mut reader, voxels)?;
        if header.volume_ct {
            volume.ct = read_f32s(&mut reader, voxels)?;
        }
        if header.volume_labels {
            volume.labels = reader.take(voxels)?.to_vec();
        }
        scan = Some(read_scan(
            &mut reader,
            [
                header.volume_shape[0] as usize,
                header.volume_shape[1] as usize,
                header.volume_shape[2] as usize,
            ],
        )?);
    }
    volume.sessions = read_sessions(&mut reader, header.volume_sessions)?;
    volume.plans = read_sessions(&mut reader, header.volume_plans)?;
    volume.companions = read_sessions(&mut reader, header.volume_companions)?;
    Ok((
        header,
        marks,
        EncodedMarks {
            lines: line_bytes,
            points: point_bytes,
            quads: quad_bytes,
        },
        volume,
        scan,
    ))
}

/// One decoded payload. A clone shares the dose cubes, so every dose cell of a
/// scene builds its plot from one decode.
#[derive(Clone)]
pub struct Decoded {
    header: PlotHeader,
    marks: Marks,
    encoded: EncodedMarks,
    dose: Option<crate::render::DoseParts>,
}

impl Decoded {
    pub fn new(bytes: &[u8]) -> Result<Self, String> {
        let (header, marks, encoded, volume, scan) = decode_plot(bytes)?;
        Ok(Self {
            header,
            marks,
            encoded,
            dose: crate::render::dose_parts(volume, scan),
        })
    }
}

impl Plot {
    pub fn from_payload(bytes: &[u8]) -> Result<Self, String> {
        Decoded::new(bytes).map(Self::from_decoded)
    }

    pub fn from_decoded(decoded: Decoded) -> Self {
        let Decoded {
            header,
            marks,
            encoded,
            dose,
        } = decoded;
        let mut plot = Self::from_parts(
            header.panels,
            header.columns,
            header.column_weights,
            header.row_weights,
            header.row_splits,
            header.side,
            marks,
            header.background,
            header.foreground,
        );
        plot.encoded = Some(encoded);
        if let Some(parts) = dose {
            plot.attach_parts(parts);
        }
        plot.carry_lines(
            header.line_token.parse::<u64>().unwrap_or(0),
            header.reuse_lines,
            header.line_prefix,
        );
        plot
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, len: usize) -> Result<&'a [u8], String> {
        let end = self
            .at
            .checked_add(len)
            .filter(|end| *end <= self.bytes.len());
        let Some(end) = end else {
            return Err("plot payload is truncated".into());
        };
        let out = &self.bytes[self.at..end];
        self.at = end;
        Ok(out)
    }
}
#[cfg(test)]
mod tests {
    use scan_kit_core::Series;

    use super::*;

    #[test]
    fn payload_round_trips_marks_and_panel_frames() {
        let scene = PlotScene {
            title: "mixed".into(),
            panels: vec![
                Panel {
                    title: "a".into(),
                    y_label: String::new(),
                    x_label: String::new(),
                    xmin: 0.0,
                    xmax: 4.0,
                    ymin: 0.0,
                    ymax: 4.0,
                    series: vec![
                        Series::Polyline {
                            xs: vec![0.0, 1.0, 2.0],
                            ys: vec![1.0, 3.0, 2.0],
                            color: [1.0, 0.0, 0.0, 1.0],
                            thickness: 2.0,
                        },
                        Series::Points {
                            xs: vec![1.0, 2.0],
                            ys: vec![1.0, 2.0],
                            color: [0.0, 1.0, 0.0, 1.0],
                            radius: 3.0,
                            times: Vec::new(),
                        },
                        Series::Bars {
                            edges: vec![0.0, 1.0, 2.0],
                            counts: vec![2.0, 3.0],
                            color: [0.0, 0.0, 1.0, 1.0],
                        },
                    ],
                    x_labels: vec!["x".into()],
                    equal: false,
                },
                Panel {
                    title: String::new(),
                    y_label: String::new(),
                    x_label: String::new(),
                    xmin: 0.0,
                    xmax: 1.0,
                    ymin: 0.0,
                    ymax: 1.0,
                    series: vec![Series::heatmap(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], 3, 2)],
                    x_labels: Vec::new(),
                    equal: false,
                },
            ],
            controls: Vec::new(),
            table: None,
            columns: 2,
            column_weights: vec![2.0, 1.0],
            row_weights: vec![2.0, 1.0, 1.0],
            side: 1,
            row_splits: Vec::new(),
            volume: scan_kit_core::VolumeMark::default(),
        };
        let bytes = encode_plot(&scene, [0.1, 0.1, 0.1, 1.0], [0.9, 0.9, 0.9, 1.0]).unwrap();
        let (header, marks, _, _, _) = decode_plot(&bytes).unwrap();
        assert_eq!(marks, build_marks(&scene.panels));
        assert_eq!(header.panels, header_panels(&scene.panels));
        assert_eq!(header.column_weights, vec![2.0, 1.0]);
        assert_eq!(header.row_weights, vec![2.0, 1.0, 1.0]);
        assert_eq!(header.side, 1);
        assert!(marks.quads.iter().any(|quad| quad.heatmap == Some(0)));
        assert!(decode_plot(&bytes[..bytes.len() - 1]).is_err());
    }

    #[test]
    fn a_cube_round_trips_every_bit() {
        let mut edge = vec![0.0f32; 3];
        edge.extend([1.0, -0.0, f32::NAN, 2.0]);
        edge.extend([0.0; ZERO_RUN - 1]);
        edge.push(3.0);
        edge.extend([0.0; ZERO_RUN]);
        edge.extend([4.0, f32::MIN_POSITIVE]);
        edge.extend([0.0; 40]);
        for values in [edge, vec![0.0; 50], vec![5.0; 9], Vec::new()] {
            let mut out = Vec::new();
            write_cube(&mut out, &values);
            let mut reader = Reader { bytes: &out, at: 0 };
            let back = read_cube(&mut reader, values.len()).unwrap();
            assert_eq!(reader.at, out.len());
            let bits = |cube: &[f32]| cube.iter().map(|value| value.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&back), bits(&values));
        }
        let mut out = Vec::new();
        write_cube(&mut out, &[0.0; 1000]);
        assert_eq!(out.len(), 8);
        let stuck = [0u8; 8];
        assert!(read_cube(
            &mut Reader {
                bytes: &stuck,
                at: 0
            },
            4
        )
        .is_err());
        let short = [5u8, 0, 0, 0, 0, 0, 0, 0];
        let mut reader = Reader {
            bytes: &short,
            at: 0,
        };
        assert!(read_cube(&mut reader, 1 << 40).is_err());
    }

    #[test]
    fn payload_keeps_the_dose_window_and_gantry() {
        let mut scene = PlotScene {
            title: "dose".into(),
            panels: vec![Panel {
                title: "Axial".into(),
                y_label: String::new(),
                x_label: String::new(),
                xmin: 0.0,
                xmax: 1.0,
                ymin: 0.0,
                ymax: 1.0,
                series: Vec::new(),
                x_labels: Vec::new(),
                equal: false,
            }],
            controls: Vec::new(),
            table: None,
            columns: 1,
            column_weights: Vec::new(),
            row_weights: Vec::new(),
            side: 0,
            row_splits: Vec::new(),
            volume: VolumeMark {
                values: vec![1.0],
                shape: [1, 1, 1],
                origin: [0.0, 0.0, 0.0],
                voxel: 1.0,
                lo: 0.0,
                hi: 1.5,
                base_lo: 0.0,
                base_hi: 4.0,
                gantry: 90.0,
                unit: "Gy".into(),
                show_phantom: true,
                field: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                ..VolumeMark::default()
            },
        };
        scene.volume.gain = 1.0;
        scene.volume.ct = vec![12.0];
        scene.volume.labels = vec![3];
        scene.volume.session_colors = vec![[0.2, 0.3, 0.8, 1.0]];
        scene.volume.session_focus = 1;
        scene.volume.sessions = vec![scan_kit_core::SessionDose {
            values: vec![4.0, 5.0],
            shape: [1, 1, 2],
            origin: [1.0, 2.0, 3.0],
            voxel: 2.0,
        }];
        scene.volume.plans = vec![scan_kit_core::SessionDose {
            values: vec![7.0],
            shape: [1, 1, 1],
            origin: [0.0; 3],
            voxel: 1.0,
        }];
        scene.volume.companions = vec![scan_kit_core::SessionDose {
            values: vec![8.0],
            shape: [1, 1, 1],
            origin: [0.0; 3],
            voxel: 1.0,
        }];
        let bytes = encode_plot(&scene, [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]).unwrap();
        let (_, _, _, volume, scan) = decode_plot(&bytes).unwrap();
        let fresh = scan_kit_core::scan_grid(&volume.values, [1, 1, 1], volume.voxel.max(1e-3));
        let scan = scan.expect("a dose cube ships its scan");
        assert_eq!(scan.peak_at, fresh.peak_at);
        assert_eq!(scan.bricks, fresh.bricks);
        assert_eq!(scan.integrals, fresh.integrals);
        assert!((volume.gantry - 90.0).abs() < 1e-4);
        assert_eq!(volume.session_focus, 1);
        assert_eq!(volume.session_colors, vec![[0.2, 0.3, 0.8, 1.0]]);
        assert_eq!(volume.sessions[0].values, vec![4.0, 5.0]);
        assert_eq!(volume.plans[0].values, vec![7.0]);
        assert_eq!(volume.companions[0].values, vec![8.0]);
        assert_eq!(volume.sessions[0].shape, [1, 1, 2]);
        assert!((volume.sessions[0].voxel - 2.0).abs() < 1e-4);
        assert_eq!(volume.unit, "Gy");
        assert!(volume.show_phantom);
        assert_eq!(volume.field, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert!((volume.base_hi - 4.0).abs() < 1e-4);
        assert!((volume.hi - 1.5).abs() < 1e-4);
        assert_eq!(volume.values, vec![1.0]);
        assert_eq!(volume.ct, vec![12.0]);
        assert_eq!(volume.labels, vec![3]);
    }

    #[test]
    fn a_settled_playhead_omits_trace_lines_the_plot_already_has() {
        let scene = trace_and_guide(0.0);
        let bytes = encode_plot(&scene, [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]).unwrap();
        let (header, _, _, _, _) = decode_plot(&bytes).unwrap();
        assert!(!header.reuse_lines);
        assert!(header.line_prefix > 1);
        let moved = trace_and_guide(0.4);
        let again = encode_plot_reusing(
            &moved,
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            "final",
            Some(&header.line_token),
        )
        .unwrap();
        let (next, _, _, _, _) = decode_plot(&again).unwrap();
        assert!(next.reuse_lines);
        assert_eq!(next.line_token, header.line_token);
        assert_eq!(next.lines, 1, "only the guide tail is in the payload");
        assert_eq!(
            next.batches,
            build_marks(&moved.panels).panels,
            "a reused payload still addresses the full line buffer"
        );
        assert!(again.len() < bytes.len());
        let mut plot = Plot::from_payload(&again).unwrap();
        let mut previous = Plot::from_payload(&bytes).unwrap();
        plot.splice_cached_lines(&mut previous);
        assert_eq!(
            plot.line_marks(),
            build_marks(&moved.panels).lines.as_slice()
        );
        let miss = encode_plot_reusing(
            &moved,
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            "final",
            Some("1"),
        )
        .unwrap();
        assert!(!decode_plot(&miss).unwrap().0.reuse_lines);
    }

    fn trace_and_guide(guide_y: f32) -> PlotScene {
        let xs: Vec<f32> = (0..48).map(|index| index as f32).collect();
        let ys = vec![1.0; xs.len()];
        PlotScene {
            title: "timeline".into(),
            panels: vec![
                Panel {
                    title: String::new(),
                    y_label: "mm".into(),
                    x_label: String::new(),
                    xmin: 0.0,
                    xmax: 47.0,
                    ymin: 0.0,
                    ymax: 2.0,
                    series: vec![Series::Polyline {
                        xs,
                        ys,
                        color: [0.2, 0.4, 0.8, 1.0],
                        thickness: 1.5,
                    }],
                    x_labels: Vec::new(),
                    equal: false,
                },
                Panel {
                    title: String::new(),
                    y_label: String::new(),
                    x_label: String::new(),
                    xmin: 0.0,
                    xmax: 1.0,
                    ymin: 0.0,
                    ymax: 1.0,
                    series: vec![Series::Guide {
                        xs: vec![0.0, 1.0],
                        ys: vec![guide_y, guide_y],
                        color: [0.5, 0.5, 0.5, 1.0],
                        thickness: 1.0,
                    }],
                    x_labels: Vec::new(),
                    equal: true,
                },
            ],
            controls: Vec::new(),
            table: None,
            columns: 2,
            column_weights: Vec::new(),
            row_weights: Vec::new(),
            side: 1,
            row_splits: Vec::new(),
            volume: scan_kit_core::VolumeMark::default(),
        }
    }
}
