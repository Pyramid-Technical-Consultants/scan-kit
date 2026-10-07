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
        for value in scene.volume.values.iter().take(voxels) {
            out.extend_from_slice(&value.to_le_bytes());
        }
        if !scene.volume.ct.is_empty() {
            for value in scene.volume.ct.iter().take(voxels) {
                out.extend_from_slice(&value.to_le_bytes());
            }
        }
        if !scene.volume.labels.is_empty() {
            out.extend_from_slice(&scene.volume.labels[..voxels.min(scene.volume.labels.len())]);
        }
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
    let mut doses = Vec::with_capacity(count as usize);
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
        let mut values = Vec::with_capacity(n);
        for _ in 0..n {
            values.push(f32::from_le_bytes(reader.take(4)?.try_into().unwrap()));
        }
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
    for value in &session.values {
        out.extend_from_slice(&value.to_le_bytes());
    }
}

pub fn plot_header(bytes: &[u8]) -> Result<PlotHeader, String> {
    decode_plot(bytes).map(|(header, _, _, _)| header)
}

/// GPU bytes already packed by [`encode_plot_quality`]. Uploading these skips a
/// second pass through the mark structs.
pub(crate) struct EncodedMarks {
    pub lines: Vec<u8>,
    pub points: Vec<u8>,
    pub quads: Vec<u8>,
}

pub(crate) fn decode_plot(
    bytes: &[u8],
) -> Result<(PlotHeader, Marks, EncodedMarks, VolumeMark), String> {
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
    let voxels = header.volume_shape[0] as usize
        * header.volume_shape[1] as usize
        * header.volume_shape[2] as usize;
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
    if voxels > 0 {
        let mut values = Vec::with_capacity(voxels);
        for _ in 0..voxels {
            let bytes = reader.take(4)?;
            values.push(f32::from_le_bytes(bytes.try_into().unwrap()));
        }
        volume.values = values;
        if header.volume_ct {
            let mut ct = Vec::with_capacity(voxels);
            for _ in 0..voxels {
                let bytes = reader.take(4)?;
                ct.push(f32::from_le_bytes(bytes.try_into().unwrap()));
            }
            volume.ct = ct;
        }
        if header.volume_labels {
            volume.labels = reader.take(voxels)?.to_vec();
        }
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
    ))
}

impl Plot {
    pub fn from_payload(bytes: &[u8]) -> Result<Self, String> {
        let (header, marks, encoded, volume) = decode_plot(bytes)?;
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
        plot.attach_volume(&volume);
        plot.carry_lines(
            header.line_token.parse::<u64>().unwrap_or(0),
            header.reuse_lines,
            header.line_prefix,
        );
        Ok(plot)
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
        let (header, marks, _, _) = decode_plot(&bytes).unwrap();
        assert_eq!(marks, build_marks(&scene.panels));
        assert_eq!(header.panels, header_panels(&scene.panels));
        assert_eq!(header.column_weights, vec![2.0, 1.0]);
        assert_eq!(header.row_weights, vec![2.0, 1.0, 1.0]);
        assert_eq!(header.side, 1);
        assert!(marks.quads.iter().any(|quad| quad.heatmap == Some(0)));
        assert!(decode_plot(&bytes[..bytes.len() - 1]).is_err());
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
        let (_, _, _, volume) = decode_plot(&bytes).unwrap();
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
    }

    #[test]
    fn a_settled_playhead_omits_trace_lines_the_plot_already_has() {
        let scene = trace_and_guide(0.0);
        let bytes = encode_plot(&scene, [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]).unwrap();
        let (header, _, _, _) = decode_plot(&bytes).unwrap();
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
        let (next, _, _, _) = decode_plot(&again).unwrap();
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
