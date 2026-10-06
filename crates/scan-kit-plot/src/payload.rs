//! One binary payload per opened view.
//!
//! Layout: `u32` little-endian header length, the JSON [`PlotHeader`], then the
//! line, point, and quad instance bytes in GPU layout, then each heatmap's
//! RGBA pixels. The shell reads the header for its controls. The wasm plot uploads
//! the rest without parsing series again.

use scan_kit_core::{Control, DataTable, Panel, PlotScene};
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
    Ok(out)
}

pub fn plot_header(bytes: &[u8]) -> Result<PlotHeader, String> {
    decode_plot(bytes).map(|(header, _, _)| header)
}

/// GPU bytes already packed by [`encode_plot_quality`]. Uploading these skips a
/// second pass through the mark structs.
pub(crate) struct EncodedMarks {
    pub lines: Vec<u8>,
    pub points: Vec<u8>,
    pub quads: Vec<u8>,
}

pub(crate) fn decode_plot(bytes: &[u8]) -> Result<(PlotHeader, Marks, EncodedMarks), String> {
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
    Ok((
        header,
        marks,
        EncodedMarks {
            lines: line_bytes,
            points: point_bytes,
            quads: quad_bytes,
        },
    ))
}

impl Plot {
    pub fn from_payload(bytes: &[u8]) -> Result<Self, String> {
        let (header, marks, encoded) = decode_plot(bytes)?;
        let mut plot = Self::from_parts(
            header.panels,
            header.columns,
            header.column_weights,
            header.row_weights,
            header.side,
            marks,
            header.background,
            header.foreground,
        );
        plot.encoded = Some(encoded);
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
        };
        let bytes = encode_plot(&scene, [0.1, 0.1, 0.1, 1.0], [0.9, 0.9, 0.9, 1.0]).unwrap();
        let (header, marks, _) = decode_plot(&bytes).unwrap();
        assert_eq!(marks, build_marks(&scene.panels));
        assert_eq!(header.panels, header_panels(&scene.panels));
        assert_eq!(header.column_weights, vec![2.0, 1.0]);
        assert_eq!(header.row_weights, vec![2.0, 1.0, 1.0]);
        assert_eq!(header.side, 1);
        assert!(marks.quads.iter().any(|quad| quad.heatmap == Some(0)));
        assert!(decode_plot(&bytes[..bytes.len() - 1]).is_err());
    }

    #[test]
    fn a_settled_playhead_omits_trace_lines_the_plot_already_has() {
        let scene = trace_and_guide(0.0);
        let bytes = encode_plot(&scene, [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]).unwrap();
        let (header, _, _) = decode_plot(&bytes).unwrap();
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
        let (next, _, _) = decode_plot(&again).unwrap();
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
        }
    }
}
