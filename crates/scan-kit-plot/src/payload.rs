//! One binary payload per opened view.
//!
//! Layout: `u32` little-endian header length, the JSON [`PlotHeader`], then the
//! line, point, and quad instance bytes in GPU layout, then each heatmap's
//! RGBA pixels. The shell reads the header for its controls. The wasm plot uploads
//! the rest without parsing series again.

use scan_kit_core::{Control, DataTable, Panel, PlotScene};
use serde::{Deserialize, Serialize};

use crate::render::{
    build_marks, decode_lines, decode_points, decode_quads, encode_lines, encode_points,
    encode_quads, header_panels, Marks, PanelBatch, Plot, LINE_STRIDE, POINT_STRIDE, QUAD_STRIDE,
};

/// `apps/desktop/src/plot-header.ts` mirrors the public fields. Rename both together.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlotHeader {
    pub title: String,
    pub controls: Vec<Control>,
    pub table: Option<DataTable>,
    pub samples: Vec<f32>,
    /// Panel frames. Series are empty. They travel as marks.
    pub panels: Vec<Panel>,
    pub columns: u32,
    pub column_weights: Vec<f32>,
    #[serde(default)]
    pub row_weights: Vec<f32>,
    pub background: [f32; 4],
    pub foreground: [f32; 4],
    batches: Vec<PanelBatch>,
    heatmap_size: Vec<(u32, u32)>,
    lines: u32,
    points: u32,
    quads: u32,
}

pub fn encode_plot(
    scene: &PlotScene,
    background: [f32; 4],
    foreground: [f32; 4],
) -> Result<Vec<u8>, String> {
    let marks = build_marks(&scene.panels);
    let header = PlotHeader {
        title: scene.title.clone(),
        controls: scene.controls.clone(),
        table: scene.table.clone(),
        samples: scene.samples.clone(),
        panels: header_panels(&scene.panels),
        columns: scene.columns,
        column_weights: scene.column_weights.clone(),
        row_weights: scene.row_weights.clone(),
        background,
        foreground,
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
    decode_plot(bytes).map(|(header, _)| header)
}

pub(crate) fn decode_plot(bytes: &[u8]) -> Result<(PlotHeader, Marks), String> {
    let mut reader = Reader { bytes, at: 0 };
    let json_len = u32::from_le_bytes(reader.take(4)?.try_into().unwrap()) as usize;
    let header: PlotHeader =
        serde_json::from_slice(reader.take(json_len)?).map_err(|err| err.to_string())?;
    let lines = decode_lines(reader.take(header.lines as usize * LINE_STRIDE as usize)?);
    let points = decode_points(reader.take(header.points as usize * POINT_STRIDE as usize)?);
    let quads = decode_quads(reader.take(header.quads as usize * QUAD_STRIDE as usize)?);
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
    Ok((header, marks))
}

impl Plot {
    pub fn from_payload(bytes: &[u8]) -> Result<Self, String> {
        let (header, marks) = decode_plot(bytes)?;
        Ok(Self::from_parts(
            header.panels,
            header.columns,
            header.column_weights,
            header.row_weights,
            marks,
            header.background,
            header.foreground,
        ))
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
            samples: vec![0.5],
            columns: 2,
            column_weights: vec![2.0, 1.0],
            row_weights: vec![2.0, 1.0, 1.0],
        };
        let bytes = encode_plot(&scene, [0.1, 0.1, 0.1, 1.0], [0.9, 0.9, 0.9, 1.0]).unwrap();
        let (header, marks) = decode_plot(&bytes).unwrap();
        assert_eq!(marks, build_marks(&scene.panels));
        assert_eq!(header.panels, header_panels(&scene.panels));
        assert_eq!(header.samples, vec![0.5]);
        assert_eq!(header.column_weights, vec![2.0, 1.0]);
        assert_eq!(header.row_weights, vec![2.0, 1.0, 1.0]);
        assert!(marks.quads.iter().any(|quad| quad.heatmap == Some(0)));
        assert!(decode_plot(&bytes[..bytes.len() - 1]).is_err());
    }
}
