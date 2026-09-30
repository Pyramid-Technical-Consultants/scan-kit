//! A plot scene is data. `scan-kit-compute` turns it into pixels.

use serde::{Deserialize, Serialize};

/// One drawable series in data coordinates.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Series {
    Polyline {
        xs: Vec<f32>,
        ys: Vec<f32>,
        color: [f32; 4],
        thickness: f32,
    },
    Points {
        xs: Vec<f32>,
        ys: Vec<f32>,
        color: [f32; 4],
        radius: f32,
    },
    Bars {
        edges: Vec<f32>,
        counts: Vec<f32>,
        color: [f32; 4],
    },
    Heatmap {
        values: Vec<f32>,
        cols: u32,
        rows: u32,
    },
}

/// One axes rectangle. Several panels share a frame.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Panel {
    pub title: String,
    pub xmin: f32,
    pub xmax: f32,
    pub ymin: f32,
    pub ymax: f32,
    pub series: Vec<Series>,
}

/// A control the shell renders with shadcn. The view does not draw it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Control {
    pub id: String,
    pub label: String,
    pub options: Vec<String>,
    pub value: String,
}

/// Tabular result for views that are not pictures. The shell uses Glide.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DataTable {
    pub columns: Vec<String>,
    pub rows: Vec<Vec<String>>,
}

/// What a view workflow returns before pixels are rendered.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlotScene {
    pub title: String,
    pub panels: Vec<Panel>,
    pub controls: Vec<Control>,
    pub table: Option<DataTable>,
    /// Audio samples at 1 kHz, when the view is Audio Explorer.
    pub samples: Vec<f32>,
}

impl PlotScene {
    pub fn empty(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            panels: Vec::new(),
            controls: Vec::new(),
            table: None,
            samples: Vec::new(),
        }
    }
}

/// Map a data value into a pixel span. A zero span stays at the start.
pub fn map_span(value: f32, vmin: f32, vmax: f32, start: f32, end: f32) -> f32 {
    let span = vmax - vmin;
    let t = if span.abs() < 1e-12 {
        0.0
    } else {
        ((value - vmin) / span).clamp(0.0, 1.0)
    };
    start + t * (end - start)
}
