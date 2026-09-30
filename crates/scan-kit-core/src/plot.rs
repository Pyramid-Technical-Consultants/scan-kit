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
    /// Filled rectangles in data coordinates. `x` and `y` are the lower-left corner.
    Rects {
        x: Vec<f32>,
        y: Vec<f32>,
        w: Vec<f32>,
        h: Vec<f32>,
        color: [f32; 4],
    },
    /// A polyline the shell palette leaves alone. Reference lines use this.
    Guide {
        xs: Vec<f32>,
        ys: Vec<f32>,
        color: [f32; 4],
        thickness: f32,
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
    /// Category names drawn at x = 0, 1, 2, …. Empty draws numeric ticks from the range.
    pub x_labels: Vec<String>,
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
    /// Columns in the panel grid. `0` packs panels into a square.
    pub columns: u32,
    /// Relative column widths. Empty means equal columns.
    pub column_weights: Vec<f32>,
}

impl PlotScene {
    pub fn empty(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            panels: Vec::new(),
            controls: Vec::new(),
            table: None,
            samples: Vec::new(),
            columns: 0,
            column_weights: Vec::new(),
        }
    }
}

/// A short tick label. Digits, a sign, and one dot, so the plot font can draw it.
pub fn format_tick(value: f32) -> String {
    if !value.is_finite() {
        return String::new();
    }
    if value.abs() < 1.0e6 && (value.abs() >= 100.0 || (value - value.round()).abs() < 1.0e-3) {
        format!("{}", value.round() as i64)
    } else if value.abs() >= 10.0 {
        format!("{value:.1}")
    } else {
        format!("{value:.2}")
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
