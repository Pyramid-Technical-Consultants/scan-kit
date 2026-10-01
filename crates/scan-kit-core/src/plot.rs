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
    /// `ramp` selects the color scale. 0 viridis, 1 turbo, 2 fades from transparent
    /// to `color`. 3–15 are the other dose and difference scales, and 16 is the
    /// gamma map that steps at the middle of the window. `lo` and `hi` are the
    /// color window; `hi <= lo` stretches each image to its own min and max.
    Heatmap {
        values: Vec<f32>,
        cols: u32,
        rows: u32,
        #[serde(default)]
        ramp: u8,
        #[serde(default = "heatmap_white")]
        color: [f32; 4],
        #[serde(default)]
        lo: f32,
        #[serde(default)]
        hi: f32,
    },
    /// Filled rectangles in data coordinates. `x` and `y` are the lower-left corner.
    Rects {
        x: Vec<f32>,
        y: Vec<f32>,
        w: Vec<f32>,
        h: Vec<f32>,
        color: [f32; 4],
    },
    /// Filled triangles in data coordinates. `xs` and `ys` are groups of three corners.
    Triangles {
        xs: Vec<f32>,
        ys: Vec<f32>,
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
    /// Quantity drawn up the left side, including units when the series has them.
    pub y_label: String,
    pub xmin: f32,
    pub xmax: f32,
    pub ymin: f32,
    pub ymax: f32,
    pub series: Vec<Series>,
    /// Category names drawn at x = 0, 1, 2, …. Empty draws numeric ticks from the range.
    pub x_labels: Vec<String>,
    /// One data unit has the same pixel length on both axes.
    #[serde(default)]
    pub equal: bool,
}

fn heatmap_white() -> [f32; 4] {
    [1.0, 1.0, 1.0, 1.0]
}

impl Series {
    /// Viridis heatmap. Existing dose and window maps use this.
    pub fn heatmap(values: Vec<f32>, cols: u32, rows: u32) -> Self {
        Self::Heatmap {
            values,
            cols,
            rows,
            ramp: 0,
            color: heatmap_white(),
            lo: 0.0,
            hi: 0.0,
        }
    }
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

/// Pixel rectangle. `y` grows downward from the top of the framebuffer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlotRect {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

/// Visible window in data coordinates. The vertex shader only multiplies by
/// [`Camera::clip_from_data`]. A later orbit camera writes the same matrix slot.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera {
    pub xmin: f32,
    pub xmax: f32,
    pub ymin: f32,
    pub ymax: f32,
}

impl Camera {
    pub fn new(xmin: f32, xmax: f32, ymin: f32, ymax: f32) -> Self {
        let (xmin, xmax) = span(xmin, xmax);
        let (ymin, ymax) = span(ymin, ymax);
        Self {
            xmin,
            xmax,
            ymin,
            ymax,
        }
    }

    /// `factor` > 1 zooms in. The data point stays fixed in the window.
    pub fn zoom_at(&mut self, x: f32, y: f32, factor: f32) {
        if !factor.is_finite() || factor < 1.0e-3 {
            return;
        }
        self.xmin = x - (x - self.xmin) / factor;
        self.xmax = x + (self.xmax - x) / factor;
        self.ymin = y - (y - self.ymin) / factor;
        self.ymax = y + (self.ymax - y) / factor;
    }

    pub fn pan(&mut self, dx: f32, dy: f32) {
        if dx.is_finite() {
            self.xmin += dx;
            self.xmax += dx;
        }
        if dy.is_finite() {
            self.ymin += dy;
            self.ymax += dy;
        }
    }

    /// Drag the picture with the pointer. `dx` and `dy` are framebuffer pixels, y down.
    pub fn pan_pixels(&mut self, dx: f32, dy: f32, plot: PlotRect) {
        let w = plot.w.max(1.0);
        let h = plot.h.max(1.0);
        self.pan(
            -dx * (self.xmax - self.xmin) / w,
            dy * (self.ymax - self.ymin) / h,
        );
    }

    pub fn zoom_at_pixel(
        &mut self,
        px: f32,
        py: f32,
        plot: PlotRect,
        width: f32,
        height: f32,
        factor: f32,
    ) {
        let [x, y] = self.data_at(px, py, plot, width, height);
        self.zoom_at(x, y, factor);
    }

    pub fn data_at(&self, px: f32, py: f32, plot: PlotRect, width: f32, height: f32) -> [f32; 2] {
        let matrix = self.clip_from_data(plot, width, height);
        let width = width.max(1.0);
        let height = height.max(1.0);
        let ndc_x = (px / width) * 2.0 - 1.0;
        let ndc_y = 1.0 - (py / height) * 2.0;
        let ax = matrix[0];
        let ay = matrix[5];
        [
            if ax.abs() < 1.0e-12 {
                self.xmin
            } else {
                (ndc_x - matrix[12]) / ax
            },
            if ay.abs() < 1.0e-12 {
                self.ymin
            } else {
                (ndc_y - matrix[13]) / ay
            },
        ]
    }

    /// Column-major `clip_from_data`. Data `y` grows up. Framebuffer `y` grows down.
    /// `z` is preserved, so these charts upload `z = 0` and a later view can write a
    /// view-projection into the same slot.
    pub fn clip_from_data(&self, plot: PlotRect, width: f32, height: f32) -> [f32; 16] {
        let width = width.max(1.0);
        let height = height.max(1.0);
        let xspan = (self.xmax - self.xmin).max(1.0e-6);
        let yspan = (self.ymax - self.ymin).max(1.0e-6);
        let ax = (2.0 * plot.w / width) / xspan;
        let ay = (2.0 * plot.h / height) / yspan;
        let bx = (2.0 * plot.x / width) - 1.0 - ax * self.xmin;
        let by = 1.0 - 2.0 * (plot.y + plot.h) / height - ay * self.ymin;
        [
            ax, 0.0, 0.0, 0.0, 0.0, ay, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, bx, by, 0.0, 1.0,
        ]
    }
}

fn span(min: f32, max: f32) -> (f32, f32) {
    if !min.is_finite() || !max.is_finite() {
        return (0.0, 1.0);
    }
    if (max - min).abs() < 1.0e-6 {
        return (min - 0.5, min + 0.5);
    }
    if min <= max {
        (min, max)
    } else {
        (max, min)
    }
}

/// Column-major multiply. The shader does the same `matrix * vec4(position, 1)`.
pub fn apply_clip(matrix: [f32; 16], x: f32, y: f32, z: f32) -> [f32; 4] {
    [
        matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12],
        matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13],
        matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14],
        matrix[3] * x + matrix[7] * y + matrix[11] * z + matrix[15],
    ]
}

/// Framebuffer pixel of a data point. `y` grows downward. Matches the vertex shader.
pub fn project(matrix: [f32; 16], point: [f32; 3], width: f32, height: f32) -> [f32; 2] {
    let clip = apply_clip(matrix, point[0], point[1], point[2]);
    let w = if clip[3].abs() < 1.0e-8 { 1.0 } else { clip[3] };
    let ndc_x = clip[0] / w;
    let ndc_y = clip[1] / w;
    let width = width.max(1.0);
    let height = height.max(1.0);
    [
        (ndc_x * 0.5 + 0.5) * width,
        (1.0 - (ndc_y * 0.5 + 0.5)) * height,
    ]
}

/// 1-2-5 tick values that cover `lo` to `hi`.
pub fn ticks(lo: f32, hi: f32) -> Vec<f32> {
    let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };
    if !lo.is_finite() || !hi.is_finite() || hi - lo < 1.0e-6 {
        return vec![lo];
    }
    let raw = (hi - lo) / 4.0;
    let mag = 10.0f32.powf(raw.abs().max(1.0e-20).log10().floor());
    let residual = raw / mag;
    let nice = if residual < 1.5 {
        1.0
    } else if residual < 3.0 {
        2.0
    } else if residual < 7.0 {
        5.0
    } else {
        10.0
    };
    let step = nice * mag;
    if step <= 0.0 || !step.is_finite() {
        return vec![lo, hi];
    }
    let mut value = (lo / step).ceil() * step;
    if value < lo {
        value += step;
    }
    let mut out = Vec::new();
    let mut guard = 0;
    while value <= hi + step * 1.0e-3 && guard < 24 {
        let snapped = (value / step).round() * step;
        if snapped >= lo - step * 1.0e-3 && snapped <= hi + step * 1.0e-3 {
            out.push(snapped);
        }
        value += step;
        guard += 1;
    }
    if out.is_empty() {
        vec![lo]
    } else {
        out
    }
}

/// A short tick label.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ticks_are_one_two_five_and_zoom_keeps_the_point() {
        let marks = ticks(0.0, 10.0);
        assert_eq!(marks, vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0]);
        let mut camera = Camera::new(0.0, 10.0, 0.0, 10.0);
        let plot = PlotRect {
            x: 10.0,
            y: 20.0,
            w: 80.0,
            h: 40.0,
        };
        let before = project(
            camera.clip_from_data(plot, 100.0, 80.0),
            [5.0, 5.0, 0.0],
            100.0,
            80.0,
        );
        camera.zoom_at(5.0, 5.0, 2.0);
        assert!((camera.xmin - 2.5).abs() < 1.0e-4);
        assert!((camera.xmax - 7.5).abs() < 1.0e-4);
        let matrix = camera.clip_from_data(plot, 100.0, 80.0);
        let after = project(matrix, [5.0, 5.0, 0.0], 100.0, 80.0);
        assert!((before[0] - after[0]).abs() < 1.0e-3);
        assert!((before[1] - after[1]).abs() < 1.0e-3);
        let clip = apply_clip(matrix, 0.0, 0.0, 0.0);
        assert_eq!(clip[2], 0.0);
        let origin = project(
            Camera::new(0.0, 10.0, 0.0, 10.0).clip_from_data(
                PlotRect {
                    x: 0.0,
                    y: 0.0,
                    w: 100.0,
                    h: 80.0,
                },
                100.0,
                80.0,
            ),
            [0.0, 0.0, 0.0],
            100.0,
            80.0,
        );
        assert!((origin[0]).abs() < 1.0e-3);
        assert!((origin[1] - 80.0).abs() < 1.0e-3);
    }
}
