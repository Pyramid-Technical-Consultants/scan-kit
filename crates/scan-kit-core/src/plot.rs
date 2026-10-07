//! A plot scene is data. `scan-kit-compute` turns it into pixels.

use serde::{Deserialize, Serialize};

/// Density or contour built from a timed cloud.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CloudStyle {
    Density,
    Contour,
}

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
        /// Sample time, one entry per point. Empty means the point is always drawn.
        /// A live scatter fills this so playback can hide rows outside the playhead
        /// without building the cloud again.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        times: Vec<f32>,
    },
    Bars {
        edges: Vec<f32>,
        counts: Vec<f32>,
        color: [f32; 4],
    },
    /// `ramp` is a catalog index. 0 is viridis, and the session index fades
    /// `color` from clear to that ink. `lo` and `hi` are the color window;
    /// `hi <= lo` stretches each image to its own min and max.
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
    /// Timed samples a density or contour was counted from. The plot does not
    /// draw these as dots. A playhead step counts the visible window into the
    /// heatmap or contour that precedes this series.
    Cloud {
        xs: Vec<f32>,
        ys: Vec<f32>,
        times: Vec<f32>,
        x0: f32,
        x1: f32,
        y0: f32,
        y1: f32,
        style: CloudStyle,
        #[serde(default)]
        cutoff: f32,
    },
}

/// One axes rectangle. Several panels share a frame.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Panel {
    pub title: String,
    /// Quantity drawn up the left side, including units when the series has them.
    pub y_label: String,
    /// Quantity drawn under the tick labels, including units.
    #[serde(default)]
    pub x_label: String,
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
            ramp: crate::ramp::VIRIDIS,
            color: heatmap_white(),
            lo: 0.0,
            hi: 0.0,
        }
    }
}

/// One menu row. `label` includes the unit when the source has one.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Choice {
    pub id: String,
    pub label: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub detail: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub icon: String,
}

impl Choice {
    pub fn plain(text: &str) -> Self {
        Self {
            id: text.to_string(),
            label: text.to_string(),
            detail: String::new(),
            icon: String::new(),
        }
    }

    pub fn full(id: &str, label: &str, detail: impl Into<String>, icon: &str) -> Self {
        Self {
            id: id.to_string(),
            label: label.to_string(),
            detail: detail.into(),
            icon: icon.to_string(),
        }
    }
}

impl PartialEq<str> for Choice {
    fn eq(&self, other: &str) -> bool {
        self.label == other || self.id == other
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
enum ChoiceWire {
    Text(String),
    Full(Choice),
}

fn choices_de<'de, D>(deserializer: D) -> Result<Vec<Choice>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let raw = Vec::<ChoiceWire>::deserialize(deserializer)?;
    Ok(raw
        .into_iter()
        .map(|item| match item {
            ChoiceWire::Text(text) => Choice::plain(&text),
            ChoiceWire::Full(choice) => choice,
        })
        .collect())
}

/// A control the shell renders with shadcn. The view does not draw it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Control {
    pub id: String,
    pub label: String,
    #[serde(deserialize_with = "choices_de")]
    pub options: Vec<Choice>,
    pub value: String,
    /// Sidebar fieldset. Empty lands in Options.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub group: String,
    /// `check` is a checkbox. `radio` is an exclusive button group. Empty is a select.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub kind: String,
}

impl Control {
    pub fn plain(
        id: impl Into<String>,
        label: impl Into<String>,
        options: impl IntoIterator<Item = impl AsRef<str>>,
        value: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            options: options
                .into_iter()
                .map(|option| Choice::plain(option.as_ref()))
                .collect(),
            value: value.into(),
            group: String::new(),
            kind: String::new(),
        }
    }

    pub fn grouped(mut self, group: &str) -> Self {
        self.group = group.to_string();
        self
    }

    pub fn checked(mut self) -> Self {
        self.kind = "check".to_string();
        self
    }

    pub fn radio(mut self) -> Self {
        self.kind = "radio".to_string();
        self
    }

    pub fn icons(mut self, icons: &[&str]) -> Self {
        for (choice, icon) in self.options.iter_mut().zip(icons) {
            choice.icon = (*icon).to_string();
        }
        self
    }

    pub fn labels(&self) -> Vec<&str> {
        self.options
            .iter()
            .map(|choice| choice.label.as_str())
            .collect()
    }
}

/// Tabular result for views that are not pictures. The shell uses Glide.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DataTable {
    pub columns: Vec<String>,
    pub rows: Vec<Vec<String>>,
}

/// Dose grid the plot uploads once. Slice and 3D cells sample it locally.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct VolumeMark {
    /// Dose samples, x-fastest.
    pub values: Vec<f32>,
    /// CT samples under the wash. Empty leaves the wash on its own.
    #[serde(default)]
    pub ct: Vec<f32>,
    /// Structure id per voxel. `0` is outside every outline.
    #[serde(default)]
    pub labels: Vec<u8>,
    pub shape: [u32; 3],
    pub origin: [f32; 3],
    pub voxel: f32,
    #[serde(default)]
    pub ramp: u8,
    #[serde(default)]
    pub lo: f32,
    #[serde(default)]
    pub hi: f32,
    /// Dose window before gain or a manual window. Both zero means the plot measures the cube.
    #[serde(default)]
    pub base_lo: f32,
    #[serde(default)]
    pub base_hi: f32,
    /// 0 is treated as 1 when the plot attaches the grid.
    #[serde(default)]
    pub gain: f32,
    #[serde(default)]
    pub opacity: f32,
    /// 0 integrate, 1 maximum, 2 transparent.
    #[serde(default)]
    pub mode: u8,
    /// 0 nearest, 1 linear, 2 cubic.
    #[serde(default)]
    pub filter: u8,
    /// Degrees about +X. 90 lays beam depth along Y. 0 leaves the lattice as deposited.
    #[serde(default)]
    pub gantry: f32,
    /// Color-bar unit. Empty draws the bar without a title.
    #[serde(default)]
    pub unit: String,
    /// Cyan phantom box in the 3D march.
    #[serde(default)]
    pub show_phantom: bool,
    /// Field box in millimetres: x0, x1, y0, y1, z0, z1. A zero span draws nothing.
    #[serde(default)]
    pub field: [f32; 6],
    /// Other loaded sessions, in selection order. An empty `values` buffer is the
    /// cube in `values` above (`session_focus`). Line plots draw every entry.
    #[serde(default)]
    pub sessions: Vec<SessionDose>,
    /// Palette ink for `sessions`, same order. Empty keeps the foreground.
    #[serde(default)]
    pub session_colors: Vec<[f32; 4]>,
    /// Which `sessions` entry is the cube shown in the image cells.
    #[serde(default)]
    pub session_focus: u32,
}

/// One session cube for the dose line plots. Empty `values` means the main cube.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SessionDose {
    pub values: Vec<f32>,
    pub shape: [u32; 3],
    pub origin: [f32; 3],
    pub voxel: f32,
}

/// What a view workflow returns before pixels are rendered.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlotScene {
    pub title: String,
    pub panels: Vec<Panel>,
    pub controls: Vec<Control>,
    pub table: Option<DataTable>,
    /// Columns in the panel grid. `0` packs panels into a square.
    pub columns: u32,
    /// Relative column widths. Empty means equal columns.
    pub column_weights: Vec<f32>,
    /// Relative row heights. Empty means equal rows.
    pub row_weights: Vec<f32>,
    /// Panels at the end of `panels` that fill the right column.
    /// `0` keeps the row-major grid.
    #[serde(default)]
    pub side: u32,
    /// Left and right weight of every row when `columns` is 2 and this holds
    /// one pair per panel. Empty shares [`Self::column_weights`] across rows.
    #[serde(default)]
    pub row_splits: Vec<f32>,
    /// Volume sampled by the dose workspace. Empty for every other view.
    #[serde(default)]
    pub volume: VolumeMark,
}

impl PlotScene {
    pub fn empty(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            panels: Vec::new(),
            controls: Vec::new(),
            table: None,
            columns: 0,
            column_weights: Vec::new(),
            row_weights: Vec::new(),
            side: 0,
            row_splits: Vec::new(),
            volume: VolumeMark::default(),
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
