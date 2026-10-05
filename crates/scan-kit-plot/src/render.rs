//! Rasterize a [`scan_kit_core::PlotScene`].
//!
//! Mark positions stay in data space (`z = 0`). The vertex shader multiplies by
//! `clip_from_data`. Pan and zoom rewrite that matrix. Stroke width stays in pixels.
//!
//! Each panel paints the grid, solid quads and heatmaps, lines, points, the
//! 1px bound, then labels. [`Plot::record`] (GPU), [`Plot::paint_cpu`] (no
//! adapter), and [`Plot::pick`] (hover, reverse order) must keep that order.
//! The plot rectangle is whole pixels. Axis-aligned strokes sit on pixel
//! centers, in the vertex shader and in [`stroke_cpu`], so a 1px line is one
//! pixel. The bound is that stroke on the rectangle's outer pixels.
//! A panel with `equal` letterboxes that rectangle to the camera's data
//! aspect, so one data unit has the same pixel length on both axes.
//!
//! A mark kind's byte layout lives in its `*_STRIDE`, `*_ATTRS`, `encode_*`,
//! `decode_*`, and the matching WGSL `vs_*` inputs. Change them together.
//! The scene's lines and points are the filtered samples. A stroke may be
//! simplified here for the current camera when several samples share a pixel,
//! and that pass keeps the extrema in the pixel. A camera that gives a sample
//! its own pixel draws it. Do not thin the series before it reaches this crate.
//! Heatmap value row 0 is the low data y. The texture is stored top-first, so
//! that row is the last row of pixels. Those pixels are the catalog color for
//! the series ramp, baked before upload.
//! Draws stay within WebGL2: no base instance (bind a buffer slice instead) and
//! one color target.

use std::num::NonZeroU64;

use scan_kit_core::{format_tick, project, ticks, Camera, Panel, PlotRect, PlotScene, Series};

use serde::{Deserialize, Serialize};

use crate::text::{self, atlas};
use crate::GpuError;

const PLOT_SHADER: &str = r#"
struct Uniforms {
    clip_from_data: mat4x4<f32>,
    resolution: vec2<f32>,
    pad: vec2<f32>,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(1) @binding(0) var mark_tex: texture_2d<f32>;
@group(1) @binding(1) var mark_samp: sampler;

fn to_px(data: vec3<f32>) -> vec2<f32> {
    let clip = u.clip_from_data * vec4<f32>(data, 1.0);
    let ndc = clip.xy / max(clip.w, 0.000001);
    return vec2<f32>(
        (ndc.x * 0.5 + 0.5) * u.resolution.x,
        (1.0 - (ndc.y * 0.5 + 0.5)) * u.resolution.y,
    );
}

fn px_to_clip(px: vec2<f32>) -> vec4<f32> {
    return vec4<f32>(
        (px.x / u.resolution.x) * 2.0 - 1.0,
        1.0 - (px.y / u.resolution.y) * 2.0,
        0.0,
        1.0,
    );
}

fn shade(rgb: vec3<f32>, alpha: f32, coverage: f32) -> vec4<f32> {
    let a = alpha * coverage;
    if (a < 0.004) {
        discard;
    }
    return vec4<f32>(rgb * a, a);
}

fn orient(a: vec2<f32>, b: vec2<f32>, p: vec2<f32>) -> f32 {
    return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
}

fn is_top_left(a: vec2<f32>, b: vec2<f32>) -> bool {
    let d = b - a;
    return d.y > 0.0 || (d.y == 0.0 && d.x < 0.0);
}

fn same_side(w: f32, area: f32, a: vec2<f32>, b: vec2<f32>) -> bool {
    if (w * area > 0.0) {
        return true;
    }
    if (w != 0.0) {
        return false;
    }
    if (area > 0.0) {
        return is_top_left(a, b);
    }
    return is_top_left(b, a);
}

// Screen y grows down. A shared edge is included on one triangle only.
fn owns_pixel(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>, c: vec2<f32>) -> bool {
    let area = orient(a, b, c);
    if (area == 0.0) {
        return false;
    }
    return same_side(orient(a, b, p), area, a, b)
        && same_side(orient(b, c, p), area, b, c)
        && same_side(orient(c, a, p), area, c, a);
}

fn corner(index: u32) -> vec2<f32> {
    switch index {
        case 0u: { return vec2<f32>(-1.0, -1.0); }
        case 1u: { return vec2<f32>(1.0, -1.0); }
        case 2u: { return vec2<f32>(1.0, 1.0); }
        case 3u: { return vec2<f32>(-1.0, -1.0); }
        case 4u: { return vec2<f32>(1.0, 1.0); }
        default: { return vec2<f32>(-1.0, 1.0); }
    }
}

struct LineOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) @interpolate(flat) a: vec2<f32>,
    @location(2) @interpolate(flat) b: vec2<f32>,
    @location(3) @interpolate(flat) radius: f32,
}

@vertex
fn vs_line(
    @builtin(vertex_index) vi: u32,
    @location(0) a: vec4<f32>,
    @location(1) b: vec4<f32>,
    @location(2) color: vec4<f32>,
    @location(3) thickness: f32,
) -> LineOut {
    var pa = to_px(a.xyz);
    var pb = to_px(b.xyz);
    // A vertical or horizontal stroke centered on a pixel boundary covers two
    // pixels. Park it on one pixel center. Diagonals keep their antialiasing.
    // `snap_hairline` is this same test.
    let dx = abs(pa.x - pb.x);
    let dy = abs(pa.y - pb.y);
    if (dx < 0.05 && dy >= dx) {
        let x = floor((pa.x + pb.x) * 0.5) + 0.5;
        pa.x = x;
        pb.x = x;
    } else if (dy < 0.05) {
        let y = floor((pa.y + pb.y) * 0.5) + 0.5;
        pa.y = y;
        pb.y = y;
    }
    let delta = pb - pa;
    let len = max(length(delta), 0.001);
    let dir = delta / len;
    let normal = vec2<f32>(-dir.y, dir.x);
    let radius = max(thickness, 1.0) * 0.5;
    let pad = radius + 1.5;
    let xy = corner(vi);
    let px = (pa + pb) * 0.5 + dir * xy.x * (len * 0.5 + pad) + normal * xy.y * pad;
    var out: LineOut;
    out.clip = px_to_clip(px);
    out.color = color;
    out.a = pa;
    out.b = pb;
    out.radius = radius;
    return out;
}

@fragment
fn fs_line(in: LineOut) -> @location(0) vec4<f32> {
    let p = in.clip.xy;
    let ab = in.b - in.a;
    let len2 = max(dot(ab, ab), 0.0001);
    let t = clamp(dot(p - in.a, ab) / len2, 0.0, 1.0);
    let dist = length(p - (in.a + ab * t));
    let coverage = clamp(in.radius + 0.5 - dist, 0.0, 1.0);
    return shade(in.color.rgb, in.color.a, coverage);
}

struct PointOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) @interpolate(flat) center: vec2<f32>,
    @location(2) @interpolate(flat) radius: f32,
}

@vertex
fn vs_point(
    @builtin(vertex_index) vi: u32,
    @location(0) center: vec4<f32>,
    @location(1) color: vec4<f32>,
) -> PointOut {
    let c = to_px(center.xyz);
    let radius = max(center.w, 1.0);
    let px = c + corner(vi) * (radius + 1.5);
    var out: PointOut;
    out.clip = px_to_clip(px);
    out.color = color;
    out.center = c;
    out.radius = radius;
    return out;
}

@fragment
fn fs_point(in: PointOut) -> @location(0) vec4<f32> {
    let dist = length(in.clip.xy - in.center);
    let coverage = clamp(in.radius + 0.5 - dist, 0.0, 1.0);
    return shade(in.color.rgb, in.color.a, coverage);
}

fn quad_uv(index: u32) -> vec2<f32> {
    switch index {
        case 0u, 3u: { return vec2<f32>(0.0, 0.0); }
        case 1u: { return vec2<f32>(1.0, 0.0); }
        case 2u, 4u: { return vec2<f32>(1.0, 1.0); }
        default: { return vec2<f32>(0.0, 1.0); }
    }
}

struct QuadOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) @interpolate(flat) bounds0: vec2<f32>,
    @location(3) @interpolate(flat) bounds1: vec2<f32>,
    @location(4) @interpolate(flat) heat: f32,
}

@vertex
fn vs_quad(
    @builtin(vertex_index) vi: u32,
    @location(0) a: vec4<f32>,
    @location(1) b: vec4<f32>,
    @location(2) color: vec4<f32>,
) -> QuadOut {
    // a.w > 0.5 is a triangle (a.xy, b.xy, b.zw). Vertices grow a pixel so the
    // rasterizer emits every sample the original triangle owns. The fragment
    // shader keeps that original triangle, top-left edges included, so a shared
    // edge composites once. Vertices 3–5 collapse and add no second copy.
    if (a.w > 0.5) {
        let c0 = to_px(vec3<f32>(a.xy, 0.0));
        let c1 = to_px(vec3<f32>(b.xy, 0.0));
        let c2 = to_px(vec3<f32>(b.zw, 0.0));
        var p = c0;
        if (vi == 1u) { p = c1; }
        else if (vi == 2u) { p = c2; }
        if (vi < 3u) {
            let mid = (c0 + c1 + c2) * (1.0 / 3.0);
            let delta = p - mid;
            let len = max(length(delta), 0.001);
            p = p + delta / len * 1.0;
        }
        var out: QuadOut;
        out.clip = px_to_clip(p);
        out.color = color;
        out.uv = c2;
        out.bounds0 = c0;
        out.bounds1 = c1;
        out.heat = -1.0;
        return out;
    }
    let lo = min(a.xy, b.xy);
    let hi = max(a.xy, b.xy);
    let uv = quad_uv(vi);
    let data = vec2<f32>(mix(lo.x, hi.x, uv.x), mix(hi.y, lo.y, uv.y));
    let p0 = to_px(vec3<f32>(lo, 0.0));
    let p1 = to_px(vec3<f32>(hi, 0.0));
    let bounds0 = min(p0, p1);
    let bounds1 = max(p0, p1);
    var px = to_px(vec3<f32>(data, 0.0));
    let mid = (bounds0 + bounds1) * 0.5;
    px = px + sign(px - mid) * 1.0;
    var out: QuadOut;
    out.clip = px_to_clip(px);
    out.color = color;
    out.uv = uv;
    out.bounds0 = bounds0;
    out.bounds1 = bounds1;
    out.heat = a.z;
    return out;
}

@fragment
fn fs_quad(in: QuadOut) -> @location(0) vec4<f32> {
    if (in.heat < -0.5) {
        let coverage = select(0.0, 1.0, owns_pixel(in.clip.xy, in.bounds0, in.bounds1, in.uv));
        return shade(in.color.rgb, in.color.a, coverage);
    }
    let p = in.clip.xy;
    let outside = max(max(in.bounds0.x - p.x, p.x - in.bounds1.x), max(in.bounds0.y - p.y, p.y - in.bounds1.y));
    let coverage = clamp(0.5 - outside, 0.0, 1.0);
    // textureSample is illegal on a branch the triangle pixels skip. Level 0
    // has no derivatives, so this heatmap lookup can sit in that branch.
    // heat > 0.5 is a baked ramp. Otherwise the quad is a solid color.
    if (in.heat > 0.5) {
        let texel = textureSampleLevel(mark_tex, mark_samp, in.uv, 0.0);
        return shade(texel.rgb, texel.a, coverage);
    }
    return shade(in.color.rgb, in.color.a, coverage);
}

struct TextOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
}

@vertex
fn vs_text(
    @builtin(vertex_index) vi: u32,
    @location(0) anchor: vec4<f32>,
    @location(1) offset_size: vec4<f32>,
    @location(2) uv: vec4<f32>,
    @location(3) color: vec4<f32>,
) -> TextOut {
    var origin = anchor.xy;
    if (anchor.w > 0.5) {
        origin = to_px(anchor.xyz);
    }
    let xy = quad_uv(vi);
    var local = offset_size.xy + xy * offset_size.zw;
    // Screen-space labels with anchor.z set run up the axis (90° CCW, y down).
    if (anchor.w < 0.5 && anchor.z > 0.5) {
        local = vec2<f32>(local.y, -local.x);
    }
    let px = origin + local;
    var out: TextOut;
    out.clip = px_to_clip(px);
    out.color = color;
    out.uv = mix(uv.xy, uv.zw, xy);
    return out;
}

@fragment
fn fs_text(in: TextOut) -> @location(0) vec4<f32> {
    let coverage = textureSample(mark_tex, mark_samp, in.uv).r;
    return shade(in.color.rgb, in.color.a, coverage);
}
"#;

pub fn plot_shader_source() -> &'static str {
    PLOT_SHADER
}

pub fn compile_plot_shader() -> Result<(), String> {
    let module = naga::front::wgsl::parse_str(PLOT_SHADER).map_err(|err| err.to_string())?;
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    validator.validate(&module).map_err(|err| err.to_string())?;
    Ok(())
}

#[derive(Clone, Copy)]
pub struct PlotInput {
    pub x: f32,
    pub y: f32,
    pub dx: f32,
    pub dy: f32,
    pub wheel: f32,
    pub drag: bool,
    pub reset: bool,
}

impl Default for PlotInput {
    fn default() -> Self {
        Self {
            x: -1.0,
            y: -1.0,
            dx: 0.0,
            dy: 0.0,
            wheel: 0.0,
            drag: false,
            reset: false,
        }
    }
}

pub struct PlotFrame {
    pub rgba: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub hover_hit: bool,
    pub hover_x: f32,
    pub hover_y: f32,
    pub series: Option<u32>,
}

// `id` on a mark is 0 for the grid and spines. Otherwise it is the series index
// across all panels, in scene order, plus 1. Hover reports `id - 1`.

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct LineRec {
    pub a: [f32; 3],
    pub b: [f32; 3],
    pub color: [f32; 4],
    pub thickness: f32,
    pub id: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PointRec {
    pub p: [f32; 3],
    pub radius: f32,
    pub color: [f32; 4],
    pub id: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct QuadRec {
    pub a: [f32; 2],
    pub b: [f32; 2],
    /// Third corner. `None` is an axis-aligned quad from `a` to `b`.
    pub c: Option<[f32; 2]>,
    /// 1 samples `heatmap`, 0 fills with `color`.
    pub heat: f32,
    pub color: [f32; 4],
    pub id: u32,
    /// Index into `Marks::heatmaps`.
    pub heatmap: Option<usize>,
}

struct GlyphRec {
    /// `w` > 0.5 anchors in data space. Screen-space text with `z` > 0.5 is rotated
    /// so the string runs up the y axis.
    anchor: [f32; 4],
    offset: [f32; 2],
    size: [f32; 2],
    uv: [f32; 4],
    color: [f32; 4],
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct PanelBatch {
    pub line_start: u32,
    pub line_count: u32,
    pub point_start: u32,
    pub point_count: u32,
    pub quad_start: u32,
    pub quad_count: u32,
    /// `(quad index, heatmap texture)` for each heatmap in the panel.
    pub heats: Vec<(u32, usize)>,
}

#[derive(Debug, PartialEq)]
pub(crate) struct Marks {
    pub lines: Vec<LineRec>,
    pub points: Vec<PointRec>,
    pub quads: Vec<QuadRec>,
    pub heatmaps: Vec<Vec<u8>>,
    pub heatmap_size: Vec<(u32, u32)>,
    pub panels: Vec<PanelBatch>,
}

struct Cell {
    cell: PlotRect,
    plot: PlotRect,
    panel: usize,
}

struct HeatGpu {
    group: wgpu::BindGroup,
    // The view inside `group` borrows this texture.
    #[allow(dead_code)]
    texture: wgpu::Texture,
}

struct GpuMarks {
    lines: wgpu::Buffer,
    points: wgpu::Buffer,
    quads: wgpu::Buffer,
    heats: Vec<HeatGpu>,
}

#[cfg(not(target_arch = "wasm32"))]
struct Offscreen {
    width: u32,
    height: u32,
    texture: wgpu::Texture,
    view: wgpu::TextureView,
}

pub(crate) const LINE_STRIDE: u64 = 64;
pub(crate) const POINT_STRIDE: u64 = 48;
pub(crate) const QUAD_STRIDE: u64 = 64;
const GLYPH_STRIDE: u64 = 64;

/// Canvas and offscreen sides are clamped to this range in device pixels.
pub(crate) const MIN_SIDE: u32 = 16;
pub(crate) const MAX_SIDE: u32 = 8192;

pub struct Plot {
    /// Panel titles, ranges, and category labels. Series live in `marks`.
    panels: Vec<Panel>,
    columns: u32,
    weights: Vec<f32>,
    row_weights: Vec<f32>,
    cameras: Vec<Camera>,
    home: Vec<Camera>,
    background: [f32; 4],
    foreground: [f32; 4],
    marks: Marks,
    /// Packed line, point, and quad bytes from the payload. Taken on upload.
    pub(crate) encoded: Option<crate::payload::EncodedMarks>,
    /// Heat textures reused when the next payload has the same pixels.
    kept_heats: Option<Vec<HeatGpu>>,
    gpu: Option<GpuMarks>,
    frame_buf: Option<wgpu::Buffer>,
    text_buf: Option<wgpu::Buffer>,
    uniform_buf: Option<wgpu::Buffer>,
    uniform_groups: Vec<wgpu::BindGroup>,
    mark_uploads: u32,
    size: (u32, u32),
    #[cfg(not(target_arch = "wasm32"))]
    offscreen: Option<Offscreen>,
    #[cfg(not(target_arch = "wasm32"))]
    stage: Option<wgpu::Buffer>,
}

#[cfg(not(target_arch = "wasm32"))]
pub fn render_plot(
    scene: &PlotScene,
    width: u32,
    height: u32,
    background: [f32; 4],
    foreground: [f32; 4],
) -> Result<Vec<u8>, String> {
    if scene.panels.is_empty() {
        return Ok(Vec::new());
    }
    Plot::new(scene, background, foreground)
        .draw(width, height, &PlotInput::default())
        .map(|frame| frame.rgba)
}

impl Plot {
    pub fn new(scene: &PlotScene, background: [f32; 4], foreground: [f32; 4]) -> Self {
        Self::from_parts(
            header_panels(&scene.panels),
            scene.columns,
            scene.column_weights.clone(),
            scene.row_weights.clone(),
            build_marks(&scene.panels),
            background,
            foreground,
        )
    }

    pub(crate) fn from_parts(
        panels: Vec<Panel>,
        columns: u32,
        weights: Vec<f32>,
        row_weights: Vec<f32>,
        marks: Marks,
        background: [f32; 4],
        foreground: [f32; 4],
    ) -> Self {
        let cameras = panels
            .iter()
            .map(|panel| Camera::new(panel.xmin, panel.xmax, panel.ymin, panel.ymax))
            .collect::<Vec<_>>();
        Self {
            marks,
            encoded: None,
            kept_heats: None,
            panels,
            columns,
            weights,
            row_weights,
            home: cameras.clone(),
            cameras,
            background,
            foreground,
            gpu: None,
            frame_buf: None,
            text_buf: None,
            uniform_buf: None,
            uniform_groups: Vec::new(),
            mark_uploads: 0,
            size: (0, 0),
            #[cfg(not(target_arch = "wasm32"))]
            offscreen: None,
            #[cfg(not(target_arch = "wasm32"))]
            stage: None,
        }
    }

    #[cfg(test)]
    fn mark_uploads(&self) -> u32 {
        self.mark_uploads
    }

    #[cfg(test)]
    fn zoom(&mut self, panel: usize, x: f32, y: f32, factor: f32) {
        if let Some(camera) = self.cameras.get_mut(panel) {
            camera.zoom_at(x, y, factor);
        }
    }

    /// Keep a zoomed window when the new panel is the same quantity and its
    /// data range is still close. A different label or a much larger span
    /// starts from the new limits. Extra panels keep their own fit.
    #[cfg_attr(not(any(test, target_arch = "wasm32")), allow(dead_code))]
    pub(crate) fn adopt_view(&mut self, previous: &Plot) {
        let count = self.cameras.len().min(previous.cameras.len());
        for index in 0..count {
            let panel = &self.panels[index];
            let previous_panel = &previous.panels[index];
            if panel.x_label != previous_panel.x_label || panel.y_label != previous_panel.y_label {
                continue;
            }
            // A fit the user has not touched tracks the new data, so a session
            // that arrives after the first partial stays on screen. A zoom or
            // pan is kept when the quantity and the span are still close.
            if previous.cameras[index] != previous.home[index]
                && axes_close(previous.home[index], self.home[index])
            {
                self.cameras[index] = previous.cameras[index];
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        self.panels.is_empty()
    }

    /// Resize, reset, pan, or zoom. `x`, `y`, `dx`, and `dy` are framebuffer pixels
    /// at `width` by `height`.
    pub fn apply(&mut self, width: u32, height: u32, input: &PlotInput) {
        let width = width.clamp(MIN_SIDE, MAX_SIDE);
        let height = height.clamp(MIN_SIDE, MAX_SIDE);
        self.size = (width, height);
        if input.reset {
            self.cameras.clone_from(&self.home);
        }
        self.apply_pointer(width, height, input);
    }

    pub fn size(&self) -> (u32, u32) {
        self.size
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn draw(
        &mut self,
        width: u32,
        height: u32,
        input: &PlotInput,
    ) -> Result<PlotFrame, String> {
        if self.panels.is_empty() {
            return Ok(PlotFrame {
                rgba: Vec::new(),
                width: 0,
                height: 0,
                hover_hit: false,
                hover_x: 0.0,
                hover_y: 0.0,
                series: None,
            });
        }
        self.apply(width, height, input);
        let (width, height) = self.size;
        let rgba = match crate::native_gpu() {
            Ok(gpu) => self
                .paint_offscreen(gpu, width, height)
                .map_err(|err| err.to_string())?,
            Err(_) => {
                let layout = self.layout(width, height);
                self.paint_cpu(width, height, &layout)
            }
        };
        let (hover_hit, hover_x, hover_y, series) = self.hover(input.x, input.y);
        Ok(PlotFrame {
            rgba,
            width,
            height,
            hover_hit,
            hover_x,
            hover_y,
            series,
        })
    }

    /// Data coordinates and the series under a framebuffer pixel of the last size.
    pub fn hover(&self, x: f32, y: f32) -> (bool, f32, f32, Option<u32>) {
        let (width, height) = self.size;
        if width == 0 {
            return (false, 0.0, 0.0, None);
        }
        let layout = self.layout(width, height);
        let Some(index) = hit_plot(&layout, x, y) else {
            return (false, 0.0, 0.0, None);
        };
        let cell = &layout[index];
        let data = self.cameras[cell.panel].data_at(x, y, cell.plot, width as f32, height as f32);
        (true, data[0], data[1], self.pick(cell, x, y, width, height))
    }

    /// Topmost series mark under the pixel, in paint order.
    // ponytail: linear scan over the panel's marks per hover. Past a few million
    // marks, bucket them into a per-panel screen grid on camera change.
    fn pick(&self, cell: &Cell, x: f32, y: f32, width: u32, height: u32) -> Option<u32> {
        let matrix =
            self.cameras[cell.panel].clip_from_data(cell.plot, width as f32, height as f32);
        let (w, h) = (width as f32, height as f32);
        let batch = &self.marks.panels[cell.panel];
        let hit = |id: u32| id.checked_sub(1);
        let points = &self.marks.points
            [batch.point_start as usize..(batch.point_start + batch.point_count) as usize];
        for point in points.iter().rev() {
            let center = project(matrix, point.p, w, h);
            let reach = point.radius.max(1.0) + 0.5;
            if (center[0] - x).powi(2) + (center[1] - y).powi(2) <= reach * reach {
                return hit(point.id);
            }
        }
        let lines = &self.marks.lines
            [batch.line_start as usize..(batch.line_start + batch.line_count) as usize];
        for line in lines.iter().rev() {
            let a = project(matrix, line.a, w, h);
            let b = project(matrix, line.b, w, h);
            if segment_distance(a, b, [x, y]) <= line.thickness.max(1.0) * 0.5 + 0.5 {
                return hit(line.id);
            }
        }
        let covers = |quad: &QuadRec| {
            let p0 = project(matrix, [quad.a[0], quad.a[1], 0.0], w, h);
            let p1 = project(matrix, [quad.b[0], quad.b[1], 0.0], w, h);
            if let Some(c) = quad.c {
                let p2 = project(matrix, [c[0], c[1], 0.0], w, h);
                return point_in_triangle([x, y], p0, p1, p2);
            }
            x >= p0[0].min(p1[0])
                && x <= p0[0].max(p1[0])
                && y >= p0[1].min(p1[1])
                && y <= p0[1].max(p1[1])
        };
        for (index, _) in batch.heats.iter().rev() {
            let quad = &self.marks.quads[*index as usize];
            if covers(quad) {
                return hit(quad.id);
            }
        }
        let quads = &self.marks.quads
            [batch.quad_start as usize..(batch.quad_start + batch.quad_count) as usize];
        quads
            .iter()
            .rev()
            .find(|quad| covers(quad))
            .and_then(|quad| hit(quad.id))
    }

    fn apply_pointer(&mut self, width: u32, height: u32, input: &PlotInput) {
        if !input.drag && input.wheel == 0.0 {
            return;
        }
        let layout = self.layout(width, height);
        let Some(index) = hit_cell(&layout, input.x, input.y) else {
            return;
        };
        let plot = layout[index].plot;
        if input.drag {
            self.cameras[index].pan_pixels(input.dx, input.dy, plot);
        }
        if input.wheel != 0.0 {
            let factor = (-input.wheel * 0.0015).exp();
            self.cameras[index].zoom_at_pixel(
                input.x,
                input.y,
                plot,
                width as f32,
                height as f32,
                factor,
            );
        }
    }

    fn layout(&self, width: u32, height: u32) -> Vec<Cell> {
        let rects = panel_rects(
            self.panels.len(),
            width,
            height,
            self.columns,
            &self.weights,
            &self.row_weights,
        );
        let font = atlas();
        rects
            .into_iter()
            .enumerate()
            .map(|(index, cell)| {
                let camera = self.cameras[index];
                let panel = &self.panels[index];
                let left = axis_name_width(&panel.y_label) + y_tick_width(&camera) + 8.0;
                let top = if shows_title(panel) {
                    font.line_height + 8.0
                } else {
                    6.0
                };
                let bottom = font.line_height
                    + 10.0
                    + if panel.x_label.is_empty() {
                        0.0
                    } else {
                        font.line_height + 4.0
                    };
                let mut plot = snap_plot(PlotRect {
                    x: cell.x + left,
                    y: cell.y + top,
                    w: (cell.w - left - 16.0).max(8.0),
                    h: (cell.h - top - bottom).max(8.0),
                });
                if panel.equal {
                    plot = snap_plot(equal_scale(plot, &camera));
                }
                Cell {
                    plot,
                    cell,
                    panel: index,
                }
            })
            .collect()
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn paint_cpu(&mut self, width: u32, height: u32, layout: &[Cell]) -> Vec<u8> {
        let mut frame = Vec::with_capacity((width * height * 4) as usize);
        let pixel = rgba_bytes(self.background);
        for _ in 0..width * height {
            frame.extend_from_slice(&pixel);
        }
        for cell in layout {
            let camera = self.cameras[cell.panel];
            let matrix = camera.clip_from_data(cell.plot, width as f32, height as f32);
            let batch = &self.marks.panels[cell.panel];
            for line in grid_lines(&camera, self.foreground) {
                stroke_cpu(&mut frame, width, height, &matrix, &line, cell.plot);
            }
            for quad in &self.marks.quads
                [batch.quad_start as usize..(batch.quad_start + batch.quad_count) as usize]
            {
                fill_quad_cpu(
                    &mut frame,
                    width,
                    height,
                    &matrix,
                    quad,
                    &self.marks,
                    cell.plot,
                );
            }
            for (index, _) in &batch.heats {
                fill_quad_cpu(
                    &mut frame,
                    width,
                    height,
                    &matrix,
                    &self.marks.quads[*index as usize],
                    &self.marks,
                    cell.plot,
                );
            }
            for line in &self.marks.lines
                [batch.line_start as usize..(batch.line_start + batch.line_count) as usize]
            {
                stroke_cpu(&mut frame, width, height, &matrix, line, cell.plot);
            }
            for point in &self.marks.points
                [batch.point_start as usize..(batch.point_start + batch.point_count) as usize]
            {
                let px = project(matrix, point.p, width as f32, height as f32);
                disc_cpu(
                    &mut frame,
                    width,
                    height,
                    px,
                    point.radius.max(1.0),
                    point.color,
                    cell.plot,
                );
            }
            for line in border_lines(&camera, &cell.plot, self.foreground) {
                stroke_cpu(&mut frame, width, height, &matrix, &line, cell.plot);
            }
            for glyph in labels_for(&self.panels[cell.panel], &camera, cell, self.foreground) {
                blit_glyph(&mut frame, width, height, &matrix, &glyph);
            }
        }
        frame
    }

    /// Draw into an offscreen target and read it back as tight RGBA rows.
    #[cfg(not(target_arch = "wasm32"))]
    fn paint_offscreen(
        &mut self,
        gpu: &PlotGpu,
        width: u32,
        height: u32,
    ) -> Result<Vec<u8>, GpuError> {
        if !self
            .offscreen
            .as_ref()
            .is_some_and(|target| target.width == width && target.height == height)
        {
            let texture = target_texture(&gpu.device, width, height, gpu.format);
            self.offscreen = Some(Offscreen {
                width,
                height,
                view: texture.create_view(&Default::default()),
                texture,
            });
        }
        let (texture, view) = {
            let target = self.offscreen.as_ref().unwrap();
            (target.texture.clone(), target.view.clone())
        };
        let mut encoder = self.record(gpu, &view, width, height)?;
        let padded = (width * 4).div_ceil(256) * 256;
        let bytes = u64::from(padded) * u64::from(height);
        if !self
            .stage
            .as_ref()
            .is_some_and(|buffer| buffer.size() >= bytes)
        {
            self.stage = Some(gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("readback"),
                size: bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }));
        }
        let stage = self.stage.as_ref().unwrap();
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: stage,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        gpu.queue.submit(Some(encoder.finish()));
        read_buffer(&gpu.device, stage, width, height, padded)
    }

    /// Record one frame into `view`. The caller submits it and, offscreen, reads it back.
    pub(crate) fn record(
        &mut self,
        gpu: &PlotGpu,
        view: &wgpu::TextureView,
        width: u32,
        height: u32,
    ) -> Result<wgpu::CommandEncoder, GpuError> {
        let layout = self.layout(width, height);
        self.ensure_uploaded(gpu)?;
        self.ensure_uniforms(gpu, layout.len());
        let frames = layout
            .iter()
            .map(|cell| {
                let camera = &self.cameras[cell.panel];
                (
                    grid_lines(camera, self.foreground),
                    border_lines(camera, &cell.plot, self.foreground),
                )
            })
            .collect::<Vec<_>>();
        let labels = layout
            .iter()
            .map(|cell| {
                labels_for(
                    &self.panels[cell.panel],
                    &self.cameras[cell.panel],
                    cell,
                    self.foreground,
                )
            })
            .collect::<Vec<_>>();
        let frame_bytes = frames
            .iter()
            .flat_map(|(grid, border)| encode_lines(grid).into_iter().chain(encode_lines(border)))
            .collect::<Vec<_>>();
        let text_bytes = labels
            .iter()
            .flat_map(|glyphs| encode_glyphs(glyphs))
            .collect::<Vec<_>>();
        write_grow(&gpu.device, &gpu.queue, &mut self.frame_buf, &frame_bytes);
        write_grow(&gpu.device, &gpu.queue, &mut self.text_buf, &text_bytes);
        let Some(uniform) = self.uniform_buf.as_ref() else {
            return Err(GpuError::Message("uniform buffer missing".into()));
        };
        for (index, cell) in layout.iter().enumerate() {
            let matrix =
                self.cameras[cell.panel].clip_from_data(cell.plot, width as f32, height as f32);
            let bytes = encode_uniform(matrix, width as f32, height as f32);
            gpu.queue
                .write_buffer(uniform, (index * 256) as u64, &bytes);
        }
        let Some(marks) = self.gpu.as_ref() else {
            return Err(GpuError::Message("marks not uploaded".into()));
        };
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("plot"),
                color_attachments: &[Some(color_attachment(view, self.background))],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            // WebGL2 has no base instance, so each range binds a slice and starts at 0.
            let mut frame_cursor = 0u64;
            let mut text_cursor = 0u64;
            for (index, cell) in layout.iter().enumerate() {
                let offset = (index * 256) as u32;
                let batch = &self.marks.panels[cell.panel];
                let (sx, sy, sw, sh) = scissor(cell.plot, width, height);
                pass.set_scissor_rect(sx, sy, sw, sh);
                pass.set_pipeline(&gpu.line_pipeline);
                pass.set_bind_group(0, &self.uniform_groups[index], &[offset]);
                let grid_count = frames[index].0.len() as u32;
                if let (true, Some(buffer)) = (grid_count > 0, self.frame_buf.as_ref()) {
                    pass.set_vertex_buffer(0, buffer.slice(frame_cursor * LINE_STRIDE..));
                    pass.draw(0..6, 0..grid_count);
                }
                frame_cursor += u64::from(grid_count);
                pass.set_pipeline(&gpu.quad_pipeline);
                pass.set_bind_group(0, &self.uniform_groups[index], &[offset]);
                if batch.quad_count > 0 {
                    pass.set_bind_group(1, &gpu.white_group, &[]);
                    pass.set_vertex_buffer(
                        0,
                        marks
                            .quads
                            .slice(u64::from(batch.quad_start) * QUAD_STRIDE..),
                    );
                    pass.draw(0..6, 0..batch.quad_count);
                }
                for (start, texture) in &batch.heats {
                    pass.set_bind_group(1, &marks.heats[*texture].group, &[]);
                    pass.set_vertex_buffer(0, marks.quads.slice(u64::from(*start) * QUAD_STRIDE..));
                    pass.draw(0..6, 0..1);
                }
                pass.set_pipeline(&gpu.line_pipeline);
                pass.set_bind_group(0, &self.uniform_groups[index], &[offset]);
                if batch.line_count > 0 {
                    pass.set_vertex_buffer(
                        0,
                        marks
                            .lines
                            .slice(u64::from(batch.line_start) * LINE_STRIDE..),
                    );
                    pass.draw(0..6, 0..batch.line_count);
                }
                if batch.point_count > 0 {
                    pass.set_pipeline(&gpu.point_pipeline);
                    pass.set_bind_group(0, &self.uniform_groups[index], &[offset]);
                    pass.set_vertex_buffer(
                        0,
                        marks
                            .points
                            .slice(u64::from(batch.point_start) * POINT_STRIDE..),
                    );
                    pass.draw(0..6, 0..batch.point_count);
                }
                let border_count = frames[index].1.len() as u32;
                if let (true, Some(buffer)) = (border_count > 0, self.frame_buf.as_ref()) {
                    pass.set_pipeline(&gpu.line_pipeline);
                    pass.set_bind_group(0, &self.uniform_groups[index], &[offset]);
                    pass.set_vertex_buffer(0, buffer.slice(frame_cursor * LINE_STRIDE..));
                    pass.draw(0..6, 0..border_count);
                }
                frame_cursor += u64::from(border_count);
                pass.set_scissor_rect(0, 0, width, height);
                let text_count = labels[index].len() as u32;
                if let (true, Some(buffer)) = (text_count > 0, self.text_buf.as_ref()) {
                    pass.set_pipeline(&gpu.text_pipeline);
                    pass.set_bind_group(0, &self.uniform_groups[index], &[offset]);
                    pass.set_bind_group(1, &gpu.atlas_group, &[]);
                    pass.set_vertex_buffer(0, buffer.slice(text_cursor * GLYPH_STRIDE..));
                    pass.draw(0..6, 0..text_count);
                }
                text_cursor += u64::from(text_count);
            }
        }
        Ok(encoder)
    }

    fn ensure_uniforms(&mut self, gpu: &PlotGpu, count: usize) {
        let needed = (count.max(1) * 256) as u64;
        if self
            .uniform_buf
            .as_ref()
            .is_some_and(|buffer| buffer.size() >= needed)
            && self.uniform_groups.len() == count
        {
            return;
        }
        let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uniforms"),
            size: needed,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.uniform_groups = (0..count)
            .map(|_| uniform_group(&gpu.device, &gpu.uniform_layout, &buffer))
            .collect();
        self.uniform_buf = Some(buffer);
    }

    /// Keep heatmap textures when the pixels did not change.
    #[cfg(target_arch = "wasm32")]
    pub(crate) fn keep_heatmaps(&mut self, mut previous: Plot) {
        if previous.marks.heatmaps != self.marks.heatmaps {
            return;
        }
        if let Some(gpu) = previous.gpu.take() {
            self.kept_heats = Some(gpu.heats);
        }
    }

    fn ensure_uploaded(&mut self, gpu: &PlotGpu) -> Result<(), GpuError> {
        if self.gpu.is_some() {
            return Ok(());
        }
        let heats = if let Some(heats) = self.kept_heats.take() {
            heats
        } else {
            self.marks
                .heatmaps
                .iter()
                .zip(&self.marks.heatmap_size)
                .map(|(pixels, (cols, rows))| {
                    let texture = upload_rgba(&gpu.device, &gpu.queue, pixels, *cols, *rows)?;
                    let group =
                        textured_group(&gpu.device, &gpu.textured_layout, &texture, &gpu.sampler);
                    Ok(HeatGpu { group, texture })
                })
                .collect::<Result<Vec<_>, GpuError>>()?
        };
        let (lines, points, quads) = if let Some(encoded) = self.encoded.take() {
            (encoded.lines, encoded.points, encoded.quads)
        } else {
            (
                encode_lines(&self.marks.lines),
                encode_points(&self.marks.points),
                encode_quads(&self.marks.quads),
            )
        };
        self.gpu = Some(GpuMarks {
            lines: upload_buffer(&gpu.device, &gpu.queue, "lines", &lines),
            points: upload_buffer(&gpu.device, &gpu.queue, "points", &points),
            quads: upload_buffer(&gpu.device, &gpu.queue, "quads", &quads),
            heats,
        });
        self.mark_uploads += 1;
        Ok(())
    }

    #[cfg(test)]
    fn project_data(&self, panel: usize, x: f32, y: f32, width: u32, height: u32) -> [f32; 2] {
        let layout = self.layout(width, height);
        let cell = &layout[panel];
        let matrix = self.cameras[panel].clip_from_data(cell.plot, width as f32, height as f32);
        project(matrix, [x, y, 0.0], width as f32, height as f32)
    }
}

/// Panel frames without their series. The marks carry the series.
pub(crate) fn header_panels(panels: &[Panel]) -> Vec<Panel> {
    panels
        .iter()
        .map(|panel| Panel {
            title: panel.title.clone(),
            y_label: panel.y_label.clone(),
            x_label: panel.x_label.clone(),
            xmin: panel.xmin,
            xmax: panel.xmax,
            ymin: panel.ymin,
            ymax: panel.ymax,
            series: Vec::new(),
            x_labels: panel.x_labels.clone(),
            equal: panel.equal,
        })
        .collect()
}

#[cfg(not(target_arch = "wasm32"))]
fn orient_px(a: [f32; 2], b: [f32; 2], p: [f32; 2]) -> f32 {
    (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
}

#[cfg(not(target_arch = "wasm32"))]
fn is_top_left_px(a: [f32; 2], b: [f32; 2]) -> bool {
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    dy > 0.0 || (dy == 0.0 && dx < 0.0)
}

#[cfg(not(target_arch = "wasm32"))]
fn same_side_px(w: f32, area: f32, a: [f32; 2], b: [f32; 2]) -> bool {
    if w * area > 0.0 {
        return true;
    }
    if w != 0.0 {
        return false;
    }
    if area > 0.0 {
        is_top_left_px(a, b)
    } else {
        is_top_left_px(b, a)
    }
}

/// Screen y grows down. Matches `owns_pixel` in the plot shader: a shared edge
/// belongs to one triangle, so a transparent fill is not composited twice.
#[cfg(not(target_arch = "wasm32"))]
fn owns_pixel_px(p: [f32; 2], a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> bool {
    let area = orient_px(a, b, c);
    if area == 0.0 {
        return false;
    }
    same_side_px(orient_px(a, b, p), area, a, b)
        && same_side_px(orient_px(b, c, p), area, b, c)
        && same_side_px(orient_px(c, a, p), area, c, a)
}

fn point_in_triangle(p: [f32; 2], a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> bool {
    let sign = |p: [f32; 2], a: [f32; 2], b: [f32; 2]| {
        (p[0] - b[0]) * (a[1] - b[1]) - (a[0] - b[0]) * (p[1] - b[1])
    };
    let d1 = sign(p, a, b);
    let d2 = sign(p, b, c);
    let d3 = sign(p, c, a);
    let has_neg = d1 < 0.0 || d2 < 0.0 || d3 < 0.0;
    let has_pos = d1 > 0.0 || d2 > 0.0 || d3 > 0.0;
    !(has_neg && has_pos)
}

fn segment_distance(a: [f32; 2], b: [f32; 2], p: [f32; 2]) -> f32 {
    let ab = [b[0] - a[0], b[1] - a[1]];
    let len2 = (ab[0] * ab[0] + ab[1] * ab[1]).max(1.0e-4);
    let t = (((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1]) / len2).clamp(0.0, 1.0);
    let dx = p[0] - (a[0] + ab[0] * t);
    let dy = p[1] - (a[1] + ab[1] * t);
    (dx * dx + dy * dy).sqrt()
}

/// Device, pipelines, and the glyph atlas for one color format.
pub(crate) struct PlotGpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    pub format: wgpu::TextureFormat,
    uniform_layout: wgpu::BindGroupLayout,
    textured_layout: wgpu::BindGroupLayout,
    line_pipeline: wgpu::RenderPipeline,
    point_pipeline: wgpu::RenderPipeline,
    quad_pipeline: wgpu::RenderPipeline,
    text_pipeline: wgpu::RenderPipeline,
    sampler: wgpu::Sampler,
    white_group: wgpu::BindGroup,
    atlas_group: wgpu::BindGroup,
    // The views inside the groups borrow these textures.
    #[allow(dead_code)]
    white: wgpu::Texture,
    #[allow(dead_code)]
    atlas_tex: wgpu::Texture,
}

impl PlotGpu {
    pub(crate) fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        format: wgpu::TextureFormat,
    ) -> Result<Self, GpuError> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("plot"),
            source: wgpu::ShaderSource::Wgsl(PLOT_SHADER.into()),
        });
        let uniform_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("uniform"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: Some(NonZeroU64::new(80).unwrap()),
                },
                count: None,
            }],
        });
        let textured_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("texture"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let line_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("lines"),
            bind_group_layouts: &[&uniform_layout],
            push_constant_ranges: &[],
        });
        let shaded_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shaded"),
            bind_group_layouts: &[&uniform_layout, &textured_layout],
            push_constant_ranges: &[],
        });
        let make = |layout: &wgpu::PipelineLayout,
                    vs: &str,
                    fs: &str,
                    buffer: wgpu::VertexBufferLayout<'_>| {
            pipeline(&device, &shader, layout, vs, fs, &[buffer], format)
        };
        let line_pipeline = make(&line_layout, "vs_line", "fs_line", line_vertex());
        let point_pipeline = make(&line_layout, "vs_point", "fs_point", point_vertex());
        let quad_pipeline = make(&shaded_layout, "vs_quad", "fs_quad", quad_vertex());
        let text_pipeline = make(&shaded_layout, "vs_text", "fs_text", text_vertex());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("plot"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let font = atlas();
        let atlas_tex = upload_r8(&device, &queue, &font.pixels, font.size, font.size)?;
        let white = upload_r8(&device, &queue, &[255], 1, 1)?;
        let atlas_group = textured_group(&device, &textured_layout, &atlas_tex, &sampler);
        let white_group = textured_group(&device, &textured_layout, &white, &sampler);
        Ok(Self {
            device,
            queue,
            format,
            uniform_layout,
            textured_layout,
            line_pipeline,
            point_pipeline,
            quad_pipeline,
            text_pipeline,
            sampler,
            white_group,
            atlas_group,
            white,
            atlas_tex,
        })
    }
}

fn pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    layout: &wgpu::PipelineLayout,
    vs: &str,
    fs: &str,
    buffers: &[wgpu::VertexBufferLayout<'_>],
    format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(vs),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some(vs),
            compilation_options: Default::default(),
            buffers,
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some(fs),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    })
}

// Byte strides keep the per-mark `id` word. Only CPU picking reads it.
const _: () = assert!(fits(&LINE_ATTRS, LINE_STRIDE));
const _: () = assert!(fits(&POINT_ATTRS, POINT_STRIDE));
const _: () = assert!(fits(&QUAD_ATTRS, QUAD_STRIDE));
const _: () = assert!(fits(&TEXT_ATTRS, GLYPH_STRIDE));

const fn fits(attrs: &[wgpu::VertexAttribute], stride: u64) -> bool {
    let mut index = 0;
    while index < attrs.len() {
        if attrs[index].offset + attrs[index].format.size() > stride {
            return false;
        }
        index += 1;
    }
    true
}

const LINE_ATTRS: [wgpu::VertexAttribute; 4] = wgpu::vertex_attr_array![
    0 => Float32x4,
    1 => Float32x4,
    2 => Float32x4,
    3 => Float32,
];
const POINT_ATTRS: [wgpu::VertexAttribute; 2] = wgpu::vertex_attr_array![
    0 => Float32x4,
    1 => Float32x4,
];
const QUAD_ATTRS: [wgpu::VertexAttribute; 3] = wgpu::vertex_attr_array![
    0 => Float32x4,
    1 => Float32x4,
    2 => Float32x4,
];
const TEXT_ATTRS: [wgpu::VertexAttribute; 4] = wgpu::vertex_attr_array![
    0 => Float32x4,
    1 => Float32x4,
    2 => Float32x4,
    3 => Float32x4,
];

fn line_vertex() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: LINE_STRIDE,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &LINE_ATTRS,
    }
}

fn point_vertex() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: POINT_STRIDE,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &POINT_ATTRS,
    }
}

fn quad_vertex() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: QUAD_STRIDE,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &QUAD_ATTRS,
    }
}

fn text_vertex() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: GLYPH_STRIDE,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &TEXT_ATTRS,
    }
}

pub(crate) fn build_marks(panels: &[Panel]) -> Marks {
    let mut lines = Vec::new();
    let mut points = Vec::new();
    let mut quads = Vec::new();
    let mut heatmaps = Vec::new();
    let mut heatmap_size = Vec::new();
    let mut batches = Vec::new();
    let mut next_id = 1u32;
    for panel in panels {
        let line_start = lines.len() as u32;
        let point_start = points.len() as u32;
        let quad_start = quads.len() as u32;
        let mut pending = Vec::new();
        for series in &panel.series {
            let id = next_id;
            next_id = next_id.saturating_add(1);
            match series {
                Series::Polyline {
                    xs,
                    ys,
                    color,
                    thickness,
                }
                | Series::Guide {
                    xs,
                    ys,
                    color,
                    thickness,
                } => {
                    push_polyline(&mut lines, xs, ys, *color, *thickness, id);
                }
                Series::Points {
                    xs,
                    ys,
                    color,
                    radius,
                } => {
                    for (x, y) in xs.iter().zip(ys) {
                        if x.is_finite() && y.is_finite() {
                            points.push(PointRec {
                                p: [*x, *y, 0.0],
                                radius: *radius,
                                color: *color,
                                id,
                            });
                        }
                    }
                }
                Series::Bars {
                    edges,
                    counts,
                    color,
                } => {
                    let baseline = if panel.ymin <= 0.0 && panel.ymax >= 0.0 {
                        0.0
                    } else {
                        panel.ymin
                    };
                    for (index, count) in counts.iter().copied().enumerate() {
                        let (Some(left), Some(right)) =
                            (edges.get(index).copied(), edges.get(index + 1).copied())
                        else {
                            continue;
                        };
                        if left.is_finite() && right.is_finite() && count.is_finite() {
                            quads.push(solid_quad([left, baseline], [right, count], *color, id));
                        }
                    }
                }
                Series::Rects { x, y, w, h, color } => {
                    for (((left, bottom), width), height) in x.iter().zip(y).zip(w).zip(h) {
                        if left.is_finite()
                            && bottom.is_finite()
                            && width.is_finite()
                            && height.is_finite()
                        {
                            quads.push(solid_quad(
                                [*left, *bottom],
                                [*left + *width, *bottom + *height],
                                *color,
                                id,
                            ));
                        }
                    }
                }
                Series::Triangles { xs, ys, color } => {
                    let mut index = 0;
                    while index + 2 < xs.len() && index + 2 < ys.len() {
                        let corners = [
                            (xs[index], ys[index]),
                            (xs[index + 1], ys[index + 1]),
                            (xs[index + 2], ys[index + 2]),
                        ];
                        index += 3;
                        if corners.iter().all(|(x, y)| x.is_finite() && y.is_finite()) {
                            quads.push(QuadRec {
                                a: [corners[0].0, corners[0].1],
                                b: [corners[1].0, corners[1].1],
                                c: Some([corners[2].0, corners[2].1]),
                                heat: 0.0,
                                color: *color,
                                id,
                                heatmap: None,
                            });
                        }
                    }
                }
                Series::Heatmap {
                    values,
                    cols,
                    rows,
                    ramp,
                    color,
                    lo,
                    hi,
                } => {
                    if *cols == 0 || *rows == 0 {
                        continue;
                    }
                    let pixels = heatmap_bytes(values, *cols, *rows, *ramp, *color, *lo, *hi);
                    if pixels.is_empty() {
                        continue;
                    }
                    pending.push((heatmaps.len(), id, 1.0, *color));
                    heatmaps.push(pixels);
                    heatmap_size.push((*cols, *rows));
                }
            }
        }
        let quad_count = quads.len() as u32 - quad_start;
        let mut heats = Vec::new();
        for (texture, id, heat, color) in pending {
            let start = quads.len() as u32;
            quads.push(QuadRec {
                a: [panel.xmin, panel.ymin],
                b: [panel.xmax, panel.ymax],
                c: None,
                heat,
                color,
                id,
                heatmap: Some(texture),
            });
            heats.push((start, texture));
        }
        batches.push(PanelBatch {
            line_start,
            line_count: lines.len() as u32 - line_start,
            point_start,
            point_count: points.len() as u32 - point_start,
            quad_start,
            quad_count,
            heats,
        });
    }
    Marks {
        lines,
        points,
        quads,
        heatmaps,
        heatmap_size,
        panels: batches,
    }
}

fn solid_quad(a: [f32; 2], b: [f32; 2], color: [f32; 4], id: u32) -> QuadRec {
    QuadRec {
        a,
        b,
        c: None,
        heat: 0.0,
        color,
        id,
        heatmap: None,
    }
}

fn push_polyline(
    lines: &mut Vec<LineRec>,
    xs: &[f32],
    ys: &[f32],
    color: [f32; 4],
    thickness: f32,
    id: u32,
) {
    let mut prev: Option<[f32; 3]> = None;
    for (x, y) in xs.iter().zip(ys) {
        if !x.is_finite() || !y.is_finite() {
            prev = None;
            continue;
        }
        let point = [*x, *y, 0.0];
        if let Some(start) = prev {
            lines.push(LineRec {
                a: start,
                b: point,
                color,
                thickness: thickness.max(1.0),
                id,
            });
        }
        prev = Some(point);
    }
}

fn heatmap_bytes(
    values: &[f32],
    cols: u32,
    rows: u32,
    ramp: u8,
    tint: [f32; 4],
    window_lo: f32,
    window_hi: f32,
) -> Vec<u8> {
    let (lo, mut hi) = if window_hi > window_lo {
        (window_lo, window_hi)
    } else {
        let mut lo = f32::MAX;
        let mut hi = f32::MIN;
        for value in values {
            if value.is_finite() {
                lo = lo.min(*value);
                hi = hi.max(*value);
            }
        }
        (lo, hi)
    };
    if !lo.is_finite() || !hi.is_finite() {
        return Vec::new();
    }
    if (hi - lo).abs() < 1.0e-8 {
        hi = lo + 1.0;
    }
    let cols_n = cols as usize;
    let rows_n = rows as usize;
    if cols_n == 0 || rows_n == 0 {
        return Vec::new();
    }
    let mut pixels = vec![0u8; cols_n * rows_n * 4];
    for (index, value) in values.iter().take(cols_n * rows_n).enumerate() {
        let t = if value.is_finite() {
            ((*value - lo) / (hi - lo)).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let row = index / cols_n;
        let col = index % cols_n;
        // The sampler treats pixel row 0 as the top of the plot.
        let dst = ((rows_n - 1 - row) * cols_n + col) * 4;
        pixels[dst..dst + 4].copy_from_slice(&ramp_pixel(ramp, tint, t));
    }
    pixels
}

fn ramp_pixel(ramp: u8, tint: [f32; 4], t: f32) -> [u8; 4] {
    let (rgb, alpha) = if scan_kit_core::is_session(ramp) {
        let color = scan_kit_core::session([tint[0], tint[1], tint[2]], t);
        ([color[0], color[1], color[2]], color[3] * tint[3])
    } else {
        (scan_kit_core::sample(ramp, t), 1.0)
    };
    [byte(rgb[0]), byte(rgb[1]), byte(rgb[2]), byte(alpha)]
}

fn byte(channel: f32) -> u8 {
    (channel.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn grid_lines(camera: &Camera, foreground: [f32; 4]) -> Vec<LineRec> {
    let mut lines = Vec::new();
    let grid = grid_color(foreground);
    for tick in ticks(camera.xmin, camera.xmax) {
        lines.push(axis_line(
            [tick, camera.ymin, 0.0],
            [tick, camera.ymax, 0.0],
            grid,
            1.0,
        ));
    }
    for tick in ticks(camera.ymin, camera.ymax) {
        lines.push(axis_line(
            [camera.xmin, tick, 0.0],
            [camera.xmax, tick, 0.0],
            grid,
            1.0,
        ));
    }
    lines
}

/// Whole-pixel plot box. The scissor, the camera, and the frame share it, so a
/// 1px stroke on the outer pixel is not cut off when the cell size is fractional.
/// Shrink the plot so one data unit is the same length on both axes.
fn equal_scale(plot: PlotRect, camera: &Camera) -> PlotRect {
    let data_w = (camera.xmax - camera.xmin).abs().max(1e-6);
    let data_h = (camera.ymax - camera.ymin).abs().max(1e-6);
    let data_aspect = data_w / data_h;
    let pixel_aspect = plot.w / plot.h.max(1.0);
    if pixel_aspect > data_aspect {
        let w = (plot.h * data_aspect).max(1.0).min(plot.w);
        PlotRect {
            x: plot.x + (plot.w - w) * 0.5,
            y: plot.y,
            w,
            h: plot.h,
        }
    } else {
        let h = (plot.w / data_aspect).max(1.0).min(plot.h);
        PlotRect {
            x: plot.x,
            y: plot.y + (plot.h - h) * 0.5,
            w: plot.w,
            h,
        }
    }
}

fn snap_plot(plot: PlotRect) -> PlotRect {
    let x = plot.x.floor();
    let y = plot.y.floor();
    let right = (plot.x + plot.w).floor().max(x + 1.0);
    let bottom = (plot.y + plot.h).floor().max(y + 1.0);
    PlotRect {
        x,
        y,
        w: right - x,
        h: bottom - y,
    }
}

/// A 1px rectangle on the plot's outer pixels. Half a pixel in lands on those
/// pixel centers; a full pixel in used to sit on the boundary and get clipped.
fn border_lines(camera: &Camera, plot: &PlotRect, foreground: [f32; 4]) -> Vec<LineRec> {
    let span_x = camera.xmax - camera.xmin;
    let span_y = camera.ymax - camera.ymin;
    let inset_x = 0.5 * span_x.abs() / plot.w.max(1.0);
    let inset_y = 0.5 * span_y.abs() / plot.h.max(1.0);
    let x0 = camera.xmin + inset_x;
    let x1 = camera.xmax - inset_x;
    let y0 = camera.ymin + inset_y;
    let y1 = camera.ymax - inset_y;
    let spine = frame_color(foreground);
    vec![
        axis_line([x0, y0, 0.0], [x1, y0, 0.0], spine, 1.0),
        axis_line([x1, y0, 0.0], [x1, y1, 0.0], spine, 1.0),
        axis_line([x1, y1, 0.0], [x0, y1, 0.0], spine, 1.0),
        axis_line([x0, y1, 0.0], [x0, y0, 0.0], spine, 1.0),
    ]
}

fn frame_color(foreground: [f32; 4]) -> [f32; 4] {
    [foreground[0], foreground[1], foreground[2], 0.45]
}

fn grid_color(foreground: [f32; 4]) -> [f32; 4] {
    [foreground[0], foreground[1], foreground[2], 0.3]
}

fn axis_line(a: [f32; 3], b: [f32; 3], color: [f32; 4], thickness: f32) -> LineRec {
    LineRec {
        a,
        b,
        color,
        thickness,
        id: 0,
    }
}

fn shows_title(panel: &Panel) -> bool {
    !panel.title.is_empty() && !panel.y_label.starts_with(&panel.title)
}

fn labels_for(panel: &Panel, camera: &Camera, cell: &Cell, color: [f32; 4]) -> Vec<GlyphRec> {
    let mut out = Vec::new();
    let font = atlas();
    if shows_title(panel) {
        push_text(
            &mut out,
            &panel.title,
            [cell.plot.x, cell.cell.y + font.ascent, 0.0],
            0.0,
            0.0,
            [0.0, 0.0],
            color,
        );
    }
    if !panel.y_label.is_empty() {
        push_up_text(
            &mut out,
            &panel.y_label,
            [
                cell.cell.x + axis_name_width(&panel.y_label) * 0.5,
                cell.plot.y + cell.plot.h * 0.5,
            ],
            color,
        );
    }
    for tick in ticks(camera.ymin, camera.ymax) {
        let label = format_tick(tick);
        push_text(
            &mut out,
            &label,
            [camera.xmin, tick, 0.0],
            1.0,
            1.0,
            [-4.0, ink_center_shift(&label)],
            color,
        );
    }
    if panel.x_labels.is_empty() {
        for tick in ticks(camera.xmin, camera.xmax) {
            push_text(
                &mut out,
                &format_tick(tick),
                [tick, camera.ymin, 0.0],
                1.0,
                0.5,
                [0.0, 4.0 + font.ascent],
                color,
            );
        }
    } else {
        let pitch = (font.line_height * 4.0).max(1.0);
        let stride = panel
            .x_labels
            .len()
            .div_ceil((cell.plot.w / pitch).max(1.0) as usize)
            .max(1);
        for (index, label) in panel.x_labels.iter().enumerate() {
            if index % stride != 0 || label.is_empty() {
                continue;
            }
            push_text(
                &mut out,
                label,
                [index as f32, camera.ymin, 0.0],
                1.0,
                0.5,
                [0.0, 4.0 + font.ascent],
                color,
            );
        }
    }
    if !panel.x_label.is_empty() {
        push_text(
            &mut out,
            &panel.x_label,
            [
                cell.plot.x + cell.plot.w * 0.5,
                x_axis_name_y(&cell.plot),
                0.0,
            ],
            0.0,
            0.5,
            [0.0, ink_center_shift(&panel.x_label)],
            color,
        );
    }
    out
}

/// Screen y of the x-axis name, centered in the band under the tick labels.
fn x_axis_name_y(plot: &PlotRect) -> f32 {
    let font = atlas();
    plot.y + plot.h + font.line_height + 10.0 + font.line_height * 0.5
}

fn push_text(
    out: &mut Vec<GlyphRec>,
    text: &str,
    anchor: [f32; 3],
    mode: f32,
    align_x: f32,
    extra: [f32; 2],
    color: [f32; 4],
) {
    let (stamps, width) = text::layout(text);
    let shift_x = extra[0] - width * align_x;
    for stamp in stamps {
        out.push(GlyphRec {
            anchor: [anchor[0], anchor[1], anchor[2], mode],
            offset: [shift_x + stamp.x, extra[1] + stamp.y],
            size: [stamp.w, stamp.h],
            uv: stamp.uv,
            color,
        });
    }
}

fn y_tick_width(camera: &Camera) -> f32 {
    ticks(camera.ymin, camera.ymax)
        .into_iter()
        .map(|value| text::text_width(&format_tick(value)))
        .fold(0.0, f32::max)
}

fn axis_name_width(label: &str) -> f32 {
    if label.is_empty() {
        0.0
    } else {
        atlas().line_height + 4.0
    }
}

/// Shift that puts the ink's vertical center on the anchor. Screen y grows down.
fn ink_center_shift(text: &str) -> f32 {
    let (stamps, _) = text::layout(text);
    if stamps.is_empty() {
        return 0.0;
    }
    let top = stamps.iter().map(|stamp| stamp.y).fold(f32::MAX, f32::min);
    let bottom = stamps
        .iter()
        .map(|stamp| stamp.y + stamp.h)
        .fold(f32::MIN, f32::max);
    -0.5 * (top + bottom)
}

fn push_up_text(out: &mut Vec<GlyphRec>, text: &str, anchor: [f32; 2], color: [f32; 4]) {
    let (stamps, width) = text::layout(text);
    if stamps.is_empty() {
        return;
    }
    let top = stamps.iter().map(|stamp| stamp.y).fold(f32::MAX, f32::min);
    let bottom = stamps
        .iter()
        .map(|stamp| stamp.y + stamp.h)
        .fold(f32::MIN, f32::max);
    let mid_y = 0.5 * (top + bottom);
    for stamp in stamps {
        out.push(GlyphRec {
            anchor: [anchor[0], anchor[1], 1.0, 0.0],
            offset: [stamp.x - width * 0.5, stamp.y - mid_y],
            size: [stamp.w, stamp.h],
            uv: stamp.uv,
            color,
        });
    }
}

fn panel_rects(
    count: usize,
    width: u32,
    height: u32,
    columns: u32,
    weights: &[f32],
    row_weights: &[f32],
) -> Vec<PlotRect> {
    if count == 0 {
        return Vec::new();
    }
    let cols = if columns == 0 {
        (count as f32).sqrt().ceil() as usize
    } else {
        columns.max(1) as usize
    };
    let rows = count.div_ceil(cols);
    let gap = 12.0f32;
    let margin = 8.0f32;
    // Extra room on the right so the 1px frame is not cut by the canvas edge.
    let margin_right = 16.0f32;
    let inner_w = width as f32 - margin - margin_right - gap * (cols.saturating_sub(1) as f32);
    let col_weight = if weights.len() == cols {
        weights.to_vec()
    } else {
        vec![1.0; cols]
    };
    let weight_sum = col_weight.iter().sum::<f32>().max(1.0e-6);
    let mut col_x = Vec::with_capacity(cols);
    let mut col_w = Vec::with_capacity(cols);
    let mut x = margin;
    for (index, weight) in col_weight.iter().enumerate() {
        let cell_w = inner_w * weight / weight_sum;
        col_x.push(x);
        col_w.push(cell_w);
        x += cell_w;
        if index + 1 < cols {
            x += gap;
        }
    }
    let row_weight = if row_weights.len() == rows {
        row_weights.to_vec()
    } else {
        vec![1.0; rows]
    };
    let row_sum = row_weight.iter().sum::<f32>().max(1.0e-6);
    let inner_h = height as f32 - margin * 2.0 - gap * (rows.saturating_sub(1) as f32);
    let mut row_y = Vec::with_capacity(rows);
    let mut row_h = Vec::with_capacity(rows);
    let mut y = margin;
    for (index, weight) in row_weight.iter().enumerate() {
        let cell_h = inner_h * weight / row_sum;
        row_y.push(y);
        row_h.push(cell_h);
        y += cell_h;
        if index + 1 < rows {
            y += gap;
        }
    }
    (0..count)
        .map(|index| {
            let col = index % cols;
            let row = index / cols;
            PlotRect {
                x: col_x[col],
                y: row_y[row],
                w: col_w[col],
                h: row_h[row],
            }
        })
        .collect()
}

fn hit_cell(layout: &[Cell], x: f32, y: f32) -> Option<usize> {
    layout.iter().position(|cell| inside(cell.cell, x, y))
}

fn hit_plot(layout: &[Cell], x: f32, y: f32) -> Option<usize> {
    layout.iter().position(|cell| inside(cell.plot, x, y))
}

fn inside(rect: PlotRect, x: f32, y: f32) -> bool {
    x >= rect.x && y >= rect.y && x < rect.x + rect.w && y < rect.y + rect.h
}

fn scissor(plot: PlotRect, width: u32, height: u32) -> (u32, u32, u32, u32) {
    // `snap_plot` makes this an exact pixel range. Truncation is that range.
    let x = (plot.x.max(0.0) as u32).min(width.saturating_sub(1));
    let y = (plot.y.max(0.0) as u32).min(height.saturating_sub(1));
    let w = (plot.w.max(1.0) as u32).min(width.saturating_sub(x)).max(1);
    let h = (plot.h.max(1.0) as u32)
        .min(height.saturating_sub(y))
        .max(1);
    (x, y, w, h)
}

#[cfg(not(target_arch = "wasm32"))]
fn fill_quad_cpu(
    frame: &mut [u8],
    width: u32,
    height: u32,
    matrix: &[f32; 16],
    quad: &QuadRec,
    marks: &Marks,
    plot: PlotRect,
) {
    if let Some(c) = quad.c {
        let p0 = project(
            *matrix,
            [quad.a[0], quad.a[1], 0.0],
            width as f32,
            height as f32,
        );
        let p1 = project(
            *matrix,
            [quad.b[0], quad.b[1], 0.0],
            width as f32,
            height as f32,
        );
        let p2 = project(*matrix, [c[0], c[1], 0.0], width as f32, height as f32);
        let pts = [p0, p1, p2];
        let min_x = pts.iter().map(|p| p[0]).fold(f32::MAX, f32::min).floor() as i32;
        let max_x = pts.iter().map(|p| p[0]).fold(f32::MIN, f32::max).ceil() as i32;
        let min_y = pts.iter().map(|p| p[1]).fold(f32::MAX, f32::min).floor() as i32;
        let max_y = pts.iter().map(|p| p[1]).fold(f32::MIN, f32::max).ceil() as i32;
        for y in min_y..=max_y {
            for x in min_x..=max_x {
                if owns_pixel_px([x as f32 + 0.5, y as f32 + 0.5], pts[0], pts[1], pts[2]) {
                    blend(frame, width, height, x, y, quad.color, Some(plot));
                }
            }
        }
        return;
    }
    let p0 = project(
        *matrix,
        [quad.a[0], quad.a[1], 0.0],
        width as f32,
        height as f32,
    );
    let p1 = project(
        *matrix,
        [quad.b[0], quad.b[1], 0.0],
        width as f32,
        height as f32,
    );
    let left = p0[0].min(p1[0]);
    let right = p0[0].max(p1[0]);
    let top = p0[1].min(p1[1]);
    let bottom = p0[1].max(p1[1]);
    let span_x = (right - left).max(1.0);
    let span_y = (bottom - top).max(1.0);
    for y in top.floor() as i32..=bottom.ceil() as i32 {
        for x in left.floor() as i32..=right.ceil() as i32 {
            let color = if let Some(index) = quad.heatmap {
                let u = ((x as f32 - left) / span_x).clamp(0.0, 0.999);
                let v = ((y as f32 - top) / span_y).clamp(0.0, 0.999);
                let (cols, rows) = marks.heatmap_size[index];
                let col = ((u * cols as f32) as u32).min(cols.saturating_sub(1));
                let row = ((v * rows as f32) as u32).min(rows.saturating_sub(1));
                let texel = (row * cols + col) as usize * 4;
                let px = &marks.heatmaps[index];
                [
                    px[texel] as f32 / 255.0,
                    px[texel + 1] as f32 / 255.0,
                    px[texel + 2] as f32 / 255.0,
                    px[texel + 3] as f32 / 255.0,
                ]
            } else {
                quad.color
            };
            blend(frame, width, height, x, y, color, Some(plot));
        }
    }
}

/// Same rule as `vs_line`: an axis-aligned stroke moves onto one pixel center.
#[cfg(not(target_arch = "wasm32"))]
fn snap_hairline(pa: &mut [f32; 2], pb: &mut [f32; 2]) {
    let dx = (pa[0] - pb[0]).abs();
    let dy = (pa[1] - pb[1]).abs();
    if dx < 0.05 && dy >= dx {
        let x = ((pa[0] + pb[0]) * 0.5).floor() + 0.5;
        pa[0] = x;
        pb[0] = x;
    } else if dy < 0.05 {
        let y = ((pa[1] + pb[1]) * 0.5).floor() + 0.5;
        pa[1] = y;
        pb[1] = y;
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn stroke_cpu(
    frame: &mut [u8],
    width: u32,
    height: u32,
    matrix: &[f32; 16],
    line: &LineRec,
    plot: PlotRect,
) {
    let mut pa = project(*matrix, line.a, width as f32, height as f32);
    let mut pb = project(*matrix, line.b, width as f32, height as f32);
    snap_hairline(&mut pa, &mut pb);
    let radius = line.thickness.max(1.0) * 0.5;
    let steps = ((pa[0] - pb[0]).abs().max((pa[1] - pb[1]).abs()) as i32).max(1);
    for step in 0..=steps {
        let t = step as f32 / steps as f32;
        disc_cpu(
            frame,
            width,
            height,
            [pa[0] + (pb[0] - pa[0]) * t, pa[1] + (pb[1] - pa[1]) * t],
            radius,
            line.color,
            plot,
        );
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn disc_cpu(
    frame: &mut [u8],
    width: u32,
    height: u32,
    center: [f32; 2],
    radius: f32,
    color: [f32; 4],
    plot: PlotRect,
) {
    let r = radius.ceil() as i32 + 1;
    let cx = center[0].round() as i32;
    let cy = center[1].round() as i32;
    for y in cy - r..=cy + r {
        for x in cx - r..=cx + r {
            let dx = x as f32 + 0.5 - center[0];
            let dy = y as f32 + 0.5 - center[1];
            let dist = (dx * dx + dy * dy).sqrt();
            let coverage = (radius + 0.5 - dist).clamp(0.0, 1.0);
            if coverage <= 0.0 {
                continue;
            }
            let mut ink = color;
            ink[3] *= coverage;
            blend(frame, width, height, x, y, ink, Some(plot));
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn blit_glyph(frame: &mut [u8], width: u32, height: u32, matrix: &[f32; 16], glyph: &GlyphRec) {
    let origin = if glyph.anchor[3] > 0.5 {
        project(
            *matrix,
            [glyph.anchor[0], glyph.anchor[1], glyph.anchor[2]],
            width as f32,
            height as f32,
        )
    } else {
        [glyph.anchor[0], glyph.anchor[1]]
    };
    let font = atlas();
    let turned = glyph.anchor[3] < 0.5 && glyph.anchor[2] > 0.5;
    let u0 = (glyph.uv[0] * font.size as f32).round() as u32;
    let v0 = (glyph.uv[1] * font.size as f32).round() as u32;
    let columns = glyph.size[0] as u32;
    let rows = glyph.size[1] as u32;
    for row in 0..rows {
        for col in 0..columns {
            let coverage = font.pixels[((v0 + row) * font.size + u0 + col) as usize] as f32 / 255.0;
            if coverage < 0.004 {
                continue;
            }
            let local_x = glyph.offset[0] + col as f32;
            let local_y = glyph.offset[1] + row as f32;
            let (dx, dy) = if turned {
                (local_y, -local_x)
            } else {
                (local_x, local_y)
            };
            let mut color = glyph.color;
            color[3] *= coverage;
            blend(
                frame,
                width,
                height,
                (origin[0] + dx).floor() as i32,
                (origin[1] + dy).floor() as i32,
                color,
                None,
            );
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn blend(
    frame: &mut [u8],
    width: u32,
    height: u32,
    x: i32,
    y: i32,
    color: [f32; 4],
    plot: Option<PlotRect>,
) {
    if x < 0 || y < 0 || x >= width as i32 || y >= height as i32 {
        return;
    }
    if let Some(plot) = plot {
        let (sx, sy, sw, sh) = scissor(plot, width, height);
        let (x, y) = (x as u32, y as u32);
        if x < sx || y < sy || x >= sx + sw || y >= sy + sh {
            return;
        }
    }
    let index = (y as u32 * width + x as u32) as usize;
    if index >= frame.len() / 4 {
        return;
    }
    let alpha = color[3].clamp(0.0, 1.0);
    let pixel = index * 4;
    for channel in 0..3 {
        let src = color[channel].clamp(0.0, 1.0) * alpha * 255.0;
        let dst = frame[pixel + channel] as f32;
        frame[pixel + channel] = (src + dst * (1.0 - alpha)).round().clamp(0.0, 255.0) as u8;
    }
    frame[pixel + 3] = 255;
}

// Instance word layouts. Words past the `*_ATTRS` tables (`id`, the heatmap
// index, padding) never reach the shader. Only the payload and hover read them.

pub(crate) fn encode_lines(lines: &[LineRec]) -> Vec<u8> {
    let mut out = Vec::with_capacity(lines.len() * LINE_STRIDE as usize);
    for line in lines {
        push4(&mut out, [line.a[0], line.a[1], line.a[2], 0.0]);
        push4(&mut out, [line.b[0], line.b[1], line.b[2], 0.0]);
        push4(&mut out, line.color);
        push_f32(&mut out, line.thickness);
        push_u32(&mut out, line.id);
        push_u32(&mut out, 0);
        push_u32(&mut out, 0);
    }
    debug_assert_eq!(out.len(), lines.len() * LINE_STRIDE as usize);
    out
}

pub(crate) fn decode_lines(bytes: &[u8]) -> Vec<LineRec> {
    let (chunks, _) = bytes.as_chunks::<{ LINE_STRIDE as usize }>();
    chunks
        .iter()
        .map(|chunk| LineRec {
            a: [f32_at(chunk, 0), f32_at(chunk, 1), f32_at(chunk, 2)],
            b: [f32_at(chunk, 4), f32_at(chunk, 5), f32_at(chunk, 6)],
            color: f32x4_at(chunk, 8),
            thickness: f32_at(chunk, 12),
            id: u32_at(chunk, 13),
        })
        .collect()
}

pub(crate) fn encode_points(points: &[PointRec]) -> Vec<u8> {
    let mut out = Vec::with_capacity(points.len() * POINT_STRIDE as usize);
    for point in points {
        push4(&mut out, [point.p[0], point.p[1], point.p[2], point.radius]);
        push4(&mut out, point.color);
        push_u32(&mut out, point.id);
        push_u32(&mut out, 0);
        push_u32(&mut out, 0);
        push_u32(&mut out, 0);
    }
    debug_assert_eq!(out.len(), points.len() * POINT_STRIDE as usize);
    out
}

pub(crate) fn decode_points(bytes: &[u8]) -> Vec<PointRec> {
    let (chunks, _) = bytes.as_chunks::<{ POINT_STRIDE as usize }>();
    chunks
        .iter()
        .map(|chunk| PointRec {
            p: [f32_at(chunk, 0), f32_at(chunk, 1), f32_at(chunk, 2)],
            radius: f32_at(chunk, 3),
            color: f32x4_at(chunk, 4),
            id: u32_at(chunk, 8),
        })
        .collect()
}

pub(crate) fn encode_quads(quads: &[QuadRec]) -> Vec<u8> {
    let mut out = Vec::with_capacity(quads.len() * QUAD_STRIDE as usize);
    for quad in quads {
        let [cx, cy] = quad.c.unwrap_or([0.0, 0.0]);
        push4(
            &mut out,
            [
                quad.a[0],
                quad.a[1],
                quad.heat,
                if quad.c.is_some() { 1.0 } else { 0.0 },
            ],
        );
        push4(&mut out, [quad.b[0], quad.b[1], cx, cy]);
        push4(&mut out, quad.color);
        push_u32(&mut out, quad.id);
        // 0 is a solid quad, so the heatmap index is stored plus 1.
        push_u32(&mut out, quad.heatmap.map_or(0, |index| index as u32 + 1));
        push_u32(&mut out, 0);
        push_u32(&mut out, 0);
    }
    debug_assert_eq!(out.len(), quads.len() * QUAD_STRIDE as usize);
    out
}

pub(crate) fn decode_quads(bytes: &[u8]) -> Vec<QuadRec> {
    let (chunks, _) = bytes.as_chunks::<{ QUAD_STRIDE as usize }>();
    chunks
        .iter()
        .map(|chunk| QuadRec {
            a: [f32_at(chunk, 0), f32_at(chunk, 1)],
            heat: f32_at(chunk, 2),
            b: [f32_at(chunk, 4), f32_at(chunk, 5)],
            c: if f32_at(chunk, 3) > 0.5 {
                Some([f32_at(chunk, 6), f32_at(chunk, 7)])
            } else {
                None
            },
            color: f32x4_at(chunk, 8),
            id: u32_at(chunk, 12),
            heatmap: u32_at(chunk, 13).checked_sub(1).map(|index| index as usize),
        })
        .collect()
}

fn f32_at(chunk: &[u8], word: usize) -> f32 {
    f32::from_le_bytes(chunk[word * 4..word * 4 + 4].try_into().unwrap())
}

fn f32x4_at(chunk: &[u8], word: usize) -> [f32; 4] {
    std::array::from_fn(|offset| f32_at(chunk, word + offset))
}

fn u32_at(chunk: &[u8], word: usize) -> u32 {
    u32::from_le_bytes(chunk[word * 4..word * 4 + 4].try_into().unwrap())
}

fn encode_glyphs(glyphs: &[GlyphRec]) -> Vec<u8> {
    let mut out = Vec::with_capacity(glyphs.len() * GLYPH_STRIDE as usize);
    for glyph in glyphs {
        push4(&mut out, glyph.anchor);
        push4(
            &mut out,
            [
                glyph.offset[0],
                glyph.offset[1],
                glyph.size[0],
                glyph.size[1],
            ],
        );
        push4(&mut out, glyph.uv);
        push4(&mut out, glyph.color);
    }
    out
}

fn encode_uniform(matrix: [f32; 16], width: f32, height: f32) -> [u8; 80] {
    let mut out = [0u8; 80];
    for (index, value) in matrix.iter().enumerate() {
        out[index * 4..index * 4 + 4].copy_from_slice(&value.to_le_bytes());
    }
    out[64..68].copy_from_slice(&width.to_le_bytes());
    out[68..72].copy_from_slice(&height.to_le_bytes());
    out
}

fn push4(out: &mut Vec<u8>, value: [f32; 4]) {
    for channel in value {
        push_f32(out, channel);
    }
}

fn push_f32(out: &mut Vec<u8>, value: f32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn uniform_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("uniform"),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer,
                offset: 0,
                size: Some(NonZeroU64::new(80).unwrap()),
            }),
        }],
    })
}

fn textured_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    texture: &wgpu::Texture,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    let view = texture.create_view(&Default::default());
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("texture"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    })
}

fn upload_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    bytes: &[u8],
) -> wgpu::Buffer {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.len().max(4) as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    if !bytes.is_empty() {
        queue.write_buffer(&buffer, 0, bytes);
    }
    buffer
}

fn write_grow(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    slot: &mut Option<wgpu::Buffer>,
    bytes: &[u8],
) {
    if bytes.is_empty() {
        return;
    }
    let needed = bytes.len() as u64;
    if slot
        .as_ref()
        .map(|buffer| buffer.size() < needed)
        .unwrap_or(true)
    {
        *slot = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dynamic"),
            size: needed,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
    }
    queue.write_buffer(slot.as_ref().unwrap(), 0, bytes);
}

fn upload_rgba(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pixels: &[u8],
    cols: u32,
    rows: u32,
) -> Result<wgpu::Texture, GpuError> {
    let cols = cols.max(1);
    let rows = rows.max(1);
    let row_bytes = cols * 4;
    let stride = row_bytes.next_multiple_of(256);
    let mut padded = vec![0u8; (stride * rows) as usize];
    for row in 0..rows {
        let src = (row * row_bytes) as usize;
        let dst = (row * stride) as usize;
        let end = src + row_bytes as usize;
        if end <= pixels.len() {
            padded[dst..dst + row_bytes as usize].copy_from_slice(&pixels[src..end]);
        }
    }
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("rgba"),
        size: wgpu::Extent3d {
            width: cols,
            height: rows,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &padded,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(stride),
            rows_per_image: Some(rows),
        },
        wgpu::Extent3d {
            width: cols,
            height: rows,
            depth_or_array_layers: 1,
        },
    );
    Ok(texture)
}

fn upload_r8(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pixels: &[u8],
    cols: u32,
    rows: u32,
) -> Result<wgpu::Texture, GpuError> {
    let cols = cols.max(1);
    let rows = rows.max(1);
    let stride = cols.next_multiple_of(256);
    let mut padded = vec![0u8; (stride * rows) as usize];
    for row in 0..rows {
        let src = (row * cols) as usize;
        let dst = (row * stride) as usize;
        let end = src + cols as usize;
        if end <= pixels.len() {
            padded[dst..dst + cols as usize].copy_from_slice(&pixels[src..end]);
        }
    }
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("r8"),
        size: wgpu::Extent3d {
            width: cols,
            height: rows,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &padded,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(stride),
            rows_per_image: Some(rows),
        },
        wgpu::Extent3d {
            width: cols,
            height: rows,
            depth_or_array_layers: 1,
        },
    );
    Ok(texture)
}

#[cfg(not(target_arch = "wasm32"))]
fn target_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("target"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

fn color_attachment(
    view: &wgpu::TextureView,
    color: [f32; 4],
) -> wgpu::RenderPassColorAttachment<'_> {
    wgpu::RenderPassColorAttachment {
        view,
        resolve_target: None,
        ops: wgpu::Operations {
            load: wgpu::LoadOp::Clear(wgpu::Color {
                r: f64::from(color[0]),
                g: f64::from(color[1]),
                b: f64::from(color[2]),
                a: f64::from(color[3]),
            }),
            store: wgpu::StoreOp::Store,
        },
        depth_slice: None,
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn read_buffer(
    device: &wgpu::Device,
    readback: &wgpu::Buffer,
    width: u32,
    height: u32,
    padded: u32,
) -> Result<Vec<u8>, GpuError> {
    let slice = readback.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|err| GpuError::Message(err.to_string()))?;
    receiver
        .recv()
        .map_err(|err| GpuError::Message(err.to_string()))?
        .map_err(|err| GpuError::Message(err.to_string()))?;
    let mapped = slice.get_mapped_range();
    let mut frame = Vec::with_capacity((width * height * 4) as usize);
    for row in 0..height {
        let start = (row * padded) as usize;
        frame.extend_from_slice(&mapped[start..start + (width * 4) as usize]);
    }
    drop(mapped);
    readback.unmap();
    Ok(frame)
}

#[cfg(not(target_arch = "wasm32"))]
fn rgba_bytes(color: [f32; 4]) -> [u8; 4] {
    color.map(|channel| (channel.clamp(0.0, 1.0) * 255.0).round() as u8)
}

#[cfg_attr(not(any(test, target_arch = "wasm32")), allow(dead_code))]
fn axes_close(before: Camera, after: Camera) -> bool {
    axis_close(before.xmin, before.xmax, after.xmin, after.xmax)
        && axis_close(before.ymin, before.ymax, after.ymin, after.ymax)
}

#[cfg_attr(not(any(test, target_arch = "wasm32")), allow(dead_code))]
fn axis_close(a0: f32, a1: f32, b0: f32, b1: f32) -> bool {
    let a_span = (a1 - a0).abs().max(1.0e-6);
    let b_span = (b1 - b0).abs().max(1.0e-6);
    let ratio = (a_span / b_span).max(b_span / a_span);
    let mid_delta = ((a0 + a1) - (b0 + b1)).abs() * 0.5;
    ratio <= 3.0 && mid_delta <= a_span.max(b_span)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line_scene() -> PlotScene {
        PlotScene {
            title: "line".into(),
            panels: vec![Panel {
                title: String::new(),
                y_label: String::new(),
                x_label: String::new(),
                xmin: 0.0,
                xmax: 10.0,
                ymin: 0.0,
                ymax: 10.0,
                series: vec![Series::Polyline {
                    xs: vec![0.0, 10.0],
                    ys: vec![5.0, 5.0],
                    color: [1.0, 0.0, 0.0, 1.0],
                    thickness: 4.0,
                }],
                x_labels: Vec::new(),
                equal: false,
            }],
            controls: Vec::new(),
            table: None,
            samples: Vec::new(),
            columns: 0,
            column_weights: Vec::new(),
            row_weights: Vec::new(),
        }
    }

    #[test]
    fn the_top_row_can_take_half_the_height() {
        let mut scene = line_scene();
        let panel = scene.panels[0].clone();
        scene.panels.push(panel.clone());
        scene.panels.push(panel);
        scene.columns = 1;
        scene.row_weights = vec![2.0, 1.0, 1.0];
        let plot = Plot::new(&scene, [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]);
        let layout = plot.layout(400, 420);
        let top = layout[0].cell.h;
        let middle = layout[1].cell.h;
        let bottom = layout[2].cell.h;
        assert!(
            (top - (middle + bottom)).abs() < 1.0,
            "{top} {middle} {bottom}"
        );
        assert!((middle - bottom).abs() < 1.0, "{middle} {bottom}");
    }

    #[test]
    fn equal_axes_stay_square_in_a_wide_frame() {
        let mut scene = line_scene();
        scene.panels[0].equal = true;
        scene.panels[0].xmin = -2.0;
        scene.panels[0].xmax = 2.0;
        scene.panels[0].ymin = -2.0;
        scene.panels[0].ymax = 2.0;
        let plot = Plot::new(&scene, [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]);
        let frame = plot.layout(480, 200)[0].plot;
        assert!(
            (frame.w - frame.h).abs() <= 1.0,
            "{} x {}",
            frame.w,
            frame.h
        );
    }

    #[test]
    fn a_baked_heatmap_matches_the_catalog_and_keeps_low_y_last() {
        let turbo = scan_kit_core::index("turbo");
        let bytes = heatmap_bytes(&[1.0], 1, 1, turbo, [1.0; 4], 0.0, 1.0);
        let rgb = scan_kit_core::sample(turbo, 1.0);
        assert_eq!(bytes[0], byte(rgb[0]));
        assert_eq!(bytes[1], byte(rgb[1]));
        assert_eq!(bytes[2], byte(rgb[2]));
        assert_eq!(bytes[3], 255);
        let gray = scan_kit_core::index("gray");
        let white = [1.0, 1.0, 1.0, 1.0];
        let bytes = heatmap_bytes(&[0.0, 5.0], 2, 1, gray, white, 0.0, 5.0);
        assert_eq!(bytes, vec![0, 0, 0, 255, 255, 255, 255, 255]);
        // Value row 0 is the low y, so it is the last texture row.
        let bytes = heatmap_bytes(&[0.0, 5.0], 1, 2, gray, white, 0.0, 5.0);
        assert_eq!(bytes, vec![255, 255, 255, 255, 0, 0, 0, 255]);
    }

    #[test]
    fn the_cpu_fill_reads_the_baked_ramp() {
        let mut scene = line_scene();
        scene.panels[0].xmin = 0.0;
        scene.panels[0].xmax = 1.0;
        scene.panels[0].ymin = 0.0;
        scene.panels[0].ymax = 1.0;
        scene.panels[0].series = vec![Series::Heatmap {
            values: vec![1.0],
            cols: 1,
            rows: 1,
            ramp: scan_kit_core::index("gray"),
            color: [1.0; 4],
            lo: 0.0,
            hi: 1.0,
        }];
        let mut plot = Plot::new(&scene, [0.0, 0.0, 0.0, 1.0], [0.9, 0.9, 0.9, 1.0]);
        let layout = plot.layout(64, 64);
        let frame = plot.paint_cpu(64, 64, &layout);
        let rect = layout[0].plot;
        let x = (rect.x + rect.w * 0.5) as u32;
        let y = (rect.y + rect.h * 0.5) as u32;
        let pixel = (y * 64 + x) as usize * 4;
        assert_eq!(&frame[pixel..pixel + 3], &[255, 255, 255]);
    }

    #[test]
    fn a_reload_keeps_the_zoomed_window_when_the_axes_match() {
        let scene = line_scene();
        let mut first = Plot::new(&scene, [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]);
        first.zoom(0, 5.0, 5.0, 2.0);
        let zoomed = first.cameras[0];
        let mut second = Plot::new(&scene, [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]);
        second.adopt_view(&first);
        assert_eq!(second.cameras[0], zoomed);

        let mut nudged = line_scene();
        nudged.panels[0].xmax = 12.0;
        let mut kept = Plot::new(&nudged, [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]);
        kept.adopt_view(&first);
        assert_eq!(kept.cameras[0], zoomed);

        let mut shifted = line_scene();
        shifted.panels[0].xmax = 40.0;
        let mut third = Plot::new(&shifted, [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]);
        third.adopt_view(&first);
        assert_eq!(third.cameras[0], third.home[0]);

        let mut relabeled = line_scene();
        relabeled.panels[0].y_label = "IC1 (nA)".into();
        let mut renamed = Plot::new(&relabeled, [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]);
        renamed.adopt_view(&first);
        assert_eq!(renamed.cameras[0], renamed.home[0]);

        let mut extra = line_scene();
        extra.panels.push(extra.panels[0].clone());
        let mut added = Plot::new(&extra, [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]);
        added.adopt_view(&first);
        assert_eq!(added.cameras[0], zoomed);
        assert_eq!(added.cameras[1], added.home[1]);

        let plain = Plot::new(&scene, [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]);
        let mut grown = line_scene();
        grown.panels[0].xmax = 18.0;
        grown.panels[0].ymax = 18.0;
        let mut refit = Plot::new(&grown, [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]);
        refit.adopt_view(&plain);
        assert_eq!(refit.cameras[0], refit.home[0]);
        assert!(refit.cameras[0].xmax > plain.cameras[0].xmax);
    }

    #[test]
    fn sk_req_009_plot_shader_compiles_and_draws_a_line() {
        compile_plot_shader().unwrap();
        let frame = render_plot(
            &line_scene(),
            80,
            60,
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        )
        .unwrap();
        assert_eq!(frame.len(), 80 * 60 * 4);
        let red = frame.chunks(4).any(|pixel| pixel[0] > 200 && pixel[1] < 40);
        assert!(red, "the polyline should paint a red pixel");
    }

    #[test]
    fn camera_change_keeps_mark_buffers_and_aligns_grid() {
        compile_plot_shader().unwrap();
        let scene = PlotScene {
            title: "grid".into(),
            panels: vec![Panel {
                title: String::new(),
                y_label: String::new(),
                x_label: String::new(),
                xmin: 0.0,
                xmax: 10.0,
                ymin: 0.0,
                ymax: 10.0,
                series: vec![Series::Polyline {
                    xs: vec![4.0, 4.0],
                    ys: vec![4.0, 6.0],
                    color: [1.0, 0.0, 0.0, 1.0],
                    thickness: 3.0,
                }],
                x_labels: Vec::new(),
                equal: false,
            }],
            controls: Vec::new(),
            table: None,
            samples: Vec::new(),
            columns: 0,
            column_weights: Vec::new(),
            row_weights: Vec::new(),
        };
        let mut plot = Plot::new(&scene, [0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]);
        let first = plot.draw(180, 140, &PlotInput::default()).unwrap();
        if crate::native_gpu().is_err() {
            return;
        }
        let uploads = plot.mark_uploads();
        plot.zoom(0, 5.0, 5.0, 2.0);
        let frame = plot.draw(180, 140, &PlotInput::default()).unwrap();
        assert_eq!(plot.mark_uploads(), uploads);
        assert_eq!(first.rgba.len(), frame.rgba.len());
        let grid = plot.project_data(0, 4.0, 3.0, 180, 140);
        let sample = plot.project_data(0, 4.0, 5.0, 180, 140);
        assert!(
            near(&frame.rgba, 180, 140, grid, |pixel| pixel[1] > 40
                && pixel[0] < 40),
            "grid pixel {grid:?}"
        );
        assert!(
            near(&frame.rgba, 180, 140, sample, |pixel| pixel[0] > 150),
            "sample pixel {sample:?}"
        );
    }

    #[test]
    fn gpu_and_cpu_pictures_agree() {
        let Ok(gpu) = crate::native_gpu() else {
            return;
        };
        let mut scene = line_scene();
        scene.panels[0].series.extend([
            Series::Points {
                xs: vec![2.0, 7.0],
                ys: vec![3.0, 8.0],
                color: [0.0, 0.0, 1.0, 1.0],
                radius: 5.0,
            },
            Series::Bars {
                edges: vec![1.0, 3.0, 5.0],
                counts: vec![2.0, 4.0],
                color: [0.0, 0.8, 0.0, 0.8],
            },
        ]);
        scene.panels.push(Panel {
            title: "heat".into(),
            y_label: String::new(),
            x_label: String::new(),
            xmin: 0.0,
            xmax: 1.0,
            ymin: 0.0,
            ymax: 1.0,
            series: vec![Series::heatmap((0..64).map(|v| v as f32).collect(), 8, 8)],
            x_labels: Vec::new(),
            equal: false,
        });
        let (width, height) = (320, 160);
        let mut plot = Plot::new(&scene, [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]);
        plot.apply(width, height, &PlotInput::default());
        let on_gpu = plot.paint_offscreen(gpu, width, height).unwrap();
        let layout = plot.layout(width, height);
        let on_cpu = plot.paint_cpu(width, height, &layout);
        // Edges rasterize up to a pixel apart, so a pixel may match any 3x3 neighbor.
        let pixel = |frame: &[u8], x: i64, y: i64| {
            let x = x.clamp(0, i64::from(width) - 1) as usize;
            let y = y.clamp(0, i64::from(height) - 1) as usize;
            let index = (y * width as usize + x) * 4;
            [frame[index], frame[index + 1], frame[index + 2]]
        };
        let mut differ = 0;
        for y in 0..i64::from(height) {
            for x in 0..i64::from(width) {
                let want = pixel(&on_gpu, x, y);
                let matched = (-1..=1).any(|dy| {
                    (-1..=1).any(|dx| {
                        let got = pixel(&on_cpu, x + dx, y + dy);
                        want.iter().zip(got).all(|(a, b)| a.abs_diff(b) <= 96)
                    })
                });
                differ += usize::from(!matched);
            }
        }
        let share = differ as f32 / (width * height) as f32;
        assert!(share < 0.001, "{:.2}% of pixels differ", share * 100.0);
    }

    #[test]
    fn hover_picks_the_topmost_series_without_a_gpu_readback() {
        let mut scene = line_scene();
        scene.panels[0].series.push(Series::Points {
            xs: vec![2.0],
            ys: vec![5.0],
            color: [0.0, 0.0, 1.0, 1.0],
            radius: 6.0,
        });
        let mut plot = Plot::new(&scene, [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]);
        plot.apply(200, 160, &PlotInput::default());
        let on_point = plot.project_data(0, 2.0, 5.0, 200, 160);
        assert_eq!(
            plot.hover(on_point[0], on_point[1]).3,
            Some(1),
            "the point draws over the line"
        );
        let on_line = plot.project_data(0, 8.0, 5.0, 200, 160);
        let hover = plot.hover(on_line[0], on_line[1]);
        assert!(hover.0);
        assert!(
            (hover.1 - 8.0).abs() < 0.2,
            "data x under the cursor {}",
            hover.1
        );
        assert_eq!(hover.3, Some(0));
        let empty = plot.project_data(0, 8.0, 9.0, 200, 160);
        assert_eq!(plot.hover(empty[0], empty[1]).3, None);
    }

    #[test]
    fn every_panel_frame_is_one_crisp_pixel() {
        let mut horizontal = [10.2, 4.0];
        let mut horizontal_end = [80.0, 4.0];
        snap_hairline(&mut horizontal, &mut horizontal_end);
        assert_eq!(horizontal[1], 4.5);
        assert_eq!(horizontal_end[1], 4.5);
        let mut vertical = [3.2, 1.0];
        let mut vertical_end = [3.2, 40.0];
        snap_hairline(&mut vertical, &mut vertical_end);
        assert_eq!(vertical[0], 3.5);
        let mut diagonal = [0.0, 0.0];
        let mut diagonal_end = [10.0, 8.0];
        snap_hairline(&mut diagonal, &mut diagonal_end);
        assert_eq!(
            [diagonal[0], diagonal[1], diagonal_end[0], diagonal_end[1]],
            [0.0, 0.0, 10.0, 8.0]
        );

        let mut scene = line_scene();
        scene.panels.push(Panel {
            title: String::new(),
            y_label: "Probability (%)".into(),
            x_label: String::new(),
            xmin: 0.0,
            xmax: 10.0,
            ymin: 0.0,
            ymax: 10.0,
            series: Vec::new(),
            x_labels: Vec::new(),
            equal: false,
        });
        scene.columns = 2;
        scene.column_weights = vec![8.0, 1.65];
        let mut plot = Plot::new(&scene, [0.05, 0.05, 0.05, 1.0], [0.9, 0.9, 0.9, 1.0]);
        let (width, height) = (640, 360);
        let layout = plot.layout(width, height);
        let frame = plot.paint_cpu(width, height, &layout);
        let luma = |x: u32, y: u32| {
            let index = ((y * width + x) * 4) as usize;
            frame[index].max(frame[index + 1]).max(frame[index + 2])
        };
        for cell in &layout {
            let plot = cell.plot;
            assert_eq!(plot.x.fract(), 0.0);
            assert_eq!(plot.y.fract(), 0.0);
            assert_eq!(plot.w.fract(), 0.0);
            assert_eq!(plot.h.fract(), 0.0);
            let x0 = plot.x as u32;
            let y0 = plot.y as u32;
            let x1 = x0 + plot.w as u32 - 1;
            let y1 = y0 + plot.h as u32 - 1;
            let column = (x0 + 2..x1)
                .find(|x| luma(*x, y0 + 3) < 30)
                .expect("a column clear of the grid");
            let row = (y0 + 2..y1)
                .find(|y| luma(x0 + 3, *y) < 30)
                .expect("a row clear of the grid");
            for (edge, inward) in [
                ((column, y0), (column, y0 + 1)),
                ((column, y1), (column, y1 - 1)),
                ((x0, row), (x0 + 1, row)),
                ((x1, row), (x1 - 1, row)),
            ] {
                let on = luma(edge.0, edge.1);
                let inside = luma(inward.0, inward.1);
                assert!(on > 60, "missing frame pixel {edge:?} luma {on}");
                assert!(
                    on > inside + 30,
                    "frame at {edge:?} bled inward ({on} vs {inside})"
                );
            }
        }
    }

    #[test]
    fn y_tick_ink_sits_on_its_gridline_and_the_name_runs_up_the_axis() {
        let panel = Panel {
            title: "IC1 X".into(),
            y_label: "IC1 X (mm)".into(),
            x_label: String::new(),
            xmin: 0.0,
            xmax: 10.0,
            ymin: -10.0,
            ymax: 10.0,
            series: Vec::new(),
            x_labels: Vec::new(),
            equal: false,
        };
        let camera = Camera::new(0.0, 10.0, -10.0, 10.0);
        let cell = Cell {
            cell: PlotRect {
                x: 8.0,
                y: 4.0,
                w: 180.0,
                h: 140.0,
            },
            plot: PlotRect {
                x: 70.0,
                y: 16.0,
                w: 100.0,
                h: 100.0,
            },
            panel: 0,
        };
        let glyphs = labels_for(&panel, &camera, &cell, [1.0, 1.0, 1.0, 1.0]);
        let row: Vec<_> = glyphs
            .iter()
            .filter(|glyph| (glyph.anchor[1]).abs() < 1e-3 && glyph.anchor[3] > 0.5)
            .collect();
        assert!(!row.is_empty(), "a zero tick should be labeled");
        let top = row
            .iter()
            .map(|glyph| glyph.offset[1])
            .fold(f32::MAX, f32::min);
        let bottom = row
            .iter()
            .map(|glyph| glyph.offset[1] + glyph.size[1])
            .fold(f32::MIN, f32::max);
        let mid = 0.5 * (top + bottom);
        assert!(mid.abs() < 0.51, "tick label center {mid}");
        let turned: Vec<_> = glyphs
            .iter()
            .filter(|glyph| glyph.anchor[2] > 0.5 && glyph.anchor[3] < 0.5)
            .collect();
        assert!(!turned.is_empty(), "the y label should be rotated");
        let name_y = cell.plot.y + cell.plot.h * 0.5;
        assert!((turned[0].anchor[1] - name_y).abs() < 0.01);
        let name_x = cell.cell.x + axis_name_width(&panel.y_label) * 0.5;
        assert!(turned
            .iter()
            .all(|glyph| (glyph.anchor[0] - name_x).abs() < 0.01));
    }

    #[test]
    fn x_axis_name_sits_under_the_ticks() {
        let mut panel = Panel {
            title: String::new(),
            y_label: "Y (mm)".into(),
            x_label: "X (mm)".into(),
            xmin: 0.0,
            xmax: 10.0,
            ymin: 0.0,
            ymax: 10.0,
            series: Vec::new(),
            x_labels: Vec::new(),
            equal: false,
        };
        let camera = Camera::new(0.0, 10.0, 0.0, 10.0);
        let cell = Cell {
            cell: PlotRect {
                x: 8.0,
                y: 4.0,
                w: 180.0,
                h: 140.0,
            },
            plot: PlotRect {
                x: 70.0,
                y: 16.0,
                w: 100.0,
                h: 100.0,
            },
            panel: 0,
        };
        let glyphs = labels_for(&panel, &camera, &cell, [1.0, 1.0, 1.0, 1.0]);
        let center = cell.plot.x + cell.plot.w * 0.5;
        let named: Vec<_> = glyphs
            .iter()
            .filter(|glyph| {
                glyph.anchor[2] < 0.5
                    && glyph.anchor[3] < 0.5
                    && (glyph.anchor[0] - center).abs() < 0.01
            })
            .collect();
        assert!(!named.is_empty(), "the x label should be centered");
        assert!((named[0].anchor[1] - x_axis_name_y(&cell.plot)).abs() < 0.01);
        assert!(named[0].anchor[1] > cell.plot.y + cell.plot.h + atlas().line_height);

        let mut scene = line_scene();
        let bare = Plot::new(&scene, [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]).layout(200, 160);
        panel.series = scene.panels[0].series.clone();
        scene.panels[0] = panel;
        let named_layout =
            Plot::new(&scene, [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]).layout(200, 160);
        assert!(
            named_layout[0].plot.h + 8.0 < bare[0].plot.h,
            "x label should take a line under the ticks"
        );
    }

    #[test]
    fn grid_stays_behind_filled_data_and_a_triangle_hits_its_interior() {
        let mut scene = line_scene();
        scene.panels[0].series = vec![
            Series::Rects {
                x: vec![0.0],
                y: vec![0.0],
                w: vec![10.0],
                h: vec![10.0],
                color: [1.0, 0.0, 0.0, 1.0],
            },
            Series::Triangles {
                xs: vec![1.0, 9.0, 5.0],
                ys: vec![1.0, 1.0, 9.0],
                color: [0.0, 0.0, 1.0, 1.0],
            },
        ];
        let mut plot = Plot::new(&scene, [0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]);
        let frame = plot.draw(180, 140, &PlotInput::default()).unwrap();
        let covered = plot.project_data(0, 2.0, 8.0, 180, 140);
        assert!(
            near(&frame.rgba, 180, 140, covered, |pixel| pixel[0] > 150
                && pixel[1] < 40),
            "filled data should cover the grid {covered:?}"
        );
        let inside = plot.project_data(0, 5.0, 3.0, 180, 140);
        assert_eq!(plot.hover(inside[0], inside[1]).3, Some(1));
        let outside = plot.project_data(0, 1.0, 8.0, 180, 140);
        assert_ne!(plot.hover(outside[0], outside[1]).3, Some(1));
    }

    #[test]
    fn shared_edge_belongs_to_one_triangle() {
        let left = [[0.0, 0.0], [10.0, 10.0], [0.0, 10.0]];
        let right = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]];
        let owns = |p, tri: [[f32; 2]; 3]| owns_pixel_px(p, tri[0], tri[1], tri[2]);
        assert!(owns([1.5, 7.5], left));
        assert!(!owns([1.5, 7.5], right));
        assert!(owns([7.5, 1.5], right));
        assert!(!owns([7.5, 1.5], left));
        let seam = [5.0, 5.0];
        assert_eq!(
            usize::from(owns(seam, left)) + usize::from(owns(seam, right)),
            1,
            "the diagonal is drawn by one triangle"
        );
    }

    #[test]
    fn triangle_seam_stays_the_same_color_as_the_fill() {
        let mut scene = line_scene();
        scene.panels[0].series = vec![Series::Triangles {
            xs: vec![0.0, 10.0, 10.0, 0.0, 10.0, 0.0],
            ys: vec![0.0, 0.0, 10.0, 0.0, 10.0, 10.0],
            color: [1.0, 0.0, 0.0, 0.5],
        }];
        let (width, height) = (180, 140);
        let mut plot = Plot::new(&scene, [0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]);
        plot.apply(width, height, &PlotInput::default());
        let layout = plot.layout(width, height);
        let cpu = plot.paint_cpu(width, height, &layout);
        assert_fill_is_even(&cpu, width, &layout[0].plot);
        if let Ok(gpu) = crate::native_gpu() {
            let on_gpu = plot.paint_offscreen(gpu, width, height).unwrap();
            assert_fill_is_even(&on_gpu, width, &layout[0].plot);
        }
    }

    fn assert_fill_is_even(frame: &[u8], width: u32, plot: &PlotRect) {
        let mut quiet = 255u8;
        let mut peak = 0u8;
        let x0 = plot.x as u32 + 8;
        let x1 = (plot.x + plot.w) as u32 - 8;
        let y0 = plot.y as u32 + 8;
        let y1 = (plot.y + plot.h) as u32 - 8;
        for y in y0..y1 {
            for x in x0..x1 {
                let red = frame[((y * width + x) * 4) as usize];
                if red > 40 {
                    quiet = quiet.min(red);
                    peak = peak.max(red);
                }
            }
        }
        assert!(quiet > 80 && quiet < 255, "fill red {quiet}");
        assert!(
            peak - quiet < 24,
            "overlapping triangles darkened the fill ({quiet}..={peak})"
        );
    }

    fn near(
        frame: &[u8],
        width: u32,
        height: u32,
        point: [f32; 2],
        pred: impl Fn([u8; 4]) -> bool,
    ) -> bool {
        let x = point[0].round() as i32;
        let y = point[1].round() as i32;
        for dy in -2..=2 {
            for dx in -2..=2 {
                let px = x + dx;
                let py = y + dy;
                if px < 0 || py < 0 || px >= width as i32 || py >= height as i32 {
                    continue;
                }
                let index = ((py as u32 * width + px as u32) * 4) as usize;
                let pixel = [
                    frame[index],
                    frame[index + 1],
                    frame[index + 2],
                    frame[index + 3],
                ];
                if pred(pixel) {
                    return true;
                }
            }
        }
        false
    }
}
