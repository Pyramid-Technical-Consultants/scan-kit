//! Rasterize a [`scan_kit_core::PlotScene`].
//!
//! Mark positions stay in data space (`z = 0`). The vertex shader multiplies by
//! `clip_from_data`. Pan and zoom rewrite that matrix. Stroke width stays in pixels.

use std::num::NonZeroU64;
use std::sync::OnceLock;

use scan_kit_core::{
    format_tick, project, ticks, Camera, Panel, PlotRect, PlotScene, Series,
};

use crate::text::{self, atlas};
use crate::{request_device, ComputeError};

const PLOT_SHADER: &str = r#"
struct Uniforms {
    clip_from_data: mat4x4<f32>,
    resolution: vec2<f32>,
    pad: vec2<f32>,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(1) @binding(0) var mark_tex: texture_2d<f32>;
@group(1) @binding(1) var mark_samp: sampler;

struct Frag {
    @location(0) color: vec4<f32>,
    @location(1) id: vec4<f32>,
}

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

fn shade(rgb: vec3<f32>, alpha: f32, coverage: f32, id: u32) -> Frag {
    let a = alpha * coverage;
    if (a < 0.004) {
        discard;
    }
    var out: Frag;
    out.color = vec4<f32>(rgb * a, a);
    out.id = vec4<f32>(f32(id % 255u) / 255.0, f32(id / 255u) / 255.0, 0.0, 1.0);
    return out;
}

fn colormap(t: f32) -> vec3<f32> {
    let stops = array<vec3<f32>, 6>(
        vec3<f32>(0.267, 0.005, 0.329),
        vec3<f32>(0.230, 0.322, 0.546),
        vec3<f32>(0.128, 0.567, 0.551),
        vec3<f32>(0.267, 0.749, 0.441),
        vec3<f32>(0.741, 0.873, 0.150),
        vec3<f32>(0.993, 0.906, 0.144),
    );
    let x = clamp(t, 0.0, 1.0) * 5.0;
    let i = min(u32(x), 4u);
    let f = fract(x);
    return mix(stops[i], stops[i + 1u], f);
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
    @location(4) @interpolate(flat) id: u32,
}

@vertex
fn vs_line(
    @builtin(vertex_index) vi: u32,
    @location(0) a: vec4<f32>,
    @location(1) b: vec4<f32>,
    @location(2) color: vec4<f32>,
    @location(3) thickness: f32,
    @location(4) id: u32,
) -> LineOut {
    let pa = to_px(a.xyz);
    let pb = to_px(b.xyz);
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
    out.id = id;
    return out;
}

@fragment
fn fs_line(in: LineOut) -> Frag {
    let p = in.clip.xy;
    let ab = in.b - in.a;
    let len2 = max(dot(ab, ab), 0.0001);
    let t = clamp(dot(p - in.a, ab) / len2, 0.0, 1.0);
    let dist = length(p - (in.a + ab * t));
    let coverage = clamp(in.radius + 0.5 - dist, 0.0, 1.0);
    return shade(in.color.rgb, in.color.a, coverage, in.id);
}

struct PointOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) @interpolate(flat) center: vec2<f32>,
    @location(2) @interpolate(flat) radius: f32,
    @location(3) @interpolate(flat) id: u32,
}

@vertex
fn vs_point(
    @builtin(vertex_index) vi: u32,
    @location(0) center: vec4<f32>,
    @location(1) color: vec4<f32>,
    @location(2) id: u32,
) -> PointOut {
    let c = to_px(center.xyz);
    let radius = max(center.w, 1.0);
    let px = c + corner(vi) * (radius + 1.5);
    var out: PointOut;
    out.clip = px_to_clip(px);
    out.color = color;
    out.center = c;
    out.radius = radius;
    out.id = id;
    return out;
}

@fragment
fn fs_point(in: PointOut) -> Frag {
    let dist = length(in.clip.xy - in.center);
    let coverage = clamp(in.radius + 0.5 - dist, 0.0, 1.0);
    return shade(in.color.rgb, in.color.a, coverage, in.id);
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
    @location(5) @interpolate(flat) id: u32,
}

@vertex
fn vs_quad(
    @builtin(vertex_index) vi: u32,
    @location(0) a: vec4<f32>,
    @location(1) b: vec4<f32>,
    @location(2) color: vec4<f32>,
    @location(3) id: u32,
) -> QuadOut {
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
    out.id = id;
    return out;
}

@fragment
fn fs_quad(in: QuadOut) -> Frag {
    let p = in.clip.xy;
    let outside = max(max(in.bounds0.x - p.x, p.x - in.bounds1.x), max(in.bounds0.y - p.y, p.y - in.bounds1.y));
    let coverage = clamp(0.5 - outside, 0.0, 1.0);
    let sample = textureSample(mark_tex, mark_samp, in.uv).r;
    let rgb = mix(in.color.rgb, colormap(sample), in.heat);
    return shade(rgb, in.color.a, coverage, in.id);
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
    let px = origin + offset_size.xy + xy * offset_size.zw;
    var out: TextOut;
    out.clip = px_to_clip(px);
    out.color = color;
    out.uv = mix(uv.xy, uv.zw, xy);
    return out;
}

@fragment
fn fs_text(in: TextOut) -> Frag {
    let coverage = textureSample(mark_tex, mark_samp, in.uv).r;
    return shade(in.color.rgb, in.color.a, coverage, 0u);
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
    validator
        .validate(&module)
        .map_err(|err| err.to_string())?;
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

struct LineRec {
    a: [f32; 3],
    b: [f32; 3],
    color: [f32; 4],
    thickness: f32,
    id: u32,
}

struct PointRec {
    p: [f32; 3],
    radius: f32,
    color: [f32; 4],
    id: u32,
}

struct QuadRec {
    a: [f32; 2],
    b: [f32; 2],
    heat: f32,
    color: [f32; 4],
    id: u32,
    heatmap: Option<usize>,
}

struct GlyphRec {
    anchor: [f32; 4],
    offset: [f32; 2],
    size: [f32; 2],
    uv: [f32; 4],
    color: [f32; 4],
}

#[derive(Clone)]
struct PanelBatch {
    line_start: u32,
    line_count: u32,
    point_start: u32,
    point_count: u32,
    quad_start: u32,
    quad_count: u32,
    heats: Vec<(u32, usize)>,
}

struct Marks {
    lines: Vec<LineRec>,
    points: Vec<PointRec>,
    quads: Vec<QuadRec>,
    heatmaps: Vec<Vec<u8>>,
    heatmap_size: Vec<(u32, u32)>,
    panels: Vec<PanelBatch>,
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
    panels: Vec<PanelBatch>,
    heats: Vec<HeatGpu>,
}

struct Targets {
    width: u32,
    height: u32,
    color_view: wgpu::TextureView,
    color: wgpu::Texture,
    id_view: wgpu::TextureView,
    id: wgpu::Texture,
}

pub struct Plot {
    panels: Vec<Panel>,
    columns: u32,
    weights: Vec<f32>,
    cameras: Vec<Camera>,
    home: Vec<Camera>,
    background: [f32; 4],
    foreground: [f32; 4],
    marks: Marks,
    gpu: Option<GpuMarks>,
    targets: Option<Targets>,
    frame_buf: Option<wgpu::Buffer>,
    text_buf: Option<wgpu::Buffer>,
    uniform_buf: Option<wgpu::Buffer>,
    uniform_groups: Vec<wgpu::BindGroup>,
    color_stage: Option<wgpu::Buffer>,
    id_stage: Option<wgpu::Buffer>,
    cpu_ids: Vec<u32>,
    mark_uploads: u32,
    on_gpu: bool,
    tried_gpu: bool,
    size: (u32, u32),
}

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
        let cameras = scene
            .panels
            .iter()
            .map(|panel| Camera::new(panel.xmin, panel.xmax, panel.ymin, panel.ymax))
            .collect::<Vec<_>>();
        Self {
            marks: build_marks(&scene.panels),
            panels: scene.panels.clone(),
            columns: scene.columns,
            weights: scene.column_weights.clone(),
            home: cameras.clone(),
            cameras,
            background,
            foreground,
            gpu: None,
            targets: None,
            frame_buf: None,
            text_buf: None,
            uniform_buf: None,
            uniform_groups: Vec::new(),
            color_stage: None,
            id_stage: None,
            cpu_ids: Vec::new(),
            mark_uploads: 0,
            on_gpu: false,
            tried_gpu: false,
            size: (0, 0),
        }
    }

    #[cfg(test)]
    fn mark_uploads(&self) -> u32 {
        self.mark_uploads
    }

    #[cfg(test)]
    fn on_gpu(&self) -> bool {
        self.on_gpu
    }

    #[cfg(test)]
    fn zoom(&mut self, panel: usize, x: f32, y: f32, factor: f32) {
        if let Some(camera) = self.cameras.get_mut(panel) {
            camera.zoom_at(x, y, factor);
        }
    }

    pub fn draw(&mut self, width: u32, height: u32, input: &PlotInput) -> Result<PlotFrame, String> {
        let width = width.clamp(16, 8192);
        let height = height.clamp(16, 8192);
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
        if input.reset {
            self.cameras.clone_from(&self.home);
        }
        self.apply_pointer(width, height, input);
        let layout = self.layout(width, height);
        let sample = pixel_of(input.x, input.y, width, height);
        let (rgba, series) = self.paint(width, height, &layout, sample)?;
        let hover = self.hover_at(input.x, input.y, width, height, &layout, false);
        self.size = (width, height);
        Ok(PlotFrame {
            rgba,
            width,
            height,
            hover_hit: hover.0,
            hover_x: hover.1,
            hover_y: hover.2,
            series: if hover.0 { series } else { None },
        })
    }

    pub fn hover(&self, x: f32, y: f32) -> (bool, f32, f32, Option<u32>) {
        let (width, height) = self.size;
        if width == 0 {
            return (false, 0.0, 0.0, None);
        }
        let layout = self.layout(width, height);
        self.hover_at(x, y, width, height, &layout, true)
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
        );
        let font = atlas();
        rects
            .into_iter()
            .enumerate()
            .map(|(index, cell)| {
                let camera = self.cameras[index];
                let left = y_label_width(&camera) + 10.0;
                let top = if self.panels[index].title.is_empty() {
                    6.0
                } else {
                    font.line_height + 8.0
                };
                let bottom = font.line_height + 10.0;
                Cell {
                    plot: PlotRect {
                        x: cell.x + left,
                        y: cell.y + top,
                        w: (cell.w - left - 8.0).max(8.0),
                        h: (cell.h - top - bottom).max(8.0),
                    },
                    cell,
                    panel: index,
                }
            })
            .collect()
    }

    fn paint(
        &mut self,
        width: u32,
        height: u32,
        layout: &[Cell],
        sample: Option<(u32, u32)>,
    ) -> Result<(Vec<u8>, Option<u32>), String> {
        self.try_gpu();
        if self.on_gpu {
            return self
                .paint_gpu(width, height, layout, sample)
                .map_err(|err| err.to_string());
        }
        let frame = self.paint_cpu(width, height, layout);
        let series = sample.and_then(|(x, y)| {
            let id = self
                .cpu_ids
                .get((y * width + x) as usize)
                .copied()
                .unwrap_or(0);
            if id == 0 { None } else { Some(id - 1) }
        });
        Ok((frame, series))
    }

    fn try_gpu(&mut self) {
        if self.tried_gpu {
            return;
        }
        self.tried_gpu = true;
        if plot_gpu().is_ok() {
            self.on_gpu = true;
        }
    }

    fn paint_cpu(&mut self, width: u32, height: u32, layout: &[Cell]) -> Vec<u8> {
        let mut frame = Vec::with_capacity((width * height * 4) as usize);
        let pixel = rgba_bytes(self.background);
        for _ in 0..width * height {
            frame.extend_from_slice(&pixel);
        }
        self.cpu_ids = vec![0; (width * height) as usize];
        for cell in layout {
            let camera = self.cameras[cell.panel];
            let matrix = camera.clip_from_data(cell.plot, width as f32, height as f32);
            let batch = &self.marks.panels[cell.panel];
            for quad in &self.marks.quads
                [batch.quad_start as usize..(batch.quad_start + batch.quad_count) as usize]
            {
                fill_quad_cpu(&mut frame, &mut self.cpu_ids, width, height, &matrix, quad, &self.marks, cell.plot);
            }
            for (index, _) in &batch.heats {
                fill_quad_cpu(
                    &mut frame,
                    &mut self.cpu_ids,
                    width,
                    height,
                    &matrix,
                    &self.marks.quads[*index as usize],
                    &self.marks,
                    cell.plot,
                );
            }
            for line in frame_lines(&camera, self.foreground) {
                stroke_cpu(
                    &mut frame,
                    &mut self.cpu_ids,
                    width,
                    height,
                    &matrix,
                    &line,
                    cell.plot,
                );
            }
            for line in &self.marks.lines
                [batch.line_start as usize..(batch.line_start + batch.line_count) as usize]
            {
                stroke_cpu(
                    &mut frame,
                    &mut self.cpu_ids,
                    width,
                    height,
                    &matrix,
                    line,
                    cell.plot,
                );
            }
            for point in &self.marks.points
                [batch.point_start as usize..(batch.point_start + batch.point_count) as usize]
            {
                let px = project(matrix, point.p, width as f32, height as f32);
                disc_cpu(
                    &mut frame,
                    &mut self.cpu_ids,
                    width,
                    height,
                    px,
                    point.radius.max(1.0),
                    point.color,
                    point.id,
                    cell.plot,
                );
            }
            for glyph in labels_for(
                &self.panels[cell.panel],
                &camera,
                cell,
                self.foreground,
            ) {
                blit_glyph(&mut frame, width, height, &matrix, &glyph);
            }
        }
        frame
    }

    fn paint_gpu(
        &mut self,
        width: u32,
        height: u32,
        layout: &[Cell],
        sample: Option<(u32, u32)>,
    ) -> Result<(Vec<u8>, Option<u32>), ComputeError> {
        self.ensure_uploaded()?;
        let gpu = plot_gpu()?;
        self.ensure_targets(width, height)?;
        self.ensure_uniforms(layout.len())?;
        let frame_lines = layout
            .iter()
            .map(|cell| frame_lines(&self.cameras[cell.panel], self.foreground))
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
        let frame_bytes = frame_lines.iter().flat_map(|lines| encode_lines(lines)).collect::<Vec<_>>();
        let text_bytes = labels.iter().flat_map(|glyphs| encode_glyphs(glyphs)).collect::<Vec<_>>();
        write_grow(&gpu.device, &gpu.queue, &mut self.frame_buf, &frame_bytes, wgpu::BufferUsages::VERTEX)?;
        write_grow(&gpu.device, &gpu.queue, &mut self.text_buf, &text_bytes, wgpu::BufferUsages::VERTEX)?;
        let Some(uniform) = self.uniform_buf.as_ref() else {
            return Err(ComputeError::Message("uniform buffer missing".into()));
        };
        for (index, cell) in layout.iter().enumerate() {
            let matrix = self.cameras[cell.panel].clip_from_data(cell.plot, width as f32, height as f32);
            let bytes = encode_uniform(matrix, width as f32, height as f32);
            gpu.queue
                .write_buffer(uniform, (index * 256) as u64, &bytes);
        }
        let Some(targets) = self.targets.as_ref() else {
            return Err(ComputeError::Message("targets missing".into()));
        };
        let color_view = targets.color_view.clone();
        let id_view = targets.id_view.clone();
        let color = targets.color.clone();
        let marks_lines = self.gpu.as_ref().unwrap().lines.clone();
        let marks_points = self.gpu.as_ref().unwrap().points.clone();
        let marks_quads = self.gpu.as_ref().unwrap().quads.clone();
        let panels = self.gpu.as_ref().unwrap().panels.clone();
        let heat_groups = self
            .gpu
            .as_ref()
            .unwrap()
            .heats
            .iter()
            .map(|heat| heat.group.clone())
            .collect::<Vec<_>>();
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("plot"),
                color_attachments: &[
                    Some(color_attachment(&color_view, self.background)),
                    Some(color_attachment(&id_view, [0.0, 0.0, 0.0, 1.0])),
                ],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            let mut frame_cursor = 0u32;
            let mut text_cursor = 0u32;
            for (index, cell) in layout.iter().enumerate() {
                let offset = (index * 256) as u32;
                let batch = &panels[cell.panel];
                let (sx, sy, sw, sh) = scissor(cell.plot, width, height);
                pass.set_scissor_rect(sx, sy, sw, sh);
                pass.set_pipeline(&gpu.quad_pipeline);
                if batch.quad_count > 0 {
                    pass.set_bind_group(0, &self.uniform_groups[index], &[offset]);
                    pass.set_bind_group(1, &gpu.white_group, &[]);
                    pass.set_vertex_buffer(0, marks_quads.slice(..));
                    pass.draw(0..6, batch.quad_start..batch.quad_start + batch.quad_count);
                }
                for (start, texture) in &batch.heats {
                    pass.set_bind_group(0, &self.uniform_groups[index], &[offset]);
                    pass.set_bind_group(1, &heat_groups[*texture], &[]);
                    pass.set_vertex_buffer(0, marks_quads.slice(..));
                    pass.draw(0..6, *start..*start + 1);
                }
                pass.set_pipeline(&gpu.line_pipeline);
                pass.set_bind_group(0, &self.uniform_groups[index], &[offset]);
                let frame_count = frame_lines[index].len() as u32;
                if frame_count > 0 {
                    if let Some(buffer) = self.frame_buf.as_ref() {
                        pass.set_vertex_buffer(0, buffer.slice(..));
                        pass.draw(0..6, frame_cursor..frame_cursor + frame_count);
                    }
                }
                frame_cursor += frame_count;
                if batch.line_count > 0 {
                    pass.set_vertex_buffer(0, marks_lines.slice(..));
                    pass.draw(0..6, batch.line_start..batch.line_start + batch.line_count);
                }
                if batch.point_count > 0 {
                    pass.set_pipeline(&gpu.point_pipeline);
                    pass.set_bind_group(0, &self.uniform_groups[index], &[offset]);
                    pass.set_vertex_buffer(0, marks_points.slice(..));
                    pass.draw(0..6, batch.point_start..batch.point_start + batch.point_count);
                }
                pass.set_scissor_rect(0, 0, width, height);
                pass.set_pipeline(&gpu.text_pipeline);
                let text_count = labels[index].len() as u32;
                if text_count > 0 {
                    pass.set_bind_group(0, &self.uniform_groups[index], &[offset]);
                    pass.set_bind_group(1, &gpu.atlas_group, &[]);
                    if let Some(buffer) = self.text_buf.as_ref() {
                        pass.set_vertex_buffer(0, buffer.slice(..));
                        pass.draw(0..6, text_cursor..text_cursor + text_count);
                    }
                }
                text_cursor += text_count;
            }
        }
        let padded = (width * 4).div_ceil(256) * 256;
        let readback = self.color_stage(u64::from(padded) * u64::from(height))?;
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &color,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
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
        let id_stage = if let Some((x, y)) = sample {
            let id_stage = self.id_stage()?;
            let id_tex = self.targets.as_ref().unwrap().id.clone();
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &id_tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x, y, z: 0 },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &id_stage,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(256),
                        rows_per_image: Some(1),
                    },
                },
                wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );
            Some(id_stage)
        } else {
            None
        };
        gpu.queue.submit(Some(encoder.finish()));
        map_color_and_id(&gpu.device, &readback, id_stage.as_ref(), width, height, padded)
    }

    fn color_stage(&mut self, bytes: u64) -> Result<wgpu::Buffer, ComputeError> {
        let gpu = plot_gpu()?;
        let fits = self.color_stage.as_ref().is_some_and(|buffer| buffer.size() >= bytes);
        if !fits {
            self.color_stage = Some(gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("readback"),
                size: bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }));
        }
        Ok(self.color_stage.as_ref().unwrap().clone())
    }

    fn id_stage(&mut self) -> Result<wgpu::Buffer, ComputeError> {
        let gpu = plot_gpu()?;
        if self.id_stage.is_none() {
            self.id_stage = Some(gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("id"),
                size: 256,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }));
        }
        Ok(self.id_stage.as_ref().unwrap().clone())
    }

    fn hover_at(
        &self,
        x: f32,
        y: f32,
        width: u32,
        height: u32,
        layout: &[Cell],
        read_series: bool,
    ) -> (bool, f32, f32, Option<u32>) {
        let Some(index) = hit_plot(layout, x, y) else {
            return (false, 0.0, 0.0, None);
        };
        let cell = &layout[index];
        let data = self.cameras[cell.panel].data_at(x, y, cell.plot, width as f32, height as f32);
        let series = if read_series {
            self.series_at(x, y, width, height)
        } else {
            None
        };
        (true, data[0], data[1], series)
    }

    fn series_at(&self, x: f32, y: f32, width: u32, height: u32) -> Option<u32> {
        let px = x.floor() as i32;
        let py = y.floor() as i32;
        if px < 0 || py < 0 || px >= width as i32 || py >= height as i32 {
            return None;
        }
        let id = if self.on_gpu {
            read_id_pixel(self, px as u32, py as u32).unwrap_or(0)
        } else {
            self.cpu_ids.get((py as u32 * width + px as u32) as usize).copied().unwrap_or(0)
        };
        if id == 0 { None } else { Some(id - 1) }
    }

    fn ensure_targets(&mut self, width: u32, height: u32) -> Result<(), ComputeError> {
        if self.targets.as_ref().is_some_and(|targets| targets.width == width && targets.height == height) {
            return Ok(());
        }
        let gpu = plot_gpu()?;
        let color = target_texture(&gpu.device, width, height, wgpu::TextureFormat::Rgba8Unorm);
        let id = target_texture(&gpu.device, width, height, wgpu::TextureFormat::Rgba8Unorm);
        self.targets = Some(Targets {
            width,
            height,
            color_view: color.create_view(&Default::default()),
            id_view: id.create_view(&Default::default()),
            color,
            id,
        });
        Ok(())
    }

    fn ensure_uniforms(&mut self, count: usize) -> Result<(), ComputeError> {
        let needed = (count.max(1) * 256) as u64;
        if self.uniform_buf.as_ref().is_some_and(|buffer| buffer.size() >= needed) && self.uniform_groups.len() == count {
            return Ok(());
        }
        let gpu = plot_gpu()?;
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
        Ok(())
    }

    fn ensure_uploaded(&mut self) -> Result<(), ComputeError> {
        if self.gpu.is_some() {
            return Ok(());
        }
        let gpu = plot_gpu()?;
        let heats = self
            .marks
            .heatmaps
            .iter()
            .zip(&self.marks.heatmap_size)
            .map(|(pixels, (cols, rows))| {
                let texture = upload_r8(&gpu.device, &gpu.queue, pixels, *cols, *rows)?;
                let group = textured_group(&gpu.device, &gpu.textured_layout, &texture, &gpu.sampler);
                Ok(HeatGpu { group, texture })
            })
            .collect::<Result<Vec<_>, ComputeError>>()?;
        self.gpu = Some(GpuMarks {
            lines: upload_buffer(&gpu.device, &gpu.queue, "lines", &encode_lines(&self.marks.lines)),
            points: upload_buffer(&gpu.device, &gpu.queue, "points", &encode_points(&self.marks.points)),
            quads: upload_buffer(&gpu.device, &gpu.queue, "quads", &encode_quads(&self.marks.quads)),
            panels: self.marks.panels.clone(),
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

struct PlotGpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
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

fn plot_gpu() -> Result<&'static PlotGpu, ComputeError> {
    static GPU: OnceLock<Option<PlotGpu>> = OnceLock::new();
    GPU.get_or_init(|| pollster_block(build_plot_gpu()).ok())
        .as_ref()
        .ok_or(ComputeError::NoAdapter)
}

async fn build_plot_gpu() -> Result<PlotGpu, ComputeError> {
    let (device, queue) = request_device().await?;
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
    let line_pipeline = pipeline(&device, &shader, &line_layout, "vs_line", "fs_line", &[line_vertex()]);
    let point_pipeline = pipeline(&device, &shader, &line_layout, "vs_point", "fs_point", &[point_vertex()]);
    let quad_pipeline = pipeline(&device, &shader, &shaded_layout, "vs_quad", "fs_quad", &[quad_vertex()]);
    let text_pipeline = pipeline(&device, &shader, &shaded_layout, "vs_text", "fs_text", &[text_vertex()]);
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
    Ok(PlotGpu {
        device,
        queue,
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

fn pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    layout: &wgpu::PipelineLayout,
    vs: &str,
    fs: &str,
    buffers: &[wgpu::VertexBufferLayout<'_>],
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
            targets: &[
                Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                }),
                Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                }),
            ],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    })
}

const LINE_ATTRS: [wgpu::VertexAttribute; 5] = wgpu::vertex_attr_array![
    0 => Float32x4,
    1 => Float32x4,
    2 => Float32x4,
    3 => Float32,
    4 => Uint32,
];
const POINT_ATTRS: [wgpu::VertexAttribute; 3] = wgpu::vertex_attr_array![
    0 => Float32x4,
    1 => Float32x4,
    2 => Uint32,
];
const QUAD_ATTRS: [wgpu::VertexAttribute; 4] = wgpu::vertex_attr_array![
    0 => Float32x4,
    1 => Float32x4,
    2 => Float32x4,
    3 => Uint32,
];
const TEXT_ATTRS: [wgpu::VertexAttribute; 4] = wgpu::vertex_attr_array![
    0 => Float32x4,
    1 => Float32x4,
    2 => Float32x4,
    3 => Float32x4,
];

fn line_vertex() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: 64,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &LINE_ATTRS,
    }
}

fn point_vertex() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: 48,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &POINT_ATTRS,
    }
}

fn quad_vertex() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: 64,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &QUAD_ATTRS,
    }
}

fn text_vertex() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: 64,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &TEXT_ATTRS,
    }
}

fn build_marks(panels: &[Panel]) -> Marks {
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
                Series::Polyline { xs, ys, color, thickness }
                | Series::Guide { xs, ys, color, thickness } => {
                    push_polyline(&mut lines, xs, ys, *color, *thickness, id);
                }
                Series::Points { xs, ys, color, radius } => {
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
                Series::Bars { edges, counts, color } => {
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
                        if left.is_finite() && bottom.is_finite() && width.is_finite() && height.is_finite() {
                            quads.push(solid_quad(
                                [*left, *bottom],
                                [*left + *width, *bottom + *height],
                                *color,
                                id,
                            ));
                        }
                    }
                }
                Series::Heatmap { values, cols, rows } => {
                    if *cols == 0 || *rows == 0 {
                        continue;
                    }
                    let pixels = heatmap_bytes(values, *cols, *rows);
                    if pixels.is_empty() {
                        continue;
                    }
                    pending.push((heatmaps.len(), id));
                    heatmaps.push(pixels);
                    heatmap_size.push((*cols, *rows));
                }
            }
        }
        let quad_count = quads.len() as u32 - quad_start;
        let mut heats = Vec::new();
        for (texture, id) in pending {
            let start = quads.len() as u32;
            quads.push(QuadRec {
                a: [panel.xmin, panel.ymin],
                b: [panel.xmax, panel.ymax],
                heat: 1.0,
                color: [1.0, 1.0, 1.0, 1.0],
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
        heat: 0.0,
        color,
        id,
        heatmap: None,
    }
}

fn push_polyline(lines: &mut Vec<LineRec>, xs: &[f32], ys: &[f32], color: [f32; 4], thickness: f32, id: u32) {
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

fn heatmap_bytes(values: &[f32], cols: u32, rows: u32) -> Vec<u8> {
    let mut lo = f32::MAX;
    let mut hi = f32::MIN;
    for value in values {
        if value.is_finite() {
            lo = lo.min(*value);
            hi = hi.max(*value);
        }
    }
    if !lo.is_finite() {
        return Vec::new();
    }
    if (hi - lo).abs() < 1.0e-8 {
        hi = lo + 1.0;
    }
    let mut pixels = vec![0u8; (cols * rows) as usize];
    for (index, value) in values.iter().take(pixels.len()).enumerate() {
        if value.is_finite() {
            let t = ((*value - lo) / (hi - lo)).clamp(0.0, 1.0);
            pixels[index] = (t * 255.0).round() as u8;
        }
    }
    pixels
}

fn frame_lines(camera: &Camera, foreground: [f32; 4]) -> Vec<LineRec> {
    let mut lines = Vec::new();
    let grid = [foreground[0], foreground[1], foreground[2], 0.45];
    let spine = [foreground[0], foreground[1], foreground[2], 1.0];
    for tick in ticks(camera.xmin, camera.xmax) {
        lines.push(axis_line([tick, camera.ymin, 0.0], [tick, camera.ymax, 0.0], grid, 1.0));
    }
    for tick in ticks(camera.ymin, camera.ymax) {
        lines.push(axis_line([camera.xmin, tick, 0.0], [camera.xmax, tick, 0.0], grid, 1.0));
    }
    lines.push(axis_line(
        [camera.xmin, camera.ymin, 0.0],
        [camera.xmax, camera.ymin, 0.0],
        spine,
        1.25,
    ));
    lines.push(axis_line(
        [camera.xmin, camera.ymin, 0.0],
        [camera.xmin, camera.ymax, 0.0],
        spine,
        1.25,
    ));
    lines
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

fn labels_for(panel: &Panel, camera: &Camera, cell: &Cell, color: [f32; 4]) -> Vec<GlyphRec> {
    let mut out = Vec::new();
    let font = atlas();
    if !panel.title.is_empty() {
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
    for tick in ticks(camera.ymin, camera.ymax) {
        push_text(
            &mut out,
            &format_tick(tick),
            [camera.xmin, tick, 0.0],
            1.0,
            1.0,
            [-8.0, -font.ascent * 0.35],
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
        let stride = panel
            .x_labels
            .len()
            .div_ceil((cell.plot.w / 72.0).max(1.0) as usize)
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
    out
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

fn y_label_width(camera: &Camera) -> f32 {
    ticks(camera.ymin, camera.ymax)
        .into_iter()
        .map(|value| text::text_width(&format_tick(value)))
        .fold(0.0, f32::max)
}

fn panel_rects(count: usize, width: u32, height: u32, columns: u32, weights: &[f32]) -> Vec<PlotRect> {
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
    let inner_w = width as f32 - margin * 2.0 - gap * (cols.saturating_sub(1) as f32);
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
    let cell_h = (height as f32 - margin * 2.0 - gap * (rows.saturating_sub(1) as f32)) / rows as f32;
    (0..count)
        .map(|index| {
            let col = index % cols;
            let row = index / cols;
            PlotRect {
                x: col_x[col],
                y: margin + row as f32 * (cell_h + gap),
                w: col_w[col],
                h: cell_h,
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
    let x = (plot.x.max(0.0) as u32).min(width.saturating_sub(1));
    let y = (plot.y.max(0.0) as u32).min(height.saturating_sub(1));
    let w = (plot.w.max(1.0) as u32).min(width.saturating_sub(x)).max(1);
    let h = (plot.h.max(1.0) as u32).min(height.saturating_sub(y)).max(1);
    (x, y, w, h)
}

fn fill_quad_cpu(
    frame: &mut [u8],
    ids: &mut [u32],
    width: u32,
    height: u32,
    matrix: &[f32; 16],
    quad: &QuadRec,
    marks: &Marks,
    plot: PlotRect,
) {
    let p0 = project(*matrix, [quad.a[0], quad.a[1], 0.0], width as f32, height as f32);
    let p1 = project(*matrix, [quad.b[0], quad.b[1], 0.0], width as f32, height as f32);
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
                let sample = marks.heatmaps[index][(row * cols + col) as usize] as f32 / 255.0;
                let rgb = colormap(sample);
                [rgb[0], rgb[1], rgb[2], 1.0]
            } else {
                quad.color
            };
            blend(frame, ids, width, height, x, y, color, quad.id, Some(plot));
        }
    }
}

fn stroke_cpu(
    frame: &mut [u8],
    ids: &mut [u32],
    width: u32,
    height: u32,
    matrix: &[f32; 16],
    line: &LineRec,
    plot: PlotRect,
) {
    let pa = project(*matrix, line.a, width as f32, height as f32);
    let pb = project(*matrix, line.b, width as f32, height as f32);
    disc_cpu(frame, ids, width, height, pa, line.thickness.max(1.0) * 0.5, line.color, line.id, plot);
    let steps = ((pa[0] - pb[0]).abs().max((pa[1] - pb[1]).abs()) as i32).max(1);
    for step in 0..=steps {
        let t = step as f32 / steps as f32;
        disc_cpu(
            frame,
            ids,
            width,
            height,
            [pa[0] + (pb[0] - pa[0]) * t, pa[1] + (pb[1] - pa[1]) * t],
            line.thickness.max(1.0) * 0.5,
            line.color,
            line.id,
            plot,
        );
    }
}

fn disc_cpu(
    frame: &mut [u8],
    ids: &mut [u32],
    width: u32,
    height: u32,
    center: [f32; 2],
    radius: f32,
    color: [f32; 4],
    id: u32,
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
            blend(frame, ids, width, height, x, y, ink, id, Some(plot));
        }
    }
}

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
    let left = (origin[0] + glyph.offset[0]).floor() as i32;
    let top = (origin[1] + glyph.offset[1]).floor() as i32;
    let u0 = (glyph.uv[0] * font.size as f32).round() as u32;
    let v0 = (glyph.uv[1] * font.size as f32).round() as u32;
    let columns = glyph.size[0] as u32;
    let rows = glyph.size[1] as u32;
    let mut ids = Vec::new();
    for row in 0..rows {
        for col in 0..columns {
            let coverage = font.pixels[((v0 + row) * font.size + u0 + col) as usize] as f32 / 255.0;
            if coverage < 0.004 {
                continue;
            }
            let mut color = glyph.color;
            color[3] *= coverage;
            blend(frame, &mut ids, width, height, left + col as i32, top + row as i32, color, 0, None);
        }
    }
}

fn blend(
    frame: &mut [u8],
    ids: &mut [u32],
    width: u32,
    height: u32,
    x: i32,
    y: i32,
    color: [f32; 4],
    id: u32,
    plot: Option<PlotRect>,
) {
    if x < 0 || y < 0 || x >= width as i32 || y >= height as i32 {
        return;
    }
    if let Some(plot) = plot {
        if (x as f32) < plot.x - 1.0
            || (y as f32) < plot.y - 1.0
            || x as f32 >= plot.x + plot.w + 1.0
            || y as f32 >= plot.y + plot.h + 1.0
        {
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
    if id > 0 && alpha > 0.3 && index < ids.len() {
        ids[index] = id;
    }
}

fn colormap(t: f32) -> [f32; 3] {
    const STOPS: [[f32; 3]; 6] = [
        [0.267, 0.005, 0.329],
        [0.230, 0.322, 0.546],
        [0.128, 0.567, 0.551],
        [0.267, 0.749, 0.441],
        [0.741, 0.873, 0.150],
        [0.993, 0.906, 0.144],
    ];
    let x = (t.clamp(0.0, 1.0) * 5.0).min(4.999);
    let index = x.floor() as usize;
    let fract = x - index as f32;
    let a = STOPS[index];
    let b = STOPS[index + 1];
    [
        a[0] + (b[0] - a[0]) * fract,
        a[1] + (b[1] - a[1]) * fract,
        a[2] + (b[2] - a[2]) * fract,
    ]
}

fn encode_lines(lines: &[LineRec]) -> Vec<u8> {
    let mut out = Vec::with_capacity(lines.len() * 64);
    for line in lines {
        push4(&mut out, [line.a[0], line.a[1], line.a[2], 0.0]);
        push4(&mut out, [line.b[0], line.b[1], line.b[2], 0.0]);
        push4(&mut out, line.color);
        push_f32(&mut out, line.thickness);
        push_u32(&mut out, line.id);
        push_u32(&mut out, 0);
        push_u32(&mut out, 0);
    }
    out
}

fn encode_points(points: &[PointRec]) -> Vec<u8> {
    let mut out = Vec::with_capacity(points.len() * 48);
    for point in points {
        push4(&mut out, [point.p[0], point.p[1], point.p[2], point.radius]);
        push4(&mut out, point.color);
        push_u32(&mut out, point.id);
        push_u32(&mut out, 0);
        push_u32(&mut out, 0);
        push_u32(&mut out, 0);
    }
    out
}

fn encode_quads(quads: &[QuadRec]) -> Vec<u8> {
    let mut out = Vec::with_capacity(quads.len() * 64);
    for quad in quads {
        push4(&mut out, [quad.a[0], quad.a[1], quad.heat, 0.0]);
        push4(&mut out, [quad.b[0], quad.b[1], 0.0, 0.0]);
        push4(&mut out, quad.color);
        push_u32(&mut out, quad.id);
        push_u32(&mut out, 0);
        push_u32(&mut out, 0);
        push_u32(&mut out, 0);
    }
    out
}

fn encode_glyphs(glyphs: &[GlyphRec]) -> Vec<u8> {
    let mut out = Vec::with_capacity(glyphs.len() * 64);
    for glyph in glyphs {
        push4(&mut out, glyph.anchor);
        push4(&mut out, [glyph.offset[0], glyph.offset[1], glyph.size[0], glyph.size[1]]);
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

fn uniform_group(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, buffer: &wgpu::Buffer) -> wgpu::BindGroup {
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

fn upload_buffer(device: &wgpu::Device, queue: &wgpu::Queue, label: &str, bytes: &[u8]) -> wgpu::Buffer {
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
    _usage: wgpu::BufferUsages,
) -> Result<(), ComputeError> {
    if bytes.is_empty() {
        return Ok(());
    }
    let needed = bytes.len() as u64;
    if slot.as_ref().map(|buffer| buffer.size() < needed).unwrap_or(true) {
        *slot = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dynamic"),
            size: needed,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
    }
    queue.write_buffer(slot.as_ref().unwrap(), 0, bytes);
    Ok(())
}

fn upload_r8(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pixels: &[u8],
    cols: u32,
    rows: u32,
) -> Result<wgpu::Texture, ComputeError> {
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

fn target_texture(device: &wgpu::Device, width: u32, height: u32, format: wgpu::TextureFormat) -> wgpu::Texture {
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

fn color_attachment(view: &wgpu::TextureView, color: [f32; 4]) -> wgpu::RenderPassColorAttachment<'_> {
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

fn pixel_of(x: f32, y: f32, width: u32, height: u32) -> Option<(u32, u32)> {
    let px = x.floor() as i32;
    let py = y.floor() as i32;
    if px < 0 || py < 0 || px >= width as i32 || py >= height as i32 {
        None
    } else {
        Some((px as u32, py as u32))
    }
}

fn map_color_and_id(
    device: &wgpu::Device,
    color: &wgpu::Buffer,
    id: Option<&wgpu::Buffer>,
    width: u32,
    height: u32,
    padded: u32,
) -> Result<(Vec<u8>, Option<u32>), ComputeError> {
    let (sender, receiver) = std::sync::mpsc::channel();
    let color_done = sender.clone();
    color.slice(..).map_async(wgpu::MapMode::Read, move |result| {
        let _ = color_done.send(result);
    });
    if let Some(id) = id {
        id.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
    }
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|err| ComputeError::Message(err.to_string()))?;
    let waits = if id.is_some() { 2 } else { 1 };
    for _ in 0..waits {
        receiver
            .recv()
            .map_err(|err| ComputeError::Message(err.to_string()))?
            .map_err(|err| ComputeError::Message(err.to_string()))?;
    }
    let mapped = color.slice(..).get_mapped_range();
    let mut frame = Vec::with_capacity((width * height * 4) as usize);
    let row_bytes = (width * 4) as usize;
    for row in 0..height {
        let start = (row * padded) as usize;
        frame.extend_from_slice(&mapped[start..start + row_bytes]);
    }
    drop(mapped);
    color.unmap();
    let series = if let Some(id) = id {
        let mapped = id.slice(..).get_mapped_range();
        let value = u32::from(mapped[0]) + u32::from(mapped[1]) * 255;
        drop(mapped);
        id.unmap();
        if value == 0 { None } else { Some(value - 1) }
    } else {
        None
    };
    Ok((frame, series))
}

fn read_buffer(
    device: &wgpu::Device,
    readback: &wgpu::Buffer,
    width: u32,
    height: u32,
    padded: u32,
) -> Result<Vec<u8>, ComputeError> {
    let slice = readback.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|err| ComputeError::Message(err.to_string()))?;
    receiver
        .recv()
        .map_err(|err| ComputeError::Message(err.to_string()))?
        .map_err(|err| ComputeError::Message(err.to_string()))?;
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

fn read_id_pixel(plot: &Plot, x: u32, y: u32) -> Result<u32, ComputeError> {
    let Some(targets) = plot.targets.as_ref() else {
        return Ok(0);
    };
    let gpu = plot_gpu()?;
    let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("id"),
        size: 256,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &targets.id,
            mip_level: 0,
            origin: wgpu::Origin3d { x, y, z: 0 },
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256),
                rows_per_image: Some(1),
            },
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    gpu.queue.submit(Some(encoder.finish()));
    let bytes = read_buffer(&gpu.device, &buffer, 1, 1, 256)?;
    Ok(u32::from(bytes[0]) + u32::from(bytes[1]) * 255)
}

fn rgba_bytes(color: [f32; 4]) -> [u8; 4] {
    color.map(|channel| (channel.clamp(0.0, 1.0) * 255.0).round() as u8)
}

fn pollster_block<T>(future: impl std::future::Future<Output = T>) -> T {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");
    runtime.block_on(future)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line_scene() -> PlotScene {
        PlotScene {
            title: "line".into(),
            panels: vec![Panel {
                title: String::new(),
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
            }],
            controls: Vec::new(),
            table: None,
            samples: Vec::new(),
            columns: 0,
            column_weights: Vec::new(),
        }
    }

    #[test]
    fn sk_req_009_plot_shader_compiles_and_draws_a_line() {
        compile_plot_shader().unwrap();
        let frame = render_plot(&line_scene(), 80, 60, [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]).unwrap();
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
            }],
            controls: Vec::new(),
            table: None,
            samples: Vec::new(),
            columns: 0,
            column_weights: Vec::new(),
        };
        let mut plot = Plot::new(&scene, [0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]);
        let first = plot.draw(180, 140, &PlotInput::default()).unwrap();
        if !plot.on_gpu() {
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
            near(&frame.rgba, 180, 140, grid, |pixel| pixel[1] > 40 && pixel[0] < 40),
            "grid pixel {grid:?}"
        );
        assert!(
            near(&frame.rgba, 180, 140, sample, |pixel| pixel[0] > 150),
            "sample pixel {sample:?}"
        );
    }

    fn near(frame: &[u8], width: u32, height: u32, point: [f32; 2], pred: impl Fn([u8; 4]) -> bool) -> bool {
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
                let pixel = [frame[index], frame[index + 1], frame[index + 2], frame[index + 3]];
                if pred(pixel) {
                    return true;
                }
            }
        }
        false
    }
}
