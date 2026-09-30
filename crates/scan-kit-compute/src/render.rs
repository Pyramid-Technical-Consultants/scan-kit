//! Rasterize a [`scan_kit_core::PlotScene`] with wgpu.
//!
//! Triangles are built on the CPU. The fragment shader paints them into an
//! RGBA target and the frame is read back. Without an adapter the same
//! triangles are filled on the CPU so a headless check still sees the picture.

use std::sync::OnceLock;

use scan_kit_core::{map_span, Panel, PlotScene, Series};

use crate::{request_device, ComputeError};

const PLOT_SHADER: &str = r#"
struct Uniforms {
    size: vec2<f32>,
}

@group(0) @binding(0) var<uniform> u: Uniforms;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn vs(@location(0) pos: vec2<f32>, @location(1) color: vec4<f32>) -> VsOut {
    let x = (pos.x / u.size.x) * 2.0 - 1.0;
    let y = 1.0 - (pos.y / u.size.y) * 2.0;
    var out: VsOut;
    out.clip = vec4<f32>(x, y, 0.0, 1.0);
    out.color = color;
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    return in.color;
}
"#;

pub fn plot_shader_source() -> &'static str {
    PLOT_SHADER
}

#[derive(Clone, Copy)]
struct Tri {
    p: [[f32; 2]; 3],
    color: [f32; 4],
}

/// RGBA8 frame. `background` and `foreground` come from the shell's shadcn tokens.
pub fn render_plot(
    scene: &PlotScene,
    width: u32,
    height: u32,
    background: [f32; 4],
    foreground: [f32; 4],
) -> Result<Vec<u8>, String> {
    let width = width.clamp(16, 1600);
    let height = height.clamp(16, 1200);
    let triangles = scene_triangles(scene, width, height, foreground);
    match raster_gpu(width, height, background, &triangles) {
        Ok(frame) => Ok(frame),
        Err(ComputeError::NoAdapter) => Ok(raster_cpu(width, height, background, &triangles)),
        Err(ComputeError::Message(message)) => Err(message),
    }
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

fn scene_triangles(scene: &PlotScene, width: u32, height: u32, foreground: [f32; 4]) -> Vec<Tri> {
    let mut triangles = Vec::new();
    let rects = panel_rects(
        scene.panels.len(),
        width,
        height,
        scene.columns,
        &scene.column_weights,
    );
    for (panel, rect) in scene.panels.iter().zip(rects) {
        push_panel(&mut triangles, panel, rect, foreground);
    }
    triangles
}

#[derive(Clone, Copy)]
struct Rect {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
}

fn panel_rects(count: usize, width: u32, height: u32, columns: u32, weights: &[f32]) -> Vec<Rect> {
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
    let col_weight: Vec<f32> = if weights.len() == cols {
        weights.to_vec()
    } else {
        vec![1.0; cols]
    };
    let weight_sum = col_weight.iter().sum::<f32>().max(1e-6);
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
    let cell_h =
        (height as f32 - margin * 2.0 - gap * (rows.saturating_sub(1) as f32)) / rows as f32;
    (0..count)
        .map(|index| {
            let col = index % cols;
            let row = index / cols;
            Rect {
                x: col_x[col],
                y: margin + row as f32 * (cell_h + gap),
                w: col_w[col],
                h: cell_h,
            }
        })
        .collect()
}

fn push_panel(triangles: &mut Vec<Tri>, panel: &Panel, rect: Rect, foreground: [f32; 4]) {
    let title_px = if panel.title.is_empty() { 0.0 } else { 16.0 };
    if title_px > 0.0 {
        draw_text(
            triangles,
            &panel.title,
            rect.x + 28.0,
            rect.y + 1.0,
            rect.w - 32.0,
            foreground,
        );
    }
    let plot = Rect {
        x: rect.x + 64.0,
        y: rect.y + 4.0 + title_px,
        w: (rect.w - 72.0).max(8.0),
        h: (rect.h - 32.0 - title_px).max(8.0),
    };
    // Axis frame, so an empty panel is still a plot.
    push_quad(
        triangles,
        plot.x,
        plot.y + plot.h - 1.0,
        plot.w,
        1.0,
        foreground,
    );
    push_quad(triangles, plot.x, plot.y, 1.0, plot.h, foreground);
    draw_ticks(triangles, panel, plot, foreground);
    for series in &panel.series {
        match series {
            Series::Polyline {
                xs,
                ys,
                color,
                thickness,
            } => {
                push_polyline(triangles, xs, ys, *color, *thickness, panel, plot);
            }
            Series::Points {
                xs,
                ys,
                color,
                radius,
            } => {
                for (x, y) in xs.iter().zip(ys) {
                    if !x.is_finite() || !y.is_finite() {
                        continue;
                    }
                    let px = map_span(*x, panel.xmin, panel.xmax, plot.x, plot.x + plot.w);
                    let py = map_span(*y, panel.ymin, panel.ymax, plot.y + plot.h, plot.y);
                    let r = radius.max(1.0);
                    push_quad(triangles, px - r, py - r, r * 2.0, r * 2.0, *color);
                }
            }
            Series::Bars {
                edges,
                counts,
                color,
            } => {
                let ymin = panel.ymin.min(0.0);
                let ymax = panel.ymax.max(counts.iter().copied().fold(0.0, f32::max));
                for (i, count) in counts.iter().copied().enumerate() {
                    let Some(left) = edges.get(i).copied() else {
                        continue;
                    };
                    let Some(right) = edges.get(i + 1).copied() else {
                        continue;
                    };
                    let x0 = map_span(left, panel.xmin, panel.xmax, plot.x, plot.x + plot.w);
                    let x1 = map_span(right, panel.xmin, panel.xmax, plot.x, plot.x + plot.w);
                    let y0 = map_span(ymin, ymin, ymax, plot.y + plot.h, plot.y);
                    let y1 = map_span(count, ymin, ymax, plot.y + plot.h, plot.y);
                    push_quad(
                        triangles,
                        x0,
                        y1.min(y0),
                        (x1 - x0).abs().max(1.0),
                        (y1 - y0).abs().max(1.0),
                        *color,
                    );
                }
            }
            Series::Guide {
                xs,
                ys,
                color,
                thickness,
            } => {
                push_polyline(triangles, xs, ys, *color, *thickness, panel, plot);
            }
            Series::Rects { x, y, w, h, color } => {
                for (((left, bottom), width), height) in x.iter().zip(y).zip(w).zip(h) {
                    push_data_rect(
                        triangles, panel, plot, *left, *bottom, *width, *height, *color,
                    );
                }
            }
            Series::Heatmap { values, cols, rows } => {
                if *cols == 0 || *rows == 0 {
                    continue;
                }
                let mut lo = f32::MAX;
                let mut hi = f32::MIN;
                for value in values {
                    if value.is_finite() {
                        lo = lo.min(*value);
                        hi = hi.max(*value);
                    }
                }
                if !lo.is_finite() {
                    continue;
                }
                if (hi - lo).abs() < 1e-8 {
                    hi = lo + 1.0;
                }
                let cw = plot.w / *cols as f32;
                let ch = plot.h / *rows as f32;
                for row in 0..*rows {
                    for col in 0..*cols {
                        let index = col as usize + *cols as usize * row as usize;
                        let Some(value) = values.get(index).copied() else {
                            continue;
                        };
                        let t = ((value - lo) / (hi - lo)).clamp(0.0, 1.0);
                        let color = [
                            0.15 + 0.7 * t,
                            0.2 + 0.3 * (1.0 - (t - 0.5).abs()),
                            0.55 - 0.3 * t,
                            1.0,
                        ];
                        let x = plot.x + col as f32 * cw;
                        let y = plot.y + row as f32 * ch;
                        push_quad(triangles, x, y, cw.max(1.0), ch.max(1.0), color);
                    }
                }
            }
        }
    }
}

fn draw_ticks(triangles: &mut Vec<Tri>, panel: &Panel, plot: Rect, color: [f32; 4]) {
    for value in nice_ticks(panel.ymin, panel.ymax, 4) {
        let py = map_span(value, panel.ymin, panel.ymax, plot.y + plot.h, plot.y);
        let text = scan_kit_core::format_tick(value);
        let width = text.chars().count() as f32 * 12.0;
        draw_text(
            triangles,
            &text,
            (plot.x - width - 6.0).max(0.0),
            py - 7.0,
            width,
            color,
        );
        push_quad(triangles, plot.x - 4.0, py, 4.0, 1.0, color);
    }
    if panel.x_labels.is_empty() {
        for value in nice_ticks(panel.xmin, panel.xmax, 4) {
            let px = map_span(value, panel.xmin, panel.xmax, plot.x, plot.x + plot.w);
            let text = scan_kit_core::format_tick(value);
            let width = text.chars().count() as f32 * 12.0;
            draw_text(
                triangles,
                &text,
                px - width * 0.5,
                plot.y + plot.h + 2.0,
                width + 8.0,
                color,
            );
        }
    } else {
        let stride = panel
            .x_labels
            .len()
            .div_ceil((plot.w / 48.0).max(1.0) as usize)
            .max(1);
        for (index, label) in panel.x_labels.iter().enumerate() {
            if index % stride != 0 || label.is_empty() {
                continue;
            }
            let px = map_span(
                index as f32,
                panel.xmin,
                panel.xmax,
                plot.x,
                plot.x + plot.w,
            );
            let width = label.chars().count() as f32 * 12.0;
            draw_text(
                triangles,
                label,
                px - width * 0.5,
                plot.y + plot.h + 2.0,
                width + 8.0,
                color,
            );
        }
    }
}

fn nice_ticks(lo: f32, hi: f32, count: usize) -> Vec<f32> {
    if !lo.is_finite() || !hi.is_finite() || count < 2 || (hi - lo).abs() < 1e-6 {
        return vec![lo];
    }
    (0..count)
        .map(|index| lo + (hi - lo) * index as f32 / (count - 1) as f32)
        .collect()
}

fn push_polyline(
    triangles: &mut Vec<Tri>,
    xs: &[f32],
    ys: &[f32],
    color: [f32; 4],
    thickness: f32,
    panel: &Panel,
    plot: Rect,
) {
    let mut prev: Option<(f32, f32)> = None;
    for (x, y) in xs.iter().zip(ys) {
        if !x.is_finite() || !y.is_finite() {
            prev = None;
            continue;
        }
        let px = map_span(*x, panel.xmin, panel.xmax, plot.x, plot.x + plot.w);
        let py = map_span(*y, panel.ymin, panel.ymax, plot.y + plot.h, plot.y);
        if let Some((x0, y0)) = prev {
            push_segment(triangles, x0, y0, px, py, thickness.max(1.0), color);
        }
        prev = Some((px, py));
    }
}

fn push_segment(
    triangles: &mut Vec<Tri>,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    thickness: f32,
    color: [f32; 4],
) {
    let dx = x1 - x0;
    let dy = y1 - y0;
    let len = (dx * dx + dy * dy).sqrt().max(1e-3);
    let nx = -dy / len * thickness * 0.5;
    let ny = dx / len * thickness * 0.5;
    let a = [x0 + nx, y0 + ny];
    let b = [x0 - nx, y0 - ny];
    let c = [x1 - nx, y1 - ny];
    let d = [x1 + nx, y1 + ny];
    triangles.push(Tri {
        p: [a, b, c],
        color,
    });
    triangles.push(Tri {
        p: [a, c, d],
        color,
    });
}

#[allow(clippy::too_many_arguments)]
fn push_data_rect(
    triangles: &mut Vec<Tri>,
    panel: &Panel,
    plot: Rect,
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    color: [f32; 4],
) {
    if !x.is_finite() || !y.is_finite() || !w.is_finite() || !h.is_finite() {
        return;
    }
    let x0 = map_span(x, panel.xmin, panel.xmax, plot.x, plot.x + plot.w);
    let x1 = map_span(x + w, panel.xmin, panel.xmax, plot.x, plot.x + plot.w);
    let y0 = map_span(y, panel.ymin, panel.ymax, plot.y + plot.h, plot.y);
    let y1 = map_span(y + h, panel.ymin, panel.ymax, plot.y + plot.h, plot.y);
    let left = x0.min(x1);
    let top = y0.min(y1);
    push_quad(
        triangles,
        left,
        top,
        (x1 - x0).abs().max(1.0),
        (y1 - y0).abs().max(1.0),
        color,
    );
}

fn push_quad(triangles: &mut Vec<Tri>, x: f32, y: f32, w: f32, h: f32, color: [f32; 4]) {
    let a = [x, y];
    let b = [x + w, y];
    let c = [x + w, y + h];
    let d = [x, y + h];
    triangles.push(Tri {
        p: [a, b, c],
        color,
    });
    triangles.push(Tri {
        p: [a, c, d],
        color,
    });
}

fn raster_cpu(width: u32, height: u32, background: [f32; 4], triangles: &[Tri]) -> Vec<u8> {
    let mut frame = Vec::with_capacity((width * height * 4) as usize);
    for _ in 0..width * height {
        push_rgba(&mut frame, background);
    }
    for tri in triangles {
        fill_triangle(&mut frame, width, height, tri);
    }
    frame
}

fn fill_triangle(frame: &mut [u8], width: u32, height: u32, tri: &Tri) {
    let [a, b, c] = tri.p;
    let min_x = a[0].min(b[0]).min(c[0]).floor().max(0.0) as u32;
    let max_x = a[0].max(b[0]).max(c[0]).ceil().min(width as f32 - 1.0) as u32;
    let min_y = a[1].min(b[1]).min(c[1]).floor().max(0.0) as u32;
    let max_y = a[1].max(b[1]).max(c[1]).ceil().min(height as f32 - 1.0) as u32;
    let area = edge(a, b, c);
    if area.abs() < 1e-4 {
        return;
    }
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let p = [x as f32 + 0.5, y as f32 + 0.5];
            let w0 = edge(b, c, p) / area;
            let w1 = edge(c, a, p) / area;
            let w2 = edge(a, b, p) / area;
            if w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0 {
                let index = ((y * width + x) * 4) as usize;
                if index + 3 < frame.len() {
                    blend_pixel(frame, index, tri.color);
                }
            }
        }
    }
}

fn edge(a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> f32 {
    (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0])
}

fn push_rgba(frame: &mut Vec<u8>, color: [f32; 4]) {
    frame.extend_from_slice(&rgba_bytes(color));
}

fn blend_pixel(frame: &mut [u8], index: usize, color: [f32; 4]) {
    let src = rgba_bytes(color);
    if src[3] == 255 {
        frame[index..index + 4].copy_from_slice(&src);
        return;
    }
    let alpha = f32::from(src[3]) / 255.0;
    for channel in 0..3 {
        let dst = f32::from(frame[index + channel]);
        frame[index + channel] =
            (f32::from(src[channel]) * alpha + dst * (1.0 - alpha)).round() as u8;
    }
    frame[index + 3] = 255;
}

fn draw_text(triangles: &mut Vec<Tri>, text: &str, x: f32, y: f32, max_w: f32, color: [f32; 4]) {
    let mut cursor = x;
    for ch in text.chars() {
        if cursor + 8.0 > x + max_w {
            break;
        }
        let Some(rows) = glyph(ch.to_ascii_uppercase()) else {
            cursor += 8.0;
            continue;
        };
        for (row, bits) in rows.iter().enumerate() {
            for col in 0..5 {
                if bits & (1 << (4 - col)) != 0 {
                    push_quad(
                        triangles,
                        cursor + col as f32 * 2.0,
                        y + row as f32 * 2.0,
                        2.0,
                        2.0,
                        color,
                    );
                }
            }
        }
        cursor += 12.0;
    }
}

fn glyph(ch: char) -> Option<[u8; 7]> {
    // 5×7, top row first, high bit on the left.
    Some(match ch {
        ' ' => [0, 0, 0, 0, 0, 0, 0],
        'A' => [
            0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001,
        ],
        'B' => [
            0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110,
        ],
        'C' => [
            0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110,
        ],
        'D' => [
            0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110,
        ],
        'E' => [
            0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111,
        ],
        'F' => [
            0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000,
        ],
        'G' => [
            0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01110,
        ],
        'H' => [
            0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001,
        ],
        'I' => [
            0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110,
        ],
        'J' => [
            0b00111, 0b00010, 0b00010, 0b00010, 0b10010, 0b10010, 0b01100,
        ],
        'K' => [
            0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001,
        ],
        'L' => [
            0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111,
        ],
        'M' => [
            0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001,
        ],
        'N' => [
            0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001,
        ],
        'O' => [
            0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110,
        ],
        'P' => [
            0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000,
        ],
        'Q' => [
            0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101,
        ],
        'R' => [
            0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001,
        ],
        'S' => [
            0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110,
        ],
        'T' => [
            0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100,
        ],
        'U' => [
            0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110,
        ],
        'V' => [
            0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100,
        ],
        'W' => [
            0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b10101, 0b01010,
        ],
        'X' => [
            0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001,
        ],
        'Y' => [
            0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100,
        ],
        'Z' => [
            0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111,
        ],
        '0' => [
            0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110,
        ],
        '1' => [
            0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110,
        ],
        '2' => [
            0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111,
        ],
        '3' => [
            0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110,
        ],
        '4' => [
            0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010,
        ],
        '5' => [
            0b11111, 0b10000, 0b10000, 0b11110, 0b00001, 0b00001, 0b11110,
        ],
        '6' => [
            0b01110, 0b10000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110,
        ],
        '7' => [
            0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000,
        ],
        '8' => [
            0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110,
        ],
        '9' => [
            0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00001, 0b01110,
        ],
        '/' => [
            0b00001, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b10000,
        ],
        '%' => [
            0b11001, 0b11010, 0b00100, 0b00100, 0b01011, 0b10011, 0b00000,
        ],
        '(' => [
            0b00010, 0b00100, 0b01000, 0b01000, 0b01000, 0b00100, 0b00010,
        ],
        ')' => [
            0b01000, 0b00100, 0b00010, 0b00010, 0b00010, 0b00100, 0b01000,
        ],
        '.' => [0, 0, 0, 0, 0, 0b00100, 0b00100],
        '-' => [0, 0, 0, 0b01110, 0, 0, 0],
        '+' => [0, 0b00100, 0b00100, 0b11111, 0b00100, 0b00100, 0],
        _ => return None,
    })
}

fn rgba_bytes(color: [f32; 4]) -> [u8; 4] {
    color.map(|channel| (channel.clamp(0.0, 1.0) * 255.0).round() as u8)
}

struct PlotGpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::RenderPipeline,
    bind_layout: wgpu::BindGroupLayout,
}

fn plot_gpu() -> Result<&'static PlotGpu, ComputeError> {
    static GPU: OnceLock<Option<PlotGpu>> = OnceLock::new();
    // ponytail: the first frame builds the device and pipeline; later frames reuse them.
    // A process restart is the recovery if the device is lost.
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
    let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_layout],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("plot"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs"),
            compilation_options: Default::default(),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: 24,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x2,
                        offset: 0,
                        shader_location: 0,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x4,
                        offset: 8,
                        shader_location: 1,
                    },
                ],
            }],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8Unorm,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    });
    Ok(PlotGpu {
        device,
        queue,
        pipeline,
        bind_layout,
    })
}

fn raster_gpu(
    width: u32,
    height: u32,
    background: [f32; 4],
    triangles: &[Tri],
) -> Result<Vec<u8>, ComputeError> {
    if triangles.is_empty() {
        return Ok(raster_cpu(width, height, background, triangles));
    }
    let gpu = plot_gpu()?;
    let device = &gpu.device;
    let queue = &gpu.queue;
    let mut vertices: Vec<f32> = Vec::with_capacity(triangles.len() * 18);
    for tri in triangles {
        for point in tri.p {
            vertices.extend_from_slice(&point);
            vertices.extend_from_slice(&tri.color);
        }
    }
    let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vertices"),
        size: (vertices.len() * 4) as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let vertex_bytes = bytemuck_f32(&vertices);
    queue.write_buffer(&vertex_buffer, 0, &vertex_bytes);

    let uniform_data = [width as f32, height as f32, 0.0, 0.0];
    let uniform = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("uniforms"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let uniform_bytes = bytemuck_f32(&uniform_data);
    queue.write_buffer(&uniform, 0, &uniform_bytes);

    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &gpu.bind_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform.as_entire_binding(),
        }],
    });

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("frame"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let padded = (width * 4).div_ceil(256) * 256;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: u64::from(padded) * u64::from(height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("plot"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: f64::from(background[0]),
                        g: f64::from(background[1]),
                        b: f64::from(background[2]),
                        a: f64::from(background[3]),
                    }),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&gpu.pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.set_vertex_buffer(0, vertex_buffer.slice(..));
        pass.draw(0..(triangles.len() * 3) as u32, 0..1);
    }
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
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
    queue.submit(Some(encoder.finish()));
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

fn bytemuck_f32(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

#[cfg(test)]
mod tests {
    use scan_kit_core::{Panel, PlotScene, Series};

    use super::*;

    #[test]
    fn sk_req_009_plot_shader_compiles_and_draws_a_line() {
        compile_plot_shader().unwrap();
        let scene = PlotScene {
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
        };
        let frame =
            render_plot(&scene, 80, 60, [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]).unwrap();
        assert_eq!(frame.len(), 80 * 60 * 4);
        let red = frame.chunks(4).any(|pixel| pixel[0] > 200 && pixel[1] < 40);
        assert!(red, "the polyline should paint a red pixel");
    }
}

fn pollster_block<T>(future: impl std::future::Future<Output = T>) -> T {
    // The desktop command is synchronous. A private current-thread runtime
    // drives the one adapter request. ponytail: one runtime per frame; a
    // resident device replaces it if view latency shows up.
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");
    runtime.block_on(future)
}
