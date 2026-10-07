//! Rasterize a [`scan_kit_core::PlotScene`].
//!
//! Mark positions stay in data space (`z = 0`). The vertex shader multiplies by
//! `clip_from_data`. Pan and zoom rewrite that matrix. Stroke width stays in pixels.
//!
//! Each panel paints the grid, solid quads and heatmaps, lines, points, the
//! 1px bound, then labels. A dose color axis is drawn with the labels, in
//! screen pixels outside the plot scissor: `heat` above 2.5 samples the ramp
//! and `heat` below −1.5 is a tick or the bar frame. [`Plot::record`] (GPU),
//! [`Plot::paint_cpu`] (no adapter), and [`Plot::pick`] (hover, reverse order)
//! must keep that order.
//! The plot rectangle is whole pixels. Axis-aligned strokes sit on pixel
//! centers, in the vertex shader and in [`stroke_cpu`], so a 1px line is one
//! pixel. The bound is that stroke on the rectangle's outer pixels.
//! A panel with `equal` letterboxes that rectangle to the camera's data
//! aspect, so one data unit has the same pixel length on both axes.
//! A dose slice fills its cell. The window grows on the longer side so a
//! millimetre has the same pixel length on both axes, including after a
//! quarter turn. That window is the ramp's zero. The volume stays on its
//! own bounds, and the crosshair runs out to the window. Slice axes sit on
//! that border. The 3D cell keeps its axes in the volume and leaves the
//! border clear. Tick marks and their numbers hang off the three edges that
//! meet at the entrance corner, on the side facing the camera, and the text
//! turns to follow the edge. The titles are X (mm), Y (mm), and Depth (mm).
//! The field box is an amber rectangle on each slice and an amber wireframe
//! through the volume. The wire is a pixel and a half wide on that picture,
//! at the edge's own depth, so orbit and zoom do not drop it. Dose line plots
//! draw every loaded session in that session's color. A second chamber uses
//! that color at reduced alpha. The plan is a dashed stroke in the same color.
//! A toolbar band names the plot, so the canvas title stays off.
//! Color gain, window, opacity, ray mode, and the sample filter are uniforms.
//! A color-scale change rewrites the 256 ramp pixels on the last atlas row.
//! Neither rebuilds the volume.
//!
//! A mark kind's byte layout lives in its `*_STRIDE`, `*_ATTRS`, `encode_*`,
//! `decode_*`, and the matching WGSL `vs_*` inputs. Change them together.
//! The scene's lines and points are the filtered samples. A stroke may be
//! simplified here for the current camera when several samples share a pixel,
//! and that pass keeps the extrema in the pixel. A camera that gives a sample
//! its own pixel draws it. Do not thin the series before it reaches this crate.
//! Points that share a cell of a 4096 grid over the panel range keep the first
//! sample. That is the mark that is uploaded, so a later zoom does not restore
//! samples that shared the cell. A point series that carries a sample time keeps
//! every sample, sorted by that time. A time trace is monotonic in x and keeps
//! every sample too. Playback draws each of those as one buffer range inside
//! the playhead window, not a new upload. A density or contour cloud keeps the
//! same timed samples in that buffer and does not draw them as dots. Playback
//! counts only the visible window into the heatmap texture or the contour mesh.
//! Guides and panels that are not time traces stay whole draws.
//! Heatmap value row 0 is the low data y. The texture is stored top-first, so
//! that row is the last row of pixels. Those pixels are the catalog color for
//! the series ramp, baked before upload.
//! Draws stay within WebGL2: no base instance (bind a buffer slice instead) and
//! one color target.

use std::collections::HashSet;
use std::num::NonZeroU64;

use scan_kit_core::{
    contour_in_frame, count_grid, format_tick, project, robust_limits, ticks, time_window, Camera,
    CloudStyle, Panel, PlotRect, PlotScene, Series,
};

use serde::{Deserialize, Serialize};

use crate::text::{self, atlas};
use crate::GpuError;

const PLOT_SHADER: &str = r#"
struct Uniforms {
    clip_from_data: mat4x4<f32>,
    resolution: vec2<f32>,
    pad: vec2<f32>,
    // plane, slice or azimuth, turns or elevation, integral
    dose0: vec4<f32>,
    // nx, ny, nz, peak
    dose1: vec4<f32>,
    // lo, hi, gain, opacity
    dose2: vec4<f32>,
    // mode, filter, voxel, integral peak
    dose3: vec4<f32>,
    // volume origin xyz, orbit zoom
    dose4: vec4<f32>,
    // film aspect, line scale, typical, march voxel
    dose5: vec4<f32>,
    // march shape, or integral cols, rows, row. Volume w picks each axis's outward side.
    dose6: vec4<f32>,
    // look-at offset xyz, field of view (0 is orthographic)
    dose7: vec4<f32>,
    // field min xyz, 1 when the field box is drawn
    dose8: vec4<f32>,
    // field max xyz, 1 when the phantom box is drawn
    dose9: vec4<f32>,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(1) @binding(0) var mark_tex: texture_2d<f32>;
@group(1) @binding(1) var mark_samp: sampler;
@group(2) @binding(0) var dose_vol: texture_3d<f32>;
@group(2) @binding(1) var dose_brick: texture_3d<f32>;
@group(2) @binding(2) var dose_samp: sampler;
@group(2) @binding(3) var dose_int: texture_2d<f32>;

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
    // Screen pixels, y down. a.xy and b.xy are the rectangle corners.
    // heat > 2.5 is the color ramp. heat < -1.5 is a tick or the bar frame.
    if (a.z > 2.5 || a.z < -1.5) {
        let uv = quad_uv(vi);
        let bounds0 = min(a.xy, b.xy);
        let bounds1 = max(a.xy, b.xy);
        var px = vec2<f32>(mix(a.x, b.x, uv.x), mix(a.y, b.y, uv.y));
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
    let p = in.clip.xy;
    let outside = max(max(in.bounds0.x - p.x, p.x - in.bounds1.x), max(in.bounds0.y - p.y, p.y - in.bounds1.y));
    let coverage = clamp(0.5 - outside, 0.0, 1.0);
    // Screen-space color axis. The ramp reads the last row of the dose atlas.
    if (in.heat > 2.5) {
        let span = max(in.bounds1.y - in.bounds0.y, 1.0);
        let t = clamp(1.0 - (in.clip.y - in.bounds0.y) / span, 0.0, 1.0);
        let bar = ramp_color(t);
        return shade(bar.rgb, bar.a, coverage);
    }
    if (in.heat < -1.5) {
        return shade(in.color.rgb, in.color.a, coverage);
    }
    if (in.heat < -0.5) {
        let coverage = select(0.0, 1.0, owns_pixel(in.clip.xy, in.bounds0, in.bounds1, in.uv));
        return shade(in.color.rgb, in.color.a, coverage);
    }
    // textureLoad and textureSampleLevel have no derivatives, so a dose atlas
    // lookup can sit in this branch. heat > 1.5 samples the uploaded volume.
    // heat > 0.5 is a baked ramp. Otherwise the quad is a solid color.
    if (in.heat > 1.5) {
        let plot_h = max(abs(in.bounds1.y - in.bounds0.y), 1.0);
        return dose_fragment(in.uv, coverage, plot_h);
    }
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
    // w < -0.5 rotates by z radians, clockwise with y down, so a 3D axis title
    // follows its edge. z > 0.5 on other screen text runs up the axis (90° CCW).
    if (anchor.w < -0.5) {
        let c = cos(anchor.z);
        let s = sin(anchor.z);
        local = vec2<f32>(c * local.x - s * local.y, s * local.x + c * local.y);
    } else if (anchor.w < 0.5 && anchor.z > 0.5) {
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

fn atlas_load(ix: i32, iy: i32, iz: i32) -> vec4<f32> {
    let nx = max(i32(u.dose1.x + 0.5), 1);
    let ny = max(i32(u.dose1.y + 0.5), 1);
    let nz = max(i32(u.dose1.z + 0.5), 1);
    let tiles = max(i32(ceil(sqrt(f32(nz)))), 1);
    let z = clamp(iz, 0, nz - 1);
    let tx = z % tiles;
    let ty = z / tiles;
    let x = tx * nx + clamp(ix, 0, nx - 1);
    let y = ty * ny + clamp(ny - 1 - iy, 0, ny - 1);
    let dims = textureDimensions(mark_tex);
    if (x < 0 || y < 0 || x >= i32(dims.x) || y >= i32(dims.y) - 1) {
        // 0.5 is zero dose. The slice atlas is signed around mid-grey.
        return vec4<f32>(0.5, 0.0, 0.0, 0.0);
    }
    return textureLoad(mark_tex, vec2<i32>(x, y), 0);
}

fn ramp_color(t: f32) -> vec4<f32> {
    let dims = textureDimensions(mark_tex);
    let x = clamp(i32(clamp(t, 0.0, 1.0) * 255.0), 0, 255);
    return textureLoad(mark_tex, vec2<i32>(x, i32(dims.y) - 1), 0);
}

fn turned_uv(uv: vec2<f32>, turns: u32) -> vec2<f32> {
    let t = turns % 4u;
    if (t == 1u) { return vec2<f32>(uv.y, 1.0 - uv.x); }
    if (t == 2u) { return vec2<f32>(1.0 - uv.x, 1.0 - uv.y); }
    if (t == 3u) { return vec2<f32>(1.0 - uv.y, uv.x); }
    return uv;
}

fn catmull(d: f32) -> f32 {
    let x = abs(d);
    if (x >= 2.0) { return 0.0; }
    if (x >= 1.0) { return ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0; }
    return (1.5 * x - 2.5) * x * x + 1.0;
}

fn atlas_dose(sample: f32) -> f32 {
    return (sample - 0.5) * 2.0 * u.dose1.w;
}

fn sample_grid(x: f32, y: f32, z: f32) -> vec4<f32> {
    let sampling = u32(u.dose3.y + 0.5);
    if (sampling == 0u) {
        return atlas_load(i32(round(x)), i32(round(y)), i32(round(z)));
    }
    let x0 = i32(floor(x));
    let y0 = i32(floor(y));
    let z0 = i32(floor(z));
    var acc = vec4<f32>(0.0);
    if (sampling == 1u) {
        let tx = x - floor(x);
        let ty = y - floor(y);
        let tz = z - floor(z);
        for (var dz = 0; dz < 2; dz = dz + 1) {
            for (var dy = 0; dy < 2; dy = dy + 1) {
                for (var dx = 0; dx < 2; dx = dx + 1) {
                    let wx = select(1.0 - tx, tx, dx == 1);
                    let wy = select(1.0 - ty, ty, dy == 1);
                    let wz = select(1.0 - tz, tz, dz == 1);
                    acc = acc + atlas_load(x0 + dx, y0 + dy, z0 + dz) * (wx * wy * wz);
                }
            }
        }
        return acc;
    }
    for (var dz = 0; dz < 4; dz = dz + 1) {
        for (var dy = 0; dy < 4; dy = dy + 1) {
            for (var dx = 0; dx < 4; dx = dx + 1) {
                let ox = dx - 1;
                let oy = dy - 1;
                let oz = dz - 1;
                let w = catmull(x - (floor(x) + f32(ox))) * catmull(y - (floor(y) + f32(oy))) * catmull(z - (floor(z) + f32(oz)));
                acc = acc + atlas_load(x0 + ox, y0 + oy, z0 + oz) * w;
            }
        }
    }
    return acc;
}

fn window_t(raw: f32, lo: f32, hi: f32) -> f32 {
    return clamp((raw * u.dose2.z - lo) / max(hi - lo, 0.000001), 0.0, 1.0);
}

fn wash(raw: f32, lo: f32, hi: f32, under: vec3<f32>, has_ct: bool) -> vec4<f32> {
    let color = ramp_color(window_t(raw, lo, hi));
    let alpha = clamp(u.dose2.w, 0.0, 1.0);
    if (!has_ct) {
        return vec4<f32>(color.rgb, alpha);
    }
    let rgb = under * (1.0 - alpha) + color.rgb * alpha;
    return vec4<f32>(rgb, 1.0);
}

fn dose_fragment(uv: vec2<f32>, coverage: f32, plot_h: f32) -> vec4<f32> {
    let plane = u32(u.dose0.x + 0.5);
    if (plane == 3u) {
        // Elevation lives in dose0.z. Turning the film with it rolls the picture.
        let marched = march_color(uv, plot_h);
        return shade(marched.rgb, marched.a, coverage);
    }
    let film = turned_uv(uv, u32(u.dose0.z + 0.5));
    let nx = max(u.dose1.x, 1.0);
    let ny = max(u.dose1.y, 1.0);
    let nz = max(u.dose1.z, 1.0);
    var raw = 0.0;
    var under = vec3<f32>(0.0);
    var has_ct = false;
    if (u.dose0.w > 0.5) {
        let cols = max(u.dose6.x, 1.0);
        let rows = max(u.dose6.y, 1.0);
        let ix = i32(round(film.x * cols - 0.5));
        let iy = i32(round((1.0 - film.y) * rows - 0.5));
        let dims = textureDimensions(dose_int);
        let x = clamp(ix, 0, i32(dims.x) - 1);
        let y = clamp(iy + i32(u.dose6.z), 0, i32(dims.y) - 1);
        raw = textureLoad(dose_int, vec2<i32>(x, y), 0).r;
        let color = wash(raw, u.dose2.x, u.dose2.y, under, false);
        return shade(color.rgb, color.a, coverage);
    }
    // Axial is X across and Y up. Coronal is X across and depth up.
    // Sagittal is Y across and depth up, the same axes as the field box.
    var vx = film.x * nx - 0.5;
    var vy = (1.0 - film.y) * ny - 0.5;
    var vz = u.dose0.y;
    if (plane == 1u) {
        vy = u.dose0.y;
        vz = (1.0 - film.y) * nz - 0.5;
    } else if (plane == 2u) {
        vx = u.dose0.y;
        vy = film.x * ny - 0.5;
        vz = (1.0 - film.y) * nz - 0.5;
    }
    let texel = sample_grid(vx, vy, vz);
    raw = atlas_dose(texel.r);
    has_ct = texel.b > 0.5;
    under = vec3<f32>(texel.g);
    let color = wash(raw, u.dose2.x, u.dose2.y, under, has_ct);
    return shade(color.rgb, color.a, coverage);
}

// Vispy turntable march. The same steps are `scan_kit_core::ray_rgba`.
// Empty 8³ bricks are skipped. The brick grid is dilated, so a skip is exact.
const VIEW_DEPTH_MM: f32 = 8.0;
const DOSE_FLOOR: f32 = 0.02;

fn vol_at(ix: i32, iy: i32, iz: i32) -> f32 {
    return textureLoad(dose_vol, vec3<i32>(ix + 2, iy + 2, iz + 2), 0).r;
}

fn vol_hw(p: vec3<f32>) -> f32 {
    let dims = vec3<f32>(textureDimensions(dose_vol));
    let voxel = max(u.dose5.w, 0.000001);
    let cell = (p - u.dose4.xyz) / voxel;
    let uv = (cell + vec3<f32>(2.0)) / dims;
    return textureSampleLevel(dose_vol, dose_samp, uv, 0.0).r;
}

fn vol_cubic(q: vec3<f32>) -> f32 {
    let x0 = i32(floor(q.x));
    let y0 = i32(floor(q.y));
    let z0 = i32(floor(q.z));
    var acc = 0.0;
    for (var dz = -1; dz < 3; dz = dz + 1) {
        for (var dy = -1; dy < 3; dy = dy + 1) {
            for (var dx = -1; dx < 3; dx = dx + 1) {
                let w = catmull(q.x - f32(x0 + dx)) * catmull(q.y - f32(y0 + dy)) * catmull(q.z - f32(z0 + dz));
                acc = acc + vol_at(x0 + dx, y0 + dy, z0 + dz) * w;
            }
        }
    }
    return acc;
}

fn vol_sample(p: vec3<f32>) -> f32 {
    let voxel = max(u.dose5.w, 0.000001);
    let q = (p - u.dose4.xyz) / voxel - 0.5;
    let sampling = u32(u.dose3.y + 0.5);
    if (sampling == 0u) {
        // Python nearest is floor(cell). q is cell - 0.5.
        let cell = q + 0.5;
        return vol_at(i32(floor(cell.x)), i32(floor(cell.y)), i32(floor(cell.z)));
    }
    if (sampling == 2u) {
        return vol_cubic(q);
    }
    return vol_hw(p);
}

fn brick_of(cell: vec3<f32>) -> f32 {
    let b = vec3<i32>(floor(cell / 8.0));
    let dims = textureDimensions(dose_brick);
    if (b.x < 0 || b.y < 0 || b.z < 0 || b.x >= i32(dims.x) || b.y >= i32(dims.y) || b.z >= i32(dims.z)) {
        return 0.0;
    }
    return textureLoad(dose_brick, b, 0).r;
}

fn axis_step(cell: f32, dir_mm: f32, voxel: f32, best: f32) -> f32 {
    let dir_i = dir_mm / voxel;
    if (abs(dir_i) < 1e-8) { return best; }
    let brick = floor(cell / 8.0) * 8.0;
    var bound = brick;
    if (dir_i > 0.0) { bound = brick + 8.0; }
    let step = (bound - cell) / dir_i;
    if (step > 0.0001) { return min(best, step); }
    return best;
}

fn steps_to_leave(cell: vec3<f32>, dir: vec3<f32>, voxel: f32) -> i32 {
    var delta = 1e30;
    delta = axis_step(cell.x, dir.x, voxel, delta);
    delta = axis_step(cell.y, dir.y, voxel, delta);
    delta = axis_step(cell.z, dir.z, voxel, delta);
    let steps = ceil(delta / voxel);
    if (steps < 1.0 || steps > 1e8) { return 1; }
    return i32(steps);
}

fn clip_axis(t: vec2<f32>, origin: f32, dir: f32, bmin: f32, bmax: f32) -> vec3<f32> {
    if (abs(dir) < 1e-8) {
        let ok = select(0.0, 1.0, origin >= bmin && origin <= bmax);
        return vec3<f32>(t.x, t.y, ok);
    }
    var near = (bmin - origin) / dir;
    var far = (bmax - origin) / dir;
    if (near > far) {
        let swap = near;
        near = far;
        far = swap;
    }
    let t0 = max(t.x, near);
    let t1 = min(t.y, far);
    return vec3<f32>(t0, t1, select(0.0, 1.0, t0 <= t1));
}

fn march_basis(azimuth: f32, elevation: f32) -> mat3x3<f32> {
    let ce = cos(elevation);
    let se = sin(elevation);
    let ca = cos(azimuth);
    let sa = sin(azimuth);
    let look = normalize(vec3<f32>(-ce * sa, ce * ca, -se));
    let right = normalize(vec3<f32>(ca, sa, 0.0));
    let up = normalize(cross(right, look));
    return mat3x3<f32>(look, right, up);
}

fn film_fy(extent: vec3<f32>, aspect: f32, zoom: f32) -> f32 {
    let pad_x = max(extent.x * 0.12, 1.0);
    let pad_y = max(extent.y * 0.12, 1.0);
    let pad_z = max(extent.z * 0.12, 1.0);
    var rx = extent.x + 2.0 * pad_x;
    var ry = extent.y + 2.0 * pad_y;
    var rz = extent.z + 2.0 * pad_z;
    if (aspect > 1.0) {
        rx = rx / aspect;
        ry = ry / aspect;
    } else {
        rz = rz * aspect;
    }
    let rxs = sqrt(rx * rx + ry * ry);
    let rys = sqrt(rx * rx + ry * ry + rz * rz);
    let scale = max(rxs, rys) * 1.04 * zoom;
    if (aspect > 1.0) { return scale; }
    return scale / aspect;
}

fn march_window(mode: u32, typical: f32) -> vec2<f32> {
    let gain = max(u.dose2.z, 0.01);
    let lo = u.dose2.x;
    let hi = u.dose2.y;
    let line = max(u.dose5.y, 0.000001);
    if (mode == 0u) {
        if (lo < 0.0) {
            let reach = line / gain;
            return vec2<f32>(-reach, reach);
        }
        let peak = max(typical, 0.000001);
        let flo = clamp(lo / peak, 0.0, 1.0);
        let fhi = max(hi / peak, flo + 0.0001);
        let scale = line / gain;
        return vec2<f32>(flo * scale, fhi * scale);
    }
    if (lo < 0.0) {
        let reach = typical / gain;
        return vec2<f32>(-reach, reach);
    }
    return vec2<f32>(lo / gain, max(hi, lo + 0.000001) / gain);
}

fn fog_sample(dose: f32, step: f32, typical: f32) -> vec4<f32> {
    let gain = clamp(u.dose2.z, 0.0, 1.0);
    let reach = max(typical, 0.000001);
    let lo = u.dose2.x;
    let hi = u.dose2.y;
    let t = clamp((dose - lo) / max(hi - lo, 0.000001), 0.0, 1.0);
    let rgb = ramp_color(t).rgb;
    var alpha = 0.0;
    if (lo < 0.0) {
        alpha = clamp(abs(dose / reach), 0.0, 1.0) * gain * step / VIEW_DEPTH_MM;
    } else {
        let tau = gain * max(dose, 0.0) / reach * step / VIEW_DEPTH_MM;
        alpha = 1.0 - exp(-tau);
    }
    return vec4<f32>(rgb, clamp(alpha, 0.0, 1.0));
}

fn gantry_cs() -> vec2<f32> {
    let th = radians(u.dose3.w);
    return vec2<f32>(cos(th), sin(th));
}

fn view_extent_of(extent: vec3<f32>) -> vec3<f32> {
    let cs = gantry_cs();
    let c = abs(cs.x);
    let s = abs(cs.y);
    return vec3<f32>(extent.x, c * extent.y + s * extent.z, s * extent.y + c * extent.z);
}

fn to_lattice(p: vec3<f32>, origin: vec3<f32>, extent: vec3<f32>) -> vec3<f32> {
    let cs = gantry_cs();
    let center = origin + extent * 0.5;
    let d = p - center;
    return center + vec3<f32>(d.x, cs.x * d.y + cs.y * d.z, -cs.y * d.y + cs.x * d.z);
}

fn lattice_direction(dir: vec3<f32>) -> vec3<f32> {
    let cs = gantry_cs();
    return vec3<f32>(dir.x, cs.x * dir.y + cs.y * dir.z, -cs.y * dir.y + cs.x * dir.z);
}

fn zero_color() -> vec4<f32> {
    let mode = u32(u.dose3.x + 0.5);
    let typical = max(u.dose5.z, 0.000001);
    let window = march_window(mode, typical);
    let t = clamp((0.0 - window.x) / max(window.y - window.x, 0.000001), 0.0, 1.0);
    let color = ramp_color(t);
    return vec4<f32>(color.rgb, 1.0);
}

fn to_view_pt(q: vec3<f32>, origin: vec3<f32>, extent: vec3<f32>) -> vec3<f32> {
    let cs = gantry_cs();
    let center = origin + extent * 0.5;
    let d = q - center;
    return center + vec3<f32>(d.x, cs.x * d.y - cs.y * d.z, cs.y * d.y + cs.x * d.z);
}

fn nearer(eye: vec3<f32>, dir: vec3<f32>, a: vec3<f32>, b: vec3<f32>, best: vec2<f32>) -> vec2<f32> {
    let v = b - a;
    let w = eye - a;
    let bv = dot(dir, v);
    let c = max(dot(v, v), 0.00000001);
    let d = dot(dir, w);
    let e = dot(v, w);
    let denom = c - bv * bv;
    var s = 0.0;
    if (abs(denom) > 0.00001) {
        s = clamp((e - bv * d) / denom, 0.0, 1.0);
    } else {
        s = clamp(dot(eye - a, v) / c, 0.0, 1.0);
    }
    let q = a + v * s;
    let t = dot(q - eye, dir);
    let gap = length((eye + dir * t) - q);
    if (gap < best.y) {
        return vec2<f32>(t, gap);
    }
    return best;
}

fn lattice_corner(lo: vec3<f32>, hi: vec3<f32>, bits: i32, origin: vec3<f32>, extent: vec3<f32>) -> vec3<f32> {
    let x = select(lo.x, hi.x, (bits & 1) == 1);
    let y = select(lo.y, hi.y, (bits & 2) == 2);
    let z = select(lo.z, hi.z, (bits & 4) == 4);
    return to_view_pt(vec3<f32>(x, y, z), origin, extent);
}

fn box_near(eye: vec3<f32>, dir: vec3<f32>, lo: vec3<f32>, hi: vec3<f32>, origin: vec3<f32>, extent: vec3<f32>) -> vec2<f32> {
    var best = vec2<f32>(0.0, 1e30);
    var axis = 0;
    loop {
        if (axis >= 3) { break; }
        var bit = 1;
        if (axis == 1) { bit = 2; }
        else if (axis == 2) { bit = 4; }
        var i = 0;
        loop {
            if (i >= 4) { break; }
            var low = i;
            if (axis == 0) {
                let yb = i - (i / 2) * 2;
                let zb = i / 2;
                low = yb * 2 + zb * 4;
            } else if (axis == 1) {
                let xb = i - (i / 2) * 2;
                let zb = i / 2;
                low = xb + zb * 4;
            }
            best = nearer(eye, dir, lattice_corner(lo, hi, low, origin, extent), lattice_corner(lo, hi, low + bit, origin, extent), best);
            i = i + 1;
        }
        axis = axis + 1;
    }
    return best;
}

fn label_every(count: f32) -> f32 {
    if (count <= 8.0) { return 1.0; }
    if (count <= 16.0) { return 2.0; }
    if (count <= 40.0) { return 5.0; }
    if (count <= 80.0) { return 10.0; }
    if (count <= 160.0) { return 20.0; }
    if (count <= 400.0) { return 50.0; }
    return 100.0;
}

fn tick_major(k: f32, every: f32, k0: f32, k1: f32) -> bool {
    let gap = min(k - k0, k1 - k);
    if (gap < 0.01) { return true; }
    let on_step = abs(k - every * round(k / every)) < 0.01;
    return on_step && gap >= 0.6 * every;
}

fn edge_root(axis: i32, mark: f32, origin: vec3<f32>, z1: f32) -> vec3<f32> {
    if (axis == 0) { return vec3<f32>(mark, origin.y, z1); }
    if (axis == 1) { return vec3<f32>(origin.x, mark, z1); }
    return vec3<f32>(origin.x, origin.y, mark);
}

fn hang_ticks(best: vec2<f32>, eye: vec3<f32>, dir: vec3<f32>, origin: vec3<f32>, extent: vec3<f32>, axis: i32, out: vec3<f32>, stub: f32) -> vec2<f32> {
    var nearest = best;
    let z1 = origin.z + extent.z;
    var lo = origin.x;
    var hi = origin.x + extent.x;
    if (axis == 1) {
        lo = origin.y;
        hi = origin.y + extent.y;
    } else if (axis == 2) {
        lo = origin.z;
        hi = z1;
    }
    let k0 = ceil(lo / 10.0 - 0.000001);
    let k1 = floor(hi / 10.0 + 0.000001);
    let every = label_every(max(k1 - k0 + 1.0, 0.0));
    var k = k0;
    var n = 0;
    loop {
        if (k > k1 + 0.01 || n >= 96) { break; }
        let mark = k * 10.0;
        let len = select(stub * 0.55, stub, tick_major(k, every, k0, k1));
        let root = edge_root(axis, mark, origin, z1);
        nearest = nearer(eye, dir, to_view_pt(root, origin, extent), to_view_pt(root + out * len, origin, extent), nearest);
        k = k + 1.0;
        n = n + 1;
    }
    if (k1 < k0 || abs(lo - k0 * 10.0) > 0.05) {
        let root = edge_root(axis, lo, origin, z1);
        nearest = nearer(eye, dir, to_view_pt(root, origin, extent), to_view_pt(root + out * stub, origin, extent), nearest);
    }
    if (k1 < k0 || abs(hi - k1 * 10.0) > 0.05) {
        let root = edge_root(axis, hi, origin, z1);
        nearest = nearer(eye, dir, to_view_pt(root, origin, extent), to_view_pt(root + out * stub, origin, extent), nearest);
    }
    return nearest;
}

// Centimetre ticks on the three edges that meet at the entrance corner.
// dose6.w bits 0..2 pick EDGE_OUT's second direction for X, Y, and depth.
// Numbered ticks are the long ones; they match axis_ticks.
fn tick_near(eye: vec3<f32>, dir: vec3<f32>, origin: vec3<f32>, extent: vec3<f32>) -> vec2<f32> {
    let stub = clamp(0.02 * max(extent.x, max(extent.y, extent.z)), 1.5, 6.0);
    let code = u32(u.dose6.w + 0.5);
    var best = vec2<f32>(0.0, 1e30);
    let x_out = select(vec3<f32>(0.0, -1.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), (code & 1u) == 1u);
    let y_out = select(vec3<f32>(-1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), (code & 2u) == 2u);
    let z_out = select(vec3<f32>(-1.0, 0.0, 0.0), vec3<f32>(0.0, -1.0, 0.0), (code & 4u) == 4u);
    best = hang_ticks(best, eye, dir, origin, extent, 0, x_out, stub);
    best = hang_ticks(best, eye, dir, origin, extent, 1, y_out, stub);
    return hang_ticks(best, eye, dir, origin, extent, 2, z_out, stub);
}

fn clip_box(eye: vec3<f32>, dir: vec3<f32>, bmin: vec3<f32>, bmax: vec3<f32>) -> vec3<f32> {
    var clip = clip_axis(vec2<f32>(0.0, 1e30), eye.x, dir.x, bmin.x, bmax.x);
    if (clip.z < 0.5) { return clip; }
    clip = clip_axis(clip.xy, eye.y, dir.y, bmin.y, bmax.y);
    if (clip.z < 0.5) { return clip; }
    return clip_axis(clip.xy, eye.z, dir.z, bmin.z, bmax.z);
}

// World size of one picture pixel at distance `t` along the ray. `plot_h` is the
// picture, not the canvas: the volume is one cell of a taller frame.
fn world_per_pixel(t: f32, fy: f32, plot_h: f32) -> f32 {
    let span = max(plot_h, 1.0);
    let fov = min(u.dose7.w, 3.124139);
    if (fov <= 0.0001) {
        return fy / span;
    }
    return max(t, 0.001) * 2.0 * tan(fov * 0.5) / span;
}

// best.xyz is ink. best.w is the gap, or 1e30 when nothing has hit.
fn keep_guide(best: vec4<f32>, t: f32, gap: f32, ink: vec3<f32>, t_lo: f32, t_hi: f32, fy: f32, plot_h: f32) -> vec4<f32> {
    let px = 1.5 * world_per_pixel(t, fy, plot_h);
    if (t < t_lo - px || t > t_hi + px || gap > px || gap >= best.w) {
        return best;
    }
    return vec4<f32>(ink, gap);
}

fn over_guides(color: vec4<f32>, eye: vec3<f32>, dir: vec3<f32>, origin: vec3<f32>, extent: vec3<f32>, hit_t: f32, t0: f32, t1: f32, fy: f32, plot_h: f32) -> vec4<f32> {
    var box_ink = vec3<f32>(0.55, 0.58, 0.62);
    if (u.dose9.w > 0.5) {
        box_ink = vec3<f32>(0.25, 0.8, 0.85);
    }
    var best = vec4<f32>(0.0, 0.0, 0.0, 1e30);
    let lattice = box_near(eye, dir, origin, origin + extent, origin, extent);
    best = keep_guide(best, lattice.x, lattice.y, box_ink, t0, hit_t, fy, plot_h);
    let ticks = tick_near(eye, dir, origin, extent);
    best = keep_guide(best, ticks.x, ticks.y, vec3<f32>(0.62, 0.66, 0.7), t0, hit_t, fy, plot_h);
    if (u.dose8.w > 0.5) {
        let field = box_near(eye, dir, u.dose8.xyz, u.dose9.xyz, origin, extent);
        // Through the slab. A nearer lattice edge that the dose hides must not erase it.
        best = keep_guide(best, field.x, field.y, vec3<f32>(0.86, 0.68, 0.22), t0, t1, fy, plot_h);
    }
    if (best.w < 1e20) {
        return vec4<f32>(best.xyz, 1.0);
    }
    return color;
}

fn march_color(uv: vec2<f32>, plot_h: f32) -> vec4<f32> {
    let voxel = max(u.dose5.w, 0.000001);
    let dims = max(u.dose6.xyz, vec3<f32>(1.0));
    let extent = dims * voxel;
    let origin = u.dose4.xyz;
    let gantry = u.dose3.w;
    var box_o = origin;
    var box_e = extent;
    if (abs(gantry) > 0.001) {
        box_e = view_extent_of(extent);
        let lcenter = origin + extent * 0.5;
        box_o = lcenter - box_e * 0.5;
    }
    let center = origin + extent * 0.5 + u.dose7.xyz;
    let aspect = max(u.dose5.x, 0.001);
    let zoom = max(u.dose4.w, 0.05);
    let fy = film_fy(box_e, aspect, zoom);
    // 179 degrees. A 180 degree field makes tan(fov/2) infinite.
    let fov = min(u.dose7.w, 3.124139);
    let basis = march_basis(u.dose0.y, u.dose0.z);
    let look = basis[0];
    let right = basis[1];
    let up = basis[2];
    let x = uv.x * 2.0 - 1.0;
    let y = (1.0 - uv.y) * 2.0 - 1.0;
    var eye: vec3<f32>;
    var dir: vec3<f32>;
    var distance: f32;
    if (fov <= 0.0001) {
        let half_y = fy * 0.5;
        distance = length(box_e) + half_y;
        eye = center - look * distance + right * x * aspect * half_y + up * y * half_y;
        dir = look;
    } else {
        distance = fy / (2.0 * tan(fov * 0.5));
        let tan_y = tan(fov * 0.5);
        eye = center - look * distance;
        dir = normalize(look + right * x * aspect * tan_y + up * y * tan_y);
    }
    let bmax = box_o + box_e;
    // Ticks stick out, and half of a silhouette stroke sits outside the cube.
    // Test that shell before giving up, so a miss still draws the wire.
    let reach = distance + length(box_e);
    let shell = max(6.0, 2.0 * world_per_pixel(reach, fy, plot_h));
    let wide = clip_box(eye, dir, box_o - vec3<f32>(shell), bmax + vec3<f32>(shell));
    if (wide.z < 0.5 || wide.y < 0.0) { return zero_color(); }
    let clip = clip_box(eye, dir, box_o, bmax);
    let t0 = max(clip.x, 0.0);
    let t1 = clip.y;
    let dist = t1 - t0;
    if (clip.z < 0.5 || dist < 0.001) {
        return over_guides(zero_color(), eye, dir, origin, extent, wide.y, 0.0, wide.y, fy, plot_h);
    }
    let nstep = i32(clamp(ceil(dist / voxel), 1.0, 1024.0));
    let entry = eye + dir * t0;
    let mode = u32(u.dose3.x + 0.5);
    let typical = max(u.dose5.z, 0.000001);
    var integ = 0.0;
    var peak = 0.0;
    var trans = 1.0;
    var col = vec3<f32>(0.0);
    var wsum = 0.0;
    var wpos = 0.0;
    var peak_t = t0;
    var fog_t = t1;
    var fog_marked = false;
    var i = 0;
    var guard = 0;
    loop {
        if (i >= nstep || guard >= 1024) { break; }
        guard = guard + 1;
        let p = entry + dir * voxel * (f32(i) + 0.5);
        var q = p;
        var step_dir = dir;
        if (abs(gantry) > 0.001) {
            q = to_lattice(p, origin, extent);
            step_dir = lattice_direction(dir);
        }
        let cell = (q - origin) / voxel;
        let occupancy = brick_of(cell);
        if (occupancy <= 0.0 || (mode == 1u && occupancy <= peak)) {
            i = i + steps_to_leave(cell, step_dir, voxel);
        } else {
            let dose = vol_sample(q);
            let t_here = t0 + voxel * (f32(i) + 0.5);
            if (mode == 1u) {
                if (dose >= peak) { peak_t = t_here; }
                peak = max(peak, dose);
            } else if (mode == 2u) {
                let before = trans;
                let fog = fog_sample(dose, voxel, typical);
                col = col + trans * fog.rgb * fog.a;
                trans = trans * (1.0 - fog.a);
                if (!fog_marked && before >= 0.5 && trans < 0.5) {
                    fog_t = t_here;
                    fog_marked = true;
                }
                if (trans < 0.02) { break; }
            } else {
                integ = integ + dose * voxel;
                if (abs(dose) >= 0.1 * typical) {
                    wsum = wsum + abs(dose);
                    wpos = wpos + abs(dose) * t_here;
                }
            }
            i = i + 1;
        }
    }
    var hit_t = t1;
    if (mode == 1u) {
        hit_t = peak_t;
    } else if (mode == 2u && fog_marked) {
        hit_t = fog_t;
    } else if (wsum > 0.0) {
        hit_t = wpos / wsum;
    }
    let opacity = clamp(u.dose2.w, 0.0, 1.0);
    var color = zero_color();
    if (mode == 2u) {
        let covered = 1.0 - trans;
        if (covered >= 0.02) {
            color = vec4<f32>(col / covered, covered * opacity);
        }
    } else {
        var shown = integ;
        if (mode == 1u) { shown = peak; }
        let window = march_window(mode, typical);
        let scale = max(max(abs(window.x), abs(window.y)), 0.000001);
        if (abs(shown) >= scale * DOSE_FLOOR) {
            let t = clamp((shown - window.x) / max(window.y - window.x, 0.000001), 0.0, 1.0);
            color = vec4<f32>(ramp_color(t).rgb, opacity);
        } else {
            hit_t = t1;
        }
    }
    return over_guides(color, eye, dir, origin, extent, hit_t, t0, t1, fy, plot_h);
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
    /// DOM `buttons`: bit 0 left, bit 1 right. Zero means left, so older callers still orbit.
    pub buttons: u8,
    pub shift: bool,
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
            buttons: 0,
            shift: false,
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

#[derive(Clone, Debug)]
pub(crate) struct PointRec {
    pub p: [f32; 3],
    pub radius: f32,
    pub color: [f32; 4],
    pub id: u32,
    /// Sample time. NaN is drawn in every playhead window.
    pub time: f32,
}

impl PartialEq for PointRec {
    fn eq(&self, other: &Self) -> bool {
        self.p == other.p
            && self.radius == other.radius
            && self.color == other.color
            && self.id == other.id
            && self.time.to_bits() == other.time.to_bits()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct QuadRec {
    pub a: [f32; 2],
    pub b: [f32; 2],
    /// Third corner. `None` is an axis-aligned quad from `a` to `b`.
    pub c: Option<[f32; 2]>,
    /// `> 2.5` is a screen-space color ramp, `> 1.5` samples the volume,
    /// `> 0.5` is a baked heatmap, `< -1.5` is a screen-space tick. Otherwise `color`.
    pub heat: f32,
    pub color: [f32; 4],
    pub id: u32,
    /// Index into `Marks::heatmaps`.
    pub heatmap: Option<usize>,
}

struct GlyphRec {
    /// `w` > 0.5 anchors in data space. `w` < −0.5 rotates screen text by `z`
    /// radians, clockwise with y down. Other screen text with `z` > 0.5 runs up the axis.
    anchor: [f32; 4],
    offset: [f32; 2],
    size: [f32; 2],
    uv: [f32; 4],
    color: [f32; 4],
}

/// One point series in a panel. A timed run is sorted by [`PointRec::time`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct PointRun {
    pub start: u32,
    pub count: u32,
    pub timed: bool,
    /// Samples for a density or contour. Playback reads them and does not draw them.
    #[serde(default)]
    pub source: bool,
}

/// A density or contour counted from a timed point run.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct CloudBatch {
    pub start: u32,
    pub count: u32,
    pub x0: f32,
    pub x1: f32,
    pub y0: f32,
    pub y1: f32,
    pub cols: u32,
    pub rows: u32,
    pub style: CloudStyle,
    #[serde(default)]
    pub cutoff: f32,
    pub heatmap: Option<u32>,
    pub ramp: u8,
    pub color: [f32; 4],
    pub lo: f32,
    pub hi: f32,
    pub fill: [f32; 4],
    pub line: [f32; 4],
    pub thickness: f32,
    pub id: u32,
}

struct LiveCloud {
    panel: usize,
    quads: Vec<QuadRec>,
    lines: Vec<LineRec>,
    dose: bool,
    /// The slice crosshair. Its quad is the ramp zero behind the volume, and
    /// its lines run to the fitted window rather than the volume edge.
    crosshair: bool,
}

/// One line series. A monotonic stroke is sorted by x, so a playhead is one slice.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct LineRun {
    pub start: u32,
    pub count: u32,
    pub monotonic: bool,
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
    /// Empty means the whole point range is untimed. A payload from this crate
    /// always fills it when the panel has points.
    #[serde(default)]
    pub point_runs: Vec<PointRun>,
    /// Empty means draw `line_start..line_count` whole. A payload from this crate
    /// always fills it when the panel has lines.
    #[serde(default)]
    pub line_runs: Vec<LineRun>,
    /// Timed clouds whose picture is a heatmap or contour of the visible window.
    #[serde(default)]
    pub clouds: Vec<CloudBatch>,
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
    /// Playback replaces density pixels through this texture.
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
/// Uniform struct bytes. The dynamic offset is still 256.
const UNIFORM_BYTES: usize = 240;
/// Vispy turntable home: azimuth 30°, elevation 25°, 45° vertical field of view.
const HOME_AZIMUTH: f32 = 30.0 * std::f32::consts::PI / 180.0;
const HOME_ELEVATION: f32 = 25.0 * std::f32::consts::PI / 180.0;
const ORBIT_STEP: f32 = 15.0 * std::f32::consts::PI / 180.0;
pub(crate) const POINT_STRIDE: u64 = 48;
pub(crate) const QUAD_STRIDE: u64 = 64;
/// Cells across one axis of the panel range. Samples in one cell upload once.
const POINT_CELLS: u32 = 4096;
const GLYPH_STRIDE: u64 = 64;

/// Canvas and offscreen sides are clamped to this range in device pixels.
pub(crate) const MIN_SIDE: u32 = 16;
pub(crate) const MAX_SIDE: u32 = 8192;

struct VolumeDraw {
    group: wgpu::BindGroup,
    origin: [f32; 3],
    voxel: f32,
    line_scale: f32,
    shape: [f32; 3],
}

pub struct Plot {
    /// Panel titles, ranges, and category labels. Series live in `marks`.
    panels: Vec<Panel>,
    columns: u32,
    weights: Vec<f32>,
    row_weights: Vec<f32>,
    row_splits: Vec<f32>,
    side: u32,
    /// Gap under the pointer while a divider is dragged. Cleared when the button lifts.
    split_grab: Option<SplitGrab>,
    /// Uploaded dose grid. Paging and orbit sample it; they do not replace it.
    dose: Option<crate::dose::DoseGrid>,
    /// Atlas heatmap that holds the dose slices and the ramp row.
    dose_atlas: Option<usize>,
    /// The ramp row changed after the atlas texture was uploaded.
    ramp_dirty: bool,
    cursor: [usize; 3],
    azimuth: f32,
    elevation: f32,
    orbit_zoom: f32,
    /// Radians. Zero is orthographic, matching vispy `fov == 0`.
    fov: f32,
    /// Last perspective field of view, restored by the ortho toggle.
    persp_fov: f32,
    /// Shift-drag look-at offset, millimetres.
    pan: [f32; 3],
    /// Float dose and dilated bricks for the 3D march. Absent until the first draw.
    volume_draw: Option<VolumeDraw>,
    turns: Vec<u8>,
    /// Field outlines in slice millimetres. Drawn on the film, so they track the picture.
    guides: Vec<Vec<LineRec>>,
    integral: Vec<bool>,
    profile_integral: Vec<bool>,
    /// Y window to restore when a profile leaves its integral.
    profile_window: Vec<Option<(f32, f32)>>,
    /// Increments only when the volume atlas is built.
    atlas_stamp: u32,
    cameras: Vec<Camera>,
    home: Vec<Camera>,
    background: [f32; 4],
    foreground: [f32; 4],
    marks: Marks,
    /// Packed line, point, and quad bytes from the payload. Taken on upload.
    pub(crate) encoded: Option<crate::payload::EncodedMarks>,
    /// Heat textures reused when the next payload has the same pixels.
    kept_heats: Option<Vec<HeatGpu>>,
    /// Trace lines already on the GPU. A later payload can replace the tail.
    kept_line_buf: Option<wgpu::Buffer>,
    line_token: u64,
    reuse_lines: bool,
    line_prefix: u32,
    /// Playhead window for timed points. `None` draws every sample.
    time_window: Option<(f32, f32)>,
    /// Contour fills and isolines for the current playhead. Rebuilt from the point buffer.
    live: Vec<LiveCloud>,
    gpu: Option<GpuMarks>,
    frame_buf: Option<wgpu::Buffer>,
    text_buf: Option<wgpu::Buffer>,
    /// Screen-space color axis. Drawn with the labels, outside the plot scissor.
    color_buf: Option<wgpu::Buffer>,
    cloud_quads: Option<wgpu::Buffer>,
    cloud_lines: Option<wgpu::Buffer>,
    heat_revision: u32,
    heat_uploaded: u32,
    uniform_buf: Option<wgpu::Buffer>,
    uniform_groups: Vec<wgpu::BindGroup>,
    mark_uploads: u32,
    size: (u32, u32),
    /// Toolbar band above a dose cell, in framebuffer pixels. Zero leaves the title gutter alone.
    chrome: f32,
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
    pub(crate) fn from_parts(
        panels: Vec<Panel>,
        columns: u32,
        weights: Vec<f32>,
        row_weights: Vec<f32>,
        row_splits: Vec<f32>,
        side: u32,
        marks: Marks,
        background: [f32; 4],
        foreground: [f32; 4],
    ) -> Self {
        let cameras = panels
            .iter()
            .map(|panel| Camera::new(panel.xmin, panel.xmax, panel.ymin, panel.ymax))
            .collect::<Vec<_>>();
        let panel_count = panels.len();
        let mut plot = Self {
            marks,
            encoded: None,
            kept_heats: None,
            kept_line_buf: None,
            line_token: 0,
            reuse_lines: false,
            line_prefix: 0,
            time_window: None,
            live: Vec::new(),
            panels,
            columns,
            weights,
            row_weights,
            row_splits,
            side,
            split_grab: None,
            dose: None,
            dose_atlas: None,
            ramp_dirty: false,
            cursor: [0, 0, 0],
            azimuth: HOME_AZIMUTH,
            elevation: HOME_ELEVATION,
            orbit_zoom: 1.0,
            fov: scan_kit_core::FOV_Y,
            persp_fov: scan_kit_core::FOV_Y,
            pan: [0.0; 3],
            volume_draw: None,
            turns: vec![0; panel_count],
            guides: vec![Vec::new(); panel_count],
            integral: vec![false; panel_count],
            profile_integral: vec![false; panel_count],
            profile_window: vec![None; panel_count],
            atlas_stamp: 0,
            home: cameras.clone(),
            cameras,
            background,
            foreground,
            gpu: None,
            frame_buf: None,
            text_buf: None,
            color_buf: None,
            cloud_quads: None,
            cloud_lines: None,
            heat_revision: 0,
            heat_uploaded: 0,
            uniform_buf: None,
            uniform_groups: Vec::new(),
            mark_uploads: 0,
            size: (0, 0),
            chrome: 0.0,
            #[cfg(not(target_arch = "wasm32"))]
            offscreen: None,
            #[cfg(not(target_arch = "wasm32"))]
            stage: None,
        };
        plot.rebuild_live();
        plot
    }

    pub fn new(scene: &PlotScene, background: [f32; 4], foreground: [f32; 4]) -> Self {
        let mut plot = Self::from_parts(
            header_panels(&scene.panels),
            scene.columns,
            scene.column_weights.clone(),
            scene.row_weights.clone(),
            scene.row_splits.clone(),
            scene.side,
            build_marks(&scene.panels),
            background,
            foreground,
        );
        plot.attach_volume(&scene.volume);
        plot
    }

    pub(crate) fn attach_volume(&mut self, mark: &scan_kit_core::VolumeMark) {
        let Some(mut grid) = crate::dose::grid_from(
            mark.values.clone(),
            mark.ct.clone(),
            mark.labels.clone(),
            mark.shape,
            mark.origin,
            mark.voxel,
            mark.ramp,
            mark.lo,
            mark.hi,
            if mark.gain == 0.0 { 1.0 } else { mark.gain },
            if mark.opacity == 0.0 {
                1.0
            } else {
                mark.opacity
            },
            mark.mode,
            mark.filter,
        ) else {
            return;
        };
        grid.gantry = mark.gantry;
        grid.unit = mark.unit.clone();
        grid.show_phantom = mark.show_phantom;
        grid.field = mark.field;
        grid.session_colors = mark.session_colors.clone();
        grid.session_focus = mark.session_focus as usize;
        grid.sessions = mark.sessions.iter().map(session_volume).collect();
        grid.companions = mark.companions.iter().map(session_volume).collect();
        grid.plans = mark.plans.iter().map(session_volume).collect();
        crate::dose::remember_window(&mut grid, mark.base_lo, mark.base_hi);
        let (pixels, cols, rows) = crate::dose::atlas_bytes(&grid);
        let atlas = self.marks.heatmaps.len();
        self.marks.heatmaps.push(pixels);
        self.marks.heatmap_size.push((cols, rows));
        self.cursor = grid.peak_at;
        self.guides = vec![Vec::new(); self.panels.len()];
        for index in 0..self.panels.len() {
            let title = self.panels[index].title.clone();
            if dose_plane(&title).is_some_and(|plane| plane < 3) {
                let start = self.marks.panels[index].line_start as usize;
                let count = self.marks.panels[index].line_count as usize;
                self.guides[index] = self
                    .marks
                    .lines
                    .get(start..start + count)
                    .map(<[LineRec]>::to_vec)
                    .unwrap_or_default();
                self.marks.panels[index].line_count = 0;
                self.marks.panels[index].line_runs.clear();
            }
            if dose_plane(&title).is_some() {
                let quad_index = self.marks.panels[index].heats.last().map(|slot| slot.0);
                if let Some(quad_index) = quad_index {
                    if let Some(quad) = self.marks.quads.get_mut(quad_index as usize) {
                        quad.heat = 2.0;
                        quad.heatmap = Some(atlas);
                    }
                    if let Some(slot) = self.marks.panels[index].heats.last_mut() {
                        slot.1 = atlas;
                    }
                }
            }
            if profile_kind(&title).is_some() {
                self.marks.panels[index].line_count = 0;
            }
        }
        self.dose = Some(grid);
        self.dose_atlas = Some(atlas);
        self.ramp_dirty = false;
        self.volume_draw = None;
        self.atlas_stamp = self.atlas_stamp.wrapping_add(1);
        self.heat_revision = self.heat_revision.wrapping_add(1);
        if let Some(encoded) = self.encoded.as_mut() {
            encoded.quads = encode_quads(&self.marks.quads);
        }
        self.refresh_dose_overlay();
    }

    fn integrate_scale(&self, panel: usize, aspect: f32) -> f32 {
        let Some(grid) = &self.dose else {
            return 0.0;
        };
        let Some(plane) = self
            .panels
            .get(panel)
            .and_then(|item| dose_plane(&item.title))
        else {
            return 0.0;
        };
        if plane != 3 || grid.mode != 0 {
            return 0.0;
        }
        scan_kit_core::view_ray_scale(
            &grid.volume,
            &grid.bricks,
            grid.brick_shape,
            self.azimuth,
            self.elevation,
            self.orbit_zoom,
            aspect,
            grid.filter,
            self.fov,
            self.pan,
            grid.gantry,
        )
    }

    fn dose_uniform(&self, panel: usize, aspect: f32, viewed: f32) -> [[f32; 4]; 10] {
        let Some(grid) = &self.dose else {
            return [[0.0; 4]; 10];
        };
        let Some(plane) = self
            .panels
            .get(panel)
            .and_then(|item| dose_plane(&item.title))
        else {
            return [[0.0; 4]; 10];
        };
        let [nx, ny, nz] = grid.atlas_shape;
        let (second, third, integral) = if plane == 3 {
            (self.azimuth, self.elevation, 0.0)
        } else {
            let (full, full_n, atlas_n) = match plane {
                1 => (self.cursor[1], grid.volume.shape[1], ny),
                2 => (self.cursor[0], grid.volume.shape[0], nx),
                _ => (self.cursor[2], grid.volume.shape[2], nz),
            };
            let mapped = if full_n == atlas_n {
                full as f32
            } else {
                full as f32 * atlas_n as f32 / full_n.max(1) as f32
            };
            let on = self.integral.get(panel).copied().unwrap_or(false);
            (
                mapped,
                self.turns.get(panel).copied().unwrap_or(0) as f32,
                if on { 1.0 } else { 0.0 },
            )
        };
        let integral_peak = if integral > 0.0 {
            crate::dose::integral_peak(grid, plane)
        } else {
            grid.peak
        };
        let (lo, hi) = if integral > 0.0 {
            crate::dose::integral_limits(grid, plane)
        } else if grid.hi > grid.lo {
            (grid.lo, grid.hi)
        } else {
            (0.0, grid.peak)
        };
        let (origin, march_voxel, axis_line) = self
            .volume_draw
            .as_ref()
            .map(|draw| (draw.origin, draw.voxel, draw.line_scale))
            .unwrap_or((grid.volume.origin, grid.volume.voxel, grid.line_scale));
        // Integrate colors against the brightest ray in this view: Python's auto window.
        let line = if plane == 3 && grid.mode == 0 {
            viewed
        } else {
            axis_line
        };
        let typical = lo.abs().max(hi.abs()).max(1e-6);
        let march_shape = self.volume_draw.as_ref().map(|draw| draw.shape).unwrap_or([
            grid.volume.shape[0] as f32,
            grid.volume.shape[1] as f32,
            grid.volume.shape[2] as f32,
        ]);
        let [full_nx, full_ny, full_nz] = grid.volume.shape;
        let dose6 = if plane == 3 {
            let sides = axis_sides(&grid.volume, dose_ray(self, grid, aspect));
            [
                march_shape[0],
                march_shape[1],
                march_shape[2],
                (sides[0] + sides[1] * 2 + sides[2] * 4) as f32,
            ]
        } else if integral > 0.0 {
            match plane {
                1 => [full_nx as f32, full_nz as f32, full_ny as f32, 0.0],
                2 => [
                    full_ny as f32,
                    full_nz as f32,
                    (full_ny + full_nz) as f32,
                    0.0,
                ],
                _ => [full_nx as f32, full_ny as f32, 0.0, 0.0],
            }
        } else {
            [0.0; 4]
        };
        let field_on = if plane == 3
            && grid.field[1] > grid.field[0]
            && grid.field[3] > grid.field[2]
            && grid.field[5] > grid.field[4]
        {
            1.0
        } else {
            0.0
        };
        let phantom_on = if plane == 3 && grid.show_phantom {
            1.0
        } else {
            0.0
        };
        [
            [plane as f32, second, third, integral],
            [nx as f32, ny as f32, nz as f32, grid.span],
            [lo, hi, grid.gain, grid.opacity],
            [
                grid.mode as f32,
                grid.filter as f32,
                grid.volume.voxel,
                if plane == 3 {
                    grid.gantry
                } else {
                    integral_peak
                },
            ],
            [origin[0], origin[1], origin[2], self.orbit_zoom],
            [aspect, line, typical, march_voxel],
            dose6,
            [self.pan[0], self.pan[1], self.pan[2], self.fov],
            [grid.field[0], grid.field[2], grid.field[4], field_on],
            [grid.field[1], grid.field[3], grid.field[5], phantom_on],
        ]
    }

    fn refresh_dose_overlay(&mut self) {
        self.live.retain(|cloud| !cloud.dose);
        if self.dose.is_none() {
            return;
        }
        let cursor = self.cursor;
        let profile_integral = self.profile_integral.clone();
        let foreground = self.foreground;
        let described: Vec<(usize, String, f32, f32, f32, f32)> = self
            .panels
            .iter()
            .enumerate()
            .map(|(index, panel)| {
                (
                    index,
                    panel.title.clone(),
                    panel.xmin,
                    panel.xmax,
                    panel.ymin,
                    panel.ymax,
                )
            })
            .collect();
        let Some(grid) = self.dose.as_ref() else {
            return;
        };
        let mut extra = Vec::new();
        for (index, title, xmin, xmax, ymin, ymax) in described {
            if let Some(kind) = profile_kind(&title) {
                let lines = profile_lines(
                    grid,
                    kind,
                    cursor,
                    profile_integral.get(index).copied().unwrap_or(false),
                    foreground,
                );
                extra.push(LiveCloud {
                    panel: index,
                    quads: Vec::new(),
                    lines,
                    dose: true,
                    crosshair: false,
                });
            }
            if let Some(plane) = dose_plane(&title) {
                if plane < 3 {
                    let turns = self.turns.get(index).copied().unwrap_or(0);
                    let bounds = [xmin, xmax, ymin, ymax];
                    extra.push(LiveCloud {
                        panel: index,
                        quads: Vec::new(),
                        lines: crosshair_lines(grid, plane, cursor, bounds, turns, foreground),
                        dose: true,
                        crosshair: true,
                    });
                    let mut lines = Vec::new();
                    for (id, a, b) in
                        crate::dose::outlines(grid, plane, cursor_index(plane, cursor))
                    {
                        let a = crate::dose::display_mm(a[0], a[1], bounds, turns);
                        let b = crate::dose::display_mm(b[0], b[1], bounds, turns);
                        lines.push(LineRec {
                            a: [a[0], a[1], 0.0],
                            b: [b[0], b[1], 0.0],
                            color: structure_color(id),
                            thickness: 1.5,
                            id: 0,
                        });
                    }
                    if !lines.is_empty() {
                        extra.push(LiveCloud {
                            panel: index,
                            quads: Vec::new(),
                            lines,
                            dose: true,
                            crosshair: false,
                        });
                    }
                    let mut lines = Vec::new();
                    for line in self.guides.get(index).into_iter().flatten() {
                        let a = crate::dose::display_mm(line.a[0], line.a[1], bounds, turns);
                        let b = crate::dose::display_mm(line.b[0], line.b[1], bounds, turns);
                        lines.push(LineRec {
                            a: [a[0], a[1], 0.0],
                            b: [b[0], b[1], 0.0],
                            color: line.color,
                            thickness: line.thickness,
                            id: line.id,
                        });
                    }
                    if !lines.is_empty() {
                        extra.push(LiveCloud {
                            panel: index,
                            quads: Vec::new(),
                            lines,
                            dose: true,
                            crosshair: false,
                        });
                    }
                    let lines = field_lines(grid.field, plane, bounds, turns);
                    if !lines.is_empty() {
                        extra.push(LiveCloud {
                            panel: index,
                            quads: Vec::new(),
                            lines,
                            dose: true,
                            crosshair: false,
                        });
                    }
                }
            }
        }
        self.live.extend(extra);
    }

    /// Pixel frames of the panel grid, for the cell pickers.
    pub fn frames_json(&self, width: u32, height: u32) -> String {
        let layout = self.layout(width.max(1), height.max(1));
        let frames: Vec<serde_json::Value> = layout
            .iter()
            .map(|cell| {
                serde_json::json!({
                    "x": cell.cell.x,
                    "y": cell.cell.y,
                    "w": cell.cell.w,
                    "h": cell.cell.h,
                    "plotX": cell.plot.x,
                    "plotY": cell.plot.y,
                    "plotW": cell.plot.w,
                    "plotH": cell.plot.h,
                })
            })
            .collect();
        serde_json::Value::Array(frames).to_string()
    }

    /// Local slice tools. `rotate` and `integral` stay on this plot.
    pub fn dose_action(&mut self, panel: usize, action: &str) {
        if self.dose.is_none() || panel >= self.panels.len() {
            return;
        }
        match action {
            "rotate" => {
                if let Some(turns) = self.turns.get_mut(panel) {
                    *turns = turns.wrapping_add(1) % 4;
                }
            }
            "integral" => {
                if dose_plane(&self.panels[panel].title).is_some_and(|plane| plane < 3) {
                    if let Some(flag) = self.integral.get_mut(panel) {
                        *flag = !*flag;
                    }
                } else if profile_kind(&self.panels[panel].title).is_some() {
                    let turning_on = !self.profile_integral.get(panel).copied().unwrap_or(false);
                    if turning_on {
                        if let Some(camera) = self.cameras.get(panel) {
                            if let Some(slot) = self.profile_window.get_mut(panel) {
                                *slot = Some((camera.ymin, camera.ymax));
                            }
                        }
                    }
                    if let Some(flag) = self.profile_integral.get_mut(panel) {
                        *flag = turning_on;
                    }
                    if turning_on {
                        self.fit_integral_profile(panel);
                    } else {
                        // A reload keeps the flag and refits y, but the saved slice
                        // window belongs to the previous lattice. Fall back to this
                        // plot's slice fit so the Gy curve is back in frame.
                        let home = self
                            .home
                            .get(panel)
                            .map(|camera| (camera.ymin, camera.ymax));
                        let saved = self
                            .profile_window
                            .get_mut(panel)
                            .and_then(|slot| slot.take());
                        if let Some(camera) = self.cameras.get_mut(panel) {
                            if let Some((ymin, ymax)) = saved.or(home) {
                                camera.ymin = ymin;
                                camera.ymax = ymax;
                            }
                        }
                        self.mark_profile_unit(panel, false);
                    }
                }
            }
            _ => return,
        }
        self.refresh_dose_overlay();
    }

    /// The integral is Gy·mm² (or the quantity times mm²). Fit y to every row of
    /// that projection so paging cannot push the curve out of the frame.
    fn fit_integral_profile(&mut self, panel: usize) {
        let Some(kind) = self
            .panels
            .get(panel)
            .and_then(|item| profile_kind(&item.title))
        else {
            return;
        };
        let Some(grid) = self.dose.as_ref() else {
            return;
        };
        let (ymin, ymax) = integral_profile_axis(grid, kind);
        if let Some(camera) = self.cameras.get_mut(panel) {
            camera.ymin = ymin;
            camera.ymax = ymax;
        }
        self.mark_profile_unit(panel, true);
    }

    fn mark_profile_unit(&mut self, panel: usize, on: bool) {
        let Some(label) = self.panels.get_mut(panel).map(|item| &mut item.y_label) else {
            return;
        };
        const SUFFIX: &str = "·mm²";
        if on {
            if !label.ends_with(SUFFIX) {
                label.push_str(SUFFIX);
            }
        } else if let Some(base) = label.strip_suffix(SUFFIX) {
            *label = base.to_string();
        }
    }

    /// Blender numpad orbit for the 3D dose cell. Number-row keys match vispy's emulate-numpad.
    pub fn dose_key(&mut self, key: &str, ctrl: bool) -> bool {
        if self.dose.is_none()
            || !self
                .panels
                .iter()
                .any(|panel| dose_plane(&panel.title) == Some(3))
        {
            return false;
        }
        let Some((azimuth, elevation, fov, persp)) = blender_step(
            self.azimuth,
            self.elevation,
            self.fov,
            self.persp_fov,
            key,
            ctrl,
        ) else {
            return false;
        };
        self.azimuth = azimuth;
        self.elevation = elevation;
        self.fov = fov;
        self.persp_fov = persp;
        true
    }

    fn pan_volume(&mut self, cell: &Cell, dx: f32, dy: f32) {
        let extent = {
            let Some(grid) = self.dose.as_ref() else {
                return;
            };
            let voxel = grid.volume.voxel;
            [
                grid.volume.shape[0] as f32 * voxel,
                grid.volume.shape[1] as f32 * voxel,
                grid.volume.shape[2] as f32 * voxel,
            ]
        };
        let aspect = cell.plot.w / cell.plot.h.max(1.0);
        let fy = scan_kit_core::film_height(extent, aspect, self.orbit_zoom);
        let scale = fy / cell.plot.h.max(1.0);
        let (right, up) = turntable_axes(self.azimuth, self.elevation);
        for axis in 0..3 {
            self.pan[axis] -= (right[axis] * dx + up[axis] * dy) * scale;
        }
    }

    #[cfg(test)]
    pub(crate) fn atlas_stamp(&self) -> u32 {
        self.atlas_stamp
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

    #[cfg_attr(not(any(test, target_arch = "wasm32")), allow(dead_code))]
    pub(crate) fn set_chrome(&mut self, px: f32) {
        self.chrome = px.max(0.0);
    }

    /// Window, gain, opacity, ray mode, sample filter, and color scale.
    /// The voxels stay. A ramp change rewrites one texture row.
    #[cfg_attr(not(any(test, target_arch = "wasm32")), allow(dead_code))]
    pub(crate) fn paint(&mut self, spec: &str) -> String {
        if self.dose.is_none() {
            return String::new();
        }
        let spec: serde_json::Value = serde_json::from_str(spec).unwrap_or(serde_json::Value::Null);
        let text = |key: &str| spec.get(key).and_then(|value| value.as_str()).unwrap_or("");
        let level = spec.get("level").and_then(|value| {
            value
                .as_str()
                .and_then(|text| text.parse().ok())
                .or_else(|| value.as_f64().map(|number| number as f32))
        });
        let auto = match spec.get("auto").and_then(|value| value.as_str()) {
            None | Some("") => true,
            Some(text) => matches!(text, "on" | "On" | "true" | "1"),
        };
        let percent = matches!(text("error"), "percent" | "Percent");
        let (base_lo, base_hi, gamma, difference, ramp_now, opacity_now, unit_now) = {
            let grid = self.dose.as_ref().unwrap();
            (
                grid.base_lo,
                grid.base_hi,
                grid.gamma,
                grid.difference,
                grid.ramp,
                grid.opacity,
                grid.unit.clone(),
            )
        };
        let wash = scan_kit_core::wash_of(scan_kit_core::PaintChoice {
            base_lo,
            base_hi,
            gamma,
            difference,
            scale: text("scale"),
            auto,
            level,
            percent,
            ray: text("ray"),
            sample: text("sample"),
        });
        let session = scan_kit_core::is_session(ramp_now);
        let ramp = if session || text("scale").is_empty() {
            ramp_now
        } else {
            wash.ramp
        };
        let unit = scan_kit_core::painted_unit(&unit_now, wash.mode, gamma);
        let ramp_changed = ramp_now != ramp;
        let opacity_changed = (opacity_now - wash.opacity).abs() > 1.0e-4;
        if let Some(index) = self.dose_atlas {
            if ramp_changed || opacity_changed {
                let size = self.marks.heatmap_size.get(index).copied();
                if let (Some((cols, rows)), Some(pixels)) =
                    (size, self.marks.heatmaps.get_mut(index))
                {
                    if ramp_changed && !session {
                        crate::dose::paint_ramp_row(pixels, cols, rows, ramp, wash.opacity);
                    } else {
                        crate::dose::paint_ramp_alpha(pixels, cols, rows, wash.opacity);
                    }
                    if self.gpu.is_some() || self.kept_heats.is_some() {
                        self.ramp_dirty = true;
                    }
                }
            }
        }
        let grid = self.dose.as_mut().unwrap();
        grid.lo = wash.lo;
        grid.hi = wash.hi;
        grid.gain = wash.gain;
        grid.opacity = wash.opacity;
        grid.ramp = ramp;
        grid.mode = wash.mode;
        grid.filter = wash.filter;
        grid.unit = unit;
        wash.level
            .map(scan_kit_core::LevelSpan::json)
            .unwrap_or_default()
    }

    /// Keep a zoomed window when the new panel is the same quantity and its
    /// data range is still close. A different label or a much larger span
    /// starts from the new limits. Extra panels keep their own fit.
    #[cfg_attr(not(any(test, target_arch = "wasm32")), allow(dead_code))]
    pub(crate) fn adopt_view(&mut self, previous: &Plot) {
        self.chrome = previous.chrome;
        if self.panels.len() == previous.panels.len()
            && self.row_splits.len() == previous.row_splits.len()
            && !previous.row_splits.is_empty()
        {
            self.row_splits.clone_from(&previous.row_splits);
            if self.row_weights.len() == previous.row_weights.len()
                && !previous.row_weights.is_empty()
            {
                self.row_weights.clone_from(&previous.row_weights);
            }
        }
        if self.dose.is_some()
            && previous.dose.is_some()
            && self.panels.len() == previous.panels.len()
        {
            self.cursor =
                if let (Some(before), Some(after)) = (previous.dose.as_ref(), self.dose.as_ref()) {
                    carried_cursor(
                        previous.cursor,
                        before.volume.origin,
                        before.volume.voxel,
                        before.volume.shape,
                        after.volume.origin,
                        after.volume.voxel,
                        after.volume.shape,
                    )
                } else {
                    previous.cursor
                };
            self.azimuth = previous.azimuth;
            self.elevation = previous.elevation;
            self.orbit_zoom = previous.orbit_zoom;
            self.fov = previous.fov;
            self.persp_fov = previous.persp_fov;
            self.pan = previous.pan;
            if previous.turns.len() == self.panels.len() {
                self.turns.clone_from(&previous.turns);
            }
            if previous.integral.len() == self.panels.len() {
                self.integral.clone_from(&previous.integral);
            }
            if previous.profile_integral.len() == self.panels.len() {
                self.profile_integral.clone_from(&previous.profile_integral);
            }
            if previous.profile_window.len() == self.panels.len()
                && previous.home.len() == self.home.len()
            {
                for index in 0..self.panels.len() {
                    if axes_close(previous.home[index], self.home[index]) {
                        self.profile_window[index] = previous.profile_window[index];
                    }
                }
            }
            self.refresh_dose_overlay();
        }
        for index in 0..self.panels.len() {
            if self.profile_integral.get(index).copied().unwrap_or(false) {
                self.mark_profile_unit(index, true);
            }
        }
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
        for index in 0..self.panels.len() {
            if self.profile_integral.get(index).copied().unwrap_or(false) {
                self.fit_integral_profile(index);
            }
        }
    }

    /// Slide time-panel cameras to `[lo, hi]`. Timed points and monotonic time
    /// traces draw the samples inside that window. Density and contour rebin
    /// that same window. `on == false` restores the full traces and every point.
    /// A zoom sticks when `force` is false. Spectrum and side panels stay put.
    /// The point buffer is not replaced. The window updates even when a zoom sticks.
    #[cfg_attr(not(any(test, target_arch = "wasm32")), allow(dead_code))]
    pub(crate) fn follow_time(&mut self, on: bool, lo: f32, hi: f32, force: bool) {
        self.time_window = (on && lo.is_finite() && hi.is_finite() && hi >= lo).then_some((lo, hi));
        let (lo, hi) = if on {
            (lo, hi)
        } else {
            (f32::NEG_INFINITY, f32::INFINITY)
        };
        let mut fits = Vec::new();
        let mut x_lo = f32::MAX;
        let mut x_hi = f32::MIN;
        for index in 0..self.panels.len() {
            if !self.is_time_panel(index) {
                continue;
            }
            if !force && self.cameras[index] != self.home[index] {
                continue;
            }
            let span = self.panel_span(index, lo, hi);
            if let Some((start, end, _, _)) = span {
                x_lo = x_lo.min(start);
                x_hi = x_hi.max(end);
            }
            fits.push((index, span));
        }
        let shared = (x_lo <= x_hi).then(|| time_window(x_lo, x_hi));
        for (index, span) in fits {
            let Some((xmin, xmax)) = shared.or_else(|| {
                (on && lo.is_finite() && hi.is_finite() && hi >= lo).then(|| time_window(lo, hi))
            }) else {
                continue;
            };
            let (ymin, ymax) = span.map_or(
                (self.cameras[index].ymin, self.cameras[index].ymax),
                |(_, _, low, high)| (low, high),
            );
            let camera = Camera {
                xmin,
                xmax,
                ymin,
                ymax,
            };
            self.cameras[index] = camera;
            self.home[index] = camera;
        }
        self.rebuild_live();
    }

    /// ponytail: a "before" window counts the whole prefix. A one-second window
    /// is about a thousand points. If that prefix gets slow, keep a running grid.
    fn rebuild_live(&mut self) {
        let window = self.time_window;
        let mut live = Vec::new();
        let mut heat_writes = Vec::new();
        for (panel_index, batch) in self.marks.panels.iter().enumerate() {
            for cloud in &batch.clouds {
                let Some(points) = self
                    .marks
                    .points
                    .get(cloud.start as usize..(cloud.start + cloud.count) as usize)
                else {
                    continue;
                };
                let shown = window_points(points, window);
                let (xs, ys) = samples_of(shown);
                match cloud.style {
                    CloudStyle::Density => {
                        let Some(index) = cloud.heatmap else {
                            continue;
                        };
                        let counts = count_grid(
                            &xs,
                            &ys,
                            cloud.x0,
                            cloud.x1,
                            cloud.y0,
                            cloud.y1,
                            cloud.cols.max(1) as usize,
                        );
                        let pixels = heatmap_bytes(
                            &counts,
                            cloud.cols.max(1),
                            cloud.rows.max(1),
                            cloud.ramp,
                            cloud.color,
                            cloud.lo,
                            cloud.hi,
                        );
                        heat_writes.push((index as usize, pixels));
                    }
                    CloudStyle::Contour => {
                        let drawn = contour_in_frame(
                            &xs,
                            &ys,
                            cloud.cutoff,
                            cloud.x0,
                            cloud.x1,
                            cloud.y0,
                            cloud.y1,
                        );
                        let (quads, lines) = contour_geometry(
                            &drawn,
                            cloud.id,
                            cloud.fill,
                            cloud.line,
                            cloud.thickness,
                        );
                        live.push(LiveCloud {
                            panel: panel_index,
                            quads,
                            lines,
                            dose: false,
                            crosshair: false,
                        });
                    }
                }
            }
        }
        let mut changed = false;
        for (index, pixels) in heat_writes {
            if self
                .marks
                .heatmaps
                .get(index)
                .is_some_and(|have| have != &pixels)
            {
                self.marks.heatmaps[index] = pixels;
                changed = true;
            }
        }
        if changed {
            self.heat_revision = self.heat_revision.wrapping_add(1);
        }
        self.live = live;
        self.refresh_dose_overlay();
    }

    #[cfg_attr(not(any(test, target_arch = "wasm32")), allow(dead_code))]
    fn is_time_panel(&self, index: usize) -> bool {
        // Time traces leave the x label empty. Spectrum, scatter, and the side
        // column keep their own axes.
        self.panels
            .get(index)
            .is_some_and(|panel| panel.x_label.is_empty())
            && index + (self.side as usize) < self.panels.len()
    }

    #[cfg_attr(not(any(test, target_arch = "wasm32")), allow(dead_code))]
    fn panel_span(&self, index: usize, lo: f32, hi: f32) -> Option<(f32, f32, f32, f32)> {
        let batch = self.marks.panels.get(index)?;
        if batch.line_runs.is_empty() {
            let start = batch.line_start as usize;
            let end = start + batch.line_count as usize;
            return follow_span(self.marks.lines.get(start..end)?, &[], lo, hi);
        }
        follow_span(&self.marks.lines, &batch.line_runs, lo, hi)
    }

    /// One draw range per stroke. Time panels clip monotonic strokes to the playhead.
    fn line_draw_ranges(&self) -> Vec<Vec<(u32, u32)>> {
        let windows: Vec<Option<(f32, f32)>> = (0..self.marks.panels.len())
            .map(|index| {
                if self.is_time_panel(index) {
                    self.time_window
                } else {
                    None
                }
            })
            .collect();
        let lines = &self.marks.lines;
        let panels = &self.marks.panels;
        panels
            .iter()
            .zip(windows)
            .map(|(batch, window)| {
                visible_line_draws(
                    lines,
                    batch.line_start,
                    batch.line_count,
                    &batch.line_runs,
                    window,
                )
            })
            .collect()
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
            self.azimuth = HOME_AZIMUTH;
            self.elevation = HOME_ELEVATION;
            self.orbit_zoom = 1.0;
            self.fov = scan_kit_core::FOV_Y;
            self.persp_fov = scan_kit_core::FOV_Y;
            self.pan = [0.0; 3];
            for index in 0..self.panels.len() {
                if self.profile_integral.get(index).copied().unwrap_or(false) {
                    self.fit_integral_profile(index);
                }
            }
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
        let data = self.view_camera(cell.panel, &cell.plot).data_at(
            x,
            y,
            cell.plot,
            width as f32,
            height as f32,
        );
        (true, data[0], data[1], self.pick(cell, x, y, width, height))
    }

    /// Topmost series mark under the pixel, in paint order.
    // ponytail: linear scan over the panel's marks per hover. Past a few million
    // marks, bucket them into a per-panel screen grid on camera change.
    fn pick(&self, cell: &Cell, x: f32, y: f32, width: u32, height: u32) -> Option<u32> {
        let matrix = self.view_camera(cell.panel, &cell.plot).clip_from_data(
            cell.plot,
            width as f32,
            height as f32,
        );
        let (w, h) = (width as f32, height as f32);
        let clip = if self.is_time_panel(cell.panel) {
            self.time_window
        } else {
            None
        };
        let batch = &self.marks.panels[cell.panel];
        let hit = |id: u32| id.checked_sub(1);
        let runs = if batch.point_runs.is_empty() {
            vec![(batch.point_start, batch.point_count)]
        } else {
            batch
                .point_runs
                .iter()
                .filter(|run| !run.source)
                .map(|run| (run.start, run.count))
                .collect::<Vec<_>>()
        };
        for (start, count) in runs.into_iter().rev() {
            let Some(points) = self
                .marks
                .points
                .get(start as usize..(start + count) as usize)
            else {
                continue;
            };
            for point in points.iter().rev() {
                if !point_shown(point.time, self.time_window) {
                    continue;
                }
                let center = project(matrix, point.p, w, h);
                let reach = point.radius.max(1.0) + 0.5;
                if (center[0] - x).powi(2) + (center[1] - y).powi(2) <= reach * reach {
                    return hit(point.id);
                }
            }
        }
        for cloud in self
            .live
            .iter()
            .rev()
            .filter(|cloud| cloud.panel == cell.panel)
        {
            for line in cloud.lines.iter().rev() {
                let a = project(matrix, line.a, w, h);
                let b = project(matrix, line.b, w, h);
                if segment_distance(a, b, [x, y]) <= line.thickness.max(1.0) * 0.5 + 0.5 {
                    return hit(line.id);
                }
            }
        }
        let draws = visible_line_draws(
            &self.marks.lines,
            batch.line_start,
            batch.line_count,
            &batch.line_runs,
            clip,
        );
        for (start, count) in draws.iter().rev() {
            let Some(segments) = self
                .marks
                .lines
                .get(*start as usize..(*start + *count) as usize)
            else {
                continue;
            };
            for line in segments.iter().rev() {
                let a = project(matrix, line.a, w, h);
                let b = project(matrix, line.b, w, h);
                if segment_distance(a, b, [x, y]) <= line.thickness.max(1.0) * 0.5 + 0.5 {
                    return hit(line.id);
                }
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
        for cloud in self
            .live
            .iter()
            .rev()
            .filter(|cloud| cloud.panel == cell.panel)
        {
            if let Some(quad) = cloud.quads.iter().rev().find(|quad| covers(quad)) {
                return hit(quad.id);
            }
        }
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
        if !input.drag {
            self.split_grab = None;
        }
        if !input.drag && input.wheel == 0.0 {
            return;
        }
        if self.drag_split(width, height, input) {
            return;
        }
        if self.dose_pointer(width, height, input) {
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

    fn dose_pointer(&mut self, width: u32, height: u32, input: &PlotInput) -> bool {
        if self.dose.is_none() || (!input.drag && input.wheel == 0.0) {
            return false;
        }
        let layout = self.layout(width, height);
        let Some(index) = hit_cell(&layout, input.x, input.y) else {
            return false;
        };
        let panel = layout[index].panel;
        let Some(plane) = dose_plane(&self.panels[panel].title) else {
            return false;
        };
        if !inside(layout[index].plot, input.x, input.y) {
            return true;
        }
        if plane == 3 {
            let step = 0.5_f32.to_radians();
            let buttons = if input.buttons == 0 { 1 } else { input.buttons };
            let left = (buttons & 1) != 0;
            let right = (buttons & 2) != 0;
            if input.drag && !(left && right) {
                if left && input.shift {
                    self.pan_volume(&layout[index], input.dx, input.dy);
                } else if right && input.shift {
                    // The saved perspective angle stays put. Dragging to 0 is ortho;
                    // the 5 key restores the angle from before that drag.
                    let degrees = (self.fov.to_degrees() - input.dy / 5.0).clamp(0.0, 179.0);
                    self.fov = degrees.to_radians();
                } else if right {
                    self.orbit_zoom = (self.orbit_zoom * 1.007_f32.powf(input.dy)).clamp(0.2, 8.0);
                } else if left {
                    self.azimuth -= input.dx * step;
                    self.elevation = (self.elevation + input.dy * step)
                        .clamp(-std::f32::consts::FRAC_PI_2, std::f32::consts::FRAC_PI_2);
                }
            }
            if input.wheel != 0.0 {
                // A positive wheel is scroll-down. That zooms out, matching the charts.
                self.orbit_zoom = (self.orbit_zoom * (input.wheel * 0.0015).exp()).clamp(0.2, 8.0);
            }
            return true;
        }
        if input.wheel != 0.0 {
            let delta = if input.wheel > 0.0 { 1 } else { -1 };
            let axis = match plane {
                1 => 1,
                2 => 0,
                _ => 2,
            };
            if let Some(grid) = self.dose.as_ref() {
                self.cursor[axis] = crate::dose::page(grid, plane, self.cursor[axis], delta);
            }
            self.refresh_dose_overlay();
            return true;
        }
        if input.drag {
            let cell = &layout[index];
            let [x, y] = self.view_camera(panel, &cell.plot).data_at(
                input.x,
                input.y,
                cell.plot,
                width as f32,
                height as f32,
            );
            let bounds = [
                self.panels[panel].xmin,
                self.panels[panel].xmax,
                self.panels[panel].ymin,
                self.panels[panel].ymax,
            ];
            let turns = self.turns.get(panel).copied().unwrap_or(0);
            let Some([x, y]) = crate::dose::source_mm(x, y, bounds, turns) else {
                return true;
            };
            if let Some(grid) = self.dose.as_ref() {
                match plane {
                    1 => {
                        self.cursor[0] = grid.volume.index_of(0, x);
                        self.cursor[2] = grid.volume.index_of(2, y);
                    }
                    2 => {
                        self.cursor[1] = grid.volume.index_of(1, x);
                        self.cursor[2] = grid.volume.index_of(2, y);
                    }
                    _ => {
                        self.cursor[0] = grid.volume.index_of(0, x);
                        self.cursor[1] = grid.volume.index_of(1, y);
                    }
                }
            }
            self.refresh_dose_overlay();
            return true;
        }
        false
    }

    /// A drag that starts in a gap resizes that row or that row's columns.
    /// The scene is not rebuilt; the weights live on this plot.
    fn drag_split(&mut self, width: u32, height: u32, input: &PlotInput) -> bool {
        if !row_split_layout(
            self.panels.len(),
            self.columns,
            &self.row_weights,
            &self.row_splits,
        ) {
            return false;
        }
        if !input.drag {
            return false;
        }
        if self.split_grab.is_none() {
            self.split_grab = hit_split(
                self.panels.len(),
                width,
                height,
                &self.row_weights,
                &self.row_splits,
                input.x,
                input.y,
            );
        }
        let Some(grab) = self.split_grab else {
            return false;
        };
        let rects = panel_rects(
            self.panels.len(),
            width,
            height,
            self.columns,
            &self.weights,
            &self.row_weights,
            &self.row_splits,
            self.side,
        );
        match grab {
            SplitGrab::Cols(row) => {
                let left = &rects[row * 2];
                let right = &rects[row * 2 + 1];
                let span = (left.w + right.w).max(1.0);
                let pair = row * 2;
                let total = (self.row_splits[pair] + self.row_splits[pair + 1]).max(1.0e-6);
                let next = (self.row_splits[pair] + input.dx / span * total)
                    .clamp(total * 0.12, total * 0.88);
                self.row_splits[pair] = next;
                self.row_splits[pair + 1] = total - next;
            }
            SplitGrab::Rows(row) => {
                let above = &rects[row * 2];
                let below = &rects[(row + 1) * 2];
                let span = (above.h + below.h).max(1.0);
                let total = (self.row_weights[row] + self.row_weights[row + 1]).max(1.0e-6);
                let next = (self.row_weights[row] + input.dy / span * total)
                    .clamp(total * 0.12, total * 0.88);
                self.row_weights[row] = next;
                self.row_weights[row + 1] = total - next;
            }
        }
        true
    }

    fn layout(&self, width: u32, height: u32) -> Vec<Cell> {
        let rects = panel_rects(
            self.panels.len(),
            width,
            height,
            self.columns,
            &self.weights,
            &self.row_weights,
            &self.row_splits,
            self.side,
        );
        let font = atlas();
        rects
            .into_iter()
            .enumerate()
            .map(|(index, cell)| {
                let camera = self.cameras[index];
                let panel = &self.panels[index];
                let plane = dose_plane(&panel.title);
                let edge = font.line_height * 0.5 + 4.0;
                let (mut left, top, bottom, right) = if plane == Some(3) {
                    (4.0, edge, edge, color_column_width())
                } else {
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
                    let right = if plane.is_some() {
                        color_column_width()
                    } else {
                        16.0
                    };
                    (left, top, bottom, right)
                };
                let right = if plane.is_some() {
                    right.min((cell.w - left - 48.0).max(28.0))
                } else {
                    right
                };
                // The view picker and its slice buttons live in this band, so the picture starts below them.
                let top = top.max(self.chrome);
                let mut plot = snap_plot(PlotRect {
                    x: cell.x + left,
                    y: cell.y + top,
                    w: (cell.w - left - right).max(8.0),
                    h: (cell.h - top - bottom).max(8.0),
                });
                if plane.is_some_and(|plane| plane < 3) {
                    let turns = self.turns.get(index).copied().unwrap_or(0);
                    let fitted = fit_domain(camera, &plot, turns);
                    let wider = axis_name_width(&panel.y_label) + y_tick_width(&fitted) + 8.0;
                    if wider > left {
                        left = wider;
                        plot = snap_plot(PlotRect {
                            x: cell.x + left,
                            y: cell.y + top,
                            w: (cell.w - left - right).max(8.0),
                            h: (cell.h - top - bottom).max(8.0),
                        });
                    }
                }
                if panel.equal && plane.is_none() {
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
    fn fill_dose_cpu(
        &self,
        frame: &mut [u8],
        width: u32,
        height: u32,
        matrix: &[f32; 16],
        quad: QuadRec,
        cell: &Cell,
        ray_scale: f32,
    ) {
        let Some(grid) = &self.dose else {
            return;
        };
        let Some(plane) = dose_plane(&self.panels[cell.panel].title) else {
            return;
        };
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
        let aspect = span_x / span_y;
        let view = crate::dose::ViewSample {
            plane,
            index: cursor_index(plane, self.cursor),
            turns: self.turns.get(cell.panel).copied().unwrap_or(0),
            integral: self.integral.get(cell.panel).copied().unwrap_or(false),
            azimuth: self.azimuth,
            elevation: self.elevation,
            zoom: self.orbit_zoom,
            aspect,
            ray_scale,
            fov: self.fov,
            center: self.pan,
        };
        let prepared = crate::dose::prepare_plane(grid, &view);
        for y in top.floor() as i32..=bottom.ceil() as i32 {
            for x in left.floor() as i32..=right.ceil() as i32 {
                let u = ((x as f32 - left) / span_x).clamp(0.0, 1.0);
                let v = ((y as f32 - top) / span_y).clamp(0.0, 1.0);
                let color = crate::dose::shade_uv(grid, &view, prepared.as_ref(), [u, v]);
                blend(frame, width, height, x, y, color, Some(cell.plot));
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn paint_cpu(&mut self, width: u32, height: u32, layout: &[Cell]) -> Vec<u8> {
        self.fit_slice_chrome(layout);
        let line_draws = self.line_draw_ranges();
        let mut frame = Vec::with_capacity((width * height * 4) as usize);
        let pixel = rgba_bytes(self.background);
        for _ in 0..width * height {
            frame.extend_from_slice(&pixel);
        }
        for cell in layout {
            let camera = self.view_camera(cell.panel, &cell.plot);
            let matrix = camera.clip_from_data(cell.plot, width as f32, height as f32);
            let batch = &self.marks.panels[cell.panel];
            let scale = self.integrate_scale(cell.panel, cell.plot.w / cell.plot.h.max(1.0));
            let (grid, border) = panel_frame(self, cell);
            for line in grid {
                stroke_cpu(&mut frame, width, height, &matrix, &line, cell.plot);
            }
            for cloud in self
                .live
                .iter()
                .filter(|cloud| cloud.panel == cell.panel && cloud.crosshair)
            {
                for quad in &cloud.quads {
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
            }
            let solid: Vec<QuadRec> = self.marks.quads
                [batch.quad_start as usize..(batch.quad_start + batch.quad_count) as usize]
                .to_vec();
            let heated: Vec<QuadRec> = batch
                .heats
                .iter()
                .filter_map(|(index, _)| self.marks.quads.get(*index as usize).copied())
                .collect();
            for quad in solid.iter().chain(&heated) {
                if quad.heat > 1.5 && quad.heat < 2.5 {
                    self.fill_dose_cpu(&mut frame, width, height, &matrix, *quad, cell, scale);
                } else {
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
            }
            for cloud in self
                .live
                .iter()
                .filter(|cloud| cloud.panel == cell.panel && !cloud.crosshair)
            {
                for quad in &cloud.quads {
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
            }
            for (start, count) in &line_draws[cell.panel] {
                let Some(segments) = self
                    .marks
                    .lines
                    .get(*start as usize..(*start + *count) as usize)
                else {
                    continue;
                };
                for line in segments {
                    stroke_cpu(&mut frame, width, height, &matrix, line, cell.plot);
                }
            }
            for cloud in self.live.iter().filter(|cloud| cloud.panel == cell.panel) {
                for line in &cloud.lines {
                    stroke_cpu(&mut frame, width, height, &matrix, line, cell.plot);
                }
            }
            let runs = if batch.point_runs.is_empty() {
                vec![(batch.point_start, batch.point_count)]
            } else {
                batch
                    .point_runs
                    .iter()
                    .filter(|run| !run.source)
                    .map(|run| (run.start, run.count))
                    .collect::<Vec<_>>()
            };
            for (start, count) in runs {
                let Some(points) = self
                    .marks
                    .points
                    .get(start as usize..(start + count) as usize)
                else {
                    continue;
                };
                for point in points {
                    if !point_shown(point.time, self.time_window) {
                        continue;
                    }
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
            }
            for line in border {
                stroke_cpu(&mut frame, width, height, &matrix, &line, cell.plot);
            }
            paint_color_axis(self, &mut frame, width, height, cell, scale);
            let mut glyphs = if cell_plane(self, cell) == Some(3) {
                Vec::new()
            } else {
                labels_for(
                    &self.panels[cell.panel],
                    &camera,
                    cell,
                    self.foreground,
                    self.chrome <= 0.0,
                )
            };
            glyphs.extend(dose_chrome(self, cell));
            glyphs.extend(color_axis_glyphs(self, cell, scale));
            for glyph in glyphs {
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
        self.fit_slice_chrome(&layout);
        self.ensure_uploaded(gpu)?;
        self.flush_ramp(&gpu.queue);
        self.ensure_volume(gpu)?;
        self.sync_heatmaps(&gpu.queue);
        let (cloud_quad_span, cloud_line_span, grounds) = self.upload_live(gpu);
        self.ensure_uniforms(gpu, layout.len());
        let window = self.time_window;
        let line_draws = self.line_draw_ranges();
        let points = &self.marks.points;
        let panels = &self.marks.panels;
        let point_draws: Vec<Vec<(u32, u32)>> = panels
            .iter()
            .map(|batch| {
                visible_draws(
                    points,
                    batch.point_start,
                    batch.point_count,
                    &batch.point_runs,
                    window,
                )
            })
            .collect();
        let scales: Vec<f32> = layout
            .iter()
            .map(|cell| self.integrate_scale(cell.panel, cell.plot.w / cell.plot.h.max(1.0)))
            .collect();
        let frames = layout
            .iter()
            .map(|cell| panel_frame(self, cell))
            .collect::<Vec<_>>();
        let labels = layout
            .iter()
            .enumerate()
            .map(|(index, cell)| {
                let mut glyphs = if cell_plane(self, cell) == Some(3) {
                    Vec::new()
                } else {
                    labels_for(
                        &self.panels[cell.panel],
                        &self.view_camera(cell.panel, &cell.plot),
                        cell,
                        self.foreground,
                        self.chrome <= 0.0,
                    )
                };
                glyphs.extend(dose_chrome(self, cell));
                glyphs.extend(color_axis_glyphs(self, cell, scales[index]));
                glyphs
            })
            .collect::<Vec<_>>();
        let mut bar_quads = Vec::new();
        let bar_ranges: Vec<(u32, u32)> = layout
            .iter()
            .enumerate()
            .map(|(index, cell)| {
                let (ramp, ticks) = color_axis_quads(self, cell, scales[index]);
                let ramp_n = u32::from(ramp.is_some());
                let tick_n = ticks.len() as u32;
                if let Some(quad) = ramp {
                    bar_quads.push(quad);
                }
                bar_quads.extend(ticks);
                (ramp_n, tick_n)
            })
            .collect();
        let frame_bytes = frames
            .iter()
            .flat_map(|(grid, border)| encode_lines(grid).into_iter().chain(encode_lines(border)))
            .collect::<Vec<_>>();
        let text_bytes = labels
            .iter()
            .flat_map(|glyphs| encode_glyphs(glyphs))
            .collect::<Vec<_>>();
        let color_bytes = encode_quads(&bar_quads);
        write_grow(&gpu.device, &gpu.queue, &mut self.frame_buf, &frame_bytes);
        write_grow(&gpu.device, &gpu.queue, &mut self.text_buf, &text_bytes);
        write_grow(&gpu.device, &gpu.queue, &mut self.color_buf, &color_bytes);
        let Some(uniform) = self.uniform_buf.as_ref() else {
            return Err(GpuError::Message("uniform buffer missing".into()));
        };
        for (index, cell) in layout.iter().enumerate() {
            let matrix = self.view_camera(cell.panel, &cell.plot).clip_from_data(
                cell.plot,
                width as f32,
                height as f32,
            );
            let bytes = encode_uniform(
                matrix,
                width as f32,
                height as f32,
                self.dose_uniform(
                    cell.panel,
                    cell.plot.w / cell.plot.h.max(1.0),
                    scales[index],
                ),
            );
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
            let mut color_cursor = 0u64;
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
                let volume = self
                    .volume_draw
                    .as_ref()
                    .map(|draw| &draw.group)
                    .unwrap_or(&gpu.empty_volume);
                pass.set_bind_group(2, volume, &[]);
                let (quad_at, quad_count) = cloud_quad_span[cell.panel];
                let ground = grounds[cell.panel];
                if let (true, Some(buffer)) = (ground > 0, self.cloud_quads.as_ref()) {
                    pass.set_bind_group(1, &gpu.white_group, &[]);
                    pass.set_vertex_buffer(0, buffer.slice(quad_at * QUAD_STRIDE..));
                    pass.draw(0..6, 0..ground as u32);
                }
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
                let rest = quad_count.saturating_sub(ground);
                if let (true, Some(buffer)) = (rest > 0, self.cloud_quads.as_ref()) {
                    pass.set_bind_group(1, &gpu.white_group, &[]);
                    pass.set_vertex_buffer(0, buffer.slice((quad_at + ground) * QUAD_STRIDE..));
                    pass.draw(0..6, 0..rest as u32);
                }
                pass.set_pipeline(&gpu.line_pipeline);
                pass.set_bind_group(0, &self.uniform_groups[index], &[offset]);
                for (start, count) in &line_draws[cell.panel] {
                    pass.set_vertex_buffer(0, marks.lines.slice(u64::from(*start) * LINE_STRIDE..));
                    pass.draw(0..6, 0..*count);
                }
                let (line_at, line_count) = cloud_line_span[cell.panel];
                if let (true, Some(buffer)) = (line_count > 0, self.cloud_lines.as_ref()) {
                    pass.set_vertex_buffer(0, buffer.slice(line_at * LINE_STRIDE..));
                    pass.draw(0..6, 0..line_count as u32);
                }
                let draws = &point_draws[cell.panel];
                if !draws.is_empty() {
                    pass.set_pipeline(&gpu.point_pipeline);
                    pass.set_bind_group(0, &self.uniform_groups[index], &[offset]);
                    for (start, count) in draws {
                        pass.set_vertex_buffer(
                            0,
                            marks.points.slice(u64::from(*start) * POINT_STRIDE..),
                        );
                        pass.draw(0..6, 0..*count);
                    }
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
                let (ramp_n, tick_n) = bar_ranges[index];
                if let (true, Some(buffer)) = (ramp_n + tick_n > 0, self.color_buf.as_ref()) {
                    pass.set_pipeline(&gpu.quad_pipeline);
                    pass.set_bind_group(0, &self.uniform_groups[index], &[offset]);
                    pass.set_bind_group(2, volume, &[]);
                    if ramp_n > 0 {
                        if let Some(texture) = dose_heat_index(self, cell.panel) {
                            pass.set_bind_group(1, &marks.heats[texture].group, &[]);
                            pass.set_vertex_buffer(0, buffer.slice(color_cursor * QUAD_STRIDE..));
                            pass.draw(0..6, 0..1);
                        }
                        color_cursor += 1;
                    }
                    if tick_n > 0 {
                        pass.set_bind_group(1, &gpu.white_group, &[]);
                        pass.set_vertex_buffer(0, buffer.slice(color_cursor * QUAD_STRIDE..));
                        pass.draw(0..6, 0..tick_n);
                        color_cursor += u64::from(tick_n);
                    }
                }
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

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn line_marks(&self) -> &[LineRec] {
        &self.marks.lines
    }

    pub(crate) fn carry_lines(&mut self, token: u64, reuse: bool, prefix: u32) {
        self.line_token = token;
        self.reuse_lines = reuse;
        self.line_prefix = prefix;
    }

    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
    pub(crate) fn needs_cached_lines(&self) -> bool {
        self.reuse_lines
    }

    /// Keep trace lines from `previous` and append this payload's tail.
    ///
    /// Returns an error before it changes `previous` when the cache does not match.
    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
    pub(crate) fn adopt_lines(
        &mut self,
        previous: &mut Plot,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<(), String> {
        if !self.reuse_lines {
            return Ok(());
        }
        let prefix = self.line_prefix as usize;
        if self.line_token == 0
            || self.line_token != previous.line_token
            || previous.marks.lines.len() < prefix
        {
            return Err("plot lines are not in the previous picture".into());
        }
        let suffix = self
            .encoded
            .as_mut()
            .map(|encoded| std::mem::take(&mut encoded.lines))
            .unwrap_or_default();
        let offset = u64::from(self.line_prefix) * LINE_STRIDE;
        let gpu_holds_prefix = previous
            .gpu
            .as_ref()
            .is_some_and(|gpu| gpu.lines.size() >= offset);
        if gpu_holds_prefix {
            self.kept_line_buf = Some(reuse_line_buffer(previous, device, queue, offset, &suffix));
            self.splice_cached_lines(previous);
            return Ok(());
        }
        if let Some(bytes) = stitched_line_bytes(previous, offset as usize, &suffix) {
            if let Some(encoded) = self.encoded.as_mut() {
                encoded.lines = bytes;
            }
            self.splice_cached_lines(previous);
            return Ok(());
        }
        Err("plot lines are not in the previous picture".into())
    }

    #[cfg_attr(not(any(test, target_arch = "wasm32")), allow(dead_code))]
    pub(crate) fn splice_cached_lines(&mut self, previous: &mut Plot) {
        let prefix = self.line_prefix as usize;
        let suffix = std::mem::take(&mut self.marks.lines);
        let mut lines = std::mem::take(&mut previous.marks.lines);
        lines.truncate(prefix);
        lines.extend(suffix);
        self.marks.lines = lines;
    }

    fn flush_ramp(&mut self, queue: &wgpu::Queue) {
        if !self.ramp_dirty {
            return;
        }
        self.ramp_dirty = false;
        let Some(index) = self.dose_atlas else {
            return;
        };
        let Some((cols, rows)) = self.marks.heatmap_size.get(index).copied() else {
            return;
        };
        if rows == 0 {
            return;
        }
        let row_bytes = (cols * 4) as usize;
        let start = (rows as usize - 1) * row_bytes;
        let Some(row) = self
            .marks
            .heatmaps
            .get(index)
            .and_then(|pixels| pixels.get(start..start + row_bytes))
            .map(<[u8]>::to_vec)
        else {
            return;
        };
        let Some(texture) = self
            .gpu
            .as_ref()
            .and_then(|gpu| gpu.heats.get(index))
            .map(|heat| &heat.texture)
        else {
            return;
        };
        write_rgba_row(queue, texture, &row, cols, rows - 1);
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
        let kept_lines = self.kept_line_buf.take();
        let (lines, points, quads) = if let Some(encoded) = self.encoded.take() {
            (encoded.lines, encoded.points, encoded.quads)
        } else {
            (
                encode_lines(&self.marks.lines),
                encode_points(&self.marks.points),
                encode_quads(&self.marks.quads),
            )
        };
        let lines = if let Some(buffer) = kept_lines {
            buffer
        } else {
            upload_buffer(&gpu.device, &gpu.queue, "lines", &lines)
        };
        self.gpu = Some(GpuMarks {
            lines,
            points: upload_buffer(&gpu.device, &gpu.queue, "points", &points),
            quads: upload_buffer(&gpu.device, &gpu.queue, "quads", &quads),
            heats,
        });
        self.mark_uploads += 1;
        self.heat_uploaded = self.heat_revision;
        Ok(())
    }

    fn ensure_volume(&mut self, gpu: &PlotGpu) -> Result<(), GpuError> {
        if self.volume_draw.is_some() || self.dose.is_none() {
            return Ok(());
        }
        let limit = gpu.device.limits().max_texture_dimension_3d.max(1);
        let grid = self.dose.as_ref().unwrap();
        let max_axis = grid.volume.shape.iter().copied().max().unwrap_or(1) as u32;
        let (dose_tex, brick_tex, origin, voxel, line_scale, shape) = if max_axis <= limit {
            let dose_tex = upload_dose_f16(
                &gpu.device,
                &gpu.queue,
                &grid.volume.values,
                grid.volume.shape,
            )?;
            let brick_tex = upload_r32_3d(&gpu.device, &gpu.queue, &grid.bricks, grid.brick_shape)?;
            (
                dose_tex,
                brick_tex,
                grid.volume.origin,
                grid.volume.voxel,
                grid.line_scale,
                [
                    grid.volume.shape[0] as f32,
                    grid.volume.shape[1] as f32,
                    grid.volume.shape[2] as f32,
                ],
            )
        } else {
            let pooled = pool_volume(&grid.volume, limit);
            let scan = scan_kit_core::scan_volume(&pooled);
            let dose_tex = upload_dose_f16(&gpu.device, &gpu.queue, &pooled.values, pooled.shape)?;
            let brick_tex = upload_r32_3d(&gpu.device, &gpu.queue, &scan.bricks, scan.brick_shape)?;
            (
                dose_tex,
                brick_tex,
                pooled.origin,
                pooled.voxel,
                scan.line_scale,
                [
                    pooled.shape[0] as f32,
                    pooled.shape[1] as f32,
                    pooled.shape[2] as f32,
                ],
            )
        };
        let (planes, plane_w, plane_h) = integral_planes(grid);
        let integral_tex = upload_r32_2d(&gpu.device, &gpu.queue, &planes, plane_w, plane_h)?;
        let group = volume_group(
            &gpu.device,
            &gpu.volume_layout,
            &gpu.sampler,
            &dose_tex,
            &brick_tex,
            &integral_tex,
        );
        self.volume_draw = Some(VolumeDraw {
            group,
            origin,
            voxel,
            line_scale,
            shape,
        });
        Ok(())
    }

    fn sync_heatmaps(&mut self, queue: &wgpu::Queue) {
        if self.heat_revision == self.heat_uploaded {
            return;
        }
        let indices: Vec<usize> = self
            .marks
            .panels
            .iter()
            .flat_map(|batch| &batch.clouds)
            .filter_map(|cloud| cloud.heatmap.map(|index| index as usize))
            .collect();
        let jobs: Vec<(usize, Vec<u8>, u32, u32)> = indices
            .into_iter()
            .filter_map(|index| {
                let pixels = self.marks.heatmaps.get(index)?.clone();
                let &(cols, rows) = self.marks.heatmap_size.get(index)?;
                Some((index, pixels, cols, rows))
            })
            .collect();
        let Some(gpu) = self.gpu.as_ref() else {
            return;
        };
        for (index, pixels, cols, rows) in jobs {
            let Some(heat) = gpu.heats.get(index) else {
                continue;
            };
            write_rgba(queue, &heat.texture, &pixels, cols, rows);
        }
        self.heat_uploaded = self.heat_revision;
    }

    fn upload_live(&mut self, gpu: &PlotGpu) -> (Vec<(u64, u64)>, Vec<(u64, u64)>, Vec<u64>) {
        let panels = self.marks.panels.len();
        let mut quad_span = vec![(0u64, 0u64); panels];
        let mut line_span = vec![(0u64, 0u64); panels];
        let mut grounds = vec![0u64; panels];
        let mut quad_bytes = Vec::new();
        let mut line_bytes = Vec::new();
        let mut quads = 0u64;
        let mut lines = 0u64;
        for panel in 0..panels {
            let quad_at = quads;
            let line_at = lines;
            for cloud in self
                .live
                .iter()
                .filter(|cloud| cloud.panel == panel && cloud.crosshair)
            {
                quad_bytes.extend(encode_quads(&cloud.quads));
                quads += cloud.quads.len() as u64;
            }
            grounds[panel] = quads - quad_at;
            for cloud in self
                .live
                .iter()
                .filter(|cloud| cloud.panel == panel && !cloud.crosshair)
            {
                quad_bytes.extend(encode_quads(&cloud.quads));
                quads += cloud.quads.len() as u64;
            }
            for cloud in self.live.iter().filter(|cloud| cloud.panel == panel) {
                line_bytes.extend(encode_lines(&cloud.lines));
                lines += cloud.lines.len() as u64;
            }
            quad_span[panel] = (quad_at, quads - quad_at);
            line_span[panel] = (line_at, lines - line_at);
        }
        write_grow(&gpu.device, &gpu.queue, &mut self.cloud_quads, &quad_bytes);
        write_grow(&gpu.device, &gpu.queue, &mut self.cloud_lines, &line_bytes);
        (quad_span, line_span, grounds)
    }

    #[cfg(test)]
    fn project_data(&self, panel: usize, x: f32, y: f32, width: u32, height: u32) -> [f32; 2] {
        let layout = self.layout(width, height);
        let cell = &layout[panel];
        let matrix = self.view_camera(panel, &cell.plot).clip_from_data(
            cell.plot,
            width as f32,
            height as f32,
        );
        project(matrix, [x, y, 0.0], width as f32, height as f32)
    }

    /// Slice windows grow so a millimetre matches on both axes. Other panels keep their camera.
    fn view_camera(&self, panel: usize, plot: &PlotRect) -> Camera {
        let camera = self.cameras[panel];
        let spatial = self
            .panels
            .get(panel)
            .and_then(|item| dose_plane(&item.title))
            .is_some_and(|plane| plane < 3);
        if !spatial {
            return camera;
        }
        let turns = self.turns.get(panel).copied().unwrap_or(0);
        fit_domain(camera, plot, turns)
    }

    /// Paint the ramp's zero across each slice window and run that crosshair to
    /// the window. The volume quad stays on the lattice.
    fn fit_slice_chrome(&mut self, layout: &[Cell]) {
        if self.dose.is_none() {
            return;
        }
        let jobs: Vec<(usize, Camera, [f32; 4])> = layout
            .iter()
            .filter_map(|cell| {
                let plane = cell_plane(self, cell).filter(|plane| *plane < 3)?;
                let camera = self.view_camera(cell.panel, &cell.plot);
                let integral = self.integral.get(cell.panel).copied().unwrap_or(false);
                let color = slice_zero(self.dose.as_ref()?, integral, plane);
                Some((cell.panel, camera, color))
            })
            .collect();
        for (panel, camera, color) in jobs {
            for cloud in self
                .live
                .iter_mut()
                .filter(|cloud| cloud.panel == panel && cloud.crosshair)
            {
                for line in &mut cloud.lines {
                    extend_crosshair(line, camera);
                }
                cloud.quads.clear();
                cloud.quads.push(QuadRec {
                    a: [camera.xmin, camera.ymin],
                    b: [camera.xmax, camera.ymax],
                    c: None,
                    heat: 0.0,
                    color,
                    id: 0,
                    heatmap: None,
                });
            }
        }
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
    volume_layout: wgpu::BindGroupLayout,
    line_pipeline: wgpu::RenderPipeline,
    point_pipeline: wgpu::RenderPipeline,
    quad_pipeline: wgpu::RenderPipeline,
    text_pipeline: wgpu::RenderPipeline,
    sampler: wgpu::Sampler,
    white_group: wgpu::BindGroup,
    atlas_group: wgpu::BindGroup,
    empty_volume: wgpu::BindGroup,
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
                    min_binding_size: Some(NonZeroU64::new(UNIFORM_BYTES as u64).unwrap()),
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
        let volume_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dose-volume"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let quad_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("quads"),
            bind_group_layouts: &[&uniform_layout, &textured_layout, &volume_layout],
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
        let quad_pipeline = make(&quad_layout, "vs_quad", "fs_quad", quad_vertex());
        let text_pipeline = make(&shaded_layout, "vs_text", "fs_text", text_vertex());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("plot"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let font = atlas();
        let atlas_tex = upload_r8(&device, &queue, &font.pixels, font.size, font.size)?;
        let white = upload_r8(&device, &queue, &[255], 1, 1)?;
        let atlas_group = textured_group(&device, &textured_layout, &atlas_tex, &sampler);
        let white_group = textured_group(&device, &textured_layout, &white, &sampler);
        let empty_dose = upload_dose_f16(&device, &queue, &[0.0], [1, 1, 1])?;
        let empty_brick = upload_r32_3d(&device, &queue, &[0.0], [1, 1, 1])?;
        let empty_integral = upload_r32_2d(&device, &queue, &[0.0], 1, 1)?;
        let empty_volume = volume_group(
            &device,
            &volume_layout,
            &sampler,
            &empty_dose,
            &empty_brick,
            &empty_integral,
        );
        Ok(Self {
            device,
            queue,
            format,
            uniform_layout,
            textured_layout,
            volume_layout,
            line_pipeline,
            point_pipeline,
            quad_pipeline,
            text_pipeline,
            sampler,
            white_group,
            atlas_group,
            empty_volume,
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
    build_marks_inner(panels, false)
}

/// Guides and other marks, with polyline samples counted but not stored.
///
/// The caller already has those line records. Batch ranges still address the
/// full buffer, prefix first.
pub(crate) fn build_marks_without_polylines(panels: &[Panel]) -> Marks {
    build_marks_inner(panels, true)
}

fn build_marks_inner(panels: &[Panel], skip_polylines: bool) -> Marks {
    let mut lines = Vec::new();
    let mut points = Vec::new();
    let mut quads = Vec::new();
    let mut heatmaps = Vec::new();
    let mut heatmap_size = Vec::new();
    let mut batches = Vec::new();
    let mut next_id = 1u32;
    let mut cursor = 0u32;
    for panel in panels {
        let line_start = cursor;
        let point_start = points.len() as u32;
        let quad_start = quads.len() as u32;
        let mut point_runs = Vec::new();
        let mut line_runs = Vec::new();
        let mut pending = Vec::new();
        let contour_live = hides_polylines(panel);
        let mut fill_color = [0.8, 0.8, 0.8, 0.13];
        let mut line_color = [0.8, 0.8, 0.8, 0.0];
        let mut line_thickness = 1.0f32;
        let mut last_heat: Option<usize> = None;
        let mut last_ramp = 0u8;
        let mut last_color = [1.0, 1.0, 1.0, 1.0];
        let mut last_lo = 0.0f32;
        let mut last_hi = 0.0f32;
        let mut last_cols = 0u32;
        let mut last_rows = 0u32;
        let mut clouds = Vec::new();
        for series in &panel.series {
            let id = next_id;
            next_id = next_id.saturating_add(1);
            match series {
                Series::Polyline {
                    xs,
                    ys,
                    color,
                    thickness,
                } => {
                    if contour_live {
                        line_color = *color;
                        line_thickness = *thickness;
                        continue;
                    }
                    let start = cursor;
                    let added = if skip_polylines {
                        polyline_segments(xs, ys)
                    } else {
                        let before = lines.len();
                        push_polyline(&mut lines, xs, ys, *color, *thickness, id);
                        (lines.len() - before) as u32
                    };
                    cursor = cursor.saturating_add(added);
                    if added > 0 {
                        line_runs.push(LineRun {
                            start,
                            count: added,
                            monotonic: true,
                        });
                    }
                }
                Series::Guide {
                    xs,
                    ys,
                    color,
                    thickness,
                } => {
                    let start = cursor;
                    let before = lines.len();
                    push_polyline(&mut lines, xs, ys, *color, *thickness, id);
                    let added = (lines.len() - before) as u32;
                    cursor = cursor.saturating_add(added);
                    if added > 0 {
                        line_runs.push(LineRun {
                            start,
                            count: added,
                            monotonic: false,
                        });
                    }
                }
                Series::Points {
                    xs,
                    ys,
                    color,
                    radius,
                    times,
                } => {
                    let before = points.len() as u32;
                    push_points(&mut points, xs, ys, times, *color, *radius, id, panel);
                    let count = points.len() as u32 - before;
                    if count > 0 {
                        point_runs.push(PointRun {
                            start: before,
                            count,
                            timed: !times.is_empty(),
                            source: false,
                        });
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
                    if contour_live {
                        fill_color = *color;
                        continue;
                    }
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
                    last_heat = Some(heatmaps.len() - 1);
                    last_ramp = *ramp;
                    last_color = *color;
                    last_lo = *lo;
                    last_hi = *hi;
                    last_cols = *cols;
                    last_rows = *rows;
                }
                Series::Cloud {
                    xs,
                    ys,
                    times,
                    x0,
                    x1,
                    y0,
                    y1,
                    style,
                    cutoff,
                } => {
                    let before = points.len() as u32;
                    push_points(&mut points, xs, ys, times, [0.0; 4], 0.0, id, panel);
                    let count = points.len() as u32 - before;
                    if count == 0 {
                        continue;
                    }
                    point_runs.push(PointRun {
                        start: before,
                        count,
                        timed: true,
                        source: true,
                    });
                    let heatmap = (*style == CloudStyle::Density)
                        .then_some(last_heat)
                        .flatten();
                    clouds.push(CloudBatch {
                        start: before,
                        count,
                        x0: *x0,
                        x1: *x1,
                        y0: *y0,
                        y1: *y1,
                        cols: last_cols,
                        rows: last_rows,
                        style: *style,
                        cutoff: *cutoff,
                        heatmap: heatmap.map(|index| index as u32),
                        ramp: last_ramp,
                        color: last_color,
                        lo: last_lo,
                        hi: last_hi,
                        fill: fill_color,
                        line: line_color,
                        thickness: line_thickness,
                        id,
                    });
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
            line_count: cursor - line_start,
            point_start,
            point_count: points.len() as u32 - point_start,
            quad_start,
            quad_count,
            heats,
            point_runs,
            line_runs,
            clouds,
        });
    }
    if !skip_polylines {
        debug_assert_eq!(cursor, lines.len() as u32);
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

#[cfg_attr(not(any(test, target_arch = "wasm32")), allow(dead_code))]
/// Raw x extent and the padded y extent of one panel's strokes inside `[lo, hi]`.
///
/// Each monotonic stroke is sorted by x, so the window is a binary search plus
/// the samples that fall inside it. `runs` addresses `lines`. An empty `runs`
/// scans `lines` by series id, which is the panel slice from an older payload.
fn follow_span(
    lines: &[LineRec],
    runs: &[LineRun],
    lo: f32,
    hi: f32,
) -> Option<(f32, f32, f32, f32)> {
    let mut x_lo = f32::MAX;
    let mut x_hi = f32::MIN;
    let mut y_lo = f32::MAX;
    let mut y_hi = f32::MIN;
    let mut absorb = |series: &[LineRec]| {
        let (first, stop) = window_range(series, lo, hi);
        if first >= stop {
            return;
        }
        let mut ys = Vec::new();
        let mut previous: Option<[f32; 3]> = None;
        for line in &series[first..stop] {
            if previous != Some(line.a) {
                take_vertex(&mut ys, &mut x_lo, &mut x_hi, line.a, lo, hi);
            }
            take_vertex(&mut ys, &mut x_lo, &mut x_hi, line.b, lo, hi);
            previous = Some(line.b);
        }
        if let Some((low, high)) = robust_limits(&ys) {
            y_lo = y_lo.min(low);
            y_hi = y_hi.max(high);
        }
    };
    if runs.is_empty() {
        let mut index = 0;
        while index < lines.len() {
            if lines[index].id == 0 {
                index += 1;
                continue;
            }
            let id = lines[index].id;
            let start = index;
            let mut end = index;
            while end + 1 < lines.len() && lines[end + 1].id == id {
                end += 1;
            }
            absorb(&lines[start..=end]);
            index = end + 1;
        }
    } else {
        for run in runs {
            if !run.monotonic || run.count == 0 {
                continue;
            }
            let Some(series) = lines.get(run.start as usize..(run.start + run.count) as usize)
            else {
                continue;
            };
            absorb(series);
        }
    }
    (x_lo <= x_hi && y_lo <= y_hi).then_some((x_lo, x_hi, y_lo, y_hi))
}

/// Half-open segment range of a monotonic stroke whose x overlaps `[lo, hi]`.
fn window_range(series: &[LineRec], lo: f32, hi: f32) -> (usize, usize) {
    let first = series.partition_point(|line| line.b[0] < lo);
    let stop = series.partition_point(|line| line.a[0] <= hi);
    (first, stop)
}

#[cfg_attr(not(any(test, target_arch = "wasm32")), allow(dead_code))]
fn take_vertex(
    ys: &mut Vec<f32>,
    x_lo: &mut f32,
    x_hi: &mut f32,
    point: [f32; 3],
    lo: f32,
    hi: f32,
) {
    let x = point[0];
    let y = point[1];
    if !x.is_finite() || !y.is_finite() || x < lo || x > hi {
        return;
    }
    *x_lo = x_lo.min(x);
    *x_hi = x_hi.max(x);
    ys.push(y);
}

pub(crate) struct LineStamp {
    pub token: u64,
    pub prefix: u32,
    pub reusable: bool,
}

/// Identity of the polyline samples. Guides are not part of it, so a scatter
/// crosshair can change without rebuilding the traces.
pub(crate) fn line_stamp(panels: &[Panel]) -> LineStamp {
    let mut state = 0x9E37_79B1_85EB_CA87u64;
    let mut prefix = 0u32;
    for panel in panels {
        if hides_polylines(panel) {
            continue;
        }
        for series in &panel.series {
            let Series::Polyline {
                xs,
                ys,
                color,
                thickness,
            } = series
            else {
                continue;
            };
            state = mix(state, xs.len() as u64);
            state = mix(state, ys.len() as u64);
            for value in xs.iter().chain(ys) {
                state = mix(state, u64::from(value.to_bits()));
            }
            for channel in color {
                state = mix(state, u64::from(channel.to_bits()));
            }
            state = mix(state, u64::from(thickness.to_bits()));
            prefix = prefix.saturating_add(polyline_segments(xs, ys));
        }
    }
    LineStamp {
        token: state.max(1),
        prefix,
        reusable: prefix > 0 && polylines_are_prefix(panels),
    }
}

fn mix(state: u64, value: u64) -> u64 {
    state
        .wrapping_add(value)
        .wrapping_mul(0x517c_c1b7_2722_0a95)
}

fn polylines_are_prefix(panels: &[Panel]) -> bool {
    let mut seen_guide = false;
    for panel in panels {
        for series in &panel.series {
            match series {
                Series::Guide { .. } => seen_guide = true,
                Series::Polyline { .. } if seen_guide && !hides_polylines(panel) => return false,
                _ => {}
            }
        }
    }
    true
}

fn polyline_segments(xs: &[f32], ys: &[f32]) -> u32 {
    let mut prev = false;
    let mut count = 0u32;
    for (x, y) in xs.iter().zip(ys) {
        let finite = x.is_finite() && y.is_finite();
        if finite && prev {
            count = count.saturating_add(1);
        }
        prev = finite;
    }
    count
}

fn hides_polylines(panel: &Panel) -> bool {
    panel.series.iter().any(|series| {
        matches!(
            series,
            Series::Cloud {
                style: CloudStyle::Contour,
                ..
            }
        )
    })
}

fn window_points(points: &[PointRec], window: Option<(f32, f32)>) -> &[PointRec] {
    let Some((lo, hi)) = window else {
        return points;
    };
    let begin = points.partition_point(|point| point.time < lo);
    let end = begin + points[begin..].partition_point(|point| point.time <= hi);
    &points[begin..end]
}

fn samples_of(points: &[PointRec]) -> (Vec<f32>, Vec<f32>) {
    let mut xs = Vec::with_capacity(points.len());
    let mut ys = Vec::with_capacity(points.len());
    for point in points {
        xs.push(point.p[0]);
        ys.push(point.p[1]);
    }
    (xs, ys)
}

fn contour_geometry(
    series: &[Series],
    id: u32,
    fill: [f32; 4],
    line: [f32; 4],
    thickness: f32,
) -> (Vec<QuadRec>, Vec<LineRec>) {
    let mut quads = Vec::new();
    let mut lines = Vec::new();
    for series in series {
        match series {
            Series::Triangles { xs, ys, .. } => {
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
                            color: fill,
                            id,
                            heatmap: None,
                        });
                    }
                }
            }
            Series::Polyline { xs, ys, .. } => {
                push_polyline(&mut lines, xs, ys, line, thickness, id);
            }
            _ => {}
        }
    }
    (quads, lines)
}

fn point_shown(time: f32, window: Option<(f32, f32)>) -> bool {
    let Some((lo, hi)) = window else {
        return true;
    };
    !time.is_finite() || (time >= lo && time <= hi)
}

/// Draw ranges for one panel. A timed run is sorted, so the playhead is one slice.
///
/// ponytail: a "before" window draws every sample up to the playhead. A
/// multi-minute dwell is one instanced draw of that prefix. If fill rate shows
/// up, decimate the visible slice into the 4096 grid and upload that.
fn visible_draws(
    points: &[PointRec],
    start: u32,
    count: u32,
    runs: &[PointRun],
    window: Option<(f32, f32)>,
) -> Vec<(u32, u32)> {
    let mut draws = Vec::new();
    if runs.is_empty() {
        if count > 0 {
            draws.push((start, count));
        }
        return draws;
    }
    for run in runs {
        if run.source {
            continue;
        }
        let (offset, shown) = if run.timed {
            let Some(range) = points.get(run.start as usize..(run.start + run.count) as usize)
            else {
                continue;
            };
            if let Some((lo, hi)) = window {
                let begin = range.partition_point(|point| point.time < lo);
                let end = begin + range[begin..].partition_point(|point| point.time <= hi);
                (begin, (end - begin) as u32)
            } else {
                (0usize, run.count)
            }
        } else {
            (0usize, run.count)
        };
        if shown > 0 {
            draws.push((run.start + offset as u32, shown));
        }
    }
    draws
}

/// Draw ranges for one panel's lines. A monotonic time trace is one slice.
///
/// ponytail: a "before" window draws every segment up to the playhead. A
/// multi-minute dwell is one instanced draw of that prefix per stroke. If fill
/// rate shows up, bin the slice to one extrema pair per pixel column and upload
/// that. The 1 s window is about a thousand segments and does not need it.
fn visible_line_draws(
    lines: &[LineRec],
    start: u32,
    count: u32,
    runs: &[LineRun],
    window: Option<(f32, f32)>,
) -> Vec<(u32, u32)> {
    let Some((lo, hi)) = window else {
        return full_line_draw(start, count);
    };
    if runs.is_empty() {
        return full_line_draw(start, count);
    }
    let mut draws = Vec::new();
    for run in runs {
        if run.count == 0 {
            continue;
        }
        if !run.monotonic {
            draws.push((run.start, run.count));
            continue;
        }
        let Some(series) = lines.get(run.start as usize..(run.start + run.count) as usize) else {
            continue;
        };
        let (first, stop) = window_range(series, lo, hi);
        if first < stop {
            draws.push((run.start + first as u32, (stop - first) as u32));
        }
    }
    draws
}

fn full_line_draw(start: u32, count: u32) -> Vec<(u32, u32)> {
    if count == 0 {
        Vec::new()
    } else {
        vec![(start, count)]
    }
}

/// One sample per cell. Overlapping dots in a scatter upload and shade once.
///
/// ponytail: the grid is fixed when the payload is built, 4096 cells on each
/// axis of the panel range. A zoom that would split a cell still draws the
/// kept sample. Upgrade: retain the series and rebuild marks on camera change.
fn push_points(
    points: &mut Vec<PointRec>,
    xs: &[f32],
    ys: &[f32],
    times: &[f32],
    color: [f32; 4],
    radius: f32,
    id: u32,
    panel: &Panel,
) {
    if !times.is_empty() {
        let start = points.len();
        for ((x, y), time) in xs.iter().zip(ys).zip(times) {
            if !x.is_finite() || !y.is_finite() || !time.is_finite() {
                continue;
            }
            points.push(PointRec {
                p: [*x, *y, 0.0],
                radius,
                color,
                id,
                time: *time,
            });
        }
        points[start..].sort_by(|left, right| left.time.total_cmp(&right.time));
        return;
    }
    let x0 = panel.xmin.min(panel.xmax);
    let y0 = panel.ymin.min(panel.ymax);
    let x_span = (panel.xmax - panel.xmin).abs().max(1.0e-12);
    let y_span = (panel.ymax - panel.ymin).abs().max(1.0e-12);
    let mut occupied = HashSet::with_capacity(xs.len().min(POINT_CELLS as usize));
    for (x, y) in xs.iter().zip(ys) {
        if !x.is_finite() || !y.is_finite() {
            continue;
        }
        let cell =
            (u64::from(point_cell(*x, x0, x_span)) << 32) | u64::from(point_cell(*y, y0, y_span));
        if !occupied.insert(cell) {
            continue;
        }
        points.push(PointRec {
            p: [*x, *y, 0.0],
            radius,
            color,
            id,
            time: f32::NAN,
        });
    }
}

fn point_cell(value: f32, origin: f32, span: f32) -> u32 {
    let t = ((value - origin) / span).clamp(0.0, 0.999_984);
    ((t * POINT_CELLS as f32) as u32).min(POINT_CELLS - 1)
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
/// Grow the window so one data unit is the same length on both axes.
///
/// The plot rectangle is unchanged. A quarter turn swaps which millimetre
/// runs along the pixel width, so the window grows to match that picture.
fn fit_domain(camera: Camera, plot: &PlotRect, turns: u8) -> Camera {
    let dw = (camera.xmax - camera.xmin).abs().max(1e-6);
    let dh = (camera.ymax - camera.ymin).abs().max(1e-6);
    let pixel = plot.w.max(1.0) / plot.h.max(1.0);
    let data = dw / dh;
    let ratio = if turns.is_multiple_of(2) {
        pixel
    } else {
        pixel * data * data
    };
    let (cw, ch) = if data < ratio {
        (ratio * dh, dh)
    } else {
        (dw, dw / ratio.max(1e-6))
    };
    let cx = (camera.xmin + camera.xmax) * 0.5;
    let cy = (camera.ymin + camera.ymax) * 0.5;
    Camera {
        xmin: cx - cw * 0.5,
        xmax: cx + cw * 0.5,
        ymin: cy - ch * 0.5,
        ymax: cy + ch * 0.5,
    }
}

/// Color of a dose of zero on this slice, the same window the volume shader uses.
fn slice_zero(grid: &crate::dose::DoseGrid, integral: bool, plane: u8) -> [f32; 4] {
    let (lo, hi) = if integral {
        crate::dose::integral_limits(grid, plane)
    } else if grid.hi > grid.lo {
        (grid.lo, grid.hi)
    } else {
        (0.0, grid.peak)
    };
    let span = (hi - lo).abs().max(1e-6);
    let t = ((0.0 * grid.gain - lo) / span).clamp(0.0, 1.0);
    let rgb = scan_kit_core::sample(grid.ramp, t);
    [rgb[0], rgb[1], rgb[2], grid.opacity.clamp(0.0, 1.0)]
}

/// Stretch one crosshair segment to the fitted window. The shared coordinate is the cursor.
fn extend_crosshair(line: &mut LineRec, camera: Camera) {
    let dx = (line.b[0] - line.a[0]).abs();
    let dy = (line.b[1] - line.a[1]).abs();
    if dx >= dy {
        let y = (line.a[1] + line.b[1]) * 0.5;
        line.a = [camera.xmin, y, line.a[2]];
        line.b = [camera.xmax, y, line.b[2]];
    } else {
        let x = (line.a[0] + line.b[0]) * 0.5;
        line.a = [x, camera.ymin, line.a[2]];
        line.b = [x, camera.ymax, line.b[2]];
    }
}

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

const COLOR_BAR_W: f32 = 14.0;
const COLOR_RAMP_HEAT: f32 = 4.0;
const COLOR_TICK_HEAT: f32 = -2.0;
const SUPERSCRIPT: [char; 10] = ['⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹'];

fn color_column_width() -> f32 {
    let font = atlas();
    let label = text::text_width("−0.000");
    8.0 + COLOR_BAR_W + 14.0 + label + 8.0 + font.line_height + 6.0
}

fn color_bar_rect(cell: &Cell) -> PlotRect {
    PlotRect {
        x: cell.plot.x + cell.plot.w + 8.0,
        y: cell.plot.y,
        w: COLOR_BAR_W,
        h: cell.plot.h.max(8.0),
    }
}

fn cell_plane(plot: &Plot, cell: &Cell) -> Option<u8> {
    plot.panels
        .get(cell.panel)
        .and_then(|panel| dose_plane(&panel.title))
}

fn panel_frame(plot: &Plot, cell: &Cell) -> (Vec<LineRec>, Vec<LineRec>) {
    let camera = plot.view_camera(cell.panel, &cell.plot);
    match cell_plane(plot, cell) {
        Some(3) => (Vec::new(), Vec::new()),
        Some(_) => (
            Vec::new(),
            border_lines(&camera, &cell.plot, plot.foreground),
        ),
        None => (
            grid_lines(&camera, plot.foreground),
            border_lines(&camera, &cell.plot, plot.foreground),
        ),
    }
}

fn dose_heat_index(plot: &Plot, panel: usize) -> Option<usize> {
    let batch = plot.marks.panels.get(panel)?;
    batch.heats.iter().rev().find_map(|(start, texture)| {
        let quad = plot.marks.quads.get(*start as usize)?;
        (quad.heat > 1.5 && quad.heat < 2.5).then_some(*texture)
    })
}

fn volume_limits(grid: &crate::dose::DoseGrid, line_scale: f32) -> (f32, f32) {
    let gain = grid.gain.max(0.01);
    let typical = grid.lo.abs().max(grid.hi.abs()).max(1e-6);
    if grid.mode == 0 {
        let line = if line_scale > 0.0 {
            line_scale
        } else {
            grid.line_scale.max(1e-6)
        };
        if grid.lo < 0.0 {
            let reach = line / gain;
            return (-reach, reach);
        }
        let peak = typical.max(1e-6);
        let flo = (grid.lo / peak).clamp(0.0, 1.0);
        let fhi = (grid.hi / peak).max(flo + 1e-4);
        let scale = line / gain;
        return (flo * scale, fhi * scale);
    }
    if grid.lo < 0.0 {
        let reach = typical / gain;
        return (-reach, reach);
    }
    (grid.lo / gain, grid.hi.max(grid.lo + 1e-6) / gain)
}

fn presented_limits(
    plot: &Plot,
    panel: usize,
    plane: u8,
    line_scale: f32,
) -> Option<(f32, f32, String)> {
    let grid = plot.dose.as_ref()?;
    let gain = grid.gain.max(0.01);
    let (lo, hi) = if plane >= 3 {
        volume_limits(grid, line_scale)
    } else if plot.integral.get(panel).copied().unwrap_or(false) {
        let (lo, hi) = crate::dose::integral_limits(grid, plane);
        (lo / gain, hi / gain)
    } else if grid.hi > grid.lo {
        (grid.lo / gain, grid.hi / gain)
    } else {
        (0.0, grid.peak / gain)
    };
    Some((lo, hi, grid.unit.clone()))
}

fn superscript_exp(exp: i32) -> String {
    exp.to_string()
        .chars()
        .map(|ch| match ch {
            '-' => '⁻',
            '0'..='9' => SUPERSCRIPT[(ch as u8 - b'0') as usize],
            _ => ch,
        })
        .collect()
}

fn short_tick(value: f32) -> String {
    if !value.is_finite() || value.abs() < 1e-8 {
        return "0".into();
    }
    let exp = value.abs().log10().floor();
    let scale = 10.0_f32.powf(exp - 3.0);
    let rounded = (value / scale).round() * scale;
    if rounded.abs() < 1e-8 {
        return "0".into();
    }
    let decimals = (3 - rounded.abs().log10().floor() as i32).max(0) as usize;
    let text = format!("{rounded:.decimals$}");
    let trimmed = if text.contains('.') {
        text.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        text
    };
    trimmed.replace('-', "−")
}

fn tick_labels(values: &[f32]) -> (Vec<String>, String) {
    if values.is_empty() {
        return (Vec::new(), String::new());
    }
    let peak = values
        .iter()
        .fold(0.0_f32, |peak, value| peak.max(value.abs()));
    let mut scale = 1.0_f32;
    let mut offset = String::new();
    if peak > 0.0 && peak.is_finite() {
        let exp = peak.log10().floor() as i32;
        if !(-2..=3).contains(&exp) {
            scale = 10.0_f32.powi(exp);
            offset = format!("×10{}", superscript_exp(exp));
        }
    }
    let labels = values
        .iter()
        .map(|value| short_tick(*value / scale))
        .collect();
    (labels, offset)
}

fn minor_parts(step: f32) -> i32 {
    if step <= 0.0 || !step.is_finite() {
        return 5;
    }
    let leading = step / 10.0_f32.powf(step.log10().floor());
    if (leading - 2.0).abs() < 0.05 || (leading - 2.5).abs() < 0.05 {
        4
    } else {
        5
    }
}

fn locator_ticks(lo: f32, hi: f32, nbins: i32) -> Vec<f32> {
    let span = (hi - lo).max(1e-12);
    let raw = span / nbins.max(2) as f32;
    if !raw.is_finite() || raw <= 0.0 {
        return vec![lo, hi];
    }
    let mag = 10.0_f32.powf(raw.log10().floor());
    let residual = raw / mag;
    let nice = [1.0_f32, 2.0, 2.5, 5.0, 10.0]
        .into_iter()
        .find(|step| residual <= *step + 1e-3)
        .unwrap_or(10.0);
    let step = (nice * mag).max(1e-20);
    let mut value = (lo / step).floor() * step;
    if value > lo {
        value -= step;
    }
    let mut out = Vec::new();
    for _ in 0..48 {
        let snapped = (value / step).round() * step;
        if snapped >= lo - span * 1e-4
            && snapped <= hi + span * 1e-4
            && out
                .last()
                .is_none_or(|prev: &f32| (snapped - *prev).abs() > step * 0.25)
        {
            out.push(snapped);
        }
        if value > hi + step {
            break;
        }
        value += step;
    }
    out
}

fn color_axis_ticks(lo: f32, hi: f32, nbins: i32) -> (Vec<f32>, Vec<f32>, Vec<String>, String) {
    if !lo.is_finite() || !hi.is_finite() {
        return (Vec::new(), Vec::new(), Vec::new(), String::new());
    }
    let (lo, hi) = if hi < lo { (hi, lo) } else { (lo, hi) };
    let span = hi - lo;
    if span <= 0.0 {
        return (vec![lo], Vec::new(), Vec::new(), String::new());
    }
    let nbins = nbins.clamp(2, 8);
    let pad = span * 1e-6;
    let mut nice = locator_ticks(lo, hi, nbins);
    nice.retain(|value| *value >= lo - pad && *value <= hi + pad);
    if lo < 0.0 && hi > 0.0 && nice.iter().all(|value| value.abs() > pad) {
        nice.push(0.0);
        nice.sort_by(f32::total_cmp);
    }
    let inner: Vec<f32> = nice
        .into_iter()
        .filter(|value| *value > lo + pad && *value < hi - pad)
        .collect();
    let mut majors = Vec::with_capacity(inner.len() + 2);
    majors.push(lo);
    majors.extend(inner);
    majors.push(hi);
    let mut minors = Vec::new();
    for pair in majors.windows(2) {
        let step = pair[1] - pair[0];
        let parts = minor_parts(step);
        for part in 1..parts {
            let value = pair[0] + step * part as f32 / parts as f32;
            if value > lo && value < hi {
                minors.push(value);
            }
        }
    }
    let (labels, offset) = tick_labels(&majors);
    (majors, minors, labels, offset)
}

fn tick_row(value: f32, lo: f32, hi: f32, bar: &PlotRect) -> f32 {
    let span = (hi - lo).max(1e-12);
    (bar.y + (hi - value) / span * bar.h).round()
}

fn major_tick_length(value: f32, span: f32, labeled: bool) -> f32 {
    if !labeled {
        5.0
    } else if value.abs() <= span * 1e-6 {
        9.0
    } else {
        7.0
    }
}

struct ColorScale {
    bar: PlotRect,
    lo: f32,
    hi: f32,
    majors: Vec<f32>,
    minors: Vec<f32>,
    labels: Vec<String>,
    title: String,
    labeled: Vec<bool>,
}

fn label_slots(majors: &[f32], lo: f32, hi: f32, bar: &PlotRect) -> Vec<bool> {
    let font = atlas();
    let mut labeled = vec![false; majors.len()];
    let mut shown = Vec::new();
    let mut order = Vec::with_capacity(majors.len());
    if !majors.is_empty() {
        order.push(0);
    }
    if majors.len() > 1 {
        order.push(majors.len() - 1);
    }
    if majors.len() > 2 {
        order.extend(1..majors.len() - 1);
    }
    for index in order {
        let y = tick_row(majors[index], lo, hi, bar);
        if shown
            .iter()
            .any(|prev: &f32| (y - *prev).abs() < font.line_height)
        {
            continue;
        }
        labeled[index] = true;
        shown.push(y);
    }
    labeled
}

fn color_scale(plot: &Plot, cell: &Cell, line_scale: f32) -> Option<ColorScale> {
    let plane = cell_plane(plot, cell)?;
    let (mut lo, mut hi, unit) = presented_limits(plot, cell.panel, plane, line_scale)?;
    if hi < lo {
        std::mem::swap(&mut lo, &mut hi);
    }
    if !lo.is_finite() || !hi.is_finite() || hi - lo < 1e-8 {
        return None;
    }
    let bar = color_bar_rect(cell);
    let nbins = (bar.h / 52.0).floor() as i32;
    let (majors, minors, labels, offset) = color_axis_ticks(lo, hi, nbins);
    if majors.len() < 2 || labels.len() != majors.len() {
        return None;
    }
    let title = if offset.is_empty() {
        unit
    } else if unit.is_empty() {
        offset
    } else {
        format!("{unit}  {offset}")
    };
    let labeled = label_slots(&majors, lo, hi, &bar);
    Some(ColorScale {
        bar,
        lo,
        hi,
        majors,
        minors,
        labels,
        title,
        labeled,
    })
}

fn screen_rect(x0: f32, y0: f32, x1: f32, y1: f32, color: [f32; 4], heat: f32) -> QuadRec {
    QuadRec {
        a: [x0, y0],
        b: [x1, y1],
        c: None,
        heat,
        color,
        id: 0,
        heatmap: None,
    }
}

fn color_axis_quads(plot: &Plot, cell: &Cell, line_scale: f32) -> (Option<QuadRec>, Vec<QuadRec>) {
    let Some(scale) = color_scale(plot, cell, line_scale) else {
        return (None, Vec::new());
    };
    let color = plot.foreground;
    let ramp = screen_rect(
        scale.bar.x,
        scale.bar.y,
        scale.bar.x + scale.bar.w,
        scale.bar.y + scale.bar.h,
        [1.0, 1.0, 1.0, 1.0],
        COLOR_RAMP_HEAT,
    );
    let mut ticks = Vec::new();
    let x0 = scale.bar.x;
    let y0 = scale.bar.y;
    let x1 = scale.bar.x + scale.bar.w;
    let y1 = scale.bar.y + scale.bar.h;
    ticks.push(screen_rect(x0, y0, x1, y0 + 1.0, color, COLOR_TICK_HEAT));
    ticks.push(screen_rect(x0, y1 - 1.0, x1, y1, color, COLOR_TICK_HEAT));
    ticks.push(screen_rect(x0, y0, x0 + 1.0, y1, color, COLOR_TICK_HEAT));
    ticks.push(screen_rect(x1 - 1.0, y0, x1, y1, color, COLOR_TICK_HEAT));
    let spine = x1;
    let span = scale.hi - scale.lo;
    for value in &scale.minors {
        let y = tick_row(*value, scale.lo, scale.hi, &scale.bar);
        ticks.push(screen_rect(
            spine,
            y,
            spine + 4.0,
            y + 1.0,
            color,
            COLOR_TICK_HEAT,
        ));
    }
    for (index, value) in scale.majors.iter().enumerate() {
        let y = tick_row(*value, scale.lo, scale.hi, &scale.bar);
        let length = major_tick_length(*value, span, scale.labeled[index]);
        ticks.push(screen_rect(
            spine,
            y,
            spine + length,
            y + 1.0,
            color,
            COLOR_TICK_HEAT,
        ));
    }
    (Some(ramp), ticks)
}

fn color_axis_glyphs(plot: &Plot, cell: &Cell, line_scale: f32) -> Vec<GlyphRec> {
    let Some(scale) = color_scale(plot, cell, line_scale) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    let spine = scale.bar.x + scale.bar.w;
    let span = scale.hi - scale.lo;
    for (index, label) in scale.labels.iter().enumerate() {
        if !scale.labeled[index] {
            continue;
        }
        let length = major_tick_length(scale.majors[index], span, true);
        let y = tick_row(scale.majors[index], scale.lo, scale.hi, &scale.bar) + 0.5;
        push_text(
            &mut out,
            label,
            [spine + length + 4.0, y, 0.0],
            0.0,
            0.0,
            [0.0, ink_center_shift(label)],
            plot.foreground,
        );
    }
    if !scale.title.is_empty() {
        let font = atlas();
        push_up_text(
            &mut out,
            &scale.title,
            [
                cell.cell.x + cell.cell.w - 4.0 - font.line_height * 0.5,
                scale.bar.y + scale.bar.h * 0.5,
            ],
            plot.foreground,
        );
    }
    out
}

#[cfg(not(target_arch = "wasm32"))]
fn paint_color_axis(
    plot: &Plot,
    frame: &mut [u8],
    width: u32,
    height: u32,
    cell: &Cell,
    line_scale: f32,
) {
    let Some(scale) = color_scale(plot, cell, line_scale) else {
        return;
    };
    let Some(grid) = &plot.dose else {
        return;
    };
    let x0 = scale.bar.x.floor() as i32;
    let x1 = (scale.bar.x + scale.bar.w).ceil() as i32;
    let y0 = scale.bar.y.floor() as i32;
    let y1 = (scale.bar.y + scale.bar.h).ceil() as i32;
    for y in y0..y1 {
        let t = (1.0 - ((y as f32 + 0.5 - scale.bar.y) / scale.bar.h)).clamp(0.0, 1.0);
        let rgb = scan_kit_core::sample(grid.ramp, t);
        for x in x0..x1 {
            blend(
                frame,
                width,
                height,
                x,
                y,
                [rgb[0], rgb[1], rgb[2], 1.0],
                None,
            );
        }
    }
    let ink = plot.foreground;
    for x in x0..x1 {
        blend(frame, width, height, x, y0, ink, None);
        blend(frame, width, height, x, y1 - 1, ink, None);
    }
    for y in y0..y1 {
        blend(frame, width, height, x0, y, ink, None);
        blend(frame, width, height, x1 - 1, y, ink, None);
    }
    let spine = (scale.bar.x + scale.bar.w).round() as i32;
    let span = scale.hi - scale.lo;
    for value in &scale.minors {
        let y = tick_row(*value, scale.lo, scale.hi, &scale.bar) as i32;
        for x in spine..(spine + 4) {
            blend(frame, width, height, x, y, ink, None);
        }
    }
    for (index, value) in scale.majors.iter().enumerate() {
        let y = tick_row(*value, scale.lo, scale.hi, &scale.bar) as i32;
        let length = major_tick_length(*value, span, scale.labeled[index]).round() as i32;
        for x in spine..(spine + length) {
            blend(frame, width, height, x, y, ink, None);
        }
    }
}

fn dose_chrome(plot: &Plot, cell: &Cell) -> Vec<GlyphRec> {
    let mut out = Vec::new();
    for label in volume_labels(plot, cell) {
        let mut color = plot.foreground;
        color[3] *= if label.title { 0.75 } else { 0.6 };
        push_text(
            &mut out,
            &label.text,
            [label.at[0], label.at[1], label.angle],
            -1.0,
            0.5,
            [0.0, ink_center_shift(&label.text)],
            color,
        );
    }
    out
}

/// Outward candidates for the X, Y, and depth edges. `tick_near` uses the same pair.
const EDGE_OUT: [[[f32; 3]; 2]; 3] = [
    [[0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
    [[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
];
const AXIS_TITLE: [&str; 3] = ["X (mm)", "Y (mm)", "Depth (mm)"];

fn depth_title(plot: &Plot, panel: usize, axis: usize) -> String {
    if axis == 2 {
        if let Some(label) = plot
            .panels
            .get(panel)
            .map(|item| item.x_label.as_str())
            .filter(|label| !label.is_empty())
        {
            return label.to_string();
        }
    }
    AXIS_TITLE[axis].to_string()
}
const LABEL_GAP_PX: f32 = 14.0;

struct VolumeLabel {
    text: String,
    at: [f32; 2],
    /// Clockwise radians, y down.
    angle: f32,
    title: bool,
}

fn dose_ray(plot: &Plot, grid: &crate::dose::DoseGrid, aspect: f32) -> scan_kit_core::RayView {
    scan_kit_core::RayView {
        azimuth: plot.azimuth,
        elevation: plot.elevation,
        zoom: plot.orbit_zoom,
        aspect,
        mode: grid.mode,
        filter: grid.filter,
        gain: grid.gain,
        opacity: grid.opacity,
        lo: grid.lo,
        hi: grid.hi,
        ramp: grid.ramp,
        line_scale: grid.line_scale,
        fov: plot.fov,
        center: plot.pan,
        gantry: grid.gantry,
    }
}

fn lattice_box(volume: &scan_kit_core::Volume) -> ([f32; 3], [f32; 3]) {
    let voxel = volume.voxel.max(1e-6);
    let extent = [
        volume.shape[0] as f32 * voxel,
        volume.shape[1] as f32 * voxel,
        volume.shape[2] as f32 * voxel,
    ];
    (volume.origin, extent)
}

fn edge_point(axis: usize, mark: f32, origin: [f32; 3], extent: [f32; 3]) -> [f32; 3] {
    let z1 = origin[2] + extent[2];
    match axis {
        0 => [mark, origin[1], z1],
        1 => [origin[0], mark, z1],
        _ => [origin[0], origin[1], mark],
    }
}

fn add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn mul3(a: [f32; 3], scale: f32) -> [f32; 3] {
    [a[0] * scale, a[1] * scale, a[2] * scale]
}

/// Tick positions every 10 mm, plus the box ends. Numbered ticks stay under eight
/// per edge; both ends always carry a number.
fn axis_ticks(lo: f32, hi: f32) -> Vec<(f32, bool)> {
    const STEP: f32 = 10.0;
    let k0 = (lo / STEP - 1e-9).ceil() as i32;
    let k1 = (hi / STEP + 1e-9).floor() as i32;
    let count = if k1 >= k0 { (k1 - k0 + 1) as usize } else { 0 };
    let every = [1i32, 2, 5, 10, 20, 50, 100]
        .into_iter()
        .find(|step| count <= *step as usize * 8)
        .unwrap_or(100);
    let mut ticks = Vec::with_capacity(count + 2);
    for index in 0..count {
        let k = k0 + index as i32;
        let gap = (k - k0).min(k1 - k);
        let major = index == 0
            || index + 1 == count
            || (k.rem_euclid(every) == 0 && gap as f32 >= 0.6 * every as f32);
        ticks.push((k as f32 * STEP, major));
    }
    if ticks.first().is_none_or(|(value, _)| *value - lo > 1e-4) {
        ticks.insert(0, (lo, true));
    }
    if ticks.last().is_none_or(|(value, _)| hi - *value > 1e-4) {
        ticks.push((hi, true));
    }
    ticks
}

fn axis_number(axis: usize, value: f32, hi: f32) -> String {
    format_tick(if axis == 2 { hi - value } else { value })
}

fn film_step(
    volume: &scan_kit_core::Volume,
    view: scan_kit_core::RayView,
    from: [f32; 3],
    to: [f32; 3],
    scale: [f32; 2],
) -> Option<[f32; 2]> {
    let a = scan_kit_core::dose_film_open(volume, view, from)?;
    let b = scan_kit_core::dose_film_open(volume, view, to)?;
    Some([(b[0] - a[0]) * scale[0], (b[1] - a[1]) * scale[1]])
}

fn hypot2(v: [f32; 2]) -> f32 {
    v[0].hypot(v[1])
}

fn clamp_on_film(at: [f32; 2], plot: &PlotRect, half: [f32; 2]) -> [f32; 2] {
    let place = |value: f32, start: f32, span: f32, pad: f32| {
        let lo = start + pad;
        let hi = start + span - pad;
        if lo <= hi {
            value.clamp(lo, hi)
        } else {
            start + span * 0.5
        }
    };
    [
        place(at[0], plot.x, plot.w, half[0]),
        place(at[1], plot.y, plot.h, half[1]),
    ]
}

fn outward_index(edge: [f32; 2], candidates: [[f32; 2]; 2], away: [f32; 2]) -> usize {
    let length = hypot2(edge);
    if length < 1e-6 {
        let score = |cand: [f32; 2]| cand[0] * away[0] + cand[1] * away[1];
        return usize::from(score(candidates[1]) > score(candidates[0]));
    }
    let mut perp = [-edge[1] / length, edge[0] / length];
    if perp[0] * away[0] + perp[1] * away[1] < 0.0 {
        perp = [-perp[0], -perp[1]];
    }
    let score = |cand: [f32; 2]| cand[0] * perp[0] + cand[1] * perp[1];
    usize::from(score(candidates[1]) > score(candidates[0]))
}

fn label_direction(edge: [f32; 2], out: [f32; 2]) -> [f32; 2] {
    let length = hypot2(edge);
    let mut direction = if length > 1e-6 {
        [-edge[1] / length, edge[0] / length]
    } else {
        out
    };
    if length > 1e-6 && direction[0] * out[0] + direction[1] * out[1] < 0.0 {
        direction = [-direction[0], -direction[1]];
    }
    let span = hypot2(direction).max(1e-9);
    [direction[0] / span, direction[1] / span]
}

fn readable_angle(edge: [f32; 2]) -> f32 {
    let mut degrees = edge[1].atan2(edge[0]).to_degrees();
    if degrees > 90.0 {
        degrees -= 180.0;
    } else if degrees < -90.0 {
        degrees += 180.0;
    }
    degrees.to_radians()
}

fn spaced_labels(tips: &[[f32; 2]], need: f32) -> Vec<usize> {
    let count = tips.len();
    if count < 2 {
        return (0..count).collect();
    }
    let mut keep = vec![0];
    for index in 1..count - 1 {
        let previous = tips[keep[keep.len() - 1]];
        if hypot2([tips[index][0] - previous[0], tips[index][1] - previous[1]]) >= need {
            keep.push(index);
        }
    }
    while keep.len() > 1 {
        let previous = tips[keep[keep.len() - 1]];
        let last = tips[count - 1];
        if hypot2([last[0] - previous[0], last[1] - previous[1]]) >= need {
            break;
        }
        keep.pop();
    }
    keep.push(count - 1);
    keep
}

/// Which of [`EDGE_OUT`] faces the camera. Bit 0 is X, bit 1 is Y, bit 2 is depth.
fn axis_sides(volume: &scan_kit_core::Volume, view: scan_kit_core::RayView) -> [usize; 3] {
    let (origin, extent) = lattice_box(volume);
    let center = [
        origin[0] + extent[0] * 0.5,
        origin[1] + extent[1] * 0.5,
        origin[2] + extent[2] * 0.5,
    ];
    let scale = [view.aspect.max(1e-3), 1.0];
    let mut sides = [0; 3];
    let Some(center_uv) = scan_kit_core::dose_film_open(volume, view, center) else {
        return sides;
    };
    for axis in 0..3 {
        let mid = edge_point(axis, origin[axis] + extent[axis] * 0.5, origin, extent);
        let Some(mid_uv) = scan_kit_core::dose_film_open(volume, view, mid) else {
            continue;
        };
        let Some(edge) = film_step(volume, view, mid, add3(mid, axis_along(axis)), scale) else {
            continue;
        };
        let Some(first) = film_step(volume, view, mid, add3(mid, EDGE_OUT[axis][0]), scale) else {
            continue;
        };
        let Some(second) = film_step(volume, view, mid, add3(mid, EDGE_OUT[axis][1]), scale) else {
            continue;
        };
        let away = [
            (mid_uv[0] - center_uv[0]) * scale[0],
            (mid_uv[1] - center_uv[1]) * scale[1],
        ];
        let away = if hypot2(away) > 1e-6 { away } else { first };
        sides[axis] = outward_index(edge, [first, second], away);
    }
    sides
}

fn axis_along(axis: usize) -> [f32; 3] {
    let mut step = [0.0; 3];
    step[axis] = 1.0;
    step
}

fn volume_labels(plot: &Plot, cell: &Cell) -> Vec<VolumeLabel> {
    let Some(grid) = &plot.dose else {
        return Vec::new();
    };
    if plot
        .panels
        .get(cell.panel)
        .and_then(|panel| dose_plane(&panel.title))
        != Some(3)
    {
        return Vec::new();
    }
    let volume = &grid.volume;
    let (origin, extent) = lattice_box(volume);
    if extent.iter().any(|span| *span <= 0.0) {
        return Vec::new();
    }
    let aspect = cell.plot.w / cell.plot.h.max(1.0);
    let view = dose_ray(plot, grid, aspect);
    let sides = axis_sides(volume, view);
    let pixel = [cell.plot.w * crate::dose::FILM_X, cell.plot.h];
    let to_px = |uv: [f32; 2]| -> [f32; 2] {
        [
            cell.plot.x + uv[0] * pixel[0],
            cell.plot.y + uv[1] * pixel[1],
        ]
    };
    let em = atlas().line_height;
    let stub = (0.02 * extent[0].max(extent[1]).max(extent[2])).clamp(1.5, 6.0);
    let mut frames = Vec::new();
    for axis in 0..3 {
        let lo = origin[axis];
        let hi = lo + extent[axis];
        let mid = edge_point(axis, lo + extent[axis] * 0.5, origin, extent);
        let out = EDGE_OUT[axis][sides[axis]];
        let Some(mid_uv) = scan_kit_core::dose_film_open(volume, view, mid) else {
            continue;
        };
        let Some(edge) = film_step(volume, view, mid, add3(mid, axis_along(axis)), pixel) else {
            continue;
        };
        let Some(side) = film_step(volume, view, mid, add3(mid, out), pixel) else {
            continue;
        };
        let per_mm = hypot2(edge);
        frames.push((axis, lo, hi, mid_uv, out, edge, side, per_mm));
    }
    let widest = frames.iter().map(|frame| frame.7).fold(0.0_f32, f32::max);
    let mut labels = Vec::new();
    for (axis, lo, hi, mid_uv, out, edge, side, per_mm) in frames {
        let shown = per_mm * extent[axis] >= 3.0 * em && per_mm >= 0.3 * widest;
        if !shown {
            continue;
        }
        let center = [
            origin[0] + extent[0] * 0.5,
            origin[1] + extent[1] * 0.5,
            origin[2] + extent[2] * 0.5,
        ];
        let away = scan_kit_core::dose_film_open(volume, view, center)
            .map(|uv| {
                let at = to_px(uv);
                let here = to_px(mid_uv);
                [here[0] - at[0], here[1] - at[1]]
            })
            .unwrap_or([0.0, 0.0]);
        let direction = label_direction(
            edge,
            if hypot2(side) >= 0.25 * per_mm {
                side
            } else {
                away
            },
        );
        let angle = readable_angle(edge);
        let gap = LABEL_GAP_PX + 0.8 * em;
        let paired: Vec<(String, [f32; 2])> = axis_ticks(lo, hi)
            .into_iter()
            .filter(|(_, major)| *major)
            .filter_map(|(value, _)| {
                let text = axis_number(axis, value, hi);
                if text.is_empty() {
                    return None;
                }
                let tip = add3(edge_point(axis, value, origin, extent), mul3(out, stub));
                let px = scan_kit_core::dose_film_open(volume, view, tip).map(to_px)?;
                Some((text, px))
            })
            .collect();
        if !paired.is_empty() {
            let wide = paired
                .iter()
                .map(|(text, _)| text::text_width(text))
                .fold(0.0, f32::max);
            let points: Vec<[f32; 2]> = paired.iter().map(|(_, px)| *px).collect();
            for index in spaced_labels(&points, wide + 4.0) {
                let (text, tip) = &paired[index];
                let at = [tip[0] + direction[0] * gap, tip[1] + direction[1] * gap];
                if at[0] < cell.plot.x
                    || at[0] > cell.plot.x + cell.plot.w
                    || at[1] < cell.plot.y
                    || at[1] > cell.plot.y + cell.plot.h
                {
                    continue;
                }
                labels.push(VolumeLabel {
                    text: text.clone(),
                    at,
                    angle,
                    title: false,
                });
            }
        }
        let tick_px = (side[0] * direction[0] + side[1] * direction[1]).abs() * stub;
        let mut title_at = to_px(mid_uv);
        let reach = tick_px + gap + em + LABEL_GAP_PX + 0.6 * em;
        title_at[0] += direction[0] * reach;
        title_at[1] += direction[1] * reach;
        let title = depth_title(plot, cell.panel, axis);
        let (sin_t, cos_t) = angle.sin_cos();
        let width = text::text_width(&title);
        let half = [
            0.5 * width * cos_t.abs() + 0.5 * em * sin_t.abs(),
            0.5 * width * sin_t.abs() + 0.5 * em * cos_t.abs(),
        ];
        title_at = clamp_on_film(title_at, &cell.plot, half);
        labels.push(VolumeLabel {
            text: title,
            at: title_at,
            angle,
            title: true,
        });
    }
    labels
}

fn labels_for(
    panel: &Panel,
    camera: &Camera,
    cell: &Cell,
    color: [f32; 4],
    title: bool,
) -> Vec<GlyphRec> {
    let mut out = Vec::new();
    let font = atlas();
    if title && shows_title(panel) {
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

#[derive(Clone, Copy)]
enum SplitGrab {
    /// Horizontal gap under this row.
    Rows(usize),
    /// Vertical gap inside this row.
    Cols(usize),
}

fn dose_plane(title: &str) -> Option<u8> {
    let title = title.to_ascii_lowercase();
    if title.starts_with("axial") {
        Some(0)
    } else if title.starts_with("coronal") {
        Some(1)
    } else if title.starts_with("sagittal") {
        Some(2)
    } else if title.starts_with("3d") {
        Some(3)
    } else {
        None
    }
}

fn cached_depth_integral(grid: &crate::dose::DoseGrid, y: usize) -> (Vec<f32>, Vec<f32>) {
    let [_, ny, nz] = grid.volume.shape;
    let y = y.min(ny.saturating_sub(1));
    let voxel = grid.volume.voxel;
    let face = grid.volume.origin[2] + nz as f32 * voxel;
    let mut depth = Vec::with_capacity(nz);
    let mut dose = Vec::with_capacity(nz);
    for z in (0..nz).rev() {
        depth.push(face - (grid.volume.origin[2] + (z as f32 + 0.5) * voxel));
        let sum = grid.integrals[2].get(y + ny * z).copied().unwrap_or(0.0);
        dose.push(sum * voxel);
    }
    (depth, dose)
}

fn cached_lateral_integral(
    grid: &crate::dose::DoseGrid,
    x: usize,
    z: usize,
) -> (Vec<f32>, Vec<f32>) {
    let [nx, _, nz] = grid.volume.shape;
    let x = x.min(nx.saturating_sub(1));
    let z = z.min(nz.saturating_sub(1));
    let voxel = grid.volume.voxel;
    let x0 = grid.volume.origin[0] + (x as f32 + 0.5) * voxel;
    let mut xs = Vec::with_capacity(nx);
    let mut dose = Vec::with_capacity(nx);
    for column in 0..nx {
        xs.push(grid.volume.origin[0] + (column as f32 + 0.5) * voxel - x0);
        let sum = grid.integrals[1]
            .get(column + nx * z)
            .copied()
            .unwrap_or(0.0);
        dose.push(sum * voxel);
    }
    (xs, dose)
}

fn cached_longitudinal_integral(
    grid: &crate::dose::DoseGrid,
    y: usize,
    z: usize,
) -> (Vec<f32>, Vec<f32>) {
    let [_, ny, nz] = grid.volume.shape;
    let y = y.min(ny.saturating_sub(1));
    let z = z.min(nz.saturating_sub(1));
    let voxel = grid.volume.voxel;
    let z0 = grid.volume.origin[2] + (z as f32 + 0.5) * voxel;
    let mut xs = Vec::with_capacity(nz);
    let mut dose = Vec::with_capacity(nz);
    for depth in 0..nz {
        let z_mm = grid.volume.origin[2] + (depth as f32 + 0.5) * voxel;
        xs.push(z0 - z_mm);
        let sum = grid.integrals[2]
            .get(y + ny * depth)
            .copied()
            .unwrap_or(0.0);
        dose.push(sum * voxel);
    }
    (xs, dose)
}

fn integral_profile_axis(grid: &crate::dose::DoseGrid, kind: u8) -> (f32, f32) {
    let voxel = grid.volume.voxel;
    let mut lo = 0.0f32;
    let mut hi = 0.0f32;
    let mut any = false;
    let fold = |values: &[f32], lo: &mut f32, hi: &mut f32, any: &mut bool| {
        for value in values {
            if !value.is_finite() {
                continue;
            }
            let shown = value * voxel;
            if *any {
                *lo = lo.min(shown);
                *hi = hi.max(shown);
            } else {
                *lo = shown;
                *hi = shown;
                *any = true;
            }
        }
    };
    match kind {
        1 => fold(&grid.integrals[1], &mut lo, &mut hi, &mut any),
        0 | 2 => fold(&grid.integrals[2], &mut lo, &mut hi, &mut any),
        _ => {
            fold(&grid.integrals[1], &mut lo, &mut hi, &mut any);
            fold(&grid.integrals[2], &mut lo, &mut hi, &mut any);
        }
    }
    for volume in grid
        .sessions
        .iter()
        .chain(grid.companions.iter())
        .chain(grid.plans.iter())
        .flatten()
    {
        fold_cube(volume, kind, &mut lo, &mut hi, &mut any);
    }
    if !any || !lo.is_finite() || !hi.is_finite() {
        return (0.0, 1.0);
    }
    if (hi - lo).abs() < 1.0e-4 {
        (lo - 0.5, hi + 0.5)
    } else if lo <= hi {
        (lo, hi)
    } else {
        (hi, lo)
    }
}

fn profile_kind(title: &str) -> Option<u8> {
    let title = title.to_ascii_lowercase();
    if title.starts_with("depth") {
        Some(0)
    } else if title.starts_with("lateral +") || title.starts_with("lateral+") {
        Some(3)
    } else if title.starts_with("lateral") {
        Some(1)
    } else if title.starts_with("longitudinal") {
        Some(2)
    } else {
        None
    }
}

fn cursor_index(plane: u8, cursor: [usize; 3]) -> usize {
    match plane {
        1 => cursor[1],
        2 => cursor[0],
        _ => cursor[2],
    }
}

fn structure_color(id: u8) -> [f32; 4] {
    const INK: [[f32; 4]; 4] = [
        [0.95, 0.55, 0.2, 0.95],
        [0.35, 0.75, 0.95, 0.95],
        [0.55, 0.9, 0.45, 0.95],
        [0.9, 0.45, 0.7, 0.95],
    ];
    INK[(id as usize).saturating_sub(1) % INK.len()]
}

fn session_volume(dose: &scan_kit_core::SessionDose) -> Option<scan_kit_core::Volume> {
    if dose.values.is_empty() {
        return None;
    }
    let shape = [
        dose.shape[0] as usize,
        dose.shape[1] as usize,
        dose.shape[2] as usize,
    ];
    let n = shape[0].saturating_mul(shape[1]).saturating_mul(shape[2]);
    if n == 0 || dose.values.len() < n {
        return None;
    }
    let mut values = dose.values.clone();
    values.truncate(n);
    Some(scan_kit_core::Volume {
        origin: dose.origin,
        shape,
        voxel: dose.voxel.max(1e-3),
        values,
    })
}

fn fold_cube(volume: &scan_kit_core::Volume, kind: u8, lo: &mut f32, hi: &mut f32, any: &mut bool) {
    let [nx, ny, nz] = volume.shape;
    if nx == 0 || ny == 0 || nz == 0 {
        return;
    }
    let area = volume.voxel * volume.voxel;
    let mut touch = |sum: f32| {
        if !sum.is_finite() {
            return;
        }
        if *any {
            *lo = lo.min(sum);
            *hi = hi.max(sum);
        } else {
            *lo = sum;
            *hi = sum;
            *any = true;
        }
    };
    let sum_x = kind != 1;
    let sum_y = kind == 1 || kind >= 3;
    if sum_x {
        for y in 0..ny {
            for z in 0..nz {
                let mut sum = 0.0;
                for x in 0..nx {
                    sum += volume.get(x, y, z);
                }
                touch(sum * area);
            }
        }
    }
    if sum_y {
        for z in 0..nz {
            for x in 0..nx {
                let mut sum = 0.0;
                for y in 0..ny {
                    sum += volume.get(x, y, z);
                }
                touch(sum * area);
            }
        }
    }
}

fn profile_curves(
    grid: &crate::dose::DoseGrid,
    volume: &scan_kit_core::Volume,
    own: bool,
    kind: u8,
    at: [usize; 3],
    integrate: bool,
) -> Vec<(Vec<f32>, Vec<f32>)> {
    let [ix, iy, iz] = at;
    let mut series = Vec::new();
    match kind {
        0 if integrate && own => series.push(cached_depth_integral(grid, iy)),
        0 if integrate => series.push(volume.depth_integral(iy)),
        0 => series.push(volume.depth_profile(ix, iy)),
        1 if integrate && own => series.push(cached_lateral_integral(grid, ix, iz)),
        1 if integrate => series.push(volume.lateral_integral(ix, iz)),
        1 => series.push(volume.lateral_profile(ix, iy, iz)),
        2 if integrate && own => series.push(cached_longitudinal_integral(grid, iy, iz)),
        2 if integrate => series.push(volume.longitudinal_integral(iy, iz)),
        2 => series.push(volume.longitudinal_profile(ix, iy, iz)),
        _ if integrate && own => {
            series.push(cached_lateral_integral(grid, ix, iz));
            series.push(cached_longitudinal_integral(grid, iy, iz));
        }
        _ if integrate => {
            series.push(volume.lateral_integral(ix, iz));
            series.push(volume.longitudinal_integral(iy, iz));
        }
        _ => {
            series.push(volume.lateral_profile(ix, iy, iz));
            series.push(volume.longitudinal_profile(ix, iy, iz));
        }
    }
    series
}

fn profile_lines(
    grid: &crate::dose::DoseGrid,
    kind: u8,
    cursor: [usize; 3],
    integrate: bool,
    fallback: [f32; 4],
) -> Vec<LineRec> {
    let count = grid.sessions.len().max(1);
    let focus = grid.session_focus.min(count - 1);
    let mut order: Vec<usize> = (0..count).filter(|index| *index != focus).collect();
    order.push(focus);
    let home = &grid.volume;
    let mm = [
        home.mm_of(0, cursor[0].min(home.shape[0].saturating_sub(1))),
        home.mm_of(1, cursor[1].min(home.shape[1].saturating_sub(1))),
        home.mm_of(2, cursor[2].min(home.shape[2].saturating_sub(1))),
    ];
    let mut lines = Vec::new();
    for index in order {
        let foreign = grid.sessions.get(index).and_then(|slot| slot.as_ref());
        let at = match foreign {
            Some(volume) => [
                volume.index_of(0, mm[0]),
                volume.index_of(1, mm[1]),
                volume.index_of(2, mm[2]),
            ],
            None => cursor,
        };
        let color = grid.session_colors.get(index).copied().unwrap_or(fallback);
        let volume = foreign.unwrap_or(home);
        push_profile_curves(
            &mut lines,
            grid,
            volume,
            foreign.is_none(),
            kind,
            at,
            integrate,
            color,
            1.0,
            false,
        );
        if let Some(extra) = grid.companions.get(index).and_then(|slot| slot.as_ref()) {
            push_profile_curves(
                &mut lines,
                grid,
                extra,
                false,
                kind,
                index_mm(extra, mm),
                integrate,
                color,
                0.55,
                false,
            );
        }
        if let Some(extra) = grid.plans.get(index).and_then(|slot| slot.as_ref()) {
            push_profile_curves(
                &mut lines,
                grid,
                extra,
                false,
                kind,
                index_mm(extra, mm),
                integrate,
                color,
                1.0,
                true,
            );
        }
    }
    lines
}

fn index_mm(volume: &scan_kit_core::Volume, mm: [f32; 3]) -> [usize; 3] {
    [
        volume.index_of(0, mm[0]),
        volume.index_of(1, mm[1]),
        volume.index_of(2, mm[2]),
    ]
}

fn push_profile_curves(
    lines: &mut Vec<LineRec>,
    grid: &crate::dose::DoseGrid,
    volume: &scan_kit_core::Volume,
    own: bool,
    kind: u8,
    at: [usize; 3],
    integrate: bool,
    color: [f32; 4],
    fade: f32,
    dashed: bool,
) {
    for (curve, (xs, ys)) in profile_curves(grid, volume, own, kind, at, integrate)
        .into_iter()
        .enumerate()
    {
        let mut ink = if curve == 0 {
            color
        } else {
            [color[0], color[1], color[2], color[3] * 0.55]
        };
        ink[3] *= fade;
        stroke_profile(lines, xs, ys, ink, dashed);
    }
}

fn stroke_profile(
    lines: &mut Vec<LineRec>,
    xs: Vec<f32>,
    ys: Vec<f32>,
    color: [f32; 4],
    dashed: bool,
) {
    let mut prev: Option<[f32; 3]> = None;
    let mut step = 0u32;
    for (x, y) in xs.into_iter().zip(ys) {
        if !x.is_finite() || !y.is_finite() {
            prev = None;
            continue;
        }
        let point = [x, y, 0.0];
        if let Some(start) = prev {
            // ponytail: 3 samples on, 2 off. A shader dash is a line attribute,
            // which means changing LINE_STRIDE with the encoder and the WGSL inputs.
            let draw = !dashed || step % 5 < 3;
            if draw {
                lines.push(LineRec {
                    a: start,
                    b: point,
                    color,
                    thickness: 1.5,
                    id: 0,
                });
            }
            step += 1;
        }
        prev = Some(point);
    }
}

const FIELD_INK: [f32; 4] = [0.95, 0.72, 0.2, 0.95];

fn field_lines(field: [f32; 6], plane: u8, bounds: [f32; 4], turns: u8) -> Vec<LineRec> {
    if plane >= 3 || !(field[1] > field[0] && field[3] > field[2] && field[5] > field[4]) {
        return Vec::new();
    }
    let (x0, x1, y0, y1) = match plane {
        1 => (field[0], field[1], field[4], field[5]),
        2 => (field[2], field[3], field[4], field[5]),
        _ => (field[0], field[1], field[2], field[3]),
    };
    let corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)];
    let end = |px: f32, py: f32| {
        let point = crate::dose::display_mm(px, py, bounds, turns);
        [point[0], point[1], 0.0]
    };
    (0..4)
        .map(|i| {
            let (ax, ay) = corners[i];
            let (bx, by) = corners[(i + 1) % 4];
            LineRec {
                a: end(ax, ay),
                b: end(bx, by),
                color: FIELD_INK,
                thickness: 1.5,
                id: 0,
            }
        })
        .collect()
}

fn crosshair_lines(
    grid: &crate::dose::DoseGrid,
    plane: u8,
    cursor: [usize; 3],
    bounds: [f32; 4],
    turns: u8,
    color: [f32; 4],
) -> Vec<LineRec> {
    let volume = &grid.volume;
    let (x, y) = match plane {
        1 => (volume.mm_of(0, cursor[0]), volume.mm_of(2, cursor[2])),
        2 => (volume.mm_of(1, cursor[1]), volume.mm_of(2, cursor[2])),
        _ => (volume.mm_of(0, cursor[0]), volume.mm_of(1, cursor[1])),
    };
    let [xmin, xmax, ymin, ymax] = bounds;
    let ink = [color[0], color[1], color[2], 0.85];
    let end = |px: f32, py: f32| {
        let point = crate::dose::display_mm(px, py, bounds, turns);
        [point[0], point[1], 0.0]
    };
    vec![
        LineRec {
            a: end(x, ymin),
            b: end(x, ymax),
            color: ink,
            thickness: 1.0,
            id: 0,
        },
        LineRec {
            a: end(xmin, y),
            b: end(xmax, y),
            color: ink,
            thickness: 1.0,
            id: 0,
        },
    ]
}

fn hit_split(
    count: usize,
    width: u32,
    height: u32,
    row_weights: &[f32],
    row_splits: &[f32],
    x: f32,
    y: f32,
) -> Option<SplitGrab> {
    let rects = panel_rects(count, width, height, 2, &[], row_weights, row_splits, 0);
    let rows = count / 2;
    for row in 0..rows {
        let left = &rects[row * 2];
        let right = &rects[row * 2 + 1];
        if y >= left.y && y <= left.y + left.h && x >= left.x + left.w && x <= right.x {
            return Some(SplitGrab::Cols(row));
        }
    }
    for row in 0..rows.saturating_sub(1) {
        let above = &rects[row * 2];
        let below = &rects[(row + 1) * 2];
        if y >= above.y + above.h && y <= below.y && x >= above.x && x <= width as f32 {
            return Some(SplitGrab::Rows(row));
        }
    }
    None
}

fn row_split_layout(count: usize, columns: u32, row_weights: &[f32], row_splits: &[f32]) -> bool {
    columns == 2
        && count >= 2
        && count.is_multiple_of(2)
        && row_splits.len() == count
        && row_weights.len() == count / 2
}

fn panel_rects(
    count: usize,
    width: u32,
    height: u32,
    columns: u32,
    weights: &[f32],
    row_weights: &[f32],
    row_splits: &[f32],
    side: u32,
) -> Vec<PlotRect> {
    if count == 0 {
        return Vec::new();
    }
    if side > 0 && (side as usize) < count {
        // A weight per grid column plus one for the side column means the main
        // panels are a row-major grid and the side panels stack beside it.
        if columns > 1 && weights.len() == columns as usize + 1 {
            return grid_with_side(
                count,
                width,
                height,
                columns as usize,
                weights,
                side as usize,
            );
        }
        return side_column(count, width, height, weights, side as usize);
    }
    if row_split_layout(count, columns, row_weights, row_splits) {
        return split_rows(count, width, height, row_weights, row_splits);
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

/// Three rows of two cells. `row_weights` is the vertical split. Each pair in
/// `row_splits` is that row's own horizontal split.
fn split_rows(
    count: usize,
    width: u32,
    height: u32,
    row_weights: &[f32],
    row_splits: &[f32],
) -> Vec<PlotRect> {
    let rows = count / 2;
    let gap = 12.0f32;
    let margin = 8.0f32;
    let margin_right = 16.0f32;
    let inner_h = height as f32 - margin * 2.0 - gap * (rows.saturating_sub(1) as f32);
    let row_sum = row_weights.iter().sum::<f32>().max(1.0e-6);
    let inner_w = width as f32 - margin - margin_right - gap;
    let mut rects = Vec::with_capacity(count);
    let mut y = margin;
    for row in 0..rows {
        let cell_h = inner_h * row_weights[row] / row_sum;
        let left = row_splits[row * 2].max(0.0);
        let right = row_splits[row * 2 + 1].max(0.0);
        let sum = (left + right).max(1.0e-6);
        let left_w = inner_w * left / sum;
        let right_w = inner_w * right / sum;
        rects.push(PlotRect {
            x: margin,
            y,
            w: left_w,
            h: cell_h,
        });
        rects.push(PlotRect {
            x: margin + left_w + gap,
            y,
            w: right_w,
            h: cell_h,
        });
        y += cell_h;
        if row + 1 < rows {
            y += gap;
        }
    }
    rects
}

/// Left column stacks `count - side` panels. The last `side` panels fill the right
/// column and share that same height.
fn side_column(
    count: usize,
    width: u32,
    height: u32,
    weights: &[f32],
    side: usize,
) -> Vec<PlotRect> {
    let main = count - side;
    let gap = 12.0f32;
    let margin = 8.0f32;
    let margin_right = 16.0f32;
    let inner_w = width as f32 - margin - margin_right - gap;
    let col_weight = if weights.len() == 2 {
        [weights[0], weights[1]]
    } else {
        [1.0, 1.0]
    };
    let weight_sum = (col_weight[0] + col_weight[1]).max(1.0e-6);
    let left_w = inner_w * col_weight[0] / weight_sum;
    let right_w = inner_w * col_weight[1] / weight_sum;
    let inner_h = height as f32 - margin * 2.0;
    let mut rects = Vec::with_capacity(count);
    rects.extend(stack_column(main, margin, left_w, margin, inner_h, gap));
    rects.extend(stack_column(
        side,
        margin + left_w + gap,
        right_w,
        margin,
        inner_h,
        gap,
    ));
    rects
}

/// Main panels fill a row-major grid. The last `side` panels stack in the
/// column to the right of that grid and share its full height.
fn grid_with_side(
    count: usize,
    width: u32,
    height: u32,
    columns: usize,
    weights: &[f32],
    side: usize,
) -> Vec<PlotRect> {
    let main = count - side;
    let cols = columns.max(1);
    let rows = main.div_ceil(cols).max(1);
    let gap = 12.0f32;
    let margin = 8.0f32;
    let margin_right = 16.0f32;
    let bands = cols + 1;
    let inner_w = width as f32 - margin - margin_right - gap * (bands.saturating_sub(1) as f32);
    let weight_sum = weights.iter().sum::<f32>().max(1.0e-6);
    let mut col_x = Vec::with_capacity(bands);
    let mut col_w = Vec::with_capacity(bands);
    let mut x = margin;
    for (index, weight) in weights.iter().enumerate() {
        let cell_w = inner_w * weight / weight_sum;
        col_x.push(x);
        col_w.push(cell_w);
        x += cell_w;
        if index + 1 < bands {
            x += gap;
        }
    }
    let inner_h = height as f32 - margin * 2.0;
    let mut rects = Vec::with_capacity(count);
    let row_gaps = gap * rows.saturating_sub(1) as f32;
    let row_h = ((inner_h - row_gaps) / rows as f32).max(1.0);
    for index in 0..main {
        let col = index % cols;
        let row = index / cols;
        rects.push(PlotRect {
            x: col_x[col],
            y: margin + row as f32 * (row_h + gap),
            w: col_w[col],
            h: row_h,
        });
    }
    rects.extend(stack_column(
        side,
        col_x[cols],
        col_w[cols],
        margin,
        inner_h,
        gap,
    ));
    rects
}

fn stack_column(rows: usize, x: f32, w: f32, top: f32, inner_h: f32, gap: f32) -> Vec<PlotRect> {
    let rows = rows.max(1);
    let gaps = gap * rows.saturating_sub(1) as f32;
    let h = ((inner_h - gaps) / rows as f32).max(1.0);
    (0..rows)
        .map(|row| PlotRect {
            x,
            y: top + row as f32 * (h + gap),
            w,
            h,
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
    let (sin_t, cos_t) = glyph.anchor[2].sin_cos();
    let turned = glyph.anchor[3] < 0.5 && glyph.anchor[2] > 0.5;
    let spun = glyph.anchor[3] < -0.5;
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
            let (dx, dy) = if spun {
                (
                    cos_t * local_x - sin_t * local_y,
                    sin_t * local_x + cos_t * local_y,
                )
            } else if turned {
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
        push_f32(&mut out, point.time);
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
            time: f32_at(chunk, 9),
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

fn encode_uniform(
    matrix: [f32; 16],
    width: f32,
    height: f32,
    dose: [[f32; 4]; 10],
) -> [u8; UNIFORM_BYTES] {
    let mut out = [0u8; UNIFORM_BYTES];
    for (index, value) in matrix.iter().enumerate() {
        out[index * 4..index * 4 + 4].copy_from_slice(&value.to_le_bytes());
    }
    out[64..68].copy_from_slice(&width.to_le_bytes());
    out[68..72].copy_from_slice(&height.to_le_bytes());
    for (index, row) in dose.iter().enumerate() {
        for (lane, value) in row.iter().enumerate() {
            let at = 80 + (index * 4 + lane) * 4;
            out[at..at + 4].copy_from_slice(&value.to_le_bytes());
        }
    }
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
                size: Some(NonZeroU64::new(UNIFORM_BYTES as u64).unwrap()),
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

#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
fn reuse_line_buffer(
    previous: &mut Plot,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    offset: u64,
    suffix: &[u8],
) -> wgpu::Buffer {
    let gpu = previous
        .gpu
        .as_mut()
        .expect("trace lines are already uploaded");
    let needed = (offset + suffix.len() as u64).max(4);
    let dummy = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("lines-moved"),
        size: 4,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let current = std::mem::replace(&mut gpu.lines, dummy);
    if current.size() >= needed {
        if !suffix.is_empty() {
            queue.write_buffer(&current, offset, suffix);
        }
        return current;
    }
    let next = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("lines"),
        size: needed,
        usage: wgpu::BufferUsages::VERTEX
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    if offset > 0 && current.size() >= offset {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(&current, 0, &next, 0, offset);
        queue.submit(Some(encoder.finish()));
    }
    if !suffix.is_empty() {
        queue.write_buffer(&next, offset, suffix);
    }
    next
}

#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
fn stitched_line_bytes(previous: &Plot, prefix_bytes: usize, suffix: &[u8]) -> Option<Vec<u8>> {
    let encoded = previous.encoded.as_ref()?;
    if encoded.lines.len() < prefix_bytes {
        return None;
    }
    let mut bytes = Vec::with_capacity(prefix_bytes + suffix.len());
    bytes.extend_from_slice(&encoded.lines[..prefix_bytes]);
    bytes.extend_from_slice(suffix);
    Some(bytes)
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
        usage: wgpu::BufferUsages::VERTEX
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
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

fn write_rgba_row(queue: &wgpu::Queue, texture: &wgpu::Texture, row: &[u8], cols: u32, y: u32) {
    let cols = cols.max(1);
    let row_bytes = cols * 4;
    let stride = row_bytes.next_multiple_of(256);
    let mut padded = vec![0u8; stride as usize];
    let take = row.len().min(row_bytes as usize);
    padded[..take].copy_from_slice(&row[..take]);
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d { x: 0, y, z: 0 },
            aspect: wgpu::TextureAspect::All,
        },
        &padded,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(stride),
            rows_per_image: Some(1),
        },
        wgpu::Extent3d {
            width: cols,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
}

fn write_rgba(queue: &wgpu::Queue, texture: &wgpu::Texture, pixels: &[u8], cols: u32, rows: u32) {
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
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
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
}

fn volume_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    dose: &wgpu::Texture,
    bricks: &wgpu::Texture,
    integral: &wgpu::Texture,
) -> wgpu::BindGroup {
    let dose_view = dose.create_view(&Default::default());
    let brick_view = bricks.create_view(&Default::default());
    let integral_view = integral.create_view(&Default::default());
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("dose-volume"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&dose_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&brick_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&integral_view),
            },
        ],
    })
}

fn integral_planes(grid: &crate::dose::DoseGrid) -> (Vec<f32>, u32, u32) {
    let [nx, ny, nz] = grid.volume.shape;
    let width = nx.max(ny).max(1);
    let height = (ny + nz + nz).max(1);
    let mut image = vec![0.0f32; width * height];
    for (index, value) in grid.integrals[0].iter().enumerate() {
        let x = index % nx.max(1);
        let y = index / nx.max(1);
        if x < width && y < height {
            image[x + width * y] = *value;
        }
    }
    for (index, value) in grid.integrals[1].iter().enumerate() {
        let x = index % nx.max(1);
        let z = index / nx.max(1);
        let row = ny + z;
        if x < width && row < height {
            image[x + width * row] = *value;
        }
    }
    for (index, value) in grid.integrals[2].iter().enumerate() {
        let y = index % ny.max(1);
        let z = index / ny.max(1);
        let row = ny + nz + z;
        if y < width && row < height {
            image[y + width * row] = *value;
        }
    }
    (image, width as u32, height as u32)
}

/// Two zero voxels on every side, so linear and cubic taps that fall outside
/// the box read 0. The march box stays the unpadded shape.
fn upload_dose_f16(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    values: &[f32],
    shape: [usize; 3],
) -> Result<wgpu::Texture, GpuError> {
    let nx = shape[0].max(1);
    let ny = shape[1].max(1);
    let nz = shape[2].max(1);
    let width = nx + 4;
    let height = ny + 4;
    let depth = nz + 4;
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("dose-volume"),
        size: wgpu::Extent3d {
            width: width as u32,
            height: height as u32,
            depth_or_array_layers: depth as u32,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::R16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let stride = ((width * 2) as u32).next_multiple_of(256);
    let mut padded = vec![0u8; stride as usize * height * depth];
    if shape[0] > 0 && shape[1] > 0 && shape[2] > 0 {
        for z in 0..nz.min(shape[2]) {
            for y in 0..ny.min(shape[1]) {
                let src = nx.min(shape[0]) * (y + shape[1] * z);
                let row = ((z + 2) * height + (y + 2)) * stride as usize;
                for x in 0..nx.min(shape[0]) {
                    let bits = f32_to_f16(values.get(src + x).copied().unwrap_or(0.0));
                    let at = row + (x + 2) * 2;
                    padded[at..at + 2].copy_from_slice(&bits.to_le_bytes());
                }
            }
        }
    }
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
            rows_per_image: Some(height as u32),
        },
        wgpu::Extent3d {
            width: width as u32,
            height: height as u32,
            depth_or_array_layers: depth as u32,
        },
    );
    Ok(texture)
}

fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let mut mantissa = bits & 0x007f_ffff;
    let mut exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    if (bits & 0x7fff_ffff) == 0 {
        return sign;
    }
    if ((bits >> 23) & 0xff) == 255 {
        return sign | 0x7c00 | if mantissa == 0 { 0 } else { 0x200 };
    }
    if exp >= 31 {
        return sign | 0x7c00;
    }
    if exp <= 0 {
        return sign;
    }
    let dropped = mantissa & 0x1fff;
    mantissa >>= 13;
    if dropped > 0x1000 || (dropped == 0x1000 && mantissa & 1 == 1) {
        mantissa += 1;
        if mantissa == 0x400 {
            mantissa = 0;
            exp += 1;
            if exp >= 31 {
                return sign | 0x7c00;
            }
        }
    }
    sign | ((exp as u16) << 10) | (mantissa as u16)
}

fn upload_r32_2d(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    values: &[f32],
    width: u32,
    height: u32,
) -> Result<wgpu::Texture, GpuError> {
    let width = width.max(1);
    let height = height.max(1);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("dose-integral"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let stride = (width * 4).next_multiple_of(256);
    let mut padded = vec![0u8; (stride * height) as usize];
    for y in 0..height {
        let row = (y * stride) as usize;
        for x in 0..width {
            let value = values.get((x + width * y) as usize).copied().unwrap_or(0.0);
            let at = row + x as usize * 4;
            padded[at..at + 4].copy_from_slice(&value.to_le_bytes());
        }
    }
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
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    Ok(texture)
}

fn upload_r32_3d(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    values: &[f32],
    shape: [usize; 3],
) -> Result<wgpu::Texture, GpuError> {
    let width = shape[0].max(1) as u32;
    let height = shape[1].max(1) as u32;
    let depth = shape[2].max(1) as u32;
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("dose-volume"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: depth,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::R32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let stride = (width * 4).next_multiple_of(256);
    let mut padded = vec![0u8; (stride * height * depth) as usize];
    for z in 0..depth {
        for y in 0..height {
            let row = ((z * height + y) * stride) as usize;
            for x in 0..width {
                let index =
                    x as usize + width as usize * (y as usize + height as usize * z as usize);
                let value = values.get(index).copied().unwrap_or(0.0);
                let at = row + x as usize * 4;
                padded[at..at + 4].copy_from_slice(&value.to_le_bytes());
            }
        }
    }
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
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: depth,
        },
    );
    Ok(texture)
}

/// Average voxels by one integer factor on every axis so the grid fits a 3D texture.
fn pool_volume(volume: &scan_kit_core::Volume, limit: u32) -> scan_kit_core::Volume {
    let factor = volume
        .shape
        .iter()
        .copied()
        .max()
        .unwrap_or(1)
        .div_ceil(limit.max(1) as usize)
        .max(1);
    let shape = [
        volume.shape[0].div_ceil(factor),
        volume.shape[1].div_ceil(factor),
        volume.shape[2].div_ceil(factor),
    ];
    let mut values = vec![0.0f32; shape[0] * shape[1] * shape[2]];
    for z in 0..shape[2] {
        for y in 0..shape[1] {
            for x in 0..shape[0] {
                let mut sum = 0.0f32;
                let mut count = 0.0f32;
                for dz in 0..factor {
                    for dy in 0..factor {
                        for dx in 0..factor {
                            let xx = x * factor + dx;
                            let yy = y * factor + dy;
                            let zz = z * factor + dz;
                            if xx < volume.shape[0] && yy < volume.shape[1] && zz < volume.shape[2]
                            {
                                sum += volume.get(xx, yy, zz);
                                count += 1.0;
                            }
                        }
                    }
                }
                values[x + shape[0] * (y + shape[1] * z)] =
                    if count > 0.0 { sum / count } else { 0.0 };
            }
        }
    }
    scan_kit_core::Volume {
        origin: volume.origin,
        shape,
        voxel: volume.voxel * factor as f32,
        values,
    }
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
    write_rgba(queue, &texture, pixels, cols, rows);
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

/// Vispy key name, or the DOM name, to the next azimuth, elevation, fov, and saved perspective fov.
fn blender_step(
    azimuth: f32,
    elevation: f32,
    fov: f32,
    persp: f32,
    key: &str,
    ctrl: bool,
) -> Option<(f32, f32, f32, f32)> {
    let digit = match key {
        "1" | "Digit1" | "Numpad1" | "End" => '1',
        "2" | "Digit2" | "Numpad2" | "ArrowDown" | "Down" => '2',
        "3" | "Digit3" | "Numpad3" | "PageDown" => '3',
        "4" | "Digit4" | "Numpad4" | "ArrowLeft" | "Left" => '4',
        "5" | "Digit5" | "Numpad5" | "Clear" => '5',
        "6" | "Digit6" | "Numpad6" | "ArrowRight" | "Right" => '6',
        "7" | "Digit7" | "Numpad7" | "Home" => '7',
        "8" | "Digit8" | "Numpad8" | "ArrowUp" | "Up" => '8',
        "9" | "Digit9" | "Numpad9" | "PageUp" => '9',
        _ => return None,
    };
    if digit == '5' {
        return Some(if fov <= 1.0e-4 {
            (azimuth, elevation, persp.max(1.0e-3), persp)
        } else {
            (azimuth, elevation, 0.0, fov)
        });
    }
    if digit == '9' {
        if elevation.abs() >= 80.0_f32.to_radians() {
            return Some((azimuth, -elevation, fov, persp));
        }
        return Some((
            (azimuth + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU),
            elevation,
            fov,
            persp,
        ));
    }
    if matches!(digit, '2' | '4' | '6' | '8') && !ctrl {
        let (daz, del) = match digit {
            '4' => (ORBIT_STEP, 0.0),
            '6' => (-ORBIT_STEP, 0.0),
            '8' => (0.0, ORBIT_STEP),
            _ => (0.0, -ORBIT_STEP),
        };
        return Some((
            azimuth + daz,
            (elevation + del).clamp(-std::f32::consts::FRAC_PI_2, std::f32::consts::FRAC_PI_2),
            fov,
            persp,
        ));
    }
    let pose = match (digit, ctrl) {
        ('1', false) => (180.0_f32.to_radians(), 0.0),
        ('1', true) => (0.0, 0.0),
        ('3', false) => (90.0_f32.to_radians(), 0.0),
        ('3', true) => ((-90.0_f32).to_radians(), 0.0),
        ('7', false) => (0.0, std::f32::consts::FRAC_PI_2),
        ('7', true) => (0.0, -std::f32::consts::FRAC_PI_2),
        _ => return None,
    };
    Some((pose.0, pose.1, fov, persp))
}

fn turntable_axes(azimuth: f32, elevation: f32) -> ([f32; 3], [f32; 3]) {
    let (sa, ca) = azimuth.sin_cos();
    let (se, ce) = elevation.sin_cos();
    let look = unit([-ce * sa, ce * ca, -se]);
    let right = unit([ca, sa, 0.0]);
    let up = [
        right[1] * look[2] - right[2] * look[1],
        right[2] * look[0] - right[0] * look[2],
        right[0] * look[1] - right[1] * look[0],
    ];
    (right, unit(up))
}

/// Slice index on a new lattice, in the same millimetres as the old one.
fn carried_cursor(
    previous: [usize; 3],
    from_origin: [f32; 3],
    from_voxel: f32,
    from_shape: [usize; 3],
    to_origin: [f32; 3],
    to_voxel: f32,
    to_shape: [usize; 3],
) -> [usize; 3] {
    let mut cursor = [0; 3];
    for axis in 0..3 {
        let next = to_shape[axis];
        if next == 0 {
            continue;
        }
        let last = next - 1;
        if from_shape[axis] == 0 || from_voxel <= 0.0 || to_voxel <= 0.0 {
            cursor[axis] = previous[axis].min(last);
            continue;
        }
        let mm = from_origin[axis] + (previous[axis] as f32 + 0.5) * from_voxel;
        let index = ((mm - to_origin[axis]) / to_voxel - 0.5).round();
        cursor[axis] = index.clamp(0.0, last as f32) as usize;
    }
    cursor
}

fn unit(value: [f32; 3]) -> [f32; 3] {
    let length = (value[0] * value[0] + value[1] * value[1] + value[2] * value[2])
        .sqrt()
        .max(1.0e-8);
    [value[0] / length, value[1] / length, value[2] / length]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blender_keys_match_the_turntable() {
        let azimuth = 30.0_f32.to_radians();
        let elevation = 25.0_f32.to_radians();
        let fov = scan_kit_core::FOV_Y;
        let (front, flat, same, _) =
            blender_step(azimuth, elevation, fov, fov, "1", false).unwrap();
        assert!((front - 180.0_f32.to_radians()).abs() < 1.0e-5);
        assert!(flat.abs() < 1.0e-5);
        assert!((same - fov).abs() < 1.0e-5);
        let (back, _, _, _) = blender_step(azimuth, elevation, fov, fov, "Digit1", true).unwrap();
        assert!(back.abs() < 1.0e-5);
        let (_, top, _, _) = blender_step(azimuth, elevation, fov, fov, "Home", false).unwrap();
        assert!((top - std::f32::consts::FRAC_PI_2).abs() < 1.0e-5);
        let (turned, _, _, _) = blender_step(0.0, 0.0, fov, fov, "ArrowLeft", false).unwrap();
        assert!((turned - ORBIT_STEP).abs() < 1.0e-5);
        let (_, _, ortho, saved) = blender_step(azimuth, elevation, fov, fov, "5", false).unwrap();
        assert!(ortho.abs() < 1.0e-6);
        assert!((saved - fov).abs() < 1.0e-5);
        let (_, _, restored, _) =
            blender_step(azimuth, elevation, 0.0, saved, "Numpad5", false).unwrap();
        assert!((restored - fov).abs() < 1.0e-5);
        assert!(blender_step(azimuth, elevation, fov, fov, "KeyA", false).is_none());
    }

    #[test]
    fn a_coarser_slice_lands_on_the_same_depth() {
        let cursor = carried_cursor(
            [4, 2, 10],
            [0.0, 0.0, -40.0],
            2.0,
            [8, 4, 20],
            [0.0, 0.0, -40.0],
            1.0,
            [16, 8, 40],
        );
        assert_eq!(cursor, [9, 5, 21]);
        assert_eq!(
            carried_cursor(
                [3, 1, 4],
                [0.0, 0.0, -8.0],
                1.0,
                [8, 4, 16],
                [0.0, 0.0, -8.0],
                1.0,
                [8, 4, 16],
            ),
            [3, 1, 4]
        );
        assert_eq!(
            carried_cursor(
                [100, 0, 0],
                [0.0; 3],
                1.0,
                [120, 4, 4],
                [0.0; 3],
                1.0,
                [8, 4, 4],
            )[0],
            7
        );
    }

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
            columns: 0,
            column_weights: Vec::new(),
            row_weights: Vec::new(),
            side: 0,
            row_splits: Vec::new(),
            volume: scan_kit_core::VolumeMark::default(),
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
    fn each_row_keeps_its_own_horizontal_split() {
        let mut scene = line_scene();
        let panel = scene.panels[0].clone();
        while scene.panels.len() < 6 {
            scene.panels.push(panel.clone());
        }
        scene.columns = 2;
        scene.row_weights = vec![1.4, 1.4, 1.0];
        scene.row_splits = vec![3.0, 1.0, 1.0, 1.0, 1.0, 3.0];
        let plot = Plot::new(&scene, [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]);
        let layout = plot.layout(600, 480);
        let top_gap = layout[0].cell.x + layout[0].cell.w;
        let bottom_gap = layout[4].cell.x + layout[4].cell.w;
        assert!(
            top_gap > bottom_gap + 40.0,
            "top {top_gap} bottom {bottom_gap}"
        );
        assert!(layout[0].cell.h > layout[4].cell.h);
    }

    #[test]
    fn dragging_a_gap_resizes_that_split_only() {
        let mut scene = line_scene();
        let panel = scene.panels[0].clone();
        while scene.panels.len() < 6 {
            scene.panels.push(panel.clone());
        }
        scene.columns = 2;
        scene.row_weights = vec![1.0, 1.0, 1.0];
        scene.row_splits = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let mut plot = Plot::new(&scene, [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]);
        let layout = plot.layout(600, 480);
        let vertical = layout[0].cell.y + layout[0].cell.h + 6.0;
        let before_bottom = plot.row_weights[2];
        plot.apply(
            600,
            480,
            &PlotInput {
                x: 200.0,
                y: vertical,
                dy: 30.0,
                drag: true,
                ..PlotInput::default()
            },
        );
        assert!(plot.row_weights[0] > 1.0);
        assert!((plot.row_weights[2] - before_bottom).abs() < 1.0e-4);
        plot.apply(600, 480, &PlotInput::default());
        let layout = plot.layout(600, 480);
        let gap_x = layout[2].cell.x + layout[2].cell.w + 4.0;
        let gap_y = layout[2].cell.y + layout[2].cell.h * 0.5;
        let other = plot.row_splits[0];
        plot.apply(
            600,
            480,
            &PlotInput {
                x: gap_x,
                y: gap_y,
                dx: 40.0,
                drag: true,
                ..PlotInput::default()
            },
        );
        assert!(plot.row_splits[2] > plot.row_splits[3]);
        assert!((plot.row_splits[0] - other).abs() < 1.0e-4);
    }

    #[test]
    fn paging_a_slice_does_not_upload_the_volume_again() {
        let mut scene = line_scene();
        let panel = scene.panels[0].clone();
        scene.panels.clear();
        for title in [
            "Axial",
            "3D",
            "Coronal",
            "Sagittal",
            "Depth dose",
            "Lateral profile",
        ] {
            let mut next = panel.clone();
            next.title = title.into();
            scene.panels.push(next);
        }
        scene.columns = 2;
        scene.row_weights = vec![1.0, 1.0, 1.0];
        scene.row_splits = vec![1.0; 6];
        scene.volume = scan_kit_core::VolumeMark {
            values: vec![2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            shape: [2, 2, 2],
            origin: [0.0, 0.0, 0.0],
            voxel: 1.0,
            ..scan_kit_core::VolumeMark::default()
        };
        let mut plot = Plot::new(&scene, [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]);
        let stamp = plot.atlas_stamp();
        let before = plot.cursor[2];
        let layout = plot.layout(600, 480);
        let cell = layout[0].cell;
        plot.apply(
            600,
            480,
            &PlotInput {
                x: cell.x + cell.w * 0.5,
                y: cell.y + cell.h * 0.5,
                wheel: 1.0,
                ..PlotInput::default()
            },
        );
        assert_eq!(plot.atlas_stamp(), stamp);
        assert!(plot.cursor[2] > before);
    }

    #[test]
    fn a_color_change_does_not_rebuild_the_volume() {
        let mut scene = line_scene();
        scene.panels[0].title = "Axial".into();
        scene.volume = scan_kit_core::VolumeMark {
            values: vec![1.0; 8],
            shape: [2, 2, 2],
            origin: [0.0, 0.0, 0.0],
            voxel: 1.0,
            ramp: 1,
            lo: 0.0,
            hi: 4.0,
            base_lo: 0.0,
            base_hi: 4.0,
            gain: 1.0,
            unit: "Gy·mm".into(),
            ..scan_kit_core::VolumeMark::default()
        };
        let mut plot = Plot::new(&scene, [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]);
        let stamp = plot.atlas_stamp();
        let ptr = plot.dose.as_ref().unwrap().volume.values.as_ptr();
        let before = ramp_corner(&plot);
        let window = plot.paint(r#"{"auto":"Off","level":"1.5","ray":"Maximum"}"#);
        let grid = plot.dose.as_ref().unwrap();
        assert_eq!(grid.volume.values.as_ptr(), ptr);
        assert_eq!(plot.atlas_stamp(), stamp);
        assert!((grid.hi - 1.5).abs() < 1e-4);
        assert!((grid.gain - 1.0).abs() < 1e-4);
        assert_eq!(grid.mode, 1);
        assert_eq!(grid.unit, "Gy");
        assert!(window.contains("Window"));
        assert!(window.contains("\"max\":\"8\""));
        assert_eq!(
            ramp_corner(&plot),
            before,
            "gain and window leave the ramp row"
        );
        plot.paint(r#"{"scale":"Viridis","auto":"On"}"#);
        assert_ne!(ramp_corner(&plot), before);
        assert_eq!(
            plot.dose.as_ref().unwrap().ramp,
            scan_kit_core::index("viridis")
        );
        assert_eq!(plot.atlas_stamp(), stamp);
        assert_eq!(plot.dose.as_ref().unwrap().volume.values.as_ptr(), ptr);
    }

    fn ramp_corner(plot: &Plot) -> [u8; 3] {
        let index = plot.dose_atlas.expect("dose atlas");
        let pixels = &plot.marks.heatmaps[index];
        let (cols, rows) = plot.marks.heatmap_size[index];
        let dst = (((rows - 1) * cols + 255) * 4) as usize;
        [pixels[dst], pixels[dst + 1], pixels[dst + 2]]
    }

    #[test]
    fn scrolling_down_on_the_volume_zooms_out() {
        let mut scene = line_scene();
        scene.panels[0].title = "3D".into();
        scene.volume = scan_kit_core::VolumeMark {
            values: vec![1.0],
            shape: [1, 1, 1],
            origin: [0.0, 0.0, 0.0],
            voxel: 1.0,
            ..scan_kit_core::VolumeMark::default()
        };
        let mut plot = Plot::new(&scene, [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]);
        let layout = plot.layout(600, 480);
        let picture = layout[0].plot;
        plot.apply(
            600,
            480,
            &PlotInput {
                x: picture.x + picture.w * 0.5,
                y: picture.y + picture.h * 0.5,
                wheel: 120.0,
                ..PlotInput::default()
            },
        );
        assert!(plot.orbit_zoom > 1.0);
    }

    #[test]
    fn the_field_box_shows_on_every_dose_view() {
        let mut scene = line_scene();
        let mut panel = scene.panels[0].clone();
        panel.series = vec![Series::Heatmap {
            values: vec![1.0; 64],
            cols: 8,
            rows: 8,
            ramp: 0,
            color: [1.0; 4],
            lo: 0.0,
            hi: 1.0,
        }];
        panel.xmin = 0.0;
        panel.xmax = 8.0;
        panel.ymin = 0.0;
        panel.ymax = 8.0;
        panel.equal = false;
        scene.panels = ["Axial", "Coronal", "Sagittal", "3D"]
            .into_iter()
            .map(|title| {
                let mut cell = panel.clone();
                cell.title = title.into();
                cell
            })
            .collect();
        scene.columns = 2;
        scene.volume = scan_kit_core::VolumeMark {
            values: vec![1.0; 512],
            shape: [8, 8, 8],
            origin: [0.0, 0.0, 0.0],
            voxel: 1.0,
            field: [1.0, 3.0, 0.0, 2.0, 4.0, 6.0],
            ..scan_kit_core::VolumeMark::default()
        };
        let plot = Plot::new(&scene, [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]);
        let expect = [
            (0usize, 1.0, 3.0, 0.0, 2.0),
            (1, 1.0, 3.0, 4.0, 6.0),
            (2, 0.0, 2.0, 4.0, 6.0),
        ];
        for (panel, x0, x1, y0, y1) in expect {
            let mut lines = Vec::new();
            for cloud in plot.live.iter().filter(|cloud| cloud.panel == panel) {
                for line in &cloud.lines {
                    if line.color[0] > 0.9 && line.color[1] > 0.6 && line.color[2] < 0.4 {
                        lines.push(line);
                    }
                }
            }
            assert_eq!(lines.len(), 4, "panel {panel}");
            let span = |axis: usize| {
                lines
                    .iter()
                    .flat_map(|line| [line.a[axis], line.b[axis]])
                    .fold((f32::MAX, f32::MIN), |(lo, hi), value| {
                        (lo.min(value), hi.max(value))
                    })
            };
            let (got_x0, got_x1) = span(0);
            let (got_y0, got_y1) = span(1);
            assert!(
                (got_x0 - x0).abs() < 1e-3 && (got_x1 - x1).abs() < 1e-3,
                "{panel}"
            );
            assert!(
                (got_y0 - y0).abs() < 1e-3 && (got_y1 - y1).abs() < 1e-3,
                "{panel}"
            );
        }
        let uniform = plot.dose_uniform(3, 1.0, 0.0);
        assert_eq!(uniform[8][3], 1.0);
        assert_eq!(uniform[8][..3], [1.0, 0.0, 4.0]);
        assert_eq!(uniform[9][..3], [3.0, 2.0, 6.0]);
    }

    fn box_edges(lo: [f32; 3], hi: [f32; 3]) -> [([f32; 3], [f32; 3]); 12] {
        let corner = |bits: i32| {
            [
                if (bits & 1) == 0 { lo[0] } else { hi[0] },
                if (bits & 2) == 0 { lo[1] } else { hi[1] },
                if (bits & 4) == 0 { lo[2] } else { hi[2] },
            ]
        };
        let mut edges = [([0.0; 3], [0.0; 3]); 12];
        let mut at = 0;
        for axis in 0..3 {
            let bit = 1 << axis;
            for i in 0..4 {
                let low = if axis == 0 {
                    (i % 2) * 2 + (i / 2) * 4
                } else if axis == 1 {
                    (i % 2) + (i / 2) * 4
                } else {
                    i
                };
                edges[at] = (corner(low), corner(low + bit));
                at += 1;
            }
        }
        edges
    }

    fn seg_dist(p: [f32; 2], a: [f32; 2], b: [f32; 2]) -> f32 {
        let ab = [b[0] - a[0], b[1] - a[1]];
        let len2 = ab[0] * ab[0] + ab[1] * ab[1];
        if len2 < 1.0e-6 {
            return (p[0] - a[0]).hypot(p[1] - a[1]);
        }
        let t = (((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1]) / len2).clamp(0.0, 1.0);
        let q = [a[0] + ab[0] * t, a[1] + ab[1] * t];
        (p[0] - q[0]).hypot(p[1] - q[1])
    }

    /// Pixels about a picture-pixel off the projected edge. That is where a stroke
    /// sized to the whole canvas used to disappear.
    fn wires_near(
        frame: &[u8],
        width: u32,
        plot: &PlotRect,
        edges: &[([f32; 2], [f32; 2])],
        avoid: &[([f32; 2], [f32; 2])],
        field: bool,
    ) -> (usize, [u8; 3]) {
        let height = frame.len() / 4 / width as usize;
        let mut hits = 0usize;
        let mut sample = [0u8; 3];
        for &(a, b) in edges {
            for step in [0.3_f32, 0.5, 0.7] {
                let mid = [a[0] + (b[0] - a[0]) * step, a[1] + (b[1] - a[1]) * step];
                let x0 = (mid[0].floor() as i32 - 3).max(0);
                let y0 = (mid[1].floor() as i32 - 3).max(0);
                let x1 = (mid[0].ceil() as i32 + 3).min(width as i32 - 1);
                let y1 = (mid[1].ceil() as i32 + 3).min(height as i32 - 1);
                for y in y0..=y1 {
                    for x in x0..=x1 {
                        let center = [x as f32 + 0.5, y as f32 + 0.5];
                        if center[0] < plot.x + 1.0 || center[0] > plot.x + plot.w - 1.0 {
                            continue;
                        }
                        if center[1] < plot.y + 1.0 || center[1] > plot.y + plot.h - 1.0 {
                            continue;
                        }
                        let dist = seg_dist(center, a, b);
                        if !(0.7..=1.2).contains(&dist) {
                            continue;
                        }
                        if field && avoid.iter().any(|&(c, d)| seg_dist(center, c, d) < 4.0) {
                            continue;
                        }
                        let index = (y as usize * width as usize + x as usize) * 4;
                        let rgb = [frame[index], frame[index + 1], frame[index + 2]];
                        sample = rgb;
                        let ink = if field {
                            rgb[0] > 190 && rgb[1] > 120 && rgb[2] < 100
                        } else {
                            rgb[1] > 100 && rgb[2] > 120 && rgb[0] < 190
                        };
                        if ink {
                            hits += 1;
                        }
                    }
                }
            }
        }
        (hits, sample)
    }

    #[test]
    fn the_3d_box_stays_visible_as_the_camera_moves() {
        let n = 8usize;
        let mut scene = dose_scene(n, vec![0.0; n * n * n], 1, 0);
        scene.volume.field = [2.0, 6.0, 2.0, 6.0, 2.0, 6.0];
        scene.volume.gantry = 90.0;
        let mut plot = Plot::new(&scene, [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]);
        let Ok(gpu) = crate::native_gpu() else {
            return;
        };
        let (width, height) = (960u32, 640u32);
        let lattice = box_edges([0.0; 3], [n as f32; 3]);
        let field = box_edges([2.0, 2.0, 2.0], [6.0, 6.0, 6.0]);
        let views = [
            (HOME_AZIMUTH, HOME_ELEVATION, 1.0_f32, "home"),
            (HOME_AZIMUTH + 0.9, HOME_ELEVATION - 0.2, 4.0, "zoomed out"),
            (0.15, 1.15, 0.4, "close"),
        ];
        for (azimuth, elevation, zoom, name) in views {
            plot.azimuth = azimuth;
            plot.elevation = elevation;
            plot.orbit_zoom = zoom;
            let layout = plot.layout(width, height);
            let cell = layout
                .iter()
                .find(|cell| plot.panels[cell.panel].title == "3D")
                .expect("3D cell");
            let old = 1.5 * cell.plot.h / height as f32;
            assert!(
                old < 0.65,
                "{name}: the 3D picture fills the frame, so this no longer tests a short cell"
            );
            let frame = plot
                .paint_offscreen(gpu, width, height)
                .expect("dose shader");
            let project = |point: [f32; 3]| {
                let grid = plot.dose.as_ref().expect("volume");
                let aspect = cell.plot.w / cell.plot.h.max(1.0);
                let uv = scan_kit_core::dose_film_open(
                    &grid.volume,
                    dose_ray(&plot, grid, aspect),
                    point,
                )?;
                Some([
                    cell.plot.x + uv[0] * cell.plot.w,
                    cell.plot.y + uv[1] * cell.plot.h,
                ])
            };
            let on_film = |edge: ([f32; 3], [f32; 3])| {
                let (a, b) = (project(edge.0)?, project(edge.1)?);
                ((a[0] - b[0]).hypot(a[1] - b[1]) >= 10.0).then_some((a, b))
            };
            let lattice_px: Vec<_> = lattice.iter().copied().filter_map(on_film).collect();
            let field_px: Vec<_> = field.iter().copied().filter_map(on_film).collect();
            let (gray, gray_at) = wires_near(&frame, width, &cell.plot, &lattice_px, &[], false);
            let (amber, amber_at) =
                wires_near(&frame, width, &cell.plot, &field_px, &lattice_px, true);
            assert!(
                gray >= 4,
                "{name}: bounding box {gray} px, sample {gray_at:?}"
            );
            assert!(
                amber >= 4,
                "{name}: field bounds {amber} px, sample {amber_at:?}"
            );
        }
    }

    #[test]
    fn the_sagittal_dose_sits_inside_its_field_box() {
        let (nx, ny, nz) = (3usize, 8usize, 16usize);
        let (hx, hy, hz) = (1usize, 6usize, 2usize);
        let mut values = vec![0.0f32; nx * ny * nz];
        values[hx + nx * (hy + ny * hz)] = 8.0;
        let heat = |cols: u32, rows: u32| Series::Heatmap {
            values: vec![0.0; (cols * rows) as usize],
            cols,
            rows,
            ramp: 1,
            color: [1.0; 4],
            lo: 0.0,
            hi: 8.0,
        };
        let mut scene = line_scene();
        let mut sagittal = scene.panels[0].clone();
        sagittal.title = "Sagittal".into();
        sagittal.xmin = 0.0;
        sagittal.xmax = ny as f32;
        sagittal.ymin = 0.0;
        sagittal.ymax = nz as f32;
        sagittal.series = vec![heat(ny as u32, nz as u32)];
        let mut coronal = sagittal.clone();
        coronal.title = "Coronal".into();
        coronal.xmin = 0.0;
        coronal.xmax = nx as f32;
        coronal.series = vec![heat(nx as u32, nz as u32)];
        scene.panels = vec![sagittal, coronal];
        scene.columns = 2;
        scene.volume = scan_kit_core::VolumeMark {
            values,
            shape: [nx as u32, ny as u32, nz as u32],
            origin: [0.0, 0.0, 0.0],
            voxel: 1.0,
            ramp: 1,
            hi: 8.0,
            field: [
                hx as f32,
                (hx + 1) as f32,
                hy as f32,
                (hy + 1) as f32,
                hz as f32,
                (hz + 1) as f32,
            ],
            ..scan_kit_core::VolumeMark::default()
        };
        let mut plot = Plot::new(&scene, [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]);
        let (width, height) = (480u32, 320u32);
        let layout = plot.layout(width, height);
        let cpu = plot.paint_cpu(width, height, &layout);
        let device = crate::native_gpu();
        let gpu = device.as_ref().map(|gpu| {
            plot.paint_offscreen(gpu, width, height)
                .expect("dose shader")
        });
        let hot = scan_kit_core::sample(1, 1.0).map(|channel| (channel * 255.0).round() as u8);
        let pixel = |frame: &[u8], x: f32, y: f32| {
            let x = x.round().clamp(0.0, width as f32 - 1.0) as usize;
            let y = y.round().clamp(0.0, height as f32 - 1.0) as usize;
            let index = (y * width as usize + x) * 4;
            [frame[index], frame[index + 1], frame[index + 2]]
        };
        let near = |got: [u8; 3], label: &str| {
            assert!(
                got.iter()
                    .zip(hot)
                    .all(|(channel, want)| (*channel as i32 - want as i32).abs() <= 8),
                "{label} {got:?} expect {hot:?}"
            );
        };
        // Off the crosshair, still inside the hot voxel and inside the field box.
        let sagittal_at = plot.project_data(0, hy as f32 + 0.7, hz as f32 + 0.2, width, height);
        near(pixel(&cpu, sagittal_at[0], sagittal_at[1]), "cpu sagittal");
        let coronal_at = plot.project_data(1, hx as f32 + 0.7, hz as f32 + 0.2, width, height);
        near(pixel(&cpu, coronal_at[0], coronal_at[1]), "cpu coronal");
        if let Ok(frame) = gpu {
            near(
                pixel(&frame, sagittal_at[0], sagittal_at[1]),
                "gpu sagittal",
            );
            near(pixel(&frame, coronal_at[0], coronal_at[1]), "gpu coronal");
        }
    }

    #[test]
    fn dose_profiles_overlay_each_session_color() {
        let mut scene = line_scene();
        scene.panels[0].title = "Depth dose".into();
        scene.panels[0].series.clear();
        scene.panels[0].y_label = "Gy".into();
        scene.volume = scan_kit_core::VolumeMark {
            values: vec![1.0, 4.0],
            shape: [1, 1, 2],
            origin: [0.0; 3],
            voxel: 1.0,
            session_colors: vec![[0.1, 0.2, 0.8, 1.0], [0.9, 0.4, 0.1, 1.0]],
            session_focus: 0,
            sessions: vec![
                scan_kit_core::SessionDose {
                    values: Vec::new(),
                    shape: [1, 1, 2],
                    origin: [0.0; 3],
                    voxel: 1.0,
                },
                scan_kit_core::SessionDose {
                    values: vec![9.0, 2.0],
                    shape: [1, 1, 2],
                    origin: [0.0; 3],
                    voxel: 1.0,
                },
            ],
            ..scan_kit_core::VolumeMark::default()
        };
        let plot = Plot::new(&scene, [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]);
        let lines: Vec<_> = plot
            .live
            .iter()
            .filter(|cloud| cloud.panel == 0)
            .flat_map(|cloud| cloud.lines.iter())
            .collect();
        assert_eq!(lines.len(), 2);
        let blue = lines.iter().find(|line| line.color[2] > 0.5).unwrap();
        let orange = lines.iter().find(|line| line.color[0] > 0.5).unwrap();
        assert!((blue.a[1].max(blue.b[1]) - 4.0).abs() < 1e-3);
        assert!((orange.a[1].max(orange.b[1]) - 9.0).abs() < 1e-3);
    }

    #[test]
    fn a_plan_is_dashed_in_the_session_color() {
        let mut scene = line_scene();
        scene.panels[0].title = "Depth dose".into();
        scene.panels[0].series.clear();
        scene.panels[0].y_label = "Gy".into();
        let color = [0.1, 0.2, 0.8, 1.0];
        scene.volume = scan_kit_core::VolumeMark {
            values: vec![1.0; 8],
            shape: [1, 1, 8],
            origin: [0.0; 3],
            voxel: 1.0,
            session_colors: vec![color],
            plans: vec![scan_kit_core::SessionDose {
                values: vec![2.0; 8],
                shape: [1, 1, 8],
                origin: [0.0; 3],
                voxel: 1.0,
            }],
            companions: vec![scan_kit_core::SessionDose {
                values: vec![3.0; 8],
                shape: [1, 1, 8],
                origin: [0.0; 3],
                voxel: 1.0,
            }],
            ..scan_kit_core::VolumeMark::default()
        };
        let plot = Plot::new(&scene, [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]);
        let lines: Vec<_> = plot
            .live
            .iter()
            .filter(|cloud| cloud.panel == 0)
            .flat_map(|cloud| cloud.lines.iter())
            .collect();
        let full = lines.iter().filter(|line| line.color[3] > 0.9).count();
        let faded = lines.iter().filter(|line| line.color[3] < 0.7).count();
        // 7 solid samples, and 5 of the 7 plan samples (3 on, 2 off).
        assert_eq!(full, 12);
        assert_eq!(faded, 7);
        assert!(lines.iter().all(|line| line.color[2] > 0.5));
        assert!(lines
            .iter()
            .all(|line| (line.color[0] - color[0]).abs() < 1e-3));
    }

    #[test]
    fn a_toolbar_hides_the_canvas_title() {
        let panel = Panel {
            title: "Depth dose".into(),
            y_label: "Gy".into(),
            x_label: String::new(),
            xmin: 0.0,
            xmax: 1.0,
            ymin: 0.0,
            ymax: 1.0,
            series: Vec::new(),
            x_labels: Vec::new(),
            equal: false,
        };
        let camera = Camera::new(0.0, 1.0, 0.0, 1.0);
        let cell = Cell {
            cell: PlotRect {
                x: 0.0,
                y: 0.0,
                w: 200.0,
                h: 160.0,
            },
            plot: PlotRect {
                x: 40.0,
                y: 28.0,
                w: 120.0,
                h: 100.0,
            },
            panel: 0,
        };
        let ink = [1.0, 1.0, 1.0, 1.0];
        let shown = labels_for(&panel, &camera, &cell, ink, true);
        let hidden = labels_for(&panel, &camera, &cell, ink, false);
        let title_y = cell.cell.y + atlas().ascent;
        let on_title = |glyphs: &[GlyphRec]| {
            glyphs
                .iter()
                .filter(|glyph| (glyph.anchor[1] - title_y).abs() < 0.5 && glyph.anchor[3] < 0.5)
                .count()
        };
        assert!(on_title(&shown) >= 8);
        assert_eq!(on_title(&hidden), 0);
        assert!(!hidden.is_empty(), "axis labels stay");
    }

    #[test]
    fn the_crosshair_sits_on_the_slice_picture() {
        let mut scene = line_scene();
        scene.panels.truncate(1);
        scene.panels[0].title = "Axial".into();
        scene.panels[0].equal = false;
        scene.panels[0].xmin = 0.0;
        scene.panels[0].xmax = 2.0;
        scene.panels[0].ymin = 0.0;
        scene.panels[0].ymax = 2.0;
        scene.panels[0].series = vec![
            Series::Heatmap {
                values: vec![0.0; 4],
                cols: 2,
                rows: 2,
                ramp: 0,
                color: [1.0; 4],
                lo: 0.0,
                hi: 1.0,
            },
            Series::Guide {
                xs: vec![0.0, 2.0, 2.0, 0.0, 0.0],
                ys: vec![0.0, 0.0, 2.0, 2.0, 0.0],
                color: [0.95, 0.72, 0.2, 0.9],
                thickness: 1.0,
            },
        ];
        scene.volume = scan_kit_core::VolumeMark {
            values: vec![1.0, 0.0, 0.0, 0.0],
            shape: [2, 2, 1],
            origin: [0.0, 0.0, 0.0],
            voxel: 1.0,
            ..scan_kit_core::VolumeMark::default()
        };
        let mut plot = Plot::new(&scene, [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]);
        let bounds = [0.0, 2.0, 0.0, 2.0];
        let volume = &plot.dose.as_ref().unwrap().volume;
        let x = volume.mm_of(0, plot.cursor[0]);
        let y = volume.mm_of(1, plot.cursor[1]);
        let expected = crate::dose::display_mm(x, y, bounds, 0);
        let cross = plot
            .live
            .iter()
            .find(|cloud| cloud.panel == 0 && cloud.lines.len() == 2)
            .unwrap();
        let hit = line_crossing(&cross.lines);
        assert!((hit[0] - expected[0]).abs() < 1e-3);
        assert!((hit[1] - expected[1]).abs() < 1e-3);
        assert!((hit[0] - x).abs() < 1e-3);
        let guide = plot
            .live
            .iter()
            .find(|cloud| cloud.panel == 0 && cloud.lines.len() > 2)
            .unwrap();
        let right = guide
            .lines
            .iter()
            .map(|line| line.a[0].max(line.b[0]))
            .fold(0.0, f32::max);
        assert!((right - 2.0 * crate::dose::FILM_X).abs() < 1e-3);

        let layout = plot.layout(200, 200);
        let cell = &layout[0];
        let matrix = plot
            .view_camera(0, &cell.plot)
            .clip_from_data(cell.plot, 200.0, 200.0);
        let data_x = 1.5;
        let data_y = 1.0;
        let ndc_x = matrix[0] * data_x + matrix[12];
        let ndc_y = matrix[5] * data_y + matrix[13];
        plot.apply(
            200,
            200,
            &PlotInput {
                x: (ndc_x + 1.0) * 0.5 * 200.0,
                y: (1.0 - ndc_y) * 0.5 * 200.0,
                drag: true,
                ..PlotInput::default()
            },
        );
        assert_eq!(plot.cursor[0], 1);

        let x = plot.dose.as_ref().unwrap().volume.mm_of(0, plot.cursor[0]);
        let y = plot.dose.as_ref().unwrap().volume.mm_of(1, plot.cursor[1]);
        plot.dose_action(0, "rotate");
        let expected = crate::dose::display_mm(x, y, bounds, 1);
        let cross = plot
            .live
            .iter()
            .find(|cloud| cloud.panel == 0 && cloud.lines.len() == 2)
            .unwrap();
        let hit = line_crossing(&cross.lines);
        assert!((hit[0] - expected[0]).abs() < 1e-3);
        assert!((hit[1] - expected[1]).abs() < 1e-3);
    }

    fn pixel_rgb(frame: &[u8], width: u32, x: i32, y: i32) -> [u8; 3] {
        if x < 0 || y < 0 {
            return [0; 3];
        }
        let index = ((y as u32 * width) + x as u32) as usize * 4;
        if index + 2 >= frame.len() {
            return [0; 3];
        }
        [frame[index], frame[index + 1], frame[index + 2]]
    }

    #[test]
    fn a_wide_slice_paints_zero_and_the_crosshair_past_the_volume() {
        let mut scene = line_scene();
        scene.panels[0].title = "Axial".into();
        scene.panels[0].equal = false;
        scene.panels[0].xmin = 0.0;
        scene.panels[0].xmax = 4.0;
        scene.panels[0].ymin = 0.0;
        scene.panels[0].ymax = 4.0;
        scene.panels[0].series = vec![Series::Heatmap {
            values: vec![0.0; 16],
            cols: 4,
            rows: 4,
            ramp: 1,
            color: [1.0; 4],
            lo: 0.0,
            hi: 4.0,
        }];
        scene.volume = scan_kit_core::VolumeMark {
            values: vec![4.0; 16],
            shape: [4, 4, 1],
            origin: [0.0, 0.0, 0.0],
            voxel: 1.0,
            ramp: 1,
            lo: 0.0,
            hi: 4.0,
            mode: 0,
            ..scan_kit_core::VolumeMark::default()
        };
        let mut plot = Plot::new(&scene, [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]);
        let dose = plot
            .marks
            .quads
            .iter()
            .find(|quad| quad.heat > 1.5 && quad.heat < 2.5)
            .unwrap();
        assert!((dose.a[0]).abs() < 1e-3 && (dose.b[0] - 4.0).abs() < 1e-3);
        let layout = plot.layout(480, 160);
        let camera = plot.view_camera(0, &layout[0].plot);
        assert!(
            camera.xmin < -0.5,
            "the window opens past the volume, xmin {}",
            camera.xmin
        );
        let cpu = plot.paint_cpu(480, 160, &layout);
        let cross = plot.live.iter().find(|cloud| cloud.crosshair).unwrap();
        let horizontal = cross
            .lines
            .iter()
            .find(|line| (line.a[1] - line.b[1]).abs() < 1e-3)
            .unwrap();
        assert!(horizontal.a[0] <= camera.xmin + 1e-3);
        assert!(horizontal.b[0] >= camera.xmax - 1e-3);
        let margin_x = camera.xmin * 0.5;
        let quiet = plot.project_data(0, margin_x, 3.0, 480, 160);
        let zero = scan_kit_core::sample(1, 0.0).map(|channel| (channel * 255.0).round() as u8);
        let got = pixel_rgb(&cpu, 480, quiet[0] as i32, quiet[1] as i32);
        assert!(
            got.iter()
                .zip(zero)
                .all(|(got, want)| (*got as i32 - want as i32).abs() <= 2),
            "margin {got:?} zero {zero:?}"
        );
        let inside = plot.project_data(0, 2.0, 2.0, 480, 160);
        let hot = pixel_rgb(&cpu, 480, inside[0] as i32, inside[1] as i32);
        assert!(
            hot.iter()
                .zip(zero)
                .any(|(got, want)| (*got as i32 - want as i32).abs() > 20),
            "the volume stays on its bounds, pixel {hot:?}"
        );
        let on_line = plot.project_data(0, margin_x, horizontal.a[1], 480, 160);
        let mut brightest = 0i32;
        for dy in -2..=2 {
            let rgb = pixel_rgb(&cpu, 480, on_line[0] as i32, on_line[1] as i32 + dy);
            brightest = brightest.max(rgb.iter().map(|channel| *channel as i32).sum());
        }
        let quiet_sum: i32 = got.iter().map(|channel| *channel as i32).sum();
        assert!(
            brightest > quiet_sum + 80,
            "crosshair sum {brightest} beside zero sum {quiet_sum}"
        );
    }

    #[test]
    fn an_integrated_profile_stays_in_the_frame() {
        let mut scene = line_scene();
        scene.panels.truncate(1);
        scene.panels[0].title = "Depth dose".into();
        scene.panels[0].y_label = "Gy".into();
        scene.panels[0].x_label = String::new();
        scene.panels[0].series = vec![Series::Polyline {
            xs: vec![0.0, 1.0],
            ys: vec![1.0, 4.0],
            color: [1.0; 4],
            thickness: 1.5,
        }];
        scene.panels[0].xmin = 0.0;
        scene.panels[0].xmax = 1.0;
        scene.panels[0].ymin = 1.0;
        scene.panels[0].ymax = 4.0;
        scene.volume = scan_kit_core::VolumeMark {
            values: vec![4.0, 0.0, 1.0, 0.0],
            shape: [2, 1, 2],
            origin: [0.0, 0.0, 0.0],
            voxel: 2.0,
            ..scan_kit_core::VolumeMark::default()
        };
        let mut plot = Plot::new(&scene, [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]);
        plot.cameras[0].ymin = 2.0;
        plot.cameras[0].ymax = 3.5;
        let before = plot.cameras[0].ymax;
        plot.dose_action(0, "integral");
        let top = plot
            .live
            .iter()
            .find(|cloud| cloud.panel == 0)
            .unwrap()
            .lines
            .iter()
            .map(|line| line.a[1].max(line.b[1]))
            .fold(f32::MIN, f32::max);
        assert!(top > before);
        assert!(plot.cameras[0].ymax + 1e-3 >= top);
        assert!(plot.cameras[0].ymin - 1e-3 <= 4.0);
        assert_eq!(plot.panels[0].y_label, "Gy·mm²");
        let mut again = Plot::new(&scene, [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]);
        again.adopt_view(&plot);
        again.dose_action(0, "integral");
        assert!((again.cameras[0].ymax - before).abs() < 1e-3);
        assert!((again.cameras[0].ymin - 2.0).abs() < 1e-3);
        assert_eq!(again.panels[0].y_label, "Gy");
        let mut wide = scene.clone();
        wide.panels[0].ymin = 0.0;
        wide.panels[0].ymax = 80.0;
        let mut reloaded = Plot::new(&wide, [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]);
        reloaded.adopt_view(&plot);
        assert_eq!(reloaded.panels[0].y_label, "Gy·mm²");
        reloaded.dose_action(0, "integral");
        assert!((reloaded.cameras[0].ymax - 80.0).abs() < 1e-3);
        assert!((reloaded.cameras[0].ymin).abs() < 1e-3);
        assert_eq!(reloaded.panels[0].y_label, "Gy");
        plot.dose_action(0, "integral");
        assert!((plot.cameras[0].ymax - before).abs() < 1e-3);
        assert!((plot.cameras[0].ymin - 2.0).abs() < 1e-3);
        assert_eq!(plot.panels[0].y_label, "Gy");
    }

    fn line_crossing(lines: &[LineRec]) -> [f32; 2] {
        let vertical = lines
            .iter()
            .find(|line| (line.a[0] - line.b[0]).abs() < 1e-3)
            .unwrap();
        let horizontal = lines
            .iter()
            .find(|line| (line.a[1] - line.b[1]).abs() < 1e-3)
            .unwrap();
        [vertical.a[0], horizontal.a[1]]
    }

    #[test]
    fn the_3d_cell_paints_the_dose_box() {
        let n = 16usize;
        let mut values = vec![0.0f32; n * n * n];
        for z in 5..11 {
            for y in 5..11 {
                for x in 5..11 {
                    values[x + n * (y + n * z)] = 4.0;
                }
            }
        }
        let mut scene = line_scene();
        scene.panels[0].title = "3D".into();
        scene.panels[0].equal = true;
        scene.panels[0].xmin = 0.0;
        scene.panels[0].xmax = n as f32;
        scene.panels[0].ymin = 0.0;
        scene.panels[0].ymax = n as f32;
        scene.panels[0].series = vec![Series::Heatmap {
            values: vec![0.0; n * n],
            cols: n as u32,
            rows: n as u32,
            ramp: 1,
            color: [1.0; 4],
            lo: 0.0,
            hi: 4.0,
        }];
        scene.volume = scan_kit_core::VolumeMark {
            values,
            shape: [n as u32, n as u32, n as u32],
            origin: [0.0, 0.0, 0.0],
            voxel: 1.0,
            ramp: 1,
            hi: 4.0,
            mode: 1,
            ..scan_kit_core::VolumeMark::default()
        };
        let expect = scan_kit_core::sample(1, 1.0).map(|channel| (channel * 255.0).round() as u8);
        for filter in [0u8, 1] {
            scene.volume.filter = filter;
            let mut plot = Plot::new(&scene, [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]);
            let layout = plot.layout(320, 320);
            let rect = layout[0].plot;
            let x = (rect.x + rect.w * 0.5) as u32;
            let y = (rect.y + rect.h * 0.5) as u32;
            let cpu = plot.paint_cpu(320, 320, &layout);
            let cpu_px = &cpu[((y * 320 + x) * 4) as usize..][..3];
            assert!(
                cpu_px
                    .iter()
                    .zip(expect)
                    .all(|(got, want)| (*got as i32 - want as i32).abs() <= 2),
                "cpu filter {filter} {cpu_px:?} expect {expect:?}"
            );
            let frame =
                render_plot(&scene, 320, 320, [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]).unwrap();
            if frame.is_empty() {
                return;
            }
            let gpu_px = &frame[((y * 320 + x) * 4) as usize..][..3];
            assert!(
                gpu_px
                    .iter()
                    .zip(expect)
                    .all(|(got, want)| (*got as i32 - want as i32).abs() <= 2),
                "gpu filter {filter} {gpu_px:?} expect {expect:?}"
            );
        }
    }

    #[test]
    fn the_3d_axes_hang_readable_titles() {
        let short = axis_ticks(0.0, 4.0);
        assert_eq!(short, vec![(0.0, true), (4.0, true)]);
        let partial = axis_ticks(0.0, 16.0);
        assert_eq!(partial.last().map(|tick| tick.0), Some(16.0));
        assert!(partial.last().is_some_and(|tick| tick.1));
        let century = axis_ticks(0.0, 100.0);
        assert_eq!(century.first().map(|tick| tick.0), Some(0.0));
        assert_eq!(century.last().map(|tick| tick.0), Some(100.0));
        let numbered = century.iter().filter(|tick| tick.1).count();
        assert!(numbered <= 8, "{numbered} numbered ticks");
        assert!(century.iter().any(|tick| !tick.1), "minor ticks");

        let n = 8usize;
        let mut scene = line_scene();
        scene.panels[0].title = "3D".into();
        scene.panels[0].series.clear();
        scene.volume = scan_kit_core::VolumeMark {
            values: vec![1.0; n * n * n],
            shape: [n as u32, n as u32, n as u32],
            origin: [0.0, 0.0, 0.0],
            voxel: 10.0,
            ramp: 1,
            hi: 1.0,
            mode: 1,
            ..scan_kit_core::VolumeMark::default()
        };
        let plot = Plot::new(&scene, [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]);
        let layout = plot.layout(360, 360);
        let cell = &layout[0];
        let labels = volume_labels(&plot, cell);
        let titles: Vec<_> = labels
            .iter()
            .filter(|label| label.title)
            .map(|label| label.text.as_str())
            .collect();
        assert_eq!(titles, ["X (mm)", "Y (mm)", "Depth (mm)"]);
        assert!(
            labels.iter().any(|label| label.angle.abs() > 0.05),
            "titles follow the edge"
        );
        for title in labels.iter().filter(|label| label.title) {
            assert!(
                title.at[0] > cell.plot.x && title.at[0] < cell.plot.x + cell.plot.w,
                "{} left the film",
                title.text
            );
        }
        assert!(
            labels.iter().any(|label| !label.title && label.text == "0"),
            "an edge starts at 0"
        );
        assert!(
            labels
                .iter()
                .any(|label| !label.title && label.text == "80"),
            "an edge names its far end"
        );
        let grid = plot.dose.as_ref().expect("volume");
        let view = dose_ray(&plot, grid, cell.plot.w / cell.plot.h.max(1.0));
        let corner = scan_kit_core::dose_film_open(&grid.volume, view, [0.0, 0.0, 80.0])
            .expect("entrance corner");
        let corner = [
            cell.plot.x + corner[0] * cell.plot.w,
            cell.plot.y + corner[1] * cell.plot.h,
        ];
        let hung = labels.iter().any(|label| {
            !label.title
                && label.text == "0"
                && (label.at[0] - corner[0]).hypot(label.at[1] - corner[1]) > 10.0
        });
        assert!(hung, "numbers hang off the tick");
        let glyphs = dose_chrome(&plot, cell);
        assert!(glyphs.iter().all(|glyph| glyph.anchor[3] < -0.5));
        let bits = plot.dose_uniform(0, cell.plot.w / cell.plot.h.max(1.0), 0.0)[6][3];
        let sides = axis_sides(&grid.volume, view);
        let packed = (sides[0] + sides[1] * 2 + sides[2] * 4) as f32;
        assert_eq!(bits, packed, "the march ticks use the same outward side");
    }

    #[test]
    fn dose_views_fill_the_cell_and_the_color_axis_is_numbered() {
        let (majors, _, labels, offset) = color_axis_ticks(0.0, 4.0, 6);
        assert!((majors[0]).abs() < 1e-4);
        assert!((majors[majors.len() - 1] - 4.0).abs() < 1e-4);
        assert_eq!(labels[0], "0");
        assert_eq!(labels[labels.len() - 1], "4");
        assert!(offset.is_empty(), "{offset}");
        let (crossed, _, _, _) = color_axis_ticks(-1.0, 1.0, 6);
        assert!(crossed.iter().any(|value| value.abs() < 1e-4));
        let (_, _, _, offset) = color_axis_ticks(0.0, 1.0e-6, 4);
        assert!(offset.contains('⁻'), "{offset}");
        let (_, _, labels, _) = color_axis_ticks(-2.0, 0.0, 4);
        assert!(labels.iter().any(|label| label.contains('−')), "{labels:?}");

        let mut scene = line_scene();
        scene.panels[0].title = "Axial".into();
        scene.panels[0].equal = true;
        scene.panels[0].x_label = "X (mm)".into();
        scene.panels[0].y_label = "Y (mm)".into();
        let mut axial_plot = Plot::new(&scene, [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]);
        let axial = axial_plot.layout(400, 160);
        let cell = &axial[0];
        assert!(
            cell.plot.w > cell.plot.h * 1.3,
            "a wide slice fills the cell, plot {}x{}",
            cell.plot.w,
            cell.plot.h
        );
        assert!(cell.plot.x > cell.cell.x + 8.0, "y ticks sit on the border");
        let origin = axial_plot.project_data(0, 0.0, 0.0, 400, 160);
        let east = axial_plot.project_data(0, 1.0, 0.0, 400, 160);
        let north = axial_plot.project_data(0, 0.0, 1.0, 400, 160);
        let dx = (east[0] - origin[0]).abs();
        let dy = (origin[1] - north[1]).abs();
        assert!(
            (dx - dy).abs() < 0.05,
            "a millimetre is {dx} px across and {dy} px up"
        );
        axial_plot.cameras[0] = scan_kit_core::Camera::new(0.0, 20.0, 0.0, 10.0);
        axial_plot.turns[0] = 1;
        let bounds = [0.0, 20.0, 0.0, 10.0];
        let here = crate::dose::display_mm(5.0, 5.0, bounds, 1);
        let along = crate::dose::display_mm(6.0, 5.0, bounds, 1);
        let across = crate::dose::display_mm(5.0, 6.0, bounds, 1);
        let p0 = axial_plot.project_data(0, here[0], here[1], 400, 160);
        let p1 = axial_plot.project_data(0, along[0], along[1], 400, 160);
        let p2 = axial_plot.project_data(0, across[0], across[1], 400, 160);
        let turned_x = (p1[0] - p0[0]).hypot(p1[1] - p0[1]);
        let turned_y = (p2[0] - p0[0]).hypot(p2[1] - p0[1]);
        assert!(
            (turned_x - turned_y).abs() < 0.05,
            "a turned millimetre is {turned_x} px by {turned_y} px"
        );

        scene.panels[0].title = "3D".into();
        scene.panels[0].x_label.clear();
        scene.panels[0].y_label.clear();
        let volume = Plot::new(&scene, [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]).layout(400, 160);
        let cell = &volume[0];
        assert!(cell.plot.x < cell.cell.x + 12.0, "3D has no y axis gutter");
        assert!(cell.plot.w > cell.cell.w - color_column_width() - 16.0);

        let mut band = Plot::new(&scene, [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]);
        let open = band.layout(400, 160);
        band.set_chrome(40.0);
        let held = band.layout(400, 160);
        assert!(
            held[0].plot.y >= open[0].cell.y + 40.0,
            "the picture starts under the toolbar"
        );

        let n = 4usize;
        scene.panels[0].series = vec![Series::Heatmap {
            values: vec![0.0; n * n],
            cols: n as u32,
            rows: n as u32,
            ramp: 1,
            color: [1.0; 4],
            lo: 0.0,
            hi: 4.0,
        }];
        scene.volume = scan_kit_core::VolumeMark {
            values: vec![1.0; n * n * n],
            shape: [n as u32, n as u32, n as u32],
            origin: [0.0, 0.0, 0.0],
            voxel: 1.0,
            ramp: 1,
            hi: 4.0,
            mode: 1,
            unit: "Gy".into(),
            ..scan_kit_core::VolumeMark::default()
        };
        let mut plot = Plot::new(&scene, [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]);
        let layout = plot.layout(160, 160);
        let glyphs = color_axis_glyphs(&plot, &layout[0], 0.0);
        assert!(glyphs.len() > 4, "tick labels");
        assert!(
            glyphs.iter().any(|glyph| glyph.anchor[2] > 0.5),
            "unit title"
        );
        assert!(glyphs
            .iter()
            .any(|glyph| glyph.anchor[0] > layout[0].plot.x + layout[0].plot.w));
        let cpu = plot.paint_cpu(160, 160, &layout);
        let bar = color_bar_rect(&layout[0]);
        let x = (bar.x + 2.0) as u32;
        let y = (bar.y + 2.0) as u32;
        let t = (1.0 - ((y as f32 + 0.5 - bar.y) / bar.h)).clamp(0.0, 1.0);
        let expect = scan_kit_core::sample(1, t).map(|channel| (channel * 255.0).round() as u8);
        let got = &cpu[((y * 160 + x) * 4) as usize..][..3];
        assert!(
            got.iter()
                .zip(expect)
                .all(|(got, want)| (*got as i32 - want as i32).abs() <= 2),
            "cpu bar {got:?} expect {expect:?}"
        );
        let frame =
            render_plot(&scene, 160, 160, [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]).unwrap();
        if frame.is_empty() {
            return;
        }
        let gpu = &frame[((y * 160 + x) * 4) as usize..][..3];
        assert!(
            gpu.iter()
                .zip(expect)
                .all(|(got, want)| (*got as i32 - want as i32).abs() <= 2),
            "gpu bar {gpu:?} expect {expect:?}"
        );
    }

    #[test]
    fn the_integrate_peak_reaches_the_top_of_the_ramp() {
        let n = 16usize;
        let mut values = vec![0.0f32; n * n * n];
        for z in 0..n {
            let depth = z as f32 / n as f32;
            let dose = 0.35 + 0.65 * (-((depth - 0.8) / 0.08).powi(2)).exp();
            for y in 0..n {
                for x in 0..n {
                    let dx = x as f32 / n as f32 - 0.5;
                    let dy = y as f32 / n as f32 - 0.5;
                    values[x + n * (y + n * z)] = dose * (-(dx * dx + dy * dy) / 0.02).exp();
                }
            }
        }
        let peak = values.iter().copied().fold(0.0f32, f32::max);
        let mut scene = line_scene();
        scene.panels[0].title = "3D".into();
        scene.panels[0].equal = true;
        scene.panels[0].xmin = 0.0;
        scene.panels[0].xmax = n as f32;
        scene.panels[0].ymin = 0.0;
        scene.panels[0].ymax = n as f32;
        scene.panels[0].series = vec![Series::Heatmap {
            values: vec![0.0; n * n],
            cols: n as u32,
            rows: n as u32,
            ramp: 1,
            color: [1.0; 4],
            lo: 0.0,
            hi: peak,
        }];
        scene.volume = scan_kit_core::VolumeMark {
            values,
            shape: [n as u32, n as u32, n as u32],
            voxel: 1.0,
            ramp: 1,
            hi: peak,
            mode: 0,
            filter: 1,
            ..scan_kit_core::VolumeMark::default()
        };
        let top = scan_kit_core::sample(1, 1.0).map(|channel| (channel * 255.0).round() as u8);
        let mut plot = Plot::new(&scene, [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]);
        let layout = plot.layout(96, 96);
        let rect = layout[0].plot;
        let cpu = plot.paint_cpu(96, 96, &layout);
        assert!(
            ramp_top_is_in_the_cloud(&cpu, 96, rect, top),
            "cpu missed the top of the ramp"
        );
        let frame =
            render_plot(&scene, 96, 96, [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]).unwrap();
        if frame.is_empty() {
            return;
        }
        assert!(
            ramp_top_is_in_the_cloud(&frame, 96, rect, top),
            "gpu missed the top of the ramp"
        );
    }

    #[test]
    fn a_solid_cube_fills_its_projection() {
        let n = 16usize;
        let mut scene = line_scene();
        scene.panels[0].title = "3D".into();
        scene.panels[0].equal = true;
        scene.panels[0].xmin = 0.0;
        scene.panels[0].xmax = n as f32;
        scene.panels[0].ymin = 0.0;
        scene.panels[0].ymax = n as f32;
        scene.panels[0].series = vec![Series::Heatmap {
            values: vec![0.0; n * n],
            cols: n as u32,
            rows: n as u32,
            ramp: 1,
            color: [1.0; 4],
            lo: 0.0,
            hi: 1.0,
        }];
        scene.volume = scan_kit_core::VolumeMark {
            values: vec![1.0; n * n * n],
            shape: [n as u32, n as u32, n as u32],
            voxel: 1.0,
            ramp: 1,
            hi: 1.0,
            mode: 0,
            filter: 1,
            ..scan_kit_core::VolumeMark::default()
        };
        let mut plot = Plot::new(&scene, [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]);
        let layout = plot.layout(96, 96);
        let rect = layout[0].plot;
        let cpu = plot.paint_cpu(96, 96, &layout);
        let cpu_fill = projection_fill(&cpu, 96, rect);
        assert!(cpu_fill > 0.5, "cpu projection fill {cpu_fill}");
        let frame =
            render_plot(&scene, 96, 96, [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]).unwrap();
        if frame.is_empty() {
            return;
        }
        let gpu_fill = projection_fill(&frame, 96, rect);
        assert!(gpu_fill > 0.5, "gpu projection fill {gpu_fill}");
    }

    #[test]
    fn half_float_keeps_the_dose_values_we_paint() {
        assert_eq!(f32_to_f16(0.0), 0);
        assert_eq!(f32_to_f16(1.0), 0x3c00);
        assert_eq!(f32_to_f16(-1.0), 0xbc00);
        assert_eq!(f32_to_f16(0.5), 0x3800);
        assert_eq!(f32_to_f16(2.0), 0x4000);
        assert_eq!(f32_to_f16(4.0), 0x4400);
    }

    #[test]
    fn a_side_panel_spans_the_column_beside_it() {
        let mut scene = line_scene();
        let panel = scene.panels[0].clone();
        scene.panels.push(panel.clone());
        scene.panels.push(panel);
        scene.columns = 2;
        scene.column_weights = vec![3.0, 1.4];
        scene.side = 1;
        let plot = Plot::new(&scene, [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]);
        let layout = plot.layout(400, 420);
        let top = &layout[0].cell;
        let lower = &layout[1].cell;
        let side = &layout[2].cell;
        assert!((top.y - side.y).abs() < 1.0, "{} {}", top.y, side.y);
        let left_bottom = lower.y + lower.h;
        let side_bottom = side.y + side.h;
        assert!(
            (left_bottom - side_bottom).abs() < 1.0,
            "{left_bottom} {side_bottom}"
        );
        assert!(side.h > top.h, "{} {}", side.h, top.h);
        assert!(side.x > top.x);
    }

    #[test]
    fn a_paired_grid_keeps_a_side_column() {
        let mut scene = line_scene();
        let panel = scene.panels[0].clone();
        scene.panels.push(panel.clone());
        scene.panels.push(panel.clone());
        scene.panels.push(panel.clone());
        scene.panels.push(panel);
        scene.columns = 2;
        scene.column_weights = vec![3.0, 1.4, 1.4];
        scene.side = 1;
        let plot = Plot::new(&scene, [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]);
        let layout = plot.layout(800, 420);
        let time = &layout[0].cell;
        let spectrum = &layout[1].cell;
        let time_below = &layout[2].cell;
        let side = &layout[4].cell;
        assert!((time.y - spectrum.y).abs() < 1.0);
        assert!(spectrum.x > time.x);
        assert!(time_below.y > time.y);
        assert!((time.y - side.y).abs() < 1.0);
        let left_bottom = time_below.y + time_below.h;
        assert!((left_bottom - (side.y + side.h)).abs() < 1.0);
        assert!(side.x > spectrum.x);
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
            columns: 0,
            column_weights: Vec::new(),
            row_weights: Vec::new(),
            side: 0,
            row_splits: Vec::new(),
            volume: scan_kit_core::VolumeMark::default(),
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

    fn time_panel(x_label: &str, xmin: f32, xmax: f32, xs: Vec<f32>, ys: Vec<f32>) -> Panel {
        Panel {
            title: String::new(),
            y_label: String::new(),
            x_label: x_label.into(),
            xmin,
            xmax,
            ymin: 0.0,
            ymax: 10.0,
            series: vec![Series::Polyline {
                xs,
                ys,
                color: [1.0, 0.0, 0.0, 1.0],
                thickness: 2.0,
            }],
            x_labels: Vec::new(),
            equal: false,
        }
    }

    #[test]
    fn follow_time_slides_time_panels_and_leaves_the_rest() {
        let scene = PlotScene {
            title: "replay".into(),
            panels: vec![
                time_panel(
                    "",
                    0.0,
                    10.0,
                    vec![0.0, 1.0, 2.0, 3.0],
                    vec![1.0, 1.0, 100.0, 1.0],
                ),
                time_panel("Hz", 1.0, 500.0, vec![10.0, 20.0], vec![0.0, 1.0]),
                time_panel("mm", 2.0, 8.0, vec![2.0, 4.0], vec![3.0, 4.0]),
            ],
            controls: Vec::new(),
            table: None,
            columns: 2,
            column_weights: vec![3.0, 1.4],
            row_weights: Vec::new(),
            side: 1,
            row_splits: Vec::new(),
            volume: scan_kit_core::VolumeMark::default(),
        };
        let mut plot = Plot::new(&scene, [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]);
        let lines = plot.marks.lines.len();
        let spectrum = plot.cameras[1];
        let side = plot.cameras[2];
        plot.follow_time(true, 0.0, 1.5, true);
        let time = plot.cameras[0];
        assert!(time.xmin >= 0.0 && time.xmin < 0.01, "{}", time.xmin);
        assert!((time.xmax - 1.02).abs() < 1.0e-4, "{}", time.xmax);
        assert!(
            time.ymax < 2.0,
            "the spike outside the window stays offscreen {}",
            time.ymax
        );
        assert_eq!(plot.cameras[1], spectrum);
        assert_eq!(plot.cameras[2], side);
        assert_eq!(plot.marks.lines.len(), lines);
        plot.zoom(0, 0.5, 1.0, 2.0);
        let zoomed = plot.cameras[0];
        plot.follow_time(true, 2.0, 3.0, false);
        assert_eq!(plot.cameras[0], zoomed);
        plot.follow_time(false, 0.0, 0.0, true);
        assert!(plot.cameras[0].xmax > 3.0, "{}", plot.cameras[0].xmax);
        assert!(plot.cameras[0].ymax > 50.0, "{}", plot.cameras[0].ymax);
        assert_eq!(plot.cameras[1], spectrum);
        assert_eq!(plot.marks.lines.len(), lines);
    }

    #[test]
    fn follow_time_hides_scatter_points_outside_the_playhead() {
        let scene = PlotScene {
            title: "replay".into(),
            panels: vec![
                time_panel(
                    "",
                    0.0,
                    4.0,
                    vec![0.0, 1.0, 2.0, 3.0],
                    vec![1.0, 1.0, 1.0, 1.0],
                ),
                Panel {
                    title: String::new(),
                    y_label: String::new(),
                    x_label: "mm".into(),
                    xmin: -1.0,
                    xmax: 4.0,
                    ymin: -1.0,
                    ymax: 4.0,
                    series: vec![
                        Series::Points {
                            xs: vec![0.0, 2.0],
                            ys: vec![0.0, 0.0],
                            color: [0.1, 0.4, 0.9, 1.0],
                            radius: 4.0,
                            times: vec![0.0, 2.0],
                        },
                        Series::Points {
                            xs: vec![1.0],
                            ys: vec![1.0],
                            color: [0.9, 0.2, 0.1, 1.0],
                            radius: 4.0,
                            times: Vec::new(),
                        },
                    ],
                    x_labels: Vec::new(),
                    equal: false,
                },
            ],
            controls: Vec::new(),
            table: None,
            columns: 2,
            column_weights: vec![3.0, 1.4],
            row_weights: Vec::new(),
            side: 1,
            row_splits: Vec::new(),
            volume: scan_kit_core::VolumeMark::default(),
        };
        let mut plot = Plot::new(&scene, [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]);
        plot.apply(320, 180, &PlotInput::default());
        let at = |plot: &Plot, x: f32, y: f32| {
            let px = plot.project_data(1, x, y, 320, 180);
            plot.hover(px[0], px[1]).3
        };
        plot.follow_time(true, 0.0, 0.5, true);
        assert_eq!(at(&plot, 0.0, 0.0), Some(1));
        assert_eq!(at(&plot, 2.0, 0.0), None);
        assert_eq!(at(&plot, 1.0, 1.0), Some(2), "a point without a time stays");
        plot.follow_time(true, 1.5, 2.5, true);
        assert_eq!(at(&plot, 0.0, 0.0), None);
        assert_eq!(at(&plot, 2.0, 0.0), Some(1));
        plot.follow_time(false, 0.0, 0.0, true);
        assert_eq!(at(&plot, 0.0, 0.0), Some(1));
        assert_eq!(at(&plot, 2.0, 0.0), Some(1));
        let draws = visible_draws(
            &plot.marks.points,
            0,
            plot.marks.points.len() as u32,
            &plot.marks.panels[1].point_runs,
            Some((1.5, 2.5)),
        );
        assert_eq!(draws, vec![(1, 1), (2, 1)]);
    }

    #[test]
    fn follow_time_counts_the_visible_density_without_replacing_points() {
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        let mut times = Vec::new();
        for _ in 0..4 {
            xs.push(0.25);
            ys.push(0.25);
            times.push(0.0);
            xs.push(1.25);
            ys.push(1.25);
            times.push(2.0);
        }
        let scene = PlotScene {
            title: "density".into(),
            panels: vec![Panel {
                title: String::new(),
                y_label: String::new(),
                x_label: "x".into(),
                xmin: 0.0,
                xmax: 2.0,
                ymin: 0.0,
                ymax: 2.0,
                series: vec![
                    Series::Heatmap {
                        values: vec![0.0; 4],
                        cols: 2,
                        rows: 2,
                        ramp: 0,
                        color: [1.0, 1.0, 1.0, 1.0],
                        lo: 0.0,
                        hi: 0.0,
                    },
                    Series::Points {
                        xs: vec![0.5],
                        ys: vec![0.5],
                        color: [0.1, 0.4, 0.9, 1.0],
                        radius: 4.0,
                        times: Vec::new(),
                    },
                    Series::Cloud {
                        xs,
                        ys,
                        times,
                        x0: 0.0,
                        x1: 2.0,
                        y0: 0.0,
                        y1: 2.0,
                        style: CloudStyle::Density,
                        cutoff: 0.0,
                    },
                ],
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
            volume: scan_kit_core::VolumeMark::default(),
        };
        let mut plot = Plot::new(&scene, [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]);
        let points = plot.marks.points.clone();
        let runs = plot.marks.panels[0].point_runs.clone();
        let full = plot.marks.heatmaps.clone();
        let source = runs.iter().find(|run| run.source).expect("timed cloud");
        plot.follow_time(true, 0.0, 0.5, true);
        assert_ne!(plot.marks.heatmaps, full);
        assert_eq!(plot.marks.points, points);
        assert_eq!(plot.marks.panels[0].point_runs, runs);
        let draws = visible_draws(
            &plot.marks.points,
            plot.marks.panels[0].point_start,
            plot.marks.panels[0].point_count,
            &plot.marks.panels[0].point_runs,
            Some((0.0, 0.5)),
        );
        assert!(draws.iter().all(|(start, count)| {
            *start >= source.start + source.count || *start + *count <= source.start
        }));
        plot.follow_time(false, 0.0, 0.0, true);
        assert_eq!(plot.marks.heatmaps, full);
        assert_eq!(plot.marks.points, points);
    }

    #[test]
    fn follow_time_draws_the_time_trace_inside_the_playhead() {
        let scene = PlotScene {
            title: "replay".into(),
            panels: vec![
                Panel {
                    title: String::new(),
                    y_label: String::new(),
                    x_label: String::new(),
                    xmin: 0.0,
                    xmax: 4.0,
                    ymin: 0.0,
                    ymax: 10.0,
                    series: vec![
                        Series::Polyline {
                            xs: vec![0.0, 1.0, 2.0, 3.0],
                            ys: vec![1.0, 1.0, 1.0, 1.0],
                            color: [1.0, 0.0, 0.0, 1.0],
                            thickness: 2.0,
                        },
                        Series::Guide {
                            xs: vec![10.0, 12.0],
                            ys: vec![5.0, 5.0],
                            color: [0.4, 0.4, 0.4, 1.0],
                            thickness: 1.0,
                        },
                    ],
                    x_labels: Vec::new(),
                    equal: false,
                },
                time_panel("Hz", 1.0, 500.0, vec![10.0, 20.0], vec![0.0, 1.0]),
            ],
            controls: Vec::new(),
            table: None,
            columns: 2,
            column_weights: Vec::new(),
            row_weights: Vec::new(),
            side: 0,
            row_splits: Vec::new(),
            volume: scan_kit_core::VolumeMark::default(),
        };
        let mut plot = Plot::new(&scene, [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]);
        assert_eq!(plot.marks.lines.len(), 5);
        let runs = &plot.marks.panels[0].line_runs;
        assert!(runs[0].monotonic);
        assert!(!runs[1].monotonic);
        plot.follow_time(true, 0.5, 1.5, true);
        assert_eq!(
            plot.line_draw_ranges(),
            vec![vec![(0, 2), (3, 1)], vec![(4, 1)]]
        );
        plot.follow_time(true, 0.0, 4.0, true);
        assert_eq!(
            plot.line_draw_ranges(),
            vec![vec![(0, 3), (3, 1)], vec![(4, 1)]]
        );
        plot.follow_time(false, 0.0, 0.0, true);
        assert_eq!(plot.line_draw_ranges(), vec![vec![(0, 4)], vec![(4, 1)]]);
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
                times: Vec::new(),
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
            times: Vec::new(),
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
    fn scatter_samples_in_one_cell_upload_once() {
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for index in 0..400 {
            xs.push(1.0 + index as f32 * 1.0e-6);
            ys.push(1.0);
        }
        xs.push(8.0);
        ys.push(8.0);
        let marks = build_marks(&[Panel {
            title: String::new(),
            y_label: String::new(),
            x_label: String::new(),
            xmin: 0.0,
            xmax: 10.0,
            ymin: 0.0,
            ymax: 10.0,
            series: vec![Series::Points {
                xs,
                ys,
                color: [0.1, 0.2, 0.8, 1.0],
                radius: 2.0,
                times: Vec::new(),
            }],
            x_labels: Vec::new(),
            equal: true,
        }]);
        assert_eq!(marks.points.len(), 2);
        assert!((marks.points[0].p[0] - 1.0).abs() < 1.0e-4);
        assert!((marks.points[1].p[0] - 8.0).abs() < 1.0e-4);
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
        let glyphs = labels_for(&panel, &camera, &cell, [1.0, 1.0, 1.0, 1.0], true);
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
        let glyphs = labels_for(&panel, &camera, &cell, [1.0, 1.0, 1.0, 1.0], true);
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

    fn projection_fill(frame: &[u8], width: u32, rect: PlotRect) -> f32 {
        let x0 = rect.x.floor() as i32;
        let x1 = (rect.x + rect.w * 0.90).ceil() as i32;
        let y0 = rect.y.floor() as i32;
        let y1 = (rect.y + rect.h).ceil() as i32;
        let mut min_x = i32::MAX;
        let mut max_x = i32::MIN;
        let mut min_y = i32::MAX;
        let mut max_y = i32::MIN;
        let mut lit = 0i32;
        for y in y0..y1 {
            for x in x0..x1 {
                if x < 0 || y < 0 {
                    continue;
                }
                let index = ((y as u32 * width + x as u32) * 4) as usize;
                let Some(px) = frame.get(index..index + 3) else {
                    continue;
                };
                if px[0] as u32 + px[1] as u32 + px[2] as u32 <= 24 {
                    continue;
                }
                lit += 1;
                min_x = min_x.min(x);
                max_x = max_x.max(x);
                min_y = min_y.min(y);
                max_y = max_y.max(y);
            }
        }
        if lit == 0 || max_x < min_x || max_y < min_y {
            return 0.0;
        }
        let area = (max_x - min_x + 1) * (max_y - min_y + 1);
        lit as f32 / area as f32
    }

    fn ramp_top_is_in_the_cloud(frame: &[u8], width: u32, rect: PlotRect, top: [u8; 3]) -> bool {
        let (hot, lit, best) = cloud_top(frame, width, rect, top);
        // Turbo's red tip is steep, and a pixel center plus the GPU reconstruction
        // sit a little under the analytic peak ray. A flat field still fails
        // because the lit cloud would not be several times the hot core.
        best <= 80 && lit > hot.max(1) * 4
    }

    fn cloud_top(frame: &[u8], width: u32, rect: PlotRect, top: [u8; 3]) -> (u32, u32, u32) {
        let x0 = rect.x.floor() as i32;
        let x1 = (rect.x + rect.w * 0.90).ceil() as i32;
        let y0 = rect.y.floor() as i32;
        let y1 = (rect.y + rect.h).ceil() as i32;
        let mut hot = 0u32;
        let mut lit = 0u32;
        let mut best = u32::MAX;
        for y in y0..y1 {
            for x in x0..x1 {
                if x < 0 || y < 0 {
                    continue;
                }
                let index = ((y as u32 * width + x as u32) * 4) as usize;
                let Some(px) = frame.get(index..index + 3) else {
                    continue;
                };
                let sum = px[0] as u32 + px[1] as u32 + px[2] as u32;
                if sum < 24 {
                    continue;
                }
                lit += 1;
                let delta = px
                    .iter()
                    .zip(top)
                    .map(|(got, want)| (*got as i32 - want as i32).unsigned_abs())
                    .max()
                    .unwrap_or(255);
                best = best.min(delta);
                if delta <= 12 {
                    hot += 1;
                }
            }
        }
        (hot, lit, best)
    }

    fn dose_values(n: usize, dense: bool) -> Vec<f32> {
        let mut values = vec![0.0; n * n * n];
        if dense {
            for z in 0..n {
                let dz = z as f32 / n as f32;
                let bragg = (-((dz - 0.72) / 0.08).powi(2)).exp();
                for y in 0..n {
                    for x in 0..n {
                        let dx = x as f32 / n as f32 - 0.5;
                        let dy = y as f32 / n as f32 - 0.5;
                        let lateral = (-(dx * dx + dy * dy) / 0.02).exp();
                        values[x + n * (y + n * z)] = bragg * lateral;
                    }
                }
            }
            return values;
        }
        for (sx, sy, sz) in [(0.3f32, 0.4, 0.7), (0.55, 0.5, 0.6), (0.7, 0.35, 0.8)] {
            let cx = (sx * n as f32) as i32;
            let cy = (sy * n as f32) as i32;
            let cz = (sz * n as f32) as i32;
            for dz in -2..=2 {
                for dy in -2..=2 {
                    for dx in -2..=2 {
                        let x = cx + dx;
                        let y = cy + dy;
                        let z = cz + dz;
                        if (0..n as i32).contains(&x)
                            && (0..n as i32).contains(&y)
                            && (0..n as i32).contains(&z)
                        {
                            values[x as usize + n * (y as usize + n * z as usize)] = 1.0;
                        }
                    }
                }
            }
        }
        values
    }

    fn dose_scene(n: usize, values: Vec<f32>, mode: u8, filter: u8) -> PlotScene {
        let mut scene = line_scene();
        let panel = scene.panels.remove(0);
        for title in [
            "Axial",
            "3D",
            "Coronal",
            "Sagittal",
            "Depth dose",
            "Lateral profile",
        ] {
            let mut next = panel.clone();
            next.title = title.into();
            next.equal = dose_plane(title).is_some();
            next.xmin = 0.0;
            next.xmax = n as f32;
            next.ymin = 0.0;
            next.ymax = n as f32;
            next.series = if dose_plane(title).is_some() {
                vec![Series::Heatmap {
                    values: vec![0.0; n * n],
                    cols: n as u32,
                    rows: n as u32,
                    ramp: 1,
                    color: [1.0; 4],
                    lo: 0.0,
                    hi: 1.0,
                }]
            } else {
                Vec::new()
            };
            scene.panels.push(next);
        }
        scene.columns = 2;
        scene.row_weights = vec![1.0, 1.0, 1.0];
        scene.row_splits = vec![1.0; 6];
        scene.volume = scan_kit_core::VolumeMark {
            values,
            shape: [n as u32, n as u32, n as u32],
            voxel: 1.0,
            ramp: 1,
            hi: 1.0,
            mode,
            filter,
            ..scan_kit_core::VolumeMark::default()
        };
        scene
    }

    fn gpu_frame_ms(plot: &mut Plot, width: u32, height: u32, frames: u32) -> f32 {
        let gpu = crate::native_gpu().expect("gpu");
        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("bench"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        for _ in 0..2 {
            let encoder = plot.record(gpu, &view, width, height).unwrap();
            gpu.queue.submit(Some(encoder.finish()));
        }
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        let start = std::time::Instant::now();
        for _ in 0..frames {
            let encoder = plot.record(gpu, &view, width, height).unwrap();
            gpu.queue.submit(Some(encoder.finish()));
        }
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        start.elapsed().as_secs_f32() * 1000.0 / frames as f32
    }

    #[test]
    #[ignore = "timing; cargo test -p scan-kit-plot -- --ignored --nocapture dose_view_frame_times"]
    fn dose_view_frame_times() {
        for (n, dense) in [(96usize, false), (96, true), (160, true)] {
            let values = dose_values(n, dense);
            let kind = if dense { "dense" } else { "sparse" };
            let volume = scan_kit_core::Volume {
                origin: [0.0, 0.0, 0.0],
                shape: [n, n, n],
                voxel: 1.0,
                values: values.clone(),
            };
            let start = std::time::Instant::now();
            let scan = scan_kit_core::scan_volume(&volume);
            eprintln!(
                "prep {n} {kind} scan {:.1} ms",
                start.elapsed().as_secs_f32() * 1000.0
            );
            let _ = scan.line_scale;
            let start = std::time::Instant::now();
            let grid = crate::dose::grid_from(
                values.clone(),
                Vec::new(),
                Vec::new(),
                [n as u32, n as u32, n as u32],
                [0.0, 0.0, 0.0],
                1.0,
                1,
                0.0,
                1.0,
                1.0,
                1.0,
                0,
                1,
            )
            .unwrap();
            eprintln!(
                "prep {n} {kind} grid {:.1} ms",
                start.elapsed().as_secs_f32() * 1000.0
            );
            let start = std::time::Instant::now();
            let _ = crate::dose::atlas_bytes(&grid);
            eprintln!(
                "prep {n} {kind} atlas {:.1} ms",
                start.elapsed().as_secs_f32() * 1000.0
            );
            let start = std::time::Instant::now();
            let _ = grid.volume.integrated_slice(0);
            eprintln!(
                "prep {n} {kind} uncached axial {:.1} ms",
                start.elapsed().as_secs_f32() * 1000.0
            );
            let start = std::time::Instant::now();
            for _ in 0..1000 {
                let _ = crate::dose::integral_peak(&grid, 0);
            }
            eprintln!(
                "prep {n} {kind} cached integral x1000 {:.3} ms",
                start.elapsed().as_secs_f32() * 1000.0
            );
        }
        if crate::native_gpu().is_err() {
            eprintln!("no gpu");
            return;
        }
        let width = 1280u32;
        let height = 720u32;
        for (n, dense, mode, filter, label) in [
            (96usize, false, 0u8, 1u8, "sparse integrate linear"),
            (96, true, 0, 1, "dense integrate linear"),
            (96, true, 1, 1, "dense maximum linear"),
            (96, true, 0, 0, "dense integrate nearest"),
            (96, true, 2, 1, "dense transparent linear"),
            (160, true, 0, 1, "160 dense integrate linear"),
            (160, true, 1, 0, "160 dense maximum nearest"),
        ] {
            let scene = dose_scene(n, dose_values(n, dense), mode, filter);
            let mut plot = Plot::new(&scene, [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]);
            let ms = gpu_frame_ms(&mut plot, width, height, 8);
            eprintln!("gpu {width}x{height} {label}: {ms:.2} ms");
        }
        let mut alone = dose_scene(160, dose_values(160, true), 0, 1);
        alone.panels.swap(0, 1);
        alone.panels.truncate(1);
        alone.columns = 1;
        alone.row_weights.clear();
        alone.row_splits.clear();
        let mut plot = Plot::new(&alone, [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]);
        for (side_w, side_h) in [(640u32, 480u32), (1280, 960), (1920, 1080)] {
            let ms = gpu_frame_ms(&mut plot, side_w, side_h, 6);
            eprintln!("gpu 3d-only {side_w}x{side_h}: {ms:.2} ms");
        }
    }
}
