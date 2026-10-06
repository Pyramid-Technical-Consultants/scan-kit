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
    // textureLoad and textureSampleLevel have no derivatives, so a dose atlas
    // lookup can sit in this branch. heat > 1.5 samples the uploaded volume.
    // heat > 0.5 is a baked ramp. Otherwise the quad is a solid color.
    if (in.heat > 1.5) {
        return dose_fragment(in.uv, coverage);
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
        return vec4<f32>(0.0);
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

fn voxel_dose(ix: i32, iy: i32, iz: i32) -> f32 {
    return atlas_load(ix, iy, iz).r * u.dose1.w;
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
    return max(acc, vec4<f32>(0.0));
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

fn dose_fragment(uv: vec2<f32>, coverage: f32) -> vec4<f32> {
    if (uv.x > 0.94) {
        let bar = ramp_color(clamp(1.0 - uv.y, 0.0, 1.0));
        return shade(bar.rgb, bar.a, coverage);
    }
    let plane = u32(u.dose0.x + 0.5);
    let film = turned_uv(vec2<f32>(uv.x / 0.94, uv.y), u32(u.dose0.z + 0.5));
    let nx = max(u.dose1.x, 1.0);
    let ny = max(u.dose1.y, 1.0);
    let nz = max(u.dose1.z, 1.0);
    if (plane == 3u) {
        let marched = march_color(film);
        return shade(marched.rgb, marched.a, coverage);
    }
    var raw = 0.0;
    var under = vec3<f32>(0.0);
    var has_ct = false;
    let ix = film.x * nx - 0.5;
    let iy = (1.0 - film.y) * ny - 0.5;
    let iz = u.dose0.y;
    if (u.dose0.w > 0.5) {
        let count = select(select(nz, ny, plane == 1u), nx, plane == 2u);
        for (var step = 0; step < 256; step = step + 1) {
            if (f32(step) >= count) { break; }
            let texel = select(
                select(atlas_load(i32(round(ix)), i32(round(iy)), step), atlas_load(i32(round(ix)), step, i32(round(iz))), plane == 1u),
                atlas_load(step, i32(round(iy)), i32(round(iz))),
                plane == 2u,
            );
            raw = raw + texel.r * u.dose1.w * u.dose3.z;
        }
        let color = wash(raw, 0.0, max(u.dose3.w, 0.000001), under, false);
        return shade(color.rgb, color.a, coverage);
    }
    let texel = select(
        select(sample_grid(ix, iy, iz), sample_grid(ix, iz, iy), plane == 1u),
        sample_grid(iz, iy, ix),
        plane == 2u,
    );
    raw = texel.r * u.dose1.w;
    has_ct = texel.b > 0.5;
    under = vec3<f32>(texel.g);
    let color = wash(raw, u.dose2.x, u.dose2.y, under, has_ct);
    return shade(color.rgb, color.a, coverage);
}

fn march_color(uv: vec2<f32>) -> vec4<f32> {
    let azimuth = u.dose0.y;
    let elevation = u.dose0.z;
    let ce = cos(elevation);
    var dir = vec3<f32>(ce * sin(azimuth), ce * cos(azimuth), sin(elevation));
    var right = vec3<f32>(cos(azimuth), -sin(azimuth), 0.0);
    dir = normalize(dir);
    right = normalize(right);
    let up = normalize(cross(dir, right));
    let nx = max(u.dose1.x, 1.0);
    let ny = max(u.dose1.y, 1.0);
    let nz = max(u.dose1.z, 1.0);
    let radius = max(max(nx, ny), nz) * u.dose3.z * 0.866;
    let center = vec3<f32>(nx, ny, nz) * 0.5;
    let ufilm = uv.x * 2.0 - 1.0;
    let vfilm = (1.0 - uv.y) * 2.0 - 1.0;
    let start = center + right * ufilm * radius + up * vfilm * radius - dir * radius;
    let step = radius * 2.0 / 24.0;
    var acc = 0.0;
    var cover = 0.0;
    var best = 0.0;
    let mode = u32(u.dose3.x + 0.5);
    for (var i = 0; i < 24; i = i + 1) {
        let point = start + dir * step * (f32(i) + 0.5);
        let texel = sample_grid(point.x - 0.5, point.y - 0.5, point.z - 0.5);
        let dose = texel.r * u.dose1.w;
        if (mode == 1u) {
            best = max(best, dose);
        } else if (mode == 2u) {
            let alpha = 1.0 - exp(-dose / max(u.dose1.w, 0.000001) * 1.5);
            acc = acc + (1.0 - cover) * dose * alpha;
            cover = cover + (1.0 - cover) * alpha;
        } else {
            acc = acc + dose * step;
        }
    }
    let raw = select(acc, best, mode == 1u);
    return wash(raw, u.dose2.x, u.dose2.y, vec3<f32>(0.0), false);
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
pub(crate) const POINT_STRIDE: u64 = 48;
pub(crate) const QUAD_STRIDE: u64 = 64;
/// Cells across one axis of the panel range. Samples in one cell upload once.
const POINT_CELLS: u32 = 4096;
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
    row_splits: Vec<f32>,
    side: u32,
    /// Gap under the pointer while a divider is dragged. Cleared when the button lifts.
    split_grab: Option<SplitGrab>,
    /// Uploaded dose grid. Paging and orbit sample it; they do not replace it.
    dose: Option<crate::dose::DoseGrid>,
    cursor: [usize; 3],
    azimuth: f32,
    elevation: f32,
    turns: Vec<u8>,
    integral: Vec<bool>,
    profile_integral: Vec<bool>,
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
    cloud_quads: Option<wgpu::Buffer>,
    cloud_lines: Option<wgpu::Buffer>,
    heat_revision: u32,
    heat_uploaded: u32,
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
            cursor: [0, 0, 0],
            azimuth: 0.6,
            elevation: 0.4,
            turns: vec![0; panel_count],
            integral: vec![false; panel_count],
            profile_integral: vec![false; panel_count],
            atlas_stamp: 0,
            home: cameras.clone(),
            cameras,
            background,
            foreground,
            gpu: None,
            frame_buf: None,
            text_buf: None,
            cloud_quads: None,
            cloud_lines: None,
            heat_revision: 0,
            heat_uploaded: 0,
            uniform_buf: None,
            uniform_groups: Vec::new(),
            mark_uploads: 0,
            size: (0, 0),
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
        let Some(grid) = crate::dose::grid_from(
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
        let (pixels, cols, rows) = crate::dose::atlas_bytes(&grid);
        let atlas = self.marks.heatmaps.len();
        self.marks.heatmaps.push(pixels);
        self.marks.heatmap_size.push((cols, rows));
        let (cx, cy, cz) = grid.volume.peak_index();
        self.cursor = [cx, cy, cz];
        for index in 0..self.panels.len() {
            let title = self.panels[index].title.clone();
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
        self.atlas_stamp = self.atlas_stamp.wrapping_add(1);
        self.heat_revision = self.heat_revision.wrapping_add(1);
        if let Some(encoded) = self.encoded.as_mut() {
            encoded.quads = encode_quads(&self.marks.quads);
        }
        self.refresh_dose_overlay();
    }

    fn dose_uniform(&self, panel: usize) -> [[f32; 4]; 4] {
        let Some(grid) = &self.dose else {
            return [[0.0; 4]; 4];
        };
        let Some(plane) = self
            .panels
            .get(panel)
            .and_then(|item| dose_plane(&item.title))
        else {
            return [[0.0; 4]; 4];
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
        let (lo, hi) = if grid.hi > grid.lo {
            (grid.lo, grid.hi)
        } else {
            (0.0, grid.peak)
        };
        [
            [plane as f32, second, third, integral],
            [nx as f32, ny as f32, nz as f32, grid.peak],
            [lo, hi, grid.gain, grid.opacity],
            [
                grid.mode as f32,
                grid.filter as f32,
                grid.volume.voxel,
                integral_peak,
            ],
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
                });
            }
            if let Some(plane) = dose_plane(&title) {
                if plane < 3 {
                    extra.push(LiveCloud {
                        panel: index,
                        quads: Vec::new(),
                        lines: crosshair_lines(
                            grid, plane, cursor, xmin, xmax, ymin, ymax, foreground,
                        ),
                        dose: true,
                    });
                    let mut lines = Vec::new();
                    for (id, a, b) in
                        crate::dose::outlines(grid, plane, cursor_index(plane, cursor))
                    {
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
                    if let Some(flag) = self.profile_integral.get_mut(panel) {
                        *flag = !*flag;
                    }
                }
            }
            _ => return,
        }
        self.refresh_dose_overlay();
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

    /// Keep a zoomed window when the new panel is the same quantity and its
    /// data range is still close. A different label or a much larger span
    /// starts from the new limits. Extra panels keep their own fit.
    #[cfg_attr(not(any(test, target_arch = "wasm32")), allow(dead_code))]
    pub(crate) fn adopt_view(&mut self, previous: &Plot) {
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
            self.cursor = previous.cursor;
            self.azimuth = previous.azimuth;
            self.elevation = previous.elevation;
            if previous.turns.len() == self.panels.len() {
                self.turns.clone_from(&previous.turns);
            }
            if previous.integral.len() == self.panels.len() {
                self.integral.clone_from(&previous.integral);
            }
            if previous.profile_integral.len() == self.panels.len() {
                self.profile_integral.clone_from(&previous.profile_integral);
            }
            self.refresh_dose_overlay();
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
        if plane == 3 {
            if input.drag {
                self.azimuth += input.dx * 0.01;
                self.elevation = (self.elevation - input.dy * 0.01).clamp(-1.2, 1.2);
                return true;
            }
            return false;
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
            let [x, y] = self.cameras[panel].data_at(
                input.x,
                input.y,
                cell.plot,
                width as f32,
                height as f32,
            );
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
    fn fill_dose_cpu(
        &self,
        frame: &mut [u8],
        width: u32,
        height: u32,
        matrix: &[f32; 16],
        quad: QuadRec,
        cell: &Cell,
    ) {
        let Some(grid) = &self.dose else {
            return;
        };
        let Some(plane) = dose_plane(&self.panels[cell.panel].title) else {
            return;
        };
        let view = crate::dose::ViewSample {
            plane,
            index: cursor_index(plane, self.cursor),
            turns: self.turns.get(cell.panel).copied().unwrap_or(0),
            integral: self.integral.get(cell.panel).copied().unwrap_or(false),
            azimuth: self.azimuth,
            elevation: self.elevation,
        };
        let prepared = crate::dose::prepare_plane(grid, &view);
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
                let u = ((x as f32 - left) / span_x).clamp(0.0, 1.0);
                let v = ((y as f32 - top) / span_y).clamp(0.0, 1.0);
                let color = crate::dose::shade_uv(grid, &view, prepared.as_ref(), [u, v]);
                blend(frame, width, height, x, y, color, Some(cell.plot));
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn paint_cpu(&mut self, width: u32, height: u32, layout: &[Cell]) -> Vec<u8> {
        let line_draws = self.line_draw_ranges();
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
            let solid: Vec<QuadRec> = self.marks.quads
                [batch.quad_start as usize..(batch.quad_start + batch.quad_count) as usize]
                .to_vec();
            let heated: Vec<QuadRec> = batch
                .heats
                .iter()
                .filter_map(|(index, _)| self.marks.quads.get(*index as usize).copied())
                .collect();
            for quad in solid.iter().chain(&heated) {
                if quad.heat > 1.5 {
                    self.fill_dose_cpu(&mut frame, width, height, &matrix, *quad, cell);
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
            for cloud in self.live.iter().filter(|cloud| cloud.panel == cell.panel) {
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
        self.sync_heatmaps(&gpu.queue);
        let (cloud_quad_span, cloud_line_span) = self.upload_live(gpu);
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
            let bytes = encode_uniform(
                matrix,
                width as f32,
                height as f32,
                self.dose_uniform(cell.panel),
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
                let (quad_at, quad_count) = cloud_quad_span[cell.panel];
                if let (true, Some(buffer)) = (quad_count > 0, self.cloud_quads.as_ref()) {
                    pass.set_bind_group(1, &gpu.white_group, &[]);
                    pass.set_vertex_buffer(0, buffer.slice(quad_at * QUAD_STRIDE..));
                    pass.draw(0..6, 0..quad_count as u32);
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

    fn upload_live(&mut self, gpu: &PlotGpu) -> (Vec<(u64, u64)>, Vec<(u64, u64)>) {
        let panels = self.marks.panels.len();
        let mut quad_span = vec![(0u64, 0u64); panels];
        let mut line_span = vec![(0u64, 0u64); panels];
        let mut quad_bytes = Vec::new();
        let mut line_bytes = Vec::new();
        let mut quads = 0u64;
        let mut lines = 0u64;
        for panel in 0..panels {
            let quad_at = quads;
            let line_at = lines;
            for cloud in self.live.iter().filter(|cloud| cloud.panel == panel) {
                quad_bytes.extend(encode_quads(&cloud.quads));
                line_bytes.extend(encode_lines(&cloud.lines));
                quads += cloud.quads.len() as u64;
                lines += cloud.lines.len() as u64;
            }
            quad_span[panel] = (quad_at, quads - quad_at);
            line_span[panel] = (line_at, lines - line_at);
        }
        write_grow(&gpu.device, &gpu.queue, &mut self.cloud_quads, &quad_bytes);
        write_grow(&gpu.device, &gpu.queue, &mut self.cloud_lines, &line_bytes);
        (quad_span, line_span)
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
                    min_binding_size: Some(NonZeroU64::new(144).unwrap()),
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

fn profile_lines(
    grid: &crate::dose::DoseGrid,
    kind: u8,
    cursor: [usize; 3],
    integrate: bool,
    color: [f32; 4],
) -> Vec<LineRec> {
    let volume = &grid.volume;
    let [ix, iy, iz] = cursor;
    let mut series = Vec::new();
    let push = |series: &mut Vec<(Vec<f32>, Vec<f32>)>, pair: (Vec<f32>, Vec<f32>)| {
        series.push(pair);
    };
    match kind {
        0 if integrate => push(&mut series, volume.depth_integral(iy)),
        0 => push(&mut series, volume.depth_profile(ix, iy)),
        1 if integrate => push(&mut series, volume.lateral_integral(iz)),
        1 => push(&mut series, volume.lateral_profile(iy, iz)),
        2 if integrate => push(&mut series, volume.longitudinal_integral(iz)),
        2 => push(&mut series, volume.longitudinal_profile(ix, iz)),
        _ if integrate => {
            push(&mut series, volume.lateral_integral(iz));
            push(&mut series, volume.longitudinal_integral(iz));
        }
        _ => {
            push(&mut series, volume.lateral_profile(iy, iz));
            push(&mut series, volume.longitudinal_profile(ix, iz));
        }
    }
    let mut lines = Vec::new();
    for (index, (xs, ys)) in series.into_iter().enumerate() {
        let ink = if index == 0 {
            color
        } else {
            [0.95, 0.65, 0.25, color[3]]
        };
        let mut prev: Option<[f32; 3]> = None;
        for (x, y) in xs.into_iter().zip(ys) {
            if !x.is_finite() || !y.is_finite() {
                prev = None;
                continue;
            }
            let point = [x, y, 0.0];
            if let Some(start) = prev {
                lines.push(LineRec {
                    a: start,
                    b: point,
                    color: ink,
                    thickness: 1.5,
                    id: 0,
                });
            }
            prev = Some(point);
        }
    }
    lines
}

fn crosshair_lines(
    grid: &crate::dose::DoseGrid,
    plane: u8,
    cursor: [usize; 3],
    xmin: f32,
    xmax: f32,
    ymin: f32,
    ymax: f32,
    color: [f32; 4],
) -> Vec<LineRec> {
    let volume = &grid.volume;
    let (x, y) = match plane {
        1 => (volume.mm_of(0, cursor[0]), volume.mm_of(2, cursor[2])),
        2 => (volume.mm_of(1, cursor[1]), volume.mm_of(2, cursor[2])),
        _ => (volume.mm_of(0, cursor[0]), volume.mm_of(1, cursor[1])),
    };
    let ink = [color[0], color[1], color[2], 0.85];
    vec![
        LineRec {
            a: [x, ymin, 0.0],
            b: [x, ymax, 0.0],
            color: ink,
            thickness: 1.0,
            id: 0,
        },
        LineRec {
            a: [xmin, y, 0.0],
            b: [xmax, y, 0.0],
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

fn encode_uniform(matrix: [f32; 16], width: f32, height: f32, dose: [[f32; 4]; 4]) -> [u8; 144] {
    let mut out = [0u8; 144];
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
                size: Some(NonZeroU64::new(144).unwrap()),
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
