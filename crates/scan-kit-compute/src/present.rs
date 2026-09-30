//! One analysis frame. The scene comes from `scan-kit-io`. This module paints it.
//!
//! The desktop keeps one plot and asks for raw frames. `run_view` still returns
//! the base64 frame that MCP and the view tests use.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use scan_kit_core::{Series, ToolKind, ToolSpec};
use scan_kit_io::analysis_scene;
use serde_json::{json, Value};

use crate::render::{Plot, PlotInput};

const TOOLS: &[ToolSpec] = &[ToolSpec {
    name: "scan_kit_run_view",
    summary: "Build an analysis view and return one RGBA frame plus its controls.",
    kind: ToolKind::Workflow,
}];

pub fn tools() -> &'static [ToolSpec] {
    TOOLS
}

pub fn tool_input_schema(name: &str) -> Value {
    match name {
        "scan_kit_run_view" => json!({
            "type": "object",
            "properties": {
                "view": { "type": "string" },
                "path": { "type": "string" },
                "session_ids": { "type": "array", "items": { "type": "string" } },
                "options": { "type": "object" },
                "width": { "type": "integer" },
                "height": { "type": "integer" },
                "background": { "type": "array", "items": { "type": "number" } },
                "foreground": { "type": "array", "items": { "type": "number" } },
                "palette": { "type": "array", "items": { "type": "array", "items": { "type": "number" } } }
            },
            "required": ["view", "path", "session_ids"],
            "additionalProperties": false
        }),
        _ => json!({ "type": "object", "additionalProperties": false }),
    }
}

pub fn invoke(name: &str, input: &Value) -> Result<Value, String> {
    match name {
        "scan_kit_run_view" => {
            let view = text(input, "view")?;
            let path = text(input, "path")?;
            let session_ids = strings(input, "session_ids")?;
            let options = input.get("options").cloned().unwrap_or_else(|| json!({}));
            let width = input.get("width").and_then(Value::as_u64).unwrap_or(960) as u32;
            let height = input.get("height").and_then(Value::as_u64).unwrap_or(640) as u32;
            let background = color4(input.get("background"), [0.11, 0.11, 0.12, 1.0]);
            let foreground = color4(input.get("foreground"), [0.92, 0.92, 0.93, 1.0]);
            let palette = palette_of(input.get("palette"));
            run_view(
                view,
                Path::new(path),
                &session_ids,
                &options,
                width,
                height,
                background,
                foreground,
                &palette,
            )
        }
        _ => Err(format!("unknown tool {name}")),
    }
}

/// Load the view, color its series from the shell palette, and read one frame back.
pub fn run_view(
    view: &str,
    root: &Path,
    session_ids: &[String],
    options: &Value,
    width: u32,
    height: u32,
    background: [f32; 4],
    foreground: [f32; 4],
    palette: &[[f32; 4]],
) -> Result<Value, String> {
    let mut scene = analysis_scene(view, root, session_ids, options)?;
    apply_palette(&mut scene, palette);
    let frame = paint_once(&scene, width, height, background, foreground)?;
    Ok(json!({
        "title": scene.title,
        "width": frame.width,
        "height": frame.height,
        "rgba_base64": base64(&frame.rgba),
        "controls": scene.controls,
        "table": scene.table,
        "samples": scene.samples,
    }))
}

struct LivePlot {
    id: u64,
    plot: Plot,
}

fn live() -> std::sync::MutexGuard<'static, Option<LivePlot>> {
    static LIVE: Mutex<Option<LivePlot>> = Mutex::new(None);
    LIVE.lock().unwrap_or_else(|poison| poison.into_inner())
}

fn next_id() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

/// Build the view once and keep the plot for pointer frames.
pub fn open_plot(
    view: &str,
    root: &Path,
    session_ids: &[String],
    options: &Value,
    background: [f32; 4],
    foreground: [f32; 4],
    palette: &[[f32; 4]],
) -> Result<Value, String> {
    let mut scene = analysis_scene(view, root, session_ids, options)?;
    apply_palette(&mut scene, palette);
    let plot = Plot::new(&scene, background, foreground);
    let id = next_id();
    *live() = Some(LivePlot { id, plot });
    Ok(json!({
        "id": id,
        "title": scene.title,
        "controls": scene.controls,
        "table": scene.table,
        "samples": scene.samples,
    }))
}

/// Redraw the retained plot. The bytes are a 24-byte header plus RGBA.
pub fn plot_frame(
    id: u64,
    width: u32,
    height: u32,
    x: f32,
    y: f32,
    dx: f32,
    dy: f32,
    wheel: f32,
    drag: bool,
    reset: bool,
) -> Result<Vec<u8>, String> {
    let mut slot = live();
    let live = slot.as_mut().ok_or("no plot")?;
    if live.id != id {
        return Err("stale plot".into());
    }
    let frame = live.plot.draw(
        width,
        height,
        &PlotInput {
            x,
            y,
            dx,
            dy,
            wheel,
            drag,
            reset,
        },
    )?;
    Ok(pack_frame(frame.width, frame.height, frame.hover_hit, frame.hover_x, frame.hover_y, frame.series, &frame.rgba))
}

/// Data coordinates under the cursor. This does not copy the frame.
pub fn plot_hover(id: u64, x: f32, y: f32) -> Result<Value, String> {
    let slot = live();
    let live = slot.as_ref().ok_or("no plot")?;
    if live.id != id {
        return Err("stale plot".into());
    }
    let (hit, hover_x, hover_y, series) = live.plot.hover(x, y);
    Ok(json!({
        "hit": hit,
        "x": hover_x,
        "y": hover_y,
        "series": series,
    }))
}

fn paint_once(
    scene: &scan_kit_core::PlotScene,
    width: u32,
    height: u32,
    background: [f32; 4],
    foreground: [f32; 4],
) -> Result<crate::render::PlotFrame, String> {
    if scene.panels.is_empty() {
        return Ok(crate::render::PlotFrame {
            rgba: Vec::new(),
            width: 0,
            height: 0,
            hover_hit: false,
            hover_x: 0.0,
            hover_y: 0.0,
            series: None,
        });
    }
    Plot::new(scene, background, foreground).draw(width, height, &PlotInput::default())
}

fn pack_frame(
    width: u32,
    height: u32,
    hover_hit: bool,
    hover_x: f32,
    hover_y: f32,
    series: Option<u32>,
    rgba: &[u8],
) -> Vec<u8> {
    let mut flags = 0u32;
    if hover_hit {
        flags |= 1;
    }
    if series.is_some() {
        flags |= 2;
    }
    let mut out = Vec::with_capacity(24 + rgba.len());
    out.extend_from_slice(&width.to_le_bytes());
    out.extend_from_slice(&height.to_le_bytes());
    out.extend_from_slice(&flags.to_le_bytes());
    out.extend_from_slice(&hover_x.to_le_bytes());
    out.extend_from_slice(&hover_y.to_le_bytes());
    out.extend_from_slice(&series.unwrap_or(u32::MAX).to_le_bytes());
    out.extend_from_slice(rgba);
    out
}

fn apply_palette(scene: &mut scan_kit_core::PlotScene, palette: &[[f32; 4]]) {
    if palette.is_empty() {
        for panel in &mut scene.panels {
            for series in &mut panel.series {
                if let Series::Polyline { color, .. } | Series::Points { color, .. } = series {
                    if color[3] == 0.0 {
                        color[3] = 1.0;
                    }
                }
            }
        }
        return;
    }
    for panel in &mut scene.panels {
        let mut index = 0usize;
        for series in &mut panel.series {
            let color = palette[index % palette.len()];
            match series {
                Series::Polyline { color: slot, .. }
                | Series::Points { color: slot, .. }
                | Series::Bars { color: slot, .. }
                | Series::Rects { color: slot, .. } => {
                    let alpha = slot[3];
                    // Alpha 0 keeps the previous series color, so a mean curve and its
                    // trend stay with that session instead of taking the next chart color.
                    if alpha == 0.0 && index > 0 {
                        let color = palette[(index - 1) % palette.len()];
                        *slot = [color[0], color[1], color[2], 1.0];
                    } else {
                        *slot = [color[0], color[1], color[2], alpha.max(0.05)];
                        index += 1;
                    }
                }
                Series::Heatmap { .. } | Series::Guide { .. } => {}
            }
        }
    }
}

fn text<'a>(input: &'a Value, key: &str) -> Result<&'a str, String> {
    input
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{key} is required"))
}

fn strings(input: &Value, key: &str) -> Result<Vec<String>, String> {
    input
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{key} is required"))?
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(str::to_owned)
                .ok_or_else(|| format!("{key} must be strings"))
        })
        .collect()
}

fn color4(value: Option<&Value>, fallback: [f32; 4]) -> [f32; 4] {
    let Some(items) = value.and_then(Value::as_array) else {
        return fallback;
    };
    let mut out = fallback;
    for (index, item) in items.iter().take(4).enumerate() {
        if let Some(number) = item.as_f64() {
            out[index] = number as f32;
        }
    }
    out
}

fn palette_of(value: Option<&Value>) -> Vec<[f32; 4]> {
    let Some(rows) = value.and_then(Value::as_array) else {
        return Vec::new();
    };
    rows.iter()
        .map(|row| color4(Some(row), [0.9, 0.9, 0.9, 1.0]))
        .collect()
}

fn base64(bytes: &[u8]) -> String {
    const TABLE: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    let mut index = 0;
    while index + 3 <= bytes.len() {
        let n = ((bytes[index] as u32) << 16)
            | ((bytes[index + 1] as u32) << 8)
            | bytes[index + 2] as u32;
        out.push(TABLE[((n >> 18) & 63) as usize] as char);
        out.push(TABLE[((n >> 12) & 63) as usize] as char);
        out.push(TABLE[((n >> 6) & 63) as usize] as char);
        out.push(TABLE[(n & 63) as usize] as char);
        index += 3;
    }
    let rest = bytes.len() - index;
    if rest == 1 {
        let n = (bytes[index] as u32) << 16;
        out.push(TABLE[((n >> 18) & 63) as usize] as char);
        out.push(TABLE[((n >> 12) & 63) as usize] as char);
        out.push('=');
        out.push('=');
    } else if rest == 2 {
        let n = ((bytes[index] as u32) << 16) | ((bytes[index + 1] as u32) << 8);
        out.push(TABLE[((n >> 18) & 63) as usize] as char);
        out.push(TABLE[((n >> 12) & 63) as usize] as char);
        out.push(TABLE[((n >> 6) & 63) as usize] as char);
        out.push('=');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sk_req_008_run_view_returns_a_frame_and_controls() {
        let root = std::env::temp_dir().join(format!("scan-kit-frame-{}", std::process::id()));
        let session = root.join("sess");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&session).unwrap();
        std::fs::write(
            session.join("input_map.csv"),
            "energy,charge_req,position_x,position_y\n70,1,0,0\n90,2,4,1\n",
        )
        .unwrap();
        std::fs::write(
            session.join("spot_data.csv"),
            "ic1_total_dose,ic2_total_dose,position_x,position_y\n1,1,0,0\n2,2,4,1\n",
        )
        .unwrap();
        let frame = run_view(
            "dose_accumulation",
            &root,
            &["sess".into()],
            &json!({}),
            320,
            180,
            [0.1, 0.1, 0.1, 1.0],
            [0.9, 0.9, 0.9, 1.0],
            &[[0.8, 0.8, 0.8, 1.0], [0.4, 0.4, 0.4, 1.0]],
        )
        .unwrap();
        assert_eq!(frame["title"], "Dose Accumulation");
        assert!(frame["rgba_base64"].as_str().unwrap().len() > 32);
        assert_eq!(frame["width"], 320);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn retained_plot_frame_is_a_raw_header_plus_rgba() {
        let root = std::env::temp_dir().join(format!("scan-kit-live-{}", std::process::id()));
        let session = root.join("sess");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&session).unwrap();
        std::fs::write(
            session.join("input_map.csv"),
            "energy,charge_req,position_x,position_y\n70,1,0,0\n90,2,4,1\n",
        )
        .unwrap();
        std::fs::write(
            session.join("spot_data.csv"),
            "ic1_total_dose,ic2_total_dose,position_x,position_y\n1,1,0,0\n2,2,4,1\n",
        )
        .unwrap();
        let opened = open_plot(
            "dose_accumulation",
            &root,
            &["sess".into()],
            &json!({}),
            [0.1, 0.1, 0.1, 1.0],
            [0.9, 0.9, 0.9, 1.0],
            &[[0.8, 0.2, 0.2, 1.0]],
        )
        .unwrap();
        let id = opened["id"].as_u64().unwrap();
        assert_eq!(opened["title"], "Dose Accumulation");
        let bytes = plot_frame(id, 80, 60, -1.0, -1.0, 0.0, 0.0, 0.0, false, false).unwrap();
        assert_eq!(u32::from_le_bytes(bytes[0..4].try_into().unwrap()), 80);
        assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), 60);
        assert_eq!(bytes.len(), 24 + 80 * 60 * 4);
        let again = plot_frame(id, 80, 60, 40.0, 30.0, 0.0, 0.0, 1.0, false, false).unwrap();
        assert_eq!(again.len(), bytes.len());
        let hover = plot_hover(id, 40.0, 30.0).unwrap();
        assert!(hover["hit"].is_boolean());
        assert!(plot_frame(id + 9, 80, 60, -1.0, -1.0, 0.0, 0.0, 0.0, false, false).is_err());
        let _ = std::fs::remove_dir_all(&root);
    }
}
