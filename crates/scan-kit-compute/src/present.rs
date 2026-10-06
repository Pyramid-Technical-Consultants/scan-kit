//! Analysis views. The scene comes from `scan-kit-io` and is colored here.
//!
//! `open_plot` packs it for the desktop, which draws it in the webview.
//! `run_view` renders the same plot offscreen and returns the base64 frame that
//! MCP and the view tests use.

use std::path::Path;

use scan_kit_core::{is_session, Series, ToolKind, ToolSpec};
use serde_json::{json, Value};

use scan_kit_plot::{Plot, PlotInput};

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
    let bytes = open_plot(
        view,
        root,
        session_ids,
        options,
        background,
        foreground,
        palette,
    )?;
    let header = scan_kit_plot::plot_header(&bytes)?;
    let mut plot = Plot::from_payload(&bytes)?;
    let frame = plot.draw(width, height, &PlotInput::default())?;
    Ok(json!({
        "title": header.title,
        "width": frame.width,
        "height": frame.height,
        "rgba_base64": base64(&frame.rgba),
        "controls": header.controls,
        "table": header.table,
    }))
}

/// Build the view and pack it for the desktop's wasm plot. See `scan_kit_plot::encode_plot`.
///
/// This drains the progressive task in one slice, so the bytes match a finished load.
pub fn open_plot(
    view: &str,
    root: &Path,
    session_ids: &[String],
    options: &Value,
    background: [f32; 4],
    foreground: [f32; 4],
    palette: &[[f32; 4]],
) -> Result<Vec<u8>, String> {
    crate::task::drain_plot(
        view,
        root,
        session_ids,
        options,
        background,
        foreground,
        palette,
    )
}

pub(crate) fn apply_palette(scene: &mut scan_kit_core::PlotScene, palette: &[[f32; 4]]) {
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
                | Series::Rects { color: slot, .. }
                | Series::Triangles { color: slot, .. } => {
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
                Series::Heatmap {
                    color: slot, ramp, ..
                } => {
                    if is_session(*ramp) {
                        *slot = [color[0], color[1], color[2], 1.0];
                        index += 1;
                    }
                }
                Series::Guide { .. } => {}
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
            "bins",
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
        assert!(frame["title"].as_str().unwrap().contains("vs"));
        assert!(frame["rgba_base64"].as_str().unwrap().len() > 32);
        assert_eq!(frame["width"], 320);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn open_plot_packs_a_header_and_marks_the_plot_can_draw() {
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
            "bins",
            &root,
            &["sess".into()],
            &json!({}),
            [0.1, 0.1, 0.1, 1.0],
            [0.9, 0.9, 0.9, 1.0],
            &[[0.8, 0.2, 0.2, 1.0]],
        )
        .unwrap();
        let header = scan_kit_plot::plot_header(&opened).unwrap();
        assert!(header.title.contains("vs"));
        assert!(header.panels.iter().all(|panel| panel.series.is_empty()));
        let mut plot = Plot::from_payload(&opened).unwrap();
        let frame = plot.draw(80, 60, &PlotInput::default()).unwrap();
        assert_eq!(frame.rgba.len(), 80 * 60 * 4);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn session_heatmaps_take_palette_colors() {
        use scan_kit_core::{Panel, PlotScene, Series, SESSION};
        let mut scene = PlotScene::empty("heat");
        scene.panels.push(Panel {
            title: String::new(),
            y_label: String::new(),
            x_label: String::new(),
            xmin: 0.0,
            xmax: 1.0,
            ymin: 0.0,
            ymax: 1.0,
            series: vec![
                Series::Heatmap {
                    values: vec![0.0, 1.0],
                    cols: 1,
                    rows: 2,
                    ramp: SESSION,
                    color: [0.0, 0.0, 0.0, 1.0],
                    lo: 0.0,
                    hi: 0.0,
                },
                Series::Heatmap {
                    values: vec![0.0, 1.0],
                    cols: 1,
                    rows: 2,
                    ramp: 1,
                    color: [1.0, 1.0, 1.0, 1.0],
                    lo: 0.0,
                    hi: 0.0,
                },
                Series::Heatmap {
                    values: vec![0.0, 1.0],
                    cols: 1,
                    rows: 2,
                    ramp: SESSION,
                    color: [0.0, 0.0, 0.0, 1.0],
                    lo: 0.0,
                    hi: 0.0,
                },
            ],
            x_labels: Vec::new(),
            equal: false,
        });
        apply_palette(&mut scene, &[[0.8, 0.1, 0.1, 1.0], [0.1, 0.7, 0.2, 1.0]]);
        let colors: Vec<_> = scene.panels[0]
            .series
            .iter()
            .map(|series| match series {
                Series::Heatmap { color, ramp, .. } => (*ramp, *color),
                _ => unreachable!(),
            })
            .collect();
        assert_eq!(colors[0].0, SESSION);
        assert_eq!(colors[0].1, [0.8, 0.1, 0.1, 1.0]);
        assert_eq!(colors[1], (1, [1.0, 1.0, 1.0, 1.0]));
        assert_eq!(colors[2].1, [0.1, 0.7, 0.2, 1.0]);
    }
}
