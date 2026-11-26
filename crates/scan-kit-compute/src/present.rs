//! Analysis views. The scene comes from `scan-kit-io` and is colored here.
//!
//! `open_plot` packs it for the desktop, which draws it in the webview.
//! `run_view` renders the same plot offscreen and returns the base64 frame that
//! MCP and the view tests use.

use std::path::{Path, PathBuf};

use scan_kit_core::Cancel;
use scan_kit_core::{is_session, Series, ToolKind, ToolSpec};
use serde_json::{json, Value};

use scan_kit_plot::{Plot, PlotInput};

const TOOLS: &[ToolSpec] = &[
    ToolSpec {
        name: "scan_kit_run_view",
        summary: "Build an analysis view and return one RGBA frame plus its controls.",
        kind: ToolKind::Workflow,
    },
    ToolSpec {
        name: "scan_kit_write_dose",
        summary: "Run the patient Monte Carlo and write the dose as an RTDOSE with provenance.",
        kind: ToolKind::Workflow,
    },
    ToolSpec {
        name: "scan_kit_write_report",
        summary: "Run the patient Monte Carlo and write the QA report as a PDF with a DVH CSV.",
        kind: ToolKind::Workflow,
    },
];

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
                "session_ids": { "type": "array", "items": { "type": "string" }, "maxItems": 5 },
                "options": { "type": "object" },
                "width": { "type": "integer", "minimum": 1, "maximum": 8192 },
                "height": { "type": "integer", "minimum": 1, "maximum": 8192 },
                "background": { "type": "array", "items": { "type": "number" }, "minItems": 4, "maxItems": 4 },
                "foreground": { "type": "array", "items": { "type": "number" }, "minItems": 4, "maxItems": 4 },
                "palette": { "type": "array", "items": { "type": "array", "items": { "type": "number" }, "minItems": 4, "maxItems": 4 } }
            },
            "required": ["view", "path", "session_ids"],
            "additionalProperties": false
        }),
        "scan_kit_write_dose" => json!({
            "type": "object",
            "properties": {
                "path": { "type": "string" },
                "session_ids": { "type": "array", "items": { "type": "string" }, "maxItems": 5 },
                "options": { "type": "object" },
                "dest": { "type": "string", "description": "An absolute .dcm write path" },
                "description": { "type": "string" }
            },
            "required": ["path", "session_ids", "dest"],
            "additionalProperties": false
        }),
        "scan_kit_write_report" => json!({
            "type": "object",
            "properties": {
                "path": { "type": "string" },
                "session_ids": { "type": "array", "items": { "type": "string" }, "maxItems": 5 },
                "options": { "type": "object" },
                "dest": { "type": "string", "description": "An absolute .pdf write path" }
            },
            "required": ["path", "session_ids", "dest"],
            "additionalProperties": false
        }),
        _ => json!({ "type": "object", "additionalProperties": false }),
    }
}

pub fn invoke(name: &str, input: &Value) -> Result<Value, String> {
    invoke_flag(name, input, &Cancel::new())
}

/// `invoke` with a cooperative cancel — an MCP client's cancelled request
/// stops the drain instead of finishing detached and still writing the file.
pub fn invoke_flag(name: &str, input: &Value, cancel: &Cancel) -> Result<Value, String> {
    match name {
        "scan_kit_run_view" => {
            let view = text(input, "view")?;
            let path = text(input, "path")?;
            let session_ids = strings(input, "session_ids")?;
            // The volumetric branch renders anything it is handed — the
            // sibling views cap selection at five via analysis_scene, so
            // this stateless tool gets the same bound.
            if session_ids.len() > 5 {
                return Err("at most five sessions can be selected".into());
            }
            let options = object_arg(input, "options")?;
            let width = dim(input, "width", 960)?;
            let height = dim(input, "height", 640)?;
            let background = color4(
                input.get("background"),
                "background",
                [0.11, 0.11, 0.12, 1.0],
            )?;
            let foreground = color4(
                input.get("foreground"),
                "foreground",
                [0.92, 0.92, 0.93, 1.0],
            )?;
            let palette = palette_of(input.get("palette"))?;
            // An absent palette would paint every mark black on the dark
            // default — the desktop always supplies one, the tool need not.
            let palette = if palette.is_empty() {
                vec![foreground]
            } else {
                palette
            };
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
                cancel,
            )
        }
        "scan_kit_write_dose" | "scan_kit_write_report" => {
            let path = text(input, "path")?;
            let session_ids = strings(input, "session_ids")?;
            if session_ids.len() > 5 {
                return Err("at most five sessions can be selected".into());
            }
            let dest = text(input, "dest")?;
            let mut options = object_arg(input, "options")?;
            if name == "scan_kit_write_dose" {
                let description = match input.get("description") {
                    None | Some(Value::Null) => "scan-kit MC",
                    Some(value) => value.as_str().ok_or("description must be a string")?,
                };
                options["description"] = json!(description);
            }
            // A write tool's dest is a file-write primitive — confine it to
            // the tool's own extension inside a real directory, like the
            // plan .csv and session .zip checks.
            let ext = if name == "scan_kit_write_dose" {
                ".dcm"
            } else {
                ".pdf"
            };
            if !dest.to_ascii_lowercase().ends_with(ext) || !Path::new(dest).is_absolute() {
                return Err(format!("dest must be an absolute {ext} path"));
            }
            options["dest"] = json!(dest);
            let view = if name == "scan_kit_write_dose" {
                "write_dose"
            } else {
                "write_report"
            };
            let bytes =
                crate::task::drain_task(view, Path::new(path), &session_ids, &options, cancel)?;
            serde_json::from_slice(&bytes).map_err(|err| format!("export result: {err}"))
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
    cancel: &Cancel,
) -> Result<Value, String> {
    let bytes = open_plot(
        view,
        root,
        session_ids,
        options,
        background,
        foreground,
        palette,
        cancel,
    )?;
    let header = scan_kit_plot::plot_header(&bytes)?;
    if header.reuses_lines() {
        // Reuse payloads carry only the line suffix — the batch indices
        // address a buffer only the previous plot holds, and this path is
        // stateless.
        return Err("payload needs the previous plot's line buffer".into());
    }
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
    cancel: &Cancel,
) -> Result<Vec<u8>, String> {
    // Task views never produce plot bytes — and ExportStage/CopyStage write files
    // as a side effect, so letting `scan_kit_run_view` (a READ_ONLY MCP tool)
    // forward them here would bypass the --allow-effects gate. The desktop
    // exports reach the same stages through `start_task` instead. slot_of is
    // the single taxonomy — a hand-list here would silently drift from it.
    if !matches!(crate::task::slot_of(view), crate::task::Slot::Plot) {
        return Err(format!("{view} is a task view, not a plot"));
    }
    crate::task::drain_plot(
        view,
        root,
        session_ids,
        options,
        background,
        foreground,
        palette,
        cancel,
    )
}

pub(crate) fn apply_palette(scene: &mut scan_kit_core::PlotScene, palette: &[[f32; 4]]) {
    if palette.is_empty() {
        for panel in &mut scene.panels {
            for series in &mut panel.series {
                if let Series::Polyline { color, .. }
                | Series::Points { color, .. }
                | Series::Bars { color, .. }
                | Series::Rects { color, .. }
                | Series::Triangles { color, .. } = series
                {
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
                    if alpha == 0.0 {
                        let color = palette[index.saturating_sub(1) % palette.len()];
                        *slot = [color[0], color[1], color[2], 1.0];
                        // A leading alpha-0 series claims palette[0] at full
                        // alpha — without the advance it would paint 0.05.
                        index += usize::from(index == 0);
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
                Series::Guide { .. } | Series::Cloud { .. } => {}
            }
        }
    }
    if !palette.is_empty() && !scene.volume.values.is_empty() {
        // session_colors indexes volume.sessions — a palette short of the
        // session list leaves later cubes on the decoder's fallback tint
        // while their panel marks wrap. Wrap here too.
        let count = scene.volume.sessions.len().max(palette.len());
        scene.volume.session_colors = (0..count)
            .map(|index| palette[index % palette.len()])
            .collect();
    }
}

/// The `*_dvh.csv` path that pairs with a report written at `dest` — shared
/// so the cancel path removes the same artifact the writer commits.
pub(crate) fn dvh_sidecar(dest: &Path) -> PathBuf {
    dest.with_file_name(format!(
        "{}_dvh.csv",
        dest.file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("report")
    ))
}

/// `scan_kit_write_report`: the QA report `scan_kit.qa.report.write_report`
/// produced — the patient pipeline's metrics as a PDF, with the DVH curves in
/// a `*_dvh.csv` sidecar. The task stage runs the transport; this is the tail.
pub(crate) fn finish_patient_report(
    inputs: &scan_kit_io::patient_view::PatientInputs,
    run: &scan_kit_io::patient_view::Transport,
    session_ids: &[String],
    options: &Value,
    dest: &str,
    gpu: &str,
) -> Result<Value, String> {
    let Some(result) = &run.ran else {
        return Err("the Monte Carlo transport did not run".into());
    };
    let mut provenance = scan_kit_io::patient_view::provenance(inputs, result, session_ids, gpu);
    // A missing/short let_d falls back to physical dose — the artifact must
    // say so, not claim the RBE model the run never applied.
    provenance["rbe_effective"] = json!(run.effective);
    let metrics = scan_kit_io::patient_view::metrics(inputs, run, options);
    let title = format!("Patient QA — {}", inputs.study.plan_label);
    let rbe = scan_kit_io::patient_view::rbe_label(inputs.rbe_key);
    let (report, curves) = scan_kit_io::report::patient_report(
        &title,
        &metrics,
        rbe,
        run.effective,
        dvh_figure(&metrics, run.effective)?,
        provenance,
    );
    let dest = Path::new(dest);
    // Build the CSV before writing the PDF - a DVH error must not leave a
    // fresh PDF beside the report the task calls failed.
    let csv = scan_kit_io::report::dvh_csv(&curves, run.effective)?;
    let pdf = scan_kit_io::report::write_pdf(&report, dest)?;
    let csv_path = dvh_sidecar(dest);
    let mut out = json!({
        "pdf": pdf.display().to_string(),
        "provenance": report.provenance,
    });
    if !csv.is_empty() {
        // write_pdf already replaced any pre-existing PDF at dest, so
        // deleting the new one is not a rollback — it compounds the failure
        // by destroying the only artifact left. Leave it and name the
        // incomplete pair; the export still reports Err.
        if let Err(err) = scan_kit_core::atomic_write(&csv_path, csv.as_bytes()) {
            return Err(format!(
                "{}: {err}; {} was written without its sidecar",
                csv_path.display(),
                pdf.display()
            ));
        }
        out["dvh_csv"] = json!(csv_path.display().to_string());
    } else {
        // A curve-less export must not leave a stale sidecar pairing with
        // the new PDF - the pair reads as one artifact.
        if let Err(err) = std::fs::remove_file(&csv_path) {
            if err.kind() != std::io::ErrorKind::NotFound {
                return Err(format!(
                    "could not remove a stale {} ({err}) beside the new report",
                    csv_path.display()
                ));
            }
        }
    }
    Ok(out)
}

/// The DVH panel rendered for the report — white paper, dark ink. The caption
/// names each curve's color because the plot has no legend.
fn dvh_figure(
    metrics: &scan_kit_io::patient_view::PatientMetrics,
    effective: bool,
) -> Result<Option<scan_kit_io::report::Figure>, String> {
    const COLORS: &[([f32; 4], &str)] = &[
        ([0.12, 0.47, 0.71, 1.0], "blue"),
        ([1.0, 0.50, 0.05, 1.0], "orange"),
        ([0.17, 0.63, 0.17, 1.0], "green"),
        ([0.84, 0.15, 0.16, 1.0], "red"),
        ([0.58, 0.40, 0.74, 1.0], "purple"),
        ([0.55, 0.34, 0.29, 1.0], "brown"),
    ];
    if metrics.structures.is_empty() {
        return Ok(None);
    }
    let mut scene = scan_kit_core::PlotScene::empty("DVH");
    let mut series = Vec::new();
    let mut names = Vec::new();
    let mut xmax = 0.0f32;
    for (index, structure) in metrics.structures.iter().enumerate() {
        let (color, name) = COLORS[index % COLORS.len()];
        // Past the palette the colors repeat — the caption is the only
        // legend, so name the repeat or two curves read as one.
        names.push(if index < COLORS.len() {
            format!("{} ({name})", structure.name)
        } else {
            format!("{} ({name}, repeat)", structure.name)
        });
        let cumulative = structure.dvh.cumulative();
        xmax = structure.dvh.edges.iter().copied().fold(xmax, f32::max);
        series.push(Series::Polyline {
            xs: structure.dvh.edges.to_vec(),
            ys: cumulative,
            color,
            thickness: 1.5,
        });
    }
    scene.panels.push(scan_kit_core::Panel {
        title: "DVH".into(),
        y_label: "Volume (fraction)".into(),
        x_label: if effective {
            "Dose (Gy(RBE))".into()
        } else {
            // A physical-dose run reports Gy like its CSV and stats tables.
            "Dose (Gy)".into()
        },
        xmin: 0.0,
        xmax,
        ymin: 0.0,
        ymax: 1.0,
        series,
        x_labels: Vec::new(),
        equal: false,
    });
    // A render failure must fail the export — .ok()? would silently ship a
    // report missing its DVH panel.
    let rgba = crate::render_plot(&scene, 880, 420, [1.0, 1.0, 1.0, 1.0], [0.1, 0.1, 0.1, 1.0])
        .map_err(|err| err.to_string())?;
    Ok(Some(scan_kit_io::report::Figure {
        rgba,
        width: 880,
        height: 420,
        caption: format!("DVH curves: {}", names.join(", ")),
    }))
}

fn text<'a>(input: &'a Value, key: &str) -> Result<&'a str, String> {
    match input.get(key) {
        Some(Value::String(value)) => Ok(value),
        Some(_) => Err(format!("{key} must be a string")),
        None => Err(format!("{key} is required")),
    }
}

fn strings(input: &Value, key: &str) -> Result<Vec<String>, String> {
    let array = match input.get(key) {
        None => return Err(format!("{key} is required")),
        Some(value) => value
            .as_array()
            .ok_or_else(|| format!("{key} must be an array"))?,
    };
    array
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(str::to_owned)
                .ok_or_else(|| format!("{key} must be strings"))
        })
        .collect()
}

fn dim(input: &Value, key: &str, fallback: u32) -> Result<u32, String> {
    match input.get(key) {
        Some(value) => {
            let dim = u32::try_from(
                value
                    .as_u64()
                    .ok_or_else(|| format!("{key} must be an integer"))?,
            )
            .map_err(|_| format!("{key} is out of range"))?;
            // The same bound Plot::apply clamps to — a huge dim would ask the
            // renderer for an oversized frame buffer; 0 yields an empty view.
            if dim == 0 || dim > 8192 {
                return Err(format!("{key} is out of range"));
            }
            Ok(dim)
        }
        None => Ok(fallback),
    }
}

fn object_arg(input: &Value, key: &str) -> Result<Value, String> {
    match input.get(key) {
        Some(value) if value.is_object() => Ok(value.clone()),
        // null means "defaults" to JSON callers — same convention the
        // analysis tools already accept.
        Some(Value::Null) | None => Ok(json!({})),
        Some(_) => Err(format!("{key} must be an object")),
    }
}

fn color4(value: Option<&Value>, key: &str, fallback: [f32; 4]) -> Result<[f32; 4], String> {
    let Some(value) = value else {
        return Ok(fallback);
    };
    let items = value
        .as_array()
        .ok_or_else(|| format!("{key} must be an array"))?;
    // A short array must not blend user channels with the fallback —
    // [0.5] would silently produce [0.5, 0.11, 0.12, 1.0].
    if items.len() != 4 {
        return Err(format!("{key} must have 4 channels"));
    }
    let mut out = fallback;
    for (index, item) in items.iter().enumerate() {
        // Every sibling arg parser errors on a mistyped element — a bad
        // channel must not silently keep the fallback color.
        out[index] = item
            .as_f64()
            .map(|n| n as f32)
            .filter(|n| n.is_finite())
            .ok_or_else(|| format!("{key} must be finite numbers"))?;
    }
    Ok(out)
}

fn palette_of(value: Option<&Value>) -> Result<Vec<[f32; 4]>, String> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    let rows = value
        .as_array()
        .ok_or_else(|| "palette must be an array".to_owned())?;
    if rows.iter().any(|row| !row.is_array()) {
        return Err("palette must contain [r, g, b, a] arrays".into());
    }
    rows.iter()
        .map(|row| color4(Some(row), "palette", [0.9, 0.9, 0.9, 1.0]))
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
    fn no_structures_is_not_a_render_failure() {
        // The empty case yields no figure — a GPU error must not be
        // conflated with it.
        let metrics = scan_kit_io::patient_view::PatientMetrics {
            matches: Vec::new(),
            dropped: Vec::new(),
            structures: Vec::new(),
            gamma: None,
        };
        assert!(super::dvh_figure(&metrics, true).unwrap().is_none());
    }

    #[test]
    fn mistyped_args_error_instead_of_falling_back() {
        let base = json!({"view": "bins", "path": ".", "session_ids": []});
        for (key, value) in [
            ("width", json!("640")),
            ("width", json!(65536)),
            ("height", json!(true)),
            ("options", json!(5)),
            ("background", json!("red")),
            ("background", json!([0.11, "x", 0.12, 1])),
            ("palette", json!(7)),
            ("palette", json!([[0.9, 0.9, 0.9, "x"]])),
        ] {
            let mut input = base.clone();
            input[key] = value;
            let err = invoke("scan_kit_run_view", &input).unwrap_err();
            assert!(err.contains(key), "{key}: {err}");
        }
        // The write tools apply the same rule to options and description.
        let err = invoke(
            "scan_kit_write_dose",
            &json!({"path": ".", "session_ids": [], "dest": "d.dcm", "description": 5}),
        )
        .unwrap_err();
        assert!(err.contains("description"), "{err}");
    }

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
            &[[0.8, 0.1, 0.1, 1.0], [0.1, 0.1, 0.8, 1.0]],
            &Cancel::new(),
        )
        .unwrap();
        assert!(frame["title"].as_str().unwrap().contains("vs"));
        assert_eq!(frame["width"], 320);
        // Positive control: decode the frame and demand series paint — a
        // blank frame base64s to the same bulk, so length proves nothing.
        let rgba = decode64(frame["rgba_base64"].as_str().unwrap());
        assert_eq!(rgba.len(), 320 * 180 * 4);
        assert!(
            rgba.chunks_exact(4)
                .any(|px| px[0] > 150 && px[1] < 90 && px[2] < 90),
            "no red series pixel — the frame painted nothing"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    fn decode64(text: &str) -> Vec<u8> {
        fn value(byte: u8) -> u32 {
            match byte {
                b'A'..=b'Z' => (byte - b'A') as u32,
                b'a'..=b'z' => (byte - b'a' + 26) as u32,
                b'0'..=b'9' => (byte - b'0' + 52) as u32,
                b'+' => 62,
                b'/' => 63,
                _ => 0,
            }
        }
        let bytes: Vec<u8> = text.bytes().filter(|byte| *byte != b'=').collect();
        let mut out = Vec::with_capacity(bytes.len() * 3 / 4);
        for quad in bytes.chunks(4) {
            let acc: u32 = quad
                .iter()
                .fold(0u32, |acc, byte| (acc << 6) | value(*byte));
            out.extend_from_slice(&acc.to_be_bytes()[1..]);
        }
        let padding = text.bytes().filter(|byte| *byte == b'=').count();
        out.truncate((text.len() / 4) * 3 - padding);
        out
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
            &Cancel::new(),
        )
        .unwrap();
        let header = scan_kit_plot::plot_header(&opened).unwrap();
        assert!(header.title.contains("vs"));
        assert!(header.panels.iter().all(|panel| panel.series.is_empty()));
        let mut plot = Plot::from_payload(&opened).unwrap();
        // 80x60 paints nothing at all — even chrome — so a size assertion
        // alone is vacuous. Draw big enough that series paint, then demand it.
        let frame = plot.draw(320, 180, &PlotInput::default()).unwrap();
        assert_eq!(frame.rgba.len(), 320 * 180 * 4);
        // Positive control: the series color is [0.8, 0.2, 0.2] — a blank
        // frame (every mark dropped upstream) must fail, not pass on size.
        assert!(
            frame
                .rgba
                .chunks_exact(4)
                .any(|px| px[0] > 150 && px[1] < 100 && px[2] < 100),
            "no red series pixel — the frame painted nothing"
        );
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
