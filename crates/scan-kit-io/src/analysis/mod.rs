//! Analysis view workflows. Each one loads columns and returns a plot scene.
//! Pixels are rendered later by `scan-kit-compute`.

use std::collections::BTreeMap;
use std::path::Path;

use scan_kit_core::{finite_minmax, resolve_concept_column, Control, Panel, PlotScene, Series};
use serde_json::{json, Value};

pub(super) use super::discover;
pub(super) use super::marks::{contour_bands, control, flag, labeled, pick, text};
pub(super) use super::tables::{slice_table, spot_table, timeslice_metric, timeslice_signals};
mod distribution;
mod session_log;
mod timeline;

/// A session series. `apply_palette` replaces the RGB and keeps this alpha.
const MARK: [f32; 4] = [0.0, 0.0, 0.0, 1.0];
/// A companion of the previous series. Alpha 0 copies that session's color.
const LINKED: [f32; 4] = [0.0, 0.0, 0.0, 0.0];
const GUIDE_COLOR: [f32; 4] = [0.62, 0.62, 0.62, 0.9];

pub fn analysis_scene(
    view: &str,
    root: &Path,
    session_ids: &[String],
    options: &Value,
) -> Result<PlotScene, String> {
    if session_ids.is_empty() {
        return Err("select a session first".to_owned());
    }
    if session_ids.len() > 5 {
        return Err("at most five sessions can be selected".to_owned());
    }
    match view {
        "distribution" => Ok(distribution::distribution(root, session_ids, options)),
        "binned_summary" => Ok(binned_summary(root, session_ids, options)),
        "timeslice_replay" => Ok(timeline::replay(root, session_ids, options)),
        "ic_hv_transient" => Ok(timeline::hv_transient(root, session_ids)),
        "session_log_compare" => Ok(session_log::session_log(root, session_ids)),
        "dose_volume" => Ok(crate::dose_view::dose_volume(
            root,
            session_ids,
            options,
            None,
        )),
        _ => Err(format!("unknown view {view}")),
    }
}

pub fn channel_catalog(root: &Path, session_id: &str) -> Vec<String> {
    session_channels(root, session_id)
        .iter()
        .filter(|(name, values)| channel_key(name) && values.iter().any(|value| value.is_finite()))
        .map(|(name, _)| name.clone())
        .collect()
}

pub fn load_timeslice_columns(root: &Path, session_id: &str) -> Value {
    let columns = load_timeslice(root, session_id);
    let mut listed = Vec::new();
    for (name, values) in columns.iter() {
        listed.push(json!({ "name": name, "len": values.len() }));
    }
    let energy = energy_lookup(root, session_id);
    json!({
        "session_id": session_id,
        "columns": listed,
        "energy_layers": energy.len(),
    })
}

fn binned_summary(root: &Path, session_ids: &[String], options: &Value) -> PlotScene {
    super::binned::binned_summary(root, session_ids, options)
}

fn scene(title: &str, panels: Vec<Panel>, controls: Vec<Control>) -> PlotScene {
    PlotScene {
        title: title.into(),
        panels,
        controls,
        table: None,
        columns: 0,
        column_weights: Vec::new(),
        row_weights: Vec::new(),
        side: 0,
    }
}

fn panel(title: String, xmin: f32, xmax: f32, ymin: f32, ymax: f32, series: Vec<Series>) -> Panel {
    let xmax = if (xmax - xmin).abs() < 1e-3 {
        xmin + 1.0
    } else {
        xmax
    };
    let ymax = if (ymax - ymin).abs() < 1e-3 {
        ymin + 1.0
    } else {
        ymax
    };
    Panel {
        title,
        y_label: String::new(),
        x_label: String::new(),
        xmin,
        xmax,
        ymin,
        ymax,
        series,
        x_labels: Vec::new(),
        equal: false,
    }
}

fn stroke(xs: Vec<f32>, ys: Vec<f32>, linked: bool) -> Series {
    Series::Polyline {
        xs,
        ys,
        color: if linked { LINKED } else { MARK },
        thickness: 1.5,
    }
}

fn guide(xs: Vec<f32>, ys: Vec<f32>) -> Series {
    Series::Guide {
        xs,
        ys,
        color: GUIDE_COLOR,
        thickness: 1.0,
    }
}

fn placed(title: String, series: Vec<Series>) -> Panel {
    let (xmin, xmax, ymin, ymax) = series_span(&series);
    panel(title, xmin, xmax, ymin, ymax, series)
}

fn drew_line(series: &[Series]) -> bool {
    series.iter().any(|item| {
        matches!(item, Series::Polyline { ys, .. } if ys.iter().any(|value| value.is_finite()))
    })
}

fn finite_col<'a>(table: &'a BTreeMap<String, Vec<f32>>, key: &str) -> Option<&'a [f32]> {
    col(table, key).filter(|values| values.iter().any(|value| value.is_finite()))
}

fn span(values: &[f32]) -> (f32, f32) {
    let Some((lo, hi)) = finite_minmax(values) else {
        return (0.0, 1.0);
    };
    if (hi - lo).abs() < 1e-4 {
        (lo - 0.5, hi + 0.5)
    } else {
        (lo, hi)
    }
}

fn load_csv(root: &Path, session_id: &str, name: &str) -> BTreeMap<String, Vec<f32>> {
    discover::read_session_file(root, session_id, name)
        .and_then(|bytes| numeric_columns(&bytes).ok())
        .unwrap_or_default()
}

fn load_timeslice(root: &Path, session_id: &str) -> super::tables::Table {
    super::tables::merged_timeslice(root, session_id)
}

fn energy_lookup(root: &Path, session_id: &str) -> Vec<f32> {
    col(&load_csv(root, session_id, "input_map.csv"), "energy")
        .unwrap_or(&[])
        .to_vec()
}

fn col<'a>(columns: &'a BTreeMap<String, Vec<f32>>, concept: &str) -> Option<&'a [f32]> {
    let names: Vec<String> = columns.keys().cloned().collect();
    let name = resolve_concept_column(&names, concept).or_else(|| {
        names
            .iter()
            .find(|name| name.eq_ignore_ascii_case(concept))
            .map(String::as_str)
    })?;
    columns.get(name).map(Vec::as_slice)
}

fn numeric_columns(bytes: &[u8]) -> Result<BTreeMap<String, Vec<f32>>, String> {
    Ok(super::tables::read_sheet(bytes).num)
}

fn session_text(root: &Path, session_id: &str, name: &str) -> String {
    discover::read_session_file(root, session_id, name)
        .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
        .unwrap_or_default()
}

fn session_channels(root: &Path, session: &str) -> super::tables::Table {
    timeslice_signals(root, session)
}

pub(super) fn channel_key(name: &str) -> bool {
    !matches!(name, "energy" | "beam_on" | "beam_on_time" | "spot_time")
}

fn series_span(series: &[Series]) -> (f32, f32, f32, f32) {
    let mut xmin = f32::MAX;
    let mut xmax = f32::MIN;
    let mut ymin = f32::MAX;
    let mut ymax = f32::MIN;
    for item in series {
        let (xs, ys) = match item {
            Series::Polyline { xs, ys, .. }
            | Series::Points { xs, ys, .. }
            | Series::Guide { xs, ys, .. } => (xs, ys),
            _ => continue,
        };
        for (x, y) in xs.iter().zip(ys) {
            if x.is_finite() && y.is_finite() {
                xmin = xmin.min(*x);
                xmax = xmax.max(*x);
                ymin = ymin.min(*y);
                ymax = ymax.max(*y);
            }
        }
    }
    if xmin > xmax {
        (0.0, 1.0, 0.0, 1.0)
    } else {
        (xmin, xmax, ymin, ymax)
    }
}
#[cfg(test)]
mod tests {
    use scan_kit_core::{Family, SESSION};

    use super::distribution::{distribution_limits, reference_ring};
    use super::timeline::{robust_span, trace};
    use super::*;

    fn write_session(root: &Path) {
        let session = root.join("sess");
        std::fs::create_dir_all(&session).unwrap();
        std::fs::write(
            session.join("input_map.csv"),
            "energy,charge_req,position_x,position_y\n70,1,0,0\n90,2,5,1\n110,3,10,2\n",
        )
        .unwrap();
        std::fs::write(
            session.join("spot_data.csv"),
            "ic1_total_dose,ic2_total_dose,position_x,position_y\n1.1,1.0,0,0\n1.8,2.1,5,1\n3.2,2.7,10,2\n",
        )
        .unwrap();
        let mut timeslice = String::from(
            "r_ic1_current_dose,rci_in_trigger,ic1_peak_amplitude_x,ic1_peak_amplitude_y,ic2_peak_amplitude_x,ic2_peak_amplitude_y,position_error_x,field_x,field_y,c_x,r_xV\n",
        );
        for i in 0..32 {
            let on = if (8..24).contains(&i) { 1 } else { 0 };
            let current = if on == 1 {
                2.0
            } else {
                (24 - i).max(0) as f32 * 0.05
            };
            let cmd = i as f32 * 0.1;
            timeslice.push_str(&format!(
                "{current},{on},{current},{current},1,1,{err},0.{i},0.2,{cmd},{read}\n",
                err = (i as f32) * 0.01,
                read = cmd + 0.05
            ));
        }
        std::fs::write(
            session.join("000_timeslice_data_device_units.csv"),
            timeslice,
        )
        .unwrap();
        let hv = session.join("ic_hv_toggle");
        std::fs::create_dir_all(&hv).unwrap();
        std::fs::write(
            hv.join("IC1_HCC.csv"),
            "time,current\n0,0\n1,10\n2,2\n3,0\n",
        )
        .unwrap();
        std::fs::write(
            session.join("SessionLogFile.log"),
            "2026-01-01 00:00:00,000 INFO [dcs] START MAP FOR LAYER: 1 - TIMELINE(begin): T=0.1s\n2026-01-01 00:00:01,000 ERROR [dcs] fault\n",
        )
        .unwrap();
    }

    #[test]
    fn sk_req_011_timeslice_load_reports_columns() {
        let root = std::env::temp_dir().join(format!("scan-kit-ts-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        write_session(&root);
        let loaded = load_timeslice_columns(&root, "sess");
        assert!(loaded["columns"].as_array().unwrap().len() >= 2);
        assert!(!channel_catalog(&root, "sess").is_empty());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn sk_req_024_session_log_table() {
        let root = std::env::temp_dir().join(format!("scan-kit-log-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        write_session(&root);
        let scene =
            analysis_scene("session_log_compare", &root, &["sess".into()], &json!({})).unwrap();
        let table = scene.table.unwrap();
        assert!(table.rows.iter().any(|row| row[1] == "Errors"));
        assert!(table.rows.iter().any(|row| row[1] == "Timeline"));
        let _ = std::fs::remove_dir_all(&root);
    }

    fn scene_of(view: &str) -> PlotScene {
        opened(view, view, &json!({}))
    }

    fn opened(view: &str, tag: &str, options: &Value) -> PlotScene {
        use std::sync::atomic::{AtomicU64, Ordering};
        // Parallel tests share a process id, so the folder also needs a call counter.
        static SCENE: AtomicU64 = AtomicU64::new(0);
        let n = SCENE.fetch_add(1, Ordering::Relaxed);
        let root =
            std::env::temp_dir().join(format!("scan-kit-{view}-{tag}-{n}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        write_session(&root);
        let scene = analysis_scene(view, &root, &["sess".into()], options).unwrap();
        let _ = std::fs::remove_dir_all(&root);
        scene
    }

    fn has_kind(scene: &PlotScene, kind: &str) -> bool {
        scene.panels.iter().any(|panel| {
            panel.series.iter().any(|series| {
                matches!(
                    (kind, series),
                    ("line", Series::Polyline { .. })
                        | ("bars", Series::Bars { .. })
                        | ("points", Series::Points { .. })
                        | ("rects", Series::Rects { .. })
                        | ("triangles", Series::Triangles { .. })
                        | ("heat", Series::Heatmap { .. })
                )
            })
        })
    }

    #[test]
    fn sk_req_015_distribution_exposes_spot_modes() {
        let scene = scene_of("distribution");
        assert!(has_kind(&scene, "points"));
        assert!(has_kind(&scene, "bars"));
        assert_eq!(scene.row_weights, vec![2.0, 1.0, 1.0]);
        assert!(scene.panels.iter().any(|panel| {
            panel.equal
                && panel.title.is_empty()
                && panel.x_label == "Plan X (mm)"
                && panel.y_label == "Plan Y (mm)"
        }));
        assert!(scene.panels.iter().any(|panel| {
            !panel.equal
                && panel.x_label == "Plan X (mm)"
                && panel.y_label == "Probability (%)"
                && panel
                    .series
                    .iter()
                    .any(|series| matches!(series, Series::Polyline { .. }))
        }));
        assert!(scene.controls.iter().any(|control| control.id == "xy"
            && control.label == "XY"
            && control.value == "Position (mm)"
            && control
                .options
                .iter()
                .any(|option| option == "Amplifier (V)")
            && control.options.iter().any(|option| option == "Probe (G)")));
        assert!(scene
            .controls
            .iter()
            .any(|control| control.id == "hist_bins" && control.value == "Auto"));
        assert!(scene
            .controls
            .iter()
            .any(|control| control.id == "plan" && control.value == "On"));
        assert!(scene
            .controls
            .iter()
            .any(|control| control.id == "draw" && control.value == "Scatter"));
        assert!(scene.controls.iter().any(|control| {
            control.id == "segments"
                && scan_kit_core::parse_segments(&control.value).is_ok_and(|items| {
                    items.iter().any(|item| {
                        matches!(
                            item,
                            scan_kit_core::Segment::Beam {
                                state: scan_kit_core::BeamGate::Both
                            }
                        )
                    })
                })
        }));
        assert!(scene
            .controls
            .iter()
            .any(|control| control.id == "source" && control.value == "Spot"));
        assert!(scene.controls.iter().any(|control| control.id == "draw"
            && control.options.iter().any(|option| option == "Contour")));
        let root = std::env::temp_dir().join(format!("scan-kit-plan-first-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let session = root.join("sess");
        std::fs::create_dir_all(&session).unwrap();
        std::fs::write(
            session.join("input_map.csv"),
            "energy,charge_req,position_x,position_y\n70,1,0,0\n",
        )
        .unwrap();
        std::fs::write(
            session.join("spot_data.csv"),
            "ic1_x_spot,ic1_y_spot,ic2_x_spot,ic2_y_spot\n1,2,3,4\n",
        )
        .unwrap();
        let ordered = analysis_scene("distribution", &root, &["sess".into()], &json!({})).unwrap();
        let checks: Vec<_> = ordered
            .controls
            .iter()
            .filter(|control| matches!(control.id.as_str(), "plan" | "ic1" | "ic2"))
            .map(|control| control.id.as_str())
            .collect();
        assert_eq!(checks, ["plan", "ic1", "ic2"]);
        let labels: Vec<_> = ordered
            .panels
            .iter()
            .filter(|panel| panel.equal)
            .map(|panel| panel.x_label.as_str())
            .collect();
        assert_eq!(labels, ["Plan X (mm)", "IC1 X (mm)", "IC2 X (mm)"]);
        let _ = std::fs::remove_dir_all(&root);
        assert!(scene
            .panels
            .iter()
            .filter(|panel| panel.equal)
            .all(|panel| {
                panel.title.is_empty() && !panel.x_label.is_empty() && !panel.y_label.is_empty()
            }));
    }

    #[test]
    fn distribution_timeslice_keeps_the_plan_column() {
        let root = std::env::temp_dir().join(format!("scan-kit-slice-plan-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let session = root.join("sess");
        std::fs::create_dir_all(session.join("layer-0/run-0")).unwrap();
        std::fs::write(
            session.join("input_map.csv"),
            "energy,layer_id,spot_no,position_x,position_y\n70,1,5,3,4\n",
        )
        .unwrap();
        std::fs::write(
            session.join("layer-0/run-0/timeslice_data_device_units.csv"),
            "layer_id,spot_no,rci_in_trigger,r_ic1_x_position\n1,5,1,64\n1,5,1,64\n",
        )
        .unwrap();
        let scene = analysis_scene(
            "distribution",
            &root,
            &["sess".into()],
            &json!({"source": "Timeslice"}),
        )
        .unwrap();
        assert!(scene
            .controls
            .iter()
            .any(|control| control.id == "plan" && control.value == "On"));
        assert!(scene.panels.iter().any(|panel| {
            panel.equal && panel.x_label == "Plan X (mm)" && panel.y_label == "Plan Y (mm)"
        }));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn distribution_position_leaves_room_around_the_spots() {
        let (lo, hi) = distribution_limits("position", &[0.0, 10.0, 0.0, 10.0]);
        assert!((lo - -0.6).abs() < 1.0e-3, "{lo}");
        assert!((hi - 10.6).abs() < 1.0e-3, "{hi}");
    }

    #[test]
    fn amplifier_and_probe_are_xy_clouds() {
        let root = std::env::temp_dir().join(format!("scan-kit-amp-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let session = root.join("sess");
        std::fs::create_dir_all(&session).unwrap();
        std::fs::write(session.join("input_map.csv"), "energy\n70\n").unwrap();
        std::fs::write(
            session.join("000_timeslice_data_device_units.csv"),
            "rci_in_trigger,field_x,field_y,c_x,c_y,r_xV,r_yV\n1,0.1,0.2,1,2,1.2,2.4\n1,0.3,0.4,1,2,1.1,2.3\n",
        )
        .unwrap();
        let probe = analysis_scene(
            "distribution",
            &root,
            &["sess".into()],
            &json!({"mode": "Probe"}),
        )
        .unwrap();
        assert!(probe
            .panels
            .iter()
            .any(|panel| { panel.equal && panel.x_label == "X (G)" && panel.y_label == "Y (G)" }));
        assert!(probe
            .controls
            .iter()
            .all(|control| control.id != "grain" && control.id != "ic1" && control.id != "plan"));
        assert!(probe.controls.iter().any(|control| {
            control.id == "segments"
                && scan_kit_core::parse_segments(&control.value).is_ok_and(|items| {
                    items.iter().any(|item| {
                        matches!(
                            item,
                            scan_kit_core::Segment::Beam {
                                state: scan_kit_core::BeamGate::On
                            }
                        )
                    })
                })
        }));
        let amplifier = analysis_scene(
            "distribution",
            &root,
            &["sess".into()],
            &json!({"mode": "Amplifier"}),
        )
        .unwrap();
        assert!(amplifier.panels.iter().any(|panel| {
            panel.equal && panel.x_label == "X Error (V)" && panel.y_label == "Y Error (V)"
        }));
        assert!(amplifier.panels.iter().any(|panel| {
            !panel.equal
                && panel.x_label == "X Error (V)"
                && panel
                    .series
                    .iter()
                    .any(|series| matches!(series, Series::Bars { .. }))
        }));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn position_error_ring_has_radius_one_mm() {
        let Series::Guide { xs, ys, .. } = reference_ring() else {
            panic!("ring");
        };
        assert!(xs.len() >= 32);
        for (x, y) in xs.iter().zip(ys) {
            let radius = (x * x + y * y).sqrt();
            assert!((radius - 1.0).abs() < 1.0e-4, "{radius}");
        }
    }

    #[test]
    fn distribution_density_overlays_sessions_on_one_column() {
        let root = std::env::temp_dir().join(format!("scan-kit-density-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        write_session(&root);
        let from = root.join("sess");
        let to = root.join("sess-b");
        std::fs::create_dir_all(&to).unwrap();
        for name in ["input_map.csv", "spot_data.csv"] {
            std::fs::copy(from.join(name), to.join(name)).unwrap();
        }
        let scene = analysis_scene(
            "distribution",
            &root,
            &["sess".into(), "sess-b".into()],
            &json!({"draw": "Density"}),
        )
        .unwrap();
        let tops: Vec<_> = scene.panels.iter().filter(|panel| panel.equal).collect();
        assert_eq!(scene.columns as usize, tops.len());
        assert!(tops.iter().all(|panel| panel.equal));
        assert!(tops.iter().all(|panel| !panel.title.contains("sess")));
        assert!(tops.iter().any(|panel| {
            panel
                .series
                .iter()
                .filter(|series| matches!(series, Series::Heatmap { ramp: SESSION, .. }))
                .count()
                == 2
        }));
        assert!(scene.panels.iter().any(|panel| {
            !panel.equal
                && panel.x_label.ends_with("X (mm)")
                && panel
                    .series
                    .iter()
                    .any(|series| matches!(series, Series::Bars { .. }))
                && panel
                    .series
                    .iter()
                    .any(|series| matches!(series, Series::Polyline { .. }))
        }));
        let planned = analysis_scene(
            "distribution",
            &root,
            &["sess".into(), "sess-b".into()],
            &json!({"draw": "Density", "plan": "Off"}),
        )
        .unwrap();
        assert!(planned
            .panels
            .iter()
            .all(|panel| panel.x_label != "Plan X (mm)"));
        assert!(scene.controls.iter().all(|control| control.id != "ramp"));
        let one = analysis_scene(
            "distribution",
            &root,
            &["sess".into()],
            &json!({"draw": "Density"}),
        )
        .unwrap();
        let ramp_control = one
            .controls
            .iter()
            .find(|control| control.id == "ramp")
            .unwrap();
        assert_eq!(ramp_control.value, "Turbo");
        assert_eq!(
            ramp_control.options.len(),
            scan_kit_core::choices(Family::Sequential).len()
        );
        assert!(ramp_control.options.iter().any(|option| option == "Gray"));
        assert!(one.panels.iter().any(|panel| {
            panel.series.iter().any(|series| {
                matches!(
                    series,
                    Series::Heatmap {
                        ramp,
                        ..
                    } if *ramp == scan_kit_core::index("turbo")
                )
            })
        }));
        let contour = analysis_scene(
            "distribution",
            &root,
            &["sess".into()],
            &json!({"draw": "Contour"}),
        )
        .unwrap();
        assert!(contour
            .controls
            .iter()
            .any(|control| control.id == "cutoff" && control.value == "5"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn sk_req_017_binned_summary_and_replay_share_the_session() {
        let binned = opened(
            "binned_summary",
            "binned_summary",
            &json!({"metric": "Dose Error (%)"}),
        );
        assert!(has_kind(&binned, "triangles"));
        assert!(binned
            .controls
            .iter()
            .any(|control| control.id == "glyph" && control.value == "Violin"));
        assert!(binned
            .controls
            .iter()
            .any(|control| control.id == "y" && control.value == "Dose Error (%)"));
        let ic1 = binned
            .panels
            .iter()
            .find(|panel| panel.title.starts_with("IC1"))
            .unwrap();
        assert!(ic1.series.iter().any(|series| match series {
            Series::Triangles { ys, .. } => ys.iter().any(|value| (*value - 10.0).abs() < 1e-3),
            _ => false,
        }));
        assert!(ic1.x_labels.iter().any(|label| label == "70"));
        let replay = scene_of("timeslice_replay");
        assert!(has_kind(&replay, "line"));
        assert_eq!(replay.columns, 1);
        assert_eq!(replay.side, 0);
        let y = replay
            .controls
            .iter()
            .find(|control| control.id == "y")
            .unwrap();
        assert_eq!(y.value, "IC Current (nA)");
        assert!(y.labels().contains(&"Amplifier Error (V)"));
        assert!(replay.controls.iter().any(|control| {
            control.id == "ch_ic1_current" && control.kind == "check" && control.value == "On"
        }));
        assert!(replay
            .controls
            .iter()
            .any(|control| control.id == "segments"));
        assert!(replay
            .panels
            .iter()
            .any(|panel| panel.title.is_empty() && panel.y_label == "IC1 (nA)"));
        let (xmin, xmax, y_span) = {
            let time = &replay.panels[0];
            (time.xmin, time.xmax, time.ymax - time.ymin)
        };
        let Series::Polyline { xs, ys, .. } = &replay.panels[0].series[0] else {
            panic!("replay should be one trace");
        };
        assert_eq!(xs.len(), 32);
        assert!(xmin > 0.004 && xmin < 0.009, "{xmin}");
        assert!(xmax > 0.022 && xmax < 0.028, "{xmax}");
        let on: Vec<f32> = ys
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .collect();
        assert_eq!(on.len(), 16, "beam-on rows stay, beam-off rows drop out");
        assert!(y_span < on[0].abs().max(1.0e-6) * 0.2, "{y_span}");
        assert!(on
            .iter()
            .all(|value| (*value - on[0]).abs() <= on[0].abs() * 1e-4));
        let scrub = replay
            .controls
            .iter()
            .find(|control| control.id == "scrub")
            .unwrap();
        assert_eq!(scrub.kind, "scrub");
        let scrub_value: serde_json::Value = serde_json::from_str(&scrub.value).unwrap();
        assert_eq!(scrub_value["on"], false);
        assert_eq!(scrub_value["speed"], 1.0);
        assert_eq!(scrub_value["window"], "second");
        assert!(replay.controls.iter().all(|control| control.id != "x"));
    }

    fn finite_ys(scene: &PlotScene) -> Vec<f32> {
        let mut kept = Vec::new();
        for panel in &scene.panels {
            for series in &panel.series {
                let ys = match series {
                    Series::Polyline { ys, .. } | Series::Points { ys, .. } => ys,
                    _ => continue,
                };
                kept.extend(ys.iter().copied().filter(|value| value.is_finite()));
            }
        }
        kept
    }

    #[test]
    fn scrub_drops_rows_past_the_playhead() {
        let scrub = r#"{"on":true,"at":0.01,"speed":1,"window":"before"}"#;
        let replay = opened("timeslice_replay", "scrub", &json!({ "scrub": scrub }));
        let Series::Polyline { ys, .. } = &replay.panels[0].series[0] else {
            panic!("replay should be one trace");
        };
        let kept = ys.iter().filter(|value| value.is_finite()).count();
        assert_eq!(kept, 16, "time traces keep every beam-on sample");
        assert!(replay.panels[0].xmax < 0.015, "{}", replay.panels[0].xmax);
        assert!(replay.panels[0].xmin > 0.004, "{}", replay.panels[0].xmin);
        assert!(ys[8].is_finite());
        assert!(
            ys[23].is_finite(),
            "rows past the playhead stay in the trace"
        );

        let full = opened(
            "binned_summary",
            "scrub-full",
            &json!({ "y": "IC Current (nA)", "glyph": "Scatter" }),
        );
        let cut = opened(
            "binned_summary",
            "scrub-cut",
            &json!({ "y": "IC Current (nA)", "glyph": "Scatter", "scrub": scrub }),
        );
        let full_n = finite_ys(&full).len();
        let cut_n = finite_ys(&cut).len();
        assert!(full_n > cut_n, "{full_n} vs {cut_n}");
        assert!(cut_n > 0);

        let spread = opened("distribution", "scrub-full", &json!({ "xy": "probe" }));
        let window = opened(
            "distribution",
            "scrub-cut",
            &json!({ "xy": "probe", "scrub": scrub }),
        );
        let spread_n = finite_ys(&spread).len();
        let window_n = finite_ys(&window).len();
        assert!(spread_n > window_n, "{spread_n} vs {window_n}");
        assert!(window_n > 0);
        for scene in [&replay, &cut, &window] {
            let listed = scene
                .controls
                .iter()
                .find(|control| control.id == "segments")
                .unwrap();
            assert!(
                !listed.value.contains("time_s"),
                "the playhead is not a sidebar segment"
            );
        }
    }

    #[test]
    fn replay_filters_beam_off_and_draws_amplifier_with_a_scatter() {
        let off = scan_kit_core::segments_json(&[scan_kit_core::Segment::Beam {
            state: scan_kit_core::BeamGate::Off,
        }]);
        let filtered = opened("timeslice_replay", "beam-off", &json!({ "segments": off }));
        let Series::Polyline { ys, .. } = &filtered.panels[0].series[0] else {
            panic!("replay should be one trace");
        };
        let off_rows: Vec<f32> = ys
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .collect();
        assert_eq!(off_rows.len(), 16);
        assert!(off_rows
            .windows(2)
            .any(|pair| (pair[0] - pair[1]).abs() > 1.0));

        let root = std::env::temp_dir().join(format!("scan-kit-replay-amp-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let session = root.join("sess");
        std::fs::create_dir_all(&session).unwrap();
        std::fs::write(
            session.join("input_map.csv"),
            "energy,charge_req,position_x,position_y\n70,1,0,0\n",
        )
        .unwrap();
        std::fs::write(
            session.join("000_timeslice_data_device_units.csv"),
            "rci_in_trigger,c_x,r_xV,c_y,r_yV\n1,1.0,1.5,2.0,2.4\n1,1.1,1.6,2.1,2.5\n1,1.2,1.7,2.2,2.6\n1,1.3,1.8,2.3,2.7\n",
        )
        .unwrap();
        let amp = analysis_scene(
            "timeslice_replay",
            &root,
            &["sess".into()],
            &json!({"y": "amplifier_error"}),
        )
        .unwrap();
        let _ = std::fs::remove_dir_all(&root);
        assert!(amp.panels.iter().all(|panel| panel.title.is_empty()));
        let labels: Vec<&str> = amp
            .panels
            .iter()
            .map(|panel| panel.y_label.as_str())
            .collect();
        assert_eq!(labels, vec!["X (V)", "Y (V)"]);

        let scatter = opened("timeslice_replay", "scatter", &json!({ "scatter": "On" }));
        let distribution = opened(
            "distribution",
            "scatter-source",
            &json!({ "source": "Timeslice" }),
        );
        assert!(scatter.side >= 1);
        assert_eq!(scatter.columns, 2);
        assert_eq!(scatter.column_weights, vec![3.0, 1.4]);
        assert!(scatter
            .controls
            .iter()
            .all(|control| control.id != "scatter_source"));
        let scatter_xy = scatter
            .controls
            .iter()
            .find(|control| control.id == "scatter_xy")
            .unwrap();
        let distribution_xy = distribution
            .controls
            .iter()
            .find(|control| control.id == "xy")
            .unwrap();
        assert_eq!(scatter_xy.labels(), distribution_xy.labels());
    }

    #[test]
    fn replay_axis_ignores_a_single_spike() {
        let mut samples = vec![1.0f32; 400];
        samples[3] = 10_000.0;
        let (lo, hi) = robust_span(&samples).unwrap();
        assert!(hi < 10.0, "{hi}");
        assert!(lo < 2.0);
    }

    #[test]
    fn replay_trace_keeps_a_narrow_pulse() {
        let mut samples = vec![0.0f32; 8_001];
        samples[1] = 40.0;
        let (xs, ys) = trace(&samples);
        assert_eq!(ys.len(), samples.len());
        assert_eq!(ys[1], 40.0);
        assert!((xs[1] - 0.001).abs() < 1e-6);
    }

    #[test]
    fn sk_req_018_fft_draws_a_spectrum() {
        let fft = opened(
            "timeslice_replay",
            "fft",
            &json!({ "fft": "On", "beam": "Both" }),
        );
        assert_eq!(fft.title, "Timeslice Replay");
        assert_eq!(fft.columns, 2);
        assert_eq!(fft.side, 0);
        assert_eq!(fft.column_weights, vec![3.0, 1.4]);
        assert!(fft.panels.len() >= 2 && fft.panels.len().is_multiple_of(2));
        for pair in fft.panels.chunks(2) {
            assert_eq!(pair[1].y_label, "log10 PSD");
            assert_eq!(pair[1].x_label, "Hz");
            assert_eq!(pair[1].xmin, 1.0);
            assert_eq!(pair[1].xmax, 500.0);
            assert!(pair[1].series.iter().any(|series| {
                matches!(series, Series::Polyline { ys, .. } if ys.iter().any(|value| value.is_finite()))
            }));
        }
        let off = scene_of("timeslice_replay");
        assert!(off.panels.iter().all(|panel| panel.y_label != "log10 PSD"));
        assert!(off.controls.iter().any(|control| control.id == "fft"));

        let both = opened(
            "timeslice_replay",
            "fft-scatter",
            &json!({ "fft": "On", "scatter": "On" }),
        );
        assert_eq!(both.columns, 2);
        assert_eq!(both.column_weights, vec![3.0, 1.4, 1.4]);
        assert!(both.side >= 1);
        let main = both.panels.len() - both.side as usize;
        assert_eq!(main % 2, 0);
        assert_eq!(both.panels[1].y_label, "log10 PSD");
    }

    #[test]
    fn sk_req_022_hv_draws_the_current() {
        assert!(has_kind(&scene_of("ic_hv_transient"), "line"));
    }

    #[test]
    fn sk_req_028_dose_volume_has_slices_dvh_and_gamma() {
        let scene = scene_of("dose_volume");
        assert!(has_kind(&scene, "heat"));
        assert!(scene.panels.iter().any(|panel| panel.title.contains("DVH")));
        assert!(scene
            .panels
            .iter()
            .any(|panel| panel.title.contains("gamma")));
        assert!(scene
            .panels
            .iter()
            .any(|panel| panel.title.contains("sagittal")));
        assert!(scene
            .panels
            .iter()
            .any(|panel| panel.title.contains("coronal")));
        assert!(scene.panels.iter().any(|panel| {
            panel.series.iter().any(|series| {
                matches!(
                    series,
                    Series::Heatmap {
                        ramp,
                        lo: 0.0,
                        ..
                    } if *ramp == scan_kit_core::index("turbo")
                )
            })
        }));
        assert!(scene.controls.iter().any(|control| control.id == "scale"));
    }

    #[test]
    fn dose_volume_difference_drops_a_sequential_scale() {
        let root = std::env::temp_dir().join(format!("scan-kit-dose-diff-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        write_session(&root);
        let scene = analysis_scene(
            "dose_volume",
            &root,
            &["sess".into()],
            &json!({"compare": "Difference", "scale": "Turbo"}),
        )
        .unwrap();
        let _ = std::fs::remove_dir_all(&root);
        assert!(scene
            .controls
            .iter()
            .any(|control| { control.id == "scale" && control.value == "Managua" }));
        assert!(scene.panels.iter().any(|panel| {
            panel.series.iter().any(|series| {
                matches!(
                    series,
                    Series::Heatmap { ramp, .. } if *ramp == scan_kit_core::index("managua")
                )
            })
        }));
        assert!(scene
            .panels
            .iter()
            .any(|panel| panel.title.contains("gamma") && panel.title.contains('%')));
    }

    #[test]
    fn patient_ct_adds_dvh_gamma_and_a_fraction_control() {
        let root =
            std::env::temp_dir().join(format!("scan-kit-patient-view-{}", std::process::id()));
        let study = root.join("study");
        let _ = std::fs::remove_dir_all(&root);
        write_session(&root);
        scan_kit_dicom::write_water_study(&study).unwrap();
        let scene = crate::dose_volume(
            &root,
            &["sess".into()],
            &serde_json::json!({"study": study.display().to_string(), "model": "Monte Carlo"}),
            Some(&|job| {
                let scan_kit_core::McJob::Patient(request) = job else {
                    return Err("expected a patient job".into());
                };
                assert!(request.dose_to_water);
                assert_eq!(request.seed, 1);
                assert_eq!(request.material.len(), 32);
                assert!(request.protons.iter().any(|weight| *weight > 0.0));
                Ok(scan_kit_core::McResult {
                    volume: scan_kit_core::Volume {
                        origin: request.origin_mm,
                        shape: request.shape,
                        voxel: request.spacing_mm[0],
                        values: vec![1.0; request.material.len()],
                    },
                    uncertainty: 0.0,
                    ledger: [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                })
            }),
        );
        let _ = std::fs::remove_dir_all(&root);
        assert!(scene
            .controls
            .iter()
            .any(|control| control.id == "fraction"));
        assert!(scene.panels.iter().any(|panel| panel.title.contains("DVH")));
        assert!(scene
            .panels
            .iter()
            .any(|panel| panel.title.contains("gamma") && panel.title.contains('%')));
        assert!(scene.table.as_ref().is_some_and(|table| {
            table.rows.iter().any(|row| {
                row.iter()
                    .any(|cell| cell.contains("Gamma") || cell.contains("PTV"))
            })
        }));
        let plain = scene_of("dose_volume");
        assert!(plain
            .controls
            .iter()
            .all(|control| control.id != "fraction"));
    }

    #[test]
    fn nested_session_reads_its_own_folder() {
        let root = std::env::temp_dir().join(format!("scan-kit-nested-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let nested = root.join("a").join("a");
        std::fs::create_dir_all(nested.join("layer-0").join("run-0")).unwrap();
        std::fs::write(nested.join("input_map.csv"), "energy,charge_req\n70,1\n").unwrap();
        std::fs::write(
            nested
                .join("layer-0")
                .join("run-0")
                .join("timeslice_data_device_units.csv"),
            "r_ic1_current_dose\n1\n",
        )
        .unwrap();
        let sibling = root.join("b");
        std::fs::create_dir_all(sibling.join("layer-0").join("run-0")).unwrap();
        std::fs::write(sibling.join("input_map.csv"), "energy,charge_req\n10,1\n").unwrap();
        std::fs::write(
            sibling
                .join("layer-0")
                .join("run-0")
                .join("timeslice_data_device_units.csv"),
            "only_sibling\n9\n",
        )
        .unwrap();
        let loaded = load_timeslice_columns(&root, "a");
        let names: Vec<&str> = loaded["columns"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|column| column["name"].as_str())
            .collect();
        assert!(names.contains(&"r_ic1_current_dose"));
        assert!(!names.contains(&"only_sibling"));
        assert_eq!(loaded["energy_layers"], 1);
        let scene = analysis_scene("binned_summary", &root, &["a".into()], &json!({})).unwrap();
        assert!(scene
            .panels
            .iter()
            .any(|panel| panel.title.contains("No finite values")));
        let _ = std::fs::remove_dir_all(&root);
    }

    fn finite(values: Option<&Vec<f32>>) -> Vec<f32> {
        values
            .into_iter()
            .flatten()
            .copied()
            .filter(|value| value.is_finite())
            .collect()
    }

    #[test]
    fn a_parked_beam_converts_strips_with_device_geometry() {
        let root = std::env::temp_dir().join("scan-kit-parked-geometry");
        let _ = std::fs::remove_dir_all(&root);
        let session = root.join("sess");
        std::fs::create_dir_all(session.join("layer-0/run-0")).unwrap();
        std::fs::create_dir_all(session.join("config/map2map")).unwrap();
        std::fs::write(
            session.join("config/map2map/devices.xml"),
            r#"<devices><ion_chamber><device name="IC_1_X"/>
            <strip_count>128</strip_count><strip_to_mm>2</strip_to_mm>
            <zero_offset_at_iso_mm>-4</zero_offset_at_iso_mm>
            <source_to_device_distance_mm>1250</source_to_device_distance_mm>
            </ion_chamber></devices>"#,
        )
        .unwrap();
        std::fs::write(
            session.join("config/map2map/scan_dose_system.xml"),
            "<MapToMap><geometry><source_to_isocenter_distance>2500</source_to_isocenter_distance></geometry></MapToMap>",
        )
        .unwrap();
        std::fs::write(
            session.join("input_map.csv"),
            "energy,layer_id,spot_no,X_POSITION,Y_POSITION\n250,1,0,0,0\n",
        )
        .unwrap();
        let mut spot = String::from(
            "spot_no,layer_id,r_ic1_x_spot_sigma_raw,r_ic1_x_spot_sigma,r_ic1_x_spot_position_raw,r_ic1_x_spot_position\n",
        );
        for index in 0..12 {
            spot.push_str(&format!("{index},1,1,4,64.5,0\n"));
        }
        std::fs::write(session.join("spot_data.csv"), spot).unwrap();
        std::fs::write(
            session.join("layer-0/run-0/timeslice_data_device_units.csv"),
            "layer_id,rci_in_trigger,r_ic1_x_spot_position,r_ic1_x_sigma\n1,1,64.5,2\n",
        )
        .unwrap();
        let table = crate::tables::slice_table(&root, "sess");
        let x = table
            .get("ic1_x")
            .and_then(|values| values.first())
            .copied();
        let sigma = table
            .get("ic1_sig_x")
            .and_then(|values| values.first())
            .copied();
        assert!(x.is_some_and(|value| value.abs() < 1e-2), "position {x:?}");
        assert!(
            sigma.is_some_and(|value| (value - 8.0).abs() < 1e-2),
            "sigma {sigma:?}"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn modern_sessions_draw_a_millimetre_distribution() {
        let root =
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../test_data/logs");
        for id in ["1905185100", "1093436476", "1771520329"] {
            if !crate::discover::session_directory(&root, id)
                .join("spot_data.csv")
                .is_file()
            {
                continue;
            }
            let spot = crate::tables::spot_table(&root, id);
            let slice = crate::tables::slice_table(&root, id);
            let spot_x = finite(spot.get("ic1_x"));
            let slice_x = finite(slice.get("ic1_x"));
            let spot_sig = finite(spot.get("ic1_sig_x"));
            let slice_sig = finite(slice.get("ic1_sig_x"));
            let plan = finite(spot.get("plan_x"));
            assert!(
                spot_x.len() > 100 && slice_x.len() > 1_000,
                "{id} spot {} slice {}",
                spot_x.len(),
                slice_x.len()
            );
            assert!(
                slice_x.iter().all(|value| value.abs() < 200.0),
                "{id} timeslice position left the millimetre field"
            );
            let plan_span = plan.iter().copied().fold(f32::MIN, f32::max)
                - plan.iter().copied().fold(f32::MAX, f32::min);
            if plan_span < 0.1 {
                let median = crate::tables::median(&slice_x);
                assert!(
                    median.abs() < 5.0,
                    "{id} parked beam median {median} is not at isocenter"
                );
            } else {
                let span = slice_x.iter().copied().fold(f32::MIN, f32::max)
                    - slice_x.iter().copied().fold(f32::MAX, f32::min);
                assert!(span > 40.0, "{id} timeslice field span {span}");
            }
            let slice_sigma = crate::tables::median(&slice_sig);
            let spot_sigma = crate::tables::median(&spot_sig);
            let ratio = spot_sigma / slice_sigma;
            assert!(
                slice_sigma > 2.0 && (1.2..=3.2).contains(&ratio),
                "{id} sigma spot {spot_sigma} slice {slice_sigma}"
            );
            let scene = analysis_scene(
                "distribution",
                &root,
                &[id.to_string()],
                &json!({ "xy": "position", "source": "timeslice" }),
            )
            .unwrap();
            let points: usize = scene
                .panels
                .iter()
                .flat_map(|panel| &panel.series)
                .map(|series| match series {
                    scan_kit_core::Series::Points { xs, .. } => xs.len(),
                    _ => 0,
                })
                .sum();
            let axis = scene.panels.first().unwrap();
            assert!(
                points > 1_000 && axis.xmin > -250.0 && axis.xmax < 250.0,
                "{id} distribution points {points} axis {}..{}",
                axis.xmin,
                axis.xmax
            );
            if plan_span < 0.1 {
                assert!(
                    axis.xmin > -30.0 && axis.xmax < 30.0,
                    "{id} parked distribution axis {}..{}",
                    axis.xmin,
                    axis.xmax
                );
            }
        }
    }
}
