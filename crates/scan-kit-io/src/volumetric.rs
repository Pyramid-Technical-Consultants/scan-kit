//! Volumetric: analytic Bragg dose, and a Monte Carlo fill when the compute
//! crate supplies a runner. The sidebar is the analysis shell.

use std::path::Path;

use scan_kit_core::{
    analytic_on, choices, dose_frame, dvh, gamma_index, index, medium, protons_from_mu, resolve,
    robust_high, Family, McJob, McResult, Panel, Pencil, PlotScene, Quantity, Series, Volume,
    IC1_Z_MM, IC2_Z_MM, IC_SEP_MM, SESSION,
};
use serde_json::Value;

use super::discover;
use super::marks::{labeled, pick};
use super::tables::{interlock_sigma_mm, spot_table};

const MARK: [f32; 4] = [0.0, 0.0, 0.0, 1.0];
const FALLBACK_SIGMA: f32 = 4.0;
/// `abs(-10000) * 0.9`, the dose loader's missing-position cut.
const ABS_INVALID_MM: f32 = 9_000.0;
const FALLBACK_KMU: f32 = 2.0e-8;
const GAP_MM: f32 = 10.0;
const MC_SEED: u32 = 1;

const XY: &[(&str, &str)] = &[
    ("ic1", "IC1"),
    ("ic2", "IC2"),
    ("iso_ray", "ISO Ray"),
    ("plan", "Plan"),
];
const QUANTITY: &[(&str, &str)] = &[("dose", "Dose"), ("mu", "MU"), ("protons", "Protons")];
const MODEL: &[(&str, &str)] = &[("analytic", "Analytic"), ("mc", "Monte Carlo")];
const SCATTER: &[(&str, &str)] = &[("on", "On"), ("off", "Off")];
const HISTORIES: &[(&str, &str)] = &[
    ("1000000", "1e6"),
    ("3000000", "3e6"),
    ("10000000", "1e7"),
    ("50000000", "5e7"),
];
const SPREAD: &[(&str, &str)] = &[
    ("0", "0%"),
    ("0.5", "0.5%"),
    ("1", "1%"),
    ("1.5", "1.5%"),
    ("2", "2%"),
    ("3", "3%"),
];
const MEDIA: &[(&str, &str)] = &[
    ("water", "Water"),
    ("pmma", "PMMA"),
    ("polystyrene", "Polystyrene"),
    ("polyethylene", "Polyethylene"),
    ("a150", "A-150"),
    ("aluminum", "Aluminum"),
    ("copper", "Copper"),
];
const MC_MEDIA: &[&str] = &["water", "pmma", "polystyrene", "aluminum", "copper"];
const PHANTOM: &[(&str, &str)] = &[
    ("0", "Auto"),
    ("50", "50 mm"),
    ("100", "100 mm"),
    ("200", "200 mm"),
    ("300", "300 mm"),
];
const WET: &[(&str, &str)] = &[("0", "0"), ("2", "2 mm"), ("5", "5 mm"), ("10", "10 mm")];
const COMPARE: &[(&str, &str)] = &[
    ("measured", "Measured"),
    ("difference", "Difference"),
    ("gamma", "Gamma"),
];
const SIGMA: &[(&str, &str)] = &[
    ("measured", "Layer"),
    ("reference", "Session"),
    ("interlock", "Interlock"),
];
const EDGE: &[(&str, &str)] = &[
    ("slice50", "Slice 50%"),
    ("peak90", "Peak 90%"),
    ("peak50", "Peak 50%"),
    ("slice20", "Slice 20%"),
    ("plan90", "Plan 90%"),
];

/// `(fraction, per slice, from the plan)`. Slice edges use each depth's own maximum.
fn field_edge(kind: &str) -> (f32, bool, bool) {
    match kind {
        "peak90" => (0.9, false, false),
        "peak50" => (0.5, false, false),
        "slice20" => (0.2, true, false),
        "plan90" => (0.9, false, true),
        _ => (0.5, true, false),
    }
}

pub(crate) type McRunner<'a> = &'a dyn Fn(&McJob) -> Result<McResult, String>;

struct Cloud {
    pencils: Vec<Pencil>,
    plan: Vec<Pencil>,
}

pub fn volumetric(
    root: &Path,
    session_ids: &[String],
    options: &Value,
    mc: Option<McRunner<'_>>,
) -> PlotScene {
    if let Some(path) = options.get("study").and_then(Value::as_str) {
        if !path.is_empty() {
            return super::patient_view::scene(root, session_ids, options, mc);
        }
    }
    let picked = crate::source::select(crate::source::Shape::Source, true, true, &[], options);
    let grain = picked.grain;
    let xy = pick(options, "xy", "ic1", XY);
    let quantity_id = pick(options, "quantity", "dose", QUANTITY);
    let quantity = match quantity_id {
        "mu" => Quantity::Mu,
        "protons" => Quantity::Protons,
        _ => Quantity::Dose,
    };
    let model = pick(options, "model", "analytic", MODEL);
    let scatter = pick(options, "scatter", "on", SCATTER) == "on";
    let histories = pick(options, "histories", "10000000", HISTORIES)
        .parse::<u32>()
        .unwrap_or(10_000_000);
    let spread = pick(options, "spread", "1", SPREAD)
        .parse::<f32>()
        .unwrap_or(1.0);
    let mut medium_key = pick(options, "medium", "water", MEDIA);
    if model == "mc" && !MC_MEDIA.contains(&medium_key) {
        medium_key = "water";
    }
    let phantom_mm = pick(options, "phantom", "0", PHANTOM)
        .parse::<f32>()
        .unwrap_or(0.0);
    let wet_mm = pick(options, "wet", "0", WET).parse::<f32>().unwrap_or(0.0);
    let compare = pick(options, "compare", "measured", COMPARE);
    let plan_sigma = pick(options, "plan_sigma", "measured", SIGMA);
    let edge = pick(options, "edge", "slice50", EDGE);
    let family = if compare == "difference" {
        Family::Divergent
    } else {
        Family::Sequential
    };
    let scale = pick(
        options,
        "scale",
        if compare == "difference" {
            "managua"
        } else {
            "turbo"
        },
        choices(family),
    );
    let sigma_ref = options
        .get("sigma_ref")
        .and_then(Value::as_str)
        .unwrap_or("");

    let mat = medium(medium_key);
    let voxel_mm = pick(
        options,
        "voxel",
        "1",
        &[("0.5", "0.5 mm"), ("1", "1 mm"), ("2", "2 mm")],
    )
    .parse::<f32>()
    .unwrap_or(1.0);
    let gap_mm = pick(
        options,
        "gap",
        "10",
        &[("5", "5 mm"), ("10", "10 mm"), ("20", "20 mm")],
    )
    .parse::<f32>()
    .unwrap_or(GAP_MM);
    let margin_sigma = pick(
        options,
        "margin",
        "5",
        &[("5", "5 σ"), ("4", "4 σ"), ("3", "3 σ"), ("1", "1 σ")],
    )
    .parse::<f32>()
    .unwrap_or(5.0);
    let gantry = pick(
        options,
        "gantry",
        "90",
        &[("0", "0°"), ("90", "90°"), ("180", "180°"), ("270", "270°")],
    )
    .parse::<f32>()
    .unwrap_or(90.0);
    let spot_cap = pick(
        options,
        "spot_cap",
        "1000000",
        &[("10000", "10k"), ("100000", "100k"), ("1000000", "1M")],
    )
    .parse::<usize>()
    .unwrap_or(1_000_000)
    .max(1);
    let clouds: Vec<Cloud> = session_ids
        .iter()
        .map(|session| {
            let mut cloud =
                load_cloud(root, session, session_ids, grain, xy, plan_sigma, sigma_ref);
            cloud.pencils.truncate(spot_cap);
            cloud.plan.truncate(spot_cap);
            cloud
        })
        .collect();
    let k_mu = session_ids
        .first()
        .map(|session| read_kmu(root, session))
        .unwrap_or(FALLBACK_KMU);

    let plots = crate::workspace::plots_from(
        [
            options.get("plot0").and_then(Value::as_str),
            options.get("plot1").and_then(Value::as_str),
        ],
        false,
    );
    let (field_fraction, field_per_slice, field_from_plan) = field_edge(edge);
    let need_plan = compare != "measured"
        || plots.contains(&crate::workspace::PlotKind::Gamma)
        || field_from_plan;
    let all: Vec<Pencil> = clouds
        .iter()
        .flat_map(|cloud| {
            let measured = cloud.pencils.iter().copied();
            if need_plan {
                measured
                    .chain(cloud.plan.iter().copied())
                    .collect::<Vec<_>>()
            } else {
                measured.collect()
            }
        })
        .collect();
    let preview = options
        .get("_preview")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let scatter_on = scatter && model != "mc";
    let mut voxel_mm = voxel_mm;
    let mut frame = dose_frame(
        mat,
        &all,
        quantity,
        spread,
        scatter_on,
        wet_mm,
        phantom_mm,
        voxel_mm,
        k_mu,
        gap_mm,
        margin_sigma,
    );
    if preview {
        let mut steps = 0;
        while steps < 3 {
            let Some(next) = next_preview_voxel(voxel_mm, frame.visits) else {
                break;
            };
            voxel_mm = next;
            frame = dose_frame(
                mat,
                &all,
                quantity,
                spread,
                scatter_on,
                wet_mm,
                phantom_mm,
                voxel_mm,
                k_mu,
                gap_mm,
                margin_sigma,
            );
            steps += 1;
        }
    }
    let grid = Some((frame.origin, frame.shape, frame.voxel));
    // Three planes through the peak raymarch as a cross. The 3D cell needs the cube.
    let planes = lattice_planes(&frame);

    let mut measured = Vec::new();
    let mut plans = Vec::new();
    let mut mc_note = None;
    for cloud in &clouds {
        let (volume, ran) = fill(
            mat,
            &cloud.pencils,
            quantity,
            spread,
            scatter,
            wet_mm,
            phantom_mm,
            k_mu,
            gap_mm,
            model,
            histories,
            medium_key,
            mc,
            grid,
            planes,
        );
        mc_note = mc_note.or(ran);
        measured.push(volume);
        if need_plan {
            let (volume, _) = fill(
                mat,
                &cloud.plan,
                quantity,
                spread,
                scatter,
                wet_mm,
                phantom_mm,
                k_mu,
                gap_mm,
                model,
                histories,
                medium_key,
                mc,
                grid,
                planes,
            );
            plans.push(volume);
        }
    }

    let (dd, dta, cutoff) = gamma_criteria(options);
    // Measured is the reference and the plan is searched, as in the Python dose view.
    let want_gamma = compare == "gamma" || plots.contains(&crate::workspace::PlotKind::Gamma);
    let gamma_fields: Vec<(Volume, f32)> = if want_gamma {
        measured
            .iter()
            .zip(&plans)
            .map(|(got, plan)| {
                let (values, passed, scored) = gamma_with(got, plan, dd, dta, cutoff);
                let rate = if scored == 0 {
                    0.0
                } else {
                    100.0 * passed as f32 / scored as f32
                };
                (gamma_image(got, &values), rate)
            })
            .collect()
    } else {
        Vec::new()
    };
    let shown: Vec<Volume> = match compare {
        "difference" => measured
            .iter()
            .zip(&plans)
            .map(|(got, plan)| difference(got, plan))
            .collect(),
        "gamma" => gamma_fields
            .iter()
            .map(|(volume, _)| volume.clone())
            .collect(),
        _ => measured.clone(),
    };
    let sessions = shown.len().max(1);
    let ramp = if compare == "gamma" {
        index("gamma")
    } else if sessions > 1 {
        SESSION
    } else {
        resolve(scale, family)
    };
    let (lo, hi) = window(&shown, compare);
    let mut panels = Vec::new();
    let has_dose = measured
        .iter()
        .any(|volume| volume.values.iter().any(|value| *value > 0.0));
    if !has_dose {
        panels.push(note("No spots"));
    }

    let mut volume_mark = scan_kit_core::VolumeMark::default();
    let mut field_size: Option<String> = None;
    let session_at = session_at(options, session_ids).min(shown.len().saturating_sub(1));
    if has_dose {
        if let Some(volume) = shown.get(session_at) {
            let (ix, iy, iz) = volume.peak_index();
            let cells = crate::workspace::cells_from(&[
                options.get("cell0").and_then(Value::as_str),
                options.get("cell1").and_then(Value::as_str),
                options.get("cell2").and_then(Value::as_str),
                options.get("cell3").and_then(Value::as_str),
            ]);
            let mut dvh_lines = Vec::new();
            if plots.contains(&crate::workspace::PlotKind::Dvh) {
                push_dvh(&mut dvh_lines, volume);
            }
            let gamma = plots
                .contains(&crate::workspace::PlotKind::Gamma)
                .then(|| gamma_fields.get(session_at).cloned())
                .flatten();
            let show_field = super::marks::flag(options, "field", true);
            let field_dose = if field_from_plan {
                plans.get(session_at)
            } else {
                measured.get(session_at)
            };
            let extent = field_dose.and_then(|dose| {
                crate::workspace::field_extent(dose, field_fraction, field_per_slice)
            });
            let field = if show_field {
                extent.map(|bounds| [bounds[0], bounds[1], bounds[2], bounds[3]])
            } else {
                None
            };
            let (mut mode, filter) = view_paint(options);
            if compare == "gamma" {
                mode = 1;
            }
            let (lo_paint, hi_paint, gain, opacity) = paint_of(options, compare, lo, hi, mode);
            let (built, mut mark) = crate::workspace::assemble(&crate::workspace::Workspace {
                dose: volume.clone(),
                ct: Vec::new(),
                labels: Vec::new(),
                cursor: [ix, iy, iz],
                cells,
                plots,
                ramp,
                lo: lo_paint,
                hi: hi_paint,
                gain,
                opacity,
                mode,
                filter,
                y_label: y_name(quantity).into(),
                dvh: dvh_lines,
                gamma,
                field,
                show_field,
            });
            mark.gantry = gantry;
            mark.show_phantom = phantom_shown(options);
            mark.field = if show_field {
                extent.unwrap_or([0.0; 6])
            } else {
                [0.0; 6]
            };
            mark.unit = volume_unit(quantity, mode, compare).into();
            panels = built;
            if let Some(depth) = panels
                .iter_mut()
                .find(|panel| panel.title.starts_with("Depth"))
            {
                if let Some(result) = &mc_note {
                    let incident = result.ledger[0];
                    let left = if incident > 0.0 {
                        100.0 * (result.ledger[2] + result.ledger[3]) / incident
                    } else {
                        0.0
                    };
                    depth.title = format!(
                        "Depth dose  {:.1}%  left {:.1}%",
                        result.uncertainty * 100.0,
                        left
                    );
                } else if model == "mc" {
                    depth.title = "Depth dose  analytic fallback".into();
                }
            }
            volume_mark = mark;
            let size = match extent {
                Some(bounds) => format!(
                    "{:.0} × {:.0} × {:.0} mm",
                    bounds[1] - bounds[0],
                    bounds[3] - bounds[2],
                    bounds[5] - bounds[4]
                ),
                None => "—".into(),
            };
            field_size = Some(size);
        }
    }

    let mut controls = picked.controls;
    for control in &mut controls {
        if control.id == "source" {
            control.group = "Dose".into();
        }
    }
    if session_ids.len() > 1 {
        let pairs: Vec<(String, String)> = session_ids
            .iter()
            .map(|id| (id.clone(), id.clone()))
            .collect();
        let refs: Vec<(&str, &str)> = pairs
            .iter()
            .map(|(a, b)| (a.as_str(), b.as_str()))
            .collect();
        let chosen = session_ids
            .get(session_at)
            .map(String::as_str)
            .unwrap_or("");
        controls.push(labeled("session", "Session", &refs, chosen).grouped("Dose"));
    }
    controls.push(labeled("xy", "Signal", XY, xy).grouped("Dose"));
    controls.push(labeled("quantity", "Quantity", QUANTITY, quantity_id).grouped("Dose"));
    controls.push(labeled("compare", "Compare", COMPARE, compare).grouped("Dose"));
    if xy == "plan" || compare != "measured" {
        controls.push(labeled("plan_sigma", "Plan σ", SIGMA, plan_sigma).grouped("Dose"));
        if session_ids.len() > 1 {
            let pairs: Vec<(String, String)> = session_ids
                .iter()
                .map(|id| (id.clone(), id.clone()))
                .collect();
            let refs: Vec<(&str, &str)> = pairs
                .iter()
                .map(|(a, b)| (a.as_str(), b.as_str()))
                .collect();
            let chosen = if refs.iter().any(|(id, _)| *id == sigma_ref) {
                sigma_ref
            } else {
                refs.first().map(|(id, _)| *id).unwrap_or("")
            };
            controls.push(labeled("sigma_ref", "Reference", &refs, chosen).grouped("Dose"));
        }
    }
    if has_dose {
        let raw_spots: usize = clouds.iter().map(|cloud| cloud.pencils.len()).sum();
        let plan_spots: usize = clouds.iter().map(|cloud| cloud.plan.len()).sum();
        let spot_note = if plan_spots == 0 {
            format!("{raw_spots} · no plan")
        } else {
            let weights = match quantity {
                Quantity::Mu => "plan MU",
                Quantity::Protons => "protons",
                Quantity::Dose => "relative",
            };
            format!("{raw_spots} measured · {plan_spots} plan · {weights}")
        };
        controls.push(crate::workspace::readout(
            "Spots", "spots", &spot_note, "Dose",
        ));
    }
    controls.push(
        labeled("model", "Model", MODEL, model)
            .grouped("Calculation")
            .radio(),
    );
    if model == "analytic" {
        controls.push(
            labeled(
                "scatter",
                "Scatter",
                SCATTER,
                if scatter { "on" } else { "off" },
            )
            .grouped("Calculation"),
        );
    } else {
        controls.push(
            labeled("histories", "Histories", HISTORIES, &histories.to_string())
                .grouped("Calculation")
                .radio(),
        );
    }
    controls
        .push(labeled("spread", "Energy spread", SPREAD, &trim_num(spread)).grouped("Calculation"));
    if quantity_id != "mu" {
        controls.push(
            labeled(
                "gap",
                "IC gap",
                &[("5", "5 mm"), ("10", "10 mm"), ("20", "20 mm")],
                &gap_mm.round().to_string(),
            )
            .grouped("Calculation"),
        );
    }
    controls.push(
        labeled(
            "spot_cap",
            "Max spots",
            &[("10000", "10k"), ("100000", "100k"), ("1000000", "1M")],
            &spot_cap.to_string(),
        )
        .grouped("Calculation"),
    );
    let media: Vec<(&str, &str)> = if model == "mc" {
        MEDIA
            .iter()
            .copied()
            .filter(|(id, _)| MC_MEDIA.contains(id))
            .collect()
    } else {
        MEDIA.to_vec()
    };
    controls.push(labeled("medium", "Medium", &media, medium_key).grouped("Phantom"));
    controls.push(
        labeled(
            "phantom",
            "Thickness",
            PHANTOM,
            &phantom_mm.round().to_string(),
        )
        .grouped("Phantom"),
    );
    controls.push(labeled("wet", "Entrance", WET, &wet_mm.round().to_string()).grouped("Phantom"));
    controls.push(
        labeled(
            "margin",
            "Auto margin",
            &[("5", "5 σ"), ("4", "4 σ"), ("3", "3 σ"), ("1", "1 σ")],
            &margin_sigma.round().to_string(),
        )
        .grouped("Phantom"),
    );
    controls.push(
        labeled(
            "gantry",
            "Gantry",
            &[("0", "0°"), ("90", "90°"), ("180", "180°"), ("270", "270°")],
            &gantry.round().to_string(),
        )
        .grouped("Phantom"),
    );
    if has_dose {
        let sizing = if phantom_mm <= 0.0 { " auto" } else { "" };
        let phantom = format!(
            "{:.0} × {:.0} × {:.0} mm{sizing}",
            volume_mark.shape[0] as f32 * volume_mark.voxel,
            volume_mark.shape[1] as f32 * volume_mark.voxel,
            volume_mark.shape[2] as f32 * volume_mark.voxel
        );
        controls.push(crate::workspace::readout(
            "Size",
            "phantom_size",
            &phantom,
            "Phantom",
        ));
        controls.push(
            scan_kit_core::Control::plain(
                "phantom_box",
                "Box",
                ["On", "Off"],
                if phantom_shown(options) { "On" } else { "Off" },
            )
            .grouped("Phantom")
            .checked(),
        );
    }
    controls.push(labeled("edge", "Edge", EDGE, edge).grouped("Field"));
    if has_dose {
        let show_box = super::marks::flag(options, "field", true);
        controls.push(
            scan_kit_core::Control::plain(
                "field",
                "Bounds",
                ["On", "Off"],
                if show_box { "On" } else { "Off" },
            )
            .grouped("Field")
            .checked(),
        );
        if let Some(size) = field_size {
            controls.push(crate::workspace::readout(
                "Size",
                "field_size",
                &size,
                "Field",
            ));
        }
        push_picture_controls(
            &mut controls,
            options,
            &volume_mark,
            model,
            compare,
            session_ids.len(),
            scale,
        );
    }
    let layout = if has_dose {
        crate::workspace::layout()
    } else {
        (1, Vec::new(), Vec::new(), Vec::new())
    };
    PlotScene {
        title: "Volumetric".into(),
        panels,
        controls,
        table: None,
        columns: layout.0,
        column_weights: layout.1,
        row_weights: layout.2,
        side: 0,
        row_splits: layout.3,
        volume: volume_mark,
    }
}

fn paint_of(options: &Value, compare: &str, lo: f32, hi: f32, mode: u8) -> (f32, f32, f32, f32) {
    if compare == "gamma" {
        return (0.0, 2.0, 1.0, 1.0);
    }
    let level = options
        .get("level")
        .and_then(Value::as_str)
        .and_then(|text| text.parse::<f32>().ok());
    if mode == 2 {
        let opacity = level.unwrap_or(1.0).clamp(0.0, 1.0);
        return (lo, hi, opacity, opacity);
    }
    if compare == "difference" {
        let percent = pick(
            options,
            "error",
            "absolute",
            &[("absolute", "Absolute"), ("percent", "Percent")],
        ) == "percent";
        let reach = if percent {
            hi.abs().max(1e-6) * level.unwrap_or(10.0).clamp(0.1, 100.0) / 100.0
        } else {
            level.unwrap_or(0.2).abs().max(1e-6)
        };
        return (-reach, reach, 1.0, 1.0);
    }
    if super::marks::flag(options, "auto", true) {
        let gain = level.unwrap_or(1.0).clamp(0.05, 8.0);
        return (lo, hi, gain, 1.0);
    }
    (0.0, level.unwrap_or(hi).max(1e-6), 1.0, 1.0)
}

fn range_control(
    id: &str,
    label: &str,
    min: f32,
    max: f32,
    step: f32,
    value: &str,
) -> scan_kit_core::Control {
    let mut control = scan_kit_core::Control::plain(id, label, Vec::<&str>::new(), value);
    control.options = vec![
        scan_kit_core::Choice::full("min", &trim_range(min), "", ""),
        scan_kit_core::Choice::full("max", &trim_range(max), "", ""),
        scan_kit_core::Choice::full("step", &trim_range(step), "", ""),
    ];
    control.kind = "range".into();
    control
}

fn trim_range(value: f32) -> String {
    if (value - value.round()).abs() < 1e-4 && value.abs() < 1.0e6 {
        format!("{}", value.round() as i32)
    } else {
        let text = format!("{value:.4}");
        text.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

fn level_text(options: &Value, default: f32, min: f32, max: f32) -> String {
    let raw = options
        .get("level")
        .and_then(Value::as_str)
        .and_then(|text| text.parse::<f32>().ok())
        .unwrap_or(default);
    trim_range(raw.clamp(min, max))
}

fn session_at(options: &Value, ids: &[String]) -> usize {
    let raw = options.get("session").and_then(Value::as_str).unwrap_or("");
    ids.iter().position(|id| id == raw).unwrap_or(0)
}

fn push_picture_controls(
    controls: &mut Vec<scan_kit_core::Control>,
    options: &Value,
    mark: &scan_kit_core::VolumeMark,
    model: &str,
    compare: &str,
    sessions: usize,
    scale: &str,
) {
    let ray = pick(
        options,
        "ray",
        "integrate",
        &[
            ("integrate", "Integrate"),
            ("maximum", "Maximum"),
            ("transparent", "Transparent"),
        ],
    );
    let auto = super::marks::flag(options, "auto", true);
    if compare != "gamma" {
        if sessions <= 1 {
            let family = if compare == "difference" {
                Family::Divergent
            } else {
                Family::Sequential
            };
            controls.push(labeled("scale", "Scale", choices(family), scale).grouped("Picture"));
        }
        controls.push(
            scan_kit_core::Control::plain(
                "auto",
                "Auto",
                ["On", "Off"],
                if auto { "On" } else { "Off" },
            )
            .grouped("Picture")
            .checked(),
        );
        if compare == "difference" {
            controls.push(
                labeled(
                    "error",
                    "Full scale",
                    &[("absolute", "Absolute"), ("percent", "Percent")],
                    pick(
                        options,
                        "error",
                        "absolute",
                        &[("absolute", "Absolute"), ("percent", "Percent")],
                    ),
                )
                .grouped("Picture"),
            );
        }
        let (level_label, level_min, level_max, level_step, level_default) =
            if compare == "difference" {
                if pick(
                    options,
                    "error",
                    "absolute",
                    &[("absolute", "Absolute"), ("percent", "Percent")],
                ) == "percent"
                {
                    ("Percent", 1.0, 100.0, 1.0, 10.0)
                } else {
                    ("Full scale", 0.01, 5.0, 0.01, 0.2)
                }
            } else if ray == "transparent" {
                ("Opacity", 0.0, 1.0, 0.05, 1.0)
            } else if auto {
                ("Gain", 0.25, 4.0, 0.05, 1.0)
            } else {
                let span = mark.hi.abs().max(mark.lo.abs()).max(1.0);
                ("Window", 0.0, span * 2.0, (span / 50.0).max(0.01), span)
            };
        controls.push(
            range_control(
                "level",
                level_label,
                level_min,
                level_max,
                level_step,
                &level_text(options, level_default, level_min, level_max),
            )
            .grouped("Picture"),
        );
    }
    controls.push(
        labeled(
            "ray",
            "Ray",
            &[
                ("integrate", "Integrate"),
                ("maximum", "Maximum"),
                ("transparent", "Transparent"),
            ],
            pick(
                options,
                "ray",
                "integrate",
                &[
                    ("integrate", "Integrate"),
                    ("maximum", "Maximum"),
                    ("transparent", "Transparent"),
                ],
            ),
        )
        .grouped("Picture"),
    );
    controls.push(
        labeled(
            "sample",
            "Sample",
            &[
                ("nearest", "Nearest"),
                ("linear", "Linear"),
                ("cubic", "Cubic"),
            ],
            pick(
                options,
                "sample",
                "linear",
                &[
                    ("nearest", "Nearest"),
                    ("linear", "Linear"),
                    ("cubic", "Cubic"),
                ],
            ),
        )
        .grouped("Picture"),
    );
    controls.push(
        labeled(
            "voxel",
            "Voxel",
            &[("0.5", "0.5 mm"), ("1", "1 mm"), ("2", "2 mm")],
            pick(
                options,
                "voxel",
                "1",
                &[("0.5", "0.5 mm"), ("1", "1 mm"), ("2", "2 mm")],
            ),
        )
        .grouped("Picture"),
    );
    let mut grid = format!("{} × {} × {}", mark.shape[0], mark.shape[1], mark.shape[2]);
    if model == "mc" {
        grid.push_str(" · MC");
    }
    controls.push(crate::workspace::readout("Grid", "grid", &grid, "Picture"));
    if compare == "gamma" {
        controls.push(
            labeled(
                "dd",
                "Dose difference",
                &[("2", "2%"), ("3", "3%"), ("5", "5%")],
                pick(options, "dd", "3", &[("2", "2%"), ("3", "3%"), ("5", "5%")]),
            )
            .grouped("Gamma"),
        );
        controls.push(
            labeled(
                "dta",
                "Distance",
                &[("2", "2 mm"), ("3", "3 mm")],
                pick(options, "dta", "2", &[("2", "2 mm"), ("3", "3 mm")]),
            )
            .grouped("Gamma"),
        );
        controls.push(
            labeled(
                "cutoff",
                "Low dose",
                &[("5", "5%"), ("10", "10%"), ("20", "20%")],
                pick(
                    options,
                    "cutoff",
                    "10",
                    &[("5", "5%"), ("10", "10%"), ("20", "20%")],
                ),
            )
            .grouped("Gamma"),
        );
        controls.push(crate::workspace::readout(
            "TG-218",
            "tg218",
            "≥95% tolerance, <90% action",
            "Gamma",
        ));
    }
    let cells = crate::workspace::cells_from(&[
        options.get("cell0").and_then(Value::as_str),
        options.get("cell1").and_then(Value::as_str),
        options.get("cell2").and_then(Value::as_str),
        options.get("cell3").and_then(Value::as_str),
    ]);
    for (index, cell) in cells.iter().enumerate() {
        controls.push(crate::workspace::cell_control(index, *cell));
    }
    let plots = crate::workspace::plots_from(
        [
            options.get("plot0").and_then(Value::as_str),
            options.get("plot1").and_then(Value::as_str),
        ],
        false,
    );
    controls.push(crate::workspace::plot_control(0, plots[0]));
    controls.push(crate::workspace::plot_control(1, plots[1]));
}

/// The cyan phantom box. Thickness stays on `phantom`. Older saves stored the
/// box on that same key as On or Off.
fn phantom_shown(options: &Value) -> bool {
    if options.get("phantom_box").is_some() {
        return super::marks::flag(options, "phantom_box", false);
    }
    matches!(
        options.get("phantom").and_then(Value::as_str),
        Some("on" | "On" | "true")
    )
}

fn view_paint(options: &Value) -> (u8, u8) {
    let mode = match pick(
        options,
        "ray",
        "integrate",
        &[
            ("integrate", "Integrate"),
            ("maximum", "Maximum"),
            ("transparent", "Transparent"),
        ],
    ) {
        "maximum" => 1,
        "transparent" => 2,
        _ => 0,
    };
    let filter = match pick(
        options,
        "sample",
        "linear",
        &[
            ("nearest", "Nearest"),
            ("linear", "Linear"),
            ("cubic", "Cubic"),
        ],
    ) {
        "nearest" => 0,
        "cubic" => 2,
        _ => 1,
    };
    (mode, filter)
}

/// The displayed lattice. A heavy visit count used to keep only three planes;
/// that shortcut is a cross in the ray march, so the view always asks for every voxel.
fn lattice_planes(_frame: &scan_kit_core::DoseFrame) -> Option<[usize; 3]> {
    None
}

/// Coarser spacing for the first picture. A 1 mm lattice of 24 million visits
/// took 59 ms to deposit in release and 617 ms in debug, and the open waited
/// for that before any picture. Doubling the spacing cuts the visits by about
/// eight. 4 mm is the coarsest cube we still show.
fn next_preview_voxel(voxel: f32, visits: u64) -> Option<f32> {
    const PREVIEW_VISITS: u64 = 4_000_000;
    if visits <= PREVIEW_VISITS || voxel >= 4.0 {
        return None;
    }
    let next = (voxel * 2.0).min(4.0);
    (next > voxel + 1e-4).then_some(next)
}

fn fill(
    mat: scan_kit_core::Medium,
    pencils: &[Pencil],
    quantity: Quantity,
    spread: f32,
    scatter: bool,
    wet_mm: f32,
    phantom_mm: f32,
    k_mu: f32,
    gap_mm: f32,
    model: &str,
    histories: u32,
    medium_key: &str,
    mc: Option<McRunner<'_>>,
    grid: Option<([f32; 3], [usize; 3], f32)>,
    planes: Option<[usize; 3]>,
) -> (Volume, Option<McResult>) {
    if model == "mc" {
        if let Some(run) = mc {
            if let Some((origin, shape, voxel)) = grid {
                let depth = if phantom_mm > 0.0 {
                    phantom_mm
                } else {
                    shape[2] as f32 * voxel
                };
                let job = McJob::Slab(scan_kit_core::SlabRequest {
                    medium: medium_key.into(),
                    x: pencils.iter().map(|p| p.x).collect(),
                    y: pencils.iter().map(|p| p.y).collect(),
                    sx: pencils.iter().map(|p| p.sx).collect(),
                    sy: pencils.iter().map(|p| p.sy).collect(),
                    energy: pencils.iter().map(|p| p.energy).collect(),
                    protons: pencils
                        .iter()
                        .map(|p| match quantity {
                            Quantity::Protons => p.amount,
                            _ => protons_from_mu(
                                f64::from(p.amount),
                                f64::from(p.energy),
                                f64::from(gap_mm),
                                f64::from(k_mu),
                            ) as f32,
                        })
                        .collect(),
                    histories,
                    seed: MC_SEED,
                    spread_pct: spread,
                    wet_mm,
                    depth_mm: depth,
                    voxel_mm: voxel,
                    origin,
                    shape,
                });
                if let Ok(result) = run(&job) {
                    let volume = result.volume.clone();
                    return (volume, Some(result));
                }
            }
        }
    }
    (
        analytic_on(
            mat,
            pencils,
            quantity,
            spread,
            scatter && model != "mc",
            wet_mm,
            phantom_mm,
            1.0,
            k_mu,
            gap_mm,
            grid,
            planes,
        ),
        None,
    )
}

fn load_cloud(
    root: &Path,
    session: &str,
    sessions: &[String],
    grain: &str,
    xy: &str,
    plan_sigma: &str,
    sigma_ref: &str,
) -> Cloud {
    let table = if grain == "timeslice" {
        super::tables::timeslice_signals(root, session)
    } else {
        spot_table(root, session)
    };
    let energy = col(&table, "energy");
    let n = energy.len();
    let target = col(&table, "target_mu");
    let dose = col(&table, "ic1_dose");
    let (mx, my, msx, msy) = match xy {
        "ic2" => (
            col(&table, "ic2_x"),
            col(&table, "ic2_y"),
            col(&table, "ic2_sig_x"),
            col(&table, "ic2_sig_y"),
        ),
        "iso_ray" => iso_ray(&table),
        "plan" => (
            col(&table, "plan_x"),
            col(&table, "plan_y"),
            Vec::new(),
            Vec::new(),
        ),
        _ => (
            or_plan(&table, "ic1_x"),
            or_plan_y(&table, "ic1_y"),
            col(&table, "ic1_sig_x"),
            col(&table, "ic1_sig_y"),
        ),
    };
    let xml = devices_xml(root, session);
    let measured = pencils_from(
        &energy, &mx, &my, &msx, &msy, &dose, &target, &xml, n, false,
    );
    let donor = if plan_sigma == "reference" {
        let id = if sessions.iter().any(|s| s == sigma_ref) {
            sigma_ref
        } else {
            sessions.first().map(String::as_str).unwrap_or(session)
        };
        spot_table(root, id)
    } else {
        table.clone()
    };
    let (psx, psy) = if plan_sigma == "interlock" {
        (Vec::new(), Vec::new())
    } else {
        (col(&donor, "ic1_sig_x"), col(&donor, "ic1_sig_y"))
    };
    let plan = pencils_from(
        &energy,
        &col(&table, "plan_x"),
        &col(&table, "plan_y"),
        &psx,
        &psy,
        &dose,
        &target,
        &xml,
        n,
        plan_sigma == "interlock",
    );
    let pencils = if xy == "plan" { plan.clone() } else { measured };
    Cloud { pencils, plan }
}

fn pencils_from(
    energy: &[f32],
    x: &[f32],
    y: &[f32],
    sx: &[f32],
    sy: &[f32],
    dose: &[f32],
    target: &[f32],
    xml: &str,
    n: usize,
    interlock: bool,
) -> Vec<Pencil> {
    let mut out = Vec::new();
    for i in 0..n {
        let e = energy.get(i).copied().unwrap_or(f32::NAN);
        let px = x.get(i).copied().unwrap_or(f32::NAN);
        let py = y.get(i).copied().unwrap_or(f32::NAN);
        // Python's dose loader treats |xy| past 0.9 × 10000 mm as a missing reading.
        if !e.is_finite()
            || !px.is_finite()
            || !py.is_finite()
            || e <= 1.0
            || px.abs() >= ABS_INVALID_MM
            || py.abs() >= ABS_INVALID_MM
        {
            continue;
        }
        let delivered = dose.get(i).copied().unwrap_or(f32::NAN);
        let mu = target.get(i).copied().unwrap_or(1.0);
        let amount = if delivered.is_finite() && delivered > 0.0 {
            delivered
        } else if mu.is_finite() {
            mu
        } else {
            1.0
        };
        let sig_x = sigma_at(sx, i, e, xml, "IC_1_X", interlock);
        let sig_y = sigma_at(sy, i, e, xml, "IC_1_Y", interlock);
        out.push(Pencil {
            x: px,
            y: py,
            sx: sig_x,
            sy: sig_y,
            energy: e,
            amount,
        });
    }
    out
}

fn sigma_at(
    values: &[f32],
    i: usize,
    energy: f32,
    xml: &str,
    device: &str,
    interlock: bool,
) -> f32 {
    if interlock {
        if let Some(mm) = interlock_sigma_mm(xml, device, f64::from(energy)) {
            return mm as f32;
        }
    }
    values
        .get(i)
        .copied()
        .filter(|v| v.is_finite() && *v > 0.2)
        .unwrap_or(FALLBACK_SIGMA)
}

fn finite_median(values: &[f32]) -> f32 {
    let mut kept: Vec<f32> = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    if kept.is_empty() {
        return 0.0;
    }
    kept.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = kept.len() / 2;
    if kept.len().is_multiple_of(2) {
        (kept[mid - 1] + kept[mid]) * 0.5
    } else {
        kept[mid]
    }
}

fn shifted(values: &[f32]) -> Vec<f32> {
    let offset = finite_median(values);
    values.iter().map(|value| value - offset).collect()
}

/// Plan-crossing isocenter, else the IC1 plane. Chamber medians are removed later.
fn iso_plane_z(table: &std::collections::BTreeMap<String, Vec<f32>>) -> f32 {
    let zx = axis_iso_z(
        &col(table, "ic2_x"),
        &col(table, "ic1_x"),
        &col(table, "plan_x"),
    );
    let zy = axis_iso_z(
        &col(table, "ic2_y"),
        &col(table, "ic1_y"),
        &col(table, "plan_y"),
    );
    match (zx, zy) {
        (Some(x), Some(y)) => finite_median(&[x, y]),
        (Some(z), None) | (None, Some(z)) => z,
        (None, None) => IC1_Z_MM,
    }
}

fn axis_iso_z(near: &[f32], far: &[f32], plan: &[f32]) -> Option<f32> {
    let n = near.len().min(far.len()).min(plan.len());
    let mut zs = Vec::new();
    for i in 0..n {
        let (a, b, spot) = (near[i], far[i], plan[i]);
        if !a.is_finite() || !b.is_finite() || !spot.is_finite() {
            continue;
        }
        let slope = (b - a) / IC_SEP_MM;
        if slope.abs() < 1e-6 {
            continue;
        }
        let z = IC2_Z_MM + (spot - a) / slope;
        if z.is_finite() && z > IC2_Z_MM {
            zs.push(z);
        }
    }
    if zs.is_empty() {
        None
    } else {
        Some(finite_median(&zs))
    }
}

fn iso_ray(
    table: &std::collections::BTreeMap<String, Vec<f32>>,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let x2 = col(table, "ic2_x");
    let x1 = col(table, "ic1_x");
    let y2 = col(table, "ic2_y");
    let y1 = col(table, "ic1_y");
    if x1.is_empty() || x2.is_empty() {
        return (
            or_plan(table, "ic1_x"),
            or_plan_y(table, "ic1_y"),
            col(table, "ic1_sig_x"),
            col(table, "ic1_sig_y"),
        );
    }
    let z = iso_plane_z(table);
    let t = (z - IC2_Z_MM) / IC_SEP_MM;
    let x2 = shifted(&x2);
    let x1 = shifted(&x1);
    let y2 = shifted(&y2);
    let y1 = shifted(&y1);
    let x = x2
        .iter()
        .zip(x1.iter().chain(std::iter::repeat(&f32::NAN)))
        .map(|(a, b)| a + t * (b - a))
        .collect();
    let y = y2
        .iter()
        .zip(y1.iter().chain(std::iter::repeat(&f32::NAN)))
        .map(|(a, b)| a + t * (b - a))
        .collect();
    let sx = lerp_sigma(&col(table, "ic2_sig_x"), &col(table, "ic1_sig_x"), t);
    let sy = lerp_sigma(&col(table, "ic2_sig_y"), &col(table, "ic1_sig_y"), t);
    (x, y, sx, sy)
}

fn lerp_sigma(near: &[f32], far: &[f32], t: f32) -> Vec<f32> {
    let n = near.len().max(far.len());
    (0..n)
        .map(|i| {
            let a = near.get(i).copied().unwrap_or(f32::NAN);
            let b = far.get(i).copied().unwrap_or(f32::NAN);
            if a.is_finite() && b.is_finite() {
                (1.0 - t) * a + t * b
            } else if a.is_finite() {
                a
            } else {
                b
            }
        })
        .collect()
}

fn or_plan(table: &std::collections::BTreeMap<String, Vec<f32>>, key: &str) -> Vec<f32> {
    let got = col(table, key);
    if got.iter().any(|v| v.is_finite()) {
        got
    } else {
        col(table, "plan_x")
    }
}

fn or_plan_y(table: &std::collections::BTreeMap<String, Vec<f32>>, key: &str) -> Vec<f32> {
    let got = col(table, key);
    if got.iter().any(|v| v.is_finite()) {
        got
    } else {
        col(table, "plan_y")
    }
}

fn col(table: &std::collections::BTreeMap<String, Vec<f32>>, key: &str) -> Vec<f32> {
    table.get(key).cloned().unwrap_or_default()
}

fn devices_xml(root: &Path, session: &str) -> String {
    let dir = discover::session_directory(root, session);
    std::fs::read_to_string(dir.join("config/map2map/devices.xml")).unwrap_or_default()
}

fn read_kmu(root: &Path, session: &str) -> f32 {
    let xml = devices_xml(root, session);
    let mut hcc = None;
    let mut any = None;
    for chunk in xml.split("<ion_chamber").skip(1) {
        let head = chunk.split('>').next().unwrap_or("");
        let name = attr(head, "name").unwrap_or_default();
        if let Some(k) = chunk
            .split("K_MU=\"")
            .nth(1)
            .and_then(|rest| rest.split('"').next())
        {
            if let Ok(value) = k.parse::<f32>() {
                if value.is_finite() && value != 0.0 {
                    if name.eq_ignore_ascii_case("IC_1_HCC") {
                        hcc = Some(value);
                    }
                    any.get_or_insert(value);
                }
            }
        }
    }
    hcc.or(any).unwrap_or(FALLBACK_KMU)
}

fn attr<'a>(tag: &'a str, name: &str) -> Option<&'a str> {
    let key = format!("{name}=\"");
    let start = tag.find(&key)? + key.len();
    let rest = &tag[start..];
    Some(&rest[..rest.find('"')?])
}

fn difference(measured: &Volume, plan: &Volume) -> Volume {
    let mut values = measured.values.clone();
    for (slot, plan) in values.iter_mut().zip(&plan.values) {
        *slot -= *plan;
    }
    Volume {
        values,
        origin: measured.origin,
        shape: measured.shape,
        voxel: measured.voxel,
    }
}

fn push_dvh(lines: &mut Vec<Series>, volume: &Volume) {
    let mask = vec![true; volume.values.len()];
    let (edges, curve) = dvh(&volume.values, &mask, 32);
    let xs = edges.iter().take(curve.len()).copied().collect();
    lines.push(line(xs, curve));
}

fn gamma_criteria(options: &Value) -> (f32, f32, f32) {
    let dd = pick(options, "dd", "3", &[("2", "2%"), ("3", "3%"), ("5", "5%")])
        .parse::<f32>()
        .unwrap_or(3.0);
    let dta = pick(options, "dta", "2", &[("2", "2 mm"), ("3", "3 mm")])
        .parse::<f32>()
        .unwrap_or(2.0);
    let cutoff = pick(
        options,
        "cutoff",
        "10",
        &[("5", "5%"), ("10", "10%"), ("20", "20%")],
    )
    .parse::<f32>()
    .unwrap_or(10.0);
    (dd, dta, cutoff)
}

/// Measured dose is the reference. The plan is what the search samples.
fn gamma_with(
    measured: &Volume,
    plan: &Volume,
    dose_percent: f32,
    distance_mm: f32,
    cutoff_pct: f32,
) -> (Vec<f32>, u32, u32) {
    let n = measured.values.len().min(plan.values.len());
    gamma_index(
        &measured.values[..n],
        &plan.values[..n],
        measured.shape,
        dose_percent,
        distance_mm,
        [measured.voxel, measured.voxel, measured.voxel],
        cutoff_pct,
    )
}

fn gamma_image(template: &Volume, values: &[f32]) -> Volume {
    Volume {
        origin: template.origin,
        shape: template.shape,
        voxel: template.voxel,
        values: values.to_vec(),
    }
}

fn window(volumes: &[Volume], compare: &str) -> (f32, f32) {
    if compare == "gamma" {
        return (0.0, 2.0);
    }
    let mut flat = Vec::new();
    for volume in volumes {
        if compare == "difference" {
            flat.extend(
                volume
                    .values
                    .iter()
                    .copied()
                    .filter(|value| *value != 0.0)
                    .map(f32::abs),
            );
        } else {
            flat.extend(volume.values.iter().copied().filter(|value| *value > 0.0));
        }
    }
    if compare == "difference" {
        let hi = robust_high(&flat);
        (-hi, hi)
    } else {
        (0.0, robust_high(&flat))
    }
}

fn line(xs: Vec<f32>, ys: Vec<f32>) -> Series {
    Series::Polyline {
        xs,
        ys,
        color: MARK,
        thickness: 1.5,
    }
}

fn note(title: &str) -> Panel {
    Panel {
        title: title.into(),
        y_label: String::new(),
        x_label: String::new(),
        xmin: 0.0,
        xmax: 1.0,
        ymin: 0.0,
        ymax: 1.0,
        series: Vec::new(),
        x_labels: Vec::new(),
        equal: false,
    }
}

fn volume_unit(quantity: Quantity, mode: u8, compare: &str) -> &'static str {
    if compare == "gamma" {
        return "γ";
    }
    match quantity {
        Quantity::Dose if mode == 0 => "Gy·mm",
        Quantity::Dose => "Gy",
        Quantity::Mu => "MU/mm³",
        Quantity::Protons => "protons/mm³",
    }
}

fn y_name(quantity: Quantity) -> &'static str {
    match quantity {
        Quantity::Dose => "Gy",
        Quantity::Mu => "MU / mm³",
        Quantity::Protons => "Protons / mm³",
    }
}

fn trim_num(value: f32) -> String {
    if (value - 0.5).abs() < 1e-3 {
        "0.5".into()
    } else if (value - 2.0).abs() < 1e-3 {
        "2".into()
    } else {
        "1".into()
    }
}

#[cfg(test)]
mod tests {
    use scan_kit_core::{dose_frame, water, Pencil, Quantity, Volume};

    use super::{lattice_planes, next_preview_voxel};

    #[test]
    fn a_heavy_field_is_still_the_full_lattice() {
        let pencils: Vec<_> = (0..40)
            .map(|i| Pencil {
                x: i as f32 * 5.0,
                y: 0.0,
                sx: 4.0,
                sy: 4.0,
                energy: 180.0,
                amount: 0.05,
            })
            .collect();
        let frame = dose_frame(
            water(),
            &pencils,
            Quantity::Dose,
            1.0,
            true,
            0.0,
            0.0,
            1.0,
            2.0e-8,
            10.0,
            0.0,
        );
        assert!(frame.planes_only(), "visits {}", frame.visits);
        assert!(lattice_planes(&frame).is_none());
        assert_eq!(next_preview_voxel(1.0, frame.visits), Some(2.0));
        assert_eq!(next_preview_voxel(1.0, 1_000), None);
        assert_eq!(next_preview_voxel(4.0, frame.visits), None);
    }

    #[test]
    fn a_heavy_preview_opens_as_a_coarser_cube() {
        let root = std::env::temp_dir().join(format!(
            "scan-kit-preview-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&root);
        let session = root.join("sess");
        std::fs::create_dir_all(&session).unwrap();
        let mut map = String::from("energy,charge_req,position_x,position_y\n");
        let mut spots = String::from("ic1_total_dose,ic2_total_dose,position_x,position_y\n");
        for i in 0..40 {
            let x = i * 5;
            map.push_str(&format!("180,0.05,{x},0\n"));
            spots.push_str(&format!("0.05,0.05,{x},0\n"));
        }
        std::fs::write(session.join("input_map.csv"), map).unwrap();
        std::fs::write(session.join("spot_data.csv"), spots).unwrap();
        let scene = super::volumetric(
            &root,
            &["sess".into()],
            &serde_json::json!({ "_preview": true }),
            None,
        );
        let _ = std::fs::remove_dir_all(&root);
        assert!(
            (scene.volume.voxel - 2.0).abs() < 0.05,
            "voxel {}",
            scene.volume.voxel
        );
        let [nx, ny, nz] = scene.volume.shape;
        let nx = nx as usize;
        let ny = ny as usize;
        let nz = nz as usize;
        let (px, py, pz) = {
            let mut best = 0.0f32;
            let mut at = (0, 0, 0);
            for z in 0..nz {
                for y in 0..ny {
                    for x in 0..nx {
                        let value = scene.volume.values[x + nx * (y + ny * z)];
                        if value > best {
                            best = value;
                            at = (x, y, z);
                        }
                    }
                }
            }
            at
        };
        let off = scene
            .volume
            .values
            .iter()
            .enumerate()
            .filter(|(index, value)| {
                if **value <= 0.0 {
                    return false;
                }
                let x = index % nx;
                let y = (index / nx) % ny;
                let z = index / (nx * ny);
                x != px && y != py && z != pz
            })
            .count();
        assert!(off > 100, "off-plane voxels {off}");
    }

    #[test]
    fn a_sentinel_position_does_not_stretch_the_lattice() {
        let root = std::env::temp_dir().join(format!(
            "scan-kit-sentinel-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&root);
        let session = root.join("sess");
        std::fs::create_dir_all(&session).unwrap();
        std::fs::write(
            session.join("input_map.csv"),
            "energy,charge_req,position_x,position_y\n180,0.05,0,0\n180,0.05,10000,0\n",
        )
        .unwrap();
        std::fs::write(
            session.join("spot_data.csv"),
            "ic1_total_dose\n0.05\n0.05\n",
        )
        .unwrap();
        let scene = super::volumetric(&root, &["sess".into()], &serde_json::json!({}), None);
        let _ = std::fs::remove_dir_all(&root);
        let span = scene.volume.shape[0] as f32 * scene.volume.voxel;
        assert!(span < 120.0, "x span {span} mm");
    }

    #[test]
    fn a_sentinel_sigma_drops_the_spot() {
        let root = std::env::temp_dir().join(format!(
            "scan-kit-sigma-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&root);
        let session = root.join("sess");
        std::fs::create_dir_all(&session).unwrap();
        std::fs::write(
            session.join("input_map.csv"),
            "energy,charge_req,position_x,position_y\n180,0.05,0,0\n180,0.05,200,0\n",
        )
        .unwrap();
        std::fs::write(
            session.join("spot_data.csv"),
            "ic1_total_dose,r_ic1_x_spot_sigma,r_ic1_y_spot_sigma\n0.05,2,2\n0.05,-1,2\n",
        )
        .unwrap();
        let scene = super::volumetric(&root, &["sess".into()], &serde_json::json!({}), None);
        let _ = std::fs::remove_dir_all(&root);
        let span = scene.volume.shape[0] as f32 * scene.volume.voxel;
        assert!(span < 120.0, "x span {span} mm");
    }

    #[test]
    fn gamma_scores_measured_points_and_searches_the_plan() {
        let measured = Volume {
            origin: [0.0, 0.0, 0.0],
            shape: [2, 2, 1],
            voxel: 1.0,
            values: vec![1.0, 0.0, 0.0, 0.0],
        };
        let plan = Volume {
            values: vec![1.0, 1.0, 1.0, 1.0],
            ..measured.clone()
        };
        let (_, passed, scored) = super::gamma_with(&measured, &plan, 3.0, 2.0, 10.0);
        assert_eq!((scored, passed), (1, 1));
    }

    #[test]
    fn ic_gap_scales_the_deposited_dose() {
        let root = std::env::temp_dir().join(format!(
            "scan-kit-gap-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&root);
        let session = root.join("sess");
        std::fs::create_dir_all(&session).unwrap();
        std::fs::write(
            session.join("input_map.csv"),
            "energy,charge_req,position_x,position_y\n180,0.05,0,0\n",
        )
        .unwrap();
        std::fs::write(session.join("spot_data.csv"), "ic1_total_dose\n0.05\n").unwrap();
        let peak = |gap: &str| {
            let scene = super::volumetric(
                &root,
                &["sess".into()],
                &serde_json::json!({ "gap": gap, "voxel": "2" }),
                None,
            );
            scene.volume.values.iter().copied().fold(0.0f32, f32::max)
        };
        let narrow = peak("5");
        let wide = peak("20");
        let _ = std::fs::remove_dir_all(&root);
        assert!(narrow > 0.0 && wide > 0.0, "narrow {narrow} wide {wide}");
        assert!(
            narrow > wide * 2.0,
            "a 5 mm gap ({narrow}) should deposit more than a 20 mm gap ({wide})"
        );
    }

    #[test]
    fn the_sidebar_follows_the_dose_workflow() {
        let root = std::env::temp_dir().join(format!(
            "scan-kit-sidebar-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&root);
        let session = root.join("sess");
        std::fs::create_dir_all(&session).unwrap();
        std::fs::write(
            session.join("input_map.csv"),
            "energy,charge_req,position_x,position_y\n180,0.05,0,0\n",
        )
        .unwrap();
        std::fs::write(session.join("spot_data.csv"), "ic1_total_dose\n0.05\n").unwrap();
        let scene = super::volumetric(&root, &["sess".into()], &serde_json::json!({}), None);
        let titles = fieldsets(&scene.controls);
        assert_eq!(
            titles,
            ["Dose", "Calculation", "Phantom", "Field", "Picture"]
        );
        assert_eq!(
            scene
                .controls
                .iter()
                .filter(|control| control.id == "phantom")
                .count(),
            1
        );
        assert!(scene
            .controls
            .iter()
            .any(|control| control.id == "phantom_box"));
        assert!(scene
            .controls
            .iter()
            .all(|control| control.id != "dd" && control.group != "Gamma"));
        let boxed = super::volumetric(
            &root,
            &["sess".into()],
            &serde_json::json!({ "phantom_box": "On", "phantom": "100" }),
            None,
        );
        assert!(boxed.volume.show_phantom);
        assert!(boxed
            .controls
            .iter()
            .any(|control| { control.id == "phantom" && control.value.starts_with("100") }));
        let old = super::volumetric(
            &root,
            &["sess".into()],
            &serde_json::json!({ "phantom": "On" }),
            None,
        );
        assert!(old.volume.show_phantom);
        let gamma = super::volumetric(
            &root,
            &["sess".into()],
            &serde_json::json!({ "compare": "gamma" }),
            None,
        );
        let gamma_sets = fieldsets(&gamma.controls);
        assert_eq!(
            gamma_sets,
            [
                "Dose",
                "Calculation",
                "Phantom",
                "Field",
                "Picture",
                "Gamma"
            ]
        );
        assert!(gamma
            .controls
            .iter()
            .all(|control| control.id != "scale" && control.id != "level" && control.id != "auto"));
        assert_eq!(
            scene
                .controls
                .iter()
                .find(|control| control.id == "model")
                .map(|control| control.kind.as_str()),
            Some("radio")
        );
        assert!(scene
            .controls
            .iter()
            .all(|control| control.id != "histories"));
        let mc = super::volumetric(
            &root,
            &["sess".into()],
            &serde_json::json!({ "model": "mc" }),
            None,
        );
        assert_eq!(
            mc.controls
                .iter()
                .find(|control| control.id == "histories")
                .map(|control| control.kind.as_str()),
            Some("radio")
        );
        assert!(mc.controls.iter().all(|control| control.id != "scatter"));
        let _ = std::fs::remove_dir_all(&root);
    }

    fn fieldsets(controls: &[scan_kit_core::Control]) -> Vec<&str> {
        let mut titles = Vec::new();
        for control in controls {
            if control.group == "Cell" || control.group == "Plot" {
                continue;
            }
            if titles.last() != Some(&control.group.as_str()) {
                titles.push(control.group.as_str());
            }
        }
        titles
    }
}
