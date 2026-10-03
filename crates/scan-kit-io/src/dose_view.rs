//! Dose Volume: analytic Bragg dose, and a Monte Carlo fill when the compute
//! crate supplies a runner. The sidebar is the analysis shell.

use std::path::Path;

use scan_kit_core::{
    analytic_on, choices, dose_frame, dvh, field_bounds, gamma_index, index, is_session, medium,
    protons_from_mu, resolve, robust_high, Family, McJob, McResult, Panel, Pencil, PlotScene,
    Quantity, Series, Volume, IC1_Z_MM, IC2_Z_MM, IC_SEP_MM, SESSION,
};
use serde_json::Value;

use super::discover;
use super::marks::{labeled, pick};
use super::tables::{interlock_sigma_mm, spot_table};

const GUIDE: [f32; 4] = [0.62, 0.62, 0.62, 0.9];
const MARK: [f32; 4] = [0.0, 0.0, 0.0, 1.0];
const FALLBACK_SIGMA: f32 = 4.0;
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
const SPREAD: &[(&str, &str)] = &[("0.5", "0.5%"), ("1", "1%"), ("2", "2%")];
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
    ("measured", "Measured"),
    ("reference", "Reference"),
    ("interlock", "Interlock"),
];
const EDGE: &[(&str, &str)] = &[
    ("slice50", "Slice 50%"),
    ("peak90", "Peak 90%"),
    ("peak50", "Peak 50%"),
    ("slice20", "Slice 20%"),
    ("plan90", "Plan 90%"),
];

pub(crate) type McRunner<'a> = &'a dyn Fn(&McJob) -> Result<McResult, String>;

struct Cloud {
    pencils: Vec<Pencil>,
    plan: Vec<Pencil>,
}

pub fn dose_volume(
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
    let clouds: Vec<Cloud> = session_ids
        .iter()
        .map(|session| load_cloud(root, session, session_ids, grain, xy, plan_sigma, sigma_ref))
        .collect();
    let k_mu = session_ids
        .first()
        .map(|session| read_kmu(root, session))
        .unwrap_or(FALLBACK_KMU);

    let all: Vec<Pencil> = clouds
        .iter()
        .flat_map(|cloud| {
            cloud
                .pencils
                .iter()
                .copied()
                .chain(cloud.plan.iter().copied())
        })
        .collect();
    let frame = dose_frame(
        mat,
        &all,
        quantity,
        spread,
        scatter && model != "mc",
        wet_mm,
        phantom_mm,
        1.0,
        k_mu,
        GAP_MM,
    );
    let grid = Some((frame.origin, frame.shape, frame.voxel));
    let planes = frame.planes_only().then_some(frame.focus);

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
            model,
            histories,
            medium_key,
            mc,
            grid,
            planes,
        );
        mc_note = mc_note.or(ran);
        measured.push(volume);
        let (volume, _) = fill(
            mat,
            &cloud.plan,
            quantity,
            spread,
            scatter,
            wet_mm,
            phantom_mm,
            k_mu,
            model,
            histories,
            medium_key,
            mc,
            grid,
            planes,
        );
        plans.push(volume);
    }

    let used_planes = mc_note.is_none() && planes.is_some();
    let gamma_slice = used_planes
        || measured
            .first()
            .is_some_and(|volume| volume.values.len() > 2_000_000);
    let primary = measured
        .first()
        .filter(|volume| volume.values.iter().any(|v| *v > 0.0));
    let (ix, iy, iz) = if used_planes {
        let [nx, ny, nz] = primary.map(|volume| volume.shape).unwrap_or([1, 1, 1]);
        (
            frame.focus[0].min(nx.saturating_sub(1)),
            frame.focus[1].min(ny.saturating_sub(1)),
            frame.focus[2].min(nz.saturating_sub(1)),
        )
    } else {
        primary.map(Volume::peak_index).unwrap_or((0, 0, 0))
    };
    let shown: Vec<Volume> = match compare {
        "difference" => measured
            .iter()
            .zip(&plans)
            .map(|(got, plan)| difference(got, plan))
            .collect(),
        "gamma" if gamma_slice => measured
            .iter()
            .zip(&plans)
            .map(|(got, plan)| gamma_planes(got, plan, frame.focus).0)
            .collect(),
        "gamma" => measured
            .iter()
            .zip(&plans)
            .map(|(got, plan)| gamma_volume(got, plan))
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
    } else {
        let volume = &shown[0];
        let [nx, ny, nz] = volume.shape;
        let iz = iz.min(nz.saturating_sub(1));
        let iy = iy.min(ny.saturating_sub(1));
        let ix = ix.min(nx.saturating_sub(1));
        panels.push(slice_panel(
            "Axial",
            &shown,
            ramp,
            lo,
            hi,
            volume.origin[0],
            volume.origin[0] + nx as f32 * volume.voxel,
            volume.origin[1],
            volume.origin[1] + ny as f32 * volume.voxel,
            true,
            |volume| volume.axial(iz),
            nx as u32,
            ny as u32,
        ));
        panels.push(slice_panel(
            "coronal",
            &shown,
            ramp,
            lo,
            hi,
            volume.origin[0],
            volume.origin[0] + nx as f32 * volume.voxel,
            volume.origin[2],
            volume.origin[2] + nz as f32 * volume.voxel,
            true,
            |volume| volume.coronal(iy),
            nx as u32,
            nz as u32,
        ));
        panels.push(slice_panel(
            "sagittal",
            &shown,
            ramp,
            lo,
            hi,
            volume.origin[1],
            volume.origin[1] + ny as f32 * volume.voxel,
            volume.origin[2],
            volume.origin[2] + nz as f32 * volume.voxel,
            true,
            |volume| volume.sagittal(ix),
            ny as u32,
            nz as u32,
        ));
        let mut depth = Vec::new();
        let mut lateral = Vec::new();
        for volume in &shown {
            let (xs, ys) = volume.depth_profile(ix, iy);
            depth.push(line(xs, ys));
            let (xs, ys) = volume.lateral_profile(iy, iz);
            lateral.push(line(xs, ys));
        }
        let mut depth_panel = lines_panel("Depth", depth, y_name(quantity));
        if let Some(result) = &mc_note {
            let incident = result.ledger[0];
            let left = if incident > 0.0 {
                100.0 * (result.ledger[2] + result.ledger[3]) / incident
            } else {
                0.0
            };
            depth_panel.title = format!(
                "Depth  {:.1}%  left {:.1}%",
                result.uncertainty * 100.0,
                left
            );
        } else if model == "mc" {
            depth_panel.title = "Depth  analytic fallback".into();
        }
        panels.push(depth_panel);
        panels.push(lines_panel("Lateral", lateral, y_name(quantity)));
        let mut dvh_lines = Vec::new();
        if used_planes {
            for volume in coarse_dvh_volumes(
                &clouds,
                &all,
                mat,
                quantity,
                spread,
                scatter && model != "mc",
                wet_mm,
                phantom_mm,
                k_mu,
                compare,
                frame.visits,
            ) {
                push_dvh(&mut dvh_lines, &volume);
            }
        } else {
            for volume in &shown {
                push_dvh(&mut dvh_lines, volume);
            }
        }
        panels.push(lines_panel("DVH", dvh_lines, "Volume"));
        if let (Some(got), Some(plan)) = (measured.first(), plans.first()) {
            let (gamma_vol, passed, scored) = if gamma_slice {
                let iz = iz.min(got.shape[2].saturating_sub(1));
                let [nx, ny, _] = got.shape;
                let (values, passed, scored) =
                    plane_gamma(&plan.axial(iz), &got.axial(iz), nx, ny, got.voxel);
                (
                    Volume {
                        origin: got.origin,
                        shape: [nx, ny, 1],
                        voxel: got.voxel,
                        values,
                    },
                    passed,
                    scored,
                )
            } else {
                let (gamma, passed, scored) = gamma_of(got, plan);
                (gamma_image(got, &gamma), passed, scored)
            };
            let rate = if scored == 0 {
                0.0
            } else {
                100.0 * passed as f32 / scored as f32
            };
            let cols = gamma_vol.shape[0] as u32;
            let rows = gamma_vol.shape[1] as u32;
            let x0 = gamma_vol.origin[0];
            let y0 = gamma_vol.origin[1];
            let x1 = x0 + gamma_vol.shape[0] as f32 * gamma_vol.voxel;
            let y1 = y0 + gamma_vol.shape[1] as f32 * gamma_vol.voxel;
            let gz = if gamma_vol.shape[2] == 1 {
                0
            } else {
                iz.min(gamma_vol.shape[2].saturating_sub(1))
            };
            panels.push(slice_panel(
                &format!("gamma {rate:.0}%"),
                &[gamma_vol],
                16,
                0.0,
                2.0,
                x0,
                x1,
                y0,
                y1,
                true,
                |volume| volume.axial(gz),
                cols,
                rows,
            ));
            if let Some(bounds) =
                field_readout(if edge == "plan90" { plan } else { got }, plan, iz, edge)
            {
                if let Some(axial) = panels.first_mut() {
                    let width = bounds[1] - bounds[0];
                    let height = bounds[3] - bounds[2];
                    axial.title = format!("Axial  {width:.0} × {height:.0} mm");
                    axial.series.push(rect_guide(bounds));
                    let x_mm = volume_cross_x(got, ix);
                    let y_mm = got.origin[1] + (iy as f32 + 0.5) * got.voxel;
                    let y0 = got.origin[1];
                    let y1 = got.origin[1] + got.shape[1] as f32 * got.voxel;
                    let x0 = got.origin[0];
                    let x1 = got.origin[0] + got.shape[0] as f32 * got.voxel;
                    axial
                        .series
                        .push(guide_line(vec![x_mm, x_mm], vec![y0, y1]));
                    axial
                        .series
                        .push(guide_line(vec![x0, x1], vec![y_mm, y_mm]));
                }
            }
        }
    }

    let mut controls = picked.controls;
    controls.push(labeled("xy", "Signal", XY, xy).grouped("Data Source"));
    controls.push(labeled("quantity", "Quantity", QUANTITY, quantity_id).grouped("Data Source"));
    if xy == "plan" || compare != "measured" {
        controls
            .push(labeled("plan_sigma", "Plan Sigma", SIGMA, plan_sigma).grouped("Data Source"));
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
            controls.push(labeled("sigma_ref", "Reference", &refs, chosen).grouped("Data Source"));
        }
    }
    controls.push(labeled("model", "Model", MODEL, model).grouped("Model"));
    if model == "analytic" {
        controls.push(
            labeled(
                "scatter",
                "Scatter",
                SCATTER,
                if scatter { "on" } else { "off" },
            )
            .grouped("Model"),
        );
    } else {
        controls.push(
            labeled("histories", "Histories", HISTORIES, &histories.to_string()).grouped("Model"),
        );
    }
    controls.push(labeled("spread", "Spread", SPREAD, &trim_num(spread)).grouped("Model"));
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
    controls.push(labeled("compare", "Compare", COMPARE, compare).grouped("Compare"));
    controls.push(labeled("edge", "Field Edge", EDGE, edge).grouped("Compare"));
    if session_ids.len() <= 1 && compare != "gamma" {
        let scales = choices(if compare == "difference" {
            Family::Divergent
        } else {
            Family::Sequential
        });
        controls.push(labeled("scale", "Scale", scales, scale).grouped("Color"));
    }
    PlotScene {
        title: "Dose Volume".into(),
        panels,
        controls,
        table: None,
        samples: Vec::new(),
        columns: 3,
        column_weights: Vec::new(),
        row_weights: Vec::new(),
    }
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
                                f64::from(GAP_MM),
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
            GAP_MM,
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
        if !e.is_finite() || !px.is_finite() || !py.is_finite() || e <= 1.0 {
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
    let z = IC1_Z_MM + 0.5 * IC_SEP_MM;
    let t = (z - IC2_Z_MM) / IC_SEP_MM;
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
        ..measured.clone()
    }
}

fn gamma_volume(measured: &Volume, plan: &Volume) -> Volume {
    let (values, _, _) = gamma_of(measured, plan);
    gamma_image(measured, &values)
}

fn push_dvh(lines: &mut Vec<Series>, volume: &Volume) {
    let mask = vec![true; volume.values.len()];
    let (edges, curve) = dvh(&volume.values, &mask, 32);
    let xs = edges.iter().take(curve.len()).copied().collect();
    lines.push(line(xs, curve));
}

fn coarse_voxel(visits: u64, voxel: f32) -> f32 {
    if visits <= 4_000_000 {
        return voxel;
    }
    let scale = (visits as f64 / 4_000_000.0).cbrt() as f32;
    (voxel * scale).clamp(voxel, 10.0)
}

fn coarse_dvh_volumes(
    clouds: &[Cloud],
    all: &[Pencil],
    mat: scan_kit_core::Medium,
    quantity: Quantity,
    spread: f32,
    scatter: bool,
    wet_mm: f32,
    phantom_mm: f32,
    k_mu: f32,
    compare: &str,
    visits: u64,
) -> Vec<Volume> {
    let vox = coarse_voxel(visits, 1.0);
    let frame = dose_frame(
        mat, all, quantity, spread, scatter, wet_mm, phantom_mm, vox, k_mu, GAP_MM,
    );
    let grid = Some((frame.origin, frame.shape, frame.voxel));
    let paint = |pencils: &[Pencil]| {
        analytic_on(
            mat, pencils, quantity, spread, scatter, wet_mm, phantom_mm, vox, k_mu, GAP_MM, grid,
            None,
        )
    };
    if compare == "difference" {
        clouds
            .iter()
            .map(|cloud| difference(&paint(&cloud.pencils), &paint(&cloud.plan)))
            .collect()
    } else {
        clouds.iter().map(|cloud| paint(&cloud.pencils)).collect()
    }
}

// ponytail: a clinical lattice's 3D gamma (thousands of offsets times the
// high-dose box) does not return. Large maps score the peak slice in plane.
// Upgrade path: search only the high-dose box on the GPU.
fn plane_gamma(
    reference: &[f32],
    evaluated: &[f32],
    cols: usize,
    rows: usize,
    voxel: f32,
) -> (Vec<f32>, u32, u32) {
    gamma_index(
        reference,
        evaluated,
        [cols, rows, 1],
        3.0,
        2.0,
        [voxel, voxel, voxel],
        10.0,
    )
}

fn gamma_planes(measured: &Volume, plan: &Volume, focus: [usize; 3]) -> (Volume, u32, u32) {
    let [nx, ny, nz] = measured.shape;
    let ix = focus[0].min(nx.saturating_sub(1));
    let iy = focus[1].min(ny.saturating_sub(1));
    let iz = focus[2].min(nz.saturating_sub(1));
    let (axial, passed, scored) =
        plane_gamma(&plan.axial(iz), &measured.axial(iz), nx, ny, measured.voxel);
    let (coronal, _, _) = plane_gamma(
        &plan.coronal(iy),
        &measured.coronal(iy),
        nx,
        nz,
        measured.voxel,
    );
    let (sagittal, _, _) = plane_gamma(
        &plan.sagittal(ix),
        &measured.sagittal(ix),
        ny,
        nz,
        measured.voxel,
    );
    let mut values = vec![0.0; measured.values.len()];
    for y in 0..ny {
        for x in 0..nx {
            values[x + nx * (y + ny * iz)] = axial[x + nx * y];
        }
    }
    for z in 0..nz {
        if z == iz {
            continue;
        }
        for x in 0..nx {
            values[x + nx * (iy + ny * z)] = coronal[x + nx * z];
        }
    }
    for z in 0..nz {
        if z == iz {
            continue;
        }
        for y in 0..ny {
            if y == iy {
                continue;
            }
            values[ix + nx * (y + ny * z)] = sagittal[y + ny * z];
        }
    }
    (
        Volume {
            origin: measured.origin,
            shape: measured.shape,
            voxel: measured.voxel,
            values,
        },
        passed,
        scored,
    )
}

fn gamma_of(measured: &Volume, plan: &Volume) -> (Vec<f32>, u32, u32) {
    let n = measured.values.len().min(plan.values.len());
    gamma_index(
        &plan.values[..n],
        &measured.values[..n],
        measured.shape,
        3.0,
        2.0,
        [measured.voxel, measured.voxel, measured.voxel],
        10.0,
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

fn field_readout(volume: &Volume, plan: &Volume, iz: usize, edge: &str) -> Option<[f32; 4]> {
    let (fraction, per_slice, from_plan) = match edge {
        "peak90" => (0.9, false, false),
        "peak50" => (0.5, false, false),
        "slice20" => (0.2, true, false),
        "plan90" => (0.9, false, true),
        _ => (0.5, true, false),
    };
    let source = if from_plan { plan } else { volume };
    let image = source.axial(iz.min(source.shape[2].saturating_sub(1)));
    let peak = if per_slice {
        image.iter().copied().fold(0.0f32, f32::max)
    } else {
        source.values.iter().copied().fold(0.0f32, f32::max)
    };
    field_bounds(
        &image,
        source.shape[0],
        source.shape[1],
        source.origin[0],
        source.origin[1],
        source.voxel,
        fraction * peak,
    )
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

fn slice_panel(
    title: &str,
    volumes: &[Volume],
    ramp: u8,
    lo: f32,
    hi: f32,
    xmin: f32,
    xmax: f32,
    ymin: f32,
    ymax: f32,
    equal: bool,
    image: impl Fn(&Volume) -> Vec<f32>,
    cols: u32,
    rows: u32,
) -> Panel {
    let series = volumes
        .iter()
        .map(|volume| Series::Heatmap {
            values: image(volume),
            cols,
            rows,
            ramp,
            color: if is_session(ramp) {
                MARK
            } else {
                [1.0, 1.0, 1.0, 1.0]
            },
            lo,
            hi,
        })
        .collect();
    Panel {
        title: title.into(),
        y_label: String::new(),
        x_label: String::new(),
        xmin,
        xmax,
        ymin,
        ymax,
        series,
        x_labels: Vec::new(),
        equal,
    }
}

fn lines_panel(title: &str, series: Vec<Series>, y_label: &str) -> Panel {
    let (xmin, xmax, ymin, ymax) = series_span(&series);
    Panel {
        title: title.into(),
        y_label: y_label.into(),
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

fn rect_guide(bounds: [f32; 4]) -> Series {
    let [x0, x1, y0, y1] = bounds;
    Series::Guide {
        xs: vec![x0, x1, x1, x0, x0],
        ys: vec![y0, y0, y1, y1, y0],
        color: GUIDE,
        thickness: 1.0,
    }
}

fn guide_line(xs: Vec<f32>, ys: Vec<f32>) -> Series {
    Series::Guide {
        xs,
        ys,
        color: GUIDE,
        thickness: 1.0,
    }
}

fn volume_cross_x(volume: &Volume, ix: usize) -> f32 {
    volume.origin[0] + (ix as f32 + 0.5) * volume.voxel
}

fn series_span(series: &[Series]) -> (f32, f32, f32, f32) {
    let mut xmin = f32::MAX;
    let mut xmax = f32::MIN;
    let mut ymin = f32::MAX;
    let mut ymax = f32::MIN;
    for series in series {
        if let Series::Polyline { xs, ys, .. } = series {
            for (x, y) in xs.iter().zip(ys) {
                if x.is_finite() && y.is_finite() {
                    xmin = xmin.min(*x);
                    xmax = xmax.max(*x);
                    ymin = ymin.min(*y);
                    ymax = ymax.max(*y);
                }
            }
        }
    }
    if !xmin.is_finite() {
        (0.0, 1.0, 0.0, 1.0)
    } else {
        if (xmax - xmin).abs() < 1e-4 {
            xmin -= 0.5;
            xmax += 0.5;
        }
        if (ymax - ymin).abs() < 1e-4 {
            ymin -= 0.5;
            ymax += 0.5;
        }
        (xmin, xmax, ymin, ymax)
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
