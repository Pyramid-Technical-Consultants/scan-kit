//! Volumetric: analytic Bragg dose, and a Monte Carlo fill when the compute
//! crate supplies a runner. The sidebar is the analysis shell.

use std::path::Path;

use scan_kit_core::{
    analytic_on, choices, dose_frame, dvh, gamma_index, medium, protons_from_mu, robust_high,
    trim_number, wash_of, Family, McJob, McResult, Panel, Pencil, PlotScene, Quantity, Series,
    Volume, IC1_Z_MM, IC2_Z_MM, IC_SEP_MM, SESSION,
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

const CHAMBERS: &[(&str, &str)] = &[("independent", "Independent"), ("combined", "Combined")];
const QUANTITY: &[(&str, &str)] = &[("dose", "Dose"), ("mu", "MU"), ("protons", "Protons")];
const MODEL: &[(&str, &str)] = &[("analytic", "Analytic"), ("mc", "Monte Carlo")];

/// `mc` and the `Monte Carlo` label are the same choice.
pub fn model_is_monte_carlo(options: &Value) -> bool {
    pick(options, "model", "analytic", MODEL) == "mc"
}

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

/// Saved Signal values stay on `xy`. A trajectory bookmark is the combined ray.
fn chamber_mode(options: &Value) -> &'static str {
    match options.get("xy").and_then(Value::as_str) {
        Some("combined" | "Combined" | "iso_ray" | "ISO Ray") => "combined",
        _ => "independent",
    }
}

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
    /// Image cube and the solid profile. IC1, or IC2 when IC1 has no spots.
    pencils: Vec<Pencil>,
    /// The other chamber. Empty for the combined ray, and when only one chamber has spots.
    companion: Vec<Pencil>,
    plan: Vec<Pencil>,
    /// The chamber positions fell back to the plan's commanded spots — the
    /// cube is drawn, but gamma/difference against the plan would compare
    /// the plan to itself and must stay "not evaluated".
    commanded: bool,
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
    let allow_spot = session_ids.is_empty()
        || session_ids
            .iter()
            .any(|id| crate::tables::grain_present(root, id, false));
    // Timeslice rows carry no spot geometry (no ic*_x/y, plan_x/y, or dose) —
    // a timeslice selection can only ever build zero pencils, so don't offer it.
    let picked = crate::source::select(
        crate::source::Shape::Source,
        allow_spot,
        false,
        &[],
        options,
    );
    let grain = picked.grain;
    let chambers = chamber_mode(options);
    let quantity_id = pick(options, "quantity", "dose", QUANTITY);
    let quantity = match quantity_id {
        "mu" => Quantity::Mu,
        "protons" => Quantity::Protons,
        _ => Quantity::Dose,
    };
    let model = pick(options, "model", "analytic", MODEL);
    // The MC engine scores physical dose — under model=mc a MU/protons
    // quantity would label Gy values with the wrong unit.
    let quantity = if model == "mc" {
        Quantity::Dose
    } else {
        quantity
    };
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
    let voxel_mm = voxel_spacing(options);
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
    let clouds: Vec<Cloud> = session_ids
        .iter()
        .map(|session| {
            load_cloud(
                root,
                session,
                session_ids,
                grain,
                chambers,
                plan_sigma,
                sigma_ref,
            )
        })
        .collect();
    // K_MU is per session — a recalibrated machine or different units means
    // each cloud needs its own MU->protons factor, not the first session's.
    let k_mus: Vec<f32> = session_ids
        .iter()
        .map(|session| read_kmu(root, session))
        .collect();
    let k_mu = k_mus.first().copied().unwrap_or(FALLBACK_KMU);

    let plots = crate::workspace::plots_from(
        [
            options.get("plot0").and_then(Value::as_str),
            options.get("plot1").and_then(Value::as_str),
        ],
        false,
    );
    let (field_fraction, field_per_slice, field_from_plan) = field_edge(edge);
    let all: Vec<Pencil> = clouds
        .iter()
        .flat_map(|cloud| {
            cloud
                .pencils
                .iter()
                .copied()
                .chain(cloud.companion.iter().copied())
                .chain(cloud.plan.iter().copied())
        })
        .collect();
    let preview = options
        .get("_preview")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let mut voxel_mm = voxel_mm;
    let mut frame = dose_frame(
        mat,
        &all,
        quantity,
        spread,
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
            // The plan and the second chamber are separate deposits on this grid.
            // Budget the preview from the heavier one so a matching plan does not
            // coarsen the cube.
            let Some(next) = next_preview_voxel(
                voxel_mm,
                heaviest_visits(&clouds, frame.visits, |pencils| {
                    dose_frame(
                        mat,
                        pencils,
                        quantity,
                        spread,
                        wet_mm,
                        phantom_mm,
                        voxel_mm,
                        k_mu,
                        gap_mm,
                        margin_sigma,
                    )
                    .visits
                }),
            ) else {
                break;
            };
            voxel_mm = next;
            frame = dose_frame(
                mat,
                &all,
                quantity,
                spread,
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
    let mut companions = Vec::new();
    let mut plans = Vec::new();
    let mut mc_note = None;
    // The focused session's measured, second-chamber, and plan cubes all come
    // from the chosen model, so gamma and the overlays never compare two dose
    // engines. The runner sees one job per fill, in that order. Other sessions
    // stay analytic on every cube.
    // An unmatched session option leaves no focus — the MC run must not
    // silently land on session 0 (scenes are infallible; no Err channel).
    let focus = session_at(options, session_ids).unwrap_or(usize::MAX);
    for (index, cloud) in clouds.iter().enumerate() {
        let runner = if index == focus { mc } else { None };
        let k_mu = k_mus.get(index).copied().unwrap_or(k_mu);
        let (mut volume, mut ran) = fill(
            mat,
            &cloud.pencils,
            quantity,
            spread,
            wet_mm,
            phantom_mm,
            k_mu,
            gap_mm,
            model,
            histories,
            medium_key,
            runner,
            grid,
            planes,
        );
        let (mut companion, ran_c) = if cloud.companion.is_empty() {
            (
                Volume {
                    origin: [0.0; 3],
                    shape: [0, 0, 0],
                    spacing: [1.0; 3],
                    values: Vec::new(),
                },
                None,
            )
        } else {
            fill(
                mat,
                &cloud.companion,
                quantity,
                spread,
                wet_mm,
                phantom_mm,
                k_mu,
                gap_mm,
                model,
                histories,
                medium_key,
                runner,
                grid,
                planes,
            )
        };
        let (mut plan, ran_p) = fill(
            mat,
            &cloud.plan,
            quantity,
            spread,
            wet_mm,
            phantom_mm,
            k_mu,
            gap_mm,
            model,
            histories,
            medium_key,
            runner,
            grid,
            planes,
        );
        // A partial MC run leaves one engine per cube — recompute the whole
        // session analytically rather than compare MC against analytic.
        let mut rolled_back = false;
        if index == focus && model == "mc" {
            let attempted = [
                cloud.pencils.is_empty(),
                cloud.companion.is_empty(),
                cloud.plan.is_empty(),
            ]
            .into_iter()
            .filter(|empty| !*empty)
            .count();
            let succeeded = [
                (ran.is_some(), !cloud.pencils.is_empty()),
                (ran_c.is_some(), !cloud.companion.is_empty()),
                (ran_p.is_some(), !cloud.plan.is_empty()),
            ]
            .into_iter()
            .filter(|(done, tried)| *done && *tried)
            .count();
            rolled_back = succeeded > 0 && succeeded < attempted;
            if rolled_back {
                (volume, _) = fill(
                    mat,
                    &cloud.pencils,
                    quantity,
                    spread,
                    wet_mm,
                    phantom_mm,
                    k_mu,
                    gap_mm,
                    model,
                    histories,
                    medium_key,
                    None,
                    grid,
                    planes,
                );
                (plan, _) = fill(
                    mat,
                    &cloud.plan,
                    quantity,
                    spread,
                    wet_mm,
                    phantom_mm,
                    k_mu,
                    gap_mm,
                    model,
                    histories,
                    medium_key,
                    None,
                    grid,
                    planes,
                );
                if !cloud.companion.is_empty() {
                    (companion, _) = fill(
                        mat,
                        &cloud.companion,
                        quantity,
                        spread,
                        wet_mm,
                        phantom_mm,
                        k_mu,
                        gap_mm,
                        model,
                        histories,
                        medium_key,
                        None,
                        grid,
                        planes,
                    );
                }
                ran = None;
            }
        }
        // Any fill's MC note counts - a plan-only run is still MC - but a
        // rolled-back session must not keep one (its cubes are analytic).
        mc_note = mc_note.or(if rolled_back {
            None
        } else {
            ran.or(ran_c).or(ran_p)
        });
        measured.push(volume);
        companions.push(companion);
        plans.push(plan);
    }

    let (dd, dta, cutoff) = gamma_criteria(options);
    // Measured is the reference and the plan is searched, as in the Python dose view.
    let want_gamma = compare == "gamma" || plots.contains(&crate::workspace::PlotKind::Gamma);
    let gamma_fields: Vec<(Volume, f32)> = if want_gamma {
        measured
            .iter()
            .zip(&plans)
            .enumerate()
            .map(|(index, (got, plan))| {
                // A cloud drawn on commanded positions is the plan compared
                // to itself — always "not evaluated", never a pass.
                let (values, passed, scored) =
                    if clouds.get(index).is_some_and(|cloud| cloud.commanded) {
                        (vec![f32::NAN; got.values.len()], 0, 0)
                    } else {
                        gamma_with(got, plan, dd, dta, cutoff)
                    };
                // Zero scored voxels is "not evaluated", not a 0% fail —
                // NaN keeps the verdict text from asserting one.
                let rate = if scored == 0 {
                    f32::NAN
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
            .enumerate()
            .map(|(index, (got, plan))| {
                if clouds.get(index).is_some_and(|cloud| cloud.commanded) {
                    Volume {
                        values: vec![f32::NAN; got.values.len()],
                        ..got.clone()
                    }
                } else {
                    difference(got, plan)
                }
            })
            .collect(),
        "gamma" => gamma_fields
            .iter()
            .map(|(volume, _)| volume.clone())
            .collect(),
        _ => measured.clone(),
    };
    let sessions = shown.len().max(1);
    let (lo, hi) = window(&shown, compare);
    let level = options
        .get("level")
        .and_then(Value::as_str)
        .and_then(|text| text.parse::<f32>().ok());
    let wash = wash_of(scan_kit_core::PaintChoice {
        base_lo: lo,
        base_hi: hi,
        gamma: compare == "gamma",
        difference: compare == "difference",
        scale,
        auto: super::marks::flag(options, "auto", true),
        level,
        percent: pick(
            options,
            "error",
            "absolute",
            &[("absolute", "Absolute"), ("percent", "Percent")],
        ) == "percent",
        ray: pick(
            options,
            "ray",
            "integrate",
            &[
                ("integrate", "Integrate"),
                ("maximum", "Maximum"),
                ("transparent", "Transparent"),
            ],
        ),
        sample: pick(
            options,
            "sample",
            "linear",
            &[
                ("nearest", "Nearest"),
                ("linear", "Linear"),
                ("cubic", "Cubic"),
            ],
        ),
    });
    let ramp = if sessions > 1 && compare != "gamma" {
        SESSION
    } else {
        wash.ramp
    };
    let mut panels = Vec::new();
    let has_dose = measured
        .iter()
        .any(|volume| volume.values.iter().any(|value| *value > 0.0));
    if !has_dose {
        panels.push(note("No spots"));
    }

    let mut volume_mark = scan_kit_core::VolumeMark::default();
    let mut integral_rates: Option<[f32; 3]> = None;
    let mut field_size: Option<String> = None;
    let session_at = match session_at(options, session_ids) {
        Ok(index) => index.min(shown.len().saturating_sub(1)),
        // A stale persisted session must not render session 0 as the focus -
        // usize::MAX makes every .get below miss, matching the MC focus.
        Err(_) => usize::MAX,
    };
    // A hidden or removed focus session still owes the view a panel -
    // an empty scene leaves the last picture painted like it is live.
    if has_dose && shown.get(session_at).is_none() {
        panels.push(note("The selected session is not in this view"));
    }
    // The entry exists but its cloud produced no pencils — a [0,0,0] cube
    // paints six dead panels indistinguishable from a render bug.
    if has_dose
        && shown
            .get(session_at)
            .is_some_and(|volume| volume.values.is_empty())
    {
        panels.push(note("The selected session has no dose"));
    }
    if has_dose {
        if let Some(volume) = shown
            .get(session_at)
            .filter(|volume| !volume.values.is_empty())
        {
            let scan = scan_kit_core::scan_volume(volume);
            let [ix, iy, iz] = scan.peak_at;
            let cells = crate::workspace::cells_from(&[
                options.get("cell0").and_then(Value::as_str),
                options.get("cell1").and_then(Value::as_str),
                options.get("cell2").and_then(Value::as_str),
                options.get("cell3").and_then(Value::as_str),
            ]);
            let mut dvh_lines = Vec::new();
            // `volume` is the displayed cube — under compare:gamma or
            // difference it is a residual field, not dose. DVH needs the
            // measured one.
            if plots.contains(&crate::workspace::PlotKind::Dvh) {
                if let Some(dose) = measured.get(session_at) {
                    push_dvh(&mut dvh_lines, dose);
                }
            }
            let gamma = plots
                .contains(&crate::workspace::PlotKind::Gamma)
                .then(|| gamma_fields.get(session_at).cloned())
                .flatten();
            let field_dose = if field_from_plan {
                plans.get(session_at)
            } else {
                measured.get(session_at)
            };
            let extent = field_dose.and_then(|dose| {
                crate::workspace::field_extent(dose, field_fraction, field_per_slice)
            });
            let mode = wash.mode;
            let filter = wash.filter;
            let (lo_paint, hi_paint, gain, opacity) = (wash.lo, wash.hi, wash.gain, wash.opacity);
            let (built, mut mark) = crate::workspace::assemble(crate::workspace::Workspace {
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
                y_label: match compare {
                    "gamma" => "gamma index".into(),
                    "difference" => format!("{} difference", y_name(quantity)),
                    _ => y_name(quantity).into(),
                },
                dvh: dvh_lines,
                gamma,
            });
            mark.base_lo = lo;
            mark.base_hi = hi;
            mark.gantry = gantry;
            mark.show_phantom = phantom_shown(options);
            // The bounds switch hides this box on the plot. The extent stays so
            // that switch does not start another calculation.
            mark.field = extent.unwrap_or([0.0; 6]);
            mark.unit = volume_unit(quantity, mode, compare).into();
            if volume.spacing.iter().all(|mm| *mm > 0.0) {
                mark.prepared = Some(scan);
            }
            panels = built;
            if let Some(view) = panels
                .iter_mut()
                .find(|panel| panel.title.starts_with("3D"))
            {
                view.x_label = depth_axis_title(medium_key);
            }
            for depth in panels
                .iter_mut()
                .filter(|panel| panel.title.starts_with("Depth"))
            {
                if let Some(result) = &mc_note {
                    let incident = result.ledger[0];
                    let left = if incident > 0.0 {
                        100.0 * (result.ledger[2] + result.ledger[3]) / incident
                    } else {
                        0.0
                    };
                    depth.y_label = format!(
                        "{} · {:.1}% left {:.1}%",
                        depth.y_label,
                        result.uncertainty * 100.0,
                        left
                    );
                } else if model == "mc" {
                    depth.y_label = format!("{} · analytic fallback", depth.y_label);
                }
            }
            let at = [
                volume.mm_of(0, ix),
                volume.mm_of(1, iy),
                volume.mm_of(2, iz),
            ];
            if shown.len() > 1 {
                let gamma_volumes: Vec<Volume> = gamma_fields
                    .iter()
                    .map(|(volume, _)| volume.clone())
                    .collect();
                for panel in &mut panels {
                    crate::workspace::overlay_sessions(
                        panel,
                        &shown,
                        at,
                        &gamma_volumes,
                        &measured,
                        y_name(quantity),
                    );
                }
                mark.session_focus = session_at as u32;
                mark.sessions = shown
                    .iter()
                    .enumerate()
                    .map(|(index, volume)| session_dose(volume, index == session_at))
                    .collect();
            }
            if compare == "measured" {
                mark.plans = plans
                    .iter()
                    .map(|volume| session_dose(volume, false))
                    .collect();
                mark.companions = companions
                    .iter()
                    .map(|volume| session_dose(volume, false))
                    .collect();
                let extras: Vec<&Volume> = plans
                    .iter()
                    .chain(&companions)
                    .filter(|volume| {
                        volume
                            .values
                            .iter()
                            .any(|value| value.is_finite() && *value > 1.0e-8)
                    })
                    .collect();
                for panel in &mut panels {
                    crate::workspace::widen_profiles(panel, &extras, at);
                }
            }
            if compare == "gamma" {
                if let (Some(got), Some(plan)) = (measured.get(session_at), plans.get(session_at)) {
                    if plan.shape == got.shape
                        && plan.values.len() == got.values.len()
                        && plan.origin == got.origin
                        && plan.spacing == got.spacing
                    {
                        // Commanded positions are the plan against itself —
                        // planes keep their shape but score nothing, like the
                        // 3D gamma's all-NaN "not evaluated".
                        let planes = if clouds.get(session_at).is_some_and(|c| c.commanded) {
                            let [nx, ny, nz] = got.shape;
                            [
                                (vec![f32::NAN; nx * ny], f32::NAN),
                                (vec![f32::NAN; nx * nz], f32::NAN),
                                (vec![f32::NAN; ny * nz], f32::NAN),
                            ]
                        } else {
                            scan_kit_core::projected_gamma(got, plan, dd, dta, cutoff)
                        };
                        integral_rates = Some(planes.each_ref().map(|(_, rate)| *rate));
                        mark.integrals = planes.into_iter().map(|(gamma, _)| gamma).collect();
                    }
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
    controls.push(labeled("xy", "Chambers", CHAMBERS, chambers).grouped("Dose"));
    controls.push(labeled("quantity", "Quantity", QUANTITY, quantity_id).grouped("Dose"));
    controls.push(labeled("compare", "Compare", COMPARE, compare).grouped("Dose"));
    controls.push(labeled("plan_sigma", "Plan σ", SIGMA, plan_sigma).grouped("Dose"));
    if plan_sigma == "reference" && session_ids.len() > 1 {
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
    if has_dose {
        let measured_spots: usize = clouds.iter().map(|cloud| cloud.pencils.len()).sum();
        let other_spots: usize = clouds.iter().map(|cloud| cloud.companion.len()).sum();
        let plan_spots: usize = clouds.iter().map(|cloud| cloud.plan.len()).sum();
        let weights = match quantity {
            Quantity::Mu => "plan MU",
            Quantity::Protons => "protons",
            Quantity::Dose => "relative",
        };
        let spot_note = if other_spots > 0 {
            format!("{measured_spots} IC1 · {other_spots} IC2 · {plan_spots} plan · {weights}")
        } else if plan_spots == 0 {
            format!("{measured_spots} · no plan")
        } else {
            format!("{measured_spots} measured · {plan_spots} plan · {weights}")
        };
        controls.push(crate::workspace::readout(
            "Spots", "spots", &spot_note, "Dose",
        ));
    }
    if has_dose {
        let mut voxel = range_control(
            "voxel",
            "Voxel",
            0.25,
            10.0,
            0.1,
            &trim_number(voxel_spacing(options)),
        );
        voxel.kind = "number".into();
        for label in ["1", "2", "3"] {
            voxel
                .options
                .push(scan_kit_core::Choice::full("quick", label, "", ""));
        }
        controls.push(voxel.grouped("Calculation"));
        let mut grid = format!(
            "{} × {} × {}",
            volume_mark.shape[0], volume_mark.shape[1], volume_mark.shape[2]
        );
        if model == "mc" {
            grid.push_str(" · MC");
        }
        controls.push(crate::workspace::readout(
            "Grid",
            "grid",
            &grid,
            "Calculation",
        ));
    }
    controls.push(
        labeled("model", "Model", MODEL, model)
            .grouped("Calculation")
            .radio()
            .icons(&["analytic", "mc"]),
    );
    // Histories is the Monte Carlo count. It stays selectable in analytic mode
    // so the next transport uses the count the user already picked.
    controls.push(
        labeled("histories", "Histories", HISTORIES, &histories.to_string())
            .grouped("Calculation")
            .radio()
            .icons(&["1e6", "3e6", "1e7", "5e7"]),
    );
    controls.push(
        labeled("spread", "Energy spread", SPREAD, &trim_number(spread)).grouped("Calculation"),
    );
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
        let axes = volume_mark.axes();
        let phantom = format!(
            "{:.0} × {:.0} × {:.0} mm{sizing}",
            volume_mark.shape[0] as f32 * axes[0],
            volume_mark.shape[1] as f32 * axes[1],
            volume_mark.shape[2] as f32 * axes[2]
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
            compare,
            session_ids.len(),
            scale,
            wash.level.as_ref(),
        );
        if let Some([axial, coronal, sagittal]) = integral_rates {
            let rate_text = |rate: f32| {
                if rate.is_nan() {
                    "—".to_owned()
                } else {
                    format!("{rate:.0}%")
                }
            };
            controls.push(crate::workspace::readout(
                "Integral",
                "gamma_integral",
                &format!(
                    "Axial {} · Coronal {} · Sagittal {}",
                    rate_text(axial),
                    rate_text(coronal),
                    rate_text(sagittal)
                ),
                "Gamma",
            ));
        }
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

/// Spacing in mm. Accepts `1`, `1.5`, and `2 mm`. Outside 0.25–10 mm clamps, matching the Python spin box.
pub fn voxel_spacing(options: &Value) -> f32 {
    let raw = options.get("voxel").and_then(Value::as_str).unwrap_or("1");
    let text = raw.trim().trim_end_matches("mm").trim();
    // "2 cm"-style input: a leading number still parses — jumping to 1.0
    // would build a grid 20x finer than the user typed.
    let value = text
        .parse::<f32>()
        .or_else(|_| text.split_whitespace().next().unwrap_or("").parse())
        .unwrap_or(1.0);
    if !value.is_finite() {
        return 1.0;
    }
    value.clamp(0.25, 10.0)
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
        scan_kit_core::Choice::full("min", &trim_number(min), "", ""),
        scan_kit_core::Choice::full("max", &trim_number(max), "", ""),
        scan_kit_core::Choice::full("step", &trim_number(step), "", ""),
    ];
    control.kind = "range".into();
    control
}

fn session_at(options: &Value, ids: &[String]) -> Result<usize, String> {
    let raw = options.get("session").and_then(Value::as_str).unwrap_or("");
    if raw.is_empty() {
        return Ok(0);
    }
    // A miss must not silently focus session 0 — the MC runner and every
    // overlay would apply to the wrong session's data.
    ids.iter()
        .position(|id| id == raw)
        .ok_or_else(|| format!("session option {raw} is not a loaded session"))
}

fn push_picture_controls(
    controls: &mut Vec<scan_kit_core::Control>,
    options: &Value,
    compare: &str,
    sessions: usize,
    scale: &str,
    level: Option<&scan_kit_core::LevelSpan>,
) {
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
        if let Some(level) = level {
            controls.push(
                range_control(
                    "level",
                    level.label,
                    level.min,
                    level.max,
                    level.step,
                    &trim_number(level.value),
                )
                .grouped("Picture"),
            );
        }
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
        .icons(&["nearest", "linear", "cubic"])
        .grouped("Picture"),
    );
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

/// The displayed lattice. A heavy visit count used to keep only three planes;
/// that shortcut is a cross in the ray march, so the view always asks for every voxel.
fn lattice_planes(_frame: &scan_kit_core::DoseFrame) -> Option<[usize; 3]> {
    None
}

fn heaviest_visits(clouds: &[Cloud], combined: u64, visits: impl Fn(&[Pencil]) -> u64) -> u64 {
    let lists: Vec<&[Pencil]> = clouds
        .iter()
        .flat_map(|cloud| {
            [
                cloud.pencils.as_slice(),
                cloud.companion.as_slice(),
                cloud.plan.as_slice(),
            ]
        })
        .filter(|pencils| !pencils.is_empty())
        .collect();
    if lists.len() <= 1 {
        return combined;
    }
    lists.into_iter().map(visits).max().unwrap_or(combined)
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
    if pencils.is_empty() {
        return (
            Volume {
                origin: [0.0; 3],
                shape: [0, 0, 0],
                spacing: [1.0; 3],
                values: Vec::new(),
            },
            None,
        );
    }
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
                    // amount is always a MU/charge reading — transport
                    // wants real proton counts for every quantity.
                    protons: pencils
                        .iter()
                        .map(|p| {
                            protons_from_mu(
                                f64::from(p.amount),
                                f64::from(p.energy),
                                f64::from(gap_mm),
                                f64::from(k_mu),
                            ) as f32
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
    let mut key: Vec<u32> = pencils
        .iter()
        .flat_map(|p| [p.x, p.y, p.sx, p.sy, p.energy, p.amount])
        .chain([spread, wet_mm, phantom_mm, k_mu, gap_mm])
        .map(f32::to_bits)
        .collect();
    key.push(quantity as u32);
    if let Some((origin, shape, voxel)) = grid {
        key.extend(origin.iter().chain([&voxel]).map(|value| value.to_bits()));
        key.extend(shape.map(|n| n as u32));
    }
    if let Some(focus) = planes {
        key.extend(focus.map(|n| n as u32));
    }
    (
        analytic_cached(mat, key, || {
            analytic_on(
                mat, pencils, quantity, spread, wet_mm, phantom_mm, 1.0, k_mu, gap_mm, grid, planes,
            )
        }),
        None,
    )
}

/// A scene that only changes the compare, window, or cells, and every Monte
/// Carlo preview, reuses the plan and chamber deposits instead of painting
/// them again. `key` holds every input of the deposit.
// ponytail: a 256 MB budget. Several large 1 mm sessions still miss; raise it
// or key a disk cache if that matters.
fn analytic_cached(
    mat: scan_kit_core::Medium,
    key: Vec<u32>,
    deposit: impl FnOnce() -> Volume,
) -> Volume {
    // Three cubes always stay. Past that, keep cubes up to 64M voxels (256 MB).
    const KEEP: usize = 3;
    const BUDGET: usize = 64 << 20;
    type Kept = Vec<((&'static str, Vec<u32>), Volume)>;
    static LAST: std::sync::Mutex<Kept> = std::sync::Mutex::new(Vec::new());
    let key = (mat.key, key);
    let held = |kept: &mut Kept| {
        let at = kept.iter().position(|(seen, _)| *seen == key)?;
        let hit = kept.remove(at);
        let volume = hit.1.clone();
        kept.push(hit);
        Some(volume)
    };
    if let Some(volume) = held(&mut LAST.lock().unwrap_or_else(|err| err.into_inner())) {
        return volume;
    }
    let volume = deposit();
    let mut kept = LAST.lock().unwrap_or_else(|err| err.into_inner());
    // A cube bigger than the whole budget is searched again rather than
    // pinning the three-cube floor at GB scale.
    if held(&mut kept).is_none() && volume.values.len() <= BUDGET {
        kept.push((key, volume.clone()));
        let total = |kept: &Kept| {
            kept.iter()
                .map(|(_, cube)| cube.values.len())
                .sum::<usize>()
        };
        while kept.len() > KEEP && total(&kept) > BUDGET {
            kept.remove(0);
        }
    }
    volume
}

fn session_dose(volume: &Volume, blank: bool) -> scan_kit_core::SessionDose {
    // Difference cubes are signed — an all-nonpositive field is still
    // data (plan above measured everywhere), not a blank.
    let live = !blank
        && volume
            .values
            .iter()
            .any(|value| value.is_finite() && value.abs() > 1.0e-8);
    scan_kit_core::SessionDose {
        values: if live {
            volume.values.clone()
        } else {
            Vec::new()
        },
        shape: [
            volume.shape[0] as u32,
            volume.shape[1] as u32,
            volume.shape[2] as u32,
        ],
        origin: volume.origin,
        voxel: volume.spacing[0],
    }
}

fn load_cloud(
    root: &Path,
    session: &str,
    sessions: &[String],
    grain: &str,
    chambers: &str,
    plan_sigma: &str,
    sigma_ref: &str,
) -> Cloud {
    // A timeslice session has no spot positions — but the merged Sample
    // table can carry geometry columns unioned in by an earlier fill ask,
    // which would fabricate one unit-weight pencil per millisecond row.
    if grain == "timeslice" {
        return Cloud {
            pencils: Vec::new(),
            companion: Vec::new(),
            plan: Vec::new(),
            commanded: true,
        };
    }
    let table = if grain == "timeslice" {
        super::tables::timeslice_signals(root, session)
    } else {
        spot_table(root, session)
    };
    let energy = col(&table, "energy");
    let n = energy.len();
    let target = col(&table, "target_mu");
    let ic1_dose = col(&table, "ic1_dose");
    let ic2_dose = col(&table, "ic2_dose");
    let xml = devices_xml(root, session);
    let (ic1_x, ic1_y, ic1_commanded) = or_plan(&table, "ic1_x", "ic1_y");
    let ic1 = pencils_from(
        &energy,
        &ic1_x,
        &ic1_y,
        &col(&table, "ic1_sig_x"),
        &col(&table, "ic1_sig_y"),
        &ic1_dose,
        &target,
        &xml,
        n,
        false,
    );
    let ic2 = pencils_from(
        &energy,
        &col(&table, "ic2_x"),
        &col(&table, "ic2_y"),
        &col(&table, "ic2_sig_x"),
        &col(&table, "ic2_sig_y"),
        &ic2_dose,
        &target,
        &xml,
        n,
        false,
    );
    let (cx, cy) = iso_positions(&table);
    // iso_positions falls back to the real ic1 pair when any column is
    // missing — the flag must reflect what it used, not all four columns.
    let combined_commanded = {
        let check = |key: &str| -> bool { col(&table, key).iter().any(|v| v.is_finite()) };
        !(check("ic1_x") && check("ic1_y"))
    };
    let combined = pencils_from(
        &energy,
        &cx,
        &cy,
        &mean_column(
            &col(&table, "ic1_sig_x"),
            &col(&table, "ic2_sig_x"),
            usable_sigma,
        ),
        &mean_column(
            &col(&table, "ic1_sig_y"),
            &col(&table, "ic2_sig_y"),
            usable_sigma,
        ),
        &mean_column(&ic1_dose, &ic2_dose, usable_dose),
        &target,
        &xml,
        n,
        false,
    );
    let (donor, donor_is_self) = if plan_sigma == "reference" {
        let id = if sessions.iter().any(|item| item == sigma_ref) {
            sigma_ref
        } else {
            sessions.first().map(String::as_str).unwrap_or(session)
        };
        (spot_table(root, id), id == session)
    } else {
        (table.clone(), false)
    };
    let (psx, psy) = if plan_sigma == "interlock" {
        (Vec::new(), Vec::new())
    } else if plan_sigma == "reference" && !donor_is_self {
        // The donor is a different session — its spot list can order or
        // filter differently, so join sigmas on energy, not row index.
        join_donor_sigmas(
            &col(&donor, "energy"),
            &mean_column(
                &col(&donor, "ic1_sig_x"),
                &col(&donor, "ic2_sig_x"),
                usable_sigma,
            ),
            &mean_column(
                &col(&donor, "ic1_sig_y"),
                &col(&donor, "ic2_sig_y"),
                usable_sigma,
            ),
            &energy,
        )
    } else {
        (
            mean_column(
                &col(&donor, "ic1_sig_x"),
                &col(&donor, "ic2_sig_x"),
                usable_sigma,
            ),
            mean_column(
                &col(&donor, "ic1_sig_y"),
                &col(&donor, "ic2_sig_y"),
                usable_sigma,
            ),
        )
    };
    let plan = pencils_from(
        &energy,
        &col(&table, "plan_x"),
        &col(&table, "plan_y"),
        &psx,
        &psy,
        &ic1_dose,
        &target,
        &xml,
        n,
        plan_sigma == "interlock",
    );
    let (pencils, companion, commanded) = if chambers == "combined" {
        (combined, Vec::new(), combined_commanded)
    } else if ic1.is_empty() || (ic1_commanded && !ic2.is_empty()) {
        // A real second chamber beats a fabricated first one.
        (ic2, Vec::new(), false)
    } else {
        (ic1, ic2, ic1_commanded)
    };
    Cloud {
        pencils,
        companion,
        commanded,
        plan,
    }
}

fn usable_sigma(value: f32) -> bool {
    // Upper bound: the beam model cannot carry a metre-wide pencil, so a
    // corrupt-but-positive reading would only blow the lattice out to the
    // 512³ cap (~500 MB per fill).
    value.is_finite() && (0.2..=100.0).contains(&value)
}

fn usable_dose(value: f32) -> bool {
    value.is_finite() && value > 0.0
}

fn mean_column(left: &[f32], right: &[f32], keep: fn(f32) -> bool) -> Vec<f32> {
    let n = left.len().max(right.len());
    (0..n)
        .map(|i| {
            let a = left.get(i).copied().unwrap_or(f32::NAN);
            let b = right.get(i).copied().unwrap_or(f32::NAN);
            match (keep(a), keep(b)) {
                (true, true) => (a + b) * 0.5,
                (true, false) => a,
                (false, true) => b,
                (false, false) => f32::NAN,
            }
        })
        .collect()
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
            || !(1.0..=300.0).contains(&e)
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
            if mm.is_finite() && mm > 0.2 {
                return mm as f32;
            }
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

fn corrected(values: &[f32], plan: &[f32]) -> Vec<f32> {
    // The chamber zero offset is median(measured) - median(plan). Subtracting
    // the measured median alone would also erase a genuinely off-axis field,
    // fabricating a displacement between the measured and plan clouds.
    // With no plan reference there is no offset to find — keeping the raw
    // positions beats re-centering the cloud on a fabricated zero.
    let offset = if plan.iter().any(|v| v.is_finite()) {
        finite_median(values) - finite_median(plan)
    } else {
        0.0
    };
    values.iter().map(|value| value - offset).collect()
}

/// Plan-crossing isocenter, else the IC1 plane.
fn iso_plane_z(x2: &[f32], x1: &[f32], px: &[f32], y2: &[f32], y1: &[f32], py: &[f32]) -> f32 {
    let zx = axis_iso_z(x2, x1, px);
    let zy = axis_iso_z(y2, y1, py);
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
        // Bound the crossing to a plausible iso plane — a noise-dominated
        // chamber pair can intersect metres past the isocenter and drag the
        // median, and t = z/IC_SEP would scale the projected positions by
        // the same bogus factor.
        if z.is_finite() && z > IC2_Z_MM && z < IC1_Z_MM + 10.0 * IC_SEP_MM {
            zs.push(z);
        }
    }
    if zs.is_empty() {
        None
    } else {
        Some(finite_median(&zs))
    }
}

/// Measured spot positions projected to the isocenter plane on the IC2->IC1
/// ray, for delivered-dose matching and the combined chamber view.
pub(crate) fn iso_positions(
    table: &std::collections::BTreeMap<String, Vec<f32>>,
) -> (Vec<f32>, Vec<f32>) {
    let x2 = col(table, "ic2_x");
    let x1 = col(table, "ic1_x");
    let y2 = col(table, "ic2_y");
    let y1 = col(table, "ic1_y");
    // Present-but-all-NaN is as missing as absent — otherwise the combined
    // ray produces NaN positions and the view reports "No spots".
    if x1.is_empty()
        || x2.is_empty()
        || !x1.iter().any(|v| v.is_finite())
        || !x2.iter().any(|v| v.is_finite())
        || y1.is_empty()
        || y2.is_empty()
        || !y1.iter().any(|v| v.is_finite())
        || !y2.iter().any(|v| v.is_finite())
    {
        let (x, y, _) = or_plan(table, "ic1_x", "ic1_y");
        return (x, y);
    }
    let px = col(table, "plan_x");
    let py = col(table, "plan_y");
    let x2 = corrected(&x2, &px);
    let x1 = corrected(&x1, &px);
    let y2 = corrected(&y2, &py);
    let y1 = corrected(&y1, &py);
    // Solve the crossing on the corrected columns — the t that interpolates
    // them must come from the same frame.
    let z = iso_plane_z(&x2, &x1, &px, &y2, &y1, &py);
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
    (x, y)
}

fn join_donor_sigmas(
    donor_energy: &[f32],
    donor_x: &[f32],
    donor_y: &[f32],
    energy: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let mut psx = Vec::with_capacity(energy.len());
    let mut psy = Vec::with_capacity(energy.len());
    for e in energy {
        let at = donor_energy
            .iter()
            .position(|other| (other - e).abs() < 0.01);
        psx.push(
            at.and_then(|at| donor_x.get(at).copied())
                .unwrap_or(f32::NAN),
        );
        psy.push(
            at.and_then(|at| donor_y.get(at).copied())
                .unwrap_or(f32::NAN),
        );
    }
    (psx, psy)
}

fn or_plan(
    table: &std::collections::BTreeMap<String, Vec<f32>>,
    key_x: &str,
    key_y: &str,
) -> (Vec<f32>, Vec<f32>, bool) {
    let x = col(table, key_x);
    let y = col(table, key_y);
    // Fall back as a pair — measured x with planned y fabricates a position
    // that exists in neither record.
    if x.iter().any(|v| v.is_finite()) && y.iter().any(|v| v.is_finite()) {
        (x, y, false)
    } else {
        (col(table, "plan_x"), col(table, "plan_y"), true)
    }
}

fn col(table: &std::collections::BTreeMap<String, Vec<f32>>, key: &str) -> Vec<f32> {
    table.get(key).cloned().unwrap_or_default()
}

fn devices_xml(root: &Path, session: &str) -> String {
    let dir = discover::session_directory(root, session);
    crate::tables::capped_text(&dir.join("config/map2map/devices.xml")).unwrap_or_default()
}

fn read_kmu(root: &Path, session: &str) -> f32 {
    let xml = devices_xml(root, session);
    let mut hcc = None;
    let mut any = None;
    for chunk in xml.split("<ion_chamber").skip(1) {
        // The chamber name sits on a <device name="..."> child in devices.xml,
        // not on the <ion_chamber> tag itself.
        let name = chunk
            .split("<device")
            .skip(1)
            .filter_map(|tail| tail.split('>').next())
            .find_map(|tag| attr(tag, "name"))
            .unwrap_or_default();
        // K_MU lives on the gain_conversion with in_units="MU"
        // (core::tune::mu_gain) — prefer that tag so a stray K_MU
        // attribute elsewhere in the block cannot win; older files carry
        // k_mu on the device tag itself, so keep the block-wide search as
        // the fallback.
        let k = chunk
            .split("<gain_conversion")
            .skip(1)
            .filter_map(|tail| tail.split('>').next())
            .filter(|tag| {
                attr(tag, "in_units").is_some_and(|units| units.eq_ignore_ascii_case("MU"))
            })
            .find_map(|tag| attr(tag, "K_MU").or_else(|| attr(tag, "k_mu")))
            .or_else(|| attr(chunk, "K_MU").or_else(|| attr(chunk, "k_mu")));
        if let Some(Ok(value)) = k.map(|raw| raw.parse::<f32>()) {
            if value.is_finite() && value > 0.0 {
                if name.eq_ignore_ascii_case("IC_1_HCC") {
                    hcc = Some(value);
                }
                any.get_or_insert(value);
            }
        }
    }
    hcc.or(any).unwrap_or(FALLBACK_KMU)
}

fn attr<'a>(tag: &'a str, name: &str) -> Option<&'a str> {
    let key = format!("{name}=");
    let mut offset = 0;
    while let Some(pos) = tag[offset..].find(&key) {
        let at = offset + pos;
        // A whole attribute boundary — `name=` must not match the tail of
        // `long_name=` or `machine-name=`.
        if at == 0 || tag.as_bytes()[at - 1].is_ascii_whitespace() {
            let rest = tag[at + key.len()..].trim_start();
            // Single-quoted values parse too (tables::xml_attr).
            if let Some(&quote) = rest.as_bytes().first() {
                if quote == b'"' || quote == b'\'' {
                    let body = &rest[1..];
                    if let Some(end) = body.find(quote as char) {
                        return Some(&body[..end]);
                    }
                }
            }
        }
        offset = at + 1;
    }
    None
}

fn difference(measured: &Volume, plan: &Volume) -> Volume {
    // Same geometry guard gamma_with enforces — a plan on a shifted or
    // rescaled lattice with matching dims would subtract the wrong voxels
    // and still paint a plausible difference field.
    if measured.shape != plan.shape
        || measured.values.len() != plan.values.len()
        || measured.origin != plan.origin
        || measured.spacing != plan.spacing
    {
        // No plan to subtract — a flat field, not the measured cube drawn
        // under the divergent wash as if it were a difference.
        return Volume {
            values: vec![0.0; measured.values.len()],
            ..measured.clone()
        };
    }
    let mut values = measured.values.clone();
    for (slot, plan) in values.iter_mut().zip(&plan.values) {
        *slot -= *plan;
    }
    Volume {
        values,
        origin: measured.origin,
        shape: measured.shape,
        spacing: measured.spacing,
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
    // A session without a plan has an empty cube. Its gamma stays unscored on
    // the measured lattice, so the image keeps the shape the cells index.
    // Shape alone is not enough — a plan on a shifted or rescaled lattice
    // with the same dims would score voxel-for-voxel against the wrong
    // geometry and still return a pass rate.
    if plan.values.len() != measured.values.len()
        || plan.shape != measured.shape
        || plan.origin != measured.origin
        || plan.spacing != measured.spacing
    {
        return (vec![f32::NAN; measured.values.len()], 0, 0);
    }
    // A compare, plot, or cell change rebuilds the scene on the same cubes.
    // Matching the cubes exactly is one pass over each, far below a search.
    // ponytail: two pairs stay resident, about three cubes each. More loaded
    // sessions search again on every rebuild.
    const KEEP: usize = 2;
    type Kept = Vec<([f32; 3], Volume, Volume, (Vec<f32>, u32, u32))>;
    static LAST: std::sync::Mutex<Kept> = std::sync::Mutex::new(Vec::new());
    let criteria = [dose_percent, distance_mm, cutoff_pct];
    let same = |a: &Volume, b: &Volume| {
        a.shape == b.shape
            && a.origin == b.origin
            && a.spacing == b.spacing
            && a.values.len() == b.values.len()
            && a.values
                .iter()
                .zip(&b.values)
                .all(|(x, y)| x.to_bits() == y.to_bits())
    };
    {
        let mut kept = LAST.lock().unwrap_or_else(|err| err.into_inner());
        if let Some(at) = kept
            .iter()
            .position(|(c, m, p, _)| *c == criteria && same(m, measured) && same(p, plan))
        {
            let hit = kept.remove(at);
            let result = hit.3.clone();
            kept.push(hit);
            return result;
        }
    }
    let result = gamma_index(
        &measured.values,
        &plan.values,
        measured.shape,
        dose_percent,
        distance_mm,
        measured.spacing,
        cutoff_pct,
    );
    // A blank Monte Carlo capture scores nothing and is instant; keeping it
    // would push out a real pair. A cube over the per-entry bound (~128 MB
    // for the two kept pairs) is searched again rather than pinned.
    if result.2 > 0 && measured.values.len() <= (64 << 20) / 6 {
        let mut kept = LAST.lock().unwrap_or_else(|err| err.into_inner());
        if kept.len() >= KEEP {
            kept.remove(0);
        }
        kept.push((criteria, measured.clone(), plan.clone(), result.clone()));
    }
    result
}

fn gamma_image(template: &Volume, values: &[f32]) -> Volume {
    Volume {
        origin: template.origin,
        shape: template.shape,
        spacing: template.spacing,
        values: values.to_vec(),
    }
}

fn window(volumes: &[Volume], compare: &str) -> (f32, f32) {
    if compare == "gamma" {
        return (0.0, 2.0);
    }
    // One session is the usual voxel change. `robust_high` already keeps the
    // positive samples, so copying them into `flat` first walks the cube twice.
    if volumes.len() == 1 && compare != "difference" {
        return (0.0, robust_high(&volumes[0].values));
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

fn depth_axis_title(medium_key: &str) -> String {
    MEDIA
        .iter()
        .find(|(id, _)| *id == medium_key)
        .map(|(_, label)| format!("Depth in {label} (mm)"))
        .unwrap_or_else(|| "Depth (mm)".into())
}

fn volume_unit(quantity: Quantity, mode: u8, compare: &str) -> &'static str {
    if compare == "gamma" {
        return "gamma index";
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

#[cfg(test)]
mod tests {
    use scan_kit_core::{dose_frame, water, McResult, Pencil, Quantity, Volume};

    use super::{difference, iso_positions, lattice_planes, next_preview_voxel, read_kmu};

    #[test]
    fn an_off_axis_field_keeps_its_offset() {
        // A delivered field centered at +30 mm with a perfect delivery:
        // only the chamber zero offset is cancelled — the field offset is
        // real and must survive, or measured and plan clouds diverge.
        let mut table = std::collections::BTreeMap::new();
        table.insert("plan_x".to_owned(), vec![28.0, 30.0, 32.0]);
        table.insert("plan_y".to_owned(), vec![0.0, 0.0, 0.0]);
        // ic readings = plan + per-chamber zero offset.
        table.insert("ic1_x".to_owned(), vec![29.0, 31.0, 33.0]);
        table.insert("ic2_x".to_owned(), vec![28.5, 30.5, 32.5]);
        table.insert("ic1_y".to_owned(), vec![2.0, 2.0, 2.0]);
        table.insert("ic2_y".to_owned(), vec![2.0, 2.0, 2.0]);
        let (x, y) = iso_positions(&table);
        for (got, want) in x.iter().zip([28.0, 30.0, 32.0]) {
            assert!((got - want).abs() < 0.6, "x={got} want {want}");
        }
        assert!(y.iter().all(|v| v.abs() < 0.6));
    }

    #[test]
    fn the_hcc_chamber_kmu_wins_over_the_first_listed() {
        // The chamber name lives on a <device> child element — a plain
        // ion_chamber tag scan can never prefer IC_1_HCC.
        let root = std::env::temp_dir().join(format!(
            "scan-kit-kmu-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        let dir = root.join("s1").join("config").join("map2map");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("devices.xml"),
            "<ion_chamber><device name=\"IC_1_X\" k_mu=\"1.5e-8\"/></ion_chamber>\
             <ion_chamber><device name=\"IC_1_HCC\" k_mu=\"2.5e-8\"/></ion_chamber>",
        )
        .unwrap();
        assert_eq!(read_kmu(&root, "s1"), 2.5e-8);
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn an_all_nan_chamber_falls_back_like_a_missing_one() {
        let mut table = std::collections::BTreeMap::new();
        table.insert("ic1_x".to_owned(), vec![f32::NAN, f32::NAN]);
        table.insert("ic2_x".to_owned(), vec![1.0, 2.0]);
        table.insert("ic1_y".to_owned(), vec![f32::NAN, f32::NAN]);
        table.insert("ic2_y".to_owned(), vec![0.5, 0.5]);
        table.insert("plan_x".to_owned(), vec![9.0, 9.0]);
        table.insert("plan_y".to_owned(), vec![7.0, 7.0]);
        let (x, y) = iso_positions(&table);
        // Combined mode used to propagate the NaN into every position and
        // empty the cloud — the plan fallback must keep the session.
        assert_eq!(x, vec![9.0, 9.0]);
        assert_eq!(y, vec![7.0, 7.0]);
    }

    #[test]
    fn missing_ic_y_columns_fall_back_to_the_plan_pair() {
        // ic x columns alone used to pass the guard — the combined ray
        // then produced no y at all and every pencil dropped as "No
        // spots". A missing axis falls back pairwise like or_plan does.
        let mut table = std::collections::BTreeMap::new();
        table.insert("ic1_x".to_owned(), vec![1.0]);
        table.insert("ic2_x".to_owned(), vec![2.0]);
        table.insert("plan_x".to_owned(), vec![9.0]);
        table.insert("plan_y".to_owned(), vec![7.0]);
        let (x, y) = iso_positions(&table);
        assert_eq!(x, vec![9.0]);
        assert_eq!(y, vec![7.0]);
    }

    #[test]
    fn a_partial_mc_run_rolls_the_whole_session_back_to_analytic() {
        // One cube falling back used to leave MC measured vs analytic plan —
        // the gamma comparison then mixed engines. A partial run must be
        // reported as a fallback, not a clean MC result.
        let root = std::env::temp_dir().join(format!(
            "scan-kit-mixed-mc-{}-{}",
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
            "energy,charge_req,position_x,position_y
180,0.05,0,0
",
        )
        .unwrap();
        std::fs::write(
            session.join("spot_data.csv"),
            "ic1_total_dose
0.05
",
        )
        .unwrap();
        let calls = std::cell::Cell::new(0u32);
        let scene = super::volumetric(
            &root,
            &["sess".into()],
            &serde_json::json!({ "model": "mc" }),
            Some(&|job: &scan_kit_core::McJob| {
                calls.set(calls.get() + 1);
                if calls.get() > 1 {
                    return Err("gpu wedged".to_owned());
                }
                let scan_kit_core::McJob::Slab(request) = job else {
                    return Err("expected a slab job".to_owned());
                };
                Ok(McResult {
                    volume: Volume {
                        origin: request.origin,
                        shape: request.shape,
                        spacing: [request.voxel_mm; 3],
                        values: vec![1.0; request.shape.iter().product()],
                    },
                    uncertainty: 0.01,
                    histories: 0,
                    ledger: [0.0; 6],
                    let_d: Vec::new(),
                })
            }),
        );
        let _ = std::fs::remove_dir_all(&root);
        assert!(calls.get() >= 2, "measured and plan both asked: {calls:?}");
        let depth = scene
            .panels
            .iter()
            .find(|panel| panel.title.starts_with("Depth"))
            .expect("a depth panel");
        assert!(
            depth.y_label.contains("analytic fallback"),
            "partial MC must not claim a clean run: {}",
            depth.y_label
        );
    }

    #[test]
    fn donor_sigmas_follow_the_energy_not_the_row() {
        // A donor whose layers sort differently must not hand its sigma to
        // the wrong spot.
        let (x, y) = super::join_donor_sigmas(
            &[180.0, 70.0],
            &[3.0, 1.0],
            &[4.0, 2.0],
            &[70.0, 180.0, 120.0],
        );
        assert_eq!(x[..2], [1.0, 3.0]);
        assert_eq!(y[..2], [2.0, 4.0]);
        assert!(x[2].is_nan() && y[2].is_nan(), "no donor layer at 120");
    }

    #[test]
    fn the_plan_fallback_keeps_a_coherent_position_pair() {
        // A measured x with a planned y fabricates a position that exists in
        // neither record — the axes must fall back together.
        let mut table = std::collections::BTreeMap::new();
        table.insert("ic1_x".to_owned(), vec![f32::NAN]);
        table.insert("ic1_y".to_owned(), vec![4.0]);
        table.insert("ic2_x".to_owned(), vec![f32::NAN]);
        table.insert("ic2_y".to_owned(), vec![f32::NAN]);
        table.insert("plan_x".to_owned(), vec![9.0]);
        table.insert("plan_y".to_owned(), vec![7.0]);
        let (x, y) = iso_positions(&table);
        assert_eq!(x, vec![9.0]);
        assert_eq!(y, vec![7.0]);
    }

    #[test]
    fn a_missing_plan_difference_is_a_flat_field_not_the_measurement() {
        let measured = Volume {
            origin: [0.0; 3],
            shape: [2, 1, 1],
            spacing: [1.0; 3],
            values: vec![3.0, 4.0],
        };
        let plan = Volume {
            origin: [0.0; 3],
            shape: [0, 0, 0],
            spacing: [1.0; 3],
            values: Vec::new(),
        };
        let diff = difference(&measured, &plan);
        assert!(diff.values.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn a_shifted_plan_lattice_is_a_flat_difference() {
        // Same shape/len but a different origin — gamma_with refuses the
        // same pair; the difference must not paint a plausible-but-wrong
        // subtracted field either.
        let measured = Volume {
            origin: [0.0; 3],
            shape: [2, 1, 1],
            spacing: [1.0; 3],
            values: vec![3.0, 4.0],
        };
        let mut plan = measured.clone();
        plan.origin[0] = 4.0;
        let diff = difference(&measured, &plan);
        assert!(diff.values.iter().all(|v| *v == 0.0));
    }

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
    fn a_repeated_deposit_matches_and_a_changed_input_repaints() {
        let pencils = [Pencil {
            x: 0.0,
            y: 0.0,
            sx: 4.0,
            sy: 4.0,
            energy: 120.0,
            amount: 0.05,
        }];
        let fill = |pencils: &[Pencil], spread: f32| {
            super::fill(
                water(),
                pencils,
                Quantity::Dose,
                spread,
                0.0,
                0.0,
                2.0e-8,
                10.0,
                "analytic",
                0,
                "water",
                None,
                None,
                None,
            )
            .0
        };
        let fill = |pencils: &[Pencil], spread: f32| {
            let volume = fill(pencils, spread);
            (volume.origin, volume.values)
        };
        let first = fill(&pencils, 1.0);
        assert_eq!(fill(&pencils, 1.0), first);
        assert_ne!(fill(&pencils, 2.0), first);
        let moved = [Pencil {
            x: 1.0,
            ..pencils[0]
        }];
        assert_ne!(fill(&moved, 1.0), first);
        assert_eq!(fill(&pencils, 1.0), first);
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
    fn a_kept_gamma_follows_its_cubes() {
        let cube = |value: f32| Volume {
            origin: [0.0; 3],
            shape: [4, 4, 4],
            spacing: [2.0; 3],
            values: vec![value; 64],
        };
        let (measured, near, far) = (cube(1.0), cube(1.01), cube(1.04));
        let first = super::gamma_with(&measured, &near, 3.0, 2.0, 10.0);
        assert_eq!(super::gamma_with(&measured, &near, 3.0, 2.0, 10.0), first);
        let changed = super::gamma_with(&measured, &far, 3.0, 2.0, 10.0);
        assert!(changed.1 < first.1, "a new plan reused the old gamma");
        let looser = super::gamma_with(&measured, &far, 5.0, 2.0, 10.0);
        assert_ne!(looser.0, changed.0, "new criteria reused the old gamma");
    }

    #[test]
    fn a_shifted_plan_lattice_is_unscored() {
        let measured = Volume {
            origin: [0.0; 3],
            shape: [2, 2, 2],
            spacing: [2.0; 3],
            values: vec![1.0; 8],
        };
        // Same dims but a moved lattice — voxel-for-voxel scoring would
        // compare the wrong geometry and still report a pass rate.
        let mut plan = measured.clone();
        plan.origin[0] = 4.0;
        let (gamma, passed, scored) = super::gamma_with(&measured, &plan, 3.0, 2.0, 10.0);
        assert_eq!((passed, scored), (0, 0));
        assert!(gamma.iter().all(|v| v.is_nan()));
        let mut plan = measured.clone();
        plan.spacing[0] = 4.0;
        let (_, passed, scored) = super::gamma_with(&measured, &plan, 3.0, 2.0, 10.0);
        assert_eq!((passed, scored), (0, 0));
    }

    #[test]
    fn monte_carlo_transports_the_measured_spots_and_the_plan() {
        let root = std::env::temp_dir().join(format!(
            "scan-kit-mc-measured-{}-{}",
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
            "energy,charge_req,position_x,position_y\n150,0.05,40,0\n",
        )
        .unwrap();
        std::fs::write(
            session.join("spot_data.csv"),
            "ic1_total_dose,ic1_x_spot,ic1_y_spot\n0.05,0,0\n",
        )
        .unwrap();
        let xs = std::cell::RefCell::new(Vec::new());
        let scene = super::volumetric(
            &root,
            &["sess".into()],
            &serde_json::json!({ "model": "mc", "voxel": "4" }),
            Some(&|job| {
                let scan_kit_core::McJob::Slab(request) = job else {
                    return Err("patient job".into());
                };
                xs.borrow_mut().push(request.x.clone());
                let n = request.shape.iter().product();
                Ok(McResult {
                    volume: Volume {
                        origin: request.origin,
                        shape: request.shape,
                        spacing: [request.voxel_mm; 3],
                        values: vec![7.0; n],
                    },
                    uncertainty: 0.0,
                    histories: 0,
                    ledger: [0.0; 6],
                    let_d: Vec::new(),
                })
            }),
        );
        let _ = std::fs::remove_dir_all(&root);
        let xs = xs.borrow();
        assert_eq!(xs.len(), 2, "measured and plan each transport: {xs:?}");
        assert!(
            xs[0].iter().all(|x| x.abs() < 1.0),
            "first transport x {xs:?}, the measured spot sits at 0 mm"
        );
        assert!(
            xs[1].iter().all(|x| (x - 40.0).abs() < 1.0),
            "second transport x {xs:?}, the plan sits at 40 mm"
        );
        assert!(
            scene
                .volume
                .values
                .iter()
                .any(|value| (*value - 7.0).abs() < 1e-6),
            "the picture is not the measured transport"
        );
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
            spacing: [1.0; 3],
            values: vec![1.0, 0.0, 0.0, 0.0],
        };
        let plan = Volume {
            values: vec![1.0, 1.0, 1.0, 1.0],
            ..measured.clone()
        };
        let (_, passed, scored) = super::gamma_with(&measured, &plan, 3.0, 2.0, 10.0);
        assert_eq!((scored, passed), (1, 1));
        let no_plan = Volume {
            shape: [0, 0, 0],
            values: Vec::new(),
            ..measured.clone()
        };
        let (values, passed, scored) = super::gamma_with(&measured, &no_plan, 3.0, 2.0, 10.0);
        assert_eq!((scored, passed), (0, 0));
        assert_eq!(
            super::gamma_image(&measured, &values).peak_index(),
            (0, 0, 0)
        );
    }

    #[test]
    fn the_voxel_box_uses_the_requested_spacing() {
        let root = std::env::temp_dir().join(format!(
            "scan-kit-voxel-{}-{}",
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
        let cube = |voxel: &str| {
            super::volumetric(
                &root,
                &["sess".into()],
                &serde_json::json!({ "voxel": voxel }),
                None,
            )
        };
        let spacing = |voxel: &str| cube(voxel).volume.voxel;
        let cells = |voxel: &str| {
            let shape = cube(voxel).volume.shape;
            shape[0] * shape[1] * shape[2]
        };
        assert!((spacing("1") - 1.0).abs() < 1e-3);
        assert!((spacing("2 mm") - 2.0).abs() < 1e-3);
        assert!((spacing("1.5") - 1.5).abs() < 1e-3);
        assert!((spacing("0.1") - 0.25).abs() < 1e-3);
        assert!((spacing("40") - 10.0).abs() < 1e-3);
        assert!(cells("2 mm") < cells("1"));
        let scene = cube("");
        let control = scene
            .controls
            .iter()
            .find(|control| control.id == "voxel")
            .unwrap();
        assert_eq!(control.kind, "number");
        assert_eq!(control.group, "Calculation");
        assert_eq!(control.value, "1");
        let ids: Vec<&str> = scene
            .controls
            .iter()
            .map(|control| control.id.as_str())
            .collect();
        let voxel_at = ids.iter().position(|id| *id == "voxel").unwrap();
        let model_at = ids.iter().position(|id| *id == "model").unwrap();
        assert!(voxel_at < model_at);
        assert_eq!(
            scene
                .controls
                .iter()
                .find(|control| control.id == "grid")
                .map(|control| control.group.as_str()),
            Some("Calculation")
        );
        let hidden = super::volumetric(
            &root,
            &["sess".into()],
            &serde_json::json!({ "field": "Off" }),
            None,
        );
        assert_eq!(scene.volume.field, hidden.volume.field);
        assert!(scene.volume.field[1] > scene.volume.field[0]);
        assert_eq!(
            control
                .options
                .iter()
                .filter(|choice| choice.id == "quick")
                .map(|choice| choice.label.as_str())
                .collect::<Vec<_>>(),
            vec!["1", "2", "3"]
        );
        assert_eq!(
            scene
                .controls
                .iter()
                .find(|control| control.id == "sample")
                .map(|control| {
                    control
                        .options
                        .iter()
                        .map(|choice| choice.icon.as_str())
                        .collect::<Vec<_>>()
                }),
            Some(vec!["nearest", "linear", "cubic"])
        );
        let _ = std::fs::remove_dir_all(&root);
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
        assert!(scene.panels.iter().any(|panel| {
            panel.title.starts_with("3D") && panel.x_label == "Depth in Water (mm)"
        }));
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
            .all(|control| control.id != "spot_cap"));
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
        let [nx, ny, nz] = gamma.volume.shape.map(|n| n as usize);
        let sizes: Vec<usize> = gamma.volume.integrals.iter().map(Vec::len).collect();
        assert_eq!(
            sizes,
            [nx * ny, nx * nz, ny * nz],
            "integral views carry 2D gamma planes"
        );
        assert!(gamma
            .controls
            .iter()
            .any(|control| control.id == "gamma_integral" && control.value.contains("Axial")));
        assert!(gamma
            .controls
            .iter()
            .all(|control| control.id != "scale" && control.id != "level" && control.id != "auto"));
        assert_eq!(
            scene
                .controls
                .iter()
                .find(|control| control.id == "model")
                .map(|control| {
                    control
                        .options
                        .iter()
                        .map(|choice| choice.icon.as_str())
                        .collect::<Vec<_>>()
                }),
            Some(vec!["analytic", "mc"])
        );
        assert_eq!(
            scene
                .controls
                .iter()
                .find(|control| control.id == "model")
                .map(|control| {
                    (
                        control.kind.as_str(),
                        control
                            .options
                            .iter()
                            .map(|choice| choice.id.as_str())
                            .collect::<Vec<_>>(),
                    )
                }),
            Some(("radio", vec!["analytic", "mc"]))
        );
        assert!(scene.controls.iter().all(|control| control.id != "scatter"));
        assert_eq!(
            scene
                .controls
                .iter()
                .find(|control| control.id == "histories")
                .map(|control| {
                    (
                        control.kind.as_str(),
                        control.value.as_str(),
                        control
                            .options
                            .iter()
                            .map(|choice| (choice.id.as_str(), choice.icon.as_str()))
                            .collect::<Vec<_>>(),
                    )
                }),
            Some((
                "radio",
                "1e7",
                vec![
                    ("1000000", "1e6"),
                    ("3000000", "3e6"),
                    ("10000000", "1e7"),
                    ("50000000", "5e7")
                ]
            ))
        );
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
                .map(|control| {
                    control
                        .options
                        .iter()
                        .map(|choice| choice.icon.as_str())
                        .collect::<Vec<_>>()
                }),
            Some(vec!["1e6", "3e6", "1e7", "5e7"])
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

    #[test]
    fn saved_signal_maps_onto_chambers() {
        assert_eq!(super::chamber_mode(&serde_json::json!({})), "independent");
        assert_eq!(
            super::chamber_mode(&serde_json::json!({ "xy": "ic2" })),
            "independent"
        );
        assert_eq!(
            super::chamber_mode(&serde_json::json!({ "xy": "plan" })),
            "independent"
        );
        assert_eq!(
            super::chamber_mode(&serde_json::json!({ "xy": "iso_ray" })),
            "combined"
        );
        assert_eq!(
            super::chamber_mode(&serde_json::json!({ "xy": "Combined" })),
            "combined"
        );
    }

    #[test]
    fn combined_averages_dose_and_sigma() {
        let root = std::env::temp_dir().join(format!(
            "scan-kit-chambers-{}-{}",
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
            "energy,charge_req,position_x,position_y\n180,1,0,0\n180,1,0,0\n",
        )
        .unwrap();
        std::fs::write(
            session.join("spot_data.csv"),
            "ic1_total_dose,ic2_total_dose,ic1_x_spot_position,ic1_y_spot_position,ic2_x_spot_position,ic2_y_spot_position,ic1_x_spot_sigma,ic1_y_spot_sigma,ic2_x_spot_sigma,ic2_y_spot_sigma\n2,4,10,0,0,0,2,2,6,6\n6,10,40,0,20,0,4,4,8,8\n",
        )
        .unwrap();
        let ids = ["sess".to_string()];
        let independent =
            super::load_cloud(&root, "sess", &ids, "spot", "independent", "measured", "");
        let combined = super::load_cloud(&root, "sess", &ids, "spot", "combined", "measured", "");
        assert_eq!(independent.pencils.len(), 2);
        assert_eq!(independent.companion.len(), 2);
        assert_eq!(combined.pencils.len(), 2);
        assert!(combined.companion.is_empty());
        assert!((independent.pencils[0].amount - 2.0).abs() < 1e-3);
        assert!((independent.companion[0].amount - 4.0).abs() < 1e-3);
        assert!((combined.pencils[0].amount - 3.0).abs() < 1e-3);
        assert!((combined.pencils[1].amount - 8.0).abs() < 1e-3);
        let mean = (independent.pencils[0].sx + independent.companion[0].sx) * 0.5;
        assert!((combined.pencils[0].sx - mean).abs() < 1e-3);
        assert!((independent.plan[0].sx - mean).abs() < 1e-3);
        assert!((combined.pencils[0].x - independent.pencils[0].x).abs() > 0.5);
        let scene = super::volumetric(&root, &ids, &serde_json::json!({}), None);
        let chambers = scene
            .controls
            .iter()
            .find(|control| control.id == "xy")
            .unwrap();
        assert_eq!(chambers.label, "Chambers");
        assert_eq!(chambers.value, "Independent");
        assert_eq!(
            chambers
                .options
                .iter()
                .map(|choice| choice.label.as_str())
                .collect::<Vec<_>>(),
            ["Independent", "Combined"]
        );
        assert!(scene
            .controls
            .iter()
            .any(|control| control.id == "plan_sigma"));
        assert!(scene
            .volume
            .plans
            .iter()
            .any(|dose| !dose.values.is_empty()));
        assert!(scene
            .volume
            .companions
            .iter()
            .any(|dose| !dose.values.is_empty()));
        let ray = super::volumetric(&root, &ids, &serde_json::json!({ "xy": "iso_ray" }), None);
        assert_eq!(
            ray.controls
                .iter()
                .find(|control| control.id == "xy")
                .map(|control| control.value.as_str()),
            Some("Combined")
        );
        assert!(ray
            .volume
            .companions
            .iter()
            .all(|dose| dose.values.is_empty()));
        assert!(ray.volume.plans.iter().any(|dose| !dose.values.is_empty()));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn an_unknown_focus_session_is_an_error() {
        // A miss used to silently focus session 0 — the MC run and every
        // overlay would then target the wrong session's data.
        assert_eq!(super::session_at(&serde_json::json!({}), &[]).unwrap(), 0);
        let ids = vec!["a".to_string(), "b".to_string()];
        assert_eq!(
            super::session_at(&serde_json::json!({ "session": "b" }), &ids).unwrap(),
            1
        );
        assert!(super::session_at(&serde_json::json!({ "session": "zzz" }), &ids).is_err());
    }

    #[test]
    fn a_noise_iso_crossing_is_rejected() {
        // near/far equal within 1e-3 → slope 1e-5 → a 10 mm plan offset
        // crosses at z = 1e6 mm. Unbounded, that drags the iso median.
        assert!(super::axis_iso_z(&[10.0], &[10.001], &[20.0]).is_none());
        // A plausible crossing (z = 200 mm) is still accepted.
        assert_eq!(super::axis_iso_z(&[5.0], &[10.0], &[15.0]), Some(200.0));
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
