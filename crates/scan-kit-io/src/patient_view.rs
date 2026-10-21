//! Dose on a loaded CT. Planned and delivered spots share one Monte Carlo seed.

use std::path::Path;

use scan_kit_core::{
    dvh, gamma_index_norm, index, rbe, resample_linear, DataTable, McJob, Panel, PatientRequest,
    PlotScene, RbeModel, Volume, DEFAULT_ALPHA_BETA, RBE_CONSTANT,
};
use scan_kit_dicom::{inside_structure, load_study, scanner_density, scanner_label, PatientStudy};
use serde_json::Value;

use super::beam::{beam_record_bdl, protons_per_mu_bdl, spot_record_bdl};
use super::marks::pick;
use super::mc_tables::mc_tables;
use super::tables::spot_table;
use super::volumetric::McRunner;

const MARK: [f32; 4] = [0.0, 0.0, 0.0, 1.0];
const SEED: u32 = 1;

pub(crate) fn scene(
    root: &Path,
    session_ids: &[String],
    options: &Value,
    mc: Option<McRunner<'_>>,
) -> PlotScene {
    let path = options.get("study").and_then(Value::as_str).unwrap_or("");
    let study = match load_study(Path::new(path)) {
        Ok(study) => study,
        Err(err) => return note_scene(&err),
    };
    let histories = pick(
        options,
        "histories",
        "10000000",
        &[
            ("1000000", "1e6"),
            ("3000000", "3e6"),
            ("10000000", "1e7"),
            ("50000000", "5e7"),
        ],
    )
    .parse::<u32>()
    .unwrap_or(10_000_000);
    let deliveries = deliveries(root, session_ids, &study);
    let groups = group_fractions(
        &deliveries
            .iter()
            .map(|hit| hit.matched.beam)
            .collect::<Vec<_>>(),
    );
    let mut names = vec!["Plan".into()];
    for index in 1..=groups.len() {
        names.push(index.to_string());
    }
    let segment = pick(
        options,
        "segment",
        if groups.is_empty() {
            "plan"
        } else {
            "delivered"
        },
        &[("plan", "Plan"), ("delivered", "Delivered")],
    );
    let sent = options
        .get("fraction")
        .and_then(Value::as_str)
        .unwrap_or("Plan");
    let fraction = if segment == "plan" {
        "Plan"
    } else if names.iter().any(|name| name == sent) && sent != "Plan" {
        sent
    } else if groups.is_empty() {
        "Plan"
    } else {
        "1"
    };
    let beam_filter = options
        .get("beams")
        .and_then(Value::as_str)
        .and_then(|text| {
            text.parse::<usize>()
                .ok()
                .and_then(|index| index.checked_sub(1))
        });
    let bdl = pick(
        options,
        "bdl",
        "un_rs",
        &[
            ("un_rs", "UN + shifter"),
            ("un", "UN"),
            ("dn_rs", "DN + shifter"),
            ("dn", "DN"),
        ],
    );
    let scanner = pick(
        options,
        "scanner",
        "default",
        &[
            ("default", "Default"),
            ("water", "Water"),
            ("solid", "Solid water"),
        ],
    );
    const RBE_PAIRS: &[(&str, &str)] = &[
        ("1", "1.0"),
        ("1.1", "1.1"),
        ("mcnamara", "McNamara 2015"),
        ("wedenberg", "Wedenberg 2013"),
        ("carabe", "Carabe 2012"),
        ("unkelbach", "Unkelbach 2016"),
    ];
    const AB_PAIRS: &[(&str, &str)] = &[
        ("1.5", "1.5 Gy"),
        ("2", "2 Gy"),
        ("3", "3 Gy"),
        ("5", "5 Gy"),
        ("10", "10 Gy"),
    ];
    let rbe_key = pick(options, "rbe", "1.1", RBE_PAIRS);
    let rbe_model = if rbe_key == "1" || rbe_key == "1.0" {
        None
    } else {
        Some(RbeModel::parse(rbe_key))
    };
    let score_let = rbe_model.is_some_and(RbeModel::uses_let);
    let alpha_beta = pick(options, "alpha_beta", "2", AB_PAIRS)
        .parse::<f64>()
        .unwrap_or(DEFAULT_ALPHA_BETA);
    let (spots, protons, beams) = records(&study, fraction, &deliveries, &groups, beam_filter, bdl);
    let tables = mc_tables();
    let [nx, ny, nz] = study.ct.shape;
    let nvox = nx * ny * nz;
    let mut material = vec![0u8; nvox];
    let mut density = vec![0.0f32; nvox];
    for (index, hu) in study.ct.hu.iter().enumerate() {
        let label = scanner_label(scanner, *hu);
        material[index] = tables.ids.iter().position(|id| *id == label).unwrap_or(0) as u8;
        density[index] = scanner_density(scanner, *hu);
    }
    let request = PatientRequest {
        spots,
        protons,
        beams,
        material,
        density,
        spacing_mm: study.ct.spacing,
        origin_mm: study.ct.origin,
        shape: study.ct.shape,
        histories,
        seed: SEED,
        dose_to_water: true,
        score_let,
        let_water: score_let,
    };
    let ran = mc.and_then(|run| run(&McJob::Patient(request)).ok());
    let mut dose = ran
        .as_ref()
        .map(|result| result.volume.clone())
        .unwrap_or(Volume {
            origin: study.ct.origin,
            shape: study.ct.shape,
            voxel: study.ct.spacing[0],
            values: vec![0.0; nvox],
        });
    let let_d = ran
        .as_ref()
        .map(|result| result.let_d.as_slice())
        .filter(|letd| letd.len() == nvox);
    // The physical dose stays for dose-weighted LETd and RBE statistics.
    let physical = match rbe_model {
        Some(model) if model.uses_let() && let_d.is_some() => Some(dose.values.clone()),
        _ => None,
    };
    match rbe_model {
        None => {}
        Some(RbeModel::Constant) => {
            for value in &mut dose.values {
                *value *= RBE_CONSTANT as f32;
            }
        }
        Some(model) => {
            if let Some(letd) = let_d {
                apply_rbe(model, &mut dose.values, letd, alpha_beta);
            }
        }
    };
    // The RBE-weighted label only applies when the weighting actually ran.
    let effective = matches!(rbe_model, Some(RbeModel::Constant))
        || (rbe_model.is_some_and(RbeModel::uses_let) && let_d.is_some());
    let (cx, cy, cz) = dose.peak_index();
    let mut labels = vec![0u8; nvox];
    let mut dvh_lines = Vec::new();
    let mut rows = vec![
        vec!["Frame".into(), study.frame.clone()],
        vec!["Beams".into(), study.beams.len().to_string()],
    ];
    if let Some(result) = &ran {
        let incident = result.ledger[0];
        let left = if incident > 0.0 {
            100.0 * (result.ledger[2] + result.ledger[3]) / incident
        } else {
            0.0
        };
        rows.push(vec![
            "Uncertainty".into(),
            format!("{:.1}%", result.uncertainty * 100.0),
        ]);
        rows.push(vec!["Left grid".into(), format!("{left:.1}%")]);
    }
    if fraction != "Plan" {
        let chosen = fraction.parse::<usize>().unwrap_or(1).saturating_sub(1);
        for hit in groups.get(chosen).into_iter().flatten() {
            let delivery = &deliveries[*hit];
            let matched = &delivery.matched;
            let label = format!("Beam {} delivered", matched.beam + 1);
            rows.push(vec![
                label.clone(),
                format!("{:.1} / {:.1} MU", matched.delivered_mu, matched.planned_mu),
            ]);
            rows.push(vec![
                format!("{label} unmatched"),
                matched.unmatched.to_string(),
            ]);
            rows.push(vec![
                format!("{label} energy error"),
                format!("{:.2} MeV", matched.energy_error),
            ]);
            rows.push(vec![
                format!("{label} position RMS"),
                format!("{:.2} mm", matched.position_rms),
            ]);
            let ratios: Vec<f32> = matched
                .layer_mu_ratio
                .iter()
                .copied()
                .filter(|value| value.is_finite())
                .collect();
            if !ratios.is_empty() {
                let low = ratios.iter().copied().fold(f32::INFINITY, f32::min);
                let high = ratios.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                rows.push(vec![
                    format!("{label} MU ratio"),
                    format!("{low:.2} – {high:.2}"),
                ]);
            }
        }
    }
    let goals_text = goal_text(options, &study);
    let voxel_cc = study.ct.spacing.iter().product::<f32>() / 1000.0;
    for (id, structure) in study.structures.iter().enumerate() {
        if !structure_on(options, id, &structure.name) {
            continue;
        }
        let mark = (id + 1) as u8;
        let mut mask = vec![false; nvox];
        for index in 0..nvox {
            let x = index % nx;
            let y = (index / nx) % ny;
            let z = index / (nx * ny);
            if inside_structure(&study, structure, x, y, z) {
                mask[index] = true;
                if labels[index] == 0 {
                    labels[index] = mark;
                }
            }
        }
        if !mask.iter().any(|keep| *keep) {
            continue;
        }
        let (edges, curve) = dvh(&dose.values, &mask, 16);
        let xs = edges.iter().take(curve.len()).copied().collect();
        dvh_lines.push(scan_kit_core::Series::Polyline {
            xs,
            ys: curve.clone(),
            color: MARK,
            thickness: 1.5,
        });
        let structure_dvh = Dvh::of(&dose.values, &mask, voxel_cc);
        let rx = study.prescription.is_finite().then_some(study.prescription);
        for line in goals_text.lines() {
            let goal = match Goal::parse(line) {
                Ok(goal) => goal,
                Err(_) => continue,
            };
            if !structure.name.eq_ignore_ascii_case(&goal.roi) {
                continue;
            }
            let body = match goal.check(&structure_dvh, rx) {
                Ok((measured, pass)) => verdict(
                    pass,
                    format!("{} {measured:.2} {}", goal.metric_text, goal.unit),
                ),
                Err(err) => format!("fail {err}"),
            };
            rows.push(vec![structure.name.clone(), body]);
        }
        rows.push(vec![
            format!("{} mean", structure.name),
            format!("{:.2} Gy", structure_dvh.mean()),
        ]);
        rows.push(vec![
            format!("{} volume", structure.name),
            format!("{:.1} cc", structure_dvh.volume_cc()),
        ]);
        if let (Some(letd), Some(phys)) = (let_d, physical.as_ref()) {
            let mut dep = 0.0f64;
            let mut del = 0.0f64;
            let mut deff = 0.0f64;
            for index in 0..nvox {
                if mask[index] {
                    let d = f64::from(phys[index]);
                    dep += d;
                    del += d * f64::from(letd[index]);
                    deff += f64::from(dose.values[index]);
                }
            }
            if dep > 0.0 {
                rows.push(vec![
                    format!("{} LETd", structure.name),
                    format!("{:.2} keV/µm", del / dep),
                ]);
                rows.push(vec![
                    format!("{} RBE", structure.name),
                    format!("{:.3}", deff / dep),
                ]);
            }
        }
        rows.push(vec!["DVH".into(), structure.name.clone()]);
        for (edge, volume) in edges.iter().zip(&curve) {
            rows.push(vec!["curve".into(), format!("{edge:.4},{volume:.4}")]);
        }
    }
    if dvh_lines.is_empty() {
        dvh_lines.push(crate::workspace::dvh_line(&dose));
    }
    let gamma = ran.as_ref().and_then(|_| {
        let tps = study.tps.as_ref()?;
        let (dd, dta, cutoff) = gamma_criteria(options);
        // The TPS dose for one fraction, searched in the MC dose resampled to
        // its grid and expressed in the TPS dose's DoseType.
        let reference = tps_per_fraction(&tps.hu, &tps.summation, study.fractions);
        let norm = reference
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .fold(0.0f32, f32::max);
        let scale = (if tps.dose_type.eq_ignore_ascii_case("effective") {
            RBE_CONSTANT as f32
        } else {
            1.0
        }) / (if effective { RBE_CONSTANT as f32 } else { 1.0 });
        let same_grid = tps.shape == study.ct.shape
            && tps.spacing == study.ct.spacing
            && tps.origin == study.ct.origin;
        let evaluated: Vec<f32> = if same_grid {
            dose.values.iter().map(|value| value * scale).collect()
        } else {
            resample_linear(
                &dose.values,
                study.ct.origin,
                study.ct.spacing,
                study.ct.shape,
                tps.origin,
                tps.spacing,
                tps.shape,
            )
            .into_iter()
            .map(|value| value * scale)
            .collect()
        };
        let (values, passed, scored) = gamma_index_norm(
            &reference,
            &evaluated,
            tps.shape,
            dd,
            dta,
            tps.spacing,
            cutoff,
            norm,
        );
        let rate = if scored == 0 {
            0.0
        } else {
            100.0 * passed as f32 / scored as f32
        };
        rows.push(vec!["Gamma".into(), format!("{rate:.0}%")]);
        // The overlay draws on the dose grid; the score stays on the TPS grid.
        let display = if same_grid {
            values
        } else {
            resample_linear(
                &values,
                tps.origin,
                tps.spacing,
                tps.shape,
                study.ct.origin,
                study.ct.spacing,
                study.ct.shape,
            )
        };
        Some((
            Volume {
                values: display,
                ..dose.clone()
            },
            rate,
        ))
    });
    let cells = crate::workspace::cells_from(&[
        options.get("cell0").and_then(Value::as_str),
        options.get("cell1").and_then(Value::as_str),
        options.get("cell2").and_then(Value::as_str),
        options.get("cell3").and_then(Value::as_str),
    ]);
    let plots = crate::workspace::plots_from(
        [
            options.get("plot0").and_then(Value::as_str),
            options.get("plot1").and_then(Value::as_str),
        ],
        true,
    );
    let (panels, mut volume) = crate::workspace::assemble(crate::workspace::Workspace {
        dose,
        ct: study.ct.hu.clone(),
        labels,
        cursor: [cx, cy, cz],
        cells,
        plots,
        ramp: index("turbo"),
        lo: 0.0,
        hi: 0.0,
        gain: 1.0,
        opacity: 0.7,
        mode: 0,
        filter: 1,
        y_label: "Dose".into(),
        dvh: dvh_lines,
        gamma,
    });
    volume.unit = if effective {
        "Gy(RBE)".into()
    } else {
        "Gy".into()
    };
    let (columns, column_weights, row_weights, row_splits) = crate::workspace::layout();
    let mut controls = vec![
        scan_kit_core::Control::plain(
            "segment",
            "Dose",
            ["Plan", "Delivered"],
            if segment == "plan" {
                "Plan"
            } else {
                "Delivered"
            },
        )
        .grouped("Study"),
        scan_kit_core::Control::plain("fraction", "Fraction", &names, fraction).grouped("Study"),
        beam_control(&study, beam_filter),
        scan_kit_core::Control::plain(
            "rbe",
            "RBE",
            RBE_PAIRS.iter().map(|(_, label)| *label),
            RBE_PAIRS
                .iter()
                .find(|(key, _)| *key == rbe_key)
                .map(|(_, label)| *label)
                .unwrap_or("1.1"),
        )
        .grouped("Study"),
        scan_kit_core::Control::plain(
            "scanner",
            "Scanner",
            ["Default", "Water", "Solid water"],
            match scanner {
                "water" => "Water",
                "solid" => "Solid water",
                _ => "Default",
            },
        )
        .grouped("Study"),
        scan_kit_core::Control::plain(
            "bdl",
            "Beam model",
            ["UN + shifter", "UN", "DN + shifter", "DN"],
            match bdl {
                "un" => "UN",
                "dn" => "DN",
                "dn_rs" => "DN + shifter",
                _ => "UN + shifter",
            },
        )
        .grouped("Study"),
        history_control(histories),
        gamma_control(
            "dd",
            "Dose difference",
            "3",
            &[("2", "2%"), ("3", "3%"), ("5", "5%")],
            options,
        ),
        gamma_control(
            "dta",
            "Distance",
            "2",
            &[("2", "2 mm"), ("3", "3 mm")],
            options,
        ),
        gamma_control(
            "cutoff",
            "Low dose",
            "10",
            &[("5", "5%"), ("10", "10%"), ("20", "20%")],
            options,
        ),
    ];
    if rbe_model.is_some_and(RbeModel::uses_alpha_beta) {
        let at = controls.iter().position(|c| c.id == "rbe").unwrap_or(0) + 1;
        controls.insert(
            at,
            scan_kit_core::Control::plain(
                "alpha_beta",
                "(α/β)x",
                AB_PAIRS.iter().map(|(_, label)| *label),
                AB_PAIRS
                    .iter()
                    .find(|(key, _)| key.parse::<f64>() == Ok(alpha_beta))
                    .map(|(_, label)| *label)
                    .unwrap_or("2 Gy"),
            )
            .grouped("Study"),
        );
    }
    let goals_value = goal_text(options, &study);
    let mut goals =
        scan_kit_core::Control::plain("goals", "Goals", Vec::<&str>::new(), &goals_value);
    goals.kind = "text".into();
    controls.push(goals.grouped("Goals"));
    for (id, structure) in study.structures.iter().enumerate() {
        let on = structure_on(options, id, &structure.name);
        controls.push(
            scan_kit_core::Control::plain(
                format!("roi{id}"),
                &structure.name,
                ["On", "Off"],
                if on { "On" } else { "Off" },
            )
            .grouped("Structures")
            .checked(),
        );
    }
    for (index, cell) in cells.iter().enumerate() {
        controls.push(crate::workspace::cell_control(index, *cell));
    }
    controls.push(crate::workspace::plot_control(0, plots[0]));
    controls.push(crate::workspace::plot_control(1, plots[1]));
    PlotScene {
        title: "Volumetric".into(),
        panels,
        controls,
        table: Some(DataTable {
            columns: vec!["Item".into(), "Value".into()],
            rows,
        }),
        columns,
        column_weights,
        row_weights,
        side: 0,
        row_splits,
        volume,
    }
}

/// One measured spot snapped to its planned layer: the layer's nominal energy
/// and shifter settings, the measured isocenter-plane position, and the MU the
/// chambers counted.
struct MatchedSpot {
    energy: f32,
    x: f32,
    y: f32,
    mu: f32,
    wet: f32,
    distance: f32,
}

/// How one delivered beam strayed from its plan, in plan terms.
struct DeliveryMatch {
    beam: usize,
    spots: Vec<MatchedSpot>,
    planned_mu: f32,
    delivered_mu: f32,
    unmatched: usize,
    energy_error: f32,
    position_rms: f32,
    layer_mu_ratio: Vec<f32>,
}

struct Delivery {
    matched: DeliveryMatch,
}

const ENERGY_TOL: f32 = 0.5;

/// Plan spots at one nominal energy, in plan order. Control points that share
/// an energy (split or repainted layers) join one group; the measured spot
/// then finds its nearest planned spot inside it.
fn energy_layers(beam: &scan_kit_dicom::Beam) -> Vec<(f32, Vec<usize>)> {
    let mut layers: Vec<(f32, Vec<usize>)> = Vec::new();
    for (index, spot) in beam.spots.iter().enumerate() {
        match layers.iter_mut().find(|(energy, _)| *energy == spot.energy) {
            Some((_, spots)) => spots.push(index),
            None => layers.push((spot.energy, vec![index])),
        }
    }
    layers
}

fn deliveries(root: &Path, sessions: &[String], study: &PatientStudy) -> Vec<Delivery> {
    sessions
        .iter()
        .filter_map(|session| {
            let table = spot_table(root, session);
            let energy = table.get("energy").cloned().unwrap_or_default();
            if energy.is_empty() {
                return None;
            }
            // Measured positions on the IC2 -> IC1 ray at the isocenter plane.
            let (x, y) = super::volumetric::iso_positions(&table);
            let mu = table
                .get("ic1_dose")
                .cloned()
                .filter(|values| values.iter().any(|value| value.is_finite()))
                .or_else(|| table.get("ic2_dose").cloned())
                .unwrap_or_default();
            let beam = pick_beam(&study.beams, &energy, &mu)?;
            let matched = match_delivery(beam, &study.beams[beam], &energy, &x, &y, &mu);
            Some(Delivery { matched })
        })
        .collect()
}

/// The beam whose layers explain the most delivered spots (then the closest
/// total MU).
fn pick_beam(beams: &[scan_kit_dicom::Beam], energy: &[f32], mu: &[f32]) -> Option<usize> {
    let delivered_mu: f32 = mu.iter().filter(|value| value.is_finite()).copied().sum();
    let mut best: Option<(usize, usize, f32)> = None;
    for (index, beam) in beams.iter().enumerate() {
        let energies = energy_layers(beam);
        let hits = energy
            .iter()
            .filter(|value| {
                energies
                    .iter()
                    .any(|(layer, _)| (**value - layer).abs() <= ENERGY_TOL)
            })
            .count();
        let planned_mu: f32 = beam.spots.iter().map(|spot| spot.mu).sum();
        let mu_gap = (planned_mu - delivered_mu).abs();
        if hits > 0
            && best.is_none_or(|(_, count, gap)| hits > count || (hits == count && mu_gap < gap))
        {
            best = Some((index, hits, mu_gap));
        }
    }
    best.map(|(index, _, _)| index)
}

/// Snap each measured spot to the plan layer nearest in energy, within
/// tolerance. A matched spot transports with the layer's nominal energy and
/// the nearest planned spot's range shifter settings; a spot no layer's energy
/// reaches within `ENERGY_TOL`, or whose measured position or MU is missing,
/// is dropped and counted as unmatched.
fn match_delivery(
    beam_index: usize,
    beam: &scan_kit_dicom::Beam,
    energy: &[f32],
    x: &[f32],
    y: &[f32],
    mu: &[f32],
) -> DeliveryMatch {
    let layers = energy_layers(beam);
    let mut delivered = vec![0.0f32; layers.len()];
    let mut spots = Vec::new();
    let mut unmatched = 0usize;
    let mut energy_error = f32::NAN;
    let mut misses = Vec::new();
    for index in 0..energy.len() {
        let (mx, my, mm) = (
            x.get(index).copied().unwrap_or(f32::NAN),
            y.get(index).copied().unwrap_or(f32::NAN),
            mu.get(index).copied().unwrap_or(f32::NAN),
        );
        let nearest = layers
            .iter()
            .enumerate()
            .map(|(layer, (e, _))| (layer, (energy[index] - e).abs()))
            .min_by(|a, b| a.1.total_cmp(&b.1));
        let usable = mx.is_finite() && my.is_finite() && mm.is_finite() && mm > 0.0;
        match nearest {
            Some((layer, gap)) if usable && gap <= ENERGY_TOL => {
                let near = layers[layer]
                    .1
                    .iter()
                    .map(|&spot| {
                        let planned = &beam.spots[spot];
                        (spot, (planned.x - mx).hypot(planned.y - my))
                    })
                    .min_by(|a, b| a.1.total_cmp(&b.1))
                    .unwrap();
                misses.push(near.1);
                let planned = &beam.spots[near.0];
                spots.push(MatchedSpot {
                    energy: layers[layer].0,
                    x: mx,
                    y: my,
                    mu: mm,
                    wet: planned.wet_mm,
                    distance: planned.shifter_distance_mm,
                });
                delivered[layer] += mm;
                energy_error = if energy_error.is_nan() {
                    gap
                } else {
                    energy_error.max(gap)
                };
            }
            _ => unmatched += 1,
        }
    }
    let layer_mu_ratio = layers
        .iter()
        .enumerate()
        .map(|(layer, (_, indices))| {
            let planned: f32 = indices.iter().map(|&spot| beam.spots[spot].mu).sum();
            if delivered[layer] > 0.0 {
                delivered[layer] / planned
            } else {
                f32::NAN
            }
        })
        .collect();
    DeliveryMatch {
        beam: beam_index,
        planned_mu: beam.spots.iter().map(|spot| spot.mu).sum(),
        delivered_mu: spots.iter().map(|spot| spot.mu).sum(),
        spots,
        unmatched,
        energy_error,
        position_rms: if misses.is_empty() {
            f32::NAN
        } else {
            (misses.iter().map(|d| d * d).sum::<f32>() / misses.len() as f32).sqrt()
        },
        layer_mu_ratio,
    }
}

/// Beam deliveries in order. A beam that appears again starts the next fraction.
fn group_fractions(beams: &[usize]) -> Vec<Vec<usize>> {
    let mut out: Vec<Vec<usize>> = Vec::new();
    for (index, beam) in beams.iter().enumerate() {
        if out
            .last()
            .is_none_or(|group| group.iter().any(|&prior| beams[prior] == *beam))
        {
            out.push(Vec::new());
        }
        out.last_mut().unwrap().push(index);
    }
    out
}

fn records(
    study: &PatientStudy,
    fraction: &str,
    deliveries: &[Delivery],
    groups: &[Vec<usize>],
    beam_filter: Option<usize>,
    bdl: &str,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut spots = Vec::new();
    let mut protons = Vec::new();
    let mut beams = Vec::new();
    for (index, beam) in study.beams.iter().enumerate() {
        if let Ok(record) =
            beam_record_bdl(beam.gantry, beam.couch, &beam.position, beam.isocenter, bdl)
        {
            beams.extend_from_slice(&record);
        }
        if fraction == "Plan" {
            if beam_filter.is_some_and(|chosen| chosen != index) {
                continue;
            }
            for spot in &beam.spots {
                push_spot(
                    &mut spots,
                    &mut protons,
                    spot.energy,
                    spot.x,
                    spot.y,
                    spot.mu,
                    index,
                    spot.wet_mm,
                    spot.shifter_distance_mm,
                    bdl,
                );
            }
        }
    }
    if fraction != "Plan" {
        let chosen = fraction.parse::<usize>().unwrap_or(1).saturating_sub(1);
        for hit in groups.get(chosen).into_iter().flatten() {
            let delivery = &deliveries[*hit];
            if beam_filter.is_some_and(|chosen| chosen != delivery.matched.beam) {
                continue;
            }
            for spot in &delivery.matched.spots {
                push_spot(
                    &mut spots,
                    &mut protons,
                    spot.energy,
                    spot.x,
                    spot.y,
                    spot.mu,
                    delivery.matched.beam,
                    spot.wet,
                    spot.distance,
                    bdl,
                );
            }
        }
    }
    (spots, protons, beams)
}

fn push_spot(
    spots: &mut Vec<f32>,
    protons: &mut Vec<f32>,
    energy: f32,
    x: f32,
    y: f32,
    mu: f32,
    beam: usize,
    wet: f32,
    distance: f32,
    bdl: &str,
) {
    if let Ok(record) = spot_record_bdl(energy, x, y, beam as f32, wet, distance, bdl) {
        spots.extend_from_slice(&record);
        protons.push(mu * protons_per_mu_bdl(energy, bdl));
    }
}

fn beam_control(study: &PatientStudy, chosen: Option<usize>) -> scan_kit_core::Control {
    let mut labels = vec!["All".to_string()];
    for index in 1..=study.beams.len() {
        labels.push(index.to_string());
    }
    let value = chosen
        .map(|index| (index + 1).to_string())
        .filter(|text| labels.iter().any(|label| label == text))
        .unwrap_or_else(|| "All".into());
    let refs: Vec<&str> = labels.iter().map(String::as_str).collect();
    scan_kit_core::Control::plain("beams", "Beams", refs, value).grouped("Study")
}

fn history_control(histories: u32) -> scan_kit_core::Control {
    const PAIRS: &[(&str, &str)] = &[
        ("1000000", "1e6"),
        ("3000000", "3e6"),
        ("10000000", "1e7"),
        ("50000000", "5e7"),
    ];
    let id = histories.to_string();
    let value = PAIRS
        .iter()
        .find(|(key, _)| *key == id)
        .map(|(_, label)| *label)
        .unwrap_or("1e7");
    scan_kit_core::Control::plain(
        "histories",
        "Histories",
        PAIRS.iter().map(|(_, label)| *label),
        value,
    )
    .grouped("Calculation")
    .radio()
    .icons(&["1e6", "3e6", "1e7", "5e7"])
}

fn note_scene(message: &str) -> PlotScene {
    PlotScene {
        title: "Volumetric".into(),
        panels: vec![Panel {
            title: message.into(),
            y_label: String::new(),
            x_label: String::new(),
            xmin: 0.0,
            xmax: 1.0,
            ymin: 0.0,
            ymax: 1.0,
            series: Vec::new(),
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
    }
}

fn structure_on(options: &Value, id: usize, name: &str) -> bool {
    super::marks::flag(
        options,
        &format!("roi{id}"),
        !name.eq_ignore_ascii_case("EXTERNAL"),
    )
}

fn goal_text(options: &Value, study: &PatientStudy) -> String {
    let stored = options
        .get("goals")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim();
    if stored.is_empty() {
        default_goals(study)
    } else {
        stored.to_string()
    }
}

fn default_goals(study: &PatientStudy) -> String {
    let target = study
        .structures
        .iter()
        .find(|structure| structure.name.to_ascii_uppercase().contains("PTV"))
        .or_else(|| {
            study
                .structures
                .iter()
                .find(|structure| !structure.name.eq_ignore_ascii_case("EXTERNAL"))
        });
    match target {
        Some(structure) => format!(
            "{}: D95% >= 95%\n{}: D2% <= 107%",
            structure.name, structure.name
        ),
        None => String::new(),
    }
}

fn gamma_control(
    id: &str,
    label: &str,
    default: &'static str,
    pairs: &[(&'static str, &'static str)],
    options: &Value,
) -> scan_kit_core::Control {
    let shown = pick(options, id, default, pairs);
    let value = pairs
        .iter()
        .find(|(key, _)| *key == shown)
        .map(|(_, label)| *label)
        .unwrap_or(shown);
    let labels: Vec<&str> = pairs.iter().map(|(_, label)| *label).collect();
    scan_kit_core::Control::plain(id, label, labels, value).grouped("Gamma")
}

fn gamma_criteria(options: &Value) -> (f32, f32, f32) {
    (
        pick(options, "dd", "3", &[("2", "2%"), ("3", "3%"), ("5", "5%")])
            .parse()
            .unwrap_or(3.0),
        pick(options, "dta", "2", &[("2", "2 mm"), ("3", "3 mm")])
            .parse()
            .unwrap_or(2.0),
        pick(
            options,
            "cutoff",
            "10",
            &[("5", "5%"), ("10", "10%"), ("20", "20%")],
        )
        .parse()
        .unwrap_or(10.0),
    )
}

const DVH_BINS: usize = 4096;

/// Cumulative dose-volume histogram for one structure: `counts[i]` voxels sit
/// in `edges[i]..edges[i+1]` Gy, `voxel_cc` is one voxel's volume.
struct Dvh {
    edges: Vec<f32>,
    counts: Vec<f64>,
    voxel_cc: f32,
    dmax: f32,
    dmin: f32,
}

impl Dvh {
    /// Histogram of `dose` inside `mask`, bins up to the whole-volume peak.
    fn of(dose: &[f32], mask: &[bool], voxel_cc: f32) -> Dvh {
        let peak = dose.iter().copied().fold(0.0f32, f32::max);
        let hi = if peak > 0.0 { peak * (1.0 + 1e-6) } else { 1.0 };
        let edges: Vec<f32> = (0..=DVH_BINS)
            .map(|i| hi * i as f32 / DVH_BINS as f32)
            .collect();
        let mut counts = vec![0.0f64; DVH_BINS];
        let mut dmax = 0.0f32;
        let mut dmin = f32::NAN;
        for (value, keep) in dose.iter().zip(mask) {
            if !keep {
                continue;
            }
            let bin = ((*value / hi * DVH_BINS as f32) as usize).min(DVH_BINS - 1);
            counts[bin] += 1.0;
            dmax = dmax.max(*value);
            dmin = if dmin.is_nan() {
                *value
            } else {
                dmin.min(*value)
            };
        }
        Dvh {
            edges,
            counts,
            voxel_cc,
            dmax,
            dmin: if dmin.is_nan() { 0.0 } else { dmin },
        }
    }

    fn voxels(&self) -> f64 {
        self.counts.iter().sum()
    }

    fn volume_cc(&self) -> f32 {
        self.voxels() as f32 * self.voxel_cc
    }

    /// Volume fraction receiving at least `edges[i]`, one entry past the last.
    fn cumulative(&self) -> Vec<f32> {
        let n = (self.voxels() as f32).max(1.0);
        let mut out = vec![0.0f32; self.edges.len()];
        let mut acc = 0.0f64;
        for i in (0..self.edges.len() - 1).rev() {
            acc += self.counts[i];
            out[i] = (acc / f64::from(n)) as f32;
        }
        out
    }

    fn mean(&self) -> f32 {
        let n = (self.voxels() as f32).max(1.0);
        let sum: f64 = self
            .counts
            .iter()
            .enumerate()
            .map(|(i, count)| 0.5 * f64::from(self.edges[i + 1] + self.edges[i]) * count)
            .sum();
        (sum / f64::from(n)) as f32
    }

    /// D at a volume fraction (0.95 for D95%): the dose that much of the
    /// structure reaches.
    fn dose_at(&self, fraction: f32) -> f32 {
        if fraction <= 0.0 {
            return self.dmax;
        }
        if fraction >= 1.0 {
            return self.dmin;
        }
        dose_at_volume(&self.edges, &self.cumulative(), fraction)
    }

    fn dose_at_cc(&self, cc: f32) -> f32 {
        if self.volume_cc() > 0.0 {
            self.dose_at(cc / self.volume_cc())
        } else {
            0.0
        }
    }

    /// V at `dose` Gy, as a fraction of the structure.
    fn volume_at(&self, dose: f32) -> f32 {
        let curve = self.cumulative();
        let last = self.edges.len() - 1;
        if dose <= self.edges[0] {
            return curve[0];
        }
        if dose >= self.edges[last] {
            return curve[last];
        }
        let span = (self.edges[last] - self.edges[0]).max(1e-9);
        let pos = (dose - self.edges[0]) / span * last as f32;
        let i = (pos.floor() as usize).min(last - 1);
        let t = pos - i as f32;
        curve[i] + t * (curve[i + 1] - curve[i])
    }
}

/// One term of a clinical goal: `D95%`, `Dmax`, `D2cc`, `V20Gy`, `V50%`.
#[derive(Clone, Copy)]
enum GoalMetric {
    Mean,
    Max,
    Min,
    DoseAt(f32),
    DoseCc(f32),
    VolumeGy(f32),
    VolumePct(f32),
}

/// A clinical goal like `PTV: D95% >= 95%`, `Cord: Dmax < 45 Gy` or
/// `Lung: V20Gy < 30%`. Dose values in % are of the prescription; V goals in %
/// are of the structure's volume.
struct Goal {
    roi: String,
    metric: GoalMetric,
    metric_text: String,
    op: String,
    value: f32,
    unit: String,
}

impl Goal {
    fn parse(text: &str) -> Result<Goal, String> {
        let (roi, rest) = text
            .split_once(':')
            .ok_or_else(|| format!("not a goal: {text:?}"))?;
        let rest = rest.trim();
        let boundary = rest
            .find(['<', '>'])
            .ok_or_else(|| format!("not a goal: {text:?}"))?;
        let metric_text = rest[..boundary].trim();
        let metric = parse_metric(metric_text).ok_or_else(|| format!("not a goal: {text:?}"))?;
        let rest = rest[boundary..].trim_start();
        let (op, rest) = ["<=", ">=", "<", ">"]
            .iter()
            .find_map(|op| rest.strip_prefix(op).map(|rest| (*op, rest)))
            .ok_or_else(|| format!("not a goal: {text:?}"))?;
        let rest = rest.trim_start();
        let (value, tail) = take_number(rest);
        let mut unit = tail.trim().to_string();
        if unit.is_empty() {
            unit = if matches!(metric, GoalMetric::VolumeGy(_) | GoalMetric::VolumePct(_)) {
                "%".into()
            } else {
                "Gy".into()
            };
        }
        let unit = match unit.to_ascii_lowercase().as_str() {
            "gy" => "Gy",
            "%" => "%",
            "cc" => "cc",
            _ => return Err(format!("not a goal: {text:?}")),
        };
        let volume = matches!(metric, GoalMetric::VolumeGy(_) | GoalMetric::VolumePct(_));
        if (volume && unit == "Gy") || (!volume && unit == "cc") {
            return Err(format!(
                "{text:?}: a {} goal can't be in {unit}",
                if volume { "volume" } else { "dose" }
            ));
        }
        Ok(Goal {
            roi: roi.trim().to_string(),
            metric,
            metric_text: metric_text.to_string(),
            op: op.to_string(),
            value,
            unit: unit.to_string(),
        })
    }

    /// The metric in the goal's own unit.
    fn measure(&self, dvh: &Dvh, rx: Option<f32>) -> Result<f32, String> {
        let need_rx = || {
            rx.filter(|value| value.is_finite() && *value > 0.0)
                .ok_or_else(|| format!("{} needs the prescription dose", self.metric_text))
        };
        match self.metric {
            GoalMetric::VolumeGy(level) => {
                let frac = dvh.volume_at(level);
                Ok(if self.unit == "cc" {
                    frac * dvh.volume_cc()
                } else {
                    frac * 100.0
                })
            }
            GoalMetric::VolumePct(pct) => {
                let frac = dvh.volume_at(pct / 100.0 * need_rx()?);
                Ok(if self.unit == "cc" {
                    frac * dvh.volume_cc()
                } else {
                    frac * 100.0
                })
            }
            _ => {
                let gy = match self.metric {
                    GoalMetric::Mean => dvh.mean(),
                    GoalMetric::Max => dvh.dmax,
                    GoalMetric::Min => dvh.dmin,
                    GoalMetric::DoseCc(cc) => dvh.dose_at_cc(cc),
                    GoalMetric::DoseAt(fraction) => dvh.dose_at(fraction),
                    GoalMetric::VolumeGy(_) | GoalMetric::VolumePct(_) => unreachable!(),
                };
                Ok(if self.unit == "%" {
                    100.0 * gy / need_rx()?
                } else {
                    gy
                })
            }
        }
    }

    fn check(&self, dvh: &Dvh, rx: Option<f32>) -> Result<(f32, bool), String> {
        let measured = self.measure(dvh, rx)?;
        Ok((measured, compare(measured, &self.op, self.value)))
    }
}

fn parse_metric(text: &str) -> Option<GoalMetric> {
    let low = text.to_ascii_lowercase();
    match low.as_str() {
        "dmean" => return Some(GoalMetric::Mean),
        "dmax" => return Some(GoalMetric::Max),
        "dmin" => return Some(GoalMetric::Min),
        _ => {}
    }
    if low.is_empty() {
        return None;
    }
    let (kind, rest) = low.split_at(1);
    let end = rest
        .find(|c: char| !(c.is_ascii_digit() || c == '.'))
        .unwrap_or(rest.len());
    let (number, unit) = rest.split_at(end);
    let level: f32 = number.parse().ok()?;
    match (kind, unit) {
        ("d", "%") => Some(GoalMetric::DoseAt(level / 100.0)),
        ("d", "cc") => Some(GoalMetric::DoseCc(level)),
        ("v", "gy") => Some(GoalMetric::VolumeGy(level)),
        ("v", "%") => Some(GoalMetric::VolumePct(level)),
        _ => None,
    }
}

/// Summation types covering every fraction (PS3.3 C.8.8.3): the dose divides
/// by the fraction count. The *_SESSION types and CONTROL_POINT are per
/// fraction already.
const COURSE_SUMMATIONS: &[&str] = &[
    "PLAN",
    "MULTI_PLAN",
    "PLAN_OVERVIEW",
    "FRACTION",
    "BEAM",
    "BRACHY",
];

/// The TPS dose for one fraction.
fn tps_per_fraction(dose: &[f32], summation: &str, fractions: i32) -> Vec<f32> {
    let course = COURSE_SUMMATIONS
        .iter()
        .any(|kind| kind.eq_ignore_ascii_case(summation.trim()));
    let scale = if course { fractions.max(1) as f32 } else { 1.0 };
    dose.iter().map(|value| value / scale).collect()
}

/// Weight each voxel's physical dose by the variable-RBE model's LETd response.
fn apply_rbe(model: RbeModel, dose: &mut [f32], letd: &[f32], alpha_beta: f64) {
    for (value, &let_v) in dose.iter_mut().zip(letd.iter()) {
        *value = (f64::from(*value) * rbe(model, f64::from(*value), f64::from(let_v), alpha_beta))
            as f32;
    }
}

fn dose_at_volume(edges: &[f32], curve: &[f32], fraction: f32) -> f32 {
    if curve.is_empty() {
        return 0.0;
    }
    for index in 0..curve.len().saturating_sub(1) {
        if curve[index] >= fraction && curve[index + 1] <= fraction {
            let span = (curve[index] - curve[index + 1]).max(1e-6);
            let t = (curve[index] - fraction) / span;
            return edges[index] + t * (edges[index + 1] - edges[index]);
        }
    }
    if curve[0] < fraction {
        edges[0]
    } else {
        edges[curve.len().min(edges.len()).saturating_sub(1)]
    }
}

fn take_number(text: &str) -> (f32, &str) {
    let end = text
        .find(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-'))
        .unwrap_or(text.len());
    let (head, rest) = text.split_at(end);
    (head.parse().unwrap_or(0.0), rest)
}

fn compare(got: f32, op: &str, want: f32) -> bool {
    match op {
        ">=" => got + 1e-4 >= want,
        "<=" => got - 1e-4 <= want,
        ">" => got > want,
        "<" => got < want,
        _ => false,
    }
}

fn verdict(passed: bool, detail: String) -> String {
    format!("{} {detail}", if passed { "pass" } else { "fail" })
}

#[cfg(test)]
mod tests {
    use scan_kit_core::RbeModel;

    use super::{apply_rbe, group_fractions, tps_per_fraction, Dvh, Goal};

    #[test]
    fn a_repeated_beam_starts_another_fraction() {
        assert_eq!(group_fractions(&[0, 1]), vec![vec![0, 1]]);
        assert_eq!(group_fractions(&[0, 0]), vec![vec![0], vec![1]]);
        assert_eq!(group_fractions(&[0, 1, 0]), vec![vec![0, 1], vec![2]]);
        // Beams 0,1,0,1,2,1: the beam-0 repeat opens fraction two, the beam-1
        // repeat inside it opens a third holding only its own delivery.
        assert_eq!(
            group_fractions(&[0, 1, 0, 1, 2, 1]),
            vec![vec![0, 1], vec![2, 3, 4], vec![5]]
        );
    }

    #[test]
    fn a_flat_dose_meets_d95_and_fails_a_tight_maximum() {
        let dose = vec![2.0; 8];
        let mask = vec![true; 8];
        let dvh = Dvh::of(&dose, &mask, 0.001);
        let check = |text: &str, rx| {
            Goal::parse(text)
                .and_then(|goal| goal.check(&dvh, rx))
                .map(|(_, pass)| pass)
        };
        assert_eq!(check("PTV: D95% >= 95%", Some(2.0)), Ok(true));
        assert_eq!(check("Cord: Dmax < 1 Gy", None), Ok(false));
        assert_eq!(check("Lung: V20Gy < 30%", None), Ok(true));
        // V just under the flat dose is the whole structure; V at it bins to the edge.
        assert_eq!(check("PTV: V1.9Gy >= 99%", None), Ok(true));
        assert_eq!(check("PTV: V2Gy >= 99%", None), Ok(false));
        // Percent metrics read the prescription, cc metrics read the volume.
        assert_eq!(check("PTV: Dmean >= 99%", Some(2.0)), Ok(true));
        assert_eq!(check("PTV: Dmean >= 60 Gy", Some(2.0)), Ok(false));
        let (got, _) = Goal::parse("PTV: V1.9Gy >= 0.008 cc")
            .unwrap()
            .check(&dvh, None)
            .unwrap();
        assert!((got - 0.008).abs() < 1e-6, "{got}");
        // A % goal without a prescription reports the missing input.
        assert!(Goal::parse("PTV: D95% >= 95%")
            .unwrap()
            .check(&dvh, None)
            .is_err());
        // The grammar's own rules: unit mismatches and malformed lines.
        assert!(Goal::parse("PTV: V20Gy < 30 Gy").is_err());
        assert!(Goal::parse("PTV: Dmax < 45 cc").is_err());
        assert!(Goal::parse("no goal here").is_err());
    }

    #[test]
    fn a_course_summation_divides_by_the_fractions() {
        let dose = vec![30.0f32; 4];
        assert_eq!(tps_per_fraction(&dose, "FRACTION", 30), vec![1.0; 4]);
        assert_eq!(tps_per_fraction(&dose, "PLAN", 30), vec![1.0; 4]);
        // One fraction's dose and an unknown summation pass through unchanged.
        assert_eq!(tps_per_fraction(&dose, "FRACTION_SESSION", 30), dose);
        assert_eq!(tps_per_fraction(&dose, "", 30), dose);
    }

    #[test]
    fn a_variable_model_weights_each_voxel_by_its_let() {
        let mut dose = vec![2.0f32, 2.0, 0.0];
        apply_rbe(RbeModel::McNamara, &mut dose, &[2.0, 8.0, 0.0], 2.0);
        // 2 Gy at LETd 2 is near the clinical constant; LETd 8 is hotter.
        assert!(dose[0] > 2.0 && dose[0] < 3.0, "{}", dose[0]);
        assert!(dose[1] > dose[0]);
        // Zero dose takes RBEmax but stays zero.
        assert_eq!(dose[2], 0.0);
    }

    fn plan_spot(energy: f32, x: f32, y: f32, mu: f32) -> scan_kit_dicom::Spot {
        scan_kit_dicom::Spot {
            energy,
            x,
            y,
            mu,
            shifter: String::new(),
            wet_mm: 0.0,
            shifter_distance_mm: 0.0,
            spot_size_mm: 0.0,
        }
    }

    fn plan_beam(spots: Vec<scan_kit_dicom::Spot>) -> scan_kit_dicom::Beam {
        scan_kit_dicom::Beam {
            number: 1,
            name: "beam".into(),
            gantry: 0.0,
            couch: 0.0,
            isocenter: [0.0; 3],
            position: "HFS".into(),
            meterset: spots.iter().map(|spot| spot.mu).sum(),
            spots,
        }
    }

    // Two layers: three spots at 100 MeV, two at 150.
    fn plan() -> scan_kit_dicom::Beam {
        plan_beam(vec![
            plan_spot(100.0, -10.0, 0.0, 1.0),
            plan_spot(100.0, 0.0, 0.0, 1.0),
            plan_spot(100.0, 10.0, 0.0, 1.0),
            plan_spot(150.0, -5.0, 5.0, 2.0),
            plan_spot(150.0, 5.0, 5.0, 2.0),
        ])
    }

    #[test]
    fn an_exact_delivery_matches_every_layer() {
        let beam = plan();
        let energy: Vec<f32> = beam.spots.iter().map(|spot| spot.energy).collect();
        let x: Vec<f32> = beam.spots.iter().map(|spot| spot.x).collect();
        let y: Vec<f32> = beam.spots.iter().map(|spot| spot.y).collect();
        let mu: Vec<f32> = beam.spots.iter().map(|spot| spot.mu).collect();
        let matched = super::match_delivery(0, &beam, &energy, &x, &y, &mu);
        assert_eq!(matched.unmatched, 0);
        assert_eq!(matched.position_rms, 0.0);
        assert_eq!(matched.energy_error, 0.0);
        assert!(matched.layer_mu_ratio.iter().all(|r| *r == 1.0));
        assert_eq!(matched.spots.len(), 5);
        assert_eq!(matched.delivered_mu, matched.planned_mu);
        // Matched spots carry the layer's nominal energy, not the measured one.
        assert!(matched
            .spots
            .iter()
            .map(|spot| spot.energy)
            .eq([100.0, 100.0, 100.0, 150.0, 150.0]));
    }

    #[test]
    fn a_shifted_delivery_reports_its_errors() {
        let beam = plan();
        let mut energy: Vec<f32> = beam.spots.iter().map(|spot| spot.energy).collect();
        let mut x: Vec<f32> = beam.spots.iter().map(|spot| spot.x).collect();
        let y: Vec<f32> = beam.spots.iter().map(|spot| spot.y).collect();
        let mut mu: Vec<f32> = beam.spots.iter().map(|spot| spot.mu).collect();
        for i in 0..3 {
            energy[i] += 0.2;
            x[i] += 2.0;
            mu[i] *= 1.05;
        }
        for i in 3..5 {
            x[i] += 2.0;
        }
        // A stray spot no layer reaches within 0.5 MeV.
        energy.push(60.0);
        x.push(0.0);
        let mut y = y;
        y.push(0.0);
        mu.push(1.0);
        let matched = super::match_delivery(0, &beam, &energy, &x, &y, &mu);
        assert_eq!(matched.unmatched, 1);
        assert!((matched.energy_error - 0.2).abs() < 1e-4);
        assert!((matched.position_rms - 2.0).abs() < 1e-6);
        assert!((matched.layer_mu_ratio[0] - 1.05).abs() < 1e-4);
        assert!(matched.layer_mu_ratio[1..].iter().all(|r| *r == 1.0));
    }

    #[test]
    fn a_spot_with_no_measured_mu_is_unmatched() {
        let beam = plan();
        let energy: Vec<f32> = beam.spots.iter().map(|spot| spot.energy).collect();
        let x: Vec<f32> = beam.spots.iter().map(|spot| spot.x).collect();
        let y: Vec<f32> = beam.spots.iter().map(|spot| spot.y).collect();
        let mut mu: Vec<f32> = beam.spots.iter().map(|spot| spot.mu).collect();
        mu[0] = 0.0;
        let matched = super::match_delivery(0, &beam, &energy, &x, &y, &mu);
        assert_eq!(matched.unmatched, 1);
        assert!((matched.layer_mu_ratio[0] - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn the_closer_mu_breaks_a_beam_tie() {
        // Two beams share layer energies; the delivered MU sits nearer beam 1's plan.
        let a = plan_beam(vec![plan_spot(100.0, 0.0, 0.0, 1.0)]);
        let b = plan_beam(vec![plan_spot(100.0, 0.0, 0.0, 4.0)]);
        assert_eq!(super::pick_beam(&[a, b], &[100.0], &[3.5]), Some(1));
    }
}
