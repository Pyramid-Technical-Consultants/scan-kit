//! Dose on a loaded CT. Planned and delivered spots share one Monte Carlo seed.

use std::path::Path;

use scan_kit_core::{
    dvh, gamma_index, index, DataTable, McJob, Panel, PatientRequest, PlotScene, Volume,
};
use scan_kit_dicom::{hu_density, hu_label, inside_structure, load_study, PatientStudy};
use serde_json::Value;

use super::beam::{beam_record, protons_per_mu, spot_record};
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
    let groups = group_fractions(&deliveries.iter().map(|hit| hit.beam).collect::<Vec<_>>());
    let mut names = vec!["Plan".into()];
    for index in 1..=groups.len() {
        names.push(index.to_string());
    }
    let sent = options
        .get("fraction")
        .and_then(Value::as_str)
        .unwrap_or("Plan");
    let fraction = if names.iter().any(|name| name == sent) {
        sent
    } else {
        "Plan"
    };
    let beam_filter = options
        .get("beams")
        .and_then(Value::as_str)
        .and_then(|text| {
            text.parse::<usize>()
                .ok()
                .and_then(|index| index.checked_sub(1))
        });
    let (spots, protons, beams) =
        records(&study, root, fraction, &deliveries, &groups, beam_filter);
    let tables = mc_tables();
    let [nx, ny, nz] = study.ct.shape;
    let nvox = nx * ny * nz;
    let mut material = vec![0u8; nvox];
    let mut density = vec![0.0f32; nvox];
    for (index, hu) in study.ct.hu.iter().enumerate() {
        let label = hu_label(*hu);
        material[index] = tables.ids.iter().position(|id| *id == label).unwrap_or(0) as u8;
        density[index] = hu_density(*hu);
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
    let rbe = match pick(options, "rbe", "1", &[("1", "1.0"), ("1.1", "1.1")]) {
        "1.1" => 1.1f32,
        _ => 1.0,
    };
    if (rbe - 1.0).abs() > 1e-3 {
        for value in &mut dose.values {
            *value *= rbe;
        }
    }
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
    let goal = study
        .tps
        .as_ref()
        .map(|tps| tps.hu.iter().copied().fold(0.0f32, f32::max) * 0.95)
        .unwrap_or(0.0);
    for (id, structure) in study.structures.iter().enumerate() {
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
            ys: curve,
            color: MARK,
            thickness: 1.5,
        });
        if goal > 0.0 {
            let ptv: Vec<f32> = dose
                .values
                .iter()
                .zip(&mask)
                .filter_map(|(value, keep)| keep.then_some(*value))
                .collect();
            let result = scan_kit_dicom::clinical_goal(&ptv, goal, 0.95);
            rows.push(vec![
                structure.name.clone(),
                format!(
                    "{} {:.0}%",
                    if result.passed { "pass" } else { "fail" },
                    result.volume_fraction * 100.0
                ),
            ]);
        }
    }
    if dvh_lines.is_empty() {
        dvh_lines.push(crate::workspace::dvh_line(&dose));
    }
    let gamma = ran.as_ref().and_then(|result| {
        let tps = study.tps.as_ref()?;
        if tps.shape != dose.shape || tps.hu.len() != result.volume.values.len() {
            return None;
        }
        let (values, passed, scored) = gamma_index(
            &tps.hu,
            &result.volume.values,
            dose.shape,
            3.0,
            2.0,
            study.ct.spacing,
            10.0,
        );
        let rate = if scored == 0 {
            0.0
        } else {
            100.0 * passed as f32 / scored as f32
        };
        rows.push(vec!["Gamma".into(), format!("{rate:.0}%")]);
        Some((
            Volume {
                values,
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
    let (panels, volume) = crate::workspace::assemble(&crate::workspace::Workspace {
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
        field: None,
        show_field: false,
    });
    let (columns, column_weights, row_weights, row_splits) = crate::workspace::layout();
    let mut controls = vec![
        history_control(histories),
        scan_kit_core::Control::plain("fraction", "Fraction", &names, fraction).grouped("Study"),
        beam_control(&study, beam_filter),
        scan_kit_core::Control::plain(
            "rbe",
            "RBE",
            ["1.0", "1.1"],
            if (rbe - 1.1).abs() < 1e-3 {
                "1.1"
            } else {
                "1.0"
            },
        )
        .grouped("Study"),
    ];
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

struct Delivery {
    session: String,
    beam: usize,
}

const ENERGY_TOL: f32 = 0.5;

fn deliveries(root: &Path, sessions: &[String], study: &PatientStudy) -> Vec<Delivery> {
    sessions
        .iter()
        .filter_map(|session| {
            let table = spot_table(root, session);
            let energy = table.get("energy").cloned().unwrap_or_default();
            let beam = best_beam(study, &energy)?;
            Some(Delivery {
                session: session.clone(),
                beam,
            })
        })
        .collect()
}

fn best_beam(study: &PatientStudy, energy: &[f32]) -> Option<usize> {
    let mut best = None;
    for (index, beam) in study.beams.iter().enumerate() {
        let hits = energy
            .iter()
            .filter(|value| nearest_spot(beam, **value).is_some())
            .count();
        if hits > 0 && best.is_none_or(|(_, count)| hits > count) {
            best = Some((index, hits));
        }
    }
    best.map(|(index, _)| index)
}

fn nearest_spot(beam: &scan_kit_dicom::Beam, energy: f32) -> Option<&scan_kit_dicom::Spot> {
    beam.spots
        .iter()
        .filter(|spot| (spot.energy - energy).abs() <= ENERGY_TOL)
        .min_by(|a, b| {
            (a.energy - energy)
                .abs()
                .total_cmp(&(b.energy - energy).abs())
        })
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
    root: &Path,
    fraction: &str,
    deliveries: &[Delivery],
    groups: &[Vec<usize>],
    beam_filter: Option<usize>,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut spots = Vec::new();
    let mut protons = Vec::new();
    let mut beams = Vec::new();
    for (index, beam) in study.beams.iter().enumerate() {
        if let Ok(record) = beam_record(beam.gantry, beam.couch, &beam.position, beam.isocenter) {
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
                );
            }
        }
    }
    if fraction != "Plan" {
        let chosen = fraction.parse::<usize>().unwrap_or(1).saturating_sub(1);
        for hit in groups.get(chosen).into_iter().flatten() {
            let delivery = &deliveries[*hit];
            if beam_filter.is_some_and(|chosen| chosen != delivery.beam) {
                continue;
            }
            let beam = &study.beams[delivery.beam];
            let table = spot_table(root, &delivery.session);
            let energy = table.get("energy").cloned().unwrap_or_default();
            let x = table.get("plan_x").cloned().unwrap_or_default();
            let y = table.get("plan_y").cloned().unwrap_or_default();
            let measured = table.get("ic1_dose").cloned().unwrap_or_default();
            let target = table.get("target_mu").cloned().unwrap_or_default();
            for i in 0..energy.len() {
                let Some(plan_spot) = nearest_spot(beam, energy[i]) else {
                    continue;
                };
                let mu = measured
                    .get(i)
                    .copied()
                    .filter(|value| value.is_finite() && *value > 0.0)
                    .or_else(|| target.get(i).copied())
                    .unwrap_or(0.0);
                push_spot(
                    &mut spots,
                    &mut protons,
                    plan_spot.energy,
                    x.get(i).copied().unwrap_or(plan_spot.x),
                    y.get(i).copied().unwrap_or(plan_spot.y),
                    mu,
                    delivery.beam,
                    plan_spot.wet_mm,
                    plan_spot.shifter_distance_mm,
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
) {
    if let Ok(record) = spot_record(energy, x, y, beam as f32, wet, distance) {
        spots.extend_from_slice(&record);
        protons.push(mu * protons_per_mu(energy));
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
    .grouped("Model")
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

#[cfg(test)]
mod tests {
    use super::group_fractions;

    #[test]
    fn a_repeated_beam_starts_another_fraction() {
        assert_eq!(group_fractions(&[0, 1]), vec![vec![0, 1]]);
        assert_eq!(group_fractions(&[0, 0]), vec![vec![0], vec![1]]);
        assert_eq!(group_fractions(&[0, 1, 0]), vec![vec![0, 1], vec![2]]);
    }
}
