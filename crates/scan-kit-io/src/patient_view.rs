//! Dose on a loaded CT. Planned and delivered spots share one Monte Carlo seed.

use std::path::Path;

use scan_kit_core::{
    dvh, gamma_index, index, DataTable, McJob, Panel, PatientRequest, PlotScene, Volume,
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
    let groups = group_fractions(&deliveries.iter().map(|hit| hit.beam).collect::<Vec<_>>());
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
    let (spots, protons, beams) = records(
        &study,
        root,
        fraction,
        &deliveries,
        &groups,
        beam_filter,
        bdl,
    );
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
    let rbe = match pick(options, "rbe", "1.1", &[("1", "1.0"), ("1.1", "1.1")]) {
        "1" | "1.0" => 1.0f32,
        _ => 1.1,
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
    let goals_text = goal_text(options, &study);
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
        let reference = study
            .tps
            .as_ref()
            .map(|tps| {
                tps.hu
                    .iter()
                    .zip(&mask)
                    .filter_map(|(value, keep)| keep.then_some(*value))
                    .fold(0.0f32, f32::max)
            })
            .filter(|value| *value > 0.0)
            .map(|peak| peak * rbe)
            .unwrap_or_else(|| {
                dose.values
                    .iter()
                    .zip(&mask)
                    .filter_map(|(value, keep)| keep.then_some(*value))
                    .fold(0.0f32, f32::max)
            });
        for (name, body) in parse_goals(&goals_text) {
            if !structure.name.eq_ignore_ascii_case(&name) {
                continue;
            }
            rows.push(vec![
                structure.name.clone(),
                eval_goal(&body, &dose.values, &mask, reference),
            ]);
        }
        rows.push(vec!["DVH".into(), structure.name.clone()]);
        for (edge, volume) in edges.iter().zip(&curve) {
            rows.push(vec!["curve".into(), format!("{edge:.4},{volume:.4}")]);
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
        let (dd, dta, cutoff) = gamma_criteria(options);
        let (values, passed, scored) = gamma_index(
            &tps.hu,
            &result.volume.values,
            dose.shape,
            dd,
            dta,
            study.ct.spacing,
            cutoff,
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
    let (panels, mut volume) = crate::workspace::assemble(&crate::workspace::Workspace {
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
    volume.unit = if (rbe - 1.0).abs() > 1e-3 {
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
            ["1.0", "1.1"],
            if (rbe - 1.0).abs() < 1e-3 {
                "1.0"
            } else {
                "1.1"
            },
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

fn parse_goals(text: &str) -> Vec<(String, String)> {
    text.lines()
        .filter_map(|line| {
            let line = line.trim();
            let (name, rest) = line.split_once(':')?;
            let name = name.trim();
            let rest = rest.trim();
            (!name.is_empty() && !rest.is_empty()).then(|| (name.to_string(), rest.to_string()))
        })
        .collect()
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

fn eval_goal(body: &str, dose: &[f32], mask: &[bool], reference: f32) -> String {
    let compact: String = body
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect::<String>()
        .to_ascii_lowercase();
    if let Some(rest) = compact.strip_prefix("dmax") {
        let (op, rest) = split_cmp(rest);
        let (limit, _) = take_number(rest);
        let max = masked_max(dose, mask);
        return verdict(compare(max, op, limit), format!("Dmax {max:.2} Gy"));
    }
    if let Some(rest) = compact.strip_prefix('v') {
        let (gy, rest) = take_number(rest);
        let rest = rest.trim_start_matches("gy");
        let (op, rest) = split_cmp(rest);
        let (pct, _) = take_number(rest);
        let got = volume_above(dose, mask, gy) * 100.0;
        return verdict(compare(got, op, pct), format!("V{gy:.0} {got:.0}%"));
    }
    if let Some(rest) = compact.strip_prefix('d') {
        let (vol, rest) = take_number(rest);
        let rest = rest.trim_start_matches('%');
        let (op, rest) = split_cmp(rest);
        let (limit, unit) = take_number(rest);
        let (edges, curve) = dvh(dose, mask, 32);
        let got = dose_at_volume(&edges, &curve, vol / 100.0);
        let want = if unit.contains("gy") {
            limit
        } else {
            reference * limit / 100.0
        };
        return verdict(compare(got, op, want), format!("D{vol:.0}% {got:.2} Gy"));
    }
    "unparsed".into()
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

fn split_cmp(text: &str) -> (&str, &str) {
    for op in [">=", "<=", ">", "<"] {
        if let Some(rest) = text.strip_prefix(op) {
            return (op, rest);
        }
    }
    (">=", text)
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

fn masked_max(dose: &[f32], mask: &[bool]) -> f32 {
    dose.iter()
        .zip(mask)
        .filter_map(|(value, keep)| keep.then_some(*value))
        .fold(0.0f32, f32::max)
}

fn volume_above(dose: &[f32], mask: &[bool], gy: f32) -> f32 {
    let mut hit = 0.0;
    let mut total = 0.0;
    for (value, keep) in dose.iter().zip(mask) {
        if *keep {
            total += 1.0;
            if *value >= gy {
                hit += 1.0;
            }
        }
    }
    if total == 0.0 {
        0.0
    } else {
        hit / total
    }
}

#[cfg(test)]
mod tests {
    use super::{eval_goal, group_fractions, parse_goals};

    #[test]
    fn a_repeated_beam_starts_another_fraction() {
        assert_eq!(group_fractions(&[0, 1]), vec![vec![0, 1]]);
        assert_eq!(group_fractions(&[0, 0]), vec![vec![0], vec![1]]);
        assert_eq!(group_fractions(&[0, 1, 0]), vec![vec![0, 1], vec![2]]);
    }

    #[test]
    fn a_flat_dose_meets_d95_and_fails_a_tight_maximum() {
        let dose = vec![2.0; 8];
        let mask = vec![true; 8];
        let pass = eval_goal("D95% >= 95%", &dose, &mask, 2.0);
        assert!(pass.starts_with("pass"), "{pass}");
        let fail = eval_goal("Dmax < 1 Gy", &dose, &mask, 2.0);
        assert!(fail.starts_with("fail"), "{fail}");
        let volume = eval_goal("V20Gy < 30%", &dose, &mask, 2.0);
        assert!(volume.starts_with("pass"), "{volume}");
        assert_eq!(parse_goals("PTV: D95% >= 95%").len(), 1);
    }
}
