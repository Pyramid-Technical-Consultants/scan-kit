//! Patient QA report: a small document model and its PDF backend.
//!
//! The model is what `scan_kit.qa.report.write_report` rendered as HTML —
//! heading, paragraphs, tables, key-value pairs, a figure and a provenance
//! block — laid out here as pages instead. Callers assemble a [`Report`];
//! [`write_pdf`] and [`dvh_csv`] are the outputs.

use std::path::{Path, PathBuf};

use krilla::color::rgb;
use krilla::geom::{PathBuilder, Point, Size, Transform};
use krilla::image::Image;
use krilla::num::NormalizedF32;
use krilla::page::PageSettings;
use krilla::paint::{Fill, FillRule, Stroke};
use krilla::text::{Font, TextDirection};
use krilla::Document;
use serde_json::Value;

pub const DISCLAIMER: &str = "scan-kit research dose, not for clinical use";

const FONT: &[u8] = include_bytes!("../../scan-kit-plot/assets/SourceSans3-Regular.ttf");
const PAGE_W: f32 = 595.0; // A4, points
const PAGE_H: f32 = 842.0;
const MARGIN: f32 = 46.0;
const ROW_H: f32 = 14.0;
const TEXT_W: f32 = PAGE_W - 2.0 * MARGIN;

/// One block of report content, top to bottom.
pub enum Section {
    Heading(String),
    Paragraph(String),
    Table {
        head: Vec<String>,
        rows: Vec<Vec<String>>,
    },
    KeyValues(Vec<(String, String)>),
    /// Raw RGBA pixels (a plot frame); scaled to the text width.
    Figure {
        rgba: Vec<u8>,
        width: u32,
        height: u32,
    },
}

pub struct Report {
    pub title: String,
    pub disclaimer: String,
    pub sections: Vec<Section>,
    pub provenance: Value,
}

/// One structure's cumulative DVH for the CSV sidecar `write_report` wrote.
pub struct DvhCurve {
    pub name: String,
    pub edges: Vec<f32>,
    pub cumulative: Vec<f32>,
}

/// `%.6g`, the format the Python sidecar used.
fn sig6(value: f64) -> String {
    if value == 0.0 {
        return "0".into();
    }
    let exponent = value.abs().log10().floor() as i32;
    // printf %g prints scientific outside [1e-4, 1e6).
    if !(-4..6).contains(&exponent) {
        let text = format!("{value:.5e}");
        let (mantissa, exp) = text.split_once('e').unwrap();
        return format!(
            "{}e{exp}",
            mantissa.trim_end_matches('0').trim_end_matches('.')
        );
    }
    let decimals = (5 - exponent).max(0) as usize;
    let mut text = format!("{value:.decimals$}");
    if text.contains('.') {
        while text.ends_with('0') {
            text.pop();
        }
        if text.ends_with('.') {
            text.pop();
        }
    }
    text
}

/// `name_dvh.csv`: dose in Gy(RBE), one volume-% column per structure.
pub fn dvh_csv(curves: &[DvhCurve]) -> String {
    let Some(first) = curves.first() else {
        return String::new();
    };
    let mut out = String::from("dose_gy_rbe");
    for curve in curves {
        out.push(',');
        out.push_str(&curve.name.replace(',', " "));
        out.push_str(" volume %");
    }
    out.push('\n');
    for (row, edge) in first.edges.iter().enumerate() {
        out.push_str(&sig6(*edge as f64));
        for curve in curves {
            let volume = curve.cumulative.get(row).copied().unwrap_or(0.0);
            out.push_str(&format!(",{}", sig6(100.0 * volume as f64)));
        }
        out.push('\n');
    }
    out
}

fn text_width(text: &str, size: f32) -> f32 {
    static METRIC: std::sync::OnceLock<fontdue::Font> = std::sync::OnceLock::new();
    let font = METRIC.get_or_init(|| {
        fontdue::Font::from_bytes(FONT, fontdue::FontSettings::default()).expect("report font")
    });
    text.chars()
        .map(|c| font.metrics(c, size).advance_width)
        .sum()
}

/// Wrap `text` to the column width at `size`; one line stays one line.
fn wrap(text: &str, size: f32, width: f32) -> Vec<String> {
    let mut lines = Vec::new();
    let mut line = String::new();
    for word in text.split(' ') {
        let candidate = if line.is_empty() {
            word.to_owned()
        } else {
            format!("{line} {word}")
        };
        if !line.is_empty() && text_width(&candidate, size) > width {
            lines.push(std::mem::take(&mut line));
            line = word.to_owned();
        } else {
            line = candidate;
        }
    }
    if !line.is_empty() {
        lines.push(line);
    }
    if lines.is_empty() {
        lines.push(String::new());
    }
    lines
}

#[derive(Clone, Copy)]
enum Ink {
    Body,
    Faint,
    Warn,
}

/// A laid-out draw op; `y` grows downward from the page top.
enum Op {
    Text {
        x: f32,
        y: f32,
        size: f32,
        ink: Ink,
        text: String,
    },
    Rule {
        y: f32,
    },
    Image {
        y: f32,
        h: f32,
        rgba: Vec<u8>,
        width: u32,
        height: u32,
    },
}

impl Op {
    fn bottom(&self) -> f32 {
        match *self {
            Op::Text { y, .. } => y + 3.0,
            Op::Rule { y } => y + 1.0,
            Op::Image { y, h, .. } => y + h,
        }
    }
}

/// Lay `report` out into ops and split them into pages.
fn lay_out(report: &Report) -> Vec<Vec<Op>> {
    fn text(ops: &mut Vec<Op>, x: f32, baseline: f32, size: f32, ink: Ink, text: &str) {
        ops.push(Op::Text {
            x,
            y: baseline,
            size,
            ink,
            text: text.to_owned(),
        });
    }
    let mut ops = Vec::new();
    let mut y = MARGIN;

    text(&mut ops, MARGIN, y + 16.0, 20.0, Ink::Body, &report.title);
    y += 30.0;
    text(
        &mut ops,
        MARGIN,
        y + 9.0,
        10.0,
        Ink::Warn,
        &report.disclaimer,
    );
    y += 20.0;

    for section in &report.sections {
        match section {
            Section::Heading(heading) => {
                y += 10.0;
                text(&mut ops, MARGIN, y + 11.0, 14.0, Ink::Body, heading);
                y += 20.0;
            }
            Section::Paragraph(body) => {
                for line in wrap(body, 10.0, TEXT_W) {
                    text(&mut ops, MARGIN, y + 9.0, 10.0, Ink::Body, &line);
                    y += 14.0;
                }
                y += 4.0;
            }
            Section::KeyValues(pairs) => {
                let key_w = pairs
                    .iter()
                    .map(|(key, _)| text_width(&format!("{key}  "), 10.0))
                    .fold(0.0f32, f32::max)
                    .min(200.0);
                for (key, value) in pairs {
                    for line in wrap(value, 10.0, TEXT_W - key_w - 8.0) {
                        text(&mut ops, MARGIN, y + 9.0, 10.0, Ink::Faint, key);
                        text(
                            &mut ops,
                            MARGIN + key_w + 8.0,
                            y + 9.0,
                            10.0,
                            Ink::Body,
                            &line,
                        );
                        y += 14.0;
                    }
                }
                y += 4.0;
            }
            Section::Table { head, rows } => {
                let cols = head.len();
                let mut natural = vec![30.0f32; cols];
                for (col, label) in head.iter().enumerate() {
                    natural[col] = natural[col].max(text_width(label, 9.0) + 10.0);
                }
                for row in rows {
                    for (col, cell) in row.iter().enumerate() {
                        if col < cols {
                            natural[col] = natural[col].max(text_width(cell, 9.0) + 10.0);
                        }
                    }
                }
                let sum: f32 = natural.iter().sum();
                let widths: Vec<f32> = natural.iter().map(|w| w * TEXT_W / sum).collect();
                let mut x = MARGIN;
                for (col, label) in head.iter().enumerate() {
                    text(&mut ops, x + 4.0, y + 10.0, 9.0, Ink::Faint, label);
                    x += widths[col];
                }
                y += ROW_H;
                ops.push(Op::Rule { y });
                for row in rows {
                    let mut x = MARGIN;
                    for (col, cell) in row.iter().enumerate() {
                        if col >= cols {
                            break;
                        }
                        text(&mut ops, x + 4.0, y + 10.0, 9.0, Ink::Body, cell);
                        x += widths[col];
                    }
                    y += ROW_H;
                }
                ops.push(Op::Rule { y });
                y += 8.0;
            }
            Section::Figure {
                rgba,
                width,
                height,
            } => {
                let scale = TEXT_W / *width as f32;
                let h = *height as f32 * scale;
                ops.push(Op::Image {
                    y,
                    h,
                    rgba: rgba.clone(),
                    width: *width,
                    height: *height,
                });
                y += h + 8.0;
            }
        }
    }

    // Provenance is one preformatted block at the end.
    y += 10.0;
    text(&mut ops, MARGIN, y + 11.0, 14.0, Ink::Body, "Provenance");
    y += 20.0;
    let dump = serde_json::to_string_pretty(&report.provenance).unwrap_or_default();
    for line in dump.lines() {
        text(&mut ops, MARGIN, y + 7.0, 7.5, Ink::Faint, line);
        y += 11.0;
    }

    // Greedy pagination: an op moves to the next page when it no longer fits,
    // then every op's y becomes page-local.
    let mut pages: Vec<Vec<Op>> = vec![Vec::new()];
    let mut offset = 0.0f32;
    for op in ops {
        if !pages.last().unwrap().is_empty() && op.bottom() - offset > PAGE_H - MARGIN {
            offset += PAGE_H - 2.0 * MARGIN;
            pages.push(Vec::new());
        }
        pages.last_mut().unwrap().push(op);
    }
    let mut cursor = 0.0f32;
    for ops in pages.iter_mut() {
        for op in ops.iter_mut() {
            match op {
                Op::Text { y, .. } | Op::Rule { y } | Op::Image { y, .. } => *y -= cursor,
            }
        }
        cursor += PAGE_H - 2.0 * MARGIN;
    }
    pages
}

fn fill(ink: Ink) -> Fill {
    let (r, g, b) = match ink {
        Ink::Body => (30, 30, 30),
        Ink::Faint => (110, 110, 110),
        Ink::Warn => (160, 32, 32),
    };
    Fill {
        paint: rgb::Color::new(r, g, b).into(),
        opacity: NormalizedF32::ONE,
        rule: FillRule::NonZero,
    }
}

/// Write `report` to `dest` as PDF; the provenance JSON is the last block.
pub fn write_pdf(report: &Report, dest: &Path) -> Result<PathBuf, String> {
    let font = Font::new(FONT.to_vec().into(), 0).ok_or("the report font did not load")?;
    let settings = PageSettings::from_wh(PAGE_W, PAGE_H).unwrap();
    let mut doc = Document::new();
    for ops in lay_out(report) {
        let mut page = doc.start_page_with(settings.clone());
        {
            let mut surface = page.surface();
            for op in ops {
                match op {
                    Op::Text {
                        x,
                        y,
                        size,
                        ink,
                        text,
                    } => {
                        surface.set_fill(Some(fill(ink)));
                        surface.draw_text(
                            Point::from_xy(x, PAGE_H - y),
                            font.clone(),
                            size,
                            &text,
                            false,
                            TextDirection::Auto,
                        );
                    }
                    Op::Rule { y } => {
                        surface.set_stroke(Some(Stroke {
                            paint: rgb::Color::new(160, 160, 160).into(),
                            opacity: NormalizedF32::ONE,
                            width: 0.5,
                            ..Default::default()
                        }));
                        let mut path = PathBuilder::new();
                        path.move_to(MARGIN, PAGE_H - y);
                        path.line_to(PAGE_W - MARGIN, PAGE_H - y);
                        surface.draw_path(&path.finish().unwrap());
                    }
                    Op::Image {
                        y,
                        h,
                        rgba,
                        width,
                        height,
                    } => {
                        let image = Image::from_rgba8(rgba, width, height);
                        surface
                            .push_transform(&Transform::from_translate(MARGIN, PAGE_H - (y + h)));
                        surface.draw_image(image, Size::from_wh(TEXT_W, h).unwrap());
                        surface.pop();
                    }
                }
            }
            surface.finish();
        }
        page.finish();
    }
    let pdf = doc.finish().map_err(|err| format!("pdf: {err:?}"))?;
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent).map_err(|err| format!("{}: {err}", parent.display()))?;
    }
    std::fs::write(dest, &pdf).map_err(|err| format!("{}: {err}", dest.display()))?;
    Ok(dest.to_path_buf())
}

/// The figure's rendered frame plus the caption naming each curve.
pub struct Figure {
    pub rgba: Vec<u8>,
    pub width: u32,
    pub height: u32,
    /// e.g. "PTV (blue), Cord (orange)" — the plot has no legend, so the
    /// caption is how a reader maps a curve to its structure.
    pub caption: String,
}

/// Assemble the patient QA document `scan_kit.qa.report.write_report` built:
/// figure, delivery, gamma, goals, dose statistics. Returns the report and the
/// DVH curves its CSV sidecar writes.
pub fn patient_report(
    title: &str,
    metrics: &crate::patient_view::PatientMetrics,
    rbe_label: &str,
    figure: Option<Figure>,
    provenance: Value,
) -> (Report, Vec<DvhCurve>) {
    let mut sections = Vec::new();
    if let Some(figure) = figure {
        sections.push(Section::Figure {
            rgba: figure.rgba,
            width: figure.width,
            height: figure.height,
        });
        if !figure.caption.is_empty() {
            sections.push(Section::Paragraph(figure.caption));
        }
    }
    if !metrics.matches.is_empty() {
        sections.push(Section::Heading("Delivery".into()));
        sections.push(Section::Table {
            head: [
                "Beam",
                "Planned MU",
                "Delivered MU",
                "Ratio",
                "Unmatched spots",
                "Worst energy (MeV)",
                "Position RMS (mm)",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
            rows: metrics
                .matches
                .iter()
                .map(|(_, m)| {
                    vec![
                        (m.beam + 1).to_string(),
                        format!("{:.2}", m.planned_mu),
                        format!("{:.2}", m.delivered_mu),
                        if m.planned_mu > 0.0 {
                            format!("{:.4}", m.delivered_mu / m.planned_mu)
                        } else {
                            String::new()
                        },
                        m.unmatched.to_string(),
                        format!("{:.2}", m.energy_error),
                        format!("{:.2}", m.position_rms),
                    ]
                })
                .collect(),
        });
    }
    if let Some(gamma) = &metrics.gamma {
        sections.push(Section::Heading("Gamma vs TPS".into()));
        sections.push(Section::Paragraph(format!(
            "{} % / {} mm, {} % cutoff: {:.2} % of {} voxels pass",
            sig6(f64::from(gamma.dose_pct)),
            sig6(f64::from(gamma.dta_mm)),
            sig6(f64::from(gamma.cutoff_pct)),
            gamma.rate,
            gamma.scored,
        )));
    }
    let goal_rows: Vec<Vec<String>> = metrics
        .structures
        .iter()
        .flat_map(|structure| {
            structure.goals.iter().map(|(goal, check)| {
                let text = format!(
                    "{}: {} {} {} {}",
                    goal.roi,
                    goal.metric_text,
                    goal.op,
                    sig6(f64::from(goal.value)),
                    goal.unit
                );
                match check {
                    Ok((measured, pass)) => vec![
                        text,
                        format!("{measured:.2} {}", goal.unit),
                        if *pass { "pass".into() } else { "FAIL".into() },
                    ],
                    Err(err) => vec![text, String::new(), format!("fail {err}")],
                }
            })
        })
        .collect();
    if !goal_rows.is_empty() {
        sections.push(Section::Heading("Clinical goals".into()));
        sections.push(Section::Table {
            head: ["Goal", "Value", "Result"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            rows: goal_rows,
        });
    }
    let curves: Vec<DvhCurve> = metrics
        .structures
        .iter()
        .map(|structure| DvhCurve {
            name: structure.name.clone(),
            edges: structure.dvh.edges.clone(),
            cumulative: structure.dvh.cumulative(),
        })
        .collect();
    if !curves.is_empty() {
        sections.push(Section::Heading(format!(
            "Dose statistics (Gy(RBE), {rbe_label})"
        )));
        sections.push(dose_stats(
            &metrics
                .structures
                .iter()
                .map(|structure| DoseStat {
                    name: structure.name.clone(),
                    volume_cc: f64::from(structure.dvh.volume_cc()),
                    mean: f64::from(structure.dvh.mean()),
                    d98: f64::from(structure.dvh.dose_at(0.98)),
                    d95: f64::from(structure.dvh.dose_at(0.95)),
                    d50: f64::from(structure.dvh.dose_at(0.5)),
                    d2: f64::from(structure.dvh.dose_at(0.02)),
                    max: f64::from(structure.dvh.dose_at(0.0)),
                    let_stats: structure.let_stats,
                })
                .collect::<Vec<_>>(),
        ));
    }
    (
        Report {
            title: title.to_string(),
            disclaimer: DISCLAIMER.to_string(),
            sections,
            provenance,
        },
        curves,
    )
}

/// A structure's dose metrics for the dose-statistics table.
pub struct DoseStat {
    pub name: String,
    pub volume_cc: f64,
    pub mean: f64,
    pub d98: f64,
    pub d95: f64,
    pub d50: f64,
    pub d2: f64,
    pub max: f64,
    pub let_stats: Option<(f64, f64)>,
}

/// The patient QA report's dose table as `DoseStat` rows, `write_report`'s
/// "Dose statistics" block.
pub fn dose_stats(stats: &[DoseStat]) -> Section {
    let let_any = stats.iter().any(|stat| stat.let_stats.is_some());
    let mut head: Vec<String> = [
        "Structure",
        "Volume (cc)",
        "Dmean",
        "D98%",
        "D95%",
        "D50%",
        "D2%",
        "Dmax",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    if let_any {
        head.push("LETd (keV/µm)".into());
        head.push("RBE".into());
    }
    let rows = stats
        .iter()
        .map(|stat| {
            let mut row = vec![
                stat.name.clone(),
                format!("{:.1}", stat.volume_cc),
                format!("{:.3}", stat.mean),
                format!("{:.3}", stat.d98),
                format!("{:.3}", stat.d95),
                format!("{:.3}", stat.d50),
                format!("{:.3}", stat.d2),
                format!("{:.3}", stat.max),
            ];
            if let_any {
                match stat.let_stats {
                    Some((letd, rbe)) => {
                        row.push(format!("{letd:.2}"));
                        row.push(format!("{rbe:.3}"));
                    }
                    None => {
                        row.push(String::new());
                        row.push(String::new());
                    }
                }
            }
            row
        })
        .collect();
    Section::Table { head, rows }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_report_writes_a_readable_pdf_and_dvh_csv() {
        let dir = std::env::temp_dir().join(format!("scan-kit-report-{}", std::process::id()));
        let dest = dir.join("qa.pdf");
        let report = Report {
            title: "Patient QA".into(),
            disclaimer: DISCLAIMER.into(),
            sections: vec![
                Section::Heading("Delivery".into()),
                Section::Table {
                    head: vec!["Beam".into(), "MU".into()],
                    rows: vec![vec!["1".into(), "10.5".into()]],
                },
                Section::Figure {
                    rgba: vec![255u8; 64 * 64 * 4],
                    width: 64,
                    height: 64,
                },
                Section::KeyValues(vec![("Gamma".into(), "99.2% pass".into())]),
            ],
            provenance: serde_json::json!({"engine": "test", "histories": 1000}),
        };
        let out = write_pdf(&report, &dest).unwrap();
        let bytes = std::fs::read(&out).unwrap();
        assert!(bytes.starts_with(b"%PDF"));
        assert!(bytes.len() > 2000);
        let csv = dvh_csv(&[
            DvhCurve {
                name: "PTV".into(),
                edges: vec![0.0, 1.0, 2.0],
                cumulative: vec![1.0, 0.5, 0.0],
            },
            DvhCurve {
                name: "a, b".into(),
                edges: vec![0.0, 1.0, 2.0],
                cumulative: vec![1.0, 1.0, 0.25],
            },
        ]);
        assert!(csv.starts_with("dose_gy_rbe,PTV volume %,a  b volume %"));
        assert!(csv.contains("\n2,0,25\n"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn patient_metrics_become_the_report_sections() {
        let dose = vec![2.0f32; 8];
        let dvh = crate::patient_view::Dvh::of(&dose, &[true; 8], 0.125);
        let goal = crate::patient_view::Goal::parse("PTV: Dmean >= 1.9 Gy").unwrap();
        let metrics = crate::patient_view::PatientMetrics {
            matches: Vec::new(),
            structures: vec![crate::patient_view::StructureMetric {
                id: 0,
                name: "PTV".into(),
                mask: vec![true; 8],
                dvh,
                goals: vec![(goal, Ok((2.0, true)))],
                let_stats: Some((4.2, 1.14)),
            }],
            gamma: Some(crate::patient_view::GammaMetric {
                display: scan_kit_core::Volume {
                    origin: [0.0; 3],
                    shape: [1, 1, 1],
                    voxel: 1.0,
                    values: vec![0.5],
                },
                rate: 98.5,
                passed: 985,
                scored: 1000,
                dose_pct: 3.0,
                dta_mm: 3.0,
                cutoff_pct: 10.0,
            }),
        };
        let (report, curves) = patient_report(
            "Patient QA",
            &metrics,
            "McNamara 2015",
            None,
            serde_json::json!({"histories": 1000}),
        );
        let heads: Vec<&str> = report
            .sections
            .iter()
            .filter_map(|section| match section {
                Section::Heading(text) => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(
            heads,
            [
                "Gamma vs TPS",
                "Clinical goals",
                "Dose statistics (Gy(RBE), McNamara 2015)"
            ]
        );
        let goals = report
            .sections
            .iter()
            .find_map(|section| match section {
                Section::Table { head, rows } if head[0] == "Goal" => Some(rows),
                _ => None,
            })
            .unwrap();
        assert_eq!(goals[0][0], "PTV: Dmean >= 1.9 Gy");
        assert_eq!(goals[0][2], "pass");
        assert_eq!(curves.len(), 1);
        assert_eq!(curves[0].name, "PTV");
    }

    #[test]
    fn wrapping_respects_the_column_width() {
        let long = "word ".repeat(200);
        let lines = wrap(&long, 10.0, 100.0);
        assert!(lines.len() > 4);
        assert!(lines.iter().all(|line| text_width(line, 10.0) <= 100.0));
    }

    #[test]
    fn sig6_matches_six_significant_digits() {
        assert_eq!(sig6(0.0), "0");
        assert_eq!(sig6(2.0), "2");
        assert_eq!(sig6(1.99975), "1.99975");
        assert_eq!(sig6(12.345678), "12.3457");
        assert_eq!(sig6(1234567.0), "1.23457e6");
        assert_eq!(sig6(0.00123), "0.00123");
    }
}
