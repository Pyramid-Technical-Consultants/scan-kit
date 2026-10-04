//! One row mask for every analysis filter.
//!
//! A segment is a predicate. The list is combined with AND. Rejected samples
//! become NaN in the data columns. `beam_on` stays readable so a later segment
//! can still see the gate. A column range is an energy or time window. A
//! compare is a threshold such as dose error over 1%.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::plot::{Choice, Control};

const MOD_Z: f32 = 3.5;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BeamGate {
    On,
    Off,
    Both,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Rank {
    #[serde(rename = "all")]
    All,
    #[serde(rename = "lower_95")]
    Lower95,
    #[serde(rename = "upper_95")]
    Upper95,
    #[serde(rename = "mad")]
    Mad,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompareOp {
    #[serde(rename = "gt")]
    Gt,
    #[serde(rename = "gte")]
    Gte,
    #[serde(rename = "lt")]
    Lt,
    #[serde(rename = "lte")]
    Lte,
    #[serde(rename = "abs_gt")]
    AbsGt,
}

/// One cut. `any` is reserved for a later OR group and is rejected by the parser.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Segment {
    Beam {
        state: BeamGate,
    },
    Rank {
        which: Rank,
    },
    Range {
        column: String,
        lo: f32,
        hi: f32,
    },
    Compare {
        column: String,
        op: CompareOp,
        threshold: f32,
    },
}

pub fn segments_json(items: &[Segment]) -> String {
    serde_json::to_string(items).unwrap_or_else(|_| "[]".to_owned())
}

pub fn parse_segments(text: &str) -> Result<Vec<Segment>, String> {
    let value: Value = serde_json::from_str(text).map_err(|err| err.to_string())?;
    let items = value.as_array().ok_or("segments must be a list")?;
    items.iter().map(parse_segment).collect()
}

fn parse_segment(value: &Value) -> Result<Segment, String> {
    if value.get("kind").and_then(Value::as_str) == Some("any") {
        return Err("or groups are not available yet".to_owned());
    }
    serde_json::from_value(value.clone()).map_err(|err| err.to_string())
}

/// `segments` wins. Absent that, `beam` and `domain` fill the matching defaults.
pub fn segments_from(options: &Value, defaults: &[Segment]) -> Vec<Segment> {
    if let Some(text) = options.get("segments").and_then(Value::as_str) {
        if let Ok(items) = parse_segments(text) {
            return items;
        }
    }
    let beam = options.get("beam").and_then(Value::as_str);
    let domain = options.get("domain").and_then(Value::as_str);
    if beam.is_none() && domain.is_none() {
        return defaults.to_vec();
    }
    let mut items = Vec::new();
    if let Some(raw) = beam {
        items.push(Segment::Beam {
            state: beam_gate(raw),
        });
    } else if let Some(state) = default_beam(defaults) {
        items.push(Segment::Beam { state });
    }
    if let Some(raw) = domain {
        items.push(Segment::Rank {
            which: rank_which(raw),
        });
    } else if let Some(which) = default_rank(defaults) {
        items.push(Segment::Rank { which });
    }
    for item in defaults {
        if !matches!(item, Segment::Beam { .. } | Segment::Rank { .. }) {
            items.push(item.clone());
        }
    }
    items
}

fn default_beam(defaults: &[Segment]) -> Option<BeamGate> {
    defaults.iter().find_map(|item| match item {
        Segment::Beam { state } => Some(*state),
        _ => None,
    })
}

fn default_rank(defaults: &[Segment]) -> Option<Rank> {
    defaults.iter().find_map(|item| match item {
        Segment::Rank { which } => Some(*which),
        _ => None,
    })
}

fn beam_gate(raw: &str) -> BeamGate {
    match raw {
        "beam_off" | "Beam Off" | "off" => BeamGate::Off,
        "beam_both" | "Both" | "both" => BeamGate::Both,
        _ => BeamGate::On,
    }
}

fn rank_which(raw: &str) -> Rank {
    match raw {
        "lower_95" | "Lower 95%" => Rank::Lower95,
        "upper_95" | "Upper 5%" => Rank::Upper95,
        "mad_outliers" | "MAD Outliers" => Rank::Mad,
        _ => Rank::All,
    }
}

/// The sidebar control. `kinds` are the segments this view can add (`id`, `label`).
pub fn segments_control(items: &[Segment], kinds: &[(&str, &str)]) -> Control {
    Control {
        id: "segments".to_owned(),
        label: "Segments".to_owned(),
        options: kinds
            .iter()
            .map(|(id, label)| Choice::full(id, label, "", ""))
            .collect(),
        value: segments_json(items),
        group: "Filter Data".to_owned(),
        kind: "segments".to_owned(),
    }
}

/// True where every segment passes. `rank_on` is the plotted columns for a rank cut.
pub fn row_mask(
    table: &BTreeMap<String, Vec<f32>>,
    items: &[Segment],
    rank_on: &[&str],
) -> Vec<bool> {
    let n = table.values().map(Vec::len).max().unwrap_or(0);
    let mut keep = vec![true; n];
    for item in items {
        match item {
            Segment::Beam { state } => {
                beam_keep(table.get("beam_on").map(Vec::as_slice), *state, &mut keep)
            }
            Segment::Rank { which } => rank_keep(table, *which, rank_on, &mut keep),
            Segment::Range { column, lo, hi } => {
                range_keep(table.get(column).map(Vec::as_slice), *lo, *hi, &mut keep);
            }
            Segment::Compare {
                column,
                op,
                threshold,
            } => compare_keep(
                table.get(column).map(Vec::as_slice),
                *op,
                *threshold,
                &mut keep,
            ),
        }
    }
    keep
}

/// NaN out rejected samples. The gate and the guide columns stay put.
pub fn apply_mask(table: &mut BTreeMap<String, Vec<f32>>, items: &[Segment], rank_on: &[&str]) {
    let keep = row_mask(table, items, rank_on);
    for (key, values) in table.iter_mut() {
        if matches!(
            key.as_str(),
            "beam_on" | "expected_sigma" | "session_avg_rate"
        ) {
            continue;
        }
        for (value, slot) in values.iter_mut().zip(&keep) {
            if !slot {
                *value = f32::NAN;
            }
        }
    }
}

fn beam_keep(gate: Option<&[f32]>, state: BeamGate, keep: &mut [bool]) {
    let Some(gate) = gate else {
        return;
    };
    for (slot, value) in keep.iter_mut().zip(gate) {
        let on = !value.is_finite() || *value > 0.5;
        let pass = match state {
            BeamGate::Off => !on,
            BeamGate::On => on,
            BeamGate::Both => true,
        };
        *slot = *slot && pass;
    }
}

fn rank_keep(table: &BTreeMap<String, Vec<f32>>, which: Rank, rank_on: &[&str], keep: &mut [bool]) {
    if which == Rank::All {
        return;
    }
    let columns: Vec<&[f32]> = rank_on
        .iter()
        .filter_map(|key| table.get(*key).map(Vec::as_slice))
        .collect();
    if columns.is_empty() {
        return;
    }
    let n = keep.len();
    if which == Rank::Mad {
        let scores: Vec<Vec<f32>> = columns.iter().map(|column| modified_z(column)).collect();
        for row in 0..n {
            let outlier = scores
                .iter()
                .any(|axis| axis.get(row).copied().unwrap_or(0.0).abs() > MOD_Z);
            keep[row] = keep[row] && outlier;
        }
        return;
    }
    let severity: Vec<f32> = (0..n)
        .map(|row| {
            columns
                .iter()
                .filter_map(|column| column.get(row).copied())
                .filter(|value| value.is_finite())
                .map(f32::abs)
                .fold(None, |acc: Option<f32>, value| {
                    Some(acc.map(|best| best.max(value)).unwrap_or(value))
                })
                .unwrap_or(f32::NAN)
        })
        .collect();
    let valid: Vec<f32> = severity
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    if valid.is_empty() {
        return;
    }
    let cutoff = percentile(&valid, 0.95);
    for (row, value) in severity.iter().enumerate() {
        let pass = value.is_finite()
            && match which {
                Rank::Upper95 => *value > cutoff,
                _ => *value <= cutoff,
            };
        keep[row] = keep[row] && pass;
    }
}

fn range_keep(column: Option<&[f32]>, lo: f32, hi: f32, keep: &mut [bool]) {
    let Some(column) = column else {
        return;
    };
    for (slot, value) in keep.iter_mut().zip(column) {
        *slot = *slot && value.is_finite() && *value >= lo && *value <= hi;
    }
}

fn compare_keep(column: Option<&[f32]>, op: CompareOp, threshold: f32, keep: &mut [bool]) {
    let Some(column) = column else {
        return;
    };
    for (slot, value) in keep.iter_mut().zip(column) {
        let pass = value.is_finite()
            && match op {
                CompareOp::Gt => *value > threshold,
                CompareOp::Gte => *value >= threshold,
                CompareOp::Lt => *value < threshold,
                CompareOp::Lte => *value <= threshold,
                CompareOp::AbsGt => value.abs() > threshold,
            };
        *slot = *slot && pass;
    }
}

fn percentile(values: &[f32], q: f64) -> f32 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if sorted.is_empty() {
        return f32::NAN;
    }
    let pos = q * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil().min((sorted.len() - 1) as f64) as usize;
    let frac = (pos - lo as f64) as f32;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

fn median(values: &[f32]) -> f32 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[mid - 1] + sorted[mid]) * 0.5
    } else {
        sorted[mid]
    }
}

fn modified_z(values: &[f32]) -> Vec<f32> {
    let finite: Vec<f32> = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    let mut out = vec![0.0; values.len()];
    if finite.is_empty() {
        return out;
    }
    let med = median(&finite);
    let devs: Vec<f32> = finite.iter().map(|value| (value - med).abs()).collect();
    let mad = median(&devs);
    let mean = finite.iter().sum::<f32>() / finite.len() as f32;
    let mean_ad = devs.iter().sum::<f32>() / devs.len() as f32;
    for (slot, value) in out.iter_mut().zip(values) {
        if !value.is_finite() {
            continue;
        }
        *slot = if mad > 1e-12 {
            0.6745 * (value - med) / mad
        } else if mean_ad > 1e-12 {
            (value - mean) / (1.253314 * mean_ad)
        } else {
            0.0
        };
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> BTreeMap<String, Vec<f32>> {
        let values: Vec<f32> = (0..20).map(|value| value as f32).chain([1000.0]).collect();
        let mut beam = vec![1.0; values.len()];
        beam[0] = 0.0;
        BTreeMap::from([("y".to_owned(), values), ("beam_on".to_owned(), beam)])
    }

    fn finite(table: &BTreeMap<String, Vec<f32>>) -> Vec<f32> {
        table["y"]
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .collect()
    }

    #[test]
    fn beam_on_off_and_both_match_the_gate() {
        let mut both = sample();
        apply_mask(
            &mut both,
            &[Segment::Beam {
                state: BeamGate::Both,
            }],
            &["y"],
        );
        assert_eq!(finite(&both).len(), 21);

        let mut off = sample();
        apply_mask(
            &mut off,
            &[Segment::Beam {
                state: BeamGate::Off,
            }],
            &["y"],
        );
        assert_eq!(finite(&off), vec![0.0]);
        assert!(off["beam_on"][0].abs() < 1e-6);

        let mut on = sample();
        apply_mask(
            &mut on,
            &[Segment::Beam {
                state: BeamGate::On,
            }],
            &["y"],
        );
        let kept = finite(&on);
        assert!(!kept.contains(&0.0));
        assert!(kept.contains(&1000.0));
    }

    #[test]
    fn a_missing_gate_keeps_every_row() {
        let mut table = BTreeMap::from([("y".to_owned(), vec![1.0, 2.0])]);
        apply_mask(
            &mut table,
            &[Segment::Beam {
                state: BeamGate::On,
            }],
            &["y"],
        );
        assert!(table["y"].iter().all(|value| value.is_finite()));
    }

    #[test]
    fn rank_keeps_the_tail_the_body_or_the_outlier() {
        let mut upper = sample();
        apply_mask(
            &mut upper,
            &[Segment::Rank {
                which: Rank::Upper95,
            }],
            &["y"],
        );
        assert_eq!(finite(&upper), vec![1000.0]);

        let mut lower = sample();
        apply_mask(
            &mut lower,
            &[Segment::Rank {
                which: Rank::Lower95,
            }],
            &["y"],
        );
        let kept = finite(&lower);
        assert!(kept.contains(&1.0));
        assert!(!kept.contains(&1000.0));

        let mut mad = sample();
        apply_mask(&mut mad, &[Segment::Rank { which: Rank::Mad }], &["y"]);
        assert_eq!(finite(&mad), vec![1000.0]);

        let mut all = sample();
        apply_mask(&mut all, &[Segment::Rank { which: Rank::All }], &["y"]);
        assert_eq!(finite(&all).len(), 21);
    }

    #[test]
    fn a_range_and_a_compare_can_stand_alone() {
        let mut energy = BTreeMap::from([("energy".to_owned(), vec![70.0, 90.0, 110.0])]);
        apply_mask(
            &mut energy,
            &[Segment::Range {
                column: "energy".to_owned(),
                lo: 80.0,
                hi: 100.0,
            }],
            &[],
        );
        assert!(energy["energy"][0].is_nan());
        assert!((energy["energy"][1] - 90.0).abs() < 1e-6);
        assert!(energy["energy"][2].is_nan());

        let mut dose = BTreeMap::from([("dose_error".to_owned(), vec![0.2, -1.5, 0.4])]);
        apply_mask(
            &mut dose,
            &[Segment::Compare {
                column: "dose_error".to_owned(),
                op: CompareOp::AbsGt,
                threshold: 1.0,
            }],
            &[],
        );
        assert!(dose["dose_error"][0].is_nan());
        assert!((dose["dose_error"][1] + 1.5).abs() < 1e-6);
        assert!(dose["dose_error"][2].is_nan());
    }

    #[test]
    fn segments_and_together() {
        let mut table = sample();
        apply_mask(
            &mut table,
            &[
                Segment::Beam {
                    state: BeamGate::Off,
                },
                Segment::Rank {
                    which: Rank::Upper95,
                },
            ],
            &["y"],
        );
        assert!(finite(&table).is_empty());
    }

    #[test]
    fn legacy_beam_and_domain_strings_fill_the_list() {
        let items = segments_from(
            &serde_json::json!({"beam": "Beam Off", "domain": "Upper 5%"}),
            &[
                Segment::Beam {
                    state: BeamGate::On,
                },
                Segment::Rank { which: Rank::All },
            ],
        );
        assert_eq!(
            items,
            vec![
                Segment::Beam {
                    state: BeamGate::Off
                },
                Segment::Rank {
                    which: Rank::Upper95
                },
            ]
        );
        let only_beam = segments_from(
            &serde_json::json!({"beam": "beam_both"}),
            &[
                Segment::Beam {
                    state: BeamGate::On,
                },
                Segment::Rank {
                    which: Rank::Lower95,
                },
            ],
        );
        assert_eq!(
            only_beam,
            vec![
                Segment::Beam {
                    state: BeamGate::Both
                },
                Segment::Rank {
                    which: Rank::Lower95
                },
            ]
        );
    }

    #[test]
    fn an_or_group_is_rejected() {
        let err = parse_segments(r#"[{"kind":"any","of":[]}]"#).unwrap_err();
        assert!(err.contains("or groups"));
    }

    #[test]
    fn a_segments_string_round_trips() {
        let items = vec![
            Segment::Beam {
                state: BeamGate::On,
            },
            Segment::Rank { which: Rank::Mad },
        ];
        assert_eq!(parse_segments(&segments_json(&items)).unwrap(), items);
    }
}
