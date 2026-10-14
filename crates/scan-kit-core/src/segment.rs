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
///
/// An on `scrub` option appends a `time_s` window. The sidebar list does not carry it.
pub fn segments_from(options: &Value, defaults: &[Segment]) -> Vec<Segment> {
    let mut items = listed_segments(options, defaults);
    if let Some(range) = scrub_range(options) {
        items.push(range);
    }
    items
}

fn listed_segments(options: &Value, defaults: &[Segment]) -> Vec<Segment> {
    if let Some(text) = options.get("segments").and_then(Value::as_str) {
        if let Ok(items) = parse_segments(text) {
            return without_playhead(items);
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
    without_playhead(items)
}

/// The playback bar owns `time_s`. A copy left in the segment list would freeze the playhead.
fn without_playhead(items: Vec<Segment>) -> Vec<Segment> {
    items
        .into_iter()
        .filter(|item| !matches!(item, Segment::Range { column, .. } if column == "time_s"))
        .collect()
}

/// `(lo, hi)` of the playhead window, or nothing when the timeline is off.
/// `before` starts at negative infinity. The one-second window starts one second earlier.
pub fn scrub_limits(options: &Value) -> Option<(f32, f32)> {
    let value = scrub_value(options)?;
    if value.get("on").and_then(Value::as_bool) != Some(true) {
        return None;
    }
    let at = finite_f32(value.get("at")).unwrap_or(0.0);
    let before = value.get("window").and_then(Value::as_str) == Some("before");
    let lo = if before { f32::NEG_INFINITY } else { at - 1.0 };
    Some((lo, at))
}

fn scrub_range(options: &Value) -> Option<Segment> {
    let (lo, hi) = scrub_limits(options)?;
    Some(Segment::Range {
        column: "time_s".to_owned(),
        lo,
        hi,
    })
}

fn scrub_value(options: &Value) -> Option<Value> {
    let raw = options.get("scrub")?.as_str()?;
    serde_json::from_str(raw).ok()
}

fn finite_f32(value: Option<&Value>) -> Option<f32> {
    let number = value?.as_f64()? as f32;
    number.is_finite().then_some(number)
}

fn scrub_speed(value: Option<f64>) -> f64 {
    let Some(value) = value else {
        return 1.0;
    };
    if (value - 0.1).abs() < 1e-3 {
        0.1
    } else if (value - 10.0).abs() < 1e-3 {
        10.0
    } else {
        1.0
    }
}

/// Times where `layer` changes, on the same rows as `time`. The origin is not a mark.
pub fn layer_edges(time: &[f32], layer: &[f32]) -> Vec<f32> {
    let mut marks = Vec::new();
    let mut previous: Option<i64> = None;
    for (at, id) in time.iter().zip(layer) {
        if !at.is_finite() || !id.is_finite() {
            continue;
        }
        let id = *id as i64;
        if previous.is_some_and(|seen| seen != id) && *at > 0.0 {
            marks.push(*at);
        }
        previous = Some(id);
    }
    marks
}

/// Largest finite `time_s`. Missing columns stay at 0.
pub fn time_end<'a>(columns: impl IntoIterator<Item = &'a [f32]>) -> f32 {
    let mut end = 0.0f32;
    for column in columns {
        for value in column {
            if value.is_finite() {
                end = end.max(*value);
            }
        }
    }
    end
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
        "beam_on" | "Beam On" | "on" | "On" => BeamGate::On,
        "beam_off" | "Beam Off" | "off" | "Off" => BeamGate::Off,
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

/// The playback bar. `end` is the largest `time_s`. `layers` are change times on that clock.
pub fn scrub_control(options: &Value, end: f32, layers: &[f32]) -> Control {
    let end = if end.is_finite() { end.max(0.0) } else { 0.0 };
    let value = scrub_value(options);
    let on = value
        .as_ref()
        .and_then(|item| item.get("on"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let at = finite_f32(value.as_ref().and_then(|item| item.get("at"))).unwrap_or(0.0);
    let speed = scrub_speed(
        value
            .as_ref()
            .and_then(|item| item.get("speed"))
            .and_then(Value::as_f64),
    );
    let window = if value
        .as_ref()
        .and_then(|item| item.get("window"))
        .and_then(Value::as_str)
        == Some("before")
    {
        "before"
    } else {
        "second"
    };
    let layers: Vec<f32> = layers
        .iter()
        .copied()
        .filter(|mark| mark.is_finite() && *mark > 0.0 && *mark <= end)
        .collect();
    let text = serde_json::json!({
        "on": on,
        "at": at.clamp(0.0, end),
        "end": end,
        "speed": speed,
        "window": window,
        "layers": layers,
    });
    Control {
        id: "scrub".to_owned(),
        label: "Timeline".to_owned(),
        options: Vec::new(),
        value: text.to_string(),
        group: String::new(),
        kind: "scrub".to_owned(),
    }
}

/// The sidebar control. `kinds` are the segments this view can add (`id`, `label`).
pub fn segments_control(items: &[Segment], kinds: &[(&str, &str)]) -> Control {
    let shown = without_playhead(items.to_vec());
    Control {
        id: "segments".to_owned(),
        label: "Segments".to_owned(),
        options: kinds
            .iter()
            .map(|(id, label)| Choice::full(id, label, "", ""))
            .collect(),
        value: segments_json(&shown),
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
    let mut values = values.to_vec();
    crate::stats::percentile_linear(&mut values, q)
}

fn median(values: &[f32]) -> f32 {
    let mut values = values.to_vec();
    crate::stats::median_unstable(&mut values)
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

    fn scrubbed(window: &str, on: bool) -> Vec<f32> {
        let mut table = BTreeMap::from([
            ("y".to_owned(), vec![1.0, 2.0, 3.0, 4.0]),
            ("time_s".to_owned(), vec![0.0, 0.5, 1.2, 2.0]),
        ]);
        let raw = format!(r#"{{"on":{on},"at":1.2,"window":"{window}"}}"#);
        let items = segments_from(&serde_json::json!({ "scrub": raw }), &[]);
        apply_mask(&mut table, &items, &["y"]);
        finite(&table)
    }

    #[test]
    fn scrub_keeps_a_second_a_prefix_or_everything() {
        assert_eq!(scrubbed("second", true), vec![2.0, 3.0]);
        assert_eq!(scrubbed("before", true), vec![1.0, 2.0, 3.0]);
        assert_eq!(scrubbed("second", false), vec![1.0, 2.0, 3.0, 4.0]);
        let plain = segments_from(&serde_json::json!({}), &[]);
        assert!(plain.is_empty());

        let mut bare = BTreeMap::from([("y".to_owned(), vec![1.0, 2.0])]);
        let raw = r#"{"on":true,"at":0.5,"window":"before"}"#;
        apply_mask(
            &mut bare,
            &segments_from(&serde_json::json!({ "scrub": raw }), &[]),
            &["y"],
        );
        assert_eq!(finite(&bare), vec![1.0, 2.0]);
    }

    #[test]
    fn the_playhead_stays_out_of_the_segment_list() {
        let stale =
            r#"[{"kind":"beam","state":"on"},{"kind":"range","column":"time_s","lo":0,"hi":0.1}]"#;
        let scrub = r#"{"on":true,"at":2.0,"speed":0.1,"window":"before"}"#;
        let options = serde_json::json!({ "segments": stale, "scrub": scrub });
        let items = segments_from(&options, &[]);
        let mut table = BTreeMap::from([
            ("y".to_owned(), vec![1.0, 2.0, 3.0]),
            ("time_s".to_owned(), vec![0.0, 0.5, 2.0]),
            ("beam_on".to_owned(), vec![1.0, 0.0, 1.0]),
        ]);
        apply_mask(&mut table, &items, &["y"]);
        assert_eq!(finite(&table), vec![1.0, 3.0]);

        let listed = segments_control(&items, &[("beam", "Beam")]);
        let shown = parse_segments(&listed.value).unwrap();
        assert!(shown.iter().all(|item| !matches!(
            item,
            Segment::Range { column, .. } if column == "time_s"
        )));

        let off = segments_from(&serde_json::json!({ "segments": stale }), &[]);
        assert!(off.iter().all(|item| !matches!(
            item,
            Segment::Range { column, .. } if column == "time_s"
        )));

        let control = scrub_control(&options, 2.0, &[0.0, 0.5, 3.0]);
        let echoed: serde_json::Value = serde_json::from_str(&control.value).unwrap();
        assert_eq!(echoed["on"], true);
        assert_eq!(echoed["speed"], 0.1);
        assert_eq!(echoed["window"], "before");
        assert!((echoed["at"].as_f64().unwrap() - 2.0).abs() < 1e-6);
        assert!((echoed["end"].as_f64().unwrap() - 2.0).abs() < 1e-6);
        let layers = echoed["layers"].as_array().unwrap();
        assert_eq!(layers.len(), 1);
        assert!((layers[0].as_f64().unwrap() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn a_layer_change_marks_the_first_row_of_the_new_layer() {
        let time = [0.0, 0.1, 0.2, 0.4];
        let layer = [1.0, 1.0, 2.0, f32::NAN];
        assert_eq!(layer_edges(&time, &layer), vec![0.2]);
        assert!(layer_edges(&time, &[1.0, 1.0, 1.0, 1.0]).is_empty());
    }

    #[test]
    fn a_one_second_window_keeps_the_millisecond_on_each_edge() {
        let times: Vec<f32> = (0..2_500).map(|index| index as f32 * 0.001).collect();
        let at = f64::from(times[2_000]);
        let raw = format!(r#"{{"on":true,"at":{at},"window":"second"}}"#);
        let mut table = BTreeMap::from([
            ("y".to_owned(), times.clone()),
            ("time_s".to_owned(), times),
        ]);
        apply_mask(
            &mut table,
            &segments_from(&serde_json::json!({ "scrub": raw }), &[]),
            &["y"],
        );
        let kept: Vec<f32> = finite(&table);
        assert!(
            kept.contains(&table_time(1_000)),
            "the sample one second back stays"
        );
        assert!(kept.contains(&table_time(2_000)));
        assert!(
            !kept.contains(&table_time(999)),
            "older than one second drops"
        );
        assert!(
            !kept.contains(&table_time(2_001)),
            "past the playhead drops"
        );
    }

    fn table_time(index: i32) -> f32 {
        index as f32 * 0.001
    }
}
