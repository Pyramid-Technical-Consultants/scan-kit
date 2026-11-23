//! Shared plot helpers: option picking and row filters.

use scan_kit_core::Control;
use serde_json::Value;

pub(crate) fn labeled(id: &str, label: &str, pairs: &[(&str, &str)], current: &str) -> Control {
    let value = pairs
        .iter()
        .find(|(key, _)| *key == current)
        .map(|(_, label)| *label)
        .unwrap_or(current);
    let mut control = control(
        id,
        label,
        &pairs.iter().map(|(_, label)| *label).collect::<Vec<_>>(),
        value,
    );
    for (choice, (key, _)) in control.options.iter_mut().zip(pairs) {
        choice.id = (*key).to_string();
    }
    control
}

pub(crate) fn control(id: &str, label: &str, options: &[&str], value: &str) -> Control {
    Control::plain(id, label, options.iter().copied(), value)
}

pub(crate) fn text<'a>(options: &'a Value, key: &str, default: &'a str) -> &'a str {
    options.get(key).and_then(Value::as_str).unwrap_or(default)
}

pub(crate) fn pick(
    options: &Value,
    key: &str,
    default_id: &'static str,
    pairs: &[(&'static str, &'static str)],
) -> &'static str {
    pick_in(options, key, pairs).unwrap_or(default_id)
}

pub(crate) fn pick_in<'a>(
    options: &Value,
    key: &str,
    pairs: &[(&'a str, &'a str)],
) -> Option<&'a str> {
    let raw = options.get(key).and_then(Value::as_str)?;
    pairs
        .iter()
        .find(|(id, label)| *id == raw || *label == raw)
        .map(|(id, _)| *id)
}

/// Normalize an options object at the trust boundary: JSON numbers and
/// booleans become their string form because every option read is `as_str`,
/// and `_preview` is removed so a caller cannot set a knob the stage writes
/// for itself (`_lines` is a legitimate client key and stays).
pub fn normalize_options(options: &mut Value) {
    let Some(map) = options.as_object_mut() else {
        return;
    };
    map.remove("_preview");
    for value in map.values_mut() {
        match value.clone() {
            Value::Number(number) => {
                *value = Value::String(match number.as_i64() {
                    Some(int) => int.to_string(),
                    // serde_json keeps integral floats as f64  render
                    // them like ints so flag()/pick() recognize them.
                    None => match number.as_f64() {
                        Some(float) if float.fract() == 0.0 && float.abs() < 9.0e15 => {
                            format!("{}", float as i64)
                        }
                        _ => number.to_string(),
                    },
                });
            }
            Value::Bool(flag) => *value = Value::String(flag.to_string()),
            _ => {}
        }
    }
}

pub(crate) fn flag(options: &Value, key: &str, default_on: bool) -> bool {
    match text(options, key, if default_on { "On" } else { "Off" }) {
        // MCP callers hand us "False"/"TRUE" — match case-insensitively
        // rather than silently falling back to the default.
        raw if raw.eq_ignore_ascii_case("off")
            || raw.eq_ignore_ascii_case("false")
            || raw == "0" =>
        {
            false
        }
        raw if raw.eq_ignore_ascii_case("on") || raw.eq_ignore_ascii_case("true") || raw == "1" => {
            true
        }
        _ => default_on,
    }
}

#[cfg(test)]
mod tests {
    use super::super::tables::signal_table;

    #[test]
    fn timeslice_signals_keep_low_confidence_and_tracking_error() {
        let device = b"\
rci_in_trigger,r_ic1_x_confidence,r_ic1_x_peak_amplitude,c_x,r_xV,r_tx2_probe_x
1,90,4,1.0,1.2,30
0,10,1,1.0,1.5,-10
1,40,2,2.0,2.0,5
";
        let other = b"rci_in_trigger,r_ic1_current\n1,3\n";
        let files = vec![device.to_vec(), other.to_vec()];
        let energies = [150.0];
        let layers = [0, 1];
        let close = |got: &[f32], want: &[f32]| {
            assert_eq!(got.len(), want.len());
            for (got, want) in got.iter().zip(want) {
                assert!((got - want).abs() < 1e-4, "{got} vs {want}");
            }
        };

        let confidence = signal_table(&files, &energies, &layers, "fit_confidence");
        // The second file has a trigger and no confidence sample. That row stays.
        close(&confidence["ic1_x_confidence"][..3], &[90.0, 10.0, 40.0]);
        assert!(confidence["ic1_x_confidence"][3].is_nan());
        close(&confidence["energy"][..3], &[150.0, 150.0, 150.0]);
        assert!(confidence["energy"][3].is_nan());
        close(&confidence["beam_on"], &[1.0, 0.0, 1.0, 1.0]);
        assert!(!confidence.contains_key("ic1_y_confidence"));

        let mut filtered = confidence;
        scan_kit_core::apply_mask(
            &mut filtered,
            &[scan_kit_core::Segment::Beam {
                state: scan_kit_core::BeamGate::On,
            }],
            &["ic1_x_confidence"],
        );
        assert!(filtered["ic1_x_confidence"][0].is_finite());
        assert!(filtered["ic1_x_confidence"][1].is_nan());
        assert!((filtered["ic1_x_confidence"][2] - 40.0).abs() < 1e-4);
        assert!(filtered["ic1_x_confidence"][3].is_nan());

        let peak = signal_table(&files, &energies, &layers, "peak_amplitude");
        close(&peak["ic1_x_peak"][..3], &[4.0, 1.0, 2.0]);
        assert!(peak["ic1_x_peak"][3].is_nan());
        assert!(!peak.contains_key("ic1_y_peak"));

        let amplifier = signal_table(&files, &energies, &layers, "amplifier_error");
        close(&amplifier["amp_x"][..3], &[0.2, 0.5, 0.0]);
        assert!(amplifier["amp_x"][3].is_nan());
        assert!(!amplifier.contains_key("amp_y"));

        let field = signal_table(&files, &energies, &layers, "probe_field");
        close(&field["field_x"][..3], &[30.0, -10.0, 5.0]);
        assert!(field["field_x"][3].is_nan());
        assert!(!field.contains_key("field_y"));
    }

    #[test]
    fn flag_reads_title_and_upper_case_booleans() {
        use serde_json::json;
        let options = json!({"a": "False", "b": "TRUE", "c": "OFF", "d": "On"});
        assert!(!super::flag(&options, "a", true));
        assert!(super::flag(&options, "b", false));
        assert!(!super::flag(&options, "c", true));
        assert!(super::flag(&options, "d", false));
    }

    #[test]
    fn integral_floats_normalize_like_integers() {
        // serde_json renders 0.0 as "0.0" - flag()/pick() only know the
        // integer spellings, so normalize must emit them.
        let mut options = serde_json::json!({"roi0": 0.0, "histories": 3000000.0, "cut": 0.5});
        super::normalize_options(&mut options);
        assert_eq!(options["roi0"], "0");
        assert_eq!(options["histories"], "3000000");
        assert_eq!(options["cut"], "0.5");
    }
}
