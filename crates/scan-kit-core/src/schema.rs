//! Column aliases and the G2 current scale.
//!
//! The tables match `scan_kit/common/schema.py`. That module stays until the
//! Qt app no longer imports it.

pub const POSITION_KEY_G2_RAW: &str = "spot_raw";
pub const POSITION_KEY_G3_RAW: &str = "spot_position_raw";

pub struct ConceptAlias {
    pub concept: &'static str,
    pub names: &'static [&'static str],
}

pub static CONCEPT_ALIASES: &[ConceptAlias] = &[
    ConceptAlias {
        concept: "energy",
        names: &[
            "ENERGY",
            "energy",
            "beam_energy",
            "energy_mev",
            "nominal_energy",
        ],
    },
    ConceptAlias {
        concept: "layer_id",
        names: &[
            "layer_id",
            "layer",
            "layerid",
            "layer_index",
            "layer_number",
        ],
    },
    ConceptAlias {
        concept: "spot_no",
        names: &["spot_no", "spot", "spot_id", "spot_number", "spot_index"],
    },
    ConceptAlias {
        concept: "timestamp",
        names: &["timestamp", "time_ms", "time_stamp", "timestamp_ms"],
    },
    ConceptAlias {
        concept: "time_s",
        names: &["time_s", "time_seconds", "time_sec"],
    },
    ConceptAlias {
        concept: "time_ns",
        names: &["time_ns", "time_nanoseconds", "time_nano"],
    },
    ConceptAlias {
        concept: "ic1_current",
        names: &[
            "ic1_primary_channel",
            "ic1",
            "ic1_current",
            "ic1_primary",
            "r_ic1_current_dose",
            "r_ic1_current_dose_filt",
        ],
    },
    ConceptAlias {
        concept: "ic2_current",
        names: &[
            "ic2_primary_channel",
            "ic2",
            "ic2_current",
            "ic2_primary",
            "r_ic2_current_dose",
            "r_ic2_current_dose_filt",
        ],
    },
    ConceptAlias {
        concept: "ic1_strip_sum",
        names: &["ic1_strip_sum_channel", "ic1_strip_sum", "ic1_stripsum"],
    },
    ConceptAlias {
        concept: "ic2_strip_sum",
        names: &["ic2_strip_sum_channel", "ic2_strip_sum", "ic2_stripsum"],
    },
    ConceptAlias {
        concept: "ic3_current_a",
        names: &[
            "ic3_current_A",
            "ic3_current_a",
            "ic3_a_current",
            "ic3_current_1",
            "r_px3_1_strip_sum",
        ],
    },
    ConceptAlias {
        concept: "ic3_current_b",
        names: &[
            "ic3_current_B",
            "ic3_current_b",
            "ic3_b_current",
            "ic3_current_2",
        ],
    },
    ConceptAlias {
        concept: "ic3_current_c",
        names: &[
            "ic3_current_C",
            "ic3_current_c",
            "ic3_c_current",
            "ic3_current_3",
        ],
    },
    ConceptAlias {
        concept: "ic3_current_d",
        names: &[
            "ic3_current_D",
            "ic3_current_d",
            "ic3_d_current",
            "ic3_current_4",
        ],
    },
    ConceptAlias {
        concept: "beam_current",
        names: &[
            "c_beam_current",
            "c_beamI",
            "c_beami",
            "r_beamI",
            "r_beami",
            "beam_current",
            "rci_beam_current",
            "beam_i",
        ],
    },
    ConceptAlias {
        concept: "ic1_total_dose",
        names: &[
            "ic1_total_dose_spot",
            "ic1_total_dose_spot_raw",
            "ic1_total_dose",
            "ic1_dose_spot_raw",
        ],
    },
    ConceptAlias {
        concept: "ic2_total_dose",
        names: &[
            "ic2_total_dose_spot",
            "ic2_total_dose_spot_raw",
            "ic2_total_dose",
            "ic2_dose_spot_raw",
        ],
    },
    ConceptAlias {
        concept: "ic3_total_dose",
        names: &[
            "r_ic3_total_dose_spot",
            "r_ic3_total_dose_spot_raw",
            "ic3_total_dose_spot_raw",
            "ic3_total_dose",
            "r_ic3_total_dose",
        ],
    },
    ConceptAlias {
        concept: "ic1_scan_total_dose",
        names: &[
            "r_ic1_scan_total_dose",
            "ic1_scan_total_dose",
            "ic1_dose_total",
        ],
    },
    ConceptAlias {
        concept: "ic2_scan_total_dose",
        names: &[
            "r_ic2_scan_total_dose",
            "ic2_scan_total_dose",
            "ic2_dose_total",
        ],
    },
    ConceptAlias {
        concept: "ic3_scan_total_dose",
        names: &[
            "ic3_dose_total",
            "ic3_scan_total_dose",
            "r_ic3_scan_total_dose",
            "r_ic3_dose_total",
        ],
    },
    ConceptAlias {
        concept: "charge_req",
        names: &["CHARGE_REQ", "charge_req", "dose_req", "prescribed_dose"],
    },
    ConceptAlias {
        concept: "mag_field_x",
        names: &[
            "r_tx2_probe_x",
            "field_c_x",
            "r_xB",
            "mag_field_x",
            "field_x",
            "b_field_x",
        ],
    },
    ConceptAlias {
        concept: "mag_field_y",
        names: &[
            "r_tx2_probe_y",
            "field_c_y",
            "r_yB",
            "mag_field_y",
            "field_y",
            "b_field_y",
        ],
    },
    ConceptAlias {
        concept: "amplifier_cmd_x",
        names: &["amplifier_x_target", "c_x", "amplifier_cmd_x"],
    },
    ConceptAlias {
        concept: "amplifier_cmd_y",
        names: &["amplifier_y_target", "c_y", "amplifier_cmd_y"],
    },
    ConceptAlias {
        concept: "amplifier_readback_x",
        names: &[
            "amplifier_x_readback",
            "r_xI",
            "r_xV",
            "amplifier_readback_x",
        ],
    },
    ConceptAlias {
        concept: "amplifier_readback_y",
        names: &[
            "amplifier_y_readback",
            "r_yI",
            "r_yV",
            "amplifier_readback_y",
        ],
    },
    ConceptAlias {
        concept: "x_position",
        names: &[
            "X_POSITION",
            "x_position",
            "xposition",
            "x_pos",
            "planned_x",
        ],
    },
    ConceptAlias {
        concept: "y_position",
        names: &[
            "Y_POSITION",
            "y_position",
            "yposition",
            "y_pos",
            "planned_y",
        ],
    },
    ConceptAlias {
        concept: "ic1_x_peak_amplitude",
        names: &[
            "r_ic1_x_peak_amplitude",
            "ic1_x_peak_amplitude",
            "ic1_peak_amplitude_x",
        ],
    },
    ConceptAlias {
        concept: "ic1_y_peak_amplitude",
        names: &[
            "r_ic1_y_peak_amplitude",
            "ic1_y_peak_amplitude",
            "ic1_peak_amplitude_y",
        ],
    },
    ConceptAlias {
        concept: "ic2_x_peak_amplitude",
        names: &[
            "r_ic2_x_peak_amplitude",
            "ic2_x_peak_amplitude",
            "ic2_peak_amplitude_x",
        ],
    },
    ConceptAlias {
        concept: "ic2_y_peak_amplitude",
        names: &[
            "r_ic2_y_peak_amplitude",
            "ic2_y_peak_amplitude",
            "ic2_peak_amplitude_y",
        ],
    },
];

pub struct DerivedSum {
    pub target: &'static str,
    pub sources: &'static [&'static str],
    pub scale: f64,
}

/// G2 per-axis strip sums, added into one canonical column when that column is absent.
pub static DERIVED_SUMS: &[DerivedSum] = &[
    DerivedSum {
        target: "ic1_strip_sum",
        sources: &["r_ic1_x_strip_sum", "r_ic1_y_strip_sum"],
        scale: 1e12,
    },
    DerivedSum {
        target: "ic2_strip_sum",
        sources: &["r_ic2_x_strip_sum", "r_ic2_y_strip_sum"],
        scale: 1e12,
    },
];

pub static IC3_QUAD_ZERO_FILL: &[&str] = &["ic3_current_b", "ic3_current_c", "ic3_current_d"];

const INTEGER_CONCEPTS: &[&str] = &["layer_id", "spot_no"];

pub fn column_is_integer(concept: &str) -> bool {
    INTEGER_CONCEPTS.contains(&concept)
}

/// G2 IC current is charge in coulombs over a 1 ms timeslice. nA = coulombs * 1e12.
/// G3 names are absent here because those columns are already nA.
pub fn column_scale_factor(raw_name: &str) -> Option<f64> {
    match normalize_column_name(raw_name).as_str() {
        "r_ic1_current_dose"
        | "r_ic1_current_dose_filt"
        | "r_ic2_current_dose"
        | "r_ic2_current_dose_filt"
        | "r_px3_1_strip_sum" => Some(1e12),
        _ => None,
    }
}

pub fn normalize_column_name(name: &str) -> String {
    let mut out = String::new();
    let mut pending_break = false;
    for ch in name.trim().chars() {
        if ch.is_ascii_alphanumeric() {
            if pending_break && !out.is_empty() {
                out.push('_');
            }
            pending_break = false;
            out.push(ch.to_ascii_lowercase());
        } else {
            pending_break = true;
        }
    }
    out
}

pub fn resolve_column_name<'a>(columns: &'a [String], requested: &str) -> Option<&'a str> {
    if let Some(found) = columns.iter().find(|column| column.as_str() == requested) {
        return Some(found);
    }
    let want = normalize_column_name(requested);
    let mut found = None;
    for column in columns {
        if normalize_column_name(column) == want {
            found = Some(column.as_str());
        }
    }
    found
}

pub fn concept_column_candidates(concept: &str, position_key: Option<&str>) -> Vec<String> {
    if let Some(alias) = CONCEPT_ALIASES
        .iter()
        .find(|alias| alias.concept == concept)
    {
        return alias.names.iter().map(|name| (*name).to_owned()).collect();
    }
    let Some(position_key) = position_key else {
        return Vec::new();
    };
    let Some((ic, axis, is_raw)) = position_axis(concept) else {
        return Vec::new();
    };
    let key = if is_raw {
        position_key
    } else {
        position_key.strip_suffix("_raw").unwrap_or(position_key)
    };
    vec![format!("r_{ic}_{axis}_{key}"), format!("{ic}_{axis}_{key}")]
}

pub fn resolve_concept_column<'a>(columns: &'a [String], concept: &str) -> Option<&'a str> {
    if let Some(found) = resolve_column_name(columns, concept) {
        return Some(found);
    }
    for candidate in concept_column_candidates(concept, None) {
        if let Some(found) = resolve_column_name(columns, &candidate) {
            return Some(found);
        }
    }
    None
}

pub fn resolve_requested_column<'a>(columns: &'a [String], requested: &str) -> Option<&'a str> {
    if let Some(found) = resolve_column_name(columns, requested) {
        return Some(found);
    }
    if let Some(rest) = requested.strip_prefix("r_") {
        resolve_column_name(columns, rest)
    } else {
        resolve_column_name(columns, &format!("r_{requested}"))
    }
}

/// Header names to request for field and amplifier timeslice columns.
pub fn timeslice_amplifier_field_columns() -> Vec<&'static str> {
    let mut names = vec!["layer_id", "timestamp"];
    for concept in [
        "mag_field_x",
        "mag_field_y",
        "amplifier_cmd_x",
        "amplifier_cmd_y",
        "amplifier_readback_x",
        "amplifier_readback_y",
    ] {
        if let Some(alias) = CONCEPT_ALIASES
            .iter()
            .find(|alias| alias.concept == concept)
        {
            names.extend(alias.names.iter().copied());
        }
    }
    names
}

fn position_axis(concept: &str) -> Option<(&'static str, &'static str, bool)> {
    Some(match concept {
        "ic1_x_pos_raw" => ("ic1", "x", true),
        "ic1_y_pos_raw" => ("ic1", "y", true),
        "ic2_x_pos_raw" => ("ic2", "x", true),
        "ic2_y_pos_raw" => ("ic2", "y", true),
        "ic1_x_pos" => ("ic1", "x", false),
        "ic1_y_pos" => ("ic1", "y", false),
        "ic2_x_pos" => ("ic2", "x", false),
        "ic2_y_pos" => ("ic2", "y", false),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sk_req_006_g2_alias_and_scale() {
        assert_eq!(column_scale_factor("r_ic1_current_dose"), Some(1e12));
        assert_eq!(column_scale_factor("R-IC1 Current Dose"), Some(1e12));
        assert_eq!(column_scale_factor("ic1_primary_channel"), None);
        assert_eq!(DERIVED_SUMS[0].scale, 1e12);

        let headers = vec!["r_ic1_current_dose".to_owned(), "ENERGY".to_owned()];
        assert_eq!(
            resolve_concept_column(&headers, "ic1_current"),
            Some("r_ic1_current_dose")
        );
        assert_eq!(resolve_concept_column(&headers, "energy"), Some("ENERGY"));
        assert_eq!(
            concept_column_candidates("ic1_x_pos_raw", Some(POSITION_KEY_G2_RAW)),
            vec!["r_ic1_x_spot_raw".to_owned(), "ic1_x_spot_raw".to_owned()]
        );
        assert!(concept_column_candidates("ic1_x_pos_raw", None).is_empty());
        assert!(timeslice_amplifier_field_columns().contains(&"r_tx2_probe_x"));
        assert!(column_is_integer("layer_id"));
        assert!(!column_is_integer("energy"));
    }
}
