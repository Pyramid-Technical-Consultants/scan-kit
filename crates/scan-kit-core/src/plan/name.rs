use serde_json::Value;

use super::{
    bool_param, int_param, number, selected_energies, text, trim_zeros, STANDARD_ENERGIES_MEV,
};

pub(super) fn suggest_filename(
    template: &str,
    params: &Value,
    dicom_label: Option<&str>,
    max_length: usize,
) -> String {
    if template == "iba_pld_plan" {
        let stem = sanitize(&path_stem(&text(params, "pld_path", "")));
        return limit_name(&[stem], max_length);
    }
    let slug = match template {
        "dicom_rt_plan" => "DicomPlan".to_owned(),
        "zero_field" => "ZeroField".to_owned(),
        "rectangular_field" => "RectField".to_owned(),
        _ => sanitize(template),
    };
    let energy = energy_part(template, params);
    let geometry = geometry_part(template, params, dicom_label);
    let weight = weight_part(template, params);
    limit_name(&[slug, energy, geometry, weight], max_length)
}

pub(super) fn energy_part(template: &str, params: &Value) -> String {
    if template == "dicom_rt_plan" || template == "iba_pld_plan" {
        return String::new();
    }
    let energies = selected_energies(params);
    if energies.is_empty() {
        return "E0L".into();
    }
    if energies.len() == STANDARD_ENERGIES_MEV.len()
        && energies
            .iter()
            .all(|energy| STANDARD_ENERGIES_MEV.contains(energy))
    {
        return format!(
            "E{:.0}-{:.0}",
            STANDARD_ENERGIES_MEV[STANDARD_ENERGIES_MEV.len() - 1],
            STANDARD_ENERGIES_MEV[0]
        );
    }
    if energies.len() == 1 {
        return format!("E{}", num(energies[0], 3));
    }
    let mut indices = Vec::new();
    for energy in &energies {
        if let Some(index) = STANDARD_ENERGIES_MEV
            .iter()
            .position(|have| *have == *energy)
        {
            indices.push(index);
        }
    }
    if indices.len() == energies.len() {
        let lo = *indices.iter().min().unwrap_or(&0);
        let hi = *indices.iter().max().unwrap_or(&0);
        if hi - lo + 1 == energies.len() {
            return format!(
                "E{}-{}",
                num(STANDARD_ENERGIES_MEV[hi], 3),
                num(STANDARD_ENERGIES_MEV[lo], 3)
            );
        }
    }
    if energies.len() <= 4 {
        return format!(
            "E{}",
            energies
                .iter()
                .map(|energy| num(*energy, 3))
                .collect::<Vec<_>>()
                .join("-")
        );
    }
    format!(
        "E{}L_{}-{}",
        energies.len(),
        num(energies[0], 3),
        num(*energies.last().unwrap_or(&0.0), 3)
    )
}

pub(super) fn geometry_part(template: &str, params: &Value, dicom_label: Option<&str>) -> String {
    match template {
        "dicom_rt_plan" => {
            if let Some(label) = dicom_label.filter(|label| !label.is_empty()) {
                sanitize(label)
            } else {
                sanitize(&path_stem(&text(params, "dicom_path", "")))
            }
        }
        "zero_field" => format!("Sp{}", int_param(params, "spots_per_layer", 0)),
        "rectangular_field" => {
            let mut segments = Vec::new();
            let center_x = number(params, "center_x_mm", 0.0);
            let center_y = number(params, "center_y_mm", 0.0);
            if center_x != 0.0 || center_y != 0.0 {
                segments.push(format!("C{}x{}", num(center_x, 3), num(center_y, 3)));
            }
            segments.push(format!(
                "{}x{}mm",
                num(number(params, "field_width_mm", 0.0), 3),
                num(number(params, "field_height_mm", 0.0), 3)
            ));
            segments.push(format!(
                "G{}x{}",
                int_param(params, "spots_x", 0),
                int_param(params, "spots_y", 0)
            ));
            segments.join("_")
        }
        _ => String::new(),
    }
}

pub(super) fn weight_part(template: &str, params: &Value) -> String {
    if template == "dicom_rt_plan" || template == "iba_pld_plan" {
        return String::new();
    }
    match text(params, "spot_weight_method", "fixed").as_str() {
        "fixed" => format!("Wfix{}", num(number(params, "spot_weight_mu", 0.0), 4)),
        "random_range" => format!(
            "Wrng{}-{}",
            num(number(params, "spot_weight_min_mu", 0.0), 4),
            num(number(params, "spot_weight_max_mu", 0.0), 4)
        ),
        "layer_even_range" => {
            let prefix = if bool_param(params, "spot_weight_layer_shuffle", false) {
                "Wlyrs"
            } else {
                "Wlyr"
            };
            format!(
                "{prefix}{}-{}",
                num(number(params, "spot_weight_min_mu", 0.0), 4),
                num(number(params, "spot_weight_max_mu", 0.0), 4)
            )
        }
        "even_total" => format!(
            "Wtot{}",
            num(number(params, "spot_weight_total_mu", 0.0), 4)
        ),
        "random_total_variance" => format!(
            "Wtot{}v{}",
            num(number(params, "spot_weight_total_mu", 0.0), 4),
            num(number(params, "spot_weight_variance_pct", 0.0), 1)
        ),
        _ => "W".into(),
    }
}

pub(super) fn num(value: f64, decimals: usize) -> String {
    let text = trim_zeros(&format!("{value:.decimals$}"));
    if text.is_empty() {
        "0".into()
    } else {
        text
    }
}

pub(super) fn path_stem(path: &str) -> String {
    std::path::Path::new(path)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("")
        .to_owned()
}

pub(super) fn sanitize(text: &str) -> String {
    text.chars()
        .filter(|ch| {
            !matches!(ch, '<' | '>' | ':' | '"' | '/' | '\\' | '|' | '?' | '*') && !ch.is_control()
        })
        .map(|ch| if ch == ' ' { '_' } else { ch })
        .collect()
}

pub(super) fn limit_name(parts: &[String], max_length: usize) -> String {
    let suffix = ".csv";
    let max_stem = max_length.saturating_sub(suffix.len()).max(1);
    let join = |parts: &[String]| {
        sanitize(
            &parts
                .iter()
                .filter(|part| !part.is_empty())
                .cloned()
                .collect::<Vec<_>>()
                .join("_"),
        )
    };
    let mut compact = parts.to_vec();
    let mut stem = join(&compact);
    if stem.is_empty() {
        stem = "input_map".into();
    }
    if stem.len() <= max_stem {
        return format!("{stem}{suffix}");
    }
    if compact.len() > 1 {
        compact[1] = energy_compact(&compact[1]);
        stem = join(&compact);
        if stem.len() <= max_stem {
            return format!("{stem}{suffix}");
        }
    }
    if compact.len() > 2 && !compact[2].is_empty() {
        compact[2] = geometry_compact(&compact[2]);
        stem = join(&compact);
        if stem.len() <= max_stem {
            return format!("{stem}{suffix}");
        }
    }
    let minimal = join(&compact[..compact.len().min(2)]);
    if minimal.len() <= max_stem {
        return format!("{minimal}{suffix}");
    }
    let cut = minimal.chars().take(max_stem).collect::<String>();
    format!("{cut}{suffix}")
}

pub(super) fn energy_compact(part: &str) -> String {
    if let Some(rest) = part.strip_prefix('E') {
        if let Some((count, _)) = rest.split_once("L_") {
            if count.chars().all(|ch| ch.is_ascii_digit()) {
                return format!("E{count}L");
            }
        }
    }
    part.to_owned()
}

pub(super) fn geometry_compact(part: &str) -> String {
    if part.starts_with("Sp") {
        return part.to_owned();
    }
    part.split('_')
        .find(|segment| segment.starts_with('G'))
        .unwrap_or_else(|| part.split('_').next_back().unwrap_or(part))
        .to_owned()
}
