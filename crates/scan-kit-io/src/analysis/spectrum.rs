use std::path::Path;

use scan_kit_core::{welch_psd, PlotScene};
use serde_json::Value;

use super::{finite_names, placed, scene, stroke, timeline};

pub(super) fn fft_view(root: &Path, session_ids: &[String], options: &Value) -> PlotScene {
    let tables = timeline(root, session_ids);
    let owned = finite_names(session_ids, &tables);
    let headers: Vec<crate::source::SessionCols<'_>> = owned
        .iter()
        .map(|(name, columns)| crate::source::SessionCols { name, columns })
        .collect();
    let picked = crate::source::select(crate::source::Shape::Y, false, true, &headers, options);
    let channel = picked.y.clone();
    let mut series = Vec::new();
    for columns in &tables {
        let Some(samples) = columns.get(&channel) else {
            continue;
        };
        let (freqs, psd) = welch_psd(samples, 1000.0, 4096, 0.5);
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for (freq, power) in freqs.iter().zip(&psd) {
            if *freq >= 1.0 && *freq <= 500.0 {
                xs.push(*freq);
                // A linear PSD sits on the axis. Decades match the Python explorer.
                ys.push(if power.is_finite() && *power > 0.0 {
                    power.log10()
                } else {
                    f32::NAN
                });
            }
        }
        if ys.iter().any(|value| value.is_finite()) {
            series.push(stroke(xs, ys, false));
        }
    }
    let mut panels = if series.is_empty() {
        Vec::new()
    } else {
        vec![placed("Spectrum".into(), series)]
    };
    if let Some(panel) = panels.first_mut() {
        panel.y_label = "log10 PSD".into();
    }
    scene("FFT Explorer", panels, picked.controls)
}
