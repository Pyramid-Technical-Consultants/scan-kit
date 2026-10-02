use std::path::Path;

use scan_kit_core::{welch_psd, PlotScene};
use serde_json::Value;

use super::{channel_pairs_of, choice_control, choose, placed, scene, stroke, timeline};

pub(super) fn fft_view(root: &Path, session_ids: &[String], options: &Value) -> PlotScene {
    spectrum(root, session_ids, options, false)
}

pub(super) fn audio_view(root: &Path, session_ids: &[String], options: &Value) -> PlotScene {
    spectrum(root, session_ids, options, true)
}

fn spectrum(root: &Path, session_ids: &[String], options: &Value, audio: bool) -> PlotScene {
    let tables = timeline(root, session_ids);
    let pairs = channel_pairs_of(&tables);
    let channel = choose(options, "channel", "ic1_current", &pairs);
    let mut series = Vec::new();
    let mut played = Vec::new();
    for (index, columns) in tables.iter().enumerate() {
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
        if audio && index == 0 {
            played = audible(samples);
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
    let mut scene = scene(
        if audio {
            "Audio Explorer"
        } else {
            "FFT Explorer"
        },
        panels,
        vec![choice_control("channel", "Channel", &pairs, &channel)],
    );
    scene.samples = played;
    scene
}

fn audible(samples: &[f32]) -> Vec<f32> {
    let peak = samples
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .fold(0.0f32, |max, value| max.max(value.abs()))
        .max(1e-6);
    samples
        .iter()
        .take(8000)
        .map(|value| (value / peak).clamp(-1.0, 1.0))
        .collect()
}
