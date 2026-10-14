//! Window, gain, and color-scale choices for a dose picture.
//!
//! These change uniforms and the 256-pixel ramp. They do not touch the voxels.

use crate::ramp::{index, resolve, Family};

/// Slider the shell shows for the one color control.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LevelSpan {
    pub label: &'static str,
    pub min: f32,
    pub max: f32,
    pub step: f32,
    pub value: f32,
}

impl LevelSpan {
    pub fn json(self) -> String {
        format!(
            "{{\"label\":\"{}\",\"min\":\"{}\",\"max\":\"{}\",\"step\":\"{}\",\"value\":\"{}\"}}",
            self.label,
            trim_number(self.min),
            trim_number(self.max),
            trim_number(self.step),
            trim_number(self.value),
        )
    }
}

/// What the shader reads. `level` is absent for gamma, which has no slider.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Wash {
    pub lo: f32,
    pub hi: f32,
    pub gain: f32,
    pub opacity: f32,
    pub ramp: u8,
    pub mode: u8,
    pub filter: u8,
    pub level: Option<LevelSpan>,
}

/// Inputs to [`wash_of`]. Names may be catalog ids or the labels the shell stores.
/// `level` is absent when the slider has not been moved.
pub struct PaintChoice<'a> {
    pub base_lo: f32,
    pub base_hi: f32,
    pub gamma: bool,
    pub difference: bool,
    pub scale: &'a str,
    pub auto: bool,
    pub level: Option<f32>,
    pub percent: bool,
    pub ray: &'a str,
    pub sample: &'a str,
}

/// `base_lo`/`base_hi` are the data window.
pub fn wash_of(choice: PaintChoice<'_>) -> Wash {
    let PaintChoice {
        base_lo,
        base_hi,
        gamma,
        difference,
        scale,
        auto,
        level,
        percent,
        ray,
        sample,
    } = choice;
    let mut mode = match ray {
        "maximum" | "Maximum" => 1,
        "transparent" | "Transparent" => 2,
        _ => 0,
    };
    if gamma {
        mode = 1;
    }
    let filter = match sample {
        "nearest" | "Nearest" => 0,
        "cubic" | "Cubic" => 2,
        _ => 1,
    };
    let ramp = if gamma {
        index("gamma")
    } else if difference {
        resolve(scale, Family::Divergent)
    } else {
        resolve(scale, Family::Sequential)
    };
    if gamma {
        return Wash {
            lo: 0.0,
            hi: 2.0,
            gain: 1.0,
            opacity: 1.0,
            ramp,
            mode,
            filter,
            level: None,
        };
    }
    if mode == 2 {
        let opacity = level.unwrap_or(1.0).clamp(0.0, 1.0);
        return Wash {
            lo: base_lo,
            hi: base_hi,
            gain: opacity,
            opacity,
            ramp,
            mode,
            filter,
            level: Some(slider("Opacity", 0.0, 1.0, 0.05, 1.0, level)),
        };
    }
    if difference {
        let reach = if percent {
            base_hi.abs().max(1e-6) * level.unwrap_or(10.0).clamp(0.1, 100.0) / 100.0
        } else {
            level.unwrap_or(0.2).abs().max(1e-6)
        };
        let span = if percent {
            slider("Percent", 1.0, 100.0, 1.0, 10.0, level)
        } else {
            slider("Full scale", 0.01, 5.0, 0.01, 0.2, level)
        };
        return Wash {
            lo: -reach,
            hi: reach,
            gain: 1.0,
            opacity: 1.0,
            ramp,
            mode,
            filter,
            level: Some(span),
        };
    }
    if auto {
        let gain = level.unwrap_or(1.0).clamp(0.05, 8.0);
        return Wash {
            lo: base_lo,
            hi: base_hi,
            gain,
            opacity: 1.0,
            ramp,
            mode,
            filter,
            level: Some(slider("Gain", 0.25, 4.0, 0.05, 1.0, level)),
        };
    }
    let data = base_hi.abs().max(base_lo.abs()).max(1.0);
    let limit = data * 2.0;
    let hi = level.unwrap_or(base_hi).clamp(1e-6, limit);
    Wash {
        lo: 0.0,
        hi,
        gain: 1.0,
        opacity: 1.0,
        ramp,
        mode,
        filter,
        level: Some(LevelSpan {
            label: "Window",
            min: 0.0,
            max: limit,
            step: (data / 50.0).max(0.01),
            value: hi,
        }),
    }
}

/// Color-bar unit. Dose switches between line integral and a point sample.
/// Gamma, MU, and protons stay as they are.
pub fn painted_unit(unit: &str, mode: u8, gamma: bool) -> String {
    if gamma || (unit != "Gy" && unit != "Gy·mm") {
        return unit.to_string();
    }
    if mode == 0 {
        "Gy·mm".to_string()
    } else {
        "Gy".to_string()
    }
}

pub fn trim_number(value: f32) -> String {
    if (value - value.round()).abs() < 1e-4 && value.abs() < 1.0e6 {
        format!("{}", value.round() as i32)
    } else {
        let text = format!("{value:.4}");
        text.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

fn slider(
    label: &'static str,
    min: f32,
    max: f32,
    step: f32,
    fallback: f32,
    level: Option<f32>,
) -> LevelSpan {
    LevelSpan {
        label,
        min,
        max,
        step,
        value: level.unwrap_or(fallback).clamp(min, max),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gain_and_window_share_the_data_span() {
        let gain = wash_of(PaintChoice {
            base_lo: 0.0,
            base_hi: 4.0,
            gamma: false,
            difference: false,
            scale: "turbo",
            auto: true,
            level: Some(2.0),
            percent: false,
            ray: "integrate",
            sample: "linear",
        });
        assert!((gain.lo - 0.0).abs() < 1e-6);
        assert!((gain.hi - 4.0).abs() < 1e-6);
        assert!((gain.gain - 2.0).abs() < 1e-6);
        assert_eq!(gain.level.unwrap().label, "Gain");

        let window = wash_of(PaintChoice {
            base_lo: 0.0,
            base_hi: 4.0,
            gamma: false,
            difference: false,
            scale: "Viridis",
            auto: false,
            level: Some(1.5),
            percent: false,
            ray: "Maximum",
            sample: "Nearest",
        });
        assert!((window.hi - 1.5).abs() < 1e-6);
        assert!((window.gain - 1.0).abs() < 1e-6);
        assert_eq!(window.mode, 1);
        assert_eq!(window.filter, 0);
        assert_eq!(window.ramp, crate::ramp::index("viridis"));
        let span = window.level.unwrap();
        assert_eq!(span.label, "Window");
        assert!((span.max - 8.0).abs() < 1e-6);
        assert!((span.value - 1.5).abs() < 1e-6);
        let again = wash_of(PaintChoice {
            base_lo: 0.0,
            base_hi: 4.0,
            gamma: false,
            difference: false,
            scale: "turbo",
            auto: false,
            level: Some(3.0),
            percent: false,
            ray: "integrate",
            sample: "linear",
        });
        assert!((again.level.unwrap().max - span.max).abs() < 1e-6);
    }

    #[test]
    fn gamma_has_no_slider_and_difference_uses_percent() {
        let gamma = wash_of(PaintChoice {
            base_lo: 0.0,
            base_hi: 9.0,
            gamma: true,
            difference: false,
            scale: "turbo",
            auto: true,
            level: Some(3.0),
            percent: false,
            ray: "integrate",
            sample: "linear",
        });
        assert!(gamma.level.is_none());
        assert_eq!(gamma.mode, 1);
        assert!((gamma.hi - 2.0).abs() < 1e-6);
        let diff = wash_of(PaintChoice {
            base_lo: -2.0,
            base_hi: 2.0,
            gamma: false,
            difference: true,
            scale: "managua",
            auto: true,
            level: Some(10.0),
            percent: true,
            ray: "integrate",
            sample: "cubic",
        });
        assert!((diff.hi - 0.2).abs() < 1e-4);
        assert!((diff.lo + 0.2).abs() < 1e-4);
        assert_eq!(diff.filter, 2);
        assert_eq!(painted_unit("Gy·mm", 1, false), "Gy");
        assert_eq!(painted_unit("γ", 0, true), "γ");
        assert_eq!(painted_unit("MU/mm³", 1, false), "MU/mm³");
    }
}
