//! Color scales for every heatmap.
//!
//! Each named map is 256 RGB samples of the matplotlib colormap used by the
//! Python dose view, in this order: viridis, turbo, magma, inferno, plasma,
//! cividis, YlGnBu_r (deep), cubehelix, afmhot (heat), gray, managua_r,
//! berlin, coolwarm, RdYlBu_r, Spectral_r, PuOr_r, then the gamma step.
//! A session overlay is not a row in that table. It darkens the series color
//! and raises alpha, so several sessions can share one pixel.

const SAMPLES: &[u8] = include_bytes!("ramp_samples.bin");
const COUNT: usize = 256;

const NAMES: &[&str] = &[
    "viridis",
    "turbo",
    "magma",
    "inferno",
    "plasma",
    "cividis",
    "deep",
    "cubehelix",
    "heat",
    "gray",
    "managua",
    "berlin",
    "coolwarm",
    "rdylbu",
    "spectral",
    "puor",
    "gamma",
];

const SEQUENTIAL: &[(&str, &str)] = &[
    ("turbo", "Turbo"),
    ("viridis", "Viridis"),
    ("magma", "Magma"),
    ("inferno", "Inferno"),
    ("plasma", "Plasma"),
    ("cividis", "Cividis"),
    ("deep", "Deep"),
    ("cubehelix", "Cubehelix"),
    ("heat", "Heat"),
    ("gray", "Gray"),
];

const DIVERGENT: &[(&str, &str)] = &[
    ("managua", "Managua"),
    ("berlin", "Berlin"),
    ("coolwarm", "Coolwarm"),
    ("rdylbu", "RdYlBu"),
    ("spectral", "Spectral"),
    ("puor", "PuOr"),
];

/// Catalog index of viridis. A missing heatmap ramp deserializes as this.
pub const VIRIDIS: u8 = 0;
/// Several sessions. The series color is the ink.
pub const SESSION: u8 = NAMES.len() as u8;

const _: () = assert!(SAMPLES.len() == NAMES.len() * COUNT * 3);

/// Sequential dose and density maps, or divergent difference maps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    Sequential,
    Divergent,
}

/// Names and labels for one family, in menu order.
pub fn choices(family: Family) -> &'static [(&'static str, &'static str)] {
    match family {
        Family::Sequential => SEQUENTIAL,
        Family::Divergent => DIVERGENT,
    }
}

/// Catalog index for a ramp name. An unknown name is turbo.
pub fn index(name: &str) -> u8 {
    if name == "session" {
        return SESSION;
    }
    NAMES
        .iter()
        .position(|id| *id == name)
        .map(|slot| slot as u8)
        .unwrap_or(1)
}

/// `true` when the heatmap should ink itself with the series color.
pub fn is_session(slot: u8) -> bool {
    slot == SESSION
}

/// The ramp to draw. A name from the other family falls back to that family's default.
pub fn resolve(name: &str, family: Family) -> u8 {
    let choices = choices(family);
    if let Some((id, _)) = choices
        .iter()
        .find(|(id, label)| *id == name || *label == name)
    {
        return index(id);
    }
    match family {
        Family::Sequential => index("turbo"),
        Family::Divergent => index("managua"),
    }
}

/// RGB at `t` in 0..=1. A session index falls back to turbo; use [`session`] for that.
pub fn sample(slot: u8, t: f32) -> [f32; 3] {
    let slot = if (slot as usize) < NAMES.len() {
        slot as usize
    } else {
        1
    };
    let x = t.clamp(0.0, 1.0) * (COUNT - 1) as f32;
    let i = x.floor() as usize;
    let f = x - i as f32;
    let a = rgb_at(slot, i);
    let b = rgb_at(slot, (i + 1).min(COUNT - 1));
    [
        a[0] + (b[0] - a[0]) * f,
        a[1] + (b[1] - a[1]) * f,
        a[2] + (b[2] - a[2]) * f,
    ]
}

/// Session ink. `t` = 0 is clear, `t` = 1 is `tint`, and the middle is a darker shade.
pub fn session(tint: [f32; 3], t: f32) -> [f32; 4] {
    let u = t.clamp(0.0, 1.0);
    let shaped = u * u * (3.0 - 2.0 * u);
    let scale = 0.35 + 0.65 * shaped;
    [tint[0] * scale, tint[1] * scale, tint[2] * scale, shaped]
}

fn rgb_at(slot: usize, sample_index: usize) -> [f32; 3] {
    let base = (slot * COUNT + sample_index) * 3;
    [
        SAMPLES[base] as f32 / 255.0,
        SAMPLES[base + 1] as f32 / 255.0,
        SAMPLES[base + 2] as f32 / 255.0,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn luma(rgb: [f32; 3]) -> f32 {
        0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    }

    fn chroma(rgb: [f32; 3]) -> f32 {
        let max = rgb[0].max(rgb[1]).max(rgb[2]);
        let min = rgb[0].min(rgb[1]).min(rgb[2]);
        max - min
    }

    fn apart(a: [f32; 3], b: [f32; 3]) -> f32 {
        let d0 = a[0] - b[0];
        let d1 = a[1] - b[1];
        let d2 = a[2] - b[2];
        (d0 * d0 + d1 * d1 + d2 * d2).sqrt()
    }

    #[test]
    fn sequential_maps_brighten_and_turbo_runs_blue_to_red() {
        for (id, _) in choices(Family::Sequential) {
            let start = sample(index(id), 0.0);
            let end = sample(index(id), 1.0);
            if *id == "turbo" {
                assert!(start[2] > start[0], "{start:?}");
                assert!(end[0] > end[2], "{end:?}");
            } else {
                assert!(
                    luma(end) > luma(start) + 0.4,
                    "{id} {} {}",
                    luma(start),
                    luma(end)
                );
            }
        }
    }

    #[test]
    fn divergent_ends_differ_and_the_center_is_the_neutral() {
        for (id, _) in choices(Family::Divergent) {
            let start = sample(index(id), 0.0);
            let mid = sample(index(id), 0.5);
            let end = sample(index(id), 1.0);
            assert!(apart(start, end) > 0.4, "{id}");
            assert!(apart(mid, start) > 0.15 && apart(mid, end) > 0.15, "{id}");
        }
        assert!(chroma(sample(index("coolwarm"), 0.5)) < 0.08);
        assert!(chroma(sample(index("puor"), 0.5)) < 0.08);
        for id in ["managua", "berlin"] {
            let mid = luma(sample(index(id), 0.5));
            assert!(mid < luma(sample(index(id), 0.0)));
            assert!(mid < luma(sample(index(id), 1.0)));
        }
    }

    #[test]
    fn gamma_steps_at_one_and_starts_dark() {
        let under = sample(index("gamma"), 0.49);
        let over = sample(index("gamma"), 0.51);
        assert!(under[1] > under[0]);
        assert!(over[0] > under[0] + 0.4);
        assert!(luma(sample(index("gamma"), 0.0)) < 0.15);
    }

    #[test]
    fn a_session_ramp_is_clear_then_the_tint() {
        let tint = [0.2, 0.4, 0.8];
        let clear = session(tint, 0.0);
        let ink = session(tint, 1.0);
        let mid = session(tint, 0.5);
        assert!(clear[3] < 1.0e-5);
        assert!((ink[0] - tint[0]).abs() < 1.0e-5);
        assert!((ink[1] - tint[1]).abs() < 1.0e-5);
        assert!((ink[2] - tint[2]).abs() < 1.0e-5);
        assert!((ink[3] - 1.0).abs() < 1.0e-5);
        assert!(mid[0] < tint[0]);
        assert!(mid[3] > 0.2 && mid[3] < 0.9);
        assert!(is_session(index("session")));
        assert!(!is_session(index("turbo")));
    }

    #[test]
    fn a_difference_cannot_keep_a_sequential_map() {
        assert_eq!(resolve("turbo", Family::Divergent), index("managua"));
        assert_eq!(resolve("Managua", Family::Divergent), index("managua"));
        assert_eq!(resolve("nope", Family::Sequential), index("turbo"));
        assert_eq!(resolve("Viridis", Family::Sequential), index("viridis"));
        assert_eq!(index("viridis"), VIRIDIS);
        assert_eq!(choices(Family::Sequential).len(), 10);
        assert_eq!(choices(Family::Divergent)[0].0, "managua");
    }
}
