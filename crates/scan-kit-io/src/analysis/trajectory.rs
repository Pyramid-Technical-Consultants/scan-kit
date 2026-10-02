use std::path::Path;

use scan_kit_core::{
    fit_iso_plane, magnet_pivot_z, Control, PlotScene, Series, IC1_Z_MM, IC2_Z_MM,
};
use serde_json::Value;

use super::{col, drew_line, finite_col, guide, placed, scene, span, spot_table, stroke};

pub(super) fn trajectory(root: &Path, session_ids: &[String], options: &Value) -> PlotScene {
    let azimuth = number_option(options, "azimuth", 0.4);
    let orbit = ["0.2", "0.4", "0.8", "1.2"]
        .into_iter()
        .find(|label| *label == format!("{azimuth:.1}"))
        .unwrap_or("0.4");
    let azimuth: f32 = orbit.parse().unwrap_or(0.4);
    let mut paths = Vec::new();
    let mut iso = Vec::new();
    let mut pivot = 0.0f32;
    let mut have_pivot = false;
    for session in session_ids {
        let table = spot_table(root, session);
        let plan_x = col(&table, "plan_x").unwrap_or(&[]);
        let plan_y = col(&table, "plan_y").unwrap_or(&[]);
        let ic1_x = finite_col(&table, "ic1_x").unwrap_or(plan_x);
        let ic1_y = finite_col(&table, "ic1_y").unwrap_or(plan_y);
        let ic2_x = finite_col(&table, "ic2_x").unwrap_or(plan_x);
        let ic2_y = finite_col(&table, "ic2_y").unwrap_or(plan_y);
        let energy = col(&table, "energy").unwrap_or(&[]);
        let n = ic1_x
            .len()
            .min(ic2_x.len())
            .min(ic1_y.len())
            .min(ic2_y.len());
        let mut sx = Vec::new();
        let mut sy = Vec::new();
        for i in 0..n.min(400) {
            let (px, py) = project(ic2_x[i], ic2_y[i], IC2_Z_MM, azimuth);
            let (qx, qy) = project(ic1_x[i], ic1_y[i], IC1_Z_MM, azimuth);
            sx.extend([px, qx, f32::NAN]);
            sy.extend([py, qy, f32::NAN]);
        }
        if sx.iter().any(|value| value.is_finite()) {
            if !have_pivot {
                pivot = magnet_pivot_z(
                    ic2_x.first().copied().unwrap_or(0.0),
                    ic1_x.first().copied().unwrap_or(0.0),
                );
                have_pivot = true;
            }
            paths.push(stroke(sx, sy, false));
        }
        let fit_x = if finite_col(&table, "ic1_x").is_some() {
            ic1_x
        } else {
            plan_x
        };
        if let Some((intercept, slope)) = fit_iso_plane(energy, fit_x) {
            let (e0, e1) = span(energy);
            iso.push(stroke(
                vec![e0, e1],
                vec![intercept + slope * e0, intercept + slope * e1],
                false,
            ));
        }
    }
    if have_pivot {
        paths.push(plane_guide(
            (-40.0, 0.0, IC2_Z_MM),
            (40.0, 0.0, IC2_Z_MM),
            azimuth,
        ));
        paths.push(plane_guide(
            (-40.0, 0.0, IC1_Z_MM),
            (40.0, 0.0, IC1_Z_MM),
            azimuth,
        ));
        paths.push(plane_guide(
            (-20.0, -4.0, pivot),
            (20.0, -4.0, pivot),
            azimuth,
        ));
        paths.push(plane_guide(
            (-20.0, 4.0, pivot),
            (20.0, 4.0, pivot),
            azimuth,
        ));
    }
    let mut panels = Vec::new();
    if !paths.is_empty() {
        panels.push(placed(format!("Planes Pivot {pivot:.0} mm"), paths));
    }
    if drew_line(&iso) {
        panels.push(placed("Iso".into(), iso));
    }
    scene(
        "IC Beam Trajectory",
        panels,
        vec![control(
            "azimuth",
            "Orbit",
            &["0.2", "0.4", "0.8", "1.2"],
            orbit,
        )],
    )
}

fn control(id: &str, label: &str, options: &[&str], value: &str) -> Control {
    Control {
        id: id.into(),
        label: label.into(),
        options: options.iter().map(|option| (*option).to_owned()).collect(),
        value: value.into(),
    }
}

fn plane_guide(from: (f32, f32, f32), to: (f32, f32, f32), azimuth: f32) -> Series {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    push_projected(&mut xs, &mut ys, from, to, azimuth);
    guide(xs, ys)
}

fn project(x: f32, y: f32, z: f32, azimuth: f32) -> (f32, f32) {
    let px = x * azimuth.cos() - z * azimuth.sin();
    let py = y + z * 0.15;
    (px, py)
}

fn number_option(options: &Value, key: &str, default: f32) -> f32 {
    options
        .get(key)
        .and_then(|value| {
            value
                .as_f64()
                .or_else(|| value.as_str().and_then(|text| text.parse().ok()))
        })
        .map(|value| value as f32)
        .unwrap_or(default)
}

fn push_projected(
    xs: &mut Vec<f32>,
    ys: &mut Vec<f32>,
    from: (f32, f32, f32),
    to: (f32, f32, f32),
    azimuth: f32,
) {
    if xs.iter().any(|value| value.is_finite()) {
        xs.push(f32::NAN);
        ys.push(f32::NAN);
    }
    let (a, b) = project(from.0, from.1, from.2, azimuth);
    let (c, d) = project(to.0, to.1, to.2, azimuth);
    xs.extend([a, c]);
    ys.extend([b, d]);
}
