use serde_json::Value;

use super::{finish_generated, int_param, number, selected_energies, text, Draft, ImportSpot, Row};

pub(super) fn zero_field(params: &Value) -> Result<Vec<Row>, String> {
    let energies = selected_energies(params);
    let spots = int_param(params, "spots_per_layer", 100).max(0) as usize;
    let mut draft = Vec::new();
    for (layer, energy) in energies.iter().enumerate() {
        for _ in 0..spots {
            draft.push(Draft {
                energy: *energy,
                layer: layer as i32,
                x: 0.0,
                y: 0.0,
            });
        }
    }
    finish_generated(draft, params)
}

pub(super) fn rectangular_field(params: &Value) -> Result<Vec<Row>, String> {
    let energies = selected_energies(params);
    let base = grid_positions(
        number(params, "center_x_mm", 0.0),
        number(params, "center_y_mm", 0.0),
        number(params, "field_width_mm", 100.0),
        number(params, "field_height_mm", 100.0),
        int_param(params, "spots_x", 33).max(1) as usize,
        int_param(params, "spots_y", 33).max(1) as usize,
        &text(params, "fast_axis", "x"),
        &text(params, "start_corner", "top_left"),
    );
    let transition = text(params, "layer_transition", "reset");
    let mut draft = Vec::new();
    for (layer, energy) in energies.iter().enumerate() {
        let positions = if transition == "continue" && layer % 2 == 1 {
            let mut reversed = base.clone();
            reversed.reverse();
            reversed
        } else {
            base.clone()
        };
        for (x, y) in positions {
            draft.push(Draft {
                energy: *energy,
                layer: layer as i32,
                x,
                y,
            });
        }
    }
    finish_generated(draft, params)
}

pub(super) fn sort_rows(rows: &mut [Row]) {
    rows.sort_by(|a, b| {
        b.energy
            .partial_cmp(&a.energy)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

pub(super) fn order_within_layers(
    spots: &[ImportSpot],
    order: &str,
    axis: &str,
) -> Vec<ImportSpot> {
    let mut energies = Vec::new();
    for spot in spots {
        if !energies.contains(&spot.energy) {
            energies.push(spot.energy);
        }
    }
    let mut ordered = Vec::with_capacity(spots.len());
    for energy in energies {
        let layer: Vec<ImportSpot> = spots
            .iter()
            .filter(|spot| spot.energy == energy)
            .cloned()
            .collect();
        let indices = if order == "minimize_travel" {
            serpentine_indices(
                &layer.iter().map(|spot| spot.x).collect::<Vec<_>>(),
                &layer.iter().map(|spot| spot.y).collect::<Vec<_>>(),
                axis,
            )
        } else {
            let mut indices: Vec<usize> = (0..layer.len()).collect();
            indices.sort_by_key(|&index| layer[index].plan_index);
            indices
        };
        for index in indices {
            ordered.push(layer[index].clone());
        }
    }
    ordered
}

pub(super) fn serpentine_indices(x: &[f64], y: &[f64], axis: &str) -> Vec<usize> {
    let n = x.len();
    if n <= 1 {
        return (0..n).collect();
    }
    let (slow, fast): (Vec<f64>, Vec<f64>) = if axis == "y" {
        (x.to_vec(), y.to_vec())
    } else {
        (y.to_vec(), x.to_vec())
    };
    let mut slow_order: Vec<usize> = (0..n).collect();
    slow_order.sort_by(|&a, &b| {
        slow[a]
            .partial_cmp(&slow[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let tolerance = row_tolerance(&slow, &slow_order);
    let mut rows: Vec<Vec<usize>> = Vec::new();
    let mut current = vec![slow_order[0]];
    for &index in &slow_order[1..] {
        if (slow[index] - slow[*current.last().unwrap()]).abs() <= tolerance {
            current.push(index);
        } else {
            rows.push(std::mem::take(&mut current));
            current.push(index);
        }
    }
    rows.push(current);
    let mut ordered = Vec::with_capacity(n);
    for (row_index, row) in rows.into_iter().enumerate() {
        let mut sorted = row;
        sorted.sort_by(|&a, &b| {
            fast[a]
                .partial_cmp(&fast[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if row_index % 2 == 1 {
            sorted.reverse();
        }
        ordered.extend(sorted);
    }
    ordered
}

pub(super) fn row_tolerance(slow: &[f64], order: &[usize]) -> f64 {
    let mut unique = Vec::new();
    for &index in order {
        let value = slow[index];
        if !unique.contains(&value) {
            unique.push(value);
        }
    }
    if unique.len() <= 1 {
        return 1.0;
    }
    let mut min_gap = f64::MAX;
    for pair in unique.windows(2) {
        let gap = pair[1] - pair[0];
        if gap > 0.0 && gap < min_gap {
            min_gap = gap;
        }
    }
    if min_gap.is_finite() {
        min_gap * 0.5
    } else {
        1.0
    }
}

pub(super) fn grid_positions(
    center_x: f64,
    center_y: f64,
    width: f64,
    height: f64,
    spots_x: usize,
    spots_y: usize,
    fast_axis: &str,
    start_corner: &str,
) -> Vec<(f64, f64)> {
    let xs = linspace(center_x - width / 2.0, center_x + width / 2.0, spots_x);
    let ys = linspace(center_y - height / 2.0, center_y + height / 2.0, spots_y);
    let mut positions = Vec::with_capacity(spots_x.saturating_mul(spots_y));
    if fast_axis == "y" {
        let start_low_x = start_corner == "bottom_left" || start_corner == "top_left";
        let start_y_forward = start_corner == "bottom_left" || start_corner == "bottom_right";
        let slow = if start_low_x {
            xs.clone()
        } else {
            let mut copy = xs.clone();
            copy.reverse();
            copy
        };
        for (slow_index, x) in slow.into_iter().enumerate() {
            let forward = if slow_index % 2 == 0 {
                start_y_forward
            } else {
                !start_y_forward
            };
            let y_vals = if forward {
                ys.clone()
            } else {
                let mut copy = ys.clone();
                copy.reverse();
                copy
            };
            for y in y_vals {
                positions.push((x, y));
            }
        }
        return positions;
    }
    let start_low_y = start_corner == "bottom_left" || start_corner == "bottom_right";
    let start_x_forward = start_corner == "bottom_left" || start_corner == "top_left";
    let slow = if start_low_y {
        ys.clone()
    } else {
        let mut copy = ys.clone();
        copy.reverse();
        copy
    };
    for (slow_index, y) in slow.into_iter().enumerate() {
        let forward = if slow_index % 2 == 0 {
            start_x_forward
        } else {
            !start_x_forward
        };
        let x_vals = if forward {
            xs.clone()
        } else {
            let mut copy = xs.clone();
            copy.reverse();
            copy
        };
        for x in x_vals {
            positions.push((x, y));
        }
    }
    positions
}

pub(super) fn linspace(start: f64, stop: f64, n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![start];
    }
    (0..n)
        .map(|index| start + (stop - start) * index as f64 / (n - 1) as f64)
        .collect()
}
