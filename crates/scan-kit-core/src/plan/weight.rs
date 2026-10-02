use serde_json::Value;

use super::field::linspace;
use super::{bool_param, number, text, Draft, WEIGHT_SCALE};

pub(super) fn spot_weights(rows: &[Draft], params: &Value) -> Result<Vec<f64>, String> {
    let method = text(params, "spot_weight_method", "fixed");
    match method.as_str() {
        "fixed" => {
            let weight = round4(number(params, "spot_weight_mu", 0.02));
            Ok(vec![weight; rows.len()])
        }
        "random_range" => {
            let min_mu = number(params, "spot_weight_min_mu", 0.002);
            let max_mu = number(params, "spot_weight_max_mu", 0.1);
            let mut rng = Rng::new(mix_seed(rows.len(), min_mu, max_mu));
            Ok((0..rows.len())
                .map(|_| round4(rng.uniform(min_mu, max_mu)))
                .collect())
        }
        "layer_even_range" => {
            let min_mu = number(params, "spot_weight_min_mu", 0.002);
            let max_mu = number(params, "spot_weight_max_mu", 0.1);
            let shuffle = bool_param(params, "spot_weight_layer_shuffle", false);
            Ok(layer_even(rows, min_mu, max_mu, shuffle))
        }
        "even_total" => even_total(rows.len(), number(params, "spot_weight_total_mu", 1.0)),
        "random_total_variance" => random_total(
            rows.len(),
            number(params, "spot_weight_total_mu", 1.0),
            number(params, "spot_weight_variance_pct", 10.0),
        ),
        _ => Err("Select a spot weight method.".into()),
    }
}

pub(super) fn layer_even(rows: &[Draft], min_mu: f64, max_mu: f64, shuffle: bool) -> Vec<f64> {
    let mut weights = vec![0.0; rows.len()];
    let mut layers = Vec::new();
    for row in rows {
        if !layers.contains(&row.layer) {
            layers.push(row.layer);
        }
    }
    for layer in layers {
        let indices: Vec<usize> = rows
            .iter()
            .enumerate()
            .filter(|(_, row)| row.layer == layer)
            .map(|(index, _)| index)
            .collect();
        let mut values = linspace(min_mu, max_mu, indices.len())
            .into_iter()
            .map(round4)
            .collect::<Vec<_>>();
        if shuffle && values.len() > 1 {
            let mut rng = Rng::new(layer as u64 + 7);
            rng.shuffle(&mut values);
        }
        for (index, value) in indices.into_iter().zip(values) {
            weights[index] = value;
        }
    }
    weights
}

pub(super) fn even_total(n: usize, target: f64) -> Result<Vec<f64>, String> {
    if n == 0 {
        return Ok(Vec::new());
    }
    let mut weights = vec![round4(target / n as f64); n];
    apply_remainder(&mut weights, target);
    if weights.last().copied().unwrap_or(0.0) <= 0.0 {
        return Err(format!(
            "Target Total Weight (MU) is too small to assign a positive weight to each of {n} spots."
        ));
    }
    Ok(weights)
}

pub(super) fn random_total(n: usize, target: f64, variance_pct: f64) -> Result<Vec<f64>, String> {
    if n == 0 {
        return Ok(Vec::new());
    }
    if variance_pct <= 0.0 {
        return even_total(n, target);
    }
    let base = target / n as f64;
    let spread = variance_pct / 100.0;
    let mut rng = Rng::new(mix_seed(n, target, variance_pct));
    let raw: Vec<f64> = (0..n)
        .map(|_| base * (1.0 + rng.uniform(-spread, spread)))
        .collect();
    let scaled_total: f64 = raw.iter().sum();
    if scaled_total <= 0.0 {
        return Err("Spot Variance (%) is too large to assign positive spot weights.".into());
    }
    let scale = target / scaled_total;
    let mut weights: Vec<f64> = raw.into_iter().map(|value| round4(value * scale)).collect();
    apply_remainder(&mut weights, target);
    if weights.last().copied().unwrap_or(0.0) <= 0.0 {
        return Err(format!(
            "Target Total Weight (MU) is too small to assign a positive weight to each of {n} spots at {variance_pct}% variance."
        ));
    }
    Ok(weights)
}

pub(super) fn apply_remainder(weights: &mut [f64], target: f64) {
    if weights.is_empty() {
        return;
    }
    let remainder = round4(target - weights.iter().sum::<f64>());
    if remainder != 0.0 {
        let last = weights.len() - 1;
        weights[last] = round4(weights[last] + remainder);
    }
}

pub(super) fn round4(value: f64) -> f64 {
    (value * WEIGHT_SCALE).round() / WEIGHT_SCALE
}

pub(super) struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn uniform(&mut self, min: f64, max: f64) -> f64 {
        let unit = (self.next() >> 11) as f64 / ((1u64 << 53) as f64);
        min + (max - min) * unit
    }

    fn shuffle(&mut self, values: &mut [f64]) {
        for index in (1..values.len()).rev() {
            let swap = (self.next() as usize) % (index + 1);
            values.swap(index, swap);
        }
    }
}

pub(super) fn mix_seed(n: usize, a: f64, b: f64) -> u64 {
    let bits = a.to_bits() ^ b.to_bits().rotate_left(17) ^ (n as u64).wrapping_mul(0x9E37_79B9);
    let tick = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos() as u64)
        .unwrap_or(1);
    bits ^ tick
}
