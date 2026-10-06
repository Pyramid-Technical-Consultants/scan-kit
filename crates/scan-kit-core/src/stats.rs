//! Column reductions.
//!
//! Min, max, and mean are one pass. Median and percentile select the ranks they
//! need, so a full sort is not required. Finite values keep the same order
//! statistics as a sort. The playhead fits a window of about a thousand samples
//! with these functions before it draws the frame.
//!
//! A release run of `stats_cpu_vs_gpu` in `scan-kit-compute` timed a warmed
//! compute reduction, including the upload and the 16-byte readback, against
//! [`reduce_finite`]. Pipeline creation stayed outside the loop. Best of five:
//! 1e3 was 1 us on the CPU and 122 us on the GPU, 1e5 was 154 us and 205 us,
//! 1e6 was 1633 us and 987 us, and 1e7 was 16760 us and 8308 us. The GPU is
//! slower through 1e5 and about twice as fast from 1e6. Loaded timeslice
//! columns are under that, and the playhead window is about 1e3, so this
//! module stays on the CPU. Median and percentile stay here too: they are
//! order statistics, not an associative reduction.

/// Min, max, sum, and count of the finite samples.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FiniteReduce {
    pub min: f32,
    pub max: f32,
    pub sum: f32,
    pub count: u32,
}

/// One pass. An empty or all-non-finite slice has `count == 0` and NaN ends.
pub fn reduce_finite(samples: &[f32]) -> FiniteReduce {
    let mut out = FiniteReduce {
        min: f32::NAN,
        max: f32::NAN,
        sum: 0.0,
        count: 0,
    };
    for value in samples.iter().copied() {
        if !value.is_finite() {
            continue;
        }
        if out.count == 0 {
            out.min = value;
            out.max = value;
        } else {
            out.min = out.min.min(value);
            out.max = out.max.max(value);
        }
        out.sum += value;
        out.count += 1;
    }
    out
}

/// Finite min and max. `None` when no sample is finite.
pub fn finite_minmax(samples: &[f32]) -> Option<(f32, f32)> {
    let reduced = reduce_finite(samples);
    (reduced.count > 0).then_some((reduced.min, reduced.max))
}

/// Mean of the finite samples. `None` when no sample is finite.
pub fn mean_finite(samples: &[f32]) -> Option<f32> {
    let reduced = reduce_finite(samples);
    (reduced.count > 0).then_some(reduced.sum / reduced.count as f32)
}

fn cmp_f32(left: &f32, right: &f32) -> std::cmp::Ordering {
    left.total_cmp(right)
}

/// The value that would sit at `index` after a total-order sort.
///
/// `index` must be in range.
pub fn select_rank(values: &mut [f32], index: usize) -> f32 {
    values.select_nth_unstable_by(index, cmp_f32);
    values[index]
}

/// The values that would sit at `lo` and `hi` after a total-order sort.
///
/// `lo <= hi`, and both must be in range.
pub fn select_ranks(values: &mut [f32], lo: usize, hi: usize) -> (f32, f32) {
    values.select_nth_unstable_by(hi, cmp_f32);
    let high = values[hi];
    let low = if lo == hi {
        high
    } else {
        values[..hi].select_nth_unstable_by(lo, cmp_f32);
        values[lo]
    };
    (low, high)
}

/// Even lengths average the two middle ranks. An empty slice panics.
pub fn median_unstable(values: &mut [f32]) -> f32 {
    let n = values.len();
    assert!(n > 0, "median of an empty slice");
    let mid = n / 2;
    if n % 2 == 1 {
        select_rank(values, mid)
    } else {
        let (low, high) = select_ranks(values, mid - 1, mid);
        (low + high) * 0.5
    }
}

/// Linear blend of the floor and ceil ranks. An empty slice returns NaN.
pub fn percentile_linear(values: &mut [f32], q: f64) -> f32 {
    if values.is_empty() {
        return f32::NAN;
    }
    let pos = q * (values.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil().min((values.len() - 1) as f64) as usize;
    let frac = (pos - lo as f64) as f32;
    let (low, high) = select_ranks(values, lo, hi);
    low * (1.0 - frac) + high * frac
}

/// Nearest rank. An empty slice returns 0, which is what distribution limits use.
pub fn percentile_nearest(values: &mut [f32], portion: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let last = values.len() - 1;
    let index = ((last as f32) * portion).round() as usize;
    select_rank(values, index.min(last))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sorted_percentile(values: &[f32], q: f64) -> f32 {
        let mut sorted = values.to_vec();
        sorted.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
        let pos = q * (sorted.len() - 1) as f64;
        let lo = pos.floor() as usize;
        let hi = pos.ceil().min((sorted.len() - 1) as f64) as usize;
        let frac = (pos - lo as f64) as f32;
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }

    #[test]
    fn ranks_match_a_full_sort_of_finite_samples() {
        let samples = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let mut work = samples;
        assert!(
            (percentile_linear(&mut work, 0.5) - sorted_percentile(&samples, 0.5)).abs() < 1e-6
        );
        assert!(
            (percentile_linear(&mut work, 0.95) - sorted_percentile(&samples, 0.95)).abs() < 1e-6
        );
        assert!(percentile_linear(&mut [], 0.5).is_nan());
        let mut even = [1.0, 100.0, 2.0, 3.0];
        assert!((median_unstable(&mut even) - 2.5).abs() < 1e-6);
        let mut odd = [1.0, 100.0, 2.0];
        assert!((median_unstable(&mut odd) - 2.0).abs() < 1e-6);
        let mut cloud = vec![1.0; 400];
        cloud.push(100.0);
        assert!((percentile_nearest(&mut cloud, 0.995) - 1.0).abs() < 1e-6);
        let reduced = reduce_finite(&[f32::NAN, -2.0, 4.0, f32::INFINITY]);
        assert_eq!(reduced.count, 2);
        assert!((reduced.min + 2.0).abs() < 1e-6);
        assert!((reduced.max - 4.0).abs() < 1e-6);
        assert!(finite_minmax(&[f32::NAN]).is_none());
        assert!((mean_finite(&[1.0, 3.0, f32::NAN]).unwrap() - 2.0).abs() < 1e-6);
    }
}
