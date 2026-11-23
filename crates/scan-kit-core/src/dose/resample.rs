//! Trilinear resample of one dose grid onto another.
//!
//! Axis-aligned affine only — a rotated frame pair never reaches here because
//! the study load rejects mismatched frames of reference. Points that land
//! outside the source sample zero, matching the `map_coordinates` order-1,
//! `mode="constant"` behaviour the Python analysis used.

/// `values` (x-fastest on `src_shape`) sampled at `dst`'s voxel centers.
pub fn resample_linear(
    values: &[f32],
    src_origin: [f32; 3],
    src_spacing: [f32; 3],
    src_shape: [usize; 3],
    dst_origin: [f32; 3],
    dst_spacing: [f32; 3],
    dst_shape: [usize; 3],
) -> Vec<f32> {
    let [nx, ny, nz] = src_shape;
    let get = |x: isize, y: isize, z: isize| -> f32 {
        if x < 0 || y < 0 || z < 0 || x >= nx as isize || y >= ny as isize || z >= nz as isize {
            return 0.0;
        }
        // The source shape is caller-supplied — a wrapped flat index could
        // sample the wrong in-bounds voxel, not just miss.
        ny.checked_mul(z as usize)
            .and_then(|yz| (y as usize).checked_add(yz))
            .and_then(|yz| nx.checked_mul(yz))
            .and_then(|xyz| (x as usize).checked_add(xyz))
            .and_then(|index| values.get(index))
            .copied()
            .unwrap_or(0.0)
    };
    let Some(len) = dst_shape[0]
        .checked_mul(dst_shape[1])
        .and_then(|n| n.checked_mul(dst_shape[2]))
    else {
        return Vec::new();
    };
    // A zero/NaN spacing makes every sample index ±inf — splat_gaussians
    // treats non-positive spacing as "no data"; this volume matches.
    let spacing_ok = src_spacing
        .iter()
        .chain(dst_spacing.iter())
        .all(|v| v.is_finite() && *v > 0.0);
    if !spacing_ok {
        return vec![0.0f32; len];
    }
    let mut out = vec![0.0f32; len];
    for z in 0..dst_shape[2] {
        for y in 0..dst_shape[1] {
            for x in 0..dst_shape[0] {
                let mut acc = 0.0f32;
                let patient = [
                    dst_origin[0] + x as f32 * dst_spacing[0],
                    dst_origin[1] + y as f32 * dst_spacing[1],
                    dst_origin[2] + z as f32 * dst_spacing[2],
                ];
                let f: [f32; 3] = std::array::from_fn(|axis| {
                    (patient[axis] - src_origin[axis]) / src_spacing[axis]
                });
                let base: [isize; 3] = std::array::from_fn(|axis| f[axis].floor() as isize);
                let t: [f32; 3] = std::array::from_fn(|axis| f[axis] - base[axis] as f32);
                for dz in 0..2 {
                    for dy in 0..2 {
                        for dx in 0..2 {
                            let weight = if dx == 0 { 1.0 - t[0] } else { t[0] }
                                * if dy == 0 { 1.0 - t[1] } else { t[1] }
                                * if dz == 0 { 1.0 - t[2] } else { t[2] };
                            // `as isize` saturates on absurd coords — the
                            // adds must not overflow in a debug build.
                            if weight == 0.0 {
                                continue;
                            }

                            acc += weight
                                * get(
                                    base[0].saturating_add(dx),
                                    base[1].saturating_add(dy),
                                    base[2].saturating_add(dz),
                                );
                        }
                    }
                }
                out[x + dst_shape[0] * (y + dst_shape[1] * z)] = acc;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::resample_linear;

    #[test]
    fn the_same_grid_gives_the_same_dose() {
        let values = vec![1.0f32, 2.0, 3.0, 4.0];
        let out = resample_linear(
            &values,
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2, 2, 1],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2, 2, 1],
        );
        assert_eq!(out, values);
    }

    #[test]
    fn a_linear_dose_resamples_exactly() {
        // dose(x) = x on a 4-voxel line: half-voxel shift lands mid-way.
        let values = vec![0.0f32, 1.0, 2.0, 3.0];
        let out = resample_linear(
            &values,
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [4, 1, 1],
            [0.5, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2, 1, 1],
        );
        assert_eq!(out, vec![0.5, 1.5]);
    }

    #[test]
    fn outside_the_source_is_zero() {
        let values = vec![1.0f32, 2.0, 3.0, 4.0];
        let out = resample_linear(
            &values,
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [4, 1, 1],
            [10.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2, 1, 1],
        );
        assert_eq!(out, vec![0.0, 0.0]);
        // Half a voxel past the last center still blends in its tail.
        let edge = resample_linear(
            &values,
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [4, 1, 1],
            [3.5, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [1, 1, 1],
        );
        assert_eq!(edge, vec![2.0]);
    }

    #[test]
    fn a_dst_shape_product_that_overflows_returns_empty() {
        let out = resample_linear(
            &[0.0f32; 8],
            [0.0; 3],
            [1.0; 3],
            [2, 2, 2],
            [0.0; 3],
            [1.0; 3],
            [usize::MAX, 2, 2],
        );
        assert!(out.is_empty());
    }

    #[test]
    fn a_wrapped_source_index_reads_zero_not_a_stray_voxel() {
        // src_shape is caller-supplied — the flat index can overflow usize and
        // wrap into a real buffer offset.
        let out = resample_linear(
            &[1.0f32; 4],
            [0.0; 3],
            [1e-20; 3],
            [usize::MAX, usize::MAX, 1],
            [0.0; 3],
            [1.0; 3],
            [2, 2, 2],
        );
        assert_eq!(out.len(), 8);
        // Out-of-range samples weigh NaN; the wrapped index must contribute 0,
        // never a stray in-bounds voxel.
        assert!(out.iter().all(|v| v.is_nan() || *v == 0.0), "{out:?}");
    }

    #[test]
    fn a_zero_source_spacing_returns_zeros_instead_of_indexing_infinity() {
        // inf as isize saturates and base + 1 overflows — splat_gaussians
        // already treats non-positive spacing as "no data".
        let out = resample_linear(
            &[1.0; 8],
            [0.0; 3],
            [0.0; 3],
            [2, 2, 2],
            [10.0; 3],
            [1.0; 3],
            [1, 1, 1],
        );
        assert_eq!(out, vec![0.0]);
    }

    #[test]
    fn a_zero_weight_nan_neighbor_does_not_poison_the_sample() {
        // dst x=0 sits exactly on the scored voxel - the NaN neighbor
        // has zero weight and must not leak into the result.
        let values = [1.0f32, f32::NAN];
        let out = resample_linear(
            &values,
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2, 1, 1],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [1, 1, 1],
        );
        assert_eq!(out, vec![1.0]);
    }
}
