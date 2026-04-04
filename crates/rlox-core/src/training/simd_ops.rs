//! SIMD-accelerated operations for training hot loops.
//!
//! Provides vectorized versions of weight updates, reward shaping, and
//! priority computation. Gated behind the `simd` feature flag.
//!
//! Strategy: use `chunks_exact` with manual unrolling to give LLVM strong
//! auto-vectorization hints. On x86_64, the compiler will emit AVX2/SSE
//! instructions for these patterns. No nightly features or `std::simd` required.

// ---------------------------------------------------------------------------
// Weight operations (f32)
// ---------------------------------------------------------------------------

/// SIMD-friendly Reptile update: `target[i] += lr * (source[i] - target[i])`
///
/// Processes 8 f32s at a time via `chunks_exact`, giving LLVM a clear
/// vectorization opportunity (256-bit AVX2 = 8 x f32).
///
/// # Panics
/// Panics if `target.len() != source.len()`.
#[inline]
pub fn reptile_update_simd(target: &mut [f32], source: &[f32], lr: f32) {
    assert_eq!(target.len(), source.len(), "length mismatch");
    let n = target.len();
    let chunks = n / 8;
    let remainder = n % 8;

    let (target_chunks, target_rest) = target.split_at_mut(chunks * 8);
    let (source_chunks, source_rest) = source.split_at(chunks * 8);

    // Process 8 elements at a time -- LLVM will auto-vectorize this
    for (t_chunk, s_chunk) in target_chunks
        .chunks_exact_mut(8)
        .zip(source_chunks.chunks_exact(8))
    {
        for i in 0..8 {
            t_chunk[i] += lr * (s_chunk[i] - t_chunk[i]);
        }
    }

    // Scalar remainder
    for i in 0..remainder {
        target_rest[i] += lr * (source_rest[i] - target_rest[i]);
    }
}

/// SIMD-friendly Polyak update: `target[i] = tau * source[i] + (1 - tau) * target[i]`
///
/// Processes 8 f32s at a time.
///
/// # Panics
/// Panics if `target.len() != source.len()`.
#[inline]
pub fn polyak_update_simd(target: &mut [f32], source: &[f32], tau: f32) {
    assert_eq!(target.len(), source.len(), "length mismatch");
    let n = target.len();
    let one_minus_tau = 1.0 - tau;

    let chunks = n / 8;
    let remainder = n % 8;

    let (target_chunks, target_rest) = target.split_at_mut(chunks * 8);
    let (source_chunks, source_rest) = source.split_at(chunks * 8);

    for (t_chunk, s_chunk) in target_chunks
        .chunks_exact_mut(8)
        .zip(source_chunks.chunks_exact(8))
    {
        for i in 0..8 {
            t_chunk[i] = tau * s_chunk[i] + one_minus_tau * t_chunk[i];
        }
    }

    for i in 0..remainder {
        target_rest[i] = tau * source_rest[i] + one_minus_tau * target_rest[i];
    }
}

// ---------------------------------------------------------------------------
// Reward shaping (f64)
// ---------------------------------------------------------------------------

/// SIMD-friendly PBRS: `result[i] = rewards[i] + gamma * phi_next[i] - phi_current[i]`
///
/// At episode boundaries (`dones[i] == 1.0`), returns the raw reward.
/// Processes 4 f64s at a time (256-bit AVX2 = 4 x f64).
///
/// # Panics
/// Panics if all slices are not the same length.
#[inline]
pub fn pbrs_simd(
    rewards: &[f64],
    phi_current: &[f64],
    phi_next: &[f64],
    gamma: f64,
    dones: &[f64],
) -> Vec<f64> {
    let n = rewards.len();
    assert_eq!(phi_current.len(), n);
    assert_eq!(phi_next.len(), n);
    assert_eq!(dones.len(), n);

    let mut output = vec![0.0f64; n];

    let chunks = n / 4;
    let remainder = n % 4;

    // Process 4 elements at a time
    for chunk_idx in 0..chunks {
        let base = chunk_idx * 4;
        for i in 0..4 {
            let idx = base + i;
            // Branchless: mask = 1.0 when not done, 0.0 when done
            // done==1.0 -> shaping=0, done==0.0 -> shaping=gamma*phi_next - phi_current
            let not_done = 1.0 - dones[idx];
            let shaping = not_done * (gamma * phi_next[idx] - phi_current[idx]);
            output[idx] = rewards[idx] + shaping;
        }
    }

    // Scalar remainder
    let base = chunks * 4;
    for i in 0..remainder {
        let idx = base + i;
        let not_done = 1.0 - dones[idx];
        let shaping = not_done * (gamma * phi_next[idx] - phi_current[idx]);
        output[idx] = rewards[idx] + shaping;
    }

    output
}

// ---------------------------------------------------------------------------
// Priority computation (f64)
// ---------------------------------------------------------------------------

/// SIMD-friendly LAP priority: `priority[i] = |loss[i]| + epsilon`
///
/// The `abs + add` portion is trivially vectorizable. The subsequent
/// `powf(alpha)` is left to the caller (not SIMD-friendly).
///
/// Processes 4 f64s at a time.
#[inline]
pub fn compute_priorities_simd(losses: &[f64], epsilon: f64) -> Vec<f64> {
    let n = losses.len();
    let mut output = vec![0.0f64; n];

    let chunks = n / 4;
    let remainder = n % 4;

    for chunk_idx in 0..chunks {
        let base = chunk_idx * 4;
        for i in 0..4 {
            output[base + i] = losses[base + i].abs() + epsilon;
        }
    }

    let base = chunks * 4;
    for i in 0..remainder {
        output[base + i] = losses[base + i].abs() + epsilon;
    }

    output
}

/// SIMD-friendly weight vector averaging: `result[i] = sum(vectors[j][i]) / n`
///
/// Accumulates across vectors using chunks of 8 f32s for the inner loop.
///
/// # Panics
/// Panics if `vectors` is empty or vectors have different lengths.
#[inline]
pub fn average_weights_simd(vectors: &[&[f32]]) -> Vec<f32> {
    assert!(!vectors.is_empty(), "cannot average zero vectors");
    let dim = vectors[0].len();
    for v in vectors.iter().skip(1) {
        assert_eq!(v.len(), dim, "all vectors must have the same length");
    }

    let n = vectors.len() as f32;
    let mut result = vec![0.0f32; dim];

    for v in vectors {
        let chunks = dim / 8;
        let remainder = dim % 8;

        let (result_chunks, result_rest) = result.split_at_mut(chunks * 8);
        let (v_chunks, v_rest) = v.split_at(chunks * 8);

        for (r_chunk, v_chunk) in result_chunks
            .chunks_exact_mut(8)
            .zip(v_chunks.chunks_exact(8))
        {
            for i in 0..8 {
                r_chunk[i] += v_chunk[i];
            }
        }

        for i in 0..remainder {
            result_rest[i] += v_rest[i];
        }
    }

    // Divide by n
    let chunks = dim / 8;
    let remainder = dim % 8;
    let (result_chunks, result_rest) = result.split_at_mut(chunks * 8);
    for chunk in result_chunks.chunks_exact_mut(8) {
        for item in chunk.iter_mut().take(8) {
            *item /= n;
        }
    }
    for item in result_rest.iter_mut().take(remainder) {
        *item /= n;
    }

    result
}

// ---------------------------------------------------------------------------
// Image augmentation helper
// ---------------------------------------------------------------------------

/// Copy a contiguous row of pixels using `copy_from_slice` (auto-vectorizes
/// to SIMD memcpy on all targets).
///
/// This is a thin wrapper that makes the intent explicit for the compiler.
#[inline]
pub fn copy_pixel_row(dst: &mut [f32], src: &[f32]) {
    debug_assert_eq!(dst.len(), src.len());
    dst.copy_from_slice(src);
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Reptile SIMD tests
    // -----------------------------------------------------------------------

    #[test]
    fn reptile_simd_lr_one_copies() {
        let mut target = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let source = vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0];
        reptile_update_simd(&mut target, &source, 1.0);
        assert_eq!(target, source);
    }

    #[test]
    fn reptile_simd_lr_zero_no_change() {
        let original = vec![1.0f32, 2.0, 3.0];
        let mut target = original.clone();
        let source = vec![10.0f32, 20.0, 30.0];
        reptile_update_simd(&mut target, &source, 0.0);
        assert_eq!(target, original);
    }

    #[test]
    fn reptile_simd_matches_scalar() {
        let mut target_simd = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let mut target_scalar = target_simd.clone();
        let source: Vec<f32> = (0..10).map(|i| i as f32 * 3.0 + 0.5).collect();
        let lr = 0.3;

        reptile_update_simd(&mut target_simd, &source, lr);

        // Scalar reference
        for (t, &s) in target_scalar.iter_mut().zip(source.iter()) {
            *t += lr * (s - *t);
        }

        for (i, (a, b)) in target_simd.iter().zip(target_scalar.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "mismatch at index {i}: simd={a}, scalar={b}"
            );
        }
    }

    #[test]
    fn reptile_simd_empty_slices() {
        let mut target: Vec<f32> = vec![];
        let source: Vec<f32> = vec![];
        reptile_update_simd(&mut target, &source, 0.5);
        assert!(target.is_empty());
    }

    // -----------------------------------------------------------------------
    // Polyak SIMD tests
    // -----------------------------------------------------------------------

    #[test]
    fn polyak_simd_tau_one_copies() {
        let mut target = vec![1.0f32; 9];
        let source = vec![5.0f32; 9];
        polyak_update_simd(&mut target, &source, 1.0);
        assert_eq!(target, source);
    }

    #[test]
    fn polyak_simd_tau_zero_no_change() {
        let original = vec![3.0f32; 5];
        let mut target = original.clone();
        let source = vec![99.0f32; 5];
        polyak_update_simd(&mut target, &source, 0.0);
        assert_eq!(target, original);
    }

    #[test]
    fn polyak_simd_matches_scalar() {
        let mut target_simd: Vec<f32> = (0..17).map(|i| i as f32 * 0.7).collect();
        let mut target_scalar = target_simd.clone();
        let source: Vec<f32> = (0..17).map(|i| i as f32 * 2.1 + 0.3).collect();
        let tau = 0.005;

        polyak_update_simd(&mut target_simd, &source, tau);

        let one_minus_tau = 1.0 - tau;
        for (t, &s) in target_scalar.iter_mut().zip(source.iter()) {
            *t = tau * s + one_minus_tau * *t;
        }

        for (i, (a, b)) in target_simd.iter().zip(target_scalar.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "mismatch at index {i}: simd={a}, scalar={b}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // PBRS SIMD tests
    // -----------------------------------------------------------------------

    #[test]
    fn pbrs_simd_known_values() {
        let rewards = &[1.0, 2.0];
        let phi = &[0.5, 0.3];
        let phi_next = &[0.3, 0.8];
        let gamma = 0.99;
        let dones = &[0.0, 0.0];
        let result = pbrs_simd(rewards, phi, phi_next, gamma, dones);
        assert!((result[0] - 0.797).abs() < 1e-10, "got {}", result[0]);
        assert!((result[1] - 2.492).abs() < 1e-10, "got {}", result[1]);
    }

    #[test]
    fn pbrs_simd_done_zeroes_shaping() {
        let rewards = &[1.0, 2.0];
        let phi = &[0.5, 0.3];
        let phi_next = &[0.3, 0.8];
        let gamma = 0.99;
        let dones = &[0.0, 1.0];
        let result = pbrs_simd(rewards, phi, phi_next, gamma, dones);
        assert!((result[1] - 2.0).abs() < 1e-10, "got {}", result[1]);
    }

    #[test]
    fn pbrs_simd_matches_scalar() {
        use crate::training::reward_shaping::shape_rewards_pbrs;

        let n = 13; // non-multiple of 4 to test remainder
        let rewards: Vec<f64> = (0..n).map(|i| i as f64 * 0.3).collect();
        let phi: Vec<f64> = (0..n).map(|i| i as f64 * 0.1 + 0.05).collect();
        let phi_next: Vec<f64> = (0..n).map(|i| i as f64 * 0.2 - 0.1).collect();
        let gamma = 0.99;
        let dones: Vec<f64> = (0..n).map(|i| if i == 5 { 1.0 } else { 0.0 }).collect();

        let simd_result = pbrs_simd(&rewards, &phi, &phi_next, gamma, &dones);
        let scalar_result = shape_rewards_pbrs(&rewards, &phi, &phi_next, gamma, &dones).unwrap();

        for (i, (a, b)) in simd_result.iter().zip(scalar_result.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-10,
                "mismatch at index {i}: simd={a}, scalar={b}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Priority SIMD tests
    // -----------------------------------------------------------------------

    #[test]
    fn priorities_simd_abs_plus_epsilon() {
        let losses = &[-3.0, 0.0, 2.5, -0.1];
        let eps = 0.01;
        let result = compute_priorities_simd(losses, eps);
        assert!((result[0] - 3.01).abs() < 1e-10);
        assert!((result[1] - 0.01).abs() < 1e-10);
        assert!((result[2] - 2.51).abs() < 1e-10);
        assert!((result[3] - 0.11).abs() < 1e-10);
    }

    #[test]
    fn priorities_simd_matches_scalar() {
        let losses: Vec<f64> = (0..11).map(|i| (i as f64 - 5.0) * 1.7).collect();
        let eps = 0.001;

        let simd_result = compute_priorities_simd(&losses, eps);
        let scalar_result: Vec<f64> = losses.iter().map(|l| l.abs() + eps).collect();

        for (i, (a, b)) in simd_result.iter().zip(scalar_result.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-12,
                "mismatch at index {i}: simd={a}, scalar={b}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Average weights SIMD tests
    // -----------------------------------------------------------------------

    #[test]
    fn average_weights_simd_single_vector() {
        let v = vec![1.0f32, 2.0, 3.0];
        let result = average_weights_simd(&[&v]);
        assert_eq!(result, v);
    }

    #[test]
    fn average_weights_simd_two_vectors() {
        let v1 = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let v2 = [9.0f32, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let result = average_weights_simd(&[&v1, &v2]);
        for &r in &result {
            assert!((r - 5.0).abs() < 1e-5, "expected 5.0, got {r}");
        }
    }

    // -----------------------------------------------------------------------
    // copy_pixel_row test
    // -----------------------------------------------------------------------

    #[test]
    fn copy_pixel_row_works() {
        let src = [1.0f32, 2.0, 3.0, 4.0];
        let mut dst = [0.0f32; 4];
        copy_pixel_row(&mut dst, &src);
        assert_eq!(dst, src);
    }

    // -----------------------------------------------------------------------
    // Proptests: SIMD == scalar for random inputs
    // -----------------------------------------------------------------------

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn prop_reptile_simd_matches_scalar(
                dim in 1usize..500,
                lr in 0.0f32..1.0,
            ) {
                let source: Vec<f32> = (0..dim).map(|i| (i as f32) * 2.1 + 0.3).collect();
                let original: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.7).collect();

                let mut target_simd = original.clone();
                reptile_update_simd(&mut target_simd, &source, lr);

                let mut target_scalar = original.clone();
                for (t, &s) in target_scalar.iter_mut().zip(source.iter()) {
                    *t += lr * (s - *t);
                }

                for (i, (a, b)) in target_simd.iter().zip(target_scalar.iter()).enumerate() {
                    prop_assert!(
                        (a - b).abs() < 1e-4,
                        "reptile mismatch at {i}: simd={a}, scalar={b}"
                    );
                }
            }

            #[test]
            fn prop_polyak_simd_matches_scalar(
                dim in 1usize..500,
                tau in 0.0f32..1.0,
            ) {
                let source: Vec<f32> = (0..dim).map(|i| (i as f32) * 3.0).collect();
                let original: Vec<f32> = (0..dim).map(|i| i as f32).collect();

                let mut target_simd = original.clone();
                polyak_update_simd(&mut target_simd, &source, tau);

                let one_minus_tau = 1.0 - tau;
                let mut target_scalar = original.clone();
                for (t, &s) in target_scalar.iter_mut().zip(source.iter()) {
                    *t = tau * s + one_minus_tau * *t;
                }

                for (i, (a, b)) in target_simd.iter().zip(target_scalar.iter()).enumerate() {
                    prop_assert!(
                        (a - b).abs() < 1e-4,
                        "polyak mismatch at {i}: simd={a}, scalar={b}"
                    );
                }
            }

            #[test]
            fn prop_pbrs_simd_matches_scalar(n in 1usize..200) {
                let rewards: Vec<f64> = (0..n).map(|i| i as f64 * 0.3).collect();
                let phi: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
                let phi_next: Vec<f64> = (0..n).map(|i| i as f64 * 0.2).collect();
                let gamma = 0.99;
                let dones: Vec<f64> = (0..n).map(|i| if i % 7 == 0 { 1.0 } else { 0.0 }).collect();

                let simd_result = pbrs_simd(&rewards, &phi, &phi_next, gamma, &dones);

                // Scalar reference
                let mut scalar_result = vec![0.0f64; n];
                for i in 0..n {
                    if dones[i] == 1.0 {
                        scalar_result[i] = rewards[i];
                    } else {
                        scalar_result[i] = rewards[i] + gamma * phi_next[i] - phi[i];
                    }
                }

                for (i, (a, b)) in simd_result.iter().zip(scalar_result.iter()).enumerate() {
                    prop_assert!(
                        (a - b).abs() < 1e-10,
                        "pbrs mismatch at {i}: simd={a}, scalar={b}"
                    );
                }
            }

            #[test]
            fn prop_priorities_simd_matches_scalar(n in 1usize..200) {
                let losses: Vec<f64> = (0..n).map(|i| (i as f64 - 50.0) * 1.3).collect();
                let eps = 0.01;

                let simd_result = compute_priorities_simd(&losses, eps);
                let scalar_result: Vec<f64> = losses.iter().map(|l| l.abs() + eps).collect();

                for (i, (a, b)) in simd_result.iter().zip(scalar_result.iter()).enumerate() {
                    prop_assert!(
                        (a - b).abs() < 1e-12,
                        "priority mismatch at {i}: simd={a}, scalar={b}"
                    );
                }
            }

            #[test]
            fn prop_average_weights_simd_matches_scalar(
                dim in 1usize..100,
                num_vecs in 1usize..10,
            ) {
                let vectors: Vec<Vec<f32>> = (0..num_vecs)
                    .map(|v| (0..dim).map(|i| (v * dim + i) as f32 * 0.1).collect())
                    .collect();
                let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

                let simd_result = average_weights_simd(&refs);

                // Scalar reference
                let n = num_vecs as f32;
                let mut scalar_result = vec![0.0f32; dim];
                for v in &vectors {
                    for (r, &val) in scalar_result.iter_mut().zip(v.iter()) {
                        *r += val;
                    }
                }
                for r in &mut scalar_result {
                    *r /= n;
                }

                for (i, (a, b)) in simd_result.iter().zip(scalar_result.iter()).enumerate() {
                    prop_assert!(
                        (a - b).abs() < 1e-3,
                        "average mismatch at {i}: simd={a}, scalar={b}"
                    );
                }
            }
        }
    }
}
