//! Weight vector operations for meta-learning and target network updates.
//!
//! Provides Reptile-style meta updates, Polyak (exponential moving average)
//! updates for SAC/TD3 target networks, and weight vector averaging.

use crate::error::RloxError;

/// Trait for weight vector update strategies.
///
/// All operations are in-place on flat f32 weight vectors.
/// This enables adding new meta-learning update rules (MAML outer step,
/// Lookahead, exponential moving average variants) without modifying
/// existing code.
pub trait WeightUpdate: Send + Sync {
    /// Apply the update rule in-place.
    ///
    /// `target` is modified. `source` is read-only. `lr` controls step size.
    ///
    /// # Errors
    /// Returns `ShapeMismatch` if `target.len() != source.len()`.
    fn apply(
        &self,
        target: &mut [f32],
        source: &[f32],
        lr: f32,
    ) -> Result<(), RloxError>;

    /// Human-readable name.
    fn name(&self) -> &str;
}

/// Reptile meta-learning update strategy.
pub struct ReptileUpdate;

impl WeightUpdate for ReptileUpdate {
    #[inline]
    fn apply(
        &self,
        target: &mut [f32],
        source: &[f32],
        lr: f32,
    ) -> Result<(), RloxError> {
        reptile_update(target, source, lr)
    }

    fn name(&self) -> &str {
        "Reptile"
    }
}

/// Polyak (EMA) target network update strategy.
pub struct PolyakUpdate;

impl WeightUpdate for PolyakUpdate {
    #[inline]
    fn apply(
        &self,
        target: &mut [f32],
        source: &[f32],
        lr: f32,
    ) -> Result<(), RloxError> {
        polyak_update(target, source, lr)
    }

    fn name(&self) -> &str {
        "Polyak"
    }
}

/// Reptile weight update: `params += lr * (task_params - params)`
///
/// Operates in-place on `meta_params`. Both slices must have the same length.
/// When `meta_lr == 1.0`, this copies `task_params` into `meta_params`.
/// When `meta_lr == 0.0`, `meta_params` is unchanged.
#[inline]
pub fn reptile_update(
    meta_params: &mut [f32],
    task_params: &[f32],
    meta_lr: f32,
) -> Result<(), RloxError> {
    if meta_params.len() != task_params.len() {
        return Err(RloxError::ShapeMismatch {
            expected: format!("target.len()={}", meta_params.len()),
            got: format!("source.len()={}", task_params.len()),
        });
    }

    for (m, &t) in meta_params.iter_mut().zip(task_params.iter()) {
        *m += meta_lr * (t - *m);
    }
    Ok(())
}

/// Exponential moving average (Polyak update):
///   `target[i] = tau * source[i] + (1 - tau) * target[i]`
///
/// Used by SAC/TD3 for target network updates. Operates in-place on `target`.
#[inline]
pub fn polyak_update(
    target: &mut [f32],
    source: &[f32],
    tau: f32,
) -> Result<(), RloxError> {
    if target.len() != source.len() {
        return Err(RloxError::ShapeMismatch {
            expected: format!("target.len()={}", target.len()),
            got: format!("source.len()={}", source.len()),
        });
    }

    let one_minus_tau = 1.0 - tau;
    for (t, &s) in target.iter_mut().zip(source.iter()) {
        *t = tau * s + one_minus_tau * *t;
    }
    Ok(())
}

/// Average N weight vectors element-wise: `result[i] = mean(vectors[j][i] for all j)`
///
/// All vectors must have the same length.
pub fn average_weight_vectors(vectors: &[&[f32]]) -> Result<Vec<f32>, RloxError> {
    if vectors.is_empty() {
        return Err(RloxError::BufferError(
            "cannot average zero weight vectors".into(),
        ));
    }

    let dim = vectors[0].len();
    for (i, v) in vectors.iter().enumerate().skip(1) {
        if v.len() != dim {
            return Err(RloxError::ShapeMismatch {
                expected: format!("all vectors length {dim}"),
                got: format!("vectors[{i}].len()={}", v.len()),
            });
        }
    }

    let n = vectors.len() as f32;
    let mut result = vec![0.0f32; dim];
    for v in vectors {
        for (r, &val) in result.iter_mut().zip(v.iter()) {
            *r += val;
        }
    }
    for r in &mut result {
        *r /= n;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reptile_update_lr_one_copies() {
        let mut meta = vec![1.0f32, 2.0, 3.0];
        let task = vec![4.0f32, 5.0, 6.0];
        reptile_update(&mut meta, &task, 1.0).unwrap();
        assert_eq!(meta, vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_reptile_update_lr_zero_no_change() {
        let mut meta = vec![1.0f32, 2.0, 3.0];
        let task = vec![4.0f32, 5.0, 6.0];
        reptile_update(&mut meta, &task, 0.0).unwrap();
        assert_eq!(meta, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_reptile_update_lr_half() {
        let mut meta = vec![0.0f32, 0.0];
        let task = vec![2.0f32, 4.0];
        reptile_update(&mut meta, &task, 0.5).unwrap();
        assert!((meta[0] - 1.0).abs() < 1e-6);
        assert!((meta[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_reptile_length_mismatch() {
        let mut meta = vec![1.0f32, 2.0, 3.0];
        let task = vec![4.0f32, 5.0];
        let result = reptile_update(&mut meta, &task, 0.5);
        assert!(matches!(result, Err(RloxError::ShapeMismatch { .. })));
    }

    #[test]
    fn test_average_weight_vectors_mean() {
        let v1 = [1.0f32, 2.0, 3.0];
        let v2 = [4.0f32, 5.0, 6.0];
        let result = average_weight_vectors(&[&v1, &v2]).unwrap();
        assert!((result[0] - 2.5).abs() < 1e-6);
        assert!((result[1] - 3.5).abs() < 1e-6);
        assert!((result[2] - 4.5).abs() < 1e-6);
    }

    #[test]
    fn test_average_single_vector() {
        let v = [7.0f32, 8.0, 9.0];
        let result = average_weight_vectors(&[&v]).unwrap();
        assert_eq!(result, vec![7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_average_empty_errors() {
        let result = average_weight_vectors(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_polyak_update_tau_one_copies() {
        let mut target = vec![1.0f32, 2.0];
        let source = vec![3.0f32, 4.0];
        polyak_update(&mut target, &source, 1.0).unwrap();
        assert_eq!(target, vec![3.0, 4.0]);
    }

    #[test]
    fn test_polyak_update_tau_zero_no_change() {
        let mut target = vec![1.0f32, 2.0];
        let source = vec![3.0f32, 4.0];
        polyak_update(&mut target, &source, 0.0).unwrap();
        assert_eq!(target, vec![1.0, 2.0]);
    }

    #[test]
    fn test_polyak_length_mismatch() {
        let mut target = vec![1.0f32, 2.0];
        let source = vec![3.0f32];
        let result = polyak_update(&mut target, &source, 0.5);
        assert!(matches!(result, Err(RloxError::ShapeMismatch { .. })));
    }

    #[test]
    fn test_trait_object_safety() {
        let update: Box<dyn WeightUpdate> = Box::new(ReptileUpdate);
        assert_eq!(update.name(), "Reptile");
        let update: Box<dyn WeightUpdate> = Box::new(PolyakUpdate);
        assert_eq!(update.name(), "Polyak");
    }

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn prop_reptile_interpolates(
                dim in 1usize..100,
                lr in 0.0f32..1.0,
            ) {
                let meta: Vec<f32> = (0..dim).map(|i| i as f32).collect();
                let task: Vec<f32> = (0..dim).map(|i| (i as f32) * 2.0 + 1.0).collect();
                let mut result = meta.clone();
                reptile_update(&mut result, &task, lr).unwrap();
                for i in 0..dim {
                    let lo = meta[i].min(task[i]);
                    let hi = meta[i].max(task[i]);
                    prop_assert!(
                        result[i] >= lo - 1e-6 && result[i] <= hi + 1e-6,
                        "result[{i}]={} not in [{lo}, {hi}]", result[i]
                    );
                }
            }

            #[test]
            fn prop_polyak_interpolates(
                dim in 1usize..100,
                tau in 0.0f32..1.0,
            ) {
                let target: Vec<f32> = (0..dim).map(|i| i as f32).collect();
                let source: Vec<f32> = (0..dim).map(|i| (i as f32) * 3.0).collect();
                let mut result = target.clone();
                polyak_update(&mut result, &source, tau).unwrap();
                for i in 0..dim {
                    let lo = target[i].min(source[i]);
                    let hi = target[i].max(source[i]);
                    prop_assert!(
                        result[i] >= lo - 1e-6 && result[i] <= hi + 1e-6,
                        "result[{i}]={} not in [{lo}, {hi}]", result[i]
                    );
                }
            }

            #[test]
            fn prop_average_idempotent(
                dim in 1usize..50,
                n in 1usize..10,
            ) {
                let v: Vec<f32> = (0..dim).map(|i| i as f32 * 0.7).collect();
                let refs: Vec<&[f32]> = (0..n).map(|_| v.as_slice()).collect();
                let result = average_weight_vectors(&refs).unwrap();
                for i in 0..dim {
                    prop_assert!(
                        (result[i] - v[i]).abs() < 1e-5,
                        "averaging {n} copies: result[{i}]={}, expected {}",
                        result[i], v[i]
                    );
                }
            }
        }
    }
}
