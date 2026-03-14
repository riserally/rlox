/// Online running statistics using Welford's algorithm.
///
/// Computes running mean, population variance, and standard deviation
/// in a numerically stable, single-pass manner.
pub struct RunningStats {
    count: u64,
    mean: f64,
    m2: f64,
}

impl RunningStats {
    /// Create a new empty statistics accumulator.
    pub fn new() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
        }
    }

    /// Update with a single observation (Welford's online algorithm).
    pub fn update(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
    }

    /// Update with a batch of observations.
    pub fn batch_update(&mut self, values: &[f64]) {
        for &v in values {
            self.update(v);
        }
    }

    /// Current running mean.
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Population variance (divide by n).
    pub fn var(&self) -> f64 {
        if self.count < 1 {
            return 0.0;
        }
        self.m2 / self.count as f64
    }

    /// Population standard deviation.
    pub fn std(&self) -> f64 {
        self.var().sqrt()
    }

    /// Normalize a value to a z-score using current mean and std.
    /// Returns 0.0 if std is near zero to avoid division by zero.
    pub fn normalize(&self, value: f64) -> f64 {
        let s = self.std();
        if s < 1e-8 {
            return 0.0;
        }
        (value - self.mean) / s
    }

    /// Number of observations seen.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Reset all accumulated statistics.
    pub fn reset(&mut self) {
        self.count = 0;
        self.mean = 0.0;
        self.m2 = 0.0;
    }
}

impl Default for RunningStats {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn running_stats_new_is_empty() {
        let stats = RunningStats::new();
        assert_eq!(stats.count(), 0);
    }

    #[test]
    fn running_stats_single_sample() {
        let mut stats = RunningStats::new();
        stats.update(5.0);
        assert!((stats.mean() - 5.0).abs() < 1e-10);
        assert_eq!(stats.count(), 1);
        let _ = stats.var();
        let _ = stats.std();
    }

    #[test]
    fn running_stats_welford_known_values() {
        let mut stats = RunningStats::new();
        for &x in &[2.0_f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
            stats.update(x);
        }
        assert!(
            (stats.mean() - 5.0).abs() < 1e-10,
            "mean should be 5.0, got {}",
            stats.mean()
        );
        assert!(
            (stats.var() - 4.0).abs() < 1e-10,
            "variance should be 4.0, got {}",
            stats.var()
        );
        assert!(
            (stats.std() - 2.0).abs() < 1e-10,
            "std should be 2.0, got {}",
            stats.std()
        );
    }

    #[test]
    fn running_stats_normalize_produces_z_score() {
        let mut stats = RunningStats::new();
        for &x in &[2.0_f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
            stats.update(x);
        }
        let z = stats.normalize(5.0);
        assert!(z.abs() < 1e-10, "normalize(mean) should be ~0, got {z}");
        let z2 = stats.normalize(7.0);
        assert!(
            (z2 - 1.0).abs() < 1e-10,
            "normalize(mean+std) should be ~1, got {z2}"
        );
    }

    #[test]
    fn running_stats_normalize_with_zero_std_does_not_panic() {
        let mut stats = RunningStats::new();
        stats.update(5.0);
        stats.update(5.0);
        stats.update(5.0);
        let z = stats.normalize(5.0);
        assert!(z.is_finite(), "normalize with zero std must be finite");
    }

    #[test]
    fn running_stats_large_stream_numerically_stable() {
        let mut stats = RunningStats::new();
        let base = 1_000_000.0f64;
        for i in 0..10_000 {
            stats.update(base + (i as f64) * 0.001);
        }
        let expected_mean = base + 5.0 - 0.001 / 2.0;
        assert!(
            (stats.mean() - expected_mean).abs() < 0.01,
            "mean imprecise for large offset: got {}, expected ~{expected_mean}",
            stats.mean()
        );
    }

    #[test]
    fn running_stats_reset_clears_state() {
        let mut stats = RunningStats::new();
        for &x in &[1.0f64, 2.0, 3.0] {
            stats.update(x);
        }
        stats.reset();
        assert_eq!(stats.count(), 0);
    }

    #[test]
    fn running_stats_nan_input_does_not_silently_corrupt() {
        let mut stats = RunningStats::new();
        stats.update(1.0);
        stats.update(2.0);
        let mean_before = stats.mean();
        stats.update(f64::NAN);
        let mean_after = stats.mean();
        if mean_after.is_finite() {
            assert!(
                (mean_after - mean_before).abs() < 1e-10 || mean_after.is_nan(),
                "NaN input corrupted finite mean: was {mean_before}, now {mean_after}"
            );
        }
    }

    #[test]
    fn running_stats_batch_update() {
        let mut stats = RunningStats::new();
        stats.batch_update(&[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
        assert!((stats.mean() - 5.0).abs() < 1e-10);
        assert_eq!(stats.count(), 8);
    }

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn running_stats_mean_matches_batch_mean(
                values in proptest::collection::vec(-1000.0f64..1000.0, 2..200)
            ) {
                let mut stats = RunningStats::new();
                for &v in &values {
                    stats.update(v);
                }
                let batch_mean = values.iter().sum::<f64>() / values.len() as f64;
                prop_assert!(
                    (stats.mean() - batch_mean).abs() < 1e-8,
                    "running mean {:.10} != batch mean {:.10}",
                    stats.mean(), batch_mean
                );
            }

            #[test]
            fn running_stats_variance_non_negative(
                values in proptest::collection::vec(-1000.0f64..1000.0, 2..200)
            ) {
                let mut stats = RunningStats::new();
                for &v in &values {
                    stats.update(v);
                }
                prop_assert!(stats.var() >= 0.0, "variance must be non-negative");
                prop_assert!(stats.std() >= 0.0, "std must be non-negative");
            }

            #[test]
            fn running_stats_std_equals_sqrt_var(
                values in proptest::collection::vec(-100.0f64..100.0, 2..100)
            ) {
                let mut stats = RunningStats::new();
                for &v in &values {
                    stats.update(v);
                }
                let computed_std = stats.var().sqrt();
                prop_assert!(
                    (stats.std() - computed_std).abs() < 1e-10,
                    "std {} != sqrt(var) {}",
                    stats.std(), computed_std
                );
            }

            #[test]
            fn running_stats_count_matches_updates(
                values in proptest::collection::vec(-100.0f64..100.0, 0..200)
            ) {
                let mut stats = RunningStats::new();
                for &v in &values {
                    stats.update(v);
                }
                prop_assert_eq!(stats.count() as usize, values.len());
            }
        }
    }
}
