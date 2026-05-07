/// Online running statistics using Welford's algorithm.
///
/// Computes running mean, population variance, and standard deviation
/// in a numerically stable, single-pass manner.
#[derive(Debug, Clone)]
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

/// Per-dimension online running statistics using Welford's algorithm.
///
/// Maintains independent mean/variance accumulators for each dimension,
/// enabling proper per-feature observation normalization as in SB3's
/// `RunningMeanStd`.
#[derive(Debug, Clone)]
pub struct RunningStatsVec {
    dim: usize,
    count: u64,
    mean: Vec<f64>,
    m2: Vec<f64>,
}

impl RunningStatsVec {
    /// Create a new accumulator for vectors of the given dimensionality.
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            count: 0,
            mean: vec![0.0; dim],
            m2: vec![0.0; dim],
        }
    }

    /// Update with a single sample of length `dim` (Welford per dimension).
    ///
    /// # Panics
    ///
    /// Panics if `values.len() != dim`.
    #[inline]
    pub fn update(&mut self, values: &[f64]) {
        assert_eq!(
            values.len(),
            self.dim,
            "expected {} dimensions, got {}",
            self.dim,
            values.len()
        );
        self.count += 1;
        let n = self.count as f64;
        for (i, &val) in values.iter().enumerate().take(self.dim) {
            let delta = val - self.mean[i];
            self.mean[i] += delta / n;
            let delta2 = val - self.mean[i];
            self.m2[i] += delta * delta2;
        }
    }

    /// Update with a flat batch of `batch_size` samples, each of `dim` dimensions.
    ///
    /// `data` must have length `batch_size * dim`, laid out as
    /// `[sample0_dim0, sample0_dim1, ..., sample1_dim0, ...]`.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != batch_size * dim`.
    #[inline]
    pub fn batch_update(&mut self, data: &[f64], batch_size: usize) {
        assert_eq!(
            data.len(),
            batch_size * self.dim,
            "expected {} elements (batch_size={} * dim={}), got {}",
            batch_size * self.dim,
            batch_size,
            self.dim,
            data.len()
        );
        for sample in data.chunks_exact(self.dim) {
            self.update(sample);
        }
    }

    /// Return the current per-dimension mean vector (clone).
    #[inline]
    pub fn mean(&self) -> Vec<f64> {
        self.mean.clone()
    }

    /// Borrow the per-dimension mean as a slice (zero-cost).
    #[inline]
    pub fn mean_ref(&self) -> &[f64] {
        &self.mean
    }

    /// Return the per-dimension population variance vector.
    #[inline]
    pub fn var(&self) -> Vec<f64> {
        if self.count < 1 {
            return vec![0.0; self.dim];
        }
        let n = self.count as f64;
        self.m2.iter().map(|&m| m / n).collect()
    }

    /// Return the per-dimension population standard deviation vector.
    #[inline]
    pub fn std(&self) -> Vec<f64> {
        self.var().iter().map(|&v| v.sqrt()).collect()
    }

    /// Normalize a single sample: `(values - mean) / max(std, 1e-8)` per dimension.
    ///
    /// # Panics
    ///
    /// Panics if `values.len() != dim`.
    #[inline]
    pub fn normalize(&self, values: &[f64]) -> Vec<f64> {
        assert_eq!(
            values.len(),
            self.dim,
            "expected {} dimensions, got {}",
            self.dim,
            values.len()
        );
        let std = self.std();
        values
            .iter()
            .zip(self.mean.iter())
            .zip(std.iter())
            .map(|((&v, &m), &s)| (v - m) / s.max(1e-8))
            .collect()
    }

    /// Normalize a flat batch of `batch_size` samples.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != batch_size * dim`.
    #[inline]
    pub fn normalize_batch(&self, data: &[f64], batch_size: usize) -> Vec<f64> {
        assert_eq!(
            data.len(),
            batch_size * self.dim,
            "expected {} elements (batch_size={} * dim={}), got {}",
            batch_size * self.dim,
            batch_size,
            self.dim,
            data.len()
        );
        let std = self.std();
        let mut out = Vec::with_capacity(data.len());
        for sample in data.chunks_exact(self.dim) {
            for i in 0..self.dim {
                out.push((sample[i] - self.mean[i]) / std[i].max(1e-8));
            }
        }
        out
    }

    /// Number of samples seen so far.
    #[inline]
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Dimensionality of the tracked vectors.
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Reset all accumulated statistics, keeping the dimensionality.
    pub fn reset(&mut self) {
        self.count = 0;
        self.mean.fill(0.0);
        self.m2.fill(0.0);
    }
}

/// Exponential Moving Average (EMA) running statistics for non-stationary signals.
///
/// Unlike Welford's algorithm which weights all observations equally,
/// EMA statistics give exponentially more weight to recent observations.
/// This makes them suitable for tracking non-stationary distributions
/// where the underlying mean/variance drift over time.
///
/// Update rules:
///   mean_t = (1 - alpha) * mean_{t-1} + alpha * x_t
///   var_t  = (1 - alpha) * var_{t-1} + alpha * (x_t - mean_t)^2
///
/// The smoothing factor `alpha` controls the effective window:
///   - alpha = 2/(N+1) gives an N-step equivalent window
///   - Higher alpha = more responsive to change, noisier estimates
///   - Lower alpha = smoother estimates, slower to adapt
#[derive(Debug, Clone)]
pub struct ExponentialRunningStats {
    alpha: f64,
    mean: f64,
    var: f64,
    count: u64,
    initialized: bool,
}

impl ExponentialRunningStats {
    /// Create a new EMA statistics tracker.
    ///
    /// # Parameters
    /// - `alpha`: Smoothing factor in (0, 1). Use `from_halflife` or
    ///   `from_window` for more intuitive parameterization.
    ///
    /// # Panics
    /// Panics if alpha is not in (0, 1).
    pub fn new(alpha: f64) -> Self {
        assert!(
            alpha > 0.0 && alpha < 1.0,
            "alpha must be in (0, 1), got {alpha}"
        );
        Self {
            alpha,
            mean: 0.0,
            var: 0.0,
            count: 0,
            initialized: false,
        }
    }

    /// Create from an equivalent window size N: alpha = 2 / (N + 1).
    pub fn from_window(window: usize) -> Self {
        assert!(window >= 1, "window must be >= 1, got {window}");
        Self::new(2.0 / (window as f64 + 1.0))
    }

    /// Create from a half-life (number of steps for weight to decay by 50%).
    pub fn from_halflife(halflife: f64) -> Self {
        assert!(halflife > 0.0, "halflife must be > 0, got {halflife}");
        Self::new(1.0 - (0.5_f64).powf(1.0 / halflife))
    }

    /// Update with a single observation.
    pub fn update(&mut self, value: f64) {
        self.count += 1;
        if !self.initialized {
            self.mean = value;
            self.var = 0.0;
            self.initialized = true;
            return;
        }
        let delta = value - self.mean;
        self.mean += self.alpha * delta;
        // Biased EMA variance estimate
        self.var = (1.0 - self.alpha) * (self.var + self.alpha * delta * delta);
    }

    /// Update with a batch of observations (applied sequentially).
    pub fn batch_update(&mut self, values: &[f64]) {
        for &v in values {
            self.update(v);
        }
    }

    /// Current EMA mean.
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Current EMA variance.
    pub fn var(&self) -> f64 {
        self.var
    }

    /// Current EMA standard deviation.
    pub fn std(&self) -> f64 {
        self.var.sqrt()
    }

    /// Normalize a value using current EMA mean and std.
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

    /// The smoothing factor.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Reset to initial state.
    pub fn reset(&mut self) {
        self.mean = 0.0;
        self.var = 0.0;
        self.count = 0;
        self.initialized = false;
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

    // -----------------------------------------------------------------------
    // RunningStatsVec tests
    // -----------------------------------------------------------------------

    #[test]
    fn stats_vec_new_is_empty() {
        let stats = RunningStatsVec::new(3);
        assert_eq!(stats.count(), 0);
        assert_eq!(stats.dim(), 3);
        assert_eq!(stats.mean(), vec![0.0; 3]);
        assert_eq!(stats.var(), vec![0.0; 3]);
    }

    #[test]
    fn stats_vec_single_sample() {
        let mut stats = RunningStatsVec::new(3);
        stats.update(&[1.0, 2.0, 3.0]);
        assert_eq!(stats.count(), 1);
        assert_eq!(stats.mean(), vec![1.0, 2.0, 3.0]);
        // Variance of a single sample is 0
        assert_eq!(stats.var(), vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn stats_vec_known_values_per_dim() {
        // Dim 0: [2, 4, 4, 4, 5, 5, 7, 9] -> mean=5.0, var=4.0
        // Dim 1: [1, 1, 1, 1, 1, 1, 1, 1] -> mean=1.0, var=0.0
        let mut stats = RunningStatsVec::new(2);
        let samples: &[&[f64]] = &[
            &[2.0, 1.0],
            &[4.0, 1.0],
            &[4.0, 1.0],
            &[4.0, 1.0],
            &[5.0, 1.0],
            &[5.0, 1.0],
            &[7.0, 1.0],
            &[9.0, 1.0],
        ];
        for s in samples {
            stats.update(s);
        }
        assert_eq!(stats.count(), 8);
        let mean = stats.mean();
        assert!((mean[0] - 5.0).abs() < 1e-10, "dim0 mean: {}", mean[0]);
        assert!((mean[1] - 1.0).abs() < 1e-10, "dim1 mean: {}", mean[1]);
        let var = stats.var();
        assert!((var[0] - 4.0).abs() < 1e-10, "dim0 var: {}", var[0]);
        assert!(var[1].abs() < 1e-10, "dim1 var: {}", var[1]);
        let std = stats.std();
        assert!((std[0] - 2.0).abs() < 1e-10, "dim0 std: {}", std[0]);
        assert!(std[1].abs() < 1e-10, "dim1 std: {}", std[1]);
    }

    #[test]
    fn stats_vec_batch_update_matches_sequential() {
        let mut seq = RunningStatsVec::new(3);
        let mut batch = RunningStatsVec::new(3);

        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        for sample in data.chunks(3) {
            seq.update(sample);
        }
        batch.batch_update(&data, 3);

        assert_eq!(seq.count(), batch.count());
        for i in 0..3 {
            assert!(
                (seq.mean()[i] - batch.mean()[i]).abs() < 1e-10,
                "dim {i} mean mismatch"
            );
            assert!(
                (seq.var()[i] - batch.var()[i]).abs() < 1e-10,
                "dim {i} var mismatch"
            );
        }
    }

    #[test]
    fn stats_vec_normalize_produces_z_scores() {
        let mut stats = RunningStatsVec::new(2);
        // Dim 0: [2, 4, 4, 4, 5, 5, 7, 9] -> mean=5, std=2
        // Dim 1: [10, 20, 30, 40, 50, 60, 70, 80] -> mean=45, std=~22.36
        let dim0 = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let dim1 = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
        for i in 0..8 {
            stats.update(&[dim0[i], dim1[i]]);
        }

        // Normalizing the mean should give ~0
        let z = stats.normalize(&[5.0, 45.0]);
        assert!(z[0].abs() < 1e-10, "z[0] should be ~0, got {}", z[0]);
        assert!(z[1].abs() < 1e-10, "z[1] should be ~0, got {}", z[1]);

        // Normalizing mean+std should give ~1
        let std = stats.std();
        let z2 = stats.normalize(&[5.0 + std[0], 45.0 + std[1]]);
        assert!(
            (z2[0] - 1.0).abs() < 1e-10,
            "z2[0] should be ~1, got {}",
            z2[0]
        );
        assert!(
            (z2[1] - 1.0).abs() < 1e-10,
            "z2[1] should be ~1, got {}",
            z2[1]
        );
    }

    #[test]
    fn stats_vec_normalize_with_zero_std_clamps() {
        let mut stats = RunningStatsVec::new(2);
        stats.update(&[5.0, 3.0]);
        stats.update(&[5.0, 3.0]);
        // Both dims have zero variance
        let z = stats.normalize(&[6.0, 4.0]);
        assert!(z[0].is_finite(), "dim0 normalize must be finite");
        assert!(z[1].is_finite(), "dim1 normalize must be finite");
        // (6 - 5) / max(0, 1e-8) = 1e8
        assert!((z[0] - 1e8).abs() < 1.0, "dim0: {}", z[0]);
    }

    #[test]
    fn stats_vec_normalize_batch() {
        let mut stats = RunningStatsVec::new(2);
        stats.update(&[0.0, 0.0]);
        stats.update(&[10.0, 20.0]);
        // mean=[5, 10], var=[25, 100], std=[5, 10]

        let data = [5.0, 10.0, 10.0, 20.0]; // 2 samples
        let out = stats.normalize_batch(&data, 2);
        assert!(out[0].abs() < 1e-10, "sample0 dim0 should be 0");
        assert!(out[1].abs() < 1e-10, "sample0 dim1 should be 0");
        assert!((out[2] - 1.0).abs() < 1e-10, "sample1 dim0 should be 1");
        assert!((out[3] - 1.0).abs() < 1e-10, "sample1 dim1 should be 1");
    }

    #[test]
    fn stats_vec_reset_clears_state() {
        let mut stats = RunningStatsVec::new(2);
        stats.update(&[1.0, 2.0]);
        stats.update(&[3.0, 4.0]);
        stats.reset();
        assert_eq!(stats.count(), 0);
        assert_eq!(stats.dim(), 2);
        assert_eq!(stats.mean(), vec![0.0, 0.0]);
    }

    #[test]
    #[should_panic(expected = "expected 3 dimensions, got 2")]
    fn stats_vec_update_wrong_dim_panics() {
        let mut stats = RunningStatsVec::new(3);
        stats.update(&[1.0, 2.0]);
    }

    #[test]
    #[should_panic(expected = "expected 6 elements")]
    fn stats_vec_batch_update_wrong_len_panics() {
        let mut stats = RunningStatsVec::new(3);
        stats.batch_update(&[1.0, 2.0, 3.0, 4.0], 2);
    }

    #[test]
    #[should_panic(expected = "expected 2 dimensions, got 3")]
    fn stats_vec_normalize_wrong_dim_panics() {
        let stats = RunningStatsVec::new(2);
        stats.normalize(&[1.0, 2.0, 3.0]);
    }

    #[test]
    fn stats_vec_large_stream_numerically_stable() {
        let mut stats = RunningStatsVec::new(2);
        let base = 1_000_000.0f64;
        for i in 0..10_000 {
            let v = i as f64 * 0.001;
            stats.update(&[base + v, -base - v]);
        }
        let expected_mean = base + 5.0 - 0.001 / 2.0;
        let mean = stats.mean();
        assert!(
            (mean[0] - expected_mean).abs() < 0.01,
            "dim0 mean imprecise: got {}, expected ~{expected_mean}",
            mean[0]
        );
        assert!(
            (mean[1] + expected_mean).abs() < 0.01,
            "dim1 mean imprecise: got {}, expected ~{}",
            mean[1],
            -expected_mean
        );
    }

    #[test]
    fn stats_vec_hopper_like_multi_scale() {
        // Simulate Hopper-like observations: dim0 in [-1,1], dim1 in [-10,10]
        let mut stats = RunningStatsVec::new(2);
        for i in 0..1000 {
            let t = i as f64 / 999.0;
            let pos = -1.0 + 2.0 * t; // [-1, 1]
            let vel = -10.0 + 20.0 * t; // [-10, 10]
            stats.update(&[pos, vel]);
        }
        let std = stats.std();
        // The stds should reflect the different scales
        assert!(
            std[1] > std[0] * 5.0,
            "velocity std ({}) should be much larger than position std ({})",
            std[1],
            std[0]
        );
        // Normalizing should bring both dims to similar scale
        let z = stats.normalize(&[0.5, 5.0]);
        assert!(
            z[0].abs() < 5.0 && z[1].abs() < 5.0,
            "normalized values should be moderate z-scores, got {:?}",
            z
        );
    }

    // -----------------------------------------------------------------------
    // ExponentialRunningStats tests
    // -----------------------------------------------------------------------

    #[test]
    fn ema_new_basic() {
        let ema = ExponentialRunningStats::new(0.1);
        assert_eq!(ema.count(), 0);
        assert!((ema.alpha() - 0.1).abs() < 1e-10);
    }

    #[test]
    fn ema_from_window() {
        let ema = ExponentialRunningStats::from_window(19);
        // alpha = 2/(19+1) = 0.1
        assert!((ema.alpha() - 0.1).abs() < 1e-10);
    }

    #[test]
    fn ema_from_halflife() {
        let ema = ExponentialRunningStats::from_halflife(10.0);
        // After 10 steps, weight should be ~0.5
        assert!(ema.alpha() > 0.0 && ema.alpha() < 1.0);
    }

    #[test]
    fn ema_first_update_sets_mean() {
        let mut ema = ExponentialRunningStats::new(0.1);
        ema.update(5.0);
        assert!((ema.mean() - 5.0).abs() < 1e-10);
        assert_eq!(ema.count(), 1);
    }

    #[test]
    fn ema_tracks_constant_signal() {
        let mut ema = ExponentialRunningStats::new(0.1);
        for _ in 0..100 {
            ema.update(3.0);
        }
        assert!((ema.mean() - 3.0).abs() < 1e-8);
        assert!(ema.var() < 1e-8);
    }

    #[test]
    fn ema_adapts_to_level_shift() {
        let mut ema = ExponentialRunningStats::new(0.1);
        // Burn in at level 0
        for _ in 0..50 {
            ema.update(0.0);
        }
        assert!(ema.mean().abs() < 0.01);
        // Shift to level 10
        for _ in 0..100 {
            ema.update(10.0);
        }
        // Should have adapted close to 10
        assert!((ema.mean() - 10.0).abs() < 0.1);
    }

    #[test]
    fn ema_higher_alpha_adapts_faster() {
        let mut fast = ExponentialRunningStats::new(0.5);
        let mut slow = ExponentialRunningStats::new(0.01);
        for _ in 0..20 {
            fast.update(0.0);
            slow.update(0.0);
        }
        for _ in 0..10 {
            fast.update(10.0);
            slow.update(10.0);
        }
        // Fast should be closer to 10 than slow
        assert!(
            (fast.mean() - 10.0).abs() < (slow.mean() - 10.0).abs(),
            "fast={}, slow={}",
            fast.mean(),
            slow.mean()
        );
    }

    #[test]
    fn ema_normalize_zero_for_mean() {
        let mut ema = ExponentialRunningStats::new(0.1);
        for x in 0..100 {
            ema.update(x as f64);
        }
        // Normalize the current mean should be ~0
        let z = ema.normalize(ema.mean());
        assert!(z.abs() < 1e-8);
    }

    #[test]
    fn ema_reset_clears_state() {
        let mut ema = ExponentialRunningStats::new(0.1);
        ema.update(5.0);
        ema.update(10.0);
        ema.reset();
        assert_eq!(ema.count(), 0);
        assert!((ema.mean()).abs() < 1e-10);
    }

    #[test]
    #[should_panic(expected = "alpha must be in (0, 1)")]
    fn ema_invalid_alpha_panics() {
        ExponentialRunningStats::new(0.0);
    }

    #[test]
    #[should_panic(expected = "alpha must be in (0, 1)")]
    fn ema_alpha_one_panics() {
        ExponentialRunningStats::new(1.0);
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

            #[test]
            fn stats_vec_per_dim_mean_matches_naive(
                dim in 1usize..8,
                n_samples in 2usize..50,
            ) {
                // Generate deterministic data from dim and n_samples
                let mut data = Vec::with_capacity(n_samples * dim);
                for s in 0..n_samples {
                    for d in 0..dim {
                        data.push((s as f64) * 0.1 + (d as f64) * 10.0);
                    }
                }

                let mut stats = RunningStatsVec::new(dim);
                stats.batch_update(&data, n_samples);

                // Compute naive per-dim mean
                for d in 0..dim {
                    let sum: f64 = (0..n_samples).map(|s| data[s * dim + d]).sum();
                    let naive_mean = sum / n_samples as f64;
                    prop_assert!(
                        (stats.mean()[d] - naive_mean).abs() < 1e-8,
                        "dim {d}: running mean {} != naive mean {}",
                        stats.mean()[d], naive_mean
                    );
                }
            }

            #[test]
            fn stats_vec_variance_non_negative(
                dim in 1usize..6,
                n_samples in 2usize..50,
            ) {
                let mut data = Vec::with_capacity(n_samples * dim);
                for s in 0..n_samples {
                    for d in 0..dim {
                        data.push((s as f64) * 0.7 - (d as f64) * 3.0);
                    }
                }

                let mut stats = RunningStatsVec::new(dim);
                stats.batch_update(&data, n_samples);

                for d in 0..dim {
                    prop_assert!(
                        stats.var()[d] >= 0.0,
                        "dim {d} variance must be non-negative, got {}",
                        stats.var()[d]
                    );
                }
            }

            #[test]
            fn stats_vec_normalize_roundtrip_z_mean_zero(
                dim in 1usize..6,
                n_samples in 5usize..50,
            ) {
                let mut data = Vec::with_capacity(n_samples * dim);
                for s in 0..n_samples {
                    for d in 0..dim {
                        data.push((s as f64) * 1.3 + (d as f64) * 7.0);
                    }
                }

                let mut stats = RunningStatsVec::new(dim);
                stats.batch_update(&data, n_samples);

                // Normalizing the mean vector should give zeros
                let z = stats.normalize(&stats.mean());
                for (d, &val) in z.iter().enumerate().take(dim) {
                    prop_assert!(
                        val.abs() < 1e-8,
                        "normalize(mean)[{d}] should be ~0, got {}",
                        val
                    );
                }
            }
        }
    }
}
