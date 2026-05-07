use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::error::RloxError;

use super::extra_columns::{ColumnHandle, ExtraColumns};
use super::ExperienceRecord;

/// Fixed-capacity ring buffer with uniform random sampling.
///
/// Pre-allocates all arrays at construction for zero-allocation push.
/// Oldest transitions are overwritten when capacity is reached.
///
/// Supports optional extra f32 columns (e.g. log-probs, value estimates)
/// via [`ColumnHandle`]. When no extra columns are registered, there is
/// zero overhead — no allocations and no branches in the hot push/sample path.
#[derive(Debug)]
pub struct ReplayBuffer {
    obs_dim: usize,
    act_dim: usize,
    capacity: usize,
    observations: Vec<f32>,
    next_observations: Vec<f32>,
    actions: Vec<f32>,
    rewards: Vec<f32>,
    terminated: Vec<bool>,
    truncated: Vec<bool>,
    write_pos: usize,
    count: usize,
    extra: ExtraColumns,
}

/// A sampled batch of transitions. Owns its data (copied from the ring buffer).
#[derive(Debug, Clone)]
pub struct SampledBatch {
    pub observations: Vec<f32>,
    pub next_observations: Vec<f32>,
    pub actions: Vec<f32>,
    pub rewards: Vec<f32>,
    pub terminated: Vec<bool>,
    pub truncated: Vec<bool>,
    pub obs_dim: usize,
    pub act_dim: usize,
    pub batch_size: usize,
    /// Extra column data, populated only when columns are registered.
    /// Each entry is `(column_name, flat_data)` where `flat_data` has
    /// length `batch_size * column_dim`.
    pub extra: Vec<(String, Vec<f32>)>,
}

impl SampledBatch {
    pub fn with_capacity(batch_size: usize, obs_dim: usize, act_dim: usize) -> Self {
        Self {
            observations: Vec::with_capacity(batch_size * obs_dim),
            next_observations: Vec::with_capacity(batch_size * obs_dim),
            actions: Vec::with_capacity(batch_size * act_dim),
            rewards: Vec::with_capacity(batch_size),
            terminated: Vec::with_capacity(batch_size),
            truncated: Vec::with_capacity(batch_size),
            obs_dim,
            act_dim,
            batch_size: 0,
            extra: Vec::new(),
        }
    }

    /// Clear all data but retain allocated capacity for reuse.
    ///
    /// Note: `extra` is cleared entirely (the outer Vec). If you alternate
    /// between buffers with different extra-column schemas, the inner Vecs'
    /// capacity is lost. This is acceptable because cross-buffer reuse of
    /// extra columns is uncommon.
    pub fn clear(&mut self) {
        self.observations.clear();
        self.next_observations.clear();
        self.actions.clear();
        self.rewards.clear();
        self.terminated.clear();
        self.truncated.clear();
        self.extra.clear();
        self.batch_size = 0;
    }
}

impl ReplayBuffer {
    /// Create a ring buffer with fixed capacity. All arrays are pre-allocated.
    pub fn new(capacity: usize, obs_dim: usize, act_dim: usize) -> Self {
        Self {
            obs_dim,
            act_dim,
            capacity,
            observations: vec![0.0; capacity * obs_dim],
            next_observations: vec![0.0; capacity * obs_dim],
            actions: vec![0.0; capacity * act_dim],
            rewards: vec![0.0; capacity],
            terminated: vec![false; capacity],
            truncated: vec![false; capacity],
            write_pos: 0,
            count: 0,
            extra: ExtraColumns::new(),
        }
    }

    /// Register an extra f32 column with the given name and dimensionality.
    ///
    /// Returns a [`ColumnHandle`] for O(1) push/sample access.
    /// Must be called before any `push()` — the column is pre-allocated to
    /// match the buffer's capacity.
    pub fn register_column(&mut self, name: &str, dim: usize) -> ColumnHandle {
        let handle = self.extra.register(name, dim);
        self.extra.allocate(self.capacity);
        handle
    }

    /// Push extra column data for the most recently pushed transition.
    ///
    /// Must be called *after* `push()` and before the next `push()`.
    /// The `values` slice length must match the column's registered dim.
    pub fn push_extra(&mut self, handle: ColumnHandle, values: &[f32]) -> Result<(), RloxError> {
        if self.count == 0 {
            return Err(RloxError::BufferError(
                "push_extra called before any push()".into(),
            ));
        }
        // The most recently written position is one step behind write_pos
        let pos = if self.write_pos == 0 {
            self.capacity - 1
        } else {
            self.write_pos - 1
        };
        self.extra.push(handle, pos, values)
    }

    /// Observation dimensionality.
    pub fn obs_dim(&self) -> usize {
        self.obs_dim
    }

    /// Action dimensionality.
    pub fn act_dim(&self) -> usize {
        self.act_dim
    }

    /// Number of valid transitions currently stored.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Current write position in the ring buffer.
    pub(crate) fn write_pos(&self) -> usize {
        self.write_pos
    }

    /// Access the record at `idx` by reference.
    ///
    /// Returns `(obs_slice, next_obs_slice, action_slice, reward, terminated, truncated)`.
    ///
    /// # Panics
    ///
    /// Panics if `idx >= self.count`.
    pub(crate) fn get(&self, idx: usize) -> (&[f32], &[f32], &[f32], f32, bool, bool) {
        assert!(
            idx < self.count,
            "index {idx} out of bounds (count={})",
            self.count
        );
        let obs_start = idx * self.obs_dim;
        let act_start = idx * self.act_dim;
        (
            &self.observations[obs_start..obs_start + self.obs_dim],
            &self.next_observations[obs_start..obs_start + self.obs_dim],
            &self.actions[act_start..act_start + self.act_dim],
            self.rewards[idx],
            self.terminated[idx],
            self.truncated[idx],
        )
    }

    /// Push a transition from borrowed slices, avoiding intermediate allocation.
    pub fn push_slices(
        &mut self,
        obs: &[f32],
        next_obs: &[f32],
        action: &[f32],
        reward: f32,
        terminated: bool,
        truncated: bool,
    ) -> Result<(), RloxError> {
        if obs.len() != self.obs_dim {
            return Err(RloxError::ShapeMismatch {
                expected: format!("obs_dim={}", self.obs_dim),
                got: format!("obs.len()={}", obs.len()),
            });
        }
        if next_obs.len() != self.obs_dim {
            return Err(RloxError::ShapeMismatch {
                expected: format!("obs_dim={}", self.obs_dim),
                got: format!("next_obs.len()={}", next_obs.len()),
            });
        }
        if action.len() != self.act_dim {
            return Err(RloxError::ShapeMismatch {
                expected: format!("act_dim={}", self.act_dim),
                got: format!("action.len()={}", action.len()),
            });
        }
        let idx = self.write_pos;
        let obs_start = idx * self.obs_dim;
        self.observations[obs_start..obs_start + self.obs_dim].copy_from_slice(obs);
        self.next_observations[obs_start..obs_start + self.obs_dim].copy_from_slice(next_obs);
        let act_start = idx * self.act_dim;
        self.actions[act_start..act_start + self.act_dim].copy_from_slice(action);
        self.rewards[idx] = reward;
        self.terminated[idx] = terminated;
        self.truncated[idx] = truncated;

        self.write_pos = (self.write_pos + 1) % self.capacity;
        if self.count < self.capacity {
            self.count += 1;
        }
        Ok(())
    }

    /// Push multiple transitions at once from flat arrays.
    ///
    /// `obs_batch` shape: `[n * obs_dim]`, `next_obs_batch`: same,
    /// `actions_batch`: `[n * act_dim]`, others: `[n]`.
    pub fn push_batch(
        &mut self,
        obs_batch: &[f32],
        next_obs_batch: &[f32],
        actions_batch: &[f32],
        rewards: &[f32],
        terminated: &[bool],
        truncated: &[bool],
    ) -> Result<(), RloxError> {
        let n = rewards.len();
        if obs_batch.len() != n * self.obs_dim
            || next_obs_batch.len() != n * self.obs_dim
            || actions_batch.len() != n * self.act_dim
            || terminated.len() != n
            || truncated.len() != n
        {
            return Err(RloxError::ShapeMismatch {
                expected: format!("n={n}, obs_dim={}, act_dim={}", self.obs_dim, self.act_dim),
                got: format!(
                    "obs={}, next_obs={}, act={}, rew={}, term={}, trunc={}",
                    obs_batch.len(),
                    next_obs_batch.len(),
                    actions_batch.len(),
                    rewards.len(),
                    terminated.len(),
                    truncated.len()
                ),
            });
        }
        for i in 0..n {
            let obs = &obs_batch[i * self.obs_dim..(i + 1) * self.obs_dim];
            let next_obs = &next_obs_batch[i * self.obs_dim..(i + 1) * self.obs_dim];
            let action = &actions_batch[i * self.act_dim..(i + 1) * self.act_dim];
            self.push_slices(
                obs,
                next_obs,
                action,
                rewards[i],
                terminated[i],
                truncated[i],
            )?;
        }
        Ok(())
    }

    /// Push a transition, overwriting the oldest if at capacity.
    ///
    /// Prefer [`push_slices`](Self::push_slices) to avoid the intermediate
    /// `Vec<f32>` allocations inside `ExperienceRecord`.
    pub fn push(&mut self, record: ExperienceRecord) -> Result<(), RloxError> {
        self.push_slices(
            &record.obs,
            &record.next_obs,
            &record.action,
            record.reward,
            record.terminated,
            record.truncated,
        )
    }

    /// Sample a batch of transitions uniformly at random.
    ///
    /// Uses ChaCha8Rng seeded with `seed` for deterministic cross-platform
    /// reproducibility. Returns owned `SampledBatch`.
    ///
    /// If extra columns have been registered, their data is included in
    /// `SampledBatch::extra`.
    pub fn sample(&self, batch_size: usize, seed: u64) -> Result<SampledBatch, RloxError> {
        if batch_size > self.count {
            return Err(RloxError::BufferError(format!(
                "batch_size {} > buffer len {}",
                batch_size, self.count
            )));
        }
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut batch = SampledBatch::with_capacity(batch_size, self.obs_dim, self.act_dim);

        let has_extra = self.extra.num_columns() > 0;
        let mut indices = if has_extra {
            Vec::with_capacity(batch_size)
        } else {
            Vec::new()
        };

        for _ in 0..batch_size {
            let idx = rng.random_range(0..self.count);
            let obs_start = idx * self.obs_dim;
            batch
                .observations
                .extend_from_slice(&self.observations[obs_start..obs_start + self.obs_dim]);
            batch
                .next_observations
                .extend_from_slice(&self.next_observations[obs_start..obs_start + self.obs_dim]);
            let act_start = idx * self.act_dim;
            batch
                .actions
                .extend_from_slice(&self.actions[act_start..act_start + self.act_dim]);
            batch.rewards.push(self.rewards[idx]);
            batch.terminated.push(self.terminated[idx]);
            batch.truncated.push(self.truncated[idx]);

            if has_extra {
                indices.push(idx);
            }
        }
        batch.batch_size = batch_size;

        if has_extra {
            batch.extra = self.extra.sample_all(&indices);
        }

        Ok(batch)
    }

    /// Sample into a pre-allocated batch, reusing its capacity.
    ///
    /// Same as `sample()` but avoids allocation by reusing `batch`.
    pub fn sample_into(
        &self,
        batch: &mut SampledBatch,
        batch_size: usize,
        seed: u64,
    ) -> Result<(), RloxError> {
        if batch_size > self.count {
            return Err(RloxError::BufferError(format!(
                "batch_size {} > buffer len {}",
                batch_size, self.count
            )));
        }
        batch.clear();
        batch.obs_dim = self.obs_dim;
        batch.act_dim = self.act_dim;

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let has_extra = self.extra.num_columns() > 0;
        let mut indices = if has_extra {
            Vec::with_capacity(batch_size)
        } else {
            Vec::new()
        };

        for _ in 0..batch_size {
            let idx = rng.random_range(0..self.count);
            let obs_start = idx * self.obs_dim;
            batch
                .observations
                .extend_from_slice(&self.observations[obs_start..obs_start + self.obs_dim]);
            batch
                .next_observations
                .extend_from_slice(&self.next_observations[obs_start..obs_start + self.obs_dim]);
            let act_start = idx * self.act_dim;
            batch
                .actions
                .extend_from_slice(&self.actions[act_start..act_start + self.act_dim]);
            batch.rewards.push(self.rewards[idx]);
            batch.terminated.push(self.terminated[idx]);
            batch.truncated.push(self.truncated[idx]);
            if has_extra {
                indices.push(idx);
            }
        }
        batch.batch_size = batch_size;
        if has_extra {
            batch.extra = self.extra.sample_all(&indices);
        }
        Ok(())
    }

    /// Sample uniformly from only the most recent `window_size` transitions.
    ///
    /// This implements a sliding window replay strategy for non-stationary RL:
    /// only recent experience is used for training, discarding stale transitions
    /// from previous MDP regimes.
    ///
    /// # Parameters
    /// - `batch_size`: number of transitions to sample
    /// - `window_size`: only consider the most recent `window_size` transitions
    /// - `seed`: RNG seed for reproducibility
    ///
    /// # Errors
    /// Returns an error if `batch_size > min(window_size, self.count)`.
    pub fn sample_recent(
        &self,
        batch_size: usize,
        window_size: usize,
        seed: u64,
    ) -> Result<SampledBatch, RloxError> {
        let effective_window = window_size.min(self.count);
        if batch_size > effective_window {
            return Err(RloxError::BufferError(format!(
                "batch_size {} > effective window {} (window_size={}, count={})",
                batch_size, effective_window, window_size, self.count
            )));
        }

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut batch = SampledBatch::with_capacity(batch_size, self.obs_dim, self.act_dim);

        let has_extra = self.extra.num_columns() > 0;
        let mut indices = if has_extra {
            Vec::with_capacity(batch_size)
        } else {
            Vec::new()
        };

        for _ in 0..batch_size {
            // Map random offset within window to actual ring buffer index.
            // The most recent transition is at (write_pos - 1) mod capacity,
            // the second most recent at (write_pos - 2) mod capacity, etc.
            let offset = rng.random_range(0..effective_window);
            let idx = if self.write_pos > offset {
                self.write_pos - 1 - offset
            } else {
                // Wrap around the ring
                self.capacity - 1 - (offset - self.write_pos)
            };

            let obs_start = idx * self.obs_dim;
            batch
                .observations
                .extend_from_slice(&self.observations[obs_start..obs_start + self.obs_dim]);
            batch
                .next_observations
                .extend_from_slice(&self.next_observations[obs_start..obs_start + self.obs_dim]);
            let act_start = idx * self.act_dim;
            batch
                .actions
                .extend_from_slice(&self.actions[act_start..act_start + self.act_dim]);
            batch.rewards.push(self.rewards[idx]);
            batch.terminated.push(self.terminated[idx]);
            batch.truncated.push(self.truncated[idx]);

            if has_extra {
                indices.push(idx);
            }
        }
        batch.batch_size = batch_size;

        if has_extra {
            batch.extra = self.extra.sample_all(&indices);
        }

        Ok(batch)
    }

    /// Access the extra columns storage (for advanced use / testing).
    pub fn extra_columns(&self) -> &ExtraColumns {
        &self.extra
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::sample_record;

    #[test]
    fn ring_buffer_respects_capacity() {
        let mut buf = ReplayBuffer::new(100, 4, 1);
        for _ in 0..200 {
            buf.push(sample_record(4)).unwrap();
        }
        assert_eq!(buf.len(), 100);
    }

    #[test]
    fn ring_buffer_overwrites_oldest() {
        let mut buf = ReplayBuffer::new(3, 4, 1);
        for i in 0..5 {
            let mut r = sample_record(4);
            r.reward = i as f32;
            buf.push(r).unwrap();
        }
        // Should contain rewards 2.0, 3.0, 4.0
        let batch = buf.sample(3, 42).unwrap();
        assert!(!batch.rewards.contains(&0.0));
        assert!(!batch.rewards.contains(&1.0));
    }

    #[test]
    fn sample_returns_requested_size() {
        let mut buf = ReplayBuffer::new(1000, 4, 1);
        for _ in 0..1000 {
            buf.push(sample_record(4)).unwrap();
        }
        let batch = buf.sample(64, 42).unwrap();
        assert_eq!(batch.batch_size, 64);
        assert_eq!(batch.observations.len(), 64 * 4);
    }

    #[test]
    fn sample_errors_when_too_few() {
        let mut buf = ReplayBuffer::new(100, 4, 1);
        buf.push(sample_record(4)).unwrap();
        assert!(buf.sample(32, 42).is_err());
    }

    #[test]
    fn sample_is_deterministic_with_same_seed() {
        let mut buf = ReplayBuffer::new(1000, 4, 1);
        for _ in 0..1000 {
            buf.push(sample_record(4)).unwrap();
        }
        let b1 = buf.sample(32, 42).unwrap();
        let b2 = buf.sample(32, 42).unwrap();
        assert_eq!(b1.observations, b2.observations);
        assert_eq!(b1.rewards, b2.rewards);
    }

    #[test]
    fn replay_buffer_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ReplayBuffer>();
    }

    #[test]
    fn empty_buffer_has_zero_len() {
        let buf = ReplayBuffer::new(100, 4, 1);
        assert_eq!(buf.len(), 0);
        assert!(buf.is_empty());
    }

    #[test]
    fn test_replay_buffer_next_obs_roundtrip() {
        let obs_dim = 4;
        let mut buf = ReplayBuffer::new(100, obs_dim, 1);
        let record = ExperienceRecord {
            obs: vec![1.0; obs_dim],
            next_obs: vec![2.0, 3.0, 4.0, 5.0],
            action: vec![0.0],
            reward: 1.0,
            terminated: false,
            truncated: false,
        };
        buf.push(record).unwrap();
        let batch = buf.sample(1, 42).unwrap();
        assert_eq!(&batch.next_observations, &[2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_replay_buffer_next_obs_shape() {
        let obs_dim = 4;
        let mut buf = ReplayBuffer::new(1000, obs_dim, 1);
        for _ in 0..100 {
            buf.push(sample_record(obs_dim)).unwrap();
        }
        let batch = buf.sample(32, 42).unwrap();
        assert_eq!(batch.next_observations.len(), 32 * obs_dim);
    }

    #[test]
    fn test_replay_buffer_next_obs_dim_mismatch_errors() {
        let mut buf = ReplayBuffer::new(100, 4, 1);
        let record = ExperienceRecord {
            obs: vec![1.0; 4],
            next_obs: vec![2.0; 3], // wrong dim
            action: vec![0.0],
            reward: 1.0,
            terminated: false,
            truncated: false,
        };
        let result = buf.push(record);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("next_obs"));
    }

    #[test]
    fn test_replay_buffer_with_extra_columns_roundtrip() {
        let mut buf = ReplayBuffer::new(100, 4, 1);
        let lp = buf.register_column("log_prob", 1);
        let val = buf.register_column("value", 1);

        for i in 0..10 {
            buf.push(sample_record(4)).unwrap();
            buf.push_extra(lp, &[i as f32 * 0.1]).unwrap();
            buf.push_extra(val, &[i as f32]).unwrap();
        }

        let batch = buf.sample(5, 42).unwrap();
        assert_eq!(batch.extra.len(), 2);
        assert_eq!(batch.extra[0].0, "log_prob");
        assert_eq!(batch.extra[0].1.len(), 5); // batch_size * dim(1)
        assert_eq!(batch.extra[1].0, "value");
        assert_eq!(batch.extra[1].1.len(), 5);
    }

    #[test]
    fn test_replay_buffer_no_extra_columns_has_empty_extra() {
        let mut buf = ReplayBuffer::new(100, 4, 1);
        for _ in 0..10 {
            buf.push(sample_record(4)).unwrap();
        }
        let batch = buf.sample(5, 42).unwrap();
        assert!(batch.extra.is_empty());
    }

    #[test]
    fn test_push_extra_before_push_errors() {
        let mut buf = ReplayBuffer::new(100, 4, 1);
        let h = buf.register_column("test", 1);
        let result = buf.push_extra(h, &[1.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_extra_columns_multidim_roundtrip() {
        let mut buf = ReplayBuffer::new(100, 4, 1);
        let h = buf.register_column("action_mean", 3);

        for i in 0..5 {
            buf.push(sample_record(4)).unwrap();
            let v = i as f32;
            buf.push_extra(h, &[v, v + 1.0, v + 2.0]).unwrap();
        }

        let batch = buf.sample(3, 42).unwrap();
        assert_eq!(batch.extra.len(), 1);
        assert_eq!(batch.extra[0].0, "action_mean");
        assert_eq!(batch.extra[0].1.len(), 9); // 3 * 3
    }

    #[test]
    fn test_sample_into_matches_sample() {
        let mut buf = ReplayBuffer::new(100, 4, 1);
        for _ in 0..50 {
            buf.push(sample_record(4)).unwrap();
        }

        let batch1 = buf.sample(16, 42).unwrap();
        let mut reusable = SampledBatch::with_capacity(16, 4, 1);
        buf.sample_into(&mut reusable, 16, 42).unwrap();

        assert_eq!(batch1.observations, reusable.observations);
        assert_eq!(batch1.next_observations, reusable.next_observations);
        assert_eq!(batch1.actions, reusable.actions);
        assert_eq!(batch1.rewards, reusable.rewards);
        assert_eq!(batch1.terminated, reusable.terminated);
        assert_eq!(batch1.batch_size, reusable.batch_size);
    }

    #[test]
    fn test_sample_into_reuses_capacity() {
        let mut buf = ReplayBuffer::new(100, 4, 1);
        for _ in 0..50 {
            buf.push(sample_record(4)).unwrap();
        }

        let mut batch = SampledBatch::with_capacity(16, 4, 1);
        buf.sample_into(&mut batch, 16, 1).unwrap();
        let obs_cap = batch.observations.capacity();

        // Second sample_into should reuse capacity, not shrink
        buf.sample_into(&mut batch, 16, 2).unwrap();
        assert!(batch.observations.capacity() >= obs_cap);
    }

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn ring_buffer_never_exceeds_capacity(capacity in 1..500usize, num_pushes in 0..2000usize) {
                let mut buf = ReplayBuffer::new(capacity, 4, 1);
                for _ in 0..num_pushes {
                    buf.push(sample_record(4)).unwrap();
                }
                prop_assert!(buf.len() <= capacity);
            }

            #[test]
            fn ring_buffer_len_is_min_of_pushes_and_capacity(capacity in 1..500usize, num_pushes in 0..2000usize) {
                let mut buf = ReplayBuffer::new(capacity, 4, 1);
                for _ in 0..num_pushes {
                    buf.push(sample_record(4)).unwrap();
                }
                prop_assert_eq!(buf.len(), num_pushes.min(capacity));
            }

            #[test]
            fn sample_returns_requested_size_prop(capacity in 10..500usize, num_pushes in 10..2000usize, batch_size in 1..50usize) {
                let mut buf = ReplayBuffer::new(capacity, 4, 1);
                for _ in 0..num_pushes {
                    buf.push(sample_record(4)).unwrap();
                }
                let effective_batch = batch_size.min(buf.len());
                let batch = buf.sample(effective_batch, 42).unwrap();
                prop_assert_eq!(batch.batch_size, effective_batch);
            }
        }
    }
}
