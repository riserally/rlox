use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::error::RloxError;

use super::ExperienceRecord;

/// Fixed-capacity ring buffer with uniform random sampling.
///
/// Pre-allocates all arrays at construction for zero-allocation push.
/// Oldest transitions are overwritten when capacity is reached.
pub struct ReplayBuffer {
    obs_dim: usize,
    act_dim: usize,
    capacity: usize,
    observations: Vec<f32>,
    actions: Vec<f32>,
    rewards: Vec<f32>,
    terminated: Vec<bool>,
    truncated: Vec<bool>,
    write_pos: usize,
    count: usize,
}

/// A sampled batch of transitions. Owns its data (copied from the ring buffer).
pub struct SampledBatch {
    pub observations: Vec<f32>,
    pub actions: Vec<f32>,
    pub rewards: Vec<f32>,
    pub terminated: Vec<bool>,
    pub truncated: Vec<bool>,
    pub obs_dim: usize,
    pub act_dim: usize,
    pub batch_size: usize,
}

impl SampledBatch {
    fn with_capacity(batch_size: usize, obs_dim: usize, act_dim: usize) -> Self {
        Self {
            observations: Vec::with_capacity(batch_size * obs_dim),
            actions: Vec::with_capacity(batch_size * act_dim),
            rewards: Vec::with_capacity(batch_size),
            terminated: Vec::with_capacity(batch_size),
            truncated: Vec::with_capacity(batch_size),
            obs_dim,
            act_dim,
            batch_size: 0,
        }
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
            actions: vec![0.0; capacity * act_dim],
            rewards: vec![0.0; capacity],
            terminated: vec![false; capacity],
            truncated: vec![false; capacity],
            write_pos: 0,
            count: 0,
        }
    }

    /// Number of valid transitions currently stored.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Push a transition, overwriting the oldest if at capacity.
    pub fn push(&mut self, record: ExperienceRecord) -> Result<(), RloxError> {
        if record.obs.len() != self.obs_dim {
            return Err(RloxError::ShapeMismatch {
                expected: format!("obs_dim={}", self.obs_dim),
                got: format!("obs.len()={}", record.obs.len()),
            });
        }
        let idx = self.write_pos;
        let obs_start = idx * self.obs_dim;
        self.observations[obs_start..obs_start + self.obs_dim]
            .copy_from_slice(&record.obs);
        self.actions[idx] = record.action;
        self.rewards[idx] = record.reward;
        self.terminated[idx] = record.terminated;
        self.truncated[idx] = record.truncated;

        self.write_pos = (self.write_pos + 1) % self.capacity;
        if self.count < self.capacity {
            self.count += 1;
        }
        Ok(())
    }

    /// Sample a batch of transitions uniformly at random.
    ///
    /// Uses ChaCha8Rng seeded with `seed` for deterministic cross-platform
    /// reproducibility. Returns owned `SampledBatch`.
    pub fn sample(&self, batch_size: usize, seed: u64) -> Result<SampledBatch, RloxError> {
        if batch_size > self.count {
            return Err(RloxError::BufferError(format!(
                "batch_size {} > buffer len {}",
                batch_size, self.count
            )));
        }
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut batch = SampledBatch::with_capacity(batch_size, self.obs_dim, self.act_dim);

        for _ in 0..batch_size {
            let idx = rng.random_range(0..self.count);
            let obs_start = idx * self.obs_dim;
            batch
                .observations
                .extend_from_slice(&self.observations[obs_start..obs_start + self.obs_dim]);
            batch.actions.push(self.actions[idx]);
            batch.rewards.push(self.rewards[idx]);
            batch.terminated.push(self.terminated[idx]);
            batch.truncated.push(self.truncated[idx]);
        }
        batch.batch_size = batch_size;
        Ok(batch)
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
}
