//! GPU-friendly flat batch sampling for replay buffers.
//!
//! Provides [`FlatBatch`] — a struct of contiguous `Vec<f32>` arrays ready
//! for zero-copy transfer to GPU via PyTorch's `torch.from_blob` or
//! `pin_memory()`.
//!
//! Unlike [`SampledBatch`](super::ringbuf::SampledBatch), `FlatBatch` stores
//! terminated/truncated as `f32` (0.0/1.0) so every field is a uniform f32
//! buffer — no bool→numpy conversion needed on the Python side.
//!
//! Gated behind the `gpu` feature flag.

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::error::RloxError;
use super::ringbuf::ReplayBuffer;

/// A sampled batch stored as flat contiguous `f32` arrays.
///
/// Every field is a single contiguous allocation suitable for direct
/// transfer to GPU memory. Booleans are encoded as `f32` (0.0 / 1.0).
#[derive(Debug, Clone)]
pub struct FlatBatch {
    /// Observations, shape `[batch_size * obs_dim]`.
    pub obs: Vec<f32>,
    /// Next observations, shape `[batch_size * obs_dim]`.
    pub next_obs: Vec<f32>,
    /// Actions, shape `[batch_size * act_dim]`.
    pub actions: Vec<f32>,
    /// Rewards, shape `[batch_size]`.
    pub rewards: Vec<f32>,
    /// Terminated flags as f32 (0.0 or 1.0), shape `[batch_size]`.
    pub terminated: Vec<f32>,
    /// Truncated flags as f32 (0.0 or 1.0), shape `[batch_size]`.
    pub truncated: Vec<f32>,
    /// Number of transitions in this batch.
    pub batch_size: usize,
    /// Observation dimensionality.
    pub obs_dim: usize,
    /// Action dimensionality.
    pub act_dim: usize,
}

impl ReplayBuffer {
    /// Sample a batch and return flat contiguous `f32` arrays ready for GPU transfer.
    ///
    /// Unlike [`sample`](Self::sample), this method encodes booleans as `f32`
    /// and skips extra columns — optimized for the hot path of shipping data
    /// to a GPU where only the core transition tuple is needed.
    ///
    /// Uses `ChaCha8Rng` seeded with `seed` for deterministic cross-platform
    /// reproducibility.
    pub fn sample_flat(
        &self,
        batch_size: usize,
        seed: u64,
    ) -> Result<FlatBatch, RloxError> {
        if self.is_empty() {
            return Err(RloxError::BufferError(
                "cannot sample from empty buffer".into(),
            ));
        }
        if batch_size > self.len() {
            return Err(RloxError::BufferError(format!(
                "batch_size {} > buffer len {}",
                batch_size,
                self.len()
            )));
        }

        let obs_dim = self.obs_dim();
        let act_dim = self.act_dim();

        let mut obs = Vec::with_capacity(batch_size * obs_dim);
        let mut next_obs = Vec::with_capacity(batch_size * obs_dim);
        let mut actions = Vec::with_capacity(batch_size * act_dim);
        let mut rewards = Vec::with_capacity(batch_size);
        let mut terminated = Vec::with_capacity(batch_size);
        let mut truncated = Vec::with_capacity(batch_size);

        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        for _ in 0..batch_size {
            let idx = rng.random_range(0..self.len());
            let (o, no, a, r, t, tr) = self.get(idx);

            obs.extend_from_slice(o);
            next_obs.extend_from_slice(no);
            actions.extend_from_slice(a);
            rewards.push(r);
            terminated.push(if t { 1.0 } else { 0.0 });
            truncated.push(if tr { 1.0 } else { 0.0 });
        }

        Ok(FlatBatch {
            obs,
            next_obs,
            actions,
            rewards,
            terminated,
            truncated,
            batch_size,
            obs_dim,
            act_dim,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::{sample_record, ExperienceRecord};

    #[test]
    fn test_sample_flat_shapes() {
        let obs_dim = 8;
        let act_dim = 3;
        let mut buf = ReplayBuffer::new(200, obs_dim, act_dim);
        for _ in 0..100 {
            buf.push(sample_record_with_act(obs_dim, act_dim)).unwrap();
        }

        let batch = buf.sample_flat(32, 42).unwrap();

        assert_eq!(batch.batch_size, 32);
        assert_eq!(batch.obs_dim, obs_dim);
        assert_eq!(batch.act_dim, act_dim);
        assert_eq!(batch.obs.len(), 32 * obs_dim);
        assert_eq!(batch.next_obs.len(), 32 * obs_dim);
        assert_eq!(batch.actions.len(), 32 * act_dim);
        assert_eq!(batch.rewards.len(), 32);
        assert_eq!(batch.terminated.len(), 32);
        assert_eq!(batch.truncated.len(), 32);
    }

    #[test]
    fn test_sample_flat_contiguous_data() {
        let obs_dim = 4;
        let act_dim = 2;
        let mut buf = ReplayBuffer::new(100, obs_dim, act_dim);

        // Push a known record so we can verify content.
        let record = ExperienceRecord {
            obs: vec![1.0, 2.0, 3.0, 4.0],
            next_obs: vec![5.0, 6.0, 7.0, 8.0],
            action: vec![0.5, -0.5],
            reward: 10.0,
            terminated: true,
            truncated: false,
        };
        buf.push(record).unwrap();

        // With only 1 transition, every sample index must hit it.
        let batch = buf.sample_flat(1, 99).unwrap();

        assert_eq!(&batch.obs, &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(&batch.next_obs, &[5.0, 6.0, 7.0, 8.0]);
        assert_eq!(&batch.actions, &[0.5, -0.5]);
        assert_eq!(&batch.rewards, &[10.0]);
        assert_eq!(&batch.terminated, &[1.0]);
        assert_eq!(&batch.truncated, &[0.0]);
    }

    #[test]
    fn test_sample_flat_deterministic() {
        let obs_dim = 4;
        let act_dim = 1;
        let mut buf = ReplayBuffer::new(500, obs_dim, act_dim);
        for _ in 0..300 {
            buf.push(sample_record(obs_dim)).unwrap();
        }

        let b1 = buf.sample_flat(64, 42).unwrap();
        let b2 = buf.sample_flat(64, 42).unwrap();

        assert_eq!(b1.obs, b2.obs);
        assert_eq!(b1.next_obs, b2.next_obs);
        assert_eq!(b1.actions, b2.actions);
        assert_eq!(b1.rewards, b2.rewards);
        assert_eq!(b1.terminated, b2.terminated);
        assert_eq!(b1.truncated, b2.truncated);
    }

    #[test]
    fn test_sample_flat_empty_buffer_errors() {
        let buf = ReplayBuffer::new(100, 4, 1);
        let result = buf.sample_flat(1, 42);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("empty"),
            "expected 'empty' in error, got: {err_msg}"
        );
    }

    #[test]
    fn test_sample_flat_batch_too_large_errors() {
        let mut buf = ReplayBuffer::new(100, 4, 1);
        buf.push(sample_record(4)).unwrap();

        let result = buf.sample_flat(10, 42);
        assert!(result.is_err());
    }

    #[test]
    fn test_sample_flat_bool_encoding() {
        let obs_dim = 2;
        let act_dim = 1;
        let mut buf = ReplayBuffer::new(10, obs_dim, act_dim);

        // Push terminated=true, truncated=false
        buf.push(ExperienceRecord {
            obs: vec![0.0; obs_dim],
            next_obs: vec![0.0; obs_dim],
            action: vec![0.0],
            reward: 0.0,
            terminated: true,
            truncated: false,
        })
        .unwrap();

        // Push terminated=false, truncated=true
        buf.push(ExperienceRecord {
            obs: vec![1.0; obs_dim],
            next_obs: vec![1.0; obs_dim],
            action: vec![1.0],
            reward: 1.0,
            terminated: false,
            truncated: true,
        })
        .unwrap();

        // Sample all — terminated/truncated should only be 0.0 or 1.0.
        let batch = buf.sample_flat(2, 0).unwrap();
        for &v in &batch.terminated {
            assert!(v == 0.0 || v == 1.0, "terminated must be 0.0 or 1.0, got {v}");
        }
        for &v in &batch.truncated {
            assert!(v == 0.0 || v == 1.0, "truncated must be 0.0 or 1.0, got {v}");
        }
    }

    /// Helper that creates a record with a given obs_dim and act_dim.
    fn sample_record_with_act(obs_dim: usize, act_dim: usize) -> ExperienceRecord {
        ExperienceRecord {
            obs: vec![1.0; obs_dim],
            next_obs: vec![2.0; obs_dim],
            action: vec![0.0; act_dim],
            reward: 1.0,
            terminated: false,
            truncated: false,
        }
    }
}
