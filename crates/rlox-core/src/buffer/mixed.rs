//! Mixed sampling from two replay buffers.
//!
//! Used for offline-to-online fine-tuning: sample a configurable ratio
//! from an offline dataset and the remainder from an online replay buffer.

use crate::error::RloxError;

use super::ringbuf::{ReplayBuffer, SampledBatch};

/// Sample a mixed batch from two buffers.
///
/// Draws `ceil(batch_size * ratio)` from `buffer_a` and the remainder from
/// `buffer_b`. Returns a merged `SampledBatch`.
///
/// # Arguments
/// * `buffer_a` - first buffer (e.g., offline dataset)
/// * `buffer_b` - second buffer (e.g., online replay)
/// * `ratio` - fraction of samples from buffer_a (0.0 to 1.0)
/// * `batch_size` - total number of transitions to sample
/// * `seed` - RNG seed
pub fn sample_mixed(
    buffer_a: &ReplayBuffer,
    buffer_b: &ReplayBuffer,
    ratio: f64,
    batch_size: usize,
    seed: u64,
) -> Result<SampledBatch, RloxError> {
    if !(0.0..=1.0).contains(&ratio) {
        return Err(RloxError::BufferError(format!(
            "ratio must be in [0.0, 1.0], got {ratio}"
        )));
    }

    if buffer_a.is_empty() && ratio > 0.0 {
        return Err(RloxError::BufferError(
            "buffer_a is empty but ratio > 0".into(),
        ));
    }
    if buffer_b.is_empty() && ratio < 1.0 {
        return Err(RloxError::BufferError(
            "buffer_b is empty but ratio < 1.0".into(),
        ));
    }

    let n_from_a = ((batch_size as f64) * ratio).ceil() as usize;
    let n_from_a = n_from_a.min(batch_size); // safety clamp
    let n_from_b = batch_size - n_from_a;

    // Use different seeds for the two buffers to avoid correlation
    let seed_a = seed;
    let seed_b = seed.wrapping_add(0x9E37_79B9_7F4A_7C15); // golden-ratio derived offset

    let batch_a = if n_from_a > 0 {
        Some(buffer_a.sample(n_from_a, seed_a)?)
    } else {
        None
    };

    let batch_b = if n_from_b > 0 {
        Some(buffer_b.sample(n_from_b, seed_b)?)
    } else {
        None
    };

    // Determine dimensions from whichever batch we have
    let (obs_dim, act_dim) = match (&batch_a, &batch_b) {
        (Some(a), Some(b)) => {
            if a.obs_dim != b.obs_dim || a.act_dim != b.act_dim {
                return Err(RloxError::ShapeMismatch {
                    expected: format!("obs_dim={}, act_dim={}", a.obs_dim, a.act_dim),
                    got: format!("obs_dim={}, act_dim={}", b.obs_dim, b.act_dim),
                });
            }
            (a.obs_dim, a.act_dim)
        }
        (Some(a), None) => (a.obs_dim, a.act_dim),
        (None, Some(b)) => (b.obs_dim, b.act_dim),
        (None, None) => {
            return Err(RloxError::BufferError(
                "batch_size is 0 or both buffers empty".into(),
            ));
        }
    };

    let mut merged = SampledBatch::with_capacity(batch_size, obs_dim, act_dim);

    if let Some(a) = batch_a {
        merged.observations.extend_from_slice(&a.observations);
        merged
            .next_observations
            .extend_from_slice(&a.next_observations);
        merged.actions.extend_from_slice(&a.actions);
        merged.rewards.extend_from_slice(&a.rewards);
        merged.terminated.extend_from_slice(&a.terminated);
        merged.truncated.extend_from_slice(&a.truncated);
    }

    if let Some(b) = batch_b {
        merged.observations.extend_from_slice(&b.observations);
        merged
            .next_observations
            .extend_from_slice(&b.next_observations);
        merged.actions.extend_from_slice(&b.actions);
        merged.rewards.extend_from_slice(&b.rewards);
        merged.terminated.extend_from_slice(&b.terminated);
        merged.truncated.extend_from_slice(&b.truncated);
    }

    merged.batch_size = batch_size;

    Ok(merged)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::sample_record;

    fn make_buffer_with_reward(
        capacity: usize,
        obs_dim: usize,
        n: usize,
        reward: f32,
    ) -> ReplayBuffer {
        let mut buf = ReplayBuffer::new(capacity, obs_dim, 1);
        for _ in 0..n {
            let mut r = sample_record(obs_dim);
            r.reward = reward;
            buf.push(r).unwrap();
        }
        buf
    }

    #[test]
    fn test_mixed_ratio_one_all_from_a() {
        let buf_a = make_buffer_with_reward(100, 4, 50, 1.0);
        let buf_b = make_buffer_with_reward(100, 4, 50, 2.0);
        let batch = sample_mixed(&buf_a, &buf_b, 1.0, 32, 42).unwrap();
        assert_eq!(batch.batch_size, 32);
        for &r in &batch.rewards {
            assert_eq!(r, 1.0, "all samples should come from buffer_a");
        }
    }

    #[test]
    fn test_mixed_ratio_zero_all_from_b() {
        let buf_a = make_buffer_with_reward(100, 4, 50, 1.0);
        let buf_b = make_buffer_with_reward(100, 4, 50, 2.0);
        let batch = sample_mixed(&buf_a, &buf_b, 0.0, 32, 42).unwrap();
        assert_eq!(batch.batch_size, 32);
        for &r in &batch.rewards {
            assert_eq!(r, 2.0, "all samples should come from buffer_b");
        }
    }

    #[test]
    fn test_mixed_ratio_half() {
        let buf_a = make_buffer_with_reward(100, 4, 50, 1.0);
        let buf_b = make_buffer_with_reward(100, 4, 50, 2.0);
        let batch = sample_mixed(&buf_a, &buf_b, 0.5, 32, 42).unwrap();
        assert_eq!(batch.batch_size, 32);
        let from_a = batch.rewards.iter().filter(|&&r| r == 1.0).count();
        let from_b = batch.rewards.iter().filter(|&&r| r == 2.0).count();
        assert_eq!(from_a, 16);
        assert_eq!(from_b, 16);
    }

    #[test]
    fn test_mixed_deterministic() {
        let buf_a = make_buffer_with_reward(100, 4, 50, 1.0);
        let buf_b = make_buffer_with_reward(100, 4, 50, 2.0);
        let b1 = sample_mixed(&buf_a, &buf_b, 0.5, 32, 42).unwrap();
        let b2 = sample_mixed(&buf_a, &buf_b, 0.5, 32, 42).unwrap();
        assert_eq!(b1.observations, b2.observations);
        assert_eq!(b1.rewards, b2.rewards);
    }

    #[test]
    fn test_mixed_batch_shape() {
        let buf_a = make_buffer_with_reward(100, 4, 50, 1.0);
        let buf_b = make_buffer_with_reward(100, 4, 50, 2.0);
        let batch = sample_mixed(&buf_a, &buf_b, 0.5, 32, 42).unwrap();
        assert_eq!(batch.observations.len(), 32 * 4);
        assert_eq!(batch.next_observations.len(), 32 * 4);
        assert_eq!(batch.actions.len(), 32);
        assert_eq!(batch.rewards.len(), 32);
        assert_eq!(batch.terminated.len(), 32);
    }

    #[test]
    fn test_mixed_empty_buffer_errors() {
        let buf_a = ReplayBuffer::new(100, 4, 1);
        let buf_b = make_buffer_with_reward(100, 4, 50, 2.0);
        let result = sample_mixed(&buf_a, &buf_b, 0.5, 32, 42);
        assert!(result.is_err());
    }

    #[test]
    fn test_mixed_validates_ratio_range() {
        let buf_a = make_buffer_with_reward(100, 4, 50, 1.0);
        let buf_b = make_buffer_with_reward(100, 4, 50, 2.0);
        let result = sample_mixed(&buf_a, &buf_b, 1.5, 32, 42);
        assert!(result.is_err());

        let result2 = sample_mixed(&buf_a, &buf_b, -0.1, 32, 42);
        assert!(result2.is_err());
    }

    mod proptests {
        use super::*;
        use crate::buffer::sample_record;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn prop_mixed_batch_size_correct(
                batch_size in 1usize..50,
                ratio in 0.0f64..1.0,
            ) {
                let mut buf_a = ReplayBuffer::new(200, 4, 1);
                let mut buf_b = ReplayBuffer::new(200, 4, 1);
                for _ in 0..100 {
                    buf_a.push(sample_record(4)).unwrap();
                    buf_b.push(sample_record(4)).unwrap();
                }
                let batch = sample_mixed(&buf_a, &buf_b, ratio, batch_size, 42).unwrap();
                prop_assert_eq!(batch.batch_size, batch_size);
                prop_assert_eq!(batch.rewards.len(), batch_size);
            }
        }
    }
}
