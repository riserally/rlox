use parking_lot::Mutex;

use super::ringbuf::{ReplayBuffer, SampledBatch};
use crate::error::RloxError;

/// Thread-safe concurrent replay buffer backed by `parking_lot::Mutex`.
///
/// Multiple actor threads can push transitions concurrently.
/// A single learner thread samples batches.
///
/// `parking_lot::Mutex` is ~2x faster than `std::Mutex` on uncontended locks
/// (~10ns), which is negligible compared to the data copy cost of each push.
///
/// # Thread Safety
///
/// Automatically `Send + Sync` because `Mutex<T: Send>` is `Send + Sync`.
pub struct ConcurrentReplayBuffer {
    inner: Mutex<ReplayBuffer>,
    capacity: usize,
}

impl ConcurrentReplayBuffer {
    /// Create a concurrent replay buffer with fixed capacity.
    ///
    /// All arrays are pre-allocated inside the inner `ReplayBuffer`.
    /// `obs_dim` and `act_dim` define the per-transition dimensionality.
    pub fn new(capacity: usize, obs_dim: usize, act_dim: usize) -> Self {
        Self {
            inner: Mutex::new(ReplayBuffer::new(capacity, obs_dim, act_dim)),
            capacity,
        }
    }

    /// Push a transition into the buffer from borrowed slices.
    ///
    /// Thread-safe: multiple threads may call `push` concurrently.
    pub fn push(
        &self,
        obs: &[f32],
        next_obs: &[f32],
        action: &[f32],
        reward: f32,
        terminated: bool,
        truncated: bool,
    ) -> Result<(), RloxError> {
        let mut buf = self.inner.lock();
        buf.push_slices(obs, next_obs, action, reward, terminated, truncated)
    }

    /// Sample a batch of transitions uniformly at random.
    ///
    /// Uses `ChaCha8Rng` seeded with `seed` for deterministic cross-platform
    /// reproducibility.
    pub fn sample(&self, batch_size: usize, seed: u64) -> Result<SampledBatch, RloxError> {
        let buf = self.inner.lock();
        buf.sample(batch_size, seed)
    }

    /// Number of transitions currently stored.
    ///
    /// This is always `<= capacity`.
    pub fn len(&self) -> usize {
        self.inner.lock().len()
    }

    /// Whether the buffer has no transitions.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Maximum number of transitions the buffer can hold.
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn test_concurrent_push_single_thread() {
        let buf = ConcurrentReplayBuffer::new(100, 4, 2);

        for i in 0..50 {
            let obs = vec![i as f32; 4];
            let next_obs = vec![(i + 1) as f32; 4];
            let action = vec![0.1, 0.2];
            buf.push(&obs, &next_obs, &action, i as f32, false, false)
                .expect("push should succeed");
        }

        assert_eq!(buf.len(), 50);
        let batch = buf.sample(10, 42).expect("sample should succeed");
        assert_eq!(batch.batch_size, 10);
        assert_eq!(batch.observations.len(), 10 * 4);
        assert_eq!(batch.actions.len(), 10 * 2);
        assert_eq!(batch.rewards.len(), 10);
    }

    #[test]
    fn test_concurrent_push_multi_thread() {
        let buf = Arc::new(ConcurrentReplayBuffer::new(1000, 4, 1));
        let n_threads = 4;
        let pushes_per_thread = 200;

        let handles: Vec<_> = (0..n_threads)
            .map(|t| {
                let buf = Arc::clone(&buf);
                std::thread::spawn(move || {
                    for i in 0..pushes_per_thread {
                        let val = (t * pushes_per_thread + i) as f32;
                        let obs = vec![val; 4];
                        let next_obs = vec![val + 1.0; 4];
                        let action = vec![val * 0.01];
                        buf.push(&obs, &next_obs, &action, val, false, false)
                            .expect("push should succeed");
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().expect("thread should not panic");
        }

        assert_eq!(buf.len(), n_threads * pushes_per_thread);

        // Verify no data corruption: sample and check shapes
        let batch = buf.sample(100, 42).expect("sample should succeed");
        assert_eq!(batch.batch_size, 100);
        assert_eq!(batch.observations.len(), 100 * 4);

        // Each obs should be [v, v, v, v] where v is consistent
        for i in 0..batch.batch_size {
            let obs_slice = &batch.observations[i * 4..(i + 1) * 4];
            let first = obs_slice[0];
            for &val in &obs_slice[1..] {
                assert!(
                    (val - first).abs() < f32::EPSILON,
                    "data corruption detected: obs={obs_slice:?}"
                );
            }
        }
    }

    #[test]
    fn test_concurrent_sample_during_push() {
        let buf = Arc::new(ConcurrentReplayBuffer::new(1000, 4, 1));

        // Pre-fill some data so sampling can start immediately
        for i in 0..100 {
            let obs = vec![i as f32; 4];
            let next_obs = vec![(i + 1) as f32; 4];
            buf.push(&obs, &next_obs, &[0.0], 1.0, false, false)
                .expect("push should succeed");
        }

        let buf_push = Arc::clone(&buf);
        let push_handle = std::thread::spawn(move || {
            for i in 100..500 {
                let obs = vec![i as f32; 4];
                let next_obs = vec![(i + 1) as f32; 4];
                buf_push
                    .push(&obs, &next_obs, &[0.0], 1.0, false, false)
                    .expect("push should succeed");
            }
        });

        // Sample concurrently while pushes are happening
        let mut sample_count = 0;
        for seed in 0..50u64 {
            let len = buf.len();
            if len >= 10 {
                let batch = buf.sample(10, seed).expect("sample should succeed");
                assert_eq!(batch.batch_size, 10);
                sample_count += 1;
            }
        }

        push_handle.join().expect("push thread should not panic");
        assert!(sample_count > 0, "should have sampled at least once");
    }

    #[test]
    fn test_concurrent_wrap_around() {
        let capacity = 50;
        let buf = Arc::new(ConcurrentReplayBuffer::new(capacity, 2, 1));
        let n_threads = 4;
        let pushes_per_thread = 100; // 400 total > capacity 50

        let handles: Vec<_> = (0..n_threads)
            .map(|t| {
                let buf = Arc::clone(&buf);
                std::thread::spawn(move || {
                    for i in 0..pushes_per_thread {
                        let val = (t * pushes_per_thread + i) as f32;
                        buf.push(
                            &[val, val],
                            &[val + 1.0, val + 1.0],
                            &[val],
                            val,
                            false,
                            false,
                        )
                        .expect("push should succeed");
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().expect("thread should not panic");
        }

        // len must not exceed capacity
        assert_eq!(buf.len(), capacity);
        assert_eq!(buf.capacity(), capacity);

        // Should still be able to sample successfully
        let batch = buf.sample(20, 42).expect("sample should succeed");
        assert_eq!(batch.batch_size, 20);
    }

    #[test]
    fn test_concurrent_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ConcurrentReplayBuffer>();
    }

    #[test]
    fn test_concurrent_deterministic_sample() {
        let buf = ConcurrentReplayBuffer::new(500, 4, 1);
        for i in 0..200 {
            let obs = vec![i as f32; 4];
            let next_obs = vec![(i + 1) as f32; 4];
            buf.push(&obs, &next_obs, &[0.0], i as f32, false, false)
                .expect("push should succeed");
        }

        let b1 = buf.sample(32, 42).expect("sample should succeed");
        let b2 = buf.sample(32, 42).expect("sample should succeed");
        assert_eq!(b1.observations, b2.observations);
        assert_eq!(b1.rewards, b2.rewards);
        assert_eq!(b1.actions, b2.actions);
    }

    #[test]
    fn test_concurrent_empty_sample_errors() {
        let buf = ConcurrentReplayBuffer::new(100, 4, 1);
        assert!(buf.sample(1, 42).is_err());
        assert!(buf.is_empty());
    }

    #[test]
    fn test_concurrent_shape_mismatch_errors() {
        let buf = ConcurrentReplayBuffer::new(100, 4, 2);
        // Wrong obs dim
        assert!(buf
            .push(&[1.0, 2.0], &[1.0; 4], &[0.0, 0.0], 1.0, false, false)
            .is_err());
        // Wrong next_obs dim
        assert!(buf
            .push(&[1.0; 4], &[1.0, 2.0], &[0.0, 0.0], 1.0, false, false)
            .is_err());
        // Wrong action dim
        assert!(buf
            .push(&[1.0; 4], &[1.0; 4], &[0.0], 1.0, false, false)
            .is_err());
    }

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn prop_concurrent_len_never_exceeds_capacity(
                capacity in 1..500usize,
                num_pushes in 0..2000usize
            ) {
                let buf = ConcurrentReplayBuffer::new(capacity, 2, 1);
                for i in 0..num_pushes {
                    let v = i as f32;
                    buf.push(&[v, v], &[v, v], &[v], v, false, false).expect("push should succeed");
                }
                prop_assert!(buf.len() <= capacity);
                prop_assert_eq!(buf.len(), num_pushes.min(capacity));
            }
        }
    }
}
