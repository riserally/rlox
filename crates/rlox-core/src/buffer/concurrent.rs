use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::error::RloxError;
use super::ringbuf::SampledBatch;

/// Lock-free MPSC (multi-producer, single-consumer) replay buffer.
///
/// Multiple actor threads push transitions concurrently without locks.
/// A single learner thread samples batches. Uses an atomic write cursor
/// for slot claiming and per-slot commit flags to ensure only fully
/// written transitions are visible to the consumer.
///
/// # Design
///
/// - Fixed-size ring buffer with pre-allocated memory (zero-allocation push).
/// - `AtomicUsize` global write position: producers call `fetch_add` to claim a slot.
/// - `AtomicBool` per slot as a commit flag: set after data is fully written.
/// - `AtomicUsize` committed count: tracks how many slots are available for sampling.
/// - `Send + Sync` — designed for cross-thread sharing via `Arc`.
///
/// # Memory Ordering
///
/// - Producers use `Release` when setting the commit flag so that all prior
///   data writes are visible to the consumer.
/// - The consumer uses `Acquire` when reading commit flags so it sees the
///   data written by the producer.
pub struct ConcurrentReplayBuffer {
    obs_dim: usize,
    act_dim: usize,
    capacity: usize,

    // Flat pre-allocated storage. Written by producers (disjoint slots),
    // read by the consumer. Safety relies on each slot being written by
    // exactly one producer at a time, and only read after the commit flag
    // is set.
    observations: Box<[f32]>,
    next_observations: Box<[f32]>,
    actions: Box<[f32]>,
    rewards: Box<[f32]>,
    terminated: Box<[u8]>, // u8 instead of bool for AtomicBool-free slot storage
    truncated: Box<[u8]>,

    /// Per-slot commit flag. Set to `true` (Release) after data is fully written.
    /// Cleared (Release) before a slot is reclaimed on wrap-around.
    committed: Box<[AtomicBool]>,

    /// Global write cursor. Producers `fetch_add(1, Relaxed)` to claim a slot.
    /// The actual index is `write_pos % capacity`.
    write_pos: AtomicUsize,

    /// Number of committed (readable) transitions. Saturates at `capacity`.
    committed_count: AtomicUsize,

    /// Total number of pushes ever performed (monotonically increasing).
    /// Used to detect wrap-around: if `total_pushes > capacity`, older
    /// slots have been overwritten.
    total_pushes: AtomicUsize,
}

// SAFETY: All mutable state is behind atomics or accessed via disjoint
// slot indices claimed by `fetch_add`. The flat arrays (observations, etc.)
// are written by exactly one producer per slot, and only read by the consumer
// after the commit flag is set with Release/Acquire ordering.
unsafe impl Send for ConcurrentReplayBuffer {}
unsafe impl Sync for ConcurrentReplayBuffer {}

impl ConcurrentReplayBuffer {
    /// Create a lock-free concurrent replay buffer with fixed capacity.
    ///
    /// All arrays are pre-allocated. `obs_dim` and `act_dim` define the
    /// per-transition dimensionality.
    pub fn new(capacity: usize, obs_dim: usize, act_dim: usize) -> Self {
        let committed: Vec<AtomicBool> = (0..capacity).map(|_| AtomicBool::new(false)).collect();

        Self {
            obs_dim,
            act_dim,
            capacity,
            observations: vec![0.0f32; capacity * obs_dim].into_boxed_slice(),
            next_observations: vec![0.0f32; capacity * obs_dim].into_boxed_slice(),
            actions: vec![0.0f32; capacity * act_dim].into_boxed_slice(),
            rewards: vec![0.0f32; capacity].into_boxed_slice(),
            terminated: vec![0u8; capacity].into_boxed_slice(),
            truncated: vec![0u8; capacity].into_boxed_slice(),
            committed: committed.into_boxed_slice(),
            write_pos: AtomicUsize::new(0),
            committed_count: AtomicUsize::new(0),
            total_pushes: AtomicUsize::new(0),
        }
    }

    /// Push a transition into the buffer from borrowed slices.
    ///
    /// Thread-safe: multiple threads may call `push` concurrently.
    /// Uses `fetch_add` to claim a unique slot, writes data, then sets
    /// the commit flag.
    pub fn push(
        &self,
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

        // Claim a slot atomically.
        let raw_pos = self.write_pos.fetch_add(1, Ordering::Relaxed);
        let idx = raw_pos % self.capacity;

        // If this slot was previously committed, we are overwriting it.
        // Clear the commit flag first so the consumer does not read
        // partially written data.
        self.committed[idx].store(false, Ordering::Release);

        // Write data into the claimed slot.
        // SAFETY: Each slot index is unique per producer at this instant
        // because `fetch_add` returns a unique value. We cast the shared
        // references to mutable pointers for the specific, disjoint range.
        // The consumer will not read this slot until the commit flag is set.
        unsafe {
            let obs_start = idx * self.obs_dim;
            let obs_dst = self.observations.as_ptr().add(obs_start) as *mut f32;
            std::ptr::copy_nonoverlapping(obs.as_ptr(), obs_dst, self.obs_dim);

            let next_obs_dst = self.next_observations.as_ptr().add(obs_start) as *mut f32;
            std::ptr::copy_nonoverlapping(next_obs.as_ptr(), next_obs_dst, self.obs_dim);

            let act_start = idx * self.act_dim;
            let act_dst = self.actions.as_ptr().add(act_start) as *mut f32;
            std::ptr::copy_nonoverlapping(action.as_ptr(), act_dst, self.act_dim);

            let rew_dst = self.rewards.as_ptr().add(idx) as *mut f32;
            *rew_dst = reward;

            let term_dst = self.terminated.as_ptr().add(idx) as *mut u8;
            *term_dst = terminated as u8;

            let trunc_dst = self.truncated.as_ptr().add(idx) as *mut u8;
            *trunc_dst = truncated as u8;
        }

        // Mark this slot as committed. The Release ordering ensures all
        // data writes above are visible before the flag becomes true.
        self.committed[idx].store(true, Ordering::Release);

        // Increment committed count (saturate at capacity).
        // Use a CAS loop to avoid exceeding capacity.
        loop {
            let current = self.committed_count.load(Ordering::Relaxed);
            if current >= self.capacity {
                break;
            }
            match self.committed_count.compare_exchange_weak(
                current,
                current + 1,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(_) => continue,
            }
        }

        self.total_pushes.fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    /// Sample a batch of transitions uniformly at random.
    ///
    /// Uses `ChaCha8Rng` seeded with `seed` for deterministic cross-platform
    /// reproducibility. Only reads from committed slots.
    ///
    /// Intended to be called from a single learner thread while producers
    /// may be pushing concurrently.
    pub fn sample(&self, batch_size: usize, seed: u64) -> Result<SampledBatch, RloxError> {
        let count = self.len();
        if batch_size > count {
            return Err(RloxError::BufferError(format!(
                "batch_size {} > buffer len {}",
                batch_size, count
            )));
        }

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut batch = SampledBatch::with_capacity(batch_size, self.obs_dim, self.act_dim);

        let mut sampled = 0;
        while sampled < batch_size {
            let idx = rng.random_range(0..count);

            // Only read slots where the commit flag is set (Acquire ordering
            // ensures we see the data written by the producer).
            if !self.committed[idx].load(Ordering::Acquire) {
                continue;
            }

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
            batch.terminated.push(self.terminated[idx] != 0);
            batch.truncated.push(self.truncated[idx] != 0);

            sampled += 1;
        }

        batch.batch_size = batch_size;
        Ok(batch)
    }

    /// Number of committed (readable) transitions currently stored.
    ///
    /// This is always `<= capacity`.
    pub fn len(&self) -> usize {
        self.committed_count.load(Ordering::Acquire)
    }

    /// Whether the buffer has no committed transitions.
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
                .unwrap();
        }

        assert_eq!(buf.len(), 50);
        let batch = buf.sample(10, 42).unwrap();
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
                            .unwrap();
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(buf.len(), n_threads * pushes_per_thread);

        // Verify no data corruption: sample and check shapes
        let batch = buf.sample(100, 42).unwrap();
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
                .unwrap();
        }

        let buf_push = Arc::clone(&buf);
        let push_handle = std::thread::spawn(move || {
            for i in 100..500 {
                let obs = vec![i as f32; 4];
                let next_obs = vec![(i + 1) as f32; 4];
                buf_push
                    .push(&obs, &next_obs, &[0.0], 1.0, false, false)
                    .unwrap();
            }
        });

        // Sample concurrently while pushes are happening
        let mut sample_count = 0;
        for seed in 0..50u64 {
            let len = buf.len();
            if len >= 10 {
                let batch = buf.sample(10, seed).unwrap();
                assert_eq!(batch.batch_size, 10);
                sample_count += 1;
            }
        }

        push_handle.join().unwrap();
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
                        buf.push(&[val, val], &[val + 1.0, val + 1.0], &[val], val, false, false)
                            .unwrap();
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        // len must not exceed capacity
        assert_eq!(buf.len(), capacity);
        assert_eq!(buf.capacity(), capacity);

        // Should still be able to sample successfully
        let batch = buf.sample(20, 42).unwrap();
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
                .unwrap();
        }

        let b1 = buf.sample(32, 42).unwrap();
        let b2 = buf.sample(32, 42).unwrap();
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
                    buf.push(&[v, v], &[v, v], &[v], v, false, false).unwrap();
                }
                prop_assert!(buf.len() <= capacity);
                prop_assert_eq!(buf.len(), num_pushes.min(capacity));
            }
        }
    }
}
