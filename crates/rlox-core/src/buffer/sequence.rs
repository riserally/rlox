//! Sequence replay buffer for recurrent/transformer-based RL algorithms.
//!
//! Wraps a [`ReplayBuffer`] and [`EpisodeTracker`] to provide sampling of
//! contiguous transition sequences that never cross episode boundaries.
//! Used by DreamerV3, R2D2, and other sequence-based algorithms.

use crate::error::RloxError;

use super::episode::EpisodeTracker;
use super::ringbuf::ReplayBuffer;

/// Replay buffer that samples contiguous sequences of transitions.
///
/// Wraps a `ReplayBuffer` for storage and an `EpisodeTracker` for
/// episode-aware sequence sampling. Sequences never cross episode
/// boundaries.
pub struct SequenceReplayBuffer {
    buffer: ReplayBuffer,
    tracker: EpisodeTracker,
    obs_dim: usize,
    act_dim: usize,
    capacity: usize,
}

/// A batch of sampled sequences.
pub struct SequenceBatch {
    /// Observations: flat `(batch_size * seq_len * obs_dim)`.
    pub observations: Vec<f32>,
    /// Next observations: flat `(batch_size * seq_len * obs_dim)`.
    pub next_observations: Vec<f32>,
    /// Actions: flat `(batch_size * seq_len * act_dim)`.
    pub actions: Vec<f32>,
    /// Rewards: flat `(batch_size * seq_len)`.
    pub rewards: Vec<f32>,
    /// Terminated flags: flat `(batch_size * seq_len)`.
    pub terminated: Vec<bool>,
    /// Truncated flags: flat `(batch_size * seq_len)`.
    pub truncated: Vec<bool>,
    pub obs_dim: usize,
    pub act_dim: usize,
    pub batch_size: usize,
    pub seq_len: usize,
}

impl SequenceReplayBuffer {
    /// Create a new sequence replay buffer with given capacity and dimensions.
    pub fn new(capacity: usize, obs_dim: usize, act_dim: usize) -> Self {
        Self {
            buffer: ReplayBuffer::new(capacity, obs_dim, act_dim),
            tracker: EpisodeTracker::new(capacity),
            obs_dim,
            act_dim,
            capacity,
        }
    }

    /// Push a single transition, notifying the episode tracker.
    pub fn push_slices(
        &mut self,
        obs: &[f32],
        next_obs: &[f32],
        action: &[f32],
        reward: f32,
        terminated: bool,
        truncated: bool,
    ) -> Result<(), RloxError> {
        let write_pos = self.buffer.write_pos();
        let was_full = self.buffer.len() == self.capacity;

        // If wrapping, invalidate episodes at the position about to be overwritten
        if was_full {
            self.tracker.invalidate_overwritten(write_pos, 1);
        }

        self.buffer
            .push_slices(obs, next_obs, action, reward, terminated, truncated)?;

        let done = terminated || truncated;
        self.tracker.notify_push(write_pos, done);

        Ok(())
    }

    /// Sample `batch_size` sequences of `seq_len` consecutive transitions.
    ///
    /// Each sequence is guaranteed to be within a single episode.
    pub fn sample_sequences(
        &self,
        batch_size: usize,
        seq_len: usize,
        seed: u64,
    ) -> Result<SequenceBatch, RloxError> {
        let windows = self.tracker.sample_windows(batch_size, seq_len, seed)?;

        let total_obs = batch_size * seq_len * self.obs_dim;
        let total_act = batch_size * seq_len * self.act_dim;
        let total_flat = batch_size * seq_len;

        let mut batch = SequenceBatch {
            observations: Vec::with_capacity(total_obs),
            next_observations: Vec::with_capacity(total_obs),
            actions: Vec::with_capacity(total_act),
            rewards: Vec::with_capacity(total_flat),
            terminated: Vec::with_capacity(total_flat),
            truncated: Vec::with_capacity(total_flat),
            obs_dim: self.obs_dim,
            act_dim: self.act_dim,
            batch_size,
            seq_len,
        };

        for window in &windows {
            for offset in 0..seq_len {
                let idx = (window.ring_start + offset) % self.capacity;
                let (obs, next_obs, action, reward, terminated, truncated) =
                    self.buffer.get(idx);
                batch.observations.extend_from_slice(obs);
                batch.next_observations.extend_from_slice(next_obs);
                batch.actions.extend_from_slice(action);
                batch.rewards.push(reward);
                batch.terminated.push(terminated);
                batch.truncated.push(truncated);
            }
        }

        Ok(batch)
    }

    /// Delegate to inner buffer for standard i.i.d. sampling.
    pub fn sample(
        &self,
        batch_size: usize,
        seed: u64,
    ) -> Result<super::ringbuf::SampledBatch, RloxError> {
        self.buffer.sample(batch_size, seed)
    }

    /// Number of valid transitions currently stored.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Number of complete episodes currently tracked.
    pub fn num_complete_episodes(&self) -> usize {
        self.tracker.num_complete_episodes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: push an episode of `length` steps with identifiable data.
    fn push_episode(buf: &mut SequenceReplayBuffer, length: usize, obs_base: f32) {
        let obs_dim = buf.obs_dim;
        let act_dim = buf.act_dim;
        for i in 0..length {
            let val = obs_base + i as f32;
            let obs = vec![val; obs_dim];
            let next_obs = vec![val + 1.0; obs_dim];
            let action = vec![0.0; act_dim];
            let reward = val;
            let done = i == length - 1;
            buf.push_slices(&obs, &next_obs, &action, reward, done, false)
                .unwrap();
        }
    }

    #[test]
    fn test_new_is_empty() {
        let buf = SequenceReplayBuffer::new(100, 4, 1);
        assert_eq!(buf.len(), 0);
        assert!(buf.is_empty());
        assert_eq!(buf.num_complete_episodes(), 0);
    }

    #[test]
    fn test_push_increments_len() {
        let mut buf = SequenceReplayBuffer::new(100, 4, 1);
        push_episode(&mut buf, 5, 0.0);
        assert_eq!(buf.len(), 5);
        assert_eq!(buf.num_complete_episodes(), 1);
    }

    #[test]
    fn test_single_episode_sequence_sample() {
        let mut buf = SequenceReplayBuffer::new(100, 4, 1);
        push_episode(&mut buf, 10, 0.0);
        let batch = buf.sample_sequences(1, 3, 42).unwrap();
        assert_eq!(batch.batch_size, 1);
        assert_eq!(batch.seq_len, 3);
        assert_eq!(batch.observations.len(), 1 * 3 * 4);
        assert_eq!(batch.rewards.len(), 3);
    }

    #[test]
    fn test_sequences_dont_cross_episodes() {
        let mut buf = SequenceReplayBuffer::new(100, 4, 1);
        // Two episodes of length 5 each
        push_episode(&mut buf, 5, 0.0);
        push_episode(&mut buf, 5, 100.0);
        assert_eq!(buf.num_complete_episodes(), 2);

        // Sample many sequences of length 4
        let batch = buf.sample_sequences(20, 4, 42).unwrap();

        // Each sequence should have rewards either all in [0,5) or all in [100,105)
        for seq_idx in 0..20 {
            let rewards: Vec<f32> = (0..4)
                .map(|t| batch.rewards[seq_idx * 4 + t])
                .collect();
            let all_low = rewards.iter().all(|&r| r < 50.0);
            let all_high = rewards.iter().all(|&r| r >= 50.0);
            assert!(
                all_low || all_high,
                "sequence {seq_idx} crosses episode boundary: {rewards:?}"
            );
        }
    }

    #[test]
    fn test_sequence_contiguity() {
        let mut buf = SequenceReplayBuffer::new(100, 4, 1);
        push_episode(&mut buf, 10, 0.0);

        let batch = buf.sample_sequences(5, 5, 42).unwrap();
        let obs_dim = 4;
        let seq_len = 5;

        for seq_idx in 0..5 {
            for t in 0..(seq_len - 1) {
                let next_obs_start = (seq_idx * seq_len + t) * obs_dim;
                let obs_next_start = (seq_idx * seq_len + t + 1) * obs_dim;

                let next_obs =
                    &batch.next_observations[next_obs_start..next_obs_start + obs_dim];
                let obs_t1 =
                    &batch.observations[obs_next_start..obs_next_start + obs_dim];

                assert_eq!(
                    next_obs, obs_t1,
                    "next_obs[{t}] != obs[{t_plus_1}] in seq {seq_idx}",
                    t_plus_1 = t + 1
                );
            }
        }
    }

    #[test]
    fn test_sequence_deterministic() {
        let mut buf = SequenceReplayBuffer::new(100, 4, 1);
        push_episode(&mut buf, 10, 0.0);
        let b1 = buf.sample_sequences(5, 3, 42).unwrap();
        let b2 = buf.sample_sequences(5, 3, 42).unwrap();
        assert_eq!(b1.observations, b2.observations);
        assert_eq!(b1.rewards, b2.rewards);
    }

    #[test]
    fn test_reject_too_long_sequence() {
        let mut buf = SequenceReplayBuffer::new(100, 4, 1);
        push_episode(&mut buf, 3, 0.0);
        let result = buf.sample_sequences(1, 5, 42);
        assert!(result.is_err());
    }

    #[test]
    fn test_capacity_respected() {
        let mut buf = SequenceReplayBuffer::new(100, 4, 1);
        // Push 200 transitions (20 episodes of length 10)
        for i in 0..20 {
            push_episode(&mut buf, 10, i as f32 * 100.0);
        }
        assert_eq!(buf.len(), 100);
    }

    #[test]
    fn test_batch_shape_correct() {
        let mut buf = SequenceReplayBuffer::new(200, 8, 2);
        push_episode(&mut buf, 20, 0.0);

        let batch = buf.sample_sequences(4, 3, 42).unwrap();
        assert_eq!(batch.observations.len(), 4 * 3 * 8);
        assert_eq!(batch.next_observations.len(), 4 * 3 * 8);
        assert_eq!(batch.actions.len(), 4 * 3 * 2);
        assert_eq!(batch.rewards.len(), 4 * 3);
        assert_eq!(batch.terminated.len(), 4 * 3);
        assert_eq!(batch.truncated.len(), 4 * 3);
    }

    #[test]
    fn test_empty_buffer_sample_errors() {
        let buf = SequenceReplayBuffer::new(100, 4, 1);
        let result = buf.sample_sequences(1, 1, 42);
        assert!(result.is_err());
    }

    #[test]
    fn test_push_slices_validates_dims() {
        let mut buf = SequenceReplayBuffer::new(100, 4, 1);
        let result = buf.push_slices(
            &[1.0, 2.0, 3.0], // wrong obs_dim: 3 instead of 4
            &[1.0, 2.0, 3.0, 4.0],
            &[0.0],
            1.0,
            false,
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_multiple_episodes_mixed_lengths() {
        let mut buf = SequenceReplayBuffer::new(200, 4, 1);
        push_episode(&mut buf, 3, 0.0);
        push_episode(&mut buf, 7, 100.0);
        push_episode(&mut buf, 2, 200.0);
        push_episode(&mut buf, 10, 300.0);

        // seq_len=4: only episodes of length >= 4 are eligible (7 and 10)
        let batch = buf.sample_sequences(10, 4, 42).unwrap();
        for seq_idx in 0..10 {
            let first_reward = batch.rewards[seq_idx * 4];
            // Should be from episode starting at 100.0 or 300.0
            assert!(
                first_reward >= 90.0,
                "seq {seq_idx} sampled from too-short episode: reward={first_reward}"
            );
        }
    }

    #[test]
    fn test_sequence_rewards_match_buffer() {
        let mut buf = SequenceReplayBuffer::new(100, 2, 1);
        // Push a known episode
        for i in 0..5 {
            let val = (i + 1) as f32 * 10.0;
            buf.push_slices(&[val, val], &[val + 1.0, val + 1.0], &[0.0], val, i == 4, false)
                .unwrap();
        }

        // Sample the entire episode
        let batch = buf.sample_sequences(1, 5, 42).unwrap();
        // The rewards should be a contiguous sub-sequence of [10, 20, 30, 40, 50]
        let expected = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        assert_eq!(batch.rewards, expected);
    }

    #[test]
    fn test_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SequenceReplayBuffer>();
    }

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn prop_batch_size_matches_request(
                batch_size in 1usize..10,
                seq_len in 1usize..5,
                ep_len in 5usize..20,
            ) {
                let cap = ep_len * 5;
                let mut buf = SequenceReplayBuffer::new(cap, 4, 1);
                push_episode(&mut buf, ep_len, 0.0);
                push_episode(&mut buf, ep_len, 100.0);
                let batch = buf.sample_sequences(batch_size, seq_len, 42).unwrap();
                prop_assert_eq!(batch.batch_size, batch_size);
                prop_assert_eq!(batch.seq_len, seq_len);
            }

            #[test]
            fn prop_len_never_exceeds_capacity(
                cap in 10usize..100,
                n_pushes in 1usize..300,
            ) {
                let mut buf = SequenceReplayBuffer::new(cap, 4, 1);
                for i in 0..n_pushes {
                    let done = i % 7 == 6;
                    buf.push_slices(
                        &[i as f32; 4],
                        &[(i + 1) as f32; 4],
                        &[0.0],
                        i as f32,
                        done,
                        false,
                    ).unwrap();
                }
                prop_assert!(buf.len() <= cap);
            }

            #[test]
            fn prop_sequence_obs_contiguous(
                ep_len in 5usize..20,
                seq_len in 2usize..5,
                batch_size in 1usize..5,
            ) {
                let mut buf = SequenceReplayBuffer::new(ep_len * 3, 4, 1);
                push_episode(&mut buf, ep_len, 0.0);
                let batch = buf.sample_sequences(batch_size, seq_len, 42).unwrap();
                let obs_dim = 4;
                for seq_idx in 0..batch_size {
                    for t in 0..(seq_len - 1) {
                        let next_start = (seq_idx * seq_len + t) * obs_dim;
                        let obs_next_start = (seq_idx * seq_len + t + 1) * obs_dim;
                        let next_obs = &batch.next_observations[next_start..next_start + obs_dim];
                        let obs_t1 = &batch.observations[obs_next_start..obs_next_start + obs_dim];
                        prop_assert_eq!(next_obs, obs_t1);
                    }
                }
            }
        }
    }
}
