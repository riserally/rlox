//! Sum-tree backed prioritized experience replay.
//!
//! Provides O(log N) prefix-sum sampling and priority updates for
//! Prioritized Experience Replay (Schaul et al., 2015).

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::error::RloxError;

use super::ExperienceRecord;

// ---------------------------------------------------------------------------
// SumTree
// ---------------------------------------------------------------------------

/// Binary sum-tree for O(log N) prefix-sum queries.
///
/// Internally stores `2 * capacity` nodes where leaves occupy indices
/// `[capacity .. 2*capacity)` and internal nodes hold partial sums.
pub struct SumTree {
    capacity: usize,
    tree: Vec<f64>,
}

impl SumTree {
    /// Create a sum-tree with `capacity` leaves, all initialised to zero.
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "SumTree capacity must be > 0");
        let capacity = capacity.next_power_of_two();
        Self {
            capacity,
            tree: vec![0.0; 2 * capacity],
        }
    }

    /// Number of leaves.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Sum of all leaf priorities.
    pub fn total(&self) -> f64 {
        self.tree[1]
    }

    /// Set the priority of leaf `index`.
    pub fn set(&mut self, index: usize, priority: f64) {
        assert!(index < self.capacity, "SumTree index out of bounds");
        let mut pos = index + self.capacity;
        self.tree[pos] = priority;
        while pos > 1 {
            pos /= 2;
            self.tree[pos] = self.tree[2 * pos] + self.tree[2 * pos + 1];
        }
    }

    /// Get the priority of leaf `index`.
    pub fn get(&self, index: usize) -> f64 {
        assert!(index < self.capacity, "SumTree index out of bounds");
        self.tree[index + self.capacity]
    }

    /// Sample a leaf index such that the probability of choosing leaf `i` is
    /// `priority[i] / total()`.
    ///
    /// `value` should be in `[0, total())`.
    pub fn sample(&self, value: f64) -> usize {
        debug_assert!(value >= 0.0 && value < self.total() + 1e-12);
        let mut pos = 1;
        let mut remaining = value;
        while pos < self.capacity {
            let left = 2 * pos;
            let right = left + 1;
            if remaining < self.tree[left] {
                pos = left;
            } else {
                remaining -= self.tree[left];
                pos = right;
            }
        }
        pos - self.capacity
    }
}

// ---------------------------------------------------------------------------
// PrioritizedReplayBuffer
// ---------------------------------------------------------------------------

/// Prioritized experience replay buffer backed by a sum-tree.
///
/// Implements proportional prioritization with importance-sampling weights.
pub struct PrioritizedReplayBuffer {
    obs_dim: usize,
    act_dim: usize,
    capacity: usize,
    alpha: f64,
    beta: f64,
    tree: SumTree,
    observations: Vec<f32>,
    next_observations: Vec<f32>,
    actions: Vec<f32>,
    rewards: Vec<f32>,
    terminated: Vec<bool>,
    truncated: Vec<bool>,
    write_pos: usize,
    count: usize,
    max_priority: f64,
}

/// A sampled batch with importance-sampling weights.
pub struct PrioritizedSampledBatch {
    pub observations: Vec<f32>,
    pub next_observations: Vec<f32>,
    pub actions: Vec<f32>,
    pub rewards: Vec<f32>,
    pub terminated: Vec<bool>,
    pub truncated: Vec<bool>,
    pub obs_dim: usize,
    pub act_dim: usize,
    pub batch_size: usize,
    pub weights: Vec<f64>,
    pub indices: Vec<usize>,
}

impl PrioritizedReplayBuffer {
    /// Create a new prioritized replay buffer.
    ///
    /// * `alpha` — prioritization exponent (0 = uniform, 1 = full prioritization)
    /// * `beta` — importance-sampling correction exponent (1 = full correction)
    pub fn new(capacity: usize, obs_dim: usize, act_dim: usize, alpha: f64, beta: f64) -> Self {
        Self {
            obs_dim,
            act_dim,
            capacity,
            alpha,
            beta,
            tree: SumTree::new(capacity),
            observations: vec![0.0; capacity * obs_dim],
            next_observations: vec![0.0; capacity * obs_dim],
            actions: vec![0.0; capacity * act_dim],
            rewards: vec![0.0; capacity],
            terminated: vec![false; capacity],
            truncated: vec![false; capacity],
            write_pos: 0,
            count: 0,
            max_priority: 1.0,
        }
    }

    /// Number of valid transitions stored.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Push a transition with the given TD-error priority.
    pub fn push(&mut self, record: ExperienceRecord, priority: f64) -> Result<(), RloxError> {
        if priority < 0.0 {
            return Err(RloxError::BufferError(
                "priority must be non-negative".into(),
            ));
        }
        if record.obs.len() != self.obs_dim {
            return Err(RloxError::ShapeMismatch {
                expected: format!("obs_dim={}", self.obs_dim),
                got: format!("obs.len()={}", record.obs.len()),
            });
        }
        if record.next_obs.len() != self.obs_dim {
            return Err(RloxError::ShapeMismatch {
                expected: format!("obs_dim={}", self.obs_dim),
                got: format!("next_obs.len()={}", record.next_obs.len()),
            });
        }
        if record.action.len() != self.act_dim {
            return Err(RloxError::ShapeMismatch {
                expected: format!("act_dim={}", self.act_dim),
                got: format!("action.len()={}", record.action.len()),
            });
        }

        let idx = self.write_pos;
        let obs_start = idx * self.obs_dim;
        self.observations[obs_start..obs_start + self.obs_dim]
            .copy_from_slice(&record.obs);
        self.next_observations[obs_start..obs_start + self.obs_dim]
            .copy_from_slice(&record.next_obs);
        let act_start = idx * self.act_dim;
        self.actions[act_start..act_start + self.act_dim]
            .copy_from_slice(&record.action);
        self.rewards[idx] = record.reward;
        self.terminated[idx] = record.terminated;
        self.truncated[idx] = record.truncated;

        let p_alpha = priority.powf(self.alpha);
        self.tree.set(idx, p_alpha);
        if p_alpha > self.max_priority {
            self.max_priority = p_alpha;
        }

        self.write_pos = (self.write_pos + 1) % self.capacity;
        if self.count < self.capacity {
            self.count += 1;
        }
        Ok(())
    }

    /// Set the importance-sampling beta parameter.
    pub fn set_beta(&mut self, beta: f64) {
        self.beta = beta;
    }

    /// Sample a batch with importance-sampling weights.
    pub fn sample(&self, batch_size: usize, seed: u64) -> Result<PrioritizedSampledBatch, RloxError> {
        if self.count == 0 {
            return Err(RloxError::BufferError(
                "cannot sample from empty buffer".into(),
            ));
        }
        if batch_size > self.count {
            return Err(RloxError::BufferError(format!(
                "batch_size {} > buffer len {}",
                batch_size, self.count
            )));
        }

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let total = self.tree.total();
        let segment = total / batch_size as f64;

        let mut batch = PrioritizedSampledBatch {
            observations: Vec::with_capacity(batch_size * self.obs_dim),
            next_observations: Vec::with_capacity(batch_size * self.obs_dim),
            actions: Vec::with_capacity(batch_size * self.act_dim),
            rewards: Vec::with_capacity(batch_size),
            terminated: Vec::with_capacity(batch_size),
            truncated: Vec::with_capacity(batch_size),
            obs_dim: self.obs_dim,
            act_dim: self.act_dim,
            batch_size,
            weights: Vec::with_capacity(batch_size),
            indices: Vec::with_capacity(batch_size),
        };

        let min_prob = self.tree_min_prob();
        let max_weight = (self.count as f64 * min_prob).powf(-self.beta);

        for i in 0..batch_size {
            let lo = segment * i as f64;
            let hi = segment * (i + 1) as f64;
            let value = rng.random_range(lo..hi);
            let idx = self.tree.sample(value.min(total - 1e-12));

            // Ensure we only sample valid indices
            let idx = idx.min(self.count - 1);

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

            let prob = self.tree.get(idx) / total;
            let weight = (self.count as f64 * prob).powf(-self.beta);
            batch.weights.push(weight / max_weight);
            batch.indices.push(idx);
        }

        Ok(batch)
    }

    /// Update priorities for previously sampled indices.
    pub fn update_priorities(
        &mut self,
        indices: &[usize],
        priorities: &[f64],
    ) -> Result<(), RloxError> {
        if indices.len() != priorities.len() {
            return Err(RloxError::BufferError(
                "indices and priorities must have same length".into(),
            ));
        }
        for (&idx, &p) in indices.iter().zip(priorities.iter()) {
            if p < 0.0 {
                return Err(RloxError::BufferError(
                    "priority must be non-negative".into(),
                ));
            }
            if idx >= self.count {
                return Err(RloxError::BufferError(format!(
                    "index {} >= buffer len {}",
                    idx, self.count
                )));
            }
            let p_alpha = p.powf(self.alpha);
            self.tree.set(idx, p_alpha);
            if p_alpha > self.max_priority {
                self.max_priority = p_alpha;
            }
        }
        Ok(())
    }

    /// Find the minimum non-zero probability among valid leaves.
    fn tree_min_prob(&self) -> f64 {
        let total = self.tree.total();
        if total == 0.0 {
            return 1.0;
        }
        let mut min_p = f64::MAX;
        for i in 0..self.count {
            let p = self.tree.get(i);
            if p > 0.0 && p < min_p {
                min_p = p;
            }
        }
        if min_p == f64::MAX {
            1.0 / self.count as f64
        } else {
            min_p / total
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::{sample_record, sample_record_multidim};

    // ---- SumTree tests ----

    #[test]
    fn sum_tree_new_has_zero_total() {
        let tree = SumTree::new(8);
        assert_eq!(tree.total(), 0.0);
        assert_eq!(tree.capacity(), 8);
    }

    #[test]
    fn sum_tree_set_and_get() {
        let mut tree = SumTree::new(4);
        tree.set(0, 1.0);
        tree.set(1, 2.0);
        tree.set(2, 3.0);
        tree.set(3, 4.0);
        assert_eq!(tree.get(0), 1.0);
        assert_eq!(tree.get(1), 2.0);
        assert_eq!(tree.get(2), 3.0);
        assert_eq!(tree.get(3), 4.0);
        assert_eq!(tree.total(), 10.0);
    }

    #[test]
    fn sum_tree_update_propagates() {
        let mut tree = SumTree::new(4);
        tree.set(0, 1.0);
        tree.set(1, 1.0);
        tree.set(2, 1.0);
        tree.set(3, 1.0);
        assert_eq!(tree.total(), 4.0);

        tree.set(2, 5.0);
        assert_eq!(tree.total(), 8.0);
        assert_eq!(tree.get(2), 5.0);
    }

    #[test]
    fn sum_tree_sample_returns_correct_leaf() {
        let mut tree = SumTree::new(4);
        tree.set(0, 1.0);
        tree.set(1, 2.0);
        tree.set(2, 3.0);
        tree.set(3, 4.0);
        // total = 10; prefix sums: [0,1) -> 0, [1,3) -> 1, [3,6) -> 2, [6,10) -> 3
        assert_eq!(tree.sample(0.0), 0);
        assert_eq!(tree.sample(0.5), 0);
        assert_eq!(tree.sample(1.0), 1);
        assert_eq!(tree.sample(2.9), 1);
        assert_eq!(tree.sample(3.0), 2);
        assert_eq!(tree.sample(5.9), 2);
        assert_eq!(tree.sample(6.0), 3);
        assert_eq!(tree.sample(9.9), 3);
    }

    #[test]
    fn sum_tree_single_leaf() {
        let mut tree = SumTree::new(1);
        tree.set(0, 5.0);
        assert_eq!(tree.total(), 5.0);
        assert_eq!(tree.sample(0.0), 0);
        assert_eq!(tree.sample(4.9), 0);
    }

    // ---- PrioritizedReplayBuffer tests ----

    #[test]
    fn prb_new_is_empty() {
        let buf = PrioritizedReplayBuffer::new(100, 4, 1, 0.6, 0.4);
        assert_eq!(buf.len(), 0);
        assert!(buf.is_empty());
    }

    #[test]
    fn prb_push_increments_len() {
        let mut buf = PrioritizedReplayBuffer::new(100, 4, 1, 0.6, 0.4);
        buf.push(sample_record(4), 1.0).unwrap();
        assert_eq!(buf.len(), 1);
    }

    #[test]
    fn prb_negative_priority_errors() {
        let mut buf = PrioritizedReplayBuffer::new(100, 4, 1, 0.6, 0.4);
        let result = buf.push(sample_record(4), -1.0);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("non-negative"));
    }

    #[test]
    fn prb_sample_empty_errors() {
        let buf = PrioritizedReplayBuffer::new(100, 4, 1, 0.6, 0.4);
        let result = buf.sample(1, 42);
        assert!(result.is_err());
    }

    #[test]
    fn prb_sample_too_large_errors() {
        let mut buf = PrioritizedReplayBuffer::new(100, 4, 1, 0.6, 0.4);
        buf.push(sample_record(4), 1.0).unwrap();
        let result = buf.sample(10, 42);
        assert!(result.is_err());
    }

    #[test]
    fn prb_sample_returns_correct_size() {
        let mut buf = PrioritizedReplayBuffer::new(100, 4, 1, 0.6, 0.4);
        for _ in 0..50 {
            buf.push(sample_record(4), 1.0).unwrap();
        }
        let batch = buf.sample(16, 42).unwrap();
        assert_eq!(batch.batch_size, 16);
        assert_eq!(batch.observations.len(), 16 * 4);
        assert_eq!(batch.actions.len(), 16);
        assert_eq!(batch.rewards.len(), 16);
        assert_eq!(batch.weights.len(), 16);
        assert_eq!(batch.indices.len(), 16);
    }

    #[test]
    fn prb_weights_are_in_zero_one() {
        let mut buf = PrioritizedReplayBuffer::new(100, 4, 1, 0.6, 0.4);
        for i in 0..50 {
            buf.push(sample_record(4), (i + 1) as f64).unwrap();
        }
        let batch = buf.sample(16, 42).unwrap();
        for &w in &batch.weights {
            assert!(w > 0.0, "weight must be positive, got {w}");
            assert!(w <= 1.0 + 1e-10, "weight must be <= 1.0, got {w}");
        }
    }

    #[test]
    fn prb_high_priority_sampled_more_often() {
        let mut buf = PrioritizedReplayBuffer::new(100, 4, 1, 1.0, 0.4);
        // Index 0 gets priority 100, rest get priority 1
        let mut rec = sample_record(4);
        rec.reward = 99.0;
        buf.push(rec, 100.0).unwrap();
        for _ in 1..50 {
            buf.push(sample_record(4), 1.0).unwrap();
        }

        // Sample many times and count how often index 0 appears
        let mut count_high = 0;
        for seed in 0..100 {
            let batch = buf.sample(10, seed).unwrap();
            for &idx in &batch.indices {
                if idx == 0 {
                    count_high += 1;
                }
            }
        }
        // With priority 100 vs 49*1 = 149 total, idx 0 should appear ~67% of time
        assert!(count_high > 200, "high priority item should be sampled frequently, got {count_high}/1000");
    }

    #[test]
    fn prb_update_priorities() {
        let mut buf = PrioritizedReplayBuffer::new(100, 4, 1, 1.0, 0.4);
        for _ in 0..10 {
            buf.push(sample_record(4), 1.0).unwrap();
        }
        // Update index 0 to very high priority
        buf.update_priorities(&[0], &[100.0]).unwrap();

        let mut count_idx0 = 0;
        for seed in 0..50 {
            let batch = buf.sample(5, seed).unwrap();
            for &idx in &batch.indices {
                if idx == 0 {
                    count_idx0 += 1;
                }
            }
        }
        assert!(count_idx0 > 100, "updated high-priority item should be sampled frequently");
    }

    #[test]
    fn prb_update_priorities_negative_errors() {
        let mut buf = PrioritizedReplayBuffer::new(100, 4, 1, 0.6, 0.4);
        buf.push(sample_record(4), 1.0).unwrap();
        let result = buf.update_priorities(&[0], &[-1.0]);
        assert!(result.is_err());
    }

    #[test]
    fn prb_update_priorities_oob_errors() {
        let mut buf = PrioritizedReplayBuffer::new(100, 4, 1, 0.6, 0.4);
        buf.push(sample_record(4), 1.0).unwrap();
        let result = buf.update_priorities(&[5], &[1.0]);
        assert!(result.is_err());
    }

    #[test]
    fn prb_update_priorities_length_mismatch_errors() {
        let mut buf = PrioritizedReplayBuffer::new(100, 4, 1, 0.6, 0.4);
        buf.push(sample_record(4), 1.0).unwrap();
        let result = buf.update_priorities(&[0], &[1.0, 2.0]);
        assert!(result.is_err());
    }

    #[test]
    fn prb_set_beta() {
        let mut buf = PrioritizedReplayBuffer::new(100, 4, 1, 0.6, 0.4);
        buf.set_beta(1.0);
        // Just ensure it doesn't panic; beta affects weight computation
    }

    #[test]
    fn prb_wraps_around() {
        let mut buf = PrioritizedReplayBuffer::new(5, 4, 1, 0.6, 0.4);
        for i in 0..10 {
            let mut rec = sample_record(4);
            rec.reward = i as f32;
            buf.push(rec, 1.0).unwrap();
        }
        assert_eq!(buf.len(), 5);
        // Should contain rewards 5..10
        let batch = buf.sample(5, 42).unwrap();
        for &r in &batch.rewards {
            assert!(r >= 5.0, "old data should be overwritten, got reward {r}");
        }
    }

    #[test]
    fn prb_multidim_actions() {
        let mut buf = PrioritizedReplayBuffer::new(100, 4, 3, 0.6, 0.4);
        buf.push(sample_record_multidim(4, 3), 1.0).unwrap();
        let batch = buf.sample(1, 42).unwrap();
        assert_eq!(batch.act_dim, 3);
        assert_eq!(batch.actions.len(), 3);
    }

    #[test]
    fn prb_deterministic_with_same_seed() {
        let mut buf = PrioritizedReplayBuffer::new(100, 4, 1, 0.6, 0.4);
        for _ in 0..50 {
            buf.push(sample_record(4), 1.0).unwrap();
        }
        let b1 = buf.sample(16, 42).unwrap();
        let b2 = buf.sample(16, 42).unwrap();
        assert_eq!(b1.indices, b2.indices);
        assert_eq!(b1.weights, b2.weights);
    }

    #[test]
    fn prb_next_obs_roundtrip() {
        let obs_dim = 4;
        let mut buf = PrioritizedReplayBuffer::new(100, obs_dim, 1, 0.6, 0.4);
        let record = ExperienceRecord {
            obs: vec![1.0; obs_dim],
            next_obs: vec![2.0, 3.0, 4.0, 5.0],
            action: vec![0.0],
            reward: 1.0,
            terminated: false,
            truncated: false,
        };
        buf.push(record, 1.0).unwrap();
        let batch = buf.sample(1, 42).unwrap();
        assert_eq!(&batch.next_observations, &[2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn prb_next_obs_shape() {
        let obs_dim = 4;
        let mut buf = PrioritizedReplayBuffer::new(200, obs_dim, 1, 0.6, 0.4);
        for _ in 0..100 {
            buf.push(sample_record(obs_dim), 1.0).unwrap();
        }
        let batch = buf.sample(32, 42).unwrap();
        assert_eq!(batch.next_observations.len(), 32 * obs_dim);
    }

    #[test]
    fn prb_obs_dim_mismatch_errors() {
        let mut buf = PrioritizedReplayBuffer::new(100, 4, 1, 0.6, 0.4);
        let result = buf.push(sample_record(8), 1.0);
        assert!(result.is_err());
    }

    // ---- Proptests ----

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn sum_tree_total_equals_sum_of_leaves(
                priorities in proptest::collection::vec(0.0f64..100.0, 1..64)
            ) {
                let n = priorities.len();
                let mut tree = SumTree::new(n);
                let mut expected = 0.0;
                for (i, &p) in priorities.iter().enumerate() {
                    tree.set(i, p);
                    expected += p;
                }
                let diff = (tree.total() - expected).abs();
                prop_assert!(diff < 1e-6, "total {} != expected {}", tree.total(), expected);
            }

            #[test]
            fn sum_tree_sample_in_range(
                priorities in proptest::collection::vec(1.0f64..100.0, 1..64)
            ) {
                let n = priorities.len();
                let mut tree = SumTree::new(n);
                for (i, &p) in priorities.iter().enumerate() {
                    tree.set(i, p);
                }
                // Sample several values
                let total = tree.total();
                for v in [0.0, total * 0.25, total * 0.5, total * 0.75, total * 0.999] {
                    let idx = tree.sample(v);
                    prop_assert!(idx < n, "sampled index {} >= capacity {}", idx, n);
                }
            }

            #[test]
            fn prb_never_exceeds_capacity(
                capacity in 1..100usize,
                num_pushes in 0..300usize,
            ) {
                let mut buf = PrioritizedReplayBuffer::new(capacity, 4, 1, 0.6, 0.4);
                for _ in 0..num_pushes {
                    buf.push(sample_record(4), 1.0).unwrap();
                }
                prop_assert!(buf.len() <= capacity);
                prop_assert_eq!(buf.len(), num_pushes.min(capacity));
            }

            #[test]
            fn prb_weights_are_valid(
                num_pushes in 10..100usize,
                batch_size in 1..10usize,
            ) {
                let mut buf = PrioritizedReplayBuffer::new(200, 4, 1, 0.6, 0.4);
                for i in 0..num_pushes {
                    buf.push(sample_record(4), (i + 1) as f64).unwrap();
                }
                let effective_batch = batch_size.min(buf.len());
                let batch = buf.sample(effective_batch, 42).unwrap();
                for &w in &batch.weights {
                    prop_assert!(w > 0.0, "weight must be positive");
                    prop_assert!(w <= 1.0 + 1e-10, "weight must be <= 1.0");
                }
                for &idx in &batch.indices {
                    prop_assert!(idx < buf.len(), "index must be < len");
                }
            }
        }
    }
}
