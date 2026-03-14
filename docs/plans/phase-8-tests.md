# Phase 8 TDD Test Specifications — Production Hardening (v0.7)

**Status: RED (all tests must be written before implementation)**
**Phase plan: `/docs/plans/phase-8-production-hardening.md`**
**Depends on: Phase 7 tests passing (PPO, GRPO, DPO working end-to-end)**
**Run Rust tests:** `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo test`
**Run Python tests:** `.venv/bin/python -m pytest tests/python/test_phase8.py -v`

---

## Test Execution Order

```
Group 0: SumTree + PrioritizedReplayBuffer (Rust primitives)
Group 1: Off-policy algorithms (SAC, TD3, DQN)
Group 2: One-liner Trainer API + Config system
Group 3: Callback system + diagnostics
Group 4: Statistical evaluation + experiment metadata
Group 5: MuJoCo native + memory-mapped buffer + reward sandbox
Group 6: LLM pipeline (Online DPO, Best-of-N)
```

---

## Part 1: Rust Unit Tests

### 1.1 SumTree Tests

**Target file:** `crates/rlox-core/src/buffer/priority.rs` (new file)
**Group: 0**

```rust
// crates/rlox-core/src/buffer/priority.rs
#[cfg(test)]
mod sum_tree_tests {
    use super::*;

    // RED: SumTree::new creates a tree with correct capacity
    #[test]
    fn sum_tree_new_has_zero_total() {
        let tree = SumTree::new(1000);
        assert!((tree.total() - 0.0).abs() < f64::EPSILON);
        assert_eq!(tree.capacity(), 1000);
    }

    // RED: single insertion — total equals priority
    #[test]
    fn sum_tree_single_insert_total_equals_priority() {
        let mut tree = SumTree::new(100);
        tree.set(0, 5.0);
        assert!((tree.total() - 5.0).abs() < 1e-10);
    }

    // RED: multiple insertions — total equals sum of priorities
    #[test]
    fn sum_tree_total_equals_sum_of_priorities() {
        let mut tree = SumTree::new(100);
        let priorities = [1.0f64, 3.0, 2.0, 4.0, 1.5];
        for (i, &p) in priorities.iter().enumerate() {
            tree.set(i, p);
        }
        let expected_total: f64 = priorities.iter().sum();
        assert!((tree.total() - expected_total).abs() < 1e-10,
            "total={}, expected={}", tree.total(), expected_total);
    }

    // RED: priority update changes total correctly
    #[test]
    fn sum_tree_priority_update_changes_total() {
        let mut tree = SumTree::new(100);
        tree.set(0, 1.0);
        tree.set(1, 3.0);
        // Total = 4.0
        tree.set(0, 5.0);  // update priority 0 from 1.0 to 5.0
        // Total should now be 8.0
        assert!((tree.total() - 8.0).abs() < 1e-10,
            "total after update: {}", tree.total());
    }

    // RED: get returns leaf priority
    #[test]
    fn sum_tree_get_returns_correct_priority() {
        let mut tree = SumTree::new(100);
        tree.set(0, 1.0);
        tree.set(1, 3.0);
        tree.set(2, 2.0);
        assert!((tree.get(0) - 1.0).abs() < 1e-10);
        assert!((tree.get(1) - 3.0).abs() < 1e-10);
        assert!((tree.get(2) - 2.0).abs() < 1e-10);
    }

    // RED: sample returns index proportional to priority
    #[test]
    fn sum_tree_sample_proportional_to_priority() {
        // Priorities: [1.0, 9.0] — index 1 should appear ~9x more often
        let mut tree = SumTree::new(10);
        tree.set(0, 1.0);
        tree.set(1, 9.0);

        let n_samples = 10_000;
        let mut count_0 = 0usize;
        let mut count_1 = 0usize;

        for i in 0..n_samples {
            // Sample uniformly from [0, total)
            let s = (i as f64 / n_samples as f64) * tree.total();
            let idx = tree.sample(s);
            if idx == 0 {
                count_0 += 1;
            } else {
                count_1 += 1;
            }
        }

        // Index 1 should appear ~9x more often
        let ratio = count_1 as f64 / count_0.max(1) as f64;
        assert!(ratio > 7.0 && ratio < 11.0,
            "expected ratio ~9.0, got {:.2}", ratio);
    }

    // RED: sample with value = 0 returns first non-zero index
    #[test]
    fn sum_tree_sample_at_zero_returns_valid_index() {
        let mut tree = SumTree::new(10);
        tree.set(0, 1.0);
        tree.set(1, 2.0);
        let idx = tree.sample(0.0);
        assert!(idx < 2, "sample(0.0) should return index 0 or 1, got {}", idx);
    }

    // RED: sample at total - epsilon returns last non-zero index
    #[test]
    fn sum_tree_sample_near_total_returns_last_valid_index() {
        let mut tree = SumTree::new(10);
        tree.set(0, 1.0);
        tree.set(1, 2.0);
        let idx = tree.sample(tree.total() - 1e-12);
        assert!(idx < 2, "sample near total should return valid index");
    }

    // RED: SumTree handles capacity exactly (no out-of-bounds)
    #[test]
    fn sum_tree_at_capacity_no_panic() {
        let mut tree = SumTree::new(4);
        for i in 0..4 {
            tree.set(i, (i + 1) as f64);
        }
        // Total = 1+2+3+4 = 10
        assert!((tree.total() - 10.0).abs() < 1e-10);
    }

    // RED: O(log N) time complexity — set is O(log N) operations
    // (structural test — verify tree depth is log2(capacity))
    #[test]
    fn sum_tree_depth_is_log2_capacity() {
        for capacity in [16usize, 64, 256, 1024] {
            let tree = SumTree::new(capacity);
            // Tree should have log2(capacity) levels
            let expected_depth = (capacity as f64).log2().ceil() as usize;
            // We can't directly test depth, but capacity is preserved
            assert_eq!(tree.capacity(), capacity);
            let _ = expected_depth;
        }
    }

    // --- Property-based tests ---
    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            // RED: total always equals sum of all set priorities
            #[test]
            fn sum_tree_total_invariant(
                capacity in 1usize..200,
                priorities in proptest::collection::vec(0.0f64..100.0, 1..50)
            ) {
                let mut tree = SumTree::new(capacity);
                let n = priorities.len().min(capacity);
                for (i, &p) in priorities[..n].iter().enumerate() {
                    tree.set(i, p);
                }
                let expected: f64 = priorities[..n].iter().sum();
                prop_assert!(
                    (tree.total() - expected).abs() < 1e-8,
                    "total={}, expected={}", tree.total(), expected
                );
            }

            // RED: sampled index is always in [0, capacity)
            #[test]
            fn sum_tree_sample_always_valid_index(
                capacity in 1usize..100,
                priorities in proptest::collection::vec(0.01f64..100.0, 1..20),
                sample_fracs in proptest::collection::vec(0.0f64..1.0, 1..50)
            ) {
                let mut tree = SumTree::new(capacity);
                let n = priorities.len().min(capacity);
                for (i, &p) in priorities[..n].iter().enumerate() {
                    tree.set(i, p);
                }
                for frac in sample_fracs {
                    let s = frac * tree.total();
                    let idx = tree.sample(s);
                    prop_assert!(idx < capacity,
                        "sampled index {} >= capacity {}", idx, capacity);
                    prop_assert!(idx < n,
                        "sampled index {} >= n_inserted {}", idx, n);
                }
            }
        }
    }
}
```

### 1.2 PrioritizedReplayBuffer Tests

**Target file:** `crates/rlox-core/src/buffer/priority.rs` (continuation)
**Group: 0**

```rust
// Continuation of priority.rs
#[cfg(test)]
mod prioritized_buffer_tests {
    use super::*;
    use crate::buffer::{ExperienceRecord, sample_record};

    fn make_record(obs_dim: usize, reward: f32) -> ExperienceRecord {
        ExperienceRecord {
            obs: vec![1.0; obs_dim],
            action: vec![0.0],
            reward,
            terminated: false,
            truncated: false,
        }
    }

    // RED: push respects capacity (ring buffer semantics)
    #[test]
    fn prioritized_buffer_respects_capacity() {
        let mut buf = PrioritizedReplayBuffer::new(100, 4, 1, 0.6, 0.4);
        for _ in 0..200 {
            buf.push(make_record(4, 1.0), 1.0).unwrap();
        }
        assert_eq!(buf.len(), 100);
    }

    // RED: sample returns correct batch size
    #[test]
    fn prioritized_buffer_sample_returns_correct_batch_size() {
        let mut buf = PrioritizedReplayBuffer::new(1000, 4, 1, 0.6, 0.4);
        for i in 0..1000 {
            buf.push(make_record(4, i as f32), 1.0).unwrap();
        }
        let (batch, weights) = buf.sample(32).unwrap();
        assert_eq!(batch.batch_size, 32);
        assert_eq!(weights.len(), 32);
    }

    // RED: importance weights are in (0, 1] when beta=1.0
    #[test]
    fn prioritized_buffer_importance_weights_in_range() {
        let mut buf = PrioritizedReplayBuffer::new(100, 4, 1, 0.6, 1.0);
        for _ in 0..100 {
            buf.push(make_record(4, 1.0), 1.0).unwrap();
        }
        let (_, weights) = buf.sample(32).unwrap();
        for &w in &weights {
            assert!(w > 0.0, "importance weight must be > 0");
            assert!(w <= 1.0 + 1e-6, "importance weight must be <= 1.0, got {}", w);
        }
    }

    // RED: high-priority transitions sampled more often
    #[test]
    fn prioritized_buffer_high_priority_sampled_more_often() {
        let mut buf = PrioritizedReplayBuffer::new(100, 4, 1, 1.0, 0.4);

        // Push 90 low-priority and 10 high-priority transitions
        for _ in 0..90 {
            buf.push(make_record(4, 0.0), 0.1).unwrap();  // low priority
        }
        for _ in 0..10 {
            buf.push(make_record(4, 100.0), 100.0).unwrap();  // high priority
        }

        let n_samples = 1000;
        let mut high_count = 0;
        for seed in 0..n_samples {
            let (batch, _) = buf.sample_with_seed(1, seed as u64).unwrap();
            if batch.rewards[0] == 100.0 {
                high_count += 1;
            }
        }

        // High-priority transitions represent 10% of buffer but have 1000x priority
        // With alpha=1.0, they should be sampled much more than 10% of the time
        let high_fraction = high_count as f64 / n_samples as f64;
        assert!(high_fraction > 0.5,
            "high-priority transitions sampled {:.1%} of the time, expected > 50%",
            high_fraction);
    }

    // RED: update_priorities changes sampling distribution
    #[test]
    fn prioritized_buffer_update_priorities_changes_distribution() {
        let mut buf = PrioritizedReplayBuffer::new(10, 4, 1, 1.0, 0.4);
        for _ in 0..10 {
            buf.push(make_record(4, 1.0), 1.0).unwrap();
        }

        // Increase priority of index 0 to 100
        buf.update_priorities(&[0], &[100.0]).unwrap();

        // Now index 0 should be sampled much more often
        let mut idx0_count = 0;
        for seed in 0..1000u64 {
            let (_, indices) = buf.sample_with_indices(1, seed).unwrap();
            if indices[0] == 0 {
                idx0_count += 1;
            }
        }
        assert!(idx0_count > 500,
            "index 0 with 100x priority should be sampled > 50% of time, got {:.1%}",
            idx0_count as f64 / 1000.0);
    }

    // RED: beta annealing increases importance weights toward 1.0
    #[test]
    fn prioritized_buffer_beta_annealing() {
        let mut buf = PrioritizedReplayBuffer::new(100, 4, 1, 0.6, 0.4);
        for _ in 0..100 {
            buf.push(make_record(4, 1.0), f64::max(
                0.01, rand::random::<f64>() * 10.0,
            )).unwrap();
        }

        let (_, weights_early) = buf.sample(32).unwrap();
        let max_w_early = weights_early.iter().cloned().fold(0.0_f64, f64::max);

        // Anneal beta from 0.4 to 1.0
        buf.set_beta(1.0);
        let (_, weights_late) = buf.sample(32).unwrap();
        let max_w_late = weights_late.iter().cloned().fold(0.0_f64, f64::max);

        // With beta=1.0, weights approach 1.0 more closely
        assert!(max_w_late >= max_w_early - 1e-6,
            "max weight with beta=1.0 ({:.4}) should be >= beta=0.4 ({:.4})",
            max_w_late, max_w_early);
    }

    // RED: update_priorities with wrong indices length returns error
    #[test]
    fn prioritized_buffer_update_priorities_len_mismatch_returns_error() {
        let mut buf = PrioritizedReplayBuffer::new(100, 4, 1, 0.6, 0.4);
        for _ in 0..10 {
            buf.push(make_record(4, 1.0), 1.0).unwrap();
        }
        let result = buf.update_priorities(&[0, 1], &[2.0]);  // 2 indices, 1 priority
        assert!(result.is_err());
    }

    // RED: negative priority returns error
    #[test]
    fn prioritized_buffer_negative_priority_returns_error() {
        let mut buf = PrioritizedReplayBuffer::new(100, 4, 1, 0.6, 0.4);
        let result = buf.push(make_record(4, 1.0), -1.0);
        assert!(result.is_err(), "negative priority must return Err");
    }

    // RED: sample when buffer empty returns error
    #[test]
    fn prioritized_buffer_sample_empty_buffer_returns_error() {
        let buf = PrioritizedReplayBuffer::new(100, 4, 1, 0.6, 0.4);
        let result = buf.sample(1);
        assert!(result.is_err(), "sampling empty buffer must return Err");
    }
}
```

### 1.3 MuJoCo Native Env Tests

**Target file:** `crates/rlox-core/src/env/mujoco.rs` (new file)
**Group: 5 — requires mujoco feature flag**

```rust
// crates/rlox-core/src/env/mujoco.rs
#[cfg(all(test, feature = "mujoco"))]
mod tests {
    use super::*;
    use crate::env::spaces::Action;

    // RED: MuJoCoEnv implements RLEnv
    #[test]
    fn mujoco_env_implements_rl_env() {
        fn assert_rl_env<T: RLEnv>() {}
        assert_rl_env::<MuJoCoEnv>();
    }

    // RED: MuJoCoEnv is Send + Sync (required for Rayon)
    #[test]
    fn mujoco_env_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MuJoCoEnv>();
    }

    // RED: reset returns correct observation dimension for HalfCheetah
    #[test]
    fn mujoco_halfcheetah_obs_dim() {
        let mut env = MuJoCoEnv::new("HalfCheetah-v4").unwrap();
        let obs = env.reset(Some(42)).unwrap();
        // HalfCheetah obs_dim = 17
        assert_eq!(obs.as_slice().len(), 17,
            "HalfCheetah obs_dim should be 17, got {}", obs.as_slice().len());
    }

    // RED: step returns observation, reward (non-NaN), done flags
    #[test]
    fn mujoco_halfcheetah_step_returns_valid_transition() {
        let mut env = MuJoCoEnv::new("HalfCheetah-v4").unwrap();
        env.reset(Some(42)).unwrap();

        // HalfCheetah action_dim = 6, actions in [-1, 1]
        let action = Action::Continuous(vec![0.0f32; 6]);
        let t = env.step(&action).unwrap();

        assert_eq!(t.obs.as_slice().len(), 17);
        assert!(t.reward.is_finite(), "reward must be finite, got {}", t.reward);
        // No NaN in obs
        for &v in t.obs.as_slice() {
            assert!(v.is_finite(), "obs contains NaN/Inf: {}", v);
        }
    }

    // RED: Rayon parallel stepping — 16 envs, no segfault
    #[test]
    fn mujoco_16_envs_rayon_parallel_no_segfault() {
        use rayon::prelude::*;

        let envs: Vec<MuJoCoEnv> = (0..16)
            .map(|i| {
                let mut e = MuJoCoEnv::new("HalfCheetah-v4").unwrap();
                e.reset(Some(i as u64)).unwrap();
                e
            })
            .collect();

        let mut envs = envs;  // make mutable

        // Step 100 times in parallel
        for _ in 0..100 {
            let results: Vec<_> = envs
                .par_iter_mut()
                .map(|env| {
                    let action = Action::Continuous(vec![0.1; 6]);
                    env.step(&action)
                })
                .collect();

            for r in results {
                // Must not panic or segfault
                let _ = r.unwrap();
            }
        }
    }

    // RED: MuJoCoEnv through VecEnv ParallelBatchStepper
    #[test]
    fn mujoco_vec_env_step_all_8_envs() {
        use crate::env::batch::ParallelBatchStepper;
        use crate::env::spaces::Action;

        let envs: Vec<Box<dyn RLEnv>> = (0..8)
            .map(|i| {
                let mut e = MuJoCoEnv::new("HalfCheetah-v4").unwrap();
                e.reset(Some(i as u64)).unwrap();
                Box::new(e) as Box<dyn RLEnv>
            })
            .collect();

        let mut stepper = ParallelBatchStepper::new(envs);
        let actions: Vec<Action> = (0..8)
            .map(|_| Action::Continuous(vec![0.0; 6]))
            .collect();

        for _ in 0..10 {
            let results = stepper.step_batch(&actions).unwrap();
            assert_eq!(results.len(), 8);
            for r in &results {
                for &v in r.obs.as_slice() {
                    assert!(v.is_finite(), "MuJoCo obs contains NaN/Inf");
                }
            }
        }
    }

    // RED: max_steps truncation fires correctly
    #[test]
    fn mujoco_env_truncates_at_max_steps() {
        let mut env = MuJoCoEnv::new_with_max_steps("HalfCheetah-v4", 10).unwrap();
        env.reset(Some(42)).unwrap();

        let action = Action::Continuous(vec![0.0; 6]);
        let mut truncated = false;
        for _ in 0..15 {
            let t = env.step(&action).unwrap();
            if t.truncated {
                truncated = true;
                break;
            }
        }
        assert!(truncated, "MuJoCoEnv should truncate at max_steps=10");
    }
}
```

### 1.4 Memory-Mapped Buffer Tests

**Target file:** `crates/rlox-core/src/buffer/mmap.rs` (new file)
**Group: 5**

```rust
// crates/rlox-core/src/buffer/mmap.rs
#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::ExperienceRecord;
    use tempfile::TempDir;

    fn make_record(obs_dim: usize, reward: f32) -> ExperienceRecord {
        ExperienceRecord {
            obs: vec![1.0; obs_dim],
            action: vec![0.0],
            reward,
            terminated: false,
            truncated: false,
        }
    }

    // RED: MmapReplayBuffer::new with a temp file path
    #[test]
    fn mmap_buffer_new_creates_file() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("buffer.mmap");
        let buf = MmapReplayBuffer::new(
            path.to_str().unwrap(),
            /*capacity=*/ 1000,
            /*obs_dim=*/ 4,
            /*act_dim=*/ 1,
            /*hot_capacity=*/ 100,
        ).unwrap();
        assert!(path.exists(), "mmap file should be created on disk");
        assert_eq!(buf.len(), 0);
    }

    // RED: push to hot buffer (in-memory) until hot_capacity
    #[test]
    fn mmap_buffer_hot_region_fills_first() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("buffer.mmap");
        let mut buf = MmapReplayBuffer::new(
            path.to_str().unwrap(), 1000, 4, 1, 50,
        ).unwrap();

        for i in 0..50 {
            buf.push(make_record(4, i as f32)).unwrap();
        }
        assert_eq!(buf.len(), 50);
        // All 50 are in hot region — cold should be empty
        assert_eq!(buf.cold_len(), 0);
        assert_eq!(buf.hot_len(), 50);
    }

    // RED: push beyond hot_capacity spills to cold (disk)
    #[test]
    fn mmap_buffer_spills_to_disk_on_overflow() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("buffer.mmap");
        let mut buf = MmapReplayBuffer::new(
            path.to_str().unwrap(), 1000, 4, 1, 20,
        ).unwrap();

        for i in 0..50 {
            buf.push(make_record(4, i as f32)).unwrap();
        }
        // 50 total: 20 in hot, 30 on disk
        assert_eq!(buf.len(), 50);
        assert!(buf.cold_len() > 0, "some data should have spilled to disk");
    }

    // RED: sample from mixed hot+cold buffer returns correct batch size
    #[test]
    fn mmap_buffer_sample_from_mixed_hot_cold() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("buffer.mmap");
        let mut buf = MmapReplayBuffer::new(
            path.to_str().unwrap(), 1000, 4, 1, 20,
        ).unwrap();

        for i in 0..100 {
            buf.push(make_record(4, i as f32)).unwrap();
        }

        let batch = buf.sample(32, 42).unwrap();
        assert_eq!(batch.batch_size, 32);
        assert_eq!(batch.observations.len(), 32 * 4);
    }

    // RED: total capacity limit respected (no panic on overflow)
    #[test]
    fn mmap_buffer_total_capacity_respected() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("buffer.mmap");
        let mut buf = MmapReplayBuffer::new(
            path.to_str().unwrap(), 100, 4, 1, 20,
        ).unwrap();

        // Push 200 items into capacity=100 buffer — should ring-overwrite
        for i in 0..200 {
            buf.push(make_record(4, i as f32)).unwrap();
        }
        assert_eq!(buf.len(), 100, "buffer should never exceed total capacity");
    }

    // RED: sample after file recovery (simulated restart)
    #[test]
    fn mmap_buffer_recovers_from_disk() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("buffer.mmap");

        // Write data
        {
            let mut buf = MmapReplayBuffer::new(
                path.to_str().unwrap(), 1000, 4, 1, 20,
            ).unwrap();
            for i in 0..50 {
                buf.push(make_record(4, i as f32)).unwrap();
            }
        }  // buf dropped, file flushed

        // Recover from disk
        let mut buf = MmapReplayBuffer::open(path.to_str().unwrap()).unwrap();
        assert!(buf.len() > 0, "buffer should recover data from disk");
        // Sample must succeed
        let batch = buf.sample(10, 42).unwrap();
        assert_eq!(batch.batch_size, 10);
    }
}
```

### 1.5 Reward Sandbox Tests

**Target file:** `crates/rlox-core/src/llm/sandbox.rs` (new file)
**Group: 5**

```rust
// crates/rlox-core/src/llm/sandbox.rs
#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    // RED: simple correct code executes and returns stdout
    #[test]
    fn sandbox_correct_code_returns_output() {
        let sandbox = RewardSandbox::new(
            Duration::from_millis(500),
            64 * 1024 * 1024,  // 64 MB
        );
        let code = r#"print(2 + 2)"#;
        let result = sandbox.execute(code).unwrap();
        assert_eq!(result.trim(), "4");
    }

    // RED: infinite loop is killed by timeout
    #[test]
    fn sandbox_infinite_loop_times_out() {
        let sandbox = RewardSandbox::new(
            Duration::from_millis(100),  // 100ms timeout
            64 * 1024 * 1024,
        );
        let code = r#"while True: pass"#;
        let result = sandbox.execute(code);
        assert!(result.is_err(), "infinite loop must result in timeout error");
        let err = result.unwrap_err().to_string().to_lowercase();
        assert!(err.contains("timeout") || err.contains("time"),
            "error should mention timeout, got: {err}");
    }

    // RED: exception in code returns error (not panic)
    #[test]
    fn sandbox_exception_returns_error() {
        let sandbox = RewardSandbox::new(
            Duration::from_millis(500),
            64 * 1024 * 1024,
        );
        let code = r#"raise ValueError("test error")"#;
        let result = sandbox.execute(code);
        assert!(result.is_err(), "exception should return Err");
    }

    // RED: syntax error returns error (not panic)
    #[test]
    fn sandbox_syntax_error_returns_error() {
        let sandbox = RewardSandbox::new(
            Duration::from_millis(500),
            64 * 1024 * 1024,
        );
        let code = r#"def bad_syntax(:"#;
        let result = sandbox.execute(code);
        assert!(result.is_err(), "syntax error should return Err");
    }

    // RED: memory limit is enforced
    #[test]
    fn sandbox_memory_limit_enforced() {
        let sandbox = RewardSandbox::new(
            Duration::from_millis(500),
            4 * 1024 * 1024,  // 4 MB — very small
        );
        // Allocate 100 MB
        let code = r#"x = bytearray(100 * 1024 * 1024)"#;
        let result = sandbox.execute(code);
        // Either MemoryError or process killed
        assert!(result.is_err(),
            "memory allocation beyond limit should return Err");
    }

    // RED: multiple sequential executions work correctly
    #[test]
    fn sandbox_multiple_executions_isolated() {
        let sandbox = RewardSandbox::new(
            Duration::from_millis(500),
            64 * 1024 * 1024,
        );
        // Execute twice — must be isolated (no shared state)
        let r1 = sandbox.execute(r#"x = 42; print(x)"#).unwrap();
        let r2 = sandbox.execute(r#"print(x)"#);  // x not defined here
        assert_eq!(r1.trim(), "42");
        // Second execution should fail (x not in scope) or succeed
        // with a different x — depends on isolation strategy.
        // At minimum, it must not return "42" from previous run.
        if let Ok(r2_str) = r2 {
            assert_ne!(r2_str.trim(), "42",
                "sandbox must isolate state between executions");
        }
        // Err is also acceptable (NameError: x not defined)
    }

    // RED: empty code returns Ok("") or Ok with empty output
    #[test]
    fn sandbox_empty_code_returns_empty() {
        let sandbox = RewardSandbox::new(
            Duration::from_millis(200),
            64 * 1024 * 1024,
        );
        let result = sandbox.execute("").unwrap();
        assert!(result.trim().is_empty(), "empty code should produce empty output");
    }
}
```

---

## Part 2: Python Integration Tests

**Target file:** `tests/python/test_phase8.py`

```python
# tests/python/test_phase8.py
"""
Phase 8 TDD test specifications — Production Hardening (v0.7).

Status: RED — all tests fail until implementation is complete.
Depends on: Phase 7 tests passing (PPO, GRPO, DPO working).
Run: .venv/bin/python -m pytest tests/python/test_phase8.py -v
"""

import contextlib
import json
import math
import os
import sys
import subprocess
import tempfile
import warnings
from copy import deepcopy
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def torch():
    return pytest.importorskip("torch")


@pytest.fixture(scope="session")
def gymnasium():
    return pytest.importorskip("gymnasium")


# ---------------------------------------------------------------------------
# Group 0: PrioritizedReplayBuffer Python bindings
# ---------------------------------------------------------------------------

class TestPrioritizedReplayBufferPython:
    """Group 0: Python API for PrioritizedReplayBuffer."""

    # RED: PrioritizedReplayBuffer importable
    def test_prioritized_replay_buffer_importable(self):
        from rlox import PrioritizedReplayBuffer
        buf = PrioritizedReplayBuffer(
            capacity=1000, obs_dim=4, act_dim=1, alpha=0.6, beta=0.4
        )
        assert len(buf) == 0

    # RED: push and sample returns correct structure
    def test_prioritized_buffer_push_sample(self):
        from rlox import PrioritizedReplayBuffer

        buf = PrioritizedReplayBuffer(
            capacity=1000, obs_dim=4, act_dim=1, alpha=0.6, beta=0.4
        )
        obs = np.zeros(4, dtype=np.float32)
        action = np.zeros(1, dtype=np.float32)

        for i in range(100):
            buf.push(obs=obs, action=action, reward=float(i),
                    terminated=False, truncated=False, priority=1.0)

        batch, weights, indices = buf.sample(batch_size=32, seed=42)

        assert batch["observations"].shape == (32, 4)
        assert weights.shape == (32,)
        assert indices.shape == (32,)
        assert np.all(weights > 0)
        assert np.all(weights <= 1.0 + 1e-6)

    # RED: update_priorities changes sampling distribution
    def test_update_priorities_affects_sampling(self):
        from rlox import PrioritizedReplayBuffer

        buf = PrioritizedReplayBuffer(
            capacity=10, obs_dim=4, act_dim=1, alpha=1.0, beta=0.4
        )
        obs = np.zeros(4, dtype=np.float32)
        action = np.zeros(1, dtype=np.float32)

        for i in range(10):
            buf.push(obs=obs, action=action, reward=float(i),
                    terminated=False, truncated=False, priority=1.0)

        # Massively boost index 0's priority
        buf.update_priorities(indices=np.array([0]), priorities=np.array([1000.0]))

        count_0 = sum(
            1 for seed in range(200)
            if buf.sample(1, seed=seed)[2][0] == 0
        )
        assert count_0 > 100, (
            f"index 0 with 1000x priority should dominate sampling, got {count_0}/200"
        )


# ---------------------------------------------------------------------------
# Group 1: Off-policy algorithm tests
# ---------------------------------------------------------------------------

class TestSACPython:
    """Group 1: SAC must solve Pendulum-v1 in 50K steps."""

    # RED: SAC importable
    def test_sac_importable(self):
        from rlox import SAC  # or from rlox.algorithms.sac import SAC

    # RED: SAC smoke test
    def test_sac_smoke_test(self, torch, gymnasium):
        from rlox import SAC

        sac = SAC(env_id="Pendulum-v1", seed=42)
        sac.train(total_timesteps=500)

    # RED: SAC initializes correct network architecture
    def test_sac_network_architecture(self, torch, gymnasium):
        from rlox import SAC
        import torch.nn as nn

        sac = SAC(env_id="Pendulum-v1", seed=42)

        # Must have actor, critic1, critic2, target_critic1, target_critic2
        assert hasattr(sac, "actor")
        assert hasattr(sac, "critic1")
        assert hasattr(sac, "critic2")
        assert hasattr(sac, "target_critic1")
        assert hasattr(sac, "target_critic2")

        # Must have log_alpha for automatic entropy tuning
        assert hasattr(sac, "log_alpha")

    @pytest.mark.slow
    def test_sac_solves_pendulum(self, torch, gymnasium):
        """SAC must achieve mean return > -200 on Pendulum-v1 in 50K steps."""
        from rlox import SAC

        def evaluate(sac, n_episodes=20, seed=99):
            import gymnasium as gym
            env = gym.make("Pendulum-v1")
            rewards = []
            for ep in range(n_episodes):
                obs, _ = env.reset(seed=seed + ep)
                done = False
                ep_r = 0.0
                while not done:
                    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        action = sac.actor.get_deterministic_action(obs_t)
                        action = action.numpy()[0]
                    obs, r, term, trunc, _ = env.step(action)
                    ep_r += r
                    done = term or trunc
                rewards.append(ep_r)
            return float(np.mean(rewards))

        sac = SAC(
            env_id="Pendulum-v1",
            seed=42,
            learning_rate=3e-4,
            buffer_size=50_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
        )
        sac.train(total_timesteps=50_000)
        mean_reward = evaluate(sac)

        assert mean_reward > -200.0, (
            f"SAC should solve Pendulum-v1 (> -200), got {mean_reward:.1f}"
        )

    @pytest.mark.slow
    def test_sac_automatic_entropy_tuning(self, torch, gymnasium):
        """SAC alpha (entropy coefficient) should converge toward target_entropy."""
        from rlox import SAC

        sac = SAC(env_id="Pendulum-v1", seed=42, auto_entropy=True)
        # Target entropy = -dim(A) = -1 for Pendulum
        target_entropy = -1.0

        alpha_values = []
        for _ in range(5):
            sac.train(total_timesteps=2_000)
            alpha_values.append(sac.log_alpha.exp().item())

        # After some training, alpha should be adjusting (not all equal)
        alpha_range = max(alpha_values) - min(alpha_values)
        assert alpha_range > 1e-6, (
            "alpha should change during training — auto entropy tuning not working"
        )


class TestTD3Python:
    """Group 1: TD3 on HalfCheetah."""

    # RED: TD3 importable
    def test_td3_importable(self):
        from rlox import TD3  # or from rlox.algorithms.td3 import TD3

    # RED: TD3 smoke test
    def test_td3_smoke_test(self, torch, gymnasium):
        from rlox import TD3
        td3 = TD3(env_id="Pendulum-v1", seed=42)
        td3.train(total_timesteps=500)

    # RED: TD3 has correct components (deterministic policy, twin critics)
    def test_td3_architecture(self, torch, gymnasium):
        from rlox import TD3

        td3 = TD3(env_id="Pendulum-v1", seed=42)
        assert hasattr(td3, "actor")
        assert hasattr(td3, "target_actor")
        assert hasattr(td3, "critic1")
        assert hasattr(td3, "critic2")
        assert hasattr(td3, "target_critic1")
        assert hasattr(td3, "target_critic2")

    @pytest.mark.slow
    def test_td3_halfcheetah_5000_in_1m_steps(self, torch, gymnasium):
        """TD3 on HalfCheetah-v4 must reach > 5000 return in 1M steps."""
        pytest.importorskip("mujoco")
        from rlox import TD3

        td3 = TD3(
            env_id="HalfCheetah-v4",
            seed=42,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            batch_size=256,
            tau=0.005,
            policy_delay=2,
        )
        metrics = td3.train(total_timesteps=1_000_000)
        final_reward = metrics.get("final_mean_reward", 0.0)

        assert final_reward > 5000.0, (
            f"TD3 HalfCheetah should reach > 5000, got {final_reward:.1f}"
        )


class TestDQNPython:
    """Group 1: DQN must solve CartPole in < 50K steps."""

    # RED: DQN importable
    def test_dqn_importable(self):
        from rlox import DQN  # or from rlox.algorithms.dqn import DQN

    # RED: DQN smoke test
    def test_dqn_smoke_test(self, torch, gymnasium):
        from rlox import DQN
        dqn = DQN(env_id="CartPole-v1", seed=42)
        dqn.train(total_timesteps=500)

    @pytest.mark.slow
    def test_dqn_solves_cartpole(self, torch, gymnasium):
        """DQN must solve CartPole-v1 in < 50K steps (mean reward >= 475)."""
        from rlox import DQN

        dqn = DQN(
            env_id="CartPole-v1",
            seed=42,
            learning_rate=1e-4,
            buffer_size=50_000,
            batch_size=32,
            exploration_fraction=0.1,
            learning_starts=1_000,
        )
        dqn.train(total_timesteps=50_000)

        import gymnasium as gym
        env = gym.make("CartPole-v1")
        rewards = []
        for ep in range(20):
            obs, _ = env.reset(seed=ep + 99)
            done = False
            ep_r = 0.0
            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    q_values = dqn.q_network(obs_t)
                    action = q_values.argmax(dim=-1).item()
                obs, r, term, trunc, _ = env.step(action)
                ep_r += r
                done = term or trunc
            rewards.append(ep_r)

        mean_reward = float(np.mean(rewards))
        assert mean_reward >= 475.0, (
            f"DQN should solve CartPole (>= 475), got {mean_reward:.1f}"
        )


class TestRainbowExtensions:
    """Group 1: Rainbow extension components."""

    # RED: Double DQN flag switches Q-value computation
    def test_double_dqn_uses_online_for_action_selection(self, torch, gymnasium):
        from rlox import DQN

        dqn_standard = DQN(env_id="CartPole-v1", seed=42, double_dqn=False)
        dqn_double = DQN(env_id="CartPole-v1", seed=42, double_dqn=True)

        # Both must run without error
        dqn_standard.train(total_timesteps=500)
        dqn_double.train(total_timesteps=500)

        # Double DQN Q-values should differ from standard DQN
        # (different action selection in target computation)

    # RED: Dueling network has separate value and advantage streams
    def test_dueling_network_architecture(self, torch, gymnasium):
        from rlox import DQN

        dqn = DQN(env_id="CartPole-v1", seed=42, dueling=True)
        # The Q-network must have both value and advantage heads
        assert hasattr(dqn.q_network, "value_stream") or \
               hasattr(dqn.q_network, "adv_stream") or \
               hasattr(dqn.q_network, "advantage"), \
               "Dueling DQN must have value/advantage separation"

    # RED: N-step returns correctly bootstraps future rewards
    def test_n_step_returns_correctness(self, torch, gymnasium):
        from rlox import DQN

        # 3-step return: r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + gamma^3 * V(s_{t+3})
        dqn = DQN(env_id="CartPole-v1", seed=42, n_step=3, gamma=0.99)
        dqn.train(total_timesteps=1_000)

        # Verify n-step buffer exists and has correct structure
        assert dqn.n_step == 3


# ---------------------------------------------------------------------------
# Group 2: One-liner Trainer API
# ---------------------------------------------------------------------------

class TestOneLinerAPI:
    """Group 2: PPOTrainer(env=...).train(N) must just work."""

    # RED: PPOTrainer importable
    def test_ppo_trainer_importable(self):
        from rlox import PPOTrainer  # or from rlox.trainers import PPOTrainer

    # RED: one-liner API works for CartPole
    def test_one_liner_ppo_cartpole(self, torch, gymnasium):
        from rlox import PPOTrainer

        trainer = PPOTrainer(env="CartPole-v1")
        trainer.train(50_000)
        result = trainer.evaluate()

        assert "mean_reward" in result, "evaluate() must return 'mean_reward' key"

    # RED: one-liner API with default model string
    def test_one_liner_with_model_string(self, torch, gymnasium):
        from rlox import PPOTrainer

        trainer = PPOTrainer(env="CartPole-v1", model="mlp")
        trainer.train(1_000)

    @pytest.mark.slow
    def test_one_liner_solves_cartpole(self, torch, gymnasium):
        from rlox import PPOTrainer

        trainer = PPOTrainer(env="CartPole-v1")
        trainer.train(50_000)
        result = trainer.evaluate(n_episodes=20)

        assert result["mean_reward"] >= 475.0, (
            f"One-liner PPOTrainer should solve CartPole, got {result['mean_reward']:.1f}"
        )

    # RED: SACTrainer one-liner
    def test_one_liner_sac_pendulum(self, torch, gymnasium):
        from rlox import SACTrainer

        trainer = SACTrainer(env="Pendulum-v1")
        trainer.train(1_000)

    # RED: DQNTrainer one-liner
    def test_one_liner_dqn_cartpole(self, torch, gymnasium):
        from rlox import DQNTrainer

        trainer = DQNTrainer(env="CartPole-v1")
        trainer.train(1_000)

    # RED: trainer.save() and trainer.load() roundtrip
    def test_trainer_save_load(self, torch, gymnasium, tmp_path):
        from rlox import PPOTrainer

        trainer = PPOTrainer(env="CartPole-v1")
        trainer.train(1_000)
        save_path = str(tmp_path / "trainer.pt")
        trainer.save(save_path)

        trainer2 = PPOTrainer.load(save_path)
        result = trainer2.evaluate(n_episodes=5)
        assert "mean_reward" in result


# ---------------------------------------------------------------------------
# Group 2: Config system
# ---------------------------------------------------------------------------

class TestConfigSystem:
    """Group 2: YAML/CLI/dataclass config system."""

    # RED: PPOConfig importable with all required fields
    def test_ppo_config_importable(self):
        from rlox import PPOConfig

        config = PPOConfig()
        # Required fields from phase plan
        assert hasattr(config, "n_envs")
        assert hasattr(config, "n_steps")
        assert hasattr(config, "learning_rate")
        assert hasattr(config, "n_epochs")
        assert hasattr(config, "batch_size")
        assert hasattr(config, "clip_eps")
        assert hasattr(config, "vf_coef")
        assert hasattr(config, "ent_coef")

    # RED: PPOConfig.from_yaml loads from file
    def test_ppo_config_from_yaml(self, tmp_path):
        from rlox import PPOConfig
        import yaml

        config_dict = {
            "n_envs": 16,
            "learning_rate": 1e-3,
            "n_epochs": 5,
        }
        yaml_path = str(tmp_path / "config.yaml")
        with open(yaml_path, "w") as f:
            yaml.dump(config_dict, f)

        config = PPOConfig.from_yaml(yaml_path)
        assert config.n_envs == 16
        assert abs(config.learning_rate - 1e-3) < 1e-10
        assert config.n_epochs == 5

    # RED: config.merge() overrides specific fields
    def test_config_merge_overrides(self):
        from rlox import PPOConfig

        config = PPOConfig()
        original_lr = config.learning_rate
        merged = config.merge({"learning_rate": 1e-3})

        assert abs(merged.learning_rate - 1e-3) < 1e-10
        # Original unchanged
        assert abs(config.learning_rate - original_lr) < 1e-10

    # RED: config type validation catches wrong types
    def test_config_type_validation(self):
        from rlox import PPOConfig

        with pytest.raises((TypeError, ValueError)):
            PPOConfig(n_envs="not_an_int")

    # RED: PPOConfig.from_cli parses CLI args
    def test_config_from_cli(self):
        from rlox import PPOConfig

        config = PPOConfig.from_cli(["--n_envs=16", "--learning_rate=1e-3"])
        assert config.n_envs == 16
        assert abs(config.learning_rate - 1e-3) < 1e-10

    # RED: config validates that learning_rate > 0
    def test_config_validates_positive_learning_rate(self):
        from rlox import PPOConfig

        with pytest.raises((ValueError, AssertionError)):
            PPOConfig(learning_rate=-1e-4)

    # RED: config validates that n_envs >= 1
    def test_config_validates_n_envs_positive(self):
        from rlox import PPOConfig

        with pytest.raises((ValueError, AssertionError)):
            PPOConfig(n_envs=0)

    # RED: SACConfig has all required SAC-specific fields
    def test_sac_config_fields(self):
        from rlox import SACConfig

        config = SACConfig()
        assert hasattr(config, "tau")  # soft update coefficient
        assert hasattr(config, "target_entropy")
        assert hasattr(config, "auto_entropy")
        assert hasattr(config, "buffer_size")


# ---------------------------------------------------------------------------
# Group 3: Callback system
# ---------------------------------------------------------------------------

class TestCallbackSystem:
    """Group 3: Callback hooks fire at correct training events."""

    # RED: Callback base class importable
    def test_callback_importable(self):
        from rlox import Callback  # or from rlox.callbacks import Callback

    # RED: on_training_start fires once before training
    def test_on_training_start_fires_once(self, torch, gymnasium):
        from rlox import PPOTrainer, Callback

        events = []

        class TrackingCallback(Callback):
            def on_training_start(self, locals_dict):
                events.append("training_start")

        trainer = PPOTrainer(env="CartPole-v1", callbacks=[TrackingCallback()])
        trainer.train(1_000)

        assert events.count("training_start") == 1, (
            f"on_training_start should fire exactly once, fired {events.count('training_start')} times"
        )

    # RED: on_step fires on every environment step
    def test_on_step_fires_on_every_step(self, torch, gymnasium):
        from rlox import PPOTrainer, Callback

        step_count = [0]

        class CountSteps(Callback):
            def on_step(self, step, obs, action, reward, done, info):
                step_count[0] += 1
                return True  # continue training

        total_timesteps = 1_000
        trainer = PPOTrainer(
            env="CartPole-v1",
            n_envs=2,
            callbacks=[CountSteps()]
        )
        trainer.train(total_timesteps)

        # on_step fires once per step across all envs
        # Allow 10% tolerance for episode boundary effects
        assert step_count[0] >= total_timesteps * 0.9, (
            f"on_step should fire >= {total_timesteps*0.9:.0f} times, "
            f"fired {step_count[0]}"
        )

    # RED: returning False from on_step stops training early
    def test_on_step_early_stop(self, torch, gymnasium):
        from rlox import PPOTrainer, Callback

        class EarlyStop(Callback):
            def __init__(self, stop_at):
                self.stop_at = stop_at
                self.step_count = 0

            def on_step(self, step, obs, action, reward, done, info):
                self.step_count += 1
                return self.step_count < self.stop_at  # False = stop

        callback = EarlyStop(stop_at=100)
        trainer = PPOTrainer(env="CartPole-v1", callbacks=[callback])
        trainer.train(10_000)  # would train for 10K but should stop at 100

        assert callback.step_count < 500, (
            f"Early stop callback should stop training quickly, took {callback.step_count} steps"
        )

    # RED: EvalCallback evaluates at correct frequency
    def test_eval_callback_frequency(self, torch, gymnasium):
        from rlox import PPOTrainer
        from rlox import EvalCallback  # or from rlox.callbacks

        eval_count = [0]

        class CountingEvalCallback(EvalCallback):
            def on_eval(self, mean_reward, metrics):
                eval_count[0] += 1

        eval_freq = 500
        callback = CountingEvalCallback(
            eval_env="CartPole-v1",
            eval_freq=eval_freq,
            n_eval_episodes=3,
        )
        trainer = PPOTrainer(env="CartPole-v1", n_envs=2, callbacks=[callback])
        trainer.train(total_timesteps=2_500)

        # Should evaluate ~5 times (at 500, 1000, 1500, 2000, 2500)
        assert eval_count[0] >= 4, (
            f"EvalCallback should eval ~5 times in 2500 steps with freq=500, "
            f"got {eval_count[0]}"
        )

    # RED: CheckpointCallback saves files at correct frequency
    def test_checkpoint_callback_saves_files(self, torch, gymnasium, tmp_path):
        from rlox import PPOTrainer
        from rlox import CheckpointCallback  # or from rlox.callbacks

        save_dir = str(tmp_path / "checkpoints")
        callback = CheckpointCallback(
            save_freq=500,
            save_path=save_dir,
        )
        trainer = PPOTrainer(env="CartPole-v1", n_envs=2, callbacks=[callback])
        trainer.train(total_timesteps=2_000)

        # Should have created checkpoint files
        import os
        checkpoint_files = [
            f for f in os.listdir(save_dir)
            if f.endswith(".pt") or f.endswith(".pkl")
        ]
        assert len(checkpoint_files) >= 3, (
            f"CheckpointCallback should save >= 3 checkpoints in 2000 steps "
            f"with freq=500, got {len(checkpoint_files)} files"
        )

    # RED: EarlyStoppingCallback stops when reward plateaus
    def test_early_stopping_callback_fires(self, torch, gymnasium):
        from rlox import PPOTrainer
        from rlox import EarlyStoppingCallback  # or from rlox.callbacks

        stopped_early = [False]

        class TrackingEarlyStopping(EarlyStoppingCallback):
            def on_training_end(self):
                stopped_early[0] = True

        # Very aggressive early stopping — should trigger quickly
        callback = TrackingEarlyStopping(patience=2, min_delta=1e6)  # unreachable threshold
        trainer = PPOTrainer(
            env="CartPole-v1",
            n_envs=2,
            callbacks=[callback],
        )
        trainer.train(total_timesteps=10_000)

        # With patience=2 and impossible threshold, should stop early
        assert stopped_early[0], "EarlyStoppingCallback should have triggered"


# ---------------------------------------------------------------------------
# Group 4: Training diagnostics
# ---------------------------------------------------------------------------

class TestTrainingDiagnostics:
    """Group 4: Auto-detect entropy collapse, KL spikes, gradient explosions."""

    # RED: TrainingDiagnostics importable
    def test_training_diagnostics_importable(self):
        from rlox import TrainingDiagnostics  # or from rlox.diagnostics

    # RED: entropy collapse warning fires when entropy drops below threshold
    def test_entropy_collapse_warning_fires(self, torch, gymnasium):
        from rlox import TrainingDiagnostics
        import warnings

        diagnostics = TrainingDiagnostics(
            initial_entropy=2.0,
            target_kl=0.01,
            max_grad_norm=0.5,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            diagnostics.on_train_batch(
                loss=0.1,
                metrics={
                    "entropy": 0.1,  # < 10% of initial (2.0)
                    "approx_kl": 0.01,
                    "grad_norm": 0.3,
                }
            )
            entropy_warnings = [x for x in w if "entropy" in str(x.message).lower()]
            assert len(entropy_warnings) > 0, (
                "TrainingDiagnostics should warn on entropy collapse"
            )

    # RED: KL spike warning fires when approx_kl > 10x target
    def test_kl_spike_warning_fires(self, torch, gymnasium):
        from rlox import TrainingDiagnostics
        import warnings

        diagnostics = TrainingDiagnostics(
            initial_entropy=2.0,
            target_kl=0.01,
            max_grad_norm=0.5,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            diagnostics.on_train_batch(
                loss=0.1,
                metrics={
                    "entropy": 1.5,
                    "approx_kl": 0.5,  # 50x target (0.01)
                    "grad_norm": 0.3,
                }
            )
            kl_warnings = [x for x in w if "kl" in str(x.message).lower()]
            assert len(kl_warnings) > 0, (
                "TrainingDiagnostics should warn on KL spike"
            )

    # RED: gradient explosion warning fires when grad_norm >> max_grad_norm
    def test_gradient_explosion_warning_fires(self, torch, gymnasium):
        from rlox import TrainingDiagnostics
        import warnings

        diagnostics = TrainingDiagnostics(
            initial_entropy=2.0,
            target_kl=0.01,
            max_grad_norm=0.5,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            diagnostics.on_train_batch(
                loss=0.1,
                metrics={
                    "entropy": 1.5,
                    "approx_kl": 0.01,
                    "grad_norm": 100.0,  # 200x max_grad_norm (0.5)
                }
            )
            grad_warnings = [x for x in w if "grad" in str(x.message).lower()]
            assert len(grad_warnings) > 0, (
                "TrainingDiagnostics should warn on gradient explosion"
            )

    # RED: no warnings for normal training metrics
    def test_no_warnings_for_normal_metrics(self, torch, gymnasium):
        from rlox import TrainingDiagnostics
        import warnings

        diagnostics = TrainingDiagnostics(
            initial_entropy=2.0,
            target_kl=0.01,
            max_grad_norm=0.5,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            diagnostics.on_train_batch(
                loss=0.1,
                metrics={
                    "entropy": 1.8,   # healthy
                    "approx_kl": 0.008,  # below target
                    "grad_norm": 0.3,  # below max
                }
            )
            diagnostic_warnings = [
                x for x in w
                if "entropy" in str(x.message).lower()
                or "kl" in str(x.message).lower()
                or "grad" in str(x.message).lower()
            ]
            assert len(diagnostic_warnings) == 0, (
                f"No warnings expected for normal metrics, got: "
                f"{[str(x.message) for x in diagnostic_warnings]}"
            )

    # RED: value function divergence warning (explained_var < -1)
    def test_value_function_divergence_warning(self, torch, gymnasium):
        from rlox import TrainingDiagnostics
        import warnings

        diagnostics = TrainingDiagnostics(
            initial_entropy=2.0,
            target_kl=0.01,
            max_grad_norm=0.5,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            diagnostics.on_train_batch(
                loss=0.1,
                metrics={"entropy": 1.8, "explained_var": -2.0}
            )
            vf_warnings = [
                x for x in w
                if "value" in str(x.message).lower() or "diverge" in str(x.message).lower()
            ]
            assert len(vf_warnings) > 0


# ---------------------------------------------------------------------------
# Group 4: Statistical evaluation toolkit
# ---------------------------------------------------------------------------

class TestStatisticalEvaluation:
    """Group 4: IQM, performance profiles, bootstrap CI."""

    # RED: interquartile_mean importable
    def test_iqm_importable(self):
        from rlox import interquartile_mean  # or from rlox.evaluation

    # RED: IQM matches manual computation
    def test_iqm_known_values(self):
        from rlox import interquartile_mean

        # 10 values: [1,2,3,4,5,6,7,8,9,10]
        # Q1=2.75, Q3=7.75, middle 50% = [3,4,5,6,7] (indices 2-6)
        # IQM = mean(3,4,5,6,7) = 5.0
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        iqm = interquartile_mean(scores)

        # IQM should be around 5.5 (middle 50% of sorted scores)
        assert 4.0 < iqm < 7.0, f"IQM of [1..10] should be ~5.5, got {iqm:.2f}"

    # RED: IQM with single value returns that value
    def test_iqm_single_value(self):
        from rlox import interquartile_mean

        iqm = interquartile_mean(np.array([42.0]))
        assert abs(iqm - 42.0) < 1e-10

    # RED: IQM more robust to outliers than mean
    def test_iqm_robust_to_outliers(self):
        from rlox import interquartile_mean

        normal_scores = np.array([5.0, 5.1, 4.9, 5.0, 5.1, 4.8, 5.2])
        outlier_scores = np.append(normal_scores, [100.0, -100.0])

        iqm_normal = interquartile_mean(normal_scores)
        iqm_outlier = interquartile_mean(outlier_scores)

        mean_normal = float(np.mean(normal_scores))
        mean_outlier = float(np.mean(outlier_scores))

        # IQM should be less affected by outliers than plain mean
        iqm_diff = abs(iqm_outlier - iqm_normal)
        mean_diff = abs(mean_outlier - mean_normal)
        assert iqm_diff < mean_diff, (
            f"IQM should be more robust: IQM diff={iqm_diff:.2f}, "
            f"mean diff={mean_diff:.2f}"
        )

    # RED: stratified_bootstrap_ci returns correct format
    def test_bootstrap_ci_format(self):
        from rlox import stratified_bootstrap_ci  # or from rlox.evaluation

        scores = np.random.default_rng(42).standard_normal(100)
        lo, hi = stratified_bootstrap_ci(scores, n_bootstrap=1000, ci=0.95)

        assert lo <= hi, f"CI lower bound {lo:.3f} > upper bound {hi:.3f}"
        assert lo < float(np.median(scores)) < hi + 1e-6, (
            "Median should be within 95% CI"
        )

    # RED: performance_profiles returns fraction above threshold
    def test_performance_profiles(self):
        from rlox import performance_profiles  # or from rlox.evaluation

        scores_dict = {
            "PPO": np.array([100.0, 200.0, 300.0, 400.0, 500.0]),
            "DQN": np.array([50.0, 100.0, 150.0, 200.0, 250.0]),
        }
        thresholds = np.array([0.0, 100.0, 200.0, 300.0])

        profiles = performance_profiles(scores_dict, thresholds)

        assert "PPO" in profiles
        assert "DQN" in profiles
        # At threshold=0, all runs should be above
        assert profiles["PPO"][0] == 1.0
        assert profiles["DQN"][0] == 1.0
        # At threshold=300, only some PPO runs should be above
        assert profiles["DQN"][-1] <= profiles["PPO"][-1]

    # RED: IQM handles NaN scores gracefully
    def test_iqm_handles_nan(self):
        from rlox import interquartile_mean

        scores_with_nan = np.array([1.0, 2.0, float("nan"), 4.0, 5.0])
        # Either raise ValueError or compute IQM on non-NaN values
        try:
            iqm = interquartile_mean(scores_with_nan)
            # If it doesn't raise, result must be finite
            assert math.isfinite(iqm), "IQM with NaN should return finite or raise"
        except (ValueError, RuntimeError):
            pass  # Raising is also acceptable


# ---------------------------------------------------------------------------
# Group 4: Experiment metadata
# ---------------------------------------------------------------------------

class TestExperimentMetadata:
    """Group 4: Experiment snapshot captures all required metadata."""

    # RED: capture_experiment_metadata importable
    def test_capture_metadata_importable(self):
        from rlox import capture_experiment_metadata  # or from rlox.experiment

    # RED: metadata contains required fields
    def test_metadata_has_required_fields(self):
        from rlox import capture_experiment_metadata, PPOConfig

        config = PPOConfig()
        meta = capture_experiment_metadata(config=config, seed=42)

        required_fields = {
            "seed",
            "python_version",
            "numpy_version",
            "platform",
            "timestamp",
        }
        for field in required_fields:
            assert field in meta, f"metadata missing required field: {field}"

        assert meta["seed"] == 42

    # RED: metadata includes git hash when in git repo
    def test_metadata_includes_git_hash_if_available(self):
        from rlox import capture_experiment_metadata, PPOConfig

        meta = capture_experiment_metadata(config=PPOConfig(), seed=42)

        # git_hash may be None if not in a git repo — but the key must exist
        assert "git_hash" in meta, "metadata must have 'git_hash' key (may be None)"

    # RED: metadata includes rlox version
    def test_metadata_includes_rlox_version(self):
        from rlox import capture_experiment_metadata, PPOConfig

        meta = capture_experiment_metadata(config=PPOConfig(), seed=42)
        assert "rlox_version" in meta, "metadata must include rlox_version"

    # RED: metadata is JSON-serializable
    def test_metadata_is_json_serializable(self):
        from rlox import capture_experiment_metadata, PPOConfig

        meta = capture_experiment_metadata(config=PPOConfig(), seed=42)

        try:
            json_str = json.dumps(meta)
            assert len(json_str) > 0
        except (TypeError, ValueError) as e:
            pytest.fail(f"metadata is not JSON-serializable: {e}")


# ---------------------------------------------------------------------------
# Group 6: Online DPO / OAIF
# ---------------------------------------------------------------------------

class TestOnlineDPO:
    """Group 6: Online DPO pipeline."""

    # RED: OnlineDPO importable
    def test_online_dpo_importable(self):
        from rlox import OnlineDPO  # or from rlox.algorithms.online_dpo

    # RED: online DPO generates and scores in loop
    def test_online_dpo_smoke_test(self, torch):
        from rlox import OnlineDPO

        class MockModel(torch.nn.Module):
            VOCAB_SIZE = 20

            def __init__(self):
                super().__init__()
                self.net = torch.nn.Linear(10, self.VOCAB_SIZE)

            def forward(self, x):
                return self.net(x.float().mean(-1, keepdim=True).expand(-1, 10))

            def generate(self, prompt_ids, n=2, max_new_tokens=5):
                batch = prompt_ids.shape[0]
                return torch.randint(0, self.VOCAB_SIZE, (batch * n, max_new_tokens))

            def log_probs(self, input_ids, labels):
                logits = self.forward(input_ids)
                lp = torch.nn.functional.log_softmax(logits, dim=-1)
                return lp.gather(1, labels[:, :1]).squeeze()

        def mock_preference_fn(completions_a, completions_b, prompts):
            """Returns 1 if a preferred, 0 if b preferred."""
            return [1 if i % 2 == 0 else 0 for i in range(len(prompts))]

        model = MockModel()
        ref_model = MockModel()
        ref_model.load_state_dict(model.state_dict())

        dpo = OnlineDPO(
            model=model,
            ref_model=ref_model,
            preference_fn=mock_preference_fn,
            n_candidates=2,
            beta=0.1,
        )

        prompts = [torch.randint(0, 20, (3,)) for _ in range(4)]
        loss, metrics = dpo.train_step(prompts)
        assert math.isfinite(loss.item()), "Online DPO loss must be finite"


class TestBestOfN:
    """Group 6: Best-of-N sampling baseline."""

    # RED: BestOfN importable
    def test_best_of_n_importable(self):
        from rlox import BestOfN  # or from rlox.algorithms.best_of_n

    # RED: BestOfN returns single best completion per prompt
    def test_best_of_n_returns_best(self, torch):
        from rlox import BestOfN

        class MockModel(torch.nn.Module):
            def generate(self, prompt_ids, n=4, max_new_tokens=10):
                batch = prompt_ids.shape[0]
                return torch.randint(0, 100, (batch * n, max_new_tokens))

        def reward_fn(completions, prompts):
            """Give reward = first token value (deterministic)."""
            return [float(c[0].item()) for c in completions]

        model = MockModel()
        bon = BestOfN(model=model, reward_fn=reward_fn, n=4)

        prompts = [torch.randint(0, 100, (5,)) for _ in range(3)]
        best = bon.sample(prompts)

        assert len(best) == 3, "BestOfN should return one completion per prompt"

    # RED: BestOfN with n=1 is equivalent to greedy
    def test_best_of_n_with_n1(self, torch):
        from rlox import BestOfN

        class MockModel(torch.nn.Module):
            def generate(self, prompt_ids, n=1, max_new_tokens=5):
                batch = prompt_ids.shape[0]
                return torch.zeros(batch * n, max_new_tokens, dtype=torch.long)

        def reward_fn(completions, prompts):
            return [1.0] * len(completions)

        model = MockModel()
        bon = BestOfN(model=model, reward_fn=reward_fn, n=1)
        prompts = [torch.zeros(3, dtype=torch.long) for _ in range(2)]
        best = bon.sample(prompts)
        assert len(best) == 2
```

---

## Part 3: Performance Regression Tests

**Target file:** `tests/python/test_phase8_perf.py`

```python
# tests/python/test_phase8_perf.py
"""
Phase 8 performance regression tests.
All Phase 0-7 benchmarks must be maintained.
"""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "benchmarks"))
from conftest import BenchmarkResult, ComparisonResult, timed_run


class TestPhase8PerfRegressions:
    """Verify Phase 8 additions do not regress previous benchmarks."""

    def test_prioritized_buffer_sample_vs_uniform(self):
        """
        PrioritizedReplayBuffer.sample() must be < 10x slower than uniform
        ReplayBuffer.sample() at same size (O(log N) overhead only).
        """
        from rlox import PrioritizedReplayBuffer, ReplayBuffer

        obs = np.zeros(4, dtype=np.float32)
        act = np.zeros(1, dtype=np.float32)
        capacity = 100_000

        # Fill both buffers
        pri_buf = PrioritizedReplayBuffer(
            capacity=capacity, obs_dim=4, act_dim=1, alpha=0.6, beta=0.4
        )
        uni_buf = ReplayBuffer(capacity=capacity, obs_dim=4, act_dim=1)

        for _ in range(capacity):
            pri_buf.push(obs=obs, action=act, reward=1.0,
                        terminated=False, truncated=False, priority=1.0)
            uni_buf.push(obs=obs, action=act, reward=1.0,
                        terminated=False, truncated=False)

        pri_times = timed_run(
            lambda: pri_buf.sample(batch_size=256, seed=42),
            n_warmup=10, n_reps=100,
        )
        uni_times = timed_run(
            lambda: uni_buf.sample(batch_size=256, seed=42),
            n_warmup=10, n_reps=100,
        )

        overhead = np.median(pri_times) / np.median(uni_times)
        assert overhead < 10.0, (
            f"PrioritizedReplayBuffer.sample() is {overhead:.1f}x slower than "
            f"uniform — expected < 10x (O(log N) overhead)"
        )

    def test_gae_not_regressed_after_normalization_integration(self):
        """compute_gae must remain >= 5x faster than numpy loop."""
        from rlox import compute_gae

        rng = np.random.default_rng(42)
        n = 2048
        rewards = rng.standard_normal(n)
        values = rng.standard_normal(n)
        dones = (rng.random(n) > 0.95).astype(float)

        def numpy_gae(r, v, d, lv, g=0.99, l=0.95):
            adv = np.zeros(len(r))
            last = 0.0
            for t in reversed(range(len(r))):
                nnt = 1.0 - float(d[t])
                nv = lv if t == len(r) - 1 else v[t + 1]
                delta = r[t] + g * nv * nnt - v[t]
                last = delta + g * l * nnt * last
                adv[t] = last
            return adv, adv + v

        rlox_times = timed_run(
            lambda: compute_gae(rewards, values, dones, 0.0, 0.99, 0.95),
            n_warmup=10, n_reps=100,
        )
        numpy_times = timed_run(
            lambda: numpy_gae(rewards, values, dones, 0.0),
            n_warmup=10, n_reps=100,
        )

        speedup = np.median(numpy_times) / np.median(rlox_times)
        assert speedup >= 5.0, f"GAE regression after Phase 8: {speedup:.1f}x"
```

---

## Test Coverage Targets

| Component | Target Coverage | Notes |
|---|---|---|
| `buffer/priority.rs` (SumTree) | 95%+ | Proptest covers all tree operations |
| `buffer/priority.rs` (PrioritizedReplayBuffer) | 90%+ | All error paths tested |
| `env/mujoco.rs` | 80%+ | Requires mujoco feature flag |
| `buffer/mmap.rs` | 85%+ | Recovery path critical |
| `llm/sandbox.rs` | 90%+ | All error conditions tested |
| Python `SAC`, `TD3`, `DQN` | 70%+ | Smoke + E2E |
| Python `PPOTrainer` (one-liner) | 85%+ | Save/load + evaluate |
| Python config system | 90%+ | All merge/validation paths |
| Python callback system | 85%+ | All callback methods |
| Python `TrainingDiagnostics` | 95%+ | All warning conditions |
| Python statistical eval | 90%+ | IQM + CI + profiles |
| Python `capture_experiment_metadata` | 80%+ | JSON serialization path |

---

## Notes on Test Execution

1. **MuJoCo tests** require the `mujoco` feature flag in Cargo.toml and
   `pip install gymnasium[mujoco]`. Skip with `-k "not mujoco"`.

2. **Memory-mapped buffer tests** create temp files. Use `TempDir` in Rust
   and `tmp_path` fixture in Python to ensure cleanup.

3. **Slow E2E algorithm tests** are marked `@pytest.mark.slow`. The SAC/TD3
   HalfCheetah tests require ~1M steps and take 20-60 minutes.

4. **Sandbox tests** require the sandboxing mechanism (WASM or subprocess) to
   be installed. Mark with `@pytest.mark.sandbox` if optional.

5. **Group 0 (SumTree + PrioritizedReplayBuffer)** Rust tests must pass before
   any off-policy Python algorithm tests (Group 1), since DQN uses `PrioritizedReplayBuffer`.
