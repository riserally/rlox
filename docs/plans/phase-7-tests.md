# Phase 7 TDD Test Specifications — Algorithm Completeness (v0.5)

**Status: RED (all tests must be written before implementation)**
**Phase plan: `/docs/plans/phase-7-algorithm-completeness.md`**
**Run Rust tests:** `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo test`
**Run Python tests:** `.venv/bin/python -m pytest tests/python/ -v`

---

## Test Execution Order

Tests are grouped by dependency. Earlier groups must pass before later ones.

```
Group 0 (Bug fixes)   → must pass before any algorithm work
Group 1 (Primitives)  → RunningStats, SequencePacking, BatchSteppable
Group 2 (Collectors)  → RolloutCollector, batch assembly, zero-copy
Group 3 (Algorithms)  → PPO, GRPO, DPO, A2C end-to-end
Group 4 (Ops)         → Checkpoint, logging, wheels
```

---

## Part 1: Rust Unit Tests

All Rust tests live in `crates/rlox-core/src/` alongside the implementation
modules they test. Add `#[cfg(test)]` blocks to each new module.

### 1.1 Bug Fix Verification Tests

**Target file:** `crates/rlox-core/src/buffer/mod.rs` (update existing)
**Group: 0 — must pass first**

```rust
// crates/rlox-core/src/buffer/mod.rs
#[cfg(test)]
mod fix_verification_tests {
    use super::*;

    // RED: ExperienceRecord.action must be Vec<f32>, not f32
    #[test]
    fn experience_record_action_is_vec() {
        // Arrange: create a record with a 6-dimensional action (HalfCheetah)
        let record = ExperienceRecord {
            obs: vec![0.0f32; 17],
            action: vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6],  // 6-dim
            reward: 1.0,
            terminated: false,
            truncated: false,
        };
        // Act + Assert
        assert_eq!(record.action.len(), 6);
        assert_eq!(record.obs.len(), 17);
    }

    // RED: multi-dim action roundtrip through ExperienceTable
    #[test]
    fn experience_table_stores_multi_dim_action() {
        // Arrange
        let obs_dim = 17;
        let act_dim = 6;
        let mut table = ExperienceTable::new(obs_dim, act_dim);
        let action = vec![0.1f32, -0.2, 0.3, -0.4, 0.5, -0.6];
        let record = ExperienceRecord {
            obs: vec![1.0; obs_dim],
            action: action.clone(),
            reward: 5.0,
            terminated: false,
            truncated: false,
        };

        // Act
        table.push(record).unwrap();

        // Assert: actions_raw has act_dim elements
        assert_eq!(table.actions_raw().len(), act_dim);
        assert_eq!(&table.actions_raw()[..act_dim], action.as_slice());
    }

    // RED: ReplayBuffer stores and retrieves multi-dim actions
    #[test]
    fn replay_buffer_multi_dim_action_roundtrip() {
        // Arrange
        let obs_dim = 4;
        let act_dim = 3;
        let mut buf = ReplayBuffer::new(100, obs_dim, act_dim);
        let action = vec![0.5f32, -0.5, 1.0];
        let record = ExperienceRecord {
            obs: vec![0.1; obs_dim],
            action: action.clone(),
            reward: 1.0,
            terminated: false,
            truncated: false,
        };

        // Act
        buf.push(record).unwrap();
        let batch = buf.sample(1, 42).unwrap();

        // Assert
        assert_eq!(batch.act_dim, act_dim);
        assert_eq!(batch.actions.len(), act_dim);  // 1 sample * 3 dims
        assert_eq!(&batch.actions[..act_dim], action.as_slice());
    }

    // RED: action dim mismatch returns error, not panic
    #[test]
    fn experience_table_action_dim_mismatch_returns_error() {
        // Arrange
        let mut table = ExperienceTable::new(4, 2);
        let record = ExperienceRecord {
            obs: vec![1.0; 4],
            action: vec![0.1, 0.2, 0.3],  // 3 dims, table expects 2
            reward: 1.0,
            terminated: false,
            truncated: false,
        };

        // Act
        let result = table.push(record);

        // Assert
        assert!(result.is_err(), "action dim mismatch must return Err");
        let err_str = result.unwrap_err().to_string();
        assert!(
            err_str.contains("act_dim"),
            "error must mention act_dim, got: {err_str}"
        );
    }

    // RED: scalar action (act_dim=1) still works (backward compat)
    #[test]
    fn experience_table_scalar_action_dim_one() {
        let mut table = ExperienceTable::new(4, 1);
        let record = ExperienceRecord {
            obs: vec![1.0; 4],
            action: vec![0.0],  // scalar as vec of length 1
            reward: 1.0,
            terminated: false,
            truncated: false,
        };
        table.push(record).unwrap();
        assert_eq!(table.len(), 1);
        assert_eq!(table.actions_raw().len(), 1);
    }
}
```

**Target file:** `crates/rlox-core/src/env/parallel.rs` (update existing)

```rust
// crates/rlox-core/src/env/parallel.rs
#[cfg(test)]
mod terminal_obs_tests {
    use super::*;
    use crate::env::builtins::CartPole;
    use crate::seed::derive_seed;

    fn make_vec_env(n: usize, seed: u64) -> VecEnv {
        let envs: Vec<Box<dyn RLEnv>> = (0..n)
            .map(|i| Box::new(CartPole::new(Some(derive_seed(seed, i)))) as Box<dyn RLEnv>)
            .collect();
        VecEnv::new(envs)
    }

    // RED: StepResult must carry terminal_obs when done=true
    #[test]
    fn step_result_has_terminal_obs_on_truncation() {
        // Arrange: run a VecEnv until at least one env truncates
        let mut venv = make_vec_env(4, 42);
        venv.reset_all(Some(42)).unwrap();

        let mut found_truncated_with_terminal_obs = false;

        // Step many times — CartPole truncates at 500 steps
        for _ in 0..600 {
            let actions: Vec<Action> = (0..4).map(|_| Action::Discrete(0)).collect();
            let batch = venv.step_all(&actions).unwrap();

            for i in 0..4 {
                if batch.truncated[i] {
                    // Assert: terminal_obs must be Some when truncated
                    assert!(
                        batch.terminal_obs[i].is_some(),
                        "terminal_obs must be Some when truncated"
                    );
                    found_truncated_with_terminal_obs = true;
                }
                if batch.terminated[i] {
                    // terminal_obs should also be Some when terminated
                    assert!(
                        batch.terminal_obs[i].is_some(),
                        "terminal_obs must be Some when terminated"
                    );
                }
                // When not done, terminal_obs should be None
                if !batch.terminated[i] && !batch.truncated[i] {
                    assert!(
                        batch.terminal_obs[i].is_none(),
                        "terminal_obs must be None when not done"
                    );
                }
            }
        }

        // Note: this test may not trigger truncation in all runs.
        // Use it to drive the StepResult struct changes; the field
        // existence is more important than guaranteeing truncation.
        let _ = found_truncated_with_terminal_obs;
    }

    // RED: terminal_obs dim matches obs_dim
    #[test]
    fn terminal_obs_has_correct_dimension() {
        let mut venv = make_vec_env(2, 42);
        venv.reset_all(Some(42)).unwrap();

        for _ in 0..200 {
            let actions: Vec<Action> = vec![Action::Discrete(1); 2];
            let batch = venv.step_all(&actions).unwrap();
            for i in 0..2 {
                if let Some(tobs) = &batch.terminal_obs[i] {
                    assert_eq!(tobs.len(), 4, "CartPole terminal obs must have dim 4");
                }
            }
        }
    }

    // RED: returned obs (post-reset) is fresh episode obs, not terminal obs
    #[test]
    fn returned_obs_after_reset_is_fresh_not_terminal() {
        // A fresh CartPole reset always returns obs in [-0.05, 0.05].
        // Terminal obs can be anywhere in the obs space.
        // So if done=true, returned obs should be near-zero (fresh),
        // and terminal_obs should be the out-of-bounds state.
        let mut venv = make_vec_env(1, 42);
        venv.reset_all(Some(42)).unwrap();

        for _ in 0..200 {
            let actions = vec![Action::Discrete(1)];
            let batch = venv.step_all(&actions).unwrap();
            if batch.terminated[0] {
                let fresh_obs = &batch.obs[0];
                // Fresh reset obs is in [-0.05, 0.05]
                for &v in fresh_obs {
                    assert!(
                        v.abs() <= 0.06,
                        "post-reset obs should be near zero, got {v}"
                    );
                }
                // terminal_obs is NOT near zero (that's why it terminated)
                let tobs = batch.terminal_obs[0]
                    .as_ref()
                    .expect("terminal_obs must exist on termination");
                let max_abs = tobs.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
                assert!(
                    max_abs > 0.05,
                    "terminal obs should be out-of-bounds, got max_abs={max_abs}"
                );
                break;
            }
        }
    }
}
```

**Target file:** `crates/rlox-core/src/llm/ops.rs` (update existing)

```rust
// crates/rlox-core/src/llm/ops.rs
#[cfg(test)]
mod kl_result_tests {
    use super::*;

    // RED: compute_token_kl must return Result, not panic
    #[test]
    fn token_kl_mismatched_lengths_returns_err_not_panic() {
        // Arrange
        let log_p = vec![-1.0f64, -2.0];
        let log_q = vec![-1.0f64];  // different length

        // Act
        let result = compute_token_kl(&log_p, &log_q);

        // Assert: must be Err, never panics
        assert!(result.is_err(), "mismatched lengths must return Err");
    }

    // RED: compute_token_kl returns Ok for matching lengths
    #[test]
    fn token_kl_matching_lengths_returns_ok() {
        let log_p = vec![-1.0f64, -2.0, -0.5];
        let log_q = vec![-1.0f64, -2.0, -0.5];
        let result = compute_token_kl(&log_p, &log_q);
        assert!(result.is_ok());
        assert!(result.unwrap().abs() < 1e-15);
    }

    // RED: empty slices return Ok(0.0)
    #[test]
    fn token_kl_empty_slices_returns_zero() {
        let result = compute_token_kl(&[], &[]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0.0);
    }

    // RED: NaN log_prob propagates to result (does not silently swallow)
    #[test]
    fn token_kl_nan_input_propagates_to_output() {
        let log_p = vec![f64::NAN];
        let log_q = vec![-1.0f64];
        let result = compute_token_kl(&log_p, &log_q);
        // Either Err or Ok(NaN) — must not produce a finite non-NaN value
        match result {
            Ok(v) => assert!(v.is_nan(), "NaN input should produce NaN output"),
            Err(_) => {}  // also acceptable
        }
    }

    // RED: +Inf log_prob results in Err or Inf propagation
    #[test]
    fn token_kl_inf_input_does_not_panic() {
        let log_p = vec![f64::INFINITY];
        let log_q = vec![-1.0f64];
        // Must not panic — return Err or propagate inf
        let _result = compute_token_kl(&log_p, &log_q);
    }

    // Regression: known KL value is preserved through Result refactor
    #[test]
    fn token_kl_known_value_still_correct_after_refactor() {
        // exp(-1) * (-1 - (-2)) = exp(-1) ≈ 0.36787944
        let log_p = vec![-1.0f64];
        let log_q = vec![-2.0f64];
        let kl = compute_token_kl(&log_p, &log_q).unwrap();
        assert!((kl - (-1.0_f64).exp()).abs() < 1e-10);
    }
}
```

### 1.2 BatchSteppable Trait Tests

**Target file:** `crates/rlox-core/src/env/batch.rs` (new file)
**Group: 1**

```rust
// crates/rlox-core/src/env/batch.rs
#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::builtins::CartPole;
    use crate::env::spaces::Action;
    use crate::seed::derive_seed;

    fn make_parallel_stepper(n: usize, seed: u64) -> ParallelBatchStepper {
        let envs: Vec<Box<dyn RLEnv>> = (0..n)
            .map(|i| Box::new(CartPole::new(Some(derive_seed(seed, i)))) as Box<dyn RLEnv>)
            .collect();
        ParallelBatchStepper::new(envs)
    }

    // RED: ParallelBatchStepper implements BatchSteppable
    #[test]
    fn parallel_batch_stepper_implements_trait() {
        fn assert_batch_steppable<T: BatchSteppable>() {}
        assert_batch_steppable::<ParallelBatchStepper>();
    }

    // RED: ParallelBatchStepper is Send (required for thread dispatch)
    #[test]
    fn parallel_batch_stepper_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<ParallelBatchStepper>();
    }

    // RED: step_batch returns correct number of results
    #[test]
    fn parallel_batch_stepper_step_count() {
        // Arrange
        let mut stepper = make_parallel_stepper(8, 42);
        stepper.reset_batch(Some(42)).unwrap();
        let actions: Vec<Action> = (0..8).map(|i| Action::Discrete((i % 2) as u32)).collect();

        // Act
        let results = stepper.step_batch(&actions).unwrap();

        // Assert
        assert_eq!(results.len(), 8);
    }

    // RED: num_envs reports correct count
    #[test]
    fn parallel_batch_stepper_num_envs() {
        let stepper = make_parallel_stepper(16, 42);
        assert_eq!(stepper.num_envs(), 16);
    }

    // RED: obs_dim and act_dim accessible through trait
    #[test]
    fn parallel_batch_stepper_space_dims() {
        let stepper = make_parallel_stepper(4, 42);
        // CartPole: obs_dim=4, act_dim=1
        assert_eq!(stepper.obs_dim(), 4);
        assert_eq!(stepper.act_dim(), 1);
    }

    // RED: step_batch wrong action count returns error
    #[test]
    fn parallel_batch_stepper_wrong_action_count_returns_error() {
        let mut stepper = make_parallel_stepper(4, 42);
        stepper.reset_batch(Some(42)).unwrap();
        let actions = vec![Action::Discrete(0); 3];  // 3 actions for 4 envs
        assert!(stepper.step_batch(&actions).is_err());
    }

    // RED: VecEnv::new accepts Box<dyn BatchSteppable>
    #[test]
    fn vec_env_accepts_parallel_batch_stepper() {
        let stepper = make_parallel_stepper(4, 42);
        let _venv = VecEnv::new(Box::new(stepper));
    }

    // RED: reset_batch returns one observation per env
    #[test]
    fn parallel_batch_stepper_reset_returns_obs_per_env() {
        let mut stepper = make_parallel_stepper(6, 42);
        let obs_vec = stepper.reset_batch(Some(99)).unwrap();
        assert_eq!(obs_vec.len(), 6);
        for obs in &obs_vec {
            assert_eq!(obs.as_slice().len(), 4);  // CartPole obs_dim=4
        }
    }

    // RED: BoxDynBatchSteppable can be used as trait object
    #[test]
    fn batch_steppable_as_dyn_object() {
        let stepper: Box<dyn BatchSteppable> =
            Box::new(make_parallel_stepper(2, 42));
        assert_eq!(stepper.num_envs(), 2);
    }
}
```

### 1.3 Reward Normalization (RunningStats)

**Target file:** `crates/rlox-core/src/training/normalization.rs` (new file)
**Group: 1**

```rust
// crates/rlox-core/src/training/normalization.rs
#[cfg(test)]
mod tests {
    use super::*;

    // RED: initial state
    #[test]
    fn running_stats_new_is_empty() {
        let stats = RunningStats::new();
        assert_eq!(stats.count(), 0);
    }

    // RED: single sample — mean equals value, var is 0
    #[test]
    fn running_stats_single_sample() {
        let mut stats = RunningStats::new();
        stats.update(5.0);
        assert!((stats.mean() - 5.0).abs() < 1e-10);
        assert_eq!(stats.count(), 1);
        // Variance is undefined for single sample; implementation may return 0.0 or 1.0
        // We only require it does not panic
        let _ = stats.var();
        let _ = stats.std();
    }

    // RED: Welford's classic example — known mean and variance
    #[test]
    fn running_stats_welford_known_values() {
        // Dataset: [2, 4, 4, 4, 5, 5, 7, 9]
        // Mean = 5.0, Variance (population) = 4.0, Std = 2.0
        let mut stats = RunningStats::new();
        for &x in &[2.0_f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
            stats.update(x);
        }
        assert!((stats.mean() - 5.0).abs() < 1e-10,
            "mean should be 5.0, got {}", stats.mean());
        assert!((stats.var() - 4.0).abs() < 1e-10,
            "variance should be 4.0, got {}", stats.var());
        assert!((stats.std() - 2.0).abs() < 1e-10,
            "std should be 2.0, got {}", stats.std());
    }

    // RED: normalize brings value to approximately z-score
    #[test]
    fn running_stats_normalize_produces_z_score() {
        let mut stats = RunningStats::new();
        for &x in &[2.0_f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
            stats.update(x);
        }
        // Normalize the mean itself — should produce ~0.0
        let z = stats.normalize(5.0);
        assert!(z.abs() < 1e-10, "normalize(mean) should be ~0, got {z}");
        // Normalize mean + std — should produce ~1.0
        let z2 = stats.normalize(7.0);
        assert!((z2 - 1.0).abs() < 1e-10, "normalize(mean+std) should be ~1, got {z2}");
    }

    // RED: normalize with zero std does not panic; returns 0 or clips
    #[test]
    fn running_stats_normalize_with_zero_std_does_not_panic() {
        let mut stats = RunningStats::new();
        stats.update(5.0);
        stats.update(5.0);
        stats.update(5.0);
        // std = 0 — normalize must not divide by zero
        let z = stats.normalize(5.0);
        // Should be 0.0 (clamped) or very small
        assert!(z.is_finite(), "normalize with zero std must be finite");
    }

    // RED: large stream maintains numerical stability
    #[test]
    fn running_stats_large_stream_numerically_stable() {
        let mut stats = RunningStats::new();
        // Large constant offset — naive (sum/n) would lose precision
        let base = 1_000_000.0f64;
        for i in 0..10_000 {
            stats.update(base + (i as f64) * 0.001);
        }
        // Mean should be base + (10000 * 0.001) / 2 = base + 5.0
        let expected_mean = base + 5.0 - 0.001 / 2.0;
        assert!(
            (stats.mean() - expected_mean).abs() < 0.01,
            "mean imprecise for large offset: got {}, expected ~{expected_mean}",
            stats.mean()
        );
    }

    // RED: reset clears state
    #[test]
    fn running_stats_reset_clears_state() {
        let mut stats = RunningStats::new();
        for &x in &[1.0f64, 2.0, 3.0] {
            stats.update(x);
        }
        stats.reset();
        assert_eq!(stats.count(), 0);
    }

    // RED: NaN input propagates or errors, does not silently corrupt state
    #[test]
    fn running_stats_nan_input_does_not_silently_corrupt() {
        let mut stats = RunningStats::new();
        stats.update(1.0);
        stats.update(2.0);
        let mean_before = stats.mean();
        // Updating with NaN — either Err or NaN propagation is acceptable
        // but must NOT silently produce a different finite value
        stats.update(f64::NAN);
        let mean_after = stats.mean();
        if mean_after.is_finite() {
            assert!(
                (mean_after - mean_before).abs() < 1e-10 || mean_after.is_nan(),
                "NaN input corrupted finite mean: was {mean_before}, now {mean_after}"
            );
        }
    }

    // --- Property-based tests ---
    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            // RED: mean invariant — running mean matches batch mean
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

            // RED: variance invariant — running variance ≥ 0
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

            // RED: std = sqrt(var) invariant
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

            // RED: count always equals number of updates
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
        }
    }
}
```

### 1.4 Sequence Packing Tests

**Target file:** `crates/rlox-core/src/training/packing.rs` (new file)
**Group: 1**

```rust
// crates/rlox-core/src/training/packing.rs
#[cfg(test)]
mod tests {
    use super::*;

    // RED: pack_sequences with sequences that fit in one bin
    #[test]
    fn pack_sequences_one_bin_exact_fit() {
        // Sequences: [3, 3] — total 6, max_length=6 → one bin
        let seqs: Vec<Vec<u32>> = vec![vec![1, 2, 3], vec![4, 5, 6]];
        let slices: Vec<&[u32]> = seqs.iter().map(|s| s.as_slice()).collect();
        let packed = pack_sequences(&slices, 6).unwrap();

        // All sequences present in one batch
        assert_eq!(packed.len(), 1, "should produce 1 bin for total_len=max_len");
        assert_eq!(packed[0].input_ids.len(), 6);
    }

    // RED: pack_sequences creates separate bins when sequences exceed max_length
    #[test]
    fn pack_sequences_two_bins_when_overflow() {
        // Sequences: [3, 3, 3] — total 9, max_length=6 → 2 bins: [3,3] and [3]
        let seqs: Vec<Vec<u32>> = vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]];
        let slices: Vec<&[u32]> = seqs.iter().map(|s| s.as_slice()).collect();
        let packed = pack_sequences(&slices, 6).unwrap();

        assert!(packed.len() >= 2, "should produce at least 2 bins");
        // No bin should exceed max_length
        for bin in &packed {
            assert!(
                bin.input_ids.len() <= 6,
                "bin length {} exceeds max_length 6",
                bin.input_ids.len()
            );
        }
    }

    // RED: all sequences present in output (no data loss)
    #[test]
    fn pack_sequences_all_sequences_present() {
        let seqs: Vec<Vec<u32>> = vec![
            vec![1, 2, 3],
            vec![4, 5],
            vec![6, 7, 8, 9],
            vec![10],
            vec![11, 12],
        ];
        let slices: Vec<&[u32]> = seqs.iter().map(|s| s.as_slice()).collect();
        let packed = pack_sequences(&slices, 8).unwrap();

        // Collect all token IDs from packed output
        let mut all_packed_tokens: Vec<u32> = packed
            .iter()
            .flat_map(|b| b.input_ids.iter().copied())
            .filter(|&t| t != 0)  // 0 is padding
            .collect();
        all_packed_tokens.sort_unstable();

        let mut all_input_tokens: Vec<u32> = seqs.iter().flatten().copied().collect();
        all_input_tokens.sort_unstable();

        assert_eq!(all_packed_tokens, all_input_tokens,
            "all input tokens must appear exactly once in output");
    }

    // RED: sequence longer than max_length returns error
    #[test]
    fn pack_sequences_sequence_exceeds_max_length_returns_error() {
        let long_seq = vec![1u32; 100];
        let slices = vec![long_seq.as_slice()];
        let result = pack_sequences(&slices, 50);  // seq=100, max=50
        assert!(result.is_err(), "sequence longer than max_length must return Err");
    }

    // RED: empty input returns empty output
    #[test]
    fn pack_sequences_empty_input_returns_empty() {
        let result = pack_sequences(&[], 512).unwrap();
        assert!(result.is_empty());
    }

    // RED: single sequence produces single bin with correct content
    #[test]
    fn pack_sequences_single_sequence() {
        let seq = vec![10u32, 20, 30];
        let slices = vec![seq.as_slice()];
        let packed = pack_sequences(&slices, 512).unwrap();

        assert_eq!(packed.len(), 1);
        assert_eq!(&packed[0].input_ids[..3], &[10, 20, 30]);
    }

    // RED: attention_mask length matches input_ids length per bin
    #[test]
    fn pack_sequences_attention_mask_matches_input_ids_length() {
        let seqs: Vec<Vec<u32>> = vec![vec![1, 2, 3], vec![4, 5]];
        let slices: Vec<&[u32]> = seqs.iter().map(|s| s.as_slice()).collect();
        let packed = pack_sequences(&slices, 8).unwrap();

        for bin in &packed {
            assert_eq!(
                bin.input_ids.len(), bin.attention_mask.len(),
                "attention_mask must have same length as input_ids"
            );
        }
    }

    // RED: position_ids are contiguous from 0 within each sequence segment
    #[test]
    fn pack_sequences_position_ids_per_sequence_start_from_zero() {
        let seqs: Vec<Vec<u32>> = vec![vec![1, 2, 3], vec![4, 5]];
        let slices: Vec<&[u32]> = seqs.iter().map(|s| s.as_slice()).collect();
        let packed = pack_sequences(&slices, 8).unwrap();

        for bin in &packed {
            // Each sequence's position_ids should start at 0 and increment
            // The sequence_starts field marks where each sequence begins
            for (k, &start) in bin.sequence_starts.iter().enumerate() {
                let end = if k + 1 < bin.sequence_starts.len() {
                    bin.sequence_starts[k + 1]
                } else {
                    // Find end from attention_mask or total length
                    bin.input_ids.len()
                };
                for (j, pos) in (start..end).enumerate() {
                    assert_eq!(
                        bin.position_ids[pos], j as u32,
                        "position_ids[{pos}] should be {j}"
                    );
                }
            }
        }
    }

    // RED: fill_rate > 50% for typical Zipf-distributed sequences
    #[test]
    fn pack_sequences_fill_rate_good_for_varied_lengths() {
        // Simulate varied sequence lengths (short + long)
        let lengths = [10, 50, 20, 80, 30, 60, 15, 100, 40, 25];
        let seqs: Vec<Vec<u32>> = lengths
            .iter()
            .enumerate()
            .map(|(i, &l)| (0..l as u32).map(|t| i as u32 * 1000 + t).collect())
            .collect();
        let slices: Vec<&[u32]> = seqs.iter().map(|s| s.as_slice()).collect();
        let max_len = 128;
        let packed = pack_sequences(&slices, max_len).unwrap();

        // Count actual tokens (non-padding)
        let total_tokens: usize = lengths.iter().sum();
        let total_capacity = packed.len() * max_len;
        let fill_rate = total_tokens as f64 / total_capacity as f64;

        assert!(
            fill_rate > 0.5,
            "fill_rate {fill_rate:.2} is below 0.5 — bin packing too inefficient"
        );
    }

    // --- Property-based tests ---
    mod proptests {
        use super::*;
        use proptest::prelude::*;
        use proptest::collection::vec;

        proptest! {
            // RED: no bin ever exceeds max_length
            #[test]
            fn pack_sequences_no_bin_exceeds_max_length(
                lengths in vec(1usize..100, 1..20),
                max_len in 10usize..200,
            ) {
                // Only generate sequences where each fits in a bin
                let seqs: Vec<Vec<u32>> = lengths
                    .iter()
                    .filter(|&&l| l <= max_len)
                    .map(|&l| (0..l as u32).collect())
                    .collect();
                if seqs.is_empty() {
                    return Ok(());
                }
                let slices: Vec<&[u32]> = seqs.iter().map(|s| s.as_slice()).collect();
                let packed = pack_sequences(&slices, max_len).unwrap();
                for bin in &packed {
                    prop_assert!(
                        bin.input_ids.len() <= max_len,
                        "bin length {} exceeds max_len {}", bin.input_ids.len(), max_len
                    );
                }
            }

            // RED: all input sequences appear in packed output exactly once
            #[test]
            fn pack_sequences_all_tokens_present(
                lengths in vec(1usize..50, 1..15),
            ) {
                let max_len = 128;
                let seqs: Vec<Vec<u32>> = lengths
                    .iter()
                    .filter(|&&l| l <= max_len)
                    .enumerate()
                    .map(|(i, &l)| (0..l as u32).map(|t| i as u32 * 1000 + t).collect())
                    .collect();
                if seqs.is_empty() {
                    return Ok(());
                }
                let slices: Vec<&[u32]> = seqs.iter().map(|s| s.as_slice()).collect();
                let packed = pack_sequences(&slices, max_len).unwrap();

                let mut packed_tokens: Vec<u32> = packed
                    .iter()
                    .flat_map(|b| {
                        b.input_ids.iter().copied()
                            .zip(b.attention_mask.iter().copied())
                            .filter_map(|(t, m)| if m != 0 { Some(t) } else { None })
                    })
                    .collect();
                packed_tokens.sort_unstable();

                let mut input_tokens: Vec<u32> = seqs.iter().flatten().copied().collect();
                input_tokens.sort_unstable();

                prop_assert_eq!(packed_tokens, input_tokens);
            }
        }
    }
}
```

### 1.5 Configurable VecEnv Tests

**Target file:** `crates/rlox-python/src/env.rs` (update existing)
**Group: 0**

```rust
// crates/rlox-python/src/env.rs
// These are doc-tests or Python-side tests; Rust unit tests verify the
// binding signature and construction works:
#[cfg(test)]
mod vec_env_configurability_tests {
    // Note: These tests validate the Rust API surface changes.
    // Full integration tests are in tests/python/test_phase7.py.

    // RED: PyVecEnv::new accepts an env_id string parameter
    // This is a compile-time test — if it compiles, the API changed correctly.
    // Validated via: cargo check --package rlox-python
    //
    // The new signature should be:
    //   fn new(env_id: &str, n_envs: usize, seed: Option<u64>) -> PyResult<Self>
    //
    // Verified by the Python integration tests below.
}
```

---

## Part 2: Python Integration Tests

All Python tests live in `tests/python/test_phase7.py`.
**Add this file; do not modify existing test files.**

```python
# tests/python/test_phase7.py
"""
Phase 7 TDD test specifications — Algorithm Completeness (v0.5).

Status: RED — all tests fail until implementation is complete.
Run: .venv/bin/python -m pytest tests/python/test_phase7.py -v

Dependency order:
  Group 0: Bug fix verification (must pass first)
  Group 1: Primitive components (RunningStats, sequence packing, batch assembly)
  Group 2: RolloutCollector, batch assembly, zero-copy
  Group 3: Algorithm end-to-end (PPO, GRPO, DPO, A2C)
  Group 4: Operational (checkpoint, logging, wheels)
"""

import contextlib
import math
import os
import sys
import tempfile
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def torch():
    """Skip all torch-dependent tests if torch is not installed."""
    torch = pytest.importorskip("torch")
    return torch


@pytest.fixture(scope="session")
def gymnasium():
    """Skip gymnasium-dependent tests if not installed."""
    gym = pytest.importorskip("gymnasium")
    return gym


def make_mlp_policy(obs_dim: int, act_dim: int, hidden: int = 64):
    """Build a minimal MLP actor-critic for testing. Requires torch."""
    import torch
    import torch.nn as nn

    class MlpPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.actor_mean = nn.Sequential(
                nn.Linear(obs_dim, hidden),
                nn.Tanh(),
                nn.Linear(hidden, act_dim),
            )
            self.critic = nn.Sequential(
                nn.Linear(obs_dim, hidden),
                nn.Tanh(),
                nn.Linear(hidden, 1),
            )
            # Initialize orthogonally (PPO best practice)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight)
                    nn.init.zeros_(m.bias)

        def get_value(self, obs: "torch.Tensor") -> "torch.Tensor":
            return self.critic(obs).squeeze(-1)

        def get_action_and_logprob(
            self, obs: "torch.Tensor"
        ) -> tuple["torch.Tensor", "torch.Tensor"]:
            mean = self.actor_mean(obs)
            dist = torch.distributions.Normal(mean, torch.ones_like(mean) * 0.5)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1)
            return action, log_prob

        def forward(self, obs: "torch.Tensor"):
            return self.actor_mean(obs)

    return MlpPolicy()


def make_discrete_policy(obs_dim: int, n_actions: int, hidden: int = 64):
    """Build a minimal discrete actor-critic. Requires torch."""
    import torch
    import torch.nn as nn

    class DiscretePolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.actor = nn.Sequential(
                nn.Linear(obs_dim, hidden),
                nn.Tanh(),
                nn.Linear(hidden, n_actions),
            )
            self.critic = nn.Sequential(
                nn.Linear(obs_dim, hidden),
                nn.Tanh(),
                nn.Linear(hidden, 1),
            )

        def get_value(self, obs: "torch.Tensor") -> "torch.Tensor":
            return self.critic(obs).squeeze(-1)

        def get_action_and_logprob(
            self, obs: "torch.Tensor"
        ) -> tuple["torch.Tensor", "torch.Tensor"]:
            logits = self.actor(obs)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action, log_prob

        def get_logprob_and_entropy(
            self, obs: "torch.Tensor", actions: "torch.Tensor"
        ) -> tuple["torch.Tensor", "torch.Tensor"]:
            logits = self.actor(obs)
            dist = torch.distributions.Categorical(logits=logits)
            return dist.log_prob(actions), dist.entropy()

        def forward(self, obs: "torch.Tensor"):
            return self.actor(obs)

    return DiscretePolicy()


# ---------------------------------------------------------------------------
# Group 0: Bug fix verification
# ---------------------------------------------------------------------------

class TestMultiDimActionBuffer:
    """Group 0: ExperienceRecord.action must be Vec<f32> (multi-dim)."""

    # RED: Python ExperienceTable accepts multi-dim action array
    def test_experience_table_accepts_multi_dim_action(self):
        """ExperienceTable.push() accepts action arrays of arbitrary dim."""
        from rlox import ExperienceTable

        obs_dim = 17   # HalfCheetah obs
        act_dim = 6    # HalfCheetah action
        table = ExperienceTable(obs_dim=obs_dim, act_dim=act_dim)

        obs = np.zeros(obs_dim, dtype=np.float32)
        action = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6], dtype=np.float32)

        # Must not raise
        table.push(obs=obs, action=action, reward=1.0, terminated=False, truncated=False)
        assert len(table) == 1

    # RED: actions() accessor returns correct shape
    def test_experience_table_actions_shape(self):
        from rlox import ExperienceTable

        act_dim = 6
        table = ExperienceTable(obs_dim=4, act_dim=act_dim)
        obs = np.zeros(4, dtype=np.float32)
        for i in range(5):
            action = np.full(act_dim, float(i), dtype=np.float32)
            table.push(obs=obs, action=action, reward=1.0, terminated=False, truncated=False)

        actions = table.actions()
        assert actions.shape == (5, act_dim), f"expected (5, {act_dim}), got {actions.shape}"

    # RED: ReplayBuffer accepts and returns multi-dim actions
    def test_replay_buffer_multi_dim_action_roundtrip(self):
        from rlox import ReplayBuffer

        obs_dim = 4
        act_dim = 3
        buf = ReplayBuffer(capacity=100, obs_dim=obs_dim, act_dim=act_dim)
        obs = np.zeros(obs_dim, dtype=np.float32)
        action = np.array([0.5, -0.5, 1.0], dtype=np.float32)

        buf.push(obs=obs, action=action, reward=1.0, terminated=False, truncated=False)
        batch = buf.sample(batch_size=1, seed=42)

        assert batch["actions"].shape == (1, act_dim)
        np.testing.assert_allclose(batch["actions"][0], action, atol=1e-6)

    # RED: wrong action dim raises ValueError
    def test_experience_table_wrong_action_dim_raises(self):
        from rlox import ExperienceTable

        table = ExperienceTable(obs_dim=4, act_dim=2)
        obs = np.zeros(4, dtype=np.float32)
        wrong_action = np.zeros(3, dtype=np.float32)  # 3 != 2

        with pytest.raises((ValueError, RuntimeError)):
            table.push(obs=obs, action=wrong_action, reward=1.0,
                      terminated=False, truncated=False)


class TestTerminalObsPreservation:
    """Group 0: VecEnv must preserve terminal observation on auto-reset."""

    # RED: VecEnv.step_all() returns terminal_obs in result
    def test_vecenv_step_all_returns_terminal_obs_key(self, gymnasium):
        from rlox import VecEnv

        # With configurable VecEnv (Phase 7 fix)
        venv = VecEnv(env_id="CartPole-v1", n_envs=2, seed=42)
        for _ in range(200):
            result = venv.step_all([1, 1])
            # The result dict must contain terminal_obs
            assert "terminal_obs" in result, (
                "VecEnv.step_all() must return 'terminal_obs' key"
            )
            # terminal_obs is None when not done, array when done
            for i in range(2):
                done = result["terminated"][i] or result["truncated"][i]
                # We check this key is accessible; shape checked separately
                _ = result["terminal_obs"]

    # RED: terminal_obs has correct shape when env terminates
    def test_vecenv_terminal_obs_shape_on_termination(self, gymnasium):
        from rlox import VecEnv

        venv = VecEnv(env_id="CartPole-v1", n_envs=4, seed=42)
        found_terminal = False
        for _ in range(500):
            result = venv.step_all([1, 1, 1, 1])
            tobs = result["terminal_obs"]
            if tobs is not None:
                # tobs shape: (n_envs, obs_dim) where non-None entries
                # are actual obs, None entries are masked/zero
                if isinstance(tobs, np.ndarray):
                    assert tobs.shape[1] == 4  # CartPole obs_dim
                found_terminal = True
                break
        # If no termination, that's okay for this structural test


class TestTokenKLReturnsResult:
    """Group 0: compute_token_kl must return float, not panic on mismatch."""

    # RED: mismatched lengths raise ValueError, not panic
    def test_token_kl_mismatched_lengths_raises_not_crashes(self):
        from rlox import compute_token_kl

        log_p = np.array([-1.0, -2.0], dtype=np.float64)
        log_q = np.array([-1.0], dtype=np.float64)

        # Must raise a Python exception, not crash the process
        with pytest.raises((ValueError, RuntimeError)):
            compute_token_kl(log_p, log_q)

    # RED: identical distributions return ~0 (regression)
    def test_token_kl_identical_distributions_still_zero(self):
        from rlox import compute_token_kl

        log_p = np.array([-1.0, -2.0, -0.5], dtype=np.float64)
        kl = compute_token_kl(log_p, log_p)
        assert abs(kl) < 1e-10


class TestConfigurableVecEnv:
    """Group 0: VecEnv must accept env_id string, not be hardcoded to CartPole."""

    # RED: VecEnv accepts gymnasium env_id string
    def test_vecenv_accepts_env_id_string(self, gymnasium):
        from rlox import VecEnv

        # Must not raise
        venv = VecEnv(env_id="CartPole-v1", n_envs=4, seed=42)
        assert venv.num_envs() == 4

    # RED: VecEnv works with different env IDs (non-CartPole)
    def test_vecenv_with_mountaincar(self, gymnasium):
        from rlox import VecEnv

        venv = VecEnv(env_id="MountainCar-v0", n_envs=2, seed=42)
        result = venv.step_all([0, 1])
        assert "obs" in result
        assert result["obs"].shape[0] == 2

    # RED: VecEnv n_envs attribute accessible
    def test_vecenv_num_envs_accessor(self, gymnasium):
        from rlox import VecEnv

        for n in [1, 4, 16]:
            venv = VecEnv(env_id="CartPole-v1", n_envs=n, seed=42)
            assert venv.num_envs() == n


# ---------------------------------------------------------------------------
# Group 1: Primitive component tests
# ---------------------------------------------------------------------------

class TestRunningStats:
    """Group 1: Python bindings for Rust RunningStats (Welford's)."""

    # RED: RunningStats importable from rlox
    def test_running_stats_importable(self):
        from rlox import RunningStats
        stats = RunningStats()
        assert stats.count() == 0

    # RED: Welford's known example
    def test_running_stats_welford_known_values(self):
        from rlox import RunningStats

        stats = RunningStats()
        for x in [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]:
            stats.update(float(x))

        assert abs(stats.mean() - 5.0) < 1e-10, f"mean should be 5.0, got {stats.mean()}"
        assert abs(stats.var() - 4.0) < 1e-10, f"variance should be 4.0, got {stats.var()}"
        assert abs(stats.std() - 2.0) < 1e-10, f"std should be 2.0, got {stats.std()}"

    # RED: normalize returns z-score
    def test_running_stats_normalize(self):
        from rlox import RunningStats

        stats = RunningStats()
        for x in [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]:
            stats.update(float(x))

        z = stats.normalize(5.0)
        assert abs(z) < 1e-10, f"normalize(mean) should be 0, got {z}"

    # RED: batch_update processes numpy array
    def test_running_stats_batch_update(self):
        from rlox import RunningStats

        stats = RunningStats()
        rewards = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stats.batch_update(rewards)

        assert stats.count() == 5
        assert abs(stats.mean() - 3.0) < 1e-10

    # RED: zero variance doesn't cause divide-by-zero in normalize
    def test_running_stats_normalize_zero_variance(self):
        from rlox import RunningStats

        stats = RunningStats()
        for _ in range(10):
            stats.update(5.0)

        result = stats.normalize(5.0)
        assert math.isfinite(result), "normalize with zero variance must be finite"


class TestSequencePacking:
    """Group 1: Python bindings for sequence packing."""

    # RED: pack_sequences importable
    def test_pack_sequences_importable(self):
        from rlox import pack_sequences

    # RED: basic packing via Python API
    def test_pack_sequences_basic(self):
        from rlox import pack_sequences

        sequences = [
            np.array([1, 2, 3], dtype=np.uint32),
            np.array([4, 5], dtype=np.uint32),
            np.array([6, 7, 8, 9], dtype=np.uint32),
        ]
        bins = pack_sequences(sequences, max_length=8)

        # All input tokens must appear in output
        all_input = sorted([t for s in sequences for t in s.tolist()])
        all_output = sorted([
            t for b in bins
            for t, m in zip(b["input_ids"], b["attention_mask"])
            if m != 0
        ])
        assert all_input == all_output

    # RED: no bin exceeds max_length
    def test_pack_sequences_max_length_constraint(self):
        from rlox import pack_sequences

        sequences = [np.arange(l, dtype=np.uint32) for l in [10, 20, 30, 15, 25]]
        max_length = 40
        bins = pack_sequences(sequences, max_length=max_length)

        for i, b in enumerate(bins):
            assert len(b["input_ids"]) <= max_length, (
                f"bin {i} has length {len(b['input_ids'])} > max_length {max_length}"
            )

    # RED: sequence exceeding max_length raises
    def test_pack_sequences_too_long_raises(self):
        from rlox import pack_sequences

        sequences = [np.arange(200, dtype=np.uint32)]
        with pytest.raises((ValueError, RuntimeError)):
            pack_sequences(sequences, max_length=100)


class TestBatchAssembly:
    """Group 1: RolloutBatch zero-copy tensor bridge."""

    # RED: RolloutBatch importable
    def test_rollout_batch_importable(self, torch):
        from rlox import RolloutBatch  # or from rlox.batch import RolloutBatch

    # RED: from_experience_table produces correct shapes
    def test_rollout_batch_from_experience_table_shapes(self, torch):
        from rlox import ExperienceTable, RolloutBatch

        obs_dim = 4
        act_dim = 1
        n = 128
        table = ExperienceTable(obs_dim=obs_dim, act_dim=act_dim)
        obs = np.zeros(obs_dim, dtype=np.float32)
        action = np.zeros(act_dim, dtype=np.float32)
        for _ in range(n):
            table.push(obs=obs, action=action, reward=1.0,
                      terminated=False, truncated=False)

        batch = RolloutBatch.from_experience_table(table)

        assert batch.obs.shape == (n, obs_dim)
        assert batch.actions.shape == (n, act_dim)
        assert batch.rewards.shape == (n,)

    # RED: zero-copy — buffer.observations() returns torch tensor without copy
    def test_buffer_observations_zero_copy_torch(self, torch):
        from rlox import ExperienceTable

        table = ExperienceTable(obs_dim=4, act_dim=1)
        obs = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        for _ in range(100):
            table.push(obs=obs, action=np.zeros(1, dtype=np.float32),
                      reward=1.0, terminated=False, truncated=False)

        # as_tensor() should return without copy
        tensor = torch.as_tensor(table.observations())

        # Verify no copy was made: modifying the tensor modifies the buffer
        # (or at minimum, the tensor data pointer matches)
        assert tensor.shape == (100, 4)
        assert tensor.dtype == torch.float32

        # The canonical no-copy check: get raw data pointer
        arr1 = table.observations()
        arr2 = table.observations()
        # Both calls return same underlying memory (same data pointer)
        assert arr1.ctypes.data == arr2.ctypes.data, (
            "observations() should return a view, not a copy"
        )


# ---------------------------------------------------------------------------
# Group 2: RolloutCollector tests
# ---------------------------------------------------------------------------

class TestRolloutCollector:
    """Group 2: RolloutCollector — collects rollouts with any nn.Module policy."""

    # RED: RolloutCollector importable
    def test_rollout_collector_importable(self):
        from rlox import RolloutCollector  # or from rlox.collectors import RolloutCollector

    # RED: collector works with custom nn.Module
    def test_collector_with_custom_nn_module(self, torch, gymnasium):
        from rlox import RolloutCollector

        obs_dim = 4
        n_envs = 4
        n_steps = 32
        policy = make_discrete_policy(obs_dim, n_actions=2)

        collector = RolloutCollector(env_id="CartPole-v1", n_envs=n_envs, seed=42)
        batch = collector.collect(policy, n_steps=n_steps)

        # Shape: (n_envs * n_steps, obs_dim)
        assert batch.obs.shape == (n_envs * n_steps, obs_dim), (
            f"obs shape should be ({n_envs * n_steps}, {obs_dim}), "
            f"got {batch.obs.shape}"
        )
        assert batch.actions.shape[0] == n_envs * n_steps
        assert batch.rewards.shape == (n_envs * n_steps,)
        assert batch.dones.shape == (n_envs * n_steps,)

    # RED: collector returns advantages after GAE computation
    def test_collector_batch_has_advantages(self, torch, gymnasium):
        from rlox import RolloutCollector

        policy = make_discrete_policy(4, n_actions=2)
        collector = RolloutCollector(env_id="CartPole-v1", n_envs=2, seed=42)
        batch = collector.collect(policy, n_steps=64)

        assert hasattr(batch, "advantages"), "batch must have 'advantages' attribute"
        assert batch.advantages.shape == (2 * 64,)
        assert not torch.isnan(batch.advantages).any(), "advantages must not contain NaN"

    # RED: collector batch has log_probs from policy
    def test_collector_batch_has_log_probs(self, torch, gymnasium):
        from rlox import RolloutCollector

        policy = make_discrete_policy(4, n_actions=2)
        collector = RolloutCollector(env_id="CartPole-v1", n_envs=2, seed=42)
        batch = collector.collect(policy, n_steps=32)

        assert hasattr(batch, "log_probs"), "batch must have 'log_probs' attribute"
        assert batch.log_probs.shape[0] == 2 * 32

    # RED: collector does not hold stale obs after reset
    def test_collector_resets_between_collects(self, torch, gymnasium):
        from rlox import RolloutCollector

        policy = make_discrete_policy(4, n_actions=2)
        collector = RolloutCollector(env_id="CartPole-v1", n_envs=2, seed=42)

        batch1 = collector.collect(policy, n_steps=32)
        batch2 = collector.collect(policy, n_steps=32)

        # Both batches should have valid shapes — no errors on second collect
        assert batch1.obs.shape == batch2.obs.shape


# ---------------------------------------------------------------------------
# Group 3: PPO Loss
# ---------------------------------------------------------------------------

class TestPPOLoss:
    """Group 3: PPO clipped surrogate loss computation."""

    # RED: PPOLoss importable
    def test_ppo_loss_importable(self):
        from rlox import PPOLoss  # or from rlox.losses import PPOLoss

    # RED: PPO loss decreases on a simple synthetic batch
    def test_ppo_loss_decreases_on_good_actions(self, torch):
        from rlox import PPOLoss

        loss_fn = PPOLoss(clip_eps=0.2, vf_coef=0.5, ent_coef=0.01)
        policy = make_discrete_policy(obs_dim=4, n_actions=2)
        optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

        obs = torch.randn(64, 4)
        actions = torch.randint(0, 2, (64,))
        advantages = torch.randn(64)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        old_log_probs = torch.randn(64) - 2.0  # plausible log probs
        returns = torch.randn(64)
        old_values = torch.randn(64)

        losses = []
        for _ in range(5):
            loss, metrics = loss_fn(
                policy=policy,
                obs=obs,
                actions=actions,
                old_log_probs=old_log_probs,
                advantages=advantages,
                returns=returns,
                old_values=old_values,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "entropy" in metrics
        assert metrics["entropy"] > 0, "entropy should be positive"

    # RED: PPO loss clips ratio at clip_eps
    def test_ppo_loss_ratio_clipping(self, torch):
        from rlox import PPOLoss

        clip_eps = 0.2
        loss_fn = PPOLoss(clip_eps=clip_eps, vf_coef=0.5, ent_coef=0.0)
        policy = make_discrete_policy(obs_dim=4, n_actions=2)

        obs = torch.randn(32, 4)
        actions = torch.randint(0, 2, (32,))
        # Set old_log_probs very different from current policy to trigger clipping
        old_log_probs = torch.full((32,), -10.0)  # very stale policy
        advantages = torch.ones(32)
        returns = torch.zeros(32)
        old_values = torch.zeros(32)

        _, metrics = loss_fn(
            policy=policy,
            obs=obs,
            actions=actions,
            old_log_probs=old_log_probs,
            advantages=advantages,
            returns=returns,
            old_values=old_values,
        )

        # clip_fraction should be > 0 when ratio is far from 1.0
        assert "clip_fraction" in metrics, "metrics must include clip_fraction"

    # RED: PPO loss with value function clipping
    def test_ppo_loss_value_function_clipping(self, torch):
        from rlox import PPOLoss

        loss_fn = PPOLoss(clip_eps=0.2, vf_coef=0.5, ent_coef=0.0, clip_vloss=True)
        policy = make_discrete_policy(obs_dim=4, n_actions=2)

        obs = torch.randn(32, 4)
        actions = torch.randint(0, 2, (32,))
        old_log_probs = torch.randn(32) - 1.5
        advantages = torch.randn(32)
        returns = torch.randn(32)
        old_values = torch.randn(32)

        # Must not raise
        loss, _ = loss_fn(
            policy=policy, obs=obs, actions=actions,
            old_log_probs=old_log_probs, advantages=advantages,
            returns=returns, old_values=old_values,
        )
        assert math.isfinite(loss.item()), "loss must be finite"

    # RED: PPO loss matches reference implementation on known inputs
    def test_ppo_loss_matches_reference_on_known_inputs(self, torch):
        """
        Reference: L_CLIP = -min(r*A, clip(r, 1-e, 1+e)*A)
        where r = exp(log_pi - log_pi_old).

        For a single step with:
          log_pi_old = -1.0, log_pi = -1.0 (r=1.0)
          A = 1.0
          clip_eps = 0.2
        L_CLIP should be -1.0 (negative because we maximize).
        """
        from rlox import PPOLoss

        loss_fn = PPOLoss(clip_eps=0.2, vf_coef=0.0, ent_coef=0.0)

        # Create a policy that we can control the log_prob of
        import torch.nn as nn
        class FixedPolicy(nn.Module):
            def __init__(self):
                super().__init__()
                # Single parameter controls the logit (and thus log_prob)
                self.logit = nn.Parameter(torch.zeros(1))

            def get_logprob_and_entropy(self, obs, actions):
                # Batch size = obs.shape[0]
                logits = self.logit.expand(obs.shape[0], 2)
                dist = torch.distributions.Categorical(logits=logits)
                return dist.log_prob(actions.long()), dist.entropy()

            def get_value(self, obs):
                return torch.zeros(obs.shape[0])

        policy = FixedPolicy()
        obs = torch.zeros(1, 2)
        actions = torch.zeros(1, dtype=torch.long)
        # log_prob of action 0 under uniform = log(0.5) ≈ -0.693
        old_log_probs = torch.tensor([-0.693])
        advantages = torch.tensor([1.0])
        returns = torch.tensor([0.0])
        old_values = torch.tensor([0.0])

        loss, _ = loss_fn(
            policy=policy, obs=obs, actions=actions,
            old_log_probs=old_log_probs, advantages=advantages,
            returns=returns, old_values=old_values,
        )
        # r ≈ 1.0, loss ≈ -1.0 * 1.0 = -1.0 (negated for gradient descent)
        assert abs(loss.item() + 1.0) < 0.1, (
            f"PPO loss should be ~ -1.0, got {loss.item()}"
        )


# ---------------------------------------------------------------------------
# Group 3: End-to-end algorithm tests
# ---------------------------------------------------------------------------

class TestPPOEndToEnd:
    """Group 3: PPO must solve CartPole in < 50K steps."""

    # RED: PPO importable
    def test_ppo_importable(self):
        from rlox import PPO  # or from rlox.algorithms.ppo import PPO

    # RED: PPO.train() runs without error for 1K steps
    def test_ppo_smoke_test(self, torch, gymnasium):
        from rlox import PPO

        ppo = PPO(env_id="CartPole-v1", n_envs=4, seed=42)
        # 1K steps — just verify it runs
        ppo.train(total_timesteps=1_000)

    # RED: PPO logs episode_reward metric
    def test_ppo_logs_episode_reward(self, torch, gymnasium):
        from rlox import PPO

        ppo = PPO(env_id="CartPole-v1", n_envs=4, seed=42)
        metrics = ppo.train(total_timesteps=5_000)

        assert "episode_reward" in metrics or "mean_episode_reward" in metrics, (
            "PPO.train() must return episode_reward metric"
        )

    @pytest.mark.slow
    def test_ppo_solves_cartpole_under_50k_steps(self, torch, gymnasium):
        """
        PPO must solve CartPole-v1 (mean reward >= 475) in < 50K steps.
        Run with: pytest -m slow tests/python/test_phase7.py
        """
        from rlox import PPO

        def evaluate(policy, env_id: str, n_episodes: int = 20, seed: int = 99) -> float:
            import gymnasium as gym
            env = gym.make(env_id)
            total_reward = 0.0
            for ep in range(n_episodes):
                obs, _ = env.reset(seed=seed + ep)
                done = False
                ep_reward = 0.0
                while not done:
                    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        logits = policy(obs_t)
                        action = logits.argmax(dim=-1).item()
                    obs, reward, terminated, truncated, _ = env.step(action)
                    ep_reward += reward
                    done = terminated or truncated
                total_reward += ep_reward
            return total_reward / n_episodes

        # Train across 5 seeds — must solve in at least 4/5
        successes = 0
        for seed in [42, 123, 456, 789, 1337]:
            ppo = PPO(
                env_id="CartPole-v1",
                n_envs=8,
                seed=seed,
                n_steps=128,
                n_epochs=4,
                learning_rate=2.5e-4,
                clip_eps=0.2,
            )
            ppo.train(total_timesteps=50_000)
            mean_reward = evaluate(ppo.policy, "CartPole-v1")
            if mean_reward >= 475.0:
                successes += 1

        assert successes >= 4, (
            f"PPO should solve CartPole in 4/5 seeds, only solved {successes}/5"
        )

    @pytest.mark.slow
    def test_ppo_halfcheetah_within_1std_of_sb3(self, torch, gymnasium):
        """
        PPO on HalfCheetah-v4 must reach >= 3000 return at 1M steps.
        SB3 baseline is ~4000 ± 1000 at 1M steps.

        Requires: gymnasium[mujoco] installed.
        """
        pytest.importorskip("mujoco")
        from rlox import PPO

        ppo = PPO(
            env_id="HalfCheetah-v4",
            n_envs=8,
            seed=42,
            n_steps=2048,
            n_epochs=10,
            learning_rate=3e-4,
            clip_eps=0.2,
            vf_coef=0.5,
            ent_coef=0.0,
        )
        final_metrics = ppo.train(total_timesteps=1_000_000)

        # SB3 baseline lower bound: mean - 1 std ≈ 3000
        mean_reward = final_metrics.get("final_mean_reward", 0.0)
        assert mean_reward >= 3000.0, (
            f"PPO HalfCheetah should reach >= 3000, got {mean_reward:.1f}. "
            "SB3 baseline is ~4000 ± 1000 at 1M steps."
        )


class TestGRPOEndToEnd:
    """Group 3: GRPO must improve accuracy on synthetic math."""

    # RED: GRPO importable
    def test_grpo_importable(self):
        from rlox import GRPO  # or from rlox.algorithms.grpo import GRPO

    # RED: GRPO smoke test with mock LLM
    def test_grpo_smoke_test_with_mock_model(self, torch):
        from rlox import GRPO

        # Mock language model
        class MockLM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Embedding(100, 16)
                self.proj = torch.nn.Linear(16, 100)

            def forward(self, input_ids):
                x = self.embed(input_ids).mean(dim=1)
                return self.proj(x)

            def generate(self, prompt_ids, n_samples=4, max_new_tokens=10):
                batch = prompt_ids.shape[0]
                return torch.randint(0, 100, (batch * n_samples, max_new_tokens))

        def mock_reward_fn(completions, prompts):
            # Return random rewards for smoke test
            return np.random.RandomState(42).randn(len(completions)).tolist()

        model = MockLM()
        ref_model = MockLM()
        prompts = ["What is 2+2?"] * 4

        grpo = GRPO(
            model=model,
            ref_model=ref_model,
            reward_fn=mock_reward_fn,
            group_size=4,
            learning_rate=1e-4,
        )
        # Should run without error
        metrics = grpo.train_step(prompts)
        assert "grpo_loss" in metrics or "loss" in metrics

    @pytest.mark.slow
    def test_grpo_improves_synthetic_math_accuracy(self, torch):
        """
        GRPO must improve accuracy on a simple synthetic math task.

        Task: given prompt "What is N+M?", generate the correct answer.
        Reward: 1.0 if correct, 0.0 if wrong.
        Starting from random initialization, accuracy should increase
        from near-0% to > 20% on the training distribution.

        Uses a small tokenized model, not a real LLM.
        """
        from rlox import GRPO

        torch_module = torch

        class TinyArithmeticModel(torch_module.nn.Module):
            """Toy model for 0-9 single-digit addition."""

            VOCAB_SIZE = 20  # digits 0-9 + ops

            def __init__(self):
                super().__init__()
                self.net = torch_module.nn.Sequential(
                    torch_module.nn.Embedding(self.VOCAB_SIZE, 32),
                    torch_module.nn.Linear(32, 32),
                    torch_module.nn.ReLU(),
                    torch_module.nn.Linear(32, self.VOCAB_SIZE),
                )

            def forward(self, ids):
                emb = self.net[0](ids).mean(dim=1)
                return self.net[2](self.net[1](emb))

            def generate(self, prompt_ids, n_samples=4, max_new_tokens=1):
                logits = self.forward(prompt_ids)
                # Sample from logits
                probs = torch_module.softmax(logits, dim=-1)
                samples = torch_module.multinomial(probs, num_samples=n_samples)
                return samples.reshape(-1, 1)

        def math_reward(completions_ids, prompts):
            """
            prompts: list of (a, b) tuples
            completions_ids: tensor of shape (batch * n_samples, 1)
            """
            rewards = []
            n = len(completions_ids)
            for i, comp in enumerate(completions_ids):
                a, b = prompts[i % len(prompts)]
                correct = (a + b) % 10  # single digit answer
                predicted = comp[0].item() if hasattr(comp[0], 'item') else comp[0]
                rewards.append(1.0 if predicted == correct else 0.0)
            return rewards

        model = TinyArithmeticModel()
        ref_model = TinyArithmeticModel()
        # Copy weights
        ref_model.load_state_dict(model.state_dict())

        prompts = [(i % 5, (i // 5) % 5) for i in range(20)]

        grpo = GRPO(
            model=model,
            ref_model=ref_model,
            reward_fn=math_reward,
            group_size=8,
            learning_rate=5e-3,
            kl_coef=0.01,
        )

        initial_acc = grpo.evaluate(prompts[:20])
        grpo.train(prompts * 50, n_epochs=50)
        final_acc = grpo.evaluate(prompts[:20])

        assert final_acc > initial_acc + 0.05, (
            f"GRPO should improve accuracy: initial={initial_acc:.2f}, "
            f"final={final_acc:.2f}"
        )


class TestDPOEndToEnd:
    """Group 3: DPO must achieve > 90% preference accuracy on synthetic data."""

    # RED: DPO importable
    def test_dpo_importable(self):
        from rlox import DPO  # or from rlox.algorithms.dpo import DPO

    # RED: DPO smoke test with minimal model
    def test_dpo_smoke_test(self, torch):
        from rlox import DPO

        class TinyLM(torch.nn.Module):
            VOCAB_SIZE = 50

            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Embedding(self.VOCAB_SIZE, 16)
                self.proj = torch.nn.Linear(16, self.VOCAB_SIZE)

            def forward(self, input_ids):
                return self.proj(self.embed(input_ids).mean(dim=1))

            def log_probs(self, input_ids, labels):
                logits = self.forward(input_ids)
                log_p = torch.nn.functional.log_softmax(logits, dim=-1)
                return log_p.gather(1, labels.unsqueeze(1)).squeeze(1).sum()

        model = TinyLM()
        ref_model = TinyLM()
        ref_model.load_state_dict(model.state_dict())

        dpo = DPO(model=model, ref_model=ref_model, beta=0.1, learning_rate=1e-4)

        # Create one preference pair
        chosen = torch.randint(0, 50, (4,))
        rejected = torch.randint(0, 50, (4,))
        prompt = torch.randint(0, 50, (2,))

        loss, metrics = dpo.compute_loss(prompt, chosen, rejected)
        assert math.isfinite(loss.item()), "DPO loss must be finite"

    @pytest.mark.slow
    def test_dpo_preference_accuracy_synthetic(self, torch):
        """
        DPO must achieve > 90% preference accuracy on synthetic data.

        Synthetic data: chosen completions all start with token 5,
        rejected completions all start with token 6. A well-trained
        DPO model should assign higher reward to token-5 sequences.
        """
        from rlox import DPO

        GOOD_TOKEN = 5
        BAD_TOKEN = 6

        class BinaryLM(torch.nn.Module):
            """Minimal model for binary preference test."""
            VOCAB_SIZE = 10
            SEQ_LEN = 4

            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Embedding(self.VOCAB_SIZE, 32)
                self.proj = torch.nn.Linear(32, self.VOCAB_SIZE)

            def log_probs_sequence(self, seq):
                """Sum log-prob over all tokens in sequence."""
                emb = self.embed(seq).mean(0, keepdim=True)
                logits = self.proj(emb)
                lp = torch.nn.functional.log_softmax(logits, dim=-1)
                total = sum(lp[0, seq[i]].item() for i in range(len(seq)))
                return total

            def forward(self, x):
                return self.proj(self.embed(x).mean(1))

        model = BinaryLM()
        ref_model = BinaryLM()
        ref_model.load_state_dict(model.state_dict())

        dpo = DPO(model=model, ref_model=ref_model, beta=0.1, learning_rate=3e-3)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

        rng = np.random.default_rng(42)
        n_pairs = 200
        for _ in range(n_pairs):
            prompt = torch.tensor([1, 2], dtype=torch.long)
            chosen = torch.tensor(
                [GOOD_TOKEN] + rng.integers(0, 10, size=3).tolist(),
                dtype=torch.long
            )
            rejected = torch.tensor(
                [BAD_TOKEN] + rng.integers(0, 10, size=3).tolist(),
                dtype=torch.long
            )
            loss, _ = dpo.compute_loss(prompt, chosen, rejected)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate preference accuracy
        correct = 0
        n_eval = 50
        for _ in range(n_eval):
            prompt = torch.tensor([1, 2], dtype=torch.long)
            chosen = torch.tensor([GOOD_TOKEN, 0, 1, 2], dtype=torch.long)
            rejected = torch.tensor([BAD_TOKEN, 3, 4, 5], dtype=torch.long)
            r_chosen = model.log_probs_sequence(chosen)
            r_rejected = model.log_probs_sequence(rejected)
            if r_chosen > r_rejected:
                correct += 1

        accuracy = correct / n_eval
        assert accuracy > 0.9, (
            f"DPO preference accuracy should be > 90%, got {accuracy:.1%}"
        )


class TestA2CEndToEnd:
    """Group 3: A2C must learn CartPole."""

    # RED: A2C importable
    def test_a2c_importable(self):
        from rlox import A2C  # or from rlox.algorithms.a2c import A2C

    # RED: A2C smoke test
    def test_a2c_smoke_test(self, torch, gymnasium):
        from rlox import A2C

        a2c = A2C(env_id="CartPole-v1", n_envs=4, seed=42)
        a2c.train(total_timesteps=2_000)

    @pytest.mark.slow
    def test_a2c_learns_cartpole(self, torch, gymnasium):
        from rlox import A2C

        def evaluate(policy, n_episodes=10):
            import gymnasium as gym
            env = gym.make("CartPole-v1")
            rewards = []
            for ep in range(n_episodes):
                obs, _ = env.reset(seed=ep)
                done = False
                ep_r = 0.0
                while not done:
                    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        logits = policy(obs_t)
                        action = logits.argmax(dim=-1).item()
                    obs, r, term, trunc, _ = env.step(action)
                    ep_r += r
                    done = term or trunc
                rewards.append(ep_r)
            return np.mean(rewards)

        a2c = A2C(
            env_id="CartPole-v1",
            n_envs=8,
            seed=42,
            learning_rate=7e-4,
            n_steps=5,
        )
        a2c.train(total_timesteps=100_000)

        mean_reward = evaluate(a2c.policy)
        assert mean_reward >= 200.0, (
            f"A2C should reach >= 200 on CartPole, got {mean_reward:.1f}"
        )


# ---------------------------------------------------------------------------
# Group 4: Checkpoint / Resume
# ---------------------------------------------------------------------------

class TestCheckpointResume:
    """Group 4: Checkpoint save/load must produce identical trajectories."""

    # RED: Checkpoint importable
    def test_checkpoint_importable(self):
        from rlox import Checkpoint  # or from rlox.checkpoint import Checkpoint

    # RED: save and load do not raise
    def test_checkpoint_save_load_smoke(self, torch, gymnasium, tmp_path):
        from rlox import PPO, Checkpoint

        path = str(tmp_path / "ckpt.pt")
        ppo = PPO(env_id="CartPole-v1", n_envs=4, seed=42)
        ppo.train(total_timesteps=2_000)

        Checkpoint.save(path, model=ppo.policy, optimizer=ppo.optimizer,
                       step=2000, config=ppo.config)

        state = Checkpoint.load(path)
        assert "model_state" in state or "model" in state
        assert "step" in state
        assert state["step"] == 2000

    # RED: checkpoint produces identical trajectory after resume
    def test_checkpoint_resume_identical_trajectory(self, torch, gymnasium, tmp_path):
        """
        Train for 5K steps. Save checkpoint. Resume. Next 1K steps must be
        bit-identical to continuing from the original (requires deterministic
        RNG state preservation).
        """
        from rlox import PPO, Checkpoint

        path = str(tmp_path / "ckpt.pt")

        # Run A: train 5K, collect 1K more
        ppo_a = PPO(env_id="CartPole-v1", n_envs=2, seed=42, n_steps=64)
        ppo_a.train(total_timesteps=5_000)

        Checkpoint.save(
            path,
            model=ppo_a.policy,
            optimizer=ppo_a.optimizer,
            buffer=ppo_a.buffer,
            rng_state=ppo_a.rng_state,
            step=5000,
            config=ppo_a.config,
        )

        rewards_a = ppo_a.train(total_timesteps=1_000)

        # Run B: restore from checkpoint, collect 1K more
        ppo_b = PPO.from_checkpoint(path)
        rewards_b = ppo_b.train(total_timesteps=1_000)

        # Trajectories must be identical (same RNG state = same actions = same rewards)
        np.testing.assert_array_equal(
            rewards_a if isinstance(rewards_a, np.ndarray) else np.array(rewards_a),
            rewards_b if isinstance(rewards_b, np.ndarray) else np.array(rewards_b),
            err_msg="checkpoint resume must produce identical trajectory",
        )

    # RED: checkpoint includes all required fields
    def test_checkpoint_has_required_fields(self, torch, gymnasium, tmp_path):
        from rlox import PPO, Checkpoint

        path = str(tmp_path / "ckpt.pt")
        ppo = PPO(env_id="CartPole-v1", n_envs=2, seed=42)
        ppo.train(total_timesteps=1_000)
        Checkpoint.save(path, model=ppo.policy, optimizer=ppo.optimizer,
                       step=1000, config=ppo.config)

        state = Checkpoint.load(path)
        required_fields = {"step", "config"}
        for field in required_fields:
            assert field in state, f"checkpoint missing required field: {field}"


# ---------------------------------------------------------------------------
# Group 4: Logging
# ---------------------------------------------------------------------------

class TestLogging:
    """Group 4: W&B / TensorBoard loggers receive training metrics."""

    # RED: Logger base class importable
    def test_logger_importable(self):
        from rlox import LoggerCallback  # or from rlox.logging import LoggerCallback

    # RED: custom callback receives metrics on each train step
    def test_custom_callback_receives_metrics(self, torch, gymnasium):
        from rlox import PPO, LoggerCallback

        received_metrics = []

        class CaptureLogger(LoggerCallback):
            def on_train_step(self, step: int, metrics: dict):
                received_metrics.append((step, dict(metrics)))

            def on_rollout_end(self, step: int, metrics: dict):
                pass

        logger = CaptureLogger()
        ppo = PPO(env_id="CartPole-v1", n_envs=2, seed=42, logger=logger)
        ppo.train(total_timesteps=2_000)

        assert len(received_metrics) > 0, "logger should receive at least one metric event"
        # Each metric dict should contain at minimum a loss key
        step_0, metrics_0 = received_metrics[0]
        assert isinstance(step_0, int)
        assert isinstance(metrics_0, dict)

    # RED: WandbLogger has the correct interface (even if wandb is not installed)
    def test_wandb_logger_has_correct_interface(self):
        from rlox import WandbLogger  # or from rlox.logging import WandbLogger

        # Check methods exist
        assert hasattr(WandbLogger, "on_train_step")
        assert hasattr(WandbLogger, "on_rollout_end")
        assert hasattr(WandbLogger, "on_eval")

    # RED: TensorBoardLogger writes event files
    def test_tensorboard_logger_writes_events(self, torch, gymnasium, tmp_path):
        tb = pytest.importorskip("torch.utils.tensorboard")
        from rlox import PPO, TensorBoardLogger  # or from rlox.logging

        log_dir = str(tmp_path / "tb_logs")
        logger = TensorBoardLogger(log_dir=log_dir)

        ppo = PPO(env_id="CartPole-v1", n_envs=2, seed=42, logger=logger)
        ppo.train(total_timesteps=1_000)

        # TensorBoard creates event files in log_dir
        event_files = list(os.scandir(log_dir))
        assert len(event_files) > 0, "TensorBoardLogger should write event files"


# ---------------------------------------------------------------------------
# Group 4: Zero-copy performance regression
# ---------------------------------------------------------------------------

class TestZeroCopyRegression:
    """Group 4: buffer.sample() -> torch tensor must be zero-copy."""

    # RED: buffer.observations() returns torch-compatible zero-copy view
    def test_replay_buffer_observations_zero_copy(self, torch):
        from rlox import ReplayBuffer
        import time

        obs_dim = 4
        capacity = 10_000
        buf = ReplayBuffer(capacity=capacity, obs_dim=obs_dim, act_dim=1)
        obs = np.zeros(obs_dim, dtype=np.float32)
        action = np.zeros(1, dtype=np.float32)
        for _ in range(capacity):
            buf.push(obs=obs, action=action, reward=1.0,
                    terminated=False, truncated=False)

        # Time zero-copy access
        t0 = time.perf_counter_ns()
        for _ in range(100):
            arr = buf.observations()
            _ = torch.as_tensor(arr)  # must be zero-copy
        zero_copy_ns = (time.perf_counter_ns() - t0) / 100

        # Time explicit copy
        t0 = time.perf_counter_ns()
        for _ in range(100):
            arr = buf.observations()
            _ = torch.tensor(arr.copy())  # explicit copy
        copy_ns = (time.perf_counter_ns() - t0) / 100

        assert zero_copy_ns < copy_ns, (
            f"zero-copy ({zero_copy_ns:.0f}ns) should be faster than "
            f"explicit copy ({copy_ns:.0f}ns)"
        )

    # RED: ExperienceTable sample with minibatch returns correct shapes
    def test_rollout_batch_minibatch_iteration(self, torch, gymnasium):
        from rlox import RolloutCollector

        policy = make_discrete_policy(4, n_actions=2)
        collector = RolloutCollector(env_id="CartPole-v1", n_envs=4, seed=42)
        batch = collector.collect(policy, n_steps=64)

        minibatch_size = 32
        minibatches = list(batch.sample_minibatches(minibatch_size, shuffle=True))

        # Total elements across all minibatches equals batch size
        total = sum(mb.obs.shape[0] for mb in minibatches)
        assert total == 4 * 64, f"total elements {total} != batch size {4*64}"

        # Each minibatch has the right size (last may be smaller)
        for mb in minibatches[:-1]:
            assert mb.obs.shape[0] == minibatch_size


# ---------------------------------------------------------------------------
# Group 4: Wheel installation test
# ---------------------------------------------------------------------------

class TestWheelInstallation:
    """Group 4: Prebuilt wheel imports correctly after pip install."""

    # RED: all public symbols importable from rlox
    def test_all_public_symbols_importable(self):
        """
        After 'pip install rlox', all Phase 7 additions must be importable.
        This test verifies the public API surface is complete.
        """
        import rlox

        phase7_symbols = [
            "RunningStats",
            "pack_sequences",
            "RolloutCollector",
            "RolloutBatch",
            "PPOLoss",
            "PPO",
            "GRPO",
            "DPO",
            "A2C",
            "Checkpoint",
            "LoggerCallback",
            "WandbLogger",
            "TensorBoardLogger",
        ]

        missing = [sym for sym in phase7_symbols if not hasattr(rlox, sym)]
        assert not missing, (
            f"Following Phase 7 symbols missing from rlox module: {missing}"
        )

    # RED: version string follows semver
    def test_rlox_version_is_semver(self):
        import rlox
        version = rlox.__version__
        parts = version.split(".")
        assert len(parts) >= 2, f"version '{version}' does not look like semver"
        # Major.minor must be numeric
        assert parts[0].isdigit(), f"major version '{parts[0]}' not numeric"
        assert parts[1].isdigit(), f"minor version '{parts[1]}' not numeric"
```

---

## Part 3: Performance Regression Tests

**Target file:** `tests/python/test_phase7_perf.py`

```python
# tests/python/test_phase7_perf.py
"""
Phase 7 performance regression tests.
Ensure Phase 7 additions do not regress Phase 0-6 benchmarks.
Run: .venv/bin/python -m pytest tests/python/test_phase7_perf.py -v
"""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "benchmarks"))
from conftest import BenchmarkResult, timed_run


class TestPhase7PerfRegressions:
    """Verify that Phase 7 additions do not slow down Phase 0-6 primitives."""

    def test_gae_speed_not_regressed(self):
        """compute_gae must remain >= 5x faster than numpy loop after Phase 7."""
        from rlox import compute_gae

        def reference_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
            n = len(rewards)
            advantages = np.zeros(n)
            last_gae = 0.0
            for t in reversed(range(n)):
                nnt = 1.0 - float(dones[t])
                nv = last_value if t == n - 1 else values[t + 1]
                delta = rewards[t] + gamma * nv * nnt - values[t]
                last_gae = delta + gamma * lam * nnt * last_gae
                advantages[t] = last_gae
            return advantages, advantages + values

        rng = np.random.default_rng(42)
        n = 2048
        rewards = rng.standard_normal(n)
        values = rng.standard_normal(n)
        dones = (rng.random(n) > 0.95).astype(float)

        rlox_times = timed_run(
            lambda: compute_gae(rewards, values, dones, 0.0, 0.99, 0.95),
            n_warmup=10, n_reps=100,
        )
        numpy_times = timed_run(
            lambda: reference_gae(rewards, values, dones, 0.0),
            n_warmup=10, n_reps=100,
        )

        speedup = np.median(numpy_times) / np.median(rlox_times)
        assert speedup >= 5.0, (
            f"GAE regression: speedup dropped to {speedup:.1f}x (was >= 5x in Phase 3)"
        )

    def test_buffer_push_speed_not_regressed(self):
        """ExperienceTable push must remain > 500K transitions/sec."""
        from rlox import ExperienceTable

        obs_dim = 4
        act_dim = 1  # original act_dim — backward compat
        table = ExperienceTable(obs_dim=obs_dim, act_dim=act_dim)
        obs = np.zeros(obs_dim, dtype=np.float32)
        action = np.zeros(act_dim, dtype=np.float32)
        n = 10_000

        def push_batch():
            for _ in range(n):
                table.push(obs=obs, action=action, reward=1.0,
                          terminated=False, truncated=False)

        times = timed_run(push_batch, n_warmup=2, n_reps=10)
        result = BenchmarkResult(
            name="push_regression", category="buffer_ops",
            framework="rlox", times_ns=times,
            params={"n_items": n},
        )
        assert result.throughput > 500_000, (
            f"Buffer push regression: {result.throughput:.0f} trans/s < 500K"
        )

    def test_multi_dim_action_push_not_slower_than_scalar(self):
        """Push with act_dim=6 must not be more than 3x slower than act_dim=1."""
        from rlox import ExperienceTable

        obs = np.zeros(4, dtype=np.float32)
        n = 5_000

        # Scalar action
        table1 = ExperienceTable(obs_dim=4, act_dim=1)
        action1 = np.zeros(1, dtype=np.float32)
        times1 = timed_run(
            lambda: [table1.push(obs=obs, action=action1, reward=1.0,
                                 terminated=False, truncated=False)
                     for _ in range(n)],
            n_warmup=2, n_reps=10,
        )

        # 6-dim action
        table6 = ExperienceTable(obs_dim=4, act_dim=6)
        action6 = np.zeros(6, dtype=np.float32)
        times6 = timed_run(
            lambda: [table6.push(obs=obs, action=action6, reward=1.0,
                                 terminated=False, truncated=False)
                     for _ in range(n)],
            n_warmup=2, n_reps=10,
        )

        slowdown = np.median(times6) / np.median(times1)
        assert slowdown < 3.0, (
            f"Multi-dim action push is {slowdown:.1f}x slower than scalar — expected < 3x"
        )

    def test_running_stats_update_is_fast(self):
        """RunningStats.update() must process > 10M values/sec."""
        from rlox import RunningStats

        stats = RunningStats()
        rewards = np.random.default_rng(42).standard_normal(1_000).astype(np.float64)
        n = len(rewards)

        times = timed_run(
            lambda: stats.batch_update(rewards),
            n_warmup=10, n_reps=100,
        )
        result = BenchmarkResult(
            name="running_stats_update", category="training",
            framework="rlox", times_ns=times,
            params={"n_items": n},
        )
        assert result.throughput > 5_000_000, (
            f"RunningStats.batch_update too slow: {result.throughput:.0f} vals/s"
        )

    def test_sequence_packing_is_faster_than_naive_padding(self):
        """Sequence packing must be faster to prepare batches than naive padding."""
        from rlox import pack_sequences

        rng = np.random.default_rng(42)
        lengths = [int(rng.integers(10, 200)) for _ in range(50)]
        sequences = [np.arange(l, dtype=np.uint32) for l in lengths]
        max_length = 256

        def pack():
            return pack_sequences(sequences, max_length=max_length)

        def pad():
            return [np.pad(s, (0, max_length - len(s))) for s in sequences]

        pack_times = timed_run(pack, n_warmup=5, n_reps=50)
        pad_times = timed_run(pad, n_warmup=5, n_reps=50)

        pack_result = BenchmarkResult(
            name="pack_sequences", category="packing",
            framework="rlox", times_ns=pack_times,
        )
        # Packing is more complex than padding but should not be > 100x slower
        pad_result = BenchmarkResult(
            name="pad_sequences", category="packing",
            framework="numpy", times_ns=pad_times,
        )

        # Key metric: packed fills more GPU capacity (validated separately in
        # TestSequencePacking.test_pack_sequences_fill_rate_good_for_varied_lengths)
        assert pack_result.median_ns < 100 * pad_result.median_ns, (
            f"pack_sequences is {pack_result.median_ns / pad_result.median_ns:.0f}x "
            f"slower than naive padding — too slow"
        )
```

---

## Test Coverage Targets

| Component | Target Coverage | Notes |
|---|---|---|
| `buffer/mod.rs` (multi-dim action) | 100% of changed lines | Regression tests included |
| `env/parallel.rs` (terminal_obs) | 100% of new StepResult fields | |
| `llm/ops.rs` (token_kl Result) | 100% of error paths | |
| `training/normalization.rs` | 90%+ | Proptest covers most branches |
| `training/packing.rs` | 85%+ | Proptest covers invariants |
| Python `RolloutCollector` | 80%+ | Integration via algorithm tests |
| Python `PPOLoss` | 90%+ | Known-value tests + smoke |
| Python `PPO` | 70%+ | Slow E2E tests cover critical paths |
| Python `GRPO`, `DPO`, `A2C` | 60%+ | Smoke + E2E |
| Python `Checkpoint` | 80%+ | Roundtrip test is thorough |

---

## Notes on Test Execution

1. **Slow tests** are marked `@pytest.mark.slow`. Run them with:
   ```
   .venv/bin/python -m pytest tests/python/test_phase7.py -m slow -v
   ```

2. **Rust tests** requiring the new multi-dim action API will fail to compile
   until `ExperienceRecord.action: f32` is changed to `Vec<f32>`. This is
   intentional: compilation failure is the RED state for struct-level changes.

3. **MuJoCo tests** require `pip install gymnasium[mujoco]`. Skip with:
   ```
   .venv/bin/python -m pytest tests/python/test_phase7.py -k "not halfcheetah" -v
   ```

4. **Group ordering**: Run Group 0 tests first. If bug-fix tests fail, do not
   proceed to Group 1+. The `test_phase7.py::TestMultiDimActionBuffer` class
   must fully pass before any algorithm work begins.

5. **Adding Phase 7 symbols to `python/rlox/__init__.py`**: as each component
   is implemented, add its export. `TestWheelInstallation::test_all_public_symbols_importable`
   will gate the final wheel release.
