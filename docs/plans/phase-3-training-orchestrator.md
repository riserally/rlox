# Phase 3 — Training Orchestrator + PPO

**Status: NOT STARTED**
**Duration: Weeks 5-8**
**PRD Features: 3.1, 3.2, 3.4, 4.1, 7.1, 7.2, 7.3, 7.9, 8.3**
**Depends on: Phase 1 (VecEnv), Phase 2 (ExperienceTable, zero-copy bridge)**

---

## Objective

Build the training loop orchestration layer in Rust and deliver PPO solving CartPole end-to-end from Python. This is the first proof-of-value: PPO matching CleanRL performance with measurably faster env stepping and GAE computation.

The Rust side handles everything EXCEPT gradient computation — collecting experience, computing advantages (GAE), assembling batches, managing the KL penalty, and orchestrating the collect-train loop. PyTorch handles forward passes, loss computation, and backpropagation.

## PRD Feature Mapping

| PRD # | Feature | Priority | Target |
|-------|---------|----------|--------|
| 3.1 | Rust-native GAE / advantage computation | P0 | 5-10x faster than SB3's numpy GAE |
| 3.2 | Batch assembly pipeline | P0 | Zero-copy PyTorch tensors from Rust buffer |
| 3.4 | KL-penalty / KL-controller | P0 | Adaptive coefficient for RLHF/PPO |
| 4.1 | PPO (clip + KL variants) | P0 | End-to-end training on CartPole |
| 7.1 | "One-liner" training API | P0 | `PPOTrainer(env=...).train(N)` |
| 7.2 | Composable config system | P0 | Dataclass-based config with defaults |
| 7.3 | Custom model support | P0 | Any nn.Module works as policy |
| 7.9 | Comprehensive error messages | P0 | Clear Python exceptions, not Rust backtraces |
| 8.3 | Checkpoint / resume | P0 | Save/restore full training state |

## Reasoning

### Why GAE in Rust?

Generalized Advantage Estimation is a sequential scan over the reward/value arrays — it cannot be easily vectorized in numpy. TorchRL showed 10.6x speedup over SB3's numpy GAE simply by using C++. Rust should match or exceed this. The algorithm is simple (~20 lines) but is called on every training iteration with large arrays (2048 steps * 128 envs = 262K elements).

### Why the Rust-Python boundary here?

The training loop has a natural split:
- **Rust does**: env stepping, experience storage, GAE computation, batch assembly, rollout orchestration, KL tracking
- **Python does**: model forward pass, loss computation, `loss.backward()`, optimizer step

This split maximizes Rust's advantage (eliminating Python interpreter overhead in the data pipeline) while keeping gradient computation in PyTorch where it's battle-tested and GPU-optimized. The boundary crossing happens once per minibatch (not per transition), so PyO3's ~50ns overhead is negligible.

### Why dataclass config over YAML/dict?

RLlib's nested dict config is a known pain point. Hydra adds complexity. Python dataclasses give us:
- Full type hints and IDE autocompletion
- Default values that are visible in source
- `asdict()` for serialization
- Composability via inheritance or nesting

### Why checkpoint Rust state separately from model weights?

Model weights are saved by PyTorch (`torch.save`). Rust state (buffer contents, RNG state, training step counter, KL coefficient) is saved via serde. This separation means the framework doesn't need to understand PyTorch's serialization format, and users can swap models while keeping training state.

## TDD Test Specifications

### Rust unit tests — GAE Computation

```rust
// crates/rlox-core/src/training/gae.rs

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gae_single_step_episode() {
        // Single step: advantage = reward + gamma*V(next)*(1-done) - V(current)
        // With done=true: advantage = reward - V(current) = 1.0 - 0.5 = 0.5
        let rewards = &[1.0];
        let values = &[0.5];
        let dones = &[true];
        let last_value = 0.0;
        let gamma = 0.99;
        let gae_lambda = 0.95;
        let (advantages, returns) = compute_gae(rewards, values, dones, last_value, gamma, gae_lambda);
        assert_eq!(advantages.len(), 1);
        assert!((advantages[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn gae_multi_step_no_termination() {
        // 3 steps, no termination, constant reward=1 and value=0
        // GAE should produce discounted advantages
        let rewards = &[1.0, 1.0, 1.0];
        let values = &[0.0, 0.0, 0.0];
        let dones = &[false, false, false];
        let last_value = 0.0;
        let gamma = 0.99;
        let gae_lambda = 0.95;
        let (advantages, returns) = compute_gae(rewards, values, dones, last_value, gamma, gae_lambda);
        assert_eq!(advantages.len(), 3);
        // Last step: delta = 1.0 + 0.99*0 - 0 = 1.0, A = 1.0
        assert!((advantages[2] - 1.0).abs() < 1e-6);
        // Second step: delta = 1.0, A = 1.0 + 0.99*0.95*1.0 = 1.9405
        assert!((advantages[1] - 1.9405).abs() < 1e-4);
        // First step: A = 1.0 + 0.99*0.95*1.9405 = 2.82...
        assert!(advantages[0] > advantages[1]); // earlier steps have larger advantages
    }

    #[test]
    fn gae_resets_at_episode_boundary() {
        // Episode terminates at step 1, advantage should NOT propagate back
        let rewards = &[1.0, 1.0, 1.0];
        let values = &[0.0, 0.0, 0.0];
        let dones = &[false, true, false]; // episode boundary at step 1
        let last_value = 0.0;
        let gamma = 0.99;
        let gae_lambda = 0.95;
        let (advantages, _) = compute_gae(rewards, values, dones, last_value, gamma, gae_lambda);
        // Step 2 starts a new episode, should not be influenced by step 1
        // Step 1 (terminal): delta = 1.0 + 0 - 0 = 1.0
        assert!((advantages[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn gae_returns_are_advantages_plus_values() {
        let rewards = &[1.0, 2.0, 3.0];
        let values = &[0.5, 1.0, 1.5];
        let dones = &[false, false, true];
        let last_value = 0.0;
        let (advantages, returns) = compute_gae(rewards, values, dones, last_value, 0.99, 0.95);
        // returns[i] = advantages[i] + values[i]
        for i in 0..3 {
            assert!((returns[i] - (advantages[i] + values[i])).abs() < 1e-6);
        }
    }

    #[test]
    fn gae_empty_input() {
        let (advantages, returns) = compute_gae(&[], &[], &[], 0.0, 0.99, 0.95);
        assert!(advantages.is_empty());
        assert!(returns.is_empty());
    }

    #[test]
    fn gae_lambda_zero_is_one_step_td() {
        // With lambda=0, GAE reduces to one-step TD error
        let rewards = &[1.0, 1.0];
        let values = &[0.5, 0.5];
        let dones = &[false, false];
        let last_value = 0.5;
        let (advantages, _) = compute_gae(rewards, values, dones, last_value, 0.99, 0.0);
        // delta_1 = 1.0 + 0.99*0.5 - 0.5 = 0.995, advantage = delta (lambda=0)
        assert!((advantages[1] - 0.995).abs() < 1e-6);
    }

    #[test]
    fn gae_lambda_one_is_monte_carlo() {
        // With lambda=1, GAE reduces to discounted returns minus value baseline
        let rewards = &[1.0, 1.0, 1.0];
        let values = &[0.0, 0.0, 0.0];
        let dones = &[false, false, true];
        let (advantages, _) = compute_gae(rewards, values, dones, 0.0, 0.99, 1.0);
        // Monte Carlo return from step 0: 1 + 0.99 + 0.99^2 = 2.9701
        assert!((advantages[0] - 2.9701).abs() < 1e-3);
    }
}
```

### Rust unit tests — Batch Assembly

```rust
// crates/rlox-core/src/training/batch.rs

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rollout_buffer_collects_n_steps() {
        let mut rb = RolloutBuffer::new(64, 4, 1); // n_steps=64, obs_dim=4, act_dim=1
        for _ in 0..64 {
            rb.push_step(/* obs, action, reward, done, log_prob, value */);
        }
        assert!(rb.is_full());
        assert_eq!(rb.len(), 64);
    }

    #[test]
    fn rollout_buffer_assembles_batch_with_gae() {
        let mut rb = RolloutBuffer::new(64, 4, 1);
        // Fill with dummy data
        for i in 0..64 {
            rb.push_step(/* ... */);
        }
        let batch = rb.compute_and_assemble(/*last_value=*/0.0, /*gamma=*/0.99, /*lambda=*/0.95);
        assert_eq!(batch.observations.len(), 64 * 4); // flat
        assert_eq!(batch.advantages.len(), 64);
        assert_eq!(batch.returns.len(), 64);
        assert_eq!(batch.log_probs.len(), 64);
    }

    #[test]
    fn minibatch_iterator_yields_correct_count() {
        let batch = TrainingBatch { /* 256 transitions */ };
        let minibatches: Vec<_> = batch.minibatches(64).collect();
        assert_eq!(minibatches.len(), 4); // 256 / 64
    }

    #[test]
    fn minibatch_indices_shuffled_with_seed() {
        let batch = TrainingBatch { /* 256 transitions */ };
        let mb1: Vec<_> = batch.minibatches_seeded(64, 42).collect();
        let mb2: Vec<_> = batch.minibatches_seeded(64, 42).collect();
        // Same seed -> same order
        assert_eq!(mb1[0].indices, mb2[0].indices);
    }

    #[test]
    fn advantage_normalization() {
        let mut advantages = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        normalize_advantages(&mut advantages);
        let mean: f64 = advantages.iter().sum::<f64>() / advantages.len() as f64;
        assert!(mean.abs() < 1e-6); // mean ~ 0
        let var: f64 = advantages.iter().map(|a| a * a).sum::<f64>() / advantages.len() as f64;
        assert!((var - 1.0).abs() < 0.1); // variance ~ 1
    }
}
```

### Rust unit tests — KL Controller

```rust
// crates/rlox-core/src/training/kl.rs

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kl_controller_initial_coefficient() {
        let kl = KLController::new(0.01, 0.02); // init_coeff=0.01, target_kl=0.02
        assert_eq!(kl.coefficient(), 0.01);
    }

    #[test]
    fn kl_controller_increases_on_high_kl() {
        let mut kl = KLController::new(0.01, 0.02);
        kl.update(0.05); // measured KL >> target
        assert!(kl.coefficient() > 0.01); // coefficient should increase
    }

    #[test]
    fn kl_controller_decreases_on_low_kl() {
        let mut kl = KLController::new(0.01, 0.02);
        kl.update(0.005); // measured KL << target
        assert!(kl.coefficient() < 0.01); // coefficient should decrease
    }

    #[test]
    fn kl_controller_stays_near_target() {
        let mut kl = KLController::new(0.01, 0.02);
        kl.update(0.02); // measured KL == target
        // Coefficient should remain approximately the same
        assert!((kl.coefficient() - 0.01).abs() < 0.005);
    }

    #[test]
    fn kl_controller_has_floor() {
        let mut kl = KLController::new(0.01, 0.02);
        for _ in 0..100 {
            kl.update(0.0001); // very low KL
        }
        assert!(kl.coefficient() > 0.0); // never goes to zero
    }
}
```

### Rust unit tests — Checkpoint

```rust
// crates/rlox-core/src/checkpoint.rs

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn save_and_load_training_state() {
        let dir = TempDir::new().unwrap();
        let state = TrainingState {
            step: 1000,
            episodes: 50,
            kl_coeff: 0.015,
            rng_state: [42u8; 32], // ChaCha state
        };
        save_training_state(&state, dir.path().join("state.json")).unwrap();
        let loaded = load_training_state(dir.path().join("state.json")).unwrap();
        assert_eq!(loaded.step, 1000);
        assert_eq!(loaded.episodes, 50);
        assert!((loaded.kl_coeff - 0.015).abs() < 1e-10);
    }

    #[test]
    fn resume_produces_identical_rng_state() {
        let seed = 42u64;
        let mut rng1 = rng_from_seed(seed);
        // Advance RNG 100 steps
        for _ in 0..100 { let _: f64 = rng1.gen(); }
        // Save state
        let state_bytes = serialize_rng(&rng1);
        // Restore
        let rng2 = deserialize_rng(&state_bytes);
        // Both should produce same next value
        let v1: f64 = rng1.gen();
        let v2: f64 = rng2.gen();
        assert_eq!(v1, v2);
    }
}
```

### Python integration tests — PPO End-to-End

```python
# tests/python/test_ppo.py

import numpy as np
import pytest

def test_gae_computation():
    """GAE computed in Rust matches expected values."""
    from rlox import compute_gae
    rewards = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    values = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    dones = np.array([False, False, True])
    advantages, returns = compute_gae(
        rewards, values, dones, last_value=0.0, gamma=0.99, lam=0.95
    )
    assert advantages.shape == (3,)
    assert returns.shape == (3,)
    # Terminal step advantage = reward - value = 1.0
    np.testing.assert_almost_equal(advantages[2], 1.0, decimal=5)

def test_gae_matches_reference_implementation():
    """Rust GAE matches the CleanRL/SB3 reference implementation."""
    from rlox import compute_gae

    # Reference GAE (numpy, from CleanRL)
    def reference_gae(rewards, values, dones, last_value, gamma, lam):
        n = len(rewards)
        advantages = np.zeros(n)
        last_gae = 0.0
        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
                next_non_terminal = 1.0 - float(dones[t])
            else:
                next_value = values[t + 1]
                next_non_terminal = 1.0 - float(dones[t])
            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + gamma * lam * next_non_terminal * last_gae
        return advantages, advantages + values

    np.random.seed(42)
    n = 2048
    rewards = np.random.randn(n)
    values = np.random.randn(n)
    dones = np.random.random(n) > 0.95  # ~5% done rate
    last_value = np.random.randn()

    ref_adv, ref_ret = reference_gae(rewards, values, dones, last_value, 0.99, 0.95)
    rust_adv, rust_ret = compute_gae(rewards, values, dones, last_value, 0.99, 0.95)

    np.testing.assert_allclose(rust_adv, ref_adv, rtol=1e-6)
    np.testing.assert_allclose(rust_ret, ref_ret, rtol=1e-6)

def test_ppo_config_defaults():
    from rlox import PPOConfig
    config = PPOConfig()
    assert config.lr == 3e-4
    assert config.gamma == 0.99
    assert config.gae_lambda == 0.95
    assert config.clip_range == 0.2
    assert config.n_steps == 2048
    assert config.n_epochs == 10
    assert config.batch_size == 64
    assert config.ent_coef == 0.0
    assert config.vf_coef == 0.5
    assert config.max_grad_norm == 0.5

def test_ppo_config_custom():
    from rlox import PPOConfig
    config = PPOConfig(lr=1e-3, n_steps=128, clip_range=0.1)
    assert config.lr == 1e-3
    assert config.n_steps == 128
    assert config.clip_range == 0.1

def test_ppo_config_to_dict():
    from rlox import PPOConfig
    config = PPOConfig()
    d = config.to_dict()
    assert isinstance(d, dict)
    assert "lr" in d
    assert "gamma" in d

def test_ppo_trainer_creates():
    torch = pytest.importorskip("torch")
    from rlox import PPOTrainer, PPOConfig
    config = PPOConfig(n_steps=64)
    trainer = PPOTrainer(env="CartPole", config=config)
    assert trainer is not None

def test_ppo_cartpole_convergence():
    """PPO must solve CartPole-v1 (avg reward >= 475) within 50K steps."""
    torch = pytest.importorskip("torch")
    from rlox import PPOTrainer, PPOConfig
    config = PPOConfig(
        n_steps=2048,
        n_epochs=10,
        batch_size=64,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        total_timesteps=50_000,
        seed=42,
    )
    trainer = PPOTrainer(env="CartPole", config=config, num_envs=8)
    result = trainer.train()
    assert result["mean_reward"] >= 400  # relaxed threshold for test stability

def test_checkpoint_save_load():
    torch = pytest.importorskip("torch")
    import tempfile, os
    from rlox import PPOTrainer, PPOConfig
    config = PPOConfig(n_steps=64, total_timesteps=256, seed=42)
    trainer = PPOTrainer(env="CartPole", config=config, num_envs=2)
    trainer.train()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "checkpoint")
        trainer.save(path)
        assert os.path.exists(path + "/training_state.json")
        assert os.path.exists(path + "/model.pt")
        # Load and verify state
        trainer2 = PPOTrainer.load(path, env="CartPole", num_envs=2)
        assert trainer2.global_step == trainer.global_step
```

## Implementation Steps

### Step 1: GAE computation (`crates/rlox-core/src/training/gae.rs`)

```rust
/// Compute Generalized Advantage Estimation.
///
/// Iterates backwards over the rollout, computing:
///   delta_t = reward_t + gamma * V(t+1) * (1 - done_t) - V(t)
///   A_t = delta_t + gamma * lambda * (1 - done_t) * A(t+1)
///   return_t = A_t + V(t)
pub fn compute_gae(
    rewards: &[f64],
    values: &[f64],
    dones: &[bool],
    last_value: f64,
    gamma: f64,
    gae_lambda: f64,
) -> (Vec<f64>, Vec<f64>) { ... }
```

### Step 2: RolloutBuffer (`crates/rlox-core/src/training/batch.rs`)

Stores N steps of experience from M parallel envs. After collection, computes GAE and assembles training batches.

```rust
pub struct RolloutBuffer {
    obs: Vec<f32>,        // [n_steps * num_envs * obs_dim]
    actions: Vec<f32>,    // [n_steps * num_envs * act_dim]
    rewards: Vec<f64>,    // [n_steps * num_envs]
    dones: Vec<bool>,     // [n_steps * num_envs]
    log_probs: Vec<f64>,  // [n_steps * num_envs]
    values: Vec<f64>,     // [n_steps * num_envs]
    n_steps: usize,
    num_envs: usize,
    obs_dim: usize,
    act_dim: usize,
    pos: usize,
}

pub struct TrainingBatch {
    pub observations: Vec<f32>,
    pub actions: Vec<f32>,
    pub advantages: Vec<f64>,
    pub returns: Vec<f64>,
    pub log_probs: Vec<f64>,
}
```

### Step 3: KL Controller (`crates/rlox-core/src/training/kl.rs`)

Adaptive KL penalty coefficient (Ziegler et al. 2019):
- If measured KL > 1.5 * target: multiply coefficient by 2
- If measured KL < target / 1.5: divide coefficient by 2

### Step 4: PPO Config (`python/rlox/config.py`)

Python dataclass with all PPO hyperparameters and sensible defaults.

### Step 5: PPO Trainer (`python/rlox/trainer.py`)

The main training loop in Python:
1. Create VecEnv (Rust) with N parallel CartPoles
2. Create nn.Module policy/value network
3. Loop: collect n_steps -> push to RolloutBuffer (Rust) -> compute GAE (Rust) -> assemble batch (Rust) -> for each epoch: for each minibatch: forward pass (PyTorch) -> PPO loss (PyTorch) -> backward (PyTorch) -> optimizer step (PyTorch)

### Step 6: Checkpoint (`crates/rlox-core/src/checkpoint.rs`)

Serialize training state (step counter, KL coeff, RNG state) via serde_json. Model weights saved separately by PyTorch.

### Step 7: PyO3 bindings for training ops

`compute_gae()` as a Python function, `RolloutBuffer` as a Python class.

## New Files to Create

| File | Purpose |
|------|---------|
| `crates/rlox-core/src/training/mod.rs` | Module exports |
| `crates/rlox-core/src/training/gae.rs` | GAE / advantage computation |
| `crates/rlox-core/src/training/batch.rs` | RolloutBuffer, TrainingBatch, minibatch iterator |
| `crates/rlox-core/src/training/kl.rs` | KL penalty controller |
| `crates/rlox-core/src/checkpoint.rs` | Training state serialization |
| `crates/rlox-python/src/training.rs` | PyO3 bindings for GAE, RolloutBuffer |
| `python/rlox/config.py` | PPOConfig dataclass |
| `python/rlox/trainer.py` | PPOTrainer class |
| `python/rlox/models.py` | Default MLP policy/value network |
| `tests/python/test_ppo.py` | PPO integration + convergence tests |
| `crates/rlox-bench/benches/gae_compute.rs` | GAE benchmark vs numpy baseline |

## Files to Modify

| File | Change |
|------|--------|
| `crates/rlox-core/src/lib.rs` | Add `pub mod training; pub mod checkpoint;` |
| `crates/rlox-python/src/lib.rs` | Register training Python classes/functions |
| `python/rlox/__init__.py` | Re-export PPOTrainer, PPOConfig, compute_gae |
| `python/rlox/_rlox_core.pyi` | Add training type stubs |

## Acceptance Criteria

- [ ] GAE computation matches CleanRL reference to within 1e-6 relative tolerance
- [ ] GAE benchmark shows 5-10x speedup over equivalent numpy implementation
- [ ] RolloutBuffer collects N steps and assembles batches correctly
- [ ] KL controller adapts coefficient based on measured KL divergence
- [ ] PPOConfig has sensible defaults and is composable
- [ ] PPOTrainer solves CartPole-v1 (avg reward >= 400) within 50K steps
- [ ] Checkpoint save/load preserves training state
- [ ] Resume from checkpoint produces identical trajectory to uninterrupted run
- [ ] `PPOTrainer(env="CartPole").train(50_000)` works as a one-liner
- [ ] All Rust tests pass
- [ ] All Python integration tests pass
