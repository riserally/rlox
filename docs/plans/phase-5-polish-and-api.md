# Phase 5 — Polish, API & Production Readiness

**Status: NOT STARTED**
**Duration: Weeks 10-12**
**PRD Features: 7.1, 7.2, 7.5, 7.6, 7.9, 8.2, 10.1, 10.2, 10.3**
**Depends on: Phase 3 (PPO working), Phase 4 (DPO/GRPO working)**
**Can partially overlap with Phase 4**

---

## Objective

Transform the working internals into a polished, researcher-friendly framework. If the Python API is worse than SB3's one-liner, adoption fails regardless of Rust performance. This phase focuses on: ergonomic API, comprehensive error messages, callbacks/hooks, logging integration, experiment tracking, and correctness/property-based testing.

## PRD Feature Mapping

| PRD # | Feature | Priority | Target |
|-------|---------|----------|--------|
| 7.1 | "One-liner" training API | P0 | `PPOTrainer(env="HalfCheetah-v4").train(1M)` |
| 7.2 | Composable config system (polish) | P0 | YAML/CLI merge, full type hints |
| 7.5 | Callback / hook system | P1 | on_step, on_episode_end, on_train_batch |
| 7.6 | Rich logging integration | P1 | W&B, MLflow, TensorBoard native support |
| 7.9 | Comprehensive error messages | P0 | No raw Rust backtraces in Python |
| 8.2 | Experiment snapshot | P1 | Git hash, config, deps, hardware |
| 10.1 | Algorithm correctness tests | P0 | PPO solves CartPole, matches CleanRL curves |
| 10.2 | Property-based buffer tests | P1 | proptest for buffer invariants |
| 10.3 | Benchmark suite | P1 | Automated comparison vs SB3, CleanRL, TorchRL |

## Reasoning

### Why callbacks over subclassing?

SB3 uses callbacks; RLlib uses subclassing. Callbacks are simpler:
- No need to understand the class hierarchy to customize behavior
- Multiple callbacks compose via a list (no diamond inheritance)
- The Rust orchestrator invokes Python callbacks at well-defined points via PyO3
- Callbacks receive **batched payloads** (not per-step) to minimize Python-Rust boundary crossings

### Why property-based testing for buffers?

Replay buffers have subtle invariants: FIFO eviction order, no data loss, correct priority ordering, thread-safety under concurrent read/write. Unit tests check known scenarios; property-based tests (via `proptest`) generate thousands of random operation sequences and verify invariants hold universally. This is how Crossbeam and TiKV test their concurrent data structures.

### Why benchmark against SB3/CleanRL continuously?

The core value proposition is "faster than existing frameworks." If a refactor regresses performance, the framework loses its reason to exist. Automated benchmarks on every release (not just once at launch) keep the claim honest.

### Why experiment snapshots matter for RL?

RL's reproducibility crisis (50-70% of claimed improvements may be spurious per Agarwal et al.) means every experiment must be fully reconstructable. Recording git hash, full config, dependency versions, hardware info, and seed alongside checkpoints makes this automatic — researchers shouldn't need to manually track this.

## TDD Test Specifications

### Rust tests — Property-Based Buffer Tests

```rust
// crates/rlox-core/src/buffer/mod.rs (proptest)

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn ring_buffer_never_exceeds_capacity(
            capacity in 1usize..1000,
            num_pushes in 0usize..5000
        ) {
            let mut buf = RingBuffer::new(capacity, ExperienceSchema::new(4, 1));
            for _ in 0..num_pushes {
                buf.push(sample_transition(4)).unwrap();
            }
            prop_assert!(buf.len() <= capacity);
        }

        #[test]
        fn ring_buffer_len_is_min_of_pushes_and_capacity(
            capacity in 1usize..1000,
            num_pushes in 0usize..5000
        ) {
            let mut buf = RingBuffer::new(capacity, ExperienceSchema::new(4, 1));
            for _ in 0..num_pushes {
                buf.push(sample_transition(4)).unwrap();
            }
            prop_assert_eq!(buf.len(), num_pushes.min(capacity));
        }

        #[test]
        fn sample_never_returns_more_than_buffer_len(
            capacity in 10usize..1000,
            num_pushes in 10usize..5000,
            batch_size in 1usize..100
        ) {
            let mut buf = RingBuffer::new(capacity, ExperienceSchema::new(4, 1));
            for _ in 0..num_pushes {
                buf.push(sample_transition(4)).unwrap();
            }
            let actual_batch = batch_size.min(buf.len());
            let result = buf.sample(actual_batch, &mut rng);
            prop_assert!(result.is_ok());
            prop_assert_eq!(result.unwrap().len(), actual_batch);
        }

        #[test]
        fn varlen_store_total_equals_sum_of_lengths(
            sequences in prop::collection::vec(
                prop::collection::vec(0u32..1000, 1..100),
                1..50
            )
        ) {
            let mut store = VarLenStore::new();
            let mut expected_total = 0;
            for seq in &sequences {
                store.push(seq);
                expected_total += seq.len();
            }
            prop_assert_eq!(store.total_elements(), expected_total);
            prop_assert_eq!(store.num_sequences(), sequences.len());
        }

        #[test]
        fn varlen_store_roundtrip(
            sequences in prop::collection::vec(
                prop::collection::vec(0u32..1000, 1..100),
                1..50
            )
        ) {
            let mut store = VarLenStore::new();
            for seq in &sequences {
                store.push(seq);
            }
            for (i, seq) in sequences.iter().enumerate() {
                prop_assert_eq!(store.get(i), seq.as_slice());
            }
        }
    }
}
```

### Rust tests — Error Message Quality

```rust
// crates/rlox-python/src/error.rs

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_action_error_message_is_descriptive() {
        let err = RloxError::InvalidAction(
            "Expected Discrete action in range [0, 2), got Discrete(5)".to_string()
        );
        let msg = format!("{}", err);
        assert!(msg.contains("Expected"));
        assert!(msg.contains("Discrete"));
        assert!(msg.contains("5"));
        assert!(!msg.contains("panic")); // no raw panic info
    }

    #[test]
    fn shape_mismatch_error_is_descriptive() {
        let err = RloxError::ShapeMismatch {
            expected: "(4,)".to_string(),
            got: "(3,)".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("(4,)"));
        assert!(msg.contains("(3,)"));
    }
}
```

### Python tests — Callback System

```python
# tests/python/test_callbacks.py

import pytest

def test_callback_on_step():
    torch = pytest.importorskip("torch")
    from rlox import PPOTrainer, PPOConfig, Callback

    class StepCounter(Callback):
        def __init__(self):
            self.count = 0
        def on_step(self, info):
            self.count += 1

    counter = StepCounter()
    config = PPOConfig(n_steps=64, total_timesteps=256, seed=42)
    trainer = PPOTrainer(env="CartPole", config=config, num_envs=2, callbacks=[counter])
    trainer.train()
    assert counter.count > 0

def test_callback_on_episode_end():
    torch = pytest.importorskip("torch")
    from rlox import PPOTrainer, PPOConfig, Callback

    class EpisodeTracker(Callback):
        def __init__(self):
            self.returns = []
        def on_episode_end(self, info):
            self.returns.append(info["episode_return"])

    tracker = EpisodeTracker()
    config = PPOConfig(n_steps=64, total_timesteps=512, seed=42)
    trainer = PPOTrainer(env="CartPole", config=config, num_envs=4, callbacks=[tracker])
    trainer.train()
    assert len(tracker.returns) > 0
    assert all(isinstance(r, float) for r in tracker.returns)

def test_callback_on_train_batch():
    torch = pytest.importorskip("torch")
    from rlox import PPOTrainer, PPOConfig, Callback

    class LossTracker(Callback):
        def __init__(self):
            self.losses = []
        def on_train_batch(self, info):
            self.losses.append(info["loss"])

    tracker = LossTracker()
    config = PPOConfig(n_steps=64, total_timesteps=256, seed=42)
    trainer = PPOTrainer(env="CartPole", config=config, num_envs=2, callbacks=[tracker])
    trainer.train()
    assert len(tracker.losses) > 0

def test_multiple_callbacks_compose():
    torch = pytest.importorskip("torch")
    from rlox import PPOTrainer, PPOConfig, Callback

    class A(Callback):
        def __init__(self): self.called = False
        def on_step(self, info): self.called = True

    class B(Callback):
        def __init__(self): self.called = False
        def on_step(self, info): self.called = True

    a, b = A(), B()
    config = PPOConfig(n_steps=64, total_timesteps=128, seed=42)
    trainer = PPOTrainer(env="CartPole", config=config, num_envs=2, callbacks=[a, b])
    trainer.train()
    assert a.called and b.called
```

### Python tests — Logging Integration

```python
# tests/python/test_logging.py

import pytest
import tempfile, json, os

def test_wandb_logger_interface():
    """W&B logger has the expected interface (doesn't require actual W&B)."""
    from rlox import WandbLogger
    logger = WandbLogger(project="test", run_name="test_run", enabled=False)
    logger.log({"reward": 100.0, "loss": 0.5}, step=1)
    # Should not raise even when disabled

def test_tensorboard_logger_writes():
    from rlox import TensorBoardLogger
    with tempfile.TemporaryDirectory() as d:
        logger = TensorBoardLogger(log_dir=d)
        logger.log({"reward": 100.0}, step=1)
        logger.close()
        # Verify event file exists
        files = os.listdir(d)
        assert any("events" in f for f in files)

def test_experiment_snapshot():
    from rlox import PPOConfig, create_experiment_snapshot
    config = PPOConfig(seed=42)
    snapshot = create_experiment_snapshot(config)
    assert "config" in snapshot
    assert "python_version" in snapshot
    assert "rlox_version" in snapshot
    assert "timestamp" in snapshot
    # Should be JSON-serializable
    json.dumps(snapshot)
```

### Python tests — Config System Polish

```python
# tests/python/test_config.py

import pytest

def test_config_from_yaml():
    import tempfile, os
    from rlox import PPOConfig
    yaml_content = """
lr: 1e-3
n_steps: 128
gamma: 0.999
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        f.flush()
        config = PPOConfig.from_yaml(f.name)
    os.unlink(f.name)
    assert config.lr == 1e-3
    assert config.n_steps == 128
    assert config.gamma == 0.999
    # Other fields should have defaults
    assert config.clip_range == 0.2

def test_config_merge():
    from rlox import PPOConfig
    base = PPOConfig(lr=3e-4, n_steps=2048)
    overrides = {"lr": 1e-3, "seed": 123}
    merged = base.merge(overrides)
    assert merged.lr == 1e-3
    assert merged.n_steps == 2048  # kept from base
    assert merged.seed == 123

def test_config_to_yaml_roundtrip():
    import tempfile, os
    from rlox import PPOConfig
    config = PPOConfig(lr=1e-3, n_steps=128, seed=42)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        config.to_yaml(f.name)
    loaded = PPOConfig.from_yaml(f.name)
    os.unlink(f.name)
    assert loaded.lr == config.lr
    assert loaded.n_steps == config.n_steps
    assert loaded.seed == config.seed

def test_config_validates_types():
    from rlox import PPOConfig
    with pytest.raises((TypeError, ValueError)):
        PPOConfig(lr="not_a_number")

def test_config_rejects_invalid_values():
    from rlox import PPOConfig
    with pytest.raises(ValueError):
        PPOConfig(gamma=1.5)  # must be in [0, 1]
    with pytest.raises(ValueError):
        PPOConfig(clip_range=-0.1)  # must be positive
```

### Python tests — Algorithm Correctness

```python
# tests/python/test_correctness.py

import numpy as np
import pytest

@pytest.mark.slow
def test_ppo_cartpole_solves():
    """PPO solves CartPole-v1 (mean reward >= 475) within 50K steps.
    This is the canonical correctness test from the PRD (feature 10.1)."""
    torch = pytest.importorskip("torch")
    from rlox import PPOTrainer, PPOConfig
    config = PPOConfig(
        total_timesteps=50_000,
        n_steps=2048,
        n_epochs=10,
        batch_size=64,
        lr=3e-4,
        seed=42,
    )
    trainer = PPOTrainer(env="CartPole", config=config, num_envs=8)
    result = trainer.train()
    assert result["mean_reward"] >= 475, (
        f"PPO failed to solve CartPole: mean_reward={result['mean_reward']:.1f} < 475"
    )

@pytest.mark.slow
def test_ppo_cartpole_multiple_seeds():
    """PPO solves CartPole across 3 different seeds (robustness check)."""
    torch = pytest.importorskip("torch")
    from rlox import PPOTrainer, PPOConfig
    for seed in [42, 123, 999]:
        config = PPOConfig(total_timesteps=50_000, seed=seed)
        trainer = PPOTrainer(env="CartPole", config=config, num_envs=8)
        result = trainer.train()
        assert result["mean_reward"] >= 400, (
            f"PPO failed with seed={seed}: {result['mean_reward']:.1f}"
        )

@pytest.mark.slow
def test_ppo_deterministic_training():
    """Two PPO runs with same seed produce identical results."""
    torch = pytest.importorskip("torch")
    from rlox import PPOTrainer, PPOConfig
    config = PPOConfig(total_timesteps=5_000, seed=42)

    trainer1 = PPOTrainer(env="CartPole", config=config, num_envs=4)
    result1 = trainer1.train()

    trainer2 = PPOTrainer(env="CartPole", config=config, num_envs=4)
    result2 = trainer2.train()

    assert result1["mean_reward"] == result2["mean_reward"]
    assert result1["global_step"] == result2["global_step"]
```

### Benchmark Specifications

```python
# benchmarks/bench_vs_sb3.py (not pytest — standalone benchmark script)

"""
Benchmark rlox vs SB3 on standardized tasks.
Outputs: JSON with fps, wall_time, peak_memory for each framework.
"""

def bench_cartpole_stepping():
    """Environment stepping throughput: rlox VecEnv vs SB3 DummyVecEnv/SubprocVecEnv."""
    # rlox: step 128 CartPoles for 10K steps
    # SB3: step 128 CartPoles for 10K steps
    # Compare: steps/second

def bench_gae_computation():
    """GAE computation: rlox Rust vs SB3 numpy vs CleanRL numpy."""
    # Generate 2048*128 transitions of random data
    # Time GAE computation for each

def bench_ppo_training():
    """Full PPO training wall-clock: rlox vs SB3 vs CleanRL on CartPole 100K steps."""
    # Compare: total wall time, final mean reward, peak GPU memory
```

## Implementation Steps

### Step 1: Callback base class (`python/rlox/callbacks.py`)

```python
class Callback:
    """Base class for training callbacks."""
    def on_step(self, info: dict) -> None: ...
    def on_episode_end(self, info: dict) -> None: ...
    def on_train_batch(self, info: dict) -> None: ...
    def on_eval(self, info: dict) -> None: ...
    def on_training_start(self, info: dict) -> None: ...
    def on_training_end(self, info: dict) -> None: ...
```

Integrate into PPOTrainer: at each hook point, iterate over callbacks and call the relevant method. Callbacks receive dicts with payloads (batched, not per-step).

### Step 2: Logging integration (`python/rlox/logging.py`)

Implement as specialized callbacks:
- `WandbLogger(Callback)` — calls `wandb.log()` on `on_train_batch`
- `TensorBoardLogger(Callback)` — writes to `SummaryWriter`
- `MLflowLogger(Callback)` — calls `mlflow.log_metrics()`

### Step 3: Config system polish (`python/rlox/config.py`)

Add to existing PPOConfig:
- `from_yaml(path)` classmethod — load from YAML file
- `to_yaml(path)` — serialize to YAML
- `merge(overrides: dict)` — create new config with overrides applied
- Validation in `__post_init__`: ranges, types

### Step 4: Error message improvement

Audit all `RloxError` variants. Ensure every error includes:
- What was expected
- What was received
- Where it happened (env name, step number if available)

In `rlox-python/src/error.rs`, convert all Rust errors to specific Python exception types:
- `RloxError::InvalidAction` → `ValueError`
- `RloxError::ShapeMismatch` → `ValueError`
- `RloxError::EnvError` → `RuntimeError`

### Step 5: Experiment snapshot (`python/rlox/experiment.py`)

```python
def create_experiment_snapshot(config) -> dict:
    return {
        "config": config.to_dict(),
        "rlox_version": rlox.__version__,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "timestamp": datetime.utcnow().isoformat(),
        "git_hash": _get_git_hash(),  # if in a git repo
        "hardware": {
            "cpu": platform.processor(),
            "gpu": _get_gpu_info(),
        },
    }
```

Automatically saved alongside checkpoints.

### Step 6: Property-based tests (`crates/rlox-core/`)

Add `proptest` to dev-dependencies. Write property tests for:
- RingBuffer: capacity invariant, FIFO order, no data loss
- VarLenStore: roundtrip, total elements
- ExperienceTable: column alignment, sample size

### Step 7: Algorithm correctness tests

- PPO CartPole convergence (3 seeds)
- PPO deterministic training (same seed = same result)
- GAE matches reference implementation

### Step 8: Benchmark suite

Automated comparison scripts in `benchmarks/`:
- Env stepping throughput
- GAE computation speed
- Full PPO training wall-clock

## New Files to Create

| File | Purpose |
|------|---------|
| `python/rlox/callbacks.py` | Callback base class and built-in callbacks |
| `python/rlox/logging.py` | W&B, TensorBoard, MLflow loggers |
| `python/rlox/experiment.py` | Experiment snapshot creation |
| `tests/python/test_callbacks.py` | Callback system tests |
| `tests/python/test_logging.py` | Logger integration tests |
| `tests/python/test_config.py` | Config YAML/merge/validation tests |
| `tests/python/test_correctness.py` | Algorithm correctness (marked slow) |
| `benchmarks/bench_vs_sb3.py` | Comparison benchmarks |
| `benchmarks/bench_gae.py` | GAE-specific benchmark |

## Files to Modify

| File | Change |
|------|--------|
| `python/rlox/config.py` | Add YAML, merge, validation |
| `python/rlox/trainer.py` | Integrate callbacks, logging, snapshots |
| `python/rlox/__init__.py` | Re-export Callback, loggers |
| `crates/rlox-core/Cargo.toml` | Add `proptest` to dev-dependencies |
| `crates/rlox-python/src/error.rs` | Improve error conversion |

## Acceptance Criteria

- [ ] Callback system works: on_step, on_episode_end, on_train_batch
- [ ] Multiple callbacks compose correctly
- [ ] W&B logger (disabled mode) doesn't raise
- [ ] TensorBoard logger writes event files
- [ ] Config loads from YAML, merges overrides, validates ranges
- [ ] Error messages include expected vs. received, no raw Rust backtraces
- [ ] Experiment snapshot captures version, config, hardware, git hash
- [ ] Property tests pass for RingBuffer, VarLenStore (1000+ random sequences)
- [ ] PPO solves CartPole (>= 475 reward) within 50K steps across 3 seeds
- [ ] PPO training is deterministic (same seed = same result)
- [ ] Benchmark suite runs and produces comparison JSON
