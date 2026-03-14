# Phase 7 — Algorithm Completeness (v0.5)

**Status: NOT STARTED**
**Target: Q3 2026**
**PRD Features: A1–A3, A14, T5, T7, T10, R3, API1–API6, API10**
**Depends on: Phases 0–6 (all complete)**

---

## Objective

Transform rlox from a collection of high-performance primitives into a complete training framework. Deliver end-to-end PPO for simulation RL and end-to-end GRPO for LLM post-training. After this phase, a researcher can `pip install rlox` and train a policy without writing buffer/GAE/loss code.

## Critical Pre-Requisites (Bugs to Fix First)

These issues were identified in the architecture review and **block correct algorithm implementations**:

| Issue | File | Problem | Fix |
|-------|------|---------|-----|
| Scalar action | `rlox-core/src/buffer/mod.rs:10` | `ExperienceRecord.action` is `f32` — cannot represent multi-dim actions (HalfCheetah=6 dims) | Change to `Vec<f32>` or columnar `ndarray` |
| Terminal obs lost | `rlox-core/src/env/parallel.rs:56` | Auto-reset discards terminal observation; PPO needs it for value bootstrapping at truncation | Return `(terminal_obs, reset_obs)` tuple |
| Panic in prod | `rlox-core/src/llm/ops.rs:29` | `assert_eq!` in `compute_token_kl` will panic | Convert to `Result<_, RloxError>` |
| Hardcoded env | `rlox-python/src/env.rs:73-83` | `PyVecEnv::new()` only creates CartPole | Accept env factory or Gymnasium env string |

## Architectural Decisions (Locked In)

From the architecture review — these decisions apply to all code written in this phase:

1. **Layer 1+ lives in Python, not Rust.** Training loops, collectors, loss functions are Python classes that call Rust primitives. Researchers must be able to read and modify them.
2. **Configs are Python dataclasses.** Never cross PyO3. Rust functions accept individual parameters.
3. **Policy/model stays in Python.** Rust never holds `nn.Module` references. Collector calls Python for inference, hands data to Rust for storage/computation.

## PRD Feature Mapping

| PRD ID | Feature | Priority | Description |
|--------|---------|----------|-------------|
| A1 | PPO (clip + KL) | P0 | Full PPO with all 37 implementation details (Huang et al. 2022) |
| A2 | DPO / IPO / KTO | P0 | DPOPair storage done → add loss computation + training loop |
| A3 | GRPO | P0 | Advantage normalization done → add generation loop, clipping, loss |
| A14 | A2C | P1 | Simpler PPO variant, useful baseline |
| T5 | Batch assembly | P1 | Collate transitions → training-ready zero-copy torch tensors |
| T7 | Reward normalization | P1 | Running mean/variance (Welford's), per-token + per-sequence |
| T10 | Sequence packing | P1 | Bin-packing variable-length sequences for GPU batch efficiency |
| R3 | Checkpoint/resume | P0 | Save/restore: weights, optimizer, buffer, RNG, step counter |
| API6 | W&B / TensorBoard | P1 | Rust collects metrics → flush to Python loggers |
| API10 | Prebuilt wheels | P0 | maturin wheels for Linux x86_64/ARM64, macOS x86_64/ARM64 |

## Implementation Order

### Step 0: Fix Blocking Issues (Week 1)

Fix the four pre-requisite issues before any algorithm work.

#### 0.1 Multi-dimensional actions

```rust
// crates/rlox-core/src/buffer/mod.rs
// BEFORE:
pub struct ExperienceRecord {
    pub obs: Vec<f32>,
    pub action: f32,  // ← scalar, blocks everything
    pub reward: f32,
    ...
}

// AFTER: columnar storage with configurable action_dim
pub struct ExperienceTable {
    observations: Vec<f32>,  // flat: [obs_dim * capacity]
    actions: Vec<f32>,       // flat: [act_dim * capacity]
    rewards: Vec<f32>,
    ...
    obs_dim: usize,
    act_dim: usize,
}
```

Update all buffer push/sample paths. Update Python bindings. Update all tests.

#### 0.2 Terminal observation preservation

```rust
// crates/rlox-core/src/env/parallel.rs
pub struct StepResult {
    pub obs: Vec<f32>,           // current observation (post-reset if done)
    pub reward: f32,
    pub terminated: bool,
    pub truncated: bool,
    pub terminal_obs: Option<Vec<f32>>,  // observation at termination (before reset)
}
```

PPO needs `terminal_obs` for value bootstrapping: `V(s_terminal)` is used when `truncated=True`.

#### 0.3 Replace assert with Result

```rust
// crates/rlox-core/src/llm/ops.rs
// BEFORE:
assert_eq!(log_probs_p.len(), log_probs_q.len());

// AFTER:
if log_probs_p.len() != log_probs_q.len() {
    return Err(RloxError::ShapeMismatch {
        expected: log_probs_p.len(),
        got: log_probs_q.len(),
        context: "compute_token_kl: log_probs_p and log_probs_q must have same length",
    });
}
```

#### 0.4 Configurable VecEnv

```rust
// crates/rlox-python/src/env.rs
// Accept an env factory function or gymnasium env string
#[pymethods]
impl PyVecEnv {
    #[new]
    #[pyo3(signature = (env_id, n_envs, seed=None))]
    fn new(env_id: &str, n_envs: usize, seed: Option<u64>) -> PyResult<Self> { ... }
}
```

### Step 1: BatchSteppable Trait + PythonBatchStepper (Week 2)

The key extensibility pattern from the architecture review. Enables any Gymnasium environment to work with rlox.

```rust
// crates/rlox-core/src/env/mod.rs

/// Trait for stepping a batch of environments.
/// Separates parallelism strategy from step logic.
pub trait BatchSteppable: Send {
    fn step_batch(&mut self, actions: &[Action]) -> Result<Vec<StepResult>, RloxError>;
    fn reset_batch(&mut self, seeds: Option<&[u64]>) -> Result<Vec<Observation>, RloxError>;
    fn num_envs(&self) -> usize;
    fn obs_dim(&self) -> usize;
    fn act_dim(&self) -> usize;
}

/// Rayon-parallel stepping for Rust-native envs (CartPole, future MuJoCo)
pub struct ParallelBatchStepper<E: RLEnv> { ... }

/// Sequential stepping for Python Gymnasium envs (one GIL acquire per batch)
pub struct PythonBatchStepper { ... }
```

### Step 2: Python Layer 1 Components (Weeks 3–4)

These live entirely in Python (`python/rlox/`), calling Rust primitives.

#### 2.1 RolloutCollector

```python
# python/rlox/collectors.py
class RolloutCollector:
    """Collects rollout data from vectorized environments."""

    def __init__(self, env: str | gymnasium.Env, n_envs: int = 1, device: str = "cpu"):
        self._vec_env = rlox.VecEnv(env, n_envs)  # Rust-backed
        self._buffer = rlox.ExperienceTable(...)    # Rust-backed

    def collect(self, policy: nn.Module, n_steps: int) -> RolloutBatch:
        """Collect n_steps per env. Returns batch with obs, actions, rewards, advantages."""
        # Python loop: step env → policy forward → store in Rust buffer
        # GAE computed in Rust at end of rollout
        ...
```

#### 2.2 PPOLoss

```python
# python/rlox/losses.py
class PPOLoss:
    """PPO clipped surrogate loss with value loss and entropy bonus."""

    def __init__(self, clip_eps=0.2, vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5):
        ...

    def __call__(self, batch: RolloutBatch, policy: nn.Module) -> torch.Tensor:
        # Standard PPO loss computation in PyTorch
        ...
```

#### 2.3 Batch assembly (T5)

```python
# python/rlox/batch.py
class RolloutBatch:
    """Training batch with zero-copy torch tensors from Rust buffers."""
    obs: torch.Tensor       # from rlox buffer, zero-copy
    actions: torch.Tensor
    advantages: torch.Tensor  # computed by rlox.compute_gae()
    returns: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
```

### Step 3: PPO End-to-End (Weeks 5–7)

Complete PPO implementation following Huang et al. 2022 "37 implementation details":

```python
# python/rlox/algorithms/ppo.py
class PPO:
    """PPO with all standard implementation details."""

    def __init__(self, env: str, policy: nn.Module, config: PPOConfig):
        self.collector = RolloutCollector(env, config.n_envs)
        self.loss_fn = PPOLoss(config.clip_eps, config.vf_coef, config.ent_coef)
        ...

    def train(self, total_timesteps: int):
        for iteration in range(num_iterations):
            # 1. Collect rollout (Rust env stepping + buffer)
            batch = self.collector.collect(self.policy, self.config.n_steps)

            # 2. Compute advantages (Rust GAE — 140x faster)
            advs, rets = rlox.compute_gae(batch.rewards, batch.values, ...)

            # 3. PPO update epochs (PyTorch)
            for epoch in range(self.config.n_epochs):
                for mb in batch.sample_minibatches(self.config.batch_size):
                    loss = self.loss_fn(mb, self.policy)
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
```

Key PPO details to implement (from Huang et al.):
- Advantage normalization (per-minibatch)
- Value function clipping
- Learning rate annealing (linear decay)
- Observation normalization (running mean/var)
- Reward normalization (running mean/var) — Rust Welford's algorithm (T7)
- Orthogonal weight initialization
- Gradient clipping by global norm

#### Correctness validation:
- CartPole-v1: solve (reward ≥ 475) in < 50K timesteps, 5/5 seeds
- HalfCheetah-v4: > 5000 return in 1M steps, within 1 std of SB3 baseline

### Step 4: Reward Normalization in Rust (Week 5, parallel)

```rust
// crates/rlox-core/src/training/normalization.rs

/// Welford's online algorithm for running mean/variance.
/// Thread-safe, O(1) per update, numerically stable.
pub struct RunningStats {
    count: u64,
    mean: f64,
    m2: f64,    // sum of squared deviations
}

impl RunningStats {
    pub fn update(&mut self, value: f64) { ... }
    pub fn normalize(&self, value: f64) -> f64 { (value - self.mean) / self.std().max(1e-8) }
    pub fn mean(&self) -> f64 { self.mean }
    pub fn var(&self) -> f64 { self.m2 / self.count.max(1) as f64 }
    pub fn std(&self) -> f64 { self.var().sqrt() }
}
```

Expose via PyO3 as `rlox.RunningStats`. Used for reward normalization and observation normalization.

### Step 5: GRPO End-to-End (Weeks 6–8)

Build on existing `compute_group_advantages` and `VarLenStore`:

```python
# python/rlox/algorithms/grpo.py
class GRPO:
    """Group Relative Policy Optimization for LLM post-training."""

    def __init__(self, model, reward_fn, config: GRPOConfig):
        self.store = rlox.VarLenStore(...)  # Rust variable-length storage
        ...

    def train(self, dataset):
        for batch in dataset:
            # 1. Generate G completions per prompt (Python — calls vLLM/model.generate)
            completions = self._generate(batch.prompts, self.config.group_size)

            # 2. Score with reward function (Python)
            rewards = self.reward_fn(completions)

            # 3. Compute group advantages (Rust — 35x faster)
            advantages = rlox.compute_group_advantages(rewards, self.config.group_size)

            # 4. Compute token KL (Rust — 4x faster)
            kl = rlox.compute_token_kl(log_probs, ref_log_probs)

            # 5. GRPO loss (PyTorch)
            loss = self._grpo_loss(log_probs, old_log_probs, advantages, kl)
            loss.backward()
            self.optimizer.step()
```

### Step 6: DPO End-to-End (Week 8)

Build on existing `DPOPair` storage:

```python
# python/rlox/algorithms/dpo.py
class DPO:
    """Direct Preference Optimization."""

    def __init__(self, model, ref_model, config: DPOConfig):
        ...

    def train(self, preference_dataset):
        for chosen, rejected in preference_dataset:
            # DPO loss: -log(sigmoid(beta * (log_ratio_chosen - log_ratio_rejected)))
            ...
```

### Step 7: Sequence Packing (Week 7, parallel)

```rust
// crates/rlox-core/src/training/packing.rs

/// First-fit-decreasing bin packing for variable-length sequences.
/// Generates: packed_input_ids, attention_mask, position_ids, sequence_boundaries.
pub fn pack_sequences(
    sequences: &[&[u32]],      // token IDs per sequence
    max_length: usize,          // max packed length (GPU batch dim)
) -> Result<Vec<PackedBatch>, RloxError> { ... }

pub struct PackedBatch {
    pub input_ids: Vec<u32>,
    pub attention_mask: Vec<u8>,
    pub position_ids: Vec<u32>,
    pub sequence_starts: Vec<usize>,  // boundaries for unpacking
}
```

30–60% GPU utilization improvement over naive padding.

### Step 8: A2C (Week 8, parallel)

Simpler than PPO (no clipping, no epochs). Useful as baseline:

```python
# python/rlox/algorithms/a2c.py
class A2C:
    """Advantage Actor-Critic. Single update per rollout, no clipping."""
    ...
```

### Step 9: Checkpoint / Resume (Week 9)

```python
# python/rlox/checkpoint.py
class Checkpoint:
    """Save/restore full training state for bit-exact resume."""

    @staticmethod
    def save(path: str, *, model, optimizer, buffer, rng_state, step, config):
        # model weights + optimizer state: torch.save
        # buffer contents: rlox buffer serialization (Rust)
        # RNG state: ChaCha8 state from Rust
        # config + metadata: JSON
        ...

    @staticmethod
    def load(path: str) -> dict:
        ...
```

### Step 10: Logging Integration (Week 9, parallel)

```python
# python/rlox/logging.py
class LoggerCallback:
    """Base callback for training metrics logging."""
    def on_train_step(self, metrics: dict): ...
    def on_rollout_end(self, metrics: dict): ...
    def on_eval(self, metrics: dict): ...

class WandbLogger(LoggerCallback): ...
class TensorBoardLogger(LoggerCallback): ...
```

### Step 11: Prebuilt Wheels (Week 10)

```yaml
# .github/workflows/wheels.yml
# Build matrix: linux-x86_64, linux-aarch64, macos-x86_64, macos-arm64
# Tool: maturin build --release --strip
# ABI: abi3-py310
```

## TDD Test Specifications

### Rust Tests

```rust
// Fix verification tests
#[test]
fn test_multi_dim_action_buffer() {
    let mut buf = ExperienceTable::new(100, 4, 6);  // obs=4, act=6
    buf.push(&obs, &action_6d, reward, done);
    let sample = buf.sample(1);
    assert_eq!(sample.actions[0].len(), 6);
}

#[test]
fn test_terminal_obs_preserved() {
    let mut env = VecEnv::new(CartPole, 4);
    let result = env.step_all(&actions);
    for r in &result {
        if r.truncated {
            assert!(r.terminal_obs.is_some());
        }
    }
}

#[test]
fn test_token_kl_mismatched_lengths_returns_error() {
    let result = compute_token_kl(&[1.0, 2.0], &[1.0]);
    assert!(result.is_err());
}

// Reward normalization
#[test]
fn test_running_stats_welford() {
    let mut stats = RunningStats::new();
    for x in [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
        stats.update(x);
    }
    assert!((stats.mean() - 5.0).abs() < 1e-10);
    assert!((stats.var() - 4.0).abs() < 1e-10);
}

// Sequence packing
#[test]
fn test_pack_sequences_no_overflow() {
    let seqs = vec![vec![1,2,3], vec![4,5], vec![6,7,8,9]];
    let packed = pack_sequences(&seqs, 6).unwrap();
    // [1,2,3,4,5,PAD] and [6,7,8,9,PAD,PAD] — two batches
    assert_eq!(packed.len(), 2);
}
```

### Python Tests

```python
# PPO correctness
def test_ppo_solves_cartpole():
    """PPO must solve CartPole-v1 in < 50K timesteps."""
    ppo = PPO(env="CartPole-v1", policy=MlpPolicy(4, 2), config=PPOConfig(n_envs=8))
    ppo.train(total_timesteps=50_000)
    rewards = evaluate(ppo.policy, "CartPole-v1", n_episodes=20)
    assert np.mean(rewards) >= 475

# GRPO correctness
def test_grpo_improves_math_accuracy():
    """GRPO must improve accuracy on synthetic math from 0% toward > 20%."""
    ...

# DPO correctness
def test_dpo_preference_accuracy():
    """DPO must achieve > 90% on synthetic preference dataset."""
    ...

# Checkpoint round-trip
def test_checkpoint_resume_produces_identical_trajectory():
    ppo = PPO(env="CartPole-v1", ...)
    ppo.train(10_000)
    Checkpoint.save("ckpt.pt", model=ppo.policy, ...)
    ppo2 = PPO.from_checkpoint("ckpt.pt")
    # Both must produce identical next-batch
    ...

# Layer 1 composability
def test_collector_works_with_custom_policy():
    collector = RolloutCollector(env="CartPole-v1", n_envs=4)
    batch = collector.collect(my_custom_nn_module, n_steps=128)
    assert batch.obs.shape == (4 * 128, 4)

# Logging
def test_wandb_logger_receives_metrics():
    ...
```

## Success Criteria

- [ ] End-to-end PPO training for MuJoCo in < 50 lines of Python (Layer 1 API)
- [ ] End-to-end GRPO training for math reasoning in < 80 lines (Layer 1 API)
- [ ] PPO solves CartPole-v1 in < 50K steps across 5 seeds
- [ ] PPO on HalfCheetah-v4 matches SB3 learning curves within 1 std
- [ ] DPO achieves > 90% preference accuracy on synthetic data
- [ ] All Phase 6 benchmarks maintained (no regression)
- [ ] 100+ Python tests, 80+ Rust tests
- [ ] `pip install rlox` works on Linux x86_64 and macOS ARM64
- [ ] Checkpoint/resume produces identical trajectory to uninterrupted run
