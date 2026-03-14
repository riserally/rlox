# Phase 8 — Production Hardening (v0.7)

**Status: NOT STARTED**
**Target: Q4 2026**
**PRD Features: A4–A6, A8–A10, B4–B5, E9, L5, API2, API5, R2, R4, O2**
**Depends on: Phase 7 (PPO, GRPO, DPO working end-to-end)**

---

## Objective

Harden rlox for production research use. Complete the algorithm set with SAC, TD3, and DQN. Build the one-liner Trainer API (Layer 2). Add the composable config system, callback hooks, statistical evaluation toolkit, and training diagnostics. After this phase, rlox is a credible SB3 alternative with superior performance.

## PRD Feature Mapping

| PRD ID | Feature | Priority | Description |
|--------|---------|----------|-------------|
| A4 | SAC | P1 | Soft Actor-Critic for continuous control, automatic entropy tuning |
| A5 | TD3 | P1 | Twin Delayed DDPG, deterministic policy |
| A6 | DQN / C51 / Rainbow | P1 | Discrete control family, prioritized replay integration |
| A8 | Online DPO / OAIF | P1 | On-policy generation + preference construction + DPO update |
| A9 | Best-of-N sampling | P1 | Generate N, score, keep best — simple LLM baseline |
| A10 | Reward model training | P1 | Bradley-Terry preference model |
| B4 | Prioritized experience replay | P1 | Sum-tree in Rust: O(log N) sample + update, thread-safe |
| B5 | Memory-mapped buffer overflow | P1 | Spill to NVMe when buffer exceeds RAM |
| E9 | MuJoCo native bindings | P1 | mujoco-rs direct bindings, eliminate segfault source |
| L5 | Verifiable reward sandbox | P1 | Sandboxed code/math execution with timeouts |
| API2 | Composable config system | P0 | Dataclass-based, YAML/CLI merge, IDE completion |
| API5 | Callback / hook system | P1 | on_step, on_episode_end, on_train_batch, on_eval |
| R2 | Experiment snapshot | P1 | Git hash, config, deps, hardware, seed — auto-captured |
| R4 | Statistical evaluation toolkit | P1 | IQM, performance profiles, stratified bootstrap CI |
| O2 | Training diagnostics | P1 | Auto-detect entropy collapse, KL spikes, gradient explosions |

## Implementation Plan

### Part A: Off-Policy Infrastructure (Weeks 1–3)

Off-policy algorithms (SAC, TD3, DQN) require different infrastructure than PPO:
- **Replay buffer with priorities** (B4) — sum-tree for O(log N) sampling
- **Target networks** — exponential moving average weight updates
- **Continuous action noise** — Gaussian (SAC) or Ornstein-Uhlenbeck (TD3)

#### A.1 Prioritized Replay Buffer (Rust)

```rust
// crates/rlox-core/src/buffer/priority.rs

/// Sum-tree backed prioritized experience replay.
/// O(log N) sample proportional to priority.
/// O(log N) priority update.
/// Thread-safe via RwLock (readers don't block each other).
pub struct PrioritizedReplayBuffer {
    data: ExperienceTable,
    tree: SumTree,
    alpha: f64,          // priority exponent
    beta: f64,           // importance sampling correction
    beta_schedule: f64,  // linear annealing to 1.0
    max_priority: f64,
}

impl PrioritizedReplayBuffer {
    pub fn push(&mut self, transition: &Transition, priority: f64) -> Result<(), RloxError>;
    pub fn sample(&self, batch_size: usize) -> Result<(PrioritizedBatch, Vec<f64>), RloxError>;
    pub fn update_priorities(&mut self, indices: &[usize], priorities: &[f64]) -> Result<(), RloxError>;
}

/// Binary indexed tree for O(log N) prefix-sum queries.
struct SumTree {
    tree: Vec<f64>,
    capacity: usize,
}
```

#### A.2 Target Network Utilities (Python)

```python
# python/rlox/networks.py
def polyak_update(source: nn.Module, target: nn.Module, tau: float = 0.005):
    """Soft update: target = tau * source + (1 - tau) * target."""
    for sp, tp in zip(source.parameters(), target.parameters()):
        tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)
```

### Part B: SAC Implementation (Weeks 3–4)

```python
# python/rlox/algorithms/sac.py
class SAC:
    """Soft Actor-Critic with automatic entropy tuning."""

    def __init__(self, env: str, config: SACConfig):
        self.actor = SquashedGaussianPolicy(obs_dim, act_dim)
        self.critic1 = QNetwork(obs_dim, act_dim)
        self.critic2 = QNetwork(obs_dim, act_dim)  # twin critics
        self.target_critic1 = deepcopy(self.critic1)
        self.target_critic2 = deepcopy(self.critic2)
        self.log_alpha = nn.Parameter(torch.zeros(1))  # auto entropy tuning
        self.target_entropy = -act_dim  # heuristic from SAC paper
        self.buffer = rlox.ReplayBuffer(config.buffer_size, obs_dim, act_dim)

    def train(self, total_timesteps: int):
        for step in range(total_timesteps):
            # 1. Collect (single step, store in Rust buffer)
            action = self.actor.sample(obs)
            next_obs, reward, done, info = self.env.step(action)
            self.buffer.push(obs, action, reward, done)  # Rust push

            # 2. Sample batch (Rust — 10x faster than SB3)
            batch = self.buffer.sample(config.batch_size)

            # 3. Critic update (PyTorch)
            # 4. Actor update (PyTorch)
            # 5. Alpha update (auto entropy tuning)
            # 6. Target network soft update
```

Key SAC details:
- Squashed Gaussian policy (tanh transform + log-prob correction)
- Twin critics (minimum for pessimistic Q-estimate)
- Automatic entropy coefficient (alpha) tuning via dual gradient descent
- Target entropy = -dim(A) heuristic
- Soft target update (tau=0.005)

### Part C: TD3 Implementation (Week 4)

```python
# python/rlox/algorithms/td3.py
class TD3:
    """Twin Delayed DDPG. Deterministic policy, delayed updates."""

    def __init__(self, env: str, config: TD3Config):
        ...
        # Key differences from SAC:
        # - Deterministic policy (no entropy)
        # - Target policy smoothing (add clipped noise to target actions)
        # - Delayed policy updates (update actor every 2 critic updates)
```

### Part D: DQN / Rainbow (Weeks 5–6)

```python
# python/rlox/algorithms/dqn.py
class DQN:
    """DQN with optional Rainbow extensions."""

    def __init__(self, env: str, config: DQNConfig):
        self.q_network = QNetwork(obs_dim, n_actions)
        self.target_network = deepcopy(self.q_network)
        self.buffer = rlox.PrioritizedReplayBuffer(...)  # Rust sum-tree

    # Rainbow extensions (configurable):
    # - Double DQN (use online network for action selection, target for evaluation)
    # - Dueling architecture (separate V and A streams)
    # - Prioritized replay (B4)
    # - N-step returns
    # - C51 distributional (categorical distribution over returns)
    # - Noisy networks (parameter-space exploration)
```

### Part E: One-Liner Trainer API — Layer 2 (Week 7)

```python
# python/rlox/trainers.py
class PPOTrainer:
    """One-liner PPO training. SB3-inspired API."""

    def __init__(
        self,
        env: str,
        model: nn.Module | str = "mlp",  # "mlp", "cnn", or custom nn.Module
        config: PPOConfig | None = None,
        callbacks: list[Callback] | None = None,
        logger: Logger | None = None,
    ):
        if isinstance(model, str):
            model = build_default_policy(model, env)
        self.algo = PPO(env=env, policy=model, config=config or PPOConfig())
        ...

    def train(self, total_timesteps: int):
        self.algo.train(total_timesteps)

    def evaluate(self, n_episodes: int = 10) -> dict:
        ...

    def save(self, path: str): ...
    def load(cls, path: str) -> "PPOTrainer": ...

# Usage:
# PPOTrainer(env="HalfCheetah-v4").train(1_000_000)
# GRPOTrainer(model="deepseek-r1", reward_fn=math_reward).train(dataset)
```

### Part F: Composable Config System (Week 7, parallel)

```python
# python/rlox/config.py
from dataclasses import dataclass, field
import yaml

@dataclass
class PPOConfig:
    # Environment
    n_envs: int = 8
    n_steps: int = 2048

    # Optimization
    learning_rate: float = 3e-4
    n_epochs: int = 10
    batch_size: int = 64
    max_grad_norm: float = 0.5

    # PPO-specific
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    target_kl: float | None = None  # early stopping threshold

    # Normalization
    normalize_obs: bool = True
    normalize_rewards: bool = True

    # Scheduling
    lr_schedule: str = "linear"  # "linear", "constant", "cosine"
    total_timesteps: int = 1_000_000

    @classmethod
    def from_yaml(cls, path: str) -> "PPOConfig":
        with open(path) as f:
            return cls(**yaml.safe_load(f))

    @classmethod
    def from_cli(cls, args: list[str] | None = None) -> "PPOConfig":
        """Parse CLI overrides: --learning_rate=1e-3 --n_envs=16"""
        ...

    def merge(self, overrides: dict) -> "PPOConfig":
        """Merge overrides into config. Returns new config."""
        ...
```

### Part G: Callback / Hook System (Week 8)

```python
# python/rlox/callbacks.py
from abc import ABC, abstractmethod

class Callback(ABC):
    """Base callback. Override any method to hook into training."""

    def on_training_start(self, locals: dict) -> None: ...
    def on_rollout_start(self) -> None: ...
    def on_step(self, step: int, obs, action, reward, done, info) -> bool:
        """Return False to stop training early."""
        return True
    def on_rollout_end(self, metrics: dict) -> None: ...
    def on_train_batch(self, loss: float, metrics: dict) -> None: ...
    def on_eval(self, mean_reward: float, metrics: dict) -> None: ...
    def on_training_end(self) -> None: ...

class EvalCallback(Callback):
    """Periodic evaluation with best model saving."""
    def __init__(self, eval_env: str, eval_freq: int = 10_000, n_eval_episodes: int = 10): ...

class EarlyStoppingCallback(Callback):
    """Stop training when reward plateaus."""
    def __init__(self, patience: int = 10, min_delta: float = 0.0): ...

class CheckpointCallback(Callback):
    """Save checkpoints at regular intervals."""
    def __init__(self, save_freq: int = 50_000, save_path: str = "checkpoints/"): ...
```

### Part H: Experiment Snapshot (Week 8, parallel)

```python
# python/rlox/experiment.py
import json, subprocess, platform

def capture_experiment_metadata(config, seed: int) -> dict:
    return {
        "git_hash": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"]).decode().strip()),
        "config": asdict(config),
        "seed": seed,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "rlox_version": rlox.__version__,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "timestamp": datetime.utcnow().isoformat(),
    }
```

### Part I: Statistical Evaluation Toolkit (Week 9)

```python
# python/rlox/evaluation.py

def interquartile_mean(scores: np.ndarray) -> float:
    """IQM: mean of middle 50% of scores. More robust than median or mean."""
    ...

def performance_profiles(scores_dict: dict[str, np.ndarray], thresholds: np.ndarray) -> dict:
    """Fraction of runs above each threshold, per algorithm."""
    ...

def stratified_bootstrap_ci(scores: np.ndarray, n_bootstrap: int = 10_000, ci: float = 0.95) -> tuple:
    """Bootstrap confidence interval for IQM."""
    ...
```

Implements Agarwal et al. (NeurIPS 2021) statistical evaluation without requiring external `rliable`.

### Part J: MuJoCo Native Bindings (Weeks 9–10)

```rust
// crates/rlox-core/src/env/mujoco.rs

/// Direct mujoco-rs bindings for GIL-free, segfault-free MuJoCo stepping.
pub struct MuJoCoEnv {
    model: mujoco::Model,
    data: mujoco::Data,
    max_steps: usize,
    step_count: usize,
}

impl RLEnv for MuJoCoEnv {
    fn step(&mut self, action: &Action) -> Result<Transition, RloxError> { ... }
    fn reset(&mut self, seed: Option<u64>) -> Result<Observation, RloxError> { ... }
}
```

This eliminates the #1 source of MuJoCo segfaults (multithreaded Python ctypes access) and enables true Rayon-parallel MuJoCo stepping.

### Part K: Training Diagnostics (Week 10)

```python
# python/rlox/diagnostics.py

class TrainingDiagnostics(Callback):
    """Auto-detect common training failures."""

    def on_train_batch(self, loss, metrics):
        # Entropy collapse: entropy drops below 10% of initial
        if metrics["entropy"] < self.initial_entropy * 0.1:
            warnings.warn("Entropy collapse detected — policy becoming deterministic")

        # KL spike: KL divergence > 10x target
        if metrics.get("approx_kl") and metrics["approx_kl"] > 10 * self.target_kl:
            warnings.warn(f"KL spike: {metrics['approx_kl']:.4f} >> target {self.target_kl}")

        # Gradient explosion: grad norm > 100x max_grad_norm
        if metrics.get("grad_norm") and metrics["grad_norm"] > 100 * self.max_grad_norm:
            warnings.warn(f"Gradient explosion: norm={metrics['grad_norm']:.1f}")

        # Value function divergence: explained variance < -1
        if metrics.get("explained_var") and metrics["explained_var"] < -1.0:
            warnings.warn("Value function diverging — predicting worse than constant")
```

### Part L: Verifiable Reward Sandbox (Week 10, parallel)

```rust
// crates/rlox-core/src/llm/sandbox.rs

/// Sandboxed code execution for verifiable rewards (math, code).
/// Uses WASM or process isolation with timeout + memory limits.
pub struct RewardSandbox {
    timeout_ms: u64,
    max_memory_bytes: usize,
}

impl RewardSandbox {
    /// Execute Python code string and return stdout.
    /// Returns Err on timeout, memory limit, or execution error.
    pub fn execute(&self, code: &str) -> Result<String, RloxError> { ... }
}
```

### Part M: Memory-Mapped Buffer Overflow (Week 10, parallel)

```rust
// crates/rlox-core/src/buffer/mmap.rs

/// When replay buffer exceeds RAM, transparently spill to NVMe.
/// Uses mmap for lazy loading — only pages actually accessed are read.
pub struct MmapReplayBuffer {
    hot: ReplayBuffer,          // in-memory ring for recent data
    cold: MmapFile,             // on-disk overflow
    hot_capacity: usize,
    total_capacity: usize,
}
```

Critical for LLM trajectories where a single sequence can be 100K+ tokens.

## TDD Test Specifications

### Rust Tests

```rust
#[test]
fn test_sum_tree_proportional_sampling() {
    let mut tree = SumTree::new(1000);
    // Insert with known priorities
    tree.set(0, 1.0);
    tree.set(1, 3.0);
    // Sample 10K times — index 1 should appear ~3x more often
    ...
}

#[test]
fn test_prioritized_buffer_importance_weights() {
    let mut buf = PrioritizedReplayBuffer::new(100, 4, 1, alpha=0.6, beta=0.4);
    // Importance weights should correct for non-uniform sampling
    ...
}

#[test]
fn test_mujoco_env_rayon_parallel() {
    // 16 MuJoCo envs stepped in parallel — no segfault
    let envs: Vec<MuJoCoEnv> = (0..16).map(|_| MuJoCoEnv::new("HalfCheetah")).collect();
    let stepper = ParallelBatchStepper::new(envs);
    for _ in 0..1000 {
        stepper.step_batch(&random_actions);
    }
}
```

### Python Tests

```python
def test_sac_solves_pendulum():
    """SAC must solve Pendulum-v1 (return > -200) in 50K steps."""
    trainer = SACTrainer(env="Pendulum-v1", config=SACConfig())
    trainer.train(50_000)
    rewards = trainer.evaluate(n_episodes=20)
    assert np.mean(rewards) > -200

def test_td3_halfcheetah():
    """TD3 on HalfCheetah-v4 > 5000 return in 1M steps."""
    ...

def test_dqn_cartpole():
    """DQN must solve CartPole-v1 in < 50K steps."""
    ...

def test_one_liner_api():
    """PPOTrainer(env='CartPole-v1').train(50_000) just works."""
    trainer = PPOTrainer(env="CartPole-v1")
    trainer.train(50_000)
    assert trainer.evaluate()["mean_reward"] >= 475

def test_config_yaml_merge():
    config = PPOConfig.from_yaml("config.yaml")
    config = config.merge({"learning_rate": 1e-3})
    assert config.learning_rate == 1e-3

def test_callback_early_stopping():
    ...

def test_experiment_metadata_captured():
    meta = capture_experiment_metadata(PPOConfig(), seed=42)
    assert "git_hash" in meta
    assert meta["seed"] == 42
```

## Success Criteria

- [ ] `PPOTrainer(env="HalfCheetah-v4").train(1_000_000)` starts in < 10 seconds
- [ ] Published benchmark: rlox PPO/SAC/DQN vs SB3/CleanRL on MuJoCo + Atari
- [ ] SAC solves Pendulum-v1 in 50K steps, HalfCheetah > 5000 in 1M steps
- [ ] DQN solves CartPole-v1 in < 50K steps
- [ ] Config: YAML + CLI overrides + dataclass — all tested
- [ ] Callbacks: eval, checkpoint, early stopping — all working
- [ ] Statistical evaluation: IQM + bootstrap CI matches rliable output
- [ ] Training diagnostics detect entropy collapse on synthetic test
- [ ] MuJoCo native: 16 parallel envs, no segfaults, 1000 steps
- [ ] Zero regressions on Phase 6/7 benchmarks
- [ ] pip install produces working wheels on all target platforms
