# Rust-Python Architecture Review: Flexibility and Extensibility

**Date**: March 2026
**Scope**: Phase 7+ implementation strategy for rlox's three-layer API, plugin system, and zero-copy data flow.
**Status**: Implementation guide

---

## A. Rust Architecture for Python Extensibility

### A.1 Current State Assessment

The existing `RLEnv` trait in `crates/rlox-core/src/env/mod.rs` is well-designed for Rust consumers:

```rust
pub trait RLEnv: Send + Sync {
    fn step(&mut self, action: &Action) -> Result<Transition, RloxError>;
    fn reset(&mut self, seed: Option<u64>) -> Result<Observation, RloxError>;
    fn action_space(&self) -> &ActionSpace;
    fn obs_space(&self) -> &ObsSpace;
    fn render(&self) -> Option<String> { None }
}
```

The `Send + Sync` bounds enable Rayon parallelism. The problem: Python users cannot implement this trait. The `PyGymEnv` wrapper in `crates/rlox-python/src/env.rs` bridges a Gymnasium env, but it holds a `PyObject` and is not `Send + Sync` -- it cannot participate in `VecEnv`'s Rayon parallel stepping. This is the fundamental tension: the GIL prevents true parallel execution of Python-defined environments.

### A.2 The Dual-Dispatch Pattern for Python-Defined Environments

The solution is a two-tier environment system. Rust-native environments (CartPole, future MuJoCo bindings) step in parallel via Rayon. Python-defined environments step sequentially with a single GIL acquisition per batch, which is still faster than SB3's SubprocVecEnv because there is no IPC serialization.

Create a `BatchSteppable` trait that separates the parallelism decision from the step logic:

```rust
// crates/rlox-core/src/env/mod.rs

/// Trait for anything that can step a batch of environments.
/// This is the trait VecEnv dispatches to. Implementations decide
/// whether to use Rayon (Rust-native) or sequential-with-GIL (Python).
pub trait BatchSteppable: Send {
    fn step_batch(&mut self, actions: &[Action]) -> Result<BatchTransition, RloxError>;
    fn reset_batch(&mut self, seed: Option<u64>) -> Result<Vec<Observation>, RloxError>;
    fn num_envs(&self) -> usize;
}
```

Then refactor `VecEnv` to hold a `Box<dyn BatchSteppable>` instead of `Vec<Box<dyn RLEnv>>`. The Rayon-parallel path becomes one implementation:

```rust
/// Rayon-parallel batch stepper for Rust-native environments.
pub struct ParallelBatchStepper {
    envs: Vec<Box<dyn RLEnv>>,
}

impl BatchSteppable for ParallelBatchStepper {
    fn step_batch(&mut self, actions: &[Action]) -> Result<BatchTransition, RloxError> {
        // Current Rayon par_iter_mut logic, unchanged
    }
}
```

And on the Python side in `crates/rlox-python/`, a sequential stepper that acquires the GIL once per batch:

```rust
/// Sequential batch stepper for Python gymnasium environments.
/// Acquires the GIL once per step_batch call, steps all envs sequentially.
pub struct PythonBatchStepper {
    gym_envs: Vec<PyObject>,
    obs_dim: usize,
}

impl BatchSteppable for PythonBatchStepper {
    fn step_batch(&mut self, actions: &[Action]) -> Result<BatchTransition, RloxError> {
        Python::with_gil(|py| {
            // Step all envs sequentially, but only one GIL acquire
            let mut obs = Vec::with_capacity(self.gym_envs.len());
            let mut rewards = Vec::with_capacity(self.gym_envs.len());
            // ... collect results ...
        })
    }
}
```

This pattern preserves the existing Rayon advantage for Rust environments while allowing Python envs to participate in the same `VecEnv` API. The `PythonBatchStepper` lives in `rlox-python` (has PyO3 dependency), not `rlox-core` (stays pure Rust).

### A.3 Python-Callable Trait Objects via PyO3

For plugin points where Python users need to provide custom logic (reward functions, advantage estimators, loggers), use the "Python callback as Rust trait object" pattern:

```rust
// crates/rlox-core/src/training/reward.rs

/// Trait for reward transformations applied to raw environment rewards.
pub trait RewardTransform: Send + Sync {
    fn transform(&mut self, rewards: &mut [f64], episode_infos: &[EpisodeInfo]);
}

/// Built-in: running mean/variance normalization (Welford's algorithm).
pub struct RunningNormalize {
    mean: f64,
    var: f64,
    count: u64,
}

impl RewardTransform for RunningNormalize {
    fn transform(&mut self, rewards: &mut [f64], _infos: &[EpisodeInfo]) {
        for r in rewards.iter_mut() {
            self.count += 1;
            let delta = *r - self.mean;
            self.mean += delta / self.count as f64;
            let delta2 = *r - self.mean;
            self.var += delta * delta2;
            let std = (self.var / self.count as f64).sqrt().max(1e-8);
            *r = (*r - self.mean) / std;
        }
    }
}
```

On the Python side, wrap a Python callable into this trait:

```rust
// crates/rlox-python/src/reward.rs

/// Wraps a Python callable as a RewardTransform.
/// The callable receives (rewards: np.ndarray,) and returns np.ndarray.
struct PyRewardTransform {
    callback: PyObject,
}

// Note: NOT Send+Sync. This means Python reward transforms
// cannot be used inside Rayon parallel sections. They execute
// on the main thread between collect and train steps.
// This is acceptable: reward transforms are not on the hot path.

impl PyRewardTransform {
    fn transform_with_gil(&self, rewards: &mut [f64]) {
        Python::with_gil(|py| {
            let arr = PyArray1::from_slice(py, rewards);
            let result = self.callback.call1(py, (arr,)).expect("reward transform failed");
            let out: PyReadonlyArray1<f64> = result.extract(py).expect("must return array");
            rewards.copy_from_slice(out.as_slice().expect("contiguous"));
        });
    }
}
```

The key architectural decision: **Python callbacks execute between Rust hot-path sections, never inside them.** The collect step runs entirely in Rust. Then Python callbacks fire (reward transforms, logging hooks). Then the training step runs in Python/PyTorch. This matches the PRD's Principle 5: "The hot path is in Rust; the decision path is in Python."

### A.4 Trait Design for Pluggable Components

The current codebase has concrete structs (`ExperienceTable`, `ReplayBuffer`) but no shared buffer trait. For Phase 7, introduce traits for each pluggable component:

```rust
// crates/rlox-core/src/buffer/mod.rs

/// Trait for any buffer that supports push + sample.
pub trait ExperienceBuffer: Send + Sync {
    fn push(&mut self, record: ExperienceRecord) -> Result<(), RloxError>;
    fn push_batch(&mut self, records: &[ExperienceRecord]) -> Result<(), RloxError> {
        for r in records {
            self.push(r.clone())?;
        }
        Ok(())
    }
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool { self.len() == 0 }
}

/// Trait for buffers that support random sampling (replay buffers).
pub trait Sampleable: ExperienceBuffer {
    fn sample(&self, batch_size: usize, rng: &mut impl Rng) -> Result<SampledBatch, RloxError>;
}

/// Trait for buffers that support full-buffer iteration (rollout buffers).
pub trait Iterable: ExperienceBuffer {
    fn iter_epochs(&self, n_epochs: usize, batch_size: usize, rng: &mut impl Rng)
        -> EpochIterator<'_>;
}
```

This allows `ExperienceTable` to implement `Iterable` (for on-policy algorithms like PPO) and `ReplayBuffer` to implement `Sampleable` (for off-policy algorithms like SAC/DQN), while both share the `ExperienceBuffer` push interface. The type system enforces that you cannot accidentally sample from an on-policy rollout buffer or iterate an off-policy replay buffer.

For advantage estimators:

```rust
// crates/rlox-core/src/training/advantage.rs

/// Trait for advantage estimation algorithms.
pub trait AdvantageEstimator: Send + Sync {
    fn compute(
        &self,
        rewards: &[f64],
        values: &[f64],
        dones: &[f64],
        last_value: f64,
    ) -> (Vec<f64>, Vec<f64>); // (advantages, returns)
}

pub struct GAE {
    pub gamma: f64,
    pub lambda: f64,
}

impl AdvantageEstimator for GAE {
    fn compute(&self, rewards: &[f64], values: &[f64], dones: &[f64], last_value: f64)
        -> (Vec<f64>, Vec<f64>)
    {
        gae::compute_gae(rewards, values, dones, last_value, self.gamma, self.lambda)
    }
}

/// V-trace for IMPALA-style async correction (Phase 9).
pub struct VTrace {
    pub gamma: f64,
    pub rho_bar: f64,
    pub c_bar: f64,
}
```

This makes the advantage estimator swappable at the component level without touching algorithm code.

---

## B. The Three API Layers

### B.1 Layer 0: Primitives (Current State -- Mostly Done)

Layer 0 is the current API surface: raw Rust-backed operations callable from Python. What exists today in `python/rlox/__init__.py` is exactly Layer 0. The only gaps:

1. **Missing `push_batch` on `ExperienceTable` and `ReplayBuffer`.** Currently each push crosses the PyO3 boundary individually. Add a batch push that accepts a 2D numpy array for observations:

```rust
// In crates/rlox-python/src/buffer.rs, add to PyReplayBuffer

#[pyo3(signature = (obs_batch, actions, rewards, terminated, truncated))]
fn push_batch(
    &mut self,
    obs_batch: PyReadonlyArray2<f32>,
    actions: PyReadonlyArray1<f32>,
    rewards: PyReadonlyArray1<f32>,
    terminated: PyReadonlyArray1<bool>,
    truncated: PyReadonlyArray1<bool>,
) -> PyResult<()> {
    let obs = obs_batch.as_array();
    let n = obs.nrows();
    let actions = actions.as_slice()?;
    let rewards = rewards.as_slice()?;
    let terminated = terminated.as_slice()?;
    let truncated = truncated.as_slice()?;

    for i in 0..n {
        let record = ExperienceRecord {
            obs: obs.row(i).to_vec(),
            action: actions[i],
            reward: rewards[i],
            terminated: terminated[i],
            truncated: truncated[i],
        };
        self.inner.push(record)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    }
    Ok(())
}
```

2. **Missing continuous action support in `PyVecEnv.step_all`.** Currently hardcoded to `Action::Discrete`. Add a `step_all_continuous` method or make `step_all` polymorphic by accepting a numpy array and inferring the action type from the environment's action space.

3. **Missing `KLController` Python binding.** The Rust `KLController` in `crates/rlox-core/src/training/kl.rs` has no PyO3 wrapper. Expose it:

```rust
#[pyclass(name = "KLController")]
pub struct PyKLController {
    inner: KLController,
}

#[pymethods]
impl PyKLController {
    #[new]
    fn new(init_coeff: f64, target_kl: f64) -> Self {
        Self { inner: KLController::new(init_coeff, target_kl) }
    }

    fn coefficient(&self) -> f64 { self.inner.coefficient() }
    fn update(&mut self, measured_kl: f64) { self.inner.update(measured_kl); }
}
```

### B.2 Layer 1: Components (Phase 7 Target)

Layer 1 wraps Layer 0 primitives into lifecycle-managed components. These are **Python classes** that own Rust primitives internally. The key insight: Layer 1 lives primarily in Python (in `python/rlox/`), not in Rust. This is because the orchestration logic (when to collect, when to compute advantages, when to yield minibatches) is not performance-critical -- only the operations those orchestrations trigger are.

```
python/rlox/
    __init__.py          # Layer 0 re-exports
    collectors.py        # RolloutCollector, AsyncCollector
    stores.py            # ExperienceStore (wraps ExperienceTable + GAE)
    losses.py            # PPOLoss, DPOLoss (thin PyTorch loss modules)
    config.py            # Dataclass configs
    trainers/             # Layer 2
        __init__.py
        ppo.py
        grpo.py
        sac.py
```

The `RolloutCollector` is the critical Layer 1 component. It encapsulates the collect-compute-store cycle:

```python
# python/rlox/collectors.py

from dataclasses import dataclass
from typing import Iterator, Optional
import numpy as np
import torch
from rlox import VecEnv, ExperienceTable, compute_gae

@dataclass
class RolloutBatch:
    """A complete rollout with computed advantages, ready for training."""
    obs: torch.Tensor          # (n_steps * n_envs, obs_dim)
    actions: torch.Tensor      # (n_steps * n_envs,)
    log_probs: torch.Tensor    # (n_steps * n_envs,)
    advantages: torch.Tensor   # (n_steps * n_envs,)
    returns: torch.Tensor      # (n_steps * n_envs,)
    values: torch.Tensor       # (n_steps * n_envs,)

    def minibatch_iter(self, batch_size: int, rng: np.random.Generator) -> Iterator["RolloutBatch"]:
        """Yield shuffled minibatches for PPO-style multi-epoch training."""
        n = self.obs.shape[0]
        indices = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = indices[start : start + batch_size]
            yield RolloutBatch(
                obs=self.obs[idx],
                actions=self.actions[idx],
                log_probs=self.log_probs[idx],
                advantages=self.advantages[idx],
                returns=self.returns[idx],
                values=self.values[idx],
            )


class RolloutCollector:
    """Collects rollouts from a VecEnv, computes advantages via Rust GAE.

    This is the main Layer 1 component. It owns a VecEnv and an
    ExperienceTable, and yields RolloutBatch objects ready for training.
    """

    def __init__(
        self,
        env: VecEnv,
        n_steps: int = 2048,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        self.env = env
        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self._last_obs: Optional[np.ndarray] = None

    def collect(self, policy: "PolicyProtocol") -> RolloutBatch:
        """Run n_steps of environment interaction, return a RolloutBatch.

        Args:
            policy: Any object with .predict(obs) -> (actions, values, log_probs).
                    This is the "decision path in Python" from the PRD.
        """
        n_envs = self.env.num_envs()

        if self._last_obs is None:
            self._last_obs = np.array(self.env.reset_all())

        # Pre-allocate storage
        all_obs = np.zeros((self.n_steps, n_envs, self._last_obs.shape[-1]), dtype=np.float32)
        all_actions = np.zeros((self.n_steps, n_envs), dtype=np.float32)
        all_log_probs = np.zeros((self.n_steps, n_envs), dtype=np.float64)
        all_rewards = np.zeros((self.n_steps, n_envs), dtype=np.float64)
        all_values = np.zeros((self.n_steps, n_envs), dtype=np.float64)
        all_dones = np.zeros((self.n_steps, n_envs), dtype=np.float64)

        obs = self._last_obs

        for step in range(self.n_steps):
            # Decision path: Python (policy inference)
            with torch.no_grad():
                actions, values, log_probs = policy.predict(obs)

            all_obs[step] = obs
            all_actions[step] = actions
            all_values[step] = values
            all_log_probs[step] = log_probs

            # Hot path: Rust (env stepping)
            result = self.env.step_all(actions.tolist())
            obs = np.array(result["obs"])
            all_rewards[step] = np.array(result["rewards"])

            terminated = np.array(result["terminated"], dtype=np.float64)
            truncated = np.array(result["truncated"], dtype=np.float64)
            all_dones[step] = np.clip(terminated + truncated, 0.0, 1.0)

        self._last_obs = obs

        # Hot path: Rust (advantage computation)
        with torch.no_grad():
            _, last_values, _ = policy.predict(obs)

        # Per-env GAE computation, all in Rust
        all_advantages = np.zeros_like(all_rewards)
        all_returns = np.zeros_like(all_rewards)
        for env_idx in range(n_envs):
            adv, ret = compute_gae(
                all_rewards[:, env_idx],
                all_values[:, env_idx],
                all_dones[:, env_idx],
                last_values[env_idx],
                self.gamma,
                self.gae_lambda,
            )
            all_advantages[:, env_idx] = np.array(adv)
            all_returns[:, env_idx] = np.array(ret)

        # Flatten (n_steps, n_envs) -> (n_steps * n_envs)
        def flatten(arr):
            return torch.from_numpy(arr.reshape(-1))

        return RolloutBatch(
            obs=torch.from_numpy(all_obs.reshape(-1, all_obs.shape[-1])),
            actions=flatten(all_actions),
            log_probs=flatten(all_log_probs),
            advantages=flatten(all_advantages),
            returns=flatten(all_returns),
            values=flatten(all_values),
        )
```

Important design decision: the per-env GAE loop above calls `compute_gae` N times (once per env). This is fine because each call is microseconds. But an even better approach is to add a batched `compute_gae_batch` function on the Rust side that handles the per-env loop internally, avoiding N PyO3 boundary crossings:

```rust
// crates/rlox-core/src/training/gae.rs

/// Compute GAE for multiple environments simultaneously.
/// rewards, values, dones: shape [n_steps, n_envs] stored row-major.
pub fn compute_gae_batch(
    rewards: &[f64],       // [n_steps * n_envs]
    values: &[f64],        // [n_steps * n_envs]
    dones: &[f64],         // [n_steps * n_envs]
    last_values: &[f64],   // [n_envs]
    n_steps: usize,
    n_envs: usize,
    gamma: f64,
    gae_lambda: f64,
) -> (Vec<f64>, Vec<f64>) {
    let mut advantages = vec![0.0; n_steps * n_envs];
    let mut returns = vec![0.0; n_steps * n_envs];

    // Process each env independently -- could parallelize with Rayon
    for env in 0..n_envs {
        let mut last_gae = 0.0;
        for t in (0..n_steps).rev() {
            let idx = t * n_envs + env;
            let next_non_terminal = 1.0 - dones[idx];
            let next_value = if t == n_steps - 1 {
                last_values[env]
            } else {
                values[(t + 1) * n_envs + env]
            };
            let delta = rewards[idx] + gamma * next_value * next_non_terminal - values[idx];
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae;
            advantages[idx] = last_gae;
            returns[idx] = last_gae + values[idx];
        }
    }

    (advantages, returns)
}
```

### B.3 Layer 2: Trainers (Phase 8 Target)

Layer 2 is pure Python. Trainers compose Layer 1 components and own the training loop:

```python
# python/rlox/trainers/ppo.py

from dataclasses import dataclass, field
from typing import Optional, Union
import torch
import torch.nn as nn
from rlox import VecEnv
from rlox.collectors import RolloutCollector
from rlox.losses import PPOLoss

@dataclass
class PPOConfig:
    """All PPO hyperparameters. Typed, discoverable, serializable."""
    # Environment
    n_envs: int = 64
    n_steps: int = 2048

    # Algorithm
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_epochs: int = 4
    batch_size: int = 64

    # Training
    total_timesteps: int = 1_000_000
    seed: int = 42

    # Logging
    log_interval: int = 1  # log every N updates


class PPOTrainer:
    """One-liner PPO training. Composes Layer 1 components."""

    def __init__(
        self,
        env: Union[str, VecEnv],
        model: nn.Module,
        config: Optional[PPOConfig] = None,
    ):
        self.config = config or PPOConfig()

        if isinstance(env, str):
            # Create VecEnv from gymnasium env ID
            self.env = self._make_vec_env(env)
        else:
            self.env = env

        self.model = model
        self.collector = RolloutCollector(
            env=self.env,
            n_steps=self.config.n_steps,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )
        self.loss_fn = PPOLoss(
            clip_eps=self.config.clip_eps,
            entropy_coef=self.config.entropy_coef,
            value_coef=self.config.value_coef,
        )
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=self.config.learning_rate
        )

    def train(self, total_timesteps: Optional[int] = None):
        total = total_timesteps or self.config.total_timesteps
        steps_per_update = self.config.n_steps * self.config.n_envs
        n_updates = total // steps_per_update

        for update in range(n_updates):
            # Collect rollout (Rust hot path)
            batch = self.collector.collect(self.model)

            # Normalize advantages
            batch.advantages = (batch.advantages - batch.advantages.mean()) / (
                batch.advantages.std() + 1e-8
            )

            # Train for n_epochs (PyTorch hot path)
            rng = np.random.default_rng(self.config.seed + update)
            for _epoch in range(self.config.n_epochs):
                for mb in batch.minibatch_iter(self.config.batch_size, rng):
                    loss = self.loss_fn(mb, self.model)
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    self.optimizer.step()
```

### B.4 Ownership Model Across Layers

| Component | Rust owns | Python owns | Notes |
|-----------|-----------|-------------|-------|
| Environment state | Yes (CartPole, MuJoCo) | Yes (Gymnasium envs) | Rust envs step without GIL; Python envs acquire GIL per batch |
| Buffer memory | Yes (pre-allocated Vec) | No (views into Rust memory) | numpy arrays from `sample()` are views, not copies |
| GAE computation | Yes (pure Rust) | No (calls Rust, gets numpy back) | Single PyO3 boundary crossing per compute |
| Policy network | No | Yes (nn.Module) | Rust never touches model weights |
| Training loop | No | Yes (Python for/while) | Trainers are pure Python |
| Config | No | Yes (dataclass) | Serialized to YAML/JSON by Python |
| Logging | Rust collects metrics | Python flushes to W&B/TB | Batched: Rust accumulates, Python writes |

---

## C. Plugin / Extension Points

### C.1 Where to Allow Python-Defined Plugins

Based on the PRD's performance requirements and the "hot path in Rust / decision path in Python" principle, here are the extension points ranked by performance sensitivity:

| Extension Point | Performance Sensitivity | Recommended Implementation |
|----------------|------------------------|---------------------------|
| Custom environments | HIGH (inner loop) | Rust trait for fast envs, `PythonBatchStepper` bridge for Gymnasium envs. Python envs step sequentially. |
| Custom reward functions | MEDIUM (once per rollout) | Python callable, invoked between collect and train. |
| Custom advantage estimators | LOW (once per rollout) | Python callable returning (advantages, returns) arrays. Rust builtins (GAE, V-trace) preferred. |
| Custom loggers | LOW (periodic) | Python protocol with `log_metrics(dict)`. Batched flushing. |
| Custom policies | N/A | Always Python (nn.Module). This is deliberate. |
| Custom loss functions | N/A | Always Python (torch loss). This is deliberate. |

### C.2 Batched Python Callbacks

The current code acquires the GIL for every individual operation. For callbacks, batch them:

```rust
// crates/rlox-python/src/callbacks.rs

/// Manages a list of Python callbacks, invokes them in batch.
pub struct CallbackManager {
    callbacks: Vec<PyObject>,
}

impl CallbackManager {
    /// Invoke all callbacks once, passing the entire metrics dict.
    /// One GIL acquisition for all callbacks.
    pub fn fire(&self, metrics: &HashMap<String, f64>) -> PyResult<()> {
        if self.callbacks.is_empty() {
            return Ok(());
        }

        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            for (k, v) in metrics {
                dict.set_item(k, v)?;
            }
            for cb in &self.callbacks {
                cb.call1(py, (dict.clone(),))?;
            }
            Ok(())
        })
    }
}
```

The design rule: **never acquire the GIL inside a Rayon parallel section.** Callbacks fire on the main thread between collect and train phases.

### C.3 Rust Calling Back into Python Without Performance Degradation

Three patterns, in order of preference:

**Pattern 1: Batched post-hoc callbacks (preferred).**
Rust completes a full operation (e.g., 2048-step rollout), then passes results to Python once. This is what `RolloutCollector.collect()` does -- the entire collect is Rust, then Python processes the result.

**Pattern 2: Periodic flush callbacks.**
Rust accumulates metrics in a `HashMap<String, f64>` over many steps. Every N steps, it acquires the GIL and flushes to a Python logger. This amortizes GIL cost.

```rust
pub struct MetricsAccumulator {
    buffer: HashMap<String, Vec<f64>>,
    flush_interval: usize,
    step_count: usize,
    callback: PyObject,
}

impl MetricsAccumulator {
    pub fn record(&mut self, key: &str, value: f64) {
        self.buffer.entry(key.to_string()).or_default().push(value);
        self.step_count += 1;

        if self.step_count % self.flush_interval == 0 {
            self.flush();
        }
    }

    fn flush(&mut self) {
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            for (k, values) in &self.buffer {
                let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
                let _ = dict.set_item(k, mean);
            }
            let _ = self.callback.call1(py, (dict,));
        });
        self.buffer.clear();
    }
}
```

**Pattern 3: GIL-aware environment stepping (fallback for Gymnasium envs).**
Already implemented via `PyGymEnv`. The key optimization is batch-stepping: acquire GIL once, step all Python envs, release. Never acquire per-env.

---

## D. Configuration System

### D.1 Dataclass-Based Config

The config system should be pure Python dataclasses with serde-compatible Rust structs for any configs that cross the PyO3 boundary:

```python
# python/rlox/config.py

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Union
import json
import yaml
from pathlib import Path

@dataclass
class EnvConfig:
    """Environment configuration."""
    env_id: str = "CartPole-v1"
    n_envs: int = 64
    seed: int = 42
    # For Rust-native envs, this bypasses Gymnasium entirely
    use_rust_native: bool = True

@dataclass
class BufferConfig:
    """Experience storage configuration."""
    capacity: int = 100_000
    # "rollout" = ExperienceTable (on-policy), "replay" = ReplayBuffer (off-policy)
    buffer_type: str = "rollout"

@dataclass
class TrainConfig:
    """Base training configuration. Algorithm configs extend this."""
    total_timesteps: int = 1_000_000
    learning_rate: float = 3e-4
    seed: int = 42
    log_interval: int = 1
    checkpoint_interval: int = 10
    device: str = "auto"  # "cpu", "cuda", "auto"

@dataclass
class PPOConfig(TrainConfig):
    """PPO-specific configuration."""
    # Environment
    env: EnvConfig = field(default_factory=EnvConfig)
    buffer: BufferConfig = field(default_factory=lambda: BufferConfig(buffer_type="rollout"))

    # PPO hyperparameters
    n_steps: int = 2048
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_epochs: int = 4
    batch_size: int = 64
    target_kl: Optional[float] = None  # Early stopping KL threshold

    def to_yaml(self, path: Union[str, Path]) -> None:
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "PPOConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @classmethod
    def from_cli(cls, args: Optional[List[str]] = None) -> "PPOConfig":
        """Parse CLI overrides: --learning_rate=1e-4 --n_envs=128"""
        import argparse
        parser = argparse.ArgumentParser()
        for f in fields(cls):
            parser.add_argument(f"--{f.name}", type=type(f.default), default=f.default)
        parsed = parser.parse_args(args)
        return cls(**vars(parsed))
```

### D.2 Config Extensibility

Custom algorithms extend the base config. No inheritance hierarchy deeper than one level:

```python
@dataclass
class GRPOConfig(TrainConfig):
    """GRPO-specific configuration."""
    group_size: int = 16
    kl_coef: float = 0.1
    kl_target: float = 0.02
    max_gen_length: int = 1024
    temperature: float = 1.0
    # Reference model
    ref_model: Optional[str] = None
    use_peft: bool = False
```

### D.3 Experiment Reproducibility via Config Serialization

Every training run saves its full config alongside checkpoints:

```python
# python/rlox/experiment.py

import json
import subprocess
from dataclasses import asdict
from datetime import datetime

def save_experiment_metadata(config, output_dir: Path):
    """Save everything needed to reproduce this experiment."""
    metadata = {
        "config": asdict(config),
        "timestamp": datetime.utcnow().isoformat(),
        "rlox_version": rlox.__version__,
        "git_hash": _get_git_hash(),
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
    }
    with open(output_dir / "experiment.json", "w") as f:
        json.dump(metadata, f, indent=2)

def _get_git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"
```

---

## E. Zero-Copy Data Flow

### E.1 Current State of the rust-numpy Bridge

The current implementation in `crates/rlox-python/src/buffer.rs` uses two patterns:

1. **`PyArray1::from_slice(py, raw)` -- copies data.** Used for `observations()`, `rewards()`. This copies from Rust-owned `Vec<f32>` to a Python-owned numpy array. Acceptable for small arrays but should be avoided for large observation buffers.

2. **`PyArray1::from_vec(py, vec)` -- transfers ownership.** Used in `sample()`. The Rust `Vec` is moved into numpy's memory management. Zero-copy, but the Rust side loses access. Fine for sampled batches (one-shot consumption).

3. **`PyArray2::from_vec2` -- copies.** Used in `VecEnv.step_all` and `reset_all`. Copies from `Vec<Vec<f32>>` to a contiguous numpy array. This is a copy because `Vec<Vec<f32>>` is not contiguous.

### E.2 Improvements for Phase 7

**Fix the VecEnv observation return to be zero-copy.** Currently `BatchTransition.obs` is `Vec<Vec<f32>>` which requires a copy to produce a contiguous numpy array. Instead, store observations flat in a single `Vec<f32>` with known dimensions:

```rust
// crates/rlox-core/src/env/parallel.rs

pub struct BatchTransition {
    /// Observations: flat [n_envs * obs_dim], reshaped on the Python side.
    pub obs_flat: Vec<f32>,
    pub obs_dim: usize,
    pub rewards: Vec<f64>,
    pub terminated: Vec<bool>,
    pub truncated: Vec<bool>,
}
```

Then on the Python side:

```rust
// crates/rlox-python/src/env.rs

fn step_all<'py>(&mut self, py: Python<'py>, actions: Vec<u32>) -> PyResult<Bound<'py, PyDict>> {
    // ... step logic ...
    let dict = PyDict::new(py);
    // Zero-copy: from_vec takes ownership of the flat vec
    let obs_1d = PyArray1::from_vec(py, batch.obs_flat);
    let obs_2d = obs_1d.reshape([n, batch.obs_dim])?;
    dict.set_item("obs", obs_2d)?;
    // ...
}
```

### E.3 DLPack Integration Plan

DLPack eliminates the numpy intermediary for PyTorch tensor creation. Instead of `torch.from_numpy(buffer.sample()["obs"])` (which goes Rust -> numpy -> PyTorch), DLPack goes Rust -> PyTorch directly.

The implementation path:

1. Add `dlpack` crate dependency to `rlox-python`.
2. Implement `__dlpack__` and `__dlpack_device__` on buffer sample results.
3. PyTorch's `torch.from_dlpack()` then creates a tensor from Rust-owned memory without any copy.

```rust
// crates/rlox-python/src/tensor.rs (future)

use pyo3::prelude::*;

/// A Rust-owned tensor that exposes DLPack protocol.
/// PyTorch can consume this via torch.from_dlpack(tensor).
#[pyclass]
pub struct RloxTensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

#[pymethods]
impl RloxTensor {
    fn __dlpack__(&self, py: Python<'_>) -> PyResult<PyObject> {
        // Create DLManagedTensor pointing to self.data
        // The DLPack capsule holds a reference to self, preventing deallocation
        // until PyTorch is done with the tensor.
        todo!("Implement DLPack capsule creation")
    }

    fn __dlpack_device__(&self) -> (i32, i32) {
        (1, 0) // kDLCPU, device_id=0
    }
}
```

The priority here is moderate. The current numpy bridge adds ~100ns of overhead per array which is negligible for batch sizes of 64+. DLPack matters more when GPU buffers are added (Phase 9+), where the Rust -> numpy -> CUDA path would require an extra device transfer.

### E.4 Variable-Length Data Across the Boundary

The `VarLenStore` already uses the Arrow ListArray pattern (flat data + offsets). For Python consumption, expose the raw flat array and offsets so Python can reconstruct without copying:

```rust
// Add to PyVarLenStore

fn flat_data<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
    PyArray1::from_slice(py, self.inner.flat_data())
}

fn offsets<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u64>> {
    PyArray1::from_slice(py, self.inner.offsets())
}
```

This lets Python reconstruct variable-length sequences without N individual `get()` calls. For sequence packing in GRPO:

```python
# Python side: efficient batch assembly from VarLenStore
flat = store.flat_data()       # One PyO3 call, returns numpy view
offsets = store.offsets()       # One PyO3 call, returns numpy view
# Now pack sequences into fixed-size batches using only numpy ops
```

---

## F. Specific Rust Patterns

### F.1 Error Handling Strategy

The current `RloxError` in `crates/rlox-core/src/error.rs` is good but incomplete. Expand it for Phase 7:

```rust
#[derive(Debug, Error)]
pub enum RloxError {
    // Existing variants
    #[error("Invalid action: {0}")]
    InvalidAction(String),

    #[error("Environment error: {0}")]
    EnvError(String),

    #[error("Shape mismatch: expected {expected}, got {got}")]
    ShapeMismatch { expected: String, got: String },

    #[error("Buffer error: {0}")]
    BufferError(String),

    // New variants for Phase 7+
    #[error("Config error: {0}")]
    ConfigError(String),

    #[error("Checkpoint error: {0}")]
    CheckpointError(String),

    #[error("Inference server error: {0}")]
    InferenceError(String),

    #[error("Timeout after {duration_ms}ms: {operation}")]
    Timeout { operation: String, duration_ms: u64 },

    #[error("Numerical instability: {0}")]
    NumericalError(String),
}
```

For Python exception mapping, create a dedicated exception hierarchy instead of mapping everything to `PyRuntimeError`:

```rust
// crates/rlox-python/src/errors.rs

use pyo3::create_exception;
use pyo3::exceptions::PyException;

create_exception!(rlox, RloxError, PyException);
create_exception!(rlox, ShapeMismatchError, RloxError);
create_exception!(rlox, BufferError, RloxError);
create_exception!(rlox, EnvError, RloxError);
create_exception!(rlox, ConfigError, RloxError);

pub fn to_py_err(e: rlox_core::error::RloxError) -> PyErr {
    match e {
        rlox_core::error::RloxError::ShapeMismatch { expected, got } => {
            ShapeMismatchError::new_err(format!("expected {expected}, got {got}"))
        }
        rlox_core::error::RloxError::BufferError(msg) => BufferError::new_err(msg),
        rlox_core::error::RloxError::EnvError(msg) => EnvError::new_err(msg),
        rlox_core::error::RloxError::InvalidAction(msg) => EnvError::new_err(msg),
        other => RloxError::new_err(other.to_string()),
    }
}
```

This lets Python users catch specific exceptions:

```python
from rlox.errors import ShapeMismatchError, BufferError

try:
    buffer.push(wrong_shaped_obs, ...)
except ShapeMismatchError as e:
    print(f"Fix your observation shape: {e}")
```

### F.2 Type-Safe Action/Observation Spaces

The current `ActionSpace` and `ObsSpace` enums in `crates/rlox-core/src/env/spaces.rs` are solid. For Phase 7, add two things:

1. **`Dict` and `Tuple` space types** (required for Gymnasium compatibility per PRD section 10.3):

```rust
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ObsSpace {
    Discrete(usize),
    Box { low: Vec<f32>, high: Vec<f32>, shape: Vec<usize> },
    MultiDiscrete(Vec<usize>),
    MultiBinary(usize),
    Dict(BTreeMap<String, ObsSpace>),
    Tuple(Vec<ObsSpace>),
}
```

2. **Compile-time dimension checking** for Rust-native environments via const generics:

```rust
/// A fixed-dimension observation for environments where obs_dim is known at compile time.
/// CartPole: Obs<4>. Atari: Obs<{84*84*4}>.
pub struct FixedObs<const N: usize>(pub [f32; N]);

/// Trait for environments with compile-time-known observation dimensions.
pub trait TypedRLEnv<const OBS: usize, const ACT: usize>: Send + Sync {
    fn step(&mut self, action: &[f32; ACT]) -> Result<TypedTransition<OBS>, RloxError>;
    fn reset(&mut self, seed: Option<u64>) -> Result<FixedObs<OBS>, RloxError>;
}
```

This is optional and parallel to the dynamic `RLEnv` trait. It enables buffer pre-allocation without runtime dimension checks for environments where dimensions are known at compile time. The dynamic `RLEnv` remains the primary interface for Python-accessible environments.

### F.3 Async Patterns for Inference Server Integration

For LLM post-training, rlox needs to communicate with inference servers (vLLM, TGI, SGLang). This is I/O-bound, so Tokio is the right tool:

```rust
// crates/rlox-core/src/llm/inference.rs (future, Phase 9)

use tokio::sync::mpsc;

/// A batch generation request to an inference server.
pub struct GenerationRequest {
    pub prompt_tokens: Vec<u32>,
    pub max_tokens: usize,
    pub temperature: f64,
}

/// Client for async batch generation against vLLM/TGI.
pub struct InferenceClient {
    endpoint: String,
    max_concurrent: usize,
}

impl InferenceClient {
    /// Generate completions for a batch of prompts.
    /// Uses Tokio for concurrent HTTP requests with backpressure.
    pub async fn generate_batch(
        &self,
        requests: Vec<GenerationRequest>,
    ) -> Result<Vec<Vec<u32>>, RloxError> {
        let semaphore = Arc::new(Semaphore::new(self.max_concurrent));
        let mut set = JoinSet::new();

        for req in requests {
            let sem = semaphore.clone();
            let endpoint = self.endpoint.clone();
            set.spawn(async move {
                let _permit = sem.acquire().await.map_err(|_| {
                    RloxError::InferenceError("semaphore closed".into())
                })?;
                // HTTP request to inference server
                Self::generate_one(&endpoint, req).await
            });
        }

        let mut results = Vec::with_capacity(set.len());
        while let Some(result) = set.join_next().await {
            results.push(result.map_err(|e| {
                RloxError::InferenceError(e.to_string())
            })??);
        }
        Ok(results)
    }
}
```

The Tokio runtime lives in Rust. Python calls the async function via a blocking bridge:

```rust
// crates/rlox-python/src/llm.rs

#[pymethods]
impl PyInferenceClient {
    fn generate_batch(&self, py: Python<'_>, prompts: Vec<Vec<u32>>) -> PyResult<Vec<Vec<u32>>> {
        // Release the GIL while waiting for async I/O
        py.allow_threads(|| {
            let rt = tokio::runtime::Runtime::new()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            rt.block_on(self.inner.generate_batch(requests))
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }
}
```

The `py.allow_threads()` is critical: it releases the GIL while the Rust code waits on I/O, allowing other Python threads to run.

### F.4 Thread Safety for Concurrent Buffer Access

The current `ReplayBuffer` is not concurrent -- it uses plain `Vec` without synchronization. For Phase 7's decoupled collect/train pattern (PRD item D2), the buffer needs concurrent access.

The recommended approach is **not** `Arc<Mutex<ReplayBuffer>>` (contention at high throughput). Instead, use a lock-free single-producer single-consumer ring buffer for the collect -> train transfer, and a separate `ReplayBuffer` owned by the trainer:

```rust
// crates/rlox-core/src/buffer/concurrent.rs

use std::sync::atomic::{AtomicUsize, Ordering};

/// Lock-free SPSC ring buffer for transferring experience batches
/// from collector thread to trainer thread.
pub struct SPSCExperienceChannel {
    slots: Vec<Option<RolloutBatch>>,
    head: AtomicUsize,   // Writer position
    tail: AtomicUsize,   // Reader position
    capacity: usize,
}

impl SPSCExperienceChannel {
    pub fn new(capacity: usize) -> Self {
        let mut slots = Vec::with_capacity(capacity);
        slots.resize_with(capacity, || None);
        Self {
            slots,
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
            capacity,
        }
    }

    /// Try to push a batch. Returns Err if full.
    pub fn try_push(&mut self, batch: RolloutBatch) -> Result<(), RolloutBatch> {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);
        if (head + 1) % self.capacity == tail {
            return Err(batch); // Full
        }
        self.slots[head] = Some(batch);
        self.head.store((head + 1) % self.capacity, Ordering::Release);
        Ok(())
    }

    /// Try to pop a batch. Returns None if empty.
    pub fn try_pop(&mut self) -> Option<RolloutBatch> {
        let tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Acquire);
        if tail == head {
            return None; // Empty
        }
        let batch = self.slots[tail].take();
        self.tail.store((tail + 1) % self.capacity, Ordering::Release);
        batch
    }
}
```

For off-policy algorithms that need true concurrent read/write to a shared replay buffer, use `parking_lot::RwLock` (lower overhead than std Mutex) with a read-copy-update strategy:

```rust
use parking_lot::RwLock;
use std::sync::Arc;

/// Thread-safe replay buffer. Writers take exclusive lock (rare, batched).
/// Readers take shared lock (frequent, concurrent).
pub struct ConcurrentReplayBuffer {
    inner: Arc<RwLock<ReplayBuffer>>,
}

impl ConcurrentReplayBuffer {
    /// Push a batch of records. Takes write lock once per batch.
    pub fn push_batch(&self, records: &[ExperienceRecord]) -> Result<(), RloxError> {
        let mut buf = self.inner.write();
        for r in records {
            buf.push(r.clone())?;
        }
        Ok(())
    }

    /// Sample without blocking writers for more than the lock acquisition.
    pub fn sample(&self, batch_size: usize, seed: u64) -> Result<SampledBatch, RloxError> {
        let buf = self.inner.read();
        buf.sample(batch_size, seed)
    }
}
```

---

## G. Recommendations for Phase 7 Implementation

### G.1 Implementation Order

Based on dependency analysis and the PRD's Phase 7 deliverables:

**Week 1-2: Foundation traits and batch operations.**
1. Add `ExperienceBuffer`, `Sampleable`, `Iterable` traits to `rlox-core`.
2. Add `push_batch` to `ExperienceTable` and `ReplayBuffer` (both Rust and PyO3).
3. Add `compute_gae_batch` to `rlox-core`.
4. Add `KLController` PyO3 binding.
5. Flatten `BatchTransition.obs` to `Vec<f32>` for zero-copy returns.

**Week 3-4: Python Layer 1 components.**
6. Create `python/rlox/collectors.py` with `RolloutCollector`.
7. Create `python/rlox/losses.py` with `PPOLoss` (torch loss module).
8. Create `python/rlox/config.py` with `PPOConfig`, `GRPOConfig`.
9. Add continuous action support to `PyVecEnv.step_all`.

**Week 5-7: PPO end-to-end.**
10. Create `python/rlox/trainers/ppo.py` with `PPOTrainer`.
11. Implement reward normalization (Welford's algorithm) in `rlox-core`.
12. Implement advantage normalization in the trainer.
13. Checkpoint/resume (model weights + optimizer + RNG state).
14. Validate: CartPole solved in < 50K steps across 5 seeds.

**Week 8-10: GRPO end-to-end.**
15. Create `python/rlox/trainers/grpo.py` with `GRPOTrainer`.
16. Implement sequence packing in `rlox-core`.
17. Integrate with HuggingFace model loading.
18. Validate: improvement on synthetic math task.

**Week 11-12: DPO, logging, wheels.**
19. DPO loss computation and training loop.
20. W&B / TensorBoard integration.
21. Prebuilt wheel infrastructure via maturin.

### G.2 Architectural Decisions to Lock In Now

These decisions have downstream consequences and should be finalized before Phase 7 coding begins:

1. **Layer 1 lives in Python, not Rust.** The `RolloutCollector`, `PPOLoss`, and `PPOTrainer` are pure Python classes. Only primitive operations cross the PyO3 boundary. This maximizes researcher flexibility (they can read and modify the training loop) while keeping the performance advantage (all hot-path ops delegate to Rust).

2. **Configs are Python dataclasses, not Rust structs.** The config system does not cross the PyO3 boundary. Rust functions accept individual parameters (`gamma: f64, lambda: f64`), not config objects. The config is unpacked on the Python side. This avoids serde-PyO3 bridging complexity and keeps configs trivially serializable.

3. **The `PolicyProtocol` is a Python Protocol, not a Rust trait.** The policy network is always a Python object (nn.Module). Rust never holds a reference to it. The collector calls `policy.predict(obs)` in Python, then passes results to Rust. This keeps the model in Python's domain where autograd, CUDA, and torch.compile work naturally.

4. **Exception hierarchy.** Switch from `PyRuntimeError::new_err` everywhere to typed exceptions (`ShapeMismatchError`, `BufferError`, `EnvError`). Do this in Phase 7 week 1 before more code depends on the current pattern.

5. **Flat observation storage in BatchTransition.** Change `Vec<Vec<f32>>` to `(Vec<f32>, usize)` now. This is a breaking internal change but nothing outside `rlox-core` and `rlox-python` depends on it.

### G.3 Decisions to Defer

1. **DLPack integration.** The numpy bridge works fine for Phase 7. DLPack adds complexity (capsule lifecycle management, device negotiation) that does not pay off until GPU-resident buffers exist. Defer to Phase 8.

2. **Lock-free concurrent buffers.** Phase 7 uses synchronous collect-then-train. Concurrent buffers are needed for Phase 9's decoupled architecture. Use `parking_lot::RwLock` if needed before that.

3. **Tokio async runtime.** Not needed until inference server integration (Phase 9). Adding Tokio to `rlox-core`'s dependency tree has compile-time cost; defer until there is a concrete use case.

4. **Const-generic typed environments.** Nice for compile-time dimension checking but adds API surface complexity. The dynamic `RLEnv` trait serves Phase 7's needs. Reconsider for Phase 9 when MuJoCo native bindings (fixed obs dims) are added.

5. **Multi-agent support.** The buffer and env traits should not be designed around multi-agent from the start. Add MARL-specific traits in Phase 9 when the requirements are concrete.

### G.4 Potential Pitfalls Based on the Current Codebase

1. **The `assert_eq!` in `compute_token_kl` will panic in production.** File: `crates/rlox-core/src/llm/ops.rs`, line 29. This should return `Result<f64, RloxError>` instead of panicking on mismatched lengths. Fix this in Phase 7 week 1.

2. **`ExperienceRecord.action` is `f32` (scalar).** This works for Discrete and 1D continuous actions but not for multi-dimensional continuous control (e.g., HalfCheetah has 6 continuous actions). Phase 7 needs `action: Vec<f32>` and corresponding `act_dim` changes in `ExperienceTable` and `ReplayBuffer`. This is a breaking change to the buffer format -- do it first before building higher layers on top.

3. **`ReplayBuffer.sample()` takes a `seed: u64` parameter, creating a new RNG per sample call.** This is deterministic but slow for repeated sampling in a training loop. Consider storing a `ChaCha8Rng` on the buffer and exposing `sample_next` that advances the RNG, while keeping `sample(seed)` for reproducibility testing.

4. **`VecEnv` auto-resets on done but discards the terminal observation.** In `crates/rlox-core/src/env/parallel.rs`, line 56, the auto-reset replaces `transition.obs` with the new episode's obs. Many algorithms (especially PPO with value bootstrapping at truncation) need the terminal observation. Store it in `transition.info` or add a `terminal_obs` field to `BatchTransition`.

5. **`PyVecEnv::new()` only creates CartPole environments.** It is hardcoded in `crates/rlox-python/src/env.rs`, line 73-83. Phase 7 must generalize this to accept a Gymnasium env ID string and create N instances, or accept a Python factory callable:

```python
# Target API:
env = VecEnv.from_gymnasium("HalfCheetah-v4", n=64, seed=42)
# or:
env = VecEnv.from_factory(lambda: gym.make("HalfCheetah-v4"), n=64, seed=42)
# or for Rust-native:
env = VecEnv.cartpole(n=64, seed=42)
```

6. **No `next_obs` in ExperienceRecord.** Off-policy algorithms (SAC, DQN, TD3) need the next observation for TD target computation. The current `ExperienceRecord` stores only the current observation. Either add `next_obs: Vec<f32>` to the record, or store transitions as `(s, a, r, s', done)` tuples. Alternatively, store observations in a separate column and compute next_obs as `obs[i+1]` at sample time (the SB3 approach, more memory-efficient). Decide this before implementing SAC/DQN.

7. **`bool` columns in ReplayBuffer use `Vec<bool>`.** In Rust, `Vec<bool>` is not bitpacked (each bool is 1 byte). For buffers with millions of entries, consider using `bitvec` or packing terminated/truncated into a single `u8` flags field. This is a micro-optimization but aligns with the PRD's "< 10% above theoretical minimum for buffer storage" goal.

---

## Summary of Key Recommendations

| Decision | Recommendation | Rationale |
|----------|---------------|-----------|
| Layer 1 location | Python | Researchers must be able to read and modify the training loop |
| Config system | Python dataclasses | Simplest serialization, IDE completion, no PyO3 crossing needed |
| Policy interface | Python Protocol | Keep nn.Module in Python's domain for autograd/CUDA/compile |
| Buffer traits | Rust traits (`ExperienceBuffer`, `Sampleable`, `Iterable`) | Type-safe algorithm/buffer pairing at compile time |
| Python callbacks | Batched, between Rust hot-path sections | Minimize GIL acquisitions; never acquire inside Rayon |
| Env extensibility | `BatchSteppable` trait with Rayon and sequential implementations | Rust envs parallel, Python envs sequential-but-no-IPC |
| Error handling | Typed Python exception hierarchy | Catchable, informative errors instead of generic RuntimeError |
| Observation storage | Flat `Vec<f32>` with dimension metadata | Zero-copy PyO3 returns, cache-friendly iteration |
| Action storage | `Vec<f32>` (variable dimension) | Support multi-dim continuous control from day 1 |
| Terminal obs | Store in `BatchTransition.terminal_obs` | Required for PPO value bootstrapping at truncation |
