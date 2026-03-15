# Implementation Plan: Phases 7-9

> rlox -- Rust-accelerated RL library with PyO3 bindings
> Date: 2026-03-15
> Baseline: 255 Rust tests, 85 Python tests, Phases 0-6 complete

---

## Table of Contents

- [Phase 7: Algorithm Completeness](#phase-7-algorithm-completeness)
  - [Sprint 7.1: Foundation Fixes](#sprint-71-foundation-fixes)
  - [Sprint 7.2: Continuous Actions](#sprint-72-continuous-actions)
  - [Sprint 7.3: End-to-End Algorithms](#sprint-73-end-to-end-algorithms)
  - [Sprint 7.4: Config, Callbacks, Persistence](#sprint-74-config-callbacks-persistence)
  - [Sprint 7.5: Logging and Distribution](#sprint-75-logging-and-distribution)
- [Phase 8: Production Hardening](#phase-8-production-hardening)
  - [Sprint 8.1: Layer 2 Trainer API](#sprint-81-layer-2-trainer-api)
  - [Sprint 8.2: Config System and Evaluation](#sprint-82-config-system-and-evaluation)
  - [Sprint 8.3: Diagnostics and Native Envs](#sprint-83-diagnostics-and-native-envs)
  - [Sprint 8.4: Memory-Mapped Buffers and Reward Sandbox](#sprint-84-memory-mapped-buffers-and-reward-sandbox)
- [Phase 9: Distributed and Scale](#phase-9-distributed-and-scale)
  - [Sprint 9.1: Decoupled Pipeline](#sprint-91-decoupled-pipeline)
  - [Sprint 9.2: gRPC Workers](#sprint-92-grpc-workers)
  - [Sprint 9.3: LLM Serving Integration](#sprint-93-llm-serving-integration)
  - [Sprint 9.4: Multi-Agent and World Models](#sprint-94-multi-agent-and-world-models)
  - [Sprint 9.5: Async Training and Stable API](#sprint-95-async-training-and-stable-api)

---

## Phase 7: Algorithm Completeness

### Sprint 7.1: Foundation Fixes

#### 7.1.1 Fix PyVecEnv Silent Fallback (CRITICAL BUG)

**Problem**: `crates/rlox-python/src/env.rs:92-104` silently creates CartPole for any unknown `env_id`. A user requesting `"HalfCheetah-v4"` gets CartPole with no warning.

**Rust-side changes**:
- `crates/rlox-python/src/env.rs`: Replace the catch-all match arm with `Err(PyValueError::new_err(...))` listing supported env IDs.
- Add `PyValueError` to `pyo3::exceptions` imports.

**Python-side changes**: None (error surfaces naturally).

**TDD specification**:
```
# Rust unit tests (crates/rlox-python, requires pyo3-test or integration test)
test_pyvecenv_unknown_env_id_raises_value_error
    - Call PyVecEnv::new(4, None, Some("NonExistent-v1"))
    - Assert returns Err with message containing "NonExistent-v1"
    - Assert CartPole still works: PyVecEnv::new(4, None, Some("CartPole-v1")) succeeds

# Python integration tests (tests/test_env.py)
test_vecenv_unknown_env_id_raises
    - rlox.VecEnv(4, env_id="FakeEnv-v99") -> raises RuntimeError
test_vecenv_cartpole_explicit_still_works
    - rlox.VecEnv(4, env_id="CartPole-v1") -> works
test_vecenv_default_env_id_is_cartpole
    - rlox.VecEnv(4) -> works (backward compat)
```

**Benchmark gate**: Run `benchmarks/bench_env_stepping.py`. CartPole throughput must not regress (accept < 2% variance).

**Profiling checkpoint**: None needed (trivial change).

**Dependencies**: None (standalone fix).

---

#### 7.1.2 BatchSteppable Trait

**Problem**: `RolloutCollector` and future algorithms need a uniform interface over `VecEnv` (Rust-native) and `GymVecEnv` (Python Gymnasium). The current code couples directly to `PyVecEnv`.

**Rust-side changes**:
- New file `crates/rlox-core/src/env/batch.rs`:
  ```rust
  pub trait BatchSteppable: Send {
      fn step_batch(&mut self, actions: &[Action]) -> Result<BatchTransition, RloxError>;
      fn reset_batch(&mut self, seed: Option<u64>) -> Result<Vec<Observation>, RloxError>;
      fn num_envs(&self) -> usize;
      fn action_space(&self) -> &ActionSpace;
      fn obs_space(&self) -> &ObsSpace;
  }
  ```
- Implement `BatchSteppable` for `VecEnv` (delegating to existing methods).
- Re-export from `crates/rlox-core/src/env/mod.rs`.

**Python-side changes**: None yet (consumed in 7.1.3).

**TDD specification**:
```
# crates/rlox-core/src/env/batch.rs
test_vecenv_implements_batch_steppable
    - Create VecEnv with 4 CartPoles
    - Call step_batch, reset_batch via trait reference (&mut dyn BatchSteppable)
    - Assert shapes match VecEnv::step_all output exactly

test_batch_steppable_action_space_propagates
    - Assert action_space() returns Discrete(2) for CartPole VecEnv

test_batch_steppable_wrong_action_count
    - step_batch with wrong number of actions -> ShapeMismatch error
```

**Benchmark gate**: None (trait is zero-cost, no runtime overhead).

**Profiling checkpoint**: None.

**Dependencies**: None.

---

#### 7.1.3 Python-side GymVecEnv Wrapper

**Problem**: The `RolloutCollector` can only use Rust `VecEnv` (CartPole). Need a Python wrapper that presents `gymnasium.vector.SyncVectorEnv` behind the same interface for arbitrary Gymnasium envs.

**Rust-side changes**: None.

**Python-side changes**:
- New file `python/rlox/gym_vec_env.py`:
  ```python
  class GymVecEnv:
      """Wraps gymnasium.vector.SyncVectorEnv with the same step_all/reset_all
      interface as rlox.VecEnv, so RolloutCollector can use either."""
      def __init__(self, env_id: str, n_envs: int, seed: int = 0): ...
      def step_all(self, actions) -> dict: ...
      def reset_all(self, seed=None) -> np.ndarray: ...
      def num_envs(self) -> int: ...
      @property
      def action_space(self): ...
      @property
      def observation_space(self): ...
  ```
- Ensure return dict keys match VecEnv: `obs`, `rewards`, `terminated`, `truncated`, `terminal_obs`.
- Update `RolloutCollector.__init__` to auto-select: if `env_id` is a known Rust env, use `rlox.VecEnv`; otherwise use `GymVecEnv`.
- Export from `python/rlox/__init__.py`.

**TDD specification**:
```
# tests/test_gym_vec_env.py
test_gymvecenv_cartpole_step_returns_correct_keys
    - GymVecEnv("CartPole-v1", 4).step_all([0,1,0,1]) has keys: obs, rewards, terminated, truncated, terminal_obs
test_gymvecenv_obs_shape
    - obs shape is (n_envs, obs_dim)
test_gymvecenv_auto_reset_on_done
    - Step 200 times, never raises, obs shape stays (n_envs, obs_dim)
test_gymvecenv_terminal_obs_present_on_done
    - When terminated[i] is True, terminal_obs[i] is not None
test_gymvecenv_pendulum_continuous
    - GymVecEnv("Pendulum-v1", 2) accepts float actions
test_collector_uses_gymvecenv_for_pendulum
    - RolloutCollector("Pendulum-v1", ...) internally creates GymVecEnv
test_collector_still_uses_rust_vecenv_for_cartpole
    - RolloutCollector("CartPole-v1", ...) uses rlox.VecEnv
```

**Benchmark gate**: Run `benchmarks/bench_env_stepping.py`. GymVecEnv CartPole should be within 2x of Rust VecEnv (expected: Python overhead is dominant).

**Profiling checkpoint**: Compare GymVecEnv vs VecEnv stepping time for CartPole-v1 at 256 envs. Document the overhead ratio.

**Dependencies**: 7.1.1 (PyVecEnv must error on unknown envs so we can detect and route correctly).

---

### Sprint 7.2: Continuous Actions

#### 7.2.1 ContinuousPolicy (Gaussian) for On-Policy

**Problem**: `DiscretePolicy` is the only policy. PPO for MuJoCo needs a Gaussian actor with tanh squashing or unbounded actions.

**Rust-side changes**: None (policy is PyTorch-only for now).

**Python-side changes**:
- New class in `python/rlox/policies.py`:
  ```python
  class ContinuousPolicy(nn.Module):
      """MLP actor-critic for continuous action spaces.

      Actor outputs mean + log_std for a diagonal Gaussian.
      Critic is a separate MLP -> scalar.
      """
      def __init__(self, obs_dim, act_dim, hidden=64, log_std_init=-0.5): ...
      def get_action_and_logprob(self, obs) -> (actions, log_probs): ...
      def get_value(self, obs) -> values: ...
      def get_logprob_and_entropy(self, obs, actions) -> (log_probs, entropy): ...
  ```
- Use state-independent log_std (learnable parameter, not network output) as in CleanRL PPO continuous.
- Orthogonal init with sqrt(2) gain on hidden, 0.01 on actor head, 1.0 on critic head.

**TDD specification**:
```
# tests/test_policies.py
test_continuous_policy_action_shape
    - obs (B, 4), act_dim=2 -> actions shape (B, 2)
test_continuous_policy_logprob_shape
    - log_probs shape (B,)
test_continuous_policy_value_shape
    - values shape (B,)
test_continuous_policy_entropy_positive
    - entropy > 0 for all samples
test_continuous_policy_deterministic_with_seed
    - Same seed -> same actions
test_continuous_policy_logprob_gradient_flows
    - log_probs.sum().backward() does not raise
test_continuous_policy_get_logprob_and_entropy_matches
    - Actions sampled via get_action_and_logprob
    - Recomputed log_probs via get_logprob_and_entropy match (within tol)
```

**Benchmark gate**: None (no Rust code changed).

**Profiling checkpoint**: None.

**Dependencies**: None.

---

#### 7.2.2 Continuous Action Support in RolloutCollector

**Problem**: `RolloutCollector.collect()` calls `env.step_all(actions_np.astype(np.uint32).tolist())` -- hardcoded for discrete. Continuous actions need float arrays.

**Rust-side changes**:
- `crates/rlox-python/src/env.rs`: Add `step_all_continuous` method on `PyVecEnv` that accepts `Vec<Vec<f32>>` and maps to `Action::Continuous`. Or better: make `step_all` polymorphic by accepting a Python list of either ints or float-lists. Use a union dispatch.
- Alternative (simpler): Add a separate `step_all_continuous(actions: Vec<Vec<f32>>)` method.

**Python-side changes**:
- Update `RolloutCollector.collect()`:
  - Detect action space type (discrete vs continuous).
  - For discrete: current `actions_np.astype(np.uint32).tolist()`.
  - For continuous: `actions_np.tolist()` (list of float-lists).
- `GymVecEnv.step_all()` already handles arbitrary actions via gymnasium.

**TDD specification**:
```
# Rust (crates/rlox-python)
test_pyvecenv_step_all_continuous_shape
    - Not applicable for CartPole, but ensure the method exists and validates

# Python
test_collector_continuous_pendulum_collect
    - RolloutCollector("Pendulum-v1", n_envs=4)
    - collector.collect(ContinuousPolicy(3, 1), n_steps=32)
    - batch.actions.shape == (128, 1)
test_collector_continuous_actions_in_bounds
    - All actions in [-2, 2] for Pendulum
test_collector_continuous_advantages_finite
    - No NaN/Inf in batch.advantages
```

**Benchmark gate**: Run `benchmarks/bench_e2e_rollout.py` for CartPole (must not regress). Add a new Pendulum benchmark row.

**Profiling checkpoint**: Profile a 256-env x 2048-step continuous rollout with `py-spy`. Check if GIL contention in GymVecEnv dominates.

**Dependencies**: 7.1.3 (GymVecEnv), 7.2.1 (ContinuousPolicy).

---

### Sprint 7.3: End-to-End Algorithms

#### 7.3.1 End-to-End PPO for MuJoCo (Continuous)

**Problem**: PPO currently only works for CartPole. Need PPO on continuous envs like HalfCheetah, Hopper, Walker2d.

**Rust-side changes**: None (all heavy lifting is in Rust GAE, buffers; policy/training loop stays in Python).

**Python-side changes**:
- Update `python/rlox/algorithms/ppo.py`:
  - Remove hardcoded `_CARTPOLE_OBS_DIM = 4` / `_CARTPOLE_N_ACTIONS = 2`.
  - Query `env.observation_space` and `env.action_space` from GymVecEnv/VecEnv.
  - Auto-select `DiscretePolicy` vs `ContinuousPolicy` based on action space type.
  - Add `action_space_type` detection utility.
- Update `PPOConfig` to add `total_timesteps` and `env_id` fields so the config is self-contained.

**TDD specification**:
```
# tests/test_ppo_e2e.py
test_ppo_cartpole_converges
    - PPO("CartPole-v1", n_envs=8).train(50_000)
    - mean_reward > 300 (CartPole solved threshold is 195)

test_ppo_pendulum_runs_without_error
    - PPO("Pendulum-v1", n_envs=4).train(10_000)
    - Returns dict with expected metric keys

test_ppo_auto_selects_discrete_policy_for_cartpole
    - PPO("CartPole-v1").policy is instance of DiscretePolicy

test_ppo_auto_selects_continuous_policy_for_pendulum
    - PPO("Pendulum-v1").policy is instance of ContinuousPolicy

test_ppo_custom_policy_accepted
    - PPO("CartPole-v1", policy=custom_net).policy is custom_net

test_ppo_lr_annealing_decreases
    - After train(), optimizer LR < initial LR
```

**Benchmark gate**:
- `benchmarks/bench_e2e_rollout.py`: CartPole throughput must not regress (< 5%).
- New benchmark: `benchmarks/bench_ppo_continuous.py` -- measure wall-clock for 100k steps on Pendulum, compare to SB3 PPO.

**Profiling checkpoint**: `cargo flamegraph` on Rust GAE computation during a Pendulum run. `py-spy` on the full training loop to identify Python bottlenecks. Look for: unnecessary numpy-torch conversions, GIL contention in GymVecEnv.

**Dependencies**: 7.2.1 (ContinuousPolicy), 7.2.2 (continuous collector), 7.1.3 (GymVecEnv).

---

#### 7.3.2 End-to-End GRPO for LLM Post-Training

**Problem**: `python/rlox/algorithms/grpo.py` exists but needs validation with a real-ish LLM workflow and the batched Rust advantages.

**Rust-side changes**: None (Rust-side `compute_batch_group_advantages` already exists).

**Python-side changes**:
- Update `GRPO.train_step()`:
  - Replace per-prompt loop with batched `rlox.compute_batch_group_advantages(all_rewards, group_size)`.
  - Use `rlox.compute_token_kl_schulman` for the KL computation (more numerically stable).
- Add `reward_fn` parameter validation: must be callable, return list[float] of correct length.
- Add `GRPOConfig` dataclass to `python/rlox/config.py`.

**TDD specification**:
```
# tests/test_grpo_e2e.py
test_grpo_train_step_returns_metrics
    - Mock model with generate() and forward()
    - reward_fn returns random floats
    - train_step(prompts) returns dict with loss, mean_reward, kl

test_grpo_batched_advantages_matches_loop
    - Compare batch result to per-group loop (must be identical)

test_grpo_kl_schulman_used
    - Patch compute_token_kl_schulman, assert it gets called

test_grpo_invalid_reward_fn_raises
    - reward_fn returning wrong length -> ValueError

test_grpo_train_multiple_epochs
    - GRPO.train(prompts, n_epochs=3) runs without error

test_grpo_config_validation
    - GRPOConfig(group_size=0) raises ValueError
    - GRPOConfig(kl_coef=-1) raises ValueError
```

**Benchmark gate**: Run `benchmarks/bench_trl_comparison.py` -- GRPO advantages must remain 35x+ vs NumPy/PyTorch.

**Profiling checkpoint**: Profile `train_step` with `py-spy` on a batch of 32 prompts x 8 group_size. Verify Rust advantage computation is < 1% of total wall-clock (rest should be generation/forward pass).

**Dependencies**: None (existing Rust ops).

---

#### 7.3.3 Reward Function Extensibility

**Problem**: `RolloutCollector` uses env rewards directly. Some applications need reward shaping, curiosity bonuses, or custom reward functions.

**Rust-side changes**: None.

**Python-side changes**:
- Add `reward_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray] | None` parameter to `RolloutCollector.__init__`.
  - Signature: `reward_fn(obs, actions, env_rewards) -> modified_rewards`.
  - If `None`, use env rewards directly (default).
- Apply after each `step_all`, before GAE.

**TDD specification**:
```
# tests/test_collectors.py
test_collector_custom_reward_fn_applied
    - reward_fn = lambda obs, act, r: r * 2.0
    - Verify batch.rewards are 2x the env rewards

test_collector_reward_fn_receives_correct_shapes
    - reward_fn checks: obs.shape == (n_envs, obs_dim), etc.

test_collector_default_no_reward_fn
    - RolloutCollector without reward_fn uses env rewards unchanged

test_collector_reward_fn_returning_wrong_shape_raises
    - reward_fn returning scalar instead of array -> ValueError
```

**Benchmark gate**: Run `benchmarks/bench_e2e_rollout.py` with no reward_fn. Must not regress (the `if reward_fn is not None` check is trivial).

**Profiling checkpoint**: None.

**Dependencies**: 7.1.3 (GymVecEnv -- so collector works for arbitrary envs).

---

### Sprint 7.4: Config, Callbacks, Persistence

#### 7.4.1 Config Consolidation

**Problem**: `PPOConfig` is defined twice -- in `python/rlox/config.py` and in `python/rlox/algorithms/ppo.py`. They have slightly different fields (`normalize_rewards`, `normalize_obs` are only in the algorithm version).

**Rust-side changes**: None.

**Python-side changes**:
- Delete `PPOConfig` from `python/rlox/algorithms/ppo.py`.
- Add `normalize_rewards` and `normalize_obs` fields to `python/rlox/config.py:PPOConfig`.
- Update `PPO.__init__` to accept a `PPOConfig` instance or kwargs.
- Update `python/rlox/algorithms/__init__.py` to re-export from `rlox.config`.
- Add `GRPOConfig`, `TD3Config` to `python/rlox/config.py` following the same pattern.

**TDD specification**:
```
# tests/test_config.py
test_ppoconfig_single_source_of_truth
    - from rlox.config import PPOConfig
    - from rlox.algorithms import PPOConfig as PPOConfig2
    - assert PPOConfig is PPOConfig2

test_ppoconfig_has_normalize_fields
    - PPOConfig().normalize_rewards is False
    - PPOConfig().normalize_obs is False

test_grpo_config_validation
    - GRPOConfig(group_size=0) raises ValueError

test_td3_config_defaults
    - TD3Config().policy_delay == 2
    - TD3Config().tau == 0.005

test_config_merge_preserves_type
    - PPOConfig().merge({"n_envs": 16}) returns PPOConfig

test_config_to_dict_roundtrip
    - PPOConfig.from_dict(PPOConfig().to_dict()) == PPOConfig()
```

**Benchmark gate**: None.

**Profiling checkpoint**: None.

**Dependencies**: None.

---

#### 7.4.2 Wire Callbacks into Training Loops

**Problem**: Callbacks exist (`callbacks.py`) but are only superficially wired in trainers. `on_step`, `on_rollout_end`, `on_train_batch` are never called with actual data.

**Rust-side changes**: None.

**Python-side changes**:
- Update `PPO.train()`:
  - Call `callbacks.on_step(step=step, reward=reward)` after each env step.
  - Call `callbacks.on_rollout_end(step=update, batch=batch)` after collection.
  - Call `callbacks.on_train_batch(step=update, epoch=epoch, metrics=metrics)` after each minibatch.
  - Respect `on_step` return value: if `False`, stop training early.
- Update `SAC.train()` and `DQN.train()` similarly.
- Make `EvalCallback.on_step` actually run evaluation episodes when `step % eval_freq == 0`.
- Make `CheckpointCallback.on_step` actually save when `step % save_freq == 0`.

**TDD specification**:
```
# tests/test_callbacks.py
test_callback_on_step_called_with_reward
    - Custom callback records kwargs
    - After PPO.train(1024), callback received "reward" kwarg

test_early_stopping_stops_training
    - EarlyStoppingCallback(patience=1, min_delta=9999)
    - PPO.train(100_000) returns early (< 100_000 steps)

test_eval_callback_runs_evaluation
    - EvalCallback(eval_freq=500, n_eval_episodes=2)
    - After train, callback._step_count > 0 and evaluation ran

test_checkpoint_callback_creates_files
    - CheckpointCallback(save_freq=500, save_path=tmpdir)
    - After train, checkpoint files exist in tmpdir

test_callback_list_composes
    - CallbackList([cb1, cb2]) calls both in order

test_on_rollout_end_receives_batch
    - Callback receives batch kwarg with RolloutBatch
```

**Benchmark gate**: Run `benchmarks/bench_e2e_rollout.py` with no callbacks. Must not regress. Then with 3 no-op callbacks: < 1% overhead.

**Profiling checkpoint**: `py-spy` a 50k-step PPO run with `EvalCallback(eval_freq=5000)`. Verify eval episodes are < 5% of total wall-clock.

**Dependencies**: 7.4.1 (config consolidation, so callbacks can access config).

---

#### 7.4.3 Checkpoint / Resume (Save/Load)

**Problem**: Algorithms have no save/load. Users cannot resume interrupted training.

**Rust-side changes**:
- `ActorCritic` trait already has `save(&self, path)` and `load(&mut self, path)`.
- Add `save`/`load` to `PyActorCritic` in `crates/rlox-python/src/nn.rs` -- delegate to inner backend.

**Python-side changes**:
- Add `save(path)` and `load(path)` class methods to `PPO`, `SAC`, `DQN`:
  ```python
  def save(self, path: str) -> None:
      """Save policy weights, optimizer state, config, and training state."""
      state = {
          "policy_state_dict": self.policy.state_dict(),
          "optimizer_state_dict": self.optimizer.state_dict(),
          "config": self.config.to_dict(),
          "step": self._global_step,
      }
      torch.save(state, path)

  @classmethod
  def load(cls, path: str, env_id: str) -> "PPO":
      """Load a saved PPO agent."""
  ```
- For Rust `ActorCritic` backend: use the trait's `save`/`load` which serializes to MessagePack or safetensors.

**TDD specification**:
```
# tests/test_checkpoint.py
test_ppo_save_load_roundtrip
    - Train PPO for 1000 steps, save to tmpfile
    - Load, train for 1000 more steps -- no error
    - Loaded config matches saved config

test_ppo_save_creates_file
    - save(path) creates a file at path

test_ppo_load_nonexistent_raises
    - PPO.load("nonexistent.pt", "CartPole-v1") raises FileNotFoundError

test_sac_save_load_preserves_alpha
    - SAC trains, save, load -> alpha matches

test_rust_actor_critic_save_load
    - PyActorCritic("burn", 4, 2).save(path)
    - PyActorCritic("burn", 4, 2).load(path) -> no error
    - Act produces same output before/after load

test_checkpoint_callback_uses_save
    - CheckpointCallback triggers save at correct intervals
```

**Benchmark gate**: Measure save/load latency for a 64-64 MLP. Must be < 10ms.

**Profiling checkpoint**: None (I/O bound, not compute).

**Dependencies**: 7.4.2 (CheckpointCallback needs save wired).

---

### Sprint 7.5: Logging and Distribution

#### 7.5.1 Logging Integration

**Problem**: `WandbLogger` and `TensorBoardLogger` exist in `python/rlox/logging.py` but are stubs that only log `on_train_step`. Need richer integration.

**Rust-side changes**: None.

**Python-side changes**:
- Extend `LoggerCallback`:
  - `on_train_step(step, metrics)` -- already exists.
  - `on_rollout_end(step, metrics)` -- add rollout stats (mean_reward, mean_ep_length).
  - `on_eval(step, metrics)` -- already exists.
  - `on_save(step, path)` -- log checkpoint events.
- `WandbLogger`:
  - Log hyperparameters via `wandb.config.update(config.to_dict())` in `__init__`.
  - Add `finish()` method calling `wandb.finish()`.
- `TensorBoardLogger`:
  - Add `flush()` and `close()` methods.
  - Log histograms of advantages, returns per rollout (optional, behind a `log_histograms=True` flag).
- Add `CSVLogger` for lightweight local logging (no external deps).

**TDD specification**:
```
# tests/test_logging.py
test_csv_logger_creates_file
    - CSVLogger(path=tmpfile).on_train_step(0, {"loss": 0.5})
    - File exists with header and one data row

test_csv_logger_appends_rows
    - Log 10 steps, file has 10 data rows

test_wandb_logger_init_requires_wandb
    - Without wandb installed: ImportError

test_tensorboard_logger_writes_events
    - TensorBoardLogger(log_dir=tmpdir).on_train_step(0, {"loss": 0.5})
    - Events file exists in tmpdir

test_logger_callback_interface
    - LoggerCallback() methods are all no-ops (base class)
```

**Benchmark gate**: None (logging should have negligible overhead if only called per-update, not per-step).

**Profiling checkpoint**: None.

**Dependencies**: 7.4.2 (callbacks wired so loggers receive data).

---

#### 7.5.2 Prebuilt Wheels (Maturin CI)

**Problem**: Users must compile from source. Need CI to build and publish wheels.

**Rust-side changes**:
- Verify `Cargo.toml` and `pyproject.toml` have correct maturin configuration.
- Ensure `abi3` compatibility (PyO3 `abi3-py38` feature).

**Python-side changes**: None.

**CI/Infra changes**:
- `.github/workflows/wheels.yml`:
  - Matrix: `{os: [ubuntu-latest, macos-latest, windows-latest], python: [3.9, 3.10, 3.11, 3.12]}`.
  - Use `maturin build --release` for each target.
  - Upload to PyPI (or TestPyPI) on tagged releases.
  - Run `pytest tests/ -x` on each built wheel to verify.

**TDD specification**:
```
# CI-level tests (run in workflow)
test_wheel_installs_cleanly
    - pip install dist/*.whl in fresh venv -> success
test_wheel_import_works
    - python -c "import rlox; print(rlox.__version__)" -> no error
test_wheel_cartpole_step
    - python -c "import rlox; e = rlox.CartPole(); e.reset(); e.step(0)" -> no error
```

**Benchmark gate**: None (CI concern).

**Profiling checkpoint**: None.

**Dependencies**: All Phase 7 code changes should be complete before cutting wheels.

---

## Phase 8: Production Hardening

### Sprint 8.1: Layer 2 Trainer API

#### 8.1.1 One-Liner SB3-Style Trainer

**Problem**: Current `PPOTrainer` is thin. Want a truly one-line API: `rlox.train("CartPole-v1", algo="ppo", total_timesteps=1_000_000)`.

**Rust-side changes**: None.

**Python-side changes**:
- Add `python/rlox/train.py`:
  ```python
  def train(
      env: str,
      algo: str = "ppo",
      total_timesteps: int = 1_000_000,
      config: dict | None = None,
      callbacks: list[Callback] | None = None,
      logger: LoggerCallback | None = None,
      seed: int = 42,
  ) -> dict[str, float]:
      """One-liner training interface."""
  ```
- Auto-selects algorithm class, config, policy based on `algo` and env's action space.
- Add `rlox.evaluate(model, env, n_episodes=10)` utility.
- Update trainers to support `learn(total_timesteps)` as alias for `train()` (SB3 compat).

**TDD specification**:
```
# tests/test_train_api.py
test_train_cartpole_ppo
    - rlox.train("CartPole-v1", algo="ppo", total_timesteps=10_000)
    - Returns dict with "mean_reward"

test_train_pendulum_sac
    - rlox.train("Pendulum-v1", algo="sac", total_timesteps=5_000)

test_train_unknown_algo_raises
    - rlox.train("CartPole-v1", algo="nonexistent") raises ValueError

test_train_with_config_overrides
    - rlox.train("CartPole-v1", config={"n_envs": 2, "n_steps": 32})

test_evaluate_returns_mean_reward
    - model = PPO("CartPole-v1"); model.train(5000)
    - mean_reward = rlox.evaluate(model, "CartPole-v1", n_episodes=5)
    - isinstance(mean_reward, float)
```

**Benchmark gate**: None (wrapper layer).

**Profiling checkpoint**: None.

**Dependencies**: Phase 7 complete (all algorithms working, callbacks wired).

---

### Sprint 8.2: Config System and Evaluation

#### 8.2.1 Composable Config System (YAML + CLI Merge)

**Problem**: Configs are code-only. Users want YAML files with CLI overrides (like Hydra but lighter).

**Rust-side changes**: None.

**Python-side changes**:
- New file `python/rlox/config_loader.py`:
  ```python
  def load_config(yaml_path: str | None, cli_overrides: dict | None = None, algo: str = "ppo") -> PPOConfig | SACConfig | ...:
      """Load config from YAML, merge CLI overrides, validate."""
  ```
- Support `rlox train --config config.yaml --set n_envs=16 learning_rate=1e-3`.
- Add `python/rlox/cli.py` with `click` or `argparse` for CLI entry point.
- Support config inheritance: `base.yaml` + `env_specific.yaml` merged.

**TDD specification**:
```
# tests/test_config_loader.py
test_load_yaml_config
    - Write minimal YAML to tmpfile, load_config returns correct PPOConfig

test_cli_overrides_yaml
    - YAML has n_envs=8, override n_envs=16 -> result.n_envs == 16

test_unknown_keys_in_yaml_ignored
    - YAML with extra key "foo" -> no error, key not in config

test_invalid_value_raises
    - YAML with n_envs=-1 -> ValueError

test_config_inheritance
    - base.yaml has gamma=0.99, child.yaml inherits and overrides n_envs=32
```

**Benchmark gate**: None.

**Profiling checkpoint**: None.

**Dependencies**: 7.4.1 (config consolidation).

---

#### 8.2.2 Statistical Evaluation Toolkit

**Problem**: RL results need proper statistical reporting -- IQM, bootstrap confidence intervals, not just mean reward.

**Rust-side changes**:
- New file `crates/rlox-core/src/training/stats.rs`:
  ```rust
  /// Interquartile mean of a slice.
  pub fn iqm(data: &[f64]) -> f64 { ... }

  /// Bootstrap confidence interval (percentile method).
  pub fn bootstrap_ci(data: &[f64], n_bootstrap: usize, ci: f64, rng: &mut ChaCha8Rng) -> (f64, f64) { ... }

  /// Stratified bootstrap for comparing two algorithms.
  pub fn paired_bootstrap_test(a: &[f64], b: &[f64], n_bootstrap: usize, rng: &mut ChaCha8Rng) -> f64 { ... }
  ```
- Add PyO3 bindings: `compute_iqm`, `compute_bootstrap_ci`, `compute_paired_bootstrap`.

**Python-side changes**:
- New file `python/rlox/evaluation.py`:
  ```python
  def evaluate_policy(algo, env_id, n_episodes=100, seeds=range(10)) -> EvalResult: ...

  @dataclass
  class EvalResult:
      mean: float
      std: float
      iqm: float
      ci_lower: float
      ci_upper: float
      raw_returns: np.ndarray
  ```

**TDD specification**:
```
# Rust (crates/rlox-core/src/training/stats.rs)
test_iqm_uniform_data
    - iqm([1, 2, 3, 4, 5, 6, 7, 8]) == mean of [3, 4, 5, 6] == 4.5

test_iqm_single_element
    - iqm([5.0]) == 5.0

test_iqm_empty_returns_nan
    - iqm([]).is_nan()

test_bootstrap_ci_contains_mean
    - For normal data, 95% CI contains true mean

test_bootstrap_ci_deterministic_with_seed
    - Same seed -> same CI

proptest_iqm_between_min_and_max
    - For any Vec<f64>, iqm is between min and max (or NaN for empty)

# Python
test_evaluate_policy_returns_eval_result
    - evaluate_policy(ppo, "CartPole-v1", n_episodes=10)
    - Returns EvalResult with all fields populated
```

**Benchmark gate**: `compute_iqm` on 10k samples must be < 1ms. `compute_bootstrap_ci` with 10k bootstrap samples must be < 50ms.

**Profiling checkpoint**: None (should be trivially fast).

**Dependencies**: None.

---

### Sprint 8.3: Diagnostics and Native Envs

#### 8.3.1 Training Diagnostics

**Problem**: Users need automated detection of entropy collapse, KL spikes, gradient explosion -- common failure modes in RL training.

**Rust-side changes**: None (diagnostics are stateful monitors in Python).

**Python-side changes**:
- New file `python/rlox/diagnostics.py`:
  ```python
  class TrainingDiagnostics(Callback):
      """Monitor training health and warn on pathological signals."""

      def __init__(
          self,
          entropy_threshold: float = 0.01,
          kl_spike_threshold: float = 0.1,
          grad_norm_threshold: float = 10.0,
          window_size: int = 100,
      ): ...

      def on_train_batch(self, **kwargs) -> None:
          # Check entropy collapse
          # Check KL divergence spike
          # Check gradient norm explosion
          # Emit warnings via logging.warning()

      def report(self) -> dict[str, Any]:
          """Return diagnostic summary."""
  ```
- Integrate as a default callback (opt-out via `diagnostics=False` in trainer).

**TDD specification**:
```
# tests/test_diagnostics.py
test_entropy_collapse_warning
    - Feed entropy values [1.0, 0.5, 0.1, 0.001]
    - DiagnosticsCallback emits warning when entropy < threshold

test_kl_spike_detection
    - Feed approx_kl values [0.01, 0.01, 0.5, 0.01]
    - Detects the spike at index 2

test_gradient_explosion_detection
    - Feed grad_norms [1.0, 2.0, 100.0]
    - Detects explosion at index 2

test_report_returns_summary
    - After feeding data, report() has keys: n_entropy_warnings, n_kl_spikes, etc.

test_no_warnings_on_healthy_training
    - Feed healthy values -> no warnings emitted
```

**Benchmark gate**: None (monitoring overhead is negligible).

**Profiling checkpoint**: None.

**Dependencies**: 7.4.2 (callbacks wired).

---

#### 8.3.2 MuJoCo Native Rust Bindings

**Problem**: GymVecEnv steps MuJoCo through Python (GIL), limiting throughput. Native Rust bindings bypass this.

**Rust-side changes**:
- New crate: `crates/rlox-mujoco/` (optional workspace member).
- Depend on `mujoco-rs` (Rust bindings to MuJoCo C API) or raw FFI to `libmujoco.so`.
- Implement `RLEnv` for a set of standard envs: `HalfCheetah`, `Hopper`, `Walker2d`, `Ant`, `Humanoid`.
  ```rust
  pub struct MuJoCoEnv {
      model: MjModel,
      data: MjData,
      max_steps: usize,
      step_count: usize,
  }
  impl RLEnv for MuJoCoEnv { ... }
  ```
- Add to `PyVecEnv::new` match arms for `"HalfCheetah-v4"`, etc.
- Add PyO3 bindings: `PyMuJoCoEnv`.

**Python-side changes**:
- Update `RolloutCollector` to detect when Rust VecEnv supports the env natively.
- Optional feature flag in `pyproject.toml`: `rlox[mujoco]`.

**TDD specification**:
```
# Rust (crates/rlox-mujoco)
test_halfcheetah_reset_obs_dim
    - HalfCheetah::new().reset() returns obs of length 17

test_halfcheetah_step_returns_reward
    - Step with zero action -> reward is finite

test_halfcheetah_action_space_is_box
    - action_space() is ActionSpace::Box { shape: [6], ... }

test_halfcheetah_terminates_on_unhealthy
    - Step with large actions until terminated == true

test_vecenv_halfcheetah_parallel
    - VecEnv with 16 HalfCheetah envs, step_all -> shapes correct

proptest_mujoco_step_no_nan
    - For arbitrary valid continuous actions, obs never contains NaN

# Python
test_pyvecenv_halfcheetah_native
    - VecEnv(4, env_id="HalfCheetah-v4") creates native envs
    - step_all with float actions works

test_native_mujoco_faster_than_gym
    - Benchmark: native VecEnv vs GymVecEnv for HalfCheetah
    - Native should be > 5x faster
```

**Benchmark gate**:
- New benchmark: `benchmarks/bench_mujoco_stepping.py` -- compare Rust-native vs GymVecEnv for HalfCheetah at 64/256/1024 envs.
- Criterion benchmark: `crates/rlox-bench/benches/mujoco_stepping.rs`.

**Profiling checkpoint**: `cargo flamegraph` on 1000 steps of 256 HalfCheetah envs. Identify if MuJoCo simulation or Rayon overhead dominates.

**Dependencies**: None (independent module, but benefits from 7.1.2 BatchSteppable trait).

---

### Sprint 8.4: Memory-Mapped Buffers and Reward Sandbox

#### 8.4.1 Memory-Mapped Buffer Overflow

**Problem**: Large replay buffers (10M+ transitions, Atari-scale) may exceed RAM. Need mmap fallback.

**Rust-side changes**:
- New file `crates/rlox-core/src/buffer/mmap.rs`:
  ```rust
  pub struct MmapReplayBuffer {
      obs_mmap: memmap2::MmapMut,
      actions_mmap: memmap2::MmapMut,
      // ...
      capacity: usize,
      len: usize,
      write_pos: usize,
      obs_dim: usize,
      act_dim: usize,
  }
  ```
- Implement same API as `ReplayBuffer`: `push`, `sample`, `len`.
- Use `memmap2` crate for portable memory-mapped files.
- Auto-fallback: `ReplayBuffer::new_auto(capacity, obs_dim, act_dim, mmap_threshold: usize)` -- if `capacity * record_size > mmap_threshold`, use mmap.

**Python-side changes**:
- Add `PyMmapReplayBuffer` or extend `PyReplayBuffer` with `mmap_path` parameter.
- If `mmap_path` is provided, use mmap backend.

**TDD specification**:
```
# Rust
test_mmap_buffer_push_sample_roundtrip
    - Push 100 records, sample 10 -> data matches

test_mmap_buffer_ring_overwrites
    - capacity=10, push 15 -> len is 10, oldest overwritten

test_mmap_buffer_deterministic_sampling
    - Same seed -> same sample

test_mmap_buffer_file_created
    - MmapReplayBuffer::new(path, ...) creates file on disk

test_mmap_buffer_cleanup_on_drop
    - After drop, file is deleted (if temp flag set)

proptest_mmap_sample_indices_in_bounds
    - For any capacity/len, sampled indices are valid

# Python
test_pymmap_buffer_push_sample
    - PyReplayBuffer(1000, 4, 1, mmap_path=tmpdir) works like in-memory

test_mmap_large_buffer_does_not_oom
    - Create buffer with 10M capacity, obs_dim=84*84*4 -- should not OOM
    - (Skip on CI if insufficient disk)
```

**Benchmark gate**:
- Compare mmap vs in-memory `ReplayBuffer` for push/sample at 1M capacity:
  - Push: mmap should be within 2x of in-memory.
  - Sample: mmap should be within 3x (page fault overhead acceptable).

**Profiling checkpoint**: `cargo flamegraph` on 10k sample operations from a 1M-entry mmap buffer. Look for page fault stalls.

**Dependencies**: None.

---

#### 8.4.2 Verifiable Reward Sandbox

**Problem**: Custom reward functions execute arbitrary user code. For safety-critical applications, need isolation.

**Rust-side changes**:
- New file `crates/rlox-core/src/training/sandbox.rs`:
  ```rust
  pub enum RewardSandbox {
      /// No sandboxing -- direct function call.
      None,
      /// WASM-based sandbox using wasmtime.
      Wasm { engine: wasmtime::Engine, module: wasmtime::Module },
      /// Process isolation via subprocess.
      Process { command: String },
  }

  impl RewardSandbox {
      pub fn evaluate(&self, obs: &[f32], action: &[f32], reward: f64) -> Result<f64, RloxError> { ... }
  }
  ```
- WASM sandbox: compile reward function to WASM, execute in wasmtime with memory limits.
- Process sandbox: serialize obs/action/reward, send to subprocess via pipe, deserialize result.

**Python-side changes**:
- `python/rlox/reward_sandbox.py`:
  ```python
  class WasmRewardFn:
      def __init__(self, wasm_path: str, memory_limit_mb: int = 64): ...
      def __call__(self, obs, actions, rewards) -> np.ndarray: ...
  ```

**TDD specification**:
```
# Rust
test_wasm_sandbox_basic_reward
    - Load a simple WASM that returns reward * 2
    - evaluate(obs, action, 1.0) == 2.0

test_wasm_sandbox_memory_limit
    - WASM that allocates > limit -> error, not crash

test_process_sandbox_basic
    - Simple echo subprocess -> returns correct reward

test_process_sandbox_timeout
    - Subprocess that sleeps forever -> timeout error

# Python
test_wasm_reward_fn_callable
    - WasmRewardFn(path) is callable, returns ndarray
```

**Benchmark gate**: WASM reward function overhead must be < 10us per call (so it doesn't dominate env stepping).

**Profiling checkpoint**: Profile WASM sandbox overhead vs native function call.

**Dependencies**: None (optional feature).

---

## Phase 9: Distributed and Scale

### Sprint 9.1: Decoupled Pipeline

#### 9.1.1 Decoupled Collection/Training Pipeline

**Problem**: Collection and training are synchronous. While the GPU trains, envs idle.

**Rust-side changes**:
- New file `crates/rlox-core/src/training/pipeline.rs`:
  ```rust
  use crossbeam_channel::{bounded, Sender, Receiver};

  pub struct RolloutPipeline {
      tx: Sender<RolloutData>,
      rx: Receiver<RolloutData>,
  }

  pub struct RolloutData {
      pub obs: Vec<f32>,        // flat [n_envs * n_steps, obs_dim]
      pub actions: Vec<f32>,
      pub rewards: Vec<f64>,
      pub dones: Vec<f64>,
      pub log_probs: Vec<f32>,
      pub values: Vec<f32>,
      pub advantages: Vec<f64>,
      pub returns: Vec<f64>,
  }

  impl RolloutPipeline {
      pub fn new(buffer_size: usize) -> Self { ... }
      pub fn collector_handle(&self) -> CollectorHandle { ... }
      pub fn trainer_handle(&self) -> TrainerHandle { ... }
  }
  ```
- Collector thread: steps envs, computes GAE, sends `RolloutData` to channel.
- Trainer thread: receives data, performs SGD updates.
- Use `crossbeam_channel::bounded(2)` for backpressure (trainer can fall behind by 2 rollouts max).

**Python-side changes**:
- New file `python/rlox/async_trainer.py`:
  ```python
  class AsyncPPOTrainer:
      """PPO with decoupled collection and training."""
      def __init__(self, env_id, n_collection_threads=1, **kwargs): ...
      def train(self, total_timesteps): ...
  ```
- Uses Rust pipeline internally, Python only does the SGD step.

**TDD specification**:
```
# Rust
test_pipeline_send_receive
    - Send RolloutData through channel, receive on other end, data matches

test_pipeline_backpressure
    - bounded(1), send 2 items -> second blocks until first consumed

test_pipeline_multi_producer
    - 4 collector threads sending to 1 trainer -> all data received

# Python
test_async_ppo_trainer_cartpole
    - AsyncPPOTrainer("CartPole-v1").train(20_000) -> returns metrics

test_async_trainer_faster_than_sync
    - For n_envs=64, async should have higher throughput (steps/sec)
```

**Benchmark gate**:
- New benchmark: `benchmarks/bench_async_pipeline.py` -- compare sync vs async PPO throughput at 64/256 envs.
- Target: > 20% throughput improvement for 256 envs (collection overlaps with training).

**Profiling checkpoint**: `py-spy` on async trainer to verify collection and training threads overlap. Check for GIL contention.

**Dependencies**: Phase 7 complete (working PPO, GAE, VecEnv).

---

### Sprint 9.2: gRPC Workers

#### 9.2.1 gRPC Distributed Env Workers

**Problem**: Single-machine Rayon parallelism has limits. Need to distribute env stepping across machines.

**Rust-side changes**:
- New crate: `crates/rlox-distributed/` with `tonic` (gRPC).
- Proto definition `proto/env_worker.proto`:
  ```protobuf
  service EnvWorker {
      rpc Step(StepRequest) returns (StepResponse);
      rpc Reset(ResetRequest) returns (ResetResponse);
      rpc GetSpaces(Empty) returns (SpacesResponse);
  }
  ```
- Server: wraps a `VecEnv` on the worker machine.
- Client: implements `BatchSteppable` trait, forwards calls via gRPC.
  ```rust
  pub struct RemoteVecEnv {
      clients: Vec<EnvWorkerClient>,
      // Each client manages a shard of environments
  }
  impl BatchSteppable for RemoteVecEnv { ... }
  ```

**Python-side changes**:
- `python/rlox/distributed.py`:
  ```python
  class DistributedVecEnv:
      """Connect to remote env workers."""
      def __init__(self, worker_addrs: list[str]): ...
      def step_all(self, actions) -> dict: ...
      def reset_all(self, seed=None) -> np.ndarray: ...
  ```
- CLI: `rlox-worker --env HalfCheetah-v4 --n-envs 64 --port 50051`.

**TDD specification**:
```
# Rust
test_grpc_step_response_shape
    - Start local server with 4 CartPole envs
    - Client sends StepRequest with 4 actions
    - Response has 4 obs, 4 rewards, 4 terminated, 4 truncated

test_grpc_reset_deterministic
    - Reset with same seed -> same obs

test_remote_vecenv_batch_steppable
    - RemoteVecEnv implements BatchSteppable
    - step_batch returns correct shapes

test_multi_worker_aggregation
    - 2 workers with 4 envs each -> RemoteVecEnv.num_envs() == 8

# Python
test_distributed_vecenv_step
    - DistributedVecEnv(["localhost:50051"]).step_all(actions) works
```

**Benchmark gate**:
- New benchmark: `benchmarks/bench_distributed_stepping.py` -- measure overhead of gRPC for local loopback vs direct VecEnv.
- Acceptable overhead: < 500us per step for local, < 2ms for LAN.
- Measure scaling: 1/2/4/8 workers on same machine.

**Profiling checkpoint**: Measure gRPC serialization/deserialization overhead. For Atari-sized obs (84x84x4), check if protobuf serialization dominates.

**Dependencies**: 7.1.2 (BatchSteppable trait).

---

### Sprint 9.3: LLM Serving Integration

#### 9.3.1 Multi-GPU Training Composition

**Problem**: Single-GPU training does not scale for large models. Need to compose with PyTorch DDP/FSDP.

**Rust-side changes**: None (Rust handles data, Python handles distributed training).

**Python-side changes**:
- New file `python/rlox/distributed_training.py`:
  ```python
  class DistributedPPO:
      """PPO with DDP for multi-GPU policy training."""
      def __init__(self, env_id, world_size=None, **kwargs): ...
  ```
- Use `torch.distributed` for gradient sync.
- Each rank collects from its own VecEnv shard.
- Centralized advantage normalization across ranks.

**TDD specification**:
```
# tests/test_distributed_training.py (skip if CUDA unavailable)
test_distributed_ppo_single_gpu
    - DistributedPPO("CartPole-v1", world_size=1).train(5000) works

test_advantage_normalization_across_ranks
    - Mock 2-rank setup, verify advantages are normalized globally
```

**Benchmark gate**: None (requires multi-GPU hardware).

**Profiling checkpoint**: If multi-GPU available, profile gradient sync overhead vs compute.

**Dependencies**: 7.3.1 (working PPO).

---

#### 9.3.2 vLLM/TGI/SGLang Integration

**Problem**: GRPO needs fast LLM inference for generation. Need async Rust client for serving engines.

**Rust-side changes**:
- New file `crates/rlox-core/src/llm/serving.rs`:
  ```rust
  use reqwest::Client;

  pub struct LLMClient {
      client: Client,
      base_url: String,
      api_type: ApiType,
  }

  pub enum ApiType { VLLM, TGI, SGLang, OpenAI }

  impl LLMClient {
      pub async fn generate(&self, prompts: &[String], params: &GenParams) -> Result<Vec<String>, RloxError> { ... }
      pub async fn logprobs(&self, sequences: &[String]) -> Result<Vec<Vec<f64>>, RloxError> { ... }
  }
  ```
- Async client using `reqwest` + `tokio`.
- Batch requests with configurable concurrency.

**Python-side changes**:
- New file `python/rlox/llm_serving.py`:
  ```python
  class LLMServingClient:
      """Client for vLLM/TGI/SGLang serving endpoints."""
      def __init__(self, url, api_type="vllm"): ...
      def generate(self, prompts, **kwargs) -> list[str]: ...
      def logprobs(self, sequences) -> list[list[float]]: ...
  ```
- Integration with GRPO: replace `model.generate()` with serving client calls.

**TDD specification**:
```
# Rust
test_llm_client_construct
    - LLMClient::new("http://localhost:8000", ApiType::VLLM) does not panic

test_llm_client_generate_mock
    - Mock HTTP server returns completions
    - client.generate(["hello"]).await returns expected text

test_llm_client_handles_timeout
    - Mock server with 10s delay, client timeout=1s -> error

test_llm_client_batch_concurrency
    - 100 prompts with max_concurrency=10 -> all complete

# Python
test_serving_client_generate
    - Mock server, LLMServingClient.generate(["hello"]) returns list[str]

test_grpo_with_serving_client
    - GRPO with LLMServingClient instead of local model -> train_step works
```

**Benchmark gate**:
- New benchmark: `benchmarks/bench_llm_serving.py` -- measure generation throughput via serving vs local model.
- Latency per prompt should be < 100ms for short completions on local vLLM.

**Profiling checkpoint**: Profile async request batching. Check if HTTP overhead or serialization dominates for small batches.

**Dependencies**: 7.3.2 (GRPO working).

---

#### 9.3.3 Reward Model Serving

**Problem**: GRPO/RLHF need reward model inference. Should be served separately for scale.

**Rust-side changes**:
- Extend `LLMClient` with reward model endpoint support.
- Add `reward_score` method: takes (prompt, completion) pairs, returns scores.

**Python-side changes**:
- `python/rlox/reward_serving.py`:
  ```python
  class RewardModelClient:
      """Client for reward model serving."""
      def __init__(self, url): ...
      def score(self, prompts, completions) -> list[float]: ...
  ```
- Integrate as `reward_fn` for GRPO.

**TDD specification**:
```
# tests/test_reward_serving.py
test_reward_client_score
    - Mock server returns scores
    - client.score(["prompt"], ["completion"]) -> [0.8]

test_reward_client_batch
    - 100 pairs scored in batch -> correct length

test_grpo_with_reward_server
    - GRPO with RewardModelClient as reward_fn -> train_step works
```

**Benchmark gate**: Reward scoring latency < 50ms per batch of 64 pairs (local server).

**Profiling checkpoint**: None.

**Dependencies**: 9.3.2 (LLM client infrastructure).

---

### Sprint 9.4: Multi-Agent and World Models

#### 9.4.1 MAPPO / QMIX Multi-Agent

**Problem**: `python/rlox/algorithms/mappo.py` exists as a stub. Need working implementation with shared/independent critics.

**Rust-side changes**:
- New file `crates/rlox-core/src/env/multi_agent.rs`:
  ```rust
  pub struct MultiAgentTransition {
      pub obs: Vec<Vec<f32>>,       // per-agent obs
      pub rewards: Vec<f64>,         // per-agent rewards
      pub terminated: Vec<bool>,
      pub truncated: Vec<bool>,
      pub global_state: Option<Vec<f32>>,  // for QMIX centralized critic
  }

  pub trait MultiAgentEnv: Send + Sync {
      fn step(&mut self, actions: &[Action]) -> Result<MultiAgentTransition, RloxError>;
      fn reset(&mut self, seed: Option<u64>) -> Result<Vec<Observation>, RloxError>;
      fn n_agents(&self) -> usize;
      fn agent_action_space(&self, agent_id: usize) -> &ActionSpace;
      fn agent_obs_space(&self, agent_id: usize) -> &ObsSpace;
  }
  ```
- Add `compute_mappo_gae` that handles per-agent advantages with centralized value.
- PyO3 bindings for `MultiAgentEnv`.

**Python-side changes**:
- Implement `MAPPO` in `python/rlox/algorithms/mappo.py`:
  - Shared parameters with per-agent heads (parameter sharing option).
  - Centralized critic with global state.
  - Per-agent GAE computation using `rlox.compute_gae`.
- Add QMIX mixing network.

**TDD specification**:
```
# Rust
test_multi_agent_transition_shape
    - 3 agents -> obs has 3 entries, rewards has 3

test_mappo_gae_per_agent
    - compute_mappo_gae with 2 agents -> advantages shape matches

# Python
test_mappo_simple_env
    - MAPPO on a 2-agent cooperative env -> trains without error

test_mappo_shared_parameters
    - MAPPO(parameter_sharing=True) uses same network for all agents

test_qmix_mixing_network
    - QMIX mixing network combines per-agent Q-values -> scalar
```

**Benchmark gate**: MAPPO GAE computation should be within 2x of single-agent GAE (per agent).

**Profiling checkpoint**: Profile MAPPO rollout collection. Check if per-agent obs handling creates excessive allocations.

**Dependencies**: 7.1.2 (BatchSteppable), 7.3.1 (working PPO foundation).

---

#### 9.4.2 DreamerV3 World Model

**Problem**: `python/rlox/algorithms/dreamer.py` exists as a stub. DreamerV3 needs a world model with RSSM, image encoder, and imagination-based training.

**Rust-side changes**:
- New file `crates/rlox-core/src/buffer/sequence.rs`:
  ```rust
  pub struct SequenceReplayBuffer {
      // Stores fixed-length sequences for world model training
      sequences: Vec<Sequence>,
      capacity: usize,
  }

  pub struct Sequence {
      pub obs: Vec<Vec<f32>>,
      pub actions: Vec<Vec<f32>>,
      pub rewards: Vec<f64>,
      pub dones: Vec<bool>,
  }
  ```
- Efficient sequence sampling with `ChaCha8Rng`.

**Python-side changes**:
- Full DreamerV3 implementation:
  - RSSM (Recurrent State Space Model).
  - Image encoder/decoder (CNN).
  - Actor-critic in imagination (lambda returns).
  - Symlog predictions.
  - Free bits KL balancing.

**TDD specification**:
```
# Rust
test_sequence_buffer_stores_sequences
    - Push 5 sequences of length 50, sample 2 -> correct shape

test_sequence_buffer_ring_behavior
    - Capacity 10, push 15 -> oldest dropped

# Python
test_rssm_forward
    - RSSM processes (obs, action) sequence -> latent states

test_dreamer_train_step
    - DreamerV3 train_step on batch of sequences -> loss dict

test_dreamer_imagination
    - Actor imagines 15 steps in latent space -> values computed
```

**Benchmark gate**: Sequence buffer push/sample at 10k sequences, length 50, obs_dim=1024. Must be < 10ms per sample batch of 32.

**Profiling checkpoint**: Profile RSSM forward pass and imagination rollout with `py-spy`.

**Dependencies**: None (independent module, but benefits from buffer infrastructure).

---

### Sprint 9.5: Async Training and Stable API

#### 9.5.1 IMPALA / V-trace Async Training

**Problem**: `python/rlox/algorithms/impala.py` is a stub. Need async actor-learner architecture using V-trace.

**Rust-side changes**:
- V-trace already exists: `crates/rlox-core/src/training/vtrace.rs`.
- Add `crates/rlox-core/src/training/impala_queue.rs`:
  ```rust
  pub struct IMPALAQueue {
      queue: crossbeam_channel::Receiver<ActorBatch>,
      n_actors: usize,
  }

  pub struct ActorBatch {
      pub obs: Vec<f32>,
      pub actions: Vec<f32>,
      pub rewards: Vec<f64>,
      pub dones: Vec<f64>,
      pub behavior_log_probs: Vec<f32>,
      pub actor_id: usize,
  }
  ```

**Python-side changes**:
- Full IMPALA implementation:
  - Multiple actor processes collecting experience.
  - Central learner with V-trace off-policy correction.
  - Model weight broadcasting from learner to actors.

**TDD specification**:
```
# Rust
test_impala_queue_receives_batches
    - 4 actors send batches, queue receives all 4

test_vtrace_with_stale_policy
    - Compute V-trace where behavior != target policy
    - Verify importance weights are clipped

# Python
test_impala_cartpole_trains
    - IMPALA("CartPole-v1", n_actors=2).train(20_000) -> metrics

test_impala_vtrace_correction
    - After training, verify V-trace was applied (behavior != target log_probs)
```

**Benchmark gate**:
- New benchmark: `benchmarks/bench_impala.py` -- compare IMPALA throughput (steps/sec) vs sync PPO at 256 envs.
- IMPALA should achieve higher throughput due to async collection.

**Profiling checkpoint**: Profile actor-learner communication overhead. Check if model weight broadcasting is a bottleneck.

**Dependencies**: 9.1.1 (decoupled pipeline).

---

#### 9.5.2 Stable API 1.0

**Problem**: Before 1.0, need API stability guarantees, deprecation policy, and comprehensive documentation.

**Rust-side changes**:
- Audit all `pub` items in `rlox-core`, `rlox-nn`:
  - Mark unstable APIs with `#[doc(hidden)]` or move to `internal` module.
  - Ensure all public types implement `Debug`, `Clone` where appropriate.
  - Add `#[non_exhaustive]` to enums that may gain variants (`RloxError`, `ActionSpace`, `ObsSpace`).
- Semantic versioning: bump to `1.0.0`.

**Python-side changes**:
- Audit `__all__` exports in all modules.
- Add deprecation warnings for any APIs that will change.
- Add `python/rlox/version.py` with version info.
- Comprehensive type stubs in `python/rlox/_rlox_core.pyi` (update existing).

**TDD specification**:
```
# Rust
test_all_public_types_are_debug
    - Compile-time assertions that all pub types implement Debug

test_error_enum_is_non_exhaustive
    - Verify #[non_exhaustive] on RloxError

# Python
test_version_accessible
    - rlox.__version__ == "1.0.0"

test_all_exports_importable
    - for name in rlox.__all__: getattr(rlox, name) does not raise

test_type_stubs_complete
    - mypy --strict on a test file that uses all public APIs -> no errors
```

**Benchmark gate**: Full benchmark suite must pass with no regressions from Phase 7 baseline:
- GAE: still > 100x vs numpy.
- Buffer push: still > 50x vs TorchRL.
- Buffer sample: still > 5x vs both.
- E2E rollout: still > 2.5x vs SB3.

**Profiling checkpoint**: Final `cargo flamegraph` on all hot paths. Document performance characteristics in API docs.

**Dependencies**: All Phases 7-9 features complete.

---

## Dependency Graph Summary

```
7.1.1 (fix PyVecEnv) ─────────────┐
7.1.2 (BatchSteppable) ───────────┼─> 7.1.3 (GymVecEnv) ──> 7.2.2 (continuous collector)
                                   │                              │
7.2.1 (ContinuousPolicy) ─────────┼──────────────────────────────┤
                                   │                              v
                                   │                     7.3.1 (PPO MuJoCo) ──> 9.3.1 (multi-GPU)
                                   │
7.3.2 (GRPO E2E) ─────────────────┼──> 9.3.2 (vLLM) ──> 9.3.3 (reward serving)
                                   │
7.4.1 (config) ──> 7.4.2 (callbacks) ──> 7.4.3 (checkpoint) ──> 7.5.1 (logging)
                          │
                          └──> 8.3.1 (diagnostics)

7.3.3 (reward_fn) ─── standalone
7.5.2 (CI wheels) ─── after all Phase 7

8.1.1 (trainer API) ─── after Phase 7
8.2.1 (YAML config) ─── after 7.4.1
8.2.2 (eval toolkit) ─── standalone
8.3.2 (MuJoCo native) ─── standalone (benefits from 7.1.2)
8.4.1 (mmap buffer) ─── standalone
8.4.2 (reward sandbox) ─── standalone

9.1.1 (async pipeline) ──> 9.5.1 (IMPALA)
9.2.1 (gRPC workers) ─── after 7.1.2
9.4.1 (MAPPO) ─── after 7.1.2, 7.3.1
9.4.2 (DreamerV3) ─── standalone
9.5.2 (stable API) ─── after everything
```

---

## Cross-Cutting Concerns

### Error Handling Convention
All new Rust code must use `Result<T, RloxError>`. Add new error variants to `RloxError` as needed:
```rust
#[error("Configuration error: {0}")]
ConfigError(String),

#[error("Serialization error: {0}")]
SerializationError(String),

#[error("Network error: {0}")]
NetworkError(String),

#[error("Sandbox error: {0}")]
SandboxError(String),
```

### Determinism Convention
- All sampling uses `ChaCha8Rng` seeded from `derive_seed(master_seed, index)`.
- New components must accept `seed: Option<u64>` and be reproducible when seeded.
- Proptest for any math/sampling code.

### Precision Convention
- `f64` for: GAE, advantages, returns, rewards, statistics (IQM, bootstrap CI).
- `f32` for: observations, actions, neural network I/O, TensorData.
- Never mix without explicit cast at the boundary.

### Performance Regression Policy
After each sprint, run the full Criterion benchmark suite:
```bash
cargo bench --package rlox-bench
```
And the Python benchmarks:
```bash
python benchmarks/run_all.py
```
Compare against the Phase 6 baselines. Any regression > 5% must be investigated and either fixed or documented with justification.

### Profiling Schedule
| Sprint | Tool | Target |
|--------|------|--------|
| 7.2 | py-spy | GymVecEnv GIL overhead |
| 7.3 | cargo flamegraph | GAE in continuous rollouts |
| 7.3 | py-spy | PPO training loop |
| 8.3 | cargo flamegraph | MuJoCo stepping |
| 8.4 | cargo flamegraph | mmap buffer page faults |
| 9.1 | py-spy | async pipeline overlap |
| 9.2 | tokio-console | gRPC serialization |
| 9.5 | cargo flamegraph | final hot path audit |
