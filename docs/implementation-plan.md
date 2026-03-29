# rlox Implementation Plan — Remaining Work

## Context

After completing 14 Rust latency optimizations, 5 Python quick wins, algorithm bug fixes (SAC/TD3/DQN/PPO), and a runner refactor, this document captures all remaining work identified across correctness, infrastructure, usability, and testing.

---

## P0 — Production-Blocking (Fix Before Release) — ALL DONE

### P0.1: IMPALA V-trace ignores episode boundaries — DONE

**File:** `python/rlox/algorithms/impala.py` lines 230-248

**Problem:** `_learner_step()` computes `all_dones` but never passes them to `rlox.compute_vtrace()`. The `bootstrap_value=0.0` is hardcoded instead of computing `V(s_T)`. Rewards leak across episode boundaries.

**Fix:**
1. After the rollout loop, compute bootstrap value:
   ```python
   with torch.no_grad():
       bootstrap_val = self.local_policy.critic(final_obs).item()
   ```
2. Pass `dones` to V-trace. The Rust `compute_vtrace` doesn't currently accept dones — need to either:
   - Add a `dones` parameter to the Rust function (preferred), or
   - Zero out rewards at done boundaries before calling V-trace

3. Additionally: IMPALA hardcodes `DiscretePolicy` for all envs (line 108-118). Add continuous env detection and use `ContinuousPolicy` when appropriate.

**Testing:** Add `tests/python/test_impala.py` with:
- CartPole convergence smoke test (5000 steps, verify return > 100)
- Test that done boundaries reset advantage computation
- Test continuous env detection

**Effort:** Medium (Rust V-trace change + Python fix + tests)

---

### P0.2: DreamerV3 gradient detach breaks actor learning

**File:** `python/rlox/algorithms/dreamer.py` lines 196-222

**Problem:** `torch.no_grad()` around world model step (line 201) detaches the computation graph. The actor gradient should flow through the dynamics model per the Dreamer paper. The actor loss loop (line 216-222) also re-samples actions instead of using trajectory actions.

**Fix:**
1. Remove `torch.no_grad()` from the imagination rollout — let gradients flow through world model to actor
2. Use the same actions from the trajectory in the loss computation (don't re-sample)
3. Only detach the value targets (not the dynamics):
   ```python
   # Imagination rollout (WITH gradients for actor)
   for t in range(horizon):
       action_dist = self.actor(h)
       action = action_dist.rsample()  # reparameterized for gradient flow
       h, _, pred_r = self.world_model.step(h, action)  # NO torch.no_grad()
       imagined_rewards.append(pred_r)

   # Value targets (detached)
   with torch.no_grad():
       values = self.critic(h_stack)
   ```

**Testing:** Add `tests/python/test_dreamer.py` with:
- Verify actor loss decreases over 100 imagination steps
- Verify gradients flow from actor loss through world model parameters

**Effort:** Medium

---

### P0.3: Remove broken `python-publish.yml`

**File:** `.github/workflows/python-publish.yml`

**Problem:** Uses `python -m build` which creates a pure-Python sdist/wheel WITHOUT the Rust extension. Publishing this would give users a broken package. The correct workflow is `wheels.yml` which uses maturin.

**Fix:** Delete `python-publish.yml` entirely, or change it to trigger `wheels.yml` instead.

**Testing:** Verify `wheels.yml` runs on release tag.

**Effort:** Trivial

---

### P0.4: MAPPO critic input dimension mismatch

**File:** `python/rlox/algorithms/mappo.py` line 131, 185

**Problem:** Critic is created with `obs_dim * n_agents` input size but receives single-agent `obs_dim` observations during training. Works only when `n_agents == 1`.

**Fix:** During training, concatenate observations from all agents before passing to critic:
```python
# Collect joint observations for centralized critic
joint_obs = obs.reshape(batch_size // n_agents, n_agents * obs_dim)
values = self.critic(joint_obs).squeeze(-1)
```

**Testing:** Add `tests/python/test_mappo.py` with n_agents=2 test.

**Effort:** Low

---

### P0.5: OnlineDPO loss accumulation on CPU tensor

**File:** `python/rlox/algorithms/online_dpo.py` line 65

**Problem:** `total_loss = torch.tensor(0.0)` creates CPU tensor. Adding GPU losses silently moves computation.

**Fix:**
```python
losses = []
for prompt in batch:
    loss, _ = self.compute_loss(prompt, ...)
    losses.append(loss)
total_loss = torch.stack(losses).mean()
```

**Testing:** Add test verifying loss is on correct device.

**Effort:** Trivial

---

### P0.6: Add gradient clipping to DPO/GRPO/OnlineDPO

**Files:** `dpo.py:117`, `grpo.py:133`, `online_dpo.py:93`

**Problem:** No `max_grad_norm` clipping. LLM fine-tuning commonly diverges without it.

**Fix:** Add `max_grad_norm` parameter (default 1.0) and clip after backward:
```python
if self.max_grad_norm > 0:
    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
```

**Testing:** Verify gradients are clipped in unit test.

**Effort:** Trivial

---

## P1 — High Impact (Fix for Research Paper)

### P1.1: Add tests for 6 untested algorithms — DONE

**Test files added:**
- `tests/python/test_algorithm_bug_fixes.py` — 13 tests for IMPALA, DreamerV3, MAPPO
- `tests/python/test_llm_algorithms.py` — 34 tests for DPO, OnlineDPO, BestOfN, GRPO, MmapReplayBuffer

| Algorithm | Tests | Coverage |
|-----------|-------|----------|
| IMPALA | Bootstrap value, continuous env, training | V-trace episode boundaries, GymVecEnv fallback |
| MAPPO | Single-agent training, multi-agent guard, critic dim | NotImplementedError for n_agents>1 |
| DreamerV3 | WM frozen during AC, unfrozen after, full training | Gradient isolation verified |
| DPO | 8 tests: gradients, ref unchanged, loss math, callbacks, checkpoint, clipping | Beta values, step counting |
| OnlineDPO | 5 tests: loss, gradients, preference fn, swap, clipping | Stochastic generation |
| BestOfN | 5 tests: shape, selection, n=1, no grad, various n | Reward-based selection |
| GRPO | 9 tests: metrics, gradients, group_size=1, eval, multi-epoch, callbacks, early stop, checkpoint, KL=0 | Full lifecycle |
| MmapReplayBuffer | 7 tests: push, sample hot, spill to cold, deterministic, hot+cold, close, DQN integration | Hot/cold tiering |

---

### P1.2: Expose MmapReplayBuffer to Python — DONE

Already exposed via PyO3 in `crates/rlox-python/src/buffer.rs` (lines 421-510).
Available as `rlox.MmapReplayBuffer` in `__init__.py`. Python integration tests added.

---

### P1.3: Implement EvalCallback and CheckpointCallback

**File:** `python/rlox/callbacks.py`

**Current state:** Both are stubs — `on_step()` only increments counter.

**Fix:**

Step 1: Add `predict()` method to each algorithm:
```python
# In SAC:
def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
    with torch.no_grad():
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        if deterministic:
            action = self.actor.deterministic(obs_t).squeeze(0).numpy()
        else:
            action, _ = self.actor.sample(obs_t)
            action = action.squeeze(0).numpy()
        return action * self.act_high
```

Similar for TD3 (deterministic only), DQN (argmax), PPO (mean of Gaussian / argmax).

Step 2: Pass `algo=self` in all training loop `on_step()` calls:
```python
should_continue = self.callbacks.on_step(reward=ep_reward, step=self._global_step, algo=self)
```

Step 3: Implement `EvalCallback.on_step()`:
```python
def on_step(self, **kwargs) -> bool:
    self._step_count += 1
    if self._step_count % self.eval_freq != 0:
        return True
    algo = kwargs.get("algo")
    if algo is None or self.eval_env is None:
        return True
    rewards = []
    for _ in range(self.n_eval_episodes):
        obs, _ = self.eval_env.reset()
        ep_reward, done = 0.0, False
        while not done:
            action = algo.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = self.eval_env.step(action)
            ep_reward += r
            done = term or trunc
        rewards.append(ep_reward)
    mean_reward = sum(rewards) / len(rewards)
    if self.best_model_path and mean_reward > self.best_reward:
        self.best_reward = mean_reward
        algo.save(self.best_model_path)
    return True
```

Step 4: Implement `CheckpointCallback.on_step()` similarly.

**Testing:** Test EvalCallback triggers evaluation at correct frequency, saves best model.

**Effort:** Medium

---

### P1.4: Fix SACTrainer/DQNTrainer callback forwarding

**File:** `python/rlox/trainers.py` lines 116, 158

**Problem:** Callbacks passed to `SACTrainer(callbacks=[...])` are never forwarded to `SAC()`.

**Fix:** Pass callbacks to algo constructor:
```python
# SACTrainer.__init__:
self.algo = SAC(env_id=env, seed=seed, callbacks=callbacks, **cfg)

# DQNTrainer.__init__:
self.algo = DQN(env_id=env, seed=seed, callbacks=callbacks, **cfg)
```

**Effort:** Trivial (one line each)

---

## P2 — Medium Impact

### P2.1: Add ProgressBarCallback

**File:** `python/rlox/callbacks.py`

```python
class ProgressBarCallback(Callback):
    def on_training_start(self, **kwargs):
        total = kwargs.get("total_timesteps", 0)
        try:
            from tqdm.auto import tqdm
            self._pbar = tqdm(total=total, unit="step")
        except ImportError:
            self._pbar = None

    def on_step(self, **kwargs) -> bool:
        if self._pbar is not None:
            self._pbar.update(1)
            reward = kwargs.get("reward")
            if reward is not None:
                self._pbar.set_postfix(reward=f"{reward:.1f}")
        return True

    def on_training_end(self, **kwargs):
        if self._pbar is not None:
            self._pbar.close()
```

Requires passing `total_timesteps=` in `on_training_start()` call in each algorithm's `train()`.

**Effort:** Low

---

### P2.2: Add TimingCallback for profiling

**File:** `python/rlox/callbacks.py`

```python
class TimingCallback(Callback):
    """Measure wall-clock time of each training phase."""

    def __init__(self):
        self.phase_times = defaultdict(float)
        self._phase_start = None
        self._current_phase = None

    def _start_phase(self, name):
        now = time.perf_counter()
        if self._current_phase:
            self.phase_times[self._current_phase] += now - self._phase_start
        self._current_phase = name
        self._phase_start = now

    def on_step(self, **kwargs) -> bool:
        self._start_phase("env_step")
        return True

    def on_rollout_end(self, **kwargs):
        self._start_phase("gae_compute")

    def on_train_batch(self, **kwargs):
        self._start_phase("gradient_update")

    def summary(self) -> dict[str, float]:
        total = sum(self.phase_times.values())
        return {k: v / total * 100 for k, v in self.phase_times.items()}
```

**Effort:** Low

---

### P2.3: Add CLI entry point

**File:** New `python/rlox/__main__.py`

```python
"""python -m rlox train --algo ppo --env CartPole-v1 --timesteps 100000"""

import argparse

def main():
    parser = argparse.ArgumentParser(prog="rlox")
    sub = parser.add_subparsers(dest="command")

    train_p = sub.add_parser("train")
    train_p.add_argument("--algo", required=True, choices=["ppo", "a2c", "sac", "td3", "dqn"])
    train_p.add_argument("--env", required=True)
    train_p.add_argument("--timesteps", type=int, default=100_000)
    train_p.add_argument("--config", default=None, help="YAML config file")
    train_p.add_argument("--seed", type=int, default=42)

    eval_p = sub.add_parser("eval")
    eval_p.add_argument("--checkpoint", required=True)
    eval_p.add_argument("--env", required=True)
    eval_p.add_argument("--episodes", type=int, default=10)

    args = parser.parse_args()
    # ... dispatch to algorithm classes
```

Also add `console_scripts` entry in `pyproject.toml`:
```toml
[project.scripts]
rlox = "rlox.__main__:main"
```

**Effort:** Medium

---

### P2.4: torch.compile fix

**File:** `python/rlox/compile.py`

**Fix:** Compile individual methods instead of whole module:
```python
for method_name in ("get_action_and_logprob", "get_value", "get_logprob_and_entropy"):
    if hasattr(policy, method_name):
        original = getattr(policy, method_name)
        compiled = torch.compile(original, **compile_kwargs)
        setattr(policy, method_name, compiled)
```

**Effort:** Low

---

### P2.5: MultiGPU off-policy support

**File:** `python/rlox/distributed/multi_gpu.py` lines 62-69

**Fix:** Extend DDP wrapping to off-policy networks:
```python
for attr in ("policy", "actor", "critic1", "critic2", "q_network"):
    net = getattr(inner, attr, None)
    if net is not None and isinstance(net, nn.Module):
        setattr(inner, attr, DDP(net.to(self.device), device_ids=[self.rank]))

# Move target networks to device WITHOUT DDP
for attr in ("critic1_target", "critic2_target", "actor_target", "target_network"):
    net = getattr(inner, attr, None)
    if net is not None:
        setattr(inner, attr, net.to(self.device))
```

**Effort:** Low

---

### P2.6: Custom environment support

**Files:** `sac.py`, `td3.py`, `dqn.py` constructors

**Fix:** Accept `env: str | gymnasium.Env`:
```python
def __init__(self, env: str | gymnasium.Env, ...):
    if isinstance(env, str):
        self.env = gymnasium.make(env)
        self.env_id = env
    else:
        self.env = env
        self.env_id = getattr(env.spec, "id", "custom") if hasattr(env, "spec") else "custom"
```

**Effort:** Low

---

### P2.7: Wire RemoteEnvPool to gRPC client

**File:** `python/rlox/distributed/remote_env.py`

**Current state:** Always raises `ConnectionError`. Rust gRPC client/server are complete.

**Fix:** Create a Python wrapper that calls the Rust gRPC client via PyO3:
1. Expose `RemoteEnvClient` from rlox-grpc via PyO3
2. Have `RemoteEnvPool` use it instead of raising

**Effort:** High (requires PyO3 binding for async Rust + tokio runtime)

---

## P3 — CI/CD and Distribution

### P3.1: Add Windows and Intel Mac wheels

**File:** `.github/workflows/wheels.yml`

Add targets:
```yaml
- target: x86_64-apple-darwin
- target: x86_64-pc-windows-msvc
```

Add multi-Python support via `abi3-py39` or explicit Python version matrix.

**Effort:** Low (maturin handles cross-compilation)

---

### P3.2: Add Python type checking to CI

**File:** `.github/workflows/ci.yml`

```yaml
- name: Type check
  run: pip install pyright && pyright python/rlox/
```

**Effort:** Low (but may require fixing type annotation issues)

---

### P3.3: Add coverage reporting

**File:** `.github/workflows/ci.yml`

Rust: `cargo tarpaulin --workspace --out xml`
Python: `pytest --cov=rlox --cov-report=xml`

Upload to Codecov/Coveralls.

**Effort:** Low

---

## Implementation Order

### Week 1: P0 items (production-blocking)
1. P0.3: Delete `python-publish.yml` (5 min)
2. P0.5: Fix OnlineDPO loss tensor (15 min)
3. P0.6: Add gradient clipping to DPO/GRPO/OnlineDPO (30 min)
4. P0.4: Fix MAPPO critic input (1 hour)
5. P0.1: Fix IMPALA V-trace dones + bootstrap (4 hours)
6. P0.2: Fix DreamerV3 gradient detach (4 hours)

### Week 2: P1 items (research paper)
7. P1.4: Fix trainer callback forwarding (15 min)
8. P1.3: Implement predict() + EvalCallback + CheckpointCallback (1 day)
9. P1.1: Add tests for 6 untested algorithms (1-2 days)
10. P1.2: Expose MmapReplayBuffer to Python (4 hours)

### Week 3: P2 items (usability)
11. P2.1: ProgressBarCallback (1 hour)
12. P2.2: TimingCallback (1 hour)
13. P2.3: CLI entry point (4 hours)
14. P2.4: torch.compile fix (1 hour)
15. P2.5: MultiGPU off-policy (2 hours)
16. P2.6: Custom env support (1 hour)

### Week 4: P3 items (distribution)
17. P3.1: Windows + Intel Mac wheels (2 hours)
18. P3.2: Python type checking in CI (2 hours)
19. P3.3: Coverage reporting (1 hour)
