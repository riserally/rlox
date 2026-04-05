# Code Review: Python Algorithm Implementations

**Date:** 2026-03-29
**Reviewer:** Code Review Agent
**Scope:** All Python algorithm implementations and infrastructure
**Test baseline:** 1048 passed, 7 skipped before fixes / 1049 passed after fixes

## Summary

The rlox Python codebase is generally well-structured with consistent patterns across all 16 algorithm implementations. RL correctness (loss signs, gradient detachment, truncation handling, target networks) is sound. The primary issues found are a dead code block that silently disables a feature in TD3, a type-safety violation in CalQL that accepts discrete envs but silently misuses them, a partial lock in IMPALA that could allow torn reads during policy updates, and an O(n) list operation in DQN's n-step buffer. Several lower-severity issues relate to unsafe pickle deserialization, missing type hints, inconsistent checkpoint coverage, and test coverage gaps.

## Verdict: REQUEST CHANGES

---

## Critical Issues (CRITICAL)

### `python/rlox/algorithms/td3.py:139-142` — Dead `compile` block, feature permanently disabled

The `if compile: compile_policy(self)` block was placed **after a `return` statement** inside `_get_config_dict()`. The `compile` flag accepted in `TD3.__init__` was stored but the effect never applied. Any user passing `compile=True` would see no error and no compilation.

**Fix applied:** Moved the block to the correct position in `__init__`, before `_get_config_dict` is defined, matching the pattern used by PPO, SAC, and all other algorithms.

---

## High Issues (HIGH)

### `python/rlox/algorithms/calql.py:103-123` — CalQL silently accepts discrete envs and misuses them

`CalQL.__init__` detected `self.discrete` but then unconditionally constructed `SquashedGaussianPolicy(obs_dim, self.act_dim=1, hidden)`. For a discrete env like `CartPole-v1`, this created a continuous 1-D actor and attempted to treat discrete integer actions as a 1-D continuous action space. Training would proceed with no error but completely wrong semantics — the actor would output values in `(-1, 1)` via tanh, which were then rounded/clipped to an integer via `int(np.clip(np.round(act_arr[0]), 0, n_actions-1))`. Cal-QL is defined only for continuous SAC backbones.

**Fix applied:** Added an explicit `ValueError` when a discrete action space is detected, matching the algorithm's design intent. Updated existing tests that erroneously used `CartPole-v1` to use `Pendulum-v1`, and added a regression test `test_calql_rejects_discrete_env`.

### `python/rlox/algorithms/impala.py:394-398` — Partial lock around gradient update creates torn parameter reads

`optimizer.zero_grad()` and `loss.backward()` were executed **outside** `_policy_lock`, while only `optimizer.step()` was inside the lock. An actor thread calling `_get_policy_snapshot()` (which also acquires the lock) could read parameter values that are in the middle of gradient accumulation — after `zero_grad` cleared gradients but before `step` applied them. This is a benign condition most of the time (actors tolerate stale parameters by design in IMPALA) but causes a correctness hazard: a snapshot read between `backward` and `step` returns the pre-update parameters even though the gradient update is logically in progress.

**Fix applied:** The entire `zero_grad → backward → clip_grad_norm → step` sequence is now inside a single `with self._policy_lock` block, ensuring actor snapshots see either the old or the new parameters, never a partially-computed state.

### `python/rlox/algorithms/dqn.py:123,182,253` — O(n) `list.pop(0)` in N-step buffer hot path

`self._n_step_buffer` was declared as `list[tuple]` and used `pop(0)` to remove the oldest element during each step. `pop(0)` on a Python list is O(n) because it shifts every remaining element. For `n_step=3` this is negligible, but for larger `n_step` values (e.g. 10+) in the episode-flush loop (`while self._n_step_buffer: ... pop(0)`) this becomes O(n^2) per episode.

**Fix applied:** Changed to `collections.deque` with `popleft()`, which is O(1). The `deque` is unbounded (appropriate since the buffer is always kept at exactly `n_step` entries).

---

## Medium Issues (MEDIUM)

### `python/rlox/checkpoint.py:59` and all `from_checkpoint` methods — Unsafe pickle deserialization via `weights_only=False`

All 12 checkpoint load paths use `torch.load(path, weights_only=False)`. This enables full pickle deserialization of arbitrary Python objects from the checkpoint file. A malicious or tampered checkpoint file can execute arbitrary code when loaded.

**Recommendation:** Switch to `weights_only=True` for all model state dict loading. The non-tensor fields (config dicts, step counter) must then be stored in a separate safe channel, e.g. a sidecar JSON file written alongside the `.pt` file. This is the standard pattern for production RL systems.

**Affected files:** `checkpoint.py`, `sac.py`, `td3.py`, `dqn.py`, `dreamer.py`, `impala.py`, `mappo.py`, `qmix.py`, `awr.py`, `calql.py`, `decision_transformer.py`, `diffusion_policy.py`.

### `python/rlox/wrappers/language.py:143,156` — `assert` used for runtime control flow in `GoalConditionedWrapper`

Two `assert self._goals is not None` statements guard safety-critical control flow in `_compute_sparse_reward` and `step_all`. `assert` statements are removed entirely when Python is run with `-O` (optimized mode). If `step_all` is called before `reset_all` in production, this results in a `TypeError` (`NoneType` has no `.shape`) rather than a clear error.

**Recommendation:** Replace with explicit `if self._goals is None: raise RuntimeError("reset_all() must be called before step_all()")`.

### `python/rlox/algorithms/mpo.py:309-311` — E-step weights computed in a Python loop over batch

`torch.stack([self.compute_e_step_weights(q_vals[i], eta) for i in range(b)])` computes normalized softmax weights per sample in a Python loop. For large batch sizes this adds overhead and prevents vectorized execution.

**Recommendation:** Replace with a single vectorized call:
```python
log_w = q_vals / max(eta, 1e-8)
log_w = log_w - log_w.max(dim=-1, keepdim=True).values
weights = torch.softmax(log_w, dim=-1)
```

### `python/rlox/algorithms/trpo.py:297` — `max(kl, 0.0)` on a Python float is cosmetic but misleading

`_compute_kl_at_params` returns `max(kl, 0.0)` claiming "KL should be non-negative". KL computed as `(log_pi_old - log_pi_new).mean()` is not mathematically guaranteed to be non-negative (due to sample approximation). Clipping at zero hides potential divergences and makes the line-search termination criterion `kl <= cfg.max_kl` always satisfy trivially when the true KL is negative.

**Recommendation:** Return the raw float without the `max` clamp, and handle near-zero KL in the line-search condition.

### `python/rlox/algorithms/impala.py:364-375` — CPU roundtrip in every learner step

`_learner_step` detaches tensors, calls `.numpy()`, passes them to the Rust `compute_vtrace`, then converts back to `torch.as_tensor`. When training on GPU this incurs a mandatory device synchronization + CPU copy every learner step. For CPU-only training this is correct behavior. For GPU training it's a bottleneck.

**Note:** This is an architectural constraint of the Rust V-trace binding, which operates on NumPy arrays. The correct long-term fix is a CUDA-native V-trace kernel. Document the known limitation.

### `python/rlox/off_policy_collector.py:179` — Dead branch in `collect_step` with identical code paths

```python
if self._n_envs > 1:
    result = self.env.step_all(actions if not self.is_continuous else actions)
else:
    result = self.env.step_all(actions)
```

Both branches call `self.env.step_all(actions)` — the condition `actions if not self.is_continuous else actions` in the `n_envs > 1` branch always evaluates to `actions`. This is dead conditional logic from a refactor.

**Recommendation:** Collapse to a single `result = self.env.step_all(actions)`.

---

## Low Issues (LOW)

### `python/rlox/algorithms/vpg.py:272` — `save()` persists only the VF optimizer, losing policy optimizer state

`VPG.save()` calls `Checkpoint.save(..., optimizer=self.vf_optimizer, ...)` but does not save `self.policy_optimizer`. After `from_checkpoint()`, the policy optimizer is re-initialized from scratch, losing momentum and adaptive learning rate state. This means resumed training starts policy gradient steps from a cold optimizer.

**Recommendation:** Serialize both optimizers: use `torch.save` directly as done in SAC/TD3, or extend `Checkpoint.save` to accept multiple optimizers.

### `python/rlox/algorithms/a2c.py:163-167` — `save()` hardcodes `step=0`, discarding global step count

`A2C.save()` passes `step=0` to `Checkpoint.save`. The class tracks `_global_step` implicitly through callbacks but does not store it as an instance attribute. After checkpoint restore, any step-dependent callbacks or logging will start from step 0.

### `python/rlox/algorithms/qmix.py:346-347` — QMIX uses agent 0's action for single-env stepping; other agents' chosen actions are discarded

In `train()`, `env_action = actions[0]` — only the first agent's action is sent to the environment. Actions for agents 1..n-1 are computed but thrown away. This is noted in a comment but has algorithmic implications: the Q-values for agents 1..n-1 are trained on an action that never actually executed.

**Note:** This reflects the structural limitation of simulating multi-agent QMIX in a single-agent Gym env. Document this limitation or add a warning.

### `python/rlox/algorithms/decision_transformer.py:59` — Attention mask set as `is_causal=False` but a manual causal mask is passed

`TransformerBlock.forward` calls `self.attn(..., attn_mask=causal_mask, is_causal=False)`. The `is_causal=True` flag in PyTorch's `MultiheadAttention` applies a flash-attention-compatible causal mask internally, while `is_causal=False` with an explicit `attn_mask` is also valid. The current approach is correct but inconsistent: passing both an explicit mask and `is_causal=True` would raise an error, so the explicit mask approach is intentional. The code could benefit from a comment explaining why `is_causal=False` is used instead of the builtin flag.

### `python/rlox/algorithms/mappo.py:602-621` — Multi-agent training reuses the same `adv` tensor across epochs without recomputing

In `_train_multi_agent`, the advantage normalization `adv = (adv - adv.mean()) / (adv.std() + 1e-8)` is computed once before the epoch loop. For `n_epochs > 1` the exact same normalized advantages are used for every epoch. This is standard practice for MAPPO (same as PPO in single-agent mode), but the comment-free code may be confusing for readers expecting epoch-wise re-normalization.

### `python/rlox/pbt.py:130` — `random.uniform` used without seeding

`PBT._explore` calls `random.uniform` using the global Python random state, which is not seeded via `self.seed`. This makes hyperparameter perturbation in PBT non-reproducible even when a seed is provided.

**Recommendation:** Use `random.Random(self.seed)` or `numpy.random.default_rng(self.seed)` per PBT instance.

### `python/rlox/deploy/docker.py:48-49` — `algo` and `env` values are interpolated directly into Dockerfile without validation

```python
ENV RLOX_ALGO={algo}
ENV RLOX_ENV={env}
```

If `algo` or `env` contain newlines or shell metacharacters, the output Dockerfile would be malformed. The inputs should be validated to be safe identifiers before interpolation.

---

## Suggestions (OPTIONAL)

### `python/rlox/algorithms/sac.py:174-175` — Action scaling applied twice in inference path

In the single-env training loop:
```python
action_t, _ = self.actor.sample(obs_t)
action = action_t.squeeze(0).numpy()
action = action * self.act_high  # scales pre-squash output
```
`SquashedGaussianPolicy.sample()` returns actions in `(-1, 1)` (post-tanh). The `* self.act_high` is correct. However, in `predict()`:
```python
action, _ = self.actor.sample(obs_t)
return action * self.act_high
```
And in `_train_with_collector` the `get_action` closure also applies `* self.act_high`. This is consistent, but it means callers must never call `predict()` and then rescale again. Consider documenting this clearly in the docstring, or making the actor output natively scaled actions.

### `python/rlox/algorithms/` — Missing `predict()` method on several algorithms

`TRPO`, `IMPALA`, `MAPPO`, `A2C`, `VPG` do not implement `predict()`. All other algorithms do. This creates an inconsistent public API. Consider defining it on a base class or adding it to each missing algorithm.

### `python/rlox/collectors.py:174-181` — Truncation bootstrap computed inside `torch.no_grad()` collector loop

The `policy.get_value(term_obs_t)` call to bootstrap truncated episodes is correct but the call happens inside a Python loop over `self.n_envs`. For large `n_envs` this serializes what could be a batched value computation. Consider batching all truncated terminal obs together and making one batched `get_value` call.

### `python/rlox/algorithms/mpo.py` — Critic update uses `next_log_prob` without entropy temperature

In `_update_critic`, the target is `rewards + gamma * (1 - done) * (min(Q1, Q2) - log_prob)` without an entropy temperature. The original MPO paper separates the E-step temperature (`eta`) from the SAC-style entropy temperature used in the critic target. The current implementation conflates the two by computing targets without any temperature scaling of the log-prob. This will work but diverges from the reference paper's formulation.

---

## Positive Highlights

- **Truncation handling is correct everywhere.** All single-env training loops properly distinguish `terminated` vs `truncated`, store both flags in the buffer, and use only `terminated` as the done mask in Bellman targets (`1.0 - terminated`). The `RolloutCollector` correctly bootstraps truncated episodes. This is a common source of bugs in RL codebases and this one handles it correctly throughout.

- **Conjugate gradient TRPO implementation is clean and correct.** The `_conjugate_gradient`, `_flat_grad`, `_get_flat_params`, `_set_flat_params` utilities are well-factored and the Fisher-vector product via double-backprop is correctly implemented. The backtracking line search correctly restores old parameters when no step is accepted.

- **IMPALA's vectorized V-trace is a clean optimization.** Batching all per-env arrays into a single loop over `compute_vtrace` calls (avoiding a separate Python for-loop in the main learner path) is a good design. The pre-allocation of `vs_all` and `pg_all` avoids repeated memory allocation.

- **Checkpoint async save pattern is safe.** The `Checkpoint.save(async_save=True)` implementation correctly serializes to a `BytesIO` buffer on the main thread before spawning the background writer thread, preventing data races where the model could be updated before the file is written.

- **DreamerV3 symlog/symexp transforms are implemented correctly.** The `symlog = sign(x) * log1p(|x|)` and `symexp = sign(x) * (exp(|x|) - 1)` are numerically stable and match the DreamerV3 paper.

- **PPO loss implementation matches CleanRL reference.** Clipped value loss, advantage normalization, KL approximation diagnostic, and clip fraction metric are all correctly computed. The `old_log_probs.detach()` is correctly handled by storing them at collection time.

---

## Test Coverage Assessment

**Well-covered:**
- PPO, SAC, DQN, A2C, TD3, MAPPO, TRPO, IMPALA, AWR, DiffusionPolicy, DecisionTransformer, QMIX, Dreamer all have smoke tests in `test_algorithm_smoke.py` and `test_new_algorithms.py`.
- CalQL has a unit test for penalty calibration.
- VecNormalize has dedicated tests in `test_vec_normalize.py`.
- Off-policy collector has tests in `test_off_policy_collector.py`.
- GPU buffer has tests in `test_gpu_buffer.py`.

**Coverage gaps:**
- No tests for `MPO` end-to-end training loop or E/M-step decomposition.
- No tests for `PBT` with a fixed seed to verify reproducibility.
- No tests for `Reptile` meta-training.
- No tests for IMPALA with `n_actors > 1` verifying V-trace correctness.
- No tests for `VPG` checkpoint round-trip verifying both optimizer states are preserved.
- No tests for `CalQL` training on a continuous env end-to-end.
- `GoalConditionedWrapper` has no test for calling `step_all` before `reset_all` (assert-guard removed case).
- DQN N-step buffer flush (the episode-end while loop) has no dedicated unit test.

---

## TDD Compliance

- [x] Most algorithms have tests written first (TDD comment in `test_new_algorithms.py`)
- [ ] `MPO`, `Reptile`, `PBT` have no corresponding test files
- [ ] VPG checkpoint round-trip test missing — optimizer state loss is undetected
- [x] CalQL discrete env regression test added as part of this review fix

---

## Fixed Files

The following files were modified to address CRITICAL and HIGH findings:

| File | Change |
|------|--------|
| `python/rlox/algorithms/td3.py` | Moved `compile` block out of `_get_config_dict` return scope into `__init__` |
| `python/rlox/algorithms/calql.py` | Added `ValueError` for discrete action spaces; updated test to use `Pendulum-v1` |
| `python/rlox/algorithms/impala.py` | Widened `_policy_lock` to cover `zero_grad`, `backward`, and `step` |
| `python/rlox/algorithms/dqn.py` | Changed `_n_step_buffer` from `list` to `collections.deque`; replaced `pop(0)` with `popleft()` |
| `tests/python/test_new_algorithms.py` | Updated CalQL tests to use continuous env; added discrete env rejection test |
