# rlox Convergence Benchmark Profiling Report

**Date:** 2026-03-28
**Platform:** macOS Darwin 25.2.0 (Apple Silicon), Python 3.12, PyTorch 2.10.0
**Build:** Release (Rust extension compiled via maturin)

---

## 1. End-to-End Algorithm Timing

### PPO on CartPole-v1 (50,000 timesteps, 8 envs)

| Metric | Value |
|--------|-------|
| Total wall time | 2.66 s |
| Init time | 0.43 s |
| Training time | 2.23 s |
| Steps per second (SPS) | 18,796 |

**Phase breakdown (TimingCallback):**

| Phase | Time (s) | Share |
|-------|----------|-------|
| env_step (collect) | 1.479 | 66.4% |
| gradient_update (SGD) | 0.674 | 30.2% |
| gae_compute | 0.076 | 3.4% |
| **Total tracked** | **2.229** | **100%** |

### SAC on Pendulum-v1 (20,000 timesteps)

| Metric | Value |
|--------|-------|
| Total wall time | 94.42 s |
| Init time | 0.29 s |
| SPS | 212 |
| Learning starts | 1,000 steps (random exploration) |
| Effective training steps | 19,000 (with gradient updates) |

---

## 2. Component-Level Micro-Benchmarks

### Rust Primitives (all times in microseconds)

| Operation | Latency |
|-----------|---------|
| VecEnv.step_all(64 envs) | 48.4 us |
| ReplayBuffer.push(single) | 0.1 us |
| ReplayBuffer.push_batch(8) | 0.3 us |
| ReplayBuffer.sample(256) | 3.7 us |
| compute_gae_batched(8x128) | 34.7 us |
| compute_batch_token_kl_schulman_f32(32x2048) | 60.6 us |

**Verdict:** All Rust primitives are extremely fast. The Rust data-plane is not a bottleneck.

### PyTorch Operations (PPO, batch=8 envs)

| Operation | Per-step latency |
|-----------|-----------------|
| policy.get_action_and_logprob (8 obs) | 123 us |
| policy.get_value (8 obs) | 19 us |
| VecEnv.step_all (8 envs) | 96 us |
| Tensor conversion (numpy<->torch) | 2 us |

### PyTorch Operations (SAC, single env)

| Operation | Latency |
|-----------|---------|
| Gym env.step (Pendulum) | 15.2 us |
| actor.sample (single obs) | 49.8 us |
| ReplayBuffer.push | 0.5 us |
| ReplayBuffer.sample(256) | 4.0 us |
| Full _update (1 gradient step) | 4,790 us |

**SAC _update() internal breakdown:**

| Component | Latency | Share |
|-----------|---------|-------|
| Actor fwd+bwd+step | 2,070 us | 44.0% |
| Critic fwd+bwd+step (both) | 1,683 us | 35.8% |
| Target Q (no_grad) | 709 us | 15.1% |
| Polyak soft update | 197 us | 4.2% |
| Alpha update | 39 us | 0.8% |
| Sample + tensor convert | 8 us | 0.2% |
| **Total** | **4,708 us** | **100%** |

---

## 3. FFI Crossing Analysis

### PPO: Per Rollout (8 envs x 128 steps = 1,024 timesteps)

| FFI Call | Count | Notes |
|----------|-------|-------|
| VecEnv.step_all | 128 | 1 per step |
| policy.get_action_and_logprob | 128 | PyTorch, not FFI per se |
| policy.get_value | 129 | +1 for bootstrap |
| compute_gae_batched | 1 | Single batched call |
| **Total Rust FFI calls** | **129** | step_all + GAE |
| **Total Python<->Rust boundary crossings** | ~129 | Negligible overhead at ~2 us each |

PPO's design is already efficient: environment stepping is batched (all 8 envs in one `step_all` call), and GAE is a single batched FFI call.

### SAC: Per Training Step (after learning_starts)

| FFI Call | Count | Notes |
|----------|-------|-------|
| gym.env.step | 1 | Python Gymnasium, not Rust |
| ReplayBuffer.push | 1 | ~0.5 us |
| ReplayBuffer.sample | 1 | ~4 us |
| **Total Rust FFI calls** | **2** | push + sample |

SAC's bottleneck is purely PyTorch gradient computation, not FFI.

---

## 4. Time Spent: Rust vs Python vs PyTorch

### PPO (per rollout of 1,024 timesteps)

| Layer | Time (ms) | Share |
|-------|-----------|-------|
| **Rust** (VecEnv.step_all + GAE) | 12.3 | 19.6% |
| **PyTorch** (policy forward, no grad) | 18.2 | 29.0% |
| **PyTorch** (SGD: forward+backward+step) | 28.5 | 45.5% |
| **Python** (tensor conversions, bookkeeping) | 3.7 | 5.9% |
| **Total** | **62.7** | **100%** |

### SAC (per training step)

| Layer | Time (us) | Share |
|-------|-----------|-------|
| **Rust** (buffer push + sample) | 4.5 | 0.1% |
| **Python Gymnasium** (env.step) | 15.2 | 0.3% |
| **PyTorch inference** (actor.sample, no grad) | 49.8 | 1.0% |
| **PyTorch training** (_update: fwd+bwd+step) | 4,790 | 98.6% |
| **Total** | **~4,860** | **100%** |

---

## 5. Bottleneck Analysis

### PPO Bottlenecks

1. **Policy forward passes during collection (29%)** -- The collector calls `get_action_and_logprob` and `get_value` separately in a Python loop, 128 times per rollout. Each call processes only 8 observations (the env batch). The actor forward takes 123 us per call, which is dominated by PyTorch dispatch overhead on such tiny tensors (8x4 input through a 64-hidden MLP).

2. **SGD phase (45.5%)** -- 4 epochs x 2 minibatches = 8 gradient steps per rollout. Each minibatch (512 samples) takes ~3.56 ms. This is healthy PyTorch compute and hard to improve without algorithmic changes.

3. **VecEnv stepping is already fast** -- 96 us per step_all(8) is good but could be further batched by collecting multiple steps per FFI call.

### SAC Bottlenecks

1. **PyTorch gradient computation (98.6%)** -- The _update function dominates. Within it, the actor backward pass (44%) and critic backward pass (36%) are the main costs. This is expected for an off-policy algorithm doing 3 backward passes per step.

2. **Single-env stepping** -- SAC uses a single Gymnasium (Python) environment. This is not a bottleneck because the gradient step is 300x slower than the env step, but vectorizing SAC data collection could help with future scaling.

3. **Polyak update (4.2% of _update)** -- Iterates over all parameter pairs in Python. Could be batched with `torch._foreach_lerp_` or moved to a fused Rust kernel.

---

## 6. Optimization Recommendations

### High Impact

| # | Optimization | Target | Estimated Impact |
|---|-------------|--------|-----------------|
| 1 | **Fuse actor+critic forward in collector** | PPO collect | Combine `get_action_and_logprob` + `get_value` into a single shared-trunk forward pass. Saves ~19 us/step (half the value forward). **~10% wall-time reduction on PPO.** |
| 2 | **torch.compile the PPO policy** | PPO collect + SGD | Apply `torch.compile(policy, mode='reduce-overhead')`. Eliminates PyTorch dispatch overhead on the small MLPs. On 8-obs batches, dispatch overhead dominates compute. **~15-25% wall-time reduction on PPO.** |
| 3 | **torch.compile SAC networks** | SAC _update | Compile actor, critic1, critic2 with `torch.compile`. The 256-batch forward/backward passes will benefit from kernel fusion. **~20-30% wall-time reduction on SAC (estimated 40-80 SPS improvement).** |
| 4 | **Batch multi-step collection in Rust** | PPO collect | Instead of calling `step_all` 128 times from Python, add a `collect_n_steps(n)` method to VecEnv that returns all 128 steps' data in one FFI call. Eliminates 127 Python<->Rust round-trips and the per-step Python loop overhead. **~5-10% wall-time reduction on PPO.** |

### Medium Impact

| # | Optimization | Target | Estimated Impact |
|---|-------------|--------|-----------------|
| 5 | **Use `torch._foreach` ops for Polyak update** | SAC _update | Replace the `for sp, tp in zip(...)` loop with `torch._foreach_lerp_`. Saves Python loop overhead across ~10 parameter tensors. **~3% of SAC _update time.** |
| 6 | **Vectorize SAC data collection** | SAC env step | Run N parallel envs and batch-insert into the replay buffer. Would improve SPS at high timestep counts by amortizing the single-env overhead. **Marginal at 20k steps, significant at 1M+.** |
| 7 | **Eliminate `.tolist()` in discrete action conversion** | PPO collect | The collector calls `.astype(np.uint32).tolist()` to convert actions for VecEnv. VecEnv already accepts numpy arrays. Remove the `.tolist()` call. **< 1% improvement (0.5 us/step), but free.** |
| 8 | **Pre-allocate rollout tensors** | PPO collect | The collector builds lists of tensors and calls `torch.stack` at the end. Pre-allocating a `(n_steps, n_envs, ...)` tensor and writing into it avoids N small allocations + one stack op. **~2-3% of collect time.** |

### Low Impact / Future Work

| # | Optimization | Target | Notes |
|---|-------------|--------|-------|
| 9 | **Move PPO SGD inner loop to Rust** | PPO SGD | Would eliminate Python overhead in the minibatch loop. Diminishing returns since PyTorch backward is the actual cost. |
| 10 | **CUDA/MPS backend** | All | Moving to GPU would transform the profile entirely. The tiny-batch overhead that dominates CPU PPO disappears on GPU. |
| 11 | **Increase n_envs for PPO** | PPO collect | Going from 8 to 32+ envs amortizes per-step PyTorch dispatch overhead across more data. This is a hyperparameter trade-off, not a code change. |

---

## 7. Summary

The Rust data-plane primitives (VecEnv, ReplayBuffer, GAE, KL divergence) are all extremely fast, typically sub-100-microsecond. They represent less than 1% of SAC wall time and about 20% of PPO wall time (dominated by env stepping, which is genuinely fast).

The primary bottleneck in both algorithms is **PyTorch forward/backward computation**:
- PPO: 74.5% of time is in PyTorch (29% inference during collection + 45.5% SGD)
- SAC: 99.6% of time is in PyTorch (98.6% gradient updates + 1% inference)

The highest-ROI optimizations are:
1. `torch.compile` on all networks (free 15-30% speedup)
2. Fusing actor+critic forward passes in the PPO collector
3. Batching multi-step collection into a single Rust FFI call

No Rust-side optimizations are needed at this scale. The Rust layer is already 10-100x faster than the PyTorch layer it serves.
