# Benchmark: End-to-End Rollout Collection

The most realistic benchmark — measures the full PPO-style rollout pipeline: reset M environments, step N times, store all transitions, compute GAE.

## What is Measured

Each framework executes the complete pipeline:

| Framework | Pipeline |
|-----------|---------|
| **rlox** | `VecEnv.reset_all()` → loop N: `VecEnv.step_all(actions)` + `ExperienceTable.push()` per env → `compute_gae()` |
| **SB3** | `DummyVecEnv.reset()` → loop N: `env.step(actions)` + collect obs/rewards → Python GAE loop |
| **TorchRL** | `SerialEnv.reset()` → loop N: `env.step(td)` + collect TensorDicts → `generalized_advantage_estimate()` |

All use a dummy policy (action=0) to isolate infrastructure cost from neural network inference.

### Configurations

| Config | Envs (M) | Steps (N) | Total Transitions |
|--------|----------|-----------|-------------------|
| Small | 16 | 128 | 2,048 |
| Medium | 64 | 512 | 32,768 |
| Large | 256 | 2,048 | 524,288 |

## Results

### Raw Timings & Throughput

| Config | rlox | SB3 | TorchRL | rlox throughput |
|--------|------|-----|---------|-----------------|
| 16 × 128 (2K) | 5.8 ms | 10.0 ms | 126.5 ms | 353,698 trans/s |
| 64 × 512 (33K) | 60.5 ms | 132.8 ms | 1,722.1 ms | 541,695 trans/s |
| 256 × 2048 (524K) | 680.7 ms | 2,048.3 ms | 27,501.5 ms | 770,168 trans/s |

### Speedup

| Config | vs SB3 | 95% CI | vs TorchRL | 95% CI |
|--------|--------|--------|------------|--------|
| 16 × 128 | **1.7x** | [1.7, 1.8] | **21.9x** | [21.4, 23.3] |
| 64 × 512 | **2.2x** | [2.2, 2.2] | **28.5x** | [28.0, 28.8] |
| 256 × 2048 | **3.0x** | [3.0, 3.0] | **40.4x** | [40.1, 40.6] |

## Analysis

### Why rlox advantage grows with scale

At 16 envs, rlox is 1.7x faster than SB3. At 256 envs, it's 3.0x. The scaling comes from:

1. **Environment stepping**: rlox `VecEnv` uses Rayon thread pool (true parallelism across cores). SB3 `DummyVecEnv` is sequential Python — stepping 256 CartPole envs sequentially is 256× the single-step cost.

2. **Transition storage**: rlox pushes transitions with a single PyO3 boundary crossing per env-step. SB3 creates Python dicts, copies numpy arrays, and does per-element validation.

3. **GAE computation**: rlox computes GAE in ~540us for 524K steps. SB3's Python loop takes ~470ms for the same data (the Python GAE loop over 524K steps is significant at scale).

4. **Compounding**: Each advantage compounds. At 256×2048, the env stepping advantage (~7x) combines with buffer push advantage (~4x for small obs) and GAE advantage (~142x) into an overall 3.0x.

### TorchRL overhead

TorchRL is 40x slower at the largest scale. The TensorDict abstraction wraps every observation, reward, done flag, and action in typed tensor metadata. For lightweight environments like CartPole, the metadata cost far exceeds the actual computation.

### Peak throughput

rlox reaches **770K transitions/second** at 256 envs × 2048 steps. This is near the theoretical limit for CartPole on this hardware when including storage and GAE.

SB3 plateaus at ~256K trans/s. TorchRL at ~19K trans/s.

Source: [`benchmarks/bench_e2e_rollout.py`](../../benchmarks/bench_e2e_rollout.py)
