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
| 16 × 128 (2K) | 6.1 ms | 10.2 ms | 129.2 ms | 333,434 trans/s |
| 64 × 512 (33K) | 43.6 ms | 134.9 ms | 1,767.8 ms | 751,383 trans/s |
| 256 × 2048 (524K) | 538.5 ms | 2,080.0 ms | 28,431.8 ms | 973,694 trans/s |

### Speedup

| Config | vs SB3 | 95% CI | vs TorchRL | 95% CI |
|--------|--------|--------|------------|--------|
| 16 × 128 | **1.7x** | [1.5, 1.7] | **21.0x** | [19.4, 22.0] |
| 64 × 512 | **3.1x** | [3.1, 3.2] | **40.5x** | [39.9, 41.6] |
| 256 × 2048 | **3.9x** | [3.8, 3.9] | **52.8x** | [52.1, 53.5] |

## Analysis

### Why rlox advantage grows with scale

At 16 envs, rlox is 1.7x faster than SB3. At 256 envs, it's 3.9x. The scaling comes from:

1. **Environment stepping**: rlox `VecEnv` uses Rayon thread pool (true parallelism across cores). SB3 `DummyVecEnv` is sequential Python — stepping 256 CartPole envs sequentially is 256× the single-step cost.

2. **Transition storage**: rlox pushes transitions with a single PyO3 boundary crossing per env-step. SB3 creates Python dicts, copies numpy arrays, and does per-element validation.

3. **GAE computation**: rlox computes GAE in ~540us for 524K steps. SB3's Python loop takes ~470ms for the same data (the Python GAE loop over 524K steps is significant at scale).

4. **Compounding**: Each advantage compounds. At 256×2048, the env stepping advantage (~5.7x) combines with buffer push advantage (~10x for small obs) and GAE advantage (~140x) into an overall 3.9x.

### TorchRL overhead

TorchRL is 53x slower at the largest scale. The TensorDict abstraction wraps every observation, reward, done flag, and action in typed tensor metadata. For lightweight environments like CartPole, the metadata cost far exceeds the actual computation.

### Peak throughput

rlox reaches **973K transitions/second** at 256 envs × 2048 steps. This is close to 1M trans/s — near the theoretical limit for CartPole on this hardware when including storage and GAE.

SB3 plateaus at ~252K trans/s. TorchRL at ~18K trans/s.

Source: [`benchmarks/bench_e2e_rollout.py`](../../benchmarks/bench_e2e_rollout.py)
