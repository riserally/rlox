# Phase 6: Three-Framework Benchmark Suite

**Goal**: Comprehensive, reproducible performance comparison of **rlox vs TorchRL vs Stable-Baselines3** across the full RL pipeline — environment stepping, buffer operations, GAE computation, and end-to-end rollout collection.

## Design Principles

1. **Fair comparisons**: Measure equivalent work in each framework's idiomatic API
2. **Statistical rigor**: Bootstrap 95% CI on speedup ratios (10,000 resamples, existing infrastructure)
3. **Reproducible**: Fixed seeds, deterministic sampling, JSON artifact output
4. **Graceful degradation**: Missing frameworks are skipped, not errors

## Frameworks Under Test

| Framework | Parallelism Model | Key Characteristic |
|-----------|------------------|--------------------|
| **rlox** | Rayon thread pool (in-process) | Zero IPC, Rust-native, GIL-free |
| **TorchRL** | Multiprocess (`ParallelEnv`) or serial (`SerialEnv`) | TensorDict abstraction, PyTorch C++ kernels |
| **SB3** | Sequential (`DummyVecEnv`) or multiprocess (`SubprocVecEnv`) | NumPy-backed, Python-level abstractions |
| **Gymnasium** | Sequential or multiprocess | Ground truth baseline |
| **NumPy** | N/A | Reference implementations for correctness + baseline |

## Environments

| Environment | obs_dim | action_space | Step Cost | Purpose |
|------------|---------|-------------|-----------|---------|
| CartPole-v1 | 4 | Discrete(2) | ~1us (Rust) | Overhead-dominated, stress-tests framework efficiency |
| LunarLander-v3 | 8 | Discrete(4) | ~10-50us (Box2D) | Medium compute, tests GymEnv bridge overhead |

CartPole is mandatory (rlox has native impl). LunarLander tests the wrapper path (all frameworks go through Gymnasium).

## Benchmark Categories

### B1: Single-Step Latency

**What**: Time one `env.step(action)` call including observation return.

| Framework | API | Notes |
|-----------|-----|-------|
| rlox | `CartPole.step(1)` | Native Rust |
| TorchRL | `GymEnv("CartPole-v1").step(td)` | Gymnasium + TensorDict wrapping |
| Gymnasium | `env.step(1)` | Raw Python baseline |

- Warmup: 200 iterations
- Measurement: 1000 reps
- Auto-reset on done
- Metric: ns/step (median, IQR, p99)

### B2: Vectorized Stepping (Scaling)

**What**: N environments stepped in parallel, throughput vs N.

| Framework | API | Parallelism |
|-----------|-----|-------------|
| rlox VecEnv | `step_all(actions)` | Rayon threads |
| TorchRL ParallelEnv | `env.step(td)` | Multiprocess |
| TorchRL SerialEnv | `env.step(td)` | Sequential |
| SB3 DummyVecEnv | `env.step(actions)` | Sequential |
| SB3 SubprocVecEnv | `env.step(actions)` | Multiprocess |
| Gymnasium SyncVectorEnv | `env.step(actions)` | Sequential |

- N values: `[1, 4, 16, 64, 128, 256, 512]`
- Per measurement: 100 batch steps
- Warmup: 5 rounds, Measurement: 50 reps (20 for subprocess-based)
- Metric: steps/sec, scaling efficiency = `throughput(N) / (N * throughput(1))`

### B3: Buffer Push Throughput

**What**: Insert K transitions into replay/experience buffer.

| Framework | API | Buffer Type |
|-----------|-----|-------------|
| rlox ExperienceTable | `table.push(obs, action, reward, term, trunc)` | Columnar append-only |
| rlox ReplayBuffer | `buf.push(...)` | Ring buffer |
| TorchRL ReplayBuffer | `rb.add(td)` / `rb.extend(td)` | LazyTensorStorage |
| SB3 ReplayBuffer | `buf.add(obs, next_obs, action, reward, done, info)` | NumPy pre-allocated |

- obs_dim: 4 (CartPole), 28224 (Atari-sized)
- K: 10,000 transitions per measurement
- Warmup: 2 rounds, Measurement: 10 reps
- Metric: transitions/sec

### B4: Buffer Sample Latency

**What**: Sample a batch from a full buffer (100K transitions).

| Framework | API |
|-----------|-----|
| rlox ReplayBuffer | `buf.sample(batch_size, seed)` |
| TorchRL ReplayBuffer | `rb.sample(batch_size)` |
| SB3 ReplayBuffer | `buf.sample(batch_size)` |

- batch_size: `[32, 64, 256, 1024]`
- Warmup: 10, Measurement: 100 reps
- Metric: ns/sample (median, p99)

### B5: GAE Computation

**What**: Compute Generalized Advantage Estimation for trajectory of length T.

| Framework | API | Implementation |
|-----------|-----|---------------|
| rlox | `compute_gae(rewards, values, dones, last_value, gamma, lam)` | Rust backward loop |
| TorchRL | `generalized_advantage_estimate(gamma, lam, ...)` | PyTorch C++ kernel |
| SB3/CleanRL | Python loop (reference_gae_numpy) | Pure Python |
| NumPy | Same as SB3 | Baseline |

- T values: `[128, 512, 2048, 8192, 32768]`
- Warmup: 10, Measurement: 100 reps
- Correctness: all must match reference within 1e-6 rtol
- Metric: ns/step, total latency

### B6: End-to-End Rollout Collection

**What**: Full pipeline — reset M envs, step N times, store transitions, compute GAE.

| Framework | Pipeline |
|-----------|---------|
| rlox | `VecEnv.reset_all()` → loop: `step_all()` + `ExperienceTable.push()` → `compute_gae()` |
| SB3 | Equivalent manual loop with DummyVecEnv + ReplayBuffer |
| TorchRL | `env.rollout()` or manual loop with ParallelEnv + ReplayBuffer |

- M (envs): `[16, 64, 256]`
- N (steps): `[128, 512, 2048]`
- Warmup: 1 full rollout, Measurement: 10 reps
- Metric: transitions/sec (total wall clock)

### B7: LLM Operations (rlox vs NumPy/PyTorch only)

No TorchRL/SB3 equivalents — compare against NumPy and PyTorch baselines.

- GRPO group advantages: n_prompts × k_completions
- Token-level KL: sequence lengths 128-8192
- Metric: ns/operation

## Statistical Methodology

- **Clock**: `time.perf_counter_ns()` (monotonic, nanosecond resolution)
- **Primary stat**: Median (robust to GC outliers)
- **Dispersion**: IQR, p99
- **Comparison**: Bootstrap 95% CI on speedup ratio (10K resamples) — existing `ComparisonResult` class
- **Significance**: CI lower bound > 1.0 = statistically faster
- **No outlier removal**: GC pauses are real; report p99 to capture tails
- **Correctness first**: Every benchmark validates output correctness before timing

## Hypotheses

| ID | Hypothesis | Expected Speedup | Rationale |
|----|-----------|-----------------|-----------|
| H1 | rlox single-step < Gymnasium | 2-5x | No Python object creation overhead |
| H2 | rlox single-step < TorchRL single-step | 5-20x | No TensorDict wrapping overhead |
| H3 | rlox VecEnv > SB3 DummyVecEnv ∀N | 2-10x | GIL-free Rayon vs GIL-bound Python |
| H4 | rlox VecEnv > SB3 SubprocVecEnv for N < 64 | 2-5x | Thread pool vs process spawn + pickle |
| H5 | rlox VecEnv > TorchRL ParallelEnv for small N | 2-5x | Rayon vs multiprocess + TensorDict |
| H6 | rlox GAE > Python loop for T > 128 | 5-20x | Compiled vs interpreted loop |
| H7 | rlox GAE ~ TorchRL GAE for large T | 0.5-2x | Both compiled (Rust vs C++/ATen) |
| H8 | rlox buffer push > SB3 buffer push | 2-5x | Less per-push Python overhead |
| H9 | End-to-end rlox > SB3 for PPO rollout | 3-10x | Compounding advantages |

## Implementation Plan

### File Structure

```
benchmarks/
  conftest.py              (existing — shared infrastructure)
  bench_env_stepping.py    (existing — extend with TorchRL)
  bench_buffer_ops.py      (NEW — B3, B4)
  bench_gae.py             (NEW — B5)
  bench_e2e_rollout.py     (NEW — B6)
  bench_llm_ops.py         (NEW — B7)
  run_all.py               (existing — add new categories)
tests/python/
  test_bench_three_way.py  (NEW — TDD tests validating benchmark results)
```

### TDD Steps

1. Write `test_bench_three_way.py` with xfail tests defining expected benchmark results
2. Implement `bench_buffer_ops.py` — rlox vs TorchRL vs SB3 buffer push/sample
3. Implement `bench_gae.py` — rlox vs TorchRL vs SB3/numpy GAE
4. Extend `bench_env_stepping.py` — add TorchRL env benchmarks
5. Implement `bench_e2e_rollout.py` — full pipeline comparison
6. Update `run_all.py` to orchestrate all benchmarks
7. Remove xfail markers as benchmarks pass

### Dependencies

```
pip install torchrl stable-baselines3 gymnasium[classic-control,box2d]
```

TorchRL and SB3 are optional — benchmarks skip gracefully if not installed.

## Fairness Notes

- **rlox CartPole is native Rust**: TorchRL/SB3 delegate to Gymnasium's Python CartPole. This is a legitimate comparison (same dynamics), clearly documented.
- **TorchRL TensorDict overhead**: TensorDict construction is part of the user-facing cost. Measure both full API and pre-constructed variants.
- **SB3 stores next_obs**: SB3's ReplayBuffer takes `next_obs` separately — provide it for fair comparison.
- **CPU only**: No GPU benchmarks. TorchRL can offload to GPU but that's a different comparison.
- **GIL**: rlox releases GIL in Rust. TorchRL releases in C++ kernels. SB3 is GIL-bound. Document this.

## Measured Results (Apple M4, Python 3.12, PyTorch 2.10)

### Buffer Operations

| Benchmark | rlox | TorchRL | SB3 | vs TorchRL | vs SB3 |
|-----------|------|---------|-----|------------|--------|
| push (obs=4, 10K) | 1.5 ms | 229 ms | 15.0 ms | **148x** | **9.7x** |
| push (obs=28224, 10K) | 135 ms | 257 ms | 115 ms | **1.9x** | 0.9x |
| sample (batch=32) | 1.5 us | 20 us | 18 us | **13x** | **11x** |
| sample (batch=256) | 4.2 us | 29 us | 41 us | **6.8x** | **9.8x** |
| sample (batch=1024) | 9.2 us | 96 us | 75 us | **10x** | **8.1x** |

### GAE Computation

| Trajectory Length | rlox | NumPy Loop | TorchRL | vs NumPy | vs TorchRL |
|------------------|------|-----------|---------|----------|------------|
| 128 | 0.7 us | 34 us | 453 us | **51x** | **679x** |
| 512 | 1.2 us | 149 us | 1725 us | **119x** | **1380x** |
| 2048 | 4.0 us | 558 us | 6798 us | **139x** | **1700x** |
| 8192 | 16 us | 2229 us | 27005 us | **141x** | **1703x** |
| 32768 | 60 us | 8906 us | 108441 us | **147x** | **1791x** |

### LLM Operations

| Benchmark | rlox | NumPy | PyTorch | vs NumPy | vs PyTorch |
|-----------|------|-------|---------|----------|------------|
| GRPO 16×4 | 2.8 us | 79 us | 81 us | **28x** | **29x** |
| GRPO 256×16 | 36 us | 1252 us | 1241 us | **35x** | **34x** |
| Token KL 128 | 0.4 us | 1.7 us | 2.5 us | **4.0x** | **5.9x** |
| Token KL 8192 | 17 us | 28 us | 51 us | **1.6x** | **3.0x** |

### End-to-End Rollout

| Config | rlox | SB3 | TorchRL | vs SB3 | vs TorchRL |
|--------|------|-----|---------|--------|------------|
| 16 envs × 128 steps | 6.1 ms | 10.2 ms | 129 ms | **1.7x** | **21x** |
| 64 envs × 512 steps | 44 ms | 135 ms | 1768 ms | **3.1x** | **41x** |
| 256 envs × 2048 steps | 539 ms | 2080 ms | 28432 ms | **3.9x** | **53x** |

## Output

Each benchmark run produces:
- JSON artifact in `benchmark_results/` with full timing data
- Console summary table with speedup ratios and significance markers
- README auto-update (via existing `scripts/update_readme.py`)
