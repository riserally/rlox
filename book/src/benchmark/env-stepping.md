# Benchmark: Environment Stepping

Single-step latency and vectorized throughput — the foundation of every RL training loop. Compares rlox (Rust/Rayon) against Gymnasium, TorchRL, and Stable-Baselines3.

## What is Measured

### Single-Step Latency
One CartPole step + reset-on-done. Measures per-call overhead without vectorization.

| Framework | Implementation |
|-----------|---------------|
| rlox | `CartPole.step(1)` — native Rust, PyO3 boundary |
| Gymnasium | `gym.make("CartPole-v1").step(1)` — Python/C |
| TorchRL | `GymEnv("CartPole-v1").step(td)` — TensorDict wrapper over Gymnasium |

### Vectorized Throughput
Step N environments in lockstep for 100 batch-steps. Measures parallelism scaling.

| Framework | Implementation |
|-----------|---------------|
| rlox `VecEnv` | Rayon thread pool, true parallelism across cores |
| Gymnasium `SyncVectorEnv` | Sequential Python loop |
| Gymnasium `AsyncVectorEnv` | Multiprocessing with shared memory |
| SB3 `DummyVecEnv` | Sequential Python loop |
| SB3 `SubprocVecEnv` | Multiprocessing with pipes |
| TorchRL `SerialEnv` | Sequential with TensorDict wrapping |
| TorchRL `ParallelEnv` | Multiprocessing (crashes at >1 env, see note) |

## Results

### Single-Step Latency

| Framework | Median | IQR |
|-----------|--------|-----|
| rlox | 292 ns | 42 ns |
| Gymnasium | 2,375 ns | 125 ns |
| TorchRL | 52,834 ns | 8,968 ns |

| Comparison | Speedup | 95% CI |
|-----------|---------|--------|
| vs Gymnasium | **8.1x** | [8.1, 8.1] |
| vs TorchRL | **180.9x** | [179.7, 182.4] |

### Vectorized Throughput

| Num Envs | rlox (ms) | rlox (steps/s) | Gym Sync (ms) | vs Gym Sync | vs SB3 Dummy | vs TorchRL Serial |
|----------|-----------|----------------|---------------|-------------|-------------|-------------------|
| 1 | 0.07 | 1,472,852 | 0.60 | **8.9x** | **9.8x** | **153x** |
| 4 | 3.61 | 110,859 | 1.49 | 0.4x | 0.6x | **16x** |
| 16 | 2.10 | 762,124 | 4.86 | **2.3x** | **2.9x** | **43x** |
| 64 | 4.44 | 1,441,739 | 18.12 | **4.1x** | **5.0x** | **80x** |
| 128 | 5.44 | 2,353,969 | 37.37 | **6.9x** | **8.6x** | **136x** |
| 256 | 12.44 | 2,058,154 | 82.79 | **6.7x** | — | **120x** |
| 512 | 19.11 | 2,679,404 | 156.12 | **8.2x** | — | — |

### Bridge Overhead

| Mode | Median |
|------|--------|
| Native rlox CartPole | 334 ns |
| GymEnv bridge (rlox wrapping Gymnasium) | 2,980 ns |
| Bridge overhead | 2,646 ns (2.6 us) |

## Analysis

### Why rlox loses at 4 envs

At 4 envs, rlox (3.6ms) is **slower** than Gymnasium sync (1.5ms). This is the Rayon thread pool startup cost: for only 4 lightweight CartPole steps (~37ns each), the overhead of dispatching to Rayon threads and synchronizing exceeds the work itself. The crossover happens at ~16 envs where the parallel work justifies the thread pool overhead.

### Scaling behavior

rlox throughput scales from 111K steps/s (4 envs) to **2.7M steps/s** (512 envs) — a 24x increase. This is near-linear scaling across the M4's cores. Gymnasium sync plateaus at ~330K steps/s regardless of env count because it's sequential.

### TorchRL overhead

TorchRL `SerialEnv` is 120-153x slower than rlox across all scales. The TensorDict metadata wrapping per step adds ~100us overhead for a ~37ns computation. TorchRL `ParallelEnv` crashes at >1 env with a tensor size mismatch error (TorchRL bug, not rlox).

### Multiprocessing (SB3 SubprocVecEnv, Gymnasium AsyncVectorEnv)

Subprocess-based parallelism has high fixed overhead from IPC (pipes, shared memory). At 1 env, SB3 SubprocVecEnv is 163x slower than rlox. At 64 envs, the gap narrows to 7.3x as the parallel work amortizes IPC cost. rlox's in-process Rayon parallelism avoids IPC entirely.

Source: [`benchmarks/bench_env_stepping.py`](../../benchmarks/bench_env_stepping.py)
