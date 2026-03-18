# Benchmark: GAE Computation

Generalized Advantage Estimation — the core advantage calculation used by PPO and most on-policy RL algorithms. GAE is inherently sequential (backward scan with data dependencies), so speedup comes from eliminating Python interpreter overhead rather than parallelism.

## What is Measured

Each framework computes GAE over a single trajectory of length T:
- **rlox**: `compute_gae(rewards, values, dones, last_value, gamma, lam)` — Rust backward loop, accepts numpy arrays
- **TorchRL**: `generalized_advantage_estimate(gamma, lam, values, next_values, rewards, dones, terminated)` — PyTorch C++ kernel via TensorDict
- **NumPy loop**: Python `for t in reversed(range(n))` loop — identical to CleanRL and SB3 internals

Parameters: `gamma=0.99`, `lam=0.95`, dones sampled at ~5% rate, `seed=42`.

## Correctness

All implementations validated against the NumPy reference to within `rtol=1e-6, atol=1e-10` before benchmarking.

## Results

### Raw Timings

| Trajectory Length | rlox (median) | NumPy Loop (median) | TorchRL (median) |
|------------------|---------------|---------------------|------------------|
| 128 steps | 0.7 us | 30.4 us | 423.3 us |
| 512 steps | 1.2 us | 135.4 us | 1,600.4 us |
| 2,048 steps | 3.9 us | 557.4 us | 6,402.6 us |
| 8,192 steps | 16.6 us | 2,166.2 us | 25,712.9 us |
| 32,768 steps | 54.7 us | 8,664.0 us | 103,448.2 us |

### Speedup vs NumPy Loop (Python)

| Trajectory Length | Speedup | 95% CI |
|------------------|---------|--------|
| 128 steps | **45.5x** | [45.4, 45.7] |
| 512 steps | **116.0x** | [115.7, 116.2] |
| 2,048 steps | **142.3x** | [140.7, 148.6] |
| 8,192 steps | **130.6x** | [119.9, 136.1] |
| 32,768 steps | **158.4x** | [157.3, 158.6] |

### Speedup vs TorchRL

| Trajectory Length | Speedup | 95% CI |
|------------------|---------|--------|
| 128 steps | **634.7x** | [632.1, 636.4] |
| 512 steps | **1,371.4x** | [1366.9, 1375.5] |
| 2,048 steps | **1,634.6x** | [1615.6, 1707.4] |
| 8,192 steps | **1,550.6x** | [1422.9, 1617.6] |
| 32,768 steps | **1,890.9x** | [1876.8, 1898.7] |

## Analysis

### Why rlox is 140x faster than Python loops

The GAE backward scan is:
```
for t in reversed(range(n)):
    delta = rewards[t] + gamma * next_value * non_terminal - values[t]
    last_gae = delta + gamma * lam * non_terminal * last_gae
    advantages[t] = last_gae
```

In Python, each iteration pays:
- Python bytecode dispatch (~50ns)
- Float object creation for intermediate results
- Array indexing overhead (bounds checking, type dispatch)

In Rust, the same loop compiles to ~5 instructions with no allocation. At 2048 steps, that's 2048 × ~50ns = ~100us of Python overhead eliminated.

### Why TorchRL is 1635x slower

TorchRL's `generalized_advantage_estimate` operates on `TensorDict` objects. Each step in the computation involves:
- TensorDict metadata validation
- PyTorch tensor operation dispatch (even for scalar ops)
- TensorDict key lookups

This per-element overhead dominates the actual arithmetic. TorchRL's GAE is designed for composability within its TensorDict ecosystem, not for raw computation speed.

### Scaling behavior

The rlox vs NumPy speedup increases from 46x (128 steps) to 158x (32768 steps) because:
- At small T, rlox's fixed PyO3 boundary-crossing overhead is a larger fraction of total time
- At large T, both converge to their steady-state per-step cost, and the ratio stabilizes around 140x

Source: [`benchmarks/bench_gae.py`](../../benchmarks/bench_gae.py)
