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

| Trajectory Length | rlox (median) | rlox (p99) | NumPy Loop (median) | TorchRL (median) |
|------------------|---------------|------------|---------------------|------------------|
| 128 steps | 0.7 us | 0.8 us | 34.3 us | 453.1 us |
| 512 steps | 1.2 us | 1.3 us | 148.5 us | 1,725.0 us |
| 2,048 steps | 4.0 us | 5.0 us | 557.5 us | 6,798.4 us |
| 8,192 steps | 15.9 us | 18.5 us | 2,228.9 us | 27,004.7 us |
| 32,768 steps | 60.5 us | 72.0 us | 8,906.0 us | 108,441.3 us |

### Speedup vs NumPy Loop (Python)

| Trajectory Length | Speedup | 95% CI |
|------------------|---------|--------|
| 128 steps | **51.4x** | [51.1, 52.2] |
| 512 steps | **118.8x** | [118.1, 119.2] |
| 2,048 steps | **139.4x** | [137.2, 141.2] |
| 8,192 steps | **140.6x** | [139.4, 141.7] |
| 32,768 steps | **147.1x** | [145.9, 151.4] |

### Speedup vs TorchRL

| Trajectory Length | Speedup | 95% CI |
|------------------|---------|--------|
| 128 steps | **679.3x** | [669.9, 689.8] |
| 512 steps | **1,380.0x** | [1364.1, 1391.2] |
| 2,048 steps | **1,699.6x** | [1690.3, 1722.6] |
| 8,192 steps | **1,703.3x** | [1690.5, 1716.2] |
| 32,768 steps | **1,791.2x** | [1778.7, 1843.1] |

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

### Why TorchRL is 1700x slower

TorchRL's `generalized_advantage_estimate` operates on `TensorDict` objects. Each step in the computation involves:
- TensorDict metadata validation
- PyTorch tensor operation dispatch (even for scalar ops)
- TensorDict key lookups

This per-element overhead dominates the actual arithmetic. TorchRL's GAE is designed for composability within its TensorDict ecosystem, not for raw computation speed.

### Scaling behavior

The rlox vs NumPy speedup increases from 51x (128 steps) to 147x (32768 steps) because:
- At small T, rlox's fixed PyO3 boundary-crossing overhead is a larger fraction of total time
- At large T, both converge to their steady-state per-step cost, and the ratio stabilizes around 140x

Source: [`benchmarks/bench_gae.py`](../../benchmarks/bench_gae.py)
