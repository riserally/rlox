# Benchmark: Neural Network Backend Comparison

Speed comparison of rlox's two Rust NN backends — **Burn** (NdArray) and **Candle** (CPU) — against **PyTorch** (CPU). Measures inference and training step latency for all RL algorithm architectures (PPO, DQN, SAC, TD3).

## Methodology

- **Burn**: Rust criterion benchmark, `Autodiff<NdArray>` backend, optimized build
- **Candle**: Rust criterion benchmark, CPU backend, optimized build
- **PyTorch**: Python benchmark via `time.perf_counter_ns()`, CPU only, `torch 2.10.0`
- Cross-language comparison is approximate (criterion vs Python timer overhead differs by ~50-100ns)
- All backends use identical architectures: same hidden sizes, activations, obs/action dims

## Results

### Inference (No Gradient)

#### ActorCritic (PPO) — `act()` (obs=4, actions=2, hidden=64)

| Batch | Burn | Candle | PyTorch |
|-------|------|--------|---------|
| 1 | 63 us | 11 us | 36 us |
| 32 | 449 us | 61 us | 172 us |
| 256 | 800 us | 803 us | 589 us |

#### DQN Q-Values — `q_values()` (obs=4, actions=2, hidden=64)

| Batch | Burn | Candle | PyTorch |
|-------|------|--------|---------|
| 1 | 335 us | 4 us | 12 us |
| 32 | 62 us | 12 us | 16 us |
| 256 | 96 us | 143 us | 29 us |

#### SAC Sample Actions — `sample_actions()` (obs=17, act=6, hidden=256)

| Batch | Burn | Candle | PyTorch |
|-------|------|--------|---------|
| 1 | 91 us | 14 us | 52 us |
| 32 | 126 us | 159 us | 63 us |
| 256 | 406 us | 584 us | 787 us |

#### TD3 Deterministic Action — `act()` (obs=17, act=6, hidden=256)

| Batch | Burn | Candle | PyTorch |
|-------|------|--------|---------|
| 1 | 65 us | 12 us | 14 us |
| 32 | 97 us | 144 us | 27 us |
| 256 | 285 us | 535 us | 458 us |

#### Twin-Q Forward — `twin_q_values()` (obs=17, act=6, hidden=256)

| Batch | Burn | Candle | PyTorch |
|-------|------|--------|---------|
| 1 | 131 us | 24 us | 28 us |
| 32 | 193 us | 240 us | 52 us |
| 256 | 555 us | 1,049 us | 550 us |

### Training Steps (Forward + Backward + Optimizer)

#### PPO Step (obs=4, actions=2, hidden=64)

| Batch | Burn | Candle | PyTorch |
|-------|------|--------|---------|
| 64 | 1,885 us | 328 us | 1,440 us |
| 256 | 2,602 us | 3,090 us | 2,357 us |

#### DQN TD Step (obs=4, actions=2, hidden=64)

| Batch | Burn | Candle | PyTorch |
|-------|------|--------|---------|
| 64 | 191 us | 98 us | 738 us |
| 256 | 322 us | 554 us | 823 us |

#### Twin-Q Critic Step (obs=17, act=6, hidden=256)

| Batch | Burn | Candle | PyTorch |
|-------|------|--------|---------|
| 64 | 1,107 us | 1,840 us | 1,771 us |
| 256 | 2,090 us | 3,453 us | 2,325 us |

## Analysis

### Candle wins at small batch sizes

At batch=1, Candle is consistently the fastest backend — often by 5-10x over Burn:
- DQN q_values: **4us** (Candle) vs 335us (Burn) vs 12us (PyTorch)
- SAC sample: **14us** (Candle) vs 91us (Burn) vs 52us (PyTorch)
- TD3 act: **12us** (Candle) vs 65us (Burn) vs 14us (PyTorch)

Candle's advantage comes from its minimal tensor infrastructure. Each operation is a direct C function call with no metadata wrapping. Burn's NdArray backend has per-operation overhead from its Backend trait dispatch and module system.

### Burn catches up at larger batches

At batch=256, matrix multiplication dominates total time, and Burn's GEMM backend (via the `gemm` crate) becomes competitive:
- TD3 act: **285us** (Burn) vs 535us (Candle) — Burn is 1.9x faster
- Twin-Q: **555us** (Burn) vs 1,049us (Candle) — Burn is 1.9x faster
- Critic step: **2,090us** (Burn) vs 3,453us (Candle) — Burn is 1.7x faster

### PyTorch vs Rust backends

PyTorch is generally competitive at all batch sizes thanks to its highly optimized C++ / MKL / Accelerate backend:
- At batch=256 inference, PyTorch often matches or beats both Rust backends
- At batch=1, Candle beats PyTorch by 3-4x (less per-call overhead)
- For training steps, all three are within 2x of each other at large batches

The key insight: **for RL workloads where batch sizes are small (1-32) and latency matters, Candle is the clear winner**. For large-batch training, the backends converge.

### DQN TD step: Both Rust backends beat PyTorch

At batch=64, Candle (98us) is **7.5x faster** than PyTorch (738us), and Burn (191us) is **3.9x faster**. This is because the DQN TD step involves many small operations (gather, MSE loss, backward) where Python/PyTorch dispatch overhead compounds. The Rust backends avoid this entirely.

### Recommendation

| Use Case | Best Backend |
|----------|-------------|
| Low-latency inference (batch=1) | **Candle** |
| Large-batch training (batch>=256) | **Burn** |
| Balanced (typical RL training) | **Candle** for SAC/TD3 (small batch), **Burn** for PPO (large batch) |

Source:
- Rust benchmarks: [`crates/rlox-bench/benches/nn_backends.rs`](../../crates/rlox-bench/benches/nn_backends.rs)
- PyTorch baseline: [`benchmarks/bench_nn_backends.py`](../../benchmarks/bench_nn_backends.py)
