# rlox Benchmark Results

Three-framework performance comparison: **rlox** (Rust/PyO3) vs **TorchRL** (PyTorch) vs **Stable-Baselines3** (Python/NumPy).

> Measured on Apple M4 (14 cores), macOS 26.2, Python 3.12.7, PyTorch 2.10.0.
> All speedups reported with bootstrap 95% confidence intervals (10,000 resamples).
> Last updated: 2026-03-14.

## Summary

### Infrastructure Operations

| Category | vs SB3 | vs TorchRL | vs NumPy |
|----------|--------|------------|----------|
| [GAE (2048 steps)](gae.md) | — | **1,635x** | **142x** |
| [Buffer push (CartPole)](buffer-ops.md) | **4.0x** | **60.8x** | — |
| [Buffer sample (batch=1024)](buffer-ops.md) | **6.1x** | **6.5x** | — |
| [E2E rollout (256×2048)](e2e-rollout.md) | **3.0x** | **40.4x** | — |
| [Env stepping (256 envs)](env-stepping.md) | — | **120x** | **6.7x** (Gym) |
| [GRPO advantages (256×16)](llm-ops.md) | — | — | **34x** |
| [Token KL (128 tokens)](llm-ops.md) | — | — | **4.7x** |

### Neural Network Backends (Burn vs Candle vs PyTorch)

| Category | Burn | Candle | PyTorch |
|----------|------|--------|---------|
| [DQN TD step (batch=64)](nn-backends.md) | 191 us | **98 us** | 738 us |
| [PPO step (batch=64)](nn-backends.md) | 1,885 us | **328 us** | 1,440 us |
| [SAC sample (batch=1)](nn-backends.md) | 91 us | **14 us** | 52 us |
| [Critic step (batch=256)](nn-backends.md) | **2,090 us** | 3,453 us | 2,325 us |

## Detailed Reports

| Report | What it measures | Frameworks |
|--------|-----------------|------------|
| [Setup & Methodology](setup.md) | Hardware, software versions, statistical methods, fairness constraints | — |
| [GAE Computation](gae.md) | Generalized Advantage Estimation at various trajectory lengths | rlox, TorchRL, NumPy |
| [Buffer Operations](buffer-ops.md) | Push throughput and sample latency | rlox, TorchRL, SB3 |
| [End-to-End Rollout](e2e-rollout.md) | Full pipeline: step + store + GAE | rlox, SB3, TorchRL |
| [Environment Stepping](env-stepping.md) | Single-step latency, vectorized throughput scaling | rlox, Gymnasium, SB3, TorchRL |
| [LLM Operations](llm-ops.md) | GRPO advantages, token-level KL divergence | rlox, NumPy, PyTorch |
| [NN Backends](nn-backends.md) | Inference and training step latency | Burn, Candle, PyTorch |

## Key Findings

1. **GAE is the standout result**: 142x vs Python loops because the sequential backward scan eliminates per-iteration interpreter overhead. 1,635x vs TorchRL due to TensorDict metadata overhead per element.

2. **Buffer operations scale inversely with observation size**: At CartPole (obs=4), rlox is 61x faster than TorchRL on push — per-call overhead dominates. At Atari-sized observations (obs=28,224), the gap narrows to 1.8x as memcpy dominates. SB3 edges ahead at Atari scale (0.8x) because pre-allocated NumPy arrays avoid reallocation.

3. **End-to-end advantage compounds**: Individual speedups (env stepping: ~7x, buffer: ~4x, GAE: ~142x) compound to 3.0x vs SB3 at the largest configuration. The GAE advantage is diluted because env stepping dominates wall clock time.

4. **Env stepping scales with parallelism**: rlox reaches **2.7M steps/s** at 512 envs (8.2x vs Gymnasium). At low env counts (4), Rayon thread pool overhead makes rlox slower than sequential Python. The crossover is at ~16 envs.

5. **Tail latency matters**: rlox's buffer sample p99 is 17.0us at batch=1024, vs 109-75us for TorchRL/SB3. Pre-allocated ring buffer + deterministic ChaCha8 RNG eliminates GC pressure.

6. **Small-array operations favor rlox**: GRPO advantages (34x) and token KL at short sequences (4.7x) win because each Python function call to NumPy/PyTorch costs ~500ns in dispatch overhead. rlox's PyO3 boundary crossing costs ~100ns.

7. **Candle excels at low-latency inference**: At batch=1, Candle is 5-6x faster than Burn and 3-4x faster than PyTorch for inference. At larger batches (256+), Burn's GEMM-optimized NdArray backend catches up and often wins for training steps.

## Reproducing

```bash
# Python benchmarks (infrastructure ops)
python benchmarks/run_all.py                   # Full suite
python benchmarks/bench_gae.py                 # Just GAE
python benchmarks/bench_buffer_ops.py          # Just buffers
python benchmarks/bench_e2e_rollout.py         # Just E2E
python benchmarks/bench_llm_ops.py             # Just LLM ops
python benchmarks/bench_env_stepping.py        # Just env stepping
python benchmarks/bench_nn_backends.py         # PyTorch NN baseline

# Rust benchmarks (NN backends)
cargo bench -p rlox-bench --bench nn_backends  # Burn vs Candle
```

Raw JSON data is written to `benchmark_results/` (gitignored). Criterion HTML reports are in `target/criterion/`.
