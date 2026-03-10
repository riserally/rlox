# rlox Benchmark Results

Three-framework performance comparison: **rlox** (Rust/PyO3) vs **TorchRL** (PyTorch) vs **Stable-Baselines3** (Python/NumPy).

> Measured on Apple M4 (14 cores), macOS 26.2, Python 3.12.7, PyTorch 2.10.0.
> All speedups reported with bootstrap 95% confidence intervals (10,000 resamples).

## Summary

| Category | vs SB3 | vs TorchRL | vs NumPy |
|----------|--------|------------|----------|
| [GAE (2048 steps)](gae.md) | — | **1,700x** | **139x** |
| [Buffer push (CartPole)](buffer-ops.md) | **9.7x** | **148x** | — |
| [Buffer sample (batch=1024)](buffer-ops.md) | **8.1x** | **10.4x** | — |
| [E2E rollout (256×2048)](e2e-rollout.md) | **3.9x** | **53x** | — |
| [GRPO advantages (256×16)](llm-ops.md) | — | — | **35x** |
| [Token KL (128 tokens)](llm-ops.md) | — | — | **4.0x** |

## Detailed Reports

| Report | What it measures | Frameworks |
|--------|-----------------|------------|
| [Setup & Methodology](setup.md) | Hardware, software versions, statistical methods, fairness constraints | — |
| [GAE Computation](gae.md) | Generalized Advantage Estimation at various trajectory lengths | rlox, TorchRL, NumPy |
| [Buffer Operations](buffer-ops.md) | Push throughput and sample latency | rlox, TorchRL, SB3 |
| [End-to-End Rollout](e2e-rollout.md) | Full pipeline: step + store + GAE | rlox, SB3, TorchRL |
| [LLM Operations](llm-ops.md) | GRPO advantages, token-level KL divergence | rlox, NumPy, PyTorch |

## Key Findings

1. **GAE is the standout result**: 140x vs Python loops because the sequential backward scan eliminates per-iteration interpreter overhead. 1,700x vs TorchRL due to TensorDict metadata overhead per element.

2. **Buffer operations scale inversely with observation size**: At CartPole (obs=4), rlox is 148x faster than TorchRL on push — per-call overhead dominates. At Atari-sized observations (obs=28,224), the gap narrows to 1.9x as memcpy dominates. SB3 actually edges ahead at Atari scale (0.9x) because pre-allocated NumPy arrays avoid reallocation.

3. **End-to-end advantage compounds**: Individual speedups (env stepping: ~5x, buffer: ~10x, GAE: ~140x) compound to 3.9x vs SB3 at the largest configuration. The GAE advantage is diluted because env stepping dominates wall clock time.

4. **Tail latency matters**: rlox's buffer sample p99 is 14.7us at batch=1024, vs 135-138us for TorchRL/SB3. Pre-allocated ring buffer + deterministic ChaCha8 RNG eliminates GC pressure.

5. **Small-array operations favor rlox**: GRPO advantages (34x) and token KL at short sequences (4x) win because each Python function call to NumPy/PyTorch costs ~500ns in dispatch overhead. rlox's PyO3 boundary crossing costs ~100ns.

## Reproducing

```bash
python benchmarks/run_all.py                   # Full suite
python benchmarks/bench_gae.py                 # Just GAE
python benchmarks/bench_buffer_ops.py          # Just buffers
python benchmarks/bench_e2e_rollout.py         # Just E2E
python benchmarks/bench_llm_ops.py             # Just LLM ops
python benchmarks/bench_env_stepping.py        # Just env stepping
```

Raw JSON data is written to `benchmark_results/` (gitignored).
