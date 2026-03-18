# Benchmark: rlox vs TRL-style Operations

Compares rlox's Rust primitives against the equivalent PyTorch operations used by [TRL](https://github.com/huggingface/trl) (HuggingFace's Transformer Reinforcement Learning library) for LLM post-training computations.

> **Important**: This compares isolated numerical primitives, not full training pipelines. In a real LLM training step, generation and backward passes dominate >95% of wall-clock. See [End-to-End Impact](#end-to-end-impact) for context.

## Methodology

- **rlox**: Rust via PyO3, CPU, f64 precision
- **TRL-style PyTorch**: Standalone reimplementation of TRL's internal computations on CPU, f32 precision
- **NumPy**: Vectorized NumPy baseline
- TRL's GRPO advantages extracted from `GRPOTrainer._compute_rewards_and_advantages` (v0.16+)
- TRL's KL estimator extracted from `GRPOTrainer._compute_loss` — Schulman (2020): `exp(r) - r - 1`
- Correctness verified: rlox and TRL-style produce numerically equivalent results (accounting for population vs sample std, f64 vs f32)

## Results

### GRPO Group Advantages

rlox uses a **batched** API: single Rust call for all groups, avoiding per-group PyO3 boundary crossings.

| Config | Total | rlox (batched) | rlox (loop) | TRL-style CPU | NumPy | vs TRL | vs NumPy |
|--------|-------|---------------|-------------|--------------|-------|--------|----------|
| 16 x 4 | 64 | **0.5 us** | 2.0 us | 6.5 us | 5.9 us | **14.2x** *** | **13.0x** *** |
| 64 x 8 | 512 | **1.6 us** | 8.0 us | 11.6 us | 8.1 us | **7.1x** *** | **5.0x** *** |
| 256 x 16 | 4,096 | **6.6 us** | 36.6 us | 44.3 us | 15.9 us | **6.7x** *** | **2.4x** *** |
| 1024 x 32 | 32,768 | **53.5 us** | 153.4 us | 212.5 us | 67.9 us | **4.0x** *** | **1.3x** *** |

### Token KL Divergence (Schulman Estimator)

rlox uses per-sequence Rust calls in a Python loop. TRL/NumPy operate on batched `(B, T)` tensors.

#### Single Sequence (B=1)

| Seq Length | rlox | TRL-style CPU | NumPy | vs TRL | vs NumPy |
|-----------|------|--------------|-------|--------|----------|
| 128 | **0.5 us** | 2.5 us | 1.2 us | **5.5x** *** | **2.5x** *** |
| 512 | **1.1 us** | 3.0 us | 2.2 us | **2.7x** *** | **2.0x** *** |
| 2,048 | **4.0 us** | 4.0 us | 6.0 us | 1.0x | **1.5x** *** |
| 8,192 | **17.0 us** | 46.1 us | 26.1 us | **2.7x** *** | **1.5x** *** |

#### Batched (B=32)

| Seq Length | Total Elements | rlox | TRL-style CPU | NumPy | vs TRL | vs NumPy |
|-----------|---------------|------|--------------|-------|--------|----------|
| 128 | 4,096 | 12.4 us | 41.6 us | **11.1 us** | **3.4x** *** | 0.9x |
| 512 | 16,384 | **39.2 us** | 52.3 us | 49.2 us | **1.3x** *** | **1.3x** *** |
| 2,048 | 65,536 | 247.4 us | **156.7 us** | 284.8 us | 0.6x | **1.2x** *** |

## Analysis

### GRPO: Batched API eliminates PyO3 overhead

The batched `compute_batch_group_advantages` is 3-5x faster than the per-group loop because it amortizes PyO3 boundary crossing cost. Each crossing costs ~100ns; for 256 groups, the loop pays 25.6us in crossing alone. The batched API pays once.

TRL's vectorized `reshape + mean + std + repeat_interleave` approach pays 6 PyTorch dispatch points per call (~500ns each on CPU), totaling ~3us of dispatch overhead. rlox's tight Rust loop avoids this entirely.

### KL: rlox wins at short sequences, loses at large batched operations

At B=1 and short sequences (128-512 tokens), rlox's advantage is clear: 2.7-5.5x faster than TRL-style PyTorch. The Rust loop over elements is faster than PyTorch's per-operation dispatch.

At B=32 x T=2048 (65K elements), TRL-style PyTorch CPU wins (0.6x) because:
1. rlox pays 32 PyO3 crossings (one per sequence)
2. PyTorch's tensor ops are SIMD-vectorized over the full (32, 2048) tensor in one call
3. At this scale, compute dominates over dispatch overhead

A batched Rust KL function would close this gap.

### Population vs Sample Standard Deviation

rlox uses population std (`ddof=0`): `sqrt(sum((x - mean)^2) / N)`. PyTorch's `Tensor.std()` uses sample std (`ddof=1`): `sqrt(sum((x - mean)^2) / (N-1))`. For group sizes K >= 16, the difference is < 3% and advantages are directionally identical.

## End-to-End Impact

In a real GRPO training step on a Llama-8B-class model:

| Component | Typical Time | Fraction |
|-----------|-------------|----------|
| Generation (model.generate) | ~2-10s | 60-80% |
| Backward pass | ~0.5-2s | 15-25% |
| Forward pass (logprob extraction) | ~0.2-0.5s | 5-10% |
| **Advantage computation** | **~0.05-0.2ms** | **<0.01%** |
| **KL computation** | **~0.01-0.05ms** | **<0.01%** |
| Data marshalling / logging | ~10-50ms | 0.5-2% |

**rlox's 7-14x speedup on GRPO advantages translates to <0.01% wall-clock improvement in a full LLM training step.** The value proposition is not end-to-end training speed but:

1. **Lower tail latency** for real-time serving (RLHF reward scoring in inference pipelines)
2. **CPU-only deployments** where PyTorch overhead matters (edge devices, inference servers)
3. **Composability** with other rlox primitives (GAE, buffers) in hybrid Rust/Python RL pipelines

## Reproducing

```bash
python benchmarks/bench_trl_comparison.py
```

No TRL installation required — TRL's computations are replicated as standalone PyTorch functions.

Source: [`benchmarks/bench_trl_comparison.py`](../../benchmarks/bench_trl_comparison.py)
