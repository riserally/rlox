# Benchmark: LLM Post-Training Operations

Operations used in LLM alignment / post-training (GRPO, RLHF, DPO). No TorchRL or SB3 equivalents exist — comparison is against NumPy and PyTorch baselines.

## What is Measured

### GRPO Group Advantages
Group-Relative Policy Optimization computes z-score normalization per completion group: `(reward - mean) / std`.

| Framework | Implementation |
|-----------|---------------|
| rlox | `compute_group_advantages(rewards)` — Rust, f64, Welford's online algorithm |
| NumPy | `(rewards - rewards.mean()) / rewards.std()` — vectorized |
| PyTorch | `(g - g.mean()) / g.std()` — tensor ops |

Tested with `n_prompts × k_completions` groups: (16×4), (64×8), (256×16).

### Token-Level KL Divergence
KL(policy ‖ reference) = `sum(exp(log_p) * (log_p - log_q))`. Used in PPO-based RLHF to constrain policy drift from the reference model.

| Framework | Implementation |
|-----------|---------------|
| rlox | `compute_token_kl(log_p, log_q)` — Rust f64 loop |
| NumPy | `np.sum(np.exp(log_p) * (log_p - log_q))` — vectorized |
| PyTorch | `torch.sum(torch.exp(log_p) * (log_p - log_q)).item()` — BLAS-backed |

Tested at sequence lengths: 128, 512, 2048, 8192.

## Results

### GRPO Group Advantages

| Config | rlox (median) | rlox (p99) | NumPy (median) | PyTorch (median) |
|--------|---------------|------------|----------------|------------------|
| 16 × 4 | 2.8 us | 3.2 us | 78.7 us | 81.1 us |
| 64 × 8 | 8.8 us | 9.0 us | 328.6 us | 327.7 us |
| 256 × 16 | 36.3 us | 42.0 us | 1,252.1 us | 1,240.8 us |

| Config | vs NumPy | 95% CI | vs PyTorch | 95% CI |
|--------|----------|--------|------------|--------|
| 16 × 4 | **28.0x** | [26.7, 36.8] | **28.8x** | [27.7, 38.1] |
| 64 × 8 | **37.2x** | [36.2, 37.5] | **37.1x** | [36.0, 37.5] |
| 256 × 16 | **34.5x** | [34.2, 34.7] | **34.1x** | [33.8, 34.7] |

### Token-Level KL Divergence

| Seq Length | rlox (median) | rlox (p99) | NumPy (median) | PyTorch (median) |
|-----------|---------------|------------|----------------|------------------|
| 128 | 0.4 us | 0.5 us | 1.7 us | 2.5 us |
| 512 | 1.1 us | 1.2 us | 2.6 us | 2.9 us |
| 2,048 | 4.4 us | 5.4 us | 7.0 us | 6.5 us |
| 8,192 | 17.1 us | 22.0 us | 27.7 us | 50.6 us |

| Seq Length | vs NumPy | 95% CI | vs PyTorch | 95% CI |
|-----------|----------|--------|------------|--------|
| 128 | **4.0x** | [3.9, 4.2] | **5.9x** | [5.9, 5.9] |
| 512 | **2.3x** | [2.3, 2.3] | **2.6x** | [2.5, 2.6] |
| 2,048 | **1.6x** | [1.6, 1.6] | **1.5x** | [1.5, 1.5] |
| 8,192 | **1.6x** | [1.6, 1.6] | **3.0x** | [2.8, 3.1] |

## Analysis

### GRPO: Why 34x faster

GRPO processes many small groups (4–16 elements each). For 256 prompts × 16 completions, that's 256 separate calls to compute mean/std/normalize.

Each Python call to `np.mean()` + `np.std()` on a 16-element array incurs:
- Function call overhead (~500ns)
- NumPy type dispatch
- Array creation for the result

rlox crosses the PyO3 boundary once per group and computes mean + std + normalization in a single Rust pass with no allocation. The per-call overhead drops from ~5us (Python) to ~140ns (Rust).

NumPy and PyTorch perform nearly identically here — both are limited by Python function call overhead, not by the actual arithmetic.

### Token KL: Advantage narrows at large sequences

At seq_len=128, rlox is 4x faster than NumPy. At seq_len=8192, only 1.6x.

The reason: NumPy's vectorized `exp()` and `sum()` call into optimized BLAS/LAPACK routines (via Accelerate framework on macOS). For large arrays, these SIMD-optimized C routines approach the same throughput as Rust's f64 loop. The rlox advantage at small sizes comes from avoiding NumPy's per-call dispatch overhead.

PyTorch's surprising regression at 8192 (3.0x slower than rlox, worse than NumPy) is likely due to PyTorch's larger per-op dispatch overhead compared to NumPy for simple element-wise operations.

Source: [`benchmarks/bench_llm_ops.py`](../../benchmarks/bench_llm_ops.py)
