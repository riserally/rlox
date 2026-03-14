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

| Config | rlox (median) | NumPy (median) | PyTorch (median) |
|--------|---------------|----------------|------------------|
| 16 × 4 | 2.2 us | 79.0 us | 76.2 us |
| 64 × 8 | 8.6 us | 309.9 us | 320.3 us |
| 256 × 16 | 36.1 us | 1,243.3 us | 1,261.6 us |

| Config | vs NumPy | 95% CI | vs PyTorch | 95% CI |
|--------|----------|--------|------------|--------|
| 16 × 4 | **36.5x** | [35.7, 37.2] | **35.2x** | [35.0, 37.6] |
| 64 × 8 | **35.9x** | [35.5, 36.5] | **37.1x** | [36.6, 37.7] |
| 256 × 16 | **34.4x** | [34.2, 34.6] | **34.9x** | [34.6, 35.5] |

### Token-Level KL Divergence

| Seq Length | rlox (median) | NumPy (median) | PyTorch (median) |
|-----------|---------------|----------------|------------------|
| 128 | 0.4 us | 1.8 us | 2.0 us |
| 512 | 1.2 us | 2.8 us | 3.2 us |
| 2,048 | 3.9 us | 6.6 us | 5.8 us |
| 8,192 | 17.0 us | 26.8 us | 50.3 us |

| Seq Length | vs NumPy | 95% CI | vs PyTorch | 95% CI |
|-----------|----------|--------|------------|--------|
| 128 | **4.7x** | [4.7, 4.7] | **5.4x** | [5.4, 5.6] |
| 512 | **2.4x** | [2.3, 2.4] | **2.7x** | [2.6, 2.7] |
| 2,048 | **1.7x** | [1.7, 1.9] | **1.5x** | [1.5, 1.5] |
| 8,192 | **1.6x** | [1.6, 1.6] | **3.0x** | [2.8, 3.1] |

## Analysis

### GRPO: Why 35x faster

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
