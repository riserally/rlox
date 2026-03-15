# Benchmarking Plan: rlox vs TRL (HuggingFace Transformer Reinforcement Learning)

**Date**: 2026-03-14
**Status**: PROPOSED

---

## 1. Problem Statement

rlox implements low-level LLM post-training primitives in Rust (GRPO group advantages, token-level KL divergence, DPO loss scaffolding). TRL is HuggingFace's high-level Python library for LLM fine-tuning with RL, providing `GRPOTrainer`, `DPOTrainer`, `PPOTrainer`, etc. This plan defines a rigorous, honest benchmark comparing the two on overlapping operations.

### The fundamental asymmetry

rlox and TRL operate at different abstraction levels:

| Dimension | rlox | TRL |
|-----------|------|-----|
| Scope | Low-level compute primitives (advantages, KL, loss) | End-to-end trainers (data loading, model, optimization) |
| Compute target | CPU (Rust via PyO3) | GPU (PyTorch CUDA) or CPU |
| Data format | `numpy.ndarray` (f64) | `torch.Tensor` (f32/f16/bf16, possibly on GPU) |
| Model coupling | Model-free: operates on pre-extracted logits/rewards | Model-integrated: logit extraction is part of the trainer |

This asymmetry means we **cannot** do a simple wall-clock-to-wall-clock comparison of full training loops. Instead, we isolate the specific numerical computations that both libraries perform and compare those.

---

## 2. What Operations Are Comparable

### 2.1 GRPO Group Advantage Computation

**What rlox does**: `compute_group_advantages(rewards: ndarray[f64]) -> ndarray[f64]`
- Input: 1D array of K reward scalars for one prompt's completion group
- Output: z-score normalized advantages: `(r - mean) / std`
- Called N_prompts times per batch (once per prompt group)

**What TRL does** (in `GRPOTrainer._compute_rewards_and_advantages`):
```python
# TRL computes per-group advantages inside the trainer
# From trl/trainer/grpo_trainer.py (v0.16+):
mean_grouped_rewards = rewards.reshape(-1, self.num_generations).mean(dim=1)
std_grouped_rewards = rewards.reshape(-1, self.num_generations).std(dim=1)
advantages = (rewards - mean_grouped_rewards.repeat_interleave(self.num_generations))
advantages = advantages / (std_grouped_rewards.repeat_interleave(self.num_generations) + 1e-8)
```

**Comparison approach**: Extract TRL's advantage computation into a standalone function and benchmark it against rlox's `compute_group_advantages`. Both receive pre-computed reward tensors/arrays.

**Key difference**: TRL computes advantages for the entire batch at once using reshape+repeat_interleave (a single vectorized PyTorch operation on a 2D tensor), while rlox processes one group at a time via a Rust loop. The TRL approach is naturally batch-parallel on GPU.

### 2.2 Token-Level KL Divergence

**What rlox does**: `compute_token_kl(log_p: ndarray[f64], log_q: ndarray[f64]) -> f64`
- `KL = sum(exp(log_p) * (log_p - log_q))`
- Per-sequence, 1D operation

**What TRL does** (in `GRPOTrainer._compute_loss`):
```python
# TRL uses a per-token KL estimator (Schulman approximation):
# From trl/trainer/grpo_trainer.py:
per_token_kl = torch.exp(per_token_logps - per_token_ref_logps) - (per_token_logps - per_token_ref_logps) - 1
# Or the simpler form:
per_token_kl = per_token_logps - per_token_ref_logps  # log-ratio form
```

**Important**: TRL uses a different KL estimator than rlox. TRL (v0.14+) uses the Schulman (2020) KL estimator: `KL_approx = exp(log_ratio) - log_ratio - 1` where `log_ratio = log_pi - log_ref`. This is an unbiased estimator of KL divergence that is numerically more stable than the exact `exp(log_p) * (log_p - log_q)` form.

**Comparison approach**: Benchmark both KL formulas. Implement the Schulman KL estimator in rlox for a direct apples-to-apples comparison, and also compare rlox's exact KL against TRL's approximate KL.

### 2.3 DPO Loss Computation

**What rlox does**: The `DPO` class in `python/rlox/algorithms/dpo.py` computes:
```
loss = -logsigmoid(beta * ((log_pi_chosen - log_ref_chosen) - (log_pi_rejected - log_ref_rejected)))
```
This uses PyTorch for the actual loss computation (rlox provides data scaffolding via `DPOPair`).

**What TRL does** (in `DPOTrainer._compute_loss`):
```python
# Same mathematical formula, but with additional features:
# - label smoothing
# - reference-free mode
# - different loss variants (sigmoid, hinge, ipo, kto, etc.)
logits = policy_chosen_logps - policy_rejected_logps - (reference_chosen_logps - reference_rejected_logps)
losses = -F.logsigmoid(beta * logits)
```

**Comparison approach**: Since both use PyTorch for the loss math, the interesting comparison is:
1. Log-probability extraction (per-token logprob gathering from model logits) -- TRL does this inside the trainer
2. The loss computation itself given pre-computed log-probs -- should be identical performance (same PyTorch ops)

The DPO comparison is less interesting for benchmarking because rlox's DPO `compute_loss` already delegates to PyTorch. The Rust core only provides `DPOPair` data structures.

### 2.4 Batch Log-Probability Extraction

**Not directly implemented in rlox's Rust core**, but both rlox's Python trainers and TRL perform:
```python
logits = model(input_ids)  # (B, T, V) -- forward pass
log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
per_token = log_probs.gather(2, target.unsqueeze(-1)).squeeze(-1)
```
This is identical PyTorch code in both. Not worth benchmarking.

---

## 3. Benchmark Design

### 3.1 Benchmark 1: GRPO Group Advantages (Primary)

**This is the strongest comparison point.** rlox shows 34x advantage over NumPy/PyTorch; the question is how TRL's batched approach compares.

#### Configurations

| Config | N_prompts | K (group_size) | Total rewards | Realism |
|--------|-----------|----------------|---------------|---------|
| Small | 16 | 4 | 64 | Math/code tasks, small batch |
| Medium | 64 | 8 | 512 | Standard GRPO training |
| Large | 256 | 16 | 4096 | Large-scale GRPO (DeepSeek-R1 style) |
| XL | 1024 | 32 | 32768 | Distributed training aggregate |

#### What to measure

1. **rlox**: `compute_group_advantages(rewards)` called N_prompts times in a Python loop
2. **rlox (batched)**: New function `compute_batch_group_advantages(all_rewards, group_size)` processing all groups in one Rust call (should be implemented for this benchmark)
3. **TRL-style PyTorch**: Vectorized reshape + repeat_interleave on GPU tensor
4. **TRL-style PyTorch (CPU)**: Same but on CPU tensor (fair comparison with rlox CPU)
5. **NumPy**: Existing baseline

#### Fairness protocol

- All inputs are pre-allocated before timing starts
- For PyTorch GPU: include `torch.cuda.synchronize()` in timing
- For rlox: include PyO3 boundary crossing in timing (as currently done)
- Report both CPU-only and CPU-vs-GPU numbers with clear labels
- The GPU comparison should be framed as "Rust CPU vs PyTorch GPU" -- both are valid deployment targets

#### Expected outcome

On CPU: rlox should maintain its ~34x advantage over PyTorch CPU because the bottleneck is per-operation dispatch overhead, and TRL's vectorized approach still pays PyTorch's per-op cost (reshape, mean, std, repeat_interleave, subtract, divide = 6+ dispatch points).

On GPU: TRL's batched approach may be competitive or faster for large configs (1024x32) because GPU parallelism across all 32K elements amortizes kernel launch overhead. For small configs (16x4), GPU kernel launch latency (~5-10us) likely makes it slower than rlox's Rust.

### 3.2 Benchmark 2: Token-Level KL Divergence (Primary)

#### Configurations

| Config | Batch size | Seq length | Total elements | Realism |
|--------|-----------|------------|----------------|---------|
| Short | 1 | 128 | 128 | Short code completions |
| Medium | 1 | 512 | 512 | Typical instruction following |
| Long | 1 | 2048 | 2048 | Long-form reasoning |
| Very Long | 1 | 8192 | 8192 | Extended context (DeepSeek-R1) |
| Batched Short | 32 | 128 | 4096 | Batched short sequences |
| Batched Long | 32 | 2048 | 65536 | Batched long sequences |

#### What to measure

1. **rlox**: `compute_token_kl(log_p, log_q)` per sequence (existing)
2. **TRL Schulman KL (CPU)**: `exp(ratio) - ratio - 1` via PyTorch on CPU
3. **TRL Schulman KL (GPU)**: Same on GPU
4. **NumPy**: Existing baseline
5. **rlox Schulman variant**: Implement `exp(r) - r - 1` in Rust for apples-to-apples

#### Batched variant

TRL computes KL for an entire batch at once: `(B, T)` tensor operations. rlox currently processes one sequence at a time. For the batched configs, measure:
- rlox: Python loop over B sequences, each calling Rust
- TRL-style: Single PyTorch operation on `(B, T)` tensor

This tests whether rlox's per-sequence Rust call overhead negates its per-element speed advantage when batch sizes are realistic.

#### Expected outcome

Per-sequence (B=1): rlox should win at short sequences (128-512) due to lower dispatch overhead, with advantage narrowing at longer sequences (as shown in existing benchmarks: 4.7x at 128, 1.6x at 8192).

Batched (B=32): PyTorch GPU should win at large total element counts because the GPU kernel operates on the full `(32, 2048)` tensor in one launch. rlox would need a batched Rust implementation to compete.

### 3.3 Benchmark 3: DPO Loss Components (Secondary)

Less interesting because rlox's DPO delegates to PyTorch for loss math. Include for completeness but de-emphasize.

#### What to measure

Given pre-computed `(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps)`:
- Compute `logits = (pc - pr) - (rc - rr)` and `loss = -logsigmoid(beta * logits).mean()`
- This is 100% PyTorch in both cases; expect identical performance
- The value of the benchmark is confirming that both produce numerically identical results

### 3.4 Benchmark 4: End-to-End Overhead Comparison (Secondary)

**Not a direct speed comparison**, but a decomposition of where time is spent in a GRPO training step.

#### Method

Profile one GRPO training step in both rlox and TRL, breaking down wall-clock into:
1. Generation (model.generate) -- should be identical (same PyTorch)
2. Reward computation -- should be identical (same function)
3. Advantage computation -- rlox Rust vs TRL PyTorch
4. KL computation -- rlox Rust vs TRL PyTorch
5. Loss computation -- both PyTorch
6. Backward pass -- both PyTorch
7. Python overhead / data marshalling

This is an **instrumented profile**, not a competitive benchmark. The goal is to quantify what fraction of wall-clock rlox can accelerate (advantages + KL) versus what fraction is dominated by model forward/backward passes.

#### Expected outcome

For a realistic LLM (e.g., Llama-3.1-8B), the model forward/backward pass dominates (>90% of wall-clock). Advantages + KL computation is <1% of total time. This means rlox's 34x speedup on advantages translates to <0.5% wall-clock improvement in a full training step.

This is an important result to report honestly.

---

## 4. Fairness Framework

### 4.1 What is a fair comparison

| Comparison | Fair? | Why |
|-----------|-------|-----|
| rlox CPU vs TRL CPU, same operation, same data | Yes | Identical task, identical hardware |
| rlox CPU vs TRL GPU, same operation | Partially | Different hardware; label clearly as "CPU vs GPU" |
| rlox full GRPO step vs TRL full GRPO step | No | TRL includes model forward/backward, tokenization, logging; rlox only does primitives |
| rlox advantage compute vs TRL advantage compute in isolation | Yes | Same mathematical operation, isolated from trainer |

### 4.2 Data format considerations

- **Type**: rlox uses f64; TRL/PyTorch typically uses f32 (or bf16/f16 for mixed precision). f64 is 2x the memory bandwidth. For fairness, implement f32 variants in rlox or cast TRL inputs to f64.
- **Location**: rlox data lives on CPU as numpy arrays. TRL data lives on GPU as CUDA tensors. Converting between them adds latency that is part of the real-world cost if rlox were to be integrated into a TRL-like pipeline.
- **Recommendation**: Benchmark f32 on CPU for both. Separately benchmark f32 on GPU for TRL. Report both with clear labels.

### 4.3 What NOT to claim

- Do not claim "rlox is Nx faster than TRL" as a headline -- the comparison is between isolated numerical primitives, not full training pipelines
- Do not compare rlox CPU primitives against TRL's full trainer wall-clock
- Do not ignore the f64/f32 type difference in reported numbers
- Do not omit GPU results when TRL naturally runs on GPU

### 4.4 What IS meaningful to claim

- "rlox computes GRPO group advantages Nx faster than the equivalent PyTorch operations used by TRL, when both run on CPU with the same data types"
- "For workloads dominated by many small group normalizations (the GRPO pattern), Rust eliminates Python dispatch overhead"
- "In a full LLM training step, advantage + KL computation represents Y% of wall-clock, so rlox's speedup translates to Z% end-to-end improvement"

---

## 5. TRL Internals: How to Extract Comparable Code

### 5.1 Installing TRL

```bash
pip install trl>=0.16
```

TRL depends on: `transformers`, `accelerate`, `datasets`, `peft` (optional). For benchmarking isolated operations, only `torch` is needed -- we replicate TRL's computation logic without importing the full trainer.

### 5.2 GRPO Advantages -- Extracting TRL's Logic

From `trl/trainer/grpo_trainer.py` (`GRPOTrainer._compute_rewards_and_advantages`):

```python
def trl_style_grpo_advantages(rewards: torch.Tensor, num_generations: int) -> torch.Tensor:
    """Replicate TRL's GRPO advantage computation.

    Args:
        rewards: (N_prompts * num_generations,) flat tensor of rewards
        num_generations: K completions per prompt

    Returns:
        advantages: same shape as rewards, z-score normalized within each group
    """
    # Reshape to (N_prompts, K)
    grouped = rewards.reshape(-1, num_generations)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, keepdim=True)
    # Normalize
    advantages = (grouped - mean) / (std + 1e-8)
    return advantages.reshape(-1)
```

This is the exact code to benchmark against rlox.

### 5.3 Token KL -- Extracting TRL's Logic

From `trl/trainer/grpo_trainer.py` (`GRPOTrainer._compute_loss`):

```python
def trl_style_token_kl(per_token_logps: torch.Tensor, per_token_ref_logps: torch.Tensor) -> torch.Tensor:
    """Replicate TRL's per-token KL computation (Schulman estimator).

    Args:
        per_token_logps: (B, T) policy log-probs
        per_token_ref_logps: (B, T) reference log-probs

    Returns:
        per_token_kl: (B, T) per-token KL estimates
    """
    log_ratio = per_token_logps - per_token_ref_logps
    per_token_kl = torch.exp(log_ratio) - log_ratio - 1
    return per_token_kl
```

Note: TRL v0.14+ switched from `exp(log_p) * (log_p - log_q)` to the Schulman estimator. The Schulman estimator is cheaper (one exp, one sub, one sub, one const) vs the exact KL (one exp, one sub, one mul).

### 5.4 DPO Loss -- Extracting TRL's Logic

From `trl/trainer/dpo_trainer.py` (`DPOTrainer._compute_loss`):

```python
def trl_style_dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """Replicate TRL's DPO loss (sigmoid variant)."""
    logits = (policy_chosen_logps - policy_rejected_logps) - \
             (reference_chosen_logps - reference_rejected_logps)
    losses = -F.logsigmoid(beta * logits)
    return losses.mean()
```

---

## 6. New Benchmarks to Implement

### 6.1 `benchmarks/bench_trl_comparison.py`

Main benchmark script. Structure mirrors `bench_llm_ops.py`.

```
bench_trl_comparison.py
  |
  +-- Section 1: GRPO Advantages
  |     +-- bench_rlox_grpo(n_prompts, k)          # existing
  |     +-- bench_rlox_grpo_batched(n_prompts, k)   # NEW: single Rust call for all groups
  |     +-- bench_trl_grpo_cpu(n_prompts, k)         # NEW: TRL-style PyTorch on CPU
  |     +-- bench_trl_grpo_gpu(n_prompts, k)         # NEW: TRL-style PyTorch on GPU
  |     +-- bench_numpy_grpo(n_prompts, k)           # existing
  |
  +-- Section 2: Token KL Divergence
  |     +-- bench_rlox_kl(batch, seq_len)             # existing (per-sequence loop)
  |     +-- bench_rlox_kl_schulman(batch, seq_len)    # NEW: Schulman estimator in Rust
  |     +-- bench_trl_kl_cpu(batch, seq_len)           # NEW: TRL-style on CPU
  |     +-- bench_trl_kl_gpu(batch, seq_len)           # NEW: TRL-style on GPU
  |     +-- bench_numpy_kl(batch, seq_len)             # existing
  |
  +-- Section 3: DPO Loss (secondary)
  |     +-- bench_dpo_loss_pytorch(batch_size)         # identical in both, sanity check
  |
  +-- Section 4: Numerical Correctness
        +-- verify_grpo_equivalence()                   # rlox vs TRL produce same values
        +-- verify_kl_values()                          # exact KL vs Schulman estimator comparison
```

### 6.2 Rust Changes Needed

1. **`compute_batch_group_advantages(rewards: &[f64], group_size: usize) -> Vec<f64>`** in `crates/rlox-core/src/llm/ops.rs` -- process all groups in a single Rust call, avoiding N_prompts PyO3 boundary crossings.

2. **`compute_token_kl_schulman(log_p: &[f64], log_q: &[f64]) -> f64`** in `crates/rlox-core/src/llm/ops.rs` -- the Schulman KL estimator `sum(exp(log_p - log_q) - (log_p - log_q) - 1)` for apples-to-apples comparison.

3. **f32 variants** of both functions (optional but recommended for fair GPU comparison).

4. PyO3 bindings for the new functions in `crates/rlox-python/src/llm.rs`.

### 6.3 `benchmarks/bench_trl_profile.py` (Optional)

Instrumented profiling of a full GRPO training step in both frameworks using a small model (GPT-2 124M). Not a speed comparison but a time-decomposition analysis.

```
bench_trl_profile.py
  |
  +-- profile_rlox_grpo_step(model, prompts)   # rlox GRPO trainer
  +-- profile_trl_grpo_step(model, prompts)     # TRL GRPOTrainer
  +-- breakdown table: generation, reward, advantages, KL, loss, backward
```

---

## 7. Benchmark Scenarios -- Realistic Configurations

### 7.1 Rationale for Configurations

Real GRPO training (DeepSeek-R1, Qwen2.5) uses:
- 8-64 prompts per batch (limited by GPU memory for generation)
- 4-32 completions per prompt (K = num_generations)
- 128-8192 tokens per completion
- bf16 or f32 precision

Our microbenchmark configurations should span this range.

### 7.2 GRPO Advantage Configs

| Name | N_prompts | K | Total | Notes |
|------|-----------|---|-------|-------|
| `grpo_16x4` | 16 | 4 | 64 | Minimal, high overhead sensitivity |
| `grpo_64x8` | 64 | 8 | 512 | Standard GRPO batch |
| `grpo_256x16` | 256 | 16 | 4096 | Large batch (multi-GPU aggregate) |
| `grpo_1024x32` | 1024 | 32 | 32768 | Stress test |

### 7.3 Token KL Configs

| Name | Batch | Seq Len | Total Elements | Notes |
|------|-------|---------|----------------|-------|
| `kl_1x128` | 1 | 128 | 128 | Single short sequence |
| `kl_1x2048` | 1 | 2048 | 2048 | Single long sequence |
| `kl_1x8192` | 1 | 8192 | 8192 | Single very long sequence |
| `kl_32x128` | 32 | 128 | 4096 | Batched short |
| `kl_32x512` | 32 | 512 | 16384 | Batched medium |
| `kl_32x2048` | 32 | 2048 | 65536 | Batched long |
| `kl_128x2048` | 128 | 2048 | 262144 | Large batch, long context |

---

## 8. Statistical Protocol

Follow the same protocol as existing benchmarks (from `conftest.py`):

- **Warmup**: 10 iterations (discarded)
- **Measured repetitions**: 50 for GRPO, 100 for KL
- **Report**: Median, p25, p75, IQR
- **Comparison**: Bootstrap 95% CI on speedup ratio (10K resamples)
- **Significance**: CI lower bound > 1.0

Additional for GPU benchmarks:
- **CUDA sync**: `torch.cuda.synchronize()` before start and after end of each timed region
- **GPU warmup**: Extra warmup iterations for GPU JIT compilation (CUDA kernel caching)

---

## 9. Expected Results and Honest Reporting

### 9.1 Predicted Results

| Benchmark | rlox vs TRL-CPU | rlox vs TRL-GPU | Confidence |
|-----------|----------------|-----------------|------------|
| GRPO 16x4 | 30-40x faster | 5-15x faster (GPU kernel launch overhead) | High |
| GRPO 64x8 | 30-40x faster | 2-5x faster | High |
| GRPO 256x16 | 25-35x faster | 0.5-2x (GPU catches up) | Medium |
| GRPO 1024x32 | 20-30x faster | 0.3-1x (GPU may win) | Medium |
| KL 1x128 | 4-5x faster | 3-8x faster (kernel launch cost) | High |
| KL 1x2048 | 1.5-2x faster | 0.5-1x | Medium |
| KL 32x2048 | 1-2x faster (loop overhead) | 0.2-0.5x (GPU wins) | Medium |
| KL 128x2048 | 0.5-1.5x | 0.1-0.3x (GPU wins decisively) | Medium |

### 9.2 What Makes This Benchmark Meaningful

1. **rlox's niche**: Many small operations where Python dispatch overhead dominates. GRPO with K=4-16 is the sweet spot (34x existing benchmark shows this).

2. **GPU crossover point**: Identifying where GPU parallelism surpasses Rust CPU is valuable -- it tells users when to use rlox (small K, CPU-only setups, inference servers without GPU) vs when to stay with TRL (GPU-rich training setups).

3. **Integration cost**: If rlox requires data to move from GPU to CPU numpy and back, the transfer cost may negate the compute speedup. Quantifying this is critical.

### 9.3 How to Report Results

The benchmark report (`docs/benchmark/trl-comparison.md`) should include:

1. **Executive summary**: "rlox's Rust primitives are N-Mx faster than TRL's PyTorch equivalents on CPU for the specific operations of GRPO advantage computation and token-level KL divergence. However, in a full LLM training step, these operations represent <Y% of wall-clock, making the end-to-end impact Z%."

2. **Per-operation tables**: With CI, clearly labeled CPU vs GPU

3. **Time decomposition**: Pie chart showing what fraction of a GRPO training step each component takes (generation >> backward > forward > advantages + KL)

4. **Crossover analysis**: At what batch size / sequence length does GPU PyTorch overtake Rust CPU?

5. **Integration overhead**: Cost of `numpy <-> torch.Tensor` conversion if rlox primitives were used inside a TRL-like pipeline

---

## 10. Implementation Plan

### Phase 1: Rust Extensions (0.5 days)

1. Add `compute_batch_group_advantages(rewards, group_size)` to `ops.rs`
2. Add `compute_token_kl_schulman(log_p, log_q)` to `ops.rs`
3. Add PyO3 bindings
4. Unit tests for new functions

### Phase 2: Benchmark Script (1 day)

1. Implement `benchmarks/bench_trl_comparison.py`
2. TRL-style PyTorch functions (standalone, no TRL dependency)
3. Numerical correctness verification
4. All configs from Section 7

### Phase 3: Run and Analyze (0.5 days)

1. Run on target hardware (Apple M4 for consistency with existing benchmarks)
2. Optionally run on GPU machine if available
3. Generate `docs/benchmark/trl-comparison.md`

### Phase 4: Profile (Optional, 1 day)

1. Implement `bench_trl_profile.py` with GPT-2 124M
2. Time decomposition of full GRPO step
3. Quantify end-to-end impact

### Total estimated effort: 2-3 days

---

## 11. Dependencies

### Required
- `torch` (already installed)
- `numpy` (already installed)
- `rlox` (already installed)

### Optional (for full TRL comparison)
- `trl>=0.16` (for verifying our extracted code matches actual TRL behavior)
- `transformers` (TRL dependency)
- GPU with CUDA (for GPU benchmarks)

### Not required
- No LLM model weights needed for primitive benchmarks
- No inference server needed
- No datasets needed

---

## 12. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| TRL changes its KL/advantage implementation between versions | Results become outdated | Pin TRL version, cite specific commit/version |
| f64 vs f32 makes comparison unfair | Misleading speedup numbers | Implement f32 Rust variants; report both |
| GPU warmup/JIT makes GPU numbers unreliable | High variance in GPU timings | Extended warmup (100 iterations), more repetitions |
| rlox advantage disappears for batched operations | Weaker story | This is a valid finding; report honestly as "crossover analysis" |
| PyO3 boundary crossing dominates for batched rlox | rlox looks slow for batched | Implement batched Rust function to amortize crossing cost |
| Apple M4 has no NVIDIA GPU | Cannot benchmark TRL GPU path | Run CPU-only on M4; separately benchmark GPU on cloud instance if available |

---

## 13. References

[1] Z. Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models," arXiv:2402.03300, 2024. (GRPO algorithm)

[2] R. Rafailov et al., "Direct Preference Optimization: Your Language Model Is Secretly a Reward Model," NeurIPS, 2023. (DPO algorithm)

[3] J. Schulman, "Approximating KL Divergence," blog post, 2020. (KL estimator used by TRL)

[4] L. von Werra et al., "TRL: Transformer Reinforcement Learning," GitHub repository, https://github.com/huggingface/trl, 2024.

[5] Q. Gallouedec et al., "Open RL Benchmark," NeurIPS Datasets and Benchmarks, 2024. (TRL benchmark methodology)
