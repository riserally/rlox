# LLM Post-Training with rlox

rlox provides Rust-accelerated primitives for LLM post-training (RLHF, DPO, GRPO) alongside pure-Python algorithm implementations. The Rust primitives handle the compute-heavy operations (KL divergence, GRPO advantages, sequence packing) while Python handles the model forward/backward passes.

## Overview

| Algorithm | Class | Use Case |
|-----------|-------|----------|
| **DPO** | `rlox.algorithms.dpo.DPO` | Offline preference optimization with static dataset |
| **GRPO** | `rlox.algorithms.grpo.GRPO` | Group-relative policy optimization (DeepSeek-R1 style) |
| **OnlineDPO** | `rlox.algorithms.online_dpo.OnlineDPO` | DPO with online generation + preference oracle |
| **BestOfN** | `rlox.algorithms.best_of_n.BestOfN` | Inference-time rejection sampling |

## Rust-Accelerated Primitives

These are the building blocks — use them directly for custom training loops:

```python
import rlox
import numpy as np

# --- KL Divergence (2-27x faster than TRL/NumPy) ---

# Single sequence (f64)
kl = rlox.compute_token_kl(log_probs_policy, log_probs_ref)

# Schulman estimator (TRL-compatible)
kl = rlox.compute_token_kl_schulman(log_probs_policy, log_probs_ref)

# Batched (single FFI call for all sequences)
# log_probs are flat arrays of shape (batch * seq_len,)
kl_per_seq = rlox.compute_batch_token_kl(
    log_p_flat, log_q_flat, seq_len=512
)

# f32 variant (2x faster, matches PyTorch native precision)
kl_per_seq = rlox.compute_batch_token_kl_schulman_f32(
    log_p_flat.astype(np.float32),
    log_q_flat.astype(np.float32),
    seq_len=512,
)

# --- GRPO Group Advantages (27x faster than TRL at small groups) ---

# Single group
advantages = rlox.compute_group_advantages(rewards)  # (G,) -> (G,)

# Batched: all prompts at once
# rewards shape: (n_prompts * group_size,)
advantages = rlox.compute_batch_group_advantages(rewards, group_size=8)

# --- Sequence Packing ---
# First-fit-decreasing bin packing for variable-length sequences
from rlox import pack_sequences
bins = pack_sequences(sequence_lengths, max_length=2048)
```

## DPO — Direct Preference Optimization

Train a language model to prefer chosen completions over rejected ones, using a reference model for KL regularization.

```python
import torch
import torch.nn as nn
from rlox.algorithms.dpo import DPO

# Your language model and a frozen reference copy
model = MyLanguageModel(vocab_size=32000, hidden=256)
ref_model = MyLanguageModel(vocab_size=32000, hidden=256)
ref_model.load_state_dict(model.state_dict())
for p in ref_model.parameters():
    p.requires_grad = False

# Create DPO trainer
dpo = DPO(
    model=model,
    ref_model=ref_model,
    beta=0.1,              # KL penalty strength
    learning_rate=1e-4,
)

# Train on preference pairs
# prompt, chosen, rejected are (1, seq_len) token ID tensors
for prompt, chosen, rejected in dataset:
    metrics = dpo.train_step(prompt, chosen, rejected)
    print(f"Loss: {metrics['loss']:.4f}")

# Save checkpoint
dpo.save("dpo_checkpoint.pt")
```

### Key Features
- Gradient clipping (`max_grad_norm=1.0` by default)
- Callback support for monitoring
- Logger integration (W&B, TensorBoard, Console)

## GRPO — Group Relative Policy Optimization

GRPO generates multiple completions per prompt, scores them with a reward function, and computes advantages relative to the group (DeepSeek-R1 approach).

```python
from rlox.algorithms.grpo import GRPO

def reward_fn(completions: list[torch.Tensor]) -> list[float]:
    """Score completions. Higher is better."""
    return [score_completion(c) for c in completions]

grpo = GRPO(
    model=model,
    ref_model=ref_model,
    reward_fn=reward_fn,
    group_size=4,          # completions per prompt
    kl_coef=0.1,           # KL penalty
    learning_rate=1e-4,
    max_new_tokens=128,
)

# Train on prompts
for prompt_batch in dataloader:
    metrics = grpo.train_step(prompt_batch)
    print(f"Loss: {metrics['loss']:.4f}, Mean reward: {metrics['mean_reward']:.2f}, KL: {metrics['mean_kl']:.4f}")

grpo.save("grpo_checkpoint.pt")
```

### How GRPO Uses Rust Primitives

Internally, GRPO uses two Rust-accelerated operations:

1. **`compute_batch_group_advantages`** — normalizes rewards within each group in a single Rust call (Rayon-parallelized for large batches)
2. **`compute_batch_token_kl`** — computes per-sequence KL divergence in a single batched call instead of looping

## OnlineDPO — Online Direct Preference Optimization

Like DPO but generates completions online and queries a preference oracle:

```python
from rlox.algorithms.online_dpo import OnlineDPO

def preference_fn(pairs: list[tuple[torch.Tensor, torch.Tensor]]) -> list[int]:
    """Return 0 if first completion is preferred, 1 if second."""
    return [0 if score(a) > score(b) else 1 for a, b in pairs]

online_dpo = OnlineDPO(
    model=model,
    ref_model=ref_model,
    preference_fn=preference_fn,
    beta=0.1,
    learning_rate=1e-4,
)

# Each step generates pairs, queries preferences, and updates
for prompt_batch in dataloader:
    metrics = online_dpo.train_step(prompt_batch)
```

## BestOfN — Rejection Sampling

Generate N completions per prompt, score them, and return the best:

```python
from rlox.algorithms.best_of_n import BestOfN

bon = BestOfN(
    model=model,
    reward_fn=reward_fn,
    n=8,                   # generate 8 candidates
    max_new_tokens=128,
)

# Generate best completions (no training, inference only)
best_completions = bon.generate(prompts)  # (B, P+T)
```

## Performance Comparison vs TRL

Benchmarked on GCP (8 vCPU, CPU-only):

| Operation | rlox | TRL (PyTorch) | Speedup |
|-----------|------|---------------|---------|
| GRPO 16x4 | 1.3 us | 34.1 us | **27x** |
| GRPO 1024x32 | 115 us | 243 us | **2.1x** |
| KL 1x128 (f32) | 0.3 us | 3.1 us | **9.4x** |
| KL 32x2048 (f32) | 72 us | 194 us | **2.7x** |

## Custom Training Loop

For full control, use the Rust primitives directly:

```python
import rlox
import torch
import numpy as np

model = MyModel()
ref_model = MyModel()
ref_model.load_state_dict(model.state_dict())
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for batch in dataloader:
    prompts, completions = batch

    # Forward pass
    policy_logprobs = get_per_token_logprobs(model, completions)
    with torch.no_grad():
        ref_logprobs = get_per_token_logprobs(ref_model, completions)

    # Compute KL using Rust (batched, f32)
    p = policy_logprobs.detach().cpu().numpy().astype(np.float32).ravel()
    r = ref_logprobs.cpu().numpy().astype(np.float32).ravel()
    seq_len = policy_logprobs.shape[-1]
    kl_per_seq = rlox.compute_batch_token_kl_schulman_f32(p, r, seq_len)
    mean_kl = float(kl_per_seq.mean())

    # Compute GRPO advantages using Rust
    rewards = np.array(reward_fn(completions))
    advantages = rlox.compute_batch_group_advantages(rewards, group_size=4)

    # Your custom loss
    loss = compute_loss(policy_logprobs, advantages, kl_per_seq)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
```

## Monitoring Training

```python
from rlox.logging import ConsoleLogger, WandbLogger

# Console output every 100 steps
logger = ConsoleLogger(log_interval=100)

# Or Weights & Biases
logger = WandbLogger(project="my-llm-training")

dpo = DPO(model=model, ref_model=ref_model, logger=logger)
```

## Saving and Loading

```python
# Save
dpo.save("checkpoint.pt")

# Load
dpo_loaded = DPO.from_checkpoint("checkpoint.pt", env_id="unused")
```

## CLI

```bash
# Train DPO (requires a data loader — see examples/)
python -m rlox train --algo dpo --config dpo_config.yaml
```
