# GRPO — Group Relative Policy Optimization

> Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models," arXiv:2402.03300, 2024.
> Guo et al., "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning," arXiv:2501.12948, 2025.

## Key Idea

GRPO eliminates the need for a separate critic (value function) model — which for LLMs would be a second multi-billion-parameter network — by estimating advantages through group-relative comparisons. For each prompt, GRPO samples a group of G completions, computes rewards for each, and normalizes advantages within the group (zero-mean, unit-variance). DeepSeek-R1 demonstrated that GRPO with rule-based rewards can elicit emergent chain-of-thought reasoning without supervised fine-tuning.

## Mathematical Formulation

For a prompt `q`, sample G completions `{o_1, ..., o_G}` from `π_θ_old`:

**Group advantage estimation:**

```
A_i = (r_i - mean({r_1,...,r_G})) / std({r_1,...,r_G})
```

**GRPO objective (per-token, with KL penalty):**

```
L_GRPO(θ) = E_{q~P, {o_i}~π_old} [
  (1/G) · Σ_i (1/|o_i|) · Σ_t
    min( r_{i,t}(θ) · A_i,  clip(r_{i,t}(θ), 1-ε, 1+ε) · A_i )
    - β · D_KL(π_θ || π_ref)
]

where r_{i,t}(θ) = π_θ(o_{i,t} | q, o_{i,<t}) / π_old(o_{i,t} | q, o_{i,<t})
```

**KL divergence (per-token, reverse approximation):**

```
D_KL = π_ref(o_{i,t}) / π_θ(o_{i,t}) - log( π_ref(o_{i,t}) / π_θ(o_{i,t}) ) - 1
```

## Properties

- On-policy (samples from current policy each iteration)
- Model-free (no environment model)
- **No critic network** — this is the key innovation
- Policy gradient with group-relative baseline

## Key Hyperparameters

| Parameter | Typical Value | Notes |
|-----------|---------------|-------|
| Group size `G` | 8–64 | Completions per prompt |
| `ε` (clip) | 0.1–0.2 | PPO-style clipping |
| `β` (KL penalty) | 0.01–0.04 | Prevents reference drift |
| Learning rate | 1e-6 to 5e-6 | |
| Batch size (prompts) | 512–1024 | |
| Max seq length | 4096–32768 | Longer for reasoning |
| Temperature | 0.7–1.0 | For group sampling |

## Complexity

- **Time:** O(G × B × L × C_forward) for generation + O(B × K × C_backward) for updates
- **Generation is bottleneck:** ~70-80% of compute
- **Memory:** Policy + reference only (no critic). Saves ~14GB for 7B model vs PPO-RLHF
- ~33% memory reduction vs PPO-RLHF

## Primary Use Cases

- LLM reasoning improvement (math, code, logic)
- LLM alignment with rule-based / verifiable rewards
- Any setting where reward can be computed programmatically
- DeepSeek-R1, DeepSeek-R1-Zero, Qwen-2.5 series

## Known Limitations

1. **Group size variance** — small G gives noisy advantage estimates
2. **Expensive generation** — G completions per prompt for large models
3. **Pathological normalization** — can assign positive advantage to all-bad groups
4. **On-policy** — cannot reuse data from previous iterations
5. **KL penalty tuning** is delicate
6. **Coarse credit assignment** — same advantage for all tokens in a completion

## Major Variants

| Variant | Reference | Key Change |
|---------|-----------|------------|
| GRPO + ORM | — | Learned outcome reward model |
| GRPO + PRM | Lightman et al., 2024 | Step-level rewards for math |
| DAPO | Bytedance, 2025 | Dynamic sampling, asymmetric clipping, no KL |
| Dr. GRPO | Liu et al., 2025 | Removes length bias |
| RLOO | Ahmadian et al., ACL 2024 | Leave-one-out baseline |
| Online DPO + GRPO | — | Hybrid sampling approaches |

## Relationship to Other Algorithms

- Simplification of **PPO** for LLM setting — removes the critic
- Alternative to **DPO** — GRPO uses explicit rewards; DPO uses preference pairs
- Builds on **REINFORCE** with group-mean baseline for variance reduction
- **RLHF-PPO** is the predecessor it aims to replace

## Industry Deployment

- **DeepSeek:** R1, R1-Zero, DeepSeekMath, DeepSeek-Coder-V2
- **Alibaba:** Qwen team
- **Open-source:** OpenRLHF, TRL, LLaMA-Factory
- **The algorithm of choice** for "reasoning via RL" paradigm as of early 2026
