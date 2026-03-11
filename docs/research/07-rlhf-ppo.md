# RLHF-PPO — Reinforcement Learning from Human Feedback

> Christiano et al., "Deep Reinforcement Learning from Human Preferences," NeurIPS, 2017.
> Ouyang et al., "Training language models to follow instructions with human feedback (InstructGPT)," NeurIPS, 2022.
> Stiennon et al., "Learning to summarize from human feedback," NeurIPS, 2020.

## Key Idea

RLHF-PPO is a three-stage pipeline: (1) Supervised fine-tuning (SFT) on demonstration data, (2) Training a reward model (RM) on human preference comparisons, (3) Optimizing the SFT model using PPO with the learned RM as reward, subject to a KL constraint against the SFT model. This was the method that produced ChatGPT, InstructGPT, and Claude's early versions.

## Mathematical Formulation

**Stage 1 — SFT:**

```
L_SFT(θ) = -E_{(x,y)~D_demo} [ Σ_t log π_θ(y_t | x, y_{<t}) ]
```

**Stage 2 — Reward Model (Bradley-Terry):**

```
L_RM(φ) = -E_{(x, y_w, y_l)~D_pref} [ log σ(r_φ(x, y_w) - r_φ(x, y_l)) ]
```

**Stage 3 — PPO with KL penalty:**

```
R_total(x, y) = r_φ(x, y) - β · D_KL(π_θ(·|x) || π_ref(·|x))

Objective: max_θ E_{x~D, y~π_θ} [ R_total(x, y) ]
```

Optimized using PPO with:
- Policy = the language model `π_θ`
- Critic = a value model `V_ψ(x, y_{<t})` (often initialized from RM)
- Actions = tokens, states = (prompt, generated tokens so far)
- Reward given at end of sequence

## Properties

- On-policy (PPO), model-free
- Requires **4 models simultaneously**: policy, reference, reward model, value model
- Three-stage pipeline (SFT → RM → RL)

## Key Hyperparameters

| Parameter | Typical Value | Notes |
|-----------|---------------|-------|
| PPO `ε` | 0.1–0.2 | Clipping |
| `β` (KL penalty) | 0.01–0.2 | Adaptive in some impls |
| PPO epochs | 1–4 | Per batch of generations |
| RM size | Same as policy or smaller | |
| Batch size | 256–1024 prompts | |
| Temperature | 0.7–1.0 | For rollouts |
| Learning rate | 1e-6 to 5e-6 | |
| Reward normalization | Yes | Running mean/std |

## Complexity

- **Memory:** 4 full models. For 7B model: ~56GB in fp16, requiring multi-GPU
- **Time:** Generation (~70%) + reward scoring (~5%) + PPO update (~25%)
- Far more complex than DPO or GRPO in engineering
- Scales poorly — each component must scale with model size

## Primary Use Cases

- InstructGPT / ChatGPT (OpenAI) — the original application
- Claude (Anthropic) — with Constitutional AI modifications
- Llama 2 (Meta) — RLHF with rejection sampling
- General-purpose LLM alignment for helpfulness and harmlessness

## Known Limitations

1. **Extreme engineering complexity** — 4 models, distributed training, generation loop
2. **Reward model bottleneck** — can be hacked, gamed, or miscalibrated
3. **KL penalty tuning** is delicate
4. **Proxy reward problem** — RM can't capture all quality aspects
5. **Very expensive** — 4× model memory overhead
6. **Mode collapse** — policy converges to narrow high-reward outputs
7. **Error amplification** — RM errors amplified by PPO optimization
8. **Being displaced** by GRPO (reasoning) and DPO (alignment)

## Major Variants

| Variant | Reference | Key Change |
|---------|-----------|------------|
| Constitutional AI | Bai et al., Anthropic 2022 | AI-generated feedback |
| ReST/RAFT | Dong et al., TMLR 2023 | Best-of-N filtering + SFT |
| RLHF + PRM | — | Step-level rewards for reasoning |
| RM ensembles | — | Robustness to reward hacking |
| Iterative RLHF | — | Continuous preference collection |

## Relationship to Other Algorithms

- **PPO** is the RL algorithm used in Stage 3
- **DPO** was invented specifically to simplify this pipeline (eliminates Stages 2 and 3)
- **GRPO** simplifies Stage 3 by removing the critic
- As of 2026: RLHF-PPO is the "legacy" approach; GRPO and DPO increasingly preferred

## Industry Deployment

- **OpenAI:** InstructGPT, GPT-4
- **Anthropic:** Claude (with Constitutional AI)
- **Meta:** Llama 2 (combined with rejection sampling)
- **Google DeepMind:** Gemini (partial RLHF)
- Still used at frontier labs for non-reasoning alignment
- **Open-source:** TRL, OpenRLHF, DeepSpeed-Chat
