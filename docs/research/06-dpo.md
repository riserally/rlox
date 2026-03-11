# DPO — Direct Preference Optimization

> Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model," NeurIPS, 2023.

## Key Idea

DPO shows that the optimal policy under the KL-constrained RLHF objective can be expressed in closed form as a function of the reward. Inverting this relationship, DPO reparameterizes the reward in terms of the policy itself, eliminating explicit reward model training or RL optimization. The result is a simple supervised-learning-style loss on preference pairs (chosen vs. rejected). This simplifies the RLHF pipeline from a multi-stage process to a single training step.

## Mathematical Formulation

**Starting from the RLHF objective:**

```
max_π E_{x~D, y~π(·|x)} [ r(x,y) ] - β · D_KL(π || π_ref)
```

**Optimal policy form:**

```
π*(y|x) = (1/Z(x)) · π_ref(y|x) · exp(r(x,y) / β)
```

**Reparameterized reward:**

```
r(x,y) = β · log(π_θ(y|x) / π_ref(y|x)) + β · log Z(x)
```

**DPO loss (substituting into Bradley-Terry model):**

```
L_DPO(θ) = -E_{(x, y_w, y_l)~D} [ log σ(
  β · log(π_θ(y_w|x) / π_ref(y_w|x))
  - β · log(π_θ(y_l|x) / π_ref(y_l|x))
) ]
```

Where `y_w` = preferred, `y_l` = dispreferred, `σ` = sigmoid.

**Implicit reward:**

```
r_θ(x,y) = β · log(π_θ(y|x) / π_ref(y|x))
```

## Properties

- Offline (trains on fixed preference dataset)
- No RL loop, no reward model, no value function
- Supervised learning formulation
- Requires paired preference data (y_w, y_l per prompt)

## Key Hyperparameters

| Parameter | Typical Value | Notes |
|-----------|---------------|-------|
| `β` | 0.1–0.5 | KL regularization strength |
| Learning rate | 1e-6 to 5e-7 | Very low to prevent forgetting |
| Batch size | 32–128 | Preference pairs |
| Epochs | 1–3 | Overfitting risk with more |
| Max seq length | 512–2048 | Task dependent |
| Label smoothing | 0.0–0.1 | Optional regularization |

## Complexity

- **Time:** O(B × L × C_forward) — two forward passes per sample (chosen + rejected)
- **Memory:** Policy + reference model (frozen). No critic, no reward model
- Reference model can be offloaded or computed via LoRA difference
- Much cheaper per iteration than PPO or GRPO (no generation)

## Primary Use Cases

- LLM alignment from human preferences (the "simpler RLHF")
- Instruction following (Zephyr, Tulu)
- Safety and harmlessness training
- Image generation (diffusion model alignment)

## Known Limitations

1. **Offline only** (base form) — cannot explore beyond dataset distribution
2. **Distribution shift** — implicit reward unreliable when policy drifts far
3. **Length exploitation** — can learn to prefer longer outputs
4. **Cannot use verifiable rewards** directly (unlike GRPO)
5. **Quality ceiling** imposed by preference data
6. **Preference data expensive** to collect at scale
7. **Bradley-Terry assumption** may not hold for complex preferences
8. **Decreases entropy/diversity** more than PPO-based methods

## Major Variants

| Variant | Reference | Key Change |
|---------|-----------|------------|
| IPO | Azar et al., AISTATS 2024 | Squared loss, robust to noise |
| KTO | Ethayarajh et al., ICML 2024 | Binary feedback only (no pairs) |
| ORPO | Hong et al., EMNLP 2024 | SFT + alignment in one loss, no ref model |
| SimPO | Meng et al., NeurIPS 2024 | Avg log-prob reward, no ref model |
| Online DPO | Guo et al., 2024 | Generate new pairs with current policy |
| RSO | Liu et al., ICLR 2024 | Rejection sampling for on-policy data |
| SPPO | Wu et al., ICML 2024 | Self-play preference optimization |

## Relationship to Other Algorithms

- Derived from same objective as **RLHF-PPO** — different optimization path, same goal
- Competes with **GRPO**: DPO=offline+preferences; GRPO=online+rewards
- Can be combined: SFT → DPO → GRPO is an emerging pipeline
- Online DPO variants blur the line between DPO and GRPO

## Industry Deployment

- **HuggingFace:** Zephyr models, TRL library
- **Meta:** Llama fine-tuning recipes
- **AI2:** Tulu models
- **Anthropic:** Explored alongside RLHF
- Default choice for alignment when paired preference data is available
- As of 2026, increasingly combined with online methods
