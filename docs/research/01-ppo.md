# PPO — Proximal Policy Optimization

> Schulman et al., "Proximal Policy Optimization Algorithms," arXiv:1707.06347, 2017.

## Key Idea

PPO constrains the policy update to a trust region by clipping the probability ratio between the new and old policy, preventing destructively large updates. It achieves performance comparable to TRPO while being far simpler to implement, requiring only first-order optimization. This makes it the de facto default for both classic RL and (until recently) LLM post-training.

## Mathematical Formulation

**Notation:**
- `π_θ` — current policy parameterized by θ
- `π_θ_old` — policy before the update
- `r_t(θ) = π_θ(a_t | s_t) / π_θ_old(a_t | s_t)` — probability ratio
- `Â_t` — advantage estimate (typically GAE-λ)
- `ε` — clipping parameter

**Clipped surrogate objective:**

```
L^CLIP(θ) = E_t [ min( r_t(θ) · Â_t,  clip(r_t(θ), 1-ε, 1+ε) · Â_t ) ]
```

**Full loss with value function and entropy bonus:**

```
L(θ) = E_t [ L^CLIP(θ) - c₁ · L^VF(θ) + c₂ · S[π_θ](s_t) ]

where:
  L^VF(θ) = (V_θ(s_t) - V_t^target)²
  S[π](s) = -Σ_a π(a|s) log π(a|s)
```

**Value function target (GAE):**

```
Â_t^GAE(γ,λ) = Σ_{l=0}^{T-t} (γλ)^l · δ_{t+l}
δ_t = r_t + γ · V(s_{t+1}) - V(s_t)
V_t^target = Â_t^GAE + V(s_t)
```

## Properties

- On-policy, model-free
- Actor-critic architecture (shared or separate backbones)
- Supports both continuous and discrete action spaces

## Key Hyperparameters

| Parameter | Typical Value | Notes |
|-----------|---------------|-------|
| `ε` (clip) | 0.1–0.2 | 0.2 most common for classic RL |
| `γ` | 0.99 | Discount factor |
| `λ` (GAE) | 0.95 | Bias-variance tradeoff |
| `c₁` (VF coef) | 0.5 | Value loss weight |
| `c₂` (entropy coef) | 0.01 | Entropy bonus weight |
| Minibatch size | 64–4096 | Larger for LLM training |
| Epochs per rollout | 3–10 | Passes over collected data |
| Learning rate | 3e-4 (Adam) | Often linearly decayed |

## Complexity

- **Time per update:** O(B × K × C_forward) — B=batch, K=epochs, C=forward/backward cost
- **Memory:** O(T × N × d_obs) for rollout buffer + model parameters
- **Sample efficiency:** Poor (on-policy — data discarded after each update)

## Primary Use Cases

- Continuous control: MuJoCo locomotion (Humanoid, Ant, HalfCheetah)
- Game playing: OpenAI Five (Dota 2), hide-and-seek emergent behavior
- Robotics: Sim-to-real transfer (OpenAI Rubik's cube)
- LLM alignment: RLHF pipeline (InstructGPT, ChatGPT, Llama 2)
- LLM reasoning: Used in early DeepSeek-R1 experiments before GRPO

## Known Limitations

1. **Sample inefficient** — on-policy data used for only a few epochs then discarded
2. **Sensitive to hyperparameters** — especially ε, learning rate, and number of epochs
3. **Implementation details matter** — "37 Implementation Details" paper identified 13+ tricks PPO relies on
4. **Reward hacking** — in LLM RLHF, PPO can overoptimize the reward model
5. **Critic overhead** — for LLMs, requires a second model of comparable size
6. **Advantage estimation degrades** in very long episodes

## Major Variants

| Variant | Reference | Key Change |
|---------|-----------|------------|
| TRPO | Schulman et al., ICML 2015 | KL constraint (PPO predecessor) |
| PPO-Penalty | — | Adaptive KL penalty instead of clipping |
| APPO | Espeholt et al., ICML 2018 | Asynchronous distributed training |
| PPG | Cobbe et al., ICML 2021 | Separate phases for policy and value |
| Dual-clip PPO | — | Clips from both sides for negative advantages |

## Relationship to Other Algorithms

- Direct ancestor of **RLHF-PPO** — same algorithm, different reward source
- **GRPO** was developed specifically to replace PPO for LLM training by removing the critic
- **DPO** bypasses PPO entirely by reformulating the RLHF objective
- **MAPPO** is the multi-agent extension
- Shares GAE advantage estimation with essentially all modern policy gradient methods

## Industry Deployment

- **OpenAI:** Dota 2 (OpenAI Five), ChatGPT (RLHF-PPO), robotics
- **DeepMind:** Some game-playing agents
- **Meta:** Llama 2 RLHF training
- **Robotics:** Numerous companies for sim-to-real transfer
- **Frameworks:** Default in SB3, CleanRL, RLlib
