# Dreamer (v1/v2/v3) — Model-Based RL with World Models

> Hafner et al., "Dream to Control: Learning Behaviors by Latent Imagination," ICLR, 2020. (v1)
> Hafner et al., "Mastering Atari with Discrete World Models," ICLR, 2021. (v2)
> Hafner et al., "Mastering Diverse Domains through World Models," arXiv:2301.04104, 2023. (v3)

## Key Idea

Dreamer learns a world model in a compact latent space (RSSM — Recurrent State-Space Model) and trains the policy entirely by "dreaming" — imagining trajectories in the learned model rather than interacting with the real environment. DreamerV3 achieves this across diverse domains (Atari, continuous control, DMLab, Minecraft) with a single set of hyperparameters using symlog predictions and entropy-regularized actor training.

## Mathematical Formulation

**World model (RSSM) components:**

```
Sequence model:     h_t = f_φ(h_{t-1}, z_{t-1}, a_{t-1})       (deterministic recurrent)
Encoder:            z_t ~ q_φ(z_t | h_t, x_t)                   (posterior)
Dynamics prior:     ẑ_t ~ p_φ(z_t | h_t)                        (used in imagination)
Decoder:            x̂_t ~ p_φ(x_t | h_t, z_t)                   (reconstruction)
Reward predictor:   r̂_t = R_φ(h_t, z_t)
Continue predictor: ĉ_t = C_φ(h_t, z_t)                         (episode termination)
```

**World model loss:**

```
L_wm(φ) = E [ -ln p_φ(x_t|h_t,z_t) - ln p_φ(r_t|h_t,z_t) - ln p_φ(c_t|h_t,z_t)
              + β · D_KL(q_φ(z_t|h_t,x_t) || p_φ(z_t|h_t)) ]
```

**DreamerV3 symlog transform:**

```
symlog(x) = sign(x) · ln(|x| + 1)
symexp(x) = sign(x) · (exp(|x|) - 1)
```

**Actor-critic in imagination:**

```
L_actor(θ) = E_imagination [ Σ_t ( λ_t · sg(V_ψ(s_t)) - η · H[π_θ(·|s_t)] ) ]
L_critic(ψ) = E [ Σ_t (V_ψ(s_t) - sg(R_t^λ))² ]
```

## Properties

- Model-based: learns explicit world model, trains policy in imagination
- On-policy w.r.t. imagination, off-policy w.r.t. real environment
- Actor-critic in latent space

## Key Hyperparameters

| Parameter | Typical Value | Notes |
|-----------|---------------|-------|
| Imagination horizon | 15 | Dreamed trajectory steps |
| `γ` | 0.997 | Discount (DreamerV3) |
| `λ` (returns) | 0.95 | Lambda-returns in imagination |
| KL `β` | 0.5 (free nats) | World model KL weight |
| Model LR | 1e-4 | |
| Actor LR | 3e-5 | |
| Critic LR | 3e-5 | |
| Latent dim | 32 classes × 32 dims | Discrete latent (V2/V3) |
| Batch size | 16 × 64 steps | Sequences |
| Entropy `η` | 3e-4 | DreamerV3 |

## Complexity

- **Time:** World model training + imagination policy training. More compute per real step, but far fewer real steps needed
- **Memory:** RSSM + decoder + reward/continue heads + actor + critic. Relatively compact
- **Sample efficiency:** Excellent — often 10-100× fewer env interactions than model-free
- **Wall-clock:** Can be slower per env step due to model training overhead

## Primary Use Cases

- Visual control from pixels (Atari, DMControl)
- Minecraft (DreamerV3 was first to obtain diamond from scratch)
- Diverse domains with a single hyperparameter set
- Environments where interactions are expensive (robotics sim)

## Known Limitations

1. **Compounding model errors** in long imagination rollouts
2. **Harder to implement and debug** than model-free methods
3. **Not trivially parallelizable** across environments (unlike PPO)
4. **Struggles** with very complex / high-frequency dynamics
5. **Discrete latent space** (V2/V3) can be limiting
6. **Wall-clock time** can be worse than PPO despite sample efficiency

## Major Variants

| Variant | Reference | Key Change |
|---------|-----------|------------|
| PlaNet | Hafner et al., ICML 2019 | Predecessor — CEM planning in latent space |
| DreamerV1 | Hafner et al., ICLR 2020 | Adds actor-critic in imagination |
| DreamerV2 | Hafner et al., ICLR 2021 | Discrete latent, Atari mastery |
| DreamerV3 | Hafner et al., 2023 | Symlog, fixed hyperparams across domains |
| TD-MPC2 | Hansen et al., ICLR 2024 | Learned model + model-predictive control |
| IRIS | Micheli et al., ICML 2023 | Transformer-based world model |
| DIAMOND | Alonso et al., NeurIPS 2024 | Diffusion world model |

## Relationship to Other Algorithms

- Orthogonal to model-free methods (PPO, SAC, DQN) — can be combined
- Competes with DQN/Rainbow on Atari, often with much better sample efficiency
- **Decision Transformer** is another non-standard approach but doesn't learn a world model
- Connects to **MuZero** (model-based + search)

## Industry Deployment

- **DeepMind:** Research (Hafner)
- **Academic:** Widely adopted
- Gaining traction in robotics where sample efficiency matters
- Less common in production than model-free methods due to complexity
