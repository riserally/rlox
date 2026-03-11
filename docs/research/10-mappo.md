# MAPPO — Multi-Agent PPO

> Yu et al., "The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games," NeurIPS, 2022.

## Key Idea

MAPPO applies PPO independently to each agent in a multi-agent setting, with a shared centralized value function (critic) that can observe the global state. Despite its simplicity — essentially PPO with parameter sharing and centralized training with decentralized execution (CTDE) — MAPPO achieved state-of-the-art on multiple multi-agent benchmarks, often outperforming more complex algorithms.

## Mathematical Formulation

**Per-agent PPO objective (N agents, shared θ):**

```
L_i^CLIP(θ) = E_t [ min( r_t^i(θ) · Â_t^i,  clip(r_t^i(θ), 1-ε, 1+ε) · Â_t^i ) ]

where r_t^i = π_θ(a_t^i | o_t^i) / π_θ_old(a_t^i | o_t^i)
```

**Centralized value function:**

```
V_ψ(s_t)  where s_t = global state (all agent observations + additional info)
```

**Advantage per agent (using centralized V):**

```
Â_t^i = GAE(γ, λ) using V_ψ(s_t) and agent i's rewards
```

**Total loss (parameter sharing):**

```
L(θ) = (1/N) · Σ_{i=1}^{N} L_i^CLIP(θ)
```

## Properties

- On-policy, model-free
- Centralized training, decentralized execution (CTDE)
- Parameter sharing (optional but common)
- Cooperative multi-agent setting

## Key Hyperparameters

| Parameter | Typical Value | Notes |
|-----------|---------------|-------|
| All PPO hyperparameters | Same as PPO | See PPO doc |
| Value input | Global state | Centralized critic |
| Parameter sharing | Yes/No | Shared or separate per agent |
| Agent observation | Local only | Decentralized policy |
| Number of agents | 2–100+ | Benchmark dependent |
| Chunk length | 10 | For recurrent policies |

## Complexity

- **Time:** N agents × PPO cost, but parallelizable
- **Memory:** With parameter sharing — same as single-agent PPO + larger rollout buffer. Without — N × single-agent
- Scales reasonably to dozens of agents; hundreds require distributed training

## Primary Use Cases

- StarCraft Multi-Agent Challenge (SMAC)
- Cooperative navigation and formation
- Traffic signal control
- Multi-robot coordination
- Hanabi (cooperative card game)
- Google Research Football

## Known Limitations

1. **Cooperative only** — not designed for competitive or mixed-motive games
2. **Centralized value function** may not scale to very large agent counts
3. **Parameter sharing** assumes agent homogeneity — heterogeneous agents need separate networks
4. **Credit assignment** across agents remains challenging
5. **Global state assumption** may not hold in partially observable settings
6. **Sensitive to same implementation details** as PPO

## Major Variants

| Variant | Reference | Key Change |
|---------|-----------|------------|
| QMIX | Rashid et al., ICML 2018 | Value factorization for multi-agent Q-learning |
| MADDPG | Lowe et al., NeurIPS 2017 | Multi-agent DDPG (off-policy) |
| MAT | Wen et al., NeurIPS 2022 | Transformer-based coordination |
| HAPPO | Kuba et al., ICLR 2022 | Sequential update for heterogeneous agents |
| IPPO | — | Independent PPO without centralized critic |

## Relationship to Other Algorithms

- Direct extension of **PPO** to multi-agent settings
- Competes with dedicated MARL algorithms (QMIX, MADDPG) and often wins
- The simplicity argument parallels how PPO itself outperforms complex single-agent methods
- Can be combined with communication protocols

## Industry Deployment

- Multi-robot warehousing and coordination
- Traffic optimization systems
- Game AI (real-time strategy, team games)
- Autonomous vehicle fleet coordination (research stage)
- **Frameworks:** EPyMARL, MARLlib
