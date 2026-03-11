# Top 10 RL Algorithms (2026) — Research Survey

A comprehensive survey of the most widely used reinforcement learning algorithms as of early 2026, covering classic RL, LLM post-training, model-based methods, offline RL, and multi-agent systems.

## Algorithm Index

| # | Algorithm | Document | Category |
|---|-----------|----------|----------|
| 1 | [PPO](01-ppo.md) | Proximal Policy Optimization | Classic RL / LLM |
| 2 | [SAC](02-sac.md) | Soft Actor-Critic | Continuous Control |
| 3 | [DQN/Rainbow](03-dqn-rainbow.md) | Deep Q-Network + Improvements | Discrete Control |
| 4 | [TD3](04-td3.md) | Twin Delayed DDPG | Continuous Control |
| 5 | [GRPO](05-grpo.md) | Group Relative Policy Optimization | LLM Reasoning |
| 6 | [DPO](06-dpo.md) | Direct Preference Optimization | LLM Alignment |
| 7 | [RLHF-PPO](07-rlhf-ppo.md) | RL from Human Feedback | LLM Alignment |
| 8 | [Dreamer](08-dreamer.md) | World Model RL (v1/v2/v3) | Model-Based |
| 9 | [Decision Transformer](09-decision-transformer.md) | RL via Sequence Modeling | Offline RL |
| 10 | [MAPPO](10-mappo.md) | Multi-Agent PPO | Multi-Agent |

---

## Taxonomy

```
Reinforcement Learning Algorithms (2026)
├── Classic RL
│   ├── On-Policy, Model-Free
│   │   ├── PPO (continuous + discrete)
│   │   └── MAPPO (multi-agent)
│   ├── Off-Policy, Model-Free
│   │   ├── DQN / Rainbow (discrete)
│   │   ├── SAC (continuous, max-entropy)
│   │   └── TD3 (continuous, deterministic)
│   ├── Model-Based
│   │   └── Dreamer v1/v2/v3 (world model + imagination)
│   └── Offline / Sequence Modeling
│       └── Decision Transformer
├── LLM Post-Training
│   ├── Online RL
│   │   ├── RLHF-PPO (reward model + PPO)
│   │   └── GRPO (critic-free, group-relative)
│   └── Offline / Preference-Based
│       └── DPO (supervised from preferences)
```

---

## Comparative Summary

| Algorithm | On/Off-Policy | Actions | Sample Eff. | Memory | Primary Domain |
|-----------|--------------|---------|-------------|--------|----------------|
| PPO | On | Both | Low | Low | General RL, LLM |
| SAC | Off | Continuous | High | Medium | Robotics, Control |
| DQN/Rainbow | Off | Discrete | Medium | Medium | Games, Discrete |
| TD3 | Off | Continuous | High | Medium | Control |
| GRPO | On | Token seq. | Low | High* | LLM Reasoning |
| DPO | Offline | Token seq. | N/A | Medium* | LLM Alignment |
| RLHF-PPO | On | Token seq. | Low | Very High* | LLM Alignment |
| Dreamer | Both | Both | Very High | Medium | Visual Control |
| DT | Offline | Both | N/A | Low | Offline RL |
| MAPPO | On | Both | Low | Low-Med | Multi-Agent |

*Memory for LLM algorithms is dominated by model parameters (billions), not buffers.

---

## Pros and Cons

### PPO
- **Pros:** Simple, parallelizable, works for both classic RL and LLMs, well-tested in production
- **Cons:** Sample inefficient, implementation-detail sensitive, requires critic (expensive for LLMs)

### SAC
- **Pros:** Sample efficient (off-policy), automatic exploration (entropy), robust policies
- **Cons:** Continuous actions only, replay buffer memory, not easily parallelizable

### DQN/Rainbow
- **Pros:** Strong for discrete actions, well-understood, many improvements available
- **Cons:** Discrete only, memory-intensive, slow convergence, base DQN now outdated

### TD3
- **Pros:** Simple, stable, slightly cheaper than SAC, good for deterministic control
- **Cons:** No inherent exploration, generally slightly worse than SAC, noise tuning required

### GRPO
- **Pros:** No critic needed (~33% memory savings), excels at reasoning, verifiable rewards
- **Cons:** Expensive generation (G completions/prompt), noisy with small groups, on-policy

### DPO
- **Pros:** Extremely simple (supervised learning), no RL loop, cheap per iteration
- **Cons:** Offline only (base), distribution shift, length exploitation, preference data expensive

### RLHF-PPO
- **Pros:** Proven at scale (ChatGPT, Claude), flexible reward modeling
- **Cons:** 4 models in memory, extreme complexity, being displaced by GRPO/DPO

### Dreamer
- **Pros:** Excellent sample efficiency (10-100×), single hyperparameter set (V3), works from pixels
- **Cons:** Compounding model errors, complex implementation, wall-clock can be worse

### Decision Transformer
- **Pros:** Conceptually elegant, leverages Transformer scaling, no TD instabilities
- **Cons:** Cannot stitch trajectories, offline only, underperforms CQL/IQL on many tasks

### MAPPO
- **Pros:** Surprisingly effective, simple (just PPO), scales to dozens of agents
- **Cons:** Cooperative only, centralized critic scales poorly, credit assignment hard

---

## When to Use Which

### By Domain

| Domain | Recommended | Alternative |
|--------|------------|-------------|
| **Continuous control / robotics** | SAC | TD3, PPO |
| **Discrete actions / games** | DQN/Rainbow | PPO |
| **LLM reasoning (math, code)** | GRPO | RLHF-PPO |
| **LLM alignment (preferences)** | DPO → GRPO | RLHF-PPO |
| **Visual control from pixels** | Dreamer | Rainbow, PPO |
| **Multi-agent cooperative** | MAPPO | QMIX |
| **Offline data only** | DT / CQL / IQL | DPO (for LLMs) |
| **Sim-to-real robotics** | PPO | SAC |

### By Constraint

| Constraint | Best Choice | Why |
|-----------|------------|-----|
| **Minimal env interactions** | SAC or Dreamer | Off-policy / model-based reuse |
| **Massive parallelism** | PPO / MAPPO | Embarrassingly parallel rollouts |
| **Limited memory** | PPO | No replay buffer, small overhead |
| **Simple implementation** | DPO (LLM) / PPO (classic) | Fewest moving parts |
| **No reward model available** | DPO | Works from preference pairs directly |
| **Verifiable rewards (math/code)** | GRPO | Designed for programmatic rewards |
| **Production deployment** | PPO or SAC | Best framework support, most battle-tested |

### Decision Flowchart

```
Is this an LLM task?
├── Yes
│   ├── Do you have preference pairs? → DPO (+ Online DPO for better results)
│   ├── Do you have verifiable rewards? → GRPO
│   └── Do you need a learned reward model? → RLHF-PPO (or GRPO + ORM)
└── No
    ├── Is it multi-agent? → MAPPO
    ├── Discrete actions? → DQN/Rainbow
    ├── Continuous actions?
    │   ├── Need sample efficiency? → SAC (or Dreamer if from pixels)
    │   └── Need massive parallelism? → PPO
    └── Only offline data? → Decision Transformer / CQL / IQL
```

---

## The 2025-2026 Landscape Shift

1. **GRPO has displaced PPO for LLM reasoning.** DeepSeek-R1's success (Jan 2025) showed critic-free RL with verifiable rewards can elicit emergent reasoning. By early 2026, GRPO and variants (DAPO, Dr. GRPO, RLOO) are the default.

2. **DPO remains dominant for preference alignment** but increasingly combined with online methods. The SFT → DPO → GRPO pipeline is becoming standard.

3. **PPO retains its position for classic RL** (robotics, control, games). No algorithm has displaced it for general-purpose on-policy training.

4. **SAC has won the off-policy continuous control competition** over TD3 due to automatic entropy tuning.

5. **DreamerV3** demonstrated single-hyperparameter viability across diverse domains, advancing model-based RL from niche to practical.

6. **Decision Transformer** has been more influential conceptually (bridging LLM and RL) than practically. The "foundation model for control" vision is still developing.

7. **Multi-agent RL** is growing, with MAPPO as the go-to baseline that is surprisingly hard to beat.

---

## References

```
[1]  Schulman et al., "Proximal Policy Optimization Algorithms," arXiv:1707.06347, 2017.
[2]  Schulman et al., "Trust Region Policy Optimization," ICML, 2015.
[3]  Huang et al., "The 37 Implementation Details of PPO," ICLR Blog, 2022.
[4]  Espeholt et al., "IMPALA," ICML, 2018.
[5]  Cobbe et al., "Phasic Policy Gradient," ICML, 2021.
[6]  Haarnoja et al., "Soft Actor-Critic," ICML, 2018.
[7]  Haarnoja et al., "SAC Algorithms and Applications," arXiv:1812.05905, 2018.
[8]  Chen et al., "REDQ," ICLR, 2021.
[9]  Hiraoka et al., "DroQ," ICLR, 2022.
[10] Bhatt et al., "CrossQ," ICLR, 2024.
[11] Mnih et al., "Human-level control through deep RL," Nature, 2015.
[12] Hessel et al., "Rainbow," AAAI, 2018.
[13] van Hasselt et al., "Double Q-learning," AAAI, 2016.
[14] Wang et al., "Dueling Networks," ICML, 2016.
[15] Bellemare et al., "Distributional RL," ICML, 2017.
[16] Schaul et al., "Prioritized Experience Replay," ICLR, 2016.
[17] Dabney et al., "QR-DQN," AAAI, 2018.
[18] Dabney et al., "IQN," ICML, 2018.
[19] Badia et al., "Agent57," ICML, 2020.
[20] Schwarzer et al., "BBF," ICML, 2023.
[21] Fujimoto et al., "TD3," ICML, 2018.
[22] Lillicrap et al., "DDPG," ICLR, 2016.
[23] Fujimoto et al., "TD7," ICML, 2023.
[24] Silver et al., "DPG," ICML, 2014.
[25] Shao et al., "DeepSeekMath," arXiv:2402.03300, 2024.
[26] Guo et al., "DeepSeek-R1," arXiv:2501.12948, 2025.
[27] Lightman et al., "Let's Verify Step by Step," ICLR, 2024.
[28] Liu et al., "DAPO," arXiv, 2025.
[29] Liu et al., "Understanding R1-Zero-Like Training," arXiv, 2025.
[30] Ahmadian et al., "Back to Basics: REINFORCE for RLHF," ACL, 2024.
[31] Williams, "REINFORCE," Machine Learning, 1992.
[32] Rafailov et al., "DPO," NeurIPS, 2023.
[33] Tunstall et al., "Zephyr," arXiv:2310.16944, 2023.
[34] Ivison et al., "Tulu 2," arXiv:2311.10702, 2023.
[35] Azar et al., "IPO," AISTATS, 2024.
[36] Ethayarajh et al., "KTO," ICML, 2024.
[37] Hong et al., "ORPO," EMNLP, 2024.
[38] Meng et al., "SimPO," NeurIPS, 2024.
[39] Guo et al., "Online AI Feedback," arXiv:2402.04792, 2024.
[40] Liu et al., "RSO," ICLR, 2024.
[41] Wu et al., "SPPO," ICML, 2024.
[42] Munos et al., "Nash Learning from Human Feedback," ICML, 2024.
[43] Christiano et al., "Deep RL from Human Preferences," NeurIPS, 2017.
[44] Ouyang et al., "InstructGPT," NeurIPS, 2022.
[45] Stiennon et al., "Learning to summarize from human feedback," NeurIPS, 2020.
[46] Bai et al., "Constitutional AI," arXiv:2212.08073, 2022.
[47] Dong et al., "RAFT," TMLR, 2023.
[48] Hafner et al., "DreamerV1," ICLR, 2020.
[49] Hafner et al., "DreamerV2," ICLR, 2021.
[50] Hafner et al., "DreamerV3," arXiv:2301.04104, 2023.
[51] Hafner et al., "PlaNet," ICML, 2019.
[52] Hansen et al., "TD-MPC2," ICLR, 2024.
[53] Micheli et al., "IRIS," ICML, 2023.
[54] Alonso et al., "DIAMOND," NeurIPS, 2024.
[55] Schrittwieser et al., "MuZero," Nature, 2020.
[56] Chen et al., "Decision Transformer," NeurIPS, 2021.
[57] Janner et al., "Trajectory Transformer," NeurIPS, 2021.
[58] Reed et al., "Gato," arXiv:2205.06175, 2022.
[59] Zheng et al., "Online Decision Transformer," ICML, 2022.
[60] Wu et al., "Elastic Decision Transformer," NeurIPS, 2023.
[61] Yamagata et al., "QDT," arXiv:2209.03993, 2022.
[62] Kumar et al., "CQL," NeurIPS, 2020.
[63] Kostrikov et al., "IQL," ICLR, 2022.
[64] Fujimoto et al., "BCQ," ICML, 2019.
[65] Yu et al., "MAPPO," NeurIPS, 2022.
[66] Rashid et al., "QMIX," ICML, 2018.
[67] Lowe et al., "MADDPG," NeurIPS, 2017.
[68] Wen et al., "MAT," NeurIPS, 2022.
[69] Kuba et al., "HAPPO," ICLR, 2022.
```

---

*Survey compiled March 2026. Algorithm rankings reflect usage in research publications, open-source frameworks, and documented industry deployments through early 2026.*
