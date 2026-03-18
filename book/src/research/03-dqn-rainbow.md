# DQN and Rainbow

> Mnih et al., "Human-level control through deep reinforcement learning," Nature, 2015.
> Hessel et al., "Rainbow: Combining Improvements in Deep Reinforcement Learning," AAAI, 2018.

## Key Idea

DQN demonstrated that a single deep neural network can learn to play Atari games from raw pixels by combining Q-learning with experience replay and a target network. Rainbow combines six orthogonal improvements to DQN (double Q-learning, prioritized replay, dueling architecture, multi-step returns, distributional RL, noisy networks) into one agent, achieving superhuman performance on most Atari games.

## Mathematical Formulation

**DQN (Q-learning with function approximation):**

```
Loss: L(θ) = E_{(s,a,r,s')~D} [ (y - Q_θ(s,a))² ]
Target: y = r + γ · max_{a'} Q_{θ'}(s', a')
where θ' are target network parameters (periodically copied from θ)
```

**Double DQN (decouples selection from evaluation):**

```
y = r + γ · Q_{θ'}(s', argmax_{a'} Q_θ(s', a'))
```

**Dueling architecture:**

```
Q_θ(s,a) = V_θ(s) + A_θ(s,a) - mean_{a'} A_θ(s,a')
```

**Distributional RL (C51):**

```
Z(s,a) ~ categorical over {z_1, ..., z_N}
Loss: KL( Φ · Z_{θ'}(s', a*) || Z_θ(s, a) )
where Φ is the distributional Bellman projection
```

**Rainbow = Double DQN + Prioritized Replay + Dueling + Multi-step + C51 + NoisyNet**

## Properties

- Off-policy, model-free
- Value-based (no explicit policy — actions via argmax)
- Discrete action spaces only

## Key Hyperparameters

| Parameter | Typical Value | Notes |
|-----------|---------------|-------|
| `γ` | 0.99 | Discount |
| Replay buffer | 1e6 | Transitions |
| Batch size | 32 | |
| Target net update | Every 10k steps | Hard copy or Polyak |
| `ε` (exploration) | 1.0 → 0.01 | Linear decay ~1M steps |
| Learning rate | 2.5e-4 (Adam) | |
| Multi-step n | 3 | Rainbow |
| C51 atoms | 51 | Distributional support |

## Complexity

- **Time:** O(B × C_forward) per gradient step; one step per env step
- **Memory:** O(|D| × d_frame × stack_size) — replay buffer dominates for Atari (84×84×4)
- Rainbow roughly 2× the computation of DQN due to distributional heads

## Primary Use Cases

- Atari 2600 games (canonical benchmark)
- Board games, card games, any discrete-action domain
- Recommendation systems (actions = items to recommend)
- Network routing, resource allocation

## Known Limitations

1. **Discrete actions only** — cannot directly handle continuous action spaces
2. **Maximization bias** (partially addressed by Double DQN)
3. **Poor exploration** in sparse reward settings
4. **Memory-intensive** — frame stacking and replay buffer for image observations
5. **Slow convergence** compared to policy gradient in some domains
6. DQN alone (without Rainbow) performs poorly by modern standards

## Major Variants

| Variant | Reference | Key Change |
|---------|-----------|------------|
| Double DQN | van Hasselt et al., AAAI 2016 | Decoupled selection/evaluation |
| Dueling DQN | Wang et al., ICML 2016 | V+A decomposition |
| PER | Schaul et al., ICLR 2016 | Prioritized replay |
| C51 | Bellemare et al., ICML 2017 | Distributional returns |
| QR-DQN | Dabney et al., AAAI 2018 | Quantile regression |
| IQN | Dabney et al., ICML 2018 | Implicit quantile networks |
| Agent57 | Badia et al., ICML 2020 | Superhuman on all 57 Atari games |
| BBF | Schwarzer et al., ICML 2023 | Sample-efficient with bigger nets |

## Relationship to Other Algorithms

- **Foundational** — DQN launched the deep RL era
- **SAC** and **TD3** extend Q-learning ideas to continuous actions
- **Decision Transformer** offers an alternative approach to the same Atari benchmarks
- **Dreamer** often compared on Atari as a model-based alternative

## Industry Deployment

- **DeepMind:** Original Atari results, 3D environments
- **Recommendation engines** at scale (discrete item selection)
- **Network optimization:** Google data center cooling (DQN-inspired)
