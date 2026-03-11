# SAC — Soft Actor-Critic

> Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor," ICML, 2018.
> Haarnoja et al., "Soft Actor-Critic Algorithms and Applications," arXiv:1812.05905, 2018. (Auto temperature tuning.)

## Key Idea

SAC maximizes a "maximum entropy" objective — the standard expected return plus an entropy bonus — encouraging exploration and leading to more robust policies. Unlike PPO, SAC is off-policy, meaning it reuses past experience from a replay buffer, dramatically improving sample efficiency. The automatic temperature parameter α balances reward maximization and entropy.

## Mathematical Formulation

**Maximum entropy objective:**

```
J(π) = Σ_t E_{(s_t,a_t)~ρ_π} [ r(s_t, a_t) + α · H(π(·|s_t)) ]
     = Σ_t E [ r(s_t, a_t) - α · log π(a_t | s_t) ]
```

**Soft Q-function (Bellman equation):**

```
Q(s_t, a_t) = r(s_t, a_t) + γ · E_{s_{t+1}} [ V(s_{t+1}) ]
V(s_t) = E_{a~π} [ Q(s_t, a) - α · log π(a | s_t) ]
```

**Critic loss (twin Q-networks):**

```
L_Q(φ_i) = E_{(s,a,r,s')~D} [ (Q_{φ_i}(s,a) - y)² ]
y = r + γ · ( min_{j=1,2} Q_{φ_j'}(s', a') - α · log π_θ(a'|s') )
where a' ~ π_θ(·|s')
```

**Actor loss (reparameterization trick):**

```
L_π(θ) = E_{s~D} [ E_{a~π_θ} [ α · log π_θ(a|s) - Q_φ(s, a) ] ]
```

**Automatic temperature tuning:**

```
L(α) = E_{a~π} [ -α · ( log π(a|s) + H_target ) ]
where H_target = -dim(A)  (heuristic)
```

## Properties

- Off-policy, model-free
- Actor-critic with twin Q-networks (clipped double-Q)
- Continuous action spaces (Gaussian policy with tanh squashing)

## Key Hyperparameters

| Parameter | Typical Value | Notes |
|-----------|---------------|-------|
| `α` (temperature) | Auto-tuned | Or fixed at 0.2 |
| `γ` | 0.99 | Discount factor |
| `τ` (target net) | 0.005 | Polyak averaging |
| Replay buffer | 1e6 | Transitions |
| Batch size | 256 | Per gradient step |
| Learning rate | 3e-4 (Adam) | Actor and critics |
| H_target | -dim(A) | Entropy target |
| UTD ratio | 1 | Gradient steps per env step |

## Complexity

- **Time per update:** O(B × C_forward) — single gradient step per env step (or more with higher UTD)
- **Memory:** O(|D| × (d_obs + d_act + 1)) for replay buffer + 5 networks (actor, 2 critics, 2 targets)
- **Sample efficiency:** High — typically 10-100x fewer env interactions than PPO

## Primary Use Cases

- Continuous control: MuJoCo benchmarks (state-of-art or competitive)
- Robotics: Real-world manipulation (Berkeley), locomotion
- Autonomous driving research
- Any task where sample efficiency matters and actions are continuous

## Known Limitations

1. **Continuous actions only** — SAC-Discrete exists but is less popular
2. **Replay buffer memory** can be prohibitive for high-dimensional observations
3. **Entropy bonus** can cause over-exploration
4. **Sensitive to reward scale** (auto-tuning helps but isn't perfect)
5. **Q-value overestimation** can still occur despite twin critics
6. **Not trivially parallelizable** across environments (unlike PPO)

## Major Variants

| Variant | Reference | Key Change |
|---------|-----------|------------|
| SAC v2 | Haarnoja et al., 2018 | Automatic temperature tuning |
| REDQ | Chen et al., ICLR 2021 | Ensemble of Q-functions, high UTD |
| DroQ | Hiraoka et al., 2022 | Dropout on Q-networks |
| CrossQ | Bhatt et al., ICLR 2024 | Batch norm across critics, no target nets |
| SAC-N | — | N-critic ensemble |

## Relationship to Other Algorithms

- Competes with **TD3** for continuous control; generally preferred for auto-exploration
- Off-policy nature contrasts with **PPO** — more sample-efficient but harder to scale
- Can be combined with **Dreamer**-style world models for even higher sample efficiency
- Shares twin Q-network trick with TD3 (SAC adopted it from TD3)

## Industry Deployment

- **Robotics labs:** Berkeley, Google DeepMind (manipulation)
- **SB3:** Default off-policy algorithm
- **Google:** MT-Opt (real-robot learning at scale)
