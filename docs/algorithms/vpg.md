# VPG -- Vanilla Policy Gradient

## Intuition

Vanilla Policy Gradient (REINFORCE) is the simplest policy gradient algorithm. It collects complete episodes, computes the total return for each timestep, and nudges the policy parameters in the direction that makes high-return actions more likely. VPG is rarely used in practice due to high variance, but it is the conceptual foundation for all policy gradient methods in rlox (A2C, PPO, TRPO).

## Key Equations

The policy gradient theorem gives the gradient of the expected return:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) \, G_t \right]
$$

where $G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k$ is the discounted return-to-go.

With a baseline $b(s_t)$ (typically a learned value function $V_\phi(s_t)$), the variance is reduced without introducing bias:

$$
\nabla_\theta J(\theta) = \mathbb{E} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) \, \hat{A}_t \right]
$$

where $\hat{A}_t = G_t - V_\phi(s_t)$ is the advantage estimate.

## Pseudocode

```
algorithm VPG:
    initialize policy network pi_theta
    initialize value network V_phi (optional baseline)

    for iteration = 1, 2, ... do
        collect trajectories {(s_t, a_t, r_t)} by running pi_theta
        compute returns G_t = sum_{k=t}^{T} gamma^{k-t} * r_k
        compute advantages A_t = G_t - V_phi(s_t)

        # Policy update
        L_policy = -mean(log pi_theta(a_t | s_t) * A_t)
        theta <- theta - alpha * grad(L_policy)

        # Baseline update
        L_value = mean((V_phi(s_t) - G_t)^2)
        phi <- phi - alpha_v * grad(L_value)
```

## Quick Start

VPG is not implemented as a standalone algorithm in rlox. Instead, use PPO with `n_epochs=1` and `clip_eps=1.0` (effectively disabling clipping) for an equivalent single-pass policy gradient:

```python
from rlox import Trainer

trainer = Trainer("ppo", env="CartPole-v1", seed=42, config={
    "n_epochs": 1,
    "clip_eps": 1.0,       # no clipping = vanilla policy gradient
    "n_steps": 2048,
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 1.0,     # Monte Carlo returns (no GAE)
})
metrics = trainer.train(total_timesteps=100_000)
```

## Hyperparameters

Since VPG is emulated via PPO, the relevant parameters from `PPOConfig`:

| Parameter | Value for VPG | Description |
|-----------|---------------|-------------|
| `n_epochs` | `1` | Single pass over rollout data |
| `clip_eps` | `1.0` | Disables ratio clipping |
| `n_steps` | `2048` | Rollout length (longer is better for VPG) |
| `learning_rate` | `3e-4` | Adam learning rate |
| `gamma` | `0.99` | Discount factor |
| `gae_lambda` | `1.0` | Set to 1.0 for Monte Carlo returns |
| `vf_coef` | `0.5` | Value function loss weight |
| `ent_coef` | `0.0` | Entropy bonus (0 for pure VPG) |

## When to Use

- **Use VPG when:** you are learning policy gradients for the first time, or need a minimal baseline for comparison.
- **Do not use VPG when:** you need stable, sample-efficient training. Use [PPO](ppo.md) or [A2C](a2c.md) instead.

## References

- Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine Learning*, 8(3-4), 229-256.
- Sutton, R. S., McAllester, D., Singh, S., & Mansour, Y. (1999). Policy gradient methods for reinforcement learning with function approximation. *NeurIPS*.
