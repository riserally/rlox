# TD3+BC -- A Minimalist Approach to Offline RL

## Intuition

TD3+BC takes the simplest possible approach to offline RL: add a behavioral cloning (BC) term to the TD3 actor loss. The BC regularizer penalizes the policy for deviating from dataset actions, preventing it from exploiting Q-value overestimation on out-of-distribution actions. The weight of the BC term is normalized by the Q-value magnitude, so the algorithm automatically balances between maximizing Q-values and staying close to the data. Despite its simplicity, TD3+BC is competitive with much more complex offline RL algorithms.

## Key Equations

The actor loss combines Q-value maximization with BC regularization:

$$
L_{\text{actor}} = -\lambda \, Q_1(s, \pi(s)) + \left\| \pi(s) - a_{\text{data}} \right\|^2
$$

where the balancing coefficient is:

$$
\lambda = \frac{\alpha}{\frac{1}{N} \sum_i |Q_1(s_i, a_i)|}
$$

The critic update follows standard TD3 with clipped double Q-learning and target policy smoothing:

$$
y = r + \gamma \min(Q_1^-(s', \tilde{a}'), Q_2^-(s', \tilde{a}')), \quad \tilde{a}' = \pi^-(s') + \text{clip}(\epsilon, -c, c)
$$

## Pseudocode

```
algorithm TD3+BC:
    initialize actor pi, twin critics Q1, Q2, target networks
    load offline dataset D

    for update = 1 to n_updates do
        sample minibatch from D

        # Critic update (same as TD3)
        target_noise = clip(N(0, sigma), -c, c)
        a' = clip(pi_target(s') + target_noise, -a_max, a_max)
        y = r + gamma * (1 - done) * min(Q1_target(s', a'), Q2_target(s', a'))
        update Q1, Q2 with MSE(Q(s,a), y)

        # Actor update (every policy_delay steps)
        if update % policy_delay == 0:
            lambda = alpha / mean(|Q1(s, a_data)|)
            L = -lambda * mean(Q1(s, pi(s))) + MSE(pi(s), a_data)
            update pi

        # Soft target updates
        polyak_update(all targets, tau)
```

## Quick Start

TD3+BC uses the offline algorithm interface:

```python
from rlox.offline import OfflineDatasetBuffer
from rlox.algorithms.td3_bc import TD3BC

dataset = OfflineDatasetBuffer.from_d4rl("halfcheetah-medium-v2")
agent = TD3BC(
    dataset=dataset,
    obs_dim=17,
    act_dim=6,
    alpha=2.5,
)
metrics = agent.train(n_updates=100_000)
```

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | `2.5` | BC regularization weight (higher = more conservative) |
| `hidden` | `256` | Hidden layer width |
| `learning_rate` | `3e-4` | Learning rate |
| `tau` | `0.005` | Soft target update rate |
| `gamma` | `0.99` | Discount factor |
| `policy_delay` | `2` | Actor update frequency (every N critic updates) |
| `target_noise` | `0.2` | Target policy smoothing noise std |
| `noise_clip` | `0.5` | Target noise clipping range |
| `act_high` | `1.0` | Action space upper bound |
| `batch_size` | `256` | Minibatch size |

## When to Use

- **Use TD3+BC when:** you want a dead-simple offline RL baseline that is easy to implement and tune, with continuous action spaces.
- **Prefer TD3+BC over CQL/IQL when:** simplicity and reproducibility matter more than squeezing out extra performance.
- **Do not use TD3+BC when:** the offline dataset is very sub-optimal (the BC term will anchor the policy too close to bad data), or you need discrete actions.

## References

- Fujimoto, S. & Gu, S. S. (2021). [A Minimalist Approach to Offline Reinforcement Learning](https://arxiv.org/abs/2106.06860). *NeurIPS 2021*. [arXiv:2106.06860](https://arxiv.org/abs/2106.06860).
