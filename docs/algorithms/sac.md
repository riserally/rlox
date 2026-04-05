# SAC -- Soft Actor-Critic

## Intuition

Soft Actor-Critic augments the standard RL objective with an entropy bonus, encouraging the policy to be as random as possible while still maximizing return. This maximum-entropy framework produces robust policies that explore effectively and are less sensitive to hyperparameters. SAC uses twin Q-networks to mitigate overestimation and automatically tunes the entropy coefficient.

## Key Equations

The maximum-entropy objective:

$$
J(\pi) = \sum_{t=0}^{T} \mathbb{E}_{(s_t, a_t) \sim \rho_\pi} \left[ r(s_t, a_t) + \alpha \, H(\pi(\cdot | s_t)) \right]
$$

Soft Bellman backup for twin Q-functions:

$$
Q_\text{target}(s_t, a_t) = r_t + \gamma (1 - d_t) \left( \min_{i=1,2} Q_{\phi_i'}(s_{t+1}, \tilde{a}_{t+1}) - \alpha \log \pi_\theta(\tilde{a}_{t+1} | s_{t+1}) \right)
$$

where $\tilde{a}_{t+1} \sim \pi_\theta(\cdot | s_{t+1})$.

Policy loss (maximize Q while maximizing entropy):

$$
L_\pi(\theta) = \mathbb{E}_{s_t} \left[ \alpha \log \pi_\theta(\tilde{a}_t | s_t) - \min_{i=1,2} Q_{\phi_i}(s_t, \tilde{a}_t) \right]
$$

Automatic entropy tuning:

$$
L(\alpha) = -\alpha \, \mathbb{E}_{a_t \sim \pi} \left[ \log \pi_\theta(a_t | s_t) + \bar{H} \right]
$$

where $\bar{H} = -\dim(\mathcal{A})$ is the target entropy.

## Pseudocode

```
algorithm SAC:
    initialize actor pi_theta, twin critics Q_phi1, Q_phi2
    initialize target networks Q_phi1', Q_phi2'
    initialize replay buffer D, entropy coefficient alpha

    for step = 1, 2, ... do
        if step < learning_starts:
            a ~ Uniform(action_space)
        else:
            a ~ pi_theta(.|s)

        store (s, a, r, s', done) in D

        if step >= learning_starts:
            sample minibatch (s, a, r, s', done) from D
            a' ~ pi_theta(.|s')

            # Critic update
            y = r + gamma * (1-done) * (min(Q_phi1'(s',a'), Q_phi2'(s',a')) - alpha * log pi(a'|s'))
            update phi1, phi2 to minimize (Q_phi_i(s,a) - y)^2

            # Actor update
            a_new ~ pi_theta(.|s)
            update theta to minimize alpha * log pi(a_new|s) - min(Q_phi1(s,a_new), Q_phi2(s,a_new))

            # Entropy tuning (if auto_entropy)
            update alpha to minimize -alpha * (log pi(a_new|s) + target_entropy)

            # Target network update
            phi_i' <- tau * phi_i + (1-tau) * phi_i'
```

## Quick Start

```python
from rlox import Trainer

trainer = Trainer("sac", env="Pendulum-v1", seed=42)
metrics = trainer.train(total_timesteps=50_000)
```

For MuJoCo locomotion:

```python
trainer = Trainer("sac", env="HalfCheetah-v4", seed=42, config={
    "learning_rate": 3e-4,
    "buffer_size": 1_000_000,
    "batch_size": 256,
    "tau": 0.005,
    "gamma": 0.99,
    "learning_starts": 5000,
})
metrics = trainer.train(total_timesteps=1_000_000)
```

## Hyperparameters

All defaults from `SACConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | `3e-4` | Learning rate for all optimizers |
| `buffer_size` | `1_000_000` | Replay buffer capacity |
| `batch_size` | `256` | Minibatch size |
| `tau` | `0.005` | Polyak averaging coefficient for target networks |
| `gamma` | `0.99` | Discount factor |
| `target_entropy` | `None` | Target entropy (auto = $-\dim(\mathcal{A})$) |
| `auto_entropy` | `True` | Automatically tune entropy coefficient |
| `learning_starts` | `1000` | Random exploration steps before training |
| `hidden` | `256` | Hidden layer width |

## When to Use

- **Use SAC when:** you need sample-efficient continuous control, want robust exploration via entropy regularization, or have a continuous action space.
- **Do not use SAC when:** your action space is discrete (use [DQN](dqn.md)) or you need on-policy training for strict policy gradient analysis (use [PPO](ppo.md)).

## References

- Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290). *ICML 2018*.
- Haarnoja, T., Zhou, A., Hartikainen, K., Tucker, G., Ha, S., Tan, J., ... & Levine, S. (2018). Soft Actor-Critic Algorithms and Applications. [arXiv:1812.05905](https://arxiv.org/abs/1812.05905).
