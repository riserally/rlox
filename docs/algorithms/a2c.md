# A2C -- Advantage Actor-Critic

## Intuition

A2C is the synchronous variant of A3C. It uses a shared actor-critic network trained with a single gradient step per rollout (no clipping, no replay). Multiple parallel environments provide diverse experience that reduces correlation between samples. A2C is simpler and faster per update than PPO, but less stable for large policy changes.

## Key Equations

The policy gradient with advantage:

$$
\nabla_\theta J(\theta) = \mathbb{E}_t \left[ \nabla_\theta \log \pi_\theta(a_t | s_t) \, \hat{A}_t \right]
$$

Advantage estimated via GAE (or N-step returns when $\lambda = 1$):

$$
\hat{A}_t = \sum_{l=0}^{n-1} (\gamma \lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

The combined loss:

$$
L(\theta) = -L^{\pi}(\theta) + c_1 \, L^{VF}(\theta) - c_2 \, H[\pi_\theta]
$$

where $L^{VF} = \frac{1}{2} \| V_\theta(s_t) - G_t \|^2$.

## Pseudocode

```
algorithm A2C:
    initialize shared actor-critic network pi_theta, V_theta

    for iteration = 1, 2, ... do
        collect n_steps transitions from n_envs parallel environments
        compute GAE advantages A_t (with gae_lambda=1.0 for n-step returns)
        compute returns G_t = A_t + V_theta(s_t)

        # Single gradient step (no minibatches, no epochs)
        L_policy = -mean(log pi_theta(a_t | s_t) * A_t)
        L_value  = mean((V_theta(s_t) - G_t)^2)
        L_entropy = -H[pi_theta]

        loss = L_policy + vf_coef * L_value + ent_coef * L_entropy
        update theta with RMSprop, clip gradients to max_grad_norm
```

## Quick Start

```python
from rlox import Trainer

trainer = Trainer("a2c", env="CartPole-v1", seed=42)
metrics = trainer.train(total_timesteps=100_000)
```

For Atari-style environments:

```python
trainer = Trainer("a2c", env="PongNoFrameskip-v4", seed=42, config={
    "learning_rate": 7e-4,
    "n_steps": 5,
    "n_envs": 16,
    "gamma": 0.99,
    "gae_lambda": 1.0,
    "ent_coef": 0.01,
})
metrics = trainer.train(total_timesteps=10_000_000)
```

## Hyperparameters

All defaults from `A2CConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | `7e-4` | RMSprop learning rate |
| `n_steps` | `5` | Rollout length per update |
| `gamma` | `0.99` | Discount factor |
| `gae_lambda` | `1.0` | GAE lambda (1.0 = full n-step returns) |
| `vf_coef` | `0.5` | Value function loss coefficient |
| `ent_coef` | `0.01` | Entropy bonus coefficient |
| `max_grad_norm` | `0.5` | Gradient clipping threshold |
| `normalize_advantages` | `False` | Normalize advantages per batch |
| `n_envs` | `8` | Number of parallel environments |
| `hidden` | `64` | Hidden layer width |

## When to Use

- **Use A2C when:** you want a fast, simple on-policy baseline that is easy to debug and scales well with parallel environments.
- **Do not use A2C when:** you need stable training with large batch sizes (use [PPO](ppo.md)) or sample efficiency (use [SAC](sac.md)).

## References

- Mnih, V., Badia, A. P., Mirza, M., et al. (2016). Asynchronous Methods for Deep Reinforcement Learning. *ICML* (A3C, the async predecessor).
- Stable Baselines3 A2C implementation (synchronous variant).
