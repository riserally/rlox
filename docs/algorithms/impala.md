# IMPALA -- Distributed Actor-Learner Architecture

## Intuition

IMPALA (Importance Weighted Actor-Learner Architecture) decouples acting from learning. Multiple actor threads collect experience in parallel while a centralized learner consumes batches from a queue. Because actors run an older policy than the learner, IMPALA uses V-trace importance sampling corrections to account for the policy lag. This enables high throughput without the staleness problems of naive asynchronous methods.

## Key Equations

V-trace target for off-policy correction:

$$
v_s = V(x_s) + \sum_{t=s}^{s+n-1} \gamma^{t-s} \left( \prod_{i=s}^{t-1} c_i \right) \delta_t V
$$

where the temporal difference is:

$$
\delta_t V = \rho_t (r_t + \gamma V(x_{t+1}) - V(x_t))
$$

The truncated importance weights:

$$
\rho_t = \min\left(\bar{\rho}, \frac{\pi(a_t | x_t)}{\mu(a_t | x_t)}\right), \quad c_t = \min\left(\bar{c}, \frac{\pi(a_t | x_t)}{\mu(a_t | x_t)}\right)
$$

Policy gradient using V-trace advantages:

$$
\nabla_\theta J = \mathbb{E} \left[ \rho_t \nabla_\theta \log \pi_\theta(a_t | x_t) (r_t + \gamma v_{t+1} - V(x_t)) \right]
$$

## Pseudocode

```
algorithm IMPALA:
    initialize learner network pi_theta, V_theta
    initialize experience queue Q (max size = queue_size)
    launch n_actors actor threads

    # Each actor thread:
    actor(i):
        copy weights from learner: mu <- theta
        for step = 1, 2, ... do
            collect n_steps transitions using mu
            enqueue (states, actions, rewards, mu_probs) to Q
            periodically: mu <- theta

    # Learner:
    for batch from Q do
        compute importance weights rho_t = pi_theta(a|s) / mu(a|s)
        clip: rho_t = min(rho_clip, rho_t)
        clip: c_t = min(c_clip, rho_t)
        compute V-trace targets v_s
        compute advantages A_t = rho_t * (r_t + gamma * v_{t+1} - V(x_t))

        L_policy  = -mean(log pi_theta(a|s) * A_t)
        L_value   = mean((V_theta(s) - v_s)^2)
        L_entropy = -H[pi_theta]

        loss = L_policy + vf_coef * L_value + ent_coef * L_entropy
        update theta with RMSprop, clip gradients to max_grad_norm
```

## Quick Start

```python
from rlox import Trainer

trainer = Trainer("impala", env="CartPole-v1", seed=42, config={
    "n_actors": 4,
    "n_steps": 20,
})
metrics = trainer.train(total_timesteps=500_000)
```

For large-scale training:

```python
trainer = Trainer("impala", env="PongNoFrameskip-v4", seed=42, config={
    "n_actors": 16,
    "n_steps": 20,
    "n_envs_per_actor": 2,
    "learning_rate": 4e-4,
    "queue_size": 32,
    "rho_clip": 1.0,
    "c_clip": 1.0,
})
metrics = trainer.train(total_timesteps=50_000_000)
```

## Hyperparameters

All defaults from `IMPALAConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | `4e-4` | RMSprop learning rate |
| `n_actors` | `4` | Number of actor threads |
| `n_steps` | `20` | Rollout length per actor per batch |
| `gamma` | `0.99` | Discount factor |
| `vf_coef` | `0.5` | Value loss coefficient |
| `ent_coef` | `0.01` | Entropy bonus coefficient |
| `max_grad_norm` | `40.0` | Maximum gradient norm for clipping |
| `rho_clip` | `1.0` | V-trace truncation for importance weights ($\bar{\rho}$) |
| `c_clip` | `1.0` | V-trace truncation for trace coefficients ($\bar{c}$) |
| `queue_size` | `16` | Maximum experience queue size |
| `hidden` | `256` | Hidden layer width |
| `n_envs_per_actor` | `1` | Environments per actor thread |

## When to Use

- **Use IMPALA when:** you need high-throughput distributed training, especially for Atari-scale problems or when you have many CPU cores available.
- **Do not use IMPALA when:** you have a single machine with limited cores (use [PPO](ppo.md) or [A2C](a2c.md)), or need sample efficiency over throughput (use [SAC](sac.md)).

## References

- Espeholt, L., Soyer, H., Munos, R., et al. (2018). IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures. *ICML 2018*.
