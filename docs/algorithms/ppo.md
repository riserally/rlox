# PPO -- Proximal Policy Optimization

## Intuition

PPO prevents destructively large policy updates by clipping the probability ratio between the new and old policy. This gives most of the stability benefits of TRPO's trust region constraint, but with a first-order optimizer and minimal implementation complexity. PPO is the default algorithm in rlox and the recommended starting point for most tasks.

## Key Equations

The clipped surrogate objective:

$$
L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right]
$$

where $r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_\text{old}}(a_t | s_t)}$ is the importance sampling ratio.

The full loss combines policy, value, and entropy terms:

$$
L(\theta) = -L^{CLIP}(\theta) + c_1 \, L^{VF}(\theta) - c_2 \, H[\pi_\theta]
$$

Advantages are computed using Generalized Advantage Estimation (GAE):

$$
\hat{A}_t = \sum_{l=0}^{T-t} (\gamma \lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

## Pseudocode

```
algorithm PPO:
    initialize actor-critic network pi_theta, V_theta
    for iteration = 1, 2, ... do
        collect n_steps * n_envs transitions using pi_theta_old
        compute GAE advantages A_t using Rust data plane
        normalize advantages (if enabled)

        for epoch = 1 to n_epochs do
            for minibatch in shuffle(rollout, batch_size) do
                r_t = pi_theta(a|s) / pi_old(a|s)
                L_clip = min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)
                L_vf = (V_theta(s) - G_t)^2
                L_ent = -H[pi_theta]
                loss = -L_clip + vf_coef * L_vf + ent_coef * L_ent
                update theta with Adam, clip gradients to max_grad_norm
```

## Quick Start

```python
from rlox import Trainer

trainer = Trainer("ppo", env="CartPole-v1", seed=42)
metrics = trainer.train(total_timesteps=100_000)
print(f"Mean reward: {metrics['mean_reward']:.1f}")
```

For continuous control (MuJoCo):

```python
trainer = Trainer("ppo", env="HalfCheetah-v4", seed=42, config={
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "n_epochs": 10,
    "batch_size": 64,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "normalize_obs": True,
    "normalize_rewards": True,
})
metrics = trainer.train(total_timesteps=1_000_000)
```

## Hyperparameters

All defaults from `PPOConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_envs` | `8` | Number of parallel environments |
| `n_steps` | `128` | Rollout length per environment per update |
| `n_epochs` | `4` | SGD passes over each rollout |
| `batch_size` | `256` | Minibatch size for SGD |
| `learning_rate` | `2.5e-4` | Adam learning rate |
| `clip_eps` | `0.2` | PPO clipping range for probability ratio |
| `vf_coef` | `0.5` | Value loss coefficient |
| `ent_coef` | `0.01` | Entropy bonus coefficient |
| `max_grad_norm` | `0.5` | Maximum gradient norm for clipping |
| `gamma` | `0.99` | Discount factor |
| `gae_lambda` | `0.95` | GAE lambda |
| `normalize_advantages` | `True` | Normalize advantages per minibatch |
| `clip_vloss` | `True` | Clip value function loss |
| `anneal_lr` | `True` | Linearly anneal learning rate |
| `normalize_rewards` | `False` | Running reward normalization |
| `normalize_obs` | `False` | Running observation normalization |

## When to Use

- **Use PPO when:** you want a reliable, general-purpose algorithm that works across discrete and continuous action spaces with minimal tuning.
- **Do not use PPO when:** sample efficiency is critical (prefer [SAC](sac.md) or [TD3](td3.md) for continuous control) or you need hard trust-region guarantees (prefer [TRPO](trpo.md)).

## References

- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347).
- Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2015). [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438).
