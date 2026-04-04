# TRPO -- Trust Region Policy Optimization

## Intuition

TRPO directly constrains the KL divergence between successive policies, guaranteeing monotonic improvement (in theory). It solves a constrained optimization problem using conjugate gradients and a backtracking line search, avoiding the need to choose a clipping parameter. TRPO is more principled than PPO but significantly more expensive per update due to the second-order optimization.

## Key Equations

The constrained optimization problem:

$$
\max_\theta \; \mathbb{E}_t \left[ \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_\text{old}}(a_t | s_t)} \hat{A}_t \right]
$$

$$
\text{subject to} \quad \mathbb{E}_t \left[ D_\text{KL}(\pi_{\theta_\text{old}}(\cdot | s_t) \| \pi_\theta(\cdot | s_t)) \right] \leq \delta
$$

This is solved approximately via the natural gradient:

$$
\Delta\theta = \sqrt{\frac{2\delta}{\mathbf{g}^T \mathbf{F}^{-1} \mathbf{g}}} \, \mathbf{F}^{-1} \mathbf{g}
$$

where $\mathbf{g}$ is the policy gradient and $\mathbf{F}$ is the Fisher information matrix. The product $\mathbf{F}^{-1} \mathbf{g}$ is computed via conjugate gradients without forming $\mathbf{F}$ explicitly.

## Pseudocode

```
algorithm TRPO:
    initialize actor pi_theta, critic V_phi

    for iteration = 1, 2, ... do
        collect n_steps * n_envs transitions using pi_theta_old
        compute GAE advantages A_t

        # Natural gradient via conjugate gradients
        g = gradient of surrogate objective w.r.t. theta
        s = CG(F, g, cg_iters)   # solve Fs = g approximately
        step_size = sqrt(2 * max_kl / (s^T F s))

        # Backtracking line search
        for j = 0, 1, ..., line_search_steps do
            theta_new = theta_old + (0.5^j) * step_size * s
            if surrogate improves AND KL(theta_old, theta_new) <= max_kl:
                theta <- theta_new
                break

        # Value function update (standard SGD)
        for epoch = 1 to vf_epochs do
            update phi to minimize (V_phi(s) - G_t)^2
```

## Quick Start

```python
from rlox import Trainer

trainer = Trainer("trpo", env="CartPole-v1", seed=42)
metrics = trainer.train(total_timesteps=100_000)
```

For continuous control:

```python
trainer = Trainer("trpo", env="HalfCheetah-v4", seed=42, config={
    "max_kl": 0.01,
    "n_steps": 2048,
    "gamma": 0.99,
    "gae_lambda": 0.97,
    "cg_iters": 10,
    "vf_lr": 1e-3,
    "vf_epochs": 5,
})
metrics = trainer.train(total_timesteps=1_000_000)
```

## Hyperparameters

All defaults from `TRPOConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_kl` | `0.01` | Maximum KL divergence per update |
| `damping` | `0.1` | Damping for Fisher vector product |
| `cg_iters` | `10` | Conjugate gradient iterations |
| `line_search_steps` | `10` | Backtracking line search steps |
| `n_envs` | `8` | Number of parallel environments |
| `n_steps` | `2048` | Rollout length per environment per update |
| `gamma` | `0.99` | Discount factor |
| `gae_lambda` | `0.97` | GAE lambda |
| `vf_lr` | `1e-3` | Value function learning rate |
| `vf_epochs` | `5` | Value function SGD epochs per update |

## When to Use

- **Use TRPO when:** you need formal trust-region guarantees on policy updates, or when PPO's clipping is insufficient for your problem's sensitivity.
- **Do not use TRPO when:** wall-clock time matters (PPO is faster), or you need off-policy sample efficiency (use [SAC](sac.md)).

## References

- Schulman, J., Levine, S., Abbeel, P., Jordan, M., & Moritz, P. (2015). Trust Region Policy Optimization. *ICML 2015*.
