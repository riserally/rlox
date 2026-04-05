# MAPPO -- Multi-Agent PPO

## Intuition

MAPPO applies PPO to multi-agent cooperative settings using the Centralized Training with Decentralized Execution (CTDE) paradigm. During training, each agent's critic has access to global state information (all agents' observations), enabling better value estimation. During execution, each agent's policy uses only its local observation. Despite its simplicity, MAPPO is competitive with or superior to more complex multi-agent algorithms across many cooperative benchmarks.

## Key Equations

Each agent $i$ has its own policy $\pi_{\theta_i}$ and a centralized value function $V_\phi(s)$ where $s$ is the global state.

PPO clipped objective per agent:

$$
L_i^{CLIP}(\theta_i) = \mathbb{E}_t \left[ \min \left( r_t^i \hat{A}_t^i, \; \text{clip}(r_t^i, 1-\epsilon, 1+\epsilon) \hat{A}_t^i \right) \right]
$$

where $r_t^i = \frac{\pi_{\theta_i}(a_t^i | o_t^i)}{\pi_{\theta_i^{\text{old}}}(a_t^i | o_t^i)}$.

Centralized advantage estimation using global state:

$$
\hat{A}_t^i = \sum_{l=0}^{T-t} (\gamma \lambda)^l \delta_{t+l}^i, \quad \delta_t^i = R_t^i + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)
$$

where $R_t^i$ denotes the per-agent reward (not to be confused with the policy ratio $r_t^i$ above).

With parameter sharing (optional), all agents share a single policy network $\pi_\theta$ conditioned on agent ID.

## Pseudocode

```
algorithm MAPPO:
    initialize per-agent policies pi_theta_i (or shared pi_theta)
    initialize centralized critic V_phi (takes global state)

    for iteration = 1, 2, ... do
        for env = 1 to n_envs do
            collect n_steps of joint transitions:
                each agent i acts: a_i ~ pi_theta_i(.|o_i)
                environment returns: (o', r, done) for all agents

        for agent i = 1 to n_agents do
            compute GAE advantages A_t^i using centralized V_phi(s)
            normalize advantages

            for epoch = 1 to n_epochs do
                for minibatch in shuffle(rollout) do
                    r_t^i = pi_theta_i(a_i|o_i) / pi_old_i(a_i|o_i)
                    L_clip = min(r_t^i * A_t^i, clip(r_t^i, 1-eps, 1+eps) * A_t^i)
                    L_vf = (V_phi(s) - G_t)^2
                    L_ent = -H[pi_theta_i]
                    loss = -L_clip + vf_coef * L_vf + ent_coef * L_ent
                    update theta_i (and phi) with Adam
```

## Quick Start

```python
from rlox import Trainer

trainer = Trainer("mappo", env="simple_spread_v3", seed=42, config={
    "n_agents": 3,
})
metrics = trainer.train(total_timesteps=500_000)
```

With parameter sharing:

```python
trainer = Trainer("mappo", env="simple_spread_v3", seed=42, config={
    "n_agents": 3,
    "share_parameters": True,
    "learning_rate": 5e-4,
    "n_steps": 128,
    "n_epochs": 5,
    "clip_range": 0.2,
    "max_grad_norm": 10.0,
})
metrics = trainer.train(total_timesteps=2_000_000)
```

## Hyperparameters

All defaults from `MAPPOConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_agents` | `2` | Number of agents |
| `learning_rate` | `5e-4` | Adam learning rate |
| `n_steps` | `128` | Rollout length per environment per update |
| `n_epochs` | `5` | SGD passes per rollout |
| `clip_range` | `0.2` | PPO clipping range |
| `gamma` | `0.99` | Discount factor |
| `gae_lambda` | `0.95` | GAE lambda |
| `vf_coef` | `0.5` | Value loss coefficient |
| `ent_coef` | `0.01` | Entropy bonus coefficient |
| `max_grad_norm` | `10.0` | Maximum gradient norm |
| `share_parameters` | `False` | Whether agents share policy parameters |
| `hidden` | `64` | Hidden layer width |
| `n_envs` | `8` | Number of parallel environments |

## When to Use

- **Use MAPPO when:** you have a cooperative multi-agent task and want a simple, effective algorithm with CTDE.
- **Do not use MAPPO when:** agents are fully competitive (consider self-play), you need value decomposition (use QMIX), or the task is single-agent (use [PPO](ppo.md)).

## References

- Yu, C., Velu, A., Vinitsky, E., et al. (2022). [The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games](https://arxiv.org/abs/2103.01955). *NeurIPS 2022*.
- de Witt, C. S., Gupta, T., Makoviichuk, D., et al. (2020). Is Independent Learning All You Need in the StarCraft Multi-Agent Challenge? [arXiv:2011.09533](https://arxiv.org/abs/2011.09533).
