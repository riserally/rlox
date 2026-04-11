# TD3 -- Twin Delayed DDPG

## Intuition

TD3 extends DDPG with three key techniques to address overestimation bias and training instability: (1) twin Q-networks where the minimum is used for targets, (2) delayed policy updates so the critic stabilizes before the actor adapts, and (3) target policy smoothing that adds clipped noise to target actions. The result is a robust deterministic policy gradient algorithm for continuous control.

## Key Equations

Twin Q-network target (take the minimum to combat overestimation):

$$
y = r + \gamma (1 - d) \min_{i=1,2} Q_{\phi_i'}(s', \tilde{a}')
$$

Target policy smoothing (regularize the target):

$$
\tilde{a}' = \pi_{\theta'}(s') + \text{clip}(\epsilon, -c, c), \quad \epsilon \sim \mathcal{N}(0, \sigma^2)
$$

Critic loss:

$$
L(\phi_i) = \mathbb{E} \left[ (Q_{\phi_i}(s, a) - y)^2 \right]
$$

Deterministic policy gradient (delayed, every $d$ critic updates):

$$
\nabla_\theta J(\theta) = \mathbb{E}_s \left[ \nabla_a Q_{\phi_1}(s, a) \big|_{a=\pi_\theta(s)} \nabla_\theta \pi_\theta(s) \right]
$$

## Pseudocode

```
algorithm TD3:
    initialize actor pi_theta, twin critics Q_phi1, Q_phi2
    initialize target networks pi_theta', Q_phi1', Q_phi2'
    initialize replay buffer D

    for step = 1, 2, ... do
        if step < learning_starts:
            a ~ Uniform(action_space)
        else:
            a = pi_theta(s) + N(0, exploration_noise)
            clip a to action bounds

        store (s, a, r, s', done) in D

        if step >= learning_starts:
            sample minibatch from D

            # Target with smoothing
            a' = pi_theta'(s') + clip(N(0, target_noise), -noise_clip, noise_clip)
            clip a' to action bounds
            y = r + gamma * (1-done) * min(Q_phi1'(s',a'), Q_phi2'(s',a'))

            # Critic update
            update phi1, phi2 to minimize (Q_phi_i(s,a) - y)^2

            # Delayed actor update
            if step % policy_delay == 0:
                update theta to maximize Q_phi1(s, pi_theta(s))
                soft update: theta' <- tau*theta + (1-tau)*theta'
                soft update: phi_i' <- tau*phi_i + (1-tau)*phi_i'
```

## Quick Start

```python
from rlox import Trainer

trainer = Trainer("td3", env="Pendulum-v1", seed=42)
metrics = trainer.train(total_timesteps=50_000)
```

For MuJoCo locomotion:

```python
trainer = Trainer("td3", env="HalfCheetah-v4", seed=42, config={
    "learning_rate": 3e-4,
    "buffer_size": 1_000_000,
    "learning_starts": 10_000,
    "batch_size": 256,
    "policy_delay": 2,
    "target_noise": 0.2,
    "noise_clip": 0.5,
    "exploration_noise": 0.1,
    "train_freq": 1,
    "gradient_steps": 1,
})
metrics = trainer.train(total_timesteps=1_000_000)
```

## Hyperparameters

All defaults from `TD3Config`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | `3e-4` | Adam learning rate for actor and critic |
| `buffer_size` | `1_000_000` | Replay buffer capacity |
| `batch_size` | `256` | Minibatch size |
| `tau` | `0.005` | Polyak averaging coefficient |
| `gamma` | `0.99` | Discount factor |
| `learning_starts` | `1000` | Random exploration steps before training |
| `policy_delay` | `2` | Actor update frequency relative to critic |
| `target_noise` | `0.2` | Noise std added to target actions |
| `noise_clip` | `0.5` | Clipping range for target noise |
| `exploration_noise` | `0.1` | Std of Gaussian exploration noise |
| `hidden` | `256` | Hidden layer width |
| `n_envs` | `1` | Number of parallel environments |
| `train_freq` | `1` | Environment steps between gradient updates |
| `gradient_steps` | `1` | Number of gradient steps per update |
| `target_policy_noise` | `None` | Alias for `target_noise` (SB3 preset compat) |
| `target_noise_clip` | `None` | Alias for `noise_clip` (SB3 preset compat) |

!!! note "SB3 preset compatibility"
    `target_policy_noise` and `target_noise_clip` are accepted as aliases for `target_noise` and `noise_clip` respectively, so that SB3/rl-zoo3 YAML presets work without key renaming. When both are provided, the `target_policy_noise` / `target_noise_clip` values take precedence.

## When to Use

- **Use TD3 when:** you have continuous actions, want deterministic policies, and need sample-efficient off-policy training.
- **Do not use TD3 when:** you want stochastic exploration (use [SAC](sac.md)), your actions are discrete (use [DQN](dqn.md)), or you prefer on-policy simplicity (use [PPO](ppo.md)).

## References

- Fujimoto, S., van Hoof, H., & Meger, D. (2018). [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477). *ICML 2018*.
- Lillicrap, T. P., Hunt, J. J., Pritzel, A., et al. (2015). [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971). *arXiv* (DDPG).
