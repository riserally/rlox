# Diffusion Policy

## Intuition

Diffusion Policy generates actions by learning to reverse a diffusion (noising) process. During training, clean action sequences from demonstrations are progressively corrupted with Gaussian noise, and a denoising network learns to reverse each step. At inference, the model starts from pure noise and iteratively refines it into a coherent action trajectory, conditioned on recent observations. This approach can represent highly multimodal action distributions (unlike Gaussian policies), making it especially powerful for robotic manipulation tasks with multiple valid solutions.

## Key Equations

**Forward diffusion** (add noise to clean actions):

$$
x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

where $\bar{\alpha}_t = \prod_{i=1}^{t} \alpha_i$ is the cumulative product of the noise schedule.

**Training objective** (DDPM loss):

$$
L(\theta) = \mathbb{E}_{x_0, t, \epsilon} \left[ \left\| \epsilon - \epsilon_\theta(x_t, t, o) \right\|^2 \right]
$$

where $\epsilon_\theta$ is the denoising network conditioned on observation history $o$.

**Reverse diffusion** (DDPM sampling step):

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t, o) \right) + \sqrt{\beta_t}\, z
$$

where $z \sim \mathcal{N}(0, I)$ for $t > 0$ and $z = 0$ for $t = 0$.

## Pseudocode

```
algorithm Diffusion Policy:
    initialize denoising network eps_theta
    precompute noise schedule {alpha_t, beta_t, alpha_bar_t}
    collect offline dataset D = {(obs_window, action_window)}

    # Training
    for update = 1 to n_updates do
        sample (obs, actions) from D
        sample t ~ Uniform(0, T-1)
        sample noise eps ~ N(0, I)

        x_t = sqrt(alpha_bar_t) * actions + sqrt(1 - alpha_bar_t) * eps
        eps_pred = eps_theta(x_t, t, obs)
        loss = MSE(eps_pred, eps)
        update theta

    # Inference
    x_T ~ N(0, I)  with shape (action_horizon, act_dim)
    for t = T-1, T-2, ..., 0 do
        eps_pred = eps_theta(x_t, t, obs)
        x_{t-1} = DDPM_step(x_t, eps_pred, t)
    return x_0[0]  # first action in the horizon
```

## Quick Start

```python
from rlox import Trainer

trainer = Trainer("diffusion", env="Pendulum-v1", seed=42, config={
    "n_diffusion_steps": 50,
    "action_horizon": 8,
    "obs_horizon": 2,
    "noise_schedule": "cosine",
})
metrics = trainer.train(total_timesteps=50_000)
print(f"Loss: {metrics['loss']:.4f}")
```

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_diffusion_steps` | `50` | Number of diffusion timesteps $T$ |
| `action_horizon` | `8` | Number of future actions to predict |
| `obs_horizon` | `2` | Number of past observations to condition on |
| `hidden_dim` | `256` | Hidden layer width for denoising network |
| `learning_rate` | `1e-4` | Adam learning rate |
| `batch_size` | `256` | Minibatch size |
| `noise_schedule` | `"cosine"` | `"cosine"` or `"linear"` |
| `beta_start` | `0.0001` | Starting beta for linear schedule |
| `beta_end` | `0.02` | Ending beta for linear schedule |
| `buffer_size` | `1_000_000` | Replay buffer capacity |
| `n_inference_steps` | `10` | Denoising steps at inference (can be < T for speed) |
| `warmup_steps` | `500` | Data collection steps before training |
| `seed` | `42` | Random seed |

## When to Use

- **Use Diffusion Policy when:** your task has multimodal action distributions (e.g., robotic manipulation with multiple grasp strategies), or you have high-quality demonstration data.
- **Prefer Diffusion Policy over BC when:** the action distribution is multimodal and a single Gaussian cannot capture it.
- **Do not use Diffusion Policy when:** inference speed is critical (diffusion requires multiple denoising steps), or you need online RL without demonstrations (prefer [SAC](sac.md) or [PPO](ppo.md)).

## References

- Chi, C., Feng, S., Du, Y., Xu, Z., Cousineau, E., Burchfiel, B., & Song, S. (2023). Diffusion Policy: Visuomotor Policy Learning via Action Diffusion. *RSS 2023*. [arXiv:2303.04137](https://arxiv.org/abs/2303.04137).
- Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. *NeurIPS 2020*. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239).
