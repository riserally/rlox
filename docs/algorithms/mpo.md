# MPO -- Maximum a Posteriori Policy Optimization

## Intuition

MPO decouples the policy improvement step into two phases inspired by Expectation-Maximization. In the E-step, it constructs a non-parametric improved policy by weighting actions according to their Q-values (softmax over Q). In the M-step, it fits the parametric policy to match this improved distribution via KL-constrained supervised learning. Dual variables automatically tune the temperature (how greedy the E-step is) and the KL constraint (how far the policy can move per update). This decomposition gives MPO strong stability guarantees while remaining fully off-policy with a replay buffer.

## Key Equations

**E-step:** Compute non-parametric action weights:

$$
q(a|s) \propto \exp\!\left(\frac{Q(s,a)}{\eta}\right)
$$

where $\eta$ is the temperature dual variable.

**M-step:** Fit the parametric policy to minimize:

$$
L_{\text{actor}}(\theta) = \mathbb{E}_s \left[ \text{KL}\!\left(q(\cdot|s) \| \pi_\theta(\cdot|s)\right) \right] = -\mathbb{E}_s \left[ \sum_a q(a|s) \log \pi_\theta(a|s) \right]
$$

**Dual variable update:** The temperature $\eta$ is optimized via:

$$
\min_\eta \; \eta \epsilon + \eta \log \frac{1}{N} \sum_{i=1}^{N} \exp\!\left(\frac{Q(s, a_i)}{\eta}\right)
$$

where $\epsilon$ is the target KL constraint.

**Critic update:** Standard clipped double Q-learning:

$$
L_{\text{critic}} = \mathbb{E}\!\left[\left(Q_1(s,a) - y\right)^2 + \left(Q_2(s,a) - y\right)^2\right], \quad y = r + \gamma (1 - d) \min(Q_1^-, Q_2^-)(s', a')
$$

## Pseudocode

```
algorithm MPO:
    initialize actor pi_theta (squashed Gaussian)
    initialize twin critics Q1, Q2 with target networks
    initialize dual variable eta (log-space)
    initialize replay buffer D

    for step = 1 to total_timesteps do
        if step < learning_starts: a = random
        else: a ~ pi_theta(.|s)

        execute a, observe r, s', done
        store (s, a, r, done, s') in D

        if step >= learning_starts then
            sample minibatch from D

            # 1. Critic update (clipped double Q)
            update Q1, Q2 with Bellman targets

            # 2. E-step: sample N actions, compute Q-values
            for each s in batch:
                sample {a_1, ..., a_N} ~ pi_theta(.|s)
                q_i = min(Q1(s, a_i), Q2(s, a_i))
                w_i = softmax(q / eta)

            # 3. Dual update: optimize eta
            minimize eta * eps + eta * logsumexp(Q / eta)

            # 4. M-step: fit policy
            L_actor = -sum(w_i * log pi_theta(a_i | s))
            update theta

            # 5. Soft target update
            polyak_update(Q_targets, tau)
```

## Quick Start

```python
from rlox import Trainer

trainer = Trainer("mpo", env="Pendulum-v1", seed=42, config={
    "learning_rate": 3e-4,
    "n_action_samples": 20,
    "epsilon": 0.1,
})
metrics = trainer.train(total_timesteps=100_000)
print(f"Mean reward: {metrics['mean_reward']:.1f}")
```

For continuous control with MuJoCo:

```python
trainer = Trainer("mpo", env="HalfCheetah-v4", seed=42, config={
    "learning_rate": 3e-4,
    "buffer_size": 1_000_000,
    "batch_size": 256,
    "n_action_samples": 20,
    "epsilon": 0.1,
    "dual_lr": 1e-2,
})
metrics = trainer.train(total_timesteps=1_000_000)
```

## Hyperparameters

All defaults from `MPOConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | `3e-4` | Adam learning rate for actor and critic |
| `buffer_size` | `1_000_000` | Replay buffer capacity |
| `batch_size` | `256` | Minibatch size |
| `gamma` | `0.99` | Discount factor |
| `tau` | `0.005` | Polyak averaging coefficient for target networks |
| `n_action_samples` | `20` | Number of action samples for the E-step |
| `epsilon` | `0.1` | KL constraint for the M-step |
| `epsilon_penalty` | `0.001` | KL penalty coefficient |
| `dual_lr` | `1e-2` | Learning rate for dual variables (temperature) |
| `hidden` | `256` | Hidden layer width |
| `learning_starts` | `1000` | Random exploration steps before training |

## When to Use

- **Use MPO when:** you want a principled off-policy algorithm with automatic temperature and KL tuning, especially for continuous control tasks.
- **Prefer MPO over SAC when:** you want explicit KL constraints rather than entropy bonuses, or need tighter control over policy update magnitude.
- **Do not use MPO when:** simplicity is paramount (prefer [SAC](sac.md) or [TD3](td3.md)), or you need discrete action support (MPO requires continuous actions).

## References

- Abdolmaleki, A., Springenberg, J. T., Tassa, Y., Munos, R., Heess, N., & Riedmiller, M. (2018). Maximum a Posteriori Policy Optimisation. *ICLR 2018*. *arXiv:1806.06920*.
