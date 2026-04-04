# AWR -- Advantage Weighted Regression

## Intuition

AWR sidesteps the instabilities of importance sampling by directly weighting policy log-probabilities with exponentiated advantages. Instead of computing probability ratios between old and new policies (as PPO does), AWR fits the actor via weighted maximum likelihood: actions that turned out to be much better than the baseline receive exponentially higher weight. This makes AWR particularly simple to implement and suitable for both online and offline settings.

## Key Equations

The actor loss uses advantage-weighted regression:

$$
L_{\text{actor}}(\theta) = -\mathbb{E}_{(s,a) \sim \mathcal{D}} \left[ \exp\!\left(\frac{A(s,a)}{\beta}\right) \log \pi_\theta(a|s) \right]
$$

where $\beta$ is a temperature parameter controlling the sharpness of the weighting and $A(s,a) = r + \gamma V(s') - V(s)$ is the TD advantage.

The critic is trained with standard TD regression:

$$
L_{\text{critic}}(\phi) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ \left( V_\phi(s) - \left( r + \gamma V_{\bar{\phi}}(s') \right) \right)^2 \right]
$$

Exponentiated advantages are clamped to prevent overflow:

$$
w(s,a) = \min\!\left( \exp\!\left(\frac{A(s,a)}{\beta}\right),\; w_{\max} \right)
$$

## Pseudocode

```
algorithm AWR:
    initialize actor pi_theta, critic V_phi
    initialize replay buffer D with capacity buffer_size

    for step = 1 to total_timesteps do
        if step < learning_starts then
            a = random action
        else
            a = sample from pi_theta(.|s)

        execute a, observe r, s', done
        store (s, a, r, done, s') in D

        if step >= learning_starts and |D| >= batch_size then
            sample minibatch {(s, a, r, done, s')} from D

            # Critic update
            targets = r + gamma * (1 - done) * V_phi(s')
            L_critic = MSE(V_phi(s), targets)
            update phi

            # Actor update (AWR)
            A = targets - V_phi(s)
            w = clamp(exp(A / beta), max=max_advantage)
            L_actor = -mean(w * log pi_theta(a|s))
            update theta
```

## Quick Start

```python
from rlox import Trainer

trainer = Trainer("awr", env="CartPole-v1", seed=42)
metrics = trainer.train(total_timesteps=50_000)
print(f"Mean reward: {metrics['mean_reward']:.1f}")
```

For continuous control:

```python
trainer = Trainer("awr", env="Pendulum-v1", seed=42, config={
    "beta": 0.5,
    "learning_rate": 3e-4,
    "batch_size": 256,
    "buffer_size": 100_000,
    "gamma": 0.99,
})
metrics = trainer.train(total_timesteps=100_000)
```

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `beta` | `1.0` | Temperature for advantage weighting (lower = sharper) |
| `learning_rate` | `3e-4` | Adam learning rate for actor and critic |
| `gamma` | `0.99` | Discount factor |
| `batch_size` | `256` | Minibatch size for SGD |
| `buffer_size` | `100_000` | Replay buffer capacity |
| `hidden` | `256` | Hidden layer width |
| `learning_starts` | `1000` | Random exploration steps before training |
| `n_critic_updates` | `1` | Number of critic updates per training step |
| `max_advantage` | `20.0` | Clamp for exponentiated advantage (prevents overflow) |
| `seed` | `42` | Random seed |

## When to Use

- **Use AWR when:** you want a simple off-policy algorithm that avoids importance sampling, or you have offline data and need a straightforward baseline.
- **Prefer AWR over PPO when:** you want to combine online and offline data, or importance sampling ratios are unstable.
- **Do not use AWR when:** you need maximum sample efficiency on continuous control (prefer [SAC](sac.md) or [TD3](td3.md)), or the task requires careful entropy tuning.

## References

- Peng, X. B., Kumar, A., Zhang, G., & Levine, S. (2019). Advantage-Weighted Regression: Simple and Scalable Off-Policy Reinforcement Learning. *arXiv:1910.00177*.
