# Cal-QL -- Calibrated Conservative Q-Learning

## Intuition

Cal-QL addresses a key limitation of CQL: vanilla CQL applies the same conservative penalty regardless of how close an action is to the training distribution, leading to overly pessimistic Q-values for near-distribution actions. Cal-QL introduces a calibration mechanism that scales the conservative penalty based on the gap between current Q-values and an empirical threshold derived from offline returns. When Q-values are already well-calibrated (close to actual returns), the penalty is reduced. This makes Cal-QL particularly effective for offline pre-training followed by online fine-tuning.

## Key Equations

The standard CQL penalty pushes down Q-values for out-of-distribution actions:

$$
L_{\text{CQL}} = \alpha \left( \log \sum_a \exp Q(s,a) - \mathbb{E}_{a \sim \mathcal{D}}[Q(s,a)] \right)
$$

Cal-QL adds a calibrated scaling factor:

$$
L_{\text{cal}} = \max\!\left(Q(s,a) - Q_{\text{cal}},\; 0\right) \cdot \tau_{\text{cal}} \cdot \left(Q(s,a) - Q_{\text{data}}(s,a)\right)
$$

where $Q_{\text{cal}}$ is the $\tau$-quantile of observed episode returns.

The full critic loss combines Bellman error, CQL penalty, and calibrated penalty:

$$
L_{\text{critic}} = L_{\text{Bellman}} + \alpha \left(L_{\text{CQL}} + L_{\text{cal}}\right)
$$

The actor follows SAC-style entropy-regularized policy optimization:

$$
L_{\text{actor}} = \mathbb{E}_{a \sim \pi} \left[ \alpha_{\text{ent}} \log \pi(a|s) - \min(Q_1(s,a), Q_2(s,a)) \right]
$$

## Pseudocode

```
algorithm Cal-QL:
    initialize SAC-style actor pi, twin critics Q1, Q2, target networks
    initialize calibration threshold Q_cal = 0
    initialize return buffer R

    for step = 1 to total_timesteps do
        collect transition (s, a, r, s', done)
        if episode ends: append episode_return to R
                         Q_cal = quantile(R, calibration_tau)

        sample minibatch from replay buffer

        # Critic update
        L_bellman = standard SAC Bellman loss with twin critics
        L_CQL = logsumexp(Q_random + Q_policy) - Q_data
        L_cal = max(Q - Q_cal, 0) * calibration_tau * (Q - Q_data)
        update critics with L_bellman + cql_alpha * (L_CQL + L_cal)

        # Actor update (SAC-style)
        update actor to maximize Q - alpha_ent * log pi

        # Soft target update
        polyak_update(Q_targets, tau)
```

## Quick Start

```python
from rlox import Trainer

trainer = Trainer("calql", env="HalfCheetah-v4", seed=42, config={
    "cql_alpha": 5.0,
    "calibration_tau": 0.5,
    "learning_rate": 3e-4,
})
metrics = trainer.train(total_timesteps=500_000)
print(f"Mean reward: {metrics['mean_reward']:.1f}")
```

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | `3e-4` | Learning rate for actor, critic, and alpha optimizers |
| `buffer_size` | `100_000` | Replay buffer capacity |
| `batch_size` | `256` | Minibatch size |
| `gamma` | `0.99` | Discount factor |
| `tau` | `0.005` | Polyak averaging coefficient for target networks |
| `cql_alpha` | `5.0` | CQL penalty weight |
| `calibration_tau` | `0.5` | Quantile for calibration threshold from offline returns |
| `auto_alpha` | `False` | Auto-tune CQL alpha via dual gradient descent |
| `hidden` | `256` | Hidden layer width |
| `n_random_actions` | `10` | Random actions for CQL penalty estimation |
| `warmup_steps` | `1000` | Random exploration steps before training |
| `seed` | `42` | Random seed |

## When to Use

- **Use Cal-QL when:** you want offline pre-training with online fine-tuning, or vanilla CQL is too pessimistic on your dataset.
- **Prefer Cal-QL over CQL when:** your offline data quality varies and you need adaptive conservatism.
- **Do not use Cal-QL when:** you have high-quality expert data (prefer [BC](bc.md) or [IQL](iql.md) for simplicity), or you are doing purely online training (prefer [SAC](sac.md)).

## References

- Nakamoto, N., Zhai, S., Singh, A., Mark, M. S., Ma, Y., Finn, C., Kumar, A., & Levine, S. (2023). Cal-QL: Calibrated Offline RL Pre-Training for Efficient Online Fine-Tuning. *NeurIPS 2023*. [arXiv:2303.05479](https://arxiv.org/abs/2303.05479).
