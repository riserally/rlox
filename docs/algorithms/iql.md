# IQL -- Implicit Q-Learning

## Intuition

IQL avoids the main pitfall of offline RL -- querying Q-values for out-of-distribution actions -- by never evaluating Q on actions outside the dataset. Instead of using $\max_a Q(s,a)$ as the target (which requires querying unseen actions), IQL fits a separate value function $V(s)$ using expectile regression on Q-values of dataset actions. A high expectile ($\tau > 0.5$) biases $V$ toward the upper quantiles of $Q(s,a)$ without ever computing Q for actions not in the data. This makes IQL remarkably simple and avoids the need for explicit conservatism penalties.

## Key Equations

**Value function** via expectile regression:

$$
L_V(\psi) = \mathbb{E}_{(s,a) \sim \mathcal{D}} \left[ L_\tau^2\!\left(Q_{\bar\phi}(s,a) - V_\psi(s)\right) \right]
$$

where the asymmetric loss is:

$$
L_\tau^2(u) = |\tau - \mathbf{1}(u < 0)| \cdot u^2
$$

**Q-function** with V as the bootstrap target (no max over actions):

$$
L_Q(\phi) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ \left(Q_\phi(s,a) - r - \gamma V_\psi(s')\right)^2 \right]
$$

**Actor extraction** via advantage-weighted regression:

$$
L_\pi(\theta) = \mathbb{E}_{(s,a) \sim \mathcal{D}} \left[ \exp\!\left(\beta \cdot (Q_{\bar\phi}(s,a) - V_\psi(s))\right) \left\| \pi_\theta(s) - a \right\|^2 \right]
$$

## Pseudocode

```
algorithm IQL:
    initialize Q1, Q2 (with targets), V, actor pi
    load offline dataset D

    for update = 1 to n_updates do
        sample minibatch from D

        # Value update (expectile regression)
        q_target = min(Q1_target(s,a), Q2_target(s,a))
        L_V = expectile_loss(V(s), q_target, tau)
        update V

        # Q-function update (Bellman with V target)
        target = r + gamma * (1 - done) * V(s')
        L_Q = MSE(Q1(s,a), target) + MSE(Q2(s,a), target)
        update Q1, Q2

        # Soft target update for Q
        polyak_update(Q1_target, Q2_target)

        # Actor update (advantage-weighted regression)
        advantage = q_target - V(s)
        weights = clamp(exp(beta * advantage), max=100)
        L_pi = mean(weights * MSE(pi(s), a))
        update pi
```

## Quick Start

IQL uses the offline algorithm interface:

```python
from rlox.offline import OfflineDatasetBuffer
from rlox.algorithms.iql import IQL

dataset = OfflineDatasetBuffer.from_d4rl("halfcheetah-medium-v2")
agent = IQL(
    dataset=dataset,
    obs_dim=17,
    act_dim=6,
    expectile=0.7,
    temperature=3.0,
)
metrics = agent.train(n_updates=100_000)
```

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `expectile` | `0.7` | Expectile $\tau$ for value function regression (higher = more optimistic) |
| `temperature` | `3.0` | Inverse temperature $\beta$ for advantage-weighted actor extraction (higher = sharper weighting) |
| `hidden` | `256` | Hidden layer width |
| `learning_rate` | `3e-4` | Learning rate for all networks |
| `tau` | `0.005` | Soft target update rate |
| `gamma` | `0.99` | Discount factor |
| `batch_size` | `256` | Minibatch size |

## When to Use

- **Use IQL when:** you need a simple, stable offline RL algorithm that avoids querying out-of-distribution actions entirely.
- **Prefer IQL over CQL when:** CQL's conservatism is too aggressive, or you want fewer hyperparameters to tune.
- **Do not use IQL when:** you need to significantly improve beyond the dataset quality (CQL or Cal-QL may be more aggressive in stitching), or you need online fine-tuning (prefer [Cal-QL](calql.md)).

## References

- Kostrikov, I., Nair, A., & Levine, S. (2022). [Offline Reinforcement Learning with Implicit Q-Learning](https://arxiv.org/abs/2110.06169). *ICLR 2022*. [arXiv:2110.06169](https://arxiv.org/abs/2110.06169).
