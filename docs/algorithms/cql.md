# CQL -- Conservative Q-Learning

## Intuition

CQL tackles the fundamental challenge of offline RL: Q-value overestimation on out-of-distribution actions. Without online interaction to correct errors, standard Q-learning can assign arbitrarily high values to state-action pairs never seen in the dataset. CQL adds a regularizer that pushes down Q-values for actions sampled from the current policy (which may be out-of-distribution) while pushing up Q-values for actions in the dataset. The result is a conservative lower bound on the true Q-function that prevents the policy from exploiting overestimated values.

## Key Equations

The CQL regularizer augments the standard Bellman loss:

$$
L_{\text{CQL}} = \alpha \left( \mathbb{E}_{s \sim \mathcal{D}} \left[ \log \sum_a \exp Q(s,a) \right] - \mathbb{E}_{(s,a) \sim \mathcal{D}} \left[ Q(s,a) \right] \right)
$$

This encourages $Q(s,a)$ to be high for dataset actions and low for out-of-distribution actions.

The full critic loss:

$$
L_{\text{critic}} = L_{\text{Bellman}} + \alpha \cdot L_{\text{CQL}}
$$

The penalty is estimated using random actions and policy-sampled actions:

$$
\log \sum_a \exp Q(s,a) \approx \log \frac{1}{2N} \left( \sum_{a_i \sim \text{Uniform}} \exp Q(s, a_i) + \sum_{a_j \sim \pi} \exp Q(s, a_j) \right)
$$

Optionally, $\alpha$ can be auto-tuned via a Lagrangian:

$$
\alpha^* = \arg\min_\alpha \; \alpha \left(L_{\text{CQL}} - \tau_{\text{target}}\right)
$$

## Pseudocode

```
algorithm CQL:
    initialize SAC-style actor pi, twin critics Q1, Q2, targets
    load offline dataset D

    for update = 1 to n_updates do
        sample minibatch from D

        # Critic update
        L_bellman = standard SAC Bellman loss
        L_CQL = logsumexp(Q_random + Q_policy) - Q_data
        update critics with L_bellman + alpha * L_CQL

        # (Optional) auto-tune alpha
        alpha_loss = alpha * (target_value - L_CQL)
        update alpha

        # Actor update (SAC-style)
        update pi to maximize Q - alpha_ent * log pi

        # Entropy alpha update (SAC-style)
        # Soft target update
```

## Quick Start

CQL uses the offline algorithm interface with a pre-loaded dataset:

```python
from rlox.offline import OfflineDatasetBuffer
from rlox.algorithms.cql import CQL

dataset = OfflineDatasetBuffer.from_d4rl("halfcheetah-medium-v2")
agent = CQL(
    dataset=dataset,
    obs_dim=17,
    act_dim=6,
    cql_alpha=5.0,
    batch_size=256,
)
metrics = agent.train(n_updates=100_000)
```

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cql_alpha` | `5.0` | CQL penalty weight |
| `n_random_actions` | `10` | Random actions for CQL penalty estimation |
| `auto_alpha` | `False` | Auto-tune CQL alpha via Lagrangian |
| `cql_target_value` | `-1.0` | Target value for Lagrangian alpha tuning |
| `hidden` | `256` | Hidden layer width |
| `learning_rate` | `3e-4` | Learning rate |
| `tau` | `0.005` | Soft target update rate |
| `gamma` | `0.99` | Discount factor |
| `batch_size` | `256` | Minibatch size |
| `auto_entropy` | `True` | Auto-tune SAC entropy alpha |
| `target_entropy` | `-act_dim` | Target entropy for SAC |

## When to Use

- **Use CQL when:** you have a fixed offline dataset and need a principled conservative estimate of Q-values to avoid overestimation.
- **Prefer CQL over BC when:** the offline data is sub-optimal and you need to improve beyond the behavior policy.
- **Do not use CQL when:** the conservatism is too aggressive for your data distribution (try [Cal-QL](calql.md) for adaptive conservatism), or you have expert-quality data (prefer [BC](bc.md) or [IQL](iql.md)).

## References

- Kumar, A., Zhou, A., Tucker, G., & Levine, S. (2020). [Conservative Q-Learning for Offline Reinforcement Learning](https://arxiv.org/abs/2006.04779). *NeurIPS 2020*. [arXiv:2006.04779](https://arxiv.org/abs/2006.04779).
