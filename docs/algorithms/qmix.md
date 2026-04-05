# QMIX -- Monotonic Value Function Factorisation

## Intuition

QMIX solves cooperative multi-agent tasks by decomposing the joint team Q-value into per-agent utility functions combined through a monotonic mixing network. Each agent learns its own Q-network from local observations, while a hypernetwork (conditioned on the global state) generates the mixing weights. The key insight is enforcing monotonicity -- the joint Q-value is monotonically increasing in each agent's utility -- which guarantees that greedy action selection on the joint Q-value equals independent greedy selection per agent (decentralized execution).

## Key Equations

Per-agent Q-values are combined via the mixing network:

$$
Q_{\text{tot}}(s, \mathbf{a}) = f_{\text{mix}}\!\left(Q_1(o_1, a_1), \ldots, Q_n(o_n, a_n); s\right)
$$

Monotonicity is enforced by constraining mixing weights to be non-negative:

$$
\frac{\partial Q_{\text{tot}}}{\partial Q_i} \geq 0, \quad \forall i
$$

This is achieved by applying $|\cdot|$ (absolute value) to hypernetwork outputs:

$$
W = |f_{\text{hyper}}(s)|
$$

The loss is standard TD error on the joint Q-value:

$$
L = \mathbb{E}\!\left[\left(r + \gamma \max_{\mathbf{a}'} Q_{\text{tot}}^-(s', \mathbf{a}') - Q_{\text{tot}}(s, \mathbf{a})\right)^2\right]
$$

## Pseudocode

```
algorithm QMIX:
    initialize per-agent Q-networks {Q_i} and target networks {Q_i^-}
    initialize mixing network M and target M^-
    initialize replay buffer D

    for step = 1 to total_timesteps do
        for each agent i:
            select a_i using epsilon-greedy on Q_i(o_i, .)

        execute joint action, observe r, s', done
        store (s, {o_i}, {a_i}, r, done, s', {o_i'}) in D

        sample minibatch from D
        for each agent i:
            q_i = Q_i(o_i, a_i)       # chosen action Q-value
            q_i_target = max_a Q_i^-(o_i', a)  # greedy target

        q_tot = M({q_i}, global_state)
        q_tot_target = M^-({q_i_target}, global_state')
        L = MSE(q_tot, r + gamma * (1 - done) * q_tot_target)

        update all Q_i and M parameters
        periodically hard-copy to target networks
```

## Quick Start

```python
from rlox import Trainer

trainer = Trainer("qmix", env="CartPole-v1", seed=42, config={
    "n_agents": 3,
    "hidden_dim": 64,
    "mixing_embed_dim": 32,
})
metrics = trainer.train(total_timesteps=50_000)
print(f"Mean reward: {metrics['mean_reward']:.1f}")
```

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_agents` | `3` | Number of cooperative agents |
| `hidden_dim` | `64` | Hidden dimension for per-agent Q-networks |
| `mixing_embed_dim` | `32` | Hidden dimension of the mixing network |
| `learning_rate` | `5e-4` | Adam learning rate |
| `buffer_size` | `50_000` | Replay buffer capacity |
| `batch_size` | `32` | Minibatch size |
| `gamma` | `0.99` | Discount factor |
| `target_update_freq` | `200` | Steps between hard target network updates |
| `epsilon_start` | `1.0` | Initial exploration epsilon |
| `epsilon_end` | `0.05` | Final exploration epsilon |
| `epsilon_decay_steps` | `5000` | Linear epsilon decay duration |
| `seed` | `42` | Random seed |

## When to Use

- **Use QMIX when:** you have a cooperative multi-agent task with discrete actions and shared rewards, and agents can only observe local information at execution time.
- **Prefer QMIX over MAPPO when:** you want value decomposition with replay buffer efficiency, or the task has a clear cooperative structure where monotonicity holds.
- **Do not use QMIX when:** the task requires non-monotonic value decomposition (some cooperative games violate this), or agents need continuous action spaces (prefer [MAPPO](mappo.md)).

## References

- Rashid, T., Samvelyan, M., Schroeder de Witt, C., Farquhar, G., Foerster, J., & Whiteson, S. (2018). QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning. *ICML 2018*. [arXiv:1803.11485](https://arxiv.org/abs/1803.11485).
