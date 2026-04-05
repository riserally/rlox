# DQN -- Deep Q-Network

## Intuition

DQN approximates the optimal action-value function $Q^*(s, a)$ with a neural network and selects actions greedily. Experience replay and a target network stabilize training. rlox's DQN includes Rainbow extensions: Double DQN (reduced overestimation), Dueling architecture (separate value and advantage streams), N-step returns, and Prioritized Experience Replay (PER).

## Key Equations

The Bellman optimality target:

$$
y_t = r_t + \gamma (1 - d_t) \max_{a'} Q_{\phi'}(s_{t+1}, a')
$$

**Double DQN** decouples action selection from evaluation to reduce overestimation:

$$
y_t = r_t + \gamma (1 - d_t) \, Q_{\phi'}(s_{t+1}, \arg\max_{a'} Q_\phi(s_{t+1}, a'))
$$

**N-step returns** extend the target horizon:

$$
y_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n (1 - d_{t+n}) \max_{a'} Q_{\phi'}(s_{t+n}, a')
$$

where $d_{t+n}$ is the terminal flag (1 if episode ended, 0 otherwise).

**Dueling architecture** decomposes Q into value and advantage:

$$
Q(s, a) = V(s) + A(s, a) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s, a')
$$

**Prioritized Experience Replay** samples proportional to TD error:

$$
p_i = |\delta_i| + \epsilon, \quad P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}
$$

## Pseudocode

```
algorithm DQN:
    initialize Q-network Q_phi, target network Q_phi'
    initialize replay buffer D (uniform or prioritized)

    for step = 1, 2, ... do
        with probability epsilon: a = random action
        else: a = argmax_a Q_phi(s, a)

        store (s, a, r, s', done) in D

        if step >= learning_starts:
            sample minibatch from D (with PER weights if enabled)

            if double_dqn:
                a* = argmax_a Q_phi(s', a)
                y = r + gamma^n * (1-done) * Q_phi'(s', a*)
            else:
                y = r + gamma^n * (1-done) * max_a Q_phi'(s', a)

            loss = mean(w_i * (Q_phi(s, a) - y)^2)
            update phi with Adam

            if prioritized: update priorities in D

        every target_update_freq steps:
            phi' <- phi

        decay epsilon from initial_eps to final_eps
```

## Quick Start

```python
from rlox import Trainer

trainer = Trainer("dqn", env="CartPole-v1", seed=42)
metrics = trainer.train(total_timesteps=100_000)
```

With Rainbow extensions:

```python
trainer = Trainer("dqn", env="LunarLander-v3", seed=42, config={
    "double_dqn": True,
    "dueling": True,
    "prioritized": True,
    "n_step": 3,
    "learning_rate": 6.3e-4,
    "buffer_size": 100_000,
    "batch_size": 128,
    "exploration_final_eps": 0.02,
})
metrics = trainer.train(total_timesteps=200_000)
```

## Hyperparameters

All defaults from `DQNConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | `1e-4` | Adam learning rate |
| `buffer_size` | `1_000_000` | Replay buffer capacity |
| `batch_size` | `64` | Minibatch size |
| `gamma` | `0.99` | Discount factor |
| `target_update_freq` | `1000` | Steps between hard target network updates |
| `exploration_fraction` | `0.1` | Fraction of training for epsilon decay |
| `exploration_initial_eps` | `1.0` | Starting epsilon |
| `exploration_final_eps` | `0.05` | Final epsilon after decay |
| `learning_starts` | `1000` | Random exploration steps before training |
| `double_dqn` | `True` | Use Double DQN action selection |
| `dueling` | `False` | Use Dueling network architecture |
| `n_step` | `1` | N-step return horizon |
| `prioritized` | `False` | Use Prioritized Experience Replay |
| `alpha` | `0.6` | PER priority exponent |
| `beta_start` | `0.4` | PER initial importance-sampling exponent |
| `hidden` | `256` | Hidden layer width |

## When to Use

- **Use DQN when:** your action space is discrete and you want sample-efficient off-policy training with replay.
- **Do not use DQN when:** your action space is continuous (use [SAC](sac.md) or [TD3](td3.md)) or you want a simpler on-policy method (use [PPO](ppo.md)).

## References

- Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015). [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236). *Nature*, 518(7540), 529-533.
- van Hasselt, H., Guez, A., & Silver, D. (2016). [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461). *AAAI*.
- Wang, Z., Schaul, T., Hessel, M., et al. (2016). Dueling Network Architectures for Deep Reinforcement Learning. *ICML*.
- Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2016). Prioritized Experience Replay. *ICLR*.
