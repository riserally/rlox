# Decision Transformer

## Intuition

Decision Transformer recasts reinforcement learning as a sequence modeling problem. Instead of learning value functions or policy gradients, it trains a causal transformer on (return-to-go, state, action) triples from offline data. At inference time, you condition on a desired return-to-go and the model autoregressively generates actions that aim to achieve that target return. This makes it a purely supervised-learning approach to RL, inheriting the scalability and stability of transformer training.

## Key Equations

The model takes as input a sequence of interleaved tokens:

$$
\tau = (R_0, s_0, a_0, R_1, s_1, a_1, \ldots, R_T, s_T, a_T)
$$

where $R_t = \sum_{t'=t}^{T} r_{t'}$ is the return-to-go at timestep $t$.

The training objective is standard supervised learning on actions:

$$
L(\theta) = \mathbb{E}_{\tau \sim \mathcal{D}} \left[ \sum_{t=1}^{T} \ell\!\left( \hat{a}_t^\theta, a_t \right) \right]
$$

where $\ell$ is cross-entropy for discrete actions or MSE for continuous actions.

At inference, the return-to-go is set to a target value:

$$
R_0 = R_{\text{target}}, \quad R_{t+1} = R_t - r_t
$$

## Pseudocode

```
algorithm Decision Transformer:
    collect offline dataset D = {(s_t, a_t, r_t)_t}
    compute return-to-go R_t for each timestep in each episode

    initialize transformer model with context_length K

    for update = 1 to n_updates do
        sample batch of subsequences from D
        for each subsequence:
            input = interleave(R_{t:t+K}, s_{t:t+K}, a_{t:t+K})
            a_pred = transformer(input) at state positions
            loss = cross_entropy(a_pred, a_true) or MSE(a_pred, a_true)
        update model with Adam

    # Inference
    set R_0 = target_return
    for t = 0, 1, ... do
        a_t = transformer.predict(R_{0:t}, s_{0:t}, a_{0:t-1})
        execute a_t, observe r_t, s_{t+1}
        R_{t+1} = R_t - r_t
```

## Quick Start

```python
from rlox import Trainer

trainer = Trainer("dt", env="CartPole-v1", seed=42, config={
    "context_length": 20,
    "target_return": 200.0,
})
metrics = trainer.train(total_timesteps=50_000)
print(f"Loss: {metrics['loss']:.4f}")
```

For continuous control:

```python
trainer = Trainer("dt", env="HalfCheetah-v4", seed=42, config={
    "context_length": 20,
    "embed_dim": 128,
    "n_heads": 4,
    "n_layers": 3,
    "batch_size": 64,
    "target_return": 6000.0,
})
metrics = trainer.train(total_timesteps=1_000_000)
```

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `context_length` | `20` | Number of timesteps in the transformer context window |
| `n_heads` | `4` | Number of attention heads |
| `n_layers` | `3` | Number of transformer layers |
| `embed_dim` | `128` | Embedding dimension |
| `learning_rate` | `1e-4` | Adam learning rate |
| `batch_size` | `64` | Minibatch size for training |
| `dropout` | `0.1` | Dropout rate |
| `target_return` | `200.0` | Desired return-to-go for evaluation |
| `warmup_steps` | `500` | Data collection steps before training |
| `seed` | `42` | Random seed |

## When to Use

- **Use Decision Transformer when:** you have a large offline dataset and want to leverage transformer architectures, or when you want return-conditioned behavior (generate policies of varying quality).
- **Prefer DT over CQL/IQL when:** the dataset is large and diverse, and you want to scale with model size rather than algorithmic complexity.
- **Do not use DT when:** online interaction is cheap (on-policy methods like [PPO](ppo.md) will be simpler), or the offline dataset is small (prefer [IQL](iql.md) or [TD3+BC](td3bc.md)).

## References

- Chen, L., Lu, K., Rajeswaran, A., Lee, K., Grover, A., Laskin, M., Abbeel, P., Srinivas, A., & Mordatch, I. (2021). Decision Transformer: Reinforcement Learning via Sequence Modeling. *NeurIPS 2021*. [arXiv:2106.01345](https://arxiv.org/abs/2106.01345).
