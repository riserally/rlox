# DreamerV3 -- World Model RL

## Intuition

DreamerV3 learns a world model from experience, then trains a policy entirely within the learned model ("imagination"). The world model uses a Recurrent State-Space Model (RSSM) that separates state into deterministic and stochastic components. Because the policy trains on imagined trajectories, DreamerV3 is extremely sample-efficient and can handle high-dimensional observations (images) directly. Version 3 introduces symlog predictions and free-nats KL balancing for improved stability across diverse domains.

## Key Equations

**RSSM dynamics** (deterministic + stochastic state):

$$
h_t = f_\phi(h_{t-1}, z_{t-1}, a_{t-1}) \quad \text{(deterministic)}
$$

$$
z_t \sim q_\phi(z_t | h_t, o_t) \quad \text{(posterior, from observation)}
$$

$$
\hat{z}_t \sim p_\phi(z_t | h_t) \quad \text{(prior, for imagination)}
$$

**World model loss** (reconstruction + KL):

$$
L_\text{model} = -\ln p_\phi(o_t | h_t, z_t) - \ln p_\phi(r_t | h_t, z_t) + \beta \, D_\text{KL}[q_\phi(z_t | h_t, o_t) \| p_\phi(z_t | h_t)]
$$

with KL balancing:

$$
L_\text{KL} = \alpha \, D_\text{KL}[\text{sg}(q) \| p] + (1 - \alpha) \, D_\text{KL}[q \| \text{sg}(p)]
$$

**Imagination** (actor-critic in latent space):

$$
\lambda\text{-return:} \quad V_t^\lambda = r_t + \gamma \left[ (1-\lambda) V(s_{t+1}) + \lambda V_{t+1}^\lambda \right]
$$

$$
L_\text{actor} = -\mathbb{E}_{\text{imagine}} \left[ V_t^\lambda \right], \quad L_\text{critic} = \mathbb{E}_{\text{imagine}} \left[ (V(s_t) - \text{sg}(V_t^\lambda))^2 \right]
$$

## Pseudocode

```
algorithm DreamerV3:
    initialize RSSM world model (encoder, dynamics, decoder, reward predictor)
    initialize actor pi_theta, critic V_psi
    initialize replay buffer D

    for step = 1, 2, ... do
        # Environment interaction
        encode o_t, infer z_t ~ q(z|h,o)
        a_t ~ pi_theta(h_t, z_t)
        store (o_t, a_t, r_t) in D

        # World model training
        sample sequences of length seq_len from D
        compute RSSM states (h_t, z_t) from sequences
        L_model = reconstruction_loss + reward_loss + KL_loss(kl_balance, free_nats)
        update world model

        # Imagination training
        imagine trajectories of length imagination_horizon from current states
        compute lambda-returns V_t^lambda along imagined trajectories
        update actor to maximize V_t^lambda
        update critic to predict V_t^lambda
```

## Quick Start

```python
from rlox import Trainer

trainer = Trainer("dreamer", env="CartPole-v1", seed=42, config={
    "batch_size": 16,
    "seq_len": 50,
    "imagination_horizon": 15,
})
metrics = trainer.train(total_timesteps=100_000)
```

For visual control tasks:

```python
trainer = Trainer("dreamer", env="DMC-Cheetah-Run", seed=42, config={
    "learning_rate": 1e-4,
    "buffer_size": 1_000_000,
    "batch_size": 16,
    "seq_len": 50,
    "gamma": 0.997,
    "lambda_": 0.95,
    "deter_dim": 512,
    "stoch_dim": 32,
    "stoch_classes": 32,
    "imagination_horizon": 15,
})
metrics = trainer.train(total_timesteps=1_000_000)
```

## Hyperparameters

All defaults from `DreamerV3Config`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | `1e-4` | Learning rate for all optimizers |
| `buffer_size` | `1_000_000` | Replay buffer capacity |
| `batch_size` | `16` | Number of sequences per training batch |
| `seq_len` | `50` | Sequence length for training |
| `gamma` | `0.997` | Discount factor |
| `lambda_` | `0.95` | Lambda for lambda-returns |
| `deter_dim` | `512` | Deterministic state dimension in RSSM |
| `stoch_dim` | `32` | Number of categorical distributions |
| `stoch_classes` | `32` | Classes per categorical |
| `hidden` | `512` | Hidden layer width |
| `imagination_horizon` | `15` | Steps to imagine ahead for actor-critic |
| `kl_balance` | `0.8` | KL balancing coefficient |
| `free_nats` | `1.0` | Free nats for KL loss |

## When to Use

- **Use DreamerV3 when:** you need maximum sample efficiency, work with pixel observations, or have complex dynamics that benefit from a learned model.
- **Do not use DreamerV3 when:** you need fast wall-clock training time (model-free methods are faster per update), or your environment is simple enough that PPO solves it in minutes.

## References

- Hafner, D., Pasukonis, J., Ba, J., & Lillicrap, T. (2023). [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104). [arXiv:2301.04104](https://arxiv.org/abs/2301.04104) (DreamerV3).
- Hafner, D., Lillicrap, T., Fischer, I., Villegas, R., Ha, D., Lee, H., & Davidson, J. (2019). Learning Latent Dynamics for Planning from Pixels. *ICML* (Dreamer V1).
