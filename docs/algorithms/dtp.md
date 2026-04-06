# DTP -- Decision Tree Policy

## Intuition

Decision Tree Policy replaces neural networks with gradient-boosted trees (XGBoost) for offline reinforcement learning. Instead of training a deep network for millions of steps, DTP fits a tree ensemble in seconds on a pre-collected dataset. Two variants exist: **RWDTP** weights samples by their return quality, while **RCDTP** conditions on the desired return (like Decision Transformer, but with trees instead of transformers).

The key advantage is speed: training takes seconds instead of hours, and the resulting policies are interpretable via feature importance.

## Key Equations

### RWDTP (Return-Weighted Decision Tree Policy)

Fit a regression tree $\hat{\pi}(s) \approx a$ with sample weights proportional to normalized discounted returns raised to power $p$:

$$w_i = \left(\frac{G_i - G_{\min}}{G_{\max} - G_{\min}}\right)^p$$

where $G_i = \sum_{t=0}^{T} \gamma^t r_t^{(i)}$ is the discounted return of trajectory $i$.

Higher $p$ concentrates weight on the best trajectories. The policy is then:

$$a = \hat{\pi}_\theta(s) = \text{XGBoost}(s; \theta)$$

### RCDTP (Return-Conditioned Decision Tree Policy)

Fit a regression tree on augmented inputs $(s, \hat{G}, t) \to a$:

$$a = \hat{\pi}_\theta(s, \hat{G}, t)$$

where $\hat{G}$ is the return-to-go and $t$ is the timestep. At inference, set $\hat{G}$ to the desired target return and decrement each step:

$$\hat{G}_{t+1} = \hat{G}_t - r_t$$

## Pseudocode

```
algorithm RWDTP:
    input: offline dataset D = {(s, a, r, s', done)}, power p, gamma
    compute discounted returns G_i for each trajectory
    normalize: w_i = ((G_i - min) / (max - min))^p
    fit XGBoost regressor: s -> a, sample_weight = w

algorithm RCDTP:
    input: offline dataset D, gamma, target_return
    compute returns-to-go RTG_t for each transition
    augment features: X = [s, RTG_t, t]
    fit XGBoost regressor: X -> a
    at inference: set RTG = target_return, decrement by reward each step
```

## Quick Start

```python
from rlox import Trainer

# RWDTP: return-weighted regression (simpler, faster)
trainer = Trainer("rwdtp", env="HalfCheetah-v4", config={
    "n_estimators": 500,
    "max_depth": 6,
    "return_power": 2.0,
})
metrics = trainer.train(total_timesteps=100_000)  # collect data then fit

# RCDTP: return-conditioned (more flexible, handles suboptimal data)
trainer = Trainer("rcdtp", env="HalfCheetah-v4", config={
    "n_estimators": 500,
    "target_return": 5000.0,
})
metrics = trainer.train(total_timesteps=100_000)
```

## Hyperparameters

All defaults from `DTPConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_estimators` | `500` | Number of boosting rounds (trees) |
| `max_depth` | `6` | Maximum tree depth |
| `learning_rate` | `0.1` | XGBoost learning rate (shrinkage) |
| `gamma` | `0.99` | Discount factor for return computation |
| `return_power` | `2.0` | RWDTP: exponent for return weighting |
| `target_return` | `None` | RCDTP: desired return for inference |
| `subsample` | `0.8` | Row subsampling ratio per tree |
| `colsample_bytree` | `0.8` | Column subsampling ratio per tree |
| `buffer_size` | `100_000` | Size of data collection buffer |

## When to Use

- **Use RWDTP when:** you have a dataset of mostly-good demonstrations and want a fast, interpretable policy. Training takes seconds, not hours.
- **Use RCDTP when:** your dataset has mixed quality (some good, some bad trajectories) and you want to specify the desired performance level at inference time.
- **Do not use DTP when:** you need online learning (DTP is offline-only), your state space is very high-dimensional images (trees struggle with raw pixels), or you need a stochastic policy for exploration.

## References

- Koirala, S. & Fleming, C. (2024). [Solving Offline Reinforcement Learning with Decision Tree Regression](https://arxiv.org/abs/2401.11630). *arXiv:2401.11630*.
- Chen, T. & Guestrin, C. (2016). [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754). *KDD 2016*.
