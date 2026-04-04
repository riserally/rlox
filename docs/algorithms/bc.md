# BC -- Behavioral Cloning

## Intuition

Behavioral Cloning is the simplest possible approach to learning from demonstrations: treat it as supervised learning. Given a dataset of expert state-action pairs, BC trains a neural network to predict the expert's action from the current state. For continuous actions this is a regression problem (MSE loss); for discrete actions it is classification (cross-entropy loss). BC serves as a strong baseline and is often the right choice when you have high-quality expert data and do not need to improve beyond the demonstrator.

## Key Equations

**Continuous actions** (regression):

$$
L(\theta) = \mathbb{E}_{(s,a) \sim \mathcal{D}} \left[ \left\| \pi_\theta(s) - a \right\|^2 \right]
$$

**Discrete actions** (classification):

$$
L(\theta) = -\mathbb{E}_{(s,a) \sim \mathcal{D}} \left[ \log \pi_\theta(a | s) \right]
$$

## Pseudocode

```
algorithm Behavioral Cloning:
    initialize policy network pi_theta
    load offline dataset D = {(s_i, a_i)}

    for update = 1 to n_updates do
        sample minibatch {(s, a)} from D

        if continuous:
            pred = pi_theta(s)
            loss = MSE(pred, a)
        else:
            logits = pi_theta(s)
            loss = cross_entropy(logits, a)

        update theta with Adam
```

## Quick Start

BC uses the offline algorithm interface:

```python
from rlox.offline import OfflineDatasetBuffer
from rlox.algorithms.bc import BC

dataset = OfflineDatasetBuffer.from_d4rl("halfcheetah-expert-v2")
agent = BC(
    dataset=dataset,
    obs_dim=17,
    act_dim=6,
    continuous=True,
    batch_size=256,
)
metrics = agent.train(n_updates=50_000)
```

For discrete actions:

```python
agent = BC(
    dataset=dataset,
    obs_dim=4,
    act_dim=2,       # number of discrete actions
    continuous=False,
    batch_size=256,
)
metrics = agent.train(n_updates=50_000)
```

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `continuous` | `True` | Whether the action space is continuous |
| `hidden` | `256` | Hidden layer width |
| `learning_rate` | `3e-4` | Adam learning rate |
| `batch_size` | `256` | Minibatch size |

## When to Use

- **Use BC when:** you have high-quality expert demonstrations and want the simplest possible baseline, or as a pre-training step before fine-tuning with RL.
- **Prefer BC over offline RL methods when:** the dataset is expert-quality and you do not need to stitch together sub-optimal trajectories.
- **Do not use BC when:** the demonstration data is sub-optimal or multi-modal (prefer [IQL](iql.md), [Diffusion Policy](diffusion.md), or [AWR](awr.md)), or when you need to improve beyond the demonstrator's performance.

## References

- Pomerleau, D. A. (1989). ALVINN: An Autonomous Land Vehicle in a Neural Network. *NeurIPS 1989*.
- Bain, M. & Sammut, C. (1995). A Framework for Behavioural Cloning. *Machine Intelligence 15*, pp. 103-129.
