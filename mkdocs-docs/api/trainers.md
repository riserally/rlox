# Trainers

## Unified Trainer

The `Trainer` class is the single entry point for all 8 algorithms. It wraps any registered algorithm via the Strategy pattern.

```python
from rlox import Trainer

# Any algorithm via string name:
trainer = Trainer("ppo", env="CartPole-v1", seed=42)
trainer = Trainer("sac", env="Pendulum-v1", config={"learning_starts": 500})
trainer = Trainer("dqn", env="CartPole-v1")
trainer = Trainer("td3", env="Pendulum-v1")
trainer = Trainer("a2c", env="CartPole-v1")
trainer = Trainer("mappo", env="CartPole-v1", config={"n_agents": 1})
trainer = Trainer("dreamer", env="CartPole-v1")
trainer = Trainer("impala", env="CartPole-v1")

# Train, save, load:
metrics = trainer.train(total_timesteps=100_000)
trainer.save("checkpoint.pt")
restored = Trainer.from_checkpoint("checkpoint.pt", algorithm="ppo")
```

::: rlox.trainer.Trainer

## Algorithm Registry

::: rlox.trainer.ALGORITHM_REGISTRY

## Deprecated Per-Algorithm Trainers

The following classes are deprecated backward-compatibility wrappers.
Use `Trainer("algo", ...)` instead.

- `PPOTrainer` -> `Trainer("ppo", ...)`
- `SACTrainer` -> `Trainer("sac", ...)`
- `DQNTrainer` -> `Trainer("dqn", ...)`
- `A2CTrainer` -> `Trainer("a2c", ...)`
- `TD3Trainer` -> `Trainer("td3", ...)`
- `MAPPOTrainer` -> `Trainer("mappo", ...)`
- `DreamerV3Trainer` -> `Trainer("dreamer", ...)`
- `IMPALATrainer` -> `Trainer("impala", ...)`
