# Trainers API

The unified `Trainer` is the recommended entry point for all training.
The legacy per-algorithm trainers below are deprecated wrappers kept for
backward compatibility.

## Unified Trainer

::: rlox.trainer.Trainer

## Legacy Trainers (Deprecated)

!!! warning
    These classes are deprecated. Use `Trainer('ppo', ...)` etc. instead.
    See the [Python User Guide](../python-guide.md) for the migration path.

::: rlox.trainers.PPOTrainer

::: rlox.trainers.SACTrainer

::: rlox.trainers.DQNTrainer

::: rlox.trainers.A2CTrainer

::: rlox.trainers.TD3Trainer

::: rlox.trainers.MAPPOTrainer

::: rlox.trainers.DreamerV3Trainer

::: rlox.trainers.IMPALATrainer
