"""Logging callbacks for training loops.

Provides pluggable loggers for Weights & Biases and TensorBoard.
Extend :class:`LoggerCallback` for custom logging backends.

Example
-------
>>> from rlox.logging import WandbLogger
>>> logger = WandbLogger(project="my-rl-project")
>>> trainer = PPOTrainer(env="CartPole-v1", logger=logger)
"""

from __future__ import annotations

from typing import Any


class LoggerCallback:
    """Base logger callback. Override methods to hook into training events.

    Subclass this to implement custom logging (e.g. CSV, MLflow, etc.).
    """

    def on_train_step(self, step: int, metrics: dict[str, Any]) -> None:
        pass

    def on_rollout_end(self, step: int, metrics: dict[str, Any]) -> None:
        pass

    def on_eval(self, step: int, metrics: dict[str, Any]) -> None:
        pass


class WandbLogger(LoggerCallback):
    """Weights & Biases logger (lazy import)."""

    def __init__(self, project: str = "rlox", **init_kwargs: Any):
        import wandb  # noqa: F401

        self._run = wandb.init(project=project, **init_kwargs)

    def on_train_step(self, step: int, metrics: dict[str, Any]) -> None:
        import wandb

        wandb.log(metrics, step=step)

    def on_eval(self, step: int, metrics: dict[str, Any]) -> None:
        import wandb

        wandb.log({f"eval/{k}": v for k, v in metrics.items()}, step=step)


class TensorBoardLogger(LoggerCallback):
    """TensorBoard logger (lazy import)."""

    def __init__(self, log_dir: str = "runs"):
        from torch.utils.tensorboard import SummaryWriter

        self._writer = SummaryWriter(log_dir=log_dir)

    def on_train_step(self, step: int, metrics: dict[str, Any]) -> None:
        for k, v in metrics.items():
            self._writer.add_scalar(k, v, step)

    def on_eval(self, step: int, metrics: dict[str, Any]) -> None:
        for k, v in metrics.items():
            self._writer.add_scalar(f"eval/{k}", v, step)
