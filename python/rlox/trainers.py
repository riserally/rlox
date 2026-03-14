"""One-liner Trainer API wrapping Layer 1 algorithms with sensible defaults."""

from __future__ import annotations

from typing import Any

from rlox.callbacks import Callback, CallbackList


class PPOTrainer:
    """High-level PPO trainer with sensible defaults."""

    def __init__(
        self,
        env: str,
        model: str = "mlp",
        config: dict[str, Any] | None = None,
        callbacks: list[Callback] | None = None,
        logger: Any | None = None,
        seed: int = 42,
    ):
        from rlox.algorithms.ppo import PPO

        cfg = config or {}
        self.algo = PPO(env_id=env, seed=seed, logger=logger, **cfg)
        self.callbacks = CallbackList(callbacks)
        self.env = env

    def train(self, total_timesteps: int) -> dict[str, float]:
        self.callbacks.on_training_start()
        metrics = self.algo.train(total_timesteps=total_timesteps)
        self.callbacks.on_training_end()
        return metrics


class SACTrainer:
    """High-level SAC trainer with sensible defaults."""

    def __init__(
        self,
        env: str,
        config: dict[str, Any] | None = None,
        callbacks: list[Callback] | None = None,
        seed: int = 42,
    ):
        from rlox.algorithms.sac import SAC

        cfg = config or {}
        self.algo = SAC(env_id=env, seed=seed, **cfg)
        self.callbacks = CallbackList(callbacks)
        self.env = env

    def train(self, total_timesteps: int) -> dict[str, float]:
        self.callbacks.on_training_start()
        metrics = self.algo.train(total_timesteps=total_timesteps)
        self.callbacks.on_training_end()
        return metrics


class DQNTrainer:
    """High-level DQN trainer with sensible defaults."""

    def __init__(
        self,
        env: str,
        config: dict[str, Any] | None = None,
        callbacks: list[Callback] | None = None,
        seed: int = 42,
    ):
        from rlox.algorithms.dqn import DQN

        cfg = config or {}
        self.algo = DQN(env_id=env, seed=seed, **cfg)
        self.callbacks = CallbackList(callbacks)
        self.env = env

    def train(self, total_timesteps: int) -> dict[str, float]:
        self.callbacks.on_training_start()
        metrics = self.algo.train(total_timesteps=total_timesteps)
        self.callbacks.on_training_end()
        return metrics
