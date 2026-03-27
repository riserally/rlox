"""One-liner Trainer API wrapping algorithm implementations with sensible defaults.

These trainers are the simplest way to get started with rlox. Each wraps
the corresponding algorithm class and provides a ``train(total_timesteps)``
interface.

Example
-------
>>> from rlox.trainers import PPOTrainer
>>> trainer = PPOTrainer(env="CartPole-v1", seed=42)
>>> metrics = trainer.train(total_timesteps=50_000)
>>> print(f"Mean reward: {metrics['mean_reward']:.1f}")
"""

from __future__ import annotations

from typing import Any

from rlox.callbacks import Callback, CallbackList


class PPOTrainer:
    """High-level PPO trainer with sensible defaults.

    Parameters
    ----------
    env : str
        Gymnasium environment ID (e.g. "CartPole-v1").
    model : str
        Network architecture identifier (currently only "mlp").
    config : dict, optional
        Override any PPOConfig fields (e.g. ``{"n_envs": 16, "n_steps": 256}``).
    callbacks : list[Callback], optional
        Training callbacks for evaluation, logging, etc.
    logger : LoggerCallback, optional
        Logger instance (WandbLogger, TensorBoardLogger, or custom).
    seed : int
        Random seed for reproducibility (default 42).
    """

    def __init__(
        self,
        env: str,
        model: str = "mlp",
        config: dict[str, Any] | None = None,
        callbacks: list[Callback] | None = None,
        logger: Any | None = None,
        seed: int = 42,
        compile: bool = False,
    ):
        from rlox.algorithms.ppo import PPO

        cfg = config or {}
        self.algo = PPO(
            env_id=env, seed=seed, logger=logger, callbacks=callbacks, **cfg
        )
        self.env = env

        if compile:
            from rlox.compile import compile_policy

            compile_policy(self)

    def train(self, total_timesteps: int) -> dict[str, float]:
        """Run PPO training.

        Returns
        -------
        dict with keys: policy_loss, value_loss, entropy, approx_kl,
        clip_fraction, mean_reward.
        """
        return self.algo.train(total_timesteps=total_timesteps)

    def save(self, path: str) -> None:
        """Save training checkpoint."""
        self.algo.save(path)

    @classmethod
    def from_checkpoint(cls, path: str, env: str | None = None) -> PPOTrainer:
        """Restore from checkpoint."""
        from rlox.algorithms.ppo import PPO

        trainer = object.__new__(cls)
        trainer.algo = PPO.from_checkpoint(path, env_id=env)
        trainer.env = env or trainer.algo.env_id
        return trainer


class SACTrainer:
    """High-level SAC trainer for continuous action spaces.

    Parameters
    ----------
    env : str
        Gymnasium environment ID with continuous actions (e.g. "Pendulum-v1").
    config : dict, optional
        Override any SACConfig fields.
    callbacks : list[Callback], optional
        Training callbacks.
    seed : int
        Random seed (default 42).
    """

    def __init__(
        self,
        env: str,
        config: dict[str, Any] | None = None,
        callbacks: list[Callback] | None = None,
        seed: int = 42,
        compile: bool = False,
    ):
        from rlox.algorithms.sac import SAC

        cfg = config or {}
        self.algo = SAC(env_id=env, seed=seed, callbacks=callbacks, **cfg)
        self.callbacks = CallbackList(callbacks)
        self.env = env

        if compile:
            from rlox.compile import compile_policy

            compile_policy(self)

    def train(self, total_timesteps: int) -> dict[str, float]:
        self.callbacks.on_training_start()
        metrics = self.algo.train(total_timesteps=total_timesteps)
        self.callbacks.on_training_end()
        return metrics


class DQNTrainer:
    """High-level DQN trainer for discrete action spaces.

    Parameters
    ----------
    env : str
        Gymnasium environment ID with discrete actions (e.g. "CartPole-v1").
    config : dict, optional
        Override any DQNConfig fields (e.g. ``{"double_dqn": True, "dueling": True}``).
    callbacks : list[Callback], optional
        Training callbacks.
    seed : int
        Random seed (default 42).
    """

    def __init__(
        self,
        env: str,
        config: dict[str, Any] | None = None,
        callbacks: list[Callback] | None = None,
        seed: int = 42,
        compile: bool = False,
    ):
        from rlox.algorithms.dqn import DQN

        cfg = config or {}
        self.algo = DQN(env_id=env, seed=seed, **cfg)
        self.callbacks = CallbackList(callbacks)
        self.env = env

        if compile:
            from rlox.compile import compile_policy

            compile_policy(self)

    def train(self, total_timesteps: int) -> dict[str, float]:
        self.callbacks.on_training_start()
        metrics = self.algo.train(total_timesteps=total_timesteps)
        self.callbacks.on_training_end()
        return metrics
