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

    def save(self, path: str) -> None:
        """Save training checkpoint."""
        self.algo.save(path)

    @classmethod
    def from_checkpoint(cls, path: str, env: str | None = None) -> SACTrainer:
        """Restore from checkpoint."""
        from rlox.algorithms.sac import SAC

        trainer = object.__new__(cls)
        trainer.algo = SAC.from_checkpoint(path, env_id=env)
        trainer.env = env or trainer.algo.env_id
        trainer.callbacks = CallbackList(None)
        return trainer


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

    def save(self, path: str) -> None:
        """Save training checkpoint."""
        self.algo.save(path)

    @classmethod
    def from_checkpoint(cls, path: str, env: str | None = None) -> DQNTrainer:
        """Restore from checkpoint."""
        from rlox.algorithms.dqn import DQN

        trainer = object.__new__(cls)
        trainer.algo = DQN.from_checkpoint(path, env_id=env)
        trainer.env = env or trainer.algo.env_id
        trainer.callbacks = CallbackList(None)
        return trainer


class A2CTrainer:
    """High-level A2C trainer with callback/logger integration.

    Parameters
    ----------
    env : str
        Gymnasium environment ID (e.g. "CartPole-v1").
    config : dict, optional
        Override any A2CConfig fields.
    callbacks : list[Callback], optional
        Training callbacks.
    logger : LoggerCallback, optional
        Logger instance.
    seed : int
        Random seed (default 42).
    """

    def __init__(
        self,
        env: str,
        config: dict[str, Any] | None = None,
        callbacks: list[Callback] | None = None,
        logger: Any | None = None,
        seed: int = 42,
        compile: bool = False,
    ):
        from rlox.algorithms.a2c import A2C

        cfg = config or {}
        self.algo = A2C(
            env_id=env, seed=seed, logger=logger, callbacks=callbacks, **cfg
        )
        self.callbacks = CallbackList(callbacks)
        self.env = env

        if compile:
            from rlox.compile import compile_policy

            compile_policy(self)

    def train(self, total_timesteps: int) -> dict[str, float]:
        """Run A2C training."""
        self.callbacks.on_training_start()
        metrics = self.algo.train(total_timesteps=total_timesteps)
        self.callbacks.on_training_end()
        return metrics

    def save(self, path: str) -> None:
        """Save training checkpoint."""
        self.algo.save(path)

    @classmethod
    def from_checkpoint(cls, path: str, env: str | None = None) -> A2CTrainer:
        """Restore from checkpoint."""
        from rlox.algorithms.a2c import A2C

        trainer = object.__new__(cls)
        trainer.algo = A2C.from_checkpoint(path, env_id=env)
        trainer.env = env or trainer.algo.env_id
        trainer.callbacks = CallbackList(None)
        return trainer


class TD3Trainer:
    """High-level TD3 trainer for continuous action spaces.

    Parameters
    ----------
    env : str
        Gymnasium environment ID with continuous actions (e.g. "Pendulum-v1").
    config : dict, optional
        Override any TD3Config fields.
    callbacks : list[Callback], optional
        Training callbacks.
    logger : LoggerCallback, optional
        Logger instance.
    seed : int
        Random seed (default 42).
    """

    def __init__(
        self,
        env: str,
        config: dict[str, Any] | None = None,
        callbacks: list[Callback] | None = None,
        logger: Any | None = None,
        seed: int = 42,
        compile: bool = False,
    ):
        from rlox.algorithms.td3 import TD3

        cfg = config or {}
        self.algo = TD3(
            env_id=env, seed=seed, logger=logger, callbacks=callbacks, **cfg
        )
        self.callbacks = CallbackList(callbacks)
        self.env = env

        if compile:
            from rlox.compile import compile_policy

            compile_policy(self)

    def train(self, total_timesteps: int) -> dict[str, float]:
        """Run TD3 training."""
        self.callbacks.on_training_start()
        metrics = self.algo.train(total_timesteps=total_timesteps)
        self.callbacks.on_training_end()
        return metrics

    def save(self, path: str) -> None:
        """Save training checkpoint."""
        self.algo.save(path)

    @classmethod
    def from_checkpoint(cls, path: str, env: str | None = None) -> TD3Trainer:
        """Restore from checkpoint."""
        from rlox.algorithms.td3 import TD3

        trainer = object.__new__(cls)
        trainer.algo = TD3.from_checkpoint(path, env_id=env)
        trainer.env = env or trainer.algo.env_id
        trainer.callbacks = CallbackList(None)
        return trainer


class MAPPOTrainer:
    """High-level MAPPO trainer for multi-agent environments.

    Parameters
    ----------
    env : str
        Environment ID or multi-agent environment identifier.
    config : dict, optional
        Override any MAPPOConfig fields.
    callbacks : list[Callback], optional
        Training callbacks.
    logger : LoggerCallback, optional
        Logger instance.
    seed : int
        Random seed (default 42).
    """

    def __init__(
        self,
        env: str,
        config: dict[str, Any] | None = None,
        callbacks: list[Callback] | None = None,
        logger: Any | None = None,
        seed: int = 42,
    ):
        from rlox.algorithms.mappo import MAPPO

        cfg = config or {}
        self.algo = MAPPO(
            env_id=env, seed=seed, logger=logger, callbacks=callbacks, **cfg
        )
        self.callbacks = CallbackList(callbacks)
        self.env = env

    def train(self, total_timesteps: int) -> dict[str, float]:
        """Run MAPPO training."""
        self.callbacks.on_training_start()
        metrics = self.algo.train(total_timesteps=total_timesteps)
        self.callbacks.on_training_end()
        return metrics

    def save(self, path: str) -> None:
        """Save training checkpoint."""
        self.algo.save(path)

    @classmethod
    def from_checkpoint(cls, path: str, env: str | None = None) -> MAPPOTrainer:
        """Restore from checkpoint."""
        from rlox.algorithms.mappo import MAPPO

        trainer = object.__new__(cls)
        trainer.algo = MAPPO.from_checkpoint(path, env_id=env)
        trainer.env = env or trainer.algo.env_id
        trainer.callbacks = CallbackList(None)
        return trainer


class DreamerV3Trainer:
    """High-level DreamerV3 trainer for world-model-based RL.

    Parameters
    ----------
    env : str
        Gymnasium environment ID.
    config : dict, optional
        Override any DreamerV3Config fields.
    callbacks : list[Callback], optional
        Training callbacks.
    logger : LoggerCallback, optional
        Logger instance (attached to inner algorithm if supported).
    seed : int
        Random seed (default 42).
    """

    def __init__(
        self,
        env: str,
        config: dict[str, Any] | None = None,
        callbacks: list[Callback] | None = None,
        logger: Any | None = None,
        seed: int = 42,
    ):
        from rlox.algorithms.dreamer import DreamerV3

        cfg = config or {}
        self.algo = DreamerV3(env_id=env, seed=seed, **cfg)
        # Attach logger if the algorithm supports it
        if logger is not None and hasattr(self.algo, "logger"):
            self.algo.logger = logger
        self.callbacks = CallbackList(callbacks)
        self.env = env

    def train(self, total_timesteps: int) -> dict[str, float]:
        """Run DreamerV3 training."""
        self.callbacks.on_training_start()
        metrics = self.algo.train(total_timesteps=total_timesteps)
        self.callbacks.on_training_end()
        return metrics

    def save(self, path: str) -> None:
        """Save training checkpoint."""
        self.algo.save(path)

    @classmethod
    def from_checkpoint(cls, path: str, env: str | None = None) -> DreamerV3Trainer:
        """Restore from checkpoint."""
        from rlox.algorithms.dreamer import DreamerV3

        trainer = object.__new__(cls)
        trainer.algo = DreamerV3.from_checkpoint(path, env_id=env)
        trainer.env = env or trainer.algo.env_id
        trainer.callbacks = CallbackList(None)
        return trainer


class IMPALATrainer:
    """High-level IMPALA trainer with actor-learner architecture.

    Parameters
    ----------
    env : str
        Gymnasium environment ID.
    config : dict, optional
        Override any IMPALAConfig fields.
    callbacks : list[Callback], optional
        Training callbacks.
    logger : LoggerCallback, optional
        Logger instance.
    seed : int
        Random seed (default 42).
    """

    def __init__(
        self,
        env: str,
        config: dict[str, Any] | None = None,
        callbacks: list[Callback] | None = None,
        logger: Any | None = None,
        seed: int = 42,
    ):
        from rlox.algorithms.impala import IMPALA

        cfg = config or {}
        self.algo = IMPALA(
            env_id=env, seed=seed, logger=logger, callbacks=callbacks, **cfg
        )
        self.callbacks = CallbackList(callbacks)
        self.env = env

    def train(self, total_timesteps: int) -> dict[str, float]:
        """Run IMPALA training."""
        self.callbacks.on_training_start()
        metrics = self.algo.train(total_timesteps=total_timesteps)
        self.callbacks.on_training_end()
        return metrics

    def save(self, path: str) -> None:
        """Save training checkpoint."""
        self.algo.save(path)

    @classmethod
    def from_checkpoint(cls, path: str, env: str | None = None) -> IMPALATrainer:
        """Restore from checkpoint."""
        from rlox.algorithms.impala import IMPALA

        trainer = object.__new__(cls)
        trainer.algo = IMPALA.from_checkpoint(path, env_id=env)
        trainer.env = env or trainer.algo.env_id
        trainer.callbacks = CallbackList(None)
        return trainer
