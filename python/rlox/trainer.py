"""Unified Trainer wrapping any registered algorithm via the Strategy pattern.

The ``Trainer`` class replaces the 8 individual ``XTrainer`` classes with a
single entry point.  Algorithms are looked up by name in the
:data:`ALGORITHM_REGISTRY` or passed directly as a class.

Example
-------
>>> from rlox.trainer import Trainer
>>> trainer = Trainer("ppo", env="CartPole-v1", config={"n_envs": 16})
>>> metrics = trainer.train(total_timesteps=100_000)
>>> trainer.save("checkpoint.pt")
"""

from __future__ import annotations

import inspect
from typing import Any

from rlox.callbacks import Callback, CallbackList


# ---------------------------------------------------------------------------
# Algorithm registry
# ---------------------------------------------------------------------------

ALGORITHM_REGISTRY: dict[str, type] = {}


def register_algorithm(name: str):
    """Class decorator that registers an algorithm under *name*."""
    def decorator(cls: type) -> type:
        ALGORITHM_REGISTRY[name.lower()] = cls
        return cls
    return decorator


def _accepts_param(cls: type, param: str) -> bool:
    """Return True if *cls.__init__* has a parameter named *param*."""
    try:
        sig = inspect.signature(cls.__init__)
        return param in sig.parameters
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# Unified Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Unified trainer wrapping any registered algorithm.

    Parameters
    ----------
    algorithm : str | type
        Algorithm name (e.g. ``"ppo"``) or the algorithm class itself.
    env : str
        Gymnasium environment ID.
    config : dict, optional
        Algorithm-specific hyperparameters forwarded to the constructor.
    callbacks : list[Callback], optional
        Training callbacks.
    logger : object, optional
        Logger instance (WandbLogger, TensorBoardLogger, etc.).
    seed : int
        Random seed (default 42).
    compile : bool
        Whether to torch.compile the policy (default False).
    """

    def __init__(
        self,
        algorithm: str | type,
        env: str,
        config: dict[str, Any] | None = None,
        callbacks: list[Callback] | None = None,
        logger: Any | None = None,
        seed: int = 42,
        compile: bool = False,
    ):
        cfg = config or {}

        # Resolve algorithm class
        if isinstance(algorithm, str):
            algo_cls = ALGORITHM_REGISTRY.get(algorithm.lower())
            if algo_cls is None:
                raise ValueError(
                    f"Unknown algorithm {algorithm!r}. "
                    f"Registered: {sorted(ALGORITHM_REGISTRY)}"
                )
        else:
            algo_cls = algorithm

        # Build algorithm -- inspect __init__ to pass only accepted params
        algo_kwargs: dict[str, Any] = {"env_id": env, "seed": seed}
        if _accepts_param(algo_cls, "logger"):
            algo_kwargs["logger"] = logger
        if _accepts_param(algo_cls, "callbacks"):
            algo_kwargs["callbacks"] = callbacks
        if _accepts_param(algo_cls, "compile"):
            algo_kwargs["compile"] = compile
        algo_kwargs.update(cfg)

        self.algo = algo_cls(**algo_kwargs)
        self.env = env
        self._logger = logger
        self._callbacks = CallbackList(callbacks)

        # Attach logger if not accepted by algo __init__
        if logger is not None and not _accepts_param(algo_cls, "logger"):
            if hasattr(self.algo, "logger"):
                self.algo.logger = logger

    def train(self, total_timesteps: int) -> dict[str, float]:
        """Run training and return metrics dict."""
        return self.algo.train(total_timesteps=total_timesteps)

    def save(self, path: str) -> None:
        """Save training checkpoint."""
        self.algo.save(path)

    def predict(self, obs: Any, deterministic: bool = True) -> Any:
        """Get action from the trained policy."""
        return self.algo.predict(obs, deterministic=deterministic)

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        algorithm: str | type,
        env: str | None = None,
    ) -> Trainer:
        """Restore a Trainer from a saved checkpoint.

        Parameters
        ----------
        path : str
            Path to the checkpoint file.
        algorithm : str | type
            Algorithm name or class.
        env : str, optional
            Environment ID (uses checkpoint's env_id if None).
        """
        if isinstance(algorithm, str):
            algo_cls = ALGORITHM_REGISTRY.get(algorithm.lower())
            if algo_cls is None:
                raise ValueError(
                    f"Unknown algorithm {algorithm!r}. "
                    f"Registered: {sorted(ALGORITHM_REGISTRY)}"
                )
        else:
            algo_cls = algorithm

        trainer = object.__new__(cls)
        trainer.algo = algo_cls.from_checkpoint(path, env_id=env)
        trainer.env = env or trainer.algo.env_id
        trainer._callbacks = CallbackList(None)
        trainer._logger = None
        return trainer


# ---------------------------------------------------------------------------
# Register all built-in algorithms (lazy to avoid circular imports)
# ---------------------------------------------------------------------------

def _register_builtins() -> None:
    """Import and register all 8 built-in algorithms."""
    from rlox.algorithms.ppo import PPO
    from rlox.algorithms.sac import SAC
    from rlox.algorithms.dqn import DQN
    from rlox.algorithms.a2c import A2C
    from rlox.algorithms.td3 import TD3
    from rlox.algorithms.mappo import MAPPO
    from rlox.algorithms.dreamer import DreamerV3
    from rlox.algorithms.impala import IMPALA

    for name, cls in [
        ("ppo", PPO),
        ("sac", SAC),
        ("dqn", DQN),
        ("a2c", A2C),
        ("td3", TD3),
        ("mappo", MAPPO),
        ("dreamer", DreamerV3),
        ("impala", IMPALA),
    ]:
        if name not in ALGORITHM_REGISTRY:
            ALGORITHM_REGISTRY[name] = cls


_register_builtins()
