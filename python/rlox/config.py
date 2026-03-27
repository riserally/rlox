"""Configuration dataclasses with validation, merging, and serialization.

Each config dataclass provides:
- ``from_dict(d)`` — construct from a dict, ignoring unknown keys
- ``merge(overrides)`` — create a new config with selective overrides
- ``to_dict()`` — serialise to a plain dict (e.g. for JSON logging)

Validation is performed in ``__post_init__`` and raises ``ValueError``
for invalid parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Any


def _validate_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _validate_min(name: str, value: int, minimum: int) -> None:
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")


@dataclass
class PPOConfig:
    """Configuration for PPO training.

    Defaults match CleanRL's PPO implementation for CartPole-v1.

    Attributes
    ----------
    n_envs : int
        Number of parallel environments (default 8).
    n_steps : int
        Rollout length per environment per update (default 128).
    n_epochs : int
        Number of SGD passes over each rollout (default 4).
    batch_size : int
        Minibatch size for SGD (default 256).
    learning_rate : float
        Adam learning rate (default 2.5e-4).
    clip_eps : float
        PPO clipping range for the probability ratio (default 0.2).
    vf_coef : float
        Value loss coefficient (default 0.5).
    ent_coef : float
        Entropy bonus coefficient (default 0.01).
    max_grad_norm : float
        Maximum gradient norm for clipping (default 0.5).
    gamma : float
        Discount factor (default 0.99).
    gae_lambda : float
        GAE lambda (default 0.95).
    normalize_advantages : bool
        Whether to normalise advantages per minibatch (default True).
    clip_vloss : bool
        Whether to clip the value loss (default True).
    anneal_lr : bool
        Whether to linearly anneal the learning rate (default True).
    """
    n_envs: int = 8
    n_steps: int = 128
    n_epochs: int = 4
    batch_size: int = 256
    learning_rate: float = 2.5e-4
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    normalize_advantages: bool = True
    clip_vloss: bool = True
    anneal_lr: bool = True
    normalize_rewards: bool = False
    normalize_obs: bool = False

    def __post_init__(self):
        _validate_positive("learning_rate", self.learning_rate)
        _validate_min("n_envs", self.n_envs, 1)
        _validate_min("n_steps", self.n_steps, 1)
        _validate_min("n_epochs", self.n_epochs, 1)
        _validate_min("batch_size", self.batch_size, 1)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PPOConfig:
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def from_yaml(cls, path: str | Path) -> PPOConfig:
        """Load config from a YAML file, ignoring unknown keys."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    def merge(self, overrides: dict[str, Any]) -> PPOConfig:
        d = asdict(self)
        d.update(overrides)
        return PPOConfig.from_dict(d)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        """Save config to a YAML file."""
        import yaml

        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)


@dataclass
class SACConfig:
    """Configuration for SAC training.

    Defaults match rl-zoo3 SAC hyperparameters.

    Attributes
    ----------
    learning_rate : float
        Learning rate for all optimisers (default 3e-4).
    buffer_size : int
        Replay buffer capacity (default 1M).
    batch_size : int
        Minibatch size for critic/actor updates (default 256).
    tau : float
        Polyak averaging coefficient for target networks (default 0.005).
    gamma : float
        Discount factor (default 0.99).
    target_entropy : float or None
        Target entropy for auto-tuning. None = ``-dim(action_space)``.
    auto_entropy : bool
        Whether to automatically tune the entropy coefficient (default True).
    learning_starts : int
        Number of random exploration steps before training (default 1000).
    hidden : int
        Hidden layer width for actor and critic networks (default 256).
    """
    learning_rate: float = 3e-4
    buffer_size: int = 1_000_000
    batch_size: int = 256
    tau: float = 0.005
    gamma: float = 0.99
    target_entropy: float | None = None  # auto = -dim(A)
    auto_entropy: bool = True
    learning_starts: int = 1000
    hidden: int = 256

    def __post_init__(self):
        _validate_positive("learning_rate", self.learning_rate)
        _validate_min("buffer_size", self.buffer_size, 1)
        _validate_min("batch_size", self.batch_size, 1)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SACConfig:
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def from_yaml(cls, path: str | Path) -> SACConfig:
        """Load config from a YAML file, ignoring unknown keys."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    def merge(self, overrides: dict[str, Any]) -> SACConfig:
        d = asdict(self)
        d.update(overrides)
        return SACConfig.from_dict(d)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        """Save config to a YAML file."""
        import yaml

        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)


@dataclass
class DQNConfig:
    """Configuration for DQN training with Rainbow extensions.

    Supports Double DQN, Dueling architecture, N-step returns, and
    Prioritized Experience Replay (PER).

    Attributes
    ----------
    learning_rate : float
        Adam learning rate (default 1e-4).
    buffer_size : int
        Replay buffer capacity (default 1M).
    batch_size : int
        Minibatch size (default 64).
    gamma : float
        Discount factor (default 0.99).
    target_update_freq : int
        Steps between hard target network updates (default 1000).
    exploration_fraction : float
        Fraction of training for epsilon decay (default 0.1).
    exploration_initial_eps : float
        Starting epsilon for exploration (default 1.0).
    exploration_final_eps : float
        Final epsilon after decay (default 0.05).
    learning_starts : int
        Random exploration steps before training (default 1000).
    double_dqn : bool
        Use Double DQN action selection (default True).
    dueling : bool
        Use Dueling network architecture (default False).
    n_step : int
        N-step return horizon (default 1).
    prioritized : bool
        Use Prioritized Experience Replay (default False).
    alpha : float
        PER priority exponent (default 0.6).
    beta_start : float
        PER initial importance-sampling exponent (default 0.4).
    hidden : int
        Hidden layer width (default 256).
    """
    learning_rate: float = 1e-4
    buffer_size: int = 1_000_000
    batch_size: int = 64
    gamma: float = 0.99
    target_update_freq: int = 1000
    exploration_fraction: float = 0.1
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.05
    learning_starts: int = 1000
    double_dqn: bool = True
    dueling: bool = False
    n_step: int = 1
    prioritized: bool = False
    alpha: float = 0.6
    beta_start: float = 0.4
    hidden: int = 256

    def __post_init__(self):
        _validate_positive("learning_rate", self.learning_rate)
        _validate_min("buffer_size", self.buffer_size, 1)
        _validate_min("batch_size", self.batch_size, 1)
        _validate_min("n_step", self.n_step, 1)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DQNConfig:
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def from_yaml(cls, path: str | Path) -> DQNConfig:
        """Load config from a YAML file, ignoring unknown keys."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    def merge(self, overrides: dict[str, Any]) -> DQNConfig:
        d = asdict(self)
        d.update(overrides)
        return DQNConfig.from_dict(d)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        """Save config to a YAML file."""
        import yaml

        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
