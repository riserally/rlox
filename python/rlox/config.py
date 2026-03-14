"""Configuration dataclasses with validation, merging, and serialization."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields
from typing import Any


def _validate_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _validate_min(name: str, value: int, minimum: int) -> None:
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")


@dataclass
class PPOConfig:
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

    def merge(self, overrides: dict[str, Any]) -> PPOConfig:
        d = asdict(self)
        d.update(overrides)
        return PPOConfig.from_dict(d)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SACConfig:
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

    def merge(self, overrides: dict[str, Any]) -> SACConfig:
        d = asdict(self)
        d.update(overrides)
        return SACConfig.from_dict(d)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DQNConfig:
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

    def merge(self, overrides: dict[str, Any]) -> DQNConfig:
        d = asdict(self)
        d.update(overrides)
        return DQNConfig.from_dict(d)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
