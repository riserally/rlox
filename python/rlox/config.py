"""Configuration dataclasses with validation, merging, and serialization.

Each config dataclass provides:
- ``from_dict(d)`` — construct from a dict, ignoring unknown keys
- ``merge(overrides)`` — create a new config with selective overrides
- ``to_dict()`` — serialise to a plain dict (e.g. for JSON logging)

Validation is performed in ``__post_init__`` and raises ``ValueError``
for invalid parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field, fields
from pathlib import Path
from typing import Any


def _validate_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _validate_min(name: str, value: int, minimum: int) -> None:
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")


def _load_toml(path: str | Path) -> dict[str, Any]:
    """Read a TOML file using tomllib (3.11+)."""
    import tomllib

    with open(path, "rb") as f:
        return tomllib.load(f)


def _write_toml(data: dict[str, Any], path: str | Path) -> None:
    """Write a dict to a TOML file.

    Uses ``tomli_w`` if available, otherwise falls back to a simple
    serialiser that handles the types we actually use.
    """
    try:
        import tomli_w

        with open(path, "wb") as f:
            tomli_w.dump(data, f)
    except ImportError:
        with open(path, "w") as f:
            f.write(_dict_to_toml(data))


def _dict_to_toml(data: dict[str, Any], _prefix: str = "") -> str:
    """Minimal TOML serialiser for primitive types + nested dicts."""
    lines: list[str] = []
    tables: list[tuple[str, dict]] = []

    for key, val in data.items():
        if isinstance(val, dict):
            tables.append((key, val))
        else:
            lines.append(f"{key} = {_toml_value(val)}")

    for key, sub in tables:
        section = f"{_prefix}{key}" if not _prefix else f"{_prefix}.{key}"
        lines.append(f"\n[{section}]")
        lines.append(_dict_to_toml(sub, section))

    return "\n".join(lines) + "\n"


def _toml_value(val: Any) -> str:
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, int):
        return str(val)
    if isinstance(val, float):
        return repr(val)
    if isinstance(val, str):
        return f'"{val}"'
    if isinstance(val, list):
        inner = ", ".join(_toml_value(v) for v in val)
        return f"[{inner}]"
    if val is None:
        # TOML has no null — use empty string as sentinel
        return '""'
    return repr(val)


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

    @classmethod
    def from_toml(cls, path: str | Path) -> PPOConfig:
        """Load config from a TOML file, ignoring unknown keys."""
        return cls.from_dict(_load_toml(path))

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

    def to_toml(self, path: str | Path) -> None:
        """Save config to a TOML file."""
        _write_toml(self.to_dict(), path)


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

    @classmethod
    def from_toml(cls, path: str | Path) -> SACConfig:
        """Load config from a TOML file, ignoring unknown keys."""
        data = _load_toml(path)
        # TOML has no null — empty string is our sentinel for None
        if data.get("target_entropy") == "":
            data["target_entropy"] = None
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

    def to_toml(self, path: str | Path) -> None:
        """Save config to a TOML file."""
        _write_toml(self.to_dict(), path)


@dataclass
class A2CConfig:
    """Configuration for A2C (Advantage Actor-Critic) training.

    A2C uses a single gradient step per rollout (no clipping, no epochs).
    Typically paired with short rollouts (n_steps=5) and RMSprop.

    Attributes
    ----------
    learning_rate : float
        RMSprop learning rate (default 7e-4).
    n_steps : int
        Rollout length per update (default 5).
    gamma : float
        Discount factor (default 0.99).
    gae_lambda : float
        GAE lambda (default 1.0, equivalent to full n-step returns).
    vf_coef : float
        Value function loss coefficient (default 0.5).
    ent_coef : float
        Entropy bonus coefficient (default 0.01).
    max_grad_norm : float
        Gradient clipping threshold (default 0.5).
    normalize_advantages : bool
        Normalize advantages per batch (default False).
    n_envs : int
        Number of parallel environments (default 8).
    hidden : int
        Hidden layer width (default 64).
    """

    learning_rate: float = 7e-4
    n_steps: int = 5
    gamma: float = 0.99
    gae_lambda: float = 1.0
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    normalize_advantages: bool = False
    n_envs: int = 8
    hidden: int = 64

    def __post_init__(self):
        _validate_positive("learning_rate", self.learning_rate)
        _validate_min("n_steps", self.n_steps, 1)
        _validate_min("n_envs", self.n_envs, 1)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> A2CConfig:
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def from_yaml(cls, path: str | Path) -> A2CConfig:
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    @classmethod
    def from_toml(cls, path: str | Path) -> A2CConfig:
        return cls.from_dict(_load_toml(path))

    def merge(self, overrides: dict[str, Any]) -> A2CConfig:
        d = asdict(self)
        d.update(overrides)
        return A2CConfig.from_dict(d)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        import yaml

        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def to_toml(self, path: str | Path) -> None:
        _write_toml(self.to_dict(), path)


@dataclass
class TD3Config:
    """Configuration for TD3 (Twin Delayed DDPG) training.

    Deterministic policy with target policy smoothing and delayed updates.

    Attributes
    ----------
    learning_rate : float
        Adam learning rate for both actor and critic (default 3e-4).
    buffer_size : int
        Replay buffer capacity (default 1M).
    batch_size : int
        Minibatch size (default 256).
    tau : float
        Polyak averaging coefficient for target networks (default 0.005).
    gamma : float
        Discount factor (default 0.99).
    learning_starts : int
        Random exploration steps before training (default 1000).
    policy_delay : int
        Actor update frequency relative to critic (default 2).
    target_noise : float
        Noise added to target actions for smoothing (default 0.2).
    noise_clip : float
        Clipping range for target noise (default 0.5).
    exploration_noise : float
        Std of Gaussian exploration noise (default 0.1).
    hidden : int
        Hidden layer width (default 256).
    n_envs : int
        Number of parallel environments (default 1).
    """

    learning_rate: float = 3e-4
    buffer_size: int = 1_000_000
    batch_size: int = 256
    tau: float = 0.005
    gamma: float = 0.99
    learning_starts: int = 1000
    policy_delay: int = 2
    target_noise: float = 0.2
    noise_clip: float = 0.5
    exploration_noise: float = 0.1
    hidden: int = 256
    n_envs: int = 1

    def __post_init__(self):
        _validate_positive("learning_rate", self.learning_rate)
        _validate_min("buffer_size", self.buffer_size, 1)
        _validate_min("batch_size", self.batch_size, 1)
        _validate_min("policy_delay", self.policy_delay, 1)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TD3Config:
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def from_yaml(cls, path: str | Path) -> TD3Config:
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    @classmethod
    def from_toml(cls, path: str | Path) -> TD3Config:
        return cls.from_dict(_load_toml(path))

    def merge(self, overrides: dict[str, Any]) -> TD3Config:
        d = asdict(self)
        d.update(overrides)
        return TD3Config.from_dict(d)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        import yaml

        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def to_toml(self, path: str | Path) -> None:
        _write_toml(self.to_dict(), path)


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

    @classmethod
    def from_toml(cls, path: str | Path) -> DQNConfig:
        """Load config from a TOML file, ignoring unknown keys."""
        return cls.from_dict(_load_toml(path))

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

    def to_toml(self, path: str | Path) -> None:
        """Save config to a TOML file."""
        _write_toml(self.to_dict(), path)


# ---------------------------------------------------------------------------
# Top-level training config (Layer 2)
# ---------------------------------------------------------------------------

_VALID_ALGORITHMS = {"ppo", "sac", "dqn", "td3", "a2c"}
_VALID_LOGGERS = {"tensorboard", "wandb", "console", None}
_VALID_CALLBACKS = {"eval", "checkpoint", "progress", "timing", "early_stopping"}


@dataclass
class TrainingConfig:
    """Top-level config for YAML/TOML-driven training.

    Combines algorithm selection, environment, seed, hyperparameters,
    callbacks, and logging into a single serializable config.
    """

    algorithm: str  # "ppo", "sac", "dqn", "td3", "a2c"
    env_id: str
    total_timesteps: int = 100_000
    seed: int = 42
    n_envs: int = 1
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    callbacks: list[str] = field(default_factory=list)
    logger: str | None = None  # "tensorboard", "wandb", "console"
    log_dir: str | None = None
    eval_freq: int = 10_000
    eval_episodes: int = 10
    checkpoint_freq: int = 50_000
    checkpoint_dir: str = "checkpoints"
    normalize_obs: bool = False
    normalize_rewards: bool = False

    def __post_init__(self) -> None:
        self.algorithm = self.algorithm.lower()
        if self.algorithm not in _VALID_ALGORITHMS:
            raise ValueError(
                f"Unknown algorithm {self.algorithm!r}, "
                f"expected one of {sorted(_VALID_ALGORITHMS)}"
            )
        _validate_min("total_timesteps", self.total_timesteps, 1)
        _validate_min("n_envs", self.n_envs, 1)

    # -- Constructors --------------------------------------------------------

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrainingConfig:
        """Construct from a dict, ignoring unknown keys."""
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        # Normalise None-sentinels from TOML (empty strings)
        for key in ("logger", "log_dir"):
            if key in filtered and filtered[key] == "":
                filtered[key] = None
        return cls(**filtered)

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainingConfig:
        """Load from a YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    @classmethod
    def from_toml(cls, path: str | Path) -> TrainingConfig:
        """Load from a TOML file."""
        return cls.from_dict(_load_toml(path))

    # -- Serialisation -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        """Save to a YAML file."""
        import yaml

        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def to_toml(self, path: str | Path) -> None:
        """Save to a TOML file."""
        _write_toml(self.to_dict(), path)
