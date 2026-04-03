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


class ConfigMixin:
    """Shared serialization/deserialization for all config dataclasses.

    Provides ``from_dict``, ``from_yaml``, ``from_toml``, ``merge``,
    ``to_dict``, ``to_yaml``, and ``to_toml`` so individual config classes
    only need to declare their fields and ``__post_init__`` validation.
    """

    @classmethod
    def from_dict(cls, d: dict[str, Any]):
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def from_yaml(cls, path: str | Path):
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    @classmethod
    def from_toml(cls, path: str | Path):
        return cls.from_dict(_load_toml(path))

    def merge(self, overrides: dict[str, Any]):
        d = asdict(self)
        d.update(overrides)
        return type(self).from_dict(d)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        import yaml

        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def to_toml(self, path: str | Path) -> None:
        _write_toml(self.to_dict(), path)


@dataclass
class PPOConfig(ConfigMixin):
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


@dataclass
class SACConfig(ConfigMixin):
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
    def from_toml(cls, path: str | Path) -> SACConfig:
        """Load config from a TOML file, ignoring unknown keys."""
        data = _load_toml(path)
        # TOML has no null -- empty string is our sentinel for None
        if data.get("target_entropy") == "":
            data["target_entropy"] = None
        return cls.from_dict(data)


@dataclass
class A2CConfig(ConfigMixin):
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


@dataclass
class TD3Config(ConfigMixin):
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


@dataclass
class DQNConfig(ConfigMixin):
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


# ---------------------------------------------------------------------------
# Top-level training config (Layer 2)
# ---------------------------------------------------------------------------

@dataclass
class MAPPOConfig(ConfigMixin):
    """Configuration for Multi-Agent PPO (MAPPO) training.

    Centralized training with decentralized execution (CTDE).

    Attributes
    ----------
    n_agents : int
        Number of agents (default 2).
    learning_rate : float
        Adam learning rate (default 5e-4).
    n_steps : int
        Rollout length per environment per update (default 128).
    n_epochs : int
        Number of SGD passes per rollout (default 5).
    clip_range : float
        PPO clipping range (default 0.2).
    gamma : float
        Discount factor (default 0.99).
    gae_lambda : float
        GAE lambda (default 0.95).
    vf_coef : float
        Value loss coefficient (default 0.5).
    ent_coef : float
        Entropy bonus coefficient (default 0.01).
    max_grad_norm : float
        Maximum gradient norm for clipping (default 10.0).
    share_parameters : bool
        Whether agents share actor parameters (default False).
    hidden : int
        Hidden layer width (default 64).
    n_envs : int
        Number of parallel environments (default 8).
    """

    n_agents: int = 2
    learning_rate: float = 5e-4
    n_steps: int = 128
    n_epochs: int = 5
    clip_range: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 10.0
    share_parameters: bool = False
    hidden: int = 64
    n_envs: int = 8

    def __post_init__(self):
        _validate_positive("learning_rate", self.learning_rate)
        _validate_min("n_agents", self.n_agents, 1)
        _validate_min("n_steps", self.n_steps, 1)
        _validate_min("n_epochs", self.n_epochs, 1)
        _validate_min("n_envs", self.n_envs, 1)


@dataclass
class DreamerV3Config(ConfigMixin):
    """Configuration for DreamerV3 (world-model-based RL) training.

    Attributes
    ----------
    learning_rate : float
        Learning rate for all optimisers (default 1e-4).
    buffer_size : int
        Replay buffer capacity (default 1M).
    batch_size : int
        Number of sequences per training batch (default 16).
    seq_len : int
        Sequence length for training (default 50).
    gamma : float
        Discount factor (default 0.997).
    lambda_ : float
        Lambda for lambda-returns (default 0.95).
    deter_dim : int
        Deterministic state dimension in RSSM (default 512).
    stoch_dim : int
        Stochastic state dimension (number of categoricals, default 32).
    stoch_classes : int
        Number of classes per categorical (default 32).
    hidden : int
        Hidden layer width (default 512).
    imagination_horizon : int
        Steps to imagine ahead for actor-critic (default 15).
    kl_balance : float
        KL balancing coefficient (default 0.8).
    free_nats : float
        Free nats for KL loss (default 1.0).
    """

    learning_rate: float = 1e-4
    buffer_size: int = 1_000_000
    batch_size: int = 16
    seq_len: int = 50
    gamma: float = 0.997
    lambda_: float = 0.95
    deter_dim: int = 512
    stoch_dim: int = 32
    stoch_classes: int = 32
    hidden: int = 512
    imagination_horizon: int = 15
    kl_balance: float = 0.8
    free_nats: float = 1.0

    def __post_init__(self):
        _validate_positive("learning_rate", self.learning_rate)
        _validate_min("buffer_size", self.buffer_size, 1)
        _validate_min("batch_size", self.batch_size, 1)
        _validate_min("seq_len", self.seq_len, 1)
        _validate_min("deter_dim", self.deter_dim, 1)
        _validate_min("stoch_dim", self.stoch_dim, 1)
        _validate_min("stoch_classes", self.stoch_classes, 1)
        _validate_min("imagination_horizon", self.imagination_horizon, 1)


@dataclass
class IMPALAConfig(ConfigMixin):
    """Configuration for IMPALA (Importance Weighted Actor-Learner Architecture).

    Attributes
    ----------
    learning_rate : float
        RMSprop learning rate (default 4e-4).
    n_actors : int
        Number of actor threads (default 4).
    n_steps : int
        Rollout length per actor per batch (default 20).
    gamma : float
        Discount factor (default 0.99).
    vf_coef : float
        Value loss coefficient (default 0.5).
    ent_coef : float
        Entropy bonus coefficient (default 0.01).
    max_grad_norm : float
        Maximum gradient norm for clipping (default 40.0).
    rho_clip : float
        V-trace truncation for importance weights (default 1.0).
    c_clip : float
        V-trace truncation for trace coefficients (default 1.0).
    queue_size : int
        Maximum experience queue size (default 16).
    hidden : int
        Hidden layer width (default 256).
    n_envs_per_actor : int
        Number of environments per actor thread (default 1).
    """

    learning_rate: float = 4e-4
    n_actors: int = 4
    n_steps: int = 20
    gamma: float = 0.99
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 40.0
    rho_clip: float = 1.0
    c_clip: float = 1.0
    queue_size: int = 16
    hidden: int = 256
    n_envs_per_actor: int = 1

    def __post_init__(self):
        _validate_positive("learning_rate", self.learning_rate)
        _validate_min("n_actors", self.n_actors, 1)
        _validate_min("n_steps", self.n_steps, 1)
        _validate_min("queue_size", self.queue_size, 1)
        _validate_min("n_envs_per_actor", self.n_envs_per_actor, 1)


@dataclass
class DecisionTransformerConfig(ConfigMixin):
    """Configuration for Decision Transformer training.

    Attributes
    ----------
    context_length : int
        Number of timesteps in the context window (default 20).
    n_heads : int
        Number of attention heads (default 4).
    n_layers : int
        Number of transformer layers (default 3).
    embed_dim : int
        Embedding dimension (default 128).
    learning_rate : float
        Adam learning rate (default 1e-4).
    batch_size : int
        Minibatch size for offline training (default 64).
    dropout : float
        Dropout rate (default 0.1).
    target_return : float
        Desired return for evaluation (default 200.0).
    warmup_steps : int
        Data collection steps before training (default 500).
    """

    context_length: int = 20
    n_heads: int = 4
    n_layers: int = 3
    embed_dim: int = 128
    learning_rate: float = 1e-4
    batch_size: int = 64
    dropout: float = 0.1
    target_return: float = 200.0
    warmup_steps: int = 500

    def __post_init__(self):
        _validate_positive("learning_rate", self.learning_rate)
        _validate_min("context_length", self.context_length, 1)
        _validate_min("n_heads", self.n_heads, 1)
        _validate_min("n_layers", self.n_layers, 1)
        _validate_min("embed_dim", self.embed_dim, 1)
        _validate_min("batch_size", self.batch_size, 1)


@dataclass
class QMIXConfig(ConfigMixin):
    """Configuration for QMIX training.

    Attributes
    ----------
    n_agents : int
        Number of agents (default 3).
    hidden_dim : int
        Hidden dimension for agent Q-networks (default 64).
    mixing_embed_dim : int
        Mixing network hidden dimension (default 32).
    lr : float
        Adam learning rate (default 5e-4).
    buffer_size : int
        Replay buffer capacity (default 50_000).
    gamma : float
        Discount factor (default 0.99).
    target_update_freq : int
        Steps between target network updates (default 200).
    batch_size : int
        Minibatch size (default 32).
    """

    n_agents: int = 3
    hidden_dim: int = 64
    mixing_embed_dim: int = 32
    lr: float = 5e-4
    buffer_size: int = 50_000
    gamma: float = 0.99
    target_update_freq: int = 200
    batch_size: int = 32

    def __post_init__(self):
        _validate_positive("lr", self.lr)
        _validate_min("n_agents", self.n_agents, 1)
        _validate_min("hidden_dim", self.hidden_dim, 1)
        _validate_min("mixing_embed_dim", self.mixing_embed_dim, 1)
        _validate_min("buffer_size", self.buffer_size, 1)
        _validate_min("target_update_freq", self.target_update_freq, 1)
        _validate_min("batch_size", self.batch_size, 1)


@dataclass
class CalQLConfig(ConfigMixin):
    """Configuration for Cal-QL (Calibrated Conservative Q-Learning).

    Attributes
    ----------
    learning_rate : float
        Learning rate for all optimisers (default 3e-4).
    buffer_size : int
        Replay buffer capacity (default 100_000).
    batch_size : int
        Minibatch size (default 256).
    gamma : float
        Discount factor (default 0.99).
    tau : float
        Polyak averaging coefficient (default 0.005).
    cql_alpha : float
        CQL penalty weight (default 5.0).
    calibration_tau : float
        Quantile for calibration threshold (default 0.5).
    auto_alpha : bool
        Whether to auto-tune cql_alpha (default False).
    """

    learning_rate: float = 3e-4
    buffer_size: int = 100_000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    cql_alpha: float = 5.0
    calibration_tau: float = 0.5
    auto_alpha: bool = False

    def __post_init__(self):
        _validate_positive("learning_rate", self.learning_rate)
        _validate_min("buffer_size", self.buffer_size, 1)
        _validate_min("batch_size", self.batch_size, 1)


@dataclass
class PBTConfig(ConfigMixin):
    """Configuration for Population-Based Training.

    Attributes
    ----------
    population_size : int
        Number of agents in the population (default 8).
    interval : int
        Training timesteps between exploit/explore cycles (default 10_000).
    n_iterations : int
        Number of PBT iterations (default 20).
    exploit_fraction : float
        Bottom fraction of population replaced each cycle (default 0.2).
    perturb_factor : float
        Hyperparameter perturbation range (default 0.2).
    """

    population_size: int = 8
    interval: int = 10_000
    n_iterations: int = 20
    exploit_fraction: float = 0.2
    perturb_factor: float = 0.2

    def __post_init__(self):
        _validate_min("population_size", self.population_size, 2)
        _validate_min("interval", self.interval, 1)
        _validate_min("n_iterations", self.n_iterations, 1)


@dataclass
class TRPOConfig(ConfigMixin):
    """Configuration for Trust Region Policy Optimization.

    Attributes
    ----------
    max_kl : float
        Maximum KL divergence per update (default 0.01).
    damping : float
        Damping coefficient for Fisher vector product (default 0.1).
    cg_iters : int
        Conjugate gradient iterations (default 10).
    line_search_steps : int
        Backtracking line search steps (default 10).
    n_envs : int
        Number of parallel environments (default 8).
    n_steps : int
        Rollout length per environment per update (default 2048).
    gamma : float
        Discount factor (default 0.99).
    gae_lambda : float
        GAE lambda (default 0.97).
    vf_lr : float
        Value function learning rate (default 1e-3).
    vf_epochs : int
        Value function SGD epochs per update (default 5).
    """

    max_kl: float = 0.01
    damping: float = 0.1
    cg_iters: int = 10
    line_search_steps: int = 10
    n_envs: int = 8
    n_steps: int = 2048
    gamma: float = 0.99
    gae_lambda: float = 0.97
    vf_lr: float = 1e-3
    vf_epochs: int = 5

    def __post_init__(self):
        _validate_positive("max_kl", self.max_kl)
        _validate_positive("damping", self.damping)
        _validate_min("cg_iters", self.cg_iters, 1)
        _validate_min("line_search_steps", self.line_search_steps, 1)
        _validate_min("n_envs", self.n_envs, 1)
        _validate_min("n_steps", self.n_steps, 1)
        _validate_positive("vf_lr", self.vf_lr)
        _validate_min("vf_epochs", self.vf_epochs, 1)


_VALID_ALGORITHMS = {"ppo", "sac", "dqn", "td3", "a2c", "mappo", "dreamer", "impala", "dt", "qmix", "calql", "trpo"}
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
