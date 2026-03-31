"""rlox -- Rust-accelerated reinforcement learning.

The Polars architecture pattern applied to RL: Rust data plane for
environments, buffers, and advantage computation; Python control plane
for training logic, policies, and neural networks.

Rust primitives (via PyO3)
--------------------------
- :class:`CartPole`, :class:`VecEnv`, :class:`GymEnv` -- environment stepping
- :class:`ExperienceTable`, :class:`ReplayBuffer`, :class:`PrioritizedReplayBuffer` -- storage
- :func:`compute_gae`, :func:`compute_vtrace` -- advantage estimation
- :func:`compute_group_advantages`, :func:`compute_token_kl` -- LLM post-training
- :class:`DPOPair`, :class:`VarLenStore`, :func:`pack_sequences` -- sequence handling
- :class:`RunningStats` -- online mean/variance
- :class:`ActorCritic` -- NN backend

Python layer
------------
- :class:`RolloutBatch` -- flat-tensor container for on-policy data
- :class:`RolloutCollector` -- VecEnv + GAE rollout collection
- :class:`PPOLoss` -- clipped PPO objective
- :class:`GymVecEnv` -- gymnasium wrapper with VecEnv interface
- :class:`ContinuousPolicy`, :class:`DiscretePolicy` -- default policy networks

For algorithm implementations, see :mod:`rlox.algorithms`.
For high-level trainers, see :mod:`rlox.trainers`.

Quick start::

    import rlox
    from rlox.trainers import PPOTrainer

    trainer = PPOTrainer(env="CartPole-v1", seed=42)
    metrics = trainer.train(total_timesteps=50_000)
    print(f"Mean reward: {metrics['mean_reward']:.1f}")
"""

# -- Rust primitives (PyO3) ---------------------------------------------------
from rlox._rlox_core import (
    CartPole,
    VecEnv,
    GymEnv,
    ExperienceTable,
    ReplayBuffer,
    PrioritizedReplayBuffer,
    MmapReplayBuffer,
    OfflineDatasetBuffer,
    VarLenStore,
    compute_gae,
    compute_gae_batched,
    compute_gae_batched_f32,
    compute_vtrace,
    compute_group_advantages,
    compute_batch_group_advantages,
    compute_token_kl,
    compute_token_kl_schulman,
    compute_batch_token_kl,
    compute_batch_token_kl_schulman,
    compute_token_kl_f32,
    compute_token_kl_schulman_f32,
    compute_batch_token_kl_f32,
    compute_batch_token_kl_schulman_f32,
    DPOPair,
    RunningStats,
    RunningStatsVec,
    pack_sequences,
    ActorCritic,
    CandleCollector,
)

# -- Python Layer 1 -----------------------------------------------------------
from rlox.batch import RolloutBatch
from rlox.collectors import RolloutCollector
from rlox.gym_vec_env import GymVecEnv
from rlox.losses import PPOLoss
from rlox.policies import ContinuousPolicy, DiscretePolicy
from rlox.vec_normalize import VecNormalize

# -- Configs -------------------------------------------------------------------
from rlox.config import PPOConfig, SACConfig, DQNConfig, A2CConfig, TD3Config, TrainingConfig

# -- Runner (config-driven training) ------------------------------------------
from rlox.runner import train_from_config

# -- Callbacks -----------------------------------------------------------------
from rlox.callbacks import (
    Callback,
    CallbackList,
    EvalCallback,
    EarlyStoppingCallback,
    CheckpointCallback,
    ProgressBarCallback,
    TimingCallback,
)

# -- Protocols -----------------------------------------------------------------
from rlox.protocols import (
    OnPolicyActor,
    StochasticActor,
    DeterministicActor,
    QFunction,
    DiscreteQFunction,
    ExplorationStrategy,
    ReplayBufferProtocol,
    VecEnv as VecEnvProtocol,
)

# -- Exploration ---------------------------------------------------------------
from rlox.exploration import GaussianNoise, EpsilonGreedy, OUNoise

# -- Collectors ----------------------------------------------------------------
from rlox.off_policy_collector import OffPolicyCollector, CollectorProtocol

# -- Builders ------------------------------------------------------------------
from rlox.builders import PPOBuilder, SACBuilder, DQNBuilder

# -- Losses (composable) ------------------------------------------------------
from rlox.losses import LossComponent, CompositeLoss

# -- Logging -------------------------------------------------------------------
from rlox.logging import LoggerCallback, WandbLogger, TensorBoardLogger, ConsoleLogger

# -- Evaluation ----------------------------------------------------------------
from rlox.evaluation import (
    interquartile_mean,
    performance_profiles,
    stratified_bootstrap_ci,
    aggregate_metrics,
    probability_of_improvement,
)

# -- Diagnostics ---------------------------------------------------------------
from rlox.diagnostics import TrainingDiagnostics

# -- Dashboard -----------------------------------------------------------------
from rlox.dashboard import MetricsCollector, TerminalDashboard, HTMLReport

# -- Checkpoint ----------------------------------------------------------------
from rlox.checkpoint import Checkpoint

# -- Hub -----------------------------------------------------------------------
from rlox.hub import push_to_hub, load_from_hub

# -- Compile -------------------------------------------------------------------
from rlox.compile import compile_policy

__version__ = "0.3.0"

__all__ = [
    # Rust primitives
    "CartPole",
    "VecEnv",
    "GymEnv",
    "ExperienceTable",
    "ReplayBuffer",
    "PrioritizedReplayBuffer",
    "MmapReplayBuffer",
    "OfflineDatasetBuffer",
    "VarLenStore",
    "compute_gae",
    "compute_gae_batched",
    "compute_gae_batched_f32",
    "compute_vtrace",
    "compute_group_advantages",
    "compute_batch_group_advantages",
    "compute_token_kl",
    "compute_token_kl_schulman",
    "compute_batch_token_kl",
    "compute_batch_token_kl_schulman",
    "compute_token_kl_f32",
    "compute_token_kl_schulman_f32",
    "compute_batch_token_kl_f32",
    "compute_batch_token_kl_schulman_f32",
    "DPOPair",
    "RunningStats",
    "RunningStatsVec",
    "pack_sequences",
    "ActorCritic",
    "CandleCollector",
    # Python Layer 1
    "RolloutBatch",
    "RolloutCollector",
    "GymVecEnv",
    "VecNormalize",
    "PPOLoss",
    "ContinuousPolicy",
    "DiscretePolicy",
    # Configs
    "PPOConfig",
    "SACConfig",
    "DQNConfig",
    "A2CConfig",
    "TD3Config",
    "TrainingConfig",
    "train_from_config",
    # Callbacks
    "Callback",
    "CallbackList",
    "EvalCallback",
    "EarlyStoppingCallback",
    "CheckpointCallback",
    "ProgressBarCallback",
    "TimingCallback",
    # Logging
    "LoggerCallback",
    "WandbLogger",
    "TensorBoardLogger",
    "ConsoleLogger",
    # Evaluation
    "interquartile_mean",
    "performance_profiles",
    "stratified_bootstrap_ci",
    "aggregate_metrics",
    "probability_of_improvement",
    # Diagnostics
    "TrainingDiagnostics",
    # Dashboard
    "MetricsCollector",
    "TerminalDashboard",
    "HTMLReport",
    # Checkpoint
    "Checkpoint",
    # Hub
    "push_to_hub",
    "load_from_hub",
    # Protocols
    "VecEnvProtocol",
    # Compile
    "compile_policy",
]
