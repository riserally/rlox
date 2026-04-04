"""rlox -- Rust-accelerated reinforcement learning (v1.0.0).

The Polars architecture pattern applied to RL: Rust data plane for
environments, buffers, and advantage computation; Python control plane
for training logic, policies, and neural networks.

8 algorithms, 8 trainers, config-driven training, diagnostics dashboard.

Rust primitives (via PyO3)
--------------------------
- :class:`CartPole`, :class:`VecEnv`, :class:`GymEnv` -- environment stepping
- :class:`ExperienceTable`, :class:`ReplayBuffer`, :class:`PrioritizedReplayBuffer`,
  :class:`MmapReplayBuffer`, :class:`OfflineDatasetBuffer` -- storage
- :func:`compute_gae`, :func:`compute_vtrace` -- advantage estimation
- :func:`compute_group_advantages`, :func:`compute_token_kl` -- LLM post-training
- :class:`DPOPair`, :class:`VarLenStore`, :func:`pack_sequences` -- sequence handling
- :class:`RunningStats`, :class:`RunningStatsVec` -- online mean/variance
- :class:`ActorCritic`, :class:`CandleCollector` -- NN backend

Python layer
------------
- :class:`VecNormalize` -- obs/reward normalization wrapper (SB3-compatible)
- :class:`RolloutBatch` -- flat-tensor container for on-policy data
- :class:`RolloutCollector`, :class:`OffPolicyCollector` -- data collection
- :class:`GymVecEnv` -- gymnasium wrapper with VecEnv interface
- :class:`ContinuousPolicy`, :class:`DiscretePolicy` -- default policy networks
- :class:`TrainingConfig` -- YAML/TOML config-driven training
- :class:`TerminalDashboard`, :class:`HTMLReport` -- diagnostics dashboard

Unified Trainer
---------------
- :class:`Trainer` -- single entry point for all 8 algorithms
- Algorithms: ``ppo``, ``sac``, ``dqn``, ``td3``, ``a2c``, ``mappo``, ``dreamer``, ``impala``
- :data:`ALGORITHM_REGISTRY` -- algorithm name -> class mapping

All algorithms expose ``train(total_timesteps)``, ``save(path)``, and
``from_checkpoint(path)`` via the unified :class:`Trainer`.

For algorithm implementations, see :mod:`rlox.algorithms`.
For config-driven training, see :func:`train_from_config`.

Quick start::

    from rlox import Trainer

    trainer = Trainer("ppo", env="CartPole-v1", seed=42)
    metrics = trainer.train(total_timesteps=50_000)
    print(f"Mean reward: {metrics['mean_reward']:.1f}")

Config-driven::

    from rlox import TrainingConfig, train_from_config

    config = TrainingConfig.from_yaml("config.yaml")
    metrics = train_from_config(config)
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
    # Wave 2/3: new Rust bindings
    random_shift_batch,
    shape_rewards_pbrs,
    compute_goal_distance_potentials,
    reptile_update,
    average_weight_vectors,
    SequenceReplayBuffer,
    HERBuffer,
    py_sample_mixed,
)

# -- Python Layer 1 -----------------------------------------------------------
from rlox.batch import RolloutBatch
from rlox.collectors import RolloutCollector
from rlox.gym_vec_env import GymVecEnv
from rlox.losses import PPOLoss
from rlox.policies import ContinuousPolicy, DiscretePolicy
from rlox.vec_normalize import VecNormalize

# -- Configs -------------------------------------------------------------------
from rlox.config import (
    PPOConfig,
    SACConfig,
    DQNConfig,
    A2CConfig,
    TD3Config,
    MAPPOConfig,
    DreamerV3Config,
    IMPALAConfig,
    DecisionTransformerConfig,
    QMIXConfig,
    CalQLConfig,
    PBTConfig,
    TRPOConfig,
    SelfPlayConfig,
    GoExploreConfig,
    MPOConfig,
    TrainingConfig,
)

# -- Trainers ------------------------------------------------------------------
from rlox.trainers import (
    PPOTrainer,
    SACTrainer,
    DQNTrainer,
    A2CTrainer,
    TD3Trainer,
    MAPPOTrainer,
    DreamerV3Trainer,
    IMPALATrainer,
)

# -- Unified Trainer -----------------------------------------------------------
from rlox.trainer import Trainer, ALGORITHM_REGISTRY

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
    Augmentation,
    RewardShaper,
    IntrinsicMotivation,
    MetaLearner,
)

# -- Wrappers (Visual RL, Language-Conditioned) --------------------------------
from rlox.wrappers import (
    FrameStack,
    ImagePreprocess,
    AtariWrapper,
    DMControlWrapper,
    LanguageWrapper,
    GoalConditionedWrapper,
)

# -- Deploy (Docker, K8s, SageMaker) ------------------------------------------
from rlox.deploy import generate_dockerfile, generate_k8s_job, SageMakerEstimator

# -- Wave 4: Python wrappers --------------------------------------------------
from rlox.augmentation import RandomShift
from rlox.reward_shaping import PotentialShaping, GoalDistanceShaping
from rlox.networks import apply_spectral_norm
from rlox.intrinsic import RND, ICM
from rlox.meta import Reptile
from rlox.offline_to_online import OfflineToOnline

# -- Population-Based Training -------------------------------------------------
from rlox.pbt import PBT

# -- Self-Play -----------------------------------------------------------------
from rlox.self_play import SelfPlay

# -- Exploration ---------------------------------------------------------------
from rlox.exploration import GaussianNoise, EpsilonGreedy, OUNoise, GoExplore

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

# -- Plugins -------------------------------------------------------------------
from rlox.plugins import (
    ENV_REGISTRY,
    BUFFER_REGISTRY,
    REWARD_REGISTRY,
    register_env,
    register_buffer,
    register_reward,
    discover_plugins,
    list_registered,
)

# -- Model Zoo -----------------------------------------------------------------
from rlox.zoo import ModelZoo, ModelCard

# -- Compile -------------------------------------------------------------------
from rlox.compile import compile_policy

# -- Distributed ---------------------------------------------------------------
from rlox.distributed import MultiGPUTrainer, RemoteEnvPool, launch_elastic

__version__ = "1.0.0"

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
    # Wave 2/3 Rust bindings
    "random_shift_batch",
    "shape_rewards_pbrs",
    "compute_goal_distance_potentials",
    "reptile_update",
    "average_weight_vectors",
    "SequenceReplayBuffer",
    "HERBuffer",
    "py_sample_mixed",
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
    "MAPPOConfig",
    "DreamerV3Config",
    "IMPALAConfig",
    "DecisionTransformerConfig",
    "QMIXConfig",
    "CalQLConfig",
    "PBTConfig",
    "TRPOConfig",
    "SelfPlayConfig",
    "GoExploreConfig",
    "MPOConfig",
    "TrainingConfig",
    # Unified Trainer
    "Trainer",
    "ALGORITHM_REGISTRY",
    # Runner
    "train_from_config",
    # Trainers
    "PPOTrainer",
    "SACTrainer",
    "DQNTrainer",
    "A2CTrainer",
    "TD3Trainer",
    "MAPPOTrainer",
    "DreamerV3Trainer",
    "IMPALATrainer",
    # Callbacks
    "Callback",
    "CallbackList",
    "EvalCallback",
    "EarlyStoppingCallback",
    "CheckpointCallback",
    "ProgressBarCallback",
    "TimingCallback",
    # Protocols
    "OnPolicyActor",
    "StochasticActor",
    "DeterministicActor",
    "QFunction",
    "DiscreteQFunction",
    "ExplorationStrategy",
    "ReplayBufferProtocol",
    "VecEnvProtocol",
    "Augmentation",
    "RewardShaper",
    "IntrinsicMotivation",
    "MetaLearner",
    # Wave 4 Python wrappers
    "RandomShift",
    "PotentialShaping",
    "GoalDistanceShaping",
    "apply_spectral_norm",
    "RND",
    "ICM",
    "Reptile",
    "OfflineToOnline",
    "PBT",
    "SelfPlay",
    # Exploration
    "GaussianNoise",
    "EpsilonGreedy",
    "OUNoise",
    "GoExplore",
    # Off-policy collectors
    "OffPolicyCollector",
    "CollectorProtocol",
    # Builders
    "PPOBuilder",
    "SACBuilder",
    "DQNBuilder",
    # Losses
    "LossComponent",
    "CompositeLoss",
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
    # Plugins
    "ENV_REGISTRY",
    "BUFFER_REGISTRY",
    "REWARD_REGISTRY",
    "register_env",
    "register_buffer",
    "register_reward",
    "discover_plugins",
    "list_registered",
    # Model Zoo
    "ModelZoo",
    "ModelCard",
    # Compile
    "compile_policy",
    # Distributed
    "MultiGPUTrainer",
    "RemoteEnvPool",
    "launch_elastic",
    # Wrappers (Visual RL, Language-Conditioned)
    "FrameStack",
    "ImagePreprocess",
    "AtariWrapper",
    "DMControlWrapper",
    "LanguageWrapper",
    "GoalConditionedWrapper",
    # Deploy
    "generate_dockerfile",
    "generate_k8s_job",
    "SageMakerEstimator",
]
