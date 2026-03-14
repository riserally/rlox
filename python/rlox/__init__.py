"""rlox — Rust-accelerated reinforcement learning.

The Polars architecture pattern applied to RL: Rust data plane for
environments, buffers, and advantage computation; Python control plane
for training logic, policies, and neural networks.

Rust primitives (via PyO3)
--------------------------
- :class:`CartPole`, :class:`VecEnv`, :class:`GymEnv` — environment stepping
- :class:`ExperienceTable`, :class:`ReplayBuffer`, :class:`PrioritizedReplayBuffer` — storage
- :func:`compute_gae`, :func:`compute_vtrace` — advantage estimation
- :func:`compute_group_advantages`, :func:`compute_token_kl` — LLM post-training
- :class:`DPOPair`, :class:`VarLenStore`, :func:`pack_sequences` — sequence handling
- :class:`RunningStats` — online mean/variance

Python layer
------------
- :class:`RolloutBatch` — flat-tensor container for on-policy data
- :class:`RolloutCollector` — VecEnv + GAE rollout collection
- :class:`PPOLoss` — clipped PPO objective

For algorithm implementations, see :mod:`rlox.algorithms`.
For high-level trainers, see :mod:`rlox.trainers`.

Quick start::

    import rlox
    from rlox.trainers import PPOTrainer

    trainer = PPOTrainer(env="CartPole-v1", seed=42)
    metrics = trainer.train(total_timesteps=50_000)
    print(f"Mean reward: {metrics['mean_reward']:.1f}")
"""

from rlox._rlox_core import (
    CartPole,
    VecEnv,
    GymEnv,
    ExperienceTable,
    ReplayBuffer,
    PrioritizedReplayBuffer,
    VarLenStore,
    compute_gae,
    compute_vtrace,
    compute_group_advantages,
    compute_token_kl,
    DPOPair,
    RunningStats,
    pack_sequences,
)

from rlox.batch import RolloutBatch
from rlox.collectors import RolloutCollector
from rlox.losses import PPOLoss

__version__ = "1.0.0"

__all__ = [
    # Rust primitives
    "CartPole",
    "VecEnv",
    "GymEnv",
    "ExperienceTable",
    "ReplayBuffer",
    "PrioritizedReplayBuffer",
    "VarLenStore",
    "compute_gae",
    "compute_vtrace",
    "compute_group_advantages",
    "compute_token_kl",
    "DPOPair",
    "RunningStats",
    "pack_sequences",
    # Python Layer 1
    "RolloutBatch",
    "RolloutCollector",
    "PPOLoss",
]
