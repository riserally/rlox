"""Reptile meta-learning (Nichol et al., 2018).

Outer loop: for each iteration, sample a task (env), train an inner
algorithm for a few steps, then move the meta-parameters toward the
task-specific parameters using ``rlox.reptile_update`` (Rust-accelerated).
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
import torch.nn as nn

import rlox._rlox_core as _core
from rlox.trainer import ALGORITHM_REGISTRY


def _flatten_params(module: nn.Module) -> np.ndarray:
    """Flatten all parameters of a module into a single 1-D float32 array."""
    return np.concatenate(
        [p.detach().cpu().numpy().ravel() for p in module.parameters()]
    ).astype(np.float32)


def _load_flat_params(module: nn.Module, flat: np.ndarray) -> None:
    """Load a flat parameter vector back into a module."""
    offset = 0
    for p in module.parameters():
        numel = p.numel()
        p.data.copy_(
            torch.from_numpy(flat[offset : offset + numel].reshape(p.shape))
        )
        offset += numel


class Reptile:
    """Reptile meta-learning outer loop.

    Trains an inner algorithm on sampled tasks and averages the resulting
    weights back into the meta-parameters using Rust-accelerated updates.

    Parameters
    ----------
    algorithm_cls_name : str
        Registered algorithm name (e.g. ``"awr"``).
    env_ids : list[str]
        List of gymnasium environment IDs to sample tasks from.
    meta_lr : float
        Meta-learning rate for the outer Reptile update.
    inner_steps : int
        Number of training steps for the inner algorithm per task.
    inner_kwargs : dict, optional
        Extra kwargs passed to the inner algorithm constructor.
    """

    def __init__(
        self,
        algorithm_cls_name: str,
        env_ids: list[str],
        meta_lr: float = 0.1,
        inner_steps: int = 500,
        inner_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.algorithm_cls_name = algorithm_cls_name
        self.env_ids = env_ids
        self.meta_lr = meta_lr
        self.inner_steps = inner_steps
        self.inner_kwargs = inner_kwargs or {}

        # Look up the algorithm class
        self._algo_cls = ALGORITHM_REGISTRY[algorithm_cls_name.lower()]

        # Initialize meta-parameters from a reference algorithm
        ref = self._algo_cls(env_id=env_ids[0], **self.inner_kwargs)
        self._meta_actor_params = _flatten_params(ref.actor)
        self._meta_critic_params = _flatten_params(ref.critic)

        # Keep reference architecture for shape info
        self._ref_algo = ref

    def meta_train(self, n_iterations: int) -> dict[str, float]:
        """Run the Reptile meta-training loop.

        Parameters
        ----------
        n_iterations : int
            Number of outer-loop iterations.

        Returns
        -------
        metrics : dict with 'mean_reward' averaged across tasks
        """
        all_rewards: list[float] = []

        for iteration in range(n_iterations):
            # Sample a task
            env_id = self.env_ids[iteration % len(self.env_ids)]

            # Create inner algorithm and load meta-params
            inner_algo = self._algo_cls(env_id=env_id, **self.inner_kwargs)
            _load_flat_params(inner_algo.actor, self._meta_actor_params.copy())
            _load_flat_params(inner_algo.critic, self._meta_critic_params.copy())

            # Inner training
            metrics = inner_algo.train(total_timesteps=self.inner_steps)
            all_rewards.append(metrics.get("mean_reward", 0.0))

            # Reptile update: meta_params += lr * (task_params - meta_params)
            task_actor_params = _flatten_params(inner_algo.actor)
            task_critic_params = _flatten_params(inner_algo.critic)

            _core.reptile_update(self._meta_actor_params, task_actor_params, self.meta_lr)
            _core.reptile_update(self._meta_critic_params, task_critic_params, self.meta_lr)

        return {"mean_reward": float(np.mean(all_rewards)) if all_rewards else 0.0}

    def adapt(self, env_id: str, n_steps: int) -> Any:
        """Adapt meta-learned weights to a specific task.

        Creates a new algorithm instance, loads meta-params, and trains
        for ``n_steps`` on the given environment.

        Parameters
        ----------
        env_id : str
            Gymnasium environment ID.
        n_steps : int
            Number of adaptation steps.

        Returns
        -------
        algorithm : adapted Algorithm instance
        """
        algo = self._algo_cls(env_id=env_id, **self.inner_kwargs)
        _load_flat_params(algo.actor, self._meta_actor_params.copy())
        _load_flat_params(algo.critic, self._meta_critic_params.copy())

        algo.train(total_timesteps=n_steps)
        return algo
