"""Protocol interfaces for rlox components.

Protocols define the structural contracts that custom components must satisfy.
Users can implement these with any class — no inheritance required.

Example
-------
>>> class MyCNNPolicy:
...     def get_action_and_logprob(self, obs):
...         # custom CNN forward
...         return actions, log_probs
...     def get_value(self, obs):
...         return values
...     def get_logprob_and_entropy(self, obs, actions):
...         return log_probs, entropy
...
>>> # Works with PPO because it satisfies OnPolicyActor
>>> ppo = PPO(env_id="CartPole-v1", policy=MyCNNPolicy())
"""

from __future__ import annotations

import sys
from typing import Any, Protocol, runtime_checkable

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing import TypeVar
    Self = TypeVar("Self")

import numpy as np
import torch


@runtime_checkable
class Algorithm(Protocol):
    """Protocol that all rlox algorithms must satisfy.

    Any class with ``env_id``, ``train()``, ``save()``, and
    ``from_checkpoint()`` can be used as an algorithm with the unified
    :class:`~rlox.trainer.Trainer`.
    """

    env_id: str

    def train(self, total_timesteps: int) -> dict[str, float]:
        """Run training and return metrics."""
        ...

    def save(self, path: str) -> None:
        """Save checkpoint."""
        ...

    @classmethod
    def from_checkpoint(cls, path: str, env_id: str | None = None) -> Self:
        """Restore from checkpoint."""
        ...


@runtime_checkable
class OnPolicyActor(Protocol):
    """Protocol for on-policy actor-critics (PPO, A2C).

    Any nn.Module implementing these three methods can be used as a PPO/A2C policy.
    """

    def get_action_and_logprob(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample actions and compute log-probabilities.

        Parameters
        ----------
        obs : (B, obs_dim) tensor

        Returns
        -------
        actions : (B,) or (B, act_dim) tensor
        log_probs : (B,) tensor
        """
        ...

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute value estimates.

        Parameters
        ----------
        obs : (B, obs_dim) tensor

        Returns
        -------
        values : (B,) tensor
        """
        ...

    def get_logprob_and_entropy(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log-probability and entropy for given actions.

        Parameters
        ----------
        obs : (B, obs_dim) tensor
        actions : (B,) or (B, act_dim) tensor

        Returns
        -------
        log_probs : (B,) tensor
        entropy : (B,) tensor
        """
        ...


@runtime_checkable
class StochasticActor(Protocol):
    """Protocol for stochastic actors (SAC).

    Any nn.Module implementing sample() and deterministic() can be used as a SAC actor.
    """

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample actions with reparameterization.

        Returns
        -------
        actions : (B, act_dim) tensor — in [-1, 1] (pre-scaling)
        log_probs : (B,) tensor
        """
        ...

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        """Return deterministic (mean) actions.

        Returns
        -------
        actions : (B, act_dim) tensor
        """
        ...


@runtime_checkable
class DeterministicActor(Protocol):
    """Protocol for deterministic actors (TD3).

    Any nn.Module implementing forward() can be used as a TD3 actor.
    """

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return deterministic actions.

        Returns
        -------
        actions : (B, act_dim) tensor
        """
        ...


@runtime_checkable
class QFunction(Protocol):
    """Protocol for Q-value networks (SAC, TD3 critics).

    Takes (obs, action) and returns scalar Q-value.
    """

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute Q-value.

        Returns
        -------
        q_value : (B, 1) tensor
        """
        ...


@runtime_checkable
class DiscreteQFunction(Protocol):
    """Protocol for discrete Q-networks (DQN).

    Takes obs and returns Q-values for all actions.
    """

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute Q-values for all actions.

        Returns
        -------
        q_values : (B, n_actions) tensor
        """
        ...


@runtime_checkable
class ExplorationStrategy(Protocol):
    """Protocol for exploration strategies.

    Used by off-policy algorithms to add noise to actions.
    """

    def select_action(
        self, action: np.ndarray, step: int, total_steps: int
    ) -> np.ndarray:
        """Add exploration noise to an action.

        Parameters
        ----------
        action : raw action from the policy
        step : current training step
        total_steps : total training steps

        Returns
        -------
        noisy_action : action with exploration noise
        """
        ...

    def reset(self) -> None:
        """Reset any internal state (e.g., OU noise)."""
        ...


@runtime_checkable
class VecEnv(Protocol):
    """Protocol for vectorized environments.

    Any object implementing step_all/reset_all/num_envs can be used
    as a vectorized environment (GymVecEnv, VecNormalize, etc.).
    """

    def step_all(self, actions: np.ndarray) -> dict[str, Any]:
        """Step all sub-environments.

        Returns
        -------
        dict with keys ``obs``, ``rewards``, ``terminated``, ``truncated``,
        ``terminal_obs``.
        """
        ...

    def reset_all(self, **kwargs: Any) -> np.ndarray:
        """Reset all sub-environments.

        Returns
        -------
        np.ndarray of shape ``(n_envs, obs_dim)``.
        """
        ...

    def num_envs(self) -> int:
        """Return the number of parallel sub-environments."""
        ...


@runtime_checkable
class ReplayBufferProtocol(Protocol):
    """Protocol for replay buffers.

    Any buffer implementing push/sample/len can be used with off-policy algorithms.
    """

    def push(self, *args: Any, **kwargs: Any) -> None:
        """Store a transition."""
        ...

    def sample(self, batch_size: int, seed: int) -> dict[str, Any]:
        """Sample a batch of transitions."""
        ...

    def __len__(self) -> int:
        """Return number of stored transitions."""
        ...


@runtime_checkable
class Augmentation(Protocol):
    """Protocol for observation augmentation transforms.

    Any callable with a ``pad`` attribute and ``__call__(obs, seed)`` works.
    """

    pad: int

    def __call__(self, obs: torch.Tensor, seed: int) -> torch.Tensor:
        """Augment a batch of observations.

        Parameters
        ----------
        obs : (B, C, H, W) tensor
        seed : RNG seed for reproducibility

        Returns
        -------
        augmented : (B, C, H, W) tensor
        """
        ...


@runtime_checkable
class RewardShaper(Protocol):
    """Protocol for reward shaping transforms."""

    def shape(
        self,
        rewards: np.ndarray,
        obs: np.ndarray,
        next_obs: np.ndarray,
        dones: np.ndarray,
    ) -> np.ndarray:
        """Compute shaped rewards.

        Parameters
        ----------
        rewards : (N,) base rewards
        obs : (N, obs_dim) current observations
        next_obs : (N, obs_dim) next observations
        dones : (N,) episode termination flags

        Returns
        -------
        shaped_rewards : (N,) array
        """
        ...


@runtime_checkable
class IntrinsicMotivation(Protocol):
    """Protocol for intrinsic motivation modules (RND, ICM, etc.).

    RND only needs ``obs``; ICM needs ``(obs, next_obs, actions)``.
    Both signatures are valid -- extra args are passed through.
    """

    def compute_intrinsic_reward(
        self, obs: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        """Compute intrinsic reward for a batch of observations.

        Parameters
        ----------
        obs : (B, obs_dim) tensor
        *args : additional tensors (next_obs, actions for ICM)

        Returns
        -------
        intrinsic_reward : (B,) tensor
        """
        ...

    def update(self, obs: torch.Tensor, *args: Any, **kwargs: Any) -> dict[str, float]:
        """Update the intrinsic motivation module.

        Parameters
        ----------
        obs : (B, obs_dim) tensor
        *args : additional tensors (next_obs, actions for ICM)

        Returns
        -------
        info : dict with loss values
        """
        ...


@runtime_checkable
class MetaLearner(Protocol):
    """Protocol for meta-learning outer loops (Reptile, MAML, etc.)."""

    def meta_train(self, n_iterations: int) -> dict[str, float]:
        """Run the meta-training loop.

        Parameters
        ----------
        n_iterations : number of outer-loop iterations

        Returns
        -------
        metrics : dict with training metrics
        """
        ...

    def adapt(self, env_id: str, n_steps: int) -> Algorithm:
        """Adapt meta-learned weights to a specific task.

        Parameters
        ----------
        env_id : gymnasium environment ID
        n_steps : number of adaptation steps

        Returns
        -------
        algorithm : adapted Algorithm instance
        """
        ...
