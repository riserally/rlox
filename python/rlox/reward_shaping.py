"""Potential-based reward shaping (PBRS) backed by Rust.

Provides composable reward shaping that preserves optimal policy invariance
(Ng et al., 1999). The heavy arithmetic is done in Rust via
``rlox.shape_rewards_pbrs`` and ``rlox.compute_goal_distance_potentials``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

import rlox._rlox_core as _core


class PotentialShaping:
    """PBRS reward shaping with a configurable potential function.

    Computes: r' = r + gamma * Phi(s') * (1 - done) - Phi(s)

    The potential function can be a neural network, a distance function,
    or any callable ``(observations) -> potentials``.

    Parameters
    ----------
    potential_fn : callable
        Maps (N, obs_dim) ndarray -> (N,) ndarray of potential values.
    gamma : float
        Discount factor (default 0.99).
    """

    def __init__(self, potential_fn: Callable[[np.ndarray], np.ndarray], gamma: float = 0.99) -> None:
        self.potential_fn = potential_fn
        self.gamma = gamma

    def shape(
        self,
        rewards: np.ndarray,
        obs: np.ndarray,
        next_obs: np.ndarray,
        dones: np.ndarray,
    ) -> np.ndarray:
        """Compute PBRS shaped rewards.

        Parameters
        ----------
        rewards : (N,) base rewards
        obs : (N, obs_dim) current observations
        next_obs : (N, obs_dim) next observations
        dones : (N,) episode termination flags (1.0 = done)

        Returns
        -------
        shaped_rewards : (N,) float64 array
        """
        phi_current = np.ascontiguousarray(self.potential_fn(obs), dtype=np.float64)
        phi_next = np.ascontiguousarray(self.potential_fn(next_obs), dtype=np.float64)
        rewards = np.ascontiguousarray(rewards, dtype=np.float64)
        dones = np.ascontiguousarray(dones, dtype=np.float64)

        return _core.shape_rewards_pbrs(rewards, phi_current, phi_next, self.gamma, dones)


class GoalDistanceShaping:
    """Goal-distance potential shaping backed by Rust.

    Potential: Phi(s) = -scale * ||s[goal_slice] - goal||_2

    Moving closer to the goal increases the potential, producing a
    positive shaping bonus. Moving farther away decreases it.

    Parameters
    ----------
    goal : (goal_dim,) target goal vector
    obs_dim : total observation dimensionality
    goal_start : index where goal-relevant dims start in obs
    goal_dim : number of goal-relevant dimensions
    scale : scaling factor for the distance potential
    gamma : discount factor (default 0.99)
    """

    def __init__(
        self,
        goal: np.ndarray,
        obs_dim: int,
        goal_start: int,
        goal_dim: int,
        scale: float = 1.0,
        gamma: float = 0.99,
    ) -> None:
        self.goal = np.asarray(goal, dtype=np.float64)
        self.obs_dim = obs_dim
        self.goal_start = goal_start
        self.goal_dim = goal_dim
        self.scale = scale
        self.gamma = gamma

    def shape(
        self,
        rewards: np.ndarray,
        obs: np.ndarray,
        next_obs: np.ndarray,
        dones: np.ndarray,
    ) -> np.ndarray:
        """Compute goal-distance shaped rewards.

        Parameters
        ----------
        rewards : (N,) base rewards
        obs : (N, obs_dim) current observations
        next_obs : (N, obs_dim) next observations
        dones : (N,) episode termination flags

        Returns
        -------
        shaped_rewards : (N,) float64 array
        """
        obs_2d = np.asarray(obs, dtype=np.float64)
        next_obs_2d = np.asarray(next_obs, dtype=np.float64)
        rewards = np.ascontiguousarray(rewards, dtype=np.float64)
        dones = np.ascontiguousarray(dones, dtype=np.float64)

        # Rust expects flat (N * obs_dim,) arrays
        obs_flat = np.ascontiguousarray(obs_2d.ravel(), dtype=np.float64)
        next_obs_flat = np.ascontiguousarray(next_obs_2d.ravel(), dtype=np.float64)

        phi_current = _core.compute_goal_distance_potentials(
            obs_flat, self.goal, self.obs_dim, self.goal_start, self.goal_dim, self.scale,
        )
        phi_next = _core.compute_goal_distance_potentials(
            next_obs_flat, self.goal, self.obs_dim, self.goal_start, self.goal_dim, self.scale,
        )

        phi_current = np.ascontiguousarray(phi_current)
        phi_next = np.ascontiguousarray(phi_next)

        return _core.shape_rewards_pbrs(rewards, phi_current, phi_next, self.gamma, dones)
