"""Exploration strategies for off-policy algorithms.

Provides pluggable noise and exploration mechanisms that satisfy
the :class:`ExplorationStrategy` protocol.

Example
-------
>>> from rlox.exploration import GaussianNoise, OUNoise, EpsilonGreedy
>>> noise = GaussianNoise(sigma=0.1)
>>> noisy_action = noise.select_action(action, step=0, total_steps=100000)
"""

from __future__ import annotations

import numpy as np


class GaussianNoise:
    """Additive Gaussian noise (used by TD3).

    Parameters
    ----------
    sigma : float
        Standard deviation of the noise.
    clip : float or None
        If set, clip noise to [-clip, clip].
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self, sigma: float = 0.1, clip: float | None = None, seed: int | None = None
    ):
        self.sigma = sigma
        self.clip = clip
        self._rng = np.random.default_rng(seed)

    def select_action(
        self, action: np.ndarray, step: int, total_steps: int
    ) -> np.ndarray:
        noise = self._rng.normal(0, self.sigma, size=action.shape)
        if self.clip is not None:
            noise = np.clip(noise, -self.clip, self.clip)
        return action + noise

    def reset(self) -> None:
        pass


class EpsilonGreedy:
    """Epsilon-greedy exploration (used by DQN).

    Linearly decays epsilon from ``eps_start`` to ``eps_end`` over
    ``decay_fraction`` of total training steps.

    Parameters
    ----------
    n_actions : int
        Number of discrete actions.
    eps_start : float
        Initial epsilon (default 1.0).
    eps_end : float
        Final epsilon (default 0.05).
    decay_fraction : float
        Fraction of total steps over which to decay (default 0.1).
    seed : int or None
        Random seed.
    """

    def __init__(
        self,
        n_actions: int,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        decay_fraction: float = 0.1,
        seed: int | None = None,
    ):
        self.n_actions = n_actions
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.decay_fraction = decay_fraction
        self._rng = np.random.default_rng(seed)

    def _get_epsilon(self, step: int, total_steps: int) -> float:
        frac = min(1.0, step / max(1, int(total_steps * self.decay_fraction)))
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def select_action(
        self, action: np.ndarray, step: int, total_steps: int
    ) -> np.ndarray:
        eps = self._get_epsilon(step, total_steps)
        if self._rng.random() < eps:
            return np.array([self._rng.integers(0, self.n_actions)])
        return action

    def reset(self) -> None:
        pass


class OUNoise:
    """Ornstein-Uhlenbeck noise for temporally correlated exploration.

    Produces smooth, mean-reverting noise that is useful for physical
    control tasks where abrupt action changes are undesirable.

    Parameters
    ----------
    action_dim : int
        Dimensionality of the action space.
    mu : float
        Mean of the noise process (default 0.0).
    theta : float
        Rate of mean reversion (default 0.15).
    sigma : float
        Volatility (default 0.2).
    seed : int or None
        Random seed.
    """

    def __init__(
        self,
        action_dim: int,
        mu: float = 0.0,
        theta: float = 0.15,
        sigma: float = 0.2,
        seed: int | None = None,
    ):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self._rng = np.random.default_rng(seed)
        self._state = np.zeros(action_dim)

    def select_action(
        self, action: np.ndarray, step: int, total_steps: int
    ) -> np.ndarray:
        dx = self.theta * (
            self.mu - self._state
        ) + self.sigma * self._rng.standard_normal(self.action_dim)
        self._state = self._state + dx
        return action + self._state

    def reset(self) -> None:
        self._state = np.zeros(self.action_dim)
