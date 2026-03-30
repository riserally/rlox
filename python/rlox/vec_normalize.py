"""VecNormalize: environment wrapper for observation and reward normalization.

Wraps a GymVecEnv (or any object with step_all/reset_all interface),
applying running mean/std normalization at the environment boundary.
This matches SB3's VecNormalize architecture.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import rlox as _rlox

    _RunningStatsVec = _rlox.RunningStatsVec
    _HAS_RUST_STATS = True
except (ImportError, AttributeError):
    _HAS_RUST_STATS = False


class _RunningMeanStd:
    """Welford's online mean/variance tracker (per-dimension).

    Pure-Python fallback when Rust ``RunningStatsVec`` is unavailable.
    """

    def __init__(self, shape: int) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count: float = 1e-4

    def update(self, batch: np.ndarray) -> None:
        batch = np.asarray(batch, dtype=np.float64)
        if batch.ndim == 1:
            batch = batch[np.newaxis]
        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)
        batch_count = batch.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(
        self,
        batch_mean: np.ndarray,
        batch_var: np.ndarray,
        batch_count: int,
    ) -> None:
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total
        self.mean = new_mean
        self.var = m2 / total
        self.count = total


class VecNormalize:
    """Environment wrapper that normalizes observations and rewards.

    Wraps a GymVecEnv (or any object with step_all/reset_all interface),
    applying running mean/std normalization at the environment boundary.
    This matches SB3's VecNormalize architecture.

    Parameters
    ----------
    env : object
        Inner vectorized environment with ``step_all`` / ``reset_all``.
    norm_obs : bool
        Whether to normalize observations (default True).
    norm_reward : bool
        Whether to normalize rewards (default True).
    gamma : float
        Discount factor for return-based reward normalization (default 0.99).
    clip_obs : float
        Clipping range for normalized observations (default 10.0).
    clip_reward : float
        Clipping range for normalized rewards (default 10.0).
    epsilon : float
        Small constant to avoid division by zero (default 1e-8).
    """

    def __init__(
        self,
        env: Any,
        norm_obs: bool = True,
        norm_reward: bool = True,
        gamma: float = 0.99,
        clip_obs: float = 10.0,
        clip_reward: float = 10.0,
        epsilon: float = 1e-8,
        obs_dim: int | None = None,
    ) -> None:
        self.env = env
        self.norm_obs = norm_obs
        self.norm_reward = norm_reward
        self.gamma = gamma
        self.clip_obs = clip_obs
        self.clip_reward = clip_reward
        self.epsilon = epsilon
        self._training = True

        # Determine obs_dim
        n_envs = self.num_envs()
        if obs_dim is not None:
            pass  # use provided obs_dim
        elif hasattr(env, "observation_space"):
            obs_dim = int(np.prod(env.observation_space.shape))
        else:
            # Fallback: reset and infer from shape
            probe_obs = env.reset_all()
            obs_dim = probe_obs.shape[-1] if probe_obs.ndim > 1 else probe_obs.shape[0]

        # Obs stats tracker
        if norm_obs:
            self._obs_rms: _RunningMeanStd | Any
            self._obs_rms_is_rust: bool
            if _HAS_RUST_STATS:
                try:
                    self._obs_rms = _RunningStatsVec(obs_dim)
                    self._obs_rms_is_rust = True
                except Exception:
                    self._obs_rms = _RunningMeanStd(obs_dim)
                    self._obs_rms_is_rust = False
            else:
                self._obs_rms = _RunningMeanStd(obs_dim)
                self._obs_rms_is_rust = False

        # Reward stats tracker (return-based, SB3-style)
        if norm_reward:
            self._return_estimate = np.zeros(n_envs, dtype=np.float64)
            self._ret_rms = _RunningMeanStd(shape=1)

    @property
    def training(self) -> bool:
        """Whether running statistics are updated on each step/reset."""
        return self._training

    @training.setter
    def training(self, value: bool) -> None:
        self._training = value

    # -------------------------------------------------------------------
    # Core interface
    # -------------------------------------------------------------------

    def step_all(self, actions: np.ndarray | list[Any]) -> dict[str, Any]:
        """Step all sub-environments, normalizing obs and rewards."""
        result = self.env.step_all(actions)

        # Reward normalization (before obs normalization -- order doesn't matter
        # but we do it first to keep return estimates in sync with dones)
        if self.norm_reward:
            rewards = result["rewards"].astype(np.float64)
            terminated = result["terminated"].astype(bool)
            truncated = result["truncated"].astype(bool)
            dones = terminated | truncated

            if self._training:
                self._return_estimate = self._return_estimate * self.gamma + rewards
                self._ret_rms.update(self._return_estimate.reshape(-1, 1))
                self._return_estimate[dones] = 0.0

            ret_std = max(np.sqrt(self._ret_rms.var[0] + self.epsilon), self.epsilon)
            norm_rewards = rewards / ret_std
            norm_rewards = np.clip(norm_rewards, -self.clip_reward, self.clip_reward)
            result["rewards"] = norm_rewards

        # Obs normalization
        if self.norm_obs:
            result["obs"] = self._normalize_obs_internal(
                result["obs"], update_stats=self._training
            )

            # Normalize terminal_obs (never update stats for these)
            terminal_obs = result.get("terminal_obs")
            if terminal_obs is not None:
                new_terminal: list[np.ndarray | None] = []
                for t_obs in terminal_obs:
                    if t_obs is not None:
                        new_terminal.append(
                            self._normalize_obs_internal(
                                t_obs[np.newaxis], update_stats=False
                            )[0]
                        )
                    else:
                        new_terminal.append(None)
                result["terminal_obs"] = new_terminal

        return result

    def reset_all(self, **kwargs: Any) -> np.ndarray:
        """Reset all sub-environments, normalizing the returned observations."""
        obs = self.env.reset_all(**kwargs)
        if self.norm_obs:
            obs = self._normalize_obs_internal(obs, update_stats=self._training)
        return obs

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observations without updating running statistics.

        Use this for evaluation with frozen stats.
        """
        if not self.norm_obs:
            return obs
        return self._normalize_obs_internal(obs, update_stats=False)

    # -------------------------------------------------------------------
    # Internal normalization helpers
    # -------------------------------------------------------------------

    def _normalize_obs_internal(
        self, obs: np.ndarray, *, update_stats: bool
    ) -> np.ndarray:
        """Normalize observations, optionally updating running stats."""
        obs = np.asarray(obs, dtype=np.float32)
        flat = obs.astype(np.float64)

        if update_stats:
            if self._obs_rms_is_rust:
                batch_size = flat.shape[0] if flat.ndim > 1 else 1
                self._obs_rms.batch_update(flat.ravel(), batch_size)
            else:
                if flat.ndim == 1:
                    self._obs_rms.update(flat[np.newaxis])
                else:
                    self._obs_rms.update(flat)

        # Get mean / std
        if self._obs_rms_is_rust:
            mean = self._obs_rms.mean()
            std = self._obs_rms.std()
        else:
            mean = self._obs_rms.mean
            std = np.sqrt(self._obs_rms.var)

        std = np.where(std < self.epsilon, 1.0, std)
        normalized = (obs.astype(np.float64) - mean) / std
        normalized = np.clip(normalized, -self.clip_obs, self.clip_obs)
        return normalized.astype(np.float32)

    # -------------------------------------------------------------------
    # Stats access
    # -------------------------------------------------------------------

    def get_obs_rms(self) -> dict[str, np.ndarray]:
        """Return current observation running statistics (mean, var)."""
        if not self.norm_obs:
            raise RuntimeError("Obs normalization is disabled (norm_obs=False)")
        if self._obs_rms_is_rust:
            return {
                "mean": self._obs_rms.mean(),
                "var": self._obs_rms.var(),
            }
        return {
            "mean": self._obs_rms.mean.copy(),
            "var": self._obs_rms.var.copy(),
        }

    # -------------------------------------------------------------------
    # Passthrough
    # -------------------------------------------------------------------

    def num_envs(self) -> int:
        """Return the number of parallel sub-environments."""
        return self.env.num_envs()

    @property
    def action_space(self):  # noqa: ANN201
        """Return the single-env action space."""
        return getattr(self.env, "action_space", None)

    @property
    def observation_space(self):  # noqa: ANN201
        """Return the single-env observation space."""
        return getattr(self.env, "observation_space", None)

    def close(self) -> None:
        """Close the inner environment."""
        if hasattr(self.env, "close"):
            self.env.close()

    @property
    def env_id(self) -> str | None:
        """Return the environment ID, if available on the inner env."""
        return getattr(self.env, "env_id", None)
