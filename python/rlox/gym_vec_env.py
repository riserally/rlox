"""GymVecEnv: gymnasium SyncVectorEnv with rlox-compatible interface.

Wraps ``gymnasium.vector.SyncVectorEnv`` to present the same step/reset
contract as ``rlox.VecEnv``, enabling the :class:`~rlox.collectors.RolloutCollector`
to drive arbitrary Gymnasium environments through the same code path.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium.vector.vector_env import AutoresetMode
import numpy as np


class GymVecEnv:
    """Vectorized environment wrapper matching the ``rlox.VecEnv`` interface.

    Uses ``AutoresetMode.SAME_STEP`` so that terminal observations are
    available in the info dict on the same step the episode ends (matching
    the behaviour of ``rlox.VecEnv``).

    Parameters
    ----------
    env_id : str
        Gymnasium environment ID (e.g. ``"Pendulum-v1"``).
    n_envs : int
        Number of parallel sub-environments.
    seed : int
        Base seed; each sub-env gets ``seed + i``.
    """

    def __init__(
        self,
        env_id: str,
        n_envs: int,
        seed: int = 0,
        record_episode_stats: bool = True,
    ) -> None:
        self._env = gym.vector.SyncVectorEnv(
            [self._make_env(env_id, seed + i, record_episode_stats) for i in range(n_envs)],
            autoreset_mode=AutoresetMode.SAME_STEP,
        )
        self._n_envs = n_envs
        self._record_episode_stats = record_episode_stats
        self._episode_rewards: list[float] = []
        self._episode_lengths: list[int] = []

    @staticmethod
    def _make_env(env_id: str, seed: int, record_stats: bool = True):  # noqa: ANN205
        def _thunk() -> gym.Env:
            env = gym.make(env_id)
            if record_stats:
                env = gym.wrappers.RecordEpisodeStatistics(env)
            env.reset(seed=seed)
            return env

        return _thunk

    @property
    def episode_rewards(self) -> list[float]:
        """Rewards of all completed episodes."""
        return self._episode_rewards

    @property
    def episode_lengths(self) -> list[int]:
        """Lengths of all completed episodes."""
        return self._episode_lengths

    def step_all(self, actions: np.ndarray | list[Any]) -> dict[str, Any]:
        """Step all sub-environments.

        Returns
        -------
        dict with keys ``obs``, ``rewards``, ``terminated``, ``truncated``,
        ``terminal_obs``.  Types match ``rlox.VecEnv``: rewards as float64,
        terminated/truncated as uint8.
        """
        if not isinstance(actions, np.ndarray):
            actions = np.asarray(actions)

        obs, rewards, terminated, truncated, infos = self._env.step(actions)

        # With SAME_STEP autoreset, gymnasium puts terminal observations
        # in infos["final_obs"] for envs that finished this step.
        terminal_obs: list[np.ndarray | None] = [None] * self._n_envs
        if "final_obs" in infos:
            dones = np.asarray(terminated, dtype=bool) | np.asarray(
                truncated, dtype=bool
            )
            mask = infos.get("_final_obs", dones)
            for i in range(self._n_envs):
                if mask[i] and infos["final_obs"][i] is not None:
                    terminal_obs[i] = np.asarray(
                        infos["final_obs"][i], dtype=np.float32
                    )

        # Extract episode statistics from RecordEpisodeStatistics wrapper
        if self._record_episode_stats and "episode" in infos:
            ep_info = infos["episode"]
            mask = infos.get("_episode", np.zeros(self._n_envs, dtype=bool))
            for i in range(self._n_envs):
                if mask[i]:
                    self._episode_rewards.append(float(ep_info["r"][i]))
                    self._episode_lengths.append(int(ep_info["l"][i]))

        return {
            "obs": np.asarray(obs, dtype=np.float32),
            "rewards": np.asarray(rewards, dtype=np.float64),
            "terminated": np.asarray(terminated, dtype=np.uint8),
            "truncated": np.asarray(truncated, dtype=np.uint8),
            "terminal_obs": terminal_obs,
        }

    def reset_all(self, seed: int | None = None) -> np.ndarray:
        """Reset all sub-environments.

        Returns
        -------
        np.ndarray of shape ``(n_envs, obs_dim)`` with dtype float32.
        """
        kwargs: dict[str, Any] = {}
        if seed is not None:
            kwargs["seed"] = seed
        obs, _info = self._env.reset(**kwargs)
        return np.asarray(obs, dtype=np.float32)

    def num_envs(self) -> int:
        """Return the number of parallel sub-environments."""
        return self._n_envs

    def close(self) -> None:
        """Close all sub-environments and release resources."""
        self._env.close()

    @property
    def action_space(self) -> gym.Space:
        """Return the single-env action space."""
        return self._env.single_action_space

    @property
    def observation_space(self) -> gym.Space:
        """Return the single-env observation space."""
        return self._env.single_observation_space
