"""Visual RL environment wrappers.

Provides frame stacking, image preprocessing, and convenience wrappers
for Atari and DMControl pixel-based environments.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np


class FrameStack:
    """Stack N consecutive frames for temporal information.

    Wraps a vectorized environment that produces image observations of
    shape ``(B, C, H, W)`` and concatenates the last ``n_frames``
    observations along the channel dimension, yielding
    ``(B, n_frames * C, H, W)``.

    Parameters
    ----------
    env : object
        Inner vectorized environment with ``step_all`` / ``reset_all``.
    n_frames : int
        Number of consecutive frames to stack (default 4).
    """

    def __init__(self, env: Any, n_frames: int = 4) -> None:
        self.env = env
        self.n_frames = n_frames
        self._frames: deque[np.ndarray] = deque(maxlen=n_frames)

    def step_all(self, actions: np.ndarray | list[Any]) -> dict[str, Any]:
        """Step the inner env and return stacked observations."""
        result = self.env.step_all(actions)
        self._frames.append(result["obs"])
        result["obs"] = np.concatenate(list(self._frames), axis=1)
        return result

    def reset_all(self, **kwargs: Any) -> np.ndarray:
        """Reset the inner env and fill all frame slots with the initial obs."""
        obs = self.env.reset_all(**kwargs)
        self._frames.clear()
        for _ in range(self.n_frames):
            self._frames.append(obs)
        return np.concatenate(list(self._frames), axis=1)

    def num_envs(self) -> int:
        """Return the number of parallel sub-environments."""
        return self.env.num_envs()

    def close(self) -> None:
        """Close the inner environment."""
        if hasattr(self.env, "close"):
            self.env.close()


class ImagePreprocess:
    """Resize, grayscale, and normalize pixel observations.

    Expects observations of shape ``(B, C, H, W)`` in float32.

    Parameters
    ----------
    env : object
        Inner vectorized environment.
    size : tuple[int, int]
        Target spatial resolution ``(H, W)`` (default ``(84, 84)``).
    grayscale : bool
        Convert RGB to grayscale (default ``True``).
    normalize : bool
        Scale pixel values to ``[0, 1]`` by dividing by 255 (default ``True``).
    """

    def __init__(
        self,
        env: Any,
        size: tuple[int, int] = (84, 84),
        grayscale: bool = True,
        normalize: bool = True,
    ) -> None:
        self.env = env
        self.size = size
        self.grayscale = grayscale
        self.normalize = normalize

    def _process(self, obs: np.ndarray) -> np.ndarray:
        """Apply preprocessing to a batch of images (B, C, H, W)."""
        # Grayscale: average over channel dim
        if self.grayscale and obs.shape[1] == 3:
            # ITU-R BT.601 luma weights
            weights = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)
            obs = np.einsum("bchw,c->bhw", obs, weights)[:, np.newaxis, :, :]

        # Resize if spatial dims don't match target
        _, c, h, w = obs.shape
        target_h, target_w = self.size
        if h != target_h or w != target_w:
            obs = self._resize(obs, target_h, target_w)

        # Normalize
        if self.normalize:
            obs = obs / 255.0

        return obs

    @staticmethod
    def _resize(obs: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        """Bilinear resize using numpy (no torch/PIL dependency)."""
        b, c, h, w = obs.shape
        # Row indices
        row_ratio = h / target_h
        col_ratio = w / target_w

        row_idx = np.arange(target_h, dtype=np.float32) * row_ratio
        col_idx = np.arange(target_w, dtype=np.float32) * col_ratio

        row_floor = np.clip(np.floor(row_idx).astype(int), 0, h - 1)
        col_floor = np.clip(np.floor(col_idx).astype(int), 0, w - 1)
        row_ceil = np.clip(row_floor + 1, 0, h - 1)
        col_ceil = np.clip(col_floor + 1, 0, w - 1)

        row_frac = row_idx - np.floor(row_idx)
        col_frac = col_idx - np.floor(col_idx)

        # Bilinear interpolation
        # obs[:, :, row_floor, col_floor] etc.
        top_left = obs[:, :, row_floor[:, None], col_floor[None, :]]
        top_right = obs[:, :, row_floor[:, None], col_ceil[None, :]]
        bottom_left = obs[:, :, row_ceil[:, None], col_floor[None, :]]
        bottom_right = obs[:, :, row_ceil[:, None], col_ceil[None, :]]

        rf = row_frac[:, None]
        cf = col_frac[None, :]

        result = (
            top_left * (1 - rf) * (1 - cf)
            + top_right * (1 - rf) * cf
            + bottom_left * rf * (1 - cf)
            + bottom_right * rf * cf
        )
        return result.astype(np.float32)

    def step_all(self, actions: np.ndarray | list[Any]) -> dict[str, Any]:
        """Step the inner env and preprocess the observation."""
        result = self.env.step_all(actions)
        result["obs"] = self._process(result["obs"])
        return result

    def reset_all(self, **kwargs: Any) -> np.ndarray:
        """Reset the inner env and preprocess the observation."""
        obs = self.env.reset_all(**kwargs)
        return self._process(obs)

    def num_envs(self) -> int:
        """Return the number of parallel sub-environments."""
        return self.env.num_envs()

    def close(self) -> None:
        """Close the inner environment."""
        if hasattr(self.env, "close"):
            self.env.close()


class AtariWrapper:
    """Standard Atari preprocessing pipeline.

    Combines NoopReset, MaxAndSkip, EpisodicLife, FireReset,
    ImagePreprocess, and FrameStack into a single convenience wrapper.

    Requires ``ale-py`` and ``gymnasium[atari]``.

    Parameters
    ----------
    env_id : str
        Atari environment ID (e.g. ``"PongNoFrameskip-v4"``).
    n_envs : int
        Number of parallel sub-environments (default 8).
    frame_stack : int
        Number of frames to stack (default 4).
    """

    def __init__(
        self,
        env_id: str,
        n_envs: int = 8,
        frame_stack: int = 4,
    ) -> None:
        import gymnasium as gym
        from gymnasium.wrappers import (
            AtariPreprocessing,
            FrameStack as GymFrameStack,
        )

        def _make_env(idx: int):
            def _thunk() -> gym.Env:
                env = gym.make(env_id)
                env = AtariPreprocessing(env, frame_skip=4, grayscale_obs=True)
                env = GymFrameStack(env, frame_stack)
                return env
            return _thunk

        self._env = gym.vector.SyncVectorEnv(
            [_make_env(i) for i in range(n_envs)],
        )
        self._n_envs = n_envs

    def step_all(self, actions: np.ndarray | list[Any]) -> dict[str, Any]:
        """Step all sub-environments."""
        if not isinstance(actions, np.ndarray):
            actions = np.asarray(actions)
        obs, rewards, terminated, truncated, infos = self._env.step(actions)
        return {
            "obs": np.asarray(obs, dtype=np.float32),
            "rewards": np.asarray(rewards, dtype=np.float64),
            "terminated": np.asarray(terminated, dtype=np.uint8),
            "truncated": np.asarray(truncated, dtype=np.uint8),
            "terminal_obs": [None] * self._n_envs,
        }

    def reset_all(self, **kwargs: Any) -> np.ndarray:
        """Reset all sub-environments."""
        obs, _info = self._env.reset(**kwargs)
        return np.asarray(obs, dtype=np.float32)

    def num_envs(self) -> int:
        """Return the number of parallel sub-environments."""
        return self._n_envs


class DMControlWrapper:
    """Wrap dm_control environments with pixel observations.

    Requires ``dm_control``.

    Parameters
    ----------
    domain : str
        DMControl domain (e.g. ``"cartpole"``).
    task : str
        DMControl task (e.g. ``"swingup"``).
    n_envs : int
        Number of parallel instances (default 1).
    size : tuple[int, int]
        Render resolution (default ``(84, 84)``).
    frame_skip : int
        Number of physics steps per action (default 2).
    """

    def __init__(
        self,
        domain: str,
        task: str,
        n_envs: int = 1,
        size: tuple[int, int] = (84, 84),
        frame_skip: int = 2,
    ) -> None:
        from dm_control import suite

        self._domain = domain
        self._task = task
        self._n_envs = n_envs
        self._size = size
        self._frame_skip = frame_skip
        self._envs = [
            suite.load(domain, task) for _ in range(n_envs)
        ]

    def _render(self, env: Any) -> np.ndarray:
        """Render a single environment to pixels (C, H, W)."""
        pixels = env.physics.render(
            height=self._size[0], width=self._size[1], camera_id=0
        )
        # pixels is (H, W, 3) uint8 -> (3, H, W) float32
        return np.transpose(pixels, (2, 0, 1)).astype(np.float32)

    def step_all(self, actions: np.ndarray | list[Any]) -> dict[str, Any]:
        """Step all environments."""
        if not isinstance(actions, np.ndarray):
            actions = np.asarray(actions)

        obs_list = []
        rewards = np.zeros(self._n_envs, dtype=np.float64)
        terminated = np.zeros(self._n_envs, dtype=np.uint8)

        for i, env in enumerate(self._envs):
            reward = 0.0
            for _ in range(self._frame_skip):
                ts = env.step(actions[i] if actions.ndim > 1 else actions)
                reward += ts.reward or 0.0
                if ts.last():
                    break
            rewards[i] = reward
            terminated[i] = 1 if ts.last() else 0
            obs_list.append(self._render(env))

            # Auto-reset on termination
            if ts.last():
                env.reset()

        return {
            "obs": np.stack(obs_list).astype(np.float32),
            "rewards": rewards,
            "terminated": terminated,
            "truncated": np.zeros(self._n_envs, dtype=np.uint8),
            "terminal_obs": [None] * self._n_envs,
        }

    def reset_all(self, **kwargs: Any) -> np.ndarray:
        """Reset all environments and return pixel observations."""
        obs_list = []
        for env in self._envs:
            env.reset()
            obs_list.append(self._render(env))
        return np.stack(obs_list).astype(np.float32)

    def num_envs(self) -> int:
        """Return the number of parallel sub-environments."""
        return self._n_envs
