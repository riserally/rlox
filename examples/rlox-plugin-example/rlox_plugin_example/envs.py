"""Example custom environment plugin.

Wraps CartPole-v1 and adds observation noise to demonstrate
how to register a custom environment with rlox.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from rlox.plugins import register_env


@register_env("NoisyCartPole-v0")
class NoisyCartPole(gym.Env):
    """CartPole with Gaussian observation noise.

    Parameters
    ----------
    noise_scale : float
        Standard deviation of additive Gaussian noise (default 0.01).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, noise_scale: float = 0.01, **kwargs):
        super().__init__()
        self._inner = gym.make("CartPole-v1")
        self.noise_scale = noise_scale
        self.observation_space = self._inner.observation_space
        self.action_space = self._inner.action_space

    def step(self, action):
        obs, reward, terminated, truncated, info = self._inner.step(action)
        noisy_obs = obs + np.random.normal(0, self.noise_scale, size=obs.shape)
        return noisy_obs.astype(np.float32), reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self._inner.reset(**kwargs)
        noisy_obs = obs + np.random.normal(0, self.noise_scale, size=obs.shape)
        return noisy_obs.astype(np.float32), info

    def close(self):
        self._inner.close()
