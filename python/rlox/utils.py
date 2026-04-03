"""Shared utilities for rlox algorithms."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np


def detect_env_spaces(env_id: str) -> tuple[int, Any, bool]:
    """Detect obs_dim, action_space, and whether the env is discrete.

    Creates a temporary environment instance to inspect its observation and
    action spaces, then closes it immediately.

    Parameters
    ----------
    env_id : str
        Gymnasium environment ID.

    Returns
    -------
    tuple[int, Any, bool]
        ``(obs_dim, action_space, is_discrete)`` where *obs_dim* is the
        flattened observation dimension, *action_space* is the raw
        ``gymnasium.spaces.Space``, and *is_discrete* is ``True`` when
        the action space is ``Discrete``.
    """
    tmp = gym.make(env_id)
    obs_dim = int(np.prod(tmp.observation_space.shape))
    action_space = tmp.action_space
    is_discrete = isinstance(action_space, gym.spaces.Discrete)
    tmp.close()
    return obs_dim, action_space, is_discrete
