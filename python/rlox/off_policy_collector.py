"""Off-policy data collection with vectorized environments.

Provides a reusable collector for SAC, TD3, DQN that supports:
- Multi-env parallel collection via GymVecEnv
- Batch buffer insertion via push_batch
- Pluggable exploration strategies
- Action scaling for continuous envs

Example
-------
>>> from rlox.off_policy_collector import OffPolicyCollector
>>> from rlox.exploration import GaussianNoise
>>>
>>> collector = OffPolicyCollector(
...     env_id="Pendulum-v1",
...     n_envs=4,
...     buffer=rlox.ReplayBuffer(100_000, 3, 1),
...     exploration=GaussianNoise(sigma=0.1),
... )
>>> obs = collector.reset()
>>> for step in range(1000):
...     obs = collector.collect_step(get_action_fn, step, total_steps=10000)
"""

from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable

import gymnasium as gym
import numpy as np

from rlox.gym_vec_env import GymVecEnv


@runtime_checkable
class CollectorProtocol(Protocol):
    """Protocol for data collectors — users can bring their own."""

    def reset(self) -> np.ndarray:
        """Reset environments and return initial observations."""
        ...

    def collect_step(
        self,
        get_action: Callable[[np.ndarray], np.ndarray],
        step: int,
        total_steps: int,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Collect one step from all environments.

        Parameters
        ----------
        get_action : callable
            Maps observations (n_envs, obs_dim) → actions (n_envs, act_dim).
        step : current training step
        total_steps : total training steps

        Returns
        -------
        obs : (n_envs, obs_dim) next observations
        rewards : (n_envs,) rewards
        ep_reward_mean : float mean episode reward (if any episodes finished)
        """
        ...

    @property
    def n_envs(self) -> int:
        """Number of parallel environments."""
        ...


class OffPolicyCollector:
    """Vectorized off-policy data collector.

    Collects transitions from multiple parallel environments and stores
    them in a replay buffer using batch insertion.

    Parameters
    ----------
    env_id : str
        Gymnasium environment ID.
    n_envs : int
        Number of parallel environments (default 1).
    buffer : ReplayBuffer or compatible
        Replay buffer to store transitions.
    exploration : ExplorationStrategy, optional
        Noise/exploration strategy applied to actions.
    act_high : float, optional
        Action scaling factor. If None, auto-detected from env.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        env_id: str,
        n_envs: int = 1,
        buffer: Any = None,
        exploration: Any = None,
        act_high: float | None = None,
        seed: int = 42,
    ):
        self.env_id = env_id
        self._n_envs = n_envs
        self.buffer = buffer
        self.exploration = exploration
        self.seed = seed

        # Create vectorized env
        if n_envs > 1:
            self.env = GymVecEnv(env_id, n_envs=n_envs)
        else:
            self.env = GymVecEnv(env_id, n_envs=1)

        # Detect spaces
        probe = gym.make(env_id)
        self.obs_dim = int(np.prod(probe.observation_space.shape))
        self.is_continuous = not isinstance(probe.action_space, gym.spaces.Discrete)
        if self.is_continuous:
            self.act_dim = int(np.prod(probe.action_space.shape))
            self.act_high = (
                act_high if act_high is not None else float(probe.action_space.high[0])
            )
        else:
            self.act_dim = 1
            self.act_high = 1.0
        probe.close()

        # Episode tracking
        self._ep_rewards = np.zeros(n_envs)
        self._completed_rewards: list[float] = []
        self._obs: np.ndarray | None = None

    @property
    def n_envs(self) -> int:
        return self._n_envs

    def reset(self) -> np.ndarray:
        """Reset all environments and return initial observations."""
        self._obs = self.env.reset_all(seed=self.seed)
        self._ep_rewards = np.zeros(self._n_envs)
        self._completed_rewards.clear()
        if self.exploration is not None:
            self.exploration.reset()
        return self._obs

    def collect_step(
        self,
        get_action: Callable[[np.ndarray], np.ndarray],
        step: int,
        total_steps: int,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Collect one step from all environments.

        Calls get_action, applies exploration, steps envs, stores in buffer.

        Returns
        -------
        next_obs, rewards, mean_episode_reward
        """
        if self._obs is None:
            self._obs = self.reset()

        obs = self._obs

        # Get actions from policy
        actions = get_action(obs)

        # Apply exploration
        if self.exploration is not None:
            actions = self.exploration.select_action(actions, step, total_steps)

        # Clip actions for continuous envs
        if self.is_continuous:
            actions = np.clip(actions, -self.act_high, self.act_high)

        # Step environments
        if self._n_envs > 1:
            result = self.env.step_all(actions if not self.is_continuous else actions)
            next_obs = result["obs"]
            rewards = result["rewards"]
            terminated = result["terminated"].astype(bool)
            truncated = result["truncated"].astype(bool)
        else:
            result = self.env.step_all(actions)
            next_obs = result["obs"]
            rewards = result["rewards"]
            terminated = result["terminated"].astype(bool)
            truncated = result["truncated"].astype(bool)

        # Track episode rewards
        self._ep_rewards += rewards
        dones = terminated | truncated
        for i in range(self._n_envs):
            if dones[i]:
                self._completed_rewards.append(float(self._ep_rewards[i]))
                self._ep_rewards[i] = 0.0

        # Store in buffer
        if self.buffer is not None:
            if self._n_envs > 1 and hasattr(self.buffer, "push_batch"):
                obs_flat = obs.reshape(-1).astype(np.float32)
                next_obs_flat = next_obs.reshape(-1).astype(np.float32)
                actions_flat = actions.reshape(-1).astype(np.float32)
                rewards_f32 = rewards.astype(np.float32)
                term_u8 = terminated.astype(np.uint8)
                trunc_u8 = truncated.astype(np.uint8)
                self.buffer.push_batch(
                    obs_flat,
                    next_obs_flat,
                    actions_flat,
                    rewards_f32,
                    term_u8,
                    trunc_u8,
                )
            else:
                for i in range(self._n_envs):
                    self.buffer.push(
                        obs[i].astype(np.float32),
                        actions[i].astype(np.float32)
                        if actions.ndim > 1
                        else np.array([actions[i]], dtype=np.float32),
                        float(rewards[i]),
                        bool(terminated[i]),
                        bool(truncated[i]),
                        next_obs[i].astype(np.float32),
                    )

        self._obs = next_obs

        mean_ep_reward = (
            float(np.mean(self._completed_rewards[-10:]))
            if self._completed_rewards
            else 0.0
        )

        return next_obs, rewards, mean_ep_reward

    def collect_n_steps(
        self,
        get_action: Callable[[np.ndarray], np.ndarray],
        n_steps: int,
        start_step: int,
        total_steps: int,
    ) -> float:
        """Collect n_steps from all environments.

        Returns mean episode reward over the collection period.
        """
        for i in range(n_steps):
            self.collect_step(get_action, start_step + i, total_steps)
        return (
            float(np.mean(self._completed_rewards[-10:]))
            if self._completed_rewards
            else 0.0
        )
