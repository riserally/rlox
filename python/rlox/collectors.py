"""RolloutCollector: gather on-policy experience using Rust VecEnv + GAE.

This module provides the bridge between environment stepping (Rust) and
policy evaluation (PyTorch). It collects ``n_steps`` of experience from
``n_envs`` parallel environments and computes GAE advantages, returning
a flat :class:`~rlox.batch.RolloutBatch` ready for SGD.

For Rust-native environments (CartPole), the collector uses ``rlox.VecEnv``
for maximum throughput. For all other Gymnasium environments, it falls back
to :class:`~rlox.gym_vec_env.GymVecEnv`.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn

import rlox
from rlox.batch import RolloutBatch
from rlox.gym_vec_env import GymVecEnv

# Environments with native Rust implementations
_NATIVE_ENV_IDS = frozenset({"CartPole-v1", "CartPole"})


class _RunningMeanStd:
    """Welford's online mean/variance tracker (per-dimension).

    Pure-Python fallback used for reward return normalization.
    For observation normalization, prefer ``rlox.RunningStatsVec`` (Rust)
    when available.
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


class RolloutCollector:
    """Collect on-policy rollout data from vectorized environments.

    Orchestrates the collect-then-compute pattern:
    1. Step ``n_envs`` environments for ``n_steps`` using the appropriate backend
    2. Compute GAE advantages per environment using ``rlox.compute_gae``
    3. Flatten and return a :class:`RolloutBatch`

    Parameters
    ----------
    env_id : str
        Gymnasium environment ID. CartPole-v1 uses the native Rust backend;
        all others use ``GymVecEnv`` (gymnasium SyncVectorEnv).
    n_envs : int
        Number of parallel environments.
    seed : int
        RNG seed for environment initialisation.
    device : str
        PyTorch device for output tensors.
    gamma : float
        Discount factor for GAE.
    gae_lambda : float
        Lambda parameter for GAE bias-variance tradeoff.
    normalize_rewards : bool
        If True, divide rewards by running standard deviation of discounted
        returns (SB3-style return-based normalization).
    normalize_obs : bool
        If True, standardise observations using per-dimension running statistics.
    reward_fn : Callable or None
        Optional reward shaping function ``(obs, actions, rewards) -> rewards``.
        Applied to raw rewards before storing.

    Example
    -------
    >>> from rlox.policies import DiscretePolicy
    >>> collector = RolloutCollector("CartPole-v1", n_envs=8, seed=0)
    >>> policy = DiscretePolicy(obs_dim=4, n_actions=2)
    >>> batch = collector.collect(policy, n_steps=128)
    >>> batch.obs.shape  # torch.Size([1024, 4])
    """

    def __init__(
        self,
        env_id: str,
        n_envs: int = 1,
        seed: int = 0,
        device: str = "cpu",
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        normalize_rewards: bool = False,
        normalize_obs: bool = False,
        reward_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
        | None = None,
    ):
        self.env_id = env_id
        self.n_envs = n_envs
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.normalize_rewards = normalize_rewards
        self.normalize_obs = normalize_obs
        self.reward_fn = reward_fn

        if env_id in _NATIVE_ENV_IDS:
            self.env = rlox.VecEnv(n=n_envs, seed=seed, env_id=env_id)
            self._is_discrete = True
        else:
            self.env = GymVecEnv(env_id, n_envs=n_envs, seed=seed)
            import gymnasium as gym

            self._is_discrete = isinstance(self.env.action_space, gym.spaces.Discrete)

        self._obs: np.ndarray | None = None  # (n_envs, obs_dim)

        # Bug 2 fix: return-based reward normalization (SB3-style)
        if normalize_rewards:
            self._return_estimate = np.zeros(n_envs, dtype=np.float64)
            self._return_var_stats = _RunningMeanStd(shape=1)

        # Bug 1 fix: per-dimension observation normalization
        if normalize_obs:
            # Determine obs_dim from environment
            if env_id in _NATIVE_ENV_IDS:
                # CartPole obs_dim is always 4
                obs_dim = 4
            else:
                obs_dim = int(np.prod(self.env.observation_space.shape))
            try:
                self._obs_stats = rlox.RunningStatsVec(obs_dim)
                self._obs_stats_is_rust = True
            except AttributeError:
                # RunningStatsVec not yet available in Rust; use Python fallback
                self._obs_stats = _RunningMeanStd(shape=obs_dim)
                self._obs_stats_is_rust = False

    def _maybe_normalize_obs(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        if not self.normalize_obs:
            return obs_tensor
        # Update stats per dimension
        flat = obs_tensor.detach().cpu().numpy().astype(np.float64)
        if self._obs_stats_is_rust:
            batch_size = flat.shape[0] if flat.ndim > 1 else 1
            self._obs_stats.batch_update(flat.ravel(), batch_size)
            mean = self._obs_stats.mean()  # np.ndarray of shape (obs_dim,)
            std = self._obs_stats.std()  # np.ndarray of shape (obs_dim,)
        else:
            self._obs_stats.update(flat)
            mean = self._obs_stats.mean
            std = np.sqrt(self._obs_stats.var)
        # Replace near-zero std with 1.0 per dimension
        std = np.where(std < 1e-8, 1.0, std)
        mean_t = torch.as_tensor(mean, dtype=torch.float32, device=obs_tensor.device)
        std_t = torch.as_tensor(std, dtype=torch.float32, device=obs_tensor.device)
        return (obs_tensor - mean_t) / std_t

    @torch.no_grad()
    def collect(self, policy: nn.Module, n_steps: int) -> RolloutBatch:
        """Run the policy for *n_steps* in each env and return a flat batch."""
        if self._obs is None:
            self._obs = self.env.reset_all()

        all_obs = []
        all_actions = []
        all_rewards = []
        all_dones = []
        all_log_probs = []
        all_values = []

        for _ in range(n_steps):
            obs_tensor = torch.as_tensor(
                self._obs, dtype=torch.float32, device=self.device
            )
            obs_input = self._maybe_normalize_obs(obs_tensor)

            if hasattr(policy, "get_action_value"):
                actions, log_probs, values = policy.get_action_value(obs_input)
            else:
                actions, log_probs = policy.get_action_and_logprob(obs_input)
                values = policy.get_value(obs_input)

            # Step environment
            actions_np = actions.cpu().numpy()
            if self._is_discrete:
                step_result = self.env.step_all(actions_np.astype(np.uint32).tolist())
            else:
                step_result = self.env.step_all(actions_np)

            obs_np = self._obs  # pre-step observations for reward_fn
            raw_rewards = step_result["rewards"]
            if self.reward_fn is not None:
                raw_rewards = self.reward_fn(obs_np, actions_np, raw_rewards)

            # Bug 4 fix: store normalized obs (obs_input), not raw (obs_tensor)
            all_obs.append(obs_input)
            all_actions.append(actions)
            all_log_probs.append(log_probs)
            all_values.append(values)

            terminated = step_result["terminated"].astype(bool)
            truncated = step_result["truncated"].astype(bool)
            dones = terminated | truncated

            # Bug 2 fix: return-based reward normalization (per-step)
            # Must normalize BEFORE truncation bootstrap so bootstrap values
            # (already in normalized space) aren't double-divided by ret_std.
            if self.normalize_rewards:
                self._return_estimate = (
                    self._return_estimate * self.gamma + raw_rewards.astype(np.float64)
                )
                self._return_var_stats.update(
                    self._return_estimate.reshape(-1, 1)
                )
                self._return_estimate[dones] = 0.0
                ret_std = max(np.sqrt(self._return_var_stats.var[0] + 1e-8), 1e-8)
                raw_rewards = raw_rewards / ret_std

            # Bug 3 fix: truncation bootstrap
            # When truncated but not terminated, add gamma * V(terminal_obs) to reward.
            # V(terminal_obs) is in normalized-return space (critic trained on normalized
            # rewards), so this must happen AFTER reward normalization.
            terminal_obs_list = step_result.get("terminal_obs")
            for i in range(self.n_envs):
                if truncated[i] and not terminated[i] and terminal_obs_list is not None:
                    term_obs = terminal_obs_list[i]
                    if term_obs is not None:
                        term_obs_t = torch.as_tensor(
                            term_obs, dtype=torch.float32, device=self.device
                        ).unsqueeze(0)
                        term_obs_input = self._maybe_normalize_obs(term_obs_t)
                        term_val = policy.get_value(term_obs_input).item()
                        raw_rewards[i] += self.gamma * term_val

            all_rewards.append(
                torch.as_tensor(raw_rewards.astype(np.float32), device=self.device)
            )
            # Bug 3 fix: pass only terminated (not dones) to GAE
            all_dones.append(
                torch.as_tensor(terminated.astype(np.float32), device=self.device)
            )

            # Next observations
            next_obs = step_result["obs"].copy()
            self._obs = next_obs

        # Bootstrap value for GAE
        last_obs_tensor = torch.as_tensor(
            self._obs, dtype=torch.float32, device=self.device
        )
        last_obs_input = self._maybe_normalize_obs(last_obs_tensor)
        last_values = policy.get_value(last_obs_input)

        # Compute GAE in a single batched call (env-major flat layout)
        rewards_stacked = torch.stack(all_rewards)  # (n_steps, n_envs)
        values_stacked = torch.stack(all_values)  # (n_steps, n_envs)
        dones_stacked = torch.stack(all_dones)  # (n_steps, n_envs)

        # Transpose to (n_envs, n_steps) env-major, flatten contiguously
        rewards_flat = (
            rewards_stacked.T.contiguous().cpu().numpy().astype(np.float64).ravel()
        )
        values_flat = (
            values_stacked.T.contiguous().cpu().numpy().astype(np.float64).ravel()
        )
        dones_flat = (
            dones_stacked.T.contiguous().cpu().numpy().astype(np.float64).ravel()
        )
        last_vals = last_values.cpu().numpy().astype(np.float64)

        adv_flat, ret_flat = rlox.compute_gae_batched(
            rewards=rewards_flat,
            values=values_flat,
            dones=dones_flat,
            last_values=last_vals,
            n_steps=n_steps,
            gamma=self.gamma,
            lam=self.gae_lambda,
        )

        # Reshape from env-major (n_envs, n_steps) back to (n_steps, n_envs)
        advantages_t = (
            torch.as_tensor(
                adv_flat,
                dtype=torch.float32,
                device=self.device,
            )
            .reshape(self.n_envs, n_steps)
            .T
        )
        returns_t = (
            torch.as_tensor(
                ret_flat,
                dtype=torch.float32,
                device=self.device,
            )
            .reshape(self.n_envs, n_steps)
            .T
        )

        # Stack: (n_steps, n_envs, ...) then flatten to (n_steps * n_envs, ...)
        obs_t = torch.stack(all_obs)  # (n_steps, n_envs, obs_dim)
        actions_t = torch.stack(
            all_actions
        )  # (n_steps, n_envs) or (n_steps, n_envs, act_dim)
        rewards_t = rewards_stacked  # already (n_steps, n_envs)
        dones_t = dones_stacked  # already (n_steps, n_envs)
        log_probs_t = torch.stack(all_log_probs)  # (n_steps, n_envs)
        values_t = values_stacked  # already (n_steps, n_envs)

        # Flatten (n_steps, n_envs, ...) -> (n_steps * n_envs, ...)
        total = n_steps * self.n_envs
        return RolloutBatch(
            obs=obs_t.reshape(total, *obs_t.shape[2:]),
            actions=actions_t.reshape(total)
            if actions_t.dim() == 2
            else actions_t.reshape(total, -1),
            rewards=rewards_t.reshape(total),
            dones=dones_t.reshape(total),
            log_probs=log_probs_t.reshape(total),
            values=values_t.reshape(total),
            advantages=advantages_t.reshape(total),
            returns=returns_t.reshape(total),
        )
