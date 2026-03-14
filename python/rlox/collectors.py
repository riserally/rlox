"""RolloutCollector: gather on-policy experience using Rust VecEnv + GAE.

This module provides the bridge between environment stepping (Rust) and
policy evaluation (PyTorch). It collects ``n_steps`` of experience from
``n_envs`` parallel environments and computes GAE advantages, returning
a flat :class:`~rlox.batch.RolloutBatch` ready for SGD.

.. note::
    Currently uses ``rlox.VecEnv`` which only supports CartPole. For other
    environments, use ``gymnasium.vector.SyncVectorEnv`` with
    ``rlox.compute_gae`` directly (see ``benchmarks/convergence/rlox_runner.py``
    for an example).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

import rlox
from rlox.batch import RolloutBatch


class RolloutCollector:
    """Collect on-policy rollout data from vectorized environments.

    Orchestrates the collect-then-compute pattern:
    1. Step ``n_envs`` environments for ``n_steps`` using ``rlox.VecEnv``
    2. Compute GAE advantages per environment using ``rlox.compute_gae``
    3. Flatten and return a :class:`RolloutBatch`

    Parameters
    ----------
    env_id : str
        Gymnasium environment ID (currently only CartPole-v1 is native).
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
        If True, divide rewards by running standard deviation.
    normalize_obs : bool
        If True, standardise observations using running statistics.

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
    ):
        self.env_id = env_id
        self.n_envs = n_envs
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.normalize_rewards = normalize_rewards
        self.normalize_obs = normalize_obs

        self.env = rlox.VecEnv(n=n_envs, seed=seed, env_id=env_id)
        self._obs: np.ndarray | None = None  # (n_envs, obs_dim)

        if normalize_rewards:
            self._reward_stats = rlox.RunningStats()
        if normalize_obs:
            self._obs_stats = rlox.RunningStats()

    def _maybe_normalize_obs(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        if not self.normalize_obs:
            return obs_tensor
        # Update stats per element
        flat = obs_tensor.detach().cpu().numpy().ravel().astype(np.float64)
        self._obs_stats.batch_update(flat)
        mean = self._obs_stats.mean()
        std = self._obs_stats.std()
        if std < 1e-8:
            std = 1.0
        return (obs_tensor - mean) / std

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
            obs_tensor = torch.as_tensor(self._obs, dtype=torch.float32, device=self.device)
            obs_input = self._maybe_normalize_obs(obs_tensor)

            actions, log_probs = policy.get_action_and_logprob(obs_input)
            values = policy.get_value(obs_input)

            # Step environment
            actions_np = actions.cpu().numpy().astype(np.uint32).tolist()
            step_result = self.env.step_all(actions_np)

            all_obs.append(obs_tensor)
            all_actions.append(actions)
            all_log_probs.append(log_probs)
            all_values.append(values)
            all_rewards.append(torch.as_tensor(
                step_result["rewards"].astype(np.float32), device=self.device
            ))

            terminated = step_result["terminated"].astype(bool)
            truncated = step_result["truncated"].astype(bool)
            dones = terminated | truncated
            all_dones.append(torch.as_tensor(dones.astype(np.float32), device=self.device))

            # Next observations
            next_obs = step_result["obs"].copy()

            # For value bootstrapping on terminal envs, use terminal_obs
            # (the actual last observation before auto-reset)
            self._obs = next_obs

        # Bootstrap value for GAE
        last_obs_tensor = torch.as_tensor(self._obs, dtype=torch.float32, device=self.device)
        last_obs_input = self._maybe_normalize_obs(last_obs_tensor)
        last_values = policy.get_value(last_obs_input)

        # Compute GAE per environment, then concatenate
        all_advantages = []
        all_returns = []
        for env_idx in range(self.n_envs):
            rewards_env = torch.stack([r[env_idx] for r in all_rewards])
            values_env = torch.stack([v[env_idx] for v in all_values])
            dones_env = torch.stack([d[env_idx] for d in all_dones])

            if self.normalize_rewards:
                r_np = rewards_env.cpu().numpy().astype(np.float64)
                self._reward_stats.batch_update(r_np)
                std = self._reward_stats.std()
                if std < 1e-8:
                    std = 1.0
                rewards_env = rewards_env / std

            adv, ret = rlox.compute_gae(
                rewards=rewards_env.cpu().numpy().astype(np.float64),
                values=values_env.cpu().numpy().astype(np.float64),
                dones=dones_env.cpu().numpy().astype(np.float64),
                last_value=float(last_values[env_idx].cpu()),
                gamma=self.gamma,
                lam=self.gae_lambda,
            )
            all_advantages.append(torch.as_tensor(adv, dtype=torch.float32, device=self.device))
            all_returns.append(torch.as_tensor(ret, dtype=torch.float32, device=self.device))

        # Stack: (n_steps, n_envs, ...) then flatten to (n_steps * n_envs, ...)
        obs_t = torch.stack(all_obs)           # (n_steps, n_envs, obs_dim)
        actions_t = torch.stack(all_actions)    # (n_steps, n_envs) or (n_steps, n_envs, act_dim)
        rewards_t = torch.stack(all_rewards)    # (n_steps, n_envs)
        dones_t = torch.stack(all_dones)        # (n_steps, n_envs)
        log_probs_t = torch.stack(all_log_probs)  # (n_steps, n_envs)
        values_t = torch.stack(all_values)      # (n_steps, n_envs)
        # advantages/returns: each is (n_steps,) per env -> stack to (n_envs, n_steps) then transpose
        advantages_t = torch.stack(all_advantages).T  # (n_steps, n_envs)
        returns_t = torch.stack(all_returns).T         # (n_steps, n_envs)

        # Flatten (n_steps, n_envs, ...) -> (n_steps * n_envs, ...)
        total = n_steps * self.n_envs
        return RolloutBatch(
            obs=obs_t.reshape(total, -1),
            actions=actions_t.reshape(total) if actions_t.dim() == 2 else actions_t.reshape(total, -1),
            rewards=rewards_t.reshape(total),
            dones=dones_t.reshape(total),
            log_probs=log_probs_t.reshape(total),
            values=values_t.reshape(total),
            advantages=advantages_t.reshape(total),
            returns=returns_t.reshape(total),
        )
