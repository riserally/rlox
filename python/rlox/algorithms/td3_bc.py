"""TD3+BC: A Minimalist Approach to Offline Reinforcement Learning.

TD3 with behavioral cloning regularization on the actor loss:
    actor_loss = -λ * Q(s, π(s)) + (π(s) - a_data)²
    where λ = α / mean(|Q(s, a_data)|)

Reference:
    S. Fujimoto and S. S. Gu,
    "A Minimalist Approach to Offline Reinforcement Learning,"
    NeurIPS, 2021.
    https://arxiv.org/abs/2106.06860
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

from rlox.callbacks import Callback
from rlox.logging import LoggerCallback
from rlox.networks import QNetwork, DeterministicPolicy, polyak_update
from rlox.offline.base import OfflineAlgorithm, OfflineDataset


class TD3BC(OfflineAlgorithm):
    """TD3+BC offline RL algorithm.

    Parameters
    ----------
    dataset : OfflineDataset
        Offline dataset (e.g., ``rlox.OfflineDatasetBuffer``).
    obs_dim : int
        Observation dimension.
    act_dim : int
        Action dimension.
    alpha : float
        BC regularization weight (default 2.5).
    hidden : int
        Hidden layer width (default 256).
    learning_rate : float
        Learning rate (default 3e-4).
    tau : float
        Soft target update rate (default 0.005).
    gamma : float
        Discount factor (default 0.99).
    policy_delay : int
        Actor update frequency (default 2).
    target_noise : float
        Target policy smoothing noise (default 0.2).
    noise_clip : float
        Target noise clipping (default 0.5).
    act_high : float
        Action space upper bound for clipping (default 1.0).
    batch_size : int
        Minibatch size (default 256).
    callbacks : list[Callback], optional
    logger : LoggerCallback, optional
    """

    def __init__(
        self,
        dataset: OfflineDataset,
        obs_dim: int,
        act_dim: int,
        alpha: float = 2.5,
        hidden: int = 256,
        learning_rate: float = 3e-4,
        tau: float = 0.005,
        gamma: float = 0.99,
        policy_delay: int = 2,
        target_noise: float = 0.2,
        noise_clip: float = 0.5,
        act_high: float = 1.0,
        batch_size: int = 256,
        callbacks: list[Callback] | None = None,
        logger: LoggerCallback | None = None,
    ):
        super().__init__(dataset, batch_size, callbacks, logger)
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.policy_delay = policy_delay
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self.act_high = act_high

        # Networks
        self.actor = DeterministicPolicy(obs_dim, act_dim, hidden, act_high)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic1 = QNetwork(obs_dim, act_dim, hidden)
        self.critic2 = QNetwork(obs_dim, act_dim, hidden)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=learning_rate
        )
        self.critic1_optimizer = torch.optim.Adam(
            self.critic1.parameters(), lr=learning_rate
        )
        self.critic2_optimizer = torch.optim.Adam(
            self.critic2.parameters(), lr=learning_rate
        )

        self._update_count = 0

    def _update(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32)
        actions = torch.as_tensor(batch["actions"], dtype=torch.float32)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32)
        terminated = torch.as_tensor(batch["terminated"], dtype=torch.float32)

        if actions.dim() == 1:
            actions = actions.unsqueeze(-1)

        # --- Critic update (same as TD3) ---
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.target_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_actions = (self.actor_target(next_obs) + noise).clamp(
                -self.act_high, self.act_high
            )
            q1_next = self.critic1_target(next_obs, next_actions).squeeze(-1)
            q2_next = self.critic2_target(next_obs, next_actions).squeeze(-1)
            target_q = rewards + self.gamma * (1.0 - terminated) * torch.min(
                q1_next, q2_next
            )

        q1 = self.critic1(obs, actions).squeeze(-1)
        q2 = self.critic2(obs, actions).squeeze(-1)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic1_optimizer.zero_grad(set_to_none=True)
        self.critic2_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        # Soft target update for critics
        polyak_update(self.critic1, self.critic1_target, self.tau)
        polyak_update(self.critic2, self.critic2_target, self.tau)

        actor_loss_val = 0.0
        self._update_count += 1

        # --- Actor update with BC regularization ---
        if self._update_count % self.policy_delay == 0:
            pi = self.actor(obs)
            q_val = self.critic1(obs, pi)

            # λ = α / mean(|Q(s, a_data)|)
            lmbda = self.alpha / q_val.abs().mean().detach()

            actor_loss = -lmbda * q_val.mean() + F.mse_loss(pi, actions)

            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()
            actor_loss_val = actor_loss.item()

            polyak_update(self.actor, self.actor_target, self.tau)

        return {
            "critic_loss": critic_loss.item() / 2,
            "actor_loss": actor_loss_val,
        }

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Get deterministic action."""
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            return self.actor(obs_t).squeeze(0).numpy()
