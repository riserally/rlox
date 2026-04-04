"""Conservative Q-Learning (CQL).

Adds a conservative penalty to Q-values for out-of-distribution actions:
    L_CQL = α * (E_π[Q(s,a)] - E_D[Q(s,a)]) + standard_bellman_loss

This pushes Q-values down for unseen actions and up for dataset actions,
preventing overestimation on out-of-distribution state-action pairs.

Reference:
    A. Kumar, A. Zhou, G. Tucker, S. Levine,
    "Conservative Q-Learning for Offline Reinforcement Learning,"
    NeurIPS, 2020.
    https://arxiv.org/abs/2006.04779
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

from rlox.callbacks import Callback
from rlox.logging import LoggerCallback
from rlox.networks import QNetwork, SquashedGaussianPolicy, polyak_update
from rlox.offline.base import OfflineAlgorithm, OfflineDataset


class CQL(OfflineAlgorithm):
    """Conservative Q-Learning.

    Parameters
    ----------
    dataset : OfflineDataset
        Offline dataset.
    obs_dim : int
        Observation dimension.
    act_dim : int
        Action dimension.
    cql_alpha : float
        CQL penalty weight (default 5.0).
    n_random_actions : int
        Number of random actions for CQL penalty (default 10).
    auto_alpha : bool
        Whether to auto-tune cql_alpha via Lagrangian (default False).
    cql_target_value : float
        Target value for Lagrangian α tuning (default -1.0).
    hidden : int
        Hidden layer width (default 256).
    learning_rate : float
        Learning rate (default 3e-4).
    tau : float
        Soft target update rate (default 0.005).
    gamma : float
        Discount factor (default 0.99).
    batch_size : int
        Minibatch size (default 256).
    auto_entropy : bool
        Whether to auto-tune SAC entropy α (default True).
    target_entropy : float, optional
        Target entropy for SAC (default -act_dim).
    callbacks : list[Callback], optional
    logger : LoggerCallback, optional
    """

    def __init__(
        self,
        dataset: OfflineDataset,
        obs_dim: int,
        act_dim: int,
        cql_alpha: float = 5.0,
        n_random_actions: int = 10,
        auto_alpha: bool = False,
        cql_target_value: float = -1.0,
        hidden: int = 256,
        learning_rate: float = 3e-4,
        tau: float = 0.005,
        gamma: float = 0.99,
        batch_size: int = 256,
        auto_entropy: bool = True,
        target_entropy: float | None = None,
        callbacks: list[Callback] | None = None,
        logger: LoggerCallback | None = None,
    ):
        super().__init__(dataset, batch_size, callbacks, logger)
        self.cql_alpha = cql_alpha
        self.n_random_actions = n_random_actions
        self.auto_cql_alpha = auto_alpha
        self.cql_target_value = cql_target_value
        self.gamma = gamma
        self.tau = tau
        self.act_dim = act_dim

        # SAC-style networks
        self.actor = SquashedGaussianPolicy(obs_dim, act_dim, hidden)
        self.critic1 = QNetwork(obs_dim, act_dim, hidden)
        self.critic2 = QNetwork(obs_dim, act_dim, hidden)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=learning_rate,
        )

        # SAC entropy tuning
        self.auto_entropy = auto_entropy
        self.target_entropy = (
            target_entropy if target_entropy is not None else -float(act_dim)
        )
        if auto_entropy:
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=learning_rate)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = 0.2

        # CQL alpha tuning (Lagrangian)
        if auto_alpha:
            self.log_cql_alpha = torch.zeros(1, requires_grad=True)
            self.cql_alpha_optimizer = torch.optim.Adam(
                [self.log_cql_alpha], lr=learning_rate
            )

    def _update(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32)
        actions = torch.as_tensor(batch["actions"], dtype=torch.float32)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32)
        terminated = torch.as_tensor(batch["terminated"], dtype=torch.float32)

        if actions.dim() == 1:
            actions = actions.unsqueeze(-1)

        batch_size = obs.shape[0]

        # --- Critic update ---
        with torch.no_grad():
            next_actions, next_log_prob = self.actor.sample(next_obs)
            q1_next = self.critic1_target(next_obs, next_actions).squeeze(-1)
            q2_next = self.critic2_target(next_obs, next_actions).squeeze(-1)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_prob
            target_q = rewards + self.gamma * (1.0 - terminated) * q_next

        # Standard Bellman loss
        q1_data = self.critic1(obs, actions).squeeze(-1)
        q2_data = self.critic2(obs, actions).squeeze(-1)
        bellman_loss = F.mse_loss(q1_data, target_q) + F.mse_loss(q2_data, target_q)

        # CQL penalty: E_π[Q(s,a)] - E_D[Q(s,a)]
        # Sample random actions
        random_actions = torch.FloatTensor(
            batch_size * self.n_random_actions, self.act_dim
        ).uniform_(-1, 1)
        obs_rep = (
            obs.unsqueeze(1)
            .expand(-1, self.n_random_actions, -1)
            .reshape(-1, obs.shape[-1])
        )

        q1_rand = self.critic1(obs_rep, random_actions).reshape(
            batch_size, self.n_random_actions
        )
        q2_rand = self.critic2(obs_rep, random_actions).reshape(
            batch_size, self.n_random_actions
        )

        # Sample policy actions
        with torch.no_grad():
            policy_actions, policy_log_probs = self.actor.sample(obs_rep)
        q1_policy = self.critic1(obs_rep, policy_actions).reshape(
            batch_size, self.n_random_actions
        )
        q2_policy = self.critic2(obs_rep, policy_actions).reshape(
            batch_size, self.n_random_actions
        )

        # CQL loss: logsumexp(Q) - E_D[Q]
        cat_q1 = torch.cat([q1_rand, q1_policy], dim=1)
        cat_q2 = torch.cat([q2_rand, q2_policy], dim=1)
        cql_loss = (
            torch.logsumexp(cat_q1, dim=1).mean()
            + torch.logsumexp(cat_q2, dim=1).mean()
            - q1_data.mean()
            - q2_data.mean()
        )

        # Auto-tune CQL alpha
        if self.auto_cql_alpha:
            cql_alpha = self.log_cql_alpha.exp()
            cql_alpha_loss = cql_alpha * (self.cql_target_value - cql_loss.detach())
            self.cql_alpha_optimizer.zero_grad(set_to_none=True)
            cql_alpha_loss.backward()
            self.cql_alpha_optimizer.step()
            self.cql_alpha = cql_alpha.item()

        critic_loss = bellman_loss + self.cql_alpha * cql_loss

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Actor update (same as SAC) ---
        new_actions, log_prob = self.actor.sample(obs)
        q1_new = self.critic1(obs, new_actions).squeeze(-1)
        q2_new = self.critic2(obs, new_actions).squeeze(-1)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_prob - q_new).mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        # Entropy alpha update
        if self.auto_entropy:
            alpha_loss = -(
                self.log_alpha * (log_prob.detach() + self.target_entropy)
            ).mean()
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        # Soft target update
        polyak_update(self.critic1, self.critic1_target, self.tau)
        polyak_update(self.critic2, self.critic2_target, self.tau)

        return {
            "critic_loss": bellman_loss.item() / 2,
            "cql_loss": cql_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha,
            "cql_alpha": self.cql_alpha,
        }

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Get action from the policy."""
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            if deterministic:
                return self.actor.deterministic(obs_t).squeeze(0).numpy()
            action, _ = self.actor.sample(obs_t)
            return action.squeeze(0).numpy()
