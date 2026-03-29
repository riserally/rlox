"""Implicit Q-Learning (IQL).

Avoids querying out-of-distribution actions by using expectile
regression on the value function:
    V(s) ≈ E_τ[Q(s,a)]
where τ > 0.5 biases toward high-Q actions without evaluating
unseen actions.

Reference:
    I. Kostrikov, A. Nair, S. Levine,
    "Offline Reinforcement Learning with Implicit Q-Learning,"
    ICLR, 2022.
    https://arxiv.org/abs/2110.06169
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rlox.callbacks import Callback
from rlox.logging import LoggerCallback
from rlox.networks import QNetwork
from rlox.offline.base import OfflineAlgorithm, OfflineDataset


class _ValueNetwork(nn.Module):
    """Simple MLP value function V(s)."""

    def __init__(self, obs_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


class _GaussianPolicy(nn.Module):
    """Simple Gaussian policy for IQL actor extraction."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(obs))


def _expectile_loss(pred: torch.Tensor, target: torch.Tensor, tau: float) -> torch.Tensor:
    """Asymmetric L2 loss: L_τ(u) = |τ - 1(u < 0)| * u²."""
    diff = pred - target
    weight = torch.where(diff < 0, tau, 1.0 - tau)
    return (weight * diff.pow(2)).mean()


class IQL(OfflineAlgorithm):
    """Implicit Q-Learning.

    Parameters
    ----------
    dataset : OfflineDataset
        Offline dataset.
    obs_dim : int
        Observation dimension.
    act_dim : int
        Action dimension.
    expectile : float
        Expectile τ for value function regression (default 0.7).
    temperature : float
        β for advantage-weighted actor extraction (default 3.0).
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
    callbacks : list[Callback], optional
    logger : LoggerCallback, optional
    """

    def __init__(
        self,
        dataset: OfflineDataset,
        obs_dim: int,
        act_dim: int,
        expectile: float = 0.7,
        temperature: float = 3.0,
        hidden: int = 256,
        learning_rate: float = 3e-4,
        tau: float = 0.005,
        gamma: float = 0.99,
        batch_size: int = 256,
        callbacks: list[Callback] | None = None,
        logger: LoggerCallback | None = None,
    ):
        super().__init__(dataset, batch_size, callbacks, logger)
        self.expectile_tau = expectile
        self.temperature = temperature
        self.gamma = gamma
        self.tau_target = tau

        # Networks
        self.q1 = QNetwork(obs_dim, act_dim, hidden)
        self.q2 = QNetwork(obs_dim, act_dim, hidden)
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)
        self.value_fn = _ValueNetwork(obs_dim, hidden)
        self.actor = _GaussianPolicy(obs_dim, act_dim, hidden)

        self.q_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=learning_rate
        )
        self.v_optimizer = torch.optim.Adam(self.value_fn.parameters(), lr=learning_rate)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)

    def _update(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32)
        actions = torch.as_tensor(batch["actions"], dtype=torch.float32)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32)
        terminated = torch.as_tensor(batch["terminated"], dtype=torch.float32)

        if actions.dim() == 1:
            actions = actions.unsqueeze(-1)

        # --- Value function update (expectile regression) ---
        with torch.no_grad():
            q1 = self.q1_target(obs, actions).squeeze(-1)
            q2 = self.q2_target(obs, actions).squeeze(-1)
            q_min = torch.min(q1, q2)

        v = self.value_fn(obs)
        value_loss = _expectile_loss(v, q_min, self.expectile_tau)

        self.v_optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        self.v_optimizer.step()

        # --- Q-function update (standard Bellman with V as target) ---
        with torch.no_grad():
            next_v = self.value_fn(next_obs)
            target_q = rewards + self.gamma * (1.0 - terminated) * next_v

        q1_pred = self.q1(obs, actions).squeeze(-1)
        q2_pred = self.q2(obs, actions).squeeze(-1)
        q_loss = F.mse_loss(q1_pred, target_q) + F.mse_loss(q2_pred, target_q)

        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()

        # Soft target update
        for p, tp in zip(self.q1.parameters(), self.q1_target.parameters()):
            tp.data.mul_(1 - self.tau_target).add_(p.data * self.tau_target)
        for p, tp in zip(self.q2.parameters(), self.q2_target.parameters()):
            tp.data.mul_(1 - self.tau_target).add_(p.data * self.tau_target)

        # --- Actor update (advantage-weighted regression) ---
        with torch.no_grad():
            v_for_adv = self.value_fn(obs)
            q_for_adv = torch.min(
                self.q1_target(obs, actions).squeeze(-1),
                self.q2_target(obs, actions).squeeze(-1),
            )
            advantage = q_for_adv - v_for_adv
            weights = torch.exp(self.temperature * advantage)
            weights = weights.clamp(max=100.0)  # Prevent overflow

        pi = self.actor(obs)
        target_actions = actions.squeeze(-1) if actions.dim() > 1 and actions.shape[-1] == 1 else actions
        if pi.shape != target_actions.shape:
            target_actions = target_actions.reshape(pi.shape)
        actor_loss = (weights * (pi - target_actions).pow(2).mean(dim=-1)).mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        return {
            "value_loss": value_loss.item(),
            "q_loss": q_loss.item() / 2,
            "actor_loss": actor_loss.item(),
        }

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Get deterministic action."""
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            return self.actor(obs_t).squeeze(0).numpy()
