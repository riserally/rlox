"""Shared network utilities for off-policy algorithms."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def polyak_update(source: nn.Module, target: nn.Module, tau: float = 0.005) -> None:
    """Soft update: target = tau * source + (1 - tau) * target."""
    with torch.no_grad():
        for sp, tp in zip(source.parameters(), target.parameters()):
            tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)


class QNetwork(nn.Module):
    """Twin Q-value network for SAC / TD3.

    Takes (obs, action) concatenated as input, outputs scalar Q-value.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1))


class SquashedGaussianPolicy(nn.Module):
    """Gaussian policy with tanh squashing for SAC.

    Outputs actions in [-1, 1] with corrected log-probabilities.
    """

    LOG_STD_MIN = -20.0
    LOG_STD_MAX = 2.0

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden, act_dim)
        self.log_std_head = nn.Linear(hidden, act_dim)

    def forward(self, obs: torch.Tensor):
        h = self.shared(obs)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs: torch.Tensor):
        """Sample action and compute log-prob with tanh correction."""
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # reparameterised
        y_t = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        # Enforce action bounds correction
        log_prob = log_prob - torch.log(1.0 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)
        return y_t, log_prob

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        """Return deterministic action (mean through tanh)."""
        mean, _ = self.forward(obs)
        return torch.tanh(mean)


class DeterministicPolicy(nn.Module):
    """Deterministic policy for TD3."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256, max_action: float = 1.0):
        super().__init__()
        self.max_action = max_action
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.max_action * self.net(obs)


class DuelingQNetwork(nn.Module):
    """Dueling DQN architecture: separate value and advantage streams."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.feature(obs)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


class SimpleQNetwork(nn.Module):
    """Standard DQN Q-network."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)
