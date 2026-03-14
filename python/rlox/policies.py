"""Default policy networks for RL algorithms."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def _orthogonal_init(module: nn.Module, gain: float = 1.0) -> None:
    """Apply orthogonal initialisation to Linear layers."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class DiscretePolicy(nn.Module):
    """MLP actor-critic for discrete action spaces (e.g. CartPole)."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.apply(lambda m: _orthogonal_init(m, gain=np.sqrt(2)))
        # Policy head gets smaller init
        _orthogonal_init(self.actor[-1], gain=0.01)
        # Value head gets unit init
        _orthogonal_init(self.critic[-1], gain=1.0)

    def get_action_and_logprob(self, obs: torch.Tensor):
        logits = self.actor(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def get_logprob_and_entropy(self, obs: torch.Tensor, actions: torch.Tensor):
        logits = self.actor(obs)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy()
