"""Default policy networks for on-policy RL algorithms.

Provides actor-critic network architectures used by PPO and A2C.
All policies use orthogonal initialisation (Andrychowicz et al., 2021)
with reduced gain on the policy head (0.01) to encourage initial
exploration.
"""

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
    """MLP actor-critic for discrete action spaces (e.g. CartPole).

    Separate actor and critic networks sharing no parameters. The actor
    outputs logits for a Categorical distribution; the critic outputs a
    scalar value estimate.

    Parameters
    ----------
    obs_dim : int
        Observation space dimensionality.
    n_actions : int
        Number of discrete actions.
    hidden : int
        Hidden layer width (default 64, matching CleanRL PPO).

    Required interface methods (called by PPOLoss / RolloutCollector):
        - ``get_action_and_logprob(obs)`` → (actions, log_probs)
        - ``get_value(obs)`` → values
        - ``get_logprob_and_entropy(obs, actions)`` → (log_probs, entropy)
    """

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


class ContinuousPolicy(nn.Module):
    """MLP actor-critic for continuous action spaces (e.g. Pendulum).

    The actor outputs a mean vector; a separate learnable ``log_std``
    parameter (state-independent) parameterises the diagonal Gaussian.
    The critic is a separate MLP producing a scalar value estimate.

    Both networks use orthogonal initialisation following the same
    convention as :class:`DiscretePolicy`.

    Parameters
    ----------
    obs_dim : int
        Observation space dimensionality.
    act_dim : int
        Action space dimensionality.
    hidden : int
        Hidden layer width (default 64).
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

        # Orthogonal init with sqrt(2) gain for hidden layers
        self.apply(lambda m: _orthogonal_init(m, gain=np.sqrt(2)))
        # Actor head: small init for exploration
        _orthogonal_init(self.actor[-1], gain=0.01)
        # Critic head: unit init
        _orthogonal_init(self.critic[-1], gain=1.0)

    def _build_dist(self, obs: torch.Tensor) -> torch.distributions.Normal:
        mean = self.actor(obs)
        std = self.log_std.exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def get_action_and_logprob(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self._build_dist(obs)
        actions = dist.sample()
        log_probs = dist.log_prob(actions).sum(dim=-1)
        return actions, log_probs

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def get_logprob_and_entropy(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self._build_dist(obs)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_probs, entropy
