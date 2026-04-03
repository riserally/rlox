"""Random Network Distillation (RND) intrinsic motivation.

Burda et al., 2019. A fixed random target network and a trainable
predictor network. The prediction error serves as an intrinsic reward
signal -- novel states have high error, familiar states have low error.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RND:
    """Random Network Distillation intrinsic motivation module.

    Parameters
    ----------
    obs_dim : int
        Observation dimensionality.
    hidden : int
        Hidden layer size for both target and predictor (default 256).
    output_dim : int
        Embedding dimensionality (default 64).
    learning_rate : float
        Predictor learning rate (default 1e-3).
    """

    def __init__(
        self,
        obs_dim: int,
        hidden: int = 256,
        output_dim: int = 64,
        learning_rate: float = 1e-3,
    ) -> None:
        self.obs_dim = obs_dim

        # Fixed random target network -- never trained
        self.target = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim),
        )
        for p in self.target.parameters():
            p.requires_grad = False

        # Trainable predictor network
        self.predictor = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim),
        )

        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=learning_rate)

    def compute_intrinsic_reward(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute intrinsic reward as MSE between target and predictor.

        Parameters
        ----------
        obs : (B, obs_dim) tensor

        Returns
        -------
        reward : (B,) tensor -- per-sample prediction error
        """
        with torch.no_grad():
            target_features = self.target(obs)
            pred_features = self.predictor(obs)
            # Per-sample MSE
            reward = ((target_features - pred_features) ** 2).mean(dim=-1)
        return reward

    def update(self, obs: torch.Tensor) -> dict[str, float]:
        """Train the predictor to match the target on given observations.

        Parameters
        ----------
        obs : (B, obs_dim) tensor

        Returns
        -------
        info : dict with 'rnd_loss'
        """
        target_features = self.target(obs).detach()
        pred_features = self.predictor(obs)

        loss = F.mse_loss(pred_features, target_features)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        return {"rnd_loss": loss.item()}
