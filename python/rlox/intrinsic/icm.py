"""Intrinsic Curiosity Module (ICM) -- forward/inverse dynamics model.

Pathak et al., 2017. Learns a feature encoder, a forward dynamics model
that predicts next-state features from (feature(s), action), and an
inverse model that predicts the action from (feature(s), feature(s')).

The intrinsic reward is the forward prediction error:
    r_i = ||predicted_feature(s') - actual_feature(s')||^2

The module trains jointly on inverse loss (action prediction) and forward
loss (next-state feature prediction).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ICM:
    """Intrinsic Curiosity Module -- forward/inverse dynamics model.

    Parameters
    ----------
    obs_dim : int
        Observation dimensionality.
    action_dim : int
        Number of discrete actions (or action dimensionality).
    feature_dim : int
        Learned feature space dimensionality (default 64).
    hidden : int
        Hidden layer size (default 256).
    learning_rate : float
        Learning rate for all ICM components (default 1e-3).
    forward_weight : float
        Weight for forward loss relative to inverse loss (default 0.2).
        Total loss = (1 - forward_weight) * inverse_loss + forward_weight * forward_loss.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        feature_dim: int = 64,
        hidden: int = 256,
        learning_rate: float = 1e-3,
        forward_weight: float = 0.2,
    ) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.feature_dim = feature_dim
        self.forward_weight = forward_weight

        # Feature encoder: obs -> learned feature space
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, feature_dim),
        )

        # Forward model: (feature(s), action_onehot) -> predicted feature(s')
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, feature_dim),
        )

        # Inverse model: (feature(s), feature(s')) -> predicted action
        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

        # Joint optimizer for all components
        params = (
            list(self.encoder.parameters())
            + list(self.forward_model.parameters())
            + list(self.inverse_model.parameters())
        )
        self.optimizer = torch.optim.Adam(params, lr=learning_rate)

    def _encode_action(self, actions: torch.Tensor) -> torch.Tensor:
        """One-hot encode discrete actions."""
        if actions.dim() == 1:
            return F.one_hot(actions.long(), self.action_dim).float()
        return actions.float()

    def compute_intrinsic_reward(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor | None = None,
        actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute intrinsic reward as forward prediction error.

        Parameters
        ----------
        obs : (B, obs_dim) tensor
        next_obs : (B, obs_dim) tensor
        actions : (B,) or (B, action_dim) tensor

        Returns
        -------
        reward : (B,) tensor -- per-sample forward prediction error
        """
        if next_obs is None or actions is None:
            raise ValueError("ICM requires next_obs and actions")

        with torch.no_grad():
            feat_s = self.encoder(obs)
            feat_s_next = self.encoder(next_obs)
            action_enc = self._encode_action(actions)

            pred_feat_next = self.forward_model(
                torch.cat([feat_s, action_enc], dim=-1)
            )
            # Per-sample MSE as intrinsic reward
            reward = ((pred_feat_next - feat_s_next) ** 2).mean(dim=-1)

        return reward

    def get_loss(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute combined ICM loss (forward + inverse).

        Parameters
        ----------
        obs : (B, obs_dim) tensor
        next_obs : (B, obs_dim) tensor
        actions : (B,) or (B, action_dim) tensor

        Returns
        -------
        loss : scalar tensor
        """
        feat_s = self.encoder(obs)
        feat_s_next = self.encoder(next_obs)
        action_enc = self._encode_action(actions)

        # Forward loss: predict next-state features
        pred_feat_next = self.forward_model(
            torch.cat([feat_s, action_enc], dim=-1)
        )
        forward_loss = F.mse_loss(pred_feat_next, feat_s_next.detach())

        # Inverse loss: predict action from state transition
        action_pred = self.inverse_model(
            torch.cat([feat_s, feat_s_next], dim=-1)
        )
        inverse_loss = F.cross_entropy(action_pred, actions.long())

        loss = (
            (1 - self.forward_weight) * inverse_loss
            + self.forward_weight * forward_loss
        )
        return loss

    def update(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor | None = None,
        actions: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Train ICM on a batch of transitions.

        Parameters
        ----------
        obs : (B, obs_dim) tensor
        next_obs : (B, obs_dim) tensor
        actions : (B,) or (B, action_dim) tensor

        Returns
        -------
        info : dict with 'forward_loss', 'inverse_loss', 'icm_loss'
        """
        if next_obs is None or actions is None:
            raise ValueError("ICM requires next_obs and actions")

        feat_s = self.encoder(obs)
        feat_s_next = self.encoder(next_obs)
        action_enc = self._encode_action(actions)

        # Forward loss
        pred_feat_next = self.forward_model(
            torch.cat([feat_s, action_enc], dim=-1)
        )
        forward_loss = F.mse_loss(pred_feat_next, feat_s_next.detach())

        # Inverse loss
        action_pred = self.inverse_model(
            torch.cat([feat_s, feat_s_next], dim=-1)
        )
        inverse_loss = F.cross_entropy(action_pred, actions.long())

        loss = (
            (1 - self.forward_weight) * inverse_loss
            + self.forward_weight * forward_loss
        )

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        return {
            "forward_loss": forward_loss.item(),
            "inverse_loss": inverse_loss.item(),
            "icm_loss": loss.item(),
        }
