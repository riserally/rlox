"""PPO loss function (Schulman et al., 2017).

Implements the clipped surrogate objective with optional clipped value loss
and entropy bonus. This is a stateless loss calculator — it does not own
an optimiser or perform gradient steps.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PPOLoss:
    """Clipped PPO objective with optional clipped value loss and entropy bonus.

    Computes three loss components:
    - **Policy loss**: clipped surrogate objective (prevents large policy updates)
    - **Value loss**: optionally clipped MSE on value predictions
    - **Entropy bonus**: encourages exploration (subtracted from total loss)

    Parameters
    ----------
    clip_eps : float
        Clipping range for the probability ratio (default 0.2).
    vf_coef : float
        Value function loss coefficient (default 0.5).
    ent_coef : float
        Entropy bonus coefficient (default 0.01).
    max_grad_norm : float
        Maximum gradient norm for clipping (stored but applied externally).
    clip_vloss : bool
        Whether to clip the value loss (default True, matches CleanRL).

    Example
    -------
    >>> loss_fn = PPOLoss(clip_eps=0.2, vf_coef=0.5, ent_coef=0.01)
    >>> total_loss, metrics = loss_fn(policy, obs, actions, old_log_probs,
    ...                               advantages, returns, old_values)
    >>> total_loss.backward()
    """

    def __init__(
        self,
        clip_eps: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        clip_vloss: bool = True,
    ):
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.clip_vloss = clip_vloss

    def __call__(
        self,
        policy: nn.Module,
        obs: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        old_values: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the PPO loss.

        Parameters
        ----------
        policy : nn.Module
            Must implement ``get_logprob_and_entropy(obs, actions)``
            and ``get_value(obs)``.
        obs : Tensor of shape (B, obs_dim)
        actions : Tensor of shape (B,) or (B, act_dim)
        old_log_probs : Tensor of shape (B,)
            Log-probs from the collection policy.
        advantages : Tensor of shape (B,)
            Should be normalised before passing in.
        returns : Tensor of shape (B,)
            Target values for the critic.
        old_values : Tensor of shape (B,)
            Value estimates from the collection policy.

        Returns
        -------
        (total_loss, metrics) : tuple
            ``total_loss`` is a scalar tensor for backprop.
            ``metrics`` dict contains: policy_loss, value_loss, entropy,
            approx_kl, clip_fraction.
        """
        new_log_probs, entropy = policy.get_logprob_and_entropy(obs, actions)

        # Policy loss (clipped surrogate)
        log_ratio = new_log_probs - old_log_probs
        ratio = log_ratio.exp()
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
        policy_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Value loss
        new_values = policy.get_value(obs)
        if self.clip_vloss:
            v_clipped = old_values + torch.clamp(
                new_values - old_values, -self.clip_eps, self.clip_eps
            )
            vf_loss1 = (new_values - returns) ** 2
            vf_loss2 = (v_clipped - returns) ** 2
            value_loss = 0.5 * torch.max(vf_loss1, vf_loss2).mean()
        else:
            value_loss = 0.5 * ((new_values - returns) ** 2).mean()

        entropy_loss = entropy.mean()

        total_loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy_loss

        # Diagnostics
        with torch.no_grad():
            approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
            clip_fraction = ((ratio - 1.0).abs() > self.clip_eps).float().mean().item()

        metrics = {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy_loss.item(),
            "approx_kl": approx_kl,
            "clip_fraction": clip_fraction,
        }
        return total_loss, metrics
