"""Direct Preference Optimization (DPO)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DPO:
    """DPO trainer for language models.

    Parameters
    ----------
    model : nn.Module
        Language model with ``forward(input_ids) -> logits``.
    ref_model : nn.Module
        Frozen reference model.
    beta : float
        Temperature parameter for the DPO loss.
    learning_rate : float
        Optimiser learning rate.
    """

    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        beta: float = 0.1,
        learning_rate: float = 1e-4,
    ):
        self.model = model
        self.ref_model = ref_model
        self.beta = beta
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def _sequence_logprobs(
        self, model: nn.Module, input_ids: torch.Tensor
    ) -> torch.Tensor:
        """Sum of per-token log-probs for positions 1..T."""
        logits = model(input_ids)
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        target = input_ids[:, 1:]
        per_token = log_probs.gather(2, target.unsqueeze(-1)).squeeze(-1)
        return per_token.sum(dim=-1)  # (B,)

    def compute_loss(
        self,
        prompt: torch.Tensor,
        chosen: torch.Tensor,
        rejected: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute DPO loss.

        Parameters
        ----------
        prompt : (B, P)
        chosen : (B, C)
        rejected : (B, R)

        Returns
        -------
        loss, metrics
        """
        chosen_ids = torch.cat([prompt, chosen], dim=1)
        rejected_ids = torch.cat([prompt, rejected], dim=1)

        policy_chosen = self._sequence_logprobs(self.model, chosen_ids)
        policy_rejected = self._sequence_logprobs(self.model, rejected_ids)

        with torch.no_grad():
            ref_chosen = self._sequence_logprobs(self.ref_model, chosen_ids)
            ref_rejected = self._sequence_logprobs(self.ref_model, rejected_ids)

        log_ratio_chosen = policy_chosen - ref_chosen
        log_ratio_rejected = policy_rejected - ref_rejected

        loss = -F.logsigmoid(self.beta * (log_ratio_chosen - log_ratio_rejected)).mean()

        with torch.no_grad():
            chosen_reward = self.beta * log_ratio_chosen.mean().item()
            rejected_reward = self.beta * log_ratio_rejected.mean().item()

        metrics = {
            "loss": loss.item(),
            "chosen_reward": chosen_reward,
            "rejected_reward": rejected_reward,
        }
        return loss, metrics
