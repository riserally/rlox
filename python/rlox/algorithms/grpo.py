"""Group Relative Policy Optimization (GRPO) for LLM post-training."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import rlox


class GRPO:
    """GRPO trainer for language models.

    Parameters
    ----------
    model : nn.Module
        Language model with ``forward(input_ids) -> logits`` and
        ``generate(prompt_ids, max_new_tokens) -> token_ids``.
    ref_model : nn.Module
        Frozen reference model (same interface as *model*).
    reward_fn : callable
        ``(completions: list[torch.Tensor], prompts: torch.Tensor) -> list[float]``
    group_size : int
        Number of completions per prompt.
    kl_coef : float
        KL penalty coefficient.
    learning_rate : float
        Optimiser learning rate.
    max_new_tokens : int
        Maximum number of tokens to generate per prompt.
    """

    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        reward_fn: Callable[..., list[float]],
        group_size: int = 4,
        learning_rate: float = 1e-4,
        kl_coef: float = 0.1,
        max_new_tokens: int = 8,
    ):
        self.model = model
        self.ref_model = ref_model
        self.reward_fn = reward_fn
        self.group_size = group_size
        self.kl_coef = kl_coef
        self.max_new_tokens = max_new_tokens

        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def _get_per_token_logprobs(
        self, model: nn.Module, input_ids: torch.Tensor
    ) -> torch.Tensor:
        """Return per-token log-probs for tokens at positions 1..T."""
        logits = model(input_ids)  # (B, T, V)
        # Shift: predict token t from position t-1
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        target = input_ids[:, 1:]
        per_token = log_probs.gather(2, target.unsqueeze(-1)).squeeze(-1)
        return per_token  # (B, T-1)

    @torch.no_grad()
    def _generate_and_score(
        self, prompts: torch.Tensor
    ) -> tuple[torch.Tensor, list[float]]:
        """Generate completions for each prompt and score them."""
        n_prompts = prompts.shape[0]
        # Expand prompts by group_size
        expanded = prompts.repeat_interleave(self.group_size, dim=0)
        completions = self.model.generate(expanded, max_new_tokens=self.max_new_tokens)

        # Build list of completion token tensors for reward_fn
        comp_list = [completions[i] for i in range(completions.shape[0])]
        # Expand prompts for reward_fn call
        expanded_prompts = prompts.repeat_interleave(self.group_size, dim=0)
        rewards = self.reward_fn(comp_list, expanded_prompts)
        return completions, rewards

    def train_step(self, prompts: torch.Tensor) -> dict[str, float]:
        """One GRPO update on a batch of prompts."""
        completions, rewards_list = self._generate_and_score(prompts)
        rewards_np = np.array(rewards_list, dtype=np.float64)

        n_prompts = prompts.shape[0]
        total_loss = torch.tensor(0.0)
        total_kl = 0.0

        for i in range(n_prompts):
            start = i * self.group_size
            end = start + self.group_size
            group_rewards = rewards_np[start:end]
            group_completions = completions[start:end]

            # Group-relative advantages via Rust
            advantages = rlox.compute_group_advantages(group_rewards)
            advantages_t = torch.as_tensor(advantages, dtype=torch.float32)

            # Per-token log probs for policy and ref
            policy_logprobs = self._get_per_token_logprobs(self.model, group_completions)
            with torch.no_grad():
                ref_logprobs = self._get_per_token_logprobs(self.ref_model, group_completions)

            # Token-level KL via Rust (per sequence, summed)
            group_kl = 0.0
            for j in range(self.group_size):
                p = policy_logprobs[j].detach().cpu().numpy().astype(np.float64)
                r = ref_logprobs[j].cpu().numpy().astype(np.float64)
                group_kl += rlox.compute_token_kl(p, r)

            total_kl += group_kl / self.group_size

            # GRPO loss: -advantage * sum(token_logprobs) + kl_coef * kl
            seq_logprobs = policy_logprobs.sum(dim=-1)  # (group_size,)
            loss = -(advantages_t * seq_logprobs).mean()
            # Add KL penalty (use differentiable approx)
            kl_diff = (policy_logprobs - ref_logprobs).sum(dim=-1).mean()
            loss = loss + self.kl_coef * kl_diff
            total_loss = total_loss + loss

        total_loss = total_loss / n_prompts

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            "loss": total_loss.item(),
            "mean_reward": float(rewards_np.mean()),
            "kl": total_kl / n_prompts,
        }

    def train(self, prompts: torch.Tensor, n_epochs: int = 1) -> dict[str, float]:
        """Train over all prompts for *n_epochs*."""
        metrics: dict[str, float] = {}
        for _epoch in range(n_epochs):
            metrics = self.train_step(prompts)
        return metrics

    @torch.no_grad()
    def evaluate(self, prompts: torch.Tensor) -> float:
        """Return mean reward over prompts (single generation per prompt)."""
        completions = self.model.generate(prompts, max_new_tokens=self.max_new_tokens)
        comp_list = [completions[i] for i in range(completions.shape[0])]
        rewards = self.reward_fn(comp_list, prompts)
        return float(np.mean(rewards))
