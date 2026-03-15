"""Group Relative Policy Optimization (GRPO) for LLM post-training."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import rlox
from rlox.callbacks import Callback, CallbackList
from rlox.checkpoint import Checkpoint
from rlox.logging import LoggerCallback


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
    callbacks : list[Callback], optional
        Training callbacks.
    logger : LoggerCallback, optional
        Logger for metrics.
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
        callbacks: list[Callback] | None = None,
        logger: LoggerCallback | None = None,
    ):
        self.model = model
        self.ref_model = ref_model
        self.reward_fn = reward_fn
        self.group_size = group_size
        self.kl_coef = kl_coef
        self.max_new_tokens = max_new_tokens

        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.callbacks = CallbackList(callbacks)
        self.logger = logger
        self._global_step = 0

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
        # Expand prompts by group_size
        expanded = prompts.repeat_interleave(self.group_size, dim=0)
        completions = self.model.generate(expanded, max_new_tokens=self.max_new_tokens)

        # Build list of completion token tensors for reward_fn
        comp_list = [completions[i] for i in range(completions.shape[0])]
        expanded_prompts = prompts.repeat_interleave(self.group_size, dim=0)
        rewards = self.reward_fn(comp_list, expanded_prompts)
        return completions, rewards

    def train_step(self, prompts: torch.Tensor) -> dict[str, float]:
        """One GRPO update on a batch of prompts.

        Uses batched group advantage computation via Rust for all prompts
        at once, avoiding a Python loop per prompt.
        """
        completions, rewards_list = self._generate_and_score(prompts)
        rewards_np = np.array(rewards_list, dtype=np.float64)

        n_prompts = prompts.shape[0]

        # Batched group advantages via Rust — all groups at once
        advantages_np = np.array(
            rlox.compute_batch_group_advantages(rewards_np, self.group_size),
            dtype=np.float32,
        )
        advantages_t = torch.as_tensor(advantages_np)

        # Per-token log probs for all completions at once
        policy_logprobs = self._get_per_token_logprobs(self.model, completions)
        with torch.no_grad():
            ref_logprobs = self._get_per_token_logprobs(self.ref_model, completions)

        # Token-level KL via Rust (batched)
        total_kl = 0.0
        for j in range(completions.shape[0]):
            p = policy_logprobs[j].detach().cpu().numpy().astype(np.float64)
            r = ref_logprobs[j].cpu().numpy().astype(np.float64)
            total_kl += rlox.compute_token_kl(p, r)
        mean_kl = total_kl / completions.shape[0]

        # GRPO loss: -advantage * sum(token_logprobs) + kl_coef * kl
        seq_logprobs = policy_logprobs.sum(dim=-1)  # (n_prompts * group_size,)
        loss = -(advantages_t * seq_logprobs).mean()
        # Add KL penalty (differentiable approximation)
        kl_diff = (policy_logprobs - ref_logprobs).sum(dim=-1).mean()
        loss = loss + self.kl_coef * kl_diff

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._global_step += 1
        metrics = {
            "loss": loss.item(),
            "mean_reward": float(rewards_np.mean()),
            "kl": mean_kl,
        }

        self.callbacks.on_train_batch(**metrics)
        if self.logger is not None:
            self.logger.on_train_step(self._global_step, metrics)

        return metrics

    def train(self, prompts: torch.Tensor, n_epochs: int = 1) -> dict[str, float]:
        """Train over all prompts for *n_epochs*."""
        self.callbacks.on_training_start()
        metrics: dict[str, float] = {}
        for _epoch in range(n_epochs):
            metrics = self.train_step(prompts)
            should_continue = self.callbacks.on_step(
                reward=metrics.get("mean_reward", 0.0), step=self._global_step
            )
            if not should_continue:
                break
        self.callbacks.on_training_end()
        return metrics

    @torch.no_grad()
    def evaluate(self, prompts: torch.Tensor) -> float:
        """Return mean reward over prompts (single generation per prompt)."""
        completions = self.model.generate(prompts, max_new_tokens=self.max_new_tokens)
        comp_list = [completions[i] for i in range(completions.shape[0])]
        rewards = self.reward_fn(comp_list, prompts)
        return float(np.mean(rewards))

    def save(self, path: str) -> None:
        """Save training checkpoint."""
        Checkpoint.save(
            path,
            model=self.model,
            optimizer=self.optimizer,
            step=self._global_step,
            config={
                "group_size": self.group_size,
                "kl_coef": self.kl_coef,
                "max_new_tokens": self.max_new_tokens,
            },
        )
