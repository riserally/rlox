"""Online DPO: generate candidates, get preferences, run DPO update.

Extends DPO to the online setting: at each step, generate candidate
completions from the current policy, query a preference oracle, then
run a DPO update on the preferred/rejected pair.

Reference:
    Z. Guo, A. Rashid, B. Suber, S. Sharma, D. Sui, et al.,
    "Direct Language Model Alignment from Online AI Feedback,"
    arXiv:2402.04792, 2024.
    https://arxiv.org/abs/2402.04792

See also:
    R. Rafailov, A. Sharma, E. Mitchell, C. Manning, S. Ermon, C. Finn,
    "Direct Preference Optimization: Your Language Model is Secretly a
    Reward Model," NeurIPS, 2023. https://arxiv.org/abs/2305.18290
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class OnlineDPO:
    """Online Direct Preference Optimization.

    Each step: generate pairs of candidates per prompt, query a preference
    function, then run a DPO update on the preferred/rejected pair.
    """

    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        preference_fn: Callable[[list[tuple[torch.Tensor, torch.Tensor]]], list[int]],
        beta: float = 0.1,
        learning_rate: float = 1e-4,
        max_new_tokens: int = 8,
    ):
        self.model = model
        self.ref_model = ref_model
        self.preference_fn = preference_fn
        self.beta = beta
        self.max_new_tokens = max_new_tokens
        self.max_grad_norm = 1.0
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def _sequence_logprobs(
        self, model: nn.Module, input_ids: torch.Tensor
    ) -> torch.Tensor:
        logits = model(input_ids)
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        target = input_ids[:, 1:]
        per_token = log_probs.gather(2, target.unsqueeze(-1)).squeeze(-1)
        return per_token.sum(dim=-1)

    @torch.no_grad()
    def _generate_pair(self, prompt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate two candidate completions for a prompt."""
        expanded = prompt.unsqueeze(0).expand(2, -1)
        completions = self.model.generate(expanded, max_new_tokens=self.max_new_tokens)
        return completions[0], completions[1]

    def train_step(self, prompts: torch.Tensor) -> dict[str, float]:
        """One OnlineDPO update on a batch of prompts."""
        n_prompts = prompts.shape[0]
        pairs = []
        prompt_list = []

        for i in range(n_prompts):
            c1, c2 = self._generate_pair(prompts[i])
            pairs.append((c1, c2))
            prompt_list.append(prompts[i])

        # Get preferences (index 0 or 1)
        preferences = self.preference_fn(pairs)

        losses = []
        for i in range(n_prompts):
            c1, c2 = pairs[i]
            pref = preferences[i]
            chosen = c1 if pref == 0 else c2
            rejected = c2 if pref == 0 else c1

            chosen_ids = chosen.unsqueeze(0)
            rejected_ids = rejected.unsqueeze(0)

            policy_chosen = self._sequence_logprobs(self.model, chosen_ids)
            policy_rejected = self._sequence_logprobs(self.model, rejected_ids)

            with torch.no_grad():
                ref_chosen = self._sequence_logprobs(self.ref_model, chosen_ids)
                ref_rejected = self._sequence_logprobs(self.ref_model, rejected_ids)

            log_ratio_chosen = policy_chosen - ref_chosen
            log_ratio_rejected = policy_rejected - ref_rejected

            loss = -F.logsigmoid(
                self.beta * (log_ratio_chosen - log_ratio_rejected)
            ).mean()
            losses.append(loss)

        total_loss = torch.stack(losses).mean()

        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        if self.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {"loss": total_loss.item()}
