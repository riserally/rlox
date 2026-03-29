"""Best-of-N sampling: generate N candidates, score, return best.

A simple but effective inference-time alignment strategy. For each prompt,
generate N completions, score them with a reward model, and return the
highest-scoring completion. No training required.

Reference:
    Y. Nakano, J. Hilton, S. Balaji, J. Wu, et al.,
    "WebGPT: Browser-assisted question-answering with human feedback,"
    arXiv:2112.09332, 2021.
    https://arxiv.org/abs/2112.09332

See also:
    A. Gao, J. Schulman, J. Hilton,
    "Scaling Laws for Reward Model Overoptimization,"
    ICML, 2023. https://arxiv.org/abs/2210.10760
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


class BestOfN:
    """Best-of-N rejection sampling.

    Generate N completions per prompt, score them with a reward function,
    and return the best completion for each prompt.
    """

    def __init__(
        self,
        model: nn.Module,
        reward_fn: Callable[..., list[float]],
        n: int = 4,
        max_new_tokens: int = 8,
    ):
        self.model = model
        self.reward_fn = reward_fn
        self.n = n
        self.max_new_tokens = max_new_tokens

    @torch.no_grad()
    def generate(self, prompts: torch.Tensor) -> torch.Tensor:
        """Generate best-of-N completions for each prompt.

        Parameters
        ----------
        prompts : (B, P) token ids

        Returns
        -------
        (B, P + T) best completions
        """
        n_prompts = prompts.shape[0]

        # Expand prompts by N
        expanded = prompts.repeat_interleave(self.n, dim=0)  # (B*N, P)
        completions = self.model.generate(expanded, max_new_tokens=self.max_new_tokens)

        # Score all completions
        comp_list = [completions[i] for i in range(completions.shape[0])]
        scores = self.reward_fn(comp_list, expanded)

        # Select best per prompt
        best_completions = []
        for i in range(n_prompts):
            start = i * self.n
            end = start + self.n
            group_scores = scores[start:end]
            best_idx = start + max(range(self.n), key=lambda j: group_scores[j])
            best_completions.append(completions[best_idx])

        return torch.stack(best_completions)
