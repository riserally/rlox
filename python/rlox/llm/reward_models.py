"""Reward model serving: single model, ensemble, and multi-objective."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn as nn


class RewardModelServer:
    """Serve a reward model for scoring prompt-completion pairs.

    The model should accept tokenized input and return scalar rewards.
    For simplicity, this implementation calls the model directly (no HTTP).
    """

    def __init__(self, model: nn.Module, batch_size: int = 32):
        self.model = model
        self.batch_size = batch_size
        self.model.eval()

    @torch.no_grad()
    def score_batch(self, prompts: list[str], completions: list[str]) -> np.ndarray:
        """Score a batch of prompt-completion pairs.

        Returns a 1-D numpy array of scalar rewards.
        """
        n = len(prompts)
        # Simple: call model with batch index as input (stub tokenization)
        # In production, this would tokenize and feed through the model
        indices = torch.arange(n, dtype=torch.long)
        scores = self.model(indices)

        if isinstance(scores, torch.Tensor):
            result = scores.detach().cpu().numpy().astype(np.float64)
        else:
            result = np.array(scores, dtype=np.float64)

        return result.ravel()[:n]


class EnsembleRewardModel:
    """Weighted ensemble of multiple reward models."""

    def __init__(
        self,
        models: list[nn.Module],
        weights: list[float] | None = None,
    ):
        self.servers = [RewardModelServer(m) for m in models]
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        total = sum(weights)
        self.weights = [w / total for w in weights]

    def score_batch(self, prompts: list[str], completions: list[str]) -> np.ndarray:
        """Weighted average of individual model scores."""
        scores = None
        for server, weight in zip(self.servers, self.weights):
            s = server.score_batch(prompts, completions)
            if scores is None:
                scores = weight * s
            else:
                scores = scores + weight * s
        return scores


class MultiObjectiveReward:
    """Weighted combination of multiple reward objectives.

    Each objective is a callable: (prompts, completions) -> np.ndarray
    """

    def __init__(
        self,
        objectives: dict[str, Callable],
        weights: dict[str, float],
    ):
        self.objectives = objectives
        self.weights = weights

    def score_batch(self, prompts: list[str], completions: list[str]) -> np.ndarray:
        """Compute weighted sum: sum(weight_i * objective_i)."""
        scores = None
        for name, objective_fn in self.objectives.items():
            weight = self.weights.get(name, 1.0)
            s = objective_fn(prompts, completions)
            if scores is None:
                scores = weight * np.asarray(s, dtype=np.float64)
            else:
                scores = scores + weight * np.asarray(s, dtype=np.float64)
        return scores
