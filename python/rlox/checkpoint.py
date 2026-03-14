"""Checkpoint save/load utilities."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class Checkpoint:
    """Save and load training checkpoints."""

    @staticmethod
    def save(
        path: str,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        config: dict[str, Any],
        buffer: Any | None = None,
        rng_state: Any | None = None,
    ) -> None:
        data: dict[str, Any] = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "config": config,
        }
        if buffer is not None:
            data["buffer"] = buffer
        if rng_state is not None:
            data["rng_state"] = rng_state
        data["torch_rng_state"] = torch.random.get_rng_state()
        torch.save(data, path)

    @staticmethod
    def load(path: str) -> dict[str, Any]:
        return torch.load(path, weights_only=False)
