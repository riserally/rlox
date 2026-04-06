"""Checkpoint save/load utilities."""

from __future__ import annotations

import io
import logging
import os
import threading
from typing import Any

import torch
import torch.nn as nn

_logger = logging.getLogger(__name__)


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
        async_save: bool = False,
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

        if async_save:
            # Serialize to buffer on main thread, write on background thread
            buf = io.BytesIO()
            torch.save(data, buf)
            raw = buf.getvalue()

            def _write():
                tmp = path + ".tmp"
                with open(tmp, "wb") as f:
                    f.write(raw)
                os.replace(tmp, path)

            threading.Thread(target=_write, daemon=True).start()
        else:
            torch.save(data, path)

    @staticmethod
    def load(path: str) -> dict[str, Any]:
        return safe_torch_load(path)


def safe_torch_load(path: str, *, allow_unsafe: bool = False) -> dict[str, Any]:
    """Load a checkpoint with ``weights_only=True``.

    Parameters
    ----------
    path : str
        Path to the checkpoint file.
    allow_unsafe : bool
        If True, falls back to ``weights_only=False`` for legacy checkpoints
        that contain pickled objects. **This enables arbitrary code execution
        from the checkpoint file.** Default False.

    Raises
    ------
    RuntimeError
        If the checkpoint cannot be loaded safely and ``allow_unsafe=False``.
    """
    try:
        return torch.load(path, weights_only=True)
    except Exception as exc:
        if allow_unsafe:
            _logger.warning(
                "Loading %s with weights_only=False (unsafe). "
                "Re-save this checkpoint to remove the pickle dependency.",
                path,
            )
            return torch.load(path, weights_only=False)
        raise RuntimeError(
            f"Cannot load {path} with weights_only=True. "
            f"The checkpoint may contain pickled objects. "
            f"Pass allow_unsafe=True to load anyway (security risk), "
            f"or re-save the checkpoint with the latest rlox version."
        ) from exc
