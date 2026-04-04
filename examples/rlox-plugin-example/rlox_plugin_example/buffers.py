"""Example custom buffer plugin.

Wraps rlox.ReplayBuffer with transition-level logging to demonstrate
how to register a custom buffer with rlox.
"""

from __future__ import annotations

import logging
from typing import Any

import rlox

from rlox.plugins import register_buffer

logger = logging.getLogger(__name__)


@register_buffer("logging-replay")
class LoggingReplayBuffer:
    """Replay buffer that logs every push and sample operation.

    Thin wrapper around :class:`rlox.ReplayBuffer` for debugging.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions.
    obs_dim : int
        Observation dimensionality.
    act_dim : int
        Action dimensionality.
    """

    def __init__(self, capacity: int, obs_dim: int, act_dim: int):
        self._inner = rlox.ReplayBuffer(capacity, obs_dim, act_dim)
        self._push_count = 0

    def push(self, *args: Any) -> None:
        self._push_count += 1
        if self._push_count % 1000 == 0:
            logger.debug("LoggingReplayBuffer: %d pushes, len=%d", self._push_count, len(self))
        self._inner.push(*args)

    def sample(self, batch_size: int, step: int = 0) -> dict[str, Any]:
        logger.debug("LoggingReplayBuffer: sampling %d at step %d", batch_size, step)
        return self._inner.sample(batch_size, step)

    def __len__(self) -> int:
        return len(self._inner)
