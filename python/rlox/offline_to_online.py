"""Offline-to-online fine-tuning with mixed buffer sampling.

Orchestrates the transition from offline pre-training to online fine-tuning
by mixing samples from an offline dataset and an online replay buffer.
Uses ``rlox.py_sample_mixed`` (Rust-accelerated) for efficient batch
construction.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import rlox._rlox_core as _core


class OfflineToOnline:
    """Offline-to-online fine-tuning with configurable mixing schedule.

    Samples mixed batches from an offline buffer and an online buffer,
    with an optional linear annealing schedule that gradually reduces the
    offline ratio to zero.

    Parameters
    ----------
    offline_buffer : ReplayBuffer
        Pre-filled offline dataset buffer.
    online_buffer : ReplayBuffer
        Online replay buffer (may start empty).
    offline_ratio : float
        Initial proportion of samples drawn from offline buffer (0.0 to 1.0).
    batch_size : int
        Total batch size for mixed sampling.
    anneal_steps : int or None
        If set, linearly anneal the offline ratio to 0 over this many steps.
        If None, the ratio stays constant.
    """

    def __init__(
        self,
        offline_buffer: Any,
        online_buffer: Any,
        offline_ratio: float = 0.5,
        batch_size: int = 256,
        anneal_steps: int | None = None,
    ) -> None:
        self.offline_buffer = offline_buffer
        self.online_buffer = online_buffer
        self.initial_offline_ratio = offline_ratio
        self.batch_size = batch_size
        self.anneal_steps = anneal_steps

    def get_offline_ratio(self, step: int) -> float:
        """Compute the current offline ratio given the training step.

        Parameters
        ----------
        step : int
            Current training step.

        Returns
        -------
        ratio : float in [0, initial_offline_ratio]
        """
        if self.anneal_steps is None or self.anneal_steps <= 0:
            return self.initial_offline_ratio

        progress = min(step / self.anneal_steps, 1.0)
        return self.initial_offline_ratio * (1.0 - progress)

    def sample_mixed(self, seed: int, step: int = 0) -> dict[str, Any]:
        """Sample a mixed batch from offline and online buffers.

        Parameters
        ----------
        seed : int
            RNG seed for reproducible sampling.
        step : int
            Current training step (used for annealing).

        Returns
        -------
        batch : dict with keys 'obs', 'actions', 'rewards', 'terminated', 'next_obs'
        """
        ratio = self.get_offline_ratio(step)
        return _core.py_sample_mixed(
            self.offline_buffer,
            self.online_buffer,
            ratio,
            self.batch_size,
            seed,
        )
