"""Base class and protocol for offline RL algorithms."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from rlox.callbacks import Callback, CallbackList
from rlox.logging import LoggerCallback


@runtime_checkable
class OfflineDataset(Protocol):
    """Protocol for offline datasets — users can bring their own.

    Any object with ``sample(batch_size, seed)`` and ``__len__()`` works.
    The built-in ``rlox.OfflineDatasetBuffer`` satisfies this protocol.
    """

    def sample(self, batch_size: int, seed: int) -> dict[str, np.ndarray]:
        """Sample a batch of transitions.

        Returns dict with keys: obs, next_obs, actions, rewards, terminated.
        """
        ...

    def __len__(self) -> int: ...


class OfflineAlgorithm:
    """Base class for offline RL algorithms.

    Handles the shared training loop: sample → update → log → callback.
    Subclasses implement ``_update()`` with algorithm-specific logic.

    Parameters
    ----------
    dataset : OfflineDataset
        Offline dataset to sample from.
    batch_size : int
        Minibatch size for SGD.
    callbacks : list[Callback], optional
        Training callbacks.
    logger : LoggerCallback, optional
        Logger for metrics.
    """

    def __init__(
        self,
        dataset: OfflineDataset,
        batch_size: int = 256,
        callbacks: list[Callback] | None = None,
        logger: LoggerCallback | None = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.callbacks = CallbackList(callbacks)
        self.logger = logger
        self._global_step = 0

    def train(self, n_gradient_steps: int) -> dict[str, float]:
        """Run offline training for n_gradient_steps.

        Returns metrics from the last update step.
        """
        self.callbacks.on_training_start()
        metrics: dict[str, float] = {}

        for step in range(n_gradient_steps):
            batch = self.dataset.sample(self.batch_size, seed=step + self._global_step)
            metrics = self._update(batch)

            self._global_step += 1
            self.callbacks.on_train_batch(**metrics)

            should_continue = self.callbacks.on_step(
                step=self._global_step, algo=self, **metrics
            )
            if not should_continue:
                break

            if self.logger is not None and self._global_step % 1000 == 0:
                self.logger.on_train_step(self._global_step, metrics)

        self.callbacks.on_training_end()
        return metrics

    def _update(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        """Subclasses implement this with algorithm-specific logic."""
        raise NotImplementedError
