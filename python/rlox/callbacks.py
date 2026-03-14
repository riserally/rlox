"""Callback system for training loops."""

from __future__ import annotations

from typing import Any


class Callback:
    """Base callback class. Override methods to hook into training events."""

    def on_training_start(self, **kwargs: Any) -> None:
        pass

    def on_step(self, **kwargs: Any) -> bool:
        """Called after each step. Return False to stop training."""
        return True

    def on_rollout_end(self, **kwargs: Any) -> None:
        pass

    def on_train_batch(self, **kwargs: Any) -> None:
        pass

    def on_eval(self, **kwargs: Any) -> None:
        pass

    def on_training_end(self, **kwargs: Any) -> None:
        pass


class EvalCallback(Callback):
    """Periodic evaluation with optional best model saving."""

    def __init__(
        self,
        eval_env: Any | None = None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        best_model_path: str | None = None,
    ):
        super().__init__()
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_model_path = best_model_path
        self.best_reward = float("-inf")
        self._step_count = 0

    def on_step(self, **kwargs: Any) -> bool:
        self._step_count += 1
        return True


class EarlyStoppingCallback(Callback):
    """Stop training when reward plateaus."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        super().__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.best_reward = float("-inf")
        self.wait = 0

    def on_step(self, **kwargs: Any) -> bool:
        reward = kwargs.get("reward", None)
        if reward is None:
            return True
        if reward > self.best_reward + self.min_delta:
            self.best_reward = reward
            self.wait = 0
        else:
            self.wait += 1
        if self.wait >= self.patience:
            return False
        return True


class CheckpointCallback(Callback):
    """Periodic checkpoint saving."""

    def __init__(self, save_freq: int = 10000, save_path: str = "checkpoints"):
        super().__init__()
        self.save_freq = save_freq
        self.save_path = save_path
        self._step_count = 0

    def on_step(self, **kwargs: Any) -> bool:
        self._step_count += 1
        return True


class CallbackList:
    """Run multiple callbacks in sequence."""

    def __init__(self, callbacks: list[Callback] | None = None):
        self.callbacks = callbacks or []

    def on_training_start(self, **kwargs: Any) -> None:
        for cb in self.callbacks:
            cb.on_training_start(**kwargs)

    def on_step(self, **kwargs: Any) -> bool:
        for cb in self.callbacks:
            if not cb.on_step(**kwargs):
                return False
        return True

    def on_rollout_end(self, **kwargs: Any) -> None:
        for cb in self.callbacks:
            cb.on_rollout_end(**kwargs)

    def on_train_batch(self, **kwargs: Any) -> None:
        for cb in self.callbacks:
            cb.on_train_batch(**kwargs)

    def on_eval(self, **kwargs: Any) -> None:
        for cb in self.callbacks:
            cb.on_eval(**kwargs)

    def on_training_end(self, **kwargs: Any) -> None:
        for cb in self.callbacks:
            cb.on_training_end(**kwargs)
