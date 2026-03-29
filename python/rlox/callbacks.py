"""Callback system for training loops.

Provides hooks for monitoring, evaluation, checkpointing, and early
stopping during training. Callbacks are composable via :class:`CallbackList`.

Example
-------
>>> from rlox.callbacks import EarlyStoppingCallback, CallbackList
>>> callbacks = CallbackList([
...     EarlyStoppingCallback(patience=10),
... ])
"""

from __future__ import annotations

from typing import Any


class Callback:
    """Base callback class. Override methods to hook into training events.

    All ``on_*`` methods are no-ops by default. Override only the ones
    you need. ``on_step`` should return ``True`` to continue training
    or ``False`` to stop early.
    """

    def on_training_start(self, **kwargs: Any) -> None:
        """Called once before the training loop begins."""
        pass

    def on_step(self, **kwargs: Any) -> bool:
        """Called after each environment step. Return False to stop training."""
        return True

    def on_rollout_end(self, **kwargs: Any) -> None:
        """Called after a complete rollout is collected."""
        pass

    def on_train_batch(self, **kwargs: Any) -> None:
        """Called after each SGD minibatch update."""
        pass

    def on_eval(self, **kwargs: Any) -> None:
        """Called after an evaluation episode completes."""
        pass

    def on_training_end(self, **kwargs: Any) -> None:
        """Called once after the training loop finishes."""
        pass


class EvalCallback(Callback):
    """Periodic evaluation with optional best model saving.

    Requires the training loop to pass ``algo=self`` in ``on_step()`` kwargs.
    The algorithm must implement ``predict(obs, deterministic=True)``.

    Parameters
    ----------
    eval_env : gymnasium.Env, optional
        Environment for evaluation. If None, created from algo's env_id.
    eval_freq : int
        Evaluate every ``eval_freq`` steps (default 10000).
    n_eval_episodes : int
        Number of evaluation episodes (default 5).
    best_model_path : str, optional
        If set, save the best model weights to this path.
    verbose : bool
        Print evaluation results (default True).
    """

    def __init__(
        self,
        eval_env: Any | None = None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        best_model_path: str | None = None,
        verbose: bool = True,
    ):
        super().__init__()
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_model_path = best_model_path
        self.best_reward = float("-inf")
        self.verbose = verbose
        self._step_count = 0
        self.eval_results: list[tuple[int, float]] = []

    def on_step(self, **kwargs: Any) -> bool:
        step = kwargs.get("step", None)
        if step is not None:
            self._step_count = step
        else:
            self._step_count += 1

        if self._step_count % self.eval_freq != 0:
            return True

        algo = kwargs.get("algo")
        if algo is None or not hasattr(algo, "predict"):
            return True

        # Create eval env if needed
        env = self.eval_env
        if env is None:
            env_id = getattr(algo, "env_id", None)
            if env_id is None:
                return True
            import gymnasium as gym

            env = gym.make(env_id)

        rewards = []
        for _ in range(self.n_eval_episodes):
            obs, _ = env.reset()
            ep_reward, done = 0.0, False
            while not done:
                action = algo.predict(obs, deterministic=True)
                obs, r, term, trunc, _ = env.step(action)
                ep_reward += r
                done = term or trunc
            rewards.append(ep_reward)

        mean_reward = sum(rewards) / len(rewards)
        self.eval_results.append((self._step_count, mean_reward))

        if self.verbose:
            print(f"  [eval] step={self._step_count}  mean_reward={mean_reward:.1f}")

        if self.best_model_path and mean_reward > self.best_reward:
            self.best_reward = mean_reward
            if hasattr(algo, "save"):
                algo.save(self.best_model_path)
                if self.verbose:
                    print(f"  [eval] New best model saved ({mean_reward:.1f})")

        # Close env if we created it
        if self.eval_env is None:
            env.close()

        return True


class EarlyStoppingCallback(Callback):
    """Stop training when reward plateaus.

    Parameters
    ----------
    patience : int
        Number of steps without improvement before stopping (default 10).
    min_delta : float
        Minimum improvement to count as progress (default 0.0).
    """

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
    """Periodic checkpoint saving.

    Requires the training loop to pass ``algo=self`` in ``on_step()`` kwargs.
    The algorithm must implement ``save(path)``.

    Parameters
    ----------
    save_freq : int
        Save a checkpoint every ``save_freq`` steps (default 10000).
    save_path : str
        Directory for checkpoint files (default "checkpoints").
    verbose : bool
        Print when saving (default True).
    """

    def __init__(
        self,
        save_freq: int = 10000,
        save_path: str = "checkpoints",
        verbose: bool = True,
    ):
        super().__init__()
        self.save_freq = save_freq
        self.save_path = save_path
        self.verbose = verbose
        self._step_count = 0

    def on_step(self, **kwargs: Any) -> bool:
        import os

        step = kwargs.get("step", None)
        if step is not None:
            self._step_count = step
        else:
            self._step_count += 1

        if self._step_count % self.save_freq != 0:
            return True

        algo = kwargs.get("algo")
        if algo is None or not hasattr(algo, "save"):
            return True

        os.makedirs(self.save_path, exist_ok=True)
        path = os.path.join(self.save_path, f"checkpoint_{self._step_count}.pt")
        algo.save(path)
        if self.verbose:
            print(f"  [checkpoint] Saved to {path}")

        return True


class ProgressBarCallback(Callback):
    """Display a tqdm progress bar during training."""

    def __init__(self) -> None:
        super().__init__()
        self._pbar: Any = None

    def on_training_start(self, **kwargs: Any) -> None:
        total = kwargs.get("total_timesteps", 0)
        try:
            from tqdm.auto import tqdm

            self._pbar = tqdm(total=total, unit="step", desc="Training")
        except ImportError:
            self._pbar = None

    def on_step(self, **kwargs: Any) -> bool:
        if self._pbar is not None:
            self._pbar.update(1)
            reward = kwargs.get("reward")
            if reward is not None:
                self._pbar.set_postfix(reward=f"{reward:.1f}")
        return True

    def on_training_end(self, **kwargs: Any) -> None:
        if self._pbar is not None:
            self._pbar.close()


class TimingCallback(Callback):
    """Measure wall-clock time of each training phase."""

    def __init__(self) -> None:
        super().__init__()
        self._phase_times: dict[str, float] = {}
        self._phase_start: float = 0.0
        self._current_phase: str | None = None

    def _switch_phase(self, name: str) -> None:
        import time

        now = time.perf_counter()
        if self._current_phase is not None:
            elapsed = now - self._phase_start
            self._phase_times[self._current_phase] = (
                self._phase_times.get(self._current_phase, 0.0) + elapsed
            )
        self._current_phase = name
        self._phase_start = now

    def on_training_start(self, **kwargs: Any) -> None:
        import time

        self._phase_start = time.perf_counter()
        self._current_phase = "env_step"

    def on_step(self, **kwargs: Any) -> bool:
        self._switch_phase("env_step")
        return True

    def on_rollout_end(self, **kwargs: Any) -> None:
        self._switch_phase("gae_compute")

    def on_train_batch(self, **kwargs: Any) -> None:
        self._switch_phase("gradient_update")

    def on_training_end(self, **kwargs: Any) -> None:
        self._switch_phase("done")

    def summary(self) -> dict[str, float]:
        """Return percentage of time spent in each phase."""
        total = sum(self._phase_times.values())
        if total == 0:
            return {}
        return {k: v / total * 100 for k, v in self._phase_times.items()}


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
