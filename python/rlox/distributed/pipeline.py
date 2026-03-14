"""Decoupled collector/learner pipeline using threading and queue.Queue."""

from __future__ import annotations

import queue
import threading
from typing import Optional

import numpy as np
import torch

import rlox
from rlox.batch import RolloutBatch


class Pipeline:
    """Async collector/learner decoupling using threading.

    A background collector thread runs VecEnv, collects rollouts with random
    actions, computes GAE, and puts RolloutBatch objects into a bounded queue.
    The learner (main thread) consumes batches via next_batch / try_next_batch.
    """

    def __init__(
        self,
        env_id: str,
        n_envs: int = 4,
        n_steps: int = 32,
        channel_capacity: int = 4,
        seed: int = 0,
    ):
        self.env_id = env_id
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.seed = seed

        self._queue: queue.Queue[RolloutBatch] = queue.Queue(maxsize=channel_capacity)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._collector_loop, daemon=True)
        self._thread.start()

    def _collector_loop(self) -> None:
        """Background thread: step envs with random actions, compute GAE, enqueue."""
        env = rlox.VecEnv(n=self.n_envs, seed=self.seed, env_id=self.env_id)
        obs = env.reset_all()  # (n_envs, obs_dim)
        n_actions = 2  # CartPole default

        gamma = 0.99
        gae_lambda = 0.95

        while not self._stop_event.is_set():
            all_obs = []
            all_actions = []
            all_rewards = []
            all_dones = []
            all_log_probs = []
            all_values = []

            for _ in range(self.n_steps):
                # Random actions
                actions = np.random.randint(0, n_actions, size=self.n_envs).tolist()
                step_result = env.step_all(actions)

                obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
                all_obs.append(obs_tensor)
                all_actions.append(torch.tensor(actions, dtype=torch.long))
                all_rewards.append(
                    torch.as_tensor(step_result["rewards"].astype(np.float32))
                )

                terminated = step_result["terminated"].astype(bool)
                truncated = step_result["truncated"].astype(bool)
                dones = terminated | truncated
                all_dones.append(torch.as_tensor(dones.astype(np.float32)))

                # Dummy log_probs and values (random policy)
                all_log_probs.append(torch.zeros(self.n_envs))
                all_values.append(torch.zeros(self.n_envs))

                obs = step_result["obs"].copy()

            # Compute GAE per environment
            all_advantages = []
            all_returns = []
            for env_idx in range(self.n_envs):
                rewards_env = torch.stack([r[env_idx] for r in all_rewards])
                values_env = torch.stack([v[env_idx] for v in all_values])
                dones_env = torch.stack([d[env_idx] for d in all_dones])

                adv, ret = rlox.compute_gae(
                    rewards=rewards_env.numpy().astype(np.float64),
                    values=values_env.numpy().astype(np.float64),
                    dones=dones_env.numpy().astype(np.float64),
                    last_value=0.0,
                    gamma=gamma,
                    lam=gae_lambda,
                )
                all_advantages.append(
                    torch.as_tensor(adv, dtype=torch.float32)
                )
                all_returns.append(
                    torch.as_tensor(ret, dtype=torch.float32)
                )

            # Stack and flatten
            obs_t = torch.stack(all_obs)
            actions_t = torch.stack(all_actions)
            rewards_t = torch.stack(all_rewards)
            dones_t = torch.stack(all_dones)
            log_probs_t = torch.stack(all_log_probs)
            values_t = torch.stack(all_values)
            advantages_t = torch.stack(all_advantages).T
            returns_t = torch.stack(all_returns).T

            total = self.n_steps * self.n_envs
            batch = RolloutBatch(
                obs=obs_t.reshape(total, -1),
                actions=actions_t.reshape(total),
                rewards=rewards_t.reshape(total),
                dones=dones_t.reshape(total),
                log_probs=log_probs_t.reshape(total),
                values=values_t.reshape(total),
                advantages=advantages_t.reshape(total),
                returns=returns_t.reshape(total),
            )

            # Bounded put with timeout so we can check stop_event
            while not self._stop_event.is_set():
                try:
                    self._queue.put(batch, timeout=0.1)
                    break
                except queue.Full:
                    continue

    def next_batch(self, timeout: float = 10.0) -> Optional[RolloutBatch]:
        """Blocking wait for next batch with timeout."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def try_next_batch(self) -> Optional[RolloutBatch]:
        """Non-blocking attempt to get next batch."""
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def close(self) -> None:
        """Signal collector to stop and join the thread."""
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)
