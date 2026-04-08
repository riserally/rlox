"""DQN with Rainbow extensions (Double DQN, Dueling, N-step, PER)."""

from __future__ import annotations

import collections
import copy
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

import rlox
from rlox.callbacks import Callback, CallbackList
from rlox.config import DQNConfig
from rlox.logging import LoggerCallback
from rlox.networks import SimpleQNetwork, DuelingQNetwork


class DQN:
    """DQN with optional Double DQN, Dueling architecture, N-step returns,
    and Prioritized Experience Replay.

    Uses rlox.ReplayBuffer or rlox.PrioritizedReplayBuffer for storage.
    """

    def __init__(
        self,
        env_id: str,
        buffer_size: int = 1_000_000,
        learning_rate: float = 1e-4,
        batch_size: int = 64,
        gamma: float = 0.99,
        target_update_freq: int = 1000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        learning_starts: int = 1000,
        double_dqn: bool = True,
        dueling: bool = False,
        n_step: int = 1,
        prioritized: bool = False,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        hidden: int = 256,
        train_freq: int = 1,
        gradient_steps: int = 1,
        max_grad_norm: float = 10.0,
        seed: int = 42,
        callbacks: list[Callback] | None = None,
        logger: LoggerCallback | None = None,
        compile: bool = False,
        q_network: nn.Module | None = None,
        buffer: object | None = None,
        collector: object | None = None,
        n_envs: int = 1,
    ):
        if isinstance(env_id, str):
            self.env = gym.make(env_id)
            self.env_id = env_id
        else:
            self.env = env_id
            self.env_id = (
                getattr(env_id.spec, "id", "custom")
                if hasattr(env_id, "spec") and env_id.spec
                else "custom"
            )
        self.gamma = gamma
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.target_update_freq = target_update_freq
        self.exploration_fraction = exploration_fraction
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.double_dqn = double_dqn
        self.dueling = dueling
        self.n_step = n_step
        self.prioritized = prioritized
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.max_grad_norm = max_grad_norm

        self.config = DQNConfig(
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            batch_size=batch_size,
            gamma=gamma,
            target_update_freq=target_update_freq,
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            learning_starts=learning_starts,
            double_dqn=double_dqn,
            dueling=dueling,
            n_step=n_step,
            prioritized=prioritized,
            alpha=alpha,
            beta_start=beta_start,
            hidden=hidden,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            max_grad_norm=max_grad_norm,
        )

        obs_dim = int(np.prod(self.env.observation_space.shape))
        act_dim = int(self.env.action_space.n)

        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # Networks — use custom if provided, otherwise default MLP
        if q_network is not None:
            self.q_network = q_network
        else:
            net_cls = DuelingQNetwork if dueling else SimpleQNetwork
            self.q_network = net_cls(obs_dim, act_dim, hidden)
        self.target_network = copy.deepcopy(self.q_network)

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Buffer — use custom if provided
        if buffer is not None:
            self.buffer = buffer
        elif prioritized:
            self.buffer = rlox.PrioritizedReplayBuffer(
                buffer_size, obs_dim, 1, alpha=alpha, beta=beta_start
            )
        else:
            self.buffer = rlox.ReplayBuffer(buffer_size, obs_dim, 1)

        # N-step return buffer (deque for O(1) popleft)
        self._n_step_buffer: collections.deque[tuple] = collections.deque()

        # Off-policy collector — enables multi-env collection
        self.collector = collector
        self.n_envs = n_envs

        # Callbacks and logger
        self.callbacks = CallbackList(callbacks)
        self.logger = logger
        self._global_step = 0

        if compile:
            from rlox.compile import compile_policy

            compile_policy(self)

    def _get_epsilon(self, step: int, total_timesteps: int) -> float:
        fraction = min(
            1.0, step / max(1, int(total_timesteps * self.exploration_fraction))
        )
        return self.exploration_initial_eps + fraction * (
            self.exploration_final_eps - self.exploration_initial_eps
        )

    def _store_transition(self, obs, action, reward, next_obs, terminated, truncated):
        self._n_step_buffer.append(
            (obs, action, reward, next_obs, terminated, truncated)
        )
        if len(self._n_step_buffer) < self.n_step:
            return

        # Compute n-step return
        R = 0.0
        for i in reversed(range(self.n_step)):
            _, _, r, _, done, trunc = self._n_step_buffer[i]
            R = r + self.gamma * R * (1.0 - float(done or trunc))

        first_obs, first_action, _, _, _, _ = self._n_step_buffer[0]
        _, _, _, last_next_obs, last_done, last_trunc = self._n_step_buffer[-1]

        if self.prioritized:
            self.buffer.push(
                np.asarray(first_obs, dtype=np.float32),
                np.array([float(first_action)], dtype=np.float32),
                float(R),
                bool(last_done),
                bool(last_trunc),
                np.asarray(last_next_obs, dtype=np.float32),
                priority=1.0,
            )
        else:
            self.buffer.push(
                np.asarray(first_obs, dtype=np.float32),
                np.array([float(first_action)], dtype=np.float32),
                float(R),
                bool(last_done),
                bool(last_trunc),
                np.asarray(last_next_obs, dtype=np.float32),
            )
        self._n_step_buffer.popleft()

    def train(self, total_timesteps: int) -> dict[str, float]:
        # Use OffPolicyCollector for multi-env, or default single-env loop
        if self.n_envs > 1 and self.collector is None:
            from rlox.off_policy_collector import OffPolicyCollector

            self.collector = OffPolicyCollector(
                env_id=self.env_id,
                n_envs=self.n_envs,
                buffer=self.buffer,
                seed=getattr(self, "seed", 42),
            )

        if self.collector is not None:
            return self._train_with_collector(total_timesteps)

        obs, _ = self.env.reset()
        episode_rewards: list[float] = []
        ep_reward = 0.0
        metrics: dict[str, float] = {}

        self.callbacks.on_training_start()

        for step in range(total_timesteps):
            eps = self._get_epsilon(step, total_timesteps)

            if np.random.random() < eps or step < self.learning_starts:
                action = int(self.env.action_space.sample())
            else:
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                    q_values = self.q_network(obs_t)
                    action = int(q_values.argmax(dim=-1).item())

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            ep_reward += float(reward)

            self._store_transition(obs, action, reward, next_obs, terminated, truncated)

            obs = next_obs
            if terminated or truncated:
                # Flush n-step buffer
                while self._n_step_buffer:
                    R = 0.0
                    for i in reversed(range(len(self._n_step_buffer))):
                        _, _, r, _, done, trunc = self._n_step_buffer[i]
                        R = r + self.gamma * R * (1.0 - float(done or trunc))
                    first_obs_b, first_action_b, _, _, _, _ = self._n_step_buffer[0]
                    _, _, _, last_next_obs_b, last_done_b, last_trunc_b = (
                        self._n_step_buffer[-1]
                    )
                    if self.prioritized:
                        self.buffer.push(
                            np.asarray(first_obs_b, dtype=np.float32),
                            np.array([float(first_action_b)], dtype=np.float32),
                            float(R),
                            bool(last_done_b),
                            bool(last_trunc_b),
                            np.asarray(last_next_obs_b, dtype=np.float32),
                            priority=1.0,
                        )
                    else:
                        self.buffer.push(
                            np.asarray(first_obs_b, dtype=np.float32),
                            np.array([float(first_action_b)], dtype=np.float32),
                            float(R),
                            bool(last_done_b),
                            bool(last_trunc_b),
                            np.asarray(last_next_obs_b, dtype=np.float32),
                        )
                    self._n_step_buffer.popleft()

                episode_rewards.append(ep_reward)
                ep_reward = 0.0
                obs, _ = self.env.reset()

            # Callback: on_step
            self._global_step += 1
            should_continue = self.callbacks.on_step(
                reward=ep_reward, step=self._global_step, algo=self
            )
            if not should_continue:
                break

            # Update — only every ``train_freq`` env steps, doing
            # ``gradient_steps`` SGD steps per training round (matches SB3
            # semantics).
            if (
                step >= self.learning_starts
                and len(self.buffer) >= self.batch_size
                and step % self.train_freq == 0
            ):
                for _ in range(self.gradient_steps):
                    metrics = self._update(step, total_timesteps)
                    self.callbacks.on_train_batch(**metrics)

                # Logger
                if self.logger is not None and self._global_step % 1000 == 0:
                    self.logger.on_train_step(self._global_step, metrics)

            # Target network update
            if step % self.target_update_freq == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())

        self.callbacks.on_training_end()

        metrics["mean_reward"] = (
            float(np.mean(episode_rewards)) if episode_rewards else 0.0
        )
        return metrics

    def _update(self, step: int, total_timesteps: int) -> dict[str, float]:
        if self.prioritized:
            batch = self.buffer.sample(self.batch_size, step)
            weights = torch.as_tensor(batch["weights"], dtype=torch.float32)
            indices = np.asarray(batch["indices"])
        else:
            batch = self.buffer.sample(self.batch_size, step)
            weights = torch.ones(self.batch_size)
            indices = None

        obs = torch.as_tensor(batch["obs"], dtype=torch.float32)
        actions = torch.as_tensor(batch["actions"], dtype=torch.long).squeeze(-1)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32)
        terminated = torch.as_tensor(batch["terminated"], dtype=torch.float32)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32)

        # Current Q
        q_values = self.q_network(obs)
        q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.double_dqn:
                # Use online network for action selection
                next_actions = self.q_network(next_obs).argmax(dim=-1)
                next_q = (
                    self.target_network(next_obs)
                    .gather(1, next_actions.unsqueeze(1))
                    .squeeze(1)
                )
            else:
                next_q = self.target_network(next_obs).max(dim=-1).values

            target_q = rewards + self.gamma**self.n_step * (1.0 - terminated) * next_q

        td_error = q - target_q
        # NOTE: rlox DQN uses MSE; SB3 DQN uses Huber (smooth_l1_loss). This
        # is an intentional divergence — see
        # docs/plans/benchmark-comparison-inconsistencies.md. Switching to
        # Huber here without other adjustments was empirically destabilizing
        # on CartPole with the SB3-zoo cadence. Tracked as a follow-up.
        loss = (weights * td_error.pow(2)).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Optional gradient clipping. ``max_grad_norm`` is exposed in
        # DQNConfig for users who want to mirror SB3's clip-at-10; the
        # default value is large enough that it is effectively a no-op,
        # preserving historical rlox behavior.
        if self.max_grad_norm < float("inf"):
            torch.nn.utils.clip_grad_norm_(
                self.q_network.parameters(), self.max_grad_norm
            )
        self.optimizer.step()

        # Update priorities
        if self.prioritized and indices is not None:
            new_priorities = td_error.abs().detach().numpy() + 1e-6
            self.buffer.update_priorities(
                indices.astype(np.uint64),
                new_priorities.astype(np.float64),
            )
            # Anneal beta
            frac = min(1.0, step / max(1, total_timesteps))
            self.buffer.set_beta(0.4 + frac * (1.0 - 0.4))

        return {"loss": loss.item()}

    def _train_with_collector(self, total_timesteps: int) -> dict[str, float]:
        """Train using OffPolicyCollector for multi-env data collection.

        Note: N-step returns are not supported with the collector path
        since the collector handles transitions directly. Use n_step=1
        when using multi-env collection.
        """
        collector = self.collector
        collector.reset()

        metrics: dict[str, float] = {}

        self.callbacks.on_training_start()

        def get_action(obs_batch: np.ndarray) -> np.ndarray:
            eps = self._get_epsilon(self._global_step, total_timesteps)
            batch_size = obs_batch.shape[0]
            if self._global_step < self.learning_starts:
                return np.random.randint(0, self.act_dim, size=(batch_size,))
            with torch.no_grad():
                obs_t = torch.as_tensor(obs_batch, dtype=torch.float32)
                q_values = self.q_network(obs_t)
                greedy = q_values.argmax(dim=-1).numpy()
            # Epsilon-greedy
            random_mask = np.random.random(batch_size) < eps
            random_actions = np.random.randint(0, self.act_dim, size=(batch_size,))
            return np.where(random_mask, random_actions, greedy)

        for step in range(total_timesteps):
            _, _, mean_ep_reward = collector.collect_step(
                get_action, step, total_timesteps
            )

            self._global_step += 1
            should_continue = self.callbacks.on_step(
                reward=mean_ep_reward, step=self._global_step, algo=self
            )
            if not should_continue:
                break

            if (
                step >= self.learning_starts
                and len(self.buffer) >= self.batch_size
                and step % self.train_freq == 0
            ):
                for _ in range(self.gradient_steps):
                    metrics = self._update(step, total_timesteps)
                    self.callbacks.on_train_batch(**metrics)

                if self.logger is not None and self._global_step % 1000 == 0:
                    self.logger.on_train_step(self._global_step, metrics)

            if step % self.target_update_freq == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())

        self.callbacks.on_training_end()

        completed = collector._completed_rewards
        metrics["mean_reward"] = float(np.mean(completed)) if completed else 0.0
        return metrics

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> int:
        """Get action from the policy."""
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            return int(self.q_network(obs_t).argmax(dim=-1).item())

    def save(self, path: str) -> None:
        """Save training checkpoint."""
        data: dict[str, Any] = {
            "q_network_state_dict": self.q_network.state_dict(),
            "target_network_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step": self._global_step,
            "config": self.config.to_dict(),
            "env_id": self.env_id,
            "torch_rng_state": torch.random.get_rng_state(),
        }
        torch.save(data, path)

    @classmethod
    def from_checkpoint(cls, path: str, env_id: str | None = None) -> DQN:
        """Restore DQN from a checkpoint."""
        from rlox.checkpoint import safe_torch_load

        data = safe_torch_load(path)
        config = data["config"]
        eid = env_id or data.get("env_id", "CartPole-v1")

        dqn = cls(env_id=eid, **config)
        dqn.q_network.load_state_dict(data["q_network_state_dict"])
        dqn.target_network.load_state_dict(data["target_network_state_dict"])
        dqn.optimizer.load_state_dict(data["optimizer_state_dict"])
        dqn._global_step = data.get("step", 0)

        if "torch_rng_state" in data:
            torch.random.set_rng_state(data["torch_rng_state"])

        return dqn
