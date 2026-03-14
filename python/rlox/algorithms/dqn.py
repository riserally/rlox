"""DQN with Rainbow extensions (Double DQN, Dueling, N-step, PER)."""

from __future__ import annotations

import copy
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

import rlox
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
        seed: int = 42,
    ):
        self.env = gym.make(env_id)
        self.env_id = env_id
        self.gamma = gamma
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.target_update_freq = target_update_freq
        self.exploration_fraction = exploration_fraction
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.double_dqn = double_dqn
        self.n_step = n_step
        self.prioritized = prioritized

        obs_dim = int(np.prod(self.env.observation_space.shape))
        act_dim = int(self.env.action_space.n)

        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # Networks
        net_cls = DuelingQNetwork if dueling else SimpleQNetwork
        self.q_network = net_cls(obs_dim, act_dim, hidden)
        self.target_network = copy.deepcopy(self.q_network)

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Buffer (actions stored as 1-dim)
        if prioritized:
            self.buffer = rlox.PrioritizedReplayBuffer(
                buffer_size, obs_dim, 1, alpha=alpha, beta=beta_start
            )
        else:
            self.buffer = rlox.ReplayBuffer(buffer_size, obs_dim, 1)

        # N-step return buffer
        self._n_step_buffer: list[tuple] = []

    def _get_epsilon(self, step: int, total_timesteps: int) -> float:
        fraction = min(1.0, step / max(1, int(total_timesteps * self.exploration_fraction)))
        return self.exploration_initial_eps + fraction * (
            self.exploration_final_eps - self.exploration_initial_eps
        )

    def _store_transition(self, obs, action, reward, terminated, truncated):
        self._n_step_buffer.append((obs, action, reward, terminated, truncated))
        if len(self._n_step_buffer) < self.n_step:
            return

        # Compute n-step return
        R = 0.0
        for i in reversed(range(self.n_step)):
            _, _, r, done, trunc = self._n_step_buffer[i]
            R = r + self.gamma * R * (1.0 - float(done or trunc))

        first_obs, first_action, _, _, _ = self._n_step_buffer[0]
        last_done = self._n_step_buffer[-1][3]
        last_trunc = self._n_step_buffer[-1][4]

        if self.prioritized:
            self.buffer.push(
                np.asarray(first_obs, dtype=np.float32),
                np.array([float(first_action)], dtype=np.float32),
                float(R),
                bool(last_done),
                bool(last_trunc),
                priority=1.0,
            )
        else:
            self.buffer.push(
                np.asarray(first_obs, dtype=np.float32),
                np.array([float(first_action)], dtype=np.float32),
                float(R),
                bool(last_done),
                bool(last_trunc),
            )
        self._n_step_buffer.pop(0)

    def train(self, total_timesteps: int) -> dict[str, float]:
        obs, _ = self.env.reset()
        episode_rewards: list[float] = []
        ep_reward = 0.0
        metrics: dict[str, float] = {}

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

            self._store_transition(obs, action, reward, terminated, truncated)

            obs = next_obs
            if terminated or truncated:
                # Flush n-step buffer
                while self._n_step_buffer:
                    R = 0.0
                    for i in reversed(range(len(self._n_step_buffer))):
                        _, _, r, done, trunc = self._n_step_buffer[i]
                        R = r + self.gamma * R * (1.0 - float(done or trunc))
                    first_obs_b, first_action_b, _, _, _ = self._n_step_buffer[0]
                    if self.prioritized:
                        self.buffer.push(
                            np.asarray(first_obs_b, dtype=np.float32),
                            np.array([float(first_action_b)], dtype=np.float32),
                            float(R),
                            True,
                            False,
                            priority=1.0,
                        )
                    else:
                        self.buffer.push(
                            np.asarray(first_obs_b, dtype=np.float32),
                            np.array([float(first_action_b)], dtype=np.float32),
                            float(R),
                            True,
                            False,
                        )
                    self._n_step_buffer.pop(0)

                episode_rewards.append(ep_reward)
                ep_reward = 0.0
                obs, _ = self.env.reset()

            # Update
            if step >= self.learning_starts and len(self.buffer) >= self.batch_size:
                metrics = self._update(step, total_timesteps)

            # Target network update
            if step % self.target_update_freq == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())

        metrics["mean_reward"] = float(np.mean(episode_rewards)) if episode_rewards else 0.0
        return metrics

    def _update(self, step: int, total_timesteps: int) -> dict[str, float]:
        if self.prioritized:
            batch = self.buffer.sample(self.batch_size, step)
            weights = torch.as_tensor(np.asarray(batch["weights"]), dtype=torch.float32)
            indices = np.asarray(batch["indices"])
        else:
            batch = self.buffer.sample(self.batch_size, step)
            weights = torch.ones(self.batch_size)
            indices = None

        obs = torch.as_tensor(np.asarray(batch["obs"]), dtype=torch.float32)
        actions = torch.as_tensor(np.asarray(batch["actions"]), dtype=torch.long).squeeze(-1)
        rewards = torch.as_tensor(np.asarray(batch["rewards"]), dtype=torch.float32)
        terminated = torch.as_tensor(np.asarray(batch["terminated"]), dtype=torch.float32)

        # Current Q
        q_values = self.q_network(obs)
        q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.double_dqn:
                # Use online network for action selection
                next_actions = self.q_network(obs).argmax(dim=-1)
                next_q = self.target_network(obs).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                next_q = self.target_network(obs).max(dim=-1).values

            target_q = rewards + self.gamma ** self.n_step * (1.0 - terminated) * next_q

        td_error = q - target_q
        loss = (weights * td_error.pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
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
