"""Twin Delayed DDPG (TD3)."""

from __future__ import annotations

import copy
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

import rlox
from rlox.networks import QNetwork, DeterministicPolicy, polyak_update


class TD3:
    """Twin Delayed DDPG.

    Deterministic policy, target policy smoothing, delayed policy updates.
    Uses rlox.ReplayBuffer for storage.
    """

    def __init__(
        self,
        env_id: str,
        buffer_size: int = 1_000_000,
        learning_rate: float = 3e-4,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        learning_starts: int = 1000,
        hidden: int = 256,
        seed: int = 42,
        policy_delay: int = 2,
        target_noise: float = 0.2,
        noise_clip: float = 0.5,
        exploration_noise: float = 0.1,
    ):
        self.env = gym.make(env_id)
        self.env_id = env_id
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.policy_delay = policy_delay
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self.exploration_noise = exploration_noise

        obs_dim = int(np.prod(self.env.observation_space.shape))
        act_dim = int(np.prod(self.env.action_space.shape))
        act_high = float(self.env.action_space.high[0])

        self.act_dim = act_dim
        self.act_high = act_high

        # Networks
        self.actor = DeterministicPolicy(obs_dim, act_dim, hidden, act_high)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic1 = QNetwork(obs_dim, act_dim, hidden)
        self.critic2 = QNetwork(obs_dim, act_dim, hidden)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=learning_rate)

        # Replay buffer
        self.buffer = rlox.ReplayBuffer(buffer_size, obs_dim, act_dim)

    def train(self, total_timesteps: int) -> dict[str, float]:
        obs, _ = self.env.reset()
        episode_rewards: list[float] = []
        ep_reward = 0.0
        metrics: dict[str, float] = {}
        update_count = 0

        for step in range(total_timesteps):
            if step < self.learning_starts:
                action = self.env.action_space.sample()
            else:
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                    action = self.actor(obs_t).squeeze(0).numpy()
                    noise = np.random.randn(self.act_dim) * self.exploration_noise
                    action = np.clip(action + noise, -self.act_high, self.act_high)

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            ep_reward += float(reward)

            self.buffer.push(
                np.asarray(obs, dtype=np.float32),
                np.asarray(action, dtype=np.float32),
                float(reward),
                bool(terminated),
                bool(truncated),
            )

            obs = next_obs
            if terminated or truncated:
                episode_rewards.append(ep_reward)
                ep_reward = 0.0
                obs, _ = self.env.reset()

            # Update
            if step >= self.learning_starts and len(self.buffer) >= self.batch_size:
                update_count += 1
                metrics = self._update(step, update_count)

        metrics["mean_reward"] = float(np.mean(episode_rewards)) if episode_rewards else 0.0
        return metrics

    def _update(self, step: int, update_count: int) -> dict[str, float]:
        batch = self.buffer.sample(self.batch_size, step)
        obs = torch.as_tensor(np.asarray(batch["obs"]), dtype=torch.float32)
        actions = torch.as_tensor(np.asarray(batch["actions"]), dtype=torch.float32)
        if actions.dim() == 1:
            actions = actions.unsqueeze(-1)
        rewards = torch.as_tensor(np.asarray(batch["rewards"]), dtype=torch.float32)
        terminated = torch.as_tensor(np.asarray(batch["terminated"]), dtype=torch.float32)

        with torch.no_grad():
            # Target policy smoothing
            noise = torch.randn_like(actions) * self.target_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(obs) + noise).clamp(-self.act_high, self.act_high)

            q1_next = self.critic1_target(obs, next_actions).squeeze(-1)
            q2_next = self.critic2_target(obs, next_actions).squeeze(-1)
            target_q = rewards + self.gamma * (1.0 - terminated) * torch.min(q1_next, q2_next)

        # Critic updates
        q1 = self.critic1(obs, actions).squeeze(-1)
        q2 = self.critic2(obs, actions).squeeze(-1)
        critic1_loss = F.mse_loss(q1, target_q)
        critic2_loss = F.mse_loss(q2, target_q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        actor_loss_val = 0.0

        # Delayed policy update
        if update_count % self.policy_delay == 0:
            actor_loss = -self.critic1(obs, self.actor(obs)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            actor_loss_val = actor_loss.item()

            polyak_update(self.actor, self.actor_target, self.tau)
            polyak_update(self.critic1, self.critic1_target, self.tau)
            polyak_update(self.critic2, self.critic2_target, self.tau)

        return {
            "critic_loss": (critic1_loss.item() + critic2_loss.item()) / 2,
            "actor_loss": actor_loss_val,
        }
