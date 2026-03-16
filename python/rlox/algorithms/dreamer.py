"""DreamerV3 (simplified): world model with actor-critic in latent space.

This is a structural scaffold, not a SOTA implementation. The world model
consists of a simple encoder + GRU dynamics + decoder. The actor-critic
operates in the latent space using imagination rollouts.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import rlox


class WorldModel(nn.Module):
    """Simple world model: encoder + GRU dynamics + decoder."""

    def __init__(self, obs_dim: int, act_dim: int, latent_dim: int = 64):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Linear(obs_dim, latent_dim)
        self.dynamics = nn.GRUCell(act_dim + latent_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, obs_dim)
        self.reward_head = nn.Linear(latent_dim, 1)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return F.elu(self.encoder(obs))

    def step(
        self, h: torch.Tensor, action_onehot: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One-step transition in latent space.

        Returns (next_h, predicted_obs, predicted_reward).
        """
        inp = torch.cat([h, action_onehot], dim=-1)
        next_h = self.dynamics(inp, h)
        pred_obs = self.decoder(next_h)
        pred_reward = self.reward_head(next_h).squeeze(-1)
        return next_h, pred_obs, pred_reward


class LatentActorCritic(nn.Module):
    """Actor-critic operating in latent space."""

    def __init__(self, latent_dim: int, n_actions: int):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ELU(),
            nn.Linear(64, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ELU(),
            nn.Linear(64, 1),
        )

    def policy(self, h: torch.Tensor) -> torch.distributions.Categorical:
        logits = self.actor(h)
        return torch.distributions.Categorical(logits=logits)

    def value(self, h: torch.Tensor) -> torch.Tensor:
        return self.critic(h).squeeze(-1)


class DreamerV3:
    """Simplified DreamerV3 agent.

    Components:
    - World model (encoder + GRU dynamics + decoder)
    - Actor-critic in latent space
    - Imagination rollouts for policy learning
    - ReplayBuffer for experience storage
    """

    def __init__(
        self,
        env_id: str,
        n_envs: int = 1,
        seed: int = 42,
        latent_dim: int = 64,
        imagination_horizon: int = 5,
        buffer_size: int = 10000,
        batch_size: int = 32,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        obs_dim: int = 4,
        n_actions: int = 2,
    ):
        self.env_id = env_id
        self.n_envs = n_envs
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.latent_dim = latent_dim
        self.imagination_horizon = imagination_horizon
        self.batch_size = batch_size
        self.gamma = gamma
        self.device = "cpu"

        # Environment
        self.env = rlox.VecEnv(n=n_envs, seed=seed, env_id=env_id)

        # Models
        self.world_model = WorldModel(obs_dim, n_actions, latent_dim)
        self.actor_critic = LatentActorCritic(latent_dim, n_actions)

        # Replay buffer
        self.buffer = rlox.ReplayBuffer(buffer_size, obs_dim, 1)

        # Optimizers
        self.wm_optimizer = torch.optim.Adam(
            self.world_model.parameters(), lr=learning_rate
        )
        self.ac_optimizer = torch.optim.Adam(
            self.actor_critic.parameters(), lr=learning_rate
        )

    def _collect_experience(self, n_steps: int) -> float:
        """Collect experience with the current policy and add to buffer."""
        obs = self.env.reset_all()
        total_reward = 0.0

        for _ in range(n_steps):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
            h = self.world_model.encode(obs_tensor)
            dist = self.actor_critic.policy(h)
            actions = dist.sample()

            actions_list = actions.cpu().numpy().astype(np.uint32).tolist()
            step_result = self.env.step_all(actions_list)

            next_obs = step_result["obs"]
            for i in range(self.n_envs):
                self.buffer.push(
                    obs[i].astype(np.float32),
                    np.array([float(actions[i])], dtype=np.float32),
                    float(step_result["rewards"][i]),
                    bool(step_result["terminated"][i]),
                    bool(step_result["truncated"][i]),
                    next_obs[i].astype(np.float32),
                )

            total_reward += step_result["rewards"].sum()
            obs = step_result["obs"].copy()

        return total_reward

    def _train_world_model(self) -> float:
        """Train world model on replay buffer data."""
        if len(self.buffer) < self.batch_size:
            return 0.0

        batch = self.buffer.sample(self.batch_size, seed=np.random.randint(0, 2**31))
        obs = torch.as_tensor(np.array(batch["obs"]), dtype=torch.float32)

        # Simple reconstruction loss
        h = self.world_model.encode(obs)
        pred_obs = self.world_model.decoder(h)
        recon_loss = F.mse_loss(pred_obs, obs)

        # Reward prediction
        rewards = torch.as_tensor(np.array(batch["rewards"]), dtype=torch.float32)
        pred_rewards = self.world_model.reward_head(h).squeeze(-1)
        reward_loss = F.mse_loss(pred_rewards, rewards)

        loss = recon_loss + reward_loss
        self.wm_optimizer.zero_grad()
        loss.backward()
        self.wm_optimizer.step()

        return loss.item()

    def _train_actor_critic(self) -> dict[str, float]:
        """Train actor-critic via imagination rollouts."""
        if len(self.buffer) < self.batch_size:
            return {}

        batch = self.buffer.sample(self.batch_size, seed=np.random.randint(0, 2**31))
        obs = torch.as_tensor(np.array(batch["obs"]), dtype=torch.float32)

        h = self.world_model.encode(obs).detach()

        # Imagination rollout
        imagined_h = [h]
        imagined_rewards = []

        for _ in range(self.imagination_horizon):
            dist = self.actor_critic.policy(h)
            actions = dist.sample()
            action_onehot = F.one_hot(actions, self.n_actions).float()

            with torch.no_grad():
                h, _, pred_r = self.world_model.step(h, action_onehot)
            imagined_h.append(h)
            imagined_rewards.append(pred_r)

        # Compute returns via discounting
        values = [self.actor_critic.value(ih) for ih in imagined_h]
        returns = values[-1].detach()
        all_returns = []
        for t in reversed(range(self.imagination_horizon)):
            returns = imagined_rewards[t] + self.gamma * returns
            all_returns.insert(0, returns)

        # Actor loss (REINFORCE-style)
        actor_loss = 0.0
        for t in range(self.imagination_horizon):
            h_t = imagined_h[t]
            dist = self.actor_critic.policy(h_t)
            advantage = (all_returns[t] - values[t]).detach()
            action = dist.sample()
            log_prob = dist.log_prob(action)
            actor_loss = actor_loss - (log_prob * advantage).mean()
        actor_loss = actor_loss / self.imagination_horizon

        # Critic loss
        critic_loss = 0.0
        for t in range(self.imagination_horizon):
            critic_loss = critic_loss + F.mse_loss(values[t], all_returns[t].detach())
        critic_loss = critic_loss / self.imagination_horizon

        loss = actor_loss + 0.5 * critic_loss
        self.ac_optimizer.zero_grad()
        loss.backward()
        self.ac_optimizer.step()

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
        }

    def train(self, total_timesteps: int) -> dict[str, float]:
        """Run DreamerV3 training loop."""
        collect_steps = 32
        n_updates = max(1, total_timesteps // (collect_steps * self.n_envs))

        all_rewards: list[float] = []
        last_metrics: dict[str, float] = {}

        for _ in range(n_updates):
            reward = self._collect_experience(collect_steps)
            all_rewards.append(reward / self.n_envs)

            wm_loss = self._train_world_model()
            ac_metrics = self._train_actor_critic()

            last_metrics = {
                "wm_loss": wm_loss,
                **ac_metrics,
            }

        last_metrics["mean_reward"] = (
            float(sum(all_rewards) / len(all_rewards)) if all_rewards else 0.0
        )
        return last_metrics
