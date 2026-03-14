"""Multi-Agent PPO (MAPPO): centralized critic, decentralized actors.

Simplified implementation that works with standard Gymnasium envs
(treats each env as a single-agent case when n_agents=1).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

import rlox
from rlox.collectors import RolloutCollector
from rlox.policies import DiscretePolicy


class MAPPO:
    """Multi-Agent PPO with centralized critic, decentralized actors.

    For n_agents=1, this reduces to standard PPO on single-agent envs.
    """

    def __init__(
        self,
        env_id: str,
        n_agents: int = 1,
        n_envs: int = 4,
        seed: int = 42,
        n_steps: int = 64,
        n_epochs: int = 4,
        batch_size: int = 128,
        learning_rate: float = 2.5e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        obs_dim: int = 4,
        n_actions: int = 2,
    ):
        self.env_id = env_id
        self.n_agents = n_agents
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = "cpu"

        # Each agent gets its own actor; shared critic
        self.actors = nn.ModuleList([
            DiscretePolicy(obs_dim=obs_dim, n_actions=n_actions)
            for _ in range(n_agents)
        ])
        # Centralized critic takes concatenated observations
        self.critic = nn.Sequential(
            nn.Linear(obs_dim * n_agents, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        all_params = list(self.actors.parameters()) + list(self.critic.parameters())
        self.optimizer = torch.optim.Adam(all_params, lr=learning_rate, eps=1e-5)

        self.collector = RolloutCollector(
            env_id=env_id,
            n_envs=n_envs,
            seed=seed,
            device=self.device,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

    def train(self, total_timesteps: int) -> dict[str, float]:
        """Run MAPPO training loop."""
        steps_per_rollout = self.n_envs * self.n_steps
        n_updates = max(1, total_timesteps // steps_per_rollout)

        all_rewards: list[float] = []
        last_metrics: dict[str, float] = {}

        # For single-agent, use agent 0's actor as the policy
        policy = self.actors[0]

        for update in range(n_updates):
            batch = self.collector.collect(policy, n_steps=self.n_steps)
            all_rewards.append(batch.rewards.sum().item() / self.n_envs)

            for _epoch in range(self.n_epochs):
                for mb in batch.sample_minibatches(self.batch_size, shuffle=True):
                    adv = mb.advantages
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                    new_log_probs, entropy = policy.get_logprob_and_entropy(
                        mb.obs, mb.actions
                    )
                    # Centralized critic
                    values = self.critic(mb.obs).squeeze(-1)

                    log_ratio = new_log_probs - mb.log_probs
                    ratio = log_ratio.exp()

                    pg_loss1 = -adv * ratio
                    pg_loss2 = -adv * torch.clamp(
                        ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps
                    )
                    policy_loss = torch.max(pg_loss1, pg_loss2).mean()
                    value_loss = 0.5 * ((values - mb.returns) ** 2).mean()
                    entropy_loss = entropy.mean()

                    loss = (
                        policy_loss
                        + self.vf_coef * value_loss
                        - self.ent_coef * entropy_loss
                    )

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        list(self.actors.parameters()) + list(self.critic.parameters()),
                        self.max_grad_norm,
                    )
                    self.optimizer.step()

                    last_metrics = {
                        "policy_loss": policy_loss.item(),
                        "value_loss": value_loss.item(),
                        "entropy": entropy_loss.item(),
                    }

        last_metrics["mean_reward"] = (
            float(sum(all_rewards) / len(all_rewards)) if all_rewards else 0.0
        )
        return last_metrics
