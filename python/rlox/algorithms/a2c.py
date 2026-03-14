"""Advantage Actor-Critic (A2C): single-update variant without PPO clipping."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from rlox.collectors import RolloutCollector
from rlox.logging import LoggerCallback
from rlox.policies import DiscretePolicy


_CARTPOLE_OBS_DIM = 4
_CARTPOLE_N_ACTIONS = 2


class A2C:
    """Synchronous A2C.

    Like PPO but: no ratio clipping, no epochs (single gradient step per
    rollout), and typically shorter ``n_steps``.
    """

    def __init__(
        self,
        env_id: str,
        n_envs: int = 8,
        seed: int = 42,
        n_steps: int = 5,
        learning_rate: float = 7e-4,
        gamma: float = 0.99,
        gae_lambda: float = 1.0,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        normalize_advantages: bool = True,
        policy: nn.Module | None = None,
        logger: LoggerCallback | None = None,
    ):
        self.env_id = env_id
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.learning_rate = learning_rate
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.normalize_advantages = normalize_advantages
        self.device = "cpu"

        if policy is not None:
            self.policy = policy
        else:
            self.policy = DiscretePolicy(
                obs_dim=_CARTPOLE_OBS_DIM, n_actions=_CARTPOLE_N_ACTIONS
            )

        self.optimizer = torch.optim.RMSprop(
            self.policy.parameters(), lr=learning_rate, eps=1e-5, alpha=0.99
        )

        self.collector = RolloutCollector(
            env_id=env_id,
            n_envs=n_envs,
            seed=seed,
            device=self.device,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

        self.logger = logger

    def train(self, total_timesteps: int) -> dict[str, float]:
        steps_per_rollout = self.n_envs * self.n_steps
        n_updates = max(1, total_timesteps // steps_per_rollout)

        all_rewards: list[float] = []
        last_metrics: dict[str, float] = {}

        for update in range(n_updates):
            batch = self.collector.collect(self.policy, n_steps=self.n_steps)
            all_rewards.append(batch.rewards.sum().item() / self.n_envs)

            # Single gradient step on the full rollout
            obs = batch.obs
            actions = batch.actions
            advantages = batch.advantages
            returns = batch.returns

            if self.normalize_advantages:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            log_probs, entropy = self.policy.get_logprob_and_entropy(obs, actions)
            values = self.policy.get_value(obs)

            policy_loss = -(log_probs * advantages.detach()).mean()
            value_loss = 0.5 * ((values - returns) ** 2).mean()
            entropy_loss = entropy.mean()

            loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            last_metrics = {
                "policy_loss": policy_loss.item(),
                "value_loss": value_loss.item(),
                "entropy": entropy_loss.item(),
            }

            if self.logger is not None:
                self.logger.on_train_step(update, {**last_metrics, "mean_reward": all_rewards[-1]})

        last_metrics["mean_reward"] = float(sum(all_rewards) / len(all_rewards)) if all_rewards else 0.0
        return last_metrics
