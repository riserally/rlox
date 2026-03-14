"""Proximal Policy Optimization (PPO) with clipped surrogate objective."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from rlox.collectors import RolloutCollector
from rlox.losses import PPOLoss
from rlox.logging import LoggerCallback
from rlox.policies import DiscretePolicy


# CartPole-v1 defaults
_CARTPOLE_OBS_DIM = 4
_CARTPOLE_N_ACTIONS = 2


@dataclass
class PPOConfig:
    n_envs: int = 8
    n_steps: int = 128
    n_epochs: int = 4
    batch_size: int = 256
    learning_rate: float = 2.5e-4
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    normalize_advantages: bool = True
    clip_vloss: bool = True
    anneal_lr: bool = True
    normalize_rewards: bool = False
    normalize_obs: bool = False


class PPO:
    """PPO trainer.

    Accepts either a ``PPOConfig`` or individual keyword overrides.
    If no policy is supplied, creates a ``DiscretePolicy`` for CartPole.
    """

    def __init__(
        self,
        env_id: str,
        n_envs: int = 8,
        seed: int = 42,
        policy: nn.Module | None = None,
        logger: LoggerCallback | None = None,
        **config_kwargs: Any,
    ):
        self.env_id = env_id
        self.seed = seed

        # Build config
        cfg_fields = {f.name for f in PPOConfig.__dataclass_fields__.values()}
        cfg_dict = {k: v for k, v in config_kwargs.items() if k in cfg_fields}
        cfg_dict["n_envs"] = n_envs
        self.config = PPOConfig(**cfg_dict)

        self.device = "cpu"

        # Policy
        if policy is not None:
            self.policy = policy
        else:
            self.policy = DiscretePolicy(
                obs_dim=_CARTPOLE_OBS_DIM, n_actions=_CARTPOLE_N_ACTIONS
            )

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.config.learning_rate, eps=1e-5
        )

        self.collector = RolloutCollector(
            env_id=env_id,
            n_envs=self.config.n_envs,
            seed=seed,
            device=self.device,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            normalize_rewards=self.config.normalize_rewards,
            normalize_obs=self.config.normalize_obs,
        )

        self.loss_fn = PPOLoss(
            clip_eps=self.config.clip_eps,
            vf_coef=self.config.vf_coef,
            ent_coef=self.config.ent_coef,
            max_grad_norm=self.config.max_grad_norm,
            clip_vloss=self.config.clip_vloss,
        )

        self.logger = logger

    def train(self, total_timesteps: int) -> dict[str, float]:
        """Run PPO training and return final metrics."""
        cfg = self.config
        steps_per_rollout = cfg.n_envs * cfg.n_steps
        n_updates = max(1, total_timesteps // steps_per_rollout)

        all_rewards: list[float] = []
        last_metrics: dict[str, float] = {}

        for update in range(n_updates):
            # LR annealing
            if cfg.anneal_lr:
                frac = 1.0 - update / n_updates
                lr = cfg.learning_rate * frac
                for pg in self.optimizer.param_groups:
                    pg["lr"] = lr

            batch = self.collector.collect(self.policy, n_steps=cfg.n_steps)
            all_rewards.append(batch.rewards.sum().item() / cfg.n_envs)

            for _epoch in range(cfg.n_epochs):
                for mb in batch.sample_minibatches(cfg.batch_size, shuffle=True):
                    adv = mb.advantages
                    if cfg.normalize_advantages:
                        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                    loss, metrics = self.loss_fn(
                        self.policy,
                        mb.obs,
                        mb.actions,
                        mb.log_probs,
                        adv,
                        mb.returns,
                        mb.values,
                    )

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.policy.parameters(), cfg.max_grad_norm
                    )
                    self.optimizer.step()
                    last_metrics = metrics

            if self.logger is not None:
                self.logger.on_train_step(update, {**last_metrics, "mean_reward": all_rewards[-1]})

        last_metrics["mean_reward"] = float(sum(all_rewards) / len(all_rewards)) if all_rewards else 0.0
        return last_metrics
