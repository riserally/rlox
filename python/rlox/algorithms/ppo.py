"""Proximal Policy Optimization (PPO) with clipped surrogate objective."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from rlox.callbacks import Callback, CallbackList
from rlox.checkpoint import Checkpoint
from rlox.collectors import RolloutCollector
from rlox.config import PPOConfig
from rlox.gym_vec_env import GymVecEnv
from rlox.logging import LoggerCallback
from rlox.losses import PPOLoss
from rlox.policies import DiscretePolicy
from rlox.utils import detect_env_spaces as _detect_env_spaces
from rlox.vec_normalize import VecNormalize


_RUST_NATIVE_ENVS = {"CartPole-v1", "CartPole"}


class PPO:
    """PPO trainer.

    Auto-detects observation and action dimensions from the environment.
    Selects DiscretePolicy or ContinuousPolicy based on action space type.
    Supports callbacks, logging, and checkpoint save/load.

    Parameters
    ----------
    env_id : str
        Gymnasium environment ID.
    n_envs : int
        Number of parallel environments (default 8).
    seed : int
        Random seed.
    policy : nn.Module, optional
        Custom policy network. If None, auto-selects based on env.
    logger : LoggerCallback, optional
        Logger for metrics.
    callbacks : list[Callback], optional
        Training callbacks.
    **config_kwargs
        Override any PPOConfig fields.
    """

    def __init__(
        self,
        env_id: str,
        n_envs: int = 8,
        seed: int = 42,
        policy: nn.Module | None = None,
        logger: LoggerCallback | None = None,
        callbacks: list[Callback] | None = None,
        compile: bool = False,
        **config_kwargs: Any,
    ):
        self.env_id = env_id
        self.seed = seed

        # Build config from validated PPOConfig (uses from_dict for alias + warning support)
        cfg_dict = dict(config_kwargs)
        cfg_dict["n_envs"] = n_envs
        self.config = PPOConfig.from_dict(cfg_dict)

        self.device = "cpu"

        # Detect environment spaces
        obs_dim, action_space, is_discrete = _detect_env_spaces(env_id)
        self._obs_dim = obs_dim
        self._is_discrete = is_discrete

        # Policy — auto-select if not provided
        if policy is not None:
            self.policy = policy
        elif is_discrete:
            n_actions = int(action_space.n)
            self.policy = DiscretePolicy(obs_dim=obs_dim, n_actions=n_actions)
        else:
            from rlox.policies import ContinuousPolicy

            act_dim = int(np.prod(action_space.shape))
            self.policy = ContinuousPolicy(obs_dim=obs_dim, act_dim=act_dim)

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.config.learning_rate, eps=1e-5
        )

        # Build environment, optionally wrapping with VecNormalize
        import rlox as _rlox

        if env_id in _RUST_NATIVE_ENVS:
            raw_env = _rlox.VecEnv(n=self.config.n_envs, seed=seed, env_id=env_id)
        else:
            raw_env = GymVecEnv(env_id, n_envs=self.config.n_envs, seed=seed)

        if self.config.normalize_obs or self.config.normalize_rewards:
            self.vec_normalize: VecNormalize | None = VecNormalize(
                raw_env,
                norm_obs=self.config.normalize_obs,
                norm_reward=self.config.normalize_rewards,
                gamma=self.config.gamma,
                obs_dim=obs_dim,
            )
            collector_env = self.vec_normalize
        else:
            self.vec_normalize = None
            collector_env = raw_env

        self.collector = RolloutCollector(
            env_id=env_id,
            n_envs=self.config.n_envs,
            seed=seed,
            device=self.device,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            env=collector_env,
        )

        self.loss_fn = PPOLoss(
            clip_eps=self.config.clip_eps,
            vf_coef=self.config.vf_coef,
            ent_coef=self.config.ent_coef,
            max_grad_norm=self.config.max_grad_norm,
            clip_vloss=self.config.clip_vloss,
        )

        self.logger = logger
        self.callbacks = CallbackList(callbacks)
        self._global_step = 0

        if compile:
            from rlox.compile import compile_policy

            compile_policy(self)

    def train(self, total_timesteps: int) -> dict[str, float]:
        """Run PPO training and return final metrics."""
        cfg = self.config
        steps_per_rollout = cfg.n_envs * cfg.n_steps
        n_updates = max(1, total_timesteps // steps_per_rollout)

        all_rewards: list[float] = []
        last_metrics: dict[str, float] = {}

        self.callbacks.on_training_start()

        for update in range(n_updates):
            # LR annealing
            if cfg.anneal_lr:
                frac = 1.0 - update / n_updates
                lr = cfg.learning_rate * frac
                for pg in self.optimizer.param_groups:
                    pg["lr"] = lr

            batch = self.collector.collect(self.policy, n_steps=cfg.n_steps)
            mean_ep_reward = batch.rewards.sum().item() / cfg.n_envs
            all_rewards.append(mean_ep_reward)

            self.callbacks.on_rollout_end(mean_reward=mean_ep_reward, update=update)

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

                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.policy.parameters(), cfg.max_grad_norm
                    )
                    self.optimizer.step()
                    last_metrics = metrics

                    self._global_step += 1
                    self.callbacks.on_train_batch(loss=loss.item(), **last_metrics)

            # Step callback (per rollout, not per env step)
            should_continue = self.callbacks.on_step(
                reward=mean_ep_reward, step=self._global_step, algo=self
            )
            if not should_continue:
                break

            if self.logger is not None:
                self.logger.on_train_step(
                    update, {**last_metrics, "mean_reward": mean_ep_reward}
                )

        self.callbacks.on_training_end()

        last_metrics["mean_reward"] = (
            float(sum(all_rewards) / len(all_rewards)) if all_rewards else 0.0
        )
        return last_metrics

    def save(self, path: str) -> None:
        """Save training checkpoint."""
        Checkpoint.save(
            path,
            model=self.policy,
            optimizer=self.optimizer,
            step=self._global_step,
            config=self.config.to_dict(),
        )

    @classmethod
    def from_checkpoint(cls, path: str, env_id: str | None = None) -> PPO:
        """Restore PPO from a checkpoint.

        Parameters
        ----------
        path : str
            Path to the checkpoint file.
        env_id : str, optional
            Environment ID. If None, uses the one stored in the checkpoint config.
        """
        data = Checkpoint.load(path)
        config = data["config"]
        eid = env_id or config.get("env_id", "CartPole-v1")

        ppo = cls(env_id=eid, **config)
        ppo.policy.load_state_dict(data["model_state_dict"])
        ppo.optimizer.load_state_dict(data["optimizer_state_dict"])
        ppo._global_step = data.get("step", 0)

        if "torch_rng_state" in data:
            torch.random.set_rng_state(data["torch_rng_state"])

        return ppo
