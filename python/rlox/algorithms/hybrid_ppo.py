"""Hybrid PPO: Candle inference for collection + PyTorch for training.

Collection runs entirely in Rust (Candle NN + VecEnv + GAE) at ~180K SPS.
Training uses PyTorch for backward pass + optimizer step.
Weight sync copies flat f32 buffer from PyTorch to Candle after each update.

Currently supports discrete action spaces with native Rust VecEnv (CartPole).

PPO Reference:
    J. Schulman, F. Wolski, P. Dhariwal, A. Radford, O. Klimov,
    "Proximal Policy Optimization Algorithms,"
    arXiv:1707.06347, 2017.
    https://arxiv.org/abs/1707.06347

Candle:
    https://github.com/huggingface/candle
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn

import rlox
from rlox.callbacks import Callback, CallbackList
from rlox.config import PPOConfig
from rlox.logging import LoggerCallback
from rlox.losses import PPOLoss
from rlox.policies import DiscretePolicy


class HybridPPO:
    """PPO with Rust-native collection via Candle.

    The collection loop (env stepping + policy inference + GAE computation)
    runs in a background Rust thread with zero Python overhead.
    PyTorch handles only the training backward pass.

    Parameters
    ----------
    env_id : str
        Environment ID. Currently only "CartPole-v1" (native Rust env).
    n_envs : int
        Number of parallel environments (default 16).
    seed : int
        Random seed.
    hidden : int
        Hidden layer width for both Candle (collection) and PyTorch (training).
    logger : LoggerCallback, optional
        Logger for metrics.
    callbacks : list[Callback], optional
        Training callbacks.
    **config_kwargs
        Override any PPOConfig fields (n_steps, learning_rate, etc.).
    """

    def __init__(
        self,
        env_id: str = "CartPole-v1",
        n_envs: int = 16,
        seed: int = 42,
        hidden: int = 64,
        logger: LoggerCallback | None = None,
        callbacks: list[Callback] | None = None,
        **config_kwargs: Any,
    ):
        self.env_id = env_id
        self.seed = seed

        # Build config
        cfg_fields = {f.name for f in PPOConfig.__dataclass_fields__.values()}
        cfg_dict = {k: v for k, v in config_kwargs.items() if k in cfg_fields}
        cfg_dict["n_envs"] = n_envs
        self.config = PPOConfig(**cfg_dict)

        # Detect env spaces
        import gymnasium as gym

        tmp = gym.make(env_id)
        obs_dim = int(np.prod(tmp.observation_space.shape))
        n_actions = int(tmp.action_space.n)
        tmp.close()

        self._obs_dim = obs_dim
        self._n_actions = n_actions
        self._hidden = hidden

        # PyTorch policy (for training backward pass)
        self.policy = DiscretePolicy(
            obs_dim=obs_dim, n_actions=n_actions, hidden=hidden
        )
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.config.learning_rate, eps=1e-5
        )
        self.loss_fn = PPOLoss(
            clip_eps=self.config.clip_eps,
            vf_coef=self.config.vf_coef,
            ent_coef=self.config.ent_coef,
            max_grad_norm=self.config.max_grad_norm,
            clip_vloss=self.config.clip_vloss,
        )

        # Candle collector (for data collection — runs in Rust thread)
        self.collector = rlox.CandleCollector(
            env_id=env_id,
            n_envs=n_envs,
            obs_dim=obs_dim,
            n_actions=n_actions,
            n_steps=self.config.n_steps,
            hidden=hidden,
            lr=self.config.learning_rate,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            seed=seed,
        )

        # Initialize PyTorch weights from Candle's random init
        self._sync_candle_to_pytorch()

        self.logger = logger
        self.callbacks = CallbackList(callbacks)
        self._global_step = 0
        self._collection_time = 0.0
        self._training_time = 0.0

    def _sync_candle_to_pytorch(self) -> None:
        """Copy weights from Candle → PyTorch (used at init)."""
        candle_weights = self.collector.get_weights()
        # Map flat buffer to PyTorch parameters
        offset = 0
        for p in self.policy.parameters():
            numel = p.numel()
            p.data.copy_(
                torch.from_numpy(
                    candle_weights[offset : offset + numel].copy()
                ).reshape(p.shape)
            )
            offset += numel

    def _restart_collector(self) -> None:
        """Restart the Candle collector with current PyTorch weights."""
        self.collector = rlox.CandleCollector(
            env_id=self.env_id,
            n_envs=self.config.n_envs,
            obs_dim=self._obs_dim,
            n_actions=self._n_actions,
            n_steps=self.config.n_steps,
            hidden=self._hidden,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            seed=self.seed,
        )
        self._sync_pytorch_to_candle()

    def _sync_pytorch_to_candle(self) -> None:
        """Copy weights from PyTorch → Candle (after each training update)."""
        flat = []
        for p in self.policy.parameters():
            flat.append(p.data.cpu().flatten())
        flat_np = torch.cat(flat).numpy().astype(np.float32)
        self.collector.sync_weights(flat_np)

    def train(self, total_timesteps: int) -> dict[str, float]:
        """Run hybrid PPO training.

        Uses collect-then-train pattern: stop the Rust collection thread
        during PyTorch training to avoid CPU thread contention between
        Candle/Rayon and PyTorch/Accelerate thread pools on macOS.
        """
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

            # === COLLECTION: Pure Rust (Candle + VecEnv + GAE) ===
            t0 = time.perf_counter()
            batch_dict = self.collector.recv()  # Blocks until batch ready
            # Stop collector during training to avoid CPU thread contention
            self.collector.stop()
            t1 = time.perf_counter()
            self._collection_time += t1 - t0

            # Convert to PyTorch tensors
            n = batch_dict["n_steps"] * batch_dict["n_envs"]
            obs = torch.from_numpy(
                batch_dict["observations"].reshape(n, self._obs_dim).copy()
            ).float()
            actions = torch.from_numpy(batch_dict["actions"].copy()).float()
            log_probs = torch.from_numpy(batch_dict["log_probs"].copy()).float()
            values = torch.from_numpy(batch_dict["values"].copy()).float()
            advantages = torch.from_numpy(batch_dict["advantages"].copy()).float()
            returns = torch.from_numpy(batch_dict["returns"].copy()).float()

            mean_ep_reward = float(batch_dict["rewards"].sum()) / cfg.n_envs
            all_rewards.append(mean_ep_reward)

            self.callbacks.on_rollout_end(mean_reward=mean_ep_reward, update=update)

            # === TRAINING: PyTorch backward pass ===
            t2 = time.perf_counter()

            for _epoch in range(cfg.n_epochs):
                # Shuffle indices for minibatch sampling
                indices = torch.randperm(n)
                for start in range(0, n, cfg.batch_size):
                    end = min(start + cfg.batch_size, n)
                    mb_idx = indices[start:end]

                    mb_obs = obs[mb_idx]
                    mb_actions = actions[mb_idx]
                    mb_log_probs = log_probs[mb_idx]
                    mb_values = values[mb_idx]
                    mb_returns = returns[mb_idx]

                    mb_adv = advantages[mb_idx]
                    if cfg.normalize_advantages:
                        mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                    loss, metrics = self.loss_fn(
                        self.policy,
                        mb_obs,
                        mb_actions,
                        mb_log_probs,
                        mb_adv,
                        mb_returns,
                        mb_values,
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

            t3 = time.perf_counter()
            self._training_time += t3 - t2

            # Restart collector with updated weights for next rollout
            self._restart_collector()

            should_continue = self.callbacks.on_step(
                reward=mean_ep_reward, step=self._global_step, algo=self
            )
            if not should_continue:
                break

            if self.logger is not None:
                self.logger.on_train_step(
                    update, {**last_metrics, "mean_reward": mean_ep_reward}
                )

        self.collector.stop()
        self.callbacks.on_training_end()

        last_metrics["mean_reward"] = (
            float(sum(all_rewards) / len(all_rewards)) if all_rewards else 0.0
        )
        last_metrics["collection_time_pct"] = (
            self._collection_time
            / (self._collection_time + self._training_time + 1e-9)
            * 100
        )
        last_metrics["training_time_pct"] = (
            self._training_time
            / (self._collection_time + self._training_time + 1e-9)
            * 100
        )
        return last_metrics

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> int:
        """Get action from PyTorch policy."""
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            if deterministic:
                logits = self.policy.actor(obs_t)
                return int(logits.argmax(dim=-1).item())
            action, _ = self.policy.get_action_and_logprob(obs_t)
            return int(action.item())

    def timing_summary(self) -> dict[str, float]:
        """Return collection vs training time breakdown."""
        total = self._collection_time + self._training_time
        return {
            "collection_s": self._collection_time,
            "training_s": self._training_time,
            "collection_pct": self._collection_time / (total + 1e-9) * 100,
            "training_pct": self._training_time / (total + 1e-9) * 100,
        }
