"""Advantage Actor-Critic (A2C): single-update variant without PPO clipping."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from rlox.callbacks import Callback, CallbackList
from rlox.collectors import RolloutCollector
from rlox.logging import LoggerCallback
from rlox.policies import DiscretePolicy
from rlox.utils import detect_env_spaces as _detect_env_spaces


class A2C:
    """Synchronous A2C.

    Like PPO but: no ratio clipping, no epochs (single gradient step per
    rollout), and typically shorter ``n_steps``.

    Auto-detects observation and action dimensions from the environment.
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
        normalize_advantages: bool = False,
        policy: nn.Module | None = None,
        logger: LoggerCallback | None = None,
        callbacks: list[Callback] | None = None,
        compile: bool = False,
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

        # Auto-detect env spaces
        if policy is not None:
            self.policy = policy
        else:
            obs_dim, action_space, is_discrete = _detect_env_spaces(env_id)
            if is_discrete:
                self.policy = DiscretePolicy(
                    obs_dim=obs_dim, n_actions=int(action_space.n)
                )
            else:
                from rlox.policies import ContinuousPolicy

                act_dim = int(np.prod(action_space.shape))
                self.policy = ContinuousPolicy(obs_dim=obs_dim, act_dim=act_dim)

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
        self.callbacks = CallbackList(callbacks)
        self._global_step = 0

        if compile:
            from rlox.compile import compile_policy

            compile_policy(self)

    def train(self, total_timesteps: int) -> dict[str, float]:
        steps_per_rollout = self.n_envs * self.n_steps
        n_updates = max(1, total_timesteps // steps_per_rollout)

        all_rewards: list[float] = []
        last_metrics: dict[str, float] = {}

        self.callbacks.on_training_start()

        for update in range(n_updates):
            batch = self.collector.collect(self.policy, n_steps=self.n_steps)
            mean_ep_reward = batch.rewards.sum().item() / self.n_envs
            all_rewards.append(mean_ep_reward)

            # Single gradient step on the full rollout
            obs = batch.obs
            actions = batch.actions
            advantages = batch.advantages
            returns = batch.returns

            if self.normalize_advantages:
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

            log_probs, entropy = self.policy.get_logprob_and_entropy(obs, actions)
            values = self.policy.get_value(obs)

            policy_loss = -(log_probs * advantages.detach()).mean()
            value_loss = 0.5 * ((values - returns) ** 2).mean()
            entropy_loss = entropy.mean()

            loss = (
                policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy_loss
            )

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            last_metrics = {
                "policy_loss": policy_loss.item(),
                "value_loss": value_loss.item(),
                "entropy": entropy_loss.item(),
            }

            self.callbacks.on_train_batch(loss=loss.item(), **last_metrics)

            self._global_step += steps_per_rollout

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

    def predict(
        self, obs: Any, deterministic: bool = True
    ) -> np.ndarray | int:
        """Get action from the trained policy.

        Parameters
        ----------
        obs : array-like
            Observation.
        deterministic : bool
            If True, return the mode of the action distribution.

        Returns
        -------
        Action as an int (discrete) or numpy array (continuous).
        """
        import torch

        obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                if hasattr(self.policy, "logits_net") or (
                    hasattr(self.policy, "actor")
                    and isinstance(self.policy, DiscretePolicy)
                ):
                    logits = self.policy.actor(obs_t)
                    action = logits.argmax(dim=-1)
                else:
                    action = self.policy.actor(obs_t)
            else:
                action, _ = self.policy.get_action_and_logprob(obs_t)
        action = action.squeeze(0)
        if action.dim() == 0:
            return int(action.item())
        return action.numpy()

    def save(self, path: str) -> None:
        """Save training checkpoint."""
        from rlox.checkpoint import Checkpoint

        Checkpoint.save(
            path,
            model=self.policy,
            optimizer=self.optimizer,
            step=self._global_step,
            config={"env_id": self.env_id},
        )

    @classmethod
    def from_checkpoint(cls, path: str, env_id: str | None = None) -> A2C:
        """Restore A2C from a checkpoint."""
        from rlox.checkpoint import Checkpoint

        data = Checkpoint.load(path)
        config = data.get("config", {})
        eid = env_id or config.get("env_id", "CartPole-v1")

        a2c = cls(env_id=eid)
        a2c.policy.load_state_dict(data["model_state_dict"])
        a2c.optimizer.load_state_dict(data["optimizer_state_dict"])
        a2c._global_step = data.get("step", 0)
        return a2c
