"""Vanilla Policy Gradient (REINFORCE with learned baseline + GAE).

The simplest policy gradient algorithm -- the pedagogical anchor for
understanding PPO, TRPO, and all actor-critic methods.

VPG computes the policy gradient::

    grad J(theta) = E[grad log pi(a|s) * A(s,a)]

where A(s,a) is the advantage estimated via Generalized Advantage
Estimation (GAE).  Unlike PPO there is no clipping and unlike TRPO
there is no KL constraint -- a *single* gradient step is taken on the
surrogate objective per rollout.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from rlox.callbacks import Callback, CallbackList
from rlox.checkpoint import Checkpoint
from rlox.collectors import RolloutCollector
from rlox.config import VPGConfig
from rlox.gym_vec_env import GymVecEnv
from rlox.logging import LoggerCallback
from rlox.policies import DiscretePolicy
from rlox.trainer import register_algorithm
from rlox.utils import detect_env_spaces as _detect_env_spaces
from rlox.vec_normalize import VecNormalize


_RUST_NATIVE_ENVS = {"CartPole-v1", "CartPole"}


@register_algorithm("vpg")
class VPG:
    """Vanilla Policy Gradient (REINFORCE with baseline).

    The simplest policy gradient algorithm -- the pedagogical anchor for
    understanding PPO, TRPO, and all actor-critic methods.

    VPG computes the policy gradient::

        grad J(theta) = E[grad log pi(a|s) * A(s,a)]

    where A(s,a) is the advantage estimated via GAE.

    Key differences from PPO:
    - Single gradient step per rollout (no epochs on same data)
    - No ratio clipping
    - Separate optimizers for policy and value function

    Parameters
    ----------
    env_id : str
        Gymnasium environment ID.
    seed : int
        Random seed (default 42).
    policy : nn.Module, optional
        Custom policy network. If None, auto-selects based on env.
    logger : LoggerCallback, optional
        Logger for metrics.
    callbacks : list[Callback], optional
        Training callbacks.
    **config_kwargs
        Override any VPGConfig fields.
    """

    def __init__(
        self,
        env_id: str,
        seed: int = 42,
        policy: nn.Module | None = None,
        logger: LoggerCallback | None = None,
        callbacks: list[Callback] | None = None,
        **config_kwargs: Any,
    ):
        self.env_id = env_id
        self.seed = seed

        # Build config
        cfg_fields = {f.name for f in VPGConfig.__dataclass_fields__.values()}
        cfg_dict = {k: v for k, v in config_kwargs.items() if k in cfg_fields}
        self.config = VPGConfig(**cfg_dict)

        self.device = "cpu"

        # Detect environment spaces
        obs_dim, action_space, is_discrete = _detect_env_spaces(env_id)
        self._obs_dim = obs_dim
        self._is_discrete = is_discrete

        # Policy -- auto-select if not provided
        if policy is not None:
            self.policy = policy
        elif is_discrete:
            n_actions = int(action_space.n)
            self.policy = DiscretePolicy(
                obs_dim=obs_dim,
                n_actions=n_actions,
                hidden=self.config.hidden,
            )
        else:
            from rlox.policies import ContinuousPolicy

            act_dim = int(np.prod(action_space.shape))
            self.policy = ContinuousPolicy(
                obs_dim=obs_dim,
                act_dim=act_dim,
                hidden=self.config.hidden,
            )

        # Separate optimizers: policy gets a single step, value fn gets many
        self.policy_optimizer = torch.optim.Adam(
            self._get_policy_params(),
            lr=self.config.learning_rate,
        )
        self.vf_optimizer = torch.optim.Adam(
            self._get_vf_params(),
            lr=self.config.vf_lr,
        )

        # Build environment
        import rlox as _rlox

        if env_id in _RUST_NATIVE_ENVS:
            raw_env = _rlox.VecEnv(
                n=self.config.n_envs, seed=seed, env_id=env_id
            )
        else:
            raw_env = GymVecEnv(
                env_id, n_envs=self.config.n_envs, seed=seed
            )

        self.collector = RolloutCollector(
            env_id=env_id,
            n_envs=self.config.n_envs,
            seed=seed,
            device=self.device,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            env=raw_env,
        )

        self.logger = logger
        self.callbacks = CallbackList(callbacks)
        self._global_step = 0

    # ------------------------------------------------------------------
    # Parameter groups
    # ------------------------------------------------------------------

    def _get_policy_params(self) -> list[nn.Parameter]:
        """Return only the actor (policy head) parameters."""
        if hasattr(self.policy, "actor"):
            return list(self.policy.actor.parameters())
        if hasattr(self.policy, "logits_net"):
            return list(self.policy.logits_net.parameters())
        return list(self.policy.parameters())

    def _get_vf_params(self) -> list[nn.Parameter]:
        """Return only the critic (value head) parameters."""
        if hasattr(self.policy, "critic"):
            return list(self.policy.critic.parameters())
        if hasattr(self.policy, "value_net"):
            return list(self.policy.value_net.parameters())
        return list(self.policy.parameters())

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, total_timesteps: int) -> dict[str, float]:
        """Run VPG training and return final metrics."""
        cfg = self.config
        steps_per_rollout = cfg.n_envs * cfg.n_steps
        n_updates = max(1, total_timesteps // steps_per_rollout)

        all_rewards: list[float] = []
        last_metrics: dict[str, float] = {}

        self.callbacks.on_training_start()

        for update in range(n_updates):
            batch = self.collector.collect(self.policy, n_steps=cfg.n_steps)
            mean_ep_reward = batch.rewards.sum().item() / cfg.n_envs
            all_rewards.append(mean_ep_reward)

            self.callbacks.on_rollout_end(mean_reward=mean_ep_reward, update=update)

            # ----- Policy update: SINGLE gradient step -----
            obs = batch.obs
            actions = batch.actions
            advantages = batch.advantages

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-8
            )

            log_probs, entropy = self.policy.get_logprob_and_entropy(
                obs, actions
            )

            # L_policy = -E[log pi(a|s) * A(s,a)]
            policy_loss = -(log_probs * advantages.detach()).mean()

            # Optional entropy bonus (default ent_coef=0.0)
            if cfg.ent_coef > 0.0:
                policy_loss = policy_loss - cfg.ent_coef * entropy.mean()

            self.policy_optimizer.zero_grad(set_to_none=True)
            policy_loss.backward()
            nn.utils.clip_grad_norm_(
                self._get_policy_params(), cfg.max_grad_norm
            )
            self.policy_optimizer.step()

            # ----- Value function update: multiple epochs -----
            vf_loss_val = 0.0
            for _epoch in range(cfg.vf_epochs):
                values = self.policy.get_value(obs)
                vf_loss = ((values - batch.returns) ** 2).mean()
                self.vf_optimizer.zero_grad(set_to_none=True)
                vf_loss.backward()
                nn.utils.clip_grad_norm_(
                    self._get_vf_params(), cfg.max_grad_norm
                )
                self.vf_optimizer.step()
                vf_loss_val = vf_loss.item()

            last_metrics = {
                "policy_loss": policy_loss.item(),
                "vf_loss": vf_loss_val,
                "entropy": entropy.mean().item(),
                "mean_reward": mean_ep_reward,
            }

            self._global_step += steps_per_rollout

            self.callbacks.on_train_batch(
                loss=policy_loss.item(), **last_metrics
            )

            should_continue = self.callbacks.on_step(
                reward=mean_ep_reward, step=self._global_step, algo=self
            )
            if not should_continue:
                break

            if self.logger is not None:
                self.logger.on_train_step(update, last_metrics)

        self.callbacks.on_training_end()

        last_metrics["mean_reward"] = (
            float(sum(all_rewards) / len(all_rewards)) if all_rewards else 0.0
        )
        return last_metrics

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save training checkpoint."""
        Checkpoint.save(
            path,
            model=self.policy,
            optimizer=self.vf_optimizer,
            step=self._global_step,
            config=self.config.to_dict(),
        )

    @classmethod
    def from_checkpoint(cls, path: str, env_id: str | None = None) -> VPG:
        """Restore VPG from a checkpoint."""
        data = Checkpoint.load(path)
        config = data["config"]
        eid = env_id or config.get("env_id", "CartPole-v1")
        vpg = cls(env_id=eid, **config)
        vpg.policy.load_state_dict(data["model_state_dict"])
        vpg.vf_optimizer.load_state_dict(data["optimizer_state_dict"])
        vpg._global_step = data.get("step", 0)
        return vpg
