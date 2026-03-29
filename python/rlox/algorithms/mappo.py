"""Multi-Agent PPO (MAPPO): centralized critic, decentralized actors.

Centralized training with decentralized execution (CTDE). Each agent has
its own actor policy, but a shared centralized critic takes joint observations
from all agents.

Simplified implementation that works with standard Gymnasium envs
(treats each env as a single-agent case when n_agents=1).

Reference:
    C. Yu, A. Velu, E. Vinitsky, J. Gao, Y. Wang, A. Baez, et al.,
    "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games,"
    NeurIPS, 2022.
    https://arxiv.org/abs/2103.01955
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from rlox.callbacks import Callback, CallbackList
from rlox.collectors import RolloutCollector
from rlox.logging import LoggerCallback
from rlox.policies import DiscretePolicy, ContinuousPolicy


def _detect_env_spaces(env_id: str) -> tuple[int, Any, bool]:
    """Detect obs_dim, action_space, and whether the env is discrete.

    Returns (obs_dim, action_space, is_discrete).
    """
    tmp = gym.make(env_id)
    obs_dim = int(np.prod(tmp.observation_space.shape))
    action_space = tmp.action_space
    is_discrete = isinstance(action_space, gym.spaces.Discrete)
    tmp.close()
    return obs_dim, action_space, is_discrete


class MAPPO:
    """Multi-Agent PPO with centralized critic, decentralized actors.

    Auto-detects observation and action dimensions from the environment.
    For n_agents=1, this reduces to standard PPO on single-agent envs.

    Parameters
    ----------
    env_id : str
        Gymnasium environment ID.
    n_agents : int
        Number of agents (default 1).
    n_envs : int
        Number of parallel environments (default 4).
    seed : int
        Random seed (default 42).
    n_steps : int
        Rollout length per environment per update (default 64).
    n_epochs : int
        Number of SGD passes per rollout (default 4).
    batch_size : int
        Minibatch size for SGD (default 128).
    learning_rate : float
        Adam learning rate (default 2.5e-4).
    gamma : float
        Discount factor (default 0.99).
    gae_lambda : float
        GAE lambda (default 0.95).
    clip_eps : float
        PPO clipping range (default 0.2).
    vf_coef : float
        Value loss coefficient (default 0.5).
    ent_coef : float
        Entropy bonus coefficient (default 0.01).
    max_grad_norm : float
        Maximum gradient norm for clipping (default 0.5).
    logger : LoggerCallback, optional
        Logger for metrics.
    callbacks : list[Callback], optional
        Training callbacks.
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
        logger: LoggerCallback | None = None,
        callbacks: list[Callback] | None = None,
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

        # Auto-detect environment spaces
        obs_dim, action_space, is_discrete = _detect_env_spaces(env_id)
        self._obs_dim = obs_dim
        self._is_discrete = is_discrete

        if is_discrete:
            n_actions = int(action_space.n)
            self.actors = nn.ModuleList([
                DiscretePolicy(obs_dim=obs_dim, n_actions=n_actions)
                for _ in range(n_agents)
            ])
        else:
            act_dim = int(np.prod(action_space.shape))
            self.actors = nn.ModuleList([
                ContinuousPolicy(obs_dim=obs_dim, act_dim=act_dim)
                for _ in range(n_agents)
            ])

        # Centralized critic — for n_agents=1, input is obs_dim.
        # For n_agents>1, input would be obs_dim * n_agents (joint obs),
        # but multi-agent collection is not yet supported by RolloutCollector.
        critic_input_dim = obs_dim * n_agents
        self.critic = nn.Sequential(
            nn.Linear(critic_input_dim, 64),
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

        self.logger = logger
        self.callbacks = CallbackList(callbacks)
        self._global_step = 0

    def train(self, total_timesteps: int) -> dict[str, float]:
        """Run MAPPO training loop."""
        steps_per_rollout = self.n_envs * self.n_steps
        n_updates = max(1, total_timesteps // steps_per_rollout)

        all_rewards: list[float] = []
        last_metrics: dict[str, float] = {}

        if self.n_agents > 1:
            raise NotImplementedError(
                "MAPPO with n_agents > 1 requires a multi-agent collector that "
                "provides joint observations for the centralized critic. "
                "Currently only n_agents=1 is supported (equivalent to PPO)."
            )

        # For single-agent, use agent 0's actor as the policy
        policy = self.actors[0]

        self.callbacks.on_training_start()

        for update in range(n_updates):
            batch = self.collector.collect(policy, n_steps=self.n_steps)
            mean_ep_reward = batch.rewards.sum().item() / self.n_envs
            all_rewards.append(mean_ep_reward)

            self.callbacks.on_rollout_end(
                mean_reward=mean_ep_reward, update=update
            )

            for _epoch in range(self.n_epochs):
                for mb in batch.sample_minibatches(self.batch_size, shuffle=True):
                    adv = mb.advantages
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                    new_log_probs, entropy = policy.get_logprob_and_entropy(
                        mb.obs, mb.actions
                    )
                    # Centralized critic — for n_agents=1, obs IS the full state
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

                    self.optimizer.zero_grad(set_to_none=True)
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

                    self._global_step += 1
                    self.callbacks.on_train_batch(
                        loss=loss.item(), **last_metrics
                    )

            # Step callback (per rollout)
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
