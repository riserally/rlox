"""Multi-Agent PPO (MAPPO): centralized critic, decentralized actors.

Centralized training with decentralized execution (CTDE). Each agent has
its own actor policy, but a shared centralized critic takes joint observations
from all agents.

For n_agents=1, uses standard RolloutCollector (single-agent PPO equivalent).
For n_agents>1, uses MultiAgentCollector wrapping PettingZoo parallel envs.

Reference:
    C. Yu, A. Velu, E. Vinitsky, J. Gao, Y. Wang, A. Baez, et al.,
    "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games,"
    NeurIPS, 2022.
    https://arxiv.org/abs/2103.01955
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from rlox.callbacks import Callback, CallbackList
from rlox.collectors import RolloutCollector
from rlox.logging import LoggerCallback
from rlox.policies import DiscretePolicy, ContinuousPolicy
from rlox.utils import detect_env_spaces as _detect_env_spaces


@dataclass
class MultiAgentRolloutBatch:
    """Per-agent rollout data with joint observations for centralized critic.

    Attributes
    ----------
    agent_obs : dict[str, torch.Tensor]
        Per-agent observations, shape (N, obs_dim) each.
    joint_obs : torch.Tensor
        Concatenated observations from all agents, shape (N, obs_dim * n_agents).
    agent_actions : dict[str, torch.Tensor]
        Per-agent actions.
    agent_log_probs : dict[str, torch.Tensor]
        Per-agent log-probabilities.
    rewards : dict[str, torch.Tensor]
        Per-agent rewards.
    dones : torch.Tensor
        Shared done flags.
    values : torch.Tensor
        Value estimates from centralized critic.
    advantages : torch.Tensor
        GAE advantages computed from centralized values.
    returns : torch.Tensor
        Discounted returns.
    """

    agent_obs: dict[str, torch.Tensor]
    joint_obs: torch.Tensor
    agent_actions: dict[str, torch.Tensor]
    agent_log_probs: dict[str, torch.Tensor]
    rewards: dict[str, torch.Tensor]
    dones: torch.Tensor
    values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class MultiAgentCollector:
    """Collects rollouts from multi-agent environments.

    Provides per-agent observations and joint observations for
    centralized critic in CTDE (Centralized Training, Decentralized Execution).

    Requires PettingZoo parallel envs. If PettingZoo is not installed,
    raises ImportError with a helpful message.

    Parameters
    ----------
    env_fn : Callable
        Factory function that returns a PettingZoo parallel environment.
    n_envs : int
        Number of parallel environment instances (default 1).
    n_agents : int
        Number of agents in the environment (default 2).
    """

    def __init__(
        self,
        env_fn: Callable[[], Any],
        n_envs: int = 1,
        n_agents: int = 2,
    ):
        try:
            import pettingzoo  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "MultiAgentCollector requires PettingZoo for multi-agent environments. "
                "Install it with: pip install pettingzoo"
            ) from exc

        self.env_fn = env_fn
        self.n_envs = n_envs
        self.n_agents = n_agents
        self.envs = [env_fn() for _ in range(n_envs)]

    def collect(
        self,
        policies: dict[str, nn.Module],
        critic: nn.Module,
        n_steps: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: str = "cpu",
    ) -> MultiAgentRolloutBatch:
        """Collect rollouts from multi-agent environments.

        Parameters
        ----------
        policies : dict[str, nn.Module]
            Mapping from agent name to policy network.
        critic : nn.Module
            Centralized critic that takes joint observations.
        n_steps : int
            Number of steps to collect.
        gamma : float
            Discount factor.
        gae_lambda : float
            GAE lambda.
        device : str
            PyTorch device.

        Returns
        -------
        MultiAgentRolloutBatch
            Per-agent observations and joint observations for CTDE training.
        """
        all_agent_obs: dict[str, list[torch.Tensor]] = {
            name: [] for name in policies
        }
        all_joint_obs: list[torch.Tensor] = []
        all_agent_actions: dict[str, list[torch.Tensor]] = {
            name: [] for name in policies
        }
        all_agent_log_probs: dict[str, list[torch.Tensor]] = {
            name: [] for name in policies
        }
        all_rewards: dict[str, list[torch.Tensor]] = {
            name: [] for name in policies
        }
        all_dones: list[torch.Tensor] = []
        all_values: list[torch.Tensor] = []

        # Reset all envs and get initial observations
        env_observations = []
        for env in self.envs:
            obs, _ = env.reset()
            env_observations.append(obs)

        with torch.no_grad():
            for _ in range(n_steps):
                step_joint_obs = []
                step_actions: dict[str, list] = {name: [] for name in policies}
                step_log_probs: dict[str, list] = {name: [] for name in policies}
                step_rewards: dict[str, list] = {name: [] for name in policies}

                for env_idx, env in enumerate(self.envs):
                    obs_dict = env_observations[env_idx]
                    agent_names = list(policies.keys())

                    # Build joint observation
                    agent_obs_list = []
                    for name in agent_names:
                        obs_t = torch.as_tensor(
                            np.asarray(obs_dict[name], dtype=np.float32),
                            device=device,
                        )
                        all_agent_obs[name].append(obs_t)
                        agent_obs_list.append(obs_t)

                    joint = torch.cat(agent_obs_list, dim=-1)
                    step_joint_obs.append(joint)

                    # Get actions from each agent's policy
                    actions_for_env = {}
                    for name in agent_names:
                        obs_t = torch.as_tensor(
                            np.asarray(obs_dict[name], dtype=np.float32),
                            device=device,
                        ).unsqueeze(0)
                        action, log_prob = policies[name].get_action_and_logprob(obs_t)
                        step_actions[name].append(action.squeeze(0))
                        step_log_probs[name].append(log_prob.squeeze(0))
                        actions_for_env[name] = action.squeeze(0).cpu().numpy()

                    # Step the environment
                    next_obs, rewards, terminations, truncations, infos = env.step(
                        actions_for_env
                    )

                    for name in agent_names:
                        step_rewards[name].append(
                            torch.tensor(rewards[name], dtype=torch.float32, device=device)
                        )

                    # Done if any agent is done
                    any_done = any(
                        terminations.get(name, False) or truncations.get(name, False)
                        for name in agent_names
                    )
                    all_dones.append(
                        torch.tensor(float(any_done), dtype=torch.float32, device=device)
                    )

                    if any_done:
                        # PettingZoo parallel envs clear their agent list on
                        # episode end; reset to get fresh observations.
                        next_obs, _ = env.reset()

                    env_observations[env_idx] = next_obs

                # Stack across envs for this step
                joint_obs_step = torch.stack(step_joint_obs)
                all_joint_obs.append(joint_obs_step)

                value = critic(joint_obs_step).squeeze(-1)
                all_values.append(value)

                for name in policies:
                    all_agent_actions[name].append(torch.stack(step_actions[name]))
                    all_agent_log_probs[name].append(torch.stack(step_log_probs[name]))
                    all_rewards[name].append(torch.stack(step_rewards[name]))

        # Stack temporal dimension: (n_steps, n_envs, ...)
        joint_obs_t = torch.stack(all_joint_obs)  # (n_steps, n_envs, joint_dim)
        values_t = torch.stack(all_values)  # (n_steps, n_envs)
        dones_t = torch.stack(all_dones).reshape(n_steps, self.n_envs)

        # Compute bootstrap value
        with torch.no_grad():
            final_joint = []
            for env_idx in range(self.n_envs):
                obs_dict = env_observations[env_idx]
                agent_obs_list = [
                    torch.as_tensor(
                        np.asarray(obs_dict[name], dtype=np.float32), device=device
                    )
                    for name in policies
                ]
                final_joint.append(torch.cat(agent_obs_list, dim=-1))
            final_joint_t = torch.stack(final_joint)
            bootstrap = critic(final_joint_t).squeeze(-1)

        # Compute GAE per environment
        import rlox as _rlox

        # Use shared rewards (mean across agents) for centralized value function
        agent_names = list(policies.keys())
        mean_rewards = torch.stack(
            [torch.stack(all_rewards[name]) for name in agent_names]
        ).mean(dim=0)  # (n_steps, n_envs)

        rewards_np = mean_rewards.cpu().numpy().astype(np.float64)
        values_np = values_t.detach().cpu().numpy().astype(np.float64)
        dones_np = dones_t.cpu().numpy().astype(np.float64)

        # Flatten in env-major order for compute_gae_batched
        rewards_flat = np.ascontiguousarray(rewards_np.T).ravel()
        values_flat = np.ascontiguousarray(values_np.T).ravel()
        dones_flat = np.ascontiguousarray(dones_np.T).ravel()
        last_vals = bootstrap.cpu().numpy().astype(np.float64)

        adv_flat, ret_flat = _rlox.compute_gae_batched(
            rewards=rewards_flat,
            values=values_flat,
            dones=dones_flat,
            last_values=last_vals,
            n_steps=n_steps,
            gamma=gamma,
            lam=gae_lambda,
        )

        advantages = (
            torch.as_tensor(adv_flat, dtype=torch.float32, device=device)
            .reshape(self.n_envs, n_steps)
            .T
        )
        returns = (
            torch.as_tensor(ret_flat, dtype=torch.float32, device=device)
            .reshape(self.n_envs, n_steps)
            .T
        )

        # Flatten temporal + env dims
        total = n_steps * self.n_envs

        flat_agent_obs = {}
        flat_agent_actions = {}
        flat_agent_log_probs = {}
        flat_rewards = {}

        for name in agent_names:
            flat_agent_obs[name] = torch.stack(all_agent_obs[name]).reshape(
                total, -1
            )
            flat_agent_actions[name] = torch.stack(
                all_agent_actions[name]
            ).reshape(total, *torch.stack(all_agent_actions[name]).shape[2:])
            flat_agent_log_probs[name] = torch.stack(
                all_agent_log_probs[name]
            ).reshape(total)
            flat_rewards[name] = torch.stack(all_rewards[name]).reshape(total)

        return MultiAgentRolloutBatch(
            agent_obs=flat_agent_obs,
            joint_obs=joint_obs_t.reshape(total, -1),
            agent_actions=flat_agent_actions,
            agent_log_probs=flat_agent_log_probs,
            rewards=flat_rewards,
            dones=dones_t.reshape(total),
            values=values_t.reshape(total),
            advantages=advantages.reshape(total),
            returns=returns.reshape(total),
        )


class MAPPO:
    """Multi-Agent PPO with centralized critic, decentralized actors.

    Auto-detects observation and action dimensions from the environment.
    For n_agents=1, this reduces to standard PPO on single-agent envs.
    For n_agents>1, uses MultiAgentCollector with PettingZoo parallel envs.

    Parameters
    ----------
    env_id : str
        Gymnasium environment ID (for n_agents=1) or PettingZoo env factory.
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
    env_fn : Callable, optional
        PettingZoo parallel env factory for n_agents > 1.
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
        env_fn: Callable[[], Any] | None = None,
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
            self.actors = nn.ModuleList(
                [
                    DiscretePolicy(obs_dim=obs_dim, n_actions=n_actions)
                    for _ in range(n_agents)
                ]
            )
        else:
            act_dim = int(np.prod(action_space.shape))
            self.actors = nn.ModuleList(
                [
                    ContinuousPolicy(obs_dim=obs_dim, act_dim=act_dim)
                    for _ in range(n_agents)
                ]
            )

        # Centralized critic -- input is joint obs (obs_dim * n_agents)
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

        # Set up the appropriate collector
        if n_agents > 1:
            self._multi_agent = True
            self._env_fn = env_fn
            # MultiAgentCollector is created lazily in train() to defer the
            # ImportError until actual multi-agent training is attempted.
            self._ma_collector: MultiAgentCollector | None = None
        else:
            self._multi_agent = False
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

    def _get_ma_collector(self) -> MultiAgentCollector:
        """Lazily create the MultiAgentCollector."""
        if self._ma_collector is None:
            if self._env_fn is None:
                raise ImportError(
                    "MAPPO with n_agents > 1 requires an env_fn argument that "
                    "returns a PettingZoo parallel environment. "
                    "Install pettingzoo with: pip install pettingzoo"
                )
            self._ma_collector = MultiAgentCollector(
                env_fn=self._env_fn,
                n_envs=self.n_envs,
                n_agents=self.n_agents,
            )
        return self._ma_collector

    def train(self, total_timesteps: int) -> dict[str, float]:
        """Run MAPPO training loop."""
        steps_per_rollout = self.n_envs * self.n_steps
        n_updates = max(1, total_timesteps // steps_per_rollout)

        all_rewards: list[float] = []
        last_metrics: dict[str, float] = {}

        if self._multi_agent:
            return self._train_multi_agent(n_updates)

        # For single-agent, use agent 0's actor as the policy
        policy = self.actors[0]

        self.callbacks.on_training_start()

        for update in range(n_updates):
            batch = self.collector.collect(policy, n_steps=self.n_steps)
            mean_ep_reward = batch.rewards.sum().item() / self.n_envs
            all_rewards.append(mean_ep_reward)

            self.callbacks.on_rollout_end(mean_reward=mean_ep_reward, update=update)

            for _epoch in range(self.n_epochs):
                for mb in batch.sample_minibatches(self.batch_size, shuffle=True):
                    adv = mb.advantages
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                    new_log_probs, entropy = policy.get_logprob_and_entropy(
                        mb.obs, mb.actions
                    )
                    # Centralized critic -- for n_agents=1, obs IS the full state
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
                    self.callbacks.on_train_batch(loss=loss.item(), **last_metrics)

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

    def _train_multi_agent(self, n_updates: int) -> dict[str, float]:
        """Multi-agent training loop using MultiAgentCollector."""
        collector = self._get_ma_collector()

        agent_names = [f"agent_{i}" for i in range(self.n_agents)]
        policies = {name: self.actors[i] for i, name in enumerate(agent_names)}

        all_rewards: list[float] = []
        last_metrics: dict[str, float] = {}

        self.callbacks.on_training_start()

        for update in range(n_updates):
            batch = collector.collect(
                policies=policies,
                critic=self.critic,
                n_steps=self.n_steps,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                device=self.device,
            )

            # Mean reward across agents
            mean_reward = sum(
                batch.rewards[name].sum().item() for name in agent_names
            ) / (self.n_agents * self.n_envs)
            all_rewards.append(mean_reward)

            self.callbacks.on_rollout_end(mean_reward=mean_reward, update=update)

            for _epoch in range(self.n_epochs):
                adv = batch.advantages
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                total_policy_loss = torch.tensor(0.0)
                total_entropy = torch.tensor(0.0)

                for name in agent_names:
                    policy = policies[name]
                    new_log_probs, entropy = policy.get_logprob_and_entropy(
                        batch.agent_obs[name], batch.agent_actions[name]
                    )

                    log_ratio = new_log_probs - batch.agent_log_probs[name]
                    ratio = log_ratio.exp()

                    pg_loss1 = -adv * ratio
                    pg_loss2 = -adv * torch.clamp(
                        ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps
                    )
                    total_policy_loss = total_policy_loss + torch.max(
                        pg_loss1, pg_loss2
                    ).mean()
                    total_entropy = total_entropy + entropy.mean()

                total_policy_loss = total_policy_loss / self.n_agents
                total_entropy = total_entropy / self.n_agents

                values = self.critic(batch.joint_obs).squeeze(-1)
                value_loss = 0.5 * ((values - batch.returns) ** 2).mean()

                loss = (
                    total_policy_loss
                    + self.vf_coef * value_loss
                    - self.ent_coef * total_entropy
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actors.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                last_metrics = {
                    "policy_loss": total_policy_loss.item(),
                    "value_loss": value_loss.item(),
                    "entropy": total_entropy.item(),
                }

                self._global_step += 1
                self.callbacks.on_train_batch(loss=loss.item(), **last_metrics)

            should_continue = self.callbacks.on_step(
                reward=mean_reward, step=self._global_step, algo=self
            )
            if not should_continue:
                break

            if self.logger is not None:
                self.logger.on_train_step(
                    update, {**last_metrics, "mean_reward": mean_reward}
                )

        self.callbacks.on_training_end()

        last_metrics["mean_reward"] = (
            float(sum(all_rewards) / len(all_rewards)) if all_rewards else 0.0
        )
        return last_metrics

    def predict(
        self, obs: Any, deterministic: bool = True, agent_idx: int = 0
    ) -> np.ndarray | int:
        """Get action from the trained policy for a single agent.

        Parameters
        ----------
        obs : array-like
            Observation.
        deterministic : bool
            If True, return the mode of the action distribution.
        agent_idx : int
            Which agent's policy to use (default 0).

        Returns
        -------
        Action as an int (discrete) or numpy array (continuous).
        """
        import torch

        actor = self.actors[agent_idx]
        obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                if self._is_discrete:
                    logits = actor.actor(obs_t)
                    action = logits.argmax(dim=-1)
                else:
                    action = actor.actor(obs_t)
            else:
                action, _ = actor.get_action_and_logprob(obs_t)
        action = action.squeeze(0)
        if self._is_discrete:
            return int(action.item())
        return action.numpy()

    def save(self, path: str) -> None:
        """Save training checkpoint."""
        import torch as _torch

        state = {
            "actors": self.actors.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": {"env_id": self.env_id, "n_agents": self.n_agents},
            "step": self._global_step,
        }
        _torch.save(state, path)

    @classmethod
    def from_checkpoint(cls, path: str, env_id: str | None = None) -> MAPPO:
        """Restore MAPPO from a checkpoint."""

        from rlox.checkpoint import safe_torch_load

        data = safe_torch_load(path)
        config = data.get("config", {})
        eid = env_id or config.get("env_id", "CartPole-v1")
        n_agents = config.get("n_agents", 1)

        mappo = cls(env_id=eid, n_agents=n_agents)
        mappo.actors.load_state_dict(data["actors"])
        mappo.critic.load_state_dict(data["critic"])
        mappo.optimizer.load_state_dict(data["optimizer"])
        mappo._global_step = data.get("step", 0)
        return mappo
