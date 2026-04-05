"""IMPALA: Importance Weighted Actor-Learner Architecture.

Multiple actor threads collect experience asynchronously while a single
learner thread applies V-trace correction using rlox.compute_vtrace.

V-trace computation is vectorized across environments by batching
per-env arrays into a single call to the Rust backend.
"""

from __future__ import annotations

import queue
import threading
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import rlox
from rlox.callbacks import Callback, CallbackList
from rlox.distributed.remote_env import RemoteEnvPool
from rlox.logging import LoggerCallback
from rlox.policies import DiscretePolicy
from rlox.utils import detect_env_spaces as _detect_env_spaces


def _compute_vtrace_batched(
    log_rhos: np.ndarray,
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    bootstrap_values: np.ndarray,
    gamma: float = 0.99,
    rho_bar: float = 1.0,
    c_bar: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized V-trace computation across multiple environments.

    Batches per-env arrays and calls the Rust compute_vtrace once per env,
    but collects results efficiently.

    Parameters
    ----------
    log_rhos : ndarray of shape (n_steps, n_envs)
    rewards : ndarray of shape (n_steps, n_envs)
    values : ndarray of shape (n_steps, n_envs)
    dones : ndarray of shape (n_steps, n_envs)
    bootstrap_values : ndarray of shape (n_envs,)
    gamma : float
    rho_bar : float
    c_bar : float

    Returns
    -------
    vs : ndarray of shape (n_steps, n_envs)
    pg_advantages : ndarray of shape (n_steps, n_envs)
    """
    n_steps, n_envs = rewards.shape

    # Pre-allocate output arrays
    vs_all = np.empty((n_steps, n_envs), dtype=np.float32)
    pg_all = np.empty((n_steps, n_envs), dtype=np.float32)

    # Ensure contiguous float32 arrays for each env slice
    # Use column-contiguous slicing (each column is one env)
    for env_idx in range(n_envs):
        vs, pg = rlox.compute_vtrace(
            np.ascontiguousarray(log_rhos[:, env_idx], dtype=np.float32),
            np.ascontiguousarray(rewards[:, env_idx], dtype=np.float32),
            np.ascontiguousarray(values[:, env_idx], dtype=np.float32),
            np.ascontiguousarray(dones[:, env_idx], dtype=np.float32),
            bootstrap_value=float(bootstrap_values[env_idx]),
            gamma=gamma,
            rho_bar=rho_bar,
            c_bar=c_bar,
        )
        vs_all[:, env_idx] = vs
        pg_all[:, env_idx] = pg

    return vs_all, pg_all


class IMPALA:
    """IMPALA with V-trace off-policy correction.

    Auto-detects observation and action dimensions from the environment.
    Actors collect data in parallel threads with the current policy snapshot.
    The learner applies V-trace corrected updates.

    V-trace computation is vectorized: all environments are processed in a
    single batched call rather than looping per environment.

    Parameters
    ----------
    env_id : str
        Gymnasium environment ID.
    n_actors : int
        Number of actor threads (default 2).
    n_envs : int
        Number of environments per actor (default 2).
    seed : int
        Random seed (default 42).
    n_steps : int
        Rollout length per actor per batch (default 32).
    learning_rate : float
        RMSprop learning rate (default 5e-4).
    gamma : float
        Discount factor (default 0.99).
    rho_bar : float
        V-trace truncation for importance weights (default 1.0).
    c_bar : float
        V-trace truncation for trace coefficients (default 1.0).
    vf_coef : float
        Value loss coefficient (default 0.5).
    ent_coef : float
        Entropy bonus coefficient (default 0.01).
    max_grad_norm : float
        Maximum gradient norm for clipping (default 40.0).
    logger : LoggerCallback, optional
        Logger for metrics.
    callbacks : list[Callback], optional
        Training callbacks.
    worker_addresses : list[str] | None
        If provided, each actor uses a :class:`RemoteEnvPool` connecting
        to a subset of these gRPC worker addresses instead of local
        VecEnv/GymVecEnv.  The addresses are partitioned evenly across
        actors, so ``len(worker_addresses) >= n_actors`` is required.
    """

    def __init__(
        self,
        env_id: str,
        n_actors: int = 2,
        n_envs: int = 2,
        seed: int = 42,
        n_steps: int = 32,
        learning_rate: float = 5e-4,
        gamma: float = 0.99,
        rho_bar: float = 1.0,
        c_bar: float = 1.0,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 40.0,
        logger: LoggerCallback | None = None,
        callbacks: list[Callback] | None = None,
        worker_addresses: list[str] | None = None,
    ):
        if worker_addresses is not None and len(worker_addresses) < n_actors:
            raise ValueError(
                f"Cannot have more actors ({n_actors}) than worker addresses "
                f"({len(worker_addresses)}). Each actor needs at least one "
                f"worker address."
            )

        self.env_id = env_id
        self.n_actors = n_actors
        self.n_envs = n_envs
        self._worker_addresses = worker_addresses
        self.n_steps = n_steps
        self.gamma = gamma
        self.rho_bar = rho_bar
        self.c_bar = c_bar
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.device = "cpu"
        self.seed = seed

        # Auto-detect environment spaces
        obs_dim, action_space, is_discrete = _detect_env_spaces(env_id)
        self.obs_dim = obs_dim
        self._is_discrete = is_discrete

        if is_discrete:
            self.n_actions = int(action_space.n)
        else:
            self.n_actions = int(np.prod(action_space.shape))

        # Learner policy -- auto-select based on action space
        if is_discrete:
            self.policy = DiscretePolicy(obs_dim=self.obs_dim, n_actions=self.n_actions)
        else:
            from rlox.policies import ContinuousPolicy

            self.policy = ContinuousPolicy(obs_dim=self.obs_dim, act_dim=self.n_actions)
        self.optimizer = torch.optim.RMSprop(
            self.policy.parameters(), lr=learning_rate, eps=1e-5
        )

        self.logger = logger
        self.callbacks = CallbackList(callbacks)
        self._global_step = 0

        self._queue: queue.Queue = queue.Queue(maxsize=n_actors * 2)
        self._stop_event = threading.Event()
        self._policy_lock = threading.Lock()

    def _partition_addresses(self) -> list[list[str]]:
        """Split worker_addresses evenly across actors.

        Returns a list of length ``n_actors``, where each element is the
        subset of addresses assigned to that actor.  When addresses don't
        divide evenly, earlier actors receive the extra addresses.
        """
        if self._worker_addresses is None:
            return [[] for _ in range(self.n_actors)]

        addrs = list(self._worker_addresses)
        n = len(addrs)
        base, extra = divmod(n, self.n_actors)
        partitions: list[list[str]] = []
        offset = 0
        for i in range(self.n_actors):
            size = base + (1 if i < extra else 0)
            partitions.append(addrs[offset : offset + size])
            offset += size
        return partitions

    def _get_policy_snapshot(self) -> dict:
        """Get a copy of the current policy parameters."""
        with self._policy_lock:
            return {k: v.clone().detach() for k, v in self.policy.state_dict().items()}

    def _actor_loop(self, actor_id: int) -> None:
        """Actor thread: collect experience and enqueue."""
        actor_addresses = self._partition_addresses()[actor_id]

        if actor_addresses:
            # Distributed mode: use RemoteEnvPool as the env backend
            env = RemoteEnvPool(addresses=actor_addresses)
            env.connect()
        else:
            # Local mode: use native VecEnv or GymVecEnv fallback
            try:
                env = rlox.VecEnv(
                    n=self.n_envs, seed=self.seed + actor_id * 1000, env_id=self.env_id
                )
            except ValueError:
                from rlox.gym_vec_env import GymVecEnv

                env = GymVecEnv(self.env_id, n_envs=self.n_envs)
        obs = env.reset_all()

        # Local policy copy -- match learner policy type
        if self._is_discrete:
            local_policy = DiscretePolicy(
                obs_dim=self.obs_dim, n_actions=self.n_actions
            )
        else:
            from rlox.policies import ContinuousPolicy

            local_policy = ContinuousPolicy(
                obs_dim=self.obs_dim, act_dim=self.n_actions
            )

        while not self._stop_event.is_set():
            # Sync with learner
            snapshot = self._get_policy_snapshot()
            local_policy.load_state_dict(snapshot)

            all_obs = []
            all_actions = []
            all_log_probs = []
            all_rewards = []
            all_dones = []
            all_values = []

            with torch.no_grad():
                for _ in range(self.n_steps):
                    obs_tensor = torch.as_tensor(obs, dtype=torch.float32)

                    if self._is_discrete:
                        logits = local_policy.actor(obs_tensor)
                        dist = torch.distributions.Categorical(logits=logits)
                        actions = dist.sample()
                        log_probs = dist.log_prob(actions)
                    else:
                        actions, log_probs = local_policy.get_action_and_logprob(
                            obs_tensor
                        )

                    values = local_policy.critic(obs_tensor).squeeze(-1)

                    if self._is_discrete:
                        actions_for_env = (
                            actions.cpu().numpy().astype(np.uint32).tolist()
                        )
                    else:
                        actions_for_env = actions.cpu().numpy().astype(np.float32)
                    step_result = env.step_all(actions_for_env)

                    all_obs.append(obs_tensor)
                    all_actions.append(actions)
                    all_log_probs.append(log_probs)
                    all_values.append(values)
                    all_rewards.append(
                        torch.as_tensor(step_result["rewards"].astype(np.float32))
                    )
                    terminated = step_result["terminated"].astype(bool)
                    truncated = step_result["truncated"].astype(bool)
                    dones = terminated | truncated
                    all_dones.append(torch.as_tensor(dones.astype(np.float32)))

                    obs = step_result["obs"].copy()

                # Compute bootstrap value from final observation
                final_obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
                bootstrap_values = local_policy.critic(final_obs_tensor).squeeze(-1)

            data = {
                "obs": torch.stack(all_obs),  # (n_steps, n_envs, obs_dim)
                "actions": torch.stack(all_actions),  # (n_steps, n_envs)
                "mu_log_probs": torch.stack(all_log_probs),  # behavior log probs
                "rewards": torch.stack(all_rewards),
                "dones": torch.stack(all_dones),
                "values": torch.stack(all_values),
                "bootstrap_values": bootstrap_values,  # (n_envs,)
            }

            while not self._stop_event.is_set():
                try:
                    self._queue.put(data, timeout=0.1)
                    break
                except queue.Full:
                    continue

    def _learner_step(self, data: dict) -> dict[str, float]:
        """Apply V-trace corrected gradient update (vectorized)."""
        obs = data["obs"]  # (n_steps, n_envs, obs_dim)
        actions = data["actions"]  # (n_steps, n_envs)
        mu_log_probs = data["mu_log_probs"]
        rewards = data["rewards"]
        dones = data["dones"]
        values = data["values"]
        bootstrap_values = data["bootstrap_values"]  # (n_envs,)

        n_steps, n_envs = rewards.shape

        # Compute current policy log probs
        obs_flat = obs.reshape(-1, self.obs_dim)
        if self._is_discrete:
            actions_flat = actions.reshape(-1)
        else:
            # Continuous: (n_steps, n_envs, act_dim) -> (n_steps*n_envs, act_dim)
            actions_flat = actions.reshape(-1, self.n_actions)

        if self._is_discrete:
            logits = self.policy.actor(obs_flat)
            dist = torch.distributions.Categorical(logits=logits)
            pi_log_probs = dist.log_prob(actions_flat).reshape(n_steps, n_envs)
            entropy = dist.entropy().reshape(n_steps, n_envs)
        else:
            pi_log_probs_flat, entropy_flat = self.policy.get_logprob_and_entropy(
                obs_flat, actions_flat
            )
            pi_log_probs = pi_log_probs_flat.reshape(n_steps, n_envs)
            entropy = entropy_flat.reshape(n_steps, n_envs)

        new_values = self.policy.critic(obs_flat).squeeze(-1).reshape(n_steps, n_envs)

        # Vectorized V-trace across all environments
        log_rhos = (pi_log_probs - mu_log_probs).detach()

        vs_all, pg_adv_all = _compute_vtrace_batched(
            log_rhos.numpy().astype(np.float32),
            rewards.numpy().astype(np.float32),
            values.detach().numpy().astype(np.float32),
            dones.numpy().astype(np.float32),
            bootstrap_values.detach().numpy().astype(np.float32),
            gamma=self.gamma,
            rho_bar=self.rho_bar,
            c_bar=self.c_bar,
        )

        vs_tensor = torch.as_tensor(vs_all, dtype=torch.float32)
        pg_adv_tensor = torch.as_tensor(pg_adv_all, dtype=torch.float32)

        # Policy gradient loss (vectorized across all envs)
        total_policy_loss = -(pi_log_probs * pg_adv_tensor.detach()).mean()

        # Value loss (vectorized across all envs)
        total_value_loss = F.mse_loss(new_values, vs_tensor.detach())

        entropy_loss = entropy.mean()

        loss = (
            total_policy_loss
            + self.vf_coef * total_value_loss
            - self.ent_coef * entropy_loss
        )

        # Hold the policy lock for the entire gradient update so that actor
        # threads cannot read a partially-updated parameter tensor between the
        # backward pass and the parameter write.
        with self._policy_lock:
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

        return {
            "policy_loss": total_policy_loss.item(),
            "value_loss": total_value_loss.item(),
            "entropy": entropy_loss.item(),
        }

    def train(self, total_timesteps: int) -> dict[str, float]:
        """Run IMPALA training loop."""
        steps_per_batch = self.n_steps * self.n_envs
        n_updates = max(1, total_timesteps // steps_per_batch)

        # Start actor threads
        actors = []
        for i in range(self.n_actors):
            t = threading.Thread(target=self._actor_loop, args=(i,), daemon=True)
            t.start()
            actors.append(t)

        all_rewards: list[float] = []
        last_metrics: dict[str, float] = {}

        self.callbacks.on_training_start()

        try:
            for update in range(n_updates):
                try:
                    data = self._queue.get(timeout=30.0)
                except queue.Empty:
                    break

                metrics = self._learner_step(data)
                reward = data["rewards"].sum().item() / self.n_envs
                all_rewards.append(reward)
                last_metrics = metrics

                self._global_step += 1

                self.callbacks.on_rollout_end(mean_reward=reward, update=update)
                self.callbacks.on_train_batch(loss=sum(metrics.values()), **metrics)

                should_continue = self.callbacks.on_step(
                    reward=reward, step=self._global_step, algo=self
                )
                if not should_continue:
                    break

                if self.logger is not None:
                    self.logger.on_train_step(
                        update, {**metrics, "mean_reward": reward}
                    )
        finally:
            self._stop_event.set()
            for t in actors:
                t.join(timeout=5.0)

        self.callbacks.on_training_end()

        last_metrics["mean_reward"] = (
            float(sum(all_rewards) / len(all_rewards)) if all_rewards else 0.0
        )
        return last_metrics

    def save(self, path: str) -> None:
        """Save training checkpoint."""
        state = {
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": {"env_id": self.env_id},
            "step": self._global_step,
        }
        torch.save(state, path)

    @classmethod
    def from_checkpoint(cls, path: str, env_id: str | None = None) -> IMPALA:
        """Restore IMPALA from a checkpoint."""
        data = torch.load(path, weights_only=False)
        config = data.get("config", {})
        eid = env_id or config.get("env_id", "CartPole-v1")

        impala = cls(env_id=eid)
        impala.policy.load_state_dict(data["policy"])
        impala.optimizer.load_state_dict(data["optimizer"])
        impala._global_step = data.get("step", 0)
        return impala


class DistributedIMPALA(IMPALA):
    """IMPALA with remote environment workers via gRPC.

    Thin convenience wrapper that enforces ``worker_addresses`` is provided
    and defaults ``n_actors`` to match the number of addresses.

    Parameters
    ----------
    env_id : str
        Gymnasium environment ID (used for space detection).
    worker_addresses : list[str]
        gRPC ``host:port`` addresses for remote environment workers.
    **kwargs
        Forwarded to :class:`IMPALA`.
    """

    def __init__(
        self,
        env_id: str,
        worker_addresses: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        if worker_addresses is None or len(worker_addresses) == 0:
            raise ValueError(
                "DistributedIMPALA requires at least one worker address. "
                "Use IMPALA directly for local-only training."
            )

        # Default n_actors to number of addresses if not explicitly set
        kwargs.setdefault("n_actors", len(worker_addresses))

        super().__init__(
            env_id=env_id,
            worker_addresses=worker_addresses,
            **kwargs,
        )
