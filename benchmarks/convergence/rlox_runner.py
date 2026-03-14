"""rlox training + evaluation harness for convergence benchmarks.

Run in a separate process to avoid import contamination with SB3.
"""

from __future__ import annotations

import copy
import resource
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

import gymnasium.vector

import rlox
from rlox.batch import RolloutBatch
from rlox.losses import PPOLoss
from rlox.networks import (
    DeterministicPolicy,
    QNetwork,
    SimpleQNetwork,
    SquashedGaussianPolicy,
    polyak_update,
)
from rlox.policies import DiscretePolicy

from common import (
    EvalRecord,
    ExperimentLog,
    evaluate_policy_gym,
    get_hardware_info,
    load_config,
    result_path,
)


# ---------------------------------------------------------------------------
# On-policy trainers (PPO, A2C) with periodic evaluation
# ---------------------------------------------------------------------------


def _collect_rollout_gym(
    vec_env: gymnasium.vector.VectorEnv,
    policy: nn.Module,
    obs: np.ndarray,
    n_steps: int,
    n_envs: int,
    gamma: float,
    gae_lambda: float,
    is_discrete: bool,
) -> tuple[RolloutBatch, np.ndarray]:
    """Collect on-policy rollout using gymnasium VectorEnv + rlox GAE.

    Uses gymnasium for env stepping, rlox.compute_gae for advantage estimation.
    Returns (batch, next_obs).
    """
    all_obs = []
    all_actions = []
    all_rewards = []
    all_dones = []
    all_log_probs = []
    all_values = []

    for _ in range(n_steps):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)

        with torch.no_grad():
            actions, log_probs = policy.get_action_and_logprob(obs_tensor)
            values = policy.get_value(obs_tensor)

        if is_discrete:
            actions_env = actions.cpu().numpy().astype(np.int64)
        else:
            actions_env = actions.cpu().numpy()

        next_obs, rewards, terminated, truncated, infos = vec_env.step(actions_env)
        dones = terminated | truncated

        all_obs.append(obs_tensor)
        all_actions.append(actions)
        all_log_probs.append(log_probs)
        all_values.append(values)
        all_rewards.append(torch.as_tensor(rewards.astype(np.float32)))
        all_dones.append(torch.as_tensor(dones.astype(np.float32)))

        obs = next_obs

    # Bootstrap value for GAE
    with torch.no_grad():
        last_values = policy.get_value(torch.as_tensor(obs, dtype=torch.float32))

    # Compute GAE per environment using rlox Rust core, then concatenate
    all_advantages = []
    all_returns = []
    for env_idx in range(n_envs):
        rewards_env = torch.stack([r[env_idx] for r in all_rewards])
        values_env = torch.stack([v[env_idx] for v in all_values])
        dones_env = torch.stack([d[env_idx] for d in all_dones])

        adv, ret = rlox.compute_gae(
            rewards=rewards_env.numpy().astype(np.float64),
            values=values_env.numpy().astype(np.float64),
            dones=dones_env.numpy().astype(np.float64),
            last_value=float(last_values[env_idx]),
            gamma=gamma,
            lam=gae_lambda,
        )
        all_advantages.append(torch.as_tensor(adv, dtype=torch.float32))
        all_returns.append(torch.as_tensor(ret, dtype=torch.float32))

    obs_t = torch.stack(all_obs)
    actions_t = torch.stack(all_actions)
    rewards_t = torch.stack(all_rewards)
    dones_t = torch.stack(all_dones)
    log_probs_t = torch.stack(all_log_probs)
    values_t = torch.stack(all_values)
    advantages_t = torch.stack(all_advantages).T
    returns_t = torch.stack(all_returns).T

    total = n_steps * n_envs
    batch = RolloutBatch(
        obs=obs_t.reshape(total, -1),
        actions=actions_t.reshape(total) if actions_t.dim() == 2 else actions_t.reshape(total, -1),
        rewards=rewards_t.reshape(total),
        dones=dones_t.reshape(total),
        log_probs=log_probs_t.reshape(total),
        values=values_t.reshape(total),
        advantages=advantages_t.reshape(total),
        returns=returns_t.reshape(total),
    )
    return batch, obs


def _run_ppo(
    env_id: str,
    hp: dict[str, Any],
    policy_cfg: dict[str, Any],
    seed: int,
    max_steps: int,
    eval_freq: int,
    eval_episodes: int,
    log: ExperimentLog,
) -> None:
    """PPO training loop with periodic evaluation.

    Uses gymnasium VectorEnv for stepping + rlox.compute_gae for advantages.
    """
    n_envs = hp.get("n_envs", 8)
    n_steps = hp.get("n_steps", 2048)
    n_epochs = hp.get("n_epochs", 10)
    batch_size = hp.get("batch_size", 64)
    lr = hp.get("learning_rate", 3e-4)
    gamma = hp.get("gamma", 0.99)
    gae_lambda = hp.get("gae_lambda", 0.95)
    clip_eps = hp.get("clip_range", 0.2)
    ent_coef = hp.get("ent_coef", 0.0)
    vf_coef = hp.get("vf_coef", 0.5)
    max_grad_norm = hp.get("max_grad_norm", 0.5)

    hidden = policy_cfg.get("hidden_sizes", [64, 64])

    # Detect obs/action dimensions from a probe env
    probe_env = gym.make(env_id)
    obs_dim = int(np.prod(probe_env.observation_space.shape))
    is_discrete = hasattr(probe_env.action_space, "n")
    if is_discrete:
        n_actions = int(probe_env.action_space.n)
    else:
        n_actions = int(np.prod(probe_env.action_space.shape))
    probe_env.close()

    if is_discrete:
        policy = DiscretePolicy(obs_dim, n_actions, hidden=hidden[0])
    else:
        policy = _ContinuousActorCritic(obs_dim, n_actions, hidden)

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr, eps=1e-5)
    loss_fn = PPOLoss(
        clip_eps=clip_eps,
        vf_coef=vf_coef,
        ent_coef=ent_coef,
        max_grad_norm=max_grad_norm,
    )

    vec_env = gymnasium.vector.SyncVectorEnv(
        [lambda i=i: gym.make(env_id) for i in range(n_envs)]
    )
    obs, _ = vec_env.reset(seed=seed)

    steps_per_rollout = n_envs * n_steps
    n_updates = max(1, max_steps // steps_per_rollout)
    total_steps = 0
    start_time = time.monotonic()
    last_eval_step = 0

    for update in range(n_updates):
        # LR annealing
        frac = 1.0 - update / n_updates
        for pg in optimizer.param_groups:
            pg["lr"] = lr * frac

        batch, obs = _collect_rollout_gym(
            vec_env, policy, obs, n_steps, n_envs, gamma, gae_lambda, is_discrete,
        )
        total_steps += steps_per_rollout

        for _epoch in range(n_epochs):
            for mb in batch.sample_minibatches(batch_size, shuffle=True):
                adv = mb.advantages
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                loss, _ = loss_fn(
                    policy, mb.obs, mb.actions, mb.log_probs,
                    adv, mb.returns, mb.values,
                )
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                optimizer.step()

        # Periodic evaluation
        if total_steps - last_eval_step >= eval_freq:
            last_eval_step = total_steps
            _do_eval(env_id, policy, is_discrete, total_steps, start_time,
                     eval_episodes, seed, log)

    vec_env.close()


def _run_a2c(
    env_id: str,
    hp: dict[str, Any],
    policy_cfg: dict[str, Any],
    seed: int,
    max_steps: int,
    eval_freq: int,
    eval_episodes: int,
    log: ExperimentLog,
) -> None:
    """A2C training loop with periodic evaluation.

    Uses gymnasium VectorEnv for stepping + rlox.compute_gae for advantages.
    """
    n_envs = hp.get("n_envs", 8)
    n_steps = hp.get("n_steps", 5)
    lr = hp.get("learning_rate", 7e-4)
    gamma = hp.get("gamma", 0.99)
    gae_lambda = hp.get("gae_lambda", 1.0)
    vf_coef = hp.get("vf_coef", 0.5)
    ent_coef = hp.get("ent_coef", 0.01)
    max_grad_norm = hp.get("max_grad_norm", 0.5)

    hidden = policy_cfg.get("hidden_sizes", [64, 64])

    probe_env = gym.make(env_id)
    obs_dim = int(np.prod(probe_env.observation_space.shape))
    n_actions = int(probe_env.action_space.n)
    probe_env.close()

    policy = DiscretePolicy(obs_dim, n_actions, hidden=hidden[0])
    optimizer = torch.optim.RMSprop(policy.parameters(), lr=lr, eps=1e-5, alpha=0.99)

    vec_env = gymnasium.vector.SyncVectorEnv(
        [lambda i=i: gym.make(env_id) for i in range(n_envs)]
    )
    obs, _ = vec_env.reset(seed=seed)

    steps_per_rollout = n_envs * n_steps
    n_updates = max(1, max_steps // steps_per_rollout)
    total_steps = 0
    start_time = time.monotonic()
    last_eval_step = 0

    for update in range(n_updates):
        batch, obs = _collect_rollout_gym(
            vec_env, policy, obs, n_steps, n_envs, gamma, gae_lambda,
            is_discrete=True,
        )
        total_steps += steps_per_rollout

        batch_obs = batch.obs
        actions = batch.actions
        advantages = batch.advantages
        returns = batch.returns
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        log_probs, entropy = policy.get_logprob_and_entropy(batch_obs, actions)
        values = policy.get_value(batch_obs)

        policy_loss = -(log_probs * advantages.detach()).mean()
        value_loss = 0.5 * ((values - returns) ** 2).mean()
        entropy_loss = entropy.mean()
        loss = policy_loss + vf_coef * value_loss - ent_coef * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
        optimizer.step()

        if total_steps - last_eval_step >= eval_freq:
            last_eval_step = total_steps
            _do_eval(env_id, policy, True, total_steps, start_time,
                     eval_episodes, seed, log)

    vec_env.close()


# ---------------------------------------------------------------------------
# Off-policy trainers (SAC, TD3, DQN) with periodic evaluation
# ---------------------------------------------------------------------------


def _run_sac(
    env_id: str,
    hp: dict[str, Any],
    policy_cfg: dict[str, Any],
    seed: int,
    max_steps: int,
    eval_freq: int,
    eval_episodes: int,
    log: ExperimentLog,
) -> None:
    """SAC training loop with periodic evaluation."""
    env = gym.make(env_id)
    env.reset(seed=seed)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    act_high = float(env.action_space.high[0])

    hidden = policy_cfg.get("hidden_sizes", [256, 256])[0]
    lr = hp.get("learning_rate", 3e-4)
    gamma = hp.get("gamma", 0.99)
    tau = hp.get("tau", 0.005)
    batch_size = hp.get("batch_size", 256)
    buffer_size = hp.get("buffer_size", 1_000_000)
    learning_starts = hp.get("learning_starts", 10_000)

    actor = SquashedGaussianPolicy(obs_dim, act_dim, hidden)
    critic1 = QNetwork(obs_dim, act_dim, hidden)
    critic2 = QNetwork(obs_dim, act_dim, hidden)
    critic1_target = copy.deepcopy(critic1)
    critic2_target = copy.deepcopy(critic2)

    actor_opt = torch.optim.Adam(actor.parameters(), lr=lr)
    critic1_opt = torch.optim.Adam(critic1.parameters(), lr=lr)
    critic2_opt = torch.optim.Adam(critic2.parameters(), lr=lr)

    # Automatic entropy tuning
    target_entropy = -float(act_dim)
    log_alpha = torch.zeros(1, requires_grad=True)
    alpha_opt = torch.optim.Adam([log_alpha], lr=lr)
    alpha = log_alpha.exp().item()

    buffer = rlox.ReplayBuffer(buffer_size, obs_dim, act_dim)

    obs, _ = env.reset()
    start_time = time.monotonic()
    last_eval_step = 0

    for step in range(max_steps):
        if step < learning_starts:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                action_t, _ = actor.sample(obs_t)
                action = action_t.squeeze(0).numpy()
            action = np.clip(action, -act_high, act_high)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        buffer.push(
            np.asarray(obs, dtype=np.float32),
            np.asarray(action, dtype=np.float32),
            float(reward), bool(terminated), bool(truncated),
        )
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()

        # Update
        if step >= learning_starts and len(buffer) >= batch_size:
            batch = buffer.sample(batch_size, step)
            obs_b = torch.as_tensor(np.asarray(batch["obs"]), dtype=torch.float32)
            act_b = torch.as_tensor(np.asarray(batch["actions"]), dtype=torch.float32)
            if act_b.dim() == 1:
                act_b = act_b.unsqueeze(-1)
            rew_b = torch.as_tensor(np.asarray(batch["rewards"]), dtype=torch.float32)
            done_b = torch.as_tensor(np.asarray(batch["terminated"]), dtype=torch.float32)

            with torch.no_grad():
                next_a, next_lp = actor.sample(obs_b)
                q1_next = critic1_target(obs_b, next_a).squeeze(-1)
                q2_next = critic2_target(obs_b, next_a).squeeze(-1)
                q_next = torch.min(q1_next, q2_next) - alpha * next_lp
                target_q = rew_b + gamma * (1.0 - done_b) * q_next

            import torch.nn.functional as F

            q1 = critic1(obs_b, act_b).squeeze(-1)
            q2 = critic2(obs_b, act_b).squeeze(-1)
            c1_loss = F.mse_loss(q1, target_q)
            c2_loss = F.mse_loss(q2, target_q)

            critic1_opt.zero_grad(); c1_loss.backward(); critic1_opt.step()
            critic2_opt.zero_grad(); c2_loss.backward(); critic2_opt.step()

            new_a, new_lp = actor.sample(obs_b)
            q_new = torch.min(critic1(obs_b, new_a).squeeze(-1),
                              critic2(obs_b, new_a).squeeze(-1))
            actor_loss = (alpha * new_lp - q_new).mean()
            actor_opt.zero_grad(); actor_loss.backward(); actor_opt.step()

            # Alpha
            alpha_loss = -(log_alpha * (new_lp.detach() + target_entropy)).mean()
            alpha_opt.zero_grad(); alpha_loss.backward(); alpha_opt.step()
            alpha = log_alpha.exp().item()

            polyak_update(critic1, critic1_target, tau)
            polyak_update(critic2, critic2_target, tau)

        # Periodic evaluation
        if (step + 1) - last_eval_step >= eval_freq:
            last_eval_step = step + 1

            def get_action(o: np.ndarray) -> np.ndarray:
                with torch.no_grad():
                    ot = torch.as_tensor(o, dtype=torch.float32).unsqueeze(0)
                    return actor.deterministic(ot).squeeze(0).numpy()

            wall_clock = time.monotonic() - start_time
            sps = (step + 1) / max(wall_clock, 1e-9)
            mean_ret, std_ret, mean_len = evaluate_policy_gym(
                env_id, get_action, eval_episodes, seed + 1000,
            )
            log.evaluations.append(EvalRecord(
                step=step + 1, wall_clock_s=wall_clock,
                mean_return=mean_ret, std_return=std_ret,
                ep_length=mean_len, sps=sps,
            ))
            print(
                f"  [rlox] step={step + 1:>8d}  "
                f"return={mean_ret:>8.1f} +/- {std_ret:>6.1f}  "
                f"SPS={sps:>7.0f}  wall={wall_clock:>6.1f}s"
            )

    env.close()


def _run_td3(
    env_id: str,
    hp: dict[str, Any],
    policy_cfg: dict[str, Any],
    seed: int,
    max_steps: int,
    eval_freq: int,
    eval_episodes: int,
    log: ExperimentLog,
) -> None:
    """TD3 training loop with periodic evaluation."""
    env = gym.make(env_id)
    env.reset(seed=seed)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    act_high = float(env.action_space.high[0])

    hidden = policy_cfg.get("hidden_sizes", [400, 300])
    lr = hp.get("learning_rate", 1e-3)
    gamma = hp.get("gamma", 0.99)
    tau = hp.get("tau", 0.005)
    batch_size = hp.get("batch_size", 256)
    buffer_size = hp.get("buffer_size", 1_000_000)
    learning_starts = hp.get("learning_starts", 10_000)
    policy_delay = hp.get("policy_delay", 2)
    target_noise = hp.get("target_policy_noise", 0.2)
    noise_clip = hp.get("target_noise_clip", 0.5)
    exploration_noise = hp.get("exploration_noise", 0.1)

    # TD3 uses two hidden layer sizes
    h1 = hidden[0] if len(hidden) >= 1 else 400
    h2 = hidden[1] if len(hidden) >= 2 else 300

    actor = DeterministicPolicy(obs_dim, act_dim, h1, act_high)
    actor_target = copy.deepcopy(actor)
    critic1 = QNetwork(obs_dim, act_dim, h1)
    critic2 = QNetwork(obs_dim, act_dim, h1)
    critic1_target = copy.deepcopy(critic1)
    critic2_target = copy.deepcopy(critic2)

    actor_opt = torch.optim.Adam(actor.parameters(), lr=lr)
    critic1_opt = torch.optim.Adam(critic1.parameters(), lr=lr)
    critic2_opt = torch.optim.Adam(critic2.parameters(), lr=lr)

    buffer = rlox.ReplayBuffer(buffer_size, obs_dim, act_dim)

    obs, _ = env.reset()
    start_time = time.monotonic()
    last_eval_step = 0
    update_count = 0

    for step in range(max_steps):
        if step < learning_starts:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                action = actor(obs_t).squeeze(0).numpy()
                noise = np.random.randn(act_dim) * exploration_noise
                action = np.clip(action + noise, -act_high, act_high)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        buffer.push(
            np.asarray(obs, dtype=np.float32),
            np.asarray(action, dtype=np.float32),
            float(reward), bool(terminated), bool(truncated),
        )
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()

        # Update
        if step >= learning_starts and len(buffer) >= batch_size:
            update_count += 1
            batch = buffer.sample(batch_size, step)
            obs_b = torch.as_tensor(np.asarray(batch["obs"]), dtype=torch.float32)
            act_b = torch.as_tensor(np.asarray(batch["actions"]), dtype=torch.float32)
            if act_b.dim() == 1:
                act_b = act_b.unsqueeze(-1)
            rew_b = torch.as_tensor(np.asarray(batch["rewards"]), dtype=torch.float32)
            done_b = torch.as_tensor(np.asarray(batch["terminated"]), dtype=torch.float32)

            import torch.nn.functional as F

            with torch.no_grad():
                noise = torch.randn_like(act_b) * target_noise
                noise = noise.clamp(-noise_clip, noise_clip)
                next_a = (actor_target(obs_b) + noise).clamp(-act_high, act_high)
                q1_next = critic1_target(obs_b, next_a).squeeze(-1)
                q2_next = critic2_target(obs_b, next_a).squeeze(-1)
                tgt = rew_b + gamma * (1.0 - done_b) * torch.min(q1_next, q2_next)

            q1 = critic1(obs_b, act_b).squeeze(-1)
            q2 = critic2(obs_b, act_b).squeeze(-1)
            c1_loss = F.mse_loss(q1, tgt)
            c2_loss = F.mse_loss(q2, tgt)
            critic1_opt.zero_grad(); c1_loss.backward(); critic1_opt.step()
            critic2_opt.zero_grad(); c2_loss.backward(); critic2_opt.step()

            if update_count % policy_delay == 0:
                a_loss = -critic1(obs_b, actor(obs_b)).mean()
                actor_opt.zero_grad(); a_loss.backward(); actor_opt.step()
                polyak_update(actor, actor_target, tau)
                polyak_update(critic1, critic1_target, tau)
                polyak_update(critic2, critic2_target, tau)

        # Periodic evaluation
        if (step + 1) - last_eval_step >= eval_freq:
            last_eval_step = step + 1

            def get_action(o: np.ndarray) -> np.ndarray:
                with torch.no_grad():
                    ot = torch.as_tensor(o, dtype=torch.float32).unsqueeze(0)
                    return actor(ot).squeeze(0).numpy()

            wall_clock = time.monotonic() - start_time
            sps = (step + 1) / max(wall_clock, 1e-9)
            mean_ret, std_ret, mean_len = evaluate_policy_gym(
                env_id, get_action, eval_episodes, seed + 1000,
            )
            log.evaluations.append(EvalRecord(
                step=step + 1, wall_clock_s=wall_clock,
                mean_return=mean_ret, std_return=std_ret,
                ep_length=mean_len, sps=sps,
            ))
            print(
                f"  [rlox] step={step + 1:>8d}  "
                f"return={mean_ret:>8.1f} +/- {std_ret:>6.1f}  "
                f"SPS={sps:>7.0f}  wall={wall_clock:>6.1f}s"
            )

    env.close()


def _run_dqn(
    env_id: str,
    hp: dict[str, Any],
    policy_cfg: dict[str, Any],
    seed: int,
    max_steps: int,
    eval_freq: int,
    eval_episodes: int,
    log: ExperimentLog,
) -> None:
    """DQN training loop with periodic evaluation."""
    env = gym.make(env_id)
    env.reset(seed=seed)
    obs_dim = int(np.prod(env.observation_space.shape))
    n_actions = int(env.action_space.n)

    hidden = policy_cfg.get("hidden_sizes", [64, 64])[0]
    lr = hp.get("learning_rate", 1e-4)
    gamma = hp.get("gamma", 0.99)
    batch_size = hp.get("batch_size", 64)
    buffer_size = hp.get("buffer_size", 100_000)
    learning_starts = hp.get("learning_starts", 1000)
    target_update_freq = hp.get("target_update_interval", 10_000)
    exploration_fraction = hp.get("exploration_fraction", 0.1)
    exploration_final_eps = hp.get("exploration_final_eps", 0.05)

    q_net = SimpleQNetwork(obs_dim, n_actions, hidden)
    target_net = copy.deepcopy(q_net)
    optimizer = torch.optim.Adam(q_net.parameters(), lr=lr)

    buffer = rlox.ReplayBuffer(buffer_size, obs_dim, 1)

    obs, _ = env.reset()
    start_time = time.monotonic()
    last_eval_step = 0

    for step in range(max_steps):
        # Epsilon schedule
        frac = min(1.0, step / max(1, int(max_steps * exploration_fraction)))
        eps = 1.0 + frac * (exploration_final_eps - 1.0)

        if np.random.random() < eps or step < learning_starts:
            action = int(env.action_space.sample())
        else:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                action = int(q_net(obs_t).argmax(dim=-1).item())

        next_obs, reward, terminated, truncated, _ = env.step(action)
        buffer.push(
            np.asarray(obs, dtype=np.float32),
            np.array([float(action)], dtype=np.float32),
            float(reward), bool(terminated), bool(truncated),
        )
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()

        # Update
        if step >= learning_starts and len(buffer) >= batch_size:
            import torch.nn.functional as F

            batch = buffer.sample(batch_size, step)
            obs_b = torch.as_tensor(np.asarray(batch["obs"]), dtype=torch.float32)
            act_b = torch.as_tensor(np.asarray(batch["actions"]), dtype=torch.long).squeeze(-1)
            rew_b = torch.as_tensor(np.asarray(batch["rewards"]), dtype=torch.float32)
            done_b = torch.as_tensor(np.asarray(batch["terminated"]), dtype=torch.float32)

            q = q_net(obs_b).gather(1, act_b.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                # Double DQN
                next_actions = q_net(obs_b).argmax(dim=-1)
                next_q = target_net(obs_b).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                target_q = rew_b + gamma * (1.0 - done_b) * next_q

            loss = F.mse_loss(q, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if step % target_update_freq == 0:
            target_net.load_state_dict(q_net.state_dict())

        # Periodic evaluation
        if (step + 1) - last_eval_step >= eval_freq:
            last_eval_step = step + 1

            def get_action(o: np.ndarray) -> int:
                with torch.no_grad():
                    ot = torch.as_tensor(o, dtype=torch.float32).unsqueeze(0)
                    return int(q_net(ot).argmax(dim=-1).item())

            wall_clock = time.monotonic() - start_time
            sps = (step + 1) / max(wall_clock, 1e-9)
            mean_ret, std_ret, mean_len = evaluate_policy_gym(
                env_id, get_action, eval_episodes, seed + 1000,
            )
            log.evaluations.append(EvalRecord(
                step=step + 1, wall_clock_s=wall_clock,
                mean_return=mean_ret, std_return=std_ret,
                ep_length=mean_len, sps=sps,
            ))
            print(
                f"  [rlox] step={step + 1:>8d}  "
                f"return={mean_ret:>8.1f} +/- {std_ret:>6.1f}  "
                f"SPS={sps:>7.0f}  wall={wall_clock:>6.1f}s"
            )

    env.close()


# ---------------------------------------------------------------------------
# Continuous-action PPO actor-critic
# ---------------------------------------------------------------------------


class _ContinuousActorCritic(nn.Module):
    """Gaussian actor-critic for continuous PPO."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: list[int]):
        super().__init__()
        h = hidden_sizes[0] if hidden_sizes else 64

        self.actor_mean = nn.Sequential(
            nn.Linear(obs_dim, h), nn.Tanh(),
            nn.Linear(h, h), nn.Tanh(),
            nn.Linear(h, act_dim),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(act_dim))

        self.critic = nn.Sequential(
            nn.Linear(obs_dim, h), nn.Tanh(),
            nn.Linear(h, h), nn.Tanh(),
            nn.Linear(h, 1),
        )

    def get_action_and_logprob(self, obs: torch.Tensor):
        mean = self.actor_mean(obs)
        std = self.actor_logstd.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def get_logprob_and_entropy(self, obs: torch.Tensor, actions: torch.Tensor):
        mean = self.actor_mean(obs)
        std = self.actor_logstd.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------


def _do_eval(
    env_id: str,
    policy: nn.Module,
    is_discrete: bool,
    total_steps: int,
    start_time: float,
    eval_episodes: int,
    seed: int,
    log: ExperimentLog,
) -> None:
    """Run evaluation and append to log."""
    wall_clock = time.monotonic() - start_time
    sps = total_steps / max(wall_clock, 1e-9)

    if is_discrete:
        def get_action(obs: np.ndarray) -> int:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                logits = policy.actor(obs_t)
                return int(logits.argmax(dim=-1).item())
    else:
        def get_action(obs: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                mean = policy.actor_mean(obs_t)
                return mean.squeeze(0).numpy()

    mean_ret, std_ret, mean_len = evaluate_policy_gym(
        env_id, get_action, eval_episodes, seed + 1000,
    )
    log.evaluations.append(EvalRecord(
        step=total_steps, wall_clock_s=wall_clock,
        mean_return=mean_ret, std_return=std_ret,
        ep_length=mean_len, sps=sps,
    ))
    print(
        f"  [rlox] step={total_steps:>8d}  "
        f"return={mean_ret:>8.1f} +/- {std_ret:>6.1f}  "
        f"SPS={sps:>7.0f}  wall={wall_clock:>6.1f}s"
    )


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

_ALGO_DISPATCH = {
    "PPO": _run_ppo,
    "A2C": _run_a2c,
    "SAC": _run_sac,
    "TD3": _run_td3,
    "DQN": _run_dqn,
}


def run_rlox(config_path: str, seed: int, results_dir: str) -> Path:
    """Run a single rlox experiment and return the result path."""
    cfg = load_config(config_path)
    algo_name = cfg["algorithm"]
    env_id = cfg["environment"]
    hp = cfg["hyperparameters"]
    policy_cfg = cfg.get("policy", {})
    max_steps = cfg["max_steps"]
    eval_freq = cfg["eval_freq"]
    eval_episodes = cfg["eval_episodes"]

    print(f"[rlox] {algo_name} on {env_id}, seed={seed}, max_steps={max_steps}")

    log = ExperimentLog(
        framework="rlox",
        algorithm=algo_name,
        environment=env_id,
        seed=seed,
        hyperparameters=hp,
        hardware=get_hardware_info(),
    )

    runner = _ALGO_DISPATCH[algo_name]
    start_time = time.monotonic()

    runner(env_id, hp, policy_cfg, seed, max_steps, eval_freq, eval_episodes, log)

    elapsed = time.monotonic() - start_time
    log.total_wall_clock_s = elapsed
    log.total_steps = max_steps
    log.mean_sps = max_steps / max(elapsed, 1e-9)

    try:
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        log.peak_memory_mb = rusage.ru_maxrss / (1024 * 1024)
    except Exception:
        log.peak_memory_mb = 0.0

    out_path = result_path(Path(results_dir), "rlox", algo_name, env_id, seed)
    log.save(out_path)
    print(f"[rlox] Done. Results saved to {out_path}")
    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run rlox convergence benchmark")
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    run_rlox(args.config, args.seed, args.results_dir)
