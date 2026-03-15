"""rlox Rust-NN convergence runner (Burn / Candle backends).

Uses rlox.ActorCritic for the full forward/backward/step pipeline in Rust,
gymnasium for env stepping, and rlox.compute_gae for advantage estimation.
No PyTorch in the training loop.

Usage:
    python benchmarks/convergence/rust_nn_runner.py configs/ppo_cartpole.yaml --backend candle
"""

from __future__ import annotations

import resource
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import gymnasium.vector
import numpy as np

import rlox

from common import (
    EvalRecord,
    ExperimentLog,
    evaluate_policy_gym,
    get_hardware_info,
    load_config,
    result_path,
)


def _collect_rollout(
    vec_env: gymnasium.vector.VectorEnv,
    ac: rlox.ActorCritic,
    obs: np.ndarray,
    n_steps: int,
    n_envs: int,
    gamma: float,
    gae_lambda: float,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Collect on-policy rollout using gymnasium VectorEnv + Rust NN + rlox GAE."""
    obs_dim = obs.shape[1]

    all_obs = []
    all_actions = []
    all_rewards = []
    all_dones = []
    all_log_probs = []
    all_values = []

    for _ in range(n_steps):
        obs_flat = obs.ravel().astype(np.float32)

        actions, log_probs = ac.act(obs_flat)
        values = ac.value(obs_flat)

        actions_int = actions.astype(np.int64)
        next_obs, rewards, terminated, truncated, infos = vec_env.step(actions_int)
        dones = terminated | truncated

        all_obs.append(obs.copy())
        all_actions.append(actions.copy())
        all_log_probs.append(log_probs.copy())
        all_values.append(values.copy())
        all_rewards.append(rewards.astype(np.float32))
        all_dones.append(dones.astype(np.float32))

        obs = next_obs

    # Bootstrap value
    last_values = ac.value(obs.ravel().astype(np.float32))

    # GAE per environment using rlox Rust core
    all_advantages = []
    all_returns = []
    for env_idx in range(n_envs):
        rewards_env = np.array([r[env_idx] for r in all_rewards])
        values_env = np.array([v[env_idx] for v in all_values])
        dones_env = np.array([d[env_idx] for d in all_dones])

        adv, ret = rlox.compute_gae(
            rewards=rewards_env.astype(np.float64),
            values=values_env.astype(np.float64),
            dones=dones_env.astype(np.float64),
            last_value=float(last_values[env_idx]),
            gamma=gamma,
            lam=gae_lambda,
        )
        all_advantages.append(adv.astype(np.float32))
        all_returns.append(ret.astype(np.float32))

    # Stack: [n_steps, n_envs, ...]
    obs_arr = np.stack(all_obs)          # [n_steps, n_envs, obs_dim]
    actions_arr = np.stack(all_actions)   # [n_steps, n_envs]
    log_probs_arr = np.stack(all_log_probs)
    values_arr = np.stack(all_values)
    advantages_arr = np.stack(all_advantages).T  # [n_steps, n_envs]
    returns_arr = np.stack(all_returns).T

    total = n_steps * n_envs
    batch = {
        "obs": obs_arr.reshape(total, obs_dim),
        "actions": actions_arr.reshape(total),
        "log_probs": log_probs_arr.reshape(total),
        "values": values_arr.reshape(total),
        "advantages": advantages_arr.reshape(total),
        "returns": returns_arr.reshape(total),
    }
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
    backend: str,
) -> None:
    """PPO training loop using Rust NN backend (no PyTorch)."""
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

    hidden = policy_cfg.get("hidden_sizes", [64, 64])[0]

    # Detect obs/action dimensions
    probe_env = gym.make(env_id)
    obs_dim = int(np.prod(probe_env.observation_space.shape))
    n_actions = int(probe_env.action_space.n)
    probe_env.close()

    ac = rlox.ActorCritic(
        backend=backend,
        obs_dim=obs_dim,
        n_actions=n_actions,
        hidden=hidden,
        lr=lr,
        seed=seed,
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
        ac.learning_rate = lr * frac

        batch, obs = _collect_rollout(
            vec_env, ac, obs, n_steps, n_envs, gamma, gae_lambda,
        )
        total_steps += steps_per_rollout

        # PPO update epochs
        total = n_steps * n_envs
        indices = np.arange(total)

        for _epoch in range(n_epochs):
            np.random.shuffle(indices)

            for start in range(0, total, batch_size):
                end = min(start + batch_size, total)
                mb_idx = indices[start:end]

                mb_obs = batch["obs"][mb_idx].ravel().astype(np.float32)
                mb_actions = batch["actions"][mb_idx].astype(np.float32)
                mb_log_probs = batch["log_probs"][mb_idx].astype(np.float32)
                mb_values = batch["values"][mb_idx].astype(np.float32)
                mb_returns = batch["returns"][mb_idx].astype(np.float32)

                mb_adv = batch["advantages"][mb_idx].astype(np.float32)
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                ac.ppo_step(
                    mb_obs, mb_actions, mb_log_probs, mb_adv,
                    mb_returns, mb_values,
                    clip_eps=clip_eps,
                    vf_coef=vf_coef,
                    ent_coef=ent_coef,
                    max_grad_norm=max_grad_norm,
                )

        # Periodic evaluation
        if total_steps - last_eval_step >= eval_freq:
            last_eval_step = total_steps
            _do_eval(env_id, ac, obs_dim, total_steps, start_time,
                     eval_episodes, seed, log)

    vec_env.close()


def _do_eval(
    env_id: str,
    ac: rlox.ActorCritic,
    obs_dim: int,
    total_steps: int,
    start_time: float,
    eval_episodes: int,
    seed: int,
    log: ExperimentLog,
) -> None:
    """Run evaluation and append to log."""
    wall_clock = time.monotonic() - start_time
    sps = total_steps / max(wall_clock, 1e-9)

    def get_action(obs: np.ndarray) -> int:
        obs_flat = obs.ravel().astype(np.float32)
        actions, _ = ac.act(obs_flat)
        return int(actions[0])

    mean_ret, std_ret, mean_len = evaluate_policy_gym(
        env_id, get_action, eval_episodes, seed + 1000,
    )
    log.evaluations.append(EvalRecord(
        step=total_steps, wall_clock_s=wall_clock,
        mean_return=mean_ret, std_return=std_ret,
        ep_length=mean_len, sps=sps,
    ))
    print(
        f"  [rlox-{log.framework}] step={total_steps:>8d}  "
        f"return={mean_ret:>8.1f} +/- {std_ret:>6.1f}  "
        f"SPS={sps:>7.0f}  wall={wall_clock:>6.1f}s"
    )


def run_rust_nn(config_path: str, seed: int, results_dir: str, backend: str = "candle") -> Path:
    """Run a single Rust-NN experiment and return the result path."""
    cfg = load_config(config_path)
    algo_name = cfg["algorithm"]
    env_id = cfg["environment"]
    hp = cfg["hyperparameters"]
    policy_cfg = cfg.get("policy", {})
    max_steps = cfg["max_steps"]
    eval_freq = cfg["eval_freq"]
    eval_episodes = cfg["eval_episodes"]

    if algo_name != "PPO":
        raise ValueError(f"Rust NN runner only supports PPO (discrete), got {algo_name}")

    framework_name = f"rlox_{backend}"
    print(f"[{framework_name}] PPO on {env_id}, seed={seed}, max_steps={max_steps}")

    log = ExperimentLog(
        framework=framework_name,
        algorithm=algo_name,
        environment=env_id,
        seed=seed,
        hyperparameters=hp,
        hardware=get_hardware_info(),
    )

    start_time = time.monotonic()
    _run_ppo(env_id, hp, policy_cfg, seed, max_steps, eval_freq, eval_episodes, log, backend)

    elapsed = time.monotonic() - start_time
    log.total_wall_clock_s = elapsed
    log.total_steps = max_steps
    log.mean_sps = max_steps / max(elapsed, 1e-9)

    try:
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        log.peak_memory_mb = rusage.ru_maxrss / (1024 * 1024)
    except Exception:
        log.peak_memory_mb = 0.0

    out_path = result_path(Path(results_dir), framework_name, algo_name, env_id, seed)
    log.save(out_path)
    print(f"[{framework_name}] Done. Results saved to {out_path}")
    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run rlox Rust-NN convergence benchmark")
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument("--backend", choices=["burn", "candle"], default="candle")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    run_rust_nn(args.config, args.seed, args.results_dir, args.backend)
