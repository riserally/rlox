"""SB3 training + evaluation harness for convergence benchmarks.

Run in a separate process to avoid import contamination with rlox.
"""

from __future__ import annotations

import resource
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO, A2C, DQN, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.noise import NormalActionNoise

from common import (
    EvalRecord,
    ExperimentLog,
    evaluate_policy_gym,
    get_hardware_info,
    load_config,
    result_path,
)


SB3_ALGO_MAP = {
    "PPO": PPO,
    "A2C": A2C,
    "DQN": DQN,
    "SAC": SAC,
    "TD3": TD3,
}


class EvalCallback(BaseCallback):
    """Periodic evaluation callback that records to ExperimentLog."""

    def __init__(
        self,
        env_id: str,
        log: ExperimentLog,
        eval_freq: int,
        eval_episodes: int,
        eval_seed: int,
        start_time: float,
        vec_normalize: VecNormalize | None = None,
    ):
        super().__init__(verbose=0)
        self.env_id = env_id
        self.log = log
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.eval_seed = eval_seed
        self.start_time = start_time
        self.vec_normalize = vec_normalize
        self._last_eval_step = 0
        self._total_eval_time = 0.0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_step >= self.eval_freq:
            self._last_eval_step = self.num_timesteps
            eval_start = time.monotonic()
            wall_clock = eval_start - self.start_time
            sps = self.num_timesteps / max(wall_clock, 1e-9)

            # Build deterministic action function from SB3 model
            model = self.model
            vn = self.vec_normalize

            def get_action(obs: np.ndarray) -> np.ndarray:
                if vn is not None:
                    obs = vn.normalize_obs(obs)
                action, _ = model.predict(obs, deterministic=True)
                return action

            mean_ret, std_ret, mean_len = evaluate_policy_gym(
                self.env_id,
                get_action,
                n_episodes=self.eval_episodes,
                seed=self.eval_seed,
            )

            self._total_eval_time += time.monotonic() - eval_start
            training_wall = wall_clock - self._total_eval_time
            training_sps = self.num_timesteps / max(training_wall, 1e-9)

            self.log.evaluations.append(
                EvalRecord(
                    step=self.num_timesteps,
                    wall_clock_s=wall_clock,
                    mean_return=mean_ret,
                    std_return=std_ret,
                    ep_length=mean_len,
                    sps=sps,
                    training_sps=training_sps,
                )
            )
            print(
                f"  [SB3] step={self.num_timesteps:>8d}  "
                f"return={mean_ret:>8.1f} +/- {std_ret:>6.1f}  "
                f"SPS={sps:>7.0f}  wall={wall_clock:>6.1f}s"
            )
        return True


def _make_sb3_on_policy(
    algo_cls,
    env_id: str,
    hp: dict[str, Any],
    policy_cfg: dict[str, Any],
    seed: int,
) -> Any:
    """Create an on-policy SB3 model (PPO or A2C)."""
    n_envs = hp.get("n_envs", 1)

    def make_env(rank: int):
        def _init():
            env = gym.make(env_id)
            env.reset(seed=seed + rank)
            return env
        return _init

    # Use DummyVecEnv (sequential) to match rlox's SyncVectorEnv for fair SPS comparison
    vec_env = DummyVecEnv([make_env(i) for i in range(n_envs)])

    # Apply observation/reward normalization when config specifies it
    normalize_obs = hp.get("normalize_obs", False)
    normalize_reward = hp.get("normalize_reward", False)
    if normalize_obs or normalize_reward:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=normalize_obs,
            norm_reward=normalize_reward,
            gamma=hp.get("gamma", 0.99),
        )

    hidden = policy_cfg.get("hidden_sizes", [64, 64])
    policy_kwargs = {"net_arch": hidden}

    kwargs: dict[str, Any] = {
        "policy": "MlpPolicy",
        "env": vec_env,
        "seed": seed,
        "verbose": 0,
        "policy_kwargs": policy_kwargs,
        "learning_rate": hp.get("learning_rate", 3e-4),
        "gamma": hp.get("gamma", 0.99),
        "gae_lambda": hp.get("gae_lambda", 0.95),
        "ent_coef": hp.get("ent_coef", 0.0),
        "vf_coef": hp.get("vf_coef", 0.5),
        "max_grad_norm": hp.get("max_grad_norm", 0.5),
        "n_steps": hp.get("n_steps", 2048),
    }

    if algo_cls is PPO:
        kwargs["batch_size"] = hp.get("batch_size", 64)
        kwargs["n_epochs"] = hp.get("n_epochs", 10)
        kwargs["clip_range"] = hp.get("clip_range", 0.2)

    return algo_cls(**kwargs)


def _make_sb3_off_policy(
    algo_cls,
    env_id: str,
    hp: dict[str, Any],
    policy_cfg: dict[str, Any],
    seed: int,
) -> Any:
    """Create an off-policy SB3 model (SAC, TD3, DQN)."""
    env = gym.make(env_id)
    env.reset(seed=seed)

    hidden = policy_cfg.get("hidden_sizes", [256, 256])
    policy_kwargs = {"net_arch": hidden}

    kwargs: dict[str, Any] = {
        "policy": "MlpPolicy",
        "env": env,
        "seed": seed,
        "verbose": 0,
        "policy_kwargs": policy_kwargs,
        "learning_rate": hp.get("learning_rate", 3e-4),
        "gamma": hp.get("gamma", 0.99),
        "buffer_size": hp.get("buffer_size", 1_000_000),
        "batch_size": hp.get("batch_size", 256),
        "learning_starts": hp.get("learning_starts", 1000),
        "train_freq": hp.get("train_freq", 1),
        "gradient_steps": hp.get("gradient_steps", 1),
    }

    if algo_cls is SAC:
        kwargs["tau"] = hp.get("tau", 0.005)
        ent = hp.get("ent_coef", "auto")
        kwargs["ent_coef"] = ent if ent != "auto" else "auto"
    elif algo_cls is TD3:
        kwargs["tau"] = hp.get("tau", 0.005)
        kwargs["policy_delay"] = hp.get("policy_delay", 2)
        kwargs["target_policy_noise"] = hp.get("target_policy_noise", 0.2)
        kwargs["target_noise_clip"] = hp.get("target_noise_clip", 0.5)
        # Action noise for exploration
        act_dim = env.action_space.shape[0]
        noise_std = hp.get("exploration_noise", 0.1)
        kwargs["action_noise"] = NormalActionNoise(
            mean=np.zeros(act_dim),
            sigma=noise_std * np.ones(act_dim),
        )
    elif algo_cls is DQN:
        kwargs["target_update_interval"] = hp.get("target_update_interval", 10_000)
        kwargs["exploration_fraction"] = hp.get("exploration_fraction", 0.1)
        kwargs["exploration_final_eps"] = hp.get("exploration_final_eps", 0.05)

    return algo_cls(**kwargs)


def run_sb3(config_path: str, seed: int, results_dir: str) -> Path:
    """Run a single SB3 experiment and return the result path."""
    cfg = load_config(config_path)
    algo_name = cfg["algorithm"]
    env_id = cfg["environment"]
    hp = cfg["hyperparameters"]
    policy_cfg = cfg.get("policy", {})
    max_steps = cfg["max_steps"]
    eval_freq = cfg["eval_freq"]
    eval_episodes = cfg["eval_episodes"]

    algo_cls = SB3_ALGO_MAP[algo_name]
    is_on_policy = algo_name in ("PPO", "A2C")

    print(f"[SB3] {algo_name} on {env_id}, seed={seed}, max_steps={max_steps}")

    # Create model
    if is_on_policy:
        model = _make_sb3_on_policy(algo_cls, env_id, hp, policy_cfg, seed)
    else:
        model = _make_sb3_off_policy(algo_cls, env_id, hp, policy_cfg, seed)

    # Set up logging
    log = ExperimentLog(
        framework="sb3",
        algorithm=algo_name,
        environment=env_id,
        seed=seed,
        hyperparameters=hp,
        hardware=get_hardware_info(),
    )

    # Check if the model's env is wrapped in VecNormalize
    vec_normalize = None
    env_wrapper = model.get_env()
    if isinstance(env_wrapper, VecNormalize):
        vec_normalize = env_wrapper

    start_time = time.monotonic()
    eval_cb = EvalCallback(
        env_id=env_id,
        log=log,
        eval_freq=eval_freq,
        eval_episodes=eval_episodes,
        eval_seed=seed + 1000,
        start_time=start_time,
        vec_normalize=vec_normalize,
    )

    # Train
    model.learn(total_timesteps=max_steps, callback=eval_cb)

    elapsed = time.monotonic() - start_time
    log.total_wall_clock_s = elapsed
    log.total_steps = max_steps
    log.mean_sps = max_steps / max(elapsed, 1e-9)

    # Peak memory (macOS/Linux)
    try:
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        log.peak_memory_mb = rusage.ru_maxrss / (1024 * 1024)  # bytes -> MB on macOS
    except Exception:
        log.peak_memory_mb = 0.0

    out_path = result_path(Path(results_dir), "sb3", algo_name, env_id, seed)
    log.save(out_path)
    print(f"[SB3] Done. Results saved to {out_path}")
    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run SB3 convergence benchmark")
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    run_sb3(args.config, args.seed, args.results_dir)
