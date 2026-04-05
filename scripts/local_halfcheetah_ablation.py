#!/usr/bin/env python3
"""Local ablation: which VecNormalize component hurts HalfCheetah?

Runs 4 short experiments (200K steps each) to isolate whether obs normalization,
reward normalization, or both cause the convergence gap on HalfCheetah-v4.

Usage:
    .venv/bin/python scripts/local_halfcheetah_ablation.py

Expected wall time: ~5-10 minutes total on Apple Silicon.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ENV_ID = "HalfCheetah-v4"
TOTAL_STEPS = 200_000
SEED = 42
N_EVAL_EPISODES = 30

# Standard MuJoCo PPO hyperparams (matching SB3/rl-zoo3)
BASE_CONFIG = {
    "n_envs": 8,
    "n_steps": 2048,
    "n_epochs": 10,
    "batch_size": 64,
    "learning_rate": 3e-4,
    "ent_coef": 0.0,
    "gamma": 0.99,
    "gae_lambda": 0.95,
}

# Ablation matrix: isolate obs vs reward normalization
CONFIGS = [
    ("norm_both", {"normalize_obs": True, "normalize_rewards": True}),
    ("norm_obs_only", {"normalize_obs": True, "normalize_rewards": False}),
    ("norm_reward_only", {"normalize_obs": False, "normalize_rewards": True}),
    ("no_norm", {"normalize_obs": False, "normalize_rewards": False}),
]

# Additional test: clip_obs sensitivity (only relevant when norm_obs=True)
CLIP_OBS_CONFIGS = [
    ("clip_obs_10", {"normalize_obs": True, "normalize_rewards": True, "clip_obs": 10.0}),
    ("clip_obs_5", {"normalize_obs": True, "normalize_rewards": True, "clip_obs": 5.0}),
    ("clip_obs_20", {"normalize_obs": True, "normalize_rewards": True, "clip_obs": 20.0}),
]


def evaluate(trainer, n_episodes: int = N_EVAL_EPISODES) -> tuple[float, float]:
    """Evaluate trained policy for n_episodes, return (mean, std)."""
    eval_env = gym.make(ENV_ID)
    vn = getattr(trainer.algo, "vec_normalize", None)

    rewards = []
    for _ in range(n_episodes):
        obs, _ = eval_env.reset()
        ep_r = 0.0
        done = False
        while not done:
            if vn is not None:
                obs = vn.normalize_obs(obs.reshape(1, -1)).flatten()
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = trainer.algo.policy.actor(obs_t).squeeze(0).numpy()
            obs, r, term, trunc, _ = eval_env.step(action)
            ep_r += r
            done = term or trunc
        rewards.append(ep_r)
    eval_env.close()
    return float(np.mean(rewards)), float(np.std(rewards))


def run_experiment(name: str, config: dict) -> dict:
    """Train PPO on HalfCheetah with given config, evaluate, return results."""
    from rlox import Trainer

    full_config = {**BASE_CONFIG, **config}

    # VecNormalize clip_obs is not a PPOConfig field -- need special handling
    clip_obs = full_config.pop("clip_obs", None)

    print(f"\n{'='*60}")
    print(f"  Experiment: {name}")
    print(f"  normalize_obs={full_config.get('normalize_obs', False)}, "
          f"normalize_rewards={full_config.get('normalize_rewards', False)}")
    if clip_obs is not None:
        print(f"  clip_obs={clip_obs}")
    print(f"{'='*60}")

    t0 = time.time()
    trainer = Trainer("ppo", env=ENV_ID, seed=SEED, config=full_config)

    # Override clip_obs if specified
    if clip_obs is not None and trainer.algo.vec_normalize is not None:
        trainer.algo.vec_normalize.clip_obs = clip_obs
        print(f"  [Override] clip_obs set to {clip_obs}")

    metrics = trainer.train(total_timesteps=TOTAL_STEPS)
    elapsed = time.time() - t0
    sps = TOTAL_STEPS / elapsed

    mean_r, std_r = evaluate(trainer)
    print(f"  Result: {mean_r:.1f} +/- {std_r:.1f}  (SPS={sps:.0f}, wall={elapsed:.1f}s)")

    # Also log the obs stats if VecNormalize is active
    obs_stats = None
    vn = getattr(trainer.algo, "vec_normalize", None)
    if vn is not None and vn.norm_obs:
        stats = vn.get_obs_rms()
        obs_stats = {
            "mean_range": [float(stats["mean"].min()), float(stats["mean"].max())],
            "var_range": [float(stats["var"].min()), float(stats["var"].max())],
            "std_range": [float(np.sqrt(stats["var"]).min()), float(np.sqrt(stats["var"]).max())],
        }
        print(f"  Obs stats: mean range={obs_stats['mean_range']}, "
              f"std range={obs_stats['std_range']}")

    return {
        "name": name,
        "config": {**full_config, **({"clip_obs": clip_obs} if clip_obs else {})},
        "mean_return": mean_r,
        "std_return": std_r,
        "sps": sps,
        "wall_time_s": elapsed,
        "obs_stats": obs_stats,
    }


def main() -> None:
    print("=" * 60)
    print("  HalfCheetah VecNormalize Ablation Study")
    print(f"  {TOTAL_STEPS:,} steps per experiment, seed={SEED}")
    print("=" * 60)

    results = []

    # Phase 1: Normalization ablation
    print("\n--- Phase 1: Normalization Ablation ---")
    for name, overrides in CONFIGS:
        result = run_experiment(name, overrides)
        results.append(result)

    # Phase 2: clip_obs sensitivity (only if norm_both was not clearly worst)
    print("\n--- Phase 2: clip_obs Sensitivity ---")
    for name, overrides in CLIP_OBS_CONFIGS:
        result = run_experiment(name, overrides)
        results.append(result)

    # Summary table
    print("\n" + "=" * 70)
    print(f"  {'Experiment':<20} {'Mean Return':>12} {'Std':>8} {'SPS':>8}")
    print("-" * 70)
    for r in results:
        print(f"  {r['name']:<20} {r['mean_return']:>12.1f} {r['std_return']:>8.1f} {r['sps']:>8.0f}")
    print("=" * 70)

    # Diagnosis
    norm_results = {r["name"]: r["mean_return"] for r in results[:4]}
    best_name = max(norm_results, key=norm_results.get)
    worst_name = min(norm_results, key=norm_results.get)
    print(f"\n  Best:  {best_name} ({norm_results[best_name]:.1f})")
    print(f"  Worst: {worst_name} ({norm_results[worst_name]:.1f})")

    gap = norm_results[best_name] - norm_results[worst_name]
    print(f"  Gap:   {gap:.1f}")

    if norm_results["no_norm"] > norm_results["norm_both"] * 1.2:
        print("\n  DIAGNOSIS: VecNormalize hurts HalfCheetah.")
        if norm_results["norm_obs_only"] > norm_results["norm_both"]:
            print("  -> Reward normalization is the culprit.")
        elif norm_results["norm_reward_only"] > norm_results["norm_both"]:
            print("  -> Obs normalization is the culprit.")
        else:
            print("  -> Both components contribute to the gap.")
    elif norm_results["norm_both"] >= norm_results["no_norm"] * 0.9:
        print("\n  DIAGNOSIS: VecNormalize is NOT the issue at 200K steps.")
        print("  The gap likely emerges at longer training or is seed-dependent.")
    else:
        print("\n  DIAGNOSIS: Inconclusive at 200K steps. Consider longer runs.")

    # Save results
    out_dir = Path("results/ablation")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "halfcheetah_vecnormalize_ablation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
