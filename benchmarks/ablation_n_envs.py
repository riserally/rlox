#!/usr/bin/env python3
"""Ablation: Rayon VecEnv scaling vs n_envs.

Measures SPS for rlox (Rust VecEnv) and SB3 (Python DummyVecEnv) as
n_envs varies from 1 to 512 on CartPole-v1.

Usage:
    python benchmarks/ablation_n_envs.py --seeds 3 --steps 50000
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


def measure_sps_rlox(env_id, n_envs, seed, total_steps, n_steps=2048):
    torch.manual_seed(seed); np.random.seed(seed)
    from rlox.algorithms.ppo import PPO
    ppo = PPO(env_id=env_id, n_envs=n_envs, seed=seed,
              n_steps=min(n_steps, total_steps // max(n_envs, 1)),
              n_epochs=4, batch_size=64, learning_rate=3e-4,
              gamma=0.99, gae_lambda=0.95, clip_eps=0.2,
              ent_coef=0.0, vf_coef=0.5)
    t0 = time.monotonic()
    ppo.train(total_timesteps=total_steps)
    return total_steps / max(time.monotonic() - t0, 1e-9)


def measure_sps_sb3(env_id, n_envs, seed, total_steps, n_steps=2048):
    torch.manual_seed(seed); np.random.seed(seed)
    import gymnasium as gym
    from stable_baselines3 import PPO as SB3PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    venv = DummyVecEnv([
        (lambda r: lambda: gym.make(env_id))(i)
        for i in range(n_envs)
    ])
    model = SB3PPO("MlpPolicy", venv, seed=seed, verbose=0,
                   n_steps=min(n_steps, total_steps // max(n_envs, 1)),
                   n_epochs=4, batch_size=64, learning_rate=3e-4,
                   gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                   ent_coef=0.0, vf_coef=0.5, device="cpu")
    t0 = time.monotonic()
    model.learn(total_timesteps=total_steps)
    elapsed = time.monotonic() - t0
    venv.close()
    return total_steps / max(elapsed, 1e-9)


def main():
    parser = argparse.ArgumentParser(description="n_envs scaling ablation")
    parser.add_argument("--env", default="CartPole-v1")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--output", default="results/ablation")
    args = parser.parse_args()

    n_envs_list = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    print(f"n_envs scaling: {args.env}, {args.steps} steps, "
          f"{args.seeds} seeds")
    print("=" * 70)

    results = []
    for n_envs in n_envs_list:
        rlox_sps = []
        sb3_sps = []
        for i in range(args.seeds):
            seed = i * 1000 + 42
            print(f"  n_envs={n_envs:>4d}, seed={seed}...", end=" ", flush=True)
            r = measure_sps_rlox(args.env, n_envs, seed, args.steps)
            s = measure_sps_sb3(args.env, n_envs, seed, args.steps)
            rlox_sps.append(r)
            sb3_sps.append(s)
            print(f"rlox={r:.0f}  sb3={s:.0f}  ratio={r/s:.1f}x")

        results.append({
            "n_envs": n_envs,
            "rlox_mean_sps": float(np.mean(rlox_sps)),
            "sb3_mean_sps": float(np.mean(sb3_sps)),
            "ratio": float(np.mean(rlox_sps) / np.mean(sb3_sps)),
        })

    print()
    print("=" * 70)
    print(f"{'n_envs':>8s} {'rlox SPS':>10s} {'SB3 SPS':>10s} {'ratio':>8s}")
    for r in results:
        print(f"{r['n_envs']:>8d} {r['rlox_mean_sps']:>10.0f} "
              f"{r['sb3_mean_sps']:>10.0f} {r['ratio']:>7.1f}x")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    fname = f"ablation_n_envs_{args.env.replace('/', '_')}.json"
    with (out / fname).open("w") as f:
        json.dump({"env": args.env, "results": results}, f, indent=2)
    print(f"\nSaved to {out / fname}")


if __name__ == "__main__":
    main()
