#!/usr/bin/env python3
"""Ablation: marginal contribution of each Rust component to end-to-end SPS.

Configs:
  A = full rlox (Rust envs + Rust buffer + Rust GAE)
  B = Python envs (GymVecEnv) + Rust buffer + Rust GAE
  C = Rust envs + Python buffer path + Rust GAE  [approximated]
  D = Rust envs + Rust buffer + Python GAE (NumPy loop)
  E = full Python (SB3 DummyVecEnv + Python buffer + Python GAE)

Usage:
    python benchmarks/ablation_component.py --env CartPole-v1 --seeds 3 --steps 100000
    python benchmarks/ablation_component.py --env HalfCheetah-v4 --seeds 3 --steps 100000
    python benchmarks/ablation_component.py --env Ant-v4 --seeds 3 --steps 100000
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Python GAE reference (replaces Rust compute_gae_batched)
# ---------------------------------------------------------------------------

def _python_gae(rewards, values, dones, last_values, n_steps, gamma, lam):
    """Pure-Python GAE matching the Rust compute_gae_batched layout.

    Inputs are flat env-major arrays of length n_envs * n_steps.
    last_values has length n_envs.
    Returns (advantages, returns) each of length n_envs * n_steps.
    """
    n_envs = len(last_values)
    adv = np.zeros_like(rewards)
    ret = np.zeros_like(rewards)
    for e in range(n_envs):
        off = e * n_steps
        last_gae = 0.0
        for t in reversed(range(n_steps)):
            idx = off + t
            nt = 1.0 - dones[idx]
            nv = last_values[e] if t == n_steps - 1 else values[idx + 1]
            delta = rewards[idx] + gamma * nv * nt - values[idx]
            last_gae = delta + gamma * lam * nt * last_gae
            adv[idx] = last_gae
            ret[idx] = last_gae + values[idx]
    return adv, ret


# ---------------------------------------------------------------------------
# Collector variant with Python GAE (Config D)
# ---------------------------------------------------------------------------

class PythonGAECollector:
    """RolloutCollector that uses Python GAE instead of Rust."""

    def __init__(self, base_collector):
        self._base = base_collector

    def collect(self, policy, n_steps):
        import rlox as _rlox_mod

        # Temporarily replace the Rust GAE call
        orig_fn = _rlox_mod.compute_gae_batched

        def _py_gae_wrapper(*, rewards, values, dones, last_values, n_steps,
                            gamma, lam):
            adv, ret = _python_gae(rewards, values, dones, last_values,
                                   n_steps, gamma, lam)
            return adv, ret

        _rlox_mod.compute_gae_batched = _py_gae_wrapper
        try:
            batch = self._base.collect(policy, n_steps)
        finally:
            _rlox_mod.compute_gae_batched = orig_fn
        return batch

    @property
    def _obs(self):
        return self._base._obs

    @_obs.setter
    def _obs(self, v):
        self._base._obs = v


# ---------------------------------------------------------------------------
# Training loop (measures SPS only, no convergence)
# ---------------------------------------------------------------------------

def measure_sps(algo_name, env_id, config_label, seed, total_steps, n_envs,
                n_steps):
    """Train PPO and return SPS."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    if config_label == "E":
        return _measure_sps_sb3(env_id, seed, total_steps, n_envs, n_steps)

    from rlox.algorithms.ppo import PPO

    # Build PPO with the right env/collector setup
    use_rust_env = config_label in ("A", "C", "D")
    use_python_gae = config_label == "D"

    if use_rust_env:
        # Let PPO build its default (Rust VecEnv or GymVecEnv based on env_id)
        ppo = PPO(env_id=env_id, n_envs=n_envs, seed=seed,
                  n_steps=n_steps, n_epochs=4, batch_size=64,
                  learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
                  clip_eps=0.2, ent_coef=0.0, vf_coef=0.5)
    else:
        # Config B: force Python envs
        from rlox.gym_vec_env import GymVecEnv
        from rlox.collectors import RolloutCollector

        gym_env = GymVecEnv(env_id, n_envs=n_envs, seed=seed)
        ppo = PPO(env_id=env_id, n_envs=n_envs, seed=seed,
                  n_steps=n_steps, n_epochs=4, batch_size=64,
                  learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
                  clip_eps=0.2, ent_coef=0.0, vf_coef=0.5)
        # Override the collector's env with the Python one
        ppo.collector = RolloutCollector(
            env_id=env_id, n_envs=n_envs, seed=seed,
            gamma=0.99, gae_lambda=0.95, env=gym_env,
        )

    if use_python_gae:
        ppo.collector = PythonGAECollector(ppo.collector)

    t0 = time.monotonic()
    ppo.train(total_timesteps=total_steps)
    elapsed = time.monotonic() - t0

    return total_steps / max(elapsed, 1e-9)


def _measure_sps_sb3(env_id, seed, total_steps, n_envs, n_steps):
    """Config E: full Python via SB3."""
    import gymnasium as gym
    from stable_baselines3 import PPO as SB3PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    def make_env(rank):
        def _f():
            e = gym.make(env_id)
            e.reset(seed=seed + rank)
            return e
        return _f

    venv = DummyVecEnv([make_env(i) for i in range(n_envs)])
    model = SB3PPO("MlpPolicy", venv, seed=seed, verbose=0,
                   n_steps=n_steps, n_epochs=4, batch_size=64,
                   learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
                   clip_range=0.2, ent_coef=0.0, vf_coef=0.5,
                   device="cpu")

    t0 = time.monotonic()
    model.learn(total_timesteps=total_steps)
    elapsed = time.monotonic() - t0
    venv.close()

    return total_steps / max(elapsed, 1e-9)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Component ablation study")
    parser.add_argument("--env", default="CartPole-v1")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--output", default="results/ablation")
    parser.add_argument("--configs", default="A,B,D,E",
                        help="Comma-separated configs to run (A,B,C,D,E)")
    args = parser.parse_args()

    configs = [c.strip() for c in args.configs.split(",")]
    config_names = {
        "A": "full rlox",
        "B": "Python envs",
        "C": "Python buffer",
        "D": "Python GAE",
        "E": "full Python (SB3)",
    }

    print(f"Component ablation: {args.env}, {args.steps} steps, "
          f"{args.seeds} seeds, configs={configs}")
    print("=" * 70)

    results = []
    for cfg in configs:
        sps_values = []
        for i in range(args.seeds):
            seed = i * 1000 + 42
            print(f"  Config {cfg} ({config_names[cfg]:20s}) seed={seed}...",
                  end=" ", flush=True)
            sps = measure_sps("ppo", args.env, cfg, seed, args.steps,
                              args.n_envs, args.n_steps)
            sps_values.append(sps)
            print(f"SPS={sps:.0f}")

        mean_sps = float(np.mean(sps_values))
        std_sps = float(np.std(sps_values))
        results.append({
            "config": cfg,
            "config_name": config_names[cfg],
            "env": args.env,
            "n_seeds": args.seeds,
            "mean_sps": mean_sps,
            "std_sps": std_sps,
            "per_seed_sps": sps_values,
        })
        print(f"  → {config_names[cfg]:20s}: {mean_sps:>8.0f} ± {std_sps:.0f} SPS")
        print()

    # Summary
    print("=" * 70)
    print(f"SUMMARY — {args.env}")
    print("=" * 70)
    if len(results) >= 2:
        full = next((r for r in results if r["config"] == "A"), results[0])
        base = next((r for r in results if r["config"] == "E"), results[-1])
        total_gain = full["mean_sps"] - base["mean_sps"]
        print(f"  Full rlox:   {full['mean_sps']:>8.0f} SPS")
        print(f"  Full Python: {base['mean_sps']:>8.0f} SPS")
        print(f"  Total gain:  {total_gain:>8.0f} SPS "
              f"({full['mean_sps']/base['mean_sps']:.1f}x)")
        print()
        for r in results:
            if r["config"] not in ("A", "E"):
                marginal = full["mean_sps"] - r["mean_sps"]
                frac = marginal / max(total_gain, 1) * 100
                print(f"  Without {r['config_name']:15s}: "
                      f"{r['mean_sps']:>8.0f} SPS → "
                      f"marginal = {marginal:>6.0f} SPS ({frac:.0f}%)")

    # Save
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    fname = f"ablation_{args.env.replace('/', '_')}.json"
    with (out / fname).open("w") as f:
        json.dump({"env": args.env, "results": results}, f, indent=2)
    print(f"\nSaved to {out / fname}")


if __name__ == "__main__":
    main()
