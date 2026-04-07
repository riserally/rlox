#!/usr/bin/env python3
"""Aggregate convergence experiment results across runs (v5/v6/v7/v8) and
print a comparison table per (algo, env) showing rlox vs SB3 reference.

Each result file is expected to contain at least:
    {"algo": ..., "env": ..., "mean_reward": ..., "total_timesteps": ...}
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOTS = {
    "v5": Path("/Users/wojciechkowalinski/Sync/work/rlox-workspace/rlox/results/convergence-v5-final"),
    "v6": Path("/Users/wojciechkowalinski/Sync/work/rlox-workspace/rlox-priv/results/convergence-v6-final/convergence"),
    "v7": Path("/Users/wojciechkowalinski/Sync/work/rlox-workspace/rlox-priv/results/v7"),
    "v8": Path("/tmp/rlox-results/v8"),
}

# SB3 reference numbers from rl-baselines3-zoo (1M steps).
SB3_REF = {
    ("PPO", "CartPole-v1"): 500.0,
    ("PPO", "Acrobot-v1"): -75.0,
    ("PPO", "Hopper-v4"): 3578.0,
    ("PPO", "HalfCheetah-v4"): 5819.0,
    ("PPO", "Walker2d-v4"): 4226.0,
    ("PPO", "Ant-v4"): 2865.0,
    ("SAC", "Pendulum-v1"): -150.0,
    ("SAC", "Hopper-v4"): 3470.0,
    ("SAC", "HalfCheetah-v4"): 9656.0,
    ("SAC", "Walker2d-v4"): 4502.0,
    ("SAC", "Humanoid-v4"): 6251.0,
    ("TD3", "Pendulum-v1"): -150.0,
    ("TD3", "HalfCheetah-v4"): 9709.0,
    ("DQN", "CartPole-v1"): 500.0,
    ("DQN", "MountainCar-v0"): -110.0,
    ("A2C", "CartPole-v1"): 500.0,
}


def load_run(path: Path) -> dict:
    with path.open() as f:
        d = json.load(f)
    name = path.stem  # e.g. rlox_PPO_Hopper-v4_seed0
    parts = name.split("_")
    src = parts[0]                   # rlox | sb3
    algo = d.get("algorithm") or d.get("algo") or parts[1]
    env = d.get("environment") or d.get("env") or "_".join(parts[2:-1])

    # Final reward = last evaluation step's mean_return
    reward = None
    final_step = None
    evals = d.get("evaluations") or []
    if evals:
        last = evals[-1]
        reward = last.get("mean_return")
        final_step = last.get("step")
    if reward is None:
        reward = d.get("mean_reward") or d.get("final_reward")
    return {
        "src": src, "algo": algo, "env": env, "reward": reward,
        "step": final_step, "sps": d.get("sps"), "path": path,
    }


def collect() -> dict[str, list[dict]]:
    runs: dict[str, list[dict]] = defaultdict(list)
    for tag, root in ROOTS.items():
        if not root.exists():
            continue
        for p in sorted(root.glob("*.json")):
            try:
                runs[tag].append(load_run(p))
            except Exception as exc:  # noqa: BLE001
                print(f"  ! failed {p}: {exc}")
    return runs


def fmt(x):
    return "    n/a" if x is None else f"{x:8.1f}"


def main():
    runs = collect()

    # Latest rlox per (algo, env) — prefer v8 > v7 > v6 > v5
    priority = ["v8", "v7", "v6", "v5"]
    latest_rlox: dict[tuple, dict] = {}
    sb3: dict[tuple, dict] = {}
    for tag in priority:
        for r in runs.get(tag, []):
            key = (r["algo"], r["env"])
            if r["src"] == "rlox" and key not in latest_rlox:
                latest_rlox[key] = {**r, "tag": tag}
            elif r["src"] == "sb3" and key not in sb3:
                sb3[key] = {**r, "tag": tag}

    print("=" * 88)
    print(f"{'Algo':<6}{'Env':<18}{'rlox':>10}{'src':>5} {'sb3-local':>11}{'src':>5}  {'sb3-zoo':>10}  {'%zoo':>7}")
    print("=" * 88)
    keys = sorted(set(latest_rlox) | set(sb3) | set(SB3_REF.keys()))
    rows = []
    for k in keys:
        rl = latest_rlox.get(k)
        sb = sb3.get(k)
        ref = SB3_REF.get(k)
        rl_r = rl["reward"] if rl else None
        sb_r = sb["reward"] if sb else None
        pct = (rl_r / ref * 100) if (rl_r is not None and ref) else None
        algo, env = k
        print(
            f"{algo:<6}{env:<18}"
            f"{fmt(rl_r)}{(rl['tag'] if rl else '-'):>5}"
            f" {fmt(sb_r)}{(sb['tag'] if sb else '-'):>5}"
            f"  {fmt(ref)}  {('  n/a' if pct is None else f'{pct:6.0f}%'):>7}"
        )
        rows.append((algo, env, rl_r, sb_r, ref, pct))
    print("=" * 88)

    # Gaps worth flagging
    print("\nGAPS vs SB3-zoo reference:")
    for algo, env, rl_r, _, ref, pct in rows:
        if rl_r is None or ref is None:
            continue
        if pct < 80:
            print(f"  ✗ {algo:<5}{env:<18}  rlox={rl_r:8.1f}  ref={ref:8.1f}  ({pct:5.0f}%)")
        elif pct < 95:
            print(f"  ~ {algo:<5}{env:<18}  rlox={rl_r:8.1f}  ref={ref:8.1f}  ({pct:5.0f}%)")
        else:
            print(f"  ✓ {algo:<5}{env:<18}  rlox={rl_r:8.1f}  ref={ref:8.1f}  ({pct:5.0f}%)")


if __name__ == "__main__":
    main()
