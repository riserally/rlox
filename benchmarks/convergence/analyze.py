#!/usr/bin/env python3
"""Load JSON experiment logs and produce aggregate tables, metrics, and reports.

Usage:
    python analyze.py results/          # Analyze all results in directory
    python analyze.py results/ --csv    # Also export CSV
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_results(results_dir: str | Path) -> list[dict[str, Any]]:
    """Load all JSON result files from a directory."""
    results = []
    for path in sorted(Path(results_dir).glob("*.json")):
        with open(path) as f:
            results.append(json.load(f))
    return results


def _final_return(result: dict) -> float:
    """Get the mean return from the last evaluation."""
    evals = result.get("evaluations", [])
    if not evals:
        return float("nan")
    # Average of last 3 evaluations for stability
    last_n = evals[-min(3, len(evals)):]
    return float(np.mean([e["mean_return"] for e in last_n]))


def _steps_to_threshold(result: dict, threshold: float) -> int | None:
    """Find the first step where mean_return >= threshold."""
    for e in result.get("evaluations", []):
        if e["mean_return"] >= threshold:
            return e["step"]
    return None


def _wall_clock_to_threshold(result: dict, threshold: float) -> float | None:
    """Find the wall-clock time when threshold was first reached."""
    for e in result.get("evaluations", []):
        if e["mean_return"] >= threshold:
            return e["wall_clock_s"]
    return None


def _mean_sps(result: dict) -> float:
    """Get mean SPS from training metrics."""
    return result.get("training_metrics", {}).get("mean_sps", 0.0)


def interquartile_mean(values: list[float]) -> float:
    """IQM: discard bottom 25% and top 25%, take mean."""
    arr = np.sort(np.asarray(values))
    n = len(arr)
    q1 = int(np.ceil(n * 0.25))
    q3 = int(np.floor(n * 0.75))
    if q1 >= q3:
        return float(np.mean(arr))
    return float(np.mean(arr[q1:q3]))


def bootstrap_ci(
    values: list[float],
    n_bootstrap: int = 10_000,
    ci: float = 0.95,
    stat_fn=np.mean,
) -> tuple[float, float]:
    """Bootstrap confidence interval for a statistic."""
    arr = np.asarray(values)
    rng = np.random.default_rng(42)
    stats = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(arr, size=len(arr), replace=True)
        stats[i] = stat_fn(sample)
    alpha = (1.0 - ci) / 2.0
    return float(np.percentile(stats, 100 * alpha)), float(np.percentile(stats, 100 * (1 - alpha)))


def probability_of_improvement(
    rlox_scores: list[float],
    sb3_scores: list[float],
    n_bootstrap: int = 10_000,
) -> float:
    """P(rlox > sb3) via paired bootstrap."""
    rng = np.random.default_rng(42)
    rlox_arr = np.asarray(rlox_scores)
    sb3_arr = np.asarray(sb3_scores)
    n = min(len(rlox_arr), len(sb3_arr))
    wins = 0
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        if np.mean(rlox_arr[idx]) > np.mean(sb3_arr[idx]):
            wins += 1
    return wins / n_bootstrap


# Known thresholds from configs
THRESHOLDS = {
    ("PPO", "CartPole-v1"): 475.0,
    ("A2C", "CartPole-v1"): 475.0,
    ("DQN", "CartPole-v1"): 475.0,
    ("PPO", "Acrobot-v1"): -100.0,
    ("DQN", "MountainCar-v0"): -110.0,
    ("SAC", "Pendulum-v1"): -200.0,
    ("TD3", "Pendulum-v1"): -200.0,
    ("PPO", "HalfCheetah-v4"): 3000.0,
    ("SAC", "HalfCheetah-v4"): 8000.0,
    ("TD3", "HalfCheetah-v4"): 8000.0,
    ("PPO", "Hopper-v4"): 2000.0,
    ("SAC", "Hopper-v4"): 2500.0,
    ("PPO", "Walker2d-v4"): 2500.0,
    ("SAC", "Walker2d-v4"): 3000.0,
    ("PPO", "Ant-v4"): 3000.0,
    ("SAC", "Humanoid-v4"): 4000.0,
}


def build_aggregate_table(results: list[dict]) -> pd.DataFrame:
    """Build the aggregate metrics table from the evaluation plan."""
    # Group by (algorithm, environment, framework)
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in results:
        key = (r["algorithm"], r["environment"], r["framework"])
        groups[key].append(r)

    rows = []
    for (algo, env, fw), runs in sorted(groups.items()):
        final_returns = [_final_return(r) for r in runs]
        threshold = THRESHOLDS.get((algo, env))

        steps_to_t = [_steps_to_threshold(r, threshold) for r in runs] if threshold else [None]
        wall_to_t = [_wall_clock_to_threshold(r, threshold) for r in runs] if threshold else [None]
        sps_vals = [_mean_sps(r) for r in runs]

        valid_returns = [v for v in final_returns if not np.isnan(v)]
        valid_steps = [v for v in steps_to_t if v is not None]
        valid_wall = [v for v in wall_to_t if v is not None]

        iqm_return = interquartile_mean(valid_returns) if valid_returns else float("nan")
        iqm_ci = bootstrap_ci(valid_returns, stat_fn=interquartile_mean) if len(valid_returns) >= 3 else (float("nan"), float("nan"))

        row = {
            "Algorithm": algo,
            "Environment": env,
            "Framework": fw,
            "Seeds": len(runs),
            "IQM Return": f"{iqm_return:.1f}",
            "IQM 95% CI": f"[{iqm_ci[0]:.1f}, {iqm_ci[1]:.1f}]",
            "Mean Final Return": f"{np.mean(valid_returns):.1f}" if valid_returns else "N/A",
            "Steps to T": f"{np.mean(valid_steps):.0f}" if valid_steps else "N/A",
            "Wall-clock to T (s)": f"{np.mean(valid_wall):.1f}" if valid_wall else "N/A",
            "Mean SPS": f"{np.mean(sps_vals):.0f}" if sps_vals else "N/A",
        }
        rows.append(row)

    return pd.DataFrame(rows)


def build_comparison_table(results: list[dict]) -> pd.DataFrame:
    """Build a pairwise comparison table (rlox vs SB3)."""
    # Group by (algorithm, environment)
    by_task: dict[tuple[str, str], dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        by_task[(r["algorithm"], r["environment"])][r["framework"]].append(r)

    rows = []
    for (algo, env), fw_runs in sorted(by_task.items()):
        rlox_runs = fw_runs.get("rlox", [])
        sb3_runs = fw_runs.get("sb3", [])
        if not rlox_runs or not sb3_runs:
            continue

        rlox_returns = [_final_return(r) for r in rlox_runs]
        sb3_returns = [_final_return(r) for r in sb3_runs]
        rlox_sps = [_mean_sps(r) for r in rlox_runs]
        sb3_sps = [_mean_sps(r) for r in sb3_runs]

        valid_rlox = [v for v in rlox_returns if not np.isnan(v)]
        valid_sb3 = [v for v in sb3_returns if not np.isnan(v)]

        p_better_return = probability_of_improvement(valid_rlox, valid_sb3) if valid_rlox and valid_sb3 else float("nan")

        sps_speedup = np.mean(rlox_sps) / max(np.mean(sb3_sps), 1e-9)

        threshold = THRESHOLDS.get((algo, env))
        rlox_wall = [_wall_clock_to_threshold(r, threshold) for r in rlox_runs] if threshold else []
        sb3_wall = [_wall_clock_to_threshold(r, threshold) for r in sb3_runs] if threshold else []
        valid_rlox_wall = [v for v in rlox_wall if v is not None]
        valid_sb3_wall = [v for v in sb3_wall if v is not None]
        wall_speedup = (
            np.mean(valid_sb3_wall) / max(np.mean(valid_rlox_wall), 1e-9)
            if valid_rlox_wall and valid_sb3_wall
            else float("nan")
        )

        rows.append({
            "Algorithm": algo,
            "Environment": env,
            "rlox IQM": f"{interquartile_mean(valid_rlox):.1f}" if valid_rlox else "N/A",
            "SB3 IQM": f"{interquartile_mean(valid_sb3):.1f}" if valid_sb3 else "N/A",
            "P(rlox > SB3)": f"{p_better_return:.2f}",
            "SPS Speedup": f"{sps_speedup:.2f}x",
            "Wall-clock Speedup": f"{wall_speedup:.2f}x" if not np.isnan(wall_speedup) else "N/A",
        })

    return pd.DataFrame(rows)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Analyze convergence benchmark results")
    parser.add_argument("results_dir", help="Directory with JSON result files")
    parser.add_argument("--csv", action="store_true", help="Export tables as CSV")
    args = parser.parse_args()

    results = load_results(args.results_dir)
    if not results:
        print(f"No results found in {args.results_dir}")
        return

    print(f"Loaded {len(results)} experiment results\n")

    # Aggregate table
    agg_table = build_aggregate_table(results)
    print("=" * 80)
    print("AGGREGATE METRICS TABLE")
    print("=" * 80)
    print(agg_table.to_string(index=False))
    print()

    # Comparison table
    comp_table = build_comparison_table(results)
    if not comp_table.empty:
        print("=" * 80)
        print("PAIRWISE COMPARISON: rlox vs SB3")
        print("=" * 80)
        print(comp_table.to_string(index=False))
        print()

    if args.csv:
        out_dir = Path(args.results_dir)
        agg_path = out_dir / "aggregate_table.csv"
        agg_table.to_csv(agg_path, index=False)
        print(f"Saved: {agg_path}")

        if not comp_table.empty:
            comp_path = out_dir / "comparison_table.csv"
            comp_table.to_csv(comp_path, index=False)
            print(f"Saved: {comp_path}")


if __name__ == "__main__":
    main()
