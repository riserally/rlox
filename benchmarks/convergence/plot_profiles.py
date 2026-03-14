#!/usr/bin/env python3
"""Performance profiles and probability of improvement plots (Agarwal et al., 2021).

Generates:
1. Performance profile: P(score >= tau * reference) for tau in [0, 1]
2. Probability of improvement bar chart per task
3. SPS comparison bar chart

Usage:
    python plot_profiles.py results/
    python plot_profiles.py results/ --output figures/
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.1)

FRAMEWORK_COLORS = {"rlox": "#E63946", "sb3": "#457B9D"}
FRAMEWORK_LABELS = {"rlox": "rlox (Rust)", "sb3": "Stable-Baselines3"}

# Reference scores for normalization: (random_score, expert_score)
# random: typical random policy performance
# expert: near-optimal / published strong baseline
REFERENCE_SCORES: dict[str, tuple[float, float]] = {
    "CartPole-v1": (20.0, 500.0),
    "Acrobot-v1": (-500.0, -80.0),
    "MountainCar-v0": (-200.0, -100.0),
    "Pendulum-v1": (-1200.0, -150.0),
    "HalfCheetah-v4": (-300.0, 10000.0),
    "Hopper-v4": (0.0, 3500.0),
    "Walker2d-v4": (0.0, 5000.0),
    "Ant-v4": (-100.0, 6000.0),
    "Humanoid-v4": (0.0, 8000.0),
}


def load_results(results_dir: Path) -> list[dict[str, Any]]:
    results = []
    for p in sorted(results_dir.glob("*.json")):
        with open(p) as f:
            results.append(json.load(f))
    return results


def _final_return(result: dict) -> float:
    evals = result.get("evaluations", [])
    if not evals:
        return float("nan")
    last_n = evals[-min(3, len(evals)):]
    return float(np.mean([e["mean_return"] for e in last_n]))


def _normalize_score(score: float, env: str) -> float:
    """Normalize score to [0, 1] range using reference scores."""
    if env not in REFERENCE_SCORES:
        return score  # Can't normalize, return raw
    random_s, expert_s = REFERENCE_SCORES[env]
    if expert_s == random_s:
        return 1.0 if score >= expert_s else 0.0
    return (score - random_s) / (expert_s - random_s)


def plot_performance_profile(results: list[dict], output_dir: Path) -> None:
    """Plot aggregate performance profiles across all environments."""
    # Collect normalized scores per framework
    fw_scores: dict[str, list[float]] = defaultdict(list)

    for r in results:
        score = _final_return(r)
        if np.isnan(score):
            continue
        norm = _normalize_score(score, r["environment"])
        fw_scores[r["framework"]].append(norm)

    if not fw_scores:
        print("No valid scores for performance profile.")
        return

    taus = np.linspace(0, 1.5, 200)

    fig, ax = plt.subplots(figsize=(8, 6))

    for fw, scores in sorted(fw_scores.items()):
        arr = np.asarray(scores)
        fractions = [float(np.mean(arr >= t)) for t in taus]
        color = FRAMEWORK_COLORS.get(fw, "#666666")
        label = FRAMEWORK_LABELS.get(fw, fw)
        ax.plot(taus, fractions, color=color, label=label, linewidth=2.5)

    ax.set_xlabel("Normalized Score Threshold (τ)")
    ax.set_ylabel("Fraction of Runs ≥ τ")
    ax.set_title("Performance Profile (Agarwal et al., 2021)", fontweight="bold")
    ax.legend(loc="upper right")
    ax.set_xlim(0, 1.5)
    ax.set_ylim(0, 1.05)

    out_path = output_dir / "performance_profile.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_probability_of_improvement(results: list[dict], output_dir: Path) -> None:
    """Bar chart: P(rlox > SB3) per (algo, env) task."""
    by_task: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for r in results:
        score = _final_return(r)
        if not np.isnan(score):
            by_task[(r["algorithm"], r["environment"])][r["framework"]].append(score)

    tasks = []
    probs = []
    ci_lower = []
    ci_upper = []

    rng = np.random.default_rng(42)

    for (algo, env), fw_scores in sorted(by_task.items()):
        rlox_s = fw_scores.get("rlox", [])
        sb3_s = fw_scores.get("sb3", [])
        if not rlox_s or not sb3_s:
            continue

        rlox_arr = np.asarray(rlox_s)
        sb3_arr = np.asarray(sb3_s)
        n = min(len(rlox_arr), len(sb3_arr))

        # Bootstrap P(rlox > sb3)
        n_bootstrap = 10_000
        wins = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            wins[b] = float(np.mean(rlox_arr[idx]) > np.mean(sb3_arr[idx]))

        p = float(np.mean(wins))
        lo = float(np.percentile(wins, 2.5))
        hi = float(np.percentile(wins, 97.5))

        tasks.append(f"{algo}\n{env}")
        probs.append(p)
        ci_lower.append(p - lo)
        ci_upper.append(hi - p)

    if not tasks:
        print("No paired comparisons available.")
        return

    fig, ax = plt.subplots(figsize=(max(8, len(tasks) * 1.2), 6))

    x = np.arange(len(tasks))
    colors = ["#E63946" if p > 0.5 else "#457B9D" for p in probs]

    ax.bar(x, probs, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.errorbar(x, probs, yerr=[ci_lower, ci_upper], fmt="none", color="black", capsize=4)
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, label="P = 0.5")

    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=9)
    ax.set_ylabel("P(rlox > SB3)")
    ax.set_title("Probability of Improvement", fontweight="bold")
    ax.set_ylim(0, 1)
    ax.legend()

    fig.tight_layout()
    out_path = output_dir / "probability_of_improvement.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_sps_comparison(results: list[dict], output_dir: Path) -> None:
    """Grouped bar chart comparing SPS between frameworks."""
    by_task: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for r in results:
        sps = r.get("training_metrics", {}).get("mean_sps", 0.0)
        if sps > 0:
            by_task[(r["algorithm"], r["environment"])][r["framework"]].append(sps)

    tasks = []
    rlox_sps = []
    sb3_sps = []

    for (algo, env), fw_sps in sorted(by_task.items()):
        if "rlox" in fw_sps and "sb3" in fw_sps:
            tasks.append(f"{algo}\n{env}")
            rlox_sps.append(np.mean(fw_sps["rlox"]))
            sb3_sps.append(np.mean(fw_sps["sb3"]))

    if not tasks:
        print("No SPS data for comparison.")
        return

    fig, ax = plt.subplots(figsize=(max(8, len(tasks) * 1.5), 6))

    x = np.arange(len(tasks))
    width = 0.35

    ax.bar(x - width / 2, rlox_sps, width, label=FRAMEWORK_LABELS["rlox"],
           color=FRAMEWORK_COLORS["rlox"], alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, sb3_sps, width, label=FRAMEWORK_LABELS["sb3"],
           color=FRAMEWORK_COLORS["sb3"], alpha=0.8, edgecolor="black", linewidth=0.5)

    # Add speedup annotations
    for i in range(len(tasks)):
        speedup = rlox_sps[i] / max(sb3_sps[i], 1e-9)
        ax.annotate(
            f"{speedup:.1f}x",
            xy=(x[i], max(rlox_sps[i], sb3_sps[i])),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=9)
    ax.set_ylabel("Steps Per Second (SPS)")
    ax.set_title("Training Throughput Comparison", fontweight="bold")
    ax.legend()

    fig.tight_layout()
    out_path = output_dir / "sps_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Plot performance profiles")
    parser.add_argument("results_dir", help="Directory with JSON result files")
    parser.add_argument("--output", default=None, help="Output directory for figures")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output) if args.output else results_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(results_dir)
    if not results:
        print("No results found.")
        return

    print(f"Loaded {len(results)} results\n")

    plot_performance_profile(results, output_dir)
    plot_probability_of_improvement(results, output_dir)
    plot_sps_comparison(results, output_dir)

    print(f"\nAll figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
