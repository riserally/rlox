#!/usr/bin/env python3
"""Main entry point for convergence benchmarks.

Usage:
    # Single experiment
    python run_experiment.py configs/ppo_cartpole.yaml --seed 0

    # All seeds for one config
    python run_experiment.py configs/ppo_cartpole.yaml --seeds 0-9

    # All Classic Control experiments (Phase E1)
    python run_experiment.py --phase e1

    # All MuJoCo Core experiments (Phase E2)
    python run_experiment.py --phase e2

    # All experiments
    python run_experiment.py --phase all

    # Only one framework
    python run_experiment.py configs/ppo_cartpole.yaml --framework rlox --seed 0

    # Warmup run (discarded, stabilizes JIT/caching)
    python run_experiment.py configs/ppo_cartpole.yaml --warmup
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

CONFIGS_DIR = Path(__file__).parent / "configs"
RESULTS_DIR = Path(__file__).parent / "results"

# Phase groupings per the evaluation plan
PHASE_E1_CONFIGS = [
    "ppo_cartpole.yaml",
    "a2c_cartpole.yaml",
    "dqn_cartpole.yaml",
    "ppo_acrobot.yaml",
    "dqn_mountaincar.yaml",
    "sac_pendulum.yaml",
    "td3_pendulum.yaml",
]

PHASE_E2_CONFIGS = [
    "ppo_halfcheetah.yaml",
    "sac_halfcheetah.yaml",
    "td3_halfcheetah.yaml",
    "ppo_hopper.yaml",
    "sac_hopper.yaml",
    "ppo_walker2d.yaml",
    "sac_walker2d.yaml",
]

PHASE_E3_CONFIGS = [
    "ppo_ant.yaml",
    "sac_humanoid.yaml",
]


def run_single(
    config: str,
    seed: int,
    framework: str,
    results_dir: str,
) -> None:
    """Run a single (config, seed, framework) experiment as a subprocess.

    Using subprocess ensures import isolation between SB3 and rlox runners.
    """
    script_dir = Path(__file__).parent

    if framework == "rlox":
        runner = script_dir / "rlox_runner.py"
    else:
        runner = script_dir / "sb3_runner.py"

    cmd = [
        sys.executable,
        str(runner),
        str(config),
        "--seed", str(seed),
        "--results-dir", results_dir,
    ]

    print(f"\n{'=' * 60}")
    print(f"Running: {framework} | {Path(config).stem} | seed={seed}")
    print(f"{'=' * 60}")

    result = subprocess.run(cmd, cwd=str(script_dir))
    if result.returncode != 0:
        print(f"ERROR: {framework} run failed with code {result.returncode}")


def parse_seeds(seeds_str: str) -> list[int]:
    """Parse seed specification like '0-9' or '0,1,2,5'."""
    if "-" in seeds_str and "," not in seeds_str:
        start, end = seeds_str.split("-")
        return list(range(int(start), int(end) + 1))
    return [int(s.strip()) for s in seeds_str.split(",")]


def main():
    parser = argparse.ArgumentParser(
        description="Run convergence benchmarks: rlox vs SB3"
    )
    parser.add_argument(
        "config",
        nargs="?",
        help="Path to YAML config file (optional if --phase is set)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Single seed")
    parser.add_argument("--seeds", type=str, default=None, help="Seed range, e.g. '0-9' or '0,1,5'")
    parser.add_argument(
        "--framework",
        choices=["rlox", "sb3", "both"],
        default="both",
        help="Which framework(s) to run",
    )
    parser.add_argument(
        "--phase",
        choices=["e1", "e2", "e3", "all"],
        default=None,
        help="Run all configs in a phase",
    )
    parser.add_argument(
        "--results-dir",
        default=str(RESULTS_DIR),
        help="Directory for JSON result files",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Run a single warmup (seed=-1, discarded)",
    )
    args = parser.parse_args()

    # Determine seeds
    if args.warmup:
        seeds = [-1]
    elif args.seeds is not None:
        seeds = parse_seeds(args.seeds)
    elif args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = [0]

    # Determine configs
    if args.phase is not None:
        phase_map = {
            "e1": PHASE_E1_CONFIGS,
            "e2": PHASE_E2_CONFIGS,
            "e3": PHASE_E3_CONFIGS,
            "all": PHASE_E1_CONFIGS + PHASE_E2_CONFIGS + PHASE_E3_CONFIGS,
        }
        configs = [str(CONFIGS_DIR / c) for c in phase_map[args.phase]]
    elif args.config is not None:
        configs = [args.config]
    else:
        parser.error("Provide either a config file or --phase")
        return

    # Determine frameworks
    frameworks = ["rlox", "sb3"] if args.framework == "both" else [args.framework]

    # Run experiments
    total = len(configs) * len(seeds) * len(frameworks)
    print(f"Running {total} experiments: {len(configs)} configs x {len(seeds)} seeds x {len(frameworks)} frameworks")

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    completed = 0
    for config in configs:
        for seed in seeds:
            for framework in frameworks:
                run_single(config, seed, framework, args.results_dir)
                completed += 1
                print(f"\nProgress: {completed}/{total}")

    if args.warmup:
        print("\nWarmup complete. Results discarded.")
    else:
        print(f"\nAll {total} experiments complete. Results in: {args.results_dir}")


if __name__ == "__main__":
    main()
