"""Multi-seed convergence benchmark runner.

Runs each algorithm/environment pair across multiple seeds and computes
IQM with bootstrap confidence intervals using rlox.evaluation.

Usage:
    python benchmarks/multi_seed_runner.py --algo ppo --env CartPole-v1 --seeds 5
    python benchmarks/multi_seed_runner.py --config benchmarks/configs/ppo_cartpole.yaml --seeds 10
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

# Per-(algo, env) hyperparameter presets live alongside the convergence
# benchmark configs. We auto-resolve them so the launch scripts only need
# to specify --algo/--env/--timesteps.
_PRESET_DIR = Path(__file__).resolve().parent / "convergence" / "configs"


def _resolve_preset(algo: str, env_id: str) -> dict | None:
    """Look up a YAML preset for (algo, env) and return its hyperparameters.

    Convention: ``benchmarks/convergence/configs/<algo>_<env_short>.yaml``
    where ``env_short = env_id.split('-')[0].lower()``.

    Returns None if no preset exists; the caller falls back to Trainer
    defaults (which are CleanRL CartPole-tuned and unsuitable for MuJoCo).
    """
    env_short = env_id.split("-")[0].lower()
    preset = _PRESET_DIR / f"{algo.lower()}_{env_short}.yaml"
    if not preset.exists():
        return None
    import yaml

    with preset.open() as f:
        data = yaml.safe_load(f) or {}
    return data.get("hyperparameters", data)


def run_single_seed(
    algo: str,
    env_id: str,
    total_timesteps: int,
    seed: int,
    config: dict | None = None,
) -> dict:
    """Train one seed and return metrics + final eval reward.

    If ``config`` is None, attempt to auto-resolve a preset YAML for the
    (algo, env) pair from ``benchmarks/convergence/configs/``. This avoids
    silently running CleanRL CartPole-tuned defaults on MuJoCo envs.
    """
    from rlox import Trainer

    if config is None:
        config = _resolve_preset(algo, env_id)
    cfg = dict(config or {})
    trainer = Trainer(algo, env=env_id, seed=seed, config=cfg)

    t0 = time.time()
    metrics = trainer.train(total_timesteps=total_timesteps)
    wall_time = time.time() - t0

    # Evaluate
    import gymnasium as gym
    import torch

    eval_env = gym.make(env_id)
    is_discrete = hasattr(eval_env.action_space, "n")
    vn = getattr(trainer.algo, "vec_normalize", None)
    rewards = []
    for _ in range(30):
        obs, _ = eval_env.reset(seed=seed + 1000)
        ep_r = 0.0
        done = False
        while not done:
            # Normalize obs if VecNormalize was used during training
            eval_obs = obs
            if vn is not None:
                eval_obs = vn.normalize_obs(obs.reshape(1, -1)).flatten()

            obs_t = torch.as_tensor(eval_obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                if hasattr(trainer.algo, "predict"):
                    action = trainer.algo.predict(obs_t, deterministic=True)
                else:
                    logits = trainer.algo.policy.actor(obs_t)
                    if is_discrete:
                        action = logits.argmax(-1).item()
                    else:
                        action = logits.squeeze(0)

            # Ensure action is correct shape for the environment
            if hasattr(action, "numpy"):
                action = action.numpy()
            if isinstance(action, np.ndarray):
                action = action.flatten()
            elif is_discrete and hasattr(action, "item"):
                action = action.item()

            obs, r, term, trunc, _ = eval_env.step(action)
            ep_r += r
            done = term or trunc
        rewards.append(ep_r)
    eval_env.close()

    return {
        "algo": algo,
        "env": env_id,
        "seed": seed,
        "total_timesteps": total_timesteps,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "wall_time_s": wall_time,
        "sps": total_timesteps / wall_time,
        "metrics": {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
    }


def run_multi_seed(
    algo: str,
    env_id: str,
    total_timesteps: int,
    n_seeds: int = 5,
    config: dict | None = None,
    output_dir: str = "results/multi_seed",
) -> dict:
    """Run multiple seeds and compute aggregate statistics."""
    from rlox.evaluation import interquartile_mean, stratified_bootstrap_ci

    results = []
    for i in range(n_seeds):
        seed = i * 1000 + 42
        print(f"  Seed {i + 1}/{n_seeds} (seed={seed})...", end=" ", flush=True)
        result = run_single_seed(algo, env_id, total_timesteps, seed, config)
        results.append(result)
        print(f"reward={result['mean_reward']:.1f}, SPS={result['sps']:.0f}")

    rewards = [r["mean_reward"] for r in results]
    iqm = interquartile_mean(rewards)
    ci_low, ci_high = stratified_bootstrap_ci(rewards)

    summary = {
        "algo": algo,
        "env": env_id,
        "n_seeds": n_seeds,
        "total_timesteps": total_timesteps,
        "iqm": float(iqm),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "mean": float(np.mean(rewards)),
        "std": float(np.std(rewards)),
        "min": float(np.min(rewards)),
        "max": float(np.max(rewards)),
        "mean_sps": float(np.mean([r["sps"] for r in results])),
        "per_seed": results,
    }

    # Save
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fname = f"{algo}_{env_id.replace('/', '_')}_seeds{n_seeds}.json"
    with open(out / fname, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  IQM={iqm:.1f} [{ci_low:.1f}, {ci_high:.1f}] saved to {out / fname}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Multi-seed convergence benchmark")
    parser.add_argument("--algo", required=True, help="Algorithm name (ppo, sac, dqn, ...)")
    parser.add_argument("--env", required=True, help="Gymnasium environment ID")
    parser.add_argument("--timesteps", type=int, default=100_000, help="Total timesteps per seed")
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds")
    parser.add_argument("--output", default="results/multi_seed", help="Output directory")
    parser.add_argument("--config", help="YAML config file for hyperparameters")
    args = parser.parse_args()

    config = None
    if args.config:
        import yaml

        with open(args.config) as f:
            data = yaml.safe_load(f) or {}
        config = data.get("hyperparameters", data)

    print(f"Multi-seed benchmark: {args.algo} on {args.env} ({args.seeds} seeds, {args.timesteps} steps)")
    run_multi_seed(args.algo, args.env, args.timesteps, args.seeds, config, args.output)


if __name__ == "__main__":
    main()
