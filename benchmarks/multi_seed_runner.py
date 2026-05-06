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
    eval_freq: int = 0,
) -> dict:
    """Train one seed and return metrics + final eval reward.

    If ``config`` is None, attempt to auto-resolve a preset YAML for the
    (algo, env) pair from ``benchmarks/convergence/configs/``. This avoids
    silently running CleanRL CartPole-tuned defaults on MuJoCo envs.

    Parameters
    ----------
    eval_freq : int
        If > 0, evaluate every ``eval_freq`` steps during training and
        record per-checkpoint learning curve data. Default 0 (off).
    """
    from rlox import Trainer
    from rlox.callbacks import EvalCallback

    if config is None:
        config = _resolve_preset(algo, env_id)
    cfg = dict(config or {})

    callbacks = []
    eval_cb = None
    if eval_freq > 0:
        eval_cb = EvalCallback(
            eval_freq=eval_freq, n_eval_episodes=10, verbose=False
        )
        callbacks.append(eval_cb)

    trainer = Trainer(algo, env=env_id, seed=seed, config=cfg, callbacks=callbacks)

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
    for ep in range(30):
        # Unique reset seed per eval episode — otherwise 30 "independent"
        # deterministic rollouts are really one playout replayed 30x
        # (small MuJoCo noise notwithstanding). See
        # docs/plans/multi-seed-pre-flight-review-2026-04-06.md.
        obs, _ = eval_env.reset(seed=seed + 1000 + ep)
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

            # Coerce action to the shape the env expects:
            #   discrete    -> Python int scalar
            #   continuous  -> 1-D np.ndarray of shape (act_dim,)
            if hasattr(action, "detach"):
                action = action.detach()
            if hasattr(action, "cpu"):
                action = action.cpu()
            if hasattr(action, "numpy"):
                action = action.numpy()
            if is_discrete:
                if isinstance(action, np.ndarray):
                    action = int(action.reshape(-1)[0])
                elif hasattr(action, "item"):
                    action = int(action.item())
                else:
                    action = int(action)
            else:
                if isinstance(action, np.ndarray):
                    action = action.astype(np.float32).flatten()
                else:
                    action = np.asarray(action, dtype=np.float32).flatten()

            obs, r, term, trunc, _ = eval_env.step(action)
            ep_r += r
            done = term or trunc
        rewards.append(ep_r)
    eval_env.close()

    result = {
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

    # Include learning curve if eval callback was used
    if eval_cb is not None and eval_cb.eval_results:
        result["learning_curve"] = [
            {"step": step, "reward": reward}
            for step, reward in eval_cb.eval_results
        ]

    return result


def _compute_learning_curve_ci(
    results: list[dict],
    ci: float = 0.95,
    n_bootstrap: int = 2000,
) -> list[dict[str, float]]:
    """Compute bootstrap CI bands on per-step learning curves across seeds.

    Aligns learning curves by step, then bootstraps IQM at each checkpoint.

    Returns
    -------
    List of dicts with keys: step, mean, ci_low, ci_high.
    """
    # Collect all learning curves
    curves = [r["learning_curve"] for r in results if "learning_curve" in r]
    if not curves:
        return []

    # Find common steps (intersection across seeds)
    step_sets = [set(pt["step"] for pt in curve) for curve in curves]
    common_steps = sorted(set.intersection(*step_sets)) if step_sets else []
    if not common_steps:
        return []

    # Build matrix: (n_seeds, n_steps)
    seed_by_step = {}
    for curve in curves:
        for pt in curve:
            seed_by_step.setdefault(pt["step"], []).append(pt["reward"])

    rng = np.random.default_rng(42)
    alpha = (1.0 - ci) / 2.0
    ci_curve = []
    for step in common_steps:
        rewards = np.array(seed_by_step[step])
        n = len(rewards)
        boot_means = np.array([
            float(np.mean(rng.choice(rewards, size=n, replace=True)))
            for _ in range(n_bootstrap)
        ])
        ci_curve.append({
            "step": int(step),
            "mean": float(np.mean(rewards)),
            "ci_low": float(np.percentile(boot_means, 100 * alpha)),
            "ci_high": float(np.percentile(boot_means, 100 * (1.0 - alpha))),
        })

    return ci_curve


def run_multi_seed(
    algo: str,
    env_id: str,
    total_timesteps: int,
    n_seeds: int = 5,
    config: dict | None = None,
    output_dir: str = "results/multi_seed",
    eval_freq: int = 0,
) -> dict:
    """Run multiple seeds and compute aggregate statistics.

    Parameters
    ----------
    eval_freq : int
        If > 0, evaluate every ``eval_freq`` steps during training.
        Enables learning curve CI bands in the output.
    """
    from rlox.evaluation import interquartile_mean, stratified_bootstrap_ci

    results = []
    for i in range(n_seeds):
        seed = i * 1000 + 42
        print(f"  Seed {i + 1}/{n_seeds} (seed={seed})...", end=" ", flush=True)
        result = run_single_seed(algo, env_id, total_timesteps, seed, config, eval_freq)
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

    # Compute learning curve CI bands if eval data is available
    ci_curve = _compute_learning_curve_ci(results)
    if ci_curve:
        summary["learning_curve_ci"] = ci_curve

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
    parser.add_argument("--eval-freq", type=int, default=0,
                        help="Eval frequency for learning curve CI bands (0=off)")
    args = parser.parse_args()

    config = None
    if args.config:
        import yaml

        with open(args.config) as f:
            data = yaml.safe_load(f) or {}
        config = data.get("hyperparameters", data)

    print(f"Multi-seed benchmark: {args.algo} on {args.env} ({args.seeds} seeds, {args.timesteps} steps)")
    run_multi_seed(args.algo, args.env, args.timesteps, args.seeds, config, args.output, args.eval_freq)


if __name__ == "__main__":
    main()
