"""Multi-seed convergence benchmark for Stable-Baselines3 in the SAME harness
as ``benchmarks/multi_seed_runner.py``.

The whole point of this file is to make rlox-vs-SB3 a same-harness comparison
rather than "rlox in our harness vs SB3 zoo single-seed point estimate". It:

- Reads the same per-(algo, env) preset YAMLs from
  ``benchmarks/convergence/configs/`` that the rlox runner reads.
- Maps rlox config keys to SB3's keyword arguments (most are identical;
  ``target_update_freq -> target_update_interval`` etc.).
- Uses the same n_envs setting (PPO/A2C only — SB3's SAC/TD3/DQN are
  single-env by default).
- Wraps with the same VecNormalize semantics when the preset asks for
  ``normalize_obs`` / ``normalize_rewards``.
- Runs **the same eval protocol**: 30 deterministic episodes per seed,
  ``env.reset(seed=base+1000+ep)`` per episode (unique seed per episode),
  ``algo.predict(obs, deterministic=True)``.
- Aggregates 5 seeds with the same IQM + bootstrap CI from
  ``rlox.evaluation``.
- Writes ``sb3_<algo>_<env>_seeds<N>.json`` next to the rlox results, in the
  same JSON shape, so the two trees can be diffed cell-by-cell.

Usage::

    python benchmarks/multi_seed_runner_sb3.py --algo ppo --env Hopper-v4 \\
        --timesteps 1000000 --seeds 5 --output results/multi-seed
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np


# Reuse the same preset resolver the rlox runner uses, so the two paths
# pull configuration from a single source of truth.
_PRESET_DIR = Path(__file__).resolve().parent / "convergence" / "configs"


# SB3-specific overrides — applied on top of the shared preset.
#
# The shared preset YAMLs are tuned for rlox's train-every-step DQN. SB3
# expects an explicit ``train_freq`` and ``gradient_steps`` setting, and
# without them SB3 DQN with ``target_update_interval=10`` (the SB3-zoo
# CartPole recipe) collapses (~9 reward at 50k). These overrides add the
# missing keys from rl-baselines3-zoo so SB3 has its best shot in the
# comparison.
#
# Convention: same path as preset YAMLs, with an ``sb3_`` prefix on the
# filename.
_SB3_OVERRIDES = {
    ("dqn", "CartPole-v1"): {
        # rl-baselines3-zoo/hyperparams/dqn.yml — CartPole-v1
        "train_freq": 256,
        "gradient_steps": 128,
    },
    ("dqn", "MountainCar-v0"): {
        # rl-baselines3-zoo/hyperparams/dqn.yml — MountainCar-v0
        "train_freq": 16,
        "gradient_steps": 8,
    },
}


def _resolve_preset(algo: str, env_id: str) -> dict | None:
    env_short = env_id.split("-")[0].lower()
    preset = _PRESET_DIR / f"{algo.lower()}_{env_short}.yaml"
    if not preset.exists():
        return None
    import yaml

    with preset.open() as f:
        data = yaml.safe_load(f) or {}
    cfg = dict(data.get("hyperparameters", data))

    # Layer SB3-specific overrides on top.
    override = _SB3_OVERRIDES.get((algo.lower(), env_id))
    if override:
        cfg.update(override)

    return cfg


# ---------------------------------------------------------------------------
# Config-key translation: rlox preset -> SB3 constructor kwargs
# ---------------------------------------------------------------------------

# Keys that mean the same thing in rlox and SB3 — identity mapping.
_PASSTHROUGH_KEYS = {
    "learning_rate",
    "batch_size",
    "buffer_size",
    "gamma",
    "gae_lambda",
    "ent_coef",
    "vf_coef",
    "max_grad_norm",
    "n_steps",
    "n_epochs",
    "tau",
    "learning_starts",
    "train_freq",
    "gradient_steps",
    "policy_delay",
    "target_policy_noise",
    "target_noise_clip",
    "exploration_fraction",
    "exploration_final_eps",
}

# Keys whose name differs between rlox and SB3.
_RENAMED_KEYS = {
    "clip_range": "clip_range",   # rlox alias for clip_eps; SB3 uses clip_range
    "clip_eps": "clip_range",
    "target_update_freq": "target_update_interval",
}

# Keys handled specially (not passed directly to the SB3 constructor).
_SPECIAL_KEYS = {
    "n_envs",
    "normalize_obs",
    "normalize_rewards",
    "normalize_reward",
    "hidden",
    "anneal_lr",
    "normalize_advantages",
    # ``clip_vloss`` is NOT in this set anymore — see _translate_config which
    # forwards it to SB3 as ``clip_range_vf=clip_range`` (the only legal SB3
    # encoding of "clip the value loss with the same clip range").
    "double_dqn",
    "dueling",
    "n_step",
    "prioritized",
    "alpha",
    "beta_start",
    "exploration_initial_eps",
    "exploration_noise",
}


def _translate_config(algo: str, preset: dict) -> tuple[dict, dict]:
    """Split a preset dict into ``(sb3_kwargs, harness_options)``.

    ``sb3_kwargs`` are passed to the SB3 algorithm constructor.
    ``harness_options`` carry settings the harness needs to interpret
    (n_envs, normalize_obs, normalize_rewards, hidden width, action noise).
    """
    sb3_kwargs: dict[str, Any] = {}
    harness: dict[str, Any] = {
        "n_envs": preset.get("n_envs", 1),
        "normalize_obs": preset.get("normalize_obs", False),
        "normalize_rewards": preset.get(
            "normalize_rewards", preset.get("normalize_reward", False)
        ),
        "hidden": preset.get("hidden"),
        "exploration_noise": preset.get("exploration_noise"),
    }

    for k, v in preset.items():
        if k in _SPECIAL_KEYS:
            continue
        if k in _PASSTHROUGH_KEYS:
            sb3_kwargs[k] = v
        elif k in _RENAMED_KEYS:
            sb3_kwargs[_RENAMED_KEYS[k]] = v
        elif k == "clip_vloss":
            # SB3 has no boolean clip_vloss flag — clipping the value loss
            # is opt-in by setting ``clip_range_vf`` to a positive number.
            # If the preset asks for clipping, mirror by reusing the same
            # clip_range as the policy ratio (matches CleanRL convention).
            if v:
                # Resolve clip_range from whatever the preset already produced.
                clip = sb3_kwargs.get("clip_range") or preset.get("clip_eps") \
                    or preset.get("clip_range") or 0.2
                sb3_kwargs["clip_range_vf"] = float(clip)
        else:
            # Unknown key — drop with a warning rather than crashing.
            print(f"  [sb3 harness] WARNING: dropping unknown preset key {k!r}")

    # Hidden width -> SB3 policy_kwargs net_arch
    if harness["hidden"] is not None:
        h = harness["hidden"]
        sb3_kwargs.setdefault("policy_kwargs", {})["net_arch"] = [h, h]
    elif algo.lower() in {"ppo", "a2c"}:
        # rlox PPO/A2C policies use [64, 64] tanh by default.
        sb3_kwargs.setdefault("policy_kwargs", {})["net_arch"] = [64, 64]

    # Action noise (TD3): rlox stores a scalar sigma; SB3 wants an object.
    if algo.lower() == "td3" and harness["exploration_noise"] is not None:
        from stable_baselines3.common.noise import NormalActionNoise

        sigma = float(harness["exploration_noise"])
        # n_actions is filled in at construction time when we know the env
        harness["_action_noise_sigma"] = sigma

    return sb3_kwargs, harness


# ---------------------------------------------------------------------------
# Algorithm dispatch
# ---------------------------------------------------------------------------


def _make_env(env_id: str, n_envs: int, seed: int, normalize_obs: bool, normalize_rewards: bool, gamma: float):
    """Build a SB3 VecEnv with optional VecNormalize, mirroring rlox."""
    import gymnasium as gym
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    def _thunk(rank: int):
        def _f():
            e = gym.make(env_id)
            e.reset(seed=seed + rank)
            return e
        return _f

    venv = DummyVecEnv([_thunk(i) for i in range(n_envs)])
    if normalize_obs or normalize_rewards:
        venv = VecNormalize(
            venv,
            norm_obs=normalize_obs,
            norm_reward=normalize_rewards,
            gamma=gamma,
        )
    return venv


def _build_sb3_model(algo: str, env_id: str, seed: int, sb3_kwargs: dict, harness: dict):
    """Construct the SB3 model with the right algorithm + venv + kwargs."""
    from stable_baselines3 import PPO, SAC, TD3, DQN, A2C

    algo_l = algo.lower()
    n_envs = harness["n_envs"]
    norm_obs = harness["normalize_obs"]
    norm_rew = harness["normalize_rewards"]
    gamma = sb3_kwargs.get("gamma", 0.99)

    # Off-policy algos (SAC/TD3/DQN) are single-env in SB3 by default — even
    # if the rlox preset asks for n_envs > 1, mirror SB3's standard usage.
    if algo_l in {"sac", "td3", "dqn"}:
        n_envs = 1

    venv = _make_env(env_id, n_envs, seed, norm_obs, norm_rew, gamma)

    cls = {"ppo": PPO, "sac": SAC, "td3": TD3, "dqn": DQN, "a2c": A2C}[algo_l]

    # TD3 action noise: needs to be constructed now that we know action dim.
    if algo_l == "td3" and "_action_noise_sigma" in harness:
        from stable_baselines3.common.noise import NormalActionNoise

        n_actions = venv.action_space.shape[-1]
        sigma = harness["_action_noise_sigma"]
        sb3_kwargs["action_noise"] = NormalActionNoise(
            mean=np.zeros(n_actions, dtype=np.float32),
            sigma=sigma * np.ones(n_actions, dtype=np.float32),
        )

    model = cls("MlpPolicy", venv, seed=seed, verbose=0, **sb3_kwargs)
    return model, venv


# ---------------------------------------------------------------------------
# Single-seed train + eval
# ---------------------------------------------------------------------------


def run_single_seed(
    algo: str,
    env_id: str,
    total_timesteps: int,
    seed: int,
    config: dict | None = None,
) -> dict:
    """Train one seed of an SB3 algo, evaluate with the rlox harness protocol."""
    import gymnasium as gym
    import torch  # noqa: F401  (ensures the same torch is loaded)

    if config is None:
        config = _resolve_preset(algo, env_id)
    if config is None:
        raise ValueError(
            f"No preset found for ({algo}, {env_id}). Provide --config or "
            f"add a YAML to benchmarks/convergence/configs/."
        )

    sb3_kwargs, harness = _translate_config(algo, dict(config))

    model, venv = _build_sb3_model(algo, env_id, seed, sb3_kwargs, harness)

    t0 = time.time()
    model.learn(total_timesteps=total_timesteps)
    wall_time = time.time() - t0

    # ---- Evaluate (mirror rlox eval protocol exactly) ----
    eval_env = gym.make(env_id)
    is_discrete = hasattr(eval_env.action_space, "n")

    # If we trained with VecNormalize, freeze stats and apply normalization
    # at eval time.
    from stable_baselines3.common.vec_env import VecNormalize

    vn = venv if isinstance(venv, VecNormalize) else None
    if vn is not None:
        vn.training = False

    rewards = []
    for ep in range(30):
        obs, _ = eval_env.reset(seed=seed + 1000 + ep)
        ep_r = 0.0
        done = False
        while not done:
            obs_in = obs
            if vn is not None:
                obs_in = vn.normalize_obs(obs.reshape(1, -1)).flatten()
            action, _ = model.predict(obs_in, deterministic=True)

            # Coerce action shape for env.step
            if is_discrete:
                if isinstance(action, np.ndarray):
                    action = int(action.reshape(-1)[0])
                else:
                    action = int(action)
            else:
                if isinstance(action, np.ndarray):
                    action = action.astype(np.float32).flatten()
                else:
                    action = np.asarray(action, dtype=np.float32).flatten()

            obs, r, term, trunc, _ = eval_env.step(action)
            ep_r += float(r)
            done = term or trunc
        rewards.append(ep_r)
    eval_env.close()
    venv.close()

    return {
        "algo": algo,
        "env": env_id,
        "framework": "sb3",
        "seed": seed,
        "total_timesteps": total_timesteps,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "wall_time_s": wall_time,
        "sps": total_timesteps / max(wall_time, 1e-9),
    }


# ---------------------------------------------------------------------------
# Multi-seed driver — same shape as the rlox runner
# ---------------------------------------------------------------------------


def run_multi_seed(
    algo: str,
    env_id: str,
    total_timesteps: int,
    n_seeds: int = 5,
    config: dict | None = None,
    output_dir: str = "results/multi-seed-sb3",
) -> dict:
    """Run multiple seeds of an SB3 algo and compute IQM + bootstrap CI."""
    from rlox.evaluation import interquartile_mean, stratified_bootstrap_ci

    results = []
    for i in range(n_seeds):
        seed = i * 1000 + 42
        print(f"  [sb3] Seed {i + 1}/{n_seeds} (seed={seed})...", end=" ", flush=True)
        result = run_single_seed(algo, env_id, total_timesteps, seed, config)
        results.append(result)
        print(f"reward={result['mean_reward']:.1f}, SPS={result['sps']:.0f}")

    rewards = [r["mean_reward"] for r in results]
    iqm = interquartile_mean(rewards)
    ci_low, ci_high = stratified_bootstrap_ci(rewards)

    summary = {
        "algo": algo,
        "env": env_id,
        "framework": "sb3",
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

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fname = f"sb3_{algo}_{env_id.replace('/', '_')}_seeds{n_seeds}.json"
    with (out / fname).open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"  [sb3] IQM={iqm:.1f} [{ci_low:.1f}, {ci_high:.1f}] saved to {out / fname}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="SB3 multi-seed convergence benchmark — mirrors the rlox harness."
    )
    parser.add_argument("--algo", required=True, help="ppo | sac | td3 | dqn | a2c")
    parser.add_argument("--env", required=True)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--output", default="results/multi-seed-sb3")
    parser.add_argument("--config", help="YAML config file (defaults to auto-resolved preset)")
    args = parser.parse_args()

    config = None
    if args.config:
        import yaml

        with open(args.config) as f:
            data = yaml.safe_load(f) or {}
        config = data.get("hyperparameters", data)

    print(
        f"[sb3 harness] {args.algo} on {args.env} "
        f"({args.seeds} seeds, {args.timesteps} steps)"
    )
    run_multi_seed(args.algo, args.env, args.timesteps, args.seeds, config, args.output)


if __name__ == "__main__":
    main()
