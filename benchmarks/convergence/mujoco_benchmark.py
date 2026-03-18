"""MuJoCo continuous-control convergence benchmarks for rlox.

Tests PPO and SAC on continuous-control tasks. Pendulum-v1 always runs
(no MuJoCo install required). HalfCheetah-v4, Hopper-v4, and Walker2d-v4
are tested when ``gymnasium[mujoco]`` is available.

Usage:
    .venv/bin/python benchmarks/convergence/mujoco_benchmark.py
    .venv/bin/python benchmarks/convergence/mujoco_benchmark.py --envs Pendulum-v1
    .venv/bin/python benchmarks/convergence/mujoco_benchmark.py --algos SAC
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch

from rlox.evaluation import interquartile_mean


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class BenchmarkSpec:
    """One (algorithm, environment) benchmark configuration."""

    algorithm: str
    env_id: str
    total_timesteps: int
    reward_threshold: float
    n_eval_episodes: int = 20
    seed: int = 1
    requires_mujoco: bool = False


# rl-zoo3-style defaults --------------------------------------------------

SPECS: list[BenchmarkSpec] = [
    # --- Pendulum (always available) ---
    BenchmarkSpec(
        algorithm="PPO",
        env_id="Pendulum-v1",
        total_timesteps=100_000,
        reward_threshold=-400.0,
        seed=1,
    ),
    BenchmarkSpec(
        algorithm="SAC",
        env_id="Pendulum-v1",
        total_timesteps=50_000,
        reward_threshold=-300.0,
        seed=1,
    ),
    # --- MuJoCo (optional) ---
    BenchmarkSpec(
        algorithm="PPO",
        env_id="HalfCheetah-v4",
        total_timesteps=1_000_000,
        reward_threshold=1000.0,
        seed=1,
        requires_mujoco=True,
    ),
    BenchmarkSpec(
        algorithm="SAC",
        env_id="HalfCheetah-v4",
        total_timesteps=300_000,
        reward_threshold=3000.0,
        seed=1,
        requires_mujoco=True,
    ),
    BenchmarkSpec(
        algorithm="PPO",
        env_id="Hopper-v4",
        total_timesteps=1_000_000,
        reward_threshold=1000.0,
        seed=1,
        requires_mujoco=True,
    ),
    BenchmarkSpec(
        algorithm="SAC",
        env_id="Hopper-v4",
        total_timesteps=300_000,
        reward_threshold=1500.0,
        seed=1,
        requires_mujoco=True,
    ),
    BenchmarkSpec(
        algorithm="PPO",
        env_id="Walker2d-v4",
        total_timesteps=1_000_000,
        reward_threshold=1000.0,
        seed=1,
        requires_mujoco=True,
    ),
    BenchmarkSpec(
        algorithm="SAC",
        env_id="Walker2d-v4",
        total_timesteps=300_000,
        reward_threshold=1500.0,
        seed=1,
        requires_mujoco=True,
    ),
]


def _mujoco_available() -> bool:
    """Check whether gymnasium MuJoCo envs can be instantiated."""
    try:
        env = gym.make("HalfCheetah-v4")
        env.close()
        return True
    except (gym.error.DependencyNotInstalled, gym.error.NameNotFound, ImportError):
        return False


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_ppo_continuous(
    agent, env_id: str, n_episodes: int, seed: int,
) -> list[float]:
    """Evaluate a PPO agent with continuous policy (mean action, no sampling)."""
    env = gym.make(env_id)
    returns: list[float] = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_return = 0.0
        while not done:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                # Use the actor mean directly (deterministic) for evaluation
                mean = agent.policy.actor(obs_t)
                action = mean.squeeze(0).numpy()
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += float(reward)
            done = terminated or truncated
        returns.append(ep_return)
    env.close()
    return returns


def evaluate_sac(
    agent, env_id: str, n_episodes: int, seed: int,
) -> list[float]:
    """Evaluate a SAC agent with deterministic policy."""
    env = gym.make(env_id)
    returns: list[float] = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_return = 0.0
        while not done:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                action = agent.actor.deterministic(obs_t).squeeze(0).numpy()
                action = action * agent.act_high
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += float(reward)
            done = terminated or truncated
        returns.append(ep_return)
    env.close()
    return returns


# ---------------------------------------------------------------------------
# Training routines
# ---------------------------------------------------------------------------

def train_ppo(spec: BenchmarkSpec) -> tuple[float, list[float]]:
    """Train PPO on a continuous env. Returns (iqm, episode_returns)."""
    from rlox.algorithms.ppo import PPO

    print(f"\n{'=' * 70}")
    print(f"PPO on {spec.env_id} ({spec.total_timesteps:,} steps, seed={spec.seed})")
    print(f"{'=' * 70}")

    torch.manual_seed(spec.seed)
    np.random.seed(spec.seed)

    agent = PPO(
        env_id=spec.env_id,
        n_envs=1,
        seed=spec.seed,
        n_steps=1024,
        n_epochs=10,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        normalize_advantages=True,
    )

    t0 = time.perf_counter()
    metrics = agent.train(total_timesteps=spec.total_timesteps)
    elapsed = time.perf_counter() - t0
    print(f"  Training took {elapsed:.1f}s")
    print(f"  Training mean_reward: {metrics.get('mean_reward', 'N/A')}")

    eval_returns = evaluate_ppo_continuous(
        agent, spec.env_id, spec.n_eval_episodes, seed=spec.seed + 1000,
    )
    iqm = interquartile_mean(eval_returns)
    mean_r = float(np.mean(eval_returns))
    std_r = float(np.std(eval_returns))
    print(f"  Eval ({spec.n_eval_episodes} ep): mean={mean_r:.1f} +/- {std_r:.1f}  IQM={iqm:.1f}")

    return iqm, eval_returns


def train_sac(spec: BenchmarkSpec) -> tuple[float, list[float]]:
    """Train SAC on a continuous env. Returns (iqm, episode_returns)."""
    from rlox.algorithms.sac import SAC

    print(f"\n{'=' * 70}")
    print(f"SAC on {spec.env_id} ({spec.total_timesteps:,} steps, seed={spec.seed})")
    print(f"{'=' * 70}")

    torch.manual_seed(spec.seed)
    np.random.seed(spec.seed)

    agent = SAC(
        env_id=spec.env_id,
        buffer_size=1_000_000,
        learning_rate=3e-4,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        learning_starts=10_000,
        hidden=256,
        seed=spec.seed,
        auto_entropy=True,
    )

    t0 = time.perf_counter()

    # Manual loop so we can print episode progress
    obs, _ = agent.env.reset(seed=spec.seed)
    episode_rewards: list[float] = []
    ep_reward = 0.0

    for step in range(spec.total_timesteps):
        if step < agent.learning_starts:
            action = agent.env.action_space.sample()
        else:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                action_t, _ = agent.actor.sample(obs_t)
                action = action_t.squeeze(0).numpy()
            action = action * agent.act_high

        next_obs, reward, terminated, truncated, _ = agent.env.step(action)
        ep_reward += float(reward)

        agent.buffer.push(
            np.asarray(obs, dtype=np.float32),
            np.asarray(action, dtype=np.float32),
            float(reward),
            bool(terminated),
            bool(truncated),
            np.asarray(next_obs, dtype=np.float32),
        )

        obs = next_obs
        if terminated or truncated:
            episode_rewards.append(ep_reward)
            if len(episode_rewards) % 20 == 0:
                recent = episode_rewards[-20:]
                print(
                    f"  Episode {len(episode_rewards):5d} | "
                    f"reward={ep_reward:8.1f} | "
                    f"last-20 mean={np.mean(recent):8.1f}"
                )
            ep_reward = 0.0
            obs, _ = agent.env.reset()

        if step >= agent.learning_starts and len(agent.buffer) >= agent.batch_size:
            agent._update(step)

    elapsed = time.perf_counter() - t0
    print(f"  Training took {elapsed:.1f}s ({len(episode_rewards)} episodes)")

    eval_returns = evaluate_sac(
        agent, spec.env_id, spec.n_eval_episodes, seed=spec.seed + 1000,
    )
    iqm = interquartile_mean(eval_returns)
    mean_r = float(np.mean(eval_returns))
    std_r = float(np.std(eval_returns))
    print(f"  Eval ({spec.n_eval_episodes} ep): mean={mean_r:.1f} +/- {std_r:.1f}  IQM={iqm:.1f}")

    return iqm, eval_returns


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

_TRAIN_FNS = {
    "PPO": train_ppo,
    "SAC": train_sac,
}


@dataclass
class Result:
    """Stores one benchmark result for the summary table."""

    name: str
    iqm: float
    threshold: float
    passed: bool
    eval_returns: list[float]
    elapsed: float


def run_benchmarks(
    specs: list[BenchmarkSpec],
    has_mujoco: bool,
) -> list[Result]:
    """Run selected benchmark specs and collect results."""
    results: list[Result] = []

    for spec in specs:
        if spec.requires_mujoco and not has_mujoco:
            print(f"\n  [SKIP] {spec.algorithm}/{spec.env_id} (MuJoCo not installed)")
            continue

        fn = _TRAIN_FNS[spec.algorithm]
        t0 = time.perf_counter()
        iqm, eval_returns = fn(spec)
        elapsed = time.perf_counter() - t0

        passed = iqm >= spec.reward_threshold
        results.append(Result(
            name=f"{spec.algorithm}/{spec.env_id}",
            iqm=iqm,
            threshold=spec.reward_threshold,
            passed=passed,
            eval_returns=eval_returns,
            elapsed=elapsed,
        ))

    return results


def print_summary(results: list[Result]) -> None:
    """Print a summary table of all benchmark results."""
    print(f"\n{'=' * 70}")
    print("MUJOCO CONTINUOUS-CONTROL BENCHMARK SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'Benchmark':<30s}  {'IQM':>8s}  {'Thresh':>8s}  {'Time':>7s}  {'Status'}")
    print(f"  {'-' * 30}  {'-' * 8}  {'-' * 8}  {'-' * 7}  {'-' * 6}")

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(
            f"  {r.name:<30s}  {r.iqm:8.1f}  {r.threshold:8.1f}  "
            f"{r.elapsed:6.1f}s  [{status}]"
        )

    all_pass = all(r.passed for r in results)
    print(f"\n  {'ALL PASSED' if all_pass else 'SOME FAILED'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MuJoCo continuous-control convergence benchmarks",
    )
    parser.add_argument(
        "--algos",
        nargs="+",
        choices=["PPO", "SAC"],
        default=None,
        help="Algorithms to benchmark (default: all)",
    )
    parser.add_argument(
        "--envs",
        nargs="+",
        default=None,
        help="Environment IDs to benchmark (default: all available)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override seed for all specs",
    )
    args = parser.parse_args()

    has_mujoco = _mujoco_available()
    if has_mujoco:
        print("MuJoCo envs detected -- running full benchmark suite.")
    else:
        print("MuJoCo not installed -- running Pendulum-v1 benchmarks only.")

    specs = SPECS

    if args.algos is not None:
        algo_set = set(args.algos)
        specs = [s for s in specs if s.algorithm in algo_set]

    if args.envs is not None:
        env_set = set(args.envs)
        specs = [s for s in specs if s.env_id in env_set]

    if args.seed is not None:
        specs = [
            BenchmarkSpec(
                algorithm=s.algorithm,
                env_id=s.env_id,
                total_timesteps=s.total_timesteps,
                reward_threshold=s.reward_threshold,
                n_eval_episodes=s.n_eval_episodes,
                seed=args.seed,
                requires_mujoco=s.requires_mujoco,
            )
            for s in specs
        ]

    results = run_benchmarks(specs, has_mujoco)
    print_summary(results)

    all_pass = all(r.passed for r in results)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
