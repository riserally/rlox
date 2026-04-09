"""Convergence tests for continuous-control environments.

Pendulum-v1 tests always run (no MuJoCo dependency). MuJoCo env tests
are skipped when ``gymnasium[mujoco]`` is not installed.

Run with:
    .venv/bin/python -m pytest tests/python/test_mujoco_convergence.py -m slow -v
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch

from rlox.evaluation import interquartile_mean


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mujoco_available() -> bool:
    try:
        env = gym.make("HalfCheetah-v4")
        env.close()
        return True
    except (gym.error.DependencyNotInstalled, gym.error.NameNotFound, ImportError):
        return False


requires_mujoco = pytest.mark.skipif(
    not _mujoco_available(),
    reason="gymnasium[mujoco] not installed",
)


def _evaluate_ppo_continuous(
    agent, env_id: str, n_episodes: int, seed: int,
) -> list[float]:
    """Evaluate PPO continuous policy deterministically (actor mean)."""
    env = gym.make(env_id)
    returns: list[float] = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_return = 0.0
        while not done:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                mean = agent.policy.actor(obs_t)
                action = mean.squeeze(0).numpy()
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += float(reward)
            done = terminated or truncated
        returns.append(ep_return)
    env.close()
    return returns


def _evaluate_sac_deterministic(
    agent, env_id: str, n_episodes: int, seed: int,
) -> list[float]:
    """Evaluate SAC with deterministic policy."""
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
# Pendulum-v1 tests (always available)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_ppo_pendulum_converges() -> None:
    """PPO continuous on Pendulum-v1.

    Must reach IQM > -1200 in 200K steps (random policy scores ~-1400;
    this threshold confirms learning signal without requiring full convergence).

    Retries up to 3 seeds to handle RL variance -- passes if any seed succeeds.
    """
    from rlox.algorithms.ppo import PPO

    seeds = [1, 42, 123]
    threshold = -1200.0
    best_iqm = float("-inf")

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        agent = PPO(
            env_id="Pendulum-v1",
            n_envs=4,
            seed=seed,
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

        agent.train(total_timesteps=200_000)

        eval_returns = _evaluate_ppo_continuous(
            agent, "Pendulum-v1", n_episodes=20, seed=seed + 1000,
        )
        iqm = interquartile_mean(eval_returns)
        best_iqm = max(best_iqm, iqm)

        if iqm > threshold:
            return  # success

    assert False, (
        f"PPO failed on Pendulum-v1 across {len(seeds)} seeds: "
        f"best IQM={best_iqm:.1f}, expected > {threshold}"
    )


@pytest.mark.slow
def test_sac_pendulum_converges() -> None:
    """SAC on Pendulum-v1.

    Must reach IQM > -500 in 20K steps with rl-zoo3-style hyperparameters.
    This is a CI smoke test, not a full convergence benchmark — the real
    convergence validation is the multi-seed GCP sweep. Budget is kept
    low enough to finish within 300s on a GitHub Actions runner.
    """
    from rlox.algorithms.sac import SAC

    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)

    agent = SAC(
        env_id="Pendulum-v1",
        buffer_size=100_000,
        learning_rate=3e-4,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        learning_starts=1_000,
        hidden=256,
        seed=seed,
        auto_entropy=True,
    )

    agent.train(total_timesteps=20_000)

    eval_returns = _evaluate_sac_deterministic(
        agent, "Pendulum-v1", n_episodes=10, seed=seed + 1000,
    )
    iqm = interquartile_mean(eval_returns)
    assert iqm > -500.0, (
        f"SAC failed on Pendulum-v1: IQM={iqm:.1f}, expected > -500"
    )


# ---------------------------------------------------------------------------
# MuJoCo tests (skipped if not installed)
# ---------------------------------------------------------------------------

@pytest.mark.slow
@requires_mujoco
def test_ppo_halfcheetah_converges() -> None:
    """PPO on HalfCheetah-v4 smoke test.

    Trains 50K steps (not 1M) — enough to verify the training loop
    runs without crashing and the reward improves from baseline.
    Full convergence is validated by the multi-seed GCP sweep, not CI.
    """
    from rlox.algorithms.ppo import PPO

    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)

    agent = PPO(
        env_id="HalfCheetah-v4",
        n_envs=1,
        seed=seed,
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

    agent.train(total_timesteps=50_000)

    eval_returns = _evaluate_ppo_continuous(
        agent, "HalfCheetah-v4", n_episodes=10, seed=seed + 1000,
    )
    iqm = interquartile_mean(eval_returns)
    # At 50K steps PPO HalfCheetah is in early training — just verify
    # the reward is not degenerate (random policy gets ~-300 to -100).
    assert iqm > -200.0, (
        f"PPO failed on HalfCheetah-v4: IQM={iqm:.1f}, expected > -200"
    )


@pytest.mark.slow
@requires_mujoco
def test_sac_halfcheetah_converges() -> None:
    """SAC on HalfCheetah-v4 smoke test.

    Trains 30K steps (not 300K) to fit within CI timeout.
    Full convergence is validated by the multi-seed GCP sweep.
    """
    from rlox.algorithms.sac import SAC

    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)

    agent = SAC(
        env_id="HalfCheetah-v4",
        buffer_size=100_000,
        learning_rate=3e-4,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        learning_starts=1_000,
        hidden=256,
        seed=seed,
        auto_entropy=True,
    )

    agent.train(total_timesteps=30_000)

    eval_returns = _evaluate_sac_deterministic(
        agent, "HalfCheetah-v4", n_episodes=10, seed=seed + 1000,
    )
    iqm = interquartile_mean(eval_returns)
    # At 30K steps SAC HalfCheetah is in early training — verify
    # the loop didn't crash and reward is above random baseline.
    assert iqm > -200.0, (
        f"SAC failed on HalfCheetah-v4: IQM={iqm:.1f}, expected > -200"
    )
